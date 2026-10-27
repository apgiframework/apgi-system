"""QC metrics, write-once raw data and the task pin."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from apgi_framework.acquisition import (
    ChannelSpec,
    DeviceInfo,
    LocalRecorder,
    PinError,
    RecorderError,
    SampleChunk,
    TaskPin,
    compute_qc,
    verify_write_once,
)
from tests.acquisition.conftest import build_session, record_session, requires_sqlcipher

FS = 100.0
INFO = DeviceInfo("dev", "sim", "m", FS, (ChannelSpec("A"),))


def chunk(seq: int, first: int, n: int, *, t0: float = 50.0, drift: float = 1.0) -> SampleChunk:
    return SampleChunk(
        "dev", seq, first, t0 + drift * first / FS, np.full((n, 1), float(seq)), host_receive_ns=seq
    )


def write(tmp_path: Path, chunks, **kw):  # type: ignore[no-untyped-def]
    rec = LocalRecorder(tmp_path / "s", INFO, fsync=False, **kw)
    rec.start()
    for c in chunks:
        rec.write_chunk(c)
    rec.close()
    return LocalRecorder.read(tmp_path / "s")


class TestQC:
    def test_clean_recording_passes_all_four_metrics(self, tmp_path: Path) -> None:
        qc = compute_qc(write(tmp_path, [chunk(i, i * 50, 50) for i in range(6)]))
        assert qc.passed and qc.dropped_samples == 0 and qc.duplicate_samples == 0
        assert qc.timestamp_discontinuities == 0
        assert qc.effective_rate_hz == pytest.approx(FS)
        assert qc.rate_error_ppm == pytest.approx(0.0, abs=1e-6)

    def test_effective_rate_reports_clock_drift(self, tmp_path: Path) -> None:
        # each sample interval takes 1.0001x nominal on the device clock: 100 ppm slow
        qc = compute_qc(write(tmp_path, [chunk(i, i * 50, 50, drift=1.0001) for i in range(10)]))
        assert qc.effective_rate_hz == pytest.approx(FS / 1.0001)
        assert qc.rate_error_ppm == pytest.approx(-100.0, rel=1e-3)
        assert qc.timestamp_discontinuities == 0  # drift this small is under tolerance

    def test_dropped_samples_are_counted(self, tmp_path: Path) -> None:
        qc = compute_qc(write(tmp_path, [chunk(0, 0, 50), chunk(2, 100, 50)]))
        assert qc.dropped_samples == 50 and qc.gaps == ((50, 50),) and not qc.passed
        assert qc.timestamp_discontinuities == 0  # time follows the index across a drop

    def test_duplicate_chunks_and_samples_are_detected(self, tmp_path: Path) -> None:
        qc = compute_qc(write(tmp_path, [chunk(0, 0, 50), chunk(1, 50, 50), chunk(1, 50, 50)]))
        assert qc.duplicate_samples == 50 and qc.duplicate_chunks == 1 and not qc.passed

    def test_partial_overlap_counts_only_the_overlap(self, tmp_path: Path) -> None:
        qc = compute_qc(write(tmp_path, [chunk(0, 0, 50), chunk(1, 40, 50)]))
        assert qc.duplicate_samples == 10

    def test_timestamp_jump_is_a_discontinuity(self, tmp_path: Path) -> None:
        late = chunk(2, 100, 50, t0=50.25)  # 250 ms ahead of where the index says
        qc = compute_qc(write(tmp_path, [chunk(0, 0, 50), chunk(1, 50, 50), late]))
        assert qc.timestamp_discontinuities == 1
        assert qc.max_timestamp_jump_s == pytest.approx(0.25)

    def test_untimed_recording_has_no_rate_and_no_false_discontinuities(
        self, tmp_path: Path
    ) -> None:
        chunks = [
            SampleChunk("dev", i, i * 50, None, np.zeros((50, 1)), host_receive_ns=i)
            for i in range(3)
        ]
        qc = compute_qc(write(tmp_path, chunks))
        assert qc.effective_rate_hz is None and qc.timestamp_discontinuities == 0
        assert qc.to_dict()["passed"] is True


class TestWriteOnce:
    def test_sealed_files_are_read_only_and_verify_clean(self, tmp_path: Path) -> None:
        write(tmp_path, [chunk(i, i * 50, 50) for i in range(4)], segment_samples=100)
        session = tmp_path / "s"
        files = sorted((session / "data").iterdir())
        assert files and all(not (f.stat().st_mode & stat.S_IWUSR) for f in files)
        for name in ("markers.jsonl", "sync.jsonl"):
            assert not (session / name).stat().st_mode & stat.S_IWUSR
        assert verify_write_once(session) == []

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permission bits")
    def test_sealed_segment_cannot_be_appended_to(self, tmp_path: Path) -> None:
        write(tmp_path, [chunk(0, 0, 50)])
        seg = next((tmp_path / "s" / "data").glob("segment-*.f32"))
        with pytest.raises(PermissionError):
            seg.open("ab")

    def test_tampering_is_reported(self, tmp_path: Path) -> None:
        write(tmp_path, [chunk(3, 0, 50)])  # non-zero samples, so zeroing them changes the hash
        seg = next((tmp_path / "s" / "data").glob("segment-*.f32"))
        seg.chmod(0o644)
        seg.write_bytes(b"\x00" * seg.stat().st_size)
        problems = verify_write_once(tmp_path / "s")
        assert any("content changed" in p for p in problems)
        assert any("still writable" in p for p in problems)

    def test_missing_file_is_reported(self, tmp_path: Path) -> None:
        write(tmp_path, [chunk(0, 0, 50)])
        next((tmp_path / "s" / "data").glob("chunks-*.jsonl")).unlink()
        assert any("missing" in p for p in verify_write_once(tmp_path / "s"))

    def test_existing_recording_is_never_overwritten(self, tmp_path: Path) -> None:
        write(tmp_path, [chunk(0, 0, 50)])
        again = LocalRecorder(tmp_path / "s", INFO, fsync=False)
        with pytest.raises(RecorderError, match="already holds a recording"):
            again.start()

    def test_recovering_a_finished_recording_is_harmless(self, tmp_path: Path) -> None:
        write(tmp_path, [chunk(0, 0, 50)])
        summary = LocalRecorder.recover(tmp_path / "s")
        assert summary.n_samples == 50 and verify_write_once(tmp_path / "s") == []

    def test_crash_after_sealing_but_before_manifest_is_recoverable(self, tmp_path: Path) -> None:
        rec = LocalRecorder(tmp_path / "s", INFO, fsync=False, segment_samples=50)
        rec.start()
        rec.write_chunk(chunk(0, 0, 50))
        rec.write_chunk(chunk(1, 50, 50))  # rotates: segment 0 sealed (read-only), no manifest yet
        summary = LocalRecorder.recover(tmp_path / "s")
        assert summary.n_samples == 100 and verify_write_once(tmp_path / "s") == []


class TestTaskPin:
    def test_pin_validates_its_fields(self) -> None:
        with pytest.raises(PinError):
            TaskPin("", "b1")
        with pytest.raises(PinError):
            TaskPin("m1", "  ")

    def test_from_manifest_hashes_canonical_json(self) -> None:
        a = TaskPin.from_manifest({"version": "2.1", "tasks": ["x", "y"]}, battery_version="B7")
        b = TaskPin.from_manifest({"tasks": ["x", "y"], "version": "2.1"}, battery_version="B7")
        c = TaskPin.from_manifest({"version": "2.1", "tasks": ["y", "x"]}, battery_version="B7")
        assert a == b and a.task_manifest_sha256 != c.task_manifest_sha256
        assert a.task_manifest_version == "2.1" and a.battery_version == "B7"

    def test_manifest_without_version_is_rejected(self) -> None:
        with pytest.raises(PinError, match="version"):
            TaskPin.from_manifest({"tasks": []}, battery_version="B7")

    def test_round_trip_and_missing_key(self) -> None:
        pin = TaskPin("2.1", "B7", "abc")
        assert TaskPin.from_dict(pin.to_dict()) == pin
        with pytest.raises(PinError, match="battery_version"):
            TaskPin.from_dict({"task_manifest_version": "2.1"})

    def test_frozen(self) -> None:
        with pytest.raises(Exception):
            TaskPin("1", "2").battery_version = "3"  # type: ignore[misc]


@requires_sqlcipher
class TestSessionPin:
    def test_pin_is_stored_at_creation_and_written_to_the_recording(self, tmp_path: Path) -> None:
        pin = TaskPin.from_manifest({"version": "2.1"}, battery_version="B7")
        clock, store, _, session = build_session(tmp_path, task_pin=pin)
        row = store.get_session("sess1")
        assert row is not None and row.meta["task_pin"] == pin.to_dict()  # before prepare()
        record_session(clock, session)
        meta = json.loads((session.directory / "recording.json").read_text())
        assert meta["meta"]["task_pin"] == pin.to_dict()
        sidecar = json.loads(
            session.export_bids(tmp_path / "bids", task="rest").sidecar.read_text()
        )
        assert sidecar["TaskManifestVersion"] == "2.1" and sidecar["BatteryVersion"] == "B7"
        store.close()

    def test_require_pin_refuses_an_unpinned_session(self, tmp_path: Path) -> None:
        with pytest.raises(PinError):
            build_session(tmp_path, require_pin=True)

    def test_unpinned_session_records_null_pin(self, tmp_path: Path) -> None:
        clock, store, _, session = build_session(tmp_path)
        record_session(clock, session)
        meta = json.loads((session.directory / "recording.json").read_text())
        assert meta["meta"]["task_pin"] is None
        store.close()
