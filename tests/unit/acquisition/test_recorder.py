"""Local recorder: layout, sealing, rotation, crash recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from apgi_framework.acquisition import (
    ChannelSpec,
    DeviceInfo,
    EventMarker,
    LocalRecorder,
    RecorderError,
    SampleChunk,
    SyncProbe,
)

INFO = DeviceInfo("dev", "sim", "m", 100.0, (ChannelSpec("A"), ChannelSpec("B")))


def chunk(seq: int, first: int, n: int, *, t0: float = 50.0, fill: float = 0.0) -> SampleChunk:
    base = np.arange(first, first + n, dtype=np.float64)[:, None] + fill
    return SampleChunk(
        device_id="dev",
        sequence=seq,
        first_sample_index=first,
        first_device_time_s=t0 + first / 100.0,
        samples=np.hstack([base, -base]),
        host_receive_ns=1_000 + seq,
    )


def record(tmp_path: Path, chunks: List[SampleChunk], **kw: object) -> Tuple[Path, object]:
    session = tmp_path / "s1"
    rec = LocalRecorder(session, INFO, fsync=False, **kw)  # type: ignore[arg-type]
    rec.start()
    for c in chunks:
        rec.write_chunk(c)
    return session, rec


class TestRoundTrip:
    def test_layout_manifest_and_checksums(self, tmp_path: Path) -> None:
        session, rec = record(tmp_path, [chunk(0, 0, 10), chunk(1, 10, 10)])
        rec.write_marker(EventMarker("go", 123, code=4))  # type: ignore[attr-defined]
        rec.write_sync_probe(SyncProbe(1, 50.0, 3))  # type: ignore[attr-defined]
        summary = rec.close()  # type: ignore[attr-defined]
        assert summary.n_samples == 20 and summary.n_chunks == 2 and summary.n_markers == 1
        assert summary.complete and (session / "COMPLETE").exists()
        manifest = json.loads((session / "manifest.json").read_text())
        seg = manifest["segments"][0]
        data = (session / seg["data_file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == seg["sha256"]
        assert len(data) == 20 * 2 * 4 == seg["size_bytes"]

    def test_read_back_matches_float32(self, tmp_path: Path) -> None:
        chunks = [chunk(0, 0, 7), chunk(1, 7, 5)]
        session, rec = record(tmp_path, chunks)
        rec.write_marker(EventMarker("x", 9))  # type: ignore[attr-defined]
        rec.write_sync_probe(SyncProbe(1, 50.0, 3))  # type: ignore[attr-defined]
        rec.close()  # type: ignore[attr-defined]
        recording = LocalRecorder.read(session)
        expected = np.vstack([c.samples for c in chunks]).astype(np.float32)
        np.testing.assert_array_equal(recording.samples, expected)
        assert recording.info == INFO and recording.complete and recording.gaps == []
        assert [m.label for m in recording.markers] == ["x"]
        assert recording.sync_probes == [SyncProbe(1, 50.0, 3)]
        assert recording.sample0_device_time_s == pytest.approx(50.0)

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        session = tmp_path / "ctx"
        with LocalRecorder(session, INFO, fsync=False) as rec:
            rec.write_chunk(chunk(0, 0, 4))
        assert (session / "COMPLETE").exists()

    def test_rotation_seals_each_segment(self, tmp_path: Path) -> None:
        ready: List[Tuple[str, str, int]] = []
        chunks = [chunk(i, i * 10, 10) for i in range(5)]
        session, rec = record(
            tmp_path, chunks, segment_samples=20, on_file_ready=lambda *a: ready.append(a)
        )
        summary = rec.close()  # type: ignore[attr-defined]
        assert len(summary.segments) == 3
        assert [s.n_samples for s in summary.segments] == [20, 20, 10]
        names = {r[0] for r in ready}
        assert {"data/segment-000000.f32", "data/chunks-000002.jsonl", "manifest.json"} <= names
        for rel, sha, size in ready:
            path = session / rel
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == sha and path.stat().st_size == size
            )
        recording = LocalRecorder.read(session)
        assert recording.n_samples == 50 and recording.gaps == []

    def test_gaps_are_reported(self, tmp_path: Path) -> None:
        session, rec = record(tmp_path, [chunk(0, 0, 10), chunk(2, 30, 10)])
        rec.close()  # type: ignore[attr-defined]
        recording = LocalRecorder.read(session)
        assert recording.n_samples == 40 and recording.gaps == [(10, 20)]
        assert recording.samples[15].tolist() == [0.0, 0.0]


class TestGuards:
    def test_wrong_channel_count(self, tmp_path: Path) -> None:
        _, rec = record(tmp_path, [])
        bad = SampleChunk("dev", 0, 0, 0.0, np.zeros((3, 5)), 0)
        with pytest.raises(RecorderError, match="channels"):
            rec.write_chunk(bad)  # type: ignore[attr-defined]

    def test_lifecycle_errors(self, tmp_path: Path) -> None:
        rec = LocalRecorder(tmp_path / "a", INFO, fsync=False)
        with pytest.raises(RecorderError, match="not started"):
            rec.write_chunk(chunk(0, 0, 1))
        rec.start()
        with pytest.raises(RecorderError, match="already started"):
            rec.start()
        rec.close()
        with pytest.raises(RecorderError, match="closed"):
            rec.write_chunk(chunk(0, 0, 1))
        with pytest.raises(RecorderError, match="already closed"):
            rec.close()

    def test_refuses_to_overwrite_a_recording(self, tmp_path: Path) -> None:
        record(tmp_path, [chunk(0, 0, 3)])[1].close()  # type: ignore[attr-defined]
        with pytest.raises(RecorderError, match="already holds"):
            LocalRecorder(tmp_path / "s1", INFO).start()

    def test_invalid_segment_size(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            LocalRecorder(tmp_path, INFO, segment_samples=0)

    def test_read_rejects_non_recordings_and_bad_versions(self, tmp_path: Path) -> None:
        with pytest.raises(RecorderError, match="not a recording"):
            LocalRecorder.read(tmp_path)
        session, rec = record(tmp_path, [chunk(0, 0, 3)])
        rec.close()  # type: ignore[attr-defined]
        meta = json.loads((session / "recording.json").read_text())
        meta["format_version"] = 99
        (session / "recording.json").write_text(json.dumps(meta))
        with pytest.raises(RecorderError, match="unsupported"):
            LocalRecorder.read(session)

    def test_read_requires_sealed_segments(self, tmp_path: Path) -> None:
        session, _ = record(tmp_path, [chunk(0, 0, 3)])
        with pytest.raises(RecorderError, match="recover"):
            LocalRecorder.read(session)


class TestCrashRecovery:
    def test_torn_last_chunk_is_dropped(self, tmp_path: Path) -> None:
        chunks = [chunk(i, i * 10, 10) for i in range(5)]
        session, rec = record(tmp_path, chunks)
        rec.write_marker(EventMarker("kept", 1))  # type: ignore[attr-defined]
        # simulate power loss: no close(), last chunk's bytes cut short, torn index/marker lines
        data = session / "data" / "segment-000000.f32"
        data.write_bytes(data.read_bytes()[:-13])
        with open(session / "data" / "chunks-000000.jsonl", "a") as fh:
            fh.write('{"seq": 5, "first": 50, "n"')
        with open(session / "markers.jsonl", "a") as fh:
            fh.write('{"label": "torn"')
        ready: List[str] = []
        summary = LocalRecorder.recover(session, on_file_ready=lambda r, *_: ready.append(r))
        assert summary.n_chunks == 4 and summary.n_samples == 40 and summary.n_markers == 1
        assert (session / "COMPLETE").exists() and "data/segment-000000.f32" in ready
        recording = LocalRecorder.read(session)
        expected = np.vstack([c.samples for c in chunks[:4]]).astype(np.float32)
        np.testing.assert_array_equal(recording.samples, expected)
        assert [m.label for m in recording.markers] == ["kept"]

    def test_corrupt_chunk_stops_recovery_at_the_damage(self, tmp_path: Path) -> None:
        session, rec = record(tmp_path, [chunk(i, i * 10, 10) for i in range(4)])
        data = session / "data" / "segment-000000.f32"
        raw = bytearray(data.read_bytes())
        raw[2 * 10 * 8 + 3] ^= 0xFF  # flip a byte inside the third chunk
        data.write_bytes(bytes(raw))
        summary = LocalRecorder.recover(session)
        assert summary.n_chunks == 2

    def test_recovery_keeps_already_sealed_segments(self, tmp_path: Path) -> None:
        chunks = [chunk(i, i * 10, 10) for i in range(5)]
        session, rec = record(tmp_path, chunks, segment_samples=20)
        # crash: segments 0 and 1 sealed by rotation, segment 2 still open
        summary = LocalRecorder.recover(session)
        assert summary.n_samples == 50 and len(summary.segments) == 3
        assert LocalRecorder.read(session).n_samples == 50

    def test_empty_segment_is_discarded(self, tmp_path: Path) -> None:
        session, _ = record(tmp_path, [])
        summary = LocalRecorder.recover(session)
        assert summary.n_samples == 0 and summary.segments == ()
        assert not list((session / "data").glob("segment-*"))
