"""BIDS export and the end-to-end offline-first session."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from apgi_framework.acquisition import (
    AcquisitionSession,
    ChannelSpec,
    ClockMapper,
    DeviceInfo,
    DeviceManager,
    EventMarker,
    ExportError,
    InvalidTransition,
    LocalDirectoryTransport,
    LocalRecorder,
    SampleChunk,
    SessionState,
    SessionStore,
    SimulatedDevice,
    SyncProbe,
    UploadAgent,
    export_recording_to_bids,
    export_session_to_bids,
    recover_interrupted_sessions,
    recover_session,
    validate_bids_eeg,
)

from .helpers import KEY, NS, FakeClock, requires_sqlcipher

FS = 100.0
INFO = DeviceInfo(
    "dev",
    "sim",
    "Model X",
    FS,
    (
        ChannelSpec("Cz"),
        ChannelSpec("Pz"),
        ChannelSpec("EOG1", kind="EOG"),
        ChannelSpec("TRIG", kind="TRIG", unit="n/a"),
    ),
    firmware="1.2",
    serial="SN9",
)
T0_DEVICE = 200.0
HOST0 = 7 * NS  # host time of sample 0 in the synthetic recording


def synthetic_recording(tmp_path: Path, *, n: int = 300, gap: bool = False):  # type: ignore[no-untyped-def]
    """Recording whose device clock = host clock + 193 s exactly (no drift)."""
    session = tmp_path / "rec"
    rec = LocalRecorder(session, INFO, fsync=False)
    rec.start()
    data = np.arange(n * 4, dtype=np.float64).reshape(n, 4) / 10.0
    pieces = [(0, 100, 0), (100, 100, 1), (200, n - 200, 2)]
    for first, size, seq in pieces:
        if gap and seq == 1:
            continue
        rec.write_chunk(
            SampleChunk("dev", seq, first, T0_DEVICE + first / FS, data[first : first + size], 0)
        )
    for host_offset_s in (0.0, 0.5, 1.0, 1.5):
        host = HOST0 + int(host_offset_s * NS)
        rec.write_sync_probe(SyncProbe(host - 100, T0_DEVICE + host_offset_s, host + 100))
    rec.write_marker(EventMarker("stimulus", HOST0 + int(0.5 * NS), code=3, duration_s=0.2))
    rec.write_marker(EventMarker("response, left", HOST0 + int(1.25 * NS), value="L"))
    rec.write_marker(EventMarker("before", HOST0 - NS))
    rec.write_marker(EventMarker("after", HOST0 + 9 * NS))
    rec.close()
    return session, LocalRecorder.read(session), data


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


class TestBidsExport:
    def test_files_and_names(self, tmp_path: Path) -> None:
        _, recording, _ = synthetic_recording(tmp_path)
        result = export_recording_to_bids(
            recording,
            tmp_path / "bids",
            subject="01",
            session="a",
            task="rest",
            run=2,
            power_line_hz=50.0,
            eeg_reference="Cz",
        )
        stem = "sub-01_ses-a_task-rest_run-2"
        assert result.header.name == f"{stem}_eeg.vhdr"
        assert result.header.parent == tmp_path / "bids" / "sub-01" / "ses-a" / "eeg"
        for name in (
            "_eeg.eeg",
            "_eeg.vmrk",
            "_eeg.json",
            "_channels.tsv",
            "_events.tsv",
            "_events.json",
        ):
            assert (result.header.parent / f"{stem}{name}").exists()
        for name in ("dataset_description.json", "participants.tsv", "participants.json", "README"):
            assert (tmp_path / "bids" / name).exists()
        assert validate_bids_eeg(tmp_path / "bids") == []

    def test_binary_data_roundtrips_as_multiplexed_float32(self, tmp_path: Path) -> None:
        _, recording, data = synthetic_recording(tmp_path)
        result = export_recording_to_bids(recording, tmp_path / "b", subject="01", task="rest")
        raw = np.fromfile(result.data, dtype="<f4").reshape(-1, 4)
        np.testing.assert_array_equal(raw, data.astype(np.float32))
        header = result.header.read_text(encoding="utf-8")
        assert "NumberOfChannels=4" in header and "SamplingInterval=10000" in header
        assert "BinaryFormat=IEEE_FLOAT_32" in header and "DataOrientation=MULTIPLEXED" in header
        assert (
            f"DataFile={result.data.name}" in header
            and f"MarkerFile={result.markers.name}" in header
        )
        assert "Ch1=Cz,,1,µV" in header and "Ch4=TRIG,,1,n/a" in header

    def test_events_are_aligned_to_the_data_through_the_clock_mapping(self, tmp_path: Path) -> None:
        _, recording, _ = synthetic_recording(tmp_path)
        result = export_recording_to_bids(recording, tmp_path / "b", subject="01", task="rest")
        rows = read_tsv(result.events)
        assert [r["trial_type"] for r in rows] == ["stimulus", "response, left"]
        assert float(rows[0]["onset"]) == pytest.approx(0.5, abs=1e-6)
        assert float(rows[0]["duration"]) == pytest.approx(0.2)
        assert rows[0]["value"] == "3" and rows[0]["sample"] == "50"
        assert float(rows[1]["onset"]) == pytest.approx(1.25, abs=1e-6)
        assert rows[1]["value"] == "L" and rows[1]["sample"] == "125"
        assert len(result.dropped_markers) == 2  # one before the first sample, one after the last

    def test_brainvision_markers_are_one_based_and_escaped(self, tmp_path: Path) -> None:
        _, recording, _ = synthetic_recording(tmp_path)
        result = export_recording_to_bids(recording, tmp_path / "b", subject="01", task="rest")
        vmrk = result.markers.read_text(encoding="utf-8")
        assert "Mk1=New Segment,,1,1,0" in vmrk
        assert "Mk2=Stimulus,stimulus,51,20,0" in vmrk
        assert "Mk3=Stimulus,response\\1 left,126,1,0" in vmrk

    def test_sidecar_and_channels(self, tmp_path: Path) -> None:
        _, recording, _ = synthetic_recording(tmp_path)
        result = export_recording_to_bids(
            recording,
            tmp_path / "b",
            subject="01",
            task="rest",
            power_line_hz=50.0,
            eeg_reference="Cz",
            task_description="eyes closed",
        )
        meta = json.loads(result.sidecar.read_text(encoding="utf-8"))
        assert meta["TaskName"] == "rest" and meta["SamplingFrequency"] == FS
        assert meta["EEGReference"] == "Cz" and meta["PowerLineFrequency"] == 50.0
        assert meta["SoftwareFilters"] == "n/a"
        assert meta["EEGChannelCount"] == 2 and meta["EOGChannelCount"] == 1
        assert meta["TriggerChannelCount"] == 1 and meta["RecordingDuration"] == 3.0
        assert meta["DeviceSerialNumber"] == "SN9" and meta["TaskDescription"] == "eyes closed"
        channels = read_tsv(result.channels)
        assert [(c["name"], c["type"], c["units"]) for c in channels] == [
            ("Cz", "EEG", "µV"),
            ("Pz", "EEG", "µV"),
            ("EOG1", "EOG", "µV"),
            ("TRIG", "TRIG", "n/a"),
        ]
        assert all(c["sampling_frequency"] == "100" and c["status"] == "good" for c in channels)

    def test_unknown_reference_and_mains_are_na(self, tmp_path: Path) -> None:
        _, recording, _ = synthetic_recording(tmp_path)
        result = export_recording_to_bids(recording, tmp_path / "b", subject="01", task="rest")
        meta = json.loads(result.sidecar.read_text(encoding="utf-8"))
        assert meta["EEGReference"] == "n/a" and meta["PowerLineFrequency"] == "n/a"

    def test_gaps_become_events_and_are_filled(self, tmp_path: Path) -> None:
        _, recording, _ = synthetic_recording(tmp_path, gap=True)
        result = export_recording_to_bids(
            recording, tmp_path / "b", subject="01", task="rest", gap_fill_value=-1.0
        )
        assert result.n_gaps == 1
        gap = [r for r in read_tsv(result.events) if r["trial_type"] == "gap"]
        assert len(gap) == 1 and float(gap[0]["onset"]) == 1.0 and float(gap[0]["duration"]) == 1.0
        raw = np.fromfile(result.data, dtype="<f4").reshape(-1, 4)
        assert (raw[100:200] == -1.0).all() and (raw[:100] != -1.0).any()

    def test_second_subject_extends_participants(self, tmp_path: Path) -> None:
        _, recording, _ = synthetic_recording(tmp_path)
        root = tmp_path / "b"
        export_recording_to_bids(
            recording, root, subject="01", task="rest", participant={"age": "30"}
        )
        export_recording_to_bids(
            recording, root, subject="02", task="rest", participant={"age": "41", "sex": "F"}
        )
        rows = read_tsv(root / "participants.tsv")
        assert [r["participant_id"] for r in rows] == ["sub-01", "sub-02"]
        assert rows[0]["age"] == "30" and rows[0]["sex"] == "n/a" and rows[1]["sex"] == "F"
        assert validate_bids_eeg(root) == []

    def test_overwrite_protection(self, tmp_path: Path) -> None:
        _, recording, _ = synthetic_recording(tmp_path)
        root = tmp_path / "b"
        export_recording_to_bids(recording, root, subject="01", task="rest")
        with pytest.raises(ExportError, match="overwrite"):
            export_recording_to_bids(recording, root, subject="01", task="rest")
        export_recording_to_bids(recording, root, subject="01", task="rest", overwrite=True)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"subject": "../x"},
            {"subject": "a b"},
            {"subject": "01", "task": "re-st"},
            {"subject": "01", "session": "a_b"},
        ],
    )
    def test_labels_must_be_alphanumeric(self, tmp_path: Path, kwargs: Dict[str, str]) -> None:
        _, recording, _ = synthetic_recording(tmp_path)
        args = {"subject": "01", "task": "rest", **kwargs}
        with pytest.raises(ExportError, match="alphanumeric"):
            export_recording_to_bids(recording, tmp_path / "b", **args)
        assert not (tmp_path / "x").exists()

    def test_run_must_be_positive(self, tmp_path: Path) -> None:
        _, recording, _ = synthetic_recording(tmp_path)
        with pytest.raises(ExportError, match="run"):
            export_recording_to_bids(recording, tmp_path / "b", subject="01", task="rest", run=0)

    def test_requires_clock_probes_or_an_explicit_mapper(self, tmp_path: Path) -> None:
        _, recording, _ = synthetic_recording(tmp_path)
        recording.sync_probes = []
        with pytest.raises(ExportError, match="sync probes"):
            export_recording_to_bids(recording, tmp_path / "b", subject="01", task="rest")
        mapper = ClockMapper()
        for off in (0.0, 1.0):
            mapper.add_probe(
                SyncProbe(HOST0 + int(off * NS), T0_DEVICE + off, HOST0 + int(off * NS))
            )
        result = export_recording_to_bids(
            recording, tmp_path / "b", subject="01", task="rest", mapper=mapper
        )
        assert result.n_events == 2

    def test_export_from_session_dir_recovers_unfinished_recordings(self, tmp_path: Path) -> None:
        session = tmp_path / "crash"
        rec = LocalRecorder(session, INFO, fsync=False)
        rec.start()
        rec.write_chunk(SampleChunk("dev", 0, 0, T0_DEVICE, np.ones((50, 4)), 0))
        rec.write_sync_probe(SyncProbe(HOST0, T0_DEVICE, HOST0))
        result = export_session_to_bids(session, tmp_path / "b", subject="01", task="rest")
        assert result.data.stat().st_size == 50 * 4 * 4

    def test_validator_detects_damage(self, tmp_path: Path) -> None:
        _, recording, _ = synthetic_recording(tmp_path)
        root = tmp_path / "b"
        result = export_recording_to_bids(recording, root, subject="01", task="rest")
        assert validate_bids_eeg(root) == []
        result.data.write_bytes(result.data.read_bytes()[:-1])
        result.markers.unlink()
        meta = json.loads(result.sidecar.read_text())
        del meta["TaskName"]
        result.sidecar.write_text(json.dumps(meta))
        (root / "dataset_description.json").unlink()
        problems = "\n".join(validate_bids_eeg(root))
        for needle in (
            "multiple of channels",
            "MarkerFile",
            "TaskName",
            "dataset_description.json",
        ):
            assert needle in problems
        assert validate_bids_eeg(tmp_path / "empty")  # nothing exported


@requires_sqlcipher
class TestSessionEndToEnd:
    def build(self, tmp_path: Path, **device_kw: object):  # type: ignore[no-untyped-def]
        clock = FakeClock()
        store = SessionStore(tmp_path / "apgi.db", key=KEY, clock_ns=lambda: clock.now // 1000)
        manager = DeviceManager(clock_ns=clock)
        manager.add_device(
            SimulatedDevice(
                clock_ns=clock,
                sampling_rate_hz=250.0,
                drift_ppm=60.0,
                device_clock_offset_s=4321.0,
                probe_jitter_ns=50_000,
                **device_kw,  # type: ignore[arg-type]
            )
        )
        session = AcquisitionSession(
            store=store,
            sessions_dir=tmp_path / "sessions",
            manager=manager,
            subject="01",
            session_id="sess1",
            fsync=False,
            clock_ns=clock,
            sleep=clock.advance,
            segment_samples=500,
        )
        return clock, store, manager, session

    def run_recording(
        self, clock: FakeClock, session: AcquisitionSession, seconds: float = 4.0
    ) -> None:
        session.prepare()
        session.start()
        steps = int(seconds / 0.1)
        for i in range(steps):
            clock.advance(0.1)
            session.pump()
            if i == 9:
                session.mark("stimulus", code=1, duration_s=0.1)
        clock.advance(0.05)

    def test_full_cycle_record_export_upload(self, tmp_path: Path) -> None:
        clock, store, manager, session = self.build(tmp_path)
        self.run_recording(clock, session)
        summary = session.stop()
        assert session.machine.state is SessionState.FINALIZED
        assert summary.n_samples >= 1000 and len(summary.segments) >= 2
        assert store.upload_counts("sess1")["pending"] >= 2 * len(summary.segments)

        result = session.export_bids(tmp_path / "bids", task="rest")
        assert validate_bids_eeg(tmp_path / "bids") == []
        stim = [r for r in read_tsv(result.events) if r["trial_type"] == "stimulus"]
        assert len(stim) == 1
        # the marker fired at t = 1.0 s of host time after streaming began, within one sample
        assert float(stim[0]["onset"]) == pytest.approx(1.0, abs=1 / 250.0)

        agent = UploadAgent(
            store, LocalDirectoryTransport(tmp_path / "remote"), clock_ns=lambda: clock.now, seed=1
        )
        report = agent.run_once()
        assert (
            report.synced_sessions == ["sess1"] and session.machine.state is SessionState.FINALIZED
        )
        assert store.get_session("sess1").state == "synced"  # type: ignore[union-attr]
        for seg in summary.segments:
            a = (tmp_path / "sessions" / "sess1" / seg.data_file).read_bytes()
            b = (tmp_path / "remote" / "sess1" / seg.data_file).read_bytes()
            assert a == b
        store.close()

    def test_hardware_marker_matches_the_software_marker(self, tmp_path: Path) -> None:
        clock, store, manager, session = self.build(tmp_path)
        session.prepare()
        session.start()
        clock.advance(2.0)
        session.drain()
        marker = session.mark("trigger", code=9, hardware=True)
        assert marker.source == "hardware"
        clock.advance(1.0)
        session.stop()  # stop() must drain the one-second backlog itself
        result = session.export_bids(tmp_path / "bids", task="rest")
        data = np.fromfile(result.data, dtype="<f4").reshape(-1, 9)  # 8 EEG + TRIG
        trig_sample = int(np.flatnonzero(data[:, -1] == 9.0)[0])
        row = next(r for r in read_tsv(result.events) if r["trial_type"] == "trigger")
        assert abs(trig_sample - int(row["sample"])) <= 1  # software and hardware stamps agree
        store.close()

    def test_pause_leaves_a_documented_gap(self, tmp_path: Path) -> None:
        clock, store, manager, session = self.build(tmp_path)
        session.prepare()
        session.start()
        for _ in range(5):
            clock.advance(0.1)
            session.pump()
        session.pause()
        for _ in range(5):
            clock.advance(0.1)
            session.pump()
        session.resume()
        for _ in range(5):
            clock.advance(0.1)
            session.pump()
        session.stop()
        result = session.export_bids(tmp_path / "bids", task="rest")
        assert result.n_gaps >= 1
        assert any(r["trial_type"] == "gap" for r in read_tsv(result.events))
        store.close()

    def test_background_polling_stops_promptly_and_loses_nothing(self, tmp_path: Path) -> None:
        """Regression: stop() once deadlocked against the polling thread for its full timeout."""
        import time

        manager = DeviceManager()
        manager.add_device(SimulatedDevice(sampling_rate_hz=250.0))
        # A slow sink keeps the polling thread busy inside the chunk callbacks. The old
        # stop() joined that thread while holding the session lock it needs, so it
        # always waited out the join timeout (5 s).
        manager.add_sink(lambda chunk: time.sleep(0.005))
        store = SessionStore(tmp_path / "bg.db", key=KEY)
        session = AcquisitionSession(
            store=store,
            sessions_dir=tmp_path,
            manager=manager,
            subject="01",
            fsync=False,
            n_sync_probes=3,
            sync_interval_s=0.01,
            drain_timeout_s=0.3,
            session_id="bg",
        )
        session.prepare()
        session.start(background=True, poll_interval_s=0.001)
        started = time.monotonic()
        time.sleep(0.4)
        began = time.monotonic()
        summary = session.stop()
        finished = time.monotonic()
        assert finished - began < 1.5  # drain budget 0.3 s; the old join stalled 5 s
        # The device samples on the wall clock, so a loaded runner that delays the sleep or the
        # drain produces proportionally more samples: bound by the elapsed time, not a constant
        assert 50 <= summary.n_samples <= 250.0 * (finished - started) + 5
        recording = LocalRecorder.read(tmp_path / "bg")
        assert recording.gaps == [] and recording.n_samples == summary.n_samples
        store.close()

    def test_markers_need_an_active_session(self, tmp_path: Path) -> None:
        clock, store, manager, session = self.build(tmp_path)
        with pytest.raises(InvalidTransition):
            session.mark("early")
        session.prepare()
        with pytest.raises(InvalidTransition):
            session.mark("still early")
        with pytest.raises(ValueError):
            session.start()
            session.mark("x", hardware=True)
        with pytest.raises(InvalidTransition):
            session.export_bids(tmp_path / "b", task="rest")
        store.close()

    def test_connection_failure_fails_the_session(self, tmp_path: Path) -> None:
        clock, store, manager, session = self.build(tmp_path, fail_on_connect=True)
        with pytest.raises(Exception, match="connection failed"):
            session.prepare()
        assert session.machine.state is SessionState.FAILED
        store.close()

    def test_device_failure_midway_keeps_recorded_data_and_can_be_salvaged(
        self, tmp_path: Path
    ) -> None:
        clock, store, manager, session = self.build(
            tmp_path, fail_after_samples=500, max_chunk_samples=100
        )
        session.prepare()
        session.start()
        for _ in range(30):  # 500 samples last 2 s; the next read then raises
            clock.advance(0.1)
            session.pump()
        assert session.machine.state is SessionState.FAILED
        summary = recover_session(store, "sess1")
        assert summary.n_samples == 500
        assert store.get_session("sess1").state == "finalized"  # type: ignore[union-attr]
        store.close()

    def test_crash_while_recording_then_restart_recovers(self, tmp_path: Path) -> None:
        clock, store, manager, session = self.build(tmp_path)
        self.run_recording(clock, session, seconds=3.0)
        store.close()  # the process dies here; stop() was never called
        reopened = SessionStore(tmp_path / "apgi.db", key=KEY)
        assert recover_interrupted_sessions(reopened) == ["sess1"]
        assert reopened.get_session("sess1").state == "interrupted"  # type: ignore[union-attr]
        summary = recover_session(reopened, "sess1")
        assert summary.n_samples > 0
        assert reopened.get_session("sess1").state == "finalized"  # type: ignore[union-attr]
        # the salvaged data uploads like any other session
        agent = UploadAgent(reopened, LocalDirectoryTransport(tmp_path / "remote"), seed=1)
        assert agent.run_once().synced_sessions == ["sess1"]
        with pytest.raises(InvalidTransition):
            recover_session(reopened, "sess1")
        reopened.close()

    def test_recording_works_with_no_network_at_all(self, tmp_path: Path) -> None:
        clock, store, manager, session = self.build(tmp_path)
        session.machine.set_online(False)
        self.run_recording(clock, session, seconds=2.0)
        session.stop()
        assert session.machine.state is SessionState.FINALIZED
        offline = UploadAgent(
            store, LocalDirectoryTransport(tmp_path / "remote"), is_online=lambda: False
        )
        assert offline.run_once().offline and not (tmp_path / "remote").exists()
        online = UploadAgent(store, LocalDirectoryTransport(tmp_path / "remote"), seed=1)
        assert online.run_once().synced_sessions == ["sess1"]
        store.close()

    def test_needs_a_device_id_when_several_devices_exist(self, tmp_path: Path) -> None:
        clock = FakeClock()
        manager = DeviceManager(clock_ns=clock)
        manager.add_device(SimulatedDevice(device_id="a", clock_ns=clock))
        manager.add_device(SimulatedDevice(device_id="b", clock_ns=clock))
        store = SessionStore(tmp_path / "s.db", key=KEY)
        with pytest.raises(Exception, match="device_id"):
            AcquisitionSession(store=store, sessions_dir=tmp_path, manager=manager, subject="01")
        store.close()
