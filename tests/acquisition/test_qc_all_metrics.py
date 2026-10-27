"""All twelve QC metrics, protocol timing tolerances and the three-domain clock."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apgi_framework.acquisition import (
    ChannelSpec,
    ClockMapper,
    DeviceInfo,
    LocalRecorder,
    SampleChunk,
    SyncProbe,
    ThreeDomainClock,
    compute_session_qc,
)
from apgi_framework.acquisition.clock import ClockSyncError
from apgi_framework.acquisition.drivers import SimulatedECGDevice
from apgi_framework.acquisition.qc import METRIC_NAMES, QCThresholds, detect_r_peaks
from apgi_framework.synchronization import ProtocolTimingRegistry, TimingTolerance, TimingViolation
from tests.acquisition.conftest import build_session, record_session, requires_sqlcipher
from tests.unit.acquisition.helpers import FakeClock

FS = 100.0
TODAY = dt.date(2026, 10, 27)


def write_recording(tmp: Path, info: DeviceInfo, data: np.ndarray, *, chunk_n: int = 50, markers=()) -> Any:  # type: ignore[no-untyped-def]
    rec = LocalRecorder(tmp / "s", info, fsync=False)
    rec.start()
    for i, first in enumerate(range(0, len(data), chunk_n)):
        part = data[first : first + chunk_n]
        rec.write_chunk(
            SampleChunk(info.device_id, i, first, 50.0 + first / info.sampling_rate_hz, part, i)
        )
    for m in markers:
        rec.write_marker(m)
    rec.close()
    return LocalRecorder.read(tmp / "s")


def eeg_info(**extra: Any) -> DeviceInfo:
    return DeviceInfo(
        "d", "usb", "m", FS, (ChannelSpec("Cz"), ChannelSpec("Pz")), firmware="1.2", extra=extra
    )


def probes(mapper: ClockMapper, n: int = 8, rtt_ns: int = 400_000, jitter=lambda i: 0) -> None:  # type: ignore[no-untyped-def]
    for i in range(n):
        send = 10**9 + i * 10**8
        mapper.add_probe(SyncProbe(send, 500.0 + i * 0.1, send + rtt_ns + jitter(i)))


def clean(n: int = 600, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 5, (n, 2))


class TestAllTwelve:
    def test_names_and_order(self, tmp_path: Path) -> None:
        qc = compute_session_qc(write_recording(tmp_path, eeg_info(), clean()))
        assert tuple(m.name for m in qc.metrics) == METRIC_NAMES and len(METRIC_NAMES) == 12
        assert qc["dropped_samples"].status == "pass"

    def test_unavailable_inputs_are_na_not_pass(self, tmp_path: Path) -> None:
        qc = compute_session_qc(write_recording(tmp_path, eeg_info(), clean()))
        for name in (
            "clock_offset_uncertainty",
            "electrode_impedance",
            "cardiac_peak_quality",
            "marker_to_signal_latency",
            "upload_completeness",
        ):
            assert qc[name].status == "n/a", name

    def test_clock_offset_uncertainty(self, tmp_path: Path) -> None:
        rec = write_recording(tmp_path, eeg_info(), clean())
        good, bad = ClockMapper(), ClockMapper()
        probes(good)
        probes(bad, rtt_ns=40_000_000)  # 40 ms round trip: midpoint can be 20 ms off
        assert compute_session_qc(rec, mapper=good)["clock_offset_uncertainty"].status in (
            "pass",
            "warn",
        )
        res = compute_session_qc(rec, mapper=bad)["clock_offset_uncertainty"]
        assert res.status == "fail" and res.value == pytest.approx(0.02, rel=0.05)
        assert (
            compute_session_qc(rec, mapper=ClockMapper())["clock_offset_uncertainty"].status
            == "fail"
        )

    def test_electrode_impedance(self, tmp_path: Path) -> None:
        rec = write_recording(tmp_path, eeg_info(), clean())
        ok = compute_session_qc(rec, impedances_kohm={"Cz": 2.0, "Pz": 3.0})["electrode_impedance"]
        bad = compute_session_qc(rec, impedances_kohm={"Cz": 2.0, "Pz": 40.0})[
            "electrode_impedance"
        ]
        assert (
            ok.status == "pass"
            and bad.status == "fail"
            and "Pz" in bad.detail
            and bad.value == 40.0
        )
        from_info = write_recording(tmp_path / "x", eeg_info(impedances_kohm={"Cz": 99}), clean())
        assert compute_session_qc(from_info)["electrode_impedance"].status == "fail"

    def test_signal_saturation(self, tmp_path: Path) -> None:
        d = clean()
        d[100:160, 0] = 1000.0  # rail
        res = compute_session_qc(write_recording(tmp_path, eeg_info(), d))["signal_saturation"]
        assert res.status == "fail" and res.value == pytest.approx(60 / d.size)
        assert (
            compute_session_qc(write_recording(tmp_path / "ok", eeg_info(), clean()))[
                "signal_saturation"
            ].status
            == "pass"
        )

    def test_motion_artifacts(self, tmp_path: Path) -> None:
        d = clean(1000)
        for w in range(0, 1000, 200):  # a large swing in every second window
            d[w + 10, :] = 800.0
        bad = compute_session_qc(write_recording(tmp_path, eeg_info(), d))["motion_artifacts"]
        ok = compute_session_qc(write_recording(tmp_path / "ok", eeg_info(), clean(1000)))[
            "motion_artifacts"
        ]
        assert bad.status == "fail" and bad.value >= 0.5 and ok.status == "pass" and ok.value == 0.0

    def test_cardiac_quality_good_and_bad(self, tmp_path: Path) -> None:
        clock = FakeClock()
        dev = SimulatedECGDevice(clock_ns=clock, max_chunk_samples=100_000, seed=2)
        dev.connect()
        dev.start_stream()
        clock.advance(30.0)
        chunk = dev.read(100_000)
        info = dev.info
        good = write_recording(tmp_path, info, chunk.samples, chunk_n=500)
        qc = compute_session_qc(good)["cardiac_peak_quality"]
        assert qc.status == "pass" and qc.value >= 0.95
        noise = np.random.default_rng(1).normal(0, 1, chunk.samples.shape)
        bad = write_recording(tmp_path / "bad", info, noise, chunk_n=500)
        assert compute_session_qc(bad)["cardiac_peak_quality"].status in ("fail", "warn")

    def test_no_peaks_is_a_failure(self, tmp_path: Path) -> None:
        info = DeviceInfo("d", "usb", "m", 500.0, (ChannelSpec("E", kind="ECG", unit="mV"),))
        rec = write_recording(tmp_path, info, np.zeros((2000, 1)))
        assert compute_session_qc(rec)["cardiac_peak_quality"].status == "fail"
        assert len(detect_r_peaks(np.zeros(10), 500.0)) == 0

    def test_upload_completeness(self, tmp_path: Path) -> None:
        rec = write_recording(tmp_path, eeg_info(), clean())
        full = compute_session_qc(
            rec, upload_counts={"pending": 0, "done": 8, "failed": 0}, server_verified=True
        )
        part = compute_session_qc(rec, upload_counts={"pending": 4, "done": 4, "failed": 0})
        failed = compute_session_qc(rec, upload_counts={"pending": 0, "done": 7, "failed": 1})
        empty = compute_session_qc(rec, upload_counts={"pending": 0, "done": 0, "failed": 0})
        assert (
            full["upload_completeness"].status == "pass"
            and full["upload_completeness"].value == 1.0
        )
        assert (
            part["upload_completeness"].status == "warn"
            and part["upload_completeness"].value == 0.5
        )
        assert (
            failed["upload_completeness"].status == "fail"
            and empty["upload_completeness"].status == "fail"
        )

    def test_firmware_and_calibration(self, tmp_path: Path) -> None:
        def run(info: DeviceInfo) -> Any:
            return compute_session_qc(
                write_recording(tmp_path / info.device_id, info, clean()), today=TODAY
            )["firmware_calibration"]

        ok = run(eeg_info(calibration_date="2026-06-01"))
        old = run(
            DeviceInfo(
                "old",
                "usb",
                "m",
                FS,
                (ChannelSpec("A"), ChannelSpec("B")),
                firmware="1",
                extra={"calibration_date": "2024-01-01"},
            )
        )
        none = run(DeviceInfo("none", "usb", "m", FS, (ChannelSpec("A"), ChannelSpec("B"))))
        syn = run(
            DeviceInfo(
                "syn",
                "simulated",
                "m",
                FS,
                (ChannelSpec("A"), ChannelSpec("B")),
                data_origin="synthetic",
            )
        )
        assert ok.status == "pass" and old.status == "fail" and "days old" in old.detail
        assert none.status == "fail" and "firmware" in none.detail and "calibration" in none.detail
        assert syn.status == "n/a"

    def test_effective_rate_threshold_is_configurable(self, tmp_path: Path) -> None:
        info = eeg_info()
        rec = LocalRecorder(tmp_path / "s", info, fsync=False)
        rec.start()
        for i in range(10):
            rec.write_chunk(
                SampleChunk("d", i, i * 50, 50.0 + i * 50 / FS * 1.0005, clean(50), i)
            )  # 500 ppm slow
        rec.close()
        r = LocalRecorder.read(tmp_path / "s")
        assert compute_session_qc(r)["effective_sampling_rate"].status == "fail"
        loose = QCThresholds(max_rate_error_ppm=1000)
        assert compute_session_qc(r, thresholds=loose)["effective_sampling_rate"].status == "pass"


@requires_sqlcipher
class TestSessionIntegration:
    def test_marker_to_signal_latency_with_a_hardware_trigger(self, tmp_path: Path) -> None:
        clock, store, manager, session = build_session(tmp_path)
        session.prepare()
        session.start()
        for i in range(30):
            clock.advance(0.1)
            session.pump()
            if i == 10:
                session.mark("stim", code=7, hardware=True)
        clock.advance(0.2)
        session.pump()
        session.stop()
        rec = LocalRecorder.read(session.directory)
        res = compute_session_qc(rec, mapper=manager.mapper("sim-0"))["marker_to_signal_latency"]
        assert res.status in ("pass", "warn") and res.value is not None and res.value < 5e-3
        store.close()

    def test_full_session_qc_has_no_failures_for_the_simulator(self, tmp_path: Path) -> None:
        clock, store, manager, session = build_session(tmp_path)
        record_session(clock, session)
        rec = LocalRecorder.read(session.directory)
        qc = compute_session_qc(
            rec, mapper=manager.mapper("sim-0"), upload_counts=store.upload_counts("sess1")
        )
        assert qc.failed == [] and qc.passed
        assert qc.to_dict()["passed"] and len(qc.to_dict()["metrics"]) == 12
        store.close()


class TestTimingTolerance:
    def tol(self, **kw: float) -> TimingTolerance:
        return TimingTolerance(
            "P01", **{"max_clock_offset_uncertainty_s": 1e-3, "max_marker_latency_s": 2e-3, **kw}
        )

    def test_each_protocol_sets_its_own_limits(self, tmp_path: Path) -> None:
        rec = write_recording(tmp_path, eeg_info(), clean())
        m = ClockMapper()
        probes(m, rtt_ns=6_000_000)  # uncertainty 3 ms
        strict = self.tol()
        lax = TimingTolerance(
            "P09", max_clock_offset_uncertainty_s=10e-3, max_marker_latency_s=10e-3
        )
        assert strict.violations(
            compute_session_qc(rec, mapper=m, thresholds=strict.qc_thresholds())
        )
        assert not lax.violations(compute_session_qc(rec, mapper=m, thresholds=lax.qc_thresholds()))

    def test_enforce_raises(self, tmp_path: Path) -> None:
        rec = write_recording(tmp_path, eeg_info(), clean())
        m = ClockMapper()
        probes(m, rtt_ns=6_000_000)
        tol = self.tol()
        with pytest.raises(TimingViolation, match="P01"):
            tol.enforce(compute_session_qc(rec, mapper=m, thresholds=tol.qc_thresholds()))

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            TimingTolerance("", 1e-3, 1e-3)
        with pytest.raises(ValueError):
            TimingTolerance("P", -1.0, 1e-3)

    def test_registry_has_no_default_and_is_write_once(self) -> None:
        reg = ProtocolTimingRegistry()
        with pytest.raises(KeyError, match="no registered timing tolerance"):
            reg.get("P01")
        reg.register(self.tol())
        reg.register(self.tol())  # identical re-registration is fine
        with pytest.raises(ValueError, match="different"):
            reg.register(self.tol(max_marker_latency_s=9e-3))
        assert "P01" in reg and reg.get("P01").max_marker_latency_s == 2e-3

    def test_thresholds_inherit_other_values(self) -> None:
        th = self.tol().qc_thresholds(QCThresholds(max_impedance_kohm=3.0))
        assert th.max_impedance_kohm == 3.0 and th.max_clock_offset_uncertainty_s == 1e-3


class TestThreeDomainClock:
    def make(self) -> ThreeDomainClock:
        m = ClockMapper()
        # device = 1000 + (host - 10^9 ns)/1e9 : host_ns = 1e9 + (dev - 1000) * 1e9
        for i in range(8):
            send = 10**9 + i * 10**8
            m.add_probe(SyncProbe(send, 1000.0 + (i * 10**8 + 200_000) / 1e9, send + 400_000))
        return ThreeDomainClock(m)

    def test_requires_an_anchor(self) -> None:
        with pytest.raises(ClockSyncError, match="anchor"):
            self.make().monotonic_to_utc_ns(5)

    def test_round_trips_through_all_three_domains(self) -> None:
        c = self.make()
        c.add_anchor(10**9, 1_700_000_000 * 10**9)
        utc = c.device_to_utc_ns(1000.5)
        assert utc == pytest.approx(1_700_000_000 * 10**9 + 5 * 10**8, abs=10**6)
        assert c.utc_to_device_s(utc) == pytest.approx(1000.5, abs=1e-6)
        assert c.utc_to_monotonic_ns(c.monotonic_to_utc_ns(2 * 10**9)) == 2 * 10**9

    def test_a_later_anchor_corrects_only_from_then_on(self) -> None:
        c = self.make()
        c.add_anchor(0, 1000)
        c.add_anchor(10**9, 1000 + 10**9 + 7_000)  # wall clock stepped by 7 us at t=1 s
        assert c.monotonic_to_utc_ns(5 * 10**8) == 1000 + 5 * 10**8  # before the step: old anchor
        assert c.monotonic_to_utc_ns(2 * 10**9) == 1000 + 2 * 10**9 + 7_000

    def test_anchor_arguments_must_come_together(self) -> None:
        with pytest.raises(ValueError):
            self.make().add_anchor(1, None)

    def test_default_anchor_reads_both_clocks(self) -> None:
        c = self.make()
        mono, utc = c.add_anchor()
        assert abs(utc - __import__("time").time_ns()) < 5 * 10**9 and mono > 0
