"""Clock mapping and event markers."""

from __future__ import annotations

import numpy as np
import pytest

from apgi_framework.acquisition import (
    ChannelSpec,
    ClockMapper,
    ClockSyncError,
    DeviceInfo,
    EventMarker,
    MarkerLog,
    SimulatedDevice,
    SyncProbe,
    marker_device_time_s,
    marker_sample_index,
)

from .helpers import NS, FakeClock


def probes_from_device(
    drift_ppm: float, offset_s: float, n: int, *, jitter_ns: int = 0, asym_ns: int = 0
) -> tuple[ClockMapper, SimulatedDevice, FakeClock]:
    clock = FakeClock()
    dev = SimulatedDevice(
        clock_ns=clock,
        drift_ppm=drift_ppm,
        device_clock_offset_s=offset_s,
        probe_jitter_ns=jitter_ns,
        probe_asymmetry_ns=asym_ns,
        seed=3,
    )
    dev.connect()
    mapper = ClockMapper()
    for _ in range(n):
        mapper.add_probe(dev.sync_probe())
        clock.advance(0.2)
    return mapper, dev, clock


class TestClockMapper:
    def test_unsynced_mapper_raises(self) -> None:
        mapper = ClockMapper()
        assert not mapper.is_synced and mapper.fit is None
        with pytest.raises(ClockSyncError):
            mapper.device_to_host_ns(1.0)
        with pytest.raises(ClockSyncError):
            mapper.host_to_device_s(1)

    def test_single_probe_gives_offset_only(self) -> None:
        probe = SyncProbe(host_send_ns=10 * NS, device_time_s=500.0, host_recv_ns=10 * NS + 1000)
        mapper = ClockMapper()
        mapper.add_probe(probe)
        assert mapper.device_to_host_ns(500.0) == probe.host_mid_ns
        assert mapper.device_to_host_ns(501.0) == pytest.approx(probe.host_mid_ns + NS, abs=1)
        assert mapper.fit is not None and mapper.fit.drift_ppm == 0.0

    def test_recovers_offset_and_drift(self) -> None:
        mapper, dev, clock = probes_from_device(drift_ppm=80.0, offset_s=1234.5, n=30)
        fit = mapper.fit
        assert fit is not None and fit.drift_ppm == pytest.approx(80.0, abs=0.5)
        # truth: device reads offset + elapsed*(1+drift) since the device was built
        host_ns = clock.now
        truth_device_s = 1234.5 + (host_ns - dev._ref_ns) * 1e-9 * (1 + 80e-6)
        assert mapper.host_to_device_s(host_ns) == pytest.approx(truth_device_s, abs=5e-6)
        assert mapper.device_to_host_ns(truth_device_s) == pytest.approx(host_ns, abs=5_000)

    def test_roundtrip_is_consistent(self) -> None:
        mapper, _, clock = probes_from_device(drift_ppm=-120.0, offset_s=9.0, n=20)
        for delta in (0.0, 1.5, 30.0):
            host = clock.now + int(delta * NS)
            back = mapper.device_to_host_ns(mapper.host_to_device_s(host))
            assert back == pytest.approx(host, abs=100)

    def test_jittery_round_trips_still_map_accurately(self) -> None:
        mapper, dev, clock = probes_from_device(
            drift_ppm=40.0, offset_s=77.0, n=60, jitter_ns=2_000_000
        )
        truth = 77.0 + (clock.now - dev._ref_ns) * 1e-9 * (1 + 40e-6)
        assert mapper.host_to_device_s(clock.now) == pytest.approx(truth, abs=1e-3)

    def test_slow_round_trips_are_discarded(self) -> None:
        """A few probes with a large one-sided delay must not bias the fit."""
        rng = np.random.default_rng(0)
        offset, drift = 100.0, 30e-6
        mapper = ClockMapper(window=64, keep_fraction=0.5)
        t_ns = 0
        for i in range(40):
            t_ns += 100_000_000
            slow = i % 5 == 0
            out = 40_000_000 if slow else 100_000 + int(rng.integers(0, 20_000))
            back = 100_000 + int(rng.integers(0, 20_000))
            send, recv = t_ns, t_ns + out + back
            device_s = offset + (send + out) * 1e-9 * (1 + drift)
            mapper.add_probe(SyncProbe(send, device_s, recv))
        truth_dev = offset + (t_ns + 1_000_000) * 1e-9 * (1 + drift)
        assert mapper.host_to_device_s(t_ns + 1_000_000) == pytest.approx(truth_dev, abs=2e-4)
        assert mapper.fit is not None and mapper.fit.n_used < mapper.fit.n_points

    def test_window_is_bounded_and_reset_clears(self) -> None:
        mapper = ClockMapper(window=5)
        for i in range(12):
            mapper.add_probe(SyncProbe(i * NS, float(i), i * NS + 10))
        assert mapper.n_probes == 5
        mapper.reset()
        assert not mapper.is_synced and mapper.n_probes == 0

    def test_invalid_arguments(self) -> None:
        with pytest.raises(ValueError):
            ClockMapper(window=1)
        with pytest.raises(ValueError):
            ClockMapper(keep_fraction=0.0)
        with pytest.raises(ValueError):
            SyncProbe(host_send_ns=5, device_time_s=0.0, host_recv_ns=4)


class TestMarkers:
    def test_emit_uses_clock_and_notifies(self) -> None:
        clock = FakeClock()
        log = MarkerLog(clock)
        seen: list[EventMarker] = []
        log.subscribe(seen.append)
        marker = log.emit("stim", code=3, duration_s=0.5, trial="a")
        assert marker.onset_host_ns == clock.now and marker.extra == {"trial": "a"}
        assert seen == [marker] and len(log) == 1

    def test_snapshot_is_sorted_by_onset(self) -> None:
        log = MarkerLog(FakeClock())
        log.emit("late", onset_host_ns=300)
        log.emit("early", onset_host_ns=100)
        assert [m.label for m in log.snapshot()] == ["early", "late"]
        assert [m.label for m in log.between(0, 200)] == ["early"]

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            EventMarker(label="", onset_host_ns=0)
        with pytest.raises(ValueError):
            EventMarker(label="x", onset_host_ns=0, duration_s=-1.0)

    def test_dict_roundtrip(self) -> None:
        m = EventMarker(
            "a", 5, code=2, duration_s=1.0, value="v", source="hardware", extra={"k": 1}
        )
        assert EventMarker.from_dict(m.to_dict()) == m
        assert EventMarker.from_dict(m.to_dict()).extra == {"k": 1}

    def test_device_time_and_sample_index(self) -> None:
        mapper, dev, clock = probes_from_device(drift_ppm=0.0, offset_s=1000.0, n=10)
        info = DeviceInfo("d", "sim", "m", 250.0, (ChannelSpec("c"),))
        marker = EventMarker("m", clock.now)
        device_s = marker_device_time_s(marker, mapper)
        assert device_s == pytest.approx(1000.0 + (clock.now - dev._ref_ns) * 1e-9, abs=1e-5)
        assert marker_sample_index(marker, info, mapper, device_s - 2.0) == 500
