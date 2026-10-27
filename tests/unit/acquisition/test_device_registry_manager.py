"""Device interface, registry, simulated driver and manager."""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pytest

from apgi_framework.acquisition import (
    DeviceError,
    DeviceManager,
    DeviceNotFoundError,
    DeviceRegistry,
    DeviceState,
    SampleChunk,
    SimulatedDevice,
    default_registry,
)
from apgi_framework.acquisition.device import Device

from .helpers import FakeClock


def make_device(clock: FakeClock, **kw: object) -> SimulatedDevice:
    return SimulatedDevice(clock_ns=clock, **kw)  # type: ignore[arg-type]


class TestRegistry:
    def test_default_registry_has_simulated(self) -> None:
        assert "simulated" in default_registry
        assert isinstance(default_registry.create("simulated"), SimulatedDevice)

    def test_register_create_and_names(self) -> None:
        reg = DeviceRegistry()
        reg.register("sim", SimulatedDevice)
        assert reg.names() == ["sim"]
        assert reg.create("sim", device_id="x").info.device_id == "x"

    def test_duplicate_requires_replace(self) -> None:
        reg = DeviceRegistry()
        reg.register("sim", SimulatedDevice)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("sim", SimulatedDevice)
        reg.register("sim", SimulatedDevice, replace=True)

    def test_unknown_driver_lists_known_ones(self) -> None:
        reg = DeviceRegistry()
        reg.register("a", SimulatedDevice)
        with pytest.raises(DeviceNotFoundError, match="registered: a"):
            reg.create("missing")

    def test_unregister(self) -> None:
        reg = DeviceRegistry()
        reg.register("a", SimulatedDevice)
        reg.unregister("a")
        assert "a" not in reg
        with pytest.raises(DeviceNotFoundError):
            reg.unregister("a")

    def test_decorator_registers_class(self) -> None:
        reg = DeviceRegistry()

        @reg.driver("deco")
        class Custom(SimulatedDevice):
            pass

        assert isinstance(reg.create("deco"), Custom)

    def test_factory_must_return_device(self) -> None:
        reg = DeviceRegistry()
        reg.register("bad", lambda **_: object())  # type: ignore[arg-type,return-value]
        with pytest.raises(TypeError, match="not a Device"):
            reg.create("bad")

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError):
            DeviceRegistry().register(" ", SimulatedDevice)


class TestDeviceLifecycle:
    def test_operations_need_the_right_state(self) -> None:
        dev = make_device(FakeClock())
        with pytest.raises(DeviceError, match="needs state"):
            dev.start_stream()
        with pytest.raises(DeviceError):
            dev.read()
        dev.connect()
        with pytest.raises(DeviceError):
            dev.connect()
        dev.start_stream()
        assert dev.state is DeviceState.STREAMING
        dev.stop_stream()
        assert dev.state is DeviceState.CONNECTED
        dev.disconnect()
        assert dev.state is DeviceState.DISCONNECTED

    def test_disconnect_while_streaming_stops_first(self) -> None:
        dev = make_device(FakeClock())
        dev.connect()
        dev.start_stream()
        dev.disconnect()
        assert dev.state is DeviceState.DISCONNECTED

    def test_connect_failure_sets_error_and_allows_retry(self) -> None:
        dev = make_device(FakeClock(), fail_on_connect=True)
        with pytest.raises(DeviceError):
            dev.connect()
        assert dev.state is DeviceState.ERROR
        dev._fail_on_connect = False
        dev.connect()
        assert dev.state is DeviceState.CONNECTED

    def test_context_manager_connects_and_disconnects(self) -> None:
        with make_device(FakeClock()) as dev:
            assert dev.state is DeviceState.CONNECTED
        assert dev.state is DeviceState.DISCONNECTED

    def test_bad_config_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_device(FakeClock(), max_chunk_samples=0)
        with pytest.raises(ValueError):
            make_device(FakeClock(), sampling_rate_hz=0)


class TestSimulatedStream:
    def test_sample_count_follows_the_clock(self) -> None:
        clock = FakeClock()
        dev = make_device(clock, sampling_rate_hz=250.0, max_chunk_samples=1000)
        dev.connect()
        dev.start_stream()
        assert dev.read() is None
        clock.advance(1.0)
        chunk = dev.read()
        assert chunk is not None and chunk.n_samples == 250
        assert chunk.samples.shape == (250, dev.info.n_channels)
        assert dev.read() is None

    def test_chunks_are_contiguous_and_capped(self) -> None:
        clock = FakeClock()
        dev = make_device(clock, max_chunk_samples=100)
        dev.connect()
        dev.start_stream()
        clock.advance(2.0)
        chunks: List[SampleChunk] = []
        while (c := dev.read()) is not None:
            chunks.append(c)
        assert [c.n_samples for c in chunks] == [100] * 5
        assert [c.first_sample_index for c in chunks] == [0, 100, 200, 300, 400]
        assert [c.sequence for c in chunks] == [0, 1, 2, 3, 4]

    def test_data_independent_of_read_sizes(self) -> None:
        def collect(max_chunk: int) -> np.ndarray:
            clock = FakeClock()
            dev = make_device(clock, seed=7, max_chunk_samples=max_chunk)
            dev.connect()
            dev.start_stream()
            clock.advance(3.0)
            parts = []
            while (c := dev.read()) is not None:
                parts.append(c.samples)
            return np.vstack(parts)

        a, b = collect(33), collect(512)
        assert a.shape == b.shape == (750, 9)
        np.testing.assert_array_equal(a, b)

    def test_seed_changes_data(self) -> None:
        def first(seed: int) -> np.ndarray:
            clock = FakeClock()
            dev = make_device(clock, seed=seed)
            dev.connect()
            dev.start_stream()
            clock.advance(0.2)
            chunk = dev.read()
            assert chunk is not None
            return chunk.samples

        assert not np.array_equal(first(1), first(2))
        np.testing.assert_array_equal(first(1), first(1))

    def test_signal_has_a_ten_hz_rhythm(self) -> None:
        clock = FakeClock()
        dev = make_device(clock, sampling_rate_hz=250.0, max_chunk_samples=5000)
        dev.connect()
        dev.start_stream()
        clock.advance(8.0)
        chunk = dev.read()
        assert chunk is not None
        spectrum = np.abs(np.fft.rfft(chunk.samples[:, 0]))
        freqs = np.fft.rfftfreq(chunk.n_samples, 1 / 250.0)
        assert abs(freqs[int(np.argmax(spectrum))] - 10.0) < 0.2

    def test_device_time_runs_at_the_device_rate(self) -> None:
        clock = FakeClock()
        dev = make_device(clock, drift_ppm=1000.0, max_chunk_samples=10_000)
        dev.connect()
        dev.start_stream()
        clock.advance(10.0)
        chunk = dev.read()
        assert chunk is not None
        # 1000 ppm fast: 10 host seconds hold 10.01 device seconds of samples
        assert chunk.n_samples == int(10.0 * 1.001 * 250)

    def test_dropped_chunk_leaves_an_index_gap(self) -> None:
        clock = FakeClock()
        dev = make_device(clock, max_chunk_samples=50, drop_chunks=[1])
        dev.connect()
        dev.start_stream()
        clock.advance(1.0)
        firsts = []
        while (c := dev.read()) is not None:
            firsts.append(c.first_sample_index)
        assert firsts[:3] == [0, 100, 150]

    def test_read_failure_marks_device_error(self) -> None:
        clock = FakeClock()
        dev = make_device(clock, fail_after_samples=50, max_chunk_samples=50)
        dev.connect()
        dev.start_stream()
        clock.advance(1.0)
        assert dev.read() is not None
        with pytest.raises(DeviceError):
            dev.read()
        assert dev.state is DeviceState.ERROR

    def test_hardware_marker_lands_in_trigger_channel(self) -> None:
        clock = FakeClock()
        dev = make_device(clock, sampling_rate_hz=100.0, max_chunk_samples=1000)
        dev.connect()
        dev.start_stream()
        clock.advance(0.5)
        dev.emit_hardware_marker(7)
        clock.advance(0.5)
        chunk = dev.read()
        assert chunk is not None
        trig = chunk.samples[:, -1]
        assert dev.supports_hardware_markers
        assert list(np.flatnonzero(trig)) == [50]
        assert trig[50] == 7.0

    def test_hardware_marker_validation(self) -> None:
        dev = make_device(FakeClock())
        dev.connect()
        with pytest.raises(DeviceError, match="active stream"):
            dev.emit_hardware_marker(1)
        dev.start_stream()
        with pytest.raises(ValueError):
            dev.emit_hardware_marker(0)
        bare = make_device(FakeClock(), trigger_channel=False)
        bare.connect()
        bare.start_stream()
        with pytest.raises(DeviceError):
            bare.emit_hardware_marker(1)
        assert not bare.supports_hardware_markers

    def test_base_class_rejects_hardware_markers(self) -> None:
        class Bare(SimulatedDevice):
            def emit_hardware_marker(self, code: int) -> None:
                Device.emit_hardware_marker(self, code)

        dev = Bare(clock_ns=FakeClock())
        with pytest.raises(DeviceError, match="does not support"):
            dev.emit_hardware_marker(1)

    def test_sync_probe_reflects_offset_and_rtt(self) -> None:
        clock = FakeClock()
        dev = make_device(clock, device_clock_offset_s=50.0, probe_rtt_ns=2_000_000)
        dev.connect()
        probe = dev.sync_probe()
        assert probe.rtt_ns == 2_000_000
        assert probe.device_time_s == pytest.approx(50.0 + 0.001)


class TestManager:
    def build(self, clock: FakeClock, **kw: object) -> DeviceManager:
        manager = DeviceManager(clock_ns=clock)
        manager.add_device(make_device(clock, **kw))
        return manager

    def test_add_by_driver_name_and_duplicates(self) -> None:
        manager = DeviceManager()
        assert manager.add("simulated", device_id="a") == "a"
        with pytest.raises(DeviceError, match="already managed"):
            manager.add("simulated", device_id="a")
        with pytest.raises(DeviceNotFoundError):
            manager.add("nope")
        assert manager.device_ids == ["a"]

    def test_unknown_device_lookup(self) -> None:
        with pytest.raises(DeviceNotFoundError):
            DeviceManager().device("ghost")

    def test_poll_delivers_chunks_to_sinks(self) -> None:
        clock = FakeClock()
        manager = self.build(clock)
        got: List[SampleChunk] = []
        manager.add_sink(got.append)
        assert all(v is None for v in manager.connect_all().values())
        manager.start_streaming()
        clock.advance(0.5)
        delivered = manager.poll()
        assert len(delivered) == 1 and got == delivered
        health = manager.health()["sim-0"]
        assert health.chunks_read == 1 and health.samples_read == delivered[0].n_samples

    def test_gap_detection_counts_missing_samples(self) -> None:
        clock = FakeClock()
        manager = self.build(clock, max_chunk_samples=50, drop_chunks=[1])
        manager.connect_all()
        manager.start_streaming()
        clock.advance(1.0)
        for _ in range(5):
            manager.poll()
        assert manager.health()["sim-0"].gaps_detected == 1

    def test_drain_reads_the_whole_backlog(self) -> None:
        clock = FakeClock()
        manager = self.build(clock, max_chunk_samples=50)
        got: List[SampleChunk] = []
        manager.add_sink(got.append)
        manager.connect_all()
        manager.start_streaming()
        clock.advance(2.0)
        assert manager.poll() and sum(c.n_samples for c in got) == 50  # poll reads one chunk
        assert manager.drain() == 9 and sum(c.n_samples for c in got) == 500
        assert manager.drain() == 0

    def test_drain_gives_up_on_a_stream_it_cannot_catch(self) -> None:
        """A consumer slower than the sample period always finds one more sample."""
        manager = DeviceManager()
        manager.add("simulated", sampling_rate_hz=1000.0)
        manager.add_sink(lambda chunk: time.sleep(0.005))
        manager.connect_all()
        manager.start_streaming()
        time.sleep(0.05)
        began = time.monotonic()
        delivered = manager.drain(timeout_s=0.2)
        assert delivered > 0 and time.monotonic() - began < 1.0
        manager.disconnect_all()

    def test_one_failing_device_does_not_stop_the_others(self) -> None:
        clock = FakeClock()
        manager = DeviceManager(clock_ns=clock)
        manager.add_device(make_device(clock, device_id="good"))
        manager.add_device(make_device(clock, device_id="bad", fail_after_samples=0))
        errors: List[str] = []
        manager.add_error_listener(lambda dev, exc: errors.append(dev))
        manager.connect_all()
        manager.start_streaming()
        clock.advance(0.5)
        delivered = manager.poll()
        assert [c.device_id for c in delivered] == ["good"]
        assert errors == ["bad"]
        health = manager.health()
        assert health["bad"].state is DeviceState.ERROR and health["bad"].last_error
        assert health["good"].state is DeviceState.STREAMING

    def test_connect_failure_is_reported_not_raised(self) -> None:
        clock = FakeClock()
        manager = DeviceManager(clock_ns=clock)
        manager.add_device(make_device(clock, device_id="ok"))
        manager.add_device(make_device(clock, device_id="broken", fail_on_connect=True))
        results = manager.connect_all()
        assert results["ok"] is None and isinstance(results["broken"], DeviceError)

    def test_failing_sink_is_isolated_from_other_sinks(self) -> None:
        clock = FakeClock()
        manager = self.build(clock)
        good: List[SampleChunk] = []

        def broken(_: SampleChunk) -> None:
            raise OSError("disk full")

        manager.add_sink(broken)
        manager.add_sink(good.append)
        manager.connect_all()
        manager.start_streaming()
        clock.advance(0.2)
        manager.poll()
        assert len(good) == 1
        assert "disk full" in (manager.health()["sim-0"].last_error or "")

    def test_sync_clocks_fits_every_device(self) -> None:
        clock = FakeClock()
        manager = self.build(clock, drift_ppm=200.0)
        manager.connect_all()
        fits = manager.sync_clocks(n_probes=10, interval_s=0.1, sleep=clock.advance)
        assert fits["sim-0"].drift_ppm == pytest.approx(200.0, abs=1.0)
        assert manager.mapper("sim-0").is_synced

    def test_remove_disconnects(self) -> None:
        clock = FakeClock()
        manager = self.build(clock)
        dev = manager.device("sim-0")
        manager.connect_all()
        manager.remove("sim-0")
        assert dev.state is DeviceState.DISCONNECTED and manager.device_ids == []

    def test_background_polling_collects_samples(self) -> None:
        manager = DeviceManager()
        manager.add("simulated", sampling_rate_hz=1000.0)
        got: List[SampleChunk] = []
        manager.add_sink(got.append)
        manager.connect_all()
        manager.start_streaming()
        manager.start_polling(0.001)
        with pytest.raises(DeviceError, match="already running"):
            manager.start_polling()
        deadline = time.monotonic() + 5.0
        while sum(c.n_samples for c in got) < 100 and time.monotonic() < deadline:
            time.sleep(0.01)
        manager.stop_streaming()
        manager.disconnect_all()
        assert sum(c.n_samples for c in got) >= 100
