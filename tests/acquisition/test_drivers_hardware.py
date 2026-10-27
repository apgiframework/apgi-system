"""EEG/ECG/PPG drivers (simulated first, LSL via an injected inlet) and the hardware registry."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pytest

from apgi_framework.acquisition import (
    DeviceError,
    DeviceManager,
    HardwareRegistry,
    StimulationControlError,
    default_hardware_registry,
    default_registry,
    require_controllable,
    surrogate_usable_in_realtime,
)
from apgi_framework.acquisition.drivers import (
    ECGDevice,
    EEGDevice,
    PPGDevice,
    SimulatedECGDevice,
    SimulatedEEGDevice,
    SimulatedPPGDevice,
)
from apgi_framework.acquisition.hardware import (
    HARDWARE_CLASSES,
    INFERENCE_CADENCE_S,
    SURROGATE_RANKING,
    Role,
    UnknownHardwareClass,
)
from apgi_framework.acquisition.qc import detect_r_peaks
from tests.unit.acquisition.helpers import FakeClock

pytestmark = pytest.mark.device_mock


def stream(dev: Any, clock: FakeClock, seconds: float) -> np.ndarray:
    dev.connect()
    dev.start_stream()
    clock.advance(seconds)
    parts = []
    while (c := dev.read(10_000)) is not None:
        parts.append(c.samples)
    dev.disconnect()
    return np.vstack(parts)


class TestSimulatedDrivers:
    @pytest.mark.parametrize(
        "cls,kind,unit,fs",
        [
            (SimulatedEEGDevice, "EEG", "uV", 250.0),
            (SimulatedECGDevice, "ECG", "mV", 500.0),
            (SimulatedPPGDevice, "PPG", "a.u.", 100.0),
        ],
    )
    def test_identity_and_labelling(self, cls: Any, kind: str, unit: str, fs: float) -> None:
        dev = cls(clock_ns=FakeClock())
        analog = [c for c in dev.info.channels if c.kind != "TRIG"]
        assert analog and all(c.kind == kind and c.unit == unit for c in analog)
        assert dev.info.sampling_rate_hz == fs
        assert (
            dev.info.data_origin == "synthetic" and dev.info.driver == f"simulated-{kind.lower()}"
        )
        assert dev.capabilities.device_timestamps

    def test_ecg_has_detectable_r_peaks_at_the_set_rate(self) -> None:
        clock = FakeClock()
        dev = SimulatedECGDevice(clock_ns=clock, heart_rate_bpm=60.0, max_chunk_samples=10_000)
        data = stream(dev, clock, 20.0)
        peaks = detect_r_peaks(data[:, 0], 500.0)
        rr = np.diff(peaks) / 500.0
        assert len(peaks) >= 18 and abs(float(np.median(rr)) - 1.0) < 0.02

    def test_ppg_period_matches_heart_rate(self) -> None:
        clock = FakeClock()
        dev = SimulatedPPGDevice(clock_ns=clock, heart_rate_bpm=90.0, max_chunk_samples=10_000)
        x = stream(dev, clock, 20.0)[:, 0]
        spectrum = np.abs(np.fft.rfft(x - x.mean()))
        freq = np.fft.rfftfreq(x.size, 1 / 100.0)[int(np.argmax(spectrum))]
        assert freq == pytest.approx(1.5, abs=0.1)

    def test_same_seed_same_data(self) -> None:
        a = stream(
            SimulatedEEGDevice(clock_ns=(c1 := FakeClock()), seed=3, max_chunk_samples=10_000),
            c1,
            2,
        )
        b = stream(
            SimulatedEEGDevice(clock_ns=(c2 := FakeClock()), seed=3, max_chunk_samples=10_000),
            c2,
            2,
        )
        assert np.array_equal(a, b)

    def test_registered_by_name_and_manageable(self) -> None:
        for name in (
            "simulated-eeg",
            "simulated-ecg",
            "simulated-ppg",
            "lsl-eeg",
            "lsl-ecg",
            "lsl-ppg",
        ):
            assert name in default_registry
        manager = DeviceManager()
        assert manager.add("simulated-ecg") == "sim-ecg-0"

    def test_plain_simulated_device_is_unchanged(self) -> None:
        dev = default_registry.create("simulated", clock_ns=FakeClock())
        assert dev.info.driver == "simulated" and dev.info.model == "APGI-SIM"
        assert dev.info.channel_names[0] == "Fp1"


class FakeInlet:
    def __init__(self, batches: List[Tuple[np.ndarray, List[float]]]) -> None:
        self.batches = batches
        self.closed = False

    def pull_chunk(self, timeout: float = 0.0, max_samples: int = 64) -> Tuple[Any, Any]:
        if not self.batches:
            return [], []
        s, t = self.batches.pop(0)
        return s.tolist(), t

    def close_stream(self) -> None:
        self.closed = True


def lsl_batch(first_t: float, n: int, fs: float, n_ch: int, base: float = 0.0):  # type: ignore[no-untyped-def]
    t = [first_t + i / fs for i in range(n)]
    return np.full((n, n_ch), base), t


class TestLslDrivers:
    def make(self, cls: Any, batches: list, **kw: Any) -> Tuple[Any, FakeInlet]:
        inlet = FakeInlet(batches)
        dev = cls(inlet_factory=lambda: inlet, lsl_clock=lambda: 123.5, **kw)
        return dev, inlet

    def test_timestamps_are_the_streams_own(self) -> None:
        dev, _ = self.make(
            EEGDevice, [lsl_batch(1000.25, 8, 250.0, 8)], channel_names=[f"c{i}" for i in range(8)]
        )
        dev.connect()
        dev.start_stream()
        chunk = dev.read()
        assert chunk.first_device_time_s == 1000.25 and chunk.first_sample_index == 0
        assert dev.info.data_origin == "real" and dev.info.driver == "lsl-eeg"

    def test_a_timestamp_gap_becomes_an_index_gap(self) -> None:
        fs = 500.0
        batches = [
            lsl_batch(10.0, 50, fs, 1),
            lsl_batch(10.0 + 150 / fs, 50, fs, 1),
        ]  # 100 samples lost
        dev, _ = self.make(ECGDevice, batches)
        dev.connect()
        dev.start_stream()
        a, b = dev.read(), dev.read()
        assert a.first_sample_index == 0 and b.first_sample_index == 150
        assert b.first_device_time_s == pytest.approx(10.3)

    def test_contiguous_batches_leave_no_gap(self) -> None:
        batches = [lsl_batch(0.0, 10, 100.0, 1), lsl_batch(0.10, 10, 100.0, 1)]
        dev, _ = self.make(PPGDevice, batches)
        dev.connect()
        dev.start_stream()
        assert [dev.read().first_sample_index, dev.read().first_sample_index] == [0, 10]

    def test_empty_pull_returns_none_and_wrong_shape_errors(self) -> None:
        dev, _ = self.make(PPGDevice, [(np.zeros((4, 3)), [0, 0.01, 0.02, 0.03])])
        dev.connect()
        dev.start_stream()
        with pytest.raises(DeviceError, match="shape"):
            dev.read()
        dev2, _ = self.make(PPGDevice, [])
        dev2.connect()
        dev2.start_stream()
        assert dev2.read() is None

    def test_sync_probe_reads_the_lsl_clock(self) -> None:
        dev, inlet = self.make(EEGDevice, [])
        dev.connect()
        probe = dev.sync_probe()
        assert probe.device_time_s == 123.5 and probe.host_recv_ns >= probe.host_send_ns
        dev.disconnect()
        assert inlet.closed

    def test_without_pylsl_connecting_fails_cleanly(self) -> None:
        import importlib.util

        if importlib.util.find_spec("pylsl"):
            pytest.skip("pylsl is installed")
        dev = EEGDevice()
        with pytest.raises(DeviceError, match="pylsl"):
            dev.connect()


class TestHardwareRegistry:
    def test_eleven_classes_with_the_specified_priorities(self) -> None:
        reg = default_hardware_registry
        assert len(reg) == 11 == len(HARDWARE_CLASSES)
        priority = {c.key: c.priority for c in reg.all()}
        assert priority["eeg"] == priority["ecg"] == priority["behavioural"] == 1
        assert priority["eye_tracking"] == 2
        assert (
            priority["ppg"]
            == priority["respiration"]
            == priority["eda"]
            == priority["actigraphy"]
            == 3
        )
        protocols = {c.key: c.protocols for c in reg.all()}
        assert protocols["eeg"] == ("P00", "P01") and protocols["fmri"] == ("P03",)
        assert protocols["stimulation"] == ("P05",) and protocols["ieeg"] == ("P06",)

    def test_roles(self) -> None:
        roles = {c.key: c.role for c in default_hardware_registry.all()}
        assert roles["eye_tracking"] is Role.REALTIME_SURROGATE
        assert roles["eda"] is Role.OFFLINE_COVARIATE and roles["actigraphy"] is Role.CONTEXT
        assert {roles[k] for k in ("fmri", "stimulation", "ieeg")} == {Role.IMPORT_WORKFLOW}

    def test_priority_order_puts_import_workflows_last(self) -> None:
        keys = [c.key for c in default_hardware_registry.by_priority()]
        assert set(keys[-3:]) == {"fmri", "stimulation", "ieeg"}
        assert keys.index("eeg") < keys.index("eye_tracking") < keys.index("ppg")

    def test_shipped_drivers_are_bound_to_their_classes(self) -> None:
        reg = default_hardware_registry
        assert set(reg.drivers("eeg")) == {"lsl-eeg", "simulated-eeg"}
        assert "simulated-ecg" in reg.drivers("ecg") and "lsl-ppg" in reg.drivers("ppg")

    def test_unknown_class(self) -> None:
        with pytest.raises(UnknownHardwareClass):
            default_hardware_registry.get("telepathy")


class TestSurrogateRanking:
    def test_pupillometry_is_realtime_and_eda_is_offline(self) -> None:
        assert surrogate_usable_in_realtime("eye_tracking")
        assert not surrogate_usable_in_realtime("eda")
        assert SURROGATE_RANKING[0] == "eye_tracking" and SURROGATE_RANKING[-1] == "eda"

    def test_eda_lag_exceeds_the_inference_cadence(self) -> None:
        eda = default_hardware_registry.get("eda")
        assert (
            eda.physiological_lag_s is not None
            and eda.physiological_lag_s > INFERENCE_CADENCE_S * 10
        )


class TestStimulationGuard:
    @pytest.mark.parametrize("key", [c.key for c in HARDWARE_CLASSES])
    def test_no_class_is_controllable(self, key: str) -> None:
        with pytest.raises(StimulationControlError, match="cannot be controlled"):
            require_controllable(key)

    def test_no_live_driver_can_be_bound_to_an_import_workflow(self) -> None:
        reg = HardwareRegistry()
        with pytest.raises(StimulationControlError, match="import workflow"):
            reg.bind_driver("stimulation", SimulatedEEGDevice, "x")

    @pytest.mark.parametrize(
        "method", ["fire", "set_intensity", "bypass_interlock", "enable_output"]
    )
    def test_driver_with_an_output_control_surface_is_refused(self, method: str) -> None:
        driver = type("Rogue", (SimulatedEEGDevice,), {method: lambda self, *a, **k: None})
        with pytest.raises(StimulationControlError, match=method):
            HardwareRegistry().bind_driver("eeg", driver, "rogue")

    def test_shipped_drivers_expose_no_control_surface(self) -> None:
        from apgi_framework.acquisition.hardware import FORBIDDEN_CONTROL_SURFACE

        for cls in (
            SimulatedEEGDevice,
            SimulatedECGDevice,
            SimulatedPPGDevice,
            EEGDevice,
            ECGDevice,
            PPGDevice,
        ):
            assert not [n for n in FORBIDDEN_CONTROL_SURFACE if callable(getattr(cls, n, None))]
