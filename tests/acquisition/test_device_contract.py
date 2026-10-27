"""Device interface: capabilities, honest timestamps and data_origin labelling.

Marked ``device_mock``: the CI gate that runs every driver check against the
simulated device only, so it needs no hardware.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from apgi_framework.acquisition import (
    ChannelSpec,
    DeviceCapabilities,
    DeviceError,
    DeviceInfo,
    SampleChunk,
    SimulatedDevice,
    SyncProbe,
    validate_bids_eeg,
)
from apgi_framework.acquisition.device import Device
from apgi_framework.acquisition.drivers.base import BaseDriver, SignalChunk
from tests.acquisition.conftest import build_session, record_session, requires_sqlcipher
from tests.unit.acquisition.helpers import FakeClock

pytestmark = pytest.mark.device_mock


class UntimedDevice(Device):
    """A driver for hardware with no clock; ``lie`` makes it invent device time."""

    def __init__(self, lie: bool = False) -> None:
        super().__init__()
        self.lie = lie
        self._info = DeviceInfo("raw-0", "untimed", "X", 100.0, (ChannelSpec("A"),))

    @property
    def info(self) -> DeviceInfo:
        return self._info

    @property
    def capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(device_timestamps=False, clock_sync=False)

    def _do_connect(self) -> None: ...
    def _do_disconnect(self) -> None: ...
    def _do_start_stream(self) -> None: ...
    def _do_stop_stream(self) -> None: ...

    def _do_read(self, max_samples: Optional[int]) -> Optional[SampleChunk]:
        return SampleChunk(
            "raw-0", 0, 0, 123.0 if self.lie else None, np.zeros((4, 1)), host_receive_ns=1
        )

    def _do_sync_probe(self) -> SyncProbe:
        raise DeviceError("no clock")


class TestInterface:
    def test_base_module_exposes_the_driver_names(self) -> None:
        assert BaseDriver is Device and SignalChunk is SampleChunk

    def test_simulated_capabilities(self) -> None:
        caps = SimulatedDevice(clock_ns=FakeClock()).capabilities
        assert caps == DeviceCapabilities(
            device_timestamps=True, hardware_markers=True, clock_sync=True, data_origin="synthetic"
        )

    def test_capabilities_reject_unknown_origin(self) -> None:
        with pytest.raises(ValueError):
            DeviceCapabilities(data_origin="made-up")


class TestDeviceTimestamps:
    def test_driver_without_a_clock_may_leave_time_empty(self) -> None:
        dev = UntimedDevice()
        dev.connect()
        dev.start_stream()
        chunk = dev.read()
        assert chunk is not None and chunk.first_device_time_s is None

    def test_driver_that_invents_a_time_is_rejected_and_marked_errored(self) -> None:
        dev = UntimedDevice(lie=True)
        dev.connect()
        dev.start_stream()
        with pytest.raises(DeviceError, match="no device timestamps"):
            dev.read()
        assert dev.state.value == "error"

    def test_timed_driver_may_not_omit_time(self) -> None:
        clock = FakeClock()
        dev = SimulatedDevice(clock_ns=clock)
        dev.connect()
        dev.start_stream()
        clock.advance(1.0)
        original = dev._do_read
        dev._do_read = lambda n: (  # type: ignore[method-assign]
            lambda c: SampleChunk(
                c.device_id, c.sequence, c.first_sample_index, None, c.samples, c.host_receive_ns
            )
        )(original(n))
        with pytest.raises(DeviceError, match="without"):
            dev.read()


class TestDataOrigin:
    def test_simulator_is_labelled_synthetic(self) -> None:
        assert SimulatedDevice(clock_ns=FakeClock()).info.data_origin == "synthetic"

    def test_default_origin_is_real(self) -> None:
        info = DeviceInfo("d", "usb-amp", "m", 100.0, (ChannelSpec("A"),))
        assert info.data_origin == "real"

    def test_invalid_origin_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="data_origin"):
            DeviceInfo("d", "x", "m", 100.0, (ChannelSpec("A"),), data_origin="simulated")

    def test_round_trip_and_legacy_default(self) -> None:
        info = SimulatedDevice(clock_ns=FakeClock()).info
        assert DeviceInfo.from_dict(info.to_dict()).data_origin == "synthetic"
        legacy = info.to_dict()
        del legacy["data_origin"]
        assert DeviceInfo.from_dict(legacy).data_origin == "synthetic"  # simulator driver
        legacy["driver"] = "usb-amp"
        assert DeviceInfo.from_dict(legacy).data_origin == "real"

    @requires_sqlcipher
    def test_label_reaches_recording_and_bids_sidecar(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import json

        from apgi_framework.acquisition import LocalRecorder

        clock, store, _, session = build_session(tmp_path)
        record_session(clock, session)
        meta = json.loads((session.directory / "recording.json").read_text())
        assert meta["device"]["data_origin"] == "synthetic"
        assert LocalRecorder.read(session.directory).info.data_origin == "synthetic"
        result = session.export_bids(tmp_path / "bids", task="rest")
        assert json.loads(result.sidecar.read_text())["DataOrigin"] == "synthetic"
        assert validate_bids_eeg(tmp_path / "bids") == []
        store.close()
