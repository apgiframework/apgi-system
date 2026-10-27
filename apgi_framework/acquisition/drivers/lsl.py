"""Lab Streaming Layer (LSL) adapter shared by the EEG, ECG and PPG drivers.

LSL stamps every sample with the sender's clock, so this driver never invents time: a
chunk's ``first_device_time_s`` is the timestamp LSL delivered. LSL has no sample counter,
so ``first_sample_index`` is derived: samples received so far, plus the number of samples
implied by any timestamp gap larger than 1.5 sampling intervals (that gap is what the QC
metrics report as dropped samples).

``pylsl`` is optional. Pass ``inlet_factory`` (any object with ``pull_chunk(timeout,
max_samples) -> (samples, timestamps)``) to use another source or a test double.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..device import ChannelSpec, Device, DeviceInfo, SampleChunk, SyncProbe
from ..errors import DeviceError

InletFactory = Callable[[], Any]


def _pylsl_inlet_factory(stream_type: str, timeout_s: float = 5.0) -> InletFactory:
    def factory() -> Any:
        try:
            import pylsl  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DeviceError("pylsl is not installed (pip install pylsl)") from exc
        found = pylsl.resolve_byprop("type", stream_type, timeout=timeout_s)
        if not found:
            raise DeviceError(f"no LSL stream of type {stream_type!r} found")
        return pylsl.StreamInlet(found[0])

    return factory


class LslDevice(Device):
    """Device reading one LSL stream."""

    def __init__(
        self,
        *,
        device_id: str,
        stream_type: str,
        kind: str,
        unit: str,
        sampling_rate_hz: float,
        channel_names: Sequence[str],
        driver_name: str,
        model: str = "LSL",
        inlet_factory: Optional[InletFactory] = None,
        lsl_clock: Optional[Callable[[], float]] = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_chunk_samples: int = 64,
    ) -> None:
        super().__init__()
        self._info = DeviceInfo(
            device_id=device_id,
            driver=driver_name,
            model=model,
            sampling_rate_hz=sampling_rate_hz,
            channels=tuple(ChannelSpec(n, kind=kind, unit=unit) for n in channel_names),
            extra={"lsl_stream_type": stream_type},
        )
        self._factory = inlet_factory or _pylsl_inlet_factory(stream_type)
        self._lsl_clock = lsl_clock
        self._clock_ns = clock_ns
        self._max = max_chunk_samples
        self._inlet: Any = None
        self._sequence = 0
        self._next_index = 0
        self._next_time: Optional[float] = None

    @property
    def info(self) -> DeviceInfo:
        return self._info

    def _do_connect(self) -> None:
        self._inlet = self._factory()

    def _do_disconnect(self) -> None:
        close = getattr(self._inlet, "close_stream", None)
        if callable(close):
            close()
        self._inlet = None

    def _do_start_stream(self) -> None:
        self._sequence = 0
        self._next_index = 0
        self._next_time = None

    def _do_stop_stream(self) -> None:
        return None

    def _do_read(self, max_samples: Optional[int]) -> Optional[SampleChunk]:
        limit = self._max if max_samples is None else min(max_samples, self._max)
        samples, stamps = self._inlet.pull_chunk(timeout=0.0, max_samples=limit)
        if not len(samples):
            return None
        data = np.asarray(samples, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self._info.n_channels:
            raise DeviceError(
                f"LSL chunk shape {data.shape} does not match {self._info.n_channels} channels"
            )
        first_time = float(stamps[0])
        fs = self._info.sampling_rate_hz
        if self._next_time is not None:
            gap_s = first_time - self._next_time
            if gap_s > 1.5 / fs:  # samples were lost upstream; leave an index gap
                self._next_index += int(round(gap_s * fs))
        chunk = SampleChunk(
            device_id=self._info.device_id,
            sequence=self._sequence,
            first_sample_index=self._next_index,
            first_device_time_s=first_time,
            samples=data,
            host_receive_ns=self._clock_ns(),
        )
        self._sequence += 1
        self._next_index += data.shape[0]
        self._next_time = float(stamps[-1]) + 1.0 / fs
        return chunk

    def _do_sync_probe(self) -> SyncProbe:
        if self._lsl_clock is None:
            try:
                import pylsl  # type: ignore[import-not-found]
            except ImportError as exc:
                raise DeviceError("pylsl is not installed; cannot read the LSL clock") from exc
            self._lsl_clock = pylsl.local_clock
        send = self._clock_ns()
        device_time = float(self._lsl_clock())
        return SyncProbe(
            host_send_ns=send, device_time_s=device_time, host_recv_ns=self._clock_ns()
        )
