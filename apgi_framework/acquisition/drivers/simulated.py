"""Deterministic simulated amplifier.

The simulator has its *own* clock (offset and drift relative to the host) so the
clock-mapping code can be exercised without hardware. Everything that depends on
time reads an injectable ``clock_ns`` callable, which tests replace with a fake
clock, so runs never sleep and are fully reproducible.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ..device import ChannelSpec, Device, DeviceInfo, SampleChunk, SyncProbe
from ..errors import DeviceError

_BLOCK = 256  # samples per independently seeded noise block
_DEFAULT_NAMES = ("Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2")


class SimulatedDevice(Device):
    """EEG-like source: a per-channel 10 Hz rhythm plus seeded Gaussian noise.

    Args:
        device_id: Identifier used in recordings and logs.
        sampling_rate_hz: Nominal sampling rate.
        channel_names: EEG channel names (default: eight 10-20 sites).
        seed: Seed for signal noise and probe jitter; same seed, same data.
        clock_ns: Host clock in nanoseconds (default ``time.monotonic_ns``).
        device_clock_offset_s: Device time at construction.
        drift_ppm: Device clock rate error; positive means the device runs fast.
        probe_rtt_ns: Base sync round-trip time.
        probe_jitter_ns: Standard deviation of extra random round-trip delay.
        probe_asymmetry_ns: Extra outbound delay; biases the midpoint estimate.
        max_chunk_samples: Largest chunk returned by one ``read``.
        drop_chunks: Chunk sequence numbers whose samples are lost in transit,
            leaving a gap in ``first_sample_index``.
        fail_after_samples: Make ``read`` raise once this many samples were sent.
        fail_on_connect: Make ``connect`` raise.
        trigger_channel: Append a ``TRIG`` channel carrying hardware markers.
    """

    def __init__(
        self,
        *,
        device_id: str = "sim-0",
        sampling_rate_hz: float = 250.0,
        channel_names: Optional[Sequence[str]] = None,
        seed: int = 0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        device_clock_offset_s: float = 1000.0,
        drift_ppm: float = 0.0,
        probe_rtt_ns: int = 400_000,
        probe_jitter_ns: int = 0,
        probe_asymmetry_ns: int = 0,
        max_chunk_samples: int = 64,
        drop_chunks: Sequence[int] = (),
        fail_after_samples: Optional[int] = None,
        fail_on_connect: bool = False,
        trigger_channel: bool = True,
        channel_kind: str = "EEG",
        channel_unit: str = "uV",
        model: str = "APGI-SIM",
        driver_name: str = "simulated",
    ) -> None:
        super().__init__()
        if max_chunk_samples <= 0:
            raise ValueError("max_chunk_samples must be positive")
        names = list(channel_names) if channel_names is not None else list(_DEFAULT_NAMES)
        channels: List[ChannelSpec] = [
            ChannelSpec(n, kind=channel_kind, unit=channel_unit) for n in names
        ]
        if trigger_channel:
            channels.append(ChannelSpec("TRIG", kind="TRIG", unit="n/a"))
        self._info = DeviceInfo(
            device_id=device_id,
            driver=driver_name,
            model=model,
            sampling_rate_hz=sampling_rate_hz,
            channels=tuple(channels),
            firmware="sim-1.0",
            serial=f"SIM-{seed:06d}",
            data_origin="synthetic",
        )
        self._n_eeg = len(names)
        self._has_trigger = trigger_channel
        self._seed = seed
        self._clock_ns = clock_ns
        self._offset_s = device_clock_offset_s
        self._drift = drift_ppm * 1e-6
        self._probe_rtt_ns = probe_rtt_ns
        self._probe_jitter_ns = probe_jitter_ns
        self._probe_asym_ns = probe_asymmetry_ns
        self._max_chunk = max_chunk_samples
        self._drop: FrozenSet[int] = frozenset(drop_chunks)
        self._fail_after = fail_after_samples
        self._fail_on_connect = fail_on_connect
        self._ref_ns = clock_ns()
        self._jitter_rng = np.random.default_rng([seed, 1])
        self._reset_stream()

    # -- Device -----------------------------------------------------------
    @property
    def info(self) -> DeviceInfo:
        return self._info

    @property
    def supports_hardware_markers(self) -> bool:
        return self._has_trigger

    def emit_hardware_marker(self, code: int) -> None:
        if not self._has_trigger:
            raise DeviceError("simulated device was built without a trigger channel")
        if not 0 < code < 65536:
            raise ValueError("trigger code must be in 1..65535")
        self._require_streaming_for_marker()
        self._triggers[self._available_index()] = code

    def _do_connect(self) -> None:
        if self._fail_on_connect:
            raise DeviceError("simulated connection failure")

    def _do_disconnect(self) -> None:
        return None

    def _do_start_stream(self) -> None:
        self._reset_stream()
        self._stream_start_ns = self._clock_ns()

    def _do_stop_stream(self) -> None:
        return None

    def _do_read(self, max_samples: Optional[int]) -> Optional[SampleChunk]:
        if self._fail_after is not None and self._sent >= self._fail_after:
            raise DeviceError("simulated read failure")
        limit = self._max_chunk if max_samples is None else min(max_samples, self._max_chunk)
        ready = self._available_index() - self._next_index
        n = min(ready, limit)
        if n <= 0:
            return None
        first = self._next_index
        self._next_index += n
        sequence = self._sequence
        self._sequence += 1
        if sequence in self._drop:
            return self._do_read(max_samples)  # samples lost; the index gap remains
        self._sent += n
        start_device_s = self._device_time_at(self._stream_start_ns)
        return SampleChunk(
            device_id=self._info.device_id,
            sequence=sequence,
            first_sample_index=first,
            first_device_time_s=start_device_s + first / self._info.sampling_rate_hz,
            samples=self._generate(first, n),
            host_receive_ns=self._clock_ns(),
        )

    def _do_sync_probe(self) -> SyncProbe:
        send = self._clock_ns()
        extra = 0
        if self._probe_jitter_ns > 0:
            extra = int(abs(self._jitter_rng.normal(0.0, self._probe_jitter_ns)))
        outbound = self._probe_rtt_ns // 2 + self._probe_asym_ns
        rtt = self._probe_rtt_ns + self._probe_asym_ns + extra
        return SyncProbe(
            host_send_ns=send,
            device_time_s=self._device_time_at(send + outbound),
            host_recv_ns=send + rtt,
        )

    # -- internals --------------------------------------------------------
    def _reset_stream(self) -> None:
        self._stream_start_ns = self._clock_ns()
        self._next_index = 0
        self._sequence = 0
        self._sent = 0
        self._triggers: Dict[int, int] = {}

    def _require_streaming_for_marker(self) -> None:
        from ..device import DeviceState

        if self.state is not DeviceState.STREAMING:
            raise DeviceError("hardware markers need an active stream")

    def _device_time_at(self, host_ns: int) -> float:
        return self._offset_s + (host_ns - self._ref_ns) * 1e-9 * (1.0 + self._drift)

    def _available_index(self) -> int:
        elapsed_dev_s = (self._clock_ns() - self._stream_start_ns) * 1e-9 * (1.0 + self._drift)
        return max(0, int(elapsed_dev_s * self._info.sampling_rate_hz))

    def _waveform(self, t: NDArray[np.float64], ch: int) -> NDArray[np.float64]:
        """Noise-free signal of analogue channel ``ch`` at times ``t`` (subclasses override)."""
        return 20.0 * np.sin(2 * np.pi * 10.0 * t + 0.3 * ch)

    def _noise_sd(self) -> float:
        return 5.0

    def _generate(self, first: int, n: int) -> NDArray[np.float64]:
        fs = self._info.sampling_rate_hz
        idx = np.arange(first, first + n)
        t = idx / fs
        out = np.zeros((n, self._info.n_channels), dtype=np.float64)
        for ch in range(self._n_eeg):
            out[:, ch] = self._waveform(t, ch)
        # noise is drawn per fixed block so values do not depend on read sizes
        noise = np.empty((n, self._n_eeg), dtype=np.float64)
        for block in range(first // _BLOCK, (first + n - 1) // _BLOCK + 1):
            rng = np.random.default_rng([self._seed, 2, block])
            block_noise = rng.normal(0.0, self._noise_sd(), size=(_BLOCK, self._n_eeg))
            lo = max(first, block * _BLOCK)
            hi = min(first + n, (block + 1) * _BLOCK)
            noise[lo - first : hi - first] = block_noise[lo - block * _BLOCK : hi - block * _BLOCK]
        out[:, : self._n_eeg] += noise
        if self._has_trigger:
            for sample_index, code in self._triggers.items():
                if first <= sample_index < first + n:
                    out[sample_index - first, self._n_eeg] = float(code)
        return out
