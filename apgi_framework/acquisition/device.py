"""Device interface for acquisition hardware.

A driver turns one physical (or simulated) amplifier into a stream of
:class:`SampleChunk` objects stamped with the *device's own* clock, and answers
clock-sync probes so the host can later map device time to host time.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
from numpy.typing import NDArray

from .errors import DeviceError

DATA_ORIGINS = ("real", "synthetic")


class DeviceState(str, enum.Enum):
    """Lifecycle of a device connection."""

    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


@dataclass(frozen=True)
class ChannelSpec:
    """One recorded channel."""

    name: str
    kind: str = "EEG"
    unit: str = "uV"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("channel name must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "unit": self.unit}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelSpec":
        return cls(
            name=str(data["name"]),
            kind=str(data.get("kind", "EEG")),
            unit=str(data.get("unit", "uV")),
        )


@dataclass(frozen=True)
class DeviceInfo:
    """Static description of a connected device."""

    device_id: str
    driver: str
    model: str
    sampling_rate_hz: float
    channels: Tuple[ChannelSpec, ...]
    firmware: str = ""
    serial: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict, compare=False)
    data_origin: str = "real"

    def __post_init__(self) -> None:
        if self.data_origin not in DATA_ORIGINS:
            raise ValueError(f"data_origin must be one of {DATA_ORIGINS}, got {self.data_origin!r}")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if not self.channels:
            raise ValueError("a device needs at least one channel")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "driver": self.driver,
            "model": self.model,
            "sampling_rate_hz": self.sampling_rate_hz,
            "channels": [c.to_dict() for c in self.channels],
            "firmware": self.firmware,
            "serial": self.serial,
            "extra": self.extra,
            "data_origin": self.data_origin,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceInfo":
        driver = str(data["driver"])
        # recordings written before data_origin existed: only the simulator was synthetic
        default_origin = "synthetic" if driver == "simulated" else "real"
        return cls(
            device_id=str(data["device_id"]),
            driver=str(data["driver"]),
            model=str(data["model"]),
            sampling_rate_hz=float(data["sampling_rate_hz"]),
            channels=tuple(ChannelSpec.from_dict(c) for c in data["channels"]),
            firmware=str(data.get("firmware", "")),
            serial=None if data.get("serial") is None else str(data["serial"]),
            extra=dict(data.get("extra") or {}),
            data_origin=str(data.get("data_origin", default_origin)),
        )

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def channel_names(self) -> Tuple[str, ...]:
        return tuple(c.name for c in self.channels)


@dataclass(frozen=True)
class DeviceCapabilities:
    """What a driver can and cannot provide.

    ``device_timestamps`` is ``False`` for hardware that reports no clock of its
    own. Such a driver must leave ``SampleChunk.first_device_time_s`` as ``None``;
    the base class rejects any chunk that invents a time.
    """

    device_timestamps: bool = True
    hardware_markers: bool = False
    clock_sync: bool = True
    data_origin: str = "real"

    def __post_init__(self) -> None:
        if self.data_origin not in DATA_ORIGINS:
            raise ValueError(f"data_origin must be one of {DATA_ORIGINS}, got {self.data_origin!r}")

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class SampleChunk:
    """A block of consecutive samples.

    ``samples`` has shape ``(n_samples, n_channels)``. ``first_device_time_s`` is
    the device-clock time of the first sample (``None`` when the device reports
    no time); later samples follow at the nominal sampling interval. ``first_sample_index`` counts samples since
    streaming started, so gaps (dropped chunks) are detectable.
    """

    device_id: str
    sequence: int
    first_sample_index: int
    first_device_time_s: Optional[float]
    samples: NDArray[np.float64]
    host_receive_ns: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 2:
            raise ValueError("samples must be 2-D (n_samples, n_channels)")

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])


@dataclass(frozen=True)
class SyncProbe:
    """One round trip used to relate the device clock to the host clock.

    The device clock was read somewhere between ``host_send_ns`` and
    ``host_recv_ns``; the host assumes the midpoint.
    """

    host_send_ns: int
    device_time_s: float
    host_recv_ns: int

    def __post_init__(self) -> None:
        if self.host_recv_ns < self.host_send_ns:
            raise ValueError("host_recv_ns precedes host_send_ns")

    @property
    def rtt_ns(self) -> int:
        return self.host_recv_ns - self.host_send_ns

    @property
    def host_mid_ns(self) -> int:
        return self.host_send_ns + self.rtt_ns // 2


class Device(ABC):
    """Abstract acquisition device.

    Drivers implement the ``_do_*`` hooks; this base class enforces the state
    machine (``DISCONNECTED -> CONNECTED -> STREAMING``) so every driver rejects
    out-of-order calls the same way.
    """

    def __init__(self) -> None:
        self._state = DeviceState.DISCONNECTED

    # -- description ------------------------------------------------------
    @property
    @abstractmethod
    def info(self) -> DeviceInfo:
        """Static description of the device."""

    @property
    def state(self) -> DeviceState:
        return self._state

    @property
    def supports_hardware_markers(self) -> bool:
        """Whether :meth:`emit_hardware_marker` writes into the data stream."""
        return False

    @property
    def capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            device_timestamps=True,
            hardware_markers=self.supports_hardware_markers,
            clock_sync=True,
            data_origin=self.info.data_origin,
        )

    # -- lifecycle --------------------------------------------------------
    def connect(self) -> None:
        self._require(DeviceState.DISCONNECTED, DeviceState.ERROR)
        self._guard(self._do_connect)
        self._state = DeviceState.CONNECTED

    def disconnect(self) -> None:
        if self._state is DeviceState.DISCONNECTED:
            return
        if self._state is DeviceState.STREAMING:
            self.stop_stream()
        try:
            self._do_disconnect()
        finally:
            self._state = DeviceState.DISCONNECTED

    def start_stream(self) -> None:
        self._require(DeviceState.CONNECTED)
        self._guard(self._do_start_stream)
        self._state = DeviceState.STREAMING

    def stop_stream(self) -> None:
        self._require(DeviceState.STREAMING)
        self._guard(self._do_stop_stream)
        self._state = DeviceState.CONNECTED

    # -- data and sync ----------------------------------------------------
    def read(self, max_samples: Optional[int] = None) -> Optional[SampleChunk]:
        """Return the next available chunk, or ``None`` if nothing is ready."""
        self._require(DeviceState.STREAMING)
        chunk = self._guard(lambda: self._do_read(max_samples))
        if chunk is not None:
            self._guard(lambda: self._check_timestamp(chunk))
        return chunk

    def sync_probe(self) -> SyncProbe:
        """Take one clock-sync measurement."""
        self._require(DeviceState.CONNECTED, DeviceState.STREAMING)
        return self._guard(self._do_sync_probe)

    def emit_hardware_marker(self, code: int) -> None:
        """Write a trigger code into the device's own data stream."""
        raise DeviceError(f"{self.info.driver} does not support hardware markers")

    # -- driver hooks -----------------------------------------------------
    @abstractmethod
    def _do_connect(self) -> None:
        """Driver hook."""

    @abstractmethod
    def _do_disconnect(self) -> None:
        """Driver hook."""

    @abstractmethod
    def _do_start_stream(self) -> None:
        """Driver hook."""

    @abstractmethod
    def _do_stop_stream(self) -> None:
        """Driver hook."""

    @abstractmethod
    def _do_read(self, max_samples: Optional[int]) -> Optional[SampleChunk]:
        """Driver hook."""

    @abstractmethod
    def _do_sync_probe(self) -> SyncProbe:
        """Driver hook."""

    # -- helpers ----------------------------------------------------------
    def _require(self, *allowed: DeviceState) -> None:
        if self._state not in allowed:
            names = ", ".join(s.value for s in allowed)
            raise DeviceError(
                f"{self.info.device_id}: operation needs state {names}, "
                f"device is {self._state.value}"
            )

    def _check_timestamp(self, chunk: SampleChunk) -> None:
        """Device time is reported by the device or absent, never made up by the host."""
        has_time = chunk.first_device_time_s is not None
        if has_time and not self.capabilities.device_timestamps:
            raise DeviceError(
                f"{self.info.device_id}: driver declares no device timestamps but stamped a chunk"
            )
        if not has_time and self.capabilities.device_timestamps:
            raise DeviceError(
                f"{self.info.device_id}: driver declares device timestamps but sent a chunk without"
            )

    def _guard(self, fn: Any) -> Any:
        """Run a driver hook; any non-DeviceError failure marks the device ERROR."""
        try:
            return fn()
        except DeviceError:
            self._state = DeviceState.ERROR
            raise
        except Exception as exc:
            self._state = DeviceState.ERROR
            raise DeviceError(f"{self.info.device_id}: driver failure: {exc}") from exc

    def __enter__(self) -> "Device":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.disconnect()
