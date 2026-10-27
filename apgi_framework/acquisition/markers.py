"""Event markers stamped on the host clock.

Stimulus code stamps events with the *host* clock; the recorder stores them as
such and :class:`~apgi_framework.acquisition.clock.ClockMapper` converts them to
device time and sample indices on demand. A device that supports hardware
triggers can additionally write the code into its own stream.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .clock import ClockMapper
from .device import DeviceInfo


@dataclass(frozen=True)
class EventMarker:
    """One experimental event."""

    label: str
    onset_host_ns: int
    code: Optional[int] = None
    duration_s: float = 0.0
    value: Optional[str] = None
    source: str = "software"
    marker_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    extra: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("marker label must not be empty")
        if self.duration_s < 0:
            raise ValueError("duration_s must not be negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marker_id": self.marker_id,
            "label": self.label,
            "onset_host_ns": self.onset_host_ns,
            "code": self.code,
            "duration_s": self.duration_s,
            "value": self.value,
            "source": self.source,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventMarker":
        return cls(
            label=str(data["label"]),
            onset_host_ns=int(data["onset_host_ns"]),
            code=None if data.get("code") is None else int(data["code"]),
            duration_s=float(data.get("duration_s", 0.0)),
            value=None if data.get("value") is None else str(data["value"]),
            source=str(data.get("source", "software")),
            marker_id=str(data["marker_id"]),
            extra=dict(data.get("extra") or {}),
        )


MarkerListener = Callable[[EventMarker], None]


class MarkerLog:
    """Thread-safe, ordered collection of markers with subscriber callbacks."""

    def __init__(self, clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._clock_ns = clock_ns
        self._markers: List[EventMarker] = []
        self._listeners: List[MarkerListener] = []
        self._lock = threading.RLock()

    def subscribe(self, listener: MarkerListener) -> None:
        with self._lock:
            self._listeners.append(listener)

    def emit(
        self,
        label: str,
        *,
        code: Optional[int] = None,
        onset_host_ns: Optional[int] = None,
        duration_s: float = 0.0,
        value: Optional[str] = None,
        source: str = "software",
        **extra: Any,
    ) -> EventMarker:
        """Record an event now (or at an explicit ``onset_host_ns``)."""
        onset = self._clock_ns() if onset_host_ns is None else onset_host_ns
        marker = EventMarker(
            label=label,
            onset_host_ns=onset,
            code=code,
            duration_s=duration_s,
            value=value,
            source=source,
            extra=dict(extra),
        )
        with self._lock:
            self._markers.append(marker)
            listeners = list(self._listeners)
        for listener in listeners:
            listener(marker)
        return marker

    def snapshot(self) -> List[EventMarker]:
        with self._lock:
            return sorted(self._markers, key=lambda m: m.onset_host_ns)

    def between(self, start_ns: int, end_ns: int) -> List[EventMarker]:
        return [m for m in self.snapshot() if start_ns <= m.onset_host_ns < end_ns]

    def __len__(self) -> int:
        with self._lock:
            return len(self._markers)


def marker_device_time_s(marker: EventMarker, mapper: ClockMapper) -> float:
    """Onset of ``marker`` on the device clock."""
    return mapper.host_to_device_s(marker.onset_host_ns)


def marker_sample_index(
    marker: EventMarker,
    info: DeviceInfo,
    mapper: ClockMapper,
    first_sample_device_time_s: float,
) -> int:
    """Index of the sample at which ``marker`` falls (may be negative or past the end)."""
    delta = marker_device_time_s(marker, mapper) - first_sample_device_time_s
    return int(round(delta * info.sampling_rate_hz))
