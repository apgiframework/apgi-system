"""Three-domain clock mapping: device, host-monotonic and UTC.

* device time (seconds): the amplifier's own clock, mapped by :class:`ClockMapper`;
* host monotonic (ns): ordering and durations inside a run, never jumps;
* UTC (ns since the Unix epoch): relates a run to other systems and wall time.

UTC is tied to the monotonic clock by *anchors*, ``(monotonic_ns, utc_ns)`` pairs read together.
Between anchors the monotonic clock is assumed to tick at the UTC rate; a later anchor
corrects accumulated difference (for example an NTP step) from that point on without
rewriting the past.
"""

from __future__ import annotations

import bisect
import threading
import time
from typing import List, Optional, Tuple

from .clock import ClockFit, ClockMapper
from .errors import ClockSyncError

__all__ = ["ClockFit", "ClockMapper", "ThreeDomainClock"]


class ThreeDomainClock:
    """Convert between device seconds, monotonic ns and UTC ns."""

    def __init__(self, mapper: ClockMapper) -> None:
        self.mapper = mapper
        self._anchors: List[Tuple[int, int]] = []  # sorted by monotonic_ns
        self._lock = threading.Lock()

    def add_anchor(
        self, monotonic_ns: Optional[int] = None, utc_ns: Optional[int] = None
    ) -> Tuple[int, int]:
        """Record a simultaneous reading of both clocks (defaults: read them now)."""
        if (monotonic_ns is None) != (utc_ns is None):
            raise ValueError("give both monotonic_ns and utc_ns, or neither")
        if monotonic_ns is None:
            monotonic_ns, utc_ns = time.monotonic_ns(), time.time_ns()
        assert utc_ns is not None
        with self._lock:
            bisect.insort(self._anchors, (monotonic_ns, utc_ns))
        return monotonic_ns, utc_ns

    def _anchor_for(self, monotonic_ns: int) -> Tuple[int, int]:
        with self._lock:
            if not self._anchors:
                raise ClockSyncError("no UTC anchor recorded; call add_anchor()")
            i = bisect.bisect_right(self._anchors, (monotonic_ns, float("inf"))) - 1  # type: ignore[arg-type]
            return self._anchors[max(i, 0)]

    def monotonic_to_utc_ns(self, monotonic_ns: int) -> int:
        m0, u0 = self._anchor_for(monotonic_ns)
        return u0 + (monotonic_ns - m0)

    def utc_to_monotonic_ns(self, utc_ns: int) -> int:
        with self._lock:
            anchors = list(self._anchors)
        if not anchors:
            raise ClockSyncError("no UTC anchor recorded; call add_anchor()")
        utcs = [u for _, u in anchors]
        i = max(bisect.bisect_right(utcs, utc_ns) - 1, 0)
        m0, u0 = anchors[i]
        return m0 + (utc_ns - u0)

    def device_to_utc_ns(self, device_time_s: float) -> int:
        return self.monotonic_to_utc_ns(self.mapper.device_to_host_ns(device_time_s))

    def utc_to_device_s(self, utc_ns: int) -> float:
        return self.mapper.host_to_device_s(self.utc_to_monotonic_ns(utc_ns))
