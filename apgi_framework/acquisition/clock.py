"""Map a device clock onto the host clock.

A device's oscillator has an unknown offset and a slightly wrong rate. Each
:class:`~apgi_framework.acquisition.device.SyncProbe` gives one noisy
``(device_time, host_time)`` pair. :class:`ClockMapper` keeps a window of probes,
discards the ones with the slowest round trips (their midpoint assumption is the
least reliable, as in NTP's clock filter), and fits ``host = a + b * device`` by
least squares.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

from .device import SyncProbe
from .errors import ClockSyncError

_NS = 1_000_000_000


@dataclass(frozen=True)
class ClockFit:
    """Current mapping parameters, in host-relative terms."""

    n_points: int
    n_used: int
    drift_ppm: float
    residual_rms_s: float
    best_rtt_s: float


class ClockMapper:
    """Convert between device time (seconds) and host time (nanoseconds).

    Args:
        window: Number of most recent probes kept.
        keep_fraction: Share of the window, by lowest RTT, used for the fit.
        min_points: Probes needed before the drift term is trusted; below this
            only an offset is estimated (slope fixed at 1).
    """

    def __init__(
        self, *, window: int = 64, keep_fraction: float = 0.5, min_points: int = 4
    ) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError("keep_fraction must be in (0, 1]")
        self._window = window
        self._keep = keep_fraction
        self._min_points = max(2, min_points)
        self._probes: Deque[SyncProbe] = deque(maxlen=window)
        self._lock = threading.RLock()
        # fit: host_s_rel = intercept + slope * (device_s - dev_ref)
        self._dev_ref = 0.0
        self._host_ref_ns = 0
        self._slope = 1.0
        self._intercept = 0.0
        self._fit: Optional[ClockFit] = None

    @property
    def is_synced(self) -> bool:
        with self._lock:
            return self._fit is not None

    @property
    def n_probes(self) -> int:
        with self._lock:
            return len(self._probes)

    @property
    def fit(self) -> Optional[ClockFit]:
        with self._lock:
            return self._fit

    def reset(self) -> None:
        with self._lock:
            self._probes.clear()
            self._fit = None
            self._slope = 1.0
            self._intercept = 0.0

    def add_probe(self, probe: SyncProbe) -> None:
        with self._lock:
            self._probes.append(probe)
            self._refit()

    def device_to_host_ns(self, device_time_s: float) -> int:
        """Host clock reading (ns) at which the device clock read ``device_time_s``."""
        with self._lock:
            self._require_synced()
            rel_s = self._intercept + self._slope * (device_time_s - self._dev_ref)
            return self._host_ref_ns + int(round(rel_s * _NS))

    def host_to_device_s(self, host_ns: int) -> float:
        """Device clock reading (s) at host time ``host_ns``."""
        with self._lock:
            self._require_synced()
            rel_s = (host_ns - self._host_ref_ns) / _NS
            return self._dev_ref + (rel_s - self._intercept) / self._slope

    # -- internals --------------------------------------------------------
    def _require_synced(self) -> None:
        if self._fit is None:
            raise ClockSyncError("no sync probes recorded yet")

    def _refit(self) -> None:
        probes: List[SyncProbe] = list(self._probes)
        n_keep = max(
            self._min_points if len(probes) >= self._min_points else 1,
            int(np.ceil(len(probes) * self._keep)),
        )
        used = sorted(probes, key=lambda p: p.rtt_ns)[:n_keep]
        self._dev_ref = used[0].device_time_s
        self._host_ref_ns = used[0].host_mid_ns
        dev = np.array([p.device_time_s - self._dev_ref for p in used], dtype=np.float64)
        host = np.array([(p.host_mid_ns - self._host_ref_ns) / _NS for p in used], dtype=np.float64)
        spread = float(dev.max() - dev.min()) if len(dev) > 1 else 0.0
        if len(used) >= self._min_points and spread > 0.0:
            slope, intercept = np.polyfit(dev, host, 1)
            self._slope, self._intercept = float(slope), float(intercept)
        else:
            self._slope = 1.0
            self._intercept = float(np.mean(host - dev))
        resid = host - (self._intercept + self._slope * dev)
        self._fit = ClockFit(
            n_points=len(probes),
            n_used=len(used),
            drift_ppm=(1.0 / self._slope - 1.0) * 1e6,
            residual_rms_s=float(np.sqrt(np.mean(resid**2))),
            best_rtt_s=min(p.rtt_ns for p in probes) / _NS,
        )
