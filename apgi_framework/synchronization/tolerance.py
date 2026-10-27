"""Per-protocol timing tolerances.

Each preregistered protocol states how precise timing must be. A tolerance is registered once
under the protocol id and cannot be changed afterwards (a preregistration is not editable);
there is no default, so a protocol that never registered one fails loudly instead of being
checked against someone else's limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..acquisition.errors import AcquisitionError
from ..acquisition.qc import QCThresholds, SessionQC


class TimingViolation(AcquisitionError):
    """A session breaks its protocol's timing tolerance."""


@dataclass(frozen=True)
class TimingTolerance:
    protocol_id: str
    max_clock_offset_uncertainty_s: float
    max_marker_latency_s: float
    max_rate_error_ppm: float = 200.0
    max_timestamp_jump_s: float = 0.5e-3

    def __post_init__(self) -> None:
        if not self.protocol_id:
            raise ValueError("protocol_id must not be empty")
        for name in (
            "max_clock_offset_uncertainty_s",
            "max_marker_latency_s",
            "max_rate_error_ppm",
            "max_timestamp_jump_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def qc_thresholds(self, base: Optional[QCThresholds] = None) -> QCThresholds:
        """QC thresholds with this protocol's timing limits applied."""
        from dataclasses import replace

        return replace(
            base or QCThresholds(),
            max_clock_offset_uncertainty_s=self.max_clock_offset_uncertainty_s,
            max_marker_latency_s=self.max_marker_latency_s,
            max_rate_error_ppm=self.max_rate_error_ppm,
        )

    def violations(self, qc: SessionQC) -> List[str]:
        """Timing metrics of ``qc`` that fail this tolerance (``n/a`` metrics are not violations)."""
        names = (
            "effective_sampling_rate",
            "timestamp_discontinuity",
            "clock_offset_uncertainty",
            "marker_to_signal_latency",
        )
        out = [f"{n}: {qc[n].detail}" for n in names if qc[n].status == "fail"]
        jump = qc["timestamp_discontinuity"].value
        if jump is not None and jump > self.max_timestamp_jump_s:
            out.append(
                f"timestamp_discontinuity: jump {jump:.2e} s > {self.max_timestamp_jump_s:.2e} s"
            )
        return sorted(set(out))

    def enforce(self, qc: SessionQC) -> None:
        found = self.violations(qc)
        if found:
            raise TimingViolation(f"protocol {self.protocol_id} timing tolerance broken: {found}")


class ProtocolTimingRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, TimingTolerance] = {}

    def register(self, tolerance: TimingTolerance) -> None:
        existing = self._items.get(tolerance.protocol_id)
        if existing is not None and existing != tolerance:
            raise ValueError(
                f"{tolerance.protocol_id} already has a different registered tolerance"
            )
        self._items[tolerance.protocol_id] = tolerance

    def get(self, protocol_id: str) -> TimingTolerance:
        try:
            return self._items[protocol_id]
        except KeyError:
            raise KeyError(
                f"protocol {protocol_id!r} has no registered timing tolerance; register one explicitly"
            ) from None

    def __contains__(self, protocol_id: object) -> bool:
        return protocol_id in self._items
