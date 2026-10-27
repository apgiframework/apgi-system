"""Quality-control metrics for a finished recording.

Four metrics: effective sampling rate, dropped samples, duplicate samples and
timestamp discontinuity. All are computed from the chunk index and the
recorded samples; nothing is corrected or interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .recorder import Recording


@dataclass(frozen=True)
class QCReport:
    nominal_rate_hz: float
    effective_rate_hz: Optional[float]  # None without device timestamps
    rate_error_ppm: Optional[float]
    dropped_samples: int
    gaps: Tuple[Tuple[int, int], ...]
    duplicate_samples: int
    duplicate_chunks: int
    timestamp_discontinuities: int
    max_timestamp_jump_s: float

    @property
    def passed(self) -> bool:
        return (
            self.dropped_samples == 0
            and self.duplicate_samples == 0
            and self.timestamp_discontinuities == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["gaps"] = [list(g) for g in self.gaps]
        out["passed"] = self.passed
        return out


def compute_qc(recording: Recording, *, discontinuity_tolerance_s: float = 0.5e-3) -> QCReport:
    """Compute the four QC metrics.

    A timestamp discontinuity is a chunk whose device time differs from the time
    implied by the previous chunk's start and sample count by more than the
    tolerance.
    """
    fs = recording.info.sampling_rate_hz
    chunks = sorted(recording.chunks, key=lambda c: c.first_sample_index)

    seen_end = 0
    dup_samples = 0
    dup_chunks = 0
    seen_seq = set()
    for c in chunks:
        overlap = max(0, min(seen_end, c.first_sample_index + c.n_samples) - c.first_sample_index)
        dup_samples += overlap
        if overlap or c.sequence in seen_seq:
            dup_chunks += 1
        seen_seq.add(c.sequence)
        seen_end = max(seen_end, c.first_sample_index + c.n_samples)

    dropped = sum(length for _, length in recording.gaps)

    jumps: List[float] = []
    effective: Optional[float] = None
    ppm: Optional[float] = None
    timed = [c for c in chunks if c.first_device_time_s is not None]
    if len(timed) == len(chunks) and len(timed) >= 2:
        for prev, cur in zip(timed, timed[1:]):
            expected = prev.first_device_time_s + (cur.first_sample_index - prev.first_sample_index) / fs  # type: ignore[operator]
            jumps.append(abs(cur.first_device_time_s - expected))  # type: ignore[operator]
        first, last = timed[0], timed[-1]
        span_s = last.first_device_time_s - first.first_device_time_s  # type: ignore[operator]
        span_n = last.first_sample_index - first.first_sample_index
        if span_s > 0 and span_n > 0:
            effective = span_n / span_s
            ppm = (effective - fs) / fs * 1e6
    return QCReport(
        nominal_rate_hz=fs,
        effective_rate_hz=effective,
        rate_error_ppm=ppm,
        dropped_samples=dropped,
        gaps=tuple(recording.gaps),
        duplicate_samples=dup_samples,
        duplicate_chunks=dup_chunks,
        timestamp_discontinuities=sum(1 for j in jumps if j > discontinuity_tolerance_s),
        max_timestamp_jump_s=max(jumps) if jumps else 0.0,
    )
