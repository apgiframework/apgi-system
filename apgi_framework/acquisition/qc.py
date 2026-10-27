"""Quality-control metrics for a finished recording.

Four metrics: effective sampling rate, dropped samples, duplicate samples and
timestamp discontinuity. All are computed from the chunk index and the
recorded samples; nothing is corrected or interpolated.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .clock import ClockMapper
from .device import DeviceInfo
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


# ---------------------------------------------------------------------------
# The twelve per-session QC metrics
# ---------------------------------------------------------------------------
PASS, WARN, FAIL, NA = "pass", "warn", "fail", "n/a"

METRIC_NAMES = (
    "effective_sampling_rate",
    "dropped_samples",
    "duplicate_samples",
    "timestamp_discontinuity",
    "clock_offset_uncertainty",
    "electrode_impedance",
    "signal_saturation",
    "motion_artifacts",
    "cardiac_peak_quality",
    "marker_to_signal_latency",
    "upload_completeness",
    "firmware_calibration",
)


@dataclass(frozen=True)
class QCThresholds:
    """Limits used to turn metric values into pass/fail. A protocol may override any."""

    max_rate_error_ppm: float = 200.0
    max_clock_offset_uncertainty_s: float = 2e-3
    max_impedance_kohm: float = 10.0
    saturation_level: float = 1000.0  # |sample| at or above this counts as saturated
    max_saturated_fraction: float = 0.001
    motion_window_s: float = 1.0
    motion_ptp: float = 300.0  # peak-to-peak within a window that marks motion
    max_motion_fraction: float = 0.2
    min_cardiac_quality: float = 0.9
    max_marker_latency_s: float = 5e-3
    max_calibration_age_days: float = 365.0


@dataclass(frozen=True)
class MetricResult:
    name: str
    status: str  # pass | warn | fail | n/a
    value: Optional[float] = None
    unit: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class SessionQC:
    """All twelve metrics for one session, in a fixed order."""

    metrics: Tuple[MetricResult, ...]

    def __getitem__(self, name: str) -> MetricResult:
        for m in self.metrics:
            if m.name == name:
                return m
        raise KeyError(name)

    @property
    def failed(self) -> List[str]:
        return [m.name for m in self.metrics if m.status == FAIL]

    @property
    def passed(self) -> bool:
        return not self.failed

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "metrics": [m.to_dict() for m in self.metrics]}


def _kind_columns(recording: Recording, *kinds: str) -> List[int]:
    return [i for i, c in enumerate(recording.info.channels) if c.kind in kinds]


def _grade(value: float, limit: float, *, warn_at: float = 0.5) -> str:
    if value > limit:
        return FAIL
    return WARN if value > limit * warn_at else PASS


def _clock_metric(mapper: Optional[ClockMapper], th: QCThresholds) -> MetricResult:
    name = "clock_offset_uncertainty"
    if mapper is None:
        return MetricResult(name, NA, detail="no clock mapper supplied")
    fit = mapper.fit
    if fit is None:
        return MetricResult(name, FAIL, detail="no clock fit: not enough sync probes")
    # the midpoint assumption can be wrong by up to half the best round trip
    value = max(fit.residual_rms_s, fit.best_rtt_s / 2.0)
    return MetricResult(
        name,
        _grade(value, th.max_clock_offset_uncertainty_s),
        value,
        "s",
        f"residual rms {fit.residual_rms_s:.2e} s, best rtt/2 {fit.best_rtt_s / 2:.2e} s",
    )


def _impedance_metric(imp: Optional[Dict[str, float]], th: QCThresholds) -> MetricResult:
    name = "electrode_impedance"
    if not imp:
        return MetricResult(name, NA, detail="device reports no impedances")
    worst_ch, worst = max(imp.items(), key=lambda kv: kv[1])
    bad = sorted(ch for ch, v in imp.items() if v > th.max_impedance_kohm)
    status = FAIL if bad else (WARN if worst > th.max_impedance_kohm * 0.5 else PASS)
    return MetricResult(name, status, worst, "kOhm", f"worst {worst_ch}; over limit: {bad}")


def _saturation_metric(rec: Recording, th: QCThresholds) -> MetricResult:
    name = "signal_saturation"
    cols = _kind_columns(rec, "EEG", "ECG", "EOG", "EMG")
    if not cols:
        return MetricResult(name, NA, detail="no analogue channels")
    data = rec.samples[:, cols]
    frac = float(np.mean(np.abs(data) >= th.saturation_level))
    return MetricResult(
        name,
        _grade(frac, th.max_saturated_fraction),
        frac,
        "fraction",
        f"level {th.saturation_level}",
    )


def _motion_metric(rec: Recording, th: QCThresholds) -> MetricResult:
    name = "motion_artifacts"
    cols = _kind_columns(rec, "EEG")
    win = int(th.motion_window_s * rec.info.sampling_rate_hz)
    if not cols or rec.n_samples < win or win <= 0:
        return MetricResult(name, NA, detail="no EEG channels or recording shorter than one window")
    n_win = rec.n_samples // win
    block = rec.samples[: n_win * win, cols].reshape(n_win, win, len(cols))
    ptp = block.max(axis=1) - block.min(axis=1)  # (windows, channels)
    frac = float(np.mean(np.any(ptp > th.motion_ptp, axis=1)))
    return MetricResult(
        name,
        _grade(frac, th.max_motion_fraction),
        frac,
        "fraction of windows",
        f"peak-to-peak limit {th.motion_ptp}",
    )


def detect_r_peaks(signal: "np.ndarray", fs: float) -> "np.ndarray":
    """Indices of R peaks: local maxima above mean + 2 sd, at least 0.25 s apart."""
    x = np.asarray(signal, dtype=float)
    if x.size < 3 or float(np.std(x)) == 0.0:
        return np.array([], dtype=int)
    thr = float(np.mean(x) + 2.0 * np.std(x))
    candidates = np.flatnonzero((x[1:-1] > thr) & (x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:])) + 1
    peaks: List[int] = []
    min_gap = int(0.25 * fs)
    for idx in candidates:
        if not peaks or idx - peaks[-1] >= min_gap:
            peaks.append(int(idx))
        elif x[idx] > x[peaks[-1]]:
            peaks[-1] = int(idx)
    return np.array(peaks, dtype=int)


def _cardiac_metric(rec: Recording, th: QCThresholds) -> MetricResult:
    name = "cardiac_peak_quality"
    cols = _kind_columns(rec, "ECG")
    if not cols:
        return MetricResult(name, NA, detail="no ECG channel")
    fs = rec.info.sampling_rate_hz
    peaks = detect_r_peaks(rec.samples[:, cols[0]], fs)
    if len(peaks) < 3:
        return MetricResult(name, FAIL, 0.0, "fraction", f"only {len(peaks)} R peaks detected")
    rr = np.diff(peaks) / fs
    plausible = (rr >= 0.3) & (rr <= 2.0)  # 30-200 bpm
    median = float(np.median(rr))
    regular = np.abs(rr - median) <= 0.3 * median
    quality = float(np.mean(plausible & regular))
    status = PASS if quality >= th.min_cardiac_quality else (WARN if quality >= 0.7 else FAIL)
    return MetricResult(
        name,
        status,
        quality,
        "fraction of usable RR intervals",
        f"{len(peaks)} peaks, median RR {median:.3f} s",
    )


def _latency_metric(
    rec: Recording, mapper: Optional[ClockMapper], th: QCThresholds
) -> MetricResult:
    name = "marker_to_signal_latency"
    trig = _kind_columns(rec, "TRIG")
    hw = [m for m in rec.markers if m.source == "hardware" and m.code]
    if not trig or not hw or mapper is None:
        return MetricResult(
            name, NA, detail="needs a TRIG channel, hardware markers and a clock mapper"
        )
    fs = rec.info.sampling_rate_hz
    t0 = rec.sample0_device_time_s
    col = rec.samples[:, trig[0]]
    latencies: List[float] = []
    for m in hw:
        hits = np.flatnonzero(col == float(m.code))
        if hits.size == 0:
            continue
        marker_dev_t = mapper.host_to_device_s(m.onset_host_ns)
        sig_t = t0 + hits / fs
        latencies.append(float(sig_t[np.argmin(np.abs(sig_t - marker_dev_t))] - marker_dev_t))
    if not latencies:
        return MetricResult(
            name, FAIL, detail="hardware markers never appeared on the TRIG channel"
        )
    worst = float(max(abs(v) for v in latencies))
    return MetricResult(
        name,
        _grade(worst, th.max_marker_latency_s),
        worst,
        "s",
        f"{len(latencies)}/{len(hw)} markers matched; median {np.median(latencies):.2e} s",
    )


def _upload_metric(counts: Optional[Dict[str, int]], verified: Optional[bool]) -> MetricResult:
    name = "upload_completeness"
    if not counts:
        return MetricResult(name, NA, detail="no upload queue supplied")
    total = sum(counts.values())
    if total == 0:
        return MetricResult(name, FAIL, 0.0, "fraction", "nothing queued for upload")
    frac = counts.get("done", 0) / total
    if counts.get("failed", 0):
        status = FAIL
    elif frac == 1.0 and verified is not False:
        status = PASS
    else:
        status = WARN  # still pending: not an error, but not complete
    return MetricResult(name, status, frac, "fraction", f"{counts}; server_verified={verified}")


def _firmware_metric(info: DeviceInfo, th: QCThresholds, today: _dt.date) -> MetricResult:
    name = "firmware_calibration"
    if info.data_origin == "synthetic":
        return MetricResult(
            name, NA, detail="synthetic data has no hardware firmware or calibration"
        )
    problems: List[str] = []
    if not info.firmware:
        problems.append("firmware version not reported")
    cal = info.extra.get("calibration_date")
    age: Optional[float] = None
    if cal is None:
        problems.append("calibration date not reported")
    else:
        age = float((today - _dt.date.fromisoformat(str(cal))).days)
        if age > th.max_calibration_age_days:
            problems.append(f"calibration is {age:.0f} days old")
    return MetricResult(
        name,
        FAIL if problems else PASS,
        age,
        "days since calibration",
        "; ".join(problems) or f"firmware {info.firmware}",
    )


def compute_session_qc(
    recording: Recording,
    *,
    mapper: Optional[ClockMapper] = None,
    impedances_kohm: Optional[Dict[str, float]] = None,
    upload_counts: Optional[Dict[str, int]] = None,
    server_verified: Optional[bool] = None,
    thresholds: Optional[QCThresholds] = None,
    today: Optional[_dt.date] = None,
) -> SessionQC:
    """Compute all twelve QC metrics for one session (see :data:`METRIC_NAMES`)."""
    th = thresholds or QCThresholds()
    base = compute_qc(recording)
    if base.effective_rate_hz is None:
        rate = MetricResult("effective_sampling_rate", NA, detail="device reports no timestamps")
    else:
        status = FAIL if abs(base.rate_error_ppm or 0.0) > th.max_rate_error_ppm else PASS
        rate = MetricResult(
            "effective_sampling_rate",
            status,
            base.effective_rate_hz,
            "Hz",
            f"nominal {base.nominal_rate_hz} Hz, error {base.rate_error_ppm:.1f} ppm",
        )
    metrics = (
        rate,
        MetricResult(
            "dropped_samples",
            FAIL if base.dropped_samples else PASS,
            float(base.dropped_samples),
            "samples",
            f"gaps {list(base.gaps)}",
        ),
        MetricResult(
            "duplicate_samples",
            FAIL if base.duplicate_samples else PASS,
            float(base.duplicate_samples),
            "samples",
            f"{base.duplicate_chunks} chunks",
        ),
        MetricResult(
            "timestamp_discontinuity",
            FAIL if base.timestamp_discontinuities else PASS,
            base.max_timestamp_jump_s,
            "s",
            f"{base.timestamp_discontinuities} jumps",
        ),
        _clock_metric(mapper, th),
        _impedance_metric(
            impedances_kohm
            or {
                k: float(v) for k, v in (recording.info.extra.get("impedances_kohm") or {}).items()
            },
            th,
        ),
        _saturation_metric(recording, th),
        _motion_metric(recording, th),
        _cardiac_metric(recording, th),
        _latency_metric(recording, mapper, th),
        _upload_metric(upload_counts, server_verified),
        _firmware_metric(recording.info, th, today or _dt.date.today()),
    )
    assert tuple(m.name for m in metrics) == METRIC_NAMES
    return SessionQC(metrics)
