"""Monitoring for the APGI Framework.

Performance tracking (:class:`PerformanceMonitor`), lightweight profiling
(:class:`SystemProfiler`), metric collection and export (:class:`MetricsCollector`) and a
simple real-time data buffer (:class:`RealTimeMonitor`).

The performance classes accept two usage styles:

* per-step timing: ``start_step()`` / ``end_step(...)`` with system time in the history;
* state snapshots: ``record_metrics(system_state, step_time_ms)`` from a ``System.step()`` dict.
"""

import contextlib
import functools
import gc
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import psutil


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class PerformanceMetrics:
    """Performance metrics for one step.

    Attributes
    ----------
    step_time_ms : float
        Time taken for a single system step in milliseconds
    memory_usage_mb : float
        Current memory usage in megabytes
    ignition_rate_hz : float
        Rate of ignition events per second
    free_energy_mean : float
        Mean free energy value
    precision_mean : float
        Mean precision value
    timestamp : float
        System time when the metrics were recorded (milliseconds)
    """

    step_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    ignition_rate_hz: float = 0.0
    free_energy_mean: float = 0.0
    precision_mean: float = 0.0
    timestamp: float = 0.0


class PerformanceMonitor:
    """Track execution time, memory usage and ignition rate over time.

    Parameters
    ----------
    config : Dict[str, Any], optional
        Settings. ``monitoring.enabled`` / ``monitoring.max_history_size`` control recording;
        top-level ``sampling_interval`` (ms), ``memory_threshold_mb``, ``enable_profiling``,
        ``step_time_threshold_ms`` and ``ignition_rate_threshold_hz`` configure analysis.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config if config is not None else {}
        monitoring_config = self.config.get("monitoring", {})
        self.enabled: bool = monitoring_config.get("enabled", True)
        self.max_history_size: int = monitoring_config.get("max_history_size", 1000)

        self.sampling_interval: float = self.config.get("sampling_interval", 1000)
        self.memory_threshold_mb: float = self.config.get("memory_threshold_mb", 2000)
        self.enable_profiling: bool = self.config.get("enable_profiling", False)
        self.step_time_threshold_ms: float = self.config.get("step_time_threshold_ms", 100.0)
        self.ignition_rate_threshold_hz: float = self.config.get("ignition_rate_threshold_hz", 0.0)

        self.history: List[PerformanceMetrics] = []
        self.step_start_time: Optional[float] = None
        self.last_step_time_ms: float = 0.0
        self.process = psutil.Process()

        # Ignition events (system time, ms) used for the sliding-window rate
        self.ignition_times: List[float] = []
        self.ignition_window_ms: float = 1000.0

        # Free-form metric log (start_monitoring / record_metric)
        self.performance_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.monitoring_active: bool = False

    @property
    def metrics_history(self) -> List[PerformanceMetrics]:
        """Alias of :attr:`history`."""
        return self.history

    @metrics_history.setter
    def metrics_history(self, value: List[PerformanceMetrics]) -> None:
        self.history = value

    # -- per-step timing -------------------------------------------------------------------
    def start_step(self) -> None:
        """Mark the start of a system step for timing."""
        if self.enabled:
            self.step_start_time = time.perf_counter()

    def end_step(
        self, system_time_ms: float, ignition_occurred: bool, free_energy: float, precision: float
    ) -> Optional[PerformanceMetrics]:
        """Mark the end of a step and record its metrics (``None`` if disabled or not started)."""
        if not self.enabled or self.step_start_time is None:
            return None

        self.last_step_time_ms = (time.perf_counter() - self.step_start_time) * 1000.0
        self.step_start_time = None
        return self._record(
            self.last_step_time_ms, system_time_ms, ignition_occurred, free_energy, precision
        )

    # -- snapshot style --------------------------------------------------------------------
    def record_metrics(self, system_state: Dict[str, Any], step_time_ms: float) -> Any:
        """Record metrics from a ``System.step()`` state dictionary."""
        if not self.enabled:
            return None
        self.last_step_time_ms = float(step_time_ms)
        return self._record(
            float(step_time_ms),
            float(system_state.get("time", time.time() * 1000.0)),
            bool(system_state.get("ignition", {}).get("ignition_occurred", False)),
            float(system_state.get("free_energy", {}).get("total_free_energy", 0.0)),
            float(system_state.get("precision", {}).get("mean_precision", 0.0)),
        )

    def _record(
        self,
        step_time_ms: float,
        system_time_ms: float,
        ignition_occurred: bool,
        free_energy: float,
        precision: float,
    ) -> PerformanceMetrics:
        memory_usage_mb = self.process.memory_info().rss / (1024 * 1024)
        if ignition_occurred:
            self.ignition_times.append(system_time_ms)

        metrics = PerformanceMetrics(
            step_time_ms=step_time_ms,
            memory_usage_mb=memory_usage_mb,
            ignition_rate_hz=self._calculate_ignition_rate(system_time_ms),
            free_energy_mean=free_energy,
            precision_mean=precision,
            timestamp=system_time_ms,
        )
        self.history.append(metrics)
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size :]
        return metrics

    def _calculate_ignition_rate(self, current_time_ms: float) -> float:
        """Ignition events per second inside the sliding window ending at ``current_time_ms``."""
        cutoff_time = current_time_ms - self.ignition_window_ms
        self.ignition_times = [t for t in self.ignition_times if t >= cutoff_time]
        if not self.ignition_times:
            return 0.0
        return len(self.ignition_times) / (self.ignition_window_ms / 1000.0)

    # -- analysis ------------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        """Summary statistics (mean/max/min step time, memory, ignition rate, sample count)."""
        if not self.history:
            return {
                "mean_step_time_ms": 0.0,
                "max_step_time_ms": 0.0,
                "min_step_time_ms": 0.0,
                "std_step_time_ms": 0.0,
                "mean_memory_mb": 0.0,
                "max_memory_mb": 0.0,
                "mean_ignition_rate_hz": 0.0,
                "total_samples": 0,
            }

        step_times = [m.step_time_ms for m in self.history]
        memory_usage = [m.memory_usage_mb for m in self.history]
        ignition_rates = [m.ignition_rate_hz for m in self.history]
        return {
            "mean_step_time_ms": float(np.mean(step_times)),
            "max_step_time_ms": float(np.max(step_times)),
            "min_step_time_ms": float(np.min(step_times)),
            "std_step_time_ms": float(np.std(step_times)),
            "mean_memory_mb": float(np.mean(memory_usage)),
            "max_memory_mb": float(np.max(memory_usage)),
            "min_memory_mb": float(np.min(memory_usage)),
            "mean_ignition_rate_hz": float(np.mean(ignition_rates)),
            "max_ignition_rate_hz": float(np.max(ignition_rates)),
            "total_samples": len(self.history),
        }

    def get_average_metrics(self) -> Dict[str, float]:
        """Mean of every metric in the history (zeros when empty), rounded to 12 places."""
        names = (
            "step_time_ms",
            "memory_usage_mb",
            "ignition_rate_hz",
            "free_energy_mean",
            "precision_mean",
        )
        if not self.history:
            return {name: 0.0 for name in names}
        return {
            name: round(float(np.mean([getattr(m, name) for m in self.history])), 12)
            for name in names
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Samples, covered time range, averages, peaks and simple trends."""
        if not self.history:
            return {
                "total_samples": 0,
                "time_range_ms": 0.0,
                "average_metrics": self.get_average_metrics(),
                "peak_metrics": {},
                "performance_trends": {},
            }

        timestamps = [m.timestamp for m in self.history]
        peaks = {
            "max_step_time_ms": max(m.step_time_ms for m in self.history),
            "max_memory_usage_mb": max(m.memory_usage_mb for m in self.history),
            "max_ignition_rate_hz": max(m.ignition_rate_hz for m in self.history),
            "max_free_energy_mean": max(m.free_energy_mean for m in self.history),
        }
        return {
            "total_samples": len(self.history),
            "time_range_ms": float(max(timestamps) - min(timestamps)),
            "average_metrics": self.get_average_metrics(),
            "peak_metrics": peaks,
            "performance_trends": {
                "step_time": self._trend([m.step_time_ms for m in self.history]),
                "memory_usage": self._trend([m.memory_usage_mb for m in self.history]),
            },
        }

    @staticmethod
    def _trend(values: List[float], tolerance: float = 0.05) -> str:
        """``increasing`` / ``decreasing`` / ``stable`` comparing first and second halves."""
        if len(values) < 2:
            return "stable"
        half = len(values) // 2
        first, second = float(np.mean(values[:half])), float(np.mean(values[-half:]))
        scale = max(abs(first), 1e-12)
        change = (second - first) / scale
        if change > tolerance:
            return "increasing"
        if change < -tolerance:
            return "decreasing"
        return "stable"

    def detect_performance_issues(self) -> List[str]:
        """Human-readable warnings when the latest sample crosses a configured threshold."""
        if not self.history:
            return []
        latest = self.history[-1]
        issues: List[str] = []
        if latest.step_time_ms > self.step_time_threshold_ms:
            issues.append(
                f"High step time: {latest.step_time_ms:.1f} ms "
                f"(threshold {self.step_time_threshold_ms:.1f} ms)"
            )
        if latest.memory_usage_mb > self.memory_threshold_mb:
            issues.append(
                f"High memory usage: {latest.memory_usage_mb:.1f} MB "
                f"(threshold {self.memory_threshold_mb:.1f} MB)"
            )
        if latest.ignition_rate_hz < self.ignition_rate_threshold_hz:
            issues.append(
                f"Low ignition rate: {latest.ignition_rate_hz:.2f} Hz "
                f"(threshold {self.ignition_rate_threshold_hz:.2f} Hz)"
            )
        return issues

    def get_recent_metrics(self, n: int = 10) -> List[PerformanceMetrics]:
        """The most recent ``n`` metrics."""
        return self.history[-n:] if self.history else []

    def reset(self) -> None:
        """Reset recorded state."""
        self.history.clear()
        self.ignition_times.clear()
        self.step_start_time = None
        self.last_step_time_ms = 0.0

    def reset_metrics(self) -> None:
        """Clear the metrics history (alias of :meth:`reset`)."""
        self.reset()

    def check_enabled(self) -> bool:
        """Whether monitoring is enabled."""
        return self.enabled

    def log_performance(self, verbose: bool = False) -> None:
        """Print current performance statistics."""
        if not self.enabled or not self.history:
            return

        stats = self.get_statistics()
        print("\n=== Performance Statistics ===")
        print(f"Total samples: {stats['total_samples']}")
        print(
            f"Step time: {stats['mean_step_time_ms']:.3f} ms "
            f"(min: {stats['min_step_time_ms']:.3f}, max: {stats['max_step_time_ms']:.3f})"
        )
        print(
            f"Memory usage: {stats['mean_memory_mb']:.1f} MB (max: {stats['max_memory_mb']:.1f} MB)"
        )
        print(
            f"Ignition rate: {stats['mean_ignition_rate_hz']:.2f} Hz "
            f"(max: {stats['max_ignition_rate_hz']:.2f} Hz)"
        )
        if verbose:
            print("\nRecent metrics (last 5):")
            for i, m in enumerate(self.get_recent_metrics(5), 1):
                print(
                    f"  {i}. Time: {m.timestamp:.1f}ms, Step: {m.step_time_ms:.3f}ms, "
                    f"Mem: {m.memory_usage_mb:.1f}MB"
                )

    # -- free-form metric log --------------------------------------------------------------
    def start_monitoring(self) -> Dict[str, str]:
        """Start recording free-form metrics."""
        self.monitoring_active = True
        return {"status": "started", "timestamp": _now_iso()}

    def stop_monitoring(self) -> Dict[str, str]:
        """Stop recording free-form metrics."""
        self.monitoring_active = False
        return {"status": "stopped", "timestamp": _now_iso()}

    def record_metric(self, metric_name: str, value: float) -> None:
        """Record a named metric while monitoring is active."""
        if self.monitoring_active:
            self.performance_metrics.setdefault(metric_name, []).append(
                {"value": value, "timestamp": _now_iso()}
            )

    def get_metrics(self, metric_name: Optional[str] = None) -> Any:
        """Free-form metrics (one series if ``metric_name`` is given)."""
        if metric_name:
            return self.performance_metrics.get(metric_name, [])
        return self.performance_metrics


class SystemProfiler:
    """Profile functions, code sections and system resources."""

    def __init__(self) -> None:
        self.profile_data: Dict[str, Dict[str, Any]] = {}
        self.is_profiling: bool = False
        self.start_time: Optional[float] = None
        self._created_at = time.time()
        self._started_perf: Optional[float] = None
        # Named profile records (record_profile / get_profiles)
        self.profiles: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def profiling_active(self) -> bool:
        """Alias of :attr:`is_profiling`."""
        return self.is_profiling

    def start_profiling(self) -> Dict[str, str]:
        """Start a profiling window."""
        self.is_profiling = True
        self.start_time = time.time()
        self._started_perf = time.perf_counter()
        return {"status": "started", "timestamp": _now_iso()}

    def stop_profiling(self) -> Dict[str, Any]:
        """End the profiling window; includes ``total_time_ms`` since :meth:`start_profiling`."""
        # perf_counter has sub-microsecond resolution; time.time() ticks at ~15 ms on Windows
        started = getattr(self, "_started_perf", None)
        total_ms = (time.perf_counter() - started) * 1000.0 if started is not None else 0.0
        self._started_perf = None
        self.is_profiling = False
        self.start_time = None
        return {"status": "stopped", "timestamp": _now_iso(), "total_time_ms": total_ms}

    def _add_time(self, name: str, elapsed_ms: float) -> None:
        entry = self.profile_data.setdefault(name, {"call_count": 0, "total_time_ms": 0.0})
        entry["call_count"] = entry.get("call_count", 0) + 1
        entry["total_time_ms"] = entry.get("total_time_ms", 0.0) + elapsed_ms

    def profile_function(self, name: Optional[str] = None) -> Callable[[Callable], Callable]:
        """Decorator recording call count and total time of a function."""

        def decorator(func: Callable) -> Callable:
            label = name or func.__name__

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    self._add_time(label, (time.perf_counter() - start) * 1000.0)

            return wrapper

        return decorator

    @contextlib.contextmanager
    def profile_section(self, name: str) -> Iterator[None]:
        """Context manager recording the time spent in a code section."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self._add_time(name, (time.perf_counter() - start) * 1000.0)

    def get_profile_summary(self, top: int = 5) -> Dict[str, Any]:
        """Number of profiled items, total time and the ``top`` most expensive ones."""
        ranked = sorted(
            self.profile_data.items(), key=lambda kv: kv[1].get("total_time_ms", 0.0), reverse=True
        )
        return {
            "total_functions": len(self.profile_data),
            "total_time_ms": float(
                sum(d.get("total_time_ms", 0.0) for d in self.profile_data.values())
            ),
            "top_functions": [{"name": key, **value} for key, value in ranked[:top]],
        }

    def record_profile(self, profile_name: str, profile_data: Dict[str, Any]) -> None:
        """Record a profile data point while profiling is active."""
        if self.is_profiling:
            self.profiles.setdefault(profile_name, []).append(profile_data)

    def get_profiles(self, profile_name: Optional[str] = None) -> Any:
        """Recorded profile data points."""
        if profile_name:
            return self.profiles.get(profile_name, [])
        return self.profiles

    def profile_system_resources(self) -> Dict[str, Any]:
        """Current system resource usage."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "process_count": len(psutil.pids()),
            "uptime_seconds": time.time() - (self.start_time or self._created_at),
        }

    def profile_python_memory(self) -> Dict[str, Any]:
        """Python object/memory statistics (``ref_count`` only exists on debug builds)."""
        gettotalrefcount = getattr(sys, "gettotalrefcount", None)
        return {
            "objects_count": len(gc.get_objects()),
            "ref_count": gettotalrefcount() if gettotalrefcount else None,
            "gc_stats": gc.get_stats(),
        }


class MetricsCollector:
    """Collect, summarise and export metrics from system states and from metric sources."""

    def __init__(self, collection_interval: float = 1.0) -> None:
        self.collection_interval = collection_interval
        self.metrics: Dict[str, List[float]] = {}
        self.timestamps: List[float] = []
        self.metrics_sources: List[Any] = []
        self.collected_metrics: Dict[str, List[float]] = {}
        self.collection_active: bool = False

    # -- state snapshots ---------------------------------------------------------------------
    def collect_metrics(self, system_state: Dict[str, Any]) -> None:
        """Append one sample taken from a ``System.step()`` state dictionary."""
        workspace = system_state.get("workspace", {})
        sample = {
            "ignition_signal": system_state.get("ignition", {}).get("total_signal", 0.0),
            "free_energy": system_state.get("free_energy", {}).get("total_free_energy", 0.0),
            "precision": system_state.get("precision", {}).get("mean_precision", 0.0),
            "workspace_activity": (
                workspace.get("broadcast_strength", 0.0)
                if workspace.get("is_broadcasting")
                else 0.0
            ),
        }
        self.timestamps.append(float(system_state.get("time", time.time() * 1000.0)))
        for name, value in sample.items():
            self.metrics.setdefault(name, []).append(float(value))

    def get_summary(self) -> Dict[str, Any]:
        """Collection period, sample count and mean/min/max/std of every metric."""
        period = float(self.timestamps[-1] - self.timestamps[0]) if self.timestamps else 0.0
        return {
            "collection_period_ms": period,
            "total_samples": len(self.timestamps),
            "metrics_summary": {
                name: {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "std": float(np.std(values)),
                }
                for name, values in self.metrics.items()
                if values
            },
        }

    def export_csv(self) -> str:
        """Metrics as CSV text: a ``timestamp`` column then one column per metric."""
        names = list(self.metrics)
        lines = [",".join(["timestamp", *names])]
        for i, stamp in enumerate(self.timestamps):
            row = [str(stamp)] + [
                str(self.metrics[name][i]) if i < len(self.metrics[name]) else "" for name in names
            ]
            lines.append(",".join(row))
        return "\n".join(lines) + "\n"

    def export_json(self) -> str:
        """Metrics as JSON text with ``timestamps`` and ``metrics`` keys."""
        return json.dumps({"timestamps": self.timestamps, "metrics": self.metrics}, indent=2)

    def clear(self) -> None:
        """Drop all collected samples."""
        self.metrics.clear()
        self.timestamps.clear()
        self.collected_metrics.clear()

    # -- metric sources ----------------------------------------------------------------------
    def add_metrics_source(self, source: Any) -> None:
        """Add an object with ``get_metrics()`` or ``get_statistics()`` to collect from."""
        self.metrics_sources.append(source)

    def collect_all_metrics(self) -> Dict[str, Any]:
        """Metrics from every registered source, keyed by source class name."""
        all_metrics: Dict[str, Any] = {}
        for source in self.metrics_sources:
            name = source.__class__.__name__
            try:
                if hasattr(source, "get_metrics"):
                    all_metrics[name] = source.get_metrics()
                elif hasattr(source, "get_statistics"):
                    all_metrics[name] = source.get_statistics()
            except Exception as e:
                all_metrics[name] = {"error": str(e)}
        return all_metrics

    def get_metrics_summary(self) -> Dict[str, Any]:
        """How many registered sources reported successfully."""
        all_metrics = self.collect_all_metrics()
        failed = sum(1 for m in all_metrics.values() if isinstance(m, dict) and "error" in m)
        return {
            "total_sources": len(self.metrics_sources),
            "successful_collections": len(all_metrics) - failed,
            "failed_collections": failed,
            "timestamp": time.time(),
        }

    # -- named metric log ----------------------------------------------------------------------
    def start_collection(self) -> Dict[str, str]:
        """Start recording named metrics."""
        self.collection_active = True
        return {"status": "started", "timestamp": _now_iso()}

    def stop_collection(self) -> Dict[str, str]:
        """Stop recording named metrics."""
        self.collection_active = False
        return {"status": "stopped", "timestamp": _now_iso()}

    def collect_metric(self, metric_name: str, value: float) -> None:
        """Record one named value while collection is active."""
        if self.collection_active:
            self.collected_metrics.setdefault(metric_name, []).append(value)

    def get_collected_metrics(self, metric_name: Optional[str] = None) -> Any:
        """Named metric values (one series if ``metric_name`` is given)."""
        if metric_name:
            return self.collected_metrics.get(metric_name, [])
        return self.collected_metrics


class RealTimeMonitor:
    """Bounded buffer of recent data points for real-time display."""

    def __init__(self) -> None:
        self.monitoring_active: bool = False
        self.data_buffer: List[Any] = []

    def start_monitoring(self) -> Dict[str, str]:
        """Start real-time monitoring."""
        self.monitoring_active = True
        return {"status": "started", "timestamp": _now_iso()}

    def stop_monitoring(self) -> Dict[str, str]:
        """Stop real-time monitoring."""
        self.monitoring_active = False
        return {"status": "stopped", "timestamp": _now_iso()}

    def add_data_point(self, data: Any) -> None:
        """Add a data point (keeps the newest 1000)."""
        if self.monitoring_active:
            self.data_buffer.append(data)
            if len(self.data_buffer) > 1000:
                self.data_buffer.pop(0)

    def get_latest_data(self, n_points: int = 10) -> List[Any]:
        """The latest ``n_points`` data points."""
        return self.data_buffer[-n_points:] if self.data_buffer else []


__all__ = [
    "RealTimeMonitor",
    "PerformanceMonitor",
    "PerformanceMetrics",
    "MetricsCollector",
    "SystemProfiler",
]
