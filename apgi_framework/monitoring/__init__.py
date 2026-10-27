"""Monitoring module for APGI Framework.

This module provides real-time monitoring capabilities for:
- EEG signals
- Pupillometry
- Cardiac signals
- Experimental progress
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


# Mock classes for testing
class SystemProfiler:
    """Mock system profiler for testing purposes."""

    def __init__(self) -> None:
        self.profiles: Dict[str, List[Dict[str, Any]]] = {}
        self.profiling_active: bool = False

    def start_profiling(self) -> Dict[str, str]:
        """Start system profiling."""
        self.profiling_active = True
        return {"status": "started", "timestamp": "2024-01-01T00:00:00Z"}

    def stop_profiling(self) -> Dict[str, str]:
        """Stop system profiling."""
        self.profiling_active = False
        return {"status": "stopped", "timestamp": "2024-01-01T00:00:00Z"}

    def record_profile(self, profile_name: str, profile_data: Dict[str, Any]) -> None:
        """Record a profile data point."""
        if self.profiling_active:
            if profile_name not in self.profiles:
                self.profiles[profile_name] = []
            self.profiles[profile_name].append(profile_data)

    def get_profiles(self, profile_name: Optional[str] = None) -> Any:
        """Get profile data."""
        if profile_name:
            return self.profiles.get(profile_name, [])
        return self.profiles


class MetricsCollector:
    """Mock metrics collector for testing purposes."""

    def __init__(self) -> None:
        self.collected_metrics: Dict[str, List[float]] = {}
        self.collection_active: bool = False

    def start_collection(self) -> Dict[str, str]:
        """Start metrics collection."""
        self.collection_active = True
        return {"status": "started", "timestamp": "2024-01-01T00:00:00Z"}

    def stop_collection(self) -> Dict[str, str]:
        """Stop metrics collection."""
        self.collection_active = False
        return {"status": "stopped", "timestamp": "2024-01-01T00:00:00Z"}

    def collect_metric(self, metric_name: str, value: float) -> None:
        """Collect a metric value."""
        if self.collection_active:
            if metric_name not in self.collected_metrics:
                self.collected_metrics[metric_name] = []
            self.collected_metrics[metric_name].append(value)

    def get_collected_metrics(self, metric_name: Optional[str] = None) -> Any:
        """Get collected metrics."""
        if metric_name:
            return self.collected_metrics.get(metric_name, [])
        return self.collected_metrics


@dataclass
class PerformanceMetrics:
    """Mock performance metrics for testing purposes."""

    step_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    ignition_rate_hz: float = 0.0
    free_energy_mean: float = 0.0
    precision_mean: float = 0.0
    timestamp: float = 0.0


class RealTimeMonitor:
    """Mock real-time monitor for testing purposes."""

    def __init__(self) -> None:
        self.monitoring_active: bool = False
        self.data_buffer: List[Any] = []

    def start_monitoring(self) -> Dict[str, str]:
        """Start real-time monitoring."""
        self.monitoring_active = True
        return {"status": "started", "timestamp": "2024-01-01T00:00:00Z"}

    def stop_monitoring(self) -> Dict[str, str]:
        """Stop real-time monitoring."""
        self.monitoring_active = False
        return {"status": "stopped", "timestamp": "2024-01-01T00:00:00Z"}

    def add_data_point(self, data: Any) -> None:
        """Add a data point to the monitoring buffer."""
        if self.monitoring_active:
            self.data_buffer.append(data)
            if len(self.data_buffer) > 1000:  # Keep buffer size manageable
                self.data_buffer.pop(0)

    def get_latest_data(self, n_points: int = 10) -> List[Any]:
        """Get the latest n data points."""
        return self.data_buffer[-n_points:] if self.data_buffer else []


class PerformanceMonitor:
    """Mock performance monitor for testing purposes."""

    def __init__(self, config: Dict[str, Any]) -> None:
        monitoring_config = config.get("monitoring", {})
        self.enabled: bool = monitoring_config.get("enabled", True)
        self.max_history_size: int = monitoring_config.get("max_history_size", 100)
        self.history: List[PerformanceMetrics] = []
        self.ignition_times: List[float] = []
        self.step_start_time: Optional[float] = None
        self.last_step_time_ms: float = 0.0
        self.performance_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.monitoring_active: bool = False

    def start_step(self) -> None:
        """Start timing a step."""
        import time

        self.step_start_time = time.time()

    def end_step(
        self,
        system_time_ms: float,
        ignition_occurred: bool,
        free_energy: float,
        precision: float,
    ) -> Optional[PerformanceMetrics]:
        """End timing a step and record metrics."""
        import time

        import psutil

        if not self.enabled or self.step_start_time is None:
            return None

        step_time_ms = (time.time() - self.step_start_time) * 1000
        self.last_step_time_ms = step_time_ms

        memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024

        # Calculate ignition rate (ignitions per second in last second)
        current_time = time.time()
        self.ignition_times = [t for t in self.ignition_times if current_time - t < 1.0]
        if ignition_occurred:
            self.ignition_times.append(current_time)
        ignition_rate_hz = len(self.ignition_times)

        metrics = PerformanceMetrics(
            step_time_ms=step_time_ms,
            memory_usage_mb=memory_usage_mb,
            ignition_rate_hz=ignition_rate_hz,
            free_energy_mean=free_energy,
            precision_mean=precision,
            timestamp=system_time_ms,
        )

        self.history.append(metrics)
        if len(self.history) > self.max_history_size:
            self.history.pop(0)

        self.step_start_time = None
        return metrics

    def get_statistics(self) -> Dict[str, float]:
        """Get statistics from recorded metrics."""
        if not self.history:
            return {
                "total_samples": 0,
                "mean_step_time_ms": 0.0,
                "max_step_time_ms": 0.0,
                "min_step_time_ms": 0.0,
                "mean_memory_mb": 0.0,
                "std_step_time_ms": 0.0,
            }

        step_times = [m.step_time_ms for m in self.history]
        memory_usage = [m.memory_usage_mb for m in self.history]

        return {
            "total_samples": len(self.history),
            "mean_step_time_ms": float(np.mean(step_times)),
            "max_step_time_ms": float(np.max(step_times)),
            "min_step_time_ms": float(np.min(step_times)),
            "mean_memory_mb": float(np.mean(memory_usage)),
            "std_step_time_ms": float(np.std(step_times)),
        }

    def get_recent_metrics(self, n: int) -> List[PerformanceMetrics]:
        """Get the n most recent metrics."""
        return self.history[-n:] if self.history else []

    def reset(self) -> None:
        """Reset the monitor."""
        self.history.clear()
        self.ignition_times.clear()
        self.step_start_time = None
        self.last_step_time_ms = 0.0

    def log_performance(self, verbose: bool = False) -> None:
        """Log performance statistics."""
        stats = self.get_statistics()
        if stats["total_samples"] == 0:
            return

        print("Performance Statistics")
        print(f"Total samples: {stats['total_samples']}")
        print(
            f"Step time: mean={stats['mean_step_time_ms']:.2f}ms, "
            f"max={stats['max_step_time_ms']:.2f}ms, "
            f"min={stats['min_step_time_ms']:.2f}ms"
        )
        print(f"Memory usage: {stats['mean_memory_mb']:.2f}MB")

        if verbose:
            print("\nRecent metrics:")
            for m in self.get_recent_metrics(5):
                print(
                    f"  t={m.timestamp:.1f}: "
                    f"step={m.step_time_ms:.2f}ms, "
                    f"mem={m.memory_usage_mb:.2f}MB, "
                    f"ignition_rate={m.ignition_rate_hz:.2f}Hz"
                )

    def start_monitoring(self) -> Dict[str, str]:
        """Start performance monitoring."""
        self.monitoring_active = True
        return {"status": "started", "timestamp": "2024-01-01T00:00:00Z"}

    def stop_monitoring(self) -> Dict[str, str]:
        """Stop performance monitoring."""
        self.monitoring_active = False
        return {"status": "stopped", "timestamp": "2024-01-01T00:00:00Z"}

    def record_metric(self, metric_name: str, value: float) -> None:
        """Record a performance metric."""
        if self.monitoring_active:
            if metric_name not in self.performance_metrics:
                self.performance_metrics[metric_name] = []
            self.performance_metrics[metric_name].append(
                {
                    "value": value,
                    "timestamp": "2024-01-01T00:00:00Z",
                }
            )

    def get_metrics(self, metric_name: Optional[str] = None) -> Any:
        """Get performance metrics."""
        if metric_name:
            return self.performance_metrics.get(metric_name, [])
        return self.performance_metrics


__all__ = [
    "RealTimeMonitor",
    "PerformanceMonitor",
    "PerformanceMetrics",
    "MetricsCollector",
    "SystemProfiler",
]
