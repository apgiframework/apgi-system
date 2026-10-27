"""Performance Profiler for APGI Framework.

Provides comprehensive performance monitoring, profiling, and optimization
tools to identify and resolve computational bottlenecks.
"""

import cProfile
import functools
import io
import json
import pstats
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
import psutil

from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    execution_time: float
    memory_usage: float  # MB
    cpu_usage: float  # percentage
    peak_memory: float  # MB
    function_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "peak_memory": self.peak_memory,
            "function_calls": self.function_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


@dataclass
class ProfileResult:
    """Container for profiling results."""

    function_name: str
    metrics: PerformanceMetrics
    timestamp: float
    duration: float
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PerformanceProfiler:
    """Main performance profiler for APGI Framework."""

    def __init__(self, enable_detailed_profiling: bool = False) -> None:
        self.enable_detailed_profiling = enable_detailed_profiling
        self.results: Dict[str, List[ProfileResult]] = defaultdict(list)
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        self.memory_monitor = MemoryMonitor()
        self.cpu_monitor = CPUMonitor()
        self._lock = threading.Lock()

    @contextmanager
    def profile(self, operation_name: str, detailed: Optional[bool] = None) -> Any:
        """Context manager for profiling operations."""
        detailed = detailed if detailed is not None else self.enable_detailed_profiling

        # Start monitoring
        start_time = time.perf_counter()
        start_memory = self.memory_monitor.get_current_usage()

        profile_data = {
            "start_time": start_time,
            "start_memory": start_memory,
            "profiler": None,
        }

        if detailed:
            profile_data["profiler"] = cProfile.Profile()  # type: ignore
            profile_data["profiler"].enable()  # type: ignore

        self.cpu_monitor.start_monitoring()

        try:
            with self._lock:
                self.active_profiles[operation_name] = profile_data

            yield

        finally:
            # Stop monitoring
            end_time = time.perf_counter()
            end_memory = self.memory_monitor.get_current_usage()
            peak_memory = self.memory_monitor.get_peak_usage()
            avg_cpu = self.cpu_monitor.stop_monitoring()

            if detailed and profile_data["profiler"]:
                profile_data["profiler"].disable()  # type: ignore

            # Calculate metrics
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory

            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=avg_cpu,
                peak_memory=peak_memory,
            )

            # Analyze bottlenecks
            bottlenecks = self._analyze_bottlenecks(
                metrics, cast(Optional[cProfile.Profile], profile_data.get("profiler"))
            )
            recommendations = self._generate_recommendations(metrics, bottlenecks)

            # Store result
            result = ProfileResult(
                function_name=operation_name,
                metrics=metrics,
                timestamp=start_time,
                duration=execution_time,
                bottlenecks=bottlenecks,
                recommendations=recommendations,
            )

            with self._lock:
                self.results[operation_name].append(result)
                if operation_name in self.active_profiles:
                    del self.active_profiles[operation_name]

            logger.info(f"Profiled {operation_name}: {execution_time:.3f}s, {memory_usage:.1f}MB")

    def profile_function(
        self,
        func: Optional[Callable[..., Any]] = None,
        *,
        name: Optional[str] = None,
        detailed: Optional[bool] = None,
    ) -> Callable[..., Any]:
        """Decorator for profiling functions."""

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            operation_name = name or f.__name__

            @functools.wraps(f)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.profile(operation_name, detailed):
                    return f(*args, **kwargs)

            return wrapper

        if func is None:
            return decorator
        else:
            return decorator(func)

    def _analyze_bottlenecks(
        self, metrics: PerformanceMetrics, profiler: Optional[cProfile.Profile]
    ) -> List[str]:
        """Analyze performance metrics to identify bottlenecks."""
        bottlenecks = []

        # Time-based bottlenecks
        if metrics.execution_time > 10.0:
            bottlenecks.append("long_execution_time")
        elif metrics.execution_time > 2.0:
            bottlenecks.append("moderate_execution_time")

        # Memory-based bottlenecks
        if metrics.memory_usage > 1000:  # 1GB
            bottlenecks.append("high_memory_usage")
        elif metrics.memory_usage > 500:  # 500MB
            bottlenecks.append("moderate_memory_usage")

        # CPU-based bottlenecks
        if metrics.cpu_usage > 90:
            bottlenecks.append("high_cpu_usage")
        elif metrics.cpu_usage > 70:
            bottlenecks.append("moderate_cpu_usage")

        # Detailed profiler analysis
        if profiler:
            # Remove unused stats variable and use ps directly
            ps = pstats.Stats(profiler)
            ps.sort_stats("cumulative")

            # Capture stats output
            s = io.StringIO()
            ps.print_stats(10)
            stats_output = s.getvalue()

            # Analyze for common bottlenecks
            if "numpy" in stats_output and metrics.execution_time > 1.0:
                bottlenecks.append("numpy_operations")

            if "scipy" in stats_output and metrics.execution_time > 1.0:
                bottlenecks.append("scipy_operations")

            if "pandas" in stats_output and metrics.execution_time > 1.0:
                bottlenecks.append("pandas_operations")

        return bottlenecks

    def _generate_recommendations(
        self, metrics: PerformanceMetrics, bottlenecks: List[str]
    ) -> List[str]:
        """Generate optimization recommendations based on bottlenecks."""
        recommendations = []

        if "long_execution_time" in bottlenecks:
            recommendations.extend(
                [
                    "Consider parallelization with multiprocessing or threading",
                    "Profile individual functions to identify specific slow operations",
                    "Consider algorithmic optimizations or caching",
                ]
            )

        if "high_memory_usage" in bottlenecks:
            recommendations.extend(
                [
                    "Process data in smaller chunks or batches",
                    "Use memory-efficient data structures (e.g., numpy arrays)",
                    "Consider data compression or streaming processing",
                ]
            )

        if "high_cpu_usage" in bottlenecks:
            recommendations.extend(
                [
                    "Optimize mathematical operations with vectorization",
                    "Consider using compiled extensions (Cython, Numba)",
                    "Reduce unnecessary computations and loops",
                ]
            )

        if "numpy_operations" in bottlenecks:
            recommendations.extend(
                [
                    "Use vectorized operations instead of loops",
                    "Consider using numba.jit for numerical computations",
                    "Optimize array operations and memory layout",
                ]
            )

        if "pandas_operations" in bottlenecks:
            recommendations.extend(
                [
                    "Use vectorized pandas operations",
                    "Consider using categorical data types for strings",
                    "Optimize DataFrame operations and indexing",
                ]
            )

        return recommendations

    def get_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for operations."""
        if operation_name:
            if operation_name not in self.results:
                return {}

            results = self.results[operation_name]
            times = [r.metrics.execution_time for r in results]
            memories = [r.metrics.memory_usage for r in results]

            return {
                "operation": operation_name,
                "total_runs": len(results),
                "avg_time": np.mean(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "std_time": np.std(times),
                "avg_memory": np.mean(memories),
                "max_memory": np.max(memories),
                "common_bottlenecks": self._get_common_bottlenecks(results),
                "recommendations": self._get_common_recommendations(results),
            }
        else:
            # Summary for all operations
            summary = {}
            for op_name in self.results:
                summary[op_name] = self.get_summary(op_name)
            return summary

    def _get_common_bottlenecks(self, results: List[ProfileResult]) -> List[str]:
        """Get most common bottlenecks across results."""
        bottleneck_counts: Dict[str, int] = defaultdict(int)
        for result in results:
            for bottleneck in result.bottlenecks:
                bottleneck_counts[bottleneck] += 1

        # Return bottlenecks that appear in >50% of runs
        threshold = len(results) * 0.5
        return [b for b, count in bottleneck_counts.items() if count >= threshold]

    def _get_common_recommendations(self, results: List[ProfileResult]) -> List[str]:
        """Get most common recommendations across results."""
        rec_counts: Dict[str, int] = defaultdict(int)
        for result in results:
            for rec in result.recommendations:
                rec_counts[rec] += 1

        # Return top 3 most common recommendations
        sorted_recs = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)
        return [rec for rec, _ in sorted_recs[:3]]

    def export_results(self, filepath: Union[str, Path]) -> None:
        """Export profiling results to JSON file."""
        filepath = Path(filepath)

        export_data: Dict[str, Any] = {"timestamp": time.time(), "results": {}}

        for op_name, results in self.results.items():
            if isinstance(results, list):
                export_data["results"][op_name] = [
                    {
                        "timestamp": r.timestamp,
                        "duration": r.duration,
                        "metrics": r.metrics.to_dict(),
                        "bottlenecks": r.bottlenecks,
                        "recommendations": r.recommendations,
                    }
                    for r in results
                ]

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported profiling results to {filepath}")

    def clear_results(self, operation_name: Optional[str] = None) -> None:
        """Clear profiling results."""
        if operation_name:
            if operation_name in self.results:
                del self.results[operation_name]
        else:
            self.results.clear()


class MemoryMonitor:
    """Monitors memory usage during operations."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak_usage: float = 0.0
        self.baseline_usage = self.get_current_usage()

    def get_current_usage(self) -> float:
        """Get current memory usage in MB."""
        return float(self.process.memory_info().rss / 1024 / 1024)

    def get_peak_usage(self) -> float:
        """Get peak memory usage since last reset."""
        current = self.get_current_usage()
        self.peak_usage = max(self.peak_usage, current)
        return float(self.peak_usage)

    def reset_peak(self) -> None:
        """Reset peak memory tracking."""
        self.peak_usage = self.get_current_usage()


class CPUMonitor:
    """Monitors CPU usage during operations."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.monitoring = False
        self.cpu_samples: deque = deque(maxlen=100)
        self.monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self) -> None:
        """Start CPU monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self.cpu_samples.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> float:
        """Stop CPU monitoring and return average usage."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        if self.cpu_samples:
            return float(sum(self.cpu_samples) / len(self.cpu_samples))
        return 0.0

    def _monitor_loop(self) -> None:
        """Internal monitoring loop."""
        while self.monitoring:
            try:
                cpu_percent = self.process.cpu_percent()
                self.cpu_samples.append(cpu_percent)
                time.sleep(0.1)  # Sample every 100ms
            except Exception:
                break


class OptimizationSuggester:
    """Suggests optimizations based on profiling results."""

    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler

    def analyze_operation(self, operation_name: str) -> Dict[str, Any]:
        """Analyze a specific operation and suggest optimizations."""
        if operation_name not in self.profiler.results:
            return {"error": f"No profiling data for operation: {operation_name}"}

        results = self.profiler.results[operation_name]
        latest_result = results[-1]

        analysis = {
            "operation": operation_name,
            "performance_grade": self._calculate_performance_grade(latest_result),
            "bottlenecks": latest_result.bottlenecks,
            "recommendations": latest_result.recommendations,
            "optimization_priority": self._calculate_optimization_priority(latest_result),
            "specific_suggestions": self._get_specific_suggestions(latest_result),
        }

        return analysis

    def _calculate_performance_grade(self, result: ProfileResult) -> str:
        """Calculate performance grade (A-F) based on metrics."""
        score = 100

        # Deduct points for execution time
        if result.metrics.execution_time > 10:
            score -= 40
        elif result.metrics.execution_time > 5:
            score -= 25
        elif result.metrics.execution_time > 2:
            score -= 15
        elif result.metrics.execution_time > 1:
            score -= 10

        # Deduct points for memory usage
        if result.metrics.memory_usage > 1000:
            score -= 30
        elif result.metrics.memory_usage > 500:
            score -= 20
        elif result.metrics.memory_usage > 200:
            score -= 10

        # Deduct points for CPU usage
        if result.metrics.cpu_usage > 90:
            score -= 20
        elif result.metrics.cpu_usage > 70:
            score -= 10

        # Convert to letter grade
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _calculate_optimization_priority(self, result: ProfileResult) -> str:
        """Calculate optimization priority (High/Medium/Low)."""
        if (
            result.metrics.execution_time > 5
            or result.metrics.memory_usage > 500
            or len(result.bottlenecks) >= 3
        ):
            return "High"
        elif (
            result.metrics.execution_time > 2
            or result.metrics.memory_usage > 200
            or len(result.bottlenecks) >= 2
        ):
            return "Medium"
        else:
            return "Low"

    def _get_specific_suggestions(self, result: ProfileResult) -> List[str]:
        """Get specific optimization suggestions."""
        suggestions = []

        # Time-based suggestions
        if result.metrics.execution_time > 5:
            suggestions.append(
                "Consider breaking this operation into smaller, parallelizable chunks"
            )
            suggestions.append(
                "Profile individual sub-operations to identify the slowest components"
            )

        # Memory-based suggestions
        if result.metrics.memory_usage > 500:
            suggestions.append("Implement data streaming or batch processing")
            suggestions.append("Use memory-mapped files for large datasets")
            suggestions.append("Consider using sparse data structures if applicable")

        # Bottleneck-specific suggestions
        if "numpy_operations" in result.bottlenecks:
            suggestions.append("Use np.einsum for complex array operations")
            suggestions.append("Consider using numba.jit for numerical loops")
            suggestions.append("Optimize array memory layout (C vs Fortran order)")

        if "pandas_operations" in result.bottlenecks:
            suggestions.append("Use pd.eval() for complex expressions")
            suggestions.append("Consider using Dask for out-of-core operations")
            suggestions.append("Optimize DataFrame dtypes (use categories for strings)")

        return suggestions


# Global profiler instance
_global_profiler = None


def get_profiler(enable_detailed: bool = False) -> PerformanceProfiler:
    """Get the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(enable_detailed)
    return _global_profiler


def profile_operation(name: str, detailed: bool = False) -> Callable[..., Any]:
    """Decorator for profiling operations."""
    return get_profiler().profile_function(name=name, detailed=detailed)


def profile_context(name: str, detailed: bool = False) -> Any:
    """Context manager for profiling code blocks."""
    return get_profiler().profile(name, detailed)
