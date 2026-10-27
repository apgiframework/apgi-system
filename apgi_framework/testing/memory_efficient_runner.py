"""Memory-Efficient Test Runner for APGI Framework.

Provides memory-efficient execution for large test suites with:
- Progress checkpointing and resume capability
- Memory usage monitoring and optimization
- Resource cleanup and garbage collection
- Memory profiling and leak detection
"""

import gc
import os
import pickle
import threading
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil

from ..logging.standardized_logging import get_logger
from ..optimization.performance_monitor import PerformanceAlert
from ..security.secure_pickle import safe_pickle_load
from .batch_runner import BatchExecutionSummary, BatchTestRunner, TestResult

logger = get_logger(__name__)


@dataclass
class MemoryCheckpoint:
    """Memory checkpoint data structure."""

    execution_id: str
    timestamp: datetime
    completed_tests: List[str]
    failed_tests: List[str]
    test_results: List[Dict[str, Any]]
    memory_stats: Dict[str, Any]
    execution_config: Dict[str, Any]
    next_test_index: int


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    current_memory_mb: float
    peak_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    gc_collections: Dict[int, int]
    tracemalloc_snapshot: Optional[Any] = None


class MemoryMonitor:
    """Real-time memory monitoring for test execution."""

    def __init__(self, memory_limit_mb: Optional[float] = None):
        self.memory_limit_mb = memory_limit_mb
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.memory_history: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        self._lock = threading.Lock()

        # Enable tracemalloc for detailed memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Memory monitoring stopped")

    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        memory_info = self.process.memory_info()
        virtual_memory = psutil.virtual_memory()

        # Get garbage collection stats
        gc_stats = {}
        for generation in range(3):
            gc_stats[generation] = gc.get_count()[generation]

        # Get tracemalloc snapshot if available
        snapshot = None
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()

        return MemoryStats(
            current_memory_mb=memory_info.rss / 1024 / 1024,
            peak_memory_mb=(
                getattr(memory_info, "peak_wset", 0) / 1024 / 1024
                if getattr(memory_info, "peak_wset", None) is not None
                else 0
            ),
            available_memory_mb=virtual_memory.available / 1024 / 1024,
            memory_percent=virtual_memory.percent,
            gc_collections=gc_stats,
            tracemalloc_snapshot=snapshot,
        )

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback for memory alerts."""
        self.alert_callbacks.append(callback)

    def _monitor_loop(self) -> None:
        """Memory monitoring loop."""
        while self.monitoring:
            try:
                stats = self.get_current_stats()

                with self._lock:
                    self.memory_history.append(
                        {
                            "timestamp": time.time(),
                            "memory_mb": stats.current_memory_mb,
                            "memory_percent": stats.memory_percent,
                        }
                    )

                    # Keep only last 1000 entries
                    if len(self.memory_history) > 1000:
                        self.memory_history = self.memory_history[-1000:]

                # Check for memory alerts
                self._check_memory_alerts(stats)

                time.sleep(1.0)  # Check every second

            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(1.0)

    def _check_memory_alerts(self, stats: MemoryStats) -> None:
        """Check for memory-related alerts."""
        alerts = []

        # High memory usage alert
        if stats.memory_percent > 90:
            alerts.append(
                PerformanceAlert(
                    timestamp=time.time(),
                    alert_type="memory",
                    severity="critical",
                    message=f"Critical memory usage: {stats.memory_percent:.1f}%",
                    value=stats.memory_percent,
                    threshold=90.0,
                )
            )
        elif stats.memory_percent > 80:
            alerts.append(
                PerformanceAlert(
                    timestamp=time.time(),
                    alert_type="memory",
                    severity="warning",
                    message=f"High memory usage: {stats.memory_percent:.1f}%",
                    value=stats.memory_percent,
                    threshold=80.0,
                )
            )

        # Memory limit exceeded
        if self.memory_limit_mb and stats.current_memory_mb > self.memory_limit_mb:
            alerts.append(
                PerformanceAlert(
                    timestamp=time.time(),
                    alert_type="memory",
                    severity="critical",
                    message=f"Memory limit exceeded: {stats.current_memory_mb:.1f}MB > {self.memory_limit_mb}MB",
                    value=stats.current_memory_mb,
                    threshold=self.memory_limit_mb,
                )
            )

        # Send alerts
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in memory alert callback: {e}")


class MemoryEfficientTestRunner(BatchTestRunner):
    """Memory-efficient test runner with checkpointing and resource management."""

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        memory_limit_mb: Optional[float] = None,
        checkpoint_interval: int = 10,
        enable_memory_profiling: bool = False,
    ):
        """Initialize memory-efficient test runner."""
        super().__init__()

        self.checkpoint_dir = Path(checkpoint_dir or "test_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.memory_limit_mb = memory_limit_mb
        self.checkpoint_interval = checkpoint_interval
        self.enable_memory_profiling = enable_memory_profiling

        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(memory_limit_mb)
        self.memory_monitor.add_alert_callback(self._handle_memory_alert)

        # Execution state
        self.current_execution_id: Optional[str] = None
        self.memory_alerts: List[PerformanceAlert] = []
        self.force_gc_after_test = True

        logger.info(f"Memory-efficient runner initialized with limit: {memory_limit_mb}MB")

    def run_batch_tests_with_checkpointing(
        self,
        test_selection: Optional[List[str]] = None,
        resume_from_checkpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> BatchExecutionSummary:
        """Run batch tests with memory-efficient checkpointing."""
        # Start memory monitoring
        self.memory_monitor.start_monitoring()  # type: ignore[no-untyped-call]

        try:
            if resume_from_checkpoint:
                return self._resume_from_checkpoint(resume_from_checkpoint, **kwargs)
            else:
                return self._run_with_checkpointing(test_selection, **kwargs)
        finally:
            self.memory_monitor.stop_monitoring()  # type: ignore[no-untyped-call]

    def _run_with_checkpointing(
        self, test_selection: Optional[List[str]], **kwargs: Any
    ) -> BatchExecutionSummary:
        """Run tests with periodic checkpointing."""
        start_time = datetime.now()
        self.current_execution_id = (
            f"memory_efficient_{start_time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        )

        # Discover tests
        if test_selection is None:
            test_files = self.discover_tests()
        else:
            test_files = test_selection

        if not test_files:
            logger.warning("No tests found to execute")
            return self._create_empty_summary(start_time)

        logger.info(f"Starting memory-efficient execution of {len(test_files)} tests")

        # Initialize execution state
        completed_tests = []
        failed_tests = []
        test_results = []

        # Process tests in batches
        for i, test_file in enumerate(test_files):
            if self._stop_execution:
                break

            logger.info(f"Running test {i + 1} / {len(test_files)}: {test_file}")

            # Run single test with memory monitoring
            result = self._run_test_with_memory_monitoring(test_file, kwargs.get("timeout", 300))
            test_results.append(result)

            if result.status == "passed":
                completed_tests.append(test_file)
            else:
                failed_tests.append(test_file)

            # Force garbage collection after each test
            if self.force_gc_after_test:
                self._force_garbage_collection()  # type: ignore[no-untyped-call]

            # Create checkpoint periodically
            if (i + 1) % self.checkpoint_interval == 0:
                if self.current_execution_id is None:
                    raise RuntimeError(
                        "Current execution ID must be set before creating checkpoint."
                    )
                self._create_checkpoint(
                    execution_id=self.current_execution_id,
                    completed_tests=completed_tests,
                    failed_tests=failed_tests,
                    test_results=[asdict(r) for r in test_results],
                    next_test_index=i + 1,
                    execution_config=kwargs,
                )

            # Check memory limits
            current_stats = self.memory_monitor.get_current_stats()
            if self.memory_limit_mb and current_stats.current_memory_mb > self.memory_limit_mb:
                logger.warning(f"Memory limit exceeded: {current_stats.current_memory_mb:.1f}MB")
                # Force more aggressive cleanup
                self._aggressive_memory_cleanup()  # type: ignore[no-untyped-call]

        # Create final summary
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        summary = BatchExecutionSummary(
            total_tests=len(test_results),
            passed=sum(1 for r in test_results if r.status == "passed"),
            failed=sum(1 for r in test_results if r.status == "failed"),
            skipped=sum(1 for r in test_results if r.status == "skipped"),
            errors=sum(1 for r in test_results if r.status == "error"),
            total_duration=total_duration,
            start_time=start_time,
            end_time=end_time,
            test_results=test_results,
            execution_metadata={
                "execution_id": self.current_execution_id,
                "memory_efficient": True,
                "memory_limit_mb": self.memory_limit_mb,
                "checkpoint_interval": self.checkpoint_interval,
                "memory_alerts": len(self.memory_alerts),
                **kwargs,
            },
        )

        # Clean up checkpoint files on successful completion
        self._cleanup_checkpoints(self.current_execution_id)

        logger.info(
            f"Memory-efficient execution completed: {summary.passed} passed, {summary.failed} failed"
        )
        return summary

    def _run_test_with_memory_monitoring(self, test_file: str, timeout: int) -> TestResult:
        """Run a single test with memory monitoring."""
        # Get memory stats before test
        pre_stats = self.memory_monitor.get_current_stats()

        # Run the test
        result = self._run_single_test(test_file, timeout)

        # Get memory stats after test
        post_stats = self.memory_monitor.get_current_stats()

        # Log memory usage for this test
        memory_delta = post_stats.current_memory_mb - pre_stats.current_memory_mb
        if memory_delta > 50:  # Log if test used more than 50MB
            logger.warning(f"Test {test_file} used {memory_delta:.1f}MB memory")

        return result

    def _create_checkpoint(
        self,
        execution_id: str,
        completed_tests: List[str],
        failed_tests: List[str],
        test_results: List[Dict[str, Any]],
        next_test_index: int,
        execution_config: Dict[str, Any],
    ) -> None:
        """Create a checkpoint file."""
        checkpoint = MemoryCheckpoint(
            execution_id=execution_id,
            timestamp=datetime.now(),
            completed_tests=completed_tests.copy(),
            failed_tests=failed_tests.copy(),
            test_results=(
                [asdict(r) for r in test_results if isinstance(r, TestResult)]
                if test_results
                else []
            ),
            memory_stats=asdict(self.memory_monitor.get_current_stats()),
            execution_config=execution_config,
            next_test_index=next_test_index,
        )

        checkpoint_file = self.checkpoint_dir / f"{execution_id}_checkpoint_{next_test_index}.pkl"

        try:
            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint, f)

            logger.info(f"Checkpoint created: {checkpoint_file}")

            # Clean up old checkpoints for this execution
            self._cleanup_old_checkpoints(execution_id, keep_latest=3)

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")

    def _resume_from_checkpoint(self, checkpoint_file: str, **kwargs: Any) -> BatchExecutionSummary:
        """Resume execution from a checkpoint."""
        checkpoint_path = Path(checkpoint_file)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        try:
            checkpoint = safe_pickle_load(checkpoint_path)

            logger.info(f"Resuming execution from checkpoint: {checkpoint_file}")
            logger.info(f"Completed tests: {len(checkpoint.completed_tests)}")
            logger.info(f"Next test index: {checkpoint.next_test_index}")

            # Restore execution state
            self.current_execution_id = checkpoint.execution_id

            # Get remaining tests
            original_config = checkpoint.execution_config
            test_selection = original_config.get("test_selection")

            if test_selection is None:
                all_tests = self.discover_tests()
            else:
                all_tests = test_selection

            remaining_tests = all_tests[checkpoint.next_test_index :]

            if not remaining_tests:
                logger.info("No remaining tests to execute")
                # Return summary from checkpoint
                return self._create_summary_from_checkpoint(checkpoint)

            # Continue execution
            logger.info(f"Continuing with {len(remaining_tests)} remaining tests")

            # Convert test results back to objects
            existing_results = [TestResult(**r) for r in checkpoint.test_results]

            # Run remaining tests
            for i, test_file in enumerate(remaining_tests):
                if self._stop_execution:
                    break

                current_index = checkpoint.next_test_index + i
                logger.info(f"Running test {current_index + 1}/{len(all_tests)}: {test_file}")

                result = self._run_test_with_memory_monitoring(
                    test_file, kwargs.get("timeout", 300)
                )
                existing_results.append(result)

                if result.status == "passed":
                    checkpoint.completed_tests.append(test_file)
                else:
                    checkpoint.failed_tests.append(test_file)

                # Force garbage collection
                if self.force_gc_after_test:
                    self._force_garbage_collection()  # type: ignore[no-untyped-call]

                # Create checkpoint periodically
                if (i + 1) % self.checkpoint_interval == 0:
                    self._create_checkpoint(
                        execution_id=self.current_execution_id or "unknown",
                        completed_tests=checkpoint.completed_tests,
                        failed_tests=checkpoint.failed_tests,
                        test_results=[asdict(r) for r in existing_results],
                        next_test_index=current_index + 1,
                        execution_config=original_config,
                    )

            # Create final summary
            end_time = datetime.now()
            total_duration = (end_time - checkpoint.timestamp).total_seconds()

            summary = BatchExecutionSummary(
                total_tests=len(existing_results),
                passed=sum(1 for r in existing_results if r.status == "passed"),
                failed=sum(1 for r in existing_results if r.status == "failed"),
                skipped=sum(1 for r in existing_results if r.status == "skipped"),
                errors=sum(1 for r in existing_results if r.status == "error"),
                total_duration=total_duration,
                start_time=checkpoint.timestamp,
                end_time=end_time,
                test_results=existing_results,
                execution_metadata={
                    "execution_id": self.current_execution_id,
                    "resumed_from_checkpoint": True,
                    "checkpoint_file": checkpoint_file,
                    "memory_efficient": True,
                    **original_config,
                },
            )

            # Clean up checkpoint files
            self._cleanup_checkpoints(self.current_execution_id or "unknown")

            return summary

        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            raise

    def _force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""
        collected = gc.collect()
        if collected > 0:
            logger.debug(f"Garbage collection freed {collected} objects")

    def _aggressive_memory_cleanup(self) -> None:
        """Perform aggressive memory cleanup."""
        logger.info("Performing aggressive memory cleanup")

        # Force garbage collection multiple times
        for generation in range(3):
            collected = gc.collect(generation)
            if collected > 0:
                logger.debug(f"GC generation {generation}: freed {collected} objects")

        # Clear any module-level caches if possible
        try:
            import sys

            for module_name, module in list(sys.modules.items()):
                if hasattr(module, "__dict__"):
                    # Clear any obvious cache attributes
                    for attr_name in list(module.__dict__.keys()):
                        if "cache" in attr_name.lower():
                            try:
                                cache_obj = getattr(module, attr_name)
                                if hasattr(cache_obj, "clear"):
                                    cache_obj.clear()
                                    logger.debug(f"Cleared cache: {module_name}.{attr_name}")
                            except Exception:
                                pass
        except Exception as e:
            logger.debug(f"Error during cache cleanup: {e}")

    def _handle_memory_alert(self, alert: PerformanceAlert) -> None:
        """Handle memory alerts."""
        self.memory_alerts.append(alert)
        logger.warning(f"Memory alert: {alert.message}")

        if alert.severity == "critical":
            # Perform immediate cleanup
            self._aggressive_memory_cleanup()  # type: ignore[no-untyped-call]

    def _cleanup_old_checkpoints(self, execution_id: str, keep_latest: int = 3) -> None:
        """Clean up old checkpoint files."""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob(f"{execution_id}_checkpoint_*.pkl"))
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remove old checkpoints
            for old_checkpoint in checkpoint_files[keep_latest:]:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")

        except Exception as e:
            logger.error(f"Error cleaning up old checkpoints: {e}")

    def _cleanup_checkpoints(self, execution_id: str) -> None:
        """Clean up all checkpoint files for an execution."""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob(f"{execution_id}_checkpoint_*.pkl"))
            for checkpoint_file in checkpoint_files:
                checkpoint_file.unlink()
                logger.debug(f"Removed checkpoint: {checkpoint_file}")
        except Exception as e:
            logger.error(f"Error cleaning up checkpoints: {e}")

    def _create_empty_summary(self, start_time: datetime) -> BatchExecutionSummary:
        """Create empty summary for no tests case."""
        return BatchExecutionSummary(
            total_tests=0,
            passed=0,
            failed=0,
            skipped=0,
            errors=0,
            total_duration=0.0,
            start_time=start_time,
            end_time=datetime.now(),
            test_results=[],
            execution_metadata={},
        )

    def _create_summary_from_checkpoint(
        self, checkpoint: MemoryCheckpoint
    ) -> BatchExecutionSummary:
        """Create summary from checkpoint data."""
        test_results = checkpoint.test_results

        return BatchExecutionSummary(
            total_tests=len(test_results),
            passed=sum(
                1 for r in test_results if isinstance(r, dict) and r.get("status") == "passed"
            ),
            failed=sum(
                1 for r in test_results if isinstance(r, dict) and r.get("status") == "failed"
            ),
            skipped=sum(
                1 for r in test_results if isinstance(r, dict) and r.get("status") == "skipped"
            ),
            errors=sum(
                1 for r in test_results if isinstance(r, dict) and r.get("status") == "error"
            ),
            total_duration=0.0,  # Duration not available from checkpoint
            start_time=checkpoint.timestamp,
            end_time=checkpoint.timestamp,  # Use same timestamp for end
            test_results=[],  # Cannot restore TestResult objects from dicts
            execution_metadata={"checkpoint_loaded": True},
        )

    def get_memory_report(self) -> Dict[str, Any]:
        """Get detailed memory usage report."""
        current_stats = self.memory_monitor.get_current_stats()

        report = {
            "current_memory_mb": current_stats.current_memory_mb,
            "peak_memory_mb": current_stats.peak_memory_mb,
            "available_memory_mb": current_stats.available_memory_mb,
            "memory_percent": current_stats.memory_percent,
            "gc_collections": current_stats.gc_collections,
            "memory_alerts": len(self.memory_alerts),
            "memory_limit_mb": self.memory_limit_mb,
            "tracemalloc_enabled": tracemalloc.is_tracing(),
        }

        # Add tracemalloc top stats if available
        if current_stats.tracemalloc_snapshot:
            top_stats = current_stats.tracemalloc_snapshot.statistics("lineno")[:10]
            report["top_memory_allocations"] = [
                {
                    "filename": stat.traceback.format()[0],
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count,
                }
                for stat in top_stats
            ]

        return report

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoint files."""
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob("*_checkpoint_*.pkl"):
            try:
                stat = checkpoint_file.stat()
                checkpoints.append(
                    {
                        "filename": checkpoint_file.name,
                        "path": str(checkpoint_file),
                        "size_mb": stat.st_size / 1024 / 1024,
                        "modified": datetime.fromtimestamp(stat.st_mtime),
                        "execution_id": checkpoint_file.name.split("_checkpoint_")[0],
                    }
                )
            except Exception as e:
                logger.error(f"Error reading checkpoint {checkpoint_file}: {e}")

        return sorted(checkpoints, key=lambda x: x["modified"], reverse=True)


@contextmanager
def memory_profiling_context(operation_name: str) -> Any:
    """Context manager for memory profiling."""
    if not tracemalloc.is_tracing():
        tracemalloc.start()

    # Take initial snapshot
    snapshot1 = tracemalloc.take_snapshot()
    start_time = time.time()

    try:
        yield
    finally:
        # Take final snapshot
        snapshot2 = tracemalloc.take_snapshot()
        end_time = time.time()

        # Compare snapshots
        top_stats = snapshot2.compare_to(snapshot1, "lineno")

        logger.info(f"Memory profiling for {operation_name}:")
        logger.info(f"Duration: {end_time - start_time:.2f}s")

        for index, stat in enumerate(top_stats[:5], 1):
            logger.info(f"#{index}: {stat}")


# Convenience functions
def run_memory_efficient_tests(
    test_selection: Optional[List[str]] = None,
    memory_limit_mb: Optional[float] = None,
    checkpoint_interval: int = 10,
    resume_from: Optional[str] = None,
) -> BatchExecutionSummary:
    """Run tests with memory-efficient execution."""
    runner = MemoryEfficientTestRunner(
        memory_limit_mb=memory_limit_mb, checkpoint_interval=checkpoint_interval
    )

    return runner.run_batch_tests_with_checkpointing(
        test_selection=test_selection, resume_from_checkpoint=resume_from
    )


def get_memory_usage_report() -> Dict[str, Any]:
    """Get current memory usage report."""
    process = psutil.Process()
    memory_info = process.memory_info()
    virtual_memory = psutil.virtual_memory()

    return {
        "current_memory_mb": memory_info.rss / 1024 / 1024,
        "available_memory_mb": virtual_memory.available / 1024 / 1024,
        "memory_percent": virtual_memory.percent,
        "gc_counts": gc.get_count(),
        "tracemalloc_enabled": tracemalloc.is_tracing(),
    }
