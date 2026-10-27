"""Process pool management for CPU-bound simulations.

Provides ProcessPoolExecutor wrapper with proper initialization,
task distribution, and result handling for neural simulations.
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class SimulationTask:
    """Represents a single simulation task for parallel execution."""

    task_id: str
    func: Callable[..., Any]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    priority: int = 0
    timeout: Optional[float] = None


@dataclass
class TaskResult(Generic[T]):
    """Result of a parallel task execution."""

    task_id: str
    success: bool
    result: Optional[T]
    error: Optional[str]
    execution_time: float


class ProcessPoolManager:
    """Manages a process pool for CPU-intensive APGI simulations.

    Features:
    - Dynamic worker pool sizing based on CPU count
    - Task prioritization and timeout handling
    - Progress tracking and graceful shutdown
    - Memory-efficient chunking for large batches
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        initializer: Optional[Callable[..., Any]] = None,
        initargs: Tuple[Any, ...] = (),
    ):
        """Initialize the process pool manager.

        Args:
            max_workers: Maximum number of worker processes.
                        Defaults to CPU count.
            initializer: Optional callable to initialize each worker
            initargs: Arguments for the initializer
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.initializer = initializer
        self.initargs = initargs
        self._executor: Optional[ProcessPoolExecutor] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the process pool."""
        if self._initialized:
            return

        mp.set_start_method("spawn", force=True)

        self._executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=self.initializer,
            initargs=self.initargs,
        )
        self._initialized = True
        logger.info(f"Process pool initialized with {self.max_workers} workers")

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the process pool gracefully."""
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None
            self._initialized = False
            logger.info("Process pool shutdown complete")

    def submit(self, task: SimulationTask) -> Any:
        """Submit a single task to the pool."""
        if not self._initialized or not self._executor:
            self.initialize()
        assert self._executor is not None, "Executor should be initialized"

        future = self._executor.submit(
            _execute_task_wrapper,
            task.func,
            task.args,
            task.kwargs,
        )
        return future

    def map(
        self,
        func: Callable[[Any], T],
        items: List[Any],
        chunksize: Optional[int] = None,
    ) -> List[T]:
        """Map a function over a list of items in parallel.

        Args:
            func: Function to apply to each item
            items: List of items to process
            chunksize: Number of items per chunk for worker distribution

        Returns:
            List of results in the same order as input items
        """
        if not self._initialized or not self._executor:
            self.initialize()
        assert self._executor is not None, "Executor should be initialized"

        effective_chunksize: int = (
            chunksize
            if chunksize is not None
            else (
                max(1, len(items) // (self.max_workers * 2))
                if len(items) > self.max_workers * 4
                else 1
            )
        )

        return list(self._executor.map(func, items, chunksize=effective_chunksize))

    def execute_batch(
        self,
        tasks: List[SimulationTask],
        timeout: Optional[float] = None,
        return_exceptions: bool = False,
    ) -> List[TaskResult[Any]]:
        """Execute a batch of tasks and return results.

        Args:
            tasks: List of simulation tasks
            timeout: Maximum time to wait for all tasks
            return_exceptions: If True, include failed tasks in results

        Returns:
            List of TaskResult objects
        """
        if not self._initialized or not self._executor:
            self.initialize()
        assert self._executor is not None, "Executor should be initialized"

        results: List[TaskResult[Any]] = []
        futures = {}

        # Sort by priority (higher priority = lower number)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority)

        for task in sorted_tasks:
            future = self._executor.submit(
                _execute_task_with_timing,
                task.func,
                task.args,
                task.kwargs,
            )
            futures[future] = task

        for future in as_completed(futures, timeout=timeout):
            task = futures[future]
            try:
                result, execution_time = future.result(timeout=task.timeout)
                results.append(
                    TaskResult(
                        task_id=task.task_id,
                        success=True,
                        result=result,
                        error=None,
                        execution_time=execution_time,
                    )
                )
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Task {task.task_id} failed: {error_msg}")
                if return_exceptions:
                    results.append(
                        TaskResult(
                            task_id=task.task_id,
                            success=False,
                            result=None,
                            error=error_msg,
                            execution_time=0.0,
                        )
                    )

        return results

    def __enter__(self) -> "ProcessPoolManager":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()


def _execute_task_wrapper(
    func: Callable[..., T],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> T:
    """Wrapper to execute a task in a worker process."""
    return func(*args, **kwargs)


def _execute_task_with_timing(
    func: Callable[..., T],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[T, float]:
    """Execute task and return result with timing."""
    import time

    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed
