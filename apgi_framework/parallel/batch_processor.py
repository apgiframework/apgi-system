"""Parallel batch processor for simulation workloads.

Provides high-level interface for running batches of simulations
with automatic parallelization and progress tracking.
"""

import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")
from tqdm import tqdm

from apgi_framework.parallel.process_pool import (
    ProcessPoolManager,
    SimulationTask,
    TaskResult,
)

logger = logging.getLogger(__name__)


class ParallelBatchProcessor:
    """High-level batch processor for parallel simulation execution.

    Automatically handles:
    - Chunking large batches for optimal worker utilization
    - Progress tracking with tqdm
    - Error handling and recovery
    - Result aggregation and ordering
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        chunk_size: int = 10,
        show_progress: bool = True,
    ):
        """Initialize the batch processor.

        Args:
            max_workers: Maximum number of worker processes
            chunk_size: Number of tasks to submit as a batch
            show_progress: Whether to show progress bar
        """
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.show_progress = show_progress
        self._pool: Optional[ProcessPoolManager] = None

    def process(
        self,
        items: List[Any],
        process_func: Callable[[Any], T],
        description: str = "Processing",
    ) -> List[T]:
        """Process a list of items in parallel.

        Args:
            items: List of items to process
            process_func: Function to apply to each item
            description: Description for progress bar

        Returns:
            List of results in same order as input
        """
        with ProcessPoolManager(max_workers=self.max_workers) as pool:
            if self.show_progress:
                with tqdm(total=len(items), desc=description) as pbar:
                    results = pool.map(process_func, items)
                    pbar.update(len(items))
                return results
            else:
                return pool.map(process_func, items)

    def process_simulations(
        self,
        simulations: List[Dict[str, Any]],
        simulation_func: Callable[..., Any],
        description: str = "Running simulations",
    ) -> List[TaskResult[Any]]:
        """Run multiple simulations in parallel.

        Args:
            simulations: List of simulation parameter dictionaries
            simulation_func: Function to run each simulation
            description: Description for progress bar

        Returns:
            List of TaskResult objects
        """
        tasks = [
            SimulationTask(
                task_id=f"sim_{i:04d}",
                func=simulation_func,
                args=(sim,),
                kwargs={},
                priority=sim.get("priority", 0),
                timeout=sim.get("timeout"),
            )
            for i, sim in enumerate(simulations)
        ]

        with ProcessPoolManager(max_workers=self.max_workers) as pool:
            if self.show_progress:
                results = []
                with tqdm(total=len(tasks), desc=description) as pbar:
                    for chunk in self._chunk_tasks(tasks):
                        chunk_results = pool.execute_batch(chunk, return_exceptions=True)
                        results.extend(chunk_results)
                        pbar.update(len(chunk))
                return results
            else:
                return pool.execute_batch(tasks, return_exceptions=True)

    def _chunk_tasks(
        self,
        tasks: List[SimulationTask],
    ) -> Iterator[List[SimulationTask]]:
        """Split tasks into chunks for batch processing."""
        for i in range(0, len(tasks), self.chunk_size):
            yield tasks[i : i + self.chunk_size]

    def map_reduce(
        self,
        items: List[Any],
        map_func: Callable[[Any], T],
        reduce_func: Callable[[List[T]], R],
        description: str = "Map-Reduce",
    ) -> R:
        """Perform map-reduce operation in parallel.

        Args:
            items: Items to map over
            map_func: Function to apply to each item
            reduce_func: Function to reduce mapped results
            description: Description for progress bar

        Returns:
            Reduced result
        """
        mapped = self.process(items, map_func, f"{description} (map)")
        return reduce_func(mapped)
