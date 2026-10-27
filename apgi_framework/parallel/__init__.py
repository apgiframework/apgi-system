"""Parallel processing module for APGI Framework.

This module provides multiprocessing capabilities for CPU-bound simulations,
including process pools, task distribution, and result aggregation.
"""

from apgi_framework.parallel.batch_processor import ParallelBatchProcessor
from apgi_framework.parallel.process_pool import ProcessPoolManager, SimulationTask

__all__ = ["ProcessPoolManager", "SimulationTask", "ParallelBatchProcessor"]
