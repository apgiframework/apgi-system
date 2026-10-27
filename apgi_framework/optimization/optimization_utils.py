"""Optimization Utilities for APGI Framework.

Provides various optimization techniques including caching, vectorization,
parallel processing, and memory management to improve performance.
"""

import functools
import gc
import hashlib
import multiprocessing as mp
import threading
import weakref
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

T = TypeVar("T")

import numpy as np
import pandas as pd

try:
    import numba  # type: ignore

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0


class LRUCache:
    """Thread-safe LRU cache with statistics."""

    def __init__(self, maxsize: int = 128) -> None:
        self.maxsize = maxsize
        self.cache: Dict[Any, Any] = {}
        self.access_order: List[Any] = []
        self.stats = CacheStats()
        self._lock = threading.RLock()

    def get(self, key: Any) -> Tuple[Any, bool]:
        """Get item from cache. Returns (value, hit)."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                self.stats.hits += 1
                return self.cache[key], True
            else:
                self.stats.misses += 1
                return None, False

    def put(self, key: Any, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            if key in self.cache:
                # Update existing
                self.cache[key] = value
                self.access_order.remove(key)
                self.access_order.append(key)
            else:
                # Add new
                if len(self.cache) >= self.maxsize:
                    # Evict least recently used
                    lru_key = self.access_order.pop(0)
                    del self.cache[lru_key]
                    self.stats.evictions += 1

                self.cache[key] = value
                self.access_order.append(key)

    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.stats.reset()

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class MemoryEfficientCache:
    """Memory-efficient cache using weak references."""

    def __init__(self, maxsize: int = 128) -> None:
        self.maxsize = maxsize
        self.cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.strong_refs: Dict[Any, Any] = {}
        self.access_order: List[Any] = []
        self.stats = CacheStats()
        self._lock = threading.RLock()

    def get(self, key: Any) -> Tuple[Any, bool]:
        """Get item from cache."""
        with self._lock:
            value = self.cache.get(key)
            if value is not None:
                # Move to end
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                self.stats.hits += 1
                return value, True
            else:
                self.stats.misses += 1
                return None, False

    def put(self, key: Any, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            # Store weak reference
            self.cache[key] = value

            # Keep strong reference for LRU management
            if len(self.strong_refs) >= self.maxsize:
                # Remove oldest strong reference
                if self.access_order:
                    old_key = self.access_order.pop(0)
                    if old_key in self.strong_refs:
                        del self.strong_refs[old_key]
                        self.stats.evictions += 1

            self.strong_refs[key] = value

            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)


def memoize(maxsize: int = 128, typed: bool = False, use_weak_refs: bool = False) -> Callable:
    """Memoization decorator with configurable cache.

    Args:
        maxsize: Maximum cache size
        typed: Whether to consider argument types in cache key
        use_weak_refs: Whether to use weak references for memory efficiency
    """

    def decorator(func: Callable) -> Callable:
        if use_weak_refs:
            cache: Union[LRUCache, MemoryEfficientCache] = MemoryEfficientCache(maxsize)
        else:
            cache = LRUCache(maxsize)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create cache key
            key = _make_cache_key(args, kwargs, typed)

            # Try to get from cache
            result, hit = cache.get(key)
            if hit:
                return result

            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result

        # Add cache management methods
        wrapper.cache_info = lambda: {  # type: ignore[attr-defined, union-attr]
            "hits": cache.stats.hits,
            "misses": cache.stats.misses,
            "hit_rate": cache.stats.hit_rate,
            "size": cache.size(),  # type: ignore[attr-defined, union-attr]
            "maxsize": maxsize,
        }
        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined, union-attr]

        return wrapper

    return decorator


def _make_cache_key(args: tuple, kwargs: dict, typed: bool) -> str:
    """Create a cache key from function arguments."""
    key_parts = []

    # Add positional arguments
    for arg in args:
        if isinstance(arg, (np.ndarray, pd.DataFrame)):
            # Use hash of array/dataframe content
            key_parts.append(f"array_{hash(arg.tobytes())}")  # type: ignore
        elif hasattr(arg, "__dict__"):
            # Use hash of object attributes
            key_parts.append(f"obj_{hash(str(sorted(list(arg.__dict__.items()))))}")
        else:
            key_parts.append(str(arg))
            if typed:
                key_parts.append(str(type(arg)))

    # Add keyword arguments
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
        if typed:
            key_parts.append(str(type(v)))

    # Create hash of combined key
    key_str = "|".join(key_parts)
    return hashlib.md5(
        key_str.encode(), usedforsecurity=False
    ).hexdigest()  # cache key, not security


class VectorizedOperations:
    """Optimized vectorized operations for common APGI computations."""

    @staticmethod
    def compute_surprise_vectorized(
        prediction_errors: np.ndarray, precisions: np.ndarray
    ) -> np.ndarray:
        """Vectorized surprise computation."""
        return precisions * np.abs(prediction_errors)  # type: ignore

    @staticmethod
    def sigmoid_vectorized(x: np.ndarray) -> np.ndarray:
        """Vectorized sigmoid function with overflow protection."""
        # Clip to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))  # type: ignore

    @staticmethod
    def compute_ignition_probability_vectorized(
        surprise: np.ndarray, threshold: np.ndarray
    ) -> np.ndarray:
        """Vectorized ignition probability computation."""
        return VectorizedOperations.sigmoid_vectorized(surprise - threshold)

    @staticmethod
    def batch_correlation(
        data1: np.ndarray, data2: np.ndarray, batch_size: int = 1000
    ) -> np.ndarray:
        """Compute correlations in batches for memory efficiency."""
        n_samples = data1.shape[0]
        correlations = np.zeros(n_samples)

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch1 = data1[i:end_idx]
            batch2 = data2[i:end_idx]

            # Compute correlation for batch
            correlations[i:end_idx] = np.array(
                [
                    np.corrcoef(b1, b2)[0, 1] if len(b1) > 1 else 0.0
                    for b1, b2 in zip(batch1, batch2)
                ]
            )

        return correlations


if NUMBA_AVAILABLE:

    @numba.jit(nopython=True, parallel=True)  # type: ignore[misc]
    def _numba_surprise_computation(prediction_errors: np.ndarray, precisions: np.ndarray) -> Any:
        """Numba-optimized surprise computation."""
        return precisions * np.abs(prediction_errors)

    @numba.jit(nopython=True, parallel=True)  # type: ignore[misc]
    def _numba_sigmoid(x: np.ndarray) -> Any:
        """Numba-optimized sigmoid function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    # Replace vectorized operations with numba versions if available
    setattr(VectorizedOperations, "compute_surprise_vectorized", _numba_surprise_computation)
    setattr(VectorizedOperations, "sigmoid_vectorized", _numba_sigmoid)


class ParallelProcessor:
    """Parallel processing utilities for APGI computations."""

    def __init__(self, n_workers: Optional[int] = None, use_processes: bool = True) -> None:
        self.n_workers = n_workers or mp.cpu_count()
        self.use_processes = use_processes

    def parallel_map(
        self, func: Callable, items: List[Any], chunk_size: Optional[int] = None
    ) -> List[Any]:
        """Apply function to items in parallel."""
        if len(items) == 0:
            return []

        if len(items) == 1:
            return [func(items[0])]

        chunk_size = chunk_size or max(1, len(items) // (self.n_workers * 4))

        if self.use_processes:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                return list(executor.map(func, items, chunksize=chunk_size))
        else:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                return list(executor.map(func, items))

    def parallel_apply(
        self,
        func: Callable,
        data: np.ndarray,
        axis: int = 0,
        chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        """Apply function along axis in parallel."""
        if axis == 0:
            chunks = np.array_split(data, self.n_workers)
        else:
            chunks = np.array_split(data, self.n_workers, axis=axis)

        results = self.parallel_map(func, chunks, chunk_size)

        if axis == 0:
            return np.concatenate(results, axis=0)
        else:
            return np.concatenate(results, axis=axis)


class MemoryManager:
    """Memory management utilities."""

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }

    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> Any:  # type: ignore[no-any-return]
        """Optimize DataFrame memory usage."""
        """Optimize DataFrame memory usage."""
        optimized_df = df.copy()

        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype

            if col_type != "object":
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()

                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized_df[col] = optimized_df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized_df[col] = optimized_df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized_df[col] = optimized_df[col].astype(np.int32)

                elif str(col_type)[:5] == "float":
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        optimized_df[col] = optimized_df[col].astype(np.float32)
            else:
                # Convert strings to categories if beneficial
                num_unique_values = len(optimized_df[col].unique())
                num_total_values = len(optimized_df[col])
                if num_unique_values / num_total_values < 0.5:
                    optimized_df[col] = optimized_df[col].astype("category")

        return optimized_df

    @staticmethod
    def force_garbage_collection() -> int:
        """Force garbage collection and return collected objects."""
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")
        return collected

    @staticmethod
    def get_large_objects(min_size_mb: float = 10) -> List[Dict[str, Any]]:
        """Get information about large objects in memory."""
        import sys

        large_objects = []
        min_size_bytes = min_size_mb * 1024 * 1024

        for obj in gc.get_objects():
            try:
                size = sys.getsizeof(obj)
                if size > min_size_bytes:
                    large_objects.append(
                        {
                            "type": type(obj).__name__,
                            "size_mb": size / 1024 / 1024,
                            "id": id(obj),
                        }
                    )
            except Exception:
                continue

        return sorted(large_objects, key=lambda x: x["size_mb"], reverse=True)


class BatchProcessor:
    """Process data in batches to manage memory usage."""

    def __init__(self, batch_size: int = 1000) -> None:
        self.batch_size = batch_size

    def process_batches(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        func: Callable[..., Any],
        **kwargs: Any,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Process data in batches."""
        if isinstance(data, np.ndarray):
            return self._process_array_batches(data, func, **kwargs)
        elif isinstance(data, pd.DataFrame):
            return self._process_dataframe_batches(data, func, **kwargs)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _process_array_batches(
        self, data: np.ndarray, func: Callable[..., Any], **kwargs: Any
    ) -> np.ndarray:
        """Process numpy array in batches."""
        n_samples = data.shape[0]
        results = []

        for i in range(0, n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, n_samples)
            batch = data[i:end_idx]
            batch_result = func(batch, **kwargs)
            results.append(batch_result)

        return np.concatenate(results, axis=0)

    def _process_dataframe_batches(
        self, data: pd.DataFrame, func: Callable[..., Any], **kwargs: Any
    ) -> pd.DataFrame:
        """Process DataFrame in batches."""
        n_samples = len(data)
        results = []

        for i in range(0, n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, n_samples)
            batch = data.iloc[i:end_idx]
            batch_result = func(batch, **kwargs)
            results.append(batch_result)

        return pd.concat(results, ignore_index=True)  # type: ignore


# Optimization decorators and utilities
def optimize_memory(func: Callable) -> Callable:
    """Decorator to optimize memory usage during function execution."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Force garbage collection before
        MemoryManager.force_garbage_collection()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Force garbage collection after
            MemoryManager.force_garbage_collection()

    return wrapper


def batch_process(
    batch_size: int = 1000,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to automatically batch process array/dataframe inputs."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        processor = BatchProcessor(batch_size)

        @functools.wraps(func)
        def wrapper(data: Any, *args: Any, **kwargs: Any) -> T:
            if isinstance(data, (np.ndarray, pd.DataFrame)) and len(data) > batch_size:
                result = processor.process_batches(data, func, *args, **kwargs)
                return result  # type: ignore[return-value]
            else:
                return func(data, *args, **kwargs)

        return wrapper

    return decorator


def parallel_process(
    n_workers: Optional[int] = None, use_processes: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to automatically parallelize function execution."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        processor = ParallelProcessor(n_workers, use_processes)

        @functools.wraps(func)
        def wrapper(items: Any, *args: Any, **kwargs: Any) -> T:
            if isinstance(items, (list, tuple)) and len(items) > 1:
                # Create partial function with additional arguments
                partial_func = functools.partial(func, *args, **kwargs)
                result = processor.parallel_map(partial_func, list(items))  # type: ignore[arg-type]
                return result  # type: ignore[return-value]
            else:
                return func(items, *args, **kwargs)

        return wrapper

    return decorator


# Global instances
_global_parallel_processor = None
_global_memory_manager = MemoryManager()


def get_parallel_processor(n_workers: Optional[int] = None) -> ParallelProcessor:
    """Get global parallel processor instance."""
    global _global_parallel_processor
    if _global_parallel_processor is None:
        _global_parallel_processor = ParallelProcessor(n_workers)
    return _global_parallel_processor
