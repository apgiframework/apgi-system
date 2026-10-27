"""Performance Optimizer for Test Execution.

Provides intelligent test result caching, parallel execution tuning,
performance benchmarking, and execution optimization for the APGI framework.
"""

import hashlib
import json
import os
import sqlite3
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from ..logging.standardized_logging import get_logger
from ..utils.serialization import deserialize_json, serialize_json
from .batch_runner import BatchExecutionSummary, BatchTestRunner, TestResult

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Test result cache entry."""

    test_file: str
    test_hash: str
    dependency_hashes: Dict[str, str]
    result: TestResult
    cached_at: datetime
    cache_hits: int = 0


@dataclass
class PerformanceBenchmark:
    """Performance benchmark data."""

    test_file: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime
    test_count: int
    parallel_workers: int


@dataclass
class ExecutionProfile:
    """Test execution performance profile."""

    test_file: str
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float
    std_execution_time: float
    execution_count: int
    last_updated: datetime
    complexity_score: float = 0.0


class ResultCache:
    """Intelligent test result caching with dependency tracking."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or "test_cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize SQLite database for cache metadata
        self.db_path = self.cache_dir / "cache.db"
        self._init_database()

        # In-memory cache for frequently accessed entries
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.Lock()

        # File hash cache with mtime tracking for incremental updates
        self._file_hash_cache: Dict[str, Tuple[str, float]] = {}  # path -> (hash, mtime)
        self._file_hash_lock = threading.Lock()

        logger.info(f"Test cache initialized at {self.cache_dir}")

    def _init_database(self) -> None:
        """Initialize cache database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    test_file TEXT PRIMARY KEY,
                    test_hash TEXT NOT NULL,
                    dependency_hashes TEXT NOT NULL,
                    result_data BLOB NOT NULL,
                    cached_at TIMESTAMP NOT NULL,
                    cache_hits INTEGER DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cached_at ON cache_entries(cached_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_test_hash ON cache_entries(test_hash)
            """)

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of a file with mtime-based caching."""
        try:
            # Get current mtime first
            current_mtime = os.path.getmtime(file_path)

            # Check if we have a cached hash for this file
            with self._file_hash_lock:
                if file_path in self._file_hash_cache:
                    cached_hash, cached_mtime = self._file_hash_cache[file_path]
                    # If file hasn't changed, return cached hash
                    if current_mtime == cached_mtime:
                        return cached_hash

            # File changed or not cached - compute new hash
            with open(file_path, "rb") as f:
                content = f.read()
            new_hash = hashlib.sha256(content).hexdigest()

            # Update cache
            with self._file_hash_lock:
                self._file_hash_cache[file_path] = (new_hash, current_mtime)

            return new_hash
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            return ""

    def _get_test_dependencies(self, test_file: str) -> List[str]:
        """Get list of files that the test depends on."""
        dependencies = []
        test_path = Path(test_file)

        # Add the test file itself
        dependencies.append(test_file)

        # Add common dependency patterns
        module_dir = test_path.parent.parent
        if module_dir.exists():
            # Add all Python files in the module directory
            for py_file in module_dir.rglob("*.py"):
                if not py_file.name.startswith("test_"):
                    dependencies.append(str(py_file))

        # Add configuration files
        config_files = ["pytest.ini", "pyproject.toml", "setup.py", "requirements.txt"]

        for config_file in config_files:
            if Path(config_file).exists():
                dependencies.append(config_file)

        return dependencies

    def _calculate_dependency_hashes(self, test_file: str) -> Dict[str, str]:
        """Calculate hashes for all test dependencies with parallel processing."""
        dependencies = self._get_test_dependencies(test_file)

        # Use thread pool for parallel hash computation on large dependency trees
        if len(dependencies) > 10:
            hashes = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_file = {
                    executor.submit(self._calculate_file_hash, dep_file): dep_file
                    for dep_file in dependencies
                }
                for future in as_completed(future_to_file):
                    dep_file = future_to_file[future]
                    try:
                        hashes[dep_file] = future.result()
                    except Exception as e:
                        logger.warning(f"Failed to hash {dep_file}: {e}")
                        hashes[dep_file] = ""
            return hashes
        else:
            # For small dependency trees, use sequential processing
            return {dep_file: self._calculate_file_hash(dep_file) for dep_file in dependencies}

    def get_cached_result(self, test_file: str) -> Optional[TestResult]:
        """Get cached test result if valid."""
        with self._cache_lock:
            # Check memory cache first
            if test_file in self._memory_cache:
                entry = self._memory_cache[test_file]

                # Validate cache entry
                if self._is_cache_valid(entry):
                    entry.cache_hits += 1
                    self._update_cache_hits(test_file, entry.cache_hits)
                    logger.debug(f"Cache hit for {test_file}")
                    return entry.result
                else:
                    # Remove invalid entry
                    del self._memory_cache[test_file]
                    self._remove_from_database(test_file)

            # Check database cache
            db_entry: Optional[CacheEntry] = self._load_from_database(test_file)
            if db_entry and self._is_cache_valid(db_entry):
                # Load into memory cache
                self._memory_cache[test_file] = db_entry
                db_entry.cache_hits += 1
                self._update_cache_hits(test_file, db_entry.cache_hits)
                logger.debug(f"Database cache hit for {test_file}")
                return db_entry.result

            return None

    def cache_result(self, test_file: str, result: TestResult) -> None:
        """Cache a test result."""
        try:
            # Calculate hashes
            test_hash = self._calculate_file_hash(test_file)
            dependency_hashes = self._calculate_dependency_hashes(test_file)

            # Create cache entry
            entry = CacheEntry(
                test_file=test_file,
                test_hash=test_hash,
                dependency_hashes=dependency_hashes,
                result=result,
                cached_at=datetime.now(),
                cache_hits=0,
            )

            with self._cache_lock:
                # Store in memory cache
                self._memory_cache[test_file] = entry

                # Store in database
                self._save_to_database(entry)

            logger.debug(f"Cached result for {test_file}")

        except Exception as e:
            logger.error(f"Error caching result for {test_file}: {e}")

    def _is_cache_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid."""
        try:
            # Check if test file hash matches
            current_hash = self._calculate_file_hash(entry.test_file)
            if current_hash != entry.test_hash:
                return False

            # Check dependency hashes
            current_dep_hashes = self._calculate_dependency_hashes(entry.test_file)
            for dep_file, cached_hash in entry.dependency_hashes.items():
                if dep_file not in current_dep_hashes:
                    continue  # Dependency no longer exists

                if current_dep_hashes[dep_file] != cached_hash:
                    return False

            # Check cache age (invalidate after 24 hours)
            age = datetime.now() - entry.cached_at
            if age > timedelta(hours=24):
                return False

            return True

        except Exception as e:
            logger.warning(f"Error validating cache entry for {entry.test_file}: {e}")
            return False

    def _save_to_database(self, entry: CacheEntry) -> None:
        """Save cache entry to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries 
                (test_file, test_hash, dependency_hashes, result_data, cached_at, cache_hits)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    entry.test_file,
                    entry.test_hash,
                    json.dumps(entry.dependency_hashes),
                    zlib.compress(serialize_json(entry.result).encode("utf-8")),
                    entry.cached_at,
                    entry.cache_hits,
                ),
            )

    def _safe_load_json_data(self, compressed_data: bytes) -> Any:
        """Safely load compressed JSON data."""
        try:
            # Decompress first
            decompressed_data = zlib.decompress(compressed_data)

            # Deserialize JSON
            return deserialize_json(decompressed_data.decode("utf-8"))

        except Exception as e:
            logger.error(f"Failed to safely load JSON data: {e}")
            raise ValueError(f"Unsafe or corrupted JSON data: {e}")

    def _load_from_database(self, test_file: str) -> Optional[CacheEntry]:
        """Load cache entry from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT test_hash, dependency_hashes, result_data, cached_at, cache_hits
                    FROM cache_entries WHERE test_file = ?
                """,
                    (test_file,),
                )

                row = cursor.fetchone()
                if not row:
                    return None

                test_hash, dep_hashes_json, result_data, cached_at, cache_hits = row

                return CacheEntry(
                    test_file=test_file,
                    test_hash=test_hash,
                    dependency_hashes=json.loads(dep_hashes_json),
                    result=self._safe_load_json_data(result_data),
                    cached_at=datetime.fromisoformat(cached_at),
                    cache_hits=cache_hits,
                )

        except Exception as e:
            logger.error(f"Error loading cache entry for {test_file}: {e}")
            return None

    def _update_cache_hits(self, test_file: str, cache_hits: int) -> None:
        """Update cache hit count in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE cache_entries SET cache_hits = ? WHERE test_file = ?
            """,
                (cache_hits, test_file),
            )

    def _remove_from_database(self, test_file: str) -> None:
        """Remove cache entry from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache_entries WHERE test_file = ?", (test_file,))

    def clear_cache(self, older_than_hours: Optional[int] = None) -> None:
        """Clear cache entries."""
        with self._cache_lock:
            if older_than_hours is None:
                # Clear all
                self._memory_cache.clear()
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM cache_entries")
                logger.info("Cleared all cache entries")
            else:
                # Clear old entries
                cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

                # Remove from memory cache
                to_remove = [
                    test_file
                    for test_file, entry in self._memory_cache.items()
                    if entry.cached_at < cutoff_time
                ]
                for test_file in to_remove:
                    del self._memory_cache[test_file]

                # Remove from database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM cache_entries WHERE cached_at < ?", (cutoff_time,))

                logger.info(f"Cleared cache entries older than {older_than_hours} hours")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*), SUM(cache_hits) FROM cache_entries")
            total_entries, total_hits = cursor.fetchone()

            cursor = conn.execute("""
                SELECT COUNT(*) FROM cache_entries 
                WHERE cached_at > datetime('now', '-24 hours')
            """)
            recent_entries = cursor.fetchone()[0]

        return {
            "total_entries": total_entries or 0,
            "total_cache_hits": total_hits or 0,
            "recent_entries": recent_entries or 0,
            "memory_cache_size": len(self._memory_cache),
            "cache_directory": str(self.cache_dir),
        }


class ParallelExecutionOptimizer:
    """Optimizes parallel test execution based on performance profiles."""

    def __init__(self) -> None:
        self.execution_profiles: Dict[str, ExecutionProfile] = {}
        self.benchmarks: List[PerformanceBenchmark] = []
        self._profiles_lock = threading.Lock()

        # Load existing profiles
        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load execution profiles from disk."""
        profiles_file = Path("test_execution_profiles.json")
        if profiles_file.exists():
            try:
                with open(profiles_file, "r") as f:
                    data = json.load(f)

                for test_file, profile_data in data.items():
                    profile_data["last_updated"] = datetime.fromisoformat(
                        profile_data["last_updated"]
                    )
                    self.execution_profiles[test_file] = ExecutionProfile(**profile_data)

                logger.info(f"Loaded {len(self.execution_profiles)} execution profiles")

            except Exception as e:
                logger.error(f"Error loading execution profiles: {e}")

    def _save_profiles(self) -> None:
        """Save execution profiles to disk."""
        try:
            profiles_data = {}
            for test_file, profile in self.execution_profiles.items():
                profile_dict = asdict(profile)
                profile_dict["last_updated"] = profile.last_updated.isoformat()
                profiles_data[test_file] = profile_dict

            with open("test_execution_profiles.json", "w") as f:
                json.dump(profiles_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving execution profiles: {e}")

    def update_profile(
        self, test_file: str, execution_time: float, memory_usage: float = 0.0
    ) -> None:
        """Update execution profile for a test."""
        with self._profiles_lock:
            if test_file in self.execution_profiles:
                profile = self.execution_profiles[test_file]

                # Update statistics with proper floating-point handling
                total_time = profile.avg_execution_time * profile.execution_count + execution_time
                profile.execution_count += 1
                new_avg = total_time / profile.execution_count

                # Update min/max first, then average to maintain consistency
                new_min = min(profile.min_execution_time, execution_time)
                new_max = max(profile.max_execution_time, execution_time)

                # Ensure average is within min/max bounds (handle floating-point precision)
                profile.avg_execution_time = max(new_min, min(new_max, new_avg))
                profile.min_execution_time = new_min
                profile.max_execution_time = new_max

                # Calculate standard deviation (simplified)
                profile.std_execution_time = (
                    profile.max_execution_time - profile.min_execution_time
                ) / 2
                profile.last_updated = datetime.now()

                # Update complexity score
                profile.complexity_score = self._calculate_complexity_score(profile)

            else:
                # Create new profile
                self.execution_profiles[test_file] = ExecutionProfile(
                    test_file=test_file,
                    avg_execution_time=execution_time,
                    min_execution_time=execution_time,
                    max_execution_time=execution_time,
                    std_execution_time=0.0,
                    execution_count=1,
                    last_updated=datetime.now(),
                    complexity_score=self._calculate_complexity_score_from_time(execution_time),
                )

        # Save profiles periodically
        if len(self.execution_profiles) % 10 == 0:
            self._save_profiles()

    def _calculate_complexity_score(self, profile: ExecutionProfile) -> float:
        """Calculate complexity score for a test."""
        # Base score on execution time and variability
        time_score = min(profile.avg_execution_time / 10.0, 10.0)  # 0-10 scale
        variability_score = (
            min(profile.std_execution_time / profile.avg_execution_time * 5, 5.0)
            if profile.avg_execution_time > 0
            else 0
        )

        return time_score + variability_score

    def _calculate_complexity_score_from_time(self, execution_time: float) -> float:
        """Calculate initial complexity score from execution time."""
        return min(execution_time / 10.0, 10.0)

    def optimize_parallel_execution(
        self, test_files: List[str], max_workers: Optional[int] = None
    ) -> Tuple[int, List[List[str]]]:
        """Optimize parallel execution by grouping tests intelligently."""
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)

        # Get profiles for all tests
        test_profiles = []
        for test_file in test_files:
            if test_file in self.execution_profiles:
                profile = self.execution_profiles[test_file]
                test_profiles.append(
                    (test_file, profile.avg_execution_time, profile.complexity_score)
                )
            else:
                # Estimate for unknown tests
                test_profiles.append((test_file, 5.0, 5.0))  # Default estimates

        # Sort by complexity (most complex first)
        test_profiles.sort(key=lambda x: x[2], reverse=True)

        # Group tests into batches for optimal load balancing
        batches: List[List[str]] = [[] for _ in range(max_workers)]
        batch_loads = [0.0] * max_workers

        for test_file, exec_time, complexity in test_profiles:
            # Find batch with minimum load
            min_load_idx = batch_loads.index(min(batch_loads))
            batches[min_load_idx].append(test_file)
            batch_loads[min_load_idx] += exec_time

        # Remove empty batches
        non_empty_batches = [batch for batch in batches if batch]
        optimal_workers = len(non_empty_batches)

        logger.info(f"Optimized execution: {optimal_workers} workers for {len(test_files)} tests")
        logger.debug(f"Batch loads: {[f'{load:.1f}s' for load in batch_loads[:optimal_workers]]}")

        return optimal_workers, non_empty_batches

    def add_benchmark(
        self,
        test_file: str,
        execution_time: float,
        memory_usage: float,
        cpu_usage: float,
        test_count: int,
        parallel_workers: int,
    ) -> None:
        """Add performance benchmark."""
        benchmark = PerformanceBenchmark(
            test_file=test_file,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            timestamp=datetime.now(),
            test_count=test_count,
            parallel_workers=parallel_workers,
        )

        self.benchmarks.append(benchmark)

        # Keep only recent benchmarks
        if len(self.benchmarks) > 1000:
            self.benchmarks = self.benchmarks[-1000:]

    def get_performance_regression_report(self) -> Dict[str, Any]:
        """Generate performance regression report."""
        regressions = []
        improvements = []

        for test_file, profile in self.execution_profiles.items():
            if profile.execution_count < 2:
                continue

            # Compare recent performance with historical average
            recent_benchmarks = [
                b
                for b in self.benchmarks
                if b.test_file == test_file and b.timestamp > datetime.now() - timedelta(days=7)
            ]

            if not recent_benchmarks:
                continue

            recent_avg = sum(b.execution_time for b in recent_benchmarks) / len(recent_benchmarks)
            historical_avg = profile.avg_execution_time

            change_percent = ((recent_avg - historical_avg) / historical_avg) * 100

            if change_percent > 20:  # 20% slower
                regressions.append(
                    {
                        "test_file": test_file,
                        "historical_avg": historical_avg,
                        "recent_avg": recent_avg,
                        "change_percent": change_percent,
                    }
                )
            elif change_percent < -20:  # 20% faster
                improvements.append(
                    {
                        "test_file": test_file,
                        "historical_avg": historical_avg,
                        "recent_avg": recent_avg,
                        "change_percent": change_percent,
                    }
                )

        return {
            "regressions": sorted(regressions, key=lambda x: x["change_percent"], reverse=True),
            "improvements": sorted(improvements, key=lambda x: x["change_percent"]),
            "total_tests_profiled": len(self.execution_profiles),
            "total_benchmarks": len(self.benchmarks),
        }


class PerformanceOptimizedTestRunner(BatchTestRunner):
    """Test runner with performance optimizations."""

    def __init__(self, enable_caching: bool = True, enable_optimization: bool = True) -> None:
        super().__init__()

        self.enable_caching = enable_caching
        self.enable_optimization = enable_optimization

        # Initialize components
        self.cache = ResultCache() if enable_caching else None
        self.optimizer = ParallelExecutionOptimizer() if enable_optimization else None

        logger.info(
            f"Performance-optimized runner initialized (caching: {enable_caching}, optimization: {enable_optimization})"
        )

    def run_optimized_batch_tests(
        self,
        test_selection: Optional[List[str]] = None,
        use_cache: bool = True,
        optimize_parallel: bool = True,
        **kwargs: Any,
    ) -> BatchExecutionSummary:
        """Run batch tests with performance optimizations."""
        start_time = datetime.now()

        # Discover tests
        if test_selection is None:
            test_files = self.discover_tests()
        else:
            test_files = test_selection

        if not test_files:
            logger.warning("No tests found to execute")
            return self._create_empty_summary(start_time)

        logger.info(f"Running optimized batch execution of {len(test_files)} tests")

        # Check cache for results
        cached_results = []
        uncached_tests = []

        if self.cache and use_cache:
            for test_file in test_files:
                cached_result = self.cache.get_cached_result(test_file)
                if cached_result:
                    cached_results.append(cached_result)
                    logger.debug(f"Using cached result for {test_file}")
                else:
                    uncached_tests.append(test_file)

            logger.info(f"Cache hits: {len(cached_results)}, Cache misses: {len(uncached_tests)}")
        else:
            uncached_tests = test_files

        # Run uncached tests
        new_results = []
        if uncached_tests:
            if self.optimizer and optimize_parallel:
                # Optimize parallel execution
                (
                    optimal_workers,
                    test_batches,
                ) = self.optimizer.optimize_parallel_execution(
                    uncached_tests, kwargs.get("max_workers")
                )
                kwargs["max_workers"] = optimal_workers

                # Run with optimized batching
                new_results = self._run_optimized_parallel_tests(test_batches, **kwargs)
            else:
                # Run normally
                summary = super().run_batch_tests(test_selection=uncached_tests, **kwargs)
                new_results = summary.test_results

        # Cache new results
        if self.cache and use_cache:
            for result in new_results:
                self.cache.cache_result(result.test_file, result)

        # Update performance profiles
        if self.optimizer:
            for result in new_results:
                self.optimizer.update_profile(result.test_file, result.duration)

        # Combine results
        all_results = cached_results + new_results

        # Create summary
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        summary = BatchExecutionSummary(
            total_tests=len(all_results),
            passed=sum(1 for r in all_results if r.status == "passed"),
            failed=sum(1 for r in all_results if r.status == "failed"),
            skipped=sum(1 for r in all_results if r.status == "skipped"),
            errors=sum(1 for r in all_results if r.status == "error"),
            total_duration=total_duration,
            start_time=start_time,
            end_time=end_time,
            test_results=all_results,
            execution_metadata={
                "performance_optimized": True,
                "cache_enabled": self.enable_caching,
                "cache_hits": len(cached_results),
                "cache_misses": len(uncached_tests),
                "optimization_enabled": self.enable_optimization,
                **kwargs,
            },
        )

        logger.info(f"Optimized execution completed in {total_duration:.2f}s")
        return summary

    def _run_optimized_parallel_tests(
        self, test_batches: List[List[str]], **kwargs: Any
    ) -> List[TestResult]:
        """Run tests with optimized parallel batching."""
        all_results = []

        with ThreadPoolExecutor(max_workers=len(test_batches)) as executor:
            # Submit batch tasks
            future_to_batch = {
                executor.submit(self._run_test_batch, batch, kwargs.get("timeout", 300)): batch
                for batch in test_batches
            }

            # Collect results
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Error executing test batch: {e}")
                    # Create error results for the batch
                    for test_file in batch:
                        error_result = TestResult(
                            test_name=test_file,
                            test_file=test_file,
                            status="error",
                            duration=0.0,
                            output="",
                            error_message=str(e),
                        )
                        all_results.append(error_result)

        return all_results

    def _run_test_batch(self, batch: List[str], timeout: int) -> List[TestResult]:
        """Run a batch of tests sequentially."""
        results = []

        for test_file in batch:
            if self._stop_execution:
                break

            result = self._run_single_test(test_file, timeout)
            results.append(result)

        return results

    def _create_empty_summary(self, start_time: datetime) -> BatchExecutionSummary:
        """Create empty summary."""
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

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {}

        if self.cache:
            report["cache_stats"] = self.cache.get_cache_stats()

        if self.optimizer:
            report["regression_report"] = self.optimizer.get_performance_regression_report()
            execution_profiles = [
                {
                    "test_file": test_file,
                    "avg_duration": profile.avg_execution_time,
                    "complexity_score": profile.complexity_score,
                }
                for test_file, profile in self.optimizer.execution_profiles.items()
            ]
            report["execution_profiles"] = cast(
                List[Dict[str, Any]], execution_profiles
            )  # type: ignore

        return report

    def clear_cache(self, older_than_hours: Optional[int] = None) -> None:
        """Clear performance cache."""
        if self.cache:
            self.cache.clear_cache(older_than_hours)


# Convenience functions
def run_performance_optimized_tests(
    test_selection: Optional[List[str]] = None,
    enable_caching: bool = True,
    enable_optimization: bool = True,
    **kwargs: Any,
) -> BatchExecutionSummary:
    """Run tests with performance optimizations."""
    runner = PerformanceOptimizedTestRunner(
        enable_caching=enable_caching, enable_optimization=enable_optimization
    )

    return runner.run_optimized_batch_tests(test_selection=test_selection, **kwargs)


def benchmark_test_performance(test_files: List[str], iterations: int = 3) -> Dict[str, Any]:
    """Benchmark test performance across multiple runs."""
    runner = PerformanceOptimizedTestRunner(enable_caching=False)  # Disable cache for benchmarking
    benchmarks = []

    for i in range(iterations):
        logger.info(f"Benchmark iteration {i + 1}/{iterations}")

        start_time = time.time()
        summary = runner.run_optimized_batch_tests(test_selection=test_files, use_cache=False)
        end_time = time.time()

        benchmarks.append(
            {
                "iteration": i + 1,
                "total_duration": end_time - start_time,
                "test_count": summary.total_tests,
                "passed": summary.passed,
                "failed": summary.failed,
                "avg_test_time": (
                    summary.total_duration / summary.total_tests if summary.total_tests > 0 else 0
                ),
            }
        )

    # Calculate statistics
    durations = [b["total_duration"] for b in benchmarks]
    avg_test_times = [b["avg_test_time"] for b in benchmarks]

    return {
        "benchmarks": benchmarks,
        "statistics": {
            "avg_total_duration": sum(durations) / len(durations),
            "min_total_duration": min(durations),
            "max_total_duration": max(durations),
            "avg_test_time": sum(avg_test_times) / len(avg_test_times),
            "min_test_time": min(avg_test_times),
            "max_test_time": max(avg_test_times),
        },
        "test_count": len(test_files),
        "iterations": iterations,
    }
