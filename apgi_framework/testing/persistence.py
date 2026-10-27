"""Test Result Persistence System for APGI Framework.

This module provides comprehensive test result storage, retrieval, and analysis capabilities including:
- Database storage for test results
- Historical trend analysis
- Performance metrics tracking
- Query and filtering capabilities
- Result aggregation and reporting
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..config import ConfigManager
from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)


@dataclass
class TestExecutionRecord:
    """Test execution record for database storage."""

    id: Optional[int] = None
    test_name: str = ""
    test_file: str = ""
    status: str = ""
    duration: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    output: Optional[str] = None
    execution_context: Optional[str] = None  # JSON string
    batch_id: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class BatchExecutionRecord:
    """Batch execution record for database storage."""

    id: Optional[int] = None
    batch_id: str = ""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    total_duration: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_metadata: Optional[str] = None  # JSON string
    created_at: Optional[datetime] = None


class TestResultPersistence:
    """Advanced test result persistence and analysis system."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        config_manager: Optional[ConfigManager] = None,
    ):
        """Initialize the persistence system."""
        self.config_manager = config_manager or ConfigManager()
        self.logger = get_logger(f"{__name__}.TestResultPersistence")

        if db_path is None:
            db_path = self.config_manager.get_experimental_config().output_directory
            db_path = str(Path(db_path) / "test_results.db")

        self.db_path: Path = Path(db_path) if db_path else Path("test_results.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._initialize_database()

    def _log_operation(self, operation: str, details: str) -> None:
        """Log a database operation for auditing."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"AUDIT | {timestamp} | {operation} | {details}")

    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # Enable dictionary-like access
            yield conn
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _initialize_database(self) -> None:
        """Initialize database tables."""
        self._log_operation("INITIALIZE", "Database tables initialization")
        with self.get_connection() as conn:
            # Create batch_executions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batch_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT UNIQUE NOT NULL,
                    total_tests INTEGER NOT NULL,
                    passed INTEGER NOT NULL,
                    failed INTEGER NOT NULL,
                    skipped INTEGER NOT NULL,
                    errors INTEGER NOT NULL,
                    total_duration REAL NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP NOT NULL,
                    execution_metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create test_executions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    test_file TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration REAL NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP NOT NULL,
                    error_message TEXT,
                    traceback TEXT,
                    output TEXT,
                    execution_context TEXT,
                    batch_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (batch_id) REFERENCES batch_executions (batch_id)
                )
            """)

            # Create indexes for better query performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_test_executions_batch_id ON test_executions(batch_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_test_executions_test_name ON test_executions(test_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_test_executions_status ON test_executions(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_test_executions_start_time ON test_executions(start_time)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_batch_executions_start_time ON batch_executions(start_time)"
            )

            # Create test_performance table for trend analysis
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    test_file TEXT NOT NULL,
                    avg_duration REAL NOT NULL,
                    min_duration REAL NOT NULL,
                    max_duration REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    total_executions INTEGER NOT NULL,
                    last_execution TIMESTAMP NOT NULL,
                    trend_data TEXT,  -- JSON array of recent execution times
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(test_name, test_file)
                )
            """)

            conn.commit()
            self.logger.info(f"Database initialized at {self.db_path}")

    def store_batch_results(self, batch_summary: Any, batch_id: Optional[str] = None) -> str:
        """Store batch execution results in database."""
        if batch_id is None:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._log_operation("WRITE", f"Storing batch results for {batch_id}")
        try:
            with self.get_connection() as conn:
                # Store batch execution record
                batch_record = BatchExecutionRecord(
                    batch_id=batch_id,
                    total_tests=batch_summary.total_tests,
                    passed=batch_summary.passed,
                    failed=batch_summary.failed,
                    skipped=batch_summary.skipped,
                    errors=batch_summary.errors,
                    total_duration=batch_summary.total_duration,
                    start_time=batch_summary.start_time,
                    end_time=batch_summary.end_time,
                    execution_metadata=json.dumps(batch_summary.execution_metadata),
                )

                conn.execute(
                    """
                    INSERT OR REPLACE INTO batch_executions 
                    (batch_id, total_tests, passed, failed, skipped, errors, 
                     total_duration, start_time, end_time, execution_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        batch_record.batch_id,
                        batch_record.total_tests,
                        batch_record.passed,
                        batch_record.failed,
                        batch_record.skipped,
                        batch_record.errors,
                        batch_record.total_duration,
                        batch_record.start_time,
                        batch_record.end_time,
                        batch_record.execution_metadata,
                    ),
                )

                # Store individual test execution records
                for result in batch_summary.test_results:
                    test_record = TestExecutionRecord(
                        test_name=result.test_name,
                        test_file=result.test_file,
                        status=result.status,
                        duration=result.duration,
                        start_time=result.start_time,
                        end_time=result.end_time,
                        error_message=result.error_message,
                        traceback=result.traceback,
                        output=result.output,
                        batch_id=batch_id,
                    )

                    conn.execute(
                        """
                        INSERT INTO test_executions 
                        (test_name, test_file, status, duration, start_time, end_time,
                         error_message, traceback, output, batch_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            test_record.test_name,
                            test_record.test_file,
                            test_record.status,
                            test_record.duration,
                            test_record.start_time,
                            test_record.end_time,
                            test_record.error_message,
                            test_record.traceback,
                            test_record.output,
                            test_record.batch_id,
                        ),
                    )

                # Update performance metrics
                self._update_performance_metrics(conn, batch_summary.test_results)

                conn.commit()
                self.logger.info(f"Stored batch results for batch_id: {batch_id}")
                return batch_id

        except Exception as e:
            self.logger.error(f"Failed to store batch results: {e}")
            raise

    def _update_performance_metrics(
        self, conn: sqlite3.Connection, test_results: List[Any]
    ) -> None:
        """Update performance metrics for trend analysis."""
        for result in test_results:
            test_name = result.test_name
            test_file = result.test_file

            # Get existing performance data
            cursor = conn.execute(
                """
                SELECT avg_duration, min_duration, max_duration, success_rate, 
                       total_executions, trend_data
                FROM test_performance 
                WHERE test_name = ? AND test_file = ?
            """,
                (test_name, test_file),
            )

            existing = cursor.fetchone()

            # Calculate new metrics
            if existing:
                avg_duration = existing["avg_duration"]
                min_duration = existing["min_duration"]
                max_duration = existing["max_duration"]
                success_rate = existing["success_rate"]
                total_executions = existing["total_executions"]
                trend_data = json.loads(existing["trend_data"] or "[]")

                # Update running averages
                total_executions += 1
                avg_duration = (
                    (avg_duration * (total_executions - 1)) + result.duration
                ) / total_executions
                min_duration = min(min_duration, result.duration)
                max_duration = max(max_duration, result.duration)

                # Update success rate
                if result.status == "passed":
                    success_rate = (
                        (success_rate * (total_executions - 1)) + 1.0
                    ) / total_executions
                else:
                    success_rate = (
                        (success_rate * (total_executions - 1)) + 0.0
                    ) / total_executions

                # Update trend data (keep last 50 executions)
                trend_data.append(
                    {
                        "timestamp": result.end_time.isoformat(),
                        "duration": result.duration,
                        "status": result.status,
                    }
                )
                if len(trend_data) > 50:
                    trend_data = trend_data[-50:]
            else:
                # First execution
                avg_duration = result.duration
                min_duration = result.duration
                max_duration = result.duration
                success_rate = 1.0 if result.status == "passed" else 0.0
                total_executions = 1
                trend_data = [
                    {
                        "timestamp": result.end_time.isoformat(),
                        "duration": result.duration,
                        "status": result.status,
                    }
                ]

            # Store updated metrics
            conn.execute(
                """
                INSERT OR REPLACE INTO test_performance 
                (test_name, test_file, avg_duration, min_duration, max_duration,
                 success_rate, total_executions, last_execution, trend_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    test_name,
                    test_file,
                    avg_duration,
                    min_duration,
                    max_duration,
                    success_rate,
                    total_executions,
                    result.end_time,
                    json.dumps(trend_data),
                ),
            )

    def get_test_history(
        self,
        test_name: Optional[str] = None,
        test_file: Optional[str] = None,
        days: int = 30,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get test execution history."""
        self._log_operation(
            "READ", f"Fetching test history (name={test_name}, file={test_file}, days={days})"
        )
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM test_executions 
                    WHERE start_time >= datetime('now', '-{} days')
                """.format(int(days))

                params: List[Any] = []
                if test_name:
                    query += " AND test_name = ?"
                    params.append(test_name)
                if test_file:
                    query += " AND test_file = ?"
                    params.append(test_file)

                query += " ORDER BY start_time DESC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            self.logger.error(f"Failed to get test history: {e}")
            return []

    def get_performance_trends(
        self, test_name: Optional[str] = None, test_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get performance trend data for tests."""
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM test_performance"
                params: List[Any] = []

                if test_name or test_file:
                    conditions = []
                    if test_name:
                        conditions.append("test_name = ?")
                        params.append(test_name)
                    if test_file:
                        conditions.append("test_file = ?")
                        params.append(test_file)
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY success_rate ASC, avg_duration DESC"

                cursor = conn.execute(query, params)
                results = []

                for row in cursor.fetchall():
                    result = dict(row)
                    if result["trend_data"]:
                        result["trend_data"] = json.loads(result["trend_data"])
                    results.append(result)

                return results

        except Exception as e:
            self.logger.error(f"Failed to get performance trends: {e}")
            return []

    def get_batch_summary(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch execution summary."""
        try:
            with self.get_connection() as conn:
                # Get batch record
                cursor = conn.execute(
                    """
                    SELECT * FROM batch_executions WHERE batch_id = ?
                """,
                    (batch_id,),
                )
                batch_record = cursor.fetchone()

                if not batch_record:
                    return None

                # Get test records for this batch
                cursor = conn.execute(
                    """
                    SELECT * FROM test_executions WHERE batch_id = ? ORDER BY start_time
                """,
                    (batch_id,),
                )
                test_records = [dict(row) for row in cursor.fetchall()]

                result = dict(batch_record)
                if result["execution_metadata"]:
                    result["execution_metadata"] = json.loads(result["execution_metadata"])
                result["test_results"] = test_records

                return result

        except Exception as e:
            self.logger.error(f"Failed to get batch summary: {e}")
            return None

    def get_recent_batches(self, days: int = 7, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent batch executions."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT * FROM batch_executions 
                    WHERE start_time >= datetime('now', '-{} days')
                    ORDER BY start_time DESC 
                    LIMIT ?
                """.format(int(days)),  # nosec B608
                    (limit,),
                )

                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    if result["execution_metadata"]:
                        result["execution_metadata"] = json.loads(result["execution_metadata"])
                    results.append(result)

                return results

        except Exception as e:
            self.logger.error(f"Failed to get recent batches: {e}")
            return []

    def analyze_failure_patterns(self, days: int = 30) -> Dict[str, Any]:
        """Analyze failure patterns to identify problematic tests."""
        try:
            with self.get_connection() as conn:
                # Get failed tests
                cursor = conn.execute("""  # nosemgrep: sql-injection-risk - interval is int()-cast
                    SELECT test_name, test_file, COUNT(*) as failure_count,
                           AVG(duration) as avg_duration,
                           GROUP_CONCAT(SUBSTR(error_message, 1, 100), ' | ') as error_patterns
                    FROM test_executions 
                    WHERE status IN ('failed', 'error') 
                    AND start_time >= datetime('now', '-{} days')
                    GROUP BY test_name, test_file
                    ORDER BY failure_count DESC
                """.format(int(days)))  # nosec B608

                failed_tests = [dict(row) for row in cursor.fetchall()]

                # Get failure rate by time of day
                cursor = conn.execute("""  # nosemgrep: sql-injection-risk - interval is int()-cast
                    SELECT strftime('%H', start_time) as hour,
                           COUNT(*) as total_executions,
                           SUM(CASE WHEN status IN ('failed', 'error') THEN 1 ELSE 0 END) as failures
                    FROM test_executions 
                    WHERE start_time >= datetime('now', '-{} days')
                    GROUP BY hour
                    ORDER BY hour
                """.format(int(days)))  # nosec B608

                failure_by_hour = [dict(row) for row in cursor.fetchall()]

                # Get most common error messages
                cursor = conn.execute("""  # nosemgrep: sql-injection-risk - interval is int()-cast
                    SELECT SUBSTR(error_message, 1, 200) as error_sample, COUNT(*) as count
                    FROM test_executions 
                    WHERE status IN ('failed', 'error') 
                    AND start_time >= datetime('now', '-{} days')
                    AND error_message IS NOT NULL
                    GROUP BY SUBSTR(error_message, 1, 100)
                    ORDER BY count DESC
                    LIMIT 10
                """.format(int(days)))  # nosec B608

                common_errors = [dict(row) for row in cursor.fetchall()]

                return {
                    "failed_tests": failed_tests,
                    "failure_by_hour": failure_by_hour,
                    "common_errors": common_errors,
                    "analysis_period_days": days,
                }

        except Exception as e:
            self.logger.error(f"Failed to analyze failure patterns: {e}")
            return {}

    def generate_performance_report(self, days: int = 30) -> str:
        """Generate a comprehensive performance report."""
        try:
            # Get various metrics
            recent_batches = self.get_recent_batches(days)
            performance_trends = self.get_performance_trends()
            failure_patterns = self.analyze_failure_patterns(days)

            # Calculate summary statistics
            total_batches = len(recent_batches)
            total_tests = sum(b["total_tests"] for b in recent_batches)
            total_passed = sum(b["passed"] for b in recent_batches)
            total_failed = sum(b["failed"] for b in recent_batches)
            total_errors = sum(b["errors"] for b in recent_batches)

            overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

            # Generate report
            report = f"""
# APGI Framework Test Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Period: Last {days} days

## Executive Summary
- Total Batch Executions: {total_batches}
- Total Tests Executed: {total_tests}
- Overall Success Rate: {overall_success_rate:.1f}%
- Passed: {total_passed}
- Failed: {total_failed}
- Errors: {total_errors}

## Performance Trends
"""

            # Add top performing tests
            if performance_trends:
                report += "\n### Best Performing Tests\n"
                for test in performance_trends[:5]:
                    report += f"- {test['test_name']}: {test['success_rate']:.1%} success rate, {test['avg_duration']:.2f}s avg duration\n"

                report += "\n### Tests Needing Attention\n"
                for test in performance_trends[-5:]:
                    if test["success_rate"] < 0.8:
                        report += f"- {test['test_name']}: {test['success_rate']:.1%} success rate, {test['avg_duration']:.2f}s avg duration\n"

            # Add failure patterns
            if failure_patterns.get("failed_tests"):
                report += "\n## Failure Patterns\n"
                report += "### Most Frequently Failing Tests\n"
                for test in failure_patterns["failed_tests"][:5]:
                    report += f"- {test['test_name']}: {test['failure_count']} failures\n"
                    if test["error_patterns"]:
                        report += f"  Common errors: {test['error_patterns'][:200]}...\n"

            return report

        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return f"Error generating report: {e}"

    def cleanup_old_records(self, days_to_keep: int = 90) -> Tuple[int, int]:
        """Clean up old test execution records."""
        self._log_operation("DELETE", f"Cleaning up records older than {days_to_keep} days")
        try:
            with self.get_connection() as conn:
                # Delete old batch executions
                cursor = conn.execute("""  # nosemgrep: sql-injection-risk - interval is int()-cast
                    DELETE FROM batch_executions 
                    WHERE start_time < datetime('now', '-{} days')
                """.format(int(days_to_keep)))  # nosec B608
                deleted_batches = cursor.rowcount

                # Delete old test executions
                cursor = conn.execute("""  # nosemgrep: sql-injection-risk - interval is int()-cast
                    DELETE FROM test_executions 
                    WHERE start_time < datetime('now', '-{} days')
                """.format(int(days_to_keep)))  # nosec B608
                deleted_tests = cursor.rowcount

                conn.commit()

                self.logger.info(
                    f"Cleaned up {deleted_batches} batch records and {deleted_tests} test records older than {days_to_keep} days"
                )
                return deleted_batches, deleted_tests

        except Exception as e:
            self.logger.error(f"Failed to cleanup old records: {e}")
            return 0, 0

    def export_results(self, output_path: str, format: str = "json", days: int = 30) -> None:
        """Export test results in various formats."""
        try:
            # Get recent data
            recent_batches = self.get_recent_batches(days)

            if format.lower() == "json":
                with open(output_path, "w") as f:
                    json.dump(recent_batches, f, indent=2, default=str)

            elif format.lower() == "csv":
                import csv

                with open(output_path, "w", newline="") as f:
                    if recent_batches:
                        writer = csv.DictWriter(f, fieldnames=recent_batches[0].keys())
                        writer.writeheader()
                        for batch in recent_batches:
                            # Convert complex fields to strings
                            flat_batch = {k: str(v) for k, v in batch.items()}
                            writer.writerow(flat_batch)

            else:
                raise ValueError(f"Unsupported export format: {format}")

            self.logger.info(f"Exported {len(recent_batches)} batch results to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            raise


# Convenience functions
def get_persistence_manager(db_path: Optional[str] = None) -> TestResultPersistence:
    """Get a test result persistence manager instance."""
    return TestResultPersistence(db_path)


def store_test_results(batch_summary: Any, batch_id: Optional[str] = None) -> str:
    """Store test results in the persistence database."""
    persistence = TestResultPersistence()
    return persistence.store_batch_results(batch_summary, batch_id)


def get_test_performance_report(days: int = 30) -> str:
    """Get a performance report for the last N days."""
    persistence = TestResultPersistence()
    return persistence.generate_performance_report(days)
