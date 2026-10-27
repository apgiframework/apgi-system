"""Batch Test Execution System for APGI Framework.

This module provides comprehensive batch test execution capabilities including:
- Parallel test execution
- Test result aggregation
- Progress monitoring
- Error handling and recovery
- Performance metrics collection
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from ..config import ConfigManager
from ..logging.standardized_logging import get_logger
from .activity_logger import (
    ActivityType,
    activity_span,
    get_activity_logger,
    log_test_case_end,
    log_test_case_start,
    log_test_execution_end,
    log_test_execution_start,
)

logger = get_logger(__name__)


@dataclass
class TestResult:
    """Individual test result data structure."""

    test_name: str
    test_file: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float
    output: str
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class BatchExecutionSummary:
    """Summary of batch test execution."""

    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    total_duration: float
    start_time: datetime
    end_time: datetime
    test_results: List[TestResult]
    execution_metadata: Dict[str, Any]


class BatchTestRunner:
    """Advanced batch test execution system."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize batch test runner."""
        self.config_manager = config_manager or ConfigManager()
        self.logger = get_logger(f"{__name__}.BatchTestRunner")
        self._stop_execution = False
        self._progress_callback: Optional[Callable[[float, Any], None]] = None

    def set_progress_callback(self, callback: Callable[[float, Any], None]) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def stop_execution(self) -> None:
        """Stop ongoing test execution."""
        self._stop_execution = True
        self.logger.info("Test execution stop requested")

    def discover_tests(
        self,
        test_paths: Optional[List[str]] = None,
        markers: Optional[List[str]] = None,
        keywords: Optional[str] = None,
    ) -> List[str]:
        """Discover available tests using pytest."""
        try:
            # Build pytest arguments
            pytest_args = ["--collect-only", "--quiet"]

            if test_paths:
                pytest_args.extend(test_paths)

            if markers:
                for marker in markers:
                    pytest_args.extend(["-m", marker])

            if keywords:
                pytest_args.extend(["-k", keywords])

            # Run pytest collection
            result = pytest.main(pytest_args)

            if result != 0:
                self.logger.warning(f"Test collection returned non-zero exit code: {result}")

            # Parse collected tests from pytest output
            collected_tests = self._parse_pytest_collection_output()

            self.logger.info(f"Discovered {len(collected_tests)} tests")
            return collected_tests

        except Exception as e:
            self.logger.error(f"Error during test discovery: {e}")
            return []

    def _parse_pytest_collection_output(self) -> List[str]:
        """Parse pytest collection output to extract test names."""
        # This is a simplified implementation
        # In practice, we'd need to capture and parse pytest's JSON output
        test_files = []

        # Default test directories
        test_dirs = ["tests", "research/*/tests", "apgi_framework/tests"]

        for test_dir in test_dirs:
            if Path(test_dir).exists():
                for test_file in Path(test_dir).rglob("test_*.py"):
                    test_files.append(str(test_file))

        return test_files

    def run_batch_tests(
        self,
        test_selection: Optional[List[str]] = None,
        test_paths: Optional[List[str]] = None,
        markers: Optional[List[str]] = None,
        keywords: Optional[str] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        timeout: Optional[int] = None,
        failfast: bool = False,
    ) -> BatchExecutionSummary:
        """Run batch tests with specified parameters."""
        start_time = datetime.now()
        execution_id = f"batch_{start_time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

        # Set up activity logging context
        activity_logger = get_activity_logger()
        activity_logger.set_context(
            execution_id=execution_id,
            test_suite="batch_execution",
            component="batch_runner",
        )

        self.logger.info(f"Starting batch test execution at {start_time}")

        # Discover tests if not explicitly provided
        if test_selection is None:
            with activity_span(ActivityType.TEST_DISCOVERY, "Discovering tests"):
                test_files = self.discover_tests(test_paths, markers, keywords)
        else:
            test_files = test_selection

        if not test_files:
            self.logger.warning("No tests found to execute")
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

        # Configure execution parameters
        max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        timeout = timeout or 300  # 5 minutes default per test

        # Log test execution start
        log_test_execution_start(
            execution_id=execution_id,
            test_suite="batch_execution",
            total_tests=len(test_files),
            configuration={
                "parallel": parallel,
                "max_workers": max_workers,
                "timeout": timeout,
                "failfast": failfast,
                "test_paths": test_paths,
                "markers": markers,
                "keywords": keywords,
            },
        )

        self.logger.info(f"Executing {len(test_files)} tests with {max_workers} workers")

        # Execute tests
        test_results = []

        try:
            if parallel and len(test_files) > 1:
                test_results = self._run_tests_parallel(test_files, max_workers, timeout, failfast)
            else:
                test_results = self._run_tests_sequential(test_files, timeout, failfast)

            # Generate summary
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
                    "execution_id": execution_id,
                    "parallel": parallel,
                    "max_workers": max_workers,
                    "timeout": timeout,
                    "failfast": failfast,
                    "test_paths": test_paths,
                    "markers": markers,
                    "keywords": keywords,
                },
            )

            # Log test execution end
            log_test_execution_end(
                execution_id=execution_id,
                test_suite="batch_execution",
                results={
                    "total_tests": summary.total_tests,
                    "passed": summary.passed,
                    "failed": summary.failed,
                    "skipped": summary.skipped,
                    "errors": summary.errors,
                },
                duration_ms=total_duration * 1000,
            )

            self.logger.info(
                f"Batch execution completed in {total_duration:.2f}s: "
                f"{summary.passed} passed, {summary.failed} failed, "
                f"{summary.skipped} skipped, {summary.errors} errors"
            )

            return summary

        except Exception as e:
            # Log execution error
            activity_logger.log_error(
                "batch_runner",
                e,
                {"execution_id": execution_id, "test_files_count": len(test_files)},
            )
            raise

    def _run_tests_parallel(
        self, test_files: List[str], max_workers: int, timeout: int, failfast: bool
    ) -> List[TestResult]:
        """Run tests in parallel using ProcessPoolExecutor."""
        test_results = []
        completed_count = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all test tasks
            future_to_test = {
                executor.submit(self._run_single_test, test_file, timeout): test_file
                for test_file in test_files
            }

            # Process completed tests
            for future in as_completed(future_to_test):
                if self._stop_execution:
                    break

                test_file = future_to_test[future]
                completed_count += 1

                try:
                    result = future.result(timeout=timeout + 60)  # Extra time for overhead
                    test_results.append(result)

                    # Report progress
                    if self._progress_callback:
                        progress = completed_count / len(test_files)
                        self._progress_callback(progress, result)

                    # Stop on failure if failfast is enabled
                    if failfast and result.status in ["failed", "error"]:
                        self.logger.warning(f"Stopping execution due to failure in {test_file}")
                        break

                except Exception as e:
                    self.logger.error(f"Error executing test {test_file}: {e}")
                    error_result = TestResult(
                        test_name=test_file,
                        test_file=test_file,
                        status="error",
                        duration=0.0,
                        output="",
                        error_message=str(e),
                    )
                    test_results.append(error_result)

        return test_results

    def _run_tests_sequential(
        self, test_files: List[str], timeout: int, failfast: bool
    ) -> List[TestResult]:
        """Run tests sequentially."""
        test_results = []

        for i, test_file in enumerate(test_files):
            if self._stop_execution:
                break

            self.logger.info(f"Running test {i + 1}/{len(test_files)}: {test_file}")

            result = self._run_single_test(test_file, timeout)
            test_results.append(result)

            # Report progress
            if self._progress_callback:
                progress = (i + 1) / len(test_files)
                self._progress_callback(progress, result)

            # Stop on failure if failfast is enabled
            if failfast and result.status in ["failed", "error"]:
                self.logger.warning(f"Stopping execution due to failure in {test_file}")
                break

        return test_results

    def _run_single_test(self, test_file: str, timeout: int) -> TestResult:
        """Run a single test file and capture results."""
        start_time = datetime.now()

        # Log test case start
        log_test_case_start(test_file, test_file)

        try:
            # Prepare pytest command
            pytest_cmd = [
                sys.executable,
                "-m",
                "pytest",
                test_file,
                "--tb=short",
                "-v",
                "--color=no",
            ]

            # Run pytest with timeout
            result = subprocess.run(
                pytest_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd(),
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Parse test results from output
            test_result = self._parse_pytest_output(
                test_file, result, start_time, end_time, duration
            )

            # Log test case end
            log_test_case_end(
                test_name=test_file,
                test_file=test_file,
                status=test_result.status,
                duration_ms=duration * 1000,
                error_message=test_result.error_message,
            )

            return test_result

        except subprocess.TimeoutExpired:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            error_msg = f"Test timed out after {timeout} seconds"

            # Log test case end with timeout
            log_test_case_end(
                test_name=test_file,
                test_file=test_file,
                status="error",
                duration_ms=duration * 1000,
                error_message=error_msg,
            )

            return TestResult(
                test_name=test_file,
                test_file=test_file,
                status="error",
                duration=duration,
                output="",
                error_message=error_msg,
                start_time=start_time,
                end_time=end_time,
            )

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            error_msg = str(e)

            # Log test case end with error
            log_test_case_end(
                test_name=test_file,
                test_file=test_file,
                status="error",
                duration_ms=duration * 1000,
                error_message=error_msg,
            )

            return TestResult(
                test_name=test_file,
                test_file=test_file,
                status="error",
                duration=duration,
                output="",
                error_message=error_msg,
                start_time=start_time,
                end_time=end_time,
            )

    def _parse_pytest_output(
        self,
        test_file: str,
        subprocess_result: subprocess.CompletedProcess,
        start_time: datetime,
        end_time: datetime,
        duration: float,
    ) -> TestResult:
        """Parse pytest output into Result."""
        output = subprocess_result.stdout
        error_output = subprocess_result.stderr

        # Parse status from return code and output
        status = "error"
        error_message = None
        traceback = None

        if subprocess_result.returncode == 0:
            status = "passed"
        elif subprocess_result.returncode == 1:
            status = "failed"
        elif subprocess_result.returncode == 2:
            status = "skipped"  # No tests collected or user interrupted
        elif subprocess_result.returncode == 3:
            status = "error"  # Internal error
        elif subprocess_result.returncode == 4:
            status = "error"  # Usage error
        elif subprocess_result.returncode == 5:
            status = "skipped"  # No tests collected

        # Extract error information from output
        if status in ["failed", "error"]:
            # Look for failure summary in output
            lines = output.split("\n")
            failure_section = False
            error_lines = []

            for line in lines:
                if "FAILED" in line or "ERROR" in line:
                    failure_section = True
                elif failure_section and line.strip():
                    if line.startswith("=") or line.startswith("---"):
                        continue
                    error_lines.append(line.strip())
                elif failure_section and line.startswith("==="):
                    break

            if error_lines:
                error_message = "\n".join(error_lines[:5])  # First 5 lines
                traceback = "\n".join(error_lines)

        # If no error message found, use stderr
        if not error_message and error_output:
            error_message = error_output.strip()
            traceback = error_output

        return TestResult(
            test_name=test_file,
            test_file=test_file,
            status=status,
            duration=duration,
            output=output,
            error_message=error_message,
            traceback=traceback,
            start_time=start_time,
            end_time=end_time,
        )

    def generate_report(
        self, summary: BatchExecutionSummary, output_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive test execution report."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"batch_test_report_{timestamp}.html"

        # Generate HTML report
        html_content = self._generate_html_report(summary)

        # Save report
        with open(output_path, "w") as f:
            f.write(html_content)

        self.logger.info(f"Test report generated: {output_path}")
        return output_path

    def _generate_html_report(self, summary: BatchExecutionSummary) -> str:
        """Generate HTML report content."""
        # Calculate statistics
        success_rate = (
            (summary.passed / summary.total_tests * 100) if summary.total_tests > 0 else 0
        )

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>APGI Batch Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .test-result {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }}
        .passed {{ background-color: #d4edda; }}
        .failed {{ background-color: #f8d7da; }}
        .skipped {{ background-color: #fff3cd; }}
        .error {{ background-color: #f5c6cb; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-box {{ padding: 10px; border: 1px solid #ddd; border-radius: 3px; text-align: center; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>APGI Framework Batch Test Report</h1>
        <p>Generated: {summary.end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Execution Duration: {summary.total_duration:.2f} seconds</p>
    </div>
    
    <div class="summary">
        <h2>Execution Summary</h2>
        <div class="stats">
            <div class="stat-box">
                <h3>{summary.total_tests}</h3>
                <p>Total Tests</p>
            </div>
            <div class="stat-box passed">
                <h3>{summary.passed}</h3>
                <p>Passed</p>
            </div>
            <div class="stat-box failed">
                <h3>{summary.failed}</h3>
                <p>Failed</p>
            </div>
            <div class="stat-box skipped">
                <h3>{summary.skipped}</h3>
                <p>Skipped</p>
            </div>
            <div class="stat-box error">
                <h3>{summary.errors}</h3>
                <p>Errors</p>
            </div>
        </div>
        <p><strong>Success Rate:</strong> {success_rate:.1f}%</p>
    </div>
    
    <div class="details">
        <h2>Test Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Duration (s)</th>
                    <th>Start Time</th>
                    <th>Error Message</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add test results to table
        for result in summary.test_results:
            status_class = result.status
            error_msg = (
                result.error_message[:100] + "..."
                if result.error_message and len(result.error_message) > 100
                else (result.error_message or "")
            )

            html_template += f"""
                <tr class="{status_class}">
                    <td>{result.test_name}</td>
                    <td>{result.status.upper()}</td>
                    <td>{result.duration:.2f}</td>
                    <td>{result.start_time.strftime('%H:%M:%S') if result.start_time else ''}</td>
                    <td>{error_msg}</td>
                </tr>
"""

        html_template += (
            """
            </tbody>
        </table>
    </div>
    
    <div class="metadata">
        <h2>Execution Metadata</h2>
        <pre>
"""
            + json.dumps(summary.execution_metadata, indent=2)
            + """
        </pre>
    </div>
</body>
</html>
"""
        )

        return html_template


# Convenience functions for common batch operations
def run_all_tests(
    parallel: bool = True, max_workers: Optional[int] = None
) -> BatchExecutionSummary:
    """Run all discovered tests."""
    runner = BatchTestRunner()
    return runner.run_batch_tests(parallel=parallel, max_workers=max_workers)


def run_unit_tests(parallel: bool = True) -> BatchExecutionSummary:
    """Run only unit tests."""
    runner = BatchTestRunner()
    return runner.run_batch_tests(markers=["unit"], parallel=parallel)


def run_integration_tests(parallel: bool = True) -> BatchExecutionSummary:
    """Run only integration tests."""
    runner = BatchTestRunner()
    return runner.run_batch_tests(markers=["integration"], parallel=parallel)


def run_research_tests(parallel: bool = True) -> BatchExecutionSummary:
    """Run only research-specific tests."""
    runner = BatchTestRunner()
    return runner.run_batch_tests(markers=["research"], parallel=parallel)


def run_failed_tests(previous_results_file: str, parallel: bool = True) -> BatchExecutionSummary:
    """Re-run only failed tests from a previous execution."""
    # Load previous results
    with open(previous_results_file, "r") as f:
        previous_data = json.load(f)

    # Extract failed test files
    failed_tests = [
        result["test_file"]
        for result in previous_data["test_results"]
        if result["status"] in ["failed", "error"]
    ]

    if not failed_tests:
        logger.info("No failed tests to re-run")
        return BatchExecutionSummary(
            total_tests=0,
            passed=0,
            failed=0,
            skipped=0,
            errors=0,
            total_duration=0.0,
            start_time=datetime.now(),
            end_time=datetime.now(),
            test_results=[],
            execution_metadata={},
        )

    runner = BatchTestRunner()
    return runner.run_batch_tests(test_selection=failed_tests, parallel=parallel)
