"""CLI Test Runner for APGI Framework Test Enhancement.

This module provides command-line interface for test execution, coverage analysis,
and reporting with full feature parity to the GUI interface.

Requirements: 9.6
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from apgi_framework.logging.standardized_logging import get_logger
from apgi_framework.testing.activity_logger import (
    ActivityLevel,
    ActivityType,
    get_activity_logger,
)

logger = get_logger(__name__)


class CLITestRunner:
    """Command-line interface for test execution and management."""

    def __init__(self, container: Any, config: Any) -> None:
        """Initialize CLI test runner with dependency container and configuration."""
        self.container = container
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.activity_logger = get_activity_logger()

        # Get components from container
        self.batch_runner = container.get_batch_runner()
        self.ci_integrator = container.get_ci_integrator()
        self.notification_manager = container.get_notification_manager()

        self._start_time: Optional[datetime] = None
        self._test_results: Dict[str, Any] = {}

    def run_all_tests(self) -> int:
        """Run all tests in the project."""
        self.logger.info("Running all tests...")
        self._start_time = datetime.now()  # type: ignore[assignment]

        try:
            self.activity_logger.log_activity(
                ActivityType.TEST_EXECUTION_START,
                ActivityLevel.INFO,
                "Starting full test suite execution",
                data={"mode": "cli", "test_type": "all"},
            )

            # Discover all test files
            test_files = self._discover_test_files()
            self.logger.info(f"Discovered {len(test_files)} test files")

            # Execute tests
            results = self.batch_runner.run_tests(
                test_files=test_files,
                parallel=self.config.parallel_execution,
                max_workers=self.config.max_workers,
            )

            # Generate reports
            self._generate_reports(results, "all_tests")

            # Log completion
            self.activity_logger.log_activity(
                ActivityType.TEST_EXECUTION_END,
                ActivityLevel.INFO,
                "Full test suite execution completed",
                data={
                    "total_tests": results.get("total_tests", 0),
                    "passed": results.get("passed", 0),
                    "failed": results.get("failed", 0),
                    "duration": (
                        str(datetime.now() - self._start_time) if self._start_time else "unknown"
                    ),
                },
            )

            return 0 if results.get("failed", 0) == 0 else 1

        except Exception as e:
            self.logger.error(f"Error running all tests: {e}")
            self.activity_logger.log_activity(
                ActivityType.ERROR_OCCURRED,
                ActivityLevel.ERROR,
                f"Test execution failed: {str(e)}",
                data={"error_type": type(e).__name__},
            )
            return 1

    def run_unit_tests(self) -> int:
        """Run unit tests only."""
        self.logger.info("Running unit tests...")
        self._start_time = datetime.now()  # type: ignore[assignment]

        try:
            self.activity_logger.log_activity(
                ActivityType.TEST_EXECUTION_START,
                ActivityLevel.INFO,
                "Starting unit test execution",
                data={"mode": "cli", "test_type": "unit"},
            )

            # Discover unit test files
            test_files = self._discover_test_files(pattern="**/test_*.py", exclude_integration=True)
            self.logger.info(f"Discovered {len(test_files)} unit test files")

            # Execute tests
            results = self.batch_runner.run_tests(
                test_files=test_files,
                parallel=self.config.parallel_execution,
                max_workers=self.config.max_workers,
            )

            # Generate reports
            self._generate_reports(results, "unit_tests")

            return 0 if results.get("failed", 0) == 0 else 1

        except Exception as e:
            self.logger.error(f"Error running unit tests: {e}")
            return 1

    def run_integration_tests(self) -> int:
        """Run integration tests only."""
        self.logger.info("Running integration tests...")
        self._start_time = datetime.now()  # type: ignore[assignment]

        try:
            self.activity_logger.log_activity(
                ActivityType.TEST_EXECUTION_START,
                ActivityLevel.INFO,
                "Starting integration test execution",
                data={"mode": "cli", "test_type": "integration"},
            )

            # Discover integration test files
            test_files = self._discover_test_files(pattern="**/test_integration_*.py")
            integration_dir = Path(self.config.project_root) / "tests" / "integration"
            if integration_dir.exists():
                integration_files = list(integration_dir.glob("test_*.py"))
                test_files.extend(integration_files)

            self.logger.info(f"Discovered {len(test_files)} integration test files")

            # Execute tests
            results = self.batch_runner.run_tests(
                test_files=test_files,
                parallel=self.config.parallel_execution,
                max_workers=self.config.max_workers,
            )

            # Generate reports
            self._generate_reports(results, "integration_tests")

            return 0 if results.get("failed", 0) == 0 else 1

        except Exception as e:
            self.logger.error(f"Error running integration tests: {e}")
            return 1

    def run_tests_by_pattern(self, pattern: str) -> int:
        """Run tests matching a specific pattern."""
        self.logger.info(f"Running tests matching pattern: {pattern}")
        self._start_time = datetime.now()  # type: ignore[assignment]

        try:
            self.activity_logger.log_activity(
                ActivityType.TEST_EXECUTION_START,
                ActivityLevel.INFO,
                f"Starting pattern-based test execution: {pattern}",
                data={"mode": "cli", "test_type": "pattern", "pattern": pattern},
            )

            # Discover test files matching pattern
            test_files = self._discover_test_files(pattern=pattern)
            self.logger.info(f"Discovered {len(test_files)} test files matching pattern")

            if not test_files:
                self.logger.warning(f"No test files found matching pattern: {pattern}")
                return 0

            # Execute tests
            results = self.batch_runner.run_tests(
                test_files=test_files,
                parallel=self.config.parallel_execution,
                max_workers=self.config.max_workers,
            )

            # Generate reports
            self._generate_reports(results, f"pattern_{pattern.replace('*', 'star')}")

            return 0 if results.get("failed", 0) == 0 else 1

        except Exception as e:
            self.logger.error(f"Error running tests by pattern: {e}")
            return 1

    def generate_coverage_report(self) -> int:
        """Generate coverage report without running tests."""
        self.logger.info("Generating coverage report...")

        try:
            self.activity_logger.log_activity(
                ActivityType.COVERAGE_COLLECTION_START,
                ActivityLevel.INFO,
                "Starting coverage report generation",
                data={"mode": "cli"},
            )

            # Use CI integrator to generate coverage report
            coverage_report = self.ci_integrator.generate_coverage_report()

            if coverage_report:
                # Save report to file
                report_file = (
                    Path(self.config.project_root)
                    / "coverage_reports"
                    / f"coverage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                report_file.parent.mkdir(parents=True, exist_ok=True)

                with open(report_file, "w") as f:
                    json.dump(coverage_report, f, indent=2)

                self.logger.info(f"Coverage report saved to: {report_file}")

                # Print summary to console
                self._print_coverage_summary(coverage_report)

                return 0
            else:
                self.logger.error("Failed to generate coverage report")
                return 1

        except Exception as e:
            self.logger.error(f"Error generating coverage report: {e}")
            return 1

    def run_interactive(self) -> int:
        """Run interactive CLI mode."""
        self.logger.info("Starting interactive CLI mode...")

        logger.info("\n" + "=" * 60)
        logger.info("APGI Framework Test Enhancement - Interactive CLI")
        logger.info("=" * 60)

        while True:
            try:
                logger.info("\nAvailable commands:")
                logger.info("1. Run all tests")
                logger.info("2. Run unit tests")
                logger.info("3. Run integration tests")
                logger.info("4. Run tests by pattern")
                logger.info("5. Generate coverage report")
                logger.info("6. View test history")
                logger.info("7. Exit")

                choice = input("\nEnter your choice (1-7): ").strip()

                if choice == "1":
                    result = self.run_all_tests()
                    logger.info(f"\nTest execution completed with exit code: {result}")
                elif choice == "2":
                    result = self.run_unit_tests()
                    logger.info(f"\nUnit test execution completed with exit code: {result}")
                elif choice == "3":
                    result = self.run_integration_tests()
                    logger.info(f"\nIntegration test execution completed with exit code: {result}")
                elif choice == "4":
                    pattern = input("Enter test pattern (e.g., 'test_core_*.py'): ").strip()
                    if pattern:
                        result = self.run_tests_by_pattern(pattern)
                        logger.info(f"\nPattern test execution completed with exit code: {result}")
                elif choice == "5":
                    result = self.generate_coverage_report()
                    logger.info(f"\nCoverage report generation completed with exit code: {result}")
                elif choice == "6":
                    self._show_test_history()
                elif choice == "7":
                    logger.info("Exiting interactive mode...")
                    break
                else:
                    logger.info("Invalid choice. Please enter a number between 1-7.")

            except KeyboardInterrupt:
                logger.info("\n\nExiting interactive mode...")
                break
            except Exception as e:
                logger.info(f"Error: {e}")

        return 0

    def _discover_test_files(
        self, pattern: str = "**/test_*.py", exclude_integration: bool = False
    ) -> List[Path]:
        """Discover test files matching the given pattern."""
        project_root = Path(self.config.project_root)
        test_files: List[Path] = []

        # Search in tests directory
        tests_dir = project_root / "tests"
        if tests_dir.exists():
            test_files.extend(tests_dir.glob(pattern))

        # Search in apgi_framework directory for embedded tests
        framework_dir = project_root / "apgi_framework"
        if framework_dir.exists():
            test_files.extend(framework_dir.glob(pattern))

        # Filter out integration tests if requested
        if exclude_integration:
            test_files = [
                f
                for f in test_files
                if "integration" not in str(f) and "test_integration" not in f.name
            ]

        return sorted(test_files)

    def _generate_reports(self, results: Dict[str, Any], report_name: str) -> None:
        """Generate test execution reports."""
        try:
            # Create reports directory
            reports_dir = Path(self.config.project_root) / "test_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Generate JSON report
            json_report = {
                "execution_info": {
                    "timestamp": datetime.now().isoformat(),
                    "duration": (
                        str(datetime.now() - self._start_time) if self._start_time else "unknown"
                    ),
                    "mode": "cli",
                    "report_name": report_name,
                },
                "results": results,
                "configuration": {
                    "parallel_execution": self.config.parallel_execution,
                    "max_workers": self.config.max_workers,
                    "coverage_threshold": self.config.coverage_threshold,
                },
            }

            json_file = (
                reports_dir / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(json_file, "w") as f:
                json.dump(json_report, f, indent=2)

            self.logger.info(f"Test report saved to: {json_file}")

            # Print summary to console
            self._print_test_summary(results)

        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")

    def _print_test_summary(self, results: Dict[str, Any]) -> None:
        """Print test execution summary to console."""
        logger.info("\n" + "=" * 50)
        logger.info("TEST EXECUTION SUMMARY")
        logger.info("=" * 50)

        total = results.get("total_tests", 0)
        passed = results.get("passed", 0)
        failed = results.get("failed", 0)
        skipped = results.get("skipped", 0)

        logger.info(f"Total Tests:  {total}")
        logger.info(f"Passed:       {passed}")
        logger.info(f"Failed:       {failed}")
        logger.info(f"Skipped:      {skipped}")

        if total > 0:
            pass_rate = (passed / total) * 100
            logger.info(f"Pass Rate:    {pass_rate:.1f}%")

        if self._start_time:
            duration = datetime.now() - self._start_time
            logger.info(f"Duration:     {duration}")

        # Show coverage if available
        coverage = results.get("coverage", {})
        if coverage:
            line_coverage = coverage.get("line_coverage", 0)
            branch_coverage = coverage.get("branch_coverage", 0)
            logger.info("\nCoverage:")
            logger.info(f"Lines:        {line_coverage:.1f}%")
            logger.info(f"Branches:     {branch_coverage:.1f}%")

        logger.info("=" * 50)

        # Show failed tests if any
        failed_tests = results.get("failed_tests", [])
        if failed_tests:
            logger.info(f"\nFAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests[:10]:  # Show first 10 failures
                logger.info(
                    f"  - {test.get('name', 'Unknown')}: {test.get('error', 'No error message')}"
                )

            if len(failed_tests) > 10:
                logger.info(f"  ... and {len(failed_tests) - 10} more failures")

    def _print_coverage_summary(self, coverage_report: Dict[str, Any]) -> None:
        """Print coverage summary to console."""
        logger.info("\n" + "=" * 50)
        logger.info("COVERAGE REPORT SUMMARY")
        logger.info("=" * 50)

        overall = coverage_report.get("overall", {})
        line_coverage = overall.get("line_coverage", 0)
        branch_coverage = overall.get("branch_coverage", 0)
        function_coverage = overall.get("function_coverage", 0)

        logger.info("Overall Coverage:")
        logger.info(f"Lines:        {line_coverage:.1f}%")
        logger.info(f"Branches:     {branch_coverage:.1f}%")
        logger.info(f"Functions:    {function_coverage:.1f}%")

        # Show module breakdown
        modules = coverage_report.get("modules", {})
        if modules:
            logger.info("Module Coverage:")
            for module, data in sorted(modules.items()):
                module_coverage = data.get("line_coverage", 0)
                status = "✓" if module_coverage >= self.config.coverage_threshold * 100 else "✗"
                logger.info(f"  {status} {module:<30} {module_coverage:.1f}%")

        logger.info("=" * 50)

    def _show_test_history(self) -> None:
        """Show recent test execution history."""
        try:
            reports_dir = Path(self.config.project_root) / "test_reports"
            if not reports_dir.exists():
                logger.info("No test history available.")
                return

            # Get recent report files
            report_files = sorted(
                reports_dir.glob("*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

            if not report_files:
                logger.info("No test history available.")
                return

            logger.info("\n" + "=" * 60)
            logger.info("RECENT TEST EXECUTIONS")
            logger.info("=" * 60)

            for i, report_file in enumerate(report_files[:10]):  # Show last 10 executions
                try:
                    with open(report_file, "r") as f:
                        report = json.load(f)

                    exec_info = report.get("execution_info", {})
                    results = report.get("results", {})

                    timestamp = exec_info.get("timestamp", "Unknown")
                    duration = exec_info.get("duration", "Unknown")
                    report_name = exec_info.get("report_name", "Unknown")

                    total = results.get("total_tests", 0)
                    passed = results.get("passed", 0)
                    failed = results.get("failed", 0)

                    status = "PASS" if failed == 0 else "FAIL"

                    logger.info(
                        f"{i + 1:2d}. {timestamp[:19]} | {report_name:<15} | {status:<4} | {passed}/{total} | {duration}"
                    )

                except Exception as e:
                    logger.info(f"{i + 1:2d}. {report_file.name} - Error reading report: {e}")

            logger.info("=" * 60)

        except Exception as e:
            logger.info(f"Error showing test history: {e}")

    def cleanup(self) -> None:
        """Cleanup CLI runner resources."""
        try:
            self.logger.info("Cleaning up CLI runner...")

            # Log final activity
            self.activity_logger.log_activity(
                ActivityType.SYSTEM_SHUTDOWN,
                ActivityLevel.INFO,
                "CLI runner cleanup completed",
                data={"mode": "cli"},
            )

        except Exception as e:
            self.logger.warning(f"Error during CLI cleanup: {e}")
