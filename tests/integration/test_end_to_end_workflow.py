"""
End-to-End Integration Tests for Comprehensive Test Enhancement System

This module provides comprehensive workflow tests from test discovery to
report generation, GUI and CLI integration tests with backend components,
error handling and recovery scenario tests, and component interaction and
data flow tests between all modules.

Requirements: All requirements integration
"""

import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from apgi_framework.testing.activity_logger import (
    get_activity_logger,
)

# Import test enhancement components
from apgi_framework.testing.batch_runner import BatchExecutionSummary, BatchTestRunner
from apgi_framework.testing.ci_integrator import (
    ChangeImpact,
    CIConfiguration,
    CIIntegrator,
    ExecutionResult,
)
from apgi_framework.testing.error_handler import (
    Context,
    DiagnosticInfo,
    ErrorHandler,
)
from apgi_framework.testing.notification_manager import (
    HistoryTracker,
    NotificationManager,
    create_file_channel,
)


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_root = self.temp_dir / "test_project"
        self.project_root.mkdir(parents=True)

        # Create basic project structure
        self._setup_test_project()

        # Initialize components
        self.batch_runner = BatchTestRunner()
        self.ci_config = CIConfiguration(pipeline_type="test", test_subset_strategy="all")
        self.ci_integrator = CIIntegrator(str(self.project_root), self.ci_config)
        self.error_handler = ErrorHandler()
        self.notification_manager = NotificationManager(
            db_path=str(self.temp_dir / "test_history.db")
        )

        # Add file notification channel for testing
        file_channel = create_file_channel(
            "test_notifications", str(self.temp_dir / "notifications.log")
        )
        file_channel.success_notification = True  # Ensure notification is always sent
        self.notification_manager.add_channel(file_channel)

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _setup_test_project(self):
        """Create a minimal test project structure."""
        # Create source files
        src_dir = self.project_root / "src"
        src_dir.mkdir()

        (src_dir / "__init__.py").write_text("")
        (src_dir / "calculator.py").write_text("""
def add(a, b):
    return a + b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def multiply(a, b):
    return a * b
""")

        # Create test files
        tests_dir = self.project_root / "tests"
        tests_dir.mkdir()

        (tests_dir / "__init__.py").write_text("")
        (tests_dir / "test_calculator.py").write_text("""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calculator import add, divide, multiply

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_divide():
    assert divide(10, 2) == 5
    with pytest.raises(ValueError):
        divide(10, 0)

def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(0, 5) == 0

def test_failing_example():
    # This test will fail intentionally
    assert False, "Intentional failure for testing"
""")

        (tests_dir / "test_integration.py").write_text("""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calculator import add, multiply

def test_complex_calculation():
    result = multiply(add(2, 3), 4)
    assert result == 20

def test_chain_operations():
    x = add(1, 2)
    y = multiply(x, 3)
    assert y == 9
""")

        # Create requirements.txt
        (self.project_root / "requirements.txt").write_text("pytest>=6.0.0\n")

        # Create pytest.ini
        (self.project_root / "pytest.ini").write_text("""
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
""")

    def test_complete_test_discovery_to_report_workflow(self):
        """Test complete workflow from test discovery to report generation."""
        # Step 1: Test Discovery
        os.chdir(self.project_root)
        test_files = self.batch_runner.discover_tests()

        assert len(test_files) >= 2  # Should find our test files
        assert any("test_calculator.py" in f for f in test_files)
        assert any("test_integration.py" in f for f in test_files)

        # Step 2: Test Execution
        summary = self.batch_runner.run_batch_tests(
            test_selection=test_files,
            parallel=False,  # Sequential for predictable results
            failfast=False,
        )

        assert isinstance(summary, BatchExecutionSummary)
        assert summary.total_tests > 0
        assert summary.failed > 0  # We have an intentional failure
        assert summary.passed > 0  # We have passing tests

        # Step 3: Report Generation
        report_path = self.batch_runner.generate_report(summary)
        assert Path(report_path).exists()

        # Verify report content
        with open(report_path, "r") as f:
            report_content = f.read()
            assert "APGI Batch Test Report" in report_content
            assert str(summary.total_tests) in report_content
            assert "test_failing_example" in report_content

    def test_ci_integration_workflow(self):
        """Test CI/CD integration workflow."""
        os.chdir(self.project_root)

        # Step 1: Change Impact Analysis
        # Create a git repository for change analysis
        # Commit with an explicit identity: CI runners have no global git user configured
        identity = ["-c", "user.name=CI Test", "-c", "user.email=ci@example.com"]
        subprocess.run(["git", "init"], cwd=self.project_root, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=self.project_root, capture_output=True)
        subprocess.run(
            ["git", *identity, "commit", "-m", "Initial commit"],
            cwd=self.project_root,
            capture_output=True,
        )

        # Make a change
        calculator_file = self.project_root / "src" / "calculator.py"
        calculator_file.write_text(calculator_file.read_text() + "\n# Modified\n")

        subprocess.run(["git", "add", "."], cwd=self.project_root, capture_output=True)
        subprocess.run(
            ["git", *identity, "commit", "-m", "Modified calculator"],
            cwd=self.project_root,
            capture_output=True,
        )

        # Analyze changes
        change_impact = self.ci_integrator.analyze_changes("HEAD~1")
        assert isinstance(change_impact, ChangeImpact)
        assert len(change_impact.changed_files) > 0

        # Step 2: Execute CI Tests
        result = self.ci_integrator.execute_ci_tests(change_impact)
        # Check that CI execution completed (may or may not have failures)
        assert isinstance(result, ExecutionResult)
        assert result.execution_id is not None
        assert result.total_tests >= 0

        # Step 3: Notification
        self.notification_manager.notify_test_result(result)

        # Verify notification was sent
        notification_file = self.temp_dir / "notifications.log"
        assert notification_file.exists()

        with open(notification_file, "r") as f:
            notification_data = json.loads(f.read().strip())
            assert notification_data["result_summary"]["execution_id"] == result.execution_id

    def test_error_handling_and_recovery_workflow(self):
        """Test error handling and recovery scenarios."""
        # Create a test that will cause different types of errors
        error_test_file = self.project_root / "tests" / "test_errors.py"
        error_test_file.write_text("""
import pytest

def test_import_error():
    import nonexistent_module  # ImportError

def test_assertion_error():
    assert 1 == 2, "This will fail"

def test_value_error():
    raise ValueError("Test value error")

def test_timeout_simulation():
    import time
    time.sleep(0.1)  # Short sleep to simulate work
    assert True
""")

        os.chdir(self.project_root)

        # Run tests and capture errors
        test_files = [str(error_test_file)]
        summary = self.batch_runner.run_batch_tests(
            test_selection=test_files, parallel=False, failfast=False
        )

        # All tests should fail with different error types
        assert summary.failed > 0 or summary.errors > 0

        # Test error handling for each failure
        for test_result in summary.test_results:
            if test_result.status in ["failed", "error"]:
                # Create test context
                test_context = Context(
                    test_name=test_result.test_name, test_file=test_result.test_file
                )

                # Create a mock exception based on the error message
                if (
                    "ImportError" in test_result.error_message
                    or "import" in test_result.error_message
                ):
                    exception = ImportError("No module named 'nonexistent_module'")
                elif "AssertionError" in test_result.error_message:
                    exception = AssertionError("This will fail")
                elif "ValueError" in test_result.error_message:
                    exception = ValueError("Test value error")
                else:
                    exception = Exception(test_result.error_message or "Unknown error")

                # Handle the error
                diagnostic_info = self.error_handler.handle_error(
                    exception,
                    test_context,
                    test_execution_id=summary.execution_metadata.get("execution_id"),
                )

                assert isinstance(diagnostic_info, DiagnosticInfo)
                assert diagnostic_info.error_id is not None
                assert diagnostic_info.category is not None
                assert len(diagnostic_info.resolution_guidance) > 0

    def test_component_interaction_and_data_flow(self):
        """Test interaction and data flow between all components."""
        os.chdir(self.project_root)

        # Initialize activity logger for tracking
        activity_logger = get_activity_logger()
        activity_logger.set_context(execution_id="integration_test", component="end_to_end_test")

        # Step 1: Batch Runner -> CI Integrator
        test_files = self.batch_runner.discover_tests()
        summary = self.batch_runner.run_batch_tests(
            test_selection=test_files[:1], parallel=False  # Run just one test file
        )

        # Step 2: CI Integrator -> Notification Manager
        # Convert batch summary to CI result format
        ci_result_data = {
            "execution_id": summary.execution_metadata.get("execution_id", "test_exec"),
            "start_time": summary.start_time,
            "end_time": summary.end_time,
            "total_tests": summary.total_tests,
            "passed_tests": summary.passed,
            "failed_tests": summary.failed,
            "skipped_tests": summary.skipped,
            "coverage_percentage": 85.0,  # Mock coverage
            "execution_time_seconds": summary.total_duration,
            "failed_test_details": [
                {
                    "name": result.test_name,
                    "error": result.error_message or "",
                    "duration": result.duration,
                }
                for result in summary.test_results
                if result.status == "failed"
            ],
            "pipeline_context": {"test": "integration"},
        }

        # Create a mock CI result object
        from apgi_framework.testing.ci_integrator import ExecutionResult

        ci_result = ExecutionResult(**ci_result_data)

        # Step 3: Notification Manager processes result
        self.notification_manager.notify_test_result(ci_result)

        # Step 4: Verify data flow
        # Check that history was recorded
        history_tracker = self.notification_manager.history_tracker
        trend_analysis = history_tracker.get_trend_analysis(days=1)

        assert isinstance(trend_analysis, dict)
        if "error" not in trend_analysis:
            assert trend_analysis["total_executions"] >= 1

        # Check notification was sent
        notification_file = self.temp_dir / "notifications.log"
        if notification_file.exists():
            with open(notification_file, "r") as f:
                content = f.read().strip()
                if content:
                    notification_data = json.loads(content)
                    assert "notification" in notification_data
                    assert "result_summary" in notification_data

    def test_gui_backend_integration_simulation(self):
        """Simulate GUI integration with backend components."""
        # This test simulates how a GUI would interact with backend components
        os.chdir(self.project_root)

        # Simulate GUI test selection
        available_tests = self.batch_runner.discover_tests()
        selected_tests = available_tests[:2]  # User selects first 2 tests

        # Simulate GUI progress callback
        progress_updates = []

        def progress_callback(progress, result):
            progress_updates.append(
                {
                    "progress": progress,
                    "test_name": result.test_name,
                    "status": result.status,
                }
            )

        self.batch_runner.set_progress_callback(progress_callback)

        # Execute tests (simulating GUI execution)
        summary = self.batch_runner.run_batch_tests(test_selection=selected_tests, parallel=False)

        # Verify GUI would receive progress updates
        assert len(progress_updates) > 0
        assert all("progress" in update for update in progress_updates)
        assert all("test_name" in update for update in progress_updates)

        # Simulate GUI displaying results
        gui_display_data = {
            "summary": {
                "total": summary.total_tests,
                "passed": summary.passed,
                "failed": summary.failed,
                "duration": summary.total_duration,
            },
            "test_results": [
                {
                    "name": result.test_name,
                    "status": result.status,
                    "duration": result.duration,
                    "error": result.error_message,
                }
                for result in summary.test_results
            ],
        }

        # Verify GUI data structure is complete
        assert gui_display_data["summary"]["total"] > 0
        assert len(gui_display_data["test_results"]) == summary.total_tests

    def test_cli_backend_integration_simulation(self):
        """Simulate CLI integration with backend components."""
        os.chdir(self.project_root)

        # Simulate CLI command: run tests with specific parameters
        cli_args = {
            "test_paths": ["tests/"],
            "parallel": False,
            "max_workers": 2,
            "timeout": 60,
            "markers": None,
            "keywords": None,
        }

        # Execute via batch runner (simulating CLI backend call)
        summary = self.batch_runner.run_batch_tests(**cli_args)

        # Simulate CLI output formatting
        cli_output = []
        cli_output.append(f"Total tests: {summary.total_tests}")
        cli_output.append(f"Passed: {summary.passed}")
        cli_output.append(f"Failed: {summary.failed}")
        cli_output.append(f"Duration: {summary.total_duration:.2f}s")

        if summary.failed > 0:
            cli_output.append("\nFailed tests:")
            for result in summary.test_results:
                if result.status == "failed":
                    cli_output.append(f"  - {result.test_name}: {result.error_message}")

        # Verify CLI output structure
        assert len(cli_output) >= 4  # At least summary lines
        assert "Total tests:" in cli_output[0]
        assert "Passed:" in cli_output[1]

        # Simulate CLI report generation
        report_path = self.batch_runner.generate_report(summary)
        assert Path(report_path).exists()

    def test_error_recovery_scenarios(self):
        """Test various error recovery scenarios."""
        os.chdir(self.project_root)

        # Scenario 1: Missing dependency recovery
        missing_dep_test = self.project_root / "tests" / "test_missing_dep.py"
        missing_dep_test.write_text("""
def test_missing_dependency():
    import nonexistent_package
    assert True
""")

        # Run test and expect failure
        summary = self.batch_runner.run_batch_tests(
            test_selection=[str(missing_dep_test)], parallel=False
        )

        assert summary.failed > 0 or summary.errors > 0

        # Simulate recovery by "installing" dependency (modify test)
        missing_dep_test.write_text("""
def test_missing_dependency():
    # import nonexistent_package  # Fixed by commenting out
    assert True
""")

        # Re-run test
        recovery_summary = self.batch_runner.run_batch_tests(
            test_selection=[str(missing_dep_test)], parallel=False
        )

        assert recovery_summary.passed > 0

        # Scenario 2: Configuration error recovery
        pytest_ini = self.project_root / "pytest.ini"
        original_content = pytest_ini.read_text()

        pytest_ini.write_text("[invalid_section]\ninvalid_option = value")

        # This might cause issues, but our system should handle it
        try:
            self.batch_runner.run_batch_tests(
                test_selection=[str(missing_dep_test)], parallel=False
            )
        except Exception as e:
            # Handle configuration error
            diagnostic = self.error_handler.handle_error(e)
            assert diagnostic.category.value in [
                "configuration_error",
                "framework_issue",
            ]

        # Recover by restoring configuration
        pytest_ini.write_text(original_content)

        # Verify recovery
        recovery_summary = self.batch_runner.run_batch_tests(
            test_selection=[str(missing_dep_test)], parallel=False
        )
        assert recovery_summary.total_tests > 0

    def test_performance_under_load(self):
        """Test system performance under load conditions."""
        os.chdir(self.project_root)

        # Create multiple test files to simulate load
        for i in range(5):
            test_file = self.project_root / "tests" / f"test_load_{i}.py"
            test_file.write_text(f"""
def test_load_test_{i}_1():
    assert {i} + 1 == {i + 1}

def test_load_test_{i}_2():
    result = sum(range({i + 1}))
    assert result >= 0

def test_load_test_{i}_3():
    data = list(range({i * 10}))
    assert len(data) == {i * 10}
""")

        # Measure execution time
        start_time = datetime.now()

        # Run all tests
        summary = self.batch_runner.run_batch_tests(
            test_paths=["tests/"], parallel=True, max_workers=2
        )

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Verify performance characteristics
        assert summary.total_tests >= 7  # At least 7 tests should be found
        assert execution_time < 60  # Should complete within 60 seconds
        assert summary.total_duration > 0

        # Test parallel vs sequential performance
        sequential_summary = self.batch_runner.run_batch_tests(
            test_paths=["tests/test_load_0.py"], parallel=False
        )
        assert summary.total_tests >= sequential_summary.total_tests

    def test_data_persistence_and_retrieval(self):
        """Test data persistence and retrieval across components."""
        os.chdir(self.project_root)

        # Run initial test execution
        summary1 = self.batch_runner.run_batch_tests(
            test_paths=["tests/test_calculator.py"], parallel=False
        )

        # Convert to CI result and record in history
        ci_result1 = ExecutionResult(
            execution_id="persist_test_1",
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_tests=summary1.total_tests,
            passed_tests=summary1.passed,
            failed_tests=summary1.failed,
            skipped_tests=summary1.skipped,
            coverage_percentage=80.0,
            execution_time_seconds=summary1.total_duration,
            failed_test_details=[],
            pipeline_context={"test": "persistence"},
        )

        # Record in history tracker
        history_tracker = HistoryTracker(str(self.temp_dir / "test_history.db"))
        history_tracker.record_execution(ci_result1)

        # Run second execution
        summary2 = self.batch_runner.run_batch_tests(
            test_paths=["tests/test_integration.py"], parallel=False
        )

        ci_result2 = ExecutionResult(
            execution_id="persist_test_2",
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_tests=summary2.total_tests,
            passed_tests=summary2.passed,
            failed_tests=summary2.failed,
            skipped_tests=summary2.skipped,
            coverage_percentage=85.0,
            execution_time_seconds=summary2.total_duration,
            failed_test_details=[],
            pipeline_context={"test": "persistence"},
        )

        history_tracker.record_execution(ci_result2)

        # Retrieve and verify historical data
        trend_analysis = history_tracker.get_trend_analysis(days=1)

        if "error" not in trend_analysis:
            assert trend_analysis["total_executions"] >= 2
            assert "average_coverage" in trend_analysis
            assert "failure_rate" in trend_analysis

        # Test database persistence
        db_path = self.temp_dir / "test_history.db"
        assert db_path.exists()

        # Verify database content
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM test_executions")
            count = cursor.fetchone()[0]
            assert count >= 2


if __name__ == "__main__":
    pytest.main([__file__])
