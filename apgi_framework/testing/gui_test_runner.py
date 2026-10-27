#!/usr/bin/env python3
"""GUI Test Runner for APGI Framework.

This module provides a graphical user interface for running and monitoring
comprehensive test suites for the APGI Framework.
"""

import json
import multiprocessing
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Dict, List, Optional

from apgi_framework.logging.standardized_logging import get_logger

logger = get_logger(__name__)


# GUI imports
try:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Core dependencies
try:
    # Just check if pytest is available
    import pytest  # noqa: F401

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

try:
    from apgi_framework.cli import APGIFrameworkCLI
    from apgi_framework.utils.framework_test_utils import FrameworkTestUtils

    UTILS_AVAILABLE = True
except ImportError as e:
    logger.info(f"Warning: Test utilities not available: {e}")
    UTILS_AVAILABLE = False


class TestRunnerGUI:
    """GUI for comprehensive test execution and monitoring."""

    def __init__(self, root: Any) -> None:
        """Initialize the test runner GUI."""
        self.root = root
        self.root.title("APGI Framework - Comprehensive Test Runner")
        self.root.geometry("1200x800")

        # Test execution state
        self.test_process: Optional[subprocess.Popen] = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.is_running = False
        self.runner = ComprehensiveTestRunner()  # type: ignore[no-untyped-call]

        self.create_widgets()
        self.setup_layout()

    def create_widgets(self) -> None:
        """Create GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=tk.NSEW)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="APGI Framework Comprehensive Test Runner",
            font=("Helvetica", 16, "bold"),
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Test Configuration", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, pady=(0, 10), sticky=tk.EW)
        control_frame.columnconfigure(1, weight=1)

        # Test category selection
        ttk.Label(control_frame, text="Test Category:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10)
        )
        self.category_var = tk.StringVar(value="all")
        category_combo = ttk.Combobox(
            control_frame,
            textvariable=self.category_var,
            values=["all", "unit", "integration", "gui", "performance"],
            state="readonly",
            width=15,
        )
        category_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))

        # Options
        self.coverage_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Coverage Report", variable=self.coverage_var).grid(
            row=0, column=2, padx=(0, 10)
        )

        self.parallel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Parallel Execution", variable=self.parallel_var).grid(
            row=0, column=3, padx=(0, 10)
        )

        self.verbose_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Verbose Output", variable=self.verbose_var).grid(
            row=0, column=4, padx=(0, 10)
        )

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=(0, 10), sticky=tk.EW)
        button_frame.columnconfigure(1, weight=1)

        self.run_button = ttk.Button(button_frame, text="Run Tests", command=self.run_tests)
        self.run_button.grid(row=0, column=0, padx=(0, 10))

        self.stop_button = ttk.Button(
            button_frame, text="Stop Tests", command=self.stop_tests, state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=(0, 10))

        ttk.Button(button_frame, text="Clear Output", command=self.clear_output).grid(
            row=0, column=2, padx=(0, 10)
        )

        ttk.Button(button_frame, text="Save Results", command=self.save_results).grid(
            row=0, column=3
        )

        # Progress frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=3, column=0, columnspan=3, pady=(0, 10), sticky=tk.EW)
        progress_frame.columnconfigure(0, weight=1)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=tk.EW)

        # Status label
        self.status_label = ttk.Label(progress_frame, text="Ready to run tests")
        self.status_label.grid(row=1, column=0, pady=(5, 0))

        # Output text area
        output_frame = ttk.LabelFrame(main_frame, text="Test Output", padding="5")
        output_frame.grid(row=4, column=0, columnspan=3, sticky=tk.NSEW)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=20)
        self.output_text.grid(row=0, column=0, sticky="nsew")

        # Results summary frame
        summary_frame = ttk.LabelFrame(main_frame, text="Test Summary", padding="5")
        summary_frame.grid(row=5, column=0, columnspan=3, pady=(10, 0), sticky=tk.EW)

        # Summary labels
        self.summary_vars = {}
        summary_items = [
            ("Total Tests", "total"),
            ("Passed", "passed"),
            ("Failed", "failed"),
            ("Errors", "errors"),
            ("Skipped", "skipped"),
            ("Duration", "duration"),
        ]

        for i, (label, key) in enumerate(summary_items):
            ttk.Label(summary_frame, text=f"{label}:").grid(
                row=0, column=i * 2, padx=(0, 5), sticky=tk.W
            )
            self.summary_vars[key] = tk.StringVar(value="0")
            ttk.Label(
                summary_frame, textvariable=self.summary_vars[key], font=("Courier", 10)
            ).grid(row=0, column=i * 2 + 1, padx=(0, 15), sticky=tk.W)

        # Configure grid weights for main frame
        main_frame.rowconfigure(4, weight=1)

    def setup_layout(self) -> None:
        """Setup additional layout configurations."""
        # Configure tags for output coloring
        self.output_text.tag_configure("error", foreground="red")
        self.output_text.tag_configure("warning", foreground="orange")
        self.output_text.tag_configure("success", foreground="green")
        self.output_text.tag_configure("info", foreground="blue")

    def run_tests(self) -> None:
        """Run the test suite."""
        if self.is_running:
            return

        self.is_running = True
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Running tests...")
        self.progress_var.set(0)
        self.clear_output()

        # Start test execution in separate thread
        test_thread = threading.Thread(target=self._execute_tests)
        test_thread.daemon = True
        test_thread.start()

        # Start output monitoring
        self.root.after(100, self._check_output)

    def stop_tests(self) -> None:
        """Stop running tests."""
        if self.test_process and self.test_process.poll() is None:
            self.test_process.terminate()
            self.status_label.config(text="Tests stopped by user")
            self._finish_test_run()

    def clear_output(self) -> None:
        """Clear the output text area."""
        self.output_text.delete(1.0, tk.END)

    def save_results(self) -> None:
        """Save test results to file."""
        if not hasattr(self, "last_results"):
            messagebox.showwarning("No Results", "No test results to save. Run tests first.")
            return

        filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, "w") as f:
                json.dump(self.last_results, f, indent=2)
            messagebox.showinfo("Success", f"Results saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {e}")

    def _execute_tests(self) -> None:
        """Execute tests in background thread."""
        try:
            category = self.category_var.get()
            if category == "all":
                results = self.runner.run_all_tests(
                    verbose=self.verbose_var.get(),
                    coverage=self.coverage_var.get(),
                    parallel=self.parallel_var.get(),
                )
            else:
                results = self.runner.run_category_tests(
                    category=category,
                    verbose=self.verbose_var.get(),
                    coverage=self.coverage_var.get(),
                )

            self.last_results = results

            # Update summary
            self.output_queue.put("=== TEST RESULTS ===")
            for key, value in results.items():
                if key in [
                    "total_tests",
                    "passed",
                    "failed",
                    "errors",
                    "skipped",
                    "duration",
                ]:
                    self.output_queue.put(f"{key}: {value}")

        except Exception as e:
            self.output_queue.put(f"Error running tests: {e}")
        finally:
            self.root.after(0, self._finish_test_run)

    def _check_output(self) -> None:
        """Check for new output from test process."""
        try:
            while True:
                line = self.output_queue.get_nowait()
                self._append_output(line)
        except queue.Empty:
            pass

        if self.is_running:
            self.root.after(100, self._check_output)

    def _append_output(self, line: str) -> None:
        """Append line to output text area with coloring."""
        # Color coding based on content
        if "ERROR" in line or "FAILED" in line or "❌" in line:
            tag = "error"
        elif "WARNING" in line or "⚠️" in line:
            tag = "warning"
        elif "PASSED" in line or "OK" in line or "✅" in line:
            tag = "success"
        elif "INFO" in line or "📊" in line:
            tag = "info"
        else:
            tag = None

        self.output_text.insert(tk.END, line + "\n")
        if tag:
            start_pos = f"end-{len(line) + 1}c"
            end_pos = "end-1c"
            self.output_text.tag_add(tag, start_pos, end_pos)

        self.output_text.see(tk.END)

    def _finish_test_run(self) -> None:
        """Finish test run and reset UI."""
        self.is_running = False
        self.run_button.config(state="normal")
        self.stop_button.config(state="disabled")

        if self.progress_var.get() < 100:
            self.progress_var.set(100)

        if "stopped" not in self.status_label.cget("text").lower():
            self.status_label.config(text="Test run completed")

        # Update summary if results are available
        if hasattr(self, "last_results"):
            results = self.last_results
            self.summary_vars["total"].set(str(results.get("total_tests", 0)))
            self.summary_vars["passed"].set(str(results.get("passed", 0)))
            self.summary_vars["failed"].set(str(results.get("failed", 0)))
            self.summary_vars["errors"].set(str(results.get("errors", 0)))
            self.summary_vars["skipped"].set(str(results.get("skipped", 0)))
            self.summary_vars["duration"].set(f"{results.get('duration', 0):.2f}s")


class ComprehensiveTestRunner:
    """Comprehensive test runner with multiple execution modes."""

    def __init__(self) -> None:
        self.cli = APGIFrameworkCLI() if UTILS_AVAILABLE else None  # type: ignore[no-untyped-call]
        self.test_utils = FrameworkTestUtils() if UTILS_AVAILABLE else None  # type: ignore[no-untyped-call]

    def run_pytest(
        self,
        args: Optional[List[str]] = None,
        capture_output: bool = False,
        timeout: int = 1800,
    ) -> Any:
        """Run pytest with proper environment setup."""
        env = setup_python_path()

        # Default pytest arguments
        if args is None:
            args = ["tests/", "-v", "--tb=short", "--color=yes"]

        # Construct pytest command
        cmd = [sys.executable, "-m", "pytest"] + args

        if not capture_output:
            logger.info(f"Running: {' '.join(cmd)}")
            logger.info(f"PYTHONPATH: {env['PYTHONPATH']}")
            logger.info("-" * 60)

        # Run pytest
        try:
            if capture_output:
                result = subprocess.run(
                    cmd,
                    env=env,
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return result
            else:
                result = subprocess.run(
                    cmd, env=env, cwd=project_root, capture_output=False, text=True
                )
                return result
        except KeyboardInterrupt:
            if not capture_output:
                logger.info("\nTest execution interrupted by user")
            if capture_output:
                return self._create_mock_result(130, "Test execution interrupted by user")
            return 130
        except subprocess.TimeoutExpired:
            if not capture_output:
                logger.info(f"\nTest execution timed out after {timeout} seconds")
            if capture_output:
                return self._create_mock_result(
                    124, f"Test execution timed out after {timeout} seconds"
                )
            return 124
        except Exception as e:
            if not capture_output:
                logger.info(f"Error running pytest: {e}")
            if capture_output:
                return self._create_mock_result(1, str(e))
            return 1

    def _create_mock_result(self, returncode: int, stderr: str) -> CompletedProcess[str]:
        """Create a mock result object for error conditions."""

        class MockResult(CompletedProcess[str]):
            def __init__(self, rc: int, err: str):
                self.returncode = rc
                self.stdout = ""
                self.stderr = err
                self.args = []

        return MockResult(returncode, stderr)

    def run_all_tests(
        self, verbose: bool = False, coverage: bool = False, parallel: bool = True
    ) -> Dict[str, Any]:
        """Run all tests with comprehensive reporting."""
        logger.info("🧪 Running Comprehensive Test Suite")
        logger.info("=" * 50)

        results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "duration": 0,
            "categories": {},
            "details": {},
        }

        # Build pytest command
        cmd = ["tests"]

        if verbose:
            cmd.append("-vv")
        else:
            cmd.append("-v")

        if coverage:
            cmd.extend(
                [
                    "--cov=apgi_framework",
                    "--cov=research",
                    "--cov-report=html",
                    "--cov-report=term-missing",
                    "--cov-report=json",
                ]
            )

        # Add parallel execution if enabled
        if parallel and multiprocessing.cpu_count() > 1:
            cmd.extend(["-n", str(multiprocessing.cpu_count())])
            logger.info(f"Running tests in parallel with {multiprocessing.cpu_count()} workers")

        # Add test markers and categories
        cmd.extend(["--tb=short", "--strict-markers", "--color=yes"])

        try:
            start_time = datetime.now()

            # Run pytest with captured output
            result = self.run_pytest(cmd, capture_output=True)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            results["duration"] = duration
            results["exit_code"] = result.returncode
            results["stdout"] = result.stdout
            results["stderr"] = result.stderr

            # Parse results from output
            self._parse_pytest_output(result.stdout, results)

            # Print summary
            self._print_summary(results)

            return results

        except Exception as e:
            logger.info(f"❌ Error running tests: {e}")
            results["errors"] = 1
            results["exception"] = str(e)
            return results

    def run_category_tests(
        self, category: str, verbose: bool = False, coverage: bool = False
    ) -> Dict[str, Any]:
        """Run tests for a specific category."""
        logger.info(f"🧪 Running {category} Tests")
        logger.info("=" * 30)

        cmd = ["-m", category]

        if verbose:
            cmd.append("-vv")
        else:
            cmd.append("-v")

        if coverage:
            cmd.extend(["--cov=apgi_framework", "--cov-report=term-missing"])

        try:
            result = self.run_pytest(cmd, capture_output=True, timeout=600)

            results = {
                "category": category,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat(),
            }

            # Parse basic stats
            self._parse_pytest_output(result.stdout, results)

            logger.info(f"✅ {category} tests completed with exit code: {result.returncode}")
            return results

        except Exception as e:
            logger.info(f"❌ Error running {category} tests: {e}")
            return {"category": category, "error": str(e)}

    def run_gui_tests(self, **kwargs: Any) -> Dict[str, Any]:
        """Run GUI-specific tests."""
        return self.run_category_tests("gui", **kwargs)

    def run_unit_tests(self, **kwargs: Any) -> Dict[str, Any]:
        """Run unit tests."""
        return self.run_category_tests("unit", **kwargs)

    def run_integration_tests(self, **kwargs: Any) -> Dict[str, Any]:
        """Run integration tests."""
        return self.run_category_tests("integration", **kwargs)

    def run_performance_tests(self, **kwargs: Any) -> Dict[str, Any]:
        """Run performance tests."""
        return self.run_category_tests("performance", **kwargs)

    def run_specific_test(self, test_path: str, **kwargs: Any) -> Dict[str, Any]:
        """Run a specific test file or test function."""
        logger.info(f"🧪 Running Specific Test: {test_path}")
        logger.info("=" * 40)

        cmd = [test_path]

        if kwargs.get("verbose", False):
            cmd.append("-vv")
        else:
            cmd.append("-v")

        if kwargs.get("coverage", False):
            cmd.extend(["--cov=apgi_framework", "--cov-report=term-missing"])

        try:
            result = self.run_pytest(cmd, capture_output=True, timeout=300)

            results = {
                "test_path": test_path,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"✅ Test {test_path} completed with exit code: {result.returncode}")
            return results

        except Exception as e:
            logger.info(f"❌ Error running test {test_path}: {e}")
            return {"test_path": test_path, "error": str(e)}

    def _parse_pytest_output(self, output: str, results: Dict[str, Any]) -> None:
        """Parse pytest output to extract test statistics."""
        try:
            lines = output.split("\n")
            for line in lines:
                if " passed" in line or " failed" in line or " skipped" in line or " error" in line:
                    # Parse the summary line
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            results["total_tests"] += int(part)
                        elif "passed" in part:
                            results["passed"] = int(part.split()[0])
                        elif "failed" in part:
                            results["failed"] = int(part.split()[0])
                        elif "skipped" in part:
                            results["skipped"] = int(part.split()[0])
                        elif "error" in part:
                            results["errors"] = int(part.split()[0])
        except Exception:
            # If parsing fails, continue with default values
            pass

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print a formatted summary of test results."""
        logger.info("\n" + "=" * 50)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total Tests: {results['total_tests']}")
        logger.info(f"✅ Passed: {results['passed']}")
        logger.info(f"❌ Failed: {results['failed']}")
        logger.info(f"⏭️  Skipped: {results['skipped']}")
        logger.info(f"🚨 Errors: {results['errors']}")
        logger.info(f"⏱️  Duration: {results['duration']:.2f}s")
        logger.info(f"📅 Timestamp: {results['timestamp']}")

        if results.get("exit_code") == 0:
            logger.info("🎉 All tests passed!")
        else:
            logger.info("⚠️  Some tests failed or had errors")

        logger.info("=" * 50)


def setup_python_path() -> Dict[str, str]:
    """Set up the Python path to include the project root."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.absolute()

    # Add project root to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Set PYTHONPATH environment variable
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(project_root)

    return env
