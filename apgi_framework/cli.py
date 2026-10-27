"""Command-Line Interface for APGI Framework Testing System.

This module provides a comprehensive CLI for running individual tests,
batch experiment execution, and configuration management.
"""

import argparse
import logging
import re
import sys
from typing import Any, Dict, List, Optional

from apgi_framework.commands import COMMAND_REGISTRY
from apgi_framework.logging.standardized_logging import get_logger
from apgi_framework.main_controller import MainApplicationController

logger = get_logger(__name__)


def sanitize_input(value: str) -> str:
    """Sanitize input string to prevent command injection and other attacks.
    
    Removes potentially dangerous characters like ; | & > < ` $ ( ) { } [ ] \\ 
    """
    if not isinstance(value, str):
        return value
    # Remove characters often used in command injection
    sanitized = re.sub(r"[;\|&><`$()\[\]{}\\\'\"]", "", value)
    return sanitized.strip()


def validate_trials_range(value: str) -> int:
    """Validate trials argument is within documented range (100-10000)."""
    try:
        ivalue = int(value)
        if not 100 <= ivalue <= 10000:
            raise argparse.ArgumentTypeError(
                f"Number of trials must be between 100 and 10000, got {ivalue}"
            )
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value for trials: {value}")


def validate_participants_range(value: str) -> int:
    """Validate participants argument is within documented range (10-1000)."""
    try:
        ivalue = int(value)
        if not 10 <= ivalue <= 1000:
            raise argparse.ArgumentTypeError(
                f"Number of participants must be between 10 and 1000, got {ivalue}"
            )
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value for participants: {value}")


def validate_threshold_range(value: str) -> float:
    """Validate threshold argument is within documented range (0.5-10.0)."""
    try:
        fvalue = float(value)
        if not 0.5 <= fvalue <= 10.0:
            raise argparse.ArgumentTypeError(
                f"Threshold must be between 0.5 and 10.0, got {fvalue}"
            )
        return fvalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value for threshold: {value}")


def validate_positive_int(value: str) -> int:
    """Validate positive integer argument."""
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"Value must be a positive integer, got {ivalue}")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value}")


def validate_workers_range(value: str) -> int:
    """Validate max-workers argument is within reasonable range (1-64)."""
    try:
        ivalue = int(value)
        if not 1 <= ivalue <= 64:
            raise argparse.ArgumentTypeError(f"Max workers must be between 1 and 64, got {ivalue}")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value for max-workers: {value}")


def validate_timeout_range(value: str) -> int:
    """Validate timeout argument is within reasonable range (1-3600 seconds)."""
    try:
        ivalue = int(value)
        if not 1 <= ivalue <= 3600:
            raise argparse.ArgumentTypeError(
                f"Timeout must be between 1 and 3600 seconds, got {ivalue}"
            )
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value for timeout: {value}")


def validate_days_range(value: str) -> int:
    """Validate days argument is within reasonable range (1-365)."""
    try:
        ivalue = int(value)
        if not 1 <= ivalue <= 365:
            raise argparse.ArgumentTypeError(f"Days must be between 1 and 365, got {ivalue}")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value for days: {value}")


def validate_coverage_threshold_range(value: str) -> float:
    """Validate coverage threshold argument is within range (0-100)."""
    try:
        fvalue = float(value)
        if not 0 <= fvalue <= 100:
            raise argparse.ArgumentTypeError(
                f"Coverage threshold must be between 0 and 100, got {fvalue}"
            )
        return fvalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value for coverage threshold: {value}")


def validate_precision_range(value: str) -> float:
    """Validate precision argument is within reasonable range (0.001-1000)."""
    try:
        fvalue = float(value)
        if not 0.001 <= fvalue <= 1000:
            raise argparse.ArgumentTypeError(
                f"Precision must be between 0.001 and 1000, got {fvalue}"
            )
        return fvalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value for precision: {value}")


def validate_steepness_range(value: str) -> float:
    """Validate steepness argument is within reasonable range (0.1-50.0)."""
    try:
        fvalue = float(value)
        if not 0.1 <= fvalue <= 50.0:
            raise argparse.ArgumentTypeError(
                f"Steepness must be between 0.1 and 50.0, got {fvalue}"
            )
        return fvalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value for steepness: {value}")


def validate_gain_range(value: str) -> float:
    """Validate gain argument is within reasonable range (-10.0 to 10.0)."""
    try:
        fvalue = float(value)
        if not -10.0 <= fvalue <= 10.0:
            raise argparse.ArgumentTypeError(f"Gain must be between -10.0 and 10.0, got {fvalue}")
        return fvalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value for gain: {value}")


class APGIFrameworkCLI:
    """Command-line interface for the APGI Framework Testing System."""

    def __init__(self) -> None:
        """Initialize the CLI."""
        self.controller: Optional[MainApplicationController] = None
        self.logger: Optional[Any] = None

    def setup_logging(self, log_level: str = "INFO") -> None:
        """Setup logging for CLI operations."""
        try:
            from apgi_framework.logging.centralized_logging import APGILogManager

            APGILogManager.setup_logging(level=log_level)
            self.logger = logging.getLogger(__name__)
        except ImportError:
            from apgi_framework.logging.standardized_logging import get_logger

            self.logger = get_logger(__name__)

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description="APGI Framework Testing System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run primary falsification test
  python -m apgi_framework.cli run-test primary --trials 1000
  
  # Run all tests in batch mode
  python -m apgi_framework.cli run-batch --config config.json
  
  # Generate default configuration
  python -m apgi_framework.cli generate-config --output config.json
  
  # Validate system components
  python -m apgi_framework.cli validate-system
            """,
        )

        # Global options
        parser.add_argument(
            "--config",
            "-c",
            type=str,
            help="Path to JSON configuration file with APGI parameters",
        )

        parser.add_argument(
            "--log-level",
            "-l",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
            help="Set logging level (default: INFO)",
        )

        parser.add_argument(
            "--output-dir",
            "-o",
            type=str,
            help="Output directory for results and reports",
        )

        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Run individual test command
        test_parser = subparsers.add_parser("run-test", help="Run individual test")
        test_parser.add_argument(
            "test_type",
            choices=[
                "primary",
                "consciousness-without-ignition",
                "threshold-insensitivity",
                "soma-bias",
            ],
            help="Type of test to run",
        )
        test_parser.add_argument(
            "--trials",
            "-n",
            type=validate_trials_range,
            default=1000,
            help="Number of trials to run (range: 100-10000, default: 1000)",
        )
        test_parser.add_argument(
            "--participants",
            "-p",
            type=validate_participants_range,
            default=100,
            help="Number of participants to simulate (range: 10-1000, default: 100)",
        )
        test_parser.add_argument(
            "--seed",
            type=validate_positive_int,
            help="Random seed for reproducible results",
        )
        test_parser.add_argument("--config", "-c", type=str, help="Path to configuration file")

        batch_parser = subparsers.add_parser("run-batch", help="Run batch experiments")
        batch_parser.add_argument("--all-tests", action="store_true", help="Run all tests")
        batch_parser.add_argument(
            "--tests",
            nargs="+",
            choices=[
                "primary",
                "consciousness-without-ignition",
                "threshold-insensitivity",
                "soma-bias",
            ],
            help="Specific tests to run in batch",
        )
        batch_parser.add_argument(
            "--parallel",
            action="store_true",
            help="Run tests in parallel (experimental)",
        )

        batch_test_parser = subparsers.add_parser(
            "batch-test", help="Advanced batch test execution"
        )
        batch_test_parser.add_argument(
            "--test-paths",
            nargs="+",
            help="Specific test file paths to run (e.g., tests/test_core.py)",
        )
        batch_test_parser.add_argument(
            "--markers",
            nargs="+",
            choices=[
                "unit",
                "integration",
                "research",
                "core",
                "slow",
                "neural",
                "behavioral",
            ],
            help="Run tests with specific markers",
        )
        batch_test_parser.add_argument(
            "--keywords",
            type=str,
            help="Run tests matching keyword patterns in test names",
        )
        batch_test_parser.add_argument(
            "--parallel",
            action="store_true",
            default=True,
            help="Run tests in parallel (default: True)",
        )
        batch_test_parser.add_argument(
            "--sequential", action="store_true", help="Run tests sequentially"
        )
        batch_test_parser.add_argument(
            "--max-workers",
            type=validate_workers_range,
            help="Maximum number of parallel workers (range: 1-64)",
        )
        batch_test_parser.add_argument(
            "--timeout",
            type=validate_timeout_range,
            default=600,
            help="Timeout per test in seconds (range: 1-3600, default: 600)",
        )
        batch_test_parser.add_argument(
            "--failfast", action="store_true", help="Stop on first failure"
        )
        batch_test_parser.add_argument("--report", type=str, help="Output path for HTML report")

        result_parser = subparsers.add_parser("test-results", help="Manage test results")
        result_parser.add_argument("--list", action="store_true", help="List recent test results")
        result_parser.add_argument(
            "--show",
            type=str,
            help="Display detailed results from specific test result file",
        )
        result_parser.add_argument(
            "--rerun-failed",
            type=str,
            help="Re-run only failed tests from specified result file",
        )
        result_parser.add_argument(
            "--clean",
            action="store_true",
            help="Remove old test result files and temporary data",
        )

        # Test analysis and reporting commands
        analysis_parser = subparsers.add_parser(
            "test-analysis", help="Analyze test results and performance"
        )
        analysis_parser.add_argument(
            "--performance-report",
            action="store_true",
            help="Generate performance report",
        )
        analysis_parser.add_argument(
            "--days",
            type=validate_days_range,
            default=30,
            help="Number of days to analyze (range: 1-365, default: 30)",
        )
        analysis_parser.add_argument(
            "--trends", action="store_true", help="Show performance trends"
        )
        analysis_parser.add_argument(
            "--failures", action="store_true", help="Analyze failure patterns"
        )
        analysis_parser.add_argument("--test-name", type=str, help="Analyze specific test")
        analysis_parser.add_argument("--export", type=str, help="Export results to file")
        analysis_parser.add_argument(
            "--format",
            choices=["json", "csv"],
            default="json",
            help="Export format (default: json)",
        )

        # Test generation and coverage commands
        coverage_parser = subparsers.add_parser(
            "test-coverage", help="Analyze and generate test coverage"
        )
        coverage_parser.add_argument(
            "--analyze", action="store_true", help="Analyze test coverage gaps"
        )
        coverage_parser.add_argument(
            "--generate", action="store_true", help="Generate missing tests"
        )
        coverage_parser.add_argument(
            "--report", action="store_true", help="Generate coverage report"
        )
        coverage_parser.add_argument(
            "--output-dir",
            type=str,
            default="generated_tests",
            help="Output directory for generated tests (default: generated_tests)",
        )
        coverage_parser.add_argument(
            "--report-file",
            type=str,
            default="coverage_report.md",
            help="Output file for coverage report (default: coverage_report.md)",
        )
        coverage_parser.add_argument(
            "--root-path",
            type=str,
            help="Root path to analyze (default: current directory)",
        )
        coverage_parser.add_argument(
            "--threshold",
            type=validate_coverage_threshold_range,
            default=90.0,
            help="Coverage threshold percentage (range: 0-100, default: 90.0)",
        )
        coverage_parser.add_argument(
            "--format",
            choices=["html", "xml", "json", "text"],
            default="html",
            help="Coverage report format (default: html)",
        )
        coverage_parser.add_argument(
            "--include-patterns",
            nargs="+",
            help="File patterns to include in coverage analysis",
        )
        coverage_parser.add_argument(
            "--exclude-patterns",
            nargs="+",
            help="File patterns to exclude from coverage analysis",
        )

        # Enhanced test execution commands with GUI feature parity
        test_exec_parser = subparsers.add_parser(
            "run-tests", help="Enhanced test execution with GUI feature parity"
        )
        test_exec_parser.add_argument("--test-paths", nargs="+", help="Specific test paths to run")
        test_exec_parser.add_argument(
            "--categories",
            nargs="+",
            choices=["unit", "integration", "property", "gui", "performance"],
            help="Test categories to run",
        )
        test_exec_parser.add_argument(
            "--modules",
            nargs="+",
            help="Specific modules to test (e.g., core, clinical, neural)",
        )
        test_exec_parser.add_argument("--tags", nargs="+", help="Test tags to filter by")
        test_exec_parser.add_argument("--filter", type=str, help="Test name filter pattern")
        test_exec_parser.add_argument(
            "--parallel",
            action="store_true",
            default=True,
            help="Run tests in parallel (default: True)",
        )
        test_exec_parser.add_argument(
            "--sequential", action="store_true", help="Run tests sequentially"
        )
        test_exec_parser.add_argument(
            "--max-workers",
            type=validate_workers_range,
            help="Maximum number of parallel workers (range: 1-64)",
        )
        test_exec_parser.add_argument(
            "--timeout",
            type=validate_timeout_range,
            default=600,
            help="Timeout per test in seconds (range: 1-3600, default: 600)",
        )
        test_exec_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
        test_exec_parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")
        test_exec_parser.add_argument(
            "--progress",
            choices=["bar", "dots", "none"],
            default="bar",
            help="Progress display style (default: bar)",
        )
        test_exec_parser.add_argument(
            "--coverage",
            action="store_true",
            help="Collect coverage during test execution",
        )
        test_exec_parser.add_argument(
            "--coverage-report",
            choices=["html", "xml", "json", "text", "none"],
            default="html",
            help="Coverage report format (default: html)",
        )
        test_exec_parser.add_argument(
            "--save-results",
            action="store_true",
            default=True,
            help="Save test results to database (default: True)",
        )
        test_exec_parser.add_argument(
            "--output-format",
            choices=["json", "xml", "html", "text"],
            default="text",
            help="Output format for results (default: text)",
        )
        test_exec_parser.add_argument("--output-file", type=str, help="Output file for results")

        # Test organization and filtering commands
        organize_parser = subparsers.add_parser(
            "organize-tests", help="Organize and categorize tests"
        )
        organize_parser.add_argument("--discover", action="store_true", help="Discover all tests")
        organize_parser.add_argument(
            "--categorize", action="store_true", help="Categorize discovered tests"
        )
        organize_parser.add_argument(
            "--list-categories", action="store_true", help="List test categories"
        )
        organize_parser.add_argument(
            "--list-modules", action="store_true", help="List test modules"
        )
        organize_parser.add_argument(
            "--list-tags", action="store_true", help="List available test tags"
        )
        organize_parser.add_argument(
            "--export-tree",
            type=str,
            help="Export test tree to file (JSON format)",
        )
        organize_parser.add_argument(
            "--root-path",
            type=str,
            help="Root path for test discovery (default: current directory)",
        )

        # Configuration management commands
        config_parser = subparsers.add_parser("generate-config", help="Generate configuration file")
        config_parser.add_argument(
            "--output",
            type=str,
            default="apgi_config.json",
            help="Output path for configuration file (default: apgi_config.json)",
        )
        config_parser.add_argument(
            "--template",
            choices=["default", "minimal", "comprehensive"],
            default="default",
            help="Configuration template to use (default: default)",
        )

        # System validation command
        validate_parser = subparsers.add_parser(
            "validate-system", help="Validate system components"
        )
        validate_parser.add_argument(
            "--detailed", action="store_true", help="Show detailed validation results"
        )

        # Status command
        subparsers.add_parser("status", help="Show system status")

        # Parameter override commands
        param_parser = subparsers.add_parser("set-params", help="Set APGI parameters")
        param_parser.add_argument(
            "--extero-precision",
            type=validate_precision_range,
            help="Exteroceptive precision (range: 0.001-1000)",
        )
        param_parser.add_argument(
            "--intero-precision",
            type=validate_precision_range,
            help="Interoceptive precision (range: 0.001-1000)",
        )
        param_parser.add_argument(
            "--threshold",
            type=validate_threshold_range,
            help="Ignition threshold (range: 0.5-10.0)",
        )
        param_parser.add_argument(
            "--steepness",
            type=validate_steepness_range,
            help="Sigmoid steepness (range: 0.1-50.0)",
        )
        param_parser.add_argument(
            "--somatic-gain",
            type=validate_gain_range,
            help="Somatic marker gain (range: -10.0 to 10.0)",
        )

        return parser

    def initialize_controller(self, config_path: Optional[str] = None) -> None:
        """Initialize the main application controller."""
        if self.logger is None:
            self.setup_logging()
        try:
            self.controller = MainApplicationController(config_path)
            self.controller.initialize_system()
            if self.logger:
                self.logger.info("System initialized successfully")
        except (RuntimeError, IOError, ValueError) as e:
            if self.logger:
                self.logger.error(f"Failed to initialize system: {e}")
            sys.exit(1)

    def run_enhanced_tests(self, args: argparse.Namespace) -> None:
        """Run enhanced tests with GUI feature parity.

        **Property 23a: Test execution feature parity**

        Delegates to RunEnhancedTestsCommand for test execution.
        """
        from apgi_framework.commands.test import RunEnhancedTestsCommand

        command = RunEnhancedTestsCommand(self.controller)
        command.execute(args)

    def organize_tests(self, args: argparse.Namespace) -> None:
        """Organize and categorize tests.

        **Property 23c: Test organization feature parity**

        Delegates to OrganizeTestsCommand for test organization.
        """
        from apgi_framework.commands.test import OrganizeTestsCommand

        command = OrganizeTestsCommand(self.controller)
        command.execute(args)

    def manage_enhanced_coverage(self, args: argparse.Namespace) -> None:
        """Manage enhanced test coverage analysis.

        **Property 23b: Coverage analysis feature parity**

        Delegates to TestCoverageCommand for coverage management.
        """
        from apgi_framework.commands.analysis import TestCoverageCommand

        command = TestCoverageCommand(self.controller)
        command.execute(args)

    def manage_test_results(self, args: argparse.Namespace) -> None:
        """Manage test results.

        **Property 23g: Complete command coverage**

        Delegates to TestResultsCommand for result management.
        """
        from apgi_framework.commands.analysis import TestResultsCommand

        command = TestResultsCommand(self.controller)
        command.execute(args)

    def analyze_test_results(self, args: argparse.Namespace) -> None:
        """Analyze test results and performance.

        **Property 23g: Complete command coverage**

        Delegates to TestAnalysisCommand for result analysis.
        """
        from apgi_framework.commands.analysis import TestAnalysisCommand

        command = TestAnalysisCommand(self.controller)
        command.execute(args)

    def generate_configuration(self, args: argparse.Namespace) -> None:
        """Generate configuration file.

        **Property 23f: Configuration management feature parity**

        Delegates to GenerateConfigCommand for config generation.
        """
        from apgi_framework.commands.config import GenerateConfigCommand

        command = GenerateConfigCommand(self.controller)
        command.execute(args)

    def validate_system(self, args: argparse.Namespace) -> None:
        """Validate system components.

        **Property 23g: Complete command coverage**

        Delegates to ValidateSystemCommand for system validation.
        """
        from apgi_framework.commands.system import ValidateSystemCommand

        command = ValidateSystemCommand(self.controller)
        command.execute(args)

    def show_status(self, args: argparse.Namespace) -> None:
        """Show system status.

        **Property 23g: Complete command coverage**

        Delegates to StatusCommand for status display.
        """
        from apgi_framework.commands.system import StatusCommand

        command = StatusCommand(self.controller)
        command.execute(args)

    def set_parameters(self, args: argparse.Namespace) -> None:
        """Set APGI parameters.

        **Property 23f: Configuration management feature parity**

        Delegates to SetParamsCommand for parameter setting.
        """
        from apgi_framework.commands.config import SetParamsCommand

        command = SetParamsCommand(self.controller)
        command.execute(args)

    def run_individual_test(self, args: argparse.Namespace) -> None:
        """Run individual test.

        **Property 23g: Complete command coverage**

        Delegates to RunTestCommand for test execution.
        """
        from apgi_framework.commands.test import RunTestCommand

        command = RunTestCommand(self.controller)
        command.execute(args)

    def run_batch_experiments(self, args: argparse.Namespace) -> None:
        """Run batch experiments.

        **Property 23g: Complete command coverage**

        Delegates to RunBatchCommand for batch execution.
        """
        from apgi_framework.commands.test import RunBatchCommand

        command = RunBatchCommand(self.controller)
        command.execute(args)

    def _display_simple_validation(self, results: Dict[str, Any]) -> None:
        """Display simple validation results.

        Helper method for displaying validation results in simple format.
        """
        if self.logger:
            overall = results.get("overall", False)
            status = "PASSED" if overall else "FAILED"
            self.logger.info(f"System Validation: {status}")

    def _display_system_status(self, status: Dict[str, Any]) -> None:
        """Display system status.

        Helper method for displaying system status information.
        """
        if self.logger:
            system_status = status.get("status", "unknown")
            self.logger.info(f"System Status: {system_status}")

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration.

        **Property 23f: Configuration management feature parity**
        """
        return {
            "apgi_parameters": {
                "extero_precision": 2.0,
                "intero_precision": 1.5,
                "extero_error": 1.2,
                "intero_error": 0.8,
                "somatic_gain": 1.3,
                "threshold": 3.5,
                "steepness": 2.0,
            },
            "experimental_config": {
                "n_trials": 1000,
                "n_participants": 100,
                "random_seed": None,
                "output_directory": "results",
                "log_level": "INFO",
                "save_intermediate": True,
            },
        }

    def _create_minimal_config(self) -> Dict[str, Any]:
        """Create minimal configuration.

        **Property 23f: Configuration management feature parity**
        """
        return {
            "apgi_parameters": {"threshold": 3.5, "steepness": 2.0},
            "experimental_config": {"n_trials": 100, "output_directory": "results"},
        }

    def _create_comprehensive_config(self) -> Dict[str, Any]:
        """Create comprehensive configuration.

        **Property 23f: Configuration management feature parity**
        """
        config = self._create_default_config()
        config["experimental_config"].update(
            {
                "detailed_logging": True,
                "save_raw_data": True,
                "generate_plots": True,
                "statistical_corrections": ["fdr", "bonferroni"],
                "bootstrap_iterations": 10000,
                "confidence_interval": 0.95,
            }
        )
        return config

    def _display_results_json(self, execution: Any) -> None:
        """Display results in JSON format.

        **Property 23d: Output format feature parity**
        """
        import json

        if self.logger:
            self.logger.info(json.dumps({"execution_id": getattr(execution, "execution_id", "")}))

    def _display_results_xml(self, execution: Any) -> None:
        """Display results in XML format.

        **Property 23d: Output format feature parity**
        """
        if self.logger:
            self.logger.info("<execution></execution>")

    def _display_results_html(self, execution: Any) -> None:
        """Display results in HTML format.

        **Property 23d: Output format feature parity**
        """
        if self.logger:
            self.logger.info("<html><body>Results</body></html>")

    def _display_results_text(
        self, execution: Any, verbose: bool = False, progress: str = "bar"
    ) -> None:
        """Display results in text format.

        **Property 23d: Output format feature parity**
        **Property 23e: Progress monitoring feature parity**
        """
        if self.logger:
            self.logger.info("Test Results (Text Format)")

    def run(self, args: Optional[List[str]] = None) -> None:
        """Main entry point for the CLI."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        # Setup logging
        self.setup_logging(parsed_args.log_level)

        # Handle case where no command is provided
        if not parsed_args.command:
            parser.print_help()
            return

        # Initialize controller for commands that need it
        commands_needing_controller = [
            "run-batch",
            "validate-system",
            "status",
            "set-params",
            "run-test",
            "batch-test",
            "run-tests",
            "organize-tests",
        ]

        if parsed_args.command in commands_needing_controller:
            self.initialize_controller(parsed_args.config)

        # Map command names to instance method names
        command_method_map = {
            "run-test": "run_individual_test",
            "run-batch": "run_batch_experiments",
            "generate-config": "generate_configuration",
            "validate-system": "validate_system",
            "status": "show_status",
            "set-params": "set_parameters",
        }

        # Execute the requested command
        try:
            # Sanitize all string values in arguments
            sanitized_args = vars(parsed_args).copy()
            for key, value in sanitized_args.items():
                if isinstance(value, str):
                    sanitized_args[key] = sanitize_input(value)
                elif isinstance(value, list):
                    sanitized_args[key] = [
                        sanitize_input(v) if isinstance(v, str) else v for v in value
                    ]

            # Check if there's an instance method for this command
            method_name = command_method_map.get(parsed_args.command)
            if method_name and hasattr(self, method_name):
                method = getattr(self, method_name)
                method(argparse.Namespace(**sanitized_args))
            else:
                # Fall back to command registry
                command_class = COMMAND_REGISTRY.get(parsed_args.command)
                if not command_class:
                    if self.logger:
                        self.logger.error(f"Unknown command: {parsed_args.command}")
                    sys.exit(2)
                command_instance = command_class(self.controller)
                command_instance.execute(argparse.Namespace(**sanitized_args))
        except KeyboardInterrupt:
            if self.logger:
                self.logger.info("Operation cancelled by user")
            sys.exit(0)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error: {e}")
            sys.exit(1)
        finally:
            if self.controller:
                try:
                    self.controller.shutdown_system()
                except Exception:
                    pass
        sys.exit(0)


def main() -> None:
    """Entry point for the CLI when run as a module."""
    cli = APGIFrameworkCLI()
    cli.run()


if __name__ == "__main__":
    main()
