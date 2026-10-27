"""Main Application Entry Points for APGI Framework Test Enhancement.

This module provides the main entry points for both GUI and CLI modes of the
comprehensive test enhancement system, including dependency injection,
configuration management, and graceful shutdown procedures.

Requirements: All requirements
"""

import argparse
import atexit
import json
import logging
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from apgi_framework.testing.activity_logger import (
    ActivityLevel,
    ActivityType,
    get_activity_logger,
)
from apgi_framework.testing.batch_runner import BatchTestRunner
from apgi_framework.testing.ci_integrator import CIConfiguration, CIIntegrator
from apgi_framework.testing.notification_manager import NotificationManager
from apgi_framework.utils.logging_utils import setup_logging


@dataclass
class ApplicationConfig:
    """Main application configuration."""

    mode: str  # 'gui' or 'cli'
    project_root: str
    log_level: str = "INFO"
    config_file: Optional[str] = None
    gui_theme: str = "default"
    parallel_execution: bool = True
    max_workers: int = 4
    coverage_threshold: float = 0.8
    notification_channels: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        if self.notification_channels is None:
            self.notification_channels = []


class DependencyContainer:
    """Dependency injection container for application components."""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self._instances: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    def get_batch_runner(self) -> BatchTestRunner:
        """Get or create BatchTestRunner instance."""
        if "batch_runner" not in self._instances:
            self._instances["batch_runner"] = BatchTestRunner()
        return cast(BatchTestRunner, self._instances["batch_runner"])

    def get_ci_integrator(self) -> CIIntegrator:
        """Get or create CIIntegrator instance."""
        if "ci_integrator" not in self._instances:
            ci_config = CIConfiguration(
                test_subset_strategy="all",
                pipeline_type="generic",
                parallel_execution=self.config.parallel_execution,
                max_workers=self.config.max_workers,
                coverage_threshold=self.config.coverage_threshold,
            )
            self._instances["ci_integrator"] = CIIntegrator(
                project_root=self.config.project_root, config=ci_config
            )
        return cast(CIIntegrator, self._instances["ci_integrator"])

    def get_notification_manager(self) -> NotificationManager:
        """Get or create NotificationManager instance."""
        if "notification_manager" not in self._instances:
            from apgi_framework.testing.notification_manager import NotificationChannel

            channels = []
            if self.config.notification_channels:
                for channel_config in self.config.notification_channels:
                    channels.append(NotificationChannel(**channel_config))

            self._instances["notification_manager"] = NotificationManager(channels)
        return cast(NotificationManager, self._instances["notification_manager"])

    def cleanup(self) -> None:
        """Cleanup all managed instances."""
        for name, instance in self._instances.items():
            try:
                if hasattr(instance, "cleanup"):
                    instance.cleanup()
                elif hasattr(instance, "close"):
                    instance.close()
            except Exception as e:
                self.logger.warning(f"Error cleaning up {name}: {e}")

        self._instances.clear()


class ApplicationLifecycleManager:
    """Manages application lifecycle including startup validation and shutdown."""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.container = DependencyContainer(config)
        self.logger = logging.getLogger(__name__)
        self._shutdown_handlers: List[Callable] = []
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum: Any, frame: Any) -> None:
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Register cleanup on normal exit
        atexit.register(self.shutdown)

    def validate_startup(self) -> bool:
        """Validate system requirements and configuration."""
        try:
            # Check Python version
            if sys.version_info < (3, 8):
                self.logger.error("Python 3.8 or higher is required")
                return False

            # Check project root exists
            project_path = Path(self.config.project_root)
            if not project_path.exists():
                self.logger.error(f"Project root does not exist: {project_path}")
                return False

            # Check required directories
            required_dirs = ["tests", "apgi_framework"]
            for dir_name in required_dirs:
                dir_path = project_path / dir_name
                if not dir_path.exists():
                    self.logger.warning(f"Expected directory not found: {dir_path}")

            # Validate configuration
            if self.config.max_workers < 1:
                self.logger.error("max_workers must be at least 1")
                return False

            if not 0 <= self.config.coverage_threshold <= 1:
                self.logger.error("coverage_threshold must be between 0 and 1")
                return False

            # Test logging setup
            activity_logger = get_activity_logger()
            activity_logger.log_activity(
                ActivityType.SYSTEM_EVENT,
                ActivityLevel.INFO,
                "Application startup validation completed",
                data={
                    "mode": self.config.mode,
                    "project_root": self.config.project_root,
                },
            )

            self.logger.info("Startup validation completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Startup validation failed: {e}")
            return False

    def register_shutdown_handler(self, handler: Callable[[], None]) -> None:
        """Register a shutdown handler function."""
        self._shutdown_handlers.append(handler)

    def shutdown(self) -> None:
        """Perform graceful shutdown."""
        self.logger.info("Initiating application shutdown...")

        # Call registered shutdown handlers
        for handler in self._shutdown_handlers:
            try:
                handler()
            except Exception as e:
                self.logger.warning(f"Error in shutdown handler: {e}")

        # Cleanup dependency container
        self.container.cleanup()

        # Log shutdown completion
        activity_logger = get_activity_logger()
        activity_logger.log_activity(
            ActivityType.SYSTEM_SHUTDOWN,
            ActivityLevel.INFO,
            "Application shutdown completed",
            data={"timestamp": datetime.now().isoformat()},
        )

        self.logger.info("Application shutdown completed")


def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_file, "r") as f:
            return cast(Dict[str, Any], json.load(f))
    except FileNotFoundError:
        logging.warning(f"Config file not found: {config_file}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file {config_file}: {e}")
        return {}


def create_default_config() -> ApplicationConfig:
    """Create default application configuration."""
    return ApplicationConfig(
        mode="cli",
        project_root=str(Path.cwd()),
        log_level="INFO",
        parallel_execution=True,
        max_workers=4,
        coverage_threshold=0.8,
    )


def parse_cli_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="APGI Framework Test Enhancement System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s gui                          # Launch GUI mode
  %(prog)s cli --run-all               # Run all tests in CLI mode
  %(prog)s cli --run-unit              # Run unit tests only
  %(prog)s cli --coverage-report       # Generate coverage report
  %(prog)s cli --config config.json    # Use custom configuration
        """,
    )

    parser.add_argument(
        "mode",
        choices=["gui", "cli"],
        help="Application mode: gui for graphical interface, cli for command line",
    )

    parser.add_argument("--config", type=str, help="Path to configuration file (JSON format)")

    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path.cwd()),
        help="Project root directory (default: current directory)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)",
    )

    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.8,
        help="Coverage threshold (0.0-1.0, default: 0.8)",
    )

    # CLI-specific options
    cli_group = parser.add_argument_group("CLI options")
    cli_group.add_argument("--run-all", action="store_true", help="Run all tests")

    cli_group.add_argument("--run-unit", action="store_true", help="Run unit tests only")

    cli_group.add_argument(
        "--run-integration", action="store_true", help="Run integration tests only"
    )

    cli_group.add_argument(
        "--coverage-report", action="store_true", help="Generate coverage report"
    )

    cli_group.add_argument(
        "--test-pattern",
        type=str,
        help='Test file pattern to run (e.g., "test_core_*.py")',
    )

    # GUI-specific options
    gui_group = parser.add_argument_group("GUI options")
    gui_group.add_argument(
        "--theme",
        choices=["default", "dark", "light"],
        default="default",
        help="GUI theme (default: default)",
    )

    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> ApplicationConfig:
    """Create application configuration from command line arguments."""
    # Load base config from file if specified
    config_data = {}
    if args.config:
        config_data = load_config_from_file(args.config)

    # Override with command line arguments
    config_data.update(
        {
            "mode": args.mode,
            "project_root": args.project_root,
            "log_level": args.log_level,
            "config_file": args.config,
            "max_workers": args.max_workers,
            "coverage_threshold": args.coverage_threshold,
        }
    )

    if args.mode == "gui":
        config_data["gui_theme"] = args.theme

    return ApplicationConfig(**config_data)


def run_gui_mode(lifecycle_manager: ApplicationLifecycleManager) -> int:
    """Run the application in GUI mode."""
    try:
        import tkinter as tk

        from apgi_framework.testing.gui_test_runner import GUI_AVAILABLE, TestRunnerGUI

        if not GUI_AVAILABLE:
            logger = logging.getLogger(__name__)
            logger.error("GUI mode not available - missing tkinter dependencies")
            return 1

        logger = logging.getLogger(__name__)
        logger.info("Starting GUI mode...")

        # Create and run GUI
        root = tk.Tk()
        TestRunnerGUI(root=root)

        # Run the GUI
        root.mainloop()
        return 0

    except ImportError as e:
        logging.error(f"GUI dependencies not available: {e}")
        logging.error("Please install GUI dependencies: pip install PySide6")
        return 1
    except Exception as e:
        logging.error(f"GUI mode failed: {e}")
        return 1


def run_cli_mode(lifecycle_manager: ApplicationLifecycleManager, args: argparse.Namespace) -> int:
    """Run the application in CLI mode."""
    try:
        from apgi_framework.testing.cli_runner import CLITestRunner

        logger = logging.getLogger(__name__)
        logger.info("Starting CLI mode...")

        # Create CLI runner
        cli_runner = CLITestRunner(
            container=lifecycle_manager.container, config=lifecycle_manager.config
        )

        # Register CLI cleanup
        lifecycle_manager.register_shutdown_handler(cli_runner.cleanup)

        # Execute CLI command
        result: Any
        if args.run_all:
            result = cli_runner.run_all_tests()
        elif args.run_unit:
            result = cli_runner.run_unit_tests()
        elif args.run_integration:
            result = cli_runner.run_integration_tests()
        elif args.coverage_report:
            result = cli_runner.generate_coverage_report()
        elif args.test_pattern:
            result = cli_runner.run_tests_by_pattern(args.test_pattern)
        else:
            # Interactive CLI mode
            result = cli_runner.run_interactive()
        return int(result) if isinstance(result, int) else 0

    except Exception as e:
        logging.error(f"CLI mode failed: {e}")
        return 1


def main() -> int:
    """Main application entry point."""
    try:
        # Parse command line arguments
        args = parse_cli_arguments()

        # Create configuration
        config = create_config_from_args(args)

        # Setup logging
        setup_logging(
            level=config.log_level,
            log_file=Path(config.project_root) / "logs" / "test_enhancement.log",
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Starting APGI Framework Test Enhancement System in {config.mode} mode")

        # Create lifecycle manager
        lifecycle_manager = ApplicationLifecycleManager(config)

        # Validate startup
        if not lifecycle_manager.validate_startup():
            logger.error("Startup validation failed")
            return 1

        # Run appropriate mode
        if config.mode == "gui":
            return run_gui_mode(lifecycle_manager)
        else:
            return run_cli_mode(lifecycle_manager, args)

    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logging.error(f"Application failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
