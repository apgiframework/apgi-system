"""Main GUI Controller for APGI Framework.

Orchestrates all GUI components and provides centralized management
for the modular GUI architecture.
"""

import logging
import queue
import sys
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import Any, Callable, Dict, List, Optional, TypedDict, cast

# Type definitions


class ValidationResult(TypedDict):
    """Type definition for system validation results."""

    valid: bool
    issues: List[str]
    warnings: List[str]


# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import GUI components
try:
    from .logging_panel import LoggingPanel
    from .parameter_config_panel import ParameterConfigPanel
    from .results_visualization_panel import ResultsVisualizationPanel
    from .test_execution_panel import TestExecutionPanel

    GUI_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main_gui_controller")
    logger.warning(f"Could not import GUI components: {e}")
    GUI_COMPONENTS_AVAILABLE = False


# Import framework components
try:
    from apgi_framework.config import ConfigManager
    from apgi_framework.logging.standardized_logging import get_logger
    from apgi_framework.main_controller import MainApplicationController

    FRAMEWORK_COMPONENTS_AVAILABLE = True
except ImportError as e:
    FRAMEWORK_COMPONENTS_AVAILABLE = False
    logger = logging.getLogger("main_gui_controller")
    logger.error(f"Framework components not available: {e}")


if "logger" not in globals():
    logger: logging.Logger = (  # type: ignore[no-redef]
        get_logger("main_gui_controller")
        if "get_logger" in globals()
        else logging.getLogger("main_gui_controller")
    )


class MainGUIController:
    """Main controller for the APGI Framework GUI.

    Orchestrates all GUI components, manages state, and provides
    centralized communication between components.
    """

    def __init__(self, parent_widget: Any) -> None:
        """Initialize the main GUI controller.

        Args:
            parent_widget: Parent widget (usually the main window)
        """
        self.parent = parent_widget

        # Framework components
        self.config_manager = ConfigManager()
        self.framework_controller = MainApplicationController()

        # GUI components
        self.parameter_panel: Optional[Any] = None
        self.test_panels: Dict[str, Any] = {}  # Dictionary of test execution panels
        self.results_panel: Optional[Any] = None
        self.logging_panel: Optional[Any] = None

        # State management
        self.current_test: Optional[str] = None
        self.test_results: Dict[str, Any] = {}
        self.system_initialized: bool = False

        # Communication channels
        self.message_queue: queue.Queue = queue.Queue()
        self.progress_callbacks: list[Callable[..., Any]] = []
        self.log_callbacks: list[Callable[..., Any]] = []
        self.results_callbacks: list[Callable[..., Any]] = []

        # Event handlers
        self.event_handlers: dict[str, Callable[..., Any]] = {}

        # Initialize components
        self._setup_event_handlers()
        self._initialize_components()

        # Check framework availability
        if not FRAMEWORK_COMPONENTS_AVAILABLE:
            messagebox.showerror(
                "Framework Not Available",
                "The APGI Framework components are not properly installed.\n\n"
                "Please ensure the framework is correctly installed before running the GUI.\n\n"
                "Missing components will cause limited functionality.",
                parent=self.parent,
            )
            logger.warning("GUI initialized with missing framework components")

        logger.info("MainGUIController initialized")

    def _initialize_components(self) -> None:
        """Initialize all GUI components."""
        if not GUI_COMPONENTS_AVAILABLE:
            logger.warning("GUI components not available - using placeholder panels")
            # Initialize placeholder panels
            self.parameter_panel = ParameterConfigPanel(self.parent)
            self.results_panel = ResultsVisualizationPanel(self.parent)
            self.logging_panel = LoggingPanel(self.parent)
            self.system_initialized = True
            return

        try:
            # Initialize parameter configuration panel
            self.parameter_panel = ParameterConfigPanel(self.parent, self.config_manager)
            self._setup_parameter_panel_callbacks()

            # Initialize results visualization panel
            self.results_panel = ResultsVisualizationPanel(
                self.parent, self._get_results_callback()
            )
            self._setup_results_panel_callbacks()

            # Initialize logging panel
            self.logging_panel = LoggingPanel(self.parent)
            self._setup_logging_panel_callbacks()

            # Initialize test panels (lazy loading)
            self._initialize_test_panels()

            logger.info("All GUI components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize GUI components: {e}")
            raise

    def _initialize_test_panels(self) -> None:
        """Initialize test execution panels."""
        test_names = [
            "Primary",
            "Consciousness Without Ignition",
            "Threshold Insensitivity",
            "Soma-Bias",
            "Cross-Species Validation",
            "Clinical Biomarkers",
            "Threshold Detection",
        ]

        for test_name in test_names:
            try:
                panel = TestExecutionPanel(
                    self.parent,
                    test_name,
                    self.framework_controller,
                    self._get_progress_callback(),
                    self._get_log_callback(),
                    self._get_results_callback(),
                )
                self.test_panels[test_name] = panel
                self._setup_test_panel_callbacks(test_name, panel)

            except Exception as e:
                logger.error(f"Failed to initialize test panel for {test_name}: {e}")

    def _setup_event_handlers(self) -> None:
        """Setup event handlers for component communication."""
        # Parameter panel events
        self.event_handlers["config_changed"] = self._on_config_changed
        self.event_handlers["config_saved"] = self._on_config_saved
        self.event_handlers["config_loaded"] = self._on_config_loaded

        # Test panel events
        self.event_handlers["test_started"] = self._on_test_started
        self.event_handlers["test_completed"] = self._on_test_completed
        self.event_handlers["test_failed"] = self._on_test_failed
        self.event_handlers["progress_updated"] = self._on_progress_updated

        # Results panel events
        self.event_handlers["data_updated"] = self._on_results_updated
        self.event_handlers["export_requested"] = self._on_export_requested

        # Logging panel events
        self.event_handlers["log_added"] = self._on_log_added
        self.event_handlers["log_cleared"] = self._on_log_cleared

    def _setup_parameter_panel_callbacks(self) -> None:
        """Setup callbacks for parameter configuration panel."""
        if self.parameter_panel:
            self.parameter_panel.set_config_changed_callback(self.event_handlers["config_changed"])
            self.parameter_panel.set_validation_error_callback(self._on_validation_error)

    def _setup_results_panel_callbacks(self) -> None:
        """Setup callbacks for results visualization panel."""
        if self.results_panel:
            self.results_panel.set_data_updated_callback(self.event_handlers["data_updated"])
            self.results_panel.set_export_requested_callback(
                self.event_handlers["export_requested"]
            )

    def _setup_logging_panel_callbacks(self) -> None:
        """Setup callbacks for logging panel."""
        if self.logging_panel:
            self.logging_panel.set_log_added_callback(self.event_handlers["log_added"])
            self.logging_panel.set_log_cleared_callback(self.event_handlers["log_cleared"])

    def _setup_test_panel_callbacks(self, test_name: str, panel: Any) -> None:
        """Setup callbacks for a test execution panel."""
        panel.set_test_started_callback(lambda name: self.event_handlers["test_started"](name))
        panel.set_test_completed_callback(self._on_test_completed)
        panel.set_test_failed_callback(self._on_test_failed)
        panel.set_progress_updated_callback(self._on_progress_updated)

    def _get_progress_callback(self) -> Callable[[float, str], None]:
        """Get progress callback for test panels."""

        def progress_callback(progress: float, message: str) -> None:
            """Handle progress updates from test panels."""
            self._broadcast_progress(progress, message)

        return progress_callback

    def _get_log_callback(self) -> Callable[[str], None]:
        """Get log callback for test panels."""

        def log_callback(message: str) -> None:
            """Handle log messages from test panels."""
            self._broadcast_log(message)

        return log_callback

    def _get_results_callback(self) -> Callable[[str, Any], None]:
        """Get results callback for test panels."""

        def results_callback(test_name: str, results: Any) -> None:
            """Handle results from test panels."""
            self._handle_test_results(test_name, results)

        return results_callback

    # Event handlers
    def _on_config_changed(self) -> None:
        """Handle configuration change event."""
        logger.info("Configuration changed")
        self._broadcast_event("config_changed", {})

    def _on_config_saved(self) -> None:
        """Handle configuration saved event."""
        logger.info("Configuration saved")
        self._broadcast_event("config_saved", {})
        self._broadcast_log("Configuration saved successfully")

    def _on_config_loaded(self) -> None:
        """Handle configuration loaded event."""
        logger.info("Configuration loaded")
        self._broadcast_event("config_loaded", {})
        self._broadcast_log("Configuration loaded successfully")

    def _on_test_started(self, test_name: str) -> None:
        """Handle test started event."""
        logger.info(f"Test started: {test_name}")
        self.current_test = test_name
        self._broadcast_event("test_started", {"test_name": test_name})
        self._broadcast_log(f"Started {test_name} test")

    def _on_test_completed(self, test_name: str, results: Any) -> None:
        """Handle test completed event."""
        logger.info(f"Test completed: {test_name}")
        self.current_test = None
        self._broadcast_event("test_completed", {"test_name": test_name, "results": results})
        self._broadcast_log(f"Completed {test_name} test successfully")

    def _on_test_failed(self, test_name: str, error: str) -> None:
        """Handle test failed event."""
        logger.error(f"Test failed: {test_name} - {error}")
        self.current_test = None
        self._broadcast_event("test_failed", {"test_name": test_name, "error": error})
        self._broadcast_log(f"Failed {test_name} test: {error}", "ERROR")

    def _on_progress_updated(self, test_name: str, progress: float, message: str) -> None:
        """Handle progress update event."""
        self._broadcast_event(
            "progress_updated",
            {"test_name": test_name, "progress": progress, "message": message},
        )

    def _on_results_updated(self, results_data: List[Dict[str, Any]]) -> None:
        """Handle results data updated event."""
        logger.info(f"Results updated: {len(results_data)} entries")
        self._broadcast_event("results_updated", {"data": results_data})

    def _on_export_requested(self, file_path: str) -> None:
        """Handle export requested event."""
        logger.info(f"Export requested: {file_path}")
        self._broadcast_event("export_requested", {"file_path": file_path})

    def _on_log_added(self, message: str) -> None:
        """Handle log added event."""
        # This is already handled by the logging panel itself
        pass

    def _on_log_cleared(self) -> None:
        """Handle log cleared event."""
        logger.info("Logs cleared")
        self._broadcast_event("logs_cleared", {})

    def _on_validation_error(self, param_name: str, error: str) -> None:
        """Handle parameter validation error."""
        logger.warning(f"Validation error for {param_name}: {error}")
        self._broadcast_log(f"Parameter validation error: {param_name} - {error}", "WARNING")

    def _handle_test_results(self, test_name: str, results: Any) -> None:
        """Handle test results from test panels."""
        self.test_results[test_name] = {"timestamp": datetime.now(), "results": results}

        # Add to results panel
        if self.results_panel:
            self.results_panel.add_results(test_name, results)

    def _broadcast_progress(self, progress: float, message: str) -> None:
        """Broadcast progress update to all listeners."""
        for callback in self.progress_callbacks:
            try:
                callback(progress, message)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def _broadcast_log(self, message: str, level: str = "INFO") -> None:
        """Broadcast log message to all listeners."""
        for callback in self.log_callbacks:
            try:
                callback(message, level)
            except Exception as e:
                logger.error(f"Error in log callback: {e}")

    def _broadcast_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast event to all listeners."""
        for callback in self.results_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    # Public API methods
    def get_parameter_panel(self) -> ParameterConfigPanel:
        """Get the parameter configuration panel."""
        return cast(ParameterConfigPanel, self.parameter_panel)

    def get_test_panel(self, test_name: str) -> Optional[TestExecutionPanel]:
        """Get a specific test execution panel."""
        return self.test_panels.get(test_name)

    def get_all_test_panels(self) -> Dict[str, TestExecutionPanel]:
        """Get all test execution panels."""
        return self.test_panels.copy()

    def get_results_panel(self) -> ResultsVisualizationPanel:
        """Get the results visualization panel."""
        return cast(ResultsVisualizationPanel, self.results_panel)

    def get_logging_panel(self) -> LoggingPanel:
        """Get the logging panel."""
        return cast(LoggingPanel, self.logging_panel)

    def get_current_test(self) -> Optional[str]:
        """Get the currently running test."""
        return self.current_test

    def get_test_results(self, test_name: Optional[str] = None) -> Dict[str, Any]:
        """Get test results."""
        if test_name:
            return cast(Dict[str, Any], self.test_results.get(test_name))
        return self.test_results.copy()

    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        try:
            if self.parameter_panel:
                result = self.parameter_panel._get_configuration_dict()
                return result if isinstance(result, dict) else {}
            return {}
        except Exception as e:
            logger.error(f"Failed to get current config: {e}")
            return {}

    def save_configuration(self, file_path: Optional[str] = None) -> bool:
        """Save current configuration."""
        try:
            if self.parameter_panel:
                if file_path:
                    # Save to specific file
                    config = self.parameter_panel._get_configuration_dict()
                    import json

                    with open(file_path, "w") as f:
                        json.dump(config, f, indent=2)
                    return True
                else:
                    # Use panel's save method
                    self.parameter_panel._save_configuration()
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def load_configuration(self, file_path: Optional[str] = None) -> bool:
        """Load configuration."""
        try:
            if self.parameter_panel:
                if file_path:
                    # Load from specific file
                    import json

                    with open(file_path, "r") as f:
                        config = json.load(f)
                    self.parameter_panel._apply_configuration(config)
                    return True
                else:
                    # Use panel's load method
                    self.parameter_panel._load_configuration()
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    def run_test(self, test_name: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """Run a specific test."""
        try:
            test_panel = self.get_test_panel(test_name)
            if not test_panel:
                logger.error(f"Test panel not found: {test_name}")
                return False

            if test_panel.is_test_running():
                logger.warning(f"Test already running: {test_name}")
                return False

            # Apply parameters if provided
            if parameters:
                for param_name, value in parameters.items():
                    if param_name in test_panel.test_vars:
                        test_panel.test_vars[param_name].set(value)

            # Start the test
            test_panel._run_test()  # type: ignore[no-untyped-call]
            return True

        except Exception as e:
            logger.error(f"Failed to run test {test_name}: {e}")
            return False

    def stop_test(self, test_name: Optional[str] = None) -> bool:
        """Stop a running test."""
        try:
            if test_name:
                test_panel = self.get_test_panel(test_name)
                if test_panel and test_panel.is_test_running():
                    test_panel._stop_test()  # type: ignore[no-untyped-call]
                    return True
            else:
                # Stop current test
                if self.current_test:
                    test_panel = self.get_test_panel(self.current_test)
                    if test_panel:
                        test_panel._stop_test()  # type: ignore[no-untyped-call]
                        return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop test: {e}")
            return False

    def reset_all_tests(self) -> bool:
        """Reset all test panels."""
        try:
            for test_name, test_panel in self.test_panels.items():
                test_panel._reset_test()
            logger.info("All test panels reset")
            return True
        except Exception as e:
            logger.error(f"Failed to reset tests: {e}")
            return False

    def clear_all_results(self) -> bool:
        """Clear all results data."""
        try:
            if self.results_panel:
                self.results_panel.clear_results()
            self.test_results.clear()
            logger.info("All results cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear results: {e}")
            return False

    def export_all_results(self, file_path: str) -> bool:
        """Export all results to file."""
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "configuration": self.get_current_config(),
                "test_results": self.test_results,
                "log_statistics": (
                    self.logging_panel.get_log_statistics() if self.logging_panel else {}
                ),
            }

            import json

            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"All results exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return False

    def add_progress_callback(self, callback: Callable[..., Any]) -> None:
        """Add a progress update callback."""
        self.progress_callbacks.append(callback)

    def add_log_callback(self, callback: Callable[..., Any]) -> None:
        """Add a log message callback."""
        self.log_callbacks.append(callback)

    def add_results_callback(self, callback: Callable[..., Any]) -> None:
        """Add a results update callback."""
        self.results_callbacks.append(callback)

    def remove_progress_callback(self, callback: Callable[..., Any]) -> None:
        """Remove a progress update callback."""
        if callback in self.progress_callbacks:
            self.progress_callbacks.remove(callback)

    def remove_log_callback(self, callback: Callable[..., Any]) -> None:
        """Remove a log message callback."""
        if callback in self.log_callbacks:
            self.log_callbacks.remove(callback)

    def remove_results_callback(self, callback: Callable[..., Any]) -> None:
        """Remove a results update callback."""
        if callback in self.results_callbacks:
            self.results_callbacks.remove(callback)

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        status = {
            "initialized": self.system_initialized,
            "current_test": self.current_test,
            "total_tests": len(self.test_panels),
            "running_tests": sum(
                1 for panel in self.test_panels.values() if panel.is_test_running()
            ),
            "total_results": len(self.test_results),
            "log_statistics": (
                self.logging_panel.get_log_statistics() if self.logging_panel else {}
            ),
        }

        return status

    def shutdown(self) -> None:
        """Shutdown the GUI controller."""
        try:
            # Stop any running tests
            if self.current_test:
                self.stop_test()

            # Save current configuration
            self.save_configuration()

            # Clear resources
            self.test_results.clear()
            self.progress_callbacks.clear()
            self.log_callbacks.clear()
            self.results_callbacks.clear()

            logger.info("GUI controller shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def refresh_all_components(self) -> None:
        """Refresh all GUI components."""
        try:
            # Refresh results panel
            if self.results_panel:
                self.results_panel.refresh_visualization()

            # Refresh logging panel statistics
            if self.logging_panel:
                self.logging_panel._update_statistics_display()

            logger.info("All components refreshed")

        except Exception as e:
            logger.error(f"Failed to refresh components: {e}")

    def validate_system(self) -> ValidationResult:
        """Validate system state and return validation results."""
        issues: List[str] = []
        warnings: List[str] = []
        validation_results: ValidationResult = {
            "valid": True,
            "issues": issues,
            "warnings": warnings,
        }

        try:
            # Check parameter panel
            if self.parameter_panel:
                try:
                    # Basic validation would go here
                    pass
                except Exception as e:
                    issues.append(f"Parameter panel error: {e}")
                    validation_results["valid"] = False

            # Check test panels
            for test_name, panel in self.test_panels.items():
                try:
                    if hasattr(panel, "is_test_running") and panel.is_test_running():
                        warnings.append(f"Test {test_name} is still running")
                except Exception as e:
                    issues.append(f"Error checking test {test_name}: {e}")
                    validation_results["valid"] = False

            # Check results panel
            if self.results_panel:
                try:
                    results_data = self.results_panel.get_results_data()
                    if hasattr(results_data, "__len__") and len(results_data) > 1000:
                        warnings.append("Large amount of results data may affect performance")
                except Exception as e:
                    issues.append(f"Error checking results panel: {e}")
                    validation_results["valid"] = False

            # Check logging panel
            if self.logging_panel:
                try:
                    stats = self.logging_panel.get_log_statistics()
                    if isinstance(stats, dict) and stats.get("ERROR", 0) > 10:
                        warnings.append("High number of error logs detected")
                except Exception as e:
                    issues.append(f"Error checking logging panel: {e}")
                    validation_results["valid"] = False

        except Exception as e:
            issues.append(f"Validation error: {e}")
            validation_results["valid"] = False

        return validation_results


# Factory function for easy instantiation
def create_main_gui_controller(parent_widget: Any) -> MainGUIController:
    """Create a main GUI controller with default settings.

    Args:
        parent_widget: Parent widget (usually the main window)

    Returns:
        Configured MainGUIController instance
    """
    return MainGUIController(parent_widget)
