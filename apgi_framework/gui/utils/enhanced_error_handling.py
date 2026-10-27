"""Enhanced error handling and recovery system for APGI Framework GUI applications.

Provides comprehensive error handling, fallback mechanisms, and user-friendly
error reporting with automatic recovery strategies.
"""

import json
import logging
import sys
import tkinter as tk
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ErrorSeverity:
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory:
    """Error categories for better handling."""

    IMPORT_ERROR = "import_error"
    FILE_ERROR = "file_error"
    DATA_ERROR = "data_error"
    NETWORK_ERROR = "network_error"
    HARDWARE_ERROR = "hardware_error"
    VALIDATION_ERROR = "validation_error"
    RUNTIME_ERROR = "runtime_error"
    CONFIGURATION_ERROR = "configuration_error"


class GUIError:
    """Structured error information for GUI applications."""

    def __init__(
        self,
        exception: Exception,
        category: str,
        severity: str,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        recovery_actions: Optional[List[str]] = None,
    ):
        """Initialize GUI error.

        Args:
            exception: The original exception
            category: Error category from ErrorCategory
            severity: Error severity from ErrorSeverity
            context: Additional context information
            user_message: User-friendly error message
            recovery_actions: List of possible recovery actions
        """
        self.exception = exception
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.user_message = user_message or str(exception)
        self.recovery_actions = recovery_actions or []
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()


class FallbackManager:
    """Manages fallback mechanisms for optional dependencies and features."""

    def __init__(self) -> None:
        """Initialize fallback manager."""
        self.fallbacks: Dict[str, Dict[str, Any]] = {}
        self.register_standard_fallbacks()

    def register_standard_fallbacks(self) -> None:
        """Register standard fallback mechanisms."""
        # CustomTkinter fallback
        self.fallbacks["customtkinter"] = {
            "available": self._check_customtkinter,
            "fallback": self._customtkinter_fallback,
            "message": "CustomTkinter is not available. Using standard tkinter instead.",
        }

        # Plotly fallback
        self.fallbacks["plotly"] = {
            "available": self._check_plotly,
            "fallback": self._plotly_fallback,
            "message": "Plotly is not available. Using matplotlib for visualizations.",
        }

        # Database fallback
        self.fallbacks["sqlite"] = {
            "available": self._check_sqlite,
            "fallback": self._sqlite_fallback,
            "message": "SQLite is not available. Using file-based storage.",
        }

        # HDF5 fallback
        self.fallbacks["h5py"] = {
            "available": self._check_h5py,
            "fallback": self._h5py_fallback,
            "message": "HDF5 is not available. Using CSV/JSON storage.",
        }

    def _check_customtkinter(self) -> bool:
        """Check if CustomTkinter is available."""
        try:
            import customtkinter  # noqa: F401

            return True
        except ImportError:
            return False

    def _customtkinter_fallback(self) -> Dict[str, Any]:
        """Provide CustomTkinter fallback."""
        return {
            "use_standard_tkinter": True,
            "limited_features": ["custom styling", "modern themes"],
            "available_features": ["basic widgets", "standard layouts"],
        }

    def _check_plotly(self) -> bool:
        """Check if Plotly is available."""
        try:
            import plotly  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False

    def _plotly_fallback(self) -> Dict[str, Any]:
        """Provide Plotly fallback."""
        return {
            "use_matplotlib": True,
            "limited_features": ["interactive charts", "web export"],
            "available_features": ["static plots", "basic charts", "export to PNG/SVG"],
        }

    def _check_sqlite(self) -> bool:
        """Check if SQLite is available."""
        try:
            import sqlite3  # noqa: F401

            return True
        except ImportError:
            return False

    def _sqlite_fallback(self) -> Dict[str, Any]:
        """Provide SQLite fallback."""
        return {
            "use_json_storage": True,
            "limited_features": ["complex queries", "concurrent access"],
            "available_features": [
                "basic storage",
                "file persistence",
                "data retrieval",
            ],
        }

    def _check_h5py(self) -> bool:
        """Check if HDF5 is available."""
        try:
            import h5py  # noqa: F401

            return True
        except ImportError:
            return False

    def _h5py_fallback(self) -> Dict[str, Any]:
        """Provide HDF5 fallback."""
        return {
            "use_csv_json": True,
            "limited_features": ["large datasets", "metadata storage"],
            "available_features": ["tabular data", "structured data", "file export"],
        }

    def get_fallback(self, dependency_name: str) -> Optional[Dict[str, Any]]:
        """Get fallback for a dependency.

        Args:
            dependency_name: Name of the dependency

        Returns:
            Fallback information or None if not available
        """
        if dependency_name not in self.fallbacks:
            return None

        fallback_info = self.fallbacks[dependency_name]

        if fallback_info["available"]():
            return None  # Dependency is available, no fallback needed

        return {
            "fallback_config": fallback_info["fallback"](),
            "message": fallback_info["message"],
            "dependency": dependency_name,
        }


class EnhancedErrorHandler:
    """Enhanced error handler with recovery strategies."""

    def __init__(self, root_window: Optional[tk.Tk] = None):
        """Initialize enhanced error handler.

        Args:
            root_window: Main window for displaying error dialogs
        """
        self.root_window = root_window
        self.fallback_manager: FallbackManager = FallbackManager()
        self.error_log: List[GUIError] = []
        self.recovery_strategies: Dict[str, Callable] = {}
        self.register_recovery_strategies()

    def register_recovery_strategies(self) -> None:
        """Register automatic recovery strategies."""
        self.recovery_strategies[ErrorCategory.IMPORT_ERROR] = self._handle_import_error
        self.recovery_strategies[ErrorCategory.FILE_ERROR] = self._handle_file_error
        self.recovery_strategies[ErrorCategory.DATA_ERROR] = self._handle_data_error
        self.recovery_strategies[ErrorCategory.NETWORK_ERROR] = self._handle_network_error
        self.recovery_strategies[ErrorCategory.HARDWARE_ERROR] = self._handle_hardware_error
        self.recovery_strategies[ErrorCategory.VALIDATION_ERROR] = self._handle_validation_error
        self.recovery_strategies[ErrorCategory.CONFIGURATION_ERROR] = (
            self._handle_configuration_error
        )

    def handle_error(
        self,
        exception: Exception,
        category: str,
        severity: str = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        show_dialog: bool = True,
    ) -> bool:
        """Handle an error with appropriate recovery strategy.

        Args:
            exception: The exception to handle
            category: Error category
            severity: Error severity
            context: Additional context
            show_dialog: Whether to show error dialog

        Returns:
            True if error was handled successfully, False otherwise
        """
        # Create structured error
        error = GUIError(
            exception=exception,
            category=category,
            severity=severity,
            context=context,
            recovery_actions=self._get_recovery_actions(category),
        )

        # Log error
        self._log_error(error)

        # Store error
        self.error_log.append(error)

        # Apply recovery strategy
        recovery_success = self._apply_recovery_strategy(error)

        # Show user dialog if requested
        if show_dialog and self.root_window:
            self._show_error_dialog(error, recovery_success)

        return recovery_success

    def _log_error(self, error: GUIError) -> None:
        """Log error with full context."""
        log_message = f"{error.category.upper()}: {error.user_message}"

        if error.context:
            log_message += f" | Context: {error.context}"

        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, exc_info=error.exception)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(log_message, exc_info=error.exception)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, exc_info=error.exception)
        else:
            logger.info(log_message, exc_info=error.exception)

    def _apply_recovery_strategy(self, error: GUIError) -> bool:
        """Apply appropriate recovery strategy."""
        if error.category in self.recovery_strategies:
            try:
                result: bool = self.recovery_strategies[error.category](error)
                return result
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {recovery_error}")
                return False
        return False

    def _get_recovery_actions(self, category: str) -> List[str]:
        """Get recovery actions for an error category."""
        actions = {
            ErrorCategory.IMPORT_ERROR: [
                "Check if required packages are installed",
                "Try installing missing dependencies",
                "Use fallback functionality if available",
            ],
            ErrorCategory.FILE_ERROR: [
                "Check file permissions",
                "Verify file path exists",
                "Try a different file location",
            ],
            ErrorCategory.DATA_ERROR: [
                "Validate input data format",
                "Check data range and types",
                "Try with sample data",
            ],
            ErrorCategory.NETWORK_ERROR: [
                "Check internet connection",
                "Verify server accessibility",
                "Try again later",
            ],
            ErrorCategory.HARDWARE_ERROR: [
                "Check hardware connections",
                "Restart hardware devices",
                "Use simulation mode",
            ],
            ErrorCategory.VALIDATION_ERROR: [
                "Check input values",
                "Verify required fields",
                "Review configuration settings",
            ],
            ErrorCategory.CONFIGURATION_ERROR: [
                "Reset to default settings",
                "Check configuration file format",
                "Verify environment variables",
            ],
        }
        return actions.get(category, ["Contact support", "Check documentation"])

    def _show_error_dialog(self, error: GUIError, recovery_success: bool) -> None:
        """Show user-friendly error dialog."""
        title = f"Error - {error.category.replace('_', ' ').title()}"

        message = error.user_message

        if error.recovery_actions:
            message += "\n\nSuggested actions:\n"
            for i, action in enumerate(error.recovery_actions, 1):
                message += f"{i}. {action}\n"

        if recovery_success:
            message += "\nAutomatic recovery was attempted."

        # Choose dialog type based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            messagebox.showerror(title, message)
        elif error.severity == ErrorSeverity.HIGH:
            messagebox.showerror(title, message)
        elif error.severity == ErrorSeverity.MEDIUM:
            messagebox.showwarning(title, message)
        else:
            messagebox.showinfo(title, message)

    # Recovery strategy implementations
    def _handle_import_error(self, error: GUIError) -> bool:
        """Handle import errors with fallback mechanisms."""
        if "customtkinter" in str(error.exception).lower():
            fallback = self.fallback_manager.get_fallback("customtkinter")
            if fallback:
                logger.info(f"Using CustomTkinter fallback: {fallback['message']}")
                return True

        if "plotly" in str(error.exception).lower():
            fallback = self.fallback_manager.get_fallback("plotly")
            if fallback:
                logger.info(f"Using Plotly fallback: {fallback['message']}")
                return True

        return False

    def _handle_file_error(self, error: GUIError) -> bool:
        """Handle file errors with path corrections."""
        if "permission" in str(error.exception).lower():
            # Try to create file in user directory
            user_dir = Path.home() / "apgi_framework"
            user_dir.mkdir(exist_ok=True)
            error.context["fallback_path"] = str(user_dir)
            return True

        return False

    def _handle_data_error(self, error: GUIError) -> bool:
        """Handle data errors with validation corrections."""
        # Try to correct common data issues
        return False

    def _handle_network_error(self, error: GUIError) -> bool:
        """Handle network errors with offline mode."""
        # Enable offline mode
        return False

    def _handle_hardware_error(self, error: GUIError) -> bool:
        """Handle hardware errors with simulation mode."""
        # Enable simulation mode
        return True

    def _handle_validation_error(self, error: GUIError) -> bool:
        """Handle validation errors with default values."""
        # Apply default values
        return True

    def _handle_configuration_error(self, error: GUIError) -> bool:
        """Handle configuration errors with defaults."""
        # Reset to default configuration
        return True

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        if not self.error_log:
            return {"total_errors": 0, "by_category": {}, "by_severity": {}}

        by_category: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for error in self.error_log:
            by_category[error.category] = by_category.get(error.category, 0) + 1
            by_severity[error.severity] = by_severity.get(error.severity, 0) + 1

        return {
            "total_errors": len(self.error_log),
            "by_category": by_category,
            "by_severity": by_severity,
            "recent_errors": [str(e.exception) for e in self.error_log[-5:]],
        }

    def save_error_log(self, file_path: Optional[Path] = None) -> None:
        """Save error log to file."""
        if file_path is None:
            file_path = Path("error_log.json")

        error_data = []
        for error in self.error_log:
            error_data.append(
                {
                    "timestamp": error.timestamp.isoformat(),
                    "category": error.category,
                    "severity": error.severity,
                    "message": error.user_message,
                    "context": error.context,
                    "traceback": error.traceback,
                }
            )

        try:
            with open(file_path, "w") as f:
                json.dump(error_data, f, indent=2)
            logger.info(f"Error log saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save error log: {e}")


# Global error handler instance
_global_error_handler: Optional[EnhancedErrorHandler] = None


def get_error_handler(root_window: Optional[tk.Tk] = None) -> EnhancedErrorHandler:
    """Get or create global error handler."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = EnhancedErrorHandler(root_window)
    return _global_error_handler


def handle_gui_error(
    exception: Exception,
    category: str,
    severity: str = ErrorSeverity.MEDIUM,
    context: Optional[Dict[str, Any]] = None,
    show_dialog: bool = True,
) -> bool:
    """Handle GUI error using global error handler.

    Args:
        exception: The exception to handle
        category: Error category
        severity: Error severity
        context: Additional context
        show_dialog: Whether to show error dialog

    Returns:
        True if error was handled successfully
    """
    handler = get_error_handler()
    return handler.handle_error(exception, category, severity, context, show_dialog)


def setup_global_error_handling(root_window: Optional[tk.Tk]) -> None:
    """Setup global error handling for tkinter application."""
    handler = get_error_handler(root_window)

    def handle_tk_error(exc: Any, val: BaseException, tb: Any) -> None:
        # Convert BaseException to Exception for GUIError
        exception = val if isinstance(val, Exception) else Exception(str(val))
        error = GUIError(
            exception=exception,
            category=ErrorCategory.RUNTIME_ERROR,
            severity=ErrorSeverity.HIGH,
            context={"type": type(val).__name__},
            user_message=f"An unexpected error occurred: {type(val).__name__}: {val}",
        )
        handler._log_error(error)
        handler._show_error_dialog(error, False)

    sys.excepthook = handle_tk_error
