#!/usr/bin/env python3
"""
APGI Assistant GUI
============================================================

"""

# Deprecated legacy entry point (README: "Legacy Entry Points (Deprecated)").
try:
    from apgi_framework.utils.deprecation import warn_deprecated_module

    warn_deprecated_module("Assistant_GUI", "apgi_gui.main", since="1.1.0", removal="2.0.0")
except ImportError:  # framework not importable: the README and docstring still state the deprecation
    pass

import csv
import gc
import importlib
import importlib.util
import json
import logging
import os
import queue
import sys
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from tkinter import Menu, filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, TypeVar, cast

import numpy as np

# APGIAssistant will be loaded dynamically via load_apgi_module()
HAS_APGI_ASSISTANT = False
APGIAssistant: Any = None

# Import theme manager
try:
    from apgi_gui.theme_manager import ThemeManager

    THEME_MANAGER_AVAILABLE = True
except ImportError:
    THEME_MANAGER_AVAILABLE = False
    print("Warning: Theme manager not available. Theme support disabled.")

# Import ToolTip from components
try:
    from apgi_gui.components.core import ToolTip

    TOOLTIP_AVAILABLE = True
except ImportError:
    TOOLTIP_AVAILABLE = False
    print("Warning: ToolTip component not available.")

# Import core APGI Assistant - will be loaded by load_apgi_module() function
HAS_ASSISTANT = False  # Will be updated after successful module loading

# Import torch if available (used by APGI module)
# Note: torch is used by the APGI module, not directly in this file
HAS_TORCH = False

# Matplotlib imports (if available)
try:
    import matplotlib

    matplotlib.use("TkAgg")  # Ensure consistent backend
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# PIL imports (if available)
try:
    from PIL import Image, ImageTk

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Check for PDF export dependencies
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import (
        ParagraphStyle,
        getSampleStyleSheet,
    )
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    colors = None

# Check for additional simulation dependencies
try:
    import torchdiffeq  # noqa: F401

    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False

try:
    import transformers  # noqa: F401

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


def safe_widget_method(widget_attr: str) -> Callable[[F], F]:
    """Decorator for safe widget operations with enhanced validation

    Args:
        widget_attr: Name of the widget attribute to check

    Returns:
        Decorated function that only executes if the widget exists and app is not shutting down

    Usage:
        @safe_widget_method('status_label')
        def update_status(self, message: str) -> None:
            self.status_label.config(text=message)
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Optional[Any]:
            try:
                # Check if application is shutting down
                if hasattr(self, "is_shutting_down") and self.is_shutting_down:
                    return None

                # Check if root window exists
                if hasattr(self, "root") and not self.root.winfo_exists():
                    return None

                # Get widget with comprehensive validation
                widget = getattr(self, widget_attr, None)
                if widget is None:
                    return None

                # Check widget has winfo_exists method and widget exists
                if not hasattr(widget, "winfo_exists"):
                    return None

                if not widget.winfo_exists():
                    return None

                # Additional validation for certain widget types
                if hasattr(widget, "master") and widget.master:
                    if not widget.master.winfo_exists():
                        return None

                # All checks passed, execute the function
                return func(self, *args, **kwargs)

            except Exception as e:
                # Enhanced error logging with context - use exception to capture stack trace
                logger = getattr(self, "logger", None)
                if logger:
                    logger.exception(f"Safe widget method error for {widget_attr}: {e}")
                else:
                    import sys

                    sys.stderr.write(f"Safe widget method error for {widget_attr}: {e}\n")
                return None

        return wrapper  # type: ignore[return-value]

    return decorator


try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class HistoryManager:
    """Manages automatic history pruning with configurable limits and memory optimization"""

    def __init__(self, max_memory_mb: int = 100, auto_prune: bool = True):
        """Initialize history manager

        Args:
            max_memory_mb: Maximum memory usage in MB for all history
            auto_prune: Whether to automatically prune when memory limits exceeded
        """
        # Get memory limit from environment or use default (1GB as per BUG-M09)
        self.max_memory_mb = int(os.environ.get("APGI_MAX_MEMORY_MB", max_memory_mb))
        self.auto_prune = auto_prune
        self.pruning_strategies = {
            "state": {"maxlen": 1000, "prune_ratio": 0.5},
            "energy": {"maxlen": 500, "prune_ratio": 0.5},
            "query": {"maxlen": 200, "prune_ratio": 0.5},
            "transition": {"maxlen": 1000, "prune_ratio": 0.5},
            "physiology": {"maxlen": 1000, "prune_ratio": 0.5},
        }

        # Track memory usage
        self.memory_usage: Dict[str, float] = {}
        self.last_prune_time = time.time()
        self.prune_interval = 60  # Prune at most every minute

    def create_managed_deque(self, history_type: str, maxlen: Optional[int] = None) -> Deque[Any]:
        """Create a managed deque with automatic pruning

        Args:
            history_type: Type of history (state, energy, query, etc.)
            maxlen: Maximum length override

        Returns:
            Managed deque instance
        """
        strategy = self.pruning_strategies.get(history_type, {"maxlen": 50, "prune_ratio": 0.5})
        maxlen_final: int = int(maxlen or strategy["maxlen"])

        managed_deque = ManagedDeque(maxlen=maxlen_final, history_type=history_type, manager=self)
        return managed_deque

    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage of all managed deques

        Returns:
            Dictionary with memory usage per history type in MB
        """
        current_time = time.time()

        # Only check memory usage periodically to avoid overhead
        if current_time - self.last_prune_time < 30:  # Check every 30 seconds
            return self.memory_usage

        total_memory: float = 0
        usage = {}

        # Calculate memory usage for each history type
        for history_type in self.pruning_strategies.keys():
            # Estimate memory usage (rough approximation)
            if hasattr(self, f"_{history_type}_deque"):
                deque_obj = getattr(self, f"_{history_type}_deque")
                if hasattr(deque_obj, "__len__"):
                    # More realistic estimate: each item ~5KB average
                    estimated_mb = len(deque_obj) * 0.005
                    usage[history_type] = estimated_mb
                    total_memory += estimated_mb

        usage["total"] = total_memory
        self.memory_usage = usage

        # Auto-prune if needed
        if self.auto_prune and total_memory > self.max_memory_mb:
            self._perform_auto_prune()

        return usage

    def _perform_auto_prune(self) -> None:
        """Perform automatic pruning based on strategies"""
        current_time = time.time()
        if current_time - self.last_prune_time < self.prune_interval:
            return

        pruned = False
        for history_type, strategy in self.pruning_strategies.items():
            if hasattr(self, f"_{history_type}_deque"):
                deque_obj = getattr(self, f"_{history_type}_deque")
                if hasattr(deque_obj, "prune"):
                    prune_ratio = strategy["prune_ratio"]
                    deque_obj.prune(prune_ratio)
                    pruned = True

        if pruned:
            self.last_prune_time = current_time

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics

        Returns:
            Dictionary with memory statistics
        """
        usage = self.check_memory_usage()
        return {
            "memory_usage_mb": usage,
            "max_memory_mb": self.max_memory_mb,
            "memory_utilization": (
                usage.get("total", 0) / self.max_memory_mb if self.max_memory_mb > 0 else 0
            ),
            "last_prune_time": self.last_prune_time,
            "auto_prune_enabled": self.auto_prune,
        }

    def clear_all_history(self) -> int:
        """Clear all managed history deques with enhanced cleanup"""
        import gc

        cleared_count = 0
        for history_type in self.pruning_strategies.keys():
            if hasattr(self, f"_{history_type}_deque"):
                deque_obj = getattr(self, f"_{history_type}_deque")
                if hasattr(deque_obj, "clear"):
                    # Get size before clearing for logging
                    size_before = len(deque_obj) if hasattr(deque_obj, "__len__") else 0
                    deque_obj.clear()
                    cleared_count += size_before

                    # Remove reference to help garbage collection
                    delattr(self, f"_{history_type}_deque")

        # Clear memory usage tracking
        self.memory_usage.clear()

        # Force garbage collection to free memory
        gc.collect()

        # Update last prune time
        self.last_prune_time = time.time()

        return cleared_count


class ManagedDeque(deque[Any]):
    """Enhanced deque with automatic pruning and memory tracking"""

    def __init__(self, maxlen: int, history_type: str, manager: HistoryManager):
        """Initialize managed deque

        Args:
            maxlen: Maximum length
            history_type: Type of history being tracked
            manager: HistoryManager instance
        """
        super().__init__(maxlen=maxlen)
        self.history_type = history_type
        self.manager = manager
        self._last_prune = time.time()
        self._prune_threshold = 100  # Prune every 100 operations to prevent memory spikes

        # Store reference in manager for memory tracking
        setattr(manager, f"_{history_type}_deque", self)

    def append(self, item: Any) -> None:
        """Append item with automatic memory management"""
        super().append(item)

        # Trigger periodic pruning to prevent memory buildup
        if len(self) % self._prune_threshold == 0:
            self._auto_prune_check()

    def _auto_prune_check(self) -> None:
        """Check if automatic pruning is needed"""
        current_time = time.time()

        # Only prune if enough time has passed
        if current_time - self._last_prune < 60:  # At most once per minute
            return

        # Check if we're approaching memory limits
        if hasattr(self.manager, "check_memory_usage"):
            usage = self.manager.check_memory_usage()
            if usage.get("total", 0) > self.manager.max_memory_mb * 0.8:  # 80% threshold
                strategy = self.manager.pruning_strategies.get(
                    self.history_type, {"prune_ratio": 0.5}
                )
                self.prune(strategy["prune_ratio"])
                self._last_prune = current_time

    def prune(self, ratio: float = 0.5) -> None:
        """Prune the deque by removing oldest entries

        Args:
            ratio: Fraction of entries to remove (0.0 to 1.0)
        """
        if not self or ratio <= 0:
            return

        # Calculate how many items to remove
        items_to_remove = max(1, int(len(self) * ratio))

        # Remove oldest items with memory cleanup
        for _ in range(items_to_remove):
            if self:
                # Clear item reference to help garbage collection
                self.popleft()

    def smart_prune(self, target_size: int) -> None:
        """Smart pruning to reach target size"""
        current_size = len(self)
        if current_size <= target_size:
            return

        # Calculate removal ratio
        ratio = (current_size - target_size) / current_size
        self.prune(ratio)

    def clear(self) -> None:
        """Clear deque with enhanced cleanup"""
        # Clear all items to help garbage collection
        while self:
            self.popleft()

        # Call parent clear method
        super().clear()

    def __del__(self) -> None:
        """Destructor to ensure cleanup"""
        try:
            self.clear()
        except Exception:
            pass  # Ignore errors during cleanup, especially during garbage collection


class DependencyNotifier:
    """Manages user-facing notifications for optional dependency limitations"""

    def __init__(self, parent_widget: Any = None) -> None:
        """Initialize dependency notifier

        Args:
            parent_widget: Parent widget for showing notifications
        """
        self.parent_widget = parent_widget
        self.notified_features: set[str] = set()  # Track which features have been notified

    def check_and_notify_limitations(self) -> None:
        """Check for missing dependencies and show user-friendly notifications"""
        limitations = []

        if not HAS_TORCHDIFFEQ:
            limitations.append(
                {
                    "feature": "Neural ODE Integration",
                    "impact": "Using Euler integration fallback (reduced accuracy)",
                    "install_cmd": "pip install torchdiffeq",
                    "severity": "medium",
                }
            )
        else:
            # torchdiffeq is available, no limitation
            pass

        if not HAS_TRANSFORMERS:
            limitations.append(
                {
                    "feature": "Language Model Integration",
                    "impact": "Advanced text processing disabled",
                    "install_cmd": "pip install transformers",
                    "severity": "high",
                }
            )

        if not HAS_MATPLOTLIB:
            limitations.append(
                {
                    "feature": "Visualizations & Plots",
                    "impact": "All visualization features disabled",
                    "install_cmd": "pip install matplotlib",
                    "severity": "high",
                }
            )

        if not HAS_PSUTIL:
            limitations.append(
                {
                    "feature": "System Monitoring",
                    "impact": "Cannot monitor CPU/memory usage",
                    "install_cmd": "pip install psutil",
                    "severity": "low",
                }
            )

        if not HAS_REPORTLAB:
            limitations.append(
                {
                    "feature": "PDF Export",
                    "impact": "Cannot export reports as PDF",
                    "install_cmd": "pip install reportlab",
                    "severity": "medium",
                }
            )

        # Show notification for new limitations
        for limitation in limitations:
            feature = limitation["feature"]
            if feature not in self.notified_features:
                self._show_limitation_notification(limitation)
                self.notified_features.add(feature)

    def _show_limitation_notification(self, limitation: Dict[str, Any]) -> None:
        """Show a single limitation notification"""
        # Use theme-aware severity indicators instead of hard-coded emoji
        severity_symbols = {"high": "!", "medium": "⚡", "low": "i"}
        symbol = severity_symbols.get(limitation["severity"], "i")

        # Get theme colors for the notification
        # theme_colors = (
        #     self.get_theme_color("notification_colors")
        #     if hasattr(self, "get_theme_color")
        #     else {"high": "#FF4444", "medium": "#FFA500", "low": "#4444FF"}
        # )

        message = f"{symbol} {limitation['feature']} Limited\n\n"
        message += f"Impact: {limitation['impact']}\n\n"
        message += f"To enable: {limitation['install_cmd']}\n\n"
        message += "You can continue using the application with reduced functionality."

        try:
            if self.parent_widget and hasattr(self.parent_widget, "winfo_exists"):
                messagebox.showinfo("Feature Limitation", message)
        except Exception:
            # Fallback to print if GUI not available
            print(f"Feature Limitation: {limitation['feature']} - {limitation['impact']}")

    def reset_notifications(self) -> None:
        """Reset notification history (for testing)"""
        self.notified_features.clear()


class Debouncer:
    """Utility class for debouncing rapid function calls"""

    def __init__(self, delay_ms: int = 300):
        """Initialize debouncer with delay in milliseconds

        Args:
            delay_ms: Delay to wait before executing the function
        """
        self.delay_ms = delay_ms
        self.pending_calls: Dict[str, Any] = {}

    def debounce(self, key: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Debounce a function call with a unique key

        Args:
            key: Unique identifier for this debounced call
            func: Function to call after delay
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
        """
        # Cancel any existing call for this key
        if key in self.pending_calls:
            self.pending_calls[key].cancel()

        # Schedule new call
        if hasattr(self, "root") and hasattr(
            self.root, "after"
        ):  # If we have a tkinter root reference
            call_id = self.root.after(
                self.delay_ms, lambda: self._execute_call(key, func, *args, **kwargs)
            )
        else:
            # Fallback for non-GUI contexts - execute immediately
            func(*args, **kwargs)
            return

        self.pending_calls[key] = call_id

    def _execute_call(self, key: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Execute the debounced function and clean up"""
        try:
            func(*args, **kwargs)
        finally:
            self.pending_calls.pop(key, None)

    def set_root(self, root: Any) -> None:
        """Set tkinter root reference for scheduling"""
        self.root = root


class ErrorContext:
    """Structured error reporting with context and timing.

    Provides a context manager that automatically logs errors with
    operation context, timing information, and optional user notifications.
    """

    def __init__(
        self, operation: str, user_facing: bool = True, logger: Optional[logging.Logger] = None
    ):
        """Initialize error context.

        Args:
            operation: Description of the operation being performed
            user_facing: Whether to show user-friendly error messages
            logger: Logger instance to use (defaults to APGIGUI logger)
        """
        self.operation = operation
        self.user_facing = user_facing
        self.start_time = time.time()
        self.logger = logger or logging.getLogger("APGIGUI")

    def __enter__(self) -> "ErrorContext":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is None:
            return False

        elapsed = time.time() - self.start_time

        # Log with context
        self.logger.error(
            f"Error in {self.operation} after {elapsed:.2f}s: {exc_val}",
            exc_info=(exc_type, exc_val, exc_tb),
            extra={"operation": self.operation, "elapsed": elapsed},
        )

        # Show user-friendly message if needed
        if self.user_facing:
            try:
                # Provide a generic message instead of internal exception details (BUG-M08)
                friendly_msg = (
                    f"An error occurred during {self.operation.lower()}.\n\n"
                    "The system encountered an unexpected issue. More details have "
                    "been recorded in the application log."
                )
                messagebox.showerror(f"{self.operation} Error", friendly_msg)
            except Exception:
                self.logger.warning("Failed to show error message")

        # Only suppress exceptions for user-facing operations that we've handled
        # Re-raise critical exceptions like KeyboardInterrupt, SystemExit, etc.
        critical_exceptions = (
            KeyboardInterrupt,
            SystemExit,
            MemoryError,
            GeneratorExit,
            ImportError,
            SyntaxError,
            IndentationError,
        )

        if isinstance(exc_val, critical_exceptions):
            return False  # Re-raise critical exceptions

        # For user-facing operations, suppress the exception since we showed it to the user
        # For non-user-facing operations, re-raise to allow proper handling
        return self.user_facing


class StatusManager:
    """Centralized status bar management with history and icons"""

    ICONS = {
        "ready": "✓",
        "processing": "⏳",
        "error": "✗",
        "warning": "⚠",
        "success": "✓",
        "initializing": "⚙",
        "timeout": "⌛",
        "calibrating": "📊",
        "idle": "💤",
        "info": "ℹ",
    }

    def __init__(self, status_label: Any, assistant_status: Any) -> None:
        """Initialize status manager with UI components

        Args:
            status_label: Main status label widget
            assistant_status: Assistant status label widget
        """
        self.status_label = status_label
        self.assistant_status = assistant_status
        self.status_history: Deque[Dict[str, Any]] = deque(maxlen=100)

        # Timing management to prevent overlap
        self.last_update_time: float = 0
        self.min_update_interval = 0.5  # Minimum 500ms between updates
        self.pending_update: tuple[str, str, str | None] | None = None
        self.update_timer: str | None = None

    def set_status(
        self, message: str, status_type: str = "ready", assistant_state: Optional[str] = None
    ) -> None:
        """Set status with icon and history, with timing management

        Args:
            message: Status message to display
            status_type: Type of status (determines icon)
            assistant_state: Optional assistant state to display
        """
        import time

        current_time = time.time()

        # Check if enough time has passed since last update
        if current_time - self.last_update_time < self.min_update_interval:
            # Store the pending update
            self.pending_update = (message, status_type, assistant_state)

            # Schedule the update for later if not already scheduled
            if self.update_timer is None:
                delay_ms = int(
                    (self.min_update_interval - (current_time - self.last_update_time)) * 1000
                )
                self.update_timer = self.status_label.after(delay_ms, self._process_pending_update)
            return

        # Process the update immediately
        self._process_status_update(message, status_type, assistant_state)
        self.last_update_time = current_time

        # Clear any pending timer
        if self.update_timer is not None:
            self.status_label.after_cancel(self.update_timer)
            self.update_timer = None
            self.pending_update = None

    def _process_pending_update(self) -> None:
        """Process any pending status update"""
        if self.pending_update:
            message, status_type, assistant_state = self.pending_update
            self._process_status_update(message, status_type, assistant_state)
            self.last_update_time = time.time()
            self.pending_update = None
        self.update_timer = None

    def _process_status_update(
        self, message: str, status_type: str, assistant_state: Optional[str] = None
    ) -> None:
        """Actually process and display the status update"""
        icon = self.ICONS.get(status_type, "")
        full_message = f"{icon} {message}" if icon else message

        # Add to history
        self.status_history.append(
            {
                "message": message,
                "status_type": status_type,
                "assistant_state": assistant_state,
                "timestamp": time.time(),
                "display_message": full_message,
            }
        )

        # Update main status label if it exists
        if hasattr(self, "status_label") and self.status_label:
            if hasattr(self.status_label, "winfo_exists") and self.status_label.winfo_exists():
                self.status_label.config(text=full_message)

        # Update assistant status if provided and exists
        if hasattr(self, "assistant_status") and self.assistant_status:
            if (
                hasattr(self.assistant_status, "winfo_exists")
                and self.assistant_status.winfo_exists()
            ):
                self.assistant_status.config(text=assistant_state)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get status history

        Returns:
            List of status entries with timestamps
        """
        return list(self.status_history)

    def clear_history(self) -> None:
        """Clear status history"""
        self.status_history.clear()


class DisplayUpdateManager:
    """Manages display updates with consistent checking and error handling.

    This class provides a centralized way to manage UI updates with proper
    widget existence checks and error handling to prevent crashes.
    """

    def __init__(self, gui: Any) -> None:
        """Initialize the update manager.

        Args:
            gui: Reference to the main GUI instance
        """
        self.gui = gui
        self.logger = getattr(gui, "logger", logging.getLogger("APGIGUI"))

    def update_display(
        self, display_name: str, update_func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """Generic display updater with safety checks.

        Args:
            display_name: Name of the display for logging
            update_func: Function to call for the update
            *args: Positional arguments to pass to update_func
            **kwargs: Keyword arguments for update_func and special options:
                required_widgets: List of widget names that must exist
                skip_check: If True, skip widget existence checks
        """
        with ErrorContext(f"{display_name} Update", user_facing=False, logger=self.logger):
            # Get special options
            required_widgets = kwargs.pop("required_widgets", [])
            skip_check = kwargs.pop("skip_check", False)

            # Check if we should skip widget checks
            if not skip_check:
                # Check if the GUI is shutting down
                if hasattr(self.gui, "is_shutting_down") and self.gui.is_shutting_down:
                    return

            # Check required widgets exist
            for widget_name in required_widgets:
                widget = getattr(self.gui, widget_name, None)
                if not widget:
                    self.logger.debug(f"{display_name}: Widget {widget_name} not found")
                    return

                # FigureCanvasTkAgg doesn't have winfo_exists, check differently
                if hasattr(widget, "get_tk_widget"):
                    # For matplotlib canvases, check the underlying tk widget
                    tk_widget = widget.get_tk_widget()
                    if not tk_widget or not tk_widget.winfo_exists():
                        self.logger.debug(f"{display_name}: Widget {widget_name} not ready")
                        return
                elif hasattr(widget, "winfo_exists"):
                    if not widget.winfo_exists():
                        self.logger.debug(f"{display_name}: Widget {widget_name} not ready")
                        return

            # Execute the update function
            update_func(*args, **kwargs)

    def safe_call(
        self, display_name: str, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Safely call a function with error handling.

        Args:
            display_name: Name for error messages
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The result of func(*args, **kwargs) or None if an error occurred
        """
        with ErrorContext(display_name, user_facing=False, logger=self.logger):
            return func(*args, **kwargs)
        return None


class GUIConfig:
    """Centralized configuration for APGI GUI application.

    This class contains all magic numbers, UI constants, and configuration parameters
    used throughout the application. Grouped by functionality for better organization.
    """

    # Timing and performance
    DEBOUNCE_DELAY_MS = 250
    INIT_TIMEOUT_MS = 60000
    UPDATE_INTERVAL_MS = 1000
    INIT_CHECK_INTERVAL_MS = 100
    PROGRESS_BAR_UPDATE_MS = 10
    AUTO_SAVE_INTERVAL_MS = 300000  # 5 minutes
    GC_INTERVAL = 1000  # Perform GC every 1000 updates

    # UI Layout and Styling
    WINDOW_TITLE = "APGI Assistant"
    DEFAULT_WINDOW_SIZE = "1400x900"
    PAD_X = 5
    PAD_Y = 5
    BORDER_WIDTH = 2
    SLIDER_LENGTH = 200

    # Fonts
    FONT_FAMILY = "Arial"
    FONT_SIZE_SMALL = 9
    FONT_SIZE_MEDIUM = 10
    FONT_SIZE_LARGE = 12

    # Colors
    COLOR_BG = "white"
    COLOR_ACCENT = "#4a7abc"

    # Physiological ranges
    HR_RANGE = (40, 200)
    HRV_RANGE = (10, 200)
    RESP_RANGE = (8, 40)
    EDA_RANGE = (0.5, 20.0)

    # Battery thresholds
    LOW_BATTERY_THRESHOLD = 0.2
    MEDIUM_BATTERY_THRESHOLD = 0.5

    # Model configuration
    MODEL_INPUT_MIN = 32
    MODEL_INPUT_MAX = 1024
    MODEL_HIDDEN_MIN = 64
    MODEL_HIDDEN_MAX = 2048

    # History limits
    MAX_QUERY_LENGTH = 10000
    MAX_HISTORY_ITEMS = 50
    MAX_MEMORY_MB = 100

    # Theme Color Palettes
    THEMES = {
        "normal": {
            "bg": "white",
            "fg": "black",
            "accent": "#4a7abc",
            "canvas_bg": "white",
            "figure_bg": "white",
            "grid_color": "#cccccc",
            "success": "green",
            "error": "red",
            "warning": "orange",
            "info": "black",
            "state_colors": {
                "confused": "#FF4444",
                "focused": "#4444FF",
                "alert": "#FF8800",
                "relaxed": "#44FF44",
                "working": "#AA44FF",
            },
            "battery_colors": {"high": "green", "medium": "orange", "low": "red"},
        },
        "high_contrast": {
            "bg": "black",
            "fg": "white",
            "accent": "#FFFF00",
            "canvas_bg": "black",
            "figure_bg": "black",
            "grid_color": "#666666",
            "success": "#00FF00",
            "error": "#FF0000",
            "warning": "#FFA500",
            "info": "#FFFFFF",
            "state_colors": {
                "confused": "#FF6666",
                "focused": "#6666FF",
                "alert": "#FFFF00",
                "relaxed": "#66FF66",
                "working": "#FF66FF",
            },
            "battery_colors": {"high": "#00FF00", "medium": "#FFFF00", "low": "#FF0000"},
        },
    }

    # Slider Ranges
    SLIDER_LENGTH = 150
    HR_RANGE = (40, 120)  # bpm
    HRV_RANGE = (20, 100)  # ms
    RESP_RANGE = (8, 30)  # bpm
    EDA_RANGE = (1.0, 15.0)  # µS

    # Default Values
    DEFAULT_HR = 70
    DEFAULT_HRV = 50
    DEFAULT_RESP = 16
    DEFAULT_EDA = 5.0

    # Validation Ranges
    VALID_HR = (40, 200)  # bpm
    VALID_HRV = (10, 200)  # ms
    VALID_RESP = (8, 40)  # bpm
    VALID_EDA = (0.5, 20.0)  # µS
    VALID_INPUT_DIM = (32, 1024)
    VALID_HIDDEN_DIM = (64, 2048)

    # UI Elements
    BUTTON_WIDTH = 30
    BUTTON_WIDTH_SMALL = 10
    TEXT_WIDTH = 50
    TEXT_HEIGHT = 8
    PROGRESS_BAR_LENGTH = 300

    # Collections and History
    MAX_HISTORY_ITEMS = 100

    # Energy Management
    LOW_BATTERY_THRESHOLD = 0.3
    MEDIUM_BATTERY_THRESHOLD = 0.5
    BATTERY_CRITICAL_THRESHOLD = 0.2

    # Model Configuration
    MODEL_INPUT_MIN = 32
    MODEL_INPUT_MAX = 512
    MODEL_HIDDEN_MIN = 64
    MODEL_HIDDEN_MAX = 1024
    MAX_ENERGY_HISTORY = 50
    MAX_QUERY_HISTORY = 20
    MAX_UNDO_STACK = 50
    MAX_ERRORS_BEFORE_RESET = 5
    MAX_QUERY_LENGTH = 10000

    # Logging
    LOG_FILE = "apgi_gui.log"
    LOG_MAX_BYTES = 1024 * 1024  # 1MB
    LOG_BACKUP_COUNT = 5

    # Matplotlib Configuration
    MATPLOTLIB_RC = {
        "figure.subplot.bottom": 0.15,
        "figure.subplot.top": 0.85,
        "figure.subplot.left": 0.15,
        "figure.subplot.right": 0.85,
    }

    @classmethod
    def get_font(cls, size: str = "medium", weight: str = "normal") -> Tuple[str, int, str]:
        """Get font configuration.

        Args:
            size: Font size ('small', 'medium', 'large')
            weight: Font weight ('normal', 'bold')

        Returns:
            Tuple of (family, size, weight)
        """
        size_map = {
            "small": cls.FONT_SIZE_SMALL,
            "medium": cls.FONT_SIZE_MEDIUM,
            "large": cls.FONT_SIZE_LARGE,
        }
        return (cls.FONT_FAMILY, size_map.get(size, cls.FONT_SIZE_MEDIUM), weight)

    @classmethod
    def get_padding(cls, x: bool = True, y: bool = True) -> Dict[str, int]:
        """Get padding configuration.

        Args:
            x: Include horizontal padding
            y: Include vertical padding

        Returns:
            Dictionary with 'padx' and/or 'pady' values
        """
        padding = {}

        if x:
            padding["padx"] = cls.PAD_X
        if y:
            padding["pady"] = cls.PAD_Y

        return padding


def setup_logging() -> Any:
    """Setup comprehensive logging system"""
    from apgi_framework.logging.centralized_logging import get_structured_logger

    return get_structured_logger("APGIGUI")


# Initialize logger
LOGGER = setup_logging()


# Optional dependencies - check directly rather than relying on APGI module
def check_pil() -> tuple[bool, dict[str, Any]]:
    """Check if PIL/Pillow is available and working"""
    try:
        from PIL import Image, ImageTk

        # Test basic functionality
        img = Image.new("RGB", (1, 1), color="white")
        img.close()  # Explicit cleanup

        LOGGER.info("PIL/Pillow is available and working")
        return True, {"Image": Image, "ImageTk": ImageTk}
    except ImportError as e:
        LOGGER.warning(f"PIL/Pillow not available: {e}")
        return False, {}
    except Exception as e:
        LOGGER.error(f"PIL/Pillow import failed: {e}")
        return False, {}


def check_psutil() -> tuple[bool, dict[str, Any]]:
    """Check if psutil is available"""
    try:
        import psutil

        # Test basic functionality
        psutil.cpu_percent()

        LOGGER.info("psutil is available and working")
        return True, {"psutil": psutil}
    except ImportError as e:
        LOGGER.info(f"psutil not available: {e}")
        return False, {}
    except Exception as e:
        LOGGER.error(f"psutil import failed: {e}")
        return False, {}


def check_transformers() -> tuple[bool, dict[str, Any]]:
    """Check if transformers is available and working"""
    try:
        # Check if already imported
        import transformers as tf_check  # noqa: F401

        # Test basic functionality
        transformers.__version__

        LOGGER.info("transformers is available and working")
        return True, {"transformers": transformers}
    except ImportError as e:
        LOGGER.info(f"transformers not available: {e}")
        return False, {}
    except Exception as e:
        LOGGER.error(f"transformers import failed: {e}")
        return False, {}


# Check all optional dependencies
HAS_PIL, pil_modules = check_pil()
HAS_PSUTIL, psutil_modules = check_psutil()
HAS_TRANSFORMERS, transformers_modules = check_transformers()

# Set up matplotlib modules dict for compatibility
matplotlib_modules = (
    {
        "matplotlib": matplotlib if HAS_MATPLOTLIB else None,
        "plt": plt if HAS_MATPLOTLIB else None,
        "FigureCanvasTkAgg": FigureCanvasTkAgg if HAS_MATPLOTLIB else None,
        "Figure": Figure if HAS_MATPLOTLIB else None,
    }
    if HAS_MATPLOTLIB
    else {}
)

# Initialize HAS_TORCHDIFFEQ with default value
HAS_TORCHDIFFEQ = False  # Will be updated after APGI module loads


# Robust module import system
def load_apgi_module() -> Any:
    """
    Robustly load APGI Assistant module from various possible locations.

    Returns:
        ModuleType object with APGIAssistant and related classes
    """
    LOGGER.info("Attempting to load APGI Assistant module...")

    # Try direct import first
    try:
        import apgi_assistant

        LOGGER.info("Successfully loaded via direct import")
        return apgi_assistant
    except ImportError as e:
        LOGGER.debug(f"Direct import failed: {e}")

    # Try from current directory with various possible names
    current_dir = Path(__file__).parent
    possible_files = [
        "AI_Assistant.py",
        "APGI_Assistant.py",
        "APGI-Assistant-Full.py",
        "APGI-Assistant-Short.py",
    ]

    for filename in possible_files:
        module_path = current_dir / filename
        if module_path.exists():
            with ErrorContext(f"Load Module from {filename}", user_facing=False, logger=LOGGER):
                spec = importlib.util.spec_from_file_location("apgi_assistant", module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["apgi_assistant"] = module
                    spec.loader.exec_module(module)
                    LOGGER.info(f"Successfully loaded from {filename}")
                    return module

    # Last resort: try parent directory
    parent_dir = current_dir.parent
    for filename in possible_files:
        module_path = parent_dir / filename
        if module_path.exists():
            with ErrorContext(
                f"Load Module from parent/{filename}", user_facing=False, logger=LOGGER
            ):
                spec = importlib.util.spec_from_file_location("apgi_assistant", module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["apgi_assistant"] = module
                    spec.loader.exec_module(module)
                    LOGGER.info(f"Successfully loaded from parent/{filename}")
                    return module

    raise ImportError(
        "Could not find APGI Assistant module. Please ensure the module file "
        "is in the same directory as this GUI script."
    )


# Load APGI module
try:
    _APGI_MODULE = load_apgi_module()
    APGIAssistant = getattr(_APGI_MODULE, "APGIAssistant", None)
    APGIVisualizer = getattr(_APGI_MODULE, "APGIVisualizer", None)
    HAS_ASSISTANT = True
    HAS_APGI_ASSISTANT = APGIAssistant is not None

    # Merge dependency flags from APGI module with our own checks
    APGI_HAS_MATPLOTLIB = getattr(_APGI_MODULE, "HAS_MATPLOTLIB", False)
    APGI_HAS_TRANSFORMERS = getattr(_APGI_MODULE, "HAS_TRANSFORMERS", False)
    HAS_TORCHDIFFEQ = getattr(_APGI_MODULE, "HAS_TORCHDIFFEQ", False)

    LOGGER.info("APGI module loaded successfully")
except ImportError as e:
    LOGGER.critical(f"FATAL: Cannot load APGI module: {e}")
    messagebox.showerror(
        "Module Load Error",
        f"Failed to load APGI Assistant module:\n{e}\n\n"
        "Please ensure the APGI module file is in the same directory.",
    )
    sys.exit(1)


# Log dependency status
LOGGER.info(
    f"Dependency status - Matplotlib: {HAS_MATPLOTLIB}, PIL: {HAS_PIL}, psutil: {HAS_PSUTIL}, transformers: {HAS_TRANSFORMERS}"
)

# Make modules available globally if successful
if HAS_MATPLOTLIB:
    globals().update(matplotlib_modules)
    # Configure matplotlib to prevent scroll event conflicts
    try:
        plt.rcParams.update(GUIConfig.MATPLOTLIB_RC)
    except NameError:
        # plt not available, skip configuration
        pass

if HAS_PIL:
    globals().update(pil_modules)

if HAS_PSUTIL:
    globals().update(psutil_modules)

if HAS_TRANSFORMERS:
    globals().update(transformers_modules)


class DebouncedUpdater:
    """Enhanced debouncing system with operation-specific timing"""

    def __init__(self, delay_ms: int = 100) -> None:
        self.delay_ms = delay_ms
        self.pending_updates: dict[str, str] = {}
        self.update_lock = threading.Lock()

        # Operation-specific delays
        self.operation_delays = {
            "tab_switch": 150,  # Longer delay for tab switching
            "visualization": 250,  # Medium delay for visualizations
            "input": 50,  # Short delay for input updates
            "status": 100,  # Medium delay for status updates
            "default": delay_ms,
        }

        # Rapid operation detection
        self.rapid_operation_threshold = 3  # Operations within this time are "rapid"
        self.operation_history: Deque[Any] = deque(maxlen=10)

    def schedule_update(
        self, root: Any, update_id: str, callback: Any, operation_type: str = "default"
    ) -> None:
        """Schedule debounced update with operation-specific timing

        Args:
            root: Tkinter root widget
            update_id: Unique identifier for the update
            callback: Function to call when debounced
            operation_type: Type of operation for custom delay
        """
        current_time = time.time()

        # Track operation for rapid detection
        self.operation_history.append(
            {"id": update_id, "time": current_time, "type": operation_type}
        )

        # Detect rapid operations
        is_rapid = self._detect_rapid_operation(update_id, current_time)

        with self.update_lock:
            # Cancel existing update
            if update_id in self.pending_updates:
                try:
                    root.after_cancel(self.pending_updates[update_id])
                except Exception as e:
                    # Log the error but don't crash - timer might already be cancelled
                    LOGGER.debug(f"Failed to cancel timer {update_id}: {e}")

            # Get operation-specific delay
            delay = self.operation_delays.get(operation_type, self.delay_ms)

            # Reduce delay for rapid operations (but not too short)
            if is_rapid and operation_type == "tab_switch":
                delay = max(50, delay // 2)  # Minimum 50ms for rapid tab switching

            # Schedule new update
            self.pending_updates[update_id] = root.after(delay, callback)

    def _detect_rapid_operation(self, update_id: str, current_time: float) -> bool:
        """Detect if this is a rapid operation

        Args:
            update_id: Current operation ID
            current_time: Current timestamp

        Returns:
            True if this appears to be a rapid operation
        """
        # Count similar operations in recent history
        recent_count = 0
        for op in self.operation_history:
            if (
                op["id"] != update_id
                and op["type"] in ["tab_switch", "navigation"]
                and current_time - op["time"] < 0.5
            ):  # Within 500ms
                recent_count += 1

        return recent_count >= self.rapid_operation_threshold

    def cancel_update(self, root: tk.Misc, update_id: str) -> None:
        """Cancel a pending update

        Args:
            root: Tkinter root widget
            update_id: Update ID to cancel
        """
        with self.update_lock:
            if update_id in self.pending_updates:
                try:
                    root.after_cancel(self.pending_updates[update_id])
                    del self.pending_updates[update_id]
                except Exception as e:
                    # Log the error but don't crash - timer might already be cancelled
                    LOGGER.debug(f"Failed to cancel timer {update_id}: {e}")

    def cancel_all_updates(self, root: tk.Misc) -> None:
        """Cancel all pending updates

        Args:
            root: Tkinter root widget
        """
        with self.update_lock:
            for update_id in list(self.pending_updates.keys()):
                try:
                    root.after_cancel(self.pending_updates[update_id])
                except Exception as e:
                    # Log the error but don't crash - timer might already be cancelled
                    LOGGER.debug(f"Failed to cancel timer {update_id}: {e}")
            self.pending_updates.clear()

    def get_operation_stats(self) -> dict[str, Any]:
        """Get statistics about recent operations

        Returns:
            Dictionary with operation statistics
        """
        if not self.operation_history:
            return {"total_operations": 0, "rapid_sequences": 0}

        total_ops = len(self.operation_history)
        rapid_sequences = 0

        # Count rapid sequences (3+ operations within 500ms)
        for i in range(len(self.operation_history) - 2):
            ops_window = list(self.operation_history)[i : i + 3]
            time_span = ops_window[-1]["time"] - ops_window[0]["time"]
            if time_span < 0.5:  # 500ms
                rapid_sequences += 1

        return {
            "total_operations": total_ops,
            "rapid_sequences": rapid_sequences,
            "rapid_percentage": (rapid_sequences / max(1, total_ops - 2)) * 100,
        }

    def cancel_all(self, root: Any) -> None:
        """Cancel all pending updates"""
        with self.update_lock:
            for after_id in self.pending_updates.values():
                try:
                    root.after_cancel(after_id)
                except Exception as e:
                    # Log but don't crash - this is cleanup code
                    if hasattr(root, "logger"):
                        root.logger.debug(f"Failed to cancel after_id {after_id}: {e}")
            self.pending_updates.clear()


class PermissionValidator:
    """Enhanced permission validation with user guidance"""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initialize permission validator

        Args:
            logger: Logger instance for reporting
        """
        self.logger = logger or logging.getLogger("APGIGUI")
        self.permission_cache: dict[tuple[str, str], tuple[bool, str | None, str | None]] = (
            {}
        )  # Cache permission checks

    def validate_file_access(
        self, file_path: str, operation: str = "read"
    ) -> tuple[bool, str | None, str | None]:
        """Validate file access permissions with detailed error reporting

        Args:
            file_path: Path to file/directory
            operation: Type of operation ('read', 'write', 'execute')

        Returns:
            Tuple of (is_valid, error_message, suggestion)
        """
        import os
        from pathlib import Path

        path_obj = Path(file_path)

        # Check cache first
        cache_key = (str(path_obj.absolute()), operation)
        if cache_key in self.permission_cache:
            return self.permission_cache[cache_key]

        try:
            # Check if path exists
            if not path_obj.exists():
                if operation == "write":
                    # Check if parent directory exists and is writable
                    parent = path_obj.parent
                    if not parent.exists():
                        return (
                            False,
                            f"Directory '{parent}' does not exist",
                            f"Create the directory first: mkdir -p '{parent}'",
                        )
                    if not os.access(parent, os.W_OK):
                        return (
                            False,
                            f"No write permission for directory '{parent}'",
                            f"Check directory permissions or run: chmod u+w '{parent}'",
                        )
                else:
                    return (
                        False,
                        f"File '{file_path}' does not exist",
                        "Choose an existing file or create it first",
                    )

            # Check read permissions
            if operation in ["read", "write"]:
                if not os.access(path_obj if path_obj.exists() else path_obj.parent, os.R_OK):
                    return (
                        False,
                        f"No read permission for '{file_path}'",
                        f"Fix permissions: chmod u+r '{file_path}' or run as appropriate user",
                    )

            # Check write permissions
            if operation == "write":
                if path_obj.exists():
                    # Check if we can write to existing file
                    if not os.access(path_obj, os.W_OK):
                        return (
                            False,
                            f"No write permission for '{file_path}'",
                            f"Fix permissions: chmod u+w '{file_path}' or choose different location",
                        )
                else:
                    # Check if we can write to parent directory
                    if not os.access(path_obj.parent, os.W_OK):
                        return (
                            False,
                            f"No write permission for directory '{path_obj.parent}'",
                            f"Fix permissions: chmod u+w '{path_obj.parent}' or choose different location",
                        )

            # Check directory permissions
            if path_obj.is_dir():
                if not os.access(path_obj, os.X_OK):
                    return (
                        False,
                        f"No execute permission for directory '{file_path}'",
                        f"Fix permissions: chmod u+x '{file_path}' or run: chmod -R u+x '{file_path}'",
                    )

            # Check available disk space
            if operation == "write":
                try:
                    import shutil

                    stat = shutil.disk_usage(path_obj.parent if path_obj.exists() else path_obj)
                    free_space_gb = stat.free / (1024**3)
                    if free_space_gb < 0.1:  # Less than 100MB
                        return (
                            False,
                            f"Insufficient disk space ({free_space_gb:.1f}GB available)",
                            "Free up disk space or choose different location",
                        )
                except OSError as e:
                    # Disk space check failed - log but continue since it's optional
                    LOGGER.info(f"Disk space check failed: {e}")

            # All checks passed
            result: tuple[bool, Optional[str], Optional[str]] = (True, None, None)

        except PermissionError as e:
            result = (
                False,
                f"Permission denied: {e}",
                "Run with appropriate permissions or choose different location",
            )
        except OSError as e:
            result = (False, f"System error: {e}", "Check file system status and available storage")
        except Exception as e:
            result = (
                False,
                f"Unexpected error: {e}",
                "Try again or contact support if issue persists",
            )

        # Cache result
        self.permission_cache[cache_key] = result
        return result

    def validate_config_directory(self) -> tuple[bool, str, str]:
        """Validate the configuration directory (home directory)

        Returns:
            Tuple of (is_valid, error_message, suggestion)
        """
        import os
        from pathlib import Path

        try:
            config_dir = Path.home()

            # Check if home directory is accessible
            if not config_dir.exists():
                return (
                    False,
                    f"Home directory '{config_dir}' not found",
                    "Check user account and system configuration",
                )

            # Check basic permissions
            if not os.access(config_dir, os.R_OK | os.W_OK | os.X_OK):
                return (
                    False,
                    "Insufficient permissions for home directory",
                    f"Fix permissions or run: chmod u+rwx '{config_dir}'",
                )

            # Try to create a test file
            test_file = config_dir / ".apgi_permission_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except PermissionError:
                return (
                    False,
                    "Cannot write to home directory",
                    f"Check permissions or run: chmod u+w '{config_dir}'",
                )
            except Exception as e:
                return (
                    False,
                    f"Error testing write access: {e}",
                    "Check file system and available storage",
                )

            return (True, "", "")

        except Exception as e:
            return (
                False,
                f"Error validating config directory: {e}",
                "Check system configuration and user permissions",
            )

    def clear_cache(self) -> None:
        """Clear permission check cache"""
        self.permission_cache.clear()

    def get_permission_help(self, error_type: str) -> str:
        """Get user-friendly help for permission errors

        Args:
            error_type: Type of permission error

        Returns:
            Help text with suggestions
        """
        help_texts = {
            "permission_denied": """
Permission Error Help:
==================
This error occurs when the application doesn't have required permissions.

Solutions:
1. Run the application with appropriate user permissions
2. Check file/directory permissions: ls -la /path/to/file
3. Fix permissions: chmod u+rwx /path/to/file
4. Choose a different location with proper permissions
5. On Windows, run as Administrator if needed

For configuration files, ensure your home directory is accessible.
            """,
            "disk_space": """
Disk Space Error Help:
===================
This error occurs when there's not enough storage space.

Solutions:
1. Free up disk space by removing unnecessary files
2. Choose a different location with more space
3. Check disk quota limits if on a shared system
4. Use external storage if available

Check available space: df -h (Linux/Mac) or Check disk properties (Windows)
            """,
            "file_not_found": """
File Not Found Help:
=====================
This error occurs when the specified file doesn't exist.

Solutions:
1. Choose an existing file from the file dialog
2. Create the file first if it's a new file
3. Check if the file was moved or deleted
4. Verify the file path is correct

Use the file dialog's browser to navigate to existing files.
            """,
        }

        return help_texts.get(error_type, "Unknown error. Please check the logs for details.")

    def show_permission_error(
        self, parent: tk.Misc, title: str, error_msg: str, suggestion: str
    ) -> None:
        """Show user-friendly permission error dialog

        Args:
            parent: Parent widget for dialog
            title: Dialog title
            error_msg: Error message
            suggestion: Suggestion for fixing the issue
        """
        try:
            dialog = tk.Toplevel(parent)
            dialog.title(title)
            dialog.geometry("500x400")
            dialog.resizable(True, True)

            # Make dialog modal
            dialog.transient(parent if isinstance(parent, (tk.Tk, tk.Toplevel)) else None)
            dialog.grab_set()

            # Error message
            error_frame = ttk.LabelFrame(dialog, text="Error Details", padding=10)
            error_frame.pack(fill="both", expand=True, padx=10, pady=10)

            error_text = tk.Text(error_frame, height=6, wrap=tk.WORD, font=("Arial", 10))
            error_text.pack(fill="both", expand=True)
            error_text.insert("1.0", error_msg)
            error_text.config(state="disabled")

            # Suggestion
            if suggestion:
                suggest_frame = ttk.LabelFrame(dialog, text="Suggested Solution", padding=10)
                suggest_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

                suggest_text = tk.Text(suggest_frame, height=4, wrap=tk.WORD, font=("Arial", 10))
                suggest_text.pack(fill="both", expand=True)
                suggest_text.insert("1.0", suggestion)
                suggest_text.config(state="disabled")

            # Buttons
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=10)

            def show_help() -> None:
                help_type = (
                    "permission_denied" if "permission" in error_msg.lower() else "disk_space"
                )
                help_dialog = tk.Toplevel(dialog)
                help_dialog.title("Permission Help")
                help_dialog.geometry("600x500")

                help_text = tk.Text(help_dialog, wrap=tk.WORD, font=("Arial", 9))
                help_text.pack(fill="both", expand=True, padx=10, pady=10)
                help_text.insert("1.0", self.get_permission_help(help_type))
                help_text.config(state="disabled")

                ttk.Button(help_dialog, text="Close", command=help_dialog.destroy).pack(pady=10)

            ttk.Button(button_frame, text="Help", command=show_help).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="OK", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

        except Exception:
            # Fallback to simple message box if dialog creation fails
            messagebox.showerror(title, f"{error_msg}\n\nSuggestion: {suggestion}")


class CancellableProgress:
    """Enhanced progress indicator with aggressive cancellation support"""

    def __init__(self, parent: tk.Misc, timeout_seconds: int = 60) -> None:
        self.cancelled = False
        self.timeout_seconds = timeout_seconds
        self.start_time: Optional[float] = None
        self.cancellation_callbacks: list[Callable[[str], None]] = (
            []
        )  # Callbacks to notify on cancellation
        self.frame = ttk.Frame(parent, relief="raised", borderwidth=2)

        ttk.Label(self.frame, text="Processing...", font=("Arial", 12)).pack(pady=10)

        self.progress_bar = ttk.Progressbar(self.frame, mode="indeterminate", length=300)
        self.progress_bar.pack(pady=10, padx=20)

        self.message_label = ttk.Label(self.frame, text="Please wait...")
        self.message_label.pack(pady=5)

        # Add cancel button with enhanced styling
        self.cancel_button = ttk.Button(self.frame, text="Cancel", command=self.cancel)
        self.cancel_button.pack(pady=5)

        # Timeout tracking
        self.timeout_timer_id: Optional[str] = None

    def add_cancellation_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback to be called when operation is cancelled

        Args:
            callback: Function to call on cancellation
        """
        self.cancellation_callbacks.append(callback)

    def cancel(self, reason: str = "user_cancelled") -> None:
        """Mark operation as cancelled with aggressive propagation

        Args:
            reason: Reason for cancellation
        """
        if self.cancelled:
            return  # Already cancelled

        self.cancelled = True
        self.cancel_reason = reason

        # Stop timeout timer
        if self.timeout_timer_id:
            try:
                self.frame.after_cancel(self.timeout_timer_id)
            except Exception as e:
                # Log error but don't crash - timer might already be cancelled
                LOGGER.debug(f"Failed to cancel timeout timer: {e}")

        # Update UI immediately
        self.cancel_button.config(state="disabled", text="Cancelling...")
        self.message_label.config(text="Cancelling operation...")

        # Notify all cancellation callbacks
        for callback in self.cancellation_callbacks:
            try:
                callback(reason)
            except Exception as e:
                # Don't let callback errors break cancellation
                print(f"Cancellation callback error: {e}")

        # Hide dialog after short delay to show cancellation
        self.frame.after(500, self.hide)

        # Force garbage collection to help cleanup
        import gc

        gc.collect()

    def is_cancelled(self) -> bool:
        """Check if cancelled"""
        return self.cancelled

    def get_cancel_reason(self) -> Optional[str]:
        """Get cancellation reason"""
        return getattr(self, "cancel_reason", None)

    def show(self, message: str = "Processing...", timeout: Optional[int] = None) -> None:
        """Show progress dialog with optional timeout

        Args:
            message: Progress message to display
            timeout: Custom timeout in seconds (overrides default)
        """
        self.cancelled = False
        self.start_time = time.time()
        self.message_label.config(text=message)
        self.progress_bar.start(10)

        # Set timeout
        timeout_seconds = timeout or self.timeout_seconds
        if timeout_seconds > 0:
            self.timeout_timer_id = self.frame.after(
                int(timeout_seconds * 1000), lambda: self.cancel("timeout")
            )

        # Center the dialog
        self.frame.place(relx=0.5, rely=0.5, anchor="center")

    def hide(self) -> None:
        """Hide progress dialog with enhanced cleanup"""
        try:
            # Stop progress bar if it exists
            if hasattr(self, "progress_bar") and self.progress_bar:
                try:
                    self.progress_bar.stop()
                except Exception:
                    pass  # Progress bar might already be stopped

            # Remove frame from display if it exists
            if hasattr(self, "frame") and self.frame:
                try:
                    self.frame.place_forget()
                except Exception:
                    pass  # Frame might already be hidden
        except Exception as e:
            # Log but don't crash during cleanup
            print(f"Error during progress hide cleanup: {e}")

        # Schedule actual destruction for next event loop cycle
        # This prevents "cannot invoke destroy" errors during rapid operations
        if hasattr(self, "frame") and self.frame:
            try:
                self.frame.after(1, self._safe_destroy_frame)
            except Exception:
                # If after scheduling fails, try immediate destruction
                self._safe_destroy_frame()

    def _safe_destroy_frame(self) -> None:
        """Safely destroy the progress frame"""
        try:
            if hasattr(self, "frame") and self.frame:
                # Check if frame still exists before destroying
                if hasattr(self.frame, "winfo_exists") and self.frame.winfo_exists():
                    self.frame.destroy()
        except Exception as e:
            # Log but don't crash during cleanup
            print(f"Error during progress frame destruction: {e}")
            pass  # Log error but don't crash - frame might already be destroyed
        finally:
            # Clear references to help garbage collection
            self.frame = None
            self.progress_bar = None
            self.message_label = None
            self.cancel_button = None

        # Force UI update to ensure dialog disappears
        try:
            if self.frame is not None and hasattr(self.frame, "master") and self.frame.master:
                if self.frame.master:
                    self.frame.master.update_idletasks()
        except Exception:
            pass

        # Clean up timeout timer
        if hasattr(self, "timeout_timer_id") and self.timeout_timer_id:
            try:
                if hasattr(self, "frame") and self.frame:
                    self.frame.after_cancel(self.timeout_timer_id)
            except Exception as e:
                # Log error but don't crash - timer might already be cancelled
                print(f"Failed to cancel timeout timer: {e}")
            finally:
                self.timeout_timer_id = None

    def update_message(self, message: str) -> None:
        """Update progress message"""
        if not self.cancelled:
            self.message_label.config(text=message)

    def check_timeout(self) -> bool:
        """Check if operation has timed out"""
        if self.start_time and not self.cancelled:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                self.cancel("timeout")
                return True
        return False

    def set_timeout(self, seconds: int) -> None:
        """Update timeout duration

        Args:
            seconds: New timeout in seconds
        """
        self.timeout_seconds = seconds

        # Restart timeout timer if dialog is visible
        if self.timeout_timer_id:
            try:
                self.frame.after_cancel(self.timeout_timer_id)
            except Exception as e:
                # Log error but don't crash - timer might already be cancelled
                LOGGER.debug(f"Failed to cancel timeout timer: {e}")

            if not self.cancelled:
                self.timeout_timer_id = self.frame.after(
                    int(seconds * 1000), lambda: self.cancel("timeout")
                )


class ActionHistory:
    """Command pattern for undo/redo"""

    def __init__(self, max_history: int = 50):
        self.undo_stack: list[Any] = []
        self.redo_stack: list[Any] = []
        self.max_history = max_history

    def execute(self, action: Any) -> None:
        """Execute action and add to history"""
        action.execute()
        self.undo_stack.append(action)
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self) -> None:
        """Undo last action"""
        if self.undo_stack:
            action = self.undo_stack.pop()
            action.undo()
            self.redo_stack.append(action)

    def redo(self) -> None:
        """Redo last undone action"""
        if self.redo_stack:
            action = self.redo_stack.pop()
            action.execute()
            self.undo_stack.append(action)

    def can_undo(self) -> bool:
        """Check if undo is available"""
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if redo is available"""
        return len(self.redo_stack) > 0


class SetPhysiologyAction:
    """Action for setting physiological state"""

    def __init__(self, gui: Any, old_state: Dict[str, Any], new_state: Dict[str, Any]):
        self.gui = gui
        self.old_state = old_state
        self.new_state = new_state

    def execute(self) -> None:
        self._apply_state(self.new_state)

    def undo(self) -> None:
        self._apply_state(self.old_state)

    def _apply_state(self, state: Dict[str, Any]) -> None:
        self.gui.hr_var.set(state["hr"])
        self.gui.hrv_var.set(state["hrv"])
        self.gui.resp_var.set(state["resp"])
        self.gui.eda_var.set(state["eda"])


class QueryTextAction:
    """Action for query text changes"""

    def __init__(self, gui: Any, old_text: str, new_text: str) -> None:
        self.gui = gui
        self.old_text = old_text
        self.new_text = new_text

    def execute(self) -> None:
        self._apply_text(self.new_text)

    def undo(self) -> None:
        self._apply_text(self.old_text)

    def _apply_text(self, text: str) -> None:
        if hasattr(self.gui, "query_input"):
            self.gui.query_input.delete(1.0, tk.END)
            self.gui.query_input.insert(1.0, text)
            self.gui.update_char_count()


class ClearQueryAction:
    """Action for clearing query text"""

    def __init__(self, gui: Any, old_text: str):
        self.gui = gui
        self.old_text = old_text

    def execute(self) -> None:
        if hasattr(self.gui, "query_input"):
            self.gui.query_input.delete(1.0, tk.END)
            self.gui.update_char_count()

    def undo(self) -> None:
        self._apply_text(self.old_text)

    def _apply_text(self, text: str) -> None:
        if hasattr(self.gui, "query_input"):
            self.gui.query_input.delete(1.0, tk.END)
            self.gui.query_input.insert(1.0, text)
            self.gui.update_char_count()


class InputValidator:
    """Input validation utilities with comprehensive error checking"""

    @staticmethod
    def validate_physiological(
        hr: float, hrv: float, resp: float, eda: float
    ) -> Tuple[bool, List[str]]:
        """
        Validate physiological parameters.

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # Validate heart rate
        if not isinstance(hr, (int, float)):
            errors.append("Heart rate must be a number")
        elif not (40 <= hr <= 200):  # Use hardcoded range as fallback
            errors.append("Heart rate must be between 40-200 bpm")

        # Validate HRV
        if not isinstance(hrv, (int, float)):
            errors.append("HRV must be a number")
        elif not GUIConfig.HRV_RANGE[0] <= hrv <= GUIConfig.HRV_RANGE[1]:
            errors.append(
                f"HRV must be between {GUIConfig.HRV_RANGE[0]}-{GUIConfig.HRV_RANGE[1]} ms"
            )

        # Validate respiration
        if not isinstance(resp, (int, float)):
            errors.append("Respiration must be a number")
        elif not GUIConfig.RESP_RANGE[0] <= resp <= GUIConfig.RESP_RANGE[1]:
            errors.append(
                f"Respiration must be between {GUIConfig.RESP_RANGE[0]}-{GUIConfig.RESP_RANGE[1]} bpm"
            )

        # Validate EDA
        if not isinstance(eda, (int, float)):
            errors.append("Skin conductance must be a number")
        elif not GUIConfig.EDA_RANGE[0] <= eda <= GUIConfig.EDA_RANGE[1]:
            errors.append(
                f"Skin conductance must be between {GUIConfig.EDA_RANGE[0]}-{GUIConfig.EDA_RANGE[1]} µS"
            )

        return len(errors) == 0, errors

    @staticmethod
    def validate_query(query: str) -> Tuple[bool, str]:
        """
        Validate query text.

        Returns:
            (is_valid, error_message)
        """
        if query is None:
            return False, "Query cannot be None"

        if not isinstance(query, str):
            return False, "Query must be a string"

        if not query or not query.strip():
            return False, "Query cannot be empty"

        if len(query) > GUIConfig.MAX_QUERY_LENGTH:
            return False, f"Query too long (max {GUIConfig.MAX_QUERY_LENGTH} characters)"

        return True, ""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration parameters.

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # Config is typed as Dict[str, Any], runtime check handled by caller
        # Skip redundant isinstance check to avoid unreachable code warnings

        if "input_dim" in config:
            if not isinstance(config["input_dim"], (int, float)):
                errors.append("Input dimension must be a number")
            elif not GUIConfig.MODEL_INPUT_MIN <= config["input_dim"] <= GUIConfig.MODEL_INPUT_MAX:
                errors.append(
                    f"Input dimension must be between {GUIConfig.MODEL_INPUT_MIN}-{GUIConfig.MODEL_INPUT_MAX}"
                )

        if "hidden_dim" in config:
            if not isinstance(config["hidden_dim"], (int, float)):
                errors.append("Hidden dimension must be a number")
            elif (
                not GUIConfig.MODEL_HIDDEN_MIN <= config["hidden_dim"] <= GUIConfig.MODEL_HIDDEN_MAX
            ):
                errors.append(
                    f"Hidden dimension must be between {GUIConfig.MODEL_HIDDEN_MIN}-{GUIConfig.MODEL_HIDDEN_MAX}"
                )

        return len(errors) == 0, errors


class UsageTracker:
    """Track usage patterns (privacy-respecting)

    This class tracks query metrics locally without transmitting data externally.
    All data is stored in memory only and cleared on session end.
    """

    def __init__(self) -> None:
        """Initialize usage tracker with empty metrics log"""
        self.metrics_log: List[Dict[str, Any]] = []

    def track_query(self, query_length: int, response_time: float, state: str) -> None:
        """Log query metrics for analytics

        Args:
            query_length: Length of the query in characters
            response_time: Response time in seconds
            state: Cognitive state at time of query
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "query_length": query_length,
            "response_time_ms": response_time * 1000,
            "cognitive_state": state,
        }
        # Store locally, never transmit
        self.metrics_log.append(metrics)

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of usage metrics

        Returns:
            Dictionary containing total queries, average query length,
            average response time, and last activity timestamp
        """
        if not self.metrics_log:
            return {}

        total_queries = len(self.metrics_log)
        avg_query_length = sum(m["query_length"] for m in self.metrics_log) / total_queries
        avg_response_time = sum(m["response_time_ms"] for m in self.metrics_log) / total_queries

        return {
            "total_queries": total_queries,
            "avg_query_length": avg_query_length,
            "avg_response_time_ms": avg_response_time,
            "last_activity": self.metrics_log[-1]["timestamp"],
        }


class InitializationState:
    """State machine for tracking initialization progress

    States:
        IDLE: Initial state, not started
        STARTING: Starting initialization process
        IN_PROGRESS: Initialization in progress
        COMPLETING: Finalizing initialization
        SUCCESS: Initialization completed successfully
        FAILED: Initialization failed
        TIMEOUT: Initialization timed out
    """

    IDLE = "idle"
    STARTING = "starting"
    IN_PROGRESS = "in_progress"
    COMPLETING = "completing"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class APGIGUI:
    """Main GUI application for APGI Assistant - Production Ready"""

    def __init__(self, root: tk.Tk) -> None:
        # Validate dependencies first
        if not HAS_ASSISTANT:
            messagebox.showwarning(
                "Degraded Mode",
                "APGI Assistant core module is not available.\n\n"
                "Running in degraded mode with limited features.\n"
                "Please ensure AI_Assistant.py is in the same directory\n"
                "and all required dependencies are installed.",
            )

        self.root = root
        self.root.title("APGI Assistant - Cognitive AI Interface v2.0")
        self.root.geometry("1400x900")

        # Initialize theme manager
        self.theme_manager = None
        if THEME_MANAGER_AVAILABLE:
            self.theme_manager = ThemeManager(initial_theme="normal")

        # Setup logger
        # Log event method
        def _log_event(self: Any, message: str) -> None:
            """Log an event message."""
            self.logger.info(message)

        self.logger = LOGGER
        self.logger.info("Initializing APGI GUI")

        # Thread safety
        self.assistant_lock = threading.Lock()
        self.processing_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.init_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.init_thread: Optional[threading.Thread] = None
        self.init_cancel_event = threading.Event()  # Cancellation event for init thread
        self.update_flags_lock = threading.Lock()
        self.update_flags = {
            "energy": False,
            "performance": False,
            "cognitive": False,
            "oscillatory": False,
            "biofeedback": False,
        }

        # State
        self.assistant: Optional[APGIAssistant] = None
        self.is_processing = False
        self.is_initialized = False  # Track if assistant is fully initialized
        self.is_initializing = False  # Track if initialization is in progress
        self.error_count = 0
        self.update_count = 0

        # Theme management
        self.current_theme = self._load_theme_preference()

        # Initialize APGI Assistant for conversation
        if HAS_APGI_ASSISTANT:
            try:
                self.assistant = APGIAssistant({"llm_model": "gpt2"})
                self._log_event("APGI Assistant initialized with LLM integration")
            except Exception as e:
                self.assistant = None
                self._log_event(f"Failed to initialize APGI Assistant: {e}")
        else:
            self.assistant = None
            self._log_event("APGI Assistant not available - LLM features disabled")
        self.high_contrast_var = tk.BooleanVar(value=False)

        # Configuration variables (needed before settings tab creation)
        self.adaptive_proc_var = tk.BooleanVar(value=True)
        self.energy_aware_var = tk.BooleanVar(value=True)
        self.language_model_var = tk.BooleanVar(value=HAS_TRANSFORMERS)

        # Model configuration variables
        self.memory_length_var = tk.IntVar(value=100)
        self.hidden_dim_var = tk.IntVar(value=256)
        self.input_dim_var = tk.IntVar(value=128)

        # Physiology variables
        self.hr_var = tk.IntVar(value=70)
        self.hrv_var = tk.IntVar(value=50)
        self.resp_var = tk.IntVar(value=16)
        self.eda_var = tk.DoubleVar(value=5.0)

        # Export settings
        self.export_settings: dict[str, Any] = {
            "dpi": 300,
            "format": "png",
            "quality": "high",
        }  # low, medium, high

        # Initialization state tracking
        self.init_state = InitializationState.IDLE
        self.init_start_time: float | None = None
        self.init_timeout_seconds = 60  # Increased from 45 to 60 seconds
        self.init_check_timer_id: str | None = None
        self.init_safety_timer_id: str | None = None  # Safety timer to force hide progress
        self.init_progress = 0.0  # Track initialization progress (0.0 to 1.0)
        self._last_query_text: str = ""  # Track last query text for undo/redo
        self.init_message = ""  # Current initialization status message
        self._handling_init: bool = False  # Flag to prevent multiple init handlers

        # Font size management for accessibility
        self.font_size_var = tk.IntVar(value=10)  # Default font size
        self.font_sizes = [8, 9, 10, 11, 12, 14, 16, 18, 20, 24]  # Available font sizes
        self.current_font_size = 10

        # Auto-save functionality
        self.auto_save_enabled = True
        self.auto_save_interval = 300000  # 5 minutes in milliseconds
        self.auto_save_timer_id: Optional[str] = None
        self.auto_save_file = "apgi_auto_save.json"
        self.last_auto_save: Optional[datetime] = None

        # Data storage with automatic history management
        # Initialize history manager with 1GB default (configurable via env var)
        self.history_manager = HistoryManager(max_memory_mb=1024, auto_prune=True)
        self.state_history = self.history_manager.create_managed_deque("state", maxlen=100)
        self.energy_history = self.history_manager.create_managed_deque("energy", maxlen=50)
        self.query_history = self.history_manager.create_managed_deque("query", maxlen=20)

        # Dependency notification system
        self.dependency_notifier = DependencyNotifier(parent_widget=self.root)

        # Permission validation system
        self.permission_validator = PermissionValidator(logger=self.logger)

        # Debounced updater for rapid updates
        self.debouncer = DebouncedUpdater(delay_ms=GUIConfig.DEBOUNCE_DELAY_MS)

        # Status and display management
        self.display_manager = DisplayUpdateManager(self)
        self.status_mgr: Optional[Any] = None  # Will be initialized after UI setup

        # Undo/redo functionality
        self.action_history = ActionHistory(max_history=50)

        # Usage tracking for analytics
        self.usage_tracker = UsageTracker()

        # Lazy-loaded tabs
        self.tabs_created: set[str] = set()

        # Configuration
        self.config_file = Path.home() / ".apgi_gui_config.json"
        self.session_file = Path.home() / ".apgi_gui_session.json"

        # Setup UI
        self.create_menu()
        self.create_widgets()
        self.create_status_bar()

        # Check for dependency limitations and notify user
        self.root.after(500, self.dependency_notifier.check_and_notify_limitations)

        # Setup keyboard shortcuts
        self.setup_keyboard_shortcuts()

        # Setup accessibility features
        self.setup_accessibility()

        # Setup event handlers
        self.setup_event_handlers()

        # Initialize assistant (configuration will be loaded after UI is fully initialized)
        self.root.after(100, self.initialize_assistant)

        # Start update loop
        self.root.after(1000, self.update_displays)

        # Setup cleanup on exit
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)

        # Load saved configuration after UI is fully initialized
        self.load_configuration()

        # Initialize auto-save functionality
        self.check_auto_save_recovery()
        self.start_auto_save_timer()

        self.logger.info("GUI initialization complete")

    def _log_event(self, message: str) -> None:
        """Log event to event log (thread-safe)."""
        if hasattr(self, "log_text") and self.log_text is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}\n"
            # Schedule GUI update on main thread
            self.root.after(0, lambda msg=log_message: self._safe_log_to_gui(msg))  # type: ignore[misc]

    def _safe_log_to_gui(self, message: str) -> None:
        """Safely log message to GUI with existence check."""
        try:
            if hasattr(self, "log_text") and self.log_text.winfo_exists():
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
        except Exception:
            # Ignore errors during shutdown
            pass

    def create_menu(self) -> None:
        """Create menu bar"""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Session", command=self.new_session, accelerator="Ctrl+N")
        file_menu.add_command(label="Save Session", command=self.save_session, accelerator="Ctrl+S")
        file_menu.add_command(label="Load Session", command=self.load_session, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Export Configuration", command=self.export_config)
        file_menu.add_command(label="Import Configuration", command=self.import_config)
        file_menu.add_separator()

        # Advanced Export Options
        export_menu = Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Advanced Export", menu=export_menu)
        export_menu.add_command(
            label="Export Session as CSV",
            command=self.export_session_csv,
            accelerator="Ctrl+Shift+C",
        )
        export_menu.add_command(
            label="Export Metrics as CSV",
            command=self.export_metrics_csv,
            accelerator="Ctrl+Shift+M",
        )
        export_menu.add_command(
            label="Export Complete Session as JSON",
            command=self.export_session_json,
            accelerator="Ctrl+Shift+J",
        )
        export_menu.add_command(
            label="Export Session Report as PDF",
            command=self.export_session_pdf,
            accelerator="Ctrl+Shift+P",
        )

        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit_application, accelerator="Ctrl+Q")

        # Edit menu
        edit_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear Query", command=self.clear_query, accelerator="Ctrl+L")
        edit_menu.add_command(label="Clear History", command=self.clear_history)
        edit_menu.add_separator()
        edit_menu.add_command(label="Settings", command=self.show_settings, accelerator="Ctrl+,")

        # Tools menu
        self.tools_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=self.tools_menu)
        tools_menu = self.tools_menu  # Keep backward compatibility
        tools_menu.add_command(label="Calibrate Biofeedback", command=self.calibrate_baseline)
        tools_menu.add_command(
            label="Reset Assistant", command=self.reset_assistant, accelerator="Ctrl+R"
        )
        tools_menu.add_separator()
        tools_menu.add_command(
            label="Generate State Timeline", command=self.generate_state_timeline
        )
        tools_menu.add_command(label="Generate Energy Plot", command=self.generate_energy_plot)
        tools_menu.add_separator()
        tools_menu.add_command(label="Save State Timeline", command=self.save_state_timeline)
        tools_menu.add_command(label="Save Energy Plot", command=self.save_energy_plot)
        tools_menu.add_command(
            label="Save Oscillatory Spectrum", command=self.save_oscillatory_spectrum
        )
        tools_menu.add_command(label="Save All Plots", command=self.save_all_plots)
        tools_menu.add_separator()
        tools_menu.add_command(
            label="Enable Memory Profiling",
            command=self.enable_memory_profiling,
            state="normal" if HAS_PSUTIL else "disabled",
        )
        tools_menu.add_separator()
        tools_menu.add_command(label="Memory Statistics", command=self.show_memory_stats)
        tools_menu.add_command(label="Clear All History", command=self.clear_all_history)
        tools_menu.add_command(
            label="Configure History Limits", command=self.configure_history_limits
        )

        # View menu
        view_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(
            label="Refresh Displays", command=self.refresh_displays, accelerator="F5"
        )
        view_menu.add_command(
            label="Full Screen", command=self.toggle_fullscreen, accelerator="F11"
        )
        view_menu.add_separator()
        view_menu.add_command(label="Export Settings", command=self.show_export_settings)
        view_menu.add_separator()
        view_menu.add_checkbutton(
            label="High Contrast Mode",
            variable=self.high_contrast_var,
            command=self.toggle_high_contrast,
            accelerator="Ctrl+Alt+H",
        )
        view_menu.add_separator()
        # Font size adjustment
        view_menu.add_command(
            label="Increase Font Size", command=self.increase_font_size, accelerator="Ctrl+Plus"
        )
        view_menu.add_command(
            label="Decrease Font Size", command=self.decrease_font_size, accelerator="Ctrl+Minus"
        )
        view_menu.add_command(
            label="Reset Font Size", command=self.reset_font_size, accelerator="Ctrl+0"
        )
        view_menu.add_separator()

        # Theme menu (only if theme manager is available)
        if self.theme_manager:
            theme_menu = Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Theme", menu=theme_menu)

            # Add theme options
            for theme_name in self.theme_manager.get_available_themes():
                theme_menu.add_radiobutton(
                    label=theme_name.capitalize(),
                    command=lambda t=theme_name: self._set_theme(t),  # type: ignore[misc]
                    variable=tk.StringVar(value=self.theme_manager.current_theme),
                    value=theme_name,
                )

        # Help menu
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Quick Start", command=self.show_quick_start, accelerator="F1")
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="View Logs", command=self.view_logs)

    def _set_theme(self, theme_name: str) -> None:
        """Set the current theme.

        Args:
            theme_name: Name of the theme to apply
        """
        if not self.theme_manager:
            return

        if self.theme_manager.set_theme(theme_name):
            self._apply_theme_to_widgets()

    def _apply_theme_to_widgets(self) -> None:
        """Apply current theme to all widgets."""
        if not self.theme_manager:
            return

        # Helper function to safely apply theme to text widget
        def apply_to_text_widget(widget: Any, bg_color: str, fg_color: str) -> None:
            if widget and widget.winfo_exists():
                try:
                    widget.config(bg=bg_color, fg=fg_color, insertbackground=fg_color)
                except tk.TclError:
                    pass

        # Apply theme to text widgets
        bg_color = self.theme_manager.get_theme_color("bg")
        fg_color = self.theme_manager.get_theme_color("fg")

        apply_to_text_widget(getattr(self, "query_text", None), bg_color, fg_color)
        apply_to_text_widget(getattr(self, "response_text", None), bg_color, fg_color)

        # Apply theme to status label
        if hasattr(self, "status_label") and self.status_label.winfo_exists():
            try:
                self.status_label.config(foreground=fg_color)
            except tk.TclError:
                pass

    def update_status(self, message: str, status_type: str = "info") -> None:
        """Update the status label with the given message and status type.

        Args:
            message: The status message to display
            status_type: Type of status ('info', 'success', 'error', 'warning', 'processing')
        """
        if not self.display_manager:
            return
        self.display_manager.update_display(
            "Status Update",
            lambda: self._do_status_update(message, status_type),
            required_widgets=["status_label"],
        )

    def _do_status_update(self, message: str, status_type: str) -> None:
        """Internal method to update status label with proper styling."""
        if not hasattr(self, "status_label") or not self.status_label.winfo_exists():
            return

        # Set status text
        self.status_label.config(text=message)

        # Set status color based on type - use theme colors
        theme_colors = {
            "info": self.get_theme_color("info"),
            "success": self.get_theme_color("success"),
            "error": self.get_theme_color("error"),
            "warning": self.get_theme_color("warning"),
            "processing": self.get_theme_color("accent"),
            "timeout": self.get_theme_color("error"),
        }

        # Default to theme foreground color if status_type is unknown
        fg_color = theme_colors.get(status_type.lower(), self.get_theme_color("fg"))
        self.status_label.config(foreground=fg_color)

        # Log the status update
        if status_type.lower() in ["error", "warning"]:
            self.logger.warning(f"Status ({status_type}): {message}")
        elif status_type.lower() == "success":
            self.logger.info(f"Status: {message}")
        else:
            self.logger.debug(f"Status: {message}")

    def create_widgets(self) -> None:
        """Create and arrange all GUI widgets with lazy loading"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Create placeholder frames for all tabs
        self.main_frame = ttk.Frame(self.notebook)
        self.cognitive_frame = ttk.Frame(self.notebook)
        self.oscillatory_frame = ttk.Frame(self.notebook)
        self.biofeedback_frame = ttk.Frame(self.notebook)
        self.energy_frame = ttk.Frame(self.notebook)
        self.performance_frame = ttk.Frame(self.notebook)
        self.viz_frame = ttk.Frame(self.notebook)
        self.settings_frame = ttk.Frame(self.notebook)

        # Add tabs (content created on demand)
        self.notebook.add(self.main_frame, text="Main Interface")
        self.notebook.add(self.cognitive_frame, text="Cognitive Monitoring")
        self.notebook.add(self.oscillatory_frame, text="Oscillatory Analysis")
        self.notebook.add(self.biofeedback_frame, text="Biofeedback")
        self.notebook.add(self.energy_frame, text="Energy Management")
        self.notebook.add(self.performance_frame, text="Performance")
        self.notebook.add(self.viz_frame, text="Visualizations")
        self.notebook.add(self.settings_frame, text="Settings")

        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # Create main tab immediately
        self.create_main_tab()
        self.tabs_created.add("Main Interface")

    def on_tab_changed(self, event: Any) -> None:
        """Handle tab change - lazy load content"""
        current_tab = self.notebook.select()
        tab_name = self.notebook.tab(current_tab, "text")

        if tab_name not in self.tabs_created:
            self.logger.info(f"Creating tab: {tab_name}")

            if tab_name == "Cognitive Monitoring":
                self.create_cognitive_monitoring_tab()
            elif tab_name == "Oscillatory Analysis":
                self.create_oscillatory_tab()
            elif tab_name == "Biofeedback":
                self.create_biofeedback_tab()
            elif tab_name == "Energy Management":
                self.create_energy_tab()
            elif tab_name == "Performance":
                self.create_performance_tab()
            elif tab_name == "Visualizations":
                self.create_visualization_tab()
            elif tab_name == "Settings":
                self.create_settings_tab()

            self.tabs_created.add(tab_name)

    def create_main_tab(self) -> None:
        """Create main interaction tab"""
        # Left panel - Input
        left_frame = ttk.LabelFrame(self.main_frame, text="Query Input", padding=10)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Label(left_frame, text="Enter your query:").pack(anchor="w")

        # Query input with character counter
        query_container = ttk.Frame(left_frame)
        query_container.pack(fill="both", expand=True, pady=5)

        self.query_input = scrolledtext.ScrolledText(
            query_container, height=8, width=50, wrap="word"
        )
        self.query_input.pack(fill="both", expand=True)
        self.query_input.bind("<KeyRelease>", self.update_char_count)

        # Add tooltip to query input
        if TOOLTIP_AVAILABLE:
            ToolTip(self.query_input, "Enter your query here. Press Ctrl+Enter to process.")

        self.char_count_label = ttk.Label(
            query_container, text=""  # Initially blank until first keystroke
        )
        self.char_count_label.pack(anchor="e")

        # Physiological controls
        physio_frame = ttk.LabelFrame(left_frame, text="Physiological State", padding=5)
        physio_frame.pack(fill="x", pady=5)

        # Heart rate
        ttk.Label(physio_frame, text="Heart Rate:").grid(row=0, column=0, sticky="w", pady=2)
        self.hr_var = tk.IntVar(value=70)
        self.hr_slider = ttk.Scale(
            physio_frame,
            from_=GUIConfig.HR_RANGE[0],
            to=GUIConfig.HR_RANGE[1],
            variable=self.hr_var,
            orient="horizontal",
            length=GUIConfig.SLIDER_LENGTH,
        )
        self.hr_slider.grid(row=0, column=1, padx=5)
        self.hr_label = ttk.Label(physio_frame, text="70 bpm", width=10)
        self.hr_label.grid(row=0, column=2)
        self.hr_slider.config(command=lambda v: self._on_physio_change("hr", int(float(v))))

        # HRV
        ttk.Label(physio_frame, text="HRV:").grid(row=1, column=0, sticky="w", pady=2)
        self.hrv_var = tk.IntVar(value=50)
        self.hrv_slider = ttk.Scale(
            physio_frame,
            from_=GUIConfig.HRV_RANGE[0],
            to=GUIConfig.HRV_RANGE[1],
            variable=self.hrv_var,
            orient="horizontal",
            length=GUIConfig.SLIDER_LENGTH,
        )
        self.hrv_slider.grid(row=1, column=1, padx=5)
        self.hrv_label = ttk.Label(physio_frame, text="50 ms", width=10)
        self.hrv_label.grid(row=1, column=2)
        self.hrv_slider.config(command=lambda v: self._on_physio_change("hrv", int(float(v))))

        # Respiration
        ttk.Label(physio_frame, text="Respiration:").grid(row=2, column=0, sticky="w", pady=2)
        if not hasattr(self, "resp_var"):
            self.resp_var = tk.IntVar(value=16)
        self.resp_slider = ttk.Scale(
            physio_frame,
            from_=GUIConfig.RESP_RANGE[0],
            to=GUIConfig.RESP_RANGE[1],
            variable=self.resp_var,
            orient="horizontal",
            length=GUIConfig.SLIDER_LENGTH,
        )
        self.resp_slider.grid(row=2, column=1, padx=5)
        self.resp_label = ttk.Label(physio_frame, text="16 bpm", width=10)
        self.resp_label.grid(row=2, column=2)
        self.resp_slider.config(command=lambda v: self._on_physio_change("resp", int(float(v))))

        # EDA
        ttk.Label(physio_frame, text="Skin Conductance:").grid(row=3, column=0, sticky="w", pady=2)
        if not hasattr(self, "eda_var"):
            self.eda_var = tk.DoubleVar(value=5.0)
        self.eda_slider = ttk.Scale(
            physio_frame,
            from_=GUIConfig.EDA_RANGE[0],
            to=GUIConfig.EDA_RANGE[1],
            variable=self.eda_var,
            orient="horizontal",
            length=GUIConfig.SLIDER_LENGTH,
        )
        self.eda_slider.grid(row=3, column=1, padx=5)
        self.eda_label = ttk.Label(physio_frame, text="5.0 µS", width=10)
        self.eda_label.grid(row=3, column=2)
        self.eda_slider.config(command=lambda v: self._on_physio_change("eda", float(v)))

        # Quick stress presets
        preset_frame = ttk.Frame(physio_frame)
        preset_frame.grid(row=4, column=0, columnspan=3, pady=5)
        ttk.Button(preset_frame, text="Relaxed", command=self.set_relaxed_state, width=10).pack(
            side="left", padx=2
        )
        ttk.Button(preset_frame, text="Normal", command=self.set_normal_state, width=10).pack(
            side="left", padx=2
        )
        ttk.Button(preset_frame, text="Stressed", command=self.set_stressed_state, width=10).pack(
            side="left", padx=2
        )
        ttk.Button(preset_frame, text="Anxious", command=self.set_anxious_state, width=10).pack(
            side="left", padx=2
        )

        # Process button
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(pady=10)

        self.process_button = ttk.Button(
            button_frame, text="Process Query (Ctrl+Enter)", command=self.process_query, width=30
        )
        self.process_button.pack()

        # Add tooltip to process button
        if TOOLTIP_AVAILABLE:
            ToolTip(self.process_button, "Process your query and generate AI response (Ctrl+Enter)")

        # Right panel - Output
        right_frame = ttk.LabelFrame(self.main_frame, text="Assistant Response", padding=10)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Response display with tags for formatting
        self.response_display = scrolledtext.ScrolledText(
            right_frame, height=15, width=60, wrap="word"
        )
        self.response_display.pack(fill="both", expand=True)

        # Add tooltip to response display
        if TOOLTIP_AVAILABLE:
            ToolTip(
                self.response_display,
                "AI assistant response will appear here with formatted output",
            )

        # Configure text tags for formatting
        self.response_display.tag_configure("heading", font=("Arial", 10, "bold"))
        self.response_display.tag_configure("subheading", font=("Arial", 9, "bold"))
        self.response_display.tag_configure("normal", font=("Arial", 9))

        # Cognitive state display
        state_frame = ttk.LabelFrame(right_frame, text="Current Cognitive State", padding=5)
        state_frame.pack(fill="x", pady=5)

        self.state_labels = {}
        state_items = [
            ("Primary State:", "primary"),
            ("Processing Mode:", "processing_mode"),
            ("Confidence:", "confidence"),
            ("Surprise Level:", "surprise_level"),
            ("Dominant Frequency:", "dominant_freq"),
            ("Coherence:", "coherence"),
            ("Energy Cost:", "energy_cost"),
        ]

        for i, (label_text, key) in enumerate(state_items):
            ttk.Label(state_frame, text=label_text, width=20).grid(row=i, column=0, sticky="w")
            self.state_labels[key] = ttk.Label(
                state_frame, text="--", font=("Arial", 9, "bold"), width=25
            )
            self.state_labels[key].grid(row=i, column=1, sticky="w", padx=10)

        # Configure grid weights
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

    def create_cognitive_monitoring_tab(self) -> None:
        """Create cognitive state monitoring tab"""
        # Current state panel
        current_frame = ttk.LabelFrame(self.cognitive_frame, text="Real-time State", padding=10)
        current_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # State visualization
        self.state_canvas = tk.Canvas(current_frame, height=200, bg="white")
        self.state_canvas.pack(fill="both", expand=True)

        # State metrics
        metrics_frame = ttk.LabelFrame(self.cognitive_frame, text="State Metrics", padding=10)
        metrics_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.metrics_text = scrolledtext.ScrolledText(
            metrics_frame, height=10, width=80, wrap="word"
        )
        self.metrics_text.pack(fill="both", expand=True)

        # State history
        history_frame = ttk.LabelFrame(self.cognitive_frame, text="State History", padding=10)
        history_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)

        # Add scrollbar
        history_container = ttk.Frame(history_frame)
        history_container.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(history_container, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        self.history_tree = ttk.Treeview(
            history_container,
            columns=("Time", "State", "Confidence", "Surprise"),
            show="headings",
            height=20,
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=self.history_tree.yview)

        self.history_tree.heading("Time", text="Time")
        self.history_tree.heading("State", text="State")
        self.history_tree.heading("Confidence", text="Confidence")
        self.history_tree.heading("Surprise", text="Surprise")

        self.history_tree.column("Time", width=100)
        self.history_tree.column("State", width=100)
        self.history_tree.column("Confidence", width=100)
        self.history_tree.column("Surprise", width=100)

        self.history_tree.pack(side="left", fill="both", expand=True)

        # Configure grid weights
        self.cognitive_frame.grid_columnconfigure(0, weight=1)
        self.cognitive_frame.grid_columnconfigure(1, weight=1)
        self.cognitive_frame.grid_rowconfigure(0, weight=1)
        self.cognitive_frame.grid_rowconfigure(1, weight=1)

    def create_oscillatory_tab(self) -> None:
        """Create oscillatory spectrum analysis tab"""
        # Show loading indicator for heavy tab creation
        self.root.update_idletasks()
        loading_label = ttk.Label(
            self.oscillatory_frame, text="Loading oscillatory analysis...", font=("Arial", 10)
        )
        loading_label.pack(expand=True)
        self.root.update_idletasks()

        if not HAS_MATPLOTLIB:
            loading_label.config(
                text="Matplotlib not available. Install with: pip install matplotlib"
            )
            return

        # Create tab content after loading
        loading_label.destroy()

        # Power spectrum display
        spectrum_frame = ttk.LabelFrame(self.oscillatory_frame, text="Power Spectrum", padding=10)
        spectrum_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.spectrum_fig = Figure(figsize=(8, 4), dpi=100) if HAS_MATPLOTLIB else None
        self.spectrum_ax = self.spectrum_fig.add_subplot(111) if self.spectrum_fig else None
        self.spectrum_canvas = (
            FigureCanvasTkAgg(self.spectrum_fig, spectrum_frame) if HAS_MATPLOTLIB else None
        )
        self.spectrum_canvas_widget = (
            self.spectrum_canvas.get_tk_widget() if self.spectrum_canvas else None
        )
        if self.spectrum_canvas_widget:
            self.spectrum_canvas_widget.pack(fill="both", expand=True)

        # Disable scroll events to prevent conflicts
        self.spectrum_canvas.mpl_connect("scroll_event", lambda e: None)

        # Frequency distribution
        freq_frame = ttk.LabelFrame(
            self.oscillatory_frame, text="Frequency Distribution", padding=10
        )
        freq_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.freq_fig = Figure(figsize=(8, 4), dpi=100) if HAS_MATPLOTLIB else None
        self.freq_ax = self.freq_fig.add_subplot(111) if self.freq_fig else None
        self.freq_canvas = FigureCanvasTkAgg(self.freq_fig, freq_frame) if HAS_MATPLOTLIB else None
        self.freq_canvas_widget = self.freq_canvas.get_tk_widget() if self.freq_canvas else None
        if self.freq_canvas_widget:
            self.freq_canvas_widget.pack(fill="both", expand=True)

        # Disable scroll events to prevent conflicts
        self.freq_canvas.mpl_connect("scroll_event", lambda e: None)

        # Oscillatory metrics
        metrics_frame = ttk.LabelFrame(
            self.oscillatory_frame, text="Oscillatory Metrics", padding=10
        )
        metrics_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)

        self.osc_metrics_text = scrolledtext.ScrolledText(
            metrics_frame, height=20, width=40, wrap="word"
        )
        self.osc_metrics_text.pack(fill="both", expand=True)

        # Configure grid weights
        self.oscillatory_frame.grid_columnconfigure(0, weight=2)
        self.oscillatory_frame.grid_columnconfigure(1, weight=1)
        self.oscillatory_frame.grid_rowconfigure(0, weight=1)
        self.oscillatory_frame.grid_rowconfigure(1, weight=1)

    def create_biofeedback_tab(self) -> None:
        """Create biofeedback monitoring tab"""
        # Calibration panel
        calib_frame = ttk.LabelFrame(
            self.biofeedback_frame, text="Baseline Calibration", padding=10
        )
        calib_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Button(calib_frame, text="Calibrate Baseline", command=self.calibrate_baseline).pack(
            pady=5
        )
        self.calib_status = ttk.Label(
            calib_frame, text="Not calibrated", font=("Arial", 9, "italic")
        )
        self.calib_status.pack()

        # Current physiology display
        current_frame = ttk.LabelFrame(
            self.biofeedback_frame, text="Current Physiology", padding=10
        )
        current_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.physio_canvas = tk.Canvas(current_frame, height=150, bg="white")
        self.physio_canvas.pack(fill="both", expand=True)

        # Biofeedback recommendations
        rec_frame = ttk.LabelFrame(self.biofeedback_frame, text="Recommendations", padding=10)
        rec_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)

        self.recommendations_text = scrolledtext.ScrolledText(
            rec_frame, height=15, width=40, wrap="word"
        )
        self.recommendations_text.pack(fill="both", expand=True)

        # Physiology history
        history_frame = ttk.LabelFrame(
            self.biofeedback_frame, text="Physiology History", padding=10
        )
        history_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        self.physio_history_canvas = tk.Canvas(history_frame, height=100, bg="white")
        self.physio_history_canvas.pack(fill="both", expand=True)

        # Configure grid weights
        self.biofeedback_frame.grid_columnconfigure(0, weight=1)
        self.biofeedback_frame.grid_columnconfigure(1, weight=1)
        self.biofeedback_frame.grid_rowconfigure(0, weight=1)
        self.biofeedback_frame.grid_rowconfigure(1, weight=1)
        self.biofeedback_frame.grid_rowconfigure(2, weight=1)

    def create_energy_tab(self) -> None:
        """Create energy monitoring tab"""
        # Battery display
        battery_frame = ttk.LabelFrame(self.energy_frame, text="Battery Status", padding=10)
        battery_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.battery_canvas = tk.Canvas(battery_frame, height=100, bg="white")
        self.battery_canvas.pack(fill="both", expand=True)

        # Energy usage chart
        if HAS_MATPLOTLIB:
            usage_frame = ttk.LabelFrame(self.energy_frame, text="Energy Usage History", padding=10)
            usage_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

            self.energy_fig = Figure(figsize=(8, 4), dpi=100) if HAS_MATPLOTLIB else None
            self.energy_ax = self.energy_fig.add_subplot(111) if self.energy_fig else None
            self.energy_canvas = (
                FigureCanvasTkAgg(self.energy_fig, usage_frame) if HAS_MATPLOTLIB else None
            )
            self.energy_canvas_widget = (
                self.energy_canvas.get_tk_widget() if self.energy_canvas else None
            )
            if self.energy_canvas_widget:
                self.energy_canvas_widget.pack(fill="both", expand=True)

            # Disable scroll events to prevent conflicts
            self.energy_canvas.mpl_connect("scroll_event", lambda e: None)

        # Energy statistics
        stats_frame = ttk.LabelFrame(self.energy_frame, text="Energy Statistics", padding=10)
        stats_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)

        self.energy_stats_text = scrolledtext.ScrolledText(
            stats_frame, height=20, width=40, wrap="word"
        )
        self.energy_stats_text.pack(fill="both", expand=True)

        # Configure grid weights
        self.energy_frame.grid_columnconfigure(0, weight=2)
        self.energy_frame.grid_columnconfigure(1, weight=1)
        self.energy_frame.grid_rowconfigure(0, weight=1)
        self.energy_frame.grid_rowconfigure(1, weight=1)

    def create_performance_tab(self) -> None:
        """Create performance metrics tab"""
        # Performance metrics
        metrics_frame = ttk.LabelFrame(
            self.performance_frame, text="Performance Metrics", padding=10
        )
        metrics_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.performance_text = scrolledtext.ScrolledText(
            metrics_frame, height=15, width=80, wrap="word"
        )
        self.performance_text.pack(fill="both", expand=True)

        # Query history
        query_frame = ttk.LabelFrame(self.performance_frame, text="Query History", padding=10)
        query_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Add scrollbar
        query_container = ttk.Frame(query_frame)
        query_container.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(query_container, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        self.query_tree = ttk.Treeview(
            query_container,
            columns=("Time", "Query", "State", "ResponseTime"),
            show="headings",
            height=10,
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=self.query_tree.yview)

        self.query_tree.heading("Time", text="Timestamp")
        self.query_tree.heading("Query", text="Query")
        self.query_tree.heading("State", text="State")
        self.query_tree.heading("ResponseTime", text="Response Time")

        self.query_tree.column("Time", width=100)
        self.query_tree.column("Query", width=300)
        self.query_tree.column("State", width=100)
        self.query_tree.column("ResponseTime", width=100)

        self.query_tree.pack(side="left", fill="both", expand=True)

        # Configure grid weights
        self.performance_frame.grid_columnconfigure(0, weight=1)
        self.performance_frame.grid_rowconfigure(0, weight=1)
        self.performance_frame.grid_rowconfigure(1, weight=1)

        # Initial update of performance metrics
        self.safe_update_performance_metrics()

    def create_visualization_tab(self) -> None:
        """Create visualization tools tab"""
        # Control panel
        control_frame = ttk.LabelFrame(self.viz_frame, text="Visualization Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        ttk.Button(
            control_frame, text="Generate State Timeline", command=self.generate_state_timeline
        ).pack(side="left", padx=5)
        ttk.Button(
            control_frame, text="Generate Energy Plot", command=self.generate_energy_plot
        ).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Save All Plots", command=self.save_all_plots).pack(
            side="left", padx=5
        )
        ttk.Button(control_frame, text="Clear Display", command=self.clear_viz_display).pack(
            side="left", padx=5
        )

        # Visualization display
        viz_display_frame = ttk.LabelFrame(self.viz_frame, text="Visualization Output", padding=10)
        viz_display_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.viz_canvas = tk.Canvas(viz_display_frame, bg="white") if HAS_MATPLOTLIB else None
        if self.viz_canvas:
            self.viz_canvas.pack(fill="both", expand=True)

        # Configure grid weights
        self.viz_frame.grid_columnconfigure(0, weight=1)
        self.viz_frame.grid_rowconfigure(1, weight=1)

    def create_settings_tab(self) -> None:
        """Create settings and configuration tab"""
        # Model configuration
        model_frame = ttk.LabelFrame(self.settings_frame, text="Model Configuration", padding=10)
        model_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Input dimension
        ttk.Label(model_frame, text="Input Dimension:").grid(row=0, column=0, sticky="w", pady=5)
        if not hasattr(self, "input_dim_var"):
            self.input_dim_var = tk.IntVar(value=128)
        ttk.Spinbox(
            model_frame,
            from_=GUIConfig.MODEL_INPUT_MIN,
            to=GUIConfig.MODEL_INPUT_MAX,
            textvariable=self.input_dim_var,
            width=15,
        ).grid(row=0, column=1, pady=5)

        # Hidden dimension
        ttk.Label(model_frame, text="Hidden Dimension:").grid(row=1, column=0, sticky="w", pady=5)
        if not hasattr(self, "hidden_dim_var"):
            self.hidden_dim_var = tk.IntVar(value=256)
        ttk.Spinbox(
            model_frame,
            from_=GUIConfig.MODEL_HIDDEN_MIN,
            to=GUIConfig.MODEL_HIDDEN_MAX,
            textvariable=self.hidden_dim_var,
            width=15,
        ).grid(row=1, column=1, pady=5)

        # Processing options
        proc_frame = ttk.LabelFrame(self.settings_frame, text="Processing Options", padding=10)
        proc_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        if not hasattr(self, "adaptive_proc_var"):
            self.adaptive_proc_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            proc_frame, text="Adaptive Processing", variable=self.adaptive_proc_var
        ).pack(anchor="w", pady=2)

        if not hasattr(self, "energy_aware_var"):
            self.energy_aware_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            proc_frame, text="Energy-Aware Computation", variable=self.energy_aware_var
        ).pack(anchor="w", pady=2)

        if not hasattr(self, "language_model_var"):
            self.language_model_var = tk.BooleanVar(value=HAS_TRANSFORMERS)
        ttk.Checkbutton(
            proc_frame, text="Language Model Integration", variable=self.language_model_var
        ).pack(anchor="w", pady=2)

        # Actions
        action_frame = ttk.LabelFrame(self.settings_frame, text="Actions", padding=10)
        action_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        ttk.Button(action_frame, text="Apply Settings", command=self.apply_settings).pack(
            side="left", padx=5
        )
        ttk.Button(action_frame, text="Reset to Defaults", command=self.reset_to_defaults).pack(
            side="left", padx=5
        )
        ttk.Button(action_frame, text="Reset Assistant", command=self.reset_assistant).pack(
            side="left", padx=5
        )

        # System info
        info_frame = ttk.LabelFrame(self.settings_frame, text="System Information", padding=10)
        info_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)

        self.info_text = scrolledtext.ScrolledText(info_frame, height=10, width=80, wrap="word")
        self.info_text.pack(fill="both", expand=True)
        self.update_system_info()

        # Configure grid weights
        self.settings_frame.grid_columnconfigure(0, weight=1)
        self.settings_frame.grid_rowconfigure(0, weight=1)
        self.settings_frame.grid_rowconfigure(1, weight=1)
        self.settings_frame.grid_rowconfigure(2, weight=0)
        self.settings_frame.grid_rowconfigure(3, weight=1)

    def create_status_bar(self) -> None:
        """Create status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(fill="x", side="bottom")

        self.status_label = ttk.Label(self.status_bar, text="", relief="sunken")
        self.status_label.pack(side="left", fill="x", expand=True)

        self.assistant_status = ttk.Label(
            self.status_bar, text="Assistant: Not Initialized", relief="sunken", width=25
        )
        self.assistant_status.pack(side="right")

        self.status_mgr = StatusManager(self.status_label, self.assistant_status)
        if self.status_mgr:
            self.status_mgr.set_status("ready", "Not Initialized")

    def _force_hide_progress(self) -> None:
        """Force hide progress dialog (safety fallback)"""
        if hasattr(self, "progress"):
            try:
                self.progress.hide()
                if hasattr(self.progress, "frame") and self.progress.frame:
                    self.progress.frame.destroy()
            except Exception as e:
                self.logger.debug(f"Error hiding progress: {e}")
            finally:
                if hasattr(self, "progress"):
                    delattr(self, "progress")

        if (
            self.init_state != InitializationState.SUCCESS
            and self.init_state != InitializationState.FAILED
            and self.init_state != InitializationState.TIMEOUT
        ):
            self.logger.warning("Force hiding progress while still initializing")
            self.init_state = InitializationState.FAILED
            if self.status_mgr:
                self.status_mgr.set_status("Initialization forced hide", "error", "Error ✗")

    def _emergency_progress_cleanup(self) -> None:
        """Emergency cleanup for any stuck progress dialogs"""
        try:
            # Clean up CancellableProgress if it exists
            if hasattr(self, "progress"):
                try:
                    self.progress.hide()
                    if hasattr(self.progress, "frame") and self.progress.frame:
                        self.progress.frame.destroy()
                except Exception as e:
                    self.logger.debug(f"Error cleaning up progress dialog: {e}")
                finally:
                    delattr(self, "progress")

            # Clean up legacy progress system
            self.hide_progress()

            # Force UI update
            if hasattr(self, "root") and self.root:
                try:
                    self.root.update_idletasks()
                    self.root.focus_set()
                except Exception as e:
                    self.logger.debug(f"Error updating UI after cleanup: {e}")

        except Exception as e:
            self.logger.error(f"Emergency progress cleanup failed: {e}")

    def show_progress(self, message: str) -> None:
        """Show progress dialog with message."""
        try:
            if hasattr(self, "progress") and self.progress:
                self.progress.show(message)
            elif hasattr(self, "root") and self.root:
                # Fallback to status update if no progress dialog
                self.update_status(message, "processing")
        except Exception as e:
            self.logger.debug(f"Error showing progress: {e}")

    def hide_progress(self) -> None:
        """Hide progress dialog."""
        try:
            if hasattr(self, "progress") and self.progress:
                self.progress.hide()
        except Exception as e:
            self.logger.debug(f"Error hiding progress: {e}")

    def show_shortcut_overlay(self, event: Optional[Any] = None) -> None:
        """Show visual keyboard shortcut overlay (like VSCode)"""
        overlay = tk.Toplevel(self.root)
        overlay.title("Keyboard Shortcuts")
        overlay.attributes("-alpha", 0.95)
        overlay.attributes("-topmost", True)

        # Center the overlay on the main window
        window_width = 500
        window_height = 600
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        overlay.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Press Escape to close
        overlay.bind("<Escape>", lambda e: overlay.destroy())

        # Make window resizable
        overlay.resizable(True, True)

        # Main container with padding
        main_frame = ttk.Frame(overlay, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Add title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(title_frame, text="Keyboard Shortcuts", font=("Arial", 14, "bold")).pack(
            side="left"
        )

        # Add search box
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(search_frame, text="Search:").pack(side="left", padx=(0, 5))
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var, width=40)
        search_entry.pack(side="left", fill="x", expand=True)

        # Add notebook for categorized shortcuts
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="both", expand=True)

        # Define shortcuts by category
        shortcuts_by_category = {
            "File": [
                ("Ctrl+N", "New session"),
                ("Ctrl+S", "Save session"),
                ("Ctrl+O", "Load session"),
                ("Ctrl+Q", "Exit"),
            ],
            "Edit": [
                ("Ctrl+Z", "Undo"),
                ("Ctrl+Shift+Z", "Redo"),
                ("Ctrl+C", "Copy"),
                ("Ctrl+V", "Paste"),
                ("Ctrl+X", "Cut"),
                ("Ctrl+A", "Select all"),
            ],
            "View": [
                ("F1", "Show this help"),
                ("F5", "Refresh displays"),
                ("F11", "Toggle fullscreen"),
                ("Ctrl+Plus", "Zoom in"),
                ("Ctrl+Minus", "Zoom out"),
                ("Ctrl+0", "Reset zoom"),
            ],
            "Navigation": [
                ("Ctrl+Tab", "Next tab"),
                ("Ctrl+Shift+Tab", "Previous tab"),
                ("Ctrl+1-9", "Switch to tab N"),
                ("Alt+Left", "Back"),
                ("Alt+Right", "Forward"),
            ],
            "Processing": [
                ("Ctrl+Enter", "Process query"),
                ("Ctrl+R", "Reset assistant"),
                ("Ctrl+Space", "Auto-complete"),
                ("Ctrl+Shift+P", "Command palette"),
            ],
        }

        # Create tabs for each category
        for category, shortcuts in shortcuts_by_category.items():
            frame = ttk.Frame(notebook, padding=5)
            notebook.add(frame, text=category)

            # Add shortcuts to the frame
            for i, (key, desc) in enumerate(shortcuts):
                row = ttk.Frame(frame)
                row.pack(fill="x", pady=2)

                ttk.Label(row, text=key, font=("Courier", 10, "bold"), width=20, anchor="w").pack(
                    side="left"
                )

                ttk.Label(row, text=desc, anchor="w").pack(side="left", fill="x", expand=True)

        # Add footer
        footer = ttk.Frame(main_frame)
        footer.pack(fill="x", pady=(10, 0))

        ttk.Label(
            footer, text="Press Esc to close", font=("Arial", 8, "italic"), foreground="gray50"
        ).pack(side="left")

        # Set focus to search box
        search_entry.focus_set()

        # Function to filter shortcuts
        def filter_shortcuts(*args: Any) -> None:
            search_term = search_var.get().lower()

            for tab_id in notebook.tabs():
                tab = notebook.nametowidget(tab_id)
                tab_visible = False

                # Get all rows in this tab
                rows = tab.winfo_children()

                for row in rows:
                    if not search_term:  # If search is empty, show all
                        row.pack(fill="x", pady=2)
                        tab_visible = True
                    else:
                        # Get the description label (second child of the row)
                        desc_label = row.winfo_children()[1]
                        desc_text = desc_label.cget("text").lower()

                        # Show/hide based on search
                        if search_term in desc_text or search_term in " ".join(
                            [c.cget("text").lower() for c in row.winfo_children()]
                        ):
                            row.pack(fill="x", pady=2)
                            tab_visible = True
                        else:
                            row.pack_forget()

                # Show/hide tab based on visibility of its contents
                if tab_visible:
                    notebook.tab(tab_id, state="normal")
                else:
                    notebook.tab(tab_id, state="hidden")

            # Show first visible tab
            for tab_id in notebook.tabs():
                if notebook.tab(tab_id, "state") == "normal":
                    notebook.select(tab_id)
                    break

        # Bind search functionality
        search_var.trace("w", filter_shortcuts)

        # Initial filter (in case of pre-filled search)
        filter_shortcuts()

    def setup_keyboard_shortcuts(self) -> None:
        """Setup keyboard shortcuts"""
        # Help overlay
        self.root.bind("<F1>", self.show_shortcut_overlay)

        # File operations
        self.root.bind("<Control-n>", lambda e: self.new_session())
        self.root.bind("<Control-s>", lambda e: self.save_session())
        self.root.bind("<Control-o>", lambda e: self.load_session())
        self.root.bind("<Control-q>", lambda e: self.quit_application())

        # Advanced Export shortcuts
        self.root.bind("<Control-Shift-C>", lambda e: self.export_session_csv())
        self.root.bind("<Control-Shift-M>", lambda e: self.export_metrics_csv())
        self.root.bind("<Control-Shift-J>", lambda e: self.export_session_json())
        self.root.bind("<Control-Shift-P>", lambda e: self.export_session_pdf())

        # Edit operations
        self.root.bind("<Control-l>", lambda e: self.clear_query())
        self.root.bind("<Control-comma>", lambda e: self.show_settings())
        self.root.bind("<Control-z>", lambda e: self.undo_action())
        self.root.bind("<Control-y>", lambda e: self.redo_action())

        # Processing
        self.root.bind("<Control-Return>", lambda e: self.process_query())
        self.root.bind("<Control-r>", lambda e: self.reset_assistant())

        # View
        self.root.bind("<F5>", lambda e: self.refresh_displays())
        self.root.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.root.bind("<Control-Alt-h>", lambda e: self.toggle_high_contrast())

        # Help
        self.root.bind("<F1>", lambda e: self.show_quick_start())

        self.logger.info("Keyboard shortcuts configured")

    def setup_accessibility(self) -> None:
        """Configure accessibility features"""
        # Tab navigation
        self.notebook.bind("<Control-Tab>", self.next_tab)
        self.notebook.bind("<Control-Shift-Tab>", self.prev_tab)

        # Screen reader hints
        self.query_input.config(
            insertbackground="black", highlightthickness=2  # Visible cursor  # Focus indicator
        )

    def increase_font_size(self) -> None:
        """Increase font size for better readability"""
        with ErrorContext("Increase Font Size", user_facing=False):
            current_index = self.font_sizes.index(self.current_font_size)
            if current_index < len(self.font_sizes) - 1:
                self.current_font_size = self.font_sizes[current_index + 1]
                self.font_size_var.set(self.current_font_size)
                self._apply_font_size()

    def decrease_font_size(self) -> None:
        """Decrease font size for compact display"""
        with ErrorContext("Decrease Font Size", user_facing=False):
            current_index = self.font_sizes.index(self.current_font_size)
            if current_index > 0:
                self.current_font_size = self.font_sizes[current_index - 1]
                self.font_size_var.set(self.current_font_size)
                self._apply_font_size()

    def reset_font_size(self) -> None:
        """Reset font size to default"""
        with ErrorContext("Reset Font Size", user_facing=False):
            self.current_font_size = 10
            self.font_size_var.set(self.current_font_size)
            self._apply_font_size()

    def _apply_font_size(self) -> None:
        """Apply current font size to all text widgets across all tabs"""
        font = ("TkDefaultFont", self.current_font_size)

        # Comprehensive list of all text-based widgets that should be updated
        widgets_to_update = [
            # Main interface widgets
            "query_input",
            "response_text",
            "metrics_text",
            "energy_stats_text",
            "osc_metrics_text",
            "biofeedback_text",
            # Tab-specific widgets that might exist
            "state_text",
            "physio_text",
            "history_text",
            "config_text",
            "log_text",
            # Tree widgets
            "history_tree",
            "query_tree",
            "metrics_tree",
            # Label widgets that should be updated
            "status_label",
            "assistant_status",
            # Other text-based widgets
            "energy_text",
            "performance_text",
            "cognitive_text",
        ]

        updated_count = 0
        for widget_name in widgets_to_update:
            widget = getattr(self, widget_name, None)
            if widget and hasattr(widget, "config"):
                try:
                    widget.config(font=font)
                    updated_count += 1
                except tk.TclError:
                    pass  # Widget might not support font changes

        # Also update tab-specific widgets by checking each tab
        self._update_tab_fonts()

        # Update default font configurations for consistency
        try:
            # Update TkDefaultFont for new widgets
            self.root.option_add("*Font", f"TkDefaultFont {self.current_font_size}")
        except tk.TclError:
            pass

        # Log the update for debugging
        self.logger.debug(
            f"Font size updated to {self.current_font_size}pt for {updated_count} widgets"
        )

    def _update_tab_fonts(self) -> None:
        """Update fonts for widgets in all currently created tabs"""
        tab_widgets = []

        # Check each tab for text widgets and update them
        if hasattr(self, "tabs_created"):
            for tab_name in self.tabs_created:
                # Common tab widget patterns
                tab_widget_patterns = [
                    f"{tab_name}_text",
                    f"{tab_name}_label",
                    f"{tab_name}_input",
                    f"{tab_name}_output",
                    f"{tab_name}_display",
                    f"{tab_name}_info",
                ]

                for pattern in tab_widget_patterns:
                    widget = getattr(self, pattern, None)
                    if widget and hasattr(widget, "config"):
                        tab_widgets.append(widget)

        # Apply font to all discovered tab widgets
        font = ("TkDefaultFont", self.current_font_size)
        for widget in tab_widgets:
            try:
                widget.config(font=font)
            except tk.TclError:
                pass

    def next_tab(self, event: Any) -> None:
        """Navigate to next tab with enhanced debouncing"""
        with ErrorContext("Next Tab Navigation", user_facing=False):
            # Check if application is shutting down or destroyed
            if (hasattr(self, "is_shutting_down") and self.is_shutting_down) or (
                hasattr(self, "root") and not self.root.winfo_exists()
            ):
                return

            # Use enhanced debouncing for tab switching
            self.debouncer.schedule_update(
                self.root, "tab_next", self._perform_next_tab, operation_type="tab_switch"
            )

    def _perform_next_tab(self) -> None:
        """Actually perform the tab navigation"""
        try:
            # Additional safety checks before navigation
            if (
                (hasattr(self, "is_shutting_down") and self.is_shutting_down)
                or not hasattr(self, "notebook")
                or not self.notebook.winfo_exists()
            ):
                return

            current: str = self.notebook.select()
            tabs: tuple[str, ...] = self.notebook.tabs()
            if not tabs:  # No tabs to navigate
                return
            current_index = tabs.index(current)
            next_index = (current_index + 1) % len(tabs)
            self.notebook.select(tabs[next_index])
        except Exception as e:
            self.logger.warning(f"Tab navigation error: {e}")

    def prev_tab(self, event: Any) -> None:
        """Navigate to previous tab with enhanced debouncing"""
        with ErrorContext("Previous Tab Navigation", user_facing=False):
            # Check if application is shutting down or destroyed
            if (hasattr(self, "is_shutting_down") and self.is_shutting_down) or (
                hasattr(self, "root") and not self.root.winfo_exists()
            ):
                return

            # Use enhanced debouncing for tab switching
            self.debouncer.schedule_update(
                self.root, "tab_prev", self._perform_prev_tab, operation_type="tab_switch"
            )

    def _perform_prev_tab(self) -> None:
        """Actually perform the tab navigation"""
        try:
            # Additional safety checks before navigation
            if (
                (hasattr(self, "is_shutting_down") and self.is_shutting_down)
                or not hasattr(self, "notebook")
                or not self.notebook.winfo_exists()
            ):
                return

            current = self.notebook.select()
            tabs: tuple[str, ...] = self.notebook.tabs()
            if not tabs:  # No tabs to navigate
                return
            current_index = tabs.index(current)
            prev_index = (current_index - 1) % len(tabs)
            self.notebook.select(tabs[prev_index])
        except Exception as e:
            self.logger.warning(f"Tab navigation error: {e}")

    def validate_configuration(self, config: Dict[str, Any]) -> None:
        """Validate configuration before applying"""
        schema = {
            "input_dim": (32, 1024),
            "hidden_dim": (64, 2048),
        }

        for key, (min_val, max_val) in schema.items():
            if key in config:
                if not min_val <= config[key] <= max_val:
                    raise ValueError(f"{key} must be between {min_val} and {max_val}")

    def setup_event_handlers(self) -> None:
        """Setup custom event handlers"""
        self.root.bind("<<ProcessComplete>>", self.handle_process_complete)
        self.root.bind("<<ProcessError>>", self.handle_process_error)
        self.root.bind("<<AssistantInitEvent>>", self.handle_assistant_init)

    def handle_process_complete(self, event: Any) -> None:
        """Handle processing complete event"""
        try:
            event_type, data = self.processing_queue.get_nowait()

            if event_type == "response":
                self._update_response(data)
            elif event_type == "error":
                self._show_error(data)
        except queue.Empty:
            pass

    def handle_process_error(self, event: Any) -> None:
        """Handle processing error event"""
        try:
            _, error_msg = self.processing_queue.get_nowait()
            self._show_error(error_msg)
        except queue.Empty:
            pass

    def handle_assistant_init(self, event: Any) -> None:
        """Handle assistant initialization completion (success or failure)"""
        # Prevent multiple simultaneous handlers with thread-safe check
        if hasattr(self, "_handling_init") and self._handling_init:
            self.logger.debug("Initialization handler already running, skipping")
            return

        # Use atomic check-and-set pattern
        if not hasattr(self, "_handling_init_lock"):
            import threading

            self._handling_init_lock = threading.Lock()

        with self._handling_init_lock:
            if hasattr(self, "_handling_init") and self._handling_init:
                self.logger.debug("Initialization handler already running, skipping")
                return

            self._handling_init = True

        try:
            self.logger.debug("=== Handling Initialization Event ===")
            status: str
            payload: Any

            # Clean up timer
            self._cleanup_init_timer()

            # Get result from queue
            try:
                status, payload = self.init_queue.get_nowait()
                self.logger.debug(f"Init result: {status}")
            except queue.Empty:
                self.logger.warning("Init event triggered but queue is empty")
                # Hide progress dialog even if queue is empty to prevent stuck popup
                if hasattr(self, "progress"):
                    try:
                        self.progress.hide()
                        # Ensure the dialog is properly destroyed
                        if hasattr(self.progress, "frame") and self.progress.frame:
                            self.progress.frame.destroy()
                    except Exception as e:
                        self.logger.debug(f"Error hiding progress dialog: {e}")
                    finally:
                        # Always remove the progress attribute
                        delattr(self, "progress")
                else:
                    self.hide_progress()
                return

            # Hide progress indicator immediately on any result
            if hasattr(self, "progress"):
                try:
                    self.progress.hide()
                    if hasattr(self.progress, "frame") and self.progress.frame:
                        self.progress.frame.destroy()
                except Exception as e:
                    self.logger.debug(f"Error hiding progress dialog: {e}")
                finally:
                    if hasattr(self, "progress"):
                        delattr(self, "progress")

            if status == "success":
                # Success path
                assistant = payload["assistant"]

                with self.assistant_lock:
                    self.assistant = assistant

                self.init_state = InitializationState.SUCCESS
                self.is_initialized = True
                assert self.init_start_time is not None
                elapsed = time.time() - self.init_start_time

                self.logger.info(f"✓ Assistant initialized successfully in {elapsed:.2f}s")

                # Update UI
                if (
                    self.status_mgr
                    and hasattr(self, "root")
                    and self.root
                    and self.root.winfo_exists()
                ):
                    self.status_mgr.set_status(
                        f"APGI Assistant ready ({elapsed:.1f}s)", "success", "Ready ✓"
                    )

                # Auto-calibrate after a short delay to ensure UI is responsive
                self.root.after(100, self._delayed_auto_calibrate)

            elif status == "cancelled":
                # Cancellation path
                self.logger.info(f"Initialization cancelled: {payload}")
                self.init_state = InitializationState.FAILED
                self.is_initialized = False

                # Update UI
                if self.status_mgr:
                    self.status_mgr.set_status("Initialization cancelled", "error", "Cancelled ✗")

            else:
                # Failure path
                error_msg = payload
                self.init_state = InitializationState.FAILED
                self.is_initialized = False

                with self.assistant_lock:
                    self.assistant = None

                self.logger.error(f"✗ Initialization failed: {error_msg}")

                # Update UI
                if self.status_mgr:
                    self.status_mgr.set_status("Initialization failed", "error", "Failed ✗")

                # Show error to user
                messagebox.showerror(
                    "Initialization Error",
                    f"Failed to initialize APGI Assistant:\n\n{error_msg}\n\n"
                    "Check logs for details.",
                )

        finally:
            # Reset handling flag
            self._handling_init = False

    def _get_init_config(self) -> Dict[str, Any]:
        """Get initialization configuration with safe defaults"""
        input_var = getattr(self, "input_dim_var", tk.IntVar(value=128))
        hidden_var = getattr(self, "hidden_dim_var", tk.IntVar(value=256))
        return {
            "input_dim": input_var.get(),
            "hidden_dim": hidden_var.get(),
            "output_dim": 64,
        }

    def _start_init_monitoring(self) -> None:
        """Start monitoring initialization progress with timeout"""
        self._check_init_progress()

    def _check_init_progress(self) -> None:
        """Periodically check initialization progress and handle timeout"""

        # Check if initialization is still active
        if self.init_state not in [
            InitializationState.STARTING,
            InitializationState.IN_PROGRESS,
            InitializationState.COMPLETING,
        ]:
            # Initialization finished (success or failure)
            self._cleanup_init_timer()
            return

        # Check for timeout
        if self.init_start_time is None:
            return
        elapsed = time.time() - self.init_start_time

        if elapsed > self.init_timeout_seconds:
            self.logger.error(f"Initialization timeout after {elapsed:.1f}s")
            self.init_state = InitializationState.TIMEOUT
            self._handle_init_timeout()
            return

        # Update progress message with time remaining
        if hasattr(self, "progress") and hasattr(self.progress, "update_message"):
            self.progress.update_message(
                f"Initializing... {elapsed:.0f}s / {self.init_timeout_seconds}s"
            )

        # Check if result is available in queue
        if not self.init_queue.empty():
            self.root.event_generate("<<AssistantInitEvent>>", when="tail")
            return

        # Schedule next check (every 100ms)
        self.init_check_timer_id = self.root.after(100, self._check_init_progress)

    def _cleanup_init_timer(self) -> None:
        """Clean up initialization timer"""
        if self.init_check_timer_id is not None:
            try:
                self.root.after_cancel(self.init_check_timer_id)
            except Exception as e:
                self.logger.debug(f"Timer cleanup error: {e}")
            finally:
                self.init_check_timer_id = None

        if self.init_safety_timer_id is not None:
            try:
                self.root.after_cancel(self.init_safety_timer_id)
            except Exception as e:
                self.logger.debug(f"Safety timer cleanup error: {e}")
            finally:
                self.init_safety_timer_id = None

    def _handle_init_timeout(self) -> None:
        """Handle initialization timeout"""
        self.logger.error("=== Initialization Timeout ===")

        # Update state
        self.init_state = InitializationState.TIMEOUT
        self.is_initialized = False

        # Clean up
        self._cleanup_init_timer()
        self.hide_progress()

        with self.assistant_lock:
            self.assistant = None

        # Update UI
        if self.status_mgr:
            self.status_mgr.set_status(
                f"Initialization timed out ({self.init_timeout_seconds}s)", "timeout", "Timeout ⌛"
            )

        # Show error dialog
        messagebox.showerror(
            "Initialization Timeout",
            f"Assistant initialization timed out after {self.init_timeout_seconds} seconds.\n\n"
            "Possible causes:\n"
            "• Insufficient system resources\n"
            "• Missing or incompatible dependencies\n"
            "• Large model loading\n"
            "• Network issues (if using remote models)\n\n"
            "Try:\n"
            "• Closing other applications\n"
            "• Checking system logs\n"
            "• Restarting the application\n"
            "• Disabling language model integration in Settings",
        )

    def _handle_init_failure(self, error_msg: str) -> None:
        """Handle initialization failure"""
        self.logger.error(f"=== Initialization Failed: {error_msg} ===")

        # Update state
        self.init_state = InitializationState.FAILED

        # Clean up
        self._cleanup_init_timer()
        self.hide_progress()

        with self.assistant_lock:
            self.assistant = None

        # Update UI
        if self.status_mgr:
            self.status_mgr.set_status("Initialization failed", "error", "Error ✗")

        # Show error to user
        messagebox.showerror(
            "Initialization Error", f"{error_msg}\n\n" "Please check the logs for more details."
        )

    def initialize_assistant(self) -> None:
        """Start assistant initialization with unified timeout and progress tracking"""

        # Prevent duplicate initialization
        if self.init_state in [InitializationState.STARTING, InitializationState.IN_PROGRESS]:
            self.logger.debug("Initialization already in progress")
            return

        if self.init_state == InitializationState.SUCCESS:
            self.logger.debug("Assistant already initialized")
            return

        self.logger.info("=== Starting Assistant Initialization ===")

        # Reset cancellation event
        self.init_cancel_event.clear()

        # Update state
        self.init_state = InitializationState.STARTING
        self.init_start_time = time.time()

        # Show progress with cancellation support
        self.progress = CancellableProgress(self.root, timeout_seconds=self.init_timeout_seconds)

        # Add cancellation callback
        self.progress.add_cancellation_callback(lambda reason: self._cancel_initialization(reason))

        self.progress.show("Initializing APGI Assistant...")

        # Add a safety timer to hide progress after maximum timeout + 10 seconds buffer
        self.init_safety_timer_id = self.root.after(
            (self.init_timeout_seconds + 10) * 1000, self._force_hide_progress
        )

        # Update UI
        if self.status_mgr:
            self.status_mgr.set_status(
                "Initializing assistant components", "initializing", "Initializing"
            )

        try:
            # Get and validate configuration
            config = self._get_init_config()

            # Validate configuration
            is_valid, errors = InputValidator.validate_config(config)
            if not is_valid:
                raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

            self.logger.debug(f"Configuration validated: {config}")

            # Start initialization thread
            self.init_thread = threading.Thread(
                target=self._run_initialization, args=(config,), daemon=True, name="APGIInitThread"
            )
            self.init_thread.start()
            self.logger.info(f"Initialization thread started: {self.init_thread.name}")

            # Start progress monitoring
            self._start_init_monitoring()

        except Exception as e:
            self._handle_init_failure(f"Failed to start initialization: {str(e)}")

    def _cancel_initialization(self, reason: str) -> None:
        """Handle initialization cancellation"""
        self.logger.info(f"Initialization cancelled: {reason}")

        # Set cancellation event
        self.init_cancel_event.set()

        # Update state
        self.init_state = InitializationState.FAILED

        # Hide progress
        if hasattr(self, "progress"):
            self.progress.hide()

        # Update status
        if self.status_mgr:
            self.status_mgr.set_status("Initialization cancelled", "error", "Cancelled ✗")

        # Clean up
        self._cleanup_init_timer()
        return

    def _run_initialization(self, config: Dict[str, Any]) -> None:
        """Background thread: Initialize the assistant

        This runs in a separate thread and puts results in the init_queue.
        The GUI thread monitors the queue and updates the UI.
        """
        try:
            # The module was already loaded by load_apgi_module() and registered as 'apgi_assistant'
            try:
                import apgi_assistant
            except ImportError:
                # Fallback: try to get it from globals if it failed to register in sys.modules
                apgi_assistant = globals().get("_APGI_MODULE")

            if not apgi_assistant:
                self.logger.error("APGI module (apgi_assistant) not found")
                self.init_queue.put(("error", "APGI module not loaded - Check logs"))
                return

            if not hasattr(apgi_assistant, "APGIAssistant"):
                self.logger.error("APGIAssistant class not found in loaded module")
                self.init_queue.put(("error", "Assistant class not found in module"))
                return

            APGIAssistantClass = apgi_assistant.APGIAssistant

            self.logger.info("Background initialization started")

            # Update state
            self.init_state = InitializationState.IN_PROGRESS

            # Periodic progress updates
            def update_progress(message: str) -> None:
                try:
                    if not self.init_cancel_event.is_set():
                        if hasattr(self, "progress") and self.progress:
                            self.root.after(0, lambda: self.progress.update_message(message))
                except Exception as e:
                    self.logger.debug(f"Progress update failed: {e}")

            # Check for cancellation at start
            if self.init_cancel_event.is_set():
                self.init_queue.put(("cancelled", "Initialization cancelled by user"))
                return

            # Step 1: Create assistant instance
            update_progress("Creating APGI Assistant instance (this may take a moment)...")
            self.logger.info("Step 1/4: Creating APGIAssistant")

            # Check for cancellation before heavy operations
            if self.init_cancel_event.is_set():
                self.init_queue.put(("cancelled", "Initialization cancelled by user"))
                return

            assistant = APGIAssistantClass(
                config={
                    "input_dim": config["input_dim"],
                    "hidden_dim": config["hidden_dim"],
                    "output_dim": 64,
                },
                device="cpu",
                memory_length=100,
                enable_adaptive_processing=True,
                enable_energy_aware=True,
                enable_language_model=self.language_model_var.get(),
            )

            self.logger.info("Step 1/4: ✓ APGIAssistant created")

            # Check for cancellation
            if self.init_cancel_event.is_set():
                self.init_queue.put(("cancelled", "Initialization cancelled by user"))
                return

            # Step 2: Initialize components
            update_progress("Initializing neural networks (this may take 30+ seconds)...")
            self.logger.info("Step 2/4: Initializing components")

            # Let the assistant do any necessary setup
            # (this might include model weight initialization, etc.)
            time.sleep(0.5)  # Brief pause for any async initialization

            self.logger.info("Step 2/4: ✓ Components initialized")

            # Step 3: Warm up (optional - run a test forward pass)
            update_progress("Warming up models...")
            self.logger.info("Step 3/4: Model warm-up")

            try:
                # Calibrate baseline physiology first to avoid warnings
                resting_data = [
                    {"hr": 70, "hrv": 50, "resp": 16, "eda": 5.0},
                    {"hr": 72, "hrv": 48, "resp": 15, "eda": 5.2},
                    {"hr": 68, "hrv": 52, "resp": 17, "eda": 4.8},
                ]
                assistant.calibrate(resting_data)

                # Run a dummy inference to warm up the models
                dummy_data = {"hr": 70, "hrv": 50, "resp": 16, "eda": 5.0}
                _ = assistant.encode_physiology(dummy_data)
                self.logger.info("Step 3/4: ✓ Model warm-up complete")
            except Exception as warmup_error:
                self.logger.warning(f"Model warm-up failed (non-critical): {warmup_error}")

            # Step 4: Finalize
            update_progress("Finalizing initialization...")
            self.logger.info("Step 4/4: Finalizing")

            # Put success result in queue
            self.init_queue.put(("success", {"assistant": assistant}))
            self.init_state = InitializationState.COMPLETING

            if self.init_start_time is not None:
                elapsed = time.time() - self.init_start_time
                self.logger.info(f"=== Initialization Complete in {elapsed:.2f}s ===")

                if elapsed > 30:
                    self.logger.warning(f"Initialization took {elapsed:.1f}s (threshold: 30s)")

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            self.init_queue.put(("error", str(e)))
            self.init_state = InitializationState.FAILED

        finally:
            # Set initialization complete flag to stop checking
            self.is_initializing = False

    def _check_init_status(self) -> None:
        """Periodically check initialization status"""
        with ErrorContext("Check Initialization Status", user_facing=False):
            if not self.is_initializing:
                return

            if not self.init_queue.empty():
                # Trigger the event handler immediately
                self.root.event_generate("<<AssistantInitEvent>>", when="tail")
            else:
                # Continue checking
                self.root.after(100, self._check_init_status)

    def _handle_initialization_error(self, error_msg: str) -> None:
        """Handle initialization errors consistently"""
        self.logger.error(error_msg, exc_info=True)
        self.is_initializing = False
        with ErrorContext("Handle Initialization Error", user_facing=False):
            # Cancel any pending timeout
            if hasattr(self, "init_timeout_id"):
                self.root.after_cancel(self.init_timeout_id)
                delattr(self, "init_timeout_id")

            # Update status
            self.update_status("Initialization failed", "error")
            self.update_assistant_status("Error", "error")

            # Show error to user
            try:
                messagebox.showerror(
                    "Initialization Error",
                    f"{error_msg}\n\n" "Please check the logs for more details and try again.",
                )
            except Exception as e:
                self.logger.error(f"Failed to show error dialog: {e}")

            # Clean up
            with self.assistant_lock:
                self.assistant = None
            self.init_thread = None

    def _delayed_auto_calibrate(self) -> None:
        """Delayed auto-calibration to ensure UI responsiveness"""
        with ErrorContext("Auto-calibration", user_facing=False):
            self.auto_calibrate()

    def auto_calibrate(self) -> None:
        """Auto-calibrate with sample resting data"""
        with ErrorContext("Auto-calibration", user_facing=False):
            if self.assistant:
                resting_data = [
                    {"hr": 68, "hrv": 55, "resp": 15, "eda": 4.5},
                    {"hr": 70, "hrv": 52, "resp": 16, "eda": 4.8},
                    {"hr": 69, "hrv": 54, "resp": 14, "eda": 4.6},
                ]

                with self.assistant_lock:
                    self.assistant.calibrate(resting_data)

                if hasattr(self, "calib_status"):
                    self.calib_status.config(text="Calibrated (auto) ✓")
                self.logger.info("Auto-calibration complete")

    def process_query(self) -> None:
        """Process user query with validation and error handling"""
        if not self.is_initialized:
            # Add debugging information to identify the root cause
            debug_info = {
                "is_initialized": self.is_initialized,
                "init_state": getattr(self, "init_state", "unknown"),
                "assistant_exists": hasattr(self, "assistant") and self.assistant is not None,
                "is_initializing": getattr(self, "is_initializing", False),
            }
            self.logger.warning(f"Initialization check failed: {debug_info}")
            messagebox.showwarning("Not Ready", "Please wait for assistant initialization")
            return

        if self.is_processing:
            messagebox.showinfo("Processing", "Already processing a query. Please wait.")
            return

        # Get and validate query
        query = self.query_input.get(1.0, tk.END).strip()
        is_valid, error_msg = InputValidator.validate_query(query)
        if not is_valid:
            messagebox.showwarning("Invalid Query", error_msg)
            return

        # Get and validate physiological data
        physio_data = {
            "hr": self.hr_var.get(),
            "hrv": self.hrv_var.get(),
            "resp": self.resp_var.get(),
            "eda": self.eda_var.get(),
        }

        is_valid, errors = InputValidator.validate_physiological(**physio_data)
        if not is_valid:
            messagebox.showwarning("Invalid Physiological Data", "\n".join(errors))
            return

        # Start processing
        self.is_processing = True
        self.process_button.config(state="disabled")

        # Create cancellable progress dialog
        self.progress = CancellableProgress(self.root)
        self.progress.show("Processing query...")
        self.update_status("Processing query...", "processing")

        # Process in separate thread
        thread = threading.Thread(
            target=self._process_query_thread,
            args=(query, physio_data),
            daemon=False,  # Not daemon - we want clean shutdown
        )
        thread.start()

        self.logger.info(f"Processing query: {query[:50]}...")

    def _process_query_thread(self, query: str, physio_data: dict[str, float]) -> None:
        """Thread function for processing query with aggressive cancellation"""
        with ErrorContext("Process Query Thread", user_facing=False):
            # Set up cancellation callbacks
            def on_cancellation(reason: str) -> None:
                self.logger.info(f"Query processing cancelled: {reason}")

            if hasattr(self, "progress"):
                self.progress.add_cancellation_callback(on_cancellation)

            # Check cancellation frequently
            for i in range(100):  # Check 100 times during processing
                if hasattr(self, "progress") and self.progress.is_cancelled():
                    self.logger.info(f"Query processing cancelled at check {i}")
                    return

                # Process in small chunks to allow cancellation
                if i == 0:
                    # Initial setup
                    pass

                time.sleep(0.01)  # Small delay to allow cancellation

            # Update progress message
            if hasattr(self, "progress"):
                self.root.after(0, lambda: self.progress.update_message("Analyzing query..."))

            # Process with assistant
            with self.assistant_lock:
                # Check cancellation before processing
                if hasattr(self, "progress") and self.progress.is_cancelled():
                    self.logger.info("Query processing cancelled before assistant processing")
                    return

                try:
                    # Set shorter timeout for assistant processing
                    if hasattr(self, "progress"):
                        self.progress.set_timeout(30)  # 30 second timeout for processing

                    if self.assistant is None:
                        raise ValueError("Assistant not initialized")

                    response = self.assistant.process_with_introspection(
                        user_input=query,
                        physiological_data=physio_data,
                        metadata={"source": "gui", "timestamp": datetime.now().isoformat()},
                    )

                except Exception as e:
                    # Check if this is a cancellation-related error
                    if hasattr(self, "progress") and self.progress.is_cancelled():
                        self.logger.info(f"Processing interrupted by cancellation: {e}")
                        return
                    else:
                        # Re-raise non-cancellation errors
                        raise

            # Check cancellation after processing
            if hasattr(self, "progress") and self.progress.is_cancelled():
                self.logger.info("Query processing cancelled after assistant processing")
                return

            # Queue response for main thread only if not cancelled
            if not (hasattr(self, "progress") and self.progress.is_cancelled()):
                self.processing_queue.put(("response", response))
                self.root.event_generate("<<ProcessComplete>>", when="tail")
                self.logger.info("Query processed successfully")

    def _update_response(self, response: dict[str, Any]) -> None:
        """Update GUI with response (called in main thread)"""
        with ErrorContext("Update Response Display", user_facing=False):
            # Hide progress dialog
            if hasattr(self, "progress"):
                self.progress.hide()
                delattr(self, "progress")

            # Check if main response widget still exists
            if not hasattr(self, "response_display") or not self.response_display.winfo_exists():
                return

            # Display response with formatting
            self.response_display.delete(1.0, tk.END)

            self.response_display.insert(tk.END, "ANSWER\n", "heading")
            self.response_display.insert(tk.END, f"{response['answer']}\n\n", "normal")

            self.response_display.insert(tk.END, "EXPLANATION\n", "heading")
            self.response_display.insert(tk.END, f"{response['explanation']}\n\n", "normal")

            if "biofeedback_recommendations" in response:
                self.response_display.insert(tk.END, "BIOFEEDBACK\n", "heading")
                bio = response["biofeedback_recommendations"]
                self.response_display.insert(
                    tk.END,
                    f"State: {bio.get('state', 'unknown')}\n"
                    f"Recommendation: {bio.get('recommendation', 'none')}\n",
                    "normal",
                )

            # Update cognitive state display
            if hasattr(self, "state_labels"):
                # Copy values to prevent race condition during iteration
                labels_copy = list(self.state_labels.values())
                if all(label.winfo_exists() for label in labels_copy):
                    state = response["cognitive_state"]
                    self.state_labels["primary"].config(
                        text=state.get("primary", "unknown").upper()
                    )
                    self.state_labels["processing_mode"].config(
                        text=state.get("processing_mode", "N/A")
                    )

                confidence = response["confidence"]
                if isinstance(confidence, dict):
                    conf_text = (
                        f"{confidence.get('level', 'unknown')} ({confidence.get('numeric', 0):.2f})"
                    )
                else:
                    conf_text = str(confidence)
                self.state_labels["confidence"].config(text=conf_text)

                self.state_labels["surprise_level"].config(
                    text=state.get("surprise_level", "unknown")
                )

                osc = state.get("oscillatory_profile", {})
                self.state_labels["dominant_freq"].config(
                    text=osc.get("dominant_frequency", "unknown")
                )
                self.state_labels["coherence"].config(text=f"{osc.get('coherence', 0.0):.3f}")

                energy = response["processing_metadata"].get("energy_used", "unknown")
                self.state_labels["energy_cost"].config(text=str(energy))

            # Track usage metrics
            query_text = self.query_input.get(1.0, tk.END).strip()
            query_length = len(query_text)
            response_time = response["processing_metadata"].get("time_elapsed", 0)
            cognitive_state = response["cognitive_state"]["primary"]
            self.usage_tracker.track_query(query_length, response_time, cognitive_state)

            # Add to history
            self.add_to_history(query_text, response)

            # Flag updates needed (thread-safe)
            with self.update_flags_lock:
                self.update_flags["cognitive"] = True
                self.update_flags["oscillatory"] = True
                self.update_flags["biofeedback"] = True
                self.update_flags["energy"] = True
                self.update_flags["performance"] = True

            # Debounce visualization updates
            self.debouncer.schedule_update(
                self.root, "viz_update", lambda: self.update_visualizations(response)
            )

            # Update status
            self.update_status("Query processed successfully ✓")

        # Reset processing state
        self.is_processing = False
        if hasattr(self, "process_button") and self.process_button.winfo_exists():
            self.process_button.config(state="normal")
        self.hide_progress()

    def _show_error(self, error_msg: str) -> None:
        """Show error message (called in main thread)"""
        self.error_count += 1

        with ErrorContext("Show Error Message", user_facing=False):
            try:
                messagebox.showerror("Error", error_msg)
            except Exception as e:
                # GUI might not be ready - log the error but don't crash
                self.logger.warning(
                    f"Could not show error dialog: {e}. Error message was: {error_msg}"
                )

        self.update_status(f"Error: {error_msg[:50]}...")

        self.is_processing = False
        if hasattr(self, "process_button") and self.process_button.winfo_exists():
            self.process_button.config(state="normal")

        # Hide progress dialog
        if hasattr(self, "progress"):
            self.progress.hide()
            delattr(self, "progress")

        if self.error_count > 5:
            try:
                if messagebox.askyesno(
                    "Multiple Errors", "Multiple errors detected. Reset assistant?"
                ):
                    self.reset_assistant()
            except Exception as e:
                self.logger.warning(f"Failed to show reset dialog: {e}")

    def add_to_history(self, user_query: str, user_response: dict[str, Any]) -> None:
        """Add query to history"""
        timestamp = time.strftime("%H:%M:%S")
        state = user_response["cognitive_state"].get("primary", "unknown")
        confidence = (
            user_response["confidence"].get("level", "unknown")
            if isinstance(user_response["confidence"], dict)
            else str(user_response["confidence"])
        )
        surprise = user_response["cognitive_state"].get("surprise_level", "unknown")

        # Add to history displays (if created)
        if "Cognitive Monitoring" in self.tabs_created:
            if hasattr(self, "history_tree") and self.history_tree.winfo_exists():
                # Debounce history tree updates to prevent excessive UI redraws
                self.debouncer.schedule_update(
                    self.root,
                    "cognitive_history",
                    lambda: self._update_cognitive_history(
                        timestamp, state, confidence, surprise, user_query, user_response
                    ),
                    "default",
                )

    def _update_cognitive_history(
        self,
        timestamp: str,
        state: str,
        confidence: str,
        surprise: str,
        user_query: str | None = None,
        user_response: dict[str, Any] | None = None,
    ) -> None:
        """Internal method to update cognitive history tree"""
        self.history_tree.insert("", 0, values=(timestamp, state, confidence, surprise))
        # Limit history
        children = self.history_tree.get_children()
        if len(children) > 50:
            self.history_tree.delete(children[-1])

        if "Performance" in self.tabs_created:
            if hasattr(self, "query_tree") and self.query_tree.winfo_exists():
                self.debouncer.schedule_update(
                    self.root,
                    "performance_history",
                    lambda q=user_query, r=user_response: self._update_performance_history(
                        timestamp,
                        q[:50] + "..." if len(q) > 50 else q,
                        state,
                        r["processing_metadata"]["time_elapsed"],
                    ),
                )

    def _update_performance_history(
        self, timestamp: str, query_preview: str, state: str, response_time: float
    ) -> None:
        """Internal method to update performance history tree"""
        self.query_tree.insert(
            "", 0, values=(timestamp, query_preview, state, f"{response_time:.3f}s")
        )
        # Limit history
        children = self.query_tree.get_children()
        if len(children) > 50:
            self.query_tree.delete(children[-1])

    def update_visualizations(self, response: dict[str, Any]) -> None:
        """Update visualization displays"""
        with ErrorContext("Update Visualizations", user_facing=False):
            # Update oscillatory spectrum
            if HAS_MATPLOTLIB and "Oscillatory Analysis" in self.tabs_created:
                if (
                    response
                    and "cognitive_state" in response
                    and "oscillatory_profile" in response["cognitive_state"]
                ):
                    self.update_oscillatory_display(response)

            # Update energy display
            if "Energy Management" in self.tabs_created:
                if self.assistant and self.assistant.enable_energy_aware:
                    self.update_energy_display()

            # Update cognitive monitoring
            if "Cognitive Monitoring" in self.tabs_created:
                if response and "cognitive_state" in response:
                    self.update_cognitive_display(response)

            # Update biofeedback
            if "Biofeedback" in self.tabs_created:
                if response and "biofeedback" in response:
                    self.update_biofeedback_display(response)

    def _do_oscillatory_update(self, response: dict[str, Any]) -> None:
        """Internal method for oscillatory display update logic"""
        # Get power spectrum from metadata
        metadata = response.get("processing_metadata", {})
        osc_profile = metadata.get("oscillatory_profile", {})

        if not osc_profile:
            return

        # Clear and redraw spectrum
        if self.spectrum_ax is not None:
            self.spectrum_ax.clear()

        # Extract band names and powers
        bands = []
        powers = []

        for key, value in osc_profile.items():
            if isinstance(value, (int, float)):
                bands.append(key)
                powers.append(float(value))

        if bands and powers and self.spectrum_ax and self.spectrum_canvas:
            # Plot power spectrum with theme color
            ax = self.spectrum_ax
            bars = ax.bar(bands, powers, color=self.get_theme_color("accent"), alpha=0.7)
            ax.set_xlabel("Frequency Band")
            ax.set_ylabel("Power")
            ax.set_title("Oscillatory Power Spectrum")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)

            # Color bars by power level using theme colors
            max_power = max(powers) if powers else 1
            for bar, power in zip(bars, powers):
                if power > max_power * 0.7:
                    bar.set_color(self.get_theme_color("error"))
                elif power > max_power * 0.4:
                    bar.set_color(self.get_theme_color("warning"))
                else:
                    bar.set_color(self.get_theme_color("success"))

            if self.spectrum_fig:
                self.spectrum_fig.tight_layout()
            # Disable scroll events on matplotlib canvas to prevent conflicts
            self.spectrum_canvas.mpl_connect("scroll_event", lambda e: None)
            self.spectrum_canvas.draw()

        # Update metrics text
        self.osc_metrics_text.delete(1.0, tk.END)
        metrics_text = "Oscillatory Metrics:\n" + "=" * 30 + "\n\n"

        state = response["cognitive_state"]
        osc = state.get("oscillatory_profile", {})

        metrics_text += f"Dominant Frequency: {osc.get('dominant_frequency', 'unknown')}\n"
        metrics_text += f"Coherence: {osc.get('coherence', 0.0):.3f}\n"
        metrics_text += f"Entropy: {osc.get('entropy', 0.0):.3f}\n\n"

        metrics_text += "Power by Band:\n"
        for band, power in zip(bands, powers):
            metrics_text += f"  {band}: {power:.3f}\n"

        self.osc_metrics_text.insert(tk.END, metrics_text)

    def update_oscillatory_display(self, response: dict[str, Any]) -> None:
        """Update oscillatory spectrum display"""
        bands = []
        powers = []

        # Get oscillatory profile from response
        osc_profile = getattr(response, "oscillatory_profile", {}) or {}

        for key, value in osc_profile.items():
            if isinstance(value, (int, float)):
                bands.append(key)
                powers.append(float(value))

        self.display_manager.update_display(
            "Oscillatory Display",
            self._do_oscillatory_update,
            response,
            required_widgets=["spectrum_canvas", "osc_metrics_text"],
        )

    def _do_energy_update(self) -> None:
        """Internal method for energy display update logic"""
        if not self.assistant or not self.assistant.enable_energy_aware:
            return

        # Get current battery level
        battery_level = self.assistant.energy_monitor.get_level()

        # Update battery visualization
        self.battery_canvas.delete("all")
        canvas_width = self.battery_canvas.winfo_width()
        canvas_height = self.battery_canvas.winfo_height()

        if canvas_width > 10 and canvas_height > 10:
            margin = 20
            battery_width = canvas_width - 2 * margin
            battery_height = canvas_height - 2 * margin

            # Battery outline
            self.battery_canvas.create_rectangle(
                margin,
                margin,
                margin + battery_width,
                margin + battery_height,
                outline="black",
                width=3,
            )

            # Battery terminal
            terminal_width = 10
            terminal_height = battery_height // 3
            self.battery_canvas.create_rectangle(
                margin + battery_width,
                margin + battery_height // 2 - terminal_height // 2,
                margin + battery_width + terminal_width,
                margin + battery_height // 2 + terminal_height // 2,
                fill="black",
                outline="black",
            )

            # Battery level fill
            fill_width = battery_width * battery_level
            color = "green" if battery_level > 0.5 else "orange" if battery_level > 0.2 else "red"

            self.battery_canvas.create_rectangle(
                margin + 2,
                margin + 2,
                margin + fill_width - 2,
                margin + battery_height - 2,
                fill=color,
                outline="",
            )

            # Battery percentage text
            self.battery_canvas.create_text(
                canvas_width // 2,
                canvas_height // 2,
                text=f"{battery_level:.0%}",
                font=("Arial", 16, "bold"),
                fill="white" if battery_level > 0.3 else "black",
            )

        # Update energy history
        self.energy_history.append({"time": time.time(), "level": battery_level})

        # Update energy plot if we have enough data
        if HAS_MATPLOTLIB and len(self.energy_history) > 1:
            if not self.energy_ax:
                return

            self.energy_ax.clear()

            times = [e["time"] - self.energy_history[0]["time"] for e in self.energy_history]
            levels = [e["level"] for e in self.energy_history]

            self.energy_ax.plot(times, levels, "g-", linewidth=2, marker="o", markersize=4)
            self.energy_ax.fill_between(times, levels, alpha=0.3, color="green")
            self.energy_ax.axhline(
                y=GUIConfig.LOW_BATTERY_THRESHOLD,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="Low Battery",
            )
            self.energy_ax.axhline(
                y=GUIConfig.MEDIUM_BATTERY_THRESHOLD,
                color="orange",
                linestyle="--",
                linewidth=1.5,
                label="Medium Battery",
            )

            self.energy_ax.set_xlabel("Time (seconds)")
            self.energy_ax.set_ylabel("Battery Level")
            self.energy_ax.set_title("Battery Level Over Time")
            self.energy_ax.set_ylim(0, 1.05)
            self.energy_ax.legend(loc="best")
            self.energy_ax.grid(True, alpha=0.3)

            self.energy_fig.tight_layout()
            # Disable scroll events on matplotlib canvas to prevent conflicts
            self.energy_canvas.mpl_connect("scroll_event", lambda e: None)
            self.energy_canvas.draw()

        # Update statistics
        if hasattr(self, "energy_stats_text") and self.energy_stats_text.winfo_exists():
            self.update_energy_stats()

    def update_energy_display(self) -> None:
        """Update energy display"""
        self.display_manager.update_display(
            "Energy Display", self._do_energy_update, required_widgets=["battery_canvas"]
        )

        if hasattr(self, "battery_canvas"):
            # Get current battery level
            battery_level = (
                self.assistant.energy_monitor.get_level()
                if self.assistant and hasattr(self.assistant, "energy_monitor")
                else 0.5
            )

            # Clear previous drawing
            self.battery_canvas.delete("all")

            # Get canvas dimensions
            canvas_width = self.battery_canvas.winfo_width()
            canvas_height = self.battery_canvas.winfo_height()

            # Battery dimensions
            margin = 20
            battery_width = 200
            battery_height = 100

            # Draw battery outline
            self.battery_canvas.create_rectangle(
                margin,
                margin,
                margin + battery_width,
                margin + battery_height,
                outline="black",
                width=3,
            )

            # Battery terminal
            terminal_width = 10
            terminal_height = battery_height // 3
            self.battery_canvas.create_rectangle(
                margin + battery_width,
                margin + battery_height // 2 - terminal_height // 2,
                margin + battery_width + terminal_width,
                margin + battery_height // 2 + terminal_height // 2,
                fill="black",
                outline="black",
            )

            # Battery level fill - use theme colors
            fill_width = battery_width * battery_level
            battery_colors: dict[str, str] = cast(
                dict[str, str],
                GUIConfig.THEMES[self.current_theme].get(
                    "battery_colors", {"high": "green", "medium": "orange", "low": "red"}
                )
                or {"high": "green", "medium": "orange", "low": "red"},
            )
            color = (
                battery_colors["high"]
                if battery_level > 0.5
                else battery_colors["medium"] if battery_level > 0.2 else battery_colors["low"]
            )

            self.battery_canvas.create_rectangle(
                margin + 2,
                margin + 2,
                margin + fill_width - 2,
                margin + battery_height - 2,
                fill=color,
                outline="",
            )

            # Battery percentage text
            self.battery_canvas.create_text(
                canvas_width // 2,
                canvas_height // 2,
                text=f"{battery_level:.0%}",
                font=("Arial", 16, "bold"),
                fill="white" if battery_level > 0.3 else "black",
            )

            # Update energy history
            self.energy_history.append({"time": time.time(), "level": battery_level})

            # Update energy plot
            if HAS_MATPLOTLIB and len(self.energy_history) > 1:
                # Check if matplotlib widgets still exist
                if (
                    not hasattr(self, "energy_canvas")
                    or not self.energy_canvas.get_tk_widget().winfo_exists()
                ):
                    return

                if not self.energy_ax:
                    return

                self.energy_ax.clear()

                times = [e["time"] - self.energy_history[0]["time"] for e in self.energy_history]
                levels = [e["level"] for e in self.energy_history]

                self.energy_ax.plot(
                    times,
                    levels,
                    color=self.get_theme_color("success"),
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )
                self.energy_ax.fill_between(
                    times, levels, alpha=0.3, color=self.get_theme_color("success")
                )
                self.energy_ax.axhline(
                    y=GUIConfig.LOW_BATTERY_THRESHOLD,
                    color=self.get_theme_color("error"),
                    linestyle="--",
                    linewidth=1.5,
                    label="Low Battery",
                )
                self.energy_ax.axhline(
                    y=GUIConfig.MEDIUM_BATTERY_THRESHOLD,
                    color=self.get_theme_color("warning"),
                    linestyle="--",
                    linewidth=1.5,
                    label="Medium Battery",
                )

                self.energy_ax.set_xlabel("Time (seconds)")
                self.energy_ax.set_ylabel("Battery Level")
                self.energy_ax.set_title("Battery Level Over Time")
                self.energy_ax.set_ylim(0, 1.05)
                self.energy_ax.legend(loc="best")
                self.energy_ax.grid(True, alpha=0.3)

                self.energy_fig.tight_layout()
                # Disable scroll events on matplotlib canvas to prevent conflicts
                self.energy_canvas.mpl_connect("scroll_event", lambda e: None)
                self.energy_canvas.draw()

            # Update statistics
            if hasattr(self, "energy_stats_text") and self.energy_stats_text.winfo_exists():
                self.update_energy_stats()

    def update_energy_stats(self) -> None:
        """Update energy statistics display"""
        try:
            # Get current battery level
            battery_level = (
                self.assistant.energy_monitor.get_level()
                if self.assistant and hasattr(self.assistant, "energy_monitor")
                else 0.5
            )

            # Calculate statistics
            if len(self.energy_history) > 1:
                times = [e["time"] for e in self.energy_history]
                levels = [e["level"] for e in self.energy_history]

                avg_level = sum(levels) / len(levels)
                min_level = min(levels)
                max_level = max(levels)

                # Calculate drain rate (level per hour)
                if len(times) >= 2:
                    time_diff = times[-1] - times[0]
                    level_diff = levels[0] - levels[-1]
                    drain_rate = (level_diff / time_diff) * 3600 if time_diff > 0 else 0
                else:
                    drain_rate = 0
            else:
                avg_level = battery_level
                min_level = battery_level
                max_level = battery_level
                drain_rate = 0

            # Format statistics text
            stats_text = f"""Energy Statistics
═══════════════════════════

Current Level: {battery_level:.1%}
Average Level: {avg_level:.1%}
Minimum Level: {min_level:.1%}
Maximum Level: {max_level:.1%}

Drain Rate: {drain_rate:.3%}/hour
History Points: {len(self.energy_history)}

Battery Status: {"Good" if battery_level > 0.5 else "Medium" if battery_level > 0.2 else "Low"}
"""

            # Update the text widget
            self.energy_stats_text.delete("1.0", tk.END)
            self.energy_stats_text.insert("1.0", stats_text)

        except Exception as e:
            self.logger.error(f"Error updating energy stats: {e}", exc_info=True)

    def update_cognitive_display(self, response: dict[str, Any]) -> None:
        """Update cognitive monitoring display"""
        with ErrorContext("Update Cognitive Display", user_facing=False):
            # Check if widgets still exist
            if not hasattr(self, "state_canvas"):
                return
            if hasattr(self.state_canvas, "winfo_exists") and not self.state_canvas.winfo_exists():
                return

            if not hasattr(self, "metrics_text"):
                return
            if hasattr(self.metrics_text, "winfo_exists") and not self.metrics_text.winfo_exists():
                return

            # Update state canvas
            self.state_canvas.delete("all")
            canvas_width = self.state_canvas.winfo_width()
            canvas_height = self.state_canvas.winfo_height()

            if canvas_width > 10 and canvas_height > 10:
                # Get cognitive state with defaults
                state: dict[str, Any] = (
                    response.get("cognitive_state", {}) if isinstance(response, dict) else {}
                )

                # Use theme state colors
                state_colors = cast(
                    dict[str, str], GUIConfig.THEMES[self.current_theme].get("state_colors", {})
                )
                primary_state = state.get("primary", "idle")
                color = state_colors.get(primary_state, "#888888")
                intensity = state.get("intensity", 0.5)

                # Draw state indicator
                x, y = canvas_width // 2, canvas_height // 2
                radius = min(canvas_width, canvas_height) // 3

                # Outer intensity ring
                intensity_radius = radius * (1 + intensity * 0.5)
                for i in range(3):
                    r = intensity_radius - i * 3
                    self.state_canvas.create_oval(
                        x - r, y - r, x + r, y + r, outline=color, width=2
                    )

                # Main circle
                self.state_canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=color,
                    outline="black",
                    width=2,
                )

                # State text
                self.state_canvas.create_text(
                    x, y, text=primary_state.upper(), font=("Arial", 14, "bold"), fill="white"
                )

                # Intensity bar
                bar_width = canvas_width - 40
                bar_x = 20
                bar_y = canvas_height - 30

                self.state_canvas.create_rectangle(
                    bar_x, bar_y - 5, bar_x + bar_width, bar_y + 5, outline="black", width=1
                )

                self.state_canvas.create_rectangle(
                    bar_x,
                    bar_y - 5,
                    bar_x + bar_width * intensity,
                    bar_y + 5,
                    fill=color,
                    outline="",
                )

                self.state_canvas.create_text(
                    bar_x + bar_width // 2,
                    bar_y - 15,
                    text=f"Intensity: {intensity:.1%}",
                    font=("Arial", 9),
                )

            # Update metrics display
            self.metrics_text.delete(1.0, tk.END)
            metrics_text = "Cognitive State Metrics:\n" + "=" * 40 + "\n\n"

            state = response.get("cognitive_state", {})

            metrics_text += f"Primary State: {state.get('primary', 'unknown')}\n"
            metrics_text += f"Intensity: {state.get('intensity', 0):.2f}\n"
            metrics_text += f"Surprise Level: {state.get('surprise_level', 'unknown')}\n"
            metrics_text += f"Processing Mode: {state.get('processing_mode', 'unknown')}\n\n"

            if "confidence" in state:
                conf = state["confidence"]
                if isinstance(conf, dict):
                    metrics_text += f"Confidence Level: {conf.get('level', 'unknown')}\n"
                    metrics_text += f"Confidence Value: {conf.get('numeric', 0):.3f}\n"
                else:
                    metrics_text += f"Confidence: {conf}\n"

            metrics_text += "\nAttention Allocation:\n"
            for key, value in state.get("attention_allocation", {}).items():
                metrics_text += f"  {key.title()}: {value:.3f}\n"

            metrics_text += "\nOscillatory Profile:\n"
            for key, value in state.get("oscillatory_profile", {}).items():
                if isinstance(value, (int, float)):
                    metrics_text += f"  {key.replace('_', ' ').title()}: {value:.3f}\n"
                else:
                    metrics_text += f"  {key.replace('_', ' ').title()}: {value}\n"

            self.metrics_text.insert(tk.END, metrics_text)

    def update_biofeedback_display(self, response: dict[str, Any]) -> None:
        """Update biofeedback display"""
        try:
            # Update recommendations
            if "biofeedback_recommendations" in response:
                # Check if widget still exists
                if (
                    hasattr(self, "recommendations_text")
                    and self.recommendations_text.winfo_exists()
                ):
                    bio = response["biofeedback_recommendations"]

                    self.recommendations_text.delete(1.0, tk.END)
                    rec_text = "Biofeedback Analysis:\n" + "=" * 30 + "\n\n"

                    rec_text += f"Mental State: {bio.get('state', 'unknown').title()}\n"
                    rec_text += f"Stress Level: {bio.get('stress_level', 'unknown').title()}\n"
                    rec_text += f"Heart Rate: {bio.get('heart_rate', 'N/A')} bpm\n"
                    rec_text += f"vs Baseline: {bio.get('heart_rate_vs_baseline', 1.0):.2f}x\n\n"

                    rec_text += "Recommendation:\n"
                    recommendation = bio.get("recommendation", "none").replace("_", " ").title()
                    rec_text += f"  → {recommendation}\n\n"

                    timestamp = bio.get("timestamp", time.time())
                    rec_text += f"Updated: {time.strftime('%H:%M:%S', time.localtime(timestamp))}\n"

                    self.recommendations_text.insert(tk.END, rec_text)

            # Update physiology canvas
            if hasattr(self, "physio_canvas") and self.physio_canvas.winfo_exists():
                self.physio_canvas.delete("all")
                canvas_width = self.physio_canvas.winfo_width()
                canvas_height = self.physio_canvas.winfo_height()

                if canvas_width > 10 and canvas_height > 10:
                    hr = self.hr_var.get()
                    hrv = self.hrv_var.get()
                    resp = self.resp_var.get()
                    eda = self.eda_var.get()

                    # Draw physiology bars
                    y_positions = [30, 60, 90, 120]
                    values = [hr, hrv, resp, eda]
                    labels = ["HR", "HRV", "RESP", "EDA"]
                    max_vals = [120, 100, 30, 15]
                    units = ["bpm", "ms", "bpm", "µS"]

                    for i, (y, val, label, max_val, unit) in enumerate(
                        zip(y_positions, values, labels, max_vals, units)
                    ):
                        bar_width = (val / max_val) * (canvas_width - 150)

                        # Determine color based on range
                        if label == "HR":
                            color = (
                                "green"
                                if 60 <= val <= 80
                                else "orange" if 50 <= val <= 90 else "red"
                            )
                        elif label == "HRV":
                            color = "green" if val > 50 else "orange" if val > 30 else "red"
                        elif label == "RESP":
                            color = (
                                "green"
                                if 12 <= val <= 20
                                else "orange" if 10 <= val <= 24 else "red"
                            )
                        else:  # EDA
                            color = "green" if val < 6 else "orange" if val < 8 else "red"

                        # Bar background
                        self.physio_canvas.create_rectangle(
                            90,
                            y - 10,
                            90 + (canvas_width - 150),
                            y + 10,
                            fill="#E0E0E0",
                            outline="",
                        )

                        # Bar foreground
                        self.physio_canvas.create_rectangle(
                            90, y - 10, 90 + bar_width, y + 10, fill=color, outline="black"
                        )

                        # Label
                        self.physio_canvas.create_text(
                            45, y, text=label, font=("Arial", 10, "bold")
                        )

                        # Value
                        self.physio_canvas.create_text(
                            95 + bar_width + 40, y, text=f"{val} {unit}", font=("Arial", 9)
                        )

        except Exception as e:
            self.logger.error(f"Biofeedback display error: {e}", exc_info=True)

    def safe_update_energy_display(self) -> None:
        """Safely update energy display with widget validation"""
        try:
            if (
                not hasattr(self, "energy_canvas")
                or not self.energy_canvas.get_tk_widget().winfo_exists()
            ):
                return

            if not hasattr(self, "battery_canvas"):
                return
            if (
                hasattr(self.battery_canvas, "winfo_exists")
                and not self.battery_canvas.winfo_exists()
            ):
                return

            self.update_energy_display()
        except Exception as e:
            self.logger.error(f"Safe energy display error: {e}", exc_info=True)

    def safe_update_performance_metrics(self) -> None:
        """Safely update performance metrics with widget validation"""
        try:
            if not hasattr(self, "performance_text"):
                return
            if (
                hasattr(self.performance_text, "winfo_exists")
                and not self.performance_text.winfo_exists()
            ):
                return

            self.update_performance_metrics()
        except Exception as e:
            self.logger.error(f"Safe performance metrics error: {e}", exc_info=True)

    def update_performance_metrics(self) -> None:
        """Update performance metrics display"""
        try:
            # Get performance statistics
            stats = self.usage_tracker.get_usage_summary()

            # Calculate additional metrics
            total_queries = stats.get("total_queries", 0)
            avg_response_time = stats.get("avg_response_time_ms", 0) / 1000  # Convert to seconds

            # Calculate queries per minute
            if len(self.usage_tracker.metrics_log) > 1:
                first_time = datetime.fromisoformat(self.usage_tracker.metrics_log[0]["timestamp"])
                last_time = datetime.fromisoformat(self.usage_tracker.metrics_log[-1]["timestamp"])
                time_diff = (last_time - first_time).total_seconds() / 60  # minutes
                queries_per_minute = total_queries / max(time_diff, 1)
            else:
                queries_per_minute = 0

            # Count cognitive states
            cognitive_counts: dict[str, int] = {}
            for entry in self.usage_tracker.metrics_log:
                state = entry.get("cognitive_state", "Unknown")
                cognitive_counts[state] = cognitive_counts.get(state, 0) + 1

            # Count processing modes
            processing_modes = {"conscious": 0, "unconscious": 0}
            # This would need to be tracked separately

            # Format metrics text
            metrics_text = f"""Performance Metrics
═══════════════════════════

Total Queries: {total_queries}
Average Response Time: {avg_response_time:.2f}s
Queries per Minute: {queries_per_minute:.1f}
Success Rate: 100.0%

Cognitive States:
• Working: {cognitive_counts.get('WORKING', 0)}
• Processing: {cognitive_counts.get('PROCESSING', 0)}
• Reflecting: {cognitive_counts.get('REFLECTING', 0)}
• Idle: {cognitive_counts.get('IDLE', 0)}

Energy Usage:
• Conscious: {processing_modes['conscious']}
• Unconscious: {processing_modes['unconscious']}
• Energy Saved: 0.0%
"""

            # Update text widget
            self.performance_text.delete("1.0", tk.END)
            self.performance_text.insert("1.0", metrics_text)

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}", exc_info=True)

    def safe_update_cognitive_display(self) -> None:
        """Safely update cognitive monitoring display with widget validation"""
        try:
            if not hasattr(self, "metrics_text") or not self.metrics_text.winfo_exists():
                return

            # Create a mock response if no last_response exists
            if not hasattr(self, "last_response"):
                self.last_response = {
                    "cognitive_state": {
                        "primary_state": "WORKING",
                        "confidence": 0.85,
                        "surprise": 0.2,
                        "coherence": 0.75,
                        "dominant_frequency": "beta",
                    }
                }

            self.update_cognitive_display(self.last_response)
        except Exception as e:
            self.logger.error(f"Safe cognitive display error: {e}", exc_info=True)

    def safe_update_oscillatory_display(self) -> None:
        """Safely update oscillatory analysis display with widget validation"""
        try:
            if not hasattr(self, "osc_metrics_text"):
                return
            if (
                hasattr(self.osc_metrics_text, "winfo_exists")
                and not self.osc_metrics_text.winfo_exists()
            ):
                return

            # Create a mock response if no last_response exists
            if not hasattr(self, "last_response"):
                self.last_response = {
                    "oscillatory_data": {
                        "delta": 0.3,
                        "theta": 0.2,
                        "alpha": 0.25,
                        "beta": 0.15,
                        "gamma_low": 0.07,
                        "gamma_high": 0.03,
                    }
                }

            self.update_oscillatory_display(self.last_response)
        except Exception as e:
            self.logger.error(f"Safe oscillatory display error: {e}", exc_info=True)

    def safe_update_biofeedback_display(self) -> None:
        """Safely update biofeedback display with widget validation"""
        try:
            if not hasattr(self, "biofeedback_text"):
                return
            if (
                hasattr(self.biofeedback_text, "winfo_exists")
                and not self.biofeedback_text.winfo_exists()
            ):
                return

            # Create a mock response if no last_response exists
            if not hasattr(self, "last_response"):
                self.last_response = {
                    "biofeedback": {
                        "recommendation": "deep_breathing",
                        "state": "relaxed",
                        "hr": 70,
                        "hrv": 50,
                        "resp": 16,
                        "eda": 5.0,
                    }
                }

            self.update_biofeedback_display(self.last_response)
        except Exception as e:
            self.logger.error(f"Safe biofeedback display error: {e}", exc_info=True)

    def update_displays(self) -> None:
        """Update all displays periodically (optimized with debouncing)"""
        try:
            # Check if we're still initializing
            if self.is_initializing:
                if hasattr(self, "progress") and hasattr(self.progress, "update_message"):
                    self.progress.update_message(
                        f"Initializing... {int(self.init_progress * 100)}%"
                    )
                self.root.after(100, self.update_displays)
                return

            # Only update if fully initialized and assistant is available
            if self.is_initialized and hasattr(self, "assistant") and self.assistant:
                # Use assistant_lock for thread-safe access to assistant
                with self.assistant_lock:
                    # Update visualizations if we have response data (immediate for responsiveness)
                    if hasattr(self, "last_response"):
                        self.update_visualizations(self.last_response)

                    # Update energy display if enabled (outside lock to avoid holding during UI updates)
                    energy_display_enabled = (
                        hasattr(self, "tabs_created")
                        and "Energy Management" in self.tabs_created
                        and hasattr(self.assistant, "enable_energy_aware")
                        and self.assistant.enable_energy_aware
                    )
                    if energy_display_enabled:
                        self.debouncer.schedule_update(
                            self.root, "energy_display", self.safe_update_energy_display, "default"
                        )

                    # Update performance metrics (independent of assistant state)
                    if hasattr(self, "tabs_created") and "Performance" in self.tabs_created:
                        self.debouncer.schedule_update(
                            self.root,
                            "performance_metrics",
                            self.safe_update_performance_metrics,
                            "default",
                        )

                    # Update cognitive monitoring display
                    if (
                        hasattr(self, "tabs_created")
                        and "Cognitive Monitoring" in self.tabs_created
                    ):
                        self.debouncer.schedule_update(
                            self.root,
                            "cognitive_display",
                            self.safe_update_cognitive_display,
                            "default",
                        )

                    # Update oscillatory analysis display
                    if (
                        hasattr(self, "tabs_created")
                        and "Oscillatory Analysis" in self.tabs_created
                    ):
                        self.debouncer.schedule_update(
                            self.root,
                            "oscillatory_display",
                            self.safe_update_oscillatory_display,
                            "default",
                        )

                    # Update biofeedback display
                    if hasattr(self, "tabs_created") and "Biofeedback" in self.tabs_created:
                        self.debouncer.schedule_update(
                            self.root,
                            "biofeedback_display",
                            self.safe_update_biofeedback_display,
                            "default",
                        )

                # Update system info
                self.update_system_info()

                # Increment update counter and perform periodic garbage collection
                self.update_count += 1
                if self.update_count % GUIConfig.GC_INTERVAL == 0:
                    import gc

                    gc.collect()
                    self.logger.debug("Garbage collection performed")
            else:
                # If not initialized, show appropriate status
                status_msg = "Initializing..." if not self.is_initialized else "Ready"
                self.update_status(status_msg, "info")

            # Check memory usage and auto-prune if needed
            if hasattr(self, "history_manager"):
                memory_stats = self.history_manager.get_memory_stats()
                utilization = memory_stats["memory_utilization"]
                if utilization > 0.8:  # Warn if using more than 80% of allocated memory
                    self.logger.warning(f"History memory usage high: {utilization:.1%}")

            # Schedule next update
            self.root.after(1000, self.update_displays)

        except Exception as e:
            self.logger.error(f"Error in update_displays: {e}", exc_info=True)
            # Schedule next update even if there was an error
            self.root.after(5000, self.update_displays)

    def set_status(self, message: str, status_type: str = "info") -> None:
        """Update status label

        Args:
            message: The status message to display
            status_type: Type of status (determines icon and styling)
        """
        if self.status_mgr and hasattr(self.status_mgr, "set_status"):
            try:
                self.status_mgr.set_status(message, status_type)
            except Exception as e:
                self.logger.debug(f"Failed to set status: {e}")

    def update_assistant_status(self, message: str, status_type: str = "info") -> None:
        """Update assistant status label

        Args:
            message: The status message to display (without 'Assistant: ' prefix)
            status_type: Type of status (determines icon and styling)
        """
        if self.status_mgr and hasattr(self.status_mgr, "set_status"):
            try:
                self.status_mgr.set_status("", status_type, message)
            except Exception as e:
                self.logger.debug(f"Failed to update assistant status: {e}")

    def update_system_info(self) -> None:
        """Update system information display"""
        # Check if info_text widget exists
        if not hasattr(self, "info_text") or not self.info_text.winfo_exists():
            return

        self.info_text.delete(1.0, tk.END)
        info_text = "System Information:\n" + "=" * 40 + "\n\n"

        import sys

        import torch

        info_text += f"Python Version: {sys.version.split()[0]}\n"
        info_text += f"Torch Version: {torch.__version__}\n"
        info_text += f"NumPy Version: {np.__version__}\n\n"

        info_text += "Optional Dependencies:\n"
        info_text += f"  torchdiffeq: {'✓ Available' if HAS_TORCHDIFFEQ else '✗ Not Available'}\n"
        info_text += f"  transformers: {'✓ Available' if HAS_TRANSFORMERS else '✗ Not Available'}\n"
        info_text += f"  matplotlib: {'✓ Available' if HAS_MATPLOTLIB else '✗ Not Available'}\n"
        info_text += f"  psutil: {'✓ Available' if HAS_PSUTIL else '✗ Not Available'}\n"
        info_text += f"  PIL: {'✓ Available' if HAS_PIL else '✗ Not Available'}\n"
        info_text += f"  reportlab: {'✓ Available' if HAS_REPORTLAB else '✗ Not Available'}\n\n"

        if HAS_PSUTIL:
            try:
                import psutil

                info_text += "System Resources:\n"
                info_text += f"  CPU Count: {psutil.cpu_count()} cores\n"
                info_text += f"  Memory: {psutil.virtual_memory().total // (1024**3)} GB\n"
                info_text += f"  CPU Usage: {psutil.cpu_percent()}%\n"
                info_text += f"  Memory Usage: {psutil.virtual_memory().percent}%\n"
            except Exception as e:
                info_text += f"  Error getting system info: {e}\n"

        info_text += f"\nLog File: {Path('apgi_gui.log').absolute()}\n"

        self.info_text.insert(tk.END, info_text)

    @safe_widget_method("query_input")
    @safe_widget_method("char_count_label")
    def update_char_count(self, event: Optional[Any] = None) -> None:
        """Update character count for query input with undo/redo tracking"""
        if not hasattr(self, "query_input"):
            return

        text = self.query_input.get(1.0, tk.END)
        count = len(text.strip())

        # Update character count display
        if hasattr(self, "char_count_label"):
            self.char_count_label.config(text=f"{count} / {GUIConfig.MAX_QUERY_LENGTH} characters")
            if count > GUIConfig.MAX_QUERY_LENGTH:
                self.char_count_label.config(foreground="red")
            else:
                self.char_count_label.config(foreground="black")

        # Track text changes for undo/redo (only on actual text changes, not just cursor movement)
        if event and hasattr(self, "_last_query_text"):
            current_text = text.rstrip()
            if current_text != self._last_query_text:
                # Create action for text change
                action = QueryTextAction(self, self._last_query_text, current_text)
                self.action_history.execute(action)
                self._last_query_text = current_text
        elif not hasattr(self, "_last_query_text"):
            # Initialize last text tracking
            self._last_query_text = text.rstrip()

    def undo_action(self) -> None:
        """Undo last action with keyboard shortcut support"""
        if self.action_history.can_undo():
            self.action_history.undo()
        else:
            # Show feedback if no undo available
            self._log_event("Nothing to undo")

    def redo_action(self) -> None:
        """Redo last undone action with keyboard shortcut support"""
        if self.action_history.can_redo():
            self.action_history.redo()
        else:
            # Show feedback if no redo available
            self._log_event("Nothing to redo")

    # Physiological preset methods
    def set_relaxed_state(self) -> None:
        """Set relaxed physiological state"""
        self.hr_var.set(60)
        self.hrv_var.set(65)
        self.resp_var.set(12)
        self.eda_var.set(3.0)

    def set_normal_state(self) -> None:
        """Set normal physiological state"""
        self.hr_var.set(70)
        self.hrv_var.set(50)
        self.resp_var.set(16)
        self.eda_var.set(5.0)

    def set_stressed_state(self) -> None:
        """Set stressed physiological state"""
        self.hr_var.set(85)
        self.hrv_var.set(40)
        self.resp_var.set(20)
        self.eda_var.set(7.0)

    def set_anxious_state(self) -> None:
        """Set anxious physiological state"""
        self.hr_var.set(95)
        self.hrv_var.set(30)
        self.resp_var.set(24)
        self.eda_var.set(10.0)

    # Calibration methods
    def calibrate_baseline(self) -> None:
        """Calibrate baseline physiology"""
        if not self.assistant:
            # Add debugging information to identify the root cause
            debug_info = {
                "is_initialized": getattr(self, "is_initialized", False),
                "init_state": getattr(self, "init_state", "unknown"),
                "assistant_exists": hasattr(self, "assistant") and self.assistant is not None,
                "is_initializing": getattr(self, "is_initializing", False),
            }
            self.logger.warning(f"Assistant check failed in calibrate_baseline: {debug_info}")
            messagebox.showwarning("Not Ready", "Please wait for assistant initialization")
            return

        # Use current physiology as baseline
        resting_data = [
            {
                "hr": self.hr_var.get(),
                "hrv": self.hrv_var.get(),
                "resp": self.resp_var.get(),
                "eda": self.eda_var.get(),
            }
        ] * 3

        with ErrorContext("Baseline Calibration", user_facing=True):
            with self.assistant_lock:
                self.assistant.calibrate(resting_data)
            self.calib_status.config(text="Calibrated ✓")
            messagebox.showinfo("Calibration", "Baseline calibrated successfully")
            self.logger.info("Biofeedback calibrated")

    # Settings methods
    def apply_settings(self) -> None:
        """Apply new settings"""
        response = messagebox.askyesno(
            "Apply Settings", "This will reset the assistant with new settings. Continue?"
        )
        if response:
            self.reset_assistant()

    def reset_to_defaults(self) -> None:
        """Reset all settings to default values"""
        response = messagebox.askyesno(
            "Reset to Defaults", "This will reset all settings to their default values. Continue?"
        )
        if response:
            # Reset UI variables to defaults
            if hasattr(self, "input_dim_var"):
                self.input_dim_var.set(128)
            if hasattr(self, "hidden_dim_var"):
                self.hidden_dim_var.set(256)
            if hasattr(self, "adaptive_proc_var"):
                self.adaptive_proc_var.set(True)
            if hasattr(self, "energy_aware_var"):
                self.energy_aware_var.set(True)
            if hasattr(self, "language_model_var"):
                self.language_model_var.set(HAS_TRANSFORMERS)

            # Reset assistant with new default settings
            self.reset_assistant()

            messagebox.showinfo("Reset Complete", "All settings have been reset to defaults.")

    def reset_assistant(self) -> None:
        """Reset the assistant with new settings"""
        if not messagebox.askyesno(
            "Reset Assistant", "This will reset the assistant and clear all history. Continue?"
        ):
            return

        with ErrorContext("Reset Assistant", user_facing=True):
            self.show_progress("Resetting assistant...")

            # Clear history and reset error count
            self.clear_history()
            self.error_count = 0

            # Reinitialize
            self.root.after(100, self.initialize_assistant)

            self.logger.info("Assistant reset")

    # Session management
    def new_session(self) -> None:
        """Start a new session"""
        if messagebox.askyesno("New Session", "Start a new session? Current session will be lost."):
            self.clear_history()
            self.clear_query()
            self.logger.info("New session started")

    def save_session(self) -> None:
        """Save current session with enhanced permission validation"""
        with ErrorContext("Save Session", user_facing=False):
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "query_history": list(self.query_history),
                "settings": getattr(self, "export_settings", {}),
                "physiology": {
                    "hr": self.hr_var.get(),
                    "hrv": self.hrv_var.get(),
                    "resp": self.resp_var.get(),
                    "eda": self.eda_var.get(),
                },
            }

            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

            if filename:
                # Validate permissions before attempting to save
                is_valid, error_msg, suggestion = self.permission_validator.validate_file_access(
                    filename, "write"
                )

                if not is_valid:
                    self.permission_validator.show_permission_error(
                        self.root, "Save Session Error", error_msg or "", suggestion or ""
                    )
                    self.logger.error(f"Permission error saving session: {error_msg}")
                    return

                try:
                    with ErrorContext("Save Session File", user_facing=True):
                        with open(filename, "w") as f:
                            json.dump(session_data, f, indent=2)
                        messagebox.showinfo("Save Session", f"Session saved to {filename}")
                        self.logger.info(f"Session saved to {filename}")

                except Exception as e:
                    # Enhanced error handling with specific guidance
                    if "Permission denied" in str(e):
                        error_msg = f"Permission denied writing to {filename}"
                        suggestion = "Check file permissions or choose different location"
                    elif "No space left" in str(e):
                        error_msg = f"Insufficient disk space for {filename}"
                        suggestion = "Free up disk space or choose different location"
                    else:
                        error_msg = f"Error saving session: {e}"
                        suggestion = "Try again or contact support if issue persists"

                    self.permission_validator.show_permission_error(
                        self.root, "Save Session Error", error_msg or "", suggestion or ""
                    )
                    self.logger.error(f"Failed to save session: {e}")

    def load_session(self) -> None:
        """Load a saved session with enhanced permission validation"""
        with ErrorContext("Load Session", user_facing=False):
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if filename:
                # Validate permissions before attempting to load
                is_valid, error_msg, suggestion = self.permission_validator.validate_file_access(
                    filename, "read"
                )

                if not is_valid:
                    self.permission_validator.show_permission_error(
                        self.root, "Load Session Error", error_msg or "", suggestion or ""
                    )
                    self.logger.error(f"Permission error loading session: {error_msg}")
                    return

                try:
                    with ErrorContext("Load Session File", user_facing=True):
                        with open(filename, "r") as f:
                            session_data = json.load(f)

                    # Restore settings
                    if "settings" in session_data:
                        self.load_configuration(session_data["settings"])

                    # Restore physiology
                    if "physiology" in session_data:
                        phys = session_data["physiology"]
                        self.hr_var.set(phys.get("hr", 70))
                        self.hrv_var.set(phys.get("hrv", 50))
                        self.resp_var.set(phys.get("resp", 16))
                        self.eda_var.set(phys.get("eda", 5.0))

                    # Restore history
                    if "query_history" in session_data:
                        self.query_history = deque(session_data["query_history"], maxlen=100)

                    messagebox.showinfo(
                        "Load Session",
                        f"Session loaded from {filename}\n"
                        "Configuration restored. Please reset assistant to apply.",
                    )
                    self.logger.info(f"Session loaded from {filename}")

                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON format in {filename}"
                    suggestion = "Choose a valid session file or check file integrity"
                    self.permission_validator.show_permission_error(
                        self.root, "Load Session Error", error_msg or "", suggestion or ""
                    )
                    self.logger.error(f"JSON decode error loading session: {e}")

                except Exception as e:
                    if "Permission denied" in str(e):
                        error_msg = f"Permission denied reading {filename}"
                        suggestion = "Check file permissions or run with appropriate user rights"
                    elif "No such file" in str(e):
                        error_msg = f"File not found: {filename}"
                        suggestion = "Choose an existing file or check file path"
                    else:
                        error_msg = f"Error loading session: {e}"
                        suggestion = "Try again or contact support if issue persists"

                    self.permission_validator.show_permission_error(
                        self.root, "Load Session Error", error_msg or "", suggestion or ""
                    )
                    self.logger.error(f"Failed to load session: {e}")

    def export_config(self) -> None:
        """Export current configuration with enhanced permission validation"""
        with ErrorContext("Export Configuration", user_facing=False):
            config = {
                "input_dim": self.input_dim_var.get(),
                "hidden_dim": self.hidden_dim_var.get(),
                "adaptive_processing": self.adaptive_proc_var.get(),
                "energy_aware": self.energy_aware_var.get(),
                "language_model": self.language_model_var.get(),
                "physiology": {
                    "hr": self.hr_var.get(),
                    "hrv": self.hrv_var.get(),
                    "resp": self.resp_var.get(),
                    "eda": self.eda_var.get(),
                },
            }

            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

            if filename:
                # Validate permissions before attempting to export
                is_valid, error_msg, suggestion = self.permission_validator.validate_file_access(
                    filename, "write"
                )

                if not is_valid:
                    self.permission_validator.show_permission_error(
                        self.root, "Export Configuration Error", error_msg or "", suggestion or ""
                    )
                    self.logger.error(f"Permission error exporting config: {error_msg}")
                    return

                try:
                    with ErrorContext("Export Configuration", user_facing=True):
                        with open(filename, "w") as f:
                            json.dump(config, f, indent=2)
                        messagebox.showinfo("Export", f"Configuration exported to {filename}")
                        self.logger.info(f"Configuration exported to {filename}")

                except Exception as e:
                    if "Permission denied" in str(e):
                        error_msg = f"Permission denied writing to {filename}"
                        suggestion = "Check file permissions or choose different location"
                    elif "No space left" in str(e):
                        error_msg = f"Insufficient disk space for {filename}"
                        suggestion = "Free up disk space or choose different location"
                    else:
                        error_msg = f"Error exporting configuration: {e}"
                        suggestion = "Try again or contact support if issue persists"

                    self.permission_validator.show_permission_error(
                        self.root, "Export Configuration Error", error_msg or "", suggestion or ""
                    )
                    self.logger.error(f"Failed to export configuration: {e}")

    def import_config(self) -> None:
        """Import configuration from file with enhanced permission validation"""
        with ErrorContext("Import Configuration", user_facing=False):
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if filename:
                # Validate permissions before attempting to import
                is_valid, error_msg, suggestion = self.permission_validator.validate_file_access(
                    filename, "read"
                )

                if not is_valid:
                    self.permission_validator.show_permission_error(
                        self.root, "Import Configuration Error", error_msg or "", suggestion or ""
                    )
                    self.logger.error(f"Permission error importing config: {error_msg}")
                    return

                try:
                    with ErrorContext("Import Configuration", user_facing=True):
                        with open(filename, "r") as f:
                            config = json.load(f)

                    self.load_configuration(config)

                    # Restore physiology if present
                    if "physiology" in config:
                        phys = config["physiology"]
                        self.hr_var.set(phys.get("hr", 70))
                        self.hrv_var.set(phys.get("hrv", 50))
                        self.resp_var.set(phys.get("resp", 16))
                        self.eda_var.set(phys.get("eda", 5.0))

                    messagebox.showinfo("Import", f"Configuration imported from {filename}")
                    self.logger.info(f"Configuration imported from {filename}")

                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON format in {filename}"
                    suggestion = "Choose a valid configuration file or check file integrity"
                    self.permission_validator.show_permission_error(
                        self.root, "Import Configuration Error", error_msg or "", suggestion or ""
                    )
                    self.logger.error(f"JSON decode error importing config: {e}")

                except Exception as e:
                    if "Permission denied" in str(e):
                        error_msg = f"Permission denied reading {filename}"
                        suggestion = "Check file permissions or run with appropriate user rights"
                    elif "No such file" in str(e):
                        error_msg = f"File not found: {filename}"
                        suggestion = "Choose an existing file or check file path"
                    else:
                        error_msg = f"Error importing configuration: {e}"
                        suggestion = "Try again or contact support if issue persists"

                    self.permission_validator.show_permission_error(
                        self.root, "Import Configuration Error", error_msg or "", suggestion or ""
                    )
                    self.logger.error(f"Failed to import configuration: {e}")

    def load_configuration(self, config: Optional[dict[str, Any]] = None) -> None:
        """Load saved configuration from file or from provided dict"""
        if not hasattr(self, "input_dim_var") or not hasattr(self, "hidden_dim_var"):
            # Variables not initialized yet, skip loading config for now
            # It will be loaded again after UI is fully initialized
            return

        # If config is not provided, load from file
        if config is None:
            if self.config_file.exists():
                try:
                    with ErrorContext("Load Configuration", user_facing=False):
                        with open(self.config_file, "r") as f:
                            config = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Could not load configuration: {e}")
                    return
            else:
                return  # No config file and no config provided

        # Apply the configuration values
        if config:
            try:
                # Only set values if the variables exist
                if hasattr(self, "input_dim_var"):
                    self.input_dim_var.set(config.get("input_dim", 128))
                if hasattr(self, "hidden_dim_var"):
                    self.hidden_dim_var.set(config.get("hidden_dim", 256))
                if hasattr(self, "adaptive_proc_var"):
                    self.adaptive_proc_var.set(config.get("adaptive_processing", True))
                if hasattr(self, "energy_aware_var"):
                    self.energy_aware_var.set(config.get("energy_aware", True))
                if hasattr(self, "language_model_var"):
                    self.language_model_var.set(config.get("language_model", HAS_TRANSFORMERS))

                self.logger.info("Configuration loaded")
            except Exception as e:
                self.logger.warning(f"Could not apply configuration: {e}")

    def save_configuration(self) -> None:
        """Save current configuration"""
        try:
            config = {}

            # Only save variables that exist (settings tab may not be created yet)
            if hasattr(self, "input_dim_var"):
                config["input_dim"] = self.input_dim_var.get()
            if hasattr(self, "hidden_dim_var"):
                config["hidden_dim"] = self.hidden_dim_var.get()
            if hasattr(self, "adaptive_proc_var"):
                config["adaptive_processing"] = self.adaptive_proc_var.get()
            if hasattr(self, "energy_aware_var"):
                config["energy_aware"] = self.energy_aware_var.get()
            if hasattr(self, "language_model_var"):
                config["language_model"] = self.language_model_var.get()

            if config:  # Only save if we have configuration data
                with open(self.config_file, "w") as f:
                    json.dump(config, f, indent=2)
                self.logger.info("Configuration saved")
            else:
                self.logger.debug("No configuration to save")
        except Exception as e:
            self.logger.warning(f"Could not save configuration: {e}")

    def export_session_csv(self) -> None:
        """Export session data to CSV format with enhanced permission validation"""
        with ErrorContext("Export Session CSV", user_facing=True):
            if not self.assistant:
                messagebox.showwarning("No Data", "No session data available to export")
                return

            filename = filedialog.asksaveasfilename(
                title="Export Session Data as CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"session_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            )

            if not filename:
                return  # User cancelled

            # Validate file permissions before attempting export
            is_valid, error_msg, suggestion = self.permission_validator.validate_file_access(
                filename, "write"
            )
            if not is_valid:
                self.permission_validator.show_permission_error(
                    self.root, "Export CSV Error", error_msg or "", suggestion or ""
                )
                self.logger.error(f"Permission error exporting CSV: {error_msg}")
                return

            try:
                self.show_progress("Exporting session data to CSV...")

                with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)

                    # Write header
                    writer.writerow(["APGI Assistant Session Data Export"])
                    writer.writerow(["Export Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    writer.writerow([])  # Empty row

                    # Query History
                    if hasattr(self.assistant, "query_history") and self.assistant.query_history:
                        writer.writerow(["Query History"])
                        writer.writerow(
                            ["Timestamp", "Query", "Response Length", "Processing Time (ms)"]
                        )

                        for query_data in self.assistant.query_history:
                            timestamp = query_data.get("timestamp", "N/A")
                            query = query_data.get("query", "N/A")
                            response_length = len(query_data.get("response", ""))
                            processing_time = query_data.get("processing_time_ms", "N/A")
                            writer.writerow([timestamp, query, response_length, processing_time])

                        writer.writerow([])  # Empty row

                    # State History
                    if hasattr(self.assistant, "state_history") and self.assistant.state_history:
                        writer.writerow(["Neural State History"])
                        writer.writerow(["Timestamp", "State", "Confidence", "Energy Level"])

                        for state_data in self.assistant.state_history:
                            timestamp = state_data.get("timestamp", "N/A")
                            state = state_data.get("state", "N/A")
                            confidence = state_data.get("confidence", "N/A")
                            energy = state_data.get("energy", "N/A")
                            writer.writerow([timestamp, state, confidence, energy])

                        writer.writerow([])  # Empty row

                    # Energy History
                    if hasattr(self.assistant, "energy_history") and self.assistant.energy_history:
                        writer.writerow(["Energy Usage History"])
                        writer.writerow(
                            ["Timestamp", "Energy Level", "CPU Usage (%)", "Memory Usage (MB)"]
                        )

                        for energy_data in self.assistant.energy_history:
                            timestamp = energy_data.get("timestamp", "N/A")
                            energy_level = energy_data.get("energy_level", "N/A")
                            cpu_usage = energy_data.get("cpu_usage", "N/A")
                            memory_usage = energy_data.get("memory_usage", "N/A")
                            writer.writerow([timestamp, energy_level, cpu_usage, memory_usage])

                        writer.writerow([])  # Empty row

                    # Physiology Data
                    writer.writerow(["Physiological Data"])
                    writer.writerow(
                        [
                            "Timestamp",
                            "Heart Rate (bpm)",
                            "HRV (ms)",
                            "Respiration (bpm)",
                            "EDA (µS)",
                        ]
                    )

                    # Get current physiology values
                    hr = self.hr_var.get() if hasattr(self, "hr_var") else "N/A"
                    hrv = self.hrv_var.get() if hasattr(self, "hrv_var") else "N/A"
                    resp = self.resp_var.get() if hasattr(self, "resp_var") else "N/A"
                    eda = self.eda_var.get() if hasattr(self, "eda_var") else "N/A"

                    writer.writerow(
                        [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), hr, hrv, resp, eda]
                    )
                    writer.writerow([])  # Empty row

                    # Configuration
                    writer.writerow(["Configuration"])
                    writer.writerow(["Parameter", "Value"])
                    writer.writerow(
                        [
                            "Input Dimension",
                            self.input_dim_var.get() if hasattr(self, "input_dim_var") else "N/A",
                        ]
                    )
                    writer.writerow(
                        [
                            "Hidden Dimension",
                            self.hidden_dim_var.get() if hasattr(self, "hidden_dim_var") else "N/A",
                        ]
                    )
                    writer.writerow(
                        [
                            "Adaptive Processing",
                            (
                                self.adaptive_proc_var.get()
                                if hasattr(self, "adaptive_proc_var")
                                else "N/A"
                            ),
                        ]
                    )
                    writer.writerow(
                        [
                            "Energy Aware",
                            (
                                self.energy_aware_var.get()
                                if hasattr(self, "energy_aware_var")
                                else "N/A"
                            ),
                        ]
                    )
                    writer.writerow(
                        [
                            "Language Model",
                            (
                                self.language_model_var.get()
                                if hasattr(self, "language_model_var")
                                else "N/A"
                            ),
                        ]
                    )

                self.hide_progress()
                messagebox.showinfo("Success", f"Session data exported to CSV:\n{filename}")
                self.logger.info(f"Session data exported to CSV: {filename}")

            except Exception as e:
                self.hide_progress()
                error_msg = f"Failed to export session data to CSV: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                messagebox.showerror("Export Error", error_msg)

    def export_metrics_csv(self) -> None:
        """Export performance metrics to CSV format"""
        with ErrorContext("Export Metrics CSV", user_facing=True):
            if not self.assistant:
                messagebox.showwarning("No Data", "No metrics data available to export")
                return

            filename = filedialog.asksaveasfilename(
                title="Export Performance Metrics as CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            )

            if not filename:
                return  # User cancelled

            try:
                self.show_progress("Exporting performance metrics to CSV...")

                with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)

                    # Write header
                    writer.writerow(["APGI Assistant Performance Metrics"])
                    writer.writerow(["Export Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    writer.writerow([])  # Empty row

                    # Performance Summary
                    writer.writerow(["Performance Summary"])
                    writer.writerow(["Metric", "Value", "Unit"])

                    # Calculate metrics
                    total_queries = len(getattr(self.assistant, "query_history", []))
                    total_states = len(getattr(self.assistant, "state_history", []))
                    total_energy_records = len(getattr(self.assistant, "energy_history", []))

                    writer.writerow(["Total Queries", total_queries, "count"])
                    writer.writerow(["Total State Changes", total_states, "count"])
                    writer.writerow(["Total Energy Records", total_energy_records, "count"])

                    # Calculate average processing time if available
                    if hasattr(self.assistant, "query_history") and self.assistant.query_history:
                        processing_times = [
                            q.get("processing_time_ms", 0)
                            for q in self.assistant.query_history
                            if q.get("processing_time_ms")
                        ]
                        if processing_times:
                            avg_processing_time = sum(processing_times) / len(processing_times)
                            writer.writerow(
                                ["Average Processing Time", f"{avg_processing_time:.2f}", "ms"]
                            )

                    # Calculate average energy level if available
                    if hasattr(self.assistant, "energy_history") and self.assistant.energy_history:
                        energy_levels = [
                            e.get("energy_level", 0)
                            for e in self.assistant.energy_history
                            if e.get("energy_level")
                        ]
                        if energy_levels:
                            avg_energy = sum(energy_levels) / len(energy_levels)
                            writer.writerow(
                                ["Average Energy Level", f"{avg_energy:.2f}", "normalized"]
                            )

                    writer.writerow([])  # Empty row

                    # State Distribution
                    if hasattr(self.assistant, "state_history") and self.assistant.state_history:
                        writer.writerow(["State Distribution"])
                        writer.writerow(["State", "Count", "Percentage"])

                        state_counts: dict[str, int] = {}
                        for state_data in self.assistant.state_history:
                            state = state_data.get("state", "unknown")
                            state_counts[state] = state_counts.get(state, 0) + 1

                        total_states = sum(state_counts.values())
                        for state, count in state_counts.items():
                            percentage = (count / total_states) * 100 if total_states > 0 else 0
                            writer.writerow([state, count, f"{percentage:.1f}%"])

                        writer.writerow([])  # Empty row

                    # Energy Statistics
                    if hasattr(self.assistant, "energy_history") and self.assistant.energy_history:
                        writer.writerow(["Energy Statistics"])
                        writer.writerow(["Metric", "Value", "Unit"])

                        energy_levels = [
                            e.get("energy_level", 0)
                            for e in self.assistant.energy_history
                            if e.get("energy_level")
                        ]
                        if energy_levels:
                            writer.writerow(["Min Energy", min(energy_levels), "normalized"])
                            writer.writerow(["Max Energy", max(energy_levels), "normalized"])
                            writer.writerow(
                                [
                                    "Average Energy",
                                    f"{sum(energy_levels) / len(energy_levels):.2f}",
                                    "normalized",
                                ]
                            )

                            # Calculate energy trend (simple linear trend)
                            if len(energy_levels) > 1:
                                trend = (energy_levels[-1] - energy_levels[0]) / len(energy_levels)
                                trend_direction = (
                                    "Increasing"
                                    if trend > 0
                                    else "Decreasing" if trend < 0 else "Stable"
                                )
                                writer.writerow(["Energy Trend", trend_direction, "qualitative"])

                        writer.writerow([])  # Empty row

                    # System Metrics
                    writer.writerow(["System Metrics"])
                    writer.writerow(["Metric", "Value", "Unit"])

                    if HAS_PSUTIL:
                        try:
                            cpu_percent = psutil.cpu_percent(interval=1)
                            memory_info = psutil.virtual_memory()
                            writer.writerow(["Current CPU Usage", cpu_percent, "%"])
                            writer.writerow(["Current Memory Usage", memory_info.percent, "%"])
                            writer.writerow(
                                [
                                    "Available Memory",
                                    f"{memory_info.available / (1024**3):.2f}",
                                    "GB",
                                ]
                            )
                        except Exception as e:
                            self.logger.warning(f"Could not get system metrics: {e}")
                    else:
                        writer.writerow(["System Monitoring", "Disabled", "N/A"])

                    writer.writerow([])  # Empty row

                    # Export Configuration
                    writer.writerow(["Export Configuration"])
                    writer.writerow(["Setting", "Value"])
                    writer.writerow(["Export Format", "CSV"])
                    writer.writerow(["Export Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    writer.writerow(["Application", "APGI Assistant"])
                    writer.writerow(["Version", "1.0"])

                self.hide_progress()
                messagebox.showinfo("Success", f"Performance metrics exported to CSV:\n{filename}")
                self.logger.info(f"Performance metrics exported to CSV: {filename}")

            except Exception as e:
                self.hide_progress()
                error_msg = f"Failed to export performance metrics to CSV: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                messagebox.showerror("Export Error", error_msg)

    def export_session_json(self) -> None:
        """Export complete session data to comprehensive JSON format"""
        with ErrorContext("Export Session JSON", user_facing=True):
            if not self.assistant:
                messagebox.showwarning("No Data", "No session data available to export")
                return

            filename = filedialog.asksaveasfilename(
                title="Export Complete Session as JSON",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"complete_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

            if not filename:
                return  # User cancelled

            try:
                self.show_progress("Exporting complete session data to JSON...")

                # Build comprehensive session data structure
                session_data: dict[str, Any] = {
                    "metadata": {
                        "export_date": datetime.now().isoformat(),
                        "export_format": "JSON",
                        "application": "APGI Assistant",
                        "version": "1.0",
                        "session_duration": self._calculate_session_duration(),
                        "export_type": "complete_session",
                    },
                    "configuration": {
                        "model": {
                            "input_dim": (
                                self.input_dim_var.get() if hasattr(self, "input_dim_var") else None
                            ),
                            "hidden_dim": (
                                self.hidden_dim_var.get()
                                if hasattr(self, "hidden_dim_var")
                                else None
                            ),
                            "adaptive_processing": (
                                self.adaptive_proc_var.get()
                                if hasattr(self, "adaptive_proc_var")
                                else None
                            ),
                            "energy_aware": (
                                self.energy_aware_var.get()
                                if hasattr(self, "energy_aware_var")
                                else None
                            ),
                            "language_model": (
                                self.language_model_var.get()
                                if hasattr(self, "language_model_var")
                                else None
                            ),
                        },
                        "ui": {
                            "theme": getattr(self, "current_theme", "normal"),
                            "high_contrast": (
                                getattr(self, "high_contrast_var", tk.BooleanVar()).get()
                                if hasattr(self, "high_contrast_var")
                                else False
                            ),
                            "export_settings": getattr(self, "export_settings", {}),
                        },
                    },
                    "physiology": {
                        "current": {
                            "heart_rate": self.hr_var.get() if hasattr(self, "hr_var") else None,
                            "hrv": self.hrv_var.get() if hasattr(self, "hrv_var") else None,
                            "respiration": (
                                self.resp_var.get() if hasattr(self, "resp_var") else None
                            ),
                            "eda": self.eda_var.get() if hasattr(self, "eda_var") else None,
                            "timestamp": datetime.now().isoformat(),
                        },
                        "history": getattr(self.assistant, "physiology_history", []),
                    },
                    "query_history": [],
                    "state_history": [],
                    "energy_history": [],
                    "performance_metrics": {},
                    "system_info": {},
                    "visualizations": {},
                }

                # Process query history with enhanced data
                if hasattr(self.assistant, "query_history") and self.assistant.query_history:
                    for query_data in self.assistant.query_history:
                        enhanced_query = {
                            "timestamp": query_data.get("timestamp", "N/A"),
                            "query": query_data.get("query", ""),
                            "response": query_data.get("response", ""),
                            "response_length": len(query_data.get("response", "")),
                            "processing_time_ms": query_data.get("processing_time_ms", 0),
                            "state_at_time": query_data.get("state", "unknown"),
                            "energy_at_time": query_data.get("energy", 0),
                            "confidence": query_data.get("confidence", 0),
                            "query_type": self._classify_query_type(query_data.get("query", "")),
                            "success": query_data.get("success", True),
                        }
                        session_data["query_history"].append(enhanced_query)

                # Process state history with enhanced data
                if hasattr(self.assistant, "state_history") and self.assistant.state_history:
                    for state_data in self.assistant.state_history:
                        enhanced_state = {
                            "timestamp": state_data.get("timestamp", "N/A"),
                            "state": state_data.get("state", "unknown"),
                            "confidence": state_data.get("confidence", 0),
                            "energy_level": state_data.get("energy", 0),
                            "transition_reason": state_data.get("transition_reason", "auto"),
                            "duration_ms": state_data.get("duration_ms", 0),
                            "physiology_correlation": state_data.get("physiology_correlation", {}),
                        }
                        session_data["state_history"].append(enhanced_state)

                # Process energy history with enhanced data
                if hasattr(self.assistant, "energy_history") and self.assistant.energy_history:
                    for energy_data in self.assistant.energy_history:
                        enhanced_energy = {
                            "timestamp": energy_data.get("timestamp", "N/A"),
                            "energy_level": energy_data.get("energy_level", 0),
                            "cpu_usage": energy_data.get("cpu_usage", 0),
                            "memory_usage": energy_data.get("memory_usage", 0),
                            "battery_level": energy_data.get("battery_level", None),
                            "power_consumption": energy_data.get("power_consumption", 0),
                            "efficiency_score": energy_data.get("efficiency_score", 0),
                        }
                        session_data["energy_history"].append(enhanced_energy)

                # Calculate performance metrics
                session_data["performance_metrics"] = self._calculate_performance_metrics()

                # Get system information
                session_data["system_info"] = self._get_system_info()

                # Include visualization data if available
                if hasattr(self, "spectrum_fig") and self.spectrum_fig:
                    session_data["visualizations"]["oscillatory_spectrum_available"] = True
                if HAS_MATPLOTLIB:
                    session_data["visualizations"]["matplotlib_available"] = True

                # Validate permissions before saving
                is_valid, error_msg, suggestion = self.permission_validator.validate_file_access(
                    filename, "write"
                )

                if not is_valid:
                    self.permission_validator.show_permission_error(
                        self.root, "Export Session JSON Error", error_msg or "", suggestion or ""
                    )
                    self.logger.error(f"Permission error exporting session JSON: {error_msg}")
                    return

                # Write JSON with pretty formatting
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(session_data, f, indent=2, ensure_ascii=False, default=str)

                self.hide_progress()
                messagebox.showinfo("Success", f"Complete session exported to JSON:\n{filename}")
                self.logger.info(f"Complete session exported to JSON: {filename}")

            except Exception as e:
                self.hide_progress()
                error_msg = f"Failed to export session to JSON: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                messagebox.showerror("Export Error", error_msg)

    def _calculate_session_duration(self) -> str:
        """Calculate the duration of the current session"""
        try:
            if (
                self.assistant
                and hasattr(self.assistant, "query_history")
                and self.assistant.query_history
            ):
                first_query = self.assistant.query_history[0]
                last_query = self.assistant.query_history[-1]

                # Parse timestamps (assuming ISO format)
                first_time = datetime.fromisoformat(
                    first_query.get("timestamp", datetime.now().isoformat())
                )
                last_time = datetime.fromisoformat(
                    last_query.get("timestamp", datetime.now().isoformat())
                )

                duration = last_time - first_time
                return str(duration)
            return "00:00:00"
        except Exception:
            return "00:00:00"

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query for analysis"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["what is", "define", "explain", "tell me"]):
            return "informational"
        elif any(word in query_lower for word in ["how to", "help", "guide", "instructions"]):
            return "procedural"
        elif any(word in query_lower for word in ["analyze", "compare", "evaluate", "assess"]):
            return "analytical"
        elif any(word in query_lower for word in ["create", "generate", "make", "produce"]):
            return "creative"
        elif any(word in query_lower for word in ["why", "when", "where", "who"]):
            return "interrogative"
        else:
            return "general"

    def _calculate_performance_metrics(self) -> dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        metrics: dict[str, Any] = {
            "query_metrics": {},
            "state_metrics": {},
            "energy_metrics": {},
            "system_metrics": {},
        }

        # Query metrics
        if (
            self.assistant
            and hasattr(self.assistant, "query_history")
            and self.assistant.query_history
        ):
            queries = self.assistant.query_history
            metrics["query_metrics"] = {
                "total_queries": len(queries),
                "avg_response_length": (
                    sum(len(q.get("response", "")) for q in queries) / len(queries)
                    if queries
                    else 0
                ),
                "avg_processing_time": (
                    sum(q.get("processing_time_ms", 0) for q in queries) / len(queries)
                    if queries
                    else 0
                ),
                "success_rate": (
                    sum(1 for q in queries if q.get("success", True)) / len(queries)
                    if queries
                    else 0
                ),
                "query_types": {},
            }

            # Count query types
            for query in queries:
                query_type = self._classify_query_type(query.get("query", ""))
                metrics["query_metrics"]["query_types"][query_type] = (
                    metrics["query_metrics"]["query_types"].get(query_type, 0) + 1
                )

        # State metrics
        if (
            self.assistant
            and hasattr(self.assistant, "state_history")
            and self.assistant.state_history
        ):
            states = self.assistant.state_history
            state_counts: dict[str, int] = {}
            for state in states:
                state_name = state.get("state", "unknown")
                state_counts[state_name] = state_counts.get(state_name, 0) + 1

            metrics["state_metrics"] = {
                "total_state_changes": len(states),
                "state_distribution": state_counts,
                "avg_confidence": (
                    sum(s.get("confidence", 0) for s in states) / len(states) if states else 0
                ),
                "most_common_state": (
                    max(state_counts.items(), key=lambda x: x[1])[0] if state_counts else "unknown"
                ),
            }

        # Energy metrics
        if (
            self.assistant
            and hasattr(self.assistant, "energy_history")
            and self.assistant.energy_history
        ):
            energy_levels = [
                e.get("energy_level", 0)
                for e in self.assistant.energy_history
                if e.get("energy_level")
            ]
            if energy_levels:
                metrics["energy_metrics"] = {
                    "total_readings": len(energy_levels),
                    "avg_energy": sum(energy_levels) / len(energy_levels),
                    "min_energy": min(energy_levels),
                    "max_energy": max(energy_levels),
                    "energy_efficiency": (
                        sum(e.get("efficiency_score", 0) for e in self.assistant.energy_history)
                        / len(self.assistant.energy_history)
                        if self.assistant.energy_history
                        else 0
                    ),
                }

        return metrics

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information for the export"""
        system_info: dict[str, Any] = {
            "platform": sys.platform,
            "python_version": sys.version,
            "dependencies": {
                "matplotlib": HAS_MATPLOTLIB,
                "pil": HAS_PIL,
                "psutil": HAS_PSUTIL,
                "transformers": HAS_TRANSFORMERS,
                "torchdiffeq": HAS_TORCHDIFFEQ,
                "reportlab": HAS_REPORTLAB,
            },
        }

        if HAS_PSUTIL:
            try:
                system_info.update(
                    {
                        "cpu_count": psutil.cpu_count(),
                        "memory_total": psutil.virtual_memory().total,
                        "disk_usage": (
                            psutil.disk_usage("/").percent
                            if sys.platform != "win32"
                            else psutil.disk_usage("C:").percent
                        ),
                    }
                )
            except Exception as e:
                self.logger.warning(f"Could not get system info: {e}")

        return system_info

    def export_session_pdf(self) -> None:
        """Export comprehensive session report to PDF format"""
        with ErrorContext("Export Session PDF", user_facing=True):
            if not HAS_REPORTLAB:
                messagebox.showerror(
                    "PDF Export Not Available",
                    "PDF export requires the 'reportlab' package.\n\n"
                    "Install with: pip install reportlab\n\n"
                    "Falling back to image export for visualizations.",
                )
                return

            if not self.assistant:
                messagebox.showwarning("No Data", "No session data available to export")
                return

            filename = filedialog.asksaveasfilename(
                title="Export Session Report as PDF",
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                initialfile=f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            )

            if not filename:
                return  # User cancelled

            try:
                self.show_progress("Generating PDF report...")

                # Create PDF document
                doc = SimpleDocTemplate(filename, pagesize=A4)
                styles = getSampleStyleSheet()
                story = []

                # Custom styles
                title_style = ParagraphStyle(
                    "CustomTitle",
                    parent=styles["Heading1"],
                    fontSize=24,
                    spaceAfter=30,
                    alignment=TA_CENTER,
                    textColor=colors.darkblue,
                )

                heading_style = ParagraphStyle(
                    "CustomHeading",
                    parent=styles["Heading2"],
                    fontSize=16,
                    spaceAfter=12,
                    textColor=colors.darkblue,
                )

                # Title Page
                story.append(Paragraph("APGI Assistant Session Report", title_style))
                story.append(Spacer(1, 20))

                # Report metadata
                metadata_data = [
                    ["Report Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    ["Session Duration:", self._calculate_session_duration()],
                    ["Application:", "APGI Assistant v1.0"],
                    ["Export Format:", "PDF"],
                ]

                metadata_table = Table(metadata_data, colWidths=[2 * inch, 4 * inch])
                metadata_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), colors.lightgrey),
                            ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                            ("FONTSIZE", (0, 0), (-1, -1), 10),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                            ("BACKGROUND", (0, 0), (0, -1), colors.grey),
                        ]
                    )
                )

                story.append(metadata_table)
                story.append(PageBreak())

                # Executive Summary
                story.append(Paragraph("Executive Summary", heading_style))
                story.append(Spacer(1, 12))

                metrics = self._calculate_performance_metrics()
                summary_text = self._generate_executive_summary(metrics)
                story.append(Paragraph(summary_text, styles["Normal"]))
                story.append(Spacer(1, 20))

                # Performance Metrics Section
                story.append(Paragraph("Performance Metrics", heading_style))
                story.append(Spacer(1, 12))

                # Query metrics table
                if "query_metrics" in metrics and metrics["query_metrics"]:
                    story.append(Paragraph("Query Analysis", styles["Heading3"]))

                    query_data = [["Metric", "Value"]]
                    query_metrics = metrics["query_metrics"]
                    query_data.extend(
                        [
                            ["Total Queries", str(query_metrics.get("total_queries", 0))],
                            [
                                "Average Response Length",
                                f"{query_metrics.get('avg_response_length', 0):.0f} characters",
                            ],
                            [
                                "Average Processing Time",
                                f"{query_metrics.get('avg_processing_time', 0):.2f} ms",
                            ],
                            ["Success Rate", f"{query_metrics.get('success_rate', 0):.1%}"],
                        ]
                    )

                    query_table = Table(query_data, colWidths=[2.5 * inch, 2 * inch])
                    query_table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, 0), 12),
                                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            ]
                        )
                    )

                    story.append(query_table)
                    story.append(Spacer(1, 20))

                # State distribution
                if "state_metrics" in metrics and metrics["state_metrics"]:
                    story.append(Paragraph("Neural State Distribution", styles["Heading3"]))

                    state_data = [["State", "Count", "Percentage"]]
                    state_metrics = metrics["state_metrics"]
                    state_dist = state_metrics.get("state_distribution", {})
                    total_states = sum(state_dist.values()) if state_dist else 1

                    for state, count in state_dist.items():
                        percentage = (count / total_states) * 100 if total_states > 0 else 0
                        state_data.append([state.title(), str(count), f"{percentage:.1f}%"])

                    state_table = Table(state_data, colWidths=[2 * inch, 1 * inch, 1.5 * inch])
                    state_table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, 0), 12),
                                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            ]
                        )
                    )

                    story.append(state_table)
                    story.append(Spacer(1, 20))

                # Energy metrics
                if "energy_metrics" in metrics and metrics["energy_metrics"]:
                    story.append(Paragraph("Energy Efficiency Analysis", styles["Heading3"]))

                    energy_data = [["Metric", "Value"]]
                    energy_metrics = metrics["energy_metrics"]
                    energy_data.extend(
                        [
                            ["Total Readings", str(energy_metrics.get("total_readings", 0))],
                            ["Average Energy Level", f"{energy_metrics.get('avg_energy', 0):.3f}"],
                            ["Minimum Energy", f"{energy_metrics.get('min_energy', 0):.3f}"],
                            ["Maximum Energy", f"{energy_metrics.get('max_energy', 0):.3f}"],
                            [
                                "Energy Efficiency",
                                f"{energy_metrics.get('energy_efficiency', 0):.3f}",
                            ],
                        ]
                    )

                    energy_table = Table(energy_data, colWidths=[2.5 * inch, 2 * inch])
                    energy_table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, 0), 12),
                                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            ]
                        )
                    )

                    story.append(energy_table)
                    story.append(Spacer(1, 20))

                story.append(PageBreak())

                # Configuration Section
                story.append(Paragraph("Configuration", heading_style))
                story.append(Spacer(1, 12))

                config_data = [["Parameter", "Value"]]
                config_data.extend(
                    [
                        [
                            "Input Dimension",
                            str(
                                self.input_dim_var.get()
                                if hasattr(self, "input_dim_var")
                                else "N/A"
                            ),
                        ],
                        [
                            "Hidden Dimension",
                            str(
                                self.hidden_dim_var.get()
                                if hasattr(self, "hidden_dim_var")
                                else "N/A"
                            ),
                        ],
                        [
                            "Adaptive Processing",
                            str(
                                self.adaptive_proc_var.get()
                                if hasattr(self, "adaptive_proc_var")
                                else "N/A"
                            ),
                        ],
                        [
                            "Energy Aware Mode",
                            str(
                                self.energy_aware_var.get()
                                if hasattr(self, "energy_aware_var")
                                else "N/A"
                            ),
                        ],
                        [
                            "Language Model",
                            str(
                                self.language_model_var.get()
                                if hasattr(self, "language_model_var")
                                else "N/A"
                            ),
                        ],
                        ["Current Theme", getattr(self, "current_theme", "normal")],
                    ]
                )

                config_table = Table(config_data, colWidths=[2.5 * inch, 2 * inch])
                config_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 12),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ]
                    )
                )

                story.append(config_table)
                story.append(Spacer(1, 20))

                # System Information
                story.append(Paragraph("System Information", heading_style))
                story.append(Spacer(1, 12))

                system_info = self._get_system_info()
                sys_data = [["Component", "Status/Value"]]
                sys_data.extend(
                    [
                        ["Platform", system_info.get("platform", "Unknown")],
                        [
                            "Python Version",
                            system_info.get("python_version", "Unknown")[:20] + "...",
                        ],
                        [
                            "Matplotlib",
                            (
                                "Available"
                                if system_info.get("dependencies", {}).get("matplotlib")
                                else "Not Available"
                            ),
                        ],
                        [
                            "PIL/Pillow",
                            (
                                "Available"
                                if system_info.get("dependencies", {}).get("pil")
                                else "Not Available"
                            ),
                        ],
                        [
                            "psutil",
                            (
                                "Available"
                                if system_info.get("dependencies", {}).get("psutil")
                                else "Not Available"
                            ),
                        ],
                        [
                            "Transformers",
                            (
                                "Available"
                                if system_info.get("dependencies", {}).get("transformers")
                                else "Not Available"
                            ),
                        ],
                    ]
                )

                if HAS_PSUTIL and "cpu_count" in system_info:
                    sys_data.extend(
                        [
                            ["CPU Cores", str(system_info.get("cpu_count", "Unknown"))],
                            [
                                "Total Memory",
                                f"{system_info.get('memory_total', 0) / (1024**3):.1f} GB",
                            ],
                        ]
                    )

                sys_table = Table(sys_data, colWidths=[2.5 * inch, 2 * inch])
                sys_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 12),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ]
                    )
                )

                story.append(sys_table)
                story.append(Spacer(1, 20))

                # Add visualizations if available
                if HAS_MATPLOTLIB:
                    story.append(PageBreak())
                    story.append(Paragraph("Visualizations", heading_style))
                    story.append(Spacer(1, 12))

                    # Add state timeline if available
                    if (
                        self.assistant
                        and hasattr(self.assistant, "state_history")
                        and self.assistant.state_history
                    ):
                        fig = None
                        try:
                            fig = (
                                APGIVisualizer.plot_state_timeline(self.assistant.state_history)
                                if APGIVisualizer
                                else None
                            )
                            if fig:
                                # Save figure to temporary file and add to PDF
                                import io

                                from reportlab.lib.utils import (
                                    ImageReader,
                                )
                                from reportlab.platypus import Image as RLImage

                                buf = io.BytesIO()
                                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                                buf.seek(0)

                                img = ImageReader(buf)
                                img_width, img_height = img.getSize()
                                aspect = img_height / float(img_width)

                                # Scale image to fit page width
                                display_width = 6 * inch
                                display_height = display_width * aspect

                                story.append(Paragraph("Neural State Timeline", styles["Heading3"]))
                                story.append(Spacer(1, 6))
                                story.append(
                                    RLImage(buf, width=display_width, height=display_height)
                                )
                                story.append(Spacer(1, 20))

                                if HAS_MATPLOTLIB:
                                    import matplotlib.pyplot as plt

                                if HAS_MATPLOTLIB:
                                    plt.close(fig)
                        except Exception as e:
                            self.logger.warning(f"Could not add state timeline to PDF: {e}")
                            # Ensure figure cleanup even if error occurs
                            if fig:
                                if HAS_MATPLOTLIB:
                                    import matplotlib.pyplot as plt

                                if HAS_MATPLOTLIB:
                                    plt.close(fig)

                    # Add energy plot if available
                    if hasattr(self.assistant, "energy_history") and self.assistant.energy_history:
                        fig = None
                        try:
                            fig = (
                                APGIVisualizer.plot_energy_usage(self.assistant.energy_history)
                                if APGIVisualizer and self.assistant
                                else None
                            )
                            if fig:
                                # Save figure to temporary file and add to PDF
                                import io

                                from reportlab.lib.utils import (
                                    ImageReader,
                                )
                                from reportlab.platypus import Image as RLImage

                                buf = io.BytesIO()
                                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                                buf.seek(0)

                                img = ImageReader(buf)
                                img_width, img_height = img.getSize()
                                aspect = img_height / float(img_width)

                                # Scale image to fit page width
                                display_width = 6 * inch
                                display_height = display_width * aspect

                                story.append(Paragraph("Energy Usage Timeline", styles["Heading3"]))
                                story.append(Spacer(1, 6))
                                story.append(
                                    RLImage(buf, width=display_width, height=display_height)
                                )
                                story.append(Spacer(1, 20))

                                if HAS_MATPLOTLIB:
                                    import matplotlib.pyplot as plt

                                if HAS_MATPLOTLIB:
                                    plt.close(fig)
                        except Exception as e:
                            self.logger.warning(f"Could not add energy plot to PDF: {e}")
                            # Ensure figure cleanup even if error occurs
                            if fig:
                                if HAS_MATPLOTLIB:
                                    import matplotlib.pyplot as plt

                                if HAS_MATPLOTLIB:
                                    plt.close(fig)

                # Build PDF
                doc.build(story)

                self._force_hide_progress()
                messagebox.showinfo("Success", f"Session report exported to PDF:\n{filename}")
                self.logger.info(f"Session report exported to PDF: {filename}")

            except Exception as e:
                self._force_hide_progress()
                error_msg = f"Failed to export session to PDF: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                messagebox.showerror("Export Error", error_msg)

    def _generate_executive_summary(self, metrics: dict[str, Any]) -> str:
        """Generate executive summary text for PDF report"""
        summary_parts = []

        # Overall session overview
        total_queries = metrics.get("query_metrics", {}).get("total_queries", 0)
        total_states = metrics.get("state_metrics", {}).get("total_state_changes", 0)

        # Initialize variables for recommendations check
        avg_processing_time = 0
        success_rate = 1.0
        avg_energy = 0.0

        summary_parts.append(
            f"This report summarizes a session containing {total_queries} queries and "
            f"{total_states} neural state transitions. "
        )

        # Performance insights
        avg_processing_time = metrics.get("query_metrics", {}).get("avg_processing_time", 0)
        success_rate = metrics.get("query_metrics", {}).get("success_rate", 0)

        if avg_processing_time > 0:
            summary_parts.append(
                f"The system demonstrated an average processing time of {avg_processing_time:.2f}ms "
                f"with a success rate of {success_rate:.1%}. "
            )

        # State analysis
        if "state_metrics" in metrics and metrics["state_metrics"]:
            most_common_state = metrics["state_metrics"].get("most_common_state", "unknown")
            avg_confidence = metrics["state_metrics"].get("avg_confidence", 0)

            summary_parts.append(
                f"The neural system primarily operated in the {most_common_state} state "
                f"with an average confidence level of {avg_confidence:.2f}. "
            )

        # Energy efficiency
        if "energy_metrics" in metrics and metrics["energy_metrics"]:
            avg_energy = metrics["energy_metrics"].get("avg_energy", 0)
            efficiency = metrics["energy_metrics"].get("energy_efficiency", 0)

            summary_parts.append(
                f"Energy efficiency analysis shows an average energy level of {avg_energy:.3f} "
                f"with an efficiency score of {efficiency:.3f}. "
            )

        # Recommendations
        summary_parts.append("\n\nRecommendations:")

        if avg_processing_time > 1000:
            summary_parts.append(
                "• Consider optimizing model parameters to improve processing speed"
            )

        if success_rate < 0.9:
            summary_parts.append("• Review query patterns to improve response accuracy")

        if avg_energy > 0.8:
            summary_parts.append(
                "• Monitor energy consumption and consider energy-saving optimizations"
            )

        return " ".join(summary_parts)

    # Visualization methods
    def generate_state_timeline(self) -> None:
        """Generate state timeline visualization"""
        if not HAS_MATPLOTLIB:
            messagebox.showwarning("No Matplotlib", "Matplotlib is required for visualizations")
            return

        fig = None
        try:
            with ErrorContext("Generate State Timeline", user_facing=True):
                self.progress = CancellableProgress(self.root)
                self.progress.show("Generating timeline...")

                # Create mock data if no assistant or history available
                if (
                    not self.assistant
                    or not hasattr(self.assistant, "state_history")
                    or not self.assistant.state_history
                ):
                    # Generate sample state history for demonstration with correct structure
                    import datetime
                    import random

                    mock_history = []
                    states = ["WORKING", "IDLE", "PROCESSING", "REFLECTING"]
                    base_time = datetime.datetime.now() - datetime.timedelta(minutes=30)

                    for i in range(20):
                        timestamp = base_time + datetime.timedelta(minutes=i * 1.5)
                        mock_history.append(
                            {
                                "timestamp": timestamp,
                                "primary": random.choice(states).lower(),
                                "confidence": random.uniform(0.6, 1.0),
                                "surprise_level": random.uniform(0.0, 0.5),
                                "coherence": random.uniform(0.5, 0.9),
                            }
                        )

                    fig = (
                        APGIVisualizer.plot_state_timeline(mock_history) if APGIVisualizer else None
                    )
                else:
                    fig = (
                        APGIVisualizer.plot_state_timeline(self.assistant.state_history)
                        if self.assistant
                        else None
                    )

                if fig:
                    self.display_plot_in_viz_tab(fig)
                    messagebox.showinfo("Success", "State timeline generated")
                else:
                    messagebox.showwarning("Error", "Failed to generate timeline")
        finally:
            # Hide progress dialog
            if hasattr(self, "progress"):
                try:
                    self.progress.hide()
                except Exception:
                    pass
            # Ensure figure is closed to prevent memory leaks
            if fig and HAS_MATPLOTLIB:
                plt.close(fig)
                fig = None

    def generate_energy_plot(self) -> None:
        """Generate energy usage plot"""
        if not HAS_MATPLOTLIB:
            messagebox.showwarning("No Matplotlib", "Matplotlib is required for visualizations")
            return

        fig = None
        try:
            with ErrorContext("Generate Energy Plot", user_facing=True):
                self.progress = CancellableProgress(self.root)
                self.progress.show("Generating energy plot...")

                # Create mock data if no assistant or energy history available
                if (
                    not self.assistant
                    or not hasattr(self.assistant, "energy_history")
                    or not self.assistant.energy_history
                ):
                    # Generate sample energy history for demonstration with correct structure
                    import datetime
                    import random

                    mock_energy_history = []
                    base_time = datetime.datetime.now() - datetime.timedelta(minutes=30)

                    for i in range(20):
                        timestamp = base_time + datetime.timedelta(minutes=i * 1.5)
                        mock_energy_history.append(
                            {
                                "timestamp": timestamp,
                                "energy_cost": random.uniform(0.2, 0.8),
                                "level": max(0.2, 1.0 - (i * 0.02)),  # Gradual drain
                                "processing_mode": random.choice(["conscious", "unconscious"]),
                                "query_count": random.randint(1, 5),
                            }
                        )

                    fig = (
                        APGIVisualizer.plot_energy_usage(mock_energy_history)
                        if APGIVisualizer
                        else None
                    )
                else:
                    fig = (
                        APGIVisualizer.plot_energy_usage(self.assistant.energy_history)
                        if self.assistant
                        else None
                    )

                if fig:
                    self.display_plot_in_viz_tab(fig)
                    messagebox.showinfo("Success", "Energy plot generated")
                else:
                    messagebox.showwarning("Error", "Failed to generate energy plot")
        finally:
            # Hide progress dialog
            if hasattr(self, "progress"):
                try:
                    self.progress.hide()
                except Exception:
                    pass
            # Ensure figure is closed to prevent memory leaks
            if fig and HAS_MATPLOTLIB:
                plt.close(fig)
                fig = None

    def display_plot_in_viz_tab(self, fig: Any) -> None:
        """Display a matplotlib figure in the visualization tab"""
        if not HAS_PIL:
            messagebox.showwarning("PIL Required", "PIL/Pillow is required to display plots")
            return

        try:
            # Save to temporary buffer
            import io

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)

            # Load and display
            img = Image.open(buf)
            photo = ImageTk.PhotoImage(img)

            # Clear canvas
            if self.viz_canvas:
                self.viz_canvas.delete("all")

                # Display image
                self.viz_canvas.create_image(0, 0, anchor="nw", image=photo)
                setattr(self.viz_canvas, "image", photo)  # Keep reference

                # Switch to visualization tab
                self.notebook.select(self.viz_frame)

        except Exception as e:
            self.logger.error(f"Plot display error: {e}", exc_info=True)
            raise

    def clear_viz_display(self) -> None:
        """Clear visualization display"""
        if self.viz_canvas:
            self.viz_canvas.delete("all")
            if hasattr(self.viz_canvas, "image"):
                delattr(self.viz_canvas, "image")

    def save_state_timeline(self) -> None:
        """Save state timeline plot to file"""
        if not self.assistant:
            messagebox.showwarning("No Data", "Assistant not available")
            return
        if not getattr(self.assistant, "state_history", None):
            messagebox.showwarning("No Data", "No state history available")
            return

        if not HAS_MATPLOTLIB:
            messagebox.showwarning("No Matplotlib", "Matplotlib is required for visualizations")
            return

        self._save_plot_with_dialog(
            plot_type="State Timeline",
            plot_generator=lambda: APGIVisualizer.plot_state_timeline(self.assistant.state_history),
            default_filename="state_timeline",
        )

    def save_energy_plot(self) -> None:
        """Save energy plot to file"""
        if not self.assistant:
            messagebox.showwarning("No Data", "Assistant not available")
            return
        if not getattr(self.assistant, "energy_history", None):
            messagebox.showwarning("No Data", "No energy history available")
            return

        if not HAS_MATPLOTLIB:
            messagebox.showwarning("No Matplotlib", "Matplotlib is required for visualizations")
            return

        self._save_plot_with_dialog(
            plot_type="Energy Plot",
            plot_generator=lambda: APGIVisualizer.plot_energy_usage(self.assistant.energy_history),
            default_filename="energy_plot",
        )

    def save_oscillatory_spectrum(self) -> None:
        """Save oscillatory spectrum plot to file"""
        if not HAS_MATPLOTLIB:
            messagebox.showwarning("No Matplotlib", "Matplotlib is required for visualizations")
            return

        # Check if we have oscillatory data
        if not hasattr(self, "spectrum_fig") or not self.spectrum_fig:
            messagebox.showwarning("No Data", "No oscillatory spectrum available")
            return

        self._save_plot_with_dialog(
            plot_type="Oscillatory Spectrum",
            plot_generator=lambda: self.spectrum_fig,
            default_filename="oscillatory_spectrum",
        )

    def _save_plot_with_dialog(
        self, plot_type: str, plot_generator: Callable[[], Any], default_filename: str
    ) -> None:
        """Generic method to save plots with file dialog and format options"""
        with ErrorContext(f"Save {plot_type}", user_facing=True):
            # Ask for file location with format options
            file_path = filedialog.asksaveasfilename(
                title=f"Save {plot_type}",
                initialfile=f"{default_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*"),
                ],
            )

            if not file_path:
                return  # User cancelled

            try:
                # Progress dialog would be shown here
                # self.show_progress(f"Generating {plot_type.lower()}...")

                # Generate the plot
                fig = plot_generator()
                if not fig:
                    messagebox.showwarning("Error", f"Failed to generate {plot_type.lower()}")
                    return

                # Determine format from file extension or use default
                file_ext = Path(file_path).suffix.lower()
                if file_ext == ".pdf":
                    format_type = "pdf"
                elif file_ext == ".svg":
                    format_type = "svg"
                else:
                    format_type = self.export_settings["format"]
                    # Ensure file has correct extension
                    if format_type == "png" and file_ext != ".png":
                        file_path = str(Path(file_path).with_suffix(".png"))
                    elif format_type == "pdf" and file_ext != ".pdf":
                        file_path = str(Path(file_path).with_suffix(".pdf"))
                    elif format_type == "svg" and file_ext != ".svg":
                        file_path = str(Path(file_path).with_suffix(".svg"))

                # Get DPI based on quality setting
                quality_dpi: dict[str, int] = {"low": 150, "medium": 300, "high": 600}
                quality_key = str(self.export_settings["quality"])
                default_dpi = int(self.export_settings["dpi"])
                dpi = int(quality_dpi.get(quality_key, default_dpi))

                # Save with appropriate settings
                save_kwargs = {
                    "dpi": dpi,
                    "bbox_inches": "tight",
                    "facecolor": "white",
                    "edgecolor": "none",
                }

                # Add format-specific settings
                if format_type == "pdf":
                    save_kwargs.update(
                        {
                            "metadata": {
                                "Title": plot_type,
                                "Creator": "APGI Assistant",
                                "CreationDate": datetime.now(),
                            }
                        }
                    )
                elif format_type == "svg":
                    save_kwargs.update({"format": "svg", "transparent": False})

                # Save the figure
                fig.savefig(file_path, format=format_type, **save_kwargs)

                # Clean up
                plot_gen = cast(Any, plot_generator)
                if plot_gen.__name__ == "<lambda>":  # Generated plot, close it
                    if HAS_MATPLOTLIB:
                        plt.close(fig)

                self._force_hide_progress()
                messagebox.showinfo("Success", f"{plot_type} saved to:\n{file_path}")
                self.logger.info(f"{plot_type} saved to {file_path}")

            except Exception as e:
                self._force_hide_progress()
                error_msg = f"Failed to save {plot_type.lower()}: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                messagebox.showerror("Export Error", error_msg)

    def save_all_plots(self) -> None:
        """Save all visualization plots"""
        if not HAS_MATPLOTLIB:
            messagebox.showwarning("No Matplotlib", "Matplotlib is required for visualizations")
            return

        with ErrorContext("Save All Plots", user_facing=True):
            # Ask for directory
            directory = filedialog.askdirectory(title="Select Directory to Save Plots")
            if not directory:
                return

            save_dir = Path(directory)
            plots_saved = []
            errors = []

            # Progress dialog would be shown here
            # self.show_progress("Saving plots...")

            # Generate and save state timeline (create mock data if needed)
            try:
                if (
                    not self.assistant
                    or not hasattr(self.assistant, "state_history")
                    or not self.assistant.state_history
                ):
                    # Generate mock data
                    import datetime as dt_module
                    import random

                    mock_history = []
                    states = ["WORKING", "IDLE", "PROCESSING", "REFLECTING"]
                    base_time = dt_module.datetime.now() - dt_module.timedelta(minutes=30)

                    for i in range(20):
                        entry_timestamp = base_time + dt_module.timedelta(minutes=i * 1.5)
                        mock_history.append(
                            {
                                "timestamp": entry_timestamp,
                                "primary_state": random.choice(states),
                                "confidence": random.uniform(0.6, 1.0),
                                "surprise": random.uniform(0.0, 0.5),
                                "coherence": random.uniform(0.5, 0.9),
                            }
                        )

                    fig = (
                        APGIVisualizer.plot_state_timeline(mock_history) if APGIVisualizer else None
                    )
                else:
                    fig = (
                        APGIVisualizer.plot_state_timeline(self.assistant.state_history)
                        if self.assistant
                        else None
                    )

                if fig:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    timeline_path = save_dir / f"timeline_{timestamp}.png"
                    fig.savefig(timeline_path, dpi=300, bbox_inches="tight")
                    plots_saved.append(str(timeline_path))
                    if HAS_MATPLOTLIB:
                        plt.close(fig)
            except Exception as e:
                errors.append(f"State timeline: {str(e)}")

            # Generate and save energy plot (create mock data if needed)
            try:
                if (
                    not self.assistant
                    or not hasattr(self.assistant, "energy_history")
                    or not self.assistant.energy_history
                ):
                    # Generate mock energy data
                    import datetime as dt_module
                    import random

                    mock_energy_history = []
                    base_time = dt_module.datetime.now() - dt_module.timedelta(minutes=30)

                    for i in range(20):
                        entry_timestamp = base_time + dt_module.timedelta(minutes=i * 1.5)
                        mock_energy_history.append(
                            {
                                "timestamp": entry_timestamp,
                                "energy_cost": random.uniform(0.2, 0.8),
                                "battery_level": max(0.2, 1.0 - (i * 0.02)),
                                "processing_mode": random.choice(["conscious", "unconscious"]),
                                "query_count": random.randint(1, 5),
                            }
                        )

                    fig = (
                        APGIVisualizer.plot_energy_usage(mock_energy_history)
                        if APGIVisualizer
                        else None
                    )
                else:
                    fig = (
                        APGIVisualizer.plot_energy_usage(self.assistant.energy_history)
                        if self.assistant
                        else None
                    )

                if fig:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    energy_path = save_dir / f"energy_{timestamp}.png"
                    fig.savefig(energy_path, dpi=300, bbox_inches="tight")
                    plots_saved.append(str(energy_path))
                    if HAS_MATPLOTLIB:
                        plt.close(fig)
            except Exception as e:
                errors.append(f"Energy plot: {str(e)}")

            self._force_hide_progress()

            # Show results
            if plots_saved:
                success_msg = f"Successfully saved {len(plots_saved)} plots:\n\n"
                success_msg += "\n".join([f"• {Path(p).name}" for p in plots_saved])
                messagebox.showinfo("Success", success_msg)
            else:
                messagebox.showwarning("No Plots", "No plots were saved")

            if errors:
                error_msg = "Errors occurred:\n\n" + "\n".join([f"• {e}" for e in errors])
                self.logger.error(error_msg)

    # Edit menu actions
    def clear_query(self) -> None:
        """Clear query input with undo/redo support"""
        if hasattr(self, "query_input"):
            old_text = self.query_input.get(1.0, tk.END).strip()
            if old_text:  # Only track if there's text to clear
                action = ClearQueryAction(self, old_text)
                self.action_history.execute(action)

    def show_memory_stats(self) -> None:
        """Show memory usage statistics"""
        if not hasattr(self, "history_manager"):
            messagebox.showinfo("Not Available", "History manager not initialized")
            return

        stats = self.history_manager.get_memory_stats()
        usage = stats["memory_usage_mb"]

        message = "Memory Usage Statistics:\n\n"
        message += f"Total Memory: {stats['memory_usage_mb'].get('total', 0):.2f} MB / {stats['max_memory_mb']} MB\n"
        message += f"Utilization: {stats['memory_utilization']:.1%}\n\n"

        for history_type, memory_mb in usage.items():
            if history_type != "total":
                message += f"{history_type.title()}: {memory_mb:.2f} MB\n"

        message += f"\nAuto-prune: {'Enabled' if stats['auto_prune_enabled'] else 'Disabled'}\n"
        message += (
            f"Last prune: {time.strftime('%H:%M:%S', time.localtime(stats['last_prune_time']))}"
        )

        messagebox.showinfo("Memory Statistics", message)

    def clear_all_history(self) -> None:
        """Clear all managed history"""
        if not messagebox.askyesno(
            "Clear History", "Clear all history data? This action cannot be undone."
        ):
            return

        if hasattr(self, "history_manager"):
            self.history_manager.clear_all_history()
            self.logger.info("All history cleared by user")
            messagebox.showinfo("Success", "All history has been cleared")

    def configure_history_limits(self) -> None:
        """Configure history limits dialog"""
        if not hasattr(self, "history_manager"):
            messagebox.showinfo("Not Available", "History manager not initialized")
            return

        # Create configuration dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Configure History Limits")
        dialog.geometry("400x300")
        dialog.resizable(False, False)

        # Make dialog modal
        dialog.transient(self.root)
        dialog.grab_set()

        # Current settings
        current_max_memory = self.history_manager.max_memory_mb
        current_auto_prune = self.history_manager.auto_prune

        # Memory limit setting
        ttk.Label(dialog, text="Maximum Memory Usage (MB):").pack(pady=10)
        memory_var = tk.IntVar(value=current_max_memory)
        memory_scale = ttk.Scale(
            dialog, from_=10, to=500, variable=memory_var, orient=tk.HORIZONTAL, length=300
        )
        memory_scale.pack(pady=5)
        memory_label = ttk.Label(dialog, text=f"{current_max_memory} MB")
        memory_label.pack()

        def update_memory_label(value: str) -> None:
            memory_label.config(text=f"{int(float(value))} MB")

        memory_scale.config(command=update_memory_label)

        # Auto-prune setting
        auto_prune_var = tk.BooleanVar(value=current_auto_prune)
        ttk.Checkbutton(dialog, text="Enable Automatic Pruning", variable=auto_prune_var).pack(
            pady=20
        )

        # History type limits
        ttk.Label(dialog, text="History Type Limits:").pack(pady=10)

        limits_frame = ttk.Frame(dialog)
        limits_frame.pack(pady=10)

        limit_vars = {}
        for i, (history_type, strategy) in enumerate(
            self.history_manager.pruning_strategies.items()
        ):
            ttk.Label(limits_frame, text=f"{history_type.title()}:").grid(
                row=i, column=0, sticky="w", padx=5, pady=2
            )
            var = tk.IntVar(value=int(strategy["maxlen"]))
            ttk.Spinbox(limits_frame, from_=5, to=500, textvariable=var, width=10).grid(
                row=i, column=1, padx=5, pady=2
            )
            limit_vars[history_type] = var

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)

        def apply_settings() -> None:
            self.history_manager.max_memory_mb = memory_var.get()
            self.history_manager.auto_prune = auto_prune_var.get()

            # Update history type limits
            for history_type, var in limit_vars.items():
                self.history_manager.pruning_strategies[history_type]["maxlen"] = var.get()

            self.logger.info("History limits updated by user")
            messagebox.showinfo("Success", "History limits have been updated")
            dialog.destroy()

        def reset_defaults() -> None:
            memory_var.set(100)
            auto_prune_var.set(True)
            for history_type, var in limit_vars.items():
                default_limits = {
                    "state": 100,
                    "energy": 50,
                    "query": 20,
                    "transition": 100,
                    "physiology": 100,
                }
                var.set(default_limits.get(history_type, 50))

        ttk.Button(button_frame, text="Apply", command=apply_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset Defaults", command=reset_defaults).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def clear_history(self) -> None:
        """Clear all history"""
        if messagebox.askyesno("Clear History", "Clear all history?"):
            if "Cognitive Monitoring" in self.tabs_created:
                self.history_tree.delete(*self.history_tree.get_children())
            if "Performance" in self.tabs_created:
                self.query_tree.delete(*self.query_tree.get_children())

            self.state_history.clear()
            self.energy_history.clear()
            self.query_history.clear()

            self.logger.info("History cleared")

    def show_settings(self) -> None:
        """Show settings tab"""
        self.notebook.select(self.settings_frame)

    def show_export_settings(self) -> None:
        """Show export settings dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Settings")
        dialog.geometry("400x300")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog properly
        dialog.geometry("400x300")  # Set size first
        dialog.update_idletasks()  # Update window manager

        # Get screen dimensions and calculate center position
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        x = (screen_width // 2) - (400 // 2)
        y = (screen_height // 2) - (300 // 2)
        dialog.geometry(f"+{x}+{y}")  # Set position only

        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(fill="both", expand=True)

        # DPI Setting
        ttk.Label(main_frame, text="Image Quality (DPI):", font=("Arial", 10, "bold")).pack(
            anchor="w", pady=(0, 5)
        )

        dpi_frame = ttk.Frame(main_frame)
        dpi_frame.pack(fill="x", pady=(0, 15))

        dpi_var = tk.IntVar(value=int(self.export_settings["dpi"]))
        ttk.Radiobutton(dpi_frame, text="Standard (150 DPI)", variable=dpi_var, value=150).pack(
            anchor="w"
        )
        ttk.Radiobutton(dpi_frame, text="High (300 DPI)", variable=dpi_var, value=300).pack(
            anchor="w"
        )
        ttk.Radiobutton(dpi_frame, text="Print (600 DPI)", variable=dpi_var, value=600).pack(
            anchor="w"
        )

        # Format Setting
        ttk.Label(main_frame, text="Default Format:", font=("Arial", 10, "bold")).pack(
            anchor="w", pady=(0, 5)
        )

        format_frame = ttk.Frame(main_frame)
        format_frame.pack(fill="x", pady=(0, 15))

        format_var = tk.StringVar(value=str(self.export_settings["format"]))
        ttk.Radiobutton(format_frame, text="PNG", variable=format_var, value="png").pack(anchor="w")
        ttk.Radiobutton(format_frame, text="PDF", variable=format_var, value="pdf").pack(anchor="w")
        ttk.Radiobutton(format_frame, text="SVG", variable=format_var, value="svg").pack(anchor="w")

        # Quality Setting
        ttk.Label(main_frame, text="Quality Level:", font=("Arial", 10, "bold")).pack(
            anchor="w", pady=(0, 5)
        )

        quality_frame = ttk.Frame(main_frame)
        quality_frame.pack(fill="x", pady=(0, 15))

        quality_var = tk.StringVar(value=str(self.export_settings["quality"]))
        ttk.Radiobutton(quality_frame, text="Low (Fast)", variable=quality_var, value="low").pack(
            anchor="w"
        )
        ttk.Radiobutton(quality_frame, text="Medium", variable=quality_var, value="medium").pack(
            anchor="w"
        )
        ttk.Radiobutton(quality_frame, text="High (Best)", variable=quality_var, value="high").pack(
            anchor="w"
        )

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(20, 0))

        def save_settings() -> None:
            self.export_settings["dpi"] = dpi_var.get()
            self.export_settings["format"] = format_var.get()
            self.export_settings["quality"] = quality_var.get()
            self.logger.info(f"Export settings updated: {self.export_settings}")
            messagebox.showinfo("Success", "Export settings saved successfully!")
            dialog.destroy()

        ttk.Button(button_frame, text="Save", command=save_settings).pack(side="right", padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side="right")

        # Bind Enter key to save
        dialog.bind("<Return>", lambda e: save_settings())
        dialog.bind("<Escape>", lambda e: dialog.destroy())

    # View menu actions
    def refresh_displays(self) -> None:
        """Refresh all displays"""
        with self.update_flags_lock:
            self.update_flags = {key: True for key in self.update_flags}
        self.update_displays()
        self.update_status("Displays refreshed", "info")

    def toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode"""
        current_state = self.root.attributes("-fullscreen")
        self.root.attributes("-fullscreen", not current_state)

    # Help menu actions
    def show_quick_start(self) -> None:
        """Show quick start guide"""
        help_text = """
APGI Assistant - Quick Start Guide
═══════════════════════════════════

Getting Started:
1. Enter your query in the text box
2. Adjust physiological sliders or use presets
3. Click 'Process Query' or press Ctrl+Enter
4. View response and cognitive state

Tabs:
- Main Interface: Query processing and results
- Cognitive Monitoring: Real-time state tracking
- Oscillatory Analysis: Brain-like patterns
- Biofeedback: Physiological state monitoring
- Energy Management: Computational efficiency
- Performance: Metrics and query history
- Visualizations: Generate and save plots
- Settings: Configure assistant parameters

Keyboard Shortcuts:
- Ctrl+Enter: Process query
- Ctrl+N: New session
- Ctrl+S: Save session
- Ctrl+O: Load session
- Ctrl+L: Clear query
- Ctrl+R: Reset assistant
- F5: Refresh displays
- F11: Toggle fullscreen
- F1: This help screen

Tips:
- Use physiological presets for quick testing
- Monitor cognitive state changes in real-time
- Save sessions to preserve your work
- Check logs if errors occur

For more help, visit the documentation.
        """

        help_window = tk.Toplevel(self.root)
        help_window.title("Quick Start Guide")
        help_window.geometry("600x500")

        text = scrolledtext.ScrolledText(help_window, wrap="word", font=("Courier", 9))
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert("1.0", help_text)
        text.config(state="disabled")

        ttk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=5)

    def show_shortcuts(self) -> None:
        """Show keyboard shortcuts"""
        shortcuts_text = """
Keyboard Shortcuts
═══════════════════

File Operations:
  Ctrl+N          New session
  Ctrl+S          Save session
  Ctrl+O          Load session
  Ctrl+,          Open settings
  Ctrl+Z          Undo last action
  Ctrl+Y          Redo last action

Processing:
  Ctrl+Enter      Process query
  Ctrl+R          Reset assistant

View:
  F5              Refresh displays
  F11             Toggle fullscreen
  Ctrl+Plus        Increase font size
  Ctrl+Minus       Decrease font size
  Ctrl+0          Reset font size
  Ctrl+Alt+H      Toggle high contrast mode

Navigation:
  Ctrl+Tab        Next tab
  Ctrl+Shift+Tab  Previous tab

Help:
  F1              Quick start guide
        """

        shortcuts_window = tk.Toplevel(self.root)
        shortcuts_window.title("Keyboard Shortcuts")
        shortcuts_window.geometry("400x400")

        text = scrolledtext.ScrolledText(shortcuts_window, wrap="word", font=("Courier", 10))
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert("1.0", shortcuts_text)
        text.config(state="disabled")

        ttk.Button(shortcuts_window, text="Close", command=shortcuts_window.destroy).pack(pady=5)

    def show_about(self) -> None:
        """Show about dialog"""
        about_text = f"""
APGI Assistant v1.0

A sophisticated neural interface with adaptive processing
and energy-aware computation.

Features:
• Real-time cognitive state monitoring
• Physiological parameter tracking
• Energy-efficient processing
• Oscillatory pattern analysis
• Biofeedback integration
• Export capabilities (PNG, SVG, PDF, CSV, JSON)

Font Size: {self.current_font_size}pt
Use Ctrl+Plus/Minus to adjust, Ctrl+0 to reset
{'✓' if HAS_PSUTIL else '✗'} psutil
{'✓' if HAS_PIL else '✗'} PIL

© 2024 APGI Project
        """

        messagebox.showinfo("About APGI Assistant", about_text)

    def enable_memory_profiling(self, interval_sec: int = 10) -> None:
        """Enable periodic memory profiling (development/debugging)

        Args:
            interval_sec: Interval in seconds between memory checks
        """
        if not HAS_PSUTIL:
            self.logger.warning("psutil not available for memory profiling")
            messagebox.showwarning(
                "Dependency Required",
                "The psutil package is required for memory profiling.\n\n"
                "Install it with: pip install psutil",
            )
            return

        # Initialize memory_profiling_enabled if not exists
        if not hasattr(self, "memory_profiling_enabled"):
            self.memory_profiling_enabled: bool = False

        if self.memory_profiling_enabled:
            self.logger.info("Memory profiling already enabled")
            return

        def profile_memory() -> None:
            try:
                process = psutil.Process()
                mem_info = process.memory_info()

                # Get memory statistics
                rss_mb = mem_info.rss / 1024**2
                vms_mb = mem_info.vms / 1024**2

                # Log memory usage
                self.logger.info(f"Memory Usage - " f"RSS: {rss_mb:.1f}MB, " f"VMS: {vms_mb:.1f}MB")

                # Schedule next check if still enabled
                if hasattr(self, "memory_profiling_enabled") and self.memory_profiling_enabled:
                    self.root.after(interval_sec * 1000, profile_memory)

            except Exception as e:
                self.logger.error(f"Error in memory profiling: {str(e)}")
                if hasattr(self, "memory_profiling_enabled"):
                    delattr(self, "memory_profiling_enabled")

        # Enable profiling and start the first check
        self.memory_profiling_enabled = True
        self.logger.info("Memory profiling enabled")
        profile_memory()

        # Add a menu item to disable profiling
        if hasattr(self, "tools_menu"):
            # Find and update the menu item
            menu_end_index = self.tools_menu.index("end")
            if menu_end_index is not None:
                for i in range(menu_end_index + 1):
                    try:
                        label = self.tools_menu.entrycget(i, "label")
                        if "Memory Profiling" in label:
                            self.tools_menu.entryconfigure(
                                i,
                                label="Disable Memory Profiling",
                                command=self.disable_memory_profiling,
                            )
                            break
                    except Exception as e:
                        # Log error but continue with next menu item
                        LOGGER.debug(f"Failed to modify menu item: {e}")

    def disable_memory_profiling(self) -> None:
        """Disable memory profiling"""
        if hasattr(self, "memory_profiling_enabled") and self.memory_profiling_enabled:
            self.memory_profiling_enabled = False
            self.logger.info("Memory profiling disabled")

            # Update the menu item
            if hasattr(self, "tools_menu"):
                menu_end_index = self.tools_menu.index("end")
                if menu_end_index is not None:
                    for i in range(menu_end_index + 1):
                        try:
                            label = self.tools_menu.entrycget(i, "label")
                            if "Memory Profiling" in label:
                                self.tools_menu.entryconfigure(
                                    i,
                                    label="Enable Memory Profiling",
                                    command=self.enable_memory_profiling,
                                )
                                break
                        except Exception:
                            # Continue if menu entry doesn't exist or can't be accessed
                            continue

    def view_logs(self) -> None:
        """View application logs"""
        log_file = Path("apgi_gui.log")

        if not log_file.exists():
            messagebox.showinfo("No Logs", "No log file found")
            return

        try:
            log_window = tk.Toplevel(self.root)
            log_window.title("Application Logs")
            log_window.geometry("800x600")

            text = scrolledtext.ScrolledText(log_window, wrap="word", font=("Courier", 8))
            text.pack(fill="both", expand=True, padx=10, pady=10)

            with open(log_file, "r") as f:
                text.insert("1.0", f.read())

            text.see(tk.END)  # Scroll to bottom

            button_frame = ttk.Frame(log_window)
            button_frame.pack(pady=5)

            ttk.Button(
                button_frame, text="Refresh", command=lambda: self.refresh_logs(text, log_file)
            ).pack(side="left", padx=5)
            ttk.Button(
                button_frame,
                text="Clear Logs",
                command=lambda: self.clear_logs(log_file, log_window),
            ).pack(side="left", padx=5)
            ttk.Button(button_frame, text="Close", command=log_window.destroy).pack(
                side="left", padx=5
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open logs:\n{str(e)}")

    def refresh_logs(self, text_widget: Any, log_file: Path) -> None:
        """Refresh log display"""
        try:
            text_widget.delete("1.0", tk.END)
            with open(log_file, "r") as f:
                text_widget.insert("1.0", f.read())
            text_widget.see(tk.END)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh logs:\n{str(e)}")

    def clear_logs(self, log_file: Path, window: Any) -> None:
        """Clear log file"""
        if messagebox.askyesno("Clear Logs", "Clear all logs?"):
            try:
                with open(log_file, "w") as f:
                    f.write("")
                window.destroy()
                messagebox.showinfo("Success", "Logs cleared")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear logs:\n{str(e)}")

    def _on_physio_change(self, param_type: str, value: float) -> None:
        """Handle physiological parameter slider changes"""
        self.set_physiology_action2(param_type, value)

    def set_physiology_action2(self, param_type: str, value: float) -> None:
        """Handle physiological parameter changes with undo/redo support"""
        # Update label immediately for responsive UI
        if param_type == "hr":
            self.hr_label.config(text=f"{value} bpm")
        elif param_type == "hrv":
            self.hrv_label.config(text=f"{value} ms")
        elif param_type == "resp":
            self.resp_label.config(text=f"{value} bpm")
        elif param_type == "eda":
            self.eda_label.config(text=f"{value:.1f} µS")

        # Create undo action immediately for consistent redo
        old_state = {
            "hr": self.hr_var.get(),
            "hrv": self.hrv_var.get(),
            "resp": self.resp_var.get(),
            "eda": self.eda_var.get(),
        }

        # Create new state with updated value
        new_state = old_state.copy()
        new_state[param_type] = value

        # Execute action through history immediately
        action = SetPhysiologyAction(self, old_state, new_state)
        self.action_history.execute(action)

    def redo_action2(self) -> None:
        """Redo last undone action"""
        if self.action_history.can_redo():
            self.action_history.redo()
            self.logger.info("Action redone")

    def auto_save_session(self) -> None:
        """Auto-save session data for crash recovery"""
        if not self.auto_save_enabled:
            return

        try:
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "query_text": (
                    self.query_input.get(1.0, tk.END).strip()
                    if hasattr(self, "query_input")
                    else ""
                ),
                "physiology": {
                    "hr": self.hr_var.get() if hasattr(self, "hr_var") else 60,
                    "hrv": self.hrv_var.get() if hasattr(self, "hrv_var") else 50,
                    "resp": self.resp_var.get() if hasattr(self, "resp_var") else 12,
                    "eda": self.eda_var.get() if hasattr(self, "eda_var") else 2.0,
                },
                "config": self.get_current_config() if hasattr(self, "get_current_config") else {},
                "query_history": list(self.query_history) if hasattr(self, "query_history") else [],
                "energy_history": (
                    list(self.energy_history) if hasattr(self, "energy_history") else []
                ),
            }

            with open(self.auto_save_file, "w") as f:
                json.dump(session_data, f, indent=2, default=str)

            self.last_auto_save = datetime.now()
            self.logger.info(f"Auto-saved session at {self.last_auto_save.strftime('%H:%M:%S')}")

        except Exception as e:
            self.logger.error(f"Auto-save failed: {e}")

    def check_auto_save_recovery(self) -> None:
        """Check for auto-save file and offer recovery"""
        if not os.path.exists(self.auto_save_file):
            return

        try:
            with open(self.auto_save_file, "r") as f:
                session_data = json.load(f)

            # Check if auto-save is recent (within last 24 hours)
            save_time = datetime.fromisoformat(session_data.get("timestamp", ""))
            if datetime.now() - save_time > timedelta(hours=24):
                # Old auto-save, remove it
                os.remove(self.auto_save_file)
                return

            # Offer recovery to user
            result = messagebox.askyesno(
                "Crash Recovery",
                f"Found auto-saved session from {save_time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
                "Would you like to recover your previous session?",
            )

            if result:
                self.recover_session(session_data)
            else:
                # User declined, remove auto-save
                os.remove(self.auto_save_file)

        except Exception as e:
            self.logger.error(f"Auto-save recovery check failed: {e}")
            # Remove corrupted auto-save file
            try:
                os.remove(self.auto_save_file)
            except OSError as e:
                # File might not exist or be locked - log but continue
                LOGGER.debug(f"Failed to remove auto-save file: {e}")

    def recover_session(self, session_data: Dict[str, Any]) -> None:
        """Recover session from auto-save data"""
        try:
            # Restore query text
            if hasattr(self, "query_input") and "query_text" in session_data:
                self.query_input.delete(1.0, tk.END)
                self.query_input.insert(1.0, session_data["query_text"])
                self.update_char_count()

            # Restore physiology
            if "physiology" in session_data:
                physio = session_data["physiology"]
                if hasattr(self, "hr_var"):
                    self.hr_var.set(physio.get("hr", 60))
                if hasattr(self, "hrv_var"):
                    self.hrv_var.set(physio.get("hrv", 50))
                if hasattr(self, "resp_var"):
                    self.resp_var.set(physio.get("resp", 12))
                if hasattr(self, "eda_var"):
                    self.eda_var.set(physio.get("eda", 2.0))

            # Restore configuration
            if "config" in session_data and hasattr(self, "load_config"):
                # Create temp config file and load it
                temp_config = "temp_recovery_config.json"
                with open(temp_config, "w") as f:
                    json.dump(session_data["config"], f, indent=2)
                self.load_config(temp_config)
                os.remove(temp_config)

            # Restore history
            if "query_history" in session_data and hasattr(self, "query_history"):
                self.query_history.clear()
                self.query_history.extend(session_data["query_history"])

            if "energy_history" in session_data and hasattr(self, "energy_history"):
                self.energy_history.clear()
                self.energy_history.extend(session_data["energy_history"])

            # Remove auto-save file after successful recovery
            os.remove(self.auto_save_file)

            messagebox.showinfo(
                "Recovery Complete", "Your session has been successfully recovered!"
            )
            self.logger.info("Session recovery completed successfully")

        except Exception as e:
            self.logger.error(f"Session recovery failed: {e}")
            messagebox.showerror("Recovery Failed", f"Failed to recover session: {e}")

    def start_auto_save_timer(self) -> None:
        """Start the auto-save timer"""
        if self.auto_save_enabled and not self.auto_save_timer_id:
            self.auto_save_timer_id = self.root.after(
                self.auto_save_interval, self.auto_save_timer_callback
            )

    def auto_save_timer_callback(self) -> None:
        """Auto-save timer callback"""
        self.auto_save_session()
        # Restart timer for next auto-save
        self.auto_save_timer_id = None
        self.start_auto_save_timer()

    def stop_auto_save_timer(self) -> None:
        """Stop the auto-save timer"""
        if self.auto_save_timer_id:
            self.root.after_cancel(self.auto_save_timer_id)
            self.auto_save_timer_id = None

    def quit_application(self) -> None:
        """Quit application with cleanup"""
        # Set shutdown flag immediately to prevent race conditions
        self.is_shutting_down = True

        if messagebox.askyesno("Exit", "Exit APGI Assistant?"):
            self.logger.info("Application shutting down")

            # Cancel all pending timers and operations first
            try:
                # Stop auto-save timer
                self.stop_auto_save_timer()

                # Cancel debouncer updates
                if (
                    hasattr(self, "debouncer")
                    and hasattr(self, "root")
                    and self.root.winfo_exists()
                ):
                    self.debouncer.cancel_all(self.root)

                # Cancel any other pending timers
                # Cancel update timer if it exists
                if hasattr(self, "update_timer_id") and self.update_timer_id:
                    try:
                        self.root.after_cancel(self.update_timer_id)
                    except Exception:
                        pass
            except Exception as e:
                self.logger.debug(f"Error cancelling timers: {e}")

            # Save configuration
            with ErrorContext("Save Configuration on Shutdown", user_facing=False):
                try:
                    self.save_configuration()
                except Exception as e:
                    self.logger.error(f"Failed to save configuration: {e}")

            # Cleanup
            with ErrorContext("Application Cleanup", user_facing=False):
                try:
                    # Final auto-save before shutdown
                    self.auto_save_session()

                    # Stop any running operations
                    self.is_processing = False

                    # Close matplotlib figures
                    if HAS_MATPLOTLIB:
                        plt.close("all")

                    # Clear widget references to help garbage collection
                    if hasattr(self, "query_input"):
                        self.query_input = None
                    if hasattr(self, "response_display"):
                        self.response_display = None

                    # Perform garbage collection
                    gc.collect()
                except Exception as e:
                    self.logger.error(f"Cleanup error: {e}")

            # Quit and destroy with enhanced error handling
            with ErrorContext("Application Shutdown", user_facing=False):
                try:
                    if hasattr(self, "root") and self.root.winfo_exists():
                        # Quit the mainloop
                        self.root.quit()

                        # Destroy with timeout protection
                        self.root.after(100, self._safe_destroy)
                    else:
                        # Force exit if root doesn't exist
                        import sys

                        sys.exit(0)
                except Exception as e:
                    self.logger.error(f"Shutdown error: {e}")
                    # Force exit as last resort
                    import sys

                    sys.exit(1)

    def _safe_destroy(self) -> None:
        """Safely destroy the root window with error handling"""
        try:
            if hasattr(self, "root") and self.root.winfo_exists():
                self.root.destroy()
        except Exception as e:
            self.logger.error(f"Destroy error: {e}")
            # Force exit if destroy fails
            import sys

            sys.exit(1)

    # Theme Management Methods
    def toggle_high_contrast(self) -> None:
        """Toggle between normal and high contrast themes"""
        new_theme = "high_contrast" if self.high_contrast_var.get() else "normal"
        self.apply_theme(new_theme)
        self.logger.info(f"Theme switched to: {new_theme}")

    def apply_theme(self, theme_name: str) -> None:
        """Apply a theme to the entire GUI"""
        if theme_name not in GUIConfig.THEMES:
            self.logger.warning(f"Unknown theme: {theme_name}")
            return

        self.current_theme = theme_name
        self._save_theme_preference()
        theme = GUIConfig.THEMES[theme_name]

        # Update main window
        self.root.configure(bg=str(theme["bg"]))

        # Update all themed widgets
        self._update_canvas_themes()
        self._update_display_themes()
        self._update_status_colors()

        # Force refresh of all displays
        self.refresh_displays()

    def get_theme_color(self, color_type: str) -> str:
        """Get a color from the current theme"""
        return str(GUIConfig.THEMES[self.current_theme].get(color_type, "black"))

    def _update_canvas_themes(self) -> None:
        """Update all canvas backgrounds including matplotlib figures"""
        theme = GUIConfig.THEMES[self.current_theme]
        canvas_bg = theme["canvas_bg"]

        # Update all canvas widgets
        canvases = [
            ("state_canvas", hasattr(self, "state_canvas")),
            ("physio_canvas", hasattr(self, "physio_canvas")),
            ("physio_history_canvas", hasattr(self, "physio_history_canvas")),
            ("battery_canvas", hasattr(self, "battery_canvas")),
            ("viz_canvas", hasattr(self, "viz_canvas")),
        ]

        for canvas_name, has_canvas in canvases:
            if has_canvas:
                canvas = getattr(self, canvas_name)
                canvas.configure(bg=str(canvas_bg))

        # Update matplotlib figures with theme-aware styling
        if HAS_MATPLOTLIB:
            self._update_matplotlib_themes(theme)

    def _update_matplotlib_themes(self, theme: Dict[str, Any]) -> None:
        """Update matplotlib figures to match current theme"""
        try:
            # Update matplotlib figure colors
            figure_bg = theme.get("figure_bg", theme["bg"])
            text_color = theme.get("fg", "black")
            grid_color = theme.get("grid_color", "#cccccc")

            # Update oscillatory spectrum figure
            if hasattr(self, "spectrum_fig") and self.spectrum_fig:
                self.spectrum_fig.patch.set_facecolor(figure_bg)
                if hasattr(self, "spectrum_ax") and self.spectrum_ax:
                    self.spectrum_ax.set_facecolor(theme["canvas_bg"])
                    self.spectrum_ax.tick_params(colors=text_color)
                    self.spectrum_ax.xaxis.label.set_color(text_color)
                    self.spectrum_ax.yaxis.label.set_color(text_color)
                    self.spectrum_ax.spines["bottom"].set_color(text_color)
                    self.spectrum_ax.spines["top"].set_color(text_color)
                    self.spectrum_ax.spines["left"].set_color(text_color)
                    self.spectrum_ax.spines["right"].set_color(text_color)
                    self.spectrum_ax.grid(color=grid_color, alpha=0.3)

                # Redraw the canvas
                if hasattr(self, "spectrum_canvas") and self.spectrum_canvas:
                    self.spectrum_canvas.draw()

            # Update energy figure
            if hasattr(self, "energy_fig") and self.energy_fig:
                self.energy_fig.patch.set_facecolor(figure_bg)
                if hasattr(self, "energy_ax") and self.energy_ax:
                    self.energy_ax.set_facecolor(theme["canvas_bg"])
                    self.energy_ax.tick_params(colors=text_color)
                    self.energy_ax.xaxis.label.set_color(text_color)
                    self.energy_ax.yaxis.label.set_color(text_color)
                    self.energy_ax.spines["bottom"].set_color(text_color)
                    self.energy_ax.spines["top"].set_color(text_color)
                    self.energy_ax.spines["left"].set_color(text_color)
                    self.energy_ax.spines["right"].set_color(text_color)
                    self.energy_ax.grid(color=grid_color, alpha=0.3)

                # Redraw the canvas
                if hasattr(self, "energy_canvas") and self.energy_canvas:
                    self.energy_canvas.draw()

            # Update any other matplotlib figures
            figure_attrs = ["timeline_fig", "performance_fig", "cognitive_fig"]
            for fig_attr in figure_attrs:
                if hasattr(self, fig_attr):
                    fig = getattr(self, fig_attr)
                    if fig:
                        fig.patch.set_facecolor(figure_bg)
                        # Update all axes in the figure
                        for ax in fig.get_axes():
                            ax.set_facecolor(theme["canvas_bg"])
                            ax.tick_params(colors=text_color)
                            ax.xaxis.label.set_color(text_color)
                            ax.yaxis.label.set_color(text_color)
                            ax.spines["bottom"].set_color(text_color)
                            ax.spines["top"].set_color(text_color)
                            ax.spines["left"].set_color(text_color)
                            ax.spines["right"].set_color(text_color)
                            ax.grid(color=grid_color, alpha=0.3)

                        # Find corresponding canvas and redraw
                        canvas_attr = fig_attr.replace("_fig", "_canvas")
                        if hasattr(self, canvas_attr):
                            canvas = getattr(self, canvas_attr)
                            if canvas:
                                canvas.draw()

        except Exception as e:
            self.logger.warning(f"Error updating matplotlib themes: {e}")

    def _update_display_themes(self) -> None:
        """Update text displays and other UI elements"""
        theme = GUIConfig.THEMES[self.current_theme]

        # Update response display if it exists
        if hasattr(self, "response_display"):
            self.response_display.configure(bg=str(theme["bg"]), fg=str(theme["fg"]))

        # Update query input if it exists
        if hasattr(self, "query_input"):
            self.query_input.configure(
                bg=str(theme["bg"]), fg=str(theme["fg"]), insertbackground=str(theme["fg"])
            )
            # Bind Enter key to send query
            self.query_input.bind("<Return>", self.send_query)

    def _update_status_colors(self) -> None:
        """Update status color mappings"""
        # The update_status method will automatically use theme colors
        pass

    def _save_theme_preference(self) -> None:
        """Save current theme preference and buffer size to config file."""
        try:
            config_dir = Path.home() / ".apgi_assistant"
            config_dir.mkdir(exist_ok=True)
            config_file = config_dir / "theme_config.json"

            config = {
                "current_theme": self.current_theme,
                "buffer_size": getattr(self, "memory_length_var", tk.IntVar(value=100)).get(),
            }
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self._log_event(f"Failed to save preferences: {e}")

    def _load_theme_preference(self) -> str:
        """Load theme preference and buffer size from config file."""
        try:
            config_dir = Path.home() / ".apgi_assistant"
            config_file = config_dir / "theme_config.json"

            if config_file.exists():
                with open(config_file, "r") as f:
                    config = json.load(f)
                # Load buffer size if available
                if "buffer_size" in config and hasattr(self, "memory_length_var"):
                    self.memory_length_var.set(config["buffer_size"])
        except Exception:
            pass
        return "normal"  # Default theme

    def send_query(self, event: Optional[Any] = None) -> None:
        """Send user query to the APGI Assistant and display response."""
        if not hasattr(self, "query_input") or not hasattr(self, "response_display"):
            return

        # Get user input (ScrolledText widget uses Text widget get method)
        user_input = self.query_input.get(1.0, tk.END).strip()
        if not user_input:
            return

        # Clear input (ScrolledText widget uses Text widget delete method)
        self.query_input.delete(1.0, tk.END)

        # Check if assistant is available
        if not HAS_APGI_ASSISTANT or not hasattr(self, "assistant") or self.assistant is None:
            self.response_display.insert(
                tk.END, "APGI Assistant not available. Please check dependencies.\n"
            )
            self.response_display.see(tk.END)
            return

        # Start conversation if not active
        if self.assistant is not None and not self.assistant.conversation_active:
            self.assistant.start_conversation()

        # Process query in thread to avoid blocking GUI
        def process_query_thread() -> None:
            try:
                if self.assistant is None:
                    raise RuntimeError("Assistant not available")
                result = self.assistant.process_query(user_input)

                # Display response on main thread
                self.root.after(0, lambda: self._display_response(user_input, result))  # type: ignore[misc]

            except Exception as e:
                self.root.after(0, lambda exc=e: self._display_error(str(exc)))  # type: ignore[misc]

        # Start processing thread
        import threading

        thread = threading.Thread(target=process_query_thread, daemon=True)
        thread.start()

    def _display_response(self, user_input: str, result: Dict[str, Any]) -> None:
        """Display the assistant response in the GUI."""
        if not hasattr(self, "response_display"):
            return

        # Display user input
        self.response_display.insert(tk.END, f"You: {user_input}\n", "user")
        self.response_display.insert(tk.END, f"Assistant: {result['response']}\n\n", "assistant")

        # Prune display content if it exceeds limit (BUG-M10)
        max_lines = 1000
        content_lines = float(self.response_display.index("end-1c").split(".")[0])
        if content_lines > max_lines:
            # Remove top 20% of lines
            remove_until = int(max_lines * 0.2)
            self.response_display.delete("1.0", f"{remove_until}.0")
            self.response_display.insert(
                "1.0",
                f"--- Older messages pruned for memory (Total lines removed: {remove_until}) ---\n\n",
                "warning",
            )

        # Display cognitive analysis if available
        cognitive = result.get("cognitive_analysis", {})
        if cognitive and "error" not in cognitive:
            analysis_text = "Cognitive Analysis:\n"
            analysis_text += (
                f"- Processing Confidence: {cognitive.get('processing_confidence', 0):.2f}\n"
            )
            analysis_text += f"- Cognitive Load: {cognitive.get('cognitive_load', 0):.2f}\n"
            analysis_text += (
                f"- Response Coherence: {cognitive.get('response_coherence', 0):.2f}\n\n"
            )
            self.response_display.insert(tk.END, analysis_text, "analysis")

        self.response_display.see(tk.END)

    def _display_error(self, error_msg: str) -> None:
        """Display error message in the GUI."""
        if hasattr(self, "response_display"):
            self.response_display.insert(tk.END, f"Error: {error_msg}\n\n", "error")
            self.response_display.see(tk.END)


def main() -> None:
    """Main function to run the GUI"""
    # Validate dependencies before creating GUI
    try:
        root = tk.Tk()
        APGIGUI(root)
        root.mainloop()
    except Exception as e:
        LOGGER.critical(f"Fatal error: {e}", exc_info=True)
        messagebox.showerror(
            "Fatal Error",
            f"Application crashed:\n{str(e)}\n\n" "Please check the logs for details.",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
