"""Loading indicator component for APGI Framework GUI.

Provides visual feedback for long-running operations with progress bars,
spinners, and status messages.
"""

import logging
import tkinter as tk
from datetime import datetime
from tkinter import ttk
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class LoadingIndicator:
    """A reusable loading indicator component for GUI applications.

    Features:
    - Progress bar with percentage
    - Animated spinner
    - Status messages
    - Cancellation support
    - Time estimation
    """

    def __init__(
        self,
        parent: tk.Widget,
        title: str = "Loading...",
        show_cancel: bool = True,
        show_time: bool = True,
        width: int = 400,
        height: int = 150,
    ):
        """Initialize loading indicator.

        Args:
            parent: Parent widget
            title: Window title
            show_cancel: Whether to show cancel button
            show_time: Whether to show elapsed time
            width: Window width
            height: Window height
        """
        self.parent = parent
        self.title = title
        self.show_cancel = show_cancel
        self.show_time = show_time

        # State
        self.is_active = False
        self.is_cancelled = False
        self.progress = 0.0
        self.status_message = ""
        self.start_time: Optional[datetime] = None
        self.cancel_callback: Optional[Callable] = None

        # Animation
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_index = 0
        self.animation_running = False

        # Create window
        self._create_window(width, height)

    def _create_window(self, width: int, height: int) -> None:
        """Create the loading window."""
        # Create modal window
        self.window = tk.Toplevel(self.parent)
        self.window.title(self.title)
        self.window.geometry(f"{width}x{height}")
        self.window.resizable(False, False)

        # Center the window
        self.window.transient(self.parent)  # type: ignore
        self.window.grab_set()

        # Calculate center position
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        x = parent_x + (parent_width // 2) - (width // 2)
        y = parent_y + (parent_height // 2) - (height // 2)

        self.window.geometry(f"{width}x{height}+{x}+{y}")

        # Create main frame
        main_frame = ttk.Frame(self.window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title label
        self.title_label = ttk.Label(
            main_frame, text=self.title, font=("TkDefaultFont", 12, "bold")
        )
        self.title_label.pack(pady=(0, 15))

        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        # Spinner label
        self.spinner_label = ttk.Label(status_frame, text="", font=("TkDefaultFont", 16))
        self.spinner_label.pack(side=tk.LEFT, padx=(0, 10))

        # Status message label
        self.status_label = ttk.Label(status_frame, text="", font=("TkDefaultFont", 10))
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame, variable=self.progress_var, maximum=100, length=width - 40
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Progress percentage label
        self.percentage_label = ttk.Label(main_frame, text="0%", font=("TkDefaultFont", 9))
        self.percentage_label.pack()

        # Time label
        if self.show_time:
            self.time_label = ttk.Label(main_frame, text="", font=("TkDefaultFont", 9))
            self.time_label.pack(pady=(5, 0))

        # Cancel button
        if self.show_cancel:
            self.cancel_button = ttk.Button(main_frame, text="Cancel", command=self._on_cancel)
            self.cancel_button.pack(pady=(10, 0))

        # Configure window close behavior
        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def start(
        self,
        status_message: str = "Initializing...",
        progress: float = 0.0,
        cancel_callback: Optional[Callable] = None,
    ) -> None:
        """Start the loading indicator.

        Args:
            status_message: Initial status message
            progress: Initial progress (0.0 to 1.0)
            cancel_callback: Function to call when cancelled
        """
        self.is_active = True
        self.is_cancelled = False
        self.progress = progress
        self.status_message = status_message
        self.start_time = datetime.now()
        self.cancel_callback = cancel_callback

        # Update UI
        self._update_status(status_message)
        self._update_progress(progress)

        # Start animation
        self._start_animation()

        logger.debug(f"Loading indicator started: {status_message}")

    def update_progress(self, progress: float, status_message: Optional[str] = None) -> None:
        """Update progress and optionally status message.

        Args:
            progress: Progress value (0.0 to 1.0)
            status_message: Optional new status message
        """
        if not self.is_active or self.is_cancelled:
            return

        self.progress = max(0.0, min(1.0, progress))

        if status_message is not None:
            self.status_message = status_message
            self._update_status(status_message)

        self._update_progress(self.progress)

    def update_status(self, status_message: str) -> None:
        """Update only the status message."""
        if not self.is_active or self.is_cancelled:
            return

        self.status_message = status_message
        self._update_status(status_message)

    def stop(self) -> None:
        """Stop the loading indicator."""
        if not self.is_active:
            return

        self.is_active = False
        self.animation_running = False

        # Update final state
        self._update_progress(1.0)
        self._update_status("Complete!")

        # Auto-close after a short delay
        self.window.after(1000, self._close_window)

        logger.debug("Loading indicator stopped")

    def cancel(self) -> None:
        """Cancel the operation."""
        if not self.is_active:
            return

        self.is_cancelled = True
        self.is_active = False
        self.animation_running = False

        # Update UI
        self._update_status("Cancelled...")

        # Call cancel callback
        if self.cancel_callback:
            try:
                self.cancel_callback()
            except Exception as e:
                logger.error(f"Cancel callback failed: {e}")

        # Close window
        self.window.after(500, self._close_window)

        logger.debug("Loading indicator cancelled")

    def _update_status(self, message: str) -> None:
        """Update status message label."""
        self.status_label.config(text=message)

    def _update_progress(self, progress: float) -> None:
        """Update progress bar and percentage."""
        percentage = int(progress * 100)
        self.progress_var.set(percentage)
        self.percentage_label.config(text=f"{percentage}%")

    def _update_time(self) -> None:
        """Update elapsed time display."""
        if not self.show_time or not self.start_time:
            return

        elapsed = datetime.now() - self.start_time

        # Format elapsed time
        total_seconds = int(elapsed.total_seconds())
        if total_seconds < 60:
            time_str = f"Elapsed: {total_seconds}s"
        else:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            time_str = f"Elapsed: {minutes}m {seconds}s"

        # Add estimated remaining time if we have progress
        if self.progress > 0.1:  # Only estimate after 10% progress
            remaining_seconds = int((total_seconds / self.progress) - total_seconds)
            if remaining_seconds > 0:
                if remaining_seconds < 60:
                    time_str += f" | Est. remaining: {remaining_seconds}s"
                else:
                    minutes = remaining_seconds // 60
                    seconds = remaining_seconds % 60
                    time_str += f" | Est. remaining: {minutes}m {seconds}s"

        self.time_label.config(text=time_str)

    def _start_animation(self) -> None:
        """Start the spinner animation."""
        self.animation_running = True
        self._animate()

    def _animate(self) -> None:
        """Animate the spinner."""
        if not self.animation_running:
            return

        # Update spinner
        spinner_char = self.spinner_chars[self.spinner_index]
        self.spinner_label.config(text=spinner_char)
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)

        # Update time
        self._update_time()

        # Schedule next frame
        self.window.after(100, self._animate)

    def _on_cancel(self) -> None:
        """Handle cancel button click."""
        self.cancel()

    def _close_window(self) -> None:
        """Close the window safely."""
        try:
            self.window.grab_release()
            self.window.destroy()
        except Exception as e:
            logger.warning(f"Failed to close loading indicator window: {e}")


class ProgressDialog:
    """A context manager for progress dialogs.

    Usage:
        with ProgressDialog(parent, "Processing data...") as progress:
            for i, item in enumerate(data):
                # Process item
                progress.update_progress((i + 1) / len(data), f"Processing item {i + 1}")
                if progress.is_cancelled:
                    break
    """

    def __init__(
        self,
        parent: tk.Widget,
        title: str = "Processing...",
        show_cancel: bool = True,
        show_time: bool = True,
    ):
        """Initialize progress dialog context manager.

        Args:
            parent: Parent widget
            title: Dialog title
            show_cancel: Whether to show cancel button
            show_time: Whether to show time estimation
        """
        self.parent = parent
        self.title = title
        self.show_cancel = show_cancel
        self.show_time = show_time
        self.indicator: Optional[LoadingIndicator] = None

    def __enter__(self) -> "ProgressDialog":
        """Enter context manager."""
        self.indicator = LoadingIndicator(
            self.parent,
            title=self.title,
            show_cancel=self.show_cancel,
            show_time=self.show_time,
        )
        self.indicator.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        if self.indicator:
            if exc_type is None:
                self.indicator.stop()
            else:
                self.indicator.cancel()

    def update_progress(self, progress: float, status_message: Optional[str] = None) -> None:
        """Update progress."""
        if self.indicator:
            self.indicator.update_progress(progress, status_message)

    def update_status(self, status_message: str) -> None:
        """Update status."""
        if self.indicator:
            self.indicator.update_status(status_message)

    @property
    def is_cancelled(self) -> bool:
        """Check if operation was cancelled."""
        return self.indicator.is_cancelled if self.indicator else False
