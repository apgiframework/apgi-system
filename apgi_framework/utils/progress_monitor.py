"""Progress monitoring utilities for long-running operations in APGI Framework.

This module provides progress bars and status updates for operations
that may take more than 2 seconds to complete.
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from apgi_framework.logging.standardized_logging import get_logger

logger = get_logger(__name__)


class ProgressStatus(Enum):
    """Status of long-running operations."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ProgressCallback:
    """Callback configuration for progress updates."""

    on_complete: Optional[Callable] = None
    on_error: Optional[Callable] = None
    on_progress: Optional[Callable] = None


class ProgressMonitor:
    """Monitor and display progress for long-running operations."""

    def __init__(
        self,
        total_steps: int,
        operation_name: str = "Operation",
        show_percentage: bool = True,
        show_time_remaining: bool = True,
        callback: Optional[ProgressCallback] = None,
    ):
        self.total_steps = total_steps
        self.current_step = 0
        self.operation_name = operation_name
        self.show_percentage = show_percentage
        self.show_time_remaining = show_time_remaining
        self.callback = callback
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.status = ProgressStatus.PENDING
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start progress monitoring."""
        with self._lock:
            self.status = ProgressStatus.RUNNING
            self.start_time = time.time()
            self.last_update_time = time.time()
            self._update_display()

    def update(self, increment: int = 1, message: str = "") -> None:
        """Update progress by specified increment."""
        with self._lock:
            if self.status == ProgressStatus.RUNNING:
                self.current_step = min(self.current_step + increment, self.total_steps)
                self.last_update_time = time.time()
                self._update_display(message)

                # Call progress callback if provided
                if self.callback and self.callback.on_progress:
                    try:
                        self.callback.on_progress(self.get_percentage(), message)
                    except Exception:
                        pass  # Ignore callback errors to avoid disrupting main operation

    def set_step(self, step: int, message: str = "") -> None:
        """Set progress to specific step."""
        with self._lock:
            if self.status == ProgressStatus.RUNNING:
                self.current_step = min(max(step, 0), self.total_steps)
                self.last_update_time = time.time()
                self._update_display(message)

                # Call progress callback if provided
                if self.callback and self.callback.on_progress:
                    try:
                        self.callback.on_progress(self.get_percentage(), message)
                    except Exception:
                        pass

    def complete(self, result: Any = None) -> None:
        """Mark operation as completed."""
        with self._lock:
            self.status = ProgressStatus.COMPLETED
            self.current_step = self.total_steps
            self._update_display("Completed")

            # Call completion callback if provided
            if self.callback and self.callback.on_complete:
                try:
                    self.callback.on_complete(result)
                except Exception:
                    pass

    def error(self, error: Exception) -> None:
        """Mark operation as failed."""
        with self._lock:
            self.status = ProgressStatus.ERROR
            self._update_display(f"Error: {str(error)}")

            # Call error callback if provided
            if self.callback and self.callback.on_error:
                try:
                    self.callback.on_error(error)
                except Exception:
                    pass

    def get_percentage(self) -> float:
        """Get current progress as percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100

    def get_time_remaining(self) -> float:
        """Get estimated time remaining in seconds."""
        if self.current_step == 0 or self.status != ProgressStatus.RUNNING:
            return 0.0

        elapsed = time.time() - self.start_time
        if self.current_step == 0:
            return 0.0

        # Estimate based on current rate
        rate = self.current_step / elapsed if elapsed > 0 else 0
        remaining_steps = self.total_steps - self.current_step
        return remaining_steps / rate if rate > 0 else 0

    def _update_display(self, message: str = "") -> None:
        """Update progress display."""
        if self.show_percentage:
            percentage = self.get_percentage()
            percentage_str = f"{percentage:.1f}%"
        else:
            percentage_str = f"{self.current_step}/{self.total_steps}"

        if self.show_time_remaining and self.status == ProgressStatus.RUNNING:
            remaining = self.get_time_remaining()
            time_str = f" ({remaining:.0f}s remaining)"
        else:
            time_str = ""

        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * self.get_percentage() / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        display = f"\r{self.operation_name}: [{percentage_str}] {bar} {message}{time_str}"

        # Only update every 0.5 seconds to avoid flickering
        current_time = time.time()
        if current_time - self.last_update_time >= 0.5:
            logger.info(display, end="", flush=True)
            self.last_update_time = current_time


def with_progress(
    total_steps: int,
    operation_name: str,
    show_percentage: bool = True,
    show_time_remaining: bool = True,
    callback: Optional[ProgressCallback] = None,
) -> ProgressMonitor:
    """Context manager for progress monitoring.

    Usage:
        with with_progress(100, "Processing data") as progress:
            for i in range(100):
                progress.update(message=f"Processing item {i+1}")
                time.sleep(0.1)
            progress.complete()
    """
    monitor = ProgressMonitor(
        total_steps=total_steps,
        operation_name=operation_name,
        show_percentage=show_percentage,
        show_time_remaining=show_time_remaining,
        callback=callback,
    )
    return monitor
