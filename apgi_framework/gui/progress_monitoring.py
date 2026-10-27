"""Real-time progress monitoring for parameter estimation tasks.

Provides classes for tracking task completion and data quality during experiments.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class RealTimeProgressMonitor:
    """Monitors task progress and data quality in real-time.

    Tracks trial completion, timing, and provides progress updates.
    """

    def __init__(self) -> None:
        """Initialize progress monitor."""
        self.current_task: Optional[str] = None
        self.total_trials: int = 0
        self.completed_trials: int = 0
        self.start_time: Optional[datetime] = None
        self.estimated_completion: Optional[datetime] = None

        # Callbacks for progress updates
        self.progress_callbacks: list[Callable[[Dict[str, Any]], None]] = []

        logger.info("RealTimeProgressMonitor initialized")

    def start_task(self, task_name: str, total_trials: int) -> None:
        """Start monitoring a new task.

        Args:
            task_name: Name of the task
            total_trials: Total number of trials
        """
        self.current_task = task_name
        self.total_trials = total_trials
        self.completed_trials = 0
        self.start_time = datetime.now()
        self.estimated_completion = None

        logger.info(f"Started monitoring task: {task_name} ({total_trials} trials)")
        self._notify_progress()

    def update_progress(self, completed_trials: int) -> None:
        """Update progress with number of completed trials.

        Args:
            completed_trials: Number of completed trials
        """
        self.completed_trials = completed_trials

        # Estimate completion time
        if self.start_time and completed_trials > 0:
            elapsed = datetime.now() - self.start_time
            trials_per_second = completed_trials / elapsed.total_seconds()
            remaining_trials = self.total_trials - completed_trials

            if trials_per_second > 0:
                remaining_seconds = remaining_trials / trials_per_second
                self.estimated_completion = datetime.now() + timedelta(seconds=remaining_seconds)

        self._notify_progress()

    def complete_task(self, success: bool = True) -> None:
        """Mark current task as completed.

        Args:
            success: Whether the task completed successfully
        """
        self.completed_trials = self.total_trials
        self._notify_progress()

        logger.info(f"Task completed: {self.current_task} (success={success})")
        self.current_task = None

    def is_active(self) -> bool:
        """Check if a task is currently being monitored.

        Returns:
            True if a task is active
        """
        return self.current_task is not None

    def add_progress_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for progress updates.

        Args:
            callback: Function to call with progress updates
        """
        self.progress_callbacks.append(callback)

    def _notify_progress(self) -> None:
        """Notify all registered callbacks of progress update."""
        progress_data = self.get_progress_data()

        for callback in self.progress_callbacks:
            try:
                callback(progress_data)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def get_progress_data(self) -> Dict[str, Any]:
        """Get current progress data.

        Returns:
            Dictionary with progress information
        """
        progress_percent = 0.0
        if self.total_trials > 0:
            progress_percent = (self.completed_trials / self.total_trials) * 100

        elapsed_time = None
        if self.start_time:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()

        return {
            "task_name": self.current_task,
            "total_trials": self.total_trials,
            "completed_trials": self.completed_trials,
            "progress_percent": progress_percent,
            "elapsed_seconds": elapsed_time,
            "estimated_completion": (
                self.estimated_completion.isoformat() if self.estimated_completion else None
            ),
        }

    def get_progress_string(self) -> str:
        """Get formatted progress string.

        Returns:
            Human-readable progress string
        """
        if not self.current_task:
            return "No task running"

        progress_data = self.get_progress_data()

        progress_str = f"{self.current_task}: {self.completed_trials}/{self.total_trials} trials "
        progress_str += f"({progress_data['progress_percent']:.1f}%)"

        if self.estimated_completion:
            progress_str += f" - ETA: {self.estimated_completion.strftime('%H:%M:%S')}"

        return progress_str

    def reset(self) -> None:
        """Reset progress monitor."""
        self.current_task = None
        self.total_trials = 0
        self.completed_trials = 0
        self.start_time = None
        self.estimated_completion = None

        logger.info("Progress monitor reset")


if __name__ == "__main__":
    """Launch the progress monitoring demo as a standalone application."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    class ProgressMonitorDemo:
        """Demo application for progress monitoring."""

        def __init__(self) -> None:
            self.root = tk.Tk()
            self.root.title("Progress Monitoring Demo")
            self.root.geometry("600x400")

            self.monitor: RealTimeProgressMonitor = RealTimeProgressMonitor()
            self.setup_ui()

        def setup_ui(self) -> None:
            """Setup demo UI."""
            # Main frame
            main_frame = ttk.Frame(self.root, padding="20")
            main_frame.grid(row=0, column=0, sticky="nsew")

            # Title
            title = ttk.Label(
                main_frame, text="Progress Monitoring Demo", font=("Arial", 16, "bold")
            )
            title.grid(row=0, column=0, columnspan=2, pady=(0, 20))

            # Start button
            start_btn = ttk.Button(main_frame, text="Start Demo Task", command=self.start_demo)
            start_btn.grid(row=1, column=0, padx=5, pady=5)

            # Status display
            self.status_text = tk.Text(main_frame, height=15, width=60)
            self.status_text.grid(row=2, column=0, columnspan=2, pady=10)

        def start_demo(self) -> None:
            """Start demo task."""
            self.monitor.start_task("Demo Task", 100)
            self.update_progress()

        def update_progress(self) -> None:
            """Update progress display."""
            if self.monitor.is_active():
                # Simulate progress
                completed = self.monitor.completed_trials + 1
                if completed <= self.monitor.total_trials:
                    self.monitor.update_progress(completed)

                    # Update display
                    status = self.monitor.get_progress_data()
                    self.status_text.delete(1.0, tk.END)
                    self.status_text.insert(tk.END, f"Task: {status['current_task']}\n")
                    self.status_text.insert(
                        tk.END, f"Progress: {status['progress_percent']:.1f}%\n"
                    )
                    self.status_text.insert(
                        tk.END,
                        f"Completed: {status['completed_trials']}/{status['total_trials']}\n",
                    )
                    self.status_text.insert(tk.END, f"Elapsed: {status['elapsed_time']}\n")
                    if status["estimated_completion"]:
                        self.status_text.insert(tk.END, f"ETA: {status['estimated_completion']}\n")

                    # Schedule next update
                    self.root.after(100, self.update_progress)
                else:
                    self.monitor.complete_task(success=True)
                    messagebox.showinfo("Complete", "Demo task completed successfully!")

        def run(self) -> None:
            """Run the demo."""
            self.root.mainloop()


if __name__ == "__main__":
    # Run demo
    demo = ProgressMonitorDemo()
    demo.run()
