"""Performance Monitor for APGI Framework.

Real-time performance monitoring with GUI integration for tracking
system resources, operation performance, and optimization opportunities.
"""

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil

from ..logging.standardized_logging import get_logger
from .performance_profiler import get_profiler

logger = get_logger(__name__)

# Try to import tkinter for GUI components
try:
    import tkinter as tk
    from tkinter import messagebox, ttk

    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    messagebox = None  # type: ignore

    # Create dummy classes for headless environments
    class DummyTk:
        @staticmethod
        def StringVar() -> None:
            return None

        class messagebox:
            @staticmethod
            def showinfo(title: str, message: str) -> None:
                pass

            @staticmethod
            def showerror(title: str, message: str) -> None:
                pass

        class Listbox:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def insert(self, *args: Any) -> None:
                pass

            def delete(self, *args: Any) -> None:
                pass

            def pack(self, *args: Any, **kwargs: Any) -> None:
                pass

            def itemconfig(self, *args: Any) -> None:
                pass

            size = 0

        class Toplevel:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def title(self, *args: Any) -> None:
                pass

            def geometry(self, *args: Any) -> None:
                pass

        class Text:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def insert(self, *args: Any) -> None:
                pass

            def configure(self, *args: Any) -> None:
                pass

            def pack(self, *args: Any, **kwargs: Any) -> None:
                pass

    class DummyTtk:
        @staticmethod
        def LabelFrame(*args: Any, **kwargs: Any) -> None:
            return None

        @staticmethod
        def Frame(*args: Any, **kwargs: Any) -> None:
            return None

        @staticmethod
        def Label(*args: Any, **kwargs: Any) -> None:
            return None

        @staticmethod
        def Progressbar(*args: Any, **kwargs: Any) -> None:
            return None

        @staticmethod
        def Scrollbar(*args: Any, **kwargs: Any) -> None:
            return None

        @staticmethod
        def Button(*args: Any, **kwargs: Any) -> None:
            return None

    tk = DummyTk  # type: ignore
    ttk = DummyTtk  # type: ignore


@dataclass
class SystemMetrics:
    """System performance metrics."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_available_mb": self.memory_available_mb,
            "disk_io_read_mb": self.disk_io_read_mb,
            "disk_io_write_mb": self.disk_io_write_mb,
            "network_sent_mb": self.network_sent_mb,
            "network_recv_mb": self.network_recv_mb,
        }


@dataclass
class PerformanceAlert:
    """Performance alert information."""

    timestamp: float
    alert_type: str  # 'cpu', 'memory', 'disk', 'network'
    severity: str  # 'warning', 'critical'
    message: str
    value: float
    threshold: float


class SystemMonitor:
    """Monitors system performance metrics."""

    def __init__(self, update_interval: float = 1.0, history_size: int = 300):
        self.update_interval = update_interval
        self.history_size = history_size
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Metrics history
        self.metrics_history: deque = deque(maxlen=history_size)

        # Alert thresholds
        self.alert_thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 80.0,
            "memory_critical": 95.0,
            "disk_io_warning": 100.0,  # MB/s
            "disk_io_critical": 500.0,
        }

        # Alert callbacks
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []

        # Previous values for rate calculations
        self._prev_disk_io: Optional[Any] = None
        self._prev_network_io: Optional[Any] = None
        self._prev_timestamp: Optional[float] = None

        self._lock = threading.Lock()

    def start_monitoring(self) -> None:
        """Start system monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")

    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("System monitoring stopped")

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)

    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent metrics."""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_history(self, duration_minutes: Optional[float] = None) -> List[SystemMetrics]:
        """Get metrics history for specified duration."""
        with self._lock:
            if duration_minutes is None:
                return list(self.metrics_history)

            cutoff_time = time.time() - (duration_minutes * 60)
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()

                with self._lock:
                    self.metrics_history.append(metrics)

                # Check for alerts
                self._check_alerts(metrics)

                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)

    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        current_time = time.time()

        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        # Calculate time delta if timestamp available
        time_delta: Optional[float] = None
        if self._prev_timestamp is not None:
            time_delta = current_time - self._prev_timestamp

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = 0.0
        disk_write_mb = 0.0

        if disk_io and self._prev_disk_io is not None and time_delta is not None and time_delta > 0:
            read_delta: int = disk_io.read_bytes - self._prev_disk_io.read_bytes
            write_delta: int = disk_io.write_bytes - self._prev_disk_io.write_bytes
            disk_read_mb = (read_delta / time_delta) / (1024 * 1024)
            disk_write_mb = (write_delta / time_delta) / (1024 * 1024)

        self._prev_disk_io = psutil.disk_io_counters() if disk_io else None

        # Network I/O
        network_io = psutil.net_io_counters()
        network_sent_mb = 0.0
        network_recv_mb = 0.0

        if (
            network_io
            and self._prev_network_io is not None
            and time_delta is not None
            and time_delta > 0
        ):
            sent_delta: int = network_io.bytes_sent - self._prev_network_io.bytes_sent
            recv_delta: int = network_io.bytes_recv - self._prev_network_io.bytes_recv
            network_sent_mb = (sent_delta / time_delta) / (1024 * 1024)
            network_recv_mb = (recv_delta / time_delta) / (1024 * 1024)

        self._prev_network_io = psutil.net_io_counters() if network_io else None
        self._prev_timestamp = current_time

        return SystemMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
        )

    def _check_alerts(self, metrics: SystemMetrics) -> None:
        """Check for performance alerts."""
        alerts = []

        # CPU alerts
        if metrics.cpu_percent >= self.alert_thresholds["cpu_critical"]:
            alerts.append(
                PerformanceAlert(
                    timestamp=metrics.timestamp,
                    alert_type="cpu",
                    severity="critical",
                    message=f"Critical CPU usage: {metrics.cpu_percent:.1f}%",
                    value=metrics.cpu_percent,
                    threshold=self.alert_thresholds["cpu_critical"],
                )
            )
        elif metrics.cpu_percent >= self.alert_thresholds["cpu_warning"]:
            alerts.append(
                PerformanceAlert(
                    timestamp=metrics.timestamp,
                    alert_type="cpu",
                    severity="warning",
                    message=f"High CPU usage: {metrics.cpu_percent:.1f}%",
                    value=metrics.cpu_percent,
                    threshold=self.alert_thresholds["cpu_warning"],
                )
            )

        # Memory alerts
        if metrics.memory_percent >= self.alert_thresholds["memory_critical"]:
            alerts.append(
                PerformanceAlert(
                    timestamp=metrics.timestamp,
                    alert_type="memory",
                    severity="critical",
                    message=f"Critical memory usage: {metrics.memory_percent:.1f}%",
                    value=metrics.memory_percent,
                    threshold=self.alert_thresholds["memory_critical"],
                )
            )
        elif metrics.memory_percent >= self.alert_thresholds["memory_warning"]:
            alerts.append(
                PerformanceAlert(
                    timestamp=metrics.timestamp,
                    alert_type="memory",
                    severity="warning",
                    message=f"High memory usage: {metrics.memory_percent:.1f}%",
                    value=metrics.memory_percent,
                    threshold=self.alert_thresholds["memory_warning"],
                )
            )

        # Disk I/O alerts
        total_disk_io = metrics.disk_io_read_mb + metrics.disk_io_write_mb
        if total_disk_io >= self.alert_thresholds["disk_io_critical"]:
            alerts.append(
                PerformanceAlert(
                    timestamp=metrics.timestamp,
                    alert_type="disk",
                    severity="critical",
                    message=f"Critical disk I/O: {total_disk_io:.1f} MB/s",
                    value=total_disk_io,
                    threshold=self.alert_thresholds["disk_io_critical"],
                )
            )
        elif total_disk_io >= self.alert_thresholds["disk_io_warning"]:
            alerts.append(
                PerformanceAlert(
                    timestamp=metrics.timestamp,
                    alert_type="disk",
                    severity="warning",
                    message=f"High disk I/O: {total_disk_io:.1f} MB/s",
                    value=total_disk_io,
                    threshold=self.alert_thresholds["disk_io_warning"],
                )
            )

        # Send alerts to callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")


class PerformanceMonitorGUI:
    """GUI component for real-time performance monitoring."""

    def __init__(self, parent: Any, monitor: Optional[SystemMonitor] = None) -> None:
        self.parent = parent
        self.monitor = monitor or SystemMonitor()
        self.update_queue: queue.Queue = queue.Queue()

        # Create GUI
        self.frame = ttk.LabelFrame(parent, text="Performance Monitor", padding=10)
        self._create_widgets()

        # Start monitoring
        self.monitor.add_alert_callback(self._on_alert)
        self.monitor.start_monitoring()

        # Start GUI update loop
        self._schedule_update()

    def _create_widgets(self) -> None:
        """Create GUI widgets."""
        # Current metrics display
        metrics_frame = ttk.LabelFrame(self.frame, text="Current Metrics", padding=5)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))

        # CPU
        cpu_frame = ttk.Frame(metrics_frame)
        cpu_frame.pack(fill=tk.X, pady=2)

        ttk.Label(cpu_frame, text="CPU:", width=12).pack(side=tk.LEFT)
        self.cpu_var = tk.StringVar(value="0.0%")
        ttk.Label(cpu_frame, textvariable=self.cpu_var, width=10).pack(side=tk.LEFT)

        self.cpu_progress = ttk.Progressbar(cpu_frame, length=200, maximum=100)
        self.cpu_progress.pack(side=tk.LEFT, padx=(10, 0))

        # Memory
        memory_frame = ttk.Frame(metrics_frame)
        memory_frame.pack(fill=tk.X, pady=2)

        ttk.Label(memory_frame, text="Memory:", width=12).pack(side=tk.LEFT)
        self.memory_var = tk.StringVar(value="0.0%")
        ttk.Label(memory_frame, textvariable=self.memory_var, width=10).pack(side=tk.LEFT)

        self.memory_progress = ttk.Progressbar(memory_frame, length=200, maximum=100)
        self.memory_progress.pack(side=tk.LEFT, padx=(10, 0))

        # Disk I/O
        disk_frame = ttk.Frame(metrics_frame)
        disk_frame.pack(fill=tk.X, pady=2)

        ttk.Label(disk_frame, text="Disk I/O:", width=12).pack(side=tk.LEFT)
        self.disk_var = tk.StringVar(value="0.0 MB/s")
        ttk.Label(disk_frame, textvariable=self.disk_var, width=15).pack(side=tk.LEFT)

        # Alerts display
        alerts_frame = ttk.LabelFrame(self.frame, text="Performance Alerts", padding=5)
        alerts_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Alert list
        self.alert_listbox = tk.Listbox(alerts_frame, height=6)
        alert_scrollbar = ttk.Scrollbar(
            alerts_frame, orient=tk.VERTICAL, command=self.alert_listbox.yview
        )
        self.alert_listbox.configure(yscrollcommand=alert_scrollbar.set)

        self.alert_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        alert_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Control buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(button_frame, text="Clear Alerts", command=self._clear_alerts).pack(side=tk.LEFT)

        ttk.Button(button_frame, text="Show Details", command=self._show_details).pack(
            side=tk.LEFT, padx=(10, 0)
        )

        ttk.Button(button_frame, text="Export Data", command=self._export_data).pack(side=tk.RIGHT)

    def _schedule_update(self) -> None:
        """Schedule GUI update."""
        self._update_display()
        self.parent.after(1000, self._schedule_update)  # Update every second

    def _update_display(self) -> None:
        """Update the display with current metrics."""
        metrics = self.monitor.get_current_metrics()
        if not metrics:
            return

        # Update CPU
        self.cpu_var.set(f"{metrics.cpu_percent:.1f}%")
        self.cpu_progress["value"] = metrics.cpu_percent

        # Color coding for CPU
        if metrics.cpu_percent >= 95:
            self.cpu_progress.configure(style="red.Horizontal.TProgressbar")
        elif metrics.cpu_percent >= 80:
            self.cpu_progress.configure(style="yellow.Horizontal.TProgressbar")
        else:
            self.cpu_progress.configure(style="green.Horizontal.TProgressbar")

        # Update Memory
        self.memory_var.set(f"{metrics.memory_percent:.1f}%")
        self.memory_progress["value"] = metrics.memory_percent

        # Color coding for Memory
        if metrics.memory_percent >= 95:
            self.memory_progress.configure(style="red.Horizontal.TProgressbar")
        elif metrics.memory_percent >= 80:
            self.memory_progress.configure(style="yellow.Horizontal.TProgressbar")
        else:
            self.memory_progress.configure(style="green.Horizontal.TProgressbar")

        # Update Disk I/O
        total_io = metrics.disk_io_read_mb + metrics.disk_io_write_mb
        self.disk_var.set(f"{total_io:.1f} MB/s")

        # Process any queued alerts
        try:
            while True:
                alert = self.update_queue.get_nowait()
                self._add_alert_to_display(alert)
        except queue.Empty:
            pass

    def _on_alert(self, alert: PerformanceAlert) -> None:
        """Handle performance alert."""
        self.update_queue.put(alert)

    def _add_alert_to_display(self, alert: PerformanceAlert) -> None:
        """Add alert to display."""
        timestamp_str = time.strftime("%H:%M:%S", time.localtime(alert.timestamp))
        alert_text = f"[{timestamp_str}] {alert.severity.upper()}: {alert.message}"

        self.alert_listbox.insert(0, alert_text)

        # Limit number of displayed alerts
        if self.alert_listbox.size() > 50:
            self.alert_listbox.delete(tk.END)

        # Color coding
        if alert.severity == "critical":
            self.alert_listbox.itemconfig(0, {"fg": "red"})
        elif alert.severity == "warning":
            self.alert_listbox.itemconfig(0, {"fg": "orange"})

    def _clear_alerts(self) -> None:
        """Clear all alerts from display."""
        self.alert_listbox.delete(0, tk.END)

    def _show_details(self) -> None:
        """Show detailed performance information."""
        details_window = tk.Toplevel(self.parent)
        details_window.title("Performance Details")
        details_window.geometry("600x400")

        # Create text widget for details
        text_widget = tk.Text(details_window, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(details_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Get recent metrics
        recent_metrics = self.monitor.get_metrics_history(duration_minutes=5)

        if recent_metrics:
            # Calculate statistics
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]

            details_text = f"""Performance Details (Last 5 minutes)

CPU Usage:
  Current: {cpu_values[-1]:.1f}%
  Average: {np.mean(cpu_values):.1f}%
  Maximum: {np.max(cpu_values):.1f}%
  Minimum: {np.min(cpu_values):.1f}%

Memory Usage:
  Current: {memory_values[-1]:.1f}%
  Average: {np.mean(memory_values):.1f}%
  Maximum: {np.max(memory_values):.1f}%
  Minimum: {np.min(memory_values):.1f}%

System Information:
  Total Memory: {recent_metrics[-1].memory_used_mb + recent_metrics[-1].memory_available_mb:.0f} MB
  Available Memory: {recent_metrics[-1].memory_available_mb:.0f} MB
  
Profiler Statistics:
"""

            # Add profiler information
            profiler = get_profiler()
            summary = profiler.get_summary()

            for operation, stats in summary.items():
                details_text += f"\n{operation}:\n"
                details_text += f"  Runs: {stats.get('total_runs', 0)}\n"
                details_text += f"  Avg Time: {stats.get('avg_time', 0):.3f}s\n"
                details_text += f"  Max Time: {stats.get('max_time', 0):.3f}s\n"

            text_widget.insert(tk.END, details_text)
        else:
            text_widget.insert(tk.END, "No performance data available.")

        text_widget.configure(state=tk.DISABLED)

    def _export_data(self) -> None:
        """Export performance data."""
        import json
        from tkinter import filedialog

        filename = filedialog.asksaveasfilename(
            title="Export Performance Data",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if filename:
            try:
                # Get all metrics history
                metrics_history = self.monitor.get_metrics_history()

                # Convert to exportable format
                export_data = {
                    "timestamp": time.time(),
                    "metrics": [m.to_dict() for m in metrics_history],
                    "profiler_summary": get_profiler().get_summary(),
                }

                with open(filename, "w") as f:
                    json.dump(export_data, f, indent=2)

                messagebox.showinfo("Success", f"Performance data exported to {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")

    def get_widget(self) -> Any:
        """Get the main widget frame."""
        return self.frame

    def pack(self, **kwargs: Any) -> None:
        """Pack the widget."""
        self.frame.pack(**kwargs)

    def grid(self, **kwargs: Any) -> None:
        """Grid the widget."""
        self.frame.grid(**kwargs)

    def destroy(self) -> None:
        """Clean up resources."""
        self.monitor.stop_monitoring()


# Global monitor instance
_global_monitor = None


def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SystemMonitor()
    return _global_monitor


def create_performance_monitor_gui(
    parent: Any, monitor: Optional[SystemMonitor] = None
) -> PerformanceMonitorGUI:
    """Create a performance monitor GUI component."""
    if not HAS_TKINTER:
        raise ImportError("tkinter is not available. GUI components cannot be created.")
    return PerformanceMonitorGUI(parent, monitor)
