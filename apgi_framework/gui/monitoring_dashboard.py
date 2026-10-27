"""Real-time multi-modal monitoring dashboard.

Provides live monitoring of EEG, pupillometry, cardiac signals, and parameter estimates.
"""

import asyncio
import json
import os
import sys
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import asdict
from datetime import datetime
from tkinter import ttk
from typing import Any, Dict, List, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from ..logging.standardized_logging import get_logger
    from ..neural.eeg_interface import EEGInterface
    from .realtime_data_stream import get_streamer
except ImportError:
    # Handle relative import when run directly
    from apgi_framework.gui.realtime_data_stream import get_streamer
    from apgi_framework.logging.standardized_logging import get_logger
    from apgi_framework.neural.eeg_interface import EEGInterface

# Check for websockets library
try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None  # type: ignore
    WEBSOCKETS_AVAILABLE = False

logger = get_logger(__name__)


class LiveEEGMonitor:
    """Real-time EEG signal quality and P3b/HEP display.

    Monitors EEG data quality, artifact rates, and neural signatures.
    """

    def __init__(self, parent_frame: tk.Widget):
        """Initialize EEG monitor.

        Args:
            parent_frame: Parent tkinter frame
        """
        self.frame = ttk.LabelFrame(parent_frame, text="EEG Monitoring", padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create main container
        main_container = ttk.Frame(self.frame)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Top section - signal quality and controls
        top_frame = ttk.Frame(main_container)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # Signal quality indicators
        quality_frame = ttk.Frame(top_frame)
        quality_frame.pack(fill=tk.X, pady=5)

        ttk.Label(quality_frame, text="Signal Quality:").pack(side=tk.LEFT, padx=5)
        self.quality_label = ttk.Label(quality_frame, text="N/A", foreground="gray")
        self.quality_label.pack(side=tk.LEFT, padx=5)

        ttk.Label(quality_frame, text="Artifact Rate:").pack(side=tk.LEFT, padx=5)
        self.artifact_label = ttk.Label(quality_frame, text="N/A", foreground="gray")
        self.artifact_label.pack(side=tk.LEFT, padx=5)

        # Control buttons
        control_frame = ttk.Frame(top_frame)
        control_frame.pack(fill=tk.X, pady=5)

        self.start_button = ttk.Button(
            control_frame, text="Start Monitoring", command=self.start_monitoring
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Monitoring",
            command=self.stop_monitoring,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Middle section - real-time signal visualization
        signal_frame = ttk.LabelFrame(main_container, text="Real-time EEG Signal", padding=5)
        signal_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create matplotlib figure for EEG signal
        self.signal_fig = Figure(figsize=(8, 3), dpi=80)
        self.signal_ax = self.signal_fig.add_subplot(111)
        self.signal_ax.set_xlabel("Time (s)")
        self.signal_ax.set_ylabel("Amplitude (μV)")
        self.signal_ax.set_title("EEG Signal (Sample Channels)")
        self.signal_ax.grid(True, alpha=0.3)

        # Initialize signal data storage
        self.signal_window = 5.0  # 5 seconds of data
        self.sampling_rate = 250.0  # Hz
        self.signal_points = int(self.signal_window * self.sampling_rate)
        self.signal_time: deque[float] = deque(maxlen=self.signal_points)
        self.signal_data: deque[float] = deque(maxlen=self.signal_points)

        # Create signal plot lines (show first 4 channels)
        self.signal_lines = []
        colors = ["blue", "red", "green", "orange"]
        for i in range(4):
            (line,) = self.signal_ax.plot(
                [], [], colors[i], alpha=0.7, linewidth=1, label=f"Ch {i + 1}"
            )
            self.signal_lines.append(line)

        self.signal_ax.legend(loc="upper right", fontsize="small")
        self.signal_ax.set_xlim(0, self.signal_window)
        self.signal_ax.set_ylim(-100, 100)  # μV range

        # Embed signal figure
        self.signal_canvas = FigureCanvasTkAgg(self.signal_fig, signal_frame)  # type: ignore[no-untyped-call]
        self.signal_canvas.draw()  # type: ignore[no-untyped-call]
        self.signal_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # type: ignore[no-untyped-call]

        # Bottom section - neural signatures
        signature_frame = ttk.LabelFrame(main_container, text="Neural Signatures", padding=5)
        signature_frame.pack(fill=tk.X, pady=5)

        ttk.Label(signature_frame, text="P3b Amplitude (μV):").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.p3b_label = ttk.Label(signature_frame, text="N/A")
        self.p3b_label.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(signature_frame, text="HEP Amplitude (μV):").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.hep_label = ttk.Label(signature_frame, text="N/A")
        self.hep_label.grid(row=1, column=1, sticky=tk.W, pady=2)

        # Bad channels display
        ttk.Label(signature_frame, text="Bad Channels:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.bad_channels_label = ttk.Label(signature_frame, text="None")
        self.bad_channels_label.grid(row=2, column=1, sticky=tk.W, pady=2)

        # Current state
        self.current_quality = 0.0
        self.artifact_rate = 0.0
        self.p3b_amplitude: Optional[float] = None
        self.hep_amplitude: Optional[float] = None
        self.bad_channels: List[str] = []

        # Real-time data streaming
        self.eeg_interface: Optional[EEGInterface] = None
        self.is_monitoring = False
        self.update_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Signal generation for visualization
        self.signal_start_time = time.time()

        logger.info("LiveEEGMonitor initialized")

    def update_signal_quality(self, quality_score: float, artifact_rate: float) -> None:
        """Update EEG signal quality display.

        Args:
            quality_score: Signal quality score (0-1)
            artifact_rate: Artifact rate (0-1)
        """
        self.current_quality = quality_score
        self.artifact_rate = artifact_rate

        # Update quality label with color coding
        quality_text = f"{quality_score:.2f}"
        if quality_score >= 0.8:
            color = "green"
        elif quality_score >= 0.6:
            color = "orange"
        else:
            color = "red"

        self.quality_label.config(text=quality_text, foreground=color)

        # Update artifact rate
        artifact_text = f"{artifact_rate * 100:.1f}%"
        artifact_color = (
            "green" if artifact_rate < 0.2 else "orange" if artifact_rate < 0.4 else "red"
        )
        self.artifact_label.config(text=artifact_text, foreground=artifact_color)

    def update_p3b(self, amplitude: float, latency: Optional[float] = None) -> None:
        """Update P3b amplitude display.

        Args:
            amplitude: P3b amplitude in μV
            latency: Optional P3b latency in ms
        """
        self.p3b_amplitude = amplitude

        if latency:
            text = f"{amplitude:.2f} μV ({latency:.0f} ms)"
        else:
            text = f"{amplitude:.2f} μV"

        self.p3b_label.config(text=text)

    def update_hep(self, amplitude: float) -> None:
        """Update HEP amplitude display.

        Args:
            amplitude: HEP amplitude in μV
        """
        self.hep_amplitude = amplitude
        self.hep_label.config(text=f"{amplitude:.2f} μV")

    def update_bad_channels(self, bad_channels: List[str]) -> None:
        """Update bad channels display.

        Args:
            bad_channels: List of bad channel names
        """
        self.bad_channels = bad_channels

        if bad_channels:
            text = ", ".join(bad_channels[:5])  # Show first 5
            if len(bad_channels) > 5:
                text += f" (+{len(bad_channels) - 5} more)"
            self.bad_channels_label.config(text=text, foreground="red")
        else:
            self.bad_channels_label.config(text="None", foreground="green")

    def start_monitoring(self) -> None:
        """Start real-time EEG monitoring."""
        if self.is_monitoring:
            return

        try:
            # Initialize EEG interface
            from apgi_framework.neural.eeg_interface import EEGConfig

            config = EEGConfig(sampling_rate=250.0)
            self.eeg_interface = EEGInterface(config)

            # Register callback for data processing
            self.eeg_interface.register_callback(self._process_eeg_data)

            # Start streaming with simulated data
            self.eeg_interface.start_streaming()

            # Start GUI update thread
            self.is_monitoring = True
            self.stop_event.clear()
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()

            # Update button states
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

            logger.info("EEG monitoring started")

        except Exception as e:
            logger.error(f"Failed to start EEG monitoring: {e}")
            self.stop_monitoring()

    def stop_monitoring(self) -> None:
        """Stop real-time EEG monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self.stop_event.set()

        # Stop EEG streaming
        if self.eeg_interface:
            self.eeg_interface.stop_streaming()  # type: ignore[no-untyped-call]
            self.eeg_interface = None

        # Wait for update thread to finish
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
            self.update_thread = None

        # Update button states
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        logger.info("EEG monitoring stopped")

    def _process_eeg_data(
        self, data: np.ndarray, timestamp: float, artifacts: Dict[str, Any]
    ) -> None:
        """Process incoming EEG data."""
        # Calculate signal quality (simplified)
        quality_score = 1.0 - artifacts.get("artifact_rate", 0.0)
        artifact_rate = artifacts.get("artifact_rate", 0.0)

        # Update GUI in main thread
        self.frame.after(0, self.update_signal_quality, quality_score, artifact_rate)

        # Simulate neural signature detection
        if np.random.random() < 0.1:  # 10% chance of detection
            p3b_amp = np.random.normal(5.0, 2.0)  # Simulated P3b
            hep_amp = np.random.normal(3.0, 1.5)  # Simulated HEP
            self.frame.after(0, self.update_p3b, p3b_amp)
            self.frame.after(0, self.update_hep, hep_amp)

    def _update_loop(self) -> None:
        """GUI update loop."""
        while self.is_monitoring and not self.stop_event.is_set():
            # Generate simulated EEG signal for visualization
            current_time = time.time() - self.signal_start_time

            # Generate 4 channels of simulated EEG data
            for i in range(4):
                # Different frequencies for each channel to show variety
                freq = 10 + i * 2  # 10Hz, 12Hz, 14Hz, 16Hz
                amplitude = 20 + i * 5  # Different amplitudes
                signal_value = amplitude * np.sin(
                    2 * np.pi * freq * current_time
                ) + np.random.normal(0, 2)

                # Add to signal data
                self.signal_time.append(current_time)
                self.signal_data.append(signal_value)

            # Update signal plot
            if len(self.signal_time) > 0:
                # Update each channel line
                for i, line in enumerate(self.signal_lines):
                    # Extract data for this channel
                    channel_data = [self.signal_data[j] for j in range(i, len(self.signal_data), 4)]
                    channel_time = [self.signal_time[j] for j in range(i, len(self.signal_time), 4)]

                    if channel_time and channel_data:
                        line.set_data(channel_time, channel_data)

                # Adjust x-axis to show sliding window
                if current_time > self.signal_window:
                    self.signal_ax.set_xlim(current_time - self.signal_window, current_time)

                # Redraw canvas
                self.signal_canvas.draw()  # type: ignore[no-untyped-call]

            # Update display at 10 Hz
            time.sleep(0.1)

    def reset(self) -> None:
        """Reset monitor display."""
        self.quality_label.config(text="N/A", foreground="gray")
        self.artifact_label.config(text="N/A", foreground="gray")
        self.p3b_label.config(text="N/A")
        self.hep_label.config(text="N/A")
        self.bad_channels_label.config(text="None", foreground="gray")


class PupillometryMonitor:
    """Live pupillometry signal monitoring.

    Displays pupil diameter, blink rate, and data quality.
    """

    def __init__(self, parent_frame: tk.Widget):
        """Initialize pupillometry monitor.

        Args:
            parent_frame: Parent tkinter frame
        """
        self.frame = ttk.LabelFrame(parent_frame, text="Pupillometry Monitoring", padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create main container
        main_container = ttk.Frame(self.frame)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Top section - signal quality
        top_frame = ttk.Frame(main_container)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        quality_frame = ttk.Frame(top_frame)
        quality_frame.pack(fill=tk.X, pady=5)

        ttk.Label(quality_frame, text="Data Quality:").pack(side=tk.LEFT, padx=5)
        self.quality_label = ttk.Label(quality_frame, text="N/A", foreground="gray")
        self.quality_label.pack(side=tk.LEFT, padx=5)

        ttk.Label(quality_frame, text="Data Loss:").pack(side=tk.LEFT, padx=5)
        self.data_loss_label = ttk.Label(quality_frame, text="N/A", foreground="gray")
        self.data_loss_label.pack(side=tk.LEFT, padx=5)

        # Middle section - real-time visualization
        viz_frame = ttk.LabelFrame(main_container, text="Real-time Pupillometry Data", padding=5)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create matplotlib figure for pupillometry data
        self.fig = Figure(figsize=(8, 4), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Pupil Diameter and Blink Rate")
        self.ax.grid(True, alpha=0.3)

        # Initialize data storage for plotting
        self.max_points = 100
        self.time_data: deque[float] = deque(maxlen=self.max_points)
        self.pupil_diameter_data: deque[float] = deque(maxlen=self.max_points)
        self.blink_rate_data: deque[float] = deque(maxlen=self.max_points)
        self.tracking_loss_data: deque[int] = deque(maxlen=self.max_points)

        # Create plot lines with different y-axes
        self.ax2 = self.ax.twinx()  # Secondary y-axis for blink rate

        # Pupil diameter line (left axis)
        (self.pupil_line,) = self.ax.plot([], [], "b-", label="Pupil Diameter (mm)", linewidth=2)
        # Blink rate line (right axis)
        (self.blink_line,) = self.ax2.plot(
            [], [], "r-", label="Blink Rate (bpm)", linewidth=2, alpha=0.7
        )

        # Configure axes
        self.ax.set_ylabel("Pupil Diameter (mm)", color="b")
        self.ax2.set_ylabel("Blink Rate (bpm)", color="r")
        self.ax.tick_params(axis="y", labelcolor="b")
        self.ax2.tick_params(axis="y", labelcolor="r")

        # Add legend
        lines1, labels1 = self.ax.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Set initial axis limits
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(2, 8)  # Pupil diameter range (mm)
        self.ax2.set_ylim(0, 30)  # Blink rate range (bpm)

        # Embed figure
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)  # type: ignore[no-untyped-call]
        self.canvas.draw()  # type: ignore[no-untyped-call]
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # type: ignore[no-untyped-call]

        # Bottom section - measurements
        measurements_frame = ttk.LabelFrame(main_container, text="Current Measurements", padding=5)
        measurements_frame.pack(fill=tk.X, pady=5)

        ttk.Label(measurements_frame, text="Pupil Diameter (mm):").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.diameter_label = ttk.Label(measurements_frame, text="N/A")
        self.diameter_label.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(measurements_frame, text="Blink Rate (bpm):").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.blink_rate_label = ttk.Label(measurements_frame, text="N/A")
        self.blink_rate_label.grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(measurements_frame, text="Tracking Loss:").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        self.tracking_loss_label = ttk.Label(measurements_frame, text="N/A")
        self.tracking_loss_label.grid(row=2, column=1, sticky=tk.W, pady=2)

        # Start time for x-axis
        self.start_time = time.time()

        logger.info("PupillometryMonitor initialized")

    def update_quality(self, quality_score: float, data_loss: float) -> None:
        """Update pupillometry data quality.

        Args:
            quality_score: Quality score (0-1)
            data_loss: Data loss proportion (0-1)
        """
        # Update quality
        quality_text = f"{quality_score:.2f}"
        quality_color = (
            "green" if quality_score >= 0.8 else "orange" if quality_score >= 0.6 else "red"
        )
        self.quality_label.config(text=quality_text, foreground=quality_color)

        # Update data loss
        loss_text = f"{data_loss * 100:.1f}%"
        loss_color = "green" if data_loss < 0.1 else "orange" if data_loss < 0.3 else "red"
        self.data_loss_label.config(text=loss_text, foreground=loss_color)

    def update_measurements(self, diameter: float, blink_rate: float, tracking_loss: int) -> None:
        """Update pupil measurements.

        Args:
            diameter: Pupil diameter in mm
            blink_rate: Blink rate in blinks per minute
            tracking_loss: Number of tracking loss episodes
        """
        self.diameter_label.config(text=f"{diameter:.2f} mm")
        self.blink_rate_label.config(text=f"{blink_rate:.1f} bpm")
        self.tracking_loss_label.config(text=str(tracking_loss))

        # Update visualization data
        current_time = time.time() - self.start_time
        self.time_data.append(current_time)
        self.pupil_diameter_data.append(diameter)
        self.blink_rate_data.append(blink_rate)
        self.tracking_loss_data.append(tracking_loss)

        self._update_plot()

    def _update_plot(self) -> None:
        """Update the matplotlib plot with current data."""
        if len(self.time_data) > 0:
            # Update line data
            self.pupil_line.set_data(list(self.time_data), list(self.pupil_diameter_data))
            self.blink_line.set_data(list(self.time_data), list(self.blink_rate_data))

            # Adjust x-axis limits to show sliding window
            if self.time_data[-1] > 10:
                self.ax.set_xlim(self.time_data[-1] - 10, self.time_data[-1])
            else:
                self.ax.set_xlim(0, 10)

            # Adjust y-axis limits based on data range
            if self.pupil_diameter_data:
                y_min, y_max = min(self.pupil_diameter_data), max(self.pupil_diameter_data)
                margin = (y_max - y_min) * 0.1
                self.ax.set_ylim(y_min - margin, y_max + margin)

            if self.blink_rate_data:
                y_min, y_max = min(self.blink_rate_data), max(self.blink_rate_data)
                margin = (y_max - y_min) * 0.1
                self.ax2.set_ylim(y_min - margin, y_max + margin)

            # Redraw canvas
            self.canvas.draw()  # type: ignore[no-untyped-call]

    def reset(self) -> None:
        """Reset monitor display."""
        self.quality_label.config(text="N/A", foreground="gray")
        self.data_loss_label.config(text="N/A", foreground="gray")
        self.diameter_label.config(text="N/A")
        self.blink_rate_label.config(text="N/A")
        self.tracking_loss_label.config(text="N/A")


class CardiacMonitor:
    """Live cardiac signal monitoring.

    Displays heart rate, HRV, and R-peak detection quality.
    """

    def __init__(self, parent_frame: tk.Widget):
        """Initialize cardiac monitor.

        Args:
            parent_frame: Parent tkinter frame
        """
        self.frame = ttk.LabelFrame(parent_frame, text="Cardiac Monitoring", padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create main container
        main_container = ttk.Frame(self.frame)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Top section - signal quality
        top_frame = ttk.Frame(main_container)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        quality_frame = ttk.Frame(top_frame)
        quality_frame.pack(fill=tk.X, pady=5)

        ttk.Label(quality_frame, text="Signal Quality:").pack(side=tk.LEFT, padx=5)
        self.quality_label = ttk.Label(quality_frame, text="N/A", foreground="gray")
        self.quality_label.pack(side=tk.LEFT, padx=5)

        ttk.Label(quality_frame, text="R-peak Confidence:").pack(side=tk.LEFT, padx=5)
        self.rpeak_label = ttk.Label(quality_frame, text="N/A", foreground="gray")
        self.rpeak_label.pack(side=tk.LEFT, padx=5)

        # Middle section - real-time visualization
        viz_frame = ttk.LabelFrame(main_container, text="Real-time Cardiac Data", padding=5)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create matplotlib figure for cardiac data
        self.fig = Figure(figsize=(8, 4), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Heart Rate and HRV")
        self.ax.grid(True, alpha=0.3)

        # Initialize data storage for plotting
        self.max_points = 100
        self.time_data: deque[float] = deque(maxlen=self.max_points)
        self.heart_rate_data: deque[float] = deque(maxlen=self.max_points)
        self.hrv_data: deque[float] = deque(maxlen=self.max_points)
        self.rr_interval_data: deque[float] = deque(maxlen=self.max_points)

        # Create plot lines with different y-axes
        self.ax2 = self.ax.twinx()  # Secondary y-axis for HRV

        # Heart rate line (left axis)
        (self.hr_line,) = self.ax.plot([], [], "r-", label="Heart Rate (BPM)", linewidth=2)
        # HRV line (right axis)
        (self.hrv_line,) = self.ax2.plot(
            [], [], "g-", label="HRV (RMSSD ms)", linewidth=2, alpha=0.7
        )

        # Configure axes
        self.ax.set_ylabel("Heart Rate (BPM)", color="r")
        self.ax2.set_ylabel("HRV (RMSSD ms)", color="g")
        self.ax.tick_params(axis="y", labelcolor="r")
        self.ax2.tick_params(axis="y", labelcolor="g")

        # Add legend
        lines1, labels1 = self.ax.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Set initial axis limits
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(50, 100)  # Heart rate range (BPM)
        self.ax2.set_ylim(10, 100)  # HRV range (ms)

        # Embed figure
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)  # type: ignore[no-untyped-call]
        self.canvas.draw()  # type: ignore[no-untyped-call]
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # type: ignore[no-untyped-call]

        # Bottom section - measurements
        measurements_frame = ttk.LabelFrame(main_container, text="Current Measurements", padding=5)
        measurements_frame.pack(fill=tk.X, pady=5)

        ttk.Label(measurements_frame, text="Heart Rate (BPM):").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.hr_label = ttk.Label(measurements_frame, text="N/A")
        self.hr_label.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(measurements_frame, text="HRV (RMSSD ms):").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.hrv_label = ttk.Label(measurements_frame, text="N/A")
        self.hrv_label.grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(measurements_frame, text="RR Interval (ms):").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        self.rr_label = ttk.Label(measurements_frame, text="N/A")
        self.rr_label.grid(row=2, column=1, sticky=tk.W, pady=2)

        # Start time for x-axis
        self.start_time = time.time()

        logger.info("CardiacMonitor initialized")

    def update_quality(self, quality_score: float, rpeak_confidence: float) -> None:
        """Update cardiac signal quality.

        Args:
            quality_score: Signal quality score (0-1)
            rpeak_confidence: R-peak detection confidence (0-1)
        """
        # Update quality
        quality_text = f"{quality_score:.2f}"
        quality_color = (
            "green" if quality_score >= 0.8 else "orange" if quality_score >= 0.6 else "red"
        )
        self.quality_label.config(text=quality_text, foreground=quality_color)

        # Update R-peak confidence
        rpeak_text = f"{rpeak_confidence:.2f}"
        rpeak_color = (
            "green" if rpeak_confidence >= 0.9 else "orange" if rpeak_confidence >= 0.7 else "red"
        )
        self.rpeak_label.config(text=rpeak_text, foreground=rpeak_color)

    def update_measurements(self, heart_rate: float, hrv: float, rr_interval: float) -> None:
        """Update cardiac measurements.

        Args:
            heart_rate: Heart rate in BPM
            hrv: Heart rate variability (RMSSD) in ms
            rr_interval: RR interval in ms
        """
        self.hr_label.config(text=f"{heart_rate:.1f} BPM")
        self.hrv_label.config(text=f"{hrv:.1f} ms")
        self.rr_label.config(text=f"{rr_interval:.1f} ms")

        # Update visualization data
        current_time = time.time() - self.start_time
        self.time_data.append(current_time)
        self.heart_rate_data.append(heart_rate)
        self.hrv_data.append(hrv)
        self.rr_interval_data.append(rr_interval)

        self._update_plot()

    def _update_plot(self) -> None:
        """Update the matplotlib plot with current data."""
        if len(self.time_data) > 0:
            # Update line data
            self.hr_line.set_data(list(self.time_data), list(self.heart_rate_data))
            self.hrv_line.set_data(list(self.time_data), list(self.hrv_data))

            # Adjust x-axis limits to show sliding window
            if self.time_data[-1] > 10:
                self.ax.set_xlim(self.time_data[-1] - 10, self.time_data[-1])
            else:
                self.ax.set_xlim(0, 10)

            # Adjust y-axis limits based on data range
            if self.heart_rate_data:
                y_min, y_max = min(self.heart_rate_data), max(self.heart_rate_data)
                margin = (y_max - y_min) * 0.1
                self.ax.set_ylim(y_min - margin, y_max + margin)

            if self.hrv_data:
                y_min, y_max = min(self.hrv_data), max(self.hrv_data)
                margin = (y_max - y_min) * 0.1
                self.ax2.set_ylim(y_min - margin, y_max + margin)

            # Redraw canvas
            self.canvas.draw()  # type: ignore[no-untyped-call]

    def reset(self) -> None:
        """Reset monitor display."""
        self.quality_label.config(text="N/A", foreground="gray")
        self.rpeak_label.config(text="N/A", foreground="gray")
        self.hr_label.config(text="N/A")
        self.hrv_label.config(text="N/A")
        self.rr_label.config(text="N/A")


class RealTimeParameterEstimateUpdater:
    """Real-time parameter estimate updates.

    Displays ongoing Bayesian parameter estimates as data accumulates.
    """

    def __init__(self, parent_frame: tk.Widget):
        """Initialize parameter estimate updater.

        Args:
            parent_frame: Parent tkinter frame
        """
        self.frame = ttk.LabelFrame(parent_frame, text="Parameter Estimates", padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create main container with text and visualization
        main_container = ttk.Frame(self.frame)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left side - text displays
        text_frame = ttk.Frame(main_container)
        text_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Parameter displays
        params_frame = ttk.Frame(text_frame)
        params_frame.pack(fill=tk.Y)

        # θ₀ (theta0)
        ttk.Label(params_frame, text="θ₀ (Ignition Threshold):").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.theta0_label = ttk.Label(params_frame, text="N/A")
        self.theta0_label.grid(row=0, column=1, sticky=tk.W, pady=2)

        # Πᵢ (pi_i)
        ttk.Label(params_frame, text="Πᵢ (Interoceptive Precision):").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.pi_i_label = ttk.Label(params_frame, text="N/A")
        self.pi_i_label.grid(row=1, column=1, sticky=tk.W, pady=2)

        # β (beta)
        ttk.Label(params_frame, text="β (Somatic Bias):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.beta_label = ttk.Label(params_frame, text="N/A")
        self.beta_label.grid(row=2, column=1, sticky=tk.W, pady=2)

        # Convergence status
        ttk.Label(params_frame, text="Convergence:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.convergence_label = ttk.Label(params_frame, text="N/A")
        self.convergence_label.grid(row=3, column=1, sticky=tk.W, pady=2)

        # Right side - visualization
        viz_frame = ttk.Frame(main_container)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create matplotlib figure for parameter trends
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Parameter Value")
        self.ax.set_title("Real-time Parameter Estimates")
        self.ax.grid(True, alpha=0.3)

        # Initialize data storage for plotting
        self.max_points = 100  # Keep last 100 data points
        self.time_data: deque[float] = deque(maxlen=self.max_points)
        self.theta0_data: deque[float] = deque(maxlen=self.max_points)
        self.pi_i_data: deque[float] = deque(maxlen=self.max_points)
        self.beta_data: deque[float] = deque(maxlen=self.max_points)

        # Create plot lines
        (self.theta0_line,) = self.ax.plot([], [], "b-", label="θ₀", linewidth=2)
        (self.pi_i_line,) = self.ax.plot([], [], "g-", label="Πᵢ", linewidth=2)
        (self.beta_line,) = self.ax.plot([], [], "r-", label="β", linewidth=2)
        self.ax.legend(loc="upper right")

        # Set initial axis limits
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-0.5, 2.0)

        # Embed matplotlib figure in tkinter
        self.canvas: Any = FigureCanvasTkAgg(self.fig, viz_frame)  # type: ignore[no-untyped-call]
        self.canvas.draw()  # type: ignore[no-untyped-call]
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # type: ignore[no-untyped-call]

        # Start time for x-axis
        self.start_time = time.time()

        logger.info("RealTimeParameterEstimateUpdater initialized")

    def update_theta0(self, mean: float, std: float, ci_lower: float, ci_upper: float) -> None:
        """Update θ₀ estimate.

        Args:
            mean: Mean estimate
            std: Standard deviation
            ci_lower: Lower 95% credible interval
            ci_upper: Upper 95% credible interval
        """
        text = f"{mean:.3f} ± {std:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
        self.theta0_label.config(text=text)

        # Update visualization data
        current_time = time.time() - self.start_time
        self.time_data.append(current_time)
        self.theta0_data.append(mean)
        self._update_plot()

    def update_pi_i(self, mean: float, std: float, ci_lower: float, ci_upper: float) -> None:
        """Update Πᵢ estimate.

        Args:
            mean: Mean estimate
            std: Standard deviation
            ci_lower: Lower 95% credible interval
            ci_upper: Upper 95% credible interval
        """
        text = f"{mean:.3f} ± {std:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
        self.pi_i_label.config(text=text)

        # Update visualization data
        current_time = time.time() - self.start_time
        if len(self.time_data) == 0 or current_time > self.time_data[-1]:
            self.time_data.append(current_time)
        self.pi_i_data.append(mean)
        self._update_plot()

    def update_beta(self, mean: float, std: float, ci_lower: float, ci_upper: float) -> None:
        """Update β estimate.

        Args:
            mean: Mean estimate
            std: Standard deviation
            ci_lower: Lower 95% credible interval
            ci_upper: Upper 95% credible interval
        """
        text = f"{mean:.3f} ± {std:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
        self.beta_label.config(text=text)

        # Update visualization data
        current_time = time.time() - self.start_time
        if len(self.time_data) == 0 or current_time > self.time_data[-1]:
            self.time_data.append(current_time)
        self.beta_data.append(mean)
        self._update_plot()

    def _update_plot(self) -> None:
        """Update the matplotlib plot with current data."""
        if len(self.time_data) > 0:
            # Ensure all data arrays have the same length
            min_length = min(len(self.theta0_data), len(self.pi_i_data), len(self.beta_data))
            time_array = list(self.time_data)[-min_length:]
            theta0_array = list(self.theta0_data)[-min_length:]
            pi_i_array = list(self.pi_i_data)[-min_length:]
            beta_array = list(self.beta_data)[-min_length:]

            # Update line data
            self.theta0_line.set_data(time_array, theta0_array)
            self.pi_i_line.set_data(time_array, pi_i_array)
            self.beta_line.set_data(time_array, beta_array)

            # Adjust x-axis limits to show sliding window
            if time_array[-1] > 10:
                self.ax.set_xlim(time_array[-1] - 10, time_array[-1])
            else:
                self.ax.set_xlim(0, 10)

            # Adjust y-axis limits based on data range
            all_data = theta0_array + pi_i_array + beta_array
            if all_data:
                y_min, y_max = min(all_data), max(all_data)
                margin = (y_max - y_min) * 0.1
                self.ax.set_ylim(y_min - margin, y_max + margin)

            # Redraw canvas
            self.canvas.draw()

    def update_convergence(self, converged: bool, r_hat: float) -> None:
        """Update convergence status.

        Args:
            converged: Whether chains have converged
            r_hat: Gelman-Rubin statistic
        """
        if converged:
            text = f"Converged (R̂={r_hat:.3f})"
            color = "green"
        else:
            text = f"Not converged (R̂={r_hat:.3f})"
            color = "orange"

        self.convergence_label.config(text=text, foreground=color)

    def reset(self) -> None:
        """Reset parameter displays."""
        self.theta0_label.config(text="N/A")
        self.pi_i_label.config(text="N/A")
        self.beta_label.config(text="N/A")
        self.convergence_label.config(text="N/A", foreground="gray")


class QualityAlertSystem:
    """Automated data quality issue detection and notifications.

    Monitors all data streams and generates alerts for quality issues.
    """

    def __init__(self, parent_frame: tk.Widget):
        """Initialize quality alert system.

        Args:
            parent_frame: Parent tkinter frame
        """
        self.frame = ttk.LabelFrame(parent_frame, text="Quality Alerts", padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Alert display
        self.alert_text = tk.Text(self.frame, height=10, state=tk.DISABLED, wrap=tk.WORD)
        self.alert_text.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for different alert levels
        self.alert_text.tag_config("info", foreground="blue")
        self.alert_text.tag_config("warning", foreground="orange")
        self.alert_text.tag_config("error", foreground="red")

        self.alerts: List[Dict[str, Any]] = []

        logger.info("QualityAlertSystem initialized")

    def add_alert(self, level: str, message: str, source: str = "") -> None:
        """Add a quality alert.

        Args:
            level: Alert level ('info', 'warning', or 'error')
            message: Alert message
            source: Source of alert (e.g., 'EEG', 'Pupillometry')
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        alert = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "source": source,
        }

        self.alerts.append(alert)

        # Update display
        self.alert_text.config(state=tk.NORMAL)

        alert_line = f"[{timestamp}] "
        if source:
            alert_line += f"[{source}] "
        alert_line += f"{message}\n"

        self.alert_text.insert(tk.END, alert_line, level)
        self.alert_text.see(tk.END)
        self.alert_text.config(state=tk.DISABLED)

        if level == "warning":
            logger.warning(alert_line)
        else:
            logger.error(alert_line)

    def check_eeg_quality(self, quality_score: float, artifact_rate: float) -> None:
        """Check EEG quality and generate alerts if needed.

        Args:
            quality_score: EEG quality score (0-1)
            artifact_rate: Artifact rate (0-1)
        """
        if quality_score < 0.5:
            self.add_alert("error", f"Poor EEG quality: {quality_score:.2f}", "EEG")
        elif quality_score < 0.7:
            self.add_alert("warning", f"Moderate EEG quality: {quality_score:.2f}", "EEG")

        if artifact_rate > 0.5:
            self.add_alert("error", f"High artifact rate: {artifact_rate * 100:.1f}%", "EEG")
        elif artifact_rate > 0.3:
            self.add_alert("warning", f"Elevated artifact rate: {artifact_rate * 100:.1f}%", "EEG")

    def check_pupil_quality(self, data_loss: float, tracking_loss: int) -> None:
        """Check pupillometry quality and generate alerts if needed.

        Args:
            data_loss: Data loss proportion (0-1)
            tracking_loss: Number of tracking loss episodes
        """
        if data_loss > 0.4:
            self.add_alert("error", f"High pupil data loss: {data_loss * 100:.1f}%", "Pupillometry")
        elif data_loss > 0.2:
            self.add_alert(
                "warning",
                f"Moderate pupil data loss: {data_loss * 100:.1f}%",
                "Pupillometry",
            )

        if tracking_loss > 10:
            self.add_alert(
                "warning",
                f"Frequent tracking loss: {tracking_loss} episodes",
                "Pupillometry",
            )

    def check_cardiac_quality(self, quality_score: float, rpeak_confidence: float) -> None:
        """Check cardiac signal quality and generate alerts if needed.

        Args:
            quality_score: Cardiac quality score (0-1)
            rpeak_confidence: R-peak detection confidence (0-1)
        """
        if quality_score < 0.6:
            self.add_alert("error", f"Poor cardiac quality: {quality_score:.2f}", "Cardiac")
        elif quality_score < 0.8:
            self.add_alert("warning", f"Moderate cardiac quality: {quality_score:.2f}", "Cardiac")

        if rpeak_confidence < 0.7:
            self.add_alert("warning", f"Low R-peak confidence: {rpeak_confidence:.2f}", "Cardiac")

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts.clear()
        self.alert_text.config(state=tk.NORMAL)
        self.alert_text.delete(1.0, tk.END)
        self.alert_text.config(state=tk.DISABLED)

    def get_alert_summary(self) -> Dict[str, int]:
        """Get summary of alerts by level.

        Returns:
            Dictionary with counts by alert level
        """
        summary = {"info": 0, "warning": 0, "error": 0}

        for alert in self.alerts:
            level = alert["level"]
            if level in summary:
                summary[level] += 1

        return summary


class MultiModalMonitoringDashboard:
    """Main dashboard that combines all monitoring components."""

    def __init__(self, root: tk.Tk):
        """Initialize the multi-modal monitoring dashboard."""
        self.root = root
        self.root.title("APGI Multi-Modal Monitoring Dashboard")
        self.root.geometry("1200x800")

        # Add stop event for simulation control
        self._stop_event = threading.Event()

        # Initialize data streamer
        self.streamer = get_streamer()

        # Create main notebook for different monitors
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create individual monitors with proper frame containers
        eeg_frame = tk.Frame(self.notebook)
        self.eeg_monitor = LiveEEGMonitor(eeg_frame)

        pupil_frame = tk.Frame(self.notebook)
        self.pupil_monitor = PupillometryMonitor(pupil_frame)

        cardiac_frame = tk.Frame(self.notebook)
        self.cardiac_monitor = CardiacMonitor(cardiac_frame)

        params_frame = tk.Frame(self.notebook)
        self.param_monitor = RealTimeParameterEstimateUpdater(params_frame)

        # Create alert system
        alert_frame = tk.Frame(self.notebook)
        self.alert_system = QualityAlertSystem(alert_frame)

        # Add tabs to notebook
        self.notebook.add(eeg_frame, text="EEG Monitoring")
        self.notebook.add(pupil_frame, text="Pupillometry")
        self.notebook.add(cardiac_frame, text="Cardiac")
        self.notebook.add(params_frame, text="Parameters")
        self.notebook.add(alert_frame, text="Alerts")

        # Start real-time streaming
        self.start_streaming()

        logger.info("MultiModalMonitoringDashboard initialized")

    def start_streaming(self) -> None:
        """Start real-time data streaming."""
        if self.streamer.start_server():  # type: ignore[no-untyped-call]
            logger.info("Real-time data streaming started")
            # Start data reception thread
            self.streaming_thread = threading.Thread(target=self._receive_stream_data, daemon=True)
            self.streaming_thread.start()
        else:
            logger.error("Failed to start real-time streaming")

    def _receive_stream_data(self) -> None:
        """Receive and process streaming data via WebSocket client."""
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("WebSocket library not available, using simulated data")
            self._simulate_data_updates()
            return

        # Create new event loop for WebSocket client
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._websocket_client_loop())
        except Exception as e:
            logger.error(f"WebSocket client error: {e}")
            # Fallback to simulation
            self._simulate_data_updates()
        finally:
            loop.close()

    async def _websocket_client_loop(self) -> None:
        """WebSocket client loop to receive data."""
        uri = f"ws://{self.streamer.host}:{self.streamer.port}"

        try:
            async with websockets.connect(uri) as websocket:
                logger.info(f"Connected to WebSocket server at {uri}")

                # Send subscription request
                subscribe_msg = {
                    "type": "subscribe",
                    "streams": ["eeg", "pupil", "cardiac", "parameters"],
                }
                await websocket.send(json.dumps(subscribe_msg))

                # Receive and process data
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        self._process_websocket_data(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON received: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            raise

    def _process_websocket_data(self, data: Dict[str, Any]) -> None:
        """Process incoming WebSocket data and update monitors."""
        data_type = data.get("data_type")
        stream_data = data.get("data", {})

        if data_type == "eeg":
            self._update_eeg_from_stream(stream_data)
        elif data_type == "pupil":
            self._update_pupil_from_stream(stream_data)
        elif data_type == "cardiac":
            self._update_cardiac_from_stream(stream_data)
        elif data_type == "parameters":
            self._update_parameters_from_stream(stream_data)

    def _update_eeg_from_stream(self, data: Dict[str, Any]) -> None:
        """Update EEG monitor from streaming data."""
        quality_score = data.get("quality_score", 0.0)
        artifact_rate = data.get("artifact_rate", 0.0)
        bad_channels = data.get("bad_channels", [])
        p3b_amplitude = data.get("p3b_amplitude")
        hep_amplitude = data.get("hep_amplitude")

        # Update GUI in main thread
        self.root.after(0, self.eeg_monitor.update_signal_quality, quality_score, artifact_rate)
        self.root.after(0, self.eeg_monitor.update_bad_channels, bad_channels)

        if p3b_amplitude is not None:
            self.root.after(0, self.eeg_monitor.update_p3b, p3b_amplitude)

        if hep_amplitude is not None:
            self.root.after(0, self.eeg_monitor.update_hep, hep_amplitude)

        # Check for quality alerts
        self.root.after(0, self.alert_system.check_eeg_quality, quality_score, artifact_rate)

    def _update_pupil_from_stream(self, data: Dict[str, Any]) -> None:
        """Update pupillometry monitor from streaming data."""
        data_quality = data.get("data_quality", 0.0)
        data_loss = data.get("data_loss", 0.0)
        pupil_diameter = data.get("pupil_diameter", 0.0)
        blink_rate = data.get("blink_rate", 0.0)
        tracking_loss = data.get("tracking_loss", 0)

        # Update GUI in main thread
        self.root.after(0, self.pupil_monitor.update_quality, data_quality, data_loss)
        self.root.after(
            0,
            self.pupil_monitor.update_measurements,
            pupil_diameter,
            blink_rate,
            tracking_loss,
        )

        # Check for quality alerts
        self.root.after(0, self.alert_system.check_pupil_quality, data_loss, tracking_loss)

    def _update_cardiac_from_stream(self, data: Dict[str, Any]) -> None:
        """Update cardiac monitor from streaming data."""
        signal_quality = data.get("signal_quality", 0.0)
        rpeak_confidence = data.get("rpeak_confidence", 0.0)
        heart_rate = data.get("heart_rate", 0.0)
        hrv = data.get("hrv", 0.0)
        rr_interval = data.get("rr_interval", 0.0)

        # Update GUI in main thread
        self.root.after(0, self.cardiac_monitor.update_quality, signal_quality, rpeak_confidence)
        self.root.after(0, self.cardiac_monitor.update_measurements, heart_rate, hrv, rr_interval)

        # Check for quality alerts
        self.root.after(
            0, self.alert_system.check_cardiac_quality, signal_quality, rpeak_confidence
        )

    def _update_parameters_from_stream(self, data: Dict[str, Any]) -> None:
        """Update parameter estimates from streaming data."""
        param_name = data.get("parameter_name", "")
        mean = data.get("mean", 0.0)
        std = data.get("std", 0.0)
        ci_lower = data.get("ci_lower", 0.0)
        ci_upper = data.get("ci_upper", 0.0)
        converged = data.get("converged", False)
        r_hat = data.get("r_hat")

        # Update GUI in main thread
        if param_name == "theta0":
            self.root.after(0, self.param_monitor.update_theta0, mean, std, ci_lower, ci_upper)
        elif param_name == "pi_i":
            self.root.after(0, self.param_monitor.update_pi_i, mean, std, ci_lower, ci_upper)
        elif param_name == "beta":
            self.root.after(0, self.param_monitor.update_beta, mean, std, ci_lower, ci_upper)

        self.root.after(0, self.param_monitor.update_convergence, converged, r_hat or 1.0)

    def _simulate_data_updates(self) -> None:
        """Fallback simulation when WebSocket is not available."""
        logger.info("Using simulated data updates")
        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                # Generate simulated data using the streamer's methods
                eeg_data = self.streamer._generate_eeg_data(current_time)
                pupil_data = self.streamer._generate_pupil_data(current_time)
                cardiac_data = self.streamer._generate_cardiac_data(current_time)

                # Process the simulated data through the same methods as WebSocket data
                self._update_eeg_from_stream(asdict(eeg_data))
                self._update_pupil_from_stream(asdict(pupil_data))
                self._update_cardiac_from_stream(asdict(cardiac_data))

                # Occasionally update parameters
                if int(current_time) % 3 == 0:  # Every 3 seconds
                    param_data = self.streamer._generate_parameter_data(current_time)
                    self._update_parameters_from_stream(asdict(param_data))

                time.sleep(0.1)  # 10 Hz updates

            except Exception as e:
                logger.error(f"Simulation error: {e}")
                break

    def stop_streaming(self) -> None:
        """Stop real-time data streaming."""
        if hasattr(self, "streamer"):
            self.streamer.stop_server()  # type: ignore[no-untyped-call]
            logger.info("Real-time data streaming stopped")


if __name__ == "__main__":
    """Launch the monitoring dashboard as a standalone application."""
    root = tk.Tk()
    app = MultiModalMonitoringDashboard(root)
    root.mainloop()
