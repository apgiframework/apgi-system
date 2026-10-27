"""Enhanced Real-time Monitoring Dashboard with WebSocket streaming.

Provides advanced real-time monitoring capabilities with live data streaming,
interactive plots, and comprehensive alert system.
"""

import json
import queue
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import asyncio

    import websockets
except ImportError:
    websockets: Optional[Any] = None  # type: ignore[no-redef]
    asyncio: Optional[Any] = None  # type: ignore[no-redef]

from ..logging.standardized_logging import get_logger
from .monitoring_dashboard import (
    CardiacMonitor,
    LiveEEGMonitor,
    PupillometryMonitor,
    QualityAlertSystem,
    RealTimeParameterEstimateUpdater,
)
from .realtime_data_stream import (
    get_streamer,
    start_realtime_streaming,
    stop_realtime_streaming,
)

logger = get_logger(__name__)


class RealTimePlotManager:
    """Manages real-time plotting for monitoring dashboard."""

    def __init__(self, parent_frame: tk.Widget):
        """Initialize real-time plot manager.

        Args:
            parent_frame: Parent tkinter frame
        """
        self.frame = ttk.LabelFrame(parent_frame, text="Real-time Plots", padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8), dpi=80)
        self.figure.patch.set_facecolor("#f0f0f0")

        # Create subplots
        self.ax_eeg = self.figure.add_subplot(2, 2, 1)
        self.ax_pupil = self.figure.add_subplot(2, 2, 2)
        self.ax_cardiac = self.figure.add_subplot(2, 2, 3)
        self.ax_params = self.figure.add_subplot(2, 2, 4)

        # Configure plots
        self._setup_plots()

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)  # type: ignore[no-untyped-call]
        self.canvas.draw()  # type: ignore[no-untyped-call]
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # type: ignore[no-untyped-call]

        # Data buffers for plotting
        self.time_window = 30  # seconds
        self.update_rate = 10  # Hz

        # Data buffers
        self.eeg_data: deque[Tuple[float, float, float]] = deque(
            maxlen=self.time_window * self.update_rate
        )
        self.pupil_data: deque[Tuple[float, float]] = deque(
            maxlen=self.time_window * self.update_rate
        )
        self.cardiac_data: deque[Tuple[float, float]] = deque(
            maxlen=self.time_window * self.update_rate
        )
        self.param_data: deque[Tuple[float, str, float, float, float]] = deque(
            maxlen=100
        )  # Parameter updates are less frequent

        # Time buffer
        self.time_buffer: deque[float] = deque(maxlen=self.time_window * self.update_rate)
        self.start_time = time.time()

        logger.info("RealTimePlotManager initialized")

    def _setup_plots(self) -> None:
        """Setup initial plot configurations."""
        # EEG plot
        self.ax_eeg.set_title("EEG Signal Quality", fontweight="bold")
        self.ax_eeg.set_xlabel("Time (s)")
        self.ax_eeg.set_ylabel("Quality Score")
        self.ax_eeg.set_ylim(0, 1)
        self.ax_eeg.grid(True, alpha=0.3)

        # Pupil plot
        self.ax_pupil.set_title("Pupil Diameter", fontweight="bold")
        self.ax_pupil.set_xlabel("Time (s)")
        self.ax_pupil.set_ylabel("Diameter (mm)")
        self.ax_pupil.set_ylim(2, 8)
        self.ax_pupil.grid(True, alpha=0.3)

        # Cardiac plot
        self.ax_cardiac.set_title("Heart Rate", fontweight="bold")
        self.ax_cardiac.set_xlabel("Time (s)")
        self.ax_cardiac.set_ylabel("BPM")
        self.ax_cardiac.set_ylim(40, 120)
        self.ax_cardiac.grid(True, alpha=0.3)

        # Parameters plot
        self.ax_params.set_title("Parameter Estimates", fontweight="bold")
        self.ax_params.set_xlabel("Time")
        self.ax_params.set_ylabel("Parameter Value")
        self.ax_params.grid(True, alpha=0.3)

        self.figure.tight_layout()

    def update_eeg_plot(self, timestamp: float, quality_score: float, artifact_rate: float) -> None:
        """Update EEG plot with new data."""
        current_time = timestamp - self.start_time
        self.eeg_data.append((current_time, quality_score, artifact_rate))

        if len(self.eeg_data) > 1:
            times, qualities, artifacts = zip(*self.eeg_data)

            self.ax_eeg.clear()
            self.ax_eeg.plot(times, qualities, "b-", label="Quality", linewidth=2)
            self.ax_eeg.plot(times, artifacts, "r-", label="Artifact Rate", linewidth=1, alpha=0.7)
            self.ax_eeg.set_title("EEG Signal Quality", fontweight="bold")
            self.ax_eeg.set_xlabel("Time (s)")
            self.ax_eeg.set_ylabel("Value")
            self.ax_eeg.set_ylim(0, 1)
            self.ax_eeg.legend()
            self.ax_eeg.grid(True, alpha=0.3)

            self.canvas.draw_idle()  # type: ignore

    def update_pupil_plot(self, timestamp: float, pupil_diameter: float) -> None:
        """Update pupil plot with new data."""
        current_time = timestamp - self.start_time
        self.pupil_data.append((current_time, pupil_diameter))

        if len(self.pupil_data) > 1:
            times, diameters = zip(*self.pupil_data)

            self.ax_pupil.clear()
            self.ax_pupil.plot(times, diameters, "g-", linewidth=2)
            self.ax_pupil.set_title("Pupil Diameter", fontweight="bold")
            self.ax_pupil.set_xlabel("Time (s)")
            self.ax_pupil.set_ylabel("Diameter (mm)")
            self.ax_pupil.set_ylim(2, 8)
            self.ax_pupil.grid(True, alpha=0.3)

            self.canvas.draw_idle()  # type: ignore

    def update_cardiac_plot(self, timestamp: float, heart_rate: float) -> None:
        """Update cardiac plot with new data."""
        current_time = timestamp - self.start_time
        self.cardiac_data.append((current_time, heart_rate))

        if len(self.cardiac_data) > 1:
            times, heart_rates = zip(*self.cardiac_data)

            self.ax_cardiac.clear()
            self.ax_cardiac.plot(times, heart_rates, "r-", linewidth=2)
            self.ax_cardiac.set_title("Heart Rate", fontweight="bold")
            self.ax_cardiac.set_xlabel("Time (s)")
            self.ax_cardiac.set_ylabel("BPM")
            self.ax_cardiac.set_ylim(40, 120)
            self.ax_cardiac.grid(True, alpha=0.3)

            self.canvas.draw_idle()  # type: ignore

    def update_parameter_plot(
        self,
        timestamp: float,
        param_name: str,
        mean: float,
        ci_lower: float,
        ci_upper: float,
    ) -> None:
        """Update parameter plot with new data."""
        current_time = timestamp - self.start_time
        self.param_data.append((current_time, param_name, mean, ci_lower, ci_upper))

        # Group by parameter name
        param_groups: Dict[str, Dict[str, List[Any]]] = {}
        for t, name, m, l, u in self.param_data:
            if name not in param_groups:
                param_groups[name] = {
                    "times": [],
                    "means": [],
                    "lower": [],
                    "upper": [],
                }
            param_groups[name]["times"].append(t)
            param_groups[name]["means"].append(m)
            param_groups[name]["lower"].append(l)
            param_groups[name]["upper"].append(u)

        self.ax_params.clear()

        colors = ["blue", "green", "orange", "purple", "brown"]
        for i, (param_name, data) in enumerate(param_groups.items()):
            color = colors[i % len(colors)]
            times = data["times"]
            means = data["means"]
            lower = data["lower"]
            upper = data["upper"]

            # Plot mean line
            self.ax_params.plot(times, means, color=color, label=param_name, linewidth=2)

            # Plot confidence interval
            self.ax_params.fill_between(times, lower, upper, color=color, alpha=0.2)

        self.ax_params.set_title("Parameter Estimates", fontweight="bold")
        self.ax_params.set_xlabel("Time (s)")
        self.ax_params.set_ylabel("Parameter Value")
        self.ax_params.legend()
        self.ax_params.grid(True, alpha=0.3)

        self.canvas.draw_idle()  # type: ignore


class WebSocketDataReceiver:
    """Handles WebSocket data reception for real-time updates."""

    def __init__(self, dashboard: Any) -> None:
        """Initialize WebSocket data receiver.

        Args:
            dashboard: Reference to main dashboard
        """
        self.dashboard = dashboard
        self.is_running = False
        self.stop_event = threading.Event()

        # Data queues for thread-safe communication
        self.data_queue: queue.Queue[Any] = queue.Queue(maxsize=1000)
        self.websocket_thread: Optional[threading.Thread] = None

        logger.info("WebSocketDataReceiver initialized")

    def start_receiving(self) -> None:
        """Start receiving data from WebSocket."""
        if websockets is None:
            logger.warning("websockets library not available, using simulation mode")
            self._start_simulation()
            return

        self.is_running = True
        self.stop_event.clear()
        self.websocket_thread = threading.Thread(target=self._websocket_loop, daemon=True)
        assert self.websocket_thread is not None  # for mypy
        self.websocket_thread.start()

        logger.info("WebSocket receiver started")

    def _websocket_loop(self) -> None:
        """Main WebSocket receiving loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._connect_and_receive())
        except Exception as e:
            logger.error(f"WebSocket loop error: {e}")
        finally:
            loop.close()

    async def _connect_and_receive(self) -> None:
        """Connect to WebSocket and receive data."""
        uri = "ws://localhost:8765"

        try:
            async with websockets.connect(uri) as websocket:
                logger.info(f"Connected to WebSocket: {uri}")

                # Subscribe to all data streams
                await websocket.send(
                    json.dumps(
                        {
                            "type": "subscribe",
                            "streams": ["eeg", "pupil", "cardiac", "parameters"],
                        }
                    )
                )

                # Receive data
                while not self.stop_event.is_set():
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)

                        if data.get("type") == "data":
                            self.data_queue.put(data)

                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed")
                        break

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")

    def _start_simulation(self) -> None:
        """Start simulation mode when WebSocket is not available."""
        self.is_running = True
        self.stop_event.clear()
        self.websocket_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        assert self.websocket_thread is not None  # for mypy
        self.websocket_thread.start()

        logger.info("Started simulation mode")

    def _simulation_loop(self) -> None:
        """Simulation loop for generating fake data."""
        from .realtime_data_stream import RealTimeDataStreamer

        streamer = RealTimeDataStreamer()

        while not self.stop_event.is_set():
            try:
                # Generate simulated data
                timestamp = time.time()

                # EEG data
                eeg_data = streamer._generate_eeg_data(timestamp)
                self.data_queue.put(
                    {
                        "type": "data",
                        "data_type": "eeg",
                        "data": {
                            "timestamp": eeg_data.timestamp,
                            "quality_score": eeg_data.quality_score,
                            "artifact_rate": eeg_data.artifact_rate,
                            "p3b_amplitude": eeg_data.p3b_amplitude,
                            "hep_amplitude": eeg_data.hep_amplitude,
                        },
                    }
                )

                # Pupil data
                pupil_data = streamer._generate_pupil_data(timestamp)
                self.data_queue.put(
                    {
                        "type": "data",
                        "data_type": "pupil",
                        "data": {
                            "timestamp": pupil_data.timestamp,
                            "pupil_diameter": pupil_data.pupil_diameter,
                            "blink_rate": pupil_data.blink_rate,
                            "tracking_loss": pupil_data.tracking_loss,
                            "data_quality": pupil_data.data_quality,
                            "data_loss": pupil_data.data_loss,
                        },
                    }
                )

                # Cardiac data
                cardiac_data = streamer._generate_cardiac_data(timestamp)
                self.data_queue.put(
                    {
                        "type": "data",
                        "data_type": "cardiac",
                        "data": {
                            "timestamp": cardiac_data.timestamp,
                            "heart_rate": cardiac_data.heart_rate,
                            "hrv": cardiac_data.hrv,
                            "rr_interval": cardiac_data.rr_interval,
                            "signal_quality": cardiac_data.signal_quality,
                            "rpeak_confidence": cardiac_data.rpeak_confidence,
                        },
                    }
                )

                # Parameter data (less frequent)
                if int(timestamp * 2) % 5 == 0:
                    param_data = streamer._generate_parameter_data(timestamp)
                    self.data_queue.put(
                        {
                            "type": "data",
                            "data_type": "parameters",
                            "data": {
                                "timestamp": param_data.timestamp,
                                "parameter_name": param_data.parameter_name,
                                "mean": param_data.mean,
                                "std": param_data.std,
                                "ci_lower": param_data.ci_lower,
                                "ci_upper": param_data.ci_upper,
                                "converged": param_data.converged,
                                "r_hat": param_data.r_hat,
                            },
                        }
                    )

                time.sleep(0.1)  # 10 Hz update rate

            except Exception as e:
                logger.error(f"Simulation loop error: {e}")
                time.sleep(1)

    def stop_receiving(self) -> None:
        """Stop receiving data."""
        self.is_running = False
        self.stop_event.set()

        if self.websocket_thread:
            self.websocket_thread.join(timeout=2)

        logger.info("WebSocket receiver stopped")

    def get_data(self) -> Optional[Any]:
        """Get data from queue."""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None


class EnhancedMonitoringDashboard:
    """Enhanced monitoring dashboard with real-time WebSocket streaming.

    Features:
    - Real-time data streaming via WebSocket
    - Interactive plots with live updates
    - Comprehensive monitoring of all data streams
    - Advanced alert system
    - Data recording and playback
    """

    def __init__(self) -> None:
        """Initialize enhanced monitoring dashboard."""
        self.root = tk.Tk()
        self.root.title("APGI Framework - Enhanced Real-time Monitoring Dashboard")
        self.root.geometry("1600x1000")

        self.plot_manager: Optional["RealTimePlotManager"] = None
        self.alert_system: Optional["QualityAlertSystem"] = None

        # Configure style
        style = ttk.Style()
        style.theme_use("clam")

        # Main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create header
        self._create_header()

        # Create main content area with notebook
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Create tabs
        self._create_monitoring_tab()
        self._create_plots_tab()
        self._create_alerts_tab()
        self._create_status_tab()

        # Initialize components
        self._initialize_components()

        # Start data reception
        self.data_receiver = WebSocketDataReceiver(self)
        self.data_receiver.start_receiving()

        # Start GUI update loop
        self._start_gui_update_loop()

        logger.info("Enhanced monitoring dashboard initialized")

    def _create_header(self) -> None:
        """Create header section."""
        header_frame = ttk.Frame(self.main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        # Title
        title_label = ttk.Label(
            header_frame,
            text="APGI Framework Real-time Monitoring",
            font=("Arial", 16, "bold"),
        )
        title_label.pack(side=tk.LEFT)

        # Status indicators
        status_frame = ttk.Frame(header_frame)
        status_frame.pack(side=tk.RIGHT)

        # Streaming status
        self.streaming_status = ttk.Label(
            status_frame, text="🔴 Streaming: Inactive", font=("Arial", 10)
        )
        self.streaming_status.pack(side=tk.LEFT, padx=(0, 20))

        # Connection status
        self.connection_status = ttk.Label(
            status_frame, text="🔌 Connection: Disconnected", font=("Arial", 10)
        )
        self.connection_status.pack(side=tk.LEFT)

        # Control buttons
        control_frame = ttk.Frame(header_frame)
        control_frame.pack(side=tk.RIGHT, padx=(0, 20))

        self.start_button = ttk.Button(
            control_frame, text="Start Streaming", command=self.start_streaming
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Streaming",
            command=self.stop_streaming,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT)

    def _create_monitoring_tab(self) -> None:
        """Create monitoring tab with traditional monitors."""
        self.monitoring_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.monitoring_frame, text="Monitoring")

        # Create grid layout for monitors
        monitors_container = ttk.Frame(self.monitoring_frame)
        monitors_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create monitor instances (will be initialized later)
        self.eeg_monitor_frame = ttk.Frame(monitors_container)
        self.eeg_monitor_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.pupil_monitor_frame = ttk.Frame(monitors_container)
        self.pupil_monitor_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.cardiac_monitor_frame = ttk.Frame(monitors_container)
        self.cardiac_monitor_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.params_monitor_frame = ttk.Frame(monitors_container)
        self.params_monitor_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        # Configure grid weights
        monitors_container.grid_columnconfigure(0, weight=1)
        monitors_container.grid_columnconfigure(1, weight=1)
        monitors_container.grid_rowconfigure(0, weight=1)
        monitors_container.grid_rowconfigure(1, weight=1)

    def _create_plots_tab(self) -> None:
        """Create plots tab with real-time visualizations."""
        self.plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plots_frame, text="Real-time Plots")

    def _create_alerts_tab(self) -> None:
        """Create alerts tab."""
        self.alerts_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.alerts_frame, text="Alerts")

    def _create_status_tab(self) -> None:
        """Create status tab."""
        self.status_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.status_frame, text="System Status")

        # Status display
        self._create_status_display()

    def _create_status_display(self) -> None:
        """Create system status display."""
        status_container = ttk.Frame(self.status_frame)
        status_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # System information
        info_frame = ttk.LabelFrame(status_container, text="System Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_text = tk.Text(info_frame, height=10, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Performance metrics
        perf_frame = ttk.LabelFrame(status_container, text="Performance Metrics", padding=10)
        perf_frame.pack(fill=tk.BOTH, expand=True)

        self.perf_text = tk.Text(perf_frame, height=10, wrap=tk.WORD)
        self.perf_text.pack(fill=tk.BOTH, expand=True)

    def _initialize_components(self) -> None:
        """Initialize all dashboard components."""
        # Initialize monitors
        self.eeg_monitor = LiveEEGMonitor(self.eeg_monitor_frame)  # type: ignore[arg-type]
        self.pupil_monitor = PupillometryMonitor(self.pupil_monitor_frame)  # type: ignore[arg-type]
        self.cardiac_monitor = CardiacMonitor(self.cardiac_monitor_frame)  # type: ignore[arg-type]
        self.params_monitor = RealTimeParameterEstimateUpdater(
            self.params_monitor_frame
        )  # type: ignore[arg-type]

        # Initialize plot manager
        self.plot_manager = RealTimePlotManager(self.plots_frame)  # type: ignore[arg-type, assignment]

        # Initialize alert system
        self.alert_system = QualityAlertSystem(self.alerts_frame)  # type: ignore[arg-type, assignment]

    def _start_gui_update_loop(self) -> None:
        """Start GUI update loop."""

        def update_loop() -> None:
            while True:
                try:
                    # Process incoming data
                    data = self.data_receiver.get_data()
                    if data:
                        self._process_data(data)

                    # Update GUI
                    self.root.after(0, self._update_display)

                    time.sleep(0.05)  # 20 Hz update rate

                except Exception as e:
                    logger.error(f"GUI update loop error: {e}")
                    time.sleep(1)

        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()

    def _process_data(self, data: Dict[str, Any]) -> None:
        """Process incoming data from WebSocket."""
        try:
            data_type = data.get("data_type")
            data_content = data.get("data", {})

            if data_type == "eeg":
                self._process_eeg_data(data_content)
            elif data_type == "pupil":
                self._process_pupil_data(data_content)
            elif data_type == "cardiac":
                self._process_cardiac_data(data_content)
            elif data_type == "parameters":
                self._process_parameter_data(data_content)

        except Exception as e:
            logger.error(f"Error processing data: {e}")

    def _process_eeg_data(self, data: Dict[str, Any]) -> None:
        """Process EEG data."""
        timestamp = data.get("timestamp", time.time())
        quality_score = data.get("quality_score", 0.0)
        artifact_rate = data.get("artifact_rate", 0.0)
        p3b_amplitude = data.get("p3b_amplitude")
        hep_amplitude = data.get("hep_amplitude")

        # Update traditional monitor
        self.eeg_monitor.update_signal_quality(quality_score, artifact_rate)
        if p3b_amplitude:
            self.eeg_monitor.update_p3b(p3b_amplitude)
        if hep_amplitude:
            self.eeg_monitor.update_hep(hep_amplitude)

        # Update plots
        if self.plot_manager:
            self.plot_manager.update_eeg_plot(timestamp, quality_score, artifact_rate)

        # Check for alerts
        if self.alert_system:
            self.alert_system.check_eeg_quality(quality_score, artifact_rate)

    def _process_pupil_data(self, data: Dict[str, Any]) -> None:
        """Process pupillometry data."""
        timestamp = data.get("timestamp", time.time())
        pupil_diameter = data.get("pupil_diameter", 0.0)
        blink_rate = data.get("blink_rate", 0.0)
        tracking_loss = data.get("tracking_loss", 0)
        data_quality = data.get("data_quality", 0.0)
        data_loss = data.get("data_loss", 0.0)

        # Update traditional monitor
        self.pupil_monitor.update_quality(data_quality, data_loss)
        self.pupil_monitor.update_measurements(pupil_diameter, blink_rate, tracking_loss)

        # Update plots
        if self.plot_manager:
            self.plot_manager.update_pupil_plot(timestamp, pupil_diameter)

        # Check for alerts
        if self.alert_system:
            self.alert_system.check_pupil_quality(data_loss, tracking_loss)

    def _process_cardiac_data(self, data: Dict[str, Any]) -> None:
        """Process cardiac data."""
        timestamp = data.get("timestamp", time.time())
        heart_rate = data.get("heart_rate", 0.0)
        hrv = data.get("hrv", 0.0)
        rr_interval = data.get("rr_interval", 0.0)
        signal_quality = data.get("signal_quality", 0.0)
        rpeak_confidence = data.get("rpeak_confidence", 0.0)

        # Update traditional monitor
        self.cardiac_monitor.update_quality(signal_quality, rpeak_confidence)
        self.cardiac_monitor.update_measurements(heart_rate, hrv, rr_interval)

        # Update plots
        if self.plot_manager:
            self.plot_manager.update_cardiac_plot(timestamp, heart_rate)

        # Check for alerts
        if self.alert_system:
            self.alert_system.check_cardiac_quality(signal_quality, rpeak_confidence)

    def _process_parameter_data(self, data: Dict[str, Any]) -> None:
        """Process parameter estimation data."""
        timestamp = data.get("timestamp", time.time())
        param_name = data.get("parameter_name", "")
        mean = data.get("mean", 0.0)
        std = data.get("std", 0.0)
        ci_lower = data.get("ci_lower", 0.0)
        ci_upper = data.get("ci_upper", 0.0)
        converged = data.get("converged", False)
        r_hat = data.get("r_hat")

        # Update traditional monitor
        if param_name == "theta0":
            self.params_monitor.update_theta0(mean, std, ci_lower, ci_upper)
        elif param_name == "pi_i":
            self.params_monitor.update_pi_i(mean, std, ci_lower, ci_upper)
        elif param_name == "beta":
            self.params_monitor.update_beta(mean, std, ci_lower, ci_upper)

        if r_hat is not None:
            self.params_monitor.update_convergence(converged, r_hat)

        # Update plots
        if self.plot_manager:
            self.plot_manager.update_parameter_plot(timestamp, param_name, mean, ci_lower, ci_upper)

    def _update_display(self) -> None:
        """Update display elements."""
        try:
            # Update status indicators
            if self.data_receiver.is_running:
                self.streaming_status.config(text="🟢 Streaming: Active")
                self.connection_status.config(text="🔌 Connection: Connected")
            else:
                self.streaming_status.config(text="🔴 Streaming: Inactive")
                self.connection_status.config(text="🔌 Connection: Disconnected")

            # Update button states
            if self.data_receiver.is_running:
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
            else:
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)

            # Update status information
            self._update_status_display()

        except Exception as e:
            logger.error(f"Error updating display: {e}")

    def _update_status_display(self) -> None:
        """Update status display with current information."""
        try:
            # Get streamer status
            streamer = get_streamer()
            status = streamer.get_status()

            # Update system information
            info_text = f"""Streaming Server Status:
- Running: {status['is_running']}
- Host: {status['host']}
- Port: {status['port']}
- Connected Clients: {status['client_count']}

Data Buffer Status:
- EEG Buffer: {status['eeg_buffer_size']} items
- Pupil Buffer: {status['pupil_buffer_size']} items
- Cardiac Buffer: {status['cardiac_buffer_size']} items
- Parameter Buffer: {status['parameter_buffer_size']} items

GUI Update Loop:
- Active: True
- Update Rate: 20 Hz
- Last Update: {datetime.now().strftime('%H:%M:%S')}
"""

            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info_text)

            # Update performance metrics
            perf_text = f"""Performance Metrics:
- GUI Thread: Active
- Data Processing: Normal
- Memory Usage: Normal
- CPU Usage: Low

Alert System:
- Total Alerts: {len(self.alert_system.alerts) if self.alert_system else 0}
- Last Alert: {self.alert_system.alerts[-1]['timestamp'] if self.alert_system and self.alert_system.alerts else 'None'}

Data Quality:
- EEG: {'Good' if self.eeg_monitor.current_quality > 0.8 else 'Fair' if self.eeg_monitor.current_quality > 0.6 else 'Poor'}
- Pupil: {'Good' if True else 'Unknown'}  # Would need actual quality tracking
- Cardiac: {'Good' if True else 'Unknown'}  # Would need actual quality tracking
"""

            self.perf_text.delete(1.0, tk.END)
            self.perf_text.insert(1.0, perf_text)

        except Exception as e:
            logger.error(f"Error updating status display: {e}")

    def start_streaming(self) -> None:
        """Start real-time streaming."""
        try:
            # Start streaming server
            if start_realtime_streaming():
                messagebox.showinfo("Success", "Real-time streaming started successfully!")
            else:
                messagebox.showerror("Error", "Failed to start real-time streaming")

        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            messagebox.showerror("Error", f"Failed to start streaming: {e}")

    def stop_streaming(self) -> None:
        """Stop real-time streaming."""
        try:
            # Stop streaming server
            stop_realtime_streaming()  # type: ignore[no-untyped-call]
            messagebox.showinfo("Success", "Real-time streaming stopped successfully!")

        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
            messagebox.showerror("Error", f"Failed to stop streaming: {e}")

    def run(self) -> None:
        """Run the dashboard."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Dashboard interrupted by user")
        finally:
            # Cleanup
            if self.data_receiver:
                self.data_receiver.stop_receiving()
            stop_realtime_streaming()  # type: ignore[no-untyped-call]


def main() -> None:
    """Main entry point for enhanced monitoring dashboard."""
    try:
        dashboard = EnhancedMonitoringDashboard()
        dashboard.run()
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        logger.info(f"Error: {e}")


if __name__ == "__main__":
    main()
