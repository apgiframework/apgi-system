#!/usr/bin/env python3
"""
APGI System - Comprehensive GUI
===============================

A modern, production-ready graphical interface for the Allostatic Precision-Gated Ignition
(APGI) cognitive system.

Features:
- Real-time cognitive state monitoring
- Oscillatory spectrum analysis
- Biofeedback integration
- Energy management
- Performance metrics
- Data export capabilities

Author: APGI Development Team
Version: 2.0.0
"""

import json
import queue
import sys
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable, Dict, Optional, Tuple, Union

# Configure matplotlib backend before any matplotlib imports
import matplotlib
import numpy as np

matplotlib.use("TkAgg")  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# Try to import APGI system
try:
    sys.path.insert(0, str(Path(__file__).parent))  # noqa: E402
    from apgi_framework.core.free_energy import FreeEnergyCalculator  # noqa: F401, E402
    from apgi_framework.system import APGISystem  # noqa: E402

    HAS_APGI = True
except ImportError as e:
    HAS_APGI = False
    print(f"Warning: APGI system not available: {e}")

# Try to import theme manager
try:
    from apgi_gui.components.core import ToolTip  # noqa: F401, E402
    from apgi_gui.theme_manager import ThemeManager  # noqa: F401, E402

    HAS_GUI_COMPONENTS = True
except ImportError:
    HAS_GUI_COMPONENTS = False
    print("Warning: GUI components not available")


class GUIConfig:
    """Centralized configuration for APGI GUI application."""

    # Window settings
    WINDOW_TITLE = "APGI System - Cognitive AI Interface"
    DEFAULT_WINDOW_SIZE = "1400x900"
    MIN_WINDOW_SIZE = (1200, 700)

    # Timing and performance
    UPDATE_INTERVAL_MS = 100
    HISTORY_UPDATE_INTERVAL_MS = 1000
    PLOT_UPDATE_INTERVAL_MS = 500
    DEBOUNCE_DELAY_MS = 300
    INIT_TIMEOUT_MS = 60000

    # UI Layout
    PAD_X = 5
    PAD_Y = 5
    BORDER_WIDTH = 2
    SLIDER_LENGTH = 200

    # Fonts
    FONT_FAMILY = "Arial"
    FONT_SIZE_SMALL = 9
    FONT_SIZE_MEDIUM = 10
    FONT_SIZE_LARGE = 12

    # Physiological ranges
    HR_RANGE = (40, 200)
    HRV_RANGE = (10, 200)
    RESP_RANGE = (8, 40)
    EDA_RANGE = (0.5, 20.0)

    # Model configuration
    MODEL_INPUT_MIN = 32
    MODEL_INPUT_MAX = 1024
    MODEL_HIDDEN_MIN = 64
    MODEL_HIDDEN_MAX = 2048

    # History limits
    MAX_STATE_HISTORY = 1000
    MAX_ENERGY_HISTORY = 500
    MAX_QUERY_HISTORY = 200
    MAX_MEMORY_MB = 1024

    # Theme colors
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
        },
        "dark": {
            "bg": "#1e1e1e",
            "fg": "#ffffff",
            "accent": "#4a7abc",
            "canvas_bg": "#2d2d2d",
            "figure_bg": "#2d2d2d",
            "grid_color": "#555555",
            "success": "#4CAF50",
            "error": "#F44336",
            "warning": "#FF9800",
            "info": "#ffffff",
            "state_colors": {
                "confused": "#FF6666",
                "focused": "#6666FF",
                "alert": "#FFFF00",
                "relaxed": "#66FF66",
                "working": "#FF66FF",
            },
        },
    }

    @classmethod
    def get_font(cls, size: str = "medium") -> Tuple[str, int]:
        """Get font configuration."""
        size_map = {
            "small": cls.FONT_SIZE_SMALL,
            "medium": cls.FONT_SIZE_MEDIUM,
            "large": cls.FONT_SIZE_LARGE,
        }
        return (cls.FONT_FAMILY, size_map.get(size, cls.FONT_SIZE_MEDIUM))


class StatusManager:
    """Centralized status bar management with history."""

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

    def __init__(
        self, status_label: Union[tk.Label, ttk.Label], assistant_status: Union[tk.Label, ttk.Label]
    ) -> None:
        self.status_label = status_label
        self.assistant_status = assistant_status
        self.status_history: deque[Dict[str, Any]] = deque(maxlen=100)
        self.last_update_time = 0.0
        self.min_update_interval = 0.5

    def set_status(
        self, message: str, status_type: str = "ready", assistant_state: Optional[str] = None
    ) -> None:
        """Set status with icon and history."""
        import time

        current_time = time.time()
        if current_time - self.last_update_time < self.min_update_interval:
            return

        icon = self.ICONS.get(status_type, "")
        full_message = f"{icon} {message}" if icon else message

        self.status_history.append(
            {
                "message": message,
                "status_type": status_type,
                "assistant_state": assistant_state,
                "timestamp": time.time(),
                "display_message": full_message,
            }
        )

        if self.status_label and self.status_label.winfo_exists():
            self.status_label.config(text=full_message)

        if assistant_state and self.assistant_status and self.assistant_status.winfo_exists():
            self.assistant_status.config(text=assistant_state)

        self.last_update_time = current_time


class APGISystemGUI:
    """Main GUI application for APGI System."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(GUIConfig.WINDOW_TITLE)
        self.root.geometry(GUIConfig.DEFAULT_WINDOW_SIZE)
        self.root.minsize(*GUIConfig.MIN_WINDOW_SIZE)

        # Setup logging
        from apgi_framework.logging.centralized_logging import get_structured_logger

        self.logger = get_structured_logger("APGIGUI")
        self.logger.info("Initializing APGI System GUI")

        # State variables
        self.system: Optional[APGISystem] = None
        self.is_running = False
        self.is_initialized = False
        self.update_count = 0
        self.current_theme = "normal"

        # Data storage
        self.state_history: deque[Any] = deque(maxlen=GUIConfig.MAX_STATE_HISTORY)
        self.energy_history: deque[Any] = deque(maxlen=GUIConfig.MAX_ENERGY_HISTORY)
        self.query_history: deque[Any] = deque(maxlen=GUIConfig.MAX_QUERY_HISTORY)

        # Threading
        self.system_lock = threading.Lock()
        self.data_queue: queue.Queue[Any] = queue.Queue()
        self.update_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Create UI
        self.create_menu()
        self.create_main_layout()
        self.create_status_bar()

        # Initialize system
        if HAS_APGI:
            self.initialize_system()

        # Start update loop
        self.start_update_loop()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_menu(self) -> None:
        """Create application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Session", command=self.new_session)
        file_menu.add_command(label="Export Data", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Refresh", command=self.refresh_display)
        view_menu.add_separator()
        view_menu.add_radiobutton(
            label="Light Theme",
            variable=tk.StringVar(value="normal"),
            command=lambda: self.set_theme("normal"),
        )
        view_menu.add_radiobutton(
            label="Dark Theme",
            variable=tk.StringVar(value="dark"),
            command=lambda: self.set_theme("dark"),
        )

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Calibrate", command=self.calibrate_system)
        tools_menu.add_command(label="Reset System", command=self.reset_system)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)

    def create_main_layout(self) -> None:
        """Create the main application layout."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Create tabs
        self.dashboard_frame = ttk.Frame(self.notebook)
        self.cognitive_frame = ttk.Frame(self.notebook)
        self.oscillatory_frame = ttk.Frame(self.notebook)
        self.biofeedback_frame = ttk.Frame(self.notebook)
        self.energy_frame = ttk.Frame(self.notebook)
        self.performance_frame = ttk.Frame(self.notebook)
        self.settings_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.dashboard_frame, text="Dashboard")
        self.notebook.add(self.cognitive_frame, text="Cognitive State")
        self.notebook.add(self.oscillatory_frame, text="Oscillatory Analysis")
        self.notebook.add(self.biofeedback_frame, text="Biofeedback")
        self.notebook.add(self.energy_frame, text="Energy Management")
        self.notebook.add(self.performance_frame, text="Performance")
        self.notebook.add(self.settings_frame, text="Settings")

        # Create tab contents
        self.create_dashboard_tab()
        self.create_cognitive_tab()
        self.create_oscillatory_tab()
        self.create_biofeedback_tab()
        self.create_energy_tab()
        self.create_performance_tab()
        self.create_settings_tab()

    def create_dashboard_tab(self) -> None:
        """Create the main dashboard tab."""
        # Left panel - System status
        left_frame = ttk.LabelFrame(self.dashboard_frame, text="System Status", padding=10)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.status_text = scrolledtext.ScrolledText(left_frame, height=15, width=50, wrap="word")
        self.status_text.pack(fill="both", expand=True)

        # Right panel - Quick controls
        right_frame = ttk.LabelFrame(self.dashboard_frame, text="Quick Controls", padding=10)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        ttk.Button(right_frame, text="Start System", command=self.start_system).pack(
            fill="x", pady=2
        )
        ttk.Button(right_frame, text="Stop System", command=self.stop_system).pack(fill="x", pady=2)
        ttk.Button(right_frame, text="Step", command=self.step_system).pack(fill="x", pady=2)
        ttk.Button(right_frame, text="Reset", command=self.reset_system).pack(fill="x", pady=2)

        # Bottom panel - Quick metrics
        bottom_frame = ttk.LabelFrame(self.dashboard_frame, text="Quick Metrics", padding=10)
        bottom_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        self.metrics_labels = {}
        metrics = [
            ("Current State", "current_state"),
            ("Free Energy", "free_energy"),
            ("Ignition Count", "ignition_count"),
            ("Energy Reserves", "energy_reserves"),
        ]

        for i, (label, key) in enumerate(metrics):
            ttk.Label(bottom_frame, text=f"{label}:").grid(row=i, column=0, sticky="w")
            self.metrics_labels[key] = ttk.Label(
                bottom_frame, text="--", font=("Arial", 10, "bold")
            )
            self.metrics_labels[key].grid(row=i, column=1, sticky="w", padx=10)

        # Configure grid
        self.dashboard_frame.grid_columnconfigure(0, weight=2)
        self.dashboard_frame.grid_columnconfigure(1, weight=1)
        self.dashboard_frame.grid_rowconfigure(0, weight=1)
        self.dashboard_frame.grid_rowconfigure(1, weight=1)

    def create_cognitive_tab(self) -> None:
        """Create cognitive state monitoring tab."""
        # State visualization
        state_frame = ttk.LabelFrame(self.cognitive_frame, text="Cognitive State", padding=10)
        state_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.state_canvas = tk.Canvas(state_frame, height=200, bg="white")
        self.state_canvas.pack(fill="both", expand=True)

        # State metrics
        metrics_frame = ttk.LabelFrame(self.cognitive_frame, text="State Metrics", padding=10)
        metrics_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.cognitive_metrics_text = scrolledtext.ScrolledText(
            metrics_frame, height=10, width=60, wrap="word"
        )
        self.cognitive_metrics_text.pack(fill="both", expand=True)

        # Configure grid
        self.cognitive_frame.grid_columnconfigure(0, weight=1)
        self.cognitive_frame.grid_rowconfigure(0, weight=2)
        self.cognitive_frame.grid_rowconfigure(1, weight=1)

    def create_oscillatory_tab(self) -> None:
        """Create oscillatory analysis tab."""
        # Power spectrum
        spectrum_frame = ttk.LabelFrame(self.oscillatory_frame, text="Power Spectrum", padding=10)
        spectrum_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.spectrum_fig = Figure(figsize=(8, 4), dpi=100)
        self.spectrum_ax = self.spectrum_fig.add_subplot(111)
        self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_fig, spectrum_frame)  # type: ignore[no-untyped-call]
        self.spectrum_canvas.get_tk_widget().pack(fill="both", expand=True)  # type: ignore[no-untyped-call]

        # Frequency distribution
        freq_frame = ttk.LabelFrame(
            self.oscillatory_frame, text="Frequency Distribution", padding=10
        )
        freq_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.freq_fig = Figure(figsize=(8, 4), dpi=100)
        self.freq_ax = self.freq_fig.add_subplot(111)
        self.freq_canvas = FigureCanvasTkAgg(self.freq_fig, freq_frame)  # type: ignore[no-untyped-call]
        self.freq_canvas.get_tk_widget().pack(fill="both", expand=True)  # type: ignore[no-untyped-call]

        # Configure grid
        self.oscillatory_frame.grid_columnconfigure(0, weight=1)
        self.oscillatory_frame.grid_rowconfigure(0, weight=1)
        self.oscillatory_frame.grid_rowconfigure(1, weight=1)

    def create_biofeedback_tab(self) -> None:
        """Create biofeedback monitoring tab."""
        # Physiological parameters
        physio_frame = ttk.LabelFrame(
            self.biofeedback_frame, text="Physiological Parameters", padding=10
        )
        physio_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.physio_vars = {}
        physio_params = [
            ("Heart Rate (bpm)", "hr", GUIConfig.HR_RANGE, 70),
            ("HRV (ms)", "hrv", GUIConfig.HRV_RANGE, 50),
            ("Respiration (bpm)", "resp", GUIConfig.RESP_RANGE, 16),
            ("Skin Conductance (µS)", "eda", GUIConfig.EDA_RANGE, 5.0),
        ]

        for i, (label, key, (min_val, max_val), default) in enumerate(physio_params):
            ttk.Label(physio_frame, text=label).grid(row=i, column=0, sticky="w", pady=2)

            var = tk.DoubleVar(value=default)
            self.physio_vars[key] = var

            slider = ttk.Scale(
                physio_frame,
                from_=min_val,
                to=max_val,
                variable=var,
                orient="horizontal",
                length=200,
            )
            slider.grid(row=i, column=1, padx=5)

            value_label = ttk.Label(physio_frame, text=str(default))
            value_label.grid(row=i, column=2)

            # Update label on slider change
            def make_command(lbl: Union[tk.Label, ttk.Label], k: str) -> Callable[[str], None]:
                return lambda v: self._update_physio_label(v, lbl, k)

            slider.config(command=make_command(value_label, key))

        # Configure grid
        self.biofeedback_frame.grid_columnconfigure(0, weight=1)
        self.biofeedback_frame.grid_rowconfigure(0, weight=1)

    def _update_physio_label(self, value: str, label: Union[tk.Label, ttk.Label], key: str) -> None:
        """Update physiological parameter label."""
        if key == "eda":
            label.config(text=f"{float(value):.1f}")
        else:
            label.config(text=f"{int(float(value))}")

    def create_energy_tab(self) -> None:
        """Create energy management tab."""
        # Energy usage plot
        energy_frame = ttk.LabelFrame(self.energy_frame, text="Energy Usage History", padding=10)
        energy_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.energy_fig = Figure(figsize=(8, 4), dpi=100)
        self.energy_ax = self.energy_fig.add_subplot(111)
        self.energy_canvas = FigureCanvasTkAgg(self.energy_fig, energy_frame)  # type: ignore[no-untyped-call]
        self.energy_canvas.get_tk_widget().pack(fill="both", expand=True)  # type: ignore[no-untyped-call]

        # Battery status
        battery_frame = ttk.LabelFrame(self.energy_frame, text="Battery Status", padding=10)
        battery_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.battery_canvas = tk.Canvas(battery_frame, height=50, bg="white")
        self.battery_canvas.pack(fill="both", expand=True)

        # Configure grid
        self.energy_frame.grid_columnconfigure(0, weight=1)
        self.energy_frame.grid_rowconfigure(0, weight=2)
        self.energy_frame.grid_rowconfigure(1, weight=1)

    def create_performance_tab(self) -> None:
        """Create performance metrics tab."""
        # Performance metrics
        metrics_frame = ttk.LabelFrame(
            self.performance_frame, text="Performance Metrics", padding=10
        )
        metrics_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.performance_text = scrolledtext.ScrolledText(
            metrics_frame, height=15, width=80, wrap="word"
        )
        self.performance_text.pack(fill="both", expand=True)

        # Configure grid
        self.performance_frame.grid_columnconfigure(0, weight=1)
        self.performance_frame.grid_rowconfigure(0, weight=1)

    def create_settings_tab(self) -> None:
        """Create settings tab."""
        # Model settings
        model_frame = ttk.LabelFrame(self.settings_frame, text="Model Configuration", padding=10)
        model_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Label(model_frame, text="Input Dimension:").grid(row=0, column=0, sticky="w")
        self.input_dim_var = tk.IntVar(value=128)
        ttk.Spinbox(model_frame, from_=32, to=1024, textvariable=self.input_dim_var).grid(
            row=0, column=1
        )

        ttk.Label(model_frame, text="Hidden Dimension:").grid(row=1, column=0, sticky="w")
        self.hidden_dim_var = tk.IntVar(value=256)
        ttk.Spinbox(model_frame, from_=64, to=2048, textvariable=self.hidden_dim_var).grid(
            row=1, column=1
        )

        # System settings
        system_frame = ttk.LabelFrame(self.settings_frame, text="System Settings", padding=10)
        system_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Label(system_frame, text="Update Interval (ms):").grid(row=0, column=0, sticky="w")
        self.update_interval_var = tk.IntVar(value=100)
        ttk.Spinbox(system_frame, from_=50, to=1000, textvariable=self.update_interval_var).grid(
            row=0, column=1
        )

        ttk.Label(system_frame, text="History Size:").grid(row=1, column=0, sticky="w")
        self.history_size_var = tk.IntVar(value=1000)
        ttk.Spinbox(system_frame, from_=100, to=10000, textvariable=self.history_size_var).grid(
            row=1, column=1
        )

        # Apply button
        ttk.Button(self.settings_frame, text="Apply Settings", command=self.apply_settings).grid(
            row=2, column=0, pady=10
        )

        # Configure grid
        self.settings_frame.grid_columnconfigure(0, weight=1)
        self.settings_frame.grid_rowconfigure(0, weight=1)
        self.settings_frame.grid_rowconfigure(1, weight=1)

    def create_status_bar(self) -> None:
        """Create status bar at bottom."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side="bottom", fill="x", padx=5, pady=2)

        self.status_label = ttk.Label(status_frame, text="Ready", relief="sunken", anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True)

        self.assistant_status = ttk.Label(status_frame, text="Idle", relief="sunken", width=20)
        self.assistant_status.pack(side="right", padx=5)

        self.status_mgr = StatusManager(self.status_label, self.assistant_status)

    def initialize_system(self) -> None:
        """Initialize the APGI system."""
        if not HAS_APGI:
            self.logger.warning("APGI system not available")
            return

        try:
            self.system = APGISystem()
            self.is_initialized = True
            # Update settings UI to reflect actual config
            input_dim = self.system.config["hierarchy"]["level_configs"][0]["nodes"]
            self.input_dim_var.set(input_dim)
            self.logger.info(f"APGI system initialized successfully (input_dim={input_dim})")
            self.status_mgr.set_status("System initialized", "success")
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            self.status_mgr.set_status(f"Initialization failed: {e}", "error")

    def start_system(self) -> None:
        """Start the system simulation."""
        if not self.is_initialized:
            self.status_mgr.set_status("System not initialized", "error")
            return

        self.is_running = True
        self.stop_event.clear()
        self.status_mgr.set_status("System running", "processing")

        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

    def stop_system(self) -> None:
        """Stop the system simulation."""
        self.is_running = False
        self.stop_event.set()
        self.status_mgr.set_status("System stopped", "idle")

    def step_system(self) -> None:
        """Execute a single system step."""
        if not self.is_initialized or not self.system:
            return

        try:
            with self.system_lock:
                # Get input dimension from system config (sensory level nodes)
                input_dim = self.system.config["hierarchy"]["level_configs"][0]["nodes"]

                # Generate extero_input with correct dimension
                t = self.system.time / 1000.0  # Convert to seconds
                base = np.sin(2 * np.pi * t / 5.0) * np.ones(input_dim)
                noise = np.random.randn(input_dim) * 0.2
                extero_input = (base + noise).astype(np.float64)

                self.system.step(extero_input)
                self._update_data()
        except Exception as e:
            self.logger.error(f"Error during step: {e}")

    def reset_system(self) -> None:
        """Reset the system to initial state."""
        if not self.is_initialized or not self.system:
            return

        try:
            with self.system_lock:
                self.system.reset()
                self.state_history.clear()
                self.energy_history.clear()
                self.query_history.clear()
            self.status_mgr.set_status("System reset", "info")
        except Exception as e:
            self.logger.error(f"Error during reset: {e}")

    def calibrate_system(self) -> None:
        """Calibrate the system."""
        self.status_mgr.set_status("Calibration not implemented", "warning")

    def _get_input_data(self) -> Dict[str, float]:
        """Get current input data from physiological parameters."""
        return {
            "hr": self.physio_vars["hr"].get(),
            "hrv": self.physio_vars["hrv"].get(),
            "resp": self.physio_vars["resp"].get(),
            "eda": self.physio_vars["eda"].get(),
        }

    def _update_loop(self) -> None:
        """Background update loop."""
        while not self.stop_event.is_set():
            try:
                self.step_system()
                time.sleep(self.update_interval_var.get() / 1000.0)
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")

    def _update_data(self) -> None:
        """Update data from system."""
        if not self.system:
            return

        try:
            state = self.system.get_state()
            self.state_history.append(state)

            # Update energy history
            if "thermodynamic" in state:
                thermo = state["thermodynamic"]
                self.energy_history.append(
                    {
                        "timestamp": time.time(),
                        "reserves": thermo.get("metabolic_reserves", 0),
                    }
                )

            # Queue UI update
            self.data_queue.put(state)
        except Exception as e:
            self.logger.error(f"Error updating data: {e}")

    def start_update_loop(self) -> None:
        """Start the UI update loop."""
        self._process_updates()

    def _process_updates(self) -> None:
        """Process pending updates in the UI thread."""
        try:
            while not self.data_queue.empty():
                state = self.data_queue.get_nowait()
                self._update_ui(state)
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error processing updates: {e}")

        # Schedule next update
        self.root.after(GUIConfig.UPDATE_INTERVAL_MS, self._process_updates)

    def _update_ui(self, state: Dict[str, Any]) -> None:
        """Update UI with new state data."""
        try:
            # Update dashboard metrics
            if "ignition" in state:
                ignition = state["ignition"]
                self.metrics_labels["ignition_count"].config(
                    text=str(ignition.get("threshold_stats", {}).get("ignition_count", 0))
                )

            if "thermodynamic" in state:
                thermo = state["thermodynamic"]
                self.metrics_labels["energy_reserves"].config(
                    text=f"{thermo.get('metabolic_reserves', 0):.2f}"
                )

            if "cognitive_state" in state:
                cog = state["cognitive_state"]
                self.metrics_labels["current_state"].config(
                    text=cog.get("primary_state", "unknown")
                )

            if "core" in state:
                core = state["core"]
                self.metrics_labels["free_energy"].config(text=f"{core.get('free_energy', 0):.2f}")

            # Update status text periodically
            self.update_count += 1
            if self.update_count % 10 == 0:
                self._update_status_text(state)

        except Exception as e:
            self.logger.error(f"Error updating UI: {e}")

    def _update_status_text(self, state: Dict[str, Any]) -> None:
        """Update the status text display."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            status_str = f"[{timestamp}] System Update\n"
            status_str += (
                f"  State: {state.get('cognitive_state', {}).get('primary_state', 'unknown')}\n"
            )
            status_str += f"  Free Energy: {state.get('core', {}).get('free_energy', 0):.2f}\n"
            status_str += f"  Ignitions: {state.get('ignition', {}).get('threshold_stats', {}).get('ignition_count', 0)}\n"
            status_str += "\n"

            self.status_text.insert("end", status_str)
            self.status_text.see("end")

            # Limit text size
            if float(self.status_text.index("end-1c").split(".")[0]) > 100:
                self.status_text.delete("1.0", "50.0")
        except Exception as e:
            self.logger.error(f"Error updating status text: {e}")

    def refresh_display(self) -> None:
        """Refresh all displays."""
        self.status_mgr.set_status("Display refreshed", "info")

    def new_session(self) -> None:
        """Create a new session."""
        self.reset_system()
        self.status_mgr.set_status("New session started", "info")

    def export_data(self) -> None:
        """Export system data to file."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                data = {
                    "state_history": list(self.state_history),
                    "energy_history": list(self.energy_history),
                    "export_time": datetime.now().isoformat(),
                }
                with open(filename, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                self.status_mgr.set_status(f"Data exported to {filename}", "success")
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            messagebox.showerror("Export Error", str(e))

    def apply_settings(self) -> None:
        """Apply current settings."""
        try:
            # Update history sizes
            new_size = self.history_size_var.get()
            self.state_history = deque(self.state_history, maxlen=new_size)
            self.energy_history = deque(self.energy_history, maxlen=new_size)

            self.status_mgr.set_status("Settings applied", "success")
        except Exception as e:
            self.logger.error(f"Error applying settings: {e}")
            messagebox.showerror("Settings Error", str(e))

    def set_theme(self, theme_name: str) -> None:
        """Set the application theme."""
        self.current_theme = theme_name
        theme = GUIConfig.THEMES.get(theme_name, GUIConfig.THEMES["normal"])

        # Apply theme colors
        self.root.configure(bg=theme["bg"])  # type: ignore[call-overload]

        self.status_mgr.set_status(f"Theme changed to {theme_name}", "info")

    def show_help(self) -> None:
        """Show help documentation."""
        help_text = """
APGI System Help
================

Getting Started:
1. Initialize the system using the dashboard controls
2. Adjust physiological parameters in the Biofeedback tab
3. Monitor cognitive state changes in real-time
4. Export data when needed

Tabs:
- Dashboard: Main system controls and quick metrics
- Cognitive State: Detailed cognitive monitoring
- Oscillatory Analysis: Frequency spectrum visualization
- Biofeedback: Physiological parameter controls
- Energy Management: Resource usage monitoring
- Performance: System performance metrics
- Settings: Configuration options

Keyboard Shortcuts:
- Ctrl+N: New session
- Ctrl+E: Export data
- F5: Refresh display

For more information, see the documentation.
        """
        messagebox.showinfo("Help", help_text)

    def show_about(self) -> None:
        """Show about dialog."""
        about_text = """
APGI System v2.0.0

A sophisticated neural interface with adaptive processing
and energy-aware computation.

Features:
• Real-time cognitive state monitoring
• Physiological parameter tracking
• Energy-efficient processing
• Oscillatory pattern analysis
• Biofeedback integration

© 2024 APGI Project
        """
        messagebox.showinfo("About", about_text)

    def on_close(self) -> None:
        """Handle application close."""
        if self.is_running:
            self.stop_system()

        if messagebox.askyesno("Exit", "Exit APGI System?"):
            self.logger.info("Application shutting down")
            self.root.destroy()


def main() -> None:
    """Main entry point."""
    root = tk.Tk()

    # Set DPI awareness for better scaling on high-DPI displays
    try:
        from ctypes import windll  # type: ignore[attr-defined]

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass  # Not on Windows or not supported

    APGISystemGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
