"""
APGI System - Comprehensive GUI Application

Full-featured Tkinter GUI for the Allostatic Precision-Gated Ignition System.
Provides complete control and visualization of all subsystems.
"""

import csv
import json
import platform
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock, RLock
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, Optional, cast

import numpy as np
import psutil
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from numpy.typing import NDArray

from apgi_framework.config_validator import ConfigValidationError, validate_config_file
from apgi_framework.logging.centralized_logging import get_structured_logger
from apgi_framework.platform_utils import get_data_dir, get_resource_path
from apgi_framework.system import APGISystem

# Import theme manager
from apgi_gui.theme_manager import get_theme_manager

# Initialize structured logger
logger = get_structured_logger(__name__)


class APGIGui:
    """Main GUI application for APGI System."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize GUI application."""
        self.root = root
        self.root.title("APGI Consciousness Modeling Framework")

        # Configuration
        self.config_file = Path.home() / ".apgi_gui_config.json"

        # Set responsive window size based on screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Use 80% of screen dimensions, with minimum and maximum bounds
        window_width = min(max(1200, int(screen_width * 0.8)), 1920)
        window_height = min(max(800, int(screen_height * 0.8)), 1080)

        # Center window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(1000, 700)  # Set minimum window size

        # Register WM_DELETE_WINDOW protocol handler to properly shut down
        self.root.protocol("WM_DELETE_WINDOW", self._confirm_exit)

        # Force window to be visible and bring to front
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after_idle(lambda: self.root.attributes("-topmost", False))

        # Platform-specific modifier key for shortcuts
        self.modifier_key = "Command" if platform.system() == "Darwin" else "Ctrl"

        # Initialize theme manager
        self.theme_manager = get_theme_manager()

        # Load saved theme from config
        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                    if "theme" in config:
                        self.theme_manager.set_theme(config["theme"])
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

        self._apply_theme_to_root()

        # System state
        self.apgi_simulation: Optional[APGISystem] = None
        self.is_running: bool = False
        self.is_paused: bool = False
        self.simulation_thread: Optional[threading.Thread] = None
        self.config_path: Path = get_resource_path("config/default.yaml")
        self.auto_save_timer: Optional[str] = None  # Timer ID for auto-save

        # Data buffers for plotting with configurable sizes
        self.buffer_size: int = 1000  # Default buffer size
        self.min_buffer_size: int = 100  # Minimum buffer size
        self.max_buffer_size: int = 10000  # Maximum buffer size

        # Load buffer size from config if available
        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                    if "buffer_size" in config:
                        loaded_size = config["buffer_size"]
                        if self.min_buffer_size <= loaded_size <= self.max_buffer_size:
                            self.buffer_size = loaded_size
        except Exception as e:
            logger.warning(f"Failed to load buffer size from config: {e}")

        # Initialize buffers with default size
        self._initialize_buffers(self.buffer_size)

        # Performance tracking
        self.last_frame_time = time.time()
        self.fps_buffer: deque[float] = deque(maxlen=100)
        self.memory_buffer: deque[float] = deque(maxlen=1000)

        # Thread safety locks
        self.data_lock: threading.RLock = threading.RLock()  # Reentrant lock for data buffers
        self.log_lock: threading.Lock = threading.Lock()  # Simple lock for log data
        self.system_lock: threading.RLock = threading.RLock()  # Reentrant lock for system access

        # Logging with bounded buffer to prevent memory leaks
        self.log_buffer_size: int = 10000  # Maximum number of log entries to keep
        self.log_data: deque[Dict[str, Any]] = deque(maxlen=self.log_buffer_size)
        self.auto_save: bool = False

        # Initialize custom input attributes
        self.custom_input_steps_remaining = 0
        self.custom_input_params: Dict[str, object] | None = None

        # Initialize view variables as tk.BooleanVar with master
        # Moved to _convert_to_tkinter_variables for proper timing
        self.view_vars: Dict[str, tk.BooleanVar] = {}  # Placeholder
        self.auto_save = False

        # Build GUI
        try:
            self._create_menu_bar()
        except Exception as e:
            self._log_event(f"❌ Error creating menu bar: {e}")
            import traceback

            traceback.print_exc()
            return

        try:
            self._create_main_layout()
        except Exception as e:
            self._log_event(f"❌ Error creating main layout: {e}")
            import traceback

            traceback.print_exc()
            return

        try:
            self._create_status_bar()
        except Exception as e:
            self._log_event(f"❌ Error creating status bar: {e}")
            import traceback

            traceback.print_exc()
            return

        # Initialize system directly (no login required)
        self._initialize_system()

        # Start update loop
        self._update_displays()

        # Schedule tkinter variable conversion for after main loop starts
        # Use a longer delay to ensure the window is fully ready
        self.root.after(200, self._convert_to_tkinter_variables)

        # Apply theme to all components after GUI is fully created
        self.root.after(300, self._apply_theme_to_root)

        # Setup keyboard shortcuts
        self._setup_keyboard_shortcuts()

    def _setup_logging(self) -> Any:
        """Setup application logging."""
        from apgi_framework.logging.centralized_logging import get_structured_logger

        return get_structured_logger("APGIGUI")

    def _setup_keyboard_shortcuts(self) -> None:
        """Setup keyboard shortcuts for the application."""
        # Zoom shortcuts (handle both =/+ and -)
        self.root.bind("<Control-equal>", lambda e: self._zoom_in())
        self.root.bind("<Control-plus>", lambda e: self._zoom_in())
        self.root.bind("<Control-minus>", lambda e: self._zoom_out())
        self.root.bind("<Control-underscore>", lambda e: self._zoom_out())

        # Simulation shortcuts
        self.root.bind("<F5>", lambda e: self._start_simulation())
        self.root.bind("<F6>", lambda e: self._pause_simulation())
        self.root.bind("<F7>", lambda e: self._stop_simulation())
        self.root.bind("<F8>", lambda e: self._reset_simulation())
        self.root.bind("<F1>", lambda e: self._show_help())

        # File operations (avoid conflicts with toolbar)
        self.root.bind("<Control-n>", lambda e: self._new_session())
        self.root.bind("<Control-o>", lambda e: self._load_config())
        self.root.bind("<Control-e>", lambda e: self._export_data())
        self.root.bind("<Control-q>", lambda e: self._exit_app())

        # View operations
        self.root.bind("<Control-p>", lambda e: self._toggle_parameter_panel())
        self.root.bind("<Control-l>", lambda e: self._toggle_log_panel())
        self.root.bind("<Control-0>", lambda e: self._zoom_fit())

        # Navigation
        self.root.bind("<Tab>", self._handle_tab_navigation)
        self.root.bind("<Shift-Tab>", self._handle_shift_tab_navigation)
        self.root.bind("<Control-Tab>", lambda e: self._cycle_notebook_tabs(1))
        self.root.bind("<Control-Shift-Tab>", lambda e: self._cycle_notebook_tabs(-1))

        # Ensure shortcuts work even when widgets have focus
        self.root.focus_set()

    def _apply_theme_to_root(self) -> None:
        """Apply current theme to root window."""
        theme_bg = self.theme_manager.get_theme_color("bg")

        try:
            self.root.configure(bg=theme_bg)
        except tk.TclError as e:
            logger.error(f"Error: {e}")  # Some themes may not support bg configuration

        # Apply to all widgets recursively
        self._apply_theme_to_child_widgets(self.root)

        # Apply to canvases if they exist
        if hasattr(self, "neural_canvas"):
            self.theme_manager.apply_theme_to_canvas(
                self.neural_canvas, self.theme_manager.current_theme
            )

        if hasattr(self, "intero_canvas"):
            self.theme_manager.apply_theme_to_canvas(
                self.intero_canvas, self.theme_manager.current_theme
            )

        if hasattr(self, "metrics_canvas"):
            self.theme_manager.apply_theme_to_canvas(
                self.metrics_canvas, self.theme_manager.current_theme
            )

        if hasattr(self, "self_canvas"):
            self.theme_manager.apply_theme_to_canvas(
                self.self_canvas, self.theme_manager.current_theme
            )

        if hasattr(self, "osc_canvas"):
            self.theme_manager.apply_theme_to_canvas(
                self.osc_canvas, self.theme_manager.current_theme
            )

        if hasattr(self, "state_canvas"):
            self.theme_manager.apply_theme_to_canvas(
                self.state_canvas, self.theme_manager.current_theme
            )

        # Force redraw of all components
        self.root.update_idletasks()

    def _apply_theme_to_child_widgets(self, parent: tk.Misc) -> None:
        """Recursively apply current theme to all child widgets."""
        for child in parent.winfo_children():
            # Skip matplotlib canvas widgets (handled separately)
            if "canvas" in str(child).lower() and hasattr(child, "get_tk_widget"):
                continue

            self.theme_manager.apply_theme_to_widget(child)
            if child.winfo_children():
                self._apply_theme_to_child_widgets(child)

    def _change_theme(self, theme_name: str) -> None:
        """Change the current theme."""
        if self.theme_manager.set_theme(theme_name):
            self._apply_theme_to_root()
            self._log_event(f"Theme changed to: {theme_name}")

            # Save theme to config
            try:
                config = {}
                if self.config_file.exists():
                    with open(self.config_file, "r") as f:
                        config = json.load(f)
                config["theme"] = theme_name
                with open(self.config_file, "w") as f:
                    json.dump(config, f)
            except Exception as e:
                self._log_event(f"Failed to save theme: {e}")
        else:
            messagebox.showerror("Theme Error", f"Unknown theme: {theme_name}")

    def _create_theme_menu(self) -> None:
        """Create theme selection menu."""
        # Add theme menu to view menu if it exists
        if hasattr(self, "menu_bar"):
            for i in range(self.menu_bar.index("end") or 0):
                if self.menu_bar.entrycget(i, "label") == "View":
                    view_menu = self.menu_bar.nametowidget(self.menu_bar.entrycget(i, "menu"))
                    if view_menu:
                        # Add theme submenu
                        theme_menu = tk.Menu(view_menu, tearoff=0)
                        view_menu.add_cascade(label="Theme", menu=theme_menu)

                        # Add theme options
                        from functools import partial

                        for theme_name in self.theme_manager.get_available_themes():
                            theme_menu.add_command(
                                label=theme_name.title().replace("_", " "),
                                command=partial(self._change_theme, theme_name),
                            )

                        # Add separator and high contrast toggle
                        theme_menu.add_separator()
                        theme_menu.add_command(
                            label="Toggle High Contrast", command=self._toggle_high_contrast
                        )
                    break

    def _toggle_high_contrast(self) -> None:
        """Toggle between normal and high contrast themes."""
        current_theme = self.theme_manager.current_theme
        if "high_contrast" in current_theme:
            # Switch to normal theme
            self._change_theme("normal")
        else:
            # Switch to high contrast
            self._change_theme("high_contrast")

    def _initialize_buffers(self, size: int) -> None:
        """Initialize data buffers with specified size."""
        self.time_buffer: deque[float] = deque(maxlen=size)
        self.data_buffers: Dict[str, deque[float]] = {
            "ignition": deque(maxlen=size),
            "free_energy": deque(maxlen=size),
            "extero_precision": deque(maxlen=size),
            "intero_precision": deque(maxlen=size),
            "metabolic_reserves": deque(maxlen=size),
            "allostatic_load": deque(maxlen=size),
            "heart_rate": deque(maxlen=size),
            "cortisol": deque(maxlen=size),
            "workspace_active": deque(maxlen=size),
            "gamma_power": deque(maxlen=size),
            "beta_power": deque(maxlen=size),
            "theta_power": deque(maxlen=size),
            "alpha_power": deque(maxlen=size),
            "somatic_markers": deque(maxlen=size),
            "minimal_self_coherence": deque(maxlen=size),
            "performance": deque(maxlen=size),
            "memory_mb": deque(maxlen=size),
            "delta_power": deque(maxlen=size),
        }
        self.buffer_size = size
        self._log_event(f"Buffer size set to {size} points")

    def _convert_to_tkinter_variables(self) -> None:
        """Convert Python variables to tkinter variables after GUI is created."""
        # Convert buffer size variable
        self.buffer_size_var = tk.IntVar(master=self.root, value=self.buffer_size)

        # Only initialize if not already initialized
        if not hasattr(self, "auto_save_var"):
            self.auto_save_var = tk.BooleanVar(master=self.root, value=self.auto_save)
        else:
            self.auto_save_var.set(self.auto_save)

        # Convert param_vars to tkinter DoubleVar
        if hasattr(self, "param_vars"):
            for key, value in self.param_vars.items():
                if not hasattr(value, "get"):
                    # Convert plain float/int to DoubleVar
                    self.param_vars[key] = tk.DoubleVar(master=self.root, value=float(value))

        self.view_vars = {
            "control_panel": tk.BooleanVar(master=self.root, value=True),
            "neural_activity": tk.BooleanVar(master=self.root, value=True),
            "interoception": tk.BooleanVar(master=self.root, value=True),
            "system_metrics": tk.BooleanVar(master=self.root, value=True),
        }
        # Assign variables to view checkbuttons
        for key, var in self.view_vars.items():
            if hasattr(self, "view_checkbuttons") and key in self.view_checkbuttons:
                # Note: add_checkbutton returns an index, not the checkbutton object
                # We need to configure the menu item by index
                menu = self._view_menu
                # Find the index of the checkbutton
                end_index = menu.index("end")
                if end_index is not None:
                    for i in range(int(end_index) + 1):
                        if (
                            menu.entrycget(i, "label")
                            == {
                                "control_panel": "Control Panel",
                                "neural_activity": "Neural Activity",
                                "interoception": "Interoception",
                                "system_metrics": "System Metrics",
                            }[key]
                        ):
                            menu.entryconfig(i, variable=var)
                            break

        # Now add the auto-save checkbutton to the file menu
        try:
            file_menu = cast(tk.Menu, self.menu_bar.winfo_children()[0])  # File menu is first
            # Find the index of the "Exit" command and insert before it
            exit_index = None
            last_index = file_menu.index("end")
            if last_index is not None:
                for i in range(int(last_index) + 1):
                    try:
                        label = file_menu.entrycget(i, "label")
                        if label == "Exit":
                            exit_index = i
                            break
                    except tk.TclError:
                        continue

            if exit_index is not None:
                # Insert auto-save checkbutton before Exit
                file_menu.insert_checkbutton(
                    exit_index,
                    label="Auto-save Data",
                    variable=self.auto_save_var,
                    command=self._toggle_auto_save,
                )
            else:
                # Fallback: append to end
                file_menu.add_checkbutton(
                    label="Auto-save Data",
                    variable=self.auto_save_var,
                    command=self._toggle_auto_save,
                )
        except Exception as e:
            self._log_event(f"Warning: Could not add auto-save checkbutton: {e}")

    def _configure_buffer_size(self) -> None:
        """Open dialog to configure buffer size."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Buffer Configuration")
        dialog.geometry("400x200")
        dialog.resizable(False, False)

        # Center dialog
        dialog.transient(self.root)
        dialog.grab_set()

        # Buffer size control
        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Data Buffer Size:", font=("Arial", 10, "bold")).pack(pady=5)

        buffer_frame = ttk.Frame(frame)
        buffer_frame.pack(pady=10)

        ttk.Label(buffer_frame, text="Size:").pack(side=tk.LEFT)

        buffer_spinbox = ttk.Spinbox(
            buffer_frame,
            from_=self.min_buffer_size,
            to=self.max_buffer_size,
            textvariable=self.buffer_size_var,
            width=10,
        )
        buffer_spinbox.pack(side=tk.LEFT, padx=5)

        ttk.Label(buffer_frame, text="points").pack(side=tk.LEFT)

        # Memory usage estimate
        memory_label = ttk.Label(frame, text="")
        memory_label.pack(pady=10)

        def update_memory_estimate(*args: Any) -> None:
            try:
                size = int(self.buffer_size_var.get())
                # Calculate memory usage accurately based on numpy float64 dtype
                num_buffers = len(self.data_buffers) + 1  # +1 for time buffer
                bytes_per_point = np.dtype(np.float64).itemsize  # 8 bytes for float64
                estimated_mb = (size * num_buffers * bytes_per_point) / (1024 * 1024)
                memory_label.config(text=f"Estimated memory usage: {estimated_mb:.1f} MB")
            except (ValueError, tk.TclError):
                memory_label.config(text="Invalid buffer size")

        self.buffer_size_var.trace_add("write", update_memory_estimate)
        update_memory_estimate()

        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=10)

        def apply_buffer_size() -> None:
            # Check if simulation is running
            if self.is_running:
                messagebox.showwarning(
                    "Simulation Running",
                    "Cannot change buffer size while simulation is running.\n\n"
                    "Please stop the simulation first.",
                )
                return

            try:
                new_size = int(self.buffer_size_var.get())
                if self.min_buffer_size <= new_size <= self.max_buffer_size:
                    with self.data_lock:
                        self._initialize_buffers(new_size)
                    self.buffer_size = new_size
                    # Save buffer size to config
                    try:
                        config = {}
                        if self.config_file.exists():
                            with open(self.config_file, "r") as f:
                                config = json.load(f)
                        config["buffer_size"] = new_size
                        with open(self.config_file, "w") as f:
                            json.dump(config, f)
                    except Exception as e:
                        self._log_event(f"Failed to save buffer size to config: {e}")
                    dialog.destroy()
                    self._log_event(f"Buffer size updated to {new_size} points")
                else:
                    messagebox.showerror(
                        "Invalid Size",
                        f"Buffer size must be between {self.min_buffer_size} and {self.max_buffer_size}",
                    )
            except ValueError:
                messagebox.showerror("Invalid Size", "Please enter a valid number")

        ttk.Button(button_frame, text="Apply", command=apply_buffer_size).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

        # Add to View menu
        view_menu = None
        last_index = self.menu_bar.index("end")
        if last_index is not None:
            for i in range(last_index + 1):
                if self.menu_bar.entrycget(i, "label") == "View":
                    view_menu = self.menu_bar.nametowidget(self.menu_bar.entrycget(i, "menu"))
                    break

        if view_menu:
            view_menu.add_separator()
            view_menu.add_command(
                label="Configure Buffer Size...", command=self._configure_buffer_size
            )

        # Thread safety locks
        self.data_lock = RLock()  # Reentrant lock for data buffers
        self.log_lock = Lock()  # Simple lock for log data
        self.system_lock = RLock()  # Reentrant lock for system access

        # Initialize panel frames to None to prevent AttributeErrors during early access
        self.param_frame: Optional[ttk.LabelFrame] = None
        self.log_frame: ttk.LabelFrame

        # Logging with bounded buffer to prevent memory leaks
        self.log_buffer_size = 10000  # Maximum number of log entries to keep
        self.log_data = deque(maxlen=self.log_buffer_size)
        self.auto_save = False

        # Build GUI
        try:
            self._create_menu_bar()
        except Exception as e:
            print(f"❌ Error creating menu bar: {e}")
            import traceback

            traceback.print_exc()
            return

        try:
            self._create_main_layout()
        except Exception as e:
            print(f"❌ Error creating main layout: {e}")
            import traceback

            traceback.print_exc()
            return

        try:
            self._create_status_bar()
        except Exception as e:
            print(f"❌ Error creating status bar: {e}")
            import traceback

            traceback.print_exc()
            return

        # Initialize system
        self._initialize_system()

        # Start update loop
        self._update_displays()

    def _create_menu_bar(self) -> None:
        """Create comprehensive menu bar."""
        self.menu_bar: tk.Menu = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # File Menu
        file_menu: tk.Menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="New Session", command=self._new_session, accelerator=f"{self.modifier_key}+N"
        )
        file_menu.add_command(
            label="Load Configuration...",
            command=self._load_config,
            accelerator=f"{self.modifier_key}+O",
        )
        file_menu.add_command(
            label="Save Configuration...",
            command=self._save_config,
            accelerator=f"{self.modifier_key}+S",
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Export Data...", command=self._export_data, accelerator=f"{self.modifier_key}+E"
        )
        file_menu.add_command(label="Export Plot...", command=self._export_plot)
        file_menu.add_separator()
        file_menu.add_separator()
        # Create auto-save variable if it doesn't exist yet
        if not hasattr(self, "auto_save_var"):
            self.auto_save_var = tk.BooleanVar(master=self.root, value=self.auto_save)

        file_menu.add_checkbutton(
            label="Auto-save Data",
            variable=self.auto_save_var,
            command=self._toggle_auto_save,
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Exit", command=self._confirm_exit, accelerator=f"{self.modifier_key}+Q"
        )

        # Edit Menu
        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="System Parameters...", command=self._edit_parameters)
        edit_menu.add_command(label="Precision Settings...", command=self._edit_precision)
        edit_menu.add_command(label="Ignition Threshold...", command=self._edit_threshold)
        edit_menu.add_separator()
        edit_menu.add_command(label="Reset to Defaults", command=self._reset_defaults)

        # Simulation Menu
        sim_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Simulation", menu=sim_menu)
        sim_menu.add_command(label="Start", command=self._start_simulation, accelerator="F5")
        sim_menu.add_command(label="Pause/Resume", command=self._pause_simulation, accelerator="F6")
        sim_menu.add_command(label="Stop", command=self._stop_simulation, accelerator="F7")
        sim_menu.add_command(label="Reset", command=self._reset_simulation, accelerator="F8")
        sim_menu.add_separator()
        sim_menu.add_command(label="Run Preset Task...", command=self._run_preset_task)

        # View Menu
        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="View", menu=view_menu)
        # Store view menu for later variable assignment
        self._view_menu = view_menu
        # Store checkbuttons for later variable assignment
        self.view_checkbuttons: Dict[str, Any] = {}
        view_menu.add_checkbutton(
            label="Control Panel",
            command=self._toggle_control_panel,
        )
        view_menu.add_checkbutton(
            label="Neural Activity",
            command=self._toggle_neural_activity,
        )
        view_menu.add_checkbutton(
            label="Interoception",
            command=self._toggle_interoception,
        )
        view_menu.add_checkbutton(
            label="System Metrics",
            command=self._toggle_system_metrics,
        )
        view_menu.add_separator()
        view_menu.add_command(
            label="Zoom In", accelerator=f"{self.modifier_key}++", command=self._zoom_in
        )
        view_menu.add_command(
            label="Zoom Out", accelerator=f"{self.modifier_key}+-", command=self._zoom_out
        )
        view_menu.add_command(
            label="Fit to Window", accelerator=f"{self.modifier_key}+0", command=self._zoom_fit
        )

        # Tools Menu
        tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Trigger Ignition Event", command=self._trigger_ignition)
        tools_menu.add_command(label="Induce Stressor", command=self._induce_stressor)
        tools_menu.add_command(label="Modulate Precision...", command=self._modulate_precision)
        tools_menu.add_separator()
        tools_menu.add_command(label="Inject Sensory Input...", command=self._inject_input)
        tools_menu.add_command(label="Set Body State...", command=self._set_body_state)
        tools_menu.add_separator()
        tools_menu.add_command(label="System Diagnostics", command=self._show_diagnostics)

        # Analysis Menu
        analysis_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Ignition Statistics", command=self._show_ignition_stats)
        analysis_menu.add_command(label="Energy Budget Report", command=self._show_energy_report)
        analysis_menu.add_command(label="Somatic Marker Analysis", command=self._analyze_markers)
        analysis_menu.add_command(label="Self-Model Coherence", command=self._analyze_coherence)
        analysis_menu.add_separator()
        analysis_menu.add_command(
            label="Statistical Analysis...", command=self._show_statistical_analysis
        )
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Generate Report...", command=self._generate_report)

        # Help Menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._show_docs)
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About APGI System", command=self._show_about)

        # Create theme menu
        self._create_theme_menu()

    def _confirm_exit(self) -> None:
        """Confirm exit with user if simulation is running."""
        if self.is_running:
            result = messagebox.askyesno(
                "Confirm Exit",
                "A simulation is currently running.\n\n"
                "Exiting will stop the simulation and unsaved data may be lost.\n\n"
                "Do you want to exit anyway?",
                icon="warning",
            )
            if not result:
                return  # User cancelled exit

        self._exit_app()

    def _exit_app(self) -> None:
        """Exit the application with proper cleanup."""
        try:
            # Stop simulation if running
            if self.is_running:
                self._stop_simulation()

            # Stop auto-save timer if active
            if hasattr(self, "auto_save_timer") and self.auto_save_timer is not None:
                self._stop_auto_save_timer()

            # Cancel any pending after callbacks
            if hasattr(self, "_update_after_id"):
                self.root.after_cancel(self._update_after_id)

            # Wait for simulation thread to finish if it exists
            if (
                hasattr(self, "simulation_thread")
                and self.simulation_thread
                and self.simulation_thread.is_alive()
            ):
                self.simulation_thread.join(timeout=2.0)  # Wait up to 2 seconds
                if self.simulation_thread.is_alive():
                    # Thread didn't finish, but continue with shutdown
                    logger.warning("Simulation thread did not finish gracefully")

            # Clean up matplotlib canvases to prevent warnings
            canvases = [
                getattr(self, attr, None)
                for attr in [
                    "neural_canvas",
                    "intero_canvas",
                    "metrics_canvas",
                    "self_canvas",
                    "osc_canvas",
                    "state_canvas",
                ]
            ]
            for canvas in canvases:
                if canvas:
                    try:
                        canvas.get_tk_widget().destroy()
                    except Exception as e:
                        logger.debug(f"Error destroying canvas: {e}")

            # Clean up auto-save files on exit
            self._cleanup_auto_save_files()

            # Clean up toplevel windows
            self._cleanup_toplevel_windows()

            # Log shutdown
            self._log_event("Application shutting down")

        except Exception as e:
            # Log error but continue with shutdown
            logger.error(f"Error during shutdown: {e}", exc_info=True)

        finally:
            # Force quit
            self.root.quit()

    def _cleanup_toplevel_windows(self) -> None:
        """Clean up and close all toplevel windows."""
        try:
            for child in self.root.winfo_children():
                if isinstance(child, tk.Toplevel):
                    try:
                        child.destroy()
                    except Exception as e:
                        logger.error(f"Error: {e}")
        except Exception as e:
            logger.error(f"Error cleaning up toplevel windows: {e}")

    def _create_main_layout(self) -> None:
        """Create main application layout."""
        # Create main container with paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Controls
        left_panel = ttk.Frame(main_paned, width=350)
        main_paned.add(left_panel, weight=0)

        # Right panel - Visualizations (tabbed)
        right_panel = ttk.Frame(main_paned)
        main_paned.add(right_panel, weight=1)
        self.right_panel = right_panel

        # Store references for toggling
        self.main_paned = main_paned
        self.left_panel = left_panel

        # Build left panel
        self._create_control_panel(left_panel)

        # Build right panel with tabs
        self._create_visualization_panel(right_panel)

    def _create_control_panel(self, parent: tk.Widget) -> None:
        """Create control panel."""
        # Simulation Controls
        control_frame = ttk.LabelFrame(parent, text="Simulation Control", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)

        self.start_btn = ttk.Button(
            btn_frame, text="▶ Start", command=self._start_simulation, width=10
        )
        self.start_btn.pack(side=tk.LEFT, padx=2)

        self.pause_btn = ttk.Button(
            btn_frame, text="⏸ Pause", command=self._pause_simulation, width=10, state=tk.DISABLED
        )
        self.pause_btn.pack(side=tk.LEFT, padx=2)

        self.stop_btn = ttk.Button(
            btn_frame, text="⏹ Stop", command=self._stop_simulation, width=10, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        self.reset_btn = ttk.Button(
            btn_frame, text="↻ Reset", command=self._reset_simulation, width=10
        )
        self.reset_btn.pack(side=tk.LEFT, padx=2)

        # Speed control
        speed_frame = ttk.Frame(control_frame)
        speed_frame.pack(fill=tk.X, pady=5)
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        if not hasattr(self, "speed_var"):
            self.speed_var = tk.DoubleVar(master=self.root, value=1.0)
        speed_scale = ttk.Scale(
            speed_frame, from_=0.1, to=10.0, orient=tk.HORIZONTAL, variable=self.speed_var
        )
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.speed_label = ttk.Label(speed_frame, text="1.0x")
        self.speed_label.pack(side=tk.LEFT)

        # System Status
        status_frame = ttk.LabelFrame(parent, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        self.status_labels = {}
        status_items = [
            ("Time", "0.00 s"),
            ("Ignition Events", "0"),
            ("Workspace", "Idle"),
            ("Metabolic Reserves", "100.0%"),
            ("Allostatic Load", "0.0%"),
        ]

        for label, initial in status_items:
            frame = ttk.Frame(status_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"{label}:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
            self.status_labels[label] = ttk.Label(frame, text=initial, font=("Arial", 9))
            self.status_labels[label].pack(side=tk.RIGHT)

        # Parameter Adjustments
        self.param_frame = ttk.LabelFrame(parent, text="Quick Parameters", padding=10)
        self.param_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollable parameter list
        canvas = tk.Canvas(self.param_frame, height=300)
        scrollbar = ttk.Scrollbar(self.param_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add parameter controls
        self.param_vars: Dict[str, Any] = {}
        self.param_scales: Dict[str, ttk.Scale] = {}
        self.param_labels: Dict[str, ttk.Label] = {}
        parameters = [
            ("Ignition Threshold", "baseline_threshold", 1.0, 5.0, 2.0),
            ("Extero Precision", "extero_precision", 0.1, 10.0, 1.0),
            ("Intero Precision", "intero_precision", 0.1, 10.0, 0.8),
            ("Arousal Level", "arousal", 0.0, 1.0, 0.0),
            ("Stress Level", "stress", 0.0, 1.0, 0.0),
            ("Activity Level", "activity", 0.0, 1.0, 0.0),
            ("Learning Rate", "learning_rate", 0.001, 0.1, 0.01),
            ("Attention Gain", "attention_gain", 0.5, 3.0, 1.0),
        ]

        for i, (label, key, min_val, max_val, default) in enumerate(parameters):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=3)

            ttk.Label(frame, text=label, width=18).pack(side=tk.LEFT)

            # Create tkinter variable immediately for test compatibility
            var = tk.DoubleVar(master=self.root, value=default)
            self.param_vars[key] = var

            scale = ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=var)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            val_label = ttk.Label(frame, text=f"{default:.2f}", width=6)
            val_label.pack(side=tk.LEFT)

            # Update label when variable changes
            var.trace_add(
                "write", lambda *args, v=var, lbl=val_label: lbl.config(text=f"{v.get():.2f}")
            )

            # Store scale and label references for later
            self.param_scales[key] = scale
            self.param_labels[key] = val_label

        # Initialize thread-safe parameter cache after all parameters are created
        self._param_cache: Dict[str, Any] = {}
        self._setup_param_cache()

        # Event Log
        self.log_frame = ttk.LabelFrame(parent, text="Event Log", padding=10)
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = scrolledtext.ScrolledText(
            self.log_frame, height=8, width=40, font=("Courier", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self._log_event("APGI System initialized")

    def _create_visualization_panel(self, parent: tk.Widget) -> None:
        """Create tabbed visualization panel."""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Neural Activity & Ignition
        neural_tab = ttk.Frame(notebook)
        notebook.add(neural_tab, text="Neural Activity")
        self._create_neural_plots(neural_tab)

        # Tab 2: Interoception & Body State
        intero_tab = ttk.Frame(notebook)
        notebook.add(intero_tab, text="Interoception")
        self._create_intero_plots(intero_tab)

        # Tab 3: System Metrics
        metrics_tab = ttk.Frame(notebook)
        notebook.add(metrics_tab, text="System Metrics")
        self._create_metrics_plots(metrics_tab)

        # Tab 4: Self-Model & Coherence
        self_tab = ttk.Frame(notebook)
        notebook.add(self_tab, text="Self-Model")
        self._create_self_plots(self_tab)

        # Tab 5: Oscillations & Spectral
        osc_tab = ttk.Frame(notebook)
        notebook.add(osc_tab, text="Oscillations")
        self._create_osc_plots(osc_tab)

        # Tab 6: 3D State Space (advanced)
        state_tab = ttk.Frame(notebook)
        notebook.add(state_tab, text="State Space")
        self._create_state_space(state_tab)

        # Store references for toggling
        self.notebook = notebook
        self.tab_frames = [neural_tab, intero_tab, metrics_tab, self_tab, osc_tab, state_tab]
        self.tab_texts = [
            "Neural Activity",
            "Interoception",
            "System Metrics",
            "Self-Model",
            "Oscillations",
            "State Space",
        ]

    def _create_neural_plots(self, parent: tk.Widget) -> None:
        """Create neural activity visualization."""
        fig = Figure(figsize=(10, 8))

        self.neural_axes = {
            "ignition": fig.add_subplot(4, 1, 1),
            "workspace": fig.add_subplot(4, 1, 2),
            "precision": fig.add_subplot(4, 1, 3),
            "free_energy": fig.add_subplot(4, 1, 4),
        }

        # Configure axes
        self.neural_axes["ignition"].set_ylabel("Ignition Events")
        self.neural_axes["ignition"].set_ylim(-0.1, 1.1)

        self.neural_axes["workspace"].set_ylabel("Workspace Activity")
        self.neural_axes["precision"].set_ylabel("Precision")
        self.neural_axes["free_energy"].set_ylabel("Free Energy")
        self.neural_axes["free_energy"].set_xlabel("Time (s)")

        # Initialize line plots
        self.neural_lines: Dict[str, Any] = {}
        self.neural_lines["ignition"] = self.neural_axes["ignition"].scatter(
            [], [], c="red", s=50, alpha=0.6
        )
        (self.neural_lines["workspace"],) = self.neural_axes["workspace"].plot(
            [], [], "b-", linewidth=2
        )
        (self.neural_lines["extero_precision"],) = self.neural_axes["precision"].plot(
            [], [], "g-", label="Extero", linewidth=2
        )
        (self.neural_lines["intero_precision"],) = self.neural_axes["precision"].plot(
            [], [], "r-", label="Intero", linewidth=2
        )
        (self.neural_lines["free_energy"],) = self.neural_axes["free_energy"].plot(
            [], [], "purple", linewidth=2
        )

        self.neural_axes["precision"].legend(loc="upper right")

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, parent)  # type: ignore
        canvas.draw()  # type: ignore
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # type: ignore

        toolbar = NavigationToolbar2Tk(canvas, parent)  # type: ignore
        toolbar.update()

        self.neural_canvas = canvas

    def _create_intero_plots(self, parent: tk.Widget) -> None:
        """Create interoception visualization."""
        fig = Figure(figsize=(10, 8))

        self.intero_axes = {
            "heart_rate": fig.add_subplot(4, 1, 1),
            "cortisol": fig.add_subplot(4, 1, 2),
            "allostatic_load": fig.add_subplot(4, 1, 3),
            "metabolic": fig.add_subplot(4, 1, 4),
        }

        self.intero_axes["heart_rate"].set_ylabel("Heart Rate (bpm)")
        self.intero_axes["cortisol"].set_ylabel("Cortisol (μg/dL)")
        self.intero_axes["allostatic_load"].set_ylabel("Allostatic Load")
        self.intero_axes["metabolic"].set_ylabel("Metabolic Reserves")
        self.intero_axes["metabolic"].set_xlabel("Time (s)")

        self.intero_lines: Dict[str, Any] = {}
        (self.intero_lines["heart_rate"],) = self.intero_axes["heart_rate"].plot(
            [], [], "r-", linewidth=2
        )
        (self.intero_lines["cortisol"],) = self.intero_axes["cortisol"].plot(
            [], [], "orange", linewidth=2
        )
        (self.intero_lines["allostatic_load"],) = self.intero_axes["allostatic_load"].plot(
            [], [], "darkred", linewidth=2
        )
        (self.intero_lines["metabolic"],) = self.intero_axes["metabolic"].plot(
            [], [], "green", linewidth=2
        )

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, parent)  # type: ignore
        canvas.draw()  # type: ignore
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # type: ignore

        toolbar = NavigationToolbar2Tk(canvas, parent)  # type: ignore
        toolbar.update()

        self.intero_canvas = canvas

    def _create_metrics_plots(self, parent: tk.Widget) -> None:
        """Create system metrics visualization."""
        fig = Figure(figsize=(10, 8))

        self.metrics_axes = {
            "somatic": fig.add_subplot(4, 1, 1),
            "gamma": fig.add_subplot(4, 1, 2),
            "performance": fig.add_subplot(4, 1, 3),
            "memory": fig.add_subplot(4, 1, 4),
        }

        self.metrics_axes["somatic"].set_ylabel("Somatic Markers")
        self.metrics_axes["gamma"].set_ylabel("Gamma Power")
        self.metrics_axes["performance"].set_ylabel("FPS")
        self.metrics_axes["memory"].set_ylabel("Memory (MB)")
        self.metrics_axes["memory"].set_xlabel("Time (s)")

        self.metrics_lines: Dict[str, Any] = {}
        (self.metrics_lines["somatic"],) = self.metrics_axes["somatic"].plot(
            [], [], "purple", linewidth=2
        )
        (self.metrics_lines["gamma"],) = self.metrics_axes["gamma"].plot(
            [], [], "darkgreen", linewidth=2
        )
        (self.metrics_lines["performance"],) = self.metrics_axes["performance"].plot(
            [], [], "red", linewidth=2
        )
        (self.metrics_lines["memory"],) = self.metrics_axes["memory"].plot(
            [], [], "orange", linewidth=2
        )

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, parent)  # type: ignore
        canvas.draw()  # type: ignore
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # type: ignore

        toolbar = NavigationToolbar2Tk(canvas, parent)  # type: ignore
        toolbar.update()

        self.metrics_canvas = canvas

    def _create_self_plots(self, parent: tk.Widget) -> None:
        """Create self-model visualization."""
        fig = Figure(figsize=(10, 8))

        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel("Coherence")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 1.1)

        self.coherence_line: Any = None
        (self.coherence_line,) = ax.plot([], [], "b-", linewidth=2, label="Minimal Self")
        ax.axhline(y=0.7, color="g", linestyle="--", label="Normal Threshold")
        ax.axhline(y=0.4, color="r", linestyle="--", label="Depersonalization")
        ax.legend()

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, parent)  # type: ignore
        canvas.draw()  # type: ignore
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # type: ignore

        toolbar = NavigationToolbar2Tk(canvas, parent)  # type: ignore
        toolbar.update()

        self.self_canvas = canvas
        self.self_ax = ax

    def _create_osc_plots(self, parent: tk.Widget) -> None:
        """Create oscillation visualization."""
        fig = Figure(figsize=(10, 8))

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.set_ylabel("Oscillation Signal")
        ax1.set_xlabel("Time (s)")
        ax2.set_ylabel("Power")
        ax2.set_xlabel("Frequency Band")

        self.osc_signal_line: Any = None
        (self.osc_signal_line,) = ax1.plot([], [], "b-", linewidth=1)

        # Bar plot for power spectrum
        self.osc_bars: Any = None
        self.osc_ax1 = ax1
        self.osc_ax2 = ax2

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, parent)  # type: ignore
        canvas.draw()  # type: ignore
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # type: ignore

        toolbar = NavigationToolbar2Tk(canvas, parent)  # type: ignore
        toolbar.update()

        self.osc_canvas = canvas

    def _create_state_space(self, parent: tk.Widget) -> None:
        """Create 3D state space visualization."""

        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlabel("Free Energy")
        ax.set_ylabel("Precision")
        ax.set_zlabel("Allostatic Load")
        ax.set_title("3D State Space Trajectory")

        self.state_scatter: Any = None
        self.state_scatter = ax.scatter([], [], [], c="b", marker="o", s=20, alpha=0.6)
        self.state_ax = ax

        canvas = FigureCanvasTkAgg(fig, parent)  # type: ignore
        canvas.draw()  # type: ignore
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # type: ignore

        toolbar = NavigationToolbar2Tk(canvas, parent)  # type: ignore
        toolbar.update()

        self.state_canvas = canvas

    def _create_status_bar(self) -> None:
        status_bar = ttk.Frame(self.root)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_text = ttk.Label(status_bar, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.fps_label = ttk.Label(status_bar, text="0 FPS", relief=tk.SUNKEN, width=10)
        self.fps_label.pack(side=tk.RIGHT)

    # System Control Methods

    def _initialize_system(self) -> None:
        """Initialize APGI system with proper error handling."""
        try:
            self.apgi_simulation = APGISystem(config_path=str(self.config_path))
            if self.apgi_simulation is None:
                raise RuntimeError("APGI system initialization returned None")

            self._log_event("System initialized successfully")
            self._update_status("System ready")

            # Enable UI controls that require a valid system
            self._enable_system_controls(True)

        except Exception as e:
            self.apgi_simulation = None
            import traceback

            error_details = (
                f"Failed to initialize system:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            )
            self._log_event(f"System initialization failed: {str(e)}")

            # Enhanced error dialog with troubleshooting guidance
            troubleshooting_msg = (
                f"{error_details}\n\n"
                "TROUBLESHOOTING STEPS:\n"
                "1. Check configuration file exists and is valid YAML\n"
                "2. Verify all dependencies are installed: pip install -r requirements.txt\n"
                "3. Check system resources (memory, disk space)\n"
                "4. Try resetting to default configuration\n"
                "5. Restart the application\n\n"
                "For detailed help, see APGI-Simulation-README.md"
            )

            messagebox.showerror("Initialization Error", troubleshooting_msg)

            self._log_event(f"CRITICAL ERROR: {str(e)}")
            self._update_status("System initialization failed - See troubleshooting steps")

            # Log the full error for debugging
            logger.error(f"System initialization failed: {str(e)}", exc_info=True)

            # Offer to reset configuration
            try:
                reset_result = messagebox.askyesno(
                    "Reset Configuration?",
                    "Would you like to reset to default configuration?\n\n"
                    "This may resolve initialization issues.",
                )
                if reset_result:
                    self._reset_to_default_config()
            except Exception as e:
                logger.error(f"Error: {e}")

    def _reset_to_default_config(self) -> None:
        """Reset configuration to default settings."""
        try:
            # Reset to default configuration path
            self.config_path = get_resource_path("config/default.yaml")

            # Reinitialize system with default config
            self._initialize_system()

            # Log the reset
            self._log_event("Configuration reset to defaults")

            # Show success message
            messagebox.showinfo(
                "Configuration Reset",
                "Configuration has been reset to default settings.\n\n"
                "The system will now attempt to initialize with defaults.",
            )

        except Exception as e:
            # Show error if reset fails
            messagebox.showerror(
                "Reset Failed",
                f"Failed to reset configuration:\n{str(e)}\n\n"
                "Please check the configuration file manually.",
            )
            self._log_event(f"Configuration reset failed: {str(e)}")

    def _enable_system_controls(self, enabled: bool) -> None:
        """Enable or disable UI controls that require a valid system."""
        # Simulation controls
        self.start_btn.config(state=tk.NORMAL if enabled else tk.DISABLED)
        self.reset_btn.config(state=tk.NORMAL if enabled else tk.DISABLED)

        # Parameter controls
        if hasattr(self, "param_scales"):
            for scale in self.param_scales.values():
                scale.config(state=tk.NORMAL if enabled else tk.DISABLED)

        # Menu items that require system (these would need to be stored as references)
        # For now, we'll handle this in the individual menu methods

    def _start_simulation(self) -> None:
        """Start simulation with system validation."""
        if self.is_running:
            return

        if self.apgi_simulation is None:
            messagebox.showerror(
                "System Not Available",
                "Cannot start simulation: APGI system is not initialized.\n\n"
                "Please check your configuration and restart the application.",
            )
            return

        self.is_running = True
        self.is_paused = False

        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)

        self._log_event("Simulation started")
        self._update_status("Running simulation...")

        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()

    def _pause_simulation(self) -> None:
        """Pause/resume simulation."""
        self.is_paused = not self.is_paused

        if self.is_paused:
            self.pause_btn.config(text="▶ Resume")
            self._log_event("Simulation paused")
            self._update_status("Paused")
        else:
            self.pause_btn.config(text="⏸ Pause")
            self._log_event("Simulation resumed")
            self._update_status("Running simulation...")

    def _update_ui_after_error(self) -> None:
        """Update UI button states after simulation error."""
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self.stop_btn.config(state=tk.DISABLED)
        self._update_status("Simulation stopped due to error")

    def _stop_simulation(self) -> None:
        """Stop simulation."""
        self.is_running = False
        self.is_paused = False

        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self.stop_btn.config(state=tk.DISABLED)

        self._log_event("Simulation stopped")
        self._update_status("Stopped")

    def _reset_simulation(self) -> None:
        """Reset simulation (thread-safe) with system validation."""
        was_running = self.is_running
        if was_running:
            self._stop_simulation()

        if self.apgi_simulation is None:
            self._log_event("Cannot reset: System not initialized")
            messagebox.showwarning(
                "System Not Available",
                "Cannot reset simulation: APGI system is not initialized.\n\n"
                "Please check your configuration and restart the application.",
            )
            return

        try:
            self.apgi_simulation.reset()
        except Exception as e:
            self._log_event(f"Error resetting system: {str(e)}")
            messagebox.showerror("Reset Error", f"Failed to reset system:\n{str(e)}")
            return

        # Clear buffers with thread safety
        with self.data_lock:
            for buffer in self.data_buffers.values():
                buffer.clear()
            self.time_buffer.clear()

        # Clear log data with thread safety
        with self.log_lock:
            self.log_data.clear()

        self._log_event("System reset")
        self._update_status("System reset")

        # Update displays
        self._update_plots()

    def _simulation_loop(self) -> None:
        """Main simulation loop (runs in separate thread) with system validation."""
        last_time = time.time()
        frame_count = 0
        fps_update_interval = 1.0

        while self.is_running:
            if not self.is_paused:
                try:
                    # Validate system before each step
                    if self.apgi_simulation is None:
                        self._log_event("Simulation stopped: System not available")
                        self.is_running = False
                        # Schedule GUI update to disable controls
                        self.root.after(0, lambda: self._safe_enable_system_controls(False))
                        break

                    # Generate input
                    current_time = self.apgi_simulation.time / 1000.0  # Convert to seconds
                    extero_input = self._generate_input(current_time)

                    # Apply parameter adjustments
                    self._apply_parameters()

                    # Step simulation
                    state = self.apgi_simulation.step(extero_input)

                    # Record data
                    self._record_state(state)

                    # Auto-save if enabled
                    if self.auto_save and len(self.log_data) % 1000 == 0:
                        self._auto_save_data()

                    frame_count += 1

                except Exception as e:
                    logger.error(f"Simulation error: {e}", exc_info=True)
                    self.is_running = False
                    self.is_paused = False
                    # Schedule GUI update with proper button states
                    self.root.after(0, self._safe_update_ui_after_error)

            # FPS calculation
            current = time.time()
            if current - last_time >= fps_update_interval:
                fps = frame_count / (current - last_time)
                self.root.after(0, lambda fps=fps: self._safe_update_fps(fps))  # type: ignore[misc]
                frame_count = 0
                last_time = current

            # Control simulation speed
            speed = self._get_thread_safe_speed()
            time.sleep(0.001 / max(0.1, speed))

    def _generate_input(self, t: float) -> NDArray[np.float64]:
        """Generate sensory input."""
        # Check for custom input
        if self.custom_input_steps_remaining > 0:
            params = self.custom_input_params
            if params:
                pattern = params["pattern"]
                intensity = params["intensity"]
                noise_level = params["noise"]

                if pattern == "constant":
                    base = np.full(256, intensity, dtype=np.float64)
                elif pattern == "pulse":
                    # Create a pulse in the middle of the step
                    base = np.zeros(256, dtype=np.float64)
                    base[intensity > 0.5] = intensity  # type: ignore[operator, call-overload]
                elif pattern == "sine":
                    base = (
                        intensity
                        * (np.sin(2 * np.pi * t / 5.0) + 1)
                        / 2
                        * np.ones(256, dtype=np.float64)
                    )
                else:  # random
                    base = np.random.randn(256) * intensity  # type: ignore[operator]

                noise = np.random.randn(256) * noise_level  # type: ignore[operator]
                input_signal: NDArray[np.float64] = base + noise

                self.custom_input_steps_remaining -= 1
                if self.custom_input_steps_remaining == 0:
                    self.custom_input_params = None
                    self._log_event("Custom input injection completed")

                return input_signal

        # Default sinusoidal with noise
        base = np.sin(2 * np.pi * t / 5.0) * np.ones(256, dtype=np.float64)
        noise = np.random.randn(256) * 0.2
        result: NDArray[np.float64] = base + noise
        return result

    def _safe_update_fps(self, fps: float) -> None:
        """Safely update FPS label with existence check."""
        try:
            if hasattr(self, "fps_label") and self.fps_label.winfo_exists():
                self.fps_label.config(text=f"{fps:.1f} FPS")
        except Exception as e:
            # Log errors during shutdown
            logger.warning(f"Error updating FPS label: {str(e)}")

    def _safe_enable_system_controls(self, enabled: bool) -> None:
        """Safely enable/disable system controls with existence check."""
        try:
            if hasattr(self, "start_btn") and self.start_btn.winfo_exists():
                self._enable_system_controls(enabled)
        except Exception as e:
            # Log errors during shutdown
            logger.warning(f"Error enabling/disabling controls: {str(e)}")

    def _safe_update_ui_after_error(self) -> None:
        """Safely update UI after error with existence check."""
        try:
            if hasattr(self, "start_btn") and self.start_btn.winfo_exists():
                self._update_ui_after_error()
        except Exception as e:
            # Ignore errors during shutdown
            logger.error(f"Error: {str(e)}")

    def _apply_parameters(self) -> None:
        """Apply parameter adjustments to system with validation and rollback."""
        if self.apgi_simulation is None:
            self._log_event("Cannot apply parameters: System not initialized")
            return

        # Validate parameter values before applying
        validation_errors = []

        # Define valid ranges for parameters
        valid_ranges = {
            "arousal": (0.0, 1.0),
            "stress": (0.0, 1.0),
            "activity": (0.0, 1.0),
            "extero_precision": (0.1, 10.0),
            "intero_precision": (0.1, 10.0),
            "baseline_threshold": (1.0, 5.0),
        }

        # Check each parameter individually
        for param_name, (min_val, max_val) in valid_ranges.items():
            if param_name in self._param_cache:
                try:
                    value = self._param_cache[param_name]
                    if not isinstance(value, (int, float)):
                        validation_errors.append(f"{param_name}: Must be a number")
                    elif not (min_val <= value <= max_val):
                        validation_errors.append(
                            f"{param_name}: Must be between {min_val} and {max_val}"
                        )
                except Exception as e:
                    validation_errors.append(f"{param_name}: Error reading value: {str(e)}")

        # Cross-parameter validation for incompatible combinations
        try:
            arousal = self._param_cache.get("arousal", 0.0)
            stress = self._param_cache.get("stress", 0.0)
            activity = self._param_cache.get("activity", 0.0)
            extero_prec = self._param_cache.get("extero_precision", 1.0)
            intero_prec = self._param_cache.get("intero_precision", 0.8)
            threshold = self._param_cache.get("baseline_threshold", 2.0)

            # Check for potentially unstable combinations
            if arousal > 0.8 and stress > 0.8 and activity > 0.8:
                validation_errors.append(
                    "High arousal + stress + activity may cause instability. "
                    "Consider reducing at least one parameter."
                )

            if extero_prec > 8.0 and intero_prec > 8.0:
                validation_errors.append(
                    "Very high precision values may cause numerical overflow. "
                    "Consider keeping precision ≤ 8.0."
                )

            if threshold < 1.5 and (arousal > 0.9 or stress > 0.9):
                validation_errors.append(
                    "Low threshold with high arousal/stress may cause excessive ignition events. "
                    "Consider increasing threshold or reducing arousal/stress."
                )

            if activity > 0.9 and threshold > 4.0:
                validation_errors.append(
                    "High activity with very high threshold may prevent ignition. "
                    "Consider reducing threshold or activity."
                )

        except Exception as e:
            validation_errors.append("Cross-parameter validation error: " + str(e))

        # If validation errors exist, show them and abort
        if validation_errors:
            error_msg = "Parameter validation failed:\n" + "\n".join(validation_errors)
            self._log_event(f"Parameter validation failed: {'; '.join(validation_errors)}")
            messagebox.showerror("Validation Error", error_msg)
            return

        # Store current values for rollback
        old_values = {}
        try:
            # Save current values
            old_values = {
                "arousal": self.apgi_simulation.body_model.arousal_level,
                "stress": self.apgi_simulation.body_model.stress_level,
                "activity": self.apgi_simulation.body_model.activity_level,
                "extero_precision": self.apgi_simulation.precision.extero_baseline,
                "intero_precision": self.apgi_simulation.precision.intero_baseline,
                "baseline_threshold": self.apgi_simulation.ignition_threshold.baseline_threshold,
            }

            # Apply body state modulations
            self.apgi_simulation.body_model.set_arousal(self._param_cache["arousal"])
            self.apgi_simulation.body_model.set_stress(self._param_cache["stress"])
            self.apgi_simulation.body_model.set_activity(self._param_cache["activity"])

            # Apply precision
            self.apgi_simulation.precision.extero_baseline = self._param_cache["extero_precision"]
            self.apgi_simulation.precision.intero_baseline = self._param_cache["intero_precision"]

            # Apply threshold
            self.apgi_simulation.ignition_threshold.baseline_threshold = self._param_cache[
                "baseline_threshold"
            ]

        except Exception as e:
            self._log_event("Error applying parameters - Rolling back...")
            logger.error(f"Parameter application error: {str(e)}", exc_info=True)

            # Rollback to previous values
            try:
                if old_values:
                    self.apgi_simulation.body_model.set_arousal(old_values["arousal"])
                    self.apgi_simulation.body_model.set_stress(old_values["stress"])
                    self.apgi_simulation.body_model.set_activity(old_values["activity"])
                    self.apgi_simulation.precision.extero_baseline = old_values["extero_precision"]
                    self.apgi_simulation.precision.intero_baseline = old_values["intero_precision"]
                    self.apgi_simulation.ignition_threshold.baseline_threshold = old_values[
                        "baseline_threshold"
                    ]
                    self._log_event("Parameters rolled back to previous values")
            except Exception as rollback_error:
                self._log_event(f"CRITICAL: Failed to rollback parameters: {str(rollback_error)}")
                logger.error(f"Rollback failed: {str(rollback_error)}", exc_info=True)

    def _record_state(self, state: Any) -> None:
        """Record state data (thread-safe)."""
        current_time = state["time"] / 1000.0  # Convert to seconds

        with self.data_lock:
            self.time_buffer.append(current_time)

            # Extract and record metrics (convert arrays to scalars)
            self.data_buffers["ignition"].append(1 if state["ignition"]["ignition_occurred"] else 0)
            self.data_buffers["free_energy"].append(
                float(
                    np.mean(state["ignition"]["total_signal"])
                    if hasattr(state["ignition"]["total_signal"], "__iter__")
                    else state["ignition"]["total_signal"]
                )
            )
            self.data_buffers["extero_precision"].append(
                float(
                    np.mean(state["precision"]["exteroceptive"])
                    if hasattr(state["precision"]["exteroceptive"], "__iter__")
                    else state["precision"]["exteroceptive"]
                )
            )
            self.data_buffers["intero_precision"].append(
                float(
                    np.mean(state["precision"]["interoceptive"])
                    if hasattr(state["precision"]["interoceptive"], "__iter__")
                    else state["precision"]["interoceptive"]
                )
            )
            self.data_buffers["metabolic_reserves"].append(
                float(
                    np.mean(state["metabolism"]["reserves"])
                    if hasattr(state["metabolism"]["reserves"], "__iter__")
                    else state["metabolism"]["reserves"]
                )
            )
            self.data_buffers["allostatic_load"].append(
                float(
                    np.mean(state["allostasis"]["allostatic_load"])
                    if hasattr(state["allostasis"]["allostatic_load"], "__iter__")
                    else state["allostasis"]["allostatic_load"]
                )
            )
            self.data_buffers["heart_rate"].append(state["body"]["current"]["heart_rate"])
            self.data_buffers["cortisol"].append(state["body"]["current"]["cortisol"])
            self.data_buffers["workspace_active"].append(
                1 if state["workspace"]["is_reportable"] else 0
            )
            # Extract oscillation data safely
            oscillations = state.get("oscillations", {})
            if isinstance(oscillations, dict):
                band_powers = oscillations.get("band_powers", {})
                gamma_power = band_powers.get("gamma", 0)
                beta_power = band_powers.get("beta", 0)
                theta_power = band_powers.get("theta", 0)
                alpha_power = band_powers.get("alpha", 0)
            else:
                gamma_power = beta_power = theta_power = alpha_power = 0

            self.data_buffers["gamma_power"].append(gamma_power)
            self.data_buffers["beta_power"].append(beta_power)
            self.data_buffers["theta_power"].append(theta_power)
            self.data_buffers["alpha_power"].append(alpha_power)
            # Calculate delta_power as difference between gamma and beta power
            self.data_buffers["delta_power"].append(abs(gamma_power - beta_power))
            if state is not None:
                self.data_buffers["minimal_self_coherence"].append(
                    state["self_model"]["minimal"]["coherence"]
                )

            # Count somatic markers
            if (
                self.apgi_simulation is not None
                and hasattr(self.apgi_simulation, "somatic_markers")
                and hasattr(self.apgi_simulation.somatic_markers, "markers")
            ):
                self.data_buffers["somatic_markers"].append(
                    len(self.apgi_simulation.somatic_markers.markers)
                )
            else:
                self.data_buffers["somatic_markers"].append(0)

            # Performance metrics
            if len(self.fps_buffer) > 0:
                self.data_buffers["performance"].append(self.fps_buffer[-1])
            else:
                self.data_buffers["performance"].append(0)

            # Memory usage
            if len(self.memory_buffer) > 0:
                self.data_buffers["memory_mb"].append(self.memory_buffer[-1])
            else:
                self.data_buffers["memory_mb"].append(0)

            # Create a snapshot for logging (outside of data_lock to avoid deadlock)
            log_entry = {
                "time": current_time,
                **{k: v[-1] if v else 0 for k, v in self.data_buffers.items()},
            }

        # Log data for export (separate lock)
        with self.log_lock:
            self.log_data.append(log_entry)

    def _update_displays(self) -> None:
        """Update all displays (called periodically)."""
        # Calculate FPS
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        if frame_time > 0:
            fps = 1.0 / frame_time
            self.fps_buffer.append(fps)
        self.last_frame_time = current_time

        # Get memory usage
        try:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.memory_buffer.append(memory_mb)
        except Exception as e:
            logger.error(f"Error: {str(e)}")

        if self.is_running and not self.is_paused:
            self._update_status_labels()
            self._update_plots()

        # Schedule next update - use 50ms for smoother 20Hz refresh rate
        self.root.after(50, self._update_displays)  # Update every 50ms (20Hz)

    def _update_status_labels(self) -> None:
        """Update status labels with system validation."""
        # Ensure status_labels are initialized before trying to update them
        if not hasattr(self, "status_labels") or not self.status_labels:
            return
        if self.apgi_simulation is None:
            # Show disabled status when system is not available
            self.status_labels["Time"].config(text="--.-- s")
            self.status_labels["Ignition Events"].config(text="--")
            self.status_labels["Workspace"].config(text="Unavailable")
            self.status_labels["Metabolic Reserves"].config(text="--.%")
            self.status_labels["Allostatic Load"].config(text="--.%")
            return

        try:
            summary = self.apgi_simulation.get_state_summary()

            self.status_labels["Time"].config(text=f"{summary['time_ms'] / 1000.0:.2f} s")

            stats = summary["ignition_stats"]
            self.status_labels["Ignition Events"].config(text=str(stats["recent_ignitions"]))

            workspace = "Broadcasting" if summary["workspace_state"] == "broadcasting" else "Idle"
            self.status_labels["Workspace"].config(text=workspace)

            reserves = summary["metabolic_reserves"]
            reserves_scalar = float(
                np.mean(reserves) if hasattr(reserves, "__iter__") else reserves
            )
            self.status_labels["Metabolic Reserves"].config(text=f"{reserves_scalar:.1f}%")

            load = summary["allostatic_load"] * 100
            load_scalar = float(np.mean(load) if hasattr(load, "__iter__") else load)
            self.status_labels["Allostatic Load"].config(text=f"{load_scalar:.1f}%")

        except Exception as e:
            self._log_event("Error updating status labels")
            logger.error(f"Status update error: {str(e)}", exc_info=True)

    def _update_plots(self) -> None:
        """Update all plot canvases (thread-safe)."""
        with self.data_lock:
            if len(self.time_buffer) < 2:
                return

            # Check all buffers have minimum required data length
            min_buffer_length = len(self.time_buffer)
            for key, buffer in self.data_buffers.items():
                if len(buffer) < min_buffer_length:
                    self._log_event(
                        f"Buffer {key} length mismatch: " f"{len(buffer)} vs {min_buffer_length}"
                    )
                    return

            # Create copies of data for thread-safe access
            time_data = np.array(self.time_buffer)
            data_copies = {}
            for key, buffer in self.data_buffers.items():
                data_copies[key] = np.array(buffer)

        try:
            # Update neural plots
            self._update_neural_plots(time_data, data_copies)

            # Update interoception plots
            self._update_intero_plots(time_data, data_copies)

            # Update metrics plots
            self._update_metrics_plots(time_data, data_copies)

            # Update self-model plot
            self._update_self_plot(time_data, data_copies)

            # Update oscillation plots
            self._update_osc_plots(time_data, data_copies)

            # Update 3D state space
            self._update_state_space(time_data, data_copies)

        except Exception as e:
            self._log_event("Error updating plots")
            logger.error(f"Plot update error: {str(e)}", exc_info=True)

    def _update_neural_plots(self, time_data: Any, data_copies: Dict[str, Any]) -> None:
        """Update neural activity plots."""
        # Ignition events (scatter plot)
        ignitions = data_copies["ignition"]
        ignition_times = time_data[ignitions > 0]
        ignition_values = ignitions[ignitions > 0]

        if len(ignition_times) > 0:
            self.neural_lines["ignition"].set_offsets(
                np.column_stack([ignition_times, ignition_values])
            )

        # Workspace activity
        workspace = data_copies["workspace_active"]
        self.neural_lines["workspace"].set_data(time_data, workspace)

        # Precision
        extero_prec = data_copies["extero_precision"]
        intero_prec = data_copies["intero_precision"]
        self.neural_lines["extero_precision"].set_data(time_data, extero_prec)
        self.neural_lines["intero_precision"].set_data(time_data, intero_prec)

        # Free energy
        fe = data_copies["free_energy"]
        self.neural_lines["free_energy"].set_data(time_data, fe)

        # Update axis limits
        for key, ax in self.neural_axes.items():
            if key != "ignition":
                ax.set_xlim(time_data[0], time_data[-1])
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

        self.neural_axes["ignition"].set_xlim(time_data[0], time_data[-1])

        self.neural_canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _update_intero_plots(self, time_data: Any, data_copies: Dict[str, Any]) -> None:
        """Update interoception plots."""
        hr = data_copies["heart_rate"]
        cortisol = data_copies["cortisol"]
        load = data_copies["allostatic_load"]
        metabolic = data_copies["metabolic_reserves"]

        self.intero_lines["heart_rate"].set_data(time_data, hr)
        self.intero_lines["cortisol"].set_data(time_data, cortisol)
        self.intero_lines["allostatic_load"].set_data(time_data, load)
        self.intero_lines["metabolic"].set_data(time_data, metabolic)

        for ax in self.intero_axes.values():
            ax.set_xlim(time_data[0], time_data[-1])
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        self.intero_canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _update_metrics_plots(self, time_data: Any, data_copies: Dict[str, Any]) -> None:
        """Update system metrics plots."""
        somatic = data_copies["somatic_markers"]
        gamma = data_copies["gamma_power"]

        # Performance metrics
        if len(self.fps_buffer) > 0:
            avg_fps = np.mean(list(self.fps_buffer)[-30:])  # Last 30 frames
            performance_data = np.full_like(time_data, avg_fps)
        else:
            performance_data = np.zeros_like(time_data)

        if len(self.memory_buffer) > 0:
            memory_data = np.full_like(time_data, self.memory_buffer[-1])
        else:
            memory_data = np.zeros_like(time_data)

        self.metrics_lines["somatic"].set_data(time_data, somatic)
        self.metrics_lines["gamma"].set_data(time_data, gamma)
        self.metrics_lines["performance"].set_data(time_data, performance_data)
        self.metrics_lines["memory"].set_data(time_data, memory_data)

        for ax in self.metrics_axes.values():
            ax.set_xlim(time_data[0], time_data[-1])
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        self.metrics_canvas.draw_idle()  # type: ignore

    def _update_self_plot(self, time_data: Any, data_copies: Dict[str, Any]) -> None:
        """Update self-model plot."""
        coherence = data_copies["minimal_self_coherence"]
        self.coherence_line.set_data(time_data, coherence)

        self.self_ax.set_xlim(time_data[0], time_data[-1])

        self.self_canvas.draw_idle()  # type: ignore

    def _update_osc_plots(self, time_data: Any, data_copies: Dict[str, Any]) -> None:
        """Update oscillation plots."""
        # For oscillation signal, show recent window
        window_size = min(500, len(time_data))
        recent_time = time_data[-window_size:]

        # Use actual oscillation data from system instead of synthetic
        oscillation_data = data_copies.get("oscillation", [])
        if len(oscillation_data) > 0:
            recent_osc = oscillation_data[-window_size:]
        else:
            # Fallback to synthetic data if no actual data available
            recent_osc = np.sin(2 * np.pi * 20 * recent_time) + 0.5 * np.sin(
                2 * np.pi * 40 * recent_time
            )
        self.osc_signal_line.set_data(recent_time, recent_osc)

        self.osc_ax1.set_xlim(recent_time[0], recent_time[-1])
        self.osc_ax1.set_ylim(-2, 2)

        # Power spectrum
        if self.osc_bars is None:
            bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
            # Initialize with dynamic values from data
            delta_power = (
                data_copies["delta_power"][-1] if len(data_copies["delta_power"]) > 0 else 0.5
            )
            theta_power = (
                data_copies["theta_power"][-1] if len(data_copies["theta_power"]) > 0 else 0.7
            )
            alpha_power = (
                data_copies["alpha_power"][-1] if len(data_copies["alpha_power"]) > 0 else 1.0
            )
            beta_power = (
                data_copies["beta_power"][-1] if len(data_copies["beta_power"]) > 0 else 0.8
            )
            gamma_power = (
                data_copies["gamma_power"][-1] if len(data_copies["gamma_power"]) > 0 else 0.6
            )
            powers = [delta_power, theta_power, alpha_power, beta_power, gamma_power]
            self.osc_bars = self.osc_ax2.bar(bands, powers)
            self.osc_ax2.set_ylim(0, 1.5)
        else:
            # Update bar heights with dynamic values
            delta_power = (
                data_copies["delta_power"][-1] if len(data_copies["delta_power"]) > 0 else 0.5
            )
            theta_power = (
                data_copies["theta_power"][-1] if len(data_copies["theta_power"]) > 0 else 0.7
            )
            alpha_power = (
                data_copies["alpha_power"][-1] if len(data_copies["alpha_power"]) > 0 else 1.0
            )
            beta_power = (
                data_copies["beta_power"][-1] if len(data_copies["beta_power"]) > 0 else 0.8
            )
            gamma_power = (
                data_copies["gamma_power"][-1] if len(data_copies["gamma_power"]) > 0 else 0.6
            )

            powers = [delta_power, theta_power, alpha_power, beta_power, gamma_power]
            for bar, power in zip(self.osc_bars, powers):
                bar.set_height(power)

        self.osc_canvas.draw_idle()  # type: ignore

    def _update_state_space(self, time_data: Any, data_copies: Dict[str, Any]) -> None:
        """Update 3D state space plot."""
        if len(time_data) < 10:
            return

        # Get recent trajectory
        fe = data_copies["free_energy"]
        prec = data_copies["extero_precision"]
        load = data_copies["allostatic_load"]

        # Color by time

        self.state_scatter._offsets3d = (fe, prec, load)

        # Fix axis limits to avoid identical min/max warning
        fe_min, fe_max = fe.min(), fe.max()
        prec_min, prec_max = prec.min(), prec.max()

        # Ensure min and max are different by at least a small epsilon
        if abs(fe_max - fe_min) < 1e-10:
            fe_center = (fe_min + fe_max) / 2
            fe_min, fe_max = fe_center - 0.5, fe_center + 0.5

        if abs(prec_max - prec_min) < 1e-10:
            prec_center = (prec_min + prec_max) / 2
            prec_min, prec_max = prec_center - 0.5, prec_center + 0.5

        self.state_ax.set_xlim(fe_min, fe_max)
        self.state_ax.set_ylim(prec_min, prec_max)
        self.state_ax.set_zlim(0, 1)

        self.state_canvas.draw_idle()  # type: ignore[no-untyped-call]

    # Menu Command Methods

    def _new_session(self) -> None:
        """Start new session (thread-safe)."""
        if messagebox.askyesno("New Session", "Start a new session? Current data will be lost."):
            self._reset_simulation()
            with self.log_lock:
                self.log_data.clear()
            self._log_event("New session started")

    def _load_config(self) -> None:
        """Load configuration file."""
        filename = filedialog.askopenfilename(
            title="Load Configuration", filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            config_path = Path(filename)

            # Validate config_path against allowlist of directories
            allowed_dirs = [Path.home(), Path.cwd()]
            if not any(
                config_path.parent == allowed_dir or config_path.parent.is_relative_to(allowed_dir)
                for allowed_dir in allowed_dirs
            ):
                messagebox.showerror(
                    "Security Error",
                    "Configuration files can only be loaded from allowed directories:\n\n"
                    f"- {Path.home()}\n- {Path.cwd()}\n\n"
                    "Please select a file from one of these locations.",
                )
                return

            try:
                with open(filename, "r") as f:
                    yaml.safe_load(f)

                # Validate configuration before using it
                try:
                    validate_config_file(filename)
                except ConfigValidationError:
                    messagebox.showerror(
                        "Configuration Error",
                        "Configuration validation failed.\n\nPlease fix the configuration file and try again.",
                    )
                    return

                self.config_path = config_path
                self._initialize_system()
                self._log_event(f"Configuration loaded: {filename}")
                messagebox.showinfo("Success", "Configuration loaded successfully")
            except yaml.YAMLError:
                messagebox.showerror("YAML Error", "Invalid YAML format")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")

    def _save_config(self) -> None:
        """Save configuration file."""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
        )
        if filename:
            try:
                # Create config from current parameters
                config = {"parameters": {k: v.get() for k, v in self.param_vars.items()}}
                with open(filename, "w") as f:
                    yaml.dump(config, f)
                self._log_event(f"Configuration saved: {filename}")
                messagebox.showinfo("Success", "Configuration saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

    def _export_data(self) -> None:
        """Export simulation data (thread-safe)."""
        with self.log_lock:
            if not self.log_data:
                messagebox.showwarning("No Data", "No simulation data to export")
                return

            # Create a copy for export to avoid holding the lock during file operations
            log_data_copy = list(self.log_data)  # Convert deque to list for export

        # Use platform-appropriate data directory as initial directory
        initial_dir = get_data_dir()
        initial_dir.mkdir(parents=True, exist_ok=True)

        filename = filedialog.asksaveasfilename(
            title="Export Data",
            initialdir=str(initial_dir),
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")],
        )

        if filename:
            # Check data size before heavy processing (BUG-M04)
            data_size = len(log_data_copy)
            if data_size > 5000:
                result = messagebox.askyesno(
                    "Large Data Export",
                    f"You are about to export {data_size} records.\n\n"
                    "This may take some time and consume significant memory. "
                    "Do you want to proceed?",
                    icon="warning",
                )
                if not result:
                    return

            try:
                if filename.endswith(".json"):
                    with open(filename, "w") as f:
                        json.dump(log_data_copy, f, indent=2)
                else:
                    # CSV export
                    with open(filename, "w", newline="") as f:
                        if log_data_copy and isinstance(log_data_copy[0], dict):
                            writer = csv.DictWriter(f, fieldnames=log_data_copy[0].keys())
                            writer.writeheader()
                            writer.writerows(log_data_copy)

                self._log_event(f"Data exported: {filename}")
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")

    def _export_plot(self) -> None:
        """Export current plot or all plots."""
        # Create export options dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Plot Options")
        dialog.geometry("400x200")
        dialog.resizable(False, False)

        # Center dialog
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Choose export option:", font=("Arial", 12, "bold")).pack(pady=20)

        export_var = tk.StringVar(value="current")

        # Radio buttons for export options
        options_frame = ttk.Frame(dialog)
        options_frame.pack(pady=10)

        ttk.Radiobutton(
            options_frame, text="Export current tab only", variable=export_var, value="current"
        ).pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(
            options_frame,
            text="Export all tabs to separate files",
            variable=export_var,
            value="separate",
        ).pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(
            options_frame,
            text="Export all tabs to multi-page PDF",
            variable=export_var,
            value="pdf",
        ).pack(anchor=tk.W, pady=5)

        def proceed_with_export() -> None:
            dialog.destroy()
            self._do_export_plot(export_var.get())

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)

        ttk.Button(button_frame, text="Export", command=proceed_with_export).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=10)

    def _do_export_plot(self, export_type: str) -> None:
        """Perform the actual plot export."""
        canvases = [
            ("Neural_Activity", self.neural_canvas),
            ("Interoception", self.intero_canvas),
            ("System_Metrics", self.metrics_canvas),
            ("Self_Model", self.self_canvas),
            ("Oscillations", self.osc_canvas),
            ("State_Space", self.state_canvas),
        ]

        if export_type == "current":
            # Export current tab only
            try:
                current_tab_index = self.notebook.index("current")  # type: ignore
                tab_name, canvas = canvases[current_tab_index]
            except (AttributeError, IndexError):
                tab_name, canvas = canvases[0]  # Fallback to neural canvas

            filename = filedialog.asksaveasfilename(
                title="Export Plot",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
            )

            if filename:
                try:
                    canvas.figure.savefig(filename, dpi=300, bbox_inches="tight")
                    self._log_event(f"Plot exported: {filename}")
                    messagebox.showinfo("Success", f"Plot saved to {filename}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to export plot: {str(e)}")

        elif export_type == "separate":
            # Export each tab to separate files
            initial_dir = get_data_dir()
            initial_dir.mkdir(parents=True, exist_ok=True)

            for tab_name, canvas in canvases:
                filename = filedialog.asksaveasfilename(
                    title=f"Export {tab_name} Plot",
                    initialdir=str(initial_dir),
                    initialfile=f"{tab_name.lower()}_plot.png",
                    defaultextension=".png",
                    filetypes=[
                        ("PNG files", "*.png"),
                        ("PDF files", "*.pdf"),
                        ("All files", "*.*"),
                    ],
                )

                if filename:
                    try:
                        canvas.figure.savefig(filename, dpi=300, bbox_inches="tight")
                        self._log_event(f"Plot exported: {filename}")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to export {tab_name} plot: {str(e)}")
                        return

            messagebox.showinfo("Success", "All plots exported successfully!")

        elif export_type == "pdf":
            # Export all tabs to multi-page PDF
            filename = filedialog.asksaveasfilename(
                title="Export All Plots to PDF",
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            )

            if filename:
                try:
                    from matplotlib.backends.backend_pdf import PdfPages

                    with PdfPages(filename) as pdf:
                        for tab_name, canvas in canvases:
                            pdf.savefig(canvas.figure, dpi=300, bbox_inches="tight")

                    self._log_event(f"All plots exported to PDF: {filename}")
                    messagebox.showinfo("Success", f"All plots saved to PDF: {filename}")

                except ImportError:
                    messagebox.showerror(
                        "Error",
                        "PDF export requires matplotlib. Please install matplotlib with PDF support.",
                    )
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to export PDF: {str(e)}")

    def _toggle_auto_save(self) -> None:
        """Toggle auto-save feature."""
        self.auto_save = self.auto_save_var.get()
        status = "enabled" if self.auto_save else "disabled"
        self._log_event(f"Auto-save {status}")

        if self.auto_save:
            # Start auto-save timer (save every 5 minutes)
            self._start_auto_save_timer()
        else:
            # Stop auto-save timer
            self._stop_auto_save_timer()

    def _start_auto_save_timer(self) -> None:
        """Start the auto-save timer."""
        if hasattr(self, "auto_save_timer") and self.auto_save_timer is not None:
            self.root.after_cancel(self.auto_save_timer)

        # Save every 5 minutes (300,000 ms)
        self.auto_save_timer = self.root.after(300000, self._auto_save_timer_callback)

    def _stop_auto_save_timer(self) -> None:
        """Stop the auto-save timer."""
        if hasattr(self, "auto_save_timer") and self.auto_save_timer is not None:
            self.root.after_cancel(self.auto_save_timer)
            self.auto_save_timer = None

    def _auto_save_timer_callback(self) -> None:
        """Callback for auto-save timer."""
        if self.auto_save:
            self._auto_save_data()
            # Schedule next save
            self.auto_save_timer = self.root.after(300000, self._auto_save_timer_callback)

    def _toggle_control_panel(self) -> None:
        """Toggle control panel visibility."""
        visible = self.view_vars["control_panel"].get()
        if visible:
            # Show control panel
            if self.left_panel not in self.main_paned.panes():  # type: ignore
                self.main_paned.insert(0, self.left_panel, weight=0)
        else:
            # Hide control panel
            if self.left_panel in self.main_paned.panes():  # type: ignore
                self.main_paned.remove(self.left_panel)
        self._log_event(f"Control panel {'shown' if visible else 'hidden'}")

    def _toggle_neural_activity(self) -> None:
        """Toggle neural activity panel visibility."""
        visible = self.view_vars["neural_activity"].get()
        tab_index = 0
        if visible:
            # Show tab
            if self.tab_frames[tab_index] not in self.notebook.tabs():  # type: ignore[no-untyped-call]
                self.notebook.add(self.tab_frames[tab_index], text=self.tab_texts[tab_index])
        else:
            # Hide tab
            if self.tab_frames[tab_index] in self.notebook.tabs():  # type: ignore[no-untyped-call]
                self.notebook.hide(self.tab_frames[tab_index])
        self._log_event(f"Neural activity panel {'shown' if visible else 'hidden'}")

    def _toggle_interoception(self) -> None:
        """Toggle interoception panel visibility."""
        visible = self.view_vars["interoception"].get()
        tab_index = 1
        if visible:
            # Show tab
            if self.tab_frames[tab_index] not in self.notebook.tabs():  # type: ignore[no-untyped-call]
                self.notebook.add(self.tab_frames[tab_index], text=self.tab_texts[tab_index])
        else:
            # Hide tab
            if self.tab_frames[tab_index] in self.notebook.tabs():  # type: ignore[no-untyped-call]
                self.notebook.hide(self.tab_frames[tab_index])
        self._log_event(f"Interoception panel {'shown' if visible else 'hidden'}")

    def _toggle_system_metrics(self) -> None:
        """Toggle system metrics panel visibility."""
        visible = self.view_vars["system_metrics"].get()
        tab_index = 2
        if visible:
            # Show tab
            if self.tab_frames[tab_index] not in self.notebook.tabs():  # type: ignore[no-untyped-call]
                self.notebook.add(self.tab_frames[tab_index], text=self.tab_texts[tab_index])
        else:
            # Hide tab
            if self.tab_frames[tab_index] in self.notebook.tabs():  # type: ignore[no-untyped-call]
                self.notebook.hide(self.tab_frames[tab_index])
        self._log_event(f"System metrics panel {'shown' if visible else 'hidden'}")

    def _auto_save_data(self) -> None:
        """Auto-save data to file (thread-safe)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = get_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)

        # Check if directory is writable
        import os

        if not os.access(data_dir, os.W_OK):
            self._log_event("Auto-save failed: Data directory not writable")
            return

        filename = data_dir / f"apgi_autosave_{timestamp}.json"
        try:
            with self.log_lock:
                log_data_copy = list(self.log_data)  # Convert deque to list for export

            # Collect full simulation state
            state = {
                "log_data": log_data_copy,
                "param_vars": {
                    k: v.get() if hasattr(v, "get") else v for k, v in self.param_vars.items()
                },
                "buffer_size": self.buffer_size,
                "data_buffers": {k: list(v) for k, v in self.data_buffers.items()},
                "time_buffer": list(self.time_buffer),
                "is_running": self.is_running,
                "is_paused": self.is_paused,
                "theme": self.theme_manager.current_theme,
                "auto_save_enabled": self.auto_save,
            }

            with open(filename, "w") as f:
                json.dump(state, f)
        except Exception as e:
            self._log_event(f"Auto-save failed: {str(e)}")

    def _cleanup_auto_save_files(self) -> None:
        """Clean up old auto-save files on exit."""
        try:
            data_dir = get_data_dir()
            if not data_dir.exists():
                return

            # Find all auto-save files
            autosave_files = list(data_dir.glob("apgi_autosave_*.json"))

            if len(autosave_files) <= 5:  # Keep at least 5 recent files
                return

            # Sort by modification time (newest first)
            autosave_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            # Remove older files, keeping only the 5 most recent
            files_to_remove = autosave_files[5:]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                except Exception as e:
                    # Log error but don't fail if cleanup fails
                    print(f"Failed to remove auto-save file {file_path}: {str(e)}")

        except Exception as e:
            # Log error but don't fail shutdown
            print(f"Error during auto-save cleanup: {str(e)}")
        """Clean up all toplevel windows to prevent Tkinter warnings."""
        try:
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Toplevel):
                    try:
                        widget.destroy()
                    except Exception as e:
                        logger.error(f"Error: {e}")
        except Exception as e:
            logger.error(f"Error: {e}")

    def _edit_parameters(self) -> None:
        """Open parameter editor dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit System Parameters")
        dialog.geometry("600x700")

        # Make dialog modal
        dialog.transient(self.root)
        # dialog.grab_set()  # Removed to prevent blocking simulation updates

        # Create main frame with scrollbar
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        ttk.Label(dialog, text="System Parameters", font=("Arial", 14, "bold")).pack(pady=10)

        # Parameter definitions
        parameters = [
            ("Ignition Threshold", "baseline_threshold", 1.0, 5.0, 2.0),
            ("Extero Precision", "extero_precision", 0.1, 10.0, 1.0),
            ("Intero Precision", "intero_precision", 0.1, 10.0, 0.8),
            ("Arousal Level", "arousal", 0.0, 1.0, 0.0),
            ("Stress Level", "stress", 0.0, 1.0, 0.0),
            ("Activity Level", "activity", 0.0, 1.0, 0.0),
            ("Learning Rate", "learning_rate", 0.001, 0.1, 0.01),
            ("Attention Gain", "attention_gain", 0.5, 3.0, 1.0),
        ]

        # Create parameter editors
        param_vars = {}
        for i, (label, key, min_val, max_val, default) in enumerate(parameters):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=5, padx=5)

            ttk.Label(frame, text=label, width=18).pack(side=tk.LEFT)

            # Get current value if it exists, otherwise use default
            current_val = default
            if hasattr(self, "param_vars") and key in self.param_vars:
                current_val = self.param_vars[key].get()

            var = tk.DoubleVar(value=current_val)
            param_vars[key] = var

            scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            val_label = ttk.Label(frame, text=f"{current_val:.3f}", width=8)
            val_label.pack(side=tk.LEFT)

            var.trace_add(
                "write", lambda *args, v=var, label=val_label: label.config(text=f"{v.get():.3f}")
            )

        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, pady=10)

        def apply_parameters() -> None:
            """Apply the edited parameters to the main system."""
            if hasattr(self, "param_vars"):
                for key, var in param_vars.items():
                    self.param_vars[key].set(var.get())
                self._apply_parameters()
                messagebox.showinfo("Success", "Parameters applied successfully!")
            else:
                messagebox.showwarning("Warning", "No active parameter system found.")

        def reset_to_defaults() -> None:
            """Reset all parameters to their default values."""
            for i, (label, key, min_val, max_val, default) in enumerate(parameters):
                param_vars[key].set(default)

        ttk.Button(button_frame, text="Apply", command=apply_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=reset_to_defaults).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

        # Pack scrollbar and canvas
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _edit_precision(self) -> None:
        """Edit precision settings."""
        if not self.apgi_simulation:
            messagebox.showwarning(
                "Warning", "No active APGI system found. Please initialize the system first."
            )
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Precision Settings")
        dialog.geometry("400x300")

        # Make dialog modal
        dialog.transient(self.root)
        # dialog.grab_set()  # Removed to prevent blocking simulation updates

        # Precision settings frame
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Precision Parameters", font=("Arial", 12, "bold")).pack(pady=10)

        # Current values display
        current_extero = self.apgi_simulation.precision.extero_baseline
        current_intero = self.apgi_simulation.precision.intero_baseline

        ttk.Label(frame, text=f"Current Exteroceptive Precision: {float(current_extero):.2f}").pack(
            pady=5
        )
        ttk.Label(frame, text=f"Current Interoceptive Precision: {float(current_intero):.2f}").pack(
            pady=5
        )

        # New value inputs
        ttk.Label(frame, text="New Exteroceptive Precision:").pack(pady=(20, 5))
        extero_var = tk.DoubleVar(value=float(current_extero))
        extero_scale = ttk.Scale(
            frame, from_=0.1, to=5.0, variable=extero_var, orient=tk.HORIZONTAL, length=300
        )
        extero_scale.pack(pady=5)
        extero_label = ttk.Label(frame, text=f"{float(current_extero):.2f}")
        extero_label.pack()
        extero_var.trace("w", lambda *args: extero_label.config(text=f"{extero_var.get():.2f}"))

        ttk.Label(frame, text="New Interoceptive Precision:").pack(pady=(20, 5))
        intero_var = tk.DoubleVar(value=float(current_intero))
        intero_scale = ttk.Scale(
            frame, from_=0.1, to=5.0, variable=intero_var, orient=tk.HORIZONTAL, length=300
        )
        intero_scale.pack(pady=5)
        intero_label = ttk.Label(frame, text=f"{float(current_intero):.2f}")
        intero_label.pack()
        intero_var.trace("w", lambda *args: intero_label.config(text=f"{intero_var.get():.2f}"))

        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=20)

        def apply_precision() -> None:
            try:
                self.param_vars["extero_precision"].set(extero_var.get())
                self.param_vars["intero_precision"].set(intero_var.get())
                self._apply_parameters()
                self._log_event(
                    f"Precision updated: Extero={extero_var.get():.2f}, Intero={intero_var.get():.2f}"
                )
                messagebox.showinfo("Success", "Precision settings updated successfully!")
                dialog.destroy()
            except Exception:
                messagebox.showerror("Error", "Failed to update precision")

        ttk.Button(button_frame, text="Apply", command=apply_precision).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def _edit_threshold(self) -> None:
        """Edit ignition threshold."""
        if not self.apgi_simulation:
            messagebox.showwarning(
                "Warning", "No active APGI system found. Please initialize the system first."
            )
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Ignition Threshold Settings")
        dialog.geometry("400x250")

        # Make dialog modal
        dialog.transient(self.root)
        # dialog.grab_set()  # Removed to prevent blocking simulation updates

        # Threshold settings frame
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Ignition Threshold", font=("Arial", 12, "bold")).pack(pady=10)

        # Current value display
        current_threshold = self.apgi_simulation.ignition_threshold.baseline_threshold
        ttk.Label(frame, text=f"Current Baseline Threshold: {current_threshold:.2f}").pack(pady=5)

        # Info label
        info_text = "Higher values make ignition less likely (more conservative)\nLower values make ignition more likely (more sensitive)"
        ttk.Label(frame, text=info_text, font=("Arial", 9), foreground="gray").pack(pady=10)

        # New value input
        ttk.Label(frame, text="New Baseline Threshold:").pack(pady=(20, 5))
        threshold_var = tk.DoubleVar(value=current_threshold)
        threshold_scale = ttk.Scale(
            frame, from_=0.5, to=5.0, variable=threshold_var, orient=tk.HORIZONTAL, length=300
        )
        threshold_scale.pack(pady=5)
        threshold_label = ttk.Label(frame, text=f"{current_threshold:.2f}")
        threshold_label.pack()
        threshold_var.trace(
            "w", lambda *args: threshold_label.config(text=f"{threshold_var.get():.2f}")
        )

        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=20)

        def apply_threshold() -> None:
            try:
                self.param_vars["baseline_threshold"].set(threshold_var.get())
                self._apply_parameters()
                self._log_event(f"Ignition threshold updated: {threshold_var.get():.2f}")
                messagebox.showinfo("Success", "Ignition threshold updated successfully!")
                dialog.destroy()
            except Exception:
                messagebox.showerror("Error", "Failed to update threshold")

        ttk.Button(button_frame, text="Apply", command=apply_threshold).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def _reset_defaults(self) -> None:
        """Reset all parameters to defaults."""
        if messagebox.askyesno("Reset", "Reset all parameters to default values?"):
            # Reset parameter sliders
            for key, var in self.param_vars.items():
                # Set to default values (simplified)
                if key == "baseline_threshold":
                    var.set(2.0)
                elif key in ["extero_precision", "intero_precision"]:
                    var.set(1.0)
                else:
                    var.set(0.0)

            self._log_event("Parameters reset to defaults")

    def _run_preset_task(self) -> None:
        """Run preset experimental task."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Run Preset Task")
        dialog.geometry("500x400")

        ttk.Label(dialog, text="Select Preset Task:", font=("Arial", 12, "bold")).pack(pady=10)

        tasks = [
            "Attentional Blink",
            "Change Blindness",
            "Binocular Rivalry",
            "Iowa Gambling Task",
            "Masking Paradigm",
            "Stroop Task",
            "N-Back Task",
        ]

        listbox = tk.Listbox(dialog, height=len(tasks))
        for task in tasks:
            listbox.insert(tk.END, task)
        listbox.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        # Task configuration frame
        config_frame = ttk.LabelFrame(dialog, text="Task Configuration", padding=10)
        config_frame.pack(pady=5, padx=20, fill=tk.BOTH)

        ttk.Label(config_frame, text="Trials per condition:").grid(row=0, column=0, sticky=tk.W)
        trials_var = tk.IntVar(value=20)
        ttk.Spinbox(config_frame, from_=5, to=100, textvariable=trials_var, width=10).grid(
            row=0, column=1
        )

        # Create Run Task button (initially disabled)
        run_button = ttk.Button(dialog, text="Run Task", state=tk.DISABLED)
        run_button.pack(pady=5)

        def on_selection_change(event: Any) -> None:
            """Enable Run Task button when a task is selected."""
            if listbox.curselection():  # type: ignore[no-untyped-call]
                run_button.config(state=tk.NORMAL)
            else:
                run_button.config(state=tk.DISABLED)

        # Bind selection change event
        listbox.bind("<<ListboxSelect>>", on_selection_change)

        def run_selected() -> None:
            selection = listbox.curselection()  # type: ignore[no-untyped-call]
            if selection:
                task_name = tasks[selection[0]]

                if task_name == "Attentional Blink":
                    dialog.destroy()
                    self._run_attentional_blink_task(num_trials=trials_var.get())
                elif task_name == "Change Blindness":
                    dialog.destroy()
                    self._run_change_blindness_task(num_trials=trials_var.get())
                elif task_name == "Binocular Rivalry":
                    dialog.destroy()
                    self._run_binocular_rivalry_task(num_trials=trials_var.get())
                elif task_name == "Iowa Gambling Task":
                    dialog.destroy()
                    self._run_iowa_gambling_task(num_trials=trials_var.get())
                elif task_name == "Masking Paradigm":
                    dialog.destroy()
                    self._run_masking_paradigm_task(trials_per_condition=trials_var.get())
                elif task_name == "Stroop Task":
                    dialog.destroy()
                    self._run_stroop_task(num_trials=trials_var.get())
                elif task_name == "N-Back Task":
                    dialog.destroy()
                    self._run_nback_task(num_trials=trials_var.get())
                else:
                    self._log_event(f"Task not yet implemented: {task_name}")
                    messagebox.showinfo(
                        "Coming Soon",
                        f"{task_name} implementation coming soon!\n\n"
                        "Currently available:\n- Attentional Blink\n- Change Blindness\n- Binocular Rivalry\n- Iowa Gambling Task\n- Masking Paradigm\n- Stroop Task\n- N-Back Task",
                    )

        run_button.config(command=run_selected)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=5)

    def _run_attentional_blink_task(self, num_trials: int = 20) -> None:
        """Run the Attentional Blink experimental task."""
        if not self.apgi_simulation:
            messagebox.showerror("Error", "System not initialized")
            return

        # Stop current simulation if running
        was_running = self.is_running
        if was_running:
            self._stop_simulation()

        self._log_event("Starting Attentional Blink task...")
        self._update_status("Running Attentional Blink task...")

        # Import and create task
        try:
            from apgi_framework.experiments.tasks import AttentionalBlinkTask

            # Create task with specified parameters
            task = AttentionalBlinkTask(
                stream_length=15,
                stimulus_duration_ms=100,
                num_trials_per_lag=num_trials,
                lags=[1, 2, 3, 4, 8],
            )

            # Create progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Running Attentional Blink Task")
            progress_dialog.geometry("400x200")

            ttk.Label(progress_dialog, text="Running trials...", font=("Arial", 12, "bold")).pack(
                pady=10
            )

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                progress_dialog, length=300, mode="determinate", variable=progress_var
            )
            progress_bar.pack(pady=10)

            status_label = ttk.Label(progress_dialog, text="Trial 0 of 0")
            status_label.pack(pady=5)

            # Results text area
            results_text = scrolledtext.ScrolledText(progress_dialog, height=6, width=50)
            results_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

            # Run task in separate thread
            def run_task_thread() -> None:
                total_trials = len(task.trials)

                for trial_idx, trial in enumerate(task.trials):
                    # Update progress
                    progress = (trial_idx / total_trials) * 100
                    self.root.after(
                        0,
                        lambda p=progress, i=trial_idx, t=total_trials: (  # type: ignore[misc]
                            progress_var.set(p),  # type: ignore[func-returns-value]
                            (
                                status_label.config(text=f"Trial {i + 1} of {t}")
                                if status_label.winfo_exists()
                                else None
                            ),
                        ),
                    )

                    # Run trial with system lock
                    with self.system_lock:
                        result = task.run_trial(self.apgi_simulation, trial)

                    # Log result
                    if trial_idx % 10 == 0:
                        msg = f"Trial {trial_idx}: Lag {result['lag']}, T1: {result['t1_detected']}, T2: {result['t2_detected']}\n"
                        self.root.after(
                            0,
                            lambda m=msg, rt=results_text: (  # type: ignore[misc]
                                rt.insert(tk.END, m) if rt.winfo_exists() else None
                            ),
                        )
                        self.root.after(
                            0, lambda rt=results_text: rt.see(tk.END) if rt.winfo_exists() else None  # type: ignore[misc]
                        )

                # Task complete
                self.root.after(0, lambda pv=progress_var: pv.set(100))  # type: ignore[misc]
                self.root.after(
                    0,
                    lambda sl=status_label: (  # type: ignore[misc]
                        sl.config(text="Analysis complete!") if sl.winfo_exists() else None
                    ),
                )

                # Analyze results
                analysis = task.analyze_results()

                # Display summary
                summary = f"\n{'=' * 50}\nRESULTS:\n{'=' * 50}\n"
                summary += f"Total Trials: {analysis['total_trials']}\n"
                summary += f"Overall T1 Accuracy: {analysis['overall_t1_accuracy']:.1%}\n"
                summary += f"Overall T2 Accuracy: {analysis['overall_t2_accuracy']:.1%}\n"
                summary += f"Overall Blink Rate: {analysis['overall_blink_rate']:.1%}\n"
                summary += f"Peak Blink at Lag {analysis['max_blink_lag']}: {analysis['max_blink_rate']:.1%}\n\n"

                summary += "By Lag:\n"
                for lag in sorted(analysis["lag_analysis"].keys()):
                    lag_data = analysis["lag_analysis"][lag]
                    summary += f"  Lag {lag}: T2|T1={lag_data['t2_given_t1_accuracy']:.1%}, Blink={lag_data['blink_rate']:.1%}\n"

                self.root.after(
                    0,
                    lambda s=summary, rt=results_text: (  # type: ignore[misc]
                        rt.insert(tk.END, s) if rt.winfo_exists() else None
                    ),
                )
                self.root.after(
                    0, lambda rt=results_text: rt.see(tk.END) if rt.winfo_exists() else None  # type: ignore[misc]
                )

                # Save results
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"attentional_blink_results_{timestamp}.json"
                task.save_results(filename)

                self.root.after(
                    0, lambda: self._log_event(f"Task complete. Results saved to {filename}")
                )

                # Add close button
                def close_and_show_results() -> None:
                    progress_dialog.destroy()
                    task.print_results(analysis)
                    messagebox.showinfo(
                        "Task Complete",
                        f"Attentional Blink task completed!\n\n"
                        f"T1 Accuracy: {analysis['overall_t1_accuracy']:.1%}\n"
                        f"T2 Accuracy: {analysis['overall_t2_accuracy']:.1%}\n"
                        f"Blink Rate: {analysis['overall_blink_rate']:.1%}\n\n"
                        f"Results saved to:\n{filename}",
                    )

                self.root.after(
                    0,
                    lambda: ttk.Button(
                        progress_dialog, text="Close", command=close_and_show_results
                    ).pack(pady=5),
                )

            # Start task thread
            import threading

            task_thread = threading.Thread(target=run_task_thread, daemon=True)
            task_thread.start()

        except Exception:
            self._log_event("ERROR running task")
            messagebox.showerror("Task Error", "Failed to run task")
            import traceback

            traceback.print_exc()

    def _run_change_blindness_task(self, num_trials: int = 10) -> None:
        """Run the Change Blindness experimental task."""
        if not self.apgi_simulation:
            messagebox.showerror("Error", "System not initialized")
            return

        # Stop current simulation if running
        was_running = self.is_running
        if was_running:
            self._stop_simulation()

        self._log_event("Starting Change Blindness task...")
        self._update_status("Running Change Blindness task...")

        # Import and create task
        try:
            from apgi_framework.experiments.tasks import ChangeBlindnessTask

            # Create task with specified parameters
            task = ChangeBlindnessTask(
                display_size=(512, 512),
                max_alternations=20,
                num_trials_per_condition=num_trials,
                change_magnitudes=[0.3, 0.5, 0.8],
                presentation_duration_ms=240.0,
                blank_duration_ms=80.0,
            )

            # Create progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Running Change Blindness Task")
            progress_dialog.geometry("400x200")

            ttk.Label(progress_dialog, text="Running trials...", font=("Arial", 12, "bold")).pack(
                pady=10
            )

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                progress_dialog, length=300, mode="determinate", variable=progress_var
            )
            progress_bar.pack(pady=10)

            status_label = ttk.Label(progress_dialog, text="Trial 0 of 0")
            status_label.pack(pady=5)

            # Results text area
            results_text = scrolledtext.ScrolledText(progress_dialog, height=6, width=50)
            results_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

            # Run task in separate thread
            def run_task_thread() -> None:
                total_trials = len(task.trials)

                for trial_idx, trial in enumerate(task.trials):
                    # Update progress
                    progress = (trial_idx / total_trials) * 100
                    self.root.after(
                        0,
                        lambda p=progress, i=trial_idx, t=total_trials: (  # type: ignore[misc]
                            progress_var.set(p),  # type: ignore[func-returns-value]
                            (
                                status_label.config(text=f"Trial {i + 1} of {t}")
                                if status_label.winfo_exists()
                                else None
                            ),
                        ),
                    )

                    # Run trial with system lock
                    with self.system_lock:
                        result = task.run_trial(self.apgi_simulation, trial)

                    # Log result
                    if trial_idx % 10 == 0:
                        detected_str = "Yes" if result["change_detected"] else "No"
                        msg = f"Trial {trial_idx}: {result['change_type'].value}, Detected: {detected_str}\n"
                        self.root.after(
                            0,
                            lambda m=msg, rt=results_text: (  # type: ignore[misc]
                                rt.insert(tk.END, m) if rt.winfo_exists() else None
                            ),
                        )
                        self.root.after(
                            0, lambda rt=results_text: rt.see(tk.END) if rt.winfo_exists() else None  # type: ignore[misc]
                        )

                # Task complete
                self.root.after(
                    0, lambda pv=progress_var: pv.set(100) if pv.winfo_exists() else None  # type: ignore[misc]
                )
                self.root.after(
                    0,
                    lambda sl=status_label: (  # type: ignore[misc]
                        sl.config(text="Analysis complete!") if sl.winfo_exists() else None
                    ),
                )

                # Analyze results
                analysis = task.analyze_results()

                # Display summary
                summary = f"\n{'=' * 50}\nRESULTS:\n{'=' * 50}\n"
                summary += f"Total Trials: {analysis['total_trials']}\n"
                summary += f"Detection Rate: {analysis['overall_detection_rate']:.1%}\n"
                summary += f"Blindness Rate: {analysis['overall_blindness_rate']:.1%}\n"
                summary += f"Avg Alternations: {analysis['overall_avg_alternations']:.1f}\n"
                summary += f"Avg Time to Detection: {analysis['overall_avg_time_ms']:.0f} ms\n\n"

                summary += "By Change Type:\n"
                for change_type, data in analysis["by_change_type"].items():
                    summary += f"  {change_type}: {data['detection_rate']:.1%}\n"

                summary += "\nBy Magnitude:\n"
                for magnitude in sorted(analysis["by_magnitude"].keys()):
                    data = analysis["by_magnitude"][magnitude]
                    summary += f"  {magnitude:.1f}: {data['detection_rate']:.1%}\n"

                self.root.after(
                    0,
                    lambda s=summary, rt=results_text: (  # type: ignore[misc]
                        rt.insert(tk.END, s) if rt.winfo_exists() else None
                    ),
                )
                self.root.after(
                    0, lambda rt=results_text: rt.see(tk.END) if rt.winfo_exists() else None  # type: ignore[misc]
                )

                # Save results
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"change_blindness_results_{timestamp}.json"
                task.save_results(filename)

                self.root.after(
                    0, lambda: self._log_event(f"Task complete. Results saved to {filename}")
                )

                # Add close button
                def close_and_show_results() -> None:
                    progress_dialog.destroy()
                    task.print_results(analysis)
                    messagebox.showinfo(
                        "Task Complete",
                        f"Change Blindness task completed!\n\n"
                        f"Detection Rate: {analysis['overall_detection_rate']:.1%}\n"
                        f"Blindness Rate: {analysis['overall_blindness_rate']:.1%}\n"
                        f"Avg Alternations: {analysis['overall_avg_alternations']:.1f}\n\n"
                        f"Results saved to:\n{filename}",
                    )

                self.root.after(
                    0,
                    lambda: ttk.Button(
                        progress_dialog, text="Close", command=close_and_show_results
                    ).pack(pady=5),
                )

            # Start task thread
            import threading

            task_thread = threading.Thread(target=run_task_thread, daemon=True)
            task_thread.start()

        except Exception:
            self._log_event("ERROR running task")
            messagebox.showerror("Task Error", "Failed to run task")
            import traceback

            traceback.print_exc()

    def _run_binocular_rivalry_task(self, num_trials: int = 10) -> None:
        """Run the Binocular Rivalry experimental task."""
        if not self.apgi_simulation:
            messagebox.showerror("Error", "System not initialized")
            return

        # Stop current simulation if running
        was_running = self.is_running
        if was_running:
            self._stop_simulation()

        self._log_event("Starting Binocular Rivalry task...")
        self._update_status("Running Binocular Rivalry task...")

        # Import and create task
        try:
            from apgi_framework.experiments.tasks import BinocularRivalryTask

            # Create task with specified parameters
            task = BinocularRivalryTask(
                image_size=(256, 256),
                trial_duration_ms=30000.0,  # 30 seconds per trial
                num_trials=num_trials,
                strength_ratios=[(1.0, 1.0), (1.0, 0.8), (1.2, 1.0)],
                sampling_interval_ms=100.0,
            )

            # Create progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Running Binocular Rivalry Task")
            progress_dialog.geometry("400x200")

            ttk.Label(progress_dialog, text="Running trials...", font=("Arial", 12, "bold")).pack(
                pady=10
            )

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                progress_dialog, length=300, mode="determinate", variable=progress_var
            )
            progress_bar.pack(pady=10)

            status_label = ttk.Label(progress_dialog, text="Trial 0 of 0")
            status_label.pack(pady=5)

            # Results text area
            results_text = scrolledtext.ScrolledText(progress_dialog, height=6, width=50)
            results_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

            # Run task in separate thread
            def run_task_thread() -> None:
                total_trials = len(task.trials)

                for trial_idx, trial in enumerate(task.trials):
                    # Update progress
                    progress = (trial_idx / total_trials) * 100
                    self.root.after(
                        0,
                        lambda p=progress, i=trial_idx, t=total_trials: (  # type: ignore[misc]
                            progress_var.set(p),  # type: ignore[func-returns-value]
                            (
                                status_label.config(text=f"Trial {i + 1} of {t}")
                                if status_label.winfo_exists()
                                else None
                            ),
                        ),
                    )

                    # Run trial with system lock
                    with self.system_lock:
                        result = task.run_trial(self.apgi_simulation, trial)

                    # Log result
                    if trial_idx % 5 == 0:
                        msg = (
                            f"Trial {trial_idx}: {result['num_alternations']} alternations, "
                            f"Pattern A dominance: {result['pattern_a_dominance_ratio']:.1%}\n"
                        )
                        self.root.after(
                            0,
                            lambda m=msg, rt=results_text: (  # type: ignore[misc]
                                rt.insert(tk.END, m) if rt.winfo_exists() else None
                            ),
                        )
                        self.root.after(
                            0, lambda rt=results_text: rt.see(tk.END) if rt.winfo_exists() else None  # type: ignore[misc]
                        )

                # Task complete
                self.root.after(0, lambda pv=progress_var: pv.set(100))  # type: ignore[misc]
                self.root.after(
                    0,
                    lambda sl=status_label: (  # type: ignore[misc]
                        sl.config(text="Analysis complete!") if sl.winfo_exists() else None
                    ),
                )

                # Analyze results
                analysis = task.analyze_results()

                # Display summary
                summary = f"\n{'=' * 50}\nRESULTS:\n{'=' * 50}\n"
                summary += f"Total Trials: {analysis['total_trials']}\n"
                summary += (
                    f"Avg Dominance Duration: {analysis['avg_dominance_duration_ms']:.0f} ms\n"
                )
                summary += (
                    f"Avg Alternation Rate: {analysis['avg_alternation_rate']:.2f} per second\n"
                )
                summary += f"Pattern A Dominance: {analysis['avg_pattern_a_dominance_ratio']:.1%}\n"
                summary += f"Total Alternations: {analysis['total_alternations']}\n\n"

                summary += "By Strength Ratio:\n"
                for ratio_key in sorted(analysis["by_strength_ratio"].keys()):
                    data = analysis["by_strength_ratio"][ratio_key]
                    summary += (
                        f"  {ratio_key}: Dom={data['avg_dominance_duration_ms']:.0f}ms, "
                        f"Alt={data['avg_alternation_rate']:.2f}/s\n"
                    )

                self.root.after(
                    0,
                    lambda s=summary, rt=results_text: (  # type: ignore[misc]
                        rt.insert(tk.END, s) if rt.winfo_exists() else None
                    ),
                )
                self.root.after(
                    0, lambda rt=results_text: rt.see(tk.END) if rt.winfo_exists() else None  # type: ignore[misc]
                )

                # Save results
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"binocular_rivalry_results_{timestamp}.json"
                task.save_results(filename)

                self.root.after(
                    0, lambda: self._log_event(f"Task complete. Results saved to {filename}")
                )

                # Add close button
                def close_and_show_results() -> None:
                    progress_dialog.destroy()
                    task.print_results(analysis)
                    messagebox.showinfo(
                        "Task Complete",
                        f"Binocular Rivalry task completed!\n\n"
                        f"Avg Dominance Duration: {analysis['avg_dominance_duration_ms']:.0f} ms\n"
                        f"Alternation Rate: {analysis['avg_alternation_rate']:.2f} per second\n"
                        f"Total Alternations: {analysis['total_alternations']}\n\n"
                        f"Results saved to:\n{filename}",
                    )

                self.root.after(
                    0,
                    lambda: ttk.Button(
                        progress_dialog, text="Close", command=close_and_show_results
                    ).pack(pady=5),
                )

            # Start task thread
            import threading

            task_thread = threading.Thread(target=run_task_thread, daemon=True)
            task_thread.start()

        except Exception:
            self._log_event("ERROR running task")
            messagebox.showerror("Task Error", "Failed to run task")
            import traceback

            traceback.print_exc()

    def _run_masking_paradigm_task(self, trials_per_condition: int = 20) -> None:
        """Run the Masking Paradigm experimental task."""
        if not self.apgi_simulation:
            messagebox.showerror("Error", "System not initialized")
            return

        was_running = self.is_running
        if was_running:
            self._stop_simulation()

        self._log_event("Starting Masking Paradigm task...")
        self._update_status("Running Masking Paradigm task...")

        try:
            from apgi_framework.experiments.tasks import MaskingParadigmTask

            task = MaskingParadigmTask(num_trials_per_condition=trials_per_condition)

            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Running Masking Paradigm Task")
            progress_dialog.geometry("420x240")

            ttk.Label(progress_dialog, text="Running trials...", font=("Arial", 12, "bold")).pack(
                pady=10
            )

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                progress_dialog, length=320, mode="determinate", variable=progress_var
            )
            progress_bar.pack(pady=10)

            status_label = ttk.Label(progress_dialog, text="Trial 0 of 0")
            status_label.pack(pady=5)

            results_text = scrolledtext.ScrolledText(progress_dialog, height=6, width=60)
            results_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

            def run_task_thread() -> None:
                total_trials = len(task.trials)

                for trial_idx, trial in enumerate(task.trials):
                    progress = (trial_idx / total_trials) * 100 if total_trials else 0
                    self.root.after(
                        0,
                        lambda p=progress, i=trial_idx, t=total_trials: (  # type: ignore[misc]
                            progress_var.set(p),  # type: ignore[func-returns-value]
                            (
                                status_label.config(text=f"Trial {i + 1} of {t}")
                                if status_label.winfo_exists()
                                else None
                            ),
                        ),
                    )

                    # Run trial with system lock
                    with self.system_lock:
                        result = task.run_trial(self.apgi_simulation, trial)

                    if trial_idx % 10 == 0:
                        detected = "Yes" if result["target_detected"] else "No"
                        msg = (
                            f"Trial {trial_idx}: SOA {result['soa_ms']}ms, Detected: {detected}, "
                            f"Ignitions: {result['ignition_count']}\n"
                        )
                        self.root.after(
                            0,
                            lambda m=msg, rt=results_text: (  # type: ignore[misc]
                                rt.insert(tk.END, m) if rt.winfo_exists() else None
                            ),
                        )
                        self.root.after(
                            0, lambda rt=results_text: rt.see(tk.END) if rt.winfo_exists() else None  # type: ignore[misc]
                        )
                self.root.after(0, lambda pv=progress_var: pv.set(100))  # type: ignore[misc]
                self.root.after(
                    0,
                    lambda sl=status_label: (  # type: ignore[misc]
                        sl.config(text="Analysis complete!") if sl.winfo_exists() else None
                    ),
                )

                analysis = task.analyze_results(task.results)

                summary = f"\n{'=' * 50}\nRESULTS:\n{'=' * 50}\n"
                summary += f"Total Trials: {analysis['total_trials']}\n"
                summary += f"Overall Detection Rate: {analysis['overall_detection_rate']:.1%}\n"
                summary += f"Overall Suppression Rate: {analysis['overall_suppression_rate']:.1%}\n"
                summary += f"Masking Effect: {analysis['masking_effect']:.1%}\n\n"
                summary += "By SOA:\n"
                for soa in sorted(analysis["by_soa"].keys()):
                    data = analysis["by_soa"][soa]
                    summary += (
                        f"  {soa:.0f}ms: Det={data['detection_rate']:.1%}, "
                        f"Supp={data['suppression_rate']:.1%}, "
                        f"Avg Strength={data['avg_ignition_strength']:.2f}\n"
                    )

                self.root.after(
                    0,
                    lambda s=summary, rt=results_text: (  # type: ignore[misc]
                        rt.insert(tk.END, s) if rt.winfo_exists() else None
                    ),
                )
                self.root.after(
                    0, lambda rt=results_text: rt.see(tk.END) if rt.winfo_exists() else None  # type: ignore[misc]
                )

                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"masking_paradigm_results_{timestamp}.json"
                task.save_results(task.results, filename)

                self.root.after(
                    0, lambda: self._log_event(f"Task complete. Results saved to {filename}")
                )

                def close_and_show_results() -> None:
                    progress_dialog.destroy()
                    # task.print_results(analysis)  # type: ignore[attr-defined]
                    messagebox.showinfo(
                        "Task Complete",
                        f"Masking Paradigm task completed!\n\n"
                        f"Detection Rate: {analysis['overall_detection_rate']:.1%}\n"
                        f"Suppression Rate: {analysis['overall_suppression_rate']:.1%}\n\n"
                        f"Results saved to:\n{filename}",
                    )

                self.root.after(
                    0,
                    lambda: ttk.Button(
                        progress_dialog, text="Close", command=close_and_show_results
                    ).pack(pady=5),
                )

            threading.Thread(target=run_task_thread, daemon=True).start()

        except Exception:
            self._log_event("ERROR running task")
            messagebox.showerror("Task Error", "Failed to run task")
            import traceback

            traceback.print_exc()

    def _run_iowa_gambling_task(self, num_trials: int = 100) -> None:
        """Run the Iowa Gambling Task experimental task."""
        if not self.apgi_simulation:
            messagebox.showerror("Error", "System not initialized")
            return

        # Stop current simulation if running
        was_running = self.is_running
        if was_running:
            self._stop_simulation()

        self._log_event("Starting Iowa Gambling Task...")
        self._update_status("Running Iowa Gambling Task...")

        # Import and create task
        try:
            from apgi_framework.experiments.tasks import IowaGamblingTask

            # Create task with specified parameters
            task = IowaGamblingTask(
                num_trials=num_trials,
                starting_balance=2000,
            )

            # Create progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Running Iowa Gambling Task")
            progress_dialog.geometry("500x250")

            ttk.Label(progress_dialog, text="Running trials...", font=("Arial", 12, "bold")).pack(
                pady=10
            )

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                progress_dialog, length=400, mode="determinate", variable=progress_var
            )
            progress_bar.pack(pady=10)

            status_label = ttk.Label(progress_dialog, text="Trial 0 of 0")
            status_label.pack(pady=5)

            # Results text area
            results_text = scrolledtext.ScrolledText(progress_dialog, height=8, width=60)
            results_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

            # Run task in separate thread
            def run_task_thread() -> None:
                total_trials = len(task.trials)

                for trial_idx, trial in enumerate(task.trials):
                    # Update progress
                    progress = (trial_idx / total_trials) * 100

                    def update_igt_progress(p: float, i: int, t: int) -> None:
                        progress_var.set(p)
                        if status_label.winfo_exists():
                            status_label.config(text=f"Trial {i + 1} of {t}")

                    self.root.after(
                        0,
                        lambda p=progress, i=trial_idx, t=total_trials: update_igt_progress(  # type: ignore[misc]
                            p, i, t
                        ),
                    )

                    # Run trial with system lock
                    with self.system_lock:
                        result = task.run_trial(self.apgi_simulation, trial)

                    # Log result
                    if trial_idx % 10 == 0:
                        deck = result["deck_choice"].value
                        net = result["net_outcome"]
                        balance = result["running_total"]
                        msg = (
                            f"Trial {trial_idx}: Deck {deck}, Net: ${net:+d}, Balance: ${balance}\n"
                        )

                        def log_igt_result(m: str, rt: "scrolledtext.ScrolledText") -> None:
                            if rt.winfo_exists():
                                rt.insert(tk.END, m)

                        self.root.after(0, lambda m=msg, rt=results_text: log_igt_result(m, rt))  # type: ignore[misc]

                        def scroll_igt_to_end(rt: "scrolledtext.ScrolledText") -> None:
                            if rt.winfo_exists():
                                rt.see(tk.END)

                        self.root.after(0, lambda rt=results_text: scroll_igt_to_end(rt))  # type: ignore[misc]

                # Task complete
                def set_igt_complete(pv: tk.DoubleVar) -> None:
                    try:
                        pv.set(100)
                    except tk.TclError:
                        pass  # Variable was destroyed

                self.root.after(0, lambda pv=progress_var: set_igt_complete(pv))  # type: ignore[misc]

                def set_igt_status_complete(sl: ttk.Label) -> None:
                    if sl.winfo_exists():
                        sl.config(text="Analysis complete!")

                self.root.after(0, lambda sl=status_label: set_igt_status_complete(sl))  # type: ignore[misc]

                # Analyze results
                analysis = task.analyze_results()

                # Display summary
                summary = f"\n{'=' * 50}\nRESULTS:\n{'=' * 50}\n"
                summary += f"Total Trials: {analysis['total_trials']}\n"
                summary += f"Final Balance: ${analysis['final_balance']}\n"
                summary += f"Total Earnings: ${analysis['total_earnings']}\n"
                summary += f"Good Deck Choices: {analysis['good_deck_choices']} ({analysis['advantageous_ratio']:.1%})\n"
                summary += f"Bad Deck Choices: {analysis['bad_deck_choices']}\n"
                summary += f"Net Score: {analysis['net_score']}\n\n"

                summary += "Deck Selections:\n"
                for deck in ["A", "B", "C", "D"]:
                    deck_data = analysis["by_deck"][deck]
                    deck_type = "Bad" if deck in ["A", "B"] else "Good"
                    summary += f"  Deck {deck} ({deck_type}): {deck_data['selections']} ({deck_data['selection_percentage']:.1f}%)\n"

                def insert_igt_summary(s: str, rt: "scrolledtext.ScrolledText") -> None:
                    if rt.winfo_exists():
                        rt.insert(tk.END, s)

                self.root.after(0, lambda s=summary, rt=results_text: insert_igt_summary(s, rt))  # type: ignore[misc]

                def scroll_igt_summary(rt: "scrolledtext.ScrolledText") -> None:
                    if rt.winfo_exists():
                        rt.see(tk.END)

                self.root.after(0, lambda rt=results_text: scroll_igt_summary(rt))  # type: ignore[misc]

                # Save results
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"iowa_gambling_results_{timestamp}.json"
                task.save_results(filename)

                self.root.after(
                    0, lambda: self._log_event(f"Task complete. Results saved to {filename}")
                )

                # Add close button
                def close_and_show_results() -> None:
                    progress_dialog.destroy()
                    task.print_results(analysis)
                    messagebox.showinfo(
                        "Task Complete",
                        f"Iowa Gambling Task completed!\n\n"
                        f"Total Earnings: ${analysis['total_earnings']}\n"
                        f"Good Deck Choices: {analysis['advantageous_ratio']:.1%}\n"
                        f"Net Score: {analysis['net_score']}\n\n"
                        f"Results saved to:\n{filename}",
                    )

                self.root.after(
                    0,
                    lambda: ttk.Button(
                        progress_dialog, text="Close", command=close_and_show_results
                    ).pack(pady=5),
                )

            import threading

            task_thread = threading.Thread(target=run_task_thread, daemon=True)
            task_thread.start()

        except Exception:
            self._log_event("ERROR running task")
            messagebox.showerror("Task Error", "Failed to run task")
            import traceback

            traceback.print_exc()

    def _run_stroop_task(self, num_trials: int = 60) -> None:
        """Run the Stroop Task experimental task."""
        if not self.apgi_simulation:
            messagebox.showerror("Error", "System not initialized")
            return

        # Stop current simulation if running
        was_running = self.is_running
        if was_running:
            self._stop_simulation()

        self._log_event("Starting Stroop Task...")
        self._update_status("Running Stroop Task...")

        # Import and create task
        try:
            from apgi_framework.experiments.tasks import StroopTask

            # Create task with specified parameters
            task = StroopTask(
                num_trials=num_trials,
            )

            # Create progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Running Stroop Task")
            progress_dialog.geometry("500x250")

            ttk.Label(progress_dialog, text="Running trials...", font=("Arial", 12, "bold")).pack(
                pady=10
            )

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                progress_dialog, length=400, mode="determinate", variable=progress_var
            )
            progress_bar.pack(pady=10)

            status_label = ttk.Label(progress_dialog, text="Trial 0 of 0")
            status_label.pack(pady=5)

            # Results text area
            results_text = scrolledtext.ScrolledText(progress_dialog, height=8, width=60)
            results_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

            def run_task_thread() -> None:
                """Run task in separate thread to keep GUI responsive."""
                try:
                    results = task.run_all_trials(self.apgi_simulation)

                    # Update progress and status
                    for i, result in enumerate(results):
                        progress = (i + 1) / len(results) * 100

                        def update_stroop_progress(p: float, pv: tk.DoubleVar) -> None:
                            pv.set(p)

                        self.root.after(
                            0, lambda p=progress, pv=progress_var: update_stroop_progress(p, pv)  # type: ignore[misc]
                        )

                        def update_stroop_status(idx: int, total: int, sl: ttk.Label) -> None:
                            if sl.winfo_exists():
                                sl.config(text=f"Trial {idx + 1} of {total}")

                        self.root.after(
                            0,
                            lambda i=i, total=len(results), sl=status_label: update_stroop_status(  # type: ignore[misc]
                                i, total, sl
                            ),
                        )

                        # Log trial results
                        def log_stroop_trial(
                            r: Dict[str, Any], rt: "scrolledtext.ScrolledText"
                        ) -> None:
                            if rt.winfo_exists():
                                rt.insert(
                                    tk.END,
                                    f"Trial {r['trial_number']}: {r['trial_type']} - RT: {r['response_time_ms']:.0f}ms - Correct: {r['is_correct']}\n",
                                )

                        self.root.after(
                            0, lambda r=result, rt=results_text: log_stroop_trial(r, rt)  # type: ignore[misc]
                        )

                        def scroll_stroop_to_end(rt: "scrolledtext.ScrolledText") -> None:
                            if rt.winfo_exists():
                                rt.see(tk.END)

                        self.root.after(0, lambda rt=results_text: scroll_stroop_to_end(rt))  # type: ignore[misc]

                        # Small delay to show progress
                        time.sleep(0.1)

                    # Complete progress
                    def set_stroop_complete(pv: tk.DoubleVar) -> None:
                        pv.set(100)

                    self.root.after(0, lambda pv=progress_var: set_stroop_complete(pv))  # type: ignore[misc]

                    def set_stroop_status_complete(sl: ttk.Label) -> None:
                        if sl.winfo_exists():
                            sl.config(text="Analysis complete!")

                    self.root.after(0, lambda sl=status_label: set_stroop_status_complete(sl))  # type: ignore[misc]

                    # Analyze results
                    analysis = task.analyze_results()

                    # Display summary
                    summary = f"\n{'=' * 50}\nRESULTS:\n{'=' * 50}\n"
                    summary += f"Total Trials: {analysis['total_trials']}\n"
                    summary += f"Overall Accuracy: {analysis['overall_accuracy']:.1%}\n"
                    summary += f"Mean RT: {analysis['mean_response_time_ms']:.0f}ms\n\n"

                    summary += "By Trial Type:\n"
                    for trial_type in ["congruent", "incongruent", "neutral"]:
                        if trial_type in analysis["by_trial_type"]:
                            data = analysis["by_trial_type"][trial_type]
                            summary += f"  {trial_type.capitalize()}: {data['accuracy']:.1%} accuracy, {data['mean_rt']:.0f}ms RT\n"

                    summary += f"\nStroop Effect: {analysis['stroop_effect_ms']:.0f}ms\n"
                    summary += f"Interference Score: {analysis['interference_score']:.2f}\n"

                    def insert_stroop_summary(s: str, rt: "scrolledtext.ScrolledText") -> None:
                        if rt.winfo_exists():
                            rt.insert(tk.END, s)

                    self.root.after(
                        0, lambda s=summary, rt=results_text: insert_stroop_summary(s, rt)  # type: ignore[misc]
                    )

                    def scroll_stroop_summary(rt: "scrolledtext.ScrolledText") -> None:
                        if rt.winfo_exists():
                            rt.see(tk.END)

                    self.root.after(0, lambda rt=results_text: scroll_stroop_summary(rt))  # type: ignore[misc]

                    # Save results
                    import datetime

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"stroop_task_results_{timestamp}.json"
                    task.save_results(filepath=filename)

                    self.root.after(
                        0, lambda: self._log_event(f"Task complete. Results saved to {filename}")
                    )

                    # Add close button
                    def close_and_show_results() -> None:
                        progress_dialog.destroy()
                        # task.print_results(analysis)  # Method may not exist
                        messagebox.showinfo(
                            "Task Complete",
                            f"Stroop Task completed!\n\n"
                            f"Overall Accuracy: {analysis['overall_accuracy']:.1%}\n"
                            f"Stroop Effect: {analysis['stroop_effect_ms']:.0f}ms\n"
                            f"Interference Score: {analysis['interference_score']:.2f}\n\n"
                            f"Results saved to:\n{filename}",
                        )

                    def add_close_button() -> None:
                        ttk.Button(
                            progress_dialog, text="Close", command=close_and_show_results
                        ).pack(pady=5)

                    self.root.after(0, add_close_button)

                except Exception:
                    self.root.after(
                        0,
                        lambda: messagebox.showerror("Task Error", "Error running task"),
                    )
                    self.root.after(
                        0,
                        lambda: (
                            progress_dialog.destroy() if progress_dialog.winfo_exists() else None
                        ),
                    )

            import threading

            task_thread = threading.Thread(target=run_task_thread, daemon=True)
            task_thread.start()

        except Exception:
            self._log_event("ERROR running task")
            messagebox.showerror("Task Error", "Failed to run task")
            import traceback

            traceback.print_exc()

    def _run_nback_task(self, num_trials: int = 30) -> None:
        """Run the N-Back Task experimental task."""
        if not self.apgi_simulation:
            messagebox.showerror("Error", "System not initialized")
            return

        # Stop current simulation if running
        was_running = self.is_running
        if was_running:
            self._stop_simulation()

        self._log_event("Starting N-Back Task...")
        self._update_status("Running N-Back Task...")

        # Import and create task
        try:
            from apgi_framework.experiments.tasks import NBackTask

            # Create task with specified parameters
            task = NBackTask(
                n_level=2,  # 2-back task
                num_trials=num_trials,
            )

            # Create progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Running N-Back Task")
            progress_dialog.geometry("500x250")

            ttk.Label(progress_dialog, text="Running trials...", font=("Arial", 12, "bold")).pack(
                pady=10
            )

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                progress_dialog, length=400, mode="determinate", variable=progress_var
            )
            progress_bar.pack(pady=10)

            status_label = ttk.Label(progress_dialog, text="Trial 0 of 0")
            status_label.pack(pady=5)

            # Results text area
            results_text = scrolledtext.ScrolledText(progress_dialog, height=8, width=60)
            results_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

            def run_task_thread() -> None:
                """Run task in separate thread to keep GUI responsive."""
                try:
                    results = task.run_all_trials(self.apgi_simulation)

                    # Update progress and status
                    for i, result in enumerate(results):
                        progress = (i + 1) / len(results) * 100
                        self.root.after(
                            0,
                            lambda p=progress, pv=progress_var: (pv.set(p)),  # type: ignore[misc]
                        )
                        self.root.after(
                            0,
                            lambda i=i, total=len(results), sl=status_label: (  # type: ignore[misc]
                                sl.config(text=f"Trial {i + 1} of {total}")
                                if sl.winfo_exists()
                                else None
                            ),
                        )

                        # Log trial results
                        self.root.after(
                            0,
                            lambda r=result, rt=results_text: (  # type: ignore[misc]
                                rt.insert(
                                    tk.END,
                                    f"Trial {r['trial_number']}: Stimulus '{r['stimulus']}' - Target: {r['is_target']} - Response: {r['response']} - Correct: {r['is_correct']}\n",
                                )
                                if rt.winfo_exists()
                                else None
                            ),
                        )
                        self.root.after(
                            0, lambda rt=results_text: rt.see(tk.END) if rt.winfo_exists() else None  # type: ignore[misc]
                        )

                    # Analyze results
                    analysis = task.analyze_results()

                    # Display summary
                    summary = f"\n{'=' * 50}\nRESULTS:\n{'=' * 50}\n"
                    summary += f"Total Trials: {analysis['total_trials']}\n"
                    summary += f"Hit Rate: {analysis['hit_rate']:.1%}\n"
                    summary += f"False Alarm Rate: {analysis['false_alarm_rate']:.1%}\n"
                    summary += f"d' (d-prime): {analysis['d_prime']:.2f}\n"
                    summary += f"Response Criterion: {analysis['response_criterion']:.2f}\n"
                    summary += f"Mean RT: {analysis['mean_response_time_ms']:.0f}ms\n\n"

                    summary += "Response Breakdown:\n"
                    summary += f"  Hits: {analysis['hits']}\n"
                    summary += f"  Misses: {analysis['misses']}\n"
                    summary += f"  False Alarms: {analysis['false_alarms']}\n"
                    summary += f"  Correct Rejections: {analysis['correct_rejections']}\n"

                    self.root.after(
                        0,
                        lambda s=summary, rt=results_text: (  # type: ignore[misc]
                            rt.insert(tk.END, s) if rt.winfo_exists() else None
                        ),
                    )
                    self.root.after(
                        0, lambda rt=results_text: rt.see(tk.END) if rt.winfo_exists() else None  # type: ignore[misc]
                    )

                    # Save results
                    import datetime

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"nback_task_results_{timestamp}.json"
                    task.save_results(filepath=filename)

                    self.root.after(
                        0, lambda: self._log_event(f"Task complete. Results saved to {filename}")
                    )

                    # Add close button
                    def close_and_show_results() -> None:
                        progress_dialog.destroy()
                        task.print_results(analysis)
                        messagebox.showinfo(
                            "Task Complete",
                            f"N-Back Task completed!\n\n"
                            f"Hit Rate: {analysis['hit_rate']:.1%}\n"
                            f"d' (d-prime): {analysis['d_prime']:.2f}\n"
                            f"Mean RT: {analysis['mean_response_time_ms']:.0f}ms\n\n"
                            f"Results saved to:\n{filename}",
                        )

                    self.root.after(
                        0,
                        lambda: ttk.Button(
                            progress_dialog, text="Close", command=close_and_show_results
                        ).pack(pady=5),
                    )

                except Exception:
                    self.root.after(
                        0,
                        lambda: messagebox.showerror("Task Error", "Error running task"),
                    )
                    self.root.after(
                        0,
                        lambda: (
                            progress_dialog.destroy() if progress_dialog.winfo_exists() else None
                        ),
                    )

            import threading

            task_thread = threading.Thread(target=run_task_thread, daemon=True)
            task_thread.start()

        except Exception as e:
            self._log_event(f"ERROR running task: {str(e)}")
            messagebox.showerror("Task Error", f"Failed to run task:\n{str(e)}")
            import traceback

            traceback.print_exc()

    def _zoom_in(self) -> None:
        """Zoom in on all matplotlib plots."""
        try:
            # Zoom in on neural plots
            if hasattr(self, "neural_axes"):
                for ax in self.neural_axes.values():
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    x_center = (xlim[0] + xlim[1]) / 2
                    y_center = (ylim[0] + ylim[1]) / 2
                    x_range = (xlim[1] - xlim[0]) * 0.8  # Zoom in by 20%
                    y_range = (ylim[1] - ylim[0]) * 0.8
                    ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
                    ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
                self.neural_canvas.draw_idle()  # type: ignore[no-untyped-call]

            # Zoom in on other plot canvases if they exist
            for canvas_attr in [
                "intero_canvas",
                "metrics_canvas",
                "self_canvas",
                "osc_canvas",
                "state_canvas",
            ]:
                if hasattr(self, canvas_attr):
                    canvas = getattr(self, canvas_attr)
                    if hasattr(canvas, "figure"):
                        for ax in canvas.figure.get_axes():
                            xlim = ax.get_xlim()
                            ylim = ax.get_ylim()
                            x_center = (xlim[0] + xlim[1]) / 2
                            y_center = (ylim[0] + ylim[1]) / 2
                            x_range = (xlim[1] - xlim[0]) * 0.8
                            y_range = (ylim[1] - ylim[0]) * 0.8
                            ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
                            ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
                        canvas.draw_idle()

        except Exception as e:
            self._log_event(f"Error zooming in: {str(e)}")

    def _zoom_out(self) -> None:
        """Zoom out on all matplotlib plots."""
        try:
            # Zoom out on neural plots
            if hasattr(self, "neural_axes"):
                for ax in self.neural_axes.values():
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    x_center = (xlim[0] + xlim[1]) / 2
                    y_center = (ylim[0] + ylim[1]) / 2
                    x_range = (xlim[1] - xlim[0]) * 1.25  # Zoom out by 25%
                    y_range = (ylim[1] - ylim[0]) * 1.25
                    ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
                    ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
                self.neural_canvas.draw_idle()  # type: ignore[no-untyped-call]

            # Zoom out on other plot canvases if they exist
            for canvas_attr in [
                "intero_canvas",
                "metrics_canvas",
                "self_canvas",
                "osc_canvas",
                "state_canvas",
            ]:
                if hasattr(self, canvas_attr):
                    canvas = getattr(self, canvas_attr)
                    if hasattr(canvas, "figure"):
                        for ax in canvas.figure.get_axes():
                            xlim = ax.get_xlim()
                            ylim = ax.get_ylim()
                            x_center = (xlim[0] + xlim[1]) / 2
                            y_center = (ylim[0] + ylim[1]) / 2
                            x_range = (xlim[1] - xlim[0]) * 1.25
                            y_range = (ylim[1] - ylim[0]) * 1.25
                            ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
                            ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
                        canvas.draw_idle()

        except Exception as e:
            self._log_event(f"Error zooming out: {str(e)}")

    def _zoom_fit(self) -> None:
        """Reset zoom to fit all data on all matplotlib plots."""
        try:
            # Check if there's data to fit
            if len(self.time_buffer) == 0:
                messagebox.showinfo("No Data", "No data to fit — start simulation first")
                return

            # Force a plot update to recalculate limits
            if len(self.time_buffer) > 0:
                self._update_plots()

            # Reset neural plots to auto-scale
            if hasattr(self, "neural_axes"):
                for ax in self.neural_axes.values():
                    ax.relim()
                    ax.autoscale_view()
                self.neural_canvas.draw_idle()  # type: ignore[no-untyped-call]

            # Reset other plot canvases if they exist
            for canvas_attr in [
                "intero_canvas",
                "metrics_canvas",
                "self_canvas",
                "osc_canvas",
                "state_canvas",
            ]:
                if hasattr(self, canvas_attr):
                    canvas = getattr(self, canvas_attr)
                    if hasattr(canvas, "figure"):
                        for ax in canvas.figure.get_axes():
                            ax.relim()
                            ax.autoscale_view()
                        canvas.draw_idle()

        except Exception as e:
            self._log_event(f"Error fitting to window: {str(e)}")

    def _trigger_ignition(self) -> None:
        """Manually trigger ignition event."""
        if not self.apgi_simulation:
            messagebox.showinfo("No System", "Please initialize the system first.")
            return

        if not self.is_running:
            messagebox.showinfo("Simulation Not Running", "Please start the simulation first.")
            return

        # Force high arousal to trigger ignition
        self.param_vars["arousal"].set(0.9)
        self.param_vars["stress"].set(0.8)
        self._log_event("Manual ignition trigger activated")

    def _induce_stressor(self) -> None:
        """Induce stressor event with configurable intensity."""
        if not self.apgi_simulation:
            messagebox.showinfo("No System", "Please initialize the system first.")
            return

        # Create dialog for intensity selection
        dialog = tk.Toplevel(self.root)
        dialog.title("Induce Stressor")
        dialog.geometry("350x200")
        dialog.resizable(False, False)

        # Make dialog modal
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Select Stressor Intensity:", font=("Arial", 12, "bold")).pack(
            pady=20
        )

        # Intensity slider
        intensity_var = tk.DoubleVar(value=0.5)
        intensity_frame = ttk.Frame(dialog)
        intensity_frame.pack(pady=10)

        ttk.Label(intensity_frame, text="Intensity:").pack(side=tk.LEFT)
        intensity_scale = ttk.Scale(
            intensity_frame,
            from_=0.1,
            to=2.0,
            variable=intensity_var,
            orient=tk.HORIZONTAL,
            length=200,
        )
        intensity_scale.pack(side=tk.LEFT, padx=10)

        intensity_label = ttk.Label(intensity_frame, text="0.50")
        intensity_label.pack(side=tk.LEFT)
        intensity_var.trace_add(
            "write", lambda *args: intensity_label.config(text=f"{intensity_var.get():.2f}")
        )

        def apply_stressor() -> None:
            intensity = intensity_var.get()
            if self.apgi_simulation:
                self.apgi_simulation.allostasis.trigger_stressor(intensity=intensity)  # type: ignore[attr-defined]
            self._log_event(f"Stressor induced (intensity: {intensity:.2f})")
            messagebox.showinfo("Success", f"Stressor induced with intensity {intensity:.2f}")
            dialog.destroy()

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)

        ttk.Button(button_frame, text="Apply", command=apply_stressor).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=10)

    def _modulate_precision(self) -> None:
        """Open precision modulation dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Modulate Precision")
        dialog.geometry("400x200")

        ttk.Label(dialog, text="Precision Modulation", font=("Arial", 12, "bold")).pack(pady=10)

        frame = ttk.Frame(dialog)
        frame.pack(pady=10)

        ttk.Label(frame, text="Modulation Factor:").pack()
        factor_var = tk.DoubleVar(value=1.0)
        ttk.Scale(frame, from_=0.1, to=3.0, variable=factor_var, orient=tk.HORIZONTAL).pack()

        def apply() -> None:
            factor = factor_var.get()
            extero_current = self.param_vars["extero_precision"].get()
            intero_current = self.param_vars["intero_precision"].get()

            extero_new = extero_current * factor
            intero_new = intero_current * factor

            # Clamp to valid precision range [0.1, 10.0]
            extero_clamped = min(10.0, max(0.1, extero_new))
            intero_clamped = min(10.0, max(0.1, intero_new))

            self.param_vars["extero_precision"].set(extero_clamped)
            self.param_vars["intero_precision"].set(intero_clamped)

            self._log_event(
                f"Precision modulated by factor: {factor:.2f} (Extero: {extero_current:.2f} -> {extero_clamped:.2f}, Intero: {intero_current:.2f} -> {intero_clamped:.2f})"
            )
            dialog.destroy()

        ttk.Button(dialog, text="Apply", command=apply).pack(pady=10)

    def _inject_input(self) -> None:
        """Inject custom sensory input."""
        if not self.apgi_simulation:
            messagebox.showwarning(
                "Warning", "No active APGI system found. Please initialize the system first."
            )
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Custom Input Injection")
        dialog.geometry("500x600")

        # Make dialog modal
        dialog.transient(self.root)
        # dialog.grab_set()  # Removed to prevent blocking simulation updates

        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(main_frame, text="Inject Custom Sensory Input", font=("Arial", 12, "bold")).pack(
            pady=10
        )

        # Input type selection
        input_frame = ttk.LabelFrame(main_frame, text="Input Type", padding=10)
        input_frame.pack(fill=tk.X, pady=5)

        input_type = tk.StringVar(value="extero")
        ttk.Radiobutton(
            input_frame, text="Exteroceptive (External)", variable=input_type, value="extero"
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            input_frame, text="Interoceptive (Internal)", variable=input_type, value="intero"
        ).pack(anchor=tk.W)

        # Input parameters
        params_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=5)

        # Intensity
        ttk.Label(params_frame, text="Intensity:").grid(row=0, column=0, sticky=tk.W, pady=2)
        intensity_var = tk.DoubleVar(value=1.0)
        intensity_scale = ttk.Scale(
            params_frame, from_=0.0, to=5.0, variable=intensity_var, orient=tk.HORIZONTAL
        )
        intensity_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        intensity_label = ttk.Label(params_frame, text="1.00")
        intensity_label.grid(row=0, column=2)
        intensity_var.trace_add(
            "write", lambda *args: intensity_label.config(text=f"{intensity_var.get():.2f}")
        )

        # Duration
        ttk.Label(params_frame, text="Duration (steps):").grid(row=1, column=0, sticky=tk.W, pady=2)
        duration_var = tk.IntVar(value=10)
        duration_spin = ttk.Spinbox(
            params_frame, from_=1, to=100, textvariable=duration_var, width=10
        )
        duration_spin.grid(row=1, column=1, sticky=tk.W, padx=5)

        # Noise level
        ttk.Label(params_frame, text="Noise Level:").grid(row=2, column=0, sticky=tk.W, pady=2)
        noise_var = tk.DoubleVar(value=0.1)
        noise_scale = ttk.Scale(
            params_frame, from_=0.0, to=1.0, variable=noise_var, orient=tk.HORIZONTAL
        )
        noise_scale.grid(row=2, column=1, sticky=tk.EW, padx=5)
        noise_label = ttk.Label(params_frame, text="0.10")
        noise_label.grid(row=2, column=2)
        noise_var.trace_add(
            "write", lambda *args: noise_label.config(text=f"{noise_var.get():.2f}")
        )

        params_frame.columnconfigure(1, weight=1)

        # Pattern selection
        pattern_frame = ttk.LabelFrame(main_frame, text="Input Pattern", padding=10)
        pattern_frame.pack(fill=tk.X, pady=5)

        pattern_var = tk.StringVar(value="constant")
        ttk.Radiobutton(
            pattern_frame, text="Constant", variable=pattern_var, value="constant"
        ).pack(anchor=tk.W)
        ttk.Radiobutton(pattern_frame, text="Pulse", variable=pattern_var, value="pulse").pack(
            anchor=tk.W
        )
        ttk.Radiobutton(pattern_frame, text="Sine Wave", variable=pattern_var, value="sine").pack(
            anchor=tk.W
        )
        ttk.Radiobutton(
            pattern_frame, text="Random Noise", variable=pattern_var, value="random"
        ).pack(anchor=tk.W)

        # Status display
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5)

        status_text = tk.Text(status_frame, height=6, width=50)
        status_text.pack(fill=tk.BOTH, expand=True)

        def inject_input() -> None:
            """Inject the custom input into the APGI system."""
            try:
                input_type_val = input_type.get()
                intensity = intensity_var.get()
                duration = duration_var.get()
                noise = noise_var.get()
                pattern = pattern_var.get()

                # Generate input pattern
                if pattern == "constant":
                    input_signal = np.full(duration, intensity)
                elif pattern == "pulse":
                    input_signal = np.zeros(duration)
                    input_signal[duration // 4 : 3 * duration // 4] = intensity
                elif pattern == "sine":
                    t = np.linspace(0, 2 * np.pi, duration)
                    input_signal = intensity * (np.sin(t) + 1) / 2
                else:  # random
                    input_signal = np.random.randn(duration) * noise + intensity

                # Add noise
                if noise > 0:
                    input_signal += np.random.randn(duration) * noise

                # Inject into system
                self.custom_input_params = {
                    "pattern": pattern,
                    "intensity": intensity,
                    "noise": noise,
                    "type": input_type_val,
                }
                self.custom_input_steps_remaining = duration

                status_text.delete(1.0, tk.END)
                status_text.insert(tk.END, f"Input Type: {input_type_val}\n")
                status_text.insert(tk.END, f"Pattern: {pattern}\n")
                status_text.insert(tk.END, f"Intensity: {intensity:.2f}\n")
                status_text.insert(tk.END, f"Duration: {duration} steps\n")
                status_text.insert(tk.END, f"Noise Level: {noise:.2f}\n\n")
                status_text.insert(tk.END, "Custom input injected into APGI system!\n")
                status_text.insert(tk.END, f"Will affect next {duration} simulation steps.\n")

                self._log_event(
                    f"Custom sensory input injected: {pattern} pattern, intensity {intensity:.2f}, duration {duration} steps"
                )

                messagebox.showinfo(
                    "Success",
                    f"Custom input injected successfully!\n\nWill affect the next {duration} simulation steps.",
                )

            except Exception as e:
                messagebox.showerror("Error", f"Failed to inject input: {str(e)}")
                status_text.delete(1.0, tk.END)
                status_text.insert(tk.END, f"Error: {str(e)}")

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Inject Input", command=inject_input).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            button_frame, text="Clear Status", command=lambda: status_text.delete(1.0, tk.END)
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def _set_body_state(self) -> None:
        """Set body state manually."""
        if not self.apgi_simulation:
            messagebox.showwarning(
                "Warning", "No active APGI system found. Please initialize the system first."
            )
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Set Body State")
        dialog.geometry("500x600")

        # Make dialog modal
        dialog.transient(self.root)
        # dialog.grab_set()  # Removed to prevent blocking simulation updates

        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(main_frame, text="Body State Configuration", font=("Arial", 12, "bold")).pack(
            pady=10
        )

        # Current values display
        current_frame = ttk.LabelFrame(main_frame, text="Current Values", padding=10)
        current_frame.pack(fill=tk.X, pady=5)

        current_values = {
            "Heart Rate": f"{self.apgi_simulation.body_model.get_current_state_dict()['heart_rate']:.1f} bpm",
            "Cortisol": f"{self.apgi_simulation.body_model.get_current_state_dict()['cortisol']:.3f} μg/dL",
            "Arousal": f"{self.apgi_simulation.body_model.arousal_level:.3f}",
            "Stress": f"{self.apgi_simulation.body_model.stress_level:.3f}",
            "Activity": f"{self.apgi_simulation.body_model.activity_level:.3f}",
        }

        for label, value in current_values.items():
            frame = ttk.Frame(current_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"{label}:", width=12, anchor="w").pack(side=tk.LEFT)
            ttk.Label(frame, text=value, anchor="w").pack(side=tk.LEFT, padx=10)

        # Parameter adjustment frame
        adjust_frame = ttk.LabelFrame(main_frame, text="Adjust Parameters", padding=10)
        adjust_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Define adjustable parameters
        params = [
            ("Arousal Level", "arousal", 0.0, 1.0, self.apgi_simulation.body_model.arousal_level),
            ("Stress Level", "stress", 0.0, 1.0, self.apgi_simulation.body_model.stress_level),
            (
                "Activity Level",
                "activity",
                0.0,
                1.0,
                self.apgi_simulation.body_model.activity_level,
            ),
            (
                "Heart Rate",
                "heart_rate",
                60.0,
                180.0,
                self.apgi_simulation.body_model.get_current_state_dict()["heart_rate"],
            ),
            (
                "Cortisol",
                "cortisol",
                0.0,
                2.0,
                self.apgi_simulation.body_model.get_current_state_dict()["cortisol"],
            ),
            (
                "Temperature",
                "temperature",
                36.0,
                40.0,
                getattr(self.apgi_simulation.body_model, "temperature", 37.0),
            ),
            (
                "Respiration",
                "respiration",
                8.0,
                30.0,
                getattr(self.apgi_simulation.body_model, "respiration_rate", 12.0),
            ),
            (
                "Glucose",
                "glucose",
                70.0,
                200.0,
                getattr(self.apgi_simulation.body_model, "glucose_level", 90.0),
            ),
        ]

        param_vars = {}
        for label, key, min_val, max_val, default in params:
            frame = ttk.Frame(adjust_frame)
            frame.pack(fill=tk.X, pady=3)

            ttk.Label(frame, text=label, width=15).pack(side=tk.LEFT)

            var = tk.DoubleVar(value=default)
            param_vars[key] = var

            scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            val_label = ttk.Label(frame, text=f"{default:.2f}", width=8)
            val_label.pack(side=tk.LEFT)

            var.trace_add(
                "write", lambda *args, v=var, label=val_label: label.config(text=f"{v.get():.2f}")
            )

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        def apply_body_state() -> None:
            """Apply the body state changes."""
            try:
                # Set the parameters
                if self.apgi_simulation is None:
                    return
                if "arousal" in param_vars:
                    self.apgi_simulation.body_model.set_arousal(param_vars["arousal"].get())
                if "stress" in param_vars:
                    self.apgi_simulation.body_model.set_stress(param_vars["stress"].get())
                if "activity" in param_vars:
                    self.apgi_simulation.body_model.set_activity(param_vars["activity"].get())

                # For other parameters, try to set them if the body_model supports it
                for key in ["heart_rate", "cortisol", "temperature", "respiration", "glucose"]:
                    if hasattr(self.apgi_simulation.body_model, f"set_{key}"):
                        val = param_vars[key].get()
                        # Extract scalar if numpy array
                        if hasattr(val, "__iter__") and not isinstance(val, str):  # type: ignore[unreachable]
                            val = float(val[0]) if len(val) > 0 else 0.0  # type: ignore[index, arg-type]
                        getattr(self.apgi_simulation.body_model, f"set_{key}")(val)
                    elif hasattr(self.apgi_simulation.body_model, key):
                        val = param_vars[key].get()
                        # Extract scalar if numpy array
                        if hasattr(val, "__iter__") and not isinstance(val, str):  # type: ignore[unreachable]
                            val = float(val[0]) if len(val) > 0 else 0.0  # type: ignore[index, arg-type]
                        setattr(self.apgi_simulation.body_model, key, val)

                self._log_event("Body state manually adjusted")
                messagebox.showinfo("Success", "Body state updated successfully!")
                dialog.destroy()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to update body state: {str(e)}")

        def reset_to_current() -> None:
            """Reset sliders to current values."""
            if self.apgi_simulation is None:
                return
            state_dict = self.apgi_simulation.body_model.get_current_state_dict()
            for key, var in param_vars.items():
                if key in state_dict:
                    val = state_dict[key]
                    # Extract scalar if numpy array
                    if hasattr(val, "__iter__") and not isinstance(val, str):
                        val = float(val[0]) if len(val) > 0 else 0.0
                    var.set(val)
                elif key == "arousal":
                    arousal_val = self.apgi_simulation.body_model.arousal_level
                    if hasattr(arousal_val, "__iter__") and not isinstance(arousal_val, str):  # type: ignore[unreachable]
                        arousal_val = float(arousal_val[0]) if len(arousal_val) > 0 else 0.0  # type: ignore[assignment]
                    var.set(arousal_val)  # type: ignore[arg-type]
                elif key == "stress":
                    stress_val = self.apgi_simulation.body_model.stress_level
                    if hasattr(stress_val, "__iter__") and not isinstance(stress_val, str):  # type: ignore[unreachable]
                        stress_val = float(stress_val[0]) if len(stress_val) > 0 else 0.0  # type: ignore[assignment]
                    var.set(stress_val)  # type: ignore[arg-type]
                elif key == "activity":
                    activity_val = self.apgi_simulation.body_model.activity_level
                    if hasattr(activity_val, "__iter__") and not isinstance(activity_val, str):  # type: ignore[unreachable]
                        activity_val = float(activity_val[0]) if len(activity_val) > 0 else 0.0  # type: ignore[assignment]
                    var.set(activity_val)  # type: ignore[arg-type]

        ttk.Button(button_frame, text="Apply", command=apply_body_state).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Current", command=reset_to_current).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def _show_diagnostics(self) -> None:
        """Show system diagnostics."""
        if not self.apgi_simulation:
            messagebox.showwarning("No System", "System not initialized")
            return

        summary = self.apgi_simulation.get_state_summary()

        # Extract values with validation
        ignition_stats = summary.get("ignition_stats", {})
        somatic_markers = summary.get("somatic_markers", {})

        mean_signal = ignition_stats.get("mean_signal", "N/A")
        mean_threshold = ignition_stats.get("mean_threshold", "N/A")
        retrieval_success_rate = somatic_markers.get("retrieval_success_rate", "N/A")

        diag_text = f"""APGI System Diagnostics

Time: {summary.get('time_ms', 0) / 1000.0:.2f} seconds

Ignition Statistics:
- Recent Events: {ignition_stats.get('recent_ignitions', 'N/A')}
- Mean Signal: {mean_signal if mean_signal != 'N/A' else 'N/A'}
- Mean Threshold: {mean_threshold if mean_threshold != 'N/A' else 'N/A'}

Workspace State: {summary.get('workspace_state', 'N/A')}

Metabolic State:
- Reserves: {summary.get('metabolic_reserves', 'N/A')}
- Allostatic Load: {summary.get('allostatic_load', 'N/A') * 100:.1f if summary.get('allostatic_load') != 'N/A' else 'N/A'}%

Somatic Markers:
- Count: {somatic_markers.get('num_markers', 'N/A')}
- Retrieval Success Rate: {retrieval_success_rate * 100:.1f if retrieval_success_rate != 'N/A' else 'N/A'}%
"""

        messagebox.showinfo("System Diagnostics", diag_text)

    def _show_ignition_stats(self) -> None:
        """Show detailed ignition statistics."""
        if not self.apgi_simulation:
            return

        stats = self.apgi_simulation.ignition_threshold.get_statistics()

        text = f"""Ignition Statistics

Mean Signal: {stats['mean_signal']:.3f}
Std Signal: {stats.get('std_signal', 0):.3f}
Mean Threshold: {stats['mean_threshold']:.3f}
Std Threshold: {stats.get('std_threshold', 0):.3f}

Recent Ignitions: {stats['recent_ignitions']}
Ignition Rate: {stats['ignition_rate']:.3f}
Current Probability: {stats['current_probability']:.3f}
"""

        messagebox.showinfo("Ignition Statistics", text)

    def _show_energy_report(self) -> None:
        """Show energy budget report."""
        if not self.apgi_simulation:
            return

        text = f"""Energy Budget Report

Current Reserves: {self.apgi_simulation.metabolism.current_reserves:.1f}
Total Consumed: {self.apgi_simulation.metabolism.total_consumed:.1f}

Baseline Rate: {self.apgi_simulation.metabolism.baseline_rate:.1f}/s
Ignition Cost: {self.apgi_simulation.metabolism.ignition_cost:.1f} per event

Reserve Fraction: {self.apgi_simulation.metabolism.current_reserves / self.apgi_simulation.metabolism.total_budget * 100:.1f}%
"""

        messagebox.showinfo("Energy Budget", text)

    def _analyze_markers(self) -> None:
        """Analyze somatic markers."""
        if not self.apgi_simulation:
            return

        stats = self.apgi_simulation.somatic_markers.get_statistics()

        text = f"""Somatic Marker Analysis

Total Markers: {stats['num_markers']}
Capacity Used: {stats.get('capacity_used', 0) * 100:.1f}%

Total Retrievals: {stats.get('total_retrievals', 0)}
Successful Retrievals: {stats.get('successful_retrievals', 0)}
Success Rate: {stats.get('retrieval_success_rate', 0) * 100:.1f}%

Average Strength: {stats.get('avg_strength', 0):.3f}
Average Outcome: {stats.get('avg_outcome', 0):.3f}
"""

        messagebox.showinfo("Somatic Marker Analysis", text)

    def _analyze_coherence(self) -> None:
        """Analyze self-model coherence."""
        if not self.apgi_simulation:
            messagebox.showwarning("No System", "Please start the simulation first")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Self-Model Coherence Analysis")
        dialog.geometry("600x500")

        # Make dialog modal
        dialog.transient(self.root)
        # dialog.grab_set()  # Removed to prevent blocking simulation updates

        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(
            main_frame, text="Self-Model Coherence Analysis", font=("Arial", 14, "bold")
        ).pack(pady=10)

        # Analysis results text area
        results_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=20)
        results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        def run_coherence_analysis() -> None:
            """Run coherence analysis and display results."""
            results_text.delete(1.0, tk.END)
            results_text.insert(tk.END, "ANALYZING SELF-MODEL COHERENCE\n")
            results_text.insert(tk.END, "=" * 50 + "\n\n")

            try:
                # Get coherence data from buffer
                with self.data_lock:
                    coherence_data = list(self.data_buffers.get("minimal_self_coherence", []))

                if not coherence_data:
                    results_text.insert(tk.END, "No coherence data available.\n")
                    results_text.insert(tk.END, "Please run the simulation to collect data.\n")
                    return

                import numpy as np

                coherence_array = np.array(coherence_data)

                # Basic statistics
                mean_coherence = np.mean(coherence_array)
                std_coherence = np.std(coherence_array)
                min_coherence = np.min(coherence_array)
                max_coherence = np.max(coherence_array)
                median_coherence = np.median(coherence_array)

                # Trend analysis
                if len(coherence_array) > 10:
                    # Simple linear trend
                    x = np.arange(len(coherence_array))
                    slope, intercept = np.polyfit(x, coherence_array, 1)
                    trend_direction = (
                        "increasing"
                        if slope > 0.001
                        else "decreasing" if slope < -0.001 else "stable"
                    )
                else:
                    trend_direction = "insufficient data"

                # State assessment
                if mean_coherence > 0.8:
                    state = "High coherence - Strong self-model integrity"
                elif mean_coherence > 0.6:
                    state = "Moderate coherence - Normal self-model function"
                elif mean_coherence > 0.4:
                    state = "Low coherence - Potential depersonalization risk"
                else:
                    state = "Very low coherence - Significant self-model disruption"

                # Variability assessment
                if std_coherence < 0.1:
                    stability = "Highly stable coherence"
                elif std_coherence < 0.2:
                    stability = "Moderately stable coherence"
                else:
                    stability = "Highly variable coherence"

                # Display results
                results_text.insert(tk.END, "STATISTICAL SUMMARY\n")
                results_text.insert(tk.END, f"Data Points: {len(coherence_array)}\n")
                results_text.insert(tk.END, f"Mean Coherence: {mean_coherence:.3f}\n")
                results_text.insert(tk.END, f"Standard Deviation: {std_coherence:.3f}\n")
                results_text.insert(tk.END, f"Median: {median_coherence:.3f}\n")
                results_text.insert(tk.END, f"Range: {min_coherence:.3f} - {max_coherence:.3f}\n")
                results_text.insert(tk.END, f"Trend: {trend_direction}\n\n")

                results_text.insert(tk.END, "ASSESSMENT\n")
                results_text.insert(tk.END, f"Current State: {state}\n")
                results_text.insert(tk.END, f"Stability: {stability}\n\n")

                results_text.insert(tk.END, "INTERPRETATION\n")
                if mean_coherence > 0.7:
                    results_text.insert(tk.END, "• Self-model maintains strong coherence\n")
                    results_text.insert(tk.END, "• Effective predictive processing\n")
                    results_text.insert(
                        tk.END, "• Good integration of exteroceptive and interoceptive signals\n"
                    )
                elif mean_coherence > 0.5:
                    results_text.insert(tk.END, "• Self-model coherence is adequate\n")
                    results_text.insert(tk.END, "• Minor fluctuations in self-representation\n")
                    results_text.insert(tk.END, "• Monitor for environmental stressors\n")
                else:
                    results_text.insert(tk.END, "• Self-model coherence is compromised\n")
                    results_text.insert(tk.END, "• Risk of depersonalization or dissociation\n")
                    results_text.insert(tk.END, "• Consider reducing cognitive load or stress\n")
                    results_text.insert(
                        tk.END, "• May benefit from increased precision weighting\n"
                    )

                if std_coherence > 0.15:
                    results_text.insert(
                        tk.END, "\n• High variability suggests environmental instability\n"
                    )
                    results_text.insert(tk.END, "• Consider stabilizing sensory input patterns\n")

                results_text.insert(tk.END, f"\nAnalysis completed at {time.strftime('%H:%M:%S')}")

            except Exception as e:
                results_text.insert(tk.END, f"Analysis failed: {str(e)}\n")
                import traceback

                results_text.insert(tk.END, traceback.format_exc())

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Run Analysis", command=run_coherence_analysis).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            button_frame, text="Clear", command=lambda: results_text.delete(1.0, tk.END)
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

        # Run initial analysis
        run_coherence_analysis()

    def _generate_report(self) -> None:
        """Generate comprehensive report with statistical analysis."""
        # Create dialog to choose export format
        dialog = tk.Toplevel(self.root)
        dialog.title("Generate Report")
        dialog.geometry("400x200")
        dialog.resizable(False, False)

        # Center dialog
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Choose export format:", font=("Arial", 12, "bold")).pack(pady=20)

        format_var = tk.StringVar(value="text")

        # Radio buttons for format selection
        format_frame = ttk.Frame(dialog)
        format_frame.pack(pady=10)

        ttk.Radiobutton(
            format_frame, text="Text Report (.txt)", variable=format_var, value="text"
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            format_frame, text="PDF Report (.pdf)", variable=format_var, value="pdf"
        ).pack(anchor=tk.W)

        def proceed_with_format() -> None:
            dialog.destroy()
            self._do_generate_report(format_var.get())

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)

        ttk.Button(button_frame, text="Generate", command=proceed_with_format).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=10)

    def _do_generate_report(self, format_type: str) -> None:
        """Generate report in specified format."""
        if format_type == "pdf":
            filename = filedialog.asksaveasfilename(
                title="Save PDF Report",
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            )
        else:
            filename = filedialog.asksaveasfilename(
                title="Save Text Report",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            )

        if filename:
            try:
                if format_type == "pdf":
                    self._generate_pdf_report(filename)
                else:
                    self._generate_text_report(filename)

                messagebox.showinfo("Success", f"Report saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report:\n{str(e)}")

    def _generate_text_report(self, filename: str) -> None:
        """Generate text report (original implementation)."""
        with open(filename, "w") as f:
            f.write("APGI System Comprehensive Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if self.apgi_simulation:
                # Basic summary
                summary = self.apgi_simulation.get_state_summary()
                f.write(f"Simulation Time: {summary['time_ms'] / 1000.0:.2f} seconds\n")
                f.write(f"Ignition Events: {summary['ignition_stats']['recent_ignitions']}\n")
                f.write(f"Metabolic Reserves: {summary['metabolic_reserves']:.1f}\n")
                f.write(f"Allostatic Load: {summary['allostatic_load'] * 100:.1f}%\n\n")

                # Enhanced statistical analysis
                try:
                    from apgi_framework.analysis import SystemAnalyzer

                    analyzer = SystemAnalyzer(self.apgi_simulation.config)
                    analysis_results = analyzer.analyze_system(self.apgi_simulation)
                    statistical_summary = analyzer.generate_statistical_summary(
                        analysis_results._data
                    )

                    f.write("STATISTICAL ANALYSIS\n")
                    f.write("-" * 30 + "\n\n")

                    # Ignition analysis
                    f.write("Ignition Dynamics:\n")
                    ign_stats = statistical_summary["ignition_analysis"]
                    f.write(f"  Rate: {ign_stats['statistics']['ignition_rate_hz']:.3f} Hz\n")
                    f.write(
                        f"  Mean Interval: {ign_stats['statistics']['mean_ignition_interval_ms']:.1f} ms\n"
                    )
                    f.write(f"  Interpretation: {ign_stats['interpretation']}\n\n")

                    # Energy analysis
                    f.write("Energy Budget:\n")
                    energy_stats = statistical_summary["energy_analysis"]
                    f.write(
                        f"  Total Consumed: {energy_stats['statistics']['total_energy_consumed']:.3f}\n"
                    )
                    f.write(
                        f"  Depletion Rate: {energy_stats['statistics']['reserve_depletion_rate']:.4f}/s\n"
                    )
                    f.write(f"  Interpretation: {energy_stats['interpretation']}\n\n")

                    # Learning analysis
                    f.write("Learning & Memory:\n")
                    learning_stats = statistical_summary["learning_analysis"]
                    f.write(f"  Total Markers: {learning_stats['statistics']['total_markers']}\n")
                    f.write(
                        f"  Retrieval Success: {learning_stats['statistics']['retrieval_success_rate']:.1%}\n"
                    )
                    f.write(f"  Interpretation: {learning_stats['interpretation']}\n\n")

                    # Coherence analysis
                    f.write("Self-Model Coherence:\n")
                    coherence_stats = statistical_summary["coherence_analysis"]
                    f.write(
                        f"  Mean Coherence: {coherence_stats['statistics']['mean_coherence']:.3f}\n"
                    )
                    f.write(f"  Interpretation: {coherence_stats['interpretation']}\n\n")

                except Exception as e:
                    f.write(f"Statistical analysis failed: {str(e)}\n\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write("Event Log:\n")
            f.write(self.log_text.get(1.0, tk.END))

        self._log_event(f"Text report saved: {filename}")

    def _generate_pdf_report(self, filename: str) -> None:
        """Generate PDF report using reportlab."""
        try:
            from reportlab.lib.pagesizes import letter  # type: ignore[import-untyped]
            from reportlab.lib.styles import (  # type: ignore[import-untyped]
                ParagraphStyle,
                getSampleStyleSheet,
            )

            # Create PDF document
            from reportlab.platypus import (  # type: ignore[import-untyped]
                PageBreak,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
            )

            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=16,
                spaceAfter=30,
                alignment=1,  # Center
            )
            story.append(Paragraph("APGI System Comprehensive Report", title_style))
            story.append(Spacer(1, 12))

            # Generation info
            story.append(
                Paragraph(
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]
                )
            )
            story.append(Spacer(1, 12))

            if self.apgi_simulation:
                # Basic summary
                summary = self.apgi_simulation.get_state_summary()
                story.append(Paragraph("<b>Simulation Summary</b>", styles["Heading2"]))
                story.append(Spacer(1, 6))

                summary_text = f"""
                <b>Simulation Time:</b> {summary['time_ms'] / 1000.0:.2f} seconds<br/>
                <b>Ignition Events:</b> {summary['ignition_stats']['recent_ignitions']}<br/>
                <b>Metabolic Reserves:</b> {summary['metabolic_reserves']:.1f}<br/>
                <b>Allostatic Load:</b> {summary['allostatic_load'] * 100:.1f}%
                """
                story.append(Paragraph(summary_text, styles["Normal"]))
                story.append(Spacer(1, 12))

                # Enhanced statistical analysis
                try:
                    from apgi_framework.analysis import SystemAnalyzer

                    analyzer = SystemAnalyzer(self.apgi_simulation.config)
                    analysis_results = analyzer.analyze_system(self.apgi_simulation)
                    statistical_summary = analyzer.generate_statistical_summary(
                        analysis_results._data
                    )

                    story.append(Paragraph("<b>Statistical Analysis</b>", styles["Heading2"]))
                    story.append(Spacer(1, 6))

                    # Ignition analysis
                    story.append(Paragraph("<b>Ignition Dynamics:</b>", styles["Heading3"]))
                    ign_stats = statistical_summary["ignition_analysis"]
                    ignition_text = f"""
                    Rate: {ign_stats['statistics']['ignition_rate_hz']:.3f} Hz<br/>
                    Mean Interval: {ign_stats['statistics']['mean_ignition_interval_ms']:.1f} ms<br/>
                    <i>{ign_stats['interpretation']}</i>
                    """
                    story.append(Paragraph(ignition_text, styles["Normal"]))
                    story.append(Spacer(1, 6))

                    # Energy analysis
                    story.append(Paragraph("<b>Energy Budget:</b>", styles["Heading3"]))
                    energy_stats = statistical_summary["energy_analysis"]
                    energy_text = f"""
                    Total Consumed: {energy_stats['statistics']['total_energy_consumed']:.3f}<br/>
                    Depletion Rate: {energy_stats['statistics']['reserve_depletion_rate']:.4f}/s<br/>
                    <i>{energy_stats['interpretation']}</i>
                    """
                    story.append(Paragraph(energy_text, styles["Normal"]))
                    story.append(Spacer(1, 6))

                    # Learning analysis
                    story.append(Paragraph("<b>Learning & Memory:</b>", styles["Heading3"]))
                    learning_stats = statistical_summary["learning_analysis"]
                    learning_text = f"""
                    Total Markers: {learning_stats['statistics']['total_markers']}<br/>
                    Retrieval Success: {learning_stats['statistics']['retrieval_success_rate']:.1%}<br/>
                    <i>{learning_stats['interpretation']}</i>
                    """
                    story.append(Paragraph(learning_text, styles["Normal"]))
                    story.append(Spacer(1, 6))

                    # Coherence analysis
                    story.append(Paragraph("<b>Self-Model Coherence:</b>", styles["Heading3"]))
                    coherence_stats = statistical_summary["coherence_analysis"]
                    coherence_text = f"""
                    Mean Coherence: {coherence_stats['statistics']['mean_coherence']:.3f}<br/>
                    <i>{coherence_stats['interpretation']}</i>
                    """
                    story.append(Paragraph(coherence_text, styles["Normal"]))
                    story.append(Spacer(1, 6))

                except Exception as e:
                    story.append(
                        Paragraph(f"Statistical analysis failed: {str(e)}", styles["Normal"])
                    )
                    story.append(Spacer(1, 12))

            # Event log
            story.append(PageBreak())
            story.append(Paragraph("<b>Event Log</b>", styles["Heading2"]))
            story.append(Spacer(1, 6))

            # Add event log as preformatted text
            log_content = self.log_text.get(1.0, tk.END)
            log_style = ParagraphStyle(
                "LogStyle", parent=styles["Normal"], fontSize=8, fontName="Courier", leftIndent=20
            )
            story.append(Paragraph(log_content.replace("\n", "<br/>"), log_style))

            # Build PDF
            doc.build(story)
            self._log_event(f"PDF report saved: {filename}")

        except ImportError:
            raise Exception(
                "PDF generation requires reportlab. Please install with: pip install reportlab"
            )
        except Exception as e:
            raise Exception(f"PDF generation failed: {str(e)}")

    def _show_statistical_analysis(self) -> None:
        """Show detailed statistical analysis dialog."""
        if not self.apgi_simulation:
            messagebox.showwarning("No System", "Please start the simulation first")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Statistical Analysis")
        dialog.geometry("800x600")
        dialog.resizable(True, True)

        # Center dialog
        dialog.transient(self.root)
        dialog.grab_set()

        # Create scrolled text widget for results
        text_widget = scrolledtext.ScrolledText(dialog, font=("Courier", 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Analysis button
        def run_analysis() -> None:
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, "Running statistical analysis...\n\n")

            try:
                from apgi_framework.analysis import SystemAnalyzer

                analyzer = SystemAnalyzer(self.apgi_simulation.config)  # type: ignore[union-attr]

                # Correlation analysis
                text_widget.insert(tk.END, "CORRELATION ANALYSIS\n")
                text_widget.insert(tk.END, "=" * 40 + "\n\n")
                correlations = analyzer.compute_correlation_analysis(self.apgi_simulation.history)  # type: ignore[union-attr]

                for var1, corr_dict in correlations.items():
                    text_widget.insert(tk.END, f"{var1} correlations:\n")
                    for var2, stats in corr_dict.items():
                        corr = stats["correlation"]
                        p_val = stats["p_value"]
                        significance = (
                            "***"
                            if p_val < 0.001
                            else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                        )
                        text_widget.insert(
                            tk.END, f"  {var2}: r={corr:.3f} {significance} (p={p_val:.3f})\n"
                        )
                    text_widget.insert(tk.END, "\n")

                # Frequency analysis
                text_widget.insert(tk.END, "\nFREQUENCY DOMAIN ANALYSIS\n")
                text_widget.insert(tk.END, "=" * 40 + "\n\n")
                freq_analysis = analyzer.compute_frequency_analysis(self.apgi_simulation.history)  # type: ignore[union-attr]

                for signal_name, freq_data in freq_analysis.items():
                    text_widget.insert(tk.END, f"{signal_name.upper()}:\n")
                    text_widget.insert(
                        tk.END,
                        f"  Dominant Frequencies: {freq_data['dominant_frequencies']['freqs']}\n",
                    )
                    text_widget.insert(
                        tk.END, f"  Spectral Entropy: {freq_data['spectral_entropy']:.3f}\n"
                    )
                    text_widget.insert(tk.END, f"  Total Power: {freq_data['total_power']:.3f}\n\n")

                # Stationarity analysis
                text_widget.insert(tk.END, "STATIONARITY ANALYSIS\n")
                text_widget.insert(tk.END, "=" * 40 + "\n\n")
                stationarity = analyzer.compute_stationarity_analysis(self.apgi_simulation.history)  # type: ignore[union-attr]

                for signal_name, stats in stationarity.items():
                    text_widget.insert(tk.END, f"{signal_name.upper()}:\n")
                    text_widget.insert(tk.END, f"  ADF Test p-value: {stats['adf_pvalue']:.4f}\n")
                    text_widget.insert(tk.END, f"  Trend Strength: {stats['trend_strength']:.6f}\n")
                    stationary = "Stationary" if stats["adf_pvalue"] < 0.05 else "Non-stationary"
                    text_widget.insert(tk.END, f"  Result: {stationary}\n\n")

                text_widget.insert(tk.END, "Analysis complete!\n")

            except Exception as e:
                text_widget.insert(tk.END, f"Analysis failed: {str(e)}\n")
                # Removed traceback exposure for security

        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(button_frame, text="Run Analysis", command=run_analysis).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            button_frame,
            text="Save Results",
            command=lambda: self._save_analysis_results(text_widget),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def _save_analysis_results(self, text_widget: Any) -> None:
        """Save statistical analysis results to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Statistical Analysis",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )

        if filename:
            try:
                with open(filename, "w") as f:
                    f.write(text_widget.get(1.0, tk.END))
                self._log_event(f"Statistical analysis saved: {filename}")
                messagebox.showinfo("Success", f"Analysis saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save analysis:\n{str(e)}")

    def _show_docs(self) -> None:
        """Show documentation."""
        try:
            docs_path = Path(__file__).parent / "docs" / "GUI.md"
            with open(docs_path, "r", encoding="utf-8") as f:
                doc_text = f.read()
        except Exception as e:
            doc_text = f"Error loading documentation: {e}\n\nPlease check if docs/GUI.md exists."

        dialog = tk.Toplevel(self.root)
        dialog.title("Documentation")
        dialog.geometry("600x400")

        text_widget = scrolledtext.ScrolledText(dialog, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(1.0, doc_text)
        text_widget.config(state=tk.DISABLED)

        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)

    def _show_shortcuts(self) -> None:
        """Show keyboard shortcuts."""
        shortcuts = """Keyboard Shortcuts

File Operations:
Ctrl+N  - New Session
Ctrl+O  - Load Configuration
Ctrl+S  - Save Configuration
Ctrl+E  - Export Data
Ctrl+Q  - Exit

Simulation Control:
F5      - Start Simulation
F6      - Pause/Resume
F7      - Stop Simulation
F8      - Reset System
Ctrl+R  - Reset System

View Navigation:
Ctrl+P  - Toggle Parameter Panel
Ctrl+L  - Toggle Log Panel
Ctrl+Tab - Cycle Through Tabs
Ctrl+Shift+Tab - Reverse Tab Cycle
Tab     - Navigate Between Controls
Shift+Tab - Reverse Navigation

Zoom Controls:
Ctrl++  - Zoom In
Ctrl+-  - Zoom Out
Ctrl+0  - Fit to Window

Help:
F1      - Show Help
"""

        messagebox.showinfo("Keyboard Shortcuts", shortcuts)

    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = """APGI System v0.1.0

Allostatic Precision-Gated Ignition Framework

A computational model of consciousness based on:
- Active Inference (Free Energy Principle)
- Predictive Processing
- Global Workspace Theory
- Allostatic Regulation
- Somatic Markers

Developed using Python, NumPy, Matplotlib, and Tkinter

For more information, visit: https://github.com/apgiframework/apgi-system
"""

        messagebox.showinfo("About APGI System", about_text)

    # Helper Methods

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
        except Exception as e:
            # Ignore errors during shutdown
            logger.error(f"Error: {e}")

    def _update_status(self, message: str) -> None:
        """Update status bar message (thread-safe)."""
        if hasattr(self, "status_text") and self.status_text is not None:
            self.root.after(0, lambda msg=message: self._safe_update_status(msg))  # type: ignore[misc]

    def _safe_update_status(self, message: str) -> None:
        """Safely update status bar with existence check."""
        try:
            if hasattr(self, "status_text") and self.status_text.winfo_exists():
                self.status_text.config(text=message)
        except Exception as e:
            # Ignore errors during shutdown
            logger.error(f"Error: {e}")

    def _get_thread_safe_speed(self) -> float:
        """Get simulation speed in a thread-safe manner."""
        if hasattr(self, "_speed_value"):
            return self._speed_value
        return 1.0

    def _update_speed_cache(self) -> None:
        """Update thread-safe speed cache and label."""
        try:
            if hasattr(self, "speed_var") and hasattr(self, "speed_label"):
                if self.speed_label.winfo_exists():
                    self._speed_value = self.speed_var.get()
                    # Safely update label on main thread
                    new_text = f"{self._speed_value:.1f}x"
                    self.root.after(0, lambda: self.speed_label.config(text=new_text))
        except Exception as e:
            # Ignore errors during shutdown
            logger.debug(f"Error updating speed cache: {e}")

    def _setup_param_cache(self) -> None:
        """Set up thread-safe parameter cache with traces."""
        if hasattr(self, "param_vars"):
            for param_name, var in self.param_vars.items():
                # Initialize cache - handle both regular variables and tkinter variables
                if hasattr(var, "get"):
                    self._param_cache[param_name] = var.get()
                    # Add trace to update cache when variable changes
                    var.trace_add("write", lambda *args, n=param_name: self._update_param_cache(n))
                else:
                    # Regular variable
                    self._param_cache[param_name] = var

    def _update_param_cache(self, param_name: str) -> None:
        """Update parameter cache when tkinter variable changes."""
        try:
            if hasattr(self, "param_vars") and param_name in self.param_vars:
                var = self.param_vars[param_name]
                if hasattr(var, "get"):
                    self._param_cache[param_name] = var.get()
        except Exception as e:
            logger.error(f"Error: {e}")

    def _show_help(self) -> None:
        """Show comprehensive help dialog."""
        help_text = """APGI System Help

GETTING STARTED
---------------
1. Load Configuration: File > Load Configuration (YAML file)
2. Start Simulation: Click 'Start' or press F5
3. Monitor: Watch the plots and status indicators
4. Adjust Parameters: Use the sliders in the left panel
5. Stop Simulation: Click 'Stop' or press F7

KEYBOARD SHORTCUTS
------------------
File Operations:
  Ctrl+N - New Session
  Ctrl+O - Load Configuration
  Ctrl+S - Save Configuration
  Ctrl+E - Export Data
  Ctrl+Q - Exit

Simulation Control:
  F5 - Start Simulation
  F6 - Pause/Resume
  F7 - Stop Simulation
  F8 - Reset System

View Operations:
  Ctrl+P - Toggle Parameter Panel
  Ctrl+L - Toggle Log Panel
  Ctrl+0 - Fit to Window
  Ctrl++/- - Zoom In/Out

Navigation:
  Tab - Navigate Between Controls
  Ctrl+Tab - Cycle Through Tabs

TOOLS
-----
- Trigger Ignition: Manually trigger ignition event
- Induce Stressor: Apply stressor with configurable intensity
- Modulate Precision: Adjust extero/intero precision
- Inject Input: Add custom sensory input patterns

ANALYSIS
--------
- Ignition Statistics: View detailed ignition data
- Energy Budget Report: Monitor system energy usage
- Somatic Marker Analysis: Examine learned associations
- Self-Model Coherence: Check self-model integrity

TROUBLESHOOTING
---------------
- If simulation won't start: Check configuration file
- If plots are empty: Ensure simulation is running
- If system unresponsive: Check log panel for errors
- For detailed help: See APGI-Simulation-README.md

For more information, visit the documentation or contact support.
"""

        dialog = tk.Toplevel(self.root)
        dialog.title("APGI System Help")
        dialog.geometry("600x500")

        text_widget = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, font=("Arial", 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(1.0, help_text)
        text_widget.config(state=tk.DISABLED)

        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)

    def _toggle_parameter_panel(self) -> None:
        """Toggle parameter panel visibility."""
        try:
            # Robust check for frame existence and validity
            param_frame = getattr(self, "param_frame", None)
            if param_frame is not None:
                try:
                    if param_frame.winfo_exists():
                        current_state = param_frame.winfo_ismapped()
                        if current_state:
                            param_frame.pack_forget()
                            self._log_event("Parameter panel hidden")
                        else:
                            param_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                            self._log_event("Parameter panel shown")
                except tk.TclError:
                    # Widget destroyed but reference still exists
                    self._log_event("Warning: Parameter panel widget destroyed")
            else:
                self._log_event("Parameter panel not yet initialized")
        except Exception as e:
            self._log_event(f"Error toggling parameter panel: {e}")

    def _toggle_log_panel(self) -> None:
        """Toggle log panel visibility."""
        try:
            # Robust check for frame existence and validity
            log_frame = getattr(self, "log_frame", None)
            if log_frame is not None:
                try:
                    if log_frame.winfo_exists():
                        current_state = log_frame.winfo_ismapped()
                        if current_state:
                            log_frame.pack_forget()
                            self._log_event("Log panel hidden")
                        else:
                            log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                            self._log_event("Log panel shown")
                except tk.TclError:
                    # Widget destroyed but reference still exists
                    self._log_event("Warning: Log panel widget destroyed")
            else:
                self._log_event("Log panel not yet initialized")
        except Exception as e:
            self._log_event(f"Error toggling log panel: {e}")

    def _handle_tab_navigation(self, event: tk.Event) -> str:
        """Handle Tab key navigation."""
        next_widget = event.widget.tk_focusNext()
        if next_widget is not None:
            next_widget.focus_set()
        return "break"

    def _handle_shift_tab_navigation(self, event: tk.Event) -> str:
        """Handle Shift+Tab key navigation."""
        prev_widget = event.widget.tk_focusPrev()
        if prev_widget is not None:
            prev_widget.focus_set()
        return "break"

    def _cycle_notebook_tabs(self, direction: int) -> None:
        """Cycle through notebook tabs."""
        try:
            # Find the notebook widget
            if hasattr(self, "right_panel"):
                for child in self.right_panel.winfo_children():
                    if isinstance(child, ttk.Notebook):
                        current = child.index("current")  # type: ignore[no-untyped-call]
                        total = child.index("end") - 1  # type: ignore[no-untyped-call]
                        new_index = (current + direction) % total
                        child.select(new_index)  # type: ignore[no-untyped-call]
                        self._log_event(f"Switched to tab: {child.tab(new_index, 'text')}")  # type: ignore[no-untyped-call]
                        break
        except Exception as e:
            self._log_event(f"Error cycling tabs: {e}")


def main() -> None:
    """Main entry point for GUI application."""
    # Check dependencies before starting with user-friendly error handling
    try:
        from utils.dependency_checker import check_dependencies

        deps_ok = check_dependencies(core_only=True)

        if not deps_ok:
            print("=" * 60)
            print("DEPENDENCY CHECK FAILED")
            print("=" * 60)
            print("Some required dependencies are missing or incompatible.")
            print("\nTo resolve this issue:")
            print("1. Run: pip install -r requirements.txt")
            print("2. Or install with: pip install -e .")
            print("3. Check Python version (requires 3.8+)")
            print("\nFor detailed installation instructions, see:")
            print("- QUICKSTART_GUI.txt in the project root")
            print("- APGI-Simulation-README.md")
            print("=" * 60)

            # Try to show GUI dialog if possible
            try:
                from tkinter import messagebox

                error_root = tk.Tk()
                error_root.withdraw()  # Hide main window
                messagebox.showerror(
                    "Missing Dependencies",
                    "Required dependencies are missing.\n\n"
                    "Please run:\n"
                    "pip install -r requirements.txt\n\n"
                    "See QUICKSTART_GUI.txt for details.",
                )
                error_root.destroy()

            except Exception as e:
                logger.error(f"Error: {e}")

            return

    except ImportError as e:
        # Silently continue if dependency checker is not available
        logger.error(f"Error: {e}")
    except Exception as e:
        # Silently continue if dependency check fails
        logger.error(f"Error: {e}")

    # Create main application
    root = tk.Tk()
    app = APGIGui(root)

    # Start the main loop (app reference needed for GUI lifecycle)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        # Handle graceful shutdown
        if hasattr(app, "root") and app.root:
            app.root.quit()


if __name__ == "__main__":
    main()
