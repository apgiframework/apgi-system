"""
APGI System - GUI Application
"""

import logging
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from apgi_framework import platform_utils
from apgi_framework.platform_utils import get_data_dir
from apgi_framework.system import APGISystem
from apgi_gui.theme_manager import get_theme_manager

from .components.control_panel import ControlPanel
from .components.menu_bar import MenuBar
from .components.status_bar import StatusBar
from .components.visualization_panel import VisualizationPanel
from .controllers.simulation_controller import SimulationController
from .mediator import GUIMediator

logger = logging.getLogger(__name__)


class APGIGui:
    """Main GUI application for APGI System (Modular)."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("APGI Consciousness Modeling Framework")

        self.data_dir = get_data_dir()
        # Default configuration shipped with the app (works from source and bundled builds)
        self.config_path: Path = platform_utils.get_resource_path("config/default.yaml")
        self.theme_manager = get_theme_manager()

        self.system = APGISystem()
        self.data_buffers: Dict[str, List[float]] = {
            "ignition": [],
            "surprise": [],
            "extero_precision": [],
            "intero_precision": [],
            "somatic_gain": [],
            "metabolic_reserves": [],
            "allostatic_load": [],
        }
        self.time_buffer: List[float] = []
        self.max_buffer_points = 1000

        self.log_data: List[Dict[str, Any]] = []
        self.speed_var = tk.DoubleVar(value=1.0)
        self.auto_save_var = tk.BooleanVar(value=False)
        self.param_vars: Dict[str, tk.DoubleVar] = {}
        self.view_vars: Dict[str, tk.BooleanVar] = {
            "control_panel": tk.BooleanVar(value=True),
            "neural_activity": tk.BooleanVar(value=True),
            "interoception": tk.BooleanVar(value=True),
            "system_metrics": tk.BooleanVar(value=True),
        }

        # Pending injected sensory input: (array, remaining_steps)
        self._pending_input: Optional[tuple] = None

        # Initialize Simulation Controller
        self.sim_controller = SimulationController(
            self.system,
            callbacks={
                "on_step": self._on_simulation_step,
                "on_error": self._on_simulation_error,
                "on_reset": self._on_simulation_reset,
                "get_custom_input": self._get_custom_input,
            },
        )

        # Initialize GUI Mediator for decoupling UI from controller
        self.mediator = GUIMediator(self.system, self.sim_controller)
        self.mediator.register_callback("on_step", self._on_simulation_step)
        self.mediator.register_callback("on_error", self._on_simulation_error)
        self.mediator.register_callback("on_simulation_reset", self._on_simulation_reset)

        # Legacy/test compatibility attributes
        self.apgi_simulation: Optional[APGISystem] = self.system
        self.buffer_size = 1000

        # Process metrics caching (sample every 10 UI cycles = ~1 second)
        self._process: Optional[Any] = psutil.Process() if HAS_PSUTIL else None
        self._ui_update_counter = 0
        self._cached_memory_mb = 0.0

        # Setup Layout
        self._setup_layout()

        # Initialize Components
        self._setup_components()

        # Start update loop
        self._update_ui_loop()

    def _setup_layout(self) -> None:
        """Create main layout containers."""
        # Main Paned Window
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left Panel (Controls)
        self.left_frame = ttk.Frame(self.main_paned, width=350)
        self.main_paned.add(self.left_frame, weight=0)

        # Right Panel (Visualizations)
        self.right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_frame, weight=1)

    def _setup_components(self) -> None:
        """Initialize modular GUI components."""
        # Callbacks for MenuBar and ControlPanel
        callbacks = {
            "start_simulation": self._start_simulation,
            "pause_simulation": self._pause_simulation,
            "stop_simulation": self._stop_simulation,
            "reset_simulation": self._reset_simulation,
            "new_session": self._new_session,
            "load_config": self._load_config,
            "save_config": self._save_config,
            "export_data": self._export_data,
            "export_plot": self._export_plot,
            "confirm_exit": self._confirm_exit,
            "toggle_auto_save": self._toggle_auto_save,
            "edit_parameters": self._edit_parameters,
            "edit_precision": self._edit_precision,
            "edit_threshold": self._edit_threshold,
            "reset_defaults": self._reset_defaults,
            "run_preset_task": self._run_preset_task,
            "zoom_in": self._zoom_in,
            "zoom_out": self._zoom_out,
            "zoom_fit": self._zoom_fit,
            "trigger_ignition": self._trigger_ignition,
            "induce_stressor": self._induce_stressor,
            "modulate_precision": self._modulate_precision,
            "inject_input": self._inject_input,
            "set_body_state": self._set_body_state,
            "show_diagnostics": self._show_diagnostics,
            "show_ignition_stats": self._show_ignition_stats,
            "show_energy_report": self._show_energy_report,
            "analyze_markers": self._analyze_markers,
            "analyze_coherence": self._analyze_coherence,
            "show_statistical_analysis": self._show_statistical_analysis,
            "generate_report": self._generate_report,
            "show_docs": self._show_docs,
            "show_shortcuts": self._show_shortcuts,
            "show_about": self._show_about,
            "toggle_control_panel": self._toggle_control_panel,
            "toggle_neural_activity": self._toggle_neural_activity,
            "toggle_interoception": self._toggle_interoception,
            "toggle_system_metrics": self._toggle_system_metrics,
            "speed_var": self.speed_var,
            "auto_save_var": self.auto_save_var,
            "param_vars": self.param_vars,
            "view_vars": self.view_vars,
        }

        self.menu_bar = MenuBar(self.root, callbacks)  # type: ignore[arg-type]
        self.status_bar = StatusBar(self.root)  # type: ignore[arg-type]
        self.control_panel = ControlPanel(self.left_frame, callbacks)  # type: ignore[arg-type]
        self.viz_panel = VisualizationPanel(self.right_frame)

        # Bind keyboard accelerators
        self._bind_accelerators()

    def _bind_accelerators(self) -> None:
        """Bind keyboard accelerators for menu actions."""
        # Determine modifier key based on platform
        is_mac = self.root.tk.call("tk", "windowingsystem") == "aqua"
        mod = "Command" if is_mac else "Control"

        # File menu accelerators
        self.root.bind(f"<{mod}-n>", lambda e: self._new_session())
        self.root.bind(f"<{mod}-o>", lambda e: self._load_config())
        self.root.bind(f"<{mod}-s>", lambda e: self._save_config())
        self.root.bind(f"<{mod}-e>", lambda e: self._export_data())
        self.root.bind(f"<{mod}-q>", lambda e: self._confirm_exit())

        # Simulation accelerators (function keys)
        self.root.bind("<F5>", lambda e: self._start_simulation())
        self.root.bind("<F6>", lambda e: self._pause_simulation())
        self.root.bind("<F7>", lambda e: self._stop_simulation())
        self.root.bind("<F8>", lambda e: self._reset_simulation())

    # Simulation Event Handlers
    def _on_simulation_step(self, data: Dict[str, Any]) -> None:
        """Handle simulation step data."""
        # Update buffers
        sim_time = data.get("time", 0.0)
        self.time_buffer.append(sim_time)

        mapping = {
            "ignition": "ignition_probability",
            "surprise": "surprise",
            "extero_precision": "extero_precision",
            "intero_precision": "intero_precision",
            "somatic_gain": "somatic_gain",
            "metabolic_reserves": "metabolic_reserves",
            "allostatic_load": "allostatic_load",
        }

        for key, data_key in mapping.items():
            val = data.get(data_key, 0.0)
            self.data_buffers[key].append(float(val))

        # Maintain buffer size
        if len(self.time_buffer) > self.max_buffer_points:
            self.time_buffer.pop(0)
            for buf in self.data_buffers.values():
                buf.pop(0)

    def _on_simulation_error(self, data: Dict[str, Any]) -> None:
        """Handle simulation error."""
        error_msg = data.get("message", "Unknown error")
        self.root.after(0, lambda: messagebox.showerror("Simulation Error", error_msg))
        self.root.after(0, self._stop_simulation)

    def _on_simulation_reset(self, data: Optional[Dict[str, Any]] = None) -> None:
        """Handle simulation reset."""
        self.time_buffer.clear()
        for buf in self.data_buffers.values():
            buf.clear()
        self.control_panel.log_event("Simulation reset")

    def _get_custom_input(self) -> Optional[Any]:
        """Get custom input from GUI for simulation step.

        Drains the pending injection set by _inject_input, returning the
        injected array for the specified number of steps, then None.
        """
        if self._pending_input is not None:
            arr, remaining = self._pending_input
            if remaining > 0:
                self._pending_input = (arr, remaining - 1)
                return arr
            else:
                self._pending_input = None
        return None

    # UI Update Loop
    def _update_ui_loop(self) -> None:
        """Periodically update UI components."""
        # Update status bar
        if self.mediator.is_running:
            current_time = self.time_buffer[-1] if self.time_buffer else 0.0
            # Rough FPS estimate (could be improved)
            fps = 10.0 if not self.mediator.is_paused else 0.0

            # Throttle memory sampling to every ~1 second (10 UI cycles)
            self._ui_update_counter += 1
            if self._ui_update_counter >= 10:
                self._ui_update_counter = 0
                if self._process is not None:
                    try:
                        self._cached_memory_mb = self._process.memory_info().rss / (1024 * 1024)
                    except Exception:
                        self._cached_memory_mb = 0.0

            self.status_bar.update_metrics(fps, self._cached_memory_mb, current_time)

            # Update plots
            self.viz_panel.update_plots(self.data_buffers, self.time_buffer)

            # Update status labels in control panel
            # Count ignition events (values > 0.5 threshold)
            ignition_count = sum(1 for v in self.data_buffers.get("ignition", []) if v > 0.5)
            status_metrics = {
                "Time": f"{current_time:.2f} s",
                "Ignition Events": str(ignition_count),
                "Workspace": "Active" if not self.mediator.is_paused else "Paused",
            }
            self.control_panel.update_status(status_metrics)

        # Update speed in controller via mediator
        self.mediator.set_simulation_speed(self.speed_var.get())

        self.root.after(100, self._update_ui_loop)

    # UI Action Callbacks (Proxies to Mediator)
    def _start_simulation(self) -> None:
        try:
            self.mediator.start_simulation()
            self.control_panel.start_btn.config(state=tk.DISABLED)
            self.control_panel.pause_btn.config(state=tk.NORMAL)
            self.control_panel.stop_btn.config(state=tk.NORMAL)
            self.control_panel.log_event("Simulation started")
            self.status_bar.set_status("Running...")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _pause_simulation(self) -> None:
        self.mediator.pause_simulation()
        is_paused = self.mediator.is_paused
        self.control_panel.pause_btn.config(text="▶ Resume" if is_paused else "⏸ Pause")
        self.control_panel.log_event("Simulation paused" if is_paused else "Simulation resumed")
        self.status_bar.set_status("Paused" if is_paused else "Running...")

    def _stop_simulation(self) -> None:
        self.mediator.stop_simulation()
        self.control_panel.start_btn.config(state=tk.NORMAL)
        self.control_panel.pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self.control_panel.stop_btn.config(state=tk.DISABLED)
        self.control_panel.log_event("Simulation stopped")
        self.status_bar.set_status("Stopped")

    def _reset_simulation(self) -> None:
        self.mediator.reset_simulation()
        self._stop_simulation()
        self.control_panel.log_event("Simulation reset")

    def _confirm_exit(self) -> None:
        if self.mediator.is_running:
            if not messagebox.askyesno("Confirm Exit", "Simulation is running. Exit anyway?"):
                return
        self.mediator.stop_simulation()
        self.root.destroy()

    # Config and session methods
    def _load_config_from_path(self, path: Path) -> bool:
        """Load configuration from a file path."""
        try:
            import json

            import yaml

            with open(path, "r") as f:
                if path.suffix in (".yaml", ".yml"):
                    config = yaml.safe_load(f)
                elif path.suffix == ".json":
                    config = json.load(f)
                else:
                    # Try YAML first, then JSON
                    content = f.read()
                    try:
                        config = yaml.safe_load(content)
                    except Exception:
                        config = json.loads(content)

            # Apply config to system if valid
            if isinstance(config, dict):
                self._apply_config(config)
                self.control_panel.log_event(f"Loaded config: {path.name}")
                return True
            else:
                logger.warning(f"Invalid config format in {path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            messagebox.showerror("Config Error", f"Failed to load config: {e}")
            return False

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply loaded configuration to the system."""
        # Apply parameters to system if they exist
        for key, value in config.items():
            if hasattr(self.system, key):
                try:
                    setattr(self.system, key, value)
                except Exception as e:
                    logger.warning(f"Could not set {key}: {e}")

    def _new_session(self):
        self._reset_simulation()
        self.control_panel.log_event("New session started")

    def _load_config(self):
        """Open file dialog to load configuration."""
        from tkinter import filedialog

        filetypes = [("Config files", "*.yaml *.yml *.json"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Load Configuration", filetypes=filetypes)
        if path:
            self._load_config_from_path(Path(path))

    def _save_config(self):
        """Open file dialog to save configuration."""
        from tkinter import filedialog

        import yaml

        filetypes = [("YAML files", "*.yaml"), ("JSON files", "*.json"), ("All files", "*.*")]
        path = filedialog.asksaveasfilename(
            title="Save Configuration", filetypes=filetypes, defaultextension=".yaml"
        )
        if path:
            try:
                # Extract current system parameters
                config = self._get_current_config()
                with open(path, "w") as f:
                    if path.endswith(".json"):
                        import json

                        json.dump(config, f, indent=2)
                    else:
                        yaml.dump(config, f, default_flow_style=False)
                self.control_panel.log_event(f"Saved config: {Path(path).name}")
            except Exception as e:
                logger.error(f"Failed to save config: {e}")
                messagebox.showerror("Save Error", f"Failed to save config: {e}")

    def _get_current_config(self) -> Dict[str, Any]:
        """Get current system configuration as a dictionary."""
        config: Dict[str, Any] = {}
        # Export relevant system attributes
        export_attrs = [
            "ignition_threshold",
            "extero_precision",
            "intero_precision",
            "arousal_level",
            "stress_level",
            "activity_level",
        ]
        for attr in export_attrs:
            if hasattr(self.system, attr):
                config[attr] = getattr(self.system, attr)
        return config

    def _export_data(self):
        """Export simulation data to file."""
        import csv
        import json
        from tkinter import filedialog

        if not self.time_buffer:
            messagebox.showwarning("Export Error", "No data to export. Run simulation first.")
            return

        filetypes = [("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")]
        path = filedialog.asksaveasfilename(
            title="Export Data", filetypes=filetypes, defaultextension=".csv"
        )
        if not path:
            return

        try:
            if path.endswith(".json"):
                # Export as JSON
                data = []
                for i, t in enumerate(self.time_buffer):
                    row = {"time": t}
                    for key, buf in self.data_buffers.items():
                        if i < len(buf):
                            row[key] = buf[i]
                    data.append(row)
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
            else:
                # Export as CSV
                with open(path, "w", newline="") as f:
                    writer = csv.writer(f)
                    headers = ["time"] + list(self.data_buffers.keys())
                    writer.writerow(headers)
                    for i, t in enumerate(self.time_buffer):
                        row = [t]
                        for key, buf in self.data_buffers.items():
                            row.append(buf[i] if i < len(buf) else "")
                        writer.writerow(row)

            self.control_panel.log_event(f"Exported data: {Path(path).name}")
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            messagebox.showerror("Export Error", f"Failed to export data: {e}")

    def _export_plot(self):
        """Save the current visualization to a file."""
        from tkinter import filedialog

        filetypes = [("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("All files", "*.*")]
        path = filedialog.asksaveasfilename(
            title="Export Plot", filetypes=filetypes, defaultextension=".png"
        )
        if not path:
            return
        try:
            # Try to save from the active notebook tab's figure
            active_tab = self.viz_panel.notebook.index(self.viz_panel.notebook.select())
            tab_keys = list(self.viz_panel.plots.keys())
            if active_tab < len(tab_keys):
                widget = self.viz_panel.plots[tab_keys[active_tab]]
                widget.fig.savefig(path, bbox_inches="tight", dpi=150)
                self.control_panel.log_event(f"Plot exported: {Path(path).name}")
            else:
                messagebox.showwarning("Export Plot", "No active plot tab to export.")
        except Exception as e:
            logger.error(f"Failed to export plot: {e}")
            messagebox.showerror("Export Error", f"Failed to export plot: {e}")

    def _toggle_auto_save(self):
        """Toggle automatic periodic data saving."""
        enabled = self.auto_save_var.get()
        state = "enabled" if enabled else "disabled"
        self.control_panel.log_event(f"Auto-save {state}")
        if enabled:
            self._schedule_auto_save()

    def _schedule_auto_save(self) -> None:
        """Periodically save data when auto-save is active."""
        if not self.auto_save_var.get():
            return
        if self.time_buffer:
            try:
                import csv

                save_path = self.data_dir / "auto_save.csv"
                with open(save_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["time"] + list(self.data_buffers.keys()))
                    for i, t in enumerate(self.time_buffer):
                        row = [t] + [
                            buf[i] if i < len(buf) else "" for buf in self.data_buffers.values()
                        ]
                        writer.writerow(row)
            except Exception as e:
                logger.warning(f"Auto-save failed: {e}")
        # Reschedule every 30 seconds
        self.root.after(30_000, self._schedule_auto_save)

    def _edit_parameters(self):
        """Open System Parameters dialog."""
        self._open_param_dialog(
            "System Parameters",
            [
                ("Arousal Level", "arousal_level", 0.0, 1.0),
                ("Stress Level", "stress_level", 0.0, 1.0),
                ("Activity Level", "activity_level", 0.0, 1.0),
                ("Learning Rate", "learning_rate", 0.001, 0.1),
                ("Attention Gain", "attention_gain", 0.5, 3.0),
            ],
        )

    def _edit_precision(self):
        """Open Precision Settings dialog."""
        self._open_param_dialog(
            "Precision Settings",
            [
                ("Exteroceptive Precision", "extero_precision", 0.1, 10.0),
                ("Interoceptive Precision", "intero_precision", 0.1, 10.0),
            ],
        )

    def _edit_threshold(self):
        """Open Ignition Threshold dialog."""
        self._open_param_dialog(
            "Ignition Threshold",
            [
                ("Ignition Threshold", "ignition_threshold", 1.0, 5.0),
            ],
        )

    def _open_param_dialog(self, title: str, params: list) -> None:
        """Generic parameter editing dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.resizable(False, False)
        dialog.grab_set()

        vars_map: Dict[str, tk.DoubleVar] = {}
        for i, (label, attr, lo, hi) in enumerate(params):
            current = getattr(self.system, attr, (lo + hi) / 2)
            var = tk.DoubleVar(value=float(current))
            vars_map[attr] = var

            ttk.Label(dialog, text=label + ":").grid(row=i, column=0, sticky=tk.W, padx=10, pady=4)
            ttk.Scale(dialog, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL, length=200).grid(
                row=i, column=1, padx=6, pady=4
            )
            ttk.Label(dialog, textvariable=var, width=6).grid(row=i, column=2, padx=4, pady=4)

        def apply() -> None:
            for attr, var in vars_map.items():
                try:
                    setattr(self.system, attr, var.get())
                except Exception as exc:
                    logger.warning(f"Could not set {attr}: {exc}")
            self.control_panel.log_event(f"{title} updated")
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=len(params), column=0, columnspan=3, pady=8)
        ttk.Button(btn_frame, text="Apply", command=apply).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=6)

    def _reset_defaults(self):
        """Reset all system parameters to default values."""
        defaults = {
            "ignition_threshold": 2.0,
            "extero_precision": 1.0,
            "intero_precision": 1.0,
            "arousal_level": 0.5,
            "stress_level": 0.0,
            "activity_level": 0.5,
        }
        for attr, val in defaults.items():
            if hasattr(self.system, attr):
                try:
                    setattr(self.system, attr, val)
                except Exception as e:
                    logger.warning(f"Could not reset {attr}: {e}")
        self.control_panel.log_event("Parameters reset to defaults")

    def _run_preset_task(self) -> None:
        """Open preset experimental task selector."""
        tasks = [
            "Attentional Blink",
            "Change Blindness",
            "Binocular Rivalry",
            "Masking Paradigm",
            "Iowa Gambling Task",
        ]
        task_params: Dict[str, Dict[str, float]] = {
            "Attentional Blink": {"arousal_level": 0.8, "stress_level": 0.3, "activity_level": 0.9},
            "Change Blindness": {"arousal_level": 0.4, "stress_level": 0.1, "activity_level": 0.5},
            "Binocular Rivalry": {
                "extero_precision": 2.0,
                "intero_precision": 0.5,
                "arousal_level": 0.6,
            },
            "Masking Paradigm": {
                "ignition_threshold": 3.5,
                "arousal_level": 0.5,
                "activity_level": 0.7,
            },
            "Iowa Gambling Task": {
                "stress_level": 0.6,
                "arousal_level": 0.7,
                "activity_level": 0.8,
            },
        }

        dialog = tk.Toplevel(self.root)
        dialog.title("Run Preset Task")
        dialog.resizable(False, False)
        dialog.grab_set()

        ttk.Label(dialog, text="Select experimental paradigm:").pack(padx=12, pady=(12, 4))
        listbox = tk.Listbox(dialog, selectmode=tk.SINGLE, height=len(tasks), width=30)
        for t in tasks:
            listbox.insert(tk.END, t)
        listbox.select_set(0)
        listbox.pack(padx=12, pady=4)

        def run_task() -> None:
            sel = listbox.curselection()
            if not sel:
                return
            name = tasks[sel[0]]
            params = task_params.get(name, {})
            for attr, val in params.items():
                if hasattr(self.system, attr):
                    try:
                        setattr(self.system, attr, val)
                    except Exception as exc:
                        logger.warning(f"Could not set {attr}: {exc}")
            self.control_panel.log_event(f"Preset task applied: {name}")
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=8)
        ttk.Button(btn_frame, text="Run Task", command=run_task).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=6)

    def _zoom_in(self):
        """Scale up the active visualization plot axes."""
        self._adjust_zoom(0.75)

    def _zoom_out(self):
        """Scale out the active visualization plot axes."""
        self._adjust_zoom(1.33)

    def _zoom_fit(self):
        """Fit active visualization to window."""
        try:
            active_tab = self.viz_panel.notebook.index(self.viz_panel.notebook.select())
            tab_keys = list(self.viz_panel.plots.keys())
            if active_tab < len(tab_keys):
                widget = self.viz_panel.plots[tab_keys[active_tab]]
                for ax in widget.fig.get_axes():
                    ax.autoscale()
                widget.fig.tight_layout()
                widget.canvas.draw_idle()
        except Exception as e:
            logger.debug(f"Zoom fit: {e}")

    def _adjust_zoom(self, factor: float) -> None:
        """Multiply the y-axis range of the active plot by factor."""
        try:
            active_tab = self.viz_panel.notebook.index(self.viz_panel.notebook.select())
            tab_keys = list(self.viz_panel.plots.keys())
            if active_tab < len(tab_keys):
                widget = self.viz_panel.plots[tab_keys[active_tab]]
                for ax in widget.fig.get_axes():
                    ylo, yhi = ax.get_ylim()
                    mid = (ylo + yhi) / 2
                    half = (yhi - ylo) / 2 * factor
                    ax.set_ylim(mid - half, mid + half)
                widget.canvas.draw_idle()
        except Exception as e:
            logger.debug(f"Zoom adjust: {e}")

    # Levels the manual ignition pulse drives the arousal / stress controls to.
    IGNITION_PULSE_LEVELS = {"arousal": 0.9, "stress": 0.8}

    def _trigger_ignition(self):
        """Trigger manual ignition by temporarily elevating arousal/stress.

        The arousal and stress sliders move to the pulse levels (never lower than they are)
        and the system attributes are raised with them; both are restored after 2 seconds.
        """
        try:
            orig_arousal = getattr(self.system, "arousal_level", 0.5)
            orig_stress = getattr(self.system, "stress_level", 0.0)
            setattr(self.system, "arousal_level", min(1.0, orig_arousal + 0.4))
            setattr(self.system, "stress_level", min(1.0, orig_stress + 0.3))

            originals = {}
            for name, level in self.IGNITION_PULSE_LEVELS.items():
                var = self.param_vars.get(name)
                if var is not None:
                    originals[name] = var.get()
                    var.set(max(originals[name], level))

            def restore() -> None:
                setattr(self.system, "arousal_level", orig_arousal)
                setattr(self.system, "stress_level", orig_stress)
                for name, value in originals.items():
                    self.param_vars[name].set(value)
                self.control_panel.log_event("Ignition pulse complete: parameters restored")

            self.control_panel.log_event("Manual ignition triggered: arousal/stress elevated")
            self.root.after(2000, restore)  # Restore after 2 seconds
        except Exception as e:
            logger.warning(f"Trigger ignition failed: {e}")
            self.control_panel.log_event(f"Trigger ignition error: {e}")

    def _induce_stressor(self):
        """Induce allostatic stressor by spiking stress and allostatic load."""
        self._open_param_dialog(
            "Induce Stressor",
            [
                ("Stress Intensity", "stress_level", 0.0, 1.0),
                ("Activity Level", "activity_level", 0.0, 1.0),
            ],
        )
        self.control_panel.log_event("Stressor dialog opened")

    def _modulate_precision(self):
        """Open precision modulation dialog."""
        self._open_param_dialog(
            "Modulate Precision",
            [
                ("Exteroceptive Precision", "extero_precision", 0.1, 10.0),
                ("Interoceptive Precision", "intero_precision", 0.1, 10.0),
            ],
        )
        self.control_panel.log_event("Precision modulation dialog opened")

    def _inject_input(self):
        """Open sensory input injection dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Inject Sensory Input")
        dialog.resizable(False, False)
        dialog.grab_set()

        ttk.Label(dialog, text="Sensory signal amplitude:").grid(
            row=0, column=0, padx=10, pady=8, sticky=tk.W
        )
        amp_var = tk.DoubleVar(value=1.0)
        ttk.Scale(
            dialog, from_=0.0, to=5.0, variable=amp_var, orient=tk.HORIZONTAL, length=200
        ).grid(row=0, column=1, padx=6, pady=8)
        ttk.Label(dialog, textvariable=amp_var, width=5).grid(row=0, column=2, padx=4)

        ttk.Label(dialog, text="Duration (steps):").grid(
            row=1, column=0, padx=10, pady=4, sticky=tk.W
        )
        dur_var = tk.IntVar(value=10)
        ttk.Spinbox(dialog, from_=1, to=100, textvariable=dur_var, width=6).grid(
            row=1, column=1, padx=6, pady=4, sticky=tk.W
        )

        def inject() -> None:
            amp = amp_var.get()
            # Store pending injection so simulation loop can pick it up
            self._pending_input = (np.ones(256) * amp, dur_var.get())
            self.control_panel.log_event(
                f"Sensory input injected: amplitude={amp:.2f}, duration={dur_var.get()} steps"
            )
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=8)
        ttk.Button(btn_frame, text="Inject", command=inject).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=6)

    def _set_body_state(self):
        """Open body state configuration dialog."""
        self._open_param_dialog(
            "Set Body State",
            [
                ("Arousal Level", "arousal_level", 0.0, 1.0),
                ("Stress Level", "stress_level", 0.0, 1.0),
                ("Activity Level", "activity_level", 0.0, 1.0),
            ],
        )
        self.control_panel.log_event("Body state dialog opened")

    def _show_diagnostics(self):
        """Show live system diagnostics in a scrollable window."""
        dialog = tk.Toplevel(self.root)
        dialog.title("System Diagnostics")
        dialog.geometry("480x360")

        text = tk.Text(dialog, wrap=tk.WORD, state=tk.DISABLED, font=("Courier", 10))
        scroll = ttk.Scrollbar(dialog, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        lines = ["=== APGI System Diagnostics ===", ""]
        # Thread state
        running = self.mediator.is_running
        paused = self.mediator.is_paused
        lines.append(f"Simulation Running : {running}")
        lines.append(f"Simulation Paused  : {paused}")
        thread = self.sim_controller.simulation_thread
        lines.append(f"Thread Alive       : {thread.is_alive() if thread else False}")
        lines.append("")
        # Buffer stats
        lines.append(f"Buffer Points      : {len(self.time_buffer)}")
        lines.append(f"Buffer Max         : {self.max_buffer_points}")
        lines.append("")
        # System attributes
        lines.append("--- System Parameters ---")
        for attr in [
            "arousal_level",
            "stress_level",
            "activity_level",
            "ignition_threshold",
            "extero_precision",
            "intero_precision",
        ]:
            val = getattr(self.system, attr, "N/A")
            lines.append(f"  {attr:<25}: {val}")
        lines.append("")
        # Memory
        if self._process is not None:
            try:
                mem_mb = self._process.memory_info().rss / (1024 * 1024)
                lines.append(f"Memory (RSS)       : {mem_mb:.1f} MB")
            except Exception:
                pass
        lines.append("")
        lines.append("=== End of Report ===")

        text.configure(state=tk.NORMAL)
        text.insert(tk.END, "\n".join(lines))
        text.configure(state=tk.DISABLED)
        self.control_panel.log_event("System diagnostics displayed")

    def _show_ignition_stats(self):
        """Show ignition statistics computed from live data buffers."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Ignition Statistics")
        dialog.geometry("420x300")

        text = tk.Text(dialog, wrap=tk.WORD, state=tk.DISABLED, font=("Courier", 10))
        scroll = ttk.Scrollbar(dialog, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        ign = self.data_buffers.get("ignition", [])
        total = len(ign)
        events = [v for v in ign if v > 0.5]
        n_events = len(events)
        duration = (
            (self.time_buffer[-1] - self.time_buffer[0]) if len(self.time_buffer) > 1 else 0.0
        )
        rate = n_events / duration if duration > 0 else 0.0
        mean_prob = float(np.mean(ign)) if ign else 0.0
        max_prob = float(np.max(ign)) if ign else 0.0

        lines = [
            "=== Ignition Statistics ===",
            "",
            f"  Total samples     : {total}",
            f"  Ignition events   : {n_events}  (threshold > 0.5)",
            f"  Simulation time   : {duration:.2f} s",
            f"  Event rate        : {rate:.3f} events/s",
            f"  Mean probability  : {mean_prob:.4f}",
            f"  Peak probability  : {max_prob:.4f}",
            "",
            "=== End of Report ===",
        ]

        text.configure(state=tk.NORMAL)
        text.insert(tk.END, "\n".join(lines))
        text.configure(state=tk.DISABLED)
        self.control_panel.log_event("Ignition statistics displayed")

    def _show_energy_report(self):
        """Show energy budget report from live data buffers."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Energy Budget Report")
        dialog.geometry("420x300")

        text = tk.Text(dialog, wrap=tk.WORD, state=tk.DISABLED, font=("Courier", 10))
        scroll = ttk.Scrollbar(dialog, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        reserves = self.data_buffers.get("metabolic_reserves", [])
        load = self.data_buffers.get("allostatic_load", [])

        def _stats(buf: list) -> str:
            if not buf:
                return "  No data"
            arr = np.array(buf, dtype=float)
            return (
                f"  Current : {arr[-1]:.4f}\n"
                f"  Mean    : {float(arr.mean()):.4f}\n"
                f"  Min     : {float(arr.min()):.4f}\n"
                f"  Max     : {float(arr.max()):.4f}"
            )

        lines = [
            "=== Energy Budget Report ===",
            "",
            "--- Metabolic Reserves ---",
            _stats(reserves),
            "",
            "--- Allostatic Load ---",
            _stats(load),
            "",
            "=== End of Report ===",
        ]

        text.configure(state=tk.NORMAL)
        text.insert(tk.END, "\n".join(lines))
        text.configure(state=tk.DISABLED)
        self.control_panel.log_event("Energy budget report displayed")

    def _analyze_markers(self):
        """Show somatic marker analysis."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Somatic Marker Analysis")
        dialog.geometry("420x280")

        text = tk.Text(dialog, wrap=tk.WORD, state=tk.DISABLED, font=("Courier", 10))
        scroll = ttk.Scrollbar(dialog, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        gain = self.data_buffers.get("somatic_gain", [])
        arr = np.array(gain, dtype=float) if gain else np.array([])

        lines = [
            "=== Somatic Marker Analysis ===",
            "",
            f"  Buffer samples   : {len(gain)}",
        ]
        if arr.size > 0:
            lines += [
                f"  Current gain     : {float(arr[-1]):.4f}",
                f"  Mean gain        : {float(arr.mean()):.4f}",
                f"  Std deviation    : {float(arr.std()):.4f}",
                f"  Min / Max        : {float(arr.min()):.4f} / {float(arr.max()):.4f}",
            ]
        else:
            lines.append("  No data — run simulation first.")
        lines += ["", "=== End of Report ==="]

        text.configure(state=tk.NORMAL)
        text.insert(tk.END, "\n".join(lines))
        text.configure(state=tk.DISABLED)
        self.control_panel.log_event("Somatic marker analysis displayed")

    def _analyze_coherence(self):
        """Show self-model coherence analysis (surprise-based proxy)."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Self-Model Coherence")
        dialog.geometry("420x280")

        text = tk.Text(dialog, wrap=tk.WORD, state=tk.DISABLED, font=("Courier", 10))
        scroll = ttk.Scrollbar(dialog, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        surprise = self.data_buffers.get("surprise", [])
        arr = np.array(surprise, dtype=float) if surprise else np.array([])
        coherence = 1.0 - float(np.clip(arr.mean(), 0, 1)) if arr.size > 0 else 0.0

        lines = [
            "=== Self-Model Coherence ===",
            "  (Derived from inverse of mean surprise)",
            "",
            f"  Samples          : {len(surprise)}",
        ]
        if arr.size > 0:
            lines += [
                f"  Mean surprise    : {float(arr.mean()):.4f}",
                f"  Coherence index  : {coherence:.4f}  (1=perfect, 0=incoherent)",
                f"  Surprise std     : {float(arr.std()):.4f}",
            ]
        else:
            lines.append("  No data — run simulation first.")
        lines += ["", "=== End of Report ==="]

        text.configure(state=tk.NORMAL)
        text.insert(tk.END, "\n".join(lines))
        text.configure(state=tk.DISABLED)
        self.control_panel.log_event("Self-model coherence analysis displayed")

    def _show_statistical_analysis(self):
        """Open statistical analysis dialog with descriptive stats for all buffers."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Statistical Analysis")
        dialog.geometry("520x400")

        text = tk.Text(dialog, wrap=tk.WORD, state=tk.DISABLED, font=("Courier", 10))
        scroll = ttk.Scrollbar(dialog, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        lines = ["=== Statistical Analysis ===", ""]
        lines.append(f"  {'Channel':<25} {'N':>5} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
        lines.append("  " + "-" * 67)
        for key, buf in self.data_buffers.items():
            arr = np.array(buf, dtype=float) if buf else np.array([])
            if arr.size > 0:
                lines.append(
                    f"  {key:<25} {arr.size:>5} {arr.mean():>9.4f} {arr.std():>9.4f} "
                    f"{arr.min():>9.4f} {arr.max():>9.4f}"
                )
            else:
                lines.append(f"  {key:<25} {'—':>5}")
        lines += ["", "=== End of Report ==="]

        text.configure(state=tk.NORMAL)
        text.insert(tk.END, "\n".join(lines))
        text.configure(state=tk.DISABLED)
        self.control_panel.log_event("Statistical analysis displayed")

    def _generate_report(self):
        """Generate and optionally save a comprehensive text report."""
        # Build report text
        import datetime
        from tkinter import filedialog

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = (
            (self.time_buffer[-1] - self.time_buffer[0]) if len(self.time_buffer) > 1 else 0.0
        )
        ign = self.data_buffers.get("ignition", [])
        n_events = sum(1 for v in ign if v > 0.5)
        rate = n_events / duration if duration > 0 else 0.0

        lines = [
            "APGI System - Comprehensive Session Report",
            "=" * 50,
            f"Generated    : {ts}",
            f"Sim Duration : {duration:.2f} s",
            f"Buffer Points: {len(self.time_buffer)}",
            "",
            "--- Ignition Summary ---",
            f"  Events (>0.5): {n_events}",
            f"  Rate         : {rate:.3f} events/s",
            "",
            "--- Descriptive Statistics ---",
            f"  {'Channel':<25} {'N':>5} {'Mean':>9} {'Std':>9}",
            "  " + "-" * 50,
        ]
        for key, buf in self.data_buffers.items():
            arr = np.array(buf, dtype=float) if buf else np.array([])
            if arr.size > 0:
                lines.append(f"  {key:<25} {arr.size:>5} {arr.mean():>9.4f} {arr.std():>9.4f}")
            else:
                lines.append(f"  {key:<25} {'—':>5}")
        lines += ["", "--- System Parameters ---"]
        for attr in [
            "arousal_level",
            "stress_level",
            "activity_level",
            "ignition_threshold",
            "extero_precision",
            "intero_precision",
        ]:
            lines.append(f"  {attr:<25}: {getattr(self.system, attr, 'N/A')}")
        lines += ["", "=" * 50, "End of Report"]
        report_text = "\n".join(lines)

        # Show preview
        preview = tk.Toplevel(self.root)
        preview.title("Report Preview")
        preview.geometry("560x440")
        text_w = tk.Text(preview, wrap=tk.WORD, font=("Courier", 10))
        sb = ttk.Scrollbar(preview, command=text_w.yview)
        text_w.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        text_w.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        text_w.insert(tk.END, report_text)
        text_w.configure(state=tk.DISABLED)

        def save_report() -> None:
            path = filedialog.asksaveasfilename(
                title="Save Report",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                defaultextension=".txt",
            )
            if path:
                try:
                    with open(path, "w") as f:
                        f.write(report_text)
                    self.control_panel.log_event(f"Report saved: {Path(path).name}")
                except Exception as exc:
                    messagebox.showerror("Save Error", str(exc))

        btn_row = ttk.Frame(preview)
        btn_row.pack(pady=6)
        ttk.Button(btn_row, text="Save...", command=save_report).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="Close", command=preview.destroy).pack(side=tk.LEFT, padx=6)
        self.control_panel.log_event("Report generated")

    def _show_docs(self):
        """Show documentation dialog."""
        docs_text = """APGI System - Documentation

The APGI (Allostatic Precision-Gated Ignition) framework models consciousness 
as a dynamic process integrating interoceptive and exteroceptive signals.

Key Concepts:
- Ignition: Phase transitions to conscious access
- Precision: Confidence weights for predictions
- Allostasis: Energy regulation and homeostatic control
- Free Energy: Prediction error minimization

Keyboard Shortcuts:
  F5 - Start Simulation
  F6 - Pause/Resume
  F7 - Stop Simulation
  F8 - Reset System
  Ctrl+N - New Session
  Ctrl+O - Load Config
  Ctrl+S - Save Config
  Ctrl+E - Export Data
  Ctrl+Q - Exit

For more information, see the project documentation."""
        messagebox.showinfo("APGI Documentation", docs_text)

    def _show_shortcuts(self):
        """Show keyboard shortcuts dialog."""
        shortcuts_text = """Keyboard Shortcuts

Simulation Controls:
  F5     - Start Simulation
  F6     - Pause/Resume
  F7     - Stop Simulation
  F8     - Reset System

File Operations:
  Ctrl+N - New Session
  Ctrl+O - Load Configuration
  Ctrl+S - Save Configuration
  Ctrl+E - Export Data
  Ctrl+Q - Exit Application

Note: Advanced features (Edit dialogs, Tools interventions, 
Analysis reports) are accessible via the menu system."""
        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)

    def _show_about(self):
        """Show about dialog."""
        about_text = """APGI Consciousness Modeling Framework

Version: 1.0.0

A computational framework for modeling consciousness through
the lens of allostatic regulation and precision-gated ignition.

Built with Python, Tkinter, and NumPy.

MIT License"""
        messagebox.showinfo("About APGI System", about_text)

    def _toggle_control_panel(self):
        """Toggle control panel (left pane) visibility."""
        current = self.view_vars["control_panel"].get()
        new_state = not current
        self.view_vars["control_panel"].set(new_state)
        if new_state:
            # Show: re-add the left frame if it was removed
            try:
                self.main_paned.add(self.left_frame, weight=0, before=self.right_frame)
            except Exception:
                pass
        else:
            try:
                self.main_paned.forget(self.left_frame)
            except Exception:
                pass
        self.control_panel.log_event(f"Control panel {'shown' if new_state else 'hidden'}")

    def _toggle_neural_activity(self):
        """Toggle neural activity tab visibility."""
        current = self.view_vars["neural_activity"].get()
        self.view_vars["neural_activity"].set(not current)
        # Show or hide the tab in the notebook
        tab = self.viz_panel.tabs.get("neural")
        if tab:
            try:
                if not current:
                    self.viz_panel.notebook.add(tab, text="Neural Activity")
                else:
                    self.viz_panel.notebook.hide(tab)
            except Exception:
                pass
        self.control_panel.log_event(f"Neural activity tab {'shown' if not current else 'hidden'}")

    def _toggle_interoception(self):
        """Toggle interoception tab visibility."""
        current = self.view_vars["interoception"].get()
        self.view_vars["interoception"].set(not current)
        tab = self.viz_panel.tabs.get("intero")
        if tab:
            try:
                if not current:
                    self.viz_panel.notebook.add(tab, text="Interoception")
                else:
                    self.viz_panel.notebook.hide(tab)
            except Exception:
                pass
        self.control_panel.log_event(f"Interoception tab {'shown' if not current else 'hidden'}")

    def _toggle_system_metrics(self):
        """Toggle system metrics tab visibility."""
        current = self.view_vars["system_metrics"].get()
        self.view_vars["system_metrics"].set(not current)
        tab = self.viz_panel.tabs.get("metrics")
        if tab:
            try:
                if not current:
                    self.viz_panel.notebook.add(tab, text="System Metrics")
                else:
                    self.viz_panel.notebook.hide(tab)
            except Exception:
                pass
        self.control_panel.log_event(f"System metrics tab {'shown' if not current else 'hidden'}")

    # Legacy/test compatibility properties
    @property
    def is_running(self) -> bool:
        """Legacy property for test compatibility."""
        return self.mediator.is_running

    @property
    def is_paused(self) -> bool:
        """Legacy property for test compatibility."""
        return self.mediator.is_paused

    # Legacy/test compatibility methods
    def _record_state(self, state: Dict[str, Any]) -> None:
        """Legacy method for test compatibility."""
        self.log_data.append(state)

    def _generate_input(self, t: float) -> Any:
        """Legacy method for test compatibility."""
        return np.random.randn(256) * 0.5

    @property
    def log_text(self) -> tk.Text:
        """Legacy property for test compatibility."""
        # Return a dummy Text widget if not available
        if not hasattr(self, "_log_text_widget"):
            self._log_text_widget = tk.Text(self.root, height=10, state=tk.DISABLED)
        return self._log_text_widget

    @property
    def status_text(self) -> tk.Label:
        """Legacy property for test compatibility."""
        # Return a dummy Label widget if not available
        if not hasattr(self, "_status_text_widget"):
            self._status_text_widget = tk.Label(self.root, text="")
        return self._status_text_widget

    def _log_event(self, message: str) -> None:
        """Legacy method for test compatibility."""
        self.control_panel.log_event(message)

    def _update_status(self, message: str) -> None:
        """Legacy method for test compatibility."""
        self.status_bar.set_status(message)

    @property
    def status_labels(self) -> Dict[str, tk.Label]:
        """Legacy property for test compatibility."""
        # Return dummy labels if not available
        if not hasattr(self, "_status_labels_dict"):
            self._status_labels_dict = {
                "Time": tk.Label(self.root, text="0.00 s"),
                "Ignition Events": tk.Label(self.root, text="0"),
                "Workspace": tk.Label(self.root, text="Inactive"),
            }
        return self._status_labels_dict

    # Additional legacy/test compatibility methods
    def _update_plots(self) -> None:
        """Legacy method for test compatibility."""
        pass

    def _update_status_labels(self) -> None:
        """Legacy method for test compatibility."""
        pass

    def _apply_parameters(self) -> None:
        """Legacy method for test compatibility."""
        pass

    def _convert_to_tkinter_variables(self) -> None:
        """Legacy method for test compatibility."""
        pass

    # Button references for test compatibility
    @property
    def start_btn(self) -> tk.Button:
        """Legacy property for test compatibility."""
        if not hasattr(self, "_start_btn_widget"):
            self._start_btn_widget = tk.Button(self.root, text="Start")
        return self._start_btn_widget

    @property
    def pause_btn(self) -> tk.Button:
        """Legacy property for test compatibility."""
        if not hasattr(self, "_pause_btn_widget"):
            self._pause_btn_widget = tk.Button(self.root, text="Pause")
        return self._pause_btn_widget

    @property
    def stop_btn(self) -> tk.Button:
        """Legacy property for test compatibility."""
        if not hasattr(self, "_stop_btn_widget"):
            self._stop_btn_widget = tk.Button(self.root, text="Stop")
        return self._stop_btn_widget

    @property
    def reset_btn(self) -> tk.Button:
        """Legacy property for test compatibility."""
        if not hasattr(self, "_reset_btn_widget"):
            self._reset_btn_widget = tk.Button(self.root, text="Reset")
        return self._reset_btn_widget


def main():
    root = tk.Tk()
    APGIGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
