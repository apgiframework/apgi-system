"""Parameter Configuration Panel for APGI Framework GUI.

Extracted from the monolithic GUI to provide a focused component
for parameter configuration and validation.
"""

import json
import logging
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import Any, Callable, Dict, Optional

import customtkinter as ctk

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from apgi_framework.config import APGIParameters, ConfigManager, ExperimentalConfig
    from apgi_framework.logging.standardized_logging import get_logger
    from apgi_framework.validation import get_validator

    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    FRAMEWORK_AVAILABLE = False
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("parameter_config_panel")
    logger.error(f"APGI Framework components not available: {e}")

    # Show error message to user
    try:
        import tkinter.messagebox as msgbox

        msgbox.showerror(
            "Framework Not Available",
            "The APGI Framework is not properly installed.\n\n"
            "Parameter configuration will not function correctly.\n\n"
            "Please install the framework before using this panel.",
        )
    except ImportError:
        pass  # Can't show messagebox without tkinter


logger = (
    get_logger("parameter_config_panel")  # type: ignore
    if "get_logger" in globals()
    else logging.getLogger("parameter_config_panel")
)


class ParameterConfigPanel(ctk.CTkFrame):
    """Panel for configuring APGI parameters and experimental settings.

    Provides comprehensive parameter configuration with real-time validation,
    tooltips, and configuration management capabilities.
    """

    def __init__(self, parent: Any, config_manager: Optional[ConfigManager] = None) -> None:
        """Initialize the parameter configuration panel.

        Args:
            parent: Parent widget
            config_manager: Configuration manager instance
        """
        super().__init__(parent)

        self.config_manager = config_manager or ConfigManager()
        self.param_vars: Dict[str, Any] = {}
        self.exp_vars: Dict[str, Any] = {}
        self.param_entries: Dict[str, ctk.CTkBaseClass] = (
            {}
        )  # Store entry widgets for validation feedback
        self.exp_entries: Dict[str, ctk.CTkBaseClass] = {}

        # Initialize validator
        try:
            self.validator: Optional[Any] = get_validator()
        except Exception as e:
            logger.warning(f"Could not initialize validator: {e}")
            self.validator = None

        # Callbacks for external components
        self.on_config_changed: Optional[Callable] = None
        self.on_validation_error: Optional[Callable] = None

        self._create_widgets()  # type: ignore[no-untyped-call]
        self._load_current_config()  # type: ignore[no-untyped-call]
        self._setup_validation()  # type: ignore[no-untyped-call]

        logger.info("ParameterConfigPanel initialized")

    def _create_widgets(self) -> None:
        """Create parameter configuration widgets."""
        # Create scrollable frame for many parameters
        self.scrollable_frame = ctk.CTkScrollableFrame(self)
        self.scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # APGI Parameters section
        self._create_apgi_parameters_section()

        # Experimental Configuration section
        self._create_experimental_config_section()

        # Control buttons section
        self._create_control_buttons()

    def _create_apgi_parameters_section(self) -> None:
        """Create APGI parameters configuration section."""
        apgi_frame = ctk.CTkFrame(self.scrollable_frame)
        apgi_frame.pack(fill="x", padx=5, pady=5)

        apgi_title = ctk.CTkLabel(
            apgi_frame, text="APGI Parameters", font=ctk.CTkFont(size=14, weight="bold")
        )
        apgi_title.pack(padx=10, pady=(10, 5))

        # APGI parameter definitions
        apgi_params = [
            (
                "extero_precision",
                "Exteroceptive Precision",
                2.0,
                "Precision of exteroceptive signals (0.01-10.0, typical: 0.5-5.0)",
                (0.01, 10.0),
            ),
            (
                "intero_precision",
                "Interoceptive Precision",
                1.5,
                "Precision of interoceptive signals (0.01-10.0, typical: 0.5-5.0)",
                (0.01, 10.0),
            ),
            (
                "extero_error",
                "Exteroceptive Error",
                1.2,
                "Exteroceptive prediction error as z-score (-10 to 10, typical: -3 to 3)",
                (-10.0, 10.0),
            ),
            (
                "intero_error",
                "Interoceptive Error",
                0.8,
                "Interoceptive prediction error as z-score (-10 to 10, typical: -3 to 3)",
                (-10.0, 10.0),
            ),
            (
                "somatic_gain",
                "Somatic Gain",
                1.3,
                "Somatic marker gain modulating interoceptive precision (0.01-5.0, typical: 0.5-2.0)",
                (0.01, 5.0),
            ),
            (
                "threshold",
                "Threshold",
                3.5,
                "Ignition threshold for conscious access (0.1-10.0, typical: 2.0-5.0)",
                (0.1, 10.0),
            ),
            (
                "steepness",
                "Steepness",
                2.0,
                "Steepness of sigmoid function (0.1-10.0, typical: 1.0-3.0)",
                (0.1, 10.0),
            ),
        ]

        for param, label, default, tooltip, valid_range in apgi_params:
            self._create_parameter_row(
                apgi_frame, param, label, default, tooltip, valid_range, "apgi"
            )

    def _create_experimental_config_section(self) -> None:
        """Create experimental configuration section."""
        exp_frame = ctk.CTkFrame(self.scrollable_frame)
        exp_frame.pack(fill="x", padx=5, pady=5)

        exp_title = ctk.CTkLabel(
            exp_frame,
            text="Experimental Configuration",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        exp_title.pack(padx=10, pady=(10, 5))

        # Experimental parameter definitions
        exp_params = [
            (
                "n_trials",
                "Number of Trials",
                1000,
                "Number of trials per test (1-100000, typical: 50-1000)",
                (1, 100000),
                int,
            ),
            (
                "n_participants",
                "Number of Participants",
                100,
                "Number of participants (1-10000, typical: 20-200)",
                (1, 10000),
                int,
            ),
            (
                "random_seed",
                "Random Seed",
                "",
                "Random seed for reproducibility (optional)",
                None,
                str,
            ),
            (
                "p3b_threshold",
                "P3b Threshold (μV)",
                5.0,
                "P3b amplitude threshold in μV (1.0-20.0, typical: 3.0-7.0)",
                (1.0, 20.0),
                float,
            ),
            (
                "gamma_plv_threshold",
                "Gamma PLV Threshold",
                0.3,
                "Gamma phase-locking value threshold (0.05-0.8, typical: 0.2-0.4)",
                (0.05, 0.8),
                float,
            ),
            (
                "bold_z_threshold",
                "BOLD Z Threshold",
                3.1,
                "BOLD Z-score threshold (1.0-5.0, typical: 2.3-3.5)",
                (1.0, 5.0),
                float,
            ),
            (
                "pci_threshold",
                "PCI Threshold",
                0.4,
                "Perturbational Complexity Index threshold (0.1-0.8, typical: 0.3-0.5)",
                (0.1, 0.8),
                float,
            ),
            (
                "alpha_level",
                "Alpha Level",
                0.05,
                "Statistical significance level (0.001-0.1, typical: 0.05)",
                (0.001, 0.1),
                float,
            ),
            (
                "effect_size_threshold",
                "Effect Size Threshold",
                0.5,
                "Minimum effect size threshold (0.1-2.0, typical: 0.3-0.8)",
                (0.1, 2.0),
                float,
            ),
            (
                "power_threshold",
                "Power Threshold",
                0.8,
                "Statistical power threshold (0.5-0.99, typical: 0.8-0.95)",
                (0.5, 0.99),
                float,
            ),
        ]

        for param, label, default, tooltip, valid_range, param_type in exp_params:
            self._create_parameter_row(
                exp_frame,
                param,
                label,
                default,
                tooltip,
                valid_range,
                "exp",
                param_type,
            )

    def _create_parameter_row(
        self,
        parent_frame: Any,
        param_name: str,
        label: str,
        default_value: Any,
        tooltip: str,
        valid_range: Optional[tuple],
        section: str,
        param_type: type = float,
    ) -> None:
        """Create a row for parameter configuration.

        Args:
            parent_frame: Parent frame widget
            param_name: Parameter name
            label: Display label
            default_value: Default value
            tooltip: Help tooltip text
            valid_range: Valid range for validation
            section: Parameter section ('apgi' or 'exp')
            param_type: Parameter type (int, float, str)
        """
        # Create a frame for each parameter row
        param_row = ctk.CTkFrame(parent_frame)
        param_row.pack(fill="x", padx=5, pady=2)

        # Label
        ctk.CTkLabel(param_row, text=f"{label}:").pack(side="left", padx=5)

        # Create variable
        var: tk.Variable
        if param_type == int:
            var = tk.IntVar(value=default_value)
        elif param_type == float:
            var = tk.DoubleVar(value=default_value)
        else:  # str
            var = tk.StringVar(value=default_value)

        # Create entry widget
        entry = ctk.CTkEntry(param_row, textvariable=var, width=120)
        entry.pack(side="left", padx=5)

        # Add tooltip
        self._create_tooltip(entry, tooltip)

        # Store variables and entries
        if section == "apgi":
            self.param_vars[param_name] = var
            self.param_entries[param_name] = entry
        else:
            self.exp_vars[param_name] = var
            self.exp_entries[param_name] = entry

        # Validation indicator (skip for random_seed)
        if param_name != "random_seed":
            indicator = ctk.CTkLabel(param_row, text="✓", text_color="#00FF00", width=2)
            indicator.pack(side="left", padx=5)

            entry_key = f"{param_name}_indicator"
            if section == "apgi":
                self.param_entries[entry_key] = indicator
            else:
                self.exp_entries[entry_key] = indicator

        # Range label if specified
        if valid_range:
            range_text = f"[{valid_range[0]}, {valid_range[1]}]"
            range_label = ctk.CTkLabel(param_row, text=range_text, text_color="gray")
            range_label.pack(side="left", padx=(5, 10))

    def _create_control_buttons(self) -> None:
        """Create control buttons for configuration management."""
        control_frame = ctk.CTkFrame(self.scrollable_frame)
        control_frame.pack(fill="x", padx=5, pady=10)

        # Load configuration button
        load_btn = ctk.CTkButton(
            control_frame, text="Load Configuration", command=self._load_configuration
        )
        load_btn.pack(side="left", padx=5, pady=5)

        # Save configuration button
        save_btn = ctk.CTkButton(
            control_frame, text="Save Configuration", command=self._save_configuration
        )
        save_btn.pack(side="left", padx=5, pady=5)

        # Reset to defaults button
        reset_btn = ctk.CTkButton(
            control_frame, text="Reset to Defaults", command=self._reset_to_defaults
        )
        reset_btn.pack(side="left", padx=5, pady=5)

        # Validate all button
        validate_btn = ctk.CTkButton(
            control_frame, text="Validate All", command=self._validate_all_parameters
        )
        validate_btn.pack(side="left", padx=5, pady=5)

    def _create_tooltip(self, widget: Any, text: str) -> None:
        """Create a tooltip for a widget."""

        def on_enter(event: Any) -> None:
            tooltip = ctk.CTkToplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

            label = ctk.CTkLabel(tooltip, text=text, font=ctk.CTkFont(size=10), wraplength=300)
            label.pack(padx=5, pady=3)

            widget.tooltip = tooltip

        def on_leave(event: Any) -> None:
            if hasattr(widget, "tooltip"):
                widget.tooltip.destroy()
                del widget.tooltip

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _setup_validation(self) -> None:
        """Setup real-time validation for parameters."""
        # Setup validation for APGI parameters
        for param_name, var in self.param_vars.items():
            if hasattr(var, "trace_add"):
                # Newer tkinter syntax (Python 3.6+)
                var.trace_add("write", lambda *args, p=param_name: self._validate_parameter(p))
            else:
                # Legacy syntax
                var.trace("w", lambda *args, p=param_name: self._validate_parameter(p))

        # Setup validation for experimental parameters (except random_seed)
        for param_name, var in self.exp_vars.items():
            if param_name != "random_seed":
                if hasattr(var, "trace_add"):
                    var.trace_add("write", lambda *args, p=param_name: self._validate_parameter(p))
                else:
                    var.trace("w", lambda *args, p=param_name: self._validate_parameter(p))

    def _validate_parameter(self, param_name: str) -> None:
        """Validate a single parameter and update indicator."""
        try:
            # Get the parameter value
            if param_name in self.param_vars:
                var = self.param_vars[param_name]
            else:
                var = self.exp_vars[param_name]

            value = var.get()  # type: ignore[no-untyped-call]

            # Skip validation for empty string (random_seed)
            if value == "" and param_name == "random_seed":
                return

            # Convert to appropriate type
            if isinstance(value, str) and param_name != "random_seed":
                value = float(value)

            # Validate using validator if available
            if self.validator and param_name in self.param_vars:
                # This is an APGI parameter
                apgi_params = self._get_current_apgi_params()
                result = self.validator.validate_apgi_parameters(**apgi_params)

                if result.is_valid:
                    self._set_validation_indicator(param_name, True, "")
                else:
                    self._set_validation_indicator(param_name, False, result.get_message())
            else:
                # Basic range validation
                self._validate_parameter_range(param_name, value)

            # Notify of configuration change
            if self.on_config_changed:
                self.on_config_changed()

        except Exception as e:
            self._set_validation_indicator(param_name, False, str(e))

            if self.on_validation_error:
                self.on_validation_error(param_name, str(e))

    def _validate_parameter_range(self, param_name: str, value: float) -> None:
        """Validate parameter against expected ranges."""
        # Define validation ranges
        validation_ranges = {
            "n_trials": (1, 100000),
            "n_participants": (1, 10000),
            "p3b_threshold": (1.0, 20.0),
            "gamma_plv_threshold": (0.05, 0.8),
            "bold_z_threshold": (1.0, 5.0),
            "pci_threshold": (0.1, 0.8),
            "alpha_level": (0.001, 0.1),
            "effect_size_threshold": (0.1, 2.0),
            "power_threshold": (0.5, 0.99),
        }

        if param_name in validation_ranges:
            min_val, max_val = validation_ranges[param_name]
            if min_val <= value <= max_val:
                self._set_validation_indicator(param_name, True, "")
            else:
                self._set_validation_indicator(
                    param_name,
                    False,
                    f"Value {value} outside range [{min_val}, {max_val}]",
                )
        else:
            # No range validation available
            self._set_validation_indicator(param_name, True, "")

    def _set_validation_indicator(self, param_name: str, is_valid: bool, message: str) -> None:
        """Set validation indicator for a parameter."""
        # Find the indicator widget
        indicator_key = f"{param_name}_indicator"

        if param_name in self.param_vars:
            entries = self.param_entries
        else:
            entries = self.exp_entries

        if indicator_key in entries:
            indicator = entries[indicator_key]
            if is_valid:
                indicator.configure(text="✓", text_color="#00FF00")
            else:
                indicator.configure(text="✗", text_color="#FF0000")

                # Show error message
                if message:
                    logger.warning(f"Parameter {param_name}: {message}")

    def _get_current_apgi_params(self) -> Dict[str, float]:
        """Get current APGI parameter values."""
        return {param: float(var.get()) for param, var in self.param_vars.items()}  # type: ignore

    def _get_current_exp_params(self) -> Dict[str, Any]:
        """Get current experimental parameter values."""
        params: Dict[str, Any] = {}
        for param, var in self.exp_vars.items():
            value = var.get()  # type: ignore
            if param == "random_seed" and value == "":
                params[param] = None
            else:
                params[param] = value
        return params

    def _load_current_config(self) -> None:
        """Load current configuration from config manager."""
        try:
            config: Any = self.config_manager.load_config()  # type: ignore[call-arg, func-returns-value]

            # Load APGI parameters
            apgi_config = config.get("apgi_parameters", {})
            for param, var in self.param_vars.items():
                if param in apgi_config:
                    var.set(apgi_config[param])

            # Load experimental parameters
            exp_config = config.get("experimental_config", {})
            for param, var in self.exp_vars.items():
                if param in exp_config:
                    value = exp_config[param]
                    if value is None and param == "random_seed":
                        var.set("")
                    else:
                        var.set(value)

            logger.info("Configuration loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")

    def _load_configuration(self) -> None:
        """Load configuration from file."""
        try:
            from tkinter import filedialog

            file_path = filedialog.askopenfilename(
                title="Load Configuration",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )

            if file_path:
                with open(file_path, "r") as f:
                    config = json.load(f)

                # Apply configuration
                self._apply_configuration(config)

                messagebox.showinfo("Success", "Configuration loaded successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {e}")
            logger.error(f"Failed to load configuration: {e}")

    def _save_configuration(self) -> None:
        """Save current configuration to file."""
        try:
            from tkinter import filedialog

            file_path = filedialog.asksaveasfilename(
                title="Save Configuration",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )

            if file_path:
                config = self._get_configuration_dict()

                with open(file_path, "w") as f:
                    json.dump(config, f, indent=2)

                messagebox.showinfo("Success", "Configuration saved successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
            logger.error(f"Failed to save configuration: {e}")

    def _reset_to_defaults(self) -> None:
        """Reset all parameters to default values."""
        if messagebox.askyesno("Reset Parameters", "Reset all parameters to default values?"):
            try:
                # Reset APGI parameters to defaults
                apgi_defaults = {
                    "extero_precision": 2.0,
                    "intero_precision": 1.5,
                    "extero_error": 1.2,
                    "intero_error": 0.8,
                    "somatic_gain": 1.3,
                    "threshold": 3.5,
                    "steepness": 2.0,
                }

                for param, default in apgi_defaults.items():
                    if param in self.param_vars:
                        self.param_vars[param].set(default)

                # Reset experimental parameters to defaults
                exp_defaults = {
                    "n_trials": 1000,
                    "n_participants": 100,
                    "random_seed": "",
                    "p3b_threshold": 5.0,
                    "gamma_plv_threshold": 0.3,
                    "bold_z_threshold": 3.1,
                    "pci_threshold": 0.4,
                    "alpha_level": 0.05,
                    "effect_size_threshold": 0.5,
                    "power_threshold": 0.8,
                }

                for param, default in exp_defaults.items():  # type: ignore[assignment]
                    if param in self.exp_vars:
                        self.exp_vars[param].set("" if param == "random_seed" else str(default))

                messagebox.showinfo("Success", "Parameters reset to defaults")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset parameters: {e}")
                logger.error(f"Failed to reset parameters: {e}")

    def _validate_all_parameters(self) -> None:
        """Validate all parameters and show results."""
        try:
            # Validate APGI parameters
            apgi_params = self._get_current_apgi_params()
            if self.validator:
                result = self.validator.validate_apgi_parameters(**apgi_params)

                if result.is_valid:
                    messagebox.showinfo("Validation", "All APGI parameters are valid")
                else:
                    messagebox.showwarning(
                        "Validation", f"APGI parameter issues:\n{result.get_message()}"
                    )
            else:
                messagebox.showinfo(
                    "Validation", "Basic validation passed (validator not available)"
                )

            # Validate experimental parameters
            exp_params = self._get_current_exp_params()
            validation_errors = []

            for param, value in exp_params.items():
                if param == "random_seed":
                    continue

                try:
                    self._validate_parameter_range(param, float(value))
                except Exception as e:
                    validation_errors.append(f"{param}: {e}")

            if validation_errors:
                messagebox.showwarning(
                    "Validation",
                    "Experimental parameter issues:\n" + "\n".join(validation_errors),
                )
            else:
                messagebox.showinfo("Validation", "All experimental parameters are valid")

        except Exception as e:
            messagebox.showerror("Error", f"Validation failed: {e}")
            logger.error(f"Validation failed: {e}")

    def _apply_configuration(self, config: Dict[str, Any]) -> None:
        """Apply configuration dictionary to widgets."""
        # Apply APGI parameters
        apgi_config = config.get("apgi_parameters", {})
        for param, value in apgi_config.items():
            if param in self.param_vars:
                self.param_vars[param].set(value)

        # Apply experimental parameters
        exp_config = config.get("experimental_config", {})
        for param, value in exp_config.items():
            if param in self.exp_vars:
                if param == "random_seed" and value is None:
                    self.exp_vars[param].set("")
                else:
                    self.exp_vars[param].set(value)

    def _get_configuration_dict(self) -> Dict[str, Any]:
        """Get current configuration as dictionary."""
        return {
            "apgi_parameters": self._get_current_apgi_params(),
            "experimental_config": self._get_current_exp_params(),
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
        }

    def get_apgi_parameters(self) -> APGIParameters:
        """Get APGI parameters object."""
        try:
            apgi_params = self._get_current_apgi_params()
            return APGIParameters(**apgi_params)
        except Exception as e:
            logger.error(f"Failed to create APGIParameters: {e}")
            return APGIParameters()

    def get_experimental_config(self) -> ExperimentalConfig:
        """Get experimental configuration object."""
        try:
            exp_params = self._get_current_exp_params()
            return ExperimentalConfig(**exp_params)
        except Exception as e:
            logger.error(f"Failed to create ExperimentalConfig: {e}")
            return ExperimentalConfig()

    def set_config_changed_callback(self, callback: Callable) -> None:
        """Set callback for configuration changes."""
        self.on_config_changed = callback

    def set_validation_error_callback(self, callback: Callable) -> None:
        """Set callback for validation errors."""
        self.on_validation_error = callback


# Factory function for easy instantiation
def create_parameter_config_panel(
    parent: Any, config_manager: Optional[ConfigManager] = None
) -> ParameterConfigPanel:
    """Create a parameter configuration panel with default settings.

    Args:
        parent: Parent widget
        config_manager: Optional configuration manager

    Returns:
        Configured ParameterConfigPanel instance
    """
    return ParameterConfigPanel(parent, config_manager)
