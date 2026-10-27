"""Test Execution Panel for APGI Framework GUI.

Extracted from the monolithic GUI to provide a focused component
for running and managing falsification tests.
"""

import logging
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import Any, Callable, Dict, Optional, Union

import customtkinter as ctk

from apgi_framework.logging.standardized_logging import get_logger

logger = get_logger(__name__)


# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from apgi_framework.logging.standardized_logging import APGILogger, get_logger
    from apgi_framework.main_controller import MainApplicationController
    from apgi_framework.validation import get_validator

    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    FRAMEWORK_AVAILABLE = False
    temp_logger = logging.getLogger("test_execution_panel")
    temp_logger.error(f"APGI Framework components not available: {e}")

    # Show error message
    try:
        import tkinter.messagebox as msgbox

        msgbox.showerror(
            "Framework Not Available",
            "The APGI Framework is not properly installed.\n\n"
            "Test execution will not function correctly.\n\n"
            "Please install the framework before running tests.",
        )
    except ImportError:
        pass


logger: Union[APGILogger, logging.Logger] = (  # type: ignore[no-redef]
    get_logger("test_execution_panel")
    if "get_logger" in globals() and get_logger is not None
    else logging.getLogger("test_execution_panel")
)


class ExecutionPanel(ctk.CTkFrame):
    """Panel for running falsification tests with progress tracking and result management.

    Provides comprehensive test execution capabilities including parameter validation,
    progress tracking, real-time logging, and result display.
    """

    def __init__(
        self,
        parent: Any,
        test_name: str,
        controller: Optional[MainApplicationController] = None,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
        results_callback: Optional[Callable] = None,
    ):
        """Initialize the test execution panel.

        Args:
            parent: Parent widget
            test_name: Name of the test to run
            controller: Main application controller
            progress_callback: Callback for progress updates
            log_callback: Callback for log messages
            results_callback: Callback for test results
        """
        super().__init__(parent)

        self.test_name = test_name
        self.controller = controller or MainApplicationController()
        self.progress_callback = progress_callback
        self.log_callback = log_callback or self._default_log_callback
        self.results_callback = results_callback

        # Test execution state
        self.is_running = False
        self.test_controller = None
        self.current_thread: Optional[threading.Thread] = None
        self.test_results = None

        # Test parameters
        self.test_vars: Dict[str, Any] = {}

        # Callbacks for external components
        self.on_test_started: Optional[Callable] = None
        self.on_test_completed: Optional[Callable] = None
        self.on_test_failed: Optional[Callable] = None
        self.on_progress_updated: Optional[Callable] = None

        self._create_widgets()
        self._setup_test_parameters()  # type: ignore[no-untyped-call]

        logger.info(f"TestExecutionPanel initialized for test: {test_name}")

    def _create_widgets(self) -> None:
        """Create test execution widgets."""
        # Create scrollable frame
        self.scrollable_frame = ctk.CTkScrollableFrame(self)
        self.scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Test description section
        self._create_description_section()  # type: ignore[no-untyped-call]

        # Test parameters section
        self._create_parameters_section()  # type: ignore[no-untyped-call]

        # Control buttons section
        self._create_control_section()  # type: ignore[no-untyped-call]

        # Progress section
        self._create_progress_section()  # type: ignore[no-untyped-call]

        # Results section
        self._create_results_section()  # type: ignore[no-untyped-call]

    def _create_description_section(self) -> None:
        """Create test description section."""
        desc_frame = ctk.CTkFrame(self.scrollable_frame)
        desc_frame.pack(fill="x", padx=5, pady=5)

        desc_title = ctk.CTkLabel(
            desc_frame,
            text="Test Description",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        desc_title.pack(padx=10, pady=(10, 5))

        # Test descriptions
        descriptions = {
            "Primary": "Tests for full ignition signatures without consciousness. This test evaluates whether complete neural ignition signatures can occur without conscious awareness.",
            "Consciousness Without Ignition": "Tests for consciousness without ignition signatures. This test evaluates whether conscious awareness can occur without complete neural ignition.",
            "Threshold Insensitivity": "Tests neuromodulatory threshold dynamics. This test evaluates how different neuromodulatory states affect threshold sensitivity.",
            "Soma-Bias": "Tests interoceptive vs exteroceptive bias. This test evaluates the balance between interoceptive and exteroceptive precision.",
            "Cross-Species Validation": "Tests cross-species generalizability of findings. This test evaluates whether results generalize across different species.",
            "Clinical Biomarkers": "Tests clinical biomarker validity. This test evaluates the effectiveness of biomarkers for clinical applications.",
            "Threshold Detection": "Tests threshold detection paradigms. This test evaluates different methods for detecting sensory thresholds.",
        }

        desc_text = tk.Text(desc_frame, height=4, wrap="word")
        desc_text.pack(fill="x", padx=5, pady=5)
        desc_text.insert(
            "1.0",
            descriptions.get(self.test_name, "Falsification test for APGI Framework"),
        )
        desc_text.configure(state="disabled")

    def _create_parameters_section(self) -> None:
        """Create test parameters section."""
        self.params_frame = ctk.CTkFrame(self.scrollable_frame)
        self.params_frame.pack(fill="x", padx=5, pady=5)

        params_title = ctk.CTkLabel(
            self.params_frame,
            text="Test Parameters",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        params_title.pack(padx=10, pady=(10, 5))

        # Common parameters for all tests
        self._create_parameter_row(
            self.params_frame,
            "n_trials",
            "Number of Trials",
            1000,
            "Total number of trials to run",
        )

        # Test-specific parameters
        if self.test_name in ["Cross-Species Validation", "Clinical Biomarkers"]:
            self._create_parameter_row(
                self.params_frame,
                "n_participants",
                "Number of Participants",
                100,
                "Number of participants to simulate",
            )

        if self.test_name == "Threshold Insensitivity":
            self._create_parameter_row(
                self.params_frame,
                "n_drug_conditions",
                "Drug Conditions",
                5,
                "Number of drug conditions to test",
            )

        if self.test_name == "Cross-Species Validation":
            self._create_parameter_row(
                self.params_frame,
                "n_species",
                "Number of Species",
                3,
                "Number of species to include",
            )

        if self.test_name == "Clinical Biomarkers":
            self._create_parameter_row(
                self.params_frame,
                "clinical_condition",
                "Clinical Condition",
                "depression",
                "Clinical condition to test",
            )

            conditions = [
                "depression",
                "anxiety",
                "schizophrenia",
                "bipolar",
                "adhd",
                "autism",
            ]

            # Create dropdown for disorder selection
            disorder_frame = ctk.CTkFrame(self.params_frame)
            disorder_frame.pack(fill="x", pady=5)

            ctk.CTkLabel(disorder_frame, text="Disorder:").pack(anchor="w")
            disorder_var = tk.StringVar(value=conditions[0])
            disorder_menu = ctk.CTkOptionMenu(
                disorder_frame, variable=disorder_var, values=conditions
            )
            disorder_menu.pack(fill="x", pady=2)

        if self.test_name == "Threshold Detection":
            self._create_parameter_row(
                self.params_frame,
                "modality",
                "Modality",
                "visual",
                "Sensory modality for threshold testing",
            )
            self._create_parameter_row(
                self.params_frame,
                "method",
                "Method",
                "adaptive_staircase",
                "Threshold estimation method",
            )

            modalities = ["visual", "auditory", "interoceptive", "somatosensory"]

            # Create dropdown for modality selection
            modality_frame = ctk.CTkFrame(self.params_frame)
            modality_frame.pack(fill="x", pady=5)

            ctk.CTkLabel(modality_frame, text="Modality:").pack(anchor="w")
            modality_var = tk.StringVar(value=modalities[0])
            modality_menu = ctk.CTkOptionMenu(
                modality_frame, variable=modality_var, values=modalities
            )
            modality_menu.pack(fill="x", pady=2)

    def _create_parameter_row(
        self,
        parent_frame: Any,
        param_name: str,
        label: str,
        default_value: Any,
        tooltip: str,
    ) -> None:
        """Create a parameter input row."""
        param_row = ctk.CTkFrame(parent_frame)
        param_row.pack(fill="x", padx=5, pady=2)

        # Label
        label_widget = ctk.CTkLabel(param_row, text=f"{label}:")
        label_widget.pack(side="left", padx=(5, 10))
        self._create_tooltip(label_widget, tooltip)

        # Entry widget
        if isinstance(default_value, int):
            var = ctk.IntVar(value=default_value)
            entry = ctk.CTkEntry(param_row, textvariable=var, width=15)
        elif isinstance(default_value, float):
            var = ctk.DoubleVar(value=default_value)
        else:
            var = ctk.StringVar(value=str(default_value))
            entry = ctk.CTkEntry(param_row, textvariable=var, width=15)

        if isinstance(default_value, float) or isinstance(default_value, int):
            entry = ctk.CTkEntry(param_row, textvariable=var, width=15)
        entry.pack(side="left", padx=5)
        self.test_vars[param_name] = var

    def _create_control_section(self) -> None:
        """Create control buttons section."""
        control_frame = ctk.CTkFrame(self.scrollable_frame)
        control_frame.pack(fill="x", padx=5, pady=5)

        # Run button
        self.run_button = ctk.CTkButton(
            control_frame,
            text="Run Test",
            command=self._run_test,
            fg_color="#2E8B57",  # Sea green
        )
        self.run_button.pack(side="left", padx=5, pady=5)

        # Stop button
        self.stop_button = ctk.CTkButton(
            control_frame,
            text="Stop Test",
            command=self._stop_test,
            fg_color="#DC143C",  # Crimson
        )
        self.stop_button.pack(side="left", padx=5, pady=5)
        self.stop_button.configure(state="disabled")

        # Reset button
        self.reset_button = ctk.CTkButton(
            control_frame,
            text="Reset",
            command=self._reset_test,
            fg_color="#4682B4",  # Steel blue
        )
        self.reset_button.pack(side="left", padx=5, pady=5)

        # Export results button
        self.export_button = ctk.CTkButton(
            control_frame,
            text="Export Results",
            command=self._export_results,
            fg_color="#8B4513",  # Saddle brown
        )
        self.export_button.pack(side="left", padx=5, pady=5)

    def _create_progress_section(self) -> None:
        """Create progress tracking section."""
        progress_frame = ctk.CTkFrame(self.scrollable_frame)
        progress_frame.pack(fill="x", padx=5, pady=5)

        progress_title = ctk.CTkLabel(
            progress_frame,
            text="Test Progress",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        progress_title.pack(padx=10, pady=(10, 5))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ctk.CTkProgressBar(
            progress_frame, variable=self.progress_var, progress_color="#2E8B57"
        )
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0)

        # Progress label
        self.progress_label = ctk.CTkLabel(progress_frame, text="Ready to run test")
        self.progress_label.pack(padx=10, pady=(0, 5))

    def _create_results_section(self) -> None:
        """Create results display section."""
        results_frame = ctk.CTkFrame(self.scrollable_frame)
        results_frame.pack(fill="both", expand=True, padx=5, pady=5)

        results_title = ctk.CTkLabel(
            results_frame, text="Test Results", font=ctk.CTkFont(size=14, weight="bold")
        )
        results_title.pack(padx=10, pady=(10, 5))

        # Results text widget with scrollbar
        self.results_text = tk.Text(results_frame, wrap="word", height=10)
        scrollbar = ctk.CTkScrollbar(results_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=5)
        scrollbar.pack(side="right", fill="y", padx=(0, 5), pady=5)

        # Configure text tags for formatting
        self.results_text.tag_configure("header", font=("Arial", 12, "bold"))
        self.results_text.tag_configure("success", foreground="#2E8B57")
        self.results_text.tag_configure("error", foreground="#DC143C")
        self.results_text.tag_configure("warning", foreground="#FF8C00")
        self.results_text.tag_configure("info", foreground="#4682B4")

    def _create_tooltip(self, widget: Any, text: str) -> None:
        """Create a tooltip for a widget."""

        def on_enter(event: Any) -> None:
            tooltip = ctk.CTkToplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+ {event.x_root + 10}+ {event.y_root + 10}")

            label = ctk.CTkLabel(tooltip, text=text, font=ctk.CTkFont(size=10), wraplength=300)
            label.pack(padx=5, pady=3)

            widget.tooltip = tooltip

        def on_leave(event: Any) -> None:
            if hasattr(widget, "tooltip"):
                widget.tooltip.destroy()
                del widget.tooltip

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _setup_test_parameters(self) -> None:
        """Setup test-specific parameters and validation."""
        # Test-specific parameter validation
        if self.test_name == "Clinical Biomarkers":
            # Add dropdown for clinical conditions
            conditions = [
                "depression",
                "anxiety",
                "schizophrenia",
                "bipolar",
                "adhd",
                "autism",
            ]

            # Create dropdown for disorder selection
            disorder_frame = ctk.CTkFrame(self.params_frame)
            disorder_frame.pack(fill="x", pady=5)

            ctk.CTkLabel(disorder_frame, text="Disorder:").pack(anchor="w")
            disorder_var = tk.StringVar(value=conditions[0])
            disorder_menu = ctk.CTkOptionMenu(
                disorder_frame, variable=disorder_var, values=conditions
            )
            disorder_menu.pack(fill="x", pady=2)

            # Store reference for later use
            self.test_vars["disorder"] = disorder_var

        elif self.test_name == "Threshold Detection":
            # Add dropdown for modalities
            modalities = ["visual", "auditory", "interoceptive", "somatosensory"]

            # Create dropdown for modality selection
            modality_frame = ctk.CTkFrame(self.params_frame)
            modality_frame.pack(fill="x", pady=5)

            ctk.CTkLabel(modality_frame, text="Modality:").pack(anchor="w")
            modality_var = tk.StringVar(value=modalities[0])
            modality_menu = ctk.CTkOptionMenu(
                modality_frame, variable=modality_var, values=modalities
            )
            modality_menu.pack(fill="x", pady=2)

            # Store reference for later use
            self.test_vars["modality"] = modality_var

    def _run_test(self) -> None:
        """Run the falsification test in a separate thread."""
        if self.is_running:
            return

        # Validate test parameters
        if not self._validate_parameters():
            return

        self.is_running = True
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.reset_button.configure(state="disabled")
        self.export_button.configure(state="disabled")

        # Clear previous results
        self.results_text.delete("1.0", "end")
        self.progress_var.set(0)
        self.progress_label.configure(text="Starting test...")

        # Notify test started
        if self.on_test_started:
            self.on_test_started(self.test_name)

        # Start test in separate thread
        self.current_thread = threading.Thread(target=self._test_worker, daemon=True)
        self.current_thread.start()

    def _stop_test(self) -> None:
        """Stop the running test."""
        self.is_running = False

        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.reset_button.configure(state="normal")

        self.progress_label.configure(text="Test stopped by user")
        self.log_callback("Test stopped by user")

        # Wait for thread to finish
        if self.current_thread and self.current_thread.is_alive():
            self.current_thread.join(timeout=2.0)

    def _reset_test(self) -> None:
        """Reset the test panel."""
        self.is_running = False
        self.test_results = None

        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.reset_button.configure(state="normal")
        self.export_button.configure(state="disabled")

        self.progress_var.set(0)
        self.progress_label.configure(text="Ready to run test")
        self.results_text.delete("1.0", "end")

        # Reset parameters to defaults
        self._reset_parameters()

        self.log_callback("Test panel reset")

    def _validate_parameters(self) -> bool:
        """Validate test parameters before running."""
        try:
            # Get parameter values
            n_trials = self.test_vars.get("n_trials", tk.IntVar(value=1000)).get()

            # Basic validation
            if n_trials < 1 or n_trials > 100000:
                messagebox.showerror(
                    "Invalid Parameters",
                    "Number of trials must be between 1 and 100000",
                )
                return False

            # Validate participants if applicable
            if "n_participants" in self.test_vars:
                n_participants = self.test_vars["n_participants"].get()
                if n_participants < 1 or n_participants > 10000:
                    messagebox.showerror(
                        "Invalid Parameters",
                        "Number of participants must be between 1 and 10000",
                    )
                    return False

            # Advanced validation if validator is available
            try:
                validator = get_validator()

                # Build validation parameters
                validation_params = {"n_trials": n_trials}
                if "n_participants" in self.test_vars:
                    validation_params["n_participants"] = self.test_vars["n_participants"].get()

                result = validator.validate_experimental_config(**validation_params)

                if not result.is_valid:
                    messagebox.showerror(
                        "Invalid Parameters",
                        f"Test parameters are invalid:\n\n{result.get_message()}",
                    )
                    return False

                # Show warnings if any
                if result.warnings:
                    response = messagebox.askyesno(
                        "Parameter Warnings",
                        f"Test parameters have warnings:\n\n{result.get_message()}\n\nRun test anyway?",
                    )
                    if not response:
                        return False

            except Exception as e:
                self.log_callback(f"Warning: Advanced validation failed: {e}")

            return True

        except Exception as e:
            messagebox.showerror("Validation Error", f"Parameter validation failed: {e}")
            return False

    def _test_worker(self) -> None:
        """Worker thread for running tests."""
        try:
            self.log_callback(f"Starting {self.test_name} falsification test...")

            # Get test parameters
            test_params = self._get_test_parameters()

            # Calculate total operations for progress tracking
            total_operations = self._calculate_total_operations(test_params)
            completed_operations = 0

            # Update progress
            self._update_progress(0, total_operations, "Initializing test...")

            # Run the test
            self.test_results = self.controller.run_test(**test_params)  # type: ignore

            # Simulate progress updates (in real implementation, this would come from the test)
            for i in range(10):
                if not self.is_running:
                    break

                completed_operations = (i + 1) * (total_operations // 10)
                self._update_progress(
                    completed_operations,
                    total_operations,
                    f"Running test... {i + 1}/10",
                )

                # Simulate work
                import time

                time.sleep(0.5)

            if self.is_running:
                self._update_progress(total_operations, total_operations, "Test completed!")
                self._display_results()

                # Notify test completed
                if self.on_test_completed:
                    self.on_test_completed(self.test_name, self.test_results)

                if self.results_callback:
                    self.results_callback(self.test_name, self.test_results)

                self.log_callback(f"{self.test_name} test completed successfully")
            else:
                self.log_callback(f"{self.test_name} test was stopped")

        except Exception as e:
            self.log_callback(f"Error in {self.test_name} test: {e}")
            self._display_error(str(e))

            # Notify test failed
            if self.on_test_failed:
                self.on_test_failed(self.test_name, str(e))

            messagebox.showerror("Test Error", f"Test execution failed: {e}")

        finally:
            # Reset UI state
            self.run_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.reset_button.configure(state="normal")
            self.export_button.configure(state="normal" if self.test_results else "disabled")
            self.is_running = False

    def _get_test_parameters(self) -> Dict[str, Any]:
        """Get current test parameters."""
        params = {}

        for param_name, var in self.test_vars.items():
            params[param_name] = var.get()

        # Add test name
        params["test_name"] = self.test_name

        return params

    def _calculate_total_operations(self, test_params: Dict[str, Any]) -> int:
        """Calculate total operations for progress tracking."""
        n_trials = int(test_params.get("n_trials", 1000))
        n_participants = int(test_params.get("n_participants", 20))

        if self.test_name == "Threshold Insensitivity":
            n_drug_conditions = int(test_params.get("n_drug_conditions", 5))
            return int(n_trials * n_participants * n_drug_conditions)
        elif self.test_name == "Cross-Species Validation":
            n_species = int(test_params.get("n_species", 3))
            return int(n_trials * n_participants * n_species)
        else:
            return int(n_trials * n_participants)

    def _update_progress(self, completed: int, total: int, message: str) -> None:
        """Update progress bar and label."""
        if total > 0:
            progress = completed / total
            self.progress_var.set(progress)

            percentage = progress * 100
            self.progress_label.configure(text=f"{message} ({percentage:.1f}%)")

            # Notify progress update
            if self.on_progress_updated:
                self.on_progress_updated(self.test_name, progress, message)

            if self.progress_callback:
                self.progress_callback(progress, message)

    def _display_results(self) -> None:
        """Display test results in the results text widget."""
        if not self.test_results:
            return

        self.results_text.insert("end", f"=== {self.test_name} Test Results ===\n\n", "header")

        # Display results based on test type
        if hasattr(self.test_results, "to_dict"):
            results_dict = self.test_results.to_dict()
        else:
            results_dict = self.test_results

        for key, value in results_dict.items():
            if isinstance(value, float):
                self.results_text.insert(
                    "end", f"{key.replace('_', ' ').title()}: {value:.3f}\n", "info"
                )
            else:
                self.results_text.insert(
                    "end", f"{key.replace('_', ' ').title()}: {value}\n", "info"
                )

        # Add summary
        self.results_text.insert("end", "\n=== Summary ===\n", "header")

        if "success_rate" in results_dict:
            success_rate = results_dict["success_rate"]
            if success_rate > 0.7:
                self.results_text.insert(
                    "end",
                    f"Test Result: SUCCESS ({success_rate:.1%} success rate)\n",
                    "success",
                )
            elif success_rate > 0.4:
                self.results_text.insert(
                    "end",
                    f"Test Result: PARTIAL ({success_rate:.1%} success rate)\n",
                    "warning",
                )
            else:
                self.results_text.insert(
                    "end",
                    f"Test Result: FAILED ({success_rate:.1%} success rate)\n",
                    "error",
                )

        if "falsification_rate" in results_dict:
            falsification_rate = results_dict["falsification_rate"]
            self.results_text.insert(
                "end", f"Falsification Rate: {falsification_rate:.1%}\n", "info"
            )

        self.results_text.insert(
            "end",
            f"Execution Time: {results_dict.get('execution_time', 'N/A')} seconds\n",
            "info",
        )
        self.results_text.insert(
            "end",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "info",
        )

    def _display_error(self, error_message: str) -> None:
        """Display error message in results."""
        self.results_text.insert("end", f"=== {self.test_name} Test Error ===\n\n", "header")
        self.results_text.insert("end", f"Error: {error_message}\n", "error")
        self.results_text.insert(
            "end",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "info",
        )

    def _export_results(self) -> None:
        """Export test results to file."""
        if not self.test_results:
            messagebox.showwarning("No Results", "No test results to export")
            return

        try:
            from tkinter import filedialog

            file_path = filedialog.asksaveasfilename(
                title="Export Test Results",
                defaultextension=".json",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*"),
                ],
            )

            if file_path:
                if file_path.endswith(".json"):
                    self._export_json(file_path)
                else:
                    self._export_text(file_path)

                messagebox.showinfo("Success", f"Results exported to {file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {e}")

    def _export_json(self, file_path: str) -> None:
        """Export results as JSON."""
        import json

        # Build results data
        results_data: Any = self.test_results
        has_to_dict = self.test_results is not None
        if has_to_dict:
            has_to_dict = hasattr(self.test_results, "to_dict")
        if has_to_dict:
            results_data = self.test_results.to_dict()  # type: ignore[attr-defined]

        export_data = {
            "test_name": self.test_name,
            "timestamp": datetime.now().isoformat(),
            "parameters": self._get_test_parameters(),
            "results": results_data,
        }

        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2)

    def _export_text(self, file_path: str) -> None:
        """Export results as text."""
        with open(file_path, "w") as f:
            f.write(f"=== {self.test_name} Test Results ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Parameters:\n")
            for param_name, var in self.test_vars.items():
                f.write(f"  {param_name}: {var.get()}\n")

            f.write("\nResults:\n")
            f.write(self.results_text.get("1.0", "end"))

    def _reset_parameters(self) -> None:
        """Reset parameters to default values."""
        # Reset to default values
        defaults = {
            "n_trials": 1000,
            "n_participants": 100,
            "n_drug_conditions": 5,
            "n_species": 3,
            "clinical_condition": "depression",
            "modality": "visual",
            "method": "adaptive_staircase",
        }

        for param_name, default_value in defaults.items():
            if param_name in self.test_vars:
                self.test_vars[param_name].set(default_value)

    def _default_log_callback(self, message: str) -> None:
        """Default log callback if none provided."""
        logger.info(f"[{self.test_name}] {message}")

    def set_test_started_callback(self, callback: Callable) -> None:
        """Set callback for test started events."""
        self.on_test_started = callback

    def set_test_completed_callback(self, callback: Callable) -> None:
        """Set callback for test completed events."""
        self.on_test_completed = callback

    def set_test_failed_callback(self, callback: Callable) -> None:
        """Set callback for test failed events."""
        self.on_test_failed = callback

    def set_progress_updated_callback(self, callback: Callable) -> None:
        """Set callback for progress update events."""
        self.on_progress_updated = callback

    def get_test_results(self) -> Optional[Any]:
        """Get current test results."""
        return self.test_results

    def is_test_running(self) -> bool:
        """Check if test is currently running."""
        return self.is_running

    def get_test_name(self) -> str:
        """Get the test name."""
        return self.test_name


# Factory function for easy instantiation
def create_test_execution_panel(
    parent: Any,
    test_name: str,
    controller: Optional[MainApplicationController] = None,
    progress_callback: Optional[Callable] = None,
    log_callback: Optional[Callable] = None,
    results_callback: Optional[Callable] = None,
) -> ExecutionPanel:
    """Create a test execution panel with default settings.

    Args:
        parent: Parent widget
        test_name: Name of the test
        controller: Optional main application controller
        progress_callback: Optional progress callback
        log_callback: Optional log callback
        results_callback: Optional results callback

    Returns:
        Configured ExecutionPanel instance
    """
    return ExecutionPanel(
        parent, test_name, controller, progress_callback, log_callback, results_callback
    )


# Alias for backward compatibility
TestExecutionPanel = ExecutionPanel
