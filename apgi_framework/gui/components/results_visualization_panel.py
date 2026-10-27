"""Results Visualization Panel for APGI Framework GUI.

Extracted from the monolithic GUI to provide a focused component
for visualizing test results and statistical analysis.
"""

import json
import logging
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Callable, Dict, List, Optional

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTk
from matplotlib.figure import Figure

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from apgi_framework.logging.standardized_logging import get_logger

    logger = get_logger("results_visualization_panel")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("results_visualization_panel")  # type: ignore[assignment]
    logger.warning(f"Could not import standardized logging: {e}")


class ResultsVisualizationPanel(ctk.CTkFrame):
    """Panel for visualizing test results and statistical analysis.

    Provides comprehensive visualization capabilities including real-time plotting,
    statistical summaries, comparison views, and export functionality.
    """

    def __init__(self, parent: Any, results_callback: Optional[Callable] = None):
        """Initialize the results visualization panel.

        Args:
            parent: Parent widget
            results_callback: Callback for getting results data
        """
        super().__init__(parent)

        self.results_data: List[Dict[str, Any]] = []
        self.test_run_history: List[Dict[str, Any]] = []  # Track multiple runs for comparison
        self.results_callback = results_callback

        # Visualization settings
        self.current_view_mode = "Overview"
        self.plot_colors = {
            "primary": "#2E8B57",  # Sea green
            "secondary": "#4682B4",  # Steel blue
            "accent": "#DC143C",  # Crimson
            "warning": "#FF8C00",  # Dark orange
            "neutral": "#708090",  # Slate gray
        }

        # Callbacks for external components
        self.on_data_updated: Optional[Callable] = None
        self.on_export_requested: Optional[Callable] = None

        self._create_widgets()  # type: ignore[no-untyped-call]
        self._setup_matplotlib()  # type: ignore[no-untyped-call]

        logger.info("ResultsVisualizationPanel initialized")

    def _create_widgets(self) -> None:
        """Create visualization widgets."""
        # Create scrollable frame
        self.scrollable_frame = ctk.CTkScrollableFrame(self)
        self.scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Control section
        self._create_control_section()

        # Summary statistics section
        self._create_summary_section()

        # Visualization section
        self._create_visualization_section()

    def _create_control_section(self) -> None:
        """Create control buttons and view selector."""
        control_frame = ctk.CTkFrame(self.scrollable_frame)
        control_frame.pack(fill="x", padx=5, pady=5)

        # Control buttons
        self.clear_btn = ctk.CTkButton(
            control_frame,
            text="Clear Data",
            command=self._clear_data,
            fg_color=self.plot_colors["accent"],
        )
        self.clear_btn.pack(side="left", padx=2, pady=2)

        self.export_btn = ctk.CTkButton(
            control_frame,
            text="Export Results",
            command=self._export_results,
            fg_color=self.plot_colors["neutral"],
        )
        self.export_btn.pack(side="left", padx=2, pady=2)

        self.compare_btn = ctk.CTkButton(
            control_frame,
            text="Compare Runs",
            command=self._show_comparison_view,
            fg_color=self.plot_colors["secondary"],
        )
        self.compare_btn.pack(side="left", padx=2, pady=2)

        self.refresh_btn = ctk.CTkButton(
            control_frame,
            text="Refresh Plots",
            command=self._update_plots,
            fg_color=self.plot_colors["primary"],
        )
        self.refresh_btn.pack(side="left", padx=2, pady=2)

        # View mode selector
        ctk.CTkLabel(control_frame, text="View:").pack(side="right", padx=(10, 2))

        self.view_mode_var = tk.StringVar(value="Overview")
        self.view_mode_combo = ctk.CTkOptionMenu(
            control_frame,
            variable=self.view_mode_var,
            values=[
                "Overview",
                "Statistical Details",
                "Time Series",
                "Comparison",
                "Neural Signatures",
            ],
            width=15,
        )
        self.view_mode_combo.pack(side="right", padx=2)
        self.view_mode_combo.configure(command=lambda choice: self._change_view_mode(choice))

    def _create_summary_section(self) -> None:
        """Create summary statistics section."""
        summary_frame = ctk.CTkFrame(self.scrollable_frame)
        summary_frame.pack(fill="x", padx=5, pady=5)

        summary_title = ctk.CTkLabel(
            summary_frame,
            text="Statistical Summary",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        summary_title.pack(padx=10, pady=(10, 5))

        # Summary text with scrollbar
        summary_container = ctk.CTkFrame(summary_frame)
        summary_container.pack(fill="x", padx=5, pady=5)

        self.summary_text = tk.Text(summary_container, height=6, wrap="word", font=("Courier", 9))
        summary_scrollbar = ctk.CTkScrollbar(summary_container, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)

        self.summary_text.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=5)
        summary_scrollbar.pack(side="right", fill="y", padx=(0, 5), pady=5)

        # Configure text tags for formatting
        self.summary_text.tag_configure("header", font=("Courier", 10, "bold"))
        self.summary_text.tag_configure("success", foreground=self.plot_colors["primary"])
        self.summary_text.tag_configure("warning", foreground=self.plot_colors["warning"])
        self.summary_text.tag_configure("error", foreground=self.plot_colors["accent"])
        self.summary_text.tag_configure("info", foreground=self.plot_colors["secondary"])

        self._update_summary_text()

    def _create_visualization_section(self) -> None:
        """Create matplotlib visualization section."""
        viz_frame = ctk.CTkFrame(self.scrollable_frame)
        viz_frame.pack(fill="both", expand=True, padx=5, pady=5)

        viz_title = ctk.CTkLabel(
            viz_frame,
            text="Data Visualization",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        viz_title.pack(padx=10, pady=(10, 5))

        # Create matplotlib figure container
        self.fig_container = ctk.CTkFrame(viz_frame)
        self.fig_container.pack(fill="both", expand=True, padx=5, pady=5)

        # Initialize plots
        self._setup_matplotlib()

    def _setup_matplotlib(self) -> None:
        """Setup matplotlib figure and axes."""
        # Create figure with better layout
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.patch.set_facecolor("#f0f0f0")

        # Set main title
        self.fig.suptitle(
            "APGI Falsification Test Results - Real-Time Analysis",
            fontsize=14,
            fontweight="bold",
        )

        # Create subplot grid
        gs = self.fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Falsification rates
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # Effect sizes
        self.ax3 = self.fig.add_subplot(gs[1, 0])  # P-values
        self.ax4 = self.fig.add_subplot(gs[1, 1])  # Statistical power

        # Style the axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("#ffffff")

        # Create canvas
        if hasattr(self, "canvas"):
            self.canvas.get_tk_widget().destroy()  # type: ignore[attr-defined, has-type]

        self.canvas = FigureCanvasTk(self.fig, self.fig_container)  # type: ignore[no-untyped-call]
        self.canvas.get_tk_widget().pack(fill="both", expand=True)  # type: ignore[no-untyped-call]

        # Initialize empty plots
        self._update_plots()

    def _change_view_mode(self, view_mode: str) -> None:
        """Change visualization view mode."""
        self.current_view_mode = view_mode
        self._update_plots()
        logger.info(f"Changed view mode to: {view_mode}")

    def _show_comparison_view(self) -> None:
        """Show comparison view for multiple test runs."""
        if len(self.test_run_history) < 2:
            messagebox.showinfo(
                "Info",
                "Need at least 2 test runs to compare. Run more tests to enable comparison.",
            )
            return

        # Switch to comparison view
        self.view_mode_var.set("Comparison")
        self._change_view_mode("Comparison")

    def _clear_data(self) -> None:
        """Clear all results data."""
        if messagebox.askyesno(
            "Clear Data", "Clear all visualization data? This cannot be undone."
        ):
            self.results_data = []
            self.test_run_history = []
            self._update_plots()
            self._update_summary_text()

            # Notify data cleared
            if self.on_data_updated:
                self.on_data_updated([])

            logger.info("Visualization data cleared")

    def _update_plots(self) -> None:
        """Update all visualization plots based on current view mode."""
        # Clear all axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("#ffffff")

        if not self.results_data:
            self._show_empty_plots()
        else:
            if self.current_view_mode == "Overview":
                self._show_overview_plots()
            elif self.current_view_mode == "Statistical Details":
                self._show_statistical_details()
            elif self.current_view_mode == "Time Series":
                self._show_time_series_plots()
            elif self.current_view_mode == "Comparison":
                self._show_comparison_plots()
            elif self.current_view_mode == "Neural Signatures":
                self._show_neural_signature_plots()

        # Refresh canvas
        self.canvas.draw()

    def _show_empty_plots(self) -> None:
        """Show empty plots with instructions."""
        empty_message = "No data available\nRun tests to see results"

        for i, ax in enumerate([self.ax1, self.ax2, self.ax3, self.ax4]):
            ax.text(
                0.5,
                0.5,
                empty_message,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color=self.plot_colors["neutral"],
            )
            ax.set_xticks([])
            ax.set_yticks([])

            # Set appropriate titles
            titles = [
                "Falsification Rates",
                "Effect Sizes",
                "P-values",
                "Statistical Power",
            ]
            ax.set_title(titles[i], fontweight="bold")

    def _show_overview_plots(self) -> None:
        """Show overview visualization plots."""
        # Extract data
        test_types = [r.get("test_type", "Unknown") for r in self.results_data]
        falsified = [r.get("is_falsified", False) for r in self.results_data]
        effect_sizes = [r.get("effect_size", 0) for r in self.results_data]
        p_values = [r.get("p_value", 1.0) for r in self.results_data]
        powers = [r.get("statistical_power", 0) for r in self.results_data]

        # Plot 1: Falsification rates by test type
        self._plot_falsification_rates(test_types, falsified)

        # Plot 2: Effect sizes distribution
        self._plot_effect_sizes(effect_sizes)

        # Plot 3: P-values distribution
        self._plot_p_values(p_values)

        # Plot 4: Statistical power
        self._plot_statistical_power(powers)

    def _plot_falsification_rates(self, test_types: List[str], falsified: List[bool]) -> None:
        """Plot falsification rates by test type."""
        unique_tests = sorted(list(set(test_types)))
        falsification_rates = []
        test_counts = []

        for test in unique_tests:
            test_results = [f for t, f in zip(test_types, falsified) if t == test]
            rate = sum(test_results) / len(test_results) if test_results else 0
            falsification_rates.append(rate)
            test_counts.append(len(test_results))

        colors = [
            self.plot_colors["accent"] if r > 0.5 else self.plot_colors["primary"]
            for r in falsification_rates
        ]

        bars = self.ax1.bar(range(len(unique_tests)), falsification_rates, color=colors, alpha=0.7)

        self.ax1.set_xticks(range(len(unique_tests)))
        self.ax1.set_xticklabels(
            [t.replace("_", "\n") for t in unique_tests],
            rotation=0,
            ha="center",
            fontsize=8,
        )
        self.ax1.set_ylabel("Falsification Rate")
        self.ax1.set_title("Falsification Rates by Test Type", fontweight="bold")
        self.ax1.set_ylim(0, 1.0)
        self.ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, test_counts)):
            height = bar.get_height()
            self.ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    def _plot_effect_sizes(self, effect_sizes: List[float]) -> None:
        """Plot effect sizes distribution."""
        if effect_sizes:
            self.ax2.hist(
                effect_sizes,
                bins=min(10, len(effect_sizes)),
                alpha=0.7,
                color=self.plot_colors["secondary"],
                edgecolor="black",
            )

            # Reference lines
            self.ax2.axvline(
                x=0.5,
                color=self.plot_colors["warning"],
                linestyle="--",
                label="Medium effect (0.5)",
                linewidth=2,
            )
            self.ax2.axvline(
                x=0.8,
                color=self.plot_colors["accent"],
                linestyle="--",
                label="Large effect (0.8)",
                linewidth=2,
            )

            self.ax2.set_xlabel("Effect Size")
            self.ax2.set_ylabel("Frequency")
            self.ax2.legend(fontsize=8)

        self.ax2.set_title("Effect Sizes Distribution", fontweight="bold")

    def _plot_p_values(self, p_values: List[float]) -> None:
        """Plot p-values distribution."""
        if p_values:
            # Create histogram with significance threshold
            self.ax3.hist(
                p_values,
                bins=min(10, len(p_values)),
                alpha=0.7,
                color=self.plot_colors["neutral"],
                edgecolor="black",
            )

            # Significance threshold
            self.ax3.axvline(
                x=0.05,
                color=self.plot_colors["accent"],
                linestyle="--",
                label="α = 0.05",
                linewidth=2,
            )

            # Count significant results
            significant = sum(1 for p in p_values if p < 0.05)
            total = len(p_values)
            sig_rate = significant / total if total > 0 else 0

            self.ax3.text(
                0.95,
                0.95,
                f"Significant: {significant}/{total} ({sig_rate:.1%})",
                transform=self.ax3.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            self.ax3.set_xlabel("P-value")
            self.ax3.legend(fontsize=8)

        self.ax3.set_title("P-values Distribution", fontweight="bold")

    def _plot_statistical_power(self, powers: List[float]) -> None:
        """Plot statistical power."""
        if powers:
            # Create histogram
            self.ax4.hist(
                powers,
                bins=min(10, len(powers)),
                alpha=0.7,
                color=self.plot_colors["primary"],
                edgecolor="black",
            )

            # Adequate power threshold
            self.ax4.axvline(
                x=0.8,
                color=self.plot_colors["warning"],
                linestyle="--",
                label="Adequate power (0.8)",
                linewidth=2,
            )

            # Count adequate power
            adequate = sum(1 for p in powers if p >= 0.8)
            total = len(powers)
            adequate_rate = adequate / total if total > 0 else 0

            self.ax4.text(
                0.95,
                0.95,
                f"Adequate: {adequate}/{total} ({adequate_rate:.1%})",
                transform=self.ax4.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            self.ax4.set_xlabel("Statistical Power")
            self.ax4.legend(fontsize=8)

        self.ax4.set_title("Statistical Power Distribution", fontweight="bold")

    def _show_statistical_details(self) -> None:
        """Show detailed statistical analysis."""
        # Extract detailed statistics
        effect_sizes = [r.get("effect_size", 0) for r in self.results_data]
        p_values = [r.get("p_value", 1.0) for r in self.results_data]
        confidence_intervals = [r.get("confidence_interval", [0, 1]) for r in self.results_data]

        # Plot 1: Effect sizes with confidence intervals
        self._plot_effect_sizes_with_ci(effect_sizes, confidence_intervals)

        # Plot 2: P-value vs Effect size scatter
        self._plot_p_vs_effect_size(p_values, effect_sizes)

        # Plot 3: Confidence interval widths
        self._plot_ci_widths(confidence_intervals)

        # Plot 4: Statistical power analysis
        self._plot_power_analysis()

    def _show_time_series_plots(self) -> None:
        """Show time series analysis of test runs."""
        if not self.test_run_history:
            self._show_empty_plots()
            return

        # Extract time series data
        timestamps = [run.get("timestamp", datetime.now()) for run in self.test_run_history]
        success_rates = [run.get("success_rate", 0) for run in self.test_run_history]
        falsification_rates = [run.get("falsification_rate", 0) for run in self.test_run_history]

        # Plot time series
        self._plot_success_rate_timeline(timestamps, success_rates)
        self._plot_falsification_timeline(timestamps, falsification_rates)
        self._plot_trend_analysis(timestamps, success_rates, falsification_rates)
        self._plot_run_comparison()

    def _show_comparison_plots(self) -> None:
        """Show comparison plots for multiple runs."""
        if len(self.test_run_history) < 2:
            self._show_empty_plots()
            return

        # Extract comparison data
        run_names = [f"Run {i + 1}" for i in range(len(self.test_run_history))]
        metrics = [
            "success_rate",
            "falsification_rate",
            "mean_effect_size",
            "mean_power",
        ]

        # Create comparison plots
        self._plot_run_comparison_bar(run_names, metrics)
        self._plot_run_radar_comparison(run_names)
        self._plot_performance_trends()
        self._plot_statistical_significance_comparison()

    def _show_neural_signature_plots(self) -> None:
        """Show neural signature specific visualizations."""
        # Extract neural data
        p3b_amplitudes = [r.get("p3b_amplitude", []) for r in self.results_data]
        gamma_powers = [r.get("gamma_power", []) for r in self.results_data]

        # Plot neural signatures
        self._plot_p3b_analysis(p3b_amplitudes)
        self._plot_gamma_analysis(gamma_powers)
        self._plot_neural_correlations()
        self._plot_signature_distributions()

    def _update_summary_text(self) -> None:
        """Update summary statistics text."""
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")

        if not self.results_data:
            self.summary_text.insert(
                "1.0", "No test results yet. Run tests to see summary statistics.\n"
            )
        else:
            # Calculate summary statistics
            total_tests = len(self.results_data)
            falsified_tests = sum(1 for r in self.results_data if r.get("is_falsified", False))
            falsification_rate = falsified_tests / total_tests if total_tests > 0 else 0

            effect_sizes = [r.get("effect_size", 0) for r in self.results_data]
            mean_effect_size = np.mean(effect_sizes) if effect_sizes else 0

            p_values = [r.get("p_value", 1.0) for r in self.results_data]
            significant_tests = sum(1 for p in p_values if p < 0.05)

            powers = [r.get("statistical_power", 0) for r in self.results_data]
            mean_power = np.mean(powers) if powers else 0

            # Build summary text
            summary = f"{'=' * 50}\n"
            summary += "SUMMARY STATISTICS\n"
            summary += f"{'=' * 50}\n\n"
            summary += f"Total Tests:        {total_tests}\n"
            summary += f"Falsified Tests:   {falsified_tests} ({falsification_rate:.1%})\n"
            summary += f"Mean Effect Size:  {mean_effect_size:.3f}\n"
            summary += (
                f"Significant Tests: {significant_tests} ({significant_tests / total_tests:.1%})\n"
            )
            summary += f"Mean Power:        {mean_power:.3f}\n\n"

            # Test type breakdown
            test_types = {}
            for result in self.results_data:
                test_type = result.get("test_type", "Unknown")
                if test_type not in test_types:
                    test_types[test_type] = {"total": 0, "falsified": 0}
                test_types[test_type]["total"] += 1
                if result.get("is_falsified", False):
                    test_types[test_type]["falsified"] += 1

            summary += "TEST TYPE BREAKDOWN\n"
            summary += f"{'-' * 30}\n"
            for test_type, stats in test_types.items():
                rate = stats["falsified"] / stats["total"] if stats["total"] > 0 else 0
                summary += (
                    f"{test_type:20s}: {stats['falsified']:2d}/{stats['total']:2d} ({rate:.1%})\n"
                )

            # Insert summary with formatting
            self.summary_text.insert("1.0", summary)

            # Apply formatting tags
            self.summary_text.tag_add("header", "1.0", "3.0")
            self.summary_text.tag_add("info", "4.0", "10.0")
            self.summary_text.tag_add("header", "11.0", "13.0")

            # Color code falsification rate
            if falsification_rate > 0.5:
                self.summary_text.tag_add("warning", "6.0", "6.0")
            else:
                self.summary_text.tag_add("success", "6.0", "6.0")

        self.summary_text.configure(state="disabled")

    def _export_results(self) -> None:
        """Export visualization results to file."""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Export Visualization Results",
                defaultextension=".json",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("PNG images", "*.png"),
                    ("CSV data", "*.csv"),
                    ("All files", "*.*"),
                ],
            )

            if file_path:
                if file_path.endswith(".json"):
                    self._export_json(file_path)
                elif file_path.endswith(".png"):
                    self._export_image(file_path)
                elif file_path.endswith(".csv"):
                    self._export_csv(file_path)
                else:
                    # Default to JSON
                    self._export_json(file_path + ".json")

                messagebox.showinfo("Success", f"Results exported to {file_path}")

                # Notify export requested
                if self.on_export_requested:
                    self.on_export_requested(file_path)

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {e}")
            logger.error(f"Failed to export results: {e}")

    def _export_json(self, file_path: str) -> None:
        """Export results as JSON."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "view_mode": self.current_view_mode,
            "summary_statistics": self._calculate_summary_stats(),
            "results_data": self.results_data,
            "test_run_history": self.test_run_history,
        }

        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

    def _export_image(self, file_path: str) -> None:
        """Export current visualization as image."""
        self.fig.savefig(file_path, dpi=300, bbox_inches="tight")

    def _export_csv(self, file_path: str) -> None:
        """Export results data as CSV."""
        import pandas as pd

        if self.results_data:
            df = pd.DataFrame(self.results_data)
            df.to_csv(file_path, index=False)

    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics for export."""
        if not self.results_data:
            return {}

        total_tests = len(self.results_data)
        falsified_tests = sum(1 for r in self.results_data if r.get("is_falsified", False))

        effect_sizes = [r.get("effect_size", 0) for r in self.results_data]
        p_values = [r.get("p_value", 1.0) for r in self.results_data]
        powers = [r.get("statistical_power", 0) for r in self.results_data]

        return {
            "total_tests": total_tests,
            "falsified_tests": falsified_tests,
            "falsification_rate": (falsified_tests / total_tests if total_tests > 0 else 0),
            "mean_effect_size": np.mean(effect_sizes) if effect_sizes else 0,
            "median_effect_size": np.median(effect_sizes) if effect_sizes else 0,
            "std_effect_size": np.std(effect_sizes) if effect_sizes else 0,
            "significant_tests": sum(1 for p in p_values if p < 0.05),
            "mean_p_value": np.mean(p_values) if p_values else 1.0,
            "mean_power": np.mean(powers) if powers else 0,
            "adequate_power_tests": sum(1 for p in powers if p >= 0.8),
        }

    def add_results(self, test_name: str, results: Dict[str, Any]) -> None:
        """Add new test results to visualization."""
        # Add timestamp
        results["timestamp"] = datetime.now()
        results["test_type"] = test_name

        self.results_data.append(results)

        # Add to run history
        run_summary = {
            "test_name": test_name,
            "timestamp": results["timestamp"],
            "success_rate": results.get("success_rate", 0),
            "falsification_rate": 1.0 if results.get("is_falsified", False) else 0.0,
            "mean_effect_size": results.get("effect_size", 0),
            "mean_power": results.get("statistical_power", 0),
        }
        self.test_run_history.append(run_summary)

        # Update visualizations
        self._update_plots()
        self._update_summary_text()

        # Notify data updated
        if self.on_data_updated:
            self.on_data_updated(self.results_data)

        logger.info(f"Added results for {test_name} test")

    def clear_results(self) -> None:
        """Clear all results data."""
        self.results_data = []
        self.test_run_history = []
        self._update_plots()
        self._update_summary_text()

    def get_results_data(self) -> List[Dict[str, Any]]:
        """Get current results data."""
        return self.results_data.copy()

    def set_data_updated_callback(self, callback: Callable[[Any], None]) -> None:
        """Set callback for data update events."""
        self.on_data_updated = callback

    def set_export_requested_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for export request events."""
        self.on_export_requested = callback

    def refresh_visualization(self) -> None:
        """Force refresh of all visualizations."""
        self._update_plots()
        self._update_summary_text()

    # Additional plotting methods for different view modes
    def _plot_effect_sizes_with_ci(
        self, effect_sizes: List[float], confidence_intervals: List[List[float]]
    ) -> None:
        """Plot effect sizes with confidence intervals."""
        if effect_sizes and confidence_intervals:
            x_pos = range(len(effect_sizes))
            ci_lower = [ci[0] for ci in confidence_intervals]
            ci_upper = [ci[1] for ci in confidence_intervals]

            self.ax1.errorbar(
                x_pos,
                effect_sizes,
                yerr=[
                    np.array(effect_sizes) - np.array(ci_lower),
                    np.array(ci_upper) - np.array(effect_sizes),
                ],
                fmt="o",
                capsize=5,
                capthick=2,
                color=self.plot_colors["secondary"],
            )
            self.ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
            self.ax1.set_xlabel("Test Index")
            self.ax1.set_ylabel("Effect Size")
            self.ax1.set_title("Effect Sizes with Confidence Intervals", fontweight="bold")

    def _plot_p_vs_effect_size(self, p_values: List[float], effect_sizes: List[float]) -> None:
        """Plot p-values vs effect sizes scatter plot."""
        if p_values and effect_sizes:
            colors = [
                self.plot_colors["accent"] if p < 0.05 else self.plot_colors["neutral"]
                for p in p_values
            ]

            self.ax2.scatter(effect_sizes, p_values, c=colors, alpha=0.7)
            self.ax2.axhline(
                y=0.05,
                color=self.plot_colors["accent"],
                linestyle="--",
                label="α = 0.05",
            )
            self.ax2.axvline(x=0, color="gray", linestyle="-", alpha=0.5)
            self.ax2.set_xlabel("Effect Size")
            self.ax2.set_ylabel("P-value")
            self.ax2.set_title("P-value vs Effect Size", fontweight="bold")
            self.ax2.legend()

    def _plot_ci_widths(self, confidence_intervals: List[List[float]]) -> None:
        """Plot confidence interval widths."""
        if confidence_intervals:
            ci_widths = [ci[1] - ci[0] for ci in confidence_intervals]
            self.ax3.hist(
                ci_widths,
                bins=min(10, len(ci_widths)),
                alpha=0.7,
                color=self.plot_colors["warning"],
                edgecolor="black",
            )
            self.ax3.set_xlabel("Confidence Interval Width")
            self.ax3.set_ylabel("Frequency")
            self.ax3.set_title("Confidence Interval Widths", fontweight="bold")

    def _plot_power_analysis(self) -> None:
        """Plot statistical power analysis."""
        # Create power curve
        effect_sizes = np.linspace(0.1, 2.0, 50)
        sample_sizes = [20, 50, 100, 200]

        for n in sample_sizes:
            # Simplified power calculation (would use proper statistical methods)
            power = 1 - (1 + (effect_sizes * np.sqrt(n / 2)) ** -1) ** -1
            self.ax4.plot(effect_sizes, power, label=f"n={n}")

        self.ax4.set_xlabel("Effect Size")
        self.ax4.set_ylabel("Statistical Power")
        self.ax4.set_title("Power Analysis", fontweight="bold")
        self.ax4.legend()
        self.ax4.grid(True, alpha=0.3)

    def _plot_3d_phase_portrait(
        self,
        x_data: List[float],
        y_data: List[float],
        z_data: List[float],
        title: str = "3D Phase Portrait",
    ) -> Optional[Figure]:
        """Plot 3D phase portrait for advanced visualization."""
        # Create 3D axes
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot the trajectory
        ax.plot(x_data, y_data, z_data, lw=2, alpha=0.8, color=self.plot_colors["primary"])

        # Add markers for start and end
        ax.scatter([x_data[0]], [y_data[0]], [z_data[0]], color="green", s=100, label="Start")
        ax.scatter([x_data[-1]], [y_data[-1]], [z_data[-1]], color="red", s=100, label="End")

        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title(title, fontweight="bold")
        ax.legend()

        return fig

    def _plot_cross_spectral_analysis(
        self, signal1: List[float], signal2: List[float], fs: int = 1000
    ) -> Optional[Figure]:
        """Plot cross-spectral analysis between two signals."""
        from scipy import signal

        # Compute cross-spectral density
        f, Pxy = signal.coherence(signal1, signal2, fs=fs)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot coherence
        ax1.plot(f, Pxy, color=self.plot_colors["primary"], lw=2)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Coherence")
        ax1.set_title("Cross-Spectral Coherence", fontweight="bold")
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)

        # Plot phase
        f, Cxy = signal.csd(signal1, signal2, fs=fs)
        phase = np.angle(Cxy)
        ax2.plot(f, phase, color=self.plot_colors["secondary"], lw=2)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (radians)")
        ax2.set_title("Cross-Spectral Phase", fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_success_rate_timeline(
        self, timestamps: List[datetime], success_rates: List[float]
    ) -> None:
        """Plot success rate over time."""
        if timestamps and success_rates:
            # Convert timestamps to numbers for plotting
            time_nums = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]  # hours

            self.ax1.plot(
                time_nums,
                success_rates,
                "o-",
                color=self.plot_colors["primary"],
                linewidth=2,
            )
            self.ax1.set_xlabel("Time (hours)")
            self.ax1.set_ylabel("Success Rate")
            self.ax1.set_title("Success Rate Timeline", fontweight="bold")
            self.ax1.grid(True, alpha=0.3)
            self.ax1.set_ylim(0, 1)

    def _plot_falsification_timeline(
        self, timestamps: List[datetime], falsification_rates: List[float]
    ) -> None:
        """Plot falsification rate over time."""
        if timestamps and falsification_rates:
            time_nums = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]

            self.ax2.plot(
                time_nums,
                falsification_rates,
                "o-",
                color=self.plot_colors["accent"],
                linewidth=2,
            )
            self.ax2.set_xlabel("Time (hours)")
            self.ax2.set_ylabel("Falsification Rate")
            self.ax2.set_title("Falsification Rate Timeline", fontweight="bold")
            self.ax2.grid(True, alpha=0.3)
            self.ax2.set_ylim(0, 1)

    def _plot_trend_analysis(
        self,
        timestamps: List[datetime],
        success_rates: List[float],
        falsification_rates: List[float],
    ) -> None:
        """Plot trend analysis of success and falsification rates."""
        if timestamps and success_rates and falsification_rates:
            time_nums = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]

            # Calculate trends (simple linear fit)
            if len(time_nums) > 1:
                success_trend = np.polyfit(time_nums, success_rates, 1)
                falsification_trend = np.polyfit(time_nums, falsification_rates, 1)

                success_line = np.polyval(success_trend, time_nums)
                falsification_line = np.polyval(falsification_trend, time_nums)

                self.ax3.plot(
                    time_nums,
                    success_rates,
                    "o",
                    color=self.plot_colors["primary"],
                    alpha=0.7,
                )
                self.ax3.plot(
                    time_nums,
                    success_line,
                    "-",
                    color=self.plot_colors["primary"],
                    linewidth=2,
                    label="Success trend",
                )
                self.ax3.plot(
                    time_nums,
                    falsification_rates,
                    "s",
                    color=self.plot_colors["accent"],
                    alpha=0.7,
                )
                self.ax3.plot(
                    time_nums,
                    falsification_line,
                    "-",
                    color=self.plot_colors["accent"],
                    linewidth=2,
                    label="Falsification trend",
                )

            self.ax3.set_xlabel("Time (hours)")
            self.ax3.set_ylabel("Rate")
            self.ax3.set_title("Trend Analysis", fontweight="bold")
            self.ax3.legend()
            self.ax3.grid(True, alpha=0.3)

    def _plot_run_comparison(self) -> None:
        """Plot comparison of individual runs."""
        if self.test_run_history:
            run_names = [f"Run {i + 1}" for i in range(len(self.test_run_history))]
            success_rates = [run.get("success_rate", 0) for run in self.test_run_history]
            falsification_rates = [
                run.get("falsification_rate", 0) for run in self.test_run_history
            ]

            x = range(len(run_names))
            width = 0.35

            self.ax4.bar(
                [i - width / 2 for i in x],
                success_rates,
                width,
                label="Success Rate",
                color=self.plot_colors["primary"],
                alpha=0.7,
            )
            self.ax4.bar(
                [i + width / 2 for i in x],
                falsification_rates,
                width,
                label="Falsification Rate",
                color=self.plot_colors["accent"],
                alpha=0.7,
            )

            self.ax4.set_xlabel("Run")
            self.ax4.set_ylabel("Rate")
            self.ax4.set_title("Run Comparison", fontweight="bold")
            self.ax4.set_xticks(x)
            self.ax4.set_xticklabels(run_names)
            self.ax4.legend()
            self.ax4.grid(True, alpha=0.3)
            self.ax4.set_ylim(0, 1)

    def _plot_run_comparison_bar(self, run_names: List[str], metrics: List[str]) -> None:
        """Plot bar comparison of runs across different metrics."""
        if run_names and metrics:
            # Extract metric values for each run
            metric_data = {}
            for metric in metrics:
                values = []
                for run in self.test_run_history:
                    if metric == "success_rate":
                        values.append(run.get("success_rate", 0))
                    elif metric == "falsification_rate":
                        values.append(run.get("falsification_rate", 0))
                    elif metric == "mean_effect_size":
                        values.append(run.get("mean_effect_size", 0))
                    elif metric == "mean_power":
                        values.append(run.get("mean_power", 0))
                metric_data[metric] = values

            x = range(len(run_names))
            width = 0.2
            colors = [
                self.plot_colors["primary"],
                self.plot_colors["secondary"],
                self.plot_colors["accent"],
                self.plot_colors["warning"],
            ]

            for i, metric in enumerate(metrics):
                if metric in metric_data:
                    offset = (i - 1.5) * width
                    self.ax1.bar(
                        [pos + offset for pos in x],
                        metric_data[metric],
                        width,
                        label=metric.replace("_", " ").title(),
                        color=colors[i % len(colors)],
                        alpha=0.7,
                    )

            self.ax1.set_xlabel("Run")
            self.ax1.set_ylabel("Value")
            self.ax1.set_title("Run Comparison by Metric", fontweight="bold")
            self.ax1.set_xticks(x)
            self.ax1.set_xticklabels(run_names)
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)

    def _plot_run_radar_comparison(self, run_names: List[str]) -> None:
        """Plot radar comparison of runs (simplified as bar plot)."""
        if run_names and len(run_names) >= 2:
            # Simplified: show average metrics for each run
            avg_metrics = []
            for run in self.test_run_history:
                avg = (
                    run.get("success_rate", 0)
                    + run.get("falsification_rate", 0)
                    + run.get("mean_effect_size", 0)
                    + run.get("mean_power", 0)
                ) / 4
                avg_metrics.append(avg)

            self.ax2.bar(run_names, avg_metrics, color=self.plot_colors["secondary"], alpha=0.7)
            self.ax2.set_ylabel("Average Metric Score")
            self.ax2.set_title("Run Performance Radar (Simplified)", fontweight="bold")
            self.ax2.grid(True, alpha=0.3)

    def _plot_performance_trends(self) -> None:
        """Plot performance trends across runs."""
        if self.test_run_history:
            run_indices = list(range(1, len(self.test_run_history) + 1))
            success_rates = [run.get("success_rate", 0) for run in self.test_run_history]
            effect_sizes = [run.get("mean_effect_size", 0) for run in self.test_run_history]
            powers = [run.get("mean_power", 0) for run in self.test_run_history]

            self.ax3.plot(
                run_indices,
                success_rates,
                "o-",
                label="Success Rate",
                color=self.plot_colors["primary"],
            )
            self.ax3.plot(
                run_indices,
                effect_sizes,
                "s-",
                label="Effect Size",
                color=self.plot_colors["secondary"],
            )
            self.ax3.plot(
                run_indices,
                powers,
                "^-",
                label="Power",
                color=self.plot_colors["accent"],
            )

            self.ax3.set_xlabel("Run Number")
            self.ax3.set_ylabel("Metric Value")
            self.ax3.set_title("Performance Trends", fontweight="bold")
            self.ax3.legend()
            self.ax3.grid(True, alpha=0.3)

    def _plot_statistical_significance_comparison(self) -> None:
        """Plot statistical significance comparison across runs."""
        if self.test_run_history:
            run_names = [f"Run {i + 1}" for i in range(len(self.test_run_history))]
            # Mock statistical significance scores (would need real calculation)
            significance_scores = [
                run.get("falsification_rate", 0) * 0.8 for run in self.test_run_history
            ]  # Simplified

            self.ax4.bar(
                run_names,
                significance_scores,
                color=self.plot_colors["warning"],
                alpha=0.7,
            )
            self.ax4.axhline(
                y=0.05,
                color=self.plot_colors["accent"],
                linestyle="--",
                label="α = 0.05",
            )
            self.ax4.set_ylabel("Significance Score")
            self.ax4.set_title("Statistical Significance Comparison", fontweight="bold")
            self.ax4.legend()
            self.ax4.grid(True, alpha=0.3)

    def _plot_p3b_analysis(self, p3b_amplitudes: List[List[float]]) -> None:
        """Plot P3b amplitude analysis."""
        if p3b_amplitudes:
            # Flatten and plot distribution
            all_amplitudes = [amp for sublist in p3b_amplitudes for amp in sublist if sublist]
            if all_amplitudes:
                self.ax1.hist(
                    all_amplitudes,
                    bins=min(10, len(all_amplitudes)),
                    alpha=0.7,
                    color=self.plot_colors["primary"],
                )
                self.ax1.set_xlabel("P3b Amplitude (μV)")
                self.ax1.set_ylabel("Frequency")
                self.ax1.set_title("P3b Amplitude Distribution", fontweight="bold")
                self.ax1.grid(True, alpha=0.3)

    def _plot_gamma_analysis(self, gamma_powers: List[List[float]]) -> None:
        """Plot gamma power analysis."""
        if gamma_powers:
            # Flatten and plot distribution
            all_powers = [power for sublist in gamma_powers for power in sublist if sublist]
            if all_powers:
                self.ax2.hist(
                    all_powers,
                    bins=min(10, len(all_powers)),
                    alpha=0.7,
                    color=self.plot_colors["secondary"],
                )
                self.ax2.set_xlabel("Gamma Power (dB)")
                self.ax2.set_ylabel("Frequency")
                self.ax2.set_title("Gamma Power Distribution", fontweight="bold")
                self.ax2.grid(True, alpha=0.3)

    def _plot_neural_correlations(self) -> None:
        """Plot neural correlations."""
        if self.results_data:
            # Extract neural features for correlation
            p3b_vals = [r.get("p3b_amplitude", [0]) for r in self.results_data]
            gamma_vals = [r.get("gamma_power", [0]) for r in self.results_data]

            # Use mean values for correlation
            p3b_means = [np.mean(vals) if vals else 0 for vals in p3b_vals]
            gamma_means = [np.mean(vals) if vals else 0 for vals in gamma_vals]

            if p3b_means and gamma_means:
                self.ax3.scatter(
                    p3b_means, gamma_means, alpha=0.7, color=self.plot_colors["accent"]
                )
                # Add correlation line
                if len(p3b_means) > 1:
                    corr_coef = np.corrcoef(p3b_means, gamma_means)[0, 1]
                    self.ax3.text(
                        0.05,
                        0.95,
                        f"Correlation: {corr_coef:.2f}",
                        transform=self.ax3.transAxes,
                        fontsize=10,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )

                self.ax3.set_xlabel("Mean P3b Amplitude")
                self.ax3.set_ylabel("Mean Gamma Power")
                self.ax3.set_title("Neural Feature Correlation", fontweight="bold")
                self.ax3.grid(True, alpha=0.3)

    def _plot_signature_distributions(self) -> None:
        """Plot neural signature distributions."""
        if self.results_data:
            # Create mock neural signatures for visualization
            signatures = []
            for r in self.results_data:
                sig = {
                    "conscious": r.get("is_falsified", False),
                    "p3b": (np.mean(r.get("p3b_amplitude", [0])) if r.get("p3b_amplitude") else 0),
                    "gamma": (np.mean(r.get("gamma_power", [0])) if r.get("gamma_power") else 0),
                }
                signatures.append(sig)

            if signatures:
                conscious_p3b = [s["p3b"] for s in signatures if s["conscious"]]
                unconscious_p3b = [s["p3b"] for s in signatures if not s["conscious"]]

                if conscious_p3b and unconscious_p3b:
                    self.ax4.hist(
                        conscious_p3b,
                        alpha=0.7,
                        label="Conscious",
                        color=self.plot_colors["primary"],
                    )
                    self.ax4.hist(
                        unconscious_p3b,
                        alpha=0.7,
                        label="Unconscious",
                        color=self.plot_colors["accent"],
                    )
                    self.ax4.set_xlabel("P3b Amplitude")
                    self.ax4.set_ylabel("Frequency")
                    self.ax4.set_title("Neural Signatures by Consciousness", fontweight="bold")
                    self.ax4.legend()
                    self.ax4.grid(True, alpha=0.3)


# Factory function for easy instantiation
def create_results_visualization_panel(
    parent: Any, results_callback: Optional[Callable] = None
) -> ResultsVisualizationPanel:
    """Create a results visualization panel with default settings.

    Args:
        parent: Parent widget
        results_callback: Optional results callback

    Returns:
        Configured ResultsVisualizationPanel instance
    """
    return ResultsVisualizationPanel(parent, results_callback)
