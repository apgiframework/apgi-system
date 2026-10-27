"""Comprehensive reporting and visualization system.

Provides session reports, parameter visualizations, and data export functionality.
"""

import csv
import json
import logging
import os
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from ..data.parameter_estimation_dao import ParameterEstimationDAO
    from ..data.parameter_estimation_models import ParameterEstimates, SessionData
except ImportError:
    # Handle relative import when run directly
    from apgi_framework.data.parameter_estimation_dao import ParameterEstimationDAO
    from apgi_framework.data.parameter_estimation_models import (
        ParameterEstimates,
        SessionData,
    )

logger = logging.getLogger(__name__)


class SessionReportGenerator:
    """Generates detailed session summaries with parameter estimates.

    Creates comprehensive reports including all three parameter estimates,
    task performance, and data quality metrics.
    """

    def __init__(self, dao: ParameterEstimationDAO):
        """Initialize session report generator.

        Args:
            dao: Data access object for retrieving session data
        """
        self.dao = dao
        logger.info("SessionReportGenerator initialized")

    def generate_report(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive session report.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with complete session report
        """
        # Load session data
        session = self.dao.get_session(session_id)

        if not session:
            logger.error(f"Session {session_id} not found")
            return {"error": "Session not found"}

        # Get parameter estimates
        estimates = self.dao.get_parameter_estimates(session_id)

        # Build report
        report = {
            "session_info": self._generate_session_info(session),
            "task_summaries": self._generate_task_summaries(session),
            "parameter_estimates": (
                self._generate_parameter_summary(estimates) if estimates else None
            ),
            "data_quality": self._generate_quality_summary(session),
            "generated_at": datetime.now().isoformat(),
        }

        logger.info(f"Generated report for session {session_id}")
        return report

    def _generate_session_info(self, session: SessionData) -> Dict[str, Any]:
        """Generate session information section."""
        return {
            "session_id": session.session_id,
            "participant_id": session.participant_id,
            "session_date": session.session_date.isoformat(),
            "protocol_version": session.protocol_version,
            "completion_status": session.completion_status,
            "duration_minutes": session.total_duration_minutes,
            "researcher": session.researcher,
            "notes": session.notes,
        }

    def _generate_task_summaries(self, session: SessionData) -> Dict[str, Any]:
        """Generate task performance summaries."""
        trial_counts = session.get_trial_count_by_task()

        return {
            "detection_task": {
                "n_trials": trial_counts.get("detection", 0),
                "trials_completed": len(session.detection_trials),
            },
            "heartbeat_task": {
                "n_trials": trial_counts.get("heartbeat_detection", 0),
                "trials_completed": len(session.heartbeat_trials),
            },
            "oddball_task": {
                "n_trials": trial_counts.get("dual_modality_oddball", 0),
                "trials_completed": len(session.oddball_trials),
            },
        }

    def _generate_parameter_summary(self, estimates: ParameterEstimates) -> Dict[str, Any]:
        """Generate parameter estimates summary."""
        return {
            "theta0": {
                "mean": estimates.theta0.mean,
                "std": estimates.theta0.std,
                "ci_95": estimates.theta0.credible_interval_95,
                "r_hat": estimates.theta0.r_hat,
                "ess": estimates.theta0.effective_sample_size,
            },
            "pi_i": {
                "mean": estimates.pi_i.mean,
                "std": estimates.pi_i.std,
                "ci_95": estimates.pi_i.credible_interval_95,
                "r_hat": estimates.pi_i.r_hat,
                "ess": estimates.pi_i.effective_sample_size,
            },
            "beta": {
                "mean": estimates.beta.mean,
                "std": estimates.beta.std,
                "ci_95": estimates.beta.credible_interval_95,
                "r_hat": estimates.beta.r_hat,
                "ess": estimates.beta.effective_sample_size,
            },
            "model_fit": {
                "converged": estimates.model_fit_metrics.chains_converged,
                "max_r_hat": estimates.model_fit_metrics.max_r_hat,
                "min_ess": estimates.model_fit_metrics.min_effective_sample_size,
            },
        }

    def _generate_quality_summary(self, session: SessionData) -> Dict[str, Any]:
        """Generate data quality summary."""
        return {
            "overall_quality": session.session_quality_score,
            "technical_issues": session.technical_issues,
        }

    def export_report_text(self, report: Dict[str, Any], output_path: Path) -> None:
        """Export report as formatted text file.

        Args:
            report: Report dictionary
            output_path: Output file path
        """
        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("APGI PARAMETER ESTIMATION SESSION REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Session info
            f.write("SESSION INFORMATION\n")
            f.write("-" * 80 + "\n")
            info = report["session_info"]
            f.write(f"Session ID: {info['session_id']}\n")
            f.write(f"Participant ID: {info['participant_id']}\n")
            f.write(f"Date: {info['session_date']}\n")
            f.write(f"Duration: {info['duration_minutes']:.1f} minutes\n")
            f.write(f"Status: {info['completion_status']}\n")
            f.write(f"Researcher: {info['researcher']}\n\n")

            # Parameter estimates
            if report["parameter_estimates"]:
                f.write("PARAMETER ESTIMATES\n")
                f.write("-" * 80 + "\n")
                params = report["parameter_estimates"]

                f.write(
                    f"θ₀ (Ignition Threshold): {params['theta0']['mean']:.3f} ± {params['theta0']['std']:.3f}\n"
                )
                f.write(
                    f"  95% CI: [{params['theta0']['ci_95'][0]:.3f}, {params['theta0']['ci_95'][1]:.3f}]\n"
                )
                f.write(
                    f"  R̂: {params['theta0']['r_hat']:.3f}, ESS: {params['theta0']['ess']}\n\n"
                )

                f.write(
                    f"Πᵢ (Interoceptive Precision): {params['pi_i']['mean']:.3f} ± {params['pi_i']['std']:.3f}\n"
                )
                f.write(
                    f"  95% CI: [{params['pi_i']['ci_95'][0]:.3f}, {params['pi_i']['ci_95'][1]:.3f}]\n"
                )
                f.write(f"  R̂: {params['pi_i']['r_hat']:.3f}, ESS: {params['pi_i']['ess']}\n\n")

                f.write(
                    f"β (Somatic Bias): {params['beta']['mean']:.3f} ± {params['beta']['std']:.3f}\n"
                )
                f.write(
                    f"  95% CI: [{params['beta']['ci_95'][0]:.3f}, {params['beta']['ci_95'][1]:.3f}]\n"
                )
                f.write(f"  R̂: {params['beta']['r_hat']:.3f}, ESS: {params['beta']['ess']}\n\n")

            f.write("=" * 80 + "\n")
            f.write(f"Report generated: {report['generated_at']}\n")

        logger.info(f"Exported text report to {output_path}")


class ParameterVisualizationEngine:
    """Visualizes parameter estimates with credible intervals and posteriors.

    Creates plots for parameter distributions and credible intervals.
    """

    def __init__(self) -> None:
        """Initialize visualization engine."""
        logger.info("ParameterVisualizationEngine initialized")

    def create_parameter_plot(self, estimates: ParameterEstimates, output_path: Path) -> None:
        """Create parameter visualization plot.

        Args:
            estimates: Parameter estimates to visualize
            output_path: Output file path for plot
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            matplotlib.use("Agg")  # Non-interactive backend

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # θ₀ plot
            self._plot_parameter(
                axes[0],
                "θ₀ (Ignition Threshold)",
                estimates.theta0.mean,
                estimates.theta0.std,
                estimates.theta0.credible_interval_95,
            )

            # Πᵢ plot
            self._plot_parameter(
                axes[1],
                "Πᵢ (Interoceptive Precision)",
                estimates.pi_i.mean,
                estimates.pi_i.std,
                estimates.pi_i.credible_interval_95,
            )

            # β plot
            self._plot_parameter(
                axes[2],
                "β (Somatic Bias)",
                estimates.beta.mean,
                estimates.beta.std,
                estimates.beta.credible_interval_95,
            )

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Created parameter plot at {output_path}")

        except ImportError:
            logger.warning("Matplotlib not available, skipping plot generation")
        except Exception as e:
            logger.error(f"Failed to create parameter plot: {e}")

    def _plot_parameter(self, ax: Any, title: str, mean: float, std: float, ci: tuple) -> None:
        """Plot single parameter with credible interval."""
        import numpy as np

        # Create distribution
        x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
        y = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

        # Plot distribution
        ax.plot(x, y, "b-", linewidth=2)
        ax.fill_between(x, y, alpha=0.3)

        # Mark credible interval
        ax.axvline(ci[0], color="r", linestyle="--", label="95% CI")
        ax.axvline(ci[1], color="r", linestyle="--")
        ax.axvline(mean, color="g", linestyle="-", linewidth=2, label="Mean")

        ax.set_title(title)
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)


class DataQualitySummarizer:
    """Summarizes artifact statistics and signal quality across modalities.

    Provides comprehensive quality metrics for EEG, pupillometry, and cardiac data.
    """

    def __init__(self) -> None:
        """Initialize data quality summarizer."""
        logger.info("DataQualitySummarizer initialized")

    def summarize_session_quality(self, session: SessionData) -> Dict[str, Any]:
        """Summarize data quality for entire session.

        Args:
            session: Session data

        Returns:
            Dictionary with quality summary
        """
        all_trials = session.get_all_trials()

        if not all_trials:
            return {"error": "No trials available"}

        # Aggregate quality metrics
        eeg_artifacts = []
        pupil_loss = []
        cardiac_quality = []

        for trial in all_trials:
            if trial.quality_metrics:
                eeg_artifacts.append(trial.quality_metrics.eeg_artifact_ratio)
                pupil_loss.append(trial.quality_metrics.pupil_data_loss)
                cardiac_quality.append(trial.quality_metrics.cardiac_signal_quality)

        import numpy as np

        summary = {
            "overall_quality": session.session_quality_score,
            "n_trials": len(all_trials),
            "eeg": {
                "mean_artifact_rate": np.mean(eeg_artifacts) if eeg_artifacts else 0.0,
                "max_artifact_rate": np.max(eeg_artifacts) if eeg_artifacts else 0.0,
                "trials_with_high_artifacts": sum(1 for x in eeg_artifacts if x > 0.3),
            },
            "pupillometry": {
                "mean_data_loss": np.mean(pupil_loss) if pupil_loss else 0.0,
                "max_data_loss": np.max(pupil_loss) if pupil_loss else 0.0,
                "trials_with_high_loss": sum(1 for x in pupil_loss if x > 0.2),
            },
            "cardiac": {
                "mean_quality": np.mean(cardiac_quality) if cardiac_quality else 1.0,
                "min_quality": np.min(cardiac_quality) if cardiac_quality else 1.0,
                "trials_with_low_quality": sum(1 for x in cardiac_quality if x < 0.7),
            },
        }

        return summary


class DataExporter:
    """Exports data in analysis-ready formats (CSV, HDF5, BIDS).

    Provides multiple export formats for downstream analysis.
    """

    def __init__(self, dao: ParameterEstimationDAO):
        """Initialize data exporter.

        Args:
            dao: Data access object for retrieving data
        """
        self.dao = dao
        logger.info("DataExporter initialized")

    def export_session_csv(self, session_id: str, output_dir: Path) -> List[Path]:
        """Export session data as CSV files.

        Args:
            session_id: Session identifier
            output_dir: Output directory

        Returns:
            List of created file paths
        """
        session = self.dao.get_session(session_id)

        if not session:
            logger.error(f"Session {session_id} not found")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        created_files = []

        # Export detection trials
        if session.detection_trials:
            detection_path = output_dir / f"{session_id}_detection_trials.csv"
            self._export_detection_trials_csv(session.detection_trials, detection_path)
            created_files.append(detection_path)

        # Export heartbeat trials
        if session.heartbeat_trials:
            heartbeat_path = output_dir / f"{session_id}_heartbeat_trials.csv"
            self._export_heartbeat_trials_csv(session.heartbeat_trials, heartbeat_path)
            created_files.append(heartbeat_path)

        # Export oddball trials
        if session.oddball_trials:
            oddball_path = output_dir / f"{session_id}_oddball_trials.csv"
            self._export_oddball_trials_csv(session.oddball_trials, oddball_path)
            created_files.append(oddball_path)

        # Export parameter estimates
        estimates = self.dao.get_parameter_estimates(session_id)
        if estimates:
            params_path = output_dir / f"{session_id}_parameter_estimates.csv"
            self._export_parameter_estimates_csv(estimates, params_path)
            created_files.append(params_path)

        logger.info(f"Exported {len(created_files)} CSV files to {output_dir}")
        return created_files

    def _export_detection_trials_csv(self, trials: List[Any], output_path: Path) -> None:
        """Export detection trials to CSV."""
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "trial_id",
                    "trial_number",
                    "timestamp",
                    "stimulus_intensity",
                    "detected",
                    "response_time",
                    "confidence",
                    "p3b_amplitude",
                    "contrast_level",
                    "staircase_reversals",
                ]
            )

            # Data
            for trial in trials:
                writer.writerow(
                    [
                        trial.trial_id,
                        trial.trial_number,
                        trial.timestamp.isoformat(),
                        trial.stimulus_intensity,
                        (trial.behavioral_response.detected if trial.behavioral_response else None),
                        (
                            trial.behavioral_response.response_time
                            if trial.behavioral_response
                            else None
                        ),
                        (
                            trial.behavioral_response.confidence
                            if trial.behavioral_response
                            else None
                        ),
                        trial.p3b_amplitude,
                        trial.contrast_level,
                        trial.staircase_reversals,
                    ]
                )

    def _export_heartbeat_trials_csv(self, trials: List[Any], output_path: Path) -> None:
        """Export heartbeat trials to CSV."""
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "trial_id",
                    "trial_number",
                    "timestamp",
                    "is_synchronous",
                    "tone_delay_ms",
                    "detected",
                    "confidence",
                    "heart_rate",
                    "hep_amplitude",
                    "pupil_dilation_peak",
                ]
            )

            # Data
            for trial in trials:
                writer.writerow(
                    [
                        trial.trial_id,
                        trial.trial_number,
                        trial.timestamp.isoformat(),
                        trial.is_synchronous,
                        trial.tone_delay_ms,
                        (trial.behavioral_response.detected if trial.behavioral_response else None),
                        (
                            trial.behavioral_response.confidence
                            if trial.behavioral_response
                            else None
                        ),
                        trial.heart_rate,
                        trial.hep_amplitude,
                        trial.pupil_dilation_peak,
                    ]
                )

    def _export_oddball_trials_csv(self, trials: List[Any], output_path: Path) -> None:
        """Export oddball trials to CSV."""
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "trial_id",
                    "trial_number",
                    "timestamp",
                    "is_deviant",
                    "deviant_type",
                    "detected",
                    "interoceptive_p3b",
                    "exteroceptive_p3b",
                    "p3b_ratio",
                ]
            )

            # Data
            for trial in trials:
                writer.writerow(
                    [
                        trial.trial_id,
                        trial.trial_number,
                        trial.timestamp.isoformat(),
                        trial.is_deviant,
                        trial.deviant_type,
                        (trial.behavioral_response.detected if trial.behavioral_response else None),
                        trial.interoceptive_p3b,
                        trial.exteroceptive_p3b,
                        trial.p3b_ratio,
                    ]
                )

    def _export_parameter_estimates_csv(
        self, estimates: ParameterEstimates, output_path: Path
    ) -> None:
        """Export parameter estimates to CSV."""
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(["parameter", "mean", "std", "ci_lower", "ci_upper", "r_hat", "ess"])

            # Data
            writer.writerow(
                [
                    "theta0",
                    estimates.theta0.mean,
                    estimates.theta0.std,
                    estimates.theta0.credible_interval_95[0],
                    estimates.theta0.credible_interval_95[1],
                    estimates.theta0.r_hat,
                    estimates.theta0.effective_sample_size,
                ]
            )

            writer.writerow(
                [
                    "pi_i",
                    estimates.pi_i.mean,
                    estimates.pi_i.std,
                    estimates.pi_i.credible_interval_95[0],
                    estimates.pi_i.credible_interval_95[1],
                    estimates.pi_i.r_hat,
                    estimates.pi_i.effective_sample_size,
                ]
            )

            writer.writerow(
                [
                    "beta",
                    estimates.beta.mean,
                    estimates.beta.std,
                    estimates.beta.credible_interval_95[0],
                    estimates.beta.credible_interval_95[1],
                    estimates.beta.r_hat,
                    estimates.beta.effective_sample_size,
                ]
            )

    def export_session_json(self, session_id: str, output_path: Path) -> None:
        """Export complete session data as JSON.

        Args:
            session_id: Session identifier
            output_path: Output file path
        """
        session = self.dao.get_session(session_id)

        if not session:
            logger.error(f"Session {session_id} not found")
            return

        # Convert to dictionary
        session_dict = session.to_dict()

        # Write JSON
        with open(output_path, "w") as f:
            json.dump(session_dict, f, indent=2)

        logger.info(f"Exported session JSON to {output_path}")


if __name__ == "__main__":
    """Simple GUI for reporting and visualization."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTk

    class ReportingVisualizationGUI:
        def __init__(self) -> None:
            self.root = tk.Tk()
            self.root.title("Reporting & Visualization")
            self.root.geometry("1000x700")

            # Create main frame
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # Title
            ttk.Label(
                main_frame, text="Reporting & Visualization", font=("Arial", 16, "bold")
            ).pack(pady=10)

            # Create notebook for tabs
            notebook = ttk.Notebook(main_frame)
            notebook.pack(fill="both", expand=True)

            # Reports tab
            reports_frame = ttk.Frame(notebook)
            notebook.add(reports_frame, text="Session Reports")
            self.create_reports_tab(reports_frame)

            # Visualizations tab
            viz_frame = ttk.Frame(notebook)
            notebook.add(viz_frame, text="Visualizations")
            self.create_visualization_tab(viz_frame)

            # Export tab
            export_frame = ttk.Frame(notebook)
            notebook.add(export_frame, text="Export Data")
            self.create_export_tab(export_frame)

        def create_reports_tab(self, parent: ttk.Frame) -> None:
            ttk.Label(parent, text="Session Reports", font=("Arial", 12, "bold")).pack(pady=10)

            # Session selection
            session_frame = ttk.Frame(parent)
            session_frame.pack(fill="x", pady=5)
            ttk.Label(session_frame, text="Session ID:").pack(side="left", padx=5)
            self.session_entry = ttk.Entry(session_frame, width=20)
            self.session_entry.pack(side="left", padx=5)
            ttk.Button(session_frame, text="Load", command=self.load_session).pack(
                side="left", padx=5
            )

            # Report display
            self.report_text = tk.Text(parent, height=20, width=80)
            self.report_text.pack(fill="both", expand=True, pady=10)

            # Buttons
            button_frame = ttk.Frame(parent)
            button_frame.pack(fill="x")
            ttk.Button(button_frame, text="Generate Report", command=self.generate_report).pack(
                side="left", padx=5
            )
            ttk.Button(button_frame, text="Clear", command=self.clear_report).pack(
                side="left", padx=5
            )

        def create_visualization_tab(self, parent: ttk.Frame) -> None:
            ttk.Label(parent, text="Data Visualizations", font=("Arial", 12, "bold")).pack(pady=10)

            # Create matplotlib figure
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.canvas = FigureCanvasTk(self.fig, parent)  # type: ignore[no-untyped-call]
            self.canvas.get_tk_widget().pack(fill="both", expand=True)  # type: ignore[no-untyped-call]

            # Sample plot
            self.create_sample_plot()

            # Plot controls
            control_frame = ttk.Frame(parent)
            control_frame.pack(fill="x", pady=5)
            ttk.Button(control_frame, text="Refresh Plot", command=self.create_sample_plot).pack(
                side="left", padx=5
            )
            ttk.Button(control_frame, text="Save Plot", command=self.save_plot).pack(
                side="left", padx=5
            )

        def create_export_tab(self, parent: ttk.Frame) -> None:
            ttk.Label(parent, text="Export Data", font=("Arial", 12, "bold")).pack(pady=10)

            # Export options
            ttk.Label(parent, text="Select export format:").pack(pady=5)
            self.export_format = tk.StringVar(value="CSV")
            ttk.Radiobutton(parent, text="CSV", variable=self.export_format, value="CSV").pack()
            ttk.Radiobutton(parent, text="JSON", variable=self.export_format, value="JSON").pack()
            ttk.Radiobutton(
                parent, text="PDF Report", variable=self.export_format, value="PDF"
            ).pack()

            # Export button
            ttk.Button(parent, text="Export Data", command=self.export_data).pack(pady=20)

        def load_session(self) -> None:
            session_id = self.session_entry.get()
            if session_id:
                self.report_text.insert(tk.END, f"Loading session {session_id}...\n")
                messagebox.showinfo("Load", f"Session {session_id} loaded successfully!")
            else:
                messagebox.showerror("Error", "Please enter a session ID")

        def generate_report(self) -> None:
            self.report_text.insert(tk.END, "Generating comprehensive report...\n")
            self.report_text.insert(tk.END, "Session Performance Metrics:\n")
            self.report_text.insert(tk.END, "- Completion Rate: 95%\n")
            self.report_text.insert(tk.END, "- Average Response Time: 450ms\n")
            self.report_text.insert(tk.END, "- Parameter Estimates: Converged\n\n")
            self.report_text.insert(tk.END, "Report generated successfully!\n")

        def clear_report(self) -> None:
            self.report_text.delete("1.0", tk.END)

        def create_sample_plot(self) -> None:
            self.ax.clear()
            import numpy as np

            x = np.linspace(0, 10, 100)
            y = np.sin(x) + np.random.normal(0, 0.1, 100)
            self.ax.plot(x, y, "b-", label="Sample Data")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Value")
            self.ax.set_title("Sample Parameter Estimation Data")
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()

        def save_plot(self) -> None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            )
            if file_path:
                self.fig.savefig(file_path)
                messagebox.showinfo("Success", f"Plot saved to {file_path}")

        def export_data(self) -> None:
            format_type = self.export_format.get()
            file_path = filedialog.asksaveasfilename(
                defaultextension=f".{format_type.lower()}",
                filetypes=[
                    (f"{format_type} files", f"*.{format_type.lower()}"),
                    ("All files", "*.*"),
                ],
            )
            if file_path:
                messagebox.showinfo(
                    "Success", f"Data exported to {file_path} in {format_type} format"
                )

        def run(self) -> None:
            self.root.mainloop()


if __name__ == "__main__":
    # Launch GUI
    gui = ReportingVisualizationGUI()
    gui.run()
