"""Integrated Data Management System for APGI Framework.

This module provides a unified interface for all data management, reporting,
and visualization capabilities.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.data_models import (
    ExperimentalTrial,
    FalsificationResult,
    StatisticalSummary,
)
from ..exceptions import DataManagementError
from .dashboard import DashboardServer, ExperimentMonitor
from .data_exporter import DataExporter
from .report_generator import ReportGenerator
from .visualizer import APGIVisualizer, InteractiveVisualizer


class IntegratedDataManager:
    """Unified data management system that coordinates all data operations."""

    def __init__(
        self,
        base_output_dir: str = "output",
        enable_dashboard: bool = True,
        dashboard_port: int = 8080,
    ):
        """Initialize the integrated data manager.

        Args:
            base_output_dir: Base directory for all outputs
            enable_dashboard: Whether to enable the web dashboard
            dashboard_port: Port for the dashboard server
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.report_generator = ReportGenerator(str(self.base_output_dir / "reports"))
        self.data_exporter = DataExporter(str(self.base_output_dir / "exports"))
        self.visualizer = APGIVisualizer(str(self.base_output_dir / "figures"))
        self.interactive_visualizer = InteractiveVisualizer()  # type: ignore[no-untyped-call]

        # Dashboard setup
        self.enable_dashboard = enable_dashboard
        self.dashboard: Optional[DashboardServer] = None
        self.experiment_monitor: ExperimentMonitor = ExperimentMonitor()

        if enable_dashboard:
            self.dashboard = DashboardServer(
                port=dashboard_port, data_dir=str(self.base_output_dir / "dashboard")
            )
            self.experiment_monitor = self.dashboard.monitor

        self.logger = logging.getLogger(__name__)

        # Active experiments tracking
        self.active_experiments: Dict[str, Any] = {}

    def start_system(self) -> None:
        """Start the data management system."""
        try:
            if self.enable_dashboard:
                if self.dashboard is not None:
                    self.dashboard.start_dashboard()
            else:
                self.experiment_monitor.start_monitoring()

            self.logger.info("Data management system started successfully")

        except Exception as e:
            raise DataManagementError(f"Failed to start data management system: {str(e)}")

    def stop_system(self) -> None:
        """Stop the data management system."""
        try:
            if self.enable_dashboard:
                if self.dashboard is not None:
                    self.dashboard.stop_dashboard()
            else:
                self.experiment_monitor.stop_monitoring()

            self.logger.info("Data management system stopped")

        except Exception as e:
            self.logger.error(f"Error stopping data management system: {str(e)}")

    def register_experiment(
        self, experiment_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a new experiment for tracking and management.

        Args:
            experiment_id: Unique experiment identifier
            metadata: Experiment metadata

        Returns:
            Experiment ID

        Raises:
            DataManagementError: If experiment is already registered or ID is invalid
        """
        # Validate experiment_id
        if not experiment_id:
            raise DataManagementError("Experiment ID cannot be empty")

        # Check for duplicate experiment
        if experiment_id in self.active_experiments:
            raise DataManagementError(f"Experiment {experiment_id} is already registered")

        try:
            # Register with monitor
            self.experiment_monitor.register_experiment(experiment_id, metadata or {})

            # Track locally
            self.active_experiments[experiment_id] = {
                "metadata": metadata or {},
                "status": "registered",
                "start_time": datetime.now(),
                "results": [],
                "trials": [],
                "reports": [],
                "exports": [],
                "figures": [],
            }

            self.logger.info(f"Registered experiment {experiment_id}")
            return experiment_id

        except Exception as e:
            raise DataManagementError(f"Failed to register experiment: {str(e)}")

    def update_experiment_data(
        self,
        experiment_id: str,
        results: Optional[List[FalsificationResult]] = None,
        trials: Optional[List[ExperimentalTrial]] = None,
        current_trial: Optional[int] = None,
    ) -> None:
        """Update experiment data and notify monitoring systems.

        Args:
            experiment_id: Experiment identifier
            results: New falsification results
            trials: New experimental trials
            current_trial: Current trial number
        """
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not registered")

            exp_data = self.active_experiments[experiment_id]

            # Update local data
            if results:
                exp_data["results"].extend(results)

            if trials:
                exp_data["trials"].extend(trials)

            # Update monitor
            if current_trial is not None:
                self.experiment_monitor.update_experiment_progress(
                    experiment_id, current_trial, results or [], trials or []
                )

            self.logger.debug(f"Updated data for experiment {experiment_id}")

        except Exception as e:
            raise DataManagementError(f"Failed to update experiment data: {str(e)}")

    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        """Update experiment status.

        Args:
            experiment_id: Experiment identifier
            status: New status (e.g., "pending", "running", "completed", "failed")

        Raises:
            DataManagementError: If experiment not found or update fails
        """
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not registered")

            self.active_experiments[experiment_id]["status"] = status
            self.logger.debug(f"Updated status for experiment {experiment_id} to {status}")

        except Exception as e:
            raise DataManagementError(f"Failed to update experiment status: {str(e)}")

    def update_experiment_progress(self, experiment_id: str, progress: int) -> None:
        """Update experiment progress percentage.

        Args:
            experiment_id: Experiment identifier
            progress: Progress percentage (0-100)

        Raises:
            DataManagementError: If experiment not found or update fails
        """
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not registered")

            self.active_experiments[experiment_id]["progress"] = progress
            self.logger.debug(f"Updated progress for experiment {experiment_id} to {progress}%")

        except Exception as e:
            raise DataManagementError(f"Failed to update experiment progress: {str(e)}")

    def complete_experiment(
        self, experiment_id: str, results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark experiment as completed."""
        try:
            if experiment_id in self.active_experiments:
                self.active_experiments[experiment_id]["status"] = "completed"
                if results:
                    self.active_experiments[experiment_id]["results"] = results
                self.experiment_monitor.complete_experiment(experiment_id)
                self.logger.info(f"Completed experiment {experiment_id}")

        except Exception as e:
            raise DataManagementError(f"Failed to complete experiment: {str(e)}")

    def generate_comprehensive_report(
        self,
        experiment_id: str,
        statistical_summary: StatisticalSummary,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Generate comprehensive report for an experiment.

        Args:
            experiment_id: Experiment identifier
            statistical_summary: Statistical analysis summary
            formats: Report formats to generate

        Returns:
            Dictionary mapping format to file path
        """
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not registered")

            exp_data = self.active_experiments[experiment_id]
            results = exp_data["results"]
            trials = exp_data["trials"]

            if not results:
                raise ValueError(f"No results available for experiment {experiment_id}")

            # Generate report
            report = self.report_generator.generate_falsification_report(
                experiment_id, results, trials, statistical_summary
            )

            # Save in multiple formats
            if formats is None:
                formats = ["json", "txt", "html"]

            report_paths = {}
            for format in formats:
                path = self.report_generator.save_report(report, format)
                report_paths[format] = path
                exp_data["reports"].append(path)

            self.logger.info(f"Generated comprehensive report for experiment {experiment_id}")
            return report_paths

        except Exception as e:
            raise DataManagementError(f"Failed to generate comprehensive report: {str(e)}")

    def export_experiment_data(
        self, experiment_id: str, formats: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, str]]:
        """Export experiment data in multiple formats.

        Args:
            experiment_id: Experiment identifier
            formats: Export formats

        Returns:
            Dictionary mapping data type and format to file path
        """
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not registered")

            exp_data = self.active_experiments[experiment_id]
            results = exp_data["results"]
            trials = exp_data["trials"]

            if formats is None:
                formats = ["csv", "json", "hdf5"]

            export_paths: Dict[str, Dict[str, str]] = {"results": {}, "trials": {}}

            # Export results
            if results:
                for format in formats:
                    path = self.data_exporter.export_falsification_results(
                        results, format, f"{experiment_id}_results"
                    )
                    export_paths["results"][format] = path
                    exp_data["exports"].append(path)

            # Export trials
            if trials:
                for format in formats:
                    path = self.data_exporter.export_experimental_trials(
                        trials, format, f"{experiment_id}_trials"
                    )
                    export_paths["trials"][format] = path
                    exp_data["exports"].append(path)

            self.logger.info(f"Exported data for experiment {experiment_id}")
            return export_paths

        except Exception as e:
            raise DataManagementError(f"Failed to export experiment data: {str(e)}")

    def generate_visualizations(
        self,
        experiment_id: str,
        statistical_summary: StatisticalSummary,
        create_publication_set: bool = True,
    ) -> List[str]:
        """Generate visualizations for an experiment.

        Args:
            experiment_id: Experiment identifier
            statistical_summary: Statistical analysis summary
            create_publication_set: Whether to create publication-ready figure set

        Returns:
            List of paths to generated figures
        """
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not registered")

            exp_data = self.active_experiments[experiment_id]
            results = exp_data["results"]
            trials = exp_data["trials"]

            if not results or not trials:
                raise ValueError(
                    f"Insufficient data for visualization in experiment {experiment_id}"
                )

            figure_paths = []

            if create_publication_set:
                # Generate complete publication figure set
                pub_figures = self.visualizer.create_publication_figure_set(
                    results, trials, statistical_summary, experiment_id
                )
                figure_paths.extend(pub_figures)
            else:
                # Generate individual figures
                figures = [
                    self.visualizer.plot_falsification_summary(results),
                    self.visualizer.plot_neural_signatures(trials),
                    self.visualizer.plot_apgi_parameter_space(trials),
                    self.visualizer.plot_statistical_summary(statistical_summary),
                ]
                figure_paths.extend(figures)

            # Update experiment data
            exp_data["figures"].extend(figure_paths)

            self.logger.info(
                f"Generated {len(figure_paths)} visualizations for experiment {experiment_id}"
            )
            return figure_paths

        except Exception as e:
            raise DataManagementError(f"Failed to generate visualizations: {str(e)}")

    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment summary."""
        try:
            if experiment_id not in self.active_experiments:
                return {}

            exp_data = self.active_experiments[experiment_id]
            monitor_data = self.experiment_monitor.get_experiment_status(experiment_id)

            summary = {
                "experiment_id": experiment_id,
                "metadata": exp_data["metadata"],
                "start_time": exp_data["start_time"].isoformat(),
                "status": (monitor_data.get("status", "unknown") if monitor_data else "unknown"),
                "progress": monitor_data.get("progress", 0.0) if monitor_data else 0.0,
                "statistics": (monitor_data.get("statistics", {}) if monitor_data else {}),
                "data_counts": {
                    "results": len(exp_data["results"]),
                    "trials": len(exp_data["trials"]),
                    "reports": len(exp_data["reports"]),
                    "exports": len(exp_data["exports"]),
                    "figures": len(exp_data["figures"]),
                },
                "files": {
                    "reports": exp_data["reports"],
                    "exports": exp_data["exports"],
                    "figures": exp_data["figures"],
                },
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error getting experiment summary: {str(e)}")
            return {}

    def get_all_experiments_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all experiments."""
        return {
            exp_id: self.get_experiment_summary(exp_id) for exp_id in self.active_experiments.keys()
        }

    def cleanup_experiment(self, experiment_id: str, keep_files: bool = True) -> None:
        """Clean up experiment data and resources.

        Args:
            experiment_id: Experiment identifier
            keep_files: Whether to keep generated files
        """
        try:
            if experiment_id in self.active_experiments:
                exp_data = self.active_experiments[experiment_id]

                if not keep_files:
                    # Delete generated files
                    all_files = exp_data["reports"] + exp_data["exports"] + exp_data["figures"]

                    for file_path in all_files:
                        try:
                            Path(file_path).unlink(missing_ok=True)
                        except Exception as e:
                            self.logger.warning(f"Could not delete file {file_path}: {str(e)}")

                # Remove from tracking
                del self.active_experiments[experiment_id]

                self.logger.info(f"Cleaned up experiment {experiment_id}")

        except Exception as e:
            self.logger.error(f"Error cleaning up experiment {experiment_id}: {str(e)}")

    def get_dashboard_url(self) -> Optional[str]:
        """Get dashboard URL if enabled."""
        if self.enable_dashboard and self.dashboard:
            return f"http://localhost:{self.dashboard.port}"
        return None

    def validate_data(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate data integrity and structure.

        Args:
            data: Data dictionary to validate (if None, validates all active experiments)

        Returns:
            Validation results with status and any warnings
        """
        try:
            if data is not None:
                # Validate provided data
                validation_results: Dict[str, Any] = {
                    "status": "valid",
                    "warnings": [],
                    "errors": [],
                    "data_keys": list(data.keys()),
                }

                # Check for empty/null values
                for key, value in data.items():
                    if value is None or (isinstance(value, (list, dict)) and len(value) == 0):
                        validation_results["warnings"].append(f"Empty/null value for key: {key}")

                return validation_results

            # Validate all active experiments
            validation_results = {
                "status": "valid",
                "warnings": [],
                "errors": [],
                "experiments_validated": 0,
            }

            for experiment_id in self.active_experiments:
                exp_data = self.active_experiments.get(experiment_id, {})
                if not exp_data.get("results") and not exp_data.get("trials"):
                    validation_results["warnings"].append(
                        f"Experiment {experiment_id} has no results or trials"
                    )
                validation_results["experiments_validated"] += 1

            return validation_results

        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            return {"status": "error", "errors": [str(e)]}

    def clean_data(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Clean data by removing empty/null values and fixing issues.

        Args:
            data: Data dictionary to clean (if None, cleans active experiment results)

        Returns:
            Cleaning results with operations performed
        """
        try:
            cleaning_results: Dict[str, Any] = {
                "operations": [],
                "warnings": [],
                "fixed_issues": 0,
            }

            if data is not None:
                # Clean provided data dictionary
                keys_to_remove = []
                for key, value in list(data.items()):
                    if value is None:
                        keys_to_remove.append(key)
                    elif isinstance(value, dict):
                        # Recursively clean nested dicts
                        nested_clean = self.clean_data(value)
                        cleaning_results["operations"].extend(nested_clean.get("operations", []))

                for key in keys_to_remove:
                    del data[key]
                    cleaning_results["operations"].append(f"Removed null value for key: {key}")
                    cleaning_results["fixed_issues"] += 1

                return cleaning_results

            # Clean active experiment results
            for experiment_id in list(self.active_experiments.keys()):
                exp_data = self.active_experiments[experiment_id]

                # Remove empty results
                if "results" in exp_data and not exp_data["results"]:
                    cleaning_results["warnings"].append(
                        f"Experiment {experiment_id} has empty results"
                    )

                # Clean up any None values in results
                if isinstance(exp_data.get("results"), list):
                    original_count = len(exp_data["results"])
                    exp_data["results"] = [r for r in exp_data["results"] if r is not None]
                    removed_count = original_count - len(exp_data["results"])
                    if removed_count > 0:
                        cleaning_results["operations"].append(
                            f"Removed {removed_count} null results from {experiment_id}"
                        )
                        cleaning_results["fixed_issues"] += removed_count

            return cleaning_results

        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            return {"operations": [], "warnings": [str(e)], "fixed_issues": 0}


# Convenience function for easy setup
def create_data_manager(
    output_dir: str = "apgi_outputs",
    enable_dashboard: bool = True,
    dashboard_port: int = 8080,
) -> IntegratedDataManager:
    """Create and configure an integrated data manager.

    Args:
        output_dir: Base output directory
        enable_dashboard: Whether to enable dashboard
        dashboard_port: Dashboard port

    Returns:
        Configured data manager
    """
    return IntegratedDataManager(
        base_output_dir=output_dir,
        enable_dashboard=enable_dashboard,
        dashboard_port=dashboard_port,
    )
