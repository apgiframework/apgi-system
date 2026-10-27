"""Comprehensive Reporting Dashboard for APGI Framework.

This module provides a web-based dashboard for experiment monitoring,
real-time progress tracking, and experiment comparison and analysis tools.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..core.data_models import ExperimentalTrial, FalsificationResult
from ..exceptions import DashboardError
from .visualizer import InteractiveVisualizer


class ExperimentMonitor:
    """Monitors experiment progress and provides real-time updates."""

    def __init__(self, update_interval: float = 1.0):
        """Initialize experiment monitor.

        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.is_monitoring = False
        self.experiments: Dict[str, Any] = {}
        self.listeners: List[Callable] = []
        self.logger = logging.getLogger(__name__)

        # Thread-safe data structures
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            self.logger.info("Started experiment monitoring")

    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Stopped experiment monitoring")

    def register_experiment(self, experiment_id: str, metadata: Dict[str, Any]) -> None:
        """Register a new experiment for monitoring."""
        with self._lock:
            self.experiments[experiment_id] = {
                "id": experiment_id,
                "metadata": metadata,
                "start_time": datetime.now(),
                "status": "running",
                "progress": 0.0,
                "current_trial": 0,
                "total_trials": metadata.get("total_trials", 0),
                "results": [],
                "trials": [],
                "last_update": datetime.now(),
                "statistics": {
                    "trials_per_second": 0.0,
                    "estimated_completion": None,
                    "falsifications_detected": 0,
                },
            }

        self._notify_listeners("experiment_registered", experiment_id)
        self.logger.info(f"Registered experiment {experiment_id} for monitoring")

    def update_experiment_progress(
        self,
        experiment_id: str,
        current_trial: int,
        new_results: Optional[List[FalsificationResult]] = None,
        new_trials: Optional[List[ExperimentalTrial]] = None,
    ) -> None:
        """Update experiment progress."""
        with self._lock:
            if experiment_id not in self.experiments:
                return

            exp = self.experiments[experiment_id]
            exp["current_trial"] = current_trial
            exp["last_update"] = datetime.now()

            if exp["total_trials"] > 0:
                exp["progress"] = current_trial / exp["total_trials"]

            if new_results:
                exp["results"].extend(new_results)
                exp["statistics"]["falsifications_detected"] = sum(
                    1 for r in exp["results"] if r.is_falsified
                )

            if new_trials:
                exp["trials"].extend(new_trials)

            # Update statistics
            self._update_experiment_statistics(experiment_id)

        self._notify_listeners("experiment_updated", experiment_id)

    def complete_experiment(self, experiment_id: str) -> None:
        """Mark experiment as completed."""
        with self._lock:
            if experiment_id in self.experiments:
                self.experiments[experiment_id]["status"] = "completed"
                self.experiments[experiment_id]["progress"] = 1.0
                self.experiments[experiment_id]["last_update"] = datetime.now()

        self._notify_listeners("experiment_completed", experiment_id)
        self.logger.info(f"Experiment {experiment_id} completed")

    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get current experiment status."""
        with self._lock:
            result: Dict[str, Any] = self.experiments.get(experiment_id, {}).copy()
            return result if result else None

    def get_all_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all experiments."""
        with self._lock:
            return {eid: exp.copy() for eid, exp in self.experiments.items()}

    def add_listener(self, callback: Callable[[str, str], None]) -> None:
        """Add event listener for experiment updates."""
        self.listeners.append(callback)

    def remove_listener(self, callback: Callable[[str, str], None]) -> None:
        """Remove event listener."""
        if callback in self.listeners:
            self.listeners.remove(callback)

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                with self._lock:
                    for experiment_id in list(self.experiments.keys()):
                        self._update_experiment_statistics(experiment_id)

                time.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.update_interval)

    def _update_experiment_statistics(self, experiment_id: str) -> None:
        """Update experiment statistics."""
        if experiment_id not in self.experiments:
            return

        exp = self.experiments[experiment_id]
        now = datetime.now()
        elapsed = (now - exp["start_time"]).total_seconds()

        if elapsed > 0 and exp["current_trial"] > 0:
            exp["statistics"]["trials_per_second"] = exp["current_trial"] / elapsed

            if exp["statistics"]["trials_per_second"] > 0 and exp["total_trials"] > 0:
                remaining_trials = exp["total_trials"] - exp["current_trial"]
                remaining_seconds = remaining_trials / exp["statistics"]["trials_per_second"]
                exp["statistics"]["estimated_completion"] = now + timedelta(
                    seconds=remaining_seconds
                )

    def _notify_listeners(self, event_type: str, experiment_id: str) -> None:
        """Notify all listeners of an event."""
        for listener in self.listeners:
            try:
                listener(event_type, experiment_id)
            except Exception as e:
                self.logger.error(f"Error notifying listener: {str(e)}")


class ExperimentComparator:
    """Provides experiment comparison and analysis capabilities."""

    def __init__(self) -> None:
        """Initialize experiment comparator."""
        self.logger = logging.getLogger(__name__)

    def compare_experiments(self, experiments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple experiments and generate analysis.

        Args:
            experiments: Dictionary of experiment data

        Returns:
            Comparison analysis results
        """
        try:
            comparison = {
                "summary": self._generate_comparison_summary(experiments),
                "statistical_comparison": self._compare_statistics(experiments),
                "parameter_analysis": self._analyze_parameters(experiments),
                "performance_metrics": self._compare_performance(experiments),
                "recommendations": self._generate_recommendations(experiments),
            }

            return comparison

        except Exception as e:
            raise DashboardError(f"Failed to compare experiments: {str(e)}")

    def _generate_comparison_summary(
        self, experiments: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate high-level comparison summary."""
        total_experiments = len(experiments)
        completed_experiments = sum(
            1 for exp in experiments.values() if exp["status"] == "completed"
        )

        all_results = []
        for exp in experiments.values():
            all_results.extend(exp.get("results", []))

        falsification_rate = (
            sum(1 for r in all_results if r.is_falsified) / len(all_results) if all_results else 0
        )

        return {
            "total_experiments": total_experiments,
            "completed_experiments": completed_experiments,
            "total_tests": len(all_results),
            "overall_falsification_rate": falsification_rate,
            "experiment_ids": list(experiments.keys()),
        }

    def _compare_statistics(self, experiments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare statistical metrics across experiments."""
        stats_by_experiment = {}

        for exp_id, exp_data in experiments.items():
            results = exp_data.get("results", [])

            if results:
                stats_by_experiment[exp_id] = {
                    "mean_effect_size": sum(r.effect_size for r in results) / len(results),
                    "mean_p_value": sum(r.p_value for r in results) / len(results),
                    "mean_power": sum(r.statistical_power for r in results) / len(results),
                    "falsification_count": sum(1 for r in results if r.is_falsified),
                    "total_tests": len(results),
                }

        # Calculate cross-experiment statistics
        if stats_by_experiment:
            all_effect_sizes = [stats["mean_effect_size"] for stats in stats_by_experiment.values()]
            all_powers = [stats["mean_power"] for stats in stats_by_experiment.values()]

            comparison_stats = {
                "effect_size_range": (min(all_effect_sizes), max(all_effect_sizes)),
                "power_range": (min(all_powers), max(all_powers)),
                "most_powerful_experiment": max(
                    stats_by_experiment.keys(),
                    key=lambda k: stats_by_experiment[k]["mean_power"],
                ),
                "highest_effect_size_experiment": max(
                    stats_by_experiment.keys(),
                    key=lambda k: stats_by_experiment[k]["mean_effect_size"],
                ),
            }
        else:
            comparison_stats = {}

        return {"by_experiment": stats_by_experiment, "comparison": comparison_stats}

    def _analyze_parameters(self, experiments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze parameter distributions across experiments."""
        parameter_analysis: Dict[str, List[float]] = {
            "extero_precision": [],
            "intero_precision": [],
            "threshold": [],
            "somatic_gain": [],
        }

        for exp_data in experiments.values():
            trials = exp_data.get("trials", [])

            for trial in trials:
                if hasattr(trial, "apgi_parameters"):
                    params = trial.apgi_parameters
                    parameter_analysis["extero_precision"].append(params.extero_precision)
                    parameter_analysis["intero_precision"].append(params.intero_precision)
                    parameter_analysis["threshold"].append(params.threshold)
                    parameter_analysis["somatic_gain"].append(params.somatic_gain)

        # Calculate statistics for each parameter
        param_stats = {}
        for param_name, values in parameter_analysis.items():
            if values:
                param_stats[param_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values))
                    ** 0.5,
                }

        return param_stats

    def _compare_performance(self, experiments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare performance metrics across experiments."""
        performance_metrics = {}

        for exp_id, exp_data in experiments.items():
            stats = exp_data.get("statistics", {})

            performance_metrics[exp_id] = {
                "trials_per_second": stats.get("trials_per_second", 0),
                "total_runtime": None,
                "efficiency_score": 0,
            }

            # Calculate runtime if experiment is completed
            if exp_data["status"] == "completed":
                start_time = exp_data.get("start_time")
                last_update = exp_data.get("last_update")

                if start_time and last_update:
                    if isinstance(start_time, str):
                        start_time = datetime.fromisoformat(start_time)
                    if isinstance(last_update, str):
                        last_update = datetime.fromisoformat(last_update)

                    runtime = (last_update - start_time).total_seconds()
                    performance_metrics[exp_id]["total_runtime"] = runtime

                    # Calculate efficiency score (trials per second normalized)
                    if runtime > 0:
                        total_trials = exp_data.get(
                            "total_trials", exp_data.get("current_trial", 0)
                        )
                        performance_metrics[exp_id]["efficiency_score"] = total_trials / runtime

        return performance_metrics

    def _generate_recommendations(self, experiments: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on experiment comparison."""
        recommendations = []

        # Analyze falsification patterns
        all_results = []
        for exp in experiments.values():
            all_results.extend(exp.get("results", []))

        if all_results:
            falsification_rate = sum(1 for r in all_results if r.is_falsified) / len(all_results)

            if falsification_rate > 0.5:
                recommendations.append(
                    "High falsification rate detected. Consider revising APGI Framework parameters or methodology."
                )
            elif falsification_rate == 0:
                recommendations.append(
                    "No falsifications detected. Framework appears robust within tested parameter space."
                )
            else:
                recommendations.append(
                    f"Moderate falsification rate ({falsification_rate:.1%}). Further investigation recommended."
                )

        # Analyze statistical power
        all_powers = []
        for exp in experiments.values():
            results = exp.get("results", [])
            if results:
                mean_power = sum(r.statistical_power for r in results) / len(results)
                all_powers.append(mean_power)

        if all_powers:
            overall_power = sum(all_powers) / len(all_powers)

            if overall_power < 0.8:
                recommendations.append(
                    f"Statistical power is low ({overall_power:.3f}). Consider increasing sample sizes."
                )

        # Performance recommendations
        completed_experiments = [
            exp for exp in experiments.values() if exp["status"] == "completed"
        ]

        if len(completed_experiments) > 1:
            runtimes = []
            for exp in completed_experiments:
                start_time = exp.get("start_time")
                last_update = exp.get("last_update")

                if start_time and last_update:
                    if isinstance(start_time, str):
                        start_time = datetime.fromisoformat(start_time)
                    if isinstance(last_update, str):
                        last_update = datetime.fromisoformat(last_update)

                    runtime = (last_update - start_time).total_seconds()
                    runtimes.append(runtime)

            if runtimes and max(runtimes) > 2 * min(runtimes):
                recommendations.append(
                    "Significant performance variation detected between experiments. Consider optimizing slower configurations."
                )

        return recommendations


class DashboardServer:
    """Web-based dashboard server for experiment monitoring and analysis."""

    def __init__(self, port: int = 8080, data_dir: str = "data"):
        self.data: Dict[str, Any] = {}
        """
        Initialize dashboard server.

        Args:
            port: Server port
            data_dir: Directory for dashboard data storage
        """
        self.port = port
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.monitor = ExperimentMonitor()  # type: ignore
        self.comparator = ExperimentComparator()  # type: ignore
        self.visualizer = InteractiveVisualizer()  # type: ignore

        self.logger = logging.getLogger(__name__)

        # Dashboard state - unified data structure
        self.data = {
            "experiments": {},
            "comparisons": {},
            "visualizations": {},
            "summary_stats": {},
            "last_update": datetime.now().isoformat(),
        }

    def start_dashboard(self) -> None:
        """Start the dashboard server."""
        try:
            self.monitor.start_monitoring()
            self.monitor.add_listener(self._on_experiment_event)

            # In a real implementation, this would start a web server
            # For now, we'll simulate the dashboard functionality
            self.logger.info(f"Dashboard server started on port {self.port}")
            self._save_dashboard_state()

        except Exception as e:
            raise DashboardError(f"Failed to start dashboard server: {str(e)}")

    def stop_dashboard(self) -> None:
        """Stop the dashboard server."""
        try:
            self.monitor.stop_monitoring()
            self._save_dashboard_state()
            self.logger.info("Dashboard server stopped")

        except Exception as e:
            self.logger.error(f"Error stopping dashboard server: {str(e)}")

    def get_data(self, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current unified dashboard data."""
        # Update with latest experiment data
        experiments_data: Dict[str, Dict[str, Any]] = self.monitor.get_all_experiments()
        self.data["experiments"] = experiments_data
        self.data["last_update"] = datetime.now().isoformat()

        return self.data

    def get_dashboard_data(self, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Backward compatibility alias for get_data method."""
        return self.get_data(experiment_id)

    def get_experiment_visualization_data(self, experiment_id: str) -> Dict[str, Any]:
        """Get visualization data for specific experiment."""
        experiment = self.monitor.get_experiment_status(experiment_id)

        if not experiment:
            return {}

        try:
            results = experiment.get("results", [])
            trials = experiment.get("trials", [])

            if results and trials:
                viz_data = self.visualizer.create_interactive_dashboard_data(results, trials)
                # Store in unified data structure
                if "visualizations" not in self.data:
                    self.data["visualizations"] = {}
                self.data["visualizations"][experiment_id] = viz_data
                return viz_data
            else:
                return {"message": "No data available for visualization"}

        except Exception as e:
            self.logger.error(f"Error creating visualization data: {str(e)}")
            return {"error": str(e)}

    def export_dashboard_report(self, format: str = "json") -> str:
        """Export comprehensive dashboard report."""
        try:
            data = self.get_data()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if format == "json":
                filename = f"dashboard_report_{timestamp}.json"
                filepath = self.data_dir / filename

                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2, default=str)

            else:
                raise ValueError(f"Unsupported export format: {format}")

            self.logger.info(f"Exported dashboard report to {filepath}")
            return str(filepath)

        except Exception as e:
            raise DashboardError(f"Failed to export dashboard report: {str(e)}")

    def _on_experiment_event(self, event_type: str, experiment_id: str) -> None:
        """Handle experiment events."""
        self.logger.debug(f"Experiment event: {event_type} for {experiment_id}")

        # Update unified data structure
        self.data["last_update"] = datetime.now().isoformat()

        # Save state periodically
        if event_type in ["experiment_completed", "experiment_registered"]:
            self._save_dashboard_state()

    def _save_dashboard_state(self) -> None:
        """Save current dashboard state to disk."""
        try:
            state_file = self.data_dir / "dashboard_state.json"

            with open(state_file, "w") as f:
                json.dump(self.data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving dashboard state: {str(e)}")

    def _load_dashboard_state(self) -> None:
        """Load dashboard state from disk."""
        try:
            state_file = self.data_dir / "dashboard_state.json"

            if state_file.exists():
                with open(state_file, "r") as f:
                    self.data = json.load(f)

        except Exception as e:
            self.logger.error(f"Error loading dashboard state: {str(e)}")


# Convenience function for easy dashboard setup
def create_dashboard(port: int = 8080, data_dir: str = "data") -> DashboardServer:
    """Create and configure a dashboard server instance.

    Args:
        port: Server port
        data_dir: Data directory

    Returns:
        Configured dashboard server
    """
    return DashboardServer(port=port, data_dir=data_dir)
