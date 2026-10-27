"""Visualization System for APGI Framework.

This module provides publication-quality plotting and figure generation
for falsification test results and interactive visualization capabilities.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..core.data_models import (
    ExperimentalTrial,
    FalsificationResult,
    StatisticalSummary,
)
from ..exceptions import VisualizationError


class APGIVisualizer:
    """Generates publication-quality plots and figures for APGI Framework results."""

    def __init__(self, output_dir: str = "figures", style: str = "publication"):
        """Initialize the visualizer.

        Args:
            output_dir: Directory to save generated figures
            style: Plotting style ("publication", "presentation", "interactive")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Set up plotting style
        self._setup_style(style)

    def _setup_style(self, style: str) -> None:
        """Configure matplotlib and seaborn styles."""
        if style == "publication":
            plt.style.use("seaborn-v0_8-whitegrid")
            sns.set_palette("husl")
            plt.rcParams.update(
                {
                    "font.size": 12,
                    "axes.titlesize": 14,
                    "axes.labelsize": 12,
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "legend.fontsize": 10,
                    "figure.titlesize": 16,
                    "figure.dpi": 300,
                    "savefig.dpi": 300,
                    "savefig.bbox": "tight",
                }
            )

        elif style == "presentation":
            plt.style.use("seaborn-v0_8-darkgrid")
            sns.set_palette("bright")
            plt.rcParams.update(
                {
                    "font.size": 14,
                    "axes.titlesize": 18,
                    "axes.labelsize": 16,
                    "xtick.labelsize": 14,
                    "ytick.labelsize": 14,
                    "legend.fontsize": 14,
                    "figure.titlesize": 20,
                    "figure.dpi": 150,
                    "savefig.dpi": 150,
                }
            )

        else:  # interactive
            plt.style.use("default")
            sns.set_palette("tab10")

    def plot_falsification_summary(
        self,
        results: List[FalsificationResult],
        save_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Create summary plot of falsification test results.

        Args:
            results: List of falsification results
            save_path: Optional custom save path

        Returns:
            Path to saved figure
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Falsification status by test type
            test_types = [r.test_type for r in results]
            falsified = [r.is_falsified for r in results]

            df = pd.DataFrame({"Test Type": test_types, "Falsified": falsified})
            falsification_counts = (
                df.groupby(["Test Type", "Falsified"]).size().unstack(fill_value=0)
            )

            falsification_counts.plot(kind="bar", ax=ax1, color=["lightcoral", "lightgreen"])
            ax1.set_title("Falsification Results by Test Type")
            ax1.set_xlabel("Test Type")
            ax1.set_ylabel("Number of Tests")
            ax1.legend(["Not Falsified", "Falsified"])
            ax1.tick_params(axis="x", rotation=45)

            # 2. Effect size distribution
            effect_sizes = [r.effect_size for r in results]
            ax2.hist(effect_sizes, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
            ax2.axvline(
                np.mean(effect_sizes),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(effect_sizes):.3f}",
            )
            ax2.set_title("Distribution of Effect Sizes")
            ax2.set_xlabel("Effect Size (Cohen's d)")
            ax2.set_ylabel("Frequency")
            ax2.legend()

            # 3. P-value vs Effect size scatter
            p_values = [r.p_value for r in results]
            colors = ["red" if f else "blue" for f in falsified]

            ax3.scatter(effect_sizes, p_values, c=colors, alpha=0.6)
            ax3.axhline(0.05, color="gray", linestyle="--", alpha=0.5, label="α = 0.05")
            ax3.set_title("Effect Size vs P-Value")
            ax3.set_xlabel("Effect Size (Cohen's d)")
            ax3.set_ylabel("P-Value")
            ax3.set_yscale("log")
            ax3.legend(["α = 0.05", "Not Falsified", "Falsified"])

            # 4. Statistical power by test type
            powers = [r.statistical_power for r in results]
            df_power = pd.DataFrame({"Test Type": test_types, "Power": powers})

            sns.boxplot(data=df_power, x="Test Type", y="Power", ax=ax4)
            ax4.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="Power = 0.8")
            ax4.set_title("Statistical Power by Test Type")
            ax4.set_xlabel("Test Type")
            ax4.set_ylabel("Statistical Power")
            ax4.tick_params(axis="x", rotation=45)
            ax4.legend()

            plt.tight_layout()

            # Save figure
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.output_dir / f"falsification_summary_{timestamp}.png"

            plt.savefig(save_path)
            plt.close()

            self.logger.info(f"Saved falsification summary plot to {save_path}")
            return str(save_path)

        except Exception as e:
            raise VisualizationError(f"Failed to create falsification summary plot: {str(e)}")

    def plot_neural_signatures(
        self,
        trials: List[ExperimentalTrial],
        save_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Create comprehensive neural signature visualization.

        Args:
            trials: List of experimental trials
            save_path: Optional custom save path

        Returns:
            Path to saved figure
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # Extract neural signature data
            p3b_amplitudes = [t.neural_signatures.p3b_amplitude for t in trials]
            gamma_plvs = [t.neural_signatures.gamma_plv for t in trials]
            pci_values = [t.neural_signatures.pci_value for t in trials]
            consciousness = [t.consciousness_assessment.subjective_report for t in trials]

            # 1. P3b amplitude distribution
            conscious_p3b = [p for p, c in zip(p3b_amplitudes, consciousness) if c]
            unconscious_p3b = [p for p, c in zip(p3b_amplitudes, consciousness) if not c]

            ax1.hist(conscious_p3b, bins=20, alpha=0.7, label="Conscious", color="green")
            ax1.hist(unconscious_p3b, bins=20, alpha=0.7, label="Unconscious", color="red")
            ax1.axvline(5.0, color="black", linestyle="--", label="Threshold (5 μV)")
            ax1.set_title("P3b Amplitude Distribution")
            ax1.set_xlabel("P3b Amplitude (μV)")
            ax1.set_ylabel("Frequency")
            ax1.legend()

            # 2. Gamma PLV distribution
            conscious_gamma = [g for g, c in zip(gamma_plvs, consciousness) if c]
            unconscious_gamma = [g for g, c in zip(gamma_plvs, consciousness) if not c]

            ax2.hist(conscious_gamma, bins=20, alpha=0.7, label="Conscious", color="green")
            ax2.hist(unconscious_gamma, bins=20, alpha=0.7, label="Unconscious", color="red")
            ax2.axvline(0.3, color="black", linestyle="--", label="Threshold (0.3)")
            ax2.set_title("Gamma-band PLV Distribution")
            ax2.set_xlabel("Phase-Locking Value")
            ax2.set_ylabel("Frequency")
            ax2.legend()

            # 3. PCI distribution
            conscious_pci = [p for p, c in zip(pci_values, consciousness) if c]
            unconscious_pci = [p for p, c in zip(pci_values, consciousness) if not c]

            ax3.hist(conscious_pci, bins=20, alpha=0.7, label="Conscious", color="green")
            ax3.hist(unconscious_pci, bins=20, alpha=0.7, label="Unconscious", color="red")
            ax3.axvline(0.4, color="black", linestyle="--", label="Threshold (0.4)")
            ax3.set_title("PCI Distribution")
            ax3.set_xlabel("Perturbational Complexity Index")
            ax3.set_ylabel("Frequency")
            ax3.legend()

            # 4. Neural signature correlation matrix
            signature_data = pd.DataFrame(
                {
                    "P3b Amplitude": p3b_amplitudes,
                    "Gamma PLV": gamma_plvs,
                    "PCI": pci_values,
                    "Consciousness": [1 if c else 0 for c in consciousness],
                }
            )

            correlation_matrix = signature_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, ax=ax4)
            ax4.set_title("Neural Signature Correlations")

            plt.tight_layout()

            # Save figure
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.output_dir / f"neural_signatures_{timestamp}.png"

            plt.savefig(save_path)
            plt.close()

            self.logger.info(f"Saved neural signatures plot to {save_path}")
            return str(save_path)

        except Exception as e:
            raise VisualizationError(f"Failed to create neural signatures plot: {str(e)}")

    def create_neural_signature_plot(
        self,
        trials: List[ExperimentalTrial],
        save_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Create neural signature visualization (alias for plot_neural_signatures).

        Args:
            trials: List of experimental trials
            save_path: Optional custom save path

        Returns:
            Path to saved figure
        """
        return self.plot_neural_signatures(trials, save_path)

    def plot_apgi_parameter_space(
        self,
        trials: List[ExperimentalTrial],
        save_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Visualize APGI parameter space and ignition patterns.

        Args:
            trials: List of experimental trials
            save_path: Optional custom save path

        Returns:
            Path to saved figure
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # Extract parameter data
            extero_precisions = [t.apgi_parameters.extero_precision for t in trials]
            intero_precisions = [t.apgi_parameters.intero_precision for t in trials]
            thresholds = [t.apgi_parameters.threshold for t in trials]
            somatic_gains = [t.apgi_parameters.somatic_gain for t in trials]
            consciousness = [t.consciousness_assessment.subjective_report for t in trials]

            # 1. Precision space (extero vs intero)
            colors = ["green" if c else "red" for c in consciousness]
            ax1.scatter(extero_precisions, intero_precisions, c=colors, alpha=0.6)
            ax1.set_title("Precision Parameter Space")
            ax1.set_xlabel("Exteroceptive Precision (Πₑ)")
            ax1.set_ylabel("Interoceptive Precision (Πᵢ)")

            # Add legend
            conscious_patch = patches.Patch(color="green", label="Conscious")
            unconscious_patch = patches.Patch(color="red", label="Unconscious")
            ax1.legend(handles=[conscious_patch, unconscious_patch])

            # 2. Threshold vs Somatic Gain
            ax2.scatter(thresholds, somatic_gains, c=colors, alpha=0.6)
            ax2.set_title("Threshold vs Somatic Marker Gain")
            ax2.set_xlabel("Ignition Threshold (θₜ)")
            ax2.set_ylabel("Somatic Marker Gain (M_{c,a})")
            ax2.legend(handles=[conscious_patch, unconscious_patch])

            # 3. Threshold distribution by consciousness
            conscious_thresholds = [t for t, c in zip(thresholds, consciousness) if c]
            unconscious_thresholds = [t for t, c in zip(thresholds, consciousness) if not c]

            ax3.hist(
                conscious_thresholds,
                bins=20,
                alpha=0.7,
                label="Conscious",
                color="green",
            )
            ax3.hist(
                unconscious_thresholds,
                bins=20,
                alpha=0.7,
                label="Unconscious",
                color="red",
            )
            ax3.set_title("Ignition Threshold Distribution")
            ax3.set_xlabel("Ignition Threshold (θₜ)")
            ax3.set_ylabel("Frequency")
            ax3.legend()

            # 4. Parameter correlation heatmap
            param_data = pd.DataFrame(
                {
                    "Extero Precision": extero_precisions,
                    "Intero Precision": intero_precisions,
                    "Threshold": thresholds,
                    "Somatic Gain": somatic_gains,
                    "Consciousness": [1 if c else 0 for c in consciousness],
                }
            )

            correlation_matrix = param_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, ax=ax4)
            ax4.set_title("APGI Parameter Correlations")

            plt.tight_layout()

            # Save figure
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.output_dir / f"apgi_parameter_space_{timestamp}.png"

            plt.savefig(save_path)
            plt.close()

            self.logger.info(f"Saved APGI parameter space plot to {save_path}")
            return str(save_path)

        except Exception as e:
            raise VisualizationError(f"Failed to create APGI parameter space plot: {str(e)}")

    def plot_statistical_summary(
        self,
        statistical_summary: StatisticalSummary,
        save_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Create statistical summary visualization.

        Args:
            statistical_summary: Statistical analysis summary
            save_path: Optional custom save path

        Returns:
            Path to saved figure
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # 1. Power analysis visualization
            effect_sizes = np.linspace(0, 2, 100)
            sample_sizes = [20, 50, 100, 200]

            for n in sample_sizes:
                # Simplified power calculation for visualization
                powers = 1 - np.exp(-(effect_sizes**2) * n / 16)
                ax1.plot(effect_sizes, powers, label=f"n = {n}")

            ax1.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="Power = 0.8")
            ax1.axvline(
                statistical_summary.mean_effect_size,
                color="black",
                linestyle=":",
                label=f"Observed d = {statistical_summary.mean_effect_size:.3f}",
            )
            ax1.set_title("Statistical Power Analysis")
            ax1.set_xlabel("Effect Size (Cohen's d)")
            ax1.set_ylabel("Statistical Power")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Confidence intervals
            metrics = ["Effect Size", "Statistical Power", "Replication Rate"]
            values = [
                statistical_summary.mean_effect_size,
                statistical_summary.statistical_power,
                statistical_summary.replication_success_rate,
            ]

            # Simulate confidence intervals (in real implementation, use actual CIs)
            errors = [v * 0.1 for v in values]  # 10% error bars for illustration

            ax2.bar(metrics, values, yerr=errors, capsize=5, alpha=0.7, color="lightblue")
            ax2.set_title("Key Statistical Metrics")
            ax2.set_ylabel("Value")
            ax2.tick_params(axis="x", rotation=45)

            # 3. Sample size adequacy
            categories = [
                "Adequate Power\n(≥0.8)",
                "Moderate Power\n(0.5-0.8)",
                "Low Power\n(<0.5)",
            ]

            if statistical_summary.statistical_power >= 0.8:
                counts = [1, 0, 0]
                colors = ["green", "lightgray", "lightgray"]
            elif statistical_summary.statistical_power >= 0.5:
                counts = [0, 1, 0]
                colors = ["lightgray", "orange", "lightgray"]
            else:
                counts = [0, 0, 1]
                colors = ["lightgray", "lightgray", "red"]

            ax3.bar(categories, counts, color=colors)
            ax3.set_title("Statistical Power Assessment")
            ax3.set_ylabel("Current Study")
            ax3.set_ylim(0, 1.2)

            # 4. Replication analysis
            replication_data = {
                "Successful": statistical_summary.replication_success_rate,
                "Failed": 1 - statistical_summary.replication_success_rate,
            }

            colors = ["lightgreen", "lightcoral"]
            ax4.pie(
                replication_data.values(),
                labels=replication_data.keys(),
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax4.set_title("Replication Success Rate")

            plt.tight_layout()

            # Save figure
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.output_dir / f"statistical_summary_{timestamp}.png"

            plt.savefig(save_path)
            plt.close()

            self.logger.info(f"Saved statistical summary plot to {save_path}")
            return str(save_path)

        except Exception as e:
            raise VisualizationError(f"Failed to create statistical summary plot: {str(e)}")

    def create_publication_figure_set(
        self,
        results: List[FalsificationResult],
        trials: List[ExperimentalTrial],
        statistical_summary: StatisticalSummary,
        experiment_id: str,
    ) -> List[str]:
        """Create complete set of publication-ready figures.

        Args:
            results: Falsification results
            trials: Experimental trials
            statistical_summary: Statistical summary
            experiment_id: Experiment identifier

        Returns:
            List of paths to generated figures
        """
        figure_paths = []

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Generate all figures with consistent naming
            figures = [
                (
                    self.plot_falsification_summary,
                    results,
                    f"fig1_falsification_summary_{experiment_id}_{timestamp}.png",
                ),
                (
                    self.plot_neural_signatures,
                    trials,
                    f"fig2_neural_signatures_{experiment_id}_{timestamp}.png",
                ),
                (
                    self.plot_apgi_parameter_space,
                    trials,
                    f"fig3_parameter_space_{experiment_id}_{timestamp}.png",
                ),
                (
                    self.plot_statistical_summary,
                    statistical_summary,
                    f"fig4_statistical_summary_{experiment_id}_{timestamp}.png",
                ),
            ]

            for plot_func, data, filename in figures:
                save_path = self.output_dir / filename
                plot_func = cast(Callable[[Any, Optional[Union[str, Path]]], str], plot_func)
                figure_path = plot_func(data, save_path)
                figure_paths.append(figure_path)

            self.logger.info(f"Generated complete figure set for experiment {experiment_id}")
            return figure_paths

        except Exception as e:
            raise VisualizationError(f"Failed to create publication figure set: {str(e)}")


class InteractiveVisualizer:
    """Provides interactive visualization capabilities for real-time analysis."""

    def __init__(self, output_dir: str = "figures") -> None:
        """Initialize interactive visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def create_interactive_dashboard_data(
        self, results: List[FalsificationResult], trials: List[ExperimentalTrial]
    ) -> Dict[str, Any]:
        """Prepare data for interactive dashboard visualization.

        Args:
            results: Falsification results
            trials: Experimental trials

        Returns:
            Dictionary containing structured data for dashboard
        """
        try:
            data = {
                "summary_stats": {
                    "total_tests": len(results),
                    "falsified_tests": sum(1 for r in results if r.is_falsified),
                    "mean_effect_size": np.mean([r.effect_size for r in results]),
                    "mean_p_value": np.mean([r.p_value for r in results]),
                    "mean_power": np.mean([r.statistical_power for r in results]),
                },
                "test_results": [
                    {
                        "test_type": r.test_type,
                        "is_falsified": r.is_falsified,
                        "effect_size": r.effect_size,
                        "p_value": r.p_value,
                        "statistical_power": r.statistical_power,
                        "confidence_level": r.confidence_level,
                    }
                    for r in results
                ],
                "trial_data": [
                    {
                        "trial_id": t.trial_id,
                        "condition": t.condition,
                        "consciousness": t.consciousness_assessment.subjective_report,
                        "p3b_amplitude": t.neural_signatures.p3b_amplitude,
                        "gamma_plv": t.neural_signatures.gamma_plv,
                        "pci_value": t.neural_signatures.pci_value,
                        "extero_precision": t.apgi_parameters.extero_precision,
                        "intero_precision": t.apgi_parameters.intero_precision,
                        "threshold": t.apgi_parameters.threshold,
                    }
                    for t in trials
                ],
                "timestamp": datetime.now().isoformat(),
            }

            return data

        except Exception as e:
            raise VisualizationError(f"Failed to create interactive dashboard data: {str(e)}")

    def plot_self_model(
        self,
        trials: List[ExperimentalTrial],
        save_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Visualize the self-model dynamics."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            intero_precisions = [t.apgi_parameters.intero_precision for t in trials]
            somatic_gains = [t.apgi_parameters.somatic_gain for t in trials]
            consciousness = [t.consciousness_assessment.subjective_report for t in trials]

            # Plot interoceptive precision vs somatic gain
            colors = ["green" if c else "red" for c in consciousness]
            ax1.scatter(intero_precisions, somatic_gains, c=colors, alpha=0.6)
            ax1.set_title("Self-Model: Interoceptive Precision vs Somatic Gain")
            ax1.set_xlabel("Interoceptive Precision (Πᵢ)")
            ax1.set_ylabel("Somatic Gain (M_{c,a})")

            # Plot distribution of somatic influence
            somatic_influence = [i * g for i, g in zip(intero_precisions, somatic_gains)]
            sns.kdeplot(somatic_influence, ax=ax2, fill=True)
            ax2.set_title("Distribution of Somatic Influence (Πᵢ * M_{c,a})")
            ax2.set_xlabel("Influence Strength")

            plt.tight_layout()
            if not save_path:
                save_path = (
                    self.output_dir / f"self_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
            plt.savefig(save_path)
            plt.close()
            return str(save_path)
        except Exception as e:
            raise VisualizationError(f"Failed to create self-model plot: {str(e)}")

    def plot_oscillations(
        self,
        trials: List[ExperimentalTrial],
        save_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Visualize oscillatory dynamics across trials."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            gamma_plvs = [t.neural_signatures.gamma_plv for t in trials]
            trial_indices = range(len(trials))

            ax.plot(trial_indices, gamma_plvs, label="Gamma PLV", color="purple")
            ax.set_title("Oscillatory Dynamics Across Trials")
            ax.set_xlabel("Trial Index")
            ax.set_ylabel("Gamma Phase-Locking Value")
            ax.legend()

            plt.tight_layout()
            if not save_path:
                save_path = (
                    self.output_dir / f"oscillations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
            plt.savefig(save_path)
            plt.close()
            return str(save_path)
        except Exception as e:
            raise VisualizationError(f"Failed to create oscillations plot: {str(e)}")

    def plot_3d_state_space(
        self,
        trials: List[ExperimentalTrial],
        save_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Visualize APGI state space in 3D."""
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            extero = [t.apgi_parameters.extero_precision for t in trials]
            intero = [t.apgi_parameters.intero_precision for t in trials]
            threshold = [t.apgi_parameters.threshold for t in trials]
            consciousness = [t.consciousness_assessment.subjective_report for t in trials]

            colors = ["green" if c else "red" for c in consciousness]
            ax.scatter(extero, intero, threshold, c=colors, alpha=0.6)

            ax.set_title("3D APGI State Space")
            ax.set_xlabel("Extero Precision")
            ax.set_ylabel("Intero Precision")
            ax.set_zlabel("Threshold")

            plt.tight_layout()
            if not save_path:
                save_path = (
                    self.output_dir
                    / f"state_space_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
            plt.savefig(save_path)
            plt.close()
            return str(save_path)
        except Exception as e:
            raise VisualizationError(f"Failed to create 3D state space plot: {str(e)}")
