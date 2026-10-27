"""Parameter recovery validation system for APGI framework.

This module implements synthetic data generation and parameter recovery
validation to ensure the Bayesian modeling pipeline can accurately recover
known ground-truth parameters.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..logging.standardized_logging import get_logger
from ..utils.progress_monitor import ProgressMonitor
from .bayesian_models import ParameterEstimates
from .parameter_estimation import JointParameterFitter


@dataclass
class GroundTruthParameters:
    """Ground truth parameters for synthetic data generation."""

    theta0: float  # Baseline ignition threshold
    pi_i: float  # Interoceptive precision
    beta: float  # Somatic bias

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {"theta0": self.theta0, "pi_i": self.pi_i, "beta": self.beta}


@dataclass
class RecoveryMetrics:
    """Metrics for parameter recovery assessment."""

    correlation: float  # Pearson correlation between true and recovered
    bias: float  # Mean bias (recovered - true)
    rmse: float  # Root mean squared error
    mae: float  # Mean absolute error
    within_ci_95: float  # Proportion of true values within 95% CI

    def __repr__(self) -> str:
        return (
            f"RecoveryMetrics(r={self.correlation:.3f}, "
            f"bias={self.bias:.3f}, RMSE={self.rmse:.3f})"
        )


@dataclass
class RecoveryResults:
    """Complete results from parameter recovery study."""

    n_datasets: int
    ground_truth: List[GroundTruthParameters]
    recovered: List[ParameterEstimates]
    metrics: Dict[str, RecoveryMetrics]  # Metrics for each parameter
    passed: bool  # Whether recovery validation passed
    validation_criteria: Dict[str, float]  # Criteria used for validation

    def summary(self) -> str:
        """Generate summary string."""
        status = "PASSED" if self.passed else "FAILED"
        summary = f"Parameter Recovery Validation: {status}\n"
        summary += f"Datasets: {self.n_datasets}\n\n"

        for param_name, metrics in self.metrics.items():
            summary += f"{param_name}:\n"
            summary += f"  Correlation: {metrics.correlation:.3f}\n"
            summary += f"  Bias: {metrics.bias:.3f}\n"
            summary += f"  RMSE: {metrics.rmse:.3f}\n"
            summary += f"  95% CI Coverage: {metrics.within_ci_95:.1%}\n\n"

        return summary


class SyntheticDataGenerator:
    """Generates synthetic behavioral and neural data with known ground-truth parameters."""

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize synthetic data generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(random_seed)
        self.logger = get_logger(__name__)

    def generate_detection_task_data(
        self, theta0: float, n_trials: int = 200, noise_level: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """Generate synthetic detection task data.

        Args:
            theta0: Ground truth baseline ignition threshold
            n_trials: Number of trials
            noise_level: Noise level for neural data

        Returns:
            Dictionary with detection task data
        """
        # Generate stimulus intensities (adaptive staircase simulation)
        stimulus_intensity = self.rng.uniform(-2, 2, n_trials)

        # Generate detection responses based on psychometric function
        detection_prob = 1 / (1 + np.exp(-theta0 * stimulus_intensity))
        detected = self.rng.binomial(1, detection_prob)

        # Generate P3b amplitudes (inversely related to threshold)
        p3b_amplitude = -theta0 + self.rng.normal(0, noise_level, n_trials)

        # Add trial-level noise for detected trials
        p3b_amplitude[detected == 1] += self.rng.normal(0, noise_level * 0.5, np.sum(detected))

        return {
            "subject_id": np.ones(n_trials, dtype=int),
            "stimulus_intensity": stimulus_intensity,
            "detected": detected,
            "p3b_amplitude": p3b_amplitude,
        }

    def generate_heartbeat_task_data(
        self, pi_i: float, n_trials: int = 60, noise_level: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """Generate synthetic heartbeat detection task data.

        Args:
            pi_i: Ground truth interoceptive precision
            n_trials: Number of trials
            noise_level: Noise level for neural data

        Returns:
            Dictionary with heartbeat task data
        """
        # Generate synchronous/asynchronous trials
        synchronous = self.rng.binomial(1, 0.5, n_trials)

        # Generate responses based on d' (interoceptive precision)
        d_prime = pi_i
        prob_correct = stats.norm.cdf(d_prime / 2)

        response_sync = np.zeros(n_trials, dtype=int)
        for i in range(n_trials):
            if synchronous[i] == 1:
                response_sync[i] = self.rng.binomial(1, prob_correct)
            else:
                response_sync[i] = self.rng.binomial(1, 1 - prob_correct)

        # Generate confidence ratings (higher for correct responses)
        confidence = np.zeros(n_trials)
        correct = response_sync == synchronous
        confidence[correct] = self.rng.uniform(0.6, 1.0, np.sum(correct))
        confidence[~correct] = self.rng.uniform(0.0, 0.6, np.sum(~correct))

        # Generate HEP amplitudes (related to interoceptive precision)
        hep_amplitude = pi_i + self.rng.normal(0, noise_level, n_trials)

        # Generate pupil responses (related to interoceptive precision)
        pupil_response = pi_i + self.rng.normal(0, noise_level, n_trials)

        return {
            "subject_id": np.ones(n_trials, dtype=int),
            "synchronous": synchronous,
            "response_sync": response_sync,
            "confidence": confidence,
            "hep_amplitude": hep_amplitude,
            "pupil_response": pupil_response,
        }

    def generate_oddball_task_data(
        self, beta: float, n_trials: int = 120, noise_level: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """Generate synthetic oddball task data.

        Args:
            beta: Ground truth somatic bias
            n_trials: Number of trials
            noise_level: Noise level for neural data

        Returns:
            Dictionary with oddball task data
        """
        # Generate trial types (0=standard, 1=intero deviant, 2=extero deviant)
        n_deviants = n_trials // 6  # ~17% deviants each
        trial_type = np.concatenate(
            [
                np.zeros(n_trials - 2 * n_deviants, dtype=int),
                np.ones(n_deviants, dtype=int),
                np.full(n_deviants, 2, dtype=int),
            ]
        )
        self.rng.shuffle(trial_type)

        # Generate P3b amplitudes
        p3b_intero = np.zeros(n_trials)
        p3b_extero = np.zeros(n_trials)

        # Interoceptive deviants: P3b scaled by beta
        intero_mask = trial_type == 1
        p3b_intero[intero_mask] = beta + self.rng.normal(0, noise_level, np.sum(intero_mask))

        # Exteroceptive deviants: P3b baseline
        extero_mask = trial_type == 2
        p3b_extero[extero_mask] = 1.0 + self.rng.normal(0, noise_level, np.sum(extero_mask))

        # Standards: minimal P3b
        standard_mask = trial_type == 0
        p3b_intero[standard_mask] = self.rng.normal(0, noise_level * 0.5, np.sum(standard_mask))
        p3b_extero[standard_mask] = self.rng.normal(0, noise_level * 0.5, np.sum(standard_mask))

        return {
            "subject_id": np.ones(n_trials, dtype=int),
            "trial_type": trial_type,
            "p3b_intero": p3b_intero,
            "p3b_extero": p3b_extero,
        }

    def generate_complete_dataset(
        self,
        ground_truth: GroundTruthParameters,
        n_trials_detection: int = 200,
        n_trials_heartbeat: int = 60,
        n_trials_oddball: int = 120,
        noise_level: float = 0.1,
    ) -> Tuple[Dict, Dict, Dict]:
        """Generate complete synthetic dataset for one subject.

        Args:
            ground_truth: Ground truth parameters
            n_trials_detection: Number of detection trials
            n_trials_heartbeat: Number of heartbeat trials
            n_trials_oddball: Number of oddball trials
            noise_level: Noise level for neural data

        Returns:
            Tuple of (detection_data, heartbeat_data, oddball_data)
        """
        detection_data = self.generate_detection_task_data(
            ground_truth.theta0, n_trials_detection, noise_level
        )

        heartbeat_data = self.generate_heartbeat_task_data(
            ground_truth.pi_i, n_trials_heartbeat, noise_level
        )

        oddball_data = self.generate_oddball_task_data(
            ground_truth.beta, n_trials_oddball, noise_level
        )

        return detection_data, heartbeat_data, oddball_data

    def generate_multiple_datasets(
        self,
        n_datasets: int,
        theta0_range: Tuple[float, float] = (-1.0, 1.0),
        pi_i_range: Tuple[float, float] = (0.5, 2.0),
        beta_range: Tuple[float, float] = (0.5, 1.5),
        noise_level: float = 0.1,
    ) -> List[Tuple[GroundTruthParameters, Dict, Dict, Dict]]:
        """Generate multiple synthetic datasets with varying parameters.

        Args:
            n_datasets: Number of datasets to generate
            theta0_range: Range for theta0 sampling
            pi_i_range: Range for pi_i sampling
            beta_range: Range for beta sampling
            noise_level: Noise level for neural data

        Returns:
            List of (ground_truth, detection_data, heartbeat_data, oddball_data)
        """
        datasets = []

        # Progress tracking for dataset generation
        progress = ProgressMonitor(n_datasets, "Generating synthetic datasets")
        progress.start()

        for i in range(n_datasets):
            # Sample ground truth parameters
            theta0 = self.rng.uniform(*theta0_range)
            pi_i = self.rng.uniform(*pi_i_range)
            beta = self.rng.uniform(*beta_range)

            ground_truth = GroundTruthParameters(theta0, pi_i, beta)

            # Generate data
            detection, heartbeat, oddball = self.generate_complete_dataset(
                ground_truth, noise_level=noise_level
            )

            datasets.append((ground_truth, detection, heartbeat, oddball))
            progress.update(message=f"Dataset {i + 1}/{n_datasets}")

        progress.complete()
        return datasets


class RecoveryAnalyzer:
    """Analyzes parameter recovery with correlation metrics and bias assessment."""

    @staticmethod
    def compute_recovery_metrics(
        true_values: np.ndarray,
        recovered_means: np.ndarray,
        recovered_ci_lower: np.ndarray,
        recovered_ci_upper: np.ndarray,
    ) -> RecoveryMetrics:
        """Compute recovery metrics for a single parameter.

        Args:
            true_values: Ground truth parameter values
            recovered_means: Recovered parameter means
            recovered_ci_lower: Lower bounds of 95% CI
            recovered_ci_upper: Upper bounds of 95% CI

        Returns:
            RecoveryMetrics object
        """
        # Correlation
        correlation = float(np.corrcoef(true_values, recovered_means)[0, 1])

        # Bias
        bias = float(np.mean(recovered_means - true_values))

        # RMSE
        rmse = float(np.sqrt(np.mean((recovered_means - true_values) ** 2)))

        # MAE
        mae = float(np.mean(np.abs(recovered_means - true_values)))

        # Coverage (proportion within 95% CI)
        within_ci = np.logical_and(
            true_values >= recovered_ci_lower, true_values <= recovered_ci_upper
        )
        within_ci_95 = float(np.mean(within_ci))

        return RecoveryMetrics(
            correlation=correlation,
            bias=bias,
            rmse=rmse,
            mae=mae,
            within_ci_95=within_ci_95,
        )

    @staticmethod
    def analyze_recovery(
        ground_truth_list: List[GroundTruthParameters],
        recovered_list: List[ParameterEstimates],
    ) -> Dict[str, RecoveryMetrics]:
        """Analyze parameter recovery across all datasets.

        Args:
            ground_truth_list: List of ground truth parameters
            recovered_list: List of recovered parameter estimates

        Returns:
            Dictionary mapping parameter name to RecoveryMetrics
        """
        # Extract arrays
        true_theta0 = np.array([gt.theta0 for gt in ground_truth_list])
        true_pi_i = np.array([gt.pi_i for gt in ground_truth_list])
        true_beta = np.array([gt.beta for gt in ground_truth_list])

        rec_theta0 = np.array([r.theta0.mean for r in recovered_list])
        rec_pi_i = np.array([r.pi_i.mean for r in recovered_list])
        rec_beta = np.array([r.beta.mean for r in recovered_list])

        rec_theta0_ci_lower = np.array([r.theta0.credible_interval_95[0] for r in recovered_list])
        rec_theta0_ci_upper = np.array([r.theta0.credible_interval_95[1] for r in recovered_list])

        rec_pi_i_ci_lower = np.array([r.pi_i.credible_interval_95[0] for r in recovered_list])
        rec_pi_i_ci_upper = np.array([r.pi_i.credible_interval_95[1] for r in recovered_list])

        rec_beta_ci_lower = np.array([r.beta.credible_interval_95[0] for r in recovered_list])
        rec_beta_ci_upper = np.array([r.beta.credible_interval_95[1] for r in recovered_list])

        # Compute metrics for each parameter
        metrics = {
            "theta0": RecoveryAnalyzer.compute_recovery_metrics(
                true_theta0, rec_theta0, rec_theta0_ci_lower, rec_theta0_ci_upper
            ),
            "pi_i": RecoveryAnalyzer.compute_recovery_metrics(
                true_pi_i, rec_pi_i, rec_pi_i_ci_lower, rec_pi_i_ci_upper
            ),
            "beta": RecoveryAnalyzer.compute_recovery_metrics(
                true_beta, rec_beta, rec_beta_ci_lower, rec_beta_ci_upper
            ),
        }

        return metrics

    @staticmethod
    def plot_recovery(
        ground_truth_list: List[GroundTruthParameters],
        recovered_list: List[ParameterEstimates],
        save_path: Optional[str] = None,
    ) -> None:
        """Create recovery plots for all parameters.

        Args:
            ground_truth_list: List of ground truth parameters
            recovered_list: List of recovered parameter estimates
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        params = ["theta0", "pi_i", "beta"]
        param_labels = [
            "θ₀ (Threshold)",
            "Πᵢ (Interoceptive Precision)",
            "β (Somatic Bias)",
        ]

        for idx, (param, label) in enumerate(zip(params, param_labels)):
            ax = axes[idx]

            # Extract values
            true_vals = np.array([getattr(gt, param) for gt in ground_truth_list])
            rec_vals = np.array([getattr(r, param).mean for r in recovered_list])
            rec_ci_lower = np.array(
                [getattr(r, param).credible_interval_95[0] for r in recovered_list]
            )
            rec_ci_upper = np.array(
                [getattr(r, param).credible_interval_95[1] for r in recovered_list]
            )

            # Scatter plot with error bars
            ax.errorbar(
                true_vals,
                rec_vals,
                yerr=[rec_vals - rec_ci_lower, rec_ci_upper - rec_vals],
                fmt="o",
                alpha=0.6,
                capsize=3,
            )

            # Identity line
            min_val = min(true_vals.min(), rec_vals.min())
            max_val = max(true_vals.max(), rec_vals.max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                alpha=0.5,
                label="Perfect Recovery",
            )

            # Compute correlation
            r = np.corrcoef(true_vals, rec_vals)[0, 1]

            ax.set_xlabel(f"True {label}")
            ax.set_ylabel(f"Recovered {label}")
            ax.set_title(f"{label}\nr = {r:.3f}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")


class ParameterRecoveryValidator:
    """Validates parameter recovery through simulation of synthetic datasets."""

    def __init__(
        self,
        validation_criteria: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None,
    ):
        """Initialize parameter recovery validator.

        Args:
            validation_criteria: Minimum correlation thresholds for each parameter
            random_seed: Random seed for reproducibility
        """
        if validation_criteria is None:
            validation_criteria = {
                "theta0": 0.85,  # r > 0.85 for θ₀
                "pi_i": 0.75,  # r > 0.75 for Πᵢ
                "beta": 0.85,  # r > 0.85 for β
            }

        self.validation_criteria = validation_criteria
        self.generator = SyntheticDataGenerator(random_seed)
        self.analyzer = RecoveryAnalyzer()
        self.logger = get_logger(__name__)

    def run_validation(
        self,
        n_datasets: int = 100,
        noise_level: float = 0.1,
        chains: int = 4,
        iter: int = 1000,
        verbose: bool = True,
    ) -> RecoveryResults:
        """Run complete parameter recovery validation.

        Args:
            n_datasets: Number of synthetic datasets to generate
            noise_level: Noise level for synthetic data
            chains: Number of MCMC chains
            iter: Number of iterations per chain
            verbose: Print progress

        Returns:
            RecoveryResults with validation outcome
        """
        if verbose:
            self.logger.info(f"Generating {n_datasets} synthetic datasets...")

        # Generate synthetic datasets
        datasets = self.generator.generate_multiple_datasets(n_datasets, noise_level=noise_level)

        ground_truth_list = []
        recovered_list = []

        # Fit each dataset
        fitter = JointParameterFitter()

        for i, (ground_truth, detection, heartbeat, oddball) in enumerate(datasets):
            if verbose and (i + 1) % 10 == 0:
                self.logger.info(f"Processing dataset {i + 1}/{n_datasets}...")

            try:
                # Fit model
                fit_results = fitter.fit_all_subjects(
                    detection,
                    heartbeat,
                    oddball,
                    participant_ids=[f"synth_{i}"],
                    session_ids=[f"session_{i}"],
                    chains=chains,
                    iter=iter,
                    verbose=False,
                )

                ground_truth_list.append(ground_truth)
                recovered_list.append(fit_results.parameter_estimates[0])

            except Exception as e:
                if verbose:
                    self.logger.warning(f"Warning: Failed to fit dataset {i}: {e}")
                continue

        if verbose:
            self.logger.info(f"Successfully fit {len(recovered_list)}/{n_datasets} datasets")
            self.logger.info("Analyzing parameter recovery...")

        # Analyze recovery
        metrics = self.analyzer.analyze_recovery(ground_truth_list, recovered_list)

        # Check validation criteria
        passed = all(
            metrics[param].correlation >= threshold
            for param, threshold in self.validation_criteria.items()
        )

        results = RecoveryResults(
            n_datasets=len(recovered_list),
            ground_truth=ground_truth_list,
            recovered=recovered_list,
            metrics=metrics,
            passed=passed,
            validation_criteria=self.validation_criteria,
        )

        if verbose:
            self.logger.info("\n" + results.summary())

        return results


class ValidationReportGenerator:
    """Generates comprehensive validation reports with pass/fail criteria."""

    @staticmethod
    def generate_report(
        recovery_results: RecoveryResults, output_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive validation report.

        Args:
            recovery_results: Results from parameter recovery validation
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        report = []
        report.append("=" * 70)
        report.append("PARAMETER RECOVERY VALIDATION REPORT")
        report.append("=" * 70)
        report.append("")

        # Overall status
        status = "✓ PASSED" if recovery_results.passed else "✗ FAILED"
        report.append(f"Validation Status: {status}")
        report.append(f"Number of Datasets: {recovery_results.n_datasets}")
        report.append("")

        # Detailed metrics for each parameter
        report.append("-" * 70)
        report.append("PARAMETER RECOVERY METRICS")
        report.append("-" * 70)
        report.append("")

        for param_name, metrics in recovery_results.metrics.items():
            threshold = recovery_results.validation_criteria[param_name]
            passed = metrics.correlation >= threshold
            status_symbol = "✓" if passed else "✗"

            report.append(f"{param_name.upper()}:")
            report.append(
                f"  {status_symbol} Correlation: {metrics.correlation:.4f} "
                f"(threshold: {threshold:.2f})"
            )
            report.append(f"  Bias: {metrics.bias:.4f}")
            report.append(f"  RMSE: {metrics.rmse:.4f}")
            report.append(f"  MAE: {metrics.mae:.4f}")
            report.append(f"  95% CI Coverage: {metrics.within_ci_95:.1%}")
            report.append("")

        # Recommendations
        report.append("-" * 70)
        report.append("RECOMMENDATIONS")
        report.append("-" * 70)
        report.append("")

        if recovery_results.passed:
            report.append("✓ Parameter recovery validation passed.")
            report.append("  The pipeline can accurately recover ground-truth parameters.")
            report.append("  Proceed with empirical data collection.")
        else:
            report.append("✗ Parameter recovery validation failed.")
            report.append("  Consider the following:")
            report.append("  - Increase number of trials per task")
            report.append("  - Reduce noise in neural measurements")
            report.append("  - Refine model specification")
            report.append("  - Check for model identifiability issues")

        report.append("")
        report.append("=" * 70)

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)

        return report_text

    @staticmethod
    def generate_detailed_statistics(recovery_results: RecoveryResults) -> pd.DataFrame:
        """Generate detailed statistics table.

        Args:
            recovery_results: Results from parameter recovery validation

        Returns:
            DataFrame with detailed statistics
        """
        rows = []

        for param_name, metrics in recovery_results.metrics.items():
            rows.append(
                {
                    "Parameter": param_name,
                    "Correlation": metrics.correlation,
                    "Bias": metrics.bias,
                    "RMSE": metrics.rmse,
                    "MAE": metrics.mae,
                    "CI_Coverage": metrics.within_ci_95,
                    "Threshold": recovery_results.validation_criteria[param_name],
                    "Passed": metrics.correlation
                    >= recovery_results.validation_criteria[param_name],
                }
            )

        return pd.DataFrame(rows)
