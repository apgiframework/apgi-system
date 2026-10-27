"""Replication tracking and power analysis for APGI falsification testing.

This module provides comprehensive replication tracking across multiple
simulated labs and power analysis capabilities for determining appropriate
sample sizes and statistical power in APGI framework validation studies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2, f, norm, t


class ReplicationStatus(Enum):
    """Status of replication attempts."""

    SUCCESS = "success"
    FAILURE = "failure"
    INCONCLUSIVE = "inconclusive"
    PENDING = "pending"


class PowerAnalysisMethod(Enum):
    """Methods for power analysis."""

    T_TEST = "t_test"
    ANOVA = "anova"
    CORRELATION = "correlation"
    PROPORTION = "proportion"
    MANN_WHITNEY = "mann_whitney"


@dataclass
class ExperimentResult:
    """Container for individual experiment results."""

    experiment_id: str
    lab_id: str
    effect_size: float
    p_value: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    test_type: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplicationSummary:
    """Summary of replication attempts across labs."""

    total_replications: int
    successful_replications: int
    failed_replications: int
    inconclusive_replications: int
    success_rate: float
    mean_effect_size: float
    effect_size_heterogeneity: float
    combined_p_value: float
    confidence_in_effect: float
    interpretation: str


@dataclass
class PowerAnalysisResult:
    """Result of power analysis calculation."""

    test_type: str
    effect_size: float
    alpha: float
    power: float
    sample_size: int
    critical_value: float
    interpretation: str
    recommendations: List[str]


class ReplicationTracker:
    """Tracks replication attempts across multiple simulated laboratories.

    Provides comprehensive analysis of replication success rates,
    effect size consistency, and meta-analytic combination of results
    for APGI framework validation studies.
    """

    def __init__(
        self,
        success_threshold: float = 0.05,
        effect_size_threshold: float = 0.1,
        min_replications: int = 3,
    ):
        """Initialize the replication tracker.

        Args:
            success_threshold: P-value threshold for successful replication
            effect_size_threshold: Minimum meaningful effect size
            min_replications: Minimum number of replications for analysis
        """
        self.success_threshold = success_threshold
        self.effect_size_threshold = effect_size_threshold
        self.min_replications = min_replications
        self.experiments: List[ExperimentResult] = []

    def add_experiment_result(self, result: ExperimentResult) -> None:
        """Add an experiment result to the tracking system.

        Args:
            result: ExperimentResult to add
        """
        self.experiments.append(result)

    def add_multiple_results(self, results: List[ExperimentResult]) -> None:
        """Add multiple experiment results.

        Args:
            results: List of ExperimentResult objects
        """
        self.experiments.extend(results)

    def evaluate_replication_success(
        self, original_effect_size: float, replication_tolerance: float = 0.5
    ) -> ReplicationSummary:
        """Evaluate overall replication success across all experiments.

        Args:
            original_effect_size: Effect size from original study
            replication_tolerance: Tolerance for effect size differences

        Returns:
            ReplicationSummary with comprehensive analysis
        """
        if len(self.experiments) < self.min_replications:
            raise ValueError(f"Need at least {self.min_replications} replications for analysis")

        # Classify each replication
        successful = 0
        failed = 0
        inconclusive = 0

        effect_sizes = []
        p_values = []

        for exp in self.experiments:
            effect_sizes.append(exp.effect_size)
            p_values.append(exp.p_value)

            # Determine replication status
            if exp.p_value < self.success_threshold and abs(
                exp.effect_size - original_effect_size
            ) <= replication_tolerance * abs(original_effect_size):
                successful += 1
            elif exp.p_value >= self.success_threshold:
                failed += 1
            else:
                inconclusive += 1

        total = len(self.experiments)
        success_rate = successful / total

        # Calculate meta-analytic statistics
        mean_effect_size = float(np.mean(effect_sizes))
        effect_size_heterogeneity = float(np.std(effect_sizes))

        # Combine p-values using Fisher's method
        combined_p_value = self._combine_p_values_fisher(p_values)

        # Calculate confidence in effect
        confidence_in_effect = self._calculate_confidence_in_effect(
            effect_sizes, original_effect_size, success_rate
        )

        # Generate interpretation
        interpretation = self._interpret_replication_results(
            success_rate, mean_effect_size, original_effect_size, combined_p_value
        )

        return ReplicationSummary(
            total_replications=total,
            successful_replications=successful,
            failed_replications=failed,
            inconclusive_replications=inconclusive,
            success_rate=success_rate,
            mean_effect_size=mean_effect_size,
            effect_size_heterogeneity=effect_size_heterogeneity,
            combined_p_value=combined_p_value,
            confidence_in_effect=confidence_in_effect,
            interpretation=interpretation,
        )

    def meta_analysis(self, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform meta-analysis of effect sizes across replications.

        Args:
            weights: Optional weights for each study (default: sample size)

        Returns:
            Dictionary with meta-analytic results
        """
        if len(self.experiments) < 2:
            raise ValueError("Need at least 2 studies for meta-analysis")

        effect_sizes = np.array([exp.effect_size for exp in self.experiments])
        sample_sizes = np.array([exp.sample_size for exp in self.experiments])

        # Use sample sizes as weights if not provided
        if weights is None:
            weights = sample_sizes
        else:
            weights = np.array(weights)

        # Weighted mean effect size
        weighted_mean = np.average(effect_sizes, weights=weights)

        # Standard error of weighted mean
        weighted_se = np.sqrt(1 / np.sum(weights))

        # Heterogeneity statistics (Q and I²)
        q_statistic = np.sum(weights * (effect_sizes - weighted_mean) ** 2)
        df = len(effect_sizes) - 1
        q_p_value = 1 - chi2.cdf(q_statistic, df)

        # I² statistic (percentage of variation due to heterogeneity)
        i_squared = max(0, (q_statistic - df) / q_statistic) * 100

        # Confidence interval for weighted mean
        alpha = 0.05
        z_crit = norm.ppf(1 - alpha / 2)
        ci_lower = weighted_mean - z_crit * weighted_se
        ci_upper = weighted_mean + z_crit * weighted_se

        return {
            "weighted_mean_effect_size": weighted_mean,
            "standard_error": weighted_se,
            "confidence_interval": (ci_lower, ci_upper),
            "q_statistic": q_statistic,
            "q_p_value": q_p_value,
            "i_squared": i_squared,
            "heterogeneity": (
                "low" if i_squared < 25 else "moderate" if i_squared < 75 else "high"
            ),
        }

    def get_lab_performance(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by laboratory.

        Returns:
            Dictionary with lab-specific performance metrics
        """
        lab_results = {}

        # Group experiments by lab
        labs: Dict[str, List[ExperimentResult]] = {}
        for exp in self.experiments:
            if exp.lab_id not in labs:
                labs[exp.lab_id] = []
            labs[exp.lab_id].append(exp)

        # Analyze each lab
        for lab_id, lab_experiments in labs.items():
            effect_sizes = [exp.effect_size for exp in lab_experiments]
            p_values = [exp.p_value for exp in lab_experiments]
            sample_sizes = [exp.sample_size for exp in lab_experiments]

            # Calculate lab statistics
            mean_effect_size = np.mean(effect_sizes)
            effect_size_consistency = 1 / (1 + np.std(effect_sizes))  # Higher = more consistent
            success_rate = np.mean([p < self.success_threshold for p in p_values])

            lab_results[lab_id] = {
                "n_experiments": len(lab_experiments),
                "mean_effect_size": mean_effect_size,
                "effect_size_std": np.std(effect_sizes),
                "effect_size_consistency": effect_size_consistency,
                "success_rate": success_rate,
                "mean_sample_size": np.mean(sample_sizes),
                "experiments": lab_experiments,
            }

        return lab_results

    def _combine_p_values_fisher(self, p_values: List[float]) -> float:
        """Combine p-values using Fisher's method."""
        # Remove any p-values that are exactly 0 or 1 to avoid log issues
        valid_p_values = [p for p in p_values if 0 < p < 1]

        if not valid_p_values:
            return 1.0

        # Fisher's combined test statistic
        chi2_stat = -2 * np.sum(np.log(valid_p_values))
        df = 2 * len(valid_p_values)

        # Combined p-value
        combined_p = float(1 - chi2.cdf(chi2_stat, df))
        return combined_p

    def _calculate_confidence_in_effect(
        self,
        effect_sizes: List[float],
        original_effect_size: float,
        success_rate: float,
    ) -> float:
        """Calculate overall confidence in the effect."""
        # Factors contributing to confidence
        consistency_factor = 1 / (1 + np.std(effect_sizes))
        magnitude_factor = min(1.0, float(abs(np.mean(effect_sizes)) / abs(original_effect_size)))
        replication_factor = success_rate

        # Combined confidence score
        confidence = (consistency_factor * magnitude_factor * replication_factor) ** (1 / 3)
        return float(confidence)

    def _interpret_replication_results(
        self,
        success_rate: float,
        mean_effect_size: float,
        original_effect_size: float,
        combined_p_value: float,
    ) -> str:
        """Generate interpretation of replication results."""
        # Success rate interpretation
        if success_rate >= 0.8:
            success_text = "excellent replication success"
        elif success_rate >= 0.6:
            success_text = "good replication success"
        elif success_rate >= 0.4:
            success_text = "moderate replication success"
        else:
            success_text = "poor replication success"

        # Effect size consistency
        effect_ratio = (
            abs(mean_effect_size / original_effect_size) if original_effect_size != 0 else 0
        )
        if 0.8 <= effect_ratio <= 1.2:
            effect_text = "consistent effect size"
        elif 0.5 <= effect_ratio <= 1.5:
            effect_text = "moderately consistent effect size"
        else:
            effect_text = "inconsistent effect size"

        # Combined significance
        if combined_p_value < 0.001:
            sig_text = "highly significant combined result"
        elif combined_p_value < 0.01:
            sig_text = "significant combined result"
        elif combined_p_value < 0.05:
            sig_text = "marginally significant combined result"
        else:
            sig_text = "non-significant combined result"

        return f"Replication shows {success_text} ({success_rate:.1%}) with {effect_text} and {sig_text}"


class PowerAnalyzer:
    """Comprehensive power analysis for experimental design.

    Provides sample size calculations and power analysis for various
    statistical tests used in APGI framework validation studies.
    """

    def __init__(self, alpha: float = 0.05):
        """Initialize the power analyzer.

        Args:
            alpha: Type I error rate (significance level)
        """
        self.alpha = alpha

    def t_test_power(
        self,
        effect_size: float,
        sample_size: Optional[int] = None,
        power: Optional[float] = None,
        test_type: str = "two_sample",
    ) -> PowerAnalysisResult:
        """Calculate power analysis for t-tests.

        Args:
            effect_size: Cohen's d effect size
            sample_size: Sample size (if calculating power)
            power: Desired power (if calculating sample size)
            test_type: Type of t-test ("one_sample", "two_sample", "paired")

        Returns:
            PowerAnalysisResult with analysis
        """
        if sample_size is not None and power is not None:
            raise ValueError("Specify either sample_size or power, not both")

        if sample_size is not None:
            # Calculate power given sample size
            if test_type == "one_sample":
                df = sample_size - 1
                ncp = effect_size * np.sqrt(sample_size)
            elif test_type == "two_sample":
                df = 2 * sample_size - 2
                ncp = effect_size * np.sqrt(sample_size / 2)
            else:  # paired
                df = sample_size - 1
                ncp = effect_size * np.sqrt(sample_size)

            # Critical value
            t_crit = t.ppf(1 - self.alpha / 2, df)

            # Power calculation using non-central t-distribution
            power_calc = 1 - t.cdf(t_crit, df, ncp) + t.cdf(-t_crit, df, ncp)

            result_power = power_calc
            result_n = sample_size

        elif power is not None:
            # Calculate sample size given power
            result_n = self._calculate_sample_size_t_test(effect_size, power, test_type)
            result_power = power
        else:
            raise ValueError("Must specify either sample_size or power")

        # Generate recommendations
        recommendations = self._generate_power_recommendations(
            result_power, result_n, effect_size, "t-test"
        )

        # Interpretation
        interpretation = self._interpret_power_analysis(
            result_power, result_n, effect_size, "t-test"
        )

        return PowerAnalysisResult(
            test_type=f"{test_type}_t_test",
            effect_size=effect_size,
            alpha=self.alpha,
            power=result_power,
            sample_size=result_n,
            critical_value=t.ppf(1 - self.alpha / 2, result_n - 1),
            interpretation=interpretation,
            recommendations=recommendations,
        )

    def anova_power(
        self,
        effect_size: float,
        n_groups: int,
        sample_size_per_group: Optional[int] = None,
        power: Optional[float] = None,
    ) -> PowerAnalysisResult:
        """Calculate power analysis for one-way ANOVA.

        Args:
            effect_size: Cohen's f effect size
            n_groups: Number of groups
            sample_size_per_group: Sample size per group
            power: Desired power

        Returns:
            PowerAnalysisResult with analysis
        """
        if sample_size_per_group is not None and power is not None:
            raise ValueError("Specify either sample_size_per_group or power, not both")

        if sample_size_per_group is not None:
            # Calculate power given sample size
            total_n = n_groups * sample_size_per_group
            df_between = n_groups - 1
            df_within = total_n - n_groups

            # Non-centrality parameter
            ncp = effect_size**2 * total_n

            # Critical F value
            f_crit = f.ppf(1 - self.alpha, df_between, df_within)

            # Power using non-central F distribution (approximation)
            power_calc = 1 - f.cdf(f_crit, df_between, df_within, ncp)

            result_power = power_calc
            result_n = sample_size_per_group

        elif power is not None:
            # Calculate sample size given power (iterative approach)
            result_n = self._calculate_sample_size_anova(effect_size, n_groups, power)
            result_power = power
        else:
            raise ValueError("Must specify either sample_size_per_group or power")

        # Generate recommendations
        recommendations = self._generate_power_recommendations(
            result_power, result_n * n_groups, effect_size, "ANOVA"
        )

        # Interpretation
        interpretation = self._interpret_power_analysis(
            result_power, result_n * n_groups, effect_size, "ANOVA"
        )

        return PowerAnalysisResult(
            test_type="one_way_anova",
            effect_size=effect_size,
            alpha=self.alpha,
            power=result_power,
            sample_size=result_n,
            critical_value=f.ppf(1 - self.alpha, n_groups - 1, result_n * n_groups - n_groups),
            interpretation=interpretation,
            recommendations=recommendations,
        )

    def correlation_power(
        self,
        effect_size: float,
        sample_size: Optional[int] = None,
        power: Optional[float] = None,
    ) -> PowerAnalysisResult:
        """Calculate power analysis for correlation tests.

        Args:
            effect_size: Pearson correlation coefficient
            sample_size: Sample size
            power: Desired power

        Returns:
            PowerAnalysisResult with analysis
        """
        if sample_size is not None and power is not None:
            raise ValueError("Specify either sample_size or power, not both")

        if sample_size is not None:
            # Calculate power given sample size
            # df = sample_size - 2  # Not needed for power calculation

            # Fisher's z transformation
            z_r = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
            se_z = 1 / np.sqrt(sample_size - 3)

            # Critical value
            z_crit = norm.ppf(1 - self.alpha / 2)

            # Power calculation
            power_calc = (
                1 - norm.cdf(z_crit - abs(z_r) / se_z) + norm.cdf(-z_crit - abs(z_r) / se_z)
            )

            result_power = power_calc
            result_n = sample_size

        elif power is not None:
            # Calculate sample size given power
            result_n = self._calculate_sample_size_correlation(effect_size, power)
            result_power = power
        else:
            raise ValueError("Must specify either sample_size or power")

        # Generate recommendations
        recommendations = self._generate_power_recommendations(
            result_power, result_n, effect_size, "correlation"
        )

        # Interpretation
        interpretation = self._interpret_power_analysis(
            result_power, result_n, effect_size, "correlation"
        )

        return PowerAnalysisResult(
            test_type="correlation",
            effect_size=effect_size,
            alpha=self.alpha,
            power=result_power,
            sample_size=result_n,
            critical_value=norm.ppf(1 - self.alpha / 2),
            interpretation=interpretation,
            recommendations=recommendations,
        )

    def _calculate_sample_size_t_test(
        self, effect_size: float, power: float, test_type: str
    ) -> int:
        """Calculate sample size for t-test using iterative approach."""
        # Initial guess
        n = 10

        # Iterative search
        for _ in range(1000):  # Maximum iterations
            if test_type == "one_sample":
                df = n - 1
                ncp = effect_size * np.sqrt(n)
            elif test_type == "two_sample":
                df = 2 * n - 2
                ncp = effect_size * np.sqrt(n / 2)
            else:  # paired
                df = n - 1
                ncp = effect_size * np.sqrt(n)

            t_crit = t.ppf(1 - self.alpha / 2, df)
            current_power = 1 - t.cdf(t_crit, df, ncp) + t.cdf(-t_crit, df, ncp)

            if abs(current_power - power) < 0.001:
                break

            # Adjust sample size
            if current_power < power:
                n += 1
            else:
                break

        return n

    def _calculate_sample_size_anova(self, effect_size: float, n_groups: int, power: float) -> int:
        """Calculate sample size per group for ANOVA using iterative approach."""
        # Initial guess
        n_per_group = 5

        # Iterative search
        for _ in range(1000):
            total_n = n_groups * n_per_group
            df_between = n_groups - 1
            df_within = total_n - n_groups

            ncp = effect_size**2 * total_n
            f_crit = f.ppf(1 - self.alpha, df_between, df_within)
            current_power = 1 - f.cdf(f_crit, df_between, df_within, ncp)

            if abs(current_power - power) < 0.001:
                break

            if current_power < power:
                n_per_group += 1
            else:
                break

        return n_per_group

    def _calculate_sample_size_correlation(self, effect_size: float, power: float) -> int:
        """Calculate sample size for correlation using iterative approach."""
        # Initial guess
        n = 10

        # Iterative search
        for _ in range(1000):
            z_r = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
            se_z = 1 / np.sqrt(n - 3)
            z_crit = norm.ppf(1 - self.alpha / 2)

            current_power = (
                1 - norm.cdf(z_crit - abs(z_r) / se_z) + norm.cdf(-z_crit - abs(z_r) / se_z)
            )

            if abs(current_power - power) < 0.001:
                break

            if current_power < power:
                n += 1
            else:
                break

        return n

    def _generate_power_recommendations(
        self, power: float, sample_size: int, effect_size: float, test_type: str
    ) -> List[str]:
        """Generate recommendations based on power analysis."""
        recommendations = []

        if power < 0.8:
            recommendations.append(f"Power ({power:.3f}) is below recommended 0.8 threshold")
            recommendations.append(f"Consider increasing sample size above {sample_size}")

        if abs(effect_size) < 0.2:
            recommendations.append("Effect size is small - consider if practically meaningful")

        if sample_size > 1000:
            recommendations.append("Large sample size required - consider feasibility")

        if power > 0.95:
            recommendations.append("Very high power - sample size might be larger than necessary")

        return recommendations

    def _interpret_power_analysis(
        self, power: float, sample_size: int, effect_size: float, test_type: str
    ) -> str:
        """Generate interpretation of power analysis."""
        power_level = "adequate" if power >= 0.8 else "inadequate"
        effect_magnitude = (
            "small" if abs(effect_size) < 0.5 else "medium" if abs(effect_size) < 0.8 else "large"
        )

        return (
            f"{test_type} with {effect_magnitude} effect size (d={effect_size:.3f}) "
            f"requires n={sample_size} for {power_level} power ({power:.3f})"
        )
