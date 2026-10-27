"""Effect size and confidence interval calculations for APGI falsification testing.

This module provides comprehensive effect size calculations including Cohen's d,
eta-squared, omega-squared, and bootstrap confidence interval estimation
for validating statistical significance in APGI framework testing.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
from scipy.stats import ncf, t


class EffectSizeType(Enum):
    """Types of effect sizes supported."""

    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"
    ETA_SQUARED = "eta_squared"
    PARTIAL_ETA_SQUARED = "partial_eta_squared"
    OMEGA_SQUARED = "omega_squared"
    COHENS_F = "cohens_f"
    CRAMER_V = "cramer_v"
    PEARSON_R = "pearson_r"
    RANK_BISERIAL = "rank_biserial"


class ConfidenceIntervalMethod(Enum):
    """Methods for calculating confidence intervals."""

    PARAMETRIC = "parametric"
    BOOTSTRAP = "bootstrap"
    BIAS_CORRECTED_BOOTSTRAP = "bc_bootstrap"
    PERCENTILE_BOOTSTRAP = "percentile_bootstrap"


@dataclass
class EffectSizeResult:
    """Container for effect size calculation results."""

    effect_size_type: str
    value: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    interpretation: str
    sample_sizes: Dict[str, int]
    method: str
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class BootstrapResult:
    """Container for bootstrap confidence interval results."""

    original_statistic: float
    bootstrap_distribution: np.ndarray
    confidence_interval: Tuple[float, float]
    confidence_level: float
    n_bootstrap: int
    bias_correction: Optional[float] = None


class EffectSizeCalculator:
    """Comprehensive effect size and confidence interval calculator.

    Provides various effect size measures and bootstrap confidence
    interval estimation for statistical analysis in APGI framework testing.
    """

    def __init__(self, random_state: Optional[int] = None):
        """Initialize the effect size calculator.

        Args:
            random_state: Random seed for reproducible bootstrap results
        """
        self.random_state = random_state

    def cohens_d(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        confidence_level: float = 0.95,
        method: ConfidenceIntervalMethod = ConfidenceIntervalMethod.BOOTSTRAP,
    ) -> EffectSizeResult:
        """Calculate Cohen's d effect size with confidence intervals.

        Args:
            group1: First group data
            group2: Second group data
            confidence_level: Confidence level for intervals (default 0.95)
            method: Method for calculating confidence intervals

        Returns:
            EffectSizeResult with Cohen's d and confidence intervals
        """
        # Validate minimum group sizes
        if len(group1) < 2 or len(group2) < 2:
            raise ValueError("Groups must have at least 2 observations for Cohen's d calculation")

        # Calculate Cohen's d
        n1, n2 = len(group1), len(group2)
        m1, m2 = np.mean(group1), np.mean(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

        # Cohen's d
        d = (m1 - m2) / pooled_std if pooled_std != 0 else 0.0

        # Calculate confidence interval
        if method == ConfidenceIntervalMethod.PARAMETRIC:
            ci = self._cohens_d_parametric_ci(d, n1, n2, confidence_level)
        else:
            ci = self._bootstrap_cohens_d_ci(group1, group2, confidence_level, method)

        # Interpretation
        interpretation = self._interpret_cohens_d(d)

        return EffectSizeResult(
            effect_size_type=EffectSizeType.COHENS_D.value,
            value=d,
            confidence_interval=ci,
            confidence_level=confidence_level,
            interpretation=interpretation,
            sample_sizes={"group1": n1, "group2": n2},
            method=method.value,
        )

    def hedges_g(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        confidence_level: float = 0.95,
        method: ConfidenceIntervalMethod = ConfidenceIntervalMethod.BOOTSTRAP,
    ) -> EffectSizeResult:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d).

        Args:
            group1: First group data
            group2: Second group data
            confidence_level: Confidence level for intervals
            method: Method for calculating confidence intervals

        Returns:
            EffectSizeResult with Hedges' g and confidence intervals
        """
        # First calculate Cohen's d
        cohens_d_result = self.cohens_d(group1, group2, confidence_level, method)
        d = cohens_d_result.value

        # Bias correction factor
        n1, n2 = len(group1), len(group2)
        df = n1 + n2 - 2
        correction_factor = 1 - (3 / (4 * df - 1))

        # Hedges' g
        g = d * correction_factor

        # Adjust confidence interval
        ci_lower, ci_upper = cohens_d_result.confidence_interval
        ci = (ci_lower * correction_factor, ci_upper * correction_factor)

        # Interpretation (same as Cohen's d)
        interpretation = self._interpret_cohens_d(g)

        return EffectSizeResult(
            effect_size_type=EffectSizeType.HEDGES_G.value,
            value=g,
            confidence_interval=ci,
            confidence_level=confidence_level,
            interpretation=interpretation,
            sample_sizes={"group1": n1, "group2": n2},
            method=method.value,
        )

    def eta_squared(
        self,
        groups: Tuple[np.ndarray, ...],
        f_statistic: float,
        confidence_level: float = 0.95,
    ) -> EffectSizeResult:
        """Calculate eta-squared effect size for ANOVA.

        Args:
            groups: Tuple of group arrays
            f_statistic: F-statistic from ANOVA
            confidence_level: Confidence level for intervals

        Returns:
            EffectSizeResult with eta-squared and confidence intervals
        """
        # Calculate eta-squared
        k = len(groups)  # number of groups
        n = sum(len(group) for group in groups)  # total sample size
        df_between = k - 1
        df_within = n - k

        # Calculate sum of squares
        grand_mean = np.mean(np.concatenate(groups))
        ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups)
        ss_within = sum(np.sum((group - np.mean(group)) ** 2) for group in groups)
        ss_total = ss_between + ss_within

        eta_squared = ss_between / ss_total

        # Confidence interval using non-central F distribution
        ci = self._eta_squared_ci(f_statistic, df_between, df_within, confidence_level)

        # Interpretation
        interpretation = self._interpret_eta_squared(eta_squared)

        sample_sizes = {f"group_{i + 1}": len(group) for i, group in enumerate(groups)}

        return EffectSizeResult(
            effect_size_type=EffectSizeType.ETA_SQUARED.value,
            value=eta_squared,
            confidence_interval=ci,
            confidence_level=confidence_level,
            interpretation=interpretation,
            sample_sizes=sample_sizes,
            method="parametric",
        )

    def omega_squared(
        self,
        groups: Tuple[np.ndarray, ...],
        f_statistic: float,
        confidence_level: float = 0.95,
    ) -> EffectSizeResult:
        """Calculate omega-squared effect size (less biased than eta-squared).

        Args:
            groups: Tuple of group arrays
            f_statistic: F-statistic from ANOVA
            confidence_level: Confidence level for intervals

        Returns:
            EffectSizeResult with omega-squared and confidence intervals
        """
        k = len(groups)  # number of groups
        n = sum(len(group) for group in groups)  # total sample size
        df_between = k - 1

        # Omega-squared
        omega_squared = (df_between * (f_statistic - 1)) / (df_between * (f_statistic - 1) + n)
        omega_squared = max(0, omega_squared)  # Ensure non-negative

        # Confidence interval (approximation using eta-squared CI)
        eta_sq_result = self.eta_squared(groups, f_statistic, confidence_level)
        ci = eta_sq_result.confidence_interval

        # Interpretation
        interpretation = self._interpret_eta_squared(omega_squared)  # Same scale as eta-squared

        sample_sizes = {f"group_{i + 1}": len(group) for i, group in enumerate(groups)}

        return EffectSizeResult(
            effect_size_type=EffectSizeType.OMEGA_SQUARED.value,
            value=omega_squared,
            confidence_interval=ci,
            confidence_level=confidence_level,
            interpretation=interpretation,
            sample_sizes=sample_sizes,
            method="parametric",
        )

    def pearson_r(
        self, x: np.ndarray, y: np.ndarray, confidence_level: float = 0.95
    ) -> EffectSizeResult:
        """Calculate Pearson correlation coefficient with confidence intervals.

        Args:
            x: First variable
            y: Second variable
            confidence_level: Confidence level for intervals

        Returns:
            EffectSizeResult with Pearson r and confidence intervals
        """
        if len(x) != len(y):
            raise ValueError("Variables must have equal length")

        # Calculate Pearson correlation
        r, _ = stats.pearsonr(x, y)
        n = len(x)

        # Fisher's z-transformation for confidence interval
        z_r = np.arctanh(r)
        se_z = 1 / np.sqrt(n - 3)

        # Critical value
        alpha = 1 - confidence_level
        z_crit = stats.norm.ppf(1 - alpha / 2)

        # Confidence interval in z-space
        z_lower = z_r - z_crit * se_z
        z_upper = z_r + z_crit * se_z

        # Transform back to r-space
        ci = (np.tanh(z_lower), np.tanh(z_upper))

        # Interpretation
        interpretation = self._interpret_correlation(r)

        return EffectSizeResult(
            effect_size_type=EffectSizeType.PEARSON_R.value,
            value=r,
            confidence_interval=ci,
            confidence_level=confidence_level,
            interpretation=interpretation,
            sample_sizes={"n": n},
            method="fisher_z_transform",
        )

    def bootstrap_confidence_interval(
        self,
        data: Union[np.ndarray, Tuple[np.ndarray, ...]],
        statistic_func: Callable[[Union[np.ndarray, Tuple[np.ndarray, ...]]], float],
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000,
        method: ConfidenceIntervalMethod = ConfidenceIntervalMethod.BIAS_CORRECTED_BOOTSTRAP,
    ) -> BootstrapResult:
        """Calculate bootstrap confidence intervals for any statistic.

        Args:
            data: Data array or tuple of arrays
            statistic_func: Function to calculate statistic
            confidence_level: Confidence level for intervals
            n_bootstrap: Number of bootstrap samples
            method: Bootstrap method to use

        Returns:
            BootstrapResult with confidence intervals
        """
        # Calculate original statistic
        original_stat = statistic_func(data)

        # Create local random generator
        rng = (
            np.random.default_rng(self.random_state)
            if self.random_state is not None
            else np.random.default_rng()
        )

        # Generate bootstrap samples
        bootstrap_stats_list: List[float] = []

        bootstrap_data: Union[np.ndarray, Tuple[np.ndarray, ...]]

        if isinstance(data, tuple):
            # Multiple arrays - resample each independently
            for _ in range(n_bootstrap):
                # Resample each array independently to avoid preserving paired structure
                bootstrap_data = tuple(rng.choice(arr, size=len(arr), replace=True) for arr in data)
                bootstrap_stat = statistic_func(bootstrap_data)
                bootstrap_stats_list.append(bootstrap_stat)
        else:
            # Single array
            n = len(data)
            for _ in range(n_bootstrap):
                indices = rng.choice(n, size=n, replace=True)
                bootstrap_data = data[indices]
                bootstrap_stat = statistic_func(bootstrap_data)
                bootstrap_stats_list.append(bootstrap_stat)

        bootstrap_stats = np.array(bootstrap_stats_list)

        # Calculate confidence interval based on method
        alpha = 1 - confidence_level

        if method == ConfidenceIntervalMethod.PERCENTILE_BOOTSTRAP:
            ci = self._percentile_ci(bootstrap_stats, alpha)
            bias_correction = None
        elif method == ConfidenceIntervalMethod.BIAS_CORRECTED_BOOTSTRAP:
            ci, bias_correction = self._bias_corrected_ci(bootstrap_stats, original_stat, alpha)
        else:  # Default to percentile
            ci = self._percentile_ci(bootstrap_stats, alpha)
            bias_correction = None

        return BootstrapResult(
            original_statistic=original_stat,
            bootstrap_distribution=bootstrap_stats,
            confidence_interval=ci,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            bias_correction=bias_correction,
        )

    def _cohens_d_parametric_ci(
        self, d: float, n1: int, n2: int, confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate parametric confidence interval for Cohen's d."""
        # Approximate standard error
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))

        # Degrees of freedom
        df = n1 + n2 - 2

        # Critical value
        alpha = 1 - confidence_level
        t_crit = t.ppf(1 - alpha / 2, df)

        # Confidence interval
        ci_lower = d - t_crit * se_d
        ci_upper = d + t_crit * se_d

        return (ci_lower, ci_upper)

    def _bootstrap_cohens_d_ci(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        confidence_level: float,
        method: ConfidenceIntervalMethod,
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for Cohen's d."""

        def cohens_d_func(data: Any) -> float:
            g1, g2 = data
            n1, n2 = len(g1), len(g2)
            m1, m2 = np.mean(g1), np.mean(g2)
            s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
            return float((m1 - m2) / pooled_std)

        bootstrap_result = self.bootstrap_confidence_interval(
            (group1, group2), cohens_d_func, confidence_level, method=method
        )

        return bootstrap_result.confidence_interval

    def _eta_squared_ci(
        self, f_stat: float, df_between: int, df_within: int, confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for eta-squared using non-central F distribution."""
        alpha = 1 - confidence_level

        # Non-centrality parameter
        lambda_nc = f_stat * df_between

        # Use non-central F distribution for proper confidence intervals
        f_lower = ncf.ppf(alpha / 2, df_between, df_within, lambda_nc)
        f_upper = ncf.ppf(1 - alpha / 2, df_between, df_within, lambda_nc)

        # Convert to eta-squared
        eta_lower = max(0, (f_lower * df_between - df_between) / (f_lower * df_between + df_within))
        eta_upper = min(1, (f_upper * df_between - df_between) / (f_upper * df_between + df_within))

        return (eta_lower, eta_upper)

    def _percentile_ci(self, bootstrap_stats: np.ndarray, alpha: float) -> Tuple[float, float]:
        """Calculate percentile confidence interval."""
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)

        return (ci_lower, ci_upper)

    def _bias_corrected_ci(
        self, bootstrap_stats: np.ndarray, original_stat: float, alpha: float
    ) -> Tuple[Tuple[float, float], float]:
        """Calculate bias-corrected confidence interval."""
        # Bias correction
        n_below = np.sum(bootstrap_stats < original_stat)
        bias_correction = stats.norm.ppf(n_below / len(bootstrap_stats))

        # Adjusted percentiles
        z_alpha_2 = stats.norm.ppf(alpha / 2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)

        lower_percentile = stats.norm.cdf(2 * bias_correction + z_alpha_2) * 100
        upper_percentile = stats.norm.cdf(2 * bias_correction + z_1_alpha_2) * 100

        # Ensure percentiles are within valid range
        lower_percentile = max(0, min(100, lower_percentile))
        upper_percentile = max(0, min(100, upper_percentile))

        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)

        return (ci_lower, ci_upper), bias_correction

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            magnitude = "negligible"
        elif abs_d < 0.5:
            magnitude = "small"
        elif abs_d < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        direction = "positive" if d > 0 else "negative" if d < 0 else "zero"
        return f"{magnitude} {direction} effect (|d| = {abs_d:.3f})"

    def _interpret_eta_squared(self, eta_sq: float) -> str:
        """Interpret eta-squared effect size."""
        if eta_sq < 0.01:
            magnitude = "negligible"
        elif eta_sq < 0.06:
            magnitude = "small"
        elif eta_sq < 0.14:
            magnitude = "medium"
        else:
            magnitude = "large"

        return f"{magnitude} effect (η² = {eta_sq:.3f})"

    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient."""
        abs_r = abs(r)
        if abs_r < 0.1:
            magnitude = "negligible"
        elif abs_r < 0.3:
            magnitude = "small"
        elif abs_r < 0.5:
            magnitude = "medium"
        else:
            magnitude = "large"

        direction = "positive" if r > 0 else "negative" if r < 0 else "zero"
        return f"{magnitude} {direction} correlation (|r| = {abs_r:.3f})"


def get_effect_size_guidelines() -> Dict[str, Dict[str, str]]:
    """Get interpretation guidelines for different effect sizes.

    Returns:
        Dictionary with effect size guidelines
    """
    return {
        "cohens_d": {
            "negligible": "< 0.2",
            "small": "0.2 - 0.5",
            "medium": "0.5 - 0.8",
            "large": "> 0.8",
        },
        "eta_squared": {
            "negligible": "< 0.01",
            "small": "0.01 - 0.06",
            "medium": "0.06 - 0.14",
            "large": "> 0.14",
        },
        "correlation": {
            "negligible": "< 0.1",
            "small": "0.1 - 0.3",
            "medium": "0.3 - 0.5",
            "large": "> 0.5",
        },
    }
