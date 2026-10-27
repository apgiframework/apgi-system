"""Core statistical testing framework for APGI falsification testing.

This module provides comprehensive statistical analysis capabilities including
t-tests, ANOVA, non-parametric tests, cluster correction, and multiple
comparisons correction for validating APGI framework falsification criteria.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
from scipy.stats import f_oneway, kruskal, mannwhitneyu, ttest_ind, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests


class TestType(Enum):
    """Statistical test types supported by the framework."""

    INDEPENDENT_T_TEST = "independent_t_test"
    PAIRED_T_TEST = "paired_t_test"
    ONE_WAY_ANOVA = "one_way_anova"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    KRUSKAL_WALLIS = "kruskal_wallis"
    FRIEDMAN = "friedman"
    MCNEMAR = "mcnemar"


class CorrectionMethod(Enum):
    """Multiple comparisons correction methods."""

    BONFERRONI = "bonferroni"
    FDR_BH = "fdr_bh"  # Benjamini-Hochberg
    FDR_BY = "fdr_by"  # Benjamini-Yekutieli
    HOLM = "holm"
    SIDAK = "sidak"
    NONE = "none"


@dataclass
class StatisticalResult:
    """Container for statistical test results."""

    test_type: str
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[Union[int, Tuple[int, int]]] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_sizes: Optional[Dict[str, int]] = None
    assumptions_met: Optional[Dict[str, bool]] = None
    interpretation: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class ClusterCorrectionResult:
    """Container for cluster-corrected statistical results."""

    original_p_values: np.ndarray
    corrected_p_values: np.ndarray
    significant_clusters: np.ndarray
    cluster_sizes: np.ndarray
    threshold: float
    method: str


class StatisticalTester:
    """Core statistical testing framework for APGI falsification testing.

    Provides comprehensive statistical analysis capabilities including
    parametric and non-parametric tests, cluster correction, and
    multiple comparisons correction.
    """

    def __init__(self, alpha: float = 0.05, random_state: Optional[int] = None):
        """Initialize the statistical tester.

        Args:
            alpha: Significance level for statistical tests
            random_state: Random seed for reproducible results
        """
        self.alpha = alpha
        self.random_state = random_state

    def independent_t_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        equal_var: bool = True,
        alternative: str = "two-sided",
    ) -> StatisticalResult:
        """Perform independent samples t-test.

        Args:
            group1: First group data
            group2: Second group data
            equal_var: Whether to assume equal variances
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')

        Returns:
            StatisticalResult with test statistics and interpretation
        """
        # Check assumptions
        assumptions = self._check_t_test_assumptions(group1, group2)

        # Perform test
        statistic, p_value = ttest_ind(group1, group2, equal_var=equal_var, alternative=alternative)

        # Calculate degrees of freedom
        if equal_var:
            df = len(group1) + len(group2) - 2
        else:
            # Welch's t-test degrees of freedom
            s1_sq = np.var(group1, ddof=1)
            s2_sq = np.var(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            df = (s1_sq / n1 + s2_sq / n2) ** 2 / (
                (s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1)
            )

        # Calculate effect size (Cohen's d)
        effect_size = self._calculate_cohens_d(group1, group2)

        # Generate interpretation
        interpretation = self._interpret_result(p_value, effect_size, "independent t-test")

        return StatisticalResult(
            test_type=TestType.INDEPENDENT_T_TEST.value,
            statistic=statistic,
            p_value=p_value,
            degrees_of_freedom=df,
            effect_size=effect_size,
            sample_sizes={"group1": len(group1), "group2": len(group2)},
            assumptions_met=assumptions,
            interpretation=interpretation,
        )

    def paired_t_test(
        self, before: np.ndarray, after: np.ndarray, alternative: str = "two-sided"
    ) -> StatisticalResult:
        """Perform paired samples t-test.

        Args:
            before: Pre-condition measurements
            after: Post-condition measurements
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')

        Returns:
            StatisticalResult with test statistics and interpretation
        """
        if len(before) != len(after):
            raise ValueError("Paired samples must have equal length")

        # Check assumptions
        differences = after - before
        assumptions = self._check_paired_t_test_assumptions(differences)

        # Perform test
        statistic, p_value = ttest_rel(before, after, alternative=alternative)

        # Calculate degrees of freedom
        df = len(before) - 1

        # Calculate effect size (Cohen's d for paired samples)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)

        # Generate interpretation
        interpretation = self._interpret_result(p_value, effect_size, "paired t-test")

        return StatisticalResult(
            test_type=TestType.PAIRED_T_TEST.value,
            statistic=statistic,
            p_value=p_value,
            degrees_of_freedom=df,
            effect_size=effect_size,
            sample_sizes={"n_pairs": len(before)},
            assumptions_met=assumptions,
            interpretation=interpretation,
        )

    def one_way_anova(self, *groups: np.ndarray) -> StatisticalResult:
        """Perform one-way ANOVA.

        Args:
            *groups: Variable number of group arrays

        Returns:
            StatisticalResult with test statistics and interpretation
        """
        if len(groups) < 2:
            raise ValueError("ANOVA requires at least 2 groups")

        # Check assumptions
        assumptions = self._check_anova_assumptions(groups)

        # Perform test
        statistic, p_value = f_oneway(*groups)

        # Calculate degrees of freedom
        k = len(groups)  # number of groups
        n = sum(len(group) for group in groups)  # total sample size
        df_between = k - 1
        df_within = n - k

        # Calculate effect size (eta-squared)
        effect_size = self._calculate_eta_squared(groups, statistic, df_between, df_within)

        # Generate interpretation
        interpretation = self._interpret_result(p_value, effect_size, "one-way ANOVA")

        sample_sizes = {f"group_{i + 1}": len(group) for i, group in enumerate(groups)}

        return StatisticalResult(
            test_type=TestType.ONE_WAY_ANOVA.value,
            statistic=statistic,
            p_value=p_value,
            degrees_of_freedom=(df_between, df_within),
            effect_size=effect_size,
            sample_sizes=sample_sizes,
            assumptions_met=assumptions,
            interpretation=interpretation,
        )

    def mann_whitney_u(
        self, group1: np.ndarray, group2: np.ndarray, alternative: str = "two-sided"
    ) -> StatisticalResult:
        """Perform Mann-Whitney U test (non-parametric alternative to independent t-test).

        Args:
            group1: First group data
            group2: Second group data
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')

        Returns:
            StatisticalResult with test statistics and interpretation
        """
        # Perform test
        statistic, p_value = mannwhitneyu(group1, group2, alternative=alternative)

        # Calculate effect size (rank-biserial correlation)
        effect_size = self._calculate_rank_biserial_correlation(group1, group2, statistic)

        # Generate interpretation
        interpretation = self._interpret_result(p_value, effect_size, "Mann-Whitney U test")

        return StatisticalResult(
            test_type=TestType.MANN_WHITNEY_U.value,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            sample_sizes={"group1": len(group1), "group2": len(group2)},
            interpretation=interpretation,
        )

    def wilcoxon_signed_rank(
        self, before: np.ndarray, after: np.ndarray, alternative: str = "two-sided"
    ) -> StatisticalResult:
        """Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

        Args:
            before: Pre-condition measurements
            after: Post-condition measurements
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')

        Returns:
            StatisticalResult with test statistics and interpretation
        """
        if len(before) != len(after):
            raise ValueError("Paired samples must have equal length")

        # Perform test
        statistic, p_value = wilcoxon(before, after, alternative=alternative)

        # Calculate effect size (matched-pairs rank-biserial correlation)
        differences = after - before
        effect_size = self._calculate_matched_pairs_rank_biserial(differences, statistic)

        # Generate interpretation
        interpretation = self._interpret_result(p_value, effect_size, "Wilcoxon signed-rank test")

        return StatisticalResult(
            test_type=TestType.WILCOXON_SIGNED_RANK.value,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            sample_sizes={"n_pairs": len(before)},
            interpretation=interpretation,
        )

    def kruskal_wallis(self, *groups: np.ndarray) -> StatisticalResult:
        """Perform Kruskal-Wallis test (non-parametric alternative to one-way ANOVA).

        Args:
            *groups: Variable number of group arrays

        Returns:
            StatisticalResult with test statistics and interpretation
        """
        if len(groups) < 2:
            raise ValueError("Kruskal-Wallis test requires at least 2 groups")

        # Perform test
        statistic, p_value = kruskal(*groups)

        # Calculate degrees of freedom
        df = len(groups) - 1

        # Calculate effect size (epsilon-squared)
        effect_size = self._calculate_epsilon_squared(groups, statistic)

        # Generate interpretation
        interpretation = self._interpret_result(p_value, effect_size, "Kruskal-Wallis test")

        sample_sizes = {f"group_{i + 1}": len(group) for i, group in enumerate(groups)}

        return StatisticalResult(
            test_type=TestType.KRUSKAL_WALLIS.value,
            statistic=statistic,
            p_value=p_value,
            degrees_of_freedom=df,
            effect_size=effect_size,
            sample_sizes=sample_sizes,
            interpretation=interpretation,
        )

    def multiple_comparisons_correction(
        self,
        p_values: np.ndarray,
        method: CorrectionMethod = CorrectionMethod.FDR_BH,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply multiple comparisons correction to p-values.

        Args:
            p_values: Array of uncorrected p-values
            method: Correction method to apply
            alpha: Significance level (uses instance alpha if None)

        Returns:
            Tuple of (rejected_hypotheses, corrected_p_values)
        """
        if alpha is None:
            alpha = self.alpha

        if method == CorrectionMethod.NONE:
            return p_values < alpha, p_values

        rejected, corrected_p_values, _, _ = multipletests(
            p_values, alpha=alpha, method=method.value
        )

        return rejected, corrected_p_values

    def cluster_correction(
        self,
        p_values: np.ndarray,
        cluster_threshold: float = 0.05,
        min_cluster_size: int = 1,
    ) -> ClusterCorrectionResult:
        """Apply cluster-based correction for multiple comparisons.

        Args:
            p_values: Array of uncorrected p-values
            cluster_threshold: Threshold for forming clusters
            min_cluster_size: Minimum cluster size to be considered significant

        Returns:
            ClusterCorrectionResult with corrected statistics
        """
        # Identify significant voxels/tests at uncorrected threshold
        significant_mask = p_values < cluster_threshold

        # Find connected clusters (simplified 1D implementation)
        clusters = self._find_clusters_1d(significant_mask)

        # Calculate cluster sizes
        cluster_sizes = np.array([np.sum(cluster) for cluster in clusters])

        # Apply cluster size threshold
        significant_clusters = cluster_sizes >= min_cluster_size

        # Create corrected p-values array
        corrected_p_values = np.ones_like(p_values)

        for i, (cluster, is_significant) in enumerate(zip(clusters, significant_clusters)):
            if is_significant:
                corrected_p_values[cluster] = p_values[cluster]

        return ClusterCorrectionResult(
            original_p_values=p_values,
            corrected_p_values=corrected_p_values,
            significant_clusters=significant_clusters,
            cluster_sizes=cluster_sizes,
            threshold=cluster_threshold,
            method="cluster_correction",
        )

    def _check_t_test_assumptions(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, bool]:
        """Check assumptions for independent t-test."""
        assumptions = {}

        # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
        if len(group1) <= 50 and len(group2) <= 50:
            _, p1 = stats.shapiro(group1)
            _, p2 = stats.shapiro(group2)
            assumptions["normality_group1"] = p1 > 0.05
            assumptions["normality_group2"] = p2 > 0.05
        else:
            # Use Anderson-Darling test for larger samples
            ad1 = stats.anderson(group1, dist="norm")
            ad2 = stats.anderson(group2, dist="norm")
            assumptions["normality_group1"] = ad1.statistic < ad1.critical_values[2]  # 5% level
            assumptions["normality_group2"] = ad2.statistic < ad2.critical_values[2]

        # Homogeneity of variance (Levene's test)
        _, p_levene = stats.levene(group1, group2)
        assumptions["equal_variances"] = p_levene > 0.05

        return assumptions

    def _check_paired_t_test_assumptions(self, differences: np.ndarray) -> Dict[str, bool]:
        """Check assumptions for paired t-test."""
        assumptions = {}

        # Normality of differences
        if len(differences) <= 50:
            _, p_norm = stats.shapiro(differences)
            assumptions["normality_differences"] = p_norm > 0.05
        else:
            ad = stats.anderson(differences, dist="norm")
            assumptions["normality_differences"] = ad.statistic < ad.critical_values[2]

        return assumptions

    def _check_anova_assumptions(self, groups: Tuple[np.ndarray, ...]) -> Dict[str, bool]:
        """Check assumptions for one-way ANOVA."""
        assumptions = {}

        # Normality for each group
        normality_results = []
        for i, group in enumerate(groups):
            if len(group) <= 50:
                _, p_norm = stats.shapiro(group)
                normality_results.append(p_norm > 0.05)
            else:
                ad = stats.anderson(group, dist="norm")
                normality_results.append(ad.statistic < ad.critical_values[2])

        assumptions["normality_all_groups"] = all(normality_results)

        # Homogeneity of variance (Levene's test)
        _, p_levene = stats.levene(*groups)
        assumptions["equal_variances"] = p_levene > 0.05

        return assumptions

    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

        # Cohen's d
        d = float(np.mean(group1) - np.mean(group2)) / pooled_std
        return float(d)

    def _calculate_eta_squared(
        self,
        groups: Tuple[np.ndarray, ...],
        f_statistic: float,
        df_between: int,
        df_within: int,
    ) -> float:
        """Calculate eta-squared effect size for ANOVA."""
        # Calculate sum of squares
        grand_mean = np.mean(np.concatenate(groups))

        # Between-groups sum of squares
        ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups)

        # Within-groups sum of squares
        ss_within = sum(np.sum((group - np.mean(group)) ** 2) for group in groups)

        # Total sum of squares
        ss_total = ss_between + ss_within

        # Eta-squared
        eta_squared = float(ss_between) / float(ss_total)
        return eta_squared

    def _calculate_rank_biserial_correlation(
        self, group1: np.ndarray, group2: np.ndarray, u_statistic: float
    ) -> float:
        """Calculate rank-biserial correlation for Mann-Whitney U test."""
        n1, n2 = len(group1), len(group2)
        # Rank-biserial correlation
        r = 1 - (2 * u_statistic) / (n1 * n2)
        return r

    def _calculate_matched_pairs_rank_biserial(
        self, differences: np.ndarray, w_statistic: float
    ) -> float:
        """Calculate matched-pairs rank-biserial correlation for Wilcoxon test."""
        # Remove zero differences
        non_zero_diffs = differences[differences != 0]
        n_non_zero = len(non_zero_diffs)

        if n_non_zero == 0:
            return 0.0

        # Matched-pairs rank-biserial correlation
        r = 1 - (4 * w_statistic) / (n_non_zero * (n_non_zero + 1))
        return r

    def _calculate_epsilon_squared(
        self, groups: Tuple[np.ndarray, ...], h_statistic: float
    ) -> float:
        """Calculate epsilon-squared effect size for Kruskal-Wallis test."""
        n = sum(len(group) for group in groups)
        k = len(groups)

        # Epsilon-squared approximation
        epsilon_squared = (h_statistic - k + 1) / (n - k)
        return max(0, epsilon_squared)  # Ensure non-negative

    def _find_clusters_1d(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find connected clusters in 1D boolean mask."""
        clusters: List[np.ndarray] = []
        if not np.any(mask):
            return clusters

        # Find start and end points of clusters
        diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        # Create cluster masks
        for start, end in zip(starts, ends):
            cluster_mask = np.zeros_like(mask, dtype=bool)
            cluster_mask[start:end] = True
            clusters.append(cluster_mask)

        return clusters

    def _interpret_result(self, p_value: float, effect_size: float, test_name: str) -> str:
        """Generate interpretation of statistical result."""
        # Significance interpretation
        if p_value < 0.001:
            sig_text = "highly significant (p < 0.001)"
        elif p_value < 0.01:
            sig_text = "very significant (p < 0.01)"
        elif p_value < 0.05:
            sig_text = "significant (p < 0.05)"
        else:
            sig_text = "not significant (p ≥ 0.05)"

        # Effect size interpretation (Cohen's conventions)
        abs_effect = abs(effect_size) if effect_size is not None else 0
        if abs_effect < 0.2:
            effect_text = "negligible effect size"
        elif abs_effect < 0.5:
            effect_text = "small effect size"
        elif abs_effect < 0.8:
            effect_text = "medium effect size"
        else:
            effect_text = "large effect size"

        return f"{test_name} result is {sig_text} with {effect_text} (d = {effect_size:.3f})"
