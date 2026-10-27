"""Predictive validity testing framework for APGI parameters.

This module implements validation tasks to test whether estimated parameters
(θ₀, Πᵢ, β) predict performance on independent behavioral measures.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

from .bayesian_models import ParameterEstimates


@dataclass
class TaskPerformance:
    """Performance metrics from a validation task."""

    participant_id: str
    task_name: str
    accuracy: float
    reaction_time_mean: float
    reaction_time_std: float
    interference_effect: Optional[float] = None
    lapse_rate: Optional[float] = None
    additional_metrics: Optional[Dict[str, float]] = None


@dataclass
class ValidityResult:
    """Results from predictive validity analysis."""

    parameter_name: str
    task_name: str
    correlation: float
    p_value: float
    r_squared: float
    n_participants: int
    significant: bool
    effect_size: str  # 'small', 'medium', 'large'

    def __repr__(self) -> str:
        sig = (
            "***"
            if self.p_value < 0.001
            else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else "ns"
        )
        return (
            f"ValidityResult({self.parameter_name} → {self.task_name}: "
            f"r={self.correlation:.3f} {sig}, n={self.n_participants})"
        )


@dataclass
class ComparativeValidityResult:
    """Results comparing APGI parameters to traditional measures."""

    apgi_parameter: str
    traditional_measure: str
    apgi_correlation: float
    traditional_correlation: float
    apgi_r_squared: float
    traditional_r_squared: float
    apgi_better: bool
    improvement_percentage: float


class EmotionalInterferenceTask:
    """Emotional interference task (Stroop/flanker) for Πᵢ validation.

    Tests whether interoceptive precision predicts performance disruption
    on emotional interference tasks.
    """

    def __init__(self, task_type: str = "stroop"):
        """Initialize emotional interference task.

        Args:
            task_type: Type of task ('stroop' or 'flanker')
        """
        self.task_type = task_type

    def run_task(
        self,
        participant_id: str,
        n_trials: int = 120,
        emotional_proportion: float = 0.5,
    ) -> TaskPerformance:
        """Run emotional interference task.

        Args:
            participant_id: Participant identifier
            n_trials: Number of trials
            emotional_proportion: Proportion of emotional trials

        Returns:
            TaskPerformance with results
        """
        # This is a placeholder - actual implementation would present stimuli
        # and collect responses

        # Simulate performance data
        accuracy = np.random.uniform(0.7, 0.95)
        rt_mean = np.random.uniform(500, 800)  # ms
        rt_std = np.random.uniform(100, 200)

        # Interference effect: RT difference between emotional and neutral
        interference_effect = np.random.uniform(50, 150)  # ms

        return TaskPerformance(
            participant_id=participant_id,
            task_name=f"emotional_{self.task_type}",
            accuracy=accuracy,
            reaction_time_mean=rt_mean,
            reaction_time_std=rt_std,
            interference_effect=interference_effect,
        )

    def compute_interference_score(self, emotional_rt: np.ndarray, neutral_rt: np.ndarray) -> float:
        """Compute interference effect.

        Args:
            emotional_rt: Reaction times for emotional trials
            neutral_rt: Reaction times for neutral trials

        Returns:
            Interference effect (emotional - neutral)
        """
        return float(np.mean(emotional_rt) - np.mean(neutral_rt))

    def predict_from_pi_i(
        self,
        pi_i: float,
        baseline_interference: float = 100.0,
        sensitivity: float = 50.0,
    ) -> float:
        """Predict interference effect from interoceptive precision.

        Higher Πᵢ should predict greater emotional interference.

        Args:
            pi_i: Interoceptive precision
            baseline_interference: Baseline interference (ms)
            sensitivity: Sensitivity of interference to Πᵢ

        Returns:
            Predicted interference effect (ms)
        """
        return baseline_interference + sensitivity * pi_i


class ContinuousPerformanceTask:
    """Continuous Performance Task (CPT) for θ₀ validation.

    Tests whether baseline ignition threshold predicts attentional lapses
    and reaction time variability.
    """

    def __init__(self, duration_minutes: int = 10):
        """Initialize CPT.

        Args:
            duration_minutes: Task duration in minutes
        """
        self.duration_minutes = duration_minutes

    def run_task(self, participant_id: str, target_frequency: float = 0.2) -> TaskPerformance:
        """Run continuous performance task.

        Args:
            participant_id: Participant identifier
            target_frequency: Proportion of target trials

        Returns:
            TaskPerformance with results
        """
        # Simulate performance
        accuracy = np.random.uniform(0.75, 0.98)
        rt_mean = np.random.uniform(300, 500)
        rt_std = np.random.uniform(50, 150)

        # Lapse rate: proportion of very slow responses (> 2 SD)
        lapse_rate = np.random.uniform(0.01, 0.15)

        return TaskPerformance(
            participant_id=participant_id,
            task_name="continuous_performance",
            accuracy=accuracy,
            reaction_time_mean=rt_mean,
            reaction_time_std=rt_std,
            lapse_rate=lapse_rate,
        )

    def compute_lapse_rate(self, reaction_times: np.ndarray, threshold_sd: float = 2.0) -> float:
        """Compute attentional lapse rate.

        Args:
            reaction_times: Array of reaction times
            threshold_sd: Threshold in standard deviations

        Returns:
            Proportion of lapses
        """
        mean_rt = np.mean(reaction_times)
        std_rt = np.std(reaction_times)
        threshold = mean_rt + threshold_sd * std_rt

        lapses = np.sum(reaction_times > threshold)
        return float(lapses / len(reaction_times))

    def compute_rt_variability(self, reaction_times: np.ndarray) -> float:
        """Compute reaction time variability (coefficient of variation).

        Args:
            reaction_times: Array of reaction times

        Returns:
            Coefficient of variation
        """
        return float(np.std(reaction_times) / np.mean(reaction_times))

    def predict_from_theta0(
        self, theta0: float, baseline_lapse_rate: float = 0.05, sensitivity: float = 0.1
    ) -> float:
        """Predict lapse rate from baseline ignition threshold.

        Higher θ₀ (higher threshold) should predict more lapses.

        Args:
            theta0: Baseline ignition threshold
            baseline_lapse_rate: Baseline lapse rate
            sensitivity: Sensitivity of lapses to θ₀

        Returns:
            Predicted lapse rate
        """
        return baseline_lapse_rate + sensitivity * theta0


class BodyVigilanceScaleAnalyzer:
    """Body Vigilance Scale analyzer for β validation.

    Analyzes correlation between somatic bias (β) and validated somatic
    symptom questionnaires.
    """

    def __init__(self) -> None:
        """Initialize Body Vigilance Scale analyzer."""
        self.scale_items = [
            "attention_to_body_sensations",
            "sensitivity_to_changes",
            "monitoring_frequency",
            "concern_about_sensations",
        ]

    def compute_bvs_score(self, responses: Dict[str, float]) -> float:
        """Compute Body Vigilance Scale total score.

        Args:
            responses: Dictionary of item responses (0-10 scale)

        Returns:
            Total BVS score
        """
        return float(np.mean([responses[item] for item in self.scale_items]))

    def collect_questionnaire(self, participant_id: str) -> Dict[str, Union[str, float]]:
        """Collect Body Vigilance Scale questionnaire.

        Args:
            participant_id: Participant identifier

        Returns:
            Dictionary of responses
        """
        # Simulate questionnaire responses
        responses: Dict[str, Union[str, float]] = {
            item: np.random.uniform(0, 10) for item in self.scale_items
        }
        responses["participant_id"] = participant_id
        responses["total_score"] = self.compute_bvs_score(
            {k: float(v) for k, v in responses.items() if k in self.scale_items}
        )

        return responses

    def predict_from_beta(
        self, beta: float, baseline_bvs: float = 5.0, sensitivity: float = 3.0
    ) -> float:
        """Predict BVS score from somatic bias.

        Higher β should predict higher body vigilance.

        Args:
            beta: Somatic bias
            baseline_bvs: Baseline BVS score
            sensitivity: Sensitivity of BVS to β

        Returns:
            Predicted BVS score
        """
        return baseline_bvs + sensitivity * (beta - 1.0)


class PredictivePowerComparator:
    """Compares predictive power of APGI parameters against traditional measures."""

    @staticmethod
    def compute_correlation(
        predictor: np.ndarray, outcome: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute correlation and related statistics.

        Args:
            predictor: Predictor variable
            outcome: Outcome variable

        Returns:
            Tuple of (correlation, p_value, r_squared)
        """
        # Remove NaN values
        mask = ~(np.isnan(predictor) | np.isnan(outcome))
        predictor = predictor[mask]
        outcome = outcome[mask]

        if len(predictor) < 3:
            return 0.0, 1.0, 0.0

        r, p = stats.pearsonr(predictor, outcome)
        r_squared = r**2

        return float(r), float(p), float(r_squared)

    @staticmethod
    def classify_effect_size(r: float) -> str:
        """Classify effect size based on correlation.

        Args:
            r: Correlation coefficient

        Returns:
            Effect size classification
        """
        abs_r = abs(r)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "small"
        elif abs_r < 0.5:
            return "medium"
        else:
            return "large"

    @staticmethod
    def test_predictive_validity(
        parameter_estimates: List[ParameterEstimates],
        task_performance: List[TaskPerformance],
        parameter_name: str,
        performance_metric: str = "interference_effect",
    ) -> ValidityResult:
        """Test predictive validity of a parameter.

        Args:
            parameter_estimates: List of parameter estimates
            task_performance: List of task performance data
            parameter_name: Name of parameter to test ('theta0', 'pi_i', 'beta')
            performance_metric: Performance metric to predict

        Returns:
            ValidityResult
        """
        # Extract parameter values
        param_values: List[float] = []
        perf_values: List[Optional[float]] = []

        for est in parameter_estimates:
            # Find matching performance data
            matching_perf = [p for p in task_performance if p.participant_id == est.participant_id]

            if not matching_perf:
                continue

            perf = matching_perf[0]

            # Extract parameter value
            param_dist = getattr(est, parameter_name)
            param_values.append(param_dist.mean)

            # Extract performance metric
            if performance_metric == "interference_effect":
                perf_values.append(perf.interference_effect)
            elif performance_metric == "lapse_rate":
                perf_values.append(perf.lapse_rate)
            elif performance_metric == "rt_variability":
                perf_values.append(perf.reaction_time_std / perf.reaction_time_mean)
            else:
                perf_values.append(getattr(perf, performance_metric, np.nan))

        param_values_array = np.array(param_values)
        perf_values_array = np.array(perf_values)

        # Compute correlation
        r, p, r_squared = PredictivePowerComparator.compute_correlation(
            param_values_array, perf_values_array
        )

        effect_size = PredictivePowerComparator.classify_effect_size(r)

        return ValidityResult(
            parameter_name=parameter_name,
            task_name=task_performance[0].task_name if task_performance else "unknown",
            correlation=r,
            p_value=p,
            r_squared=r_squared,
            n_participants=len(param_values),
            significant=p < 0.05,
            effect_size=effect_size,
        )

    @staticmethod
    def compare_to_traditional_measures(
        apgi_parameter: np.ndarray,
        traditional_measure: np.ndarray,
        outcome: np.ndarray,
        apgi_name: str,
        traditional_name: str,
    ) -> ComparativeValidityResult:
        """Compare APGI parameter to traditional measure.

        Args:
            apgi_parameter: APGI parameter values
            traditional_measure: Traditional measure values
            outcome: Outcome variable
            apgi_name: Name of APGI parameter
            traditional_name: Name of traditional measure

        Returns:
            ComparativeValidityResult
        """
        # Compute correlations
        apgi_r, _, apgi_r2 = PredictivePowerComparator.compute_correlation(apgi_parameter, outcome)
        trad_r, _, trad_r2 = PredictivePowerComparator.compute_correlation(
            traditional_measure, outcome
        )

        # Determine which is better
        apgi_better = abs(apgi_r) > abs(trad_r)

        # Compute improvement percentage
        if abs(trad_r) > 0:
            improvement = ((abs(apgi_r) - abs(trad_r)) / abs(trad_r)) * 100
        else:
            improvement = 0.0

        return ComparativeValidityResult(
            apgi_parameter=apgi_name,
            traditional_measure=traditional_name,
            apgi_correlation=apgi_r,
            traditional_correlation=trad_r,
            apgi_r_squared=apgi_r2,
            traditional_r_squared=trad_r2,
            apgi_better=apgi_better,
            improvement_percentage=improvement,
        )

    @staticmethod
    def generate_comparison_report(comparisons: List[ComparativeValidityResult]) -> str:
        """Generate report comparing APGI to traditional measures.

        Args:
            comparisons: List of comparative results

        Returns:
            Report as string
        """
        report = []
        report.append("=" * 70)
        report.append("PREDICTIVE VALIDITY COMPARISON REPORT")
        report.append("=" * 70)
        report.append("")

        for comp in comparisons:
            symbol = "✓" if comp.apgi_better else "✗"
            report.append(f"{symbol} {comp.apgi_parameter} vs {comp.traditional_measure}")
            report.append(
                f"  APGI: r = {comp.apgi_correlation:.3f}, R² = {comp.apgi_r_squared:.3f}"
            )
            report.append(
                f"  Traditional: r = {comp.traditional_correlation:.3f}, "
                f"R² = {comp.traditional_r_squared:.3f}"
            )

            if comp.apgi_better:
                report.append(f"  Improvement: {comp.improvement_percentage:.1f}%")
            else:
                report.append("  Traditional measure performs better")
            report.append("")

        # Summary
        n_better = sum(1 for c in comparisons if c.apgi_better)
        report.append("-" * 70)
        report.append(
            f"Summary: APGI parameters outperformed traditional measures "
            f"in {n_better}/{len(comparisons)} comparisons"
        )
        report.append("=" * 70)

        return "\n".join(report)


class PredictiveValidityFramework:
    """Complete framework for testing predictive validity of APGI parameters."""

    def __init__(self) -> None:
        """Initialize predictive validity framework."""
        self.emotional_task = EmotionalInterferenceTask()
        self.cpt_task = ContinuousPerformanceTask()
        self.bvs_analyzer = BodyVigilanceScaleAnalyzer()
        self.comparator = PredictivePowerComparator()

    def run_complete_validation(
        self,
        parameter_estimates: List[ParameterEstimates],
        collect_behavioral_data: bool = True,
    ) -> Dict[str, ValidityResult]:
        """Run complete predictive validity testing.

        Args:
            parameter_estimates: List of parameter estimates
            collect_behavioral_data: Whether to collect new behavioral data

        Returns:
            Dictionary of validity results
        """
        results = {}

        # Test Πᵢ with emotional interference
        if collect_behavioral_data:
            emotional_performance = [
                self.emotional_task.run_task(est.participant_id) for est in parameter_estimates
            ]
        else:
            emotional_performance = []  # Would load from database

        if emotional_performance:
            results["pi_i_emotional"] = self.comparator.test_predictive_validity(
                parameter_estimates,
                emotional_performance,
                "pi_i",
                "interference_effect",
            )

        # Test θ₀ with CPT
        if collect_behavioral_data:
            cpt_performance = [
                self.cpt_task.run_task(est.participant_id) for est in parameter_estimates
            ]
        else:
            cpt_performance = []

        if cpt_performance:
            results["theta0_lapses"] = self.comparator.test_predictive_validity(
                parameter_estimates, cpt_performance, "theta0", "lapse_rate"
            )

        # Test β with BVS (questionnaire data)
        # This would typically be collected separately

        return results

    def generate_validity_report(
        self,
        validity_results: Dict[str, ValidityResult],
        comparative_results: Optional[List[ComparativeValidityResult]] = None,
    ) -> str:
        """Generate comprehensive validity report.

        Args:
            validity_results: Dictionary of validity results
            comparative_results: Optional comparative results

        Returns:
            Report as string
        """
        report = []
        report.append("=" * 70)
        report.append("PREDICTIVE VALIDITY REPORT")
        report.append("=" * 70)
        report.append("")

        # Individual validity results
        for key, result in validity_results.items():
            sig_symbol = (
                "***"
                if result.p_value < 0.001
                else ("**" if result.p_value < 0.01 else "*" if result.p_value < 0.05 else "ns")
            )

            report.append(f"{result.parameter_name} → {result.task_name}")
            report.append(f"  r = {result.correlation:.3f} {sig_symbol}")
            report.append(f"  R² = {result.r_squared:.3f}")
            report.append(f"  p = {result.p_value:.4f}")
            report.append(f"  Effect size: {result.effect_size}")
            report.append(f"  N = {result.n_participants}")
            report.append("")

        # Comparative results
        if comparative_results:
            report.append("-" * 70)
            report.append(self.comparator.generate_comparison_report(comparative_results))

        return "\n".join(report)
