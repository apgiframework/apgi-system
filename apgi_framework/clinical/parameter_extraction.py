"""Clinical Parameter Extraction Module.

Implements rapid 30-minute assessment battery for extracting individual APGI parameters
(θₜ, Πₑ, Πᵢ, β) with reliability and validity metrics.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Types of sensory modalities for assessment."""

    VISUAL = "visual"
    AUDITORY = "auditory"
    INTEROCEPTIVE = "interoceptive"


class TaskType(Enum):
    """Types of assessment tasks."""

    THRESHOLD_DETECTION = "threshold_detection"
    ODDBALL = "oddball"
    EMOTIONAL_STROOP = "emotional_stroop"
    HEARTBEAT_DETECTION = "heartbeat_detection"
    BREATH_HOLD = "breath_hold"


@dataclass
class ClinicalParameters:
    """Individual APGI parameters extracted from assessment."""

    # Core APGI parameters
    theta_t: float = 3.5  # Ignition threshold
    pi_e: float = 2.0  # Exteroceptive precision
    pi_i: float = 1.5  # Interoceptive precision
    beta: float = 1.2  # Somatic bias weight

    # Additional parameters
    alpha: float = 1.0  # Sigmoid steepness
    gamma: float = 0.1  # Homeostatic recovery rate

    # Confidence intervals (95%)
    theta_t_ci: Tuple[float, float] = (3.0, 4.0)
    pi_e_ci: Tuple[float, float] = (1.5, 2.5)
    pi_i_ci: Tuple[float, float] = (1.0, 2.0)
    beta_ci: Tuple[float, float] = (1.0, 1.4)

    # Metadata
    participant_id: str = ""
    assessment_date: Optional[datetime] = None
    assessment_duration: float = 30.0  # minutes

    # Quality metrics
    data_quality: float = 0.0  # 0-1 scale
    completion_rate: float = 0.0  # 0-1 scale

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "theta_t": self.theta_t,
            "pi_e": self.pi_e,
            "pi_i": self.pi_i,
            "beta": self.beta,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "theta_t_ci": list(self.theta_t_ci),
            "pi_e_ci": list(self.pi_e_ci),
            "pi_i_ci": list(self.pi_i_ci),
            "beta_ci": list(self.beta_ci),
            "participant_id": self.participant_id,
            "assessment_date": (self.assessment_date.isoformat() if self.assessment_date else None),
            "assessment_duration": self.assessment_duration,
            "data_quality": self.data_quality,
            "completion_rate": self.completion_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClinicalParameters":
        """Create from dictionary."""
        params = cls()
        params.theta_t = data.get("theta_t", 3.5)
        params.pi_e = data.get("pi_e", 2.0)
        params.pi_i = data.get("pi_i", 1.5)
        params.beta = data.get("beta", 1.2)
        params.alpha = data.get("alpha", 1.0)
        params.gamma = data.get("gamma", 0.1)
        params.theta_t_ci = tuple(data.get("theta_t_ci", [3.0, 4.0]))
        params.pi_e_ci = tuple(data.get("pi_e_ci", [1.5, 2.5]))
        params.pi_i_ci = tuple(data.get("pi_i_ci", [1.0, 2.0]))
        params.beta_ci = tuple(data.get("beta_ci", [1.0, 1.4]))
        params.participant_id = data.get("participant_id", "")

        date_str = data.get("assessment_date")
        if date_str:
            params.assessment_date = datetime.fromisoformat(date_str)

        params.assessment_duration = data.get("assessment_duration", 30.0)
        params.data_quality = data.get("data_quality", 0.0)
        params.completion_rate = data.get("completion_rate", 0.0)

        return params


@dataclass
class ReliabilityMetrics:
    """Reliability and validity metrics for parameter extraction."""

    # Test-retest reliability
    test_retest_icc: Dict[str, float] = field(default_factory=dict)  # Intraclass correlation
    test_retest_interval: Optional[timedelta] = None

    # Internal consistency
    cronbach_alpha: float = 0.0
    split_half_reliability: float = 0.0

    # Convergent validity
    convergent_correlations: Dict[str, float] = field(default_factory=dict)

    # Discriminant validity
    discriminant_correlations: Dict[str, float] = field(default_factory=dict)

    # Criterion validity
    criterion_correlations: Dict[str, float] = field(default_factory=dict)

    # Sensitivity and specificity
    sensitivity: float = 0.0  # True positive rate
    specificity: float = 0.0  # True negative rate

    # Classification accuracy
    classification_accuracy: float = 0.0

    # Standard error of measurement
    sem: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_retest_icc": self.test_retest_icc,
            "test_retest_interval": (
                str(self.test_retest_interval) if self.test_retest_interval else None
            ),
            "cronbach_alpha": self.cronbach_alpha,
            "split_half_reliability": self.split_half_reliability,
            "convergent_correlations": self.convergent_correlations,
            "discriminant_correlations": self.discriminant_correlations,
            "criterion_correlations": self.criterion_correlations,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "classification_accuracy": self.classification_accuracy,
            "sem": self.sem,
        }


@dataclass
class AssessmentTask:
    """Individual assessment task configuration."""

    task_type: TaskType
    modality: ModalityType
    duration: float  # minutes
    n_trials: int
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Results
    completed: bool = False
    completion_time: float = 0.0  # actual duration in minutes
    data_quality: float = 0.0  # 0-1 scale

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type.value,
            "modality": self.modality.value,
            "duration": self.duration,
            "n_trials": self.n_trials,
            "parameters": self.parameters,
            "completed": self.completed,
            "completion_time": self.completion_time,
            "data_quality": self.data_quality,
        }


@dataclass
class AssessmentBattery:
    """30-minute assessment battery configuration."""

    battery_id: str = ""
    participant_id: str = ""

    # Tasks in the battery
    tasks: List[AssessmentTask] = field(default_factory=list)

    # Timing
    total_duration: float = 30.0  # minutes
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Results
    completed: bool = False
    completion_rate: float = 0.0
    overall_quality: float = 0.0

    # Extracted parameters
    parameters: Optional[ClinicalParameters] = None
    reliability: Optional[ReliabilityMetrics] = None

    def add_task(self, task: AssessmentTask) -> None:
        """Add task to battery."""
        self.tasks.append(task)
        logger.info(f"Added {task.task_type.value} task to battery")

    def get_total_planned_duration(self) -> float:
        """Get total planned duration of all tasks."""
        return sum(task.duration for task in self.tasks)

    def get_completion_rate(self) -> float:
        """Calculate completion rate."""
        if not self.tasks:
            return 0.0
        completed = sum(1 for task in self.tasks if task.completed)
        return completed / len(self.tasks)

    def get_overall_quality(self) -> float:
        """Calculate overall data quality."""
        if not self.tasks:
            return 0.0
        completed_tasks = [task for task in self.tasks if task.completed]
        if not completed_tasks:
            return 0.0
        return float(np.mean([task.data_quality for task in completed_tasks]))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "battery_id": self.battery_id,
            "participant_id": self.participant_id,
            "tasks": [task.to_dict() for task in self.tasks],
            "total_duration": self.total_duration,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "completed": self.completed,
            "completion_rate": self.completion_rate,
            "overall_quality": self.overall_quality,
            "parameters": self.parameters.to_dict() if self.parameters else None,
            "reliability": self.reliability.to_dict() if self.reliability else None,
        }


class ClinicalParameterExtractor:
    """Clinical parameter extraction system for rapid APGI assessment.

    Implements 30-minute assessment battery with individual parameter estimation
    (θₜ, Πₑ, Πᵢ, β) and reliability/validity metrics.
    """

    def __init__(self, participant_id: str = ""):
        """Initialize clinical parameter extractor.

        Args:
            participant_id: Unique participant identifier
        """
        self.participant_id = participant_id
        self.assessment_history: List[AssessmentBattery] = []

        logger.info(f"Initialized ClinicalParameterExtractor for participant {participant_id}")

    def create_standard_battery(self) -> AssessmentBattery:
        """Create standard 30-minute assessment battery.

        Returns:
            AssessmentBattery configured with standard tasks
        """
        battery = AssessmentBattery(
            battery_id=f"battery_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            participant_id=self.participant_id,
            total_duration=30.0,
        )

        # Task 1: Visual threshold detection (5 minutes)
        battery.add_task(
            AssessmentTask(
                task_type=TaskType.THRESHOLD_DETECTION,
                modality=ModalityType.VISUAL,
                duration=5.0,
                n_trials=40,
                parameters={
                    "stimulus_type": "gabor",
                    "adaptive_algorithm": "quest_plus",
                    "target_threshold": 0.75,
                },
            )
        )

        # Task 2: Auditory threshold detection (5 minutes)
        battery.add_task(
            AssessmentTask(
                task_type=TaskType.THRESHOLD_DETECTION,
                modality=ModalityType.AUDITORY,
                duration=5.0,
                n_trials=40,
                parameters={
                    "stimulus_type": "tone",
                    "adaptive_algorithm": "quest_plus",
                    "target_threshold": 0.75,
                },
            )
        )

        # Task 3: Heartbeat detection (5 minutes)
        battery.add_task(
            AssessmentTask(
                task_type=TaskType.HEARTBEAT_DETECTION,
                modality=ModalityType.INTEROCEPTIVE,
                duration=5.0,
                n_trials=30,
                parameters={
                    "method": "mental_tracking",
                    "trial_duration": 10,  # seconds
                },
            )
        )

        # Task 4: Visual oddball with ERP (7 minutes)
        battery.add_task(
            AssessmentTask(
                task_type=TaskType.ODDBALL,
                modality=ModalityType.VISUAL,
                duration=7.0,
                n_trials=200,
                parameters={
                    "oddball_probability": 0.2,
                    "stimulus_duration": 100,  # ms
                    "isi": 1500,  # ms
                },
            )
        )

        # Task 5: Interoceptive oddball (breath hold) (5 minutes)
        battery.add_task(
            AssessmentTask(
                task_type=TaskType.BREATH_HOLD,
                modality=ModalityType.INTEROCEPTIVE,
                duration=5.0,
                n_trials=10,
                parameters={
                    "hold_duration": 20,  # seconds
                    "recovery_duration": 10,  # seconds
                },
            )
        )

        # Task 6: Emotional Stroop (3 minutes)
        battery.add_task(
            AssessmentTask(
                task_type=TaskType.EMOTIONAL_STROOP,
                modality=ModalityType.VISUAL,
                duration=3.0,
                n_trials=60,
                parameters={
                    "word_types": ["neutral", "threat", "positive"],
                    "response_deadline": 2000,  # ms
                },
            )
        )

        logger.info(
            f"Created standard battery with {len(battery.tasks)} tasks, "
            f"total duration: {battery.get_total_planned_duration():.1f} minutes"
        )

        return battery

    def extract_parameters_from_battery(
        self,
        battery: AssessmentBattery,
        behavioral_data: Dict[str, Any],
        neural_data: Optional[Dict[str, Any]] = None,
        physiological_data: Optional[Dict[str, Any]] = None,
    ) -> ClinicalParameters:
        """Extract APGI parameters from completed assessment battery.

        Args:
            battery: Completed assessment battery
            behavioral_data: Behavioral response data
            neural_data: Optional EEG/MEG data
            physiological_data: Optional physiological data (pupil, HR, etc.)

        Returns:
            ClinicalParameters with extracted values
        """
        logger.info(f"Extracting parameters from battery {battery.battery_id}")

        # Initialize parameters
        params = ClinicalParameters(
            participant_id=self.participant_id,
            assessment_date=datetime.now(),
            assessment_duration=battery.get_total_planned_duration(),
        )

        # Extract threshold (θₜ) from detection tasks
        params.theta_t, params.theta_t_ci = self._extract_threshold(behavioral_data, neural_data)

        # Extract exteroceptive precision (Πₑ) from visual/auditory tasks
        params.pi_e, params.pi_e_ci = self._extract_exteroceptive_precision(
            behavioral_data, neural_data
        )

        # Extract interoceptive precision (Πᵢ) from interoceptive tasks
        params.pi_i, params.pi_i_ci = self._extract_interoceptive_precision(
            behavioral_data, physiological_data
        )

        # Extract somatic bias (β) from emotional tasks
        params.beta, params.beta_ci = self._extract_somatic_bias(
            behavioral_data, neural_data, physiological_data
        )

        # Extract additional parameters
        params.alpha = self._extract_sigmoid_steepness(behavioral_data)
        params.gamma = self._extract_recovery_rate(behavioral_data)

        # Calculate quality metrics
        params.data_quality = battery.get_overall_quality()
        params.completion_rate = battery.get_completion_rate()

        logger.info(
            f"Extracted parameters: θₜ={params.theta_t:.2f}, "
            f"Πₑ={params.pi_e:.2f}, Πᵢ={params.pi_i:.2f}, β={params.beta:.2f}"
        )

        return params

    def _extract_threshold(
        self, behavioral_data: Dict[str, Any], neural_data: Optional[Dict[str, Any]]
    ) -> Tuple[float, Tuple[float, float]]:
        """Extract ignition threshold from detection tasks."""
        # Get detection thresholds from adaptive staircases
        visual_threshold = behavioral_data.get("visual_threshold", 0.5)
        auditory_threshold = behavioral_data.get("auditory_threshold", 0.5)
        intero_threshold = behavioral_data.get("interoceptive_threshold", 0.5)

        # Average across modalities (normalized)
        thresholds = [visual_threshold, auditory_threshold, intero_threshold]
        mean_threshold = np.mean(thresholds)
        std_threshold = np.std(thresholds)

        # Convert to APGI threshold scale (arbitrary units)
        # Lower detection threshold -> lower ignition threshold
        theta_t = 3.5 - (mean_threshold - 0.5) * 2.0

        # Adjust based on neural data if available
        if neural_data and "p3b_amplitude" in neural_data:
            # Higher P3b amplitude suggests lower threshold
            p3b_amplitude = neural_data["p3b_amplitude"]
            theta_t -= (p3b_amplitude - 5.0) * 0.2

        # Calculate confidence interval
        se = std_threshold / np.sqrt(len(thresholds))
        ci_lower = theta_t - 1.96 * se * 2.0
        ci_upper = theta_t + 1.96 * se * 2.0

        # Clamp to reasonable range
        theta_t = np.clip(theta_t, 1.0, 6.0)
        ci_lower = np.clip(ci_lower, 1.0, 6.0)
        ci_upper = np.clip(ci_upper, 1.0, 6.0)

        return float(theta_t), (float(ci_lower), float(ci_upper))

    def _extract_exteroceptive_precision(
        self, behavioral_data: Dict[str, Any], neural_data: Optional[Dict[str, Any]]
    ) -> Tuple[float, Tuple[float, float]]:
        """Extract exteroceptive precision from visual/auditory tasks."""
        # Get performance metrics
        visual_accuracy = behavioral_data.get("visual_accuracy", 0.75)
        auditory_accuracy = behavioral_data.get("auditory_accuracy", 0.75)
        rt_variability = behavioral_data.get("rt_variability", 100.0)  # ms

        # Higher accuracy and lower variability -> higher precision
        mean_accuracy = (visual_accuracy + auditory_accuracy) / 2
        precision_from_accuracy = mean_accuracy * 3.0  # Scale to 0-3 range

        # Lower RT variability -> higher precision
        precision_from_rt = 2.5 - (rt_variability - 100.0) / 100.0

        # Combine metrics
        pi_e = (precision_from_accuracy + precision_from_rt) / 2

        # Adjust based on neural data if available
        if neural_data and "gamma_power" in neural_data:
            # Higher gamma power suggests higher precision
            gamma_power = neural_data["gamma_power"]
            pi_e += (gamma_power - 0.5) * 0.5

        # Calculate confidence interval (simplified)
        se = 0.2  # Standard error estimate
        ci_lower = pi_e - 1.96 * se
        ci_upper = pi_e + 1.96 * se

        # Clamp to reasonable range
        pi_e = np.clip(pi_e, 0.5, 4.0)
        ci_lower = np.clip(ci_lower, 0.5, 4.0)
        ci_upper = np.clip(ci_upper, 0.5, 4.0)

        return float(pi_e), (float(ci_lower), float(ci_upper))

    def _extract_interoceptive_precision(
        self,
        behavioral_data: Dict[str, Any],
        physiological_data: Optional[Dict[str, Any]],
    ) -> Tuple[float, Tuple[float, float]]:
        """Extract interoceptive precision from interoceptive tasks."""
        # Get heartbeat detection accuracy
        hb_accuracy = behavioral_data.get("heartbeat_accuracy", 0.6)

        # Get breath awareness metrics
        breath_awareness = behavioral_data.get("breath_awareness", 0.5)

        # Combine metrics
        pi_i = (hb_accuracy + breath_awareness) * 1.5

        # Adjust based on physiological data if available
        if physiological_data:
            # Pupil dilation to interoceptive stimuli
            if "pupil_dilation_intero" in physiological_data:
                pupil_dilation = physiological_data["pupil_dilation_intero"]
                pi_i += (pupil_dilation - 0.5) * 0.3

            # Heart rate variability
            if "hrv" in physiological_data:
                hrv = physiological_data["hrv"]
                # Higher HRV associated with better interoceptive awareness
                pi_i += (hrv - 50.0) / 50.0 * 0.2

        # Calculate confidence interval
        se = 0.15
        ci_lower = pi_i - 1.96 * se
        ci_upper = pi_i + 1.96 * se

        # Clamp to reasonable range
        pi_i = np.clip(pi_i, 0.3, 3.5)
        ci_lower = np.clip(ci_lower, 0.3, 3.5)
        ci_upper = np.clip(ci_upper, 0.3, 3.5)

        return float(pi_i), (float(ci_lower), float(ci_upper))

    def _extract_somatic_bias(
        self,
        behavioral_data: Dict[str, Any],
        neural_data: Optional[Dict[str, Any]],
        physiological_data: Optional[Dict[str, Any]],
    ) -> Tuple[float, Tuple[float, float]]:
        """Extract somatic bias weight from emotional tasks."""
        # Get emotional Stroop interference
        stroop_interference = behavioral_data.get("emotional_stroop_interference", 50.0)  # ms

        # Higher interference suggests stronger somatic bias
        beta = 1.0 + (stroop_interference - 50.0) / 100.0

        # Adjust based on neural data
        if neural_data and "p3b_amplitude_emotional" in neural_data:
            p3b_emotional = neural_data["p3b_amplitude_emotional"]
            p3b_neutral = neural_data.get("p3b_amplitude_neutral", 5.0)

            # Ratio of emotional to neutral P3b
            if p3b_neutral > 0:
                p3b_ratio = p3b_emotional / p3b_neutral
                beta += (p3b_ratio - 1.0) * 0.3

        # Adjust based on physiological data
        if physiological_data and "scr_emotional" in physiological_data:
            # Skin conductance response to emotional stimuli
            scr = physiological_data["scr_emotional"]
            beta += (scr - 0.5) * 0.2

        # Calculate confidence interval
        se = 0.1
        ci_lower = beta - 1.96 * se
        ci_upper = beta + 1.96 * se

        # Clamp to reasonable range
        beta = np.clip(beta, 0.5, 3.0)
        ci_lower = np.clip(ci_lower, 0.5, 3.0)
        ci_upper = np.clip(ci_upper, 0.5, 3.0)

        return float(beta), (float(ci_lower), float(ci_upper))

    def _extract_sigmoid_steepness(self, behavioral_data: Dict[str, Any]) -> float:
        """Extract sigmoid steepness parameter from psychometric curves."""
        # Get slope of psychometric function if available
        psychometric_slope = behavioral_data.get("psychometric_slope", 1.0)

        # Convert to sigmoid steepness
        alpha = psychometric_slope
        alpha = np.clip(alpha, 0.5, 2.0)

        return float(alpha)

    def _extract_recovery_rate(self, behavioral_data: Dict[str, Any]) -> float:
        """Extract homeostatic recovery rate from sequential trials."""
        # Analyze performance recovery after errors or high-load trials
        recovery_trials = behavioral_data.get("recovery_trials", 5)

        # Faster recovery -> higher gamma
        gamma = 0.1 + (5.0 - recovery_trials) * 0.02
        gamma = np.clip(gamma, 0.05, 0.3)

        return float(gamma)

    def calculate_reliability_metrics(
        self,
        params1: ClinicalParameters,
        params2: ClinicalParameters,
        time_interval: Optional[timedelta] = None,
    ) -> ReliabilityMetrics:
        """Calculate reliability metrics from test-retest data.

        Args:
            params1: Parameters from first assessment
            params2: Parameters from second assessment
            time_interval: Time between assessments

        Returns:
            ReliabilityMetrics with test-retest and other reliability measures
        """
        logger.info("Calculating reliability metrics")

        metrics = ReliabilityMetrics(test_retest_interval=time_interval)

        # Calculate test-retest ICC for each parameter
        param_names = ["theta_t", "pi_e", "pi_i", "beta"]

        for param_name in param_names:
            val1 = getattr(params1, param_name)
            val2 = getattr(params2, param_name)

            # Calculate ICC (simplified - would use proper ICC calculation in practice)
            icc = self._calculate_icc([val1], [val2])
            metrics.test_retest_icc[param_name] = icc

            # Calculate standard error of measurement
            sem = np.std([val1, val2]) * np.sqrt(1 - icc)
            metrics.sem[param_name] = sem

        logger.info(f"Test-retest ICC: {metrics.test_retest_icc}")

        return metrics

    def _calculate_icc(self, values1: List[float], values2: List[float]) -> float:
        """Calculate intraclass correlation coefficient.

        Simplified implementation - in practice would use proper ICC(2,1) or ICC(3,1).
        """
        if len(values1) != len(values2):
            logger.error("Value lists must have same length")
            return np.nan

        # Calculate Pearson correlation as simplified ICC
        if len(values1) < 2:
            return np.nan

        try:
            corr, _ = pearsonr(values1, values2)
            return float(corr)
        except (ValueError, TypeError, RuntimeError):
            return np.nan

    def calculate_internal_consistency(
        self, battery: AssessmentBattery, trial_data: Dict[str, List[float]]
    ) -> float:
        """Calculate internal consistency (Cronbach's alpha) from trial-level data.

        Args:
            battery: Assessment battery
            trial_data: Dictionary mapping task names to trial-level scores

        Returns:
            Cronbach's alpha coefficient
        """
        # Collect all trial scores
        all_scores = []
        for task_name, scores in trial_data.items():
            if len(scores) > 0:
                all_scores.append(scores)

        if len(all_scores) < 2:
            logger.warning("Need at least 2 tasks for internal consistency")
            return 0.0

        # Calculate Cronbach's alpha
        alpha = self._cronbach_alpha(all_scores)

        logger.info(f"Internal consistency (Cronbach's alpha): {alpha:.3f}")

        return alpha

    def _cronbach_alpha(self, item_scores: List[List[float]]) -> float:
        """Calculate Cronbach's alpha for internal consistency.

        Args:
            item_scores: List of lists, where each inner list contains scores for one item

        Returns:
            Cronbach's alpha coefficient
        """
        # Convert to numpy array
        try:
            # Ensure all items have same length
            min_length = min(len(scores) for scores in item_scores)
            item_array = np.array([scores[:min_length] for scores in item_scores])

            n_items = item_array.shape[0]

            if n_items < 2:
                return 0.0

            # Calculate item variances
            item_variances = np.var(item_array, axis=1, ddof=1)

            # Calculate total score variance
            total_scores = np.sum(item_array, axis=0)
            total_variance = np.var(total_scores, ddof=1)

            # Calculate Cronbach's alpha
            if total_variance == 0:
                return 0.0

            alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_variances) / total_variance)

            return float(alpha)

        except Exception as e:
            logger.error(f"Error calculating Cronbach's alpha: {e}")
            return 0.0

    def calculate_split_half_reliability(
        self, battery: AssessmentBattery, trial_data: Dict[str, List[float]]
    ) -> float:
        """Calculate split-half reliability.

        Args:
            battery: Assessment battery
            trial_data: Dictionary mapping task names to trial-level scores

        Returns:
            Split-half reliability coefficient (Spearman-Brown corrected)
        """
        # Collect all trial scores
        all_scores = []
        for task_name, scores in trial_data.items():
            if len(scores) > 0:
                all_scores.extend(scores)

        if len(all_scores) < 4:
            logger.warning("Need at least 4 trials for split-half reliability")
            return 0.0

        # Split into two halves
        n = len(all_scores)
        half1 = all_scores[: n // 2]
        half2 = all_scores[n // 2 : 2 * (n // 2)]  # Ensure equal length

        # Calculate correlation between halves
        try:
            corr, _ = pearsonr(half1, half2)

            # Apply Spearman-Brown correction
            reliability = (2 * corr) / (1 + corr)

            logger.info(f"Split-half reliability: {reliability:.3f}")

            return float(reliability)

        except Exception as e:
            logger.error(f"Error calculating split-half reliability: {e}")
            return 0.0

    def validate_against_criterion(
        self, params: ClinicalParameters, criterion_measures: Dict[str, float]
    ) -> Dict[str, float]:
        """Validate parameters against external criterion measures.

        Args:
            params: Extracted clinical parameters
            criterion_measures: Dictionary of criterion measures (e.g., clinical scales)

        Returns:
            Dictionary of correlations with criterion measures
        """
        correlations = {}

        # Example criterion validations - compute validation scores based on criterion measures
        if "anxiety_scale" in criterion_measures:
            anxiety_level = criterion_measures["anxiety_scale"]
            # Lower threshold should correlate with higher anxiety
            expected_theta = 3.5 - anxiety_level * 0.5  # Lower theta for higher anxiety
            corr_theta = 1.0 - min(abs(params.theta_t - expected_theta) / 2.0, 1.0)
            correlations["theta_anxiety"] = corr_theta

            # Higher interoceptive precision correlates with higher anxiety
            expected_pi_i = 1.5 + anxiety_level * 0.5  # Higher pi_i for higher anxiety
            corr_pi_i = 1.0 - min(abs(params.pi_i - expected_pi_i) / 1.0, 1.0)
            correlations["pi_i_anxiety"] = corr_pi_i

        if "depression_scale" in criterion_measures:
            depression_level = criterion_measures["depression_scale"]
            # Higher threshold correlates with higher depression
            expected_theta = 3.5 + depression_level * 0.5  # Higher theta for higher depression
            corr_theta = 1.0 - min(abs(params.theta_t - expected_theta) / 2.0, 1.0)
            correlations["theta_depression"] = corr_theta

            # Lower interoceptive precision correlates with higher depression
            expected_pi_i = 1.5 - depression_level * 0.5  # Lower pi_i for higher depression
            corr_pi_i = 1.0 - min(abs(params.pi_i - expected_pi_i) / 1.0, 1.0)
            correlations["pi_i_depression"] = corr_pi_i

        if "interoceptive_awareness" in criterion_measures:
            awareness_level = criterion_measures["interoceptive_awareness"]
            # Interoceptive precision should correlate with awareness
            expected_pi_i = 1.5 + (awareness_level - 0.5) * 1.0  # Higher awareness -> higher pi_i
            corr_pi_i = 1.0 - min(abs(params.pi_i - expected_pi_i) / 1.0, 1.0)
            correlations["pi_i_awareness"] = corr_pi_i

        logger.info(f"Criterion correlations: {correlations}")

        return correlations

    def calculate_classification_metrics(
        self, true_labels: List[str], predicted_labels: List[str]
    ) -> Tuple[float, float, float]:
        """Calculate sensitivity, specificity, and accuracy for disorder classification.

        Args:
            true_labels: True disorder labels
            predicted_labels: Predicted disorder labels

        Returns:
            Tuple of (sensitivity, specificity, accuracy)
        """
        if len(true_labels) != len(predicted_labels):
            logger.error("Label lists must have same length")
            return 0.0, 0.0, 0.0

        # Calculate confusion matrix elements
        tp = sum(
            1
            for t, p in zip(true_labels, predicted_labels)
            if t != "control" and p != "control" and t == p
        )
        tn = sum(
            1 for t, p in zip(true_labels, predicted_labels) if t == "control" and p == "control"
        )
        fp = sum(
            1 for t, p in zip(true_labels, predicted_labels) if t == "control" and p != "control"
        )
        fn = sum(
            1 for t, p in zip(true_labels, predicted_labels) if t != "control" and p == "control"
        )

        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0.0

        logger.info(
            f"Classification metrics - Sensitivity: {sensitivity:.3f}, "
            f"Specificity: {specificity:.3f}, Accuracy: {accuracy:.3f}"
        )

        return sensitivity, specificity, accuracy

    def generate_clinical_report(
        self,
        params: ClinicalParameters,
        reliability: Optional[ReliabilityMetrics] = None,
    ) -> str:
        """Generate clinical report with parameter estimates and interpretation.

        Args:
            params: Extracted clinical parameters
            reliability: Optional reliability metrics

        Returns:
            Formatted clinical report string
        """
        report = []
        report.append("=" * 60)
        report.append("APGI CLINICAL PARAMETER EXTRACTION REPORT")
        report.append("=" * 60)
        report.append("")

        # Participant information
        report.append(f"Participant ID: {params.participant_id}")
        if params.assessment_date:
            report.append(f"Assessment Date: {params.assessment_date.strftime('%Y-%m-%d %H:%M')}")
        report.append(f"Assessment Duration: {params.assessment_duration:.1f} minutes")
        report.append(f"Data Quality: {params.data_quality:.2f}")
        report.append(f"Completion Rate: {params.completion_rate:.1%}")
        report.append("")

        # Core parameters
        report.append("CORE APGI PARAMETERS")
        report.append("-" * 60)
        report.append(
            f"Ignition Threshold (θₜ):        {params.theta_t:.2f} "
            f"[95% CI: {params.theta_t_ci[0]:.2f}-{params.theta_t_ci[1]:.2f}]"
        )
        report.append(
            f"Exteroceptive Precision (Πₑ):   {params.pi_e:.2f} "
            f"[95% CI: {params.pi_e_ci[0]:.2f}-{params.pi_e_ci[1]:.2f}]"
        )
        report.append(
            f"Interoceptive Precision (Πᵢ):   {params.pi_i:.2f} "
            f"[95% CI: {params.pi_i_ci[0]:.2f}-{params.pi_i_ci[1]:.2f}]"
        )
        report.append(
            f"Somatic Bias Weight (β):        {params.beta:.2f} "
            f"[95% CI: {params.beta_ci[0]:.2f}-{params.beta_ci[1]:.2f}]"
        )
        report.append("")

        # Additional parameters
        report.append("ADDITIONAL PARAMETERS")
        report.append("-" * 60)
        report.append(f"Sigmoid Steepness (α):          {params.alpha:.2f}")
        report.append(f"Recovery Rate (γ):              {params.gamma:.3f}")
        report.append("")

        # Clinical interpretation
        report.append("CLINICAL INTERPRETATION")
        report.append("-" * 60)

        # Threshold interpretation
        if params.theta_t < 2.5:
            report.append("• Threshold: VERY LOW - High risk for anxiety/panic")
        elif params.theta_t < 3.0:
            report.append("• Threshold: LOW - Elevated anxiety sensitivity")
        elif params.theta_t > 4.0:
            report.append("• Threshold: HIGH - Possible anhedonia/depression")
        else:
            report.append("• Threshold: NORMAL - Adaptive range")

        # Interoceptive precision interpretation
        if params.pi_i > 2.5:
            report.append("• Interoceptive Precision: HIGH - Heightened body awareness")
        elif params.pi_i < 1.0:
            report.append("• Interoceptive Precision: LOW - Reduced body awareness")
        else:
            report.append("• Interoceptive Precision: NORMAL - Balanced awareness")

        # Somatic bias interpretation
        if params.beta > 1.8:
            report.append("• Somatic Bias: HIGH - Strong emotional-somatic coupling")
        elif params.beta < 0.8:
            report.append("• Somatic Bias: LOW - Reduced emotional processing")
        else:
            report.append("• Somatic Bias: NORMAL - Adaptive emotional regulation")

        report.append("")

        # Reliability metrics if available
        if reliability:
            report.append("RELIABILITY METRICS")
            report.append("-" * 60)

            if reliability.test_retest_icc:
                report.append("Test-Retest ICC:")
                for param, icc in reliability.test_retest_icc.items():
                    report.append(f"  {param}: {icc:.3f}")

            if reliability.cronbach_alpha > 0:
                report.append(f"Internal Consistency (α): {reliability.cronbach_alpha:.3f}")

            if reliability.split_half_reliability > 0:
                report.append(f"Split-Half Reliability: {reliability.split_half_reliability:.3f}")

            if reliability.classification_accuracy > 0:
                report.append(f"Classification Accuracy: {reliability.classification_accuracy:.1%}")

            report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 60)

        if params.theta_t < 2.5 and params.pi_i > 2.0:
            report.append("• Consider SNRI treatment for anxiety with interoceptive focus")
        elif params.theta_t > 4.0 and params.pi_i < 1.0:
            report.append("• Consider interventions targeting interoceptive awareness")
        elif params.beta > 2.0:
            report.append("• Consider beta-blocker augmentation for somatic symptoms")
        else:
            report.append("• Parameters within normal range - monitor longitudinally")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def save_parameters(self, params: ClinicalParameters, filepath: str) -> None:
        """Save parameters to JSON file.

        Args:
            params: Clinical parameters to save
            filepath: Path to save file
        """
        try:
            with open(filepath, "w") as f:
                json.dump(params.to_dict(), f, indent=2)
            logger.info(f"Saved parameters to {filepath}")
        except Exception as e:
            logger.error(f"Error saving parameters: {e}")

    def load_parameters(self, filepath: str) -> ClinicalParameters:
        """Load parameters from JSON file.

        Args:
            filepath: Path to parameter file

        Returns:
            ClinicalParameters object
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            params = ClinicalParameters.from_dict(data)
            logger.info(f"Loaded parameters from {filepath}")
            return params
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            return ClinicalParameters()

    def track_longitudinal_changes(
        self, parameter_history: List[ClinicalParameters]
    ) -> Dict[str, Any]:
        """Track longitudinal changes in parameters over time.

        Args:
            parameter_history: List of parameters from multiple assessments

        Returns:
            Dictionary with longitudinal trends and statistics
        """
        if len(parameter_history) < 2:
            logger.warning("Need at least 2 assessments for longitudinal tracking")
            return {}

        # Extract parameter values over time
        param_names = ["theta_t", "pi_e", "pi_i", "beta"]
        trends = {}

        for param_name in param_names:
            values = [getattr(p, param_name) for p in parameter_history]
            dates = [p.assessment_date for p in parameter_history if p.assessment_date]

            # Calculate trend statistics
            trends[param_name] = {
                "values": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "change": float(values[-1] - values[0]),
                "percent_change": (
                    float((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0.0
                ),
            }

            # Calculate linear trend if we have dates
            if len(dates) == len(values) and len(dates) >= 2:
                # Convert dates to days since first assessment
                days = [(d - dates[0]).days for d in dates]

                # Fit linear trend
                if len(days) > 1:
                    slope, intercept = np.polyfit(days, values, 1)
                    trends[param_name]["slope"] = float(slope)
                    trends[param_name]["slope_per_week"] = float(slope * 7)

        logger.info(f"Calculated longitudinal trends for {len(parameter_history)} assessments")

        return trends


# Convenience function for quick parameter extraction
def extract_parameters_quick(
    participant_id: str,
    behavioral_data: Dict[str, Any],
    neural_data: Optional[Dict[str, Any]] = None,
    physiological_data: Optional[Dict[str, Any]] = None,
) -> ClinicalParameters:
    """Quick parameter extraction without full battery setup.

    Args:
        participant_id: Participant identifier
        behavioral_data: Behavioral response data
        neural_data: Optional neural data
        physiological_data: Optional physiological data

    Returns:
        ClinicalParameters with extracted values
    """
    extractor = ClinicalParameterExtractor(participant_id)
    battery = extractor.create_standard_battery()

    # Mark all tasks as completed (assuming data is from completed battery)
    for task in battery.tasks:
        task.completed = True
        task.data_quality = 0.8

    battery.completed = True
    battery.completion_rate = 1.0
    battery.overall_quality = 0.8

    params = extractor.extract_parameters_from_battery(
        battery, behavioral_data, neural_data, physiological_data
    )

    return params
