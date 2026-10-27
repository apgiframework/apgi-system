"""Parameter estimation specific data models for the APGI Framework.

Defines data structures for parameter estimation tasks including behavioral responses,
trial results, parameter estimates, and quality metrics for the three core tasks:
- Detection task (θ₀ estimation)
- Heartbeat detection task (Πᵢ estimation)
- Dual-modality oddball task (β estimation)
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TaskType(Enum):
    """Types of parameter estimation tasks."""

    DETECTION = "detection"
    HEARTBEAT_DETECTION = "heartbeat_detection"
    DUAL_MODALITY_ODDBALL = "dual_modality_oddball"


class StimulusModality(Enum):
    """Stimulus modalities for tasks."""

    VISUAL = "visual"
    AUDITORY = "auditory"
    INTEROCEPTIVE = "interoceptive"
    EXTEROCEPTIVE = "exteroceptive"


@dataclass
class BehavioralResponse:
    """Behavioral response data for parameter estimation tasks."""

    response_time: float  # milliseconds
    detected: bool
    confidence: Optional[float] = None  # 0-1 scale
    response_key: str = ""
    reaction_time_valid: bool = True

    def validate(self) -> bool:
        """Validate behavioral response data."""
        if self.response_time < 0:
            return False
        if self.confidence is not None and not (0 <= self.confidence <= 1):
            return False
        return True


@dataclass
class QualityMetrics:
    """Data quality metrics for multi-modal recordings."""

    eeg_artifact_ratio: float = 0.0  # 0-1, proportion of trials with artifacts
    pupil_data_loss: float = 0.0  # 0-1, proportion of missing pupil data
    cardiac_signal_quality: float = 1.0  # 0-1, cardiac signal quality score
    overall_quality_score: float = 1.0  # 0-1, composite quality score

    # EEG specific metrics
    electrode_impedances: Dict[str, float] = field(default_factory=dict)  # kΩ
    bad_channels: List[str] = field(default_factory=list)

    # Pupillometry specific metrics
    blink_rate: float = 0.0  # blinks per minute
    tracking_loss_episodes: int = 0

    # Cardiac specific metrics
    r_peak_detection_confidence: float = 1.0  # 0-1
    heart_rate_variability: float = 0.0  # RMSSD in ms

    def validate(self) -> bool:
        """Validate quality metrics."""
        metrics = [
            self.eeg_artifact_ratio,
            self.pupil_data_loss,
            self.cardiac_signal_quality,
            self.overall_quality_score,
        ]
        return all(0 <= metric <= 1 for metric in metrics)


@dataclass
class TrialData:
    """Core trial data structure for parameter estimation tasks."""

    trial_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    participant_id: str = ""
    session_id: str = ""
    task_type: TaskType = TaskType.DETECTION
    trial_number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    # Stimulus parameters
    stimulus_parameters: Dict[str, Any] = field(default_factory=dict)
    stimulus_modality: StimulusModality = StimulusModality.VISUAL
    stimulus_intensity: float = 0.0

    # Response data
    behavioral_response: Optional[BehavioralResponse] = None

    # Physiological data (optional, stored as references or arrays)
    eeg_data: Optional[np.ndarray] = None
    pupil_data: Optional[np.ndarray] = None
    cardiac_data: Optional[np.ndarray] = None

    # Quality assessment
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate trial data."""
        if not self.trial_id or not self.participant_id:
            return False
        if self.behavioral_response and not self.behavioral_response.validate():
            return False
        if not self.quality_metrics.validate():
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trial_id": self.trial_id,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "task_type": self.task_type.value,
            "trial_number": self.trial_number,
            "timestamp": self.timestamp.isoformat(),
            "stimulus_parameters": self.stimulus_parameters,
            "stimulus_modality": self.stimulus_modality.value,
            "stimulus_intensity": self.stimulus_intensity,
            "behavioral_response": (
                self.behavioral_response.__dict__ if self.behavioral_response else None
            ),
            "quality_metrics": self.quality_metrics.__dict__,
            "metadata": self.metadata,
        }


@dataclass
class DetectionTrialResult(TrialData):
    """Specific trial result for detection task (θ₀ estimation)."""

    task_type: TaskType = field(default=TaskType.DETECTION, init=False)

    # Detection-specific data
    gabor_orientation: Optional[float] = None  # degrees
    tone_frequency: Optional[float] = None  # Hz
    contrast_level: float = 0.0  # 0-1

    # Neural markers
    p3b_amplitude: Optional[float] = None  # μV at Pz electrode
    p3b_latency: Optional[float] = None  # ms

    # Adaptive staircase data
    staircase_intensity: float = 0.0
    staircase_reversals: int = 0

    def validate(self) -> bool:
        """Validate detection trial data."""
        if not super().validate():
            return False
        if not (0 <= self.contrast_level <= 1):
            return False
        return True


@dataclass
class HeartbeatTrialResult(TrialData):
    """Specific trial result for heartbeat detection task (Πᵢ estimation)."""

    task_type: TaskType = field(default=TaskType.HEARTBEAT_DETECTION, init=False)

    # Heartbeat-specific data
    is_synchronous: bool = False
    tone_delay_ms: float = 0.0  # delay from R-peak
    r_peak_timestamp: Optional[datetime] = None

    # Cardiac measurements
    heart_rate: float = 0.0  # BPM
    rr_interval: float = 0.0  # ms

    # Neural markers
    hep_amplitude: Optional[float] = None  # μV, heartbeat-evoked potential
    interoceptive_p3b: Optional[float] = None  # μV

    # Pupillometry
    pupil_baseline: Optional[float] = None  # mm
    pupil_dilation_peak: Optional[float] = None  # mm
    pupil_time_to_peak: Optional[float] = None  # ms

    def validate(self) -> bool:
        """Validate heartbeat trial data."""
        if not super().validate():
            return False
        if self.heart_rate < 0 or self.heart_rate > 200:
            return False
        return True


@dataclass
class OddballTrialResult(TrialData):
    """Specific trial result for dual-modality oddball task (β estimation)."""

    task_type: TaskType = field(default=TaskType.DUAL_MODALITY_ODDBALL, init=False)

    # Oddball-specific data
    is_deviant: bool = False
    deviant_type: Optional[str] = None  # 'interoceptive' or 'exteroceptive'

    # Interoceptive stimuli
    co2_puff_duration: Optional[float] = None  # ms
    co2_concentration: Optional[float] = None  # %
    heartbeat_flash_delay: Optional[float] = None  # ms from R-peak

    # Exteroceptive stimuli
    gabor_orientation_deviation: Optional[float] = None  # degrees from standard
    auditory_deviant_frequency: Optional[float] = None  # Hz

    # Neural responses
    interoceptive_p3b: Optional[float] = None  # μV
    exteroceptive_p3b: Optional[float] = None  # μV
    p3b_ratio: Optional[float] = None  # interoceptive/exteroceptive

    # Precision matching
    interoceptive_precision: Optional[float] = None  # Πᵢ
    exteroceptive_precision: Optional[float] = None  # Πₑ

    def validate(self) -> bool:
        """Validate oddball trial data."""
        if not super().validate():
            return False
        if self.is_deviant and not self.deviant_type:
            return False
        if self.deviant_type and self.deviant_type not in [
            "interoceptive",
            "exteroceptive",
        ]:
            return False
        return True


@dataclass
class ParameterDistribution:
    """Statistical distribution for a parameter estimate."""

    mean: float
    std: float
    credible_interval_95: Tuple[float, float]
    posterior_samples: Optional[np.ndarray] = None

    # Convergence diagnostics
    r_hat: Optional[float] = None  # Gelman-Rubin statistic
    effective_sample_size: Optional[int] = None

    def validate(self) -> bool:
        """Validate parameter distribution."""
        if self.std < 0:
            return False
        if self.credible_interval_95[0] > self.credible_interval_95[1]:
            return False
        if self.r_hat is not None and self.r_hat > 1.1:
            return False  # Poor convergence
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mean": self.mean,
            "std": self.std,
            "credible_interval_95": self.credible_interval_95,
            "r_hat": self.r_hat,
            "effective_sample_size": self.effective_sample_size,
        }


@dataclass
class ModelFitMetrics:
    """Model fitting quality metrics."""

    log_likelihood: float = 0.0
    aic: float = 0.0  # Akaike Information Criterion
    bic: float = 0.0  # Bayesian Information Criterion
    waic: float = 0.0  # Watanabe-Akaike Information Criterion

    # Convergence metrics
    chains_converged: bool = False
    max_r_hat: float = 1.0
    min_effective_sample_size: int = 0

    # Cross-validation
    loo_cv_score: Optional[float] = None  # Leave-one-out cross-validation

    def validate(self) -> bool:
        """Validate model fit metrics."""
        return self.max_r_hat >= 1.0 and self.min_effective_sample_size >= 0


@dataclass
class ReliabilityMetrics:
    """Test-retest reliability metrics."""

    icc_theta0: Optional[float] = None  # Intraclass correlation for θ₀
    icc_pi_i: Optional[float] = None  # Intraclass correlation for Πᵢ
    icc_beta: Optional[float] = None  # Intraclass correlation for β

    test_retest_interval_days: Optional[int] = None
    n_participants_reliability: Optional[int] = None

    def validate(self) -> bool:
        """Validate reliability metrics."""
        iccs = [self.icc_theta0, self.icc_pi_i, self.icc_beta]
        return all(icc is None or 0 <= icc <= 1 for icc in iccs)


@dataclass
class ParameterEstimates:
    """Complete parameter estimates for an individual."""

    participant_id: str
    session_id: str

    # Core APGI parameters
    theta0: ParameterDistribution  # Baseline ignition threshold
    pi_i: ParameterDistribution  # Interoceptive precision
    beta: ParameterDistribution  # Somatic bias

    estimation_timestamp: datetime = field(default_factory=datetime.now)

    # Model quality
    model_fit_metrics: ModelFitMetrics = field(default_factory=ModelFitMetrics)
    reliability_metrics: ReliabilityMetrics = field(default_factory=ReliabilityMetrics)

    # Task performance summaries
    detection_accuracy: Optional[float] = None  # Proportion correct
    heartbeat_d_prime: Optional[float] = None  # d' for heartbeat detection
    oddball_discrimination: Optional[float] = None  # d' for oddball detection

    # Data quality summary
    overall_data_quality: float = 1.0  # 0-1 composite score
    n_trials_used: Dict[str, int] = field(default_factory=dict)  # trials per task
    n_trials_excluded: Dict[str, int] = field(default_factory=dict)  # excluded trials per task

    def validate(self) -> bool:
        """Validate parameter estimates."""
        if not self.participant_id or not self.session_id:
            return False

        parameters = [self.theta0, self.pi_i, self.beta]
        if not all(param.validate() for param in parameters):
            return False

        if not self.model_fit_metrics.validate():
            return False

        if not self.reliability_metrics.validate():
            return False

        if not (0 <= self.overall_data_quality <= 1):
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "estimation_timestamp": self.estimation_timestamp.isoformat(),
            "theta0": self.theta0.to_dict(),
            "pi_i": self.pi_i.to_dict(),
            "beta": self.beta.to_dict(),
            "model_fit_metrics": self.model_fit_metrics.__dict__,
            "reliability_metrics": self.reliability_metrics.__dict__,
            "detection_accuracy": self.detection_accuracy,
            "heartbeat_d_prime": self.heartbeat_d_prime,
            "oddball_discrimination": self.oddball_discrimination,
            "overall_data_quality": self.overall_data_quality,
            "n_trials_used": self.n_trials_used,
            "n_trials_excluded": self.n_trials_excluded,
        }


@dataclass
class SessionData:
    """Complete session data for parameter estimation."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    participant_id: str = ""
    session_date: datetime = field(default_factory=datetime.now)
    protocol_version: str = "1.0.0"

    # Session status
    completion_status: str = "in_progress"  # in_progress, completed, aborted
    total_duration_minutes: Optional[float] = None

    # Trial data
    detection_trials: List[DetectionTrialResult] = field(default_factory=list)
    heartbeat_trials: List[HeartbeatTrialResult] = field(default_factory=list)
    oddball_trials: List[OddballTrialResult] = field(default_factory=list)

    # Parameter estimates (computed after session)
    parameter_estimates: Optional[ParameterEstimates] = None

    # Session-level quality metrics
    session_quality_score: float = 1.0
    technical_issues: List[str] = field(default_factory=list)

    # Metadata
    researcher: str = ""
    notes: str = ""

    def get_all_trials(self) -> List[TrialData]:
        """Get all trials from session."""
        all_trials: List[TrialData] = []
        all_trials.extend(self.detection_trials)
        all_trials.extend(self.heartbeat_trials)
        all_trials.extend(self.oddball_trials)
        return all_trials

    def get_trial_count_by_task(self) -> Dict[str, int]:
        """Get trial counts by task type."""
        return {
            "detection": len(self.detection_trials),
            "heartbeat_detection": len(self.heartbeat_trials),
            "dual_modality_oddball": len(self.oddball_trials),
        }

    def validate(self) -> bool:
        """Validate session data."""
        if not self.session_id or not self.participant_id:
            return False

        # Validate all trials
        all_trials = self.get_all_trials()
        if not all(trial.validate() for trial in all_trials):
            return False

        # Validate parameter estimates if present
        if self.parameter_estimates and not self.parameter_estimates.validate():
            return False

        if not (0 <= self.session_quality_score <= 1):
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "session_date": self.session_date.isoformat(),
            "protocol_version": self.protocol_version,
            "completion_status": self.completion_status,
            "total_duration_minutes": self.total_duration_minutes,
            "detection_trials": [trial.to_dict() for trial in self.detection_trials],
            "heartbeat_trials": [trial.to_dict() for trial in self.heartbeat_trials],
            "oddball_trials": [trial.to_dict() for trial in self.oddball_trials],
            "parameter_estimates": (
                self.parameter_estimates.to_dict() if self.parameter_estimates else None
            ),
            "session_quality_score": self.session_quality_score,
            "technical_issues": self.technical_issues,
            "researcher": self.researcher,
            "notes": self.notes,
        }
