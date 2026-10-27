"""QUEST+ adaptive staircase implementation for optimal stimulus selection.

Implements the QUEST+ algorithm for efficient threshold estimation in
detection tasks, with state management and convergence monitoring.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)


@dataclass
class QuestPlusParameters:
    """Parameters for QUEST+ algorithm configuration."""

    # Stimulus parameter space
    stimulus_min: float = 0.01  # Minimum stimulus intensity
    stimulus_max: float = 1.0  # Maximum stimulus intensity
    stimulus_steps: int = 50  # Number of stimulus levels to test

    # Psychometric function parameters
    threshold_min: float = 0.01  # Minimum threshold
    threshold_max: float = 1.0  # Maximum threshold
    threshold_steps: int = 40  # Number of threshold values in prior

    slope_min: float = 1.0  # Minimum psychometric function slope
    slope_max: float = 10.0  # Maximum psychometric function slope
    slope_steps: int = 20  # Number of slope values in prior

    lapse_rate: float = 0.02  # Lapse rate (0-1)
    guess_rate: float = 0.5  # Guess rate for 2AFC tasks

    # Prior distributions (uniform by default)
    threshold_prior: Optional[np.ndarray] = None
    slope_prior: Optional[np.ndarray] = None

    # Convergence criteria
    min_trials: int = 20  # Minimum trials before convergence check
    max_trials: int = 200  # Maximum trials
    convergence_criterion: float = 0.05  # Threshold change criterion
    min_reversals: int = 4  # Minimum reversals for convergence

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Validate stimulus ranges
        if self.stimulus_min >= self.stimulus_max:
            raise ValueError("stimulus_min must be less than stimulus_max")

        # Validate threshold ranges
        if self.threshold_min >= self.threshold_max:
            raise ValueError("threshold_min must be less than threshold_max")

        # Validate slope ranges
        if self.slope_min >= self.slope_max:
            raise ValueError("slope_min must be less than slope_max")

        # Validate steps
        if self.stimulus_steps <= 0:
            raise ValueError("stimulus_steps must be positive")

        if self.threshold_steps <= 0:
            raise ValueError("threshold_steps must be positive")

        if self.slope_steps <= 0:
            raise ValueError("slope_steps must be positive")

        # Validate rates
        if not 0 <= self.lapse_rate <= 1:
            raise ValueError("lapse_rate must be between 0 and 1")

        if not 0 <= self.guess_rate <= 1:
            raise ValueError("guess_rate must be between 0 and 1")

        # Validate convergence criteria
        if self.convergence_criterion <= 0:
            raise ValueError("convergence_criterion must be positive")

        # Validate trial counts
        if self.min_trials <= 0:
            raise ValueError("min_trials must be positive")

        if self.max_trials <= 0:
            raise ValueError("max_trials must be positive")

        if self.min_reversals <= 0:
            raise ValueError("min_reversals must be positive")

        # Validate prior distributions
        if self.threshold_prior is not None:
            if not isinstance(self.threshold_prior, np.ndarray):
                raise ValueError("threshold_prior must be a numpy array")

            if len(self.threshold_prior) != self.threshold_steps:
                raise ValueError(f"threshold_prior must have length {self.threshold_steps}")

            if not np.allclose(np.sum(self.threshold_prior), 1.0):
                raise ValueError("threshold_prior must sum to 1 (normalized)")

        if self.slope_prior is not None:
            if not isinstance(self.slope_prior, np.ndarray):
                raise ValueError("slope_prior must be a numpy array")

            if len(self.slope_prior) != self.slope_steps:
                raise ValueError(f"slope_prior must have length {self.slope_steps}")

            if not np.allclose(np.sum(self.slope_prior), 1.0):
                raise ValueError("slope_prior must sum to 1 (normalized)")


@dataclass
class StaircaseState:
    """Current state of the adaptive staircase."""

    trial_number: int = 0
    current_intensity: float = 0.5
    threshold_estimate: float = 0.5
    threshold_std: float = 0.2

    # Trial history
    intensities: List[float] = field(default_factory=list)
    responses: List[bool] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    # Convergence tracking
    reversals: int = 0
    last_direction: Optional[int] = None  # 1 for up, -1 for down
    converged: bool = False
    convergence_trial: Optional[int] = None

    # Posterior distribution
    posterior: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "trial_number": self.trial_number,
            "current_intensity": self.current_intensity,
            "threshold_estimate": self.threshold_estimate,
            "threshold_std": self.threshold_std,
            "intensities": self.intensities,
            "responses": self.responses,
            "timestamps": [ts.isoformat() for ts in self.timestamps],
            "reversals": self.reversals,
            "last_direction": self.last_direction,
            "converged": self.converged,
            "convergence_trial": self.convergence_trial,
            "posterior": (self.posterior.tolist() if self.posterior is not None else None),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StaircaseState":
        """Create state from dictionary."""
        state = cls()
        state.trial_number = data["trial_number"]
        state.current_intensity = data["current_intensity"]
        state.threshold_estimate = data["threshold_estimate"]
        state.threshold_std = data["threshold_std"]
        state.intensities = data["intensities"]
        state.responses = data["responses"]
        state.timestamps = [datetime.fromisoformat(ts) for ts in data["timestamps"]]
        state.reversals = data["reversals"]
        state.last_direction = data["last_direction"]
        state.converged = data["converged"]
        state.convergence_trial = data["convergence_trial"]
        if data["posterior"] is not None:
            state.posterior = np.array(data["posterior"])
        return state


class QuestPlusStaircase:
    """QUEST+ adaptive staircase for optimal stimulus selection.

    Implements the QUEST+ algorithm which uses Bayesian inference to
    select stimulus intensities that maximize information gain about
    the psychometric function parameters.
    """

    def __init__(
        self,
        parameters: Optional[QuestPlusParameters] = None,
        participant_id: str = "",
        task_id: str = "",
    ):
        """Initialize QUEST+ staircase.

        Args:
            parameters: QUEST+ algorithm parameters
            participant_id: Unique participant identifier
            task_id: Unique task identifier
        """
        self.parameters = parameters or QuestPlusParameters()
        self.participant_id = participant_id
        self.task_id = task_id
        self.state = StaircaseState()

        # Initialize parameter spaces
        self._initialize_parameter_spaces()

        # Initialize prior distribution
        self._initialize_prior()

        logger.info(
            f"Initialized QUEST+ staircase for participant {participant_id}, task {task_id}"
        )

    def _initialize_parameter_spaces(self) -> None:
        """Initialize the parameter spaces for stimulus and psychometric function."""
        # Stimulus intensity space
        self.stimulus_space = np.linspace(
            self.parameters.stimulus_min,
            self.parameters.stimulus_max,
            self.parameters.stimulus_steps,
        )

        # Threshold parameter space
        self.threshold_space = np.linspace(
            self.parameters.threshold_min,
            self.parameters.threshold_max,
            self.parameters.threshold_steps,
        )

        # Slope parameter space
        self.slope_space = np.linspace(
            self.parameters.slope_min,
            self.parameters.slope_max,
            self.parameters.slope_steps,
        )

        logger.debug(
            f"Parameter spaces initialized: {len(self.stimulus_space)} stimuli, "
            f"{len(self.threshold_space)} thresholds, {len(self.slope_space)} slopes"
        )

    def _initialize_prior(self) -> None:
        """Initialize prior distribution over psychometric function parameters."""
        # Use provided priors or uniform distributions
        if self.parameters.threshold_prior is not None:
            threshold_prior = self.parameters.threshold_prior
        else:
            threshold_prior = np.ones(len(self.threshold_space))

        if self.parameters.slope_prior is not None:
            slope_prior = self.parameters.slope_prior
        else:
            slope_prior = np.ones(len(self.slope_space))

        # Create joint prior (outer product)
        self.prior = np.outer(threshold_prior, slope_prior)
        self.prior = self.prior / np.sum(self.prior)  # Normalize

        # Initialize posterior as prior
        self.state.posterior = self.prior.copy()

        logger.debug(f"Prior distribution initialized with shape {self.prior.shape}")

    def _psychometric_function(self, intensity: float, threshold: float, slope: float) -> float:
        """Compute psychometric function value.

        Uses cumulative normal distribution with lapse and guess rates.

        Args:
            intensity: Stimulus intensity
            threshold: Detection threshold
            slope: Psychometric function slope

        Returns:
            Probability of detection (0-1)
        """
        # Convert to z-score
        z = slope * (intensity - threshold)

        # Cumulative normal with lapse and guess rates
        p_detect = self.parameters.guess_rate + (
            1 - self.parameters.guess_rate - self.parameters.lapse_rate
        ) * norm.cdf(z)

        return float(np.clip(p_detect, 1e-10, 1 - 1e-10))  # Avoid numerical issues

    def _likelihood(
        self, intensity: float, response: bool, threshold: float, slope: float
    ) -> float:
        """Compute likelihood of response given parameters.

        Args:
            intensity: Stimulus intensity
            response: Participant response (True = detected)
            threshold: Detection threshold
            slope: Psychometric function slope

        Returns:
            Likelihood of the response
        """
        p_detect = self._psychometric_function(intensity, threshold, slope)

        if response:
            return p_detect
        else:
            return 1 - p_detect

    def _expected_entropy(self, intensity: float) -> float:
        """Calculate expected entropy after presenting stimulus at given intensity.

        Args:
            intensity: Stimulus intensity to evaluate

        Returns:
            Expected entropy (lower is better for information gain)
        """
        # Calculate expected posterior for each possible response
        entropy_detected = 0.0
        entropy_not_detected = 0.0
        p_detected_total = 0.0

        # Iterate over all parameter combinations
        for i, threshold in enumerate(self.threshold_space):
            for j, slope in enumerate(self.slope_space):
                if self.state.posterior is not None:
                    prior_prob = self.state.posterior[i, j]

                    if prior_prob > 1e-10:  # Skip negligible probabilities
                        # Probability of detection for this parameter combination
                        p_detect = self._psychometric_function(intensity, threshold, slope)
                        posterior_detected = prior_prob * p_detect
                        posterior_not_detected = prior_prob * (1 - p_detect)

                        p_detected_total += posterior_detected
                        entropy_detected += posterior_detected * np.log(posterior_detected)
                        entropy_not_detected += posterior_not_detected * np.log(
                            posterior_not_detected
                        )

        # Normalize posteriors and calculate entropies
        p_not_detected_total = 1.0 - p_detected_total
        if p_detected_total > 1e-10:
            entropy_detected = -entropy_detected / p_detected_total + np.log(p_detected_total)
        else:
            entropy_detected = 0.0

        if p_not_detected_total > 1e-10:
            entropy_not_detected = -entropy_not_detected / p_not_detected_total + np.log(
                p_not_detected_total
            )
        else:
            entropy_not_detected = 0.0

        # Expected entropy
        expected_entropy = (
            p_detected_total * entropy_detected + p_not_detected_total * entropy_not_detected
        )

        return expected_entropy

    def get_next_intensity(self) -> float:
        """Get the optimal stimulus intensity for the next trial.

        Returns:
            Optimal stimulus intensity
        """
        if self.state.trial_number == 0:
            # First trial: start at middle of range
            intensity = (self.parameters.stimulus_min + self.parameters.stimulus_max) / 2
        else:
            # Find intensity that minimizes expected entropy
            entropies = [self._expected_entropy(intensity) for intensity in self.stimulus_space]
            optimal_idx = np.argmin(entropies)
            intensity = self.stimulus_space[optimal_idx]

        self.state.current_intensity = intensity
        logger.debug(f"Selected intensity {intensity:.4f} for trial {self.state.trial_number + 1}")

        return intensity

    def update(self, intensity: float, response: bool) -> None:
        """Update the staircase with the response to the current trial.

        Args:
            intensity: Stimulus intensity that was presented
            response: Participant response (True = detected)
        """
        # Validate response type
        if not isinstance(response, bool):
            raise TypeError("Response must be a boolean (True or False)")

        # Validate intensity
        if not isinstance(intensity, (int, float)):
            raise TypeError("Intensity must be a number")

        if intensity < self.parameters.stimulus_min or intensity > self.parameters.stimulus_max:
            raise ValueError(
                f"Intensity must be between {self.parameters.stimulus_min} and {self.parameters.stimulus_max}"
            )

        # Record trial data
        self.state.trial_number += 1
        self.state.intensities.append(intensity)
        self.state.responses.append(response)
        self.state.timestamps.append(datetime.now())

        # Update posterior distribution using Bayes' rule
        self._update_posterior(intensity, response)

        # Update threshold estimate
        self._update_threshold_estimate()

        # Check for reversals
        self._check_reversals(response)

        # Check convergence
        self._check_convergence()

        logger.debug(
            f"Updated staircase: trial {self.state.trial_number}, "
            f"threshold estimate {self.state.threshold_estimate:.4f}"
        )

    def _update_posterior(self, intensity: float, response: bool) -> None:
        """Update posterior distribution using Bayes' rule."""
        # Calculate likelihood for each parameter combination
        if self.state.posterior is None:
            return
        for i, threshold in enumerate(self.threshold_space):
            for j, slope in enumerate(self.slope_space):
                likelihood = self._likelihood(intensity, response, threshold, slope)
                self.state.posterior[i, j] *= likelihood

        # Normalize posterior
        if self.state.posterior is not None:
            posterior_sum = np.sum(self.state.posterior)
            if posterior_sum > 1e-10:
                self.state.posterior /= posterior_sum
            else:
                # Reset to prior if posterior becomes degenerate
                self.state.posterior = self.prior.copy()
                logger.warning("Posterior became degenerate, reset to prior")

    def _update_threshold_estimate(self) -> None:
        """Update threshold estimate from posterior distribution."""
        # Calculate marginal distribution over thresholds
        if self.state.posterior is None:
            return
        threshold_marginal = np.sum(self.state.posterior, axis=1)

        # Calculate mean and standard deviation
        self.state.threshold_estimate = np.sum(threshold_marginal * self.threshold_space)

        # Calculate standard deviation
        threshold_variance = np.sum(
            threshold_marginal * (self.threshold_space - self.state.threshold_estimate) ** 2
        )
        self.state.threshold_std = np.sqrt(threshold_variance)

    def _check_reversals(self, response: bool) -> None:
        """Check for reversals in the staircase direction."""
        if len(self.state.responses) < 2:
            return

        # Determine current direction based on response pattern
        current_direction = None
        if response and not self.state.responses[-2]:
            current_direction = 1  # Up (increase intensity after miss)
        elif not response and self.state.responses[-2]:
            current_direction = -1  # Down (decrease intensity after hit)

        # Check for reversal
        if (
            current_direction is not None
            and self.state.last_direction is not None
            and current_direction != self.state.last_direction
        ):
            self.state.reversals += 1
            logger.debug(
                f"Reversal {self.state.reversals} detected at trial {self.state.trial_number}"
            )

        if current_direction is not None:
            self.state.last_direction = current_direction

    def _check_convergence(self) -> None:
        """Check if the staircase has converged."""
        if self.state.converged:
            return

        # Check minimum trials and reversals
        if (
            self.state.trial_number < self.parameters.min_trials
            or self.state.reversals < self.parameters.min_reversals
        ):
            return

        # Check threshold stability (change in estimate over recent trials)
        if len(self.state.intensities) >= 10:
            recent_estimates = []
            # Calculate threshold estimates for recent windows
            for i in range(len(self.state.intensities) - 5, len(self.state.intensities)):
                if i >= 5:  # Need some history for estimate
                    # Simplified estimate based on recent responses
                    recent_responses = self.state.responses[max(0, i - 10) : i]
                    recent_intensities = self.state.intensities[max(0, i - 10) : i]

                    if recent_responses and recent_intensities:
                        # Simple threshold estimate: intensity at 50% detection
                        hits = [
                            intensity
                            for intensity, response in zip(recent_intensities, recent_responses)
                            if response
                        ]
                        misses = [
                            intensity
                            for intensity, response in zip(recent_intensities, recent_responses)
                            if not response
                        ]

                        if hits and misses:
                            estimate = (max(misses) + min(hits)) / 2
                            recent_estimates.append(estimate)

            # Check stability of recent estimates
            if len(recent_estimates) >= 3:
                estimate_std = np.std(recent_estimates)
                if estimate_std < self.parameters.convergence_criterion:
                    self.state.converged = True
                    self.state.convergence_trial = self.state.trial_number
                    logger.info(
                        f"Staircase converged at trial {self.state.trial_number}, "
                        f"threshold estimate: {self.state.threshold_estimate:.4f}"
                    )

        # Force convergence at maximum trials
        if self.state.trial_number >= self.parameters.max_trials:
            self.state.converged = True
            self.state.convergence_trial = self.state.trial_number
            logger.info(
                f"Staircase forced convergence at maximum trials ({self.parameters.max_trials})"
            )

    def get_threshold_estimate(self) -> Tuple[float, float]:
        """Get current threshold estimate with uncertainty.

        Returns:
            Tuple of (threshold_estimate, threshold_std)
        """
        return self.state.threshold_estimate, self.state.threshold_std

    def get_psychometric_curve(
        self, intensities: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the estimated psychometric curve.

        Args:
            intensities: Stimulus intensities to evaluate (default: parameter space)

        Returns:
            Tuple of (intensities, detection_probabilities)
        """
        if intensities is None:
            intensities = self.stimulus_space

        # Calculate expected detection probability for each intensity
        detection_probs = np.zeros_like(intensities)

        for k, intensity in enumerate(intensities):
            prob_sum = 0.0
            for i, threshold in enumerate(self.threshold_space):
                for j, slope in enumerate(self.slope_space):
                    if self.state.posterior is not None:
                        posterior_prob = self.state.posterior[i, j]
                        p_detect = self._psychometric_function(intensity, threshold, slope)
                        prob_sum += posterior_prob * p_detect

            detection_probs[k] = prob_sum

        return intensities, detection_probs

    def is_converged(self) -> bool:
        """Check if the staircase has converged."""
        return self.state.converged

    def should_continue(self) -> bool:
        """Check if the staircase should continue."""
        return not self.state.converged and self.state.trial_number < self.parameters.max_trials

    def save_state(self, filepath: str) -> None:
        """Save staircase state to file."""
        state_data = {
            "parameters": self.parameters.__dict__,
            "participant_id": self.participant_id,
            "task_id": self.task_id,
            "state": self.state.to_dict(),
        }

        with open(filepath, "w") as f:
            json.dump(state_data, f, indent=2)

        logger.info(f"Saved staircase state to {filepath}")

    def load_state(self, filepath: str) -> None:
        """Load staircase state from file."""
        with open(filepath, "r") as f:
            state_data = json.load(f)

        # Restore parameters
        self.parameters = QuestPlusParameters(**state_data["parameters"])
        self.participant_id = state_data["participant_id"]
        self.task_id = state_data["task_id"]

        # Restore state
        self.state = StaircaseState.from_dict(state_data["state"])

        # Reinitialize parameter spaces and prior
        self._initialize_parameter_spaces()
        self._initialize_prior()

        logger.info(f"Loaded staircase state from {filepath}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of staircase performance."""
        if not self.state.intensities:
            return {"error": "No trials completed"}

        hit_rate = np.mean(self.state.responses) if self.state.responses else 0.0

        return {
            "total_trials": self.state.trial_number,
            "converged": self.state.converged,
            "convergence_trial": self.state.convergence_trial,
            "reversals": self.state.reversals,
            "threshold_estimate": self.state.threshold_estimate,
            "threshold_std": self.state.threshold_std,
            "hit_rate": hit_rate,
            "intensity_range": (
                min(self.state.intensities),
                max(self.state.intensities),
            ),
            "final_intensity": (self.state.intensities[-1] if self.state.intensities else None),
        }
