"""Phase Transition Module.

Implements phase transition dynamics for APGI consciousness experiments.
Provides the SomaticAgent class for active inference with somatic markers.
"""

from typing import Tuple

import numpy as np

from apgi_framework.core.precision import PrecisionCalculator
from apgi_framework.core.prediction_error import PredictionErrorProcessor


class SomaticAgent:
    """Active inference agent with somatic markers for decision making.

    The agent uses expected free energy minimization and somatic markers
    to make decisions between habitual (unconscious) and conscious modes.
    """

    def __init__(
        self,
        n_states: int = 4,
        n_actions: int = 3,
        n_contexts: int = 2,
        precision: float = 1.0,
    ):
        """Initialize SomaticAgent.

        Args:
            n_states: Number of hidden states
            n_actions: Number of possible actions
            n_contexts: Number of environmental contexts
            precision: Precision parameter for decision making
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_contexts = n_contexts
        self.precision = precision

        # Initialize somatic markers (context x action matrix)
        self.somatic_markers = np.zeros((n_contexts, n_actions))

        # Learning rate for somatic marker updates
        self.learning_rate = 0.1

        # Initialize precision calculator
        self.precision_calc = PrecisionCalculator()

        # Initialize prediction error processor
        self.prediction_error_proc = PredictionErrorProcessor()

    def expected_free_energy(
        self, beliefs: np.ndarray, context: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate expected free energy for each action.

        Args:
            beliefs: Current belief state
            context: Current environmental context

        Returns:
            Tuple of (G_modified, G_basic) energy arrays
        """
        # Mock implementation
        G_modified = np.random.random(self.n_actions)
        G_basic = np.random.random(self.n_actions)
        return G_modified, G_basic

    def update_somatic_marker(self, context: int, action: int, outcome: float) -> None:
        """Update somatic marker based on outcome.

        Args:
            context: Environmental context
            action: Action taken
            outcome: Outcome value (positive or negative)
        """
        self.somatic_markers[context, action] += self.learning_rate * outcome

    def decide(
        self, beliefs: np.ndarray, context: int = 0, surprise: float = 0.1
    ) -> Tuple[int, bool, dict]:
        """Make a decision based on beliefs and surprise.

        Args:
            beliefs: Current belief state
            context: Current environmental context
            surprise: Surprise level (higher triggers conscious processing)

        Returns:
            Tuple of (action, is_conscious, info_dict)
        """
        # Simple decision making - select action with best somatic marker
        action = int(np.argmax(self.somatic_markers[context]))

        # High surprise triggers conscious access
        conscious = surprise > 1.0

        return action, conscious, {"surprise": surprise}


class PhaseTransitionAnalyzer:
    """Analyzes phase transitions in APGI consciousness experiments.

    Provides methods for detecting phase transitions and analyzing
    transition dynamics between conscious and unconscious states.
    """

    def __init__(self, data: np.ndarray):
        """Initialize PhaseTransitionAnalyzer.

        Args:
            data: Neural or behavioral data to analyze
        """
        self.data = data

    def detect_transitions(self, threshold: float = 0.5) -> np.ndarray:
        """Detect phase transitions based on energy threshold.

        Args:
            threshold: Energy threshold for transition detection

        Returns:
            Boolean array indicating transition points
        """
        # Mock implementation
        return np.random.random(len(self.data)) > threshold

    def analyze_transition_dynamics(self) -> dict:
        """Analyze transition dynamics between states.

        Returns:
            Dictionary with transition statistics
        """
        # Mock implementation
        return {
            "transition_frequency": np.random.random(),
            "state_stability": np.random.random(),
            "transition_latency": np.random.random(),
        }


class ConsciousnessDetector:
    """Detects consciousness indicators in neural/behavioral data.

    Uses various metrics to identify conscious vs unconscious
    processing patterns.
    """

    def __init__(self, data: np.ndarray, sampling_rate: float = 1000.0):
        """Initialize ConsciousnessDetector.

        Args:
            data: Neural data to analyze
            sampling_rate: Data sampling rate in Hz
        """
        self.data = data
        self.sampling_rate = sampling_rate

    def detect_consciousness_level(self) -> float:
        """Detect current consciousness level (0-1 scale).

        Returns:
            Consciousness level between 0 (unconscious) and 1 (conscious)
        """
        # Mock implementation
        return np.random.random()

    def analyze_consciousness_metrics(self) -> dict:
        """Analyze various consciousness-related metrics.

        Returns:
            Dictionary with consciousness metrics
        """
        # Mock implementation
        return {
            "consciousness_level": self.detect_consciousness_level(),
            "neural_coherence": np.random.random(),
            "information_integration": np.random.random(),
            "metacognitive_awareness": np.random.random(),
        }
