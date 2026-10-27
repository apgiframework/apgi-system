"""Core neural models for APGI Framework.

This module contains the main neural network models including:
- SomaticAgent: Active inference agent with somatic markers
- PredictiveIgnitionNetwork: Neural network with ignition dynamics
"""

from typing import Tuple

import numpy as np

from .precision_engine import PrecisionCalculator
from .prediction_error_engine import PredictionErrorProcessor


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
        """Initialize the SomaticAgent.

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
            beliefs: Current belief state over hidden states
            context: Current environmental context

        Returns:
            Tuple of (modified_free_energy, basic_free_energy) arrays
        """
        # Basic expected free energy (without somatic markers)
        # Simplified calculation: -log(beliefs) for each action
        basic_G = -np.log(beliefs + 1e-8)  # Add small epsilon to avoid log(0)

        # Extend to match number of actions (repeat for each action)
        basic_G_extended = np.tile(basic_G.mean(), self.n_actions)

        # Modify with somatic markers
        somatic_bias = self.somatic_markers[context, :] * self.precision
        modified_G = basic_G_extended - somatic_bias

        return modified_G, basic_G_extended

    def update_somatic_marker(self, context: int, action: int, outcome_valence: float) -> None:
        """Update somatic markers based on action outcome.

        Args:
            context: Environmental context
            action: Action taken
            outcome_valence: Valence of outcome (positive for good, negative for bad)
        """
        # Simple learning rule: update towards outcome valence
        current_marker = self.somatic_markers[context, action]
        self.somatic_markers[context, action] += self.learning_rate * (
            outcome_valence - current_marker
        )

        # Clamp to reasonable bounds
        self.somatic_markers[context, action] = np.clip(
            self.somatic_markers[context, action], -2.0, 2.0
        )

    def decide(
        self, beliefs: np.ndarray, context: int, surprise: float
    ) -> Tuple[int, bool, np.ndarray]:
        """Make a decision based on beliefs, context, and surprise.

        Args:
            beliefs: Current belief state
            context: Environmental context
            surprise: Current surprise level (high surprise -> conscious processing)

        Returns:
            Tuple of (action, conscious_flag, free_energy_array)
        """
        # Calculate expected free energy
        G_modified, G_basic = self.expected_free_energy(beliefs, context)

        # Determine if conscious processing based on surprise
        # Higher surprise leads to conscious processing
        consciousness_threshold = 1.5  # Threshold for conscious ignition
        conscious = bool(surprise > consciousness_threshold)

        # Action selection: minimize expected free energy
        if conscious:
            # Conscious: use modified free energy with somatic markers
            action = np.argmin(G_modified)
            G = G_modified
        else:
            # Habitual: use basic free energy
            action = np.argmin(G_basic)
            G = G_basic

        return int(action), conscious, G


class PredictiveIgnitionNetwork:
    """Neural network with predictive coding and ignition dynamics.

    Implements a predictive coding network where high surprise can trigger
    global ignition, broadcasting information throughout the network.
    """

    def __init__(
        self,
        n_features: int = 10,
        n_global_units: int = 5,
        threshold: float = 1.0,
        alpha: float = 2.0,
    ):
        """Initialize the PredictiveIgnitionNetwork.

        Args:
            n_features: Number of input features
            n_global_units: Number of global broadcasting units
            threshold: Ignition threshold
            alpha: Precision/weighting parameter
        """
        self.n_features = n_features
        self.n_global_units = n_global_units
        self.threshold = threshold
        self.alpha = alpha

        # Initialize weight matrix W (global_units x features) as expected by tests
        # Use seeded random for reproducibility
        rng = np.random.RandomState(42)
        self.W = rng.randn(n_global_units, n_features) * 0.1

        # Initialize precision vector for features
        self.precision = np.ones(n_features)

        # Initialize global activation state
        self.global_activation = np.zeros(n_global_units, dtype=np.float64)

        # Initialize precision calculator
        self.precision_calc = PrecisionCalculator()

        # Initialize prediction error processor
        self.prediction_error_proc = PredictionErrorProcessor()

    def forward_pass(
        self, sensory_input: np.ndarray, somatic_gain: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, float]:
        """Perform a forward pass through the network.

        Args:
            sensory_input: Input sensory data
            somatic_gain: Gain factor from somatic markers

        Returns:
            Tuple of (predictions, errors, weighted_errors, ignited, ignition_probability)
        """
        # Generate predictions based on current global activation and weights
        predictions = self.W.T @ self.global_activation

        # Calculate prediction errors
        errors = sensory_input - predictions

        # Weight errors by precision and somatic gain
        weighted_errors = errors * self.precision * somatic_gain * self.alpha

        # Calculate ignition probability based on error magnitude
        error_magnitude = np.linalg.norm(weighted_errors)

        # Sigmoid function for ignition probability
        ignition_prob = 1.0 / (1.0 + np.exp(-self.alpha * (error_magnitude - self.threshold)))

        # Determine if ignition occurs (ensure Python bool, not numpy bool)
        ignited = bool(ignition_prob > 0.5)

        if ignited:
            # Update global activation based on prediction errors
            activation_update = self.W @ weighted_errors
            self.global_activation += 0.1 * activation_update  # Learning rate

            # Apply activation function (tanh for stability)
            self.global_activation[:] = np.tanh(self.global_activation)

        return predictions, errors, weighted_errors, ignited, ignition_prob

    def reset(self) -> None:
        """Reset the network state."""
        self.global_activation = np.zeros(self.n_global_units)


# Aliases for backward compatibility with tests
AgentModel = SomaticAgent
