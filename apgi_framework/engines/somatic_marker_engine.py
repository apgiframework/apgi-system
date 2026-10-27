"""Somatic marker engine for context-dependent gain modulation.

This module implements the somatic marker system that modulates interoceptive
precision based on contextual factors like emotional valence, stakes, and arousal.

Implementation Note:
    The specification proposes exp(β·M) for somatic modulation. This
    implementation uses a bounded factor-based approach:
        M_{c,a} = baseline × context_factor × arousal_factor × valence_factor × stakes_factor
    This sigmoid-like bounded modulation is more physiologically realistic
    than unbounded exponential growth, while maintaining the same qualitative
    behavior of amplifying precision in high-stakes/emotional contexts.
"""

import warnings
from enum import Enum
from typing import Dict

import numpy as np

from ..exceptions import MathematicalError


class ContextType(Enum):
    """Types of contextual situations for somatic marker modulation."""

    ROUTINE = "routine"
    HIGH_STAKES = "high_stakes"
    EMOTIONAL = "emotional"
    THREAT = "threat"
    REWARD = "reward"
    NEUTRAL = "neutral"


class SomaticMarkerEngine:
    """Engine for calculating somatic marker gain (M_{c,a}) based on context.

    The somatic marker system modulates interoceptive precision based on
    the emotional and contextual significance of the current situation.
    """

    def __init__(self, baseline_gain: float = 1.0, max_gain: float = 5.0, min_gain: float = 0.1):
        """Initialize somatic marker engine.

        Args:
            baseline_gain: Default gain for neutral contexts
            max_gain: Maximum allowed gain value
            min_gain: Minimum allowed gain value
        """
        if baseline_gain <= 0:
            raise MathematicalError("Baseline gain must be positive")
        if max_gain <= min_gain:
            raise MathematicalError("Maximum gain must be greater than minimum gain")
        if not (min_gain <= baseline_gain <= max_gain):
            raise MathematicalError("Baseline gain must be within min/max bounds")

        self.baseline_gain = baseline_gain
        self.max_gain = max_gain
        self.min_gain = min_gain

        # Default context modulation factors
        self._context_modulations = {
            ContextType.ROUTINE: 1.0,
            ContextType.HIGH_STAKES: 2.5,
            ContextType.EMOTIONAL: 2.0,
            ContextType.THREAT: 3.0,
            ContextType.REWARD: 1.8,
            ContextType.NEUTRAL: 1.0,
        }

    def calculate_somatic_gain(
        self,
        context: ContextType,
        arousal: float = 1.0,
        valence: float = 0.0,
        stakes: float = 1.0,
    ) -> float:
        """Calculate somatic marker gain for given context and parameters.

        Args:
            context: Type of contextual situation
            arousal: Arousal level (0.1 to 3.0, default 1.0)
            valence: Emotional valence (-1.0 to 1.0, default 0.0)
            stakes: Situational stakes (0.1 to 3.0, default 1.0)

        Returns:
            Somatic marker gain value (M_{c,a})

        Raises:
            MathematicalError: If parameters are out of valid ranges
        """
        # Validate inputs
        if not (0.1 <= arousal <= 3.0):
            raise MathematicalError("Arousal must be between 0.1 and 3.0")
        if not (-1.0 <= valence <= 1.0):
            raise MathematicalError("Valence must be between -1.0 and 1.0")
        if not (0.1 <= stakes <= 3.0):
            raise MathematicalError("Stakes must be between 0.1 and 3.0")

        # Get base context modulation
        base_modulation = self._context_modulations.get(context, 1.0)

        # Calculate arousal contribution (higher arousal increases gain)
        arousal_factor = 0.5 + 0.5 * arousal  # Maps [0.1, 3.0] to roughly [0.55, 2.0]

        # Calculate valence contribution (negative valence increases gain more)
        valence_factor = 1.0 + 0.3 * abs(valence)  # Absolute valence increases gain
        if valence < 0:  # Negative emotions have stronger effect
            valence_factor *= 1.2

        # Calculate stakes contribution
        stakes_factor = 0.7 + 0.3 * stakes  # Maps [0.1, 3.0] to roughly [0.73, 1.6]

        # Combine all factors
        gain = (
            self.baseline_gain * base_modulation * arousal_factor * valence_factor * stakes_factor
        )

        # Apply bounds
        gain = np.clip(gain, self.min_gain, self.max_gain)

        return float(gain)

    def update_context_modulation(self, context: ContextType, modulation: float) -> None:
        """Update modulation factor for a specific context type.

        Args:
            context: Context type to update
            modulation: New modulation factor (must be positive)

        Raises:
            MathematicalError: If modulation is non-positive
        """
        if modulation <= 0:
            raise MathematicalError("Context modulation must be positive")

        self._context_modulations[context] = modulation

    def get_context_modulations(self) -> Dict[ContextType, float]:
        """Get current context modulation factors."""
        return self._context_modulations.copy()

    def calculate_adaptive_gain(
        self,
        recent_errors: np.ndarray,
        context: ContextType,
        adaptation_rate: float = 0.1,
    ) -> float:
        """Calculate adaptive somatic gain based on recent prediction errors.

        Args:
            recent_errors: Array of recent prediction errors
            context: Current context type
            adaptation_rate: Rate of adaptation (0.0 to 1.0)

        Returns:
            Adaptive somatic marker gain

        Raises:
            MathematicalError: If adaptation rate is out of bounds
        """
        if not (0.0 <= adaptation_rate <= 1.0):
            raise MathematicalError("Adaptation rate must be between 0.0 and 1.0")

        if len(recent_errors) == 0:
            warnings.warn("No recent errors provided, using baseline gain")
            return self.baseline_gain

        recent_errors = np.asarray(recent_errors)

        # Calculate error magnitude and variability
        error_magnitude = np.mean(np.abs(recent_errors))
        error_variability = np.std(recent_errors)

        # Base gain from context
        base_gain = self.calculate_somatic_gain(context)

        # Adaptive component based on error patterns
        # Higher errors and variability increase gain
        error_factor = 1.0 + adaptation_rate * (error_magnitude + 0.5 * error_variability)

        adaptive_gain = base_gain * error_factor

        # Apply bounds
        adaptive_gain = np.clip(adaptive_gain, self.min_gain, self.max_gain)

        return float(adaptive_gain)

    def get_engine_info(self) -> dict:
        """Get information about the somatic marker engine configuration.

        Returns:
            Dictionary with engine settings and parameters
        """
        return {
            "baseline_gain": self.baseline_gain,
            "max_gain": self.max_gain,
            "min_gain": self.min_gain,
            "context_types": [ctx.value for ctx in ContextType],
            "context_modulations": {
                ctx.value: mod for ctx, mod in self._context_modulations.items()
            },
            "parameter_ranges": {
                "arousal": [0.1, 3.0],
                "valence": [-1.0, 1.0],
                "stakes": [0.1, 3.0],
                "adaptation_rate": [0.0, 1.0],
            },
        }
