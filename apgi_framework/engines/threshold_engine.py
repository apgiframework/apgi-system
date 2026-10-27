"""Threshold management system for dynamic ignition threshold adjustment.

This module implements the threshold manager that handles dynamic adjustment
of the ignition threshold (θₜ) based on context, adaptation, and task demands.
"""

from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from ..exceptions import MathematicalError
from .somatic_marker_engine import ContextType


class ThresholdAdaptationType(Enum):
    """Types of threshold adaptation mechanisms."""

    FIXED = "fixed"
    HOMEOSTATIC = "homeostatic"
    CONTEXTUAL = "contextual"
    ADAPTIVE = "adaptive"


class ThresholdManager:
    """Manager for dynamic ignition threshold (θₜ) adjustment.

    Handles threshold modulation based on context, recent ignition history,
    and adaptive mechanisms to maintain optimal ignition rates.
    """

    def __init__(
        self,
        baseline_threshold: float = 3.5,
        min_threshold: float = 0.5,
        max_threshold: float = 10.0,
        adaptation_type: ThresholdAdaptationType = ThresholdAdaptationType.ADAPTIVE,
    ):
        """Initialize threshold manager.

        Args:
            baseline_threshold: Default threshold value
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
            adaptation_type: Type of threshold adaptation to use
        """
        if baseline_threshold <= 0:
            raise MathematicalError("Baseline threshold must be positive")
        if max_threshold <= min_threshold:
            raise MathematicalError("Maximum threshold must be greater than minimum")
        if not (min_threshold <= baseline_threshold <= max_threshold):
            raise MathematicalError("Baseline threshold must be within min/max bounds")

        self.baseline_threshold = baseline_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adaptation_type = adaptation_type

        # Adaptation parameters
        self.target_ignition_rate = 0.3  # Target 30% ignition rate
        self.adaptation_rate = 0.05  # Rate of threshold adjustment
        self.context_sensitivity = 0.5  # Sensitivity to context changes

        # History tracking
        self._ignition_history: List[bool] = []
        self._threshold_history: List[float] = []
        self._current_threshold = baseline_threshold

        # Context-specific threshold modulations
        self._context_modulations = {
            ContextType.ROUTINE: 0.0,
            ContextType.HIGH_STAKES: -0.5,  # Lower threshold for high stakes
            ContextType.EMOTIONAL: -0.3,
            ContextType.THREAT: -0.8,  # Much lower threshold for threats
            ContextType.REWARD: -0.2,
            ContextType.NEUTRAL: 0.0,
        }

    def get_current_threshold(
        self, context: Optional[ContextType] = None, arousal: float = 1.0
    ) -> float:
        """Get current ignition threshold with optional context modulation.

        Args:
            context: Current contextual situation
            arousal: Current arousal level (0.1 to 3.0)

        Returns:
            Current ignition threshold value

        Raises:
            MathematicalError: If arousal is out of valid range
        """
        if not (0.1 <= arousal <= 3.0):
            raise MathematicalError("Arousal must be between 0.1 and 3.0")

        threshold = self._current_threshold

        # Apply context modulation if provided
        if context is not None:
            context_adjustment = self._context_modulations.get(context, 0.0)
            threshold += self.context_sensitivity * context_adjustment

        # Apply arousal modulation (higher arousal lowers threshold)
        arousal_adjustment = -0.2 * (arousal - 1.0)  # Maps arousal to threshold adjustment
        threshold += arousal_adjustment

        # Apply bounds
        threshold = np.clip(threshold, self.min_threshold, self.max_threshold)

        return float(threshold)

    def update_threshold(
        self, ignition_occurred: bool, context: Optional[ContextType] = None
    ) -> float:
        """Update threshold based on recent ignition and adaptation type.

        Args:
            ignition_occurred: Whether ignition occurred in the last trial
            context: Current contextual situation

        Returns:
            Updated threshold value
        """
        # Record ignition in history
        self._ignition_history.append(ignition_occurred)

        # Limit history size
        if len(self._ignition_history) > 100:
            self._ignition_history = self._ignition_history[-100:]

        # Update threshold based on adaptation type
        if self.adaptation_type == ThresholdAdaptationType.FIXED:
            # No adaptation - keep baseline
            new_threshold = self.baseline_threshold

        elif self.adaptation_type == ThresholdAdaptationType.HOMEOSTATIC:
            # Homeostatic adaptation to maintain target ignition rate
            new_threshold = self._homeostatic_adaptation()

        elif self.adaptation_type == ThresholdAdaptationType.CONTEXTUAL:
            # Context-dependent adaptation
            new_threshold = self._contextual_adaptation(context)

        else:
            # ADAPTIVE: Full adaptive mechanism
            new_threshold = self._adaptive_threshold_update(context)

        # Apply bounds and update
        self._current_threshold = np.clip(new_threshold, self.min_threshold, self.max_threshold)

        # Record in history
        self._threshold_history.append(self._current_threshold)
        if len(self._threshold_history) > 100:
            self._threshold_history = self._threshold_history[-100:]

        return self._current_threshold

    def _homeostatic_adaptation(self) -> float:
        """Implement homeostatic threshold adaptation."""
        if len(self._ignition_history) < 10:
            return self._current_threshold

        # Calculate recent ignition rate
        recent_rate = np.mean(self._ignition_history[-20:])

        # Adjust threshold to move toward target rate
        rate_error = recent_rate - self.target_ignition_rate
        threshold_adjustment = self.adaptation_rate * rate_error

        return float(self._current_threshold + threshold_adjustment)

    def _contextual_adaptation(self, context: Optional[ContextType]) -> float:
        """Implement context-dependent threshold adaptation."""
        base_threshold = self._homeostatic_adaptation()

        if context is None:
            return base_threshold

        # Apply context-specific adjustment
        context_adjustment = self._context_modulations.get(context, 0.0)
        return base_threshold + self.context_sensitivity * context_adjustment

    def _adaptive_threshold_update(self, context: Optional[ContextType]) -> float:
        """Implement sophisticated adaptive threshold mechanism with trend analysis."""
        # Start with homeostatic adaptation
        threshold = self._homeostatic_adaptation()

        # Add context sensitivity
        if context is not None:
            context_adjustment = self._context_modulations.get(context, 0.0)
            threshold += self.context_sensitivity * context_adjustment

        # Enhanced variability-based adjustment with trend analysis
        if len(self._ignition_history) >= 20:
            # Calculate multiple metrics for sophisticated analysis
            recent_ignitions = self._ignition_history[-20:]

            # Basic variability
            recent_variability = np.std(recent_ignitions)

            # Trend analysis using linear regression
            time_points = np.arange(len(recent_ignitions))
            if len(recent_ignitions) >= 5:
                # Calculate trend slope
                trend_slope = np.polyfit(time_points, recent_ignitions, 1)[0]

                # Rate of change analysis
                if len(recent_ignitions) >= 10:
                    # Short-term vs long-term trend comparison
                    short_term_trend = np.polyfit(time_points[-5:], recent_ignitions[-5:], 1)[0]
                    long_term_trend = np.polyfit(time_points, recent_ignitions, 1)[0]
                    trend_acceleration = short_term_trend - long_term_trend
                else:
                    trend_acceleration = 0.0

                # Volatility clustering detection
                if len(recent_ignitions) >= 10:
                    recent_volatility = float(np.std(recent_ignitions[-5:]))
                    historical_volatility = float(np.std(recent_ignitions[:-5]))
                    volatility_ratio = recent_volatility / (historical_volatility + 1e-8)
                else:
                    volatility_ratio = 1.0

                # Momentum indicator
                momentum = (
                    np.mean(recent_ignitions[-3:]) - np.mean(recent_ignitions[-10:-3])
                    if len(recent_ignitions) >= 10
                    else 0.0
                )

                # Adaptive adjustments based on sophisticated metrics
                variability_adjustment = 0.1 * recent_variability
                trend_adjustment = 0.05 * trend_slope  # Positive trend increases threshold
                acceleration_adjustment = (
                    0.02 * trend_acceleration
                )  # Acceleration affects threshold
                volatility_adjustment = 0.03 * (volatility_ratio - 1.0)  # Volatility clustering
                momentum_adjustment = 0.04 * momentum  # Momentum-based adjustment

                # Combine adjustments with adaptive weighting
                total_adjustment = (
                    variability_adjustment
                    + trend_adjustment
                    + acceleration_adjustment
                    + volatility_adjustment
                    + momentum_adjustment
                )

                # Apply adaptive scaling based on system stability
                stability_factor = self._calculate_system_stability(np.array(recent_ignitions))
                scaled_adjustment = total_adjustment * stability_factor

                threshold += scaled_adjustment

        # Predictive adjustment based on expected future states
        if len(self._ignition_history) >= 30:
            predicted_adjustment = self._predictive_threshold_adjustment()
            threshold += predicted_adjustment

        return threshold

    def _calculate_system_stability(self, recent_ignitions: np.ndarray) -> float:
        """Calculate system stability factor for adaptive scaling."""
        if len(recent_ignitions) < 10:
            return 1.0

        # Multiple stability metrics
        # 1. Variance stability (coefficient of variation)
        mean_ignition = np.mean(recent_ignitions)
        cv = np.std(recent_ignitions) / (mean_ignition + 1e-8)
        variance_stability = 1.0 / (1.0 + cv)  # Lower CV = higher stability

        # 2. Trend stability (how consistent is the trend)
        if len(recent_ignitions) >= 10:
            # Calculate trend consistency
            first_half_trend = np.polyfit(np.arange(5), recent_ignitions[:5], 1)[0]
            second_half_trend = np.polyfit(np.arange(5), recent_ignitions[-5:], 1)[0]
            trend_consistency = 1.0 / (1.0 + abs(first_half_trend - second_half_trend))
        else:
            trend_consistency = 1.0

        # 3. Autocorrelation stability
        if len(recent_ignitions) >= 8:
            # Simple autocorrelation at lag 1
            autocorr = np.corrcoef(recent_ignitions[:-1], recent_ignitions[1:])[0, 1]
            autocorr_stability = abs(autocorr)  # Higher autocorr = more predictable = more stable
        else:
            autocorr_stability = 0.5

        # Combine stability metrics
        overall_stability: float = (
            variance_stability + trend_consistency + autocorr_stability
        ) / 3.0

        # Scale to reasonable range (0.5 to 1.5)
        return 0.5 + overall_stability

    def _predictive_threshold_adjustment(self) -> float:
        """Calculate predictive adjustment based on expected future states."""
        if len(self._ignition_history) < 30:
            return 0.0

        recent_history = self._ignition_history[-30:]

        # Simple time series prediction using exponential smoothing
        alpha = 0.3  # Smoothing factor

        # Calculate smoothed values
        smoothed = [float(recent_history[0])]
        for i in range(1, len(recent_history)):
            smoothed.append(alpha * recent_history[i] + (1 - alpha) * smoothed[-1])

        # Predict next value based on trend
        if len(smoothed) >= 5:
            recent_trend = smoothed[-1] - smoothed[-5]
            predicted_next = smoothed[-1] + recent_trend * 0.5  # Conservative prediction

            # Calculate adjustment to prepare for predicted state
            current_avg = np.mean(recent_history[-5:])
            predictive_adjustment = 0.02 * (predicted_next - current_avg)

            return float(predictive_adjustment)

        return 0.0

    def reset_threshold(self, new_baseline: Optional[float] = None) -> None:
        """Reset threshold to baseline and clear history.

        Args:
            new_baseline: New baseline threshold (if None, uses current baseline)
        """
        if new_baseline is not None:
            if new_baseline <= 0:
                raise MathematicalError("New baseline threshold must be positive")
            self.baseline_threshold = new_baseline

        self._current_threshold = self.baseline_threshold
        self._ignition_history.clear()
        self._threshold_history.clear()

    def get_ignition_statistics(self) -> Dict[str, float]:
        """Get statistics about recent ignition patterns.

        Returns:
            Dictionary with ignition statistics
        """
        if len(self._ignition_history) == 0:
            return {
                "ignition_rate": 0.0,
                "n_trials": 0,
                "variability": 0.0,
                "recent_rate": 0.0,
            }

        ignition_array = np.array(self._ignition_history, dtype=float)

        stats = {
            "ignition_rate": float(np.mean(ignition_array)),
            "n_trials": len(self._ignition_history),
            "variability": float(np.std(ignition_array)),
            "recent_rate": (
                float(np.mean(ignition_array[-10:]))
                if len(ignition_array) >= 10
                else float(np.mean(ignition_array))
            ),
        }

        return stats

    def update_context_modulation(self, context: ContextType, modulation: float) -> None:
        """Update threshold modulation for a specific context.

        Args:
            context: Context type to update
            modulation: New modulation value (can be negative)
        """
        self._context_modulations[context] = modulation

    def get_threshold_info(self) -> dict:
        """Get information about threshold manager configuration.

        Returns:
            Dictionary with threshold manager settings
        """
        return {
            "baseline_threshold": self.baseline_threshold,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "current_threshold": self._current_threshold,
            "adaptation_type": self.adaptation_type.value,
            "target_ignition_rate": self.target_ignition_rate,
            "adaptation_rate": self.adaptation_rate,
            "context_sensitivity": self.context_sensitivity,
            "context_modulations": {
                ctx.value: mod for ctx, mod in self._context_modulations.items()
            },
            "history_length": len(self._ignition_history),
            "threshold_history_length": len(self._threshold_history),
        }


# Aliases for backward compatibility with tests
ThresholdDetector = ThresholdManager
