"""Precision calculation module for the APGI Framework.

This module implements precision calculations for both exteroceptive (Πₑ)
and interoceptive (Πᵢ) components, including validation and edge case handling.
"""

import warnings
from enum import Enum
from typing import Any, Tuple, cast

import numpy as np

from apgi_framework.exceptions import MathematicalError


class NeuromodulatorType(Enum):
    """Types of neuromodulators for precision modulation."""

    ACETYLCHOLINE = "acetylcholine"
    NOREPINEPHRINE = "norepinephrine"
    DOPAMINE = "dopamine"
    SEROTONIN = "serotonin"


class PrecisionWeighting:
    """Weighting mechanism for precision values with neuromodulation."""

    extero_baseline: np.ndarray
    intero_baseline: np.ndarray
    extero_precision: np.ndarray
    intero_precision: np.ndarray
    attention_focus: list[str | None]
    fatigue_level: np.ndarray
    cognitive_load: np.ndarray
    neuromodulators: dict[NeuromodulatorType, np.ndarray]
    attention_gain: np.ndarray
    extero_volatility: float
    intero_volatility: float
    attention_gain_range: list[float]
    volatility_sensitivity: float
    neuromodulator_effects: dict[str, float]
    precision_range: list[float]
    _error_variance_history: list[float]

    def __init__(self, config: dict[str, float] | float = 1.0):
        """Initialize precision weighting.

        Parameters
        ----------
        config : dict[str, float] | float, optional
            Configuration dict or base precision value, by default 1.0
        """
        if isinstance(config, dict):
            precision_config: dict[str, Any] = cast(
                dict[str, Any], config.get("precision", {}) or {}
            )
            self.extero_baseline = np.array([precision_config.get("exteroceptive_baseline", 1.0)])
            self.intero_baseline = np.array([precision_config.get("interoceptive_baseline", 0.8)])
            self.attention_gain_range = precision_config.get("attention_gain_range", [0.5, 3.0])
            self.volatility_sensitivity = precision_config.get("volatility_sensitivity", 0.1)
            self.neuromodulator_effects = precision_config.get("neuromodulator_effects", {})
            active_inference: dict[str, Any] = cast(
                dict[str, Any], config.get("active_inference", {}) or {}
            )
            self.precision_range = active_inference.get("precision_range", [0.1, 10.0])
        else:
            self.extero_baseline = np.array([config])
            self.intero_baseline = np.array([1.0])
            self.attention_gain_range = [0.5, 3.0]
            self.volatility_sensitivity = 0.1
            self.neuromodulator_effects = {}
            self.precision_range = [0.1, 10.0]

        self.extero_precision = self.extero_baseline.copy()
        self.intero_precision = self.intero_baseline.copy()
        self.attention_focus = [None]
        self.fatigue_level = np.array([0.0])
        self.cognitive_load = np.array([0.0])
        self.neuromodulators = {nm: np.array([0.5]) for nm in NeuromodulatorType}
        self.attention_gain = np.array([1.0])
        self.extero_volatility = 0.0
        self.intero_volatility = 0.0
        self._error_variance_history: list[float] = []
        self.neuromodulator_weights: dict[NeuromodulatorType, float] = {}
        self.base_precision = self.extero_baseline.copy()

    def set_neuromodulator(self, neuromodulator: NeuromodulatorType, level: float) -> None:
        """Set neuromodulator level.

        Parameters
        ----------
        neuromodulator : NeuromodulatorType
            Type of neuromodulator
        level : float
            Neuromodulator level (0.0 to 1.0)
        """
        level = max(0.0, min(1.0, level))
        self.neuromodulators[neuromodulator] = np.array([level])

    def set_fatigue(self, level: float) -> None:
        """Set fatigue level.

        Parameters
        ----------
        level : float
            Fatigue level (0.0 to 1.0)
        """
        level = max(0.0, min(1.0, level))
        self.fatigue_level = np.array([level])

    def set_cognitive_load(self, level: float) -> None:
        """Set cognitive load level.

        Parameters
        ----------
        level : float
            Cognitive load level (0.0 to 1.0)
        """
        level = max(0.0, min(1.0, level))
        self.cognitive_load = np.array([level])

    def set_neuromodulator_weight(self, neuromodulator: NeuromodulatorType, weight: float) -> None:
        """Set weight for a specific neuromodulator.

        Parameters
        ----------
        neuromodulator : NeuromodulatorType
            Type of neuromodulator
        weight : float
            Weight value
        """
        self.neuromodulators[neuromodulator] = np.array([weight])

    def calculate_weighted_precision(self, prediction_error: float) -> float:
        """Calculate precision weighted by neuromodulators.

        Parameters
        ----------
        prediction_error : float
            Prediction error value

        Returns
        -------
        float
            Weighted precision
        """
        if not hasattr(self, "neuromodulator_weights") or not self.neuromodulator_weights:
            return float(
                self.base_precision[0]
                if isinstance(self.base_precision, np.ndarray)
                else self.base_precision
            )

        total_weight = sum(self.neuromodulator_weights.values())
        return float(self.base_precision * total_weight / len(self.neuromodulator_weights))

    def update(
        self,
        extero_error_variance: float | None = None,
        intero_error_variance: float | None = None,
        attention_target: np.ndarray | None = None,
        context: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Update precision weighting based on error variances and context.

        Parameters
        ----------
        extero_error_variance : float, optional
            Exteroceptive error variance
        intero_error_variance : float, optional
            Interoceptive error variance
        attention_target : np.ndarray, optional
            Attention target array
        context : dict[str, float], optional
            Context dictionary

        Returns
        -------
        dict[str, float]
            Dictionary with updated precision values
        """
        # Validate error variances
        if extero_error_variance is not None:
            if extero_error_variance < 0 or not np.isfinite(extero_error_variance):
                raise ValueError("Error variance must be non-negative and finite")
        if intero_error_variance is not None:
            if intero_error_variance < 0 or not np.isfinite(intero_error_variance):
                raise ValueError("Error variance must be non-negative and finite")

        # Calculate base precision from error variance
        if extero_error_variance is not None:
            extero_prec = 1.0 / (extero_error_variance + 1e-6)
            self._error_variance_history.append(extero_error_variance)
            if len(self._error_variance_history) > 10:
                self._error_variance_history.pop(0)
            self.extero_volatility = float(
                np.std(self._error_variance_history)
                if len(self._error_variance_history) > 1
                else 0.0
            )
        else:
            extero_prec = float(self.extero_precision[0])

        if intero_error_variance is not None:
            intero_prec = 1.0 / (intero_error_variance + 1e-6)
        else:
            intero_prec = float(self.intero_precision[0])

        # Apply neuromodulator effects
        ne_level = float(self.neuromodulators[NeuromodulatorType.NOREPINEPHRINE][0])
        ach_level = float(self.neuromodulators[NeuromodulatorType.ACETYLCHOLINE][0])
        ne_effect = self.neuromodulator_effects.get("norepinephrine", 1.5)
        ach_effect = self.neuromodulator_effects.get("acetylcholine", 1.2)

        extero_prec *= 1.0 + ne_level * (ne_effect - 1.0)
        intero_prec *= 1.0 + ach_level * (ach_effect - 1.0)

        # Apply attention modulation
        if attention_target is not None:
            if "extero" in attention_target:
                self.attention_gain = np.array([self.attention_gain_range[1]])
                self.attention_focus = ["extero"]
            elif "intero" in attention_target:
                self.attention_gain = np.array([self.attention_gain_range[1]])
                self.attention_focus = ["intero"]
            else:
                raise ValueError("Invalid attention target")
        else:
            self.attention_gain = np.array([1.0])
            self.attention_focus = [None]

        extero_prec *= float(self.attention_gain[0])
        intero_prec *= float(self.attention_gain[0])

        # Apply fatigue penalty
        fatigue_penalty = 1.0 - 0.5 * float(self.fatigue_level[0])
        extero_prec *= fatigue_penalty
        intero_prec *= fatigue_penalty

        # Apply cognitive load penalty
        load_penalty = 1.0 - 0.3 * float(self.cognitive_load[0])
        extero_prec *= load_penalty
        intero_prec *= load_penalty

        # Apply context modulation
        if context is not None:
            threat_level = context.get("threat_level", 0.0)
            intero_prec *= 1.0 + 0.5 * threat_level

            task_demand = context.get("task_demand", 0.0)
            extero_prec *= 1.0 + 0.3 * task_demand

        # Clamp to precision range
        extero_prec = max(self.precision_range[0], min(self.precision_range[1], extero_prec))
        intero_prec = max(self.precision_range[0], min(self.precision_range[1], intero_prec))

        # Update internal state
        self.extero_precision = np.array([extero_prec])
        self.intero_precision = np.array([intero_prec])

        return {
            "exteroceptive": extero_prec,
            "interoceptive": intero_prec,
            "attention_gain": float(self.attention_gain[0]),
            "fatigue_penalty": fatigue_penalty,
        }

    def reset(self) -> None:
        """Reset precision to baseline values."""
        self.extero_precision = self.extero_baseline.copy()
        self.intero_precision = self.intero_baseline.copy()
        self.attention_focus = [None]
        self.fatigue_level = np.array([0.0])
        self.cognitive_load = np.array([0.0])
        self.neuromodulators = {nm: np.array([0.5]) for nm in NeuromodulatorType}
        self.attention_gain = np.array([1.0])
        self._error_variance_history = []

    def get_precision_matrix(self, modality: str, size: int) -> np.ndarray:
        """Generate precision matrix for given modality.

        Parameters
        ----------
        modality : str
            Modality type ('extero' or 'intero')
        size : int
            Matrix size

        Returns
        -------
        np.ndarray
            Precision matrix
        """
        if modality == "extero":
            prec = float(self.extero_precision[0])
        elif modality == "intero":
            prec = float(self.intero_precision[0])
        else:
            raise ValueError("Invalid modality")

        matrix = np.eye(size) * prec
        return matrix

    def _apply_attention(self, attention_target: np.ndarray | None) -> None:
        """Apply attention modulation.

        Parameters
        ----------
        attention_target : np.ndarray, optional
            Attention target array
        """
        if attention_target is None:
            self.attention_gain = np.array([1.0])
            self.attention_focus = [None]
        elif "extero" in attention_target:
            self.attention_gain = np.array([self.attention_gain_range[1]])
            self.attention_focus = ["extero"]
        elif "intero" in attention_target:
            self.attention_gain = np.array([self.attention_gain_range[1]])
            self.attention_focus = ["intero"]
        else:
            self.attention_gain = np.array([1.0])
            self.attention_focus = [None]


class PrecisionCalculator:
    """Calculator for exteroceptive and interoceptive precision values.

    Precision represents the inverse of uncertainty/variance in prediction
    errors, with higher precision indicating more reliable predictions.
    """

    def __init__(self, min_precision: float = 1e-6, max_precision: float = 100.0):
        """Initialize precision calculator with bounds.

        Args:
            min_precision: Minimum allowed precision value to prevent division by zero
            max_precision: Maximum allowed precision value to prevent numerical issues
        """
        if min_precision <= 0:
            raise MathematicalError("Minimum precision must be positive")
        if max_precision <= min_precision:
            raise MathematicalError("Maximum precision must be greater than minimum")

        self.min_precision = min_precision
        self.max_precision = max_precision

    def calculate_exteroceptive_precision(self, variance: float, confidence: float = 1.0) -> float:
        """Calculate exteroceptive precision (Πₑ).

        Precision is typically the inverse of variance, modulated by confidence.

        Args:
            variance: Variance of exteroceptive prediction errors
            confidence: Confidence modulation factor (default 1.0)

        Returns:
            Exteroceptive precision value

        Raises:
            MathematicalError: If variance is non-positive or confidence is non-positive
        """
        if variance <= 0:
            raise MathematicalError("Variance must be positive")
        if confidence <= 0:
            raise MathematicalError("Confidence must be positive")

        # Calculate precision as inverse variance, modulated by confidence
        precision = confidence / variance

        # Apply bounds
        precision = self._apply_precision_bounds(precision)

        return float(precision)

    def calculate_interoceptive_precision(
        self, variance: float, attention: float = 1.0, arousal: float = 1.0
    ) -> float:
        """Calculate base interoceptive precision (Πᵢ).

        Interoceptive precision can be modulated by attention and arousal states.

        Args:
            variance: Variance of interoceptive prediction errors
            attention: Attention modulation factor (default 1.0)
            arousal: Arousal modulation factor (default 1.0)

        Returns:
            Base interoceptive precision value

        Raises:
            MathematicalError: If variance is non-positive or modulation factors are non-positive
        """
        if variance <= 0:
            raise MathematicalError("Variance must be positive")
        if attention <= 0:
            raise MathematicalError("Attention factor must be positive")
        if arousal <= 0:
            raise MathematicalError("Arousal factor must be positive")

        # Calculate precision with attention and arousal modulation
        precision = (attention * arousal) / variance

        # Apply bounds
        precision = self._apply_precision_bounds(precision)

        return float(precision)

    def calculate_precision_from_samples(
        self, samples: np.ndarray, method: str = "inverse_variance"
    ) -> float:
        """Calculate precision from sample data.

        Args:
            samples: Array of prediction error samples
            method: Calculation method ("inverse_variance", "inverse_std", "fisher_information")

        Returns:
            Calculated precision value

        Raises:
            MathematicalError: If samples are invalid or method is unknown
        """
        if len(samples) == 0:
            raise MathematicalError("Sample array cannot be empty")

        samples = np.asarray(samples)

        if method == "inverse_variance":
            variance = np.var(samples, ddof=1)  # Sample variance
            if variance <= 0:
                warnings.warn("Zero or negative variance, using minimum precision")
                return self.min_precision
            precision = 1.0 / variance

        elif method == "inverse_std":
            std = np.std(samples, ddof=1)  # Sample standard deviation
            if std <= 0:
                warnings.warn("Zero or negative standard deviation, using minimum precision")
                return self.min_precision
            precision = 1.0 / std

        elif method == "fisher_information":
            # Simplified Fisher information approximation
            variance = np.var(samples, ddof=1)
            if variance <= 0:
                warnings.warn("Zero or negative variance, using minimum precision")
                return self.min_precision
            precision = len(samples) / variance

        else:
            raise MathematicalError(f"Unknown precision calculation method: {method}")

        # Apply bounds
        precision = self._apply_precision_bounds(precision)

        return float(precision)

    def validate_precision_pair(
        self, extero_precision: float, intero_precision: float
    ) -> Tuple[bool, str]:
        """Validate a pair of exteroceptive and interoceptive precision values.

        Args:
            extero_precision: Exteroceptive precision value
            intero_precision: Interoceptive precision value

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check individual bounds
            if not (self.min_precision <= extero_precision <= self.max_precision):
                return (
                    False,
                    f"Exteroceptive precision {extero_precision} out of bounds [{self.min_precision}, {self.max_precision}]",
                )

            if not (self.min_precision <= intero_precision <= self.max_precision):
                return (
                    False,
                    f"Interoceptive precision {intero_precision} out of bounds [{self.min_precision}, {self.max_precision}]",
                )

            # Check for reasonable ratios (optional validation)
            ratio = extero_precision / intero_precision
            if ratio > 1000 or ratio < 0.001:
                return (
                    False,
                    f"Precision ratio {ratio:.3f} may indicate unrealistic values",
                )

            return True, "Valid precision pair"

        except (TypeError, ValueError) as e:
            return False, f"Invalid precision values: {e}"

    def _apply_precision_bounds(self, precision: float) -> float:
        """Apply minimum and maximum bounds to precision value.

        Args:
            precision: Raw precision value

        Returns:
            Bounded precision value
        """
        if precision < self.min_precision:
            warnings.warn(f"Precision {precision} below minimum, clipping to {self.min_precision}")
            return self.min_precision
        elif precision > self.max_precision:
            warnings.warn(f"Precision {precision} above maximum, clipping to {self.max_precision}")
            return self.max_precision
        else:
            return precision

    def calculate_precision(self, data: np.ndarray, method: str = "inverse_variance") -> float:
        """Calculate precision from sample data (alias for calculate_precision_from_samples).

        Args:
            data: Array of prediction error samples
            method: Calculation method ("inverse_variance", "inverse_std", "fisher_information")

        Returns:
            Calculated precision value
        """
        return self.calculate_precision_from_samples(data, method)

    def confidence_interval(self, data: np.ndarray, confidence: float = 0.95) -> tuple:
        """Calculate confidence interval for data.

        Args:
            data: Array of samples
            confidence: Confidence level (default 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(data) == 0:
            raise MathematicalError("Cannot calculate CI for empty data")
        if len(data) < 2:
            raise MathematicalError("Need at least 2 samples for CI calculation")

        from scipy import stats

        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of mean
        df = len(data) - 1  # Degrees of freedom

        # Get t-value for confidence level
        t_value = stats.t.ppf((1 + confidence) / 2, df)

        margin = t_value * sem
        return (mean - margin, mean + margin)

    def relative_precision(self, data: np.ndarray) -> float:
        """Calculate relative precision (coefficient of variation inverse).

        Args:
            data: Array of samples

        Returns:
            Relative precision value
        """
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        if std == 0:
            return float("inf") if mean == 0 else self.max_precision

        cv = std / abs(mean) if mean != 0 else float("inf")
        if cv == 0:
            return self.max_precision

        return min(1.0 / cv, self.max_precision)

    def coefficient_of_variation(self, data: np.ndarray) -> float:
        """Calculate coefficient of variation (CV).

        Args:
            data: Array of samples

        Returns:
            Coefficient of variation
        """
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        if mean == 0:
            return 0.0 if std == 0 else float("inf")

        return float(std / abs(mean))

    def standard_error(self, data: np.ndarray) -> float:
        """Calculate standard error of the mean.

        Args:
            data: Array of samples

        Returns:
            Standard error
        """
        if len(data) == 0:
            raise MathematicalError("Cannot calculate SE for empty data")

        return float(np.std(data, ddof=1) / np.sqrt(len(data)))

    def calculate_precision_batch(self, datasets: list, method: str = "inverse_variance") -> list:
        """Calculate precision for multiple datasets.

        Args:
            datasets: List of numpy arrays
            method: Calculation method

        Returns:
            List of precision values
        """
        return [self.calculate_precision(d, method) for d in datasets]

    def precision_metrics(self, data: np.ndarray) -> dict:
        """Calculate comprehensive precision metrics.

        Args:
            data: Array of samples

        Returns:
            Dictionary with precision metrics
        """
        return {
            "precision": self.calculate_precision(data),
            "confidence_interval_95": self.confidence_interval(data, 0.95),
            "relative_precision": self.relative_precision(data),
            "coefficient_of_variation": self.coefficient_of_variation(data),
            "standard_error": self.standard_error(data),
        }

    def get_precision_info(self) -> dict:
        """Get information about precision calculation settings.

        Returns:
            Dictionary with precision calculator configuration
        """
        return {
            "min_precision": self.min_precision,
            "max_precision": self.max_precision,
            "calculation_methods": [
                "inverse_variance",
                "inverse_std",
                "fisher_information",
            ],
            "extero_modulation_factors": ["confidence"],
            "intero_modulation_factors": ["attention", "arousal"],
        }
