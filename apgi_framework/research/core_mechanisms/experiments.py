"""Experimental paradigms for APGI core mechanisms.

This module provides experimental implementations and utilities
for core APGI experimental paradigms and mechanisms.
"""

from typing import Any, Dict

import numpy as np


class ExperimentalParadigm:
    """Base class for experimental paradigms."""

    def __init__(self, paradigm_id: str) -> None:
        self.paradigm_id = paradigm_id
        self.parameters: Dict[str, Any] = {}
        self.results: list[Dict[str, Any]] = []

    def setup(self, **kwargs: Any) -> None:
        """Setup the experimental paradigm."""
        self.parameters.update(kwargs)

    def run(self, trials: int = 100) -> Dict[str, Any]:
        """Run the experimental paradigm."""
        results = {
            "paradigm_id": self.paradigm_id,
            "trials": trials,
            "parameters": self.parameters,
            "data": self._generate_data(trials),
        }
        self.results.append(results)
        return results

    def _generate_data(self, trials: int) -> np.ndarray[Any, Any]:
        """Generate synthetic data for the paradigm."""
        return np.random.randn(trials)


class ThresholdDetectionExperiment(ExperimentalParadigm):
    """Threshold detection experimental paradigm."""

    def __init__(self) -> None:
        super().__init__("threshold_detection")
        self.threshold = 0.5

    def setup(self, threshold: float = 0.5, **kwargs: Any) -> None:
        """Setup threshold detection parameters."""
        super().setup(**kwargs)
        self.threshold = threshold

    def _generate_data(self, trials: int) -> np.ndarray[Any, Any]:
        """Generate threshold detection data."""
        base_data = np.random.randn(trials)
        # Add threshold effect
        threshold_effect = np.where(base_data > self.threshold, base_data * 1.5, base_data * 0.5)
        return threshold_effect


class ConsciousnessExperiment(ExperimentalParadigm):
    """Consciousness-related experimental paradigm."""

    def __init__(self) -> None:
        super().__init__("consciousness")
        self.consciousness_level = 0.5

    def setup(self, consciousness_level: float = 0.5, **kwargs: Any) -> None:
        """Setup consciousness experiment parameters."""
        super().setup(**kwargs)
        self.consciousness_level = consciousness_level

    def _generate_data(self, trials: int) -> np.ndarray[Any, Any]:
        """Generate consciousness experiment data."""
        base_data = np.random.randn(trials)
        # Modulate by consciousness level
        consciousness_modulation = base_data * (1 + self.consciousness_level)
        return consciousness_modulation  # type: ignore[no-any-return]


class SomaticMarkerExperiment(ExperimentalParadigm):
    """Somatic marker experimental paradigm."""

    def __init__(self) -> None:
        super().__init__("somatic_marker")
        self.somatic_gain = 1.0

    def setup(self, somatic_gain: float = 1.0, **kwargs: Any) -> None:
        """Setup somatic marker parameters."""
        super().setup(**kwargs)
        self.somatic_gain = somatic_gain

    def _generate_data(self, trials: int) -> np.ndarray[Any, Any]:
        """Generate somatic marker data."""
        base_data = np.random.randn(trials)
        # Apply somatic gain
        somatic_effect = base_data * self.somatic_gain
        return somatic_effect  # type: ignore[no-any-return]


# Experimental module interface
experimental = {
    "ThresholdDetectionExperiment": ThresholdDetectionExperiment,
    "ConsciousnessExperiment": ConsciousnessExperiment,
    "SomaticMarkerExperiment": SomaticMarkerExperiment,
    "ExperimentalParadigm": ExperimentalParadigm,
}


def get_experiment(experiment_type: str) -> ExperimentalParadigm:
    """Get an experiment instance by type."""
    experiments = {
        "threshold_detection": ThresholdDetectionExperiment,
        "consciousness": ConsciousnessExperiment,
        "somatic_marker": SomaticMarkerExperiment,
    }

    if experiment_type not in experiments:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    return experiments[experiment_type]()


def run_standard_experiment(
    experiment_type: str, trials: int = 100, **kwargs: Any
) -> Dict[str, Any]:
    """Run a standard experiment by type."""
    experiment = get_experiment(experiment_type)
    experiment.setup(**kwargs)
    return experiment.run(trials)


__all__ = [
    "ExperimentalParadigm",
    "ThresholdDetectionExperiment",
    "ConsciousnessExperiment",
    "SomaticMarkerExperiment",
    "experimental",
    "get_experiment",
    "run_standard_experiment",
]
