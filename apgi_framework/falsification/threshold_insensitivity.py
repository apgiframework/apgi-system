"""Threshold insensitivity falsification test."""

import logging
from datetime import datetime
from typing import Any, Dict, List, cast


class ThresholdInsensitivityTest:
    """Tests whether the framework is insensitive to reasonable threshold variations.

    This falsification test examines whether small changes in the thresholds
    for neural signatures and consciousness measures lead to dramatically different
    conclusions about framework validity.
    """

    def __init__(self) -> None:
        """Initialize the test with default parameters."""
        self.logger = logging.getLogger(__name__)
        self.base_thresholds = {
            "p3b_amplitude": 5.0,
            "gamma_plv": 0.3,
            "bold_activation": 0.1,
            "pci_value": 0.3,
            "forced_choice_accuracy": 0.6,
            "confidence_rating": 0.5,
        }

    def run_test(self, n_trials: int = 100, threshold_variations: float = 0.2) -> dict:
        """Run the threshold insensitivity test.

        Args:
            n_trials: Number of trials per threshold configuration
            threshold_variations: Fraction to vary thresholds (e.g., 0.2 = ±20%)

        Returns:
            Dictionary containing test results and sensitivity analysis
        """
        import numpy as np

        # Test different threshold configurations
        threshold_configs = []
        results_by_config = []

        # Base configuration
        threshold_configs.append({"name": "baseline", "thresholds": self.base_thresholds.copy()})

        # Lower thresholds
        lower_thresholds = {}
        for key, value in self.base_thresholds.items():
            lower_thresholds[key] = value * (1 - threshold_variations)
        threshold_configs.append({"name": "lower", "thresholds": lower_thresholds})

        # Higher thresholds
        higher_thresholds = {}
        for key, value in self.base_thresholds.items():
            higher_thresholds[key] = value * (1 + threshold_variations)
        threshold_configs.append({"name": "higher", "thresholds": higher_thresholds})

        # Run simulation for each configuration
        for config in threshold_configs:
            config_results = self._run_configuration_simulation(
                n_trials, cast(Dict[str, Any], config["thresholds"])
            )
            results_by_config.append(
                {
                    "config_name": config["name"],
                    "thresholds": config["thresholds"],
                    "results": config_results,
                }
            )

        # Analyze sensitivity
        falsification_rates: List[float] = []
        for r in results_by_config:
            r_dict = cast(Dict[str, Any], r)
            results_dict = cast(Dict[str, Any], r_dict["results"])
            falsification_rates.append(cast(float, results_dict["falsification_rate"]))
        sensitivity = float(np.std(falsification_rates) / (np.mean(falsification_rates) + 1e-8))

        # Determine if framework is falsified (too sensitive)
        falsified = sensitivity > 0.5  # High sensitivity indicates threshold dependence

        return {
            "status": "completed",
            "test_name": "ThresholdInsensitivityTest",
            "timestamp": datetime.now().isoformat(),
            "n_trials_per_config": n_trials,
            "threshold_variations": threshold_variations,
            "configurations": results_by_config,
            "sensitivity_analysis": {
                "falsification_rates": falsification_rates,
                "mean_rate": float(np.mean(falsification_rates)),
                "std_rate": float(np.std(falsification_rates)),
                "sensitivity_coefficient": sensitivity,
            },
            "framework_falsified": falsified,
            "interpretation": self._interpret_sensitivity(sensitivity),
        }

    def _run_configuration_simulation(self, n_trials: int, thresholds: dict) -> dict:
        """Run simulation for a specific threshold configuration."""
        import numpy as np

        falsifying_trials = 0

        for trial in range(n_trials):
            # Simulate neural data
            p3b_amp = np.random.normal(6, 2)  # μV
            gamma_plv = np.random.beta(3, 7)  # Phase locking value
            bold_act = np.random.normal(0.12, 0.05)  # Percent signal change
            pci_val = np.random.normal(0.35, 0.1)  # PCI value

            # Check if signatures meet thresholds
            signatures_met = (
                p3b_amp > thresholds["p3b_amplitude"]
                and gamma_plv > thresholds["gamma_plv"]
                and bold_act > thresholds["bold_activation"]
                and pci_val > thresholds["pci_value"]
            )

            # Simulate consciousness measures
            forced_choice_acc = np.random.beta(8, 4) if signatures_met else np.random.beta(4, 8)
            confidence = forced_choice_acc + np.random.normal(0, 0.1)
            confidence = np.clip(confidence, 0, 1)

            consciousness_met = (
                forced_choice_acc > thresholds["forced_choice_accuracy"]
                and confidence > thresholds["confidence_rating"]
            )

            # Check for falsification (signatures without consciousness)
            if signatures_met and not consciousness_met:
                falsifying_trials += 1

        return {
            "falsifying_trials": falsifying_trials,
            "falsification_rate": falsifying_trials / n_trials,
        }

    def _interpret_sensitivity(self, sensitivity: float) -> str:
        """Interpret the sensitivity results."""
        if sensitivity > 1.0:
            return "Extremely high threshold sensitivity - framework falsified"
        elif sensitivity > 0.5:
            return "High threshold sensitivity - framework unreliable"
        elif sensitivity > 0.2:
            return "Moderate threshold sensitivity - framework questionable"
        elif sensitivity > 0.1:
            return "Low threshold sensitivity - framework reasonably robust"
        else:
            return "Very low threshold sensitivity - framework highly robust"
