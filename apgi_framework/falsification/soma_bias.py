"""
Soma bias falsification test.
"""

import logging
from datetime import datetime
from typing import Any, Dict, cast


class SomaBiasTest:
    """
    Tests whether somatic (interoceptive) biases systematically affect framework predictions.

    This falsification test examines whether the framework's predictions are
    systematically distorted by somatic/interoceptive influences, which would
    indicate a bias in the model.
    """

    def __init__(self) -> None:
        """Initialize the test with default parameters."""
        self.logger = logging.getLogger(__name__)
        self.soma_bias_levels = [-0.5, -0.2, 0.0, 0.2, 0.5]  # Negative to positive bias
        self.neutral_baseline = 0.0

    def run_test(self, n_trials: int = 100, bias_strength: float = 0.3) -> dict:
        """
        Run the soma bias test.

        Args:
            n_trials: Number of trials per bias level
            bias_strength: Maximum strength of somatic bias to apply

        Returns:
            Dictionary containing test results and bias analysis
        """
        import numpy as np
        from scipy import stats

        results_by_bias = []

        for bias_level in self.soma_bias_levels:
            bias_results = self._run_bias_simulation(n_trials, bias_level, bias_strength)
            results_by_bias.append({"bias_level": bias_level, "results": bias_results})

        # Analyze bias effects
        falsification_rates = [
            cast(Dict[str, Any], r)["results"]["falsification_rate"] for r in results_by_bias
        ]
        bias_levels = [r["bias_level"] for r in results_by_bias]

        # Test for systematic bias (correlation between bias level and falsification rate)
        correlation, p_value = stats.pearsonr(bias_levels, falsification_rates)

        # Test for bias asymmetry (different effects for positive vs negative bias)
        positive_bias_rates = []
        negative_bias_rates = []
        for r in results_by_bias:
            r_dict = cast(Dict[str, Any], r)
            bias_level = cast(float, r_dict["bias_level"])
            results_dict = cast(Dict[str, Any], r_dict["results"])
            falsification_rate = cast(float, results_dict["falsification_rate"])
            if bias_level > 0:
                positive_bias_rates.append(falsification_rate)
            elif bias_level < 0:
                negative_bias_rates.append(falsification_rate)

        asymmetry_test = None
        asymmetry_p = None
        if positive_bias_rates and negative_bias_rates:
            t_stat, t_p = stats.ttest_ind(positive_bias_rates, negative_bias_rates)
            asymmetry_test = float(t_stat)
            asymmetry_p = float(t_p)

        # Determine if framework is falsified
        falsified = (abs(correlation) > 0.7 and p_value < 0.05) or (
            asymmetry_p is not None and asymmetry_p < 0.05
        )

        return {
            "status": "completed",
            "test_name": "SomaBiasTest",
            "timestamp": datetime.now().isoformat(),
            "n_trials_per_bias": n_trials,
            "bias_strength": bias_strength,
            "bias_levels_tested": self.soma_bias_levels,
            "results_by_bias": results_by_bias,
            "bias_analysis": {
                "correlation_coefficient": float(correlation),
                "correlation_p_value": float(p_value),
                "asymmetry_t_stat": asymmetry_test,
                "asymmetry_p_value": asymmetry_p,
                "positive_bias_mean": (
                    float(np.mean(positive_bias_rates)) if positive_bias_rates else None
                ),
                "negative_bias_mean": (
                    float(np.mean(negative_bias_rates)) if negative_bias_rates else None
                ),
            },
            "framework_falsified": falsified,
            "interpretation": self._interpret_bias(correlation, p_value, asymmetry_p or 0.0),
        }

    def _run_bias_simulation(self, n_trials: int, bias_level: float, bias_strength: float) -> dict:
        """Run simulation for a specific bias level."""
        import numpy as np

        falsifying_trials = 0

        for trial in range(n_trials):
            # Apply somatic bias to neural signature generation
            bias_effect = bias_level * bias_strength

            # Simulate neural signatures with bias
            p3b_amp = np.random.normal(6 + bias_effect * 2, 2)  # Bias affects P3b
            gamma_plv = np.random.beta(3 + bias_effect * 2, 7 - bias_effect)  # Bias affects gamma
            bold_act = np.random.normal(0.12 + bias_effect * 0.05, 0.05)  # Bias affects BOLD
            pci_val = np.random.normal(0.35 + bias_effect * 0.1, 0.1)  # Bias affects PCI

            # Standard thresholds
            signatures_met = p3b_amp > 5.0 and gamma_plv > 0.3 and bold_act > 0.1 and pci_val > 0.3

            # Simulate consciousness (also affected by bias through interoceptive awareness)
            bias_consciousness_effect = bias_level * bias_strength * 0.3
            forced_choice_acc = np.random.beta(
                8 + bias_consciousness_effect * 4, 4 - bias_consciousness_effect * 2
            )
            confidence = forced_choice_acc + np.random.normal(0, 0.1)
            confidence = np.clip(confidence, 0, 1)

            consciousness_met = forced_choice_acc > 0.6 and confidence > 0.5

            # Check for falsification
            if signatures_met and not consciousness_met:
                falsifying_trials += 1

        return {
            "falsifying_trials": falsifying_trials,
            "falsification_rate": falsifying_trials / n_trials,
        }

    def _interpret_bias(self, correlation: float, corr_p: float, asymmetry_p: float) -> str:
        """Interpret the bias test results."""
        if abs(correlation) > 0.8 and corr_p < 0.01:
            return f"Strong systematic bias detected (r={correlation:.3f}) - framework falsified"
        elif abs(correlation) > 0.6 and corr_p < 0.05:
            return f"Moderate systematic bias detected (r={correlation:.3f}) - framework challenged"
        elif asymmetry_p < 0.05:
            return "Significant bias asymmetry detected - framework questionable"
        elif abs(correlation) > 0.3:
            return (
                f"Weak bias effects detected (r={correlation:.3f}) - framework partially supported"
            )
        else:
            # Spec 1.1.0: nonsignificance alone establishes neither absence of bias nor support.
            return "No significant bias effects detected - indeterminate (absence not established)"
