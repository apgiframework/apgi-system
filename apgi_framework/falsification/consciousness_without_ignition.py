"""Consciousness without ignition falsification test."""

import logging
from datetime import datetime


class ConsciousnessWithoutIgnitionTest:
    """Tests whether consciousness can occur without full neural ignition signatures.

    This falsification test examines the converse of the primary test: whether
    subjective reports of consciousness can occur in the absence of the complete
    set of neural signatures (P3b, gamma synchrony, BOLD activation, PCI).
    """

    def __init__(self) -> None:
        """Initialize the test with default parameters."""
        self.logger = logging.getLogger(__name__)
        self.signature_thresholds = {
            "p3b_amplitude": 5.0,  # μV
            "gamma_plv": 0.3,  # Phase locking value
            "bold_activation": 0.1,  # Percent signal change
            "pci_value": 0.3,  # Perturbational complexity index
        }
        self.consciousness_thresholds = {
            "subjective_report": True,
            "forced_choice_accuracy": 0.6,  # Above chance
            "confidence_rating": 0.5,
        }

    def run_test(self, n_trials: int = 100, noise_level: float = 0.1) -> dict:
        """Run the consciousness without ignition test.

        Args:
            n_trials: Number of experimental trials to simulate
            noise_level: Level of neural noise to add (0.0-1.0)

        Returns:
            Dictionary containing test results and statistics
        """
        import numpy as np

        # Simulate trials
        consciousness_without_ignition = 0
        ignition_without_consciousness = 0
        both_present = 0
        both_absent = 0

        trial_results = []

        for trial in range(n_trials):
            # Simulate neural signatures with noise
            p3b_present = np.random.random() > (0.3 + noise_level)
            gamma_present = np.random.random() > (0.3 + noise_level)
            bold_present = np.random.random() > (0.3 + noise_level)
            pci_present = np.random.random() > (0.3 + noise_level)

            # Full ignition requires all signatures
            full_ignition = p3b_present and gamma_present and bold_present and pci_present

            # Simulate consciousness measures
            subjective_conscious = np.random.random() > (0.4 - noise_level * 0.5)
            forced_choice_acc = (
                np.random.beta(8, 4) if subjective_conscious else np.random.beta(4, 8)
            )
            confidence = forced_choice_acc + np.random.normal(0, 0.1)
            confidence = np.clip(confidence, 0, 1)

            full_consciousness = (
                subjective_conscious
                and forced_choice_acc > self.consciousness_thresholds["forced_choice_accuracy"]
                and confidence > self.consciousness_thresholds["confidence_rating"]
            )

            # Categorize trial
            if full_consciousness and not full_ignition:
                consciousness_without_ignition += 1
                category = "consciousness_without_ignition"
            elif full_ignition and not full_consciousness:
                ignition_without_consciousness += 1
                category = "ignition_without_consciousness"
            elif full_consciousness and full_ignition:
                both_present += 1
                category = "both_present"
            else:
                both_absent += 1
                category = "both_absent"

            trial_results.append(
                {
                    "trial_id": trial,
                    "category": category,
                    "full_ignition": full_ignition,
                    "full_consciousness": full_consciousness,
                    "p3b_present": p3b_present,
                    "gamma_present": gamma_present,
                    "bold_present": bold_present,
                    "pci_present": pci_present,
                    "subjective_conscious": subjective_conscious,
                    "forced_choice_acc": forced_choice_acc,
                    "confidence": confidence,
                }
            )

        # Calculate statistics
        consciousness_without_ignition_rate = consciousness_without_ignition / n_trials
        ignition_without_consciousness_rate = ignition_without_consciousness / n_trials

        # Statistical test
        from scipy import stats

        # Ensure non-negative values for contingency table
        contingency_table = [
            [
                max(0, consciousness_without_ignition),
                max(0, both_present - consciousness_without_ignition),
            ],
            [max(0, both_absent), max(0, ignition_without_consciousness)],
        ]

        chi2_stat = None
        p_value = None
        try:
            chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
        except (ValueError, RuntimeError) as e:
            self.logger.warning(
                f"Chi-square test failed: {e}. Using conservative fallback p-value."
            )
            p_value = 0.5  # Conservative fallback that won't falsely indicate significance

        # Determine if framework is falsified
        falsified = consciousness_without_ignition_rate > 0.1 and p_value < 0.05

        return {
            "status": "completed",
            "test_name": "ConsciousnessWithoutIgnitionTest",
            "timestamp": datetime.now().isoformat(),
            "n_trials": n_trials,
            "results": {
                "consciousness_without_ignition_count": consciousness_without_ignition,
                "consciousness_without_ignition_rate": consciousness_without_ignition_rate,
                "ignition_without_consciousness_count": ignition_without_consciousness,
                "ignition_without_consciousness_rate": ignition_without_consciousness_rate,
                "both_present_count": both_present,
                "both_absent_count": both_absent,
            },
            "statistical_test": {"chi2_statistic": chi2_stat, "p_value": p_value},
            "framework_falsified": falsified,
            "interpretation": self._interpret_results(consciousness_without_ignition_rate, p_value),
            "trial_data": trial_results,
        }

    def _interpret_results(self, rate: float, p_value: float) -> str:
        """Interpret the test results."""
        if rate > 0.2 and p_value < 0.01:
            return "Strong evidence that consciousness can occur without full ignition - framework falsified"
        elif rate > 0.1 and p_value < 0.05:
            return "Moderate evidence that consciousness can occur without full ignition - framework challenged"
        elif rate > 0.05:
            return "Weak evidence that consciousness can occur without full ignition - framework partially supported"
        else:
            return (
                "No evidence that consciousness occurs without full ignition - framework supported"
            )
