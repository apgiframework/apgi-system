"""Stub module for masking paradigm task.

This module provides type stubs for masking paradigm experimental task
that is being migrated from the old apgi_simulation structure.
"""

from enum import Enum
from typing import Any, Dict, List

import numpy as np


class MaskType(Enum):
    """Types of masking in masking paradigm."""

    FORWARD = "forward"
    BACKWARD = "backward"
    META_CONTRAST = "meta_contrast"


class MaskingParadigmTask:
    """Masking paradigm experimental task."""

    def __init__(
        self,
        target_duration_ms: int = 20,
        mask_duration_ms: int = 40,
        soas: List[int] | None = None,
        num_trials_per_condition: int = 10,
    ):
        """Initialize masking paradigm task.

        Parameters
        ----------
        target_duration_ms : int, optional
            Duration of target stimulus in milliseconds, by default 20
        mask_duration_ms : int, optional
            Duration of mask stimulus in milliseconds, by default 40
        soas : list[int], optional
            List of stimulus onset asynchronies, by default None
        num_trials_per_condition : int, optional
            Number of trials per condition, by default 10
        """
        self.target_duration_ms = target_duration_ms
        self.mask_duration_ms = mask_duration_ms
        self.soas = soas or [20, 40, 60, 80, 100, 120, 140, 160]
        self.num_trials_per_condition = num_trials_per_condition
        self.mask_type = MaskType.FORWARD
        self.trials = self._generate_trials()
        self.results: List[Dict[str, Any]] = []

    def create_target_stimulus(self) -> np.ndarray:
        """Create a target stimulus.

        Returns
        -------
        np.ndarray
            Target stimulus image
        """
        return np.random.rand(64, 64)

    def create_mask_stimulus(self) -> np.ndarray:
        """Create a mask stimulus.

        Returns
        -------
        np.ndarray
            Mask stimulus image
        """
        return np.random.rand(64, 64)

    def calculate_masking_effect(self, soa_ms: int) -> float:
        """Calculate masking effect given stimulus onset asynchrony.

        Parameters
        ----------
        soa_ms : int
            Stimulus onset asynchrony in milliseconds

        Returns
        -------
        float
            Masking effect magnitude
        """
        # Simplified masking curve
        if self.mask_type == MaskType.FORWARD:
            if soa_ms < 50:
                return 0.9
            elif soa_ms < 100:
                return 0.5
            else:
                return 0.1
        else:
            return 0.3

    def run_trial(self, apgi_framework: Any, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single masking trial.

        Parameters
        ----------
        apgi_framework : Any
            APGI framework instance
        trial : Dict[str, Any]
            Trial configuration

        Returns
        -------
        Dict[str, Any]
            Trial results
        """
        soa_ms = trial["soa_ms"]
        masking_effect = self.calculate_masking_effect(soa_ms)
        return {
            "trial_number": trial["trial_number"],
            "soa_ms": soa_ms,
            "mask_type": self.mask_type,
            "target_detected": np.random.random() > masking_effect,
            "masking_effect": masking_effect,
            "ignition_count": np.random.randint(0, 5),
        }

    def _generate_trials(self) -> List[Dict[str, Any]]:
        """Generate trial configurations.

        Returns
        -------
        List[Dict[str, Any]]
            List of trial configurations
        """
        trials = []
        trial_num = 0
        for soa in self.soas:
            for _ in range(self.num_trials_per_condition):
                trial = {
                    "trial_number": trial_num,
                    "soa_ms": soa,
                }
                trials.append(trial)
                trial_num += 1
        return trials

    def run_all_trials(self, apgi_framework: Any) -> Dict[str, Any]:
        """Run all trials and return analysis.

        Parameters
        ----------
        apgi_framework : Any
            APGI framework instance

        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        self.results = []
        for trial in self.trials:
            result = self.run_trial(apgi_framework, trial)
            self.results.append(result)

        # Calculate analysis by SOA
        by_soa = {}
        for soa in self.soas:
            soa_results = [r for r in self.results if r["soa_ms"] == soa]
            if soa_results:
                detections = sum(1 for r in soa_results if r["target_detected"])
                by_soa[soa] = {
                    "total_trials": len(soa_results),
                    "detections": detections,
                    "detection_rate": detections / len(soa_results),
                    "suppression_rate": 1.0 - (detections / len(soa_results)),
                    "avg_ignition_strength": 0.5,
                }

        # Calculate masking curve
        masking_curve = [
            {"soa_ms": soa, "detection_rate": by_soa[soa]["detection_rate"]}
            for soa in self.soas
            if soa in by_soa
        ]

        # Calculate masking effect
        if len(masking_curve) >= 2:
            masking_effect = (
                masking_curve[-1]["detection_rate"] - masking_curve[0]["detection_rate"]
            )
        else:
            masking_effect = 0.0

        return {
            "total_trials": len(self.results),
            "overall_detection_rate": sum(1 for r in self.results if r["target_detected"])
            / len(self.results),
            "overall_suppression_rate": 1.0
            - (sum(1 for r in self.results if r["target_detected"]) / len(self.results)),
            "masking_effect": masking_effect,
            "by_soa": by_soa,
            "masking_curve": masking_curve,
            "task_parameters": {
                "target_duration_ms": self.target_duration_ms,
                "mask_duration_ms": self.mask_duration_ms,
                "soas": self.soas,
                "num_trials_per_condition": self.num_trials_per_condition,
            },
        }

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trial results.

        Parameters
        ----------
        results : List[Dict[str, Any]]
            Trial results

        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        if not results:
            return {"total_trials": 0, "overall_detection_rate": 0.0}

        overall_detection_rate = sum(1 for r in results if r["target_detected"]) / len(results)
        return {
            "total_trials": len(results),
            "overall_detection_rate": overall_detection_rate,
            "overall_suppression_rate": 1.0 - overall_detection_rate,
        }

    def save_results(self, results: List[Dict[str, Any]], filepath: str) -> None:
        """Save results to file.

        Parameters
        ----------
        results : List[Dict[str, Any]]
            Trial results
        filepath : str
            Output file path
        """
        import json

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

    def reset(self) -> None:
        """Reset task to initial state."""
        self.trials = self._generate_trials()
        self.results = []
