"""Stub module for change blindness task.

This module provides type stubs for change blindness experimental task
that is being migrated from the old apgi_simulation structure.
"""

from enum import Enum
from typing import Any, Dict

import numpy as np


class ChangeType(Enum):
    """Types of changes in change blindness task."""

    COLOR = "color"
    POSITION = "position"
    APPEARANCE = "appearance"
    DISAPPEARANCE = "disappearance"


class ChangeBlindnessTask:
    """Change blindness experimental task."""

    def __init__(
        self,
        display_size: tuple = (512, 512),
        max_alternations: int = 10,
        num_trials_per_condition: int = 5,
        change_magnitudes: list[float] | None = None,
        presentation_duration_ms: float = 240.0,
        blank_duration_ms: float = 80.0,
    ):
        """Initialize change blindness task.

        Parameters
        ----------
        display_size : tuple, optional
            Size of display area, by default (512, 512)
        max_alternations : int, optional
            Maximum number of alternations, by default 10
        num_trials_per_condition : int, optional
            Number of trials per condition, by default 5
        change_magnitudes : list[float], optional
            List of change magnitudes, by default None
        """
        self.display_size = display_size
        self.change_type = ChangeType.COLOR
        self.max_alternations = max_alternations
        self.num_trials_per_condition = num_trials_per_condition
        self.change_magnitudes = change_magnitudes or [0.3, 0.5, 0.7]
        self.presentation_duration_ms = presentation_duration_ms
        self.blank_duration_ms = blank_duration_ms
        self.trials = self._generate_trials()

    def create_scene(self, num_objects: int = 5) -> Dict[str, Any]:
        """Create a scene with multiple objects.

        Parameters
        ----------
        num_objects : int, optional
            Number of objects in the scene, by default 5

        Returns
        -------
        Dict[str, Any]
            Scene description
        """
        objects = []
        for i in range(num_objects):
            objects.append(
                {
                    "id": i,
                    "position": np.random.rand(2) * self.display_size[0],
                    "color": np.random.rand(3),
                    "size": np.random.uniform(10, 50),
                }
            )
        return {"objects": objects, "change_type": self.change_type}

    def introduce_change(self, scene: Dict[str, Any], change_type: ChangeType) -> Dict[str, Any]:
        """Introduce a change to the scene.

        Parameters
        ----------
        scene : Dict[str, Any]
            Original scene
        change_type : ChangeType
            Type of change to introduce

        Returns
        -------
        Dict[str, Any]
            Modified scene with change
        """
        modified_scene = scene.copy()
        if scene["objects"]:
            obj_idx = np.random.randint(len(scene["objects"]))
            if change_type == ChangeType.COLOR:
                modified_scene["objects"][obj_idx]["color"] = np.random.rand(3)
            elif change_type == ChangeType.POSITION:
                modified_scene["objects"][obj_idx]["position"] = (
                    np.random.rand(2) * self.display_size[0]
                )
        return modified_scene

    def detect_change(self, original: Dict[str, Any], modified: Dict[str, Any]) -> bool:
        """Detect if a change occurred between scenes.

        Parameters
        ----------
        original : Dict[str, Any]
            Original scene
        modified : Dict[str, Any]
            Modified scene

        Returns
        -------
        bool
            Whether change was detected
        """
        # Simplified detection
        return len(original["objects"]) != len(modified["objects"])

    def _generate_trials(self) -> list[Dict[str, Any]]:
        """Generate trial configurations.

        Returns
        -------
        list[Dict[str, Any]]
            List of trial configurations
        """
        trials = []
        for i, magnitude in enumerate(self.change_magnitudes):
            for j in range(self.num_trials_per_condition):
                trial = {
                    "trial_number": i * self.num_trials_per_condition + j,
                    "change_type": self.change_type,
                    "change_magnitude": magnitude,
                }
                trials.append(trial)
        return trials

    def run_trial(self, apgi_framework: Any, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single trial.

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
        # Simulate trial execution
        import random

        change_detected = random.random() > 0.3  # 70% detection rate
        alternations_to_detection = (
            random.randint(1, self.max_alternations) if change_detected else self.max_alternations
        )
        time_to_detection = random.uniform(100, 2000) if change_detected else 2000
        ignition_signal_strength = random.uniform(0.1, 1.0)

        return {
            "trial_number": trial["trial_number"],
            "change_type": trial["change_type"],
            "change_magnitude": trial["change_magnitude"],
            "change_detected": change_detected,
            "alternations_to_detection": alternations_to_detection,
            "time_to_detection": time_to_detection,
            "ignition_signal_strength": ignition_signal_strength,
        }

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

        # Calculate analysis by change magnitude
        by_magnitude = {}
        for magnitude in self.change_magnitudes:
            mag_results = [r for r in self.results if r["change_magnitude"] == magnitude]
            if mag_results:
                detections = sum(1 for r in mag_results if r["change_detected"])
                by_magnitude[magnitude] = {
                    "total_trials": len(mag_results),
                    "detections": detections,
                    "detection_rate": detections / len(mag_results),
                    "avg_alternations_to_detection": sum(
                        r["alternations_to_detection"] for r in mag_results if r["change_detected"]
                    )
                    / max(detections, 1),
                    "avg_time_to_detection": sum(
                        r["time_to_detection"] for r in mag_results if r["change_detected"]
                    )
                    / max(detections, 1),
                }

        # Calculate overall statistics
        overall_detection_rate = sum(1 for r in self.results if r["change_detected"]) / len(
            self.results
        )

        return {
            "total_trials": len(self.results),
            "overall_detection_rate": overall_detection_rate,
            "overall_blindness_rate": 1.0 - overall_detection_rate,
            "overall_avg_alternations": sum(
                r["alternations_to_detection"] for r in self.results if r["change_detected"]
            )
            / max(sum(1 for r in self.results if r["change_detected"]), 1),
            "overall_avg_time_ms": sum(
                r["time_to_detection"] for r in self.results if r["change_detected"]
            )
            / max(sum(1 for r in self.results if r["change_detected"]), 1),
            "by_magnitude": by_magnitude,
            "by_change_type": {self.change_type.value: by_magnitude},
            "task_parameters": {
                "change_magnitudes": self.change_magnitudes,
                "num_trials_per_condition": self.num_trials_per_condition,
                "max_alternations": self.max_alternations,
            },
        }

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze trial results.

        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        if not self.results:
            return {"total_trials": 0, "overall_detection_rate": 0.0}

        overall_detection_rate = sum(1 for r in self.results if r["change_detected"]) / len(
            self.results
        )
        return {
            "total_trials": len(self.results),
            "overall_detection_rate": overall_detection_rate,
        }

    def save_results(self, filename: str) -> None:
        """Save results to a JSON file.

        Parameters
        ----------
        filename : str
            Path to save results
        """
        import json

        with open(filename, "w") as f:
            json.dump({"results": self.results}, f, indent=2)

    def print_results(self, analysis: Dict[str, Any]) -> None:
        """Print analysis results to console.

        Parameters
        ----------
        analysis : Dict[str, Any]
            Analysis results to print
        """
        print("\n" + "=" * 50)
        print("CHANGE BLINDNESS TASK RESULTS")
        print("=" * 50)
        print(f"Total Trials: {analysis['total_trials']}")
        print(f"Detection Rate: {analysis['overall_detection_rate']:.1%}")
        print(f"Blindness Rate: {analysis['overall_blindness_rate']:.1%}")
        print(f"Avg Alternations: {analysis['overall_avg_alternations']:.1f}")
        print(f"Avg Time to Detection: {analysis['overall_avg_time_ms']:.0f} ms")
