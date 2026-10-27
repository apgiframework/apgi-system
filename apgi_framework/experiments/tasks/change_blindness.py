"""Change blindness (flicker paradigm) on the APGI system.

A scene and a modified copy alternate, separated by blanks. Each alternation is presented to
the APGI system; the change is noticed when the modified display evokes a clearly stronger
response than the original (an ignition only for the modified scene, or a signal peak that is
higher by a margin that grows with the size of the change). Small changes can go unnoticed for
many alternations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ._records import (
    STIMULUS_DIM,
    TaskRecord,
    blank_stimulus,
    draw_seed,
    present_stimulus,
    save_json,
    scaled_timestep,
)

#: Task time represented by one system step (ms).
STEP_MS = 60.0
#: A modified display must beat the original's signal peak by this fraction of the threshold.
DETECTION_MARGIN = 0.15
#: Differences smaller than this are not a change.
TOLERANCE = 1e-9


class ChangeType(Enum):
    """Types of changes in change blindness task."""

    COLOR = "color"
    POSITION = "position"
    APPEARANCE = "appearance"
    DISAPPEARANCE = "disappearance"


@dataclass
class ChangeBlindnessTrial(TaskRecord):
    """One planned flicker trial."""

    trial_number: int
    change_type: ChangeType
    change_magnitude: float


@dataclass
class ChangeBlindnessResult(TaskRecord):
    """Outcome of one executed trial."""

    trial_number: int
    change_type: ChangeType
    change_magnitude: float
    change_detected: bool
    alternations_to_detection: int
    time_to_detection: float
    ignition_signal_strength: float


class ChangeBlindnessTask:
    """Change blindness experimental task.

    Parameters
    ----------
    display_size : tuple
        Width and height of the scene.
    max_alternations : int
        Alternations shown before giving up.
    num_trials_per_condition : int
        Trials per change magnitude.
    change_magnitudes : list of float, optional
        Change sizes in (0, 1].
    presentation_duration_ms, blank_duration_ms : float
        Time of each display and of each blank in between.
    change_type : ChangeType
        What changes between the displays.
    seed : int, optional
        Seed for scenes and changes; by default drawn from numpy's global state.
    """

    def __init__(
        self,
        display_size: tuple = (512, 512),
        max_alternations: int = 10,
        num_trials_per_condition: int = 5,
        change_magnitudes: Optional[List[float]] = None,
        presentation_duration_ms: float = 240.0,
        blank_duration_ms: float = 80.0,
        change_type: ChangeType = ChangeType.COLOR,
        seed: Optional[int] = None,
    ):
        self.display_size = display_size
        self.change_type = change_type
        self.max_alternations = max_alternations
        self.num_trials_per_condition = num_trials_per_condition
        self.change_magnitudes = list(change_magnitudes) if change_magnitudes else [0.3, 0.5, 0.7]
        self.presentation_duration_ms = presentation_duration_ms
        self.blank_duration_ms = blank_duration_ms
        self._seed = draw_seed(seed)
        self.rng = np.random.default_rng(self._seed)

        self.results: List[ChangeBlindnessResult] = []
        self.trials: List[ChangeBlindnessTrial] = self._generate_trials()

    # -- scenes -----------------------------------------------------------------------------
    def create_scene(self, num_objects: int = 5) -> Dict[str, Any]:
        """A scene of ``num_objects`` coloured objects at random positions."""
        objects = [
            {
                "id": i,
                "position": self.rng.random(2) * self.display_size[0],
                "color": self.rng.random(3),
                "size": float(self.rng.uniform(10, 50)),
            }
            for i in range(num_objects)
        ]
        return {"objects": objects, "change_type": self.change_type}

    def introduce_change(
        self, scene: Dict[str, Any], change_type: ChangeType, magnitude: float = 0.5
    ) -> Dict[str, Any]:
        """A copy of ``scene`` with one object changed; ``scene`` itself is left untouched.

        ``magnitude`` in (0, 1] scales the change: colour shift (per channel, towards the far
        end of the range), position shift (fraction of the display width), size change.
        """
        modified = {**scene, "objects": copy.deepcopy(scene["objects"])}
        objects = modified["objects"]
        if not objects:
            return modified

        index = int(self.rng.integers(len(objects)))
        target = objects[index]
        magnitude = float(np.clip(magnitude, 0.05, 1.0))
        if change_type == ChangeType.COLOR:
            color = np.asarray(target["color"], dtype=float)
            # Move every channel towards the far end so the change is always visible
            target["color"] = np.clip(
                np.where(color < 0.5, color + magnitude, color - magnitude), 0, 1
            )
        elif change_type == ChangeType.POSITION:
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            shift = (
                magnitude * self.display_size[0] / 4.0 * np.array([np.cos(angle), np.sin(angle)])
            )
            target["position"] = np.clip(
                np.asarray(target["position"], dtype=float) + shift, 0, self.display_size[0]
            )
        elif change_type == ChangeType.APPEARANCE:
            target["size"] = float(target["size"] * (1.0 + magnitude))
        elif change_type == ChangeType.DISAPPEARANCE:
            del objects[index]
        modified["change_type"] = change_type
        return modified

    def detect_change(self, original: Dict[str, Any], modified: Dict[str, Any]) -> bool:
        """Whether two scenes differ (object count, colour, position or size)."""
        a, b = original["objects"], modified["objects"]
        if len(a) != len(b):
            return True
        for first, second in zip(a, b):
            if not np.allclose(first["color"], second["color"], atol=TOLERANCE):
                return True
            if not np.allclose(first["position"], second["position"], atol=TOLERANCE):
                return True
            if abs(float(first["size"]) - float(second["size"])) > TOLERANCE:
                return True
        return False

    def render_scene(self, scene: Dict[str, Any]) -> np.ndarray:
        """Feature vector the system sees: one coloured bump per object along the input axis."""
        stimulus = np.zeros(STIMULUS_DIM)
        positions = np.arange(STIMULUS_DIM)
        width = self.display_size[0]
        for obj in scene["objects"]:
            centre = float(np.asarray(obj["position"], dtype=float)[0]) / width * (STIMULUS_DIM - 1)
            spread = max(2.0, float(obj["size"]) / 4.0)
            bump = np.exp(-0.5 * ((positions - centre) / spread) ** 2)
            stimulus += bump * float(np.mean(obj["color"])) * 2.0
            # The vertical position shifts the bump's second harmonic, so position changes in
            # y are visible too
            y = float(np.asarray(obj["position"], dtype=float)[1]) / width
            stimulus += 0.5 * bump * np.sin(2.0 * np.pi * y * positions / STIMULUS_DIM * 8.0)
        return stimulus

    # -- trials -----------------------------------------------------------------------------
    def _generate_trials(self) -> List[ChangeBlindnessTrial]:
        trials: List[ChangeBlindnessTrial] = []
        for magnitude in self.change_magnitudes:
            for _ in range(self.num_trials_per_condition):
                trials.append(
                    ChangeBlindnessTrial(
                        trial_number=len(trials),
                        change_type=self.change_type,
                        change_magnitude=magnitude,
                    )
                )
        return trials

    def run_trial(self, apgi_framework: Any, trial: ChangeBlindnessTrial) -> ChangeBlindnessResult:
        """Flicker the original and modified scene on ``apgi_framework`` until the change is seen."""
        original = self.create_scene()
        modified = self.introduce_change(original, trial.change_type, trial.change_magnitude)
        display_a, display_b = self.render_scene(original), self.render_scene(modified)
        display_steps = max(1, int(round(self.presentation_duration_ms / STEP_MS)))
        blank_steps = max(1, int(round(self.blank_duration_ms / STEP_MS)))
        cycle_ms = 2.0 * (self.presentation_duration_ms + self.blank_duration_ms)

        detected_at: Optional[int] = None
        strength = 0.0
        with scaled_timestep(apgi_framework, STEP_MS):
            for alternation in range(1, self.max_alternations + 1):
                first = present_stimulus(apgi_framework, display_a, display_steps)
                present_stimulus(apgi_framework, blank_stimulus(), blank_steps)
                second = present_stimulus(apgi_framework, display_b, display_steps)
                present_stimulus(apgi_framework, blank_stimulus(), blank_steps)

                margin = DETECTION_MARGIN * max(second.threshold, 1e-9)
                noticed = (second.ignited and not first.ignited) or (
                    second.peak_signal - first.peak_signal > margin
                )
                strength = max(strength, second.peak_signal)
                if noticed:
                    detected_at = alternation
                    break

        detected = detected_at is not None
        result = ChangeBlindnessResult(
            trial_number=trial.trial_number,
            change_type=trial.change_type,
            change_magnitude=trial.change_magnitude,
            change_detected=detected,
            alternations_to_detection=(
                detected_at if detected_at is not None else self.max_alternations
            ),
            time_to_detection=float(
                (detected_at if detected_at is not None else self.max_alternations) * cycle_ms
            ),
            ignition_signal_strength=float(strength),
        )
        self.results.append(result)
        return result

    def run_all_trials(self, apgi_framework: Any) -> Dict[str, Any]:
        """Run every trial and return the analysis."""
        self.results = []
        for trial in self.trials:
            self.run_trial(apgi_framework, trial)
        return self.analyze_results()

    # -- analysis ---------------------------------------------------------------------------
    def analyze_results(self) -> Dict[str, Any]:
        """Detection rates overall and by change magnitude (``error`` entry if no results)."""
        results = self.results
        if not results:
            return {
                "error": "No results to analyze",
                "total_trials": 0,
                "overall_detection_rate": 0.0,
            }

        def summarize(subset: List[ChangeBlindnessResult]) -> Dict[str, float]:
            hits = [r for r in subset if r["change_detected"]]
            return {
                "total_trials": len(subset),
                "detections": len(hits),
                "detection_rate": len(hits) / len(subset),
                "avg_alternations_to_detection": (
                    float(np.mean([r["alternations_to_detection"] for r in hits])) if hits else 0.0
                ),
                "avg_time_to_detection": (
                    float(np.mean([r["time_to_detection"] for r in hits])) if hits else 0.0
                ),
            }

        by_magnitude = {
            magnitude: summarize(subset)
            for magnitude in self.change_magnitudes
            if (subset := [r for r in results if r["change_magnitude"] == magnitude])
        }
        overall = summarize(results)
        return {
            "total_trials": len(results),
            "overall_detection_rate": overall["detection_rate"],
            "overall_blindness_rate": 1.0 - overall["detection_rate"],
            "overall_avg_alternations": overall["avg_alternations_to_detection"],
            "overall_avg_time_ms": overall["avg_time_to_detection"],
            "by_magnitude": by_magnitude,
            "by_change_type": {self.change_type.value: by_magnitude},
            "task_parameters": {
                "change_magnitudes": self.change_magnitudes,
                "num_trials_per_condition": self.num_trials_per_condition,
                "max_alternations": self.max_alternations,
            },
        }

    def reset(self) -> None:
        """Return to the initial state with freshly generated trials."""
        self.results = []
        self.trials = self._generate_trials()

    def save_results(self, filename: str) -> None:
        """Save the trial results to a JSON file."""
        save_json(filename, self.results)

    def print_results(self, analysis: Dict[str, Any]) -> None:
        """Print an analysis to the console."""
        print("\n" + "=" * 50)
        print("CHANGE BLINDNESS TASK RESULTS")
        print("=" * 50)
        print(f"Total Trials: {analysis['total_trials']}")
        print(f"Detection Rate: {analysis['overall_detection_rate']:.1%}")
        print(f"Blindness Rate: {analysis['overall_blindness_rate']:.1%}")
        print(f"Avg Alternations: {analysis['overall_avg_alternations']:.1f}")
        print(f"Avg Time to Detection: {analysis['overall_avg_time_ms']:.0f} ms")
