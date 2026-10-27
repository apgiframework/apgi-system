"""Visual masking paradigm on the APGI system.

A brief target is followed by a mask at a variable stimulus onset asynchrony (SOA). The
combined timeline is presented to the APGI system one 10 ms step at a time. The target counts
as detected when the system ignites; an ignition that the mask arrives too early to let
consolidate is suppressed with a probability that falls off with SOA (backward masking), so
detection recovers as the SOA grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ._records import (
    STIMULUS_DIM,
    PhaseOutcome,
    TaskRecord,
    draw_seed,
    present_stimulus,
    save_json,
    scaled_timestep,
)

#: Task time represented by one system step (ms).
STEP_MS = 10.0
#: Time constant (ms) of the masking effect's decay with SOA.
MASKING_TAU_MS = 60.0
#: Extra time after the mask is gone in which an ignition can still occur (ms).
SETTLE_MS = 100.0


class MaskType(Enum):
    """Types of masking in masking paradigm."""

    FORWARD = "forward"
    BACKWARD = "backward"
    META_CONTRAST = "meta_contrast"
    PATTERN = "pattern"


@dataclass
class MaskingTrial(TaskRecord):
    """One planned trial."""

    trial_number: int
    soa_ms: float
    target_duration_ms: float
    mask_duration_ms: float
    mask_type: MaskType


@dataclass
class MaskingResult(TaskRecord):
    """Outcome of one executed trial."""

    trial_number: int
    soa_ms: float
    mask_type: MaskType
    target_detected: bool
    detection_time: Optional[float]
    ignition_strength: float
    ignition_count: int
    mask_suppression_occurred: bool
    masking_effect: float = 0.0


class MaskingParadigmTask:
    """Masking paradigm experimental task.

    Parameters
    ----------
    target_duration_ms, mask_duration_ms : float
        Durations of the target and of the mask.
    soas : list of float, optional
        Target-to-mask onset asynchronies; SOA 0 presents both at once.
    num_trials_per_condition : int
        Trials per SOA.
    target_strength, mask_strength : float
        Stimulus amplitudes; a stronger mask suppresses more.
    mask_type : MaskType
        Mask kind reported in the results.
    seed : int, optional
        Seed for stimulus noise and suppression draws; by default drawn from numpy's global
        state.
    """

    def __init__(
        self,
        target_duration_ms: float = 20,
        mask_duration_ms: float = 40,
        soas: Optional[List[int]] = None,
        num_trials_per_condition: int = 10,
        target_strength: float = 2.0,
        mask_strength: float = 4.0,
        mask_type: MaskType = MaskType.PATTERN,
        seed: Optional[int] = None,
    ):
        self.target_duration_ms = target_duration_ms
        self.mask_duration_ms = mask_duration_ms
        self.soas = list(soas) if soas else [20, 40, 60, 80, 100, 120, 140, 160]
        self.num_trials_per_condition = num_trials_per_condition
        self.target_strength = target_strength
        self.mask_strength = mask_strength
        self.mask_type = mask_type
        self._seed = draw_seed(seed)
        self.rng = np.random.default_rng(self._seed)

        self.current_trial_idx = 0
        self.results: List[MaskingResult] = []
        self.trials: List[MaskingTrial] = self._generate_trials()

    # -- stimuli ----------------------------------------------------------------------------
    def _generate_target(self) -> np.ndarray:
        """A localised Gaussian bump (plus faint noise) of amplitude ``target_strength``."""
        centre = STIMULUS_DIM // 2
        positions = np.arange(STIMULUS_DIM)
        bump = np.exp(-0.5 * ((positions - centre) / (STIMULUS_DIM / 16.0)) ** 2)
        return np.asarray(
            self.target_strength * bump + 0.05 * self.rng.standard_normal(STIMULUS_DIM)
        )

    def _generate_mask(self) -> np.ndarray:
        """A dense, high-energy noise pattern covering the whole input."""
        return np.asarray(0.5 * self.mask_strength * self.rng.standard_normal(STIMULUS_DIM))

    # Backward-compatible names
    def create_target_stimulus(self) -> np.ndarray:
        return self._generate_target()

    def create_mask_stimulus(self) -> np.ndarray:
        return self._generate_mask()

    def calculate_masking_effect(self, soa_ms: float) -> float:
        """Probability that an ignition is suppressed by a mask at ``soa_ms``.

        Relative mask strength times an exponential decay with SOA (time constant 60 ms).
        """
        relative = self.mask_strength / (self.mask_strength + self.target_strength)
        return float(relative * np.exp(-max(soa_ms, 0.0) / MASKING_TAU_MS))

    # -- trials -----------------------------------------------------------------------------
    def _generate_trials(self) -> List[MaskingTrial]:
        trials: List[MaskingTrial] = []
        for soa in self.soas:
            for _ in range(self.num_trials_per_condition):
                trials.append(
                    MaskingTrial(
                        trial_number=len(trials),
                        soa_ms=soa,
                        target_duration_ms=self.target_duration_ms,
                        mask_duration_ms=self.mask_duration_ms,
                        mask_type=self.mask_type,
                    )
                )
        return trials

    def _present_timeline(self, apgi_framework: Any, trial: MaskingTrial) -> PhaseOutcome:
        """Present target and mask on their schedule, one system step per ``STEP_MS``."""
        target, mask = self._generate_target(), self._generate_mask()
        mask_onset = trial.soa_ms
        mask_offset = mask_onset + trial.mask_duration_ms
        total_ms = max(trial.target_duration_ms, mask_offset) + SETTLE_MS

        timeline = PhaseOutcome()
        with scaled_timestep(apgi_framework, STEP_MS):
            for step in range(max(1, int(round(total_ms / STEP_MS)))):
                t = step * STEP_MS
                stimulus = np.zeros(STIMULUS_DIM)
                if t < trial.target_duration_ms:
                    stimulus = stimulus + target
                if mask_onset <= t < mask_offset:
                    stimulus = stimulus + mask
                timeline.extend(present_stimulus(apgi_framework, stimulus, 1))
        return timeline

    def run_trial(self, apgi_framework: Any, trial: MaskingTrial) -> MaskingResult:
        """Run one trial on ``apgi_framework`` (an :class:`APGISystem`) and record the result."""
        timeline = self._present_timeline(apgi_framework, trial)
        first = timeline.first_ignition_step
        effect = self.calculate_masking_effect(trial.soa_ms)

        suppressed = False
        detected = False
        detection_time: Optional[float] = None
        if first is not None:
            ignition_ms = (first + 1) * STEP_MS
            # An ignition before the mask arrives has consolidated; later ones can be masked
            mask_arrived_first = ignition_ms > trial.soa_ms
            suppressed = bool(mask_arrived_first and self.rng.random() < effect)
            detected = not suppressed
            if detected:
                detection_time = float(ignition_ms)

        result = MaskingResult(
            trial_number=trial.trial_number,
            soa_ms=trial.soa_ms,
            mask_type=trial.mask_type,
            target_detected=detected,
            detection_time=detection_time,
            ignition_strength=float(timeline.peak_signal),
            ignition_count=int(timeline.ignition_count),
            mask_suppression_occurred=suppressed,
            masking_effect=effect,
        )
        self.results.append(result)
        self.current_trial_idx = len(self.results)
        return result

    def run_all_trials(self, apgi_framework: Any) -> Dict[str, Any]:
        """Run every trial and return the analysis."""
        self.results = []
        for trial in self.trials:
            self.run_trial(apgi_framework, trial)
        return self.analyze_results()

    # -- analysis ---------------------------------------------------------------------------
    def analyze_results(self, results: Optional[List[MaskingResult]] = None) -> Dict[str, Any]:
        """Detection and suppression by SOA (``{"error": ..., "total_trials": 0}`` if none)."""
        results = self.results if results is None else results
        if not results:
            return {"error": "No results to analyze", "total_trials": 0}

        total = len(results)
        detected = sum(1 for r in results if r["target_detected"])
        suppressed = sum(1 for r in results if r["mask_suppression_occurred"])

        by_soa: Dict[Any, Dict[str, float]] = {}
        for soa in self.soas:
            subset = [r for r in results if r["soa_ms"] == soa]
            if not subset:
                continue
            hits = sum(1 for r in subset if r["target_detected"])
            by_soa[soa] = {
                "total_trials": len(subset),
                "detections": hits,
                "detection_rate": hits / len(subset),
                "suppression_rate": sum(1 for r in subset if r["mask_suppression_occurred"])
                / len(subset),
                "avg_ignition_strength": float(np.mean([r["ignition_strength"] for r in subset])),
            }

        curve = [{"soa_ms": soa, "detection_rate": by_soa[soa]["detection_rate"]} for soa in by_soa]
        masking_effect = (
            float(curve[-1]["detection_rate"] - curve[0]["detection_rate"])
            if len(curve) >= 2
            else 0.0
        )
        return {
            "total_trials": total,
            "overall_detection_rate": detected / total,
            "overall_suppression_rate": suppressed / total,
            "masking_effect": masking_effect,
            "by_soa": by_soa,
            "masking_curve": curve,
            "task_parameters": {
                "target_duration_ms": self.target_duration_ms,
                "mask_duration_ms": self.mask_duration_ms,
                "soas": self.soas,
                "num_trials_per_condition": self.num_trials_per_condition,
                "target_strength": self.target_strength,
                "mask_strength": self.mask_strength,
            },
        }

    def save_results(self, results: Optional[List[Any]], filepath: str) -> None:
        """Save ``results`` (default: the task's own) to a JSON file."""
        save_json(filepath, self.results if results is None else results)

    def reset(self) -> None:
        """Return to the initial state with freshly generated trials."""
        self.results = []
        self.current_trial_idx = 0
        self.trials = self._generate_trials()
