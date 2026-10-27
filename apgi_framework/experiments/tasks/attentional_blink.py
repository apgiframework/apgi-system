"""Attentional blink task (rapid serial visual presentation) on the APGI system.

A stream of distractors contains two targets, T1 and T2, ``lag`` items apart. The stream is
presented to the APGI system item by item. A target counts as detected when the system ignites
while it is on screen. Because an ignition is followed by a refractory period, T2 presented
shortly after an ignited T1 is missed: the blink (T1 detected, T2 missed).
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

#: Number of system steps used to present one stream item.
STEPS_PER_ITEM = 3


class StimulusType(Enum):
    """Types of stimuli in attentional blink task."""

    TARGET = "target"
    DISTRACTOR = "distractor"
    PROBE = "probe"
    TARGET_1 = "target_1"
    TARGET_2 = "target_2"


@dataclass
class AttentionalBlinkTrial(TaskRecord):
    """One planned RSVP stream."""

    trial_number: int
    t1_position: int
    t2_position: int
    lag: int
    stream: List[np.ndarray]
    stream_types: List[StimulusType]


@dataclass
class AttentionalBlinkResult(TaskRecord):
    """Outcome of one executed stream."""

    trial_number: int
    lag: int
    t1_detected: bool
    t2_detected: bool
    t1_ignition_time: Optional[float]
    t2_ignition_time: Optional[float]
    t1_signal_strength: float
    t2_signal_strength: float
    blink_occurred: bool


class AttentionalBlinkTask:
    """Attentional blink experimental task.

    Parameters
    ----------
    stimulus_duration_ms, item_duration_ms : float, optional
        Time each stream item is shown (the two names are aliases; default 100 ms).
    isi_ms : float
        Inter-stimulus interval (kept for compatibility; items are shown back to back).
    lags : list of int, optional
        T2 - T1 distances in items.
    num_trials_per_lag : int
        Trials per lag.
    stream_length : int
        Items per stream; every lag must be shorter than the stream.
    target_salience : float
        Target amplitude; distractors are much weaker.
    seed : int, optional
        Seed for streams and positions; by default drawn from numpy's global state.
    """

    def __init__(
        self,
        stimulus_duration_ms: Optional[float] = None,
        isi_ms: float = 200,
        lags: Optional[List[int]] = None,
        num_trials_per_lag: int = 10,
        stream_length: int = 10,
        item_duration_ms: Optional[float] = None,
        target_salience: float = 2.0,
        seed: Optional[int] = None,
    ):
        duration = item_duration_ms if item_duration_ms is not None else stimulus_duration_ms
        self.item_duration_ms = float(duration) if duration is not None else 100.0
        self.stimulus_duration_ms = self.item_duration_ms
        self.isi_ms = isi_ms
        self.lags = list(lags) if lags else [1, 2, 3, 4, 5, 6, 7, 8]
        self.num_trials_per_lag = num_trials_per_lag
        self.stream_length = stream_length
        self.target_salience = target_salience
        self._seed = draw_seed(seed)
        self.rng = np.random.default_rng(self._seed)

        self.stimulus_sequence: List[StimulusType] = []
        self.current_trial_idx = 0
        self.results: List[AttentionalBlinkResult] = []
        self.trials: List[AttentionalBlinkTrial] = self._generate_trials()

    # -- stimuli ----------------------------------------------------------------------------
    def _generate_target(self, target_num: int = 1) -> np.ndarray:
        """T1 drives the first half of the input, T2 the second half (different patterns)."""
        stimulus = np.zeros(STIMULUS_DIM)
        half = STIMULUS_DIM // 2
        start = 0 if target_num == 1 else half
        phase = np.linspace(0.0, 2.0 * np.pi * target_num, half)
        stimulus[start : start + half] = self.target_salience * (1.0 + np.sin(phase)) / 2.0
        stimulus[start] += self.target_salience
        return stimulus

    def _generate_distractor(self) -> np.ndarray:
        """Weak noise, well below target amplitude."""
        return np.asarray(0.1 * self.target_salience * self.rng.standard_normal(STIMULUS_DIM))

    def generate_sequence(self, num_stimuli: int = 10) -> List[StimulusType]:
        """A fixed illustrative sequence: target at position 2, probe at position 5."""
        sequence = [StimulusType.DISTRACTOR] * num_stimuli
        if num_stimuli > 2:
            sequence[2] = StimulusType.TARGET
        if num_stimuli > 5:
            sequence[5] = StimulusType.PROBE
        self.stimulus_sequence = sequence
        return sequence

    def calculate_blink_probability(self, lag: int) -> float:
        """Textbook blink probability by lag (reference curve; not used to score trials)."""
        if lag < 2:
            return 0.9
        if lag < 4:
            return 0.7
        return 0.1

    # -- trials -----------------------------------------------------------------------------
    def _generate_trials(self) -> List[AttentionalBlinkTrial]:
        trials: List[AttentionalBlinkTrial] = []
        for lag in self.lags:
            last_t1 = self.stream_length - lag - 1
            if last_t1 < 0:
                raise ValueError(
                    f"lag {lag} does not fit in a stream of {self.stream_length} items"
                )
            for _ in range(self.num_trials_per_lag):
                # Leave a couple of distractors before T1 when the stream allows it
                t1 = int(self.rng.integers(min(2, last_t1), last_t1 + 1))
                t2 = t1 + lag
                stream = [self._generate_distractor() for _ in range(self.stream_length)]
                types = [StimulusType.DISTRACTOR] * self.stream_length
                stream[t1], types[t1] = self._generate_target(1), StimulusType.TARGET_1
                stream[t2], types[t2] = self._generate_target(2), StimulusType.TARGET_2
                trials.append(
                    AttentionalBlinkTrial(
                        trial_number=len(trials),
                        t1_position=t1,
                        t2_position=t2,
                        lag=lag,
                        stream=stream,
                        stream_types=types,
                    )
                )
        return trials

    def run_trial(
        self, apgi_framework: Any, trial: AttentionalBlinkTrial
    ) -> AttentionalBlinkResult:
        """Present the stream to ``apgi_framework`` (an :class:`APGISystem`) and score T1/T2."""
        step_ms = self.item_duration_ms / STEPS_PER_ITEM
        windows: List[PhaseOutcome] = []
        with scaled_timestep(apgi_framework, step_ms):
            for item in trial.stream:
                windows.append(present_stimulus(apgi_framework, item, STEPS_PER_ITEM))

        def ignition_time(position: int) -> Optional[float]:
            first = windows[position].first_ignition_step
            if first is None:
                return None
            return float(position * self.item_duration_ms + (first + 1) * step_ms)

        t1, t2 = windows[trial.t1_position], windows[trial.t2_position]
        result = AttentionalBlinkResult(
            trial_number=trial.trial_number,
            lag=trial.lag,
            t1_detected=t1.ignited,
            t2_detected=t2.ignited,
            t1_ignition_time=ignition_time(trial.t1_position),
            t2_ignition_time=ignition_time(trial.t2_position),
            t1_signal_strength=t1.peak_signal,
            t2_signal_strength=t2.peak_signal,
            blink_occurred=t1.ignited and not t2.ignited,
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
    def analyze_results(self) -> Dict[str, Any]:
        """T1/T2 accuracy and blink rate overall and by lag (``error`` entry if no results)."""
        results = self.results
        if not results:
            return {"error": "No results to analyze", "total_trials": 0, "blink_probabilities": {}}

        total = len(results)
        lag_analysis: Dict[int, Dict[str, float]] = {}
        for lag in self.lags:
            subset = [r for r in results if r["lag"] == lag]
            if not subset:
                continue
            t1_hits = [r for r in subset if r["t1_detected"]]
            t2_hits = [r for r in subset if r["t2_detected"]]
            blinks = [r for r in subset if r["blink_occurred"]]
            t2_given_t1 = (
                sum(1 for r in t1_hits if r["t2_detected"]) / len(t1_hits) if t1_hits else 0.0
            )
            lag_analysis[lag] = {
                "total_trials": len(subset),
                "t1_accuracy": len(t1_hits) / len(subset),
                "t2_accuracy": len(t2_hits) / len(subset),
                "t2_given_t1_accuracy": t2_given_t1,
                "blink_rate": len(blinks) / len(subset),
                "t1_detections": len(t1_hits),
                "t2_detections": len(t2_hits),
                "blink_occurrences": len(blinks),
            }

        max_blink_lag: Optional[int] = None
        max_blink_rate = 0.0
        for lag, data in lag_analysis.items():
            if data["blink_rate"] > max_blink_rate:
                max_blink_lag, max_blink_rate = lag, data["blink_rate"]

        return {
            "total_trials": total,
            "overall_t1_accuracy": sum(1 for r in results if r["t1_detected"]) / total,
            "overall_t2_accuracy": sum(1 for r in results if r["t2_detected"]) / total,
            "overall_blink_rate": sum(1 for r in results if r["blink_occurred"]) / total,
            "lag_analysis": lag_analysis,
            "max_blink_lag": max_blink_lag,
            "max_blink_rate": max_blink_rate,
            "blink_probabilities": {lag: d["blink_rate"] for lag, d in lag_analysis.items()},
            "task_parameters": {
                "lags": self.lags,
                "num_trials_per_lag": self.num_trials_per_lag,
                "stream_length": self.stream_length,
                "item_duration_ms": self.item_duration_ms,
                "target_salience": self.target_salience,
            },
        }

    def reset(self) -> None:
        """Return to the initial state with freshly generated streams."""
        self.results = []
        self.current_trial_idx = 0
        self.trials = self._generate_trials()

    def save_results(self, filename: str) -> None:
        """Save the trial results to a JSON file."""
        save_json(filename, self.results)

    def print_results(self, analysis: Dict[str, Any]) -> None:
        """Print an analysis to the console."""
        print("\n" + "=" * 50)
        print("ATTENTIONAL BLINK TASK RESULTS")
        print("=" * 50)
        print(f"Total Trials: {analysis['total_trials']}")
        print(f"T1 Accuracy: {analysis['overall_t1_accuracy']:.1%}")
        print(f"T2 Accuracy: {analysis['overall_t2_accuracy']:.1%}")
        print(f"Blink Rate: {analysis['overall_blink_rate']:.1%}")
        print("\nBy Lag:")
        for lag, data in analysis["lag_analysis"].items():
            print(
                f"  Lag {lag}: T2|T1={data['t2_given_t1_accuracy']:.1%}, "
                f"Blink={data['blink_rate']:.1%}"
            )
