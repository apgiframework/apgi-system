"""Binocular rivalry task.

Two incompatible images, one per eye, compete for perceptual dominance. Dominance is simulated
with a standard mutual-inhibition model: each percept's population is driven by its stimulus
strength, inhibits the other, and slowly adapts, so dominance alternates with noisy durations.
Dominance is sampled every 100 ms and expressed as the share of pattern A in [0, 1]
(above 0.5 means A is dominant).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._records import TaskRecord, draw_seed, save_json

#: Interval at which dominance is sampled (ms).
SAMPLE_MS = 100.0
#: Integration step of the rivalry model (ms).
MODEL_DT_MS = 10.0

# Mutual-inhibition model parameters (ms time constants)
TAU_ACTIVITY_MS = 20.0
TAU_ADAPTATION_MS = 4000.0
INHIBITION = 2.0
ADAPTATION_GAIN = 1.2
NOISE = 0.03


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-6.0 * (x - 0.5))))


@dataclass
class BinocularRivalryTrial(TaskRecord):
    """One planned rivalry trial."""

    trial_number: int
    pattern_a_strength: float
    pattern_b_strength: float


@dataclass
class BinocularRivalryResult(TaskRecord):
    """Outcome of one executed trial."""

    trial_number: int
    pattern_a_strength: float
    pattern_b_strength: float
    total_duration_ms: float
    dominance_periods: List[float]
    num_alternations: int
    pattern_a_total_duration: float
    pattern_b_total_duration: float
    pattern_a_dominance_ratio: float
    average_dominance_duration: float
    alternation_rate: float


class BinocularRivalryTask:
    """Binocular rivalry experimental task.

    Parameters
    ----------
    image_size : tuple
        Size of the rivalrous images.
    trial_duration_ms : float
        Length of each trial.
    num_trials : int
        Number of trials; they cycle through ``strength_ratios``.
    strength_ratios : list of (float, float), optional
        Relative strengths of the two patterns; 1.0 is the baseline strength of 0.5.
    sampling_interval_ms : float
        Kept for compatibility; dominance is sampled every 100 ms.
    seed : int, optional
        Seed for noise and images; by default drawn from numpy's global state.
    """

    def __init__(
        self,
        image_size: tuple = (256, 256),
        trial_duration_ms: float = 1000.0,
        num_trials: int = 10,
        strength_ratios: Optional[List[Tuple[float, float]]] = None,
        sampling_interval_ms: float = 100.0,
        seed: Optional[int] = None,
    ):
        self.image_size = image_size
        self.trial_duration_ms = trial_duration_ms
        self.num_trials = num_trials
        self.strength_ratios = strength_ratios or [(1.0, 1.0)]
        self.sampling_interval_ms = sampling_interval_ms
        self._seed = draw_seed(seed)
        self.rng = np.random.default_rng(self._seed)

        self.dominance_history: List[float] = []
        self.results: List[BinocularRivalryResult] = []
        self.trials: List[BinocularRivalryTrial] = self._generate_trials()

    # -- stimuli and dominance --------------------------------------------------------------
    def create_stimulus_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """Orthogonal gratings (left: 45 degrees, right: 135 degrees), pixel values in [0, 1]."""
        height, width = self.image_size
        y, x = np.mgrid[0:height, 0:width]
        frequency = 8.0 / max(width, 1)
        left = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * (x + y))
        right = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * (x - y))
        return left, right

    def simulate_dominance_switching(
        self, duration_ms: int = 10000, strengths: Tuple[float, float] = (0.5, 0.5)
    ) -> List[float]:
        """Simulate dominance of pattern A over ``duration_ms``; replaces the stored history.

        Returns one value per 100 ms: the share ``rA / (rA + rB)`` of pattern A's activity.
        """
        steps = int(duration_ms // SAMPLE_MS)
        substeps = int(SAMPLE_MS // MODEL_DT_MS)
        drive = np.asarray(strengths, dtype=float)
        rate = np.array([0.6, 0.4])
        adaptation = np.zeros(2)

        dominance: List[float] = []
        for _ in range(steps):
            for _ in range(substeps):
                noise = NOISE * self.rng.standard_normal(2)
                net = drive - INHIBITION * rate[::-1] - ADAPTATION_GAIN * adaptation + noise
                rate = rate + MODEL_DT_MS / TAU_ACTIVITY_MS * (-rate + _sigmoid(net))
                adaptation = adaptation + MODEL_DT_MS / TAU_ADAPTATION_MS * (-adaptation + rate)
            share = rate[0] / (rate[0] + rate[1] + 1e-12)
            dominance.append(float(np.clip(share, 0.0, 1.0)))
        self.dominance_history = dominance
        return list(dominance)

    def get_switch_rate(self) -> float:
        """Dominance switches per second in the stored history (0 if there is none)."""
        if not self.dominance_history:
            return 0.0
        switches = sum(
            1
            for i in range(1, len(self.dominance_history))
            if (self.dominance_history[i] - 0.5) * (self.dominance_history[i - 1] - 0.5) < 0
        )
        return switches / (len(self.dominance_history) * SAMPLE_MS / 1000.0)

    # -- trials -----------------------------------------------------------------------------
    def _generate_trials(self) -> List[BinocularRivalryTrial]:
        trials: List[BinocularRivalryTrial] = []
        for i in range(self.num_trials):
            ratio_a, ratio_b = self.strength_ratios[i % len(self.strength_ratios)]
            trials.append(
                BinocularRivalryTrial(
                    trial_number=i,
                    pattern_a_strength=0.5 * ratio_a,
                    pattern_b_strength=0.5 * ratio_b,
                )
            )
        return trials

    def _calculate_dominance_periods(self, dominance: List[float]) -> List[float]:
        """Durations (ms) of consecutive runs of the same dominant pattern."""
        if not dominance:
            return []
        periods: List[float] = []
        current_state = dominance[0] > 0.5
        current = 0.0
        for value in dominance:
            state = value > 0.5
            if state != current_state:
                periods.append(current)
                current, current_state = 0.0, state
            current += SAMPLE_MS
        periods.append(current)
        return periods

    def run_trial(
        self, apgi_framework: Any, trial: BinocularRivalryTrial
    ) -> BinocularRivalryResult:
        """Simulate one rivalry trial (the rivalry model is independent of the APGI system)."""
        dominance = self.simulate_dominance_switching(
            int(self.trial_duration_ms), (trial.pattern_a_strength, trial.pattern_b_strength)
        )
        periods = self._calculate_dominance_periods(dominance)

        a_ms = SAMPLE_MS * sum(1 for d in dominance if d > 0.5)
        b_ms = SAMPLE_MS * sum(1 for d in dominance if d <= 0.5)
        total_ms = a_ms + b_ms
        alternations = max(len(periods) - 1, 0)
        result = BinocularRivalryResult(
            trial_number=trial.trial_number,
            pattern_a_strength=trial.pattern_a_strength,
            pattern_b_strength=trial.pattern_b_strength,
            total_duration_ms=total_ms,
            dominance_periods=periods,
            num_alternations=alternations,
            pattern_a_total_duration=a_ms,
            pattern_b_total_duration=b_ms,
            pattern_a_dominance_ratio=a_ms / total_ms if total_ms > 0 else 0.5,
            average_dominance_duration=float(np.mean(periods)) if periods else 0.0,
            alternation_rate=alternations / (total_ms / 1000.0) if total_ms > 0 else 0.0,
        )
        self.results.append(result)
        return result

    def run_all_trials(self, apgi_framework: Any) -> Dict[str, Any]:
        """Run every trial and return the analysis."""
        self.results = []
        for trial in self.trials:
            self.run_trial(apgi_framework, trial)
        analysis = self.analyze_results()
        analysis["task_parameters"] = {
            "trial_duration_ms": self.trial_duration_ms,
            "num_trials": self.num_trials,
            "image_size": self.image_size,
        }
        return analysis

    # -- analysis ---------------------------------------------------------------------------
    def analyze_results(self) -> Dict[str, Any]:
        """Averages overall and by strength ratio."""
        results = self.results
        if not results:
            return {
                "total_trials": 0,
                "avg_alternation_rate": 0.0,
                "avg_dominance_duration_ms": 0.0,
                "avg_pattern_a_dominance_ratio": 0.0,
                "total_alternations": 0,
                "by_strength_ratio": {},
            }

        by_ratio: Dict[str, Dict[str, float]] = {}
        for r in results:
            key = f"{r['pattern_a_strength']:.2f}_{r['pattern_b_strength']:.2f}"
            entry = by_ratio.setdefault(
                key, {"avg_dominance_duration_ms": 0.0, "avg_alternation_rate": 0.0, "count": 0}
            )
            entry["avg_dominance_duration_ms"] += r["average_dominance_duration"]
            entry["avg_alternation_rate"] += r["alternation_rate"]
            entry["count"] += 1
        for entry in by_ratio.values():
            entry["avg_dominance_duration_ms"] /= entry["count"]
            entry["avg_alternation_rate"] /= entry["count"]

        return {
            "total_trials": len(results),
            "avg_alternation_rate": float(np.mean([r["alternation_rate"] for r in results])),
            "avg_dominance_duration_ms": float(
                np.mean([r["average_dominance_duration"] for r in results])
            ),
            "avg_pattern_a_dominance_ratio": float(
                np.mean([r["pattern_a_dominance_ratio"] for r in results])
            ),
            "total_alternations": int(sum(r["num_alternations"] for r in results)),
            "by_strength_ratio": by_ratio,
        }

    def reset(self) -> None:
        """Return to the initial state with freshly generated trials."""
        self.results = []
        self.dominance_history = []
        self.trials = self._generate_trials()

    def save_results(self, filename: str) -> None:
        """Save the trial results to a JSON file."""
        save_json(filename, self.results)

    def print_results(self, analysis: Dict[str, Any]) -> None:
        """Print an analysis to the console."""
        print("\n" + "=" * 50)
        print("BINOCULAR RIVALRY TASK RESULTS")
        print("=" * 50)
        print(f"Total Trials: {analysis['total_trials']}")
        print(f"Avg Alternation Rate: {analysis['avg_alternation_rate']:.2f} per second")
        print(f"Avg Dominance Duration: {analysis['avg_dominance_duration_ms']:.0f} ms")
