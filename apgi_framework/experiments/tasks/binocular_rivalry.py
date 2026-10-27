"""Stub module for binocular rivalry task.

This module provides type stubs for binocular rivalry experimental task
that is being migrated from the old apgi_simulation structure.
"""

from typing import Any

import numpy as np


class BinocularRivalryTask:
    """Binocular rivalry experimental task."""

    def __init__(
        self,
        image_size: tuple = (256, 256),
        trial_duration_ms: float = 1000.0,
        num_trials: int = 10,
        strength_ratios: list[tuple[float, float]] | None = None,
        sampling_interval_ms: float = 100.0,
    ):
        """Initialize binocular rivalry task.

        Parameters
        ----------
        image_size : tuple, optional
            Size of images for rivalry, by default (256, 256)
        trial_duration_ms : float, optional
            Duration of each trial in milliseconds, by default 1000.0
        num_trials : int, optional
            Number of trials, by default 10
        """
        self.image_size = image_size
        self.trial_duration_ms = trial_duration_ms
        self.num_trials = num_trials
        self.strength_ratios = strength_ratios or [(1.0, 1.0)]
        self.sampling_interval_ms = sampling_interval_ms
        self.dominance_history: list[float] = []
        self.trials = self._generate_trials()

    def create_stimulus_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """Create a pair of rivalrous stimuli.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pair of images for left and right eye
        """
        left_image = np.random.rand(*self.image_size)
        right_image = np.random.rand(*self.image_size)
        return left_image, right_image

    def simulate_dominance_switching(self, duration_ms: int = 10000) -> list[float]:
        """Simulate dominance switching over time.

        Parameters
        ----------
        duration_ms : int, optional
            Duration of simulation in milliseconds, by default 10000

        Returns
        -------
        list[float]
            Dominance values over time
        """
        # Simplified dominance simulation
        num_steps = duration_ms // 100
        dominance = np.zeros(num_steps)
        for i in range(num_steps):
            dominance[i] = np.sin(i * 0.1) * 0.5 + 0.5
        self.dominance_history = dominance.tolist()
        return self.dominance_history

    def get_switch_rate(self) -> float:
        """Calculate the rate of dominance switches.

        Returns
        -------
        float
            Switch rate in Hz
        """
        if not self.dominance_history:
            return 0.0
        # Count zero crossings
        switches = sum(
            1
            for i in range(1, len(self.dominance_history))
            if (self.dominance_history[i] - 0.5) * (self.dominance_history[i - 1] - 0.5) < 0
        )
        return switches / (len(self.dominance_history) / 10.0)  # Assuming 100ms steps

    def _generate_trials(self) -> list[dict[str, Any]]:
        """Generate trial configurations.

        Returns
        -------
        list[dict[str, Any]]
            List of trial configurations
        """
        trials = []
        for i in range(self.num_trials):
            trial = {
                "trial_number": i,
                "pattern_a_strength": 0.5 + np.random.uniform(-0.2, 0.2),
                "pattern_b_strength": 0.5 + np.random.uniform(-0.2, 0.2),
            }
            trials.append(trial)
        return trials

    def run_trial(self, apgi_framework: Any, trial: dict[str, Any]) -> dict[str, Any]:
        """Run a single trial.

        Parameters
        ----------
        apgi_framework : Any
            APGI framework instance
        trial : dict[str, Any]
            Trial configuration

        Returns
        -------
        dict[str, Any]
            Trial results
        """
        # Simulate dominance switching
        dominance = self.simulate_dominance_switching(int(self.trial_duration_ms))

        # Calculate dominance periods
        dominance_periods = self._calculate_dominance_periods(dominance)

        pattern_a_duration = sum(d for d in dominance if d > 0.5) * 100
        pattern_b_duration = sum(d for d in dominance if d <= 0.5) * 100
        total_duration = pattern_a_duration + pattern_b_duration

        pattern_a_ratio = pattern_a_duration / total_duration if total_duration > 0 else 0.5
        avg_dominance_duration = (
            total_duration / (len(dominance_periods) + 1) if dominance_periods else 0
        )
        alternation_rate = self.get_switch_rate()

        return {
            "trial_number": trial["trial_number"],
            "pattern_a_strength": trial["pattern_a_strength"],
            "pattern_b_strength": trial["pattern_b_strength"],
            "total_duration_ms": total_duration,
            "dominance_periods": dominance_periods,
            "num_alternations": len(dominance_periods),
            "pattern_a_total_duration": pattern_a_duration,
            "pattern_b_total_duration": pattern_b_duration,
            "pattern_a_dominance_ratio": pattern_a_ratio,
            "average_dominance_duration": avg_dominance_duration,
            "alternation_rate": alternation_rate,
        }

    def _calculate_dominance_periods(self, dominance: list[float]) -> list[float]:
        """Calculate dominance periods from dominance history.

        Parameters
        ----------
        dominance : list[float]
            Dominance values over time

        Returns
        -------
        list[float]
            List of dominance period durations
        """
        periods: list[float] = []
        current_period: float = 0.0
        current_state = dominance[0] > 0.5 if dominance else True

        for d in dominance:
            state = d > 0.5
            if state != current_state:
                periods.append(current_period)
                current_period = 0.0
                current_state = state
            current_period += 100.0  # 100ms per step

        if current_period > 0:
            periods.append(current_period)

        return periods

    def run_all_trials(self, apgi_framework: Any) -> dict[str, Any]:
        """Run all trials and return analysis.

        Parameters
        ----------
        apgi_framework : Any
            APGI framework instance

        Returns
        -------
        dict[str, Any]
            Analysis results
        """
        self.results = []
        for trial in self.trials:
            result = self.run_trial(apgi_framework, trial)
            self.results.append(result)

        # Calculate overall statistics
        avg_alternation_rate = sum(r["alternation_rate"] for r in self.results) / len(self.results)
        avg_pattern_a_ratio = sum(r["pattern_a_dominance_ratio"] for r in self.results) / len(
            self.results
        )
        avg_dominance_duration = sum(r["average_dominance_duration"] for r in self.results) / len(
            self.results
        )

        return {
            "total_trials": len(self.results),
            "avg_alternation_rate": avg_alternation_rate,
            "avg_pattern_a_dominance_ratio": avg_pattern_a_ratio,
            "avg_dominance_duration_ms": avg_dominance_duration,
            "task_parameters": {
                "trial_duration_ms": self.trial_duration_ms,
                "num_trials": self.num_trials,
                "image_size": self.image_size,
            },
        }

    def analyze_results(self) -> dict[str, Any]:
        """Analyze trial results.

        Returns
        -------
        dict[str, Any]
            Analysis results
        """
        if not self.results:
            return {
                "total_trials": 0,
                "avg_alternation_rate": 0.0,
                "avg_dominance_duration_ms": 0.0,
                "avg_pattern_a_dominance_ratio": 0.0,
                "total_alternations": 0,
                "by_strength_ratio": {},
            }

        avg_alternation_rate = sum(r["alternation_rate"] for r in self.results) / len(self.results)
        avg_pattern_a_ratio = sum(r["pattern_a_dominance_ratio"] for r in self.results) / len(
            self.results
        )
        avg_dominance_duration = sum(r["average_dominance_duration"] for r in self.results) / len(
            self.results
        )
        total_alternations = sum(r["num_alternations"] for r in self.results)

        # Group by strength ratio
        by_strength_ratio = {}
        for r in self.results:
            ratio_key = f"{r['pattern_a_strength']:.2f}_{r['pattern_b_strength']:.2f}"
            if ratio_key not in by_strength_ratio:
                by_strength_ratio[ratio_key] = {
                    "avg_dominance_duration_ms": 0.0,
                    "avg_alternation_rate": 0.0,
                    "count": 0,
                }
            by_strength_ratio[ratio_key]["avg_dominance_duration_ms"] += r[
                "average_dominance_duration"
            ]
            by_strength_ratio[ratio_key]["avg_alternation_rate"] += r["alternation_rate"]
            by_strength_ratio[ratio_key]["count"] += 1

        # Calculate averages
        for key in by_strength_ratio:
            count = by_strength_ratio[key]["count"]
            by_strength_ratio[key]["avg_dominance_duration_ms"] /= count
            by_strength_ratio[key]["avg_alternation_rate"] /= count

        return {
            "total_trials": len(self.results),
            "avg_alternation_rate": avg_alternation_rate,
            "avg_dominance_duration_ms": avg_dominance_duration,
            "avg_pattern_a_dominance_ratio": avg_pattern_a_ratio,
            "total_alternations": total_alternations,
            "by_strength_ratio": by_strength_ratio,
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

    def print_results(self, analysis: dict[str, Any]) -> None:
        """Print analysis results to console.

        Parameters
        ----------
        analysis : dict[str, Any]
            Analysis results to print
        """
        print("\n" + "=" * 50)
        print("BINOCULAR RIVALRY TASK RESULTS")
        print("=" * 50)
        print(f"Total Trials: {analysis['total_trials']}")
        print(f"Avg Alternation Rate: {analysis['avg_alternation_rate']:.2f} per second")
        print(f"Avg Dominance Duration: {analysis['avg_dominance_duration_ms']:.0f} ms")
