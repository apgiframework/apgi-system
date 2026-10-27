"""Stub module for attentional blink task.

This module provides type stubs for attentional blink experimental task
that is being migrated from the old apgi_simulation structure.
"""

from enum import Enum
from typing import Any, Dict, List


class StimulusType(Enum):
    """Types of stimuli in attentional blink task."""

    TARGET = "target"
    DISTRACTOR = "distractor"
    PROBE = "probe"


class AttentionalBlinkTask:
    """Attentional blink experimental task."""

    def __init__(
        self,
        stimulus_duration_ms: int = 100,
        isi_ms: int = 200,
        lags: list[int] | None = None,
        num_trials_per_lag: int = 10,
        stream_length: int = 10,
    ):
        """Initialize attentional blink task.

        Parameters
        ----------
        stimulus_duration_ms : int, optional
            Duration of each stimulus in milliseconds, by default 100
        isi_ms : int, optional
            Inter-stimulus interval in milliseconds, by default 200
        lags : list[int], optional
            List of lags to test, by default None
        num_trials_per_lag : int, optional
            Number of trials per lag, by default 10
        stream_length : int, optional
            Length of stimulus stream, by default 10
        """
        self.stimulus_duration_ms = stimulus_duration_ms
        self.isi_ms = isi_ms
        self.lags = lags or [1, 2, 3, 4, 5, 6, 7, 8]
        self.num_trials_per_lag = num_trials_per_lag
        self.stream_length = stream_length
        self.stimulus_sequence: List[StimulusType] = []
        self.trials = self._generate_trials()
        self.results: List[Dict[str, Any]] = []
        self.current_trial_idx = 0

    def generate_sequence(self, num_stimuli: int = 10) -> List[StimulusType]:
        """Generate a stimulus sequence for the task.

        Parameters
        ----------
        num_stimuli : int, optional
            Number of stimuli in the sequence, by default 10

        Returns
        -------
        List[StimulusType]
            Generated stimulus sequence
        """
        sequence = [StimulusType.DISTRACTOR] * num_stimuli
        # Place target at position 2
        if num_stimuli > 2:
            sequence[2] = StimulusType.TARGET
        # Place probe at position 5
        if num_stimuli > 5:
            sequence[5] = StimulusType.PROBE
        self.stimulus_sequence = sequence
        return sequence

    def calculate_blink_probability(self, lag: int) -> float:
        """Calculate blink probability given lag between target and probe.

        Parameters
        ----------
        lag : int
            Lag between target and probe in stimuli

        Returns
        -------
        float
            Blink probability
        """
        # Simplified blink probability curve
        if lag < 2:
            return 0.9
        elif lag < 4:
            return 0.7
        else:
            return 0.1

    def run_trial(self, apgi_framework: Any, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single trial of the attentional blink task.

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
        import random

        lag = trial.get("lag", 3)
        blink_prob = self.calculate_blink_probability(lag)
        t1_detected = random.random() > 0.1
        t2_detected = random.random() > blink_prob
        blink_occurred = t1_detected and not t2_detected

        return {
            "trial_number": trial["trial_number"],
            "lag": lag,
            "t1_detected": t1_detected,
            "t2_detected": t2_detected,
            "blink_occurred": blink_occurred,
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
        for lag in self.lags:
            for _ in range(self.num_trials_per_lag):
                trial = {
                    "trial_number": trial_num,
                    "lag": lag,
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

        # Calculate analysis by lag
        lag_analysis = {}
        for lag in self.lags:
            lag_results = [r for r in self.results if r["lag"] == lag]
            if lag_results:
                t1_detections = sum(1 for r in lag_results if r["t1_detected"])
                t2_detections = sum(1 for r in lag_results if r["t2_detected"])
                blink_occurrences = sum(1 for r in lag_results if r["blink_occurred"])
                t2_given_t1 = [r["t2_detected"] for r in lag_results if r["t1_detected"]]
                t2_given_t1_accuracy = sum(t2_given_t1) / len(t2_given_t1) if t2_given_t1 else 0.0

                lag_analysis[lag] = {
                    "total_trials": len(lag_results),
                    "t1_accuracy": t1_detections / len(lag_results),
                    "t2_accuracy": t2_detections / len(lag_results),
                    "t2_given_t1_accuracy": t2_given_t1_accuracy,
                    "blink_rate": blink_occurrences / len(lag_results),
                    "t1_detections": t1_detections,
                    "t2_detections": t2_detections,
                    "blink_occurrences": blink_occurrences,
                }

        # Calculate overall statistics
        overall_t1_accuracy = sum(1 for r in self.results if r["t1_detected"]) / len(self.results)
        overall_t2_accuracy = sum(1 for r in self.results if r["t2_detected"]) / len(self.results)
        overall_blink_rate = sum(1 for r in self.results if r["blink_occurred"]) / len(self.results)

        # Find max blink lag
        max_blink_lag = None
        max_blink_rate = 0.0
        for lag, data in lag_analysis.items():
            if data["blink_rate"] > max_blink_rate:
                max_blink_rate = data["blink_rate"]
                max_blink_lag = lag

        return {
            "total_trials": len(self.results),
            "overall_t1_accuracy": overall_t1_accuracy,
            "overall_t2_accuracy": overall_t2_accuracy,
            "overall_blink_rate": overall_blink_rate,
            "lag_analysis": lag_analysis,
            "max_blink_lag": max_blink_lag,
            "max_blink_rate": max_blink_rate,
            "task_parameters": {
                "lags": self.lags,
                "num_trials_per_lag": self.num_trials_per_lag,
                "stream_length": self.stream_length,
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
            return {"blink_probabilities": {}}

        blink_probs = {}
        for lag in self.lags:
            lag_results = [r for r in self.results if r["lag"] == lag]
            if lag_results:
                blink_rate = sum(1 for r in lag_results if r["blink_occurred"]) / len(lag_results)
                blink_probs[lag] = blink_rate

        return {"blink_probabilities": blink_probs}

    def reset(self) -> None:
        """Reset the task state."""
        self.results = []
        self.current_trial_idx = 0
        self.trials = self._generate_trials()

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
