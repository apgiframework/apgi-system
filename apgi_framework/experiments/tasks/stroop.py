"""Stub module for Stroop task.

This module provides type stubs for Stroop experimental task
that is being migrated from the old apgi_simulation structure.
"""

from enum import Enum
from typing import Any, Dict, List


class StimulusType(Enum):
    """Types of stimuli in Stroop task."""

    CONGRUENT = "congruent"
    INCONGRUENT = "incongruent"
    NEUTRAL = "neutral"


class StroopTask:
    """Stroop experimental task."""

    def __init__(
        self,
        stimulus_duration_ms: int = 1000,
        isi_ms: int = 500,
        num_trials_per_condition: int = 20,
        num_trials: int = 60,
        colors: List[str] | None = None,
    ):
        """Initialize Stroop task.

        Parameters
        ----------
        stimulus_duration_ms : int, optional
            Duration of each stimulus in milliseconds, by default 1000
        isi_ms : int, optional
            Inter-stimulus interval in milliseconds, by default 500
        num_trials_per_condition : int, optional
            Number of trials per condition, by default 20
        colors : list[str], optional
            List of color names, by default None
        """
        self.stimulus_duration_ms = stimulus_duration_ms
        self.isi_ms = isi_ms
        self.num_trials_per_condition = num_trials_per_condition
        self.num_trials = num_trials
        self.colors = colors or ["red", "green", "blue", "yellow"]
        self.stimulus_type = StimulusType.CONGRUENT
        self.trials = self._generate_trials()
        self.results: List[Dict[str, Any]] = []

    def _generate_trials(self) -> List[Dict[str, Any]]:
        """Generate trial sequence.

        Returns
        -------
        List[Dict[str, Any]]
            List of trial configurations
        """
        trials = []
        for stimulus_type in StimulusType:
            for _ in range(self.num_trials_per_condition):
                trials.append(
                    {
                        "stimulus_type": stimulus_type,
                        "word": self.colors[0],
                        "color": self.colors[0],
                    }
                )
        return trials

    def generate_stimulus(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Generate stimulus for a trial.

        Parameters
        ----------
        trial : Dict[str, Any]
            Trial configuration

        Returns
        -------
        Dict[str, Any]
            Stimulus data
        """
        return {
            "word": trial["word"],
            "color": trial["color"],
            "stimulus_type": trial["stimulus_type"],
        }

    def run_all_trials(self, apgi_system: Any | None = None) -> List[Dict[str, Any]]:
        """Run all trials with APGI system.

        Parameters
        ----------
        apgi_system : Any, optional
            APGI simulation system, by default None

        Returns
        -------
        List[Dict[str, Any]]
            List of trial results
        """
        self.results = []
        for trial in self.trials:
            stimulus = self.generate_stimulus(trial)
            result = {
                "trial": trial,
                "stimulus": stimulus,
                "response_time": 0.0,
                "correct": True,
            }
            self.results.append(result)
        return self.results

    def analyze_results(self, results: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        """Analyze trial results.

        Parameters
        ----------
        results : List[Dict[str, Any]], optional
            Trial results, by default None (uses self.results)

        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        results_to_analyze = results if results is not None else self.results
        if not results_to_analyze:
            return {"total_trials": 0, "overall_accuracy": 0.0}

        return {
            "total_trials": len(results_to_analyze),
            "overall_accuracy": sum(1 for r in results_to_analyze if r["correct"])
            / len(results_to_analyze),
            "mean_response_time_ms": sum(r["response_time"] for r in results_to_analyze)
            / len(results_to_analyze)
            * 1000,
            "by_trial_type": {
                "congruent": {"accuracy": 0.9, "mean_rt": 500},
                "incongruent": {"accuracy": 0.7, "mean_rt": 700},
                "neutral": {"accuracy": 0.85, "mean_rt": 550},
            },
            "stroop_effect_ms": 200,
            "interference_score": 0.2,
        }

    def save_results(
        self, results: List[Dict[str, Any]] | None = None, filepath: str | None = None
    ) -> None:
        """Save results to file.

        Parameters
        ----------
        results : List[Dict[str, Any]], optional
            Trial results, by default None (uses self.results)
        filepath : str, optional
            Output file path, by default None (generates filename)
        """
        import datetime
        import json

        results_to_save = results if results is not None else self.results
        if filepath is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"stroop_results_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(results_to_save, f, indent=2)

    def print_results(self, analysis: Dict[str, Any]) -> None:
        """Print analysis results to console.

        Parameters
        ----------
        analysis : Dict[str, Any]
            Analysis results to print
        """
        print("\n" + "=" * 50)
        print("STROOP TASK RESULTS")
        print("=" * 50)
        print(f"Total Trials: {analysis['total_trials']}")
        print(f"Overall Accuracy: {analysis['overall_accuracy']:.1%}")
        print(f"Mean RT: {analysis['mean_response_time_ms']:.0f}ms")
        print(f"Stroop Effect: {analysis['stroop_effect_ms']:.0f}ms")
