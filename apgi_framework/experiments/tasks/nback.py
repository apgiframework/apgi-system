"""Stub module for N-back task.

This module provides type stubs for N-back experimental task
that is being migrated from the old apgi_simulation structure.
"""

from enum import Enum
from typing import Any, Dict, List

import numpy as np


class NBackLevel(Enum):
    """N-back difficulty levels."""

    ONE_BACK = 1
    TWO_BACK = 2
    THREE_BACK = 3


class StimulusType(Enum):
    """Types of stimuli in N-back task."""

    LETTERS = "letters"
    NUMBERS = "numbers"
    SHAPES = "shapes"


class NBackTask:
    """N-back experimental task."""

    def __init__(
        self,
        n_level: int = 2,
        stimulus_type: str = "letters",
        sequence_length: int = 20,
        isi_ms: int = 1500,
        num_blocks: int = 5,
        num_trials: int = 100,
        num_trials_total: int = 100,
    ):
        """Initialize N-back task.

        Parameters
        ----------
        n_level : int, optional
            N-back level (1, 2, or 3), by default 2
        stimulus_type : str, optional
            Type of stimuli, by default "letters"
        sequence_length : int, optional
            Length of stimulus sequence, by default 20
        isi_ms : int, optional
            Inter-stimulus interval in milliseconds, by default 1500
        num_blocks : int, optional
            Number of experimental blocks, by default 5
        num_trials : int, optional
            Total number of trials, by default 100
        """
        self.n_level = n_level
        self.stimulus_type = stimulus_type
        self.sequence_length = sequence_length
        self.isi_ms = isi_ms
        self.num_blocks = num_blocks
        self.num_trials = num_trials
        self.num_trials_total = num_trials_total
        self.stimuli = self._generate_stimuli()
        self.target_positions = self._generate_targets()
        self.results: List[Dict[str, Any]] = []

    def _generate_stimuli(self) -> List[str]:
        """Generate stimulus sequence.

        Returns
        -------
        List[str]
            List of stimuli
        """
        if self.stimulus_type == "letters":
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            return [alphabet[i % len(alphabet)] for i in range(self.sequence_length)]
        elif self.stimulus_type == "numbers":
            return [str(i % 10) for i in range(self.sequence_length)]
        else:
            return ["shape_" + str(i % 5) for i in range(self.sequence_length)]

    def _generate_targets(self) -> List[bool]:
        """Generate target positions.

        Returns
        -------
        List[bool]
            List indicating target positions
        """
        targets = [False] * self.sequence_length
        target_probability = 0.3
        for i in range(self.n_level, self.sequence_length):
            if np.random.random() < target_probability:
                targets[i] = True
                # Make the stimulus match the one n positions back
                self.stimuli[i] = self.stimuli[i - self.n_level]
        return targets

    def is_target(self, position: int) -> bool:
        """Check if position is a target.

        Parameters
        ----------
        position : int
            Position in sequence

        Returns
        -------
        bool
            True if target
        """
        if position < self.n_level:
            return False
        return self.stimuli[position] == self.stimuli[position - self.n_level]

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
        for i, stimulus in enumerate(self.stimuli):
            result = {
                "position": i,
                "stimulus": stimulus,
                "is_target": self.is_target(i),
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
            return {"total_trials": 0, "accuracy": 0.0}

        return {
            "total_trials": len(results_to_analyze),
            "targets": sum(1 for r in results_to_analyze if r["is_target"]),
            "correct": sum(1 for r in results_to_analyze if r["correct"]),
            "accuracy": sum(1 for r in results_to_analyze if r["correct"])
            / len(results_to_analyze),
            "mean_rt": sum(r["response_time"] for r in results_to_analyze)
            / len(results_to_analyze)
            * 1000,
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
            filepath = f"nback_results_{timestamp}.json"

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
        print("N-BACK TASK RESULTS")
        print("=" * 50)
        print(f"Total Trials: {analysis['total_trials']}")
        print(f"Targets: {analysis['targets']}")
        print(f"Accuracy: {analysis['accuracy']:.1%}")
        print(f"Mean RT: {analysis['mean_rt']:.0f}ms")
