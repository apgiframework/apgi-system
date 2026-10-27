"""Stub module for Iowa gambling task.

This module provides type stubs for Iowa gambling experimental task
that is being migrated from the old apgi_simulation structure.
"""

from enum import Enum
from typing import Any, Dict, List

import numpy as np


class DeckType(Enum):
    """Deck types for Iowa gambling task."""

    ADVANTAGEOUS_A = "advantageous_a"
    ADVANTAGEOUS_B = "advantageous_b"
    DISADVANTAGEOUS_A = "disadvantageous_a"
    DISADVANTAGEOUS_B = "disadvantageous_b"


# Default deck schedules for the Iowa gambling task
DECK_SCHEDULES = {
    DeckType.ADVANTAGEOUS_A: {"mean": 100, "variance": 50},
    DeckType.ADVANTAGEOUS_B: {"mean": 100, "variance": 50},
    DeckType.DISADVANTAGEOUS_A: {"mean": 100, "variance": 200},
    DeckType.DISADVANTAGEOUS_B: {"mean": 100, "variance": 200},
}


class IowaGamblingTask:
    """Iowa gambling experimental task."""

    def __init__(
        self,
        num_decks: int = 4,
        starting_balance: int = 1000,
        num_trials: int = 100,
        initial_balance: int = 1000,
        deck_stimulus_strength: float = 1.0,
        outcome_stimulus_strength: float = 1.0,
        interoceptive_gain: float = 1.0,
        deck_selection_strategy: str = "random",
    ):
        """Initialize Iowa gambling task.

        Parameters
        ----------
        num_decks : int, optional
            Number of card decks, by default 4
        starting_balance : int, optional
            Starting balance in points, by default 1000
        num_trials : int, optional
            Number of trials, by default 100
        initial_balance : int, optional
            Initial balance, by default 1000
        deck_stimulus_strength : float, optional
            Strength of deck stimulus, by default 1.0
        outcome_stimulus_strength : float, optional
            Strength of outcome stimulus, by default 1.0
        interoceptive_gain : float, optional
            Interoceptive gain, by default 1.0
        deck_selection_strategy : str, optional
            Deck selection strategy, by default "random"
        """
        self.num_decks = num_decks
        self.starting_balance = starting_balance
        self.num_trials = num_trials
        self.initial_balance = initial_balance
        self.deck_stimulus_strength = deck_stimulus_strength
        self.outcome_stimulus_strength = outcome_stimulus_strength
        self.interoceptive_gain = interoceptive_gain
        self.deck_selection_strategy = deck_selection_strategy
        self.deck_rewards: List[Dict[str, float]] = [
            {"mean": 100, "variance": 50},  # Advantageous
            {"mean": 100, "variance": 50},  # Advantageous
            {"mean": 100, "variance": 200},  # Disadvantageous
            {"mean": 100, "variance": 200},  # Disadvantageous
        ]
        self.selection_history: List[int] = []
        self.trials = self._generate_trials()
        self.results: List[Dict[str, Any]] = []
        self.current_trial = 0

    def draw_card(self, deck_index: int) -> float:
        """Draw a card from a specific deck.

        Parameters
        ----------
        deck_index : int
            Index of deck to draw from

        Returns
        -------
        float
            Reward value
        """
        reward = np.random.normal(
            self.deck_rewards[deck_index]["mean"],
            self.deck_rewards[deck_index]["variance"],
        )
        self.selection_history.append(deck_index)
        return reward

    def calculate_net_score(self) -> float:
        """Calculate net score based on deck selections.

        Returns
        -------
        float
            Net score
        """
        # Simplified scoring
        advantageous_selections = sum(1 for s in self.selection_history if s < 2)
        disadvantageous_selections = len(self.selection_history) - advantageous_selections
        return advantageous_selections - disadvantageous_selections

    def get_deck_statistics(self) -> Dict[str, Any]:
        """Get statistics about deck selections.

        Returns
        -------
        Dict[str, Any]
            Deck selection statistics
        """
        selection_counts = [self.selection_history.count(i) for i in range(self.num_decks)]
        return {
            "selection_counts": selection_counts,
            "total_selections": len(self.selection_history),
            "advantageous_ratio": (selection_counts[0] + selection_counts[1])
            / max(len(self.selection_history), 1),
        }

    def _generate_trials(self) -> List[Dict[str, Any]]:
        """Generate trial configurations.

        Returns
        -------
        List[Dict[str, Any]]
            List of trial configurations
        """
        trials = []
        for i in range(self.num_trials):
            trial = {
                "trial_number": i,
                "deck_index": i % self.num_decks,
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
        reward = self.draw_card(trial["deck_index"])
        return {
            "trial_number": trial["trial_number"],
            "deck_index": trial["deck_index"],
            "reward": reward,
            "balance": self.starting_balance
            + sum(self.deck_rewards[s]["mean"] for s in self.selection_history),
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

        # Calculate analysis
        deck_choices = [r["deck_index"] for r in self.results]
        good_deck_choices = sum(1 for d in deck_choices if d < 2)
        bad_deck_choices = sum(1 for d in deck_choices if d >= 2)

        # Add running_total to each result
        running_balance = self.starting_balance
        for r in self.results:
            running_balance += r["reward"]
            r["running_total"] = running_balance
            r["deck_choice"] = DeckType(list(DeckType)[r["deck_index"]])
            r["net_outcome"] = r["reward"]

        return {
            "total_trials": len(self.results),
            "initial_balance": self.starting_balance,
            "final_balance": self.starting_balance + sum(r["reward"] for r in self.results),
            "total_earnings": sum(r["reward"] for r in self.results),
            "good_deck_choices": good_deck_choices,
            "bad_deck_choices": bad_deck_choices,
            "advantageous_ratio": good_deck_choices / max(len(self.results), 1),
            "net_score": self.calculate_net_score(),
            "by_deck": {
                "A": {
                    "selections": deck_choices.count(0),
                    "selection_percentage": deck_choices.count(0) / len(self.results) * 100,
                    "avg_net_outcome": 0.0,
                    "avg_somatic_marker": 0.0,
                },
                "B": {
                    "selections": deck_choices.count(1),
                    "selection_percentage": deck_choices.count(1) / len(self.results) * 100,
                    "avg_net_outcome": 0.0,
                    "avg_somatic_marker": 0.0,
                },
                "C": {
                    "selections": deck_choices.count(2),
                    "selection_percentage": deck_choices.count(2) / len(self.results) * 100,
                    "avg_net_outcome": 0.0,
                    "avg_somatic_marker": 0.0,
                },
                "D": {
                    "selections": deck_choices.count(3),
                    "selection_percentage": deck_choices.count(3) / len(self.results) * 100,
                    "avg_net_outcome": 0.0,
                    "avg_somatic_marker": 0.0,
                },
            },
            "by_block": [
                {"good_deck_percentage": good_deck_choices / max(len(self.results), 1) * 100}
            ],
            "somatic_markers": {"good_decks_avg": 0.5, "bad_decks_avg": 0.3},
            "learning_improvement": 0,
            "task_parameters": {"num_trials": self.num_trials, "num_decks": self.num_decks},
        }

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze trial results.

        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        if not self.results:
            return {"total_trials": 0, "advantageous_ratio": 0.0}

        deck_choices = [r["deck_index"] for r in self.results]
        good_deck_choices = sum(1 for d in deck_choices if d < 2)
        return {
            "total_trials": len(self.results),
            "advantageous_ratio": good_deck_choices / len(self.results),
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
        print("IOWA GAMBLING TASK RESULTS")
        print("=" * 50)
        print(f"Total Trials: {analysis['total_trials']}")
        print(f"Final Balance: {analysis['final_balance']:.0f}")
        print(f"Total Earnings: {analysis['total_earnings']:.0f}")
        print(f"Good Deck Choices: {analysis['good_deck_choices']}")
        print(f"Bad Deck Choices: {analysis['bad_deck_choices']}")
        print(f"Advantageous Ratio: {analysis['advantageous_ratio']:.2%}")

    def reset(self) -> None:
        """Reset task to initial state."""
        self.selection_history = []
        self.trials = self._generate_trials()
        self.results = []
        self.current_trial = 0
