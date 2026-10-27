"""Iowa Gambling Task (Bechara et al., 1994) on the APGI system.

Four decks pay a fixed reward per card and occasionally charge a penalty. Decks A and B pay
100 but cost more than they pay (net -250 per 10 cards); decks C and D pay 50 and net +250
per 10 cards. Each trial presents the chosen deck's stimulus to the APGI system, lets it
settle, shows the outcome, and lets the system's somatic-marker memory learn the result, so
the strength of the retrieved marker for a deck can be compared across good and bad decks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ._records import (
    STIMULUS_DIM,
    TaskRecord,
    draw_seed,
    present_stimulus,
    save_json,
    scaled_timestep,
)


class DeckType(Enum):
    """The four decks. A and B are disadvantageous, C and D advantageous."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


#: Per-deck payoff schedule. ``penalties`` are shuffled over the penalised cards of every
#: block of 10 draws (``penalty_frequency`` x 10 of them), so ``net_per_10`` holds exactly.
DECK_SCHEDULES: Dict[DeckType, Dict[str, Any]] = {
    DeckType.A: {
        "reward": 100,
        "penalty_frequency": 0.5,
        "penalties": [150, 200, 250, 300, 350],
        "net_per_10": -250,
    },
    DeckType.B: {
        "reward": 100,
        "penalty_frequency": 0.1,
        "penalties": [1250],
        "net_per_10": -250,
    },
    DeckType.C: {
        "reward": 50,
        "penalty_frequency": 0.5,
        "penalties": [25, 25, 50, 50, 100],
        "net_per_10": 250,
    },
    DeckType.D: {
        "reward": 50,
        "penalty_frequency": 0.1,
        "penalties": [250],
        "net_per_10": 250,
    },
}

GOOD_DECKS = (DeckType.C, DeckType.D)
BAD_DECKS = (DeckType.A, DeckType.B)

#: System steps spent on the deck stimulus and on the outcome, and the task time each step
#: stands for (ms).
DECISION_STEPS = 3
OUTCOME_STEPS = 2
STEP_MS = 50.0
BLOCK_SIZE = 20
#: Outcomes are divided by this before being stored as somatic markers (about one big loss).
OUTCOME_SCALE = 250.0


@dataclass
class IowaTrial(TaskRecord):
    """One planned card draw."""

    trial_number: int
    deck_choice: DeckType
    reward: int
    penalty: int
    deck_stimulus: np.ndarray
    outcome_stimulus: np.ndarray
    net_outcome: int = field(init=False)

    def __post_init__(self) -> None:
        self.net_outcome = self.reward - self.penalty


@dataclass
class IowaResult(TaskRecord):
    """Outcome of one executed trial."""

    trial_number: int
    deck_choice: DeckType
    reward: int
    penalty: int
    net_outcome: int
    running_total: int
    ignition_occurred: bool
    ignition_strength: float
    somatic_marker_strength: float
    decision_time: float
    anticipatory_response: float


class IowaGamblingTask:
    """Iowa Gambling Task.

    Parameters
    ----------
    num_decks : int
        Number of decks (the task is defined for the four standard decks).
    starting_balance, initial_balance : int, optional
        Starting balance; ``initial_balance`` wins when both are given. Default 2000.
    num_trials : int
        Number of card draws.
    deck_stimulus_strength, outcome_stimulus_strength : float
        Amplitude of the deck cue and of the outcome feedback presented to the system.
    interoceptive_gain : float
        Scales the anticipatory (interoceptive) response.
    deck_selection_strategy : {"random", "balanced", "cycle"}
        How the planned deck sequence is drawn: uniformly, equal counts shuffled, or A-B-C-D.
    seed : int, optional
        Seed for the schedule; by default drawn from numpy's global state.
    """

    def __init__(
        self,
        num_decks: int = 4,
        starting_balance: Optional[int] = None,
        num_trials: int = 100,
        initial_balance: Optional[int] = None,
        deck_stimulus_strength: float = 1.0,
        outcome_stimulus_strength: float = 1.0,
        interoceptive_gain: float = 1.0,
        deck_selection_strategy: str = "random",
        seed: Optional[int] = None,
    ):
        if initial_balance is None:
            initial_balance = starting_balance if starting_balance is not None else 2000
        self.num_decks = num_decks
        self.initial_balance = int(initial_balance)
        self.starting_balance = self.initial_balance
        self.num_trials = num_trials
        self.deck_stimulus_strength = deck_stimulus_strength
        self.outcome_stimulus_strength = outcome_stimulus_strength
        self.interoceptive_gain = interoceptive_gain
        self.deck_selection_strategy = deck_selection_strategy
        self._seed = draw_seed(seed)

        self.current_balance = self.initial_balance
        self.current_trial_idx = 0
        self.selection_history: List[int] = []
        self.results: List[IowaResult] = []
        self.trials: List[IowaTrial] = self._generate_trials()

    # -- stimuli ----------------------------------------------------------------------------
    def _deck_stimulus(self, deck: DeckType) -> np.ndarray:
        """A cue occupying the quarter of the input vector assigned to the deck."""
        stimulus = np.zeros(STIMULUS_DIM)
        width = STIMULUS_DIM // len(DeckType)
        start = list(DeckType).index(deck) * width
        phase = np.linspace(0.0, 4.0 * np.pi, width)
        stimulus[start : start + width] = self.deck_stimulus_strength * (1.0 + np.sin(phase)) / 2.0
        stimulus[start] += self.deck_stimulus_strength  # never all-zero, even for strength 0 phase
        return stimulus

    def _outcome_stimulus(self, reward: int, penalty: int) -> np.ndarray:
        """Feedback: a positive reward channel and a negative penalty channel."""
        stimulus = np.zeros(STIMULUS_DIM)
        half = STIMULUS_DIM // 2
        stimulus[:half] = self.outcome_stimulus_strength * np.tanh(reward / OUTCOME_SCALE)
        stimulus[half:] = -self.outcome_stimulus_strength * np.tanh(penalty / OUTCOME_SCALE)
        return stimulus

    # -- schedule ---------------------------------------------------------------------------
    def _deck_sequence(self, rng: np.random.Generator) -> List[DeckType]:
        decks = list(DeckType)
        if self.deck_selection_strategy == "balanced":
            sequence = [decks[i % len(decks)] for i in range(self.num_trials)]
            order = rng.permutation(self.num_trials)
            return [sequence[i] for i in order]
        if self.deck_selection_strategy == "cycle":
            return [decks[i % len(decks)] for i in range(self.num_trials)]
        return [decks[int(i)] for i in rng.integers(0, len(decks), self.num_trials)]

    @staticmethod
    def _card_block(deck: DeckType, rng: np.random.Generator) -> List[int]:
        """Penalties for the next 10 cards of ``deck`` (0 for unpenalised cards)."""
        schedule = DECK_SCHEDULES[deck]
        penalties = list(schedule["penalties"])
        block = [0] * 10
        slots = rng.choice(10, size=len(penalties), replace=False)
        for slot, penalty in zip(slots, rng.permutation(penalties)):
            block[int(slot)] = int(penalty)
        return block

    def _generate_trials(self) -> List[IowaTrial]:
        rng = np.random.default_rng(self._seed)
        blocks: Dict[DeckType, List[int]] = {deck: [] for deck in DeckType}
        trials: List[IowaTrial] = []
        for number, deck in enumerate(self._deck_sequence(rng)):
            if not blocks[deck]:
                blocks[deck] = self._card_block(deck, rng)
            penalty = blocks[deck].pop(0)
            reward = int(DECK_SCHEDULES[deck]["reward"])
            trials.append(
                IowaTrial(
                    trial_number=number,
                    deck_choice=deck,
                    reward=reward,
                    penalty=penalty,
                    deck_stimulus=self._deck_stimulus(deck),
                    outcome_stimulus=self._outcome_stimulus(reward, penalty),
                )
            )
        return trials

    # -- running ----------------------------------------------------------------------------
    def run_trial(self, apgi_framework: Any, trial: IowaTrial) -> IowaResult:
        """Run one trial on ``apgi_framework`` (an :class:`APGISystem`) and record the result."""
        deck_index = list(DeckType).index(trial.deck_choice)
        # The system stores one-dimensional actions; the deck cue (the context) tells decks apart
        action = np.array([deck_index + 1.0])

        markers = getattr(apgi_framework, "somatic_markers", None)
        marker_strength = 0.0
        if markers is not None:
            gain, _ = markers.retrieve(trial.deck_stimulus, action)
            marker_strength = float(np.max(np.abs(np.asarray(gain, dtype=float) - 1.0)))

        with scaled_timestep(apgi_framework, STEP_MS):
            decision = present_stimulus(apgi_framework, trial.deck_stimulus, DECISION_STEPS)
            present_stimulus(apgi_framework, trial.outcome_stimulus, OUTCOME_STEPS)

        if markers is not None:
            markers.learn(
                trial.deck_stimulus,
                action,
                float(np.clip(trial.net_outcome / OUTCOME_SCALE, -1.0, 1.0)),
                float(getattr(apgi_framework, "time", 0.0)),
            )

        first = decision.first_ignition_step
        decision_steps = decision.steps if first is None else first + 1
        self.current_balance += trial.net_outcome
        self.selection_history.append(deck_index)
        result = IowaResult(
            trial_number=trial.trial_number,
            deck_choice=trial.deck_choice,
            reward=trial.reward,
            penalty=trial.penalty,
            net_outcome=trial.net_outcome,
            running_total=self.current_balance,
            ignition_occurred=decision.ignited,
            ignition_strength=decision.peak_signal,
            somatic_marker_strength=marker_strength,
            decision_time=float(decision_steps * STEP_MS),
            anticipatory_response=float(decision.mean_intero_signal * self.interoceptive_gain),
        )
        self.results.append(result)
        self.current_trial_idx = len(self.results)
        return result

    def run_all_trials(self, apgi_framework: Any) -> Dict[str, Any]:
        """Run every planned trial from a fresh balance and return the analysis."""
        self.results = []
        self.current_balance = self.initial_balance
        self.selection_history = []
        for trial in self.trials:
            self.run_trial(apgi_framework, trial)
        return self.analyze_results()

    # -- analysis ---------------------------------------------------------------------------
    def calculate_net_score(self) -> float:
        """Net score: advantageous (C, D) minus disadvantageous (A, B) choices."""
        good = sum(1 for r in self.results if r.deck_choice in GOOD_DECKS)
        return float(good - (len(self.results) - good))

    def get_deck_statistics(self) -> Dict[str, Any]:
        counts = [self.selection_history.count(i) for i in range(len(DeckType))]
        total = len(self.selection_history)
        return {
            "selection_counts": counts,
            "total_selections": total,
            "advantageous_ratio": (counts[2] + counts[3]) / max(total, 1),
        }

    def analyze_results(self) -> Dict[str, Any]:
        """Summary of the executed trials (``{"error": ..., "total_trials": 0}`` if none)."""
        results = self.results
        if not results:
            return {"error": "No results to analyze", "total_trials": 0}

        total = len(results)
        good = sum(1 for r in results if r.deck_choice in GOOD_DECKS)

        by_deck: Dict[str, Dict[str, float]] = {}
        for deck in DeckType:
            picks = [r for r in results if r.deck_choice is deck]
            by_deck[deck.value] = {
                "selections": len(picks),
                "selection_percentage": len(picks) / total * 100.0,
                "avg_net_outcome": float(np.mean([r.net_outcome for r in picks])) if picks else 0.0,
                "avg_somatic_marker": (
                    float(np.mean([r.somatic_marker_strength for r in picks])) if picks else 0.0
                ),
            }

        by_block: List[Dict[str, float]] = []
        for index, start in enumerate(range(0, total, BLOCK_SIZE)):
            block = results[start : start + BLOCK_SIZE]
            block_good = sum(1 for r in block if r.deck_choice in GOOD_DECKS)
            by_block.append(
                {
                    "block": index + 1,
                    "trials": len(block),
                    "good_deck_choices": block_good,
                    "good_deck_percentage": block_good / len(block) * 100.0,
                    "net_earnings": float(sum(r.net_outcome for r in block)),
                }
            )

        def marker_average(decks: Any) -> float:
            values = [r.somatic_marker_strength for r in results if r.deck_choice in decks]
            return float(np.mean(values)) if values else 0.0

        first, last = results[:BLOCK_SIZE], results[-BLOCK_SIZE:]
        improvement = sum(1 for r in last if r.deck_choice in GOOD_DECKS) - sum(
            1 for r in first if r.deck_choice in GOOD_DECKS
        )
        final_balance = results[-1].running_total
        return {
            "total_trials": total,
            "initial_balance": self.initial_balance,
            "final_balance": final_balance,
            "total_earnings": final_balance - self.initial_balance,
            "good_deck_choices": good,
            "bad_deck_choices": total - good,
            "advantageous_ratio": good / total,
            "net_score": float(good - (total - good)),
            "by_deck": by_deck,
            "by_block": by_block,
            "somatic_markers": {
                "good_decks_avg": marker_average(GOOD_DECKS),
                "bad_decks_avg": marker_average(BAD_DECKS),
                "overall_avg": float(np.mean([r.somatic_marker_strength for r in results])),
            },
            "learning_improvement": improvement,
            "task_parameters": {
                "num_trials": self.num_trials,
                "num_decks": self.num_decks,
                "initial_balance": self.initial_balance,
                "deck_selection_strategy": self.deck_selection_strategy,
            },
        }

    def save_results(self, filename: str) -> None:
        """Save the trial results to a JSON file."""
        save_json(filename, self.results)

    def print_results(self, analysis: Dict[str, Any]) -> None:
        """Print an analysis to the console."""
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
        """Return to the initial state with a freshly generated schedule."""
        self.current_balance = self.initial_balance
        self.current_trial_idx = 0
        self.selection_history = []
        self.results = []
        self.trials = self._generate_trials()
