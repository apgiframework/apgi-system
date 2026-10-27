"""Shared building blocks for the experimental tasks.

* :class:`TaskRecord` - trial and result dataclasses that also read like read-only mappings,
  so ``result.reward``, ``result["reward"]`` and ``"reward" in result`` all work (the API,
  the GUI and the analysis code use different styles).
* :func:`present_stimulus` / :func:`scaled_timestep` - drive an :class:`APGISystem` with a
  stimulus and read back its ignition dynamics.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

#: Length of the exteroceptive observation vector the APGI system consumes.
STIMULUS_DIM = 256


def jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays, enums and records to plain JSON types."""
    if isinstance(value, TaskRecord):
        return value.to_dict()
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(jsonable(k)): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


def save_json(filename: str, results: Sequence[Any]) -> None:
    """Write ``{"results": [...]}`` to ``filename``."""
    with open(filename, "w") as f:
        json.dump({"results": jsonable(list(results))}, f, indent=2)


class TaskRecord:
    """Mixin for dataclasses that should also behave like read-only mappings."""

    def keys(self) -> List[str]:
        return [f.name for f in dataclasses.fields(self)]  # type: ignore[arg-type]

    def __getitem__(self, key: str) -> Any:
        if key in self.keys():
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in self.keys()

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key) if key in self.keys() else default

    def items(self) -> List[Tuple[str, Any]]:
        return [(k, getattr(self, k)) for k in self.keys()]

    def to_dict(self) -> Dict[str, Any]:
        return {k: jsonable(v) for k, v in self.items()}


@dataclass
class PhaseOutcome:
    """What the APGI system did while a stimulus was presented."""

    signals: List[float] = field(default_factory=list)
    ignitions: List[bool] = field(default_factory=list)
    intero_signals: List[float] = field(default_factory=list)
    threshold: float = 0.0

    @property
    def steps(self) -> int:
        return len(self.signals)

    @property
    def ignition_count(self) -> int:
        return int(sum(self.ignitions))

    @property
    def ignited(self) -> bool:
        return any(self.ignitions)

    @property
    def peak_signal(self) -> float:
        return float(max(self.signals)) if self.signals else 0.0

    @property
    def mean_intero_signal(self) -> float:
        return float(np.mean(self.intero_signals)) if self.intero_signals else 0.0

    @property
    def first_ignition_step(self) -> Optional[int]:
        for i, occurred in enumerate(self.ignitions):
            if occurred:
                return i
        return None

    def extend(self, other: "PhaseOutcome") -> None:
        self.signals += other.signals
        self.ignitions += other.ignitions
        self.intero_signals += other.intero_signals
        self.threshold = other.threshold or self.threshold


def _scalar(value: Any) -> float:
    return float(np.max(np.asarray(value, dtype=float)))


def present_stimulus(system: Any, stimulus: np.ndarray, steps: int) -> PhaseOutcome:
    """Step ``system`` ``steps`` times with ``stimulus`` and collect its ignition dynamics."""
    outcome = PhaseOutcome()
    observation = np.asarray(stimulus, dtype=np.float64)
    for _ in range(max(0, steps)):
        state = system.step(observation)
        ignition = state["ignition"]
        outcome.ignitions.append(bool(np.any(ignition["ignition_occurred"])))
        outcome.signals.append(_scalar(ignition["total_signal"]))
        outcome.intero_signals.append(_scalar(ignition.get("intero_signal", 0.0)))
        outcome.threshold = _scalar(ignition.get("threshold", 0.0))
    return outcome


@contextlib.contextmanager
def scaled_timestep(system: Any, timestep_ms: float) -> Iterator[None]:
    """Temporarily make each system step stand for ``timestep_ms`` of task time."""
    previous = getattr(system, "timestep_ms", None)
    if previous is not None:
        system.timestep_ms = float(timestep_ms)
    try:
        yield
    finally:
        if previous is not None:
            system.timestep_ms = previous


def blank_stimulus() -> np.ndarray:
    """An empty (all-zero) observation."""
    return np.zeros(STIMULUS_DIM)


def draw_seed(seed: Optional[int]) -> int:
    """``seed`` if given, else one drawn from numpy's global state (so ``np.random.seed`` works)."""
    return int(seed) if seed is not None else int(np.random.randint(0, 2**31 - 1))
