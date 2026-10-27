"""Somatic Marker System.

Implements Damasio's somatic marker hypothesis:
Learns associations between contexts/actions and body states,
then uses these to bias decision-making.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


@dataclass
class SomaticMarker:
    """A somatic marker: association between situation and body response.

    Stores: (context_pattern, action_pattern) -> body_outcome
    """

    context_pattern: np.ndarray[Any, np.dtype[Any]]
    action_pattern: np.ndarray[Any, np.dtype[Any]]
    body_outcome: float  # Valence: positive = good, negative = bad
    strength: float = 1.0  # Association strength
    access_count: int = 0
    last_update_time: float = 0.0


class SomaticMarkerSystem:
    """Stores and retrieves somatic markers for decision guidance.

    Implements Damasio's somatic marker hypothesis: the brain learns
    associations between situations/actions and their bodily outcomes
    (positive or negative), then uses these learned associations to bias
    future decision-making. When a similar situation is encountered, the
    retrieved marker modulates interoceptive gain to guide behavior.

    The system performs four key functions:
    1. **Learning**: Store associations (context, action) -> body_outcome
    2. **Retrieval**: Find markers matching current context-action pairs
    3. **Gain Modulation**: Convert marker valence to interoceptive gain
    4. **Decision Biasing**: Bias action selection toward positive outcomes

    Markers are stored with:
    - Context pattern: State representation when marker was learned
    - Action pattern: Action taken in that context
    - Body outcome: Valence of resulting body state (-1 to +1)
    - Strength: Association strength (0-1), decays if unused
    - Access count: Number of times retrieved

    Retrieval uses cosine similarity for pattern matching. When a marker
    is retrieved, its body outcome is converted to a gain modulation factor:
    - Positive outcome (good) -> increase gain (enhance interoceptive signal)
    - Negative outcome (bad) -> decrease gain (suppress interoceptive signal)

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing:
        - somatic_markers.capacity: Maximum number of markers (default: 10000)
        - somatic_markers.learning_rate: Learning rate for updates (default: 0.05)
        - somatic_markers.decay_rate: Decay rate for unused markers (default: 0.001)
        - somatic_markers.retrieval_threshold: Minimum strength for retrieval (default: 0.3)
        - somatic_markers.gain_modulation_range: [min, max] gain values (default: [0.5, 2.0])

    Attributes
    ----------
    markers : List[SomaticMarker]
        Stored somatic markers
    capacity : int
        Maximum number of markers to store
    learning_rate : float
        Rate at which marker outcomes are updated
    decay_rate : float
        Rate at which unused markers decay
    retrieval_threshold : float
        Minimum strength required for successful retrieval
    gain_range : List[float]
        [min, max] range for gain modulation
    total_retrievals : int
        Total number of retrieval attempts
    successful_retrievals : int
        Number of successful retrievals

    Examples
    --------
    >>> config = {'somatic_markers': {'capacity': 1000}}
    >>> sm_system = SomaticMarkerSystem(config)
    >>> context = np.random.randn(10)
    >>> action = np.random.randn(5)
    >>> sm_system.learn(context, action, body_outcome=0.8, current_time=100.0)
    >>> gain, found = sm_system.retrieve(context, action)
    >>> print(f"Gain: {gain}, Found: {found}")
    Gain: 1.65, Found: True
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize vectorized somatic marker system."""
        self.config = config
        self.batch_size = config.get("active_inference", {}).get("batch_size", 1)
        marker_config = config.get("somatic_markers", {})

        # System parameters
        self.capacity = marker_config.get("capacity", 10000)
        self.learning_rate = marker_config.get("learning_rate", 0.05)
        self.decay_rate = marker_config.get("decay_rate", 0.001)
        self.retrieval_threshold = marker_config.get("retrieval_threshold", 0.3)
        self.gain_range = marker_config.get("gain_modulation_range", [0.5, 2.0])

        # Storage: use parallel arrays for vectorized similarity search
        self.context_patterns: Optional[np.ndarray] = None  # (N, D_ctx)
        self.action_patterns: Optional[np.ndarray] = None  # (N, D_act)
        self.outcomes: Optional[np.ndarray] = None  # (N,)
        self.strengths: Optional[np.ndarray] = None  # (N,)
        self.num_markers = 0

        # Statistics
        self.total_retrievals = 0
        self.successful_retrievals = 0

    def learn(
        self,
        context: np.ndarray,
        action: np.ndarray,
        body_outcome: np.ndarray,
        current_time: float = 0.0,
    ) -> None:
        """Learn or update markers for a batch (B, D)."""
        # Sequential update for each agent in batch (learning is less frequent)
        for i in range(self.batch_size):
            c = context[i] if context.ndim > 1 else context
            a = action[i] if action.ndim > 1 else action
            o = body_outcome[i] if hasattr(body_outcome, "__len__") else body_outcome
            self._learn_single_marker(c, a, o, current_time)

    def _learn_single_marker(
        self,
        context: np.ndarray,
        action: np.ndarray,
        outcome: Union[np.ndarray, float],
        current_time: float,
    ) -> None:
        # Initialize storage if needed
        if (
            self.context_patterns is None
            or self.action_patterns is None
            or self.outcomes is None
            or self.strengths is None
        ):
            self.context_patterns = np.zeros((self.capacity, len(context)))
            self.action_patterns = np.zeros((self.capacity, len(action)))
            self.outcomes = np.zeros(self.capacity)
            self.strengths = np.zeros(self.capacity)

        # Check for similar
        idx = self._find_best_match_idx(context, action)
        if idx is not None:
            self.outcomes[idx] = (1 - self.learning_rate) * self.outcomes[
                idx
            ] + self.learning_rate * outcome
            self.strengths[idx] = min(1.0, self.strengths[idx] + 0.1)
        elif self.num_markers < self.capacity:
            idx = self.num_markers
            self.context_patterns[idx] = context
            self.action_patterns[idx] = action
            self.outcomes[idx] = outcome
            self.strengths[idx] = 0.5
            self.num_markers += 1
        else:
            weakest = np.argmin(self.strengths[: self.num_markers])
            self.context_patterns[weakest] = context
            self.action_patterns[weakest] = action
            self.outcomes[weakest] = outcome
            self.strengths[weakest] = 0.5

    def retrieve(self, context: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve markers for a batch (B, D)."""
        if (
            self.num_markers == 0
            or self.context_patterns is None
            or self.action_patterns is None
            or self.strengths is None
            or self.outcomes is None
        ):
            return np.ones(self.batch_size), np.zeros(self.batch_size, dtype=bool)

        # Vectorized similarity search
        # Context sim (B, N)
        if context.ndim == 1:
            context = context[np.newaxis, :]
        if action.ndim == 1:
            action = action[np.newaxis, :]

        ctx_norms = np.linalg.norm(context, axis=1, keepdims=True)
        marker_ctx_norms = np.linalg.norm(self.context_patterns[: self.num_markers], axis=1)
        sim_ctx = (context @ self.context_patterns[: self.num_markers].T) / (
            ctx_norms @ marker_ctx_norms[np.newaxis, :] + 1e-9
        )

        act_norms = np.linalg.norm(action, axis=1, keepdims=True)
        marker_act_norms = np.linalg.norm(self.action_patterns[: self.num_markers], axis=1)
        sim_act = (action @ self.action_patterns[: self.num_markers].T) / (
            act_norms @ marker_act_norms[np.newaxis, :] + 1e-9
        )

        sim = 0.7 * sim_ctx + 0.3 * sim_act

        best_marker_indices = np.argmax(sim, axis=1)
        best_sims = np.max(sim, axis=1)

        found_mask = (best_sims > 0.8) & (
            self.strengths[best_marker_indices] > self.retrieval_threshold
        )

        gains = np.ones(self.batch_size)
        marker_outcomes = self.outcomes[best_marker_indices]
        normalized_outcomes = (marker_outcomes + 1.0) / 2.0
        calculated_gains = self.gain_range[0] + normalized_outcomes * (
            self.gain_range[1] - self.gain_range[0]
        )

        gains[found_mask] = calculated_gains[found_mask]

        return gains, found_mask

    def _find_best_match_idx(
        self, context: np.ndarray, action: np.ndarray, similarity_threshold: float = 0.8
    ) -> Optional[int]:
        if self.num_markers == 0 or self.context_patterns is None or self.action_patterns is None:
            return None

        ctx_norm = np.linalg.norm(context)
        marker_ctx_norms = np.linalg.norm(self.context_patterns[: self.num_markers], axis=1)
        sim_ctx = (self.context_patterns[: self.num_markers] @ context) / (
            marker_ctx_norms * ctx_norm + 1e-9
        )

        act_norm = np.linalg.norm(action)
        marker_act_norms = np.linalg.norm(self.action_patterns[: self.num_markers], axis=1)
        sim_act = (self.action_patterns[: self.num_markers] @ action) / (
            marker_act_norms * act_norm + 1e-9
        )

        sim = 0.7 * sim_ctx + 0.3 * sim_act
        best_idx = int(np.argmax(sim))
        if sim[best_idx] > similarity_threshold:
            return best_idx
        return None

    def decay_markers(self, dt: float) -> None:
        """Apply decay for all markers."""
        if self.num_markers > 0 and self.strengths is not None:
            self.strengths[: self.num_markers] *= 1.0 - self.decay_rate * dt / 1000.0
            self.strengths[: self.num_markers] = np.maximum(0.0, self.strengths[: self.num_markers])

    def reset(self) -> None:
        """Reset storage."""
        self.num_markers = 0
        if self.strengths is not None:
            self.strengths.fill(0.0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the somatic marker system."""
        # Calculate derived statistics
        capacity_used = self.num_markers / self.capacity if self.capacity > 0 else 0.0
        retrieval_success_rate = (
            self.successful_retrievals / self.total_retrievals if self.total_retrievals > 0 else 0.0
        )
        avg_strength = (
            float(np.mean(self.strengths[: self.num_markers]))
            if self.num_markers > 0 and self.strengths is not None
            else 0.0
        )
        avg_outcome = (
            float(np.mean(self.outcomes[: self.num_markers]))
            if self.num_markers > 0 and self.outcomes is not None
            else 0.0
        )

        return {
            "num_markers": self.num_markers,
            "total_retrievals": self.total_retrievals,
            "successful_retrievals": self.successful_retrievals,
            "capacity": self.capacity,
            "capacity_used": capacity_used,
            "retrieval_success_rate": retrieval_success_rate,
            "avg_strength": avg_strength,
            "avg_outcome": avg_outcome,
        }
