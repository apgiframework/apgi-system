"""Narrative Self - long-term identity and autobiographical memory."""

from collections import deque
from typing import Any, Dict

import numpy as np


class NarrativeSelf:
    """Narrative self-model with episodic memory and long-term identity.

    Implements the narrative self - the extended, autobiographical sense
    of identity that emerges from episodic memory and self-reflection.
    This represents the reflective, story-based understanding of oneself
    as a continuous agent with a past, present, and future.

    **Core Components**:
    - **Episodic Memory**: Storage of significant life experiences
    - **Identity Vector**: Slowly-evolving representation of personal identity
    - **Goal Hierarchy**: Long-term goals and values (currently placeholder)
    - **Narrative Coherence**: Consistency of self-story over time

    **Key Functions**:
    - Selectively encodes significant experiences into episodic memory
    - Gradually consolidates identity through experience integration
    - Maintains narrative coherence across different time scales
    - Supports autobiographical reasoning and future planning

    **Memory Dynamics**:
    - Selective encoding: Only ~1% of experiences stored (significance-based)
    - Capacity limits: Fixed maximum number of episodes
    - Consolidation: Identity slowly updated based on stored experiences
    - Forgetting: Oldest memories displaced when capacity exceeded

    The narrative self operates on much longer timescales than the minimal
    self, reflecting the slow accumulation of autobiographical knowledge.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing narrative self settings:
        - self_model.narrative_self.episodic_memory_capacity: Maximum episodes (default: 1000)
        - self_model.narrative_self.consolidation_rate: Identity update rate (default: 0.01)

    Attributes
    ----------
    episodic_memory : deque
        Circular buffer storing significant experiences
    identity_vector : np.ndarray
        64-dimensional identity representation
    goal_hierarchy : list
        Hierarchical goal structure (currently unused)
    memory_capacity : int
        Maximum number of stored episodes
    consolidation_rate : float
        Rate of identity consolidation

    Examples
    --------
    >>> config = {'self_model': {'narrative_self': {'episodic_memory_capacity': 500}}}
    >>> narrative_self = NarrativeSelf(config)
    >>> experience = {'event': 'ignition', 'context': 'high_arousal', 'outcome': 'success'}
    >>> result = narrative_self.update(experience, dt=1.0)
    >>> print(f"Identity strength: {result['identity_strength']:.2f}")
    Identity strength: 0.63
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("self_model", {}).get("narrative_self", {})
        self.memory_capacity = self.config.get("episodic_memory_capacity", 1000)
        self.consolidation_rate = self.config.get("consolidation_rate", 0.01)

        self.episodic_memory: deque[Dict[str, Any]] = deque(maxlen=self.memory_capacity)
        self.identity_vector = np.random.randn(64) * 0.1
        self.goal_hierarchy: list[Any] = []

    def update(self, experience: Dict[str, Any], dt: float = 1.0) -> Dict[str, Any]:
        """Update narrative self with new experience.
        Processes new experiences for potential storage using importance-weighting
        based on surprise magnitude (Free Energy).
        """
        # 1. Importance-weighted selective encoding
        # Use Free Energy as Surprise Magnitude
        surprise = experience.get("free_energy", 0.0)

        # Scaling surprise to a probability
        # High surprise -> High probability of encoding
        # k=1.5, x0=50 can be tuned via config
        threshold = self.config.get("encoding_threshold", 50.0)
        encoding_prob = 1.0 / (1.0 + np.exp(-0.1 * (surprise - threshold)))

        # Always store extreme surprises, otherwise probabilistic
        if surprise > threshold * 2 or np.random.rand() < encoding_prob:
            self.episodic_memory.append(
                {
                    "time": experience.get("time", 0.0),
                    "surprise": surprise,
                    "content": experience.get("beliefs", []),  # Store belief state as memory
                    "type": "significant_event" if surprise > threshold else "routine",
                }
            )

        # 2. Gradual identity consolidation
        # Identity vector rotates towards significant experiences
        if len(self.episodic_memory) > 0 and surprise > threshold:
            # Simplified: Use a portion of the top-level belief to influence identity
            beliefs = experience.get("beliefs", [])
            if beliefs:
                top_belief = beliefs[-1].mean
                # Project or pad to identity dim (64)
                if len(top_belief) >= 64:
                    update_vec = top_belief[:64]
                else:
                    update_vec = np.pad(top_belief, (0, 64 - len(top_belief)))

                # Update identity: slow moving average towards current significant state
                alpha = self.consolidation_rate * dt / 1000.0
                self.identity_vector = (1 - alpha) * self.identity_vector + alpha * update_vec

        # 3. Compute Coherence
        # Coherence is high if current surprise is low, and if identity is stable
        coherence = 1.0 / (1.0 + 0.01 * surprise)

        return {
            "identity_strength": float(np.linalg.norm(self.identity_vector)),
            "memory_count": len(self.episodic_memory),
            "narrative_coherence": float(coherence),
            "encoded_this_step": surprise > threshold or (np.random.rand() < encoding_prob),
        }

    def get_current_state(self) -> Dict[str, Any]:
        """Get current narrative self state."""
        coherence = 1.0 - 0.1 * np.random.rand()  # Simplified

        return {
            "identity_strength": float(np.linalg.norm(self.identity_vector)),
            "memory_count": len(self.episodic_memory),
            "narrative_coherence": float(coherence),
            "identity_vector": self.identity_vector.copy(),
        }

    def reset(self) -> None:
        self.episodic_memory.clear()
        self.identity_vector = np.random.randn(64) * 0.1
