"""Minimal Self - moment-to-moment embodied self."""

from typing import Any, Dict

import numpy as np


class MinimalSelf:
    """Minimal self-model based on interoceptive processing and embodied awareness.

    Implements the minimal self - the most basic form of self-awareness that
    emerges from interoceptive processing and bodily self-recognition. This
    represents the pre-reflective, moment-to-moment sense of being an
    embodied agent distinct from the environment.

    **Core Components**:
    - **Body Schema**: Integrated representation of current body state
    - **Temporal Continuity**: Sense of persistence through time
    - **Sense of Agency**: Feeling of control over actions and their consequences
    - **Coherence**: Overall integration of self-related information

    **Key Functions**:
    - Integrates interoceptive signals into coherent body representation
    - Tracks prediction accuracy to maintain sense of agency
    - Monitors temporal continuity of self-experience
    - Detects disruptions that may lead to depersonalization

    **Pathological States**:
    - **Depersonalization**: When coherence drops below threshold (< 0.4)
    - **Derealization**: Disruption of temporal continuity
    - **Agency Loss**: Poor prediction accuracy affecting sense of control

    The minimal self updates at a slower rate (50ms by default) than
    basic sensory processing, reflecting the integrative nature of
    self-awareness.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing minimal self settings:
        - self_model.minimal_self.update_rate_ms: Update rate in milliseconds (default: 50)
        - self_model.minimal_self.coherence_threshold: Coherence threshold (default: 0.7)
        - self_model.minimal_self.depersonalization_threshold: Threshold for depersonalization (default: 0.4)

    Attributes
    ----------
    body_schema : np.ndarray
        Integrated body state representation (6D vector)
    temporal_continuity : float
        Measure of temporal coherence (0-1)
    sense_of_agency : float
        Measure of perceived control (0-1)
    update_counter : float
        Counter for update timing
    update_rate_ms : float
        Update rate in milliseconds
    coherence_threshold : float
        Threshold for normal coherence

    Examples
    --------
    >>> config = {'self_model': {'minimal_self': {'update_rate_ms': 100}}}
    >>> minimal_self = MinimalSelf(config)
    >>> body_state = np.array([70, 15, 37.0, 5.0, 10, 120])  # HR, RR, temp, glucose, cortisol, BP
    >>> result = minimal_self.update(body_state, prediction_accuracy=0.8, dt=1.0)
    >>> print(f"Coherence: {result['coherence']:.2f}")
    Coherence: 0.80
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("self_model", {}).get("minimal_self", {})
        self.update_rate_ms = self.config.get("update_rate_ms", 50)
        self.coherence_threshold = self.config.get("coherence_threshold", 0.7)

        self.body_schema = np.zeros(6)  # Body state representation
        self.temporal_continuity = 1.0
        self.sense_of_agency = 1.0
        self.update_counter = 0.0

    def update(
        self, body_state: np.ndarray, prediction_accuracy: float, dt: float = 1.0
    ) -> Dict[str, Any]:
        """Update minimal self representation.

        Integrates current body state and prediction accuracy to maintain
        the minimal self-model. Updates occur at the configured rate to
        reflect the slower, integrative nature of self-awareness.

        Parameters
        ----------
        body_state : np.ndarray
            Current body state vector (6D: heart rate, respiration, temperature,
            glucose, cortisol, blood pressure)
        prediction_accuracy : float
            Accuracy of recent predictions (0-1), used to update sense of agency
        dt : float, optional
            Timestep in milliseconds, by default 1.0

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'body_schema': Current integrated body representation
            - 'coherence': Overall self-coherence (0-1)
            - 'agency': Sense of agency (0-1)
            - 'continuity': Temporal continuity (0-1)
            - 'depersonalization': Boolean indicating if coherence is critically low
        """
        self.update_counter += dt

        if self.update_counter >= self.update_rate_ms:
            self.update_counter = 0
            self.body_schema = 0.9 * self.body_schema + 0.1 * body_state
            self.sense_of_agency = 0.9 * self.sense_of_agency + 0.1 * prediction_accuracy

        self.temporal_continuity = prediction_accuracy

        coherence = min(self.sense_of_agency, self.temporal_continuity)
        depersonalization = coherence < self.config.get("depersonalization_threshold", 0.4)

        return {
            "body_schema": self.body_schema.copy(),
            "coherence": float(coherence),
            "agency": float(self.sense_of_agency),
            "continuity": float(self.temporal_continuity),
            "depersonalization": depersonalization,
        }

    def get_current_state(self) -> Dict[str, Any]:
        """Get current minimal self state."""
        coherence = min(self.sense_of_agency, self.temporal_continuity)
        depersonalization = coherence < self.config.get("depersonalization_threshold", 0.4)

        return {
            "body_schema": self.body_schema.copy(),
            "coherence": float(coherence),
            "agency": float(self.sense_of_agency),
            "continuity": float(self.temporal_continuity),
            "depersonalization": depersonalization,
        }

    def reset(self) -> None:
        self.body_schema = np.zeros(6)
        self.temporal_continuity = 1.0
        self.sense_of_agency = 1.0
