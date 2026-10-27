"""Coherence maintenance between minimal and narrative self."""

from typing import Any, Dict


class CoherenceMaintenance:
    """Maintains coherence and unity between minimal and narrative self-models.

    Integrates the moment-to-moment minimal self with the long-term narrative
    self to maintain a unified sense of personal identity. This component
    ensures that immediate bodily self-awareness remains consistent with
    autobiographical self-knowledge.

    **Integration Functions**:
    - Monitors consistency between minimal and narrative self-representations
    - Detects conflicts between immediate experience and long-term identity
    - Maintains overall phenomenal unity of self-experience
    - Provides integrated coherence metrics for the complete self-model

    **Coherence Computation**:
    - Weighted combination: 60% minimal coherence + 40% narrative coherence
    - Reflects greater weight of immediate experience in moment-to-moment awareness
    - Phenomenal unity threshold: coherence > 0.7 indicates unified self-experience

    **Clinical Relevance**:
    - Low coherence may indicate dissociative states
    - Conflicts between minimal and narrative self can signal identity disturbances
    - Coherence tracking enables detection of self-related pathology

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary (currently unused but maintained for consistency)

    Attributes
    ----------
    coherence_score : float
        Current overall coherence between minimal and narrative self (0-1)

    Examples
    --------
    >>> coherence = CoherenceMaintenance({})
    >>> minimal_info = {'coherence': 0.8, 'agency': 0.7}
    >>> narrative_info = {'narrative_coherence': 0.9, 'identity_strength': 0.6}
    >>> result = coherence.update(minimal_info, narrative_info)
    >>> print(f"Phenomenal unity: {result['phenomenal_unity']}")
    Phenomenal unity: True
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.coherence_score = 1.0

    def update(
        self, minimal_info: Dict[str, Any], narrative_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update coherence between minimal and narrative self-models.

        Computes integrated coherence by combining coherence measures from
        both minimal and narrative self-components. Uses weighted averaging
        to reflect the relative importance of immediate vs. long-term
        self-experience.

        Parameters
        ----------
        minimal_info : Dict[str, Any]
            Information from minimal self, must contain 'coherence' key
        narrative_info : Dict[str, Any]
            Information from narrative self, must contain 'narrative_coherence' key

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'overall_coherence': Integrated coherence score (0-1)
            - 'phenomenal_unity': Boolean indicating unified self-experience (coherence > 0.7)
        """
        # Check consistency
        minimal_coherence = minimal_info.get("coherence", 1.0)
        narrative_coherence = narrative_info.get("narrative_coherence", 1.0)

        self.coherence_score = 0.6 * minimal_coherence + 0.4 * narrative_coherence

        return {
            "overall_coherence": float(self.coherence_score),
            "phenomenal_unity": self.coherence_score > 0.7,
        }

    def get_coherence_metrics(self) -> Dict[str, Any]:
        """Get current coherence metrics."""
        return {
            "overall_coherence": float(self.coherence_score),
            "phenomenal_unity": self.coherence_score > 0.7,
        }

    def reset(self) -> None:
        self.coherence_score = 1.0
