"""State Classifier for Emergent Psychological States.

Enables psychological states to be emergent properties of system dynamics
by classifying current parameters into the closest archetypal state.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class StateClassifier:
    """Classifies the current dynamical state of the APGI system into
    psychological categories based on information-theoretic and
    physiological parameters.
    """

    def __init__(self, state_library: Union[Any, Dict[str, Any]]):
        """Initialize the classifier with a library of archetypal states.

        Args:
            state_library: Instance of APGIStateLibrary or Dict[str, PsychologicalState]
        """
        self.state_profiles: Dict[str, Dict[str, float]] = {}
        if isinstance(state_library, dict):
            self._build_from_dict(state_library)
        else:
            self.library = state_library
            self._build_profiles()

    def _build_from_dict(self, state_dict: Dict[str, Any]) -> None:
        """Extract archetypal parameters from a dictionary."""
        for name, state in state_dict.items():
            # Handle both object and dict-like states
            if hasattr(state, "Pi_e_actual"):  # APGI-Equations style
                self.state_profiles[name] = {
                    "Pi_e": state.Pi_e_actual,
                    "Pi_i": state.Pi_i_eff_actual,
                    "M": state.M_ca,
                    "A": getattr(state, "arousal_level", 0.5),
                    "theta": state.theta_t,
                }
            elif hasattr(state, "Pi_e"):  # Psychological-States-GUI style
                self.state_profiles[name] = {
                    "Pi_e": state.Pi_e,
                    "Pi_i": getattr(state, "Pi_i_eff", 1.0),
                    "M": state.M_ca,
                    "A": 0.5,  # Default arousal if not in params
                    "theta": state.theta_t,
                }
            elif isinstance(state, dict):
                self.state_profiles[name] = state

    def _build_profiles(self) -> None:
        """Extract archetypal parameters from library object."""
        for name, state in self.library.states.items():
            # Handle both object and dict-like states
            if hasattr(state, "Pi_e_actual"):  # APGI-Equations style
                self.state_profiles[name] = {
                    "Pi_e": state.Pi_e_actual,
                    "Pi_i": state.Pi_i_eff_actual,
                    "M": state.M_ca,
                    "A": getattr(state, "arousal_level", 0.5),
                    "theta": state.theta_t,
                }
            elif hasattr(state, "Pi_e"):  # Psychological-States-GUI style
                self.state_profiles[name] = {
                    "Pi_e": state.Pi_e,
                    "Pi_i": getattr(state, "Pi_i_eff", 1.0),
                    "M": state.M_ca,
                    "A": 0.5,  # Default arousal if not in params
                    "theta": state.theta_t,
                }
            elif isinstance(state, dict):
                self.state_profiles[name] = state

    def classify(self, current_params: Dict[str, float]) -> Tuple[str, float]:
        """Identify the closest psychological state.

        Args:
            current_params: Dictionary with keys 'Pi_e', 'Pi_i', 'M', 'A', 'theta'

        Returns:
            Tuple of (state_name, distance_score)
        """
        best_state = "unelaborated"
        min_dist = float("inf")

        # Normalize current params
        # (This is simplified; real implementation might use weighted Euclidean or Mahalanobis)
        p = np.array(
            [
                current_params.get("Pi_e", 1.0),
                current_params.get("Pi_i", 1.0),
                current_params.get("M", 0.0),
                current_params.get("A", 0.5),
                current_params.get("theta", 1.0),
            ]
        )

        for name, profile in self.state_profiles.items():
            target = np.array(
                [profile["Pi_e"], profile["Pi_i"], profile["M"], profile["A"], profile["theta"]]
            )

            # Weighted Euclidean distance
            # Precision parameters have higher weight as they define the 'flavor' of perception
            weights = np.array([1.5, 1.5, 1.0, 1.0, 1.2])
            dist = np.sqrt(np.sum(weights * (p - target) ** 2))

            if dist < min_dist:
                min_dist = dist
                best_state = name

        return best_state, min_dist

    def get_state_details(self, name: str) -> Optional[Any]:
        """Return the PsychologicalState object for the given name."""
        return self.library.states.get(name)
