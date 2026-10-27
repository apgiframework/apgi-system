"""Allostatic Regulation.

Maintains homeostatic set points and manages allostatic load.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class AllostaticSetPoint:
    """Set point for a physiological variable."""

    name: str
    target: float
    acceptable_range: Tuple[float, float]
    current_deviation: float = 0.0
    load_contribution: float = 0.0


class AllostaticRegulator:
    """Manages allostatic regulation and cumulative load tracking.

    Allostasis refers to achieving stability through change - the process by
    which the body maintains homeostasis through adaptive responses to stressors.
    This class implements allostatic regulation by:

    1. Maintaining homeostatic set points for key physiological variables
    2. Computing deviations from optimal states
    3. Tracking cumulative allostatic load (wear-and-tear from adaptation)
    4. Generating regulatory signals to restore homeostasis

    Allostatic load accumulates when physiological variables remain outside
    their acceptable ranges, representing the cumulative cost of adaptation.

    The regulator tracks four key variables:
    - Heart rate: Index 0
    - Temperature: Index 2
    - Glucose: Index 3
    - Cortisol: Index 4

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing:
        - interoception.allostatic_ranges: Range factors for regulation
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize vectorized allostatic regulator."""
        self.config = config
        self.batch_size = config.get("active_inference", {}).get("batch_size", 1)
        intero_config = config.get("interoception", {})
        ranges_config = intero_config.get("allostatic_ranges", {})

        # Define set points for key variables (4,)
        # heart_rate, temperature, glucose, cortisol
        self.num_vars = 4
        self.targets = np.array([70.0, 37.0, 5.0, 10.0])
        self.lower_range = np.array([65.0, 36.8, 4.5, 8.0])
        self.upper_range = np.array([75.0, 37.2, 5.5, 12.0])
        self.range_widths = self.upper_range - self.lower_range

        # State tracking (B, 4)
        self.current_deviations = np.zeros((self.batch_size, self.num_vars))
        self.total_load = np.zeros(self.batch_size)

        self.load_decay_rate = 0.001
        self.load_accumulation_rate = 0.01

        # Regulation parameters
        self.tight_range_factor = ranges_config.get("tight", 0.1)
        self.moderate_range_factor = ranges_config.get("moderate", 0.2)
        self.wide_range_factor = ranges_config.get("wide", 0.3)

    def update(self, body_state: np.ndarray, dt: float = 1.0) -> Dict[str, np.ndarray]:
        """Update allostasis for batch.

        Args:
            body_state: State array (B, D) where D includes indices 0, 2, 3, 4
            dt: Timestep in ms
        """
        # heart_rate (0), temp (2), gluc (3), cort (4)
        selected_indices = [0, 2, 3, 4]
        current_vars = body_state[:, selected_indices]

        # Calculate deviations (B, 4)
        self.current_deviations = current_vars - self.targets

        # Calculate load contributions (B, 4)
        outside_lower = current_vars < self.lower_range
        outside_upper = current_vars > self.upper_range

        normalized_deviations = np.zeros_like(current_vars)
        load_masks = outside_lower | outside_upper

        # Avoid division by zero, though range_widths > 0 for these variables
        normalized_deviations[load_masks] = (
            np.abs(self.current_deviations[load_masks])
            / np.broadcast_to(self.range_widths, (self.batch_size, self.num_vars))[load_masks]
        )

        # Accumulate load (B,)
        total_normalized_dev = np.sum(normalized_deviations, axis=1)
        # Scale accumulation by dt/1000 to convert to seconds if needed, or just stay in ms
        self.total_load += self.load_accumulation_rate * total_normalized_dev * (dt / 1000.0)

        # Decay load
        self.total_load *= 1.0 - self.load_decay_rate * (dt / 1000.0)
        self.total_load = np.clip(self.total_load, 0.0, 1.0)

        # Regulatory signals (B, 4) - proportional drive
        regulation_signals = -0.1 * self.current_deviations

        return {
            "regulation_signals": regulation_signals,
            "allostatic_load": self.total_load.copy(),
            "total_deviation": np.sum(np.abs(self.current_deviations), axis=1),
            "stability": np.clip(1.0 - np.mean(normalized_deviations, axis=1), 0.0, 1.0),
        }

    def get_allostatic_load(self) -> np.ndarray:
        """Get current load (B,)."""
        return self.total_load.copy()

    def reset(self) -> None:
        """Reset for all agents in batch."""
        self.total_load.fill(0.0)
        self.current_deviations.fill(0.0)
