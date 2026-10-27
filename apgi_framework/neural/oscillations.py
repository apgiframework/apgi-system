"""Stub module for oscillation engine.

This module provides type stubs for oscillation engine
that is being migrated from the old apgi_simulation structure.
"""

from typing import Any, Dict, Optional

import numpy as np


class OscillationEngine:
    """Engine for neural oscillations."""

    def __init__(self, config: int | dict = 5):
        """Initialize oscillation engine.

        Parameters
        ----------
        config : int | dict, optional
            Number of frequencies or configuration dict, by default 5
        """
        if isinstance(config, dict):
            num_frequencies = config.get("neural", {}).get("num_frequencies", 5)
        else:
            num_frequencies = config

        self.num_frequencies = num_frequencies
        self.frequencies = np.linspace(1, 50, num_frequencies)
        self.amplitudes = np.zeros(num_frequencies)
        self.phases = np.zeros(num_frequencies)

    def update(self, time_ms: float) -> np.ndarray:
        """Update oscillation state.

        Parameters
        ----------
        time_ms : float
            Current time in milliseconds

        Returns
        -------
        np.ndarray
            Current oscillation values
        """
        self.phases = (self.phases + self.frequencies * time_ms / 1000.0) % (2 * np.pi)
        return self.amplitudes * np.sin(self.phases)  # type: ignore[no-any-return]

    def generate(
        self, time_ms: float = 1.0, modulation: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Generate current oscillation values with optional modulation.

        Parameters
        ----------
        time_ms : float, optional
            Time step in milliseconds, by default 1.0
        modulation : Dict[str, float], optional
            Modulation factors for frequency bands, by default None
        """
        # Apply modulation to amplitudes if provided
        if modulation:
            # Simplified: if 'gamma' in modulation, scale first few frequencies
            gamma_mod = modulation.get("gamma", 1.0)
            self.amplitudes *= gamma_mod

        activity = self.update(time_ms)
        return {
            "activity": activity,
            "spectrum": self.get_spectrum(),
            "band_powers": self.get_band_powers(),
        }

    #: Canonical EEG band edges in Hz (lower inclusive, upper exclusive).
    BANDS: Dict[str, tuple] = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 150.0),
    }

    def get_band_powers(self) -> Dict[str, float]:
        """Return summed squared amplitude of the components inside each EEG band."""
        powers: Dict[str, float] = {}
        for name, (low, high) in self.BANDS.items():
            mask = (self.frequencies >= low) & (self.frequencies < high)
            powers[name] = float(np.sum(self.amplitudes[mask] ** 2))
        return powers

    def set_amplitude(self, frequency_idx: int, amplitude: float) -> None:
        """Set amplitude for a frequency band.

        Parameters
        ----------
        frequency_idx : int
            Index of frequency band
        amplitude : float
            Amplitude value
        """
        if 0 <= frequency_idx < self.num_frequencies:
            self.amplitudes[frequency_idx] = amplitude

    def get_spectrum(self) -> Dict[str, Any]:
        """Get oscillation spectrum.

        Returns
        -------
        Dict[str, Any]
            Spectrum data
        """
        return {
            "frequencies": self.frequencies.tolist(),
            "amplitudes": self.amplitudes.tolist(),
            "phases": self.phases.tolist(),
            "dominant_frequency": float(self.frequencies[np.argmax(self.amplitudes)]),
        }
