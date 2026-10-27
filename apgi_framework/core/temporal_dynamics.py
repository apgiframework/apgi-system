"""Temporal dynamics for neural oscillations.

Implements phase evolution, Phase-Amplitude Coupling (PAC),
and synchrony metrics for multi-band neural oscillation models.
"""

import math
from typing import Any, Dict, Optional

import numpy as np


class TemporalDynamics:
    """Model of temporal dynamics in neural systems with multi-band oscillations."""

    DEFAULT_BANDS: Dict[str, Dict[str, Any]] = {
        "delta": {"range": [1, 4], "amplitude": 0.5},
        "theta": {"range": [4, 8], "amplitude": 0.7},
        "alpha": {"range": [8, 12], "amplitude": 1.0},
        "beta": {"range": [12, 30], "amplitude": 0.8},
        "gamma": {"range": [30, 80], "amplitude": 0.6},
    }

    def __init__(
        self,
        config: Dict[str, Any],
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize temporal dynamics model.

        Parameters
        ----------
        config : dict
            Configuration dict with 'oscillations' key containing
            'bands' and 'coupling_strength'.
        rng : np.random.Generator, optional
            Random number generator for reproducible initialization.
            If None, uses a private RNG that won't affect global state.
        """
        osc_config: Dict[str, Any] = dict(config.get("oscillations", {}))

        bands_data = osc_config.get("bands", self.DEFAULT_BANDS.copy())
        self.bands: Dict[str, Dict[str, Any]] = dict(bands_data)
        self.coupling_strength: float = osc_config.get("coupling_strength", 0.3)

        self._rng = rng if rng is not None else np.random.default_rng()

        self.time: float = 0.0
        self._initial_rng_state = dict(self._rng.bit_generator.state)

        self.phases: Dict[str, float] = {}
        self._initialize_phases()

    def _initialize_phases(self) -> None:
        """Initialize random phases for all bands in [0, 2π)."""
        for band in self.bands:
            self.phases[band] = float(self._rng.uniform(0, 2 * math.pi))

    def reset(self) -> None:
        """Reset time and reinitialize phases using the stored RNG state."""
        self.time = 0.0
        self._rng.bit_generator.state = dict(self._initial_rng_state)
        self._initialize_phases()

    def _compute_synchrony(self, amplitudes: Dict[str, float]) -> float:
        """Compute synchrony metric from band amplitudes.

        Returns the ratio of max amplitude to sum of amplitudes.
        Returns 0.0 for empty amplitudes.

        Parameters
        ----------
        amplitudes : dict
            Dict mapping band names to amplitude values.

        Returns
        -------
        float
            Synchrony in [0, 1], or 0.0 if amplitudes is empty.
        """
        if not amplitudes:
            return 0.0
        total = sum(amplitudes.values())
        if total == 0:
            return 0.0
        return max(amplitudes.values()) / total

    def _compute_amplitudes(self, pac_factor: float) -> Dict[str, float]:
        """Compute amplitudes for all bands with PAC modulation.

        Parameters
        ----------
        pac_factor : float
            Phase-amplitude coupling factor.

        Returns
        -------
        dict
            Dict mapping band names to amplitude values.
        """
        amplitudes: Dict[str, float] = {}
        for band, params in self.bands.items():
            base_amp = params["amplitude"]
            amplitudes[band] = base_amp * pac_factor
        return amplitudes

    def get_gain_modulation(self, band: str) -> float:
        """Get gain modulation for a specific band.

        Gain = 1 + 0.5 * sin(phase), which is in [0.5, 1.5].
        Returns 1.0 for unknown bands.

        Parameters
        ----------
        band : str
            Band name.

        Returns
        -------
        float
            Gain modulation factor.
        """
        phase = self.phases.get(band, 0.0)
        return 1.0 + 0.5 * math.sin(phase)

    def update(self, dt: float) -> Dict[str, Any]:
        """Update dynamics by time step dt.

        Advances time, evolves phases (dφ/dt = 2πf), computes
        PAC factor, amplitudes, and synchrony.

        Parameters
        ----------
        dt : float
            Time step in seconds.

        Returns
        -------
        dict
            Result with keys: 'phases', 'amplitudes', 'pac_factor', 'synchrony'.
        """
        self.time += dt

        for band, params in self.bands.items():
            freq = (params["range"][0] + params["range"][1]) / 2.0
            self.phases[band] = (self.phases[band] + 2 * math.pi * freq * dt) % (2 * math.pi)

        theta_phase = self.phases.get("theta", 0.0)
        pac_factor = 1.0 + self.coupling_strength * math.sin(theta_phase)

        amplitudes = self._compute_amplitudes(pac_factor)
        synchrony = self._compute_synchrony(amplitudes)

        return {
            "phases": self.phases.copy(),
            "amplitudes": amplitudes,
            "pac_factor": pac_factor,
            "synchrony": synchrony,
        }
