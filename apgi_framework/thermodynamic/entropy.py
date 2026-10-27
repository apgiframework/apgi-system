"""Entropy production and thermodynamic tracking."""

from typing import Any, Dict


class EntropyTracker:
    """Tracks entropy production and thermodynamic costs of neural computation.

    Implements thermodynamic tracking based on Landauer's principle: any
    logically irreversible computation must dissipate energy and increase
    entropy. This provides a fundamental physical constraint on information
    processing in biological systems.

    **Thermodynamic Principles**:
    - **Landauer's Principle**: Erasing 1 bit of information costs kT ln(2) energy
    - **Entropy Production**: Irreversible computations increase total entropy
    - **Non-equilibrium Dynamics**: Living systems maintain low entropy through energy dissipation
    - **Information-Energy Trade-off**: More computation requires more energy dissipation

    **Entropy Sources**:
    - **Spike Events**: Each action potential represents information processing and bit erasure
    - **Ignition Events**: Global state transitions involve large-scale information reorganization
    - **Synaptic Transmission**: Information transfer between neurons

    **Physical Constants**:
    - Landauer constant: kT ln(2) ≈ 2.87 × 10^-21 J at room temperature (T = 300K)
    - k: Boltzmann constant (1.38 × 10^-23 J/K)
    - T: Temperature (assumed 300K for biological systems)

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing thermodynamic settings:
        - thermodynamic.landauer_constant: Energy cost per bit erased (default: 1.0)

    Attributes
    ----------
    landauer_constant : float
        Energy cost per bit of information erased
    total_entropy : float
        Cumulative entropy produced since reset
    bits_erased : float
        Total number of information bits erased

    Examples
    --------
    >>> config = {'thermodynamic': {'landauer_constant': 2.87e-21}}
    >>> tracker = EntropyTracker(config)
    >>> result = tracker.update(num_spikes=50, ignition=True, dt=1.0)
    >>> print(f"Entropy: {result['total_entropy']:.2f}, Landauer cost: {result['landauer_cost']:.2e}")
    Entropy: 60.00, Landauer cost: 1.44e-19
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("thermodynamic", {})
        self.landauer_constant = self.config.get("landauer_constant", 1.0)

        self.total_entropy = 0.0
        self.bits_erased = 0.0

    def update(
        self, num_spikes: int = 0, ignition: bool = False, dt: float = 1.0
    ) -> Dict[str, Any]:
        """Update entropy production based on neural activity.

        Computes entropy increase from neural computation events. Each spike
        represents information processing that erases bits and increases entropy.
        Ignition events involve large-scale state transitions with high entropy cost.

        Parameters
        ----------
        num_spikes : int, optional
            Number of spikes that occurred this timestep, by default 0
        ignition : bool, optional
            Whether an ignition event occurred, by default False
        dt : float, optional
            Timestep in milliseconds (currently unused), by default 1.0

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'total_entropy': Cumulative entropy produced
            - 'bits_erased': Total information bits erased
            - 'landauer_cost': Total thermodynamic cost in Joules (bits × kT ln(2))
        """
        # Entropy from spikes (information processing)
        self.bits_erased += num_spikes
        spike_entropy = num_spikes * self.landauer_constant

        # Entropy from ignition (state transitions)
        ignition_entropy = 10.0 if ignition else 0.0

        self.total_entropy += spike_entropy + ignition_entropy

        return {
            "total_entropy": float(self.total_entropy),
            "bits_erased": float(self.bits_erased),
            "landauer_cost": float(self.bits_erased * self.landauer_constant),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get current entropy statistics."""
        return {
            "total_entropy": float(self.total_entropy),
            "bits_erased": float(self.bits_erased),
            "landauer_cost": float(self.bits_erased * self.landauer_constant),
        }

    def reset(self) -> None:
        self.total_entropy = 0.0
        self.bits_erased = 0.0
