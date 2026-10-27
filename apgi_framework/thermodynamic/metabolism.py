"""Metabolic energy budget tracking."""

from typing import Any, Dict


class MetabolicBudget:
    """Tracks ATP consumption and metabolic energy constraints for neural computation.

    Implements a biologically-inspired energy budget system that constrains
    neural computation based on metabolic availability. The brain consumes
    ~20% of the body's total energy, and this constraint shapes cognitive
    processing and consciousness.

    **Energy Components**:
    - **Baseline Consumption**: Continuous metabolic cost of maintaining neural activity
    - **Ignition Cost**: High energy cost of global workspace broadcasting (7.5 ATP units)
    - **Task Overhead**: Additional cost during active cognitive tasks
    - **Recovery**: Gradual replenishment of energy reserves

    **Metabolic Constraints**:
    - Total energy budget limits sustained high-level processing
    - Depletion affects ignition threshold (energy-gated consciousness)
    - Recovery rate determines sustainable cognitive load
    - Depletion threshold triggers metabolic stress responses

    **Biological Basis**:
    - ATP is the universal energy currency for neural computation
    - Action potentials and synaptic transmission are metabolically expensive
    - Consciousness (global ignition) requires coordinated activity across brain regions
    - Energy constraints may explain attention limitations and cognitive fatigue

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing thermodynamic settings:
        - thermodynamic.total_energy_budget: Maximum energy capacity (default: 100.0)
        - thermodynamic.baseline_consumption: Baseline consumption rate per ms (default: 20.0)
        - thermodynamic.ignition_cost: Energy cost per ignition event (default: 7.5)
        - thermodynamic.task_overhead: Additional cost during tasks per ms (default: 27.5)
        - thermodynamic.recovery_rate: Energy recovery rate per ms (default: 5.0)
        - thermodynamic.depletion_threshold: Threshold for energy depletion (default: 10.0)

    Attributes
    ----------
    total_budget : float
        Maximum energy capacity
    current_reserves : float
        Current available energy
    total_consumed : float
        Cumulative energy consumed since reset
    baseline_rate : float
        Continuous consumption rate
    ignition_cost : float
        Energy cost per ignition event
    task_overhead : float
        Additional cost during active tasks
    recovery_rate : float
        Rate of energy replenishment
    depletion_threshold : float
        Threshold below which system is considered depleted

    Examples
    --------
    >>> config = {'thermodynamic': {'total_energy_budget': 150.0, 'ignition_cost': 10.0}}
    >>> budget = MetabolicBudget(config)
    >>> result = budget.update(ignition_occurred=True, task_active=True, dt=1.0)
    >>> print(f"Reserves: {result['reserves']:.1f}, Depleted: {result['depleted']}")
    Reserves: 132.5, Depleted: False
    """

    def __init__(self, config: Dict[str, Any]):
        thermo_config = config.get("thermodynamic", {})
        self.total_budget = thermo_config.get("total_energy_budget", 100.0)
        self.baseline_rate = thermo_config.get("baseline_consumption", 20.0)
        self.ignition_cost = thermo_config.get("ignition_cost", 7.5)
        self.task_overhead = thermo_config.get("task_overhead", 27.5)
        self.recovery_rate = thermo_config.get("recovery_rate", 5.0)
        self.depletion_threshold = thermo_config.get("depletion_threshold", 10.0)

        self.current_reserves = self.total_budget
        self.total_consumed = 0.0
        self.reference_magnitude = thermo_config.get("reference_magnitude", 1.0)

    def update(
        self,
        ignition_occurred: bool,
        task_active: bool,
        dt: float = 1.0,
        broadcast_content: Any = None,
    ) -> Dict[str, Any]:
        """Update metabolic budget based on system activity.

        Computes energy consumption based on current activity and updates
        energy reserves. Includes baseline consumption, ignition costs,
        task overhead, and energy recovery.

        Parameters
        ----------
        ignition_occurred : bool
            Whether an ignition event occurred this timestep
        task_active : bool
            Whether the system is actively engaged in a cognitive task
        dt : float, optional
            Timestep in milliseconds, by default 1.0

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'reserves': Current energy reserves
            - 'consumed': Total energy consumed since reset
            - 'depleted': Boolean indicating if reserves below depletion threshold
            - 'reserve_fraction': Fraction of total budget remaining (0-1)
        """
        # Consumption
        consumption = self.baseline_rate * dt / 1000.0

        if ignition_occurred:
            import numpy as np

            # Scale ignition cost by broadcast content magnitude — larger ignitions cost more
            broadcast_magnitude = (
                np.linalg.norm(broadcast_content) if broadcast_content is not None else 1.0
            )
            normalized_magnitude = np.clip(broadcast_magnitude / self.reference_magnitude, 0.5, 3.0)
            consumption += self.ignition_cost * normalized_magnitude

        if task_active:
            consumption += self.task_overhead * dt / 1000.0

        self.current_reserves -= consumption
        self.total_consumed += consumption

        # Recovery is mathematically only enabled during pauses from task demands
        if not task_active:
            self.current_reserves += self.recovery_rate * dt / 1000.0

        # Clamp reserves to valid range [0, max_capacity]
        self.current_reserves = max(0.0, min(self.current_reserves, self.total_budget))

        depleted = self.current_reserves < self.depletion_threshold

        return {
            "reserves": float(self.current_reserves),
            "consumed": float(self.total_consumed),
            "depleted": depleted,
            "reserve_fraction": float(self.current_reserves / self.total_budget),
        }

    def reset(self) -> None:
        self.current_reserves = self.total_budget
        self.total_consumed = 0.0
