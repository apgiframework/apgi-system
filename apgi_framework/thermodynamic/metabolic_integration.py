"""Integration module: Virtual Metabolic Layer with APGI Ignition System.

This module provides the bridge between the Virtual Metabolic Layer (using Neural Mass Models
for ATP flux estimation) and the existing APGI ignition and metabolic budget systems.

Key Features:
-------------
1. Dynamic κ estimation during ignition events (replacing static order-of-magnitude values)
2. Real-time ATP flux tracking integrated with Global Workspace broadcasting
3. Biophysically-grounded metabolic cost for glutamate recycling and ion pumping
4. Metabolic budget updates based on Neural Mass Model predictions

Usage:
------
    # Create integrated metabolic system
    metabolic_system = IntegratedMetabolicSystem(config)

    # During simulation loop
    metabolic_system.update(
        ignition_occurred=True,
        ignition_signal=2.5,
        threshold=2.0,
        workspace_content=content_vector,
        dt_ms=1.0
    )

    # Get dynamic κ for thermodynamic calculations
    kappa = metabolic_system.get_current_kappa()

Integration Points:
------------------
- Connects to IgnitionThreshold via metabolic state updates
- Connects to MetabolicBudget via energy consumption tracking
- Connects to GlobalWorkspace via content-based ATP estimation
"""

from typing import Any, Dict, Optional

import numpy as np

from apgi_framework.thermodynamic.metabolism import MetabolicBudget
from apgi_framework.thermodynamic.neural_mass_metabolism import (
    MetabolicCostFactors,
    NeuralMassParameters,
    VirtualMetabolicLayer,
)


class IntegratedMetabolicSystem:
    """Integrated metabolic system combining Virtual Metabolic Layer with APGI budget.

    This class serves as the primary integration point, providing:
    1. Dynamic κ values based on biophysical ATP calculations
    2. Metabolic budget tracking with NMM-based estimates
    3. Ignition cost prediction and actual cost tracking

    The system computes κ using:
        κ = ATP_total / (bits × kT × ln(2))

    where ATP_total is estimated from Neural Mass Model dynamics.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the integrated metabolic system.

        Args:
            config: Configuration dictionary with keys:
                - thermodynamic.total_energy_budget: Maximum energy capacity
                - thermodynamic.baseline_consumption: Baseline consumption rate
                - thermodynamic.ignition_cost: Static ignition cost (fallback)
                - thermodynamic.use_dynamic_kappa: Enable NMM-based κ estimation
                - thermodynamic.workspace_neurons: Number of neurons in workspace
                - neural_mass.*: Neural mass model parameters
            rng: Random number generator
        """
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()

        # Thermodynamic settings
        thermo_config = config.get("thermodynamic", {})
        self.use_dynamic_kappa = thermo_config.get("use_dynamic_kappa", True)
        self.workspace_neurons = thermo_config.get("workspace_neurons", 100_000)
        self.ignition_duration_ms = thermo_config.get("ignition_duration_ms", 300.0)

        # Initialize traditional metabolic budget
        self.metabolic_budget = MetabolicBudget(config)

        # Initialize Virtual Metabolic Layer (if dynamic kappa enabled)
        if self.use_dynamic_kappa:
            neural_params = self._extract_neural_params(config)
            metabolic_factors = self._extract_metabolic_factors(config)
            self.vml = VirtualMetabolicLayer(
                neural_params=neural_params,
                metabolic_factors=metabolic_factors,
                rng=self.rng,
            )
            self.atp_calculator = self.vml.atp_calculator
        else:
            self.vml = None
            self.atp_calculator = None

        # Tracking
        self.ignition_count: int = 0
        self.total_atp_consumed: float = 0.0
        self.kappa_history: list[float] = []
        self.ignition_cost_history: list[Dict[str, Any]] = []
        self.max_history: int = 1000

    def _extract_neural_params(self, config: Dict[str, Any]) -> NeuralMassParameters:
        """Extract neural mass parameters from config."""
        neural_config = config.get("neural_mass", {})

        return NeuralMassParameters(
            excitatory_time_constant_ms=neural_config.get("excitatory_tau_ms", 10.0),
            inhibitory_time_constant_ms=neural_config.get("inhibitory_tau_ms", 20.0),
            synaptic_gain_excitatory=neural_config.get("excitatory_gain", 3.25),
            synaptic_gain_inhibitory=neural_config.get("inhibitory_gain", 22.0),
            connectivity_pyramid_to_pyramid=neural_config.get("connectivity_c1", 135.0),
            num_neurons=self.workspace_neurons,
            num_synapses=neural_config.get("num_synapses", self.workspace_neurons * 100),
        )

    def _extract_metabolic_factors(self, config: Dict[str, Any]) -> MetabolicCostFactors:
        """Extract metabolic cost factors from config."""
        metabolic_config = config.get("metabolic_costs", {})

        return MetabolicCostFactors(
            atp_per_glutamate_cycle=metabolic_config.get("atp_per_vesicle", 20_000.0),
            resting_atp_per_neuron_s=metabolic_config.get("resting_atp_per_neuron", 1.0e9),
            astrocyte_lactate_factor=metabolic_config.get("astrocyte_factor", 1.2),
        )

    def update(
        self,
        ignition_occurred: bool,
        ignition_signal: float = 0.0,
        threshold: float = 2.0,
        workspace_content: Optional[np.ndarray] = None,
        task_active: bool = False,
        dt_ms: float = 1.0,
    ) -> Dict[str, Any]:
        """Update metabolic system state.

        This is the main integration method called during simulation loops.

        Args:
            ignition_occurred: Whether ignition occurred this timestep
            ignition_signal: Accumulated surprise signal S_t
            threshold: Ignition threshold θ_t
            workspace_content: Content being broadcast (for ATP estimation)
            task_active: Whether active cognitive task is ongoing
            dt_ms: Timestep in milliseconds

        Returns:
            Dictionary with metabolic state and κ value
        """
        # Update traditional metabolic budget
        budget_result = self.metabolic_budget.update(
            ignition_occurred=ignition_occurred,
            task_active=task_active,
            dt=dt_ms,
            broadcast_content=workspace_content,
        )

        # Update Virtual Metabolic Layer
        if self.vml is not None:
            # Simulate neural activity during this timestep
            input_drive = self._compute_input_drive(ignition_signal, threshold)
            neural_result = self.vml.simulate_neural_activity(
                input_drive=input_drive,
                duration_ms=dt_ms,
            )

            # If ignition occurred, compute detailed cost
            ignition_cost = None
            if ignition_occurred:
                ignition_cost = self.vml.compute_ignition_cost(
                    ignition_signal=ignition_signal,
                    threshold=threshold,
                    workspace_content=workspace_content,
                    ignition_duration_ms=self.ignition_duration_ms,
                )
                self.ignition_count += 1
                self.kappa_history.append(ignition_cost["kappa_landauer"])
                self.ignition_cost_history.append(ignition_cost)
                self.total_atp_consumed += ignition_cost["atp_total"]

                # Trim history
                if len(self.kappa_history) > self.max_history:
                    self.kappa_history.pop(0)
                    self.ignition_cost_history.pop(0)

        # Compute current κ
        current_kappa = self.get_current_kappa()

        return {
            "budget_reserves": budget_result["reserves"],
            "budget_depleted": budget_result["depleted"],
            "reserve_fraction": budget_result["reserve_fraction"],
            "current_kappa": current_kappa,
            "kappa_is_dynamic": self.use_dynamic_kappa,
            "ignition_count": self.ignition_count,
            "total_atp_consumed": self.total_atp_consumed,
            "neural_activity": neural_result if self.vml else None,
            "last_ignition_cost": ignition_cost if ignition_occurred else None,
        }

    def _compute_input_drive(self, ignition_signal: float, threshold: float) -> float:
        """Compute input drive to neural mass model from ignition signal.

        Scales the signal-to-threshold ratio to appropriate input range.
        """
        # Normalize by threshold
        if threshold > 0:
            normalized_signal = ignition_signal / threshold
        else:
            normalized_signal = ignition_signal

        # Scale to appropriate input range (0.1 - 0.5 for neural mass)
        # Higher signals produce stronger input drive
        input_drive = 0.1 + 0.4 * min(1.0, normalized_signal / 3.0)

        return float(input_drive)

    def get_current_kappa(self, use_recent_average: bool = True) -> float:
        """Get the current κ value for thermodynamic calculations.

        If dynamic kappa is enabled, returns NMM-based estimate.
        Otherwise returns static configuration value.

        Args:
            use_recent_average: Use average of recent κ values for stability

        Returns:
            Current κ value (ATP cost per bit normalized by Landauer limit)
        """
        if self.use_dynamic_kappa and self.vml is not None:
            return self.vml.get_dynamic_kappa(use_recent_average)

        # Return static κ from config (order-of-magnitude fallback)
        thermo_config = self.config.get("thermodynamic", {})
        return thermo_config.get("static_kappa", 1.0e6)

    def compute_ignition_cost_estimate(
        self,
        ignition_signal: float,
        threshold: float,
        workspace_content: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute estimated ATP cost for a hypothetical ignition.

        This allows pre-computation of ignition cost without triggering
        actual state updates.

        Args:
            ignition_signal: Projected surprise signal S_t
            threshold: Ignition threshold θ_t
            workspace_content: Content to be broadcast

        Returns:
            Dictionary with estimated ignition cost and κ
        """
        if self.vml is None:
            # Return fallback estimate based on static ignition cost
            static_cost = self.config.get("thermodynamic", {}).get("ignition_cost", 7.5)
            return {
                "atp_total_estimate": static_cost * 1e12,  # Rough conversion
                "kappa_landauer": 1.0e6,  # Default order-of-magnitude
                "is_estimate": True,
                "method": "static_fallback",
            }

        # Use Virtual Metabolic Layer for detailed estimate
        cost = self.vml.compute_ignition_cost(
            ignition_signal=ignition_signal,
            threshold=threshold,
            workspace_content=workspace_content,
            ignition_duration_ms=self.ignition_duration_ms,
        )

        cost["is_estimate"] = True
        cost["method"] = "neural_mass_model"

        return cost

    def get_metabolic_summary(self) -> Dict[str, Any]:
        """Get comprehensive metabolic system summary."""
        summary: Dict[str, Any] = {
            "dynamic_kappa_enabled": self.use_dynamic_kappa,
            "ignition_count": self.ignition_count,
            "total_atp_consumed": self.total_atp_consumed,
            "budget_state": {
                "reserves": self.metabolic_budget.current_reserves,
                "total_budget": self.metabolic_budget.total_budget,
                "depleted": self.metabolic_budget.current_reserves
                < self.metabolic_budget.depletion_threshold,
            },
        }

        # Add κ statistics
        if self.kappa_history:
            summary["kappa_stats"] = {
                "current": self.get_current_kappa(),
                "mean": float(np.mean(self.kappa_history)),
                "std": float(np.std(self.kappa_history)),
                "min": float(np.min(self.kappa_history)),
                "max": float(np.max(self.kappa_history)),
                "num_samples": len(self.kappa_history),
            }
        else:
            summary["kappa_stats"] = {
                "current": self.get_current_kappa(),
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "num_samples": 0,
            }

        # Add VML state if available
        if self.vml is not None:
            summary["virtual_metabolic_layer"] = self.vml.get_metabolic_state()

        return summary

    def reset(self) -> None:
        """Reset the integrated metabolic system."""
        self.metabolic_budget.reset()
        if self.vml is not None:
            self.vml.reset()
        self.ignition_count = 0
        self.total_atp_consumed = 0.0
        self.kappa_history = []
        self.ignition_cost_history = []


class MetabolicIgnitionAdapter:
    """Adapter class for integrating metabolic tracking with ignition threshold.

    This adapter connects the Virtual Metabolic Layer to the IgnitionThreshold
    class, providing real-time metabolic state updates for threshold modulation.

    Usage:
        threshold = IgnitionThreshold(config)
        metabolic = IntegratedMetabolicSystem(config)
        adapter = MetabolicIgnitionAdapter(threshold, metabolic)

        # During simulation
        adapter.update(ignition_signal, threshold_value, ignition_occurred)
        kappa = adapter.get_kappa_for_cost_update()
    """

    def __init__(
        self,
        ignition_threshold: Any,  # IgnitionThreshold instance
        metabolic_system: IntegratedMetabolicSystem,
    ):
        """Initialize the adapter."""
        self.ignition_threshold = ignition_threshold
        self.metabolic_system = metabolic_system

        # Recent ignition tracking
        self.last_ignition_cost: Optional[Dict[str, Any]] = None
        self.recent_ignition_costs: list[float] = []
        self.max_recent_costs = 10

    def update(
        self,
        ignition_signal: float,
        threshold: float,
        ignition_occurred: bool,
        workspace_content: Optional[np.ndarray] = None,
        dt_ms: float = 1.0,
    ) -> Dict[str, Any]:
        """Update both ignition threshold and metabolic system.

        This method coordinates the two systems, using metabolic state
        for threshold modulation and ignition events for metabolic cost tracking.

        Args:
            ignition_signal: Current accumulated signal S_t
            threshold: Current ignition threshold θ_t
            ignition_occurred: Whether ignition occurred
            workspace_content: Content being broadcast
            dt_ms: Timestep in milliseconds

        Returns:
            Dictionary with updated states from both systems
        """
        # Update metabolic system
        metabolic_result = self.metabolic_system.update(
            ignition_occurred=ignition_occurred,
            ignition_signal=ignition_signal,
            threshold=threshold,
            workspace_content=workspace_content,
            task_active=True,  # Assume active during ignition
            dt_ms=dt_ms,
        )

        # Get metabolic reserves for threshold modulation
        reserves = metabolic_result["reserve_fraction"]
        allostatic_load = 1.0 - reserves  # Inverse of reserves = allostatic load

        # Update ignition threshold with metabolic state
        # This updates the vectorized metabolic state in IgnitionThreshold
        if hasattr(self.ignition_threshold, "update_metabolic_state"):
            # Create arrays for vectorized update (assuming batch_size=1)
            reserves_array = np.array([reserves])
            load_array = np.array([allostatic_load])
            self.ignition_threshold.update_metabolic_state(reserves_array, load_array)

        # Track ignition cost
        if ignition_occurred and metabolic_result.get("last_ignition_cost"):
            cost = metabolic_result["last_ignition_cost"]["atp_total"]
            self.recent_ignition_costs.append(cost)
            if len(self.recent_ignition_costs) > self.max_recent_costs:
                self.recent_ignition_costs.pop(0)
            self.last_ignition_cost = metabolic_result["last_ignition_cost"]

        return {
            "metabolic": metabolic_result,
            "threshold": {
                "current_threshold": threshold,
                "modulated_by_reserves": reserves,
            },
        }

    def get_kappa_for_cost_update(self) -> float:
        """Get current κ value for metabolic cost calculations.

        Returns:
            Current κ (dynamic or static based on configuration)
        """
        return self.metabolic_system.get_current_kappa()

    def get_average_ignition_cost(self) -> float:
        """Get average ATP cost of recent ignition events."""
        if not self.recent_ignition_costs:
            return 0.0
        return float(np.mean(self.recent_ignition_costs))

    def estimate_future_cost(
        self,
        projected_signal: float,
        threshold: float,
        workspace_content: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Estimate cost of a potential future ignition.

        Args:
            projected_signal: Projected surprise signal
            threshold: Expected threshold
            workspace_content: Projected broadcast content

        Returns:
            Cost estimate dictionary
        """
        return self.metabolic_system.compute_ignition_cost_estimate(
            ignition_signal=projected_signal,
            threshold=threshold,
            workspace_content=workspace_content,
        )


def create_metabolic_system_with_vml(
    workspace_neurons: int = 100_000,
    use_dynamic_kappa: bool = True,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> IntegratedMetabolicSystem:
    """Factory function to create an integrated metabolic system with Virtual Metabolic Layer.

    Args:
        workspace_neurons: Number of neurons in the global workspace
        use_dynamic_kappa: Enable NMM-based dynamic κ estimation
        config_overrides: Additional configuration overrides

    Returns:
        Configured IntegratedMetabolicSystem
    """
    config: Dict[str, Any] = {
        "thermodynamic": {
            "total_energy_budget": 100.0,
            "baseline_consumption": 20.0,
            "ignition_cost": 7.5,  # Fallback static value
            "task_overhead": 27.5,
            "recovery_rate": 5.0,
            "depletion_threshold": 10.0,
            "use_dynamic_kappa": use_dynamic_kappa,
            "workspace_neurons": workspace_neurons,
            "ignition_duration_ms": 300.0,
        },
        "neural_mass": {
            "excitatory_tau_ms": 10.0,
            "inhibitory_tau_ms": 20.0,
            "excitatory_gain": 3.25,
            "inhibitory_gain": 22.0,
            "connectivity_c1": 135.0,
            "num_synapses": workspace_neurons * 100,
        },
        "metabolic_costs": {
            "atp_per_vesicle": 20_000.0,
            "resting_atp_per_neuron": 1.0e9,
            "astrocyte_factor": 1.2,
        },
    }

    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

    return IntegratedMetabolicSystem(config)
