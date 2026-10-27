"""Virtual Metabolic Layer: Neural Mass Models for ATP Flux Estimation.

This module implements a biophysically-inspired metabolic estimation system using
Neural Mass Models (NMMs) to compute ATP consumption during Global Ignition events.

The Virtual Metabolic Layer addresses the limitation of order-of-magnitude κ estimates
by providing a mechanistic calculation based on:
1. Neural population dynamics (Jansen-Rit inspired neural mass model)
2. Glutamate recycling costs (excitatory neurotransmission)
3. Ion pumping costs (Na+/K+-ATPase, Ca2+-ATPase)
4. Global workspace broadcast energy requirements

Key Equations:
-------------
ATP_flux = ATP_glutamate + ATP_ion_pumping + ATP_baseline

where:
- ATP_glutamate ≈ N_synapses × f_firing × Q_glutamate (ATP per vesicle cycle)
- ATP_ion_pumping = ATP_NaK + ATP_Ca (restoration of ion gradients)
- κ = ATP_flux / (information_bits × kT × ln(2))

References:
----------
- Jansen & Rit (1995): Electroencephalogram and visual evoked potential generation
- Attwell & Laughlin (2001): Energy budget for signaling in the grey matter
- Lennie (2003): The cost of cortical computation
- Howarth et al. (2012): Updated energy budgets for neural computation
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class NeuralMassParameters:
    """Parameters for the Neural Mass Model.

    Based on Jansen-Rit model with modifications for metabolic estimation.
    Default values represent a canonical cortical column (~10^5 neurons).

    Attributes:
        excitatory_time_constant_ms: Membrane time constant for pyramidal cells (ms)
        inhibitory_time_constant_ms: Membrane time constant for interneurons (ms)
        synaptic_gain_excitatory: Gain of excitatory synapses (mV)
        synaptic_gain_inhibitory: Gain of inhibitory synapses (mV)
        connectivity_pyramid_to_pyramid: Recurrent excitation strength
        connectivity_pyramid_to_interneuron: Excitation of interneurons
        connectivity_interneuron_to_pyramid: Inhibition of pyramidal cells
        max_firing_rate: Maximum firing rate (Hz)
        slope_firing_function: Slope of sigmoid firing function (mV^-1)
        baseline_firing_excitatory: Baseline excitatory firing rate (Hz)
        baseline_firing_inhibitory: Baseline inhibitory firing rate (Hz)
        num_synapses: Number of synapses in the neural mass
        num_neurons: Number of neurons in the population
    """

    # Time constants (ms)
    excitatory_time_constant_ms: float = 10.0  # τ_e: Pyramidal cell membrane time constant
    inhibitory_time_constant_ms: float = 20.0  # τ_i: Interneuron membrane time constant

    # Synaptic gains (mV)
    synaptic_gain_excitatory: float = 3.25  # A: Excitatory postsynaptic potential amplitude
    synaptic_gain_inhibitory: float = 22.0  # B: Inhibitory postsynaptic potential amplitude

    # Connectivity constants (dimensionless, represents average synaptic contacts)
    connectivity_pyramid_to_pyramid: float = 135.0  # C1: Recurrent excitation
    connectivity_pyramid_to_interneuron: float = 0.8 * 135.0  # C2: Excitation of interneurons
    connectivity_interneuron_to_pyramid: float = 0.25 * 135.0  # C3: Inhibition of pyramidal cells
    connectivity_input_to_pyramid: float = 0.8 * 135.0  # C4: External input to pyramidal cells
    connectivity_input_to_interneuron: float = 0.25 * 135.0  # C5: External input to interneurons

    # Firing function parameters
    max_firing_rate: float = 5.0  # ν_max: Maximum firing rate (Hz)
    slope_firing_function: float = 0.56  # r: Slope of sigmoid (mV^-1)
    firing_threshold: float = 6.0  # v_0: Firing threshold (mV)

    # Baseline firing rates (Hz)
    baseline_firing_excitatory: float = 5.0
    baseline_firing_inhibitory: float = 10.0

    # Population sizes
    num_synapses: int = 10_000_000  # ~10^7 synapses per cortical column
    num_neurons: int = 100_000  # ~10^5 neurons per cortical column
    excitatory_fraction: float = 0.8  # Fraction of excitatory neurons


@dataclass
class MetabolicCostFactors:
    """ATP cost factors for neural computation (in ATP molecules per event).

    Based on comprehensive energy budget estimates from Attwell & Laughlin (2001)
    and subsequent updates from Howarth et al. (2012).

    Attributes:
        atp_per_glutamate_cycle: ATP to recycle one glutamate vesicle
        atp_per_na_k_pump_cycle: ATP per Na+/K+-ATPase cycle (3 Na+ out, 2 K+ in)
        atp_per_ca_pump_cycle: ATP per Ca2+-ATPase cycle
        na_pump_rate_per_spike: Na+ influx per action potential (10^6 ions)
        ca_pump_rate_per_spike: Ca2+ influx per action potential (10^4 ions)
        resting_atp_per_neuron_s: Baseline ATP consumption per neuron per second
        astrocyte_lactate_factor: Multiplier for astrocyte metabolic support
    """

    # Synaptic transmission costs
    atp_per_glutamate_cycle: float = 20_000.0  # ~20,000 ATP per vesicle recycling
    vesicles_per_spike: float = 1.0  # Average vesicles released per spike

    # Ion pumping costs
    atp_per_na_k_pump_cycle: float = 1.0  # 1 ATP per 3 Na+ out / 2 K+ in
    na_pump_rate_per_spike: float = 1.0  # ~10^6 Na+ per spike (normalized units)
    atp_per_ca_pump_cycle: float = 1.0  # 1 ATP per Ca2+ pumped
    ca_pump_rate_per_spike: float = 0.01  # ~10^4 Ca2+ per spike (normalized units)

    # Baseline metabolism (resting)
    resting_atp_per_neuron_s: float = 1.0e9  # ~10^9 ATP/s per neuron at rest

    # Astrocyte support factor (lactate shuttle)
    astrocyte_lactate_factor: float = 1.2  # 20% additional ATP from astrocytes

    # Action potential propagation cost
    atp_per_ap_propagation: float = 1.0e7  # ~10^7 ATP per action potential

    # Temperature factor (Q10 ≈ 2.3 for metabolic processes)
    q10_factor: float = 2.3
    reference_temp_c: float = 37.0


class NeuralMassModel:
    """Neural Mass Model for estimating population-level neural activity.

    Implements a Jansen-Rit inspired model to estimate firing rates and
    synaptic activity that drive metabolic demand calculations.

    The model tracks three populations:
    1. Pyramidal cells (main excitatory output)
    2. Excitatory interneurons
    3. Inhibitory interneurons

    State variables:
    - y[0], y[1]: Postsynaptic potential (PSP) at pyramidal cells from excitatory inputs
    - y[2], y[3]: PSP at pyramidal cells from inhibitory inputs
    - y[4], y[5]: PSP at excitatory interneurons
    - y[6], y[7]: PSP at inhibitory interneurons

    where y[2i] represents the PSP and y[2i+1] represents its time derivative.
    """

    def __init__(
        self,
        params: Optional[NeuralMassParameters] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the neural mass model."""
        self.params = params or NeuralMassParameters()
        self.rng = rng if rng is not None else np.random.default_rng()

        # Initialize state: [pyramid_exc, d_pyramid_exc, pyramid_inh, d_pyramid_inh,
        #                    interneuron_exc, d_interneuron_exc, interneuron_inh, d_interneuron_inh]
        self.state = np.zeros(8)

        # Firing rate history for metabolic calculations
        self.pyramidal_firing_history: list[float] = []
        self.interneuron_firing_history: list[float] = []
        self.synaptic_activity_history: list[float] = []

        # History limits
        self.max_history = 1000

    def sigmoid(self, v: float) -> float:
        """Sigmoid firing rate function.

        S(v) = 2e_0 / (1 + exp(r(v_0 - v)))

        where:
        - e_0: maximum firing rate
        - r: slope parameter
        - v_0: threshold potential
        """
        p = self.params
        # Clip argument to prevent overflow in exp()
        exp_arg = np.clip(p.slope_firing_function * (p.firing_threshold - v), -500.0, 500.0)
        return (2.0 * p.max_firing_rate) / (1.0 + np.exp(exp_arg))

    def derivatives(self, state: np.ndarray, input_current: float) -> np.ndarray:
        """Compute state derivatives for the neural mass model.

        Args:
            state: Current state vector [8]
            input_current: External input current (simulating thalamic input)

        Returns:
            Derivatives of state variables [8]
        """
        p = self.params
        y = state

        # Compute firing rates from PSPs
        # Pyramidal output drives interneurons
        v_pyramid = y[0] - y[2]  # Net PSP at pyramidal cells
        sp = self.sigmoid(v_pyramid)

        # Inhibitory interneuron output
        vi_inh = self.sigmoid(y[6])

        # Derivatives (Jansen-Rit equations)
        dydt = np.zeros(8)

        # y[0]: PSP from excitatory inputs to pyramidal cells (via C3 from interneurons)
        # Actually C3 is inhibitory, so let's correct the structure

        # Correct Jansen-Rit structure:
        # y[0], y[1]: excitatory PSP at pyramidal (from other pyramidal via C1 + input via C4)
        # y[2], y[3]: inhibitory PSP at pyramidal (from interneurons via C3)
        # y[4], y[5]: excitatory PSP at excitatory interneurons (from pyramidal via C2)
        # y[6], y[7]: inhibitory PSP at inhibitory interneurons (from interneurons)

        # d(PSP_exc_pyramid)/dt
        dydt[0] = y[1]
        # d²(PSP_exc_pyramid)/dt²
        dydt[1] = (
            p.synaptic_gain_excitatory * p.connectivity_pyramid_to_pyramid * sp
            + p.synaptic_gain_excitatory * p.connectivity_input_to_pyramid * input_current
            - 2.0 * y[1]
            - y[0] / p.excitatory_time_constant_ms**2
        )

        # d(PSP_inh_pyramid)/dt
        dydt[2] = y[3]
        # d²(PSP_inh_pyramid)/dt²
        dydt[3] = (
            p.synaptic_gain_inhibitory * p.connectivity_interneuron_to_pyramid * vi_inh
            - 2.0 * y[3]
            - y[2] / p.inhibitory_time_constant_ms**2
        )

        # d(PSP_exc_interneuron)/dt
        dydt[4] = y[5]
        # d²(PSP_exc_interneuron)/dt²
        dydt[5] = (
            p.synaptic_gain_excitatory * p.connectivity_pyramid_to_interneuron * sp
            + p.synaptic_gain_excitatory * p.connectivity_input_to_interneuron * input_current
            - 2.0 * y[5]
            - y[4] / p.excitatory_time_constant_ms**2
        )

        # d(PSP_inh_interneuron)/dt (self-inhibition among interneurons)
        dydt[6] = y[7]
        # d²(PSP_inh_interneuron)/dt²
        dydt[7] = (
            p.synaptic_gain_inhibitory * 0.1 * vi_inh  # Weak self-inhibition
            - 2.0 * y[7]
            - y[6] / p.inhibitory_time_constant_ms**2
        )

        return dydt

    def step(self, input_current: float, dt_ms: float = 1.0) -> Dict[str, float]:
        """Advance the neural mass model by one timestep.

        Args:
            input_current: External input current (arbitrary units, scaled by connectivity)
            dt_ms: Timestep in milliseconds

        Returns:
            Dictionary with firing rates and PSPs
        """
        # Compute derivatives
        dydt = self.derivatives(self.state, input_current)

        # Euler integration (could use RK4 for better stability)
        # For metabolic estimation, Euler with small dt is sufficient
        self.state += dydt * dt_ms

        # Compute outputs
        v_pyramid = self.state[0] - self.state[2]
        v_interneuron = self.state[4]

        pyramidal_firing = self.sigmoid(v_pyramid)
        interneuron_firing = self.sigmoid(v_interneuron)

        # Synaptic activity (proxy for neurotransmitter release)
        synaptic_activity = (
            pyramidal_firing * self.params.connectivity_pyramid_to_pyramid
            + interneuron_firing * self.params.connectivity_interneuron_to_pyramid
        )

        # Store history
        self.pyramidal_firing_history.append(float(pyramidal_firing))
        self.interneuron_firing_history.append(float(interneuron_firing))
        self.synaptic_activity_history.append(float(synaptic_activity))

        # Trim history
        if len(self.pyramidal_firing_history) > self.max_history:
            self.pyramidal_firing_history.pop(0)
            self.interneuron_firing_history.pop(0)
            self.synaptic_activity_history.pop(0)

        return {
            "pyramidal_firing_rate_hz": float(pyramidal_firing),
            "interneuron_firing_rate_hz": float(interneuron_firing),
            "net_pyramidal_potential_mv": float(v_pyramid),
            "synaptic_activity": float(synaptic_activity),
            "excitatory_psp": float(self.state[0]),
            "inhibitory_psp": float(self.state[2]),
        }

    def get_average_firing_rates(self, window_ms: float = 100.0) -> Dict[str, float]:
        """Get average firing rates over recent window."""
        window_steps = int(window_ms)
        if len(self.pyramidal_firing_history) < window_steps:
            window_steps = len(self.pyramidal_firing_history)

        if window_steps == 0:
            return {"pyramidal_hz": 0.0, "interneuron_hz": 0.0}

        return {
            "pyramidal_hz": float(np.mean(self.pyramidal_firing_history[-window_steps:])),
            "interneuron_hz": float(np.mean(self.interneuron_firing_history[-window_steps:])),
        }

    def reset(self) -> None:
        """Reset the neural mass model to baseline state."""
        self.state = np.zeros(8)
        self.pyramidal_firing_history = []
        self.interneuron_firing_history = []
        self.synaptic_activity_history = []


class ATPFluxCalculator:
    """Calculate ATP flux based on neural mass model activity.

    Converts firing rates and synaptic activity into biophysical ATP consumption
    estimates using established energy budgets for neural computation.

    The calculator produces:
    1. Instantaneous ATP flux (ATP/s)
    2. Cumulative ATP consumption for an ignition event
    3. Information-to-energy efficiency (κ value)

    Key Calculations:
    ----------------
    ATP_glutamate = N_synapses × f_release × ATP_per_vesicle

    ATP_NaK = (Na_influx × ATP_per_Na) / pump_efficiency

    ATP_Ca = Ca_influx × ATP_per_Ca

    ATP_baseline = N_neurons × resting_rate

    κ = ATP_total / (bits_processed × kT × ln(2))
    """

    def __init__(
        self,
        metabolic_factors: Optional[MetabolicCostFactors] = None,
        neural_params: Optional[NeuralMassParameters] = None,
    ):
        """Initialize the ATP flux calculator."""
        self.factors = metabolic_factors or MetabolicCostFactors()
        self.neural_params = neural_params or NeuralMassParameters()

        # Physical constants
        self.k_boltzmann_J_per_K = 1.380649e-23  # Boltzmann constant (J/K)
        self.temp_kelvin = 310.15  # Body temperature (~37°C)

        # Tracking
        self.total_atp_consumed: float = 0.0
        self.atp_history: list[Dict[str, float]] = []
        self.max_history = 10000

    def compute_atp_flux(
        self,
        pyramidal_firing_rate_hz: float,
        interneuron_firing_rate_hz: float,
        synaptic_activity: float,
        num_active_neurons: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute ATP flux components from neural activity.

        Args:
            pyramidal_firing_rate_hz: Firing rate of pyramidal cells (Hz)
            interneuron_firing_rate_hz: Firing rate of interneurons (Hz)
            synaptic_activity: Normalized synaptic activity metric
            num_active_neurons: Number of neurons active (defaults to neural_params.num_neurons)

        Returns:
            Dictionary with ATP flux components (ATP per second)
        """
        f = self.factors
        n = num_active_neurons or self.neural_params.num_neurons
        n_excitatory = int(n * self.neural_params.excitatory_fraction)
        n_inhibitory = n - n_excitatory

        # 1. Glutamate recycling cost (synaptic transmission)
        # Based on: ~20,000 ATP per vesicle recycled
        vesicles_released_per_s = pyramidal_firing_rate_hz * f.vesicles_per_spike * n_excitatory
        atp_glutamate_per_s = vesicles_released_per_s * f.atp_per_glutamate_cycle

        # 2. Na+/K+-ATPase cost (restoring ion gradients after spikes)
        # Action potentials cause Na+ influx (~10^6 ions per spike)
        # 3 Na+ pumped per ATP molecule
        na_influx_per_s = (
            pyramidal_firing_rate_hz * f.na_pump_rate_per_spike * n_excitatory
            + interneuron_firing_rate_hz * f.na_pump_rate_per_spike * n_inhibitory
        )
        atp_na_k_pump_per_s = na_influx_per_s * f.atp_per_na_k_pump_cycle / 3.0

        # 3. Ca2+-ATPase cost (restoring calcium gradients)
        # Ca2+ enters during spikes and synaptic activity
        ca_influx_per_s = (
            pyramidal_firing_rate_hz * f.ca_pump_rate_per_spike * n_excitatory
            + synaptic_activity * 1000.0  # Synaptic Ca2+ influx
        )
        atp_ca_pump_per_s = ca_influx_per_s * f.atp_per_ca_pump_cycle

        # 4. Baseline metabolism (resting ATP consumption)
        # ~10^9 ATP/s per neuron at rest (maintenance of gradients, protein synthesis)
        atp_baseline_per_s = n * f.resting_atp_per_neuron_s

        # 5. Action potential propagation cost
        # Includes axonal signaling and dendritic processing
        total_firing_rate = (
            pyramidal_firing_rate_hz * n_excitatory + interneuron_firing_rate_hz * n_inhibitory
        )
        atp_propagation_per_s = total_firing_rate * f.atp_per_ap_propagation

        # Total ATP flux
        atp_total_per_s = (
            atp_glutamate_per_s
            + atp_na_k_pump_per_s
            + atp_ca_pump_per_s
            + atp_baseline_per_s
            + atp_propagation_per_s
        )

        # Apply astrocyte support factor (lactate shuttle increases efficiency)
        atp_total_per_s *= f.astrocyte_lactate_factor

        # Temperature correction (Q10 effect)
        # ATP production/consumption is temperature-dependent
        temp_factor = self._temperature_factor()
        atp_total_per_s *= temp_factor

        result = {
            "atp_glutamate_per_s": float(atp_glutamate_per_s),
            "atp_na_k_pump_per_s": float(atp_na_k_pump_per_s),
            "atp_ca_pump_per_s": float(atp_ca_pump_per_s),
            "atp_baseline_per_s": float(atp_baseline_per_s),
            "atp_propagation_per_s": float(atp_propagation_per_s),
            "atp_total_per_s": float(atp_total_per_s),
            "vesicles_released_per_s": float(vesicles_released_per_s),
            "na_influx_per_s": float(na_influx_per_s),
            "total_firing_rate_hz": float(total_firing_rate),
        }

        # Track history
        self.atp_history.append(result)
        self.total_atp_consumed += atp_total_per_s * 0.001  # Assume 1ms steps

        if len(self.atp_history) > self.max_history:
            self.atp_history.pop(0)

        return result

    def _temperature_factor(self) -> float:
        """Compute temperature-dependent metabolic factor."""
        # Simplified: assume physiological temperature (37°C)
        # Q10 ≈ 2.3 means rate doubles every 10°C
        return 1.0  # At 37°C reference

    def compute_kappa(
        self,
        atp_total: float,
        bits_processed: float,
        consider_landauer: bool = True,
    ) -> float:
        """Compute the κ (kappa) value: metabolic cost per bit of information.

        κ = ATP_total / (bits × kT × ln(2))

        This represents the thermodynamic efficiency of information processing.
        The Landauer limit (kT ln 2) is the theoretical minimum energy per bit.

        Args:
            atp_total: Total ATP molecules consumed
            bits_processed: Information content in bits
            consider_landauer: If True, normalize by Landauer limit

        Returns:
            κ value (dimensionless if normalized by Landauer, ATP/bit otherwise)
        """
        if bits_processed <= 0:
            return float("inf")

        atp_per_bit = atp_total / bits_processed

        if consider_landauer:
            # Landauer limit: E_min = kT ln(2) ≈ 0.018 eV at 37°C
            # ATP hydrolysis: ~20 kJ/mol = ~0.32 eV per ATP
            energy_per_atp_joules = 20_000 / 6.022e23  # ~3.3e-20 J per ATP
            energy_per_bit_landauer = self.k_boltzmann_J_per_K * self.temp_kelvin * np.log(2)

            # Energy consumed in Joules
            energy_consumed = atp_total * energy_per_atp_joules

            # κ = E_consumed / E_landauer
            kappa = energy_consumed / (bits_processed * energy_per_bit_landauer)
            return float(kappa)

        return float(atp_per_bit)

    def compute_ignition_cost(
        self,
        ignition_duration_ms: float = 300.0,
        workspace_size_neurons: int = 100_000,
        broadcast_amplitude: float = 1.0,
    ) -> Dict[str, float]:
        """Compute the total ATP cost for a Global Ignition event.

        This calculates the metabolic cost of the global workspace broadcast,
        including heightened neural activity, increased synaptic transmission,
        and sustained ion pumping during the ignition event.

        Args:
            ignition_duration_ms: Duration of the ignition event (ms)
            workspace_size_neurons: Number of neurons involved in global broadcast
            broadcast_amplitude: Amplitude of the broadcast signal (0-1 scale)

        Returns:
            Dictionary with ignition cost breakdown
        """
        # Ignition involves heightened activity in frontoparietal regions
        # Estimate elevated firing rates during ignition
        baseline_firing = 5.0  # Hz
        ignition_firing = baseline_firing + 15.0 * broadcast_amplitude  # Up to 20 Hz

        # Average firing rate during ignition
        avg_firing_rate = (baseline_firing + ignition_firing) / 2.0

        # Duration in seconds
        duration_s = ignition_duration_ms / 1000.0

        # Estimate vesicle release during ignition
        vesicles_per_neuron = avg_firing_rate * duration_s * self.factors.vesicles_per_spike
        total_vesicles = vesicles_per_neuron * workspace_size_neurons

        # ATP for glutamate recycling during ignition
        atp_glutamate = total_vesicles * self.factors.atp_per_glutamate_cycle

        # ATP for ion pumping (elevated due to increased firing)
        na_influx = (
            avg_firing_rate
            * self.factors.na_pump_rate_per_spike
            * workspace_size_neurons
            * duration_s
        )
        atp_na_k = na_influx * self.factors.atp_per_na_k_pump_cycle / 3.0

        ca_influx = (
            avg_firing_rate
            * self.factors.ca_pump_rate_per_spike
            * workspace_size_neurons
            * duration_s
        )
        atp_ca = ca_influx * self.factors.atp_per_ca_pump_cycle

        # Baseline consumption during ignition period
        atp_baseline = workspace_size_neurons * self.factors.resting_atp_per_neuron_s * duration_s

        # Total ignition cost
        atp_total = atp_glutamate + atp_na_k + atp_ca + atp_baseline

        # Apply astrocyte factor
        atp_total *= self.factors.astrocyte_lactate_factor

        # Estimate information content (bits)
        # Global ignition broadcasts content with dimensionality ~256 (typical workspace)
        # Information content depends on broadcast amplitude
        workspace_dimensionality = 256.0
        # Effective bits = workspace_dim × amplitude (simplified)
        bits_broadcast = workspace_dimensionality * broadcast_amplitude

        # Compute κ for this ignition event
        kappa = self.compute_kappa(atp_total, bits_broadcast, consider_landauer=True)

        return {
            "atp_total": float(atp_total),
            "atp_glutamate": float(atp_glutamate),
            "atp_na_k_pump": float(atp_na_k),
            "atp_ca_pump": float(atp_ca),
            "atp_baseline": float(atp_baseline),
            "duration_ms": float(ignition_duration_ms),
            "workspace_neurons": int(workspace_size_neurons),
            "vesicles_released": float(total_vesicles),
            "bits_broadcast": float(bits_broadcast),
            "kappa_landauer": float(kappa),
            "atp_per_bit": float(atp_total / bits_broadcast) if bits_broadcast > 0 else 0.0,
        }

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics of ATP consumption."""
        if not self.atp_history:
            return {"total_atp_consumed": 0.0, "mean_atp_per_s": 0.0}

        atp_totals = [h["atp_total_per_s"] for h in self.atp_history]

        return {
            "total_atp_consumed": float(self.total_atp_consumed),
            "mean_atp_per_s": float(np.mean(atp_totals)),
            "max_atp_per_s": float(np.max(atp_totals)),
            "min_atp_per_s": float(np.min(atp_totals)),
        }

    def reset(self) -> None:
        """Reset tracking variables."""
        self.total_atp_consumed = 0.0
        self.atp_history = []


class VirtualMetabolicLayer:
    """Virtual Metabolic Layer integrating Neural Mass Models with ATP flux estimation.

    This is the main interface that combines:
    1. Neural Mass Model for population-level neural dynamics
    2. ATP Flux Calculator for metabolic cost estimation
    3. Integration with APGI ignition system for dynamic κ value generation

    The layer provides accurate metabolic cost estimates for ignition events,
    replacing order-of-magnitude approximations with biophysically-grounded calculations.

    Usage:
        layer = VirtualMetabolicLayer()
        layer.simulate_pre_ignition(input_drive=0.5, duration_ms=100)
        ignition_cost = layer.compute_ignition_cost(
            ignition_signal=2.5,
            threshold=2.0,
            workspace_content=np.random.randn(256)
        )
        kappa = ignition_cost["kappa_landauer"]
    """

    def __init__(
        self,
        neural_params: Optional[NeuralMassParameters] = None,
        metabolic_factors: Optional[MetabolicCostFactors] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the Virtual Metabolic Layer.

        Args:
            neural_params: Neural mass model parameters
            metabolic_factors: Metabolic cost factors
            rng: Random number generator for stochastic elements
        """
        self.neural_model = NeuralMassModel(neural_params, rng)
        self.atp_calculator = ATPFluxCalculator(metabolic_factors, neural_params)

        # Integration state
        self.current_input_drive: float = 0.0
        self.accumulated_atp: float = 0.0
        self.ignition_event_count: int = 0

        # Kappa tracking for dynamic updates
        self.kappa_history: list[float] = []
        self.max_kappa_history = 100

    def simulate_neural_activity(
        self,
        input_drive: float,
        duration_ms: float,
        dt_ms: float = 1.0,
    ) -> Dict[str, Any]:
        """Simulate neural mass activity and compute ATP consumption.

        Args:
            input_drive: Input current to the neural mass (0-1 scale)
            duration_ms: Simulation duration in milliseconds
            dt_ms: Timestep in milliseconds

        Returns:
            Dictionary with neural activity and ATP consumption data
        """
        num_steps = int(duration_ms / dt_ms)
        atp_components_sum = {
            "atp_glutamate_per_s": 0.0,
            "atp_na_k_pump_per_s": 0.0,
            "atp_ca_pump_per_s": 0.0,
            "atp_baseline_per_s": 0.0,
            "atp_propagation_per_s": 0.0,
        }

        # Run simulation
        neural_data = []
        for _ in range(num_steps):
            # Scale input drive to appropriate range for neural mass
            scaled_input = input_drive * 0.1  # Scale to mV range
            neural_state = self.neural_model.step(scaled_input, dt_ms)
            neural_data.append(neural_state)

            # Compute ATP flux
            atp_components = self.atp_calculator.compute_atp_flux(
                pyramidal_firing_rate_hz=neural_state["pyramidal_firing_rate_hz"],
                interneuron_firing_rate_hz=neural_state["interneuron_firing_rate_hz"],
                synaptic_activity=neural_state["synaptic_activity"],
            )

            # Accumulate ATP components
            for key in atp_components_sum:
                if key in atp_components:
                    atp_components_sum[key] += atp_components[key] * dt_ms / 1000.0

        # Average firing rates over simulation
        avg_firing = self.neural_model.get_average_firing_rates(duration_ms)

        # Total ATP consumed
        total_atp = sum(atp_components_sum.values())
        self.accumulated_atp += total_atp

        return {
            "duration_ms": duration_ms,
            "input_drive": input_drive,
            "average_pyramidal_firing_hz": avg_firing["pyramidal_hz"],
            "average_interneuron_firing_hz": avg_firing["interneuron_hz"],
            "total_atp_consumed": total_atp,
            "atp_breakdown": atp_components_sum,
            "final_neural_state": neural_data[-1] if neural_data else None,
        }

    def compute_ignition_cost(
        self,
        ignition_signal: float,
        threshold: float,
        workspace_content: Optional[np.ndarray] = None,
        ignition_duration_ms: float = 300.0,
    ) -> Dict[str, Any]:
        """Compute the metabolic cost of a Global Ignition event.

        This method provides the key functionality for dynamic κ estimation.
        It calculates the ATP cost and computes κ based on the Landauer limit.

        Args:
            ignition_signal: Accumulated surprise signal S_t
            threshold: Ignition threshold θ_t
            workspace_content: Content being broadcast (for information estimation)
            ignition_duration_ms: Duration of the ignition event

        Returns:
            Dictionary with ignition cost breakdown and κ value
        """
        # Compute signal amplitude relative to threshold
        signal_excess = max(0.0, ignition_signal - threshold)
        broadcast_amplitude = min(1.0, signal_excess / threshold) if threshold > 0 else 0.0

        # Determine workspace content information
        if workspace_content is not None:
            # Estimate information content from content magnitude
            content_magnitude = float(np.linalg.norm(workspace_content))
            # Normalize to reasonable bit estimate
            bits_content = min(256.0, content_magnitude)
        else:
            bits_content = 128.0  # Default: half of typical workspace capacity

        # Use ATP calculator to compute ignition cost
        ignition_cost = self.atp_calculator.compute_ignition_cost(
            ignition_duration_ms=ignition_duration_ms,
            workspace_size_neurons=self.neural_model.params.num_neurons,
            broadcast_amplitude=broadcast_amplitude,
        )

        # Override bits estimate if we have content
        ignition_cost["bits_broadcast"] = bits_content

        # Recompute κ with actual bits
        kappa = self.atp_calculator.compute_kappa(
            atp_total=ignition_cost["atp_total"],
            bits_processed=bits_content,
            consider_landauer=True,
        )

        ignition_cost["kappa_landauer"] = kappa
        ignition_cost["signal_excess"] = signal_excess
        ignition_cost["broadcast_amplitude"] = broadcast_amplitude
        ignition_cost["threshold"] = threshold
        ignition_cost["ignition_signal"] = ignition_signal

        # Track kappa history
        self.kappa_history.append(kappa)
        if len(self.kappa_history) > self.max_kappa_history:
            self.kappa_history.pop(0)

        self.ignition_event_count += 1

        return ignition_cost

    def get_dynamic_kappa(self, use_recent_average: bool = True) -> float:
        """Get the current κ value for metabolic cost calculations.

        Args:
            use_recent_average: If True, return average of recent κ values

        Returns:
            Current κ value (ATP cost normalized by Landauer limit)
        """
        if not self.kappa_history:
            # Default value if no ignition events yet
            return 1.0e6  # ~1 million times Landauer limit (typical biological estimate)

        if use_recent_average and len(self.kappa_history) >= 5:
            # Use recent average for stability
            return float(np.mean(self.kappa_history[-5:]))

        return self.kappa_history[-1]

    def get_metabolic_state(self) -> Dict[str, Any]:
        """Get current metabolic state summary."""
        atp_stats = self.atp_calculator.get_summary_stats()
        avg_firing = self.neural_model.get_average_firing_rates(100.0)

        return {
            "accumulated_atp": self.accumulated_atp,
            "ignition_event_count": self.ignition_event_count,
            "current_kappa": self.get_dynamic_kappa(),
            "kappa_history_mean": float(np.mean(self.kappa_history)) if self.kappa_history else 0.0,
            "atp_stats": atp_stats,
            "recent_firing_rates": avg_firing,
        }

    def reset(self) -> None:
        """Reset the metabolic layer to initial state."""
        self.neural_model.reset()
        self.atp_calculator.reset()
        self.current_input_drive = 0.0
        self.accumulated_atp = 0.0
        self.ignition_event_count = 0
        self.kappa_history = []


# Convenience function for quick κ estimation
def estimate_kappa_for_ignition(
    ignition_signal: float,
    threshold: float,
    workspace_content: Optional[np.ndarray] = None,
    ignition_duration_ms: float = 300.0,
    workspace_size_neurons: int = 100_000,
) -> float:
    """Quick estimate of κ for an ignition event.

    This is a convenience function for one-off κ calculations.
    For repeated use, create a VirtualMetabolicLayer instance.

    Args:
        ignition_signal: Accumulated surprise S_t
        threshold: Ignition threshold θ_t
        workspace_content: Optional content for information estimation
        ignition_duration_ms: Duration of ignition in milliseconds
        workspace_size_neurons: Size of the broadcasting workspace

    Returns:
        Estimated κ value (ATP cost relative to Landauer limit)
    """
    layer = VirtualMetabolicLayer()

    # Simulate pre-ignition buildup
    layer.simulate_neural_activity(input_drive=0.3, duration_ms=100.0)

    # Compute ignition cost
    cost = layer.compute_ignition_cost(
        ignition_signal=ignition_signal,
        threshold=threshold,
        workspace_content=workspace_content,
        ignition_duration_ms=ignition_duration_ms,
    )

    return float(cost["kappa_landauer"])
