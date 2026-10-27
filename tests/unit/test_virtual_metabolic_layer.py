"""
Tests for Virtual Metabolic Layer with Neural Mass Models.

This test suite validates:
1. Neural Mass Model dynamics (Jansen-Rit inspired)
2. ATP flux calculations for glutamate recycling and ion pumping
3. Kappa (κ) estimation from biophysical principles
4. Integration with existing APGI metabolic systems
"""

import numpy as np
import pytest

from apgi_framework.thermodynamic.metabolic_integration import (
    IntegratedMetabolicSystem,
    MetabolicIgnitionAdapter,
    create_metabolic_system_with_vml,
)
from apgi_framework.thermodynamic.neural_mass_metabolism import (
    ATPFluxCalculator,
    MetabolicCostFactors,
    NeuralMassModel,
    NeuralMassParameters,
    VirtualMetabolicLayer,
    estimate_kappa_for_ignition,
)


class TestNeuralMassParameters:
    """Tests for NeuralMassParameters dataclass."""

    def test_default_parameters(self):
        """Test default parameter initialization."""
        params = NeuralMassParameters()

        assert params.excitatory_time_constant_ms == 10.0
        assert params.inhibitory_time_constant_ms == 20.0
        assert params.synaptic_gain_excitatory == 3.25
        assert params.synaptic_gain_inhibitory == 22.0
        assert params.num_neurons == 100_000
        assert params.excitatory_fraction == 0.8

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        params = NeuralMassParameters(
            excitatory_time_constant_ms=15.0,
            num_neurons=50_000,
            excitatory_fraction=0.75,
        )

        assert params.excitatory_time_constant_ms == 15.0
        assert params.num_neurons == 50_000
        assert params.excitatory_fraction == 0.75
        # Default values should still work
        assert params.inhibitory_time_constant_ms == 20.0


class TestMetabolicCostFactors:
    """Tests for MetabolicCostFactors dataclass."""

    def test_default_factors(self):
        """Test default metabolic cost factors."""
        factors = MetabolicCostFactors()

        assert factors.atp_per_glutamate_cycle == 20_000.0
        assert factors.resting_atp_per_neuron_s == 1.0e9
        assert factors.astrocyte_lactate_factor == 1.2

    def test_atp_cost_calculations(self):
        """Test that ATP costs are positive and reasonable."""
        factors = MetabolicCostFactors()

        # Glutamate recycling should cost ATP
        assert factors.atp_per_glutamate_cycle > 0
        # Resting metabolism should be substantial
        assert factors.resting_atp_per_neuron_s > 1.0e8


class TestNeuralMassModel:
    """Tests for NeuralMassModel dynamics."""

    def test_initialization(self):
        """Test model initialization."""
        model = NeuralMassModel()

        assert model.state.shape == (8,)
        assert np.all(model.state == 0.0)
        assert len(model.pyramidal_firing_history) == 0

    def test_sigmoid_function(self):
        """Test sigmoid firing rate function."""
        model = NeuralMassModel()

        # At high input, firing rate should approach max
        high_output = model.sigmoid(20.0)
        assert high_output > 8.0  # Should be near max_firing_rate

        # At low input, firing rate should be low
        low_output = model.sigmoid(-10.0)
        assert low_output < 1.0

        # At threshold, firing rate should be around half max
        mid_output = model.sigmoid(6.0)  # near v_0
        assert 3.0 < mid_output < 7.0

    def test_step_integration(self):
        """Test that step integration runs without errors."""
        model = NeuralMassModel()

        # Run a few steps
        for _ in range(10):
            result = model.step(input_current=0.2, dt_ms=1.0)

            # Check required keys
            assert "pyramidal_firing_rate_hz" in result
            assert "interneuron_firing_rate_hz" in result
            assert "synaptic_activity" in result

            # Check firing rates are non-negative
            assert result["pyramidal_firing_rate_hz"] >= 0
            assert result["interneuron_firing_rate_hz"] >= 0

    def test_history_tracking(self):
        """Test that firing history is tracked."""
        model = NeuralMassModel()

        # Run many steps to populate history
        for _ in range(50):
            model.step(input_current=0.3, dt_ms=1.0)

        assert len(model.pyramidal_firing_history) == 50
        assert len(model.interneuron_firing_history) == 50

    def test_reset(self):
        """Test model reset functionality."""
        model = NeuralMassModel()

        # Run some steps
        for _ in range(20):
            model.step(input_current=0.3, dt_ms=1.0)

        # Reset
        model.reset()

        assert np.all(model.state == 0.0)
        assert len(model.pyramidal_firing_history) == 0

    def test_response_to_input(self):
        """Test that model responds appropriately to input."""
        model = NeuralMassModel()

        # The Jansen-Rit column oscillates, so compare time-averaged rates after the transient
        low_firing = _settled_firing_rate(model, 0.1)
        model.reset()
        high_firing = _settled_firing_rate(model, 0.5)

        # High input should produce higher firing rate
        assert high_firing > low_firing


def _settled_firing_rate(model: NeuralMassModel, input_current: float) -> float:
    """Mean pyramidal firing rate (Hz) after a 100 ms transient, over the next 300 ms."""
    rates = [
        model.step(input_current=input_current, dt_ms=1.0)["pyramidal_firing_rate_hz"]
        for _ in range(400)
    ]
    return float(np.mean(rates[100:]))


class TestATPFluxCalculator:
    """Tests for ATP flux calculation."""

    def test_initialization(self):
        """Test calculator initialization."""
        calc = ATPFluxCalculator()

        assert calc.total_atp_consumed == 0.0
        assert len(calc.atp_history) == 0

    def test_atp_flux_computation(self):
        """Test ATP flux computation."""
        calc = ATPFluxCalculator()

        result = calc.compute_atp_flux(
            pyramidal_firing_rate_hz=5.0,
            interneuron_firing_rate_hz=10.0,
            synaptic_activity=1.0,
        )

        # Check all components exist
        assert "atp_glutamate_per_s" in result
        assert "atp_na_k_pump_per_s" in result
        assert "atp_ca_pump_per_s" in result
        assert "atp_baseline_per_s" in result
        assert "atp_total_per_s" in result

        # All ATP fluxes should be positive
        assert result["atp_total_per_s"] > 0
        assert result["atp_baseline_per_s"] > 0

    def test_total_atp_greater_than_baseline(self):
        """Test that total ATP exceeds baseline during activity."""
        calc = ATPFluxCalculator()

        # Baseline firing
        baseline = calc.compute_atp_flux(
            pyramidal_firing_rate_hz=0.0,
            interneuron_firing_rate_hz=0.0,
            synaptic_activity=0.0,
        )

        calc.reset()

        # Active firing
        active = calc.compute_atp_flux(
            pyramidal_firing_rate_hz=10.0,
            interneuron_firing_rate_hz=5.0,
            synaptic_activity=2.0,
        )

        # Active should be higher than baseline
        assert active["atp_total_per_s"] > baseline["atp_total_per_s"]

    def test_ignition_cost_computation(self):
        """Test ignition cost computation."""
        calc = ATPFluxCalculator()

        cost = calc.compute_ignition_cost(
            ignition_duration_ms=300.0,
            workspace_size_neurons=100_000,
            broadcast_amplitude=1.0,
        )

        # Check cost components
        assert "atp_total" in cost
        assert "atp_glutamate" in cost
        assert "atp_na_k_pump" in cost
        assert "kappa_landauer" in cost
        assert "bits_broadcast" in cost

        # All ATP costs should be positive
        assert cost["atp_total"] > 0
        assert cost["atp_glutamate"] > 0

        # Kappa should be positive and substantial
        assert cost["kappa_landauer"] > 1000  # Much higher than Landauer limit

    def test_kappa_computation(self):
        """Test kappa computation."""
        calc = ATPFluxCalculator()

        # Compute kappa for sample values
        atp_total = 1e15  # 1 quadrillion ATP molecules
        bits = 256.0

        kappa = calc.compute_kappa(atp_total, bits, consider_landauer=True)

        # Kappa should be positive
        assert kappa > 0

        # For biological systems, kappa should be much larger than 1
        # (Landauer limit is ~18 kT per bit)
        assert kappa > 1e5

    def test_kappa_increases_with_atp(self):
        """Test that kappa increases with ATP consumption."""
        calc = ATPFluxCalculator()

        bits = 100.0

        kappa_low = calc.compute_kappa(1e12, bits, consider_landauer=True)
        kappa_high = calc.compute_kappa(1e15, bits, consider_landauer=True)

        assert kappa_high > kappa_low


class TestVirtualMetabolicLayer:
    """Tests for the VirtualMetabolicLayer main class."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = VirtualMetabolicLayer()

        assert layer.current_input_drive == 0.0
        assert layer.accumulated_atp == 0.0
        assert layer.ignition_event_count == 0
        assert layer.neural_model is not None
        assert layer.atp_calculator is not None

    def test_simulate_neural_activity(self):
        """Test neural activity simulation."""
        layer = VirtualMetabolicLayer()

        result = layer.simulate_neural_activity(
            input_drive=0.3,
            duration_ms=100.0,
        )

        assert "duration_ms" in result
        assert "total_atp_consumed" in result
        assert "average_pyramidal_firing_hz" in result
        assert result["duration_ms"] == 100.0
        assert result["total_atp_consumed"] > 0

    def test_compute_ignition_cost(self):
        """Test ignition cost computation."""
        layer = VirtualMetabolicLayer()

        # Simulate some pre-ignition activity
        layer.simulate_neural_activity(input_drive=0.3, duration_ms=100.0)

        # Compute ignition cost
        content = np.random.randn(256)
        cost = layer.compute_ignition_cost(
            ignition_signal=3.0,
            threshold=2.0,
            workspace_content=content,
            ignition_duration_ms=300.0,
        )

        assert "atp_total" in cost
        assert "kappa_landauer" in cost
        assert "signal_excess" in cost
        assert "broadcast_amplitude" in cost

        # Signal excess should be positive
        assert cost["signal_excess"] > 0

        # Ignition count should increment
        assert layer.ignition_event_count == 1

    def test_get_dynamic_kappa(self):
        """Test dynamic kappa retrieval."""
        layer = VirtualMetabolicLayer()

        # Before any ignitions, should return default
        default_kappa = layer.get_dynamic_kappa()
        assert default_kappa > 0

        # After ignition, kappa should be from history
        layer.simulate_neural_activity(input_drive=0.3, duration_ms=50.0)
        layer.compute_ignition_cost(
            ignition_signal=3.0,
            threshold=2.0,
            workspace_content=np.random.randn(256),
        )

        kappa = layer.get_dynamic_kappa()
        assert kappa > 0
        assert len(layer.kappa_history) == 1

    def test_get_metabolic_state(self):
        """Test metabolic state retrieval."""
        layer = VirtualMetabolicLayer()

        # Run some activity
        layer.simulate_neural_activity(input_drive=0.3, duration_ms=50.0)

        state = layer.get_metabolic_state()

        assert "accumulated_atp" in state
        assert "ignition_event_count" in state
        assert "current_kappa" in state
        assert "recent_firing_rates" in state

    def test_reset(self):
        """Test layer reset."""
        layer = VirtualMetabolicLayer()

        # Run activity and ignition
        layer.simulate_neural_activity(input_drive=0.3, duration_ms=50.0)
        layer.compute_ignition_cost(
            ignition_signal=3.0,
            threshold=2.0,
            workspace_content=np.random.randn(256),
        )

        # Reset
        layer.reset()

        assert layer.accumulated_atp == 0.0
        assert layer.ignition_event_count == 0
        assert len(layer.kappa_history) == 0


class TestEstimateKappaForIgnition:
    """Tests for the convenience function."""

    def test_convenience_function(self):
        """Test the estimate_kappa_for_ignition convenience function."""
        kappa = estimate_kappa_for_ignition(
            ignition_signal=3.0,
            threshold=2.0,
            workspace_content=np.random.randn(256),
            ignition_duration_ms=300.0,
        )

        # Should return positive kappa
        assert kappa > 0
        # Should be in biological range (millions times Landauer)
        assert kappa > 1e5


class TestIntegratedMetabolicSystem:
    """Tests for the IntegratedMetabolicSystem."""

    def test_initialization(self):
        """Test system initialization."""
        config = {
            "thermodynamic": {
                "total_energy_budget": 100.0,
                "use_dynamic_kappa": True,
                "workspace_neurons": 100_000,
            }
        }

        system = IntegratedMetabolicSystem(config)

        assert system.use_dynamic_kappa is True
        assert system.workspace_neurons == 100_000
        assert system.metabolic_budget is not None
        assert system.vml is not None

    def test_static_kappa_fallback(self):
        """Test that static kappa fallback works."""
        config = {
            "thermodynamic": {
                "total_energy_budget": 100.0,
                "use_dynamic_kappa": False,
                "static_kappa": 5.0e5,
            }
        }

        system = IntegratedMetabolicSystem(config)

        assert system.use_dynamic_kappa is False
        assert system.vml is None

        kappa = system.get_current_kappa()
        assert kappa == 5.0e5

    def test_update_with_ignition(self):
        """Test system update with ignition event."""
        config = {
            "thermodynamic": {
                "total_energy_budget": 100.0,
                "use_dynamic_kappa": True,
                "workspace_neurons": 50_000,
            }
        }

        system = IntegratedMetabolicSystem(config)

        result = system.update(
            ignition_occurred=True,
            ignition_signal=3.0,
            threshold=2.0,
            workspace_content=np.random.randn(256),
            task_active=True,
            dt_ms=1.0,
        )

        assert "budget_reserves" in result
        assert "current_kappa" in result
        assert "ignition_count" in result

        # Ignition should be counted
        assert result["ignition_count"] == 1
        assert system.ignition_count == 1

    def test_update_without_ignition(self):
        """Test system update without ignition."""
        config = {
            "thermodynamic": {
                "total_energy_budget": 100.0,
                "use_dynamic_kappa": True,
            }
        }

        system = IntegratedMetabolicSystem(config)

        result = system.update(
            ignition_occurred=False,
            ignition_signal=1.5,  # Below threshold
            threshold=2.0,
            task_active=True,
            dt_ms=1.0,
        )

        assert result["ignition_count"] == 0
        assert result["last_ignition_cost"] is None

    def test_ignition_cost_estimate(self):
        """Test ignition cost estimation."""
        config = {
            "thermodynamic": {
                "total_energy_budget": 100.0,
                "use_dynamic_kappa": True,
            }
        }

        system = IntegratedMetabolicSystem(config)

        estimate = system.compute_ignition_cost_estimate(
            ignition_signal=3.0,
            threshold=2.0,
            workspace_content=np.random.randn(256),
        )

        assert "atp_total" in estimate
        assert "kappa_landauer" in estimate
        assert estimate["is_estimate"] is True

    def test_metabolic_summary(self):
        """Test metabolic summary generation."""
        config = {
            "thermodynamic": {
                "total_energy_budget": 100.0,
                "use_dynamic_kappa": True,
            }
        }

        system = IntegratedMetabolicSystem(config)

        # Run some activity
        for _ in range(5):
            system.update(
                ignition_occurred=True,
                ignition_signal=3.0,
                threshold=2.0,
                workspace_content=np.random.randn(256),
                task_active=True,
                dt_ms=1.0,
            )

        summary = system.get_metabolic_summary()

        assert "dynamic_kappa_enabled" in summary
        assert "ignition_count" in summary
        assert "kappa_stats" in summary
        assert summary["ignition_count"] == 5


class TestFactoryFunction:
    """Tests for the create_metabolic_system_with_vml factory."""

    def test_factory_basic(self):
        """Test basic factory usage."""
        system = create_metabolic_system_with_vml(
            workspace_neurons=100_000,
            use_dynamic_kappa=True,
        )

        assert isinstance(system, IntegratedMetabolicSystem)
        assert system.workspace_neurons == 100_000
        assert system.use_dynamic_kappa is True

    def test_factory_with_overrides(self):
        """Test factory with config overrides."""
        system = create_metabolic_system_with_vml(
            workspace_neurons=50_000,
            use_dynamic_kappa=True,
            config_overrides={
                "thermodynamic": {"total_energy_budget": 200.0},
                "neural_mass": {"excitatory_tau_ms": 12.0},
            },
        )

        assert system.workspace_neurons == 50_000
        assert system.metabolic_budget.total_budget == 200.0


class TestMetabolicIgnitionAdapter:
    """Tests for the MetabolicIgnitionAdapter."""

    def test_adapter_initialization(self):
        """Test adapter initialization."""

        # Mock ignition threshold object
        class MockThreshold:
            def update_metabolic_state(self, reserves, load):
                pass

        threshold = MockThreshold()
        config = {
            "thermodynamic": {
                "total_energy_budget": 100.0,
                "use_dynamic_kappa": True,
            }
        }
        metabolic_system = IntegratedMetabolicSystem(config)

        adapter = MetabolicIgnitionAdapter(threshold, metabolic_system)

        assert adapter.ignition_threshold is threshold
        assert adapter.metabolic_system is metabolic_system

    def test_adapter_update(self):
        """Test adapter update method."""

        class MockThreshold:
            def __init__(self):
                self.last_reserves = None
                self.last_load = None

            def update_metabolic_state(self, reserves, load):
                self.last_reserves = reserves
                self.last_load = load

        threshold = MockThreshold()
        config = {
            "thermodynamic": {
                "total_energy_budget": 100.0,
                "use_dynamic_kappa": True,
            }
        }
        metabolic_system = IntegratedMetabolicSystem(config)
        adapter = MetabolicIgnitionAdapter(threshold, metabolic_system)

        result = adapter.update(
            ignition_signal=3.0,
            threshold=2.0,
            ignition_occurred=True,
            workspace_content=np.random.randn(256),
            dt_ms=1.0,
        )

        assert "metabolic" in result
        assert "threshold" in result

        # Threshold should have been updated
        assert threshold.last_reserves is not None
        assert threshold.last_load is not None


@pytest.mark.parametrize(
    "input_drive,expected_response",
    [
        (0.1, "low"),
        (0.3, "moderate"),
        (0.5, "high"),
    ],
)
def test_neural_response_scaling(input_drive, expected_response):
    """Test that neural model responds appropriately to different inputs."""
    model = NeuralMassModel()

    firing_rate = _settled_firing_rate(model, input_drive)

    # Verify firing rate is in reasonable range
    assert 0 <= firing_rate <= 10.0  # Max firing rate is 5.0, allow margin

    # Check response pattern
    if expected_response == "low":
        assert firing_rate < 3.0
    elif expected_response == "moderate":
        assert 1.0 <= firing_rate <= 5.0
    else:  # high
        assert firing_rate > 2.0
