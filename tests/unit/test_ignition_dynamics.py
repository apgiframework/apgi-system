"""
Unit tests for ignition dynamics components.

Tests specific threshold scenarios, broadcasting mechanics, and temporal orchestration
for IgnitionThreshold, GlobalWorkspace, and IgnitionTimeline components.
"""

from typing import Any, Dict, Union

import numpy as np
import pytest

from apgi_framework.ignition.global_workspace import GlobalWorkspace, WorkspaceState
from apgi_framework.ignition.threshold import IgnitionThreshold


class TestIgnitionThreshold:
    """Unit tests for IgnitionThreshold component."""

    @pytest.fixture
    def threshold_config(self) -> Dict[str, Any]:
        """Configuration for threshold tests."""
        return {
            "ignition": {
                "baseline_threshold": 2.0,
                "threshold_range": [1.0, 5.0],
                "sigmoid_alpha": 5.0,
                "refractory_period_ms": 200,
            }
        }

    @pytest.fixture
    def threshold(self, threshold_config: Dict[str, Any]) -> IgnitionThreshold:
        """Create IgnitionThreshold instance."""
        return IgnitionThreshold(threshold_config)

    def test_initialization(self, threshold) -> None:
        """Test proper initialization of IgnitionThreshold."""
        assert threshold.baseline_threshold == 2.0
        assert threshold.threshold_range == [1.0, 5.0]
        assert threshold.sigmoid_alpha == 5.0
        assert threshold.refractory_period_ms == 200
        assert np.all(threshold.current_threshold == 2.0)
        assert np.all(threshold.current_signal == 0.0)
        assert np.all(threshold.metabolic_reserves == 1.0)
        assert np.all(threshold.allostatic_load == 0.0)

    def test_basic_signal_computation(self, threshold) -> None:
        """
        Test basic ignition signal computation.
        Verifies alignment with APGI Eq. 1.2 (Accumulated Signal).
        """
        extero_error = np.array([[1.0, 2.0, 1.5]])
        intero_error = np.array([[0.5, 0.3]])

        # First call to set start time
        threshold.compute_ignition_signal(
            extero_error=extero_error,
            extero_precision=np.array([1.5]),
            intero_error=intero_error,
            intero_precision=np.array([1.0]),
            somatic_marker_gain=np.array([1.2]),
            current_time=0.0,
        )

        # Second call with positive dt (1000ms = 1s)
        ignited, components = threshold.compute_ignition_signal(
            extero_error=extero_error,
            extero_precision=np.array([1.5]),
            intero_error=intero_error,
            intero_precision=np.array([1.0]),
            somatic_marker_gain=np.array([1.2]),
            current_time=1000.0,
        )

        # Check signal components
        # Vectorized version: extero_scalar = np.mean(extero_error**2, axis=1)
        extero_scalar = np.mean(extero_error**2)
        intero_scalar = np.mean(intero_error**2)

        expected_extero = 0.5 * 1.5 * extero_scalar
        expected_intero = 0.5 * 1.0 * 1.2 * intero_scalar
        expected_total_drive = expected_extero + expected_intero

        expected_total = expected_total_drive

        assert abs(components["total_signal"][0] - expected_total) < 0.2
        # The threshold adapts allostatically (signal history lowers it slightly per step),
        # so after two updates it stays close to, not exactly at, the 2.0 baseline.
        assert components["threshold"][0] == pytest.approx(2.0, abs=0.1)
        assert 0 <= components["probability"][0] <= 1
        assert isinstance(ignited, np.ndarray)
        assert ignited.dtype == bool

    def test_threshold_exceeding_ignition(self, threshold) -> None:
        """
        Test ignition when signal exceeds threshold.
        Verifies alignment with APGI Eq. 2.3 (Ignition Probability).
        """
        # Create large error to exceed threshold
        extero_error = np.array([[3.0, 4.0, 2.0]])  # Large error
        intero_error = np.array([[1.0, 1.5]])

        # Multiple attempts to account for stochastic nature
        ignition_occurred = False
        current_time = 0.0
        for _ in range(50):  # Try multiple times
            current_time += 10.0  # Progress time (10ms)
            ignited, components = threshold.compute_ignition_signal(
                extero_error=extero_error,
                extero_precision=np.array([2.0]),
                intero_error=intero_error,
                intero_precision=np.array([1.5]),
                somatic_marker_gain=np.array([1.0]),
                current_time=100.0 + current_time,
            )
            if ignited[0]:
                ignition_occurred = True
                break

        # With such large signals, ignition should occur eventually
        assert ignition_occurred, "Large signal should eventually trigger ignition"

    def test_refractory_period_enforcement(self, threshold) -> None:
        """Test that refractory period prevents immediate re-ignition."""
        # Create large error that should trigger ignition
        extero_error = np.array([[5.0, 5.0]])
        intero_error = np.array([[2.0]])

        # First ignition
        ignited1, _ = threshold.compute_ignition_signal(
            extero_error=extero_error,
            extero_precision=np.array([3.0]),
            intero_error=intero_error,
            intero_precision=np.array([2.0]),
            somatic_marker_gain=np.array([1.0]),
            current_time=100.0,
        )

        # If first attempt didn't ignite, keep trying until it does
        current_time = 100.0
        while not ignited1[0]:
            current_time += 1.0
            ignited1, _ = threshold.compute_ignition_signal(
                extero_error=extero_error,
                extero_precision=np.array([3.0]),
                intero_error=intero_error,
                intero_precision=np.array([2.0]),
                somatic_marker_gain=np.array([1.0]),
                current_time=current_time,
            )

        # Immediate second attempt (within refractory period)
        ignited2, _ = threshold.compute_ignition_signal(
            extero_error=extero_error,
            extero_precision=np.array([3.0]),
            intero_error=intero_error,
            intero_precision=np.array([2.0]),
            somatic_marker_gain=np.array([1.0]),
            current_time=current_time + 50.0,  # 50ms later, within 200ms refractory
        )

        # Should not ignite due to refractory period
        assert not ignited2[0], "Should not ignite within refractory period"

        # After refractory period
        ignited3, _ = threshold.compute_ignition_signal(
            extero_error=extero_error,
            extero_precision=np.array([3.0]),
            intero_error=intero_error,
            intero_precision=np.array([2.0]),
            somatic_marker_gain=np.array([1.0]),
            current_time=current_time + 250.0,  # 250ms later, after refractory
        )

        # May ignite again (stochastic, but not blocked by refractory period)
        assert isinstance(ignited3, np.ndarray)

    def test_metabolic_state_modulation(self, threshold) -> None:
        """Test threshold modulation by metabolic state."""
        # Test with full reserves
        threshold.update_metabolic_state(reserves=np.array([1.0]), allostatic_load=np.array([0.0]))
        threshold._update_threshold(100.0)
        full_reserves_threshold = threshold.current_threshold[0]

        # Test with depleted reserves
        threshold.update_metabolic_state(reserves=np.array([0.2]), allostatic_load=np.array([0.0]))
        threshold._update_threshold(100.0)
        depleted_reserves_threshold = threshold.current_threshold[0]

        # Depleted reserves should increase threshold
        assert depleted_reserves_threshold > full_reserves_threshold

        # Test with high allostatic load
        threshold.update_metabolic_state(reserves=np.array([1.0]), allostatic_load=np.array([0.8]))
        threshold._update_threshold(100.0)
        high_load_threshold = threshold.current_threshold[0]

        # High load should increase threshold
        assert high_load_threshold > full_reserves_threshold

    def test_somatic_marker_gain_modulation(self, threshold):
        """
        Test somatic marker gain effects on interoceptive signal.
        Verifies alignment with APGI Eq. 2.2 (Effective Interoceptive Precision).
        """
        extero_error = np.array([[1.0]])
        intero_error = np.array([[1.0]])

        # Test with low gain (aversive context)
        _, components_low = threshold.compute_ignition_signal(
            extero_error=extero_error,
            extero_precision=np.array([1.0]),
            intero_error=intero_error,
            intero_precision=np.array([1.0]),
            somatic_marker_gain=np.array([0.6]),
            current_time=100.0,
        )

        # Test with high gain (appetitive context)
        _, components_high = threshold.compute_ignition_signal(
            extero_error=extero_error,
            extero_precision=np.array([1.0]),
            intero_error=intero_error,
            intero_precision=np.array([1.0]),
            somatic_marker_gain=np.array([1.8]),
            current_time=200.0,
        )

        # Higher gain should produce higher interoceptive signal
        assert components_high["intero_signal"][0] > components_low["intero_signal"][0]
        assert components_high["total_signal"][0] > components_low["total_signal"][0]

    def test_input_validation(self, threshold) -> None:
        """Test input validation for compute_ignition_signal."""
        valid_extero = np.array([[1.0, 2.0]])
        valid_intero = np.array([[0.5]])

        # Test invalid extero_error type
        with pytest.raises(TypeError):
            threshold.compute_ignition_signal(
                extero_error=[1.0, 2.0],  # List instead of array
                extero_precision=np.array([1.0]),
                intero_error=valid_intero,
                intero_precision=np.array([1.0]),
                somatic_marker_gain=np.array([1.0]),
            )

        # Test negative precision
        with pytest.raises(ValueError):
            threshold.compute_ignition_signal(
                extero_error=valid_extero,
                extero_precision=np.array([-1.0]),  # Negative precision
                intero_error=valid_intero,
                intero_precision=np.array([1.0]),
                somatic_marker_gain=np.array([1.0]),
            )

        # Test somatic marker gain out of range
        with pytest.raises(ValueError):
            threshold.compute_ignition_signal(
                extero_error=valid_extero,
                extero_precision=np.array([1.0]),
                intero_error=valid_intero,
                intero_precision=np.array([1.0]),
                somatic_marker_gain=np.array([3.0]),  # Out of range [0.5, 2.0]
            )

    def test_statistics_computation(self, threshold) -> None:
        """Test statistics computation."""
        # Initially empty
        stats = threshold.get_statistics()
        assert stats["mean_signal"] == 0.0
        assert stats["recent_ignitions"] == 0

        # Add some history
        extero_error = np.array([[1.0, 1.5]])
        intero_error = np.array([[0.5]])

        for i in range(10):
            threshold.compute_ignition_signal(
                extero_error=extero_error,
                extero_precision=np.array([1.0]),
                intero_error=intero_error,
                intero_precision=np.array([1.0]),
                somatic_marker_gain=np.array([1.0]),
                current_time=float(i * 10),
            )

        stats = threshold.get_statistics()
        assert stats["mean_signal"] > 0
        assert "std_signal" in stats
        assert "mean_threshold" in stats
        assert "ignition_rate" in stats

    def test_threshold_bit_flip_sensitivity(self, threshold: IgnitionThreshold) -> None:
        """
        Test that even minute changes (single-bit equivalent) impact ignition probability.
        Critical for research integrity and mutation testing robustness.
        Verifies alignment with APGI Eq. 2.3 (Sigmoid Sharpness).
        """
        extero_error = np.array([[2.0]])
        intero_error = np.array([[1.0]])

        # Point of maximum sensitivity (S ≈ theta)
        threshold.baseline_threshold = 2.5
        threshold.reset()

        # Step twice to accumulate signal
        threshold.compute_ignition_signal(
            extero_error=extero_error,
            extero_precision=np.array([1.0]),
            intero_error=intero_error,
            intero_precision=np.array([1.0]),
            somatic_marker_gain=np.array([1.0]),
            current_time=0.0,
        )
        _, components = threshold.compute_ignition_signal(
            extero_error=extero_error,
            extero_precision=np.array([1.0]),
            intero_error=intero_error,
            intero_precision=np.array([1.0]),
            somatic_marker_gain=np.array([1.0]),
            current_time=1000.0,
        )
        base_prob = components["probability"][0]

        # Perturb error by a tiny amount (simulating near-epsilon change)
        eps = 1e-10
        threshold.reset()
        threshold.compute_ignition_signal(
            extero_error=extero_error + eps,
            extero_precision=np.array([1.0]),
            intero_error=intero_error,
            intero_precision=np.array([1.0]),
            somatic_marker_gain=np.array([1.0]),
            current_time=0.0,
        )
        _, components_eps = threshold.compute_ignition_signal(
            extero_error=extero_error + eps,
            extero_precision=np.array([1.0]),
            intero_error=intero_error,
            intero_precision=np.array([1.0]),
            somatic_marker_gain=np.array([1.0]),
            current_time=1000.0,
        )
        eps_prob = components_eps["probability"][0]

        # Sigmoid probability should change
        assert (
            eps_prob != base_prob
        ), f"Probability {base_prob} did not change for tiny perturbation"

    def test_latency_to_ignition_prediction(self, threshold: IgnitionThreshold) -> None:
        """
        Test deterministic latency to ignition prediction.
        Verifies alignment with APGI Eq. 5.1 (Latency to Ignition).
        """
        # Massive signal
        extero_error = np.array([[10.0]])

        threshold.reset()
        t = 0.0
        ignited: Union[bool, np.ndarray] = False
        while not (isinstance(ignited, np.ndarray) and ignited[0]) and t < 1000:
            t += 10.0  # 10ms steps
            ignited, _ = threshold.compute_ignition_signal(
                extero_error=extero_error,
                extero_precision=np.array([10.0]),
                intero_error=np.zeros((1, 1)),
                intero_precision=np.array([0.0]),
                somatic_marker_gain=np.array([1.0]),
                current_time=t,
            )

        # High signal should ignite quickly (< 100ms)
        assert t < 100, f"Expected fast ignition for high signal, but took {t}ms"

    def test_reset_functionality(self, threshold: IgnitionThreshold) -> None:
        """Test reset restores initial state."""
        # Modify state
        threshold.metabolic_reserves[0] = 0.5
        threshold.allostatic_load[0] = 0.3
        threshold.current_signal[0] = 5.0

        # Reset
        threshold.reset()

        # Check restoration
        assert np.all(threshold.current_threshold == threshold.baseline_threshold)
        assert np.all(threshold.current_signal == 0.0)
        assert np.all(threshold.metabolic_reserves == 1.0)
        assert np.all(threshold.allostatic_load == 0.0)


class TestGlobalWorkspace:
    """Unit tests for GlobalWorkspace component."""

    @pytest.fixture
    def workspace_config(self) -> Dict[str, Any]:
        """Configuration for workspace tests."""
        return {"ignition": {"amplification_duration_ms": 300, "workspace_dim": 256}}

    @pytest.fixture
    def workspace(self, workspace_config: Dict[str, Any]) -> GlobalWorkspace:
        """Create GlobalWorkspace instance."""
        return GlobalWorkspace(workspace_config)

    def test_initialization(self, workspace: GlobalWorkspace) -> None:
        """Test proper initialization of GlobalWorkspace."""
        assert workspace.amplification_duration_ms == 300
        assert np.all(workspace.states == WorkspaceState.IDLE.value)
        assert workspace.current_content.shape == (1, 256)
        assert np.all(workspace.state_times == 0.0)

    def test_idle_to_igniting_transition(self, workspace: GlobalWorkspace) -> None:
        """Test transition from IDLE to IGNITING state."""
        # Set candidate
        candidate = np.random.randn(1, 256)

        # Idle update
        state = workspace.update(ignition_mask=np.array([False]), candidates=candidate)
        assert state["states"][0] == WorkspaceState.IDLE.value

        # Trigger ignition
        state = workspace.update(ignition_mask=np.array([True]))
        assert state["states"][0] == WorkspaceState.IGNITING.value
        assert workspace.state_times[0] >= 0.0

    def test_competition_resolution(self, workspace: GlobalWorkspace) -> None:
        """Test winner selection (simple in this version)."""
        candidate = np.random.randn(1, 256)

        # Trigger ignition
        workspace.update(ignition_mask=np.array([True]), candidates=candidate)

        # Progress through igniting phase (50ms)
        for _ in range(60):
            state = workspace.update(ignition_mask=np.array([False]), dt=1.0)
            if state["states"][0] == WorkspaceState.BROADCASTING.value:
                break

        # Should have transitioned to broadcasting
        assert state["states"][0] == WorkspaceState.BROADCASTING.value
        # Content should be non-zero (account for amplification/noise)
        assert np.linalg.norm(state["content"][0]) > 0

    def test_broadcasting_phase(self, workspace: GlobalWorkspace) -> None:
        """Test broadcasting phase mechanics."""
        candidate = np.random.randn(1, 256)
        workspace.update(ignition_mask=np.array([True]), candidates=candidate)

        # Progress to broadcasting
        for _ in range(60):
            state = workspace.update(ignition_mask=np.array([False]), dt=1.0)
            if state["states"][0] == WorkspaceState.BROADCASTING.value:
                break

        # Should be broadcasting and reportable
        assert state["states"][0] == WorkspaceState.BROADCASTING.value
        assert state["is_reportable"][0]

    def test_state_machine_progression(self, workspace: GlobalWorkspace) -> None:
        """Test complete state machine progression."""
        candidate = np.random.randn(1, 256)
        workspace.update(ignition_mask=np.array([True]), candidates=candidate)

        # Should be IGNITING
        assert workspace.states[0] == WorkspaceState.IGNITING.value

        # Progress through IGNITING (50ms)
        for _ in range(100):
            state = workspace.update(ignition_mask=np.array([False]), dt=1.0)
            if state["states"][0] != WorkspaceState.IGNITING.value:
                break
        assert state["states"][0] == WorkspaceState.BROADCASTING.value

        # Progress through BROADCASTING (300ms)
        for _ in range(400):
            state = workspace.update(ignition_mask=np.array([False]), dt=1.0)
            if state["states"][0] != WorkspaceState.BROADCASTING.value:
                break
        assert state["states"][0] == WorkspaceState.MAINTAINING.value

        # Progress through MAINTAINING (1000ms)
        for _ in range(1100):
            state = workspace.update(ignition_mask=np.array([False]), dt=1.0)
            if state["states"][0] != WorkspaceState.MAINTAINING.value:
                break
        assert state["states"][0] == WorkspaceState.FADING.value

        # Progress through FADING (200ms)
        for _ in range(300):
            state = workspace.update(ignition_mask=np.array([False]), dt=1.0)
            if state["states"][0] != WorkspaceState.FADING.value:
                break
        assert state["states"][0] == WorkspaceState.IDLE.value

    def test_reset_functionality(self, workspace: GlobalWorkspace):
        """Test reset restores initial state."""
        workspace.states[0] = WorkspaceState.BROADCASTING.value
        workspace.state_times[0] = 100.0
        workspace.current_content[0] = np.random.randn(256)

        workspace.reset()

        assert workspace.states[0] == WorkspaceState.IDLE.value
        assert np.all(workspace.current_content == 0.0)
        assert workspace.state_times[0] == 0.0
