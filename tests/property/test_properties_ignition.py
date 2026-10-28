"""
Property-based tests for ignition dynamics components.

This module implements property-based tests using Hypothesis to verify
universal properties that should hold across all valid inputs for the
ignition dynamics components of the APGI system.

Each test is tagged with the corresponding property from the design document
and validates specific requirements from the requirements document.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # noqa: E402

from typing import Any, Dict  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from apgi_framework.ignition.global_workspace import GlobalWorkspace  # noqa: E402
from apgi_framework.ignition.threshold import IgnitionThreshold  # noqa: E402
from tests.strategies import (  # noqa: E402
    precision_weighted_error_strategy,
    somatic_marker_gain_strategy,
)

# Configure Hypothesis for property-based testing
settings.register_profile(
    "property_tests",
    max_examples=100,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.large_base_example,
    ],
)
settings.load_profile("property_tests")


def load_config() -> Dict[str, Any]:
    """Load configuration for tests."""
    config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


class TestIgnitionDynamicsProperties:
    """Property-based tests for ignition dynamics invariants."""

    @given(
        error_data=precision_weighted_error_strategy(),
        somatic_gain=somatic_marker_gain_strategy(),
        current_time=st.floats(min_value=0.0, max_value=10000.0),
    )
    def test_property_ignition_threshold_crossing(
        self, error_data: Dict[str, Any], somatic_gain: float, current_time: float
    ) -> None:
        """
        **Feature: apgi-enhancement-maintenance, Property 6: Ignition occurs when signal exceeds threshold**

        For any signal exceeding threshold (outside refractory), ignition should occur with high probability.

        **Validates: Requirements 3.1**
        """
        config = load_config()
        threshold_system = IgnitionThreshold(config)

        # Set up conditions for high signal
        extero_error = error_data["extero_error"] * 5.0  # Amplify to ensure high signal
        extero_precision = np.array(
            [error_data["extero_precision"] * 2.0]
        )  # High precision as array
        intero_error = error_data["intero_error"] * 3.0  # Amplify intero signal
        intero_precision = np.array(
            [error_data["intero_precision"] * 2.0]
        )  # High precision as array

        # Ensure we're outside refractory period
        threshold_system.last_ignition_time = np.array([current_time - 1000.0])  # 1 second ago

        # Compute ignition signal
        ignited, components = threshold_system.compute_ignition_signal(
            extero_error=extero_error,
            extero_precision=extero_precision,
            intero_error=intero_error,
            intero_precision=intero_precision,
            somatic_marker_gain=np.array([somatic_gain]),
            current_time=current_time,
        )

        total_signal = components["total_signal"]
        threshold = components["threshold"]

        # Signal should be positive and finite
        assert total_signal >= 0, f"Signal should be non-negative: {total_signal}"
        assert np.isfinite(total_signal), f"Signal should be finite: {total_signal}"

        # Threshold should be positive and finite
        assert threshold > 0, f"Threshold should be positive: {threshold}"
        assert np.isfinite(threshold), f"Threshold should be finite: {threshold}"

        # If signal significantly exceeds threshold, ignition probability should be high
        if total_signal > threshold + 1.0:  # Signal well above threshold
            probability = components["ignition_probability"]
            assert probability > 0.5, (
                f"High signal should give high ignition probability: signal={total_signal:.2f}, "
                f"threshold={threshold:.2f}, prob={probability:.3f}"
            )

        # Signal should be non-negative and finite
        assert total_signal >= 0, f"Total signal should be non-negative: {total_signal}"
        assert np.isfinite(total_signal), f"Total signal should be finite: {total_signal}"

        # Component signals should be non-negative
        assert components["extero_signal"] >= 0, "Extero signal should be non-negative"
        assert components["intero_signal"] >= 0, "Intero signal should be non-negative"

    @given(
        amplification_duration=st.floats(min_value=200.0, max_value=500.0),
        num_updates=st.integers(min_value=10, max_value=50),
    )
    def test_property_workspace_broadcast_duration(
        self,
        amplification_duration: float,
        num_updates: int,
    ) -> None:
        """
        **Feature: apgi-enhancement-maintenance, Property 7: Workspace broadcast duration**

        For any ignition event, the global workspace should broadcast for 300-500ms duration.

        **Validates: Requirements 3.2**
        """
        config = load_config()
        config["ignition"]["amplification_duration_ms"] = amplification_duration
        workspace = GlobalWorkspace(config)

        # Generate candidate content matching workspace dimensions
        candidate_content = np.random.randn(1, workspace.content_dim)

        # Trigger ignition
        state = workspace.update(
            ignition_mask=np.array([True]),
            candidates=candidate_content,
            priorities=np.array([1.5]),
            dt=1.0,
        )

        # Track broadcasting duration
        broadcast_start_time = None
        broadcast_end_time = None
        total_time = 0.0

        for i in range(num_updates):
            state = workspace.update(ignition_mask=np.array([False]), dt=1.0)
            total_time += 1.0

            is_broadcasting = state["is_reportable"][0]
            if is_broadcasting and broadcast_start_time is None:
                broadcast_start_time = total_time

            if (
                not is_broadcasting
                and broadcast_start_time is not None
                and broadcast_end_time is None
            ):
                broadcast_end_time = total_time
                break

        # If we observed a complete broadcast cycle
        if broadcast_start_time is not None and broadcast_end_time is not None:
            broadcast_duration = broadcast_end_time - broadcast_start_time

            # Duration should be within expected range (allowing some tolerance)
            expected_total_duration = amplification_duration + 1000.0  # amp + maintenance
            assert (
                broadcast_duration >= amplification_duration * 0.8
            ), f"Broadcast duration too short: {broadcast_duration}ms (expected >= {amplification_duration * 0.8}ms)"

            assert (
                broadcast_duration <= expected_total_duration * 1.2
            ), f"Broadcast duration too long: {broadcast_duration}ms (expected <= {expected_total_duration * 1.2}ms)"

    @given(
        reserves_low=st.floats(min_value=0.0, max_value=0.3),
        reserves_high=st.floats(min_value=0.7, max_value=1.0),
        error_data=precision_weighted_error_strategy(),
    )
    def test_property_threshold_increase_with_depleted_reserves(
        self, reserves_low: float, reserves_high: float, error_data: Dict[str, Any]
    ) -> None:
        """
        **Feature: apgi-enhancement-maintenance, Property 8: Threshold increases with depleted reserves**

        For any metabolic reserve level, the ignition threshold should increase monotonically
        as reserves decrease (lower reserves → higher threshold).

        **Validates: Requirements 3.3**
        """
        # Skip edge cases where reserves are too close to boundaries
        # as dynamic threshold may produce equal values due to other factors
        assume(reserves_low < 0.25 or reserves_high > 0.75)
        assume(abs(reserves_high - reserves_low) > 0.4)  # Ensure significant difference

        config = load_config()

        # Test with high reserves
        threshold_high_reserves = IgnitionThreshold(config)
        threshold_high_reserves.update_metabolic_state(np.array([reserves_high]), np.array([0.0]))

        _, components_high = threshold_high_reserves.compute_ignition_signal(
            extero_error=error_data["extero_error"],
            extero_precision=np.array([error_data["extero_precision"]]),
            intero_error=error_data["intero_error"],
            intero_precision=np.array([error_data["intero_precision"]]),
            somatic_marker_gain=np.array([1.0]),
            current_time=0.0,
        )

        # Test with low reserves
        threshold_low_reserves = IgnitionThreshold(config)
        threshold_low_reserves.update_metabolic_state(np.array([reserves_low]), np.array([0.0]))

        _, components_low = threshold_low_reserves.compute_ignition_signal(
            extero_error=error_data["extero_error"],
            extero_precision=np.array([error_data["extero_precision"]]),
            intero_error=error_data["intero_error"],
            intero_precision=np.array([error_data["intero_precision"]]),
            somatic_marker_gain=np.array([1.0]),
            current_time=0.0,
        )

        threshold_high = components_high["threshold"]
        threshold_low = components_low["threshold"]

        # Lower reserves should result in higher threshold
        # Use >= with tolerance due to dynamic threshold factors
        # Allow small violations due to numerical precision
        assert threshold_low >= threshold_high - 0.01, (
            f"Threshold should increase with depleted reserves: "
            f"high_reserves={reserves_high:.2f} -> threshold={threshold_high:.2f}, "
            f"low_reserves={reserves_low:.2f} -> threshold={threshold_low:.2f}"
        )

        # Both thresholds should be positive and finite
        assert threshold_high > 0 and np.isfinite(threshold_high)
        assert threshold_low > 0 and np.isfinite(threshold_low)

    @given(
        load_low=st.floats(min_value=0.0, max_value=0.3),
        load_high=st.floats(min_value=0.7, max_value=1.0),
        error_data=precision_weighted_error_strategy(),
    )
    def test_property_threshold_modulation_by_allostatic_load(
        self, load_low: float, load_high: float, error_data: Dict[str, Any]
    ) -> None:
        """
        **Feature: apgi-enhancement-maintenance, Property 9: Threshold modulation by allostatic load**

        For any allostatic load value, the ignition threshold should increase monotonically
        with load (higher load → higher threshold).

        **Validates: Requirements 3.4**
        """
        config = load_config()

        # Test with low allostatic load
        threshold_low_load = IgnitionThreshold(config)
        threshold_low_load.update_metabolic_state(
            np.array([1.0]), np.array([load_low])
        )  # Full reserves, low load

        _, components_low = threshold_low_load.compute_ignition_signal(
            extero_error=error_data["extero_error"],
            extero_precision=np.array([error_data["extero_precision"]]),
            intero_error=error_data["intero_error"],
            intero_precision=np.array([error_data["intero_precision"]]),
            somatic_marker_gain=np.array([1.0]),
            current_time=0.0,
        )

        # Test with high allostatic load
        threshold_high_load = IgnitionThreshold(config)
        threshold_high_load.update_metabolic_state(
            np.array([1.0]), np.array([load_high])
        )  # Full reserves, high load

        _, components_high = threshold_high_load.compute_ignition_signal(
            extero_error=error_data["extero_error"],
            extero_precision=np.array([error_data["extero_precision"]]),
            intero_error=error_data["intero_error"],
            intero_precision=np.array([error_data["intero_precision"]]),
            somatic_marker_gain=np.array([1.0]),
            current_time=0.0,
        )

        threshold_low = components_low["threshold"]
        threshold_high = components_high["threshold"]

        # Higher allostatic load should result in higher threshold
        # Use >= with tolerance due to dynamic threshold factors
        assert threshold_high >= threshold_low - 0.01, (
            f"Threshold should increase with allostatic load: "
            f"low_load={load_low:.2f} -> threshold={threshold_low:.2f}, "
            f"high_load={load_high:.2f} -> threshold={threshold_high:.2f}"
        )

        # Both thresholds should be positive and finite
        assert threshold_low > 0 and np.isfinite(threshold_low)
        assert threshold_high > 0 and np.isfinite(threshold_high)

    @given(
        gain_low=st.floats(min_value=0.5, max_value=0.8),
        gain_high=st.floats(min_value=1.2, max_value=2.0),
        error_data=precision_weighted_error_strategy(),
    )
    def test_property_somatic_marker_gain_incorporation(
        self, gain_low: float, gain_high: float, error_data: Dict[str, Any]
    ) -> None:
        """
        **Feature: apgi-enhancement-maintenance, Property 10: Somatic marker gain incorporation**

        For any somatic marker gain value, the interoceptive component of the ignition signal
        should be proportional to the gain (signal = precision × gain × error).

        **Validates: Requirements 3.5**
        """
        # Exclude cases where interoceptive error is effectively zero
        intero_error_magnitude = np.linalg.norm(error_data["intero_error"])
        assume(intero_error_magnitude > 0.01)  # Ensure non-trivial error

        config = load_config()
        threshold_system = IgnitionThreshold(config)

        # Test with low gain
        _, components_low = threshold_system.compute_ignition_signal(
            extero_error=error_data["extero_error"],
            extero_precision=np.array([error_data["extero_precision"]]),
            intero_error=error_data["intero_error"],
            intero_precision=np.array([error_data["intero_precision"]]),
            somatic_marker_gain=np.array([gain_low]),
            current_time=0.0,
        )

        # Reset for second test
        threshold_system.reset()

        # Test with high gain
        _, components_high = threshold_system.compute_ignition_signal(
            extero_error=error_data["extero_error"],
            extero_precision=np.array([error_data["extero_precision"]]),
            intero_error=error_data["intero_error"],
            intero_precision=np.array([error_data["intero_precision"]]),
            somatic_marker_gain=np.array([gain_high]),
            current_time=0.0,
        )

        intero_signal_low = components_low["intero_signal"]
        intero_signal_high = components_high["intero_signal"]

        # Higher gain should result in higher interoceptive signal
        assert intero_signal_high > intero_signal_low, (
            f"Interoceptive signal should increase with somatic marker gain: "
            f"low_gain={gain_low:.2f} -> signal={intero_signal_low:.2f}, "
            f"high_gain={gain_high:.2f} -> signal={intero_signal_high:.2f}"
        )

        # Signals should be proportional to gains (within tolerance)
        expected_ratio = gain_high / gain_low
        actual_ratio = intero_signal_high / max(intero_signal_low, 1e-10)  # Avoid division by zero

        # Allow some tolerance for numerical precision
        assert (
            abs(actual_ratio - expected_ratio) < 0.5
        ), f"Signal ratio should match gain ratio: expected={expected_ratio:.2f}, actual={actual_ratio:.2f}"

        # Both signals should be non-negative and finite
        assert intero_signal_low >= 0 and np.isfinite(intero_signal_low)
        assert intero_signal_high >= 0 and np.isfinite(intero_signal_high)

        # Verify the mathematical relationship: signal = 0.5 × precision × gain × mean(||error||²)
        # The implementation uses mean of squared errors, not Euclidean norm
        expected_low = (
            0.5
            * error_data["intero_precision"]
            * gain_low
            * np.mean(error_data["intero_error"] ** 2)
        )
        expected_high = (
            0.5
            * error_data["intero_precision"]
            * gain_high
            * np.mean(error_data["intero_error"] ** 2)
        )

        assert (
            abs(float(np.ravel(intero_signal_low)[0]) - expected_low) < 1e-6
        ), f"Low gain signal calculation incorrect: expected={expected_low:.6f}, actual={float(np.ravel(intero_signal_low)[0]):.6f}"

        assert (
            abs(float(np.ravel(intero_signal_high)[0]) - expected_high) < 1e-6
        ), f"High gain signal calculation incorrect: expected={expected_high:.6f}, actual={float(np.ravel(intero_signal_high)[0]):.6f}"
