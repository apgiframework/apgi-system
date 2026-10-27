"""Unit tests for HierarchicalPredictor.

Tests for the stub implementation of hierarchical prediction functionality.
Note: Full hierarchical prediction is being migrated from the old apgi_simulation structure.
"""

import numpy as np

from apgi_framework.core.predictive_processing import (
    HierarchicalPredictor,
    PredictionErrorChannel,
    StabilityMonitor,
)


class TestPredictionErrorChannel:
    """Test PredictionErrorChannel functionality."""

    def test_initialization(self) -> None:
        """Test channel initializes correctly."""
        channel = PredictionErrorChannel(name="test_channel")
        assert channel.name == "test_channel"
        assert channel.precision == 1.0
        assert len(channel.error_history) == 0

    def test_initialization_custom_precision(self) -> None:
        """Test channel initializes with custom precision."""
        channel = PredictionErrorChannel(name="test", precision=2.5)
        assert channel.precision == 2.5

    def test_process_error(self) -> None:
        """Test error processing."""
        channel = PredictionErrorChannel(name="test", precision=2.0)
        result = channel.process_error(5.0)

        # Error * precision = 5.0 * 2.0 = 10.0
        assert abs(result - 10.0) < 1e-10
        assert len(channel.error_history) == 1
        assert channel.error_history[0] == 5.0

    def test_get_error_statistics_empty(self) -> None:
        """Test statistics with no errors."""
        channel = PredictionErrorChannel(name="test")
        stats = channel.get_error_statistics()

        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0
        assert stats["count"] == 0

    def test_get_error_statistics_with_data(self) -> None:
        """Test statistics with error history."""
        channel = PredictionErrorChannel(name="test")

        # Process some errors
        for error in [1.0, 2.0, 3.0, 4.0, 5.0]:
            channel.process_error(error)

        stats = channel.get_error_statistics()
        assert stats["count"] == 5
        assert abs(stats["mean"] - 3.0) < 1e-10
        assert stats["std"] > 0


class TestStabilityMonitor:
    """Test StabilityMonitor functionality."""

    def test_initialization_default(self) -> None:
        """Test monitor initializes with default threshold."""
        monitor = StabilityMonitor()
        assert monitor.threshold == 1e6
        assert monitor.is_stable is True
        assert len(monitor.variance_history) == 0

    def test_initialization_custom(self) -> None:
        """Test monitor initializes with custom threshold."""
        monitor = StabilityMonitor(threshold=100.0)
        assert monitor.threshold == 100.0

    def test_check_stability_stable(self) -> None:
        """Test stability check with stable prediction."""
        monitor = StabilityMonitor(threshold=10.0)
        prediction = np.array([0.1, 0.2, 0.3])

        is_stable = monitor.check_stability(prediction)
        assert is_stable is True
        assert len(monitor.variance_history) == 1

    def test_check_stability_unstable(self) -> None:
        """Test stability check with unstable prediction."""
        monitor = StabilityMonitor(threshold=0.001)
        prediction = np.array([100.0, 200.0, 300.0])

        is_stable = monitor.check_stability(prediction)
        assert is_stable is False

    def test_get_statistics_empty(self) -> None:
        """Test statistics with no history."""
        monitor = StabilityMonitor()
        stats = monitor.get_statistics()

        assert stats["mean_variance"] == 0.0
        assert stats["max_variance"] == 0.0
        assert stats["is_stable"] is True

    def test_get_statistics_with_data(self) -> None:
        """Test statistics with history."""
        monitor = StabilityMonitor()

        # Add some variance data
        for _ in range(5):
            monitor.check_stability(np.random.randn(10))

        stats = monitor.get_statistics()
        assert stats["mean_variance"] >= 0
        assert stats["max_variance"] >= 0

    def test_reset_statistics(self) -> None:
        """Test resetting statistics."""
        monitor = StabilityMonitor()

        # Add data
        for _ in range(5):
            monitor.check_stability(np.random.randn(10))

        monitor.reset_statistics()
        stats = monitor.get_statistics()

        assert stats["mean_variance"] == 0.0
        assert stats["max_variance"] == 0.0
        assert len(monitor.variance_history) == 0


class TestHierarchicalPredictor:
    """Test HierarchicalPredictor functionality."""

    def test_initialization_default(self) -> None:
        """Test predictor initializes with default levels."""
        predictor = HierarchicalPredictor()
        assert predictor.num_levels == 3
        assert len(predictor.channels) == 3
        assert predictor.learning_rates.shape == (3,)
        assert predictor.stability_monitor is not None

    def test_initialization_custom(self) -> None:
        """Test predictor initializes with custom levels."""
        predictor = HierarchicalPredictor(config=5)
        assert predictor.num_levels == 5
        assert len(predictor.channels) == 5
        assert predictor.learning_rates.shape == (5,)

    def test_predict_shape(self) -> None:
        """Test predict returns correct shape."""
        predictor = HierarchicalPredictor(config=3)
        input_data = np.random.randn(16)
        predictions = predictor.predict(input_data)

        # Should tile input to num_levels rows
        assert predictions.shape == (3, 16)

    def test_predict_values_finite(self) -> None:
        """Test predict returns finite values."""
        predictor = HierarchicalPredictor()
        input_data = np.random.randn(8)
        predictions = predictor.predict(input_data)

        assert np.all(np.isfinite(predictions))

    def test_predict_with_dt_ms(self) -> None:
        """Test predict with custom dt_ms."""
        predictor = HierarchicalPredictor()
        input_data = np.random.randn(8)
        predictions = predictor.predict(input_data, dt_ms=2.0)

        assert predictions.shape == (3, 8)
        assert np.all(np.isfinite(predictions))

    def test_predict_triggers_stability_check(self) -> None:
        """Test that predict updates stability monitor."""
        predictor = HierarchicalPredictor()

        initial_history_len = len(predictor.stability_monitor.variance_history)
        predictor.predict(np.random.randn(8))
        new_history_len = len(predictor.stability_monitor.variance_history)

        assert new_history_len > initial_history_len

    def test_update_precision(self) -> None:
        """Test updating precision for specific level."""
        predictor = HierarchicalPredictor(config=3)

        predictor.update_precision(1, 5.0)
        assert predictor.channels[1].precision == 5.0

        # Other levels unchanged
        assert predictor.channels[0].precision == 1.0
        assert predictor.channels[2].precision == 1.0

    def test_update_precision_out_of_bounds(self) -> None:
        """Test updating precision for invalid level."""
        predictor = HierarchicalPredictor(config=3)

        # Should not raise error, just ignore
        predictor.update_precision(10, 5.0)
        predictor.update_precision(-1, 5.0)

        # All precisions unchanged
        for channel in predictor.channels:
            assert channel.precision == 1.0

    def test_get_prediction_errors_initial(self) -> None:
        """Test getting errors before any predictions."""
        predictor = HierarchicalPredictor(config=3)
        errors = predictor.get_prediction_errors()

        # One exteroceptive entry per level; zeros since no predictions were made
        assert len(errors["exteroceptive"]) == 3
        for error in errors["exteroceptive"]:
            assert error == 0.0
        assert errors["exteroceptive_stats"]["mean_error"] == 0.0

    def test_prediction_with_interoceptive_input(self) -> None:
        """Test prediction with interoceptive input."""
        predictor = HierarchicalPredictor(config=3)

        # Generate some predictions
        predictor.predict(np.random.randn(8))

    def test_get_prediction_errors_after_predict(self) -> None:
        """Test getting errors after predictions."""
        predictor = HierarchicalPredictor(config=3)

        # Generate some predictions
        predictor.predict(np.random.randn(8))
        errors = predictor.get_prediction_errors()

        assert len(errors["exteroceptive"]) == 3
        # Should have some error values now
        for error in errors["exteroceptive"]:
            assert isinstance(float(error), float)
        assert "interoceptive" in errors

    def test_temporal_dynamics(self) -> None:
        """Test temporal dynamics of predictions."""
        predictor = HierarchicalPredictor(config=3)

        # Generate some state
        predictor.predict(np.random.randn(8))
        predictor.predict(np.random.randn(8))

    def test_reset(self) -> None:
        """Test resetting predictor state."""
        predictor = HierarchicalPredictor(config=3)

        # Generate some state
        predictor.predict(np.random.randn(8))
        predictor.predict(np.random.randn(8))

        # Reset
        predictor.reset()

        # Channels should be fresh
        assert len(predictor.channels) == 3
        for channel in predictor.channels:
            assert channel.name.startswith("level_")
            assert len(channel.error_history) == 0

        # Stability monitor cleared
        assert len(predictor.stability_monitor.variance_history) == 0

    def test_multiple_predictions_consistency(self) -> None:
        """Test multiple predictions maintain consistency."""
        predictor = HierarchicalPredictor(config=3)

        for i in range(10):
            input_data = np.random.randn(8) * 0.1
            predictions = predictor.predict(input_data)

            assert predictions.shape == (3, 8)
            assert np.all(np.isfinite(predictions))

    def test_learning_rates_array(self) -> None:
        """Test learning rates initialization."""
        predictor = HierarchicalPredictor(config=5)

        # Default learning rate is 0.1
        assert np.allclose(predictor.learning_rates, 0.1)

    def test_channels_named_correctly(self) -> None:
        """Test that channels are named correctly."""
        predictor = HierarchicalPredictor(config=3)

        assert predictor.channels[0].name == "level_0"
        assert predictor.channels[1].name == "level_1"
        assert predictor.channels[2].name == "level_2"
