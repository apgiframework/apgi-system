"""Integration tests for numerical stability with core components.

These tests verify that the stub implementations handle basic operations correctly.
Note: Full stability monitoring is being migrated from the old apgi_simulation structure.
"""

import numpy as np
import pytest

from apgi_framework.core.free_energy import FreeEnergyCalculator
from apgi_framework.core.predictive_processing import HierarchicalPredictor


class TestFreeEnergyCalculator:
    """Test FreeEnergyCalculator functionality."""

    def test_initialization_defaults(self) -> None:
        """Test calculator initializes with default values."""
        calc = FreeEnergyCalculator()
        assert calc.precision_exteroceptive == 1.0
        assert calc.precision_interoceptive == 1.0
        assert calc.prediction_error_weight == 1.0

    def test_initialization_custom(self) -> None:
        """Test calculator initializes with custom values."""
        calc = FreeEnergyCalculator(
            config={
                "precision_exteroceptive": 2.0,
                "precision_interoceptive": 3.0,
                "prediction_error_weight": 0.5,
            }
        )
        assert calc.precision_exteroceptive == 2.0
        assert calc.precision_interoceptive == 3.0
        assert calc.prediction_error_weight == 0.5

    def test_calculate_free_energy(self) -> None:
        """Test basic free energy calculation."""
        calc = FreeEnergyCalculator()
        prediction_error = np.array([1.0, 2.0, 3.0])
        fe = calc.calculate_free_energy(prediction_error)

        assert isinstance(fe, float)
        assert fe >= 0
        assert np.isfinite(fe)

    def test_calculate_free_energy_with_prior(self) -> None:
        """Test free energy calculation with prior belief."""
        calc = FreeEnergyCalculator()
        prediction_error = np.array([1.0, 1.0, 1.0])
        prior_belief = np.array([0.5, 0.5, 0.5])
        fe = calc.calculate_free_energy(prediction_error, prior_belief)

        assert isinstance(fe, float)
        assert fe >= 0
        assert np.isfinite(fe)

    def test_calculate_surprise(self) -> None:
        """Test surprise calculation."""
        calc = FreeEnergyCalculator()
        prediction_error = np.array([2.0, 0.0, 0.0])
        surprise = calc.calculate_surprise(prediction_error)

        assert isinstance(surprise, float)
        assert surprise >= 0
        assert np.isfinite(surprise)
        assert abs(surprise - 4.0) < 1e-10

    def test_get_free_energy_components(self) -> None:
        """Test retrieving free energy components."""
        calc = FreeEnergyCalculator(
            config={
                "precision_exteroceptive": 2.5,
                "precision_interoceptive": 1.5,
                "prediction_error_weight": 0.8,
            }
        )
        components = calc.get_free_energy_components()

        assert isinstance(components, dict)
        assert components["precision_exteroceptive"] == 2.5
        assert components["precision_interoceptive"] == 1.5
        assert components["prediction_error_weight"] == 0.8

    def test_compute_variational_free_energy(self) -> None:
        """Test variational free energy computation."""
        calc = FreeEnergyCalculator(config={"precision_exteroceptive": 2.0})
        observation = np.array([1.0, 2.0, 3.0])
        prediction = np.array([1.5, 2.5, 3.5])
        precision = 2.0
        posterior_mean = np.array([1.0, 2.0, 3.0])
        posterior_cov = np.eye(3)
        prior_mean = np.array([0.0, 0.0, 0.0])
        prior_cov = np.eye(3) * 2
        fe, components = calc.compute_variational_free_energy(
            observation, prediction, precision, posterior_mean, posterior_cov, prior_mean, prior_cov
        )

        assert isinstance(fe, float)
        assert fe >= 0
        assert fe == pytest.approx(components["accuracy"] + components["complexity"])

    def test_stability_monitor_property(self) -> None:
        """Test stability_monitor property returns expected structure."""
        calc = FreeEnergyCalculator()
        monitor = calc.stability_monitor

        assert isinstance(monitor, dict)
        assert "stable" in monitor
        assert "variance" in monitor
        assert monitor["stable"] is True


class TestHierarchicalPredictorStability:
    """Test HierarchicalPredictor stability characteristics."""

    def test_initialization_default(self) -> None:
        """Test predictor initializes with default levels."""
        predictor = HierarchicalPredictor(3)
        assert predictor.num_levels == 3
        assert len(predictor.channels) == 3

    def test_initialization_custom(self) -> None:
        """Test predictor initializes with custom levels."""
        predictor = HierarchicalPredictor(5)
        assert predictor.num_levels == 5
        assert len(predictor.channels) == 5

    def test_predict_returns_array(self) -> None:
        """Test predict returns array of predictions."""
        predictor = HierarchicalPredictor(3)
        input_data = np.random.randn(16)
        predictions = predictor.predict(input_data)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == 3

    def test_predict_with_dt(self) -> None:
        """Test predict with custom dt_ms."""
        predictor = HierarchicalPredictor(3)
        input_data = np.random.randn(8)
        predictions = predictor.predict(input_data, dt_ms=2.0)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == 3

    def test_update_precision(self) -> None:
        """Test precision update for specific level."""
        predictor = HierarchicalPredictor(3)
        predictor.update_precision(1, 5.0)

        assert predictor.channels[1].precision == 5.0

    def test_get_prediction_errors(self) -> None:
        """Test retrieving prediction errors."""
        predictor = HierarchicalPredictor(3)
        input_data = np.random.randn(8)
        predictor.predict(input_data)

        errors = predictor.get_prediction_errors()
        assert isinstance(errors, dict)
        assert len(errors["exteroceptive"]) == 3

    def test_stability_monitor_tracks_predictions(self) -> None:
        """Test that stability monitor tracks predictions."""
        predictor = HierarchicalPredictor(3)

        for _ in range(5):
            input_data = np.random.randn(8) * 0.1
            predictor.predict(input_data)

        assert len(predictor.stability_monitor.variance_history) > 0

    def test_stability_monitor_check_stability(self) -> None:
        """Test stability check returns boolean."""
        predictor = HierarchicalPredictor(3)
        input_data = np.random.randn(8)
        predictions = predictor.predict(input_data)

        is_stable = predictor.stability_monitor.check_stability(predictions)
        assert isinstance(is_stable, bool)

    def test_reset_clears_state(self) -> None:
        """Test that reset clears predictor state."""
        predictor = HierarchicalPredictor(3)

        for _ in range(5):
            predictor.predict(np.random.randn(8))

        predictor.reset()

        assert len(predictor.channels) == 3
        for channel in predictor.channels:
            assert len(channel.error_history) == 0

    def test_predict_with_stable_input(self) -> None:
        """Test prediction with small, stable inputs."""
        predictor = HierarchicalPredictor()

        for _ in range(10):
            input_data = np.random.randn(8) * 0.01
            predictions = predictor.predict(input_data)
            assert np.all(np.isfinite(predictions))

    def test_get_statistics(self) -> None:
        """Test stability monitor statistics."""
        predictor = HierarchicalPredictor()

        for _ in range(5):
            predictor.predict(np.random.randn(8))

        stats = predictor.stability_monitor.get_statistics()
        assert isinstance(stats, dict)
        assert "mean_variance" in stats
        assert "max_variance" in stats
        assert "is_stable" in stats

    def test_reset_statistics(self) -> None:
        """Test resetting stability statistics."""
        predictor = HierarchicalPredictor()

        for _ in range(5):
            predictor.predict(np.random.randn(8))

        predictor.stability_monitor.reset_statistics()
        stats = predictor.stability_monitor.get_statistics()

        assert stats["mean_variance"] == 0.0
        assert stats["max_variance"] == 0.0
