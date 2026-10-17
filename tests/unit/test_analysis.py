"""Unit tests for extended analysis capabilities."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Import from analysis.py file directly to avoid conflict with analysis/ package
spec = importlib.util.spec_from_file_location(
    "analysis_module", Path(__file__).resolve().parents[2] / "apgi_framework" / "analysis.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Failed to load analysis module")
analysis_module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(analysis_module)  # type: ignore[union-attr]

AnalysisResults = analysis_module.AnalysisResults
SystemAnalyzer = analysis_module.SystemAnalyzer
analyze_simulation_run = analysis_module.analyze_simulation_run
from apgi_framework.system import APGISystem


def test_analysis_results_dataclass():
    """Test AnalysisResults dataclass initialization."""
    results = AnalysisResults()

    assert isinstance(results.ignition_statistics, dict)
    assert isinstance(results.energy_budget_summary, dict)
    assert isinstance(results.somatic_marker_stats, dict)
    assert isinstance(results.coherence_metrics, dict)
    assert isinstance(results.temporal_dynamics, dict)


def test_analysis_results_with_data() -> None:
    """Test AnalysisResults with actual data."""
    results = AnalysisResults(
        ignition_statistics={"total_ignitions": 10, "ignition_rate_hz": 2.0},
        energy_budget_summary={"total_energy_consumed": 0.5},
        somatic_marker_stats={"total_markers": 5},
        coherence_metrics={"mean_coherence": 0.85},
        temporal_dynamics={"time": [0.0, 1.0, 2.0]},
    )

    assert results.ignition_statistics["total_ignitions"] == 10
    assert results.energy_budget_summary["total_energy_consumed"] == 0.5
    assert results.somatic_marker_stats["total_markers"] == 5
    assert results.coherence_metrics["mean_coherence"] == 0.85
    assert len(results.temporal_dynamics["time"]) == 3


def test_system_analyzer_initialization(config: dict) -> None:
    """Test SystemAnalyzer initialization."""
    analyzer = SystemAnalyzer(config)

    assert analyzer.config == config


def test_compute_ignition_statistics_empty(config: dict) -> None:
    """Test ignition statistics computation with no ignitions."""
    analyzer = SystemAnalyzer(config)
    system = APGISystem()

    stats = analyzer.compute_ignition_statistics(system.ignition_threshold, system.history)

    assert stats["total_ignitions"] == 0
    assert stats["ignition_rate_hz"] == 0.0
    assert stats["mean_ignition_interval_ms"] == 0.0


def test_compute_ignition_statistics_with_data(config: dict) -> None:
    """Test ignition statistics computation with actual data."""
    analyzer = SystemAnalyzer(config)
    system = APGISystem()

    # Simulate some ignition events
    system.ignition_threshold.recent_ignitions.append(100.0)
    system.ignition_threshold.recent_ignitions.append(500.0)
    system.ignition_threshold.recent_ignitions.append(1000.0)

    # Add time history
    system.history["time"] = [0.0, 100.0, 500.0, 1000.0, 1500.0]

    stats = analyzer.compute_ignition_statistics(system.ignition_threshold, system.history)

    assert stats["total_ignitions"] == 3
    assert stats["ignition_rate_hz"] > 0.0
    assert stats["mean_ignition_interval_ms"] > 0.0
    assert stats["min_ignition_interval_ms"] > 0.0
    assert stats["max_ignition_interval_ms"] > 0.0


def test_compute_energy_budget_summary_empty(config: dict) -> None:
    """Test energy budget summary with no history."""
    analyzer = SystemAnalyzer(config)
    system = APGISystem()

    summary = analyzer.compute_energy_budget_summary(system.metabolism, system.history)

    assert summary["total_energy_consumed"] == 0.0
    assert summary["final_reserves"] == 1.0
    assert summary["min_reserves"] == 1.0


def test_compute_energy_budget_summary_with_data(config: dict) -> None:
    """Test energy budget summary with actual data."""
    analyzer = SystemAnalyzer(config)
    system = APGISystem()

    # Simulate energy consumption
    system.history["metabolic_reserves"] = [1.0, 0.95, 0.90, 0.85, 0.80]
    system.history["ignitions"] = [0, 1, 0, 1, 0]
    system.history["time"] = [0.0, 100.0, 200.0, 300.0, 400.0]

    summary = analyzer.compute_energy_budget_summary(system.metabolism, system.history)

    assert summary["total_energy_consumed"] == pytest.approx(0.2, abs=0.01)
    assert summary["final_reserves"] == 0.80
    assert summary["min_reserves"] == 0.80
    assert summary["mean_energy_per_step"] > 0.0
    assert summary["energy_per_ignition"] > 0.0


def test_compute_somatic_marker_statistics(config: dict) -> None:
    """Test somatic marker statistics computation."""
    analyzer = SystemAnalyzer(config)
    system = APGISystem()

    # Add some markers
    context = np.random.randn(10)
    action = np.random.randn(5)
    system.somatic_markers.learn(context, action, body_outcome=np.array([0.5]), current_time=0.0)
    system.somatic_markers.learn(context, action, body_outcome=np.array([0.7]), current_time=100.0)

    stats = analyzer.compute_somatic_marker_statistics(system.somatic_markers)

    assert stats["total_markers"] > 0
    assert 0.0 <= stats["capacity_used"] <= 1.0
    assert 0.0 <= stats["retrieval_success_rate"] <= 1.0
    assert stats["mean_marker_strength"] > 0.0


def test_compute_coherence_metrics(config: dict) -> None:
    """Test coherence metrics computation."""
    analyzer = SystemAnalyzer(config)
    system = APGISystem()

    metrics = analyzer.compute_coherence_metrics(system.coherence)

    assert "mean_coherence" in metrics
    assert "current_coherence" in metrics
    assert "phenomenal_unity" in metrics
    assert 0.0 <= metrics["mean_coherence"] <= 1.0


def test_extract_temporal_dynamics(config: dict) -> None:
    """Test temporal dynamics extraction."""
    analyzer = SystemAnalyzer(config)
    system = APGISystem()

    # Add some history
    system.history["time"] = [0.0, 1.0, 2.0, 3.0]
    system.history["free_energy"] = [1.0, 0.9, 0.8, 0.7]
    system.history["precision"] = [1.0, 1.1, 1.2, 1.3]
    system.history["metabolic_reserves"] = [1.0, 0.99, 0.98, 0.97]

    # Add signal history (numpy arrays as expected by signal_history type)
    system.ignition_threshold.signal_history = [
        np.array([2.0]),
        np.array([2.5]),
        np.array([3.0]),
        np.array([2.8]),
    ]
    # Also add threshold history
    system.ignition_threshold.threshold_history = [2.0, 2.0, 2.0, 2.0]

    dynamics = analyzer.extract_temporal_dynamics(system.history, system.ignition_threshold)

    assert len(dynamics["time"]) == 4
    assert len(dynamics["free_energy"]) == 4
    assert len(dynamics["precision"]) == 4
    assert len(dynamics["metabolic_reserves"]) == 4
    # signal_history contains arrays, so ignition_signal will be a list of arrays
    assert len(dynamics["ignition_signal"]) == 4
    assert len(dynamics["threshold"]) == 4


def test_analyze_system_integration(config: dict) -> None:
    """Test full system analysis integration."""
    system = APGISystem()

    # Run a short simulation
    system.run(duration_ms=100.0)

    analyzer = SystemAnalyzer(system.config)
    results = analyzer.analyze_system(system)

    # Verify all components are present
    assert isinstance(results, AnalysisResults)
    assert isinstance(results.ignition_statistics, dict)
    assert isinstance(results.energy_budget_summary, dict)
    assert isinstance(results.somatic_marker_stats, dict)
    assert isinstance(results.coherence_metrics, dict)
    assert isinstance(results.temporal_dynamics, dict)

    # Verify some basic properties
    assert results.ignition_statistics["total_ignitions"] >= 0
    assert results.energy_budget_summary["final_reserves"] >= 0.0  # Reserves are in absolute units
    assert results.somatic_marker_stats["total_markers"] >= 0


def test_analyze_simulation_run_convenience_function():
    """Test the convenience function for analysis."""
    system = APGISystem()

    # Run a short simulation
    system.run(duration_ms=100.0)

    results = analyze_simulation_run(system)

    # Verify results
    assert isinstance(results, AnalysisResults)
    assert "total_ignitions" in results.ignition_statistics
    assert "total_energy_consumed" in results.energy_budget_summary
    assert "total_markers" in results.somatic_marker_stats
    assert "mean_coherence" in results.coherence_metrics
    assert "time" in results.temporal_dynamics


def test_analysis_with_longer_simulation():
    """Test analysis with a longer simulation run."""
    system = APGISystem()

    # Run a longer simulation to get more interesting data
    system.run(duration_ms=1000.0)

    results = analyze_simulation_run(system)

    # Verify we have meaningful data
    assert len(results.temporal_dynamics["time"]) > 0
    assert results.energy_budget_summary["total_energy_consumed"] >= 0.0

    # If there were ignitions, verify statistics
    if results.ignition_statistics["total_ignitions"] > 0:
        assert results.ignition_statistics["ignition_rate_hz"] > 0.0
        assert results.energy_budget_summary["energy_per_ignition"] > 0.0
