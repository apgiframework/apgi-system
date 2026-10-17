"""
Extended tests for analysis module to improve coverage.

Tests the comprehensive analysis capabilities including system analysis,
results processing, and reporting functionality.
"""

import importlib.util
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np

# Import from analysis.py file directly to avoid conflict with analysis/ package
spec = importlib.util.spec_from_file_location(
    "analysis_module", Path(__file__).resolve().parents[2] / "apgi_framework" / "analysis.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Could not load analysis module")
analysis_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis_module)

AnalysisResults = analysis_module.AnalysisResults
SystemAnalyzer = analysis_module.SystemAnalyzer
analyze_simulation_run = analysis_module.analyze_simulation_run


class TestAnalysisResults:
    """Test the AnalysisResults dataclass."""

    def test_analysis_results_initialization(self) -> None:
        """Test that AnalysisResults can be initialized with default values."""
        results = AnalysisResults()

        assert isinstance(results.ignition_statistics, dict)
        assert isinstance(results.energy_budget_summary, dict)
        assert isinstance(results.somatic_marker_stats, dict)
        assert isinstance(results.coherence_metrics, dict)
        assert isinstance(results.temporal_dynamics, dict)

    def test_analysis_results_with_data(self) -> None:
        """Test AnalysisResults with actual data."""
        ignition_stats = {
            "total_ignitions": 10,
            "ignition_rate_hz": 0.5,
            "mean_ignition_interval_ms": 2000.0,
        }

        energy_stats = {
            "total_energy_consumed": 100.0,
            "mean_energy_per_step": 0.1,
            "final_reserves": 0.8,
        }

        temporal_data = {
            "time": [1000.0, 2000.0, 3000.0],
            "ignition_signal": [1.0, 2.0, 1.5],
            "free_energy": [1.2, 1.8, 1.5],
        }

        results = AnalysisResults(
            ignition_statistics=ignition_stats,
            energy_budget_summary=energy_stats,
            temporal_dynamics=temporal_data,
        )

        assert results.ignition_statistics["total_ignitions"] == 10
        assert results.energy_budget_summary["final_reserves"] == 0.8
        assert len(results.temporal_dynamics["time"]) == 3


class TestSystemAnalyzer:
    """Test the SystemAnalyzer class."""

    def test_analyzer_initialization(self) -> None:
        """Test that SystemAnalyzer initializes correctly."""
        config = {
            "analysis_window_ms": 1000.0,
            "ignition_threshold": 2.0,
            "enable_detailed_analysis": True,
        }

        analyzer = SystemAnalyzer(config)

        assert analyzer.config == config
        assert hasattr(analyzer, "config")

    def test_analyzer_default_config(self) -> None:
        """Test SystemAnalyzer with default configuration."""
        analyzer = SystemAnalyzer({})

        assert isinstance(analyzer.config, dict)

    def test_analyze_system_basic(self) -> None:
        """Test basic system analysis."""
        config: Dict[str, Any] = {"analysis_window_ms": 5000.0}
        analyzer = SystemAnalyzer(config)

        # Mock system with basic state
        mock_system = MagicMock()
        mock_system.get_state.return_value = {
            "ignition": {
                "ignition_occurred": True,
                "total_signal": 2.5,
                "ignition_times": [1000.0, 3000.0],
                "ignition_durations": [100.0, 150.0],
            },
            "metabolism": {
                "energy_consumed": 0.2,
                "current_reserves": 0.8,
                "energy_history": [0.1, 0.2, 0.15],
                "reserve_history": [1.0, 0.9, 0.8],
            },
            "time": 5000.0,
        }

        # Mock history data
        mock_system.get_history.return_value = {
            "ignition_signals": [1.0, 2.5, 1.8, 0.5],
            "free_energy_values": [1.2, 1.8, 1.5, 1.3],
            "timestamps": [1000.0, 2000.0, 3000.0, 4000.0],
        }

        results = analyzer.analyze_system(mock_system)

        assert isinstance(results, AnalysisResults)
        assert isinstance(results.ignition_statistics, dict)
        assert isinstance(results.energy_budget_summary, dict)

    def test_compute_ignition_statistics(self) -> None:
        """Test ignition statistics computation."""
        config: Dict[str, Any] = {}
        analyzer = SystemAnalyzer(config)

        # Mock ignition data
        ignition_times: List[float] = [1000.0, 3000.0, 7000.0]
        ignition_durations: List[float] = [50.0, 75.0, 100.0]
        total_time = 10000.0

        stats = analyzer._compute_ignition_statistics(
            ignition_times, ignition_durations, total_time
        )

        assert stats["total_ignitions"] == 3
        assert stats["ignition_rate_hz"] == 0.3  # type: ignore[comparison-overlap]  # 3 events in 10 seconds
        assert stats["mean_ignition_duration_ms"] == 75.0  # (50+75+100)/3

    def test_compute_ignition_statistics_empty(self) -> None:
        """Test ignition statistics with no ignition events."""
        config: Dict[str, Any] = {}
        analyzer = SystemAnalyzer(config)

        ignition_times: List[float] = []
        ignition_durations: List[float] = []
        total_time = 10000.0

        stats = analyzer._compute_ignition_statistics(
            ignition_times, ignition_durations, total_time
        )

        assert stats["total_ignitions"] == 0
        assert stats["ignition_rate_hz"] == 0.0
        assert stats["mean_ignition_duration_ms"] == 0.0

    def test_compute_energy_budget_summary(self) -> None:
        """Test energy budget summary computation."""
        config: Dict[str, Any] = {}
        analyzer = SystemAnalyzer(config)

        energy_history = [0.1, 0.2, 0.15, 0.3, 0.25]  # Total = 1.0
        reserve_history = [1.0, 0.9, 0.7, 0.4, 0.15]
        ignition_count = 2

        summary = analyzer._compute_energy_budget_summary(
            energy_history, reserve_history, ignition_count
        )

        assert summary["total_energy_consumed"] == 1.0
        assert summary["mean_energy_per_step"] == 0.2  # 1.0 / 5
        assert summary["energy_per_ignition"] == 0.5  # 1.0 / 2
        assert summary["final_reserves"] == 0.15
        assert summary["min_reserves"] == 0.15

    def test_compute_coherence_metrics(self) -> None:
        """Test coherence metrics computation."""
        config: Dict[str, Any] = {}
        analyzer = SystemAnalyzer(config)

        coherence_history = [0.8, 0.6, 0.9, 0.7, 0.85]

        metrics = analyzer._compute_coherence_metrics(coherence_history)

        assert metrics["mean_coherence"] == 0.77  # (0.8+0.6+0.9+0.7+0.85)/5
        assert metrics["min_coherence"] == 0.6
        assert metrics["max_coherence"] == 0.9
        assert "std_coherence" in metrics

    def test_generate_report(self) -> None:
        """Test analysis report generation."""
        config: Dict[str, Any] = {}
        analyzer = SystemAnalyzer(config)

        results = AnalysisResults(
            ignition_statistics={
                "total_ignitions": 5,
                "ignition_rate_hz": 0.25,
                "mean_ignition_duration_ms": 120.0,
            },
            energy_budget_summary={
                "total_energy_consumed": 2.5,
                "final_reserves": 0.6,
                "energy_per_ignition": 0.5,
            },
            coherence_metrics={"mean_coherence": 0.75, "std_coherence": 0.1},
        )

        report = analyzer.generate_report(results)

        assert isinstance(report, str)
        assert "APGI System Analysis Report" in report
        assert "5" in report  # Should contain ignition count
        assert "0.25" in report  # Should contain ignition rate

    def test_analyze_temporal_patterns(self) -> None:
        """Test temporal pattern analysis."""
        config: Dict[str, Any] = {}
        analyzer = SystemAnalyzer(config)

        results = AnalysisResults(
            ignition_statistics={"total_ignitions": 3},
            energy_budget_summary={"total_energy_consumed": 1.5},
            temporal_dynamics={"time": [1000.0, 2000.0], "ignition_signal": [1.0, 2.0]},
        )

        # Mock ignition_threshold object with required attributes
        mock_ignition_threshold = MagicMock()
        mock_ignition_threshold.signal_history = [1.0, 2.0]
        mock_ignition_threshold.threshold_history = [1.5, 1.5]

        # Test temporal pattern analysis
        temporal_patterns = analyzer.extract_temporal_dynamics(
            results.temporal_dynamics, ignition_threshold=mock_ignition_threshold
        )

        assert isinstance(temporal_patterns, dict)
        assert "time" in temporal_patterns
        assert "ignition_signal" in temporal_patterns

    def test_export_results(self) -> None:
        """Test exporting results to different formats."""
        config: Dict[str, Any] = {}
        analyzer = SystemAnalyzer(config)

        results = AnalysisResults(
            ignition_statistics={"total_ignitions": 3},
            energy_budget_summary={"total_energy_consumed": 1.5},
            temporal_dynamics={"time": [1000.0, 2000.0], "ignition_signal": [1.0, 2.0]},
        )

        # Test JSON export
        json_data = analyzer.export_json(results)
        assert isinstance(json_data, str)
        assert "total_ignitions" in json_data

        # Test CSV export
        csv_data = analyzer.export_csv(results)
        assert isinstance(csv_data, str)
        assert "time" in csv_data


class TestAnalyzeSimulationRun:
    """Test the analyze_simulation_run function."""

    def test_analyze_simulation_run_basic(self) -> None:
        """Test basic simulation run analysis."""
        # Mock system
        mock_system = MagicMock()
        mock_system.get_state.return_value = {
            "ignition": {"ignition_times": [1000.0, 2000.0], "ignition_durations": [100.0, 120.0]},
            "metabolism": {"energy_history": [0.1, 0.2], "reserve_history": [1.0, 0.9]},
            "time": 3000.0,
        }

        mock_system.get_history.return_value = {
            "ignition_signals": [1.0, 2.0],
            "timestamps": [1000.0, 2000.0],
        }

        results = analyze_simulation_run(mock_system)

        assert isinstance(results, AnalysisResults)
        assert isinstance(results.ignition_statistics, dict)

    def test_analyze_simulation_run_empty_system(self) -> None:
        """Test analysis with empty system state."""
        mock_system = MagicMock()
        mock_system.get_state.return_value = {
            "ignition": {"ignition_times": [], "ignition_durations": []},
            "metabolism": {"energy_history": [], "reserve_history": []},
            "time": 0.0,
        }

        mock_system.get_history.return_value = {"ignition_signals": [], "timestamps": []}

        results = analyze_simulation_run(mock_system)

        assert isinstance(results, AnalysisResults)
        # Should handle empty data gracefully


class TestAnalysisIntegration:
    """Integration tests for analysis functionality."""

    def test_full_analysis_pipeline(self) -> None:
        """Test the complete analysis pipeline."""
        config = {"analysis_window_ms": 10000.0, "enable_detailed_analysis": True}
        analyzer = SystemAnalyzer(config)

        # Create comprehensive mock system with required attributes
        mock_system = MagicMock()
        mock_system.config = config

        # Mock ignition_threshold with recent_ignitions (list of time values)
        mock_ignition_threshold = MagicMock()
        mock_ignition_threshold.recent_ignitions = [1000.0, 2500.0, 4200.0, 6800.0]
        mock_ignition_threshold.signal_history = [1.0 + 0.5 * np.sin(i * 0.2) for i in range(50)]
        mock_ignition_threshold.threshold_history = [2.0] * 50
        mock_system.ignition_threshold = mock_ignition_threshold

        # Mock metabolism with proper energy history
        mock_metabolism = MagicMock()
        mock_metabolism.energy_history = [0.1 + 0.05 * np.random.randn() for _ in range(50)]
        mock_metabolism.reserve_history = [1.0 - i * 0.02 for i in range(50)]
        mock_system.metabolism = mock_metabolism

        # Mock somatic_markers
        mock_somatic_markers = MagicMock()
        mock_somatic_markers.get_statistics.return_value = {
            "num_markers": 10,
            "capacity_used": 0.2,
            "retrieval_success_rate": 0.8,
            "avg_strength": 0.7,
            "avg_outcome": 0.5,
        }
        mock_system.somatic_markers = mock_somatic_markers

        # Mock coherence
        mock_coherence = MagicMock()
        mock_coherence.get_coherence_metrics.return_value = {
            "overall_coherence": 0.75,
            "phenomenal_unity": True,
        }
        mock_system.coherence = mock_coherence

        # Mock history data
        mock_system.history = {
            "time": [i * 200.0 for i in range(50)],
            "ignition_signals": [1.0 + 0.5 * np.sin(i * 0.2) for i in range(50)],
            "free_energy": [1.5 + 0.3 * np.cos(i * 0.15) for i in range(50)],
            "precision": [0.8 + 0.1 * np.sin(i * 0.1) for i in range(50)],
            "metabolic_reserves": [1.0 - i * 0.02 for i in range(50)],
            "ignitions": [1, 0, 1, 0, 1, 0, 1, 0] + [0] * 42,
        }

        # Perform analysis
        results = analyzer.analyze_system(mock_system)

        # Verify comprehensive results
        assert isinstance(results, AnalysisResults)
        assert len(results.ignition_statistics) > 0
        assert len(results.energy_budget_summary) > 0
        assert len(results.temporal_dynamics) > 0

        # Generate report
        report = analyzer.generate_report(results)
        assert len(report) > 500  # Should be substantial

        # Export data
        json_export = analyzer.export_json(results)  # type: ignore[attr-defined]
        csv_export = analyzer.export_csv(results)  # type: ignore[attr-defined]

        assert len(json_export) > 100
        assert len(csv_export) > 100

        # Verify key metrics are reasonable
        if "total_ignitions" in results.ignition_statistics:
            assert results.ignition_statistics["total_ignitions"] == 4

    def test_analysis_with_missing_data(self) -> None:
        """Test analysis robustness with missing data."""
        config: dict[str, Any] = {}
        analyzer = SystemAnalyzer(config)

        # Mock system with incomplete data
        mock_system = MagicMock()
        mock_system.get_state.return_value = {
            "ignition": {"ignition_times": [1000.0]},  # Missing durations
            "metabolism": {"energy_history": [0.1, 0.2]},  # Missing reserves
            "time": 2000.0,
        }

        mock_system.get_history.return_value = {
            "timestamps": [1000.0, 2000.0]
            # Missing other history data
        }

        # Should handle missing data gracefully
        results = analyzer.analyze_system(mock_system)

        assert isinstance(results, AnalysisResults)
        # Should not raise exceptions even with incomplete data

    def test_performance_with_large_dataset(self) -> None:
        """Test analysis performance with large datasets."""
        config: dict[str, Any] = {}
        analyzer = SystemAnalyzer(config)

        # Create large mock dataset
        n_points = 10000
        mock_system = MagicMock()

        mock_system.get_state.return_value = {
            "ignition": {
                "ignition_times": [i * 100.0 for i in range(100)],  # 100 ignitions
                "ignition_durations": [50.0 + 10.0 * np.random.randn() for _ in range(100)],
            },
            "metabolism": {
                "energy_history": [0.1 + 0.05 * np.random.randn() for _ in range(n_points)],
                "reserve_history": [1.0 - i * 0.0001 for i in range(n_points)],
            },
            "time": n_points * 10.0,
        }

        mock_system.get_history.return_value = {
            "ignition_signals": [1.0 + 0.5 * np.random.randn() for _ in range(n_points)],
            "free_energy_values": [1.5 + 0.3 * np.random.randn() for _ in range(n_points)],
            "timestamps": [i * 10.0 for i in range(n_points)],
        }

        # Analysis should complete in reasonable time
        import time

        start_time = time.time()
        results = analyzer.analyze_system(mock_system)
        analysis_time = time.time() - start_time

        assert isinstance(results, AnalysisResults)
        assert analysis_time < 10.0  # Should complete within 10 seconds

        # Results should be valid
        assert len(results.ignition_statistics) > 0
        assert len(results.energy_budget_summary) > 0
