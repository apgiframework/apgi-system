"""Advanced Data Export Module.

Provides comprehensive data export capabilities for APGI system results,
including scientific formats, statistical analysis, and visualization exports.
"""

import importlib.util
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from numpy.typing import NDArray

# Import from analysis.py file directly to avoid conflict with analysis/ package
spec = importlib.util.spec_from_file_location(
    "analysis_module", Path(__file__).with_name("analysis.py")
)
if spec is None or spec.loader is None:
    raise ImportError("Failed to load analysis module")
analysis_module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(analysis_module)

AnalysisResults = analysis_module.AnalysisResults  # type: ignore[misc]


class DataExporter:
    """Advanced data exporter for APGI system results.

    Supports multiple export formats including:
    - CSV/TSV for spreadsheet analysis
    - JSON for web applications
    - HDF5 for large datasets
    - MATLAB .mat files for MATLAB analysis
    - NetCDF for scientific computing
    - Parquet for big data analytics
    - Statistical summaries and reports
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data exporter with configuration."""
        self.config = config or {}
        self.default_precision = self.config.get("precision", 6)
        self.include_metadata = self.config.get("include_metadata", True)
        self.compress_output = self.config.get("compress_output", True)

    def export_csv(
        self,
        data: Any,  # AnalysisResults or Dict[str, Any]
        filepath: Union[str, Path],
        include_temporal: bool = True,
        delimiter: str = ",",
    ) -> bool:
        """Export data to CSV format.

        Args:
            data: Analysis results or data dictionary
            filepath: Output file path
            include_temporal: Whether to include time series data
            delimiter: CSV delimiter (default: comma)

        Returns:
            True if export successful
        """
        try:
            filepath = Path(filepath)

            if isinstance(data, AnalysisResults):
                # Convert AnalysisResults to exportable format
                export_data = self._prepare_tabular_data(data, include_temporal)
            else:
                export_data = data

            # Create DataFrame for easier CSV export
            if "temporal_dynamics" in export_data and include_temporal:
                return self._export_temporal_csv(export_data, filepath)
            else:
                return self._export_summary_csv(export_data, filepath)

        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False

    def _export_temporal_csv(self, export_data: Dict[str, Any], filepath: Path) -> bool:
        """Export temporal dynamics data to CSV."""
        temporal_data = export_data["temporal_dynamics"]
        df = pd.DataFrame(temporal_data)

        # Add metadata as header comments
        if self.include_metadata:
            with open(filepath, "w", newline="") as f:
                # Write metadata as comments
                f.write("# APGI System Data Export\n")
                f.write(f"# Export Date: {datetime.now().isoformat()}\n")
                f.write("# Data Type: Temporal Dynamics\n")

                # Write summary statistics as comments
                for category, stats in export_data.items():
                    if category != "temporal_dynamics" and isinstance(stats, dict):
                        f.write(f"# {category.title()}:\n")
                        for key, value in stats.items():
                            f.write(f"#   {key}: {value}\n")
                f.write("#\n")

                # Write CSV data
                df.to_csv(f, index=False, float_format=f"%.{self.default_precision}f")
        else:
            df.to_csv(filepath, index=False, float_format=f"%.{self.default_precision}f")
        return True

    def _export_summary_csv(self, export_data: Dict[str, Any], filepath: Path) -> bool:
        """Export summary statistics to CSV."""
        rows = []
        for category, stats in export_data.items():
            if isinstance(stats, dict):
                for key, value in stats.items():
                    rows.append({"category": category, "metric": key, "value": value})

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        return True

    def export_json(
        self,
        data: Any,  # AnalysisResults or Dict[str, Any]
        filepath: Union[str, Path],
        pretty_print: bool = True,
    ) -> bool:
        """Export data to JSON format.

        Args:
            data: Analysis results or data dictionary
            filepath: Output file path
            pretty_print: Whether to format JSON with indentation

        Returns:
            True if export successful
        """
        try:
            filepath = Path(filepath)

            if isinstance(data, AnalysisResults):
                export_data = asdict(data)
            else:
                export_data = data

            # Add metadata
            if self.include_metadata:
                export_data["_metadata"] = {
                    "export_date": datetime.now().isoformat(),
                    "exporter_version": "1.0.0",
                    "data_type": "APGI_Analysis_Results",
                }

            # Convert numpy arrays to lists for JSON serialization
            export_data = self._convert_numpy_to_json(export_data)

            with open(filepath, "w") as f:
                if pretty_print:
                    json.dump(export_data, f, indent=2, default=str)
                else:
                    json.dump(export_data, f, default=str)

            return True

        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False

    def export_hdf5(
        self,
        data: Any,  # AnalysisResults or Dict[str, Any]
        filepath: Union[str, Path],
        compression: str = "gzip",
    ) -> bool:
        """Export data to HDF5 format for large datasets.

        Args:
            data: Analysis results or data dictionary
            filepath: Output file path
            compression: Compression algorithm ('gzip', 'lzf', 'szip')

        Returns:
            True if export successful
        """
        try:
            filepath = Path(filepath)

            if isinstance(data, AnalysisResults):
                export_data = asdict(data)
            else:
                export_data = data

            with h5py.File(filepath, "w") as f:
                # Add metadata
                if self.include_metadata:
                    f.attrs["export_date"] = datetime.now().isoformat()
                    f.attrs["data_type"] = "APGI_Analysis_Results"

                # Recursively write data groups
                self._write_hdf5_group(f, export_data, compression)

            return True

        except Exception as e:
            print(f"Error exporting to HDF5: {e}")
            return False

    def export_matlab(
        self, data: Any, filepath: Union[str, Path]  # AnalysisResults or Dict[str, Any]
    ) -> bool:
        """Export data to MATLAB .mat format.

        Args:
            data: Analysis results or data dictionary
            filepath: Output file path

        Returns:
            True if export successful
        """
        try:
            filepath = Path(filepath)

            if isinstance(data, AnalysisResults):
                export_data = asdict(data)
            else:
                export_data = data

            # Convert data for MATLAB compatibility
            matlab_data = self._convert_for_matlab(export_data)

            # Add metadata
            if self.include_metadata:
                matlab_data["metadata"] = {
                    "export_date": datetime.now().isoformat(),
                    "data_type": "APGI_Analysis_Results",
                }

            scipy.io.savemat(filepath, matlab_data)

            return True

        except Exception as e:
            print(f"Error exporting to MATLAB: {e}")
            return False

    def export_parquet(
        self, data: Any, filepath: Union[str, Path]  # AnalysisResults or Dict[str, Any]
    ) -> bool:
        """Export temporal data to Parquet format for big data analytics.

        Args:
            data: Analysis results or data dictionary
            filepath: Output file path

        Returns:
            True if export successful
        """
        try:
            filepath = Path(filepath)

            if isinstance(data, AnalysisResults):
                export_data = asdict(data)
            else:
                export_data = data

            # Extract temporal data for Parquet export
            if "temporal_dynamics" in export_data:
                df = pd.DataFrame(export_data["temporal_dynamics"])

                # Add metadata columns
                if self.include_metadata:
                    df["export_date"] = datetime.now().isoformat()
                    df["data_type"] = "APGI_Temporal_Data"

                df.to_parquet(filepath, compression="snappy" if self.compress_output else None)
                return True
            else:
                print("No temporal data available for Parquet export")
                return False

        except Exception as e:
            print(f"Error exporting to Parquet: {e}")
            return False

    def export_statistical_report(
        self,
        data: Any,  # AnalysisResults or Dict[str, Any]
        filepath: Union[str, Path],
        include_plots: bool = True,
    ) -> bool:
        """Export comprehensive statistical analysis report.

        Args:
            data: Analysis results or data dictionary
            filepath: Output file path (HTML format)
            include_plots: Whether to include visualization plots

        Returns:
            True if export successful
        """
        try:
            filepath = Path(filepath)

            if isinstance(data, AnalysisResults):
                export_data = asdict(data)
            else:
                export_data = data

            # Generate HTML report
            html_content = self._generate_statistical_html(export_data, include_plots)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)

            return True

        except Exception as e:
            print(f"Error exporting statistical report: {e}")
            return False

    def export_visualization_plots(
        self,
        data: Any,  # AnalysisResults or Dict[str, Any]
        output_dir: Union[str, Path],
        formats: List[str] = ["png", "pdf"],
    ) -> bool:
        """Export visualization plots of the analysis results.

        Args:
            data: Analysis results or data dictionary
            output_dir: Output directory for plots
            formats: List of image formats to export

        Returns:
            True if export successful
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if isinstance(data, AnalysisResults):
                export_data = asdict(data)
            else:
                export_data = data

            # Generate various plots
            plots_created = []

            # 1. Temporal dynamics plot
            if "temporal_dynamics" in export_data:
                fig = self._create_temporal_plot(export_data["temporal_dynamics"])
                for fmt in formats:
                    plot_path = output_dir / f"temporal_dynamics.{fmt}"
                    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plots_created.append(plot_path)
                plt.close(fig)

            # 2. Ignition statistics plot
            if "ignition_statistics" in export_data:
                fig = self._create_ignition_plot(export_data["ignition_statistics"])
                for fmt in formats:
                    plot_path = output_dir / f"ignition_statistics.{fmt}"
                    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plots_created.append(plot_path)
                plt.close(fig)

            # 3. Energy budget plot
            if "energy_budget_summary" in export_data:
                fig = self._create_energy_plot(export_data["energy_budget_summary"])
                for fmt in formats:
                    plot_path = output_dir / f"energy_budget.{fmt}"
                    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plots_created.append(plot_path)
                plt.close(fig)

            print(f"Created {len(plots_created)} visualization plots")
            return True

        except Exception as e:
            print(f"Error exporting visualization plots: {e}")
            return False

    def _prepare_tabular_data(
        self, results: Any, include_temporal: bool  # AnalysisResults
    ) -> Dict[str, Any]:
        """Prepare AnalysisResults for tabular export."""
        data = asdict(results)

        if not include_temporal and "temporal_dynamics" in data:
            del data["temporal_dynamics"]

        return data

    def _convert_numpy_to_json(self, obj: Any) -> Any:
        """Recursively convert numpy arrays to JSON-serializable lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    def _write_hdf5_group(self, group: h5py.Group, data: Dict[str, Any], compression: str) -> None:
        """Recursively write data to HDF5 group."""
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._write_hdf5_group(subgroup, value, compression)
            elif isinstance(value, (list, np.ndarray)):
                # Convert to numpy array for HDF5 storage
                array_data = np.array(value)
                if array_data.size > 0:
                    group.create_dataset(key, data=array_data, compression=compression)
            else:
                # Store scalar values as attributes
                group.attrs[key] = value

    def _convert_for_matlab(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert data structure for MATLAB compatibility."""
        matlab_data = {}

        for key, value in data.items():
            # Replace invalid MATLAB variable name characters
            matlab_key = key.replace("-", "_").replace(" ", "_")

            if isinstance(value, dict):
                matlab_data[matlab_key] = self._convert_for_matlab(value)
            elif isinstance(value, list):
                converted_array = np.array(value, dtype=np.float64)
                matlab_data[matlab_key] = converted_array.tolist()
            elif isinstance(value, np.ndarray):
                matlab_data[matlab_key] = value.tolist()
            else:
                matlab_data[matlab_key] = value

        return matlab_data

    def _generate_statistical_html(self, data: Dict[str, Any], include_plots: bool) -> str:
        """Generate HTML statistical report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>APGI System Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ margin: 10px 0; }}
                .section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>APGI System Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """

        # Add sections for each data category
        for category, stats in data.items():
            if isinstance(stats, dict) and stats:
                html += f"""
                <div class="section">
                    <h2>{category.replace('_', ' ').title()}</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                """

                for key, value in stats.items():
                    if isinstance(value, float):
                        formatted_value = f"{value:.{self.default_precision}f}"
                    else:
                        formatted_value = str(value)

                    html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{formatted_value}</td></tr>"

                html += "</table></div>"

        html += """
        </body>
        </html>
        """

        return html

    def _create_temporal_plot(self, temporal_data: Dict[str, List[float]]) -> plt.Figure:
        """Create temporal dynamics visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("APGI System Temporal Dynamics", fontsize=16)

        if "time" in temporal_data:
            time = temporal_data["time"]

            # Plot ignition signal
            if "ignition_signal" in temporal_data:
                axes[0, 0].plot(time, temporal_data["ignition_signal"])
                axes[0, 0].set_title("Ignition Signal")
                axes[0, 0].set_xlabel("Time (ms)")
                axes[0, 0].set_ylabel("Signal Strength")

            # Plot free energy
            if "free_energy" in temporal_data:
                axes[0, 1].plot(time, temporal_data["free_energy"])
                axes[0, 1].set_title("Free Energy")
                axes[0, 1].set_xlabel("Time (ms)")
                axes[0, 1].set_ylabel("Free Energy")

            # Plot precision
            if "precision" in temporal_data:
                axes[1, 0].plot(time, temporal_data["precision"])
                axes[1, 0].set_title("Precision")
                axes[1, 0].set_xlabel("Time (ms)")
                axes[1, 0].set_ylabel("Precision")

            # Plot metabolic reserves
            if "metabolic_reserves" in temporal_data:
                axes[1, 1].plot(time, temporal_data["metabolic_reserves"])
                axes[1, 1].set_title("Metabolic Reserves")
                axes[1, 1].set_xlabel("Time (ms)")
                axes[1, 1].set_ylabel("Reserve Level")

        plt.tight_layout()
        return fig

    def _create_ignition_plot(self, ignition_stats: Dict[str, float]) -> plt.Figure:
        """Create ignition statistics visualization."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Create bar plot of key ignition metrics
        metrics = []
        values = []

        key_metrics = ["total_ignitions", "ignition_rate_hz", "mean_ignition_duration_ms"]
        for metric in key_metrics:
            if metric in ignition_stats:
                metrics.append(metric.replace("_", " ").title())
                values.append(ignition_stats[metric])

        if metrics:
            ax.bar(metrics, values)
            ax.set_title("Ignition Statistics")
            ax.set_ylabel("Value")
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        return fig

    def _create_energy_plot(self, energy_stats: Dict[str, float]) -> plt.Figure:
        """Create energy budget visualization."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Create pie chart of energy distribution
        if "total_energy_consumed" in energy_stats and "final_reserves" in energy_stats:
            consumed = energy_stats["total_energy_consumed"]
            remaining = energy_stats.get("final_reserves", 0)

            labels = ["Energy Consumed", "Remaining Reserves"]
            sizes = [consumed, remaining]
            colors = ["#ff9999", "#66b3ff"]

            ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            ax.set_title("Energy Budget Distribution")

        return fig


class AdvancedAnalytics:
    """Advanced statistical analysis and machine learning tools for APGI data."""

    def __init__(self) -> None:
        """Initialize advanced analytics."""
        pass

    def compute_correlation_matrix(
        self, temporal_data: Dict[str, List[float]]
    ) -> NDArray[np.floating]:
        """Compute correlation matrix between temporal variables."""
        df = pd.DataFrame(temporal_data)
        return df.corr().values

    def detect_anomalies(
        self, data: List[float], method: str = "zscore", threshold: float = 3.0
    ) -> List[int]:
        """Detect anomalies in time series data."""
        data_array = np.array(data)

        if method == "zscore":
            z_scores = np.abs((data_array - np.mean(data_array)) / np.std(data_array))
            return np.where(z_scores > threshold)[0].tolist()  # type: ignore[no-any-return]
        elif method == "iqr":
            q1, q3 = np.percentile(data_array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return np.where((data_array < lower_bound) | (data_array > upper_bound))[0].tolist()  # type: ignore[no-any-return]
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")

    def compute_spectral_analysis(
        self, signal: List[float], sampling_rate: float = 1.0
    ) -> Dict[str, Any]:
        """Perform spectral analysis on signal data."""
        from scipy import signal as scipy_signal

        signal_array = np.array(signal)

        # Compute power spectral density
        frequencies, psd = scipy_signal.welch(signal_array, fs=sampling_rate)

        # Find dominant frequency
        dominant_freq_idx = np.argmax(psd)
        dominant_frequency = frequencies[dominant_freq_idx]

        # Calculate total power
        total_power = np.sum(psd)

        return {
            "dominant_frequency": float(dominant_frequency),
            "frequencies": frequencies.tolist(),
            "power_spectral_density": psd.tolist(),
            "total_power": float(total_power),
        }

    @staticmethod
    def _sample_entropy(data: NDArray[np.floating], m: int = 2, r: float = 0.2) -> float:
        """Compute sample entropy for a signal."""
        N = len(data)

        def _maxdist(xi: NDArray[np.floating], xj: NDArray[np.floating]) -> float:
            return float(max(abs(ua - va) for ua, va in zip(xi, xj)))

        phi = np.zeros(2)
        for m_i in [m, m + 1]:
            patterns_m = np.array([data[i : i + m_i] for i in range(N - m_i + 1)])
            C = np.zeros(N - m_i + 1)

            for i in range(N - m_i + 1):
                template = patterns_m[i]
                matches = sum(
                    [1 for j in range(N - m_i + 1) if _maxdist(template, patterns_m[j]) <= r]
                )
                C[i] = matches / (N - m_i + 1.0)

            phi[m_i - m] = np.mean(np.log(C))

        return phi[0] - phi[1]  # type: ignore[no-any-return]

    @staticmethod
    def _approximate_entropy(data: NDArray[np.floating], m: int = 2, r: float = 0.2) -> float:
        """Compute approximate entropy for a signal."""
        N = len(data)

        def _maxdist(xi: NDArray[np.floating], xj: NDArray[np.floating]) -> float:
            return float(max(abs(ua - va) for ua, va in zip(xi, xj)))

        def _phi(m_val: int) -> float:
            patterns = np.array([data[i : i + m_val] for i in range(N - m_val + 1)])
            C = np.zeros(N - m_val + 1)

            for i in range(N - m_val + 1):
                template = patterns[i]
                matches = sum(
                    [1 for j in range(N - m_val + 1) if _maxdist(template, patterns[j]) <= r]
                )
                C[i] = matches / (N - m_val + 1.0)

            return float(np.mean(np.log(C)))  # type: ignore[no-any-return]

        return _phi(m) - _phi(m + 1)

    def compute_complexity_measures(self, signal: List[float]) -> Dict[str, float]:
        """Compute complexity measures for signal analysis."""
        signal_array = np.array(signal)

        try:
            sample_ent = self._sample_entropy(signal_array)
            approx_ent = self._approximate_entropy(signal_array)
        except Exception:
            sample_ent = np.nan
            approx_ent = np.nan

        return {
            "sample_entropy": sample_ent,
            "approximate_entropy": approx_ent,
            "variance": np.var(signal_array),
            "coefficient_of_variation": (
                np.std(signal_array) / np.mean(signal_array)
                if np.mean(signal_array) != 0
                else np.nan
            ),
        }
