"""Data Export and Visualization System for APGI Framework.

This module provides multi-format data export capabilities and publication-quality
plotting and figure generation for falsification test results.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

import h5py
import numpy as np

from ..core.data_models import ExperimentalTrial, FalsificationResult
from ..exceptions import DataExportError


class DataExporter:
    """Handles multi-format data export (CSV, JSON, HDF5) for experimental results."""

    def __init__(self, output_dir: str | None = None, config: dict[str, Any] | None = None):
        """Initialize the data exporter.

        Args:
            output_dir: Directory to save exported data (default: current directory)
            config: Optional configuration dictionary
        """
        if output_dir is None:
            output_dir = "."
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.export_dir = self.output_dir  # Alias for compatibility
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def export_falsification_results(
        self,
        results: List[FalsificationResult],
        format: str = "csv",
        filename: Optional[str] = None,
    ) -> str:
        """Export falsification results to specified format.

        Args:
            results: List of falsification results
            format: Export format ("csv", "json", "hdf5")
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"falsification_results_{timestamp}"

        try:
            if format == "csv":
                return self._export_results_csv(results, filename)
            elif format == "json":
                return self._export_results_json(results, filename)
            elif format == "hdf5":
                return self._export_results_hdf5(results, filename)
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            raise DataExportError(f"Failed to export falsification results: {str(e)}")

    def export_experimental_trials(
        self,
        trials: List[ExperimentalTrial],
        format: str = "csv",
        filename: Optional[str] = None,
    ) -> str:
        """Export experimental trials to specified format.

        Args:
            trials: List of experimental trials
            format: Export format ("csv", "json", "hdf5")
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experimental_trials_{timestamp}"

        try:
            if format == "csv":
                return self._export_trials_csv(trials, filename)
            elif format == "json":
                return self._export_trials_json(trials, filename)
            elif format == "hdf5":
                return self._export_trials_hdf5(trials, filename)
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            raise DataExportError(f"Failed to export experimental trials: {str(e)}")

    def _export_results_csv(self, results: List[FalsificationResult], filename: str) -> str:
        """Export falsification results to CSV."""
        filepath = self.output_dir / f"{filename}.csv"

        # Convert results to flat dictionary format
        rows = []
        for result in results:
            row = {
                "test_type": result.test_type,
                "is_falsified": result.is_falsified,
                "confidence_level": result.confidence_level,
                "effect_size": result.effect_size,
                "p_value": result.p_value,
                "statistical_power": result.statistical_power,
                "replication_count": result.replication_count,
            }

            # Add detailed results as separate columns
            if result.detailed_results:
                for key, value in result.detailed_results.items():
                    if isinstance(value, (int, float, str, bool)):
                        row[f"detail_{key}"] = value

            rows.append(row)

        # Write to CSV
        if rows:
            with open(filepath, "w", newline="") as csvfile:
                fieldnames = rows[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        self.logger.info(f"Exported {len(results)} falsification results to {filepath}")
        return str(filepath)

    def _export_results_json(self, results: List[FalsificationResult], filename: str) -> str:
        """Export falsification results to JSON."""
        filepath = self.output_dir / f"{filename}.json"

        # Convert to serializable format
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_results": len(results),
            "results": [asdict(result) for result in results],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info(f"Exported {len(results)} falsification results to {filepath}")
        return str(filepath)

    def _export_results_hdf5(self, results: List[FalsificationResult], filename: str) -> str:
        """Export falsification results to HDF5."""
        filepath = self.output_dir / f"{filename}.h5"

        with h5py.File(filepath, "w") as f:
            # Create groups for different data types
            results_group = f.create_group("falsification_results")

            # Store metadata
            results_group.attrs["export_timestamp"] = datetime.now().isoformat()
            results_group.attrs["total_results"] = len(results)

            if results:
                # Convert to arrays for efficient storage
                test_types = [r.test_type.encode("utf-8") for r in results]
                is_falsified = [r.is_falsified for r in results]
                confidence_levels = [r.confidence_level for r in results]
                effect_sizes = [r.effect_size for r in results]
                p_values = [r.p_value for r in results]
                statistical_powers = [r.statistical_power for r in results]
                replication_counts = [r.replication_count for r in results]

                # Store arrays
                results_group.create_dataset("test_types", data=test_types)
                results_group.create_dataset("is_falsified", data=is_falsified)
                results_group.create_dataset("confidence_levels", data=confidence_levels)
                results_group.create_dataset("effect_sizes", data=effect_sizes)
                results_group.create_dataset("p_values", data=p_values)
                results_group.create_dataset("statistical_powers", data=statistical_powers)
                results_group.create_dataset("replication_counts", data=replication_counts)

        self.logger.info(f"Exported {len(results)} falsification results to {filepath}")
        return str(filepath)

    def _export_trials_csv(self, trials: List[ExperimentalTrial], filename: str) -> str:
        """Export experimental trials to CSV."""
        filepath = self.output_dir / f"{filename}.csv"

        rows = []
        for trial in trials:
            row = {
                "trial_id": trial.trial_id,
                "participant_id": trial.participant_id,
                "condition": trial.condition,
                "timestamp": trial.timestamp.isoformat() if trial.timestamp else "",
                # APGI Parameters
                "extero_precision": trial.apgi_parameters.extero_precision,
                "intero_precision": trial.apgi_parameters.intero_precision,
                "extero_error": trial.apgi_parameters.extero_error,
                "intero_error": trial.apgi_parameters.intero_error,
                "somatic_gain": trial.apgi_parameters.somatic_gain,
                "threshold": trial.apgi_parameters.threshold,
                "steepness": trial.apgi_parameters.steepness,
                # Neural Signatures
                "p3b_amplitude": trial.neural_signatures.p3b_amplitude,
                "p3b_latency": trial.neural_signatures.p3b_latency,
                "gamma_plv": trial.neural_signatures.gamma_plv,
                "gamma_duration": trial.neural_signatures.gamma_duration,
                "pci_value": trial.neural_signatures.pci_value,
                # Consciousness Assessment
                "subjective_report": trial.consciousness_assessment.subjective_report,
                "forced_choice_accuracy": trial.consciousness_assessment.forced_choice_accuracy,
                "confidence_rating": trial.consciousness_assessment.confidence_rating,
                "wagering_behavior": trial.consciousness_assessment.wagering_behavior,
                "metacognitive_sensitivity": trial.consciousness_assessment.metacognitive_sensitivity,
            }

            # Add BOLD activations as separate columns
            for region, activation in trial.neural_signatures.bold_activations.items():
                row[f"bold_{region}"] = activation

            rows.append(row)

        if rows:
            with open(filepath, "w", newline="") as csvfile:
                fieldnames = rows[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        self.logger.info(f"Exported {len(trials)} experimental trials to {filepath}")
        return str(filepath)

    def _export_trials_json(self, trials: List[ExperimentalTrial], filename: str) -> str:
        """Export experimental trials to JSON."""
        filepath = self.output_dir / f"{filename}.json"

        data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_trials": len(trials),
            "trials": [asdict(trial) for trial in trials],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info(f"Exported {len(trials)} experimental trials to {filepath}")
        return str(filepath)

    def _export_trials_hdf5(self, trials: List[ExperimentalTrial], filename: str) -> str:
        """Export experimental trials to HDF5."""
        filepath = self.output_dir / f"{filename}.h5"

        with h5py.File(filepath, "w") as f:
            trials_group = f.create_group("experimental_trials")
            trials_group.attrs["export_timestamp"] = datetime.now().isoformat()
            trials_group.attrs["total_trials"] = len(trials)

            if trials:
                # Create subgroups for different data types
                params_group = trials_group.create_group("apgi_parameters")
                neural_group = trials_group.create_group("neural_signatures")
                trials_group.create_group("consciousness_assessment")

                # Extract and store parameter arrays
                extero_precisions = [t.apgi_parameters.extero_precision for t in trials]
                intero_precisions = [t.apgi_parameters.intero_precision for t in trials]
                extero_errors = [t.apgi_parameters.extero_error for t in trials]
                intero_errors = [t.apgi_parameters.intero_error for t in trials]
                somatic_gains = [t.apgi_parameters.somatic_gain for t in trials]
                thresholds = [t.apgi_parameters.threshold for t in trials]
                steepnesses = [t.apgi_parameters.steepness for t in trials]

                params_group.create_dataset("extero_precisions", data=extero_precisions)
                params_group.create_dataset("intero_precisions", data=intero_precisions)
                params_group.create_dataset("extero_errors", data=extero_errors)
                params_group.create_dataset("intero_errors", data=intero_errors)
                params_group.create_dataset("somatic_gains", data=somatic_gains)
                params_group.create_dataset("thresholds", data=thresholds)
                params_group.create_dataset("steepnesses", data=steepnesses)

                # Extract and store neural signature arrays
                p3b_amplitudes = [t.neural_signatures.p3b_amplitude for t in trials]
                p3b_latencies = [t.neural_signatures.p3b_latency for t in trials]
                gamma_plvs = [t.neural_signatures.gamma_plv for t in trials]
                gamma_durations = [t.neural_signatures.gamma_duration for t in trials]
                pci_values = [t.neural_signatures.pci_value for t in trials]

                neural_group.create_dataset("p3b_amplitudes", data=p3b_amplitudes)
                neural_group.create_dataset("p3b_latencies", data=p3b_latencies)
                neural_group.create_dataset("gamma_plvs", data=gamma_plvs)
                neural_group.create_dataset("gamma_durations", data=gamma_durations)
                neural_group.create_dataset("pci_values", data=pci_values)

        self.logger.info(f"Exported {len(trials)} experimental trials to {filepath}")
        return str(filepath)

    # Convenience methods for simpler API
    def export_to_json(self, data: dict, filepath: Union[str, Path]) -> str:
        """Export dictionary data to JSON file.

        Args:
            data: Dictionary to export
            filepath: Path for output file

        Returns:
            Path to exported file
        """
        filepath = Path(filepath)
        # Prepend output_dir if filepath is relative
        if not filepath.is_absolute():
            filepath = self.output_dir / filepath
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info(f"Exported data to {filepath}")
        return str(filepath)

    def export_to_csv(
        self,
        data: Any,
        filepath: Union[str, Path],
        delimiter: str = ",",
        compression: Optional[Literal["infer", "gzip", "bz2", "zip", "xz", "zstd", "tar"]] = None,
    ) -> str:
        """Export data to CSV file.

        Args:
            data: DataFrame or dict/list to export
            filepath: Path for output file
            delimiter: CSV delimiter
            compression: Compression format (None, 'gzip', 'zip')

        Returns:
            Path to exported file
        """
        import pandas as pd

        filepath = Path(filepath)
        # Prepend output_dir if filepath is relative
        if not filepath.is_absolute():
            filepath = self.output_dir / filepath
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame if needed
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Cannot export type {type(data)} to CSV")

        df.to_csv(filepath, index=False, sep=delimiter, compression=compression)
        self.logger.info(f"Exported data to {filepath}")
        return str(filepath)

    def export_to_excel(self, data: Any, filepath: Union[str, Path]) -> str:
        """Export data to Excel file."""
        import pandas as pd

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, dict) and all(
            isinstance(v, (pd.DataFrame, list)) for v in data.values()
        ):
            # Multiple sheets
            with pd.ExcelWriter(filepath) as writer:
                for sheet_name, sheet_data in data.items():
                    df = (
                        pd.DataFrame(sheet_data)
                        if not isinstance(sheet_data, pd.DataFrame)
                        else sheet_data
                    )
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            df.to_excel(filepath, index=False)

        return str(filepath)

    def export_to_parquet(self, data: Any, filepath: Union[str, Path]) -> str:
        """Export data to Parquet file."""
        import pandas as pd

        filepath = Path(filepath)
        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        df.to_parquet(filepath, index=False)
        return str(filepath)

    def export_to_hdf5(self, data: Any, filepath: Union[str, Path]) -> str:
        """Export data to HDF5 file using h5py."""
        import pandas as pd

        filepath = Path(filepath)
        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data

        with h5py.File(filepath, "w") as f:
            group = f.create_group("data")
            for column in df.columns:
                values = df[column].values
                if values.dtype == object:
                    values = values.astype(str).astype(h5py.special_dtype(vlen=str))
                group.create_dataset(column, data=values)

        self.logger.info(f"Exported data to {filepath} using h5py")
        return str(filepath)

    def export_to_pickle(self, data: Any, filepath: Union[str, Path]) -> str:
        """Export data to Pickle file."""
        import pickle

        filepath = Path(filepath)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        return str(filepath)

    def export_to_numpy(self, data: Any, filepath: Union[str, Path]) -> str:
        """Export data to Numpy file."""
        filepath = Path(filepath)
        np.save(filepath, data)
        return str(filepath)

    def batch_export(self, datasets: list[tuple[str, Any]], format: str = "csv") -> dict[str, str]:
        """Export multiple datasets at once."""
        results = {}
        for filename, data in datasets:
            results[filename] = getattr(self, f"export_to_{format}")(data, filename)
        return results

    def export_experiment_bundle(
        self, experiment_data: dict[str, Any], filepath: Union[str, Path]
    ) -> str:
        """Export a complete experiment bundle as a zip file."""
        import zipfile

        filepath = Path(filepath)
        with zipfile.ZipFile(filepath, "w") as zipf:
            # For simplicity, just export everything inside as JSON
            for key, data in experiment_data.items():
                zipf.writestr(f"{key}.json", json.dumps(data, default=str))

        return str(filepath)

    def export_with_metadata(
        self, data: Any, filepath: Union[str, Path], metadata: dict[str, Any]
    ) -> str:
        """Export data along with metadata."""
        # For simplicity, combine them into one JSON or export separately
        filepath = Path(filepath)
        if filepath.suffix == ".json":
            combined = {"data": data, "metadata": metadata}
            return self.export_to_json(combined, filepath)
        else:
            # Export data normally and metadata as separate JSON
            self.export_to_csv(data, filepath)
            meta_path = filepath.with_suffix(".meta.json")
            self.export_to_json(metadata, meta_path)
            return str(filepath)

    def generate_export_metadata(self, **kwargs: Any) -> dict[str, Any]:
        """Generate metadata for an export."""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "timestamp": datetime.now().isoformat(),
            "generator": "APGI DataExporter",
        }
        metadata.update(kwargs)
        return metadata

    def validate_export_path(self, filepath: Union[str, Path]) -> bool:
        """Validate if the export path is valid."""
        path = Path(filepath)
        return path.parent.exists()

    def validate_export_format(self, format: str) -> bool:
        """Validate if the export format is supported."""
        return format.lower() in self.get_supported_formats()

    def export_numpy(self, data: np.ndarray, filepath: Union[str, Path]) -> str:
        """Export numpy array to file.

        Args:
            data: Numpy array to export
            filepath: Path for output file

        Returns:
            Path to exported file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        np.save(filepath, data)
        self.logger.info(f"Exported array to {filepath}")
        return str(filepath)

    def get_supported_formats(self) -> list:
        """Get list of supported export formats.

        Returns:
            List of format names
        """
        return ["csv", "json", "hdf5", "npy"]
