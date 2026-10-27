"""BIDS (Brain Imaging Data Structure) Export Module for APGI Framework
Provides standardized export functionality for neuroimaging data.
"""

import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from apgi_framework.logging.standardized_logging import get_logger

logger = get_logger(__name__)


class BIDSExporter:
    """Export APGI data to BIDS-compliant format."""

    def __init__(self, bids_root: Union[str, Path] = "data/bids_dataset"):
        """Initialize BIDS exporter.

        Args:
            bids_root: Root directory for BIDS dataset (default: "data/bids_dataset")
        """
        self.bids_root = Path(bids_root)
        self.bids_root.mkdir(parents=True, exist_ok=True)

        # Create required BIDS directories
        self._create_bids_structure()

        # BIDS entity order (as per specification)
        self.entity_order = [
            "sub",
            "ses",
            "task",
            "acq",
            "ce",
            "rec",
            "dir",
            "run",
            "echo",
            "flip",
        ]

    def _create_bids_structure(self) -> None:
        """Create BIDS directory structure."""
        directories = [
            "sub-01",
            "sub-01/ses-01",
            "sub-01/ses-01/eeg",
            "sub-01/ses-01/beh",
            "sub-01/ses-01/physio",
            "derivatives",
            "derivatives/analysis",
            "code",
            "stimuli",
            "sourcedata",
        ]

        for directory in directories:
            (self.bids_root / directory).mkdir(parents=True, exist_ok=True)

    def _sanitize_string(self, string: str) -> str:
        """Sanitize string for BIDS compliance and path traversal protection."""
        # Validate input to prevent path traversal
        if not string or not isinstance(string, str):
            raise ValueError(f"Invalid string input: {string}")

        # Check for path traversal attempts
        if ".." in string or "/" in string or "\\" in string:
            raise ValueError(f"Invalid characters in string (path traversal attempt): {string}")

        # Check for spaces (not allowed)
        if " " in string:
            raise ValueError(f"String contains spaces: {string}")

        # Remove invalid characters (only allow alphanumeric, underscore, hyphen)
        sanitized = re.sub(r"[^\w\-]", "_", string)

        # Replace multiple hyphens with single
        sanitized = re.sub(r"-+", "-", sanitized)

        # Replace multiple underscores with single
        sanitized = re.sub(r"_+", "_", sanitized)

        # Remove leading/trailing hyphens and underscores
        sanitized = sanitized.strip("_-")

        # Ensure result is not empty
        if not sanitized:
            raise ValueError(f"String became empty after sanitization: {string}")

        return sanitized

    def _create_bids_filename(self, entities: Dict[str, str], suffix: str, extension: str) -> str:
        """Create BIDS-compliant filename.

        Args:
            entities: Dictionary of BIDS entities (sub, ses, task, etc.)
            suffix: File suffix (eeg, beh, etc.)
            extension: File extension

        Returns:
            BIDS-compliant filename
        """
        # Build filename in correct order
        filename_parts = []

        for entity in self.entity_order:
            if entity in entities:
                value = self._sanitize_string(str(entities[entity]))
                filename_parts.append(f"{entity}-{value}")

        # Add suffix and extension
        filename_parts.append(suffix)
        filename = "_".join(filename_parts) + f".{extension}"

        return filename

    def export_eeg_data(
        self,
        data: np.ndarray,
        subject_id: str,
        session_id: Optional[str] = None,
        task_name: str = "rest",
        channels: Optional[List[str]] = None,
        sampling_rate: float = 1000.0,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """Export EEG data to BIDS format.

        Args:
            data: EEG data array (channels x timepoints)
            subject_id: Subject identifier
            session_id: Optional session identifier
            task_name: Task name
            channels: List of channel names
            sampling_rate: Sampling rate in Hz
            metadata: Additional metadata

        Returns:
            Path to exported file
        """
        # Create entities
        entities = {"sub": subject_id, "task": task_name}
        if session_id:
            entities["ses"] = session_id

        # Create filename
        filename = self._create_bids_filename(entities, "eeg", "set")

        # Determine output directory
        sanitized_subject_id = self._sanitize_string(subject_id)
        output_dir = self.bids_root / f"sub-{sanitized_subject_id}"
        if session_id:
            output_dir = output_dir / f"ses-{session_id}"
        output_dir = output_dir / "eeg"

        # Verify containment
        if not output_dir.resolve().is_relative_to(self.bids_root.resolve()):
            raise ValueError(f"Invalid subject_id: {subject_id}")

        output_path = output_dir / filename

        # Export EEG data
        output_path = self._export_eeg_set(data, channels, sampling_rate, output_path)

        # Create channels.tsv
        self._create_channels_tsv(channels, output_dir, filename.replace(".set", "_channels.tsv"))

        # Create sidecar JSON
        self._create_eeg_json(
            sampling_rate,
            channels,
            metadata,
            output_dir,
            filename.replace(".set", ".json"),
        )

        return output_path

    def _export_eeg_set(
        self,
        data: np.ndarray,
        channels: Optional[List[str]],
        sampling_rate: float,
        output_path: Path,
    ) -> Path:
        """Export EEG data in SET format (simplified)."""
        # For simplicity, we'll export as CSV with proper BIDS structure
        csv_path = output_path.with_suffix(".csv")

        # Ensure parent directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(data.T, columns=channels)
        df.to_csv(csv_path, index=False)

        # Also create a simple binary format
        np.save(output_path.with_suffix(".npy"), data)

        return csv_path

    def _create_channels_tsv(
        self, channels: Optional[List[str]], output_dir: Path, filename: str
    ) -> None:
        """Create channels.tsv file."""
        if not channels:
            return

        # Create channel information
        channel_data = []
        for i, channel in enumerate(channels):
            # Determine channel type based on name
            if (
                channel.startswith("F")
                or channel.startswith("C")
                or channel.startswith("P")
                or channel.startswith("O")
            ):
                ch_type = "EEG"
            elif channel.lower() in ["eog", "heog", "veog"]:
                ch_type = "EOG"
            elif channel.lower() in ["ecg", "ekg"]:
                ch_type = "ECG"
            elif channel.lower() in ["emg"]:
                ch_type = "EMG"
            else:
                ch_type = "MISC"

            channel_data.append(
                {
                    "name": channel,
                    "type": ch_type,
                    "units": "µV",
                    "sampling_frequency": 1000.0,
                    "description": f"{ch_type} channel {channel}",
                }
            )

        df = pd.DataFrame(channel_data)
        df.to_csv(output_dir / filename, sep="\t", index=False)

    def _create_eeg_json(
        self,
        sampling_rate: float,
        channels: Optional[List[str]],
        metadata: Optional[Dict],
        output_dir: Path,
        filename: str,
    ) -> None:
        """Create EEG sidecar JSON file."""
        json_data = {
            "SamplingFrequency": sampling_rate,
            "PowerLineFrequency": 50.0,  # Default, should be configurable
            "EEGReference": "common_average",
            "EEGGround": "FPz",
            "SoftwareFilters": {
                "HighPass": {"HalfAmplitudeCutOffHz": 0.1, "RollOff": "6dB/octave"},
                "LowPass": {"HalfAmplitudeCutOffHz": 100.0, "RollOff": "12dB/octave"},
            },
        }

        # Add channel information
        if channels:
            json_data["ChannelCount"] = len(channels)

        # Add custom metadata
        if metadata:
            json_data.update(metadata)

        with open(output_dir / filename, "w") as f:
            json.dump(json_data, f, indent=2)

    def export_behavioral_data(
        self,
        data: pd.DataFrame,
        subject_id: str,
        session_id: Optional[str] = None,
        task_name: str = "experiment",
        metadata: Optional[Dict] = None,
    ) -> Path:
        """Export behavioral data to BIDS format.

        Args:
            data: Behavioral data DataFrame
            subject_id: Subject identifier
            session_id: Optional session identifier
            task_name: Task name
            metadata: Additional metadata

        Returns:
            Path to exported file
        """
        # Create entities
        entities = {"sub": subject_id, "task": task_name}
        if session_id:
            entities["ses"] = session_id

        # Create filename
        filename = self._create_bids_filename(entities, "beh", "tsv")

        # Determine output directory
        sanitized_subject_id = self._sanitize_string(subject_id)
        output_dir = self.bids_root / f"sub-{sanitized_subject_id}"
        if session_id:
            output_dir = output_dir / f"ses-{session_id}"
        output_dir = output_dir / "beh"

        # Verify containment
        if not output_dir.resolve().is_relative_to(self.bids_root.resolve()):
            raise ValueError(f"Invalid subject_id: {subject_id}")

        output_path = output_dir / filename

        # Export data
        output_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, sep="\t", index=False)

        # Create sidecar JSON
        self._create_beh_json(
            data.columns.tolist(),
            metadata,
            output_dir,
            filename.replace(".tsv", ".json"),
        )

        return output_path

    def _create_beh_json(
        self,
        columns: List[str],
        metadata: Optional[Dict],
        output_dir: Path,
        filename: str,
    ) -> None:
        """Create behavioral sidecar JSON file."""
        json_data = {
            "LongName": "Behavioral data",
            "Description": "Behavioral measurements from experiment",
        }

        # Add column descriptions
        column_descriptions: Dict[str, str] = {}
        for col in columns:
            if "reaction_time" in col.lower() or "rt" in col.lower():
                column_descriptions[col] = "Reaction time in milliseconds"
            elif "accuracy" in col.lower() or "correct" in col.lower():
                column_descriptions[col] = "Response accuracy (1=correct, 0=incorrect)"
            elif "condition" in col.lower():
                column_descriptions[col] = "Experimental condition"
            elif "trial" in col.lower():
                column_descriptions[col] = "Trial number"
            else:
                column_descriptions[col] = f"Measurement: {col}"

        json_data["Columns"] = column_descriptions  # type: ignore

        # Add custom metadata
        if metadata:
            json_data.update(metadata)

        with open(output_dir / filename, "w") as f:
            json.dump(json_data, f, indent=2)

    def export_physiological_data(
        self,
        data: Dict[str, np.ndarray],
        subject_id: str,
        session_id: Optional[str] = None,
        task_name: str = "rest",
        sampling_rates: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Path]:
        """Export physiological data (pupillometry, cardiac, etc.) to BIDS format.

        Args:
            data: Dictionary of physiological data arrays
            subject_id: Subject identifier
            session_id: Optional session identifier
            task_name: Task name
            sampling_rates: Dictionary of sampling rates for each data type
            metadata: Additional metadata

        Returns:
            Dictionary of paths to exported files
        """
        exported_files = {}

        for data_type, array in data.items():
            # Create entities
            entities = {"sub": subject_id, "task": task_name}
            if session_id:
                entities["ses"] = session_id

            # Create filename
            filename = self._create_bids_filename(entities, data_type, "tsv.gz")

            # Determine output directory
            sanitized_subject_id = self._sanitize_string(subject_id)
            output_dir = self.bids_root / f"sub-{sanitized_subject_id}"
            if session_id:
                output_dir = output_dir / f"ses-{session_id}"
            output_dir = output_dir / "physio"

            # Verify containment
            if not output_dir.resolve().is_relative_to(self.bids_root.resolve()):
                raise ValueError(f"Invalid subject_id: {subject_id}")

            output_path = output_dir / filename

            # Export data
            output_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(array)
            df.to_csv(output_path, sep="\t", index=False, compression="gzip")

            # Create sidecar JSON
            sampling_rate = sampling_rates.get(data_type, 1000.0) if sampling_rates else 1000.0
            self._create_physio_json(
                data_type,
                sampling_rate,
                metadata,
                output_dir,
                filename.replace(".tsv.gz", ".json"),
            )

            exported_files[data_type] = output_path

        return exported_files

    def _create_physio_json(
        self,
        data_type: str,
        sampling_rate: float,
        metadata: Optional[Dict],
        output_dir: Path,
        filename: str,
    ) -> None:
        """Create physiological sidecar JSON file."""
        json_data = {
            "SamplingFrequency": sampling_rate,
            "StartTime": 0.0,
            "Columns": [],
        }

        # Add data type specific information with proper column descriptions
        if data_type.lower() == "pupil":
            json_data["Description"] = "Pupillometry data measuring pupil diameter over time"
            json_data["Units"] = "mm"
            json_data["Columns"] = [
                {
                    "name": "pupil_diameter_left",
                    "description": "Left eye pupil diameter in millimeters",
                    "units": "mm",
                },
                {
                    "name": "pupil_diameter_right",
                    "description": "Right eye pupil diameter in millimeters",
                    "units": "mm",
                },
            ]
        elif data_type.lower() == "cardiac":
            json_data["Description"] = (
                "Cardiac physiological data including heart rate and heart rate variability"
            )
            json_data["Units"] = ["bpm", "ms"]
            json_data["Columns"] = [
                {
                    "name": "heart_rate",
                    "description": "Heart rate in beats per minute",
                    "units": "bpm",
                },
                {
                    "name": "hrv_rmssd",
                    "description": "Heart rate variability - root mean square of successive differences",
                    "units": "ms",
                },
            ]
        else:
            json_data["Description"] = f"{data_type} physiological data"
            json_data["Units"] = "unknown"

        # Add custom metadata
        if metadata:
            json_data.update(metadata)

        with open(output_dir / filename, "w") as f:
            json.dump(json_data, f, indent=2)

    def create_dataset_description(
        self,
        name: str,
        authors: List[str],
        description: str,
        version: str = "1.0.0",
        doi: Optional[str] = None,
        license_type: str = "CC0",
    ) -> Path:
        """Create dataset_description.json file.

        Args:
            name: Dataset name
            authors: List of authors
            description: Dataset description
            version: Dataset version
            doi: Optional DOI
            license_type: License type

        Returns:
            Path to dataset_description.json
        """
        description_data = {
            "Name": name,
            "BIDSVersion": "1.7.0",
            "DatasetType": "raw",
            "Authors": authors,
            "Description": description,
            "Version": version,
            "License": license_type,
        }

        if doi:
            description_data["DOI"] = doi

        output_path = self.bids_root / "dataset_description.json"

        with open(output_path, "w") as f:
            json.dump(description_data, f, indent=2)

        return output_path

    def create_participants_tsv(self, participants: List[Dict[str, Any]]) -> Path:
        """Create participants.tsv file.

        Args:
            participants: List of participant dictionaries

        Returns:
            Path to participants.tsv
        """
        df = pd.DataFrame(participants)
        output_path = self.bids_root / "participants.tsv"
        df.to_csv(output_path, sep="\t", index=False)

        # Create participants.json
        json_data = {
            "participant_id": {
                "Description": "Unique participant identifier",
                "Levels": [f"sub-{i:03d}" for i in range(1, len(participants) + 1)],
            },
            "age": {"Description": "Age of participant", "Units": "years"},
            "sex": {"Description": "Sex of participant", "Levels": ["M", "F", "O"]},
            "handedness": {
                "Description": "Handedness of participant",
                "Levels": ["R", "L", "A"],
            },
        }

        json_path = self.bids_root / "participants.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        return output_path

    def create_readme(self, content: str) -> Path:
        """Create README file."""
        readme_path = self.bids_root / "README.md"
        with open(readme_path, "w") as f:
            f.write(content)
        return readme_path

    def validate_bids_structure(self) -> Dict[str, List[str]]:
        """Validate BIDS structure and return validation results.

        Returns:
            Dictionary with validation results
        """
        validation_results: Dict[str, List[str]] = {
            "errors": [],
            "warnings": [],
            "info": [],
        }

        # Check required files
        required_files = ["dataset_description.json", "README.md"]
        for file in required_files:
            if not (self.bids_root / file).exists():
                validation_results["errors"].append(f"Missing required file: {file}")

        # Check subject directories
        sub_dirs = [d for d in self.bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")]
        if not sub_dirs:
            validation_results["errors"].append("No subject directories found")

        # Check each subject
        for sub_dir in sub_dirs:
            # Check for modality directories
            modalities = ["eeg", "beh", "physio"]
            for modality in modalities:
                mod_dir = sub_dir / modality
                if mod_dir.exists():
                    files = list(mod_dir.glob("*"))
                    if not files:
                        validation_results["warnings"].append(f"No files in {mod_dir}")
                    else:
                        validation_results["info"].append(f"Found {len(files)} files in {mod_dir}")

        return validation_results


def export_apgi_to_bids(
    data: Dict[str, Any],
    bids_root: Union[str, Path],
    dataset_name: str = "APGI_Experiment",
    authors: Optional[List[str]] = None,
    description: str = "APGI Framework experiment data",
) -> Dict[str, Path]:
    """Convenience function to export complete APGI dataset to BIDS format.

    Args:
        data: Dictionary containing all APGI data
        bids_root: Root directory for BIDS dataset
        dataset_name: Name of the dataset
        authors: List of authors
        description: Dataset description

    Returns:
        Dictionary of exported file paths
    """
    if authors is None:
        authors = ["APGI Research Team"]

    exporter = BIDSExporter(bids_root)

    # Create dataset description
    dataset_desc = exporter.create_dataset_description(
        name=dataset_name, authors=authors, description=description
    )

    exported_files = {"dataset_description": dataset_desc}

    # Create participants.tsv from subject IDs (empty list if no data)
    participants = [
        {"participant_id": sid.replace("sub-", "") if sid.startswith("sub-") else sid}
        for sid in data.keys()
    ]
    participants_file = exporter.create_participants_tsv(participants)
    exported_files["participants"] = participants_file

    # Export each data type
    for subject_id, subject_data in data.items():
        if isinstance(subject_data, dict):
            # Export EEG data
            if "eeg" in subject_data:
                eeg_data = subject_data["eeg"]
                eeg_file = exporter.export_eeg_data(
                    data=eeg_data.get("data"),
                    subject_id=subject_id,
                    task_name=eeg_data.get("task", "rest"),
                    channels=eeg_data.get("channels"),
                    sampling_rate=eeg_data.get("sampling_rate", 1000.0),
                    metadata=eeg_data.get("metadata"),
                )
                exported_files[f"{subject_id}_eeg"] = eeg_file

            # Export behavioral data
            if "behavioral" in subject_data:
                beh_data = subject_data["behavioral"]
                beh_file = exporter.export_behavioral_data(
                    data=beh_data.get("data"),
                    subject_id=subject_id,
                    task_name=beh_data.get("task", "experiment"),
                    metadata=beh_data.get("metadata"),
                )
                exported_files[f"{subject_id}_behavioral"] = beh_file

            # Export physiological data
            if "physiological" in subject_data:
                physio_data = subject_data["physiological"]
                physio_files = exporter.export_physiological_data(
                    data=physio_data.get("data"),
                    subject_id=subject_id,
                    task_name=physio_data.get("task", "rest"),
                    sampling_rates=physio_data.get("sampling_rates"),
                    metadata=physio_data.get("metadata"),
                )
                exported_files.update({f"{subject_id}_{k}": v for k, v in physio_files.items()})

    # Validate structure
    validation = exporter.validate_bids_structure()
    if validation["errors"]:
        warnings.warn(f"BIDS validation errors: {validation['errors']}")

    return exported_files


if __name__ == "__main__":
    # Example usage
    exporter = BIDSExporter("data/test_bids_dataset")

    # Create sample data
    eeg_data = np.random.rand(8, 60000)  # 8 channels, 60 seconds at 1kHz
    channels = ["Fz", "Cz", "Pz", "Oz", "F3", "F4", "P3", "P4"]

    # Export EEG data
    eeg_file = exporter.export_eeg_data(
        data=eeg_data,
        subject_id="01",
        task_name="rest",
        channels=channels,
        sampling_rate=1000.0,
    )

    logger.info(f"Exported EEG data to: {eeg_file}")

    # Create sample behavioral data
    beh_data = pd.DataFrame(
        {
            "trial_number": range(1, 101),
            "reaction_time": np.random.normal(300, 50, 100),
            "accuracy": np.random.choice([0, 1], 100),
            "condition": np.random.choice(["easy", "hard"], 100),
        }
    )

    # Export behavioral data
    beh_file = exporter.export_behavioral_data(
        data=beh_data, subject_id="01", task_name="experiment"
    )

    logger.info(f"Exported behavioral data to: {beh_file}")

    # Validate structure
    validation = exporter.validate_bids_structure()
    logger.info(f"Validation results: {validation}")
