"""Unified Data Loader for HDF5-based Formats (.h5, .mat, .nwb).

This module provides a unified interface for loading neuroscience datasets in
HDF5-based formats, specifically for validating the Landauer Bridge (E_min ≥ kTln2).

Supported Formats:
- .h5: HDF5 files (native h5py format)
- .mat: MATLAB files (v7.3 HDF5-based format via scipy.io.loadmat)
- .nwb: Neurodata Without Borders (HDF5-based via pynwb)

Primary Use Cases:
1. CRCNS.org datasets (.mat format) - Visual/Auditory cortex electrophysiology
2. Allen Brain Map datasets (.nwb format) - Two-photon calcium imaging
3. Hugging Face neuromorphic datasets (.h5 format) - Event-based spiking data

References:
- CRCNS: https://crcns.org/
- Allen Brain Map: https://portal.brain-map.org/
- Neurodata Without Borders: https://www.nwb.org/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import h5py
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class LoadedDataset:
    """Unified container for loaded neuroscience data.

    Attributes:
        data: Main data array (e.g., fluorescence traces, spike times, LFP)
        time: Time vector in milliseconds
        metadata: Dictionary of metadata from the file
        source_format: Original format of the file
        source_path: Path to the source file
        sampling_rate_hz: Sampling rate in Hz
        channel_names: Names of recording channels (if available)
        unit: Physical unit of the data (e.g., "ΔF/F0", "spikes/s", "μV")
    """

    data: npt.NDArray[np.float64]
    time: npt.NDArray[np.float64]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_format: Literal["h5", "mat", "nwb"] = "h5"
    source_path: Optional[Path] = None
    sampling_rate_hz: float = 1000.0
    channel_names: List[str] = field(default_factory=list)
    unit: str = "arbitrary"

    def __post_init__(self) -> None:
        """Validate data dimensions."""
        if len(self.data.shape) not in [1, 2]:
            raise ValueError(f"Data must be 1D or 2D, got shape {self.data.shape}")

        if len(self.time) != self.data.shape[0]:
            raise ValueError(
                f"Time vector length ({len(self.time)}) must match data first dimension ({self.data.shape[0]})"
            )

    @property
    def duration_ms(self) -> float:
        """Total duration in milliseconds."""
        if len(self.time) < 2:
            return 0.0
        return float(self.time[-1] - self.time[0])

    @property
    def num_channels(self) -> int:
        """Number of channels (1 for 1D data)."""
        if len(self.data.shape) == 1:
            return 1
        return self.data.shape[1]

    @property
    def temporal_resolution_ms(self) -> float:
        """Temporal resolution in milliseconds."""
        if len(self.time) < 2:
            return 1000.0 / self.sampling_rate_hz
        return float(np.mean(np.diff(self.time)))


class UnifiedDataLoader:
    """Unified loader for HDF5-based neuroscience data formats.

    Provides a consistent interface for loading .h5, .mat, and .nwb files,
    with automatic format detection and standardized output.

    Example:
        loader = UnifiedDataLoader()
        dataset = loader.load("data.mat")  # Auto-detects format
        print(f"Loaded {dataset.source_format} file with {dataset.num_channels} channels")
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the unified data loader.

        Args:
            cache_dir: Directory for caching downloaded datasets
        """
        self.cache_dir = cache_dir or Path.home() / ".apgi" / "datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Format availability flags
        self.h5_available = True
        self.mat_available = True
        self.nwb_available = self._check_pynwb()

    def _check_pynwb(self) -> bool:
        """Check if pynwb is available."""
        try:
            import pynwb  # noqa: F401

            return True
        except ImportError:
            logger.warning("pynwb not available. .nwb files will not be loadable.")
            return False

    def load(
        self,
        file_path: Union[str, Path],
        format: Optional[Literal["h5", "mat", "nwb", "auto"]] = "auto",
        **kwargs: Any,
    ) -> LoadedDataset:
        """Load a dataset from file with automatic format detection.

        Args:
            file_path: Path to the data file
            format: Format of the file ("auto" for automatic detection)
            **kwargs: Additional format-specific arguments

        Returns:
            LoadedDataset with standardized structure

        Raises:
            ValueError: If file format is unsupported or file cannot be loaded
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect format if not specified
        if format == "auto":
            format = self._detect_format(file_path)

        # Load based on format
        if format == "h5":
            return self._load_h5(file_path, **kwargs)
        elif format == "mat":
            return self._load_mat(file_path, **kwargs)
        elif format == "nwb":
            return self._load_nwb(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _detect_format(self, file_path: Path) -> Literal["h5", "mat", "nwb"]:
        """Detect file format from extension and content."""
        suffix = file_path.suffix.lower()

        if suffix == ".h5" or suffix == ".hdf5":
            return "h5"
        elif suffix == ".mat":
            return "mat"
        elif suffix == ".nwb":
            return "nwb"
        else:
            # Try to detect from content
            try:
                import h5py

                with h5py.File(file_path, "r") as f:
                    # Check for NWB-specific groups
                    if "processing" in f or "acquisition" in f:
                        return "nwb"
                    # Check for MATLAB-specific attributes
                    if "#refs#" in f or "MATLAB_class" in f.attrs:
                        return "mat"
                    # Default to h5
                    return "h5"
            except Exception:
                raise ValueError(f"Cannot detect format for {file_path}")

    def _load_h5(self, file_path: Path, **kwargs: Any) -> LoadedDataset:
        """Load HDF5 file.

        Args:
            file_path: Path to .h5 file
            **kwargs: Additional arguments (group_path, dataset_name)

        Returns:
            LoadedDataset
        """
        import h5py

        group_path = kwargs.get("group_path", "/")
        dataset_name = kwargs.get("dataset_name", None)

        with h5py.File(file_path, "r") as f:
            # Navigate to group
            if group_path != "/":
                group = f[group_path]
            else:
                group = f

            # Find dataset
            if dataset_name is None:
                # Try to find the main dataset
                dataset_name = self._find_main_dataset(group)

            if dataset_name not in group:
                raise ValueError(f"Dataset '{dataset_name}' not found in {file_path}")

            data_array = group[dataset_name][:]

            # Extract metadata
            metadata = self._extract_h5_metadata(group, dataset_name)

            # Extract time vector
            time_vector = self._extract_time_vector(group, metadata, data_array.shape[0])

            # Extract channel names
            channel_names = self._extract_channel_names(group, dataset_name, data_array.shape)

        return LoadedDataset(
            data=data_array.astype(np.float64),
            time=time_vector.astype(np.float64),
            metadata=metadata,
            source_format="h5",
            source_path=file_path,
            sampling_rate_hz=metadata.get("sampling_rate_hz", 1000.0),
            channel_names=channel_names,
            unit=metadata.get("unit", "arbitrary"),
        )

    def _load_mat(self, file_path: Path, **kwargs: Any) -> LoadedDataset:
        """Load MATLAB file (v7.3 HDF5-based format).

        Args:
            file_path: Path to .mat file
            **kwargs: Additional arguments (variable_name)

        Returns:
            LoadedDataset
        """
        from scipy.io import loadmat

        variable_name = kwargs.get("variable_name", None)

        # Load MATLAB file
        mat_data = loadmat(file_path, struct_as_record=False, squeeze_me=True)

        # Find main variable
        if variable_name is None:
            variable_name = self._find_main_mat_variable(mat_data)

        if variable_name not in mat_data:
            raise ValueError(f"Variable '{variable_name}' not found in {file_path}")

        data_array = mat_data[variable_name]

        # Convert to numpy array if needed
        if not isinstance(data_array, np.ndarray):
            data_array = np.array(data_array)

        # Extract metadata
        metadata = {
            "matlab_header": mat_data.get("__header__", b"").decode("utf-8", errors="ignore"),
            "matlab_version": mat_data.get("__version__", ""),
            "variable_name": variable_name,
        }

        # Extract time vector (common in MATLAB files)
        time_vector = self._extract_time_from_mat(mat_data, data_array.shape[0])

        # Extract channel names
        channel_names = self._extract_channel_names_from_mat(
            mat_data, variable_name, data_array.shape
        )

        return LoadedDataset(
            data=data_array.astype(np.float64),
            time=time_vector.astype(np.float64),
            metadata=metadata,
            source_format="mat",
            source_path=file_path,
            sampling_rate_hz=metadata.get("sampling_rate_hz", 1000.0),
            channel_names=channel_names,
            unit=metadata.get("unit", "arbitrary"),
        )

    def _load_nwb(self, file_path: Path, **kwargs: Any) -> LoadedDataset:
        """Load Neurodata Without Borders file.

        Args:
            file_path: Path to .nwb file
            **kwargs: Additional arguments (processing_module, acquisition_name)

        Returns:
            LoadedDataset
        """
        if not self.nwb_available:
            raise ImportError("pynwb is required for .nwb files. Install: pip install pynwb")

        from pynwb import NWBHDF5IO

        processing_module = kwargs.get("processing_module", None)
        acquisition_name = kwargs.get("acquisition_name", None)

        with NWBHDF5IO(str(file_path), "r") as io:
            nwbfile = io.read()

            # Try to get data from processing module
            if processing_module and processing_module in nwbfile.processing:
                module = nwbfile.processing[processing_module]
                # Get first data interface
                data_interface = list(module.data_interfaces.values())[0]
                data_array = data_interface.data[:]
                time_vector = data_interface.timestamps[:]
                metadata = {"processing_module": processing_module}

            # Try to get data from acquisition
            elif acquisition_name and acquisition_name in nwbfile.acquisition:
                acquisition = nwbfile.acquisition[acquisition_name]
                data_array = acquisition.data[:]
                time_vector = (
                    acquisition.timestamps[:]
                    if hasattr(acquisition, "timestamps")
                    else np.arange(len(data_array))
                )
                metadata = {"acquisition_name": acquisition_name}

            # Default: try to get first acquisition
            elif nwbfile.acquisition:
                acquisition_name = list(nwbfile.acquisition.keys())[0]
                acquisition = nwbfile.acquisition[acquisition_name]
                data_array = acquisition.data[:]
                time_vector = (
                    acquisition.timestamps[:]
                    if hasattr(acquisition, "timestamps")
                    else np.arange(len(data_array))
                )
                metadata = {"acquisition_name": acquisition_name}

            else:
                raise ValueError(f"No suitable data found in {file_path}")

            # Extract additional metadata
            metadata.update(
                {
                    "session_description": nwbfile.session_description,
                    "identifier": nwbfile.identifier,
                    "session_start_time": str(nwbfile.session_start_time),
                }
            )

            # Extract channel names from electrodes if available
            channel_names = []
            if hasattr(nwbfile, "electrodes") and nwbfile.electrodes is not None:
                channel_names = list(nwbfile.electrodes["id"][:])

            # Get sampling rate
            sampling_rate = metadata.get("sampling_rate", 1000.0)
            if hasattr(acquisition, "rate"):
                sampling_rate = acquisition.rate

        return LoadedDataset(
            data=data_array.astype(np.float64),
            time=time_vector.astype(np.float64),
            metadata=metadata,
            source_format="nwb",
            source_path=file_path,
            sampling_rate_hz=float(sampling_rate),
            channel_names=channel_names,
            unit=metadata.get("unit", "arbitrary"),
        )

    def _find_main_dataset(self, group: Any) -> str:
        """Find the main dataset in an HDF5 group."""
        # Look for common dataset names
        common_names = ["data", "traces", "signal", "fluorescence", "spikes", "lfp", "voltage"]

        for name in common_names:
            if name in group and isinstance(group[name], h5py.Dataset):
                return name

        # Return first dataset
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset):
                return key

        raise ValueError("No dataset found in group")

    def _find_main_mat_variable(self, mat_data: Dict[str, Any]) -> str:
        """Find the main variable in a MATLAB file."""
        # Skip internal MATLAB variables
        skip_vars = {"__header__", "__version__", "__globals__"}

        # Look for common variable names
        common_names = ["data", "traces", "signal", "fluorescence", "spikes", "lfp", "voltage"]

        for name in common_names:
            if name in mat_data and name not in skip_vars:
                return name

        # Return first non-internal variable
        for key in mat_data.keys():
            if key not in skip_vars:
                return key

        raise ValueError("No suitable variable found in MATLAB file")

    def _extract_h5_metadata(self, group: Any, dataset_name: str) -> Dict[str, Any]:
        """Extract metadata from HDF5 group."""
        metadata = {}

        # Extract dataset attributes
        if dataset_name in group:
            dataset = group[dataset_name]
            for attr_name, attr_value in dataset.attrs.items():
                metadata[attr_name] = self._convert_attr_value(attr_value)

        # Extract group attributes
        for attr_name, attr_value in group.attrs.items():
            metadata[attr_name] = self._convert_attr_value(attr_value)

        return metadata

    def _convert_attr_value(self, value: Any) -> Any:
        """Convert HDF5 attribute to Python type."""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        elif isinstance(value, (np.ndarray, list)):
            if len(value) == 1:
                return value[0]
            return list(value)
        return value

    def _extract_time_vector(
        self, group: Any, metadata: Dict[str, Any], length: int
    ) -> npt.NDArray[np.float64]:
        """Extract time vector from HDF5 group."""
        # Try to find time dataset
        time_names = ["time", "time_ms", "timestamps", "t"]

        for name in time_names:
            if name in group and isinstance(group[name], h5py.Dataset):
                time_data = group[name][:]
                if len(time_data) == length:
                    return time_data.astype(np.float64)

        # Generate from sampling rate
        sampling_rate = metadata.get("sampling_rate_hz", 1000.0)
        return np.arange(length, dtype=np.float64) * (1000.0 / sampling_rate)

    def _extract_channel_names(self, group: Any, dataset_name: str, shape: tuple) -> List[str]:
        """Extract channel names from HDF5 group."""
        channel_names = []

        # Try to find channel names dataset
        channel_names_datasets = ["channel_names", "channels", "electrodes", "labels"]

        for name in channel_names_datasets:
            if name in group and isinstance(group[name], h5py.Dataset):
                names = group[name][:]
                if len(names) == shape[1] if len(shape) == 2 else len(names):
                    channel_names = [self._convert_attr_value(n) for n in names]
                    break

        # Generate default names if not found
        if not channel_names:
            num_channels = shape[1] if len(shape) == 2 else 1
            channel_names = [f"channel_{i}" for i in range(num_channels)]

        return channel_names

    def _extract_time_from_mat(
        self, mat_data: Dict[str, Any], length: int
    ) -> npt.NDArray[np.float64]:
        """Extract time vector from MATLAB data."""
        time_names = ["time", "time_ms", "timestamps", "t"]

        for name in time_names:
            if name in mat_data:
                time_data = mat_data[name]
                if isinstance(time_data, np.ndarray) and len(time_data) == length:
                    return time_data.astype(np.float64)

        # Generate default time vector
        return np.arange(length, dtype=np.float64)

    def _extract_channel_names_from_mat(
        self, mat_data: Dict[str, Any], variable_name: str, shape: tuple
    ) -> List[str]:
        """Extract channel names from MATLAB data."""
        channel_names = []

        # Try to find channel names variable
        channel_names_vars = ["channel_names", "channels", "electrodes", "labels"]

        for name in channel_names_vars:
            if name in mat_data:
                names = mat_data[name]
                if isinstance(names, np.ndarray):
                    if len(names) == shape[1] if len(shape) == 2 else len(names):
                        channel_names = [str(n) for n in names]
                        break

        # Generate default names if not found
        if not channel_names:
            num_channels = shape[1] if len(shape) == 2 else 1
            channel_names = [f"channel_{i}" for i in range(num_channels)]

        return channel_names


# Convenience functions for specific dataset types


def load_crcns_dataset(file_path: Union[str, Path], **kwargs: Any) -> LoadedDataset:
    """Load CRCNS dataset (typically .mat format).

    CRCNS datasets include:
    - V1-1: Visual Cortex Spiking and LFP
    - AC-1: Auditory Cortex Metabolic Proxies

    Args:
        file_path: Path to CRCNS dataset file
        **kwargs: Additional arguments for UnifiedDataLoader

    Returns:
        LoadedDataset with CRCNS-specific metadata
    """
    loader = UnifiedDataLoader()
    dataset = loader.load(file_path, format="mat", **kwargs)
    dataset.metadata["source"] = "CRCNS"
    return dataset


def load_allen_dataset(file_path: Union[str, Path], **kwargs: Any) -> LoadedDataset:
    """Load Allen Brain Map dataset (typically .nwb format).

    Allen datasets include:
    - Visual Coding: Two-Photon Calcium Imaging
    - Cell Types: Electrophysiology

    Args:
        file_path: Path to Allen dataset file
        **kwargs: Additional arguments for UnifiedDataLoader

    Returns:
        LoadedDataset with Allen-specific metadata
    """
    loader = UnifiedDataLoader()
    dataset = loader.load(file_path, format="nwb", **kwargs)
    dataset.metadata["source"] = "Allen Brain Map"
    return dataset


def load_huggingface_dataset(file_path: Union[str, Path], **kwargs: Any) -> LoadedDataset:
    """Load Hugging Face neuromorphic dataset (typically .h5 format).

    Hugging Face datasets include:
    - Tonic: Spiking-Dataset-Collection
    - N-MNIST, DVS-Gestures (event-based)

    Args:
        file_path: Path to Hugging Face dataset file
        **kwargs: Additional arguments for UnifiedDataLoader

    Returns:
        LoadedDataset with Hugging Face-specific metadata
    """
    loader = UnifiedDataLoader()
    dataset = loader.load(file_path, format="h5", **kwargs)
    dataset.metadata["source"] = "Hugging Face"
    return dataset
