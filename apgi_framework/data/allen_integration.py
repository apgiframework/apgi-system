"""Allen Brain Map Dataset Integration for APGI Framework.

This module provides specialized loaders for Allen Brain Map datasets, which
are critical for validating the Landauer Bridge (E_min ≥ kTln2) through
high-resolution two-photon calcium imaging data.

Primary Datasets:
- Visual Coding: Two-Photon Calcium Imaging (PCI-like complexity metrics)
- Cell Types: Electrophysiology (cross-species complexity gradient)

References:
- Allen Brain Map: https://portal.brain-map.org/
- Visual Coding: https://portal.brain-map.org/explore/circuits/visual-coding-2p
- Allen SDK: https://github.com/AllenInstitute/AllenSDK
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from .unified_data_loader import UnifiedDataLoader, load_allen_dataset

logger = logging.getLogger(__name__)


@dataclass
class AllenVisualCodingDataset:
    """Allen Visual Coding: Two-Photon Calcium Imaging dataset.

    This dataset contains single-neuron activity from thousands of neurons
    simultaneously, enabling calculation of PCI-like complexity metrics and
    reservoir state vector x(t) to validate Liquid State Machine layer
    dynamics [§10].

    Attributes:
        fluorescence: Fluorescence traces (ΔF/F0) for each neuron
        time: Time vector in milliseconds
        neuron_ids: Unique identifiers for each neuron
        stimulus_id: Identifier for the visual stimulus
        metadata: Dataset metadata
        sampling_rate_hz: Sampling rate
        brain_region: Recorded brain region
        imaging_depth_um: Imaging depth in micrometers
    """

    fluorescence: npt.NDArray[np.float64]  # Shape: (time, neurons)
    time: npt.NDArray[np.float64]
    neuron_ids: List[str] = field(default_factory=list)
    stimulus_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    sampling_rate_hz: float = 30.0  # Typical for 2P imaging
    brain_region: str = "VISp"
    imaging_depth_um: float = 0.0

    def __post_init__(self) -> None:
        """Validate data dimensions."""
        if self.fluorescence.ndim != 2:
            raise ValueError(
                f"Fluorescence must be 2D (time, neurons), got shape {self.fluorescence.shape}"
            )

        if len(self.time) != self.fluorescence.shape[0]:
            raise ValueError(
                f"Time vector length ({len(self.time)}) must match fluorescence first dimension ({self.fluorescence.shape[0]})"
            )

        if not self.neuron_ids:
            self.neuron_ids = [f"neuron_{i}" for i in range(self.fluorescence.shape[1])]

    @property
    def num_neurons(self) -> int:
        """Number of recorded neurons."""
        return self.fluorescence.shape[1]

    @property
    def duration_ms(self) -> float:
        """Total duration in milliseconds."""
        if len(self.time) < 2:
            return 0.0
        return float(self.time[-1] - self.time[0])

    def compute_pci_like_complexity(
        self, time_window_ms: float = 500.0, n_bins: int = 50
    ) -> Dict[str, float]:
        """Compute PCI-like complexity metric from neural activity.

        This metric quantifies the complexity of neural activity patterns,
        analogous to the Perturbational Complexity Index (PCI) used in
        TMS-EEG studies.

        Args:
            time_window_ms: Time window for analysis (ms)
            n_bins: Number of bins for discretization

        Returns:
            Dictionary with complexity metrics
        """
        # Select time window
        window_samples = int(time_window_ms * self.sampling_rate_hz / 1000.0)
        window_samples = min(window_samples, len(self.time))

        # Get fluorescence in window
        window_data = self.fluorescence[:window_samples, :]

        # Normalize each neuron's trace
        normalized_data = (window_data - np.mean(window_data, axis=0)) / (
            np.std(window_data, axis=0) + 1e-10
        )

        # Compute spatial correlation matrix
        corr_matrix = np.corrcoef(normalized_data.T)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(corr_matrix)
        eigenvalues = np.real(eigenvalues)

        # Compute participation ratio (effective dimensionality)
        participation_ratio = np.sum(eigenvalues) ** 2 / np.sum(eigenvalues**2)

        # Compute entropy of eigenvalue distribution
        eigenvalue_probs = eigenvalues / (np.sum(eigenvalues) + 1e-10)
        eigenvalue_probs = eigenvalue_probs[eigenvalue_probs > 0]
        spectral_entropy = -np.sum(eigenvalue_probs * np.log(eigenvalue_probs))

        # Compute Lempel-Ziv complexity (simplified)
        # Convert to binary string by thresholding
        binary_data = (normalized_data > 0).astype(int)
        lz_complexity = self._compute_lempel_ziv(binary_data.flatten())

        return {
            "participation_ratio": float(participation_ratio),
            "spectral_entropy": float(spectral_entropy),
            "lempel_ziv_complexity": float(lz_complexity),
            "num_neurons": self.num_neurons,
            "effective_dimensionality": float(participation_ratio),
        }

    def _compute_lempel_ziv(self, binary_sequence: npt.NDArray[np.int64]) -> int:
        """Compute Lempel-Ziv complexity of binary sequence.

        Args:
            binary_sequence: Binary sequence (0s and 1s)

        Returns:
            Lempel-Ziv complexity
        """
        n = len(binary_sequence)
        complexity = 1
        i = 0

        while i < n - 1:
            j = i + 1
            while j < n:
                # Check if substring binary_sequence[i:j+1] has been seen before
                substring = binary_sequence[i : j + 1]
                if self._substring_seen(binary_sequence, substring, i):
                    j += 1
                else:
                    complexity += 1
                    i = j
                    break
            i = j

        return complexity

    def _substring_seen(
        self, sequence: npt.NDArray[np.int64], substring: npt.NDArray[np.int64], current_pos: int
    ) -> bool:
        """Check if substring has been seen before current position."""
        if len(substring) == 1:
            return True

        for i in range(current_pos):
            if i + len(substring) <= len(sequence):
                if np.array_equal(sequence[i : i + len(substring)], substring):
                    return True
        return False

    def compute_reservoir_state(
        self, reservoir_size: int = 100, time_window_ms: float = 100.0
    ) -> npt.NDArray[np.float64]:
        """Compute reservoir state vector x(t) for Liquid State Machine validation.

        The reservoir state represents the high-dimensional projection of neural
        activity, which is used in LSM models for temporal processing.

        Args:
            reservoir_size: Dimensionality of reservoir state
            time_window_ms: Time window for state computation

        Returns:
            Reservoir state vector
        """
        window_samples = int(time_window_ms * self.sampling_rate_hz / 1000.0)
        window_samples = min(window_samples, len(self.time))

        # Get fluorescence in window
        window_data = self.fluorescence[:window_samples, :]

        # Perform PCA to reduce dimensionality
        from sklearn.decomposition import PCA

        pca = PCA(n_components=min(reservoir_size, self.num_neurons))
        reservoir_state = pca.fit_transform(window_data.T)

        return reservoir_state.T  # Shape: (reservoir_size, time)

    def detect_ignition_events(
        self, threshold_std: float = 3.0, min_duration_ms: float = 50.0
    ) -> List[Dict[str, Any]]:
        """Detect ignition events from population activity.

        Ignition events are characterized by widespread, sustained neural
        activity across many neurons.

        Args:
            threshold_std: Threshold in standard deviations
            min_duration_ms: Minimum duration for ignition

        Returns:
            List of detected ignition events
        """
        # Compute population activity (mean across neurons)
        population_activity = np.mean(self.fluorescence, axis=1)

        # Detect bursts
        baseline = np.mean(population_activity[:10])
        threshold = baseline + threshold_std * np.std(population_activity[:10])

        above_threshold = population_activity > threshold
        changes = np.diff(above_threshold.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        ignition_events = []
        for start, end in zip(starts, ends):
            duration_ms = (end - start) * 1000.0 / self.sampling_rate_hz
            if duration_ms > min_duration_ms:
                # Compute fraction of neurons active during ignition
                activity_window = self.fluorescence[start:end, :]
                neuron_fractions = np.mean(activity_window > threshold, axis=0)
                active_fraction = np.mean(neuron_fractions > 0.5)

                ignition_events.append(
                    {
                        "start_ms": float(self.time[start]),
                        "end_ms": float(self.time[end]),
                        "duration_ms": duration_ms,
                        "peak_activity": float(np.max(population_activity[start:end])),
                        "active_neuron_fraction": float(active_fraction),
                        "num_active_neurons": int(np.sum(neuron_fractions > 0.5)),
                    }
                )

        return ignition_events


class AllenLoader:
    """Specialized loader for Allen Brain Map datasets.

    Provides convenience methods for loading specific Allen datasets with
    appropriate preprocessing for APGI validation protocols.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize Allen loader.

        Args:
            cache_dir: Directory for caching downloaded datasets
        """
        self.cache_dir = cache_dir or Path.home() / ".apgi" / "datasets" / "allen"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.unified_loader = UnifiedDataLoader(cache_dir=cache_dir)

    def load_visual_coding(
        self, file_path: Union[str, Path], **kwargs: Any
    ) -> AllenVisualCodingDataset:
        """Load Allen Visual Coding dataset (Two-Photon Calcium Imaging).

        Args:
            file_path: Path to Visual Coding dataset file (.nwb format)
            **kwargs: Additional arguments for loader

        Returns:
            AllenVisualCodingDataset with fluorescence data
        """
        # Load using unified loader
        dataset = load_allen_dataset(file_path, **kwargs)

        # Extract metadata
        stimulus_id = dataset.metadata.get("stimulus_id", "")
        brain_region = dataset.metadata.get("brain_region", "VISp")
        imaging_depth = dataset.metadata.get("imaging_depth_um", 0.0)

        # Extract neuron IDs from metadata or generate defaults
        neuron_ids = dataset.channel_names
        if not neuron_ids or len(neuron_ids) != dataset.data.shape[1]:
            neuron_ids = [f"neuron_{i}" for i in range(dataset.data.shape[1])]

        return AllenVisualCodingDataset(
            fluorescence=dataset.data.astype(np.float64),
            time=dataset.time,
            neuron_ids=neuron_ids,
            stimulus_id=stimulus_id,
            metadata=dataset.metadata,
            sampling_rate_hz=dataset.sampling_rate_hz,
            brain_region=brain_region,
            imaging_depth_um=float(imaging_depth),
        )

    def download_visual_coding(self, experiment_id: str, output_dir: Optional[Path] = None) -> Path:
        """Download Allen Visual Coding dataset using Allen SDK.

        Note: This requires Allen SDK installation and API access.
        Users should download manually from https://portal.brain-map.org/

        Args:
            experiment_id: Allen experiment ID
            output_dir: Directory to save downloaded data

        Returns:
            Path to downloaded data directory
        """
        output_dir = output_dir or self.cache_dir / "visual-coding"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Placeholder for Allen SDK download
            # from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
            logger.info(f"Downloading Allen Visual Coding dataset {experiment_id}...")
            logger.warning(
                "Allen SDK download requires authentication. "
                "Please download manually from https://portal.brain-map.org/ "
                f"and place in {output_dir}"
            )

        except ImportError:
            logger.warning(
                "Allen SDK not installed. Install with: pip install allensdk. "
                "Please download manually from https://portal.brain-map.org/"
            )

        return output_dir

    def list_available_experiments(self) -> List[str]:
        """List available Allen Brain Map experiments.

        Returns:
            List of experiment IDs
        """
        # This would query the Allen Brain Map API
        # For now, return a placeholder list
        return [
            "visual-coding-2p",
            "cell-types-electrophysiology",
            "mouse-connectivity",
        ]
