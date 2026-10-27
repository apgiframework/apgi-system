"""Hugging Face Neuromorphic Dataset Integration for APGI Framework.

This module provides specialized loaders for Hugging Face neuromorphic datasets,
which are critical for validating the Landauer Bridge (E_min ≥ kTln2) through
event-based spiking data that enables testing of the Allostatic Threshold θ(t).

Primary Datasets:
- Tonic: Spiking-Dataset-Collection (N-MNIST, DVS-Gestures, etc.)
- Event-based sensors (DVS) for energy-efficient neuromorphic computing

References:
- Hugging Face Datasets: https://huggingface.co/datasets
- Tonic: https://tonic.readthedocs.io/
- N-MNIST: Neuromorphic MNIST dataset
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from .unified_data_loader import UnifiedDataLoader, load_huggingface_dataset

logger = logging.getLogger(__name__)


@dataclass
class NeuromorphicDataset:
    """Hugging Face neuromorphic dataset (event-based spiking data).

    Event-based systems only consume power when pixels change, providing an
    ideal substrate for testing the Allostatic Threshold θ(t)'s ability to
    minimize metabolic cost while maximizing information value V(t) [§4].

    Attributes:
        events: Event data (x, y, timestamp, polarity)
        time: Time vector in milliseconds
        metadata: Dataset metadata
        sensor_type: Type of event-based sensor (DVS, etc.)
        spatial_resolution: Spatial resolution (height, width)
        temporal_resolution_us: Temporal resolution in microseconds
    """

    events: npt.NDArray[np.float64]  # Shape: (n_events, 4) or (n_events, 3)
    time: npt.NDArray[np.float64]
    metadata: Dict[str, Any] = field(default_factory=dict)
    sensor_type: str = "DVS"
    spatial_resolution: tuple[int, int] = (128, 128)
    temporal_resolution_us: float = 1.0

    def __post_init__(self) -> None:
        """Validate event data structure."""
        if self.events.ndim != 2:
            raise ValueError(f"Events must be 2D, got shape {self.events.shape}")

        if self.events.shape[1] not in [3, 4]:
            raise ValueError(
                f"Events must have 3 or 4 columns (x, y, t, [polarity]), got {self.events.shape[1]}"
            )

    @property
    def num_events(self) -> int:
        """Total number of events."""
        return self.events.shape[0]

    @property
    def duration_ms(self) -> float:
        """Total duration in milliseconds."""
        if len(self.time) < 2:
            return 0.0
        return float(self.time[-1] - self.time[0])

    def compute_event_rate(self, window_ms: float = 100.0) -> npt.NDArray[np.float64]:
        """Compute event rate over time windows.

        Args:
            window_ms: Window size in milliseconds

        Returns:
            Event rate in events per second
        """
        window_samples = int(window_ms * 1000.0 / self.temporal_resolution_us)
        n_windows = int(np.ceil(self.num_events / window_samples))
        event_rates = np.zeros(n_windows)

        for i in range(n_windows):
            start_idx = i * window_samples
            end_idx = min((i + 1) * window_samples, self.num_events)
            window_events = end_idx - start_idx
            event_rates[i] = window_events / (window_ms / 1000.0)

        return event_rates

    def convert_to_spike_train(
        self, spatial_bins: tuple[int, int] = (32, 32), temporal_bins: int = 100
    ) -> npt.NDArray[np.float64]:
        """Convert event data to spike train raster.

        Args:
            spatial_bins: Number of spatial bins (height, width)
            temporal_bins: Number of temporal bins

        Returns:
            Spike train raster (temporal_bins, spatial_bins[0], spatial_bins[1])
        """
        # Create raster
        spike_raster = np.zeros((temporal_bins, spatial_bins[0], spatial_bins[1]))

        # Normalize spatial coordinates
        x_norm = (self.events[:, 0] / self.spatial_resolution[1] * spatial_bins[1]).astype(int)
        y_norm = (self.events[:, 1] / self.spatial_resolution[0] * spatial_bins[0]).astype(int)

        # Normalize temporal coordinates
        t_norm = (self.events[:, 2] / self.duration_ms * temporal_bins).astype(int)

        # Clip to valid ranges
        x_norm = np.clip(x_norm, 0, spatial_bins[1] - 1)
        y_norm = np.clip(y_norm, 0, spatial_bins[0] - 1)
        t_norm = np.clip(t_norm, 0, temporal_bins - 1)

        # Fill raster
        for t, y, x in zip(t_norm, y_norm, x_norm):
            spike_raster[t, y, x] += 1

        return spike_raster

    def compute_information_value(
        self, spike_raster: Optional[npt.NDArray[np.float64]] = None
    ) -> Dict[str, float]:
        """Compute information value V(t) from spike train.

        Information value quantifies the information content of neural activity,
        which should be maximized while minimizing metabolic cost.

        Args:
            spike_raster: Pre-computed spike train (optional)

        Returns:
            Dictionary with information value metrics
        """
        if spike_raster is None:
            spike_raster = self.convert_to_spike_train()

        # Compute entropy of spatial activity pattern
        spatial_activity = np.sum(spike_raster, axis=0)  # Sum over time
        spatial_probs = spatial_activity / (np.sum(spatial_activity) + 1e-10)
        spatial_entropy = -np.sum(spatial_probs * np.log(spatial_probs + 1e-10))

        # Compute entropy of temporal activity pattern
        temporal_activity = np.sum(spike_raster, axis=(1, 2))  # Sum over space
        temporal_probs = temporal_activity / (np.sum(temporal_activity) + 1e-10)
        temporal_entropy = -np.sum(temporal_probs * np.log(temporal_probs + 1e-10))

        # Compute total entropy
        total_activity = np.sum(spike_raster)
        total_probs = spike_raster / (total_activity + 1e-10)
        total_entropy = -np.sum(total_probs * np.log(total_probs + 1e-10))

        return {
            "spatial_entropy": float(spatial_entropy),
            "temporal_entropy": float(temporal_entropy),
            "total_entropy": float(total_entropy),
            "total_events": float(total_activity),
        }

    def compute_allostatic_threshold(
        self,
        spike_raster: Optional[npt.NDArray[np.float64]] = None,
        target_information: float = 0.5,
    ) -> Dict[str, float]:
        """Compute Allostatic Threshold θ(t) for metabolic cost minimization.

        The Allostatic Threshold represents the optimal balance between
        metabolic cost and information value, minimizing cost while
        maximizing information.

        Args:
            spike_raster: Pre-computed spike train (optional)
            target_information: Target information value (0-1)

        Returns:
            Dictionary with threshold metrics
        """
        if spike_raster is None:
            spike_raster = self.convert_to_spike_train()

        # Compute information value
        info_metrics = self.compute_information_value(spike_raster)
        current_info = info_metrics["total_entropy"]

        # Normalize information to [0, 1]
        max_entropy = np.log2(spike_raster.size)
        normalized_info = current_info / max_entropy

        # Compute allostatic threshold
        # θ(t) should be low when information is high (efficient processing)
        # θ(t) should be high when information is low (need more resources)
        threshold = 1.0 - normalized_info

        # Compute metabolic cost proxy (proportional to total events)
        metabolic_cost = info_metrics["total_events"]

        # Compute efficiency (information per metabolic cost)
        efficiency = current_info / (metabolic_cost + 1e-10)

        return {
            "allostatic_threshold": float(threshold),
            "normalized_information": float(normalized_info),
            "metabolic_cost_proxy": float(metabolic_cost),
            "efficiency": float(efficiency),
            "target_information": target_information,
        }


class HuggingFaceLoader:
    """Specialized loader for Hugging Face neuromorphic datasets.

    Provides convenience methods for loading specific Hugging Face datasets
    with appropriate preprocessing for APGI validation protocols.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize Hugging Face loader.

        Args:
            cache_dir: Directory for caching downloaded datasets
        """
        self.cache_dir = cache_dir or Path.home() / ".apgi" / "datasets" / "huggingface"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.unified_loader = UnifiedDataLoader(cache_dir=cache_dir)

    def load_neuromorphic(self, file_path: Union[str, Path], **kwargs: Any) -> NeuromorphicDataset:
        """Load Hugging Face neuromorphic dataset.

        Args:
            file_path: Path to dataset file (.h5 format)
            **kwargs: Additional arguments for loader

        Returns:
            NeuromorphicDataset with event data
        """
        # Load using unified loader
        dataset = load_huggingface_dataset(file_path, **kwargs)

        # Extract event data structure
        # Neuromorphic datasets typically store events as (x, y, t, polarity)
        if dataset.data.ndim == 2 and dataset.data.shape[1] in [3, 4]:
            events = dataset.data
        else:
            # Try to reshape to event format
            events = dataset.data.reshape(-1, 4)

        # Extract metadata
        sensor_type = dataset.metadata.get("sensor_type", "DVS")
        spatial_res = dataset.metadata.get("spatial_resolution", (128, 128))
        temporal_res = dataset.metadata.get("temporal_resolution_us", 1.0)

        return NeuromorphicDataset(
            events=events.astype(np.float64),
            time=dataset.time,
            metadata=dataset.metadata,
            sensor_type=sensor_type,
            spatial_resolution=spatial_res,
            temporal_resolution_us=float(temporal_res),
        )

    def download_tonic_dataset(
        self, dataset_name: str, output_dir: Optional[Path] = None, revision: Optional[str] = None
    ) -> Path:
        """Download Tonic neuromorphic dataset from Hugging Face.

        Args:
            dataset_name: Name of the Tonic dataset (e.g., "tonic/nmnist")
            output_dir: Directory to save downloaded data

        Returns:
            Path to downloaded data directory
        """
        output_dir = output_dir or self.cache_dir / "tonic"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from datasets import load_dataset

            logger.info(f"Downloading Tonic dataset {dataset_name}...")
            dataset = load_dataset(dataset_name, revision=revision)  # pass a commit hash to pin

            # Save to local cache
            dataset.save_to_disk(str(output_dir))
            logger.info(f"Dataset saved to {output_dir}")

        except ImportError:
            logger.warning(
                "Hugging Face datasets library not installed. " "Install with: pip install datasets"
            )
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")

        return output_dir

    def list_available_datasets(self) -> List[str]:
        """List available Hugging Face neuromorphic datasets.

        Returns:
            List of dataset names
        """
        return [
            "tonic/nmnist",
            "tonic/dvsgesture",
            "tonic/cifar10-dvs",
            "tonic/ncaltech101",
        ]
