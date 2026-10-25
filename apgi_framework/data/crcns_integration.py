"""
CRCNS Dataset Integration for APGI Framework

This module provides specialized loaders for CRCNS (Collaborative Research in
Computational Neuroscience) datasets, which are critical for validating the
Landauer Bridge (E_min ≥ kTln2) through empirical metabolic and electrophysiological data.

Primary Datasets:
- V1-1: Visual Cortex Spiking and LFP (Critical Slowing Down validation)
- AC-1: Auditory Cortex Metabolic Proxies (Metabolic Cost C(t) validation)

References:
- CRCNS: https://crcns.org/
- V1-1: https://crcns.org/datasets/v1/v1-1
- AC-1: https://crcns.org/datasets/ac/ac-1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.integrate import trapezoid

from .unified_data_loader import UnifiedDataLoader, load_crcns_dataset

logger = logging.getLogger(__name__)


@dataclass
class CRCNSV1Dataset:
    """
    CRCNS V1-1: Visual Cortex Spiking and LFP dataset.

    This dataset contains multi-unit activity (MUA) and Local Field Potential (LFP)
    recordings from visual cortex, essential for testing Innovation 11 (Three
    Ignition Signatures), specifically the "Critical Slowing Down" signature in
    LFP autocorrelation.

    Attributes:
        spikes: Spike times or spike trains (in ms)
        lfp: Local field potential signal (in μV)
        time: Time vector in milliseconds
        metadata: Dataset metadata
        sampling_rate_hz: Sampling rate
        brain_region: Recorded brain region
        subject_id: Subject identifier
    """

    spikes: npt.NDArray[np.float64]
    lfp: npt.NDArray[np.float64]
    time: npt.NDArray[np.float64]
    metadata: Dict[str, Any] = field(default_factory=dict)
    sampling_rate_hz: float = 1000.0
    brain_region: str = "V1"
    subject_id: str = ""

    def compute_lfp_autocorrelation(self, max_lag_ms: float = 100.0) -> npt.NDArray[np.float64]:
        """
        Compute LFP autocorrelation for Critical Slowing Down detection.

        Critical Slowing Down is characterized by increased autocorrelation
        time constants near phase transitions (ignition events).

        Args:
            max_lag_ms: Maximum lag in milliseconds

        Returns:
            Autocorrelation function
        """
        max_lag_samples = int(max_lag_ms * self.sampling_rate_hz / 1000.0)
        autocorr = np.correlate(self.lfp, self.lfp, mode="full")
        autocorr = autocorr[len(autocorr) // 2 : len(autocorr) // 2 + max_lag_samples]
        return autocorr / autocorr[0]  # Normalize

    def detect_critical_slowing_down(
        self, threshold_std: float = 2.0, window_ms: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Detect Critical Slowing Down signatures in LFP.

        CSD manifests as increased autocorrelation time constants during
        state transitions (e.g., ignition events).

        Args:
            threshold_std: Threshold in standard deviations
            window_ms: Analysis window in milliseconds

        Returns:
            List of detected CSD events with metadata
        """
        autocorr = self.compute_lfp_autocorrelation()
        baseline = np.mean(autocorr[:10])  # Use first 10 lags as baseline
        threshold = baseline + threshold_std * np.std(autocorr[:10])

        # Find periods where autocorrelation exceeds threshold
        above_threshold = autocorr > threshold
        changes = np.diff(above_threshold.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        csd_events = []
        for start, end in zip(starts, ends):
            duration_ms = (end - start) * 1000.0 / self.sampling_rate_hz
            if duration_ms > window_ms:
                csd_events.append(
                    {
                        "start_lag_ms": float(start * 1000.0 / self.sampling_rate_hz),
                        "end_lag_ms": float(end * 1000.0 / self.sampling_rate_hz),
                        "duration_ms": duration_ms,
                        "peak_autocorr": float(np.max(autocorr[start:end])),
                        "mean_autocorr": float(np.mean(autocorr[start:end])),
                    }
                )

        return csd_events

    def compute_firing_rate(self, window_ms: float = 100.0) -> npt.NDArray[np.float64]:
        """
        Compute instantaneous firing rate from spike data.

        Args:
            window_ms: Window size in milliseconds

        Returns:
            Firing rate in spikes per second
        """
        window_samples = int(window_ms * self.sampling_rate_hz / 1000.0)
        n_windows = len(self.time) // window_samples
        firing_rates = np.zeros(n_windows)

        for i in range(n_windows):
            start_idx = i * window_samples
            end_idx = (i + 1) * window_samples
            window_spikes = np.sum(
                (self.spikes >= self.time[start_idx]) & (self.spikes < self.time[end_idx])
            )
            firing_rates[i] = window_spikes / (window_ms / 1000.0)

        return firing_rates


@dataclass
class CRCNSAC1Dataset:
    """
    CRCNS AC-1: Auditory Cortex Metabolic Proxies dataset.

    This dataset contains intrinsic optical imaging (IOS) and flavoprotein
    fluorescence data, serving as empirical proxies for the Metabolic Cost
    C(t) equation [§4.2], allowing the framework to move beyond Arbitrary
    Units (AU) toward actual energetic flux.

    Attributes:
        fluorescence: Fluorescence signal (ΔF/F0 or absolute)
        time: Time vector in milliseconds
        stimulation_times: Times of auditory stimulation
        metadata: Dataset metadata
        sampling_rate_hz: Sampling rate
        sensor_type: Type of metabolic sensor (IOS, flavoprotein, etc.)
    """

    fluorescence: npt.NDArray[np.float64]
    time: npt.NDArray[np.float64]
    stimulation_times: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    metadata: Dict[str, Any] = field(default_factory=dict)
    sampling_rate_hz: float = 10.0
    sensor_type: Literal["IOS", "flavoprotein", "unknown"] = "unknown"

    def compute_metabolic_cost_proxy(
        self, baseline_window_ms: float = 500.0
    ) -> npt.NDArray[np.float64]:
        """
        Compute metabolic cost proxy from fluorescence signal.

        The integrated fluorescence signal serves as a proxy for ATP consumption,
        enabling calibration of the c_1 coefficient in the Landauer Bridge.

        Args:
            baseline_window_ms: Window for baseline normalization (ms)

        Returns:
            Metabolic cost proxy (integrated signal above baseline)
        """
        # Compute baseline from pre-stimulation period
        if len(self.stimulation_times) > 0 and self.stimulation_times[0] > baseline_window_ms:
            baseline_mask = self.time < self.stimulation_times[0]
        else:
            baseline_mask = self.time < baseline_window_ms

        if np.any(baseline_mask):
            baseline = np.mean(self.fluorescence[baseline_mask])
        else:
            baseline = np.mean(self.fluorescence)

        # Compute signal above baseline
        signal_above_baseline = np.maximum(self.fluorescence - baseline, 0)

        # Integrate to get metabolic cost proxy
        dt_ms = np.mean(np.diff(self.time)) if len(self.time) > 1 else 1.0
        metabolic_proxy = np.cumsum(signal_above_baseline) * dt_ms

        return metabolic_proxy

    def detect_metabolic_bursts(
        self, threshold_std: float = 2.0, min_duration_ms: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Detect metabolic bursts (periods of elevated ATP consumption).

        Args:
            threshold_std: Threshold in standard deviations above baseline
            min_duration_ms: Minimum burst duration in milliseconds

        Returns:
            List of detected metabolic bursts
        """
        metabolic_proxy = self.compute_metabolic_cost_proxy()
        baseline = np.mean(metabolic_proxy[:10])
        threshold = baseline + threshold_std * np.std(metabolic_proxy[:10])

        above_threshold = metabolic_proxy > threshold
        changes = np.diff(above_threshold.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        bursts = []
        for start, end in zip(starts, ends):
            duration_ms = (end - start) * 1000.0 / self.sampling_rate_hz
            if duration_ms > min_duration_ms:
                bursts.append(
                    {
                        "start_ms": float(self.time[start]),
                        "end_ms": float(self.time[end]),
                        "duration_ms": duration_ms,
                        "peak_proxy": float(np.max(metabolic_proxy[start:end])),
                        "integrated_proxy": float(
                            trapezoid(metabolic_proxy[start:end], self.time[start:end])
                        ),
                    }
                )

        return bursts


class CRCNSLoader:
    """
    Specialized loader for CRCNS datasets.

    Provides convenience methods for loading specific CRCNS datasets with
    appropriate preprocessing for APGI validation protocols.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize CRCNS loader.

        Args:
            cache_dir: Directory for caching downloaded datasets
        """
        self.cache_dir = cache_dir or Path.home() / ".apgi" / "datasets" / "crcns"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.unified_loader = UnifiedDataLoader(cache_dir=cache_dir)

    def load_v1_1(self, file_path: Union[str, Path], **kwargs: Any) -> CRCNSV1Dataset:
        """
        Load CRCNS V1-1 dataset (Visual Cortex Spiking and LFP).

        Args:
            file_path: Path to V1-1 dataset file (.mat format)
            **kwargs: Additional arguments for loader

        Returns:
            CRCNSV1Dataset with spikes and LFP data
        """
        # Load using unified loader
        dataset = load_crcns_dataset(file_path, **kwargs)

        # Extract spikes and LFP from data
        # V1-1 typically stores spikes and LFP in separate variables
        # This is a simplified extraction - actual structure may vary

        if dataset.data.ndim == 2:
            # Assume first column is spikes, second is LFP
            spikes = dataset.data[:, 0]
            lfp = dataset.data[:, 1]
        else:
            # Single channel - assume LFP
            spikes = np.array([])
            lfp = dataset.data

        return CRCNSV1Dataset(
            spikes=spikes.astype(np.float64),
            lfp=lfp.astype(np.float64),
            time=dataset.time,
            metadata=dataset.metadata,
            sampling_rate_hz=dataset.sampling_rate_hz,
            brain_region=dataset.metadata.get("brain_region", "V1"),
            subject_id=dataset.metadata.get("subject_id", ""),
        )

    def load_ac_1(self, file_path: Union[str, Path], **kwargs: Any) -> CRCNSAC1Dataset:
        """
        Load CRCNS AC-1 dataset (Auditory Cortex Metabolic Proxies).

        Args:
            file_path: Path to AC-1 dataset file (.mat format)
            **kwargs: Additional arguments for loader

        Returns:
            CRCNSAC1Dataset with fluorescence data
        """
        # Load using unified loader
        dataset = load_crcns_dataset(file_path, **kwargs)

        # Extract stimulation times if available
        stim_times = dataset.metadata.get("stimulation_times", np.array([]))
        if isinstance(stim_times, list):
            stim_times = np.array(stim_times)

        # Determine sensor type from metadata
        sensor_type = dataset.metadata.get("sensor_type", "unknown")
        if sensor_type not in ["IOS", "flavoprotein"]:
            sensor_type = "unknown"

        return CRCNSAC1Dataset(
            fluorescence=dataset.data.astype(np.float64),
            time=dataset.time,
            stimulation_times=stim_times.astype(np.float64),
            metadata=dataset.metadata,
            sampling_rate_hz=dataset.sampling_rate_hz,
            sensor_type=sensor_type,
        )

    def download_v1_1(self, output_dir: Optional[Path] = None) -> Path:
        """
        Download CRCNS V1-1 dataset from CRCNS repository.

        Note: This requires CRCNS account authentication.
        Users should download manually from https://crcns.org/datasets/v1/v1-1

        Args:
            output_dir: Directory to save downloaded data

        Returns:
            Path to downloaded data directory
        """
        output_dir = output_dir or self.cache_dir / "v1-1"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.warning(
            "CRCNS datasets require manual download with authentication. "
            "Please download from https://crcns.org/datasets/v1/v1-1 "
            f"and place in {output_dir}"
        )

        return output_dir

    def download_ac_1(self, output_dir: Optional[Path] = None) -> Path:
        """
        Download CRCNS AC-1 dataset from CRCNS repository.

        Note: This requires CRCNS account authentication.
        Users should download manually from https://crcns.org/datasets/ac/ac-1

        Args:
            output_dir: Directory to save downloaded data

        Returns:
            Path to downloaded data directory
        """
        output_dir = output_dir or self.cache_dir / "ac-1"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.warning(
            "CRCNS datasets require manual download with authentication. "
            "Please download from https://crcns.org/datasets/ac/ac-1 "
            f"and place in {output_dir}"
        )

        return output_dir
