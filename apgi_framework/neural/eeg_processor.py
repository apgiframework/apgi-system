"""EEG signal processing pipeline for parameter estimation.

Provides real-time filtering, artifact detection/rejection, and ERP extraction
for P3b and HEP components in APGI experiments.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.stats import zscore

from apgi_framework.utils.progress_monitor import ProgressMonitor


class FilterType(Enum):
    """Filter types for EEG processing."""

    BANDPASS = "bandpass"
    HIGHPASS = "highpass"
    LOWPASS = "lowpass"
    NOTCH = "notch"


@dataclass
class ProcessedEEG:
    """Processed EEG data container."""

    data: np.ndarray  # Filtered/processed EEG data (channels x samples)
    timestamps: np.ndarray
    sampling_rate: float
    channels: List[str]
    artifact_mask: Optional[np.ndarray] = None  # Boolean mask of artifacts
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ERPFeatures:
    """Event-related potential features."""

    p3b_amplitude: Optional[float] = None  # P3b amplitude at Pz (250-500ms)
    p3b_latency: Optional[float] = None  # P3b peak latency
    hep_amplitude: Optional[float] = None  # HEP amplitude (250-400ms post R-peak)
    hep_latency: Optional[float] = None  # HEP peak latency
    baseline_mean: Optional[float] = None
    baseline_std: Optional[float] = None
    snr: Optional[float] = None  # Signal-to-noise ratio


class EEGProcessor:
    """Real-time EEG signal processor with filtering and preprocessing.

    Implements bandpass filtering (0.1-30 Hz), artifact detection,
    and real-time signal conditioning for ERP analysis.
    """

    def __init__(
        self,
        sampling_rate: float = 1000.0,
        highpass: float = 0.1,
        lowpass: float = 30.0,
        notch_freq: Optional[float] = 60.0,
    ):
        """Initialize EEG processor.

        Args:
            sampling_rate: EEG sampling rate in Hz
            highpass: Highpass filter cutoff in Hz
            lowpass: Lowpass filter cutoff in Hz
            notch_freq: Notch filter frequency for power line noise (None to disable)
        """
        self.sampling_rate = sampling_rate
        self.highpass = highpass
        self.lowpass = lowpass
        self.notch_freq = notch_freq

        # Design filters
        self._design_filters()

        # Filter state for real-time processing
        self.filter_state: Optional[Any] = None

    def _design_filters(self) -> None:
        """Design Butterworth filters for EEG processing."""
        nyquist = self.sampling_rate / 2.0

        # Bandpass filter (0.1-30 Hz)
        self.bp_b, self.bp_a = signal.butter(
            4, [self.highpass / nyquist, self.lowpass / nyquist], btype="bandpass"
        )

        # Notch filter for power line noise
        if self.notch_freq:
            # Design notch filter with Q=30 (narrow notch)
            self.notch_b, self.notch_a = signal.iirnotch(
                self.notch_freq, Q=30.0, fs=self.sampling_rate
            )

    def apply_bandpass_filter(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """Apply bandpass filter to EEG data.

        Args:
            data: EEG data array (channels x samples or samples)
            axis: Axis along which to filter

        Returns:
            Filtered data
        """
        # Apply bandpass filter
        filtered = signal.filtfilt(self.bp_b, self.bp_a, data, axis=axis)

        return np.asarray(filtered)

    def apply_notch_filter(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """Apply notch filter to remove power line noise.

        Args:
            data: EEG data array
            axis: Axis along which to filter

        Returns:
            Filtered data
        """
        if not self.notch_freq:
            return data

        filtered = signal.filtfilt(self.notch_b, self.notch_a, data, axis=axis)

        return np.asarray(filtered)

    def process_realtime(
        self, raw_data: np.ndarray, timestamps: np.ndarray, channels: List[str]
    ) -> ProcessedEEG:
        """Process EEG data in real-time with filtering.

        Args:
            raw_data: Raw EEG data (channels x samples)
            timestamps: Timestamp array
            channels: Channel names

        Returns:
            ProcessedEEG object with filtered data
        """
        # Apply filters
        filtered = self.apply_bandpass_filter(raw_data, axis=1)
        filtered = self.apply_notch_filter(filtered, axis=1)

        return ProcessedEEG(
            data=filtered,
            timestamps=timestamps,
            sampling_rate=self.sampling_rate,
            channels=channels,
            metadata={"processing_time": time.time()},
        )

    def apply_reference(
        self,
        data: np.ndarray,
        reference_type: str = "average",
        reference_channels: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Apply reference scheme to EEG data.

        Args:
            data: EEG data (channels x samples)
            reference_type: Type of reference (average, mastoid, custom)
            reference_channels: Indices of reference channels for custom reference

        Returns:
            Referenced data
        """
        if reference_type == "average":
            # Average reference
            reference = np.mean(data, axis=0, keepdims=True)
            return np.asarray(data - reference)

        elif reference_type == "custom" and reference_channels:
            # Custom reference from specified channels
            reference = np.mean(data[reference_channels, :], axis=0, keepdims=True)
            return np.asarray(data - reference)

        else:
            return data


class FASTERArtifactDetector:
    """FASTER (Fully Automated Statistical Thresholding for EEG artifact Rejection).

    Implements automated artifact detection using statistical thresholds
    across multiple dimensions: channels, epochs, and time points.
    """

    def __init__(self, z_threshold: float = 3.0):
        """Initialize FASTER artifact detector.

        Args:
            z_threshold: Z-score threshold for artifact detection
        """
        self.z_threshold = z_threshold
        self.artifact_stats: Dict[str, Any] = {}

    def detect_bad_channels(self, data: np.ndarray) -> np.ndarray:
        """Detect bad channels based on statistical properties.

        Args:
            data: EEG data (channels x samples)

        Returns:
            Boolean array indicating bad channels
        """
        n_channels = data.shape[0]

        # Compute channel statistics
        variance = np.var(data, axis=1)
        mean_correlation = np.zeros(n_channels)

        # Compute mean correlation with other channels
        progress = ProgressMonitor(n_channels, "Computing channel correlations")
        progress.start()

        for i in range(n_channels):
            correlations = []
            for j in range(n_channels):
                if i != j:
                    corr = np.corrcoef(data[i, :], data[j, :])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            mean_correlation[i] = np.mean(correlations) if correlations else 0
            progress.update(message=f"Channel {i + 1}/{n_channels}")

        progress.complete()

        # Detect outliers using z-scores
        variance_z = np.abs(zscore(variance))
        correlation_z = np.abs(zscore(mean_correlation))

        bad_channels = (variance_z > self.z_threshold) | (correlation_z > self.z_threshold)

        self.artifact_stats["bad_channels"] = {
            "count": np.sum(bad_channels),
            "indices": np.where(bad_channels)[0].tolist(),
        }

        return np.asarray(bad_channels)

    def detect_bad_epochs(self, epochs: np.ndarray) -> np.ndarray:
        """Detect bad epochs based on amplitude and variance.

        Args:
            epochs: Epoched data (epochs x channels x samples)

        Returns:
            Boolean array indicating bad epochs
        """
        n_epochs = epochs.shape[0]

        # Compute epoch statistics
        epoch_variance = np.var(epochs, axis=(1, 2))
        epoch_range = np.ptp(epochs, axis=(1, 2))
        epoch_mean_amplitude = np.mean(np.abs(epochs), axis=(1, 2))

        # Detect outliers
        variance_z = np.abs(zscore(epoch_variance))
        range_z = np.abs(zscore(epoch_range))
        amplitude_z = np.abs(zscore(epoch_mean_amplitude))

        bad_epochs = (
            (variance_z > self.z_threshold)
            | (range_z > self.z_threshold)
            | (amplitude_z > self.z_threshold)
        )

        self.artifact_stats["bad_epochs"] = {
            "count": np.sum(bad_epochs),
            "indices": np.where(bad_epochs)[0].tolist(),
            "rejection_rate": np.sum(bad_epochs) / n_epochs,
        }

        return np.asarray(bad_epochs)

    def detect_bad_samples(self, data: np.ndarray, window_size: int = 100) -> np.ndarray:
        """Detect bad samples using sliding window statistics.

        Args:
            data: EEG data (channels x samples)
            window_size: Window size for local statistics

        Returns:
            Boolean array indicating bad samples (channels x samples)
        """
        n_channels, n_samples = data.shape
        bad_samples = np.zeros((n_channels, n_samples), dtype=bool)

        # Compute local statistics in sliding windows
        for i in range(0, n_samples - window_size, window_size // 2):
            window = data[:, i : i + window_size]

            # Compute window statistics
            window_variance = np.var(window, axis=1)
            window_amplitude = np.max(np.abs(window), axis=1)

            # Detect outliers
            variance_z = np.abs(zscore(window_variance))
            amplitude_z = np.abs(zscore(window_amplitude))

            bad_window = (variance_z > self.z_threshold) | (amplitude_z > self.z_threshold)

            # Mark samples in bad windows
            bad_samples[:, i : i + window_size] |= bad_window[:, np.newaxis]

        return bad_samples

    def clean(
        self,
        data: np.ndarray,
        detect_channels: bool = True,
        detect_samples: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Clean EEG data by detecting and handling artifacts.

        Args:
            data: EEG data (channels x samples)
            detect_channels: Whether to detect bad channels
            detect_samples: Whether to detect bad samples

        Returns:
            Tuple of (cleaned data, artifact report)
        """
        cleaned = data.copy()
        artifact_report = {}

        # Detect bad channels
        if detect_channels:
            bad_channels = self.detect_bad_channels(data)
            artifact_report["bad_channels"] = self.artifact_stats["bad_channels"]

            # Interpolate bad channels (simple average of neighbors)
            for ch_idx in np.where(bad_channels)[0]:
                # Use average of all good channels
                good_channels = ~bad_channels
                if np.any(good_channels):
                    cleaned[ch_idx, :] = np.mean(cleaned[good_channels, :], axis=0)

        # Detect bad samples
        if detect_samples:
            bad_samples = self.detect_bad_samples(cleaned)
            artifact_report["bad_samples"] = {
                "total_count": np.sum(bad_samples),
                "percentage": np.sum(bad_samples) / bad_samples.size * 100,
            }

            # Interpolate bad samples (linear interpolation)
            for ch_idx in range(cleaned.shape[0]):
                bad_mask = bad_samples[ch_idx, :]
                if np.any(bad_mask):
                    good_indices = np.where(~bad_mask)[0]
                    bad_indices = np.where(bad_mask)[0]

                    if len(good_indices) >= 2:
                        cleaned[ch_idx, bad_indices] = np.interp(
                            bad_indices, good_indices, cleaned[ch_idx, good_indices]
                        )

        return cleaned, artifact_report


class EpochExtractor:
    """Extract epochs (time-locked segments) from continuous EEG data.

    Supports event-locked epoching for ERP analysis with baseline correction.
    """

    def __init__(self, sampling_rate: float = 1000.0):
        """Initialize epoch extractor.

        Args:
            sampling_rate: EEG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate

    def extract_epochs(
        self,
        data: np.ndarray,
        event_times: np.ndarray,
        timestamps: np.ndarray,
        tmin: float = -0.2,
        tmax: float = 0.8,
    ) -> np.ndarray:
        """Extract epochs around events.

        Args:
            data: Continuous EEG data (channels x samples)
            event_times: Event timestamps
            timestamps: Data timestamps
            tmin: Start time relative to event (seconds)
            tmax: End time relative to event (seconds)

        Returns:
            Epoched data (epochs x channels x samples)
        """
        n_channels = data.shape[0]

        # Convert time to samples
        samples_before = int(abs(tmin) * self.sampling_rate)
        samples_after = int(tmax * self.sampling_rate)
        epoch_length = samples_before + samples_after

        epochs = []

        for event_time in event_times:
            # Find closest timestamp
            time_diffs = np.abs(timestamps - event_time)
            event_idx = np.argmin(time_diffs)

            # Extract epoch
            start_idx = event_idx - samples_before
            end_idx = event_idx + samples_after

            # Check bounds
            if start_idx >= 0 and end_idx <= data.shape[1]:
                epoch = data[:, start_idx:end_idx]

                # Ensure correct length
                if epoch.shape[1] == epoch_length:
                    epochs.append(epoch)

        if not epochs:
            return np.array([]).reshape(0, n_channels, epoch_length)

        return np.array(epochs)


class BaselineCorrector:
    """Baseline correction for ERP analysis.

    Removes pre-stimulus baseline activity to isolate event-related responses.
    """

    def __init__(self, sampling_rate: float = 1000.0):
        """Initialize baseline corrector.

        Args:
            sampling_rate: EEG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate

    def apply_baseline_correction(
        self,
        epochs: np.ndarray,
        baseline_window: Tuple[float, float] = (-0.2, 0.0),
        method: str = "mean",
    ) -> np.ndarray:
        """Apply baseline correction to epochs.

        Args:
            epochs: Epoched data (epochs x channels x samples)
            baseline_window: Baseline time window (start, end) in seconds relative to event
            method: Baseline method (mean, median)

        Returns:
            Baseline-corrected epochs
        """
        if epochs.size == 0:
            return epochs

        # Assume epoch starts at tmin (e.g., -0.2s)
        # Convert baseline window to sample indices
        # This assumes the epoch structure from extract_epochs

        # For simplicity, assume baseline is the first portion of the epoch
        baseline_start_sample = 0
        baseline_end_sample = int(abs(baseline_window[0]) * self.sampling_rate)

        # Extract baseline period
        baseline = epochs[:, :, baseline_start_sample:baseline_end_sample]

        # Compute baseline value
        if method == "mean":
            baseline_value = np.mean(baseline, axis=2, keepdims=True)
        elif method == "median":
            baseline_value = np.median(baseline, axis=2, keepdims=True)
        else:
            raise ValueError(f"Unknown baseline method: {method}")

        # Subtract baseline
        corrected = epochs - baseline_value

        return np.asarray(corrected)


class ERPExtractor:
    """Extract event-related potentials (ERPs) from EEG data.

    Specialized for P3b (250-500ms at Pz) and HEP (250-400ms post R-peak) extraction.
    """

    def __init__(self, sampling_rate: float = 1000.0):
        """Initialize ERP extractor.

        Args:
            sampling_rate: EEG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.epoch_extractor = EpochExtractor(sampling_rate)
        self.baseline_corrector = BaselineCorrector(sampling_rate)

    def extract_p3b(
        self,
        epochs: np.ndarray,
        channel_idx: int,
        time_window: Tuple[float, float] = (0.25, 0.5),
        tmin: float = -0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract P3b amplitude from epochs.

        Args:
            epochs: Epoched data (epochs x channels x samples)
            channel_idx: Index of Pz channel (or target channel)
            time_window: Time window for P3b (start, end) in seconds
            tmin: Start time of epoch relative to event

        Returns:
            Tuple of (P3b amplitudes, P3b latencies) for each epoch
        """
        if epochs.size == 0:
            return np.array([]), np.array([])

        # Convert time window to sample indices
        window_start_sample = int((time_window[0] - tmin) * self.sampling_rate)
        window_end_sample = int((time_window[1] - tmin) * self.sampling_rate)

        # Ensure valid indices
        window_start_sample = max(0, window_start_sample)
        window_end_sample = min(epochs.shape[2], window_end_sample)

        # Extract P3b window
        p3b_window = epochs[:, channel_idx, window_start_sample:window_end_sample]

        # Find peak amplitude and latency in window
        p3b_amplitudes = np.max(p3b_window, axis=1)
        p3b_latencies = np.argmax(p3b_window, axis=1) / self.sampling_rate + time_window[0]

        return p3b_amplitudes, p3b_latencies

    def extract_hep(
        self,
        data: np.ndarray,
        r_peak_times: np.ndarray,
        timestamps: np.ndarray,
        channel_indices: Optional[List[int]] = None,
        time_window: Tuple[float, float] = (0.25, 0.4),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract heartbeat-evoked potential (HEP) around R-peaks.

        Args:
            data: Continuous EEG data (channels x samples)
            r_peak_times: R-peak timestamps
            timestamps: Data timestamps
            channel_indices: Channels to extract HEP from (None for all)
            time_window: Time window post R-peak (start, end) in seconds

        Returns:
            Tuple of (HEP epochs, HEP amplitudes)
        """
        # Extract epochs around R-peaks
        tmin = -0.2  # Pre-R-peak baseline
        tmax = 0.8  # Post-R-peak window

        epochs = self.epoch_extractor.extract_epochs(data, r_peak_times, timestamps, tmin, tmax)

        if epochs.size == 0:
            return np.array([]), np.array([])

        # Apply baseline correction
        epochs = self.baseline_corrector.apply_baseline_correction(
            epochs, baseline_window=(-0.2, 0.0)
        )

        # Select channels
        if channel_indices:
            epochs = epochs[:, channel_indices, :]

        # Convert time window to sample indices
        window_start_sample = int((time_window[0] - tmin) * self.sampling_rate)
        window_end_sample = int((time_window[1] - tmin) * self.sampling_rate)

        # Extract HEP window
        hep_window = epochs[:, :, window_start_sample:window_end_sample]

        # Compute mean amplitude in HEP window
        hep_amplitudes = np.mean(hep_window, axis=2)

        return epochs, hep_amplitudes

    def extract_features(
        self,
        data: np.ndarray,
        event_times: np.ndarray,
        timestamps: np.ndarray,
        pz_channel_idx: int,
        r_peak_times: Optional[np.ndarray] = None,
    ) -> ERPFeatures:
        """Extract comprehensive ERP features.

        Args:
            data: Continuous EEG data (channels x samples)
            event_times: Event timestamps for P3b
            timestamps: Data timestamps
            pz_channel_idx: Index of Pz channel
            r_peak_times: R-peak timestamps for HEP (optional)

        Returns:
            ERPFeatures object with extracted features
        """
        features = ERPFeatures()

        # Extract P3b
        if len(event_times) > 0:
            epochs = self.epoch_extractor.extract_epochs(
                data, event_times, timestamps, tmin=-0.2, tmax=0.8
            )

            if epochs.size > 0:
                epochs = self.baseline_corrector.apply_baseline_correction(epochs)
                p3b_amps, p3b_lats = self.extract_p3b(epochs, pz_channel_idx)

                if len(p3b_amps) > 0:
                    features.p3b_amplitude = float(np.mean(p3b_amps))
                    features.p3b_latency = float(np.mean(p3b_lats))

        # Extract HEP
        if r_peak_times is not None and len(r_peak_times) > 0:
            hep_epochs, hep_amps = self.extract_hep(data, r_peak_times, timestamps)

            if hep_amps.size > 0:
                # Average across channels and epochs
                features.hep_amplitude = float(np.mean(hep_amps))

        return features
