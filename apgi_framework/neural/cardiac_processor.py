"""
Cardiac signal processing pipeline for parameter estimation.

Provides R-peak detection, HRV analysis, HEP extraction, and quality assessment
for interoceptive precision estimation in APGI experiments.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
from scipy.stats import zscore

# Try to import numba for optimization
try:
    from numba import jit, prange  # type: ignore[import-untyped]

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Fallback decorators that do nothing
    def jit(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func

        return decorator

    def prange(x: int) -> range:
        return range(x)


class RPeakAlgorithm(Enum):
    """R-peak detection algorithms."""

    SIMPLE_THRESHOLD = "simple_threshold"
    PAN_TOMPKINS = "pan_tompkins"
    WAVELET = "wavelet"
    ADAPTIVE = "adaptive"


@jit(nopython=True, cache=True)
def _optimized_derivative(signal_data: np.ndarray) -> np.ndarray:
    """Optimized derivative calculation using numba."""
    n = len(signal_data)
    derivative = np.zeros(n)
    for i in range(1, n):
        derivative[i] = signal_data[i] - signal_data[i - 1]
    return derivative


@jit(nopython=True, cache=True)
def _optimized_moving_average(signal_data: np.ndarray, window_size: int) -> np.ndarray:
    """Optimized moving average using cumulative sum."""
    n = len(signal_data)
    result = np.zeros(n)
    cumsum = np.zeros(n + 1)
    cumsum[1:] = np.cumsum(signal_data)

    half_window = window_size // 2
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        result[i] = (cumsum[end] - cumsum[start]) / (end - start)

    return result


@jit(nopython=True, cache=True)
def _find_peaks_vectorized(
    signal_data: np.ndarray, threshold: float, min_distance: int
) -> np.ndarray:
    """Vectorized peak finding using numba."""
    n = len(signal_data)
    peaks = []

    for i in range(min_distance, n - min_distance):
        if signal_data[i] > threshold:
            # Check if it's a local maximum
            is_peak = True
            for j in range(i - min_distance, i + min_distance + 1):
                if j != i and signal_data[j] >= signal_data[i]:
                    is_peak = False
                    break

            if is_peak:
                peaks.append(i)

    return np.array(peaks)


@jit(nopython=True, cache=True)
def _filter_peaks_physiological(peak_times: np.ndarray, min_interval: float) -> np.ndarray:
    """Filter peaks based on physiological constraints."""
    if len(peak_times) <= 1:
        return peak_times

    filtered = [peak_times[0]]
    for i in range(1, len(peak_times)):
        if peak_times[i] - filtered[-1] >= min_interval:
            filtered.append(peak_times[i])

    return np.array(filtered)


@dataclass
class HRVMetrics:
    """Heart rate variability metrics."""

    # Time domain metrics
    mean_hr: float  # Mean heart rate (bpm)
    sdnn: float  # Standard deviation of NN intervals (ms)
    rmssd: float  # Root mean square of successive differences (ms)
    pnn50: float  # Percentage of successive differences > 50ms

    # Frequency domain metrics
    lf_power: Optional[float] = None  # Low frequency power (0.04-0.15 Hz)
    hf_power: Optional[float] = None  # High frequency power (0.15-0.4 Hz)
    lf_hf_ratio: Optional[float] = None  # LF/HF ratio
    total_power: Optional[float] = None  # Total spectral power

    # Additional metrics
    n_beats: int = 0  # Number of beats analyzed
    recording_duration: float = 0.0  # Duration in seconds


@dataclass
class CardiacQualityMetrics:
    """Quality assessment for cardiac signals."""

    overall_quality: float  # Overall quality score (0-1)
    signal_quality_index: float  # SQI based on template matching
    ectopic_beat_percentage: float  # Percentage of ectopic beats
    missed_beat_percentage: float  # Estimated missed beats
    noise_level: float  # Estimated noise level
    recommendation: str  # Quality recommendation


class CardiacProcessor:
    """
    Cardiac signal processor with multiple R-peak detection algorithms.

    Integrates existing R-peak detection with additional algorithms
    for robust heartbeat detection in various signal conditions.
    """

    def __init__(
        self,
        sampling_rate: float = 1000.0,
        algorithm: RPeakAlgorithm = RPeakAlgorithm.ADAPTIVE,
        heart_rate_monitor: Optional[Any] = None,
    ):
        """
        Initialize cardiac processor.

        Args:
            sampling_rate: ECG/PPG sampling rate in Hz
            algorithm: R-peak detection algorithm to use
            heart_rate_monitor: HeartRateMonitor instance (from physiological_monitoring)
        """
        self.sampling_rate = sampling_rate
        self.algorithm = algorithm
        self.heart_rate_monitor = heart_rate_monitor

    def preprocess_ecg(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Preprocess ECG signal with filtering.

        Args:
            ecg_signal: Raw ECG signal

        Returns:
            Preprocessed ECG signal
        """
        # Bandpass filter (0.5-40 Hz for ECG)
        nyquist = self.sampling_rate / 2.0
        b, a = signal.butter(4, [0.5 / nyquist, 40.0 / nyquist], btype="bandpass")
        filtered = signal.filtfilt(b, a, ecg_signal)

        # Remove baseline wander with high-pass filter
        b_hp, a_hp = signal.butter(2, 0.5 / nyquist, btype="highpass")
        filtered = signal.filtfilt(b_hp, a_hp, filtered)

        return np.asarray(filtered)

    def detect_r_peaks_pan_tompkins(
        self, ecg_signal: np.ndarray, timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Detect R-peaks using optimized Pan-Tompkins algorithm.

        Args:
            ecg_signal: Preprocessed ECG signal
            timestamps: Timestamp array

        Returns:
            Array of R-peak timestamps
        """
        # Use optimized derivative calculation
        if NUMBA_AVAILABLE:
            derivative = _optimized_derivative(ecg_signal)
        else:
            # Fallback to numpy
            derivative = np.diff(ecg_signal)
            derivative = np.pad(derivative, (0, 1), mode="edge")

        # Squaring (emphasizes higher frequencies) - vectorized
        squared = derivative**2

        # Optimized moving window integration
        window_size = int(0.15 * self.sampling_rate)  # 150ms window

        if NUMBA_AVAILABLE and len(squared) < 100000:  # Only use numba for smaller arrays
            integrated = _optimized_moving_average(squared, window_size)
        else:
            # Use numpy convolution for large arrays
            kernel = np.ones(window_size) / window_size
            integrated = np.convolve(squared, kernel, mode="same")

        # Vectorized adaptive thresholding
        signal_mean = np.mean(integrated)
        signal_std = np.std(integrated)
        threshold = signal_mean + 0.5 * signal_std

        # Use optimized peak finding
        min_distance = int(0.3 * self.sampling_rate)  # Min 300ms between peaks

        if NUMBA_AVAILABLE and len(integrated) < 100000:
            peak_indices = _find_peaks_vectorized(integrated, threshold, min_distance)
        else:
            # Fallback to scipy
            peak_indices, _ = signal.find_peaks(integrated, height=threshold, distance=min_distance)

        # Convert to timestamps
        if len(peak_indices) > 0:
            peaks = timestamps[peak_indices]
        else:
            peaks = np.array([])

        # Apply physiological constraints using optimized function
        min_rr_interval = 60.0 / 200.0  # 200 bpm max

        if NUMBA_AVAILABLE and len(peaks) > 0:
            peaks = _filter_peaks_physiological(peaks, min_rr_interval)
        else:
            # Fallback for empty or large arrays
            if len(peaks) > 1:
                filtered_peaks = [peaks[0]]
                for peak in peaks[1:]:
                    if peak - filtered_peaks[-1] >= min_rr_interval:
                        filtered_peaks.append(peak)
                peaks = np.array(filtered_peaks)

        return np.asarray(peaks)

    def detect_r_peaks_adaptive(self, ecg_signal: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks using adaptive threshold method.

        Args:
            ecg_signal: Preprocessed ECG signal
            timestamps: Timestamp array

        Returns:
            Array of R-peak timestamps
        """
        # Compute adaptive threshold using moving statistics
        window_size = int(0.5 * self.sampling_rate)  # 500ms window

        peaks = []
        threshold_history = []

        for i in range(window_size, len(ecg_signal) - window_size, window_size // 4):
            window = ecg_signal[i - window_size : i + window_size]

            # Adaptive threshold: mean + k * std
            threshold = np.mean(window) + 0.6 * np.std(window)
            threshold_history.append(threshold)

            # Find peaks in window
            window_peaks = signal.find_peaks(
                ecg_signal[i : i + window_size],
                height=threshold,
                distance=int(0.3 * self.sampling_rate),  # Min 300ms between peaks
            )[0]

            # Convert to timestamps
            for peak_idx in window_peaks:
                peaks.append(timestamps[i + peak_idx])

        return np.array(peaks)

    def detect_r_peaks(self, ecg_signal: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks using configured algorithm.

        Args:
            ecg_signal: ECG signal (raw or preprocessed)
            timestamps: Timestamp array

        Returns:
            Array of R-peak timestamps
        """
        # Preprocess signal
        preprocessed = self.preprocess_ecg(ecg_signal)

        # Use configured algorithm
        if self.algorithm == RPeakAlgorithm.PAN_TOMPKINS:
            r_peaks = self.detect_r_peaks_pan_tompkins(preprocessed, timestamps)

        elif self.algorithm == RPeakAlgorithm.ADAPTIVE:
            r_peaks = self.detect_r_peaks_adaptive(preprocessed, timestamps)

        elif self.algorithm == RPeakAlgorithm.SIMPLE_THRESHOLD:
            # Use existing heart rate monitor if available
            if self.heart_rate_monitor:
                r_peaks = self.heart_rate_monitor.detect_r_peaks(preprocessed, timestamps)
            else:
                # Fallback to adaptive
                r_peaks = self.detect_r_peaks_adaptive(preprocessed, timestamps)

        else:
            # Default to adaptive
            r_peaks = self.detect_r_peaks_adaptive(preprocessed, timestamps)

        return r_peaks

    def correct_ectopic_beats(
        self, rr_intervals: np.ndarray, threshold: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and correct ectopic beats (abnormal heartbeats).

        Args:
            rr_intervals: RR interval array (ms)
            threshold: Threshold for detecting ectopic beats (relative change)

        Returns:
            Tuple of (corrected RR intervals, ectopic beat mask)
        """
        if len(rr_intervals) < 3:
            return rr_intervals, np.zeros(len(rr_intervals), dtype=bool)

        # Compute successive differences
        successive_diffs = np.abs(np.diff(rr_intervals))
        relative_diffs = successive_diffs / rr_intervals[:-1]

        # Detect ectopic beats (large relative changes)
        ectopic = np.zeros(len(rr_intervals), dtype=bool)
        ectopic[1:] = relative_diffs > threshold

        # Correct ectopic beats by interpolation
        corrected = rr_intervals.copy()
        for i in np.where(ectopic)[0]:
            if i > 0 and i < len(corrected) - 1:
                # Interpolate from neighbors
                corrected[i] = (corrected[i - 1] + corrected[i + 1]) / 2

        return corrected, ectopic


class HRVAnalyzer:
    """
    Heart rate variability analyzer with time and frequency domain metrics.

    Extends existing HRV metrics with frequency domain analysis
    for comprehensive autonomic assessment.
    """

    def __init__(self, sampling_rate: float = 1000.0):
        """
        Initialize HRV analyzer.

        Args:
            sampling_rate: Sampling rate for interpolation
        """
        self.sampling_rate = sampling_rate

    def compute_time_domain_metrics(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Compute time domain HRV metrics.

        Args:
            rr_intervals: RR interval array (ms)

        Returns:
            Dictionary with time domain metrics
        """
        if len(rr_intervals) < 2:
            return {"mean_hr": 0.0, "sdnn": 0.0, "rmssd": 0.0, "pnn50": 0.0}

        # Mean heart rate
        mean_rr = np.mean(rr_intervals)
        mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0

        # SDNN: Standard deviation of NN intervals
        sdnn = np.std(rr_intervals)

        # RMSSD: Root mean square of successive differences
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs**2))

        # pNN50: Percentage of successive differences > 50ms
        pnn50 = np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100

        return {
            "mean_hr": float(mean_hr),
            "sdnn": float(sdnn),
            "rmssd": float(rmssd),
            "pnn50": float(pnn50),
        }

    def compute_frequency_domain_metrics(
        self, rr_intervals: np.ndarray, r_peak_times: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute frequency domain HRV metrics using Welch's method.

        Args:
            rr_intervals: RR interval array (ms)
            r_peak_times: R-peak timestamps (seconds)

        Returns:
            Dictionary with frequency domain metrics
        """
        if len(rr_intervals) < 10:
            return {
                "lf_power": 0.0,
                "hf_power": 0.0,
                "lf_hf_ratio": 0.0,
                "total_power": 0.0,
            }

        # Interpolate RR intervals to uniform sampling
        # Create uniform time base
        duration = r_peak_times[-1] - r_peak_times[0]
        uniform_time = np.arange(0, duration, 1.0 / 4.0)  # 4 Hz sampling

        # Interpolate RR intervals
        interp_rr = np.interp(uniform_time, r_peak_times[1:], rr_intervals)

        # Compute power spectral density using Welch's method
        freqs, psd = signal.welch(
            interp_rr,
            fs=4.0,  # 4 Hz sampling
            nperseg=min(256, len(interp_rr)),
            scaling="density",
        )

        # Define frequency bands
        lf_band = (0.04, 0.15)  # Low frequency
        hf_band = (0.15, 0.4)  # High frequency

        # Compute band powers
        lf_indices = np.where((freqs >= lf_band[0]) & (freqs < lf_band[1]))[0]
        hf_indices = np.where((freqs >= hf_band[0]) & (freqs < hf_band[1]))[0]

        lf_power = trapezoid(psd[lf_indices], freqs[lf_indices]) if len(lf_indices) > 0 else 0.0
        hf_power = trapezoid(psd[hf_indices], freqs[hf_indices]) if len(hf_indices) > 0 else 0.0
        total_power = trapezoid(psd, freqs)

        # LF/HF ratio
        lf_hf_ratio = float(lf_power) / float(hf_power) if float(hf_power) > 0 else 0.0

        return {
            "lf_power": float(lf_power),
            "hf_power": float(hf_power),
            "lf_hf_ratio": float(lf_hf_ratio),
            "total_power": float(total_power),
        }

    def compute_comprehensive_hrv(
        self, rr_intervals: np.ndarray, r_peak_times: np.ndarray
    ) -> HRVMetrics:
        """
        Compute comprehensive HRV metrics (time and frequency domain).

        Args:
            rr_intervals: RR interval array (ms)
            r_peak_times: R-peak timestamps (seconds)

        Returns:
            HRVMetrics object with all metrics
        """
        # Time domain metrics
        time_metrics = self.compute_time_domain_metrics(rr_intervals)

        # Frequency domain metrics
        freq_metrics = self.compute_frequency_domain_metrics(rr_intervals, r_peak_times)

        # Recording duration
        duration = r_peak_times[-1] - r_peak_times[0] if len(r_peak_times) > 1 else 0.0

        return HRVMetrics(
            mean_hr=time_metrics["mean_hr"],
            sdnn=time_metrics["sdnn"],
            rmssd=time_metrics["rmssd"],
            pnn50=time_metrics["pnn50"],
            lf_power=freq_metrics["lf_power"],
            hf_power=freq_metrics["hf_power"],
            lf_hf_ratio=freq_metrics["lf_hf_ratio"],
            total_power=freq_metrics["total_power"],
            n_beats=len(rr_intervals) + 1,
            recording_duration=duration,
        )


class HEPExtractor:
    """
    Heartbeat-evoked potential (HEP) extractor.

    Extracts EEG epochs time-locked to R-peaks for interoceptive
    neural processing analysis.
    """

    def __init__(self, sampling_rate: float = 1000.0):
        """
        Initialize HEP extractor.

        Args:
            sampling_rate: EEG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate

    def extract_hep_epochs(
        self,
        eeg_data: np.ndarray,
        r_peak_times: np.ndarray,
        eeg_timestamps: np.ndarray,
        tmin: float = -0.2,
        tmax: float = 0.8,
    ) -> np.ndarray:
        """
        Extract HEP epochs around R-peaks.

        Args:
            eeg_data: EEG data (channels x samples)
            r_peak_times: R-peak timestamps
            eeg_timestamps: EEG timestamps
            tmin: Start time relative to R-peak (seconds)
            tmax: End time relative to R-peak (seconds)

        Returns:
            HEP epochs (epochs x channels x samples)
        """
        n_channels = eeg_data.shape[0]

        # Convert time to samples
        samples_before = int(abs(tmin) * self.sampling_rate)
        samples_after = int(tmax * self.sampling_rate)
        epoch_length = samples_before + samples_after

        epochs = []

        for r_peak_time in r_peak_times:
            # Find closest EEG timestamp
            time_diffs = np.abs(eeg_timestamps - r_peak_time)
            r_peak_idx = np.argmin(time_diffs)

            # Extract epoch
            start_idx = r_peak_idx - samples_before
            end_idx = r_peak_idx + samples_after

            # Check bounds
            if start_idx >= 0 and end_idx <= eeg_data.shape[1]:
                epoch = eeg_data[:, start_idx:end_idx]

                # Ensure correct length
                if epoch.shape[1] == epoch_length:
                    epochs.append(epoch)

        if not epochs:
            return np.array([]).reshape(0, n_channels, epoch_length)

        return np.array(epochs)

    def compute_hep_amplitude(
        self,
        hep_epochs: np.ndarray,
        time_window: Tuple[float, float] = (0.25, 0.4),
        tmin: float = -0.2,
    ) -> np.ndarray:
        """
        Compute HEP amplitude in specified time window.

        Args:
            hep_epochs: HEP epochs (epochs x channels x samples)
            time_window: Time window for HEP (start, end) in seconds post R-peak
            tmin: Start time of epoch relative to R-peak

        Returns:
            HEP amplitudes (epochs x channels)
        """
        if hep_epochs.size == 0:
            return np.array([])

        # Convert time window to sample indices
        window_start_sample = int((time_window[0] - tmin) * self.sampling_rate)
        window_end_sample = int((time_window[1] - tmin) * self.sampling_rate)

        # Ensure valid indices
        window_start_sample = max(0, window_start_sample)
        window_end_sample = min(hep_epochs.shape[2], window_end_sample)

        # Extract HEP window and compute mean amplitude
        hep_window = hep_epochs[:, :, window_start_sample:window_end_sample]
        hep_amplitudes = np.mean(hep_window, axis=2)

        return np.asarray(hep_amplitudes)

    def apply_baseline_correction(
        self,
        hep_epochs: np.ndarray,
        baseline_window: Tuple[float, float] = (-0.2, 0.0),
        tmin: float = -0.2,
    ) -> np.ndarray:
        """
        Apply baseline correction to HEP epochs.

        Args:
            hep_epochs: HEP epochs (epochs x channels x samples)
            baseline_window: Baseline time window (start, end) in seconds
            tmin: Start time of epoch relative to R-peak

        Returns:
            Baseline-corrected HEP epochs
        """
        if hep_epochs.size == 0:
            return hep_epochs

        # Convert baseline window to sample indices
        baseline_start_sample = int((baseline_window[0] - tmin) * self.sampling_rate)
        baseline_end_sample = int((baseline_window[1] - tmin) * self.sampling_rate)

        # Ensure valid indices
        baseline_start_sample = max(0, baseline_start_sample)
        baseline_end_sample = min(hep_epochs.shape[2], baseline_end_sample)

        # Compute baseline
        baseline = np.mean(
            hep_epochs[:, :, baseline_start_sample:baseline_end_sample],
            axis=2,
            keepdims=True,
        )

        # Subtract baseline
        corrected = hep_epochs - baseline

        return np.asarray(corrected)


class CardiacQualityAssessor:
    """
    Cardiac signal quality assessment.

    Evaluates signal quality, detects artifacts, and validates beat detection
    for reliable HRV and HEP analysis.
    """

    def __init__(self, sampling_rate: float = 1000.0):
        """
        Initialize cardiac quality assessor.

        Args:
            sampling_rate: Cardiac signal sampling rate in Hz
        """
        self.sampling_rate = sampling_rate

    def preprocess_ecg(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Preprocess ECG signal with filtering.

        Args:
            ecg_signal: Raw ECG signal

        Returns:
            Preprocessed ECG signal
        """
        # Bandpass filter (0.5-40 Hz for ECG)
        nyquist = self.sampling_rate / 2.0
        b, a = signal.butter(4, [0.5 / nyquist, 40.0 / nyquist], btype="bandpass")
        filtered = signal.filtfilt(b, a, ecg_signal)

        # Remove baseline wander with high-pass filter
        b_hp, a_hp = signal.butter(2, 0.5 / nyquist, btype="highpass")
        filtered = signal.filtfilt(b_hp, a_hp, filtered)

        return np.asarray(filtered)

    def detect_r_peaks_pan_tompkins(
        self, ecg_signal: np.ndarray, timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Detect R-peaks using optimized Pan-Tompkins algorithm.

        Args:
            ecg_signal: Preprocessed ECG signal
            timestamps: Timestamp array

        Returns:
            Array of R-peak timestamps
        """
        # Use optimized derivative calculation
        if NUMBA_AVAILABLE:
            derivative = _optimized_derivative(ecg_signal)
        else:
            # Fallback to numpy
            derivative = np.diff(ecg_signal)
            derivative = np.pad(derivative, (0, 1), mode="edge")

        # Squaring (emphasizes higher frequencies) - vectorized
        squared = derivative**2

        # Optimized moving window integration
        window_size = int(0.15 * self.sampling_rate)  # 150ms window

        if NUMBA_AVAILABLE and len(squared) < 100000:  # Only use numba for smaller arrays
            integrated = _optimized_moving_average(squared, window_size)
        else:
            # Use numpy convolution for large arrays
            kernel = np.ones(window_size) / window_size
            integrated = np.convolve(squared, kernel, mode="same")

        # Vectorized adaptive thresholding
        signal_mean = np.mean(integrated)
        signal_std = np.std(integrated)
        threshold = signal_mean + 0.5 * signal_std

        # Use optimized peak finding
        min_distance = int(0.3 * self.sampling_rate)  # Min 300ms between peaks

        if NUMBA_AVAILABLE and len(integrated) < 100000:
            peak_indices = _find_peaks_vectorized(integrated, threshold, min_distance)
        else:
            # Fallback to scipy
            peak_indices, _ = signal.find_peaks(integrated, height=threshold, distance=min_distance)

        # Convert to timestamps
        if len(peak_indices) > 0:
            peaks = timestamps[peak_indices]
        else:
            peaks = np.array([])

        # Apply physiological constraints using optimized function
        min_rr_interval = 60.0 / 200.0  # 200 bpm max

        if NUMBA_AVAILABLE and len(peaks) > 0:
            peaks = _filter_peaks_physiological(peaks, min_rr_interval)
        else:
            # Fallback for empty or large arrays
            if len(peaks) > 1:
                filtered_peaks = [peaks[0]]
                for peak in peaks[1:]:
                    if peak - filtered_peaks[-1] >= min_rr_interval:
                        filtered_peaks.append(peak)
                peaks = np.array(filtered_peaks)

        return np.asarray(peaks)

    def detect_r_peaks_adaptive(self, ecg_signal: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks using adaptive threshold method.

        Args:
            ecg_signal: Preprocessed ECG signal
            timestamps: Timestamp array

        Returns:
            Array of R-peak timestamps
        """
        # Compute adaptive threshold using moving statistics
        window_size = int(0.5 * self.sampling_rate)  # 500ms window

        peaks = []
        threshold_history = []

        for i in range(window_size, len(ecg_signal) - window_size, window_size // 4):
            window = ecg_signal[i - window_size : i + window_size]

            # Adaptive threshold: mean + k * std
            threshold = np.mean(window) + 0.6 * np.std(window)
            threshold_history.append(threshold)

            # Find peaks in window
            window_peaks = signal.find_peaks(
                ecg_signal[i : i + window_size],
                height=threshold,
                distance=int(0.3 * self.sampling_rate),  # Min 300ms between peaks
            )[0]

            # Convert to timestamps
            for peak_idx in window_peaks:
                peaks.append(timestamps[i + peak_idx])

        return np.array(peaks)

    def compute_signal_quality_index(
        self,
        ecg_signal: np.ndarray,
        r_peaks: np.ndarray,
        timestamps: np.ndarray,
    ) -> float:
        """
        Compute signal quality index based on template matching.

        Args:
            ecg_signal: ECG signal
            r_peaks: Detected R-peak timestamps
            timestamps: Signal timestamps

        Returns:
            Signal quality index (0-1)
        """
        if len(r_peaks) < 3:
            return 0.0

        # Extract beat templates
        templates = []
        window_size = int(0.3 * self.sampling_rate)  # 300ms window

        for r_peak in r_peaks:
            # Find R-peak index
            peak_idx = np.argmin(np.abs(timestamps - r_peak))

            # Extract template
            start_idx = max(0, int(peak_idx) - window_size // 2)
            end_idx = min(len(ecg_signal), int(peak_idx) + window_size // 2)

            if end_idx - start_idx == window_size:
                template = ecg_signal[start_idx:end_idx]
                templates.append(template)

        if len(templates) < 3:
            return 0.0

        # Compute mean template
        mean_template = np.mean(templates, axis=0)

        # Compute correlation of each beat with mean template
        correlations = []
        for template in templates:
            corr = np.corrcoef(template, mean_template)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        # SQI is mean correlation
        sqi = np.mean(correlations) if correlations else 0.0

        return float(max(0.0, min(1.0, sqi)))

    def detect_ectopic_beats(self, rr_intervals: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Detect ectopic beats in RR interval series.

        Args:
            rr_intervals: RR interval array (ms)

        Returns:
            Tuple of (ectopic percentage, ectopic mask)
        """
        if len(rr_intervals) < 3:
            return 0.0, np.zeros(len(rr_intervals), dtype=bool)

        # Compute z-scores of RR intervals
        rr_z = np.abs(zscore(rr_intervals))

        # Detect outliers (z-score > 3)
        ectopic = rr_z > 3.0

        ectopic_percentage = np.sum(ectopic) / len(ectopic) * 100

        return float(ectopic_percentage), ectopic

    def estimate_missed_beats(self, rr_intervals: np.ndarray) -> float:
        """
        Estimate percentage of missed beats.

        Args:
            rr_intervals: RR interval array (ms)

        Returns:
            Estimated missed beat percentage
        """
        if len(rr_intervals) < 2:
            return 0.0

        # Expected RR interval (median)
        expected_rr = np.median(rr_intervals)

        # Count intervals that are approximately double the expected
        # (indicating a missed beat)
        missed = np.sum(rr_intervals > 1.5 * expected_rr)

        missed_percentage = missed / len(rr_intervals) * 100

        return float(missed_percentage)

    def assess_quality(
        self,
        ecg_signal: np.ndarray,
        r_peaks: np.ndarray,
        rr_intervals: np.ndarray,
        timestamps: np.ndarray,
    ) -> CardiacQualityMetrics:
        """
        Comprehensive cardiac signal quality assessment.

        Args:
            ecg_signal: ECG signal
            r_peaks: Detected R-peak timestamps
            rr_intervals: RR interval array (ms)
            timestamps: Signal timestamps

        Returns:
            CardiacQualityMetrics object
        """
        # Signal quality index
        sqi = self.compute_signal_quality_index(ecg_signal, r_peaks, timestamps)

        # Ectopic beats
        ectopic_percentage, _ = self.detect_ectopic_beats(rr_intervals)

        # Missed beats
        missed_percentage = self.estimate_missed_beats(rr_intervals)

        # Noise level (estimated from high-frequency content)
        noise_level = np.std(np.diff(ecg_signal)) / np.std(ecg_signal)

        # Overall quality score
        quality_components = [
            sqi * 0.4,  # 40% weight
            (1 - ectopic_percentage / 100) * 0.3,  # 30% weight
            (1 - missed_percentage / 100) * 0.2,  # 20% weight
            (1 - min(noise_level, 1.0)) * 0.1,  # 10% weight
        ]
        overall_quality = sum(quality_components)

        # Recommendation
        if overall_quality >= 0.8:
            recommendation = "excellent"
        elif overall_quality >= 0.6:
            recommendation = "good"
        elif overall_quality >= 0.4:
            recommendation = "acceptable"
        else:
            recommendation = "poor"

        return CardiacQualityMetrics(
            overall_quality=float(overall_quality),
            signal_quality_index=float(sqi),
            ectopic_beat_percentage=float(ectopic_percentage),
            missed_beat_percentage=float(missed_percentage),
            noise_level=float(noise_level),
            recommendation=recommendation,
        )

    def benchmark_performance(
        self, signal_length: int = 60000, n_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark cardiac processing performance.

        Args:
            signal_length: Length of test signal in samples (default: 60s at 1kHz)
            n_iterations: Number of iterations to average

        Returns:
            Dictionary with performance metrics
        """
        # Generate synthetic ECG signal
        t = np.linspace(0, signal_length / self.sampling_rate, signal_length)
        ecg_signal = (
            np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz heart rhythm
            + 0.2 * np.sin(2 * np.pi * 10 * t)  # QRS component
            + 0.1 * np.random.randn(signal_length)  # Noise
        )
        timestamps = t

        # Preprocess once
        preprocessed = self.preprocess_ecg(ecg_signal)

        # Benchmark different algorithms
        algorithms = [
            (RPeakAlgorithm.PAN_TOMPKINS, self.detect_r_peaks_pan_tompkins),
            (RPeakAlgorithm.ADAPTIVE, self.detect_r_peaks_adaptive),
        ]

        results = {}

        for algorithm_name, algorithm_func in algorithms:
            times = []

            # Warm up
            algorithm_func(preprocessed, timestamps)

            # Benchmark
            for _ in range(n_iterations):
                start_time = time.perf_counter()
                algorithm_func(preprocessed, timestamps)
                end_time = time.perf_counter()
                times.append(end_time - start_time)

            # Calculate statistics
            mean_time = np.mean(times)
            std_time = np.std(times)
            samples_per_second = signal_length / mean_time

            results[f"{algorithm_name.value}_mean_time"] = float(mean_time)
            results[f"{algorithm_name.value}_std_time"] = float(std_time)
            results[f"{algorithm_name.value}_samples_per_second"] = float(samples_per_second)
            results[f"{algorithm_name.value}_realtime_factor"] = float(
                samples_per_second / self.sampling_rate
            )

        # Overall performance summary
        results["numba_available"] = float(NUMBA_AVAILABLE)
        results["signal_length"] = float(signal_length)
        results["sampling_rate"] = float(self.sampling_rate)
        results["n_iterations"] = float(n_iterations)

        return results
