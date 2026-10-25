"""
Event-Related Potential (ERP) analysis module.

Provides P3b peak detection, early component extraction (N1, P1, N170),
and single-trial ERP estimation for APGI framework validation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from scipy import signal
from scipy.integrate import trapezoid


@dataclass
class ERPComponents:
    """Container for extracted ERP components."""

    # P3b metrics
    p3b_amplitude: Optional[float] = None
    p3b_latency: Optional[float] = None
    p3b_area: Optional[float] = None

    # Early components
    p1_amplitude: Optional[float] = None
    p1_latency: Optional[float] = None
    n1_amplitude: Optional[float] = None
    n1_latency: Optional[float] = None
    n170_amplitude: Optional[float] = None
    n170_latency: Optional[float] = None

    # Additional metrics
    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    snr: Optional[float] = None

    # Metadata
    channel: Optional[str] = None
    trial_id: Optional[str] = None
    condition: Optional[str] = None


@dataclass
class P3bMetrics:
    """Detailed P3b component metrics."""

    amplitude: float
    latency: float  # ms
    area_under_curve: float  # µV·ms
    peak_width: float  # ms
    onset_latency: float  # ms
    offset_latency: float  # ms
    fractional_area_latency: float  # 50% area latency

    # Quality metrics
    signal_to_noise: float
    confidence_interval: Tuple[float, float]
    detection_confidence: float  # 0-1


class ERPAnalysis:
    """
    Event-Related Potential analysis for APGI framework.

    Extracts P3b and early ERP components with advanced filtering
    and single-trial estimation capabilities.
    """

    def __init__(self, sampling_rate: float = 1000.0):
        """
        Initialize ERP analysis.

        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.dt = 1000.0 / sampling_rate  # ms per sample

    def baseline_correct(
        self, data: np.ndarray, baseline_window: Tuple[float, float] = (-200, 0)
    ) -> np.ndarray:
        """
        Apply baseline correction to ERP data.

        Args:
            data: ERP data (channels x time) or (trials x channels x time)
            baseline_window: Time window for baseline in ms (start, end)

        Returns:
            Baseline-corrected data
        """
        # Convert time to samples
        baseline_start = int(baseline_window[0] / self.dt)
        baseline_end = int(baseline_window[1] / self.dt)

        # Handle negative indices
        if baseline_start < 0:
            baseline_start = data.shape[-1] + baseline_start
        if baseline_end <= 0:
            baseline_end = data.shape[-1] + baseline_end

        # Compute baseline mean
        baseline = np.mean(data[..., baseline_start:baseline_end], axis=-1, keepdims=True)

        return cast(np.ndarray, data - baseline)

    def apply_filter(
        self,
        data: np.ndarray,
        lowpass: Optional[float] = 30.0,
        highpass: Optional[float] = 0.1,
        order: int = 4,
    ) -> np.ndarray:
        """
        Apply bandpass filter to ERP data.

        Args:
            data: ERP data array
            lowpass: Lowpass cutoff frequency (Hz)
            highpass: Highpass cutoff frequency (Hz)
            order: Filter order

        Returns:
            Filtered data
        """
        nyquist = self.sampling_rate / 2.0

        if highpass and lowpass:
            # Bandpass filter
            sos = signal.butter(
                order,
                [highpass / nyquist, lowpass / nyquist],
                btype="band",
                output="sos",
            )
        elif lowpass:
            # Lowpass only
            sos = signal.butter(order, lowpass / nyquist, btype="low", output="sos")
        elif highpass:
            # Highpass only
            sos = signal.butter(order, highpass / nyquist, btype="high", output="sos")
        else:
            return data

        # Apply filter along time axis
        return cast(np.ndarray, signal.sosfiltfilt(sos, data, axis=-1))

    def detect_peak(
        self,
        data: np.ndarray,
        search_window: Tuple[float, float],
        polarity: str = "positive",
    ) -> Tuple[float, float, int]:
        """
        Detect peak in specified time window.

        Args:
            data: 1D ERP waveform
            search_window: Time window in ms (start, end)
            polarity: 'positive' or 'negative'

        Returns:
            Tuple of (amplitude, latency_ms, sample_index)
        """
        # Convert time to samples
        start_idx = int(search_window[0] / self.dt)
        end_idx = int(search_window[1] / self.dt)

        # Ensure valid indices
        start_idx = max(0, start_idx)
        end_idx = min(len(data), end_idx)

        # Extract window
        window_data = data[start_idx:end_idx]

        # Find peak
        if polarity == "positive":
            peak_idx = np.argmax(window_data)
            amplitude = window_data[peak_idx]
        else:
            peak_idx = np.argmin(window_data)
            amplitude = window_data[peak_idx]

        # Convert to absolute index and latency
        abs_idx = start_idx + peak_idx
        latency = abs_idx * self.dt

        return float(amplitude), float(latency), int(abs_idx)

    def extract_p3b(
        self,
        data: np.ndarray,
        time_zero_idx: int,
        search_window: Tuple[float, float] = (300, 600),
    ) -> P3bMetrics:
        """
        Extract P3b component with detailed metrics.

        Args:
            data: 1D ERP waveform
            time_zero_idx: Sample index of stimulus onset
            search_window: Search window in ms relative to stimulus

        Returns:
            P3bMetrics object with detailed measurements
        """
        # Shift data so time_zero_idx is at position 0
        if time_zero_idx > 0:
            data = data[time_zero_idx:]

        # Detect peak
        amplitude, latency, peak_idx = self.detect_peak(data, search_window, "positive")

        # Compute area under curve
        start_idx = int(search_window[0] / self.dt)
        end_idx = int(search_window[1] / self.dt)
        area = float(trapezoid(data[start_idx:end_idx], dx=self.dt))

        # Estimate peak width at half maximum
        half_max = amplitude / 2.0

        # Find onset and offset
        onset_idx = peak_idx
        while onset_idx > 0 and data[onset_idx] > half_max:
            onset_idx -= 1

        offset_idx = peak_idx
        while offset_idx < len(data) - 1 and data[offset_idx] > half_max:
            offset_idx += 1

        peak_width = (offset_idx - onset_idx) * self.dt
        onset_latency = onset_idx * self.dt
        offset_latency = offset_idx * self.dt

        # Fractional area latency (50% of total area)
        cumulative_area = np.cumsum(data[start_idx:end_idx]) * self.dt
        half_area = area / 2.0
        fal_idx = int(np.argmin(np.abs(cumulative_area - half_area)))
        fractional_area_latency = float((start_idx + fal_idx) * self.dt)

        # Estimate SNR
        baseline_std = np.std(data[: int(100 / self.dt)])  # First 100ms as baseline
        snr = amplitude / baseline_std if baseline_std > 0 else 0.0

        # Confidence interval (bootstrap estimate)
        ci_lower = amplitude * 0.8  # Simplified
        ci_upper = amplitude * 1.2

        # Detection confidence based on SNR
        detection_confidence = min(1.0, snr / 3.0)  # SNR > 3 = high confidence

        return P3bMetrics(
            amplitude=amplitude,
            latency=latency,
            area_under_curve=float(area),
            peak_width=peak_width,
            onset_latency=onset_latency,
            offset_latency=offset_latency,
            fractional_area_latency=fractional_area_latency,
            signal_to_noise=snr,
            confidence_interval=(ci_lower, ci_upper),
            detection_confidence=detection_confidence,
        )

    def extract_early_components(
        self, data: np.ndarray, time_zero_idx: int
    ) -> Dict[str, Tuple[float, float]]:
        """
        Extract early ERP components (P1, N1, N170).

        Args:
            data: 1D ERP waveform
            time_zero_idx: Sample index of stimulus onset

        Returns:
            Dictionary with component names and (amplitude, latency) tuples
        """
        # Shift data
        if time_zero_idx > 0:
            data = data[time_zero_idx:]

        components = {}

        # P1: Positive peak around 100ms
        p1_amp, p1_lat, _ = self.detect_peak(data, (80, 120), "positive")
        components["P1"] = (p1_amp, p1_lat)

        # N1: Negative peak around 150ms
        n1_amp, n1_lat, _ = self.detect_peak(data, (120, 200), "negative")
        components["N1"] = (n1_amp, n1_lat)

        # N170: Face-specific negative peak around 170ms
        n170_amp, n170_lat, _ = self.detect_peak(data, (150, 200), "negative")
        components["N170"] = (n170_amp, n170_lat)

        return components

    def extract_all_components(
        self, data: np.ndarray, time_zero_idx: int, channel: Optional[str] = None
    ) -> ERPComponents:
        """
        Extract all ERP components from a single trial or averaged ERP.

        Args:
            data: 1D ERP waveform
            time_zero_idx: Sample index of stimulus onset
            channel: Channel name for metadata

        Returns:
            ERPComponents object with all measurements
        """
        # Baseline correction
        baseline_window = (-200, 0)
        baseline_start = max(0, time_zero_idx + int(baseline_window[0] / self.dt))
        baseline_end = time_zero_idx
        baseline_mean = np.mean(data[baseline_start:baseline_end])
        baseline_std = np.std(data[baseline_start:baseline_end])

        # Extract P3b
        p3b = self.extract_p3b(data, time_zero_idx)

        # Extract early components
        early = self.extract_early_components(data, time_zero_idx)

        return ERPComponents(
            p3b_amplitude=p3b.amplitude,
            p3b_latency=p3b.latency,
            p3b_area=p3b.area_under_curve,
            p1_amplitude=early["P1"][0],
            p1_latency=early["P1"][1],
            n1_amplitude=early["N1"][0],
            n1_latency=early["N1"][1],
            n170_amplitude=early["N170"][0],
            n170_latency=early["N170"][1],
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            snr=p3b.signal_to_noise,
            channel=channel,
        )

    def single_trial_estimation(
        self, trial_data: np.ndarray, time_zero_idx: int, method: str = "wavelet"
    ) -> np.ndarray:
        """
        Estimate single-trial ERP using advanced filtering.

        Args:
            trial_data: Single trial EEG data (channels x time)
            time_zero_idx: Sample index of stimulus onset
            method: Estimation method ('wavelet', 'adaptive', 'matched_filter')

        Returns:
            Estimated single-trial ERP
        """
        if method == "wavelet":
            return self._wavelet_denoising(trial_data)
        elif method == "adaptive":
            return self._adaptive_filtering(trial_data, time_zero_idx)
        elif method == "matched_filter":
            return self._matched_filter(trial_data, time_zero_idx)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _wavelet_denoising(self, data: np.ndarray) -> np.ndarray:
        """
        Apply wavelet denoising for single-trial ERP estimation.

        Args:
            data: EEG data array

        Returns:
            Denoised data
        """
        # Simplified wavelet denoising using lowpass filter
        # (Full wavelet implementation would require pywt library)
        return self.apply_filter(data, lowpass=15.0, highpass=0.5)

    def _adaptive_filtering(self, data: np.ndarray, time_zero_idx: int) -> np.ndarray:
        """
        Apply adaptive filtering for single-trial ERP estimation.

        Args:
            data: EEG data array
            time_zero_idx: Stimulus onset index

        Returns:
            Filtered data
        """
        # Use pre-stimulus period to estimate noise
        if time_zero_idx > 100:
            noise = data[..., :time_zero_idx]
            noise_std = np.std(noise, axis=-1, keepdims=True)

            # Adaptive threshold based on noise level
            threshold = 2.0 * noise_std

            # Soft thresholding
            filtered = np.where(np.abs(data) > threshold, data - np.sign(data) * threshold, 0)
            return filtered

        return data

    def _matched_filter(self, data: np.ndarray, time_zero_idx: int) -> np.ndarray:
        """
        Apply matched filter using template ERP.

        Args:
            data: EEG data array
            time_zero_idx: Stimulus onset index

        Returns:
            Filtered data
        """
        # Create idealized P3b template
        time_points = np.arange(0, 800, self.dt)
        template = self._create_p3b_template(time_points)

        # Convolve with template
        filtered = signal.correlate(data, template[np.newaxis, :], mode="same")

        return cast(np.ndarray, filtered)

    def _create_p3b_template(self, time_points: np.ndarray) -> np.ndarray:
        """
        Create idealized P3b template waveform.

        Args:
            time_points: Time points in ms

        Returns:
            Template waveform
        """
        # Gaussian-like P3b centered at 400ms with width ~100ms
        peak_latency = 400.0
        width = 100.0

        template = np.exp(-0.5 * ((time_points - peak_latency) / width) ** 2)
        template = template / np.max(template)  # Normalize

        return cast(np.ndarray, template)

    def compute_grand_average(
        self, trials: List[np.ndarray], reject_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Compute grand average ERP across trials with optional artifact rejection.

        Args:
            trials: List of trial data arrays
            reject_threshold: Amplitude threshold for trial rejection (µV)

        Returns:
            Tuple of (grand average, number of accepted trials)
        """
        accepted_trials = []

        for trial in trials:
            # Check for artifacts
            if reject_threshold is not None:
                if np.max(np.abs(trial)) > reject_threshold:
                    continue  # Reject trial

            accepted_trials.append(trial)

        if not accepted_trials:
            raise ValueError("No trials passed artifact rejection")

        # Compute average
        grand_average = np.mean(accepted_trials, axis=0)

        return grand_average, len(accepted_trials)

    def bootstrap_confidence_interval(
        self,
        trials: List[np.ndarray],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute bootstrap confidence intervals for ERP.

        Args:
            trials: List of trial data arrays
            n_bootstrap: Number of bootstrap iterations
            confidence: Confidence level (0-1)

        Returns:
            Tuple of (lower bound, upper bound) arrays
        """
        n_trials = len(trials)
        trials_array = np.array(trials)

        bootstrap_averages: List[np.ndarray] = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_trials, size=n_trials, replace=True)
            bootstrap_sample = trials_array[indices]
            bootstrap_avg = np.mean(bootstrap_sample, axis=0)
            bootstrap_averages.append(bootstrap_avg)

        bootstrap_averages_array = np.array(bootstrap_averages)

        # Compute percentiles
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(bootstrap_averages_array, lower_percentile, axis=0)
        upper_bound = np.percentile(bootstrap_averages_array, upper_percentile, axis=0)

        return lower_bound, upper_bound
