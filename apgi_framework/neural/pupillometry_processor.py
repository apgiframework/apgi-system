"""Pupillometry signal processing pipeline for parameter estimation.

Provides pupil feature extraction, quality assessment, and metrics calculation
for interoceptive precision estimation in APGI experiments.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.integrate import trapz
except ImportError:
    # Fallback for newer scipy versions where trapz was moved
    from scipy.integrate import trapezoid as trapz


@dataclass
class PupilFeatures:
    """Pupil dilation features for a single trial or window."""

    peak_dilation: float  # Maximum dilation in window
    mean_dilation: float  # Mean dilation in window
    time_to_peak: float  # Time to peak dilation (ms)
    area_under_curve: float  # Total dilation area
    baseline_diameter: float  # Pre-stimulus baseline
    percent_change: float  # Percent change from baseline
    latency: float  # Onset latency (ms)
    duration: float  # Duration of dilation response (ms)
    quality_score: float  # Quality metric (0-1)


@dataclass
class PupilQualityMetrics:
    """Quality assessment metrics for pupillometry data."""

    overall_quality: float  # Overall quality score (0-1)
    data_loss_percentage: float  # Percentage of missing/invalid data
    blink_rate: float  # Blinks per minute
    tracking_confidence: float  # Mean tracking confidence
    signal_to_noise_ratio: float  # SNR estimate
    artifact_percentage: float  # Percentage of artifact samples
    recommendation: str  # Quality recommendation (excellent, good, acceptable, poor)


class PupillometryProcessor:
    """Pupillometry signal processor integrating blink detection and interpolation.

    Provides comprehensive pupil signal processing including artifact handling,
    baseline correction, and feature extraction for parameter estimation.
    """

    def __init__(
        self,
        sampling_rate: float = 1000.0,
        blink_detector: Optional[Any] = None,
        artifact_interpolator: Optional[Any] = None,
    ):
        """Initialize pupillometry processor.

        Args:
            sampling_rate: Pupillometry sampling rate in Hz
            blink_detector: BlinkDetector instance (from pupillometry_interface)
            artifact_interpolator: ArtifactInterpolator instance
        """
        self.sampling_rate = sampling_rate
        self.blink_detector = blink_detector
        self.artifact_interpolator = artifact_interpolator

    def process_trial(
        self,
        pupil_data: np.ndarray,
        timestamps: np.ndarray,
        confidence: np.ndarray,
        baseline_window: Tuple[float, float] = (-0.5, 0.0),
        stimulus_time: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Process pupil data for a single trial.

        Args:
            pupil_data: Pupil diameter array (mm)
            timestamps: Timestamp array (seconds)
            confidence: Tracking confidence array (0-1)
            baseline_window: Baseline time window (start, end) relative to stimulus
            stimulus_time: Stimulus onset time

        Returns:
            Dictionary with processed data arrays
        """
        result = {
            "raw_data": pupil_data.copy(),
            "timestamps": timestamps,
            "confidence": confidence,
        }

        # Detect blinks if detector available
        if self.blink_detector:
            blinks = self.blink_detector.detect_blinks(pupil_data, timestamps, confidence)
            result["blinks"] = blinks
        else:
            blinks = np.zeros(len(pupil_data), dtype=bool)
            result["blinks"] = blinks

        # Detect artifacts if interpolator available
        if self.artifact_interpolator:
            artifacts = self.artifact_interpolator.detect_artifacts(pupil_data)
            result["artifacts"] = artifacts

            # Combine blinks and artifacts
            invalid_mask = blinks | artifacts

            # Interpolate
            clean_data = self.artifact_interpolator.interpolate(
                pupil_data, invalid_mask, method="cubic"
            )
            result["clean_data"] = clean_data
        else:
            clean_data = pupil_data.copy()
            result["clean_data"] = clean_data

        # Apply baseline correction
        baseline_start_idx = np.searchsorted(timestamps, stimulus_time + baseline_window[0])
        baseline_end_idx = np.searchsorted(timestamps, stimulus_time + baseline_window[1])

        if baseline_end_idx > baseline_start_idx:
            baseline = np.median(clean_data[baseline_start_idx:baseline_end_idx])
            result["baseline"] = baseline
            result["baseline_corrected"] = clean_data - baseline
            result["percent_change"] = (
                ((clean_data - baseline) / baseline) * 100
                if baseline > 0
                else np.zeros_like(clean_data)
            )
        else:
            result["baseline"] = np.median(clean_data)
            result["baseline_corrected"] = clean_data - result["baseline"]
            result["percent_change"] = np.zeros_like(clean_data)

        return result

    def smooth_signal(self, data: np.ndarray, window_size: int = 50) -> np.ndarray:
        """Smooth pupil signal using moving average.

        Args:
            data: Pupil diameter array
            window_size: Smoothing window size in samples

        Returns:
            Smoothed data
        """
        if len(data) < window_size:
            return data

        # Apply moving average
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(data, kernel, mode="same")

        return smoothed


class PupilFeatureExtractor:
    """Extract pupil dilation features from post-stimulus windows.

    Computes peak, mean, time-to-peak, and other metrics for
    200-500ms post-stimulus windows.
    """

    def __init__(self, sampling_rate: float = 1000.0):
        """Initialize pupil feature extractor.

        Args:
            sampling_rate: Pupillometry sampling rate in Hz
        """
        self.sampling_rate = sampling_rate

    def extract_window_features(
        self,
        pupil_data: np.ndarray,
        timestamps: np.ndarray,
        stimulus_time: float,
        window: Tuple[float, float] = (0.2, 0.5),
        baseline: Optional[float] = None,
    ) -> PupilFeatures:
        """Extract features from a post-stimulus time window.

        Args:
            pupil_data: Pupil diameter array (baseline-corrected or raw)
            timestamps: Timestamp array
            stimulus_time: Stimulus onset time
            window: Time window (start, end) in seconds relative to stimulus
            baseline: Baseline diameter (if None, computed from data)

        Returns:
            PupilFeatures object
        """
        # Find window indices
        window_start_idx = np.searchsorted(timestamps, stimulus_time + window[0])
        window_end_idx = np.searchsorted(timestamps, stimulus_time + window[1])

        if window_end_idx <= window_start_idx or window_end_idx > len(pupil_data):
            # Invalid window, return default features
            return PupilFeatures(
                peak_dilation=0.0,
                mean_dilation=0.0,
                time_to_peak=0.0,
                area_under_curve=0.0,
                baseline_diameter=baseline if baseline else 0.0,
                percent_change=0.0,
                latency=0.0,
                duration=0.0,
                quality_score=0.0,
            )

        # Extract window data
        window_data = pupil_data[window_start_idx:window_end_idx]
        window_times = timestamps[window_start_idx:window_end_idx]

        # Compute baseline if not provided
        if baseline is None:
            # Use pre-stimulus period
            baseline_idx = np.searchsorted(timestamps, stimulus_time - 0.5)
            baseline_end_idx = np.searchsorted(timestamps, stimulus_time)
            if baseline_end_idx > baseline_idx:
                baseline = np.median(pupil_data[baseline_idx:baseline_end_idx])
            else:
                baseline = (
                    np.median(pupil_data[:window_start_idx])
                    if window_start_idx > 0
                    else np.median(pupil_data)
                )

        # Peak dilation
        peak_dilation = np.max(window_data)
        peak_idx = np.argmax(window_data)

        # Time to peak (ms)
        time_to_peak = (window_times[peak_idx] - (stimulus_time + window[0])) * 1000

        # Mean dilation
        mean_dilation = np.mean(window_data)

        # Area under curve
        area_under_curve = trapz(window_data, window_times)

        # Percent change from baseline
        percent_change = ((peak_dilation - baseline) / baseline * 100) if baseline > 0 else 0.0

        # Onset latency (time when signal exceeds threshold)
        threshold = baseline + 0.1 * (peak_dilation - baseline)
        onset_indices = np.where(window_data > threshold)[0]
        if len(onset_indices) > 0:
            latency = (window_times[onset_indices[0]] - stimulus_time) * 1000
        else:
            latency = 0.0

        # Duration (time above threshold)
        duration = len(onset_indices) / self.sampling_rate * 1000

        # Quality score (based on signal characteristics)
        quality_score = self._compute_quality_score(window_data, baseline)

        return PupilFeatures(
            peak_dilation=float(peak_dilation),
            mean_dilation=float(mean_dilation),
            time_to_peak=float(time_to_peak),
            area_under_curve=float(area_under_curve),
            baseline_diameter=float(baseline),
            percent_change=float(percent_change),
            latency=float(latency),
            duration=float(duration),
            quality_score=float(quality_score),
        )

    def _compute_quality_score(self, data: np.ndarray, baseline: float) -> float:
        """Compute quality score for pupil data.

        Args:
            data: Pupil data array
            baseline: Baseline diameter

        Returns:
            Quality score (0-1)
        """
        # Check for sufficient signal variation
        signal_range = np.ptp(data)
        if signal_range < 0.1:  # Less than 0.1mm variation
            return 0.3

        # Check for physiological plausibility
        if baseline < 2.0 or baseline > 8.0:  # Outside normal range
            return 0.5

        # Check for smooth signal (not too noisy)
        noise_estimate = np.std(np.diff(data))
        if noise_estimate > 0.5:  # High noise
            return 0.6

        # Good quality
        return 0.9

    def extract_trial_features(
        self,
        pupil_data: np.ndarray,
        timestamps: np.ndarray,
        event_times: List[float],
        window: Tuple[float, float] = (0.2, 0.5),
    ) -> List[PupilFeatures]:
        """Extract features for multiple events/trials.

        Args:
            pupil_data: Continuous pupil diameter array
            timestamps: Timestamp array
            event_times: List of event onset times
            window: Time window for feature extraction

        Returns:
            List of PupilFeatures for each event
        """
        features_list = []

        for event_time in event_times:
            features = self.extract_window_features(pupil_data, timestamps, event_time, window)
            features_list.append(features)

        return features_list


class PupilMetricsCalculator:
    """Calculate comprehensive pupil dilation metrics.

    Computes area under curve, velocity, and other derived metrics
    for interoceptive precision estimation.
    """

    def __init__(self, sampling_rate: float = 1000.0):
        """Initialize pupil metrics calculator.

        Args:
            sampling_rate: Pupillometry sampling rate in Hz
        """
        self.sampling_rate = sampling_rate

    def compute_dilation_velocity(
        self, pupil_data: np.ndarray, timestamps: np.ndarray
    ) -> np.ndarray:
        """Compute pupil dilation velocity.

        Args:
            pupil_data: Pupil diameter array
            timestamps: Timestamp array

        Returns:
            Velocity array (mm/s)
        """
        if len(pupil_data) < 2:
            return np.array([])

        # Compute velocity
        dt = np.diff(timestamps)
        dt[dt == 0] = 1e-6  # Avoid division by zero
        velocity = np.diff(pupil_data) / dt

        # Pad to match original length
        velocity = np.pad(velocity, (0, 1), mode="edge")

        return velocity

    def compute_area_under_curve(
        self,
        pupil_data: np.ndarray,
        timestamps: np.ndarray,
        baseline: Optional[float] = None,
    ) -> float:
        """Compute area under the pupil dilation curve.

        Args:
            pupil_data: Pupil diameter array
            timestamps: Timestamp array
            baseline: Baseline to subtract (if None, uses minimum)

        Returns:
            Area under curve
        """
        if len(pupil_data) < 2:
            return 0.0

        # Subtract baseline
        if baseline is not None:
            data = pupil_data - baseline
        else:
            data = pupil_data - np.min(pupil_data)

        # Compute area using trapezoidal integration
        area = trapz(data, timestamps)

        return float(area)

    def compute_peak_metrics(
        self,
        pupil_data: np.ndarray,
        timestamps: np.ndarray,
        baseline: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute peak-related metrics.

        Args:
            pupil_data: Pupil diameter array
            timestamps: Timestamp array
            baseline: Baseline diameter

        Returns:
            Dictionary with peak metrics
        """
        if len(pupil_data) == 0:
            return {"peak_amplitude": 0.0, "peak_latency": 0.0, "peak_velocity": 0.0}

        # Find peak
        peak_idx = np.argmax(pupil_data)
        peak_amplitude = pupil_data[peak_idx]
        peak_latency = timestamps[peak_idx] - timestamps[0]

        # Compute velocity at peak
        velocity = self.compute_dilation_velocity(pupil_data, timestamps)
        peak_velocity = velocity[peak_idx] if len(velocity) > peak_idx else 0.0

        # Adjust for baseline
        if baseline is not None:
            peak_amplitude -= baseline

        return {
            "peak_amplitude": float(peak_amplitude),
            "peak_latency": float(peak_latency),
            "peak_velocity": float(peak_velocity),
        }

    def compute_response_metrics(
        self,
        pupil_data: np.ndarray,
        timestamps: np.ndarray,
        stimulus_time: float,
        response_window: Tuple[float, float] = (0.2, 0.5),
    ) -> Dict[str, float]:
        """Compute comprehensive response metrics for a stimulus.

        Args:
            pupil_data: Pupil diameter array
            timestamps: Timestamp array
            stimulus_time: Stimulus onset time
            response_window: Time window for response (start, end) in seconds

        Returns:
            Dictionary with response metrics
        """
        # Find response window
        window_start_idx = np.searchsorted(timestamps, stimulus_time + response_window[0])
        window_end_idx = np.searchsorted(timestamps, stimulus_time + response_window[1])

        if window_end_idx <= window_start_idx:
            return {
                "mean_response": 0.0,
                "peak_response": 0.0,
                "auc": 0.0,
                "latency": 0.0,
            }

        # Extract window
        window_data = pupil_data[window_start_idx:window_end_idx]
        window_times = timestamps[window_start_idx:window_end_idx]

        # Compute baseline
        baseline_idx = np.searchsorted(timestamps, stimulus_time - 0.5)
        baseline_end_idx = np.searchsorted(timestamps, stimulus_time)
        if baseline_end_idx > baseline_idx:
            baseline = np.median(pupil_data[baseline_idx:baseline_end_idx])
        else:
            baseline = (
                np.median(pupil_data[:window_start_idx])
                if window_start_idx > 0
                else np.median(pupil_data)
            )

        # Compute metrics
        mean_response = np.mean(window_data) - baseline
        peak_response = np.max(window_data) - baseline
        auc = self.compute_area_under_curve(window_data, window_times, baseline)

        # Latency to response onset
        threshold = baseline + 0.1 * peak_response
        onset_indices = np.where(window_data > threshold)[0]
        latency = (
            (window_times[onset_indices[0]] - stimulus_time) if len(onset_indices) > 0 else 0.0
        )

        return {
            "mean_response": float(mean_response),
            "peak_response": float(peak_response),
            "auc": float(auc),
            "latency": float(latency),
        }


class PupilQualityScorer:
    """Comprehensive data quality assessment for pupillometry.

    Evaluates tracking quality, artifact levels, and provides
    recommendations for data usability.
    """

    def __init__(self, sampling_rate: float = 1000.0):
        """Initialize pupil quality scorer.

        Args:
            sampling_rate: Pupillometry sampling rate in Hz
        """
        self.sampling_rate = sampling_rate

    def assess_quality(
        self,
        pupil_data: np.ndarray,
        confidence: np.ndarray,
        blinks: Optional[np.ndarray] = None,
        artifacts: Optional[np.ndarray] = None,
    ) -> PupilQualityMetrics:
        """Assess comprehensive data quality.

        Args:
            pupil_data: Pupil diameter array
            confidence: Tracking confidence array
            blinks: Boolean array indicating blinks
            artifacts: Boolean array indicating artifacts

        Returns:
            PupilQualityMetrics object
        """
        n_samples = len(pupil_data)

        # Data loss percentage
        missing_data = np.isnan(pupil_data) | (pupil_data == 0)
        data_loss_percentage = np.sum(missing_data) / n_samples * 100

        # Blink rate
        if blinks is not None:
            n_blinks = np.sum(np.diff(blinks.astype(int)) == 1)  # Count blink onsets
            duration_minutes = n_samples / self.sampling_rate / 60
            blink_rate = n_blinks / duration_minutes if duration_minutes > 0 else 0
        else:
            blink_rate = 0.0

        # Tracking confidence
        valid_confidence = confidence[~np.isnan(confidence)]
        tracking_confidence = np.mean(valid_confidence) if len(valid_confidence) > 0 else 0.0

        # Signal-to-noise ratio estimate
        valid_data = pupil_data[~missing_data]
        if len(valid_data) > 10:
            signal_power = np.var(valid_data)
            noise_power = np.var(np.diff(valid_data))
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        else:
            snr = 0.0

        # Artifact percentage
        if artifacts is not None:
            artifact_percentage = np.sum(artifacts) / n_samples * 100
        else:
            artifact_percentage = 0.0

        # Overall quality score (weighted combination)
        quality_components = [
            (1 - data_loss_percentage / 100) * 0.3,  # 30% weight
            tracking_confidence * 0.3,  # 30% weight
            (1 - artifact_percentage / 100) * 0.2,  # 20% weight
            min(snr / 20, 1.0) * 0.2,  # 20% weight (normalize SNR to 0-1)
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

        return PupilQualityMetrics(
            overall_quality=float(overall_quality),
            data_loss_percentage=float(data_loss_percentage),
            blink_rate=float(blink_rate),
            tracking_confidence=float(tracking_confidence),
            signal_to_noise_ratio=float(snr),
            artifact_percentage=float(artifact_percentage),
            recommendation=recommendation,
        )

    def assess_trial_quality(
        self,
        pupil_data: np.ndarray,
        confidence: np.ndarray,
        trial_window: Tuple[int, int],
    ) -> float:
        """Assess quality for a specific trial window.

        Args:
            pupil_data: Pupil diameter array
            confidence: Tracking confidence array
            trial_window: (start_idx, end_idx) for trial

        Returns:
            Quality score (0-1) for the trial
        """
        start_idx, end_idx = trial_window

        # Extract trial data
        trial_data = pupil_data[start_idx:end_idx]
        trial_confidence = confidence[start_idx:end_idx]

        # Check for missing data
        missing = np.isnan(trial_data) | (trial_data == 0)
        data_completeness = 1 - (np.sum(missing) / len(trial_data))

        # Check confidence
        mean_confidence = np.mean(trial_confidence[~np.isnan(trial_confidence)])

        # Check for physiological plausibility
        valid_data = trial_data[~missing]
        if len(valid_data) > 0:
            mean_diameter = np.mean(valid_data)
            plausibility = 1.0 if 2.0 <= mean_diameter <= 8.0 else 0.5
        else:
            plausibility = 0.0

        # Combined quality score
        quality = data_completeness * 0.4 + mean_confidence * 0.4 + plausibility * 0.2

        return float(quality)
