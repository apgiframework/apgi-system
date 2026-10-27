"""Pupillometry interface for high-speed eye tracking.

Provides pupil diameter extraction, blink detection, baseline correction,
artifact interpolation, and luminance-independent dilation measurement
for APGI experiments.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np


class EyeType(Enum):
    """Eye tracking types."""

    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


class BlinkDetectionMethod(Enum):
    """Blink detection methods."""

    VELOCITY = "velocity"  # Based on pupil size change rate
    MISSING_DATA = "missing_data"  # Based on tracking loss
    COMBINED = "combined"  # Both methods


@dataclass
class PupilSample:
    """Single pupil measurement sample."""

    timestamp: float
    diameter: float  # mm
    x_position: float  # screen coordinates
    y_position: float  # screen coordinates
    confidence: float  # tracking confidence 0-1
    is_blink: bool = False
    is_artifact: bool = False
    luminance: Optional[float] = None  # cd/m²


@dataclass
class PupillometryConfig:
    """Configuration for pupillometry data acquisition."""

    sampling_rate: float = 1000.0  # Hz (high-speed tracking)
    buffer_size: int = 60000  # samples (60 seconds at 1000 Hz)
    eye_tracked: EyeType = EyeType.BOTH

    # Blink detection parameters
    blink_detection_method: BlinkDetectionMethod = BlinkDetectionMethod.COMBINED
    blink_velocity_threshold: float = 5.0  # mm/s
    missing_data_threshold: int = 3  # consecutive missing samples

    # Artifact detection
    min_valid_diameter: float = 2.0  # mm
    max_valid_diameter: float = 8.0  # mm
    max_diameter_change: float = 1.0  # mm per sample
    confidence_threshold: float = 0.7  # minimum tracking confidence

    # Baseline correction
    baseline_window: float = 1.0  # seconds
    baseline_method: str = "median"  # median, mean, mode

    # Luminance correction
    enable_luminance_correction: bool = True
    luminance_response_time: float = 0.3  # seconds (pupil light reflex latency)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")


class BlinkDetector:
    """Real-time blink detection for pupillometry data."""

    def __init__(self, config: PupillometryConfig):
        """Initialize blink detector.

        Args:
            config: Pupillometry configuration
        """
        self.config = config
        self.detection_history: deque = deque(maxlen=1000)

    def detect_velocity_blinks(self, diameters: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Detect blinks based on rapid pupil size changes.

        Args:
            diameters: Pupil diameter array
            timestamps: Timestamp array

        Returns:
            Boolean array indicating blink presence
        """
        if len(diameters) < 2:
            return np.zeros(len(diameters), dtype=bool)

        # Compute velocity (change in diameter over time)
        dt = np.diff(timestamps)
        dt[dt == 0] = 1e-6  # Avoid division by zero
        velocity = np.abs(np.diff(diameters)) / dt

        # Pad to match original length
        velocity = np.pad(velocity, (0, 1), mode="edge")

        # Detect excessive velocity
        blinks = velocity > self.config.blink_velocity_threshold

        # Extend blink periods (blinks typically last 100-400ms)
        blink_extension = int(0.4 * self.config.sampling_rate)
        extended_blinks = np.zeros_like(blinks)

        for i in np.where(blinks)[0]:
            start = max(0, i - blink_extension // 2)
            end = min(len(blinks), i + blink_extension // 2)
            extended_blinks[start:end] = True

        return extended_blinks

    def detect_missing_data_blinks(
        self, diameters: np.ndarray, confidence: np.ndarray
    ) -> np.ndarray:
        """Detect blinks based on tracking loss or low confidence.

        Args:
            diameters: Pupil diameter array
            confidence: Tracking confidence array

        Returns:
            Boolean array indicating blink presence
        """
        # Detect missing data (NaN or zero diameter)
        missing = np.isnan(diameters) | (diameters == 0)

        # Detect low confidence
        low_confidence = confidence < self.config.confidence_threshold

        # Combine indicators
        blinks = missing | low_confidence

        # Detect consecutive missing samples
        consecutive_missing = np.zeros_like(blinks)
        count = 0
        for i in range(len(blinks)):
            if blinks[i]:
                count += 1
                if count >= self.config.missing_data_threshold:
                    consecutive_missing[i] = True
            else:
                count = 0

        return cast(np.ndarray, consecutive_missing)

    def detect_blinks(
        self, diameters: np.ndarray, timestamps: np.ndarray, confidence: np.ndarray
    ) -> np.ndarray:
        """Detect blinks using configured method.

        Args:
            diameters: Pupil diameter array
            timestamps: Timestamp array
            confidence: Tracking confidence array

        Returns:
            Boolean array indicating blink presence
        """
        if self.config.blink_detection_method == BlinkDetectionMethod.VELOCITY:
            blinks = self.detect_velocity_blinks(diameters, timestamps)

        elif self.config.blink_detection_method == BlinkDetectionMethod.MISSING_DATA:
            blinks = self.detect_missing_data_blinks(diameters, confidence)

        else:  # COMBINED
            velocity_blinks = self.detect_velocity_blinks(diameters, timestamps)
            missing_blinks = self.detect_missing_data_blinks(diameters, confidence)
            blinks = velocity_blinks | missing_blinks

        # Store detection statistics
        blink_rate = np.sum(blinks) / len(blinks) if len(blinks) > 0 else 0
        self.detection_history.append(
            {
                "timestamp": time.time(),
                "blink_rate": blink_rate,
                "n_blinks": np.sum(blinks),
            }
        )

        return blinks


class ArtifactInterpolator:
    """Interpolate artifacts and blinks in pupillometry data."""

    def __init__(self, config: PupillometryConfig):
        """Initialize artifact interpolator.

        Args:
            config: Pupillometry configuration
        """
        self.config = config

    def detect_artifacts(self, diameters: np.ndarray) -> np.ndarray:
        """Detect physiologically implausible pupil measurements.

        Args:
            diameters: Pupil diameter array

        Returns:
            Boolean array indicating artifact presence
        """
        # Check diameter bounds
        out_of_bounds = (diameters < self.config.min_valid_diameter) | (
            diameters > self.config.max_valid_diameter
        )

        # Check for excessive changes
        if len(diameters) > 1:
            changes = np.abs(np.diff(diameters))
            changes = np.pad(changes, (0, 1), mode="edge")
            excessive_change = changes > self.config.max_diameter_change
        else:
            excessive_change = np.zeros_like(diameters, dtype=bool)

        artifacts = out_of_bounds | excessive_change

        return artifacts

    def interpolate_linear(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Linear interpolation of masked data points.

        Args:
            data: Data array
            mask: Boolean mask (True = invalid, needs interpolation)

        Returns:
            Interpolated data array
        """
        result = data.copy()

        # Find valid data points
        valid_indices = np.where(~mask)[0]

        if len(valid_indices) < 2:
            # Not enough valid points for interpolation
            return np.asarray(result)

        # Interpolate invalid points
        invalid_indices = np.where(mask)[0]

        for idx in invalid_indices:
            # Find nearest valid points
            left_valid = valid_indices[valid_indices < idx]
            right_valid = valid_indices[valid_indices > idx]

            if len(left_valid) > 0 and len(right_valid) > 0:
                # Linear interpolation between nearest valid points
                left_idx = left_valid[-1]
                right_idx = right_valid[0]

                # Interpolation weight
                weight = (idx - left_idx) / (right_idx - left_idx)
                result[idx] = (1 - weight) * data[left_idx] + weight * data[right_idx]

            elif len(left_valid) > 0:
                # Use last valid value
                result[idx] = data[left_valid[-1]]

            elif len(right_valid) > 0:
                # Use first valid value
                result[idx] = data[right_valid[0]]

        return np.asarray(result)

    def interpolate_cubic(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Cubic spline interpolation of masked data points.

        Args:
            data: Data array
            mask: Boolean mask (True = invalid, needs interpolation)

        Returns:
            Interpolated data array
        """
        from scipy.interpolate import CubicSpline

        result = data.copy()
        valid_indices = np.where(~mask)[0]

        if len(valid_indices) < 4:
            # Fall back to linear interpolation
            return self.interpolate_linear(data, mask)

        # Create cubic spline from valid points
        cs = CubicSpline(valid_indices, data[valid_indices])

        # Interpolate invalid points
        invalid_indices = np.where(mask)[0]
        result[invalid_indices] = cs(invalid_indices)

        return np.asarray(result)

    def interpolate(self, data: np.ndarray, mask: np.ndarray, method: str = "linear") -> np.ndarray:
        """Interpolate masked data points.

        Args:
            data: Data array
            mask: Boolean mask (True = invalid, needs interpolation)
            method: Interpolation method (linear, cubic)

        Returns:
            Interpolated data array
        """
        if method == "cubic":
            try:
                return self.interpolate_cubic(data, mask)
            except ImportError:
                # scipy not available, fall back to linear
                return self.interpolate_linear(data, mask)
        else:
            return self.interpolate_linear(data, mask)


class BaselineCorrector:
    """Baseline correction for pupillometry data."""

    def __init__(self, config: PupillometryConfig):
        """Initialize baseline corrector.

        Args:
            config: Pupillometry configuration
        """
        self.config = config
        self.baseline_value: Optional[float] = None

    def compute_baseline(self, data: np.ndarray, method: Optional[str] = None) -> float:
        """Compute baseline pupil diameter.

        Args:
            data: Pupil diameter array
            method: Baseline method (median, mean, mode)

        Returns:
            Baseline value
        """
        if method is None:
            method = self.config.baseline_method

        # Remove invalid values
        valid_data = data[~np.isnan(data) & (data > 0)]

        if len(valid_data) == 0:
            return 0.0

        if method == "median":
            baseline = np.median(valid_data)
        elif method == "mean":
            baseline = np.mean(valid_data)
        elif method == "mode":
            # Use histogram to find mode
            hist, edges = np.histogram(valid_data, bins=50)
            mode_idx = np.argmax(hist)
            baseline = (edges[mode_idx] + edges[mode_idx + 1]) / 2
        else:
            raise ValueError(f"Unknown baseline method: {method}")

        self.baseline_value = baseline
        return float(baseline)

    def apply_baseline_correction(
        self, data: np.ndarray, baseline: Optional[float] = None
    ) -> np.ndarray:
        """Apply baseline correction to data.

        Args:
            data: Pupil diameter array
            baseline: Baseline value (if None, uses stored baseline)

        Returns:
            Baseline-corrected data
        """
        if baseline is None:
            if self.baseline_value is None:
                baseline = self.compute_baseline(data)
            else:
                baseline = self.baseline_value

        # Subtract baseline
        corrected = data - baseline

        return corrected

    def apply_percent_change(
        self, data: np.ndarray, baseline: Optional[float] = None
    ) -> np.ndarray:
        """Convert to percent change from baseline.

        Args:
            data: Pupil diameter array
            baseline: Baseline value (if None, uses stored baseline)

        Returns:
            Percent change from baseline
        """
        if baseline is None:
            if self.baseline_value is None:
                baseline = self.compute_baseline(data)
            else:
                baseline = self.baseline_value

        if baseline == 0:
            return np.zeros_like(data)

        # Compute percent change
        percent_change = ((data - baseline) / baseline) * 100

        return percent_change


class LuminanceCorrector:
    """Luminance-independent pupil dilation measurement."""

    def __init__(self, config: PupillometryConfig):
        """Initialize luminance corrector.

        Args:
            config: Pupillometry configuration
        """
        self.config = config
        self.luminance_model_params: Optional[Dict[str, float]] = None

    def fit_luminance_model(
        self, diameters: np.ndarray, luminance: np.ndarray, timestamps: np.ndarray
    ) -> Dict[str, float]:
        """Fit model of pupil response to luminance changes.

        Args:
            diameters: Pupil diameter array
            luminance: Luminance array (cd/m²)
            timestamps: Timestamp array

        Returns:
            Model parameters
        """
        # Simple linear model: diameter = a * log(luminance) + b
        # This captures the logarithmic relationship between luminance and pupil size

        valid_mask = ~np.isnan(diameters) & ~np.isnan(luminance) & (luminance > 0) & (diameters > 0)

        if np.sum(valid_mask) < 10:
            # Not enough valid data
            return {"slope": 0.0, "intercept": np.mean(diameters[~np.isnan(diameters)])}

        valid_diameters = diameters[valid_mask]
        valid_luminance = luminance[valid_mask]

        # Log-transform luminance
        log_luminance = np.log(valid_luminance)

        # Linear regression
        slope, intercept = np.polyfit(log_luminance, valid_diameters, 1)

        self.luminance_model_params = {"slope": slope, "intercept": intercept}

        return self.luminance_model_params

    def correct_luminance_effect(
        self, diameters: np.ndarray, luminance: np.ndarray, timestamps: np.ndarray
    ) -> np.ndarray:
        """Remove luminance-driven pupil changes to isolate cognitive dilation.

        Args:
            diameters: Pupil diameter array
            luminance: Luminance array (cd/m²)
            timestamps: Timestamp array

        Returns:
            Luminance-corrected pupil diameter
        """
        if self.luminance_model_params is None:
            # Fit model if not already done
            self.fit_luminance_model(diameters, luminance, timestamps)

        # Predict luminance-driven component
        valid_mask = ~np.isnan(luminance) & (luminance > 0)
        predicted = np.zeros_like(diameters)

        if self.luminance_model_params is not None and np.any(valid_mask):
            log_luminance = np.log(luminance[valid_mask])
            predicted[valid_mask] = (
                self.luminance_model_params["slope"] * log_luminance
                + self.luminance_model_params["intercept"]
            )

        # Remove predicted luminance effect
        corrected = diameters - predicted
        if self.luminance_model_params is not None:
            corrected = diameters - predicted + self.luminance_model_params["intercept"]

        return cast(np.ndarray, corrected)


class PupillometryInterface:
    """Interface for high-speed pupillometry data acquisition and processing.

    Provides pupil diameter extraction, blink detection, baseline correction,
    artifact interpolation, and luminance-independent dilation measurement.
    """

    def __init__(self, config: PupillometryConfig):
        """Initialize pupillometry interface.

        Args:
            config: Pupillometry configuration
        """
        self.config = config
        self.blink_detector = BlinkDetector(config)
        self.artifact_interpolator = ArtifactInterpolator(config)
        self.baseline_corrector = BaselineCorrector(config)
        self.luminance_corrector = LuminanceCorrector(config)

        # Data buffers
        self.sample_buffer: deque = deque(maxlen=config.buffer_size)

        # Streaming state
        self.is_streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self.data_callbacks: List[Callable] = []

        # Statistics
        self.samples_acquired = 0
        self.blinks_detected = 0
        self.artifacts_detected = 0
        self.start_time: Optional[float] = None

    def start_streaming(self, data_source: Optional[Callable] = None) -> None:
        """Start real-time pupillometry data streaming.

        Args:
            data_source: Optional callback function that returns PupilSample
                        If None, generates simulated data for testing
        """
        if self.is_streaming:
            raise RuntimeError("Streaming already active")

        self.is_streaming = True
        self.start_time = time.time()

        def stream_loop() -> None:
            """Internal streaming loop."""
            while self.is_streaming:
                # Get data from source or simulate
                if data_source:
                    sample = data_source()
                else:
                    # Simulate data for testing
                    sample = PupilSample(
                        timestamp=time.time(),
                        diameter=4.0 + np.random.randn() * 0.3,
                        x_position=960 + np.random.randn() * 10,
                        y_position=540 + np.random.randn() * 10,
                        confidence=0.95 + np.random.randn() * 0.05,
                        luminance=50.0,
                    )

                # Buffer sample
                self.sample_buffer.append(sample)
                self.samples_acquired += 1

                # Notify callbacks
                for callback in self.data_callbacks:
                    callback(sample)

                # Sleep to match sampling rate
                time.sleep(1.0 / self.config.sampling_rate)

        self.stream_thread = threading.Thread(target=stream_loop, daemon=True)
        self.stream_thread.start()

    def stop_streaming(self) -> None:
        """Stop data streaming."""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)

    def register_callback(self, callback: Callable) -> None:
        """Register callback for real-time data processing.

        Args:
            callback: Function with signature (sample: PupilSample) -> None
        """
        self.data_callbacks.append(callback)

    def get_buffer_data(self, n_samples: Optional[int] = None) -> List[PupilSample]:
        """Retrieve samples from buffer.

        Args:
            n_samples: Number of recent samples to retrieve (None for all)

        Returns:
            List of pupil samples
        """
        samples = list(self.sample_buffer)

        if n_samples:
            samples = samples[-n_samples:]

        return samples

    def extract_diameter_timeseries(
        self, samples: Optional[List[PupilSample]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract diameter and timestamp arrays from samples.

        Args:
            samples: List of pupil samples (None to use buffer)

        Returns:
            Tuple of (diameters, timestamps)
        """
        if samples is None:
            samples = self.get_buffer_data()

        if not samples:
            return np.array([]), np.array([])

        diameters = np.array([s.diameter for s in samples])
        timestamps = np.array([s.timestamp for s in samples])

        return diameters, timestamps

    def process_data(
        self,
        samples: Optional[List[PupilSample]] = None,
        apply_blink_detection: bool = True,
        apply_artifact_correction: bool = True,
        apply_baseline_correction: bool = True,
        apply_luminance_correction: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Process pupillometry data with full pipeline.

        Args:
            samples: List of pupil samples (None to use buffer)
            apply_blink_detection: Whether to detect and mark blinks
            apply_artifact_correction: Whether to interpolate artifacts
            apply_baseline_correction: Whether to apply baseline correction
            apply_luminance_correction: Whether to correct for luminance

        Returns:
            Dictionary with processed data arrays
        """
        if samples is None:
            samples = self.get_buffer_data()

        if not samples:
            return {}

        # Extract arrays
        diameters = np.array([s.diameter for s in samples])
        timestamps = np.array([s.timestamp for s in samples])
        confidence = np.array([s.confidence for s in samples])
        luminance = np.array([s.luminance if s.luminance is not None else 50.0 for s in samples])

        result = {
            "raw_diameters": diameters.copy(),
            "timestamps": timestamps,
            "confidence": confidence,
            "luminance": luminance,
        }

        # Detect blinks
        if apply_blink_detection:
            blinks = self.blink_detector.detect_blinks(diameters, timestamps, confidence)
            result["blinks"] = blinks
            self.blinks_detected += np.sum(blinks)
        else:
            blinks = np.zeros(len(diameters), dtype=bool)
            result["blinks"] = blinks

        # Detect artifacts
        artifacts = self.artifact_interpolator.detect_artifacts(diameters)
        result["artifacts"] = artifacts
        self.artifacts_detected += np.sum(artifacts)

        # Combine blinks and artifacts for interpolation
        invalid_mask = blinks | artifacts

        # Interpolate artifacts
        if apply_artifact_correction and np.any(invalid_mask):
            diameters = self.artifact_interpolator.interpolate(diameters, invalid_mask)

        result["interpolated_diameters"] = diameters.copy()

        # Apply luminance correction
        if apply_luminance_correction and self.config.enable_luminance_correction:
            diameters = self.luminance_corrector.correct_luminance_effect(
                diameters, luminance, timestamps
            )
            result["luminance_corrected_diameters"] = diameters.copy()

        # Apply baseline correction
        if apply_baseline_correction:
            baseline = self.baseline_corrector.compute_baseline(diameters)
            corrected = self.baseline_corrector.apply_baseline_correction(diameters, baseline)
            percent_change = self.baseline_corrector.apply_percent_change(diameters, baseline)

            baseline_array = np.array([baseline]) if np.isscalar(baseline) else np.asarray(baseline)
            result["baseline"] = baseline_array
            result["baseline_corrected_diameters"] = corrected
            result["percent_change"] = percent_change

        result["final_diameters"] = diameters

        return result

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Assess pupillometry data quality.

        Returns:
            Dictionary with quality metrics
        """
        if not self.sample_buffer:
            return {"status": "no_data"}

        samples = self.get_buffer_data()

        # Extract confidence values
        confidence_values = np.array([s.confidence for s in samples])
        mean_confidence = np.mean(confidence_values)

        # Compute rates
        blink_rate = self.blinks_detected / max(self.samples_acquired, 1)
        artifact_rate = self.artifacts_detected / max(self.samples_acquired, 1)

        # Assess quality
        quality_score = mean_confidence * (1 - artifact_rate)

        return {
            "status": "good" if quality_score > 0.8 else "poor",
            "quality_score": quality_score,
            "mean_confidence": mean_confidence,
            "blink_rate": blink_rate,
            "artifact_rate": artifact_rate,
            "samples_acquired": self.samples_acquired,
            "duration_seconds": time.time() - self.start_time if self.start_time else 0,
        }

    def clear_buffer(self) -> None:
        """Clear sample buffer."""
        self.sample_buffer.clear()

    def export_data(self, filename: str, format: str = "numpy") -> None:
        """Export pupillometry data to file.

        Args:
            filename: Output filename
            format: Export format (numpy, csv)
        """
        samples = self.get_buffer_data()

        if not samples:
            raise ValueError("No data to export")

        if format == "numpy":
            data_dict = {
                "timestamps": np.array([s.timestamp for s in samples]),
                "diameters": np.array([s.diameter for s in samples]),
                "x_positions": np.array([s.x_position for s in samples]),
                "y_positions": np.array([s.y_position for s in samples]),
                "confidence": np.array([s.confidence for s in samples]),
                "is_blink": np.array([s.is_blink for s in samples]),
                "is_artifact": np.array([s.is_artifact for s in samples]),
                "sampling_rate": self.config.sampling_rate,
            }
            np.savez(filename, **data_dict)  # type: ignore

        elif format == "csv":
            import csv

            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "diameter",
                        "x_position",
                        "y_position",
                        "confidence",
                        "is_blink",
                        "is_artifact",
                        "luminance",
                    ]
                )
                for s in samples:
                    writer.writerow(
                        [
                            s.timestamp,
                            s.diameter,
                            s.x_position,
                            s.y_position,
                            s.confidence,
                            s.is_blink,
                            s.is_artifact,
                            s.luminance,
                        ]
                    )

        else:
            raise ValueError(f"Unsupported export format: {format}")
