"""EEG/MEG interface with real-time processing capabilities.

Provides data streaming, buffering, artifact detection, and channel management
for neural data acquisition in APGI experiments.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np


class ChannelType(Enum):
    """EEG channel types."""

    EEG = "eeg"
    EOG = "eog"
    ECG = "ecg"
    EMG = "emg"
    REFERENCE = "reference"
    GROUND = "ground"


@dataclass
class ChannelInfo:
    """Information about a single EEG channel."""

    name: str
    type: ChannelType
    position: Optional[Tuple[float, float, float]] = None  # 3D coordinates
    impedance: Optional[float] = None  # in kOhms
    enabled: bool = True


@dataclass
class EEGConfig:
    """Configuration for EEG/MEG data acquisition."""

    sampling_rate: float = 1000.0  # Hz
    buffer_size: int = 10000  # samples
    channels: List[ChannelInfo] = field(default_factory=list)
    reference_type: str = "average"  # average, mastoid, cz
    highpass_freq: float = 0.1  # Hz
    lowpass_freq: float = 100.0  # Hz
    notch_freq: Optional[float] = 60.0  # Hz (power line)
    artifact_threshold: float = 100.0  # microvolts
    enable_realtime_filtering: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")
        if self.highpass_freq >= self.lowpass_freq:
            raise ValueError("Highpass frequency must be less than lowpass")


class ArtifactDetector:
    """Real-time artifact detection for EEG data."""

    def __init__(self, config: EEGConfig):
        """Initialize artifact detector.

        Args:
            config: EEG configuration with artifact thresholds
        """
        self.config = config
        self.threshold = config.artifact_threshold
        self.detection_history: deque = deque(maxlen=1000)

    def detect_amplitude_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Detect amplitude-based artifacts (e.g., muscle artifacts, electrode pops).

        Args:
            data: EEG data array (channels x samples)

        Returns:
            Boolean array indicating artifact presence (channels x samples)
        """
        # Detect samples exceeding threshold
        artifacts = np.abs(data) > self.threshold

        # Store detection statistics
        artifact_rate = np.mean(artifacts)
        self.detection_history.append(
            {
                "timestamp": time.time(),
                "artifact_rate": artifact_rate,
                "affected_channels": np.sum(np.any(artifacts, axis=1)),
            }
        )

        return cast(np.ndarray, artifacts)

    def detect_gradient_artifacts(
        self, data: np.ndarray, gradient_threshold: float = 50.0
    ) -> np.ndarray:
        """Detect rapid voltage changes indicating artifacts.

        Args:
            data: EEG data array (channels x samples)
            gradient_threshold: Maximum allowed voltage change (µV/sample)

        Returns:
            Boolean array indicating artifact presence
        """
        # Handle single sample case
        if data.shape[1] < 2:
            return np.zeros_like(data, dtype=bool)

        # Compute temporal gradient
        gradient = np.diff(data, axis=1)

        # Pad to match original shape
        gradient = np.pad(gradient, ((0, 0), (0, 1)), mode="constant", constant_values=0)

        # Detect excessive gradients
        artifacts = np.abs(gradient) > gradient_threshold

        return artifacts

    def detect_flatline(
        self, data: np.ndarray, window_size: int = 100, variance_threshold: float = 0.1
    ) -> np.ndarray:
        """Detect flatline artifacts (disconnected electrodes).

        Args:
            data: EEG data array (channels x samples)
            window_size: Window for variance calculation
            variance_threshold: Minimum variance to consider signal valid

        Returns:
            Boolean array indicating flatline channels
        """
        n_channels, n_samples = data.shape
        flatline = np.zeros((n_channels, n_samples), dtype=bool)

        # Check variance in sliding windows
        for i in range(0, n_samples - window_size, window_size // 2):
            window = data[:, i : i + window_size]
            variance = np.var(window, axis=1)
            flatline[:, i : i + window_size] = variance[:, np.newaxis] < variance_threshold

        return flatline

    def detect_all(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Run all artifact detection methods.

        Args:
            data: EEG data array (channels x samples)

        Returns:
            Dictionary with artifact types and boolean masks
        """
        return {
            "amplitude": self.detect_amplitude_artifacts(data),
            "gradient": self.detect_gradient_artifacts(data),
            "flatline": self.detect_flatline(data),
            "any": (
                self.detect_amplitude_artifacts(data)
                | self.detect_gradient_artifacts(data)
                | self.detect_flatline(data)
            ),
        }


class EEGInterface:
    """Interface for EEG/MEG data acquisition with real-time processing.

    Provides streaming, buffering, artifact detection, and channel management
    for neural data collection in APGI experiments.
    """

    def __init__(self, config: EEGConfig):
        """Initialize EEG interface.

        Args:
            config: EEG configuration parameters
        """
        self.config = config
        self.artifact_detector = ArtifactDetector(config)

        # Data buffers
        self.buffer: deque[np.ndarray] = deque(maxlen=config.buffer_size)
        self.timestamp_buffer: deque[float] = deque(maxlen=config.buffer_size)

        # Channel management
        self.channels = {ch.name: ch for ch in config.channels}
        self.active_channels = [ch.name for ch in config.channels if ch.enabled]

        # Streaming state
        self.is_streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self.data_callbacks: List[Callable] = []

        # Statistics
        self.samples_acquired = 0
        self.artifacts_detected = 0
        self.start_time: Optional[float] = None

    def add_channels(self, channels: List[ChannelInfo]) -> None:
        """Add channels to the montage.

        Args:
            channels: List of channel information objects
        """
        for ch in channels:
            self.channels[ch.name] = ch
            if ch.enabled:
                self.active_channels.append(ch.name)

    def set_reference(
        self, reference_type: str, reference_channels: Optional[List[str]] = None
    ) -> None:
        """Configure reference scheme.

        Args:
            reference_type: Type of reference (average, mastoid, cz, custom)
            reference_channels: Specific channels for custom reference
        """
        self.config.reference_type = reference_type

        if reference_type == "custom" and reference_channels:
            # Validate reference channels exist
            for ch in reference_channels:
                if ch not in self.channels:
                    raise ValueError(f"Reference channel {ch} not found")

    def apply_reference(self, data: np.ndarray) -> np.ndarray:
        """Apply reference scheme to data.

        Args:
            data: EEG data array (channels x samples)

        Returns:
            Referenced data
        """
        if self.config.reference_type == "average":
            # Average reference
            reference = np.mean(data, axis=0, keepdims=True)
            return cast(np.ndarray, np.subtract(data, reference))

        elif self.config.reference_type == "none":
            return data

        else:
            # For other reference types, return data as-is
            # (would need specific channel indices for mastoid, Cz, etc.)
            return data

    def start_streaming(self, data_source: Optional[Callable] = None) -> None:
        """Start real-time data streaming.

        Args:
            data_source: Optional callback function that returns (data, timestamp)
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
                    data, timestamp = data_source()
                else:
                    # Simulate data for testing
                    n_channels = len(self.active_channels)
                    data = np.random.randn(n_channels, 1) * 10.0
                    timestamp = time.time()

                # Apply reference
                data = self.apply_reference(data)

                # Detect artifacts
                artifacts = self.artifact_detector.detect_all(data)

                # Buffer data
                self.buffer.append(data)
                self.timestamp_buffer.append(timestamp)
                self.samples_acquired += data.shape[1]

                if np.any(artifacts["any"]):
                    self.artifacts_detected += np.sum(artifacts["any"])

                # Notify callbacks
                for callback in self.data_callbacks:
                    callback(data, timestamp, artifacts)

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
            callback: Function with signature (data, timestamp, artifacts) -> None
        """
        self.data_callbacks.append(callback)

    def get_buffer_data(self, n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve data from buffer.

        Args:
            n_samples: Number of recent samples to retrieve (None for all)

        Returns:
            Tuple of (data array, timestamp array)
        """
        if not self.buffer:
            return np.array([]), np.array([])

        # Convert buffer to array
        data_list = list(self.buffer)
        timestamp_list = list(self.timestamp_buffer)

        if n_samples:
            data_list = data_list[-n_samples:]
            timestamp_list = timestamp_list[-n_samples:]

        # Concatenate along time axis
        data = np.concatenate(data_list, axis=1)
        timestamps = np.array(timestamp_list)

        return data, timestamps

    def get_channel_impedances(self) -> Dict[str, float]:
        """Get current channel impedances.

        Returns:
            Dictionary mapping channel names to impedances (kOhms)
        """
        return {
            name: ch.impedance for name, ch in self.channels.items() if ch.impedance is not None
        }

    def check_signal_quality(self) -> Dict[str, Any]:
        """Assess overall signal quality.

        Returns:
            Dictionary with quality metrics
        """
        if not self.buffer:
            return {"status": "no_data"}

        data, _ = self.get_buffer_data(n_samples=1000)

        # Compute quality metrics
        signal_variance = np.var(data, axis=1)
        artifact_rate = self.artifacts_detected / max(self.samples_acquired, 1)

        # Check impedances
        impedances = self.get_channel_impedances()
        high_impedance_channels = [
            name for name, imp in impedances.items() if imp > 10.0
        ]  # > 10 kOhm

        return {
            "status": ("good" if artifact_rate < 0.05 and not high_impedance_channels else "poor"),
            "artifact_rate": artifact_rate,
            "mean_variance": float(np.mean(signal_variance)),
            "high_impedance_channels": high_impedance_channels,
            "samples_acquired": self.samples_acquired,
            "duration_seconds": time.time() - self.start_time if self.start_time else 0,
        }

    def clear_buffer(self) -> None:
        """Clear data buffers."""
        self.buffer.clear()
        self.timestamp_buffer.clear()

    def export_data(self, filename: str, format: str = "numpy") -> None:
        """Export buffered data to file.

        Args:
            filename: Output filename
            format: Export format (numpy, csv, edf)
        """
        data, timestamps = self.get_buffer_data()

        if format == "numpy":
            np.savez(
                filename,
                data=data,
                timestamps=timestamps,
                channels=self.active_channels,
                sampling_rate=self.config.sampling_rate,
            )

        elif format == "csv":
            # Simple CSV export
            import csv

            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp"] + self.active_channels)
                for i, ts in enumerate(timestamps):
                    row = [ts] + data[:, i].tolist()
                    writer.writerow(row)

        else:
            raise ValueError(f"Unsupported export format: {format}")
