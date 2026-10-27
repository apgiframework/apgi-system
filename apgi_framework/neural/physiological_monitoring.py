"""Physiological monitoring for multi-modal biosignals.

Provides heart rate, skin conductance, respiratory monitoring,
and synchronized physiological data streaming for APGI experiments.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..logging.standardized_logging import get_logger


class SignalType(Enum):
    """Physiological signal types."""

    ECG = "ecg"  # Electrocardiogram
    PPG = "ppg"  # Photoplethysmogram
    SCR = "scr"  # Skin conductance response
    RESP = "resp"  # Respiration
    TEMP = "temp"  # Temperature
    BP = "bp"  # Blood pressure


class RespirationPhase(Enum):
    """Respiration cycle phases."""

    INSPIRATION = "inspiration"
    EXPIRATION = "expiration"
    PAUSE = "pause"


@dataclass
class PhysiologicalSample:
    """Single multi-modal physiological measurement."""

    timestamp: float
    heart_rate: Optional[float] = None  # bpm
    rr_interval: Optional[float] = None  # ms
    scr_level: Optional[float] = None  # microsiemens
    scr_response: Optional[float] = None  # microsiemens
    respiration_rate: Optional[float] = None  # breaths per minute
    respiration_phase: Optional[RespirationPhase] = None
    respiration_amplitude: Optional[float] = None
    temperature: Optional[float] = None  # Celsius
    blood_pressure_systolic: Optional[float] = None  # mmHg
    blood_pressure_diastolic: Optional[float] = None  # mmHg
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhysiologicalConfig:
    """Configuration for physiological monitoring."""

    sampling_rate: float = 1000.0  # Hz
    buffer_size: int = 60000  # samples (60 seconds at 1000 Hz)

    # Signal acquisition settings
    enable_ecg: bool = True
    enable_scr: bool = True
    enable_respiration: bool = True
    enable_temperature: bool = False
    enable_blood_pressure: bool = False

    # Heart rate detection
    hr_min: float = 40.0  # bpm
    hr_max: float = 200.0  # bpm
    r_peak_threshold: float = 0.6  # relative to signal range

    # SCR detection
    scr_threshold: float = 0.01  # microsiemens
    scr_rise_time_min: float = 1.0  # seconds
    scr_rise_time_max: float = 3.0  # seconds

    # Respiration detection
    resp_min: float = 8.0  # breaths per minute
    resp_max: float = 30.0  # breaths per minute

    # Synchronization
    enable_synchronization: bool = True
    sync_tolerance: float = 0.01  # seconds

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")


class HeartRateMonitor:
    """Heart rate and HRV monitoring from ECG/PPG signals."""

    def __init__(self, config: PhysiologicalConfig):
        """Initialize heart rate monitor.

        Args:
            config: Physiological monitoring configuration
        """
        self.config = config
        self.r_peaks: List[float] = []
        self.rr_intervals: List[float] = []

    def detect_r_peaks(self, ecg_signal: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Detect R-peaks in ECG signal.

        Args:
            ecg_signal: ECG signal array
            timestamps: Timestamp array

        Returns:
            Array of R-peak timestamps
        """
        if len(ecg_signal) < 10:
            return np.array([])

        # Normalize signal
        signal_min = np.min(ecg_signal)
        signal_max = np.max(ecg_signal)
        signal_range = signal_max - signal_min

        if signal_range == 0:
            return np.array([])

        normalized = (ecg_signal - signal_min) / signal_range

        # Simple peak detection with threshold
        threshold = self.config.r_peak_threshold

        # Find peaks above threshold
        above_threshold = normalized > threshold

        # Find rising edges
        peaks = []
        in_peak = False
        peak_start = 0

        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_peak:
                # Start of peak
                in_peak = True
                peak_start = i
            elif not above_threshold[i] and in_peak:
                # End of peak - find maximum in this region
                peak_region = normalized[peak_start:i]
                if len(peak_region) > 0:
                    peak_idx = peak_start + np.argmax(peak_region)
                    peaks.append(timestamps[peak_idx])
                in_peak = False

        # Filter by physiological constraints
        if len(peaks) > 1:
            # Remove peaks that are too close together
            min_rr_interval = 60.0 / self.config.hr_max  # seconds
            filtered_peaks = [peaks[0]]

            for peak in peaks[1:]:
                if peak - filtered_peaks[-1] >= min_rr_interval:
                    filtered_peaks.append(peak)

            peaks = filtered_peaks

        self.r_peaks = peaks
        return np.array(peaks)

    def compute_rr_intervals(self, r_peaks: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute RR intervals from R-peaks.

        Args:
            r_peaks: Array of R-peak timestamps (None to use stored peaks)

        Returns:
            Array of RR intervals in milliseconds
        """
        if r_peaks is None:
            r_peaks = np.array(self.r_peaks)

        if len(r_peaks) < 2:
            return np.array([])

        # Compute intervals
        rr_intervals = np.diff(r_peaks) * 1000  # Convert to ms

        # Filter physiologically implausible intervals
        min_rr = 60000.0 / self.config.hr_max  # ms
        max_rr = 60000.0 / self.config.hr_min  # ms

        valid_mask = (rr_intervals >= min_rr) & (rr_intervals <= max_rr)
        rr_intervals = rr_intervals[valid_mask]

        self.rr_intervals = rr_intervals.tolist()
        return rr_intervals

    def compute_heart_rate(self, rr_intervals: Optional[np.ndarray] = None) -> float:
        """Compute instantaneous heart rate from RR intervals.

        Args:
            rr_intervals: Array of RR intervals in ms (None to use stored)

        Returns:
            Heart rate in bpm
        """
        if rr_intervals is None:
            rr_intervals = np.array(self.rr_intervals)

        if len(rr_intervals) == 0:
            return 0.0

        # Use most recent RR interval
        mean_rr = np.mean(rr_intervals[-5:])  # Average last 5 intervals

        if mean_rr == 0:
            return 0.0

        heart_rate = 60000.0 / mean_rr  # Convert ms to bpm

        return float(heart_rate)

    def compute_hrv_metrics(self, rr_intervals: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute heart rate variability metrics.

        Args:
            rr_intervals: Array of RR intervals in ms (None to use stored)

        Returns:
            Dictionary with HRV metrics
        """
        if rr_intervals is None:
            rr_intervals = np.array(self.rr_intervals)

        if len(rr_intervals) < 5:
            return {"sdnn": 0.0, "rmssd": 0.0, "pnn50": 0.0}

        # SDNN: Standard deviation of NN intervals
        sdnn = np.std(rr_intervals)

        # RMSSD: Root mean square of successive differences
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs**2))

        # pNN50: Percentage of successive differences > 50ms
        pnn50 = np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100

        return {"sdnn": sdnn, "rmssd": rmssd, "pnn50": pnn50}


class SkinConductanceMonitor:
    """Skin conductance response (SCR) monitoring."""

    def __init__(self, config: PhysiologicalConfig):
        """Initialize skin conductance monitor.

        Args:
            config: Physiological monitoring configuration
        """
        self.config = config
        self.scr_events: List[Dict[str, float]] = []

    def decompose_signal(
        self, scr_signal: np.ndarray, timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose SCR signal into tonic (SCL) and phasic (SCR) components.

        Args:
            scr_signal: Raw skin conductance signal
            timestamps: Timestamp array

        Returns:
            Tuple of (tonic_component, phasic_component)
        """
        if len(scr_signal) < 10:
            return np.zeros_like(scr_signal), np.zeros_like(scr_signal)

        # Simple low-pass filter for tonic component
        # Use moving average with window of ~5 seconds
        window_size = int(5.0 * self.config.sampling_rate)
        window_size = min(window_size, len(scr_signal) // 2)

        if window_size < 3:
            window_size = 3

        # Compute tonic component (slow-varying baseline)
        tonic = np.convolve(scr_signal, np.ones(window_size) / window_size, mode="same")

        # Phasic component is the difference
        phasic = scr_signal - tonic

        return tonic, phasic

    def detect_scr_events(
        self, phasic_signal: np.ndarray, timestamps: np.ndarray
    ) -> List[Dict[str, float]]:
        """Detect SCR events in phasic signal.

        Args:
            phasic_signal: Phasic SCR component
            timestamps: Timestamp array

        Returns:
            List of SCR event dictionaries
        """
        if len(phasic_signal) < 10:
            return []

        events = []

        # Find peaks in phasic signal
        threshold = self.config.scr_threshold

        # Simple peak detection
        above_threshold = phasic_signal > threshold

        in_event = False
        event_start_idx = 0

        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_event:
                # Start of event
                in_event = True
                event_start_idx = i
            elif not above_threshold[i] and in_event:
                # End of event
                event_region = phasic_signal[event_start_idx:i]

                if len(event_region) > 0:
                    # Find peak in event
                    peak_idx = event_start_idx + np.argmax(event_region)
                    peak_amplitude = phasic_signal[peak_idx]

                    # Compute rise time
                    rise_time = timestamps[peak_idx] - timestamps[event_start_idx]

                    # Check if rise time is physiologically plausible
                    if self.config.scr_rise_time_min <= rise_time <= self.config.scr_rise_time_max:
                        events.append(
                            {
                                "onset_time": timestamps[event_start_idx],
                                "peak_time": timestamps[peak_idx],
                                "amplitude": peak_amplitude,
                                "rise_time": rise_time,
                            }
                        )

                in_event = False

        self.scr_events = events
        return events

    def compute_scr_rate(
        self, events: Optional[List[Dict[str, float]]] = None, window: float = 60.0
    ) -> float:
        """Compute SCR event rate.

        Args:
            events: List of SCR events (None to use stored)
            window: Time window in seconds

        Returns:
            SCR rate (events per minute)
        """
        if events is None:
            events = self.scr_events

        if not events:
            return 0.0

        # Get events in recent window
        current_time = time.time()
        recent_events = [e for e in events if current_time - e["onset_time"] <= window]

        # Compute rate
        rate = len(recent_events) / window * 60.0

        return rate


class RespirationMonitor:
    """Respiratory monitoring for interoceptive tasks."""

    def __init__(self, config: PhysiologicalConfig):
        """Initialize respiration monitor.

        Args:
            config: Physiological monitoring configuration
        """
        self.config = config
        self.breath_cycles: List[Dict[str, Any]] = []

    def detect_breath_cycles(
        self, resp_signal: np.ndarray, timestamps: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect inspiration and expiration cycles.

        Args:
            resp_signal: Respiration signal (e.g., chest expansion)
            timestamps: Timestamp array

        Returns:
            List of breath cycle dictionaries
        """
        if len(resp_signal) < 10:
            return []

        # Find peaks (end of inspiration) and troughs (end of expiration)
        # Use simple derivative-based detection

        # Smooth signal
        window_size = int(0.5 * self.config.sampling_rate)
        window_size = max(3, min(window_size, len(resp_signal) // 4))

        smoothed = np.convolve(resp_signal, np.ones(window_size) / window_size, mode="same")

        # Find zero crossings of derivative
        derivative = np.diff(smoothed)
        derivative = np.pad(derivative, (0, 1), mode="edge")

        # Find peaks (inspiration peaks)
        peaks = []
        troughs = []

        for i in range(1, len(derivative) - 1):
            # Peak: derivative changes from positive to negative
            if derivative[i - 1] > 0 and derivative[i + 1] < 0:
                peaks.append(i)
            # Trough: derivative changes from negative to positive
            elif derivative[i - 1] < 0 and derivative[i + 1] > 0:
                troughs.append(i)

        # Match peaks and troughs to form breath cycles
        cycles = []

        for i in range(len(peaks) - 1):
            peak_idx = peaks[i]
            next_peak_idx = peaks[i + 1]

            # Find trough between peaks
            troughs_between = [t for t in troughs if peak_idx < t < next_peak_idx]

            if troughs_between:
                trough_idx = troughs_between[0]

                # Compute cycle parameters
                if i > 0 and troughs_between:
                    inspiration_time = timestamps[peak_idx] - timestamps[trough_idx]
                else:
                    inspiration_time = 0

                expiration_time = timestamps[trough_idx] - timestamps[peak_idx]
                cycle_time = timestamps[next_peak_idx] - timestamps[peak_idx]

                amplitude = smoothed[peak_idx] - smoothed[trough_idx]

                cycles.append(
                    {
                        "inspiration_start": (
                            timestamps[trough_idx] if i > 0 else timestamps[peak_idx]
                        ),
                        "inspiration_peak": timestamps[peak_idx],
                        "expiration_end": timestamps[trough_idx],
                        "inspiration_time": inspiration_time,
                        "expiration_time": expiration_time,
                        "cycle_duration": cycle_time,
                        "amplitude": amplitude,
                    }
                )

        self.breath_cycles = cycles
        return cycles

    def compute_respiration_rate(self, cycles: Optional[List[Dict[str, Any]]] = None) -> float:
        """Compute respiration rate from breath cycles.

        Args:
            cycles: List of breath cycles (None to use stored)

        Returns:
            Respiration rate in breaths per minute
        """
        if cycles is None:
            cycles = self.breath_cycles

        if len(cycles) < 2:
            return 0.0

        # Use recent cycles
        recent_cycles = cycles[-5:]

        # Compute mean cycle duration
        durations = [c["cycle_duration"] for c in recent_cycles]
        mean_duration = np.mean(durations)

        if mean_duration == 0:
            return 0.0

        # Convert to breaths per minute
        resp_rate = 60.0 / mean_duration

        # Clamp to physiological range
        resp_rate = np.clip(resp_rate, self.config.resp_min, self.config.resp_max)

        return float(resp_rate)

    def get_current_phase(
        self, resp_signal: np.ndarray, timestamps: np.ndarray, current_time: float
    ) -> RespirationPhase:
        """Determine current respiration phase.

        Args:
            resp_signal: Respiration signal
            timestamps: Timestamp array
            current_time: Current timestamp

        Returns:
            Current respiration phase
        """
        if len(resp_signal) < 10:
            return RespirationPhase.PAUSE

        # Find most recent samples
        recent_mask = timestamps >= (current_time - 1.0)

        if not np.any(recent_mask):
            return RespirationPhase.PAUSE

        recent_signal = resp_signal[recent_mask]

        # Compute derivative to determine phase
        if len(recent_signal) < 2:
            return RespirationPhase.PAUSE

        derivative = np.diff(recent_signal)
        mean_derivative = np.mean(derivative)

        # Threshold for determining phase
        threshold = 0.01 * (np.max(resp_signal) - np.min(resp_signal))

        if mean_derivative > threshold:
            return RespirationPhase.INSPIRATION
        elif mean_derivative < -threshold:
            return RespirationPhase.EXPIRATION
        else:
            return RespirationPhase.PAUSE


class PhysiologicalMonitoring:
    """Multi-modal physiological monitoring system.

    Integrates heart rate, skin conductance, respiration, and other biosignals
    with synchronized data streaming for APGI experiments.
    """

    def __init__(self, config: PhysiologicalConfig):
        """Initialize physiological monitoring system.

        Args:
            config: Physiological monitoring configuration
        """
        self.config = config

        # Setup logging
        self.logger = get_logger(__name__)

        # Initialize component monitors
        self.heart_rate_monitor = HeartRateMonitor(config)
        self.scr_monitor = SkinConductanceMonitor(config)
        self.respiration_monitor = RespirationMonitor(config)

        # Data buffers for each signal type
        self.ecg_buffer: deque = deque(maxlen=config.buffer_size)
        self.scr_buffer: deque = deque(maxlen=config.buffer_size)
        self.resp_buffer: deque = deque(maxlen=config.buffer_size)
        self.temp_buffer: deque = deque(maxlen=config.buffer_size)
        self.bp_buffer: deque = deque(maxlen=config.buffer_size)

        # Integrated sample buffer
        self.sample_buffer: deque = deque(maxlen=config.buffer_size)

        # Streaming state
        self.is_streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self.data_callbacks: List[Callable] = []

        # Synchronization
        self.sync_lock = threading.Lock()
        self.last_sync_time: Optional[float] = None

        # Statistics
        self.samples_acquired = 0
        self.start_time: Optional[float] = None

    def start_streaming(self, data_sources: Optional[Dict[SignalType, Callable]] = None) -> None:
        """Start real-time multi-modal physiological data streaming.

        Args:
            data_sources: Dictionary mapping signal types to data source callbacks
                         Each callback should return (timestamp, value) tuple
                         If None, generates simulated data for testing
        """
        if self.is_streaming:
            raise RuntimeError("Streaming already active")

        self.is_streaming = True
        self.start_time = time.time()

        def stream_loop() -> None:
            """Internal streaming loop."""
            while self.is_streaming:
                current_time = time.time()

                # Initialize sample
                sample = PhysiologicalSample(timestamp=current_time)

                # Acquire ECG/heart rate data
                if self.config.enable_ecg:
                    if data_sources and SignalType.ECG in data_sources:
                        ecg_time, ecg_value = data_sources[SignalType.ECG]()
                    else:
                        # Simulate ECG data
                        ecg_time = current_time
                        ecg_value = np.sin(2 * np.pi * 1.2 * current_time) + np.random.randn() * 0.1

                    self.ecg_buffer.append((ecg_time, ecg_value))

                    # Update heart rate if enough data
                    if len(self.ecg_buffer) >= 100:
                        ecg_data = np.array([v for _, v in self.ecg_buffer])
                        timestamps = np.array([t for t, _ in self.ecg_buffer])

                        r_peaks = self.heart_rate_monitor.detect_r_peaks(ecg_data, timestamps)
                        if len(r_peaks) > 0:
                            rr_intervals = self.heart_rate_monitor.compute_rr_intervals(r_peaks)
                            if len(rr_intervals) > 0:
                                sample.heart_rate = self.heart_rate_monitor.compute_heart_rate(
                                    rr_intervals
                                )
                                sample.rr_interval = (
                                    rr_intervals[-1] if len(rr_intervals) > 0 else None
                                )

                # Acquire SCR data
                if self.config.enable_scr:
                    if data_sources and SignalType.SCR in data_sources:
                        scr_time, scr_value = data_sources[SignalType.SCR]()
                    else:
                        # Simulate SCR data
                        scr_time = current_time
                        scr_value = 5.0 + np.random.randn() * 0.5 + 0.1 * np.sin(0.1 * current_time)

                    self.scr_buffer.append((scr_time, scr_value))

                    # Update SCR metrics if enough data
                    if len(self.scr_buffer) >= 100:
                        scr_data = np.array([v for _, v in self.scr_buffer])
                        timestamps = np.array([t for t, _ in self.scr_buffer])

                        tonic, phasic = self.scr_monitor.decompose_signal(scr_data, timestamps)
                        sample.scr_level = tonic[-1] if len(tonic) > 0 else None
                        sample.scr_response = phasic[-1] if len(phasic) > 0 else None

                # Acquire respiration data
                if self.config.enable_respiration:
                    if data_sources and SignalType.RESP in data_sources:
                        resp_time, resp_value = data_sources[SignalType.RESP]()
                    else:
                        # Simulate respiration data
                        resp_time = current_time
                        resp_value = (
                            np.sin(2 * np.pi * 0.25 * current_time) + np.random.randn() * 0.05
                        )

                    self.resp_buffer.append((resp_time, resp_value))

                    # Update respiration metrics if enough data
                    if len(self.resp_buffer) >= 100:
                        resp_data = np.array([v for _, v in self.resp_buffer])
                        timestamps = np.array([t for t, _ in self.resp_buffer])

                        cycles = self.respiration_monitor.detect_breath_cycles(
                            resp_data, timestamps
                        )
                        if cycles:
                            sample.respiration_rate = (
                                self.respiration_monitor.compute_respiration_rate(cycles)
                            )
                            sample.respiration_phase = self.respiration_monitor.get_current_phase(
                                resp_data, timestamps, current_time
                            )
                            sample.respiration_amplitude = (
                                cycles[-1]["amplitude"] if cycles else None
                            )

                # Acquire temperature data
                if self.config.enable_temperature:
                    if data_sources and SignalType.TEMP in data_sources:
                        temp_time, temp_value = data_sources[SignalType.TEMP]()
                    else:
                        # Simulate temperature data
                        temp_time = current_time
                        temp_value = 36.5 + np.random.randn() * 0.1

                    self.temp_buffer.append((temp_time, temp_value))
                    sample.temperature = temp_value

                # Acquire blood pressure data
                if self.config.enable_blood_pressure:
                    if data_sources and SignalType.BP in data_sources:
                        bp_time, bp_value = data_sources[SignalType.BP]()
                        sample.blood_pressure_systolic = (
                            bp_value[0] if isinstance(bp_value, tuple) else bp_value
                        )
                        sample.blood_pressure_diastolic = (
                            bp_value[1]
                            if isinstance(bp_value, tuple) and len(bp_value) > 1
                            else None
                        )
                    else:
                        # Simulate blood pressure data
                        sample.blood_pressure_systolic = 120 + np.random.randn() * 5
                        sample.blood_pressure_diastolic = 80 + np.random.randn() * 3

                    self.bp_buffer.append(
                        (
                            current_time,
                            (
                                sample.blood_pressure_systolic,
                                sample.blood_pressure_diastolic,
                            ),
                        )
                    )

                # Store integrated sample
                with self.sync_lock:
                    self.sample_buffer.append(sample)
                    self.samples_acquired += 1
                    self.last_sync_time = current_time

                # Notify callbacks
                for callback in self.data_callbacks:
                    try:
                        callback(sample)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")

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
            callback: Function with signature (sample: PhysiologicalSample) -> None
        """
        self.data_callbacks.append(callback)

    def get_buffer_data(self, n_samples: Optional[int] = None) -> List[PhysiologicalSample]:
        """Retrieve samples from buffer.

        Args:
            n_samples: Number of recent samples to retrieve (None for all)

        Returns:
            List of physiological samples
        """
        with self.sync_lock:
            samples = list(self.sample_buffer)

        if n_samples:
            samples = samples[-n_samples:]

        return samples

    def get_signal_data(
        self, signal_type: SignalType, n_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract specific signal type data.

        Args:
            signal_type: Type of signal to extract
            n_samples: Number of recent samples (None for all)

        Returns:
            Tuple of (values, timestamps)
        """
        if signal_type == SignalType.ECG:
            buffer = self.ecg_buffer
        elif signal_type == SignalType.SCR:
            buffer = self.scr_buffer
        elif signal_type == SignalType.RESP:
            buffer = self.resp_buffer
        elif signal_type == SignalType.TEMP:
            buffer = self.temp_buffer
        elif signal_type == SignalType.BP:
            buffer = self.bp_buffer
        else:
            return np.array([]), np.array([])

        with self.sync_lock:
            data = list(buffer)

        if n_samples:
            data = data[-n_samples:]

        if not data:
            return np.array([]), np.array([])

        timestamps = np.array([t for t, _ in data])
        values = np.array([v for _, v in data])

        return values, timestamps

    def compute_comprehensive_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive physiological metrics from buffered data.

        Returns:
            Dictionary with all computed metrics
        """
        metrics = {
            "timestamp": time.time(),
            "duration_seconds": time.time() - self.start_time if self.start_time else 0,
            "samples_acquired": self.samples_acquired,
        }

        # Heart rate metrics
        if self.config.enable_ecg and len(self.heart_rate_monitor.rr_intervals) > 0:
            rr_intervals = np.array(self.heart_rate_monitor.rr_intervals)
            metrics["heart_rate"] = self.heart_rate_monitor.compute_heart_rate(rr_intervals)
            metrics["hrv"] = self.heart_rate_monitor.compute_hrv_metrics(rr_intervals).get(
                "rmssd", 0.0
            )

        # SCR metrics
        if self.config.enable_scr and len(self.scr_buffer) > 0:
            scr_data, timestamps = self.get_signal_data(SignalType.SCR)
            tonic, phasic = self.scr_monitor.decompose_signal(scr_data, timestamps)
            events = self.scr_monitor.detect_scr_events(phasic, timestamps)

            metrics["scr_level"] = np.mean(tonic) if len(tonic) > 0 else 0.0
            metrics["scr_events"] = len(events)
            metrics["scr_rate"] = self.scr_monitor.compute_scr_rate(events)

        # Respiration metrics
        if self.config.enable_respiration and len(self.resp_buffer) > 0:
            resp_data, timestamps = self.get_signal_data(SignalType.RESP)
            cycles = self.respiration_monitor.detect_breath_cycles(resp_data, timestamps)

            if cycles:
                metrics["respiration_rate"] = self.respiration_monitor.compute_respiration_rate(
                    cycles
                )
                metrics["breath_cycles"] = len(cycles)
                phase_enum = self.respiration_monitor.get_current_phase(
                    resp_data, timestamps, time.time()
                )
                # Convert enum to integer for metrics compatibility
                phase_mapping = {
                    RespirationPhase.INSPIRATION: 0,
                    RespirationPhase.EXPIRATION: 1,
                    RespirationPhase.PAUSE: 2,
                }
                metrics["current_phase"] = phase_mapping.get(phase_enum, 0)

        return metrics

    def synchronize_with_external(
        self, external_timestamp: float, tolerance: Optional[float] = None
    ) -> Optional[PhysiologicalSample]:
        """Find physiological sample synchronized with external timestamp.

        Args:
            external_timestamp: External event timestamp to synchronize with
            tolerance: Maximum time difference for synchronization (uses config default if None)

        Returns:
            Synchronized physiological sample or None if not found
        """
        if tolerance is None:
            tolerance = self.config.sync_tolerance

        samples = self.get_buffer_data()

        if not samples:
            return None

        # Find closest sample within tolerance
        closest_sample = None
        min_diff = float("inf")

        for sample in samples:
            diff = abs(sample.timestamp - external_timestamp)
            if diff < min_diff and diff <= tolerance:
                min_diff = diff
                closest_sample = sample

        return closest_sample

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Assess physiological data quality.

        Returns:
            Dictionary with quality metrics
        """
        if not self.sample_buffer:
            return {"status": "no_data"}

        samples = self.get_buffer_data()

        # Count valid measurements
        valid_hr = sum(1 for s in samples if s.heart_rate is not None)
        valid_scr = sum(1 for s in samples if s.scr_level is not None)
        valid_resp = sum(1 for s in samples if s.respiration_rate is not None)

        total_samples = len(samples)

        # Compute quality scores
        hr_quality = valid_hr / total_samples if total_samples > 0 else 0
        scr_quality = valid_scr / total_samples if total_samples > 0 else 0
        resp_quality = valid_resp / total_samples if total_samples > 0 else 0

        overall_quality = np.mean([hr_quality, scr_quality, resp_quality])

        return {
            "status": "good" if overall_quality > 0.8 else "poor",
            "overall_quality": overall_quality,
            "heart_rate_quality": hr_quality,
            "scr_quality": scr_quality,
            "respiration_quality": resp_quality,
            "samples_acquired": self.samples_acquired,
            "duration_seconds": time.time() - self.start_time if self.start_time else 0,
        }

    def clear_buffers(self) -> None:
        """Clear all data buffers."""
        with self.sync_lock:
            self.ecg_buffer.clear()
            self.scr_buffer.clear()
            self.resp_buffer.clear()
            self.temp_buffer.clear()
            self.bp_buffer.clear()
            self.sample_buffer.clear()

    def export_data(self, filename: str, format: str = "numpy") -> None:
        """Export physiological data to file.

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
                "heart_rate": np.array(
                    [s.heart_rate if s.heart_rate is not None else np.nan for s in samples]
                ),
                "rr_interval": np.array(
                    [s.rr_interval if s.rr_interval is not None else np.nan for s in samples]
                ),
                "scr_level": np.array(
                    [s.scr_level if s.scr_level is not None else np.nan for s in samples]
                ),
                "scr_response": np.array(
                    [s.scr_response if s.scr_response is not None else np.nan for s in samples]
                ),
                "respiration_rate": np.array(
                    [
                        s.respiration_rate if s.respiration_rate is not None else np.nan
                        for s in samples
                    ]
                ),
                "respiration_amplitude": np.array(
                    [
                        (s.respiration_amplitude if s.respiration_amplitude is not None else np.nan)
                        for s in samples
                    ]
                ),
                "temperature": np.array(
                    [s.temperature if s.temperature is not None else np.nan for s in samples]
                ),
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
                        "heart_rate",
                        "rr_interval",
                        "scr_level",
                        "scr_response",
                        "respiration_rate",
                        "respiration_phase",
                        "respiration_amplitude",
                        "temperature",
                        "bp_systolic",
                        "bp_diastolic",
                    ]
                )
                for s in samples:
                    writer.writerow(
                        [
                            s.timestamp,
                            s.heart_rate if s.heart_rate is not None else "",
                            s.rr_interval if s.rr_interval is not None else "",
                            s.scr_level if s.scr_level is not None else "",
                            s.scr_response if s.scr_response is not None else "",
                            (s.respiration_rate if s.respiration_rate is not None else ""),
                            (s.respiration_phase.value if s.respiration_phase is not None else ""),
                            (
                                s.respiration_amplitude
                                if s.respiration_amplitude is not None
                                else ""
                            ),
                            s.temperature if s.temperature is not None else "",
                            (
                                s.blood_pressure_systolic
                                if s.blood_pressure_systolic is not None
                                else ""
                            ),
                            (
                                s.blood_pressure_diastolic
                                if s.blood_pressure_diastolic is not None
                                else ""
                            ),
                        ]
                    )

        else:
            raise ValueError(f"Unsupported export format: {format}")
