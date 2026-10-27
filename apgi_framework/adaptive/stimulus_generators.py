"""Stimulus generators for parameter estimation tasks.

Implements stimulus generation and control for visual, auditory, and interoceptive
stimuli used in the three core parameter estimation tasks.
"""

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

import numpy as np

T = TypeVar("T", bound="StimulusParameters")

from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)


class StimulusType(Enum):
    """Types of stimuli for parameter estimation tasks."""

    GABOR_PATCH = "gabor_patch"
    TONE = "tone"
    CO2_PUFF = "co2_puff"
    HEARTBEAT_FLASH = "heartbeat_flash"


@dataclass
class StimulusParameters:
    """Base parameters for stimulus generation."""

    stimulus_type: StimulusType
    intensity: float = 0.5
    duration_ms: float = 500.0
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate stimulus parameters."""
        return 0.0 <= self.intensity <= 1.0 and self.duration_ms > 0


@dataclass
class GaborParameters(StimulusParameters):
    """Parameters for Gabor patch stimuli."""

    stimulus_type: StimulusType = field(default=StimulusType.GABOR_PATCH, init=False)

    # Visual properties
    contrast: float = 0.5  # 0-1 scale
    spatial_frequency: float = 2.0  # cycles per degree
    orientation: float = 0.0  # degrees
    phase: float = 0.0  # radians
    size_degrees: float = 2.0  # visual angle

    # Presentation properties
    background_luminance: float = 0.5  # 0-1 scale
    position_x: float = 0.0  # screen coordinates
    position_y: float = 0.0

    def validate(self) -> bool:
        """Validate Gabor parameters."""
        return (
            super().validate()
            and 0.0 <= self.contrast <= 1.0
            and self.spatial_frequency > 0
            and 0.0 <= self.orientation < 360.0
            and self.size_degrees > 0
        )


@dataclass
class ToneParameters(StimulusParameters):
    """Parameters for auditory tone stimuli."""

    stimulus_type: StimulusType = field(default=StimulusType.TONE, init=False)

    # Auditory properties
    frequency_hz: float = 1000.0
    amplitude: float = 0.5  # 0-1 scale
    envelope: str = "linear"  # linear, exponential, gaussian

    # Timing properties
    onset_ramp_ms: float = 10.0
    offset_ramp_ms: float = 10.0

    # Synchronization
    sync_to_heartbeat: bool = False
    heartbeat_delay_ms: float = 0.0

    def validate(self) -> bool:
        """Validate tone parameters."""
        return (
            super().validate()
            and 20.0 <= self.frequency_hz <= 20000.0
            and 0.0 <= self.amplitude <= 1.0
            and self.onset_ramp_ms >= 0
            and self.offset_ramp_ms >= 0
        )


@dataclass
class CO2PuffParameters(StimulusParameters):
    """Parameters for CO2 puff stimuli."""

    stimulus_type: StimulusType = field(default=StimulusType.CO2_PUFF, init=False)

    # Gas properties
    co2_concentration: float = 10.0  # percentage
    flow_rate: float = 2.0  # L/min
    pressure: float = 1.0  # bar

    # Safety parameters
    max_duration_ms: float = 500.0
    min_interval_ms: float = 5000.0  # Minimum time between puffs
    safety_interlock: bool = True

    # Delivery properties
    nozzle_distance_cm: float = 5.0
    target_area: str = "nostril"  # nostril, face, hand

    def validate(self) -> bool:
        """Validate CO2 puff parameters."""
        return (
            super().validate()
            and 0.0 <= self.co2_concentration <= 100.0
            and self.flow_rate > 0
            and self.pressure > 0
            and self.duration_ms <= self.max_duration_ms
            and self.nozzle_distance_cm > 0
        )


class StimulusGenerator(ABC, Generic[T]):
    """Abstract base class for stimulus generators."""

    def __init__(self, generator_id: str = ""):
        self.generator_id = generator_id
        self.is_initialized = False
        self.last_stimulus_time: Optional[datetime] = None

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the stimulus generator."""

    @abstractmethod
    def generate_stimulus(self, parameters: T) -> bool:
        """Generate and present stimulus."""

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""

    def validate_timing(self, min_interval_ms: float) -> bool:
        """Validate minimum interval between stimuli."""
        if self.last_stimulus_time is None:
            return True

        elapsed = (datetime.now() - self.last_stimulus_time).total_seconds() * 1000
        return elapsed >= min_interval_ms


class GaborPatchGenerator(StimulusGenerator[GaborParameters]):
    """Generator for Gabor patch visual stimuli.

    Creates Gabor patches with precise intensity control for visual detection tasks.
    """

    def __init__(
        self,
        generator_id: str = "gabor_generator",
        screen_width: int = 1920,
        screen_height: int = 1080,
        viewing_distance_cm: float = 60.0,
    ):
        """Initialize Gabor patch generator.

        Args:
            generator_id: Unique identifier for this generator
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            viewing_distance_cm: Viewing distance in centimeters
        """
        super().__init__(generator_id)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.viewing_distance_cm = viewing_distance_cm

        # Display properties
        self.pixels_per_degree = None
        self.display_initialized = False

        logger.info(f"Initialized GaborPatchGenerator {generator_id}")

    def initialize(self) -> bool:
        """Initialize display and calculate visual parameters."""
        try:
            # Calculate pixels per degree of visual angle
            # Assuming standard monitor size (this should be calibrated)
            screen_width_cm = 50.0  # Typical 24" monitor width
            degrees_per_pixel = (
                np.degrees(np.arctan(screen_width_cm / self.viewing_distance_cm))
                / self.screen_width
            )
            self.pixels_per_degree = 1.0 / degrees_per_pixel

            self.display_initialized = True
            self.is_initialized = True

            logger.info(f"Display initialized: {self.pixels_per_degree:.2f} pixels per degree")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize display: {e}")
            return False

    def _create_gabor_patch(self, parameters: GaborParameters) -> np.ndarray:
        """Create Gabor patch stimulus array."""
        # Convert size from degrees to pixels
        size_degrees = parameters.size_degrees or 2.0  # Default size if None
        ppd: Optional[float] = self.pixels_per_degree
        size_pixels = 100 if ppd is None else int(size_degrees * ppd)

        # Create coordinate grids
        x = np.linspace(-size_degrees / 2, size_degrees / 2, size_pixels)
        y = np.linspace(-size_degrees / 2, size_degrees / 2, size_pixels)
        X, Y = np.meshgrid(x, y)

        # Rotate coordinates
        theta = np.radians(parameters.orientation)
        X_rot = X * np.cos(theta) + Y * np.sin(theta)
        Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

        # Create Gaussian envelope
        sigma = size_degrees / 6.0  # Standard deviation
        gaussian = np.exp(-(X_rot**2 + Y_rot**2) / (2 * sigma**2))

        # Create sinusoidal grating
        grating = np.cos(2 * np.pi * parameters.spatial_frequency * X_rot + parameters.phase)

        # Combine to create Gabor patch
        gabor = parameters.contrast * gaussian * grating

        # Add background luminance
        gabor = parameters.background_luminance + gabor

        # Clip to valid range
        gabor_clipped: np.ndarray = np.clip(gabor, 0.0, 1.0)
        return gabor_clipped

    def generate_stimulus(self, parameters: GaborParameters) -> bool:
        """Generate and present Gabor patch stimulus.

        Args:
            parameters: Gabor patch parameters

        Returns:
            True if stimulus was successfully presented
        """
        if not self.is_initialized:
            logger.error("Generator not initialized")
            return False

        if not parameters.validate():
            logger.error("Invalid Gabor parameters")
            return False

        try:
            # Create Gabor patch
            gabor_array = self._create_gabor_patch(parameters)

            # Record timing
            stimulus_start = datetime.now()
            parameters.timestamp = stimulus_start

            # Present stimulus (placeholder - would interface with actual display)
            logger.debug(
                f"Presenting Gabor patch: contrast={parameters.contrast:.3f}, "
                f"orientation={parameters.orientation:.1f}°, "
                f"duration={parameters.duration_ms:.1f}ms"
            )

            # Simulate presentation timing
            time.sleep(parameters.duration_ms / 1000.0)

            self.last_stimulus_time = stimulus_start

            # Store stimulus data in metadata
            parameters.metadata.update(
                {
                    "gabor_array_shape": gabor_array.shape,
                    "pixels_per_degree": self.pixels_per_degree,
                    "presentation_time": stimulus_start.isoformat(),
                }
            )

            return True

        except Exception as e:
            logger.error(f"Failed to generate Gabor stimulus: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up display resources."""
        self.display_initialized = False
        self.is_initialized = False
        logger.info("GaborPatchGenerator cleaned up")


class ToneGenerator(StimulusGenerator[ToneParameters]):
    """Generator for auditory tone stimuli.

    Creates pure tones with precise timing for auditory detection and
    heartbeat synchronization tasks.
    """

    def __init__(
        self,
        generator_id: str = "tone_generator",
        sample_rate: int = 44100,
        buffer_size: int = 1024,
    ):
        """Initialize tone generator.

        Args:
            generator_id: Unique identifier for this generator
            sample_rate: Audio sample rate in Hz
            buffer_size: Audio buffer size in samples
        """
        super().__init__(generator_id)
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        # Audio system
        self.audio_initialized = False
        self.heartbeat_synchronizer: Optional["HeartbeatSynchronizer"] = None

        logger.info(f"Initialized ToneGenerator {generator_id}")

    def initialize(self) -> bool:
        """Initialize audio system."""
        try:
            # Initialize audio system (placeholder - would use actual audio library)
            self.audio_initialized = True
            self.is_initialized = True

            logger.info(
                f"Audio system initialized: {self.sample_rate} Hz, buffer {self.buffer_size}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize audio system: {e}")
            return False

    def set_heartbeat_synchronizer(self, synchronizer: "HeartbeatSynchronizer") -> None:
        """Set heartbeat synchronizer for cardiac-locked tones."""
        self.heartbeat_synchronizer = synchronizer
        logger.info("Heartbeat synchronizer connected to ToneGenerator")

    def _create_tone(self, parameters: ToneParameters) -> np.ndarray:
        """Create tone audio signal."""
        # Calculate number of samples
        duration_samples = int(parameters.duration_ms * self.sample_rate / 1000.0)

        # Create time array
        t = np.linspace(0, parameters.duration_ms / 1000.0, duration_samples)

        # Generate pure tone
        tone = parameters.amplitude * np.sin(2 * np.pi * parameters.frequency_hz * t)

        # Apply envelope
        if parameters.envelope == "linear":
            # Linear onset/offset ramps
            onset_samples = int(parameters.onset_ramp_ms * self.sample_rate / 1000.0)
            offset_samples = int(parameters.offset_ramp_ms * self.sample_rate / 1000.0)

            # Onset ramp
            if onset_samples > 0:
                ramp_up = np.linspace(0, 1, onset_samples)
                tone[:onset_samples] *= ramp_up

            # Offset ramp
            if offset_samples > 0:
                ramp_down = np.linspace(1, 0, offset_samples)
                tone[-offset_samples:] *= ramp_down

        elif parameters.envelope == "gaussian":
            # Gaussian envelope
            sigma = duration_samples / 6.0
            center = duration_samples / 2.0
            envelope = np.exp(-((np.arange(duration_samples) - center) ** 2) / (2 * sigma**2))
            tone *= envelope

        return tone

    def generate_stimulus(self, parameters: ToneParameters) -> bool:
        """Generate and present auditory tone stimulus.

        Args:
            parameters: Tone parameters

        Returns:
            True if stimulus was successfully presented
        """
        if not self.is_initialized:
            logger.error("Generator not initialized")
            return False

        if not parameters.validate():
            logger.error("Invalid tone parameters")
            return False

        try:
            # Handle heartbeat synchronization
            if parameters.sync_to_heartbeat:
                if self.heartbeat_synchronizer is None:
                    logger.error("Heartbeat synchronizer not available")
                    return False

                # Wait for next R-peak and apply delay
                r_peak_time = self.heartbeat_synchronizer.wait_for_r_peak()
                if r_peak_time is None:
                    logger.error("Failed to detect R-peak for synchronization")
                    return False

                # Calculate presentation time
                presentation_time = r_peak_time + timedelta(
                    milliseconds=parameters.heartbeat_delay_ms
                )

                # Wait until presentation time
                wait_time = (presentation_time - datetime.now()).total_seconds()
                if wait_time > 0:
                    time.sleep(wait_time)

            # Create tone
            tone_array = self._create_tone(parameters)

            # Record timing
            stimulus_start = datetime.now()
            parameters.timestamp = stimulus_start

            # Present tone (placeholder - would interface with actual audio system)
            logger.debug(
                f"Presenting tone: {parameters.frequency_hz:.1f} Hz, "
                f"amplitude={parameters.amplitude:.3f}, "
                f"duration={parameters.duration_ms:.1f}ms"
            )

            # Simulate presentation timing
            time.sleep(parameters.duration_ms / 1000.0)

            self.last_stimulus_time = stimulus_start

            # Store stimulus data in metadata
            parameters.metadata.update(
                {
                    "tone_array_length": len(tone_array),
                    "sample_rate": self.sample_rate,
                    "presentation_time": stimulus_start.isoformat(),
                    "synchronized_to_heartbeat": parameters.sync_to_heartbeat,
                }
            )

            return True

        except Exception as e:
            logger.error(f"Failed to generate tone stimulus: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up audio resources."""
        self.audio_initialized = False
        self.is_initialized = False
        logger.info("ToneGenerator cleaned up")


class CO2PuffController(StimulusGenerator[CO2PuffParameters]):
    """Controller for CO2 puff interoceptive stimuli.

    Manages CO2 delivery with safety interlocks for interoceptive oddball tasks.
    """

    def __init__(self, generator_id: str = "co2_controller", safety_enabled: bool = True):
        """Initialize CO2 puff controller.

        Args:
            generator_id: Unique identifier for this controller
            safety_enabled: Enable safety interlocks
        """
        super().__init__(generator_id)
        self.safety_enabled = safety_enabled

        # Hardware interface
        self.valve_controller: Optional[str] = None
        self.pressure_sensor: Optional[str] = None
        self.flow_meter: Optional[str] = None

        # Safety tracking
        self.last_puff_time: Optional[datetime] = None
        self.total_puffs_session = 0
        self.max_puffs_per_session = 100
        self.emergency_stop = False

        logger.info(f"Initialized CO2PuffController {generator_id}")

    def initialize(self) -> bool:
        """Initialize CO2 delivery system and safety checks."""
        try:
            # Initialize hardware interfaces (placeholder)
            self.valve_controller = "mock_valve"  # Would be actual hardware interface
            self.pressure_sensor = "mock_pressure"
            self.flow_meter = "mock_flow"

            # Perform safety checks
            if self.safety_enabled:
                if not self._perform_safety_checks():
                    logger.error("Safety checks failed")
                    return False

            self.is_initialized = True
            logger.info("CO2 delivery system initialized with safety interlocks")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize CO2 system: {e}")
            return False

    def _perform_safety_checks(self) -> bool:
        """Perform comprehensive safety checks."""
        checks = {
            "pressure_within_limits": True,  # Would check actual pressure
            "flow_rate_calibrated": True,  # Would check flow calibration
            "valve_responsive": True,  # Would test valve operation
            "emergency_stop_functional": True,  # Would test emergency stop
            "concentration_accurate": True,  # Would verify CO2 concentration
        }

        all_passed = all(checks.values())

        if all_passed:
            logger.info("All safety checks passed")
        else:
            failed_checks = [check for check, passed in checks.items() if not passed]
            logger.error(f"Safety checks failed: {failed_checks}")

        return all_passed

    def _check_safety_constraints(self, parameters: CO2PuffParameters) -> bool:
        """Check safety constraints before puff delivery."""
        if self.emergency_stop:
            logger.error("Emergency stop activated")
            return False

        # Check session limits
        if self.total_puffs_session >= self.max_puffs_per_session:
            logger.error(f"Session puff limit reached: {self.max_puffs_per_session}")
            return False

        # Check minimum interval
        if self.last_puff_time is not None:
            elapsed_ms = (datetime.now() - self.last_puff_time).total_seconds() * 1000
            if elapsed_ms < parameters.min_interval_ms:
                logger.error(
                    f"Minimum interval not met: {elapsed_ms:.1f}ms < {parameters.min_interval_ms:.1f}ms"
                )
                return False

        # Check duration limits
        if parameters.duration_ms > parameters.max_duration_ms:
            logger.error(
                f"Duration exceeds safety limit: {parameters.duration_ms:.1f}ms > {parameters.max_duration_ms:.1f}ms"
            )
            return False

        # Check concentration limits
        if parameters.co2_concentration > 15.0:  # Safety limit
            logger.error(f"CO2 concentration too high: {parameters.co2_concentration:.1f}% > 15.0%")
            return False

        return True

    def generate_stimulus(self, parameters: CO2PuffParameters) -> bool:
        """Generate CO2 puff stimulus with safety monitoring.

        Args:
            parameters: CO2 puff parameters

        Returns:
            True if puff was successfully delivered
        """
        if not self.is_initialized:
            logger.error("Controller not initialized")
            return False

        if not parameters.validate():
            logger.error("Invalid CO2 puff parameters")
            return False

        if self.safety_enabled and not self._check_safety_constraints(parameters):
            logger.error("Safety constraints not met")
            return False

        try:
            # Record timing
            stimulus_start = datetime.now()
            parameters.timestamp = stimulus_start

            # Open valve for specified duration
            logger.debug(
                f"Delivering CO2 puff: {parameters.co2_concentration:.1f}% CO2, "
                f"flow={parameters.flow_rate:.1f} L/min, "
                f"duration={parameters.duration_ms:.1f}ms"
            )

            # Simulate puff delivery (would control actual hardware)
            time.sleep(parameters.duration_ms / 1000.0)

            # Update tracking
            self.last_puff_time = stimulus_start
            self.total_puffs_session += 1
            self.last_stimulus_time = stimulus_start

            # Store delivery data in metadata
            parameters.metadata.update(
                {
                    "delivery_time": stimulus_start.isoformat(),
                    "session_puff_number": self.total_puffs_session,
                    "safety_checks_passed": True,
                    "actual_pressure": 1.0,  # Would read from sensor
                    "actual_flow_rate": parameters.flow_rate,  # Would read from flow meter
                }
            )

            logger.info(
                f"CO2 puff delivered successfully (session puff #{self.total_puffs_session})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to deliver CO2 puff: {e}")
            # Emergency valve closure would happen here
            return False

    def emergency_stop_puff(self) -> None:
        """Emergency stop for CO2 delivery."""
        self.emergency_stop = True
        # Would immediately close all valves
        logger.warning("EMERGENCY STOP: CO2 delivery halted")

    def reset_emergency_stop(self) -> None:
        """Reset emergency stop after safety verification."""
        if self.safety_enabled and self._perform_safety_checks():
            self.emergency_stop = False
            logger.info("Emergency stop reset - system ready")
        else:
            logger.error("Cannot reset emergency stop - safety checks failed")

    def cleanup(self) -> None:
        """Clean up CO2 system and ensure safe shutdown."""
        # Close all valves
        self.emergency_stop = True

        # Reset session counters
        self.total_puffs_session = 0
        self.last_puff_time = None

        self.is_initialized = False
        logger.info("CO2PuffController cleaned up safely")


class HeartbeatSynchronizer:
    """Synchronizes stimulus presentation to cardiac cycle.

    Provides real-time R-peak detection and timing for heartbeat-locked
    stimulus presentation.
    """

    def __init__(
        self,
        synchronizer_id: str = "heartbeat_sync",
        detection_method: str = "pan_tompkins",
    ):
        """Initialize heartbeat synchronizer.

        Args:
            synchronizer_id: Unique identifier
            detection_method: R-peak detection algorithm
        """
        self.synchronizer_id = synchronizer_id
        self.detection_method = detection_method

        # Cardiac monitoring
        self.ecg_interface: Optional[Any] = None
        self.is_monitoring = False
        self.last_r_peak: Optional[datetime] = None
        self.r_peak_history: List[datetime] = []

        # Detection parameters
        self.detection_threshold = 0.5
        self.refractory_period_ms = 300  # Minimum time between R-peaks

        # Threading for real-time monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        self._lock = threading.Lock()  # Thread-safe access to shared data

        logger.info(f"Initialized HeartbeatSynchronizer {synchronizer_id}")

    def initialize(self, ecg_interface: Optional[Any] = None) -> bool:
        """Initialize cardiac monitoring."""
        try:
            self.ecg_interface = ecg_interface or "mock_ecg"  # Would be actual ECG interface

            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitor_cardiac_signal)
            self.monitoring_thread.daemon = True
            self.stop_monitoring.clear()
            self.monitoring_thread.start()

            self.is_monitoring = True
            logger.info("Cardiac monitoring started")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize cardiac monitoring: {e}")
            return False

    def _monitor_cardiac_signal(self) -> None:
        """Monitor cardiac signal for R-peak detection."""
        logger.info("Cardiac signal monitoring thread started")

        while not self.stop_monitoring.is_set():
            try:
                # Simulate ECG signal acquisition and R-peak detection
                # In real implementation, would process actual ECG data

                # Simulate heartbeat at ~60 BPM with some variability
                base_interval = 1.0  # 60 BPM = 1 beat per second
                variability = np.random.normal(0, 0.1)  # HRV simulation
                sleep_time = max(0.5, base_interval + variability)

                time.sleep(sleep_time)

                # Simulate R-peak detection
                current_time = datetime.now()

                # Check refractory period
                if (
                    self.last_r_peak is None
                    or (current_time - self.last_r_peak).total_seconds() * 1000
                    >= self.refractory_period_ms
                ):
                    with self._lock:
                        self.last_r_peak = current_time
                        self.r_peak_history.append(current_time)

                        # Keep only recent history (last 100 beats)
                        if len(self.r_peak_history) > 100:
                            self.r_peak_history.pop(0)

                    logger.debug(f"R-peak detected at {current_time.isoformat()}")

            except Exception as e:
                logger.error(f"Error in cardiac monitoring: {e}")
                time.sleep(0.1)  # Brief pause before retry

        logger.info("Cardiac signal monitoring thread stopped")

    def wait_for_r_peak(self, timeout_ms: float = 2000.0) -> Optional[datetime]:
        """Wait for the next R-peak detection.

        Args:
            timeout_ms: Maximum time to wait in milliseconds

        Returns:
            Timestamp of detected R-peak or None if timeout
        """
        if not self.is_monitoring:
            logger.error("Cardiac monitoring not active")
            return None

        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() * 1000 < timeout_ms:
            with self._lock:
                if self.last_r_peak is not None:
                    logger.debug(
                        f"R-peak detected for synchronization: {self.last_r_peak.isoformat()}"
                    )
                    return self.last_r_peak

            time.sleep(0.001)  # 1ms polling interval

        logger.warning(f"R-peak detection timeout after {timeout_ms:.1f}ms")
        return None

    def get_heart_rate(self) -> Optional[float]:
        """Calculate current heart rate from recent R-peaks.

        Returns:
            Heart rate in BPM or None if insufficient data
        """
        if len(self.r_peak_history) < 2:
            return None

        # Calculate intervals from recent beats
        recent_peaks = self.r_peak_history[-10:]  # Last 10 beats
        intervals = []

        for i in range(1, len(recent_peaks)):
            interval_ms = (recent_peaks[i] - recent_peaks[i - 1]).total_seconds() * 1000
            intervals.append(interval_ms)

        if intervals:
            mean_interval_ms = np.mean(intervals)
            heart_rate_bpm = 60000.0 / mean_interval_ms  # Convert to BPM
            return float(heart_rate_bpm)

        return None

    def get_rr_interval(self) -> Optional[float]:
        """Get the most recent R-R interval.

        Returns:
            R-R interval in milliseconds or None if insufficient data
        """
        if len(self.r_peak_history) < 2:
            return None

        last_interval = (self.r_peak_history[-1] - self.r_peak_history[-2]).total_seconds() * 1000
        return last_interval

    def predict_next_r_peak(self) -> Optional[datetime]:
        """Predict timing of next R-peak based on recent history.

        Returns:
            Predicted timestamp of next R-peak
        """
        if len(self.r_peak_history) < 3:
            return None

        # Calculate mean interval from recent beats
        recent_intervals = []
        for i in range(1, min(6, len(self.r_peak_history))):  # Last 5 intervals
            interval = (self.r_peak_history[-i] - self.r_peak_history[-i - 1]).total_seconds()
            recent_intervals.append(interval)

        if recent_intervals:
            mean_interval = np.mean(recent_intervals)
            predicted_time = self.r_peak_history[-1] + timedelta(seconds=float(mean_interval))
            return predicted_time

        return None

    def cleanup(self) -> None:
        """Clean up cardiac monitoring."""
        self.stop_monitoring.set()

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)

        self.is_monitoring = False
        logger.info("HeartbeatSynchronizer cleaned up")
