"""Hardware Configuration Manager for different EEG systems, eye trackers, and cardiac sensors.

Provides unified interface for configuring and managing various hardware devices
used in the APGI Framework parameter estimation system.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class HardwareType(Enum):
    """Types of hardware devices."""

    EEG = "eeg"
    EYE_TRACKER = "eye_tracker"
    CARDIAC_SENSOR = "cardiac_sensor"
    STIMULUS_DEVICE = "stimulus_device"


class HardwareStatus(Enum):
    """Status of hardware device."""

    NOT_CONFIGURED = "not_configured"
    CONFIGURED = "configured"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class HardwareDevice:
    """Configuration for a hardware device."""

    device_id: str
    device_type: HardwareType
    manufacturer: str
    model: str
    status: HardwareStatus = HardwareStatus.NOT_CONFIGURED

    # Connection settings
    connection_type: str = "usb"  # usb, network, serial, etc.
    connection_params: Dict[str, Any] = field(default_factory=dict)

    # Device-specific settings
    sampling_rate: Optional[int] = None
    channels: Optional[List[str]] = None
    calibration_params: Dict[str, Any] = field(default_factory=dict)

    # LSL settings
    lsl_stream_name: Optional[str] = None
    lsl_stream_type: Optional[str] = None

    error_message: Optional[str] = None


@dataclass
class EEGSystemConfig:
    """Configuration for EEG system."""

    system_name: str
    n_channels: int
    sampling_rate: int
    channel_names: List[str]
    reference_scheme: str = "average"  # average, mastoid, cz, etc.
    ground_electrode: Optional[str] = None
    impedance_threshold: float = 10.0  # kOhms

    # LSL configuration
    lsl_stream_name: str = "EEG"
    lsl_stream_type: str = "EEG"

    # Filtering
    highpass_freq: float = 0.1
    lowpass_freq: float = 30.0
    notch_freq: Optional[float] = 60.0  # Power line frequency


@dataclass
class EyeTrackerConfig:
    """Configuration for eye tracker."""

    tracker_name: str
    sampling_rate: int
    tracking_mode: str = "binocular"  # binocular, monocular

    # Calibration
    calibration_points: int = 9
    validation_points: int = 4
    accuracy_threshold: float = 1.0  # degrees

    # LSL configuration
    lsl_stream_name: str = "EyeTracker"
    lsl_stream_type: str = "Gaze"

    # Processing
    blink_detection_threshold: float = 0.5
    interpolation_method: str = "cubic"  # linear, cubic


@dataclass
class CardiacSensorConfig:
    """Configuration for cardiac sensor."""

    sensor_name: str
    sampling_rate: int
    sensor_type: str = "ecg"  # ecg, ppg

    # Signal processing
    r_peak_detection_method: str = "pan_tompkins"
    hrv_window_size: int = 300  # seconds

    # LSL configuration
    lsl_stream_name: str = "Cardiac"
    lsl_stream_type: str = "ECG"


class HardwareConfigurationManager:
    """Manages hardware configuration for the APGI Framework.

    Provides unified interface for configuring EEG systems, eye trackers,
    cardiac sensors, and other hardware devices.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize hardware configuration manager.

        Args:
            config_dir: Directory for storing hardware configurations.
        """
        self.logger = logging.getLogger(__name__)
        self.config_dir = config_dir or Path.home() / ".apgi_framework" / "hardware_configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Hardware devices
        self.devices: Dict[str, HardwareDevice] = {}

        # Predefined configurations
        self.eeg_configs = self._define_eeg_configs()
        self.eye_tracker_configs = self._define_eye_tracker_configs()
        self.cardiac_configs = self._define_cardiac_configs()

    def _define_eeg_configs(self) -> Dict[str, EEGSystemConfig]:
        """Define predefined EEG system configurations."""
        configs = {}

        # BioSemi ActiveTwo 128-channel
        configs["biosemi_128"] = EEGSystemConfig(
            system_name="BioSemi ActiveTwo 128",
            n_channels=128,
            sampling_rate=1024,
            channel_names=self._generate_biosemi_channel_names(128),
            reference_scheme="average",
            lsl_stream_name="BioSemi",
            lsl_stream_type="EEG",
        )

        # EGI HydroCel 128-channel
        configs["egi_128"] = EEGSystemConfig(
            system_name="EGI HydroCel 128",
            n_channels=128,
            sampling_rate=1000,
            channel_names=self._generate_egi_channel_names(128),
            reference_scheme="cz",
            lsl_stream_name="EGI",
            lsl_stream_type="EEG",
        )

        # Brain Products actiCHamp 64-channel
        configs["brainproducts_64"] = EEGSystemConfig(
            system_name="Brain Products actiCHamp 64",
            n_channels=64,
            sampling_rate=1000,
            channel_names=self._generate_standard_10_20_names(64),
            reference_scheme="average",
            lsl_stream_name="BrainProducts",
            lsl_stream_type="EEG",
        )

        # Generic 32-channel system
        configs["generic_32"] = EEGSystemConfig(
            system_name="Generic 32-channel",
            n_channels=32,
            sampling_rate=1000,
            channel_names=self._generate_standard_10_20_names(32),
            reference_scheme="average",
            lsl_stream_name="EEG",
            lsl_stream_type="EEG",
        )

        return configs

    def _define_eye_tracker_configs(self) -> Dict[str, EyeTrackerConfig]:
        """Define predefined eye tracker configurations."""
        configs = {}

        # Tobii Pro Spectrum
        configs["tobii_spectrum"] = EyeTrackerConfig(
            tracker_name="Tobii Pro Spectrum",
            sampling_rate=1200,
            tracking_mode="binocular",
            calibration_points=9,
            lsl_stream_name="TobiiSpectrum",
            lsl_stream_type="Gaze",
        )

        # EyeLink 1000 Plus
        configs["eyelink_1000"] = EyeTrackerConfig(
            tracker_name="EyeLink 1000 Plus",
            sampling_rate=1000,
            tracking_mode="binocular",
            calibration_points=9,
            lsl_stream_name="EyeLink",
            lsl_stream_type="Gaze",
        )

        # Pupil Labs Core
        configs["pupil_core"] = EyeTrackerConfig(
            tracker_name="Pupil Labs Core",
            sampling_rate=200,
            tracking_mode="binocular",
            calibration_points=9,
            lsl_stream_name="PupilCore",
            lsl_stream_type="Gaze",
        )

        return configs

    def _define_cardiac_configs(self) -> Dict[str, CardiacSensorConfig]:
        """Define predefined cardiac sensor configurations."""
        configs = {}

        # BioSemi ECG
        configs["biosemi_ecg"] = CardiacSensorConfig(
            sensor_name="BioSemi ECG",
            sensor_type="ecg",
            sampling_rate=1024,
            r_peak_detection_method="pan_tompkins",
            lsl_stream_name="BioSemi_ECG",
            lsl_stream_type="ECG",
        )

        # Polar H10 PPG
        configs["polar_h10"] = CardiacSensorConfig(
            sensor_name="Polar H10",
            sensor_type="ppg",
            sampling_rate=130,
            r_peak_detection_method="threshold",
            lsl_stream_name="PolarH10",
            lsl_stream_type="PPG",
        )

        # Generic ECG
        configs["generic_ecg"] = CardiacSensorConfig(
            sensor_name="Generic ECG",
            sensor_type="ecg",
            sampling_rate=1000,
            r_peak_detection_method="pan_tompkins",
            lsl_stream_name="ECG",
            lsl_stream_type="ECG",
        )

        return configs

    def _generate_biosemi_channel_names(self, n_channels: int) -> List[str]:
        """Generate BioSemi channel names."""
        # Simplified - actual BioSemi has specific naming
        return (
            [f"A{i + 1}" for i in range(min(32, n_channels))]
            + [f"B{i + 1}" for i in range(min(32, max(0, n_channels - 32)))]
            + [f"C{i + 1}" for i in range(min(32, max(0, n_channels - 64)))]
            + [f"D{i + 1}" for i in range(min(32, max(0, n_channels - 96)))]
        )

    def _generate_egi_channel_names(self, n_channels: int) -> List[str]:
        """Generate EGI channel names."""
        return [f"E{i + 1}" for i in range(n_channels)]

    def _generate_standard_10_20_names(self, n_channels: int) -> List[str]:
        """Generate standard 10-20 system channel names."""
        standard_names = [
            "Fp1",
            "Fp2",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "T7",
            "C3",
            "Cz",
            "C4",
            "T8",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "O1",
            "Oz",
            "O2",
            "F9",
            "F10",
            "T9",
            "T10",
            "P9",
            "P10",
            "FC1",
            "FC2",
            "FC5",
            "FC6",
            "CP1",
            "CP2",
            "CP5",
            "CP6",
        ]

        if n_channels <= len(standard_names):
            return standard_names[:n_channels]
        else:
            # Add extra channels
            extra = [f"EX{i + 1}" for i in range(n_channels - len(standard_names))]
            return standard_names + extra

    def configure_eeg_system(
        self, config_name: str, custom_params: Optional[Dict] = None
    ) -> HardwareDevice:
        """Configure EEG system.

        Args:
            config_name: Name of predefined configuration or 'custom'.
            custom_params: Custom parameters for configuration.

        Returns:
            Configured hardware device.
        """
        self.logger.info(f"Configuring EEG system: {config_name}")

        if config_name in self.eeg_configs:
            config = self.eeg_configs[config_name]
        elif custom_params:
            config = EEGSystemConfig(**custom_params)
        else:
            raise ValueError(f"Unknown EEG configuration: {config_name}")

        device = HardwareDevice(
            device_id=f"eeg_{config_name}",
            device_type=HardwareType.EEG,
            manufacturer=config.system_name.split()[0],
            model=config.system_name,
            status=HardwareStatus.CONFIGURED,
            sampling_rate=config.sampling_rate,
            channels=config.channel_names,
            lsl_stream_name=config.lsl_stream_name,
            lsl_stream_type=config.lsl_stream_type,
            calibration_params={
                "reference_scheme": config.reference_scheme,
                "impedance_threshold": config.impedance_threshold,
                "highpass_freq": config.highpass_freq,
                "lowpass_freq": config.lowpass_freq,
                "notch_freq": config.notch_freq,
            },
        )

        self.devices[device.device_id] = device
        self.logger.info(f"EEG system configured: {device.device_id}")

        return device

    def configure_eye_tracker(
        self, config_name: str, custom_params: Optional[Dict] = None
    ) -> HardwareDevice:
        """Configure eye tracker.

        Args:
            config_name: Name of predefined configuration or 'custom'.
            custom_params: Custom parameters for configuration.

        Returns:
            Configured hardware device.
        """
        self.logger.info(f"Configuring eye tracker: {config_name}")

        if config_name in self.eye_tracker_configs:
            config = self.eye_tracker_configs[config_name]
        elif custom_params:
            config = EyeTrackerConfig(**custom_params)
        else:
            raise ValueError(f"Unknown eye tracker configuration: {config_name}")

        device = HardwareDevice(
            device_id=f"eyetracker_{config_name}",
            device_type=HardwareType.EYE_TRACKER,
            manufacturer=config.tracker_name.split()[0],
            model=config.tracker_name,
            status=HardwareStatus.CONFIGURED,
            sampling_rate=config.sampling_rate,
            lsl_stream_name=config.lsl_stream_name,
            lsl_stream_type=config.lsl_stream_type,
            calibration_params={
                "tracking_mode": config.tracking_mode,
                "calibration_points": config.calibration_points,
                "validation_points": config.validation_points,
                "accuracy_threshold": config.accuracy_threshold,
                "blink_detection_threshold": config.blink_detection_threshold,
                "interpolation_method": config.interpolation_method,
            },
        )

        self.devices[device.device_id] = device
        self.logger.info(f"Eye tracker configured: {device.device_id}")

        return device

    def configure_cardiac_sensor(
        self, config_name: str, custom_params: Optional[Dict] = None
    ) -> HardwareDevice:
        """Configure cardiac sensor.

        Args:
            config_name: Name of predefined configuration or 'custom'.
            custom_params: Custom parameters for configuration.

        Returns:
            Configured hardware device.
        """
        self.logger.info(f"Configuring cardiac sensor: {config_name}")

        if config_name in self.cardiac_configs:
            config = self.cardiac_configs[config_name]
        elif custom_params:
            config = CardiacSensorConfig(**custom_params)
        else:
            raise ValueError(f"Unknown cardiac sensor configuration: {config_name}")

        device = HardwareDevice(
            device_id=f"cardiac_{config_name}",
            device_type=HardwareType.CARDIAC_SENSOR,
            manufacturer=config.sensor_name.split()[0],
            model=config.sensor_name,
            status=HardwareStatus.CONFIGURED,
            sampling_rate=config.sampling_rate,
            lsl_stream_name=config.lsl_stream_name,
            lsl_stream_type=config.lsl_stream_type,
            calibration_params={
                "sensor_type": config.sensor_type,
                "r_peak_detection_method": config.r_peak_detection_method,
                "hrv_window_size": config.hrv_window_size,
            },
        )

        self.devices[device.device_id] = device
        self.logger.info(f"Cardiac sensor configured: {device.device_id}")

        return device

    def save_configuration(self, filename: str) -> None:
        """Save current hardware configuration to file.

        Args:
            filename: Name of configuration file.
        """
        config_path = self.config_dir / filename

        config_data = {
            "devices": {
                device_id: {
                    "device_type": device.device_type.value,
                    "manufacturer": device.manufacturer,
                    "model": device.model,
                    "sampling_rate": device.sampling_rate,
                    "channels": device.channels,
                    "lsl_stream_name": device.lsl_stream_name,
                    "lsl_stream_type": device.lsl_stream_type,
                    "calibration_params": device.calibration_params,
                    "connection_type": device.connection_type,
                    "connection_params": device.connection_params,
                }
                for device_id, device in self.devices.items()
            }
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        self.logger.info(f"Configuration saved to {config_path}")

    def load_configuration(self, filename: str) -> None:
        """Load hardware configuration from file.

        Args:
            filename: Name of configuration file.
        """
        config_path = self.config_dir / filename

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = json.load(f)

        self.devices.clear()

        for device_id, device_data in config_data["devices"].items():
            device = HardwareDevice(
                device_id=device_id,
                device_type=HardwareType(device_data["device_type"]),
                manufacturer=device_data["manufacturer"],
                model=device_data["model"],
                status=HardwareStatus.CONFIGURED,
                sampling_rate=device_data.get("sampling_rate"),
                channels=device_data.get("channels"),
                lsl_stream_name=device_data.get("lsl_stream_name"),
                lsl_stream_type=device_data.get("lsl_stream_type"),
                calibration_params=device_data.get("calibration_params", {}),
                connection_type=device_data.get("connection_type", "usb"),
                connection_params=device_data.get("connection_params", {}),
            )
            self.devices[device_id] = device

        self.logger.info(f"Configuration loaded from {config_path}")

    def get_device(self, device_id: str) -> Optional[HardwareDevice]:
        """Get hardware device by ID."""
        return self.devices.get(device_id)

    def get_devices_by_type(self, device_type: HardwareType) -> List[HardwareDevice]:
        """Get all devices of a specific type."""
        return [device for device in self.devices.values() if device.device_type == device_type]

    def list_available_configs(self) -> Dict[str, List[str]]:
        """List all available predefined configurations."""
        return {
            "eeg": list(self.eeg_configs.keys()),
            "eye_tracker": list(self.eye_tracker_configs.keys()),
            "cardiac": list(self.cardiac_configs.keys()),
        }
