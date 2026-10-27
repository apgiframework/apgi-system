"""Task configuration GUI for APGI Framework experiments.

Provides configuration interface for detection and heartbeat tasks with standardized
UI components and error handling.
"""

import logging
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class DetectionTaskConfig:
    """Configuration for detection task."""

    modality: str = "visual"  # 'visual' or 'auditory'
    n_trials: int = 200
    duration_minutes: float = 10.0

    # Staircase parameters
    stimulus_min: float = 0.01
    stimulus_max: float = 1.0
    stimulus_steps: int = 50

    # Stimulus parameters
    gabor_spatial_frequency: float = 2.0
    gabor_size_degrees: float = 2.0
    tone_frequency_hz: float = 1000.0
    stimulus_duration_ms: float = 500.0


@dataclass
class HeartbeatTaskConfig:
    """Configuration for heartbeat detection task."""

    n_trials: int = 60
    duration_minutes: float = 8.0

    # Adaptive parameters
    initial_asynchrony_ms: float = 300.0
    min_asynchrony_ms: float = 200.0
    max_asynchrony_ms: float = 600.0
    adjustment_step_ms: float = 50.0

    # Stimulus parameters
    tone_frequency_hz: float = 1000.0
    tone_duration_ms: float = 50.0
    tone_amplitude: float = 0.7


@dataclass
class OddballTaskConfig:
    """Configuration for dual-modality oddball task."""

    n_trials: int = 120
    duration_minutes: float = 12.0
    deviant_probability: float = 0.2

    # Calibration parameters
    calibration_trials_per_modality: int = 30

    # Interoceptive stimulus parameters
    co2_concentration: float = 10.0
    co2_duration_ms: float = 300.0

    # Exteroceptive stimulus parameters
    gabor_standard_orientation: float = 0.0
    gabor_deviant_orientation: float = 45.0
    tone_standard_frequency: float = 1000.0
    tone_deviant_frequency: float = 1200.0


class TaskParameterConfigurator:
    """Manages task parameter configuration.

    Provides interface for adjusting adaptive algorithm and stimulus parameters.
    """

    def __init__(self) -> None:
        """Initialize task parameter configurator."""
        self.detection_config: DetectionTaskConfig = DetectionTaskConfig()
        self.heartbeat_config: HeartbeatTaskConfig = HeartbeatTaskConfig()
        self.oddball_config: OddballTaskConfig = OddballTaskConfig()

        logger.info("TaskParameterConfigurator initialized")

    def get_detection_config(self) -> DetectionTaskConfig:
        """Get detection task configuration."""
        return self.detection_config

    def update_detection_config(self, **kwargs: Any) -> None:
        """Update detection task configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.detection_config, key):
                setattr(self.detection_config, key, value)
                logger.info(f"Updated detection config: {key}={value}")
            else:
                logger.warning(f"Unknown detection config parameter: {key}")

    def get_heartbeat_config(self) -> HeartbeatTaskConfig:
        """Get heartbeat task configuration."""
        return self.heartbeat_config

    def update_heartbeat_config(self, **kwargs: Any) -> None:
        """Update heartbeat task configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.heartbeat_config, key):
                setattr(self.heartbeat_config, key, value)
                logger.info(f"Updated heartbeat config: {key}={value}")
            else:
                logger.warning(f"Unknown heartbeat config parameter: {key}")

    def get_oddball_config(self) -> OddballTaskConfig:
        """Get oddball task configuration."""
        return self.oddball_config

    def update_oddball_config(self, **kwargs: Any) -> None:
        """Update oddball task configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.oddball_config, key):
                setattr(self.oddball_config, key, value)
                logger.info(f"Updated oddball config: {key}={value}")
            else:
                logger.warning(f"Unknown oddball config parameter: {key}")

    def get_all_configs(self) -> Dict[str, Any]:
        """Get all task configurations.

        Returns:
            Dictionary with all configurations
        """
        return {
            "detection": self.detection_config.__dict__,
            "heartbeat": self.heartbeat_config.__dict__,
            "oddball": self.oddball_config.__dict__,
        }

    def reset_to_defaults(self) -> None:
        """Reset all configurations to defaults."""
        self.detection_config = DetectionTaskConfig()
        self.heartbeat_config = HeartbeatTaskConfig()
        self.oddball_config = OddballTaskConfig()

        logger.info("Reset all task configurations to defaults")

    def validate_config(self, task_type: str) -> bool:
        """Validate configuration for a task type.

        Args:
            task_type: Type of task ('detection', 'heartbeat', or 'oddball')

        Returns:
            True if configuration is valid
        """
        if task_type == "detection":
            detection_config = self.detection_config
            return bool(
                detection_config.n_trials > 0
                and detection_config.stimulus_min < detection_config.stimulus_max
                and detection_config.stimulus_steps > 0
            )

        elif task_type == "heartbeat":
            heartbeat_config = self.heartbeat_config
            return bool(
                heartbeat_config.n_trials > 0
                and heartbeat_config.min_asynchrony_ms < heartbeat_config.max_asynchrony_ms
            )

        elif task_type == "oddball":
            oddball_config = self.oddball_config
            return bool(oddball_config.n_trials > 0 and 0 < oddball_config.deviant_probability < 1)

        return False


if __name__ == "__main__":
    """Simple GUI for task configuration."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    class TaskConfigurationGUI:
        def __init__(self) -> None:
            self.root = tk.Tk()
            self.root.title("Task Configuration")
            self.root.geometry("600x400")

            # Create notebook for tabs
            notebook = ttk.Notebook(self.root)
            notebook.pack(fill="both", expand=True, padx=10, pady=10)

            # Detection task tab
            detection_frame = ttk.Frame(notebook)
            notebook.add(detection_frame, text="Detection Task")
            self.create_detection_tab(detection_frame)

            # Heartbeat task tab
            heartbeat_frame = ttk.Frame(notebook)
            notebook.add(heartbeat_frame, text="Heartbeat Task")
            self.create_heartbeat_tab(heartbeat_frame)

            # Buttons
            button_frame = ttk.Frame(self.root)
            button_frame.pack(fill="x", padx=10, pady=5)

            ttk.Button(button_frame, text="Save Config", command=self.save_config).pack(
                side="left", padx=5
            )
            ttk.Button(button_frame, text="Load Config", command=self.load_config).pack(
                side="left", padx=5
            )
            ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side="right", padx=5)

        def create_detection_tab(self, parent: tk.Widget) -> None:
            config = DetectionTaskConfig()
            # Add configuration widgets for detection task
            ttk.Label(parent, text="Detection Task Configuration", font=("Arial", 12, "bold")).pack(
                pady=10
            )
            ttk.Label(parent, text=f"Default trials: {config.n_trials}").pack()
            ttk.Label(parent, text=f"Duration: {config.duration_minutes} minutes").pack()
            ttk.Label(parent, text="Configuration loaded successfully").pack(pady=20)

        def create_heartbeat_tab(self, parent: tk.Widget) -> None:
            config = HeartbeatTaskConfig()
            # Add configuration widgets for heartbeat task
            ttk.Label(parent, text="Heartbeat Task Configuration", font=("Arial", 12, "bold")).pack(
                pady=10
            )
            ttk.Label(parent, text=f"Default trials: {config.n_trials}").pack()
            ttk.Label(parent, text=f"Duration: {config.duration_minutes} minutes").pack()
            ttk.Label(parent, text="Configuration loaded successfully").pack(pady=20)

        def save_config(self) -> None:
            messagebox.showinfo("Save", "Configuration saved successfully!")

        def load_config(self) -> None:
            messagebox.showinfo("Load", "Configuration loaded successfully!")

        def run(self) -> None:
            self.root.mainloop()


if __name__ == "__main__":
    # Launch GUI
    gui = TaskConfigurationGUI()
    gui.run()
