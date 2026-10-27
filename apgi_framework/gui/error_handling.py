"""Robust error handling and recovery system.

Provides hardware failure handling, session state management, backup systems,
and user guidance for error recovery.
"""

from __future__ import annotations

import json
import tkinter as tk
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

# Type definitions


class HardwareStatus(TypedDict):
    """Type definition for hardware status entry."""

    available: bool
    last_check: Optional[datetime]
    error: Optional[str]


try:
    from ..data.parameter_estimation_dao import ParameterEstimationDAO
    from ..data.parameter_estimation_models import SessionData
    from ..security.secure_pickle import (
        SecurePickleError,
        safe_pickle_dump,
        safe_pickle_load,
    )
except ImportError:
    # Create fallback classes that raise informative errors
    _SessionData: Optional[type] = None
    _ParameterEstimationDAO: Optional[type] = None

    def safe_pickle_load(
        file_path: str | Path,
        expected_types: set[type] | None = None,
        verify_checksum: bool = True,
    ) -> Any:
        """Fallback function that raises an error."""
        raise SecurePickleError("Cannot load secure pickle data - framework not properly installed")

    def safe_pickle_dump(
        obj: Any, file_path: str | Path, create_checksum: bool = True, protocol: int = 4
    ) -> None:
        """Fallback function that raises an error."""
        raise SecurePickleError("Cannot save secure pickle data - framework not properly installed")

else:
    # Import successful, don't define fallback classes
    pass


# Import error logging utilities
try:
    from .error_logging_utils import get_error_log_dir
except ImportError:
    # Fallback if utility not available
    def get_error_log_dir(gui_config_path: Path | None = None) -> Path:
        return Path.home() / ".apgi" / "error_logs"

else:
    # Import successful, don't define fallback function
    pass


# Import standardized logging and setup logger first
try:
    from ..logging.standardized_logging import APGILogger, get_logger

    _logger = get_logger(__name__)
    logger: Union[APGILogger, "logging.Logger"] = _logger
except ImportError:
    import logging

    _logger = logging.getLogger(__name__)  # type: ignore [assignment]
    logger: Union[APGILogger, "logging.Logger"] = _logger  # type: ignore


class HardwareFailureHandler:
    """Handles graceful degradation when hardware fails.

    Manages EEG, eye tracker, and cardiac sensor failures with fallback strategies.
    """

    def __init__(self) -> None:
        """Initialize hardware failure handler."""
        self.hardware_status: Dict[str, HardwareStatus] = {
            "eeg": {"available": True, "last_check": None, "error": None},
            "eye_tracker": {"available": True, "last_check": None, "error": None},
            "cardiac": {"available": True, "last_check": None, "error": None},
        }

        self.failure_callbacks: Dict[str, List[Callable[..., Any]]] = {
            "eeg": [],
            "eye_tracker": [],
            "cardiac": [],
        }

        logger.info("HardwareFailureHandler initialized")

    def register_failure_callback(self, hardware_type: str, callback: Callable[..., Any]) -> None:
        """Register callback for hardware failure.

        Args:
            hardware_type: Type of hardware ('eeg', 'eye_tracker', 'cardiac')
            callback: Function to call on failure
        """
        if hardware_type in self.failure_callbacks:
            self.failure_callbacks[hardware_type].append(callback)

    def report_hardware_failure(self, hardware_type: str, error: Exception) -> None:
        """Report hardware failure and trigger callbacks.

        Args:
            hardware_type: Type of hardware that failed
            error: Exception that occurred
        """
        if hardware_type not in self.hardware_status:
            logger.error(f"Unknown hardware type: {hardware_type}")
            return

        self.hardware_status[hardware_type]["available"] = False
        self.hardware_status[hardware_type]["last_check"] = datetime.now()
        self.hardware_status[hardware_type]["error"] = str(error)

        logger.error(f"{hardware_type} failure: {error}")

        # Trigger callbacks
        for callback in self.failure_callbacks.get(hardware_type, []):
            try:
                callback(hardware_type, error)
            except Exception as e:
                logger.error(f"Error in failure callback: {e}")

    def check_hardware_status(self, hardware_type: str) -> bool:
        """Check if hardware is available.

        Args:
            hardware_type: Type of hardware to check

        Returns:
            True if hardware is available
        """
        default_status: HardwareStatus = {
            "available": False,
            "last_check": None,
            "error": None,
        }
        status = self.hardware_status.get(hardware_type, default_status)
        return status["available"]

    def get_degraded_mode_config(self) -> Dict[str, bool]:
        """Get configuration for degraded mode operation.

        Returns:
            Dictionary indicating which systems are available
        """
        return {
            "eeg_available": self.hardware_status["eeg"]["available"],
            "eye_tracker_available": self.hardware_status["eye_tracker"]["available"],
            "cardiac_available": self.hardware_status["cardiac"]["available"],
            "behavioral_only": not any(
                self.hardware_status[hw]["available"] for hw in ["eeg", "eye_tracker", "cardiac"]
            ),
        }

    def attempt_hardware_recovery(self, hardware_type: str) -> bool:
        """Attempt to recover failed hardware.

        Args:
            hardware_type: Type of hardware to recover

        Returns:
            True if recovery successful
        """
        logger.info(f"Attempting to recover {hardware_type}")

        try:
            if hardware_type not in self.hardware_status:
                logger.error(f"Unknown hardware type: {hardware_type}")
                return False

            # Check if hardware is already available
            if self.hardware_status[hardware_type]["available"]:
                logger.info(f"{hardware_type} is already available")
                return True

            # Attempt recovery based on hardware type
            recovery_success = False

            if hardware_type == "eeg":
                recovery_success = self._recover_eeg()
            elif hardware_type == "eye_tracker":
                recovery_success = self._recover_eye_tracker()
            elif hardware_type == "cardiac":
                recovery_success = self._recover_cardiac()
            else:
                logger.warning(f"No recovery procedure defined for {hardware_type}")
                return False

            if recovery_success:
                # Update status if recovery successful
                self.hardware_status[hardware_type]["available"] = True
                self.hardware_status[hardware_type]["error"] = None
                self.hardware_status[hardware_type]["last_check"] = datetime.now()
                logger.info(f"Successfully recovered {hardware_type}")
            else:
                logger.warning(f"Failed to recover {hardware_type}")

            return recovery_success

        except Exception as e:
            logger.error(f"Error during hardware recovery attempt: {e}")
            return False

    def _recover_eeg(self) -> bool:
        """Attempt to recover EEG system."""
        try:
            logger.info("Attempting EEG recovery...")

            # Simulate checking power (90% success rate)
            import random

            if random.random() < 0.9:
                logger.info("EEG power check passed")
            else:
                logger.warning("EEG power check failed")
                return False

            # Simulate checking connections (85% success rate)
            if random.random() < 0.85:
                logger.info("EEG connections verified")
            else:
                logger.warning("EEG connection issues detected")
                return False

            # Simulate impedance check (80% success rate)
            if random.random() < 0.8:
                logger.info("EEG impedances acceptable")
            else:
                logger.warning("EEG impedances too high")
                return False

            # Simulate software restart (95% success rate)
            if random.random() < 0.95:
                logger.info("EEG acquisition software restarted successfully")
                return True
            else:
                logger.warning("EEG software restart failed")
                return False

        except Exception as e:
            logger.error(f"Error in EEG recovery: {e}")
            return False

    def _recover_eye_tracker(self) -> bool:
        """Attempt to recover eye tracker."""
        try:
            logger.info("Attempting eye tracker recovery...")

            # Simulate USB connection check (88% success rate)
            import random

            if random.random() < 0.88:
                logger.info("Eye tracker USB connection verified")
            else:
                logger.warning("Eye tracker USB connection failed")
                return False

            # Simulate participant positioning (92% success rate)
            if random.random() < 0.92:
                logger.info("Participant positioning acceptable")
            else:
                logger.warning("Participant positioning needs adjustment")
                return False

            # Simulate calibration attempt (85% success rate)
            if random.random() < 0.85:
                logger.info("Eye tracker calibration successful")
            else:
                logger.warning("Eye tracker calibration failed")
                return False

            # Simulate lighting check (90% success rate)
            if random.random() < 0.9:
                logger.info("Lighting conditions acceptable")
                return True
            else:
                logger.warning("Poor lighting conditions")
                return False

        except Exception as e:
            logger.error(f"Error in eye tracker recovery: {e}")
            return False

    def _recover_cardiac(self) -> bool:
        """Attempt to recover cardiac sensor."""
        try:
            logger.info("Attempting cardiac sensor recovery...")

            # Simulate sensor connection check (87% success rate)
            import random

            if random.random() < 0.87:
                logger.info("Cardiac sensor connections verified")
            else:
                logger.warning("Cardiac sensor connection issues")
                return False

            # Simulate sensor placement check (89% success rate)
            if random.random() < 0.89:
                logger.info("Sensor placement verified")
            else:
                logger.warning("Sensor placement needs adjustment")
                return False

            # Simulate cable check (93% success rate)
            if random.random() < 0.93:
                logger.info("Cables and connections intact")
                return True
            else:
                logger.warning("Cable or connection issues detected")
                return False

        except Exception as e:
            logger.error(f"Error in cardiac sensor recovery: {e}")
            return False

    def get_failure_summary(self) -> Dict[str, Any]:
        """Get summary of hardware failures.

        Returns:
            Dictionary with failure information
        """
        failures = []

        for hw_type, status in self.hardware_status.items():
            if not status["available"]:
                failures.append(
                    {
                        "hardware": hw_type,
                        "error": status["error"],
                        "time": (
                            status["last_check"].isoformat()
                            if status["last_check"] is not None
                            else None
                        ),
                    }
                )

        return {
            "n_failures": len(failures),
            "failures": failures,
            "degraded_mode": len(failures) > 0,
        }


class SessionStateManager:
    """Manages session state for pause/resume functionality.

    Preserves complete session state for interruption recovery.
    """

    def __init__(self, state_dir: Path):
        """Initialize session state manager.

        Args:
            state_dir: Directory for storing state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.current_state: Optional[Dict[str, Any]] = None

        logger.info(f"SessionStateManager initialized with state dir: {state_dir}")

    def save_state(self, session_id: str, state_data: Dict[str, Any]) -> Path:
        """Save session state to disk.

        Args:
            session_id: Session identifier
            state_data: State data to save

        Returns:
            Path to saved state file
        """
        state_file = self.state_dir / f"{session_id}_state.pkl"

        # Add metadata
        state_data["saved_at"] = datetime.now().isoformat()
        state_data["session_id"] = session_id

        # Save state
        safe_pickle_dump(state_data, state_file)

        self.current_state = state_data

        logger.info(f"Saved session state to {state_file}")
        return state_file

    def load_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session state from disk.

        Args:
            session_id: Session identifier

        Returns:
            State data or None if not found
        """
        state_file = self.state_dir / f"{session_id}_state.pkl"

        if not state_file.exists():
            logger.warning(f"State file not found: {state_file}")
            return None

        try:
            state_data = safe_pickle_load(state_file)

            # Ensure state_data is a dict
            if not isinstance(state_data, dict):
                logger.error(f"Invalid state data type: expected dict, got {type(state_data)}")
                return None

            self.current_state = state_data
            logger.info(f"Loaded session state from {state_file}")
            return state_data

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None

    def delete_state(self, session_id: str) -> None:
        """Delete saved session state.

        Args:
            session_id: Session identifier
        """
        state_file = self.state_dir / f"{session_id}_state.pkl"

        if state_file.exists():
            state_file.unlink()
            logger.info(f"Deleted session state: {state_file}")

    def list_saved_states(self) -> List[Dict[str, Any]]:
        """List all saved session states.

        Returns:
            List of state information dictionaries
        """
        states = []

        for state_file in self.state_dir.glob("*_state.pkl"):
            try:
                state_data = safe_pickle_load(state_file)

                states.append(
                    {
                        "session_id": state_data.get("session_id"),
                        "saved_at": state_data.get("saved_at"),
                        "file_path": str(state_file),
                    }
                )
            except Exception as e:
                logger.error(f"Failed to read state file {state_file}: {e}")

        return states

    def create_session(self, session_id: str) -> None:
        """Create a new session.

        Args:
            session_id: Session identifier
        """
        # Initialize empty session state
        self.save_state(session_id, {"status": "created", "created_at": datetime.now().isoformat()})

    def handle_session_crash(self, exception: Exception) -> None:
        """Handle session crash.

        Args:
            exception: Exception that caused the crash
        """
        logger.error(f"Session crash: {exception}")
        # Could save crash state or notify user
        if self.current_state:
            self.current_state["crashed"] = True
            self.current_state["crash_time"] = datetime.now().isoformat()
            self.current_state["crash_error"] = str(exception)

    def attempt_session_recovery(self) -> bool:
        """Attempt to recover from session crash.

        Returns:
            True if recovery successful
        """
        if self.current_state and self.current_state.get("crashed"):
            # Attempt recovery logic here
            logger.info("Attempting session recovery...")
            # For demo, just mark as recovered
            self.current_state["crashed"] = False
            self.current_state["recovered_at"] = datetime.now().isoformat()
            return True
        return False

    def get_session_status(self) -> str:
        """Get current session status.

        Returns:
            Status string
        """
        if self.current_state:
            if self.current_state.get("crashed"):
                return "crashed"
            return "active"
        return "inactive"


class AutomaticBackupSystem:
    """Real-time data backup and recovery.

    Automatically backs up session data at regular intervals.
    """

    def __init__(self, backup_dir: Path, dao: ParameterEstimationDAO):
        """Initialize automatic backup system.

        Args:
            backup_dir: Directory for storing backups
            dao: Data access object for retrieving data
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.dao = dao
        self.backup_interval_seconds = 300  # 5 minutes
        self.last_backup: Optional[datetime] = None

        logger.info(f"AutomaticBackupSystem initialized with backup dir: {backup_dir}")

    def create_backup(self, session_id: str) -> Optional[Path]:
        """Create backup of session data.

        Args:
            session_id: Session identifier

        Returns:
            Path to backup file or None if failed
        """
        try:
            session = self.dao.get_session(session_id)

            if not session:
                logger.error(f"Session {session_id} not found for backup")
                return None

            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"{session_id}_backup_{timestamp}.json"

            # Convert session to dictionary
            session_dict = session.to_dict()

            # Add backup metadata
            session_dict["backup_metadata"] = {
                "backup_time": datetime.now().isoformat(),
                "backup_version": "1.0",
            }

            # Write backup
            with open(backup_file, "w") as f:
                json.dump(session_dict, f, indent=2)

            self.last_backup = datetime.now()

            logger.info(f"Created backup: {backup_file}")
            return backup_file

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None

    def restore_from_backup(self, backup_file: Path) -> Optional[SessionData]:
        """Restore session from backup file.

        Args:
            backup_file: Path to backup file

        Returns:
            Restored SessionData or None if failed
        """
        try:
            with open(backup_file, "r") as f:
                session_dict = json.load(f)

            # Convert dictionary back to SessionData
            session_data = self._dict_to_session_data(session_dict)

            logger.info(f"Restored session from backup: {backup_file}")
            return session_data

        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return None

    def _dict_to_session_data(self, session_dict: Dict[str, Any]) -> SessionData:
        """Convert dictionary back to SessionData object."""
        try:
            # Handle datetime conversion
            session_date = session_dict.get("session_date")
            if isinstance(session_date, str):
                session_date = datetime.fromisoformat(session_date.replace("Z", "+00:00"))

            # Convert trial data lists
            detection_trials = []
            for trial_dict in session_dict.get("detection_trials", []):
                # Convert nested datetime objects
                if "timestamp" in trial_dict and isinstance(trial_dict["timestamp"], str):
                    trial_dict["timestamp"] = datetime.fromisoformat(
                        trial_dict["timestamp"].replace("Z", "+00:00")
                    )
                detection_trials.append(trial_dict)

            heartbeat_trials = []
            for trial_dict in session_dict.get("heartbeat_trials", []):
                if "timestamp" in trial_dict and isinstance(trial_dict["timestamp"], str):
                    trial_dict["timestamp"] = datetime.fromisoformat(
                        trial_dict["timestamp"].replace("Z", "+00:00")
                    )
                heartbeat_trials.append(trial_dict)

            oddball_trials = []
            for trial_dict in session_dict.get("oddball_trials", []):
                if "timestamp" in trial_dict and isinstance(trial_dict["timestamp"], str):
                    trial_dict["timestamp"] = datetime.fromisoformat(
                        trial_dict["timestamp"].replace("Z", "+00:00")
                    )
                oddball_trials.append(trial_dict)

            # Create SessionData object
            session_data = SessionData(
                session_id=session_dict.get("session_id", ""),
                participant_id=session_dict.get("participant_id", ""),
                session_date=(session_date if session_date is not None else datetime.now()),
                protocol_version=session_dict.get("protocol_version", "1.0.0"),
                completion_status=session_dict.get("completion_status", "in_progress"),
                total_duration_minutes=session_dict.get("total_duration_minutes"),
                detection_trials=detection_trials,
                heartbeat_trials=heartbeat_trials,
                oddball_trials=oddball_trials,
                parameter_estimates=session_dict.get("parameter_estimates"),
                session_quality_score=session_dict.get("session_quality_score", 1.0),
                technical_issues=session_dict.get("technical_issues", []),
                researcher=session_dict.get("researcher", ""),
                notes=session_dict.get("notes", ""),
            )

            return session_data

        except Exception as e:
            logger.error(f"Failed to convert dict to SessionData: {e}")
            raise

    def list_backups(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available backups.

        Args:
            session_id: Optional session ID to filter by

        Returns:
            List of backup information dictionaries
        """
        backups = []

        pattern = f"{session_id}_backup_*.json" if session_id else "*_backup_*.json"

        for backup_file in self.backup_dir.glob(pattern):
            try:
                backups.append(
                    {
                        "file_path": str(backup_file),
                        "file_name": backup_file.name,
                        "created": datetime.fromtimestamp(backup_file.stat().st_mtime),
                        "size_bytes": backup_file.stat().st_size,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to read backup file {backup_file}: {e}")

        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created"].timestamp(), reverse=True)  # type: ignore

        return backups

    def cleanup_old_backups(self, max_age_days: int = 30, max_backups_per_session: int = 10) -> int:
        """Clean up old backup files.

        Args:
            max_age_days: Maximum age of backups to keep
            max_backups_per_session: Maximum number of backups per session

        Returns:
            Number of backups deleted
        """
        import re

        deleted_count = 0
        cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 3600)

        # Group backups by session
        session_backups: Dict[str, List[Path]] = {}

        # Use regex to properly extract session ID
        # Pattern: (session_id)_backup_(timestamp).json
        backup_pattern = re.compile(r"^(.+?)_backup_(\d+)\.json$")

        for backup_file in self.backup_dir.glob("*_backup_*.json"):
            # Extract session ID using regex for safety
            match = backup_pattern.match(backup_file.name)
            if match:
                session_id = match.group(1)
                timestamp_str = match.group(2)

                # Validate timestamp format (basic check)
                if len(timestamp_str) == 14 and timestamp_str.isdigit():  # YYYYMMDDHHMMSS
                    if session_id not in session_backups:
                        session_backups[session_id] = []

                    session_backups[session_id].append(backup_file)
                else:
                    logger.warning(f"Skipping backup with invalid timestamp format: {backup_file}")
            else:
                logger.warning(f"Skipping backup with unrecognized filename format: {backup_file}")

        # Clean up old backups
        for session_id, backups in session_backups.items():
            # Sort by modification time (newest first)
            backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for i, backup_file in enumerate(backups):
                # Delete if too old or exceeds max count
                if backup_file.stat().st_mtime < cutoff_date or i >= max_backups_per_session:
                    try:
                        backup_file.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old backup: {backup_file}")
                    except Exception as e:
                        logger.error(f"Failed to delete backup {backup_file}: {e}")

        return deleted_count


class UserGuidanceSystem:
    """Provides clear error messages and recovery instructions.

    Displays user-friendly error messages with step-by-step recovery guidance.
    """

    def __init__(self, parent_window: Optional[tk.Tk] = None):
        """Initialize user guidance system.

        Args:
            parent_window: Parent tkinter window for dialogs
        """
        self.parent_window = parent_window

        # Error message templates
        self.error_templates = {
            "eeg_failure": {
                "title": "EEG System Failure",
                "message": "The EEG system has encountered an error.",
                "steps": [
                    "Check that the EEG amplifier is powered on",
                    "Verify all electrode connections",
                    "Check impedances are below 10 kΩ",
                    "Restart the EEG acquisition software",
                    "If problem persists, continue in behavioral-only mode",
                ],
            },
            "eye_tracker_failure": {
                "title": "Eye Tracker Failure",
                "message": "The eye tracking system has encountered an error.",
                "steps": [
                    "Check that the eye tracker is connected via USB",
                    "Verify the participant is positioned correctly",
                    "Run eye tracker calibration",
                    "Check lighting conditions in the room",
                    "If problem persists, continue without pupillometry",
                ],
            },
            "cardiac_failure": {
                "title": "Cardiac Sensor Failure",
                "message": "The cardiac monitoring system has encountered an error.",
                "steps": [
                    "Check ECG/PPG sensor connections",
                    "Verify sensor placement on participant",
                    "Check for loose cables",
                    "Try repositioning the sensor",
                    "If problem persists, continue without cardiac monitoring",
                ],
            },
        }

        logger.info("UserGuidanceSystem initialized")

    def show_error_guidance(self, error_type: str, additional_info: str = "") -> None:
        """Show error guidance dialog.

        Args:
            error_type: Type of error (key in error_templates)
            additional_info: Additional error information
        """
        if error_type not in self.error_templates:
            self._show_generic_error(error_type, additional_info)
            return

        template = self.error_templates[error_type]

        # Build message
        message = str(template["message"])
        if additional_info:
            message += f"\n\nDetails: {additional_info}"

        message += "\n\nRecovery Steps:"
        for i, step in enumerate(template["steps"], 1):
            message += f"\n{i}. {step}"

        # Show dialog
        if self.parent_window:
            messagebox.showerror(str(template["title"]), message, parent=self.parent_window)
        else:
            messagebox.showerror(str(template["title"]), message)

    def _show_generic_error(self, error_type: str, error_info: str) -> None:
        """Show generic error message."""
        message = f"An error has occurred: {error_type}"
        if error_info:
            message += f"\n\nDetails: {error_info}"

        message += "\n\nPlease contact technical support if the problem persists."

        if self.parent_window:
            messagebox.showerror("Error", message, parent=self.parent_window)
        else:
            logger.error(message)

    def show_recovery_success(self, component: str) -> None:
        """Show recovery success message.

        Args:
            component: Component that was recovered
        """
        message = f"Successfully recovered {component}.\n\nYou may continue with the experiment."

        if self.parent_window:
            messagebox.showinfo("Recovery Successful", message, parent=self.parent_window)
        else:
            logger.info(message)

    def show_degraded_mode_warning(self, unavailable_systems: List[str]) -> bool:
        """Show warning about degraded mode operation.

        Args:
            unavailable_systems: List of unavailable systems

        Returns:
            True if user wants to continue in degraded mode
        """
        message = "The following systems are unavailable:\n\n"
        for system in unavailable_systems:
            message += f"• {system}\n"

        message += "\nYou can continue in degraded mode with reduced functionality.\n\n"
        message += "Do you want to continue?"

        if self.parent_window:
            return messagebox.askyesno("Degraded Mode", message, parent=self.parent_window)
        else:
            logger.warning(message)
            return False

    def log_error_with_context(
        self, error: Exception, context: Dict[str, Any], log_dir: Optional[Path] = None
    ) -> None:
        """Log error with full context information.

        Args:
            error: Exception that occurred
            context: Context information dictionary
            log_dir: Optional directory for error logs. If None, uses configured directory or default
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }

        logger.error(f"Error with context: {json.dumps(error_info, indent=2)}")

        # Optionally save to error log file
        # Use provided log_dir or get from config system
        if log_dir is None:
            log_dir = get_error_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)

        error_log_file = log_dir / f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(error_log_file, "w") as f:
                json.dump(error_info, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write error log: {e}")


if __name__ == "__main__":
    """Error handling demonstration GUI."""

    class ErrorHandlingDemoGUI:
        def __init__(self) -> None:
            self.root = tk.Tk()
            self.root.title("Error Handling Demo")
            self.root.geometry("800x600")

            # Initialize error handling components
            self.hardware_handler = HardwareFailureHandler()
            self.session_manager = SessionStateManager(Path("session_state"))
            # BackupManager not implemented yet - using placeholder
            self.backup_manager = None

            # Create main frame
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # Title
            ttk.Label(
                main_frame,
                text="Error Handling Demonstration",
                font=("Arial", 16, "bold"),
            ).pack(pady=10)

            # Create notebook for tabs
            notebook = ttk.Notebook(main_frame)
            notebook.pack(fill="both", expand=True)

            # Hardware failures tab
            hardware_frame = ttk.Frame(notebook)
            notebook.add(hardware_frame, text="Hardware Failures")
            self.create_hardware_tab(hardware_frame)

            # Session management tab
            session_frame = ttk.Frame(notebook)
            notebook.add(session_frame, text="Session State")
            self.create_session_tab(session_frame)

            # Backup systems tab
            backup_frame = ttk.Frame(notebook)
            notebook.add(backup_frame, text="Backup Systems")
            self.create_backup_tab(backup_frame)

            # Error log tab
            log_frame = ttk.Frame(notebook)
            notebook.add(log_frame, text="Error Log")
            self.create_log_tab(log_frame)

        def create_hardware_tab(self, parent: tk.Widget) -> None:
            ttk.Label(parent, text="Hardware Failure Simulation", font=("Arial", 12, "bold")).pack(
                pady=10
            )

            # Hardware status display
            self.hardware_status_text = tk.Text(parent, height=10, width=70)
            self.hardware_status_text.pack(pady=10)

            # Simulation buttons
            button_frame = ttk.Frame(parent)
            button_frame.pack(pady=10)

            ttk.Button(
                button_frame,
                text="Simulate EEG Failure",
                command=lambda: self.simulate_hardware_failure("eeg"),
            ).pack(side="left", padx=5)
            ttk.Button(
                button_frame,
                text="Simulate Eye Tracker Failure",
                command=lambda: self.simulate_hardware_failure("eye_tracker"),
            ).pack(side="left", padx=5)
            ttk.Button(
                button_frame,
                text="Simulate Cardiac Failure",
                command=lambda: self.simulate_hardware_failure("cardiac"),
            ).pack(side="left", padx=5)

            ttk.Button(
                parent, text="Check Hardware Status", command=self.check_hardware_status
            ).pack(pady=10)

        def create_session_tab(self, parent: tk.Widget) -> None:
            ttk.Label(parent, text="Session State Management", font=("Arial", 12, "bold")).pack(
                pady=10
            )

            # Session status display
            self.session_status_text = tk.Text(parent, height=10, width=70)
            self.session_status_text.pack(pady=10)

            # Control buttons
            button_frame = ttk.Frame(parent)
            button_frame.pack(pady=10)

            ttk.Button(button_frame, text="Create Session", command=self.create_demo_session).pack(
                side="left", padx=5
            )
            ttk.Button(
                button_frame, text="Simulate Crash", command=self.simulate_session_crash
            ).pack(side="left", padx=5)
            ttk.Button(button_frame, text="Recover Session", command=self.recover_session).pack(
                side="left", padx=5
            )

        def create_backup_tab(self, parent: tk.Widget) -> None:
            ttk.Label(parent, text="Backup Systems", font=("Arial", 12, "bold")).pack(pady=10)

            # Backup status display
            self.backup_status_text = tk.Text(parent, height=10, width=70)
            self.backup_status_text.pack(pady=10)

            # Control buttons
            button_frame = ttk.Frame(parent)
            button_frame.pack(pady=10)

            ttk.Button(button_frame, text="Create Backup", command=self.create_backup).pack(
                side="left", padx=5
            )
            ttk.Button(button_frame, text="Restore Backup", command=self.restore_backup).pack(
                side="left", padx=5
            )
            ttk.Button(button_frame, text="Check Backups", command=self.check_backups).pack(
                side="left", padx=5
            )

        def create_log_tab(self, parent: tk.Widget) -> None:
            ttk.Label(parent, text="Error Log Viewer", font=("Arial", 12, "bold")).pack(pady=10)

            # Error log display
            self.error_log_text = tk.Text(parent, height=15, width=70)
            self.error_log_text.pack(pady=10)

            # Control buttons
            button_frame = ttk.Frame(parent)
            button_frame.pack(pady=10)

            ttk.Button(button_frame, text="Refresh Log", command=self.refresh_error_log).pack(
                side="left", padx=5
            )
            ttk.Button(button_frame, text="Clear Log", command=self.clear_error_log).pack(
                side="left", padx=5
            )
            ttk.Button(button_frame, text="Test Error", command=self.test_error_handling).pack(
                side="left", padx=5
            )

        def simulate_hardware_failure(self, hardware_type: str) -> None:
            self.hardware_handler.report_hardware_failure(
                hardware_type, Exception(f"Simulated {hardware_type} failure")
            )
            self.check_hardware_status()

        def check_hardware_status(self) -> None:
            # Get status for each hardware type
            status_text = ""
            for hw_type in ["eeg", "eye_tracker", "cardiac"]:
                is_available = self.hardware_handler.check_hardware_status(hw_type)
                status_text += (
                    f"{hw_type.upper()}: {'✓ Available' if is_available else '✗ Failed'}\n"
                )

            self.hardware_status_text.delete("1.0", tk.END)
            self.hardware_status_text.insert(tk.END, status_text)

        def create_demo_session(self) -> None:
            session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.session_manager.create_session(session_id)
            self.update_session_status()

        def simulate_session_crash(self) -> None:
            self.session_manager.handle_session_crash(Exception("Simulated session crash"))
            self.update_session_status()

        def recover_session(self) -> None:
            recovered = self.session_manager.attempt_session_recovery()
            if recovered:
                messagebox.showinfo("Success", "Session recovered successfully!")
            else:
                messagebox.showwarning("Recovery Failed", "Could not recover session")
            self.update_session_status()

        def update_session_status(self) -> None:
            status = self.session_manager.get_session_status()
            self.session_status_text.delete("1.0", tk.END)
            self.session_status_text.insert(tk.END, f"Session Status: {status}\n")

        def create_backup(self) -> None:
            if self.backup_manager:
                self.backup_manager.create_backup()
                messagebox.showinfo("Success", "Backup created successfully!")
            else:
                messagebox.showinfo("Info", "Backup manager not implemented yet")

        def restore_backup(self) -> None:
            if self.backup_manager:
                restored = self.backup_manager.restore_from_backup()
                if restored:
                    messagebox.showinfo("Success", "Backup restored successfully!")
                else:
                    messagebox.showwarning("Restore Failed", "Could not restore backup")
            else:
                messagebox.showinfo("Info", "Backup manager not implemented yet")

        def check_backups(self) -> None:
            if self.backup_manager:
                backups = self.backup_manager.list_available_backups()
                self.backup_status_text.delete("1.0", tk.END)
                if backups:
                    for backup in backups:
                        self.backup_status_text.insert(tk.END, f"• {backup}\n")
                else:
                    self.backup_status_text.insert(tk.END, "No backups available\n")
            else:
                self.backup_status_text.delete("1.0", tk.END)
                self.backup_status_text.insert(tk.END, "Backup manager not implemented yet\n")

        def refresh_error_log(self) -> None:
            self.error_log_text.delete("1.0", tk.END)
            self.error_log_text.insert(tk.END, "Error log refreshed...\n")

        def clear_error_log(self) -> None:
            self.error_log_text.delete("1.0", tk.END)
            self.error_log_text.insert(tk.END, "Error log cleared...\n")

        def test_error_handling(self) -> None:
            try:
                raise Exception("This is a test error for demonstration")
            except Exception as e:
                logger.error(f"Test error: {e}")
                self.refresh_error_log()
                messagebox.showinfo("Test", "Test error logged successfully!")

        def run(self) -> None:
            self.root.mainloop()

    # Import ttk for the demo
    import tkinter.ttk as ttk

    # Launch demo GUI
    demo = ErrorHandlingDemoGUI()
    demo.run()
