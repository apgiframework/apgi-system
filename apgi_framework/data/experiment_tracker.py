"""Experiment tracking and logging system for APGI Framework.

Provides comprehensive experiment logging, parameter tracking, and timeline management
for falsification testing experiments.
"""

import json
import logging
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..logging.standardized_logging import get_logger


@dataclass
class ExperimentSession:
    """Represents a single experiment session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    session_name: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, cancelled

    # Parameters and configuration
    parameters: Dict[str, Any] = field(default_factory=dict)
    random_seed: Optional[int] = None
    configuration: Dict[str, Any] = field(default_factory=dict)

    # Experimental conditions
    conditions: List[str] = field(default_factory=list)
    current_condition: Optional[str] = None
    trial_count: int = 0
    completed_trials: int = 0

    # Timeline and events
    events: List[Dict[str, Any]] = field(default_factory=list)
    milestones: Dict[str, datetime] = field(default_factory=dict)

    # Results and metrics
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    final_results: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    # Resource usage
    memory_usage_mb: List[float] = field(default_factory=list)
    cpu_usage_percent: List[float] = field(default_factory=list)
    execution_time_seconds: Optional[float] = None


@dataclass
class ExperimentLog:
    """Log entry for experiment events."""

    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    category: str = "general"  # general, parameter, condition, trial, result, error
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # component that generated the log

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "details": self.details,
            "source": self.source,
        }


class ExperimentTracker:
    """Comprehensive experiment tracking and logging system.

    Tracks experiment sessions, parameters, conditions, timelines, and results
    with detailed logging and performance monitoring.
    """

    def __init__(self, storage_path: Union[str, Path] = "experiments"):
        """Initialize experiment tracker.

        Args:
            storage_path: Path to store experiment logs and data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Active sessions
        self.active_sessions: Dict[str, ExperimentSession] = {}
        self.session_lock = threading.Lock()

        # Logging configuration
        self.log_file = self.storage_path / "experiment_logs.jsonl"
        self.setup_logging()

        # Performance monitoring
        self.monitoring_enabled = True
        self.monitoring_interval = 5.0  # seconds

    def setup_logging(self) -> None:
        """Set up logging configuration."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.storage_path / "tracker.log"),
                logging.StreamHandler(),
            ],
        )
        # Setup logging
        self.logger = get_logger(__name__)

    def create_session(
        self,
        experiment_name: str,
        parameters: Dict[str, Any],
        conditions: List[str],
        random_seed: Optional[int] = None,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new experiment session.

        Args:
            experiment_name: Name of the experiment
            parameters: Experiment parameters
            conditions: List of experimental conditions
            random_seed: Random seed for reproducibility
            configuration: Additional configuration settings

        Returns:
            Session ID
        """
        session = ExperimentSession(
            experiment_id=experiment_name,
            session_name=f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            parameters=parameters.copy(),
            conditions=conditions.copy(),
            random_seed=random_seed,
            configuration=configuration or {},
        )

        with self.session_lock:
            self.active_sessions[session.session_id] = session

        # Log session creation
        self.log_event(
            session.session_id,
            "INFO",
            "session",
            f"Created experiment session: {session.session_name}",
            {
                "experiment_id": experiment_name,
                "parameters": parameters,
                "conditions": conditions,
                "random_seed": random_seed,
            },
        )

        # Save session to disk
        self._save_session(session)

        return session.session_id

    def start_session(self, session_id: str) -> None:
        """Start an experiment session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        session.start_time = datetime.now()
        session.status = "running"

        self.log_event(
            session_id,
            "INFO",
            "session",
            "Started experiment session",
            {"start_time": session.start_time.isoformat()},
        )

        # Add milestone
        session.milestones["session_started"] = session.start_time
        self._save_session(session)

    def end_session(self, session_id: str, status: str = "completed") -> None:
        """End an experiment session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        session.end_time = datetime.now()
        session.status = status

        if session.start_time:
            session.execution_time_seconds = (session.end_time - session.start_time).total_seconds()

        self.log_event(
            session_id,
            "INFO",
            "session",
            f"Ended experiment session with status: {status}",
            {
                "end_time": session.end_time.isoformat(),
                "execution_time_seconds": session.execution_time_seconds,
                "completed_trials": session.completed_trials,
                "total_trials": session.trial_count,
            },
        )

        # Add milestone
        session.milestones["session_ended"] = session.end_time
        self._save_session(session)

    def log_parameter_change(
        self,
        session_id: str,
        parameter_name: str,
        old_value: Any,
        new_value: Any,
        reason: str = "",
    ) -> None:
        """Log a parameter change during experiment."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        session.parameters[parameter_name] = new_value

        self.log_event(
            session_id,
            "INFO",
            "parameter",
            f"Parameter changed: {parameter_name}",
            {
                "parameter_name": parameter_name,
                "old_value": old_value,
                "new_value": new_value,
                "reason": reason,
            },
        )

        self._save_session(session)

    def log_condition_change(
        self,
        session_id: str,
        new_condition: str,
        condition_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a condition change during experiment."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        old_condition = session.current_condition
        session.current_condition = new_condition

        self.log_event(
            session_id,
            "INFO",
            "condition",
            f"Condition changed: {old_condition} -> {new_condition}",
            {
                "old_condition": old_condition,
                "new_condition": new_condition,
                "condition_parameters": condition_parameters or {},
            },
        )

        self._save_session(session)

    def log_trial_start(
        self,
        session_id: str,
        trial_number: int,
        trial_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log the start of a trial."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        session.trial_count = max(session.trial_count, trial_number)

        self.log_event(
            session_id,
            "DEBUG",
            "trial",
            f"Started trial {trial_number}",
            {
                "trial_number": trial_number,
                "trial_parameters": trial_parameters or {},
                "condition": session.current_condition,
            },
        )

    def log_trial_end(
        self,
        session_id: str,
        trial_number: int,
        trial_results: Optional[Dict[str, Any]] = None,
        success: bool = True,
    ) -> None:
        """Log the end of a trial."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        if success:
            session.completed_trials += 1

        self.log_event(
            session_id,
            "DEBUG",
            "trial",
            f"Completed trial {trial_number} (success: {success})",
            {
                "trial_number": trial_number,
                "trial_results": trial_results or {},
                "success": success,
                "completed_trials": session.completed_trials,
            },
        )

        # Update intermediate results
        if trial_results and success:
            trial_key = f"trial_{trial_number}"
            session.intermediate_results[trial_key] = trial_results

        self._save_session(session)

    def log_milestone(
        self,
        session_id: str,
        milestone_name: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an experiment milestone."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        timestamp = datetime.now()
        session.milestones[milestone_name] = timestamp

        self.log_event(
            session_id,
            "INFO",
            "milestone",
            f"Reached milestone: {milestone_name}",
            details or {},
        )

        self._save_session(session)

    def log_results(
        self, session_id: str, results: Dict[str, Any], result_type: str = "final"
    ) -> None:
        """Log experiment results."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]

        if result_type == "final":
            session.final_results = results
        else:
            session.intermediate_results[result_type] = results

        self.log_event(
            session_id,
            "INFO",
            "result",
            f"Logged {result_type} results",
            {"result_keys": list(results.keys())},
        )

        self._save_session(session)

    def log_error(
        self,
        session_id: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
    ) -> None:
        """Log an error during experiment."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "message": error_message,
                "details": error_details or {},
                "exception_type": type(exception).__name__ if exception else None,
                "exception_str": str(exception) if exception else None,
            }
            session.errors.append(error_entry)

        self.log_event(session_id, "ERROR", "error", error_message, error_details or {})

        if session_id in self.active_sessions:
            self._save_session(self.active_sessions[session_id])

    def log_warning(
        self,
        session_id: str,
        warning_message: str,
        warning_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a warning during experiment."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            warning_entry = {
                "timestamp": datetime.now().isoformat(),
                "message": warning_message,
                "details": warning_details or {},
            }
            session.warnings.append(warning_entry)

        self.log_event(session_id, "WARNING", "warning", warning_message, warning_details or {})

        if session_id in self.active_sessions:
            self._save_session(self.active_sessions[session_id])

    def log_event(
        self,
        session_id: str,
        level: str,
        category: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        source: str = "tracker",
    ) -> None:
        """Log a general event."""
        log_entry = ExperimentLog(
            session_id=session_id,
            level=level,
            category=category,
            message=message,
            details=details or {},
            source=source,
        )

        # Write to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry.to_dict()) + "\n")

        # Add to session events if session exists
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.events.append(log_entry.to_dict())

    def get_session(self, session_id: str) -> Optional[ExperimentSession]:
        """Get session by ID."""
        return self.active_sessions.get(session_id)

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of session."""
        session = self.get_session(session_id)
        if not session:
            return {}

        return {
            "session_id": session.session_id,
            "experiment_id": session.experiment_id,
            "session_name": session.session_name,
            "status": session.status,
            "start_time": (session.start_time.isoformat() if session.start_time else None),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "execution_time_seconds": session.execution_time_seconds,
            "trial_count": session.trial_count,
            "completed_trials": session.completed_trials,
            "conditions": session.conditions,
            "current_condition": session.current_condition,
            "error_count": len(session.errors),
            "warning_count": len(session.warnings),
            "milestone_count": len(session.milestones),
            "has_final_results": session.final_results is not None,
        }

    def list_sessions(
        self, experiment_id: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List sessions with optional filtering."""
        sessions = []

        for session in self.active_sessions.values():
            if experiment_id and session.experiment_id != experiment_id:
                continue
            if status and session.status != status:
                continue

            sessions.append(self.get_session_summary(session.session_id))

        return sessions

    def export_session_logs(
        self, session_id: str, output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Export session logs to file."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if output_path is None:
            output_path = self.storage_path / f"session_{session_id}_logs.json"
        else:
            output_path = Path(output_path)

        # Prepare export data
        export_data = {
            "session_info": asdict(session),
            "logs": [event for event in session.events],
        }

        # Write to file
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        return output_path

    def _save_session(self, session: ExperimentSession) -> None:
        """Save session to disk using secure pickle."""
        session_file = self.storage_path / f"session_{session.session_id}.pkl"
        from ..security.secure_pickle import safe_pickle_dump

        safe_pickle_dump(session, session_file)

    def _load_session(self, session_id: str) -> Optional[ExperimentSession]:
        """Load session from disk using secure pickle."""
        session_file = self.storage_path / f"session_{session_id}.pkl"
        if not session_file.exists():
            return None

        try:
            from ..security.secure_pickle import safe_pickle_load

            session: ExperimentSession = safe_pickle_load(
                session_file, expected_types={ExperimentSession}
            )
            return session
        except Exception as e:
            self.logger.error(f"Failed to load session {session_id}: {e}")
            return None

    @contextmanager
    def session_context(
        self,
        experiment_name: str,
        parameters: Dict[str, Any],
        conditions: List[str],
        random_seed: Optional[int] = None,
    ) -> Any:
        """Context manager for experiment sessions."""
        session_id = self.create_session(experiment_name, parameters, conditions, random_seed)

        try:
            self.start_session(session_id)
            yield session_id
            self.end_session(session_id, "completed")
        except Exception as e:
            self.log_error(session_id, f"Session failed: {str(e)}", exception=e)
            self.end_session(session_id, "failed")
            raise
        finally:
            # Clean up active session
            with self.session_lock:
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
