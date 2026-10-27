"""Task control and timing infrastructure for parameter estimation tasks.

Implements precision timing, task state management, response collection,
and session management for the three core parameter estimation tasks.
"""

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)


class TaskState(Enum):
    """States for task state machine."""

    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class ResponseType(Enum):
    """Types of participant responses."""

    DETECTION = "detection"
    CONFIDENCE = "confidence"
    FORCED_CHOICE = "forced_choice"
    SYNCHRONY_JUDGMENT = "synchrony_judgment"


@dataclass
class TimingEvent:
    """Represents a timed event in the task."""

    event_id: str
    event_type: str
    scheduled_time: datetime
    actual_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_timing_error_ms(self) -> Optional[float]:
        """Calculate timing error in milliseconds."""
        if self.actual_time is None:
            return None
        return (self.actual_time - self.scheduled_time).total_seconds() * 1000


@dataclass
class ResponseData:
    """Participant response data."""

    response_id: str
    response_type: ResponseType
    response_time: datetime
    reaction_time_ms: float
    response_value: Any
    confidence: Optional[float] = None
    valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "response_id": self.response_id,
            "response_type": self.response_type.value,
            "response_time": self.response_time.isoformat(),
            "reaction_time_ms": self.reaction_time_ms,
            "response_value": self.response_value,
            "confidence": self.confidence,
            "valid": self.valid,
            "metadata": self.metadata,
        }


class PrecisionTimer:
    """High-precision timer for microsecond-accurate stimulus timing.

    Provides precise timing control for stimulus presentation and
    response collection with sub-millisecond accuracy.
    """

    def __init__(self, timer_id: str = "precision_timer"):
        """Initialize precision timer.

        Args:
            timer_id: Unique identifier for this timer
        """
        self.timer_id = timer_id
        self.reference_time: Optional[float] = None
        self.timing_events: List[TimingEvent] = []

        # Performance monitoring
        self.timing_errors: List[float] = []
        self.max_error_ms = 1.0  # Maximum acceptable timing error

        logger.info(f"Initialized PrecisionTimer {timer_id}")

    def start_session(self) -> None:
        """Start timing session and set reference time."""
        self.reference_time = time.perf_counter()
        self.timing_events.clear()
        self.timing_errors.clear()
        logger.info("Precision timing session started")

    def get_current_time_ms(self) -> float:
        """Get current time in milliseconds since session start.

        Returns:
            Time in milliseconds since session start
        """
        if self.reference_time is None:
            raise RuntimeError("Timer session not started")

        return (time.perf_counter() - self.reference_time) * 1000.0

    def schedule_event(
        self,
        event_id: str,
        event_type: str,
        delay_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TimingEvent:
        """Schedule a timed event.

        Args:
            event_id: Unique event identifier
            event_type: Type of event
            delay_ms: Delay in milliseconds from now
            metadata: Optional event metadata

        Returns:
            TimingEvent object
        """
        if self.reference_time is None:
            raise RuntimeError("Timer session not started")

        scheduled_time = datetime.now() + timedelta(milliseconds=delay_ms)

        event = TimingEvent(
            event_id=event_id,
            event_type=event_type,
            scheduled_time=scheduled_time,
            metadata=metadata or {},
        )

        self.timing_events.append(event)
        logger.debug(f"Scheduled event {event_id} in {delay_ms:.3f}ms")

        return event

    def wait_until(self, target_time_ms: float) -> float:
        """Wait until specified time with high precision.

        Args:
            target_time_ms: Target time in milliseconds since session start

        Returns:
            Actual timing error in milliseconds
        """
        if self.reference_time is None:
            raise RuntimeError("Timer session not started")

        target_absolute = self.reference_time + (target_time_ms / 1000.0)

        # Hybrid sleep + busy-wait for high precision with low CPU usage
        while True:
            current_time = time.perf_counter()
            if current_time >= target_absolute:
                break
            time_left = target_absolute - current_time
            if time_left > 0.001:  # More than 1ms left
                time.sleep(time_left - 0.001)
            else:
                # Short sleep instead of busy-wait to reduce CPU usage
                time.sleep(0.0001)

        actual_time = time.perf_counter()
        error_ms = (actual_time - target_absolute) * 1000.0

        self.timing_errors.append(error_ms)

        if abs(error_ms) > self.max_error_ms:
            logger.warning(f"Timing error exceeded threshold: {error_ms:.3f}ms")

        return error_ms

    def sleep_precise(self, duration_ms: float) -> float:
        """Sleep for precise duration.

        Args:
            duration_ms: Sleep duration in milliseconds

        Returns:
            Actual timing error in milliseconds
        """
        start_time = time.perf_counter()
        target_time = start_time + (duration_ms / 1000.0)

        # Use system sleep for most of the duration
        if duration_ms > 2.0:
            time.sleep((duration_ms - 1.0) / 1000.0)

        # Busy wait for final precision with hybrid sleep
        while time.perf_counter() < target_time:
            time_left = target_time - time.perf_counter()
            if time_left > 0.001:
                time.sleep(time_left - 0.001)
            else:
                time.sleep(0.0001)

        actual_duration = (time.perf_counter() - start_time) * 1000.0
        error_ms = actual_duration - duration_ms

        self.timing_errors.append(error_ms)
        return error_ms

    def mark_event_executed(self, event: TimingEvent) -> None:
        """Mark event as executed and record actual timing."""
        event.actual_time = datetime.now()

        timing_error = event.get_timing_error_ms()
        if timing_error is not None:
            self.timing_errors.append(timing_error)
            logger.debug(f"Event {event.event_id} executed with {timing_error:.3f}ms error")

    def get_timing_statistics(self) -> Dict[str, float]:
        """Get timing performance statistics."""
        if not self.timing_errors:
            return {
                "mean_error_ms": 0.0,
                "std_error_ms": 0.0,
                "max_error_ms": 0.0,
                "min_error_ms": 0.0,
                "errors_over_threshold": 0,
                "total_events": 0,
            }
        errors = self.timing_errors
        return {
            "mean_error_ms": sum(errors) / len(errors),
            "std_error_ms": (
                sum((e - sum(errors) / len(errors)) ** 2 for e in errors) / len(errors)
            )
            ** 0.5,
            "max_error_ms": max(errors),
            "min_error_ms": min(errors),
            "errors_over_threshold": sum(1 for e in errors if abs(e) > self.max_error_ms),
            "total_events": len(errors),
        }


class ResponseCollector:
    """Collects participant responses with confidence ratings and reaction times.

    Handles multiple response types and provides precise reaction time measurement.
    """

    def __init__(self, collector_id: str = "response_collector"):
        """Initialize response collector.

        Args:
            collector_id: Unique identifier for this collector
        """
        self.collector_id = collector_id
        self.is_collecting = False
        self.response_queue: queue.Queue[Any] = queue.Queue()

        # Response timing
        self.stimulus_onset_time: Optional[datetime] = None
        self.response_window_ms = 5000.0  # Default 5 second response window

        # Input handling
        self.input_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()  # Thread-safe access to shared data

        logger.info(f"Initialized ResponseCollector {collector_id}")

    def start_collection(
        self, stimulus_onset_time: datetime, response_window_ms: float = 5000.0
    ) -> None:
        """Start collecting responses.

        Args:
            stimulus_onset_time: Time when stimulus was presented
            response_window_ms: Response window duration in milliseconds
        """
        self.stimulus_onset_time = stimulus_onset_time
        self.response_window_ms = response_window_ms
        self.is_collecting = True

        # Clear any previous responses
        while not self.response_queue.empty():
            try:
                self.response_queue.get_nowait()
            except queue.Empty:
                break

        # Start input monitoring thread
        self._stop_event.clear()
        self.input_thread = threading.Thread(target=self._monitor_input)
        self.input_thread.daemon = True
        self.input_thread.start()

        logger.debug(f"Started response collection with {response_window_ms:.1f}ms window")

    def _monitor_input(self) -> None:
        """Monitor for input responses (placeholder implementation)."""
        # This would interface with actual input devices (keyboard, button box, etc.)
        # For now, simulate random responses for testing

        start_time = time.time()

        while not self._stop_event.is_set():
            # Simulate response detection
            time.sleep(0.01)  # 10ms polling

            # Check if response window has expired
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.response_window_ms:
                break

            # Simulate random response (would be actual input detection)
            if elapsed_ms > 200 and not self.response_queue.qsize():  # Minimum 200ms reaction time
                import random

                if random.random() < 0.001:  # Low probability per poll
                    response_time = datetime.now()
                    if self.stimulus_onset_time:
                        reaction_time = (
                            response_time - self.stimulus_onset_time
                        ).total_seconds() * 1000
                    else:
                        reaction_time = 0.0

                    # Simulate response
                    response = ResponseData(
                        response_id=f"resp_{int(time.time() * 1000)}",
                        response_type=ResponseType.DETECTION,
                        response_time=response_time,
                        reaction_time_ms=reaction_time,
                        response_value=random.choice([True, False]),
                        confidence=random.uniform(0.5, 1.0),
                    )

                    self.response_queue.put(response)
                    logger.debug(f"Simulated response: RT={reaction_time:.1f}ms")

    def wait_for_response(self, timeout_ms: Optional[float] = None) -> Optional[ResponseData]:
        """Wait for participant response.

        Args:
            timeout_ms: Timeout in milliseconds (uses response window if None)

        Returns:
            ResponseData if response received, None if timeout
        """
        if not self.is_collecting:
            logger.error("Response collection not started")
            return None

        timeout = (timeout_ms or self.response_window_ms) / 1000.0

        try:
            response: ResponseData = self.response_queue.get(timeout=timeout)
            logger.debug(f"Response received: RT={response.reaction_time_ms:.1f}ms")
            return response
        except queue.Empty:
            logger.debug("Response timeout")
            return None

    def collect_confidence_rating(
        self,
        prompt: str = "How confident are you?",
        scale_min: float = 0.0,
        scale_max: float = 1.0,
    ) -> Optional[float]:
        """Collect confidence rating from participant.

        Args:
            prompt: Confidence rating prompt
            scale_min: Minimum scale value
            scale_max: Maximum scale value

        Returns:
            Confidence rating or None if timeout
        """
        logger.debug(f"Collecting confidence rating: {prompt}")

        # Simulate confidence collection (would be actual UI)
        import random

        time.sleep(random.uniform(0.5, 2.0))  # Simulate response time

        confidence = random.uniform(scale_min, scale_max)
        logger.debug(f"Confidence rating collected: {confidence:.3f}")

        return confidence

    def stop_collection(self) -> None:
        """Stop response collection."""
        self.is_collecting = False
        self._stop_event.set()

        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=0.1)

        logger.debug("Response collection stopped")


class TaskStateMachine:
    """State machine for session flow control across parameter estimation tasks.

    Manages task transitions, error handling, and session coordination.
    """

    def __init__(self, machine_id: str = "task_state_machine"):
        """Initialize task state machine.

        Args:
            machine_id: Unique identifier for this state machine
        """
        self.machine_id = machine_id
        self.current_state = TaskState.IDLE
        self.previous_state: Optional[TaskState] = None

        # State transition callbacks
        self.state_callbacks: Dict[TaskState, List[Callable]] = {state: [] for state in TaskState}

        # State history
        self.state_history: List[Tuple[TaskState, datetime]] = []

        # Error handling
        self.error_message: Optional[str] = None
        self.recovery_callbacks: List[Callable] = []

        logger.info(f"Initialized TaskStateMachine {machine_id}")

    def transition_to(self, new_state: TaskState, message: str = "") -> bool:
        """Transition to new state.

        Args:
            new_state: Target state
            message: Optional transition message

        Returns:
            True if transition successful
        """
        if not self._is_valid_transition(self.current_state, new_state):
            logger.error(f"Invalid transition from {self.current_state.value} to {new_state.value}")
            return False

        # Record transition
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_history.append((new_state, datetime.now()))

        # Clear error message on successful transition (unless transitioning to error)
        if new_state != TaskState.ERROR:
            self.error_message = None

        logger.info(
            f"State transition: {self.previous_state.value} -> {new_state.value} ({message})"
        )

        # Execute state callbacks
        self._execute_state_callbacks(new_state)

        return True

    def _is_valid_transition(self, from_state: TaskState, to_state: TaskState) -> bool:
        """Check if state transition is valid."""
        valid_transitions = {
            TaskState.IDLE: [TaskState.INITIALIZING, TaskState.ERROR],
            TaskState.INITIALIZING: [TaskState.READY, TaskState.ERROR],
            TaskState.READY: [TaskState.RUNNING, TaskState.ERROR],
            TaskState.RUNNING: [TaskState.PAUSED, TaskState.COMPLETED, TaskState.ERROR],
            TaskState.PAUSED: [TaskState.RUNNING, TaskState.COMPLETED, TaskState.ERROR],
            TaskState.COMPLETED: [TaskState.IDLE],
            TaskState.ERROR: [TaskState.IDLE, TaskState.INITIALIZING],
        }

        return to_state in valid_transitions.get(from_state, [])

    def _execute_state_callbacks(self, state: TaskState) -> None:
        """Execute callbacks for state entry."""
        for callback in self.state_callbacks[state]:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Error in state callback for {state.value}: {e}")

    def add_state_callback(self, state: TaskState, callback: Callable) -> None:
        """Add callback for state entry."""
        self.state_callbacks[state].append(callback)
        logger.debug(f"Added callback for state {state.value}")

    def set_error(self, error_message: str) -> None:
        """Set error state with message."""
        self.error_message = error_message
        self.transition_to(TaskState.ERROR, error_message)

    def can_start_task(self) -> bool:
        """Check if task can be started."""
        return self.current_state in [TaskState.READY, TaskState.PAUSED]

    def can_pause_task(self) -> bool:
        """Check if task can be paused."""
        return self.current_state == TaskState.RUNNING

    def can_resume_task(self) -> bool:
        """Check if task can be resumed."""
        return self.current_state == TaskState.PAUSED

    def is_task_active(self) -> bool:
        """Check if task is currently active."""
        return self.current_state in [TaskState.RUNNING, TaskState.PAUSED]

    def get_state_duration(self) -> Optional[float]:
        """Get duration in current state in seconds."""
        if not self.state_history:
            return None

        last_transition_time = self.state_history[-1][1]
        return (datetime.now() - last_transition_time).total_seconds()

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of state machine status."""
        return {
            "current_state": self.current_state.value,
            "previous_state": (self.previous_state.value if self.previous_state else None),
            "state_duration_s": self.get_state_duration(),
            "error_message": self.error_message,
            "total_transitions": len(self.state_history),
            "can_start": self.can_start_task(),
            "can_pause": self.can_pause_task(),
            "can_resume": self.can_resume_task(),
            "is_active": self.is_task_active(),
        }


@dataclass
class SessionConfiguration:
    """Configuration for experimental session."""

    session_id: str
    participant_id: str
    protocol_version: str = "1.0.0"

    # Task configuration
    tasks_to_run: List[str] = field(default_factory=lambda: ["detection", "heartbeat", "oddball"])
    task_order: str = "fixed"  # fixed, randomized, counterbalanced

    # Timing configuration
    inter_task_interval_s: float = 60.0
    max_session_duration_min: float = 90.0

    # Data collection
    save_raw_data: bool = True
    backup_interval_min: float = 5.0

    # Quality control
    min_data_quality: float = 0.7
    auto_pause_on_poor_quality: bool = True

    def validate(self) -> bool:
        """Validate session configuration."""
        return (
            len(self.session_id) > 0
            and len(self.participant_id) > 0
            and len(self.tasks_to_run) > 0
            and self.max_session_duration_min > 0
        )


class SessionManager:
    """Manages experimental sessions with participant tracking and task sequencing.

    Coordinates multiple tasks, handles session state, and manages data collection.
    """

    def __init__(self, manager_id: str = "session_manager"):
        """Initialize session manager.

        Args:
            manager_id: Unique identifier for this manager
        """
        self.manager_id = manager_id
        self.current_session: Optional[SessionConfiguration] = None

        # Session state
        self.session_start_time: Optional[datetime] = None
        self.current_task_index = 0
        self.completed_tasks: List[str] = []

        # Components
        self.state_machine = TaskStateMachine(f"{manager_id}_state")
        self.timer = PrecisionTimer(f"{manager_id}_timer")
        self.response_collector = ResponseCollector(f"{manager_id}_responses")

        # Data management
        self.session_data: Dict[str, Any] = {}
        self.data_backup_thread: Optional[threading.Thread] = None
        self.stop_backup = threading.Event()
        self._data_lock = threading.Lock()  # Thread-safe access to session data

        logger.info(f"Initialized SessionManager {manager_id}")

    def configure_session(self, config: SessionConfiguration) -> bool:
        """Configure new experimental session.

        Args:
            config: Session configuration

        Returns:
            True if configuration successful
        """
        if not config.validate():
            logger.error("Invalid session configuration")
            return False

        self.current_session = config
        self.current_task_index = 0
        self.completed_tasks.clear()
        self.session_data.clear()

        # Initialize session data structure
        self.session_data = {
            "session_id": config.session_id,
            "participant_id": config.participant_id,
            "protocol_version": config.protocol_version,
            "start_time": None,
            "end_time": None,
            "tasks": {},
            "quality_metrics": {},
            "events": [],
        }

        logger.info(
            f"Session configured: {config.session_id} for participant {config.participant_id}"
        )
        return True

    def start_session(self) -> bool:
        """Start experimental session.

        Returns:
            True if session started successfully
        """
        if self.current_session is None:
            logger.error("No session configured")
            return False

        if not self.state_machine.transition_to(TaskState.INITIALIZING, "Starting session"):
            return False

        try:
            # Initialize components
            self.timer.start_session()

            # Record session start
            self.session_start_time = datetime.now()
            self.session_data["start_time"] = self.session_start_time.isoformat()

            # Start data backup thread
            self._start_data_backup()

            # Transition to ready state
            self.state_machine.transition_to(TaskState.READY, "Session initialized")

            logger.info(f"Session started: {self.current_session.session_id}")
            return True

        except Exception as e:
            self.state_machine.set_error(f"Session start failed: {e}")
            logger.error(f"Failed to start session: {e}")
            return False

    def run_next_task(self) -> Optional[str]:
        """Run the next task in the session.

        Returns:
            Task name if started successfully, None if no more tasks
        """
        if self.current_session is None:
            logger.error("No active session")
            return None

        if self.current_task_index >= len(self.current_session.tasks_to_run):
            logger.info("All tasks completed")
            self.complete_session()
            return None

        task_name = self.current_session.tasks_to_run[self.current_task_index]

        if not self.state_machine.transition_to(TaskState.RUNNING, f"Starting task {task_name}"):
            return None

        try:
            # Initialize task data
            self.session_data["tasks"][task_name] = {
                "start_time": datetime.now().isoformat(),
                "trials": [],
                "parameters": {},
                "quality_metrics": {},
            }

            logger.info(f"Started task: {task_name}")
            return task_name

        except Exception as e:
            self.state_machine.set_error(f"Task start failed: {e}")
            logger.error(f"Failed to start task {task_name}: {e}")
            return None

    def complete_current_task(self) -> bool:
        """Complete the current task and prepare for next.

        Returns:
            True if task completed successfully
        """
        if self.current_session is None:
            logger.error("No active session")
            return False

        if self.current_task_index >= len(self.current_session.tasks_to_run):
            logger.error("No active task to complete")
            return False

        task_name = self.current_session.tasks_to_run[self.current_task_index]

        # Record task completion with thread-safe access
        with self._data_lock:
            self.session_data["tasks"][task_name]["end_time"] = datetime.now().isoformat()
            self.completed_tasks.append(task_name)
            self.current_task_index += 1

        # Add inter-task interval if more tasks remain
        if self.current_task_index < len(self.current_session.tasks_to_run):
            self.state_machine.transition_to(
                TaskState.READY, f"Task {task_name} completed, preparing next"
            )

            # Schedule inter-task interval
            if self.current_session.inter_task_interval_s > 0:
                logger.info(
                    f"Inter-task interval: {self.current_session.inter_task_interval_s:.1f}s"
                )
                time.sleep(self.current_session.inter_task_interval_s)

        logger.info(f"Task completed: {task_name}")
        return True

    def complete_session(self) -> bool:
        """Complete the experimental session.

        Returns:
            True if session completed successfully
        """
        if not self.state_machine.transition_to(TaskState.COMPLETED, "Session completed"):
            return False

        try:
            # Final data save
            self._save_session_data()

            if self.current_session:
                logger.info(f"Session completed: {self.current_session.session_id}")
            else:
                logger.info("Session completed: unknown session")
            return True

        except Exception as e:
            self.state_machine.set_error(f"Session completion failed: {e}")
            logger.error(f"Failed to complete session: {e}")
            return False

    def pause_session(self) -> bool:
        """Pause the current session."""
        if self.state_machine.transition_to(TaskState.PAUSED, "Session paused by user"):
            logger.info("Session paused")
            return True
        return False

    def resume_session(self) -> bool:
        """Resume the paused session."""
        if self.state_machine.transition_to(TaskState.RUNNING, "Session resumed"):
            logger.info("Session resumed")
            return True
        return False

    def abort_session(self, reason: str = "User abort") -> bool:
        """Abort the current session."""
        self.state_machine.set_error(f"Session aborted: {reason}")

        # Stop data backup
        self._stop_data_backup()

        # Save partial data
        if self.session_data:
            self.session_data["aborted"] = True
            self.session_data["abort_reason"] = reason
            self.session_data["abort_time"] = datetime.now().isoformat()
            self._save_session_data()

        logger.warning(f"Session aborted: {reason}")
        return True

    def add_trial_data(self, task_name: str, trial_data: Dict[str, Any]) -> None:
        """Add trial data to session."""
        if task_name in self.session_data["tasks"]:
            self.session_data["tasks"][task_name]["trials"].append(trial_data)
            logger.debug(f"Added trial data for task {task_name}")

    def add_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Add event to session log."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": event_data,
        }
        self.session_data["events"].append(event)
        logger.debug(f"Added event: {event_type}")

    def get_session_progress(self) -> Dict[str, Any]:
        """Get current session progress."""
        if self.current_session is None:
            return {"error": "No active session"}

        total_tasks = len(self.current_session.tasks_to_run)
        completed_tasks = len(self.completed_tasks)

        progress = {
            "session_id": self.current_session.session_id,
            "participant_id": self.current_session.participant_id,
            "current_state": self.state_machine.current_state.value,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "current_task_index": self.current_task_index,
            "progress_percent": ((completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0),
            "session_duration_min": None,
        }

        if self.session_start_time:
            duration = (datetime.now() - self.session_start_time).total_seconds() / 60.0
            progress["session_duration_min"] = duration

        return progress

    def _start_data_backup(self) -> None:
        """Start automatic data backup thread."""
        if self.current_session and self.current_session.backup_interval_min > 0:
            self.stop_backup.clear()
            self.data_backup_thread = threading.Thread(target=self._backup_loop)
            self.data_backup_thread.daemon = True
            self.data_backup_thread.start()
            logger.info(
                f"Data backup started: {self.current_session.backup_interval_min:.1f} min intervals"
            )

    def _backup_loop(self) -> None:
        """Automatic backup loop."""
        if self.current_session is None:
            return

        interval_s = self.current_session.backup_interval_min * 60.0

        while not self.stop_backup.wait(interval_s):
            try:
                self._save_session_data(backup=True)
                logger.debug("Automatic data backup completed")
            except Exception as e:
                logger.error(f"Backup failed: {e}")

    def _stop_data_backup(self) -> None:
        """Stop automatic data backup."""
        self.stop_backup.set()
        if self.data_backup_thread and self.data_backup_thread.is_alive():
            self.data_backup_thread.join(timeout=1.0)
        logger.info("Data backup stopped")

    def _save_session_data(self, backup: bool = False) -> None:
        """Save session data to file."""
        if not self.session_data or self.current_session is None:
            return

        # Create data directory
        data_dir = Path("session_data")
        data_dir.mkdir(exist_ok=True)

        # Generate filename
        suffix = "_backup" if backup else ""
        filename = (
            f"{self.current_session.session_id}_{self.current_session.participant_id}{suffix}.json"
        )
        filepath = data_dir / filename

        # Save data
        with open(filepath, "w") as f:
            json.dump(self.session_data, f, indent=2)

        logger.debug(f"Session data saved: {filepath}")

    def cleanup(self) -> None:
        """Clean up session manager resources."""
        self._stop_data_backup()

        if self.current_session:
            self._save_session_data()

        self.response_collector.stop_collection()

        logger.info("SessionManager cleaned up")
