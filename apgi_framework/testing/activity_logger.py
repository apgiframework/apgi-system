"""Activity Logger for APGI Framework Test Enhancement.

This module provides comprehensive activity logging for all test execution activities
with structured logging, JSON formatting, log rotation, and retention management.
Integrates with all test execution components to provide detailed audit trails.

Requirements: 10.6
"""

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..utils.file_utils import FileUtils
from ..utils.logging_utils import get_logging_utils


class ActivityType(Enum):
    """Types of activities that can be logged."""

    TEST_EXECUTION_START = "test_execution_start"
    TEST_EXECUTION_END = "test_execution_end"
    TEST_CASE_START = "test_case_start"
    TEST_CASE_END = "test_case_end"
    COVERAGE_COLLECTION_START = "coverage_collection_start"
    COVERAGE_COLLECTION_END = "coverage_collection_end"
    TEST_DISCOVERY = "test_discovery"
    TEST_GENERATION = "test_generation"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_EVENT = "system_event"
    USER_ACTION = "user_action"
    CI_INTEGRATION = "ci_integration"
    NOTIFICATION_SENT = "notification_sent"
    BATCH_OPERATION = "batch_operation"
    SYSTEM_SHUTDOWN = "system_shutdown"


class ActivityLevel(Enum):
    """Activity logging levels."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ActivityContext:
    """Context information for an activity."""

    session_id: str
    execution_id: Optional[str] = None
    test_suite: Optional[str] = None
    test_file: Optional[str] = None
    test_name: Optional[str] = None
    component: Optional[str] = None
    user_id: Optional[str] = None
    environment: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivityEntry:
    """Individual activity log entry."""

    activity_id: str
    timestamp: datetime
    activity_type: ActivityType
    level: ActivityLevel
    message: str
    context: ActivityContext
    duration_ms: Optional[float] = None
    data: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    thread_id: int = field(default_factory=lambda: threading.get_ident())
    process_id: int = field(default_factory=os.getpid)

    def get_entries_by_type(self, activity_type: ActivityType) -> List["ActivityEntry"]:
        return [entry for entry in [self] if entry.activity_type == activity_type]


class ActivityBuffer:
    """Thread-safe buffer for activity entries."""

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self.buffer: List[ActivityEntry] = []
        self.lock = threading.Lock()

    def add(self, entry: ActivityEntry) -> None:
        """Add an entry to the buffer."""
        with self.lock:
            self.buffer.append(entry)
            if len(self.buffer) > self.max_size:
                # Remove oldest entries if buffer is full
                self.buffer = self.buffer[-self.max_size :]

    def get_entries_by_level(self, level: ActivityLevel) -> List[ActivityEntry]:
        """Flush and return all buffered entries."""
        with self.lock:
            entries = self.buffer.copy()
            self.buffer.clear()
            return [entry for entry in entries if entry.level == level]

    def flush(self) -> List[ActivityEntry]:
        """Flush and return all buffered entries."""
        with self.lock:
            entries = self.buffer.copy()
            self.buffer.clear()
            return entries

    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)


class ActivityFormatter:
    """Formatter for activity log entries."""

    def __init__(self, structured: bool = True) -> None:
        self.structured = structured

    def format(self, entry: ActivityEntry) -> str:
        """Format an activity entry."""
        if self.structured:
            return self._format_json(entry)
        else:
            return self._format_text(entry)

    def _format_json(self, entry: ActivityEntry) -> str:
        """Format entry as JSON string."""
        data = {
            "activity_id": entry.activity_id,
            "timestamp": entry.timestamp.isoformat(),
            "activity_type": entry.activity_type.value,
            "level": entry.level.value,
            "message": entry.message,
            "context": {
                "session_id": entry.context.session_id,
                "execution_id": entry.context.execution_id,
                "test_suite": entry.context.test_suite,
                "test_file": entry.context.test_file,
                "test_name": entry.context.test_name,
                "component": entry.context.component,
                "user_id": entry.context.user_id,
                "environment": entry.context.environment,
                "metadata": self._serialize_metadata(entry.context.metadata),
            },
            "duration_ms": entry.duration_ms,
            "data": self._serialize_metadata(entry.data),
            "thread_id": entry.thread_id,
            "process_id": entry.process_id,
        }
        if entry.error_info:
            data["error_info"] = entry.error_info
        if entry.stack_trace:
            data["stack_trace"] = entry.stack_trace
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    def export_to_json(self, output_path: Path, entry: ActivityEntry) -> None:
        """Format entry as JSON."""
        data = {
            "activity_id": entry.activity_id,
            "timestamp": entry.timestamp.isoformat(),
            "activity_type": entry.activity_type.value,
            "level": entry.level.value,
            "message": entry.message,
            "context": {
                "session_id": entry.context.session_id,
                "execution_id": entry.context.execution_id,
                "test_suite": entry.context.test_suite,
                "test_file": entry.context.test_file,
                "test_name": entry.context.test_name,
                "component": entry.context.component,
                "user_id": entry.context.user_id,
                "environment": entry.context.environment,
                "metadata": self._serialize_metadata(entry.context.metadata),
            },
            "duration_ms": entry.duration_ms,
            "data": self._serialize_metadata(entry.data),
            "thread_id": entry.thread_id,
            "process_id": entry.process_id,
        }

        if entry.error_info:
            data["error_info"] = entry.error_info

        if entry.stack_trace:
            data["stack_trace"] = entry.stack_trace

        with open(output_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize metadata to JSON-compatible format."""
        serialized = {}
        for key, value in metadata.items():
            try:
                # Test if value is JSON serializable
                json.dumps(value)
                serialized[key] = value
            except (TypeError, ValueError):
                # Convert non-serializable values to strings
                serialized[key] = str(value)
        return serialized

    def export_filtered(
        self,
        output_path: Path,
        filter_func: Callable[[ActivityEntry], bool],
        entry: ActivityEntry,
    ) -> None:
        """Format entry as human-readable text."""
        timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        context_parts = []
        if entry.context.execution_id:
            context_parts.append(f"exec:{entry.context.execution_id[:8]}")
        if entry.context.test_name:
            context_parts.append(f"test:{entry.context.test_name}")
        if entry.context.component:
            context_parts.append(f"comp:{entry.context.component}")

        context_str = f"[{','.join(context_parts)}]" if context_parts else ""

        duration_str = f" ({entry.duration_ms:.1f}ms)" if entry.duration_ms else ""

        if filter_func(entry):
            with open(output_path, "a") as f:
                f.write(
                    f"{timestamp} {entry.level.value:8} {entry.activity_type.value:25} {context_str} {entry.message}{duration_str}\n"
                )

    def _format_text(self, entry: ActivityEntry) -> str:
        """Format entry as human-readable text."""
        timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        context_parts = []
        if entry.context.execution_id:
            context_parts.append(f"exec:{entry.context.execution_id[:8]}")
        if entry.context.test_name:
            context_parts.append(f"test:{entry.context.test_name}")
        if entry.context.component:
            context_parts.append(f"comp:{entry.context.component}")

        context_str = f"[{','.join(context_parts)}]" if context_parts else ""

        duration_str = f" ({entry.duration_ms:.1f}ms)" if entry.duration_ms else ""

        return f"{timestamp} {entry.level.value:8} {entry.activity_type.value:25} {context_str} {entry.message}{duration_str}"


@dataclass
class LoggingConfiguration:
    """Configuration for activity logging."""

    log_directory: Path = field(default_factory=lambda: Path("logs/activity"))
    max_file_size_mb: int = 10
    backup_count: int = 5
    retention_days: int = 30
    enable_console_output: bool = True
    enable_file_output: bool = True
    enable_structured_format: bool = True
    buffer_size: int = 1000
    flush_interval_seconds: float = 5.0
    log_level: ActivityLevel = ActivityLevel.INFO


class ActivityLogger:
    """Comprehensive activity logger for test execution activities.

    Provides structured logging with JSON formatting, log rotation, retention management,
    and integration with all test execution components.
    """

    def __init__(self, config: Optional[LoggingConfiguration] = None) -> None:
        """Initialize the activity logger."""
        self.config = config or LoggingConfiguration()
        self.file_utils = FileUtils()
        self.logging_utils = get_logging_utils()

        # Create log directory
        self.config.log_directory.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.buffer = ActivityBuffer(self.config.buffer_size)
        self.formatter = ActivityFormatter(self.config.enable_structured_format)

        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.current_context = ActivityContext(session_id=self.session_id)

        # Threading
        self.lock = threading.Lock()
        self.flush_thread: Optional[threading.Thread] = None
        self.stop_flush_thread = threading.Event()

        # File handlers
        self.file_handler: Optional[logging.handlers.RotatingFileHandler] = None
        self.console_handler: Optional[logging.StreamHandler] = None

        # Initialize logging
        self._setup_logging()
        self._start_flush_thread()

        # Log initialization
        self.log_activity(
            ActivityType.SYSTEM_EVENT,
            ActivityLevel.INFO,
            "Activity logger initialized",
            data={
                "session_id": self.session_id,
                "config": self._serialize_config(self.config),
            },
        )

    def _serialize_config(self, config: LoggingConfiguration) -> Dict[str, Any]:
        """Serialize configuration to JSON-compatible format."""
        config_dict = asdict(config)
        # Convert Path objects to strings
        if "log_directory" in config_dict:
            config_dict["log_directory"] = str(config_dict["log_directory"])
        return config_dict

    def _setup_logging(self) -> None:
        """Setup file and console logging handlers."""
        # Setup file handler
        if self.config.enable_file_output:
            log_file = self.config.log_directory / "activity.log"

            self.file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count,
                encoding="utf-8",
            )

        # Setup console handler
        if self.config.enable_console_output:
            self.console_handler = logging.StreamHandler(sys.stdout)

    def _start_flush_thread(self) -> None:
        """Start the background flush thread."""
        if self.config.flush_interval_seconds > 0:
            self.flush_thread = threading.Thread(
                target=self._flush_worker, daemon=True, name="ActivityLoggerFlush"
            )
            if self.flush_thread:
                self.flush_thread.start()

    def _flush_worker(self) -> None:
        """Background worker to flush buffered entries."""
        while not self.stop_flush_thread.wait(self.config.flush_interval_seconds):
            try:
                self.flush_buffer()
            except Exception as e:
                # Use standard logging to avoid recursion
                logging.getLogger(__name__).error(f"Error in flush worker: {e}")

    def set_context(self, **kwargs: Any) -> None:
        """Update the current activity context."""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.current_context, key):
                    setattr(self.current_context, key, value)
                else:
                    self.current_context.metadata[key] = value

    def create_context(self, **kwargs: Any) -> ActivityContext:
        """Create a new activity context based on current context."""
        with self.lock:
            new_context = ActivityContext(
                session_id=self.current_context.session_id,
                execution_id=self.current_context.execution_id,
                test_suite=self.current_context.test_suite,
                test_file=self.current_context.test_file,
                test_name=self.current_context.test_name,
                component=self.current_context.component,
                user_id=self.current_context.user_id,
                environment=self.current_context.environment,
                metadata=self.current_context.metadata.copy(),
            )

            # Apply overrides
            for key, value in kwargs.items():
                if hasattr(new_context, key):
                    setattr(new_context, key, value)
                else:
                    new_context.metadata[key] = value

            return new_context

    def log_activity(
        self,
        activity_type: ActivityType,
        level: ActivityLevel,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        context: Optional[ActivityContext] = None,
        duration_ms: Optional[float] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Log an activity entry."""
        # Skip if level is below configured threshold
        level_values = {
            ActivityLevel.TRACE: 0,
            ActivityLevel.DEBUG: 1,
            ActivityLevel.INFO: 2,
            ActivityLevel.WARNING: 3,
            ActivityLevel.ERROR: 4,
            ActivityLevel.CRITICAL: 5,
        }

        if level_values[level] < level_values[self.config.log_level]:
            return

        # Create activity entry
        entry = ActivityEntry(
            activity_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            activity_type=activity_type,
            level=level,
            message=message,
            context=context or self.current_context,
            duration_ms=duration_ms,
            data=data or {},
        )

        # Add error information if provided
        if error:
            entry.error_info = {
                "type": type(error).__name__,
                "message": str(error),
                "args": error.args,
            }
            entry.stack_trace = traceback.format_exc()

        # Add to buffer
        self.buffer.add(entry)

        # Immediate flush for critical errors
        if level == ActivityLevel.CRITICAL:
            self.flush_buffer()

    def clear_buffer(self) -> None:
        """Clear the buffer."""
        self.buffer.buffer.clear()

    def flush_buffer(self) -> None:
        """Flush buffered entries to output handlers."""
        entries = self.buffer.flush()

        if not entries:
            return

        try:
            # Write to file
            if self.file_handler and self.config.enable_file_output:
                for entry in entries:
                    formatted = self.formatter.format(entry)
                    self.file_handler.stream.write(formatted + "\n")
                self.file_handler.flush()

            # Write to console
            if self.console_handler and self.config.enable_console_output:
                text_formatter = ActivityFormatter(structured=False)
                for entry in entries:
                    # Only show INFO and above on console
                    if entry.level in [
                        ActivityLevel.INFO,
                        ActivityLevel.WARNING,
                        ActivityLevel.ERROR,
                        ActivityLevel.CRITICAL,
                    ]:
                        formatted = text_formatter.format(entry)
                        try:
                            self.console_handler.stream.write(formatted + "\n")
                        except ValueError:
                            # Stream closed, disable console output
                            self.config.enable_console_output = False
                            break
                try:
                    self.console_handler.flush()
                except ValueError:
                    # Stream closed, disable console output
                    self.config.enable_console_output = False

        except Exception as e:
            # Use standard logging to avoid recursion
            logging.getLogger(__name__).error(f"Error flushing activity log: {e}")

    @contextmanager
    def activity_span(
        self,
        activity_type: ActivityType,
        message: str,
        level: ActivityLevel = ActivityLevel.INFO,
        context: Optional[ActivityContext] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Context manager for timing activities."""
        start_time = time.time()
        span_context = context or self.current_context

        # Log start
        self.log_activity(
            activity_type, level, f"Started: {message}", context=span_context, data=data
        )

        try:
            yield span_context

            # Log successful completion
            duration_ms = (time.time() - start_time) * 1000
            self.log_activity(
                activity_type,
                level,
                f"Completed: {message}",
                context=span_context,
                duration_ms=duration_ms,
                data=data,
            )

        except Exception as e:
            # Log error
            duration_ms = (time.time() - start_time) * 1000
            self.log_activity(
                ActivityType.ERROR_OCCURRED,
                ActivityLevel.ERROR,
                f"Failed: {message}",
                context=span_context,
                duration_ms=duration_ms,
                data=data,
                error=e,
            )
            raise

    def log_test_execution_start(
        self,
        execution_id: str,
        test_suite: str,
        total_tests: int,
        configuration: Dict[str, Any],
    ) -> None:
        """Log test execution start."""
        context = self.create_context(execution_id=execution_id, test_suite=test_suite)

        self.log_activity(
            ActivityType.TEST_EXECUTION_START,
            ActivityLevel.INFO,
            f"Starting test execution: {test_suite}",
            context=context,
            data={"total_tests": total_tests, "configuration": configuration},
        )

    def log_test_execution_end(
        self,
        execution_id: str,
        test_suite: str,
        results: Dict[str, Any],
        duration_ms: float,
    ) -> None:
        """Log test execution end."""
        context = self.create_context(execution_id=execution_id, test_suite=test_suite)

        level = ActivityLevel.ERROR if results.get("failed", 0) > 0 else ActivityLevel.INFO

        self.log_activity(
            ActivityType.TEST_EXECUTION_END,
            level,
            f"Completed test execution: {test_suite}",
            context=context,
            duration_ms=duration_ms,
            data=results,
        )

    def log_test_case_start(self, test_name: str, test_file: str) -> None:
        """Log individual test case start."""
        context = self.create_context(test_name=test_name, test_file=test_file)

        self.log_activity(
            ActivityType.TEST_CASE_START,
            ActivityLevel.DEBUG,
            f"Starting test: {test_name}",
            context=context,
        )

    def log_test_case_end(
        self,
        test_name: str,
        test_file: str,
        status: str,
        duration_ms: float,
        error_message: Optional[str] = None,
    ) -> None:
        """Log individual test case end."""
        context = self.create_context(test_name=test_name, test_file=test_file)

        level = ActivityLevel.ERROR if status == "failed" else ActivityLevel.DEBUG

        self.log_activity(
            ActivityType.TEST_CASE_END,
            level,
            f"Completed test: {test_name} ({status})",
            context=context,
            duration_ms=duration_ms,
            data={"status": status, "error_message": error_message},
        )

    def log_coverage_collection(
        self, module: str, coverage_data: Dict[str, Any], duration_ms: float
    ) -> None:
        """Log coverage collection activity."""
        self.log_activity(
            ActivityType.COVERAGE_COLLECTION_END,
            ActivityLevel.INFO,
            f"Collected coverage for {module}",
            duration_ms=duration_ms,
            data=coverage_data,
        )

    def log_test_discovery(
        self,
        discovered_tests: List[str],
        discovery_time_ms: float,
        patterns: Optional[List[str]] = None,
    ) -> None:
        """Log test discovery activity."""
        self.log_activity(
            ActivityType.TEST_DISCOVERY,
            ActivityLevel.INFO,
            f"Discovered {len(discovered_tests)} tests",
            duration_ms=discovery_time_ms,
            data={
                "test_count": len(discovered_tests),
                "tests": discovered_tests[:10],  # Limit for log size
                "patterns": patterns,
            },
        )

    def log_test_generation(
        self, component: str, generated_tests: List[str], generation_time_ms: float
    ) -> None:
        """Log test generation activity."""
        context = self.create_context(component=component)

        self.log_activity(
            ActivityType.TEST_GENERATION,
            ActivityLevel.INFO,
            f"Generated {len(generated_tests)} tests for {component}",
            context=context,
            duration_ms=generation_time_ms,
            data={
                "test_count": len(generated_tests),
                "generated_tests": generated_tests,
            },
        )

    def log_performance_metric(
        self, metric_name: str, value: float, unit: str, component: Optional[str] = None
    ) -> None:
        """Log performance metric."""
        context = self.create_context(component=component) if component else None

        self.log_activity(
            ActivityType.PERFORMANCE_METRIC,
            ActivityLevel.DEBUG,
            f"Performance metric: {metric_name} = {value} {unit}",
            context=context,
            data={"metric_name": metric_name, "value": value, "unit": unit},
        )

    def log_configuration_change(
        self, component: str, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> None:
        """Log configuration change."""
        context = self.create_context(component=component)

        self.log_activity(
            ActivityType.CONFIGURATION_CHANGE,
            ActivityLevel.INFO,
            f"Configuration changed for {component}",
            context=context,
            data={"old_config": old_config, "new_config": new_config},
        )

    def log_user_action(
        self,
        action: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log user action."""
        context = self.create_context(user_id=user_id) if user_id else None

        self.log_activity(
            ActivityType.USER_ACTION,
            ActivityLevel.INFO,
            f"User action: {action}",
            context=context,
            data=details or {},
        )

    def log_ci_integration(self, pipeline_type: str, event: str, details: Dict[str, Any]) -> None:
        """Log CI/CD integration event."""
        self.log_activity(
            ActivityType.CI_INTEGRATION,
            ActivityLevel.INFO,
            f"CI event: {event} ({pipeline_type})",
            data={"pipeline_type": pipeline_type, "event": event, **details},
        )

    def log_notification_sent(
        self, channel: str, notification_type: str, recipient_count: int, success: bool
    ) -> None:
        """Log notification sending."""
        level = ActivityLevel.INFO if success else ActivityLevel.WARNING

        self.log_activity(
            ActivityType.NOTIFICATION_SENT,
            level,
            f"Notification sent via {channel}: {notification_type}",
            data={
                "channel": channel,
                "notification_type": notification_type,
                "recipient_count": recipient_count,
                "success": success,
            },
        )

    def log_error(
        self,
        component: str,
        error: Exception,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log error with full context."""
        context = self.create_context(component=component)

        self.log_activity(
            ActivityType.ERROR_OCCURRED,
            ActivityLevel.ERROR,
            f"Error in {component}: {str(error)}",
            context=context,
            data=context_data or {},
            error=error,
        )

    def cleanup_old_logs(self) -> int:
        """Clean up old log files based on retention policy."""
        if self.config.retention_days <= 0:
            return 0

        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        cleaned_count = 0

        try:
            for log_file in self.config.log_directory.glob("*.log*"):
                if log_file.is_file():
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        try:
                            log_file.unlink()
                            cleaned_count += 1
                        except Exception as e:
                            logging.getLogger(__name__).error(
                                f"Failed to clean up log file {log_file}: {e}"
                            )

            if cleaned_count > 0:
                self.log_activity(
                    ActivityType.SYSTEM_EVENT,
                    ActivityLevel.INFO,
                    f"Cleaned up {cleaned_count} old log files",
                    data={
                        "cleaned_count": cleaned_count,
                        "retention_days": self.config.retention_days,
                    },
                )

        except Exception as e:
            logging.getLogger(__name__).error(f"Error during log cleanup: {e}")

        return cleaned_count

    def get_statistics(self) -> Dict[str, Any]:
        """Get activity logging statistics."""
        stats = {
            "session_id": self.session_id,
            "buffer_size": self.buffer.get_buffer_size(),
            "configuration": self._serialize_config(self.config),
            "log_files": [],
        }

        # Add log file information
        try:
            for log_file in self.config.log_directory.glob("*.log*"):
                if log_file.is_file():
                    file_stats = log_file.stat()
                    stats["log_files"].append(  # type: ignore
                        {
                            "name": log_file.name,
                            "size": file_stats.st_size,
                            "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        }
                    )
        except Exception as e:
            stats["log_files_error"] = str(e)

        return stats

    def shutdown(self) -> None:
        """Shutdown the activity logger."""
        self.log_activity(
            ActivityType.SYSTEM_EVENT,
            ActivityLevel.INFO,
            "Activity logger shutting down",
        )

        # Stop flush thread
        if self.flush_thread:
            self.stop_flush_thread.set()
            self.flush_thread.join(timeout=5)

        # Final flush
        self.flush_buffer()

        # Close handlers
        if self.file_handler:
            self.file_handler.close()
        if self.console_handler:
            self.console_handler.close()


# Global activity logger instance
_activity_logger: Optional[ActivityLogger] = None
_logger_lock = threading.Lock()


def get_activity_logger(
    config: Optional[LoggingConfiguration] = None,
) -> ActivityLogger:
    """Get the global activity logger instance."""
    global _activity_logger

    with _logger_lock:
        if _activity_logger is None:
            _activity_logger = ActivityLogger(config)
        return _activity_logger


def initialize_activity_logging(config: Optional[LoggingConfiguration] = None) -> None:
    """Initialize global activity logging."""
    global _activity_logger

    with _logger_lock:
        if _activity_logger is not None:
            _activity_logger.shutdown()
        _activity_logger = ActivityLogger(config)


def shutdown_activity_logging() -> None:
    """Shutdown global activity logging."""
    global _activity_logger

    with _logger_lock:
        if _activity_logger is not None:
            _activity_logger.shutdown()
            _activity_logger = None


# Convenience functions for common logging operations
def log_test_execution_start(
    execution_id: str, test_suite: str, total_tests: int, configuration: Dict[str, Any]
) -> None:
    """Convenience function to log test execution start."""
    get_activity_logger().log_test_execution_start(
        execution_id, test_suite, total_tests, configuration
    )


def log_test_execution_end(
    execution_id: str, test_suite: str, results: Dict[str, Any], duration_ms: float
) -> None:
    """Convenience function to log test execution end."""
    get_activity_logger().log_test_execution_end(execution_id, test_suite, results, duration_ms)


def log_test_case_start(test_name: str, test_file: str) -> None:
    """Convenience function to log test case start."""
    get_activity_logger().log_test_case_start(test_name, test_file)


def log_test_case_end(
    test_name: str,
    test_file: str,
    status: str,
    duration_ms: float,
    error_message: Optional[str] = None,
) -> None:
    """Convenience function to log test case end."""
    get_activity_logger().log_test_case_end(
        test_name, test_file, status, duration_ms, error_message
    )


def log_coverage_collection(module: str, coverage_data: Dict[str, Any], duration_ms: float) -> None:
    """Convenience function to log coverage collection."""
    get_activity_logger().log_coverage_collection(module, coverage_data, duration_ms)


def log_error(
    component: str, error: Exception, context_data: Optional[Dict[str, Any]] = None
) -> None:
    """Convenience function to log errors."""
    get_activity_logger().log_error(component, error, context_data)


def activity_span(
    activity_type: ActivityType, message: str, level: ActivityLevel = ActivityLevel.INFO
) -> Any:
    """Convenience function for activity spans."""
    return get_activity_logger().activity_span(activity_type, message, level)
