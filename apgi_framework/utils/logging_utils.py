"""Centralized logging utilities for the APGI Framework test enhancement system.

This module provides structured logging with appropriate levels, log rotation
and retention management, and component-specific loggers for comprehensive
activity logging and audit capabilities.
"""

import json
import logging
import logging.handlers
import sys
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from apgi_framework.logging.standardized_logging import get_logger

from .file_utils import FileUtils
from .path_utils import get_path_manager

logger = get_logger(__name__)


class LogLevel(Enum):
    """Logging levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogEntry:
    """Structured log entry."""

    timestamp: datetime
    level: str
    logger_name: str
    component: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    extra_data: Dict[str, Any]
    exception_info: Optional[str] = None


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""

    def __init__(self, include_extra: bool = True):
        """Initialize structured formatter.

        Args:
            include_extra: Whether to include extra fields in output
        """
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Create base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "component": getattr(record, "component", "unknown"),
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if requested
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "getMessage",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "component",
                ]:
                    try:
                        # Only include JSON-serializable values
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)

            if extra_fields:
                log_entry["extra"] = extra_fields

        return json.dumps(log_entry, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def __init__(self) -> None:
        """Initialize colored formatter."""
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get color for level
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Format component
        component = getattr(record, "component", record.name)

        # Create formatted message
        formatted = (
            f"{color}[{timestamp}] {record.levelname:8} {component:15} {record.getMessage()}{reset}"
        )

        # Add exception information if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


class LoggingUtils:
    """Centralized logging utilities with structured logging and rotation.

    Provides comprehensive logging functionality including structured logging,
    log rotation, retention management, and component-specific loggers.
    """

    def __init__(self, base_log_dir: Optional[Union[str, Path]] = None):
        """Initialize logging utilities.

        Args:
            base_log_dir: Base directory for log files
        """
        self.file_utils = FileUtils()
        self.path_manager = get_path_manager()

        # Set up log directory
        if base_log_dir:
            self.log_dir = self.path_manager.resolve_path(base_log_dir)
        else:
            self.log_dir = self.path_manager.get_dir("logs")

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.default_level = LogLevel.INFO
        self.console_level = LogLevel.INFO
        self.file_level = LogLevel.DEBUG

        # Rotation settings
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.backup_count = 5
        self.retention_days = 30

        # Component loggers
        self._component_loggers: Dict[str, logging.Logger] = {}
        self._handlers: Dict[str, logging.Handler] = {}

        # Thread lock for thread-safe operations
        self._lock = threading.Lock()

        # Initialize root logger
        self._setup_root_logger()

    def _setup_root_logger(self) -> None:
        """Set up the root logger configuration."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add console handler
        self._add_console_handler(root_logger)

        # Add file handler
        self._add_file_handler(root_logger, "apgi_framework.log")

    def _add_console_handler(self, logger: logging.Logger) -> None:
        """Add colored console handler to logger."""
        if "console" not in self._handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.console_level.value)
            console_handler.setFormatter(ColoredConsoleFormatter())
            self._handlers["console"] = console_handler

        logger.addHandler(self._handlers["console"])

    def _add_file_handler(self, logger: logging.Logger, filename: str) -> None:
        """Add rotating file handler to logger."""
        handler_key = f"file_{filename}"

        if handler_key not in self._handlers:
            log_file = self.log_dir / filename

            # Use rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(self.file_level.value)
            file_handler.setFormatter(StructuredFormatter())
            self._handlers[handler_key] = file_handler

        logger.addHandler(self._handlers[handler_key])

    def get_logger(
        self,
        component: str,
        log_file: Optional[str] = None,
        level: Optional[LogLevel] = None,
    ) -> logging.Logger:
        """Get or create component-specific logger.

        Args:
            component: Component name for the logger
            log_file: Optional separate log file for this component
            level: Optional logging level override

        Returns:
            Configured logger instance
        """
        with self._lock:
            logger_name = f"apgi.{component}"

            if logger_name not in self._component_loggers:
                logger = logging.getLogger(logger_name)
                logger.setLevel((level or self.default_level).value)

                # Don't propagate to avoid duplicate messages
                logger.propagate = False

                # Add console handler
                self._add_console_handler(logger)

                # Add component-specific file handler if requested
                if log_file:
                    self._add_file_handler(logger, log_file)
                else:
                    # Use default file handler
                    self._add_file_handler(logger, "apgi_framework.log")

                self._component_loggers[logger_name] = logger

            return self._component_loggers[logger_name]

    def log_structured(
        self, component: str, level: LogLevel, message: str, **extra_data: Any
    ) -> None:
        """Log structured message with extra data.

        Args:
            component: Component name
            level: Log level
            message: Log message
            **extra_data: Additional structured data
        """
        logger = self.get_logger(component)

        # Add component to extra data
        extra_data["component"] = component

        # Log with extra data
        logger.log(level.value, message, extra=extra_data)

    def log_test_execution(
        self,
        test_name: str,
        status: str,
        duration: float,
        component: str = "test_execution",
        **extra_data: Any,
    ) -> None:
        """Log test execution with structured data.

        Args:
            test_name: Name of the test
            status: Test execution status
            duration: Test duration in seconds
            component: Component name
            **extra_data: Additional test data
        """
        self.log_structured(
            component=component,
            level=LogLevel.INFO,
            message=f"Test {test_name} {status} in {duration:.3f}s",
            test_name=test_name,
            test_status=status,
            test_duration=duration,
            **extra_data,
        )

    def log_coverage_data(
        self,
        module: str,
        line_coverage: float,
        branch_coverage: float,
        component: str = "coverage",
        **extra_data: Any,
    ) -> None:
        """Log coverage data with structured information.

        Args:
            module: Module name
            line_coverage: Line coverage percentage
            branch_coverage: Branch coverage percentage
            component: Component name
            **extra_data: Additional coverage data
        """
        self.log_structured(
            component=component,
            level=LogLevel.INFO,
            message=f"Coverage for {module}: {line_coverage:.1f}% lines, {branch_coverage:.1f}% branches",
            module=module,
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            **extra_data,
        )

    def log_error_with_context(
        self,
        component: str,
        error: Exception,
        context: Dict[str, Any],
        message: Optional[str] = None,
    ) -> None:
        """Log error with full context and traceback.

        Args:
            component: Component name
            error: Exception that occurred
            context: Additional context data
            message: Optional custom message
        """
        logger = self.get_logger(component)

        error_message = message or f"Error in {component}: {str(error)}"

        # Prepare context data
        error_context = {
            "component": component,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
        }

        logger.error(error_message, extra=error_context, exc_info=True)

    def log_performance_metric(
        self,
        operation: str,
        duration: float,
        component: str = "performance",
        **extra_data: Any,
    ) -> None:
        """Log performance metrics.

        Args:
            operation: Operation name
            duration: Operation duration in seconds
            component: Component name
            **extra_data: Additional performance data
        """
        self.log_structured(
            component=component,
            level=LogLevel.INFO,
            message=f"Performance: {operation} took {duration:.3f}s",
            operation=operation,
            duration=duration,
            **extra_data,
        )

    def configure_logging(self, config: Dict[str, Any]) -> None:
        """Configure logging from configuration dictionary.

        Args:
            config: Logging configuration
        """
        # Update levels
        if "default_level" in config:
            self.default_level = LogLevel(config["default_level"])

        if "console_level" in config:
            self.console_level = LogLevel(config["console_level"])

        if "file_level" in config:
            self.file_level = LogLevel(config["file_level"])

        # Update rotation settings
        if "max_file_size" in config:
            self.max_file_size = config["max_file_size"]

        if "backup_count" in config:
            self.backup_count = config["backup_count"]

        if "retention_days" in config:
            self.retention_days = config["retention_days"]

        # Reconfigure existing loggers
        self._reconfigure_loggers()

    def _reconfigure_loggers(self) -> None:
        """Reconfigure existing loggers with new settings."""
        with self._lock:
            # Clear existing handlers
            for handler in self._handlers.values():
                handler.close()
            self._handlers.clear()

            # Reconfigure root logger
            self._setup_root_logger()

            # Reconfigure component loggers
            for logger_name, logger in self._component_loggers.items():
                # Remove existing handlers
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)

                # Add new handlers
                self._add_console_handler(logger)
                self._add_file_handler(logger, "apgi_framework.log")

    def cleanup_old_logs(self) -> int:
        """Clean up old log files based on retention policy.

        Returns:
            Number of files cleaned up
        """
        if self.retention_days <= 0:
            return 0

        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        cleaned_count = 0

        try:
            for log_file in self.log_dir.glob("*.log*"):
                if log_file.is_file():
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        try:
                            log_file.unlink()
                            cleaned_count += 1
                        except Exception as e:
                            # Log cleanup failure to console only to avoid recursion
                            logger.info(f"Failed to clean up log file {log_file}: {e}")
        except Exception as e:
            logger.info(f"Error during log cleanup: {e}")

        if cleaned_count > 0:
            self.log_structured(
                component="logging",
                level=LogLevel.INFO,
                message=f"Cleaned up {cleaned_count} old log files",
                cleaned_count=cleaned_count,
                retention_days=self.retention_days,
            )

        return cleaned_count

    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics and information.

        Returns:
            Dictionary with logging statistics
        """
        stats = {
            "log_directory": str(self.log_dir),
            "active_loggers": len(self._component_loggers),
            "active_handlers": len(self._handlers),
            "configuration": {
                "default_level": self.default_level.name,
                "console_level": self.console_level.name,
                "file_level": self.file_level.name,
                "max_file_size": self.max_file_size,
                "backup_count": self.backup_count,
                "retention_days": self.retention_days,
            },
        }

        # Add log file information
        log_files = []
        try:
            for log_file in self.log_dir.glob("*.log*"):
                if log_file.is_file():
                    file_stats = log_file.stat()
                    log_files.append(
                        {
                            "name": log_file.name,
                            "size": file_stats.st_size,
                            "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        }
                    )
        except Exception as e:
            stats["log_files_error"] = str(e)

        stats["log_files"] = log_files
        stats["total_log_size"] = sum(f["size"] for f in log_files)  # type: ignore

        return stats

    def export_logs(
        self,
        output_path: Union[str, Path],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        components: Optional[List[str]] = None,
        levels: Optional[List[LogLevel]] = None,
    ) -> Path:
        """Export filtered logs to file.

        Yields:
            Path to exported file
        """
        filtered_entries = []

        # Read all log files
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.is_file() and not log_file.name.endswith(".gz"):
                try:
                    content = self.file_utils.safe_read_text(log_file)

                    for line in content.splitlines():
                        if line.strip():
                            try:
                                entry = json.loads(line)

                                # Apply filters
                                entry_date = datetime.fromisoformat(entry["timestamp"])

                                if start_date and entry_date < start_date:
                                    continue
                                if end_date and entry_date > end_date:
                                    continue
                                if components and entry.get("component") not in components:
                                    continue
                                if levels and entry.get("level") not in [
                                    level.name for level in levels
                                ]:
                                    continue

                                filtered_entries.append(entry)

                            except (json.JSONDecodeError, KeyError, ValueError):
                                # Skip malformed entries
                                continue

                except Exception as e:
                    logger.info(f"Error reading log file {log_file}: {e}")

        # Sort by timestamp
        filtered_entries.sort(key=lambda x: x["timestamp"])

        # Export to file
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "filters": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "components": components,
                "levels": [level.name for level in levels] if levels else None,
            },
            "entry_count": len(filtered_entries),
            "entries": filtered_entries,
        }

        output_file = self.file_utils.write_json(output_path, export_data, indent=2)

        self.log_structured(
            component="logging",
            level=LogLevel.INFO,
            message=f"Exported {len(filtered_entries)} log entries to {output_file}",
            exported_entries=len(filtered_entries),
            output_file=str(output_file),
        )

        return output_file


# Global logging utilities instance
_logging_utils: Optional[LoggingUtils] = None
_logging_lock = threading.Lock()


def get_logging_utils(base_log_dir: Optional[Union[str, Path]] = None) -> LoggingUtils:
    """Get global logging utilities instance.

    Args:
        base_log_dir: Base directory for log files

    Returns:
        LoggingUtils instance
    """
    global _logging_utils
    with _logging_lock:
        should_create = _logging_utils is None
        if base_log_dir is not None:
            should_create = True
        if should_create:
            _logging_utils = LoggingUtils(base_log_dir)
    return _logging_utils  # type: ignore[return-value]


def get_component_logger(
    component: str, log_file: Optional[str] = None, level: Optional[LogLevel] = None
) -> logging.Logger:
    """Convenience function to get component logger.

    Args:
        component: Component name
        log_file: Optional separate log file
        level: Optional logging level

    Returns:
        Configured logger
    """
    return get_logging_utils().get_logger(component, log_file, level)


def log_test_activity(test_name: str, status: str, duration: float, **extra_data: Any) -> None:
    """Convenience function to log test activity."""
    get_logging_utils().log_test_execution(test_name, status, duration, **extra_data)


def log_coverage_info(
    module: str,
    line_coverage: float,
    branch_coverage: float,
    **extra_data: Any,
) -> None:
    """Convenience function to log coverage information."""
    get_logging_utils().log_coverage_data(module, line_coverage, branch_coverage, **extra_data)


def log_error(
    component: str,
    error: Exception,
    context: Dict[str, Any],
    message: Optional[str] = None,
) -> None:
    """Convenience function to log errors with context."""
    get_logging_utils().log_error_with_context(component, error, context, message)


def setup_logging(
    level: Union[str, LogLevel] = LogLevel.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    structured_format: bool = False,
) -> LoggingUtils:
    """Setup logging configuration for the application.

    Args:
        level: Logging level (string or LogLevel enum)
        log_file: Optional log file path
        console_output: Whether to output to console
        structured_format: Whether to use structured JSON format

    Returns:
        Configured LoggingUtils instance
    """
    # Convert string level to LogLevel if needed
    if isinstance(level, str):
        level = LogLevel[level.upper()]

    # Get or create logging utils
    log_dir = None
    if log_file:
        log_file_path = Path(log_file)
        log_dir = log_file_path.parent

    logging_utils = get_logging_utils(log_dir)

    # Configure logging
    config = {
        "default_level": level.value,
        "console_level": level.value if console_output else logging.CRITICAL,
        "file_level": level.value,
    }

    logging_utils.configure_logging(config)

    return logging_utils
