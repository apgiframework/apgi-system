"""Standardized logging configuration for APGI Framework.

Provides consistent logging across all modules with proper levels,
formatting, and security considerations. Replaces inconsistent print() usage.
"""

import json
import logging
import logging.handlers
import sys
import threading
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union


class CorrelationContext:
    """Thread-safe correlation ID management for distributed tracing.

    Provides a way to track requests across multiple components and services,
    enabling end-to-end observability and debugging.
    """

    _local = threading.local()

    @classmethod
    def get_correlation_id(cls) -> str:
        """Get the current correlation ID or generate a new one."""
        if not hasattr(cls._local, "correlation_id"):
            cls._local.correlation_id = str(uuid.uuid4())
        return str(cls._local.correlation_id)

    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set the correlation ID for the current thread."""
        cls._local.correlation_id = correlation_id

    @classmethod
    def clear_correlation_id(cls) -> None:
        """Clear the correlation ID for the current thread."""
        if hasattr(cls._local, "correlation_id"):
            delattr(cls._local, "correlation_id")

    @classmethod
    @contextmanager
    def correlation_scope(cls, correlation_id: Optional[str] = None) -> Any:
        """Context manager for correlation ID scope.

        Args:
            correlation_id: Optional correlation ID to use. If None, generates new.

        Example:
            with CorrelationContext.correlation_scope() as cid:
                logger.info("Processing request", correlation_id=cid)
        """
        previous_id = cls.get_correlation_id() if hasattr(cls._local, "correlation_id") else None
        new_id = correlation_id or str(uuid.uuid4())
        cls.set_correlation_id(new_id)
        try:
            yield new_id
        finally:
            if previous_id:
                cls.set_correlation_id(previous_id)
            else:
                cls.clear_correlation_id()


class APGILogger:
    """Standardized logger for the APGI Framework.

    Provides consistent logging with security features, structured output,
    and proper error handling to replace inconsistent print() statements.
    """

    def __init__(
        self,
        name: str,
        log_level: str = "INFO",
        log_file: Optional[Union[str, Path]] = None,
        enable_console: bool = True,
        enable_structured: bool = False,
    ):
        """Initialize APGI logger.

        Args:
            name: Logger name (usually module name)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            enable_console: Whether to output to console
            enable_structured: Whether to use JSON structured logging
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.enable_structured = enable_structured

        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Set log level
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Create formatters
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        self.structured_formatter = StructuredFormatter()

        # Add console handler if enabled
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                self.structured_formatter if enable_structured else self.formatter
            )
            self.logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Use rotating file handler to prevent large log files
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5
            )
            file_handler.setFormatter(
                self.structured_formatter if enable_structured else self.formatter
            )
            self.logger.addHandler(file_handler)

        # Prevent propagation to root logger to avoid duplicate messages
        self.logger.propagate = False

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log error message with optional exception details."""
        if exception:
            kwargs["exception_type"] = type(exception).__name__
            kwargs["exception_message"] = str(exception)
            kwargs["traceback"] = traceback.format_exc()
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log critical message with optional exception details."""
        if exception:
            kwargs["exception_type"] = type(exception).__name__
            kwargs["exception_message"] = str(exception)
            kwargs["traceback"] = traceback.format_exc()
        self._log(logging.CRITICAL, message, **kwargs)

    def experiment_start(self, experiment_id: str, parameters: Dict[str, Any]) -> None:
        """Log experiment start."""
        self.info(
            f"Experiment started: {experiment_id}",
            experiment_id=experiment_id,
            parameters=parameters,
            event_type="experiment_start",
        )

    def experiment_end(self, experiment_id: str, status: str, duration: float) -> None:
        """Log experiment end."""
        self.info(
            f"Experiment completed: {experiment_id} ({status})",
            experiment_id=experiment_id,
            status=status,
            duration_seconds=duration,
            event_type="experiment_end",
        )

    def security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security-related event."""
        self.warning(
            f"Security event: {event_type}",
            security_event_type=event_type,
            details=details,
            event_type="security",
        )

    def performance_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """Log performance metric."""
        self.debug(
            f"Performance: {metric_name} = {value} {unit}",
            metric_name=metric_name,
            metric_value=value,
            metric_unit=unit,
            event_type="performance",
        )

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Internal logging method with extra data."""
        extra_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": self.name,
            "correlation_id": CorrelationContext.get_correlation_id(),
            **kwargs,
        }

        if self.enable_structured:
            self.logger.log(level, message, extra={"extra_data": extra_data})
        else:
            # For non-structured logging, add key info to message
            cid = CorrelationContext.get_correlation_id()[:8]  # Short ID for readability
            if kwargs:
                key_info = ", ".join(f"{k}={v}" for k, v in list(kwargs.items())[:3])
                message = f"[{cid}] {message} [{key_info}]"
            else:
                message = f"[{cid}] {message}"
            self.logger.log(level, message)

    @contextmanager
    def log_execution(self, operation: str) -> Any:
        """Context manager to log operation execution time."""
        start_time = datetime.now()
        self.debug(f"Starting operation: {operation}")

        try:
            yield
            duration = (datetime.now() - start_time).total_seconds()
            self.info(
                f"Completed operation: {operation}",
                operation=operation,
                duration_seconds=duration,
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.error(
                f"Failed operation: {operation}",
                exception=e,
                operation=operation,
                duration_seconds=duration,
            )
            raise


class StructuredFormatter(logging.Formatter):
    """JSON structured formatter for machine-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add execution time if available
        if hasattr(record, "execution_time"):
            log_entry["execution_time_seconds"] = record.execution_time.total_seconds()

        # Add extra data if available
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)

        # Ensure correlation_id is always present
        if "correlation_id" not in log_entry:
            log_entry["correlation_id"] = CorrelationContext.get_correlation_id()

        return json.dumps(log_entry)


class LoggingManager:
    """Centralized logging manager for the APGI Framework.

    Manages logger instances, configuration, and provides
    factory methods for consistent logging setup.
    """

    _loggers: Dict[str, APGILogger] = {}
    _global_config: Dict[str, Any] = {}

    @classmethod
    def configure_global_logging(
        cls,
        log_level: str = "INFO",
        log_dir: Optional[Union[str, Path]] = None,
        enable_structured: bool = False,
        enable_console: bool = True,
    ) -> None:
        """Configure global logging settings.

        Args:
            log_level: Default logging level
            log_dir: Directory for log files
            enable_structured: Enable JSON structured logging
            enable_console: Enable console output
        """
        cls._global_config = {
            "log_level": log_level,
            "log_dir": Path(log_dir) if log_dir else None,
            "enable_structured": enable_structured,
            "enable_console": enable_console,
        }

        # Create log directory if specified
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_logger(
        cls, name: str, log_file: Optional[Union[str, Path]] = None, **kwargs: Any
    ) -> APGILogger:
        """Get or create a logger instance.

        Args:
            name: Logger name
            log_file: Optional specific log file
            **kwargs: Additional logger configuration

        Returns:
            APGILogger instance
        """
        if name not in cls._loggers:
            # Merge global config with specific config
            config = cls._global_config.copy()
            config.update(kwargs)

            # Determine log file path
            if log_file is None and config.get("log_dir"):
                log_file = Path(config["log_dir"]) / f"{name.replace('.', '_')}.log"

            # Create logger
            cls._loggers[name] = APGILogger(
                name=name,
                log_level=config.get("log_level", "INFO"),
                log_file=log_file,
                enable_console=config.get("enable_console", True),
                enable_structured=config.get("enable_structured", False),
            )

        return cls._loggers[name]

    @classmethod
    def replace_print_with_logging(cls, module_name: str) -> Any:
        """Replace print() statements with logging for a specific module.

        This is a utility function to help migrate from print() to proper logging.

        Args:
            module_name: Name of the module to modify
        """
        logger = cls.get_logger(module_name)

        # Create a print replacement function
        def safe_print(*args: Any, **kwargs: Any) -> None:
            """Print replacement that logs instead."""
            message = " ".join(str(arg) for arg in args)
            logger.info(message)

        # This would need to be called from the target module
        return safe_print


def get_logger(name: str, **kwargs: Any) -> APGILogger:
    """Convenience function to get a logger.

    Args:
        name: Logger name
        **kwargs: Additional configuration

    Returns:
        APGILogger instance
    """
    return LoggingManager.get_logger(name, **kwargs)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_structured: bool = False,
) -> None:
    """Setup global logging configuration.

    Args:
        log_level: Default logging level
        log_dir: Directory for log files
        enable_structured: Enable JSON structured logging
    """
    LoggingManager.configure_global_logging(
        log_level=log_level, log_dir=log_dir, enable_structured=enable_structured
    )


# Common logger instances for core modules
def get_security_logger() -> APGILogger:
    """Get logger for security-related events."""
    return get_logger("apgi.security")


def get_experiment_logger() -> APGILogger:
    """Get logger for experiment operations."""
    return get_logger("apgi.experiments")


def get_data_logger() -> APGILogger:
    """Get logger for data operations."""
    return get_logger("apgi.data")


def get_gui_logger() -> APGILogger:
    """Get logger for GUI operations."""
    return get_logger("apgi.gui")


def get_analysis_logger() -> APGILogger:
    """Get logger for analysis operations."""
    return get_logger("apgi.analysis")


def get_correlation_id() -> str:
    """Get the current correlation ID for distributed tracing.

    Returns:
        Current correlation ID or newly generated one

    Example:
        cid = get_correlation_id()
        logger.info("Processing request", correlation_id=cid)
        # Pass cid to downstream services for tracing
    """
    return CorrelationContext.get_correlation_id()


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current thread.

    Args:
        correlation_id: Correlation ID to set

    Example:
        # When receiving a request from another service
        set_correlation_id(request.headers.get("X-Correlation-ID"))
        logger.info("Processing incoming request")
    """
    CorrelationContext.set_correlation_id(correlation_id)


__all__ = [
    "APGILogger",
    "CorrelationContext",
    "LoggingManager",
    "StructuredFormatter",
    "get_logger",
    "setup_logging",
    "get_security_logger",
    "get_experiment_logger",
    "get_data_logger",
    "get_gui_logger",
    "get_analysis_logger",
    "get_correlation_id",
    "set_correlation_id",
]
