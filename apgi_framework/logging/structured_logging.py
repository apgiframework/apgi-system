"""Structured Logging with Correlation IDs.

Provides standardized structured logs with trace IDs/correlation IDs across
API and async tasks for end-to-end observability.
"""

import json
import logging
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .standardized_logging import CorrelationContext


class StructuredLogFormatter(logging.Formatter):
    """Formats logs as structured JSON with correlation IDs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": CorrelationContext.get_correlation_id(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class StructuredLogger:
    """Logger with structured output and correlation ID support."""

    def __init__(self, name: str, enable_structured: bool = True):
        """Initialize structured logger.

        Args:
            name: Logger name
            enable_structured: Whether to use JSON structured output
        """
        self.logger = logging.getLogger(name)
        self.enable_structured = enable_structured

        if enable_structured:
            # Add structured formatter
            handler = logging.StreamHandler()
            handler.setFormatter(StructuredLogFormatter())
            self.logger.addHandler(handler)

    def _add_correlation_id(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add correlation ID to extra fields."""
        if extra is None:
            extra = {}
        extra["correlation_id"] = CorrelationContext.get_correlation_id()
        return extra

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        extra = self._add_correlation_id(kwargs)
        self.logger.debug(message, extra={"extra_fields": extra})

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        extra = self._add_correlation_id(kwargs)
        self.logger.info(message, extra={"extra_fields": extra})

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        extra = self._add_correlation_id(kwargs)
        self.logger.warning(message, extra={"extra_fields": extra})

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        extra = self._add_correlation_id(kwargs)
        self.logger.error(message, extra={"extra_fields": extra})

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        extra = self._add_correlation_id(kwargs)
        self.logger.critical(message, extra={"extra_fields": extra})


class RequestContextManager:
    """Manages request context with correlation IDs."""

    _local = threading.local()

    @classmethod
    def set_request_context(
        cls,
        request_id: str,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
    ) -> None:
        """Set request context.

        Args:
            request_id: Unique request ID
            user_id: User making request
            endpoint: API endpoint
            method: HTTP method
        """
        cls._local.request_context = {
            "request_id": request_id,
            "user_id": user_id,
            "endpoint": endpoint,
            "method": method,
            "start_time": datetime.now(timezone.utc),
        }
        CorrelationContext.set_correlation_id(request_id)

    @classmethod
    def get_request_context(cls) -> Optional[Dict[str, Any]]:
        """Get current request context."""
        return getattr(cls._local, "request_context", None)

    @classmethod
    def clear_request_context(cls) -> None:
        """Clear request context."""
        if hasattr(cls._local, "request_context"):
            delattr(cls._local, "request_context")
        CorrelationContext.clear_correlation_id()

    @classmethod
    @contextmanager
    def request_scope(
        cls,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
    ):
        """Context manager for request scope.

        Args:
            user_id: User making request
            endpoint: API endpoint
            method: HTTP method

        Example:
            with RequestContextManager.request_scope(user_id="user123", endpoint="/api/data"):
                logger.info("Processing request")
        """
        request_id = str(uuid.uuid4())
        cls.set_request_context(request_id, user_id, endpoint, method)
        try:
            yield request_id
        finally:
            cls.clear_request_context()


class TaskContextManager:
    """Manages async task context with correlation IDs."""

    _local = threading.local()

    @classmethod
    def set_task_context(
        cls,
        task_id: str,
        task_name: str,
        parent_request_id: Optional[str] = None,
    ) -> None:
        """Set task context.

        Args:
            task_id: Unique task ID
            task_name: Name of task
            parent_request_id: Optional parent request ID for tracing
        """
        cls._local.task_context = {
            "task_id": task_id,
            "task_name": task_name,
            "parent_request_id": parent_request_id,
            "start_time": datetime.now(timezone.utc),
        }
        # Use parent request ID if available, otherwise use task ID
        correlation_id = parent_request_id or task_id
        CorrelationContext.set_correlation_id(correlation_id)

    @classmethod
    def get_task_context(cls) -> Optional[Dict[str, Any]]:
        """Get current task context."""
        return getattr(cls._local, "task_context", None)

    @classmethod
    def clear_task_context(cls) -> None:
        """Clear task context."""
        if hasattr(cls._local, "task_context"):
            delattr(cls._local, "task_context")
        CorrelationContext.clear_correlation_id()

    @classmethod
    @contextmanager
    def task_scope(
        cls,
        task_name: str,
        parent_request_id: Optional[str] = None,
    ):
        """Context manager for task scope.

        Args:
            task_name: Name of task
            parent_request_id: Optional parent request ID

        Example:
            with TaskContextManager.task_scope("data_export", parent_request_id=request_id):
                logger.info("Starting async task")
        """
        task_id = str(uuid.uuid4())
        cls.set_task_context(task_id, task_name, parent_request_id)
        try:
            yield task_id
        finally:
            cls.clear_task_context()


def get_structured_logger(name: str) -> StructuredLogger:
    """Get structured logger instance."""
    return StructuredLogger(name)
