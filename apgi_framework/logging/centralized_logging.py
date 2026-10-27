"""Centralized logging configuration for APGI Framework.

This module provides a single point of configuration for all logging
across the framework to eliminate redundancy and ensure consistency.
"""

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional


class APGILogManager:
    """Centralized logging manager for APGI Framework."""

    _configured = False
    _loggers: dict[str, logging.Logger] = {}

    @classmethod
    def setup_logging(
        cls,
        level: str = "INFO",
        log_file: Optional[str] = None,
        format_string: Optional[str] = None,
    ) -> None:
        """Setup centralized logging configuration.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for log output
            format_string: Custom format string
        """
        if cls._configured:
            return  # Already configured

        # Default format
        if format_string is None:
            format_string = (
                "%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
            )

        # Configure root logger
        handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]

        # Add file handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(getattr(logging, level.upper()))
            handlers.append(file_handler)

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=format_string,
            handlers=handlers,
            force=True,  # Override any existing configuration
        )

        cls._configured = True

    @classmethod
    def get_logger(cls, name: str, level: Optional[str] = None) -> logging.Logger:
        """Get a logger with consistent configuration.

        Args:
            name: Logger name
            level: Optional override level for this specific logger

        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]

        # Ensure logging is configured
        if not cls._configured:
            cls.setup_logging()

        logger = logging.getLogger(name)

        # Set specific level if provided
        if level:
            logger.setLevel(getattr(logging, level.upper()))

        cls._loggers[name] = logger
        return logger

    @classmethod
    def reset(cls) -> None:
        """Reset logging configuration (useful for testing)."""
        cls._configured = False
        cls._loggers.clear()
        logging.getLogger().handlers.clear()


# Convenience function for backward compatibility
def get_logger(
    name: str, level: Optional[str] = None, log_file: Optional[str] = None
) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name
        level: Optional logging level override
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    # Setup logging if file is provided (first time setup)
    if log_file and not APGILogManager._configured:
        APGILogManager.setup_logging(log_file=log_file)

    return APGILogManager.get_logger(name, level)


class StructuredAPGILogger:
    """Structured logger that outputs JSON-formatted log entries and supports correlation IDs."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def _format_log_entry(self, level: str, message: str, **kwargs: Any) -> str:
        """Format log entry as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "logger": self.logger.name,
            "message": message,
            "correlation_id": kwargs.pop("correlation_id", str(uuid.uuid4())),
            **kwargs,
        }
        # exception classes, paths and other objects must not make logging itself fail
        return json.dumps(log_entry, default=str)

    def info(self, message: str, **kwargs: Any) -> None:
        self.logger.info(self._format_log_entry("INFO", message, **kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        self.logger.warning(self._format_log_entry("WARNING", message, **kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        self.logger.error(self._format_log_entry("ERROR", message, **kwargs))

    def debug(self, message: str, **kwargs: Any) -> None:
        self.logger.debug(self._format_log_entry("DEBUG", message, **kwargs))

    def critical(self, message: str, **kwargs: Any) -> None:
        self.logger.critical(self._format_log_entry("CRITICAL", message, **kwargs))

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log at ERROR level with the active exception's traceback."""
        self.logger.exception(self._format_log_entry("ERROR", message, **kwargs))


# Convenience function for structured logger
def get_structured_logger(name: str) -> StructuredAPGILogger:
    """Get a structured logger instance."""
    return StructuredAPGILogger(name)


# Auto-configure with sensible defaults when module is imported
try:
    APGILogManager.setup_logging(level="INFO")
except Exception:
    # Fallback to basic configuration if there's an issue
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
