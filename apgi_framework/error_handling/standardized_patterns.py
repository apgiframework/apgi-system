"""Standardized Error Handling Patterns for APGI Framework
===============================================

This module provides standardized error handling patterns and utilities for the APGI Framework,
ensuring consistent error reporting, logging, and user-friendly error messages across all modules.

Features:
- Unified error handling interface
- Consistent error categorization and severity levels
- Automatic error recovery suggestions
- User-friendly error message formatting
- Error statistics and monitoring
- Decorators for common error handling patterns
"""

from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, cast

from ..exceptions import APGIFrameworkError
from ..logging.standardized_logging import get_logger


class ErrorCategory(Enum):
    """Error category classification."""

    DATA = "DATA"
    IMPORT = "IMPORT"
    VALIDATION = "VALIDATION"
    SIMULATION = "SIMULATION"
    PROCESSING = "PROCESSING"
    IO = "IO"
    PERMISSION = "PERMISSION"
    NETWORK = "NETWORK"
    RUNTIME = "RUNTIME"
    MEMORY = "MEMORY"
    CONFIGURATION = "CONFIGURATION"
    USER_INPUT = "USER_INPUT"
    BACKUP = "BACKUP"
    CACHE = "CACHE"
    CRITICAL = "CRITICAL"
    ANALYSIS = "ANALYSIS"


class ErrorSeverity(Enum):
    """Error severity levels."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ErrorInfo:
    """Error information container."""

    def __init__(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        code: str,
        details: str,
        suggestions: Optional[List[str]] = None,
        user_action: Optional[str] = None,
        traceback: Optional[str] = None,
        message: Optional[str] = None,
        **kwargs: Any,
    ):
        self.category = category
        self.severity = severity
        self.code = code
        self.details = details
        self.suggestions = suggestions or []
        self.user_action = user_action
        self.traceback = traceback
        self.message = message or details
        self.extra = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert error info to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "code": self.code,
            "details": self.details,
            "suggestions": self.suggestions,
            "user_action": self.user_action,
            "traceback": self.traceback,
            "message": self.message,
            **self.extra,
        }


class ErrorHandler:
    """Base error handler class."""

    def __init__(self) -> None:
        self.error_handlers: Dict[ErrorCategory, Callable[[ErrorInfo], None]] = {}
        self.error_counts: Dict[str, int] = {}

    def create_error(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        code: str,
        details: Optional[str],
        suggestions: Optional[List[str]] = None,
        user_action: Optional[str] = None,
        traceback: Optional[str] = None,
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> ErrorInfo:
        """Create an error info object."""
        return ErrorInfo(
            category=category,
            severity=severity,
            code=code,
            details=details or "",
            suggestions=suggestions,
            user_action=user_action,
            traceback=traceback,
            message=message or details or "",
            **kwargs,
        )

    def register_handler(
        self, category: ErrorCategory, handler: Callable[[ErrorInfo], None]
    ) -> None:
        """Register a custom error handler."""
        self.error_handlers[category] = handler

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "by_category": dict(self.error_counts),
            "most_frequent": {},
            "error_distribution": {},
        }


def format_user_message(error_info: ErrorInfo) -> str:
    """Format error info into a user-friendly message."""
    message = f"[{error_info.severity.value}] {error_info.category.value}: {error_info.message}"
    if error_info.suggestions:
        message += "\n\nSuggestions:\n" + "\n".join(f"  - {s}" for s in error_info.suggestions)
    if error_info.user_action:
        message += f"\n\nAction: {error_info.user_action}"
    return message


def handle_error(
    category: ErrorCategory,
    severity: ErrorSeverity,
    code: str,
    reraise: bool,
    log_level: str,
    message: str = "",
    **context: Any,
) -> Optional[APGIFrameworkError]:
    """Handle error with given parameters."""
    error = APGIFrameworkError(message or code, **context)
    if reraise:
        raise error
    return error


def error_boundary(
    error_type: Type[Exception],
    category: Any = "RUNTIME",
    severity: Any = "MEDIUM",
    code: str = "UNHANDLED_EXCEPTION",
    reraise: bool = True,
    log_level: str = "error",
    include_traceback: bool = True,
    error_message: str = "Operation failed",
    default_return: Any = None,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    """Error boundary decorator."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exceptions:
                if reraise:
                    raise
                return default_return

        return wrapper

    return decorator


class StandardizedErrorHandler:
    """Standardized error handler for consistent error handling across the APGI Framework.

    Provides a unified interface for error categorization, logging, and user-friendly
    message formatting with automatic recovery suggestions.
    """

    def __init__(self) -> None:
        """Initialize the standardized error handler."""
        self.logger = get_logger(__name__)
        self.error_handler = ErrorHandler()

        # Error handling statistics
        self.error_counts: Dict[str, int] = {}
        self.last_errors: List[Dict[str, Any]] = []
        self.error_patterns: Dict[str, Any] = {}

        # Initialize common error patterns
        self._init_error_patterns()

    def _init_error_patterns(self) -> None:
        """Initialize common error patterns."""
        patterns = {
            "file_operations": {
                "file_not_found": {
                    "category": ErrorCategory.IO,
                    "severity": ErrorSeverity.HIGH,
                    "suggestions": [
                        "Check file path for typos",
                        "Verify file exists",
                        "Check permissions",
                        "Check working directory",
                    ],
                    "user_action": "Verify file path and permissions",
                },
                "permission_denied": {
                    "category": ErrorCategory.PERMISSION,
                    "severity": ErrorSeverity.HIGH,
                    "suggestions": [
                        "Run with appropriate permissions",
                        "Check file/directory ownership",
                        "Use sudo if necessary",
                    ],
                    "user_action": "Check permissions or run with sudo",
                },
                "invalid_data": {
                    "category": ErrorCategory.DATA,
                    "severity": ErrorSeverity.HIGH,
                    "suggestions": [
                        "Validate data format",
                        "Check data structure",
                        "Verify field requirements",
                    ],
                    "user_action": "Check data format and structure",
                },
                "validation_failed": {
                    "category": ErrorCategory.VALIDATION,
                    "severity": ErrorSeverity.HIGH,
                    "suggestions": [
                        "Check validation criteria",
                        "Verify input parameters",
                        "Review validation protocol",
                    ],
                    "user_action": "Review validation logic",
                },
                "timeout": {
                    "category": ErrorCategory.RUNTIME,
                    "severity": ErrorSeverity.MEDIUM,
                    "suggestions": [
                        "Increase timeout value",
                        "Check network connectivity",
                        "Simplify operation",
                    ],
                    "user_action": "Increase timeout or simplify operation",
                },
                "memory_error": {
                    "category": ErrorCategory.MEMORY,
                    "severity": ErrorSeverity.CRITICAL,
                    "suggestions": [
                        "Reduce data size",
                        "Close other applications",
                        "Increase system memory",
                    ],
                    "user_action": "Free up memory",
                },
                "import_error": {
                    "category": ErrorCategory.IMPORT,
                    "severity": ErrorSeverity.HIGH,
                    "suggestions": [
                        "Install missing dependencies",
                        "Check Python environment",
                        "Verify package installation",
                    ],
                    "user_action": "Install missing package",
                },
            },
            "configuration": {
                "invalid_config": {
                    "category": ErrorCategory.CONFIGURATION,
                    "severity": ErrorSeverity.HIGH,
                    "suggestions": [
                        "Check configuration syntax",
                        "Validate configuration file",
                        "Review configuration documentation",
                    ],
                    "user_action": "Check configuration file",
                },
                "missing_config": {
                    "category": ErrorCategory.CONFIGURATION,
                    "severity": ErrorSeverity.HIGH,
                    "suggestions": [
                        "Create configuration file",
                        "Copy from template",
                        "Check default locations",
                    ],
                    "user_action": "Create configuration file",
                },
                "parameter_error": {
                    "category": ErrorCategory.CONFIGURATION,
                    "severity": ErrorSeverity.HIGH,
                    "suggestions": [
                        "Check parameter range",
                        "Check parameter type",
                        "Review parameter documentation",
                    ],
                    "user_action": "Check parameter values",
                },
            },
            "simulation": {
                "convergence_failed": {
                    "category": ErrorCategory.SIMULATION,
                    "severity": ErrorSeverity.HIGH,
                    "suggestions": [
                        "Check simulation parameters",
                        "Adjust convergence criteria",
                        "Reduce step size",
                    ],
                    "user_action": "Adjust simulation parameters",
                },
                "numerical_instability": {
                    "category": ErrorCategory.SIMULATION,
                    "severity": ErrorSeverity.CRITICAL,
                    "suggestions": [
                        "Check numerical precision",
                        "Adjust algorithm parameters",
                        "Reduce time step",
                    ],
                    "user_action": "Check numerical settings",
                },
                "simulation_crashed": {
                    "category": ErrorCategory.SIMULATION,
                    "severity": ErrorSeverity.CRITICAL,
                    "suggestions": [
                        "Check simulation input",
                        "Verify simulation state",
                        "Check resource availability",
                    ],
                    "user_action": "Check simulation setup",
                },
            },
            "analysis": {
                "processing_error": {
                    "category": ErrorCategory.PROCESSING,
                    "severity": ErrorSeverity.MEDIUM,
                    "suggestions": [
                        "Check input data format",
                        "Verify processing parameters",
                        "Check memory availability",
                    ],
                    "user_action": "Check processing parameters",
                },
                "analysis_failed": {
                    "category": ErrorCategory.ANALYSIS,
                    "severity": ErrorSeverity.HIGH,
                    "suggestions": [
                        "Check analysis method",
                        "Verify data integrity",
                        "Check parameter settings",
                    ],
                    "user_action": "Check analysis setup",
                },
            },
        }
        self.error_patterns = patterns

    def categorize_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None,
    ) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Categorize an error based on its type and context.

        Args:
            error: The exception to categorize
            context: Additional context information
            severity: Override severity level
            category: Override error category

        Returns:
            Tuple of (category, severity)
        """
        # Use provided values if specified
        if category:
            return category, severity or ErrorSeverity.MEDIUM

        if severity:
            return category or ErrorCategory.VALIDATION, severity

        # Auto-categorize based on exception type
        error_type = type(error).__name__

        # Map exception types to categories
        category_mapping = {
            # Configuration errors
            "ConfigurationError": ErrorCategory.CONFIGURATION,
            "KeyError": ErrorCategory.CONFIGURATION,
            "ValueError": ErrorCategory.CONFIGURATION,
            # Data errors
            "DataError": ErrorCategory.DATA,
            "TypeError": ErrorCategory.DATA,
            "AttributeError": ErrorCategory.DATA,
            "IndexError": ErrorCategory.DATA,
            # Import errors
            "ImportError": ErrorCategory.IMPORT,
            "ModuleNotFoundError": ErrorCategory.IMPORT,
            # Validation errors
            "ValidationError": ErrorCategory.VALIDATION,
            "AssertionError": ErrorCategory.VALIDATION,
            # Simulation errors
            "RuntimeError": ErrorCategory.SIMULATION,
            "OverflowError": ErrorCategory.SIMULATION,
            "ZeroDivisionError": ErrorCategory.SIMULATION,
            "ConvergenceError": ErrorCategory.SIMULATION,
            # Processing errors
            "ProcessingError": ErrorCategory.PROCESSING,
            # I/O errors
            "FileNotFoundError": ErrorCategory.IO,
            "PermissionError": ErrorCategory.PERMISSION,
            "IsADirectoryError": ErrorCategory.IO,
            "FileExistsError": ErrorCategory.IO,
            "ConnectionError": ErrorCategory.NETWORK,
            "TimeoutError": ErrorCategory.RUNTIME,
            # Memory errors
            "MemoryError": ErrorCategory.MEMORY,
            "BufferError": ErrorCategory.MEMORY,
            # GUI errors
            "Tkinter.TclError": ErrorCategory.RUNTIME,
            # Network errors
            "HTTPError": ErrorCategory.NETWORK,
            # System errors
            "OSError": ErrorCategory.RUNTIME,
        }

        category = category_mapping.get(error_type, ErrorCategory.RUNTIME)

        # Determine severity based on category
        if severity is None:
            severity_mapping = {
                ErrorCategory.CRITICAL: ErrorSeverity.CRITICAL,
                ErrorCategory.MEMORY: ErrorSeverity.CRITICAL,
                ErrorCategory.CONFIGURATION: ErrorSeverity.HIGH,
                ErrorCategory.VALIDATION: ErrorSeverity.HIGH,
                ErrorCategory.SIMULATION: ErrorSeverity.HIGH,
                ErrorCategory.DATA: ErrorSeverity.HIGH,
                ErrorCategory.IO: ErrorSeverity.HIGH,
                ErrorCategory.NETWORK: ErrorSeverity.HIGH,
                ErrorCategory.PERMISSION: ErrorSeverity.HIGH,
                ErrorCategory.IMPORT: ErrorSeverity.HIGH,
                ErrorCategory.RUNTIME: ErrorSeverity.MEDIUM,
                ErrorCategory.PROCESSING: ErrorSeverity.MEDIUM,
                ErrorCategory.USER_INPUT: ErrorSeverity.LOW,
                ErrorCategory.BACKUP: ErrorSeverity.MEDIUM,
                ErrorCategory.CACHE: ErrorSeverity.LOW,
            }
            severity = severity_mapping.get(category, ErrorSeverity.MEDIUM)

        return category, severity

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        reraise: bool = True,
        log_level: str = "error",
        include_traceback: bool = True,
    ) -> Optional[APGIFrameworkError]:
        """Handle error with standardized formatting and logging.

        Args:
            error: The exception to handle
            context: Additional context information
            user_message: Custom user message
            suggestions: Recovery suggestions
            reraise: Whether to re-raise the error
            log_level: Logging level
            include_traceback: Whether to include traceback

        Returns:
            APGIError instance if not reraising
        """
        # Categorize error
        category, severity = self.categorize_error(error, context, severity=None)

        # Get error pattern
        if context:
            for pattern_name, pattern_info in self.error_patterns.items():
                if pattern_name in context.get("pattern", {}):
                    # Use pattern for suggestions if available
                    if not suggestions and "suggestions" in pattern_info:
                        suggestions = pattern_info["suggestions"]
                    break

        # Create error info
        error_info = self.error_handler.create_error(
            category=category,
            severity=severity,
            code=type(error).__name__,
            details=str(error),
            suggestions=suggestions,
            user_action=None,
            **(context or {}),
        )

        # Format user message if not provided
        if user_message is None:
            user_message = format_user_message(error_info)

        # Log error
        log_level = log_level.lower()
        if log_level == "debug":
            self.logger.debug(user_message)
        elif log_level == "info":
            self.logger.info(user_message)
        elif log_level == "warning":
            self.logger.warning(user_message)
        elif log_level == "critical":
            self.logger.critical(user_message)
        else:
            self.logger.error(user_message)

        # Include traceback for debugging if requested
        if include_traceback and error_info.traceback:
            self.logger.error(f"Traceback: {error_info.traceback}")

        # Call custom handler if registered
        try:
            if category in self.error_handler.error_handlers:
                self.error_handler.error_handlers[category](error_info)
        except Exception as e:
            self.logger.error(f"Error in custom error handler: {e}")

        # Update statistics
        self.error_counts[category.value] = self.error_counts.get(category.value, 0) + 1
        self.last_errors.append(error_info.to_dict())

        # Keep only last 100 errors
        if len(self.last_errors) > 100:
            self.last_errors = self.last_errors[-100:]

        # Reraise if requested
        if reraise:
            raise error

        return error_info  # type: ignore

    def create_error_message(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.RUNTIME,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        user_action: Optional[str] = None,
    ) -> str:
        """Create a standardized error message.

        Args:
            message: Base error message
            category: Error category
            severity: Error severity
            details: Additional details
            suggestions: Recovery suggestions
            user_action: User action

        Returns:
            Formatted error message
        """
        error_info = self.error_handler.create_error(
            category=category,
            severity=severity,
            code="CUSTOM_ERROR",
            message=message,
            details=details,
            suggestions=suggestions,
            user_action=user_action,
        )
        return format_user_message(error_info)  # type: ignore

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        summary = self.error_handler.get_error_summary()

        return {
            "total_errors": summary["total_errors"],
            "by_category": summary["by_category"],
            "most_frequent": summary["most_frequent"],
            "error_distribution": summary["error_distribution"],
            "recent_errors": self.last_errors[-10:],  # Last 10 errors
            "error_patterns": self.error_patterns,
        }

    def register_error_handler(self, category: Any, handler: Callable[[Any], None]) -> None:
        """Register a custom error handler for a category."""
        self.error_handler.register_handler(category, handler)

    def safe_execute(
        self,
        func: Callable,
        *args: Any,
        error_message: str = "Operation failed",
        default_return: Any = None,
        log_level: str = "error",
        include_traceback: bool = True,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Safely execute a function with comprehensive error handling.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            error_message: Custom error message
            default_return: Default return value on error
            log_level: Logging level
            include_traceback: Whether to include traceback

        Returns:
            Function result or default return
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return self.handle_error(
                error=e,
                context=context,
                user_message=error_message,
                reraise=False,
                log_level=log_level,
                include_traceback=include_traceback,
            )

    def retry_with_backoff(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        context: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """Decorator for retrying with exponential backoff.

        Args:
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries
            backoff: Multiplier for delay after each retry
            exceptions: Exception types to retry on
            context: Additional context

        Returns:
            Decorated function
        """
        return retry_on_error(  # type: ignore
            max_retries=max_retries,
            delay=delay,
            backoff=backoff,
            exceptions=exceptions,
            context=context,
        )

    def log_error_summary(self) -> None:
        """Log a summary of recent errors."""
        stats = self.get_error_statistics()

        if stats["total_errors"] == 0:
            self.logger.info("✅ No errors in the current session")
            return

        self.logger.info("📊 Error Summary:")
        self.logger.info(f"   Total: {stats['total_errors']}")

        if stats["by_category"]:
            self.logger.info("   By Category:")
            for category, count in stats["by_category"].items():
                self.logger.info(f"     {category}: {count}")

        if stats["most_frequent"]:
            self.logger.info(f"   Most Frequent: {stats['most_frequent']['message']}")

        if stats["recent_errors"]:
            self.logger.info("   Recent Errors:")
            for error in stats["recent_errors"]:
                self.logger.info(f"     - {error['message']}")

        self.logger.info(f"   Error Distribution: {stats['error_distribution']}")

    def clear_error_history(self) -> None:
        """Clear error history."""
        self.error_handler.error_counts.clear()
        self.last_errors.clear()
        self.logger.info("Error history cleared")

    def get_recovery_suggestions(
        self, error_code: str, category: Optional[ErrorCategory] = None
    ) -> List[str]:
        """Get recovery suggestions for an error code."""
        if category:
            pattern = self.error_patterns.get(category.value, {})
            suggestions = pattern.get(error_code, ["Contact support if issue persists"])
        else:
            suggestions = ["Contact support if issue persists"]

        return suggestions  # type: ignore

    def create_error_report(self) -> Dict[str, Any]:
        """Create a comprehensive error report."""
        stats = self.get_error_statistics()

        return {
            "timestamp": datetime.now().isoformat(),
            "error_summary": stats,
            "error_patterns": self.error_patterns,
            "recent_errors": self.last_errors,
            "total_errors": stats["total_errors"],
            "categories_with_errors": len(stats["by_category"]),
            "most_frequent_category": stats["most_frequent"],
        }


# Decorators for common error handling patterns
def handle_errors(
    category: Any = "RUNTIME",
    severity: Any = "MEDIUM",
    code: str = "UNHANDLED_EXCEPTION",
    reraise: bool = True,
    log_level: str = "error",
    include_traceback: bool = True,
) -> Callable[[Callable], Callable]:  # type: ignore[no-any-return]
    """Decorator for automatic error handling."""
    return error_boundary(  # type: ignore[no-any-return]
        error_type=APGIFrameworkError,
        category=category,
        severity=severity,
        code=code,
        reraise=reraise,
        log_level=log_level,
        include_traceback=include_traceback,
    )


def safe_execute(
    error_message: str = "Operation failed",
    default_return: Any = None,
    log_level: str = "error",
    include_traceback: bool = True,
    context: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable], Callable]:  # type: ignore[no-any-return]
    """Decorator for safe function execution."""
    return error_boundary(  # type: ignore[no-any-return]
        error_type=APGIFrameworkError,
        error_message=error_message,
        default_return=default_return,
        log_level=log_level,
        include_traceback=include_traceback,
        context=context,
    )


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    context: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable], Callable]:  # type: ignore[no-any-return]
    """Decorator for retrying with exponential backoff."""
    return error_boundary(  # type: ignore[no-any-return]
        error_type=APGIFrameworkError,
        max_retries=max_retries,
        delay=delay,
        backoff=backoff,
        exceptions=exceptions,
        context=context,
    )


# Convenience functions for common error types
def config_error(message: str, **context: Any) -> APGIFrameworkError:
    """Create a configuration error."""
    return cast(
        APGIFrameworkError,
        handle_error(
            ErrorCategory.CONFIGURATION,
            ErrorSeverity.HIGH,
            "CONFIG_ERROR",
            False,
            "error",
            context=context,
        ),
    )


def validation_error(message: str, **context: Any) -> APGIFrameworkError:
    """Create a validation error."""
    return cast(
        APGIFrameworkError,
        handle_error(
            ErrorCategory.VALIDATION,
            ErrorSeverity.HIGH,
            "VALIDATION_ERROR",
            False,
            "error",
            context=context,
        ),
    )


def data_error(message: str, **context: Any) -> APGIFrameworkError:
    """Create a data error."""
    return cast(
        APGIFrameworkError,
        handle_error(
            ErrorCategory.DATA,
            ErrorSeverity.HIGH,
            "DATA_ERROR",
            False,
            "error",
            context=context,
        ),
    )


def io_error(message: str, **context: Any) -> APGIFrameworkError:
    """Create an I/O error."""
    return cast(
        APGIFrameworkError,
        handle_error(
            ErrorCategory.IO,
            ErrorSeverity.HIGH,
            "IO_ERROR",
            False,
            "error",
            context=context,
        ),
    )


def critical_error(message: str, **context: Any) -> APGIFrameworkError:
    """Create a critical error."""
    return cast(
        APGIFrameworkError,
        handle_error(
            ErrorCategory.RUNTIME,
            ErrorSeverity.CRITICAL,
            "CRITICAL_ERROR",
            False,
            "error",
            context=context,
        ),
    )


def user_error(message: str, **context: Any) -> APGIFrameworkError:
    """Create a user input error."""
    return cast(
        APGIFrameworkError,
        handle_error(
            ErrorCategory.USER_INPUT,
            ErrorSeverity.LOW,
            "USER_ERROR",
            False,
            "error",
            context=context,
        ),
    )


# Global instance
standard_error_handler = StandardizedErrorHandler()


# Utility functions
def get_error_handler() -> StandardizedErrorHandler:
    """Get the global error handler instance."""
    return standard_error_handler


def log_error_summary() -> None:
    """Log error summary (alias for global instance)."""
    standard_error_handler.log_error_summary()


def handle_error_with_context(
    error: Exception, context: Optional[Dict[str, Any]] = None
) -> Optional[APGIFrameworkError]:
    """Handle error with additional context (alias for global instance)."""
    return standard_error_handler.handle_error(error, context)


def safe_execute_with_context(
    func: Callable, context: Optional[Dict[str, Any]] = None, *args: Any, **kwargs: Any
) -> Any:
    """Safe execution with context (alias for global instance)."""
    return standard_error_handler.safe_execute(func, *args, **kwargs, context=context)
