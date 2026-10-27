"""Enhanced Error Handling System for APGI Framework.

Provides comprehensive error handling, recovery strategies, and user-friendly
error reporting with detailed diagnostics and suggested solutions.
"""

import abc
import datetime
import json
import logging
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..logging.standardized_logging import get_logger
from .error_dialog_manager import ErrorDialogType, show_error_dialog


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Implementation of the Circuit Breaker pattern."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_requests: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests

        self.state = CircuitBreakerState.CLOSED
        self.failures = 0
        self.successes = 0
        self.last_failure_time: Optional[datetime.datetime] = None
        self.logger = get_logger(f"circuit_breaker.{name}")

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call the function through the circuit breaker."""
        self._check_state()

        if self.state == CircuitBreakerState.OPEN:
            raise RuntimeError(f"Circuit breaker '{self.name}' is OPEN. Request rejected.")

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise e

    def _check_state(self) -> None:
        """Transition from OPEN to HALF_OPEN if timeout expired."""
        if self.state == CircuitBreakerState.OPEN and self.last_failure_time:
            elapsed = (datetime.datetime.now() - self.last_failure_time).total_seconds()
            if elapsed >= self.recovery_timeout:
                self.logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                self.state = CircuitBreakerState.HALF_OPEN
                self.successes = 0

    def _record_success(self) -> None:
        """Handle successful call."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.successes += 1
            if self.successes >= self.half_open_max_requests:
                self.logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED")
                self.state = CircuitBreakerState.CLOSED
                self.failures = 0

    def _record_failure(self) -> None:
        """Handle failed call."""
        self.failures += 1
        self.last_failure_time = datetime.datetime.now()

        if self.state == CircuitBreakerState.CLOSED:
            if self.failures >= self.failure_threshold:
                self.logger.warning(f"Circuit breaker '{self.name}' transitioning to OPEN")
                self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.logger.warning(f"Circuit breaker '{self.name}' transitioning back to OPEN")
            self.state = CircuitBreakerState.OPEN


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    DATA_ERROR = "data_error"
    COMPUTATION_ERROR = "computation_error"
    IO_ERROR = "io_error"
    VALIDATION_ERROR = "validation_error"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_ERROR = "dependency_error"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"
    USER_ERROR = "user_error"
    SYSTEM_ERROR = "system_error"


@dataclass
class ErrorContext:
    """Context information for error analysis."""

    function_name: str
    module_name: str
    line_number: int
    local_variables: Dict[str, Any] = field(default_factory=dict)
    stack_trace: List[str] = field(default_factory=list)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    user_action: Optional[str] = None
    system_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorSolution:
    """Suggested solution for an error."""

    description: str
    action_type: str  # "automatic", "user_action", "configuration", "restart"
    steps: List[str]
    code_example: Optional[str] = None
    documentation_link: Optional[str] = None
    success_probability: float = 0.5  # 0.0 to 1.0


@dataclass
class APGIError:
    """Enhanced error information with context and solutions."""

    original_exception: Exception
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    user_message: str
    context: ErrorContext
    solutions: List[ErrorSolution] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/reporting."""
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "user_message": self.user_message,
            "timestamp": self.context.timestamp.isoformat(),
            "function": self.context.function_name,
            "module": self.context.module_name,
            "line": self.context.line_number,
            "solutions_count": len(self.solutions),
            "metadata": self.metadata,
        }


class ErrorRecoveryStrategy(abc.ABC):
    """Base class for error recovery strategies."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abc.abstractmethod
    def can_handle(self, error: APGIError) -> bool:
        """Check if this strategy can handle the error."""

    @abc.abstractmethod
    def recover(self, error: APGIError, **kwargs: Any) -> bool:
        """Attempt to recover from the error."""


class DataValidationRecovery(ErrorRecoveryStrategy):
    """Recovery strategy for data validation errors."""

    def __init__(self) -> None:
        super().__init__(
            "data_validation_recovery",
            "Attempts to clean and validate data automatically",
        )

    def can_handle(self, error: APGIError) -> bool:
        return error.category == ErrorCategory.DATA_ERROR

    def recover(self, error: APGIError, **kwargs: Any) -> bool:
        """Attempt data cleaning and validation."""
        try:
            data = kwargs.get("data")
            if data is None:
                return False

            # Attempt basic data cleaning based on data type
            if hasattr(data, "dropna"):
                # Pandas DataFrame/Series
                cleaned_data = data.dropna()
                if len(cleaned_data) > 0:
                    kwargs["cleaned_data"] = cleaned_data
                    return True
            elif hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
                # Lists, arrays, etc.
                try:
                    import numpy as np

                    if isinstance(data, np.ndarray):
                        # Remove NaN values from numpy array
                        cleaned_data = data[~np.isnan(data)]
                        if len(cleaned_data) > 0:
                            kwargs["cleaned_data"] = cleaned_data
                            return True
                    else:
                        # Filter out None values from list
                        cleaned_data = [x for x in data if x is not None]
                        if cleaned_data:
                            kwargs["cleaned_data"] = cleaned_data
                            return True
                except ImportError:
                    # Fallback for list filtering
                    cleaned_data = [x for x in data if x is not None]
                    if cleaned_data:
                        kwargs["cleaned_data"] = cleaned_data
                        return True
            elif isinstance(data, dict):
                # Clean dictionary values
                cleaned_data = {k: v for k, v in data.items() if v is not None}
                if cleaned_data:
                    kwargs["cleaned_data"] = cleaned_data
                    return True

            # If we reach here, data couldn't be cleaned
            return False
        except Exception as e:
            # Log the recovery attempt failure
            logging.warning(f"Data validation recovery failed: {e}")
            return False


class MemoryRecovery(ErrorRecoveryStrategy):
    """Recovery strategy for memory errors."""

    def __init__(self) -> None:
        super().__init__("memory_recovery", "Reduces memory usage by processing data in chunks")

    def can_handle(self, error: APGIError) -> bool:
        return error.category == ErrorCategory.MEMORY_ERROR

    def recover(self, error: APGIError, **kwargs: Any) -> bool:
        """Attempt memory optimization."""
        try:
            # Suggest chunked processing
            batch_size = kwargs.get("batch_size", 1000)
            total_size = kwargs.get("total_size", 0)

            if total_size > batch_size:
                kwargs["suggested_chunks"] = max(1, total_size // batch_size)
                return True

            return False
        except Exception:
            return False


class ConfigurationRecovery(ErrorRecoveryStrategy):
    """Recovery strategy for configuration errors."""

    def __init__(self) -> None:
        super().__init__("configuration_recovery", "Resets configuration to default values")

    def can_handle(self, error: APGIError) -> bool:
        return error.category == ErrorCategory.CONFIGURATION_ERROR

    def recover(self, error: APGIError, **kwargs: Any) -> bool:
        """Attempt configuration reset."""
        try:
            config_path = kwargs.get("config_path")
            if config_path and Path(config_path).exists():
                # Backup current config
                backup_path = Path(config_path).with_suffix(".backup")
                Path(config_path).rename(backup_path)

                # Create default config
                default_config = kwargs.get("default_config", {})
                with open(config_path, "w") as f:
                    json.dump(default_config, f, indent=2)

                return True

            return False
        except Exception:
            return False


class CircuitBreakerRecovery(ErrorRecoveryStrategy):
    """Recovery strategy for circuit breaker events."""

    def __init__(self, breaker: CircuitBreaker) -> None:
        super().__init__(
            f"circuit_breaker_recovery_{breaker.name}",
            f"Manages recovery for circuit breaker '{breaker.name}'",
        )
        self.breaker = breaker

    def can_handle(self, error: APGIError) -> bool:
        # This strategy is triggered when an error occurs in a circuit-protected call
        return True

    def recover(self, error: APGIError, **kwargs: Any) -> bool:
        """Attempt to handle error by recording failure in breaker."""
        self.breaker._record_failure()
        return False  # Still let the error bubble up or be handled by others


class EnhancedErrorHandler:
    """Enhanced error handler with context analysis and recovery."""

    def __init__(self) -> None:
        self.logger = get_logger("enhanced_error_handler")
        self.recovery_strategies: List[ErrorRecoveryStrategy] = []
        self.error_patterns: Dict[str, Dict[str, Any]] = {}
        self.error_history: List[APGIError] = []

        # Register default recovery strategies
        self._register_default_strategies()

        # Load error patterns
        self._load_error_patterns()

    def _register_default_strategies(self) -> None:
        """Register default recovery strategies."""
        self.recovery_strategies.extend(
            [DataValidationRecovery(), MemoryRecovery(), ConfigurationRecovery()]
        )

    def _load_error_patterns(self) -> None:
        """Load known error patterns and solutions."""
        self.error_patterns = {
            # Data errors
            "FileNotFoundError": {
                "category": ErrorCategory.IO_ERROR,
                "severity": ErrorSeverity.MEDIUM,
                "solutions": [
                    ErrorSolution(
                        description="Check if the file path is correct",
                        action_type="user_action",
                        steps=[
                            "Verify the file exists at the specified path",
                            "Check file permissions",
                            "Ensure the file is not locked by another process",
                        ],
                    )
                ],
            },
            # Memory errors
            "MemoryError": {
                "category": ErrorCategory.MEMORY_ERROR,
                "severity": ErrorSeverity.HIGH,
                "solutions": [
                    ErrorSolution(
                        description="Reduce memory usage by processing data in smaller chunks",
                        action_type="automatic",
                        steps=[
                            "Split data into smaller batches",
                            "Process one batch at a time",
                            "Clear intermediate results from memory",
                        ],
                        code_example="""
# Process data in chunks
chunk_size = 1000
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    result = process_chunk(chunk)
    # Process result immediately
                        """,
                    )
                ],
            },
            # Import errors
            "ModuleNotFoundError": {
                "category": ErrorCategory.DEPENDENCY_ERROR,
                "severity": ErrorSeverity.MEDIUM,
                "solutions": [
                    ErrorSolution(
                        description="Install missing dependency",
                        action_type="user_action",
                        steps=[
                            "Install the missing package using pip",
                            "Check requirements.txt for version requirements",
                            "Restart the application after installation",
                        ],
                        code_example="pip install <missing_package>",
                    )
                ],
            },
            # Validation errors
            "ValueError": {
                "category": ErrorCategory.VALIDATION_ERROR,
                "severity": ErrorSeverity.MEDIUM,
                "solutions": [
                    ErrorSolution(
                        description="Check input data format and values",
                        action_type="user_action",
                        steps=[
                            "Validate input data types",
                            "Check for NaN or infinite values",
                            "Ensure data is within expected ranges",
                        ],
                    )
                ],
            },
            # Convergence errors
            "ConvergenceWarning": {
                "category": ErrorCategory.COMPUTATION_ERROR,
                "severity": ErrorSeverity.LOW,
                "solutions": [
                    ErrorSolution(
                        description="Adjust optimization parameters",
                        action_type="configuration",
                        steps=[
                            "Increase maximum iterations",
                            "Adjust tolerance parameters",
                            "Try different optimization algorithms",
                            "Check data quality and preprocessing",
                        ],
                    )
                ],
            },
        }

    def register_recovery_strategy(self, strategy: ErrorRecoveryStrategy) -> None:
        """Register a new recovery strategy."""
        self.recovery_strategies.append(strategy)

    def _extract_context(self, exception: Exception) -> ErrorContext:
        """Extract context information from exception."""
        tb = traceback.extract_tb(exception.__traceback__)

        function_name = "unknown"
        module_name = "unknown"
        line_number: int = 0

        if tb:
            last_frame = tb[-1]
            function_name = last_frame.name
            module_name = last_frame.filename
            line_number = last_frame.lineno or 0

        # Get stack trace
        stack_trace = traceback.format_tb(exception.__traceback__)

        # Try to get local variables (limited for security)
        local_vars = {}
        if hasattr(exception, "__context__") and exception.__context__:
            # Extract some safe variable information
            exc_info = sys.exc_info()
            frame = None
            if exc_info[2] is not None:
                frame = exc_info[2].tb_frame
            if frame:
                for key, value in frame.f_locals.items():
                    if not key.startswith("_") and isinstance(value, (str, int, float, bool)):
                        local_vars[key] = str(value)[:100]  # Limit string length

        return ErrorContext(
            function_name=function_name,
            module_name=module_name,
            line_number=line_number,
            local_variables=local_vars,
            stack_trace=stack_trace,
        )

    def _classify_error(self, exception: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify error by type and determine severity."""
        exception_name = type(exception).__name__

        if exception_name in self.error_patterns:
            pattern = self.error_patterns[exception_name]
            return pattern["category"], pattern["severity"]

        # Default classification based on exception type
        if isinstance(exception, (FileNotFoundError, PermissionError, IOError)):
            return ErrorCategory.IO_ERROR, ErrorSeverity.MEDIUM
        elif isinstance(exception, (MemoryError, OverflowError)):
            return ErrorCategory.MEMORY_ERROR, ErrorSeverity.HIGH
        elif isinstance(exception, (ValueError, TypeError)):
            return ErrorCategory.VALIDATION_ERROR, ErrorSeverity.MEDIUM
        elif isinstance(exception, (ImportError, ModuleNotFoundError)):
            return ErrorCategory.DEPENDENCY_ERROR, ErrorSeverity.MEDIUM
        elif isinstance(exception, KeyboardInterrupt):
            return ErrorCategory.USER_ERROR, ErrorSeverity.LOW
        else:
            return ErrorCategory.SYSTEM_ERROR, ErrorSeverity.MEDIUM

    def _generate_solutions(
        self, exception: Exception, category: ErrorCategory
    ) -> List[ErrorSolution]:
        """Generate solutions based on error type."""
        exception_name = type(exception).__name__

        if exception_name in self.error_patterns:
            solutions = self.error_patterns[exception_name]["solutions"]
            # Ensure we return List[ErrorSolution]
            if isinstance(solutions, list) and all(isinstance(s, ErrorSolution) for s in solutions):
                return solutions
            return []

        # Generate generic solutions based on category
        generic_solutions = {
            ErrorCategory.DATA_ERROR: [
                ErrorSolution(
                    description="Validate and clean input data",
                    action_type="user_action",
                    steps=[
                        "Check data format and structure",
                        "Remove or handle missing values",
                        "Verify data types are correct",
                    ],
                )
            ],
            ErrorCategory.IO_ERROR: [
                ErrorSolution(
                    description="Check file system permissions and paths",
                    action_type="user_action",
                    steps=[
                        "Verify file/directory exists",
                        "Check read/write permissions",
                        "Ensure sufficient disk space",
                    ],
                )
            ],
            ErrorCategory.MEMORY_ERROR: [
                ErrorSolution(
                    description="Reduce memory usage",
                    action_type="configuration",
                    steps=[
                        "Process data in smaller chunks",
                        "Clear unused variables",
                        "Consider using more efficient data structures",
                    ],
                )
            ],
        }

        return generic_solutions.get(category, [])

    def handle_error(
        self, exception: Exception, user_action: Optional[str] = None, **context: Any
    ) -> APGIError:
        """Handle an error with enhanced analysis and recovery."""
        # Generate unique error ID
        error_id = f"APGI_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(exception)}"

        # Extract context
        error_context = self._extract_context(exception)
        if user_action:
            error_context.user_action = user_action

        # Classify error
        category, severity = self._classify_error(exception)

        # Generate user-friendly message
        user_message = self._generate_user_message(exception, category)

        # Generate solutions
        solutions = self._generate_solutions(exception, category)

        # Create enhanced error
        apgi_error = APGIError(
            original_exception=exception,
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(exception),
            user_message=user_message,
            context=error_context,
            solutions=solutions,
            metadata=context,
        )

        # Log error
        self._log_error(apgi_error)

        # Add to history
        self.error_history.append(apgi_error)

        # Attempt recovery
        self._attempt_recovery(apgi_error, **context)

        # Show user-friendly dialog if severity is high enough
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._show_error_dialog(apgi_error)

        return apgi_error

    def _show_error_dialog(self, apgi_error: APGIError) -> None:
        """Show a user-friendly error dialog using the error dialog manager."""
        try:
            # Determine dialog type based on severity
            if apgi_error.severity == ErrorSeverity.CRITICAL:
                dialog_type = ErrorDialogType.ERROR
            elif apgi_error.severity == ErrorSeverity.HIGH:
                dialog_type = ErrorDialogType.WARNING
            else:
                dialog_type = ErrorDialogType.INFO

            # Create user-friendly title
            title = f"APGI {apgi_error.severity.value.upper()}"

            # Use user message if available, otherwise use technical message
            message = apgi_error.user_message or apgi_error.message

            # Show dialog using the error dialog manager
            show_error_dialog(
                message=message,
                title=title,
                dialog_type=dialog_type,
                details=apgi_error.message if apgi_error.user_message else None,
            )

        except Exception as e:
            # Fallback to logging if dialog fails
            self.logger.error(f"Failed to show error dialog: {e}")

    def _generate_user_message(self, exception: Exception, category: ErrorCategory) -> str:
        """Generate user-friendly error message."""
        base_messages = {
            ErrorCategory.DATA_ERROR: "There was an issue with the input data.",
            ErrorCategory.IO_ERROR: "There was a problem accessing a file or directory.",
            ErrorCategory.MEMORY_ERROR: "The operation requires more memory than available.",
            ErrorCategory.VALIDATION_ERROR: "The provided input is not valid.",
            ErrorCategory.DEPENDENCY_ERROR: "A required software component is missing.",
            ErrorCategory.CONFIGURATION_ERROR: "There is an issue with the configuration settings.",
            ErrorCategory.USER_ERROR: "The operation was cancelled by the user.",
            ErrorCategory.SYSTEM_ERROR: "An unexpected system error occurred.",
        }

        base_message = base_messages.get(category, "An error occurred")
        return f"{base_message} Details: {str(exception)}"

    def _log_error(self, error: APGIError) -> None:
        """Log error with appropriate level."""
        log_level_map = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }

        level = log_level_map[error.severity]
        self.logger._log(level, f"Error {error.error_id}: {error.message}")

        # Log detailed information at debug level
        self.logger.debug(f"Error details: {json.dumps(error.to_dict(), indent=2)}")

    def _attempt_recovery(self, error: APGIError, **context: Any) -> bool:
        """Attempt to recover from error using registered strategies."""
        for strategy in self.recovery_strategies:
            if strategy.can_handle(error):
                try:
                    if strategy.recover(error, **context):
                        self.logger.info(
                            f"Successfully recovered from error {error.error_id} using {strategy.name}"
                        )
                        return True
                except Exception as recovery_error:
                    self.logger.warning(
                        f"Recovery strategy {strategy.name} failed: {recovery_error}"
                    )

        return False

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about handled errors."""
        if not self.error_history:
            return {"total_errors": 0}

        category_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {}

        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1

        return {
            "total_errors": len(self.error_history),
            "by_category": category_counts,
            "by_severity": severity_counts,
            "most_recent": (self.error_history[-1].to_dict() if self.error_history else None),
        }


# Global error handler instance
_global_error_handler = None


def get_error_handler() -> EnhancedErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = EnhancedErrorHandler()
    return _global_error_handler


@contextmanager
def error_context(user_action: Optional[str] = None, **context: Any) -> Any:
    """Context manager for enhanced error handling."""
    try:
        yield
    except Exception as e:
        error_handler = get_error_handler()
        apgi_error = error_handler.handle_error(e, user_action, **context)

        # Re-raise with enhanced information
        raise APGIException(apgi_error) from e


class APGIException(Exception):
    """Custom exception class that wraps enhanced error information."""

    def __init__(self, apgi_error: APGIError) -> None:
        self.apgi_error = apgi_error
        super().__init__(apgi_error.user_message)

    def get_solutions(self) -> List[ErrorSolution]:
        """Get suggested solutions for this error."""
        return self.apgi_error.solutions

    def get_error_id(self) -> str:
        """Get the unique error ID."""
        return self.apgi_error.error_id

    def can_recover(self) -> bool:
        """Check if this error has recovery options."""
        return len(self.apgi_error.solutions) > 0


# Convenience functions
def handle_error(
    exception: Exception, user_action: Optional[str] = None, **context: Any
) -> APGIError:
    """Handle an error with enhanced analysis."""
    return get_error_handler().handle_error(exception, user_action, **context)


def register_recovery_strategy(strategy: ErrorRecoveryStrategy) -> None:
    """Register a new error recovery strategy."""
    get_error_handler().register_recovery_strategy(strategy)


def get_error_stats() -> Dict[str, Any]:
    """Get error handling statistics."""
    return get_error_handler().get_error_statistics()
