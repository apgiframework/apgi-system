"""Error Recovery Module.

Provides automatic error recovery mechanisms and retry logic for transient failures.
"""

import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

from ..exceptions import DataError, SimulationError
from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 30.0,
        retriable_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        """Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay between retries (seconds)
            backoff_factor: Exponential backoff multiplier
            max_delay: Maximum delay between retries (seconds)
            retriable_exceptions: List of exception types that should trigger retry
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.retriable_exceptions = retriable_exceptions or [
            SimulationError,
            DataError,
            IOError,
            OSError,
        ]


def with_retry(config: Optional[RetryConfig] = None) -> Any:
    """Decorator for automatic retry on transient failures.

    Args:
        config: RetryConfig instance (uses defaults if None)

    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            delay = config.initial_delay

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)

                except tuple(config.retriable_exceptions) as e:
                    last_exception = e

                    if attempt < config.max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * config.backoff_factor, config.max_delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed for {func.__name__}: {str(e)}"
                        )

                except Exception as e:
                    # Non-retriable exception - fail immediately
                    logger.error(f"Non-retriable exception in {func.__name__}: {str(e)}")
                    raise

            # All retries exhausted
            if last_exception is not None:
                raise last_exception
            else:
                raise RuntimeError(
                    f"All {config.max_attempts} attempts failed but no exception was captured"
                )

        return wrapper

    return decorator


class ErrorRecoveryManager:
    """Manages error recovery strategies and fallback mechanisms."""

    def __init__(self) -> None:
        self.error_log: List[Dict[str, Any]] = []
        self.recovery_strategies: Dict[
            Type[Exception], Callable[[Exception, Dict[str, Any]], Dict[str, Any]]
        ] = {}

    def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log error with context information.

        Args:
            error: Exception that occurred
            context: Dictionary with context information
        """
        error_entry = {
            "timestamp": datetime.now(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
        }
        self.error_log.append(error_entry)
        logger.error(f"Error logged: {error_entry}")

    def register_recovery_strategy(self, error_type: Type[Exception], strategy: Callable) -> None:
        """Register a recovery strategy for a specific error type.

        Args:
            error_type: Type of exception to handle
            strategy: Callable that implements recovery logic
        """
        self.recovery_strategies[error_type] = strategy
        logger.info(f"Registered recovery strategy for {error_type.__name__}")

    def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]:
        """Attempt to recover from an error using registered strategies.

        Args:
            error: Exception to recover from
            context: Context information

        Returns:
            Recovery result if successful, None otherwise
        """
        error_type = type(error)

        # Try exact match first
        if error_type in self.recovery_strategies:
            strategy = self.recovery_strategies[error_type]
            try:
                logger.info(f"Attempting recovery for {error_type.__name__}")
                result = strategy(error, context)
                logger.info(f"Recovery successful for {error_type.__name__}")
                return result
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {str(recovery_error)}")
                return None

        # Try parent classes
        for registered_type, strategy in self.recovery_strategies.items():
            if isinstance(error, registered_type):
                try:
                    logger.info(f"Attempting recovery using {registered_type.__name__} strategy")
                    result = strategy(error, context)
                    logger.info(f"Recovery successful using {registered_type.__name__} strategy")
                    return result
                except Exception as recovery_error:
                    logger.error(f"Recovery failed: {str(recovery_error)}")
                    continue

        logger.warning(f"No recovery strategy found for {error_type.__name__}")
        return None

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about logged errors."""
        if not self.error_log:
            return {"total_errors": 0}

        error_types: Dict[str, int] = {}
        for entry in self.error_log:
            error_type = entry["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "total_errors": len(self.error_log),
            "error_types": error_types,
            "first_error": self.error_log[0]["timestamp"],
            "last_error": self.error_log[-1]["timestamp"],
        }

    def clear_error_log(self) -> None:
        """Clear error log."""
        self.error_log.clear()
        logger.info("Error log cleared")


def safe_execute(
    func: Callable,
    fallback_value: Any = None,
    log_errors: bool = True,
    context: Optional[dict] = None,
) -> Any:
    """Safely execute a function with error handling and fallback.

    Args:
        func: Function to execute
        fallback_value: Value to return if function fails
        log_errors: Whether to log errors
        context: Context information for error logging

    Returns:
        Function result or fallback value
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.error(f"Error in safe_execute: {str(e)}")
            if context:
                logger.error(f"Context: {context}")
        return fallback_value


def validate_and_fix(
    value: Any,
    validator: Callable[[Any], bool],
    fixer: Callable[[Any], Any],
    error_message: str = "Validation failed",
) -> Any:
    """Validate a value and attempt to fix it if invalid.

    Args:
        value: Value to validate
        validator: Function that returns True if value is valid
        fixer: Function that attempts to fix invalid value
        error_message: Error message if fixing fails

    Returns:
        Valid value (original or fixed)

    Raises:
        ValueError: If value cannot be fixed
    """
    if validator(value):
        return value

    try:
        fixed_value = fixer(value)
        if validator(fixed_value):
            logger.warning(f"Value fixed: {value} -> {fixed_value}")
            return fixed_value
        else:
            raise ValueError(f"{error_message}: Could not fix value {value}")
    except Exception as e:
        raise ValueError(f"{error_message}: {str(e)}")


# Global error recovery manager
_recovery_manager = None


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager instance."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = ErrorRecoveryManager()
    return _recovery_manager


# Default recovery strategies


def recover_from_simulation_error(
    error: SimulationError, context: Dict[str, Any]
) -> Dict[str, Any]:
    """Default recovery strategy for simulation errors."""
    logger.info("Attempting to recover from simulation error by regenerating with different seed")

    # Try with a different random seed
    import numpy as np

    new_seed = np.random.randint(0, 1000000)
    np.random.seed(new_seed)

    return {"action": "retry"}


def recover_from_data_error(error: DataError, context: Dict[str, Any]) -> Dict[str, Any]:
    """Default recovery strategy for data errors."""
    logger.info("Attempting to recover from data error by using backup or default values")

    # Return default/empty data structure
    return {"action": "return", "result": {}}


# Register default strategies
def initialize_default_recovery_strategies() -> None:
    """Initialize default recovery strategies for common error types.

    This function registers recovery strategies for:
    - SimulationError: Retry with different random seed
    - DataError: Use backup or default values
    - IOError/OSError: Create missing directories, retry file operations
    """
    manager = get_recovery_manager()

    # Register simulation error recovery
    manager.register_recovery_strategy(SimulationError, recover_from_simulation_error)

    # Register data error recovery
    manager.register_recovery_strategy(DataError, recover_from_data_error)

    # Register I/O error recovery
    def recover_from_io_error(error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recovery strategy for I/O errors."""
        logger.info("Attempting to recover from I/O error")

        # Try creating missing directories
        if "path" in context:
            try:
                import os

                path = context["path"]
                directory = os.path.dirname(path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                    logger.info(f"Created directory: {directory}")
                    return {"action": "retry"}
            except Exception as e:
                logger.error(f"Failed to create directory: {e}")

        return {"action": "fail"}

    manager.register_recovery_strategy(IOError, recover_from_io_error)
    manager.register_recovery_strategy(OSError, recover_from_io_error)

    logger.info("Default recovery strategies initialized successfully")
