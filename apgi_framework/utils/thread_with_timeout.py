"""Thread with Timeout Support.

Provides a drop-in replacement for threading.Thread with timeout support.
Can be used to replace direct thread creation in GUI applications.
"""

import threading
from typing import Any, Callable, Optional


class ThreadTimeoutError(Exception):
    """Exception raised when thread execution times out."""


class ThreadWithTimeout(threading.Thread):
    """Thread class with timeout support.

    Provides same interface as threading.Thread but adds timeout capability
    to prevent long-running experiments from hanging indefinitely.
    """

    def __init__(
        self,
        target: Callable[..., Any],
        timeout: Optional[float] = None,
        on_timeout: Optional[Callable[[], None]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize thread with timeout support.

        Args:
            target: Target function to execute
            timeout: Timeout in seconds (None for no timeout)
            on_timeout: Callback function to execute on timeout
            *args: Positional arguments for target
            **kwargs: Keyword arguments for target
        """
        super().__init__(target=self._run_with_timeout, args=args, kwargs=kwargs)
        self._actual_target = target
        self._timeout = timeout
        self._on_timeout = on_timeout
        self._timeout_event = threading.Event()
        self._completed = threading.Event()
        self._result = None
        self._exception: Optional[Exception] = None

    def _run_with_timeout(self, *args: Any, **kwargs: Any) -> None:
        """Run target function with timeout monitoring."""
        if self._timeout is None:
            # No timeout, run normally
            try:
                self._result = self._actual_target(*args, **kwargs)
            except Exception as e:
                self._exception = e
            finally:
                self._completed.set()
            return

        # Run with timeout monitoring
        def timeout_monitor() -> None:
            if not self._timeout_event.wait(self._timeout):
                # Timeout occurred
                self._timeout_event.set()
                if self._on_timeout:
                    try:
                        self._on_timeout()
                    except Exception:
                        pass  # Ignore timeout callback errors

        # Start timeout monitor thread
        monitor_thread = threading.Thread(target=timeout_monitor, daemon=True)
        monitor_thread.start()

        # Run actual target
        try:
            self._result = self._actual_target(*args, **kwargs)
        except Exception as e:
            self._exception = e
        finally:
            self._completed.set()
            self._timeout_event.set()

    def get_result(self, timeout: Optional[float] = None) -> Any:
        """Get result from thread execution.

        Args:
            timeout: Optional timeout for waiting for result

        Returns:
            Result from target function

        Raises:
            ThreadTimeoutError: If thread times out
            Exception: If target function raised an exception
        """
        if not self._completed.wait(timeout):
            raise ThreadTimeoutError(f"Thread timed out after {timeout or self._timeout}s")

        if self._exception:
            raise self._exception

        return self._result

    def is_completed(self) -> bool:
        """Check if thread has completed."""
        return self._completed.is_set()

    def is_timeout(self) -> bool:
        """Check if thread timed out."""
        return self._timeout_event.is_set() and not self._completed.is_set()


def run_with_timeout(
    target: Callable[..., Any],
    timeout: Optional[float] = None,
    on_timeout: Optional[Callable[[], None]] = None,
    daemon: bool = True,
    *args: Any,
    **kwargs: Any,
) -> ThreadWithTimeout:
    """Convenience function to run a function with timeout support.

    Args:
        target: Target function to execute
        timeout: Timeout in seconds (None for no timeout)
        on_timeout: Callback function to execute on timeout
        daemon: Whether to run as daemon thread
        *args: Positional arguments for target
        **kwargs: Keyword arguments for target

    Returns:
        ThreadWithTimeout instance
    """
    # Filter out reserved keyword arguments to prevent conflicts
    filtered_kwargs = {
        k: v for k, v in kwargs.items() if k not in ("target", "timeout", "on_timeout")
    }

    thread = ThreadWithTimeout(
        target=target, timeout=timeout, on_timeout=on_timeout, *args, **filtered_kwargs  # type: ignore
    )
    thread.daemon = daemon
    thread.start()
    return thread


# Convenience function for GUI applications
def run_experiment_with_timeout(
    experiment_func: Callable[..., Any],
    timeout: float = 300,  # Default 5 minutes
    on_complete: Optional[Callable[[Any], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None,
    on_timeout: Optional[Callable[[], None]] = None,
    *args: Any,
    **kwargs: Any,
) -> ThreadWithTimeout:
    """Run experiment function with timeout support.

    Args:
        experiment_func: Experiment function to run
        timeout: Timeout in seconds (default 300s = 5 minutes)
        on_complete: Callback for successful completion
        on_error: Callback for errors
        on_timeout: Callback for timeout
        *args: Arguments for experiment function
        **kwargs: Keyword arguments for experiment function

    Returns:
        ThreadWithTimeout instance
    """

    def on_timeout_wrapper() -> None:
        if on_timeout:
            on_timeout()
        if on_error:
            on_error(ThreadTimeoutError(f"Experiment timed out after {timeout}s"))

    # Filter out reserved keyword arguments to prevent conflicts
    filtered_kwargs = {
        k: v for k, v in kwargs.items() if k not in ("target", "timeout", "on_timeout")
    }

    thread = run_with_timeout(
        target=experiment_func,
        timeout=timeout,
        on_timeout=on_timeout_wrapper,
        *args,
        **filtered_kwargs,  # type: ignore
    )

    # Add completion callback
    if on_complete or on_error:

        def check_completion() -> None:
            try:
                result = thread.get_result(timeout=timeout + 1)
                if on_complete:
                    on_complete(result)
            except ThreadTimeoutError:
                # Already handled by on_timeout_wrapper
                pass
            except Exception as e:
                if on_error:
                    on_error(e)

        # Start completion checker in separate thread
        completion_thread = threading.Thread(target=check_completion, daemon=True)
        completion_thread.start()

    return thread
