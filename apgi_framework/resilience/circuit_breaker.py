"""Circuit breaker pattern implementation for resilient external calls.

Provides automatic fault detection and graceful degradation for
external service dependencies.
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Failing, reject calls
    HALF_OPEN = auto()  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2
    excluded_exceptions: tuple = (ValueError, TypeError)


class CircuitBreaker:
    """Circuit breaker for resilient external service calls.

    Automatically detects failures and prevents cascade failures
    by temporarily rejecting calls to failing services.

    Example:
        breaker = CircuitBreaker("payment_api")

        @breaker
        def call_payment_api(amount: float) -> dict:
            return requests.post("...", json={"amount": amount}).json()
    """

    _instances: Dict[str, "CircuitBreaker"] = {}
    _instance_lock: threading.RLock = threading.RLock()

    def __new__(cls, name: str, config: Optional[CircuitBreakerConfig] = None):
        with cls._instance_lock:
            if name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[name] = instance
            return cls._instances[name]

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        if hasattr(self, "_initialized"):
            return

        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[float] = None
        self._instance_lock: threading.RLock = threading.RLock()
        self._initialized = True

        logger.info(f"Circuit breaker '{name}' initialized")

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        with self._instance_lock:
            return self._state

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        with self._instance_lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"Circuit '{self.name}' entering HALF_OPEN state")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit '{self.name}' is OPEN - service unavailable"
                    )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"Circuit '{self.name}' HALF_OPEN call limit reached"
                    )
                self._half_open_calls += 1

        # Execute the call outside the lock
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            if not isinstance(e, self.config.excluded_exceptions):
                self._record_failure()
            raise

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for circuit breaker protection."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.call(func, *args, **kwargs)

        return wrapper

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try reset."""
        if self._last_failure_time is None:
            return True
        return (time.time() - self._last_failure_time) >= self.config.recovery_timeout

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._instance_lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._half_open_calls = 0
                    logger.info(f"Circuit '{self.name}' CLOSED (recovered)")
            else:
                self._failure_count = max(0, self._failure_count - 1)

    def _record_failure(self) -> None:
        """Record a failed call."""
        with self._instance_lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
                logger.warning(f"Circuit '{self.name}' OPEN (half-open test failed)")
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.config.failure_threshold
            ):
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit '{self.name}' OPEN ({self._failure_count} failures)")

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._instance_lock:
            return {
                "name": self.name,
                "state": self._state.name,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "half_open_calls": self._half_open_calls,
                "last_failure_time": self._last_failure_time,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                },
            }

    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuit breakers."""
        with cls._instance_lock:
            return {name: cb.get_stats() for name, cb in cls._instances.items()}

    @classmethod
    def reset_all(cls) -> None:
        """Reset all circuit breakers (useful for testing)."""
        with cls._instance_lock:
            for cb in cls._instances.values():
                with cb._instance_lock:
                    cb._state = CircuitState.CLOSED
                    cb._failure_count = 0
                    cb._success_count = 0
                    cb._half_open_calls = 0
                    cb._last_failure_time = None


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class ResilientClient:
    """HTTP client with built-in circuit breaker and retry logic.

    Provides resilient external API calls with automatic failure handling.
    """

    def __init__(
        self,
        service_name: str,
        base_url: str,
        config: Optional[CircuitBreakerConfig] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.service_name = service_name
        self.base_url = base_url
        self.circuit_breaker = CircuitBreaker(service_name, config)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> Any:
        """Make resilient HTTP request."""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        url = f"{self.base_url}{endpoint}"

        # Use requests retry adapter for transient failures
        session = requests.Session()
        retries = Retry(
            total=self.max_retries,
            backoff_factor=self.retry_delay,
            status_forcelist=[500, 502, 503, 504],
        )
        session.mount("http://", HTTPAdapter(max_retries=retries))
        session.mount("https://", HTTPAdapter(max_retries=retries))

        def make_request() -> Any:
            response = session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()

        return self.circuit_breaker.call(make_request)

    def get(self, endpoint: str, **kwargs: Any) -> Any:
        """Make GET request."""
        return self.request("GET", endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs: Any) -> Any:
        """Make POST request."""
        return self.request("POST", endpoint, **kwargs)
