"""Framework-wide Rate Limiting and Brute-Force Protection.

Provides comprehensive rate limiting, brute-force detection, and DDoS protection
across the APGI framework.
"""

import hashlib
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"  # Simple counter reset per window
    SLIDING_WINDOW = "sliding_window"  # Rolling window for smoother limiting
    TOKEN_BUCKET = "token_bucket"  # Burstable rate limiting
    LEAKY_BUCKET = "leaky_bucket"  # Constant rate limiting


class SecurityAction(Enum):
    """Security response actions."""

    ALLOW = "allow"
    WARN = "warn"
    DELAY = "delay"
    BLOCK = "block"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    block_duration: int = 3600  # seconds (1 hour)
    max_failures: int = 5
    failure_window: int = 300  # seconds (5 minutes)


@dataclass
class ClientState:
    """Tracks rate limit state for a client."""

    client_id: str
    requests: List[datetime] = field(default_factory=list)
    failures: List[datetime] = field(default_factory=list)
    blocked_until: Optional[datetime] = None
    consecutive_failures: int = 0
    last_request: datetime = field(default_factory=datetime.now)

    def is_blocked(self) -> bool:
        """Check if client is currently blocked."""
        if self.blocked_until and datetime.now() < self.blocked_until:
            return True
        self.blocked_until = None
        return False


class BruteForceDetector:
    """Detects and prevents brute-force attacks."""

    def __init__(self) -> None:
        self._attempts: Dict[str, List[datetime]] = defaultdict(list)
        self._blocked: Dict[str, datetime] = {}
        self._lock = threading.Lock()

        # Configuration
        self.max_attempts = 5
        self.time_window = 300  # 5 minutes
        self.block_duration = 3600  # 1 hour
        self.backoff_multiplier = 2

    def record_attempt(
        self,
        identifier: str,
        success: bool,
        action: str = "auth",
    ) -> SecurityAction:
        """Record an authentication attempt and return security action."""
        with self._lock:
            now = datetime.now()

            # Check if already blocked
            if identifier in self._blocked:
                if now < self._blocked[identifier]:
                    logger.warning(f"Brute-force: blocked attempt from {identifier} ({action})")
                    return SecurityAction.BLOCK
                else:
                    del self._blocked[identifier]

            # Clean old attempts
            cutoff = now - timedelta(seconds=self.time_window)
            self._attempts[identifier] = [t for t in self._attempts[identifier] if t > cutoff]

            if success:
                # Clear attempts on success
                self._attempts[identifier] = []
                return SecurityAction.ALLOW

            # Record failure
            self._attempts[identifier].append(now)

            # Check threshold
            attempt_count = len(self._attempts[identifier])
            if attempt_count >= self.max_attempts:
                # Calculate exponential backoff block duration
                blocks = attempt_count // self.max_attempts
                block_time = self.block_duration * (self.backoff_multiplier ** (blocks - 1))
                block_until = now + timedelta(seconds=min(block_time, 86400))  # Max 1 day
                self._blocked[identifier] = block_until

                logger.warning(
                    f"Brute-force: Blocking {identifier} for {block_time}s "
                    f"after {attempt_count} failed attempts"
                )
                return SecurityAction.BLOCK

            if attempt_count >= self.max_attempts * 0.8:
                return SecurityAction.DELAY

            return SecurityAction.ALLOW

    def get_delay(self, identifier: str) -> float:
        """Get delay duration for delayed responses."""
        with self._lock:
            attempts = len(self._attempts.get(identifier, []))
            if attempts > 0:
                # Exponential delay: 0.1s, 0.2s, 0.4s, 0.8s, 1.6s...
                delay: float = min(0.1 * (2 ** (attempts - 1)), 5.0)
                return delay
            return 0.0

    def reset(self, identifier: str) -> None:
        """Reset attempts for an identifier."""
        with self._lock:
            self._attempts.pop(identifier, None)
            self._blocked.pop(identifier, None)


class RateLimiter:
    """Framework-wide rate limiter with multiple strategies."""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._clients: Dict[str, ClientState] = {}
        self._lock = threading.RLock()
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        logger.info(f"RateLimiter initialized with {self.config.strategy.value} strategy")

    def _cleanup_loop(self) -> None:
        """Background thread to clean up old entries."""
        while True:
            time.sleep(self._cleanup_interval)
            try:
                self._cleanup_old_entries()
            except Exception as e:
                logger.error(f"Rate limiter cleanup error: {e}")

    def _cleanup_old_entries(self) -> None:
        """Remove old client entries."""
        now = datetime.now()
        cutoff = now - timedelta(days=1)

        with self._lock:
            to_remove = [
                client_id
                for client_id, state in self._clients.items()
                if state.last_request < cutoff and not state.is_blocked()
            ]
            for client_id in to_remove:
                del self._clients[client_id]

            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} stale rate limit entries")

    def _get_client_id(self, request_context: Dict[str, Any]) -> str:
        """Extract client identifier from request context."""
        # Priority: API key > Session ID > IP + User-Agent hash
        if "api_key" in request_context:
            return f"api:{request_context['api_key']}"

        if "session_id" in request_context:
            return f"session:{request_context['session_id']}"

        # Fall back to IP + User-Agent hash
        ip = request_context.get("remote_addr", "unknown")
        ua = request_context.get("user_agent", "")
        return f"ip:{hashlib.sha256(f'{ip}:{ua}'.encode()).hexdigest()[:16]}"

    def check_rate_limit(self, request_context: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits.

        Returns:
            (allowed, response_headers)
        """
        client_id = self._get_client_id(request_context)
        now = datetime.now()

        with self._lock:
            # Get or create client state
            if client_id not in self._clients:
                self._clients[client_id] = ClientState(client_id=client_id)

            state = self._clients[client_id]
            state.last_request = now

            # Check if blocked
            if state.is_blocked():
                remaining = (
                    int((state.blocked_until - now).total_seconds()) if state.blocked_until else 0
                )
                logger.warning(f"Rate limit: {client_id} blocked for {remaining}s")
                return False, {
                    "X-RateLimit-Allowed": "false",
                    "X-RateLimit-Retry-After": str(remaining),
                    "Retry-After": str(remaining),
                }

            # Clean old requests
            state.requests = [t for t in state.requests if now - t < timedelta(hours=24)]

            # Check rate limits based on strategy
            allowed = self._check_limits(state, now)

            if not allowed:
                # Block the client
                state.blocked_until = now + timedelta(seconds=self.config.block_duration)
                logger.warning(f"Rate limit exceeded: {client_id} blocked")
                return False, {
                    "X-RateLimit-Allowed": "false",
                    "X-RateLimit-Retry-After": str(self.config.block_duration),
                    "Retry-After": str(self.config.block_duration),
                }

            # Record request
            state.requests.append(now)

            # Calculate remaining quota
            remaining = self._calculate_remaining(state, now)

            return True, {
                "X-RateLimit-Limit": str(self.config.requests_per_minute),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int((now + timedelta(minutes=1)).timestamp())),
            }

    def _check_limits(self, state: ClientState, now: datetime) -> bool:
        """Check if client is within all rate limits."""
        # Check minute limit
        minute_ago = now - timedelta(minutes=1)
        minute_requests = sum(1 for t in state.requests if t > minute_ago)
        if minute_requests >= self.config.requests_per_minute:
            return False

        # Check hour limit
        hour_ago = now - timedelta(hours=1)
        hour_requests = sum(1 for t in state.requests if t > hour_ago)
        if hour_requests >= self.config.requests_per_hour:
            return False

        # Check day limit
        day_ago = now - timedelta(days=1)
        day_requests = sum(1 for t in state.requests if t > day_ago)
        if day_requests >= self.config.requests_per_day:
            return False

        # Check burst limit (last 10 seconds)
        ten_sec_ago = now - timedelta(seconds=10)
        burst_requests = sum(1 for t in state.requests if t > ten_sec_ago)
        if burst_requests >= self.config.burst_limit:
            return False

        return True

    def _calculate_remaining(self, state: ClientState, now: datetime) -> int:
        """Calculate remaining requests in current window."""
        minute_ago = now - timedelta(minutes=1)
        minute_requests = sum(1 for t in state.requests if t > minute_ago)
        return max(0, self.config.requests_per_minute - minute_requests)

    def record_success(self, request_context: Dict[str, Any]) -> None:
        """Record successful request for metrics."""
        client_id = self._get_client_id(request_context)
        with self._lock:
            if client_id in self._clients:
                self._clients[client_id].consecutive_failures = 0

    def record_failure(self, request_context: Dict[str, Any]) -> int:
        """Record failed request and return consecutive failure count."""
        client_id = self._get_client_id(request_context)
        with self._lock:
            if client_id not in self._clients:
                self._clients[client_id] = ClientState(client_id=client_id)
            state = self._clients[client_id]
            state.consecutive_failures += 1
            state.failures.append(datetime.now())
            return state.consecutive_failures


class RateLimitDecorator:
    """Decorator for applying rate limits to functions."""

    def __init__(
        self,
        limiter: RateLimiter,
        get_context: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        self.limiter = limiter
        self.get_context = get_context

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Apply rate limiting to function."""

        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Extract request context
            if self.get_context:
                context = self.get_context(*args, **kwargs)
            else:
                context = self._default_context(*args, **kwargs)

            # Check rate limit
            allowed, headers = self.limiter.check_rate_limit(context)

            if not allowed:
                raise RateLimitExceeded("Rate limit exceeded", headers=headers)

            try:
                result = func(*args, **kwargs)
                self.limiter.record_success(context)
                return result
            except Exception:
                self.limiter.record_failure(context)
                raise

        return wrapper

    def _default_context(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Default context extractor."""
        # Try to find request-like object in args/kwargs
        for arg in args:
            if hasattr(arg, "remote_addr"):
                return {
                    "remote_addr": getattr(arg, "remote_addr", "unknown"),
                    "user_agent": getattr(arg, "user_agent", ""),
                }

        return {"remote_addr": "unknown", "user_agent": ""}


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str, headers: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.headers = headers or {}


# Global instances for framework-wide use
_global_rate_limiter: Optional[RateLimiter] = None
_global_brute_force_detector: Optional[BruteForceDetector] = None
_global_lock = threading.Lock()


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Get or create global rate limiter instance."""
    global _global_rate_limiter
    with _global_lock:
        if _global_rate_limiter is None:
            _global_rate_limiter = RateLimiter(config)
        return _global_rate_limiter


def get_brute_force_detector() -> BruteForceDetector:
    """Get or create global brute force detector instance."""
    global _global_brute_force_detector
    with _global_lock:
        if _global_brute_force_detector is None:
            _global_brute_force_detector = BruteForceDetector()  # type: ignore[no-untyped-call]
        return _global_brute_force_detector


def reset_global_instances() -> None:
    """Reset global instances (mainly for testing)."""
    global _global_rate_limiter, _global_brute_force_detector
    with _global_lock:
        _global_rate_limiter = None
        _global_brute_force_detector = None
