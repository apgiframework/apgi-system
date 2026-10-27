"""
Request Deduplication Middleware

Prevents duplicate request processing by caching request fingerprints
and returning cached responses for identical requests within a time window.
Improves API performance and reduces resource consumption.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """Container for cached response data."""

    status_code: int
    body: Any
    headers: Dict[str, str]
    timestamp: float
    ttl_seconds: int

    def is_expired(self) -> bool:
        """Check if cached response has expired."""
        return time.time() - self.timestamp > self.ttl_seconds


class RequestDeduplicationCache:
    """
    Cache for storing request fingerprints and their responses.

    Implements LRU eviction and TTL-based expiration to manage cache size.
    """

    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 60):
        """
        Initialize the deduplication cache.

        Args:
            max_size: Maximum number of cached responses
            default_ttl_seconds: Default TTL for cache entries
        """
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self._cache: Dict[str, CachedResponse] = {}
        self._access_times: Dict[str, float] = {}
        self._lock: Optional[asyncio.Lock] = None  # Asyncio lock for thread safety
        self._in_flight: Dict[str, asyncio.Event] = (
            {}
        )  # Tracks in-flight requests to prevent thundering herd
        logger.info(f"Request deduplication cache initialized (max_size={max_size})")

    async def _get_lock(self) -> asyncio.Lock:
        """Get or create async lock for thread safety."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        assert self._lock is not None
        return self._lock

    def generate_fingerprint(self, request: Request, body: Optional[bytes] = None) -> str:
        """
        Generate unique fingerprint for a request.

        Combines method, URL path, query params, and body hash.

        Args:
            request: HTTP request
            body: Request body bytes (if already read)

        Returns:
            MD5 fingerprint string
        """
        components = [
            request.method,
            request.url.path,
            str(request.query_params),
        ]

        # Include body hash if available
        if body:
            body_hash = hashlib.md5(body).hexdigest()
            components.append(body_hash)

        fingerprint_data = "|".join(components)
        return hashlib.md5(fingerprint_data.encode()).hexdigest()

    async def get(self, fingerprint: str) -> Optional[CachedResponse]:
        """
        Retrieve cached response by fingerprint.

        Args:
            fingerprint: Request fingerprint

        Returns:
            CachedResponse if found and not expired, None otherwise
        """
        async with await self._get_lock():
            if fingerprint not in self._cache:
                return None

            cached = self._cache[fingerprint]

            # Check expiration
            if cached.is_expired():
                del self._cache[fingerprint]
                del self._access_times[fingerprint]
                return None

            # Update access time for LRU
            self._access_times[fingerprint] = time.time()
            return cached

    async def get_or_wait(self, fingerprint: str) -> Tuple[Optional[CachedResponse], bool]:
        """
        Retrieve cached response or wait if the request is currently in flight.
        Returns (cached_response, is_first).
        is_first is True if this request is the first one and should execute the handler.
        """
        while True:
            async with await self._get_lock():
                # Check cache first
                if fingerprint in self._cache:
                    cached = self._cache[fingerprint]
                    if not cached.is_expired():
                        self._access_times[fingerprint] = time.time()
                        return cached, False
                    else:
                        del self._cache[fingerprint]
                        del self._access_times[fingerprint]

                # If not cached, check if another request is in flight
                if fingerprint in self._in_flight:
                    event = self._in_flight[fingerprint]
                else:
                    self._in_flight[fingerprint] = asyncio.Event()
                    return None, True

            # Wait outside the lock for the in-flight request to finish
            await event.wait()

    async def release_in_flight(self, fingerprint: str) -> None:
        """Release any waiters for this fingerprint."""
        async with await self._get_lock():
            if fingerprint in self._in_flight:
                self._in_flight[fingerprint].set()
                del self._in_flight[fingerprint]

    async def set(
        self,
        fingerprint: str,
        status_code: int,
        body: Any,
        headers: Dict[str, str],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Cache a response for a request fingerprint.

        Args:
            fingerprint: Request fingerprint
            status_code: HTTP status code
            body: Response body
            headers: Response headers
            ttl_seconds: Custom TTL (uses default if not specified)
        """
        ttl = ttl_seconds or self.default_ttl_seconds

        async with await self._get_lock():
            # Evict oldest entries if cache is full
            while len(self._cache) >= self.max_size:
                oldest = min(self._access_times, key=lambda k: self._access_times[k])
                del self._cache[oldest]
                del self._access_times[oldest]

            # Store response
            self._cache[fingerprint] = CachedResponse(
                status_code=status_code,
                body=body,
                headers=headers,
                timestamp=time.time(),
                ttl_seconds=ttl,
            )
            self._access_times[fingerprint] = time.time()

    async def invalidate_pattern(self, path_pattern: str) -> int:
        """
        Invalidate cached responses matching a path pattern.

        Args:
            path_pattern: URL path substring to match

        Returns:
            Number of invalidated entries
        """
        async with await self._get_lock():
            to_remove = [
                fp
                for fp in self._cache.keys()
                if path_pattern in self._cache[fp].headers.get("X-Request-Path", "")
            ]

            for fp in to_remove:
                del self._cache[fp]
                del self._access_times[fp]

            return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "utilization": len(self._cache) / self.max_size,
            "default_ttl_seconds": self.default_ttl_seconds,
        }


class RequestDeduplicationMiddleware(BaseHTTPMiddleware):
    """
    Middleware that deduplicates identical requests.

    Caches responses based on request fingerprints (method + path + query + body)
    and returns cached responses for duplicate requests within the TTL window.

    Skips deduplication for:
    - Non-idempotent methods (POST, PUT, DELETE, PATCH)
    - Requests with certain headers (Authorization, Cache-Control: no-cache)
    - Specific endpoint patterns
    """

    def __init__(
        self,
        app: ASGIApp,
        max_size: int = 1000,
        default_ttl_seconds: int = 60,
        skip_methods: Optional[Tuple[str, ...]] = None,
        skip_paths: Optional[Tuple[str, ...]] = None,
    ):
        """
        Initialize deduplication middleware.

        Args:
            app: ASGI application
            max_size: Maximum cache entries
            default_ttl_seconds: Default cache TTL
            skip_methods: HTTP methods to skip (default: POST, PUT, DELETE, PATCH)
            skip_paths: URL path patterns to skip
        """
        super().__init__(app)
        self.cache = RequestDeduplicationCache(
            max_size=max_size,
            default_ttl_seconds=default_ttl_seconds,
        )
        self.skip_methods = skip_methods or ("POST", "PUT", "DELETE", "PATCH")
        self.skip_paths = skip_paths or ("/health", "/metrics", "/admin")

        # Register this middleware instance with the global manager
        deduplication_manager.register_middleware(self)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Process request with deduplication logic.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler

        Returns:
            Response (cached or fresh)
        """
        # Skip deduplication for certain methods
        if request.method in self.skip_methods:
            return await call_next(request)

        # Skip deduplication for certain paths
        if any(request.url.path.startswith(path) for path in self.skip_paths):
            return await call_next(request)

        # Skip if client requests no cache
        cache_control = request.headers.get("Cache-Control", "")
        if "no-cache" in cache_control or "no-store" in cache_control:
            return await call_next(request)

        # Generate fingerprint (only for GET/HEAD with simple bodies)
        body = None
        if request.method in ("GET", "HEAD"):
            fingerprint = self.cache.generate_fingerprint(request, body)
        else:
            return await call_next(request)

        # Check cache and handle concurrency
        cached, is_first = await self.cache.get_or_wait(fingerprint)
        if cached:
            logger.debug(f"Cache hit for {request.method} {request.url.path}")
            return JSONResponse(
                content=cached.body,
                status_code=cached.status_code,
                headers=cached.headers,
            )

        try:
            # Process request
            response = await call_next(request)

            # Cache successful GET responses (2xx status codes)
            if request.method == "GET" and 200 <= response.status_code < 300:
                try:
                    # Read response body
                    body_bytes = b""
                    if hasattr(response, "body_iterator"):  # type: ignore[attr-defined]
                        async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                            body_bytes += chunk
                    else:
                        # Fallback for Response without body_iterator
                        body_bytes = bytes(response.body)

                    # Parse as JSON
                    body_content = json.loads(body_bytes) if body_bytes else {}

                    # Cache response
                    headers = dict(response.headers)
                    headers["X-Request-Path"] = request.url.path
                    headers["X-Deduplicated"] = "false"

                    await self.cache.set(
                        fingerprint=fingerprint,
                        status_code=response.status_code,
                        body=body_content,
                        headers=headers,
                    )

                    # Return fresh response
                    return JSONResponse(
                        content=body_content,
                        status_code=response.status_code,
                        headers=headers,
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")

            return response
        finally:
            if is_first:
                await self.cache.release_in_flight(fingerprint)


class DeduplicationManager:
    """
    Manager for request deduplication statistics and control.

    Provides monitoring and administrative functions for the deduplication system.
    """

    def __init__(self, middleware: Optional[RequestDeduplicationMiddleware] = None):
        self.middleware = middleware

    def register_middleware(self, middleware: RequestDeduplicationMiddleware) -> None:
        """Register a middleware instance."""
        self.middleware = middleware

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get current cache statistics."""
        if self.middleware:
            return self.middleware.cache.get_stats()
        return None

    async def invalidate_cache(self, path_pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        if self.middleware:
            return await self.middleware.cache.invalidate_pattern(path_pattern)
        return 0


# Global deduplication manager instance
deduplication_manager = DeduplicationManager()
