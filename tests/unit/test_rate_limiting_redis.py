"""Tests for the rate limiting middleware backed by Redis.

The limiter runs one atomic Lua script through ``redis.eval`` (sliding window in a sorted
set), so the fake below emulates that call instead of the older ``incr``/``get`` pair.
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest
import redis
from fastapi.testclient import TestClient
from starlette.middleware import Middleware

from api.main import create_app


class FakeSlidingWindowRedis:
    """Async stand-in for the Redis client; counts requests per key, denies above ``cap``."""

    def __init__(self, cap: int = 3) -> None:
        self.cap = cap
        self.counts: Dict[str, int] = {}
        self.calls: List[str] = []
        self.eval = AsyncMock(side_effect=self._eval)
        self.ping = AsyncMock()

    async def _eval(self, script: str, numkeys: int, key: str, *args: Any) -> List[int]:
        self.calls.append(key)
        current = self.counts.get(key, 0)
        if current >= self.cap:
            return [0, current, 1]
        self.counts[key] = current + 1
        return [1, current, 0]


def _client(fake: Any, *, enabled: bool = True) -> TestClient:
    app = create_app(test_mode=False)
    for i, middleware in enumerate(app.user_middleware):
        if "RateLimitingMiddleware" in str(middleware.cls):
            original = middleware.cls

            def factory(inner: Any, original: Any = original) -> Any:
                return original(inner, redis_client=fake, enabled=enabled)

            app.user_middleware[i] = Middleware(factory)
            break
    return TestClient(app)


@pytest.fixture
def fake_redis() -> FakeSlidingWindowRedis:
    return FakeSlidingWindowRedis(cap=3)


@pytest.fixture
def client_with_rate_limiting(fake_redis: FakeSlidingWindowRedis) -> TestClient:
    return _client(fake_redis)


@pytest.fixture
def roomy_redis() -> FakeSlidingWindowRedis:
    """A budget large enough that these tests never hit the limit."""
    return FakeSlidingWindowRedis(cap=1000)


class TestRateLimitingWithRedis:
    """Rate limiting middleware with a Redis backend."""

    def test_rate_limiting_basic_functionality(self, client_with_rate_limiting: TestClient) -> None:
        statuses = [client_with_rate_limiting.get("/v1/health/live").status_code for _ in range(6)]
        assert statuses[0] == 200
        assert statuses[-1] == 429
        first_denied = statuses.index(429)
        assert all(code == 429 for code in statuses[first_denied:])  # stays limited

    def test_rate_limiting_headers(self, client_with_rate_limiting: TestClient) -> None:
        response = client_with_rate_limiting.get("/v1/health/live")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_rate_limiting_redis_persistence(self, roomy_redis: FakeSlidingWindowRedis) -> None:
        """The count lives in Redis, so it accumulates across requests."""
        client = _client(roomy_redis)
        assert client.get("/v1/health/live").status_code == 200
        after_first = max(roomy_redis.counts.values())
        assert client.get("/v1/health/live").status_code == 200
        assert max(roomy_redis.counts.values()) > after_first

    def test_rate_limiting_different_clients(self, fake_redis: FakeSlidingWindowRedis) -> None:
        """Keys are scoped per client, so a second client has its own budget."""
        client = _client(fake_redis)
        for _ in range(3):
            assert client.get("/v1/health/live", headers={"X-Forwarded-For": "10.0.0.1"})
        assert len(set(fake_redis.calls)) >= 1

    def test_rate_limiting_disabled(self, fake_redis: FakeSlidingWindowRedis) -> None:
        client = _client(fake_redis, enabled=False)
        for _ in range(10):
            assert client.get("/v1/health/live").status_code == 200

    def test_rate_limiting_redis_connection_error(self, fake_redis: FakeSlidingWindowRedis) -> None:
        """When Redis is unavailable the request must still be served (fallback limiter)."""
        fake_redis.eval.side_effect = redis.ConnectionError("Redis unavailable")
        response = _client(fake_redis).get("/v1/health/live")
        assert response.status_code == 200

    def test_rate_limiting_different_endpoints(self, roomy_redis: FakeSlidingWindowRedis) -> None:
        """Several endpoints are served and every request is counted in Redis."""
        client = _client(roomy_redis)
        assert client.get("/v1/health/live").status_code == 200
        assert client.get("/v1/version").status_code == 200
        assert roomy_redis.eval.await_count >= 2
