"""Shared test helpers for the acquisition tests."""

from __future__ import annotations

import importlib.util

import pytest

NS = 1_000_000_000


class FakeClock:
    """Manually advanced nanosecond clock, shared by devices, markers and agents."""

    def __init__(self, start_ns: int = 5 * 10**12) -> None:
        self.now = start_ns

    def __call__(self) -> int:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += int(round(seconds * NS))


HAS_SQLCIPHER = (
    importlib.util.find_spec("sqlcipher3") is not None
    or importlib.util.find_spec("pysqlcipher3") is not None
)

requires_sqlcipher = pytest.mark.skipif(
    not HAS_SQLCIPHER, reason="SQLCipher is not installed (pip install sqlcipher3-binary on Linux)"
)

KEY = bytes(range(32))
