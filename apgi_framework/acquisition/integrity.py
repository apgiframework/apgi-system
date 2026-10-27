"""Checksums for recorded chunks: SHA-256 and CRC32C (Castagnoli).

``zlib.crc32`` uses the IEEE polynomial, which is not CRC32C. CRC32C is computed
with the optional ``crc32c`` package when installed and a table-driven pure
Python fallback otherwise; both give identical results.
"""

from __future__ import annotations

import hashlib
from typing import Callable, List

_POLY = 0x82F63B78  # reflected Castagnoli polynomial


def _build_table() -> List[int]:
    table: List[int] = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ _POLY if crc & 1 else crc >> 1
        table.append(crc)
    return table


_TABLE = _build_table()


def _crc32c_py(data: bytes, value: int = 0) -> int:
    crc = value ^ 0xFFFFFFFF
    for byte in data:
        crc = _TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _select() -> Callable[[bytes, int], int]:
    try:
        import crc32c as _native  # type: ignore[import-not-found]

        return lambda data, value=0: int(_native.crc32c(data, value))
    except ImportError:
        return _crc32c_py


_crc32c_impl = _select()


def crc32c(data: bytes, value: int = 0) -> int:
    """CRC32C of ``data`` as an unsigned 32-bit integer."""
    return _crc32c_impl(bytes(data), value)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunk_digests(raw: bytes) -> dict:
    """Both integrity values stored in a chunk's index line."""
    return {"crc32c": crc32c(raw), "sha256": sha256_hex(raw)}
