"""Per-chunk SHA-256 and CRC32C."""

from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import numpy as np

from apgi_framework.acquisition import (
    ChannelSpec,
    DeviceInfo,
    LocalRecorder,
    SampleChunk,
    crc32c,
)
from apgi_framework.acquisition.integrity import _crc32c_py

INFO = DeviceInfo("dev", "sim", "m", 100.0, (ChannelSpec("A"),), data_origin="synthetic")


def make_chunk(seq: int, first: int, n: int) -> SampleChunk:
    return SampleChunk("dev", seq, first, 10.0 + first / 100, np.full((n, 1), seq + 0.5), 1)


class TestCrc32c:
    def test_standard_check_value(self) -> None:
        # RFC 3720 / Castagnoli check value for "123456789"
        assert crc32c(b"123456789") == 0xE3069283
        assert _crc32c_py(b"123456789") == 0xE3069283

    def test_differs_from_zlib_crc32(self) -> None:
        assert crc32c(b"123456789") != zlib.crc32(b"123456789")

    def test_rfc3720_zero_and_ff_vectors(self) -> None:
        assert _crc32c_py(bytes(32)) == 0x8A9136AA
        assert _crc32c_py(b"\xff" * 32) == 0x62A8AB43

    def test_native_and_fallback_agree(self) -> None:
        data = bytes(range(256)) * 7
        assert crc32c(data) == _crc32c_py(data)


class TestChunkIndex:
    def test_every_chunk_line_has_sha256_and_crc32c(self, tmp_path: Path) -> None:
        rec = LocalRecorder(tmp_path / "s", INFO, fsync=False)
        rec.start()
        chunks = [make_chunk(0, 0, 10), make_chunk(1, 10, 7)]
        for c in chunks:
            rec.write_chunk(c)
        summary = rec.close()
        lines = [
            json.loads(line)
            for line in (tmp_path / "s" / summary.segments[0].index_file).read_text().splitlines()
        ]
        for c, line in zip(chunks, lines):
            raw = c.samples.astype("<f4").tobytes()
            assert line["sha256"] == hashlib.sha256(raw).hexdigest()
            assert line["crc32c"] == crc32c(raw)
            assert line["crc32"] == zlib.crc32(raw)  # kept for older readers

    def test_recovery_drops_a_chunk_whose_sha256_no_longer_matches(self, tmp_path: Path) -> None:
        rec = LocalRecorder(tmp_path / "s", INFO, fsync=False)
        rec.start()
        rec.write_chunk(make_chunk(0, 0, 10))
        rec.write_chunk(make_chunk(1, 10, 10))
        # simulate a crash: no close(), then flip a bit in the second chunk
        data = tmp_path / "s" / "data" / "segment-000000.f32"
        blob = bytearray(data.read_bytes())
        blob[-1] ^= 0x01
        data.write_bytes(bytes(blob))
        summary = LocalRecorder.recover(tmp_path / "s")
        assert summary.n_chunks == 1 and summary.n_samples == 10
