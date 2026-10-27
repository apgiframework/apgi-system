"""Crash-safe local recorder.

Layout of a session directory::

    recording.json            metadata, written at start and updated at close
    markers.jsonl             one JSON event per line (host-clock onsets)
    data/segment-000000.f32   little-endian float32, samples interleaved by channel
    data/chunks-000000.jsonl  one index line per chunk: offsets, times, CRC32
    manifest.json             sealed segments with SHA-256 (written at close)
    COMPLETE                  empty file, present only after a clean close

Data bytes are flushed before their index line, so the index never describes
bytes that are missing. After a crash :meth:`LocalRecorder.recover` drops any
partially written trailing chunk and seals what is left. Nothing here touches the
network; uploading is a separate concern.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from numpy.typing import NDArray

from .device import DeviceInfo, SampleChunk, SyncProbe
from .errors import RecorderError
from .markers import EventMarker

FORMAT_VERSION = 1
DTYPE = np.dtype("<f4")

FileReadyCallback = Callable[[str, str, int], None]


@dataclass(frozen=True)
class SegmentRecord:
    """A sealed segment: immutable once written, safe to upload."""

    index: int
    data_file: str
    index_file: str
    n_samples: int
    n_chunks: int
    size_bytes: int
    sha256: str
    index_sha256: str
    first_sample_index: int

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class RecordingSummary:
    session_dir: Path
    n_samples: int
    n_chunks: int
    n_markers: int
    segments: Tuple[SegmentRecord, ...]
    complete: bool


@dataclass(frozen=True)
class ChunkRecord:
    sequence: int
    first_sample_index: int
    n_samples: int
    first_device_time_s: float
    host_receive_ns: int


@dataclass
class Recording:
    """A recording read back from disk."""

    info: DeviceInfo
    meta: Dict[str, Any]
    samples: NDArray[np.float32]
    chunks: List[ChunkRecord]
    markers: List[EventMarker]
    gaps: List[Tuple[int, int]] = field(default_factory=list)
    complete: bool = False
    sync_probes: List[SyncProbe] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def sample0_device_time_s(self) -> float:
        """Device-clock time of sample index 0."""
        first = self.chunks[0]
        return first.first_device_time_s - first.first_sample_index / self.info.sampling_rate_hz


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class LocalRecorder:
    """Append chunks and markers to disk, rotating and sealing segments.

    Args:
        session_dir: Directory to create (must not already hold a recording).
        info: Description of the recorded device.
        session_meta: Free-form metadata stored in ``recording.json``.
        segment_samples: Samples per segment before rotation.
        fsync: ``fsync`` after every flush (disable only in tests).
        on_file_ready: Called as ``(relative_path, sha256, size)`` when a file can
            no longer change, so an upload queue can pick it up.
    """

    def __init__(
        self,
        session_dir: Path,
        info: DeviceInfo,
        *,
        session_meta: Optional[Dict[str, Any]] = None,
        segment_samples: int = 60_000,
        fsync: bool = True,
        on_file_ready: Optional[FileReadyCallback] = None,
    ) -> None:
        if segment_samples <= 0:
            raise ValueError("segment_samples must be positive")
        self._dir = Path(session_dir)
        self._info = info
        self._meta = dict(session_meta or {})
        self._segment_samples = segment_samples
        self._fsync = fsync
        self._on_file_ready = on_file_ready
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._seg_index = 0
        self._seg_samples = 0
        self._seg_chunks = 0
        self._seg_first_sample = 0
        self._data_fh: Optional[Any] = None
        self._index_fh: Optional[Any] = None
        self._markers_fh: Optional[Any] = None
        self._sync_fh: Optional[Any] = None
        self._segments: List[SegmentRecord] = []
        self._total_samples = 0
        self._total_chunks = 0
        self._n_markers = 0

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RecorderError("recorder already started")
            if (self._dir / "recording.json").exists():
                raise RecorderError(f"{self._dir} already holds a recording")
            (self._dir / "data").mkdir(parents=True, exist_ok=True)
            _atomic_write_json(
                self._dir / "recording.json",
                {
                    "format_version": FORMAT_VERSION,
                    "dtype": DTYPE.str,
                    "device": self._info.to_dict(),
                    "meta": self._meta,
                    "complete": False,
                },
            )
            self._markers_fh = open(self._dir / "markers.jsonl", "a", encoding="utf-8")
            self._sync_fh = open(self._dir / "sync.jsonl", "a", encoding="utf-8")
            self._open_segment()
            self._started = True

    def write_chunk(self, chunk: SampleChunk) -> None:
        with self._lock:
            self._require_open()
            if chunk.samples.shape[1] != self._info.n_channels:
                raise RecorderError(
                    f"chunk has {chunk.samples.shape[1]} channels, device has "
                    f"{self._info.n_channels}"
                )
            if self._seg_samples >= self._segment_samples:
                self._seal_segment()
                self._seg_first_sample = chunk.first_sample_index
                self._open_segment()
            raw = np.ascontiguousarray(chunk.samples, dtype=DTYPE).tobytes()
            assert self._data_fh is not None and self._index_fh is not None
            offset = self._data_fh.tell()
            self._data_fh.write(raw)
            self._flush(self._data_fh)  # data first, so the index never overruns it
            line = {
                "seq": chunk.sequence,
                "first": chunk.first_sample_index,
                "n": chunk.n_samples,
                "t0": chunk.first_device_time_s,
                "host_ns": chunk.host_receive_ns,
                "offset": offset,
                "crc32": zlib.crc32(raw),
            }
            self._index_fh.write(json.dumps(line) + "\n")
            self._flush(self._index_fh)
            self._seg_samples += chunk.n_samples
            self._seg_chunks += 1
            self._total_samples += chunk.n_samples
            self._total_chunks += 1

    def write_marker(self, marker: EventMarker) -> None:
        with self._lock:
            self._require_open()
            assert self._markers_fh is not None
            self._markers_fh.write(json.dumps(marker.to_dict()) + "\n")
            self._flush(self._markers_fh)
            self._n_markers += 1

    def write_sync_probe(self, probe: SyncProbe) -> None:
        """Persist a clock-sync measurement so exports need nothing but this folder."""
        with self._lock:
            self._require_open()
            assert self._sync_fh is not None
            line = {
                "host_send_ns": probe.host_send_ns,
                "device_time_s": probe.device_time_s,
                "host_recv_ns": probe.host_recv_ns,
            }
            self._sync_fh.write(json.dumps(line) + "\n")
            self._flush(self._sync_fh)

    def close(self) -> RecordingSummary:
        with self._lock:
            if self._closed:
                raise RecorderError("recorder already closed")
            self._require_open()
            self._seal_segment()
            assert self._markers_fh is not None
            self._markers_fh.close()
            self._markers_fh = None
            if self._sync_fh is not None:
                self._sync_fh.close()
                self._sync_fh = None
            self._closed = True
            return self._finalize()

    @property
    def segments(self) -> Tuple[SegmentRecord, ...]:
        with self._lock:
            return tuple(self._segments)

    def __enter__(self) -> "LocalRecorder":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        if not self._closed and self._started:
            self.close()

    # -- recovery and reading --------------------------------------------
    @classmethod
    def recover(
        cls, session_dir: Path, *, on_file_ready: Optional[FileReadyCallback] = None
    ) -> RecordingSummary:
        """Repair a recording left behind by a crash and finalize it."""
        session_dir = Path(session_dir)
        meta = _read_meta(session_dir)
        info = DeviceInfo.from_dict(meta["device"])
        recorder = cls(
            session_dir, info, session_meta=meta.get("meta"), on_file_ready=on_file_ready
        )
        recorder._dir.mkdir(parents=True, exist_ok=True)
        sealed = _read_manifest_segments(session_dir)
        sealed_ids = {s.index for s in sealed}
        recorder._segments = list(sealed)
        n_samples = sum(s.n_samples for s in sealed)
        n_chunks = sum(s.n_chunks for s in sealed)
        for data_path in sorted((session_dir / "data").glob("segment-*.f32")):
            idx = int(data_path.stem.split("-")[1])
            if idx in sealed_ids:
                continue
            record = _repair_segment(session_dir, idx, info)
            if record is not None:
                recorder._segments.append(record)
                n_samples += record.n_samples
                n_chunks += record.n_chunks
                recorder._notify(record)
        recorder._total_samples = n_samples
        recorder._total_chunks = n_chunks
        recorder._n_markers = _repair_markers(session_dir / "markers.jsonl")
        _repair_markers(session_dir / "sync.jsonl")
        recorder._started = True
        recorder._closed = True
        recorder._segments.sort(key=lambda s: s.index)
        return recorder._finalize()

    @staticmethod
    def read(session_dir: Path) -> Recording:
        """Load a finished or recovered recording into memory."""
        session_dir = Path(session_dir)
        meta = _read_meta(session_dir)
        info = DeviceInfo.from_dict(meta["device"])
        segments = _read_manifest_segments(session_dir)
        if not segments:
            raise RecorderError(f"{session_dir} has no sealed segments; run recover() first")
        chunks: List[ChunkRecord] = []
        parts: List[Tuple[int, NDArray[np.float32]]] = []
        for seg in segments:
            raw = np.fromfile(session_dir / "data" / seg.data_file.split("/")[-1], dtype=DTYPE)
            if seg.n_chunks and raw.size % info.n_channels:
                raise RecorderError(f"{seg.data_file} length is not a multiple of channel count")
            block = raw.reshape(-1, info.n_channels)
            for entry in _read_index(session_dir / "data" / seg.index_file.split("/")[-1]):
                chunks.append(
                    ChunkRecord(
                        sequence=int(entry["seq"]),
                        first_sample_index=int(entry["first"]),
                        n_samples=int(entry["n"]),
                        first_device_time_s=float(entry["t0"]),
                        host_receive_ns=int(entry["host_ns"]),
                    )
                )
                start = int(entry["offset"]) // (DTYPE.itemsize * info.n_channels)
                parts.append((int(entry["first"]), block[start : start + int(entry["n"])]))
        if not chunks:
            raise RecorderError(f"{session_dir} contains no chunks")
        total = max(first + part.shape[0] for first, part in parts)
        samples = np.zeros((total, info.n_channels), dtype=np.float32)
        covered = np.zeros(total, dtype=bool)
        for first, part in parts:
            samples[first : first + part.shape[0]] = part
            covered[first : first + part.shape[0]] = True
        return Recording(
            info=info,
            meta=meta,
            samples=samples,
            chunks=chunks,
            markers=_read_markers(session_dir / "markers.jsonl"),
            gaps=_find_gaps(covered),
            complete=(session_dir / "COMPLETE").exists(),
            sync_probes=_read_probes(session_dir / "sync.jsonl"),
        )

    # -- internals --------------------------------------------------------
    def _require_open(self) -> None:
        if not self._started:
            raise RecorderError("recorder not started")
        if self._closed:
            raise RecorderError("recorder is closed")

    def _flush(self, fh: Any) -> None:
        fh.flush()
        if self._fsync:
            os.fsync(fh.fileno())

    def _seg_names(self, index: int) -> Tuple[str, str]:
        return f"segment-{index:06d}.f32", f"chunks-{index:06d}.jsonl"

    def _open_segment(self) -> None:
        data_name, index_name = self._seg_names(self._seg_index)
        self._data_fh = open(self._dir / "data" / data_name, "ab")
        self._index_fh = open(self._dir / "data" / index_name, "a", encoding="utf-8")
        self._seg_samples = 0
        self._seg_chunks = 0

    def _seal_segment(self) -> None:
        if self._data_fh is None or self._index_fh is None:
            return
        self._data_fh.close()
        self._index_fh.close()
        self._data_fh = self._index_fh = None
        data_name, index_name = self._seg_names(self._seg_index)
        data_path = self._dir / "data" / data_name
        index_path = self._dir / "data" / index_name
        if self._seg_chunks == 0:
            data_path.unlink(missing_ok=True)
            index_path.unlink(missing_ok=True)
            return
        record = SegmentRecord(
            index=self._seg_index,
            data_file=f"data/{data_name}",
            index_file=f"data/{index_name}",
            n_samples=self._seg_samples,
            n_chunks=self._seg_chunks,
            size_bytes=data_path.stat().st_size,
            sha256=_sha256_file(data_path),
            index_sha256=_sha256_file(index_path),
            first_sample_index=self._seg_first_sample,
        )
        self._segments.append(record)
        self._notify(record)
        self._seg_index += 1

    def _notify(self, record: SegmentRecord) -> None:
        if self._on_file_ready is None:
            return
        self._on_file_ready(record.data_file, record.sha256, record.size_bytes)
        index_path = self._dir / record.index_file
        self._on_file_ready(record.index_file, record.index_sha256, index_path.stat().st_size)

    def _finalize(self) -> RecordingSummary:
        manifest = {
            "format_version": FORMAT_VERSION,
            "n_samples": self._total_samples,
            "n_chunks": self._total_chunks,
            "n_markers": self._n_markers,
            "segments": [s.to_dict() for s in self._segments],
        }
        _atomic_write_json(self._dir / "manifest.json", manifest)
        meta = _read_meta(self._dir)
        meta["complete"] = True
        _atomic_write_json(self._dir / "recording.json", meta)
        (self._dir / "COMPLETE").write_text("", encoding="utf-8")
        if self._on_file_ready is not None:
            for name in ("markers.jsonl", "sync.jsonl", "manifest.json", "recording.json"):
                path = self._dir / name
                if path.exists():
                    self._on_file_ready(name, _sha256_file(path), path.stat().st_size)
        return RecordingSummary(
            session_dir=self._dir,
            n_samples=self._total_samples,
            n_chunks=self._total_chunks,
            n_markers=self._n_markers,
            segments=tuple(self._segments),
            complete=True,
        )


# -- module helpers --------------------------------------------------------
def _read_meta(session_dir: Path) -> Dict[str, Any]:
    path = session_dir / "recording.json"
    if not path.exists():
        raise RecorderError(f"{session_dir} is not a recording (no recording.json)")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RecorderError("recording.json is malformed")
    if data.get("format_version") != FORMAT_VERSION:
        raise RecorderError(f"unsupported recording format {data.get('format_version')!r}")
    return data


def _read_manifest_segments(session_dir: Path) -> List[SegmentRecord]:
    path = session_dir / "manifest.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [SegmentRecord(**s) for s in data.get("segments", [])]


def _read_index(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _read_markers(path: Path) -> List[EventMarker]:
    if not path.exists():
        return []
    markers = [EventMarker.from_dict(e) for e in _read_index(path)]
    return sorted(markers, key=lambda m: m.onset_host_ns)


def _read_probes(path: Path) -> List[SyncProbe]:
    if not path.exists():
        return []
    return [
        SyncProbe(int(e["host_send_ns"]), float(e["device_time_s"]), int(e["host_recv_ns"]))
        for e in _read_index(path)
    ]


def _find_gaps(covered: NDArray[np.bool_]) -> List[Tuple[int, int]]:
    gaps: List[Tuple[int, int]] = []
    missing = ~covered
    if not missing.any():
        return gaps
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for start, end in zip(starts, ends):
        gaps.append((int(start), int(end - start)))
    return gaps


def _repair_segment(session_dir: Path, index: int, info: DeviceInfo) -> Optional[SegmentRecord]:
    """Keep only chunks whose index line and CRC are intact; seal the segment."""
    data_path = session_dir / "data" / f"segment-{index:06d}.f32"
    index_path = session_dir / "data" / f"chunks-{index:06d}.jsonl"
    frame = DTYPE.itemsize * info.n_channels
    kept: List[Dict[str, Any]] = []
    good_bytes = 0
    if index_path.exists():
        with open(data_path, "rb") as dfh, open(index_path, encoding="utf-8") as ifh:
            for line in ifh:
                try:
                    entry = json.loads(line)
                    size = int(entry["n"]) * frame
                    if int(entry["offset"]) != good_bytes:
                        break
                    dfh.seek(good_bytes)
                    raw = dfh.read(size)
                except (ValueError, KeyError):
                    break
                if len(raw) != size or zlib.crc32(raw) != int(entry["crc32"]):
                    break
                kept.append(entry)
                good_bytes += size
    if not kept:
        data_path.unlink(missing_ok=True)
        index_path.unlink(missing_ok=True)
        return None
    with open(data_path, "r+b") as fh:
        fh.truncate(good_bytes)
    with open(index_path, "w", encoding="utf-8") as fh:
        for entry in kept:
            fh.write(json.dumps(entry) + "\n")
    return SegmentRecord(
        index=index,
        data_file=f"data/{data_path.name}",
        index_file=f"data/{index_path.name}",
        n_samples=sum(int(e["n"]) for e in kept),
        n_chunks=len(kept),
        size_bytes=good_bytes,
        sha256=_sha256_file(data_path),
        index_sha256=_sha256_file(index_path),
        first_sample_index=int(kept[0]["first"]),
    )


def _repair_markers(path: Path) -> int:
    """Drop a torn final line from ``markers.jsonl``; return the marker count."""
    if not path.exists():
        return 0
    good: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            json.loads(line)
        except ValueError:
            break
        good.append(line)
    path.write_text("".join(f"{ln}\n" for ln in good), encoding="utf-8")
    return len(good)
