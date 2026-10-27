"""Server-side logic of the three-stage chunk upload (framework-agnostic).

Gigabytes never pass through the API process:

* **Stage A** :meth:`ChunkIngestService.init_upload` takes a manifest and returns a signed,
  resumable URL for the private bucket (V4 signing for GCS in production, supplied by an
  :class:`ObjectStore`).
* **Stage B** the client ``PUT`` s the bytes straight to the bucket.
* **Stage C** :meth:`ChunkIngestService.commit` checks that the object exists, its byte
  length, the caller's authorisation, the sequence number and the CRC32C, then enqueues a
  job that recomputes SHA-256 (:meth:`ChunkIngestService.run_verification_jobs`).

Chunk status:

``accepted``     SHA-256 recomputed from the stored bytes and equals the manifest.
``quarantined``  stored but not trusted yet: CRC32C mismatch, or a sequence gap. A chunk held
                 for a gap is re-examined once the missing predecessors are accepted.
``rejected``     unusable: object missing, wrong size, duplicate sequence, or SHA-256 mismatch.

An unauthorised caller gets :class:`UploadForbidden` and no state changes.
"""

from __future__ import annotations

import enum
import hashlib
import secrets
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .errors import AcquisitionError
from .integrity import crc32c


class UploadForbidden(AcquisitionError):
    """The principal may not upload to this session."""


class UnknownUpload(AcquisitionError, KeyError):
    """No upload with this id."""


class ChunkStatus(str, enum.Enum):
    PENDING = "pending"  # committed, SHA-256 verification queued
    ACCEPTED = "accepted"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ChunkManifest:
    session_id: str
    object_name: str  # path inside the session, e.g. data/segment-000000.f32
    size_bytes: int
    crc32c: int
    sha256: str
    sequence: Optional[int] = None  # None for non-chunk files (no sequence check)

    def key(self) -> str:
        return f"{self.session_id}/{self.object_name}"


@dataclass(frozen=True)
class InitResponse:
    upload_id: str
    signed_url: str
    expires_in_s: int
    already_accepted: bool = False


@dataclass
class UploadRecord:
    upload_id: str
    manifest: ChunkManifest
    principal: str
    status: ChunkStatus = ChunkStatus.PENDING
    committed: bool = False
    reason: str = ""


class ObjectStore(ABC):
    """The private bucket."""

    @abstractmethod
    def signed_upload_url(self, key: str, *, size_bytes: int, expires_in_s: int) -> str:
        """Return a time-limited URL the client PUTs the chunk to."""

    @abstractmethod
    def size(self, key: str) -> Optional[int]:
        """Stored byte length, or ``None`` if the object does not exist."""

    @abstractmethod
    def read(self, key: str) -> bytes:
        """Return the stored object bytes."""


class LocalObjectStore(ObjectStore):
    """Directory-backed store for tests and single-machine deployments.

    The "signed URL" is ``local://<token>/<key>``; :meth:`put_via_url` plays the role of the
    bucket accepting the direct ``PUT``.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._tokens: Dict[str, str] = {}

    def signed_upload_url(self, key: str, *, size_bytes: int, expires_in_s: int) -> str:
        token = secrets.token_urlsafe(16)
        self._tokens[token] = key
        return f"local://{token}/{key}"

    def put_via_url(self, url: str, data: bytes) -> None:
        token = url[len("local://") :].split("/", 1)[0]
        key = self._tokens.get(token)
        if key is None or not url.endswith("/" + key):
            raise PermissionError("signed URL is not valid")
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def size(self, key: str) -> Optional[int]:
        p = self.root / key
        return p.stat().st_size if p.is_file() else None

    def read(self, key: str) -> bytes:
        return (self.root / key).read_bytes()


Authorizer = Callable[[str, str], bool]  # (principal, session_id) -> allowed


@dataclass
class ChunkIngestService:
    store: ObjectStore
    authorize: Authorizer
    url_ttl_s: int = 900
    _uploads: Dict[str, UploadRecord] = field(default_factory=dict)
    _by_key: Dict[str, str] = field(default_factory=dict)
    _queue: List[str] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # -- stage A ------------------------------------------------------------
    def init_upload(self, manifest: ChunkManifest, principal: str) -> InitResponse:
        self._require_auth(principal, manifest.session_id)
        if manifest.size_bytes < 0 or not manifest.sha256:
            raise ValueError("manifest needs a non-negative size and a sha256")
        with self._lock:
            existing = self._by_key.get(manifest.key())
            if existing is not None:
                rec = self._uploads[existing]
                if rec.manifest.sha256 == manifest.sha256 and rec.status is ChunkStatus.ACCEPTED:
                    return InitResponse(rec.upload_id, "", 0, already_accepted=True)
                if rec.manifest.sha256 != manifest.sha256 and rec.status is ChunkStatus.ACCEPTED:
                    raise ValueError("an accepted object with different content already exists")
            upload_id = secrets.token_hex(12)
            self._uploads[upload_id] = UploadRecord(upload_id, manifest, principal)
            self._by_key[manifest.key()] = upload_id
        url = self.store.signed_upload_url(
            manifest.key(), size_bytes=manifest.size_bytes, expires_in_s=self.url_ttl_s
        )
        return InitResponse(upload_id, url, self.url_ttl_s)

    # -- stage C ------------------------------------------------------------
    def commit(self, upload_id: str, principal: str) -> UploadRecord:
        rec = self._get(upload_id)
        self._require_auth(principal, rec.manifest.session_id)
        if principal != rec.principal:
            raise UploadForbidden("upload was initialised by a different principal")
        with self._lock:
            if rec.committed:
                return rec  # idempotent
            rec.committed = True
            m = rec.manifest
            size = self.store.size(m.key())
            if size is None:
                return self._set(rec, ChunkStatus.REJECTED, "object does not exist in the bucket")
            if size != m.size_bytes:
                return self._set(
                    rec, ChunkStatus.REJECTED, f"size {size} != manifest {m.size_bytes}"
                )
            if m.sequence is not None and self._sequence_taken(rec):
                return self._set(rec, ChunkStatus.REJECTED, f"duplicate sequence {m.sequence}")
            data = self.store.read(m.key())
            if crc32c(data) != m.crc32c:
                return self._set(rec, ChunkStatus.QUARANTINED, "CRC32C mismatch")
            if m.sequence is not None and not self._predecessors_accepted(rec):
                return self._set(
                    rec, ChunkStatus.QUARANTINED, "sequence gap: earlier chunk not accepted yet"
                )
            rec.status, rec.reason = ChunkStatus.PENDING, ""
            self._queue.append(upload_id)
            return rec

    # -- verification job ------------------------------------------------------
    def run_verification_jobs(self) -> Dict[str, ChunkStatus]:
        """Recompute SHA-256 for queued chunks; also release gap-quarantined chunks."""
        done: Dict[str, ChunkStatus] = {}
        with self._lock:
            progressed = True
            while progressed:
                progressed = False
                for rec in self._uploads.values():
                    if rec.status is ChunkStatus.QUARANTINED and rec.reason.startswith(
                        "sequence gap"
                    ):
                        if self._predecessors_accepted(rec):
                            rec.status, rec.reason = ChunkStatus.PENDING, ""
                            self._queue.append(rec.upload_id)
                queue, self._queue = self._queue, []
                for upload_id in queue:
                    rec = self._uploads[upload_id]
                    if rec.status is not ChunkStatus.PENDING:
                        continue
                    digest = hashlib.sha256(self.store.read(rec.manifest.key())).hexdigest()
                    if digest == rec.manifest.sha256:
                        self._set(rec, ChunkStatus.ACCEPTED, "")
                    else:
                        self._set(rec, ChunkStatus.REJECTED, "SHA-256 mismatch")
                    done[upload_id] = rec.status
                    progressed = True
        return done

    # -- queries ------------------------------------------------------------
    def status(self, upload_id: str) -> UploadRecord:
        return self._get(upload_id)

    def status_for_key(self, key: str, principal: str) -> Optional[UploadRecord]:
        with self._lock:
            upload_id = self._by_key.get(key)
        if upload_id is None:
            return None
        rec = self._uploads[upload_id]
        self._require_auth(principal, rec.manifest.session_id)
        return rec

    # -- internals ------------------------------------------------------------
    def _get(self, upload_id: str) -> UploadRecord:
        try:
            return self._uploads[upload_id]
        except KeyError:
            raise UnknownUpload(upload_id) from None

    def _require_auth(self, principal: str, session_id: str) -> None:
        if not self.authorize(principal, session_id):
            raise UploadForbidden(f"{principal!r} may not upload to session {session_id!r}")

    @staticmethod
    def _set(rec: UploadRecord, status: ChunkStatus, reason: str) -> UploadRecord:
        rec.status, rec.reason = status, reason
        return rec

    def _chunks_of(self, rec: UploadRecord) -> List[UploadRecord]:
        sid = rec.manifest.session_id
        return [
            r
            for r in self._uploads.values()
            if r.manifest.session_id == sid and r.manifest.sequence is not None and r is not rec
        ]

    def _sequence_taken(self, rec: UploadRecord) -> bool:
        return any(
            r.manifest.sequence == rec.manifest.sequence
            and r.committed
            and r.status in (ChunkStatus.PENDING, ChunkStatus.ACCEPTED)
            for r in self._chunks_of(rec)
        )

    def _predecessors_accepted(self, rec: UploadRecord) -> bool:
        accepted = {
            r.manifest.sequence
            for r in self._chunks_of(rec)
            if r.status in (ChunkStatus.ACCEPTED, ChunkStatus.PENDING)
        }
        assert rec.manifest.sequence is not None
        return all(i in accepted for i in range(rec.manifest.sequence))
