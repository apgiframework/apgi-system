"""Encrypted session store (SQLCipher).

The store is the durable memory of an offline-first deployment: session state,
the state-change log, clock-sync points, markers and the upload queue all live
here, encrypted at rest. If SQLCipher is not installed the store refuses to open
rather than silently writing plaintext SQLite.

Keys are never hard-coded. Pass a 32-byte raw ``key``, or a ``passphrase`` that
SQLCipher stretches with PBKDF2, or use :meth:`SessionStore.from_env`.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .device import SyncProbe
from .errors import StoreError, StoreKeyError, StoreUnavailableError
from .markers import EventMarker

SCHEMA_VERSION = 3
KEY_ENV_VAR = "APGI_STORE_KEY"
SQLITE_MAGIC = b"SQLite format 3\x00"

STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

_SCHEMA = """
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    directory  TEXT NOT NULL,
    subject    TEXT,
    meta_json  TEXT NOT NULL DEFAULT '{}',
    created_ns INTEGER NOT NULL,
    updated_ns INTEGER NOT NULL
);
CREATE TABLE state_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    from_state TEXT,
    to_state   TEXT NOT NULL,
    event      TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    at_ns      INTEGER NOT NULL
);
CREATE TABLE sync_points (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    device_id    TEXT NOT NULL,
    host_send_ns INTEGER NOT NULL,
    device_time_s REAL NOT NULL,
    host_recv_ns INTEGER NOT NULL
);
CREATE TABLE markers (
    marker_id  TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    label      TEXT NOT NULL,
    onset_host_ns INTEGER NOT NULL,
    json       TEXT NOT NULL,
    seq_no     INTEGER NOT NULL DEFAULT 0,
    onset_utc_ns INTEGER
);
CREATE TABLE upload_items (
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    rel_path        TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    status          TEXT NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_ns INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT NOT NULL DEFAULT '',
    updated_ns      INTEGER NOT NULL,
    server_verified INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, rel_path)
);
CREATE INDEX idx_upload_due ON upload_items(status, next_attempt_ns);
CREATE TABLE chunk_journal (
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    device_id   TEXT NOT NULL,
    sequence    INTEGER NOT NULL,
    first_sample_index INTEGER NOT NULL,
    n_samples   INTEGER NOT NULL,
    host_receive_ns INTEGER NOT NULL,
    crc32c      INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    PRIMARY KEY (session_id, device_id, sequence)
);
CREATE TABLE milestones (
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    milestone  TEXT NOT NULL,
    at_ns      INTEGER NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (session_id, milestone)
);
CREATE INDEX idx_state_log_session ON state_log(session_id, id);
CREATE UNIQUE INDEX idx_markers_seq ON markers(session_id, seq_no) WHERE seq_no > 0;
"""

_MIGRATIONS = {
    3: (
        """CREATE TABLE IF NOT EXISTS chunk_journal (
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    device_id   TEXT NOT NULL,
    sequence    INTEGER NOT NULL,
    first_sample_index INTEGER NOT NULL,
    n_samples   INTEGER NOT NULL,
    host_receive_ns INTEGER NOT NULL,
    crc32c      INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    PRIMARY KEY (session_id, device_id, sequence)
)""",
        """CREATE TABLE IF NOT EXISTS milestones (
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    milestone  TEXT NOT NULL,
    at_ns      INTEGER NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (session_id, milestone)
)""",
    ),
    2: (
        "ALTER TABLE markers ADD COLUMN seq_no INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE markers ADD COLUMN onset_utc_ns INTEGER",
        "ALTER TABLE upload_items ADD COLUMN server_verified INTEGER NOT NULL DEFAULT 0",
        "CREATE UNIQUE INDEX idx_markers_seq ON markers(session_id, seq_no) WHERE seq_no > 0",
    ),
}


@dataclass(frozen=True)
class SessionRow:
    session_id: str
    state: str
    directory: str
    subject: Optional[str]
    meta: Dict[str, Any]
    created_ns: int
    updated_ns: int


@dataclass(frozen=True)
class StateLogEntry:
    from_state: Optional[str]
    to_state: str
    event: str
    detail: str
    at_ns: int


@dataclass(frozen=True)
class UploadItem:
    session_id: str
    rel_path: str
    sha256: str
    size_bytes: int
    status: str
    attempts: int
    next_attempt_ns: int
    last_error: str = ""
    server_verified: bool = False


@dataclass(frozen=True)
class StoredSyncPoint:
    device_id: str
    probe: SyncProbe
    extra: Dict[str, Any] = field(default_factory=dict)


def load_sqlcipher() -> ModuleType:
    """Import the SQLCipher DB-API module or raise :class:`StoreUnavailableError`."""
    for name in ("sqlcipher3.dbapi2", "pysqlcipher3.dbapi2"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise StoreUnavailableError(
        "SQLCipher is not installed. On Linux run `pip install sqlcipher3-binary`; "
        "on macOS/Windows install libsqlcipher and `pip install sqlcipher3`."
    )


def is_plaintext_sqlite(path: Path) -> bool:
    """True if ``path`` starts with the plain SQLite header (i.e. is *not* encrypted)."""
    with open(path, "rb") as fh:
        return fh.read(len(SQLITE_MAGIC)) == SQLITE_MAGIC


def _sha256_hex_of_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SessionStore:
    """Encrypted persistence for sessions, state history and the upload queue."""

    def __init__(
        self,
        path: Path,
        *,
        key: Optional[bytes] = None,
        passphrase: Optional[str] = None,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if (key is None) == (passphrase is None):
            raise ValueError("provide exactly one of key or passphrase")
        if key is not None and len(key) != 32:
            raise ValueError("key must be exactly 32 bytes")
        if passphrase is not None and not passphrase:
            raise ValueError("passphrase must not be empty")
        self._path = Path(path)
        self._clock_ns = clock_ns
        self._lock = threading.RLock()
        self._sqlcipher = load_sqlcipher()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Any = self._sqlcipher.connect(str(self._path), check_same_thread=False)
        try:
            self._unlock(key, passphrase)
            self._configure()
            self._migrate()
        except BaseException:
            self._conn.close()
            raise

    @classmethod
    def from_env(
        cls, path: Path, *, var: str = KEY_ENV_VAR, clock_ns: Callable[[], int] = time.time_ns
    ) -> "SessionStore":
        """Open with the key from an environment variable (64 hex chars or a passphrase)."""
        value = os.environ.get(var)
        if not value:
            raise StoreKeyError(f"environment variable {var} is not set")
        if len(value) == 64:
            try:
                return cls(path, key=bytes.fromhex(value), clock_ns=clock_ns)
            except ValueError:
                pass
        return cls(path, passphrase=value, clock_ns=clock_ns)

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SessionStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- sessions ---------------------------------------------------------
    def create_session(
        self,
        session_id: str,
        *,
        directory: str,
        subject: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        state: str = "created",
    ) -> None:
        now = self._clock_ns()
        with self._tx() as cur:
            try:
                cur.execute(
                    "INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
                    (session_id, state, directory, subject, json.dumps(meta or {}), now, now),
                )
            except self._sqlcipher.IntegrityError as exc:
                raise StoreError(f"session {session_id!r} already exists") from exc
            cur.execute(
                "INSERT INTO state_log(session_id, from_state, to_state, event, detail, at_ns)"
                " VALUES (?,?,?,?,?,?)",
                (session_id, None, state, "create", "", now),
            )

    def get_session(self, session_id: str) -> Optional[SessionRow]:
        with self._tx() as cur:
            cur.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,))
            row = cur.fetchone()
        return None if row is None else self._session_row(row)

    def list_sessions(self, states: Optional[Sequence[str]] = None) -> List[SessionRow]:
        with self._tx() as cur:
            if states:
                marks = ",".join("?" for _ in states)
                cur.execute(
                    # only "?" placeholders are interpolated, values are bound
                    f"SELECT * FROM sessions WHERE state IN ({marks}) ORDER BY created_ns",  # nosec B608
                    tuple(states),
                )
            else:
                cur.execute("SELECT * FROM sessions ORDER BY created_ns")
            return [self._session_row(r) for r in cur.fetchall()]

    def set_session_state(
        self,
        session_id: str,
        new_state: str,
        *,
        event: str,
        detail: str = "",
        expected_old: Optional[str] = None,
    ) -> None:
        """Change state and append to the log in one transaction."""
        now = self._clock_ns()
        with self._tx() as cur:
            cur.execute("SELECT state FROM sessions WHERE session_id=?", (session_id,))
            row = cur.fetchone()
            if row is None:
                raise StoreError(f"unknown session {session_id!r}")
            old = str(row[0])
            if expected_old is not None and old != expected_old:
                raise StoreError(f"session {session_id!r} is {old!r}, expected {expected_old!r}")
            cur.execute(
                "UPDATE sessions SET state=?, updated_ns=? WHERE session_id=?",
                (new_state, now, session_id),
            )
            cur.execute(
                "INSERT INTO state_log(session_id, from_state, to_state, event, detail, at_ns)"
                " VALUES (?,?,?,?,?,?)",
                (session_id, old, new_state, event, detail, now),
            )

    def state_history(self, session_id: str) -> List[StateLogEntry]:
        with self._tx() as cur:
            cur.execute(
                "SELECT from_state,to_state,event,detail,at_ns FROM state_log"
                " WHERE session_id=? ORDER BY id",
                (session_id,),
            )
            return [StateLogEntry(r[0], r[1], r[2], r[3], r[4]) for r in cur.fetchall()]

    # -- sync points and markers -----------------------------------------
    def add_sync_point(self, session_id: str, device_id: str, probe: SyncProbe) -> None:
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO sync_points(session_id,device_id,host_send_ns,device_time_s,"
                "host_recv_ns) VALUES (?,?,?,?,?)",
                (
                    session_id,
                    device_id,
                    probe.host_send_ns,
                    probe.device_time_s,
                    probe.host_recv_ns,
                ),
            )

    def list_sync_points(self, session_id: str, device_id: Optional[str] = None) -> List[SyncProbe]:
        query = "SELECT host_send_ns,device_time_s,host_recv_ns FROM sync_points WHERE session_id=?"
        args: Tuple[Any, ...] = (session_id,)
        if device_id is not None:
            query += " AND device_id=?"
            args += (device_id,)
        with self._tx() as cur:
            cur.execute(query + " ORDER BY id", args)
            return [SyncProbe(r[0], r[1], r[2]) for r in cur.fetchall()]

    def add_marker(self, session_id: str, marker: EventMarker) -> None:
        with self._tx() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO markers(marker_id,session_id,label,onset_host_ns,json,"
                "seq_no,onset_utc_ns) VALUES (?,?,?,?,?,?,?)",
                (
                    marker.marker_id,
                    session_id,
                    marker.label,
                    marker.onset_host_ns,
                    json.dumps(marker.to_dict()),
                    marker.seq_no,
                    marker.onset_utc_ns,
                ),
            )

    def list_markers(self, session_id: str) -> List[EventMarker]:
        with self._tx() as cur:
            cur.execute(
                "SELECT json FROM markers WHERE session_id=? ORDER BY onset_host_ns", (session_id,)
            )
            return [EventMarker.from_dict(json.loads(r[0])) for r in cur.fetchall()]

    # -- chunk journal and milestones --------------------------------------
    def journal_chunk(
        self,
        session_id: str,
        device_id: str,
        sequence: int,
        first_sample_index: int,
        n_samples: int,
        host_receive_ns: int,
        crc32c: int,
        sha256: str,
    ) -> None:
        """Record a chunk's index entry and digests in the encrypted journal (idempotent)."""
        with self._tx() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO chunk_journal VALUES (?,?,?,?,?,?,?,?)",
                (
                    session_id,
                    device_id,
                    sequence,
                    first_sample_index,
                    n_samples,
                    host_receive_ns,
                    crc32c,
                    sha256,
                ),
            )

    def list_chunk_journal(self, session_id: str) -> List[Dict[str, Any]]:
        with self._tx() as cur:
            cur.execute(
                "SELECT device_id,sequence,first_sample_index,n_samples,host_receive_ns,crc32c,"
                "sha256 FROM chunk_journal WHERE session_id=? ORDER BY device_id, sequence",
                (session_id,),
            )
            keys = (
                "device_id",
                "sequence",
                "first_sample_index",
                "n_samples",
                "host_receive_ns",
                "crc32c",
                "sha256",
            )
            return [dict(zip(keys, r)) for r in cur.fetchall()]

    def record_milestone(self, session_id: str, milestone: str, detail: str = "") -> bool:
        """Record a milestone once; returns False if it was already recorded."""
        with self._tx() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO milestones VALUES (?,?,?,?)",
                (session_id, milestone, self._clock_ns(), detail),
            )
            return bool(cur.rowcount)

    def milestones(self, session_id: str) -> Dict[str, int]:
        """Milestone name -> time (store clock, ns) for one session."""
        with self._tx() as cur:
            cur.execute("SELECT milestone, at_ns FROM milestones WHERE session_id=?", (session_id,))
            return {str(m): int(t) for m, t in cur.fetchall()}

    # -- upload queue -----------------------------------------------------
    def enqueue_file(self, session_id: str, rel_path: str, sha256: str, size_bytes: int) -> None:
        """Queue a file for upload. Re-queuing a changed file resets its attempts."""
        now = self._clock_ns()
        with self._tx() as cur:
            cur.execute(
                "SELECT sha256 FROM upload_items WHERE session_id=? AND rel_path=?",
                (session_id, rel_path),
            )
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO upload_items(session_id,rel_path,sha256,size_bytes,status,"
                    "updated_ns) VALUES (?,?,?,?,?,?)",
                    (session_id, rel_path, sha256, size_bytes, STATUS_PENDING, now),
                )
            elif row[0] != sha256:
                cur.execute(
                    "UPDATE upload_items SET sha256=?,size_bytes=?,status=?,attempts=0,"
                    "next_attempt_ns=0,last_error='',updated_ns=? WHERE session_id=? AND rel_path=?",
                    (sha256, size_bytes, STATUS_PENDING, now, session_id, rel_path),
                )

    def due_uploads(self, *, now_ns: Optional[int] = None, limit: int = 20) -> List[UploadItem]:
        """Pending items whose retry time has arrived, oldest first."""
        now = self._clock_ns() if now_ns is None else now_ns
        with self._tx() as cur:
            cur.execute(
                "SELECT session_id,rel_path,sha256,size_bytes,status,attempts,next_attempt_ns,"
                "last_error,server_verified FROM upload_items WHERE status=? AND next_attempt_ns<=?"
                " ORDER BY updated_ns, rel_path LIMIT ?",
                (STATUS_PENDING, now, limit),
            )
            return [
                UploadItem(
                    session_id=r[0],
                    rel_path=r[1],
                    sha256=r[2],
                    size_bytes=r[3],
                    status=r[4],
                    attempts=r[5],
                    next_attempt_ns=r[6],
                    last_error=r[7],
                    server_verified=bool(r[8]),
                )
                for r in cur.fetchall()
            ]

    def mark_upload_done(self, session_id: str, rel_path: str, *, server_verified: bool) -> None:
        """Mark an item uploaded.

        ``server_verified`` is True only after the server's own copy was read back and its
        SHA-256 matched.
        """
        now = self._clock_ns()
        with self._tx() as cur:
            cur.execute(
                "UPDATE upload_items SET status=?,last_error='',updated_ns=?,server_verified=?"
                " WHERE session_id=? AND rel_path=?",
                (STATUS_DONE, now, int(server_verified), session_id, rel_path),
            )

    def verified_paths(self, session_id: str) -> List[str]:
        """Relative paths of this session's files that are server-verified."""
        with self._tx() as cur:
            cur.execute(
                "SELECT rel_path FROM upload_items WHERE session_id=? AND server_verified=1"
                " ORDER BY rel_path",
                (session_id,),
            )
            return [str(r[0]) for r in cur.fetchall()]

    def all_server_verified(self, session_id: str) -> bool:
        """True when the session has queued files and every one is server-verified."""
        with self._tx() as cur:
            cur.execute(
                "SELECT COUNT(*), COALESCE(SUM(server_verified),0) FROM upload_items"
                " WHERE session_id=?",
                (session_id,),
            )
            total, verified = cur.fetchone()
        return int(total) > 0 and int(total) == int(verified)

    def mark_upload_failed(
        self,
        session_id: str,
        rel_path: str,
        error: str,
        *,
        retry_at_ns: Optional[int] = None,
        permanent: bool = False,
    ) -> None:
        now = self._clock_ns()
        status = STATUS_FAILED if permanent else STATUS_PENDING
        with self._tx() as cur:
            cur.execute(
                "UPDATE upload_items SET status=?,attempts=attempts+1,next_attempt_ns=?,"
                "last_error=?,updated_ns=? WHERE session_id=? AND rel_path=?",
                (status, retry_at_ns or 0, error[:500], now, session_id, rel_path),
            )

    def requeue_failed(self, session_id: str) -> int:
        with self._tx() as cur:
            cur.execute(
                "UPDATE upload_items SET status=?,attempts=0,next_attempt_ns=0,updated_ns=?"
                " WHERE session_id=? AND status=?",
                (STATUS_PENDING, self._clock_ns(), session_id, STATUS_FAILED),
            )
            return int(cur.rowcount)

    def upload_counts(self, session_id: str) -> Dict[str, int]:
        counts = {STATUS_PENDING: 0, STATUS_DONE: 0, STATUS_FAILED: 0}
        with self._tx() as cur:
            cur.execute(
                "SELECT status, COUNT(*) FROM upload_items WHERE session_id=? GROUP BY status",
                (session_id,),
            )
            for status, n in cur.fetchall():
                counts[str(status)] = int(n)
        return counts

    def sessions_with_pending_uploads(self) -> List[str]:
        with self._tx() as cur:
            cur.execute(
                "SELECT DISTINCT session_id FROM upload_items WHERE status=?", (STATUS_PENDING,)
            )
            return [str(r[0]) for r in cur.fetchall()]

    # -- internals --------------------------------------------------------
    def _unlock(self, key: Optional[bytes], passphrase: Optional[str]) -> None:
        if key is not None:
            pragma = f"PRAGMA key = \"x'{key.hex()}'\""
        else:
            assert passphrase is not None
            pragma = "PRAGMA key = '" + passphrase.replace("'", "''") + "'"
        self._conn.execute(pragma)
        version = self._conn.execute("PRAGMA cipher_version").fetchone()
        if not version or not version[0]:
            raise StoreUnavailableError(
                "the loaded SQLite library is not SQLCipher; refusing to store data unencrypted"
            )
        try:
            self._conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except self._sqlcipher.DatabaseError as exc:
            raise StoreKeyError("wrong key, or the file is not an encrypted APGI store") from exc

    def _configure(self) -> None:
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")

    def _migrate(self) -> None:
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise StoreError(f"store schema {version} is newer than this code ({SCHEMA_VERSION})")
        if version == 0:
            with self._lock, self._conn:
                for statement in _SCHEMA.strip().split(";"):
                    if statement.strip():
                        self._conn.execute(statement)
                _sql = f"PRAGMA user_version = {SCHEMA_VERSION}"  # nosemgrep: sql-injection-risk - PRAGMA with an int constant
                self._conn.execute(_sql)
        elif version < SCHEMA_VERSION:
            with self._lock, self._conn:
                for target in range(version + 1, SCHEMA_VERSION + 1):
                    for statement in _MIGRATIONS[target]:
                        self._conn.execute(statement)
                _sql = f"PRAGMA user_version = {SCHEMA_VERSION}"  # nosemgrep: sql-injection-risk - PRAGMA with an int constant
                self._conn.execute(_sql)

    @contextlib.contextmanager
    def _tx(self) -> Iterator[Any]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                with self._conn:  # commits on success, rolls back on error
                    yield cur
            finally:
                cur.close()

    @staticmethod
    def _session_row(row: Any) -> SessionRow:
        return SessionRow(
            session_id=row[0],
            state=row[1],
            directory=row[2],
            subject=row[3],
            meta=json.loads(row[4]),
            created_ns=row[5],
            updated_ns=row[6],
        )
