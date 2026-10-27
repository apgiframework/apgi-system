"""Background upload of sealed recording files.

The agent drains the persistent upload queue in the encrypted store. It is safe
to run on a flaky link: uploads are idempotent (checked by SHA-256), failures
back off exponentially with jitter, loss of the network pauses the queue without
consuming retries, and the queue survives restarts.
"""

from __future__ import annotations

import os
import random
import shutil
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

if TYPE_CHECKING:
    import httpx

from .errors import NetworkUnavailable, PermanentTransportError, TransportError
from .state_machine import SessionEvent, SessionMachine, SessionState
from .store import SessionStore, UploadItem, _sha256_hex_of_file


class Transport(ABC):
    """Destination for uploaded files."""

    @abstractmethod
    def exists(self, remote_key: str, sha256: str) -> bool:
        """True if ``remote_key`` already holds a file with this digest."""

    @abstractmethod
    def upload(self, local_path: Path, remote_key: str, sha256: str) -> None:
        """Store the file; must be idempotent. Raise a :class:`TransportError` on failure."""


def _safe_key(remote_key: str) -> str:
    parts = remote_key.replace("\\", "/").split("/")
    if remote_key.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise PermanentTransportError(f"unsafe remote key {remote_key!r}")
    return "/".join(parts)


class LocalDirectoryTransport(Transport):
    """Copy files into a directory (a mounted share, USB disk or staging area)."""

    def __init__(self, destination: Path) -> None:
        self._dest = Path(destination)

    def exists(self, remote_key: str, sha256: str) -> bool:
        target = self._dest / _safe_key(remote_key)
        return target.is_file() and _sha256_hex_of_file(target) == sha256

    def upload(self, local_path: Path, remote_key: str, sha256: str) -> None:
        target = self._dest / _safe_key(remote_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        try:
            shutil.copyfile(local_path, tmp)
            if _sha256_hex_of_file(tmp) != sha256:
                raise TransportError(f"checksum mismatch after copying {remote_key}")
            os.replace(tmp, target)
        except OSError as exc:
            raise TransportError(f"copy failed for {remote_key}: {exc}") from exc
        finally:
            tmp.unlink(missing_ok=True)


class HttpTransport(Transport):
    """``HEAD``/``PUT`` files to ``{base_url}/{remote_key}`` with an ``X-Content-SHA256`` header."""

    def __init__(
        self,
        base_url: str,
        *,
        client: Optional["httpx.Client"] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_s: float = 30.0,
    ) -> None:
        import httpx  # imported lazily so the rest of the package works without it

        self._httpx = httpx
        self._base = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s)
        self._headers = dict(headers or {})

    def exists(self, remote_key: str, sha256: str) -> bool:
        response = self._request("HEAD", remote_key, {})
        if response.status_code == 404:
            return False
        if response.status_code >= 400:
            self._raise_for(response)
        return response.headers.get("X-Content-SHA256") == sha256

    def upload(self, local_path: Path, remote_key: str, sha256: str) -> None:
        with open(local_path, "rb") as fh:
            response = self._request(
                "PUT", remote_key, {"X-Content-SHA256": sha256}, content=fh.read()
            )
        if response.status_code >= 300:
            self._raise_for(response)

    def _request(
        self, method: str, remote_key: str, headers: Dict[str, str], content: Optional[bytes] = None
    ) -> "httpx.Response":
        httpx = self._httpx
        url = f"{self._base}/{_safe_key(remote_key)}"
        try:
            return self._client.request(
                method, url, headers={**self._headers, **headers}, content=content
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as exc:
            raise NetworkUnavailable(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise TransportError(str(exc)) from exc

    @staticmethod
    def _raise_for(response: "httpx.Response") -> None:
        code = response.status_code
        message = f"HTTP {code} from {response.request.url}"
        if code in (408, 425, 429) or code >= 500:
            raise TransportError(message)
        raise PermanentTransportError(message)


@dataclass
class UploadReport:
    uploaded: int = 0
    retry_scheduled: int = 0
    failed_permanently: int = 0
    already_present: int = 0
    offline: bool = False
    synced_sessions: List[str] = field(default_factory=list)
    blocked_sessions: List[str] = field(default_factory=list)


class UploadAgent:
    """Drain the upload queue whenever the network is available.

    Args:
        store: Encrypted store holding sessions and the queue.
        transport: Where files go.
        is_online: Connectivity probe (default: assume online; a
            :class:`NetworkUnavailable` from the transport also marks offline).
        backoff_base_s / backoff_max_s: Retry delay is ``base * 2**attempts``,
            capped, with +/- ``jitter`` randomisation.
        max_attempts: Failures before an item is parked as permanently failed.
    """

    def __init__(
        self,
        store: SessionStore,
        transport: Transport,
        *,
        is_online: Callable[[], bool] = lambda: True,
        clock_ns: Callable[[], int] = time.time_ns,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 900.0,
        jitter: float = 0.2,
        max_attempts: int = 8,
        batch_size: int = 20,
        seed: Optional[int] = None,
    ) -> None:
        self._store = store
        self._transport = transport
        self._is_online = is_online
        self._clock_ns = clock_ns
        self._base = backoff_base_s
        self._max = backoff_max_s
        self._jitter = jitter
        self._max_attempts = max_attempts
        self._batch = batch_size
        self._rng = random.Random(seed)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def run_once(self) -> UploadReport:
        """Process every item due now; returns what happened."""
        report = UploadReport()
        online = self._is_online()
        sessions = self._store.sessions_with_pending_uploads()
        self._propagate_connectivity(online)
        if not online:
            report.offline = True
            return report
        touched: Dict[str, None] = {}
        while True:
            items = self._store.due_uploads(now_ns=self._clock_ns(), limit=self._batch)
            items = [i for i in items if i.session_id not in report.blocked_sessions]
            if not items:
                break
            progressed = False
            for item in items:
                touched[item.session_id] = None
                outcome = self._process(item, report)
                if outcome == "offline":
                    report.offline = True
                    self._propagate_connectivity(False)
                    return report
                progressed = progressed or outcome in ("done", "present")
            if not progressed:
                break
        for session_id in list(touched) + [s for s in sessions if s not in touched]:
            self._settle_session(session_id, report)
        return report

    def start(self, interval_s: float = 5.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("upload agent already running")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(interval_s,), name="apgi-upload-agent", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        thread.join(timeout=10.0)
        self._thread = None

    # -- internals --------------------------------------------------------
    def _loop(self, interval_s: float) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:  # keep the agent alive; the queue is persistent
                pass
            self._stop.wait(interval_s)

    def _process(self, item: UploadItem, report: UploadReport) -> str:
        row = self._store.get_session(item.session_id)
        if row is None:
            self._store.mark_upload_failed(
                item.session_id, item.rel_path, "unknown session", permanent=True
            )
            report.failed_permanently += 1
            return "failed"
        local = Path(row.directory) / item.rel_path
        remote_key = f"{item.session_id}/{item.rel_path}"
        self._begin_upload(item.session_id)
        try:
            if not local.is_file():
                raise PermanentTransportError(f"local file missing: {item.rel_path}")
            if _sha256_hex_of_file(local) != item.sha256:
                raise PermanentTransportError(f"local file changed since sealing: {item.rel_path}")
            if self._transport.exists(remote_key, item.sha256):
                self._store.mark_upload_done(item.session_id, item.rel_path)
                report.already_present += 1
                return "present"
            self._transport.upload(local, remote_key, item.sha256)
        except NetworkUnavailable:
            return "offline"
        except PermanentTransportError as exc:
            self._store.mark_upload_failed(item.session_id, item.rel_path, str(exc), permanent=True)
            report.failed_permanently += 1
            return "failed"
        except TransportError as exc:
            self._schedule_retry(item, str(exc), report)
            return "retry"
        self._store.mark_upload_done(item.session_id, item.rel_path)
        report.uploaded += 1
        return "done"

    def _schedule_retry(self, item: UploadItem, error: str, report: UploadReport) -> None:
        attempts_after = item.attempts + 1
        if attempts_after >= self._max_attempts:
            self._store.mark_upload_failed(
                item.session_id, item.rel_path, f"gave up: {error}", permanent=True
            )
            report.failed_permanently += 1
            return
        delay_s = min(self._max, self._base * (2**item.attempts))
        delay_s *= 1.0 + self._jitter * (2.0 * self._rng.random() - 1.0)
        retry_at = self._clock_ns() + int(delay_s * 1e9)
        self._store.mark_upload_failed(item.session_id, item.rel_path, error, retry_at_ns=retry_at)
        report.retry_scheduled += 1

    def _machine(self, session_id: str) -> Optional[SessionMachine]:
        if self._store.get_session(session_id) is None:
            return None
        return SessionMachine.load(self._store, session_id, online=True)

    def _begin_upload(self, session_id: str) -> None:
        machine = self._machine(session_id)
        if machine is not None and machine.state is SessionState.FINALIZED:
            machine.fire(SessionEvent.BEGIN_UPLOAD)

    def _propagate_connectivity(self, online: bool) -> None:
        if online:
            return
        for row in self._store.list_sessions([SessionState.UPLOADING.value]):
            machine = SessionMachine.load(self._store, row.session_id, online=True)
            machine.set_online(False)

    def _settle_session(self, session_id: str, report: UploadReport) -> None:
        machine = self._machine(session_id)
        if machine is None or machine.state is not SessionState.UPLOADING:
            return
        counts = self._store.upload_counts(session_id)
        if counts["pending"] > 0:
            return
        if counts["failed"] > 0:
            machine.fire(
                SessionEvent.UPLOAD_BLOCKED, f"{counts['failed']} file(s) failed permanently"
            )
            report.blocked_sessions.append(session_id)
        else:
            machine.fire(SessionEvent.UPLOAD_COMPLETE)
            report.synced_sessions.append(session_id)
