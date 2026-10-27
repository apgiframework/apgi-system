"""One recording session: devices, clock sync, markers, recorder and state machine.

:class:`AcquisitionSession` is the integration point. It never touches the
network; finished files are queued in the encrypted store and a separate
:class:`~apgi_framework.acquisition.uploader.UploadAgent` sends them whenever a
connection exists.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from .bids import BidsExportResult, export_recording_to_bids
from .device import SampleChunk
from .errors import DeviceError, InvalidTransition
from .manager import DeviceManager
from .markers import EventMarker, MarkerLog
from .recorder import LocalRecorder, RecordingSummary
from .state_machine import SessionEvent, SessionMachine, SessionState
from .store import SessionStore


class AcquisitionSession:
    """Record one device into a session directory with full offline-first bookkeeping.

    Args:
        store: Encrypted store for state, markers and the upload queue.
        sessions_dir: Parent directory; the session lives in ``sessions_dir/<session_id>``.
        manager: Device manager holding the device to record.
        subject: Participant label (kept in the store; used for BIDS export).
        device_id: Device to record; optional when the manager has exactly one.
        n_sync_probes: Clock-sync probes taken in :meth:`prepare`.
        sync_interval_s: Pause between probes, so the mapper sees a time spread to fit drift.
        sleep: Injected for tests; defaults to ``time.sleep``.
        drain_timeout_s: Longest :meth:`stop` spends reading what the devices still hold.
    """

    def __init__(
        self,
        *,
        store: SessionStore,
        sessions_dir: Path,
        manager: DeviceManager,
        subject: str,
        device_id: Optional[str] = None,
        session_id: Optional[str] = None,
        segment_samples: int = 60_000,
        fsync: bool = True,
        n_sync_probes: int = 8,
        sync_interval_s: float = 0.05,
        sleep: Callable[[float], None] = time.sleep,
        drain_timeout_s: float = 2.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        online: bool = True,
    ) -> None:
        ids = manager.device_ids
        if device_id is None:
            if len(ids) != 1:
                raise DeviceError("specify device_id when the manager holds several devices")
            device_id = ids[0]
        self._manager = manager
        self._device_id = device_id
        self._store = store
        self._subject = subject
        self._n_probes = n_sync_probes
        self._sync_interval_s = sync_interval_s
        self._sleep = sleep
        self._drain_timeout_s = drain_timeout_s
        self._segment_samples = segment_samples
        self._fsync = fsync
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.directory = Path(sessions_dir) / self.session_id
        self.markers = MarkerLog(clock_ns)
        self.machine = SessionMachine.create(
            store, self.session_id, directory=str(self.directory), subject=subject, online=online
        )
        self._recorder: Optional[LocalRecorder] = None
        self._lock = threading.RLock()
        self._summary: Optional[RecordingSummary] = None

    # -- lifecycle --------------------------------------------------------
    def prepare(self) -> None:
        """Connect the device, sync its clock and open the recorder."""
        with self._lock:
            if not self.machine.can(SessionEvent.PREPARE):
                raise InvalidTransition(f"cannot prepare in state {self.machine.state.value!r}")
            try:
                errors = self._manager.connect_all()
                failed = {k: v for k, v in errors.items() if v is not None}
                if failed:
                    raise DeviceError(f"device connection failed: {failed}")
                device = self._manager.device(self._device_id)
                self._recorder = LocalRecorder(
                    self.directory,
                    device.info,
                    session_meta={"session_id": self.session_id, "subject": self._subject},
                    segment_samples=self._segment_samples,
                    fsync=self._fsync,
                    on_file_ready=self._enqueue,
                )
                self._recorder.start()
                mapper = self._manager.mapper(self._device_id)
                for i in range(self._n_probes):
                    if i and self._sync_interval_s > 0:
                        self._sleep(self._sync_interval_s)
                    probe = device.sync_probe()
                    mapper.add_probe(probe)
                    self._recorder.write_sync_probe(probe)
                    self._store.add_sync_point(self.session_id, self._device_id, probe)
            except Exception as exc:
                self._fail(f"prepare failed: {exc}")
                raise
            self.machine.fire(SessionEvent.PREPARE)

    def start(self, *, background: bool = False, poll_interval_s: float = 0.005) -> None:
        """Begin streaming and recording. With ``background`` a thread polls the device."""
        with self._lock:
            if not self.machine.can(SessionEvent.START):
                raise InvalidTransition(f"cannot start in state {self.machine.state.value!r}")
            self._manager.add_sink(self._on_chunk)
            self._manager.add_error_listener(self._on_device_error)
            errors = self._manager.start_streaming()
            failed = {k: v for k, v in errors.items() if v is not None}
            if failed:
                self._fail(f"could not start streaming: {failed}")
                raise DeviceError(f"could not start streaming: {failed}")
            self.machine.fire(SessionEvent.START)
            if background:
                self._manager.start_polling(poll_interval_s)

    def pump(self) -> int:
        """Poll the devices once (foreground mode); returns the number of chunks recorded."""
        return len(self._manager.poll())

    def drain(self) -> int:
        """Read everything currently buffered on the devices; returns chunks recorded."""
        return self._manager.drain(timeout_s=self._drain_timeout_s)

    def pause(self) -> None:
        self.machine.fire(SessionEvent.PAUSE)

    def resume(self) -> None:
        self.machine.fire(SessionEvent.RESUME)

    def mark(
        self,
        label: str,
        *,
        code: Optional[int] = None,
        hardware: bool = False,
        **kwargs: Any,
    ) -> EventMarker:
        """Record an event now. ``hardware=True`` also writes ``code`` to the device stream."""
        if self.machine.state not in (SessionState.RECORDING, SessionState.PAUSED):
            raise InvalidTransition("markers can only be recorded while recording or paused")
        if hardware:
            if code is None:
                raise ValueError("hardware markers need a numeric code")
            self._manager.device(self._device_id).emit_hardware_marker(code)
        marker = self.markers.emit(
            label, code=code, source="hardware" if hardware else "software", **kwargs
        )
        with self._lock:
            assert self._recorder is not None
            self._recorder.write_marker(marker)
        self._store.add_marker(self.session_id, marker)
        return marker

    def stop(self) -> RecordingSummary:
        """Stop streaming, drain the device, seal the recording and queue it for upload."""
        self.machine.fire(SessionEvent.STOP)  # chunks are still recorded while STOPPING
        # The polling thread records chunks under self._lock, so it must be joined
        # without holding that lock or the join would wait out its timeout.
        self._manager.stop_polling()
        self._manager.drain(timeout_s=self._drain_timeout_s)  # keep what is buffered
        self._manager.stop_streaming()
        with self._lock:
            assert self._recorder is not None
            self._summary = self._recorder.close()
            self.machine.fire(SessionEvent.FINALIZE)
            return self._summary

    def export_bids(self, bids_root: Path, *, task: str, **kwargs: Any) -> BidsExportResult:
        """Export the finished recording as a BIDS-EEG dataset."""
        if self.machine.state not in (
            SessionState.FINALIZED,
            SessionState.UPLOADING,
            SessionState.SYNCED,
        ):
            raise InvalidTransition("export needs a finalized recording")
        recording = LocalRecorder.read(self.directory)
        return export_recording_to_bids(
            recording,
            bids_root,
            subject=self._subject,
            task=task,
            mapper=self._manager.mapper(self._device_id),
            **kwargs,
        )

    @property
    def summary(self) -> Optional[RecordingSummary]:
        return self._summary

    # -- callbacks --------------------------------------------------------
    def _on_chunk(self, chunk: SampleChunk) -> None:
        if chunk.device_id != self._device_id:
            return
        with self._lock:
            recording = self.machine.state in (SessionState.RECORDING, SessionState.STOPPING)
            if not recording or self._recorder is None:
                return  # paused or not started: the index gap documents the loss
            self._recorder.write_chunk(chunk)

    def _on_device_error(self, device_id: str, exc: Exception) -> None:
        if device_id == self._device_id and self.machine.can(SessionEvent.FAIL):
            self._fail(f"device error: {exc}")

    def _enqueue(self, rel_path: str, sha256: str, size: int) -> None:
        self._store.enqueue_file(self.session_id, rel_path, sha256, size)

    def _fail(self, detail: str) -> None:
        if self.machine.can(SessionEvent.FAIL):
            self.machine.fire(SessionEvent.FAIL, detail)


def recover_session(store: SessionStore, session_id: str) -> RecordingSummary:
    """Finish a session left ``INTERRUPTED`` by a crash: repair data, queue files, finalize."""
    row = store.get_session(session_id)
    if row is None:
        raise InvalidTransition(f"unknown session {session_id!r}")
    machine = SessionMachine.load(store, session_id)
    if machine.state not in (SessionState.INTERRUPTED, SessionState.FAILED):
        raise InvalidTransition(f"session is {machine.state.value!r}, nothing to recover")

    def enqueue(rel_path: str, sha256: str, size: int) -> None:
        store.enqueue_file(session_id, rel_path, sha256, size)

    summary = LocalRecorder.recover(Path(row.directory), on_file_ready=enqueue)
    machine.fire(SessionEvent.FINALIZE, "recovered after interruption")
    return summary
