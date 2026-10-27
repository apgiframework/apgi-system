"""Offline-first session state machine.

Recording never depends on the network. Connectivity is tracked separately from
the session lifecycle and only gates the upload phase:

    CREATED -> READY -> RECORDING <-> PAUSED -> STOPPING -> FINALIZED
    FINALIZED -> UPLOADING -> SYNCED
    UPLOADING -> FINALIZED            (connection lost or items failed permanently)
    RECORDING/PAUSED/STOPPING -> INTERRUPTED   (process died; found at restart)
    INTERRUPTED/FAILED -> FINALIZED   (after the recorder repaired the data)

Every transition is written to the encrypted store *before* the in-memory state
changes, so a crash can never leave memory ahead of disk.
"""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .errors import InvalidTransition, StoreError
from .store import SessionStore


class SessionState(str, enum.Enum):
    CREATED = "created"
    READY = "ready"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"
    FINALIZED = "finalized"
    UPLOADING = "uploading"
    SYNCED = "synced"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class SessionEvent(str, enum.Enum):
    PREPARE = "prepare"
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    FINALIZE = "finalize"
    BEGIN_UPLOAD = "begin_upload"
    CONNECTION_LOST = "connection_lost"
    UPLOAD_BLOCKED = "upload_blocked"
    UPLOAD_COMPLETE = "upload_complete"
    CRASH_DETECTED = "crash_detected"
    FAIL = "fail"


S, E = SessionState, SessionEvent

_RECORDING_PHASE = (S.RECORDING, S.PAUSED, S.STOPPING)

_TABLE: Dict[Tuple[SessionState, SessionEvent], SessionState] = {
    (S.CREATED, E.PREPARE): S.READY,
    (S.READY, E.START): S.RECORDING,
    (S.RECORDING, E.PAUSE): S.PAUSED,
    (S.PAUSED, E.RESUME): S.RECORDING,
    (S.RECORDING, E.STOP): S.STOPPING,
    (S.PAUSED, E.STOP): S.STOPPING,
    (S.STOPPING, E.FINALIZE): S.FINALIZED,
    (S.INTERRUPTED, E.FINALIZE): S.FINALIZED,
    (S.FAILED, E.FINALIZE): S.FINALIZED,
    (S.FINALIZED, E.BEGIN_UPLOAD): S.UPLOADING,
    (S.UPLOADING, E.CONNECTION_LOST): S.FINALIZED,
    (S.UPLOADING, E.UPLOAD_BLOCKED): S.FINALIZED,
    (S.UPLOADING, E.UPLOAD_COMPLETE): S.SYNCED,
    **{(state, E.CRASH_DETECTED): S.INTERRUPTED for state in _RECORDING_PHASE},
    **{
        (state, E.FAIL): S.FAILED
        for state in (S.CREATED, S.READY, S.RECORDING, S.PAUSED, S.STOPPING)
    },
}

TransitionListener = Callable[[SessionState, SessionState, SessionEvent], None]


@dataclass(frozen=True)
class Transition:
    old: SessionState
    new: SessionState
    event: SessionEvent
    detail: str = ""


class SessionMachine:
    """State holder for one session, optionally persisted in a :class:`SessionStore`."""

    def __init__(
        self,
        session_id: str,
        *,
        store: Optional[SessionStore] = None,
        state: SessionState = SessionState.CREATED,
        online: bool = True,
    ) -> None:
        self.session_id = session_id
        self._store = store
        self._state = state
        self._online = online
        self._lock = threading.RLock()
        self._listeners: List[TransitionListener] = []
        self._history: List[Transition] = []

    # -- construction -----------------------------------------------------
    @classmethod
    def create(
        cls,
        store: SessionStore,
        session_id: str,
        *,
        directory: str,
        subject: Optional[str] = None,
        online: bool = True,
    ) -> "SessionMachine":
        store.create_session(session_id, directory=directory, subject=subject)
        return cls(session_id, store=store, online=online)

    @classmethod
    def load(cls, store: SessionStore, session_id: str, *, online: bool = True) -> "SessionMachine":
        row = store.get_session(session_id)
        if row is None:
            raise StoreError(f"unknown session {session_id!r}")
        return cls(session_id, store=store, state=SessionState(row.state), online=online)

    # -- queries ----------------------------------------------------------
    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    @property
    def online(self) -> bool:
        with self._lock:
            return self._online

    @property
    def history(self) -> List[Transition]:
        with self._lock:
            return list(self._history)

    def can(self, event: SessionEvent) -> bool:
        with self._lock:
            return self._check(event) is None

    def add_listener(self, listener: TransitionListener) -> None:
        with self._lock:
            self._listeners.append(listener)

    # -- actions ----------------------------------------------------------
    def fire(self, event: SessionEvent, detail: str = "") -> SessionState:
        """Apply ``event`` or raise :class:`InvalidTransition`."""
        with self._lock:
            problem = self._check(event)
            if problem is not None:
                raise InvalidTransition(problem)
            return self._apply(event, _TABLE[(self._state, event)], detail)

    def set_online(self, online: bool) -> None:
        """Record connectivity. Losing it mid-upload parks the session as FINALIZED."""
        with self._lock:
            changed = online != self._online
            self._online = online
            if changed and not online and self._state is S.UPLOADING:
                self._apply(E.CONNECTION_LOST, S.FINALIZED, "network unavailable")

    # -- internals --------------------------------------------------------
    def _check(self, event: SessionEvent) -> Optional[str]:
        target = _TABLE.get((self._state, event))
        if target is None:
            return f"event {event.value!r} is not allowed in state {self._state.value!r}"
        if event is E.BEGIN_UPLOAD and not self._online:
            return "cannot begin upload while offline"
        return None

    def _apply(self, event: SessionEvent, new: SessionState, detail: str) -> SessionState:
        old = self._state
        if self._store is not None:
            self._store.set_session_state(
                self.session_id, new.value, event=event.value, detail=detail, expected_old=old.value
            )
        self._state = new
        self._history.append(Transition(old, new, event, detail))
        for listener in list(self._listeners):
            listener(old, new, event)
        return new


def recover_interrupted_sessions(store: SessionStore) -> List[str]:
    """Run at start-up: reconcile sessions left mid-flight by a crash or power loss.

    Sessions that were recording become ``INTERRUPTED`` (their data on disk must
    be repaired with ``LocalRecorder.recover`` and then finalized). Sessions that
    were uploading go back to ``FINALIZED``; their upload queue is persisted, so
    nothing is lost. Returns the ids of sessions needing recorder recovery.
    """
    needs_recovery: List[str] = []
    for row in store.list_sessions():
        state = SessionState(row.state)
        machine = SessionMachine(row.session_id, store=store, state=state)
        if state in _RECORDING_PHASE:
            machine.fire(E.CRASH_DETECTED, "process stopped while recording")
            needs_recovery.append(row.session_id)
        elif state is S.INTERRUPTED:
            needs_recovery.append(row.session_id)
        elif state is S.UPLOADING:
            machine.set_online(True)
            machine._apply(E.CONNECTION_LOST, S.FINALIZED, "restart during upload")
    return needs_recovery
