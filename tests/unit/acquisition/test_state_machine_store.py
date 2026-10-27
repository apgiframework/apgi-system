"""Offline-first state machine and the SQLCipher store."""

from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path
from typing import List, Tuple

import pytest

from apgi_framework.acquisition import (
    EventMarker,
    InvalidTransition,
    SessionEvent,
    SessionMachine,
    SessionState,
    SessionStore,
    StoreError,
    StoreKeyError,
    StoreUnavailableError,
    SyncProbe,
    recover_interrupted_sessions,
)
from apgi_framework.acquisition import store as store_module

from .helpers import KEY, requires_sqlcipher

S, E = SessionState, SessionEvent


def walk(machine: SessionMachine, *events: SessionEvent) -> None:
    for event in events:
        machine.fire(event)


class TestStateMachineInMemory:
    def test_happy_path(self) -> None:
        m = SessionMachine("s")
        walk(m, E.PREPARE, E.START, E.PAUSE, E.RESUME, E.STOP, E.FINALIZE, E.BEGIN_UPLOAD)
        assert m.state is S.UPLOADING
        m.fire(E.UPLOAD_COMPLETE)
        assert m.state is S.SYNCED
        assert [t.new for t in m.history][-1] is S.SYNCED

    @pytest.mark.parametrize(
        "state,event",
        [
            (S.CREATED, E.START),
            (S.READY, E.STOP),
            (S.RECORDING, E.BEGIN_UPLOAD),
            (S.PAUSED, E.FINALIZE),
            (S.SYNCED, E.START),
            (S.FINALIZED, E.UPLOAD_COMPLETE),
            (S.FINALIZED, E.FAIL),
        ],
    )
    def test_invalid_transitions_raise(self, state: SessionState, event: SessionEvent) -> None:
        m = SessionMachine("s", state=state)
        assert not m.can(event)
        with pytest.raises(InvalidTransition):
            m.fire(event)
        assert m.state is state

    def test_upload_requires_connectivity(self) -> None:
        m = SessionMachine("s", state=S.FINALIZED, online=False)
        assert not m.can(E.BEGIN_UPLOAD)
        with pytest.raises(InvalidTransition, match="offline"):
            m.fire(E.BEGIN_UPLOAD)
        m.set_online(True)
        m.fire(E.BEGIN_UPLOAD)
        assert m.state is S.UPLOADING

    def test_recording_never_depends_on_connectivity(self) -> None:
        m = SessionMachine("s", online=False)
        walk(m, E.PREPARE, E.START, E.STOP, E.FINALIZE)
        assert m.state is S.FINALIZED

    def test_losing_the_network_parks_an_upload(self) -> None:
        m = SessionMachine("s", state=S.UPLOADING)
        m.set_online(False)
        assert m.state is S.FINALIZED and not m.online
        m.set_online(False)  # idempotent
        m.set_online(True)
        assert m.state is S.FINALIZED  # resuming is the agent's decision

    def test_losing_the_network_does_not_touch_recording(self) -> None:
        m = SessionMachine("s", state=S.RECORDING)
        m.set_online(False)
        assert m.state is S.RECORDING

    def test_failure_paths_and_salvage(self) -> None:
        for start in (S.CREATED, S.READY, S.RECORDING, S.PAUSED, S.STOPPING):
            m = SessionMachine("s", state=start)
            m.fire(E.FAIL, "device lost")
            assert m.state is S.FAILED
            m.fire(E.FINALIZE)
            assert m.state is S.FINALIZED

    def test_crash_detection(self) -> None:
        for start in (S.RECORDING, S.PAUSED, S.STOPPING):
            m = SessionMachine("s", state=start)
            m.fire(E.CRASH_DETECTED)
            assert m.state is S.INTERRUPTED
            m.fire(E.FINALIZE)
            assert m.state is S.FINALIZED

    def test_upload_blocked_returns_to_finalized(self) -> None:
        m = SessionMachine("s", state=S.UPLOADING)
        m.fire(E.UPLOAD_BLOCKED, "1 file failed")
        assert m.state is S.FINALIZED

    def test_listeners_see_every_transition(self) -> None:
        m = SessionMachine("s")
        seen: List[Tuple[SessionState, SessionState, SessionEvent]] = []
        m.add_listener(lambda *a: seen.append(a))
        walk(m, E.PREPARE, E.START)
        assert seen == [(S.CREATED, S.READY, E.PREPARE), (S.READY, S.RECORDING, E.START)]


@requires_sqlcipher
class TestStore:
    def open(self, tmp_path: Path) -> SessionStore:
        return SessionStore(tmp_path / "apgi.db", key=KEY)

    def test_database_is_encrypted_at_rest(self, tmp_path: Path) -> None:
        with self.open(tmp_path) as store:
            store.create_session("secret-session-id", directory="/data/x", subject="alice")
        raw = (tmp_path / "apgi.db").read_bytes()
        assert not store_module.is_plaintext_sqlite(tmp_path / "apgi.db")
        for needle in (b"secret-session-id", b"alice", b"sessions", b"SQLite format"):
            assert needle not in raw

    def test_wrong_key_or_passphrase_is_rejected(self, tmp_path: Path) -> None:
        self.open(tmp_path).close()
        with pytest.raises(StoreKeyError):
            SessionStore(tmp_path / "apgi.db", key=bytes(32))
        with pytest.raises(StoreKeyError):
            SessionStore(tmp_path / "apgi.db", passphrase="guess")

    def test_passphrase_store_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "p.db"
        with SessionStore(path, passphrase='it\'s a "pass" phrase') as store:
            store.create_session("a", directory="/d")
        with SessionStore(path, passphrase='it\'s a "pass" phrase') as store:
            assert store.get_session("a") is not None

    def test_key_arguments_are_validated(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            SessionStore(tmp_path / "a.db")
        with pytest.raises(ValueError, match="exactly one"):
            SessionStore(tmp_path / "a.db", key=KEY, passphrase="x")
        with pytest.raises(ValueError, match="32 bytes"):
            SessionStore(tmp_path / "a.db", key=b"short")
        with pytest.raises(ValueError, match="empty"):
            SessionStore(tmp_path / "a.db", passphrase="")

    def test_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("APGI_STORE_KEY", raising=False)
        with pytest.raises(StoreKeyError, match="not set"):
            SessionStore.from_env(tmp_path / "e.db")
        monkeypatch.setenv("APGI_STORE_KEY", KEY.hex())
        with SessionStore.from_env(tmp_path / "e.db") as store:
            store.create_session("a", directory="/d")
        with SessionStore(tmp_path / "e.db", key=KEY) as store:  # hex form is the raw key
            assert store.get_session("a") is not None
        monkeypatch.setenv("APGI_STORE_KEY", "a plain passphrase")
        with SessionStore.from_env(tmp_path / "f.db") as store:
            store.create_session("b", directory="/d")

    def test_refuses_plain_sqlite(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(store_module, "load_sqlcipher", lambda: sqlite3)
        with pytest.raises(StoreUnavailableError, match="unencrypted"):
            SessionStore(tmp_path / "plain.db", key=KEY)

    def test_missing_library_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(name: str) -> None:
            raise ImportError(name)

        monkeypatch.setattr(importlib, "import_module", boom)
        with pytest.raises(StoreUnavailableError, match="sqlcipher3-binary"):
            store_module.load_sqlcipher()

    def test_sessions_and_state_log(self, tmp_path: Path) -> None:
        with self.open(tmp_path) as store:
            store.create_session("a", directory="/d", subject="01", meta={"k": 1})
            store.set_session_state("a", "ready", event="prepare", expected_old="created")
            row = store.get_session("a")
            assert row is not None and row.state == "ready" and row.meta == {"k": 1}
            assert [e.to_state for e in store.state_history("a")] == ["created", "ready"]
            assert [s.session_id for s in store.list_sessions(["ready"])] == ["a"]
            assert store.list_sessions(["synced"]) == []
            with pytest.raises(StoreError, match="expected"):
                store.set_session_state("a", "recording", event="start", expected_old="created")
            with pytest.raises(StoreError, match="unknown"):
                store.set_session_state("zzz", "ready", event="x")
            with pytest.raises(StoreError, match="already exists"):
                store.create_session("a", directory="/d")
            assert store.get_session("nope") is None

    def test_state_survives_reopen(self, tmp_path: Path) -> None:
        with self.open(tmp_path) as store:
            store.create_session("a", directory="/d")
            m = SessionMachine("a", store=store)
            walk(m, E.PREPARE, E.START)
        with self.open(tmp_path) as store:
            assert SessionMachine.load(store, "a").state is S.RECORDING
            with pytest.raises(StoreError):
                SessionMachine.load(store, "missing")

    def test_sync_points_and_markers_roundtrip(self, tmp_path: Path) -> None:
        with self.open(tmp_path) as store:
            store.create_session("a", directory="/d")
            store.add_sync_point("a", "dev", SyncProbe(1, 2.5, 3))
            store.add_sync_point("a", "other", SyncProbe(4, 5.5, 6))
            assert store.list_sync_points("a", "dev") == [SyncProbe(1, 2.5, 3)]
            assert len(store.list_sync_points("a")) == 2
            marker = EventMarker("m", 99, code=2, extra={"x": 1})
            store.add_marker("a", marker)
            store.add_marker("a", marker)  # idempotent by id
            assert store.list_markers("a") == [marker]

    def test_upload_queue_lifecycle(self, tmp_path: Path) -> None:
        now = [1_000]
        with SessionStore(tmp_path / "q.db", key=KEY, clock_ns=lambda: now[0]) as store:
            store.create_session("a", directory="/d")
            store.enqueue_file("a", "f1", "h1", 10)
            store.enqueue_file("a", "f2", "h2", 20)
            store.enqueue_file("a", "f1", "h1", 10)  # same digest: untouched
            assert {i.rel_path for i in store.due_uploads()} == {"f1", "f2"}
            store.mark_upload_done("a", "f1")
            store.mark_upload_failed("a", "f2", "timeout", retry_at_ns=5_000)
            assert store.due_uploads() == []  # f2 waits for its retry time
            assert [i.rel_path for i in store.due_uploads(now_ns=5_000)] == ["f2"]
            item = store.due_uploads(now_ns=5_000)[0]
            assert item.attempts == 1 and item.last_error == "timeout"
            assert store.upload_counts("a") == {"pending": 1, "done": 1, "failed": 0}
            store.mark_upload_failed("a", "f2", "gone", permanent=True)
            assert store.upload_counts("a")["failed"] == 1
            assert store.requeue_failed("a") == 1
            assert store.due_uploads()[0].attempts == 0
            assert store.sessions_with_pending_uploads() == ["a"]

    def test_changed_file_is_requeued(self, tmp_path: Path) -> None:
        with self.open(tmp_path) as store:
            store.create_session("a", directory="/d")
            store.enqueue_file("a", "f", "old", 1)
            store.mark_upload_done("a", "f")
            store.enqueue_file("a", "f", "new", 2)
            item = store.due_uploads()[0]
            assert item.sha256 == "new" and item.status == "pending"

    def test_newer_schema_is_refused(self, tmp_path: Path) -> None:
        with self.open(tmp_path) as store:
            store._conn.execute("PRAGMA user_version = 99")
        with pytest.raises(StoreError, match="newer"):
            self.open(tmp_path)

    def test_failed_transaction_rolls_back(self, tmp_path: Path) -> None:
        with self.open(tmp_path) as store:
            store.create_session("a", directory="/d")
            with pytest.raises(StoreError):
                store.set_session_state("a", "ready", event="x", expected_old="wrong")
            row = store.get_session("a")
            assert row is not None and row.state == "created"
            assert len(store.state_history("a")) == 1


@requires_sqlcipher
class TestPersistedMachine:
    def test_every_transition_is_logged_before_memory_changes(self, tmp_path: Path) -> None:
        with SessionStore(tmp_path / "m.db", key=KEY) as store:
            m = SessionMachine.create(store, "a", directory="/d", subject="01")
            walk(m, E.PREPARE, E.START, E.STOP, E.FINALIZE)
            log = store.state_history("a")
            assert [e.event for e in log] == ["create", "prepare", "start", "stop", "finalize"]

    def test_store_rejection_leaves_memory_unchanged(self, tmp_path: Path) -> None:
        with SessionStore(tmp_path / "m.db", key=KEY) as store:
            m = SessionMachine.create(store, "a", directory="/d")
            store.set_session_state("a", "ready", event="external")  # someone else moved it
            with pytest.raises(StoreError):
                m.fire(E.PREPARE)
            assert m.state is S.CREATED

    def test_restart_reconciliation(self, tmp_path: Path) -> None:
        path = tmp_path / "r.db"
        with SessionStore(path, key=KEY) as store:
            for sid, events in {
                "rec": (E.PREPARE, E.START),
                "paused": (E.PREPARE, E.START, E.PAUSE),
                "up": (E.PREPARE, E.START, E.STOP, E.FINALIZE, E.BEGIN_UPLOAD),
                "done": (E.PREPARE, E.START, E.STOP, E.FINALIZE, E.BEGIN_UPLOAD, E.UPLOAD_COMPLETE),
            }.items():
                walk(SessionMachine.create(store, sid, directory=f"/d/{sid}"), *events)
        with SessionStore(path, key=KEY) as store:  # "process restart"
            needs = recover_interrupted_sessions(store)
            states = {r.session_id: r.state for r in store.list_sessions()}
        assert sorted(needs) == ["paused", "rec"]
        assert states == {
            "rec": "interrupted",
            "paused": "interrupted",
            "up": "finalized",
            "done": "synced",
        }
