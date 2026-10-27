"""Encrypted chunk journal, session milestones and the retention policy."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from apgi_framework.acquisition import (
    LocalDirectoryTransport,
    RetentionPolicy,
)
from apgi_framework.acquisition import SessionMilestone as M
from apgi_framework.acquisition import SessionStore, UploadAgent, apply_retention_policy
from apgi_framework.acquisition.milestones import MILESTONE_ORDER, missing_predecessors
from tests.acquisition.conftest import KEY, build_session, record_session, requires_sqlcipher

pytestmark = requires_sqlcipher


def test_chunks_are_journaled_in_the_encrypted_store(tmp_path: Path) -> None:
    clock, store, _, session = build_session(tmp_path)
    summary = record_session(clock, session)
    journal = store.list_chunk_journal("sess1")
    assert len(journal) == summary.n_chunks
    assert sum(j["n_samples"] for j in journal) == summary.n_samples
    assert [j["sequence"] for j in journal] == sorted(j["sequence"] for j in journal)
    assert all(len(j["sha256"]) == 64 and j["crc32c"] >= 0 for j in journal)
    store.close()


def test_journal_matches_the_recorder_index_digests(tmp_path: Path) -> None:
    import json

    clock, store, _, session = build_session(tmp_path)
    summary = record_session(clock, session)
    on_disk = {}
    for seg in summary.segments:
        for line in (session.directory / seg.index_file).read_text().splitlines():
            e = json.loads(line)
            on_disk[e["seq"]] = (e["crc32c"], e["sha256"])
    for j in store.list_chunk_journal("sess1"):
        assert (j["crc32c"], j["sha256"]) == on_disk[j["sequence"]]
    store.close()


def test_the_database_file_is_not_plaintext_sqlite(tmp_path: Path) -> None:
    clock, store, _, session = build_session(tmp_path)
    record_session(clock, session)
    store.close()
    with pytest.raises(sqlite3.DatabaseError):
        sqlite3.connect(tmp_path / "apgi.db").execute("select * from chunk_journal").fetchall()


def test_journal_is_idempotent(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db", key=KEY)
    store.create_session("a", directory=str(tmp_path))
    for _ in range(3):
        store.journal_chunk("a", "d", 0, 0, 10, 5, 7, "ab" * 32)
    assert len(store.list_chunk_journal("a")) == 1
    store.close()


def test_schema_v2_store_is_migrated(tmp_path: Path) -> None:
    from apgi_framework.acquisition.store import SCHEMA_VERSION, load_sqlcipher

    path = tmp_path / "old.db"
    store = SessionStore(path, key=KEY)
    store.create_session("a", directory=str(tmp_path))
    store._conn.execute("DROP TABLE chunk_journal")
    store._conn.execute("DROP TABLE milestones")
    store._conn.execute("PRAGMA user_version = 2")
    store._conn.commit()
    store.close()
    reopened = SessionStore(path, key=KEY)
    reopened.journal_chunk("a", "d", 0, 0, 1, 1, 1, "00")
    assert reopened.record_milestone("a", "session_created")
    assert int(reopened._conn.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
    reopened.close()
    _ = load_sqlcipher


class TestMilestones:
    def test_a_full_session_reaches_the_first_seven(self, tmp_path: Path) -> None:
        clock, store, _, session = build_session(tmp_path)
        record_session(clock, session)
        reached = store.milestones("sess1")
        for m in MILESTONE_ORDER[:7]:
            assert m.value in reached, m
        assert M.SERVER_VERIFIED.value not in reached
        assert M.LOCAL_RETENTION_POLICY_APPLIED.value not in reached
        store.close()

    def test_milestones_are_recorded_once_with_the_first_time(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "s.db", key=KEY)
        store.create_session("a", directory=str(tmp_path))
        assert store.record_milestone("a", "x", "first") is True
        first = store.milestones("a")["x"]
        assert store.record_milestone("a", "x", "again") is False
        assert store.milestones("a")["x"] == first
        store.close()

    def test_order_and_predecessors(self) -> None:
        assert [m.value for m in MILESTONE_ORDER][0] == "session_created"
        assert len(MILESTONE_ORDER) == 9
        assert missing_predecessors([M.SESSION_CREATED], M.RECORDING_STARTED) == [
            M.CONFIGURATION_VERIFIED
        ]

    def test_server_verified_is_reached_only_after_a_confirmed_upload(self, tmp_path: Path) -> None:
        clock, store, _, session = build_session(tmp_path)
        record_session(clock, session)
        UploadAgent(store, LocalDirectoryTransport(tmp_path / "r"), seed=1).run_once()
        assert M.SERVER_VERIFIED.value in store.milestones("sess1")
        store.close()

    def test_interrupted_upload_never_reaches_server_verified(self, tmp_path: Path) -> None:
        from tests.integration.test_acquisition_offline import CuttableTransport

        clock, store, _, session = build_session(tmp_path)
        record_session(clock, session)
        UploadAgent(store, CuttableTransport(tmp_path / "r", cut_after=1), seed=1).run_once()
        assert M.SERVER_VERIFIED.value not in store.milestones("sess1")
        assert (session.directory / "manifest.json").exists()  # local data untouched
        store.close()


class TestRetention:
    def verified_session(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        clock, store, _, session = build_session(tmp_path)
        record_session(clock, session)
        UploadAgent(store, LocalDirectoryTransport(tmp_path / "r"), seed=1).run_once()
        return store, session

    def test_default_policy_keeps_everything_but_records_the_milestone(
        self, tmp_path: Path
    ) -> None:
        store, session = self.verified_session(tmp_path)
        assert apply_retention_policy(store, "sess1", RetentionPolicy()) == []
        assert (session.directory / "manifest.json").exists()
        assert M.LOCAL_RETENTION_POLICY_APPLIED.value in store.milestones("sess1")
        store.close()

    def test_purge_policy_deletes_only_after_verification(self, tmp_path: Path) -> None:
        store, session = self.verified_session(tmp_path)
        removed = apply_retention_policy(store, "sess1", RetentionPolicy(purge_after_verified=True))
        assert "manifest.json" in removed and not (session.directory / "manifest.json").exists()
        store.close()

    def test_keep_days_delays_the_purge(self, tmp_path: Path) -> None:
        store, session = self.verified_session(tmp_path)
        policy = RetentionPolicy(purge_after_verified=True, keep_days=7)
        verified_at = store.milestones("sess1")[M.SERVER_VERIFIED.value]  # store clock, ns
        assert (
            apply_retention_policy(store, "sess1", policy, now_ns=verified_at + 6 * 86_400 * 10**9)
            == []
        )
        assert apply_retention_policy(
            store, "sess1", policy, now_ns=verified_at + 8 * 86_400 * 10**9
        )
        store.close()

    def test_unverified_session_is_never_purged(self, tmp_path: Path) -> None:
        clock, store, _, session = build_session(tmp_path)
        record_session(clock, session)
        assert (
            apply_retention_policy(store, "sess1", RetentionPolicy(purge_after_verified=True)) == []
        )
        assert (session.directory / "manifest.json").exists()
        assert M.LOCAL_RETENTION_POLICY_APPLIED.value not in store.milestones("sess1")
        store.close()
