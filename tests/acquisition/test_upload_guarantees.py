"""server_verified, local-data retention and upload-failure reporting."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from apgi_framework.acquisition import (
    LocalDirectoryTransport,
    TransportError,
    UploadAgent,
    purge_verified_local_data,
)
from apgi_framework.acquisition.uploader import Transport
from tests.unit.acquisition.helpers import requires_sqlcipher
from tests.unit.acquisition.test_uploader import env  # noqa: F401  (fixture)
from tests.unit.acquisition.test_uploader import Env, ScriptedTransport, sha  # noqa: F401

pytestmark = requires_sqlcipher


class LyingTransport(Transport):
    """Accepts uploads but never confirms them on read-back."""

    def exists(self, remote_key: str, sha256: str) -> bool:
        return False

    def upload(self, local_path: Path, remote_key: str, sha256: str) -> None:
        return None


class TestServerVerified:
    def test_upload_is_verified_only_after_server_read_back(self, env: Env, tmp_path: Path) -> None:
        env.add("a.bin", b"aaa")
        report = env.agent(LocalDirectoryTransport(tmp_path / "remote")).run_once()
        assert report.uploaded == 1
        assert env.store.verified_paths("sid") == ["a.bin"]
        assert env.store.all_server_verified("sid")

    def test_unconfirmed_upload_is_not_done_and_is_retried(self, env: Env) -> None:
        env.add("a.bin")
        report = env.agent(LyingTransport()).run_once()
        assert report.uploaded == 0 and report.retry_scheduled == 1
        assert env.store.verified_paths("sid") == []
        assert env.store.upload_counts("sid")["done"] == 0
        assert env.state() != "synced"

    def test_already_present_files_count_as_verified(self, env: Env) -> None:
        env.add("a.bin")
        t = ScriptedTransport()
        t.present.add("sid/a.bin")
        env.agent(t).run_once()
        assert env.store.all_server_verified("sid")


class TestLocalDataRetention:
    def test_nothing_is_deleted_until_every_file_is_verified(self, env: Env) -> None:
        env.add("a.bin")
        env.add("b.bin")
        assert purge_verified_local_data(env.store, "sid") == []  # nothing uploaded yet
        assert (env.dir / "a.bin").exists() and (env.dir / "b.bin").exists()

    def test_partial_verification_keeps_everything(self, env: Env) -> None:
        env.add("a.bin")
        env.add("b.bin")
        env.store.mark_upload_done("sid", "a.bin", server_verified=True)
        env.store.mark_upload_done("sid", "b.bin", server_verified=False)
        assert purge_verified_local_data(env.store, "sid") == []
        assert (env.dir / "a.bin").exists()

    def test_verified_session_may_be_purged(self, env: Env, tmp_path: Path) -> None:
        env.add("a.bin", b"aaa")
        env.agent(LocalDirectoryTransport(tmp_path / "remote")).run_once()
        assert (env.dir / "a.bin").exists()  # the agent itself never deletes
        assert purge_verified_local_data(env.store, "sid") == ["a.bin"]
        assert not (env.dir / "a.bin").exists()
        assert (tmp_path / "remote" / "sid" / "a.bin").read_bytes() == b"aaa"


class TestFailuresAreReported:
    def test_unexpected_error_in_the_loop_is_logged_and_counted(
        self, env: Env, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent = env.agent(ScriptedTransport())

        def boom() -> None:
            agent._stop.set()  # one pass only
            raise RuntimeError("disk on fire")

        agent.run_once = boom  # type: ignore[method-assign]
        with caplog.at_level(logging.ERROR):
            agent._loop(0.0)
        assert agent.loop_errors == 1
        assert agent.last_loop_error == "RuntimeError: disk on fire"
        assert "disk on fire" in caplog.text

    def test_per_file_failure_keeps_its_message(self, env: Env) -> None:
        env.add("a.bin")
        t = ScriptedTransport()
        t.failures = [TransportError("503 from server")]
        env.agent(t).run_once()
        item = env.store.due_uploads(now_ns=10**18)[0]
        assert "503" in item.last_error


def test_agent_constructor_defaults_have_no_error_state(env: Env) -> None:
    agent = UploadAgent(env.store, ScriptedTransport())
    assert agent.loop_errors == 0 and agent.last_loop_error is None
