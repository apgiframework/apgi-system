"""Upload agent and transports."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, List, Optional

import pytest

from apgi_framework.acquisition import (
    HttpTransport,
    LocalDirectoryTransport,
    NetworkUnavailable,
    PermanentTransportError,
    SessionEvent,
    SessionMachine,
    SessionState,
    SessionStore,
    Transport,
    TransportError,
    UploadAgent,
)

from .helpers import KEY, FakeClock, requires_sqlcipher

httpx = pytest.importorskip("httpx")
pytestmark = requires_sqlcipher

S, E = SessionState, SessionEvent


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ScriptedTransport(Transport):
    """Transport whose behaviour per call is scripted by the test."""

    def __init__(self) -> None:
        self.uploaded: List[str] = []
        self.present: set[str] = set()
        self.failures: List[Exception] = []

    def exists(self, remote_key: str, sha256: str) -> bool:
        return remote_key in self.present

    def upload(self, local_path: Path, remote_key: str, sha256: str) -> None:
        if self.failures:
            raise self.failures.pop(0)
        self.uploaded.append(remote_key)


class Env:
    def __init__(self, tmp_path: Path) -> None:
        self.clock = FakeClock(start_ns=10**15)
        self.store = SessionStore(tmp_path / "s.db", key=KEY, clock_ns=self.clock)
        self.dir = tmp_path / "sess"
        self.dir.mkdir()
        self.store.create_session("sid", directory=str(self.dir))
        self.machine = SessionMachine("sid", store=self.store)
        for ev in (E.PREPARE, E.START, E.STOP, E.FINALIZE):
            self.machine.fire(ev)
        self.files: dict[str, bytes] = {}

    def add(self, name: str, data: bytes = b"payload") -> None:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.files[name] = data
        self.store.enqueue_file("sid", name, sha(data), len(data))

    def agent(self, transport: Transport, **kw: Any) -> UploadAgent:
        kw.setdefault("is_online", lambda: True)
        kw.setdefault("seed", 1)
        return UploadAgent(self.store, transport, clock_ns=self.clock, **kw)

    def state(self) -> str:
        row = self.store.get_session("sid")
        assert row is not None
        return row.state


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


class TestAgent:
    def test_uploads_everything_and_marks_session_synced(self, env: Env, tmp_path: Path) -> None:
        env.add("a.bin", b"aaa")
        env.add("data/b.bin", b"bbbb")
        dest = tmp_path / "remote"
        report = env.agent(LocalDirectoryTransport(dest)).run_once()
        assert report.uploaded == 2 and report.synced_sessions == ["sid"]
        assert (dest / "sid" / "a.bin").read_bytes() == b"aaa"
        assert (dest / "sid" / "data" / "b.bin").read_bytes() == b"bbbb"
        assert env.state() == "synced"
        assert env.store.upload_counts("sid") == {"pending": 0, "done": 2, "failed": 0}

    def test_files_already_present_are_not_resent(self, env: Env) -> None:
        env.add("a.bin")
        transport = ScriptedTransport()
        transport.present.add("sid/a.bin")
        report = env.agent(transport).run_once()
        assert report.already_present == 1 and transport.uploaded == []
        assert env.state() == "synced"

    def test_offline_keeps_the_queue_and_the_session(self, env: Env) -> None:
        env.add("a.bin")
        transport = ScriptedTransport()
        report = env.agent(transport, is_online=lambda: False).run_once()
        assert report.offline and report.uploaded == 0 and transport.uploaded == []
        assert env.state() == "finalized"
        assert env.store.upload_counts("sid")["pending"] == 1

    def test_comes_back_online_and_finishes(self, env: Env) -> None:
        env.add("a.bin")
        online = [False]
        transport = ScriptedTransport()
        agent = env.agent(transport, is_online=lambda: online[0])
        assert agent.run_once().offline
        online[0] = True
        assert agent.run_once().synced_sessions == ["sid"]

    def test_network_loss_mid_upload_costs_no_attempt_and_parks_the_session(self, env: Env) -> None:
        env.add("a.bin")
        env.add("b.bin")
        transport = ScriptedTransport()
        transport.failures = [NetworkUnavailable("link down")]
        report = env.agent(transport).run_once()
        assert report.offline and report.uploaded == 0
        assert env.state() == "finalized"
        assert all(i.attempts == 0 for i in env.store.due_uploads())
        assert env.agent(transport).run_once().synced_sessions == ["sid"]

    def test_transient_failure_backs_off_exponentially_then_succeeds(self, env: Env) -> None:
        env.add("a.bin")
        transport = ScriptedTransport()
        transport.failures = [TransportError("503"), TransportError("503")]
        agent = env.agent(transport, backoff_base_s=10.0, jitter=0.0)
        r1 = agent.run_once()
        assert r1.retry_scheduled == 1 and agent.run_once().uploaded == 0  # still backing off
        env.clock.advance(9.9)
        assert agent.run_once().uploaded == 0
        env.clock.advance(0.2)  # 10 s elapsed: first retry fails again, delay doubles
        assert agent.run_once().retry_scheduled == 1
        env.clock.advance(19.0)
        assert agent.run_once().uploaded == 0
        env.clock.advance(2.0)
        report = agent.run_once()
        assert report.uploaded == 1 and report.synced_sessions == ["sid"]

    def test_backoff_is_capped_and_jittered(self, env: Env) -> None:
        env.add("a.bin")
        transport = ScriptedTransport()
        transport.failures = [TransportError("x")] * 5
        agent = env.agent(transport, backoff_base_s=100.0, backoff_max_s=150.0, jitter=0.2)
        delays = []
        for _ in range(3):
            agent.run_once()
            item = env.store.get_session("sid") and next((i for i in _all_items(env)), None)
            assert item is not None
            delays.append((item.next_attempt_ns - env.clock.now) / 1e9)
            env.clock.advance(200.0)
        assert delays[0] == pytest.approx(100.0, rel=0.2)
        assert all(d <= 150.0 * 1.2 + 1e-6 for d in delays) and delays[2] > 100.0

    def test_gives_up_after_max_attempts_and_blocks_the_session(self, env: Env) -> None:
        env.add("a.bin")
        transport = ScriptedTransport()
        transport.failures = [TransportError("nope")] * 5
        agent = env.agent(transport, max_attempts=3, backoff_base_s=1.0, jitter=0.0)
        report = None
        for _ in range(3):
            report = agent.run_once()
            env.clock.advance(100.0)
        assert report is not None and report.failed_permanently == 1
        assert report.blocked_sessions == ["sid"]
        assert env.state() == "finalized"
        assert env.store.upload_counts("sid")["failed"] == 1

    def test_requeue_after_permanent_failure(self, env: Env) -> None:
        env.add("a.bin")
        transport = ScriptedTransport()
        transport.failures = [PermanentTransportError("403")]
        env.agent(transport).run_once()
        assert env.store.requeue_failed("sid") == 1
        assert env.agent(transport).run_once().synced_sessions == ["sid"]

    def test_missing_local_file_is_a_permanent_failure(self, env: Env) -> None:
        env.add("a.bin")
        (env.dir / "a.bin").unlink()
        report = env.agent(ScriptedTransport()).run_once()
        assert report.failed_permanently == 1
        item = env.store.requeue_failed("sid")
        assert item == 1

    def test_file_changed_after_sealing_is_rejected(self, env: Env) -> None:
        env.add("a.bin", b"original")
        (env.dir / "a.bin").write_bytes(b"tampered")
        transport = ScriptedTransport()
        report = env.agent(transport).run_once()
        assert report.failed_permanently == 1 and transport.uploaded == []

    def test_sealed_files_upload_while_recording_continues(self, tmp_path: Path) -> None:
        env = Env(tmp_path)
        env.store.create_session("live", directory=str(env.dir))
        live = SessionMachine("live", store=env.store)
        live.fire(E.PREPARE)
        live.fire(E.START)
        path = env.dir / "seg.bin"
        path.write_bytes(b"x")
        env.store.enqueue_file("live", "seg.bin", sha(b"x"), 1)
        transport = ScriptedTransport()
        report = env.agent(transport).run_once()
        assert report.uploaded == 1 and transport.uploaded == ["live/seg.bin"]
        assert live.state is S.RECORDING and env.store.get_session("live").state == "recording"  # type: ignore[union-attr]

    def test_background_thread_drains_the_queue(self, env: Env, tmp_path: Path) -> None:
        import time

        env.add("a.bin")
        agent = env.agent(LocalDirectoryTransport(tmp_path / "r"))
        agent.start(interval_s=0.01)
        with pytest.raises(RuntimeError):
            agent.start()
        deadline = time.monotonic() + 5
        while env.state() != "synced" and time.monotonic() < deadline:
            time.sleep(0.01)
        agent.stop()
        assert env.state() == "synced"


def _all_items(env: Env) -> List[Any]:
    return env.store.due_uploads(now_ns=env.clock.now + 10**18)


class TestLocalDirectoryTransport:
    def test_copy_is_atomic_and_verified(self, tmp_path: Path) -> None:
        src = tmp_path / "f"
        src.write_bytes(b"hello")
        transport = LocalDirectoryTransport(tmp_path / "dest")
        assert not transport.exists("s/f", sha(b"hello"))
        transport.upload(src, "s/f", sha(b"hello"))
        assert transport.exists("s/f", sha(b"hello")) and not transport.exists("s/f", "other")
        assert not list((tmp_path / "dest").rglob("*.part"))

    def test_checksum_mismatch_is_not_published(self, tmp_path: Path) -> None:
        src = tmp_path / "f"
        src.write_bytes(b"hello")
        transport = LocalDirectoryTransport(tmp_path / "dest")
        with pytest.raises(TransportError, match="checksum"):
            transport.upload(src, "s/f", "0" * 64)
        assert not (tmp_path / "dest" / "s" / "f").exists()

    @pytest.mark.parametrize("key", ["../x", "/abs", "a//b", "a/../b", "./a"])
    def test_unsafe_keys_are_rejected(self, tmp_path: Path, key: str) -> None:
        with pytest.raises(PermanentTransportError, match="unsafe"):
            LocalDirectoryTransport(tmp_path).exists(key, "x")


class TestHttpTransport:
    def make(self, handler: Any) -> HttpTransport:
        client = httpx.Client(transport=httpx.MockTransport(handler))
        return HttpTransport("https://store.example/v1/", client=client, headers={"X-Token": "t"})

    def test_put_carries_digest_and_credentials(self, tmp_path: Path) -> None:
        seen: List[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(201)

        src = tmp_path / "f"
        src.write_bytes(b"data")
        self.make(handler).upload(src, "s/f", sha(b"data"))
        req = seen[0]
        assert req.method == "PUT" and str(req.url) == "https://store.example/v1/s/f"
        assert req.headers["X-Content-SHA256"] == sha(b"data") and req.headers["X-Token"] == "t"
        assert req.content == b"data"

    def test_head_reports_existence_by_digest(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/present"):
                return httpx.Response(200, headers={"X-Content-SHA256": "abc"})
            return httpx.Response(404)

        t = self.make(handler)
        assert t.exists("s/present", "abc") and not t.exists("s/present", "different")
        assert not t.exists("s/absent", "abc")

    @pytest.mark.parametrize("status", [429, 500, 503])
    def test_retryable_statuses(self, tmp_path: Path, status: int) -> None:
        src = tmp_path / "f"
        src.write_bytes(b"x")
        t = self.make(lambda r: httpx.Response(status))
        with pytest.raises(TransportError) as info:
            t.upload(src, "s/f", "h")
        assert not isinstance(info.value, PermanentTransportError)

    @pytest.mark.parametrize("status", [400, 401, 403, 413])
    def test_permanent_statuses(self, tmp_path: Path, status: int) -> None:
        src = tmp_path / "f"
        src.write_bytes(b"x")
        with pytest.raises(PermanentTransportError):
            self.make(lambda r: httpx.Response(status)).upload(src, "s/f", "h")

    def test_connection_errors_mean_offline(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route", request=request)

        src = tmp_path / "f"
        src.write_bytes(b"x")
        with pytest.raises(NetworkUnavailable):
            self.make(handler).upload(src, "s/f", "h")

    def test_other_http_errors_are_retryable(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        src = tmp_path / "f"
        src.write_bytes(b"x")
        with pytest.raises(TransportError) as info:
            self.make(handler).upload(src, "s/f", "h")
        assert not isinstance(info.value, (NetworkUnavailable, PermanentTransportError))

    def test_agent_over_http_end_to_end(self, env: Env) -> None:
        env.add("a.bin", b"one")
        stored: dict[str, bytes] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            key = request.url.path
            if request.method == "HEAD":
                return httpx.Response(404)
            stored[key] = request.content
            return httpx.Response(200)

        report = env.agent(self.make(handler)).run_once()
        assert report.synced_sessions == ["sid"] and stored["/v1/sid/a.bin"] == b"one"
        _ = Optional  # keep typing import used
