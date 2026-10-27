"""Three-stage chunk upload: server logic, the API client and the full agent round trip."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple

import pytest

httpx = pytest.importorskip("httpx")

from apgi_framework.acquisition import (  # noqa: E402
    ChunkIngestService,
    ChunkManifest,
    ChunkStatus,
    LocalObjectStore,
    LocalRecorder,
    UploadAgent,
    UploadForbidden,
    crc32c,
)
from apgi_framework.acquisition.chunk_ingest import UnknownUpload  # noqa: E402
from apgi_framework.acquisition.errors import PermanentTransportError, TransportError  # noqa: E402
from apgi_framework.interoperability import ApiClient, SignedUrlTransport  # noqa: E402
from tests.acquisition.conftest import (  # noqa: E402
    build_session,
    record_session,
    requires_sqlcipher,
)


def manifest(data: bytes, seq=None, session="s1", name="data/c.bin", **over) -> ChunkManifest:  # type: ignore[no-untyped-def]
    fields = dict(
        session_id=session,
        object_name=name,
        size_bytes=len(data),
        crc32c=crc32c(data),
        sha256=hashlib.sha256(data).hexdigest(),
        sequence=seq,
    )
    fields.update(over)
    return ChunkManifest(**fields)  # type: ignore[arg-type]


@pytest.fixture
def svc(tmp_path: Path) -> Tuple[ChunkIngestService, LocalObjectStore]:
    store = LocalObjectStore(tmp_path / "bucket")
    return (
        ChunkIngestService(store, authorize=lambda p, s: p == "alice" and s.startswith("s")),
        store,
    )


def push(svc, store, data: bytes, seq=None, **over):  # type: ignore[no-untyped-def]
    m = manifest(data, seq, **over)
    init = svc.init_upload(m, "alice")
    store.put_via_url(init.signed_url, data)
    return init, svc.commit(init.upload_id, "alice")


class TestStages:
    def test_good_chunk_is_pending_then_accepted_after_sha_recheck(self, svc) -> None:  # type: ignore[no-untyped-def]
        service, store = svc
        init, rec = push(service, store, b"hello" * 100, seq=0)
        assert rec.status is ChunkStatus.PENDING and init.signed_url.startswith("local://")
        assert service.run_verification_jobs() == {init.upload_id: ChunkStatus.ACCEPTED}
        assert service.status(init.upload_id).status is ChunkStatus.ACCEPTED

    def test_missing_object_is_rejected(self, svc) -> None:  # type: ignore[no-untyped-def]
        service, _ = svc
        init = service.init_upload(manifest(b"abc", 0), "alice")
        rec = service.commit(init.upload_id, "alice")
        assert rec.status is ChunkStatus.REJECTED and "does not exist" in rec.reason

    def test_wrong_size_is_rejected(self, svc) -> None:  # type: ignore[no-untyped-def]
        service, store = svc
        init = service.init_upload(manifest(b"abcdef", 0), "alice")
        store.put_via_url(init.signed_url, b"abc")
        rec = service.commit(init.upload_id, "alice")
        assert rec.status is ChunkStatus.REJECTED and "size" in rec.reason

    def test_crc_mismatch_is_quarantined_not_accepted(self, svc) -> None:  # type: ignore[no-untyped-def]
        service, store = svc
        _, rec = push(service, store, b"payload!", seq=0, crc32c=123)
        assert rec.status is ChunkStatus.QUARANTINED and "CRC32C" in rec.reason
        assert service.run_verification_jobs() == {}

    def test_sha_mismatch_with_matching_crc_is_rejected_by_the_job(self, svc) -> None:  # type: ignore[no-untyped-def]
        service, store = svc
        init, rec = push(service, store, b"payload!", seq=0, sha256="0" * 64)
        assert rec.status is ChunkStatus.PENDING
        service.run_verification_jobs()
        done = service.status(init.upload_id)
        assert done.status is ChunkStatus.REJECTED and "SHA-256" in done.reason

    def test_duplicate_sequence_is_rejected(self, svc) -> None:  # type: ignore[no-untyped-def]
        service, store = svc
        push(service, store, b"one", seq=0, name="data/a.bin")
        _, rec = push(service, store, b"two", seq=0, name="data/b.bin")
        assert rec.status is ChunkStatus.REJECTED and "duplicate sequence" in rec.reason

    def test_gap_is_quarantined_until_the_missing_chunk_is_accepted(self, svc) -> None:  # type: ignore[no-untyped-def]
        service, store = svc
        i2, r2 = push(service, store, b"two", seq=1, name="data/b.bin")
        assert r2.status is ChunkStatus.QUARANTINED and "gap" in r2.reason
        i1, _ = push(service, store, b"one", seq=0, name="data/a.bin")
        service.run_verification_jobs()
        assert service.status(i1.upload_id).status is ChunkStatus.ACCEPTED
        assert service.status(i2.upload_id).status is ChunkStatus.ACCEPTED  # released by the job

    def test_commit_is_idempotent(self, svc) -> None:  # type: ignore[no-untyped-def]
        service, store = svc
        init, first = push(service, store, b"x" * 10, seq=0)
        assert service.commit(init.upload_id, "alice") is first
        assert len(service._queue) == 1

    def test_files_without_sequence_skip_the_sequence_checks(self, svc) -> None:  # type: ignore[no-untyped-def]
        service, store = svc
        _, rec = push(service, store, b"manifest", seq=None, name="manifest.json")
        assert rec.status is ChunkStatus.PENDING


class TestAuthorisation:
    def test_unauthorised_principal_changes_nothing(self, svc) -> None:  # type: ignore[no-untyped-def]
        service, _ = svc
        with pytest.raises(UploadForbidden):
            service.init_upload(manifest(b"x"), "mallory")
        with pytest.raises(UploadForbidden):
            service.init_upload(
                manifest(b"x", session="other"), "alice"
            )  # not authorised for 'other'
        assert not service._uploads

    def test_only_the_initialising_principal_may_commit(self, tmp_path: Path) -> None:
        store = LocalObjectStore(tmp_path / "b")
        service = ChunkIngestService(store, authorize=lambda p, s: True)
        init = service.init_upload(manifest(b"x"), "alice")
        with pytest.raises(UploadForbidden, match="different principal"):
            service.commit(init.upload_id, "bob")
        with pytest.raises(UnknownUpload):
            service.commit("nope", "alice")

    def test_forged_signed_url_is_refused_by_the_bucket(self, svc) -> None:  # type: ignore[no-untyped-def]
        service, store = svc
        init = service.init_upload(manifest(b"abc"), "alice")
        with pytest.raises(PermissionError):
            store.put_via_url(init.signed_url.replace("data/c.bin", "data/other.bin"), b"abc")
        with pytest.raises(PermissionError):
            store.put_via_url("local://forged/data/c.bin", b"abc")

    def test_resubmitting_an_accepted_object_is_a_no_op_but_changed_content_is_refused(self, svc) -> None:  # type: ignore[no-untyped-def]
        service, store = svc
        push(service, store, b"same", seq=0)
        service.run_verification_jobs()
        assert service.init_upload(manifest(b"same", 0), "alice").already_accepted
        with pytest.raises(ValueError, match="different content"):
            service.init_upload(manifest(b"different", 0), "alice")


def make_api(
    service: ChunkIngestService, store: LocalObjectStore, *, user: str = "alice"
) -> ApiClient:
    """An ApiClient wired by httpx.MockTransport to the service; the bucket 'PUT' goes to the store."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "PUT":  # signed URL: straight to the bucket, no API credentials
            assert "authorization" not in request.headers
            try:
                store.put_via_url(
                    str(request.url).replace("https://bucket.invalid/", "local://"), request.content
                )
            except PermissionError:
                return httpx.Response(403)
            return httpx.Response(200)
        assert request.headers["authorization"] == f"Bearer {user}"
        try:
            if path == "/v1/acquisition/uploads" and request.method == "POST":
                body = json.loads(request.content)
                init = service.init_upload(ChunkManifest(**body), user)
                url = init.signed_url.replace("local://", "https://bucket.invalid/")
                return httpx.Response(
                    200,
                    json={
                        "upload_id": init.upload_id,
                        "signed_url": url,
                        "expires_in_s": init.expires_in_s,
                        "already_accepted": init.already_accepted,
                    },
                )
            if path.endswith("/commit"):
                rec = service.commit(path.split("/")[-2], user)
                return httpx.Response(200, json={"status": rec.status.value, "reason": rec.reason})
            if path.endswith("/status"):
                rec = service.status_for_key(request.url.params["key"], user)
                if rec is None:
                    return httpx.Response(404)
                return httpx.Response(
                    200, json={"status": rec.status.value, "sha256": rec.manifest.sha256}
                )
        except UploadForbidden:
            return httpx.Response(403)
        return httpx.Response(404)

    return ApiClient(
        "https://api.invalid", user, client=httpx.Client(transport=httpx.MockTransport(handler))
    )


class TestClientAndTransport:
    def test_file_round_trip_and_server_verification(self, svc, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        service, store = svc
        transport = SignedUrlTransport(make_api(service, store))
        f = tmp_path / "f.bin"
        f.write_bytes(b"data" * 50)
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        assert not transport.exists("s1/f.bin", sha)
        transport.upload(f, "s1/f.bin", sha)
        assert not transport.exists("s1/f.bin", sha)  # committed but SHA job not run yet
        service.run_verification_jobs()
        assert transport.exists("s1/f.bin", sha) and not transport.exists("s1/f.bin", "0" * 64)
        transport.upload(f, "s1/f.bin", sha)  # idempotent: already accepted

    def test_bytes_go_to_the_bucket_not_through_the_api(self, svc, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        service, store = svc
        seen = []
        api = make_api(service, store)
        original = api._client._transport.handle_request  # type: ignore[attr-defined]

        def spy(request):  # type: ignore[no-untyped-def]
            seen.append((request.method, request.url.host, len(request.content)))
            return original(request)

        api._client._transport.handle_request = spy  # type: ignore[attr-defined]
        f = tmp_path / "big.bin"
        f.write_bytes(b"z" * 100_000)
        SignedUrlTransport(api).upload(f, "s1/big.bin", hashlib.sha256(f.read_bytes()).hexdigest())
        big = [h for _, h, n in seen if n >= 100_000]
        assert big == ["bucket.invalid"]
        assert all(n < 1000 for _, h, n in seen if h == "api.invalid")

    def test_server_side_rejection_is_permanent_and_quarantine_is_retryable(self, svc, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        service, store = svc
        api = make_api(service, store)
        transport = SignedUrlTransport(api)
        f = tmp_path / "f.bin"
        f.write_bytes(b"abc")
        # Lie about the CRC: the server quarantines it, which the agent treats as retryable.
        orig = api.init_upload
        api.init_upload = lambda m: orig({**m, "crc32c": 5})  # type: ignore[method-assign]
        with pytest.raises(TransportError) as info:
            transport.upload(f, "s1/f.bin", hashlib.sha256(b"abc").hexdigest())
        assert not isinstance(info.value, PermanentTransportError)
        # An unauthorised caller is a permanent failure.
        denied = SignedUrlTransport(make_api(service, store, user="mallory"))
        with pytest.raises(PermanentTransportError):
            denied.upload(f, "s1/g.bin", hashlib.sha256(b"abc").hexdigest())

    def test_network_down_is_reported_as_offline(self, tmp_path: Path) -> None:
        from apgi_framework.acquisition import NetworkUnavailable

        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        api = ApiClient(
            "https://api.invalid", "t", client=httpx.Client(transport=httpx.MockTransport(boom))
        )
        f = tmp_path / "f.bin"
        f.write_bytes(b"x")
        with pytest.raises(NetworkUnavailable):
            SignedUrlTransport(api).upload(f, "s/f.bin", "0" * 64)


@requires_sqlcipher
def test_upload_agent_end_to_end_through_the_signed_url_flow(tmp_path: Path) -> None:
    from apgi_framework.acquisition import SessionMilestone

    clock, store, _, session = build_session(tmp_path / "local")
    record_session(clock, session)
    bucket = LocalObjectStore(tmp_path / "bucket")
    service = ChunkIngestService(bucket, authorize=lambda p, s: True)
    transport = SignedUrlTransport(make_api(service, bucket, user="lab"))
    agent = UploadAgent(store, transport, seed=1)

    first = agent.run_once()  # uploaded, but the server has not SHA-verified anything yet
    assert first.uploaded == 0 and first.retry_scheduled > 0
    service.run_verification_jobs()
    agent2 = UploadAgent(store, transport, seed=1, clock_ns=lambda: 4 * 10**18)  # past any backoff
    report = agent2.run_once()
    assert report.synced_sessions == ["sess1"], report
    assert store.all_server_verified("sess1")
    assert SessionMilestone.SERVER_VERIFIED.value in store.milestones("sess1")
    local = LocalRecorder.read(session.directory)
    remote = LocalRecorder.read(bucket.root / "sess1")
    assert (remote.samples == local.samples).all()
    store.close()
