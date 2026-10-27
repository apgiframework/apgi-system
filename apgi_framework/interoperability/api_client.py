"""Client for the apgi-api three-stage chunk upload.

Contract (the API implements it; see :mod:`apgi_framework.acquisition.chunk_ingest`):

* ``POST {base}/v1/acquisition/uploads``            manifest -> ``{upload_id, signed_url, expires_in_s, already_accepted}``
* ``PUT  <signed_url>``                              bytes straight to the private bucket
* ``POST {base}/v1/acquisition/uploads/{id}/commit`` -> ``{status, reason}``
* ``GET  {base}/v1/acquisition/uploads/status?key=`` -> ``{status, sha256}``

:class:`SignedUrlTransport` plugs this into :class:`~apgi_framework.acquisition.UploadAgent`:
the agent's idempotency and retry rules apply unchanged, and a file counts as uploaded only
when the server reports it ``accepted`` (SHA-256 recomputed server-side).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..acquisition.errors import NetworkUnavailable, PermanentTransportError, TransportError
from ..acquisition.integrity import crc32c
from ..acquisition.uploader import Transport

if TYPE_CHECKING:
    import httpx


def _raise_for(response: "httpx.Response", what: str) -> None:
    code = response.status_code
    if code < 400:
        return
    message = f"{what}: HTTP {code}"
    if code in (408, 425, 429) or code >= 500:
        raise TransportError(message)
    raise PermanentTransportError(message)


class ApiClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        client: Optional["httpx.Client"] = None,
        timeout_s: float = 30.0,
    ) -> None:
        import httpx

        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._client = client or httpx.Client(timeout=timeout_s)
        self._httpx = httpx

    def _send(self, method: str, url: str, **kw: Any) -> "httpx.Response":
        try:
            return self._client.request(method, url, **kw)
        except (
            self._httpx.ConnectError,
            self._httpx.TimeoutException,
            self._httpx.NetworkError,
        ) as exc:
            raise NetworkUnavailable(str(exc)) from exc

    def init_upload(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        r = self._send(
            "POST", f"{self._base}/v1/acquisition/uploads", json=manifest, headers=self._headers
        )
        _raise_for(r, "init upload")
        return dict(r.json())

    def put_bytes(self, signed_url: str, data: bytes) -> None:
        """Stage B: direct to the bucket; the API token is deliberately not sent."""
        r = self._send("PUT", signed_url, content=data)
        _raise_for(r, "bucket upload")

    def commit(self, upload_id: str) -> Dict[str, Any]:
        r = self._send(
            "POST", f"{self._base}/v1/acquisition/uploads/{upload_id}/commit", headers=self._headers
        )
        _raise_for(r, "commit upload")
        return dict(r.json())

    def status(self, key: str) -> Optional[Dict[str, Any]]:
        r = self._send(
            "GET",
            f"{self._base}/v1/acquisition/uploads/status",
            params={"key": key},
            headers=self._headers,
        )
        if r.status_code == 404:
            return None
        _raise_for(r, "upload status")
        return dict(r.json())


class SignedUrlTransport(Transport):
    """:class:`Transport` using :class:`ApiClient`; ``remote_key`` is ``<session_id>/<path>``."""

    def __init__(self, api: ApiClient) -> None:
        self._api = api

    def exists(self, remote_key: str, sha256: str) -> bool:
        info = self._api.status(remote_key)
        return bool(info and info.get("status") == "accepted" and info.get("sha256") == sha256)

    def upload(self, local_path: Path, remote_key: str, sha256: str) -> None:
        session_id, _, object_name = remote_key.partition("/")
        data = Path(local_path).read_bytes()
        init = self._api.init_upload(
            {
                "session_id": session_id,
                "object_name": object_name,
                "size_bytes": len(data),
                "crc32c": crc32c(data),
                "sha256": sha256,
            }
        )
        if init.get("already_accepted"):
            return
        self._api.put_bytes(init["signed_url"], data)
        result = self._api.commit(init["upload_id"])
        if result["status"] == "rejected":
            raise PermanentTransportError(f"server rejected {remote_key}: {result.get('reason')}")
        if result["status"] == "quarantined":
            raise TransportError(f"server quarantined {remote_key}: {result.get('reason')}")
