"""Offline-upload gate: the network is cut mid-upload, nothing is lost, the session recovers.

Marked ``offline_upload`` so CI can run exactly this gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from apgi_framework.acquisition import (
    LocalDirectoryTransport,
    LocalRecorder,
    NetworkUnavailable,
    SessionState,
    UploadAgent,
    purge_verified_local_data,
    verify_write_once,
)
from tests.acquisition.conftest import build_session, record_session, requires_sqlcipher

pytestmark = [pytest.mark.offline_upload, pytest.mark.integration, requires_sqlcipher]


class CuttableTransport(LocalDirectoryTransport):
    """Local transport that loses the network after ``cut_after`` successful uploads."""

    def __init__(self, destination: Path, cut_after: int) -> None:
        super().__init__(destination)
        self.cut_after = cut_after
        self.uploads: List[str] = []

    def upload(self, local_path: Path, remote_key: str, sha256: str) -> None:
        if len(self.uploads) >= self.cut_after:
            raise NetworkUnavailable("link down")
        super().upload(local_path, remote_key, sha256)
        self.uploads.append(remote_key)


def test_network_cut_mid_upload_then_resume(tmp_path: Path) -> None:
    clock, store, _, session = build_session(tmp_path)
    summary = record_session(clock, session)
    assert session.machine.state is SessionState.FINALIZED
    queued = store.upload_counts("sess1")["pending"]
    assert queued >= 4

    remote = tmp_path / "remote"
    flaky = CuttableTransport(remote, cut_after=2)
    first = UploadAgent(store, flaky, seed=1).run_once()
    assert first.offline and first.uploaded == 2
    counts = store.upload_counts("sess1")
    assert counts["done"] == 2 and counts["pending"] == queued - 2 and counts["failed"] == 0
    assert session.machine.state is not SessionState.SYNCED
    # the interrupted session still has all its local data; nothing may be purged yet
    assert purge_verified_local_data(store, "sess1") == []
    assert verify_write_once(session.directory) == []

    resumed = UploadAgent(store, LocalDirectoryTransport(remote), seed=1).run_once()
    assert resumed.synced_sessions == ["sess1"] and resumed.failed_permanently == 0
    assert store.upload_counts("sess1") == {"pending": 0, "done": queued, "failed": 0}
    assert store.all_server_verified("sess1")

    # the server copy reads back identically to the local recording
    local = LocalRecorder.read(session.directory)
    assert local.n_samples == summary.n_samples
    copy = LocalRecorder.read(remote / "sess1")
    assert (copy.samples == local.samples).all()
    store.close()


def test_local_data_survives_until_every_file_is_server_verified(tmp_path: Path) -> None:
    clock, store, _, session = build_session(tmp_path)
    record_session(clock, session)
    UploadAgent(store, CuttableTransport(tmp_path / "r", cut_after=1), seed=1).run_once()
    assert purge_verified_local_data(store, "sess1") == []
    assert (session.directory / "manifest.json").exists()
    UploadAgent(store, LocalDirectoryTransport(tmp_path / "r"), seed=1).run_once()
    removed = purge_verified_local_data(store, "sess1")
    assert "manifest.json" in removed and not (session.directory / "manifest.json").exists()
    store.close()
