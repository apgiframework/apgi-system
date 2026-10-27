#!/usr/bin/env python3
"""Headless acquisition smoke test, used as the entry point of the desktop-build CI gate.

Records two seconds from the simulated device into an encrypted store, uploads to a local
directory, verifies everything the way a lab would depend on it and exits non-zero on any
failure. It runs unchanged from source and from a PyInstaller bundle, which is the point:
the frozen build must contain SQLCipher, the drivers and every acquisition module.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def main() -> int:
    from apgi_framework.acquisition import (
        AcquisitionSession,
        DeviceManager,
        LocalDirectoryTransport,
        LocalRecorder,
        SessionStore,
        SimulatedDevice,
        UploadAgent,
        compute_session_qc,
        verify_write_once,
    )

    class Clock:
        now = 5 * 10**12

        def __call__(self) -> int:
            return self.now

        def advance(self, s: float) -> None:
            self.now += int(s * 1e9)

    clock = Clock()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = SessionStore(root / "apgi.db", key=bytes(range(32)))
        manager = DeviceManager(clock_ns=clock)
        manager.add_device(SimulatedDevice(clock_ns=clock, sampling_rate_hz=250.0))
        session = AcquisitionSession(
            store=store,
            sessions_dir=root / "sessions",
            manager=manager,
            subject="01",
            session_id="smoke",
            fsync=False,
            clock_ns=clock,
            sleep=clock.advance,
        )
        session.prepare()
        session.start()
        for i in range(20):
            clock.advance(0.1)
            session.pump()
            if i == 5:
                session.mark("stimulus", code=1)
        clock.advance(0.05)
        session.stop()
        report = UploadAgent(store, LocalDirectoryTransport(root / "remote"), seed=1).run_once()
        recording = LocalRecorder.read(session.directory)
        qc = compute_session_qc(recording, mapper=manager.mapper("sim-0"))
        checks = {
            "synced": report.synced_sessions == ["smoke"],
            "server_verified": store.all_server_verified("smoke"),
            "raw_intact": verify_write_once(session.directory) == [],
            "journal": len(store.list_chunk_journal("smoke")) > 0,
            "qc": qc.passed,
            "synthetic": recording.info.data_origin == "synthetic",
        }
        store.close()
    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
