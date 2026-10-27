"""Record from a simulated amplifier, export to BIDS and upload, all offline-first.

Run with::

    python examples/acquisition_quickstart.py

Needs SQLCipher (``pip install sqlcipher3-binary`` on Linux). In real use take the
store key from a secret manager or the ``APGI_STORE_KEY`` environment variable;
the throw-away key below only exists so the example is self-contained.
"""

from __future__ import annotations

import secrets
import tempfile
import time
from pathlib import Path

from apgi_framework.acquisition import (
    AcquisitionSession,
    DeviceManager,
    LocalDirectoryTransport,
    SessionStore,
    UploadAgent,
    validate_bids_eeg,
)


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="apgi-acq-"))
    key = secrets.token_bytes(32)

    manager = DeviceManager()
    manager.add("simulated", device_id="amp1", sampling_rate_hz=250.0, drift_ppm=40.0)

    with SessionStore(work / "apgi.db", key=key) as store:
        session = AcquisitionSession(
            store=store,
            sessions_dir=work / "sessions",
            manager=manager,
            subject="01",
            segment_samples=500,
        )
        session.prepare()  # connect, sync the device clock, open the recorder
        session.start(background=True)  # a thread polls the device and records
        time.sleep(0.5)
        session.mark("stimulus", code=1, duration_s=0.2)
        time.sleep(0.5)
        session.mark("trigger", code=2, hardware=True)  # also written into the data stream
        time.sleep(0.5)
        summary = session.stop()
        print(f"recorded {summary.n_samples} samples in {len(summary.segments)} segment(s)")

        result = session.export_bids(work / "bids", task="demo", power_line_hz=50.0)
        print(f"BIDS events: {result.n_events}, problems: {validate_bids_eeg(work / 'bids')}")

        # Upload whenever a connection exists; here a local folder stands in for the server.
        agent = UploadAgent(store, LocalDirectoryTransport(work / "remote"))
        report = agent.run_once()
        print(f"uploaded {report.uploaded} file(s); synced sessions: {report.synced_sessions}")

    print(f"everything is under {work}")


if __name__ == "__main__":
    main()
