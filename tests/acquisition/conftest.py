"""Fixtures shared by the acquisition tests: a fully simulated, clock-controlled session."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import pytest

from apgi_framework.acquisition import (
    AcquisitionSession,
    DeviceManager,
    SessionStore,
    SimulatedDevice,
)
from tests.unit.acquisition.helpers import KEY, FakeClock, requires_sqlcipher

__all__ = ["KEY", "FakeClock", "requires_sqlcipher", "build_session", "record_session"]


def build_session(
    tmp_path: Path, **session_kw: Any
) -> Tuple[FakeClock, SessionStore, DeviceManager, AcquisitionSession]:
    clock = FakeClock()
    store = SessionStore(tmp_path / "apgi.db", key=KEY, clock_ns=lambda: clock.now // 1000)
    manager = DeviceManager(clock_ns=clock)
    manager.add_device(SimulatedDevice(clock_ns=clock, sampling_rate_hz=250.0))
    session = AcquisitionSession(
        store=store,
        sessions_dir=tmp_path / "sessions",
        manager=manager,
        subject="01",
        session_id="sess1",
        fsync=False,
        clock_ns=clock,
        sleep=clock.advance,
        segment_samples=500,
        **session_kw,
    )
    return clock, store, manager, session


def record_session(clock: FakeClock, session: AcquisitionSession, seconds: float = 3.0) -> Any:
    session.prepare()
    session.start()
    for i in range(int(seconds / 0.1)):
        clock.advance(0.1)
        session.pump()
        if i == 5:
            session.mark("stimulus", code=1)
        if i == 15:
            session.mark("response", code=2)
    clock.advance(0.05)
    return session.stop()


@pytest.fixture
def sim_session(tmp_path: Path):  # type: ignore[no-untyped-def]
    clock, store, manager, session = build_session(tmp_path)
    yield clock, store, manager, session
    store.close()
