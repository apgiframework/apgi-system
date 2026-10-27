"""Events carry a collision-safe id, a per-session seq_no and monotonic + UTC stamps."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from apgi_framework.acquisition import EventMarker, MarkerLog
from apgi_framework.core.event_bus import Event, EventBus
from tests.acquisition.conftest import build_session, record_session, requires_sqlcipher


class TestMarkerIdentity:
    def test_ids_are_unique_uuids_and_seq_is_gapless_from_one(self) -> None:
        log = MarkerLog(clock_ns=lambda: 1, utc_ns=lambda: 2)  # identical stamps on purpose
        markers = [log.emit("e") for _ in range(500)]
        assert len({m.event_id for m in markers}) == 500
        assert [m.seq_no for m in markers] == list(range(1, 501))
        assert all(len(m.event_id) == 32 and int(m.event_id, 16) >= 0 for m in markers)

    def test_each_event_has_monotonic_and_utc_stamps(self) -> None:
        log = MarkerLog(clock_ns=lambda: 111, utc_ns=lambda: 1_700_000_000_000_000_000)
        m = log.emit("go")
        assert m.onset_host_ns == 111
        assert m.onset_utc_ns == 1_700_000_000_000_000_000

    def test_round_trip_keeps_identity(self) -> None:
        m = MarkerLog(utc_ns=lambda: 5).emit("go", code=3)
        again = EventMarker.from_dict(json.loads(json.dumps(m.to_dict())))
        assert (again.event_id, again.seq_no, again.onset_utc_ns) == (m.event_id, 1, 5)

    def test_markers_written_before_this_change_still_load(self) -> None:
        old = {"marker_id": "abc", "label": "x", "onset_host_ns": 9}
        m = EventMarker.from_dict(old)
        assert m.seq_no == 0 and m.onset_utc_ns is None and m.event_id == "abc"

    def test_concurrent_emitters_never_share_a_seq_no(self) -> None:
        import threading

        log = MarkerLog()
        threads = [
            threading.Thread(target=lambda: [log.emit("e") for _ in range(200)]) for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        seqs = [m.seq_no for m in log.snapshot()]
        assert sorted(seqs) == list(range(1, 1601))


@requires_sqlcipher
class TestJournalBeforeSend:
    def test_marker_is_journaled_locally_with_identity(self, tmp_path: Path) -> None:
        clock, store, _, session = build_session(tmp_path)
        record_session(clock, session)
        stored = store.list_markers("sess1")
        on_disk = [
            json.loads(line)
            for line in (session.directory / "markers.jsonl").read_text().splitlines()
        ]
        assert [m.seq_no for m in stored] == [1, 2]
        assert [e["seq_no"] for e in on_disk] == [1, 2]
        assert all(e["onset_utc_ns"] and e["event_id"] for e in on_disk)
        store.close()


class TestEventBusIds:
    def test_ids_do_not_collide_within_one_timestamp(self) -> None:
        ids = {Event("t", None).event_id for _ in range(2000)}
        assert len(ids) == 2000

    def test_publish_assigns_increasing_seq_no_and_stamps(self) -> None:
        async def run() -> list:
            bus = EventBus()
            await bus.start()
            events = [Event("t", i) for i in range(3)]
            for e in events:
                await bus.publish(e)
            return events

        events = asyncio.run(run())
        assert [e.seq_no for e in events] == [1, 2, 3]
        d = events[0].to_dict()
        assert d["utc_ns"] > 0 and d["monotonic_ns"] > 0

    @pytest.mark.parametrize("field", ["event_id", "seq_no", "monotonic_ns", "utc_ns"])
    def test_serialised_event_exposes_identity(self, field: str) -> None:
        assert field in Event("t", None).to_dict()
