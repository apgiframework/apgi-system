"""Bounded live stream, derived datasets and keychain-wrapped store keys."""

from __future__ import annotations

import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from apgi_framework.acquisition import (
    LiveStream,
    LocalRecorder,
    ProcessLiveConsumer,
    RecorderError,
    SampleChunk,
    SessionStore,
    StoreKeyError,
    derive,
    open_store_with_keychain,
    verify_write_once,
    write_derived,
)
from apgi_framework.acquisition.key_management import KeychainKeyProvider
from tests.acquisition.conftest import build_session, record_session, requires_sqlcipher


def chunk(seq: int) -> SampleChunk:
    return SampleChunk("d", seq, seq * 4, float(seq), np.full((4, 1), float(seq)), seq)


class TestLiveStream:
    def test_put_never_blocks_and_drops_the_oldest(self) -> None:
        live = LiveStream(maxlen=3)
        for i in range(10):
            live.put(chunk(i))
        assert live.received == 10 and live.dropped == 7 and len(live) == 3
        assert [c.sequence for c in live.drain()] == [7, 8, 9]

    def test_get_waits_then_times_out(self) -> None:
        live = LiveStream(4)
        assert live.get(timeout=0.01) is None
        threading.Timer(0.02, live.put, args=(chunk(1),)).start()
        got = live.get(timeout=2.0)
        assert got is not None and got.sequence == 1

    def test_invalid_size(self) -> None:
        with pytest.raises(ValueError):
            LiveStream(0)

    @requires_sqlcipher
    def test_a_stalled_display_never_costs_recorded_data(self, tmp_path: Path) -> None:
        """Acceptance: slow rendering never blocks acquisition."""
        clock, store, manager, session = build_session(tmp_path)
        live = LiveStream(maxlen=2)  # tiny buffer and nobody reading it: a stalled display
        manager.add_sink(live)
        began = time.monotonic()
        summary = record_session(clock, session, seconds=6.0)
        assert time.monotonic() - began < 30
        assert live.dropped > 0 and len(live) == 2
        assert LocalRecorder.read(session.directory).gaps == []
        assert summary.n_chunks == live.received
        store.close()


def _count_to_file(path: str, item: Any) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{item.sequence}\n")


def _sleepy(item: Any) -> None:
    time.sleep(0.5)


class TestProcessConsumer:
    def test_chunks_reach_a_separate_process(self, tmp_path: Path) -> None:
        out = tmp_path / "seen.txt"
        consumer = ProcessLiveConsumer(partial(_count_to_file, str(out)), maxsize=32)
        consumer.start()
        for i in range(5):
            assert consumer.feed(chunk(i))
        consumer.close(timeout=30)
        assert not consumer.alive
        assert out.read_text().split() == ["0", "1", "2", "3", "4"]

    def test_slow_consumer_process_drops_instead_of_blocking(self) -> None:
        consumer = ProcessLiveConsumer(_sleepy, maxsize=2)
        consumer.start()
        began = time.monotonic()
        results = [consumer.feed(chunk(i)) for i in range(200)]
        assert time.monotonic() - began < 5.0  # feeding 200 chunks never waited on the consumer
        assert consumer.dropped > 0 and results.count(False) == consumer.dropped
        consumer.close(timeout=0.5)
        assert not consumer.alive


class TestDerived:
    def sealed(self, tmp_path: Path) -> Path:
        from apgi_framework.acquisition import ChannelSpec, DeviceInfo

        info = DeviceInfo("d", "usb", "m", 100.0, (ChannelSpec("A"), ChannelSpec("B")))
        rec = LocalRecorder(tmp_path / "s", info, fsync=False)
        rec.start()
        rec.write_chunk(SampleChunk("d", 0, 0, 1.0, np.arange(40.0).reshape(20, 2), 1))
        rec.close()
        return tmp_path / "s"

    def test_correction_is_a_new_dataset_with_provenance(self, tmp_path: Path) -> None:
        import hashlib
        import json

        s = self.sealed(tmp_path)
        before = hashlib.sha256((s / "manifest.json").read_bytes()).hexdigest()
        out = derive(s, "demeaned", lambda x: x - x.mean(axis=0), operation="demean")
        data = np.load(out / "data.npy")
        assert data.shape == (20, 2) and abs(float(data.mean())) < 1e-5
        prov = json.loads((out / "provenance.json").read_text())
        assert prov["source_manifest_sha256"] == before and prov["raw_modified"] is False
        assert prov["operation"] == "demean" and len(prov["data_sha256"]) == 64
        assert verify_write_once(s) == []  # raw untouched

    def test_the_function_gets_a_copy_so_raw_cannot_be_edited_through_it(
        self, tmp_path: Path
    ) -> None:
        s = self.sealed(tmp_path)
        raw_before = LocalRecorder.read(s).samples.copy()

        def mutate(x: np.ndarray) -> np.ndarray:
            x[:] = 0
            return x

        derive(s, "zeros", mutate, operation="zero")
        assert np.array_equal(LocalRecorder.read(s).samples, raw_before)

    def test_refuses_to_overwrite_and_to_derive_from_damaged_raw(self, tmp_path: Path) -> None:
        s = self.sealed(tmp_path)
        write_derived(s, "a", np.zeros((2, 2)), operation="x")
        with pytest.raises(RecorderError, match="already exists"):
            write_derived(s, "a", np.zeros((2, 2)), operation="x")
        seg = next((s / "data").glob("segment-*.f32"))
        seg.chmod(0o644)
        seg.write_bytes(b"\x01" * seg.stat().st_size)
        with pytest.raises(RecorderError, match="not intact"):
            write_derived(s, "b", np.zeros((2, 2)), operation="x")

    @pytest.mark.parametrize("name", ["../evil", "a/b", "", ".hidden", "x" * 80])
    def test_bad_names_are_refused(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(RecorderError, match="invalid derived dataset name"):
            write_derived(self.sealed(tmp_path), name, np.zeros((1, 1)), operation="x")


class FakeKeychain:
    def __init__(self) -> None:
        self.items: Dict[Any, str] = {}

    def get_password(self, service: str, account: str) -> Any:
        return self.items.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.items[(service, account)] = value


@requires_sqlcipher
class TestKeychainKeys:
    def test_store_reopens_with_the_same_wrapped_key(self, tmp_path: Path) -> None:
        kc = FakeKeychain()
        store, _ = open_store_with_keychain(tmp_path / "a.db", keychain=kc)
        store.create_session("s", directory=str(tmp_path))
        store.close()
        again, _ = open_store_with_keychain(tmp_path / "a.db", keychain=kc)
        assert again.get_session("s") is not None
        again.close()

    def test_the_data_key_is_never_written_in_the_clear(self, tmp_path: Path) -> None:
        kc = FakeKeychain()
        provider = KeychainKeyProvider(tmp_path / "k", keychain=kc)
        key = provider.get_or_create()
        blob = (tmp_path / "k").read_bytes()
        assert key not in blob and len(key) == 32 and len(blob) == 12 + 32 + 16
        assert (tmp_path / "k").stat().st_mode & 0o077 == 0
        assert provider.get_or_create() == key

    def test_without_the_keychain_entry_the_store_cannot_be_opened(self, tmp_path: Path) -> None:
        store, _ = open_store_with_keychain(tmp_path / "a.db", keychain=FakeKeychain())
        store.close()
        with pytest.raises(StoreKeyError, match="master key"):
            open_store_with_keychain(tmp_path / "a.db", keychain=FakeKeychain())

    def test_a_different_master_key_or_tampering_is_detected(self, tmp_path: Path) -> None:
        kc = FakeKeychain()
        p = KeychainKeyProvider(tmp_path / "k", keychain=kc)
        p.get_or_create()
        blob = bytearray((tmp_path / "k").read_bytes())
        blob[-1] ^= 1
        (tmp_path / "k").write_bytes(bytes(blob))
        with pytest.raises(StoreKeyError, match="could not be unwrapped"):
            p.get_or_create()
        other = FakeKeychain()
        other.set_password("apgi-acquisition", "store-master-key", "A" * 43 + "=")
        with pytest.raises(StoreKeyError):
            KeychainKeyProvider(tmp_path / "k", keychain=other).get_or_create()

    def test_wrong_key_still_fails_the_database_itself(self, tmp_path: Path) -> None:
        store, _ = open_store_with_keychain(tmp_path / "a.db", keychain=FakeKeychain())
        store.close()
        with pytest.raises(StoreKeyError):
            SessionStore(tmp_path / "a.db", key=bytes(32))

    def test_missing_keyring_package_is_a_clear_error(self, tmp_path: Path) -> None:
        import importlib.util

        from apgi_framework.acquisition import StoreUnavailableError

        if importlib.util.find_spec("keyring"):
            pytest.skip("keyring is installed")
        with pytest.raises(StoreUnavailableError, match="keyring"):
            KeychainKeyProvider(tmp_path / "k").get_or_create()
