# Device acquisition

`apgi_framework.acquisition` records data from an amplifier (real or simulated),
keeps working with no network, and later uploads and exports it. It has three
layers: *devices* produce samples, a *session* records them safely, and *sync and
export* move the finished recording elsewhere.

```
Device (driver) --chunks--> DeviceManager --> AcquisitionSession --> LocalRecorder (files)
   ^  sync probes                |                  |  state, markers, queue
   +---- ClockMapper <-----------+                  v
                                         SessionStore (SQLCipher)  <-- UploadAgent --> Transport
                                                    \--> BIDS export (BrainVision + TSV/JSON)
```

## Quick start

`examples/acquisition_quickstart.py` runs the whole cycle against the simulated
device (record, mark events, export to BIDS, upload to a folder):

```bash
pip install sqlcipher3-binary      # Linux x86-64; see "SQLCipher" below for other platforms
python examples/acquisition_quickstart.py
```

```python
manager = DeviceManager()
manager.add("simulated", sampling_rate_hz=250.0)

with SessionStore(path, key=key) as store:                 # encrypted at rest
    session = AcquisitionSession(store=store, sessions_dir=root, manager=manager, subject="01")
    session.prepare()                                      # connect, sync clock, open recorder
    session.start(background=True)                         # a thread polls and records
    session.mark("stimulus", code=1)                       # host-clock event
    session.mark("trigger", code=2, hardware=True)         # also written into the data stream
    summary = session.stop()                               # drain, seal, queue for upload
    session.export_bids(bids_root, task="rest", power_line_hz=50.0)
    UploadAgent(store, LocalDirectoryTransport(dest)).run_once()
```

## Devices

* `Device` (`device.py`) is the driver interface. A driver implements `_do_connect`,
  `_do_disconnect`, `_do_start_stream`, `_do_stop_stream`, `_do_read` and
  `_do_sync_probe`; the base class enforces `DISCONNECTED -> CONNECTED -> STREAMING`
  and turns any unexpected driver exception into a `DeviceError` and state `ERROR`.
* A read returns a `SampleChunk`: `(n_samples, n_channels)` data, the device-clock time
  of the first sample, and `first_sample_index`, so a lost chunk shows up as an index gap.
* `DeviceRegistry` maps names to factories. `default_registry` contains `simulated`;
  third-party drivers can register through the `apgi.acquisition.drivers` entry-point
  group or `@registry.driver("name")`.
* `DeviceManager` owns devices, one `ClockMapper` each, and the polling loop. One failing
  device or sink is reported to error listeners and never stops the others.
  `drain()` reads what is buffered (bounded by a timeout, because a live stream can
  outrun a slow consumer).
* `SimulatedDevice` is deterministic (same seed, same data, independent of read sizes) and
  has its own clock with configurable offset, drift and probe jitter, plus fault
  injection (`drop_chunks`, `fail_after_samples`, `fail_on_connect`). All time comes from an
  injectable `clock_ns`, so tests never sleep.

## Clock mapping and markers

A device clock differs from the host clock by an offset and a small rate error.
`ClockMapper` takes `SyncProbe` round trips (host send, device reading, host receive),
keeps the fastest half of a sliding window (slow round trips make the midpoint
assumption unreliable, as in NTP's clock filter) and fits `host = a + b * device` by
least squares. With fewer than four probes only an offset is estimated.

Markers (`EventMarker`, `MarkerLog`) are stamped on the **host** clock by stimulus code,
then mapped to device time and sample index on demand. Hardware markers additionally write
their code into the device's own stream, so the two stamps can be cross-checked.

## Recording

`LocalRecorder` writes a session folder: `recording.json`, `markers.jsonl`, `sync.jsonl`,
`data/segment-N.f32` (little-endian float32, interleaved) with `data/chunks-N.jsonl`
(index with CRC32), and `manifest.json` (SHA-256 per sealed segment) plus a `COMPLETE`
file after a clean close. Data is flushed before its index line, segments rotate and seal
at a configurable size, and `LocalRecorder.recover()` repairs a crash by dropping a torn or
corrupt trailing chunk. Sealed files are reported through `on_file_ready`, which is how
they reach the upload queue while recording continues.

## Offline-first state machine

Recording never needs the network. Connectivity only gates the upload phase:

```
CREATED -> READY -> RECORDING <-> PAUSED -> STOPPING -> FINALIZED -> UPLOADING -> SYNCED
                                                            ^            |
                              (connection lost / items failed)+----------+
RECORDING|PAUSED|STOPPING -> INTERRUPTED   (found at restart)      INTERRUPTED|FAILED -> FINALIZED
```

Every transition is written to the encrypted store before the in-memory state changes. At
start-up, `recover_interrupted_sessions(store)` marks sessions that were recording as
`INTERRUPTED` and returns their ids; `recover_session(store, id)` repairs the files, queues
them and finalizes. Sessions that were uploading return to `FINALIZED` with their queue
intact.

## Encrypted store

`SessionStore` keeps sessions, the state log, sync points, markers and the upload queue in a
SQLCipher database (WAL, `synchronous=FULL`, versioned schema).

* It never falls back to plain SQLite: if SQLCipher is missing, or the loaded SQLite is not
  SQLCipher, it raises `StoreUnavailableError`.
* Give it exactly one of a 32-byte `key` or a `passphrase`, or use
  `SessionStore.from_env(path)` with `APGI_STORE_KEY` (64 hex characters or a passphrase).
  Do not hard-code keys; a lost key means lost data, and there is no recovery path.
* The store protects the *database*. Recording files under `sessions_dir` are plain files
  (needed for BIDS and upload); put them on an encrypted volume if that matters.

### SQLCipher

`sqlcipher3-binary` provides wheels for x86-64 Linux only. `requirements.txt` and the
`acquisition` extra install it there; on macOS, Windows or ARM Linux install libsqlcipher
and `pip install sqlcipher3`. Without it the store tests are skipped and the store cannot be opened.

## Uploading

`UploadAgent` drains the queue through a `Transport`:

* `LocalDirectoryTransport` copies atomically and verifies SHA-256 (mounted share, USB disk).
* `HttpTransport` does `HEAD`/`PUT` to `{base_url}/{session_id}/{path}` with an
  `X-Content-SHA256` header. 408/425/429/5xx are retried, other 4xx are permanent, and
  connection failures mean *offline*.
* Uploads are idempotent (already-present files are skipped), failures back off
  exponentially with jitter (`base * 2^attempts`, capped), network loss pauses the queue
  without consuming attempts, and after `max_attempts` an item is parked as failed
  (`store.requeue_failed(session_id)` retries it). A session reaches `SYNCED` only when every
  file is confirmed; permanent failures return it to `FINALIZED`.
* Files are re-hashed before sending; a file changed since sealing is rejected.

## BIDS export

`export_recording_to_bids` / `AcquisitionSession.export_bids` write a BIDS 1.9 raw EEG
dataset: BrainVision `.vhdr/.vmrk/.eeg` (binary float32, multiplexed), `_eeg.json`,
`_channels.tsv`, `_events.tsv`/`.json`, plus `dataset_description.json`,
`participants.tsv/.json` and `README`.

* Event onsets are seconds from the first stored sample, computed through the clock mapping;
  markers outside the recorded span are dropped and listed in `result.dropped_markers`.
* Lost samples (a pause or dropped chunks) are written as `gap_fill_value` (default 0) **and**
  appear as `gap` rows in `events.tsv`; nothing is interpolated.
* Mains frequency and reference are `n/a` unless you pass them.
* `validate_bids_eeg(root)` is a structural self-check. Run the official `bids-validator`
  as well before publishing.
* `apgi_framework.export.BIDSExporter` is older and is not BIDS-EEG compliant (CSV/NPY
  output, no events file); use this module for new work.

## Limits

* One device is recorded per `AcquisitionSession` (the manager can hold several devices; start
  one session per device).
* The recorder is single-threaded per session; sustained rates of tens of thousands of
  chunks per second are not supported (real EEG rates are well below that).
* There is no driver for real hardware yet; implement `Device` for your amplifier.
