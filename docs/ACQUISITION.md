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
* A read returns a `SampleChunk` (also exported as `SignalChunk` from `drivers/base.py`):
  `(n_samples, n_channels)` data, the device-clock time of the first sample, and
  `first_sample_index`, so a lost chunk shows up as an index gap.
* `DeviceCapabilities` (`Device.capabilities`) states what a driver provides: device
  timestamps, hardware markers, clock sync and `data_origin`. Device time is reported by the
  device or absent, never invented: a driver with `device_timestamps=False` must leave
  `first_device_time_s` as `None`, and the base class raises `DeviceError` if a chunk
  disagrees with the declared capability. `drivers/base.py` re-exports the driver names
  (`BaseDriver`, `SignalChunk`, `DeviceCapabilities`).
* `DeviceInfo.data_origin` is `"real"` or `"synthetic"`. `SimulatedDevice` is `"synthetic"`;
  the label is written to `recording.json` and to the BIDS `_eeg.json` (`DataOrigin`).
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
then mapped to device time and sample index on demand. Every marker also carries a stable
`event_id` (UUID4 hex), a per-session `seq_no` (1, 2, 3, ... without gaps, assigned under the
log's lock) and `onset_utc_ns` (wall clock, taken at the same moment as the monotonic
`onset_host_ns`). The monotonic stamp orders events within a run; the UTC stamp relates them
to other systems. Markers are appended to `markers.jsonl` and the encrypted store before
anything is uploaded; the store holds a unique `(session_id, seq_no)` index. The framework
`EventBus` stamps its events the same way (`event_id`, `seq_no`, `monotonic_ns`, `utc_ns`). Hardware markers additionally write
their code into the device's own stream, so the two stamps can be cross-checked.

## Recording

`LocalRecorder` writes a session folder: `recording.json`, `markers.jsonl`, `sync.jsonl`,
`data/segment-N.f32` (little-endian float32, interleaved) with `data/chunks-N.jsonl`
(index with, per chunk, SHA-256, CRC32C and the legacy CRC32), and `manifest.json` (SHA-256 per
sealed segment) plus a `COMPLETE` file after a clean close. CRC32C is the Castagnoli
polynomial (`zlib.crc32` is a different one); it uses the optional `crc32c` package and a
pure-Python fallback with identical results. Data is flushed before its index line, segments rotate and seal
at a configurable size, and `LocalRecorder.recover()` repairs a crash by dropping a torn or
corrupt trailing chunk (any of the three checks failing marks a chunk corrupt). Sealed
segments, their index files, `markers.jsonl` and `sync.jsonl` are made read-only when sealed;
`verify_write_once(session_dir)` lists any file that is missing, has changed since sealing
or is writable again, and a recorder refuses to start in a folder that already holds a
recording. Sealed files are reported through `on_file_ready`, which is how
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
* After each upload the agent reads the file back (`Transport.exists` compares the server's
  SHA-256). Only that answer sets `server_verified` on the queue item; an upload the server
  does not confirm is retried. The agent never deletes local data;
  `purge_verified_local_data(store, session_id)` removes a session's queued files, and only
  when every one is `server_verified`.
* Unexpected errors in the agent's background loop are logged with their traceback and
  counted (`agent.loop_errors`, `agent.last_loop_error`); per-file failures keep their
  message in the queue (`UploadItem.last_error`).

## Hardware classes, drivers and stimulation safety

`default_hardware_registry` holds eleven hardware classes with Track B priorities: EEG, ECG and
behavioural devices (1), eye tracking/pupillometry (2, the real-time surrogate), PPG,
respiration, EDA (offline covariate) and actigraphy (3, context), and fMRI (P03), TMS/tFUS (P05)
and iEEG (P06) as **import workflows** for data produced elsewhere. `SURROGATE_RANKING` puts
pupillometry first and EDA last: skin-conductance responses lag by seconds, far beyond the 100 ms
inference cadence, so `surrogate_usable_in_realtime("eda")` is false.

This software never drives or bypasses stimulation equipment. `require_controllable()` raises
`StimulationControlError` for every class, no live driver can be bound to an import workflow, and
`bind_driver` refuses any driver exposing methods such as `fire`, `set_intensity` or
`bypass_interlock`.

Drivers (`drivers/`): `SimulatedEEGDevice`, `SimulatedECGDevice`, `SimulatedPPGDevice` (synthetic,
registered as `simulated-eeg|ecg|ppg`) and `EEGDevice`, `ECGDevice`, `PPGDevice` (registered as
`lsl-eeg|ecg|ppg`) which read Lab Streaming Layer streams through the optional `pylsl` package.
LSL timestamps are used as delivered; a timestamp gap becomes an index gap that QC reports as
dropped samples. Vendor-SDK drivers are not included (decision D10 is open).

## Live stream and process split

`LiveStream` is a bounded drop-oldest buffer that never blocks the acquisition path;
`ProcessLiveConsumer` feeds a function in another process through a bounded queue with the same
rule. The recorder stays the lossless path, so a stalled display costs only live frames.

## Session milestones, journal and retention

Besides the state machine, nine milestones are recorded once each in the encrypted store:
`session_created`, `configuration_verified`, `recording_started`, `events_and_chunks_journaled`,
`recording_completed`, `local_manifest_finalized`, `upload_pending`, `server_verified` and
`local_retention_policy_applied`. Every chunk's index entry and digests are journaled in the
SQLCipher store (`chunk_journal`). `RetentionPolicy` / `apply_retention_policy` delete local data
only when asked to, only after every file is server-verified, and after `keep_days`.
Raw data is immutable; corrections are written with `derive()` / `write_derived()` under
`derived/<name>/` with provenance naming the raw manifest hash.

`open_store_with_keychain()` keeps the SQLCipher key wrapped (AES-256-GCM) by a master key held
in the OS keychain (`keyring`, optional) so a copy of the data folder alone cannot be opened.

## Chunk upload to apgi-api

`ChunkIngestService` implements the server side of the three-stage upload (signed URL, direct
bucket `PUT`, commit with existence, size, authorisation, sequence and CRC32C checks, then a job
that recomputes SHA-256) and sets each chunk `accepted`, `quarantined` or `rejected`.
`interoperability.ApiClient` and `SignedUrlTransport` are the client; `UploadAgent` uses the
transport unchanged. The HTTP routes and V4 signing for the real bucket belong to the apgi-api
service; `LocalObjectStore` is the in-repo reference implementation.

## Quality control and task pin

`compute_session_qc()` computes all twelve per-session metrics (the four below plus clock-offset
uncertainty, electrode impedance, signal saturation, motion artifacts, cardiac peak-detection
quality, marker-to-signal latency, upload completeness and firmware/calibration). Each is
`pass`, `warn`, `fail` or `n/a` (never `pass` when its input is missing), against `QCThresholds`
that a protocol may override through `synchronization.TimingTolerance`; a tolerance is registered
once per preregistered protocol and has no default. `ThreeDomainClock` maps device, monotonic and
UTC time using UTC anchors.

`compute_qc(recording)` returns a `QCReport` with four metrics: effective sampling rate (and
its error in ppm against nominal, from device timestamps), dropped samples (gaps), duplicate
samples/chunks, and timestamp discontinuities (a chunk whose device time disagrees with its
sample index by more than 0.5 ms). Nothing is corrected. Recordings without device timestamps
report no rate and no discontinuities.

`TaskPin(task_manifest_version, battery_version, task_manifest_sha256)` fixes the protocol a
session ran. Pass it as `AcquisitionSession(task_pin=...)`; it is stored with the session when
it is created, written to `recording.json` and to the BIDS sidecar, and `require_pin=True`
refuses to create a session without one. `TaskPin.from_manifest()` hashes the canonical JSON
of a manifest mapping.

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
