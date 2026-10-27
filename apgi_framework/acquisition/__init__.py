"""Device acquisition: drivers, clock mapping, recording, offline-first sync and BIDS export.

Typical use::

    manager = DeviceManager()
    manager.add("simulated", sampling_rate_hz=250.0)
    with SessionStore(path, key=key) as store:
        session = AcquisitionSession(store=store, sessions_dir=root, manager=manager, subject="01")
        session.prepare(); session.start(background=True)
        session.mark("stimulus", code=1)
        session.stop()
        session.export_bids(bids_root, task="rest")

The encrypted store needs SQLCipher (``pip install sqlcipher3-binary`` on Linux).
"""

from .bids import (
    BidsExportResult,
    export_recording_to_bids,
    export_session_to_bids,
    validate_bids_eeg,
)
from .clock import ClockFit, ClockMapper
from .device import (
    ChannelSpec,
    Device,
    DeviceCapabilities,
    DeviceInfo,
    DeviceState,
    SampleChunk,
    SyncProbe,
)
from .drivers.simulated import SimulatedDevice
from .errors import (
    AcquisitionError,
    ClockSyncError,
    DeviceError,
    DeviceNotFoundError,
    ExportError,
    InvalidTransition,
    NetworkUnavailable,
    PermanentTransportError,
    RecorderError,
    StoreError,
    StoreKeyError,
    StoreUnavailableError,
    TransportError,
)
from .integrity import chunk_digests, crc32c
from .manager import DeviceHealth, DeviceManager
from .markers import EventMarker, MarkerLog, marker_device_time_s, marker_sample_index
from .pin import PinError, TaskPin
from .qc import QCReport, compute_qc
from .recorder import (
    LocalRecorder,
    Recording,
    RecordingSummary,
    SegmentRecord,
    verify_write_once,
)
from .registry import DeviceRegistry, default_registry
from .session import AcquisitionSession, recover_session
from .state_machine import (
    SessionEvent,
    SessionMachine,
    SessionState,
    recover_interrupted_sessions,
)
from .store import SessionStore
from .uploader import (
    HttpTransport,
    LocalDirectoryTransport,
    Transport,
    UploadAgent,
    UploadReport,
    purge_verified_local_data,
)

__all__ = [
    "AcquisitionError",
    "AcquisitionSession",
    "BidsExportResult",
    "ChannelSpec",
    "ClockFit",
    "ClockMapper",
    "ClockSyncError",
    "Device",
    "DeviceCapabilities",
    "DeviceError",
    "DeviceHealth",
    "DeviceInfo",
    "DeviceManager",
    "DeviceNotFoundError",
    "DeviceRegistry",
    "DeviceState",
    "EventMarker",
    "ExportError",
    "HttpTransport",
    "InvalidTransition",
    "LocalDirectoryTransport",
    "LocalRecorder",
    "MarkerLog",
    "NetworkUnavailable",
    "PermanentTransportError",
    "PinError",
    "QCReport",
    "RecorderError",
    "Recording",
    "RecordingSummary",
    "SampleChunk",
    "SegmentRecord",
    "SessionEvent",
    "SessionMachine",
    "SessionState",
    "SessionStore",
    "SimulatedDevice",
    "StoreError",
    "StoreKeyError",
    "StoreUnavailableError",
    "SyncProbe",
    "TaskPin",
    "Transport",
    "TransportError",
    "UploadAgent",
    "UploadReport",
    "chunk_digests",
    "compute_qc",
    "crc32c",
    "default_registry",
    "export_recording_to_bids",
    "export_session_to_bids",
    "marker_device_time_s",
    "marker_sample_index",
    "purge_verified_local_data",
    "recover_interrupted_sessions",
    "recover_session",
    "validate_bids_eeg",
    "verify_write_once",
]
