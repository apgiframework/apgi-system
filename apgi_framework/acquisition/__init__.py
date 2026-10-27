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
from .chunk_ingest import (
    ChunkIngestService,
    ChunkManifest,
    ChunkStatus,
    LocalObjectStore,
    UploadForbidden,
)
from .clock import ClockFit, ClockMapper
from .clock_sync import ThreeDomainClock
from .derived import derive, write_derived
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
from .hardware import (
    HardwareClass,
    HardwareRegistry,
    StimulationControlError,
    default_hardware_registry,
    require_controllable,
    surrogate_usable_in_realtime,
)
from .integrity import chunk_digests, crc32c
from .key_management import KeychainKeyProvider, open_store_with_keychain
from .live import LiveStream, ProcessLiveConsumer
from .manager import DeviceHealth, DeviceManager
from .markers import EventMarker, MarkerLog, marker_device_time_s, marker_sample_index
from .milestones import SessionMilestone
from .pin import PinError, TaskPin
from .qc import QCReport, QCThresholds, SessionQC, compute_qc, compute_session_qc
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
    RetentionPolicy,
    Transport,
    UploadAgent,
    UploadReport,
    apply_retention_policy,
    purge_verified_local_data,
)

__all__ = [
    "AcquisitionError",
    "AcquisitionSession",
    "apply_retention_policy",
    "BidsExportResult",
    "ChannelSpec",
    "chunk_digests",
    "ChunkIngestService",
    "ChunkManifest",
    "ChunkStatus",
    "ClockFit",
    "ClockMapper",
    "ClockSyncError",
    "compute_qc",
    "compute_session_qc",
    "crc32c",
    "default_hardware_registry",
    "default_registry",
    "derive",
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
    "export_recording_to_bids",
    "export_session_to_bids",
    "ExportError",
    "HardwareClass",
    "HardwareRegistry",
    "HttpTransport",
    "InvalidTransition",
    "KeychainKeyProvider",
    "LiveStream",
    "LocalDirectoryTransport",
    "LocalObjectStore",
    "LocalRecorder",
    "marker_device_time_s",
    "marker_sample_index",
    "MarkerLog",
    "NetworkUnavailable",
    "open_store_with_keychain",
    "PermanentTransportError",
    "PinError",
    "ProcessLiveConsumer",
    "purge_verified_local_data",
    "QCReport",
    "QCThresholds",
    "RecorderError",
    "Recording",
    "RecordingSummary",
    "recover_interrupted_sessions",
    "recover_session",
    "require_controllable",
    "RetentionPolicy",
    "SampleChunk",
    "SegmentRecord",
    "SessionEvent",
    "SessionMachine",
    "SessionMilestone",
    "SessionQC",
    "SessionState",
    "SessionStore",
    "SimulatedDevice",
    "StimulationControlError",
    "StoreError",
    "StoreKeyError",
    "StoreUnavailableError",
    "surrogate_usable_in_realtime",
    "SyncProbe",
    "TaskPin",
    "ThreeDomainClock",
    "Transport",
    "TransportError",
    "UploadAgent",
    "UploadForbidden",
    "UploadReport",
    "validate_bids_eeg",
    "verify_write_once",
    "write_derived",
]
