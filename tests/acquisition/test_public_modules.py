"""The register's module names exist and expose the implementations (no divergent copies)."""

from __future__ import annotations

import importlib

import pytest

CASES = [
    (
        "apgi_framework.acquisition.device_registry",
        "DeviceRegistry",
        "apgi_framework.acquisition.registry",
    ),
    (
        "apgi_framework.acquisition.device_manager",
        "DeviceManager",
        "apgi_framework.acquisition.manager",
    ),
    ("apgi_framework.acquisition.event_markers", "MarkerLog", "apgi_framework.acquisition.markers"),
    (
        "apgi_framework.acquisition.local_recorder",
        "LocalRecorder",
        "apgi_framework.acquisition.recorder",
    ),
    (
        "apgi_framework.acquisition.upload_agent",
        "UploadAgent",
        "apgi_framework.acquisition.uploader",
    ),
    ("apgi_framework.acquisition.clock_sync", "ClockMapper", "apgi_framework.acquisition.clock"),
    (
        "apgi_framework.interoperability.bids_export",
        "export_recording_to_bids",
        "apgi_framework.acquisition.bids",
    ),
    ("apgi_framework.acquisition.drivers.base", "BaseDriver", "apgi_framework.acquisition.device"),
    ("apgi_framework.synchronization", "ClockMapper", "apgi_framework.acquisition.clock"),
]


@pytest.mark.parametrize("public,name,impl", CASES)
def test_public_module_is_the_implementation(public: str, name: str, impl: str) -> None:
    a = getattr(importlib.import_module(public), name)
    b = getattr(importlib.import_module(impl), "Device" if name == "BaseDriver" else name)
    assert a is b


@pytest.mark.parametrize("module", ["eeg", "ecg", "ppg", "simulated", "base", "lsl"])
def test_driver_modules_exist(module: str) -> None:
    assert importlib.import_module(f"apgi_framework.acquisition.drivers.{module}")


def test_package_level_names() -> None:
    import apgi_framework.acquisition as acq

    for name in (
        "SessionMilestone",
        "compute_session_qc",
        "LiveStream",
        "ChunkIngestService",
        "default_hardware_registry",
        "ThreeDomainClock",
        "KeychainKeyProvider",
        "derive",
    ):
        assert hasattr(acq, name) and name in acq.__all__
