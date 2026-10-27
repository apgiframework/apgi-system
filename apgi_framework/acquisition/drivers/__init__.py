"""Acquisition drivers. ``simulated`` is the reference; the others read LSL or simulate one modality."""

from .base import BaseDriver, DeviceCapabilities, DeviceInfo, SignalChunk, SyncProbe
from .ecg import ECGDevice, SimulatedECGDevice
from .eeg import EEGDevice, SimulatedEEGDevice
from .lsl import LslDevice
from .ppg import PPGDevice, SimulatedPPGDevice
from .simulated import SimulatedDevice

__all__ = [
    "BaseDriver",
    "DeviceCapabilities",
    "DeviceInfo",
    "ECGDevice",
    "EEGDevice",
    "LslDevice",
    "PPGDevice",
    "SignalChunk",
    "SimulatedDevice",
    "SimulatedECGDevice",
    "SimulatedEEGDevice",
    "SimulatedPPGDevice",
    "SyncProbe",
]
