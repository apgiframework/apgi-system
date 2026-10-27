"""Public driver interface.

Drivers subclass :class:`BaseDriver` (the abstract ``Device``), emit
:class:`SignalChunk` blocks and describe themselves with
:class:`DeviceCapabilities`. The names are aliases of the classes in
``acquisition.device`` so there is a single implementation.
"""

from __future__ import annotations

from ..device import Device as BaseDriver
from ..device import DeviceCapabilities, DeviceInfo, SampleChunk as SignalChunk, SyncProbe

__all__ = ["BaseDriver", "DeviceCapabilities", "DeviceInfo", "SignalChunk", "SyncProbe"]
