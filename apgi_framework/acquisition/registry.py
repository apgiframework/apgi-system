"""Registry that maps driver names to device factories."""

from __future__ import annotations

import threading
from importlib import metadata
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .device import Device
from .errors import DeviceNotFoundError

DeviceFactory = Callable[..., Device]
ENTRY_POINT_GROUP = "apgi.acquisition.drivers"


class DeviceRegistry:
    """Thread-safe name -> factory table.

    Third-party drivers can register themselves through the
    ``apgi.acquisition.drivers`` entry-point group; those are loaded lazily the
    first time an unknown name is requested.
    """

    def __init__(self, *, load_entry_points: bool = False) -> None:
        self._factories: Dict[str, DeviceFactory] = {}
        self._lock = threading.RLock()
        self._load_entry_points = load_entry_points
        self._entry_points_loaded = False

    def register(
        self, name: str, factory: Union[Type[Device], DeviceFactory], *, replace: bool = False
    ) -> None:
        if not name or not name.strip():
            raise ValueError("driver name must not be empty")
        with self._lock:
            if name in self._factories and not replace:
                raise ValueError(f"driver {name!r} is already registered")
            self._factories[name] = factory

    def unregister(self, name: str) -> None:
        with self._lock:
            if name not in self._factories:
                raise DeviceNotFoundError(f"driver {name!r} is not registered")
            del self._factories[name]

    def driver(self, name: str, *, replace: bool = False) -> Callable[[Any], Any]:
        """Return a class decorator that registers the class under ``name``."""

        def decorator(cls: Any) -> Any:
            self.register(name, cls, replace=replace)
            return cls

        return decorator

    def create(self, name: str, **config: Any) -> Device:
        factory = self._lookup(name)
        device = factory(**config)
        if not isinstance(device, Device):
            raise TypeError(f"factory for {name!r} returned {type(device).__name__}, not a Device")
        return device

    def names(self) -> List[str]:
        self._maybe_load_entry_points()
        with self._lock:
            return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.names()

    def _lookup(self, name: str) -> DeviceFactory:
        with self._lock:
            factory: Optional[DeviceFactory] = self._factories.get(name)
        if factory is None:
            self._maybe_load_entry_points()
            with self._lock:
                factory = self._factories.get(name)
        if factory is None:
            raise DeviceNotFoundError(
                f"no driver named {name!r}; registered: {', '.join(self.names()) or 'none'}"
            )
        return factory

    def _maybe_load_entry_points(self) -> None:
        if not self._load_entry_points or self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
            with self._lock:
                if ep.name not in self._factories:
                    self._factories[ep.name] = ep.load()


def _build_default_registry() -> DeviceRegistry:
    from .drivers import (
        ECGDevice,
        EEGDevice,
        PPGDevice,
        SimulatedDevice,
        SimulatedECGDevice,
        SimulatedEEGDevice,
        SimulatedPPGDevice,
    )

    registry = DeviceRegistry(load_entry_points=True)
    registry.register("simulated", SimulatedDevice)
    registry.register("simulated-eeg", SimulatedEEGDevice)
    registry.register("simulated-ecg", SimulatedECGDevice)
    registry.register("simulated-ppg", SimulatedPPGDevice)
    registry.register("lsl-eeg", EEGDevice)
    registry.register("lsl-ecg", ECGDevice)
    registry.register("lsl-ppg", PPGDevice)
    return registry


default_registry = _build_default_registry()
