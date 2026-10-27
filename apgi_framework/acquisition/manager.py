"""Lifecycle management for several acquisition devices.

The manager owns device objects, one :class:`ClockMapper` per device and the
polling loop that moves chunks to sinks (normally the recorder). One misbehaving
device never stops the others: failures are recorded per device and reported to
error listeners.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .clock import ClockFit, ClockMapper
from .device import Device, DeviceState, SampleChunk
from .errors import DeviceError, DeviceNotFoundError
from .registry import DeviceRegistry, default_registry

ChunkSink = Callable[[SampleChunk], None]
ErrorListener = Callable[[str, Exception], None]


@dataclass
class DeviceHealth:
    """Observed behaviour of one device."""

    device_id: str
    state: DeviceState
    chunks_read: int = 0
    samples_read: int = 0
    gaps_detected: int = 0
    last_chunk_host_ns: Optional[int] = None
    last_error: Optional[str] = None
    clock: Optional[ClockFit] = None


@dataclass
class _Managed:
    device: Device
    mapper: ClockMapper = field(default_factory=ClockMapper)
    chunks_read: int = 0
    samples_read: int = 0
    gaps_detected: int = 0
    next_expected_index: int = 0
    last_chunk_host_ns: Optional[int] = None
    last_error: Optional[str] = None


class DeviceManager:
    def __init__(
        self,
        registry: Optional[DeviceRegistry] = None,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._registry = registry or default_registry
        self._clock_ns = clock_ns
        self._devices: Dict[str, _Managed] = {}
        self._sinks: List[ChunkSink] = []
        self._error_listeners: List[ErrorListener] = []
        self._lock = threading.RLock()
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_polling = threading.Event()

    # -- registration -----------------------------------------------------
    def add(self, driver: str, **config: Any) -> str:
        """Create a device from a registered driver and manage it."""
        return self.add_device(self._registry.create(driver, **config))

    def add_device(self, device: Device) -> str:
        device_id = device.info.device_id
        with self._lock:
            if device_id in self._devices:
                raise DeviceError(f"device {device_id!r} is already managed")
            self._devices[device_id] = _Managed(device)
        return device_id

    def remove(self, device_id: str) -> None:
        managed = self._get(device_id)
        managed.device.disconnect()
        with self._lock:
            del self._devices[device_id]

    @property
    def device_ids(self) -> List[str]:
        with self._lock:
            return sorted(self._devices)

    def device(self, device_id: str) -> Device:
        return self._get(device_id).device

    def mapper(self, device_id: str) -> ClockMapper:
        return self._get(device_id).mapper

    def add_sink(self, sink: ChunkSink) -> None:
        with self._lock:
            self._sinks.append(sink)

    def add_error_listener(self, listener: ErrorListener) -> None:
        with self._lock:
            self._error_listeners.append(listener)

    # -- lifecycle --------------------------------------------------------
    def connect_all(self) -> Dict[str, Optional[Exception]]:
        """Connect every device; returns ``{id: None | error}`` without raising."""
        return self._for_each(lambda m: m.device.connect())

    def sync_clocks(
        self,
        device_id: Optional[str] = None,
        *,
        n_probes: int = 8,
        interval_s: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Dict[str, ClockFit]:
        """Probe device clocks ``n_probes`` times and refit each mapper."""
        fits: Dict[str, ClockFit] = {}
        targets = [device_id] if device_id else self.device_ids
        for dev_id in targets:
            managed = self._get(dev_id)
            for i in range(n_probes):
                managed.mapper.add_probe(managed.device.sync_probe())
                if interval_s > 0 and i < n_probes - 1:
                    sleep(interval_s)
            fit = managed.mapper.fit
            if fit is not None:
                fits[dev_id] = fit
        return fits

    def start_streaming(self) -> Dict[str, Optional[Exception]]:
        for managed in self._snapshot():
            managed.next_expected_index = 0
        return self._for_each(lambda m: m.device.start_stream())

    def stop_streaming(self) -> Dict[str, Optional[Exception]]:
        self.stop_polling()
        return self._for_each(
            lambda m: m.device.stop_stream() if m.device.state is DeviceState.STREAMING else None
        )

    def disconnect_all(self) -> Dict[str, Optional[Exception]]:
        self.stop_polling()
        return self._for_each(lambda m: m.device.disconnect())

    # -- data path --------------------------------------------------------
    def poll(self) -> List[SampleChunk]:
        """Read every streaming device once and hand new chunks to the sinks."""
        delivered: List[SampleChunk] = []
        for managed in self._snapshot():
            device_id = managed.device.info.device_id
            if managed.device.state is not DeviceState.STREAMING:
                continue
            try:
                chunk = managed.device.read()
            except Exception as exc:  # isolate: other devices keep streaming
                self._record_error(managed, exc)
                continue
            if chunk is None:
                continue
            if chunk.first_sample_index > managed.next_expected_index:
                managed.gaps_detected += 1
            managed.next_expected_index = chunk.first_sample_index + chunk.n_samples
            managed.chunks_read += 1
            managed.samples_read += chunk.n_samples
            managed.last_chunk_host_ns = chunk.host_receive_ns
            with self._lock:
                sinks = list(self._sinks)
            for sink in sinks:
                try:
                    sink(chunk)
                except Exception as exc:
                    self._record_error(managed, exc, label=f"sink failed for {device_id}")
            delivered.append(chunk)
        return delivered

    def drain(self, *, timeout_s: float = 2.0, max_rounds: int = 1_000_000) -> int:
        """Poll until the devices have nothing buffered; returns chunks delivered.

        ``poll`` reads one chunk per device, so a backlog needs repeated calls. On a
        live stream whose consumer is slower than the sample rate there may never be
        a moment with nothing buffered, so the loop also stops after ``timeout_s``
        of wall-clock time (and ``max_rounds`` polls).
        """
        deadline = time.monotonic() + timeout_s
        total = 0
        for _ in range(max_rounds):
            delivered = len(self.poll())
            if not delivered or time.monotonic() >= deadline:
                break
            total += delivered
        return total

    def start_polling(self, interval_s: float = 0.01) -> None:
        with self._lock:
            if self._poll_thread is not None and self._poll_thread.is_alive():
                raise DeviceError("polling already running")
            self._stop_polling.clear()
            self._poll_thread = threading.Thread(
                target=self._poll_loop, args=(interval_s,), name="apgi-device-poll", daemon=True
            )
            self._poll_thread.start()

    def stop_polling(self) -> None:
        thread = self._poll_thread
        if thread is None:
            return
        self._stop_polling.set()
        thread.join(timeout=5.0)
        self._poll_thread = None

    def health(self) -> Dict[str, DeviceHealth]:
        return {
            m.device.info.device_id: DeviceHealth(
                device_id=m.device.info.device_id,
                state=m.device.state,
                chunks_read=m.chunks_read,
                samples_read=m.samples_read,
                gaps_detected=m.gaps_detected,
                last_chunk_host_ns=m.last_chunk_host_ns,
                last_error=m.last_error,
                clock=m.mapper.fit,
            )
            for m in self._snapshot()
        }

    # -- internals --------------------------------------------------------
    def _get(self, device_id: str) -> _Managed:
        with self._lock:
            managed = self._devices.get(device_id)
        if managed is None:
            raise DeviceNotFoundError(f"no managed device {device_id!r}")
        return managed

    def _snapshot(self) -> List[_Managed]:
        with self._lock:
            return list(self._devices.values())

    def _for_each(self, action: Callable[[_Managed], Any]) -> Dict[str, Optional[Exception]]:
        results: Dict[str, Optional[Exception]] = {}
        for managed in self._snapshot():
            device_id = managed.device.info.device_id
            try:
                action(managed)
                results[device_id] = None
            except Exception as exc:
                self._record_error(managed, exc)
                results[device_id] = exc
        return results

    def _record_error(self, managed: _Managed, exc: Exception, label: str = "") -> None:
        managed.last_error = f"{label}: {exc}" if label else str(exc)
        with self._lock:
            listeners = list(self._error_listeners)
        for listener in listeners:
            listener(managed.device.info.device_id, exc)

    def _poll_loop(self, interval_s: float) -> None:
        while not self._stop_polling.is_set():
            if not self.poll():
                self._stop_polling.wait(interval_s)
