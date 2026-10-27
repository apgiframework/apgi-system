"""Bounded live stream: slow rendering must never block acquisition.

The recorder is the authoritative, lossless path. The live stream is a separate, bounded,
lossy tap for displays and online processing: :meth:`LiveStream.put` never blocks and never
raises; when the buffer is full the *oldest* chunk is discarded and counted. Consumers see
``dropped`` so a display can show that it is behind.

:class:`ProcessLiveConsumer` runs a consumer function in a separate process, fed through a
bounded multiprocessing queue with the same drop-on-full rule, so acquisition, processing
and GUI can be split across processes where one process cannot keep up.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
from collections import deque
from typing import Any, Callable, Deque, List, Optional

from .device import SampleChunk


class LiveStream:
    """Thread-safe drop-oldest ring of chunks."""

    def __init__(self, maxlen: int = 256) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self._buf: Deque[SampleChunk] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self.maxlen = maxlen
        self.received = 0
        self.dropped = 0

    def put(self, chunk: SampleChunk) -> None:
        """Add a chunk in O(1) without blocking; discards the oldest if full."""
        with self._lock:
            if len(self._buf) == self.maxlen:
                self.dropped += 1
            self._buf.append(chunk)
            self.received += 1
            self._ready.notify()

    # usable directly as a DeviceManager sink
    __call__ = put

    def get(self, timeout: Optional[float] = None) -> Optional[SampleChunk]:
        """Oldest buffered chunk, waiting up to ``timeout`` seconds; ``None`` if none arrived."""
        with self._ready:
            if not self._buf:
                self._ready.wait(timeout)
            return self._buf.popleft() if self._buf else None

    def drain(self) -> List[SampleChunk]:
        with self._lock:
            out = list(self._buf)
            self._buf.clear()
            return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


def _consumer_main(q: "mp.Queue[Any]", fn: Callable[[Any], None]) -> None:
    while True:
        item = q.get()
        if item is None:
            return
        fn(item)


class ProcessLiveConsumer:
    """Feed chunks to ``fn`` in another process through a bounded queue.

    ``fn`` must be a picklable top-level function. :meth:`feed` never blocks: when the queue
    is full the chunk is dropped and counted in ``dropped``.
    """

    def __init__(self, fn: Callable[[Any], None], maxsize: int = 64) -> None:
        ctx = mp.get_context("spawn")
        self._queue: "mp.Queue[Any]" = ctx.Queue(maxsize=maxsize)
        self._proc = ctx.Process(target=_consumer_main, args=(self._queue, fn), daemon=True)
        self.dropped = 0
        self.sent = 0

    def start(self) -> None:
        self._proc.start()

    def feed(self, chunk: SampleChunk) -> bool:
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            self.dropped += 1
            return False
        self.sent += 1
        return True

    __call__ = feed

    def close(self, timeout: float = 10.0) -> None:
        """Ask the consumer to finish and wait for it; terminates it if it does not."""
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            pass
        self._proc.join(timeout)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout)

    @property
    def alive(self) -> bool:
        return self._proc.is_alive()
