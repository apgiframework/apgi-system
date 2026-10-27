"""Event markers (public module name; implementation in :mod:`.markers`)."""

from .markers import EventMarker, MarkerLog, marker_device_time_s, marker_sample_index

__all__ = ["EventMarker", "MarkerLog", "marker_device_time_s", "marker_sample_index"]
