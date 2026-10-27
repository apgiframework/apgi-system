"""Pin the task manifest and battery version a session was run with.

A recording is only interpretable against the exact protocol it ran. The pin is
fixed when the session is created, written into ``recording.json`` and the
store, and cannot be changed afterwards.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .errors import AcquisitionError


class PinError(AcquisitionError):
    """A task pin is missing, malformed or was changed after session start."""


@dataclass(frozen=True)
class TaskPin:
    """Immutable identity of the protocol a session ran."""

    task_manifest_version: str
    battery_version: str
    task_manifest_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.task_manifest_version.strip():
            raise PinError("task_manifest_version must not be empty")
        if not self.battery_version.strip():
            raise PinError("battery_version must not be empty")

    @classmethod
    def from_manifest(
        cls, manifest: Mapping[str, Any], *, battery_version: str, version_key: str = "version"
    ) -> "TaskPin":
        """Pin a manifest mapping: its version string plus a hash of its canonical JSON."""
        if version_key not in manifest:
            raise PinError(f"manifest has no {version_key!r} field")
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(str(manifest[version_key]), battery_version, digest)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_manifest_version": self.task_manifest_version,
            "battery_version": self.battery_version,
            "task_manifest_sha256": self.task_manifest_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskPin":
        try:
            return cls(
                str(data["task_manifest_version"]),
                str(data["battery_version"]),
                None if data.get("task_manifest_sha256") is None else str(data["task_manifest_sha256"]),
            )
        except KeyError as exc:
            raise PinError(f"pin is missing {exc.args[0]!r}") from exc
