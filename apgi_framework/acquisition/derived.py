"""Derived datasets: corrections never touch raw data.

Raw recordings are immutable (sealed segments are read-only and hashed in the manifest).
Any correction - re-referencing, filtering, gap filling, timestamp repair - is written as a
new dataset under ``<session>/derived/<name>/`` with a ``provenance.json`` that names the raw
manifest it came from. :func:`write_derived` verifies the raw files were not changed by the
operation and refuses to overwrite an existing derived dataset.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from .errors import RecorderError
from .recorder import LocalRecorder, verify_write_once

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_derived(
    session_dir: Path,
    name: str,
    samples: NDArray[np.floating[Any]],
    *,
    operation: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``samples`` as derived dataset ``name`` with provenance; returns its directory."""
    session_dir = Path(session_dir)
    if not _NAME.match(name):
        raise RecorderError(f"invalid derived dataset name {name!r}")
    problems = verify_write_once(session_dir)
    if problems:
        raise RecorderError(f"raw data is not intact, refusing to derive from it: {problems}")
    out = session_dir / "derived" / name
    if out.exists():
        raise RecorderError(f"derived dataset {name!r} already exists")
    manifest = session_dir / "manifest.json"
    out.mkdir(parents=True)
    data_path = out / "data.npy"
    np.save(data_path, np.asarray(samples, dtype=np.float32))
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "name": name,
                "operation": operation,
                "parameters": parameters or {},
                "source_manifest_sha256": _sha256_file(manifest),
                "data_sha256": _sha256_file(data_path),
                "raw_modified": False,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    return out


def derive(
    session_dir: Path,
    name: str,
    fn: Callable[[NDArray[np.float32]], NDArray[np.floating[Any]]],
    *,
    operation: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Path:
    """Load the raw samples, apply ``fn`` to a *copy*, write the result as a derived dataset."""
    recording = LocalRecorder.read(Path(session_dir))
    result = fn(recording.samples.copy())
    out = write_derived(session_dir, name, result, operation=operation, parameters=parameters)
    after = verify_write_once(session_dir)
    if after:  # fn must not have reached the files behind the copy
        raise RecorderError(f"raw data changed during derivation: {after}")
    return out
