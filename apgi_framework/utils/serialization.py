"""Serialization utilities for APGI Framework.

This module provides safe serialization alternatives to pickle, supporting JSON and msgpack
formats with custom encoders for APGI-specific data types.
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class APGIJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for APGI data types."""

    def default(self, obj: Any) -> Any:
        """Encode APGI-specific types to JSON-serializable format."""
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "numpy.ndarray",
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": obj.tolist(),
            }
        # Handle numpy scalars
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        # Handle pandas DataFrames
        elif isinstance(obj, pd.DataFrame):
            return {
                "__type__": "pandas.DataFrame",
                "columns": obj.columns.tolist(),
                "data": obj.to_dict(orient="records"),
            }
        # Handle pandas Series
        elif isinstance(obj, pd.Series):
            return {
                "__type__": "pandas.Series",
                "name": obj.name,
                "data": obj.to_dict(),
            }
        # Handle datetime objects
        elif isinstance(obj, datetime):
            return {"__type__": "datetime", "value": obj.isoformat()}
        # Handle date objects
        elif isinstance(obj, date):
            return {"__type__": "date", "value": obj.isoformat()}
        # Handle Path objects
        elif isinstance(obj, Path):
            return {"__type__": "Path", "value": str(obj)}
        # Handle objects with to_json method
        elif hasattr(obj, "to_json"):
            return obj.to_json()
        # Handle objects with __dict__ (simple objects)
        elif hasattr(obj, "__dict__"):
            return {
                "__type__": obj.__class__.__name__,
                "__module__": obj.__class__.__module__,
                "__dict__": obj.__dict__,
            }
        # Default fallback
        return super().default(obj)


class APGIJSONDecoder(json.JSONDecoder):
    """Custom JSON decoder for APGI data types."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize decoder with object hook."""
        kwargs["object_hook"] = self._object_hook
        super().__init__(*args, **kwargs)

    def _object_hook(self, obj: Dict[str, Any]) -> Any:
        """Decode JSON-serialized APGI types back to original types."""
        if not isinstance(obj, dict):
            return obj

        # Check if this is a special APGI type
        obj_type = obj.get("__type__")

        if obj_type == "numpy.ndarray":
            return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
        elif obj_type == "pandas.DataFrame":
            return pd.DataFrame(obj["data"])
        elif obj_type == "pandas.Series":
            return pd.Series(obj["data"], name=obj["name"])
        elif obj_type == "datetime":
            return datetime.fromisoformat(obj["value"])
        elif obj_type == "date":
            return date.fromisoformat(obj["value"])
        elif obj_type == "Path":
            return Path(obj["value"])
        elif obj_type and "__dict__" in obj:
            # Reconstruct simple objects (note: this is basic and may not work for complex objects)
            # For complex objects, use secure pickle or implement proper serialization
            return obj

        return obj


def serialize_json(data: Any, indent: int = 2) -> str:
    """Serialize data to JSON with custom encoder.

    Args:
        data: Data to serialize
        indent: JSON indentation level

    Returns:
        JSON string
    """
    return json.dumps(data, cls=APGIJSONEncoder, indent=indent)


def deserialize_json(json_str: str) -> Any:
    """Deserialize JSON string with custom decoder.

    Args:
        json_str: JSON string to deserialize

    Returns:
        Deserialized data
    """
    return json.loads(json_str, cls=APGIJSONDecoder)


def serialize_json_to_file(data: Any, file_path: Path, indent: int = 2) -> None:
    """Serialize data to JSON file.

    Args:
        data: Data to serialize
        file_path: Path to output JSON file
        indent: JSON indentation level
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=APGIJSONEncoder, indent=indent)


def deserialize_json_from_file(file_path: Path) -> Any:
    """Deserialize JSON file with custom decoder.

    Args:
        file_path: Path to JSON file

    Returns:
        Deserialized data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f, cls=APGIJSONDecoder)


def can_serialize_as_json(data: Any) -> bool:
    """Check if data can be safely serialized as JSON.

    Args:
        data: Data to check

    Returns:
        True if data is JSON-serializable, False otherwise
    """
    try:
        serialize_json(data)
        return True
    except (TypeError, ValueError):
        return False


# Migration utilities


def migrate_pickle_to_json(pickle_file: Path, json_file: Optional[Path] = None) -> bool:
    """Migrate a pickle file to JSON format.

    Args:
        pickle_file: Path to pickle file
        json_file: Path to output JSON file (defaults to pickle_file with .json extension)

    Returns:
        True if migration successful, False otherwise

    Note:
        This uses secure_pickle for loading pickle data to maintain security
    """
    try:
        from apgi_framework.security.secure_pickle import SecurePickleValidator

        if json_file is None:
            json_file = pickle_file.with_suffix(".json")

        # Load pickle data securely
        data = SecurePickleValidator.load(pickle_file, trusted=True)  # type: ignore[attr-defined]

        # Save as JSON
        serialize_json_to_file(data, json_file)

        return True
    except Exception:
        # Migration failed - data may not be JSON-serializable
        return False


def auto_detect_and_load(file_path: Path) -> Any:
    """Load data from file with automatic format detection.

    Supports: .json, .msgpack, .pkl (with secure_pickle)

    Args:
        file_path: Path to data file

    Returns:
        Deserialized data
    """
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        return deserialize_json_from_file(file_path)
    elif suffix in (".msgpack", ".mp"):
        try:
            import msgpack  # type: ignore

            with open(file_path, "rb") as f:
                return msgpack.unpackb(f.read(), raw=False)
        except ImportError:
            raise ImportError(
                "msgpack is required for .msgpack files. " "Install with: pip install msgpack"
            )
    elif suffix == ".pkl":
        import logging

        from apgi_framework.security.secure_pickle import SecurePickleValidator

        logging.warning(f"Loading legacy pickle file: {file_path}")
        return SecurePickleValidator.load(file_path, trusted=True)  # type: ignore[attr-defined]
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
