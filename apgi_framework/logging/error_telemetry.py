"""Error Telemetry for APGI Framework.

Provides basic error reporting and telemetry collection for debugging and
improving the framework.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .standardized_logging import get_logger

logger = get_logger(__name__)


class ErrorTelemetry:
    """Basic error telemetry and reporting system."""

    def __init__(self, telemetry_dir: str = "logs/telemetry"):
        """Initialize error telemetry.

        Args:
            telemetry_dir: Directory to store telemetry data
        """
        self.telemetry_dir = Path(telemetry_dir)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_file = self.telemetry_dir / "error_telemetry.json"

        # Load existing telemetry
        self._load_telemetry()

        # Create initial telemetry file if it doesn't exist
        self._save_telemetry()

    def _load_telemetry(self) -> None:
        """Load existing telemetry data."""
        if self.telemetry_file.exists():
            try:
                with open(self.telemetry_file, "r") as f:
                    self.telemetry_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load telemetry data: {e}")
                self.telemetry_data = {"errors": [], "system_info": {}}
        else:
            self.telemetry_data = {"errors": [], "system_info": {}}

    def _save_telemetry(self) -> None:
        """Save telemetry data to file."""
        try:
            with open(self.telemetry_file, "w") as f:
                json.dump(self.telemetry_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save telemetry data: {e}")

    def report_error(
        self,
        error_type: str,
        error_message: str,
        traceback: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Report an error for telemetry collection.

        Args:
            error_type: Type of error (e.g., 'ValueError', 'NetworkError')
            error_message: Error message
            traceback: Full traceback if available
            context: Additional context information
            user_id: Anonymous user identifier (optional)
        """
        # Sanitize context to prevent serialization issues
        sanitized_context = self._sanitize_for_json(context or {}, max_depth=3, max_items=10)
        sanitized_system_info = self._sanitize_for_json(
            self._get_system_info(), max_depth=2, max_items=5
        )

        error_report = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback,
            "context": sanitized_context,
            "user_id": user_id,
            "system_info": sanitized_system_info,
        }

        self.telemetry_data["errors"].append(error_report)

        # Keep only last 1000 errors
        if len(self.telemetry_data["errors"]) > 1000:
            self.telemetry_data["errors"] = self.telemetry_data["errors"][-1000:]

        self._save_telemetry()
        logger.debug(f"Error reported: {error_type} - {error_message}")

    def _sanitize_for_json(
        self, data: Any, max_depth: int = 3, max_items: int = 10, current_depth: int = 0
    ) -> Any:
        """Sanitize data for JSON serialization by limiting depth and size."""
        if current_depth >= max_depth:
            return str(data)[:100] + "..." if len(str(data)) > 100 else str(data)

        if isinstance(data, dict):
            sanitized_dict: Dict[str, Any] = {}
            for i, (key, value) in enumerate(data.items()):
                if i >= max_items:
                    sanitized_dict["...truncated"] = f"{len(data) - max_items} more items"
                    break
                sanitized_dict[str(key)[:50]] = self._sanitize_for_json(
                    value, max_depth, max_items, current_depth + 1
                )
            return sanitized_dict

        elif isinstance(data, (list, tuple)):
            sanitized_list: List[Any] = []
            for i, item in enumerate(data):
                if i >= max_items:
                    sanitized_list.append(f"...{len(data) - max_items} more items")
                    break
                sanitized_list.append(
                    self._sanitize_for_json(item, max_depth, max_items, current_depth + 1)
                )
            return sanitized_list

        elif isinstance(data, (int, float, bool, type(None))):
            return data

        else:
            # Convert other types to string representation
            str_repr = str(data)
            return str_repr[:200] + "..." if len(str_repr) > 200 else str_repr

    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information for telemetry."""
        try:
            import platform
            import sys

            return {
                "platform": platform.platform(),
                "python_version": sys.version,
                "architecture": platform.architecture(),
                "processor": platform.processor(),
            }
        except Exception:
            return {"error": "Failed to get system info"}

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of collected errors."""
        errors = self.telemetry_data.get("errors", [])

        if not errors:
            return {"total_errors": 0, "error_types": {}, "recent_errors": []}

        # Count error types
        error_types: Dict[str, int] = {}
        for error in errors:
            error_type = error.get("error_type", "Unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1

        # Get recent errors (last 10)
        recent_errors = errors[-10:] if len(errors) >= 10 else errors

        return {
            "total_errors": len(errors),
            "error_types": error_types,
            "recent_errors": recent_errors,
            "first_error": errors[0]["timestamp"] if errors else None,
            "last_error": errors[-1]["timestamp"] if errors else None,
        }

    def export_telemetry(self, filepath: str) -> None:
        """Export telemetry data to a file."""
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "summary": self.get_error_summary(),
                "full_data": self.telemetry_data,
            }

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Telemetry exported to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export telemetry: {e}")

    def clear_telemetry(self) -> None:
        """Clear all telemetry data."""
        self.telemetry_data = {"errors": [], "system_info": {}}
        self._save_telemetry()
        logger.info("Telemetry data cleared")


# Global telemetry instance
_telemetry: Optional[ErrorTelemetry] = None


def get_error_telemetry() -> ErrorTelemetry:
    """Get or create the global error telemetry instance."""
    global _telemetry
    if _telemetry is None:
        _telemetry = ErrorTelemetry()
    return _telemetry


def report_error(
    error_type: str,
    error_message: str,
    traceback: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> None:
    """Convenience function to report an error.

    Args:
        error_type: Type of error
        error_message: Error message
        traceback: Full traceback
        context: Additional context
        user_id: Anonymous user identifier
    """
    telemetry = get_error_telemetry()
    telemetry.report_error(error_type, error_message, traceback, context, user_id)


def enable_error_reporting(enable: bool = True) -> None:
    """Enable or disable automatic error reporting.

    Args:
        enable: Whether to enable error reporting
    """
    global _telemetry
    if enable and _telemetry is None:
        _telemetry = ErrorTelemetry()
    elif not enable:
        _telemetry = None

    logger.info(f"Error reporting {'enabled' if enable else 'disabled'}")
