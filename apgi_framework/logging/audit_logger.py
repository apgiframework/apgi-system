"""Comprehensive Audit Logging System for APGI.

Provides immutable, tamper-evident logging of sensitive operations,
user actions, and configuration changes.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from apgi_framework.security.pii_protector import get_pii_protector

logger = logging.getLogger("audit")


class AuditLogger:
    """Handles secure audit logging for sensitive system operations."""

    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m')}.log"
        self._last_hash: str = self._load_last_hash()

    def _load_last_hash(self) -> str:
        """Load the hash of the last log entry for chain integrity."""
        hash_file = self.log_dir / ".last_hash"
        if hash_file.exists():
            return hash_file.read_text().strip()
        return "0" * 64

    def _save_last_hash(self, new_hash: str) -> None:
        """Save the hash of the latest log entry."""
        hash_file = self.log_dir / ".last_hash"
        hash_file.write_text(new_hash)
        self._last_hash = new_hash

    def log_event(
        self,
        user_id: str,
        event_type: str,
        resource_id: str,
        action: str,
        details: Dict[str, Any],
        severity: str = "INFO",
    ) -> str:
        """Log a sensitive event to the audit trail.

        Args:
            user_id: ID of the user performing the action
            event_type: Category of event (e.g., 'DATA_ACCESS', 'CONFIG_CHANGE')
            resource_id: ID of the affected resource (e.g., session ID)
            action: Specific action performed
            details: Additional context and data
            severity: Event severity level

        Returns:
            Hash of the log entry for verification
        """
        timestamp = datetime.now().isoformat()

        # Build entry
        entry = {
            "timestamp": timestamp,
            "user_id": user_id,
            "event_type": event_type,
            "resource_id": resource_id,
            "action": action,
            "details": details,
            "severity": severity,
            "previous_hash": self._last_hash,
        }

        # Redact PII from details
        entry["details"] = get_pii_protector().redact_log_record({"details": entry["details"]})

        entry_json = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_json.encode()).hexdigest()

        # Final entry with its own hash
        final_entry = entry.copy()
        final_entry["hash"] = entry_hash

        # Append to log file
        with open(self.current_log_file, "a") as f:
            f.write(json.dumps(final_entry) + "\n")

        self._save_last_hash(entry_hash)

        logger.info(f"Audit event logged: {event_type} - {action} by {user_id}")
        return entry_hash

    def verify_integrity(self) -> bool:
        """Verify the integrity of the audit log chain.

        Returns:
            True if integrity is intact, False if tampering detected.
        """
        if not self.current_log_file.exists():
            return True

        expected_prev_hash = "0" * 64

        try:
            with open(self.current_log_file, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    actual_hash = entry.pop("hash")

                    # Verify previous hash chain
                    if entry["previous_hash"] != expected_prev_hash:
                        logger.error(
                            f"Integrity check failed: Hash chain broken at {entry['timestamp']}"
                        )
                        return False

                    # Verify entry hash
                    entry_json = json.dumps(entry, sort_keys=True)
                    calculated_hash = hashlib.sha256(entry_json.encode()).hexdigest()

                    if actual_hash != calculated_hash:
                        logger.error(
                            f"Integrity check failed: Hash mismatch at {entry['timestamp']}"
                        )
                        return False

                    expected_prev_hash = actual_hash

            return True
        except Exception as e:
            logger.error(f"Integrity check error: {e}")
            return False

    def query_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query the audit logs with filters."""
        results: List[Dict[str, Any]] = []
        if not self.current_log_file.exists():
            return results

        with open(self.current_log_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                entry_time = datetime.fromisoformat(entry["timestamp"])

                if start_time and entry_time < start_time:
                    continue
                if end_time and entry_time > end_time:
                    continue
                if user_id and entry["user_id"] != user_id:
                    continue
                if event_type and entry["event_type"] != event_type:
                    continue

                results.append(entry)

        return results


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
