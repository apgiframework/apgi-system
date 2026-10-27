"""GDPR Compliance Framework for APGI.

Handles data subject right requests (access, deletion, portability),
consent management, and privacy by design assessments.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from apgi_framework.logging.audit_logger import get_audit_logger

logger = logging.getLogger("gdpr")


class GDPRManager:
    """Manages GDPR compliance workflows."""

    def __init__(self, data_dir: str = "data/participants", consent_dir: str = "data/consent"):
        self.data_dir = Path(data_dir)
        self.consent_dir = Path(consent_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.consent_dir.mkdir(parents=True, exist_ok=True)

    def record_consent(self, participant_id: str, consent_data: Dict[str, Any]) -> None:
        """Record participant consent."""
        consent_file = self.consent_dir / f"{participant_id}_consent.json"

        consent_record = {
            "participant_id": participant_id,
            "timestamp": datetime.now().isoformat(),
            "consent_data": consent_data,
            "version": "1.0",
        }

        with open(consent_file, "w") as f:
            json.dump(consent_record, f, indent=2)

        get_audit_logger().log_event(
            user_id="SYSTEM",
            event_type="CONSENT_MANAGEMENT",
            resource_id=participant_id,
            action="RECORD_CONSENT",
            details={"consent_version": "1.0"},
        )

    def export_data_portability(self, participant_id: str) -> Optional[Path]:
        """Export all data for a participant in a portable format (JSON)."""
        participant_files = list(self.data_dir.glob(f"{participant_id}*"))
        if not participant_files:
            return None

        export_dir = Path("exports/portability")
        export_dir.mkdir(parents=True, exist_ok=True)
        export_file = (
            export_dir / f"gdpr_export_{participant_id}_{datetime.now().strftime('%Y%m%d')}.json"
        )

        all_data = {}
        for file in participant_files:
            try:
                if file.suffix == ".json":
                    with open(file, "r") as f:
                        all_data[file.name] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read {file} for export: {e}")

        with open(export_file, "w") as f:
            json.dump(all_data, f, indent=2)

        get_audit_logger().log_event(
            user_id="SYSTEM",
            event_type="DATA_SUBJECT_RIGHT",
            resource_id=participant_id,
            action="DATA_PORTABILITY_EXPORT",
            details={"export_path": str(export_file)},
        )

        return export_file

    def delete_participant_data(self, participant_id: str) -> bool:
        """Delete all data for a participant (Right to Erasure)."""
        participant_files = list(self.data_dir.glob(f"{participant_id}*"))
        consent_files = list(self.consent_dir.glob(f"{participant_id}*"))

        all_files = participant_files + consent_files
        if not all_files:
            return False

        for file in all_files:
            try:
                if file.is_dir():
                    shutil.rmtree(file)
                else:
                    file.unlink()
            except Exception as e:
                logger.error(f"Failed to delete {file}: {e}")
                return False

        get_audit_logger().log_event(
            user_id="SYSTEM",
            event_type="DATA_SUBJECT_RIGHT",
            resource_id=participant_id,
            action="RIGHT_TO_ERASURE",
            details={"files_deleted": len(all_files)},
        )

        return True

    def perform_dpia(self, project_name: str, assessment_data: Dict[str, Any]) -> Path:
        """Record a Data Protection Impact Assessment (DPIA)."""
        dpia_dir = Path("docs/compliance/dpia")
        dpia_dir.mkdir(parents=True, exist_ok=True)
        dpia_file = dpia_dir / f"dpia_{project_name}_{datetime.now().strftime('%Y%m%d')}.json"

        dpia_record = {
            "project_name": project_name,
            "timestamp": datetime.now().isoformat(),
            "assessment": assessment_data,
            "status": "completed",
        }

        with open(dpia_file, "w") as f:
            json.dump(dpia_record, f, indent=2)

        return dpia_file


# Global GDPR manager instance
_gdpr_manager: Optional[GDPRManager] = None


def get_gdpr_manager() -> GDPRManager:
    global _gdpr_manager
    if _gdpr_manager is None:
        _gdpr_manager = GDPRManager()
    return _gdpr_manager
