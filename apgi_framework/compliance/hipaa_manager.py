"""HIPAA Compliance Framework for APGI.

Handles Protected Health Information (PHI) identification,
access controls, and audit trails for healthcare data.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from apgi_framework.logging.audit_logger import get_audit_logger

logger = logging.getLogger("hipaa")


class HIPAAManager:
    """Manages HIPAA compliance workflows."""

    def __init__(self, phi_log_dir: str = "logs/hipaa"):
        self.phi_log_dir = Path(phi_log_dir)
        self.phi_log_dir.mkdir(parents=True, exist_ok=True)

    def identify_phi(self, data: Dict[str, Any]) -> List[str]:
        """Identify PHI fields in a data dictionary."""
        phi_indicators = {
            "name",
            "address",
            "birth_date",
            "phone",
            "email",
            "ssn",
            "medical_record_number",
            "health_plan_id",
            "biometric_id",
            "account_number",
            "certificate_number",
            "license_number",
            "vehicle_id",
            "device_id",
            "web_url",
            "ip_address",
        }

        found_phi = []
        for key in data:
            if key.lower() in phi_indicators:
                found_phi.append(key)
        return found_phi

    def log_phi_access(
        self, user_id: str, participant_id: str, fields_accessed: List[str], reason: str
    ) -> None:
        """Log access to PHI (Required by HIPAA)."""
        get_audit_logger().log_event(
            user_id=user_id,
            event_type="PHI_ACCESS",
            resource_id=participant_id,
            action="ACCESS_PHI",
            details={"fields": fields_accessed, "reason": reason, "compliance_standard": "HIPAA"},
            severity="MEDIUM",
        )

    def manage_baa(self, associate_name: str, status: str = "signed") -> None:
        """Manage Business Associate Agreements (BAA)."""
        baa_dir = Path("docs/compliance/baa")
        baa_dir.mkdir(parents=True, exist_ok=True)
        baa_file = baa_dir / f"baa_{associate_name}_{datetime.now().strftime('%Y%m%d')}.json"

        baa_record = {
            "associate_name": associate_name,
            "status": status,
            "last_reviewed": datetime.now().isoformat(),
            "compliance_checked": True,
        }

        with open(baa_file, "w") as f:
            json.dump(baa_record, f, indent=2)

        get_audit_logger().log_event(
            user_id="SYSTEM",
            event_type="COMPLIANCE_MANAGEMENT",
            resource_id=associate_name,
            action="UPDATE_BAA",
            details={"status": status},
        )


# Global HIPAA manager instance
_hipaa_manager: Optional[HIPAAManager] = None


def get_hipaa_manager() -> HIPAAManager:
    global _hipaa_manager
    if _hipaa_manager is None:
        _hipaa_manager = HIPAAManager()
    return _hipaa_manager
