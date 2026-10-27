"""CCPA (California Consumer Privacy Act) Compliance Framework for APGI.

Handles California consumer rights including:
- Right to Know (what personal information is collected)
- Right to Delete (deletion of personal information)
- Right to Opt-Out (of sale of personal information)
- Right to Non-Discrimination
- Right to Access (portable copy of personal information)
- Notice at Collection requirements
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from apgi_framework.logging.audit_logger import get_audit_logger

logger = logging.getLogger("ccpa")


class CCPARights:
    """CCPA consumer rights enumeration."""

    RIGHT_TO_KNOW = "right_to_know"
    RIGHT_TO_DELETE = "right_to_delete"
    RIGHT_TO_OPT_OUT = "right_to_opt_out"
    RIGHT_TO_ACCESS = "right_to_access"
    RIGHT_TO_NON_DISCRIMINATION = "right_to_non_discrimination"


class CCPAManager:
    """Manages CCPA compliance workflows for California consumers.

    Implements:
    - Consumer request handling (know, delete, access)
    - Opt-out preference management
    - Notice at Collection tracking
    - Data sale tracking and opt-out enforcement
    - Verification workflows for consumer requests
    """

    def __init__(
        self,
        data_dir: str = "data/consumers",
        consent_dir: str = "data/ccpa_consent",
        requests_dir: str = "data/ccpa_requests",
    ):
        """Initialize CCPA manager.

        Args:
            data_dir: Directory containing consumer data
            consent_dir: Directory for opt-out preferences
            requests_dir: Directory for request tracking
        """
        self.data_dir = Path(data_dir)
        self.consent_dir = Path(consent_dir)
        self.requests_dir = Path(requests_dir)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.consent_dir.mkdir(parents=True, exist_ok=True)
        self.requests_dir.mkdir(parents=True, exist_ok=True)

        # Categories of personal information under CCPA
        self.personal_info_categories = {
            "identifiers": ["name", "email", "phone", "address", "ip_address", "user_id"],
            "commercial": ["purchase_history", "payment_info", "transaction_data"],
            "biometric": ["health_data", "physiological_data", "behavioral_data"],
            "internet": ["browsing_history", "search_history", "interaction_data"],
            "geolocation": ["location_data", "gps_data", "zip_code"],
            "sensory": ["audio", "video", "photos"],
            "employment": ["job_title", "company", "professional_info"],
            "education": ["education_level", "institution", "degrees"],
            "inferences": ["psychometric_profiles", "simulation_results", "apgi_scores"],
        }

        # Third parties data may be shared with
        self.third_parties: Dict[str, List[str]] = {
            "analytics": ["Google Analytics", "Mixpanel"],
            "hosting": ["AWS", "Heroku"],
            "research": ["Academic Partners", "Clinical Research Orgs"],
        }

    def record_notice_at_collection(
        self,
        consumer_id: str,
        categories_collected: List[str],
        collection_purposes: List[str],
    ) -> Path:
        """Record Notice at Collection for a consumer.

        Required at or before point of collection under CCPA § 999.305.

        Args:
            consumer_id: Consumer identifier
            categories_collected: Categories of personal information collected
            collection_purposes: Business/commercial purposes for collection

        Returns:
            Path to the recorded notice file
        """
        notice_file = self.consent_dir / f"{consumer_id}_notice.json"

        notice_record = {
            "consumer_id": consumer_id,
            "notice_type": "at_collection",
            "timestamp": datetime.now().isoformat(),
            "categories_collected": categories_collected,
            "purposes": collection_purposes,
            "third_parties": self.third_parties,
            "retention_periods": self._get_retention_periods(categories_collected),
            "privacy_policy_url": "https://apgi-system.example.com/privacy",
        }

        with open(notice_file, "w") as f:
            json.dump(notice_record, f, indent=2)

        get_audit_logger().log_event(
            user_id="SYSTEM",
            event_type="CCPA_COMPLIANCE",
            resource_id=consumer_id,
            action="NOTICE_AT_COLLECTION",
            details={
                "categories": categories_collected,
                "purposes": collection_purposes,
            },
        )

        return notice_file

    def record_opt_out(self, consumer_id: str, opt_out_type: str) -> Path:
        """Record consumer opt-out preference.

        Args:
            consumer_id: Consumer identifier
            opt_out_type: Type of opt-out ("sale", "sharing", "all")

        Returns:
            Path to the opt-out record
        """
        opt_out_file = self.consent_dir / f"{consumer_id}_opt_out.json"

        opt_out_record = {
            "consumer_id": consumer_id,
            "opt_out_type": opt_out_type,
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "verified": False,  # Requires verification under CCPA
        }

        with open(opt_out_file, "w") as f:
            json.dump(opt_out_record, f, indent=2)

        get_audit_logger().log_event(
            user_id="SYSTEM",
            event_type="CCPA_COMPLIANCE",
            resource_id=consumer_id,
            action="OPT_OUT_RECORDED",
            details={"opt_out_type": opt_out_type},
        )

        logger.info(f"Recorded {opt_out_type} opt-out for consumer {consumer_id}")
        return opt_out_file

    def check_opt_out(self, consumer_id: str, data_sale_type: str) -> bool:
        """Check if consumer has opted out of data sale/sharing.

        Args:
            consumer_id: Consumer identifier
            data_sale_type: Type of data sharing being checked

        Returns:
            True if opted out, False otherwise
        """
        opt_out_file = self.consent_dir / f"{consumer_id}_opt_out.json"

        if not opt_out_file.exists():
            return False

        try:
            with open(opt_out_file, "r") as f:
                opt_out = json.load(f)

            if opt_out.get("status") != "active":
                return False

            # Check if this opt-out applies
            opt_type = opt_out.get("opt_out_type", "")
            return opt_type in [data_sale_type, "all", "sale"]

        except Exception as e:
            logger.error(f"Error checking opt-out for {consumer_id}: {e}")
            return False

    def get_consumer_data_categories(self, consumer_id: str) -> Dict[str, List[str]]:
        """Get all categories of personal information for a consumer.

        Implements Right to Know - categories of information collected.

        Args:
            consumer_id: Consumer identifier

        Returns:
            Dict mapping category names to list of data fields found
        """
        consumer_files = list(self.data_dir.glob(f"{consumer_id}*"))

        found_categories: Dict[str, Set[str]] = {
            cat: set() for cat in self.personal_info_categories.keys()
        }

        for file in consumer_files:
            try:
                if file.suffix == ".json":
                    with open(file, "r") as f:
                        data = json.load(f)

                    # Check each category
                    for category, fields in self.personal_info_categories.items():
                        for field in fields:
                            if self._has_field(data, field):
                                found_categories[category].add(field)

            except Exception as e:
                logger.error(f"Error reading {file}: {e}")

        # Convert sets to sorted lists
        return {cat: sorted(list(fields)) for cat, fields in found_categories.items() if fields}

    def export_consumer_data(self, consumer_id: str) -> Optional[Path]:
        """Export all consumer data in portable format.

        Implements Right to Access (portable copy).

        Args:
            consumer_id: Consumer identifier

        Returns:
            Path to export file or None if no data found
        """
        consumer_files = list(self.data_dir.glob(f"{consumer_id}*"))
        if not consumer_files:
            return None

        export_dir = Path("exports/ccpa")
        export_dir.mkdir(parents=True, exist_ok=True)
        export_file = (
            export_dir / f"ccpa_access_{consumer_id}_{datetime.now().strftime('%Y%m%d')}.json"
        )

        all_data: Dict[str, Any] = {
            "consumer_id": consumer_id,
            "export_timestamp": datetime.now().isoformat(),
            "categories": self.get_consumer_data_categories(consumer_id),
            "data_sources": [],
            "third_parties_shared_with": self.third_parties,
            "files": {},
        }

        for file in consumer_files:
            try:
                if file.suffix == ".json":
                    with open(file, "r") as f:
                        all_data["files"][file.name] = json.load(f)
                    all_data["data_sources"].append(str(file))
            except Exception as e:
                logger.error(f"Error reading {file} for export: {e}")

        with open(export_file, "w") as f:
            json.dump(all_data, f, indent=2)

        get_audit_logger().log_event(
            user_id="SYSTEM",
            event_type="CCPA_COMPLIANCE",
            resource_id=consumer_id,
            action="RIGHT_TO_ACCESS_EXPORT",
            details={"export_path": str(export_file)},
        )

        return export_file

    def delete_consumer_data(self, consumer_id: str) -> Dict[str, Any]:
        """Delete all personal information for a consumer.

        Implements Right to Delete with verification tracking.

        Args:
            consumer_id: Consumer identifier

        Returns:
            Dict with deletion results
        """
        deleted_files: List[str] = []
        failed_deletions: List[Dict[str, str]] = []
        retained_files: List[str] = []

        results: Dict[str, Any] = {
            "consumer_id": consumer_id,
            "deleted_files": deleted_files,
            "failed_deletions": failed_deletions,
            "retained_files": retained_files,
            "timestamp": datetime.now().isoformat(),
        }

        consumer_files = list(self.data_dir.glob(f"{consumer_id}*"))

        for file in consumer_files:
            try:
                if file.is_dir():
                    shutil.rmtree(file)
                else:
                    file.unlink()
                results["deleted_files"].append(str(file))
            except Exception as e:
                logger.error(f"Failed to delete {file}: {e}")
                results["failed_deletions"].append({"file": str(file), "error": str(e)})

        # Note: Some data may need to be retained for legal/security purposes
        # CCPA allows exceptions for security, fraud prevention, legal compliance
        results["retention_exceptions"] = [
            "Security incident records (fraud prevention)",
            "Legal compliance records",
            "Aggregated/anonymized research data",
        ]

        get_audit_logger().log_event(
            user_id="SYSTEM",
            event_type="CCPA_COMPLIANCE",
            resource_id=consumer_id,
            action="RIGHT_TO_DELETE",
            details={
                "files_deleted": len(results["deleted_files"]),
                "files_failed": len(results["failed_deletions"]),
            },
        )

        return results

    def record_consumer_request(
        self,
        consumer_id: str,
        request_type: str,
        request_details: Optional[Dict] = None,
    ) -> str:
        """Record a consumer rights request.

        Args:
            consumer_id: Consumer identifier
            request_type: Type of CCPA right being exercised
            request_details: Additional request information

        Returns:
            Request ID for tracking
        """
        request_id = f"CCPA-{consumer_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        request_file = self.requests_dir / f"{request_id}.json"

        request_record = {
            "request_id": request_id,
            "consumer_id": consumer_id,
            "request_type": request_type,
            "status": "received",
            "received_at": datetime.now().isoformat(),
            "details": request_details or {},
            "response_deadline": (
                datetime.now() + __import__("datetime").timedelta(days=45)
            ).isoformat(),
        }

        with open(request_file, "w") as f:
            json.dump(request_record, f, indent=2)

        get_audit_logger().log_event(
            user_id="SYSTEM",
            event_type="CCPA_COMPLIANCE",
            resource_id=consumer_id,
            action="CONSUMER_REQUEST_RECEIVED",
            details={
                "request_id": request_id,
                "request_type": request_type,
            },
        )

        return request_id

    def update_request_status(
        self, request_id: str, status: str, notes: Optional[str] = None
    ) -> bool:
        """Update the status of a consumer request.

        Args:
            request_id: The request ID
            status: New status (received, processing, completed, denied)
            notes: Optional notes

        Returns:
            True if updated, False if not found
        """
        request_file = self.requests_dir / f"{request_id}.json"

        if not request_file.exists():
            return False

        try:
            with open(request_file, "r") as f:
                record = json.load(f)

            record["status"] = status
            record["updated_at"] = datetime.now().isoformat()
            if notes:
                record["notes"] = notes

            if status == "completed":
                record["completed_at"] = datetime.now().isoformat()

            with open(request_file, "w") as f:
                json.dump(record, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Error updating request {request_id}: {e}")
            return False

    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get all pending consumer requests."""
        pending = []

        for request_file in self.requests_dir.glob("*.json"):
            try:
                with open(request_file, "r") as f:
                    record = json.load(f)
                if record.get("status") in ["received", "processing"]:
                    pending.append(record)
            except Exception:
                continue

        return pending

    def generate_privacy_report(self, consumer_id: str) -> Dict[str, Any]:
        """Generate comprehensive privacy report for a consumer.

        Combines all CCPA-related information for the consumer.
        """
        categories = self.get_consumer_data_categories(consumer_id)

        # Count total fields across all categories
        total_fields = sum(len(fields) for fields in categories.values())

        return {
            "consumer_id": consumer_id,
            "generated_at": datetime.now().isoformat(),
            "personal_information": {
                "categories_found": list(categories.keys()),
                "total_fields": total_fields,
                "field_breakdown": categories,
            },
            "third_parties": self.third_parties,
            "opt_out_status": {
                "has_opted_out": self.check_opt_out(consumer_id, "all"),
                "opt_out_types": (
                    ["sale", "sharing"] if self.check_opt_out(consumer_id, "sale") else []
                ),
            },
            "consumer_rights": {
                "right_to_know": True,
                "right_to_delete": True,
                "right_to_opt_out": True,
                "right_to_non_discrimination": True,
                "right_to_access": True,
            },
        }

    def _has_field(self, data: Any, field_name: str) -> bool:
        """Recursively check if a field exists in nested data."""
        if isinstance(data, dict):
            if field_name in data:
                return True
            return any(self._has_field(v, field_name) for v in data.values())
        elif isinstance(data, list):
            return any(self._has_field(item, field_name) for item in data)
        return False

    def _get_retention_periods(self, categories: List[str]) -> Dict[str, str]:
        """Get retention periods for data categories."""
        periods = {
            "identifiers": "Duration of relationship + 7 years",
            "commercial": "7 years (tax/compliance)",
            "biometric": "Duration of study + 10 years",
            "internet": "13 months",
            "geolocation": "13 months",
            "sensory": "Duration of study",
            "employment": "Duration of relationship",
            "education": "Duration of relationship",
            "inferences": "Duration of study + 5 years",
        }
        return {cat: periods.get(cat, "Varies by purpose") for cat in categories}


# Global CCPA manager instance
_ccpa_manager: Optional[CCPAManager] = None


def get_ccpa_manager() -> CCPAManager:
    """Get or create global CCPA manager instance."""
    global _ccpa_manager
    if _ccpa_manager is None:
        _ccpa_manager = CCPAManager()
    return _ccpa_manager
