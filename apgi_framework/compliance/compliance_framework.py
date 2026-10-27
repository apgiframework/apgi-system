"""Compliance Framework for APGI Framework.

This module provides regulatory/privacy compliance controls including:
- Data classification system
- Retention policy enforcement
- Audit logging
- Consent lineage tracking
- Data minimization controls
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data sensitivity classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"  # Personally Identifiable Information
    PHI = "phi"  # Protected Health Information
    SENSITIVE_RESEARCH = "sensitive_research"


class RetentionCategory(Enum):
    """Data retention categories with default retention periods."""

    TRANSIENT = timedelta(days=7)
    SHORT_TERM = timedelta(days=90)
    MEDIUM_TERM = timedelta(days=365)
    LONG_TERM = timedelta(days=365 * 7)
    PERMANENT = timedelta(days=365 * 100)


@dataclass
class DataClassificationRule:
    """Rule for classifying data based on content."""

    classification: DataClassification
    retention_period: timedelta
    requires_encryption: bool
    requires_consent: bool
    access_log_required: bool
    minimization_required: bool


@dataclass
class ConsentRecord:
    """Record of participant consent for data usage."""

    consent_id: str
    participant_id: str
    consent_type: str
    consent_date: datetime
    expiry_date: Optional[datetime]
    data_types: List[str]
    purposes: List[str]
    is_active: bool = True
    version: str = "1.0"


@dataclass
class AuditLogEntry:
    """Audit log entry for compliance tracking."""

    timestamp: datetime
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    classification: DataClassification
    outcome: str
    details: Dict[str, Any] = field(default_factory=dict)


class ComplianceFramework:
    """Main compliance enforcement framework.

    Provides centralized compliance controls for data classification,
    retention, consent management, and audit logging.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize compliance framework.

        Args:
            config: Configuration parameters for compliance rules
        """
        self.config = config or self._get_default_config()
        self.classification_rules = self._load_classification_rules()
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.audit_log: List[AuditLogEntry] = []
        self.data_retention_schedule: Dict[str, datetime] = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default compliance configuration."""
        return {
            "default_retention": "MEDIUM_TERM",
            "audit_log_retention": "LONG_TERM",
            "encryption_required_for": [
                "CONFIDENTIAL",
                "RESTRICTED",
                "PII",
                "PHI",
                "SENSITIVE_RESEARCH",
            ],
            "consent_required_for": ["PII", "PHI", "SENSITIVE_RESEARCH"],
            "minimization_required_for": ["PII", "PHI"],
            "audit_all_access": True,
        }

    def _load_classification_rules(
        self,
    ) -> Dict[DataClassification, DataClassificationRule]:
        """Load data classification rules."""
        return {
            DataClassification.PUBLIC: DataClassificationRule(
                classification=DataClassification.PUBLIC,
                retention_period=RetentionCategory.PERMANENT.value,
                requires_encryption=False,
                requires_consent=False,
                access_log_required=False,
                minimization_required=False,
            ),
            DataClassification.INTERNAL: DataClassificationRule(
                classification=DataClassification.INTERNAL,
                retention_period=RetentionCategory.LONG_TERM.value,
                requires_encryption=False,
                requires_consent=False,
                access_log_required=True,
                minimization_required=False,
            ),
            DataClassification.CONFIDENTIAL: DataClassificationRule(
                classification=DataClassification.CONFIDENTIAL,
                retention_period=RetentionCategory.MEDIUM_TERM.value,
                requires_encryption=True,
                requires_consent=False,
                access_log_required=True,
                minimization_required=False,
            ),
            DataClassification.RESTRICTED: DataClassificationRule(
                classification=DataClassification.RESTRICTED,
                retention_period=RetentionCategory.SHORT_TERM.value,
                requires_encryption=True,
                requires_consent=True,
                access_log_required=True,
                minimization_required=True,
            ),
            DataClassification.PII: DataClassificationRule(
                classification=DataClassification.PII,
                retention_period=RetentionCategory.MEDIUM_TERM.value,
                requires_encryption=True,
                requires_consent=True,
                access_log_required=True,
                minimization_required=True,
            ),
            DataClassification.PHI: DataClassificationRule(
                classification=DataClassification.PHI,
                retention_period=RetentionCategory.LONG_TERM.value,
                requires_encryption=True,
                requires_consent=True,
                access_log_required=True,
                minimization_required=True,
            ),
            DataClassification.SENSITIVE_RESEARCH: DataClassificationRule(
                classification=DataClassification.SENSITIVE_RESEARCH,
                retention_period=RetentionCategory.LONG_TERM.value,
                requires_encryption=True,
                requires_consent=True,
                access_log_required=True,
                minimization_required=False,
            ),
        }

    def classify_data(
        self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> DataClassification:
        """Classify data based on content and context.

        Args:
            data: Data to classify
            context: Additional context for classification

        Returns:
            Data classification level
        """
        # Check for PII indicators
        pii_indicators = ["name", "email", "phone", "address", "ssn", "social_security"]
        if any(indicator in str(data).lower() for indicator in pii_indicators):
            return DataClassification.PII

        # Check for PHI indicators
        phi_indicators = ["medical", "health", "diagnosis", "treatment", "patient"]
        if any(indicator in str(data).lower() for indicator in phi_indicators):
            return DataClassification.PHI

        # Check for sensitive research data
        if context and context.get("research_type") in [
            "clinical",
            "biomarker",
            "genetic",
        ]:
            return DataClassification.SENSITIVE_RESEARCH

        # Default classification
        return DataClassification.INTERNAL

    def record_consent(
        self,
        participant_id: str,
        consent_type: str,
        data_types: List[str],
        purposes: List[str],
        expiry_date: Optional[datetime] = None,
    ) -> ConsentRecord:
        """Record participant consent for data usage.

        Args:
            participant_id: Participant identifier
            consent_type: Type of consent (e.g., "data_collection", "research_participation")
            data_types: Types of data consented to
            purposes: Purposes for which data can be used
            expiry_date: Optional expiry date for consent

        Returns:
            Consent record
        """
        consent_id = self._generate_consent_id(participant_id, consent_type)
        consent_record = ConsentRecord(
            consent_id=consent_id,
            participant_id=participant_id,
            consent_type=consent_type,
            consent_date=datetime.now(),
            expiry_date=expiry_date,
            data_types=data_types,
            purposes=purposes,
            is_active=True,
        )
        self.consent_records[consent_id] = consent_record
        self._log_audit(
            user_id="system",
            action="consent_recorded",
            resource_type="consent",
            resource_id=consent_id,
            classification=DataClassification.RESTRICTED,
            outcome="success",
            details={
                "participant_id": participant_id,
                "consent_type": consent_type,
                "data_types": data_types,
            },
        )
        return consent_record

    def check_consent(self, participant_id: str, data_type: str, purpose: str) -> bool:
        """Check if valid consent exists for data usage.

        Args:
            participant_id: Participant identifier
            data_type: Type of data to be used
            purpose: Purpose for data usage

        Returns:
            True if valid consent exists, False otherwise
        """
        now = datetime.now()
        for consent in self.consent_records.values():
            if (
                consent.participant_id == participant_id
                and consent.is_active
                and data_type in consent.data_types
                and purpose in consent.purposes
            ):
                # Check expiry
                if consent.expiry_date and consent.expiry_date < now:
                    consent.is_active = False
                    continue
                return True
        return False

    def set_retention_schedule(
        self, resource_id: str, classification: DataClassification
    ) -> datetime:
        """Set retention schedule for a resource based on classification.

        Args:
            resource_id: Resource identifier
            classification: Data classification

        Returns:
            Expiry date for the resource
        """
        rule = self.classification_rules[classification]
        expiry_date = datetime.now() + rule.retention_period
        self.data_retention_schedule[resource_id] = expiry_date
        self._log_audit(
            user_id="system",
            action="retention_set",
            resource_type="data",
            resource_id=resource_id,
            classification=classification,
            outcome="success",
            details={"expiry_date": expiry_date.isoformat()},
        )
        return expiry_date

    def check_retention(self, resource_id: str) -> bool:
        """Check if a resource should be retained based on retention schedule.

        Args:
            resource_id: Resource identifier

        Returns:
            True if resource should be retained, False if expired
        """
        if resource_id not in self.data_retention_schedule:
            return True  # No schedule set, retain by default

        expiry_date = self.data_retention_schedule[resource_id]
        return datetime.now() < expiry_date

    def apply_data_minimization(
        self, data: Dict[str, Any], classification: DataClassification
    ) -> Dict[str, Any]:
        """Apply data minimization based on classification.

        Args:
            data: Data to minimize
            classification: Data classification

        Returns:
            Minimized data
        """
        rule = self.classification_rules[classification]

        if not rule.minimization_required:
            return data

        # Remove sensitive fields for PII/PHI
        if classification in [DataClassification.PII, DataClassification.PHI]:
            sensitive_fields = [
                "name",
                "email",
                "phone",
                "address",
                "ssn",
                "medical_record",
            ]
            minimized = {k: v for k, v in data.items() if k.lower() not in sensitive_fields}
            return minimized

        return data

    def _log_audit(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        classification: DataClassification,
        outcome: str,
        details: Dict[str, Any],
    ) -> None:
        """Log audit entry."""
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            classification=classification,
            outcome=outcome,
            details=details,
        )
        self.audit_log.append(entry)

        # Log to file if configured
        if self.config.get("audit_log_file"):
            self._write_audit_log_to_file(entry)

    def _write_audit_log_to_file(self, entry: AuditLogEntry) -> None:
        """Write audit log entry to file."""
        log_file = Path(self.config["audit_log_file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)

        log_entry = {
            "timestamp": entry.timestamp.isoformat(),
            "user_id": entry.user_id,
            "action": entry.action,
            "resource_type": entry.resource_type,
            "resource_id": entry.resource_id,
            "classification": entry.classification.value,
            "outcome": entry.outcome,
            "details": entry.details,
        }

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_audit_log(self, user_id: Optional[str] = None, limit: int = 100) -> List[AuditLogEntry]:
        """Retrieve audit log entries.

        Args:
            user_id: Filter by user ID (optional)
            limit: Maximum number of entries to return

        Returns:
            List of audit log entries
        """
        logs = self.audit_log
        if user_id:
            logs = [log for log in logs if log.user_id == user_id]
        return logs[-limit:]

    def cleanup_expired_data(self) -> List[str]:
        """Identify expired data based on retention schedules.

        Returns:
            List of resource IDs that have expired
        """
        now = datetime.now()
        expired = []
        for resource_id, expiry_date in self.data_retention_schedule.items():
            if expiry_date < now:
                expired.append(resource_id)
        return expired

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report.

        Returns:
            Compliance report with metrics and status
        """
        return {
            "report_timestamp": datetime.now().isoformat(),
            "total_consent_records": len(self.consent_records),
            "active_consent_records": sum(1 for c in self.consent_records.values() if c.is_active),
            "total_audit_entries": len(self.audit_log),
            "resources_with_retention": len(self.data_retention_schedule),
            "expired_resources": len(self.cleanup_expired_data()),
            "classification_rules": {
                cls.value: {
                    "retention_days": rule.retention_period.days,
                    "requires_encryption": rule.requires_encryption,
                    "requires_consent": rule.requires_consent,
                }
                for cls, rule in self.classification_rules.items()
            },
        }

    def _generate_consent_id(self, participant_id: str, consent_type: str) -> str:
        """Generate unique consent ID."""
        content = f"{participant_id}_{consent_type}_{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# Global compliance framework instance
_compliance_framework: Optional[ComplianceFramework] = None


def get_compliance_framework() -> ComplianceFramework:
    """Get or create global compliance framework instance."""
    global _compliance_framework
    if _compliance_framework is None:
        _compliance_framework = ComplianceFramework()
    return _compliance_framework
