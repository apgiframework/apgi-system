"""Compliance Enforcement Middleware.

Wires PII detection/masking and residency checks into request/response lifecycle
with mandatory audit logging for regulated flows.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ..logging.standardized_logging import get_logger, get_security_logger

logger = get_logger(__name__)
security_logger = get_security_logger()


class ComplianceAction(Enum):
    """Compliance enforcement actions."""

    ALLOW = "allow"
    MASK = "mask"
    REDACT = "redact"
    BLOCK = "block"
    AUDIT = "audit"


@dataclass
class ComplianceViolation:
    """Record of a compliance violation."""

    violation_type: str
    severity: str  # "low", "medium", "high", "critical"
    field_name: str
    detected_value: str
    action_taken: ComplianceAction
    timestamp: datetime
    request_id: str
    user_id: Optional[str] = None
    remediation: Optional[str] = None


class PIIDetector:
    """Detects Personally Identifiable Information in data."""

    # PII patterns
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b",
        "ssn": r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0{4})\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "passport": r"\b[A-Z]{1,2}\d{6,9}\b",
        "ip_address": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        "medical_record": r"\b(?:MRN|Medical Record Number)[:\s]*([A-Z0-9]{6,})\b",
        "dob": r"\b(?:DOB|Date of Birth)[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
    }

    def __init__(self):
        """Initialize PII detector with compiled patterns."""
        self.compiled_patterns = {
            key: re.compile(pattern, re.IGNORECASE) for key, pattern in self.PATTERNS.items()
        }

    def detect_pii(self, data: Any, field_name: str = "data") -> List[Dict[str, Any]]:
        """Detect PII in data.

        Args:
            data: Data to scan
            field_name: Name of field being scanned

        Returns:
            List of detected PII with type and location
        """
        detections = []

        if isinstance(data, str):
            for pii_type, pattern in self.compiled_patterns.items():
                matches = pattern.finditer(data)
                for match in matches:
                    detections.append(
                        {
                            "type": pii_type,
                            "field": field_name,
                            "value": match.group(0),
                            "position": match.span(),
                        }
                    )

        elif isinstance(data, dict):
            for key, value in data.items():
                detections.extend(self.detect_pii(value, f"{field_name}.{key}"))

        elif isinstance(data, (list, tuple)):
            for idx, item in enumerate(data):
                detections.extend(self.detect_pii(item, f"{field_name}[{idx}]"))

        return detections

    def mask_pii(self, data: Any, mask_char: str = "*") -> Any:
        """Mask PII in data.

        Args:
            data: Data to mask
            mask_char: Character to use for masking

        Returns:
            Data with PII masked
        """
        if isinstance(data, str):
            masked = data
            for pii_type, pattern in self.compiled_patterns.items():
                masked = pattern.sub(lambda m: mask_char * len(m.group(0)), masked)
            return masked

        elif isinstance(data, dict):
            return {key: self.mask_pii(value, mask_char) for key, value in data.items()}

        elif isinstance(data, (list, tuple)):
            return type(data)(self.mask_pii(item, mask_char) for item in data)

        return data


class DataResidencyValidator:
    """Validates data residency compliance."""

    def __init__(self, allowed_regions: Optional[Set[str]] = None):
        """Initialize residency validator.

        Args:
            allowed_regions: Set of allowed data regions (e.g., {'US', 'EU'})
        """
        self.allowed_regions = allowed_regions or {"US", "EU"}

    def validate_residency(self, data_region: str) -> bool:
        """Validate data is in allowed region.

        Args:
            data_region: Region where data is stored

        Returns:
            True if residency is compliant
        """
        return data_region.upper() in {r.upper() for r in self.allowed_regions}

    def get_residency_from_request(self, request_headers: Dict[str, str]) -> Optional[str]:
        """Extract data residency from request headers.

        Args:
            request_headers: HTTP request headers

        Returns:
            Data region or None
        """
        return request_headers.get("X-Data-Residency") or request_headers.get("X-Data-Region")


class ComplianceEnforcer:
    """Enforces compliance policies in request/response lifecycle."""

    def __init__(
        self,
        pii_detector: Optional[PIIDetector] = None,
        residency_validator: Optional[DataResidencyValidator] = None,
    ):
        """Initialize compliance enforcer.

        Args:
            pii_detector: PII detector instance
            residency_validator: Data residency validator instance
        """
        self.pii_detector = pii_detector or PIIDetector()
        self.residency_validator = residency_validator or DataResidencyValidator()
        self.violations: List[ComplianceViolation] = []

    def check_request_compliance(
        self,
        request_body: Any,
        request_headers: Dict[str, str],
        request_id: str,
        user_id: Optional[str] = None,
    ) -> tuple[bool, List[ComplianceViolation]]:
        """Check request for compliance violations.

        Args:
            request_body: Request body data
            request_headers: Request headers
            request_id: Unique request ID
            user_id: User making request

        Returns:
            Tuple of (compliant, violations)
        """
        violations = []

        # Check for PII in request
        pii_detections = self.pii_detector.detect_pii(request_body)
        for detection in pii_detections:
            violation = ComplianceViolation(
                violation_type="PII_DETECTED",
                severity="high",
                field_name=detection["field"],
                detected_value=detection["type"],
                action_taken=ComplianceAction.AUDIT,
                timestamp=datetime.now(timezone.utc),
                request_id=request_id,
                user_id=user_id,
                remediation="PII should not be transmitted in requests; use tokenization",
            )
            violations.append(violation)
            security_logger.warning(
                f"PII detected in request: {detection['type']} in {detection['field']}",
                extra={
                    "request_id": request_id,
                    "user_id": user_id,
                    "pii_type": detection["type"],
                },
            )

        # Check data residency
        data_region = self.residency_validator.get_residency_from_request(request_headers)
        if data_region and not self.residency_validator.validate_residency(data_region):
            violation = ComplianceViolation(
                violation_type="RESIDENCY_VIOLATION",
                severity="critical",
                field_name="X-Data-Residency",
                detected_value=data_region,
                action_taken=ComplianceAction.BLOCK,
                timestamp=datetime.now(timezone.utc),
                request_id=request_id,
                user_id=user_id,
                remediation=f"Data must be in allowed regions: {self.residency_validator.allowed_regions}",
            )
            violations.append(violation)
            security_logger.error(
                f"Data residency violation: {data_region}",
                extra={"request_id": request_id, "user_id": user_id},
            )

        self.violations.extend(violations)
        return len(violations) == 0, violations

    def mask_response_pii(self, response_body: Any) -> Any:
        """Mask PII in response body.

        Args:
            response_body: Response data

        Returns:
            Response with PII masked
        """
        return self.pii_detector.mask_pii(response_body)

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get compliance audit log.

        Returns:
            List of compliance violations
        """
        return [
            {
                "timestamp": v.timestamp.isoformat(),
                "violation_type": v.violation_type,
                "severity": v.severity,
                "field_name": v.field_name,
                "action_taken": v.action_taken.value,
                "request_id": v.request_id,
                "user_id": v.user_id,
                "remediation": v.remediation,
            }
            for v in self.violations
        ]

    def clear_violations(self) -> None:
        """Clear violation history."""
        self.violations.clear()


# Global compliance enforcer instance
_compliance_enforcer: Optional[ComplianceEnforcer] = None


def get_compliance_enforcer() -> ComplianceEnforcer:
    """Get or create global compliance enforcer."""
    global _compliance_enforcer
    if _compliance_enforcer is None:
        _compliance_enforcer = ComplianceEnforcer()
    return _compliance_enforcer


def compliance_middleware(
    request_body: Any,
    request_headers: Dict[str, str],
    request_id: str,
    user_id: Optional[str] = None,
) -> tuple[bool, List[ComplianceViolation]]:
    """Middleware function for compliance checking.

    Args:
        request_body: Request body
        request_headers: Request headers
        request_id: Request ID
        user_id: User ID

    Returns:
        Tuple of (compliant, violations)
    """
    enforcer = get_compliance_enforcer()
    return enforcer.check_request_compliance(request_body, request_headers, request_id, user_id)
