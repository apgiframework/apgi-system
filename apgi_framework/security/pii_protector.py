"""PII Protection Mechanisms for APGI.

Provides utilities for identifying, masking, and encrypting
Personally Identifiable Information (PII).
"""

import hashlib
import re
from typing import Any, Dict, List, Optional, Pattern

# PII Patterns
PII_PATTERNS: Dict[str, Pattern[str]] = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
}


class PIIProtector:
    """Utility for protecting PII in logs and data storage."""

    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or "default_masking_key"

    def mask_pii(self, text: str) -> str:
        """Mask PII in a string with placeholders."""
        masked_text = text
        for pii_type, pattern in PII_PATTERNS.items():
            masked_text = pattern.sub(f"[MASKED_{pii_type.upper()}]", masked_text)
        return masked_text

    def anonymize_data(self, data: Dict[str, Any], fields_to_mask: List[str]) -> Dict[str, Any]:
        """Anonymize specific fields in a dictionary."""
        anonymized = data.copy()
        for field in fields_to_mask:
            if field in anonymized:
                val = str(anonymized[field])
                # One-way hash for anonymization while preserving uniqueness
                h = hashlib.sha256((val + self.encryption_key).encode()).hexdigest()
                anonymized[field] = f"anon_{h[:12]}"
        return anonymized

    def redact_log_record(self, record_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Redact PII from a log record dictionary."""
        # Redact the message
        if "message" in record_dict:
            record_dict["message"] = self.mask_pii(str(record_dict["message"]))

        # Redact any sensitive keys in extra/details
        sensitive_keys = {"email", "phone", "address", "name", "password", "token"}

        def _redact_nested(d: Any) -> Any:
            if isinstance(d, dict):
                return {
                    k: ("[REDACTED]" if k.lower() in sensitive_keys else _redact_nested(v))
                    for k, v in d.items()
                }
            elif isinstance(d, list):
                return [_redact_nested(item) for item in d]
            return d

        result = _redact_nested(record_dict)
        return result if isinstance(result, dict) else {"data": result}


# Global protector instance
_pii_protector: Optional[PIIProtector] = None


def get_pii_protector() -> PIIProtector:
    global _pii_protector
    if _pii_protector is None:
        _pii_protector = PIIProtector()
    return _pii_protector
