"""Enhanced Data Validator for APGI Framework.

Provides comprehensive data validation, integrity checks, and sanitization
for experimental data, parameters, and user inputs.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""

    field_name: str
    severity: ValidationSeverity
    message: str
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    suggestion: Optional[str] = None
