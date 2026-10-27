"""Validation module for APGI Framework.

Provides parameter validation, system health checks, and diagnostic utilities.
"""

from typing import Any, Callable, Dict, List, Optional

from .error_recovery import (
    ErrorRecoveryManager,
    RetryConfig,
    get_recovery_manager,
    initialize_default_recovery_strategies,
    safe_execute,
    validate_and_fix,
    with_retry,
)
from .parameter_validator import ParameterValidator, ValidationResult, get_validator
from .system_health import HealthCheckResult, SystemHealthChecker, get_health_checker


# Mock classes for testing
class InputValidator:
    """Mock input validator for testing purposes."""

    def __init__(self) -> None:
        self.validation_rules: Dict[str, Callable[[Any], Dict[str, Any]]] = {}
        self.validation_history: List[Dict[str, Any]] = []

    def add_rule(self, rule_name: str, rule_function: Callable[[Any], Dict[str, Any]]) -> None:
        """Add a validation rule."""
        self.validation_rules[rule_name] = rule_function

    def validate_input(self, input_data: Any) -> Dict[str, Any]:
        """Validate input against all rules."""
        validation_id = f"input_validation_{hash(str(input_data)) % 10000:04d}"
        results: Dict[str, Dict[str, Any]] = {}

        for rule_name, rule_function in self.validation_rules.items():
            try:
                results[rule_name] = rule_function(input_data)
            except Exception as e:
                results[rule_name] = {"valid": False, "error": str(e)}

        validation_result = {
            "validation_id": validation_id,
            "input_data": input_data,
            "rule_results": results,
            "overall_valid": all(r.get("valid", False) for r in results.values()),
            "timestamp": "2024-01-01T00:00:00Z",
        }

        self.validation_history.append(validation_result)
        return validation_result

    def get_validation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get validation history."""
        return self.validation_history[-limit:] if self.validation_history else []


class DesignValidator:
    """Mock design validator for testing purposes."""

    def __init__(self) -> None:
        self.validation_rules: Dict[str, Callable[[Any], Dict[str, Any]]] = {}
        self.validation_results: Dict[str, Dict[str, Any]] = {}

    def add_validation_rule(
        self, rule_name: str, rule_function: Callable[[Any], Dict[str, Any]]
    ) -> None:
        """Add a validation rule."""
        self.validation_rules[rule_name] = rule_function

    def validate_design(self, design_data: Any) -> Dict[str, Any]:
        """Validate a design against all rules."""
        validation_id = f"validation_{hash(str(design_data)) % 10000:04d}"
        results: Dict[str, Dict[str, Any]] = {}
        results = {}

        for rule_name, rule_function in self.validation_rules.items():
            try:
                results[rule_name] = rule_function(design_data)
            except Exception as e:
                results[rule_name] = {"valid": False, "error": str(e)}

        validation_result = {
            "validation_id": validation_id,
            "design_data": design_data,
            "rule_results": results,
            "overall_valid": all(r.get("valid", False) for r in results.values()),
            "timestamp": "2024-01-01T00:00:00Z",
        }

        self.validation_results[validation_id] = validation_result
        return validation_result

    def get_validation_result(self, validation_id: str) -> Optional[Dict[str, Any]]:
        """Get validation result by ID."""
        return self.validation_results.get(validation_id)


# Initialize default recovery strategies on import
initialize_default_recovery_strategies()  # type: ignore[no-untyped-call]

__all__ = [
    "ParameterValidator",
    "ValidationResult",
    "get_validator",
    "SystemHealthChecker",
    "HealthCheckResult",
    "get_health_checker",
    "ErrorRecoveryManager",
    "RetryConfig",
    "with_retry",
    "get_recovery_manager",
    "safe_execute",
    "validate_and_fix",
    # Mock classes for testing
    "DesignValidator",
    "InputValidator",
]
