"""
JSON Schema Validation for APGI Parameters
======================================

Provides schema validation for APGI model parameters to ensure
proper structure and value ranges.
"""

from typing import Any, Dict, List, Optional


class APGIParameterValidator:
    """Validator for APGI model parameters using JSON schema"""

    def __init__(self) -> None:
        self.schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "APGI Formal Model Parameters",
            "description": "Schema for APGI formal model simulation parameters",
            "type": "object",
            "properties": {
                "tau_S": {
                    "description": "Surprise accumulation timescale (seconds)",
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 1.0,
                    "default": 0.5,
                },
                "tau_theta": {
                    "description": "Threshold adaptation timescale (seconds)",
                    "type": "number",
                    "minimum": 5.0,
                    "maximum": 60.0,
                    "default": 30.0,
                },
                "theta_0": {
                    "description": "Baseline threshold",
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 0.9,
                    "default": 0.5,
                },
                "alpha": {
                    "description": "Sigmoid steepness parameter",
                    "type": "number",
                    "minimum": 2.0,
                    "maximum": 20.0,
                    "default": 10.0,
                },
                "beta": {
                    "description": "Somatic bias parameter",
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 2.0,
                    "default": 1.0,
                },
                "Pi_e": {
                    "description": "Exteroceptive precision",
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 5.0,
                    "default": 1.0,
                },
                "Pi_i": {
                    "description": "Interoceptive precision",
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 5.0,
                    "default": 1.0,
                },
                "sigma_S": {
                    "description": "Surprise noise standard deviation",
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 1.0,
                    "default": 0.1,
                },
                "sigma_theta": {
                    "description": "Threshold noise standard deviation",
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 0.1,
                    "default": 0.01,
                },
                "gamma_M": {
                    "description": "Metabolic coupling strength",
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.1,
                },
                "gamma_A": {
                    "description": "Attentional coupling strength",
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.2,
                },
                "rho": {
                    "description": "Neuromodulatory gain",
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 3.0,
                    "default": 1.0,
                },
            },
            "additionalProperties": False,
            "required": [],
        }

    def _validate_parameter_type(
        self, key: str, value: Any, prop_schema: Dict[str, Any], errors: List[str]
    ) -> bool:
        """Validate parameter type, return False if invalid"""
        expected_type = prop_schema["type"]
        if expected_type == "number":
            if not isinstance(value, (int, float)):
                errors.append(f"Parameter '{key}' must be a number, got {type(value).__name__}")
                return False
        elif not isinstance(value, expected_type):
            errors.append(f"Parameter '{key}' must be {expected_type}, got {type(value).__name__}")
            return False
        return True

    def _validate_parameter_range(
        self, key: str, value: Any, prop_schema: Dict[str, Any], errors: List[str]
    ) -> None:
        """Validate parameter range for numeric values"""
        if isinstance(value, (int, float)):
            if "minimum" in prop_schema and value < prop_schema["minimum"]:
                errors.append(
                    f"Parameter '{key}' = {value} is below minimum {prop_schema['minimum']}"
                )
            if "maximum" in prop_schema and value > prop_schema["maximum"]:
                errors.append(
                    f"Parameter '{key}' = {value} is above maximum {prop_schema['maximum']}"
                )

    def _check_missing_parameters(self, parameters: Dict[str, Any], warnings: List[str]) -> None:
        """Check for missing recommended parameters"""
        recommended_params = ["tau_S", "tau_theta", "theta_0", "alpha"]
        for param in recommended_params:
            if param not in parameters:
                properties_dict = self.schema["properties"]
                if isinstance(properties_dict, dict) and param in properties_dict:
                    param_info = properties_dict[param]
                    if isinstance(param_info, dict) and "default" in param_info:
                        warnings.append(
                            f"Recommended parameter '{param}' not specified, will use default: {param_info['default']}"
                        )

    def validate(self, parameters: Any) -> Dict[str, Any]:
        """
        Validate parameters against schema

        Returns:
            Dict with validation results
        """
        try:
            # Check if parameters is a dict
            if not isinstance(parameters, dict):
                return {
                    "valid": False,
                    "errors": [f"Parameters must be a dictionary, got {type(parameters).__name__}"],
                    "warnings": [],
                }

            errors = []
            warnings: List[str] = []

            # Check each parameter
            for key, value in parameters.items():
                properties_dict = self.schema["properties"]
                if not isinstance(properties_dict, dict) or key not in properties_dict:
                    valid_params = (
                        list(properties_dict.keys()) if isinstance(properties_dict, dict) else []
                    )
                    errors.append(f"Unknown parameter: '{key}'. Valid parameters: {valid_params}")
                    continue

                prop_schema = properties_dict[key]

                # Check type
                if not self._validate_parameter_type(key, value, prop_schema, errors):
                    continue

                # Check range
                self._validate_parameter_range(key, value, prop_schema, errors)

            # Check for missing recommended parameters
            self._check_missing_parameters(parameters, warnings)

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "validated_params": len(parameters),
            }

        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
            }

    def get_parameter_info(self, param_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific parameter"""
        properties_dict = self.schema["properties"]
        if isinstance(properties_dict, dict):
            result = properties_dict.get(param_name)
            if isinstance(result, dict):
                return result
        return None

    def list_valid_parameters(self) -> List[str]:
        """Get list of all valid parameter names"""
        properties_dict = self.schema["properties"]
        if isinstance(properties_dict, dict):
            return list(properties_dict.keys())
        return []


# Global validator instance
parameter_validator = APGIParameterValidator()


def validate_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to validate parameters"""
    return parameter_validator.validate(parameters)


def require_valid_parameters(
    parameters: Dict[str, Any], model_id: str = "M-APGI-FORMAL"
) -> Dict[str, Any]:
    """Validate parameters, raising instead of returning an error report.

    A name the model registry does not map for ``model_id`` raises
    ``UnmappedParameterError``; a mapped name with a bad type or out-of-range value
    raises ``ValueError``. Returns the validation report when everything is acceptable.
    """
    from apgi_framework.model_registry import require_mapped_parameters

    require_mapped_parameters(model_id, parameters)
    report = parameter_validator.validate(parameters)
    if not report["valid"]:
        raise ValueError("; ".join(report["errors"]))
    return report
