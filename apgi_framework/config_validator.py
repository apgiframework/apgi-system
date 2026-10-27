"""Configuration Validation Tools.

Provides comprehensive validation for APGI system configuration files,
including parameter range validation, schema validation, and helpful
error messages for common configuration mistakes.
"""

import warnings
from typing import Any, Dict, List, Tuple, Union


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


class ConfigValidationWarning(UserWarning):
    """Warning for non-critical configuration issues."""

    pass


class ConfigValidator:
    """Validates APGI system configuration files.

    This class provides comprehensive validation for all configuration
    sections, including parameter range checking, schema validation,
    and helpful error messages for common mistakes.

    Examples
    --------
    >>> validator = ConfigValidator()
    >>> config = load_config("config.yaml")
    >>> validator.validate(config)  # Raises ConfigValidationError if invalid
    >>> errors = validator.validate_with_details(config)
    >>> if errors:
    ...     print(f"Found {len(errors)} validation errors")
    """

    def __init__(self) -> None:
        """Initialize the configuration validator."""
        self._define_schema()

    def _define_schema(self) -> None:
        """Define the configuration schema with validation rules."""
        self.schema = {
            "system": {
                "required": True,
                "fields": {
                    "name": {"type": str, "required": False},
                    "timestep_ms": {
                        "type": (int, float),
                        "required": True,
                        "range": (0.01, 100.0),
                        "description": "Simulation timestep in milliseconds",
                    },
                    "random_seed": {"type": int, "required": False, "range": (0, 2**31 - 1)},
                },
            },
            "hierarchy": {
                "required": True,
                "fields": {
                    "num_levels": {
                        "type": int,
                        "required": True,
                        "range": (1, 10),
                        "description": "Number of hierarchical levels",
                    },
                    "level_configs": {
                        "type": list,
                        "required": True,
                        "description": "Configuration for each hierarchical level",
                    },
                },
            },
            "active_inference": {
                "required": True,
                "fields": {
                    "learning_rate": {
                        "type": (int, float),
                        "required": True,
                        "range": (0.0, 1.0),
                        "description": "Learning rate for active inference",
                    },
                    "precision_init": {
                        "type": (int, float),
                        "required": True,
                        "range": (0.01, 100.0),
                        "description": "Initial precision value",
                    },
                    "precision_range": {
                        "type": list,
                        "required": True,
                        "length": 2,
                        "description": "Min and max precision values",
                    },
                    "free_energy_threshold": {
                        "type": (int, float),
                        "required": True,
                        "range": (0.0, 10000.0),
                    },
                },
            },
            "predictive_processing": {
                "required": True,
                "fields": {
                    "prediction_horizon_ms": {
                        "type": (int, float),
                        "required": True,
                        "range": (1.0, 10000.0),
                    },
                    "error_accumulation_window_ms": {
                        "type": (int, float),
                        "required": True,
                        "range": (1.0, 10000.0),
                    },
                    "temporal_discount": {
                        "type": (int, float),
                        "required": True,
                        "range": (0.0, 1.0),
                    },
                },
            },
            "precision": {
                "required": True,
                "fields": {
                    "exteroceptive_baseline": {
                        "type": (int, float),
                        "required": True,
                        "range": (0.01, 100.0),
                    },
                    "interoceptive_baseline": {
                        "type": (int, float),
                        "required": True,
                        "range": (0.01, 100.0),
                    },
                    "intero_somatic_beta": {
                        "type": (int, float),
                        "required": False,
                        "range": (0.0, 10.0),
                    },
                    "attention_gain_range": {"type": list, "required": True, "length": 2},
                    "volatility_sensitivity": {
                        "type": (int, float),
                        "required": True,
                        "range": (0.0, 10.0),
                    },
                },
            },
            "ignition": {
                "required": True,
                "fields": {
                    "baseline_threshold": {
                        "type": (int, float),
                        "required": True,
                        "range": (0.1, 100.0),
                    },
                    "threshold_range": {"type": list, "required": True, "length": 2},
                    "sigmoid_alpha": {
                        "type": (int, float),
                        "required": True,
                        "range": (0.1, 100.0),
                    },
                    "tau_S_ms": {
                        "type": (int, float),
                        "required": False,
                        "range": (1.0, 60000.0),
                    },
                    "sigma_S": {
                        "type": (int, float),
                        "required": False,
                        "range": (0.0, 10.0),
                    },
                    "tau_theta_ms": {
                        "type": (int, float),
                        "required": False,
                        "range": (1.0, 600000.0),
                    },
                    "sigma_theta": {
                        "type": (int, float),
                        "required": False,
                        "range": (0.0, 10.0),
                    },
                    "theta_0_sleep": {
                        "type": (int, float),
                        "required": False,
                        "range": (0.01, 100.0),
                    },
                    "theta_0_alert": {
                        "type": (int, float),
                        "required": False,
                        "range": (0.01, 100.0),
                    },
                    "gamma_M": {
                        "type": (int, float),
                        "required": False,
                        "range": (-10.0, 10.0),
                    },
                    "lambda_S": {
                        "type": (int, float),
                        "required": False,
                        "range": (-10.0, 10.0),
                    },
                    "amplification_duration_ms": {
                        "type": (int, float),
                        "required": True,
                        "range": (10.0, 10000.0),
                    },
                    "refractory_period_ms": {
                        "type": (int, float),
                        "required": True,
                        "range": (10.0, 10000.0),
                    },
                    "workspace_nodes": {"type": int, "required": True, "range": (10, 100000)},
                },
            },
            "interoception": {
                "required": True,
                "fields": {
                    "body_states": {"type": list, "required": True, "element_type": dict},
                    "prediction_lead_ms": {
                        "type": (int, float),
                        "required": True,
                        "range": (1.0, 10000.0),
                    },
                    "allostatic_ranges": {"type": dict, "required": True},
                },
            },
            "somatic_markers": {
                "required": True,
                "fields": {
                    "capacity": {"type": int, "required": True, "range": (1, 1000000)},
                    "learning_rate": {"type": (int, float), "required": True, "range": (0.0, 1.0)},
                    "decay_rate": {"type": (int, float), "required": True, "range": (0.0, 1.0)},
                    "retrieval_threshold": {
                        "type": (int, float),
                        "required": True,
                        "range": (0.0, 1.0),
                    },
                    "gain_modulation_range": {"type": list, "required": True, "length": 2},
                },
            },
            "thermodynamic": {
                "required": True,
                "fields": {
                    "total_energy_budget": {
                        "type": (int, float),
                        "required": True,
                        "range": (1.0, 100000.0),
                    },
                    "baseline_consumption": {
                        "type": (int, float),
                        "required": True,
                        "range": (0.0, 100000.0),
                    },
                    "ignition_cost": {
                        "type": (int, float),
                        "required": True,
                        "range": (0.0, 100000.0),
                    },
                    "recovery_rate": {
                        "type": (int, float),
                        "required": True,
                        "range": (0.0, 100000.0),
                    },
                    "depletion_threshold": {
                        "type": (int, float),
                        "required": True,
                        "range": (0.0, 100000.0),
                    },
                },
            },
            "self_model": {"required": False, "fields": {}},
            "oscillations": {
                "required": False,
                "fields": {
                    "bands": {"type": dict, "required": False},
                    "coupling_strength": {
                        "type": (int, float),
                        "required": False,
                        "range": (0.0, 1.0),
                    },
                    "criticality_parameter": {
                        "type": (int, float),
                        "required": False,
                        "range": (0.1, 10.0),
                    },
                },
            },
            "multi_modal": {
                "required": False,
                "fields": {
                    "fusion_method": {"type": str, "required": False},
                    "modalities": {"type": dict, "required": False},
                },
            },
            "validation": {"required": False, "fields": {}},
            "visualization": {"required": False, "fields": {}},
            "free_energy": {"required": False, "fields": {}},
            "api": {"required": False, "fields": {}},
            "server": {"required": False, "fields": {}},
            "database": {"required": False, "fields": {}},
            "simulation": {"required": False, "fields": {}},
            "logging": {"required": False, "fields": {}},
            "security": {"required": False, "fields": {}},
        }

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration and raise exception if invalid.
        Returns a list of warnings if validation succeeds.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to validate

        Returns
        -------
        List[str]
            List of validation warnings (if any)

        Raises
        ------
        ConfigValidationError
            If configuration is invalid
        """
        errors = self.validate_with_details(config)
        warnings_list = self.validate_parameter_ranges(config)

        # Always surface warnings via the warnings module
        for warning in warnings_list:
            warnings.warn(warning, ConfigValidationWarning)

        # Critical errors block execution
        if errors:
            error_msg = self._format_errors(errors)
            raise ConfigValidationError(f"Configuration validation failed:\n{error_msg}")

        return warnings_list

    def validate_with_details(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return list of error messages.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to validate

        Returns
        -------
        List[str]
            List of error messages (empty if valid)

        Examples
        --------
        >>> validator = ConfigValidator()
        >>> errors = validator.validate_with_details(config)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"Error: {error}")
        """
        errors = []

        # Check for required sections
        for section_name, section_schema in self.schema.items():
            if section_schema.get("required", False):
                if section_name not in config:
                    errors.append(
                        f"Missing required section: '{section_name}'. "
                        f"This section is required for proper system operation."
                    )
                    continue

            # Validate section fields if section exists
            if section_name in config:
                section_errors = self._validate_section(
                    section_name, config[section_name], section_schema
                )
                errors.extend(section_errors)

        # Check for unknown sections (warning only)
        for section_name in config:
            if section_name not in self.schema:
                warnings.warn(
                    f"Unknown configuration section: '{section_name}'. "
                    f"This section will be ignored.",
                    ConfigValidationWarning,
                )

        return errors

    def _validate_section(
        self, section_name: str, section_config: Any, section_schema: Dict[str, Any]
    ) -> List[str]:
        """Validate a configuration section."""
        errors = []

        if not isinstance(section_config, dict):
            errors.append(
                f"Section '{section_name}' must be a dictionary, "
                f"got {type(section_config).__name__}"
            )
            return errors

        fields_schema = section_schema.get("fields", {})

        # Check required fields
        for field_name, field_schema in fields_schema.items():
            if field_schema.get("required", False):
                if field_name not in section_config:
                    desc = field_schema.get("description", "")
                    error_msg = f"Missing required field: '{section_name}.{field_name}'"
                    if desc:
                        error_msg += f" ({desc})"
                    errors.append(error_msg)
                    continue

            # Validate field if it exists
            if field_name in section_config:
                field_errors = self._validate_field(
                    f"{section_name}.{field_name}", section_config[field_name], field_schema
                )
                errors.extend(field_errors)

        return errors

    def _validate_field(
        self, field_path: str, field_value: Any, field_schema: Dict[str, Any]
    ) -> List[str]:
        """Validate a configuration field."""
        errors = []

        # Type validation
        type_errors = self._validate_field_type(field_path, field_value, field_schema)
        errors.extend(type_errors)
        if type_errors:
            return errors  # Don't continue if type is wrong

        # Range validation for numeric types
        range_errors = self._validate_field_range(field_path, field_value, field_schema)
        errors.extend(range_errors)

        # Length validation for lists
        length_errors = self._validate_field_length(field_path, field_value, field_schema)
        errors.extend(length_errors)

        # Validate list elements are numeric for range fields
        element_errors = self._validate_list_elements(field_path, field_value, field_schema)
        errors.extend(element_errors)

        # Validate range lists have min < max
        range_order_errors = self._validate_range_order(field_path, field_value, field_schema)
        errors.extend(range_order_errors)

        return errors

    def _validate_field_type(
        self, field_path: str, field_value: Any, field_schema: Dict[str, Any]
    ) -> List[str]:
        """Validate field type."""
        errors = []
        expected_type = field_schema.get("type")
        if expected_type and not isinstance(field_value, expected_type):
            type_name = self._get_type_name(expected_type)
            errors.append(
                f"Field '{field_path}' has wrong type: "
                f"expected {type_name}, got {type(field_value).__name__}"
            )
        return errors

    def _validate_field_range(
        self, field_path: str, field_value: Any, field_schema: Dict[str, Any]
    ) -> List[str]:
        """Validate field range for numeric types."""
        errors = []
        if "range" in field_schema and isinstance(field_value, (int, float)):
            min_val, max_val = field_schema["range"]
            if field_value < min_val or field_value > max_val:
                errors.append(
                    f"Field '{field_path}' value {field_value} is out of range "
                    f"[{min_val}, {max_val}]"
                )
        return errors

    def _validate_field_length(
        self, field_path: str, field_value: Any, field_schema: Dict[str, Any]
    ) -> List[str]:
        """Validate field length for lists."""
        errors = []
        if "length" in field_schema and isinstance(field_value, list):
            expected_length = field_schema["length"]
            if len(field_value) != expected_length:
                errors.append(
                    f"Field '{field_path}' has wrong length: "
                    f"expected {expected_length}, got {len(field_value)}"
                )
        return errors

    def _validate_list_elements(
        self, field_path: str, field_value: Any, field_schema: Dict[str, Any]
    ) -> List[str]:
        """Validate list elements are numeric for range fields."""
        errors = []
        if isinstance(field_value, list):
            element_type = field_schema.get("element_type")
            for i, val in enumerate(field_value):
                if element_type and not isinstance(val, element_type):
                    errors.append(
                        f"Field '{field_path}[{i}]' has wrong type: "
                        f"expected {element_type.__name__}, got {type(val).__name__}"
                    )
                elif "length" in field_schema and not isinstance(val, (int, float)):
                    # Default for range-limited lists is numeric
                    errors.append(
                        f"Field '{field_path}[{i}]' must be numeric, got {type(val).__name__}"
                    )
        return errors

    def _validate_range_order(
        self, field_path: str, field_value: Any, field_schema: Dict[str, Any]
    ) -> List[str]:
        """Validate range lists have min < max."""
        errors = []
        if isinstance(field_value, list) and field_schema.get("length") == 2:
            if len(field_value) == 2 and all(isinstance(v, (int, float)) for v in field_value):
                if field_value[0] >= field_value[1]:
                    errors.append(
                        f"Field '{field_path}' range invalid: "
                        f"min ({field_value[0]}) must be less than max ({field_value[1]})"
                    )
        return errors

    def _get_type_name(self, type_spec: Union[type, Tuple[type, ...]]) -> str:
        """Get human-readable type name."""
        if isinstance(type_spec, tuple):
            names = [t.__name__ for t in type_spec]
            return " or ".join(names)
        return type_spec.__name__

    def _format_errors(self, errors: List[str]) -> str:
        """Format error messages for display."""
        formatted = []
        for i, error in enumerate(errors, 1):
            formatted.append(f"  {i}. {error}")
        return "\n".join(formatted)

    def validate_parameter_ranges(self, config: Dict[str, Any]) -> List[str]:
        """Validate that all numeric parameters are within acceptable ranges.

        This method performs additional validation beyond basic schema checking,
        including cross-parameter validation and consistency checks.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to validate

        Returns
        -------
        List[str]
            List of validation warnings/errors

        Examples
        --------
        >>> validator = ConfigValidator()
        >>> warnings = validator.validate_parameter_ranges(config)
        >>> for warning in warnings:
        ...     print(f"Warning: {warning}")
        """
        warnings_list = []

        # Validate thermodynamic consistency
        thermo_warnings = self._validate_thermodynamic_consistency(config)
        warnings_list.extend(thermo_warnings)

        # Validate ignition timing consistency
        ignition_warnings = self._validate_ignition_consistency(config)
        warnings_list.extend(ignition_warnings)

        # Validate precision range consistency
        precision_warnings = self._validate_precision_consistency(config)
        warnings_list.extend(precision_warnings)

        # Validate hierarchy consistency
        hierarchy_warnings = self._validate_hierarchy_consistency(config)
        warnings_list.extend(hierarchy_warnings)

        return warnings_list

    def _validate_thermodynamic_consistency(self, config: Dict[str, Any]) -> List[str]:
        """Validate thermodynamic configuration consistency."""
        warnings_list = []
        if "thermodynamic" in config:
            thermo = config["thermodynamic"]
            if all(
                k in thermo
                for k in ["total_energy_budget", "baseline_consumption", "ignition_cost"]
            ):
                total = thermo["total_energy_budget"]
                baseline = thermo["baseline_consumption"]
                ignition = thermo["ignition_cost"]

                if baseline + ignition > total:
                    warnings_list.append(
                        "Thermodynamic configuration warning: "
                        f"baseline_consumption ({baseline}) + ignition_cost ({ignition}) "
                        f"exceeds total_energy_budget ({total}). "
                        "System may run out of energy quickly."
                    )

                if "depletion_threshold" in thermo:
                    depletion = thermo["depletion_threshold"]
                    if depletion > total * 0.5:
                        warnings_list.append(
                            "Thermodynamic configuration warning: "
                            f"depletion_threshold ({depletion}) is more than 50% "
                            f"of total_energy_budget ({total}). "
                            "Consider lowering the threshold."
                        )
        return warnings_list

    def _validate_ignition_consistency(self, config: Dict[str, Any]) -> List[str]:
        """Validate ignition timing configuration consistency."""
        warnings_list = []
        if "ignition" in config:
            ignition = config["ignition"]
            if all(k in ignition for k in ["amplification_duration_ms", "refractory_period_ms"]):
                amp_duration = ignition["amplification_duration_ms"]
                refractory = ignition["refractory_period_ms"]

                if amp_duration < refractory:
                    warnings_list.append(
                        "Ignition configuration warning: "
                        f"amplification_duration_ms ({amp_duration}) is less than "
                        f"refractory_period_ms ({refractory}). "
                        "This may cause unexpected behavior."
                    )
        return warnings_list

    def _validate_precision_consistency(self, config: Dict[str, Any]) -> List[str]:
        """Validate precision range configuration consistency."""
        warnings_list = []
        if "active_inference" in config:
            ai = config["active_inference"]
            if "precision_range" in ai and "precision_init" in ai:
                prec_range = ai["precision_range"]
                prec_init = ai["precision_init"]

                if len(prec_range) == 2:
                    if prec_init < prec_range[0] or prec_init > prec_range[1]:
                        warnings_list.append(
                            "Active inference configuration warning: "
                            f"precision_init ({prec_init}) is outside "
                            f"precision_range [{prec_range[0]}, {prec_range[1]}]. "
                            "Initial precision will be clamped to range."
                        )
        return warnings_list

    def _validate_hierarchy_consistency(self, config: Dict[str, Any]) -> List[str]:
        """Validate hierarchy configuration consistency."""
        warnings_list = []
        if "hierarchy" in config:
            hierarchy = config["hierarchy"]
            if "num_levels" in hierarchy and "level_configs" in hierarchy:
                num_levels = hierarchy["num_levels"]
                level_configs = hierarchy["level_configs"]

                if len(level_configs) != num_levels:
                    warnings_list.append(
                        "Hierarchy configuration warning: "
                        f"num_levels ({num_levels}) does not match "
                        f"number of level_configs ({len(level_configs)}). "
                        "This may cause initialization errors."
                    )
        return warnings_list

    def get_common_mistakes_help(self) -> str:
        """Get help text for common configuration mistakes.

        Returns
        -------
        str
            Help text describing common configuration mistakes and how to fix them

        Examples
        --------
        >>> validator = ConfigValidator()
        >>> print(validator.get_common_mistakes_help())
        """
        return """
Common Configuration Mistakes and Solutions:

1. Missing Required Sections
   Problem: Configuration file is missing required sections like 'system', 'ignition', etc.
   Solution: Ensure all required sections are present. Use the default.yaml as a template.

2. Wrong Parameter Types
   Problem: Using string values where numbers are expected (e.g., timestep_ms: "1.0")
   Solution: Remove quotes from numeric values in YAML files.

3. Out of Range Values
   Problem: Parameter values outside acceptable ranges (e.g., learning_rate: 5.0)
   Solution: Check parameter ranges in documentation. Most rates should be between 0 and 1.

4. Invalid Range Specifications
   Problem: Range lists where min >= max (e.g., precision_range: [10.0, 1.0])
   Solution: Ensure first value is less than second value in all range specifications.

5. Thermodynamic Budget Issues
   Problem: baseline_consumption + ignition_cost > total_energy_budget
   Solution: Increase total_energy_budget or reduce consumption/cost values.

6. Hierarchy Mismatch
   Problem: num_levels doesn't match length of level_configs list
   Solution: Ensure level_configs has exactly num_levels entries.

7. Timing Inconsistencies
   Problem: amplification_duration_ms < refractory_period_ms
   Solution: Amplification duration should typically be longer than refractory period.

8. Precision Initialization
   Problem: precision_init outside precision_range
   Solution: Set precision_init to a value within the specified precision_range.

9. Missing Nested Fields
   Problem: Required nested fields missing (e.g., body_states in interoception)
   Solution: Check that all nested structures are complete.

10. Invalid YAML Syntax
    Problem: Indentation errors or invalid YAML syntax
    Solution: Use a YAML validator or linter to check syntax before loading.

For more information, see the configuration documentation or use validate_with_details()
to get specific error messages for your configuration.
"""

    def suggest_fixes(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest automatic fixes for common configuration issues.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to analyze

        Returns
        -------
        Dict[str, Any]
            Dictionary of suggested fixes with explanations

        Examples
        --------
        >>> validator = ConfigValidator()
        >>> fixes = validator.suggest_fixes(config)
        >>> for field, suggestion in fixes.items():
        ...     print(f"{field}: {suggestion}")
        """
        suggestions = {}

        # Check precision initialization
        if "active_inference" in config:
            ai = config["active_inference"]
            if "precision_range" in ai and "precision_init" in ai:
                prec_range = ai["precision_range"]
                prec_init = ai["precision_init"]

                if len(prec_range) == 2:
                    if prec_init < prec_range[0]:
                        suggestions["active_inference.precision_init"] = {
                            "current": prec_init,
                            "suggested": prec_range[0],
                            "reason": "Value below minimum range",
                        }
                    elif prec_init > prec_range[1]:
                        suggestions["active_inference.precision_init"] = {
                            "current": prec_init,
                            "suggested": prec_range[1],
                            "reason": "Value above maximum range",
                        }

        # Check thermodynamic budget
        if "thermodynamic" in config:
            thermo = config["thermodynamic"]
            if all(
                k in thermo
                for k in ["total_energy_budget", "baseline_consumption", "ignition_cost"]
            ):
                total = thermo["total_energy_budget"]
                baseline = thermo["baseline_consumption"]
                ignition = thermo["ignition_cost"]

                if baseline + ignition > total:
                    suggested_total = (baseline + ignition) * 1.5
                    suggestions["thermodynamic.total_energy_budget"] = {
                        "current": total,
                        "suggested": suggested_total,
                        "reason": "Insufficient budget for baseline + ignition costs",
                    }

        return suggestions


def validate_config_file(config_path: str) -> None:
    """Convenience function to validate a configuration file.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file

    Raises
    ------
    ConfigValidationError
        If configuration is invalid

    Examples
    --------
    >>> validate_config_file("config/default.yaml")
    """
    from pathlib import Path

    import yaml

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    validator = ConfigValidator()
    validator.validate(config)

    # Also check for warnings
    warnings_list = validator.validate_parameter_ranges(config)
    if warnings_list:
        print(f"\nConfiguration loaded successfully with {len(warnings_list)} warning(s):")
        for warning in warnings_list:
            print(f"  - {warning}")
