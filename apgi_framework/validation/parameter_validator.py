"""Parameter Validation Module.

Provides comprehensive validation for all APGI Framework parameters including
range checking, type validation, and helpful error messages.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ValidationResult:
    """Result of parameter validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

    def __bool__(self) -> bool:
        return self.is_valid

    def get_message(self) -> str:
        """Get formatted validation message."""
        messages = []

        if self.errors:
            messages.append("ERRORS:")
            for error in self.errors:
                messages.append(f"  ❌ {error}")

        if self.warnings:
            messages.append("WARNINGS:")
            for warning in self.warnings:
                messages.append(f"  ⚠️  {warning}")

        if self.suggestions:
            messages.append("SUGGESTIONS:")
            for suggestion in self.suggestions:
                messages.append(f"  💡 {suggestion}")

        return "\n".join(messages) if messages else "✓ All parameters valid"


class ParameterValidator:
    """Comprehensive parameter validator for APGI Framework.

    Validates all parameter types including:
    - APGI equation parameters
    - Experimental configuration
    - Neural signature thresholds
    - Statistical parameters
    - Pharmacological conditions
    """

    # Parameter ranges and constraints
    APGI_PARAM_RANGES = {
        "extero_precision": (
            0.01,
            10.0,
            "Exteroceptive precision must be positive (typical: 0.5-5.0)",
        ),
        "intero_precision": (
            0.01,
            10.0,
            "Interoceptive precision must be positive (typical: 0.5-5.0)",
        ),
        "extero_error": (
            -10.0,
            10.0,
            "Exteroceptive error should be standardized z-score (typical: -3 to 3)",
        ),
        "intero_error": (
            -10.0,
            10.0,
            "Interoceptive error should be standardized z-score (typical: -3 to 3)",
        ),
        "somatic_gain": (0.01, 5.0, "Somatic gain must be positive (typical: 0.5-2.0)"),
        "threshold": (
            0.1,
            10.0,
            "Ignition threshold must be positive (typical: 2.0-5.0)",
        ),
        "steepness": (
            0.1,
            10.0,
            "Steepness parameter must be positive (typical: 1.0-3.0)",
        ),
    }

    EXPERIMENTAL_PARAM_RANGES = {
        "n_trials": (1, 100000, "Number of trials must be positive (typical: 50-1000)"),
        "n_participants": (
            1,
            10000,
            "Number of participants must be positive (typical: 20-200)",
        ),
        "alpha_level": (
            0.001,
            0.1,
            "Alpha level must be between 0.001 and 0.1 (typical: 0.05)",
        ),
        "effect_size_threshold": (0.1, 2.0, "Effect size threshold (typical: 0.3-0.8)"),
        "power_threshold": (
            0.5,
            0.99,
            "Statistical power threshold (typical: 0.8-0.95)",
        ),
    }

    NEURAL_SIGNATURE_RANGES = {
        "p3b_threshold": (
            1.0,
            20.0,
            "P3b amplitude threshold in μV (typical: 3.0-7.0)",
        ),
        "gamma_plv_threshold": (0.05, 0.8, "Gamma PLV threshold (typical: 0.2-0.4)"),
        "bold_z_threshold": (1.0, 5.0, "BOLD Z-score threshold (typical: 2.3-3.5)"),
        "pci_threshold": (0.1, 0.8, "PCI threshold (typical: 0.3-0.5)"),
    }

    RETRY_PARAM_RANGES = {
        "max_attempts": (1, 10, "Max attempts must be between 1 and 10"),
        "initial_delay": (
            0.1,
            10.0,
            "Initial delay must be between 0.1 and 10.0 seconds",
        ),
        "backoff_factor": (1.0, 5.0, "Backoff factor must be between 1.0 and 5.0"),
        "max_delay": (1.0, 300.0, "Max delay must be between 1.0 and 300.0 seconds"),
    }

    PERFORMANCE_PARAM_RANGES = {
        "min_rt": (0.05, 2.0, "Minimum RT (typical: 0.1-0.3s)"),
        "max_rt": (0.5, 30.0, "Maximum RT (typical: 2.0-10.0s)"),
        "min_accuracy": (0.0, 1.0, "Minimum accuracy (0.0-1.0)"),
        "target_accuracy": (0.0, 1.0, "Target accuracy (0.0-1.0)"),
    }

    STIMULUS_PARAM_RANGES = {
        "stimulus_min": (0.0, 1.0, "Min stimulus intensity"),
        "stimulus_max": (0.0, 1.0, "Max stimulus intensity"),
        "lapse_rate": (0.0, 0.5, "Lapse rate (typical: 0.01-0.05)"),
        "guess_rate": (0.0, 1.0, "Guess rate (typical: 0.5 for 2AFC)"),
    }

    def __init__(self) -> None:
        self.validation_history: List[ValidationResult] = []

    def validate_apgi_parameters(self, **params: Any) -> ValidationResult:
        """Validate APGI equation parameters.

        Args:
            **params: APGI parameters to validate

        Returns:
            ValidationResult with errors, warnings, and suggestions
        """
        errors = []
        warnings = []
        suggestions = []

        # Check for required parameters
        required_params = [
            "extero_precision",
            "intero_precision",
            "extero_error",
            "intero_error",
            "somatic_gain",
            "threshold",
            "steepness",
        ]

        for param in required_params:
            if param not in params:
                errors.append(f"Missing required parameter: {param}")

        # Validate each parameter
        for param_name, value in params.items():
            if param_name not in self.APGI_PARAM_RANGES:
                warnings.append(f"Unknown parameter: {param_name}")
                continue

            min_val, max_val, description = self.APGI_PARAM_RANGES[param_name]

            # Type check
            if not isinstance(value, (int, float, np.number)):
                errors.append(f"{param_name}: Must be numeric, got {type(value).__name__}")
                continue

            value = float(value)  # Cast to float for subsequent operations

            # Range check
            if not (min_val <= value <= max_val):
                errors.append(
                    f"{param_name}: Value {value} outside valid range [{min_val}, {max_val}]. {description}"
                )

            # Warnings for unusual values
            if param_name in ["extero_precision", "intero_precision"]:
                if value < 0.5:
                    warnings.append(
                        f"{param_name}: Very low precision ({value}) may indicate weak signal"
                    )
                elif value > 5.0:
                    warnings.append(f"{param_name}: Very high precision ({value}) is unusual")

            if param_name in ["extero_error", "intero_error"]:
                if abs(value) > 3.0:
                    warnings.append(
                        f"{param_name}: Large error ({value}) is unusual for standardized values"
                    )

            if param_name == "somatic_gain":
                if value < 0.8:
                    suggestions.append(
                        f"Low somatic gain ({value}) reduces interoceptive influence"
                    )
                elif value > 2.0:
                    suggestions.append(
                        f"High somatic gain ({value}) strongly amplifies interoceptive signals"
                    )

            if param_name == "threshold":
                if value < 2.0:
                    suggestions.append(f"Low threshold ({value}) makes ignition easier to trigger")
                elif value > 5.0:
                    suggestions.append(f"High threshold ({value}) makes ignition harder to trigger")

        # Cross-parameter validation
        if "extero_precision" in params and "intero_precision" in params:
            ratio = params["intero_precision"] / params["extero_precision"]
            if ratio > 2.0:
                suggestions.append(
                    f"Interoceptive precision is {ratio:.1f}x higher than exteroceptive - strong interoceptive bias"
                )
            elif ratio < 0.5:
                suggestions.append(
                    f"Exteroceptive precision is {1 / ratio:.1f}x higher than interoceptive - strong exteroceptive bias"
                )

        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )

        self.validation_history.append(result)
        return result

    def validate_experimental_config(self, **config: Any) -> ValidationResult:
        """Validate experimental configuration parameters.

        Args:
            **config: Experimental configuration to validate

        Returns:
            ValidationResult with errors, warnings, and suggestions
        """
        errors = []
        warnings = []
        suggestions = []

        # Validate each parameter
        for param_name, value in config.items():
            if param_name not in self.EXPERIMENTAL_PARAM_RANGES:
                continue  # Skip unknown parameters (may be valid)

            min_val, max_val, description = self.EXPERIMENTAL_PARAM_RANGES[param_name]

            # Type check
            if not isinstance(value, (int, float, np.number)):
                errors.append(f"{param_name}: Must be numeric, got {type(value).__name__}")
                continue

            # Range check
            if not (min_val <= value <= max_val):
                errors.append(
                    f"{param_name}: Value {value} outside valid range [{min_val}, {max_val}]. {description}"
                )

            # Specific parameter warnings
            if param_name == "n_trials":
                if value < 50:
                    warnings.append(
                        f"Low trial count ({value}) may have insufficient statistical power"
                    )
                elif value > 5000:
                    warnings.append(
                        f"Very high trial count ({value}) may be computationally expensive"
                    )

            if param_name == "n_participants":
                if value < 20:
                    warnings.append(f"Low participant count ({value}) may not generalize well")
                elif value > 500:
                    warnings.append(f"Very high participant count ({value}) may be unnecessary")

            if param_name == "alpha_level":
                if value > 0.05:
                    warnings.append(f"Alpha level ({value}) higher than conventional 0.05")

            if param_name == "power_threshold":
                if value < 0.8:
                    warnings.append(f"Power threshold ({value}) below conventional 0.8")

        # Cross-parameter validation
        if "n_trials" in config and "n_participants" in config:
            total_observations = config["n_trials"] * config["n_participants"]
            if total_observations < 1000:
                suggestions.append(
                    f"Total observations ({total_observations}) may be low for robust statistics"
                )
            elif total_observations > 100000:
                suggestions.append(
                    f"Total observations ({total_observations}) is very large - consider reducing for efficiency"
                )

        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )

        self.validation_history.append(result)
        return result

    def validate_neural_thresholds(self, **thresholds: Any) -> ValidationResult:
        """Validate neural signature threshold parameters.

        Args:
            **thresholds: Neural signature thresholds to validate

        Returns:
            ValidationResult with errors, warnings, and suggestions
        """
        errors = []
        warnings = []
        suggestions = []

        for param_name, value in thresholds.items():
            if param_name not in self.NEURAL_SIGNATURE_RANGES:
                continue

            min_val, max_val, description = self.NEURAL_SIGNATURE_RANGES[param_name]

            # Type check
            if not isinstance(value, (int, float, np.number)):
                errors.append(f"{param_name}: Must be numeric, got {type(value).__name__}")
                continue

            # Range check
            if not (min_val <= value <= max_val):
                errors.append(
                    f"{param_name}: Value {value} outside valid range [{min_val}, {max_val}]. {description}"
                )

            # Specific warnings
            if param_name == "p3b_threshold":
                if value < 3.0:
                    suggestions.append(
                        f"Low P3b threshold ({value} μV) may increase false positives"
                    )
                elif value > 7.0:
                    suggestions.append(
                        f"High P3b threshold ({value} μV) may miss genuine ignition events"
                    )

            if param_name == "gamma_plv_threshold":
                if value < 0.2:
                    suggestions.append(f"Low gamma PLV threshold ({value}) may be too permissive")
                elif value > 0.4:
                    suggestions.append(f"High gamma PLV threshold ({value}) may be too strict")

            if param_name == "bold_z_threshold":
                if value < 2.3:
                    warnings.append(
                        f"BOLD threshold ({value}) below conventional significance (Z=2.3, p<0.05)"
                    )
                elif value > 3.5:
                    suggestions.append(f"BOLD threshold ({value}) is very conservative")

        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )

        self.validation_history.append(result)
        return result

    def validate_pharmacological_condition(
        self, drug_type: str, dosage: float, administration_time: float
    ) -> ValidationResult:
        """Validate pharmacological condition parameters.

        Args:
            drug_type: Type of drug
            dosage: Drug dosage
            administration_time: Time before testing (minutes)

        Returns:
            ValidationResult with errors, warnings, and suggestions
        """
        errors = []
        warnings = []
        suggestions = []

        # Valid drug types
        valid_drugs = ["propranolol", "l_dopa", "ssri", "physostigmine", "placebo"]

        if drug_type.lower() not in valid_drugs:
            errors.append(f"Unknown drug type: {drug_type}. Valid types: {', '.join(valid_drugs)}")

        # Dosage validation
        dosage_ranges = {
            "propranolol": (10, 160, "mg"),
            "l_dopa": (50, 400, "mg"),
            "ssri": (10, 100, "mg"),
            "physostigmine": (0.5, 4, "mg"),
            "placebo": (0, 0, "mg"),
        }

        if drug_type.lower() in dosage_ranges:
            min_dose, max_dose, unit = dosage_ranges[drug_type.lower()]

            if dosage < 0:
                errors.append(f"Dosage cannot be negative: {dosage}")
            elif not (min_dose <= dosage <= max_dose):
                warnings.append(
                    f"{drug_type} dosage {dosage} {unit} outside typical range [{min_dose}-{max_dose}] {unit}"
                )

        # Administration time validation
        if administration_time < 0:
            errors.append(f"Administration time cannot be negative: {administration_time}")
        elif administration_time > 240:
            warnings.append(
                f"Long administration time ({administration_time} min) - drug effects may have peaked"
            )

        # Drug-specific suggestions
        if drug_type.lower() == "propranolol":
            if administration_time < 30:
                suggestions.append(
                    "Propranolol typically requires 30-60 minutes to reach peak effect"
                )
        elif drug_type.lower() == "l_dopa":
            if administration_time < 20:
                suggestions.append("L-DOPA typically requires 20-30 minutes to reach peak effect")
        elif drug_type.lower() == "physostigmine":
            if administration_time < 10:
                suggestions.append("Physostigmine has rapid onset (10-15 minutes)")

        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )

        self.validation_history.append(result)
        return result

    def validate_statistical_parameters(self, **params: Any) -> ValidationResult:
        """Validate statistical analysis parameters.

        Args:
            **params: Statistical parameters to validate

        Returns:
            ValidationResult with errors, warnings, and suggestions
        """
        errors = []
        warnings = []
        suggestions = []

        # Alpha level
        if "alpha" in params:
            alpha = params["alpha"]
            if not (0 < alpha < 1):
                errors.append(f"Alpha level must be between 0 and 1, got {alpha}")
            elif alpha > 0.1:
                warnings.append(f"Alpha level ({alpha}) is unusually high")

        # Effect size
        if "effect_size" in params:
            effect_size = params["effect_size"]
            if effect_size < 0:
                errors.append(f"Effect size cannot be negative: {effect_size}")
            elif effect_size < 0.2:
                suggestions.append(f"Small effect size ({effect_size}) requires large sample")
            elif effect_size > 1.0:
                suggestions.append(f"Large effect size ({effect_size}) - easy to detect")

        # Power
        if "power" in params:
            power = params["power"]
            if not (0 < power < 1):
                errors.append(f"Statistical power must be between 0 and 1, got {power}")
            elif power < 0.8:
                warnings.append(f"Statistical power ({power}) below conventional 0.8")

        # Sample size
        if "sample_size" in params:
            n = params["sample_size"]
            if n < 1:
                errors.append(f"Sample size must be positive: {n}")
            elif n < 30:
                warnings.append(f"Small sample size ({n}) may violate normality assumptions")

        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )

        self.validation_history.append(result)
        return result

    def validate_all(
        self,
        apgi_params: Optional[Dict] = None,
        experimental_config: Optional[Dict] = None,
        neural_thresholds: Optional[Dict] = None,
    ) -> ValidationResult:
        """Validate all parameter types at once.

        Args:
            apgi_params: APGI equation parameters
            experimental_config: Experimental configuration
            neural_thresholds: Neural signature thresholds

        Returns:
            Combined ValidationResult
        """
        all_errors = []
        all_warnings = []
        all_suggestions = []

        if apgi_params:
            result = self.validate_apgi_parameters(**apgi_params)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            all_suggestions.extend(result.suggestions)

        if experimental_config:
            result = self.validate_experimental_config(**experimental_config)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            all_suggestions.extend(result.suggestions)

        if neural_thresholds:
            result = self.validate_neural_thresholds(**neural_thresholds)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            all_suggestions.extend(result.suggestions)

        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            suggestions=all_suggestions,
        )

    def validate_retry_config(self, **config: Any) -> ValidationResult:
        """Validate retry configuration parameters."""
        errors = []
        for param_name, value in config.items():
            if param_name in self.RETRY_PARAM_RANGES:
                min_val, max_val, desc = self.RETRY_PARAM_RANGES[param_name]
                if not isinstance(value, (int, float, np.number)):
                    errors.append(f"{param_name}: Must be numeric")
                    continue
                if not (min_val <= float(value) <= max_val):
                    errors.append(
                        f"{param_name}: Value {value} outside range [{min_val}, {max_val}]. {desc}"
                    )

        result = ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=[], suggestions=[]
        )
        self.validation_history.append(result)
        return result

    def validate_performance_thresholds(self, **config: Any) -> ValidationResult:
        """Validate performance threshold parameters."""
        errors = []
        for param_name, value in config.items():
            if param_name in self.PERFORMANCE_PARAM_RANGES:
                min_val, max_val, desc = self.PERFORMANCE_PARAM_RANGES[param_name]
                if not isinstance(value, (int, float, np.number)):
                    errors.append(f"{param_name}: Must be numeric")
                    continue
                if not (min_val <= float(value) <= max_val):
                    errors.append(
                        f"{param_name}: Value {value} outside range [{min_val}, {max_val}]. {desc}"
                    )

        result = ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=[], suggestions=[]
        )
        self.validation_history.append(result)
        return result

    def validate_stimulus_parameters(self, **config: Any) -> ValidationResult:
        """Validate stimulus generation parameters."""
        errors = []
        for param_name, value in config.items():
            if param_name in self.STIMULUS_PARAM_RANGES:
                min_val, max_val, desc = self.STIMULUS_PARAM_RANGES[param_name]
                if not isinstance(value, (int, float, np.number)):
                    errors.append(f"{param_name}: Must be numeric")
                    continue
                if not (min_val <= float(value) <= max_val):
                    errors.append(
                        f"{param_name}: Value {value} outside range [{min_val}, {max_val}]. {desc}"
                    )

        result = ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=[], suggestions=[]
        )
        self.validation_history.append(result)
        return result

    def get_parameter_info(self, param_name: str) -> str:
        """Get detailed information about a parameter.

        Args:
            param_name: Name of parameter

        Returns:
            Formatted information string
        """
        # Check all parameter dictionaries
        for param_dict, category in [
            (self.APGI_PARAM_RANGES, "APGI Equation Parameter"),
            (self.EXPERIMENTAL_PARAM_RANGES, "Experimental Parameter"),
            (self.NEURAL_SIGNATURE_RANGES, "Neural Signature Threshold"),
        ]:
            if param_name in param_dict:
                min_val, max_val, description = param_dict[param_name]
                return (
                    f"{category}: {param_name}\n"
                    f"Valid range: [{min_val}, {max_val}]\n"
                    f"Description: {description}"
                )

        return f"Unknown parameter: {param_name}"

    def get_validation_summary(self) -> str:
        """Get summary of all validation history."""
        if not self.validation_history:
            return "No validations performed yet"

        total_validations = len(self.validation_history)
        successful = sum(1 for v in self.validation_history if v.is_valid)
        failed = total_validations - successful

        return (
            f"Validation Summary:\n"
            f"  Total validations: {total_validations}\n"
            f"  Successful: {successful}\n"
            f"  Failed: {failed}\n"
            f"  Success rate: {successful / total_validations * 100:.1f}%"
        )


# Global validator instance
_validator = None


def get_validator() -> ParameterValidator:
    """Get global parameter validator instance."""
    global _validator
    if _validator is None:
        _validator = ParameterValidator()
    return _validator
