"""Data validation and integrity checking for the APGI Framework.

Provides comprehensive validation of experimental data, metadata,
and storage integrity with automated quality assessment.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..exceptions import APGIFrameworkError
from .data_models import ExperimentalDataset, ExperimentMetadata


class ValidationError(APGIFrameworkError):
    """Errors in data validation."""


class DataValidator:
    """Comprehensive data validation and integrity checking system.

    Validates experimental data structure, content, metadata consistency,
    and storage integrity with detailed quality assessment.
    """

    def __init__(self, strict_mode: bool = True):
        """Initialize data validator.

        Args:
            strict_mode: Whether to enforce strict validation rules
        """
        self.strict_mode = strict_mode
        self.validation_rules = self._load_validation_rules()
        self.quality_thresholds = self._load_quality_thresholds()

    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules configuration."""
        return {
            "required_metadata_fields": [
                "experiment_id",
                "experiment_name",
                "created_at",
                "researcher",
                "n_participants",
                "n_trials",
            ],
            "required_data_fields": [
                "apgi_parameters",
                "neural_signatures",
                "consciousness_assessments",
            ],
            "parameter_ranges": {
                "extero_precision": (0.1, 10.0),
                "intero_precision": (0.1, 10.0),
                "extero_error": (-5.0, 5.0),
                "intero_error": (-5.0, 5.0),
                "somatic_gain": (0.1, 5.0),
                "threshold": (0.1, 10.0),
                "steepness": (0.1, 10.0),
            },
            "signature_ranges": {
                "p3b_amplitude": (-2.0, 20.0),  # μV
                "p3b_latency": (200, 600),  # ms
                "gamma_plv": (0.0, 1.0),
                "gamma_duration": (50, 1000),  # ms
                "bold_z_score": (-5.0, 10.0),
                "pci_value": (0.0, 1.0),
            },
            "consciousness_ranges": {
                "forced_choice_accuracy": (0.0, 1.0),
                "confidence_rating": (0.0, 1.0),
                "wagering_behavior": (0.0, 1.0),
                "metacognitive_sensitivity": (-2.0, 2.0),
            },
            "min_sample_sizes": {
                "primary_falsification": 20,
                "consciousness_without_ignition": 50,
                "threshold_insensitivity": 30,
                "soma_bias": 100,
            },
        }

    def _load_quality_thresholds(self) -> Dict[str, float]:
        """Load data quality thresholds."""
        return {
            "completeness_threshold": 0.95,
            "consistency_threshold": 0.90,
            "accuracy_threshold": 0.85,
            "reliability_threshold": 0.80,
            "overall_quality_threshold": 0.85,
        }

    def validate_dataset(self, dataset: ExperimentalDataset) -> Dict[str, Any]:
        """Comprehensive dataset validation.

        Args:
            dataset: ExperimentalDataset to validate

        Returns:
            Dict containing validation results and quality metrics
        """
        validation_result: Dict[str, Any] = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "quality_score": 1.0,
            "completeness": 1.0,
            "consistency": 1.0,
            "accuracy": 1.0,
            "reliability": 1.0,
            "validation_timestamp": datetime.now(),
            "details": {},
        }

        try:
            # Validate metadata
            metadata_result = self.validate_metadata(dataset.metadata)
            validation_result["details"]["metadata"] = metadata_result

            errors_list: List[str] = validation_result["errors"]
            warnings_list: List[str] = validation_result["warnings"]

            if not metadata_result["is_valid"]:
                validation_result["is_valid"] = False
                errors_list.extend(metadata_result["errors"])

            warnings_list.extend(metadata_result["warnings"])

            # Validate data structure
            structure_result = self.validate_data_structure(dataset.data)
            validation_result["details"]["structure"] = structure_result

            if not structure_result["is_valid"]:
                validation_result["is_valid"] = False
                validation_result["errors"].extend(structure_result["errors"])

            # Validate data content
            content_result = self.validate_data_content(dataset.data)
            validation_result["details"]["content"] = content_result

            if not content_result["is_valid"]:
                validation_result["is_valid"] = False
                validation_result["errors"].extend(content_result["errors"])

            # Calculate quality metrics
            quality_metrics = self.calculate_quality_metrics(dataset)
            validation_result.update(quality_metrics)

            # Validate sample sizes
            sample_result = self.validate_sample_sizes(dataset)
            validation_result["details"]["sample_sizes"] = sample_result

            if not sample_result["is_valid"]:
                validation_result["warnings"].extend(sample_result["warnings"])

            # Overall quality assessment
            validation_result["quality_score"] = self._calculate_overall_quality(
                validation_result["completeness"],
                validation_result["consistency"],
                validation_result["accuracy"],
                validation_result["reliability"],
            )

            # Update dataset metadata with validation results
            dataset.metadata.data_quality_score = validation_result["quality_score"]
            dataset.metadata.completeness_percentage = validation_result["completeness"] * 100
            dataset.metadata.validation_status = (
                "validated" if validation_result["is_valid"] else "failed"
            )

        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Validation failed: {str(e)}")
            validation_result["quality_score"] = 0.0

        return validation_result

    def validate_metadata(self, metadata: ExperimentMetadata) -> Dict[str, Any]:
        """Validate experiment metadata."""
        result: Dict[str, Any] = {"is_valid": True, "errors": [], "warnings": []}

        # Check required fields
        for field in self.validation_rules["required_metadata_fields"]:
            if not hasattr(metadata, field) or getattr(metadata, field) is None:
                result["errors"].append(f"Missing required metadata field: {field}")
                result["is_valid"] = False

        # Validate field values
        if metadata.n_participants <= 0:
            result["errors"].append("Number of participants must be positive")
            result["is_valid"] = False

        if metadata.n_trials <= 0:
            result["errors"].append("Number of trials must be positive")
            result["is_valid"] = False

        # Check date consistency
        if metadata.updated_at < metadata.created_at:
            result["errors"].append("Updated date cannot be before created date")
            result["is_valid"] = False

        # Validate experiment name
        if not metadata.experiment_name or len(metadata.experiment_name.strip()) == 0:
            result["warnings"].append("Experiment name is empty or whitespace")

        # Check for reasonable date ranges
        now = datetime.now()
        if metadata.created_at > now + timedelta(days=1):
            result["warnings"].append("Created date is in the future")

        if metadata.created_at < datetime(2020, 1, 1):
            result["warnings"].append("Created date is unusually old")

        return result

    def validate_data_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data structure and required fields."""
        result: Dict[str, Any] = {"is_valid": True, "errors": [], "warnings": []}

        if not data:
            result["errors"].append("Data dictionary is empty")
            result["is_valid"] = False
            return result

        # Check for required top-level fields
        for field in self.validation_rules["required_data_fields"]:
            if field not in data:
                result["errors"].append(f"Missing required data field: {field}")
                result["is_valid"] = False

        # Validate APGI parameters structure
        if "apgi_parameters" in data:
            apgi_result = self._validate_apgi_parameters_structure(data["apgi_parameters"])
            if not apgi_result["is_valid"]:
                result["errors"].extend(apgi_result["errors"])
                result["is_valid"] = False
            result["warnings"].extend(apgi_result["warnings"])

        # Validate neural signatures structure
        if "neural_signatures" in data:
            neural_result = self._validate_neural_signatures_structure(data["neural_signatures"])
            if not neural_result["is_valid"]:
                result["errors"].extend(neural_result["errors"])
                result["is_valid"] = False
            result["warnings"].extend(neural_result["warnings"])

        # Validate consciousness assessments structure
        if "consciousness_assessments" in data:
            consciousness_result = self._validate_consciousness_structure(
                data["consciousness_assessments"]
            )
            if not consciousness_result["is_valid"]:
                result["errors"].extend(consciousness_result["errors"])
                result["is_valid"] = False
            result["warnings"].extend(consciousness_result["warnings"])
        return result

    def validate_numeric_array(self, data: Any) -> tuple[bool, list[str]]:
        """Validate that data is a numeric array without NaN or Inf.

        Args:
            data: Data to validate

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        if not isinstance(data, (np.ndarray, list)):
            errors.append("Data must be a numpy array or list")
            return False, errors

        arr = np.asanyarray(data)
        if arr.size == 0:
            errors.append("Array is empty")
            return False, errors

        if not np.issubdtype(arr.dtype, np.number):
            errors.append("Array must be numeric")
            return False, errors

        if np.any(np.isnan(arr)):
            errors.append("Array contains NaN values")

        if np.any(np.isinf(arr)):
            errors.append("Array contains infinity values")

        return len(errors) == 0, errors

    def validate_required_fields(
        self, data: dict[str, Any], required_fields: list[str]
    ) -> tuple[bool, list[str]]:
        """Validate that all required fields are present in data.

        Args:
            data: Data dictionary
            required_fields: List of required field names

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        return len(errors) == 0, errors

    def validate_types(
        self, data: dict[str, Any], type_schema: dict[str, type]
    ) -> tuple[bool, list[str]]:
        """Validate data types based on a schema.

        Args:
            data: Data dictionary
            type_schema: Dictionary mapping fields to expected types

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        for field, expected_type in type_schema.items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    errors.append(
                        f"Field {field} must be {expected_type.__name__}, "
                        f"got {type(data[field]).__name__}"
                    )

        return len(errors) == 0, errors

    def validate_ranges(
        self, data: dict[str, Any], ranges: dict[str, tuple[float, float]]
    ) -> tuple[bool, list[str]]:
        """Validate that values are within specified ranges.

        Args:
            data: Data dictionary
            ranges: Dictionary mapping fields to (min, max) tuples

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        for field, (min_val, max_val) in ranges.items():
            if field in data:
                value = data[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"Field {field} must be numeric")
                elif not (min_val <= value <= max_val):
                    errors.append(
                        f"Field {field} ({value}) outside valid range [{min_val}, {max_val}]"
                    )

        return len(errors) == 0, errors

    def validate_data_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data content and value ranges."""
        result: Dict[str, Any] = {"is_valid": True, "errors": [], "warnings": []}

        # Validate APGI parameter values
        if "apgi_parameters" in data:
            param_result = self._validate_parameter_values(data["apgi_parameters"])
            if not param_result["is_valid"]:
                result["errors"].extend(param_result["errors"])
                result["is_valid"] = False
            result["warnings"].extend(param_result["warnings"])

        # Validate neural signature values
        if "neural_signatures" in data:
            signature_result = self._validate_signature_values(data["neural_signatures"])
            if not signature_result["is_valid"]:
                result["errors"].extend(signature_result["errors"])
                result["is_valid"] = False
            result["warnings"].extend(signature_result["warnings"])

        # Validate consciousness assessment values
        if "consciousness_assessments" in data:
            consciousness_result = self._validate_consciousness_values(
                data["consciousness_assessments"]
            )
            if not consciousness_result["is_valid"]:
                result["errors"].extend(consciousness_result["errors"])
                result["is_valid"] = False
            result["warnings"].extend(consciousness_result["warnings"])
        return result

    def _validate_apgi_parameters_structure(self, parameters: Any) -> Dict[str, Any]:
        """Validate APGI parameters structure."""
        result: Dict[str, Any] = {"is_valid": True, "errors": [], "warnings": []}

        if not isinstance(parameters, (dict, list)):
            result["errors"].append("APGI parameters must be dict or list")
            result["is_valid"] = False
            return result

        # Handle list of parameter sets
        if isinstance(parameters, list):
            for i, param_set in enumerate(parameters):
                if not isinstance(param_set, dict):
                    result["errors"].append(f"Parameter set {i} must be a dictionary")
                    result["is_valid"] = False

        return result

    def _validate_neural_signatures_structure(self, signatures: Any) -> Dict[str, Any]:
        """Validate neural signatures structure."""
        result: Dict[str, Any] = {"is_valid": True, "errors": [], "warnings": []}

        if not isinstance(signatures, (dict, list)):
            result["errors"].append("Neural signatures must be dict or list")
            result["is_valid"] = False
            return result

        return result

    def _validate_consciousness_structure(self, assessments: Any) -> Dict[str, Any]:
        """Validate consciousness assessments structure."""
        result: Dict[str, Any] = {"is_valid": True, "errors": [], "warnings": []}

        if not isinstance(assessments, (dict, list)):
            result["errors"].append("Consciousness assessments must be dict or list")
            result["is_valid"] = False
            return result

        return result

    def _validate_parameter_values(self, parameters: Any) -> Dict[str, Any]:
        """Validate APGI parameter value ranges."""
        result: Dict[str, Any] = {"is_valid": True, "errors": [], "warnings": []}

        param_ranges = self.validation_rules["parameter_ranges"]

        def validate_single_params(params: Dict[str, Any], index: str = "") -> None:
            for param_name, (min_val, max_val) in param_ranges.items():
                if param_name in params:
                    value = params[param_name]
                    if not isinstance(value, (int, float)):
                        result["errors"].append(f"Parameter {param_name}{index} must be numeric")
                        result["is_valid"] = False
                    elif not (min_val <= value <= max_val):
                        result["errors"].append(
                            f"Parameter {param_name}{index} ({value}) outside valid range [{min_val}, {max_val}]"
                        )
                        result["is_valid"] = False

        if isinstance(parameters, dict):
            validate_single_params(parameters)
        elif isinstance(parameters, list):
            for i, param_set in enumerate(parameters):
                if isinstance(param_set, dict):
                    validate_single_params(param_set, f"[{i}]")

        return result

    def _validate_signature_values(self, signatures: Any) -> Dict[str, Any]:
        """Validate neural signature value ranges."""
        result: Dict[str, Any] = {"is_valid": True, "errors": [], "warnings": []}

        signature_ranges = self.validation_rules["signature_ranges"]

        def validate_single_signatures(sigs: Dict[str, Any], index: str = "") -> None:
            for sig_name, (min_val, max_val) in signature_ranges.items():
                if sig_name in sigs:
                    value = sigs[sig_name]
                    if not isinstance(value, (int, float)):
                        result["errors"].append(f"Signature {sig_name}{index} must be numeric")
                        result["is_valid"] = False
                    elif not (min_val <= value <= max_val):
                        result["errors"].append(
                            f"Signature {sig_name}{index} ({value}) outside valid range [{min_val}, {max_val}]"
                        )
                        result["is_valid"] = False

        if isinstance(signatures, dict):
            validate_single_signatures(signatures)
        elif isinstance(signatures, list):
            for i, sig_set in enumerate(signatures):
                if isinstance(sig_set, dict):
                    validate_single_signatures(sig_set, f"[{i}]")

        return result

    def _validate_consciousness_values(self, assessments: Any) -> Dict[str, Any]:
        """Validate consciousness assessment value ranges."""
        result: Dict[str, Any] = {"is_valid": True, "errors": [], "warnings": []}

        consciousness_ranges = self.validation_rules["consciousness_ranges"]

        def validate_single_assessment(assess: Dict[str, Any], index: str = "") -> None:
            for assess_name, (min_val, max_val) in consciousness_ranges.items():
                if assess_name in assess:
                    value = assess[assess_name]
                    if not isinstance(value, (int, float)):
                        result["errors"].append(f"Assessment {assess_name}{index} must be numeric")
                        result["is_valid"] = False
                    elif not (min_val <= value <= max_val):
                        result["errors"].append(
                            f"Assessment {assess_name}{index} ({value}) outside valid range [{min_val}, {max_val}]"
                        )
                        result["is_valid"] = False

        if isinstance(assessments, dict):
            validate_single_assessment(assessments)
        elif isinstance(assessments, list):
            for i, assess_set in enumerate(assessments):
                if isinstance(assess_set, dict):
                    validate_single_assessment(assess_set, f"[{i}]")

        return result

    def validate_sample_sizes(self, dataset: ExperimentalDataset) -> Dict[str, Any]:
        """Validate sample sizes for different test types."""
        result: Dict[str, Any] = {"is_valid": True, "warnings": []}

        min_sizes = self.validation_rules["min_sample_sizes"]
        n_participants = dataset.metadata.n_participants

        # Determine test type from metadata or data
        test_type = dataset.metadata.category

        if test_type in min_sizes:
            min_required = min_sizes[test_type]
            if n_participants < min_required:
                result["warnings"].append(
                    f"Sample size ({n_participants}) below recommended minimum "
                    f"for {test_type} ({min_required})"
                )

        return result

    def calculate_quality_metrics(self, dataset: ExperimentalDataset) -> Dict[str, float]:
        """Calculate comprehensive quality metrics."""
        metrics = {
            "completeness": 1.0,
            "consistency": 1.0,
            "accuracy": 1.0,
            "reliability": 1.0,
        }

        # Calculate completeness
        metrics["completeness"] = self._calculate_completeness(dataset)

        # Calculate consistency
        metrics["consistency"] = self._calculate_consistency(dataset)

        # Calculate accuracy (based on validation results)
        metrics["accuracy"] = self._calculate_accuracy(dataset)

        # Calculate reliability (based on data variability)
        metrics["reliability"] = self._calculate_reliability(dataset)

        return metrics

    def _calculate_completeness(self, dataset: ExperimentalDataset) -> float:
        """Calculate data completeness score."""
        required_fields = self.validation_rules["required_data_fields"]
        present_fields = sum(1 for field in required_fields if field in dataset.data)

        completeness = present_fields / len(required_fields) if required_fields else 1.0

        # Adjust for missing values within present fields
        if dataset.data:
            total_values = 0
            missing_values = 0

            for field_data in dataset.data.values():
                if isinstance(field_data, list):
                    total_values += len(field_data)
                    missing_values += sum(1 for item in field_data if item is None)
                elif isinstance(field_data, dict):
                    total_values += len(field_data)
                    missing_values += sum(1 for item in field_data.values() if item is None)

            if total_values > 0:
                value_completeness = 1.0 - (missing_values / total_values)
                completeness = (completeness + value_completeness) / 2

        return completeness

    def _calculate_consistency(self, dataset: ExperimentalDataset) -> float:
        """Calculate data consistency score."""
        consistency = 1.0

        # Check for consistent data types and structures
        if "apgi_parameters" in dataset.data:
            params = dataset.data["apgi_parameters"]
            if isinstance(params, list) and len(params) > 1:
                # Check parameter consistency across trials
                first_keys = set(params[0].keys()) if params else set()
                for param_set in params[1:]:
                    if set(param_set.keys()) != first_keys:
                        consistency -= 0.1
                        break

        # Check temporal consistency
        if dataset.metadata.created_at and dataset.metadata.updated_at:
            if dataset.metadata.updated_at < dataset.metadata.created_at:
                consistency -= 0.2

        return max(0.0, consistency)

    def _calculate_accuracy(self, dataset: ExperimentalDataset) -> float:
        """Calculate data accuracy score based on validation."""
        accuracy = 1.0

        # This would be enhanced with domain-specific accuracy checks
        # For now, base it on parameter range compliance
        if "apgi_parameters" in dataset.data:
            param_ranges = self.validation_rules["parameter_ranges"]
            params = dataset.data["apgi_parameters"]

            if isinstance(params, list):
                total_params = 0
                valid_params = 0

                for param_set in params:
                    for param_name, value in param_set.items():
                        if param_name in param_ranges and value is not None:
                            total_params += 1
                            min_val, max_val = param_ranges[param_name]
                            if min_val <= value <= max_val:
                                valid_params += 1

                if total_params > 0:
                    accuracy = valid_params / total_params

        return accuracy

    def _calculate_reliability(self, dataset: ExperimentalDataset) -> float:
        """Calculate data reliability score."""
        reliability = 1.0

        # Check for reasonable variability in measurements
        if "neural_signatures" in dataset.data:
            signatures = dataset.data["neural_signatures"]

            if isinstance(signatures, list) and len(signatures) > 1:
                # Calculate coefficient of variation for key signatures
                for sig_name in ["p3b_amplitude", "gamma_plv", "pci_value"]:
                    values = [
                        sig.get(sig_name) for sig in signatures if sig.get(sig_name) is not None
                    ]

                    if len(values) > 1:
                        mean_val = np.mean(values)
                        std_val = np.std(values)

                        if mean_val != 0:
                            cv = std_val / abs(mean_val)
                            # Penalize extremely high or low variability
                            if cv > 2.0 or cv < 0.01:
                                reliability -= 0.1

        return max(0.0, reliability)

    def _calculate_overall_quality(
        self,
        completeness: float,
        consistency: float,
        accuracy: float,
        reliability: float,
    ) -> float:
        """Calculate overall quality score."""
        # Weighted average of quality metrics
        weights = {
            "completeness": 0.3,
            "consistency": 0.2,
            "accuracy": 0.3,
            "reliability": 0.2,
        }

        overall = (
            weights["completeness"] * completeness
            + weights["consistency"] * consistency
            + weights["accuracy"] * accuracy
            + weights["reliability"] * reliability
        )

        return overall

    def check_data_integrity(self, file_path: Path) -> Dict[str, Any]:
        """Check file integrity using checksums."""
        result: Dict[str, Any] = {
            "is_intact": True,
            "checksum": "",
            "size_bytes": 0,
            "errors": [],
        }

        try:
            if not file_path.exists():
                result["is_intact"] = False
                result["errors"].append(f"File does not exist: {file_path}")
                return result

            # Calculate checksum
            with open(file_path, "rb") as f:
                content = f.read()
                result["checksum"] = hashlib.sha256(content).hexdigest()
                result["size_bytes"] = len(content)

        except Exception as e:
            result["is_intact"] = False
            result["errors"].append(f"Failed to check integrity: {str(e)}")

        return result
