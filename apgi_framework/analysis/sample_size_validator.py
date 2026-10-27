"""Sample size validation and power reporting for APGI falsification testing.

This module provides comprehensive sample size validation, power analysis
reporting, and statistical power warnings for ensuring adequate statistical
power in APGI framework validation studies, with special requirements for
soma-bias testing (n > 100).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Union

import numpy as np

from .replication_tracker import PowerAnalyzer


class ValidationStatus(Enum):
    """Status of sample size validation."""

    ADEQUATE = "adequate"
    INADEQUATE = "inadequate"
    MARGINAL = "marginal"
    EXCESSIVE = "excessive"


class TestRequirement(Enum):
    """Specific test requirements for APGI framework."""

    SOMA_BIAS_MINIMUM = "soma_bias_minimum"  # n > 100
    NEURAL_SIGNATURE_MINIMUM = "neural_signature_minimum"  # n > 20
    FALSIFICATION_MINIMUM = "falsification_minimum"  # n > 30
    REPLICATION_MINIMUM = "replication_minimum"  # n > 50
    GENERAL_MINIMUM = "general_minimum"  # n > 10


@dataclass
class SampleSizeRequirement:
    """Container for sample size requirements."""

    test_type: str
    minimum_n: int
    recommended_n: int
    power_threshold: float
    effect_size_assumption: float
    justification: str


@dataclass
class ValidationResult:
    """Result of sample size validation."""

    test_type: str
    current_n: int
    required_n: int
    recommended_n: int
    status: ValidationStatus
    power: float
    effect_size: float
    warnings: List[str]
    recommendations: List[str]
    meets_requirements: bool


@dataclass
class PowerReport:
    """Comprehensive power analysis report."""

    study_id: str
    test_configurations: List[Dict[str, Any]]
    validation_results: List[ValidationResult]
    overall_adequacy: bool
    critical_issues: List[str]
    recommendations: List[str]
    power_summary: Dict[str, Union[float, int]]


class SampleSizeValidator:
    """Comprehensive sample size validation for APGI framework studies.

    Provides validation against specific requirements for different types
    of APGI falsification tests, with special attention to soma-bias testing
    requirements and statistical power considerations.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        power_threshold: float = 0.8,
        strict_mode: bool = True,
    ):
        """Initialize the sample size validator.

        Args:
            alpha: Type I error rate
            power_threshold: Minimum acceptable statistical power
            strict_mode: Whether to apply strict APGI framework requirements
        """
        self.alpha = alpha
        self.power_threshold = power_threshold
        self.strict_mode = strict_mode
        self.power_analyzer = PowerAnalyzer(alpha=alpha)

        # Define APGI-specific requirements
        self.requirements = self._define_apgi_requirements()

    def validate_soma_bias_sample_size(
        self, n_participants: int, expected_effect_size: float = 0.3
    ) -> ValidationResult:
        """Validate sample size for soma-bias testing (requires n > 100).

        Args:
            n_participants: Number of participants
            expected_effect_size: Expected Cohen's d for soma-bias effect

        Returns:
            ValidationResult with validation details
        """
        requirement = self.requirements[TestRequirement.SOMA_BIAS_MINIMUM]

        # Check minimum requirement
        meets_minimum = n_participants > requirement.minimum_n

        # Calculate power for current sample size
        power_result = self.power_analyzer.t_test_power(
            effect_size=expected_effect_size,
            sample_size=n_participants,
            test_type="two_sample",
        )

        # Determine status
        if not meets_minimum:
            status = ValidationStatus.INADEQUATE
        elif power_result.power < self.power_threshold:
            status = ValidationStatus.MARGINAL
        elif n_participants >= requirement.recommended_n:
            status = ValidationStatus.ADEQUATE
        else:
            status = ValidationStatus.MARGINAL

        # Generate warnings and recommendations
        warnings_list = []
        recommendations = []

        if not meets_minimum:
            warnings_list.append(
                f"Sample size ({n_participants}) below APGI framework minimum (>{requirement.minimum_n})"
            )
            recommendations.append(f"Increase sample size to at least {requirement.minimum_n + 1}")

        if power_result.power < self.power_threshold:
            warnings_list.append(
                f"Statistical power ({power_result.power:.3f}) below threshold ({self.power_threshold})"
            )

            # Calculate required sample size for adequate power
            required_power_result = self.power_analyzer.t_test_power(
                effect_size=expected_effect_size,
                power=self.power_threshold,
                test_type="two_sample",
            )
            recommendations.append(
                f"Increase sample size to {required_power_result.sample_size} for adequate power"
            )

        if n_participants > 500:
            warnings_list.append("Very large sample size - consider feasibility and cost")

        return ValidationResult(
            test_type="soma_bias",
            current_n=n_participants,
            required_n=max(
                requirement.minimum_n + 1,
                self.power_analyzer.t_test_power(
                    expected_effect_size,
                    power=self.power_threshold,
                    test_type="two_sample",
                ).sample_size,
            ),
            recommended_n=requirement.recommended_n,
            status=status,
            power=power_result.power,
            effect_size=expected_effect_size,
            warnings=warnings_list,
            recommendations=recommendations,
            meets_requirements=meets_minimum and power_result.power >= self.power_threshold,
        )

    def validate_neural_signature_sample_size(
        self,
        n_participants: int,
        n_trials_per_participant: int,
        expected_effect_size: float = 0.5,
    ) -> ValidationResult:
        """Validate sample size for neural signature detection.

        Args:
            n_participants: Number of participants
            n_trials_per_participant: Number of trials per participant
            expected_effect_size: Expected effect size for neural signatures

        Returns:
            ValidationResult with validation details
        """
        requirement = self.requirements[TestRequirement.NEURAL_SIGNATURE_MINIMUM]

        # Check minimum requirement
        meets_minimum = n_participants >= requirement.minimum_n

        # Calculate power
        power_result = self.power_analyzer.t_test_power(
            effect_size=expected_effect_size,
            sample_size=n_participants,
            test_type="one_sample",
        )

        # Determine status
        if not meets_minimum:
            status = ValidationStatus.INADEQUATE
        elif power_result.power < self.power_threshold:
            status = ValidationStatus.MARGINAL
        elif n_participants >= requirement.recommended_n:
            status = ValidationStatus.ADEQUATE
        else:
            status = ValidationStatus.MARGINAL

        # Generate warnings and recommendations
        warnings_list = []
        recommendations = []

        if not meets_minimum:
            warnings_list.append(
                f"Participant count ({n_participants}) below minimum ({requirement.minimum_n})"
            )
            recommendations.append(f"Increase participants to at least {requirement.minimum_n}")

        if n_trials_per_participant < 50:
            warnings_list.append(
                f"Low trial count ({n_trials_per_participant}) may reduce signal-to-noise ratio"
            )
            recommendations.append("Consider increasing trials per participant to 50-100")

        if power_result.power < self.power_threshold:
            warnings_list.append(f"Statistical power ({power_result.power:.3f}) below threshold")

            required_power_result = self.power_analyzer.t_test_power(
                effect_size=expected_effect_size,
                power=self.power_threshold,
                test_type="one_sample",
            )
            recommendations.append(
                f"Increase participants to {required_power_result.sample_size} for adequate power"
            )

        return ValidationResult(
            test_type="neural_signature",
            current_n=n_participants,
            required_n=max(
                requirement.minimum_n,
                self.power_analyzer.t_test_power(
                    expected_effect_size,
                    power=self.power_threshold,
                    test_type="one_sample",
                ).sample_size,
            ),
            recommended_n=requirement.recommended_n,
            status=status,
            power=power_result.power,
            effect_size=expected_effect_size,
            warnings=warnings_list,
            recommendations=recommendations,
            meets_requirements=meets_minimum and power_result.power >= self.power_threshold,
        )

    def validate_falsification_sample_size(
        self,
        n_participants: int,
        expected_effect_size: float = 0.4,
        test_type: str = "primary",
    ) -> ValidationResult:
        """Validate sample size for falsification testing.

        Args:
            n_participants: Number of participants
            expected_effect_size: Expected effect size for falsification
            test_type: Type of falsification test ("primary", "secondary")

        Returns:
            ValidationResult with validation details
        """
        requirement = self.requirements[TestRequirement.FALSIFICATION_MINIMUM]

        # Adjust requirements based on test type
        if test_type == "primary":
            # Primary falsification requires higher power
            adjusted_power_threshold = 0.9
            adjusted_minimum = requirement.minimum_n
        else:
            # Secondary falsification tests
            adjusted_power_threshold = self.power_threshold
            adjusted_minimum = max(20, requirement.minimum_n - 10)

        # Check minimum requirement
        meets_minimum = n_participants >= adjusted_minimum

        # Calculate power
        power_result = self.power_analyzer.t_test_power(
            effect_size=expected_effect_size,
            sample_size=n_participants,
            test_type="two_sample",
        )

        # Determine status
        if not meets_minimum:
            status = ValidationStatus.INADEQUATE
        elif power_result.power < adjusted_power_threshold:
            status = ValidationStatus.MARGINAL
        elif n_participants >= requirement.recommended_n:
            status = ValidationStatus.ADEQUATE
        else:
            status = ValidationStatus.MARGINAL

        # Generate warnings and recommendations
        warnings_list = []
        recommendations = []

        if not meets_minimum:
            warnings_list.append(
                f"Sample size ({n_participants}) below minimum for {test_type} falsification ({adjusted_minimum})"
            )
            recommendations.append(f"Increase sample size to at least {adjusted_minimum}")

        if power_result.power < adjusted_power_threshold:
            warnings_list.append(
                f"Power ({power_result.power:.3f}) below {test_type} falsification threshold ({adjusted_power_threshold})"
            )

            required_power_result = self.power_analyzer.t_test_power(
                effect_size=expected_effect_size,
                power=adjusted_power_threshold,
                test_type="two_sample",
            )
            recommendations.append(
                f"Increase sample size to {required_power_result.sample_size} for adequate power"
            )

        if test_type == "primary" and n_participants < 50:
            warnings_list.append(
                "Primary falsification tests should have robust sample sizes (≥50)"
            )

        return ValidationResult(
            test_type=f"{test_type}_falsification",
            current_n=n_participants,
            required_n=max(
                adjusted_minimum,
                self.power_analyzer.t_test_power(
                    expected_effect_size,
                    power=adjusted_power_threshold,
                    test_type="two_sample",
                ).sample_size,
            ),
            recommended_n=requirement.recommended_n,
            status=status,
            power=power_result.power,
            effect_size=expected_effect_size,
            warnings=warnings_list,
            recommendations=recommendations,
            meets_requirements=meets_minimum and power_result.power >= adjusted_power_threshold,
        )

    def validate_replication_sample_size(
        self, n_participants_per_lab: int, n_labs: int, original_effect_size: float
    ) -> ValidationResult:
        """Validate sample size for replication studies.

        Args:
            n_participants_per_lab: Participants per laboratory
            n_labs: Number of replicating laboratories
            original_effect_size: Effect size from original study

        Returns:
            ValidationResult with validation details
        """
        requirement = self.requirements[TestRequirement.REPLICATION_MINIMUM]

        # Total effective sample size
        total_n = n_participants_per_lab * n_labs

        # Check minimum requirements
        meets_minimum_per_lab = n_participants_per_lab >= requirement.minimum_n
        meets_minimum_labs = n_labs >= 3  # Minimum for replication analysis

        # Calculate power for individual lab
        power_result = self.power_analyzer.t_test_power(
            effect_size=original_effect_size,
            sample_size=n_participants_per_lab,
            test_type="two_sample",
        )

        # Calculate power for meta-analysis (combined effect)
        meta_power_result = self.power_analyzer.t_test_power(
            effect_size=original_effect_size,
            sample_size=total_n,
            test_type="two_sample",
        )

        # Determine status
        if not (meets_minimum_per_lab and meets_minimum_labs):
            status = ValidationStatus.INADEQUATE
        elif power_result.power < 0.5:  # Lower threshold for individual replications
            status = ValidationStatus.MARGINAL
        elif meta_power_result.power >= 0.95:
            status = ValidationStatus.ADEQUATE
        else:
            status = ValidationStatus.MARGINAL

        # Generate warnings and recommendations
        warnings_list = []
        recommendations = []

        if not meets_minimum_per_lab:
            warnings_list.append(
                f"Sample size per lab ({n_participants_per_lab}) below minimum ({requirement.minimum_n})"
            )
            recommendations.append(
                f"Increase sample size per lab to at least {requirement.minimum_n}"
            )

        if not meets_minimum_labs:
            warnings_list.append(f"Number of labs ({n_labs}) below minimum (3)")
            recommendations.append("Include at least 3 independent laboratories")

        if power_result.power < 0.5:
            warnings_list.append(f"Individual lab power ({power_result.power:.3f}) very low")
            recommendations.append(
                "Consider increasing sample size per lab for better individual replication power"
            )

        if meta_power_result.power < 0.9:
            warnings_list.append(f"Meta-analytic power ({meta_power_result.power:.3f}) below 0.9")
            recommendations.append("Consider increasing total sample size across labs")

        return ValidationResult(
            test_type="replication",
            current_n=n_participants_per_lab,
            required_n=requirement.minimum_n,
            recommended_n=requirement.recommended_n,
            status=status,
            power=power_result.power,
            effect_size=original_effect_size,
            warnings=warnings_list,
            recommendations=recommendations,
            meets_requirements=meets_minimum_per_lab
            and meets_minimum_labs
            and power_result.power >= 0.5,
        )

    def generate_power_report(
        self, study_configurations: List[Dict[str, Any]], study_id: str = "APGI_Study"
    ) -> PowerReport:
        """Generate comprehensive power analysis report for study.

        Args:
            study_configurations: List of test configurations
            study_id: Identifier for the study

        Returns:
            PowerReport with comprehensive analysis
        """
        validation_results = []
        critical_issues = []
        recommendations = []

        # Validate each configuration
        for config in study_configurations:
            test_type = config.get("test_type", "general")
            n_participants = config.get("n_participants", 0)
            effect_size = config.get("effect_size", 0.3)

            if test_type == "soma_bias":
                result = self.validate_soma_bias_sample_size(n_participants, effect_size)
            elif test_type == "neural_signature":
                n_trials = config.get("n_trials_per_participant", 100)
                result = self.validate_neural_signature_sample_size(
                    n_participants, n_trials, effect_size
                )
            elif test_type in ["primary_falsification", "secondary_falsification"]:
                falsification_type = "primary" if "primary" in test_type else "secondary"
                result = self.validate_falsification_sample_size(
                    n_participants, effect_size, falsification_type
                )
            elif test_type == "replication":
                n_labs = config.get("n_labs", 1)
                result = self.validate_replication_sample_size(n_participants, n_labs, effect_size)
            else:
                # General validation
                result = self._validate_general_sample_size(n_participants, effect_size)

            validation_results.append(result)

            # Collect critical issues
            if result.status == ValidationStatus.INADEQUATE:
                critical_issues.extend(result.warnings)

            # Collect recommendations
            recommendations.extend(result.recommendations)

        # Calculate overall adequacy
        overall_adequacy = all(result.meets_requirements for result in validation_results)

        # Generate power summary
        power_summary: Dict[str, Union[float, int]] = {
            "mean_power": float(np.mean([result.power for result in validation_results])),
            "min_power": min([result.power for result in validation_results]),
            "max_power": max([result.power for result in validation_results]),
            "adequate_tests": sum(1 for result in validation_results if result.meets_requirements),
            "total_tests": len(validation_results),
        }

        # Add overall recommendations
        if not overall_adequacy:
            recommendations.append(
                "Study design requires revision to meet APGI framework standards"
            )

        if power_summary["min_power"] < 0.5:
            recommendations.append("Some tests have very low power - consider major design changes")

        return PowerReport(
            study_id=study_id,
            test_configurations=study_configurations,
            validation_results=validation_results,
            overall_adequacy=overall_adequacy,
            critical_issues=list(set(critical_issues)),  # Remove duplicates
            recommendations=list(set(recommendations)),
            power_summary=power_summary,
        )

    def _define_apgi_requirements(self) -> Dict[TestRequirement, SampleSizeRequirement]:
        """Define APGI framework-specific sample size requirements."""
        return {
            TestRequirement.SOMA_BIAS_MINIMUM: SampleSizeRequirement(
                test_type="soma_bias",
                minimum_n=100,
                recommended_n=150,
                power_threshold=0.8,
                effect_size_assumption=0.3,
                justification="APGI framework requires n > 100 for soma-bias testing to detect interoceptive vs exteroceptive differences",
            ),
            TestRequirement.NEURAL_SIGNATURE_MINIMUM: SampleSizeRequirement(
                test_type="neural_signature",
                minimum_n=20,
                recommended_n=40,
                power_threshold=0.8,
                effect_size_assumption=0.5,
                justification="Neural signature detection requires adequate signal-to-noise ratio",
            ),
            TestRequirement.FALSIFICATION_MINIMUM: SampleSizeRequirement(
                test_type="falsification",
                minimum_n=30,
                recommended_n=60,
                power_threshold=0.8,
                effect_size_assumption=0.4,
                justification="Falsification tests require robust sample sizes for definitive conclusions",
            ),
            TestRequirement.REPLICATION_MINIMUM: SampleSizeRequirement(
                test_type="replication",
                minimum_n=50,
                recommended_n=80,
                power_threshold=0.8,
                effect_size_assumption=0.3,
                justification="Replication studies need adequate power to detect original effects",
            ),
            TestRequirement.GENERAL_MINIMUM: SampleSizeRequirement(
                test_type="general",
                minimum_n=10,
                recommended_n=30,
                power_threshold=0.8,
                effect_size_assumption=0.5,
                justification="General statistical tests minimum requirements",
            ),
        }

    def _validate_general_sample_size(
        self, n_participants: int, effect_size: float
    ) -> ValidationResult:
        """Validate general sample size requirements."""
        requirement = self.requirements[TestRequirement.GENERAL_MINIMUM]

        meets_minimum = n_participants >= requirement.minimum_n

        power_result = self.power_analyzer.t_test_power(
            effect_size=effect_size, sample_size=n_participants, test_type="two_sample"
        )

        if not meets_minimum:
            status = ValidationStatus.INADEQUATE
        elif power_result.power < self.power_threshold:
            status = ValidationStatus.MARGINAL
        else:
            status = ValidationStatus.ADEQUATE

        warnings_list = []
        recommendations = []

        if not meets_minimum:
            warnings_list.append(
                f"Sample size ({n_participants}) below minimum ({requirement.minimum_n})"
            )
            recommendations.append(f"Increase sample size to at least {requirement.minimum_n}")

        if power_result.power < self.power_threshold:
            warnings_list.append(f"Statistical power ({power_result.power:.3f}) below threshold")

        return ValidationResult(
            test_type="general",
            current_n=n_participants,
            required_n=max(
                requirement.minimum_n,
                self.power_analyzer.t_test_power(
                    effect_size, power=self.power_threshold, test_type="two_sample"
                ).sample_size,
            ),
            recommended_n=requirement.recommended_n,
            status=status,
            power=power_result.power,
            effect_size=effect_size,
            warnings=warnings_list,
            recommendations=recommendations,
            meets_requirements=meets_minimum and power_result.power >= self.power_threshold,
        )
