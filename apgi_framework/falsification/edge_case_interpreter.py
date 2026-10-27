"""Edge Case Interpreter.

Implements edge case classification and interpretation for APGI Framework boundaries.
Handles special cases like anesthesia awareness, blindsight, dreams, and locked-in syndrome
that may not fit standard falsification criteria.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from ..simulators.signature_validator import CombinedSignature

if TYPE_CHECKING:
    from . import ConsciousnessAssessment


class EdgeCaseType(Enum):
    """Types of edge cases for framework boundary testing."""

    ANESTHESIA_AWARENESS = "anesthesia_awareness"
    BLINDSIGHT = "blindsight"
    DREAMS = "dreams"
    LOCKED_IN_SYNDROME = "locked_in_syndrome"
    VEGETATIVE_STATE = "vegetative_state"
    MINIMALLY_CONSCIOUS_STATE = "minimally_conscious_state"
    SPLIT_BRAIN = "split_brain"
    HEMISPATIAL_NEGLECT = "hemispatial_neglect"
    NORMAL_CONTROL = "normal_control"


class FrameworkBoundary(Enum):
    """Framework boundary classifications."""

    WITHIN_FRAMEWORK = "within_framework"
    BOUNDARY_CASE = "boundary_case"
    OUTSIDE_FRAMEWORK = "outside_framework"
    FRAMEWORK_VIOLATION = "framework_violation"


@dataclass
class EdgeCaseProfile:
    """Profile defining characteristics of an edge case."""

    case_type: EdgeCaseType
    description: str

    # Expected neural signature patterns
    expected_p3b_range: Tuple[float, float]
    expected_gamma_range: Tuple[float, float]
    expected_bold_pattern: Dict[str, Tuple[float, float]]
    expected_pci_range: Tuple[float, float]

    # Expected consciousness characteristics
    subjective_report_probability: float
    forced_choice_accuracy_range: Tuple[float, float]
    metacognitive_sensitivity_range: Tuple[float, float]

    # Framework boundary expectations
    framework_boundary: FrameworkBoundary
    interpretation_notes: str


@dataclass
class EdgeCaseClassification:
    """Result of edge case classification."""

    case_type: EdgeCaseType
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]

    # Signature analysis
    signature_consistency: float
    consciousness_consistency: float

    # Framework boundary assessment
    framework_boundary: FrameworkBoundary
    boundary_confidence: float

    # Interpretation
    interpretation: str
    recommendations: List[str]


@dataclass
class EdgeCaseAnalysisResult:
    """Complete edge case analysis result."""

    analysis_id: str
    timestamp: datetime

    # Input data
    neural_signatures: CombinedSignature
    consciousness_assessment: "ConsciousnessAssessment"

    # Classification results
    primary_classification: EdgeCaseClassification
    alternative_classifications: List[EdgeCaseClassification]

    # Framework implications
    framework_implications: str
    falsification_relevance: str

    # Recommendations
    further_testing_needed: bool
    recommended_tests: List[str]


class EdgeCaseInterpreter:
    """Classifier and interpreter for edge cases in APGI Framework testing.

    Handles special neurological and clinical conditions that may not fit
    standard falsification criteria and provides framework boundary validation.
    """

    def __init__(self) -> None:
        """Initialize edge case interpreter with predefined profiles."""
        self.edge_case_profiles = self._initialize_edge_case_profiles()

        # Classification thresholds
        self.classification_confidence_threshold = 0.7
        self.boundary_confidence_threshold = 0.6

    def _initialize_edge_case_profiles(self) -> Dict[EdgeCaseType, EdgeCaseProfile]:
        """Initialize profiles for different edge case types."""
        profiles = {}

        # Anesthesia Awareness
        profiles[EdgeCaseType.ANESTHESIA_AWARENESS] = EdgeCaseProfile(
            case_type=EdgeCaseType.ANESTHESIA_AWARENESS,
            description="Conscious awareness during general anesthesia with paralysis",
            expected_p3b_range=(2.0, 6.0),  # Variable, often reduced
            expected_gamma_range=(0.1, 0.4),  # May be present but altered
            expected_bold_pattern={
                "dlpfc": (1.5, 3.5),  # Reduced but present
                "ips": (1.0, 3.0),
                "anterior_insula": (2.0, 4.0),  # May be preserved
                "acc": (1.5, 3.5),
            },
            expected_pci_range=(0.2, 0.5),  # Intermediate values
            subjective_report_probability=0.8,  # High if able to report
            forced_choice_accuracy_range=(0.6, 0.8),  # Above chance but impaired
            metacognitive_sensitivity_range=(0.3, 0.7),  # Variable
            framework_boundary=FrameworkBoundary.BOUNDARY_CASE,
            interpretation_notes="Consciousness present with altered neural signatures due to anesthetic effects",
        )

        # Blindsight
        profiles[EdgeCaseType.BLINDSIGHT] = EdgeCaseProfile(
            case_type=EdgeCaseType.BLINDSIGHT,
            description="Above-chance visual discrimination without conscious awareness",
            expected_p3b_range=(0.5, 2.5),  # Absent or severely reduced
            expected_gamma_range=(0.05, 0.2),  # Minimal synchrony
            expected_bold_pattern={
                "v1": (0.5, 2.0),  # Reduced primary visual
                "v4": (1.0, 3.0),  # Some preserved processing
                "mt": (2.0, 4.0),  # Motion area may be preserved
                "dlpfc": (0.5, 2.0),  # Reduced frontal
                "ips": (0.5, 2.0),
            },
            expected_pci_range=(0.1, 0.3),  # Low complexity
            subjective_report_probability=0.1,  # No conscious awareness
            forced_choice_accuracy_range=(0.6, 0.8),  # Above chance performance
            metacognitive_sensitivity_range=(0.0, 0.2),  # No metacognition
            framework_boundary=FrameworkBoundary.WITHIN_FRAMEWORK,
            interpretation_notes="Unconscious processing without ignition - supports framework",
        )

        # Dreams
        profiles[EdgeCaseType.DREAMS] = EdgeCaseProfile(
            case_type=EdgeCaseType.DREAMS,
            description="Conscious experience during REM sleep",
            expected_p3b_range=(1.0, 4.0),  # Reduced but present
            expected_gamma_range=(0.2, 0.5),  # Present but altered
            expected_bold_pattern={
                "dlpfc": (0.5, 2.5),  # Reduced frontal control
                "ips": (1.5, 3.5),  # Parietal activity
                "visual_cortex": (3.0, 5.0),  # High visual activity
                "limbic": (3.5, 5.5),  # High emotional processing
            },
            expected_pci_range=(0.3, 0.6),  # Intermediate to high
            subjective_report_probability=0.6,  # If awakened during REM
            forced_choice_accuracy_range=(0.4, 0.7),  # Variable, often impaired
            metacognitive_sensitivity_range=(0.1, 0.4),  # Reduced metacognition
            framework_boundary=FrameworkBoundary.BOUNDARY_CASE,
            interpretation_notes="Altered consciousness state with modified ignition patterns",
        )

        # Locked-in Syndrome
        profiles[EdgeCaseType.LOCKED_IN_SYNDROME] = EdgeCaseProfile(
            case_type=EdgeCaseType.LOCKED_IN_SYNDROME,
            description="Full consciousness with motor paralysis",
            expected_p3b_range=(4.0, 8.0),  # Normal or enhanced
            expected_gamma_range=(0.25, 0.45),  # Normal synchrony
            expected_bold_pattern={
                "dlpfc": (3.0, 5.0),  # Normal frontal
                "ips": (3.0, 5.0),  # Normal parietal
                "anterior_insula": (3.5, 5.5),  # Enhanced awareness
                "motor_cortex": (0.5, 2.0),  # Reduced motor output
                "brainstem": (0.5, 2.0),  # Lesioned brainstem
            },
            expected_pci_range=(0.4, 0.7),  # Normal to high
            subjective_report_probability=1.0,  # Full awareness
            forced_choice_accuracy_range=(0.8, 1.0),  # Normal performance
            metacognitive_sensitivity_range=(0.7, 1.0),  # Normal metacognition
            framework_boundary=FrameworkBoundary.WITHIN_FRAMEWORK,
            interpretation_notes="Normal consciousness with motor output impairment - supports framework",
        )

        # Vegetative State
        profiles[EdgeCaseType.VEGETATIVE_STATE] = EdgeCaseProfile(
            case_type=EdgeCaseType.VEGETATIVE_STATE,
            description="Wakefulness without awareness",
            expected_p3b_range=(0.0, 2.0),  # Absent or minimal
            expected_gamma_range=(0.0, 0.15),  # Minimal synchrony
            expected_bold_pattern={
                "dlpfc": (0.0, 2.0),  # Severely reduced
                "ips": (0.0, 2.0),
                "anterior_insula": (0.0, 2.0),
                "thalamus": (0.5, 2.5),  # Some preserved function
                "brainstem": (2.0, 4.0),  # Preserved arousal
            },
            expected_pci_range=(0.0, 0.25),  # Very low
            subjective_report_probability=0.0,  # No awareness
            forced_choice_accuracy_range=(0.45, 0.55),  # Chance level
            metacognitive_sensitivity_range=(0.0, 0.1),  # No metacognition
            framework_boundary=FrameworkBoundary.WITHIN_FRAMEWORK,
            interpretation_notes="No consciousness, no ignition - supports framework",
        )

        # Minimally Conscious State
        profiles[EdgeCaseType.MINIMALLY_CONSCIOUS_STATE] = EdgeCaseProfile(
            case_type=EdgeCaseType.MINIMALLY_CONSCIOUS_STATE,
            description="Fluctuating minimal consciousness",
            expected_p3b_range=(1.0, 4.0),  # Variable, often reduced
            expected_gamma_range=(0.1, 0.3),  # Intermittent synchrony
            expected_bold_pattern={
                "dlpfc": (1.0, 3.5),  # Reduced but present
                "ips": (1.0, 3.5),
                "anterior_insula": (1.5, 4.0),
                "thalamus": (2.0, 4.0),  # Better than VS
                "default_mode": (1.0, 3.0),  # Partially preserved
            },
            expected_pci_range=(0.2, 0.4),  # Low to intermediate
            subjective_report_probability=0.3,  # Intermittent
            forced_choice_accuracy_range=(0.5, 0.7),  # Variable
            metacognitive_sensitivity_range=(0.1, 0.3),  # Minimal
            framework_boundary=FrameworkBoundary.BOUNDARY_CASE,
            interpretation_notes="Fluctuating consciousness with variable ignition patterns",
        )

        # Normal Control
        profiles[EdgeCaseType.NORMAL_CONTROL] = EdgeCaseProfile(
            case_type=EdgeCaseType.NORMAL_CONTROL,
            description="Normal conscious processing",
            expected_p3b_range=(5.0, 8.0),  # Above threshold
            expected_gamma_range=(0.3, 0.6),  # Strong synchrony
            expected_bold_pattern={
                "dlpfc": (3.1, 5.5),  # Above threshold
                "ips": (3.1, 5.5),
                "anterior_insula": (3.1, 5.5),
                "acc": (3.1, 5.5),
            },
            expected_pci_range=(0.4, 0.8),  # Above threshold
            subjective_report_probability=0.9,  # High awareness
            forced_choice_accuracy_range=(0.75, 0.95),  # Good performance
            metacognitive_sensitivity_range=(0.6, 0.9),  # Good metacognition
            framework_boundary=FrameworkBoundary.WITHIN_FRAMEWORK,
            interpretation_notes="Normal consciousness with full ignition - supports framework",
        )

        return profiles

    def classify_edge_case(
        self,
        neural_signatures: CombinedSignature,
        consciousness_assessment: "ConsciousnessAssessment",
        analysis_id: Optional[str] = None,
    ) -> EdgeCaseAnalysisResult:
        """Classify and interpret an edge case based on neural signatures and consciousness assessment.

        Args:
            neural_signatures: Combined neural signature data
            consciousness_assessment: Consciousness assessment results
            analysis_id: Optional analysis identifier

        Returns:
            Complete edge case analysis result
        """
        if analysis_id is None:
            analysis_id = f"edge_case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Calculate classification scores for each edge case type
        classification_scores = {}
        for case_type, profile in self.edge_case_profiles.items():
            score = self._calculate_classification_score(
                neural_signatures, consciousness_assessment, profile
            )
            classification_scores[case_type] = score

        # Find primary classification (highest score)
        primary_case_type = max(
            classification_scores.keys(), key=lambda x: classification_scores[x]
        )
        primary_score = classification_scores[primary_case_type]

        # Create primary classification
        primary_classification = self._create_classification(
            primary_case_type,
            primary_score,
            neural_signatures,
            consciousness_assessment,
        )

        # Find alternative classifications (scores > threshold)
        alternative_classifications = []
        for case_type, score in classification_scores.items():
            if case_type != primary_case_type and score > self.classification_confidence_threshold:
                alt_classification = self._create_classification(
                    case_type, score, neural_signatures, consciousness_assessment
                )
                alternative_classifications.append(alt_classification)

        # Sort alternatives by confidence
        alternative_classifications.sort(key=lambda x: x.confidence, reverse=True)

        # Generate framework implications
        framework_implications = self._generate_framework_implications(
            primary_classification, neural_signatures, consciousness_assessment
        )

        # Assess falsification relevance
        falsification_relevance = self._assess_falsification_relevance(
            primary_classification, neural_signatures, consciousness_assessment
        )

        # Determine if further testing is needed
        further_testing_needed = (
            primary_classification.confidence < 0.8 or len(alternative_classifications) > 0
        )

        # Generate recommendations
        recommended_tests = self._generate_test_recommendations(
            primary_classification, alternative_classifications
        )

        return EdgeCaseAnalysisResult(
            analysis_id=analysis_id,
            timestamp=datetime.now(),
            neural_signatures=neural_signatures,
            consciousness_assessment=consciousness_assessment,
            primary_classification=primary_classification,
            alternative_classifications=alternative_classifications,
            framework_implications=framework_implications,
            falsification_relevance=falsification_relevance,
            further_testing_needed=further_testing_needed,
            recommended_tests=recommended_tests,
        )

    def _calculate_classification_score(
        self,
        neural_signatures: CombinedSignature,
        consciousness_assessment: "ConsciousnessAssessment",
        profile: EdgeCaseProfile,
    ) -> float:
        """Calculate how well the data matches an edge case profile."""
        scores = []

        # P3b amplitude score
        p3b_score = self._calculate_range_score(
            neural_signatures.p3b_amplitude, profile.expected_p3b_range
        )
        scores.append(p3b_score)

        # Gamma synchrony score
        gamma_score = self._calculate_range_score(
            neural_signatures.gamma_plv, profile.expected_gamma_range
        )
        scores.append(gamma_score)

        # BOLD activation scores
        bold_scores = []
        for region, (min_val, max_val) in profile.expected_bold_pattern.items():
            if region in neural_signatures.bold_activations:
                bold_score = self._calculate_range_score(
                    neural_signatures.bold_activations[region], (min_val, max_val)
                )
                bold_scores.append(bold_score)

        if bold_scores:
            scores.append(float(np.mean(bold_scores)))

        # PCI score
        pci_score = self._calculate_range_score(
            neural_signatures.pci_value, profile.expected_pci_range
        )
        scores.append(pci_score)

        # Consciousness assessment scores
        if consciousness_assessment.subjective_report:
            report_score = profile.subjective_report_probability
        else:
            report_score = 1.0 - profile.subjective_report_probability
        scores.append(report_score)

        # Forced choice accuracy score
        fc_score = self._calculate_range_score(
            consciousness_assessment.forced_choice_accuracy,
            profile.forced_choice_accuracy_range,
        )
        scores.append(fc_score)

        # Metacognitive sensitivity score
        meta_score = self._calculate_range_score(
            consciousness_assessment.metacognitive_sensitivity,
            profile.metacognitive_sensitivity_range,
        )
        scores.append(meta_score)

        # Return weighted average (equal weights for now)
        return float(np.mean(scores))

    def _calculate_range_score(self, value: float, expected_range: Tuple[float, float]) -> float:
        """Calculate how well a value fits within an expected range."""
        min_val, max_val = expected_range

        if min_val <= value <= max_val:
            # Value is within range - score based on position in range
            range_size = max_val - min_val
            if range_size == 0:
                return 1.0

            # Score is highest at center of range
            center = (min_val + max_val) / 2
            distance_from_center = abs(value - center)
            normalized_distance = distance_from_center / (range_size / 2)
            return max(0.5, 1.0 - normalized_distance * 0.5)
        else:
            # Value is outside range - score based on distance from range
            if value < min_val:
                distance = min_val - value
                range_size = max_val - min_val
            else:
                distance = value - max_val
                range_size = max_val - min_val

            # Exponential decay based on distance
            normalized_distance = distance / range_size if range_size > 0 else distance
            return max(0.0, float(np.exp(-normalized_distance)))

    def _create_classification(
        self,
        case_type: EdgeCaseType,
        confidence: float,
        neural_signatures: CombinedSignature,
        consciousness_assessment: "ConsciousnessAssessment",
    ) -> EdgeCaseClassification:
        """Create a classification result for a specific edge case type."""
        profile = self.edge_case_profiles[case_type]

        # Analyze supporting and contradicting evidence
        supporting_evidence = []
        contradicting_evidence = []

        # Check P3b evidence
        if (
            profile.expected_p3b_range[0]
            <= neural_signatures.p3b_amplitude
            <= profile.expected_p3b_range[1]
        ):
            supporting_evidence.append(
                f"P3b amplitude ({neural_signatures.p3b_amplitude:.2f} μV) within expected range"
            )
        else:
            contradicting_evidence.append(
                f"P3b amplitude ({neural_signatures.p3b_amplitude:.2f} μV) outside expected range"
            )

        # Check gamma evidence
        if (
            profile.expected_gamma_range[0]
            <= neural_signatures.gamma_plv
            <= profile.expected_gamma_range[1]
        ):
            supporting_evidence.append(
                f"Gamma PLV ({neural_signatures.gamma_plv:.3f}) within expected range"
            )
        else:
            contradicting_evidence.append(
                f"Gamma PLV ({neural_signatures.gamma_plv:.3f}) outside expected range"
            )

        # Check PCI evidence
        if (
            profile.expected_pci_range[0]
            <= neural_signatures.pci_value
            <= profile.expected_pci_range[1]
        ):
            supporting_evidence.append(
                f"PCI ({neural_signatures.pci_value:.3f}) within expected range"
            )
        else:
            contradicting_evidence.append(
                f"PCI ({neural_signatures.pci_value:.3f}) outside expected range"
            )

        # Check consciousness evidence
        expected_report = profile.subjective_report_probability > 0.5
        if consciousness_assessment.subjective_report == expected_report:
            supporting_evidence.append(
                f"Subjective report matches expectation ({consciousness_assessment.subjective_report})"
            )
        else:
            contradicting_evidence.append(
                f"Subjective report contradicts expectation ({consciousness_assessment.subjective_report})"
            )

        # Calculate signature and consciousness consistency
        signature_consistency = self._calculate_signature_consistency(neural_signatures, profile)
        consciousness_consistency = self._calculate_consciousness_consistency(
            consciousness_assessment, profile
        )

        # Determine framework boundary
        boundary_confidence = min(confidence, 0.9)  # Cap at 0.9 for boundary decisions

        # Generate interpretation
        interpretation = self._generate_interpretation(
            case_type, profile, supporting_evidence, contradicting_evidence
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(case_type, profile, contradicting_evidence)

        return EdgeCaseClassification(
            case_type=case_type,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            signature_consistency=signature_consistency,
            consciousness_consistency=consciousness_consistency,
            framework_boundary=profile.framework_boundary,
            boundary_confidence=boundary_confidence,
            interpretation=interpretation,
            recommendations=recommendations,
        )

    def _calculate_signature_consistency(
        self, neural_signatures: CombinedSignature, profile: EdgeCaseProfile
    ) -> float:
        """Calculate consistency of neural signatures with profile expectations."""
        scores = []

        # P3b consistency
        p3b_score = self._calculate_range_score(
            neural_signatures.p3b_amplitude, profile.expected_p3b_range
        )
        scores.append(p3b_score)

        # Gamma consistency
        gamma_score = self._calculate_range_score(
            neural_signatures.gamma_plv, profile.expected_gamma_range
        )
        scores.append(gamma_score)

        # PCI consistency
        pci_score = self._calculate_range_score(
            neural_signatures.pci_value, profile.expected_pci_range
        )
        scores.append(pci_score)

        # BOLD consistency
        bold_scores = []
        for region, (min_val, max_val) in profile.expected_bold_pattern.items():
            if region in neural_signatures.bold_activations:
                bold_score = self._calculate_range_score(
                    neural_signatures.bold_activations[region], (min_val, max_val)
                )
                bold_scores.append(bold_score)

        if bold_scores:
            scores.append(float(np.mean(bold_scores)))

        return float(np.mean(scores))

    def _calculate_consciousness_consistency(
        self,
        consciousness_assessment: "ConsciousnessAssessment",
        profile: EdgeCaseProfile,
    ) -> float:
        """Calculate consistency of consciousness measures with profile expectations."""
        scores = []

        # Subjective report consistency
        if consciousness_assessment.subjective_report:
            report_score = profile.subjective_report_probability
        else:
            report_score = 1.0 - profile.subjective_report_probability
        scores.append(report_score)

        # Forced choice accuracy consistency
        fc_score = self._calculate_range_score(
            consciousness_assessment.forced_choice_accuracy,
            profile.forced_choice_accuracy_range,
        )
        scores.append(fc_score)

        # Metacognitive sensitivity consistency
        meta_score = self._calculate_range_score(
            consciousness_assessment.metacognitive_sensitivity,
            profile.metacognitive_sensitivity_range,
        )
        scores.append(meta_score)

        return float(np.mean(scores))

    def _generate_interpretation(
        self,
        case_type: EdgeCaseType,
        profile: EdgeCaseProfile,
        supporting_evidence: List[str],
        contradicting_evidence: List[str],
    ) -> str:
        """Generate interpretation text for the classification."""
        interpretation_parts = [f"Classification: {case_type.value.replace('_', ' ').title()}"]
        interpretation_parts.append(f"Description: {profile.description}")
        interpretation_parts.append(
            f"Framework Boundary: {profile.framework_boundary.value.replace('_', ' ').title()}"
        )

        if supporting_evidence:
            interpretation_parts.append(f"Supporting Evidence: {'; '.join(supporting_evidence)}")

        if contradicting_evidence:
            interpretation_parts.append(
                f"Contradicting Evidence: {'; '.join(contradicting_evidence)}"
            )

        interpretation_parts.append(f"Clinical Notes: {profile.interpretation_notes}")

        return " | ".join(interpretation_parts)

    def _generate_recommendations(
        self,
        case_type: EdgeCaseType,
        profile: EdgeCaseProfile,
        contradicting_evidence: List[str],
    ) -> List[str]:
        """Generate recommendations based on classification."""
        recommendations = []

        if contradicting_evidence:
            recommendations.append(
                "Consider alternative classifications due to contradicting evidence"
            )
            recommendations.append("Conduct additional neural signature measurements")

        if case_type == EdgeCaseType.ANESTHESIA_AWARENESS:
            recommendations.extend(
                [
                    "Verify anesthetic depth with BIS/entropy monitoring",
                    "Check for muscle relaxant effects on motor responses",
                    "Consider post-operative interview for awareness confirmation",
                ]
            )
        elif case_type == EdgeCaseType.BLINDSIGHT:
            recommendations.extend(
                [
                    "Confirm visual field defects with perimetry",
                    "Test multiple visual discrimination tasks",
                    "Verify absence of conscious visual experience",
                ]
            )
        elif case_type == EdgeCaseType.DREAMS:
            recommendations.extend(
                [
                    "Confirm REM sleep stage with polysomnography",
                    "Conduct immediate awakening protocols",
                    "Assess dream content complexity and vividness",
                ]
            )
        elif case_type == EdgeCaseType.LOCKED_IN_SYNDROME:
            recommendations.extend(
                [
                    "Verify brainstem lesion location with MRI",
                    "Test eye movement communication systems",
                    "Assess cognitive function with adapted tests",
                ]
            )
        elif case_type in [
            EdgeCaseType.VEGETATIVE_STATE,
            EdgeCaseType.MINIMALLY_CONSCIOUS_STATE,
        ]:
            recommendations.extend(
                [
                    "Conduct repeated assessments over multiple sessions",
                    "Use standardized consciousness scales (CRS-R)",
                    "Consider command-following paradigms",
                ]
            )

        return recommendations

    def _generate_framework_implications(
        self,
        classification: EdgeCaseClassification,
        neural_signatures: CombinedSignature,
        consciousness_assessment: "ConsciousnessAssessment",
    ) -> str:
        """Generate framework implications based on classification."""
        implications = []

        if classification.framework_boundary == FrameworkBoundary.WITHIN_FRAMEWORK:
            implications.append("This case falls within expected APGI Framework boundaries.")
            implications.append("The observed pattern supports framework predictions.")

            if classification.case_type == EdgeCaseType.NORMAL_CONTROL:
                implications.append("Normal ignition pattern with conscious access as expected.")
            elif classification.case_type == EdgeCaseType.BLINDSIGHT:
                implications.append("Unconscious processing without ignition supports framework.")
            elif classification.case_type == EdgeCaseType.LOCKED_IN_SYNDROME:
                implications.append(
                    "Preserved consciousness with motor impairment supports framework."
                )
            elif classification.case_type == EdgeCaseType.VEGETATIVE_STATE:
                implications.append("Absence of consciousness and ignition supports framework.")

        elif classification.framework_boundary == FrameworkBoundary.BOUNDARY_CASE:
            implications.append("This case represents a boundary condition for the APGI Framework.")
            implications.append("Framework may need refinement to fully account for this pattern.")

            if classification.case_type == EdgeCaseType.ANESTHESIA_AWARENESS:
                implications.append(
                    "Altered consciousness under anesthesia challenges standard ignition patterns."
                )
            elif classification.case_type == EdgeCaseType.DREAMS:
                implications.append("Dream consciousness shows modified ignition characteristics.")
            elif classification.case_type == EdgeCaseType.MINIMALLY_CONSCIOUS_STATE:
                implications.append(
                    "Fluctuating consciousness suggests variable ignition thresholds."
                )

        elif classification.framework_boundary == FrameworkBoundary.OUTSIDE_FRAMEWORK:
            implications.append("This case falls outside current APGI Framework scope.")
            implications.append("Framework limitations may be revealed by this pattern.")

        elif classification.framework_boundary == FrameworkBoundary.FRAMEWORK_VIOLATION:
            implications.append("This case potentially violates core APGI Framework assumptions.")
            implications.append("Significant framework revision may be required.")

        return " ".join(implications)

    def _assess_falsification_relevance(
        self,
        classification: EdgeCaseClassification,
        neural_signatures: CombinedSignature,
        consciousness_assessment: "ConsciousnessAssessment",
    ) -> str:
        """Assess relevance to falsification testing."""
        relevance_parts = []

        # Check for primary falsification relevance
        has_full_signatures = (
            neural_signatures.p3b_amplitude > 5.0
            and neural_signatures.gamma_plv > 0.3
            and neural_signatures.pci_value > 0.4
            and any(z > 3.1 for z in neural_signatures.bold_activations.values())
        )

        has_consciousness = consciousness_assessment.subjective_report

        if has_full_signatures and not has_consciousness:
            relevance_parts.append(
                "CRITICAL: Full ignition signatures without consciousness - potential primary falsification"
            )
        elif has_consciousness and not has_full_signatures:
            relevance_parts.append(
                "IMPORTANT: Consciousness without full ignition - potential secondary falsification"
            )
        else:
            relevance_parts.append("Standard pattern - no immediate falsification implications")

        # Framework boundary relevance
        if classification.framework_boundary == FrameworkBoundary.FRAMEWORK_VIOLATION:
            relevance_parts.append("Framework violation detected - high falsification relevance")
        elif classification.framework_boundary == FrameworkBoundary.BOUNDARY_CASE:
            relevance_parts.append("Boundary case - moderate falsification relevance")
        else:
            relevance_parts.append("Within framework - low falsification relevance")

        # Edge case specific relevance
        if classification.case_type == EdgeCaseType.ANESTHESIA_AWARENESS:
            relevance_parts.append("Anesthesia awareness may reveal threshold modulation effects")
        elif classification.case_type == EdgeCaseType.BLINDSIGHT:
            relevance_parts.append(
                "Blindsight demonstrates unconscious processing without ignition"
            )
        elif classification.case_type == EdgeCaseType.DREAMS:
            relevance_parts.append("Dream consciousness shows altered ignition patterns")

        return " | ".join(relevance_parts)

    def _generate_test_recommendations(
        self,
        primary_classification: EdgeCaseClassification,
        alternative_classifications: List[EdgeCaseClassification],
    ) -> List[str]:
        """Generate recommendations for further testing."""
        recommendations = []

        # General recommendations based on confidence
        if primary_classification.confidence < 0.8:
            recommendations.append(
                "Low classification confidence - conduct additional measurements"
            )
            recommendations.append("Increase sample size for more reliable classification")

        # Recommendations based on alternatives
        if alternative_classifications:
            alt_types = [alt.case_type.value for alt in alternative_classifications[:2]]
            recommendations.append(f"Consider alternative classifications: {', '.join(alt_types)}")
            recommendations.append("Conduct discriminative tests between top classifications")

        # Specific test recommendations
        if primary_classification.framework_boundary == FrameworkBoundary.BOUNDARY_CASE:
            recommendations.extend(
                [
                    "Conduct longitudinal assessment to track changes over time",
                    "Test under multiple experimental conditions",
                    "Compare with matched control participants",
                ]
            )

        if primary_classification.framework_boundary == FrameworkBoundary.FRAMEWORK_VIOLATION:
            recommendations.extend(
                [
                    "Replicate findings with independent research groups",
                    "Conduct comprehensive neurological assessment",
                    "Consider framework modification or extension",
                ]
            )

        # Add case-specific recommendations from primary classification
        recommendations.extend(primary_classification.recommendations)

        return list(set(recommendations))  # Remove duplicates

    def validate_framework_boundaries(
        self, analysis_results: List[EdgeCaseAnalysisResult]
    ) -> Dict[str, Any]:
        """Validate framework boundaries across multiple edge case analyses.

        Args:
            analysis_results: List of edge case analysis results

        Returns:
            Framework boundary validation summary
        """
        boundary_counts = {boundary.value: 0 for boundary in FrameworkBoundary}
        case_type_counts = {case_type.value: 0 for case_type in EdgeCaseType}

        violation_cases = []
        boundary_cases = []

        for result in analysis_results:
            # Count boundary types
            boundary_counts[result.primary_classification.framework_boundary.value] += 1
            case_type_counts[result.primary_classification.case_type.value] += 1

            # Collect special cases
            if (
                result.primary_classification.framework_boundary
                == FrameworkBoundary.FRAMEWORK_VIOLATION
            ):
                violation_cases.append(result)
            elif (
                result.primary_classification.framework_boundary == FrameworkBoundary.BOUNDARY_CASE
            ):
                boundary_cases.append(result)

        # Calculate statistics
        total_cases = len(analysis_results)
        violation_rate = len(violation_cases) / total_cases if total_cases > 0 else 0
        boundary_rate = len(boundary_cases) / total_cases if total_cases > 0 else 0

        # Assess framework robustness
        framework_robustness = 1.0 - (violation_rate * 0.8 + boundary_rate * 0.3)

        return {
            "total_cases_analyzed": total_cases,
            "boundary_distribution": boundary_counts,
            "case_type_distribution": case_type_counts,
            "violation_rate": violation_rate,
            "boundary_case_rate": boundary_rate,
            "framework_robustness_score": framework_robustness,
            "violation_cases": [case.analysis_id for case in violation_cases],
            "boundary_cases": [case.analysis_id for case in boundary_cases],
            "recommendations": self._generate_boundary_validation_recommendations(
                violation_rate, boundary_rate, framework_robustness
            ),
        }

    def _generate_boundary_validation_recommendations(
        self, violation_rate: float, boundary_rate: float, robustness_score: float
    ) -> List[str]:
        """Generate recommendations based on boundary validation results."""
        recommendations = []

        if violation_rate > 0.1:  # More than 10% violations
            recommendations.append(
                "High violation rate detected - framework may need significant revision"
            )
            recommendations.append("Conduct systematic review of violation cases")
            recommendations.append(
                "Consider expanding framework scope or modifying core assumptions"
            )

        if boundary_rate > 0.3:  # More than 30% boundary cases
            recommendations.append(
                "High boundary case rate - framework boundaries may need clarification"
            )
            recommendations.append("Develop more specific criteria for boundary conditions")
            recommendations.append("Consider creating sub-categories for boundary cases")

        if robustness_score < 0.7:
            recommendations.append("Low framework robustness - comprehensive review recommended")
            recommendations.append("Increase sample sizes for edge case testing")
            recommendations.append("Conduct multi-site validation studies")
        elif robustness_score > 0.9:
            recommendations.append("High framework robustness - framework appears well-validated")
            recommendations.append("Consider expanding to new edge case types")

        return recommendations
