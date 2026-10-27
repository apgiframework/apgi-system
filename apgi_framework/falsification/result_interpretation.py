"""Result Interpretation System.

This module implements comprehensive result interpretation for APGI Framework
falsification testing. Provides classification of signature patterns, falsification
assessment, and detailed result logging and interpretation.

Key components:
- Complete vs partial signature pattern detection
- Falsification vs subthreshold ignition classification
- Detailed result logging and interpretation
- Statistical significance assessment
- Publication-ready result formatting
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class SignaturePattern(Enum):
    """Types of neural signature patterns."""

    COMPLETE_IGNITION = "complete_ignition"
    PARTIAL_IGNITION = "partial_ignition"
    SUBTHRESHOLD = "subthreshold"
    ABSENT = "absent"


class FalsificationOutcome(Enum):
    """Possible falsification outcomes."""

    FALSIFIED = "falsified"
    NOT_FALSIFIED = "not_falsified"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_DATA = "insufficient_data"


class ConsciousnessState(Enum):
    """Consciousness assessment states."""

    CONSCIOUS = "conscious"
    UNCONSCIOUS = "unconscious"
    AMBIGUOUS = "ambiguous"
    NOT_ASSESSED = "not_assessed"


@dataclass
class SignatureThresholds:
    """Thresholds for neural signature classification.

    Defines the criteria for determining complete vs partial ignition signatures.
    """

    p3b_amplitude_threshold: float = 5.0  # μV
    p3b_amplitude_strong: float = 7.0  # μV for strong signature

    gamma_plv_threshold: float = 0.3  # Phase-locking value
    gamma_plv_strong: float = 0.5  # Strong gamma synchrony

    bold_z_threshold: float = 3.1  # Z-score for significance
    bold_z_strong: float = 4.0  # Strong BOLD activation

    pci_threshold: float = 0.4  # Perturbational Complexity Index
    pci_strong: float = 0.6  # Strong PCI value

    # Minimum number of regions/signatures required
    min_bold_regions: int = 3  # Minimum BOLD regions for complete pattern
    min_signatures_complete: int = 3  # Minimum signatures for complete ignition


@dataclass
class SignatureClassification:
    """Classification result for neural signature patterns.

    Contains detailed analysis of signature completeness and strength.
    """

    pattern_type: SignaturePattern
    signature_strength: float  # Overall strength score (0.0-1.0)
    completeness_score: float  # Completeness score (0.0-1.0)

    # Individual signature assessments
    p3b_present: bool
    p3b_strength: float

    gamma_present: bool
    gamma_strength: float

    bold_present: bool
    bold_strength: float
    bold_regions_active: int

    pci_present: bool
    pci_strength: float

    # Pattern details
    signatures_present: int
    signatures_required: int
    pattern_confidence: float

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FalsificationAssessment:
    """Assessment of falsification criteria and outcomes.

    Contains detailed analysis of whether falsification criteria are met.
    """

    outcome: FalsificationOutcome
    confidence_level: float  # Confidence in the outcome (0.0-1.0)

    # Primary falsification criteria
    complete_signatures_present: bool
    consciousness_absent: bool
    ai_acc_engagement_absent: bool
    experimental_controls_valid: bool

    # Detailed assessments
    signature_classification: SignatureClassification
    consciousness_state: ConsciousnessState
    consciousness_confidence: float

    # Statistical measures
    effect_size: Optional[float] = None
    p_value: Optional[float] = None
    statistical_power: Optional[float] = None

    # Interpretation details
    interpretation_summary: str = ""
    detailed_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialInterpretation:
    """Complete interpretation result for a single trial.

    Contains all analysis and interpretation for one falsification test trial.
    """

    trial_id: str
    participant_id: str
    timestamp: datetime

    # Core assessments
    signature_classification: SignatureClassification
    falsification_assessment: FalsificationAssessment

    # Raw measurements (references to other result objects)
    neural_signatures: Dict[str, Any]
    consciousness_assessment: Dict[str, Any]
    ai_acc_validation: Dict[str, Any]
    experimental_controls: Dict[str, Any]

    # Summary metrics
    overall_confidence: float
    result_quality: float
    interpretation_reliability: float

    # Detailed logging
    analysis_log: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)


class SignaturePatternClassifier:
    """Classifier for neural signature patterns.

    Determines whether signatures represent complete ignition, partial ignition,
    subthreshold activity, or absent signatures.
    """

    def __init__(self, thresholds: Optional[SignatureThresholds] = None):
        """Initialize the signature pattern classifier.

        Args:
            thresholds: Custom thresholds for signature classification
        """
        self.thresholds = thresholds or SignatureThresholds()

    def classify_p3b_signature(
        self, amplitude: float, latency: Optional[float] = None
    ) -> Tuple[bool, float]:
        """Classify P3b ERP signature.

        Args:
            amplitude: P3b amplitude in μV
            latency: P3b latency in ms (optional)

        Returns:
            Tuple of (is_present, strength_score)
        """
        if amplitude >= self.thresholds.p3b_amplitude_strong:
            return True, 1.0
        elif amplitude >= self.thresholds.p3b_amplitude_threshold:
            # Scale strength based on amplitude
            strength = (amplitude - self.thresholds.p3b_amplitude_threshold) / (
                self.thresholds.p3b_amplitude_strong - self.thresholds.p3b_amplitude_threshold
            )
            return True, min(strength + 0.5, 1.0)
        else:
            # Below threshold - calculate weak strength
            strength = amplitude / self.thresholds.p3b_amplitude_threshold * 0.4
            return False, max(strength, 0.0)

    def classify_gamma_signature(
        self, plv: float, duration: Optional[float] = None
    ) -> Tuple[bool, float]:
        """Classify gamma synchrony signature.

        Args:
            plv: Phase-locking value
            duration: Duration of synchrony in ms (optional)

        Returns:
            Tuple of (is_present, strength_score)
        """
        if plv >= self.thresholds.gamma_plv_strong:
            return True, 1.0
        elif plv >= self.thresholds.gamma_plv_threshold:
            # Scale strength based on PLV
            strength = (plv - self.thresholds.gamma_plv_threshold) / (
                self.thresholds.gamma_plv_strong - self.thresholds.gamma_plv_threshold
            )
            return True, min(strength + 0.5, 1.0)
        else:
            # Below threshold
            strength = plv / self.thresholds.gamma_plv_threshold * 0.4
            return False, max(strength, 0.0)

    def classify_bold_signature(self, activations: Dict[str, float]) -> Tuple[bool, float, int]:
        """Classify BOLD activation signature.

        Args:
            activations: Dictionary of region -> z-score activations

        Returns:
            Tuple of (is_present, strength_score, active_regions_count)
        """
        if not activations:
            return False, 0.0, 0

        active_regions = 0
        strength_scores = []

        for region, z_score in activations.items():
            if z_score >= self.thresholds.bold_z_strong:
                active_regions += 1
                strength_scores.append(1.0)
            elif z_score >= self.thresholds.bold_z_threshold:
                active_regions += 1
                # Scale strength based on z-score
                strength = (z_score - self.thresholds.bold_z_threshold) / (
                    self.thresholds.bold_z_strong - self.thresholds.bold_z_threshold
                )
                strength_scores.append(min(strength + 0.5, 1.0))
            else:
                # Below threshold
                strength = z_score / self.thresholds.bold_z_threshold * 0.4
                strength_scores.append(max(strength, 0.0))

        overall_strength = float(np.mean(strength_scores)) if strength_scores else 0.0
        is_present = active_regions >= self.thresholds.min_bold_regions

        return is_present, overall_strength, active_regions

    def classify_pci_signature(self, pci_value: float) -> Tuple[bool, float]:
        """Classify Perturbational Complexity Index signature.

        Args:
            pci_value: PCI value

        Returns:
            Tuple of (is_present, strength_score)
        """
        if pci_value >= self.thresholds.pci_strong:
            return True, 1.0
        elif pci_value >= self.thresholds.pci_threshold:
            # Scale strength based on PCI value
            strength = (pci_value - self.thresholds.pci_threshold) / (
                self.thresholds.pci_strong - self.thresholds.pci_threshold
            )
            return True, min(strength + 0.5, 1.0)
        else:
            # Below threshold
            strength = pci_value / self.thresholds.pci_threshold * 0.4
            return False, max(strength, 0.0)

    def classify_complete_signature_pattern(
        self,
        p3b_amplitude: float,
        gamma_plv: float,
        bold_activations: Dict[str, float],
        pci_value: float,
    ) -> SignatureClassification:
        """Classify complete neural signature pattern.

        Args:
            p3b_amplitude: P3b amplitude in μV
            gamma_plv: Gamma phase-locking value
            bold_activations: BOLD activations by region
            pci_value: PCI value

        Returns:
            Complete signature classification
        """
        # Classify individual signatures
        p3b_present, p3b_strength = self.classify_p3b_signature(p3b_amplitude)
        gamma_present, gamma_strength = self.classify_gamma_signature(gamma_plv)
        bold_present, bold_strength, bold_regions = self.classify_bold_signature(bold_activations)
        pci_present, pci_strength = self.classify_pci_signature(pci_value)

        # Count present signatures
        signatures_present = sum([p3b_present, gamma_present, bold_present, pci_present])
        signatures_required = self.thresholds.min_signatures_complete

        # Determine overall pattern type
        if signatures_present >= signatures_required:
            if all([p3b_present, gamma_present, bold_present, pci_present]):
                pattern_type = SignaturePattern.COMPLETE_IGNITION
            else:
                pattern_type = SignaturePattern.PARTIAL_IGNITION
        elif signatures_present > 0:
            pattern_type = SignaturePattern.SUBTHRESHOLD
        else:
            pattern_type = SignaturePattern.ABSENT

        # Calculate overall scores
        signature_strength = float(
            np.mean([p3b_strength, gamma_strength, bold_strength, pci_strength])
        )
        completeness_score = signatures_present / 4.0  # 4 total signature types

        # Calculate pattern confidence
        pattern_confidence = self._calculate_pattern_confidence(
            signatures_present,
            [p3b_strength, gamma_strength, bold_strength, pci_strength],
        )

        return SignatureClassification(
            pattern_type=pattern_type,
            signature_strength=signature_strength,
            completeness_score=completeness_score,
            p3b_present=p3b_present,
            p3b_strength=p3b_strength,
            gamma_present=gamma_present,
            gamma_strength=gamma_strength,
            bold_present=bold_present,
            bold_strength=bold_strength,
            bold_regions_active=bold_regions,
            pci_present=pci_present,
            pci_strength=pci_strength,
            signatures_present=signatures_present,
            signatures_required=signatures_required,
            pattern_confidence=pattern_confidence,
            metadata={
                "p3b_amplitude": p3b_amplitude,
                "gamma_plv": gamma_plv,
                "bold_activations": bold_activations,
                "pci_value": pci_value,
            },
        )

    def _calculate_pattern_confidence(
        self, signatures_present: int, strength_scores: List[float]
    ) -> float:
        """Calculate confidence in pattern classification."""
        base_confidence = 0.5

        # Boost confidence based on number of signatures
        signature_boost = (signatures_present / 4.0) * 0.3

        # Boost confidence based on strength of signatures
        mean_strength = np.mean(strength_scores)
        strength_boost = mean_strength * 0.2

        return min(float(base_confidence + signature_boost + strength_boost), 1.0)


class FalsificationInterpreter:
    """Interpreter for falsification test results.

    Determines whether falsification criteria are met and provides detailed
    interpretation of experimental outcomes.
    """

    def __init__(self) -> None:
        """Initialize the falsification interpreter."""
        self.signature_classifier = SignaturePatternClassifier()

    def assess_consciousness_state(
        self,
        subjective_report: bool,
        forced_choice_accuracy: float,
        confidence_rating: Optional[float] = None,
    ) -> Tuple[ConsciousnessState, float]:
        """Assess consciousness state from behavioral measures.

        Args:
            subjective_report: Whether participant reported awareness
            forced_choice_accuracy: Accuracy in forced-choice task (0.0-1.0)
            confidence_rating: Confidence rating (optional)

        Returns:
            Tuple of (consciousness_state, confidence_level)
        """
        chance_level = 0.5
        significance_threshold = 0.6  # Above chance performance

        if subjective_report and forced_choice_accuracy > significance_threshold:
            return ConsciousnessState.CONSCIOUS, 0.9
        elif not subjective_report and forced_choice_accuracy <= chance_level + 0.05:
            return ConsciousnessState.UNCONSCIOUS, 0.8
        elif subjective_report and forced_choice_accuracy <= chance_level + 0.05:
            # Subjective report but no objective evidence
            return ConsciousnessState.AMBIGUOUS, 0.6
        elif not subjective_report and forced_choice_accuracy > significance_threshold:
            # No subjective report but objective evidence
            return ConsciousnessState.AMBIGUOUS, 0.7
        else:
            return ConsciousnessState.AMBIGUOUS, 0.5

    def assess_primary_falsification(
        self,
        signature_classification: SignatureClassification,
        consciousness_state: ConsciousnessState,
        ai_acc_absent: bool,
        controls_valid: bool,
    ) -> FalsificationAssessment:
        """Assess primary falsification criterion.

        Primary falsification occurs when:
        1. Complete ignition signatures are present
        2. Consciousness is absent
        3. AI/ACC engagement is absent
        4. Experimental controls are valid

        Args:
            signature_classification: Neural signature classification
            consciousness_state: Assessed consciousness state
            ai_acc_absent: Whether AI/ACC engagement is absent
            controls_valid: Whether experimental controls are valid

        Returns:
            Complete falsification assessment
        """
        # Check primary falsification criteria
        complete_signatures = (
            signature_classification.pattern_type == SignaturePattern.COMPLETE_IGNITION
        )
        consciousness_absent = consciousness_state == ConsciousnessState.UNCONSCIOUS

        # Determine falsification outcome
        if complete_signatures and consciousness_absent and ai_acc_absent and controls_valid:
            outcome = FalsificationOutcome.FALSIFIED
            confidence = min(signature_classification.pattern_confidence * 0.9, 0.95)
        elif not controls_valid:
            outcome = FalsificationOutcome.INSUFFICIENT_DATA
            confidence = 0.3
        elif signature_classification.pattern_type == SignaturePattern.PARTIAL_IGNITION:
            outcome = FalsificationOutcome.INCONCLUSIVE
            confidence = 0.6
        else:
            outcome = FalsificationOutcome.NOT_FALSIFIED
            confidence = 0.8

        # Generate interpretation summary
        interpretation_summary = self._generate_interpretation_summary(
            outcome,
            complete_signatures,
            consciousness_absent,
            ai_acc_absent,
            controls_valid,
        )

        # Generate detailed findings
        detailed_findings = self._generate_detailed_findings(
            signature_classification, consciousness_state, ai_acc_absent, controls_valid
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            outcome, signature_classification, controls_valid
        )

        return FalsificationAssessment(
            outcome=outcome,
            confidence_level=confidence,
            complete_signatures_present=complete_signatures,
            consciousness_absent=consciousness_absent,
            ai_acc_engagement_absent=ai_acc_absent,
            experimental_controls_valid=controls_valid,
            signature_classification=signature_classification,
            consciousness_state=consciousness_state,
            consciousness_confidence=0.8,  # Would be calculated from consciousness assessment
            interpretation_summary=interpretation_summary,
            detailed_findings=detailed_findings,
            recommendations=recommendations,
        )

    def interpret_complete_trial(
        self,
        trial_id: str,
        participant_id: str,
        neural_signatures: Dict[str, Any],
        consciousness_assessment: Dict[str, Any],
        ai_acc_validation: Dict[str, Any],
        experimental_controls: Dict[str, Any],
    ) -> TrialInterpretation:
        """Provide complete interpretation for a trial.

        Args:
            trial_id: Trial identifier
            participant_id: Participant identifier
            neural_signatures: Neural signature measurements
            consciousness_assessment: Consciousness assessment results
            ai_acc_validation: AI/ACC validation results
            experimental_controls: Experimental control validation results

        Returns:
            Complete trial interpretation
        """
        timestamp = datetime.now()
        analysis_log = []
        warnings = []

        # Extract key measurements
        try:
            p3b_amplitude = neural_signatures.get("p3b_amplitude", 0.0)
            gamma_plv = neural_signatures.get("gamma_plv", 0.0)
            bold_activations = neural_signatures.get("bold_activations", {})
            pci_value = neural_signatures.get("pci_value", 0.0)

            subjective_report = consciousness_assessment.get("subjective_report", False)
            forced_choice_accuracy = consciousness_assessment.get("forced_choice_accuracy", 0.5)

            ai_acc_absent = ai_acc_validation.get("ai_acc_engagement_absent", False)
            controls_valid = experimental_controls.get("experimental_controls_valid", False)

            analysis_log.append(
                f"Extracted measurements: P3b={p3b_amplitude}, Gamma={gamma_plv}, PCI={pci_value}"
            )

        except Exception as e:
            warnings.append(f"Error extracting measurements: {str(e)}")
            # Use default values
            p3b_amplitude = gamma_plv = pci_value = 0.0
            bold_activations = {}
            subjective_report = False
            forced_choice_accuracy = 0.5
            ai_acc_absent = controls_valid = False

        # Classify signature pattern
        signature_classification = self.signature_classifier.classify_complete_signature_pattern(
            p3b_amplitude, gamma_plv, bold_activations, pci_value
        )
        analysis_log.append(
            f"Signature pattern classified as: {signature_classification.pattern_type.value}"
        )

        # Assess consciousness state
        consciousness_state, consciousness_confidence = self.assess_consciousness_state(
            subjective_report, forced_choice_accuracy
        )
        analysis_log.append(f"Consciousness state assessed as: {consciousness_state.value}")

        # Assess falsification
        falsification_assessment = self.assess_primary_falsification(
            signature_classification, consciousness_state, ai_acc_absent, controls_valid
        )
        analysis_log.append(f"Falsification outcome: {falsification_assessment.outcome.value}")

        # Calculate overall metrics
        overall_confidence = self._calculate_overall_confidence(
            signature_classification.pattern_confidence,
            consciousness_confidence,
            falsification_assessment.confidence_level,
        )

        result_quality = self._calculate_result_quality(
            signature_classification, controls_valid, len(warnings)
        )

        interpretation_reliability = self._calculate_interpretation_reliability(
            signature_classification.completeness_score,
            falsification_assessment.confidence_level,
            result_quality,
        )

        return TrialInterpretation(
            trial_id=trial_id,
            participant_id=participant_id,
            timestamp=timestamp,
            signature_classification=signature_classification,
            falsification_assessment=falsification_assessment,
            neural_signatures=neural_signatures,
            consciousness_assessment=consciousness_assessment,
            ai_acc_validation=ai_acc_validation,
            experimental_controls=experimental_controls,
            overall_confidence=overall_confidence,
            result_quality=result_quality,
            interpretation_reliability=interpretation_reliability,
            analysis_log=analysis_log,
            warnings=warnings,
        )

    def _generate_interpretation_summary(
        self,
        outcome: FalsificationOutcome,
        complete_signatures: bool,
        consciousness_absent: bool,
        ai_acc_absent: bool,
        controls_valid: bool,
    ) -> str:
        """Generate interpretation summary text."""
        if outcome == FalsificationOutcome.FALSIFIED:
            return (
                "Primary falsification criterion met: Complete ignition signatures present "
                "without consciousness, AI/ACC engagement absent, controls valid."
            )
        elif outcome == FalsificationOutcome.NOT_FALSIFIED:
            return (
                "Primary falsification criterion not met: Either signatures incomplete "
                "or consciousness present."
            )
        elif outcome == FalsificationOutcome.INCONCLUSIVE:
            return (
                "Inconclusive result: Partial signatures present, requires further investigation."
            )
        else:
            return (
                "Insufficient data: Experimental controls invalid, results cannot be interpreted."
            )

    def _generate_detailed_findings(
        self,
        signature_classification: SignatureClassification,
        consciousness_state: ConsciousnessState,
        ai_acc_absent: bool,
        controls_valid: bool,
    ) -> List[str]:
        """Generate detailed findings list."""
        findings = []

        # Signature findings
        findings.append(f"Neural signature pattern: {signature_classification.pattern_type.value}")
        findings.append(
            f"Signature completeness: {signature_classification.completeness_score:.2f}"
        )
        findings.append(f"Signatures present: {signature_classification.signatures_present}/4")

        if signature_classification.p3b_present:
            findings.append(
                f"P3b ERP present (strength: {signature_classification.p3b_strength:.2f})"
            )

        if signature_classification.gamma_present:
            findings.append(
                f"Gamma synchrony present (strength: {signature_classification.gamma_strength:.2f})"
            )

        if signature_classification.bold_present:
            findings.append(
                f"BOLD activation present in {signature_classification.bold_regions_active} regions"
            )

        if signature_classification.pci_present:
            findings.append(
                f"PCI above threshold (strength: {signature_classification.pci_strength:.2f})"
            )

        # Consciousness findings
        findings.append(f"Consciousness state: {consciousness_state.value}")

        # AI/ACC findings
        if ai_acc_absent:
            findings.append("AI/ACC engagement absent as required for falsification")
        else:
            findings.append("AI/ACC engagement present - falsification criterion not met")

        # Control findings
        if controls_valid:
            findings.append("Experimental controls valid")
        else:
            findings.append("Experimental controls invalid - results unreliable")

        return findings

    def _generate_recommendations(
        self,
        outcome: FalsificationOutcome,
        signature_classification: SignatureClassification,
        controls_valid: bool,
    ) -> List[str]:
        """Generate recommendations list."""
        recommendations = []

        if outcome == FalsificationOutcome.FALSIFIED:
            recommendations.append("Replicate finding with independent sample")
            recommendations.append("Verify AI/ACC measurement techniques")
            recommendations.append("Consider additional consciousness measures")

        elif outcome == FalsificationOutcome.INCONCLUSIVE:
            recommendations.append("Increase sample size for better statistical power")
            recommendations.append("Optimize experimental parameters for clearer signatures")
            recommendations.append("Consider alternative consciousness assessment methods")

        elif not controls_valid:
            recommendations.append("Improve experimental control procedures")
            recommendations.append("Validate participant selection criteria")
            recommendations.append("Enhance stimulus presentation protocols")

        if signature_classification.completeness_score < 0.8:
            recommendations.append("Optimize neural signature measurement protocols")
            recommendations.append("Consider individual differences in signature expression")

        return recommendations

    def _calculate_overall_confidence(self, *confidence_scores: float) -> float:
        """Calculate overall confidence from multiple confidence scores."""
        return float(np.mean(confidence_scores))

    def _calculate_result_quality(
        self,
        signature_classification: SignatureClassification,
        controls_valid: bool,
        warning_count: int,
    ) -> float:
        """Calculate result quality score."""
        base_quality = 0.5

        # Boost for signature completeness
        signature_boost = signature_classification.completeness_score * 0.3

        # Boost for valid controls
        control_boost = 0.2 if controls_valid else -0.3

        # Penalty for warnings
        warning_penalty = warning_count * 0.05

        return max(
            0.0,
            min(1.0, base_quality + signature_boost + control_boost - warning_penalty),
        )

    def _calculate_interpretation_reliability(
        self, completeness_score: float, confidence_level: float, result_quality: float
    ) -> float:
        """Calculate interpretation reliability score."""
        return float(np.mean([completeness_score, confidence_level, result_quality]))


class ResultLogger:
    """Logger for detailed result tracking and export.

    Provides structured logging and export capabilities for falsification test results.
    """

    def __init__(self, log_file: Optional[str] = None):
        """Initialize the result logger.

        Args:
            log_file: Optional file path for persistent logging
        """
        self.log_file = log_file
        self.results_log: List[Dict[str, Any]] = []

    def log_trial_result(self, trial_interpretation: TrialInterpretation) -> None:
        """Log a complete trial interpretation result.

        Args:
            trial_interpretation: Complete trial interpretation to log
        """
        log_entry = {
            "timestamp": trial_interpretation.timestamp.isoformat(),
            "trial_id": trial_interpretation.trial_id,
            "participant_id": trial_interpretation.participant_id,
            "signature_pattern": trial_interpretation.signature_classification.pattern_type.value,
            "falsification_outcome": trial_interpretation.falsification_assessment.outcome.value,
            "overall_confidence": trial_interpretation.overall_confidence,
            "result_quality": trial_interpretation.result_quality,
            "interpretation_reliability": trial_interpretation.interpretation_reliability,
            "warnings_count": len(trial_interpretation.warnings),
            "analysis_log": trial_interpretation.analysis_log,
            "detailed_findings": trial_interpretation.falsification_assessment.detailed_findings,
        }

        self.results_log.append(log_entry)

        if self.log_file:
            self._write_to_file(log_entry)

    def export_results(self, format_type: str = "json") -> Union[str, Dict]:
        """Export logged results in specified format.

        Args:
            format_type: Export format ('json', 'csv', 'summary')

        Returns:
            Exported results in requested format
        """
        if format_type == "json":
            return json.dumps(self.results_log, indent=2)
        elif format_type == "summary":
            return self._generate_summary_report()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _write_to_file(self, log_entry: Dict[str, Any]) -> None:
        """Write log entry to file."""
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of all logged results."""
        if not self.results_log:
            return {"message": "No results logged"}

        total_trials = len(self.results_log)
        falsified_count = sum(
            1 for r in self.results_log if r["falsification_outcome"] == "falsified"
        )

        pattern_counts: Dict[str, int] = {}
        for result in self.results_log:
            pattern = result["signature_pattern"]
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        avg_confidence = np.mean([r["overall_confidence"] for r in self.results_log])
        avg_quality = np.mean([r["result_quality"] for r in self.results_log])

        return {
            "total_trials": total_trials,
            "falsified_trials": falsified_count,
            "falsification_rate": falsified_count / total_trials,
            "signature_pattern_distribution": pattern_counts,
            "average_confidence": avg_confidence,
            "average_result_quality": avg_quality,
            "summary_generated": datetime.now().isoformat(),
        }
