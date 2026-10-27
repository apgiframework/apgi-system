"""Neural signature validation and combination system for APGI Framework falsification testing.

This module implements the SignatureValidator class that validates individual neural
signatures and combines them into complete patterns for consciousness detection and
falsification testing.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .bold_simulator import BOLDSignature
from .gamma_simulator import GammaSignature

# Import signature classes
from .p3b_simulator import P3bSignature
from .pci_calculator import PCISignature


class ConsciousnessLevel(Enum):
    """Enumeration of consciousness levels based on signature patterns."""

    CONSCIOUS = "conscious"
    UNCONSCIOUS = "unconscious"
    SUBTHRESHOLD = "subthreshold"
    PARTIAL_IGNITION = "partial_ignition"
    INDETERMINATE = "indeterminate"


class SignatureType(Enum):
    """Enumeration of neural signature types."""

    P3B = "p3b"
    GAMMA = "gamma"
    BOLD = "bold"
    PCI = "pci"


@dataclass
class CombinedSignature:
    """Data class representing a combined neural signature pattern."""

    p3b_signature: Optional[P3bSignature]
    gamma_signature: Optional[GammaSignature]
    bold_signature: Optional[BOLDSignature]
    pci_signature: Optional[PCISignature]

    # Validation results
    individual_validations: Dict[str, bool]
    threshold_validations: Dict[str, bool]

    # Combined analysis
    consciousness_level: ConsciousnessLevel
    ignition_probability: float
    signature_coherence: float
    missing_signatures: List[str]

    # Falsification analysis
    is_complete_ignition: bool
    is_falsification_pattern: bool
    falsification_confidence: float

    @property
    def p3b_amplitude(self) -> float:
        """Get P3b amplitude value."""
        return self.p3b_signature.amplitude if self.p3b_signature else 0.0

    @property
    def gamma_plv(self) -> float:
        """Get gamma phase-locking value."""
        if self.gamma_signature and self.gamma_signature.plv_values:
            # Return the maximum PLV value across all region pairs
            return max(self.gamma_signature.plv_values.values())
        return 0.0

    @property
    def pci_value(self) -> float:
        """Get PCI value."""
        return self.pci_signature.pci_value if self.pci_signature else 0.0

    @property
    def bold_activations(self) -> Dict[str, float]:
        """Get BOLD activation values by region."""
        if self.bold_signature:
            return self.bold_signature.activations
        return {}


class SignatureValidator:
    """Validator for neural signatures with threshold checking and pattern combination.

    This class validates individual neural signatures against physiological constraints
    and consciousness thresholds, then combines them into complete patterns for
    falsification testing.
    """

    def __init__(self) -> None:
        """Initialize signature validator with threshold parameters."""
        # Consciousness thresholds
        self.p3b_threshold = 5.0  # μV
        self.gamma_plv_threshold = 0.3
        self.gamma_duration_threshold = 200.0  # ms
        self.bold_z_threshold = 3.1
        self.pci_threshold = 0.4

        # Unconscious thresholds
        self.p3b_unconscious_threshold = 2.0  # μV
        self.gamma_unconscious_plv = 0.15
        self.gamma_unconscious_duration = 150.0  # ms
        self.bold_unconscious_z = 2.0
        self.pci_unconscious_threshold = 0.3

        # Signature weights for combination
        self.signature_weights = {
            SignatureType.P3B: 0.25,
            SignatureType.GAMMA: 0.25,
            SignatureType.BOLD: 0.25,
            SignatureType.PCI: 0.25,
        }

    def validate_p3b_signature(self, signature: P3bSignature) -> Tuple[bool, Dict[str, Any]]:
        """Validate P3b ERP signature against physiological constraints.

        Args:
            signature: P3bSignature to validate

        Returns:
            Tuple of (is_valid, validation_details)
        """
        validation_details: Dict[str, Any] = {}
        is_valid = True

        # Check amplitude range
        if signature.amplitude < 0 or signature.amplitude > 50.0:
            is_valid = False
            validation_details["amplitude_error"] = (
                f"Amplitude {signature.amplitude} μV out of range [0, 50]"
            )

        # Check latency range
        if not (200 <= signature.latency <= 600):
            is_valid = False
            validation_details["latency_error"] = (
                f"Latency {signature.latency} ms out of range [200, 600]"
            )

        # Check electrode location
        if signature.electrode != "Pz":
            is_valid = False
            validation_details["electrode_error"] = (
                f"Invalid electrode {signature.electrode}, expected Pz"
            )

        # Check consciousness threshold
        validation_details["conscious_threshold"] = signature.amplitude > self.p3b_threshold
        validation_details["unconscious_threshold"] = (
            signature.amplitude < self.p3b_unconscious_threshold
        )
        validation_details["amplitude_level"] = self._classify_amplitude_level(signature.amplitude)

        return is_valid, validation_details

    def validate_gamma_signature(self, signature: GammaSignature) -> Tuple[bool, Dict[str, Any]]:
        """Validate gamma-band synchrony signature against physiological constraints.

        Args:
            signature: GammaSignature to validate

        Returns:
            Tuple of (is_valid, validation_details)
        """
        validation_details: Dict[str, Any] = {}
        is_valid = True

        # Check PLV values range
        for connection, plv in signature.plv_values.items():
            if not (0.0 <= plv <= 1.0):
                is_valid = False
                validation_details[f"plv_error_{connection}"] = f"PLV {plv} out of range [0, 1]"

        # Check duration
        if signature.duration <= 0:
            is_valid = False
            validation_details["duration_error"] = f"Invalid duration {signature.duration} ms"

        # Check frequency range
        if (
            signature.frequency_range[0] < 20
            or signature.frequency_range[1] > 100
            or signature.frequency_range[0] >= signature.frequency_range[1]
        ):
            is_valid = False
            validation_details["frequency_error"] = (
                f"Invalid frequency range {signature.frequency_range}"
            )

        # Check peak frequency
        if not (
            signature.frequency_range[0] <= signature.peak_frequency <= signature.frequency_range[1]
        ):
            is_valid = False
            validation_details["peak_frequency_error"] = (
                f"Peak frequency {signature.peak_frequency} outside range"
            )

        # Check consciousness thresholds
        max_plv = max(signature.plv_values.values()) if signature.plv_values else 0.0
        validation_details["conscious_threshold"] = (
            max_plv > self.gamma_plv_threshold
            and signature.duration > self.gamma_duration_threshold
        )
        validation_details["unconscious_threshold"] = (
            max_plv < self.gamma_unconscious_plv
            and signature.duration < self.gamma_unconscious_duration
        )
        validation_details["max_plv"] = max_plv
        validation_details["plv_level"] = self._classify_plv_level(max_plv)

        return is_valid, validation_details

    def validate_bold_signature(self, signature: BOLDSignature) -> Tuple[bool, Dict[str, Any]]:
        """Validate BOLD activation signature against physiological constraints.

        Args:
            signature: BOLDSignature to validate

        Returns:
            Tuple of (is_valid, validation_details)
        """
        validation_details: Dict[str, Any] = {}
        is_valid = True

        # Check Z-score ranges
        for region, z_score in signature.activations.items():
            if z_score < 0 or z_score > 15.0:
                is_valid = False
                validation_details[f"z_score_error_{region}"] = (
                    f"Z-score {z_score} out of range [0, 15]"
                )

        # Check peak activation consistency
        expected_peak = max(signature.activations.values()) if signature.activations else 0.0
        if abs(signature.peak_activation - expected_peak) > 0.01:
            is_valid = False
            validation_details["peak_consistency_error"] = (
                "Peak activation inconsistent with maximum"
            )

        # Check active regions consistency
        expected_active = [
            region for region, z in signature.activations.items() if z > self.bold_z_threshold
        ]
        if set(signature.active_regions) != set(expected_active):
            is_valid = False
            validation_details["active_regions_error"] = (
                "Active regions inconsistent with thresholds"
            )

        # Check network coherence
        if not (0.0 <= signature.network_coherence <= 1.0):
            is_valid = False
            validation_details["coherence_error"] = (
                f"Network coherence {signature.network_coherence} out of range"
            )

        # Check consciousness thresholds
        validation_details["conscious_threshold"] = (
            signature.peak_activation > self.bold_z_threshold
        )
        validation_details["unconscious_threshold"] = (
            signature.peak_activation < self.bold_unconscious_z
        )
        validation_details["bilateral_dlpfc"] = signature.bilateral_dlpfc
        validation_details["activation_level"] = self._classify_activation_level(
            signature.peak_activation
        )

        return is_valid, validation_details

    def validate_pci_signature(self, signature: PCISignature) -> Tuple[bool, Dict[str, Any]]:
        """Validate PCI signature against physiological constraints.

        Args:
            signature: PCISignature to validate

        Returns:
            Tuple of (is_valid, validation_details)
        """
        validation_details: Dict[str, Any] = {}
        is_valid = True

        # Check PCI value range
        if not (0.0 <= signature.pci_value <= 1.0):
            is_valid = False
            validation_details["pci_error"] = f"PCI value {signature.pci_value} out of range [0, 1]"

        # Check complexity components
        for component, value in signature.complexity_components.items():
            if not (0.0 <= value <= 1.0):
                is_valid = False
                validation_details[f"complexity_error_{component}"] = (
                    f"Component {value} out of range"
                )

        # Check connectivity and perturbation response
        if not (0.0 <= signature.connectivity_strength <= 1.0):
            is_valid = False
            validation_details["connectivity_error"] = "Connectivity strength out of range"

        if not (0.0 <= signature.perturbation_response <= 1.0):
            is_valid = False
            validation_details["perturbation_error"] = "Perturbation response out of range"

        # Check consciousness thresholds
        validation_details["conscious_threshold"] = signature.pci_value > self.pci_threshold
        validation_details["unconscious_threshold"] = (
            signature.pci_value < self.pci_unconscious_threshold
        )
        validation_details["pci_level"] = self._classify_pci_level(signature.pci_value)

        return is_valid, validation_details

    def combine_signatures(
        self,
        p3b_signature: Optional[P3bSignature] = None,
        gamma_signature: Optional[GammaSignature] = None,
        bold_signature: Optional[BOLDSignature] = None,
        pci_signature: Optional[PCISignature] = None,
    ) -> CombinedSignature:
        """Combine individual neural signatures into a complete pattern.

        Args:
            p3b_signature: Optional P3b ERP signature
            gamma_signature: Optional gamma synchrony signature
            bold_signature: Optional BOLD activation signature
            pci_signature: Optional PCI signature

        Returns:
            CombinedSignature with complete analysis
        """
        # Validate individual signatures
        individual_validations = {}
        threshold_validations = {}

        if p3b_signature:
            is_valid, details = self.validate_p3b_signature(p3b_signature)
            individual_validations["p3b"] = is_valid
            threshold_validations["p3b"] = details.get("conscious_threshold", False)

        if gamma_signature:
            is_valid, details = self.validate_gamma_signature(gamma_signature)
            individual_validations["gamma"] = is_valid
            threshold_validations["gamma"] = details.get("conscious_threshold", False)

        if bold_signature:
            is_valid, details = self.validate_bold_signature(bold_signature)
            individual_validations["bold"] = is_valid
            threshold_validations["bold"] = details.get("conscious_threshold", False)

        if pci_signature:
            is_valid, details = self.validate_pci_signature(pci_signature)
            individual_validations["pci"] = is_valid
            threshold_validations["pci"] = details.get("conscious_threshold", False)

        # Determine missing signatures
        all_signature_types = ["p3b", "gamma", "bold", "pci"]
        present_signatures = [
            sig for sig in all_signature_types if locals()[f"{sig}_signature"] is not None
        ]
        missing_signatures = [sig for sig in all_signature_types if sig not in present_signatures]

        # Calculate ignition probability
        ignition_probability = self._calculate_ignition_probability(
            p3b_signature, gamma_signature, bold_signature, pci_signature
        )

        # Calculate signature coherence
        signature_coherence = self._calculate_signature_coherence(
            threshold_validations, present_signatures
        )

        # Determine consciousness level
        consciousness_level = self._determine_consciousness_level(
            threshold_validations, present_signatures
        )

        # Analyze falsification patterns
        is_complete_ignition = self._check_complete_ignition(threshold_validations)
        (
            is_falsification_pattern,
            falsification_confidence,
        ) = self._analyze_falsification_pattern(
            p3b_signature,
            gamma_signature,
            bold_signature,
            pci_signature,
            threshold_validations,
        )

        return CombinedSignature(
            p3b_signature=p3b_signature,
            gamma_signature=gamma_signature,
            bold_signature=bold_signature,
            pci_signature=pci_signature,
            individual_validations=individual_validations,
            threshold_validations=threshold_validations,
            consciousness_level=consciousness_level,
            ignition_probability=ignition_probability,
            signature_coherence=signature_coherence,
            missing_signatures=missing_signatures,
            is_complete_ignition=is_complete_ignition,
            is_falsification_pattern=is_falsification_pattern,
            falsification_confidence=falsification_confidence,
        )

    def _classify_amplitude_level(self, amplitude: float) -> str:
        """Classify P3b amplitude level."""
        if amplitude > self.p3b_threshold:
            return "conscious"
        elif amplitude < self.p3b_unconscious_threshold:
            return "unconscious"
        else:
            return "intermediate"

    def _classify_plv_level(self, max_plv: float) -> str:
        """Classify gamma PLV level."""
        if max_plv > self.gamma_plv_threshold:
            return "conscious"
        elif max_plv < self.gamma_unconscious_plv:
            return "unconscious"
        else:
            return "intermediate"

    def _classify_activation_level(self, peak_activation: float) -> str:
        """Classify BOLD activation level."""
        if peak_activation > self.bold_z_threshold:
            return "conscious"
        elif peak_activation < self.bold_unconscious_z:
            return "unconscious"
        else:
            return "intermediate"

    def _classify_pci_level(self, pci_value: float) -> str:
        """Classify PCI level."""
        if pci_value > self.pci_threshold:
            return "conscious"
        elif pci_value < self.pci_unconscious_threshold:
            return "unconscious"
        else:
            return "intermediate"

    def _calculate_ignition_probability(
        self,
        p3b_signature: Optional[P3bSignature],
        gamma_signature: Optional[GammaSignature],
        bold_signature: Optional[BOLDSignature],
        pci_signature: Optional[PCISignature],
    ) -> float:
        """Calculate overall ignition probability from signatures."""
        probabilities = []

        if p3b_signature:
            # Convert amplitude to probability
            p3b_prob = min(1.0, max(0.0, (p3b_signature.amplitude - 2.0) / 8.0))
            probabilities.append(p3b_prob)

        if gamma_signature:
            # Convert max PLV to probability
            max_plv = (
                max(gamma_signature.plv_values.values()) if gamma_signature.plv_values else 0.0
            )
            gamma_prob = min(1.0, max(0.0, (max_plv - 0.1) / 0.6))
            probabilities.append(gamma_prob)

        if bold_signature:
            # Convert peak activation to probability
            bold_prob = min(1.0, max(0.0, (bold_signature.peak_activation - 1.0) / 6.0))
            probabilities.append(bold_prob)

        if pci_signature:
            # Convert PCI to probability
            pci_prob = min(1.0, max(0.0, (pci_signature.pci_value - 0.1) / 0.6))
            probabilities.append(pci_prob)

        if not probabilities:
            return 0.0

        # Weighted average
        return float(np.mean(probabilities))

    def _calculate_signature_coherence(
        self, threshold_validations: Dict[str, bool], present_signatures: List[str]
    ) -> float:
        """Calculate coherence between signatures."""
        if len(present_signatures) < 2:
            return 1.0 if present_signatures else 0.0

        # Count how many signatures agree on consciousness level
        conscious_count = sum(
            1 for sig in present_signatures if threshold_validations.get(sig, False)
        )
        unconscious_count = len(present_signatures) - conscious_count

        # Coherence is high when signatures agree
        max_agreement = max(conscious_count, unconscious_count)
        coherence = max_agreement / len(present_signatures)

        return coherence

    def _determine_consciousness_level(
        self, threshold_validations: Dict[str, bool], present_signatures: List[str]
    ) -> ConsciousnessLevel:
        """Determine overall consciousness level from signatures."""
        if not present_signatures:
            return ConsciousnessLevel.INDETERMINATE

        conscious_count = sum(
            1 for sig in present_signatures if threshold_validations.get(sig, False)
        )
        total_count = len(present_signatures)
        conscious_ratio = conscious_count / total_count

        if conscious_ratio >= 0.75:
            return ConsciousnessLevel.CONSCIOUS
        elif conscious_ratio <= 0.25:
            return ConsciousnessLevel.UNCONSCIOUS
        elif conscious_ratio >= 0.5:
            return ConsciousnessLevel.PARTIAL_IGNITION
        else:
            return ConsciousnessLevel.SUBTHRESHOLD

    def _check_complete_ignition(self, threshold_validations: Dict[str, bool]) -> bool:
        """Check if all signatures indicate complete ignition."""
        required_signatures = ["p3b", "gamma", "bold", "pci"]
        return all(threshold_validations.get(sig, False) for sig in required_signatures)

    def _analyze_falsification_pattern(
        self,
        p3b_signature: Optional[P3bSignature],
        gamma_signature: Optional[GammaSignature],
        bold_signature: Optional[BOLDSignature],
        pci_signature: Optional[PCISignature],
        threshold_validations: Dict[str, bool],
    ) -> Tuple[bool, float]:
        """Analyze if pattern represents a falsification scenario."""
        # Check for complete ignition signatures
        has_complete_ignition = all(
            threshold_validations.get(sig, False) for sig in ["p3b", "gamma", "bold", "pci"]
        )

        if not has_complete_ignition:
            return False, 0.0

        # Check for AI/ACC suppression (key falsification criterion)
        ai_acc_suppressed = False
        if bold_signature:
            ai_activation, acc_activation = (
                bold_signature.activations.get("anterior_insula", 0.0),
                bold_signature.activations.get("acc", 0.0),
            )
            ai_acc_suppressed = ai_activation < 2.0 and acc_activation < 2.0

        # Check for gamma coherence suppression in AI/ACC
        gamma_ai_acc_suppressed = False
        if gamma_signature:
            # Look for low PLV involving AI/ACC regions
            ai_acc_connections = [
                conn
                for conn in gamma_signature.plv_values.keys()
                if "insula" in conn.lower() or "acc" in conn.lower()
            ]
            if ai_acc_connections:
                ai_acc_plv = np.mean(
                    [gamma_signature.plv_values[conn] for conn in ai_acc_connections]
                )
                gamma_ai_acc_suppressed = bool(ai_acc_plv <= 0.25)

        # Calculate falsification confidence
        falsification_indicators = [
            has_complete_ignition,
            ai_acc_suppressed,
            gamma_ai_acc_suppressed,
        ]

        confidence = sum(falsification_indicators) / len(falsification_indicators)
        is_falsification = confidence >= 0.67  # At least 2/3 indicators

        return is_falsification, confidence

    def validate_combined_signature(self, combined_signature: CombinedSignature) -> bool:
        """Validate a combined signature for consistency and completeness.

        Args:
            combined_signature: CombinedSignature to validate

        Returns:
            True if combined signature is valid
        """
        # Check that at least one signature is present
        signatures_present = [
            combined_signature.p3b_signature is not None,
            combined_signature.gamma_signature is not None,
            combined_signature.bold_signature is not None,
            combined_signature.pci_signature is not None,
        ]

        if not any(signatures_present):
            return False

        # Check that all present signatures are individually valid
        for sig_type, is_valid in combined_signature.individual_validations.items():
            if not is_valid:
                return False

        # Check ignition probability is in valid range
        if not (0.0 <= combined_signature.ignition_probability <= 1.0):
            return False

        # Check signature coherence is in valid range
        if not (0.0 <= combined_signature.signature_coherence <= 1.0):
            return False

        # Check falsification confidence is in valid range
        if not (0.0 <= combined_signature.falsification_confidence <= 1.0):
            return False

        return True

    def get_signature_summary(self, combined_signature: CombinedSignature) -> Dict[str, Any]:
        """Get a summary of the combined signature for reporting.

        Args:
            combined_signature: CombinedSignature to summarize

        Returns:
            Dictionary with signature summary
        """
        summary = {
            "consciousness_level": combined_signature.consciousness_level.value,
            "ignition_probability": combined_signature.ignition_probability,
            "signature_coherence": combined_signature.signature_coherence,
            "is_complete_ignition": combined_signature.is_complete_ignition,
            "is_falsification_pattern": combined_signature.is_falsification_pattern,
            "falsification_confidence": combined_signature.falsification_confidence,
            "present_signatures": [
                sig
                for sig in ["p3b", "gamma", "bold", "pci"]
                if sig not in combined_signature.missing_signatures
            ],
            "missing_signatures": combined_signature.missing_signatures,
            "threshold_validations": combined_signature.threshold_validations,
            "individual_validations": combined_signature.individual_validations,
        }

        # Add signature-specific details
        if combined_signature.p3b_signature:
            summary["p3b_amplitude"] = combined_signature.p3b_signature.amplitude
            summary["p3b_latency"] = combined_signature.p3b_signature.latency

        if combined_signature.gamma_signature:
            max_plv = (
                max(combined_signature.gamma_signature.plv_values.values())
                if combined_signature.gamma_signature.plv_values
                else 0.0
            )
            summary["gamma_max_plv"] = max_plv
            summary["gamma_duration"] = combined_signature.gamma_signature.duration

        if combined_signature.bold_signature:
            summary["bold_peak_activation"] = combined_signature.bold_signature.peak_activation
            summary["bold_bilateral_dlpfc"] = combined_signature.bold_signature.bilateral_dlpfc

        if combined_signature.pci_signature:
            summary["pci_value"] = combined_signature.pci_signature.pci_value
            summary["pci_state"] = combined_signature.pci_signature.state_classification

        return summary
