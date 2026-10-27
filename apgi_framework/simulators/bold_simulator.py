"""BOLD activation simulator for APGI Framework falsification testing.

This module implements the BOLDSimulator class that generates realistic BOLD fMRI
activation patterns with Z-scores for testing consciousness-related brain activation
in key regions like dorsolateral prefrontal cortex, intraparietal sulcus, and
anterior insula/ACC.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class BrainRegion(Enum):
    """Enumeration of brain regions for BOLD activation analysis."""

    DLPFC_LEFT = "dlpfc_left"
    DLPFC_RIGHT = "dlpfc_right"
    IPS_LEFT = "ips_left"
    IPS_RIGHT = "ips_right"
    ANTERIOR_INSULA = "anterior_insula"
    ACC = "acc"
    POSTERIOR_PARIETAL_LEFT = "posterior_parietal_left"
    POSTERIOR_PARIETAL_RIGHT = "posterior_parietal_right"
    FRONTAL_EYE_FIELDS_LEFT = "fef_left"
    FRONTAL_EYE_FIELDS_RIGHT = "fef_right"


@dataclass
class BOLDSignature:
    """Data class representing BOLD activation signature."""

    activations: Dict[str, float]  # Z-scores for each brain region
    peak_activation: float  # Maximum Z-score across regions
    active_regions: List[str]  # Regions above threshold
    bilateral_dlpfc: bool  # Whether bilateral DLPFC is active
    is_conscious_threshold: bool  # Whether Z > 3.1 threshold met
    network_coherence: float  # Coherence across frontoparietal network


class BOLDSimulator:
    """Simulator for BOLD activation patterns with realistic Z-score characteristics.

    BOLD activation with Z > 3.1 in bilateral dorsolateral prefrontal cortex,
    intraparietal sulcus, and anterior insula/ACC indicates conscious processing
    and global workspace activation.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize BOLD activation simulator.

        Args:
            random_seed: Optional seed for reproducible random number generation
        """
        self.rng = np.random.RandomState(random_seed)
        self.conscious_z_threshold = 3.1
        self.unconscious_z_threshold = 2.0

        # Define key regions for consciousness
        self.core_consciousness_regions = [
            BrainRegion.DLPFC_LEFT.value,
            BrainRegion.DLPFC_RIGHT.value,
            BrainRegion.IPS_LEFT.value,
            BrainRegion.IPS_RIGHT.value,
            BrainRegion.ANTERIOR_INSULA.value,
            BrainRegion.ACC.value,
        ]

        self.frontoparietal_network = [
            BrainRegion.DLPFC_LEFT.value,
            BrainRegion.DLPFC_RIGHT.value,
            BrainRegion.IPS_LEFT.value,
            BrainRegion.IPS_RIGHT.value,
            BrainRegion.POSTERIOR_PARIETAL_LEFT.value,
            BrainRegion.POSTERIOR_PARIETAL_RIGHT.value,
            BrainRegion.FRONTAL_EYE_FIELDS_LEFT.value,
            BrainRegion.FRONTAL_EYE_FIELDS_RIGHT.value,
        ]

        # Define bilateral pairs
        self.bilateral_pairs = [
            (BrainRegion.DLPFC_LEFT.value, BrainRegion.DLPFC_RIGHT.value),
            (BrainRegion.IPS_LEFT.value, BrainRegion.IPS_RIGHT.value),
            (
                BrainRegion.POSTERIOR_PARIETAL_LEFT.value,
                BrainRegion.POSTERIOR_PARIETAL_RIGHT.value,
            ),
            (
                BrainRegion.FRONTAL_EYE_FIELDS_LEFT.value,
                BrainRegion.FRONTAL_EYE_FIELDS_RIGHT.value,
            ),
        ]

    def generate_conscious_signature(
        self,
        z_range: Tuple[float, float] = (3.5, 8.0),
        noise_level: float = 0.3,
        bilateral_coherence: float = 0.8,
    ) -> BOLDSignature:
        """Generate BOLD activation signature indicating conscious processing.

        Args:
            z_range: Range for Z-score generation, default ensures >3.1
            noise_level: Standard deviation of noise to add to Z-scores
            bilateral_coherence: Correlation between bilateral regions (0-1)

        Returns:
            BOLDSignature with Z > 3.1 threshold in key regions
        """
        activations = {}

        # Generate high activation for core consciousness regions
        for region in self.core_consciousness_regions:
            base_z = self.rng.uniform(z_range[0], z_range[1])
            noise = self.rng.normal(0, noise_level)
            final_z = base_z + noise
            # Ensure stays above threshold
            final_z = max(final_z, self.conscious_z_threshold + 0.1)
            activations[region] = final_z

        # Apply bilateral coherence for paired regions
        for left_region, right_region in self.bilateral_pairs:
            if left_region in activations and right_region in activations:
                # Make right hemisphere similar to left with some correlation
                left_z = activations[left_region]
                coherence_factor = self.rng.uniform(
                    bilateral_coherence - 0.1, bilateral_coherence + 0.1
                )
                right_base = left_z * coherence_factor
                right_noise = self.rng.normal(0, noise_level * 0.5)
                activations[right_region] = max(
                    right_base + right_noise, self.conscious_z_threshold + 0.1
                )

        # Generate moderate activation for extended frontoparietal network
        for region in self.frontoparietal_network:
            if region not in activations:
                base_z = self.rng.uniform(z_range[0] * 0.7, z_range[1] * 0.8)
                noise = self.rng.normal(0, noise_level)
                final_z = base_z + noise
                final_z = max(final_z, self.conscious_z_threshold * 0.8)
                activations[region] = final_z

        return self._create_signature(activations)

    def generate_unconscious_signature(
        self, z_range: Tuple[float, float] = (0.5, 1.8), noise_level: float = 0.2
    ) -> BOLDSignature:
        """Generate BOLD activation signature indicating unconscious/subthreshold processing.

        Args:
            z_range: Range for Z-score generation, default ensures <3.1
            noise_level: Standard deviation of noise to add to Z-scores

        Returns:
            BOLDSignature with Z < 3.1 threshold
        """
        activations = {}

        # Generate low activation for all regions
        all_regions = self.core_consciousness_regions + [
            r for r in self.frontoparietal_network if r not in self.core_consciousness_regions
        ]

        for region in all_regions:
            base_z = self.rng.uniform(z_range[0], z_range[1])
            noise = self.rng.normal(0, noise_level)
            final_z = base_z + noise
            # Ensure stays below threshold
            final_z = min(final_z, self.unconscious_z_threshold - 0.1)
            final_z = max(final_z, 0.0)  # Ensure non-negative
            activations[region] = final_z

        return self._create_signature(activations)

    def generate_signature(
        self, target_regions: List[str], target_z: float, noise_level: float = 0.25
    ) -> BOLDSignature:
        """Generate BOLD activation signature with specific target parameters.

        Args:
            target_regions: List of regions to activate
            target_z: Target Z-score for specified regions
            noise_level: Standard deviation of noise to add

        Returns:
            BOLDSignature with specified characteristics
        """
        activations = {}

        # Set target activation for specified regions
        for region in target_regions:
            noise = self.rng.normal(0, noise_level)
            final_z = max(0.0, target_z + noise)
            activations[region] = final_z

        # Generate background activation for other regions
        all_regions = self.core_consciousness_regions + [
            r for r in self.frontoparietal_network if r not in self.core_consciousness_regions
        ]

        for region in all_regions:
            if region not in activations:
                background_z = self.rng.uniform(0.2, 1.5)
                noise = self.rng.normal(0, noise_level * 0.5)
                activations[region] = max(0.0, background_z + noise)

        return self._create_signature(activations)

    def generate_ai_acc_suppressed_signature(
        self,
        dlpfc_z_range: Tuple[float, float] = (3.5, 7.0),
        ai_acc_z_range: Tuple[float, float] = (0.2, 1.5),
        noise_level: float = 0.3,
    ) -> BOLDSignature:
        """Generate signature with high DLPFC/IPS but suppressed AI/ACC activation.
        This pattern is critical for primary falsification testing.

        Args:
            dlpfc_z_range: Z-score range for DLPFC/IPS regions
            ai_acc_z_range: Z-score range for AI/ACC regions (should be low)
            noise_level: Standard deviation of noise to add

        Returns:
            BOLDSignature with selective activation pattern
        """
        activations = {}

        # High activation for DLPFC and IPS
        dlpfc_ips_regions = [
            BrainRegion.DLPFC_LEFT.value,
            BrainRegion.DLPFC_RIGHT.value,
            BrainRegion.IPS_LEFT.value,
            BrainRegion.IPS_RIGHT.value,
        ]

        for region in dlpfc_ips_regions:
            base_z = self.rng.uniform(dlpfc_z_range[0], dlpfc_z_range[1])
            noise = self.rng.normal(0, noise_level)
            final_z = max(base_z + noise, self.conscious_z_threshold + 0.1)
            activations[region] = final_z

        # Low activation for AI/ACC
        ai_acc_regions = [BrainRegion.ANTERIOR_INSULA.value, BrainRegion.ACC.value]

        for region in ai_acc_regions:
            base_z = self.rng.uniform(ai_acc_z_range[0], ai_acc_z_range[1])
            noise = self.rng.normal(0, noise_level * 0.5)
            final_z = min(max(base_z + noise, 0.0), self.unconscious_z_threshold)
            activations[region] = final_z

        # Moderate activation for other frontoparietal regions
        other_regions = [r for r in self.frontoparietal_network if r not in dlpfc_ips_regions]

        for region in other_regions:
            base_z = self.rng.uniform(2.0, 4.0)
            noise = self.rng.normal(0, noise_level)
            activations[region] = max(0.0, base_z + noise)

        return self._create_signature(activations)

    def _create_signature(self, activations: Dict[str, float]) -> BOLDSignature:
        """Create BOLDSignature from activation dictionary."""
        peak_activation = max(activations.values()) if activations else 0.0
        active_regions = [
            region for region, z in activations.items() if z > self.conscious_z_threshold
        ]

        # Check bilateral DLPFC activation
        bilateral_dlpfc = (
            activations.get(BrainRegion.DLPFC_LEFT.value, 0) > self.conscious_z_threshold
            and activations.get(BrainRegion.DLPFC_RIGHT.value, 0) > self.conscious_z_threshold
        )

        # Check conscious threshold
        is_conscious = any(z > self.conscious_z_threshold for z in activations.values())

        # Calculate network coherence
        network_coherence = self._calculate_network_coherence(activations)

        return BOLDSignature(
            activations=activations,
            peak_activation=peak_activation,
            active_regions=active_regions,
            bilateral_dlpfc=bilateral_dlpfc,
            is_conscious_threshold=is_conscious,
            network_coherence=network_coherence,
        )

    def _calculate_network_coherence(self, activations: Dict[str, float]) -> float:
        """Calculate coherence across frontoparietal network."""
        network_activations = [
            activations.get(region, 0.0) for region in self.frontoparietal_network
        ]

        if not network_activations:
            return 0.0

        # Calculate coefficient of variation (inverse of coherence)
        mean_activation = np.mean(network_activations)
        if mean_activation == 0:
            return 0.0

        std_activation = np.std(network_activations)
        cv = std_activation / mean_activation

        # Convert to coherence measure (0-1, higher is more coherent)
        coherence = 1.0 / (1.0 + cv)
        return float(coherence)

    def validate_signature(self, signature: BOLDSignature) -> bool:
        """Validate that BOLD signature meets basic physiological constraints.

        Args:
            signature: BOLDSignature to validate

        Returns:
            True if signature is physiologically plausible
        """
        # Check Z-scores are non-negative and within reasonable range
        for z_score in signature.activations.values():
            if z_score < 0 or z_score > 15.0:  # Very high Z-scores are implausible
                return False

        # Check peak activation matches maximum
        expected_peak = max(signature.activations.values()) if signature.activations else 0.0
        if abs(signature.peak_activation - expected_peak) > 0.01:
            return False

        # Check active regions consistency
        expected_active = [
            region for region, z in signature.activations.items() if z > self.conscious_z_threshold
        ]
        if set(signature.active_regions) != set(expected_active):
            return False

        # Check network coherence is in valid range
        if not (0.0 <= signature.network_coherence <= 1.0):
            return False

        return True

    def is_conscious_level(self, signature: BOLDSignature) -> bool:
        """Determine if BOLD signature indicates conscious-level processing.

        Args:
            signature: BOLDSignature to evaluate

        Returns:
            True if activation exceeds conscious threshold (Z > 3.1)
        """
        return signature.is_conscious_threshold

    def has_bilateral_dlpfc_activation(self, signature: BOLDSignature) -> bool:
        """Check if signature has bilateral DLPFC activation."""
        return signature.bilateral_dlpfc

    def get_ai_acc_activation(self, signature: BOLDSignature) -> Tuple[float, float]:
        """Get AI and ACC activation levels."""
        ai_activation = signature.activations.get(BrainRegion.ANTERIOR_INSULA.value, 0.0)
        acc_activation = signature.activations.get(BrainRegion.ACC.value, 0.0)
        return ai_activation, acc_activation

    def is_ai_acc_suppressed(self, signature: BOLDSignature) -> bool:
        """Check if AI/ACC activation is suppressed (below threshold)."""
        ai_activation, acc_activation = self.get_ai_acc_activation(signature)
        return (
            ai_activation < self.unconscious_z_threshold
            and acc_activation < self.unconscious_z_threshold
        )

    def simulate_bold_activation(
        self, regions: List[str], z_range: Tuple[float, float]
    ) -> "BOLDResult":
        """Legacy method for backward compatibility with falsification tests.

        Args:
            regions: List of brain regions to simulate
            z_range: Range for Z-score generation

        Returns:
            BOLDResult with activations attribute for compatibility
        """
        # Determine if this should be conscious or unconscious based on Z range
        if z_range[0] > self.conscious_z_threshold:
            signature = self.generate_conscious_signature(z_range)
        elif z_range[1] < self.conscious_z_threshold:
            signature = self.generate_unconscious_signature(z_range)
        else:
            # Mixed range - use target regions with midpoint Z-score
            target_z = (z_range[0] + z_range[1]) / 2
            signature = self.generate_signature(regions, target_z)

        # Return compatibility object
        return BOLDResult(signature)


class BOLDResult:
    """Compatibility wrapper for BOLD simulation results."""

    def __init__(self, signature: BOLDSignature):
        self.signature = signature
        self.activations = signature.activations
