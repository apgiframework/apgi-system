"""AI/ACC Engagement Validation System.

This module implements validation for Anterior Insula (AI) and Anterior Cingulate Cortex (ACC)
engagement in falsification testing scenarios. According to the APGI Framework, consciousness
requires AI/ACC engagement, so falsification would occur if full ignition signatures are
present without AI/ACC activation.

Key components:
- AI/ACC BOLD response validation
- Gamma coherence checking for AI/ACC regions
- Effective connectivity analysis for AI/ACC-frontoparietal pathways
- Integration with primary falsification testing
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..exceptions import ValidationError


class AIACCRegion(Enum):
    """AI/ACC brain regions for validation."""

    ANTERIOR_INSULA_LEFT = "anterior_insula_left"
    ANTERIOR_INSULA_RIGHT = "anterior_insula_right"
    ANTERIOR_CINGULATE = "anterior_cingulate"
    DORSAL_ACC = "dorsal_acc"
    ROSTRAL_ACC = "rostral_acc"


class ConnectivityType(Enum):
    """Types of connectivity analysis."""

    EFFECTIVE_CONNECTIVITY = "effective"
    FUNCTIONAL_CONNECTIVITY = "functional"
    PHASE_COUPLING = "phase_coupling"


@dataclass
class AIACCActivation:
    """Data model for AI/ACC activation measurements.

    Represents BOLD activation levels in AI/ACC regions for falsification testing.
    """

    region: AIACCRegion
    bold_z_score: float  # Z-score for BOLD activation
    peak_latency: Optional[float] = None  # Peak activation latency (ms)
    activation_duration: Optional[float] = None  # Duration of activation (ms)
    baseline_corrected: bool = True
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate activation parameters."""
        if not isinstance(self.region, AIACCRegion):
            raise ValidationError(f"Invalid AI/ACC region: {self.region}")

    def is_significant_activation(self, threshold: float = 3.1) -> bool:
        """Check if activation is statistically significant.

        Args:
            threshold: Z-score threshold for significance (default 3.1, p < 0.001)

        Returns:
            True if activation exceeds threshold
        """
        return self.bold_z_score > threshold

    def is_absent_activation(self, threshold: float = 3.1) -> bool:
        """Check if activation is absent (below significance threshold).

        Args:
            threshold: Z-score threshold for significance

        Returns:
            True if activation is below threshold (absent)
        """
        return self.bold_z_score <= threshold


@dataclass
class GammaCoherence:
    """Data model for gamma-band coherence in AI/ACC regions.

    Represents phase-locking values and coherence measures for AI/ACC regions.
    """

    source_region: AIACCRegion
    target_region: str  # Can be any brain region
    plv: float  # Phase-locking value (0.0-1.0)
    coherence: float  # Coherence measure (0.0-1.0)
    frequency_band: Tuple[float, float] = (30.0, 80.0)  # Gamma frequency range (Hz)
    duration: float = 0.0  # Duration of coherence (ms)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate coherence parameters."""
        if not 0.0 <= self.plv <= 1.0:
            raise ValidationError(f"PLV must be between 0.0 and 1.0, got {self.plv}")

        if not 0.0 <= self.coherence <= 1.0:
            raise ValidationError(f"Coherence must be between 0.0 and 1.0, got {self.coherence}")

    def is_low_coherence(self, threshold: float = 0.25) -> bool:
        """Check if gamma coherence is low (indicating absent AI/ACC engagement).

        Args:
            threshold: PLV threshold for low coherence

        Returns:
            True if PLV is below threshold
        """
        return self.plv <= threshold

    def is_high_coherence(self, threshold: float = 0.3) -> bool:
        """Check if gamma coherence is high (indicating AI/ACC engagement).

        Args:
            threshold: PLV threshold for high coherence

        Returns:
            True if PLV exceeds threshold
        """
        return self.plv > threshold


@dataclass
class EffectiveConnectivity:
    """Data model for effective connectivity between AI/ACC and other regions.

    Represents directed connectivity measures for AI/ACC-frontoparietal pathways.
    """

    source_region: str
    target_region: str
    connectivity_strength: float  # Connectivity strength measure
    directionality: float  # Directionality index (-1 to 1)
    connectivity_type: ConnectivityType
    frequency_band: Optional[Tuple[float, float]] = None
    latency: Optional[float] = None  # Connection latency (ms)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate connectivity parameters."""
        if not -1.0 <= self.directionality <= 1.0:
            raise ValidationError(
                f"Directionality must be between -1.0 and 1.0, got {self.directionality}"
            )

    def is_strong_connectivity(self, threshold: float = 0.3) -> bool:
        """Check if connectivity is strong.

        Args:
            threshold: Connectivity strength threshold

        Returns:
            True if connectivity exceeds threshold
        """
        return abs(self.connectivity_strength) > threshold

    def is_ai_acc_to_frontoparietal(self) -> bool:
        """Check if this represents AI/ACC to frontoparietal connectivity.

        Returns:
            True if source is AI/ACC and target is frontoparietal
        """
        ai_acc_regions = ["anterior_insula", "anterior_cingulate", "acc"]
        frontoparietal_regions = ["dlpfc", "ips", "parietal", "frontal"]

        source_is_ai_acc = any(region in self.source_region.lower() for region in ai_acc_regions)
        target_is_frontoparietal = any(
            region in self.target_region.lower() for region in frontoparietal_regions
        )

        return source_is_ai_acc and target_is_frontoparietal


@dataclass
class AIACCValidationResult:
    """Complete result of AI/ACC engagement validation.

    Contains all validation measures for AI/ACC engagement in falsification testing.
    """

    trial_id: str
    participant_id: str
    timestamp: datetime

    # AI/ACC BOLD activations
    ai_acc_activations: List[AIACCActivation]

    # Gamma coherence measures
    gamma_coherences: List[GammaCoherence]

    # Effective connectivity measures
    effective_connectivities: List[EffectiveConnectivity]

    # Validation results
    ai_acc_activation_absent: bool
    gamma_coherence_low: bool
    effective_connectivity_disrupted: bool

    # Overall validation
    ai_acc_engagement_absent: bool
    confidence_level: float

    # Additional metrics
    mean_ai_acc_activation: float
    mean_gamma_plv: float
    mean_connectivity_strength: float

    metadata: Dict[str, Any] = field(default_factory=dict)


class AIACCValidator:
    """Validator for AI/ACC engagement in falsification testing.

    Implements validation logic for AI/ACC BOLD responses, gamma coherence,
    and effective connectivity according to falsification criteria.
    """

    def __init__(self) -> None:
        """Initialize the AI/ACC validator with default thresholds."""
        self.validation_thresholds = {
            "bold_significance": 3.1,  # Z-score threshold for significant BOLD activation
            "gamma_plv_low": 0.25,  # PLV threshold for low gamma coherence
            "gamma_plv_high": 0.3,  # PLV threshold for high gamma coherence
            "connectivity_strength": 0.3,  # Threshold for strong connectivity
            "min_duration": 200.0,  # Minimum duration for sustained activity (ms)
        }

        # Key AI/ACC regions for validation
        self.key_ai_acc_regions = [
            AIACCRegion.ANTERIOR_INSULA_LEFT,
            AIACCRegion.ANTERIOR_INSULA_RIGHT,
            AIACCRegion.ANTERIOR_CINGULATE,
        ]

    def validate_ai_acc_activation_absence(
        self, activations: List[AIACCActivation]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate that AI/ACC BOLD activation is absent.

        For falsification, AI/ACC regions should not show significant BOLD activation
        despite the presence of full ignition signatures.

        Args:
            activations: List of AI/ACC activation measurements

        Returns:
            Tuple of (is_absent, validation_details)
        """
        if not activations:
            return True, {"reason": "no_activations_measured", "activations": []}

        validation_details: Dict[str, Any] = {
            "activations": [],
            "significant_activations": 0,
            "mean_activation": 0.0,
            "max_activation": 0.0,
        }

        significant_count = 0
        activation_values = []

        for activation in activations:
            is_significant = activation.is_significant_activation(
                self.validation_thresholds["bold_significance"]
            )

            validation_details["activations"].append(
                {
                    "region": activation.region.value,
                    "z_score": activation.bold_z_score,
                    "is_significant": is_significant,
                }
            )

            if is_significant:
                significant_count += 1

            activation_values.append(activation.bold_z_score)

        validation_details["significant_activations"] = significant_count
        validation_details["mean_activation"] = float(np.mean(activation_values))
        validation_details["max_activation"] = float(np.max(activation_values))

        # AI/ACC activation is considered absent if no regions show significant activation
        is_absent = significant_count == 0

        return is_absent, validation_details

    def validate_gamma_coherence_low(
        self, coherences: List[GammaCoherence]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate that gamma coherence in AI/ACC regions is low.

        For falsification, AI/ACC gamma coherence should be ≤ 0.25 PLV.

        Args:
            coherences: List of gamma coherence measurements

        Returns:
            Tuple of (is_low, validation_details)
        """
        if not coherences:
            return True, {"reason": "no_coherences_measured", "coherences": []}

        validation_details: Dict[str, Any] = {
            "coherences": [],
            "high_coherence_count": 0,
            "mean_plv": 0.0,
            "max_plv": 0.0,
        }

        high_coherence_count = 0
        plv_values = []

        for coherence in coherences:
            is_low = coherence.is_low_coherence(self.validation_thresholds["gamma_plv_low"])

            validation_details["coherences"].append(
                {
                    "source_region": coherence.source_region.value,
                    "target_region": coherence.target_region,
                    "plv": coherence.plv,
                    "is_low": is_low,
                }
            )

            if not is_low:
                high_coherence_count += 1

            plv_values.append(coherence.plv)

        validation_details["high_coherence_count"] = high_coherence_count
        validation_details["mean_plv"] = float(np.mean(plv_values))
        validation_details["max_plv"] = float(np.max(plv_values))

        # Gamma coherence is considered low if all measurements are below threshold
        is_low = high_coherence_count == 0

        return is_low, validation_details

    def validate_effective_connectivity_disruption(
        self, connectivities: List[EffectiveConnectivity]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate that effective connectivity from AI/ACC to frontoparietal regions is disrupted.

        For falsification, AI/ACC-frontoparietal connectivity should be weak or absent.

        Args:
            connectivities: List of effective connectivity measurements

        Returns:
            Tuple of (is_disrupted, validation_details)
        """
        if not connectivities:
            return True, {"reason": "no_connectivities_measured", "connectivities": []}

        validation_details: Dict[str, Any] = {
            "connectivities": [],
            "strong_connections": 0,
            "ai_acc_frontoparietal_connections": 0,
            "mean_connectivity": 0.0,
        }

        strong_connections = 0
        ai_acc_fp_connections = 0
        connectivity_values = []

        for connectivity in connectivities:
            is_strong = connectivity.is_strong_connectivity(
                self.validation_thresholds["connectivity_strength"]
            )
            is_ai_acc_fp = connectivity.is_ai_acc_to_frontoparietal()

            validation_details["connectivities"].append(
                {
                    "source": connectivity.source_region,
                    "target": connectivity.target_region,
                    "strength": connectivity.connectivity_strength,
                    "is_strong": is_strong,
                    "is_ai_acc_frontoparietal": is_ai_acc_fp,
                }
            )

            if is_strong:
                strong_connections += 1
                if is_ai_acc_fp:
                    ai_acc_fp_connections += 1

            connectivity_values.append(abs(connectivity.connectivity_strength))

        validation_details["strong_connections"] = strong_connections
        validation_details["ai_acc_frontoparietal_connections"] = ai_acc_fp_connections
        validation_details["mean_connectivity"] = float(np.mean(connectivity_values))

        # Connectivity is considered disrupted if no strong AI/ACC-frontoparietal connections
        is_disrupted = ai_acc_fp_connections == 0

        return is_disrupted, validation_details

    def validate_complete_ai_acc_engagement_absence(
        self,
        activations: List[AIACCActivation],
        coherences: List[GammaCoherence],
        connectivities: List[EffectiveConnectivity],
        trial_id: str,
        participant_id: str,
    ) -> AIACCValidationResult:
        """Perform complete AI/ACC engagement validation.

        Args:
            activations: AI/ACC BOLD activations
            coherences: Gamma coherence measurements
            connectivities: Effective connectivity measurements
            trial_id: Trial identifier
            participant_id: Participant identifier

        Returns:
            Complete AI/ACC validation result
        """
        timestamp = datetime.now()

        # Validate each component
        activation_absent, activation_details = self.validate_ai_acc_activation_absence(activations)
        coherence_low, coherence_details = self.validate_gamma_coherence_low(coherences)
        (
            connectivity_disrupted,
            connectivity_details,
        ) = self.validate_effective_connectivity_disruption(connectivities)

        # Overall AI/ACC engagement is absent if all criteria are met
        engagement_absent = activation_absent and coherence_low and connectivity_disrupted

        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(
            activation_details, coherence_details, connectivity_details
        )

        # Calculate summary metrics
        mean_activation = float(activation_details.get("mean_activation", 0.0))
        mean_plv = float(coherence_details.get("mean_plv", 0.0))
        mean_connectivity = float(connectivity_details.get("mean_connectivity", 0.0))

        return AIACCValidationResult(
            trial_id=trial_id,
            participant_id=participant_id,
            timestamp=timestamp,
            ai_acc_activations=activations,
            gamma_coherences=coherences,
            effective_connectivities=connectivities,
            ai_acc_activation_absent=activation_absent,
            gamma_coherence_low=coherence_low,
            effective_connectivity_disrupted=connectivity_disrupted,
            ai_acc_engagement_absent=engagement_absent,
            confidence_level=confidence_level,
            mean_ai_acc_activation=mean_activation,
            mean_gamma_plv=mean_plv,
            mean_connectivity_strength=mean_connectivity,
            metadata={
                "activation_details": activation_details,
                "coherence_details": coherence_details,
                "connectivity_details": connectivity_details,
            },
        )

    def _calculate_confidence_level(
        self,
        activation_details: Dict,
        coherence_details: Dict,
        connectivity_details: Dict,
    ) -> float:
        """Calculate confidence level for AI/ACC validation.

        Higher confidence when measurements are clearly below thresholds.

        Args:
            activation_details: BOLD activation validation details
            coherence_details: Gamma coherence validation details
            connectivity_details: Connectivity validation details

        Returns:
            Confidence level (0.0-1.0)
        """
        confidence = 0.5  # Base confidence

        # Boost confidence based on how clearly below thresholds the measurements are

        # BOLD activation confidence
        max_activation = activation_details.get("max_activation", 0.0)
        if max_activation < 2.0:  # Well below threshold
            confidence += 0.2
        elif max_activation < self.validation_thresholds["bold_significance"]:
            confidence += 0.1

        # Gamma coherence confidence
        max_plv = coherence_details.get("max_plv", 0.0)
        if max_plv < 0.2:  # Well below threshold
            confidence += 0.2
        elif max_plv <= self.validation_thresholds["gamma_plv_low"]:
            confidence += 0.1

        # Connectivity confidence
        mean_connectivity = connectivity_details.get("mean_connectivity", 0.0)
        if mean_connectivity < 0.2:  # Well below threshold
            confidence += 0.1

        return min(confidence, 1.0)


class AIACCSimulator:
    """Simulator for AI/ACC engagement measures.

    Generates realistic AI/ACC activation, coherence, and connectivity data
    for falsification testing scenarios.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the AI/ACC simulator.

        Args:
            random_seed: Random seed for reproducible simulations
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.regions = [
            AIACCRegion.ANTERIOR_INSULA_LEFT,
            AIACCRegion.ANTERIOR_INSULA_RIGHT,
            AIACCRegion.ANTERIOR_CINGULATE,
        ]

        self.frontoparietal_regions = [
            "dlpfc_left",
            "dlpfc_right",
            "ips_left",
            "ips_right",
            "parietal_left",
            "parietal_right",
        ]

    def simulate_ai_acc_activations(
        self, engagement_level: str = "absent", n_regions: Optional[int] = None
    ) -> List[AIACCActivation]:
        """Simulate AI/ACC BOLD activations.

        Args:
            engagement_level: 'absent', 'present', or 'threshold'
            n_regions: Number of regions to simulate (default: all key regions)

        Returns:
            List of simulated AI/ACC activations
        """
        if n_regions is None:
            regions_to_simulate = self.regions
        else:
            regions_to_simulate = self.regions[:n_regions]

        activations = []

        for region in regions_to_simulate:
            if engagement_level == "absent":
                # Below significance threshold
                z_score = np.random.uniform(0.5, 2.8)
            elif engagement_level == "present":
                # Above significance threshold
                z_score = np.random.uniform(3.5, 6.0)
            else:  # threshold
                # Around threshold
                z_score = np.random.uniform(2.8, 3.5)

            activation = AIACCActivation(
                region=region,
                bold_z_score=z_score,
                peak_latency=np.random.uniform(200, 400),
                activation_duration=np.random.uniform(150, 300),
            )

            activations.append(activation)

        return activations

    def simulate_gamma_coherences(
        self, engagement_level: str = "absent", n_connections: int = 6
    ) -> List[GammaCoherence]:
        """Simulate gamma coherence measures for AI/ACC regions.

        Args:
            engagement_level: 'absent', 'present', or 'threshold'
            n_connections: Number of connections to simulate

        Returns:
            List of simulated gamma coherence measures
        """
        coherences = []

        for i in range(n_connections):
            source_region = np.random.choice(self.regions)  # type: ignore
            target_region = np.random.choice(self.frontoparietal_regions)

            if engagement_level == "absent":
                # Low PLV (≤ 0.25)
                plv = np.random.uniform(0.05, 0.25)
                coherence_val = np.random.uniform(0.1, 0.3)
            elif engagement_level == "present":
                # High PLV (> 0.3)
                plv = np.random.uniform(0.35, 0.7)
                coherence_val = np.random.uniform(0.4, 0.8)
            else:  # threshold
                # Around threshold
                plv = np.random.uniform(0.2, 0.35)
                coherence_val = np.random.uniform(0.25, 0.45)

            gamma_coherence = GammaCoherence(
                source_region=source_region,
                target_region=target_region,
                plv=plv,
                coherence=coherence_val,
                duration=np.random.uniform(200, 400),
            )

            coherences.append(gamma_coherence)

        return coherences

    def simulate_effective_connectivities(
        self, engagement_level: str = "absent", n_connections: int = 8
    ) -> List[EffectiveConnectivity]:
        """Simulate effective connectivity measures.

        Args:
            engagement_level: 'absent', 'present', or 'threshold'
            n_connections: Number of connections to simulate

        Returns:
            List of simulated effective connectivity measures
        """
        connectivities = []

        ai_acc_region_names = [
            "anterior_insula_left",
            "anterior_insula_right",
            "anterior_cingulate",
        ]

        for i in range(n_connections):
            source_region = np.random.choice(ai_acc_region_names)
            target_region = np.random.choice(self.frontoparietal_regions)

            if engagement_level == "absent":
                # Weak connectivity
                strength = np.random.uniform(0.05, 0.25)
            elif engagement_level == "present":
                # Strong connectivity
                strength = np.random.uniform(0.4, 0.8)
            else:  # threshold
                # Around threshold
                strength = np.random.uniform(0.25, 0.4)

            # Add some bidirectional connections
            if np.random.random() < 0.3:
                strength = -strength  # Reverse direction

            connectivity = EffectiveConnectivity(
                source_region=source_region,
                target_region=target_region,
                connectivity_strength=strength,
                directionality=np.random.uniform(-0.5, 0.8),
                connectivity_type=ConnectivityType.EFFECTIVE_CONNECTIVITY,
                frequency_band=(30.0, 80.0),
                latency=np.random.uniform(50, 150),
            )

            connectivities.append(connectivity)

        return connectivities

    def simulate_complete_ai_acc_measures(
        self,
        engagement_level: str = "absent",
        trial_id: str = "sim_trial",
        participant_id: str = "sim_participant",
    ) -> AIACCValidationResult:
        """Simulate complete AI/ACC engagement measures.

        Args:
            engagement_level: 'absent', 'present', or 'threshold'
            trial_id: Trial identifier
            participant_id: Participant identifier

        Returns:
            Complete simulated AI/ACC validation result
        """
        activations = self.simulate_ai_acc_activations(engagement_level)
        coherences = self.simulate_gamma_coherences(engagement_level)
        connectivities = self.simulate_effective_connectivities(engagement_level)

        validator = AIACCValidator()

        return validator.validate_complete_ai_acc_engagement_absence(
            activations, coherences, connectivities, trial_id, participant_id
        )
