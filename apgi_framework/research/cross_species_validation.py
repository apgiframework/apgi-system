"""Cross-species validation module for APGI Framework.

Provides comparative analysis across different species to validate the generalizability
of active inference principles and neural signatures of consciousness.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

from ..exceptions import ValidationError
from ..logging.standardized_logging import get_logger

logger = get_logger("cross_species_validation")


class Species(Enum):
    """Supported species for cross-species validation."""

    HUMAN = "human"
    RODENT = "rodent"
    PRIMATE = "primate"
    CANINE = "canine"
    FELINE = "feline"
    AVIAN = "avian"


class NeuralSignatureType(Enum):
    """Types of neural signatures for cross-species comparison."""

    P3B = "p3b"
    GAMMA_SYNC = "gamma_synchrony"
    HEP = "heartbeat_evoked_potential"
    MICROSATE = "microstate"
    BOLD = "bold_response"


@dataclass
class SpeciesParameters:
    """Species-specific physiological and neural parameters."""

    species: Species
    brain_mass_g: float  # Brain mass in grams
    metabolic_rate_w: float  # Metabolic rate in watts
    heart_rate_bpm: float  # Resting heart rate
    body_temp_c: float  # Body temperature in Celsius
    lifespan_years: float  # Average lifespan

    # Neural parameters
    eeg_frequency_range: Tuple[float, float]  # Hz
    p3b_latency_range: Tuple[float, float]  # ms
    gamma_frequency_range: Tuple[float, float]  # Hz

    # Scaling factors for cross-species comparison
    temporal_scaling: float = 1.0  # Time scaling factor
    amplitude_scaling: float = 1.0  # Amplitude scaling factor
    frequency_scaling: float = 1.0  # Frequency scaling factor


@dataclass
class CrossSpeciesSignature:
    """Neural signature data for cross-species validation."""

    species: Species
    signature_type: NeuralSignatureType
    participant_id: str
    raw_data: np.ndarray
    timestamps: np.ndarray

    # Extracted features
    amplitude: Optional[float] = None
    latency: Optional[float] = None
    frequency: Optional[float] = None
    duration: Optional[float] = None

    # Quality metrics
    signal_to_noise: Optional[float] = None
    artifact_rate: Optional[float] = None

    # Metadata
    experimental_condition: Optional[str] = None
    recording_location: Optional[str] = None


@dataclass
class ValidationResult:
    """Results of cross-species validation analysis."""

    signature_type: NeuralSignatureType
    species_comparison: Tuple[Species, Species]

    # Statistical comparison
    correlation_coefficient: float
    p_value: float
    effect_size: float

    # Similarity metrics
    temporal_similarity: float
    amplitude_similarity: float
    frequency_similarity: float

    # Overall validation score
    validation_score: float
    significance_threshold: float = 0.05

    is_significant: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_significant = self.p_value < self.significance_threshold


class CrossSpeciesValidator:
    """Main class for cross-species validation of APGI signatures.

    Provides comprehensive analysis tools for comparing neural signatures
    across different species and validating the generalizability of findings.
    """

    def __init__(self, significance_threshold: float = 0.05):
        """Initialize cross-species validator.

        Args:
            significance_threshold: Statistical significance threshold
        """
        self.significance_threshold = significance_threshold
        self.scaler = StandardScaler()
        self.species_params = self._initialize_species_parameters()

        logger.info(f"CrossSpeciesValidator initialized with threshold {significance_threshold}")

    def _initialize_species_parameters(self) -> Dict[Species, SpeciesParameters]:
        """Initialize species-specific parameters."""
        return {
            Species.HUMAN: SpeciesParameters(
                species=Species.HUMAN,
                brain_mass_g=1350,
                metabolic_rate_w=85.0,
                heart_rate_bpm=70,
                body_temp_c=37.0,
                lifespan_years=80,
                eeg_frequency_range=(0.5, 100),
                p3b_latency_range=(250, 500),
                gamma_frequency_range=(30, 100),
                temporal_scaling=1.0,
                amplitude_scaling=1.0,
                frequency_scaling=1.0,
            ),
            Species.RODENT: SpeciesParameters(
                species=Species.RODENT,
                brain_mass_g=2.0,
                metabolic_rate_w=0.25,
                heart_rate_bpm=350,
                body_temp_c=37.5,
                lifespan_years=3,
                eeg_frequency_range=(1.0, 200),
                p3b_latency_range=(50, 100),
                gamma_frequency_range=(40, 150),
                temporal_scaling=0.1,  # Faster processing
                amplitude_scaling=0.1,  # Smaller amplitudes
                frequency_scaling=2.0,  # Higher frequencies
            ),
            Species.PRIMATE: SpeciesParameters(
                species=Species.PRIMATE,
                brain_mass_g=400,
                metabolic_rate_w=15.0,
                heart_rate_bpm=80,
                body_temp_c=38.0,
                lifespan_years=25,
                eeg_frequency_range=(0.5, 120),
                p3b_latency_range=(200, 400),
                gamma_frequency_range=(30, 120),
                temporal_scaling=0.8,
                amplitude_scaling=0.7,
                frequency_scaling=1.2,
            ),
            Species.CANINE: SpeciesParameters(
                species=Species.CANINE,
                brain_mass_g=70,
                metabolic_rate_w=5.0,
                heart_rate_bpm=100,
                body_temp_c=38.5,
                lifespan_years=15,
                eeg_frequency_range=(1.0, 150),
                p3b_latency_range=(150, 300),
                gamma_frequency_range=(35, 130),
                temporal_scaling=0.6,
                amplitude_scaling=0.5,
                frequency_scaling=1.5,
            ),
            Species.FELINE: SpeciesParameters(
                species=Species.FELINE,
                brain_mass_g=30,
                metabolic_rate_w=2.5,
                heart_rate_bpm=120,
                body_temp_c=38.8,
                lifespan_years=18,
                eeg_frequency_range=(1.0, 160),
                p3b_latency_range=(120, 250),
                gamma_frequency_range=(40, 140),
                temporal_scaling=0.5,
                amplitude_scaling=0.4,
                frequency_scaling=1.6,
            ),
            Species.AVIAN: SpeciesParameters(
                species=Species.AVIAN,
                brain_mass_g=10,
                metabolic_rate_w=1.0,
                heart_rate_bpm=200,
                body_temp_c=40.0,
                lifespan_years=10,
                eeg_frequency_range=(2.0, 250),
                p3b_latency_range=(80, 150),
                gamma_frequency_range=(50, 200),
                temporal_scaling=0.3,
                amplitude_scaling=0.3,
                frequency_scaling=2.5,
            ),
        }

    def add_signature(self, signature: CrossSpeciesSignature) -> None:
        """Add a neural signature for cross-species validation.

        Args:
            signature: Cross-species signature data
        """
        # Validate signature data
        self._validate_signature(signature)

        # Extract features if not already done
        if signature.amplitude is None:
            signature.amplitude = self._extract_amplitude(signature)
        if signature.latency is None:
            signature.latency = self._extract_latency(signature)
        if signature.frequency is None:
            signature.frequency = self._extract_frequency(signature)

        logger.info(
            f"Added {signature.signature_type.value} signature for {signature.species.value}"
        )

    def _validate_signature(self, signature: CrossSpeciesSignature) -> None:
        """Validate signature data integrity."""
        if signature.species not in self.species_params:
            raise ValidationError(f"Unsupported species: {signature.species}")

        if len(signature.raw_data) != len(signature.timestamps):
            raise ValidationError("Raw data and timestamps must have same length")

        if len(signature.raw_data) < 10:
            raise ValidationError("Insufficient data points for analysis")

    def _extract_amplitude(self, signature: CrossSpeciesSignature) -> float:
        """Extract amplitude from neural signature."""
        if signature.signature_type == NeuralSignatureType.P3B:
            # Find peak amplitude in expected time window
            params = self.species_params[signature.species]
            latency_min, latency_max = params.p3b_latency_range

            # Convert to sample indices
            fs = 1.0 / np.mean(np.diff(signature.timestamps))
            start_idx = int(latency_min * fs / 1000)
            end_idx = int(latency_max * fs / 1000)

            # Extract peak amplitude
            window_data = signature.raw_data[start_idx:end_idx]
            return float(np.max(np.abs(window_data)))

        elif signature.signature_type == NeuralSignatureType.GAMMA_SYNC:
            # Extract gamma band power
            params = self.species_params[signature.species]
            freq_min, freq_max = params.gamma_frequency_range

            # Simple power calculation (would use FFT in production)
            return float(np.sqrt(np.mean(signature.raw_data**2)))

        else:
            # Generic amplitude extraction
            return float(np.max(np.abs(signature.raw_data)))

    def _extract_latency(self, signature: CrossSpeciesSignature) -> float:
        """Extract latency from neural signature."""
        if signature.signature_type == NeuralSignatureType.P3B:
            # Find peak latency
            params = self.species_params[signature.species]
            latency_min, latency_max = params.p3b_latency_range

            fs = 1.0 / np.mean(np.diff(signature.timestamps))
            start_idx = int(latency_min * fs / 1000)
            end_idx = int(latency_max * fs / 1000)

            window_data = signature.raw_data[start_idx:end_idx]
            peak_idx = np.argmax(np.abs(window_data)) + start_idx

            return float(signature.timestamps[peak_idx] * 1000)  # Convert to ms

        else:
            # Generic latency (time to peak)
            peak_idx = np.argmax(np.abs(signature.raw_data))
            return float(signature.timestamps[peak_idx] * 1000)

    def _extract_frequency(self, signature: CrossSpeciesSignature) -> float:
        """Extract dominant frequency from neural signature."""
        # Simple frequency estimation (would use FFT in production)
        fs = 1.0 / np.mean(np.diff(signature.timestamps))

        # Count zero crossings as frequency estimate
        zero_crossings = np.where(np.diff(np.sign(signature.raw_data)))[0]
        if len(zero_crossings) > 1:
            period = np.mean(np.diff(zero_crossings)) / fs
            return float(1.0 / (2 * period))  # Convert to Hz
        else:
            return float(fs / 2)  # Nyquist frequency as fallback

    def compare_signatures(
        self, species1: Species, species2: Species, signature_type: NeuralSignatureType
    ) -> ValidationResult:
        """Compare neural signatures between two species.

        Args:
            species1: First species to compare
            species2: Second species to compare
            signature_type: Type of neural signature to compare

        Returns:
            ValidationResult with comparison statistics
        """
        logger.info(
            f"Comparing {signature_type.value} between {species1.value} and {species2.value}"
        )

        # Get signatures for both species (would retrieve from database in production)
        signatures1 = self._get_signatures_for_species(species1, signature_type)
        signatures2 = self._get_signatures_for_species(species2, signature_type)

        if not signatures1 or not signatures2:
            raise ValidationError("No signatures found for comparison")

        # Extract features for comparison
        features1 = self._extract_features_batch(signatures1)
        features2 = self._extract_features_batch(signatures2)

        # Apply species-specific scaling
        scaled_features1 = self._apply_species_scaling(features1, species1)
        scaled_features2 = self._apply_species_scaling(features2, species2)

        # Statistical comparison
        correlation, p_value = self._compute_correlation(scaled_features1, scaled_features2)
        effect_size = self._compute_effect_size(scaled_features1, scaled_features2)

        # Similarity metrics
        temporal_sim = self._compute_temporal_similarity(signatures1, signatures2)
        amplitude_sim = self._compute_amplitude_similarity(signatures1, signatures2)
        frequency_sim = self._compute_frequency_similarity(signatures1, signatures2)

        # Overall validation score
        validation_score = self._compute_validation_score(
            correlation, temporal_sim, amplitude_sim, frequency_sim
        )

        result = ValidationResult(
            signature_type=signature_type,
            species_comparison=(species1, species2),
            correlation_coefficient=correlation,
            p_value=p_value,
            effect_size=effect_size,
            temporal_similarity=temporal_sim,
            amplitude_similarity=amplitude_sim,
            frequency_similarity=frequency_sim,
            validation_score=validation_score,
            significance_threshold=self.significance_threshold,
        )

        logger.info(f"Validation completed: score={validation_score:.3f}, p={p_value:.3f}")
        return result

    def _get_signatures_for_species(
        self, species: Species, signature_type: NeuralSignatureType
    ) -> List[CrossSpeciesSignature]:
        """Get signatures for a specific species (mock implementation)."""
        # In production, this would retrieve from database
        # For now, return mock data
        return []

    def _extract_features_batch(self, signatures: List[CrossSpeciesSignature]) -> np.ndarray:
        """Extract features from a batch of signatures."""
        features = []
        for sig in signatures:
            feature_vector = [
                sig.amplitude or 0,
                sig.latency or 0,
                sig.frequency or 0,
                sig.signal_to_noise or 0,
                1.0 - (sig.artifact_rate or 0),  # Quality score
            ]
            features.append(feature_vector)

        return np.array(features)

    def _apply_species_scaling(self, features: np.ndarray, species: Species) -> np.ndarray:
        """Apply species-specific scaling to features."""
        params = self.species_params[species]

        # Apply scaling factors
        scaled_features = features.copy()
        scaled_features[:, 0] *= params.amplitude_scaling  # Amplitude
        scaled_features[:, 1] *= params.temporal_scaling  # Latency
        scaled_features[:, 2] *= params.frequency_scaling  # Frequency

        return cast(np.ndarray, scaled_features)

    def _compute_correlation(
        self, features1: np.ndarray, features2: np.ndarray
    ) -> Tuple[float, float]:
        """Compute correlation between feature sets."""
        # Average features across samples
        avg_features1 = np.mean(features1, axis=0)
        avg_features2 = np.mean(features2, axis=0)

        # Compute correlation
        correlation, p_value = stats.pearsonr(avg_features1, avg_features2)

        return float(correlation), float(p_value)

    def _compute_effect_size(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        mean1, mean2 = np.mean(features1, axis=0), np.mean(features2, axis=0)
        std1, std2 = np.std(features1, axis=0), np.std(features2, axis=0)

        pooled_std = np.sqrt(
            ((len(features1) - 1) * std1**2 + (len(features2) - 1) * std2**2)
            / (len(features1) + len(features2) - 2)
        )

        effect_sizes = (mean1 - mean2) / pooled_std
        return float(np.mean(effect_sizes))

    def _compute_temporal_similarity(
        self,
        signatures1: List[CrossSpeciesSignature],
        signatures2: List[CrossSpeciesSignature],
    ) -> float:
        """Compute temporal similarity between signature sets."""
        latencies1 = [s.latency for s in signatures1 if s.latency is not None]
        latencies2 = [s.latency for s in signatures2 if s.latency is not None]

        if not latencies1 or not latencies2:
            return 0.0

        # Normalize latencies and compute similarity
        norm_latencies1 = np.array(latencies1) / np.mean(latencies1)
        norm_latencies2 = np.array(latencies2) / np.mean(latencies2)

        similarity = 1.0 - np.mean(np.abs(norm_latencies1 - norm_latencies2))
        return float(max(0.0, similarity))

    def _compute_amplitude_similarity(
        self,
        signatures1: List[CrossSpeciesSignature],
        signatures2: List[CrossSpeciesSignature],
    ) -> float:
        """Compute amplitude similarity between signature sets."""
        amplitudes1 = [s.amplitude for s in signatures1 if s.amplitude is not None]
        amplitudes2 = [s.amplitude for s in signatures2 if s.amplitude is not None]

        if not amplitudes1 or not amplitudes2:
            return 0.0

        # Normalize amplitudes and compute similarity
        norm_amplitudes1 = np.array(amplitudes1) / np.mean(amplitudes1)
        norm_amplitudes2 = np.array(amplitudes2) / np.mean(amplitudes2)

        similarity = 1.0 - np.mean(np.abs(norm_amplitudes1 - norm_amplitudes2))
        return float(max(0.0, similarity))

    def _compute_frequency_similarity(
        self,
        signatures1: List[CrossSpeciesSignature],
        signatures2: List[CrossSpeciesSignature],
    ) -> float:
        """Compute frequency similarity between signature sets."""
        frequencies1 = [s.frequency for s in signatures1 if s.frequency is not None]
        frequencies2 = [s.frequency for s in signatures2 if s.frequency is not None]

        if not frequencies1 or not frequencies2:
            return 0.0

        # Normalize frequencies and compute similarity
        norm_frequencies1 = np.array(frequencies1) / np.mean(frequencies1)
        norm_frequencies2 = np.array(frequencies2) / np.mean(frequencies2)

        similarity = 1.0 - np.mean(np.abs(norm_frequencies1 - norm_frequencies2))
        return float(max(0.0, similarity))

    def _compute_validation_score(
        self,
        correlation: float,
        temporal_sim: float,
        amplitude_sim: float,
        frequency_sim: float,
    ) -> float:
        """Compute overall validation score."""
        # Weighted combination of similarity metrics
        weights = {
            "correlation": 0.4,
            "temporal": 0.2,
            "amplitude": 0.2,
            "frequency": 0.2,
        }

        score = (
            weights["correlation"] * abs(correlation)
            + weights["temporal"] * temporal_sim
            + weights["amplitude"] * amplitude_sim
            + weights["frequency"] * frequency_sim
        )

        return float(score)

    def run_comprehensive_validation(
        self, signature_types: List[NeuralSignatureType]
    ) -> Dict[str, List[ValidationResult]]:
        """Run comprehensive cross-species validation for all signature types.

        Args:
            signature_types: List of signature types to validate

        Returns:
            Dictionary mapping signature types to validation results
        """
        logger.info("Starting comprehensive cross-species validation")

        results: Dict[str, List[ValidationResult]] = {}
        species_list = list(Species)

        for signature_type in signature_types:
            results[signature_type.value] = []

            # Compare all species pairs
            for i in range(len(species_list)):
                for j in range(i + 1, len(species_list)):
                    species1, species2 = species_list[i], species_list[j]

                    try:
                        result = self.compare_signatures(species1, species2, signature_type)
                        results[signature_type.value].append(result)
                    except ValidationError as e:
                        logger.warning(
                            f"Validation failed for {species1.value}-{species2.value}: {e}"
                        )
                        continue

        logger.info(
            f"Comprehensive validation completed for {len(signature_types)} signature types"
        )
        return results

    def generate_validation_report(self, results: Dict[str, List[ValidationResult]]) -> str:
        """Generate comprehensive validation report.

        Args:
            results: Validation results from comprehensive validation

        Returns:
            Formatted report string
        """
        report = []
        report.append("Cross-Species Validation Report")
        report.append("=" * 50)
        report.append("")

        for signature_type, validation_results in results.items():
            report.append(f"## {signature_type.upper()} SIGNATURES")
            report.append("-" * 30)

            if not validation_results:
                report.append("No validation results available.")
                report.append("")
                continue

            # Summary statistics
            scores = [r.validation_score for r in validation_results]
            significant_results = [r for r in validation_results if r.is_significant]

            report.append(f"Total comparisons: {len(validation_results)}")
            report.append(f"Significant results: {len(significant_results)}")
            report.append(f"Mean validation score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            report.append("")

            # Detailed results
            for result in validation_results:
                species1, species2 = result.species_comparison
                report.append(f"  {species1.value} vs {species2.value}:")
                report.append(f"    Score: {result.validation_score:.3f}")
                report.append(
                    f"    Correlation: {result.correlation_coefficient:.3f} (p={result.p_value:.3f})"
                )
                report.append(f"    Effect size: {result.effect_size:.3f}")
                report.append(f"    Significant: {'Yes' if result.is_significant else 'No'}")
                report.append("")

        return "\n".join(report)


# Factory function for easy instantiation
def create_cross_species_validator(
    significance_threshold: float = 0.05,
) -> CrossSpeciesValidator:
    """Create a cross-species validator with default settings.

    Args:
        significance_threshold: Statistical significance threshold

    Returns:
        Configured CrossSpeciesValidator instance
    """
    return CrossSpeciesValidator(significance_threshold=significance_threshold)
