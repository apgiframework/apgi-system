"""Core data models for the APGI Framework.

Defines the fundamental data structures used throughout the system for
experimental trials, results, parameters, and statistical summaries.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class APGIParameters:
    """APGI Framework mathematical parameters."""

    extero_precision: float = 2.0
    intero_precision: float = 1.5
    extero_error: float = 1.0
    intero_error: float = 0.8
    somatic_gain: float = 1.2
    threshold: float = 3.5
    steepness: float = 2.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "extero_precision": self.extero_precision,
            "intero_precision": self.intero_precision,
            "extero_error": self.extero_error,
            "intero_error": self.intero_error,
            "somatic_gain": self.somatic_gain,
            "threshold": self.threshold,
            "steepness": self.steepness,
        }


@dataclass
class NeuralSignatures:
    """Neural signature measurements."""

    p3b_amplitude: float = 0.0  # μV
    p3b_latency: float = 350.0  # ms
    gamma_plv: float = 0.0  # Phase-locking value
    gamma_duration: float = 200.0  # ms
    bold_activations: Dict[str, float] = field(default_factory=dict)  # region -> z-score
    pci_value: float = 0.0  # Perturbational Complexity Index

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "p3b_amplitude": self.p3b_amplitude,
            "p3b_latency": self.p3b_latency,
            "gamma_plv": self.gamma_plv,
            "gamma_duration": self.gamma_duration,
            "bold_activations": self.bold_activations,
            "pci_value": self.pci_value,
        }


@dataclass
class ConsciousnessAssessment:
    """Consciousness assessment measurements."""

    subjective_report: bool = False
    forced_choice_accuracy: float = 0.5
    confidence_rating: float = 0.5
    wagering_behavior: float = 0.5
    metacognitive_sensitivity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subjective_report": self.subjective_report,
            "forced_choice_accuracy": self.forced_choice_accuracy,
            "confidence_rating": self.confidence_rating,
            "wagering_behavior": self.wagering_behavior,
            "metacognitive_sensitivity": self.metacognitive_sensitivity,
        }


@dataclass
class ExperimentalTrial:
    """Single experimental trial data."""

    trial_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    participant_id: str = ""
    condition: str = ""
    trial_number: int = 0

    # Core data
    apgi_parameters: APGIParameters = field(default_factory=APGIParameters)
    neural_signatures: NeuralSignatures = field(default_factory=NeuralSignatures)
    consciousness_assessment: ConsciousnessAssessment = field(
        default_factory=ConsciousnessAssessment
    )

    # Timing and metadata
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trial_id": self.trial_id,
            "participant_id": self.participant_id,
            "condition": self.condition,
            "trial_number": self.trial_number,
            "apgi_parameters": self.apgi_parameters.to_dict(),
            "neural_signatures": self.neural_signatures.to_dict(),
            "consciousness_assessment": self.consciousness_assessment.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class FalsificationResult:
    """Results from a falsification test."""

    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    test_type: str = ""  # primary, consciousness_without_ignition, etc.
    experiment_id: str = ""

    # Core results
    is_falsified: bool = False
    confidence_level: float = 0.95
    effect_size: float = 0.0
    p_value: float = 1.0
    statistical_power: float = 0.0

    # Sample information
    n_participants: int = 0
    n_trials: int = 0
    replication_count: int = 1

    # Detailed results
    detailed_results: Dict[str, Any] = field(default_factory=dict)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    researcher: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "test_type": self.test_type,
            "experiment_id": self.experiment_id,
            "is_falsified": self.is_falsified,
            "confidence_level": self.confidence_level,
            "effect_size": self.effect_size,
            "p_value": self.p_value,
            "statistical_power": self.statistical_power,
            "n_participants": self.n_participants,
            "n_trials": self.n_trials,
            "replication_count": self.replication_count,
            "detailed_results": self.detailed_results,
            "statistical_tests": self.statistical_tests,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "researcher": self.researcher,
        }


@dataclass
class StatisticalSummary:
    """Statistical summary of experimental results."""

    summary_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_ids: List[str] = field(default_factory=list)

    # Effect size statistics
    mean_effect_size: float = 0.0
    effect_size_ci_lower: float = 0.0
    effect_size_ci_upper: float = 0.0

    # Statistical power
    statistical_power: float = 0.0
    power_ci_lower: float = 0.0
    power_ci_upper: float = 0.0

    # Replication analysis
    replication_success_rate: float = 0.0
    replication_ci_lower: float = 0.0
    replication_ci_upper: float = 0.0

    # Sample size analysis
    total_participants: int = 0
    total_trials: int = 0
    adequate_power_studies: int = 0

    # Meta-analysis results
    meta_analysis_results: Dict[str, Any] = field(default_factory=dict)
    heterogeneity_stats: Dict[str, float] = field(default_factory=dict)

    # Quality metrics
    publication_bias_tests: Dict[str, Any] = field(default_factory=dict)
    quality_assessment: Dict[str, float] = field(default_factory=dict)

    # Additional statistical attributes for report generation
    confidence_level: float = 0.95
    correction_method: str = "bonferroni"
    cross_lab_consistency: float = 0.0
    total_sample_size: int = 0
    effective_sample_size: int = 0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    analyst: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary_id": self.summary_id,
            "experiment_ids": self.experiment_ids,
            "mean_effect_size": self.mean_effect_size,
            "effect_size_ci_lower": self.effect_size_ci_lower,
            "effect_size_ci_upper": self.effect_size_ci_upper,
            "statistical_power": self.statistical_power,
            "power_ci_lower": self.power_ci_lower,
            "power_ci_upper": self.power_ci_upper,
            "replication_success_rate": self.replication_success_rate,
            "replication_ci_lower": self.replication_ci_lower,
            "replication_ci_upper": self.replication_ci_upper,
            "total_participants": self.total_participants,
            "total_trials": self.total_trials,
            "adequate_power_studies": self.adequate_power_studies,
            "meta_analysis_results": self.meta_analysis_results,
            "heterogeneity_stats": self.heterogeneity_stats,
            "publication_bias_tests": self.publication_bias_tests,
            "quality_assessment": self.quality_assessment,
            "created_at": self.created_at.isoformat(),
            "analyst": self.analyst,
        }


@dataclass
class PharmacologicalCondition:
    """Pharmacological manipulation condition."""

    condition_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    drug_name: str = ""
    dosage: float = 0.0
    dosage_unit: str = "mg"
    administration_route: str = "oral"
    administration_time: datetime = field(default_factory=datetime.now)

    # Expected effects
    expected_threshold_change: float = 0.0
    expected_duration_hours: float = 0.0

    # Control measures
    control_measures: Dict[str, float] = field(default_factory=dict)
    side_effects: List[str] = field(default_factory=list)

    # Metadata
    researcher: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition_id": self.condition_id,
            "drug_name": self.drug_name,
            "dosage": self.dosage,
            "dosage_unit": self.dosage_unit,
            "administration_route": self.administration_route,
            "administration_time": self.administration_time.isoformat(),
            "expected_threshold_change": self.expected_threshold_change,
            "expected_duration_hours": self.expected_duration_hours,
            "control_measures": self.control_measures,
            "side_effects": self.side_effects,
            "researcher": self.researcher,
            "notes": self.notes,
        }
