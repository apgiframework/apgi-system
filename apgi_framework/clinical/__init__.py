"""Clinical assessment and classification tools for APGI Framework.

This module provides tools for:
- Disorder classification (GAD, panic disorder, social anxiety)
- Treatment response prediction
- Clinical parameter extraction
- Longitudinal tracking
"""

from typing import Any, Dict, List, Optional

from .disorder_classification import (
    ClassificationResult,
    DisorderClassification,
    DisorderType,
    NeuralSignatureProfile,
)
from .parameter_extraction import (
    AssessmentBattery,
    ClinicalParameterExtractor,
    ClinicalParameters,
    ReliabilityMetrics,
)
from .treatment_prediction import (
    BaselineParameters,
    TreatmentPrediction,
    TreatmentPredictor,
    TreatmentType,
)


# Mock classes for testing
class PatientDataManager:
    """Mock patient data manager for testing purposes."""

    def __init__(self) -> None:
        self.patients: Dict[str, Any] = {}

    def add_patient(self, patient_id: str, patient_data: Dict[str, Any]) -> None:
        """Add a patient record."""
        self.patients[patient_id] = patient_data

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient data by ID."""
        return self.patients.get(patient_id)

    def update_patient(self, patient_id: str, updates: Dict[str, Any]) -> None:
        """Update patient data."""
        if patient_id in self.patients:
            self.patients[patient_id].update(updates)

    def list_patients(self) -> List[str]:
        """List all patient IDs."""
        return list(self.patients.keys())


class InterventionTracker:
    """Mock intervention tracker for testing purposes."""

    def __init__(self) -> None:
        self.interventions: Dict[str, Dict[str, Any]] = {}
        self.patient_interventions: Dict[str, List[str]] = {}

    def add_intervention(self, intervention_data: Dict[str, Any]) -> Dict[str, Any]:
        intervention_id = f"intervention_{hash(str(intervention_data)) % 10000:04d}"
        intervention = {
            "intervention_id": intervention_id,
            "data": intervention_data,
            "start_date": "2024-01-01",
            "status": "active",
        }
        self.interventions[intervention_id] = intervention

        # Link to patient if patient_id is provided
        patient_id = intervention_data.get("patient_id")
        if patient_id:
            if patient_id not in self.patient_interventions:
                self.patient_interventions[patient_id] = []
            self.patient_interventions[patient_id].append(intervention_id)

        return intervention

    def get_patient_interventions(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get all interventions for a patient."""
        intervention_ids = self.patient_interventions.get(patient_id, [])
        return [self.interventions[iid] for iid in intervention_ids if iid in self.interventions]


class OutcomeAnalyzer:
    """Mock outcome analyzer for testing purposes."""

    def __init__(self) -> None:
        self.outcomes: Dict[str, Dict[str, Any]] = {}

    def analyze_outcomes(self, outcome_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze treatment outcomes."""
        analysis_id = f"analysis_{hash(str(outcome_data)) % 10000:04d}"

        # Mock analysis logic
        baseline_score = outcome_data.get("baseline_score", 50)
        post_treatment_score = outcome_data.get("post_treatment_score", 70)
        improvement = post_treatment_score - baseline_score

        # Determine significance based on improvement
        significant_improvement = improvement > 15
        effect_size = "small" if improvement <= 10 else "medium" if improvement <= 20 else "large"
        clinical_significance = 0.05 if significant_improvement else 0.15

        result = {
            "analysis_id": analysis_id,
            "significant_improvement": significant_improvement,
            "effect_size": effect_size,
            "clinical_significance": clinical_significance,
            "improvement_score": improvement,
            "baseline_score": baseline_score,
            "post_treatment_score": post_treatment_score,
        }

        self.outcomes[analysis_id] = result
        return result

    def track_progress(self, intervention_data: Dict[str, Any]) -> Dict[str, float]:
        """Track intervention progress."""
        sessions_completed = intervention_data.get("sessions_completed", 0)
        total_sessions = intervention_data.get("total_sessions", 36)
        adherence_rate = sessions_completed / total_sessions if total_sessions > 0 else 0

        return {
            "sessions_completed": sessions_completed,
            "total_sessions": total_sessions,
            "adherence_rate": round(adherence_rate, 2),
            "progress_percentage": round(adherence_rate * 100, 1),
        }


__all__ = [
    "DisorderClassification",
    "DisorderType",
    "ClassificationResult",
    "NeuralSignatureProfile",
    "TreatmentPredictor",
    "TreatmentType",
    "TreatmentPrediction",
    "BaselineParameters",
    "ClinicalParameterExtractor",
    "ClinicalParameters",
    "AssessmentBattery",
    "ReliabilityMetrics",
    "PatientDataManager",
    "InterventionTracker",
    "OutcomeAnalyzer",
]
