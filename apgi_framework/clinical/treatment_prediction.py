"""Treatment prediction module for APGI Framework.

Implements treatment response prediction based on baseline APGI parameters.
"""

from dataclasses import dataclass
from enum import Enum


class TreatmentType(Enum):
    """Types of treatments."""

    SSRI = "ssri"
    SNRI = "snri"
    BETA_BLOCKER = "beta_blocker"
    CBT = "cbt"
    EXPOSURE_THERAPY = "exposure_therapy"


@dataclass
class BaselineParameters:
    """Baseline APGI parameters for treatment prediction."""

    theta_t: float = 3.5
    pi_e: float = 2.0
    pi_i: float = 1.5
    beta: float = 1.2


@dataclass
class TreatmentPrediction:
    """Treatment response prediction."""

    recommended_treatment: TreatmentType
    predicted_response: float  # 0-1 scale
    confidence: float  # 0-1 scale


class TreatmentPredictor:
    """Treatment response prediction system."""

    def __init__(self) -> None:
        """Initialize treatment predictor."""

    def predict(self, params: BaselineParameters) -> TreatmentPrediction:
        """Predict treatment response based on baseline APGI parameters.

        Uses a multi-factor analysis of theta_t, pi_e, pi_i, and beta parameters
        to recommend appropriate treatments and predict response probabilities.

        Args:
            params: Baseline APGI parameters

        Returns:
            TreatmentPrediction with recommended treatment, predicted response, and confidence
        """
        # Calculate neural complexity score
        complexity_score = params.theta_t * params.pi_e / (params.pi_i + 0.1)

        # Determine treatment recommendation based on parameter profile
        if complexity_score < 2.0:
            # Low complexity - may benefit from SSRIs
            recommended = TreatmentType.SSRI
            predicted_response = 0.65 + (params.beta * 0.1)
            confidence = 0.75
        elif complexity_score < 4.0:
            # Moderate complexity - may benefit from SNRIs or CBT
            if params.pi_e > params.pi_i:
                recommended = TreatmentType.SNRI
                predicted_response = 0.70 + (params.beta * 0.08)
                confidence = 0.80
            else:
                recommended = TreatmentType.CBT
                predicted_response = 0.72 + (params.beta * 0.07)
                confidence = 0.78
        else:
            # High complexity - may need combination therapy
            recommended = TreatmentType.EXPOSURE_THERAPY
            predicted_response = 0.60 + (params.beta * 0.12)
            confidence = 0.70

        # Adjust based on beta (integration parameter)
        if params.beta > 1.5:
            # High integration may benefit from CBT
            recommended = TreatmentType.CBT
            predicted_response = min(predicted_response + 0.1, 0.95)

        # Ensure values are in valid ranges
        predicted_response = max(0.1, min(0.95, predicted_response))
        confidence = max(0.5, min(0.95, confidence))

        return TreatmentPrediction(
            recommended_treatment=recommended,
            predicted_response=predicted_response,
            confidence=confidence,
        )
