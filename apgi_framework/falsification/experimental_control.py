"""Experimental Control Validation System.

This module implements validation for experimental controls required in APGI Framework
falsification testing. Ensures proper experimental conditions and rules out confounding
factors that could invalidate falsification test results.

Key components:
- Motor/verbal response system validation
- Metacognitive sensitivity assessment
- Task comprehension verification
- Stimulus validation
- Experimental integrity checking
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..exceptions import ValidationError


class ControlType(Enum):
    """Types of experimental controls."""

    MOTOR_SYSTEM = "motor_system"
    VERBAL_SYSTEM = "verbal_system"
    METACOGNITIVE = "metacognitive"
    TASK_COMPREHENSION = "task_comprehension"
    STIMULUS_VALIDATION = "stimulus_validation"


class ResponseModality(Enum):
    """Response modalities for control validation."""

    BUTTON_PRESS = "button_press"
    VERBAL_REPORT = "verbal_report"
    EYE_MOVEMENT = "eye_movement"
    GESTURE = "gesture"


@dataclass
class MotorResponse:
    """Data model for motor response validation.

    Represents motor system integrity measurements for experimental control.
    """

    response_type: ResponseModality
    reaction_time: float  # Reaction time in milliseconds
    accuracy: float  # Response accuracy (0.0-1.0)
    consistency: float  # Response consistency across trials (0.0-1.0)
    force_variability: Optional[float] = None  # Force variability for button presses
    movement_precision: Optional[float] = None  # Movement precision for gestures
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate motor response parameters."""
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValidationError(f"Accuracy must be between 0.0 and 1.0, got {self.accuracy}")

        if not 0.0 <= self.consistency <= 1.0:
            raise ValidationError(
                f"Consistency must be between 0.0 and 1.0, got {self.consistency}"
            )

        if self.reaction_time < 0:
            raise ValidationError(f"Reaction time must be positive, got {self.reaction_time}")

    def is_intact_motor_function(
        self,
        rt_threshold: float = 1500.0,
        accuracy_threshold: float = 0.8,
        consistency_threshold: float = 0.7,
    ) -> bool:
        """Check if motor function is intact.

        Args:
            rt_threshold: Maximum acceptable reaction time (ms)
            accuracy_threshold: Minimum acceptable accuracy
            consistency_threshold: Minimum acceptable consistency

        Returns:
            True if motor function meets all criteria
        """
        return (
            self.reaction_time <= rt_threshold
            and self.accuracy >= accuracy_threshold
            and self.consistency >= consistency_threshold
        )


@dataclass
class VerbalResponse:
    """Data model for verbal response validation.

    Represents verbal system integrity measurements for experimental control.
    """

    response_clarity: float  # Clarity of verbal response (0.0-1.0)
    response_latency: float  # Latency to verbal response (ms)
    vocabulary_complexity: float  # Complexity of vocabulary used (0.0-1.0)
    articulation_quality: float  # Quality of articulation (0.0-1.0)
    comprehension_accuracy: float  # Accuracy of comprehension (0.0-1.0)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate verbal response parameters."""
        for attr_name in [
            "response_clarity",
            "vocabulary_complexity",
            "articulation_quality",
            "comprehension_accuracy",
        ]:
            value = getattr(self, attr_name)
            if not 0.0 <= value <= 1.0:
                raise ValidationError(f"{attr_name} must be between 0.0 and 1.0, got {value}")

        if self.response_latency < 0:
            raise ValidationError(f"Response latency must be positive, got {self.response_latency}")

    def is_intact_verbal_function(
        self,
        clarity_threshold: float = 0.8,
        latency_threshold: float = 3000.0,
        comprehension_threshold: float = 0.85,
    ) -> bool:
        """Check if verbal function is intact.

        Args:
            clarity_threshold: Minimum acceptable response clarity
            latency_threshold: Maximum acceptable response latency (ms)
            comprehension_threshold: Minimum acceptable comprehension accuracy

        Returns:
            True if verbal function meets all criteria
        """
        return (
            self.response_clarity >= clarity_threshold
            and self.response_latency <= latency_threshold
            and self.comprehension_accuracy >= comprehension_threshold
        )


@dataclass
class MetacognitiveAssessment:
    """Data model for metacognitive sensitivity assessment.

    Represents metacognitive function measurements for experimental control.
    """

    confidence_accuracy_correlation: float  # Correlation between confidence and accuracy
    type_2_sensitivity: float  # Type 2 sensitivity (d')
    metacognitive_efficiency: float  # Metacognitive efficiency measure
    confidence_calibration: float  # Confidence calibration accuracy
    wagering_consistency: float  # Consistency in wagering behavior
    introspective_accuracy: float  # Accuracy of introspective reports
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate metacognitive assessment parameters."""
        if not -1.0 <= self.confidence_accuracy_correlation <= 1.0:
            raise ValidationError("Confidence-accuracy correlation must be between -1.0 and 1.0")

        for attr_name in [
            "metacognitive_efficiency",
            "confidence_calibration",
            "wagering_consistency",
            "introspective_accuracy",
        ]:
            value = getattr(self, attr_name)
            if not 0.0 <= value <= 1.0:
                raise ValidationError(f"{attr_name} must be between 0.0 and 1.0, got {value}")

    def is_intact_metacognitive_function(
        self,
        correlation_threshold: float = 0.3,
        sensitivity_threshold: float = 0.5,
        efficiency_threshold: float = 0.6,
    ) -> bool:
        """Check if metacognitive function is intact.

        Args:
            correlation_threshold: Minimum confidence-accuracy correlation
            sensitivity_threshold: Minimum type 2 sensitivity
            efficiency_threshold: Minimum metacognitive efficiency

        Returns:
            True if metacognitive function meets all criteria
        """
        return (
            self.confidence_accuracy_correlation >= correlation_threshold
            and self.type_2_sensitivity >= sensitivity_threshold
            and self.metacognitive_efficiency >= efficiency_threshold
        )


@dataclass
class TaskComprehension:
    """Data model for task comprehension validation.

    Represents task understanding and training validation measurements.
    """

    training_accuracy: float  # Accuracy during training phase (0.0-1.0)
    instruction_comprehension: float  # Understanding of instructions (0.0-1.0)
    task_strategy_consistency: float  # Consistency of task strategy (0.0-1.0)
    performance_stability: float  # Stability of performance across blocks (0.0-1.0)
    attention_maintenance: float  # Ability to maintain attention (0.0-1.0)
    rule_following_accuracy: float  # Accuracy in following task rules (0.0-1.0)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate task comprehension parameters."""
        for attr_name in [
            "training_accuracy",
            "instruction_comprehension",
            "task_strategy_consistency",
            "performance_stability",
            "attention_maintenance",
            "rule_following_accuracy",
        ]:
            value = getattr(self, attr_name)
            if not 0.0 <= value <= 1.0:
                raise ValidationError(f"{attr_name} must be between 0.0 and 1.0, got {value}")

    def is_adequate_task_comprehension(
        self,
        training_threshold: float = 0.8,
        comprehension_threshold: float = 0.85,
        consistency_threshold: float = 0.7,
    ) -> bool:
        """Check if task comprehension is adequate.

        Args:
            training_threshold: Minimum training accuracy
            comprehension_threshold: Minimum instruction comprehension
            consistency_threshold: Minimum strategy consistency

        Returns:
            True if task comprehension meets all criteria
        """
        return (
            self.training_accuracy >= training_threshold
            and self.instruction_comprehension >= comprehension_threshold
            and self.task_strategy_consistency >= consistency_threshold
        )


@dataclass
class StimulusValidation:
    """Data model for stimulus validation.

    Represents stimulus presentation and detection validation measurements.
    """

    stimulus_visibility: float  # Stimulus visibility/detectability (0.0-1.0)
    presentation_duration: float  # Actual presentation duration (ms)
    contrast_level: float  # Stimulus contrast level (0.0-1.0)
    signal_to_noise_ratio: float  # Signal-to-noise ratio
    detection_accuracy: float  # Accuracy of stimulus detection (0.0-1.0)
    supraliminal_threshold: float  # Threshold for supraliminal presentation
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate stimulus parameters."""
        for attr_name in [
            "stimulus_visibility",
            "contrast_level",
            "detection_accuracy",
        ]:
            value = getattr(self, attr_name)
            if not 0.0 <= value <= 1.0:
                raise ValidationError(f"{attr_name} must be between 0.0 and 1.0, got {value}")

        if self.presentation_duration < 0:
            raise ValidationError("Presentation duration must be positive")

    def is_supraliminal_presentation(
        self, visibility_threshold: float = 0.8, detection_threshold: float = 0.9
    ) -> bool:
        """Check if stimulus presentation is supraliminal.

        Args:
            visibility_threshold: Minimum visibility for supraliminal
            detection_threshold: Minimum detection accuracy for supraliminal

        Returns:
            True if stimulus is clearly supraliminal
        """
        return (
            self.stimulus_visibility >= visibility_threshold
            and self.detection_accuracy >= detection_threshold
        )


@dataclass
class ExperimentalControlResult:
    """Complete result of experimental control validation.

    Contains all control validation measures for falsification testing.
    """

    trial_id: str
    participant_id: str
    timestamp: datetime

    # Control measurements
    motor_responses: List[MotorResponse]
    verbal_responses: List[VerbalResponse]
    metacognitive_assessment: Optional[MetacognitiveAssessment]
    task_comprehension: Optional[TaskComprehension]
    stimulus_validation: Optional[StimulusValidation]

    # Validation results
    motor_system_intact: bool
    verbal_system_intact: bool
    metacognitive_function_intact: bool
    task_comprehension_adequate: bool
    stimulus_presentation_valid: bool

    # Overall validation
    experimental_controls_valid: bool
    confidence_level: float

    # Summary metrics
    overall_response_quality: float
    control_reliability_score: float

    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperimentalControlValidator:
    """Validator for experimental controls in falsification testing.

    Implements validation logic for motor/verbal systems, metacognitive function,
    task comprehension, and stimulus presentation according to experimental requirements.
    """

    def __init__(self) -> None:
        """Initialize the experimental control validator with default thresholds."""
        self.validation_thresholds = {
            # Motor system thresholds
            "motor_rt_max": 1500.0,  # Maximum reaction time (ms)
            "motor_accuracy_min": 0.8,  # Minimum accuracy
            "motor_consistency_min": 0.7,  # Minimum consistency
            # Verbal system thresholds
            "verbal_clarity_min": 0.8,  # Minimum response clarity
            "verbal_latency_max": 3000.0,  # Maximum response latency (ms)
            "verbal_comprehension_min": 0.85,  # Minimum comprehension accuracy
            # Metacognitive thresholds
            "metacog_correlation_min": 0.3,  # Minimum confidence-accuracy correlation
            "metacog_sensitivity_min": 0.5,  # Minimum type 2 sensitivity
            "metacog_efficiency_min": 0.6,  # Minimum metacognitive efficiency
            # Task comprehension thresholds
            "task_training_min": 0.8,  # Minimum training accuracy
            "task_comprehension_min": 0.85,  # Minimum instruction comprehension
            "task_consistency_min": 0.7,  # Minimum strategy consistency
            # Stimulus validation thresholds
            "stimulus_visibility_min": 0.8,  # Minimum visibility for supraliminal
            "stimulus_detection_min": 0.9,  # Minimum detection accuracy
        }

    def validate_motor_system(
        self, motor_responses: List[MotorResponse]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate motor system integrity.

        Args:
            motor_responses: List of motor response measurements

        Returns:
            Tuple of (is_intact, validation_details)
        """
        if not motor_responses:
            return False, {"reason": "no_motor_responses", "responses": []}

        validation_details: Dict[str, Any] = {
            "responses": [],
            "intact_responses": 0,
            "mean_rt": 0.0,
            "mean_accuracy": 0.0,
            "mean_consistency": 0.0,
        }

        intact_count = 0
        rt_values = []
        accuracy_values = []
        consistency_values = []

        for response in motor_responses:
            is_intact = response.is_intact_motor_function(
                self.validation_thresholds["motor_rt_max"],
                self.validation_thresholds["motor_accuracy_min"],
                self.validation_thresholds["motor_consistency_min"],
            )

            validation_details["responses"].append(
                {
                    "type": response.response_type.value,
                    "reaction_time": response.reaction_time,
                    "accuracy": response.accuracy,
                    "consistency": response.consistency,
                    "is_intact": is_intact,
                }
            )

            if is_intact:
                intact_count += 1

            rt_values.append(response.reaction_time)
            accuracy_values.append(response.accuracy)
            consistency_values.append(response.consistency)

        validation_details["intact_responses"] = intact_count
        validation_details["mean_rt"] = float(np.mean(rt_values))
        validation_details["mean_accuracy"] = float(np.mean(accuracy_values))
        validation_details["mean_consistency"] = float(np.mean(consistency_values))

        # Motor system is intact if majority of responses are intact
        is_intact = intact_count >= len(motor_responses) * 0.8

        return is_intact, validation_details

    def validate_verbal_system(
        self, verbal_responses: List[VerbalResponse]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate verbal system integrity.

        Args:
            verbal_responses: List of verbal response measurements

        Returns:
            Tuple of (is_intact, validation_details)
        """
        if not verbal_responses:
            return False, {"reason": "no_verbal_responses", "responses": []}

        validation_details: Dict[str, Any] = {
            "responses": [],
            "intact_responses": 0,
            "mean_clarity": 0.0,
            "mean_latency": 0.0,
            "mean_comprehension": 0.0,
        }

        intact_count = 0
        clarity_values = []
        latency_values = []
        comprehension_values = []

        for response in verbal_responses:
            is_intact = response.is_intact_verbal_function(
                self.validation_thresholds["verbal_clarity_min"],
                self.validation_thresholds["verbal_latency_max"],
                self.validation_thresholds["verbal_comprehension_min"],
            )

            validation_details["responses"].append(
                {
                    "clarity": response.response_clarity,
                    "latency": response.response_latency,
                    "comprehension": response.comprehension_accuracy,
                    "is_intact": is_intact,
                }
            )

            if is_intact:
                intact_count += 1

            clarity_values.append(response.response_clarity)
            latency_values.append(response.response_latency)
            comprehension_values.append(response.comprehension_accuracy)

        validation_details["intact_responses"] = intact_count
        validation_details["mean_clarity"] = float(np.mean(clarity_values))
        validation_details["mean_latency"] = float(np.mean(latency_values))
        validation_details["mean_comprehension"] = float(np.mean(comprehension_values))

        # Verbal system is intact if majority of responses are intact
        is_intact = intact_count >= len(verbal_responses) * 0.8

        return is_intact, validation_details

    def validate_metacognitive_function(
        self, metacognitive_assessment: Optional[MetacognitiveAssessment]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate metacognitive function integrity.

        Args:
            metacognitive_assessment: Metacognitive assessment measurements

        Returns:
            Tuple of (is_intact, validation_details)
        """
        if metacognitive_assessment is None:
            return False, {"reason": "no_metacognitive_assessment"}

        is_intact = metacognitive_assessment.is_intact_metacognitive_function(
            self.validation_thresholds["metacog_correlation_min"],
            self.validation_thresholds["metacog_sensitivity_min"],
            self.validation_thresholds["metacog_efficiency_min"],
        )

        validation_details = {
            "confidence_accuracy_correlation": metacognitive_assessment.confidence_accuracy_correlation,
            "type_2_sensitivity": metacognitive_assessment.type_2_sensitivity,
            "metacognitive_efficiency": metacognitive_assessment.metacognitive_efficiency,
            "confidence_calibration": metacognitive_assessment.confidence_calibration,
            "is_intact": is_intact,
        }

        return is_intact, validation_details

    def validate_task_comprehension(
        self, task_comprehension: Optional[TaskComprehension]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate task comprehension adequacy.

        Args:
            task_comprehension: Task comprehension measurements

        Returns:
            Tuple of (is_adequate, validation_details)
        """
        if task_comprehension is None:
            return False, {"reason": "no_task_comprehension_assessment"}

        is_adequate = task_comprehension.is_adequate_task_comprehension(
            self.validation_thresholds["task_training_min"],
            self.validation_thresholds["task_comprehension_min"],
            self.validation_thresholds["task_consistency_min"],
        )

        validation_details = {
            "training_accuracy": task_comprehension.training_accuracy,
            "instruction_comprehension": task_comprehension.instruction_comprehension,
            "task_strategy_consistency": task_comprehension.task_strategy_consistency,
            "performance_stability": task_comprehension.performance_stability,
            "is_adequate": is_adequate,
        }

        return is_adequate, validation_details

    def validate_stimulus_presentation(
        self, stimulus_validation: Optional[StimulusValidation]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate stimulus presentation validity.

        Args:
            stimulus_validation: Stimulus validation measurements

        Returns:
            Tuple of (is_valid, validation_details)
        """
        if stimulus_validation is None:
            return False, {"reason": "no_stimulus_validation"}

        is_valid = stimulus_validation.is_supraliminal_presentation(
            self.validation_thresholds["stimulus_visibility_min"],
            self.validation_thresholds["stimulus_detection_min"],
        )

        validation_details = {
            "stimulus_visibility": stimulus_validation.stimulus_visibility,
            "detection_accuracy": stimulus_validation.detection_accuracy,
            "contrast_level": stimulus_validation.contrast_level,
            "signal_to_noise_ratio": stimulus_validation.signal_to_noise_ratio,
            "is_valid": is_valid,
        }

        return is_valid, validation_details

    def validate_complete_experimental_controls(
        self,
        motor_responses: List[MotorResponse],
        verbal_responses: List[VerbalResponse],
        metacognitive_assessment: Optional[MetacognitiveAssessment],
        task_comprehension: Optional[TaskComprehension],
        stimulus_validation: Optional[StimulusValidation],
        trial_id: str,
        participant_id: str,
    ) -> ExperimentalControlResult:
        """Perform complete experimental control validation.

        Args:
            motor_responses: Motor response measurements
            verbal_responses: Verbal response measurements
            metacognitive_assessment: Metacognitive assessment
            task_comprehension: Task comprehension assessment
            stimulus_validation: Stimulus validation measurements
            trial_id: Trial identifier
            participant_id: Participant identifier

        Returns:
            Complete experimental control validation result
        """
        timestamp = datetime.now()

        # Validate each component
        motor_intact, motor_details = self.validate_motor_system(motor_responses)
        verbal_intact, verbal_details = self.validate_verbal_system(verbal_responses)
        metacog_intact, metacog_details = self.validate_metacognitive_function(
            metacognitive_assessment
        )
        task_adequate, task_details = self.validate_task_comprehension(task_comprehension)
        stimulus_valid, stimulus_details = self.validate_stimulus_presentation(stimulus_validation)

        # Overall experimental controls are valid if all components pass
        controls_valid = all(
            [motor_intact, verbal_intact, metacog_intact, task_adequate, stimulus_valid]
        )

        # Calculate confidence level and quality scores
        confidence_level = self._calculate_confidence_level(
            motor_details,
            verbal_details,
            metacog_details,
            task_details,
            stimulus_details,
        )

        response_quality = self._calculate_response_quality(motor_details, verbal_details)
        reliability_score = self._calculate_reliability_score(
            motor_intact, verbal_intact, metacog_intact, task_adequate, stimulus_valid
        )

        return ExperimentalControlResult(
            trial_id=trial_id,
            participant_id=participant_id,
            timestamp=timestamp,
            motor_responses=motor_responses,
            verbal_responses=verbal_responses,
            metacognitive_assessment=metacognitive_assessment,
            task_comprehension=task_comprehension,
            stimulus_validation=stimulus_validation,
            motor_system_intact=motor_intact,
            verbal_system_intact=verbal_intact,
            metacognitive_function_intact=metacog_intact,
            task_comprehension_adequate=task_adequate,
            stimulus_presentation_valid=stimulus_valid,
            experimental_controls_valid=controls_valid,
            confidence_level=confidence_level,
            overall_response_quality=response_quality,
            control_reliability_score=reliability_score,
            metadata={
                "motor_details": motor_details,
                "verbal_details": verbal_details,
                "metacognitive_details": metacog_details,
                "task_details": task_details,
                "stimulus_details": stimulus_details,
            },
        )

    def _calculate_confidence_level(
        self,
        motor_details: Dict,
        verbal_details: Dict,
        metacog_details: Dict,
        task_details: Dict,
        stimulus_details: Dict,
    ) -> float:
        """Calculate confidence level for experimental control validation."""
        confidence = 0.5  # Base confidence

        # Boost confidence based on quality of measurements
        if motor_details.get("mean_accuracy", 0) > 0.9:
            confidence += 0.1

        if verbal_details.get("mean_comprehension", 0) > 0.9:
            confidence += 0.1

        if metacog_details.get("confidence_accuracy_correlation", 0) > 0.5:
            confidence += 0.1

        if task_details.get("training_accuracy", 0) > 0.9:
            confidence += 0.1

        if stimulus_details.get("detection_accuracy", 0) > 0.95:
            confidence += 0.1

        return min(confidence, 1.0)

    def _calculate_response_quality(self, motor_details: Dict, verbal_details: Dict) -> float:
        """Calculate overall response quality score."""
        motor_quality = float(motor_details.get("mean_accuracy", 0)) * float(
            motor_details.get("mean_consistency", 0)
        )
        verbal_quality = float(verbal_details.get("mean_clarity", 0)) * float(
            verbal_details.get("mean_comprehension", 0)
        )

        return float((motor_quality + verbal_quality) / 2.0)

    def _calculate_reliability_score(
        self,
        motor_intact: bool,
        verbal_intact: bool,
        metacog_intact: bool,
        task_adequate: bool,
        stimulus_valid: bool,
    ) -> float:
        """Calculate control reliability score."""
        valid_controls = sum(
            [motor_intact, verbal_intact, metacog_intact, task_adequate, stimulus_valid]
        )
        return valid_controls / 5.0


class ExperimentalControlSimulator:
    """Simulator for experimental control measures.

    Generates realistic experimental control data for falsification testing scenarios.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the experimental control simulator.

        Args:
            random_seed: Random seed for reproducible simulations
        """
        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate_motor_responses(
        self, system_integrity: str = "intact", n_responses: int = 10
    ) -> List[MotorResponse]:
        """Simulate motor response measurements.

        Args:
            system_integrity: 'intact', 'impaired', or 'threshold'
            n_responses: Number of responses to simulate

        Returns:
            List of simulated motor responses
        """
        responses = []

        for i in range(n_responses):
            response_type = np.random.choice(list(ResponseModality))  # type: ignore

            if system_integrity == "intact":
                rt = np.random.uniform(300, 800)
                accuracy = np.random.uniform(0.85, 0.98)
                consistency = np.random.uniform(0.8, 0.95)
            elif system_integrity == "impaired":
                rt = np.random.uniform(800, 2000)
                accuracy = np.random.uniform(0.4, 0.7)
                consistency = np.random.uniform(0.3, 0.6)
            else:  # threshold
                rt = np.random.uniform(600, 1200)
                accuracy = np.random.uniform(0.75, 0.85)
                consistency = np.random.uniform(0.65, 0.75)

            response = MotorResponse(
                response_type=response_type,
                reaction_time=rt,
                accuracy=accuracy,
                consistency=consistency,
                force_variability=np.random.uniform(0.1, 0.3),
                movement_precision=np.random.uniform(0.7, 0.95),
            )

            responses.append(response)

        return responses

    def simulate_verbal_responses(
        self, system_integrity: str = "intact", n_responses: int = 5
    ) -> List[VerbalResponse]:
        """Simulate verbal response measurements.

        Args:
            system_integrity: 'intact', 'impaired', or 'threshold'
            n_responses: Number of responses to simulate

        Returns:
            List of simulated verbal responses
        """
        responses = []

        for i in range(n_responses):
            if system_integrity == "intact":
                clarity = np.random.uniform(0.85, 0.98)
                latency = np.random.uniform(800, 2000)
                comprehension = np.random.uniform(0.9, 0.98)
                articulation = np.random.uniform(0.85, 0.95)
            elif system_integrity == "impaired":
                clarity = np.random.uniform(0.4, 0.7)
                latency = np.random.uniform(2000, 5000)
                comprehension = np.random.uniform(0.5, 0.75)
                articulation = np.random.uniform(0.4, 0.7)
            else:  # threshold
                clarity = np.random.uniform(0.75, 0.85)
                latency = np.random.uniform(1500, 3000)
                comprehension = np.random.uniform(0.8, 0.9)
                articulation = np.random.uniform(0.75, 0.85)

            response = VerbalResponse(
                response_clarity=clarity,
                response_latency=latency,
                vocabulary_complexity=np.random.uniform(0.6, 0.9),
                articulation_quality=articulation,
                comprehension_accuracy=comprehension,
            )

            responses.append(response)

        return responses

    def simulate_metacognitive_assessment(
        self, function_level: str = "intact"
    ) -> MetacognitiveAssessment:
        """Simulate metacognitive assessment.

        Args:
            function_level: 'intact', 'impaired', or 'threshold'

        Returns:
            Simulated metacognitive assessment
        """
        if function_level == "intact":
            correlation = np.random.uniform(0.4, 0.8)
            sensitivity = np.random.uniform(0.6, 1.2)
            efficiency = np.random.uniform(0.7, 0.9)
            calibration = np.random.uniform(0.8, 0.95)
        elif function_level == "impaired":
            correlation = np.random.uniform(-0.2, 0.2)
            sensitivity = np.random.uniform(0.1, 0.4)
            efficiency = np.random.uniform(0.2, 0.5)
            calibration = np.random.uniform(0.3, 0.6)
        else:  # threshold
            correlation = np.random.uniform(0.25, 0.4)
            sensitivity = np.random.uniform(0.4, 0.6)
            efficiency = np.random.uniform(0.55, 0.7)
            calibration = np.random.uniform(0.7, 0.8)

        return MetacognitiveAssessment(
            confidence_accuracy_correlation=correlation,
            type_2_sensitivity=sensitivity,
            metacognitive_efficiency=efficiency,
            confidence_calibration=calibration,
            wagering_consistency=np.random.uniform(0.6, 0.9),
            introspective_accuracy=np.random.uniform(0.7, 0.9),
        )

    def simulate_task_comprehension(
        self, comprehension_level: str = "adequate"
    ) -> TaskComprehension:
        """Simulate task comprehension assessment.

        Args:
            comprehension_level: 'adequate', 'inadequate', or 'threshold'

        Returns:
            Simulated task comprehension assessment
        """
        if comprehension_level == "adequate":
            training = np.random.uniform(0.85, 0.98)
            comprehension = np.random.uniform(0.9, 0.98)
            consistency = np.random.uniform(0.8, 0.95)
            stability = np.random.uniform(0.85, 0.95)
        elif comprehension_level == "inadequate":
            training = np.random.uniform(0.4, 0.7)
            comprehension = np.random.uniform(0.5, 0.75)
            consistency = np.random.uniform(0.3, 0.6)
            stability = np.random.uniform(0.4, 0.7)
        else:  # threshold
            training = np.random.uniform(0.75, 0.85)
            comprehension = np.random.uniform(0.8, 0.9)
            consistency = np.random.uniform(0.65, 0.75)
            stability = np.random.uniform(0.7, 0.8)

        return TaskComprehension(
            training_accuracy=training,
            instruction_comprehension=comprehension,
            task_strategy_consistency=consistency,
            performance_stability=stability,
            attention_maintenance=np.random.uniform(0.7, 0.9),
            rule_following_accuracy=np.random.uniform(0.8, 0.95),
        )

    def simulate_stimulus_validation(
        self, presentation_type: str = "supraliminal"
    ) -> StimulusValidation:
        """Simulate stimulus validation measurements.

        Args:
            presentation_type: 'supraliminal', 'subliminal', or 'threshold'

        Returns:
            Simulated stimulus validation
        """
        if presentation_type == "supraliminal":
            visibility = np.random.uniform(0.85, 0.98)
            detection = np.random.uniform(0.92, 0.99)
            contrast = np.random.uniform(0.8, 0.95)
            snr = np.random.uniform(15, 30)
        elif presentation_type == "subliminal":
            visibility = np.random.uniform(0.1, 0.4)
            detection = np.random.uniform(0.45, 0.55)
            contrast = np.random.uniform(0.1, 0.3)
            snr = np.random.uniform(2, 8)
        else:  # threshold
            visibility = np.random.uniform(0.7, 0.85)
            detection = np.random.uniform(0.85, 0.92)
            contrast = np.random.uniform(0.6, 0.8)
            snr = np.random.uniform(10, 18)

        return StimulusValidation(
            stimulus_visibility=visibility,
            presentation_duration=np.random.uniform(100, 500),
            contrast_level=contrast,
            signal_to_noise_ratio=snr,
            detection_accuracy=detection,
            supraliminal_threshold=0.8,
        )
