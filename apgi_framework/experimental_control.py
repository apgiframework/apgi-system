"""Experimental Control and Validation System.

This module implements comprehensive experimental control mechanisms and validation
systems for the APGI Framework testing. It ensures proper experimental
conditions, validates participant responses, and maintains experimental integrity.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ResponseSystemType(Enum):
    """Types of response systems to validate."""

    MOTOR = "motor"
    VERBAL = "verbal"
    BUTTON_PRESS = "button_press"
    EYE_MOVEMENT = "eye_movement"


class TaskType(Enum):
    """Types of experimental tasks."""

    DETECTION = "detection"
    DISCRIMINATION = "discrimination"
    IDENTIFICATION = "identification"
    METACOGNITIVE = "metacognitive"


@dataclass
class ResponseSystemValidation:
    """Results of response system validation."""

    system_type: ResponseSystemType
    is_intact: bool
    response_time_mean: float
    response_time_std: float
    accuracy: float
    consistency_score: float
    validation_timestamp: datetime
    notes: str = ""


@dataclass
class TaskComprehensionResult:
    """Results of task comprehension verification."""

    task_type: TaskType
    comprehension_score: float
    training_trials_completed: int
    performance_criterion_met: bool
    response_consistency: float
    instruction_clarity_rating: float
    validation_timestamp: datetime


@dataclass
class ExperimentalControlReport:
    """Comprehensive experimental control validation report."""

    participant_id: str
    session_id: str
    response_systems: List[ResponseSystemValidation]
    task_comprehension: List[TaskComprehensionResult]
    overall_validity: bool
    control_score: float
    recommendations: List[str]
    timestamp: datetime


class ExperimentalControlManager:
    """Manages experimental control validation and orchestration.

    This class coordinates validation of motor/verbal response systems,
    task comprehension verification, and overall experimental integrity.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the experimental control manager.

        Args:
            config: Configuration parameters for validation thresholds
        """
        self.config = config or self._get_default_config()
        self.validation_history: Dict[str, List[ExperimentalControlReport]] = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for experimental control validation."""
        return {
            "response_time_thresholds": {
                "motor_min": 0.15,  # 150ms minimum
                "motor_max": 2.0,  # 2s maximum
                "verbal_min": 0.5,  # 500ms minimum
                "verbal_max": 5.0,  # 5s maximum
            },
            "accuracy_thresholds": {
                "minimum_accuracy": 0.7,  # 70% minimum
                "consistency_threshold": 0.8,  # 80% consistency
            },
            "comprehension_thresholds": {
                "minimum_score": 0.8,  # 80% comprehension
                "training_trials_required": 20,
                "performance_criterion": 0.75,  # 75% accuracy in training
            },
        }

    def validate_response_system(
        self,
        participant_id: str,
        system_type: ResponseSystemType,
        response_times: List[float],
        accuracies: List[bool],
    ) -> ResponseSystemValidation:
        """Validate motor/verbal response system integrity.

        Args:
            participant_id: Unique participant identifier
            system_type: Type of response system to validate
            response_times: List of response times in seconds
            accuracies: List of accuracy values (True/False)

        Returns:
            ResponseSystemValidation with validation results
        """
        logger.info(
            f"Validating {system_type.value} response system for participant {participant_id}"
        )

        # Calculate response time statistics
        rt_array = np.array(response_times)
        rt_mean = np.mean(rt_array)
        rt_std = np.std(rt_array)

        # Calculate accuracy
        accuracy = np.mean(accuracies)

        # Calculate consistency (inverse of coefficient of variation)
        consistency_score = min(1.0, max(0.0, 1.0 - (rt_std / rt_mean))) if rt_mean > 0 else 0.0

        # Determine if system is intact based on thresholds
        thresholds = self.config["response_time_thresholds"]
        accuracy_thresholds = self.config["accuracy_thresholds"]

        if system_type == ResponseSystemType.MOTOR:
            rt_valid = thresholds["motor_min"] <= rt_mean <= thresholds["motor_max"]
        elif system_type == ResponseSystemType.VERBAL:
            rt_valid = thresholds["verbal_min"] <= rt_mean <= thresholds["verbal_max"]
        else:
            rt_valid = thresholds["motor_min"] <= rt_mean <= thresholds["motor_max"]

        accuracy_valid = accuracy >= accuracy_thresholds["minimum_accuracy"]
        consistency_valid = consistency_score >= accuracy_thresholds["consistency_threshold"]

        is_intact = rt_valid and accuracy_valid and consistency_valid

        # Generate notes
        notes = []
        if not rt_valid:
            notes.append(f"Response time out of range: {rt_mean:.3f}s")
        if not accuracy_valid:
            notes.append(f"Accuracy below threshold: {accuracy:.3f}")
        if not consistency_valid:
            notes.append(f"Consistency below threshold: {consistency_score:.3f}")

        return ResponseSystemValidation(
            system_type=system_type,
            is_intact=is_intact,
            response_time_mean=rt_mean,
            response_time_std=rt_std,
            accuracy=float(accuracy),
            consistency_score=consistency_score,
            validation_timestamp=datetime.now(),
            notes="; ".join(notes),
        )

    def verify_task_comprehension(
        self,
        participant_id: str,
        task_type: TaskType,
        training_responses: List[bool],
        instruction_clarity_rating: float,
    ) -> TaskComprehensionResult:
        """Verify participant task comprehension through training performance.

        Args:
            participant_id: Unique participant identifier
            task_type: Type of experimental task
            training_responses: List of training trial accuracies
            instruction_clarity_rating: Participant rating of instruction clarity (0-1)

        Returns:
            TaskComprehensionResult with comprehension assessment
        """
        logger.info(
            f"Verifying task comprehension for {task_type.value} task, participant {participant_id}"
        )

        thresholds = self.config["comprehension_thresholds"]

        # Calculate comprehension metrics
        training_trials_completed = len(training_responses)
        performance_score = np.mean(training_responses) if training_responses else 0.0

        # Calculate response consistency in training
        if len(training_responses) >= 10:
            # Calculate consistency as stability of performance over time
            window_size = min(5, len(training_responses) // 4)
            windows = [
                training_responses[i : i + window_size]
                for i in range(0, len(training_responses) - window_size + 1, window_size)
            ]
            window_means = [np.mean(window) for window in windows if len(window) == window_size]
            response_consistency = 1.0 - np.std(window_means) if len(window_means) > 1 else 1.0
        else:
            response_consistency = 0.5  # Insufficient data

        # Determine if comprehension criteria are met
        sufficient_trials = training_trials_completed >= thresholds["training_trials_required"]
        performance_criterion_met = performance_score >= thresholds["performance_criterion"]
        comprehension_score = (
            performance_score + response_consistency + instruction_clarity_rating
        ) / 3.0

        overall_comprehension = (
            sufficient_trials
            and performance_criterion_met
            and comprehension_score >= thresholds["minimum_score"]
        )

        return TaskComprehensionResult(
            task_type=task_type,
            comprehension_score=float(comprehension_score),
            training_trials_completed=training_trials_completed,
            performance_criterion_met=overall_comprehension,
            response_consistency=response_consistency,
            instruction_clarity_rating=instruction_clarity_rating,
            validation_timestamp=datetime.now(),
        )

    def generate_control_report(
        self,
        participant_id: str,
        session_id: str,
        response_validations: List[ResponseSystemValidation],
        comprehension_results: List[TaskComprehensionResult],
    ) -> ExperimentalControlReport:
        """Generate comprehensive experimental control validation report.

        Args:
            participant_id: Unique participant identifier
            session_id: Unique session identifier
            response_validations: List of response system validation results
            comprehension_results: List of task comprehension results

        Returns:
            ExperimentalControlReport with overall validation assessment
        """
        logger.info(
            f"Generating control report for participant {participant_id}, session {session_id}"
        )

        # Calculate overall validity
        response_validity = all(rv.is_intact for rv in response_validations)
        comprehension_validity = all(cr.performance_criterion_met for cr in comprehension_results)
        overall_validity = response_validity and comprehension_validity

        # Calculate control score (0-1)
        response_scores = [rv.accuracy * rv.consistency_score for rv in response_validations]
        comprehension_scores = [cr.comprehension_score for cr in comprehension_results]

        all_scores = response_scores + comprehension_scores
        control_score = np.mean(all_scores) if all_scores else 0.0

        # Generate recommendations
        recommendations = []
        if not response_validity:
            invalid_systems = [
                rv.system_type.value for rv in response_validations if not rv.is_intact
            ]
            recommendations.append(f"Response system issues detected: {', '.join(invalid_systems)}")

        if not comprehension_validity:
            failed_tasks = [
                cr.task_type.value
                for cr in comprehension_results
                if not cr.performance_criterion_met
            ]
            recommendations.append(f"Task comprehension issues: {', '.join(failed_tasks)}")

        if control_score < 0.8:
            recommendations.append("Consider additional training or participant exclusion")

        if not recommendations:
            recommendations.append("All experimental controls validated successfully")

        report = ExperimentalControlReport(
            participant_id=participant_id,
            session_id=session_id,
            response_systems=response_validations,
            task_comprehension=comprehension_results,
            overall_validity=overall_validity,
            control_score=float(control_score),
            recommendations=recommendations,
            timestamp=datetime.now(),
        )

        # Store in history
        if participant_id not in self.validation_history:
            self.validation_history[participant_id] = []
        self.validation_history[participant_id].append(report)

        return report

    def get_validation_history(self, participant_id: str) -> List[ExperimentalControlReport]:
        """Get validation history for a participant."""
        return self.validation_history.get(participant_id, [])

    def is_participant_valid(self, participant_id: str) -> bool:
        """Check if participant has valid experimental controls."""
        history = self.get_validation_history(participant_id)
        if not history:
            return False

        latest_report = history[-1]
        return latest_report.overall_validity and latest_report.control_score >= 0.8


@dataclass
class ParticipantProfile:
    """Individual participant characteristics and parameters."""

    participant_id: str
    age: int
    cognitive_ability: float  # 0-1 scale
    attention_span: float  # 0-1 scale
    response_speed: float  # multiplier for base response times
    consistency: float  # 0-1 scale for response consistency
    fatigue_rate: float  # rate of performance degradation over time
    learning_rate: float  # rate of improvement in training
    individual_differences: Dict[str, float]  # additional individual factors


@dataclass
class SimulatedResponse:
    """A simulated participant response."""

    response_time: float
    accuracy: bool
    confidence: float
    response_value: Any
    trial_number: int
    fatigue_level: float
    timestamp: datetime


class ParticipantSimulator:
    """Simulates realistic participant behavior with individual differences.

    This class models participant responses including individual differences,
    response variability, consistency patterns, and performance changes over time.
    """

    def __init__(self, profile: ParticipantProfile, random_seed: Optional[int] = None):
        """Initialize participant simulator with individual profile.

        Args:
            profile: ParticipantProfile with individual characteristics
            random_seed: Random seed for reproducible simulations
        """
        self.profile = profile
        self.trial_count = 0
        self.fatigue_accumulation = 0.0
        self.learning_progress = 0.0

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        logger.info(f"Initialized participant simulator for {profile.participant_id}")

    def simulate_response(
        self,
        task_difficulty: float,
        base_response_time: float,
        correct_answer: Any,
        response_options: List[Any],
    ) -> SimulatedResponse:
        """Simulate a participant response to a task.

        Args:
            task_difficulty: Difficulty level (0-1)
            base_response_time: Base response time for the task
            correct_answer: The correct answer
            response_options: Available response options

        Returns:
            SimulatedResponse with simulated participant behavior
        """
        self.trial_count += 1

        # Update fatigue and learning
        self._update_fatigue()
        self._update_learning()

        # Calculate performance modifiers
        difficulty_modifier = 1.0 - (task_difficulty * 0.3)  # Harder tasks reduce performance
        fatigue_modifier = 1.0 - (self.fatigue_accumulation * 0.2)  # Fatigue reduces performance
        learning_modifier = 1.0 + (self.learning_progress * 0.1)  # Learning improves performance
        attention_modifier = self.profile.attention_span

        # Calculate overall performance
        performance = (
            self.profile.cognitive_ability
            * difficulty_modifier
            * fatigue_modifier
            * learning_modifier
            * attention_modifier
        )

        # Add individual differences
        for factor, value in self.profile.individual_differences.items():
            if factor == "impulsivity":
                performance *= 1.0 - value * 0.1  # Impulsivity reduces accuracy
            elif factor == "motivation":
                performance *= 1.0 + value * 0.1  # Motivation improves performance

        # Simulate accuracy with noise
        accuracy_probability = np.clip(performance, 0.1, 0.95)  # Keep within reasonable bounds
        accuracy_noise = np.random.normal(0, 0.05)  # Small random variation
        final_accuracy_prob = np.clip(accuracy_probability + accuracy_noise, 0.05, 0.95)

        is_correct = np.random.random() < final_accuracy_prob

        # Simulate response selection
        if is_correct:
            response_value = correct_answer
        else:
            # Select incorrect response
            incorrect_options = [opt for opt in response_options if opt != correct_answer]
            response_value = (
                np.random.choice(incorrect_options) if incorrect_options else correct_answer
            )

        # Simulate response time
        rt_base = base_response_time * self.profile.response_speed
        rt_difficulty = rt_base * (1.0 + task_difficulty * 0.5)  # Harder tasks take longer
        rt_fatigue = rt_difficulty * (
            1.0 + self.fatigue_accumulation * 0.3
        )  # Fatigue slows responses

        # Add individual consistency variation
        consistency_noise = np.random.normal(0, (1.0 - self.profile.consistency) * 0.2)
        response_time = max(0.1, rt_fatigue + consistency_noise)  # Minimum 100ms

        # Simulate confidence (related to accuracy and individual factors)
        base_confidence = final_accuracy_prob
        confidence_noise = np.random.normal(0, 0.1)
        confidence = np.clip(base_confidence + confidence_noise, 0.1, 0.9)

        return SimulatedResponse(
            response_time=response_time,
            accuracy=is_correct,
            confidence=confidence,
            response_value=response_value,
            trial_number=self.trial_count,
            fatigue_level=self.fatigue_accumulation,
            timestamp=datetime.now(),
        )

    def simulate_training_session(
        self,
        task_difficulty: float,
        n_trials: int,
        correct_answers: List[Any],
        response_options: List[List[Any]],
    ) -> List[SimulatedResponse]:
        """Simulate a complete training session with learning effects.

        Args:
            task_difficulty: Base difficulty level
            n_trials: Number of training trials
            correct_answers: List of correct answers for each trial
            response_options: List of response options for each trial

        Returns:
            List of SimulatedResponse objects for the training session
        """
        logger.info(
            f"Simulating training session: {n_trials} trials for {self.profile.participant_id}"
        )

        responses = []
        base_rt = 1.0  # Base response time for training

        for i in range(n_trials):
            # Adjust difficulty based on learning progress
            adjusted_difficulty = task_difficulty * (1.0 - self.learning_progress * 0.3)

            response = self.simulate_response(
                task_difficulty=adjusted_difficulty,
                base_response_time=base_rt,
                correct_answer=correct_answers[i % len(correct_answers)],
                response_options=response_options[i % len(response_options)],
            )

            responses.append(response)

        return responses

    def simulate_experimental_session(
        self,
        task_difficulties: List[float],
        base_response_times: List[float],
        correct_answers: List[Any],
        response_options: List[List[Any]],
    ) -> List[SimulatedResponse]:
        """Simulate a complete experimental session.

        Args:
            task_difficulties: Difficulty level for each trial
            base_response_times: Base response time for each trial
            correct_answers: Correct answer for each trial
            response_options: Response options for each trial

        Returns:
            List of SimulatedResponse objects for the experimental session
        """
        logger.info(
            f"Simulating experimental session: {len(task_difficulties)} trials for {self.profile.participant_id}"
        )

        responses = []

        for i, (difficulty, base_rt, correct, options) in enumerate(
            zip(
                task_difficulties,
                base_response_times,
                correct_answers,
                response_options,
            )
        ):
            response = self.simulate_response(
                task_difficulty=difficulty,
                base_response_time=base_rt,
                correct_answer=correct,
                response_options=options,
            )

            responses.append(response)

        return responses

    def _update_fatigue(self) -> None:
        """Update fatigue accumulation based on trial count and individual fatigue rate."""
        self.fatigue_accumulation = min(1.0, self.trial_count * self.profile.fatigue_rate * 0.001)

    def _update_learning(self) -> None:
        """Update learning progress based on trial count and individual learning rate."""
        # Learning follows a logarithmic curve
        if self.trial_count > 0:
            self.learning_progress = min(
                1.0, self.profile.learning_rate * np.log(1 + self.trial_count * 0.1)
            )

    def get_performance_metrics(self, responses: List[SimulatedResponse]) -> Dict[str, float]:
        """Calculate performance metrics from simulated responses.

        Args:
            responses: List of simulated responses

        Returns:
            Dictionary with performance metrics
        """
        if not responses:
            return {}

        accuracies = [r.accuracy for r in responses]
        response_times = [r.response_time for r in responses]
        confidences = [r.confidence for r in responses]

        return {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "response_time_mean": float(np.mean(response_times)),
            "response_time_std": float(np.std(response_times)),
            "confidence_mean": float(np.mean(confidences)),
            "confidence_std": float(np.std(confidences)),
            "consistency_score": float(1.0 - (np.std(response_times) / np.mean(response_times))),
            "final_fatigue": responses[-1].fatigue_level if responses else 0.0,
            "total_trials": len(responses),
        }

    def reset_session(self) -> None:
        """Reset simulator state for a new session."""
        self.trial_count = 0
        self.fatigue_accumulation = 0.0
        self.learning_progress = 0.0
        logger.info(f"Reset session state for participant {self.profile.participant_id}")


class ParticipantPopulationGenerator:
    """Generates populations of diverse participant profiles for simulation studies."""

    @staticmethod
    def generate_population(
        n_participants: int, random_seed: Optional[int] = None
    ) -> List[ParticipantProfile]:
        """Generate a diverse population of participant profiles.

        Args:
            n_participants: Number of participants to generate
            random_seed: Random seed for reproducible generation

        Returns:
            List of ParticipantProfile objects
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        profiles = []

        for i in range(n_participants):
            # Generate realistic individual differences
            age = int(np.random.normal(25, 8))  # Mean age 25, std 8
            age = np.clip(age, 18, 65)  # Reasonable age range

            # Cognitive ability follows normal distribution
            cognitive_ability = np.clip(np.random.normal(0.7, 0.15), 0.3, 1.0)

            # Attention span correlated with cognitive ability but with noise
            attention_span = np.clip(cognitive_ability + np.random.normal(0, 0.1), 0.3, 1.0)

            # Response speed varies independently
            response_speed = np.clip(np.random.normal(1.0, 0.2), 0.5, 2.0)

            # Consistency varies
            consistency = np.clip(np.random.normal(0.8, 0.1), 0.5, 1.0)

            # Fatigue and learning rates
            fatigue_rate = np.clip(np.random.normal(1.0, 0.3), 0.3, 2.0)
            learning_rate = np.clip(np.random.normal(1.0, 0.2), 0.5, 1.5)

            # Individual differences
            individual_differences = {
                "impulsivity": np.clip(np.random.normal(0.3, 0.2), 0.0, 1.0),
                "motivation": np.clip(np.random.normal(0.7, 0.2), 0.0, 1.0),
                "anxiety": np.clip(np.random.normal(0.3, 0.2), 0.0, 1.0),
                "experience": np.clip(np.random.normal(0.5, 0.3), 0.0, 1.0),
            }

            profile = ParticipantProfile(
                participant_id=f"P{i + 1:03d}",
                age=age,
                cognitive_ability=cognitive_ability,
                attention_span=attention_span,
                response_speed=response_speed,
                consistency=consistency,
                fatigue_rate=fatigue_rate,
                learning_rate=learning_rate,
                individual_differences=individual_differences,
            )

            profiles.append(profile)

        logger.info(f"Generated population of {n_participants} participant profiles")
        return profiles

    @staticmethod
    def generate_clinical_population(
        n_participants: int, condition: str, random_seed: Optional[int] = None
    ) -> List[ParticipantProfile]:
        """Generate participant profiles for specific clinical conditions.

        Args:
            n_participants: Number of participants to generate
            condition: Clinical condition ('adhd', 'depression', 'anxiety', 'control')
            random_seed: Random seed for reproducible generation

        Returns:
            List of ParticipantProfile objects with condition-specific characteristics
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        profiles = []

        for i in range(n_participants):
            # Base profile
            age = int(np.clip(np.random.normal(30, 10), 18, 65))

            if condition.lower() == "adhd":
                # ADHD characteristics
                cognitive_ability = np.clip(np.random.normal(0.65, 0.2), 0.3, 1.0)
                attention_span = np.clip(np.random.normal(0.4, 0.15), 0.1, 0.8)
                response_speed = np.clip(np.random.normal(0.8, 0.2), 0.4, 1.5)
                consistency = np.clip(np.random.normal(0.5, 0.2), 0.2, 0.9)
                individual_differences = {
                    "impulsivity": np.clip(np.random.normal(0.7, 0.2), 0.3, 1.0),
                    "motivation": np.clip(np.random.normal(0.6, 0.2), 0.2, 1.0),
                    "anxiety": np.clip(np.random.normal(0.4, 0.2), 0.0, 1.0),
                    "experience": np.clip(np.random.normal(0.5, 0.3), 0.0, 1.0),
                }

            elif condition.lower() == "depression":
                # Depression characteristics
                cognitive_ability = np.clip(np.random.normal(0.6, 0.2), 0.3, 1.0)
                attention_span = np.clip(np.random.normal(0.5, 0.2), 0.2, 0.9)
                response_speed = np.clip(np.random.normal(1.3, 0.3), 0.8, 2.5)  # Slower responses
                consistency = np.clip(np.random.normal(0.6, 0.2), 0.3, 1.0)
                individual_differences = {
                    "impulsivity": np.clip(np.random.normal(0.2, 0.15), 0.0, 0.6),
                    "motivation": np.clip(np.random.normal(0.3, 0.2), 0.0, 0.8),
                    "anxiety": np.clip(np.random.normal(0.6, 0.2), 0.2, 1.0),
                    "experience": np.clip(np.random.normal(0.5, 0.3), 0.0, 1.0),
                }

            else:  # Control condition
                cognitive_ability = np.clip(np.random.normal(0.75, 0.15), 0.4, 1.0)
                attention_span = np.clip(np.random.normal(0.8, 0.1), 0.5, 1.0)
                response_speed = np.clip(np.random.normal(1.0, 0.2), 0.6, 1.8)
                consistency = np.clip(np.random.normal(0.8, 0.1), 0.6, 1.0)
                individual_differences = {
                    "impulsivity": np.clip(np.random.normal(0.3, 0.15), 0.0, 0.8),
                    "motivation": np.clip(np.random.normal(0.7, 0.15), 0.3, 1.0),
                    "anxiety": np.clip(np.random.normal(0.3, 0.15), 0.0, 0.8),
                    "experience": np.clip(np.random.normal(0.5, 0.3), 0.0, 1.0),
                }

            fatigue_rate = np.clip(np.random.normal(1.0, 0.3), 0.3, 2.0)
            learning_rate = np.clip(np.random.normal(1.0, 0.2), 0.5, 1.5)

            profile = ParticipantProfile(
                participant_id=f"{condition.upper()}{i + 1:03d}",
                age=age,
                cognitive_ability=cognitive_ability,
                attention_span=attention_span,
                response_speed=response_speed,
                consistency=consistency,
                fatigue_rate=fatigue_rate,
                learning_rate=learning_rate,
                individual_differences=individual_differences,
            )

            profiles.append(profile)

        logger.info(f"Generated {condition} population of {n_participants} participant profiles")
        return profiles


@dataclass
class StimulusProperties:
    """Properties of experimental stimuli."""

    stimulus_id: str
    presentation_duration: float  # seconds
    contrast: float  # 0-1 scale
    size: float  # visual angle or other units
    luminance: float  # cd/m²
    is_supraliminal: bool
    detection_threshold: float
    stimulus_type: str  # 'visual', 'auditory', 'tactile'


@dataclass
class TaskValidationResult:
    """Results of task validation assessment."""

    task_id: str
    is_valid: bool
    stimulus_validation: bool
    training_validation: bool
    performance_validation: bool
    validation_score: float
    issues_detected: List[str]
    recommendations: List[str]
    timestamp: datetime


@dataclass
class TrainingAssessment:
    """Assessment of participant training adequacy."""

    task_id: str
    participant_id: str
    training_trials_completed: int
    final_performance: float
    learning_curve_slope: float
    performance_stability: float
    criterion_met: bool
    additional_training_needed: int
    timestamp: datetime


class StimulusValidator:
    """Validates experimental stimuli for proper presentation parameters.

    Ensures stimuli are presented supraliminally and meet detection thresholds
    for valid experimental conditions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize stimulus validator.

        Args:
            config: Configuration parameters for stimulus validation
        """
        self.config = config or self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for stimulus validation."""
        return {
            "detection_thresholds": {
                "visual_contrast": 0.05,  # 5% contrast minimum
                "visual_duration": 0.05,  # 50ms minimum
                "auditory_intensity": 40,  # 40 dB above threshold
                "tactile_intensity": 2.0,  # 2x detection threshold
            },
            "supraliminal_criteria": {
                "visual_contrast_multiplier": 3.0,  # 3x detection threshold
                "duration_multiplier": 2.0,  # 2x minimum duration
                "intensity_multiplier": 2.5,  # 2.5x detection threshold
            },
        }

    def validate_stimulus(self, stimulus: StimulusProperties) -> Tuple[bool, List[str]]:
        """Validate stimulus properties for experimental adequacy.

        Args:
            stimulus: StimulusProperties to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check supraliminal presentation
        if not stimulus.is_supraliminal:
            issues.append("Stimulus marked as subliminal - requires supraliminal presentation")

        # Validate based on stimulus type
        if stimulus.stimulus_type == "visual":
            issues.extend(self._validate_visual_stimulus(stimulus))
        elif stimulus.stimulus_type == "auditory":
            issues.extend(self._validate_auditory_stimulus(stimulus))
        elif stimulus.stimulus_type == "tactile":
            issues.extend(self._validate_tactile_stimulus(stimulus))

        # Check presentation duration
        min_duration = self.config["detection_thresholds"]["visual_duration"]
        if stimulus.presentation_duration < min_duration:
            issues.append(
                f"Presentation duration too short: {stimulus.presentation_duration:.3f}s < {min_duration:.3f}s"
            )

        is_valid = len(issues) == 0
        return is_valid, issues

    def _validate_visual_stimulus(self, stimulus: StimulusProperties) -> List[str]:
        """Validate visual stimulus properties."""
        issues = []

        min_contrast = self.config["detection_thresholds"]["visual_contrast"]
        supraliminal_contrast = (
            min_contrast * self.config["supraliminal_criteria"]["visual_contrast_multiplier"]
        )

        if stimulus.contrast < supraliminal_contrast:
            issues.append(
                f"Visual contrast too low: {stimulus.contrast:.3f} < {supraliminal_contrast:.3f}"
            )

        if stimulus.size <= 0:
            issues.append("Visual stimulus size must be positive")

        if stimulus.luminance <= 0:
            issues.append("Visual stimulus luminance must be positive")

        return issues

    def _validate_auditory_stimulus(self, stimulus: StimulusProperties) -> List[str]:
        """Validate auditory stimulus properties."""
        issues = []

        min_intensity = self.config["detection_thresholds"]["auditory_intensity"]
        supraliminal_intensity = (
            min_intensity * self.config["supraliminal_criteria"]["intensity_multiplier"]
        )

        # Assuming luminance field is repurposed for intensity in auditory stimuli
        if stimulus.luminance < supraliminal_intensity:
            issues.append(
                f"Auditory intensity too low: {stimulus.luminance:.1f} < {supraliminal_intensity:.1f} dB"
            )

        return issues

    def _validate_tactile_stimulus(self, stimulus: StimulusProperties) -> List[str]:
        """Validate tactile stimulus properties."""
        issues = []

        min_intensity = self.config["detection_thresholds"]["tactile_intensity"]
        supraliminal_intensity = (
            min_intensity * self.config["supraliminal_criteria"]["intensity_multiplier"]
        )

        if stimulus.detection_threshold * supraliminal_intensity > stimulus.luminance:
            issues.append("Tactile intensity insufficient for supraliminal detection")

        return issues


class TaskValidator:
    """Validates experimental tasks for proper training and performance criteria.

    Ensures participants are well-trained and tasks meet experimental standards.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize task validator.

        Args:
            config: Configuration parameters for task validation
        """
        self.config = config or self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for task validation."""
        return {
            "training_criteria": {
                "minimum_trials": 50,
                "performance_threshold": 0.8,  # 80% accuracy
                "stability_window": 10,  # trials for stability check
                "stability_threshold": 0.05,  # max std dev for stability
                "learning_threshold": 0.1,  # minimum improvement slope
            },
            "task_requirements": {
                "minimum_task_duration": 0.5,  # 500ms minimum
                "maximum_task_duration": 10.0,  # 10s maximum
                "response_window": 5.0,  # 5s response window
            },
        }

    def assess_training_adequacy(
        self,
        task_id: str,
        participant_id: str,
        training_responses: List[SimulatedResponse],
    ) -> TrainingAssessment:
        """Assess whether participant training is adequate for experimental participation.

        Args:
            task_id: Unique task identifier
            participant_id: Unique participant identifier
            training_responses: List of training trial responses

        Returns:
            TrainingAssessment with training adequacy evaluation
        """
        logger.info(f"Assessing training adequacy for task {task_id}, participant {participant_id}")

        if not training_responses:
            return TrainingAssessment(
                task_id=task_id,
                participant_id=participant_id,
                training_trials_completed=0,
                final_performance=0.0,
                learning_curve_slope=0.0,
                performance_stability=0.0,
                criterion_met=False,
                additional_training_needed=self.config["training_criteria"]["minimum_trials"],
                timestamp=datetime.now(),
            )

        # Calculate performance metrics
        accuracies = [r.accuracy for r in training_responses]
        trials_completed = len(accuracies)

        # Calculate final performance (last 10 trials or all if fewer)
        final_window = min(10, len(accuracies))
        final_performance = np.mean(accuracies[-final_window:]) if accuracies else 0.0

        # Calculate learning curve slope
        if len(accuracies) >= 10:
            # Fit linear trend to accuracy over time
            x = np.arange(len(accuracies))
            y = np.array(accuracies, dtype=float)
            slope, _ = np.polyfit(x, y, 1)
            learning_curve_slope = slope
        else:
            learning_curve_slope = 0.0

        # Calculate performance stability (std dev of final window)
        stability_window = min(
            self.config["training_criteria"]["stability_window"], len(accuracies)
        )
        if stability_window > 1:
            stability_data = accuracies[-stability_window:]
            performance_stability = float(1.0 - np.std(stability_data))  # Higher = more stable
        else:
            performance_stability = 0.0

        # Determine if training criteria are met
        criteria = self.config["training_criteria"]
        sufficient_trials = trials_completed >= criteria["minimum_trials"]
        sufficient_performance = final_performance >= criteria["performance_threshold"]
        sufficient_stability = performance_stability >= (1.0 - criteria["stability_threshold"])
        sufficient_learning = (
            learning_curve_slope >= -criteria["learning_threshold"]
        )  # Not declining

        criterion_met = (
            sufficient_trials
            and sufficient_performance
            and sufficient_stability
            and sufficient_learning
        )

        # Calculate additional training needed
        if criterion_met:
            additional_training_needed = 0
        else:
            additional_needed = max(0, criteria["minimum_trials"] - trials_completed)
            if not sufficient_performance:
                additional_needed = max(additional_needed, 20)  # At least 20 more trials
            if not sufficient_stability:
                additional_needed = max(additional_needed, 15)  # At least 15 more trials
            additional_training_needed = additional_needed

        return TrainingAssessment(
            task_id=task_id,
            participant_id=participant_id,
            training_trials_completed=trials_completed,
            final_performance=float(final_performance),
            learning_curve_slope=learning_curve_slope,
            performance_stability=float(performance_stability),
            criterion_met=criterion_met,
            additional_training_needed=additional_training_needed,
            timestamp=datetime.now(),
        )

    def validate_task_design(
        self,
        task_id: str,
        stimuli: List[StimulusProperties],
        task_duration: float,
        response_window: float,
    ) -> TaskValidationResult:
        """Validate overall task design for experimental adequacy.

        Args:
            task_id: Unique task identifier
            stimuli: List of stimuli used in the task
            task_duration: Total task duration in seconds
            response_window: Response window duration in seconds

        Returns:
            TaskValidationResult with comprehensive task validation
        """
        logger.info(f"Validating task design for task {task_id}")

        issues = []
        recommendations = []

        # Validate task timing
        requirements = self.config["task_requirements"]
        if task_duration < requirements["minimum_task_duration"]:
            issues.append(
                f"Task duration too short: {task_duration:.1f}s < {requirements['minimum_task_duration']:.1f}s"
            )

        if task_duration > requirements["maximum_task_duration"]:
            issues.append(
                f"Task duration too long: {task_duration:.1f}s > {requirements['maximum_task_duration']:.1f}s"
            )

        if response_window > requirements["response_window"]:
            recommendations.append(f"Consider shorter response window: {response_window:.1f}s")

        # Validate stimuli
        stimulus_validator = StimulusValidator(self.config)
        stimulus_issues = []

        for stimulus in stimuli:
            is_valid, stim_issues = stimulus_validator.validate_stimulus(stimulus)
            if not is_valid:
                stimulus_issues.extend(
                    [f"Stimulus {stimulus.stimulus_id}: {issue}" for issue in stim_issues]
                )

        issues.extend(stimulus_issues)

        # Calculate validation scores
        stimulus_validation = len(stimulus_issues) == 0
        training_validation = True  # Will be assessed per participant
        performance_validation = (
            requirements["minimum_task_duration"]
            <= task_duration
            <= requirements["maximum_task_duration"]
        )

        # Overall validation score
        validation_components = [
            stimulus_validation,
            training_validation,
            performance_validation,
        ]
        validation_score = sum(validation_components) / len(validation_components)

        # Generate recommendations
        if not stimulus_validation:
            recommendations.append(
                "Review and adjust stimulus parameters for supraliminal presentation"
            )

        if not performance_validation:
            recommendations.append(
                "Adjust task timing parameters to meet experimental requirements"
            )

        if not recommendations:
            recommendations.append("Task design meets all validation criteria")

        is_valid = len(issues) == 0

        return TaskValidationResult(
            task_id=task_id,
            is_valid=is_valid,
            stimulus_validation=stimulus_validation,
            training_validation=training_validation,
            performance_validation=performance_validation,
            validation_score=validation_score,
            issues_detected=issues,
            recommendations=recommendations,
            timestamp=datetime.now(),
        )


class WellTrainedTaskChecker:
    """Checks if participants are well-trained on experimental tasks.

    Implements comprehensive assessment of task proficiency and readiness
    for experimental participation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize well-trained task checker.

        Args:
            config: Configuration parameters for training assessment
        """
        self.config = config or self._get_default_config()
        self.task_validator = TaskValidator(config)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for training assessment."""
        return {
            "proficiency_criteria": {
                "expert_threshold": 0.9,  # 90% accuracy for expert level
                "proficient_threshold": 0.8,  # 80% accuracy for proficient
                "minimum_sessions": 3,  # Minimum training sessions
                "consistency_threshold": 0.85,  # 85% consistency across sessions
            },
            "readiness_criteria": {
                "performance_stability": 0.9,  # 90% stability required
                "cross_session_correlation": 0.7,  # 70% correlation between sessions
                "fatigue_resistance": 0.8,  # 80% performance maintenance
            },
        }

    def assess_task_proficiency(
        self,
        participant_id: str,
        task_id: str,
        session_responses: List[List[SimulatedResponse]],
    ) -> Dict[str, Any]:
        """Assess participant proficiency across multiple training sessions.

        Args:
            participant_id: Unique participant identifier
            task_id: Unique task identifier
            session_responses: List of response lists, one per training session

        Returns:
            Dictionary with proficiency assessment results
        """
        logger.info(f"Assessing task proficiency for {participant_id} on task {task_id}")

        if not session_responses:
            return {
                "proficiency_level": "insufficient",
                "overall_score": 0.0,
                "is_ready": False,
                "sessions_completed": 0,
                "recommendations": ["Complete initial training sessions"],
            }

        # Calculate per-session performance
        session_performances = []
        session_stabilities = []

        for session in session_responses:
            if session:
                accuracies = [r.accuracy for r in session]
                performance = np.mean(accuracies)
                stability = 1.0 - np.std(accuracies) if len(accuracies) > 1 else 1.0

                session_performances.append(performance)
                session_stabilities.append(stability)

        if not session_performances:
            return {
                "proficiency_level": "insufficient",
                "overall_score": 0.0,
                "is_ready": False,
                "sessions_completed": 0,
                "recommendations": ["Complete training trials in sessions"],
            }

        # Calculate overall metrics
        overall_performance = float(np.mean([float(p) for p in session_performances]))
        performance_consistency = (
            1.0 - np.std(session_performances) if len(session_performances) > 1 else 1.0
        )
        average_stability = float(np.mean([float(s) for s in session_stabilities]))

        # Calculate cross-session correlation (learning trajectory)
        if len(session_performances) >= 3:
            x = np.arange(len(session_performances))
            correlation = np.corrcoef(x, session_performances)[0, 1]
            cross_session_correlation = max(0, correlation)  # Only positive correlations count
        else:
            cross_session_correlation = 0.5  # Neutral for insufficient data

        # Assess fatigue resistance (performance maintenance across trials within sessions)
        fatigue_resistance_scores = []
        for session in session_responses:
            if len(session) >= 20:  # Need sufficient trials
                first_half = [r.accuracy for r in session[: len(session) // 2]]
                second_half = [r.accuracy for r in session[len(session) // 2 :]]

                first_performance = np.mean(first_half)
                second_performance = np.mean(second_half)

                # Fatigue resistance = how well performance is maintained
                if first_performance > 0:
                    resistance = second_performance / first_performance
                    fatigue_resistance_scores.append(min(1.0, float(resistance)))

        fatigue_resistance = (
            np.mean(fatigue_resistance_scores) if fatigue_resistance_scores else 0.5
        )

        # Determine proficiency level
        criteria = self.config["proficiency_criteria"]
        readiness_criteria = self.config["readiness_criteria"]

        sessions_completed = len(session_responses)
        sufficient_sessions = sessions_completed >= criteria["minimum_sessions"]

        if overall_performance >= criteria["expert_threshold"]:
            proficiency_level = "expert"
        elif overall_performance >= criteria["proficient_threshold"]:
            proficiency_level = "proficient"
        else:
            proficiency_level = "novice"

        # Check readiness criteria
        stability_met = average_stability >= readiness_criteria["performance_stability"]
        correlation_met = (
            cross_session_correlation >= readiness_criteria["cross_session_correlation"]
        )
        fatigue_met = fatigue_resistance >= readiness_criteria["fatigue_resistance"]
        consistency_met = performance_consistency >= criteria["consistency_threshold"]

        is_ready = (
            sufficient_sessions
            and proficiency_level in ["proficient", "expert"]
            and stability_met
            and correlation_met
            and fatigue_met
            and consistency_met
        )

        # Calculate overall score
        score_components = [
            overall_performance,
            performance_consistency,
            average_stability,
            cross_session_correlation,
            fatigue_resistance,
        ]
        overall_score = np.mean(score_components)

        # Generate recommendations
        recommendations = []
        if not sufficient_sessions:
            recommendations.append(
                f"Complete {criteria['minimum_sessions'] - sessions_completed} more training sessions"
            )

        if proficiency_level == "novice":
            recommendations.append("Continue training to reach proficient level (80% accuracy)")

        if not stability_met:
            recommendations.append("Improve within-session performance stability")

        if not correlation_met:
            recommendations.append("Focus on consistent learning progression across sessions")

        if not fatigue_met:
            recommendations.append("Build resistance to fatigue effects during long sessions")

        if not consistency_met:
            recommendations.append("Improve consistency of performance across sessions")

        if is_ready:
            recommendations = ["Participant ready for experimental participation"]

        return {
            "proficiency_level": proficiency_level,
            "overall_score": overall_score,
            "is_ready": is_ready,
            "sessions_completed": sessions_completed,
            "overall_performance": overall_performance,
            "performance_consistency": performance_consistency,
            "average_stability": average_stability,
            "cross_session_correlation": cross_session_correlation,
            "fatigue_resistance": fatigue_resistance,
            "recommendations": recommendations,
            "detailed_metrics": {
                "session_performances": session_performances,
                "session_stabilities": session_stabilities,
                "criteria_met": {
                    "sufficient_sessions": sufficient_sessions,
                    "stability_met": stability_met,
                    "correlation_met": correlation_met,
                    "fatigue_met": fatigue_met,
                    "consistency_met": consistency_met,
                },
            },
        }


@dataclass
class ConsciousnessMeasure:
    """Individual consciousness measurement result."""

    measure_type: (
        str  # 'subjective_report', 'forced_choice', 'confidence', 'wagering', 'metacognitive'
    )
    value: float
    confidence_interval: Tuple[float, float]
    reliability: float
    validity_score: float
    timestamp: datetime


@dataclass
class MetacognitiveAssessment:
    """Metacognitive sensitivity assessment results."""

    participant_id: str
    type_1_performance: float  # Primary task accuracy
    type_2_performance: float  # Metacognitive accuracy
    meta_d_prime: float  # Metacognitive sensitivity
    confidence_accuracy_correlation: float
    calibration_score: float  # How well confidence matches accuracy
    resolution_score: float  # Ability to discriminate correct/incorrect
    is_intact: bool
    assessment_details: Dict[str, Any]
    timestamp: datetime


@dataclass
class ConsciousnessValidationReport:
    """Comprehensive consciousness measurement validation report."""

    participant_id: str
    session_id: str
    measures: List[ConsciousnessMeasure]
    metacognitive_assessment: MetacognitiveAssessment
    convergent_validity: float
    measurement_reliability: float
    overall_validity: bool
    recommendations: List[str]
    timestamp: datetime


class ConsciousnessMeasurementValidator:
    """Validates consciousness measurements using multiple converging measures.

    Implements comprehensive assessment of consciousness measurement validity,
    including metacognitive sensitivity and confidence-accuracy correspondence.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize consciousness measurement validator.

        Args:
            config: Configuration parameters for consciousness measurement validation
        """
        self.config = config or self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for consciousness measurement validation."""
        return {
            "validity_thresholds": {
                "minimum_correlation": 0.3,  # Minimum inter-measure correlation
                "reliability_threshold": 0.7,  # Minimum reliability score
                "metacognitive_threshold": 0.5,  # Minimum meta-d' for intact metacognition
                "calibration_threshold": 0.6,  # Minimum calibration score
            },
            "measurement_criteria": {
                "minimum_measures": 3,  # Minimum number of consciousness measures
                "convergent_threshold": 0.4,  # Minimum convergent validity
                "confidence_accuracy_r": 0.2,  # Minimum confidence-accuracy correlation
            },
        }

    def assess_metacognitive_sensitivity(
        self,
        participant_id: str,
        primary_responses: List[bool],
        confidence_ratings: List[float],
        correct_answers: List[bool],
    ) -> MetacognitiveAssessment:
        """Assess metacognitive sensitivity and confidence-accuracy correspondence.

        Args:
            participant_id: Unique participant identifier
            primary_responses: List of participant responses (True/False)
            confidence_ratings: List of confidence ratings (0-1)
            correct_answers: List of correct answers (True/False)

        Returns:
            MetacognitiveAssessment with detailed metacognitive evaluation
        """
        logger.info(f"Assessing metacognitive sensitivity for participant {participant_id}")

        if len(primary_responses) != len(confidence_ratings) or len(primary_responses) != len(
            correct_answers
        ):
            raise ValueError("All input lists must have the same length")

        # Calculate Type 1 performance (primary task accuracy)
        type_1_accuracy = np.mean(
            [resp == correct for resp, correct in zip(primary_responses, correct_answers)]
        )

        # Calculate confidence-accuracy correlation
        accuracies = [
            1.0 if resp == correct else 0.0
            for resp, correct in zip(primary_responses, correct_answers)
        ]
        if len(set(confidence_ratings)) > 1 and len(set(accuracies)) > 1:
            conf_acc_correlation = np.corrcoef(confidence_ratings, accuracies)[0, 1]
        else:
            conf_acc_correlation = 0.0

        # Calculate metacognitive sensitivity (simplified meta-d')
        # This is a simplified version - in practice, you'd use more sophisticated SDT analysis
        correct_trials = [
            i
            for i, (resp, correct) in enumerate(zip(primary_responses, correct_answers))
            if resp == correct
        ]
        incorrect_trials = [
            i
            for i, (resp, correct) in enumerate(zip(primary_responses, correct_answers))
            if resp != correct
        ]

        if correct_trials and incorrect_trials:
            correct_confidences = [confidence_ratings[i] for i in correct_trials]
            incorrect_confidences = [confidence_ratings[i] for i in incorrect_trials]

            # Meta-d' approximation: difference in confidence between correct and incorrect trials
            mean_conf_correct = np.mean(correct_confidences)
            mean_conf_incorrect = np.mean(incorrect_confidences)
            pooled_std = np.sqrt((np.var(correct_confidences) + np.var(incorrect_confidences)) / 2)

            if pooled_std > 0:
                meta_d_prime = (mean_conf_correct - mean_conf_incorrect) / pooled_std
            else:
                meta_d_prime = 0.0
        else:
            meta_d_prime = 0.0

        # Calculate calibration (how well confidence matches accuracy)
        # Bin confidence ratings and calculate calibration curve
        n_bins = 5
        bin_edges = np.linspace(0, 1, n_bins + 1)
        calibration_scores = []

        for i in range(n_bins):
            bin_mask = (confidence_ratings >= bin_edges[i]) & (
                confidence_ratings < bin_edges[i + 1]
            )
            if i == n_bins - 1:  # Include upper bound for last bin
                bin_mask = (confidence_ratings >= bin_edges[i]) & (
                    confidence_ratings <= bin_edges[i + 1]
                )

            if np.any(bin_mask):
                bin_confidences = np.array(confidence_ratings)[bin_mask]
                bin_accuracies = np.array(accuracies)[bin_mask]

                mean_confidence = np.mean(bin_confidences)
                mean_accuracy = np.mean(bin_accuracies)

                # Calibration error for this bin
                calibration_error = abs(mean_confidence - mean_accuracy)
                calibration_scores.append(
                    1.0 - calibration_error
                )  # Convert to score (higher = better)

        calibration_score = np.mean(calibration_scores) if calibration_scores else 0.0

        # Calculate resolution (ability to discriminate correct from incorrect)
        # This is related to the variance in confidence ratings
        confidence_variance = np.var(confidence_ratings)
        max_variance = 0.25  # Maximum variance for uniform distribution on [0,1]
        resolution_score = confidence_variance / max_variance if max_variance > 0 else 0.0

        # Calculate Type 2 performance (metacognitive accuracy)
        # Simplified: correlation between confidence and accuracy
        type_2_performance = max(0, conf_acc_correlation)

        # Determine if metacognition is intact
        validity_thresholds = self.config["validity_thresholds"]
        measurement_criteria = self.config["measurement_criteria"]
        is_intact = (
            meta_d_prime >= validity_thresholds["metacognitive_threshold"]
            and conf_acc_correlation >= measurement_criteria["confidence_accuracy_r"]
            and calibration_score >= validity_thresholds["calibration_threshold"]
        )

        assessment_details = {
            "n_trials": len(primary_responses),
            "correct_trials": len(correct_trials),
            "incorrect_trials": len(incorrect_trials),
            "confidence_range": (min(confidence_ratings), max(confidence_ratings)),
            "calibration_bins": n_bins,
            "resolution_details": {
                "confidence_variance": confidence_variance,
                "max_possible_variance": max_variance,
            },
        }

        return MetacognitiveAssessment(
            participant_id=participant_id,
            type_1_performance=float(type_1_accuracy),
            type_2_performance=type_2_performance,
            meta_d_prime=meta_d_prime,
            confidence_accuracy_correlation=conf_acc_correlation,
            calibration_score=calibration_score,
            resolution_score=float(resolution_score),
            is_intact=is_intact,
            assessment_details=assessment_details,
            timestamp=datetime.now(),
        )

    def validate_consciousness_measure(
        self,
        measure_type: str,
        values: List[float],
        reference_values: Optional[List[float]] = None,
    ) -> ConsciousnessMeasure:
        """Validate individual consciousness measurement.

        Args:
            measure_type: Type of consciousness measure
            values: List of measurement values
            reference_values: Optional reference values for validation

        Returns:
            ConsciousnessMeasure with validation results
        """
        if not values:
            return ConsciousnessMeasure(
                measure_type=measure_type,
                value=0.0,
                confidence_interval=(0.0, 0.0),
                reliability=0.0,
                validity_score=0.0,
                timestamp=datetime.now(),
            )

        # Calculate basic statistics
        mean_value = np.mean(values)
        std_value = np.std(values)
        n = len(values)

        # Calculate confidence interval (95%)
        if n > 1:
            sem = std_value / np.sqrt(n)
            ci_margin = 1.96 * sem  # 95% CI
            confidence_interval = (mean_value - ci_margin, mean_value + ci_margin)
        else:
            confidence_interval = (mean_value, mean_value)

        # Calculate reliability (internal consistency)
        if n >= 10:
            # Split-half reliability
            half1 = values[: n // 2]
            half2 = values[n // 2 : n // 2 * 2]  # Ensure equal lengths

            if len(half1) == len(half2) and len(set(half1)) > 1 and len(set(half2)) > 1:
                reliability = max(0, np.corrcoef(half1, half2)[0, 1])
            else:
                reliability = 0.5  # Moderate reliability assumed
        else:
            reliability = 0.3  # Low reliability for insufficient data

        # Calculate validity score
        validity_components = []

        # Range validity (values within expected range)
        if measure_type in ["confidence", "wagering"]:
            range_validity = float(np.mean([(0 <= v <= 1) for v in values]))
        elif measure_type == "forced_choice":
            range_validity = float(np.mean([(0 <= v <= 1) for v in values]))
        else:
            range_validity = 1.0  # Assume valid range for other measures

        validity_components.append(range_validity)

        # Variability validity (sufficient variability to be informative)
        if std_value > 0.05:  # Some minimum variability expected
            variability_validity = min(1.0, float(std_value) / 0.3)  # Normalize to reasonable range
        else:
            variability_validity = 0.1  # Very low validity for no variability

        validity_components.append(variability_validity)

        # Reference validity (correlation with reference if provided)
        if reference_values and len(reference_values) == len(values):
            if len(set(values)) > 1 and len(set(reference_values)) > 1:
                ref_correlation = max(0.0, float(np.corrcoef(values, reference_values)[0, 1]))
                validity_components.append(ref_correlation)

        validity_score = float(np.mean(validity_components))

        return ConsciousnessMeasure(
            measure_type=measure_type,
            value=float(mean_value),
            confidence_interval=confidence_interval,
            reliability=reliability,
            validity_score=validity_score,
            timestamp=datetime.now(),
        )

    def generate_consciousness_validation_report(
        self,
        participant_id: str,
        session_id: str,
        subjective_reports: List[bool],
        forced_choice_responses: List[bool],
        confidence_ratings: List[float],
        wagering_responses: List[float],
        correct_answers: List[bool],
    ) -> ConsciousnessValidationReport:
        """Generate comprehensive consciousness measurement validation report.

        Args:
            participant_id: Unique participant identifier
            session_id: Unique session identifier
            subjective_reports: List of subjective awareness reports
            forced_choice_responses: List of forced-choice responses
            confidence_ratings: List of confidence ratings
            wagering_responses: List of wagering behavior responses
            correct_answers: List of correct answers

        Returns:
            ConsciousnessValidationReport with comprehensive validation
        """
        logger.info(
            f"Generating consciousness validation report for {participant_id}, session {session_id}"
        )

        # Validate individual measures
        measures = []

        # Subjective reports
        subjective_values = [1.0 if report else 0.0 for report in subjective_reports]
        subjective_measure = self.validate_consciousness_measure(
            "subjective_report", subjective_values
        )
        measures.append(subjective_measure)

        # Forced choice
        forced_choice_values = [1.0 if resp else 0.0 for resp in forced_choice_responses]
        forced_choice_measure = self.validate_consciousness_measure(
            "forced_choice", forced_choice_values
        )
        measures.append(forced_choice_measure)

        # Confidence ratings
        confidence_measure = self.validate_consciousness_measure("confidence", confidence_ratings)
        measures.append(confidence_measure)

        # Wagering responses
        wagering_measure = self.validate_consciousness_measure("wagering", wagering_responses)
        measures.append(wagering_measure)

        # Assess metacognitive sensitivity
        metacognitive_assessment = self.assess_metacognitive_sensitivity(
            participant_id, forced_choice_responses, confidence_ratings, correct_answers
        )

        # Calculate convergent validity (inter-measure correlations)
        measure_values = [m.value for m in measures]
        if len(set(measure_values)) > 1:
            # Calculate pairwise correlations
            correlations = []
            for i in range(len(measures)):
                for j in range(i + 1, len(measures)):
                    # Get raw values for correlation
                    if measures[i].measure_type == "subjective_report":
                        values_i = subjective_values
                    elif measures[i].measure_type == "forced_choice":
                        values_i = forced_choice_values
                    elif measures[i].measure_type == "confidence":
                        values_i = confidence_ratings
                    else:  # wagering
                        values_i = wagering_responses

                    if measures[j].measure_type == "subjective_report":
                        values_j = subjective_values
                    elif measures[j].measure_type == "forced_choice":
                        values_j = forced_choice_values
                    elif measures[j].measure_type == "confidence":
                        values_j = confidence_ratings
                    else:  # wagering
                        values_j = wagering_responses

                    if len(set(values_i)) > 1 and len(set(values_j)) > 1:
                        corr = np.corrcoef(values_i, values_j)[0, 1]
                        correlations.append(max(0, corr))  # Only positive correlations

            convergent_validity = np.mean(correlations) if correlations else 0.0
        else:
            convergent_validity = 0.0

        # Calculate overall measurement reliability
        reliabilities = [m.reliability for m in measures]
        measurement_reliability = np.mean(reliabilities)

        # Determine overall validity
        criteria = self.config["measurement_criteria"]
        thresholds = self.config["validity_thresholds"]

        sufficient_measures = len(measures) >= criteria["minimum_measures"]
        sufficient_convergence = convergent_validity >= criteria["convergent_threshold"]
        sufficient_reliability = measurement_reliability >= thresholds["reliability_threshold"]
        intact_metacognition = metacognitive_assessment.is_intact

        overall_validity = (
            sufficient_measures
            and sufficient_convergence
            and sufficient_reliability
            and intact_metacognition
        )

        # Generate recommendations
        recommendations = []
        if not sufficient_measures:
            recommendations.append(
                f"Add {criteria['minimum_measures'] - len(measures)} more consciousness measures"
            )

        if not sufficient_convergence:
            recommendations.append("Improve convergent validity between consciousness measures")

        if not sufficient_reliability:
            recommendations.append(
                "Increase measurement reliability through more trials or better instructions"
            )

        if not intact_metacognition:
            recommendations.append(
                "Address metacognitive sensitivity issues - consider participant exclusion"
            )

        if overall_validity:
            recommendations = ["All consciousness measurement criteria met successfully"]

        return ConsciousnessValidationReport(
            participant_id=participant_id,
            session_id=session_id,
            measures=measures,
            metacognitive_assessment=metacognitive_assessment,
            convergent_validity=convergent_validity,
            measurement_reliability=float(measurement_reliability),
            overall_validity=overall_validity,
            recommendations=recommendations,
            timestamp=datetime.now(),
        )


@dataclass
class ExperimentalCondition:
    """Definition of an experimental condition."""

    condition_id: str
    condition_name: str
    parameters: Dict[str, Any]
    expected_outcomes: Dict[str, Any]
    control_requirements: List[str]
    validation_criteria: Dict[str, float]


@dataclass
class IntegrityCheckResult:
    """Result of a single integrity check."""

    check_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    issues: List[str]
    recommendations: List[str]
    timestamp: datetime


@dataclass
class ExperimentalIntegrityReport:
    """Comprehensive experimental integrity assessment report."""

    experiment_id: str
    participant_id: str
    session_id: str
    condition_checks: List[IntegrityCheckResult]
    control_checks: List[IntegrityCheckResult]
    validation_checks: List[IntegrityCheckResult]
    overall_integrity_score: float
    is_valid_experiment: bool
    critical_issues: List[str]
    recommendations: List[str]
    timestamp: datetime


class ExperimentalIntegrityChecker:
    """Comprehensive experimental integrity checker and validator.

    This class orchestrates all experimental control and validation mechanisms
    to ensure experimental integrity and validity of testing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize experimental integrity checker.

        Args:
            config: Configuration parameters for integrity checking
        """
        self.config = config or self._get_default_config()

        # Initialize component validators
        self.control_manager = ExperimentalControlManager(config)
        self.consciousness_validator = ConsciousnessMeasurementValidator(config)
        self.task_validator = TaskValidator(config)
        self.stimulus_validator = StimulusValidator(config)

        logger.info("Initialized ExperimentalIntegrityChecker")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for integrity checking."""
        return {
            "integrity_thresholds": {
                "minimum_overall_score": 0.8,  # 80% overall integrity required
                "critical_check_threshold": 0.9,  # 90% for critical checks
                "warning_threshold": 0.7,  # 70% threshold for warnings
                "participant_validity_threshold": 0.8,  # 80% participant validity
            },
            "critical_checks": [
                "response_system_integrity",
                "consciousness_measurement_validity",
                "experimental_condition_compliance",
                "data_quality_assessment",
            ],
            "validation_weights": {
                "participant_controls": 0.3,
                "task_validation": 0.2,
                "consciousness_measures": 0.25,
                "experimental_conditions": 0.25,
            },
        }

    def check_response_system_integrity(
        self, participant_id: str, response_data: Dict[str, Dict[str, Any]]
    ) -> IntegrityCheckResult:
        """Check integrity of participant response systems.

        Args:
            participant_id: Unique participant identifier
            response_data: Dictionary with response times and accuracies by system type

        Returns:
            IntegrityCheckResult for response system integrity
        """
        logger.info(f"Checking response system integrity for participant {participant_id}")

        issues = []
        recommendations = []
        system_scores = []

        for system_type_str, data in response_data.items():
            try:
                system_type = ResponseSystemType(system_type_str.lower())

                # Extract response times and accuracies
                if "response_times" in data and "accuracies" in data:
                    response_times = list(data["response_times"])
                    accuracies = list(data["accuracies"])

                    validation = self.control_manager.validate_response_system(
                        participant_id, system_type, response_times, accuracies
                    )

                    if validation.is_intact:
                        system_scores.append(validation.accuracy * validation.consistency_score)
                    else:
                        issues.append(
                            f"{system_type.value} system integrity compromised: {validation.notes}"
                        )
                        system_scores.append(0.5)  # Partial score for compromised system
                        recommendations.append(
                            f"Investigate {system_type.value} response system issues"
                        )

            except ValueError:
                issues.append(f"Unknown response system type: {system_type_str}")
                recommendations.append("Verify response system type specification")

        # Calculate overall score
        overall_score = np.mean(system_scores) if system_scores else 0.0
        passed = overall_score >= self.config["integrity_thresholds"]["critical_check_threshold"]

        details = {
            "systems_checked": len(response_data),
            "systems_intact": sum(1 for score in system_scores if score > 0.8),
            "average_system_score": overall_score,
            "individual_scores": dict(zip(response_data.keys(), system_scores)),
        }

        return IntegrityCheckResult(
            check_name="response_system_integrity",
            passed=passed,
            score=float(overall_score),
            details=details,
            issues=issues,
            recommendations=recommendations,
            timestamp=datetime.now(),
        )

    def check_consciousness_measurement_validity(
        self, participant_id: str, session_id: str, consciousness_data: Dict[str, List]
    ) -> IntegrityCheckResult:
        """Check validity of consciousness measurements.

        Args:
            participant_id: Unique participant identifier
            session_id: Unique session identifier
            consciousness_data: Dictionary with consciousness measurement data

        Returns:
            IntegrityCheckResult for consciousness measurement validity
        """
        logger.info(
            f"Checking consciousness measurement validity for {participant_id}, session {session_id}"
        )

        try:
            # Extract consciousness measurement data
            subjective_reports = consciousness_data.get("subjective_reports", [])
            forced_choice = consciousness_data.get("forced_choice_responses", [])
            confidence_ratings = consciousness_data.get("confidence_ratings", [])
            wagering_responses = consciousness_data.get("wagering_responses", [])
            correct_answers = consciousness_data.get("correct_answers", [])

            # Generate validation report
            validation_report = (
                self.consciousness_validator.generate_consciousness_validation_report(
                    participant_id,
                    session_id,
                    subjective_reports,
                    forced_choice,
                    confidence_ratings,
                    wagering_responses,
                    correct_answers,
                )
            )

            passed = validation_report.overall_validity
            score = (
                validation_report.convergent_validity
                + validation_report.measurement_reliability
                + (1.0 if validation_report.metacognitive_assessment.is_intact else 0.0)
            ) / 3.0

            issues = []
            if not passed:
                issues.extend(validation_report.recommendations)

            details = {
                "measures_validated": len(validation_report.measures),
                "convergent_validity": validation_report.convergent_validity,
                "measurement_reliability": validation_report.measurement_reliability,
                "metacognitive_intact": validation_report.metacognitive_assessment.is_intact,
                "meta_d_prime": validation_report.metacognitive_assessment.meta_d_prime,
            }

            return IntegrityCheckResult(
                check_name="consciousness_measurement_validity",
                passed=passed,
                score=score,
                details=details,
                issues=issues,
                recommendations=validation_report.recommendations,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error in consciousness measurement validation: {e}")
            return IntegrityCheckResult(
                check_name="consciousness_measurement_validity",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                issues=[f"Consciousness measurement validation failed: {e}"],
                recommendations=["Review consciousness measurement data and validation procedures"],
                timestamp=datetime.now(),
            )

    def check_experimental_condition_compliance(
        self, condition: ExperimentalCondition, actual_parameters: Dict[str, Any]
    ) -> IntegrityCheckResult:
        """Check compliance with experimental condition requirements.

        Args:
            condition: Expected experimental condition
            actual_parameters: Actual experimental parameters

        Returns:
            IntegrityCheckResult for condition compliance
        """
        logger.info(f"Checking experimental condition compliance for {condition.condition_id}")

        issues = []
        recommendations = []
        compliance_scores = []

        # Check parameter compliance
        for param_name, expected_value in condition.parameters.items():
            if param_name in actual_parameters:
                actual_value = actual_parameters[param_name]

                # Calculate compliance based on parameter type
                if isinstance(expected_value, (int, float)):
                    # Numerical parameter - check within tolerance
                    tolerance = condition.validation_criteria.get(f"{param_name}_tolerance", 0.1)
                    if abs(actual_value - expected_value) <= tolerance * abs(expected_value):
                        compliance_scores.append(1.0)
                    else:
                        compliance_scores.append(0.0)
                        issues.append(
                            f"Parameter {param_name}: expected {expected_value}, got {actual_value}"
                        )

                elif isinstance(expected_value, str):
                    # String parameter - exact match required
                    if actual_value == expected_value:
                        compliance_scores.append(1.0)
                    else:
                        compliance_scores.append(0.0)
                        issues.append(
                            f"Parameter {param_name}: expected '{expected_value}', got '{actual_value}'"
                        )

                elif isinstance(expected_value, bool):
                    # Boolean parameter - exact match required
                    if actual_value == expected_value:
                        compliance_scores.append(1.0)
                    else:
                        compliance_scores.append(0.0)
                        issues.append(
                            f"Parameter {param_name}: expected {expected_value}, got {actual_value}"
                        )

                else:
                    # Other types - assume compliance if present
                    compliance_scores.append(0.8)
            else:
                issues.append(f"Missing required parameter: {param_name}")
                compliance_scores.append(0.0)
                recommendations.append(f"Ensure parameter {param_name} is properly set")

        # Check control requirements
        for requirement in condition.control_requirements:
            # This would be expanded based on specific control requirement types
            if requirement not in actual_parameters.get("controls_verified", []):
                issues.append(f"Control requirement not met: {requirement}")
                recommendations.append(f"Verify control requirement: {requirement}")

        # Calculate overall compliance score
        overall_score = np.mean(compliance_scores) if compliance_scores else 0.0
        passed = overall_score >= self.config["integrity_thresholds"]["critical_check_threshold"]

        details = {
            "parameters_checked": len(condition.parameters),
            "parameters_compliant": sum(compliance_scores),
            "control_requirements": len(condition.control_requirements),
            "compliance_rate": overall_score,
        }

        return IntegrityCheckResult(
            check_name="experimental_condition_compliance",
            passed=passed,
            score=float(overall_score),
            details=details,
            issues=issues,
            recommendations=recommendations,
            timestamp=datetime.now(),
        )

    def check_data_quality(
        self, participant_id: str, session_data: Dict[str, Any]
    ) -> IntegrityCheckResult:
        """Check overall data quality and completeness.

        Args:
            participant_id: Unique participant identifier
            session_data: Complete session data

        Returns:
            IntegrityCheckResult for data quality
        """
        logger.info(f"Checking data quality for participant {participant_id}")

        issues = []
        recommendations = []
        quality_scores = []

        # Check data completeness
        required_fields = [
            "response_times",
            "accuracies",
            "confidence_ratings",
            "trial_timestamps",
        ]
        for field in required_fields:
            if field in session_data and session_data[field]:
                quality_scores.append(1.0)
            else:
                issues.append(f"Missing or empty required field: {field}")
                quality_scores.append(0.0)
                recommendations.append(f"Ensure {field} data is collected and stored")

        # Check data consistency
        if "response_times" in session_data and "accuracies" in session_data:
            rt_length = len(session_data["response_times"])
            acc_length = len(session_data["accuracies"])

            if rt_length == acc_length:
                quality_scores.append(1.0)
            else:
                issues.append(
                    f"Data length mismatch: {rt_length} response times vs {acc_length} accuracies"
                )
                quality_scores.append(0.5)
                recommendations.append("Verify data collection synchronization")

        # Check for outliers in response times
        if "response_times" in session_data:
            response_times = session_data["response_times"]
            if response_times:
                rt_array = np.array(response_times)
                q1, q3 = np.percentile(rt_array, [25, 75])
                iqr = q3 - q1
                outlier_threshold = 3.0 * iqr

                outliers = np.sum(
                    (rt_array < q1 - outlier_threshold) | (rt_array > q3 + outlier_threshold)
                )
                outlier_rate = outliers / len(rt_array)

                if outlier_rate < 0.05:  # Less than 5% outliers
                    quality_scores.append(1.0)
                elif outlier_rate < 0.1:  # Less than 10% outliers
                    quality_scores.append(0.8)
                    recommendations.append("Consider reviewing outlier trials")
                else:
                    quality_scores.append(0.5)
                    issues.append(f"High outlier rate in response times: {outlier_rate:.1%}")
                    recommendations.append("Investigate causes of response time outliers")

        # Check temporal consistency
        if "trial_timestamps" in session_data:
            timestamps = session_data["trial_timestamps"]
            if len(timestamps) > 1:
                # Check for reasonable inter-trial intervals
                intervals = np.diff(timestamps)
                mean_interval = np.mean(intervals)

                if 1.0 <= mean_interval <= 10.0:  # Reasonable trial intervals
                    quality_scores.append(1.0)
                else:
                    issues.append(f"Unusual trial timing: mean interval {mean_interval:.2f}s")
                    quality_scores.append(0.7)
                    recommendations.append("Review experimental timing parameters")

        # Calculate overall quality score
        overall_score = np.mean(quality_scores) if quality_scores else 0.0
        passed = overall_score >= self.config["integrity_thresholds"]["warning_threshold"]

        details = {
            "fields_checked": len(required_fields),
            "fields_complete": sum(quality_scores[: len(required_fields)]),
            "outlier_checks": len(quality_scores) - len(required_fields),
            "overall_quality": overall_score,
        }

        return IntegrityCheckResult(
            check_name="data_quality_assessment",
            passed=passed,
            score=float(overall_score),
            details=details,
            issues=issues,
            recommendations=recommendations,
            timestamp=datetime.now(),
        )

    def generate_integrity_report(
        self,
        experiment_id: str,
        participant_id: str,
        session_id: str,
        experimental_data: Dict[str, Any],
    ) -> ExperimentalIntegrityReport:
        """Generate comprehensive experimental integrity report.

        Args:
            experiment_id: Unique experiment identifier
            participant_id: Unique participant identifier
            session_id: Unique session identifier
            experimental_data: Complete experimental data

        Returns:
            ExperimentalIntegrityReport with comprehensive integrity assessment
        """
        logger.info(
            f"Generating integrity report for experiment {experiment_id}, participant {participant_id}"
        )

        # Perform all integrity checks
        condition_checks = []
        control_checks = []
        validation_checks = []

        # Response system integrity check
        if "response_data" in experimental_data:
            response_check = self.check_response_system_integrity(
                participant_id, experimental_data["response_data"]
            )
            control_checks.append(response_check)

        # Consciousness measurement validity check
        if "consciousness_data" in experimental_data:
            consciousness_check = self.check_consciousness_measurement_validity(
                participant_id, session_id, experimental_data["consciousness_data"]
            )
            validation_checks.append(consciousness_check)

        # Experimental condition compliance check
        if (
            "experimental_condition" in experimental_data
            and "actual_parameters" in experimental_data
        ):
            condition_check = self.check_experimental_condition_compliance(
                experimental_data["experimental_condition"],
                experimental_data["actual_parameters"],
            )
            condition_checks.append(condition_check)

        # Data quality check
        if "session_data" in experimental_data:
            quality_check = self.check_data_quality(
                participant_id, experimental_data["session_data"]
            )
            validation_checks.append(quality_check)

        # Calculate overall integrity score
        all_checks = condition_checks + control_checks + validation_checks
        weights = self.config["validation_weights"]

        weighted_scores = []
        if condition_checks:
            condition_score = np.mean([check.score for check in condition_checks])
            weighted_scores.append(condition_score * weights["experimental_conditions"])

        if control_checks:
            control_score = np.mean([check.score for check in control_checks])
            weighted_scores.append(control_score * weights["participant_controls"])

        if validation_checks:
            validation_score = np.mean([check.score for check in validation_checks])
            weighted_scores.append(validation_score * weights["consciousness_measures"])

        overall_integrity_score = sum(weighted_scores) if weighted_scores else 0.0

        # Determine if experiment is valid
        critical_checks_passed = all(
            check.passed
            for check in all_checks
            if check.check_name in self.config["critical_checks"]
        )

        sufficient_score = (
            overall_integrity_score >= self.config["integrity_thresholds"]["minimum_overall_score"]
        )
        is_valid_experiment = critical_checks_passed and sufficient_score

        # Collect critical issues and recommendations
        critical_issues = []
        all_recommendations = []

        for check in all_checks:
            if not check.passed and check.check_name in self.config["critical_checks"]:
                critical_issues.extend(check.issues)
            all_recommendations.extend(check.recommendations)

        # Remove duplicate recommendations
        unique_recommendations = list(set(all_recommendations))

        return ExperimentalIntegrityReport(
            experiment_id=experiment_id,
            participant_id=participant_id,
            session_id=session_id,
            condition_checks=condition_checks,
            control_checks=control_checks,
            validation_checks=validation_checks,
            overall_integrity_score=overall_integrity_score,
            is_valid_experiment=is_valid_experiment,
            critical_issues=critical_issues,
            recommendations=unique_recommendations,
            timestamp=datetime.now(),
        )

    def validate_experiment_batch(
        self, experiments: List[Dict[str, Any]]
    ) -> Dict[str, ExperimentalIntegrityReport]:
        """Validate a batch of experiments for integrity.

        Args:
            experiments: List of experiment data dictionaries

        Returns:
            Dictionary mapping experiment IDs to integrity reports
        """
        logger.info(f"Validating batch of {len(experiments)} experiments")

        reports = {}

        for exp_data in experiments:
            try:
                experiment_id = exp_data.get("experiment_id", "unknown")
                participant_id = exp_data.get("participant_id", "unknown")
                session_id = exp_data.get("session_id", "unknown")

                report = self.generate_integrity_report(
                    experiment_id, participant_id, session_id, exp_data
                )

                reports[f"{experiment_id}_{participant_id}_{session_id}"] = report

            except Exception as e:
                logger.error(
                    f"Error validating experiment {exp_data.get('experiment_id', 'unknown')}: {e}"
                )
                # Create error report
                error_report = ExperimentalIntegrityReport(
                    experiment_id=exp_data.get("experiment_id", "unknown"),
                    participant_id=exp_data.get("participant_id", "unknown"),
                    session_id=exp_data.get("session_id", "unknown"),
                    condition_checks=[],
                    control_checks=[],
                    validation_checks=[],
                    overall_integrity_score=0.0,
                    is_valid_experiment=False,
                    critical_issues=[f"Validation error: {e}"],
                    recommendations=["Review experimental data and validation procedures"],
                    timestamp=datetime.now(),
                )
                reports[f"error_{exp_data.get('experiment_id', 'unknown')}"] = error_report

        return reports
