"""Behavioral task implementations for APGI experimental paradigms.

This module provides concrete task implementations for behavioral experiments
including detection tasks, heartbeat detection, and dual-modality oddball tasks.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DetectionTask:
    """Detection task for consciousness threshold estimation.

    Participants detect near-threshold visual or auditory stimuli.
    Used to establish the relationship between objective stimulus intensity
    and subjective detection reports.
    """

    def __init__(
        self,
        task_id: str = "detection_task",
        n_trials: int = 100,
        stimulus_intensity: float = 0.5,
        modality: str = "visual",
        duration_ms: float = 50.0,
        callback: Optional[Callable] = None,
    ):
        """Initialize detection task.

        Args:
            task_id: Unique identifier for this task instance
            n_trials: Number of trials to run
            stimulus_intensity: Intensity of stimulus (0-1, where 0.5 is threshold)
            modality: Sensory modality ('visual', 'auditory', 'tactile')
            duration_ms: Stimulus duration in milliseconds
            callback: Optional callback function for trial updates
        """
        self.task_id = task_id
        self.n_trials = n_trials
        self.stimulus_intensity = stimulus_intensity
        self.modality = modality
        self.duration_ms = duration_ms
        self.callback = callback

        self.current_trial = 0
        self.running = False
        self.results: list[Dict[str, Any]] = []

        logger.info(f"Initialized DetectionTask {task_id} with {n_trials} trials")

    def run(self) -> Dict[str, Any]:
        """Run the detection task.

        Returns:
            Dictionary containing task results
        """
        self.running = True
        self.current_trial = 0
        self.results = []

        logger.info(f"Starting DetectionTask {self.task_id}")

        for trial in range(self.n_trials):
            if not self.running:
                logger.info(f"DetectionTask {self.task_id} stopped at trial {trial}")
                break

            self.current_trial = trial

            # Simulate trial with detection probability based on intensity
            detection_probability = self._calculate_detection_probability()
            detected = np.random.random() < detection_probability
            response_time = self._generate_response_time(detected)

            trial_result = {
                "trial_id": trial,
                "stimulus_intensity": self.stimulus_intensity,
                "detected": detected,
                "response_time_ms": response_time,
                "modality": self.modality,
                "timestamp": datetime.now().isoformat(),
            }

            self.results.append(trial_result)

            if self.callback:
                self.callback(trial_result)

        self.running = False

        # Calculate summary statistics
        detection_rate = np.mean([r["detected"] for r in self.results]) if self.results else 0
        mean_rt = (
            np.mean([r["response_time_ms"] for r in self.results if r["detected"]])
            if any(r["detected"] for r in self.results)
            else 0
        )

        summary = {
            "task_id": self.task_id,
            "task_type": "detection",
            "n_trials_completed": len(self.results),
            "detection_rate": detection_rate,
            "mean_response_time_ms": mean_rt,
            "stimulus_intensity": self.stimulus_intensity,
            "modality": self.modality,
            "trial_data": self.results,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"DetectionTask {self.task_id} completed: detection_rate={detection_rate:.3f}")

        return summary

    def _calculate_detection_probability(self) -> float:
        """Calculate detection probability based on stimulus intensity."""
        # Sigmoid psychometric function
        return float(1.0 / (1.0 + np.exp(-10 * (self.stimulus_intensity - 0.5))))

    def _generate_response_time(self, detected: bool) -> float:
        """Generate response time based on detection."""
        if detected:
            return np.random.normal(400, 100)  # ms, faster when detected
        else:
            return np.random.normal(800, 200)  # ms, slower or no response

    def stop(self) -> None:
        """Stop the task execution."""
        self.running = False
        logger.info(f"DetectionTask {self.task_id} stop requested")


class HeartbeatDetectionTask:
    """Heartbeat detection task for interoceptive awareness assessment.

    Participants count their own heartbeats during specified intervals
    without checking their pulse. Tests interoceptive sensitivity.
    """

    def __init__(
        self,
        task_id: str = "heartbeat_task",
        n_trials: int = 10,
        interval_seconds: float = 30.0,
        feedback: bool = True,
        callback: Optional[Callable] = None,
    ):
        """Initialize heartbeat detection task.

        Args:
            task_id: Unique identifier for this task instance
            n_trials: Number of intervals to count
            interval_seconds: Duration of each counting interval
            feedback: Whether to provide feedback after each trial
            callback: Optional callback function for trial updates
        """
        self.task_id = task_id
        self.n_trials = n_trials
        self.interval_seconds = interval_seconds
        self.feedback = feedback
        self.callback = callback

        self.current_trial = 0
        self.running = False
        self.results: list[Dict[str, Any]] = []

        # Simulated heart rate (BPM)
        self.base_hr = np.random.normal(72, 8)

        logger.info(f"Initialized HeartbeatDetectionTask {task_id} with {n_trials} trials")

    def run(self) -> Dict[str, Any]:
        """Run the heartbeat detection task.

        Returns:
            Dictionary containing task results
        """
        self.running = True
        self.current_trial = 0
        self.results = []

        logger.info(f"Starting HeartbeatDetectionTask {self.task_id}")

        for trial in range(self.n_trials):
            if not self.running:
                logger.info(f"HeartbeatDetectionTask {self.task_id} stopped at trial {trial}")
                break

            self.current_trial = trial

            # Simulate actual heartbeats during interval
            actual_hr = self.base_hr + np.random.normal(0, 5)
            actual_beats = int(actual_hr * self.interval_seconds / 60)

            # Simulated participant count with some error
            interoceptive_accuracy = np.random.beta(3, 2)  # Individual differences
            reported_beats = int(actual_beats * interoceptive_accuracy + np.random.normal(0, 2))
            reported_beats = max(0, reported_beats)

            error = abs(reported_beats - actual_beats)
            accuracy_score = max(0, 1 - error / actual_beats) if actual_beats > 0 else 0

            trial_result = {
                "trial_id": trial,
                "interval_seconds": self.interval_seconds,
                "actual_heart_rate": actual_hr,
                "actual_beats": actual_beats,
                "reported_beats": reported_beats,
                "error": error,
                "accuracy_score": accuracy_score,
                "timestamp": datetime.now().isoformat(),
            }

            self.results.append(trial_result)

            if self.callback:
                self.callback(trial_result)

        self.running = False

        # Calculate summary statistics
        mean_accuracy = np.mean([r["accuracy_score"] for r in self.results]) if self.results else 0

        summary = {
            "task_id": self.task_id,
            "task_type": "heartbeat_detection",
            "n_trials_completed": len(self.results),
            "mean_accuracy": mean_accuracy,
            "interoceptive_sensitivity": mean_accuracy,
            "base_heart_rate": self.base_hr,
            "interval_seconds": self.interval_seconds,
            "trial_data": self.results,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"HeartbeatDetectionTask {self.task_id} completed: accuracy={mean_accuracy:.3f}"
        )

        return summary

    def stop(self) -> None:
        """Stop the task execution."""
        self.running = False
        logger.info(f"HeartbeatDetectionTask {self.task_id} stop requested")


class DualModalityOddballTask:
    """Dual-modality oddball task for exteroceptive-interoceptive integration.

    Combines standard auditory/visual oddball with cardiac-gated stimulus
    presentation to test integration of exteroceptive and interoceptive signals.
    """

    def __init__(
        self,
        task_id: str = "oddball_task",
        n_trials: int = 200,
        oddball_probability: float = 0.2,
        cardiac_gating: bool = True,
        modality: str = "auditory",
        callback: Optional[Callable] = None,
    ):
        """Initialize dual-modality oddball task.

        Args:
            task_id: Unique identifier for this task instance
            n_trials: Number of trials to run
            oddball_probability: Probability of oddball stimulus (0-1)
            cardiac_gating: Whether to gate stimuli by cardiac phase
            modality: Primary sensory modality ('auditory', 'visual', 'both')
            callback: Optional callback function for trial updates
        """
        self.task_id = task_id
        self.n_trials = n_trials
        self.oddball_probability = oddball_probability
        self.cardiac_gating = cardiac_gating
        self.modality = modality
        self.callback = callback

        self.current_trial = 0
        self.running = False
        self.results: list[Dict[str, Any]] = []

        # Simulated heart rate for cardiac gating
        self.hr_ms = 800  # ~75 BPM

        logger.info(f"Initialized DualModalityOddballTask {task_id} with {n_trials} trials")

    def run(self) -> Dict[str, Any]:
        """Run the dual-modality oddball task.

        Returns:
            Dictionary containing task results
        """
        self.running = True
        self.current_trial = 0
        self.results = []

        logger.info(f"Starting DualModalityOddballTask {self.task_id}")

        # Generate trial sequence
        trial_types = np.random.choice(
            ["standard", "oddball"],
            size=self.n_trials,
            p=[1 - self.oddball_probability, self.oddball_probability],
        )

        # Cardiac phases (if gating enabled)
        if self.cardiac_gating:
            cardiac_phases = np.random.choice(
                ["systole", "diastole"], size=self.n_trials, p=[0.5, 0.5]
            )
        else:
            cardiac_phases = np.array(["none"] * self.n_trials)

        for trial in range(self.n_trials):
            if not self.running:
                logger.info(f"DualModalityOddballTask {self.task_id} stopped at trial {trial}")
                break

            self.current_trial = trial
            trial_type = trial_types[trial]
            cardiac_phase = cardiac_phases[trial]

            # Simulate response with oddball and cardiac effects
            response_data = self._generate_trial_response(trial_type, cardiac_phase)

            trial_result = {
                "trial_id": trial,
                "trial_type": trial_type,
                "cardiac_phase": cardiac_phase,
                "cardiac_gating": self.cardiac_gating,
                "modality": self.modality,
                "response_time_ms": response_data["rt"],
                "correct": response_data["correct"],
                "p3b_amplitude": response_data["p3b"],
                "gamma_synchrony": response_data["gamma"],
                "timestamp": datetime.now().isoformat(),
            }

            self.results.append(trial_result)

            if self.callback:
                self.callback(trial_result)

        self.running = False

        # Calculate summary statistics
        oddball_trials = [r for r in self.results if r["trial_type"] == "oddball"]
        standard_trials = [r for r in self.results if r["trial_type"] == "standard"]

        oddball_accuracy = np.mean([r["correct"] for r in oddball_trials]) if oddball_trials else 0
        standard_accuracy = (
            np.mean([r["correct"] for r in standard_trials]) if standard_trials else 0
        )

        # Calculate P3b difference (oddball - standard)
        mean_p3b_oddball = (
            np.mean([r["p3b_amplitude"] for r in oddball_trials]) if oddball_trials else 0
        )
        mean_p3b_standard = (
            np.mean([r["p3b_amplitude"] for r in standard_trials]) if standard_trials else 0
        )
        p3b_difference = mean_p3b_oddball - mean_p3b_standard

        summary = {
            "task_id": self.task_id,
            "task_type": "dual_modality_oddball",
            "n_trials_completed": len(self.results),
            "n_oddball_trials": len(oddball_trials),
            "n_standard_trials": len(standard_trials),
            "oddball_accuracy": oddball_accuracy,
            "standard_accuracy": standard_accuracy,
            "accuracy_difference": oddball_accuracy - standard_accuracy,
            "p3b_difference": p3b_difference,
            "cardiac_gating": self.cardiac_gating,
            "modality": self.modality,
            "trial_data": self.results,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"DualModalityOddballTask {self.task_id} completed: oddball_acc={oddball_accuracy:.3f}, p3b_diff={p3b_difference:.3f}"
        )

        return summary

    def _generate_trial_response(self, trial_type: str, cardiac_phase: str) -> Dict[str, Any]:
        """Generate response data for a single trial."""
        is_oddball = trial_type == "oddball"

        # Response time faster for oddballs
        if is_oddball:
            rt = np.random.normal(350, 80)
        else:
            rt = np.random.normal(400, 100)

        rt = max(200, rt)

        # Accuracy higher for oddballs
        if is_oddball:
            correct = np.random.random() < 0.95
        else:
            correct = np.random.random() < 0.90

        # P3b amplitude larger for oddballs
        if is_oddball:
            base_p3b = np.random.normal(12, 3)
        else:
            base_p3b = np.random.normal(5, 2)

        # Cardiac phase modulation (systole = reduced P3b)
        if cardiac_phase == "systole":
            p3b = base_p3b * 0.85
        else:
            p3b = base_p3b

        # Gamma synchrony
        if is_oddball:
            gamma = np.random.normal(0.45, 0.1)
        else:
            gamma = np.random.normal(0.25, 0.08)

        gamma = np.clip(gamma, 0, 1)

        return {"rt": rt, "correct": correct, "p3b": p3b, "gamma": gamma}

    def stop(self) -> None:
        """Stop the task execution."""
        self.running = False
        logger.info(f"DualModalityOddballTask {self.task_id} stop requested")


# Export all task classes
__all__ = ["DetectionTask", "HeartbeatDetectionTask", "DualModalityOddballTask"]
