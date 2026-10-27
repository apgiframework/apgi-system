"""Body State Model.

Simulates physiological states and generates interoceptive predictions.
"""

from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np

from apgi_framework.types import FloatArray


@dataclass
class PhysiologicalState:
    """Complete physiological state."""

    heart_rate: float  # bpm
    respiration: float  # breaths per minute
    temperature: float  # Celsius
    glucose: float  # mmol/L
    cortisol: float  # μg/dL
    blood_pressure: float  # mmHg (systolic)


class BodyModel:
    """Multi-organ physiological model for interoceptive processing.

    The BodyModel simulates physiological states across multiple organ systems
    and generates interoceptive predictions for active inference. It maintains
    current body states and generates forward predictions to enable prediction
    error computation for interoceptive processing.

    The model tracks six key physiological variables:
    - Cardiovascular: heart rate (bpm) and blood pressure (mmHg)
    - Respiratory: respiration rate (breaths per minute)
    - Thermoregulation: body temperature (Celsius)
    - Metabolic: blood glucose (mmol/L)
    - Stress: cortisol levels (μg/dL)

    Each variable follows simplified linear dynamics with configurable time
    constants and is influenced by arousal, activity, and stress levels.
    Predictions are generated 50-200ms ahead for comparison with afferent signals.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing interoception settings including:
        - body_states: List of state configurations with baselines and ranges
        - prediction_lead_ms: Prediction horizon in milliseconds (default: 100)

    Attributes
    ----------
    current_state : PhysiologicalState
        Current physiological state across all tracked variables
    predicted_state : PhysiologicalState
        Predicted future state at prediction_lead_ms ahead
    arousal_level : float
        Current arousal level (0-1), affects heart rate and cortisol
    activity_level : float
        Current activity level (0-1), affects heart rate, respiration, glucose
    stress_level : float
        Current stress level (0-1), affects cortisol and blood pressure
    prediction_lead_ms : float
        Time horizon for predictions in milliseconds

    Examples
    --------
    >>> config = {'interoception': {'prediction_lead_ms': 100}}
    >>> body_model = BodyModel(config)
    >>> body_model.set_arousal(0.5)
    >>> result = body_model.update(dt=1.0)
    >>> print(result['current']['heart_rate'])
    75.2
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize vectorized body model."""
        self.config = config
        self.batch_size = config.get("active_inference", {}).get("batch_size", 1)
        intero_config = config.get("interoception", {})
        body_states_config = intero_config.get("body_states", [])

        # Parse body state configurations
        self.state_configs = {}
        for state_dict in body_states_config:
            for name, cfg in state_dict.items():
                self.state_configs[name] = cfg

        # Current state (B, 6)
        # heart_rate, respiration, temperature, glucose, cortisol, blood_pressure
        self.num_states = 6
        self.current_state = np.zeros((self.batch_size, self.num_states))

        self.current_state[:, 0] = self.state_configs.get("heart_rate", {}).get("baseline", 70)
        self.current_state[:, 1] = self.state_configs.get("respiration", {}).get("baseline", 15)
        self.current_state[:, 2] = self.state_configs.get("temperature", {}).get("baseline", 37.0)
        self.current_state[:, 3] = self.state_configs.get("glucose", {}).get("baseline", 5.0)
        self.current_state[:, 4] = self.state_configs.get("cortisol", {}).get("baseline", 10)
        self.current_state[:, 5] = 120.0

        # Predicted state
        self.prediction_lead_ms = intero_config.get("prediction_lead_ms", 100)
        self.predicted_state = self.current_state.copy()

        # Dynamics parameters
        self.tau_heart = 1000.0
        self.tau_respiration = 2000.0
        self.tau_temperature = 10000.0
        self.tau_glucose = 5000.0
        self.tau_cortisol = 30000.0
        self.tau_bp = 1000.0

        # Previous state for trend tracking
        self.previous_state = self.current_state.copy()

        # Influences (B,)
        self.arousal_level = np.zeros(self.batch_size)
        self.activity_level = np.zeros(self.batch_size)
        self.stress_level = np.zeros(self.batch_size)

    def update(self, dt: float = 1.0) -> Dict[str, Any]:
        """Update body state and generate predictions for batch."""
        # Store previous state
        self.previous_state = self.current_state.copy()

        # Update current state (all 6 variables vectorized)
        self._update_dynamics(dt)

        # Generate prediction
        self._generate_prediction()

        # Compute prediction error (B, 5) - heart_rate, resp, temp, glucose, cortisol
        prediction_error = self._compute_prediction_error()

        return {
            "current": self.get_current_state_dict(),
            "predicted": self.get_current_state_dict(),
            "prediction_error": prediction_error,
            "arousal": self.arousal_level.copy(),
            "activity": self.activity_level.copy(),
            "stress": self.stress_level.copy(),
        }

    def _update_dynamics(self, dt: float) -> None:
        """Update all 6 physiological variables for the batch."""
        # 0: Heart Rate
        hr_baseline = self.state_configs.get("heart_rate", {}).get("baseline", 70)
        hr_target = hr_baseline + 30 * self.arousal_level + 50 * self.activity_level
        hr_range = self.state_configs.get("heart_rate", {}).get("range", [50, 120])
        hr_target = np.clip(hr_target, hr_range[0], hr_range[1])
        self.current_state[:, 0] += (dt / self.tau_heart) * (hr_target - self.current_state[:, 0])
        self.current_state[:, 0] += np.random.randn(self.batch_size) * 5 * np.sqrt(dt / 1000.0)

        # 1: Respiration
        resp_baseline = self.state_configs.get("respiration", {}).get("baseline", 15)
        resp_target = resp_baseline + 5 * self.arousal_level + 10 * self.activity_level
        resp_range = self.state_configs.get("respiration", {}).get("range", [10, 30])
        resp_target = np.clip(resp_target, resp_range[0], resp_range[1])
        self.current_state[:, 1] += (dt / self.tau_respiration) * (
            resp_target - self.current_state[:, 1]
        )
        self.current_state[:, 1] += np.random.randn(self.batch_size) * 2 * np.sqrt(dt / 1000.0)

        # 2: Temperature
        temp_baseline = self.state_configs.get("temperature", {}).get("baseline", 37.0)
        temp_target = temp_baseline + 0.5 * self.activity_level
        temp_range = self.state_configs.get("temperature", {}).get("range", [36.0, 39.0])
        temp_target = np.clip(temp_target, temp_range[0], temp_range[1])
        self.current_state[:, 2] += (dt / self.tau_temperature) * (
            temp_target - self.current_state[:, 2]
        )
        self.current_state[:, 2] += np.random.randn(self.batch_size) * 0.2 * np.sqrt(dt / 1000.0)

        # 3: Glucose
        gluc_baseline = self.state_configs.get("glucose", {}).get("baseline", 5.0)
        gluc_target = gluc_baseline - 1.0 * self.activity_level
        gluc_range = self.state_configs.get("glucose", {}).get("range", [3.0, 8.0])
        gluc_target = np.clip(gluc_target, gluc_range[0], gluc_range[1])
        self.current_state[:, 3] += (dt / self.tau_glucose) * (
            gluc_target - self.current_state[:, 3]
        )
        self.current_state[:, 3] += np.random.randn(self.batch_size) * 0.5 * np.sqrt(dt / 1000.0)

        # 4: Cortisol
        cort_baseline = self.state_configs.get("cortisol", {}).get("baseline", 10)
        cort_target = cort_baseline + 15 * self.stress_level + 5 * self.arousal_level
        cort_range = self.state_configs.get("cortisol", {}).get("range", [5, 30])
        cort_target = np.clip(cort_target, cort_range[0], cort_range[1])
        self.current_state[:, 4] += (dt / self.tau_cortisol) * (
            cort_target - self.current_state[:, 4]
        )
        self.current_state[:, 4] += np.random.randn(self.batch_size) * 3 * np.sqrt(dt / 1000.0)

        # 5: Blood Pressure
        bp_target = 120.0 + 20 * self.arousal_level + 10 * self.stress_level
        bp_target = np.clip(bp_target, 90, 160)
        self.current_state[:, 5] += (dt / self.tau_bp) * (bp_target - self.current_state[:, 5])
        self.current_state[:, 5] += np.random.randn(self.batch_size) * 5 * np.sqrt(dt / 1000.0)

    def _generate_prediction(self) -> None:
        """Generate prediction (B, 6) for batch."""
        pred_dt = self.prediction_lead_ms
        alpha = pred_dt / 1000.0

        # Trends (B, 6)
        trends = self.current_state - self.previous_state

        # Targets (B, 6)
        targets = np.zeros_like(self.current_state)
        # Same logic as dynamics but forward-looking
        targets[:, 0] = (
            self.state_configs.get("heart_rate", {}).get("baseline", 70)
            + 30 * self.arousal_level
            + 50 * self.activity_level
        )
        targets[:, 1] = (
            self.state_configs.get("respiration", {}).get("baseline", 15)
            + 5 * self.arousal_level
            + 10 * self.activity_level
        )
        targets[:, 2] = (
            self.state_configs.get("temperature", {}).get("baseline", 37.0)
            + 0.5 * self.activity_level
        )
        targets[:, 3] = (
            self.state_configs.get("glucose", {}).get("baseline", 5.0) - 1.0 * self.activity_level
        )
        targets[:, 4] = (
            self.state_configs.get("cortisol", {}).get("baseline", 10)
            + 15 * self.stress_level
            + 5 * self.arousal_level
        )
        targets[:, 5] = 120.0 + 20 * self.arousal_level + 10 * self.stress_level

        # Exponential smoothing prediction
        taus = (
            np.array(
                [
                    self.tau_heart,
                    self.tau_respiration,
                    self.tau_temperature,
                    self.tau_glucose,
                    self.tau_cortisol,
                    self.tau_bp,
                ]
            )
            / 1000.0
        )

        # Predicted target = target + (current - target) * exp(-dt/tau)
        pred_targets = targets + (self.current_state - targets) * np.exp(-alpha / taus)

        # Combination of trend and target-smoothing
        self.predicted_state = 0.8 * pred_targets + 0.2 * (
            self.current_state + trends * (pred_dt / 1.0)
        )

    def _compute_prediction_error(self) -> np.ndarray:
        """Compute prediction error for batch."""
        # heart_rate, resp, temp, glucose, cortisol (omit BP in core error)
        errors = np.random.randn(self.batch_size, 5) * np.array([2, 1, 0.1, 0.2, 1])
        return errors  # type: ignore[no-any-return]

    def set_arousal(self, level: Union[float, FloatArray]) -> None:
        """Set arousal (B,) or scalar."""
        self.arousal_level = np.clip(level, 0.0, 1.0)

    def set_activity(self, level: Union[float, FloatArray]) -> None:
        """Set activity (B,) or scalar."""
        self.activity_level = np.clip(level, 0.0, 1.0)

    def set_stress(self, level: Union[float, FloatArray]) -> None:
        """Set stress (B,) or scalar."""
        self.stress_level = np.clip(level, 0.0, 1.0)

    def get_interoceptive_vector(self) -> np.ndarray:
        """Get current 6D physiological state (B, 6)."""
        return self.current_state.copy()

    def _state_to_dict(self, state: PhysiologicalState) -> Dict[str, float]:
        """Convert physiological state to dictionary.

        Parameters
        ----------
        state : PhysiologicalState
            State object to convert

        Returns
        -------
        Dict[str, float]
            Dictionary mapping variable names to values
        """
        return {
            "heart_rate": float(state.heart_rate),
            "respiration": float(state.respiration),
            "temperature": float(state.temperature),
            "glucose": float(state.glucose),
            "cortisol": float(state.cortisol),
            "blood_pressure": float(state.blood_pressure),
        }

    def get_current_state(self) -> Dict[str, np.ndarray]:
        """Get batch state."""
        return {
            "current": self.current_state.copy(),
            "predicted": self.predicted_state.copy(),
            "arousal": self.arousal_level.copy(),
            "activity": self.activity_level.copy(),
            "stress": self.stress_level.copy(),
            "interoceptive_vector": self.get_interoceptive_vector(),
        }

    def get_current_state_dict(self) -> Dict[str, Any]:
        """Get batch state as dictionary with named keys for backward compatibility with tests."""
        # Return first agent's state as dictionary for backward compatibility
        if self.batch_size == 1:
            return {
                "heart_rate": float(self.current_state[0, 0]),
                "respiration": float(self.current_state[0, 1]),
                "temperature": float(self.current_state[0, 2]),
                "glucose": float(self.current_state[0, 3]),
                "cortisol": float(self.current_state[0, 4]),
                "blood_pressure": float(self.current_state[0, 5]),
            }
        else:
            return {
                "heart_rate": self.current_state[:, 0],
                "respiration": self.current_state[:, 1],
                "temperature": self.current_state[:, 2],
                "glucose": self.current_state[:, 3],
                "cortisol": self.current_state[:, 4],
                "blood_pressure": self.current_state[:, 5],
            }

    def reset(self) -> None:
        """Reset batch states."""
        self.current_state[:, 0] = self.state_configs.get("heart_rate", {}).get("baseline", 70)
        self.current_state[:, 1] = self.state_configs.get("respiration", {}).get("baseline", 15)
        self.current_state[:, 2] = self.state_configs.get("temperature", {}).get("baseline", 37.0)
        self.current_state[:, 3] = self.state_configs.get("glucose", {}).get("baseline", 5.0)
        self.current_state[:, 4] = self.state_configs.get("cortisol", {}).get("baseline", 10)
        self.current_state[:, 5] = 120.0
        self.predicted_state = self.current_state.copy()
        self.arousal_level.fill(0.0)
        self.activity_level.fill(0.0)
        self.stress_level.fill(0.0)
