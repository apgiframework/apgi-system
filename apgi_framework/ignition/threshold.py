"""Vectorized Ignition Threshold Computation."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..types import ConfigDict, FloatArray


class IgnitionThreshold:
    """Vectorized ignition threshold for multi-agent conscious access."""

    def __init__(self, config: ConfigDict, rng: Optional[np.random.Generator] = None) -> None:
        """Initialize vectorized ignition threshold."""
        self.config = config
        self.batch_size = config.get("active_inference", {}).get("batch_size", 1)
        ignition_config = config.get("ignition", {})

        # Threshold parameters
        self.baseline_threshold = ignition_config.get("baseline_threshold", 2.0)
        self.threshold_range = ignition_config.get("threshold_range", [1.0, 5.0])
        self.sigmoid_alpha = ignition_config.get("sigmoid_alpha", 5.0)

        # Current state (B,)
        self.current_threshold = np.full(self.batch_size, self.baseline_threshold)
        self.current_signal = np.zeros(self.batch_size)
        self.ignition_probability = np.zeros(self.batch_size)

        # Components (B,)
        self.extero_signal = np.zeros(self.batch_size)
        self.intero_signal = np.zeros(self.batch_size)
        self.accumulated_signal = np.zeros(self.batch_size)

        self.tau_S = ignition_config.get("tau_S_ms", 500.0)
        self.sigma_S = ignition_config.get("sigma_S", 0.05)

        self._last_signal_update_time: Optional[float] = None
        self.rng = rng if rng is not None else np.random.default_rng()

        # Recent ignition history (B,)
        self.refractory_period_ms = ignition_config.get("refractory_period_ms", 200)
        self.last_ignition_time = np.full(self.batch_size, -np.inf)

        # Metabolic tracking (B,)
        self.metabolic_reserves = np.ones(self.batch_size)
        self.allostatic_load = np.zeros(self.batch_size)

        # History tracking (for stats)
        self.signal_history: List[np.ndarray] = []
        self.threshold_history: List[float] = []
        self.recent_ignitions: List[float] = []

    def compute_ignition_signal(
        self,
        extero_error: FloatArray,
        extero_precision: np.ndarray,
        intero_error: FloatArray,
        intero_precision: np.ndarray,
        somatic_marker_gain: np.ndarray,
        current_time: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Compute ignition for batch (B,)."""
        # Validate types
        if not isinstance(extero_error, np.ndarray):
            raise TypeError(f"extero_error must be np.ndarray, got {type(extero_error)}")
        if not isinstance(intero_error, np.ndarray):
            raise TypeError(f"intero_error must be np.ndarray, got {type(intero_error)}")
        if not isinstance(extero_precision, np.ndarray):
            raise TypeError(f"extero_precision must be np.ndarray, got {type(extero_precision)}")
        if not isinstance(intero_precision, np.ndarray):
            raise TypeError(f"intero_precision must be np.ndarray, got {type(intero_precision)}")
        if not isinstance(somatic_marker_gain, np.ndarray):
            raise TypeError(
                f"somatic_marker_gain must be np.ndarray, got {type(somatic_marker_gain)}"
            )

        # Validate ranges
        if np.any(extero_precision < 0):
            raise ValueError("extero_precision must be non-negative")
        if np.any(somatic_marker_gain < 0.5) or np.any(somatic_marker_gain > 2.0):
            raise ValueError("somatic_marker_gain must be in range [0.5, 2.0]")
        # extero_error: (B, D_extero), intero_error: (B, D_intero)
        # Note: input arrays might be 1D if batch_size=1, ensure 2D
        if extero_error.ndim == 1:
            extero_error = extero_error[np.newaxis, :]
        if intero_error.ndim == 1:
            intero_error = intero_error[np.newaxis, :]

        extero_scalar = np.mean(extero_error**2, axis=1)
        intero_scalar = np.mean(intero_error**2, axis=1)

        self.extero_signal = 0.5 * extero_precision * extero_scalar
        self.intero_signal = 0.5 * intero_precision * somatic_marker_gain * intero_scalar

        # Leaky integration
        input_drive = self.extero_signal + self.intero_signal
        dt = (
            0.0
            if self._last_signal_update_time is None
            else max(0.0, current_time - self._last_signal_update_time)
        )
        self._last_signal_update_time = current_time
        dt_sec = dt / 1000.0
        tau_s_sec = max(self.tau_S / 1000.0, 1e-6)

        decay = -self.accumulated_signal / tau_s_sec
        noise = self.sigma_S * np.sqrt(max(dt_sec, 0.0)) * self.rng.normal(size=self.batch_size)

        self.accumulated_signal = np.maximum(
            0.0, self.accumulated_signal + (decay + input_drive) * dt_sec + noise
        )
        self.current_signal = self.accumulated_signal

        # Track history for stats
        self.signal_history.append(self.current_signal.copy())
        if len(self.signal_history) > 1000:
            self.signal_history.pop(0)

        # Update dynamic threshold (vectorized)
        self._update_threshold(current_time)

        # Stochastic ignition probability
        delta = self.current_signal - self.current_threshold
        self.ignition_probability = 1.0 / (1.0 + np.exp(-self.sigmoid_alpha * delta))

        # Check refractory period
        refractory_mask = (current_time - self.last_ignition_time) < self.refractory_period_ms
        self.ignition_probability[refractory_mask] = 0.0

        # Stochastic choice
        ignited = self.rng.random(size=self.batch_size) < self.ignition_probability
        if np.any(ignited):
            self.last_ignition_time[ignited] = current_time
            self.recent_ignitions.append(current_time)
            if len(self.recent_ignitions) > 100:
                self.recent_ignitions.pop(0)

        return ignited, {
            "total_signal": self.current_signal.copy(),
            "threshold": self.current_threshold.copy(),
            "probability": self.ignition_probability.copy(),
            "extero_signal": self.extero_signal.copy(),
            "intero_signal": self.intero_signal.copy(),
        }

    def _update_threshold(self, current_time: float) -> None:
        """Update dynamic threshold based on metabolic and allostatic state (B,).

        Implements allostatic update: θ_{t+1} = θ_t + η(C_metabolic - V_information)
        where:
            C_metabolic: cost of maintaining current processing level (inverse of reserves)
            V_information: value of information (proportional to recent signal activity)
            η: learning rate for threshold adaptation
        """
        # Metabolic cost (C_metabolic) - increases as reserves deplete
        c_metabolic = 2.0 * (1.0 - self.metabolic_reserves)

        # Information value (V_information) - derived from recent signal history
        if self.signal_history:
            recent_mean_signal = float(np.mean(self.signal_history[-10:]))
            v_information = 0.3 * recent_mean_signal  # Information value scales with signal
        else:
            v_information = 0.0

        # Allostatic load penalty
        allostatic_penalty = 1.5 * self.allostatic_load

        # Allostatic update: θ_{t+1} = θ_t + η(C_metabolic - V_information + allostatic_penalty)
        eta = 0.1  # Learning rate for threshold adaptation
        delta_theta = eta * (c_metabolic - v_information + allostatic_penalty)

        threshold = self.current_threshold + delta_theta

        # Apply bounds
        self.current_threshold = np.clip(
            threshold, self.threshold_range[0], self.threshold_range[1]
        )

    def update_metabolic_state(self, reserves: np.ndarray, allostatic_load: np.ndarray) -> None:
        """Update metabolic and allostatic state for threshold modulation."""
        self.metabolic_reserves = np.clip(reserves, 0.0, 1.0)
        self.allostatic_load = np.clip(allostatic_load, 0.0, 1.0)

    def get_statistics(self) -> Dict[str, float]:
        """Compute ignition statistics across history."""
        if not self.signal_history:
            return {
                "mean_signal": 0.0,
                "recent_ignitions": 0,
                "ignition_rate": 0.0,
            }

        all_signals = np.stack(self.signal_history)
        return {
            "mean_signal": float(np.mean(all_signals)),
            "std_signal": float(np.std(all_signals)),
            "mean_threshold": float(np.mean(self.current_threshold)),
            "recent_ignitions": len(self.recent_ignitions),
            "ignition_rate": len(self.recent_ignitions) / 10.0,  # Approximate rate
        }

    def reset(self) -> None:
        """Reset terminal states for batch."""
        self.current_threshold[:] = self.baseline_threshold
        self.current_signal.fill(0.0)
        self.accumulated_signal.fill(0.0)
        self.last_ignition_time.fill(-np.inf)
        self.metabolic_reserves.fill(1.0)
        self.allostatic_load.fill(0.0)
        self._last_signal_update_time = None
        self.signal_history = []
        self.recent_ignitions = []
