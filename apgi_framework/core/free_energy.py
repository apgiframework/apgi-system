"""Stub module for free energy calculations.

This module provides type stubs for free energy calculation functionality
that is being migrated from the old apgi_simulation structure.
"""

from typing import Any, Dict, Optional

import numpy as np


def compute_accuracy(
    observation: np.ndarray,
    prediction: np.ndarray,
    precision: np.ndarray | float | None = None,
) -> float:
    """Compute accuracy of predictions."""
    if precision is None:
        precision = 1.0

    if isinstance(precision, (int, float)):
        precision = np.full_like(observation, float(precision))

    pred_error = observation - prediction
    if isinstance(precision, np.ndarray):
        weighted_error = np.sum(precision * pred_error**2)
    else:
        weighted_error = precision * np.sum(pred_error**2)

    return float(weighted_error)


def compute_complexity(
    posterior_mean: np.ndarray,
    posterior_cov: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
) -> float:
    """Compute complexity penalty using KL divergence."""
    calc = FreeEnergyCalculator()
    return calc._kl_divergence_gaussian(posterior_mean, posterior_cov, prior_mean, prior_cov)


def compute_epistemic_value(
    policy: np.ndarray,
    predicted_states: np.ndarray,
    uncertainty: np.ndarray,
    horizon: int = 1,
) -> float:
    """Compute epistemic value from information gain."""
    return float(-np.mean(uncertainty))


def compute_expected_free_energy(
    policy: np.ndarray,
    predicted_states: list[np.ndarray],
    predicted_observations: np.ndarray,
    preferences: np.ndarray,
    state_uncertainty: np.ndarray,
    horizon: int = 1,
) -> tuple[float, dict[str, float]]:
    """Compute expected free energy."""
    calc = FreeEnergyCalculator()
    return calc.compute_expected_free_energy(
        policy,
        predicted_states,
        predicted_observations,
        preferences,
        state_uncertainty,
        horizon,
    )


def compute_pragmatic_value(
    predicted_observations: np.ndarray,
    preferences: np.ndarray,
    horizon: int = 1,
) -> float:
    """Compute pragmatic value from utility."""
    return float(np.mean(predicted_observations * preferences))


def compute_variational_free_energy(
    observation: np.ndarray,
    prediction: np.ndarray,
    precision: float | np.ndarray,
    posterior_mean: np.ndarray,
    posterior_cov: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Compute variational free energy.

    Parameters
    ----------
    observation : np.ndarray
        Observed values
    prediction : np.ndarray
        Predicted values
    precision : float | np.ndarray
        Precision values
    posterior_mean : np.ndarray
        Posterior mean
    posterior_cov : np.ndarray
        Posterior covariance
    prior_mean : np.ndarray
        Prior mean
    prior_cov : np.ndarray
        Prior covariance

    Returns
    -------
    tuple[float, dict[str, float]]
        Free energy value and components dictionary
    """
    calc = FreeEnergyCalculator()
    return calc.compute_variational_free_energy(
        observation,
        prediction,
        precision,
        posterior_mean,
        posterior_cov,
        prior_mean,
        prior_cov,
    )


class FreeEnergyCalculator:
    """Calculator for free energy in predictive processing models."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize free energy calculator.

        Parameters
        ----------
        config : dict[str, Any], optional
            Configuration dictionary, by default None
        """
        self.config = config or {}
        self.eps = self.config.get("numerical_tolerance", 1e-10)
        self.eps = self.config.get("eps", self.eps)
        self.precision_min = self.config.get("precision_min", 1e-6)
        self.precision_max = self.config.get("precision_max", 1e6)
        self.precision_exteroceptive = self.config.get("precision_exteroceptive", 1.0)
        self.precision_interoceptive = self.config.get("precision_interoceptive", 1.0)
        self.prediction_error_weight = self.config.get("prediction_error_weight", 1.0)

    def calculate_free_energy(
        self,
        prediction_error: np.ndarray,
        prior_belief: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate free energy given prediction error.

        Parameters
        ----------
        prediction_error : np.ndarray
            Array of prediction errors
        prior_belief : np.ndarray, optional
            Prior belief states, by default None

        Returns
        -------
        float
            Calculated free energy value
        """
        # Simplified free energy calculation
        if prior_belief is not None:
            error = np.sum(prediction_error**2) * self.precision_exteroceptive
            prior_term = np.sum((prior_belief - 0) ** 2) * self.precision_interoceptive
            return float(error + prior_term)
        return float(np.sum(prediction_error**2) * self.precision_exteroceptive)

    def _kl_divergence_gaussian(
        self, mu_q: np.ndarray, sigma_q: np.ndarray, mu_p: np.ndarray, sigma_p: np.ndarray
    ) -> float:
        """Compute KL divergence between two Gaussian distributions.

        Parameters
        ----------
        mu_q : np.ndarray
            Mean of distribution q
        sigma_q : np.ndarray
            Covariance of distribution q
        mu_p : np.ndarray
            Mean of distribution p
        sigma_p : np.ndarray
            Covariance of distribution p

        Returns
        -------
        float
            KL divergence D_KL(q || p)
        """
        dim = len(mu_q)

        # Handle singular covariance matrices
        try:
            inv_sigma_p = np.linalg.inv(sigma_p + self.eps * np.eye(dim))
            det_p = np.linalg.det(sigma_p + self.eps * np.eye(dim))
            det_q = np.linalg.det(sigma_q + self.eps * np.eye(dim))
        except np.linalg.LinAlgError:
            # Return large value for singular matrices
            return 1e10

        # KL divergence formula for Gaussians
        trace_term = np.trace(inv_sigma_p @ sigma_q)
        diff = mu_p - mu_q
        quad_term = diff.T @ inv_sigma_p @ diff
        log_term = np.log(det_p / (det_q + self.eps) + self.eps)

        kl = 0.5 * (trace_term + quad_term - dim + log_term)
        return float(max(0.0, kl))

    def compute_accuracy(
        self,
        observation: np.ndarray,
        prediction: np.ndarray,
        precision: np.ndarray | float | None = None,
    ) -> float:
        """Compute accuracy of predictions.

        Parameters
        ----------
        observation : np.ndarray
            Observed values
        prediction : np.ndarray
            Predicted values
        precision : np.ndarray | float, optional
            Precision weights, by default None (uses 1.0)

        Returns
        -------
        float
            Accuracy value
        """
        if precision is None:
            precision = 1.0

        if isinstance(precision, (int, float)):
            precision = np.full_like(observation, float(precision))

        # Compute weighted prediction error
        pred_error = observation - prediction
        if isinstance(precision, np.ndarray):
            weighted_error = np.sum(precision * pred_error**2)
        else:
            weighted_error = precision * np.sum(pred_error**2)

        return float(weighted_error)

    def compute_complexity(
        self,
        posterior_mean: np.ndarray,
        posterior_cov: np.ndarray,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
    ) -> float:
        """Compute complexity penalty using KL divergence.

        Parameters
        ----------
        posterior_mean : np.ndarray
            Posterior mean
        posterior_cov : np.ndarray
            Posterior covariance
        prior_mean : np.ndarray
            Prior mean
        prior_cov : np.ndarray
            Prior covariance

        Returns
        -------
        float
            Complexity value (KL divergence)
        """
        return self._kl_divergence_gaussian(posterior_mean, posterior_cov, prior_mean, prior_cov)

    def compute_epistemic_value(
        self,
        policy: np.ndarray,
        predicted_states: np.ndarray,
        uncertainty: np.ndarray,
        horizon: int = 1,
    ) -> float:
        """Compute epistemic value from information gain.

        Parameters
        ----------
        policy : np.ndarray
            Policy vector
        predicted_states : np.ndarray
            Predicted states
        uncertainty : np.ndarray
            State uncertainty values
        horizon : int, optional
            Planning horizon, by default 1

        Returns
        -------
        float
            Epistemic value
        """
        return float(-np.mean(uncertainty))

    def compute_pragmatic_value(
        self,
        predicted_observations: np.ndarray,
        preferences: np.ndarray,
        horizon: int = 1,
    ) -> float:
        """Compute pragmatic value from utility.

        Parameters
        ----------
        predicted_observations : np.ndarray
            Predicted observations
        preferences : np.ndarray
            Preference values
        horizon : int, optional
            Planning horizon, by default 1

        Returns
        -------
        float
            Pragmatic value
        """
        return float(np.mean(predicted_observations * preferences))

    def compute_prediction_error(
        self,
        observation: np.ndarray,
        prediction: np.ndarray,
        precision: np.ndarray | float | None = None,
    ) -> dict[str, float]:
        """Compute prediction error metrics.

        Parameters
        ----------
        observation : np.ndarray
            Observed values
        prediction : np.ndarray
            Predicted values
        precision : np.ndarray | float, optional
            Precision weights, by default None (uses 1.0)

        Returns
        -------
        dict[str, float]
            Dictionary with error metrics
        """
        if precision is None:
            precision = 1.0

        if isinstance(precision, (int, float)):
            precision = np.full_like(observation, float(precision))

        raw_error = np.linalg.norm(observation - prediction)
        weighted_error = np.linalg.norm((observation - prediction) * np.sqrt(precision))
        mean_absolute_error = np.mean(np.abs(observation - prediction))
        max_error = np.max(np.abs(observation - prediction))
        precision_weighted_elements = np.abs(observation - prediction) * precision

        return {
            "raw_error": float(raw_error),
            "weighted_error": float(weighted_error),
            "mean_absolute_error": float(mean_absolute_error),
            "max_error": float(max_error),
            "precision_weighted_elements": precision_weighted_elements.tolist(),
        }

    def compute_surprise(self, observation: np.ndarray, probability: np.ndarray) -> float:
        """Compute surprise (negative log probability).

        Parameters
        ----------
        observation : np.ndarray
            Observed values
        probability : np.ndarray
            Probability values

        Returns
        -------
        float
            Surprise value
        """
        # Clamp probabilities to avoid log(0)
        prob_clamped = np.clip(probability, self.eps, 1.0)
        surprise = -np.sum(np.log(prob_clamped))
        return float(max(0.0, surprise))

    def calculate_surprise(self, prediction_error: np.ndarray) -> float:
        """Calculate surprise (negative log probability).

        Parameters
        ----------
        prediction_error : np.ndarray
            Array of prediction errors

        Returns
        -------
        float
            Surprise value
        """
        return float(np.sum(prediction_error**2))

    def get_free_energy_components(self) -> Dict[str, Any]:
        """Get components of free energy calculation.

        Returns
        -------
        Dict[str, Any]
            Dictionary with free energy components
        """
        return {
            "precision_exteroceptive": self.precision_exteroceptive,
            "precision_interoceptive": self.precision_interoceptive,
            "prediction_error_weight": self.prediction_error_weight,
        }

    def compute_variational_free_energy(
        self,
        observation: np.ndarray,
        prediction: np.ndarray,
        precision: float | np.ndarray,
        posterior_mean: np.ndarray,
        posterior_cov: np.ndarray,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Compute variational free energy.

        Parameters
        ----------
        observation : np.ndarray
            Observed values
        prediction : np.ndarray
            Predicted values
        precision : float | np.ndarray
            Precision values
        posterior_mean : np.ndarray
            Posterior mean
        posterior_cov : np.ndarray
            Posterior covariance
        prior_mean : np.ndarray
            Prior mean
        prior_cov : np.ndarray
            Prior covariance

        Returns
        -------
        tuple[float, dict[str, float]]
            Free energy value and components dictionary
        """
        # Validate inputs
        if observation.shape != prediction.shape:
            raise ValueError("Observation and prediction must have same shape")

        if isinstance(precision, (int, float)):
            if precision <= 0:
                raise ValueError("Precision must be positive")
        elif isinstance(precision, np.ndarray):
            if np.any(precision <= 0):
                raise ValueError(
                    "Array precision must be positive: all precision values must be > 0"
                )

        if not np.all(np.isfinite(observation)):
            raise ValueError("Observation must be finite")

        if observation.size == 0:
            raise ValueError("Observation must not be empty")

        if np.shape(posterior_mean) != np.shape(prior_mean):
            raise ValueError(
                f"Posterior and prior mean shapes must match, got "
                f"{np.shape(posterior_mean)} and {np.shape(prior_mean)}"
            )

        # Accuracy term: prediction error weighted by precision
        pred_error = observation - prediction
        if isinstance(precision, (int, float)):
            accuracy = 0.5 * precision * np.sum(pred_error**2)
        else:
            accuracy = 0.5 * np.sum(precision * pred_error**2)

        # Complexity term: KL divergence between posterior and prior
        complexity = self._kl_divergence_gaussian(
            posterior_mean, posterior_cov, prior_mean, prior_cov
        )

        # Total free energy
        fe = accuracy + complexity

        components = {
            "accuracy": float(accuracy),
            "complexity": float(complexity),
            "prediction_error": float(np.linalg.norm(pred_error)),
        }

        return float(fe), components

    def compute_expected_free_energy(
        self,
        policy: np.ndarray,
        predicted_states: list[np.ndarray],
        predicted_observations: np.ndarray,
        preferences: np.ndarray,
        state_uncertainty: np.ndarray,
        horizon: int = 1,
    ) -> tuple[float, dict[str, float]]:
        """Compute expected free energy.

        Parameters
        ----------
        policy : np.ndarray
            Policy vector
        predicted_states : list[np.ndarray]
            List of predicted states
        predicted_observations : np.ndarray
            Predicted observations
        preferences : np.ndarray
            Preference values
        state_uncertainty : np.ndarray
            State uncertainty values
        horizon : int, optional
            Planning horizon, by default 1

        Returns
        -------
        tuple[float, dict[str, float]]
            Expected free energy and components
        """
        # Epistemic value: information gain (negative of uncertainty)
        epistemic_value = -np.mean(state_uncertainty)

        # Pragmatic value: alignment with preferences
        pragmatic_value = np.mean(predicted_observations * preferences)

        # Exploration drive: negative of epistemic value
        exploration_drive = -epistemic_value

        # Exploitation drive: pragmatic value
        exploitation_drive = pragmatic_value

        # Expected free energy
        efe = epistemic_value + pragmatic_value

        components = {
            "epistemic_value": float(epistemic_value),
            "pragmatic_value": float(pragmatic_value),
            "exploration_drive": float(exploration_drive),
            "exploitation_drive": float(exploitation_drive),
        }

        return float(efe), components

    @property
    def stability_monitor(self) -> Dict[str, Any]:
        """Get stability monitoring data.

        Returns
        -------
        Dict[str, Any]
            Stability monitoring data
        """
        return {"stable": True, "variance": 0.1}
