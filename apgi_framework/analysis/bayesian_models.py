"""Hierarchical Bayesian modeling framework for APGI parameter estimation.

This module implements the core Bayesian modeling infrastructure for joint
parameter estimation of θ₀ (baseline ignition threshold), Πᵢ (interoceptive precision),
and β (somatic bias) using Stan/PyMC3.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import standardized logging and security
from ..logging.standardized_logging import get_logger
from ..security.secure_pickle import safe_pickle_dump, safe_pickle_load

# Initialize logger
logger = get_logger(__name__)


@dataclass
class ParameterDistribution:
    """Distribution of a single parameter with uncertainty quantification."""

    mean: float
    std: float
    credible_interval_95: Tuple[float, float]
    posterior_samples: np.ndarray

    def __repr__(self) -> str:
        return (
            f"ParameterDistribution(mean={self.mean:.4f}, "
            f"std={self.std:.4f}, "
            f"95% CI=[{self.credible_interval_95[0]:.4f}, "
            f"{self.credible_interval_95[1]:.4f}])"
        )


@dataclass
class ParameterEstimates:
    """Complete parameter estimates for a participant."""

    participant_id: str
    session_id: str
    theta0: ParameterDistribution  # Baseline ignition threshold
    pi_i: ParameterDistribution  # Interoceptive precision
    beta: ParameterDistribution  # Somatic bias

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "theta0": {
                "mean": self.theta0.mean,
                "std": self.theta0.std,
                "ci_95": self.theta0.credible_interval_95,
            },
            "pi_i": {
                "mean": self.pi_i.mean,
                "std": self.pi_i.std,
                "ci_95": self.pi_i.credible_interval_95,
            },
            "beta": {
                "mean": self.beta.mean,
                "std": self.beta.std,
                "ci_95": self.beta.credible_interval_95,
            },
        }


class SurpriseAccumulator:
    """Implements surprise accumulation dynamics.

    Equation: dSₜ/dt = –Sₜ/τ + f(Πₑ·|εₑ|, β·Πᵢ·|εᵢ|)

    Where:
    - Sₜ: Surprise at time t
    - τ: Time constant for decay
    - Πₑ: Exteroceptive precision
    - εₑ: Exteroceptive prediction error
    - Πᵢ: Interoceptive precision
    - εᵢ: Interoceptive prediction error
    - β: Somatic bias (weighting of interoceptive vs exteroceptive)
    """

    def __init__(self, tau: float = 1.0, dt: float = 0.001):
        """Initialize surprise accumulator.

        Args:
            tau: Time constant for surprise decay (seconds)
            dt: Time step for numerical integration (seconds)
        """
        self.tau = tau
        self.dt = dt
        self.surprise = 0.0
        self.history: List[float] = []

    def reset(self) -> None:
        """Reset surprise accumulator to initial state."""
        self.surprise = 0.0
        self.history = []

    def compute_weighted_prediction_error(
        self, pi_e: float, epsilon_e: float, pi_i: float, epsilon_i: float, beta: float
    ) -> float:
        """Compute weighted prediction error: f(Πₑ·|εₑ|, β·Πᵢ·|εᵢ|).

        Args:
            pi_e: Exteroceptive precision
            epsilon_e: Exteroceptive prediction error
            pi_i: Interoceptive precision
            epsilon_i: Interoceptive prediction error
            beta: Somatic bias

        Returns:
            Weighted prediction error
        """
        extero_term = pi_e * float(np.abs(epsilon_e))
        intero_term = beta * pi_i * float(np.abs(epsilon_i))
        return float(extero_term + intero_term)

    def step(
        self, pi_e: float, epsilon_e: float, pi_i: float, epsilon_i: float, beta: float
    ) -> float:
        """Perform one integration step of surprise accumulation.

        Args:
            pi_e: Exteroceptive precision
            epsilon_e: Exteroceptive prediction error
            pi_i: Interoceptive precision
            epsilon_i: Interoceptive prediction error
            beta: Somatic bias

        Returns:
            Current surprise level
        """
        weighted_pe = self.compute_weighted_prediction_error(pi_e, epsilon_e, pi_i, epsilon_i, beta)

        # Euler integration: dS/dt = -S/τ + f(...)
        dS_dt = -self.surprise / self.tau + weighted_pe
        self.surprise += dS_dt * self.dt

        # Ensure non-negative surprise
        self.surprise = max(0.0, self.surprise)

        self.history.append(self.surprise)
        return self.surprise

    def integrate(
        self,
        pi_e: np.ndarray,
        epsilon_e: np.ndarray,
        pi_i: np.ndarray,
        epsilon_i: np.ndarray,
        beta: float,
        duration: float,
    ) -> np.ndarray:
        """Integrate surprise over a time period.

        Args:
            pi_e: Exteroceptive precision time series
            epsilon_e: Exteroceptive prediction error time series
            pi_i: Interoceptive precision time series
            epsilon_i: Interoceptive prediction error time series
            beta: Somatic bias
            duration: Duration to integrate (seconds)

        Returns:
            Surprise time series
        """
        self.reset()
        n_steps = int(duration / self.dt)
        surprise_trace = np.zeros(n_steps)

        for i in range(n_steps):
            idx = min(i, len(pi_e) - 1)
            surprise_trace[i] = self.step(
                pi_e[idx], epsilon_e[idx], pi_i[idx], epsilon_i[idx], beta
            )

        return surprise_trace


class IgnitionProbabilityCalculator:
    """Calculates ignition probability from surprise and threshold.

    Equation: Bₜ = σ(α(Sₜ – θₜ))

    Where:
    - Bₜ: Ignition probability at time t
    - σ: Sigmoid function
    - α: Gain parameter (steepness of sigmoid)
    - Sₜ: Surprise at time t
    - θₜ: Threshold at time t
    """

    def __init__(self, alpha: float = 10.0):
        """Initialize ignition probability calculator.

        Args:
            alpha: Gain parameter controlling sigmoid steepness
        """
        self.alpha = alpha

    @staticmethod
    def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Numerically stable sigmoid function."""
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def compute_probability(self, surprise: float, threshold: float) -> float:
        """Compute ignition probability for a single time point.

        Args:
            surprise: Current surprise level
            threshold: Current threshold

        Returns:
            Ignition probability [0, 1]
        """
        return float(self.sigmoid(self.alpha * (surprise - threshold)))

    def compute_probability_trace(
        self, surprise_trace: np.ndarray, threshold: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Compute ignition probability over time.

        Args:
            surprise_trace: Time series of surprise values
            threshold: Threshold value (can be scalar or array)

        Returns:
            Time series of ignition probabilities
        """
        if isinstance(threshold, (int, float)):
            threshold = np.full_like(surprise_trace, threshold)

        return np.asarray(self.sigmoid(self.alpha * (surprise_trace - threshold)))

    def find_ignition_time(
        self,
        surprise_trace: np.ndarray,
        threshold: Union[float, np.ndarray],
        probability_threshold: float = 0.5,
        dt: float = 0.001,
    ) -> Optional[float]:
        """Find the time when ignition probability exceeds threshold.

        Args:
            surprise_trace: Time series of surprise values
            threshold: Ignition threshold
            probability_threshold: Probability threshold for ignition
            dt: Time step

        Returns:
            Time of ignition (seconds), or None if no ignition
        """
        prob_trace = self.compute_probability_trace(surprise_trace, threshold)
        ignition_indices = np.where(prob_trace >= probability_threshold)[0]

        if len(ignition_indices) > 0:
            return float(ignition_indices[0] * dt)
        return None


class StanModelCompiler:
    """Manages Stan model compilation and caching for efficient reuse."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize Stan model compiler.

        Args:
            cache_dir: Directory for caching compiled models
        """
        if cache_dir is None:
            cache_dir = os.path.join(Path.home(), ".apgi_framework", "stan_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._compiled_models: Dict[str, Any] = {}

    def _get_model_hash(self, model_code: str) -> str:
        """Generate hash of model code for caching."""
        return hashlib.sha256(model_code.encode()).hexdigest()

    def _get_cache_path(self, model_hash: str) -> Path:
        """Get path to cached compiled model."""
        return self.cache_dir / f"model_{model_hash}.pkl"

    def compile_stan_model(self, model_code: str, force_recompile: bool = False) -> Any:
        """Compile Stan model with caching.

        Args:
            model_code: Stan model code
            force_recompile: Force recompilation even if cached

        Returns:
            Compiled Stan model
        """
        model_hash = self._get_model_hash(model_code)
        cache_path = self._get_cache_path(model_hash)

        # Check memory cache first
        if model_hash in self._compiled_models and not force_recompile:
            return self._compiled_models[model_hash]

        # Check disk cache
        if cache_path.exists() and not force_recompile:
            try:
                model = safe_pickle_load(cache_path)
                self._compiled_models[model_hash] = model
                return model
            except Exception as e:
                logger.warning(f"Failed to load cached model: {e}")

        # Compile new model
        try:
            import pystan  # type: ignore[import-not-found]

            model = pystan.StanModel(model_code=model_code)

            # Cache to disk
            try:
                safe_pickle_dump(model, cache_path)
            except Exception as e:
                logger.warning(f"Failed to cache model: {e}")

            self._compiled_models[model_hash] = model
            return model

        except ImportError:
            raise ImportError(
                "PyStan is required for Bayesian modeling. " "Install with: pip install pystan"
            )

    def clear_cache(self) -> None:
        """Clear all cached compiled models."""
        for cache_file in self.cache_dir.glob("model_*.pkl"):
            cache_file.unlink()
        self._compiled_models.clear()


class HierarchicalBayesianModel:
    """Hierarchical Bayesian model for joint parameter estimation.

    Estimates θ₀ (baseline ignition threshold), Πᵢ (interoceptive precision),
    and β (somatic bias) from behavioral and neural data.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize hierarchical Bayesian model.

        Args:
            cache_dir: Directory for caching compiled models
        """
        self.compiler = StanModelCompiler(cache_dir)
        self.model = None
        self.fit_result = None

    def get_stan_model_code(self) -> str:
        """Get Stan model code for hierarchical parameter estimation.

        Returns:
            Stan model code as string
        """
        return """
        data {
            int<lower=0> N_subjects;
            int<lower=0> N_trials_detection;
            int<lower=0> N_trials_heartbeat;
            int<lower=0> N_trials_oddball;
            
            // Detection task data (θ₀ estimation)
            int<lower=1,upper=N_subjects> subject_detection[N_trials_detection];
            vector[N_trials_detection] stimulus_intensity;
            int<lower=0,upper=1> detected[N_trials_detection];
            vector[N_trials_detection] p3b_amplitude;
            
            // Heartbeat detection task data (Πᵢ estimation)
            int<lower=1,upper=N_subjects> subject_heartbeat[N_trials_heartbeat];
            int<lower=0,upper=1> synchronous[N_trials_heartbeat];
            int<lower=0,upper=1> response_sync[N_trials_heartbeat];
            vector[N_trials_heartbeat] confidence;
            vector[N_trials_heartbeat] hep_amplitude;
            vector[N_trials_heartbeat] pupil_response;
            
            // Oddball task data (β estimation)
            int<lower=1,upper=N_subjects> subject_oddball[N_trials_oddball];
            int<lower=0,upper=2> trial_type[N_trials_oddball];  // 0=standard, 1=intero, 2=extero
            vector[N_trials_oddball] p3b_intero;
            vector[N_trials_oddball] p3b_extero;
        }
        
        parameters {
            // Population-level parameters (hyperparameters)
            real mu_theta0;
            real<lower=0> sigma_theta0;
            real mu_pi_i;
            real<lower=0> sigma_pi_i;
            real mu_beta;
            real<lower=0> sigma_beta;
            
            // Individual-level parameters (non-centered parameterization)
            vector[N_subjects] theta0_raw;
            vector[N_subjects] pi_i_raw;
            vector[N_subjects] beta_raw;
            
            // Auxiliary parameters
            real<lower=0> sigma_p3b;
            real<lower=0> sigma_hep;
            real<lower=0> sigma_pupil;
        }
        
        transformed parameters {
            // Non-centered parameterization for better sampling
            vector[N_subjects] theta0 = mu_theta0 + sigma_theta0 * theta0_raw;
            vector[N_subjects] pi_i = mu_pi_i + sigma_pi_i * pi_i_raw;
            vector[N_subjects] beta = mu_beta + sigma_beta * beta_raw;
        }
        
        model {
            // Hyperpriors
            mu_theta0 ~ normal(0, 1);
            sigma_theta0 ~ exponential(1);
            mu_pi_i ~ normal(0, 1);
            sigma_pi_i ~ exponential(1);
            mu_beta ~ normal(1, 0.5);
            sigma_beta ~ exponential(2);
            
            // Individual-level priors (non-centered)
            theta0_raw ~ std_normal();
            pi_i_raw ~ std_normal();
            beta_raw ~ std_normal();
            
            // Auxiliary parameter priors
            sigma_p3b ~ exponential(1);
            sigma_hep ~ exponential(1);
            sigma_pupil ~ exponential(1);
            
            // Detection task likelihood
            for (n in 1:N_trials_detection) {
                int subj = subject_detection[n];
                detected[n] ~ bernoulli_logit(
                    theta0[subj] * stimulus_intensity[n]
                );
            }
            
            // P3b amplitude relates to threshold
            for (n in 1:N_trials_detection) {
                int subj = subject_detection[n];
                p3b_amplitude[n] ~ normal(-theta0[subj], sigma_p3b);
            }
            
            // Heartbeat detection likelihood
            for (n in 1:N_trials_heartbeat) {
                int subj = subject_heartbeat[n];
                real d_prime = pi_i[subj];
                real prob_correct = Phi(d_prime / 2);
                
                if (synchronous[n] == 1) {
                    response_sync[n] ~ bernoulli(prob_correct);
                } else {
                    response_sync[n] ~ bernoulli(1 - prob_correct);
                }
            }
            
            // HEP amplitude relates to interoceptive precision
            for (n in 1:N_trials_heartbeat) {
                int subj = subject_heartbeat[n];
                hep_amplitude[n] ~ normal(pi_i[subj], sigma_hep);
            }
            
            // Pupil response relates to interoceptive precision
            for (n in 1:N_trials_heartbeat) {
                int subj = subject_heartbeat[n];
                pupil_response[n] ~ normal(pi_i[subj], sigma_pupil);
            }
            
            // Oddball task: β relates P3b ratio
            for (n in 1:N_trials_oddball) {
                int subj = subject_oddball[n];
                if (trial_type[n] == 1) {  // Interoceptive deviant
                    p3b_intero[n] ~ normal(beta[subj], sigma_p3b);
                } else if (trial_type[n] == 2) {  // Exteroceptive deviant
                    p3b_extero[n] ~ normal(1.0, sigma_p3b);
                }
            }
        }
        
        generated quantities {
            // Posterior predictive checks
            vector[N_subjects] theta0_pred;
            vector[N_subjects] pi_i_pred;
            vector[N_subjects] beta_pred;
            
            for (s in 1:N_subjects) {
                theta0_pred[s] = normal_rng(mu_theta0, sigma_theta0);
                pi_i_pred[s] = normal_rng(mu_pi_i, sigma_pi_i);
                beta_pred[s] = normal_rng(mu_beta, sigma_beta);
            }
        }
        """

    def compile_model(self, force_recompile: bool = False) -> None:
        """Compile the Stan model.

        Args:
            force_recompile: Force recompilation even if cached
        """
        model_code = self.get_stan_model_code()
        self.model = self.compiler.compile_stan_model(model_code, force_recompile)

    def prepare_data(
        self,
        detection_data: Dict[str, np.ndarray],
        heartbeat_data: Dict[str, np.ndarray],
        oddball_data: Dict[str, np.ndarray],
        n_subjects: int,
    ) -> Dict[str, Any]:
        """Prepare data for Stan model.

        Args:
            detection_data: Detection task data
            heartbeat_data: Heartbeat detection task data
            oddball_data: Oddball task data
            n_subjects: Number of subjects

        Returns:
            Dictionary formatted for Stan
        """
        return {
            "N_subjects": n_subjects,
            "N_trials_detection": len(detection_data["stimulus_intensity"]),
            "N_trials_heartbeat": len(heartbeat_data["synchronous"]),
            "N_trials_oddball": len(oddball_data["trial_type"]),
            # Detection task
            "subject_detection": detection_data["subject_id"],
            "stimulus_intensity": detection_data["stimulus_intensity"],
            "detected": detection_data["detected"],
            "p3b_amplitude": detection_data["p3b_amplitude"],
            # Heartbeat task
            "subject_heartbeat": heartbeat_data["subject_id"],
            "synchronous": heartbeat_data["synchronous"],
            "response_sync": heartbeat_data["response_sync"],
            "confidence": heartbeat_data["confidence"],
            "hep_amplitude": heartbeat_data["hep_amplitude"],
            "pupil_response": heartbeat_data["pupil_response"],
            # Oddball task
            "subject_oddball": oddball_data["subject_id"],
            "trial_type": oddball_data["trial_type"],
            "p3b_intero": oddball_data["p3b_intero"],
            "p3b_extero": oddball_data["p3b_extero"],
        }

    def fit(
        self,
        data: Dict[str, Any],
        chains: int = 4,
        iter: int = 2000,
        warmup: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Fit the hierarchical Bayesian model.

        Args:
            data: Prepared data dictionary
            chains: Number of MCMC chains
            iter: Number of iterations per chain
            warmup: Number of warmup iterations (default: iter // 2)
            **kwargs: Additional arguments for Stan sampling

        Returns:
            Stan fit object
        """
        if self.model is None:
            self.compile_model()

        if self.model is None:
            raise RuntimeError("Model compilation failed")

        if warmup is None:  # type: ignore[unreachable]
            warmup = iter // 2

        self.fit_result = self.model.sampling(
            data=data, chains=chains, iter=iter, warmup=warmup, **kwargs
        )

        return self.fit_result

    def extract_parameters(
        self, subject_id: int, participant_id: str, session_id: str
    ) -> ParameterEstimates:
        """Extract parameter estimates for a specific subject.

        Args:
            subject_id: Subject index (0-based)
            participant_id: Participant identifier
            session_id: Session identifier

        Returns:
            Parameter estimates with uncertainty
        """
        if self.fit_result is None:
            raise ValueError("Model must be fit before extracting parameters")

        # Extract posterior samples
        theta0_samples = self.fit_result.extract("theta0")["theta0"][:, subject_id]  # type: ignore[unreachable]
        pi_i_samples = self.fit_result.extract("pi_i")["pi_i"][:, subject_id]
        beta_samples = self.fit_result.extract("beta")["beta"][:, subject_id]

        # Compute statistics
        def make_distribution(samples: np.ndarray) -> ParameterDistribution:
            return ParameterDistribution(
                mean=float(np.mean(samples)),
                std=float(np.std(samples)),
                credible_interval_95=(
                    float(np.percentile(samples, 2.5)),
                    float(np.percentile(samples, 97.5)),
                ),
                posterior_samples=samples,
            )

        return ParameterEstimates(
            participant_id=participant_id,
            session_id=session_id,
            theta0=make_distribution(theta0_samples),
            pi_i=make_distribution(pi_i_samples),
            beta=make_distribution(beta_samples),
        )
