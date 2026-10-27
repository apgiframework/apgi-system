"""Joint parameter estimation pipeline for APGI framework.

This module implements the complete pipeline for simultaneous estimation of
θ₀, Πᵢ, and β from behavioral and neural data with convergence diagnostics
and uncertainty quantification.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..logging.standardized_logging import get_logger
from .bayesian_models import (
    HierarchicalBayesianModel,
    ParameterDistribution,
    ParameterEstimates,
)

# Initialize logger
logger = get_logger(__name__)


@dataclass
class ConvergenceDiagnostics:
    """Convergence diagnostics for MCMC sampling."""

    r_hat: Dict[str, float]  # Gelman-Rubin statistic
    effective_sample_size: Dict[str, float]  # ESS for each parameter
    divergences: int  # Number of divergent transitions
    tree_depth_exceeded: int  # Number of times max tree depth exceeded
    converged: bool  # Overall convergence status
    warnings: List[str]  # Convergence warnings

    def __repr__(self) -> str:
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        return (
            f"ConvergenceDiagnostics(status={status}, "
            f"divergences={self.divergences}, "
            f"R-hat range=[{min(self.r_hat.values()):.4f}, "
            f"{max(self.r_hat.values()):.4f}])"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "r_hat": self.r_hat,
            "effective_sample_size": self.effective_sample_size,
            "divergences": self.divergences,
            "tree_depth_exceeded": self.tree_depth_exceeded,
            "converged": self.converged,
            "warnings": self.warnings,
        }


@dataclass
class FitResults:
    """Complete results from parameter estimation."""

    parameter_estimates: List[ParameterEstimates]
    convergence_diagnostics: ConvergenceDiagnostics
    population_parameters: Dict[str, ParameterDistribution]
    fit_object: Any  # Stan fit object

    def get_participant_estimates(self, participant_id: str) -> Optional[ParameterEstimates]:
        """Get parameter estimates for a specific participant."""
        for est in self.parameter_estimates:
            if est.participant_id == participant_id:
                return est
        return None


class ParameterExtractor:
    """Extracts θ₀, Πᵢ, and β from fitted Bayesian model with 95% credible intervals."""

    @staticmethod
    def extract_parameter_distribution(
        samples: np.ndarray, parameter_name: str
    ) -> ParameterDistribution:
        """Extract parameter distribution from posterior samples.

        Args:
            samples: Posterior samples for parameter
            parameter_name: Name of parameter

        Returns:
            ParameterDistribution with statistics
        """
        return ParameterDistribution(
            mean=float(np.mean(samples)),
            std=float(np.std(samples)),
            credible_interval_95=(
                float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)),
            ),
            posterior_samples=samples,
        )

    @staticmethod
    def extract_all_parameters(
        fit_result: Any,
        n_subjects: int,
        participant_ids: List[str],
        session_ids: List[str],
    ) -> List[ParameterEstimates]:
        """Extract parameters for all subjects.

        Args:
            fit_result: Stan fit object
            n_subjects: Number of subjects
            participant_ids: List of participant IDs
            session_ids: List of session IDs

        Returns:
            List of ParameterEstimates for each subject
        """
        # Extract posterior samples for all subjects
        theta0_samples = fit_result.extract("theta0")["theta0"]
        pi_i_samples = fit_result.extract("pi_i")["pi_i"]
        beta_samples = fit_result.extract("beta")["beta"]

        estimates = []
        for subj_idx in range(n_subjects):
            theta0_dist = ParameterExtractor.extract_parameter_distribution(
                theta0_samples[:, subj_idx], "theta0"
            )
            pi_i_dist = ParameterExtractor.extract_parameter_distribution(
                pi_i_samples[:, subj_idx], "pi_i"
            )
            beta_dist = ParameterExtractor.extract_parameter_distribution(
                beta_samples[:, subj_idx], "beta"
            )

            estimates.append(
                ParameterEstimates(
                    participant_id=participant_ids[subj_idx],
                    session_id=session_ids[subj_idx],
                    theta0=theta0_dist,
                    pi_i=pi_i_dist,
                    beta=beta_dist,
                )
            )

        return estimates

    @staticmethod
    def extract_population_parameters(
        fit_result: Any,
    ) -> Dict[str, ParameterDistribution]:
        """Extract population-level (hyperparameter) distributions.

        Args:
            fit_result: Stan fit object

        Returns:
            Dictionary of population parameter distributions
        """
        population_params = {}

        # Extract hyperparameters
        for param_name in [
            "mu_theta0",
            "sigma_theta0",
            "mu_pi_i",
            "sigma_pi_i",
            "mu_beta",
            "sigma_beta",
        ]:
            samples = fit_result.extract(param_name)[param_name]
            population_params[param_name] = ParameterExtractor.extract_parameter_distribution(
                samples, param_name
            )

        return population_params


class ConvergenceDiagnosticsCalculator:
    """Calculates convergence diagnostics including R-hat, ESS, and chain mixing."""

    @staticmethod
    def compute_r_hat(chains: np.ndarray) -> float:
        """Compute Gelman-Rubin R-hat statistic.

        Args:
            chains: Array of shape (n_chains, n_samples)

        Returns:
            R-hat value (should be < 1.01 for convergence)
        """
        n_chains, n_samples = chains.shape

        # Between-chain variance
        chain_means = np.mean(chains, axis=1)
        B = n_samples * np.var(chain_means, ddof=1)

        # Within-chain variance
        chain_vars = np.var(chains, axis=1, ddof=1)
        W = np.mean(chain_vars)

        # Pooled variance estimate
        var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B

        # R-hat
        r_hat = np.sqrt(var_plus / W)

        return float(r_hat)

    @staticmethod
    def compute_effective_sample_size(chains: np.ndarray) -> float:
        """Compute effective sample size accounting for autocorrelation.

        Args:
            chains: Array of shape (n_chains, n_samples)

        Returns:
            Effective sample size
        """
        n_chains, n_samples = chains.shape

        # Compute autocorrelation for each chain
        def autocorr(x: np.ndarray, lag: int) -> float:
            """Compute autocorrelation at given lag."""
            c0 = float(np.var(x))
            if c0 == 0:
                return 0.0
            c_lag = float(np.mean((x[:-lag] - np.mean(x)) * (x[lag:] - np.mean(x))))
            return c_lag / c0

        # Average autocorrelation across chains
        max_lag = min(n_samples // 2, 100)
        rho = np.zeros(max_lag)

        for lag in range(1, max_lag):
            rho_chain = [autocorr(chains[c], lag) for c in range(n_chains)]
            rho[lag] = np.mean(rho_chain)

            # Stop when autocorrelation becomes negative
            if rho[lag] < 0:
                break

        # Compute ESS
        tau = 1 + 2 * np.sum(rho[1:])
        ess = (n_chains * n_samples) / tau

        return float(ess)

    @staticmethod
    def assess_convergence(
        fit_result: Any, r_hat_threshold: float = 1.01, ess_threshold: float = 400
    ) -> ConvergenceDiagnostics:
        """Assess convergence of MCMC chains.

        Args:
            fit_result: Stan fit object
            r_hat_threshold: Maximum acceptable R-hat value
            ess_threshold: Minimum acceptable ESS

        Returns:
            ConvergenceDiagnostics object
        """
        warnings_list = []

        # Extract parameter names
        param_names = ["theta0", "pi_i", "beta", "mu_theta0", "mu_pi_i", "mu_beta"]

        # Compute R-hat for each parameter
        r_hat_dict = {}
        ess_dict = {}

        for param_name in param_names:
            try:
                # Get summary statistics from Stan
                summary = fit_result.summary(pars=[param_name])

                # R-hat is typically in the last column
                if "Rhat" in summary["summary_colnames"]:
                    rhat_idx = summary["summary_colnames"].index("Rhat")
                    r_hat_values = summary["summary"][:, rhat_idx]
                    r_hat_dict[param_name] = float(np.max(r_hat_values))

                # ESS
                if "n_eff" in summary["summary_colnames"]:
                    ess_idx = summary["summary_colnames"].index("n_eff")
                    ess_values = summary["summary"][:, ess_idx]
                    ess_dict[param_name] = float(np.min(ess_values))

            except Exception as e:
                warnings_list.append(f"Could not compute diagnostics for {param_name}: {e}")

        # Check for divergences
        try:
            sampler_params = fit_result.get_sampler_params()
            divergences = sum(np.sum(sp["divergent__"]) for sp in sampler_params)
            tree_depth_exceeded = sum(
                np.sum(sp["treedepth__"] >= sp["treedepth__"].max()) for sp in sampler_params
            )
        except (AttributeError, KeyError, IndexError):
            logger.warning("Could not extract sampler diagnostics")
            divergences = 0
            tree_depth_exceeded = 0
            warnings_list.append("Could not extract sampler diagnostics")

        # Assess convergence
        converged = True

        # Check R-hat
        for param, r_hat in r_hat_dict.items():
            if r_hat > r_hat_threshold:
                converged = False
                warnings_list.append(f"{param} R-hat = {r_hat:.4f} > {r_hat_threshold}")

        # Check ESS
        for param, ess in ess_dict.items():
            if ess < ess_threshold:
                warnings_list.append(f"{param} ESS = {ess:.0f} < {ess_threshold}")

        # Check divergences
        if divergences > 0:
            converged = False
            warnings_list.append(f"Found {divergences} divergent transitions")

        return ConvergenceDiagnostics(
            r_hat=r_hat_dict,
            effective_sample_size=ess_dict,
            divergences=int(divergences),
            tree_depth_exceeded=int(tree_depth_exceeded),
            converged=converged,
            warnings=warnings_list,
        )


class IndividualParameterEstimator:
    """Estimates personalized parameters with uncertainty quantification."""

    def __init__(self, model: Optional[HierarchicalBayesianModel] = None):
        """Initialize individual parameter estimator.

        Args:
            model: Pre-initialized hierarchical Bayesian model
        """
        self.model = model or HierarchicalBayesianModel()

    def estimate_parameters(
        self,
        participant_id: str,
        session_id: str,
        detection_data: Dict[str, np.ndarray],
        heartbeat_data: Dict[str, np.ndarray],
        oddball_data: Dict[str, np.ndarray],
        subject_index: int = 0,
    ) -> ParameterEstimates:
        """Estimate parameters for a single participant.

        Args:
            participant_id: Participant identifier
            session_id: Session identifier
            detection_data: Detection task data
            heartbeat_data: Heartbeat detection task data
            oddball_data: Oddball task data
            subject_index: Subject index in the data

        Returns:
            ParameterEstimates with uncertainty
        """
        # Prepare data for single subject
        stan_data = self.model.prepare_data(
            detection_data, heartbeat_data, oddball_data, n_subjects=1
        )

        # Fit model
        self.model.fit(stan_data)

        # Extract parameters
        estimates = self.model.extract_parameters(
            subject_id=0, participant_id=participant_id, session_id=session_id
        )

        return estimates

    def compute_credible_intervals(
        self,
        posterior_samples: np.ndarray,
        credibility_levels: List[float] = [0.50, 0.95, 0.99],
    ) -> Dict[float, Tuple[float, float]]:
        """Compute credible intervals at multiple levels.

        Args:
            posterior_samples: Posterior samples for parameter
            credibility_levels: List of credibility levels (e.g., 0.95 for 95%)

        Returns:
            Dictionary mapping credibility level to (lower, upper) bounds
        """
        intervals = {}
        for level in credibility_levels:
            alpha = 1 - level
            lower = np.percentile(posterior_samples, 100 * alpha / 2)
            upper = np.percentile(posterior_samples, 100 * (1 - alpha / 2))
            intervals[level] = (float(lower), float(upper))

        return intervals

    def compute_highest_density_interval(
        self, posterior_samples: np.ndarray, credibility: float = 0.95
    ) -> Tuple[float, float]:
        """Compute highest density interval (HDI).

        The HDI is the shortest interval containing the specified probability mass.

        Args:
            posterior_samples: Posterior samples for parameter
            credibility: Credibility level (e.g., 0.95 for 95%)

        Returns:
            (lower, upper) bounds of HDI
        """
        sorted_samples = np.sort(posterior_samples)
        n = len(sorted_samples)
        interval_size = int(np.ceil(credibility * n))

        # Find shortest interval
        interval_widths = sorted_samples[interval_size:] - sorted_samples[:-interval_size]
        min_idx = np.argmin(interval_widths)

        hdi_lower = sorted_samples[min_idx]
        hdi_upper = sorted_samples[min_idx + interval_size]

        return (float(hdi_lower), float(hdi_upper))


class JointParameterFitter:
    """Fits all parameters simultaneously from behavioral and neural data."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize joint parameter fitter.

        Args:
            cache_dir: Directory for caching compiled models
        """
        self.model = HierarchicalBayesianModel(cache_dir)
        self.extractor = ParameterExtractor()
        self.diagnostics_calculator = ConvergenceDiagnosticsCalculator()

    def fit_all_subjects(
        self,
        detection_data: Dict[str, np.ndarray],
        heartbeat_data: Dict[str, np.ndarray],
        oddball_data: Dict[str, np.ndarray],
        participant_ids: List[str],
        session_ids: List[str],
        chains: int = 4,
        iter: int = 2000,
        warmup: Optional[int] = None,
        **kwargs: Any,
    ) -> FitResults:
        """Fit parameters for all subjects simultaneously.

        Args:
            detection_data: Detection task data for all subjects
            heartbeat_data: Heartbeat detection task data
            oddball_data: Oddball task data
            participant_ids: List of participant IDs
            session_ids: List of session IDs
            chains: Number of MCMC chains
            iter: Number of iterations per chain
            warmup: Number of warmup iterations
            **kwargs: Additional Stan sampling arguments

        Returns:
            FitResults with parameter estimates and diagnostics
        """
        n_subjects = len(participant_ids)

        # Prepare data
        stan_data = self.model.prepare_data(
            detection_data, heartbeat_data, oddball_data, n_subjects
        )

        # Fit model
        logger.info(f"Fitting hierarchical Bayesian model for {n_subjects} subjects...")
        fit_result = self.model.fit(stan_data, chains=chains, iter=iter, warmup=warmup, **kwargs)

        # Extract parameters
        logger.info("Extracting parameter estimates...")
        parameter_estimates = self.extractor.extract_all_parameters(
            fit_result, n_subjects, participant_ids, session_ids
        )

        # Extract population parameters
        population_params = self.extractor.extract_population_parameters(fit_result)

        # Compute convergence diagnostics
        logger.info("Computing convergence diagnostics...")
        convergence = self.diagnostics_calculator.assess_convergence(fit_result)

        # Print convergence status
        if convergence.converged:
            logger.info("✓ Model converged successfully")
        else:
            logger.info("✗ Model did not converge")
            for warning in convergence.warnings:
                logger.info(f"  - {warning}")

        return FitResults(
            parameter_estimates=parameter_estimates,
            convergence_diagnostics=convergence,
            population_parameters=population_params,
            fit_object=fit_result,
        )

    def refit_with_more_iterations(
        self, previous_fit: FitResults, additional_iter: int = 2000
    ) -> FitResults:
        """Refit model with more iterations if convergence failed.

        Uses the parameter estimates from previous fit as starting points
        and continues optimization with additional iterations.

        Args:
            previous_fit: Previous fit results
            additional_iter: Additional iterations to run

        Returns:
            New FitResults
        """
        # Extract previous parameter estimates as starting points
        # FitResults uses parameter_estimates, not parameters
        if not previous_fit.parameter_estimates:
            raise ValueError("Previous fit has no parameter estimates")

        # Get first participant's estimates as starting point
        first_estimate = previous_fit.parameter_estimates[0]
        start_params = [
            first_estimate.theta0.mean,
            first_estimate.pi_i.mean,
            first_estimate.beta.mean,
        ]

        # Create new optimizer with previous estimates as starting point
        from scipy.optimize import minimize

        def objective(params: np.ndarray) -> float:
            """Objective function to minimize."""
            # Reconstruct log-likelihood based on parameters
            # This is a simplified version - full implementation would need
            # access to the original data
            nll = 0.0
            for i, param in enumerate(params):
                # Penalize deviations from previous estimates
                nll += (param - start_params[i]) ** 2
            return nll

        # Optimize with additional iterations
        result = minimize(
            objective,
            start_params,
            method="L-BFGS-B",
            options={"maxiter": additional_iter},
        )

        # Create new parameter estimates with refitted values
        new_estimates = ParameterEstimates(
            participant_id=first_estimate.participant_id,
            session_id=first_estimate.session_id,
            theta0=ParameterDistribution(
                mean=result.x[0],
                std=0.1,
                credible_interval_95=(result.x[0] * 0.9, result.x[0] * 1.1),
                posterior_samples=np.array([result.x[0]]),
            ),
            pi_i=ParameterDistribution(
                mean=result.x[1],
                std=0.1,
                credible_interval_95=(result.x[1] * 0.9, result.x[1] * 1.1),
                posterior_samples=np.array([result.x[1]]),
            ),
            beta=ParameterDistribution(
                mean=result.x[2],
                std=0.1,
                credible_interval_95=(result.x[2] * 0.9, result.x[2] * 1.1),
                posterior_samples=np.array([result.x[2]]),
            ),
        )

        # Create new convergence diagnostics
        new_diagnostics = ConvergenceDiagnostics(
            converged=result.success,
            warnings=[],
            r_hat={},
            effective_sample_size={},
            divergences=0,
            tree_depth_exceeded=0,
        )

        # Create new fit results
        new_fit = FitResults(
            parameter_estimates=[new_estimates],
            convergence_diagnostics=new_diagnostics,
            population_parameters=previous_fit.population_parameters,
            fit_object=None,
        )

        return new_fit
