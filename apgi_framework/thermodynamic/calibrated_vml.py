"""Calibrated Virtual Metabolic Layer.

This module extends the VirtualMetabolicLayer with ground-truth calibrated
cost coefficients (c_1, c_2) from high-resolution metabolic imaging data.

Integration:
------------
The CalibratedVirtualMetabolicLayer wraps the existing VirtualMetabolicLayer
and replaces the generic ATP cost estimates with biophysically-grounded
values from Two-photon and P-MRS datasets.

Usage:
------
    # Method 1: Use default literature-based coefficients
    from apgi_framework.thermodynamic import CalibratedVirtualMetabolicLayer
    vml = CalibratedVirtualMetabolicLayer()  # Uses Attwell & Laughlin values

    # Method 2: Calibrate from datasets
    from apgi_framework.thermodynamic import MetabolicCalibrator
    calibrator = MetabolicCalibrator()
    calibrator.load_two_photon_dataset("zenodo://marvin_2024_iatpsnfr2")
    calibrator.load_pmrs_dataset("chen_2020_7t_fmrs.par")
    coeffs = calibrator.fit_coefficients()

    vml = CalibratedVirtualMetabolicLayer(calibrator=calibrator)

    # Use in simulation
    cost = vml.compute_ignition_cost(
        ignition_signal=3.5,
        threshold=2.0,
        use_calibrated_coefficients=True
    )
    print(f"c_1: {vml.c_1:.2e} ATP/AP, c_2: {vml.c_2:.2e} ATP/ms/neuron")
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import numpy as np
import numpy.typing as npt

from apgi_framework.thermodynamic.metabolic_calibration import (
    CalibratedCostCoefficients,
    MetabolicCalibrator,
    get_default_coefficients,
)
from apgi_framework.thermodynamic.neural_mass_metabolism import (
    MetabolicCostFactors,
    VirtualMetabolicLayer,
)


class CalibratedVirtualMetabolicLayer(VirtualMetabolicLayer):
    """Virtual Metabolic Layer with ground-truth calibrated cost coefficients.

    This class extends VirtualMetabolicLayer to use experimentally-measured
    cost coefficients c_1 (dynamic) and c_2 (static) from high-resolution
    metabolic imaging (Two-photon, P-MRS).

    The calibration partitions ATP consumption into:
    - c_1 × N_ignitions: Activity-driven costs (ion pumping, glutamate recycling)
    - c_2 × T × N_neurons: Baseline maintenance costs (resting potential, etc.)

    Attributes:
        c_1: ATP molecules per ignition event per neuron (from Two-photon data)
        c_2: ATP molecules per millisecond per neuron at rest (from P-MRS data)
        calibrator: MetabolicCalibrator instance used for fitting
        coefficients: CalibratedCostCoefficients with full metadata

    Example:
        >>> # Use literature defaults
        >>> vml = CalibratedVirtualMetabolicLayer()
        >>> print(f"c_1 = {vml.c_1:.2e} ATP/AP")
        c_1 = 1.50e+07 ATP/AP

        >>> # Calibrate from datasets
        >>> calibrator = MetabolicCalibrator()
        >>> calibrator.load_two_photon_dataset("path/to/iATPSnFR2_data.csv")
        >>> vml = CalibratedVirtualMetabolicLayer(calibrator=calibrator)
        >>> print(f"Calibrated c_1 = {vml.c_1:.2e} ± {vml.c_1_uncertainty:.2e}")
    """

    def __init__(
        self,
        neural_params: Optional[Any] = None,
        metabolic_factors: Optional[MetabolicCostFactors] = None,
        calibrator: Optional[MetabolicCalibrator] = None,
        coefficients: Optional[CalibratedCostCoefficients] = None,
        use_calibrated_defaults: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the Calibrated Virtual Metabolic Layer.

        Args:
            neural_params: Neural mass model parameters (passed to parent)
            metabolic_factors: Metabolic cost factors (passed to parent)
            calibrator: MetabolicCalibrator to fit coefficients from data
            coefficients: Pre-computed CalibratedCostCoefficients
            use_calibrated_defaults: Use literature-based defaults if no data
            rng: Random number generator

        Note:
            Priority for coefficients:
            1. Explicit `coefficients` argument
            2. `calibrator` with fitted coefficients
            3. `use_calibrated_defaults=True` → literature values
            4. Fallback → generic estimates from parent class
        """
        # Initialize parent class
        super().__init__(
            neural_params=neural_params,
            metabolic_factors=metabolic_factors,
            rng=rng,
        )

        # Store calibrator reference
        self.calibrator = calibrator

        # Determine coefficients to use
        if coefficients is not None:
            self.coefficients = coefficients
        elif calibrator is not None:
            # Fit from loaded data
            self.coefficients = calibrator.fit_coefficients()
        elif use_calibrated_defaults:
            self.coefficients = get_default_coefficients()
        else:
            # Use placeholder - will rely on parent class estimates
            self.coefficients = CalibratedCostCoefficients(
                c_1_dynamic=0.0,
                c_2_static=0.0,
                calibration_source="literature",
            )

        # Extract convenient access
        self.c_1 = self.coefficients.c_1_dynamic
        self.c_2 = self.coefficients.c_2_static
        self.c_1_uncertainty = self.coefficients.c_1_uncertainty
        self.c_2_uncertainty = self.coefficients.c_2_uncertainty

        # Tracking for validation
        self._calibrated_computations: List[Dict[str, Any]] = []
        self._max_computation_history = 100

    def compute_ignition_cost_calibrated(
        self,
        ignition_signal: float,
        threshold: float,
        workspace_content: Optional[npt.NDArray[np.float64]] = None,
        ignition_duration_ms: float = 300.0,
        num_neurons: Optional[int] = None,
        use_calibrated: bool = True,
    ) -> Dict[str, Any]:
        """Compute ignition cost using calibrated c_1/c_2 coefficients.

        This method provides a ground-truth based estimate that partitions
        ATP consumption into:
        - Dynamic component: c_1 × N_neurons (cost of ignition event itself)
        - Static component: c_2 × duration_ms × N_neurons (baseline during event)

        Args:
            ignition_signal: Accumulated surprise S_t
            threshold: Ignition threshold θ_t
            workspace_content: Broadcast content (for info content estimation)
            ignition_duration_ms: Duration of ignition in ms
            num_neurons: Number of neurons involved (default: neural_params.num_neurons)
            use_calibrated: If True, use c_1/c_2; if False, use parent method

        Returns:
            Dictionary with cost breakdown:
            - atp_total: Total ATP molecules consumed
            - atp_dynamic: c_1 component (activity-driven)
            - atp_static: c_2 component (baseline maintenance)
            - c_1_used: c_1 value used in calculation
            - c_2_used: c_2 value used in calculation
            - kappa_landauer: Thermodynamic efficiency ratio
            - bits_broadcast: Estimated information content
        """
        if not use_calibrated or self.c_1 == 0 or self.c_2 == 0:
            # Fall back to parent class estimation
            return super().compute_ignition_cost(
                ignition_signal=ignition_signal,
                threshold=threshold,
                workspace_content=workspace_content,
                ignition_duration_ms=ignition_duration_ms,
            )

        # Get number of neurons
        n_neurons = num_neurons or self.neural_model.params.num_neurons

        # Compute signal excess and broadcast amplitude
        signal_excess = max(0.0, ignition_signal - threshold)
        broadcast_amplitude = min(1.0, signal_excess / threshold) if threshold > 0 else 0.0

        # Determine workspace content information
        if workspace_content is not None:
            content_magnitude = float(np.linalg.norm(workspace_content))
            bits_content = min(256.0, content_magnitude)
        else:
            bits_content = 128.0  # Default: half of typical workspace capacity

        # Compute ATP using calibrated coefficients
        # Model: ATP = c_1 × N_neurons (dynamic) + c_2 × duration_ms × N_neurons (static)
        atp_dynamic = self.c_1 * n_neurons * (1 + 0.5 * broadcast_amplitude)
        atp_static = self.c_2 * ignition_duration_ms * n_neurons
        atp_total = atp_dynamic + atp_static

        # Compute κ (Landauer-normalized efficiency)
        # Use parent class ATP calculator for consistency
        kappa = self.atp_calculator.compute_kappa(
            atp_total=atp_total,
            bits_processed=bits_content,
            consider_landauer=True,
        )

        result = {
            "atp_total": float(atp_total),
            "atp_dynamic": float(atp_dynamic),
            "atp_static": float(atp_static),
            "dynamic_fraction": float(atp_dynamic / atp_total) if atp_total > 0 else 0.0,
            "c_1_used": self.c_1,
            "c_2_used": self.c_2,
            "c_1_uncertainty": self.c_1_uncertainty,
            "c_2_uncertainty": self.c_2_uncertainty,
            "signal_excess": signal_excess,
            "broadcast_amplitude": broadcast_amplitude,
            "ignition_signal": ignition_signal,
            "threshold": threshold,
            "duration_ms": ignition_duration_ms,
            "workspace_neurons": n_neurons,
            "bits_broadcast": float(bits_content),
            "kappa_landauer": float(kappa),
            "atp_per_bit": float(atp_total / bits_content) if bits_content > 0 else 0.0,
            "calibration_source": self.coefficients.calibration_source,
            "temporal_resolution_ms": self.coefficients.temporal_resolution_ms,
        }

        # Track for history
        self._calibrated_computations.append(result)
        if len(self._calibrated_computations) > self._max_computation_history:
            self._calibrated_computations.pop(0)

        return result

    def compare_calibration_methods(
        self,
        ignition_signal: float = 3.5,
        threshold: float = 2.0,
        ignition_duration_ms: float = 300.0,
    ) -> Dict[str, Any]:
        """Compare calibrated vs. generic (parent class) cost estimates.

        This is useful for validation and understanding the impact of
        ground-truth calibration on metabolic cost estimates.

        Returns:
            Comparison dictionary with both estimates and differences
        """
        # Calibrated estimate
        calibrated = self.compute_ignition_cost_calibrated(
            ignition_signal=ignition_signal,
            threshold=threshold,
            ignition_duration_ms=ignition_duration_ms,
            use_calibrated=True,
        )

        # Generic estimate (parent class)
        generic = self.compute_ignition_cost(
            ignition_signal=ignition_signal,
            threshold=threshold,
            ignition_duration_ms=ignition_duration_ms,
        )

        # Compute differences
        atp_diff = calibrated["atp_total"] - generic["atp_total"]
        atp_ratio = (
            calibrated["atp_total"] / generic["atp_total"] if generic["atp_total"] > 0 else 0.0
        )

        kappa_diff = calibrated["kappa_landauer"] - generic.get("kappa_landauer", 0)

        return {
            "calibrated": calibrated,
            "generic": generic,
            "atp_difference": float(atp_diff),
            "atp_ratio": float(atp_ratio),
            "kappa_difference": float(kappa_diff),
            "calibration_source": self.coefficients.calibration_source,
            "c_1": self.c_1,
            "c_2": self.c_2,
        }

    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration state and recent computations.

        Returns:
            Summary dictionary with coefficient values and statistics
        """
        summary: Dict[str, Any] = {
            "calibration": {
                "c_1_dynamic": self.c_1,
                "c_2_static": self.c_2,
                "c_1_uncertainty": self.c_1_uncertainty,
                "c_2_uncertainty": self.c_2_uncertainty,
                "source": self.coefficients.calibration_source,
                "temporal_resolution_ms": self.coefficients.temporal_resolution_ms,
                "dataset_doi": self.coefficients.dataset_doi,
            },
            "validation": {
                "total_baseline_per_second": self.coefficients.total_baseline_per_second,
                "cost_per_ap": self.coefficients.cost_per_action_potential,
            },
        }

        # Add computation statistics if available
        if self._calibrated_computations:
            atp_totals = [c["atp_total"] for c in self._calibrated_computations]
            kappas = [c["kappa_landauer"] for c in self._calibrated_computations]

            summary["recent_computations"] = {
                "count": len(self._calibrated_computations),
                "atp_mean": float(np.mean(atp_totals)),
                "atp_std": float(np.std(atp_totals)),
                "kappa_mean": float(np.mean(kappas)),
                "kappa_std": float(np.std(kappas)),
            }

        return summary

    def validate_against_literature(self) -> Dict[str, Any]:
        """Validate coefficients against established literature values.

        Returns:
            Validation report with comparison to Attwell & Laughlin, etc.
        """
        from apgi_framework.thermodynamic.metabolic_calibration import (
            CostCoefficientValidator,
        )

        validator = CostCoefficientValidator()

        return {
            "literature_comparison": validator.compare_to_literature(self.coefficients),
            "confidence_score": validator.compute_confidence_score(self.coefficients),
            "c_1_deviation_from_attwell": ((self.c_1 - 1.91e7) / 1.91e7 if 1.91e7 > 0 else 0.0),
            "c_2_deviation_from_attwell": ((self.c_2 * 1000 - 1e9) / 1e9 if 1e9 > 0 else 0.0),
        }


def create_calibrated_vml_from_datasets(
    two_photon_path: Optional[str] = None,
    pmrs_path: Optional[str] = None,
    two_photon_sensor: Literal["iATPSnFR2", "ATeam", "Peredox", "unknown"] = "unknown",
    pmrs_format: Literal["parrec", "csv", "twix"] = "csv",
    workspace_neurons: int = 100_000,
) -> CalibratedVirtualMetabolicLayer:
    """Convenience factory to create calibrated VML from datasets.

    Args:
        two_photon_path: Path to Two-photon dataset (or None for literature c_1)
        pmrs_path: Path to P-MRS dataset (or None for literature c_2)
        two_photon_sensor: Sensor type used in Two-photon recording
        pmrs_format: Format of P-MRS data
        workspace_neurons: Number of neurons in workspace

    Returns:
        Configured CalibratedVirtualMetabolicLayer

    Example:
        >>> vml = create_calibrated_vml_from_datasets(
        ...     two_photon_path="./iatpsnfr2_traces.csv",
        ...     pmrs_path="./fmrs_flux.csv",
        ... )
        >>> print(f"Calibrated: c_1={vml.c_1:.2e}, c_2={vml.c_2:.2e}")
    """
    # Create calibrator
    calibrator = MetabolicCalibrator()

    # Load Two-photon data if provided
    if two_photon_path is not None:
        calibrator.load_two_photon_dataset(two_photon_path, two_photon_sensor)

    # Load P-MRS data if provided
    if pmrs_path is not None:
        calibrator.load_pmrs_dataset(pmrs_path, pmrs_format)

    # Create VML with calibrator
    return CalibratedVirtualMetabolicLayer(
        calibrator=calibrator,
        neural_params=None,  # Use defaults
    )


__all__ = [
    "CalibratedVirtualMetabolicLayer",
    "create_calibrated_vml_from_datasets",
]
