"""Prediction error processing module for the APGI Framework.

This module implements prediction error processing and z-score standardization
for both exteroceptive and interoceptive prediction errors used in the APGI equation.
"""

import warnings
from typing import Optional, Tuple, Union

import numpy as np

from ..exceptions import MathematicalError


class PredictionErrorProcessor:
    """Processor for exteroceptive and interoceptive prediction errors.

    Handles z-score standardization, validation, and preprocessing of
    prediction errors before they are used in the APGI equation.
    """

    def __init__(self, standardize: bool = True, outlier_threshold: float = 3.0):
        """Initialize prediction error processor.

        Args:
            standardize: Whether to apply z-score standardization
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.standardize = standardize
        self.outlier_threshold = outlier_threshold

    def process_exteroceptive_error(
        self,
        error: Union[float, np.ndarray],
        reference_mean: Optional[float] = None,
        reference_std: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """Process exteroceptive prediction error with standardization.

        Args:
            error: Raw exteroceptive prediction error(s)
            reference_mean: Reference mean for standardization (if None, uses error mean)
            reference_std: Reference std for standardization (if None, uses error std)

        Returns:
            Processed exteroceptive prediction error(s)

        Raises:
            MathematicalError: If standardization parameters are invalid
        """
        error = np.asarray(error)

        if self.standardize:
            # Handle single values - don't standardize if only one value
            if error.size == 1 and reference_mean is None and reference_std is None:
                warnings.warn(
                    "Cannot standardize single value without reference parameters, returning raw error"
                )
                return float(error.item()) if error.size == 1 else error

            if reference_mean is None:
                reference_mean = np.mean(error)
            if reference_std is None:
                reference_std = np.std(error, ddof=1) if error.size > 1 else 1.0

            if reference_std <= 0:
                warnings.warn("Standard deviation is zero or negative, using raw error")
                return error

            # Z-score standardization
            standardized_error = (error - reference_mean) / reference_std

            # Check for outliers
            if self.outlier_threshold > 0:
                outlier_mask = np.abs(standardized_error) > self.outlier_threshold
                if np.any(outlier_mask):
                    warnings.warn(f"Found {np.sum(outlier_mask)} outliers in exteroceptive errors")

            return (
                float(standardized_error.item())
                if standardized_error.size == 1
                else standardized_error
            )
        else:
            return error

    def process_interoceptive_error(
        self,
        error: Union[float, np.ndarray],
        reference_mean: Optional[float] = None,
        reference_std: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """Process interoceptive prediction error with standardization.

        Args:
            error: Raw interoceptive prediction error(s)
            reference_mean: Reference mean for standardization (if None, uses error mean)
            reference_std: Reference std for standardization (if None, uses error std)

        Returns:
            Processed interoceptive prediction error(s)

        Raises:
            MathematicalError: If standardization parameters are invalid
        """
        error = np.asarray(error)

        if self.standardize:
            # Handle single values - don't standardize if only one value
            if error.size == 1 and reference_mean is None and reference_std is None:
                warnings.warn(
                    "Cannot standardize single value without reference parameters, returning raw error"
                )
                return float(error.item()) if error.size == 1 else error

            if reference_mean is None:
                reference_mean = np.mean(error)
            if reference_std is None:
                reference_std = np.std(error, ddof=1) if error.size > 1 else 1.0

            if reference_std <= 0:
                warnings.warn("Standard deviation is zero or negative, using raw error")
                return error

            # Z-score standardization
            standardized_error = (error - reference_mean) / reference_std

            # Check for outliers
            if self.outlier_threshold > 0:
                outlier_mask = np.abs(standardized_error) > self.outlier_threshold
                if np.any(outlier_mask):
                    warnings.warn(f"Found {np.sum(outlier_mask)} outliers in interoceptive errors")

            return (
                float(standardized_error.item())
                if standardized_error.size == 1
                else standardized_error
            )
        else:
            return error

    def validate_error_pair(self, extero_error: float, intero_error: float) -> Tuple[bool, str]:
        """Validate a pair of exteroceptive and interoceptive prediction errors.

        Args:
            extero_error: Exteroceptive prediction error
            intero_error: Interoceptive prediction error

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check for NaN or infinite values
            if not np.isfinite(extero_error):
                return False, f"Exteroceptive error is not finite: {extero_error}"

            if not np.isfinite(intero_error):
                return False, f"Interoceptive error is not finite: {intero_error}"

            # Check for extreme outliers (if standardized, should be roughly within [-5, 5])
            if self.standardize:
                if abs(extero_error) > 5:
                    return (
                        False,
                        f"Exteroceptive error {extero_error} is an extreme outlier",
                    )

                if abs(intero_error) > 5:
                    return (
                        False,
                        f"Interoceptive error {intero_error} is an extreme outlier",
                    )

            return True, "Valid error pair"

        except (TypeError, ValueError) as e:
            return False, f"Invalid error values: {e}"

    def calculate_error_statistics(self, errors: np.ndarray) -> dict:
        """Calculate descriptive statistics for prediction errors.

        Args:
            errors: Array of prediction errors

        Returns:
            Dictionary with error statistics
        """
        errors = np.asarray(errors)

        if len(errors) == 0:
            raise MathematicalError("Error array cannot be empty")

        stats = {
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors, ddof=1)),
            "median": float(np.median(errors)),
            "min": float(np.min(errors)),
            "max": float(np.max(errors)),
            "n_samples": len(errors),
            "n_outliers": 0,
        }

        # Count outliers if standardization is enabled
        if self.standardize and self.outlier_threshold > 0:
            z_scores = np.abs((errors - stats["mean"]) / stats["std"])
            stats["n_outliers"] = int(np.sum(z_scores > self.outlier_threshold))

        return stats

    def get_processor_info(self) -> dict:
        """Get information about the prediction error processor configuration.

        Returns:
            Dictionary with processor settings
        """
        return {
            "standardize": self.standardize,
            "outlier_threshold": self.outlier_threshold,
            "processing_methods": ["z_score_standardization", "outlier_detection"],
            "validation_checks": ["finite_values", "extreme_outliers"],
        }
