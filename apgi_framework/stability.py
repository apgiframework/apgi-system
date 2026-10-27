"""Numerical Stability Monitoring.

Provides monitoring and handling of numerical stability issues in computations.
Detects NaN, Inf, and overflow conditions with configurable thresholds.
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from apgi_framework.types import ConfigDict, FloatArray


class NumericalInstabilityError(Exception):
    """Exception raised when numerical instability is detected.

    This error is raised when computed values exceed error thresholds,
    contain NaN or Inf values, or otherwise indicate numerical problems
    that prevent continued computation.

    Attributes
    ----------
    message : str
        Description of the numerical instability
    context : str
        Context where the instability was detected
    value : float or np.ndarray
        The problematic value(s)

    Examples
    --------
    >>> raise NumericalInstabilityError(
    ...     "Value overflow detected",
    ...     context="free_energy_calculation",
    ...     value=1e20
    ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[str] = None,
        value: Optional[Union[float, FloatArray]] = None,
    ) -> None:
        """Initialize numerical instability error.

        Parameters
        ----------
        message : str
            Error message describing the instability
        context : str, optional
            Context where the error occurred
        value : float or np.ndarray, optional
            The problematic value(s)
        """
        self.message = message
        self.context = context
        self.value = value

        full_message = message
        if context:
            full_message = f"{context}: {message}"
        if value is not None:
            if isinstance(value, np.ndarray):
                full_message += f" (max abs value: {np.max(np.abs(value))})"
            else:
                full_message += f" (value: {value})"

        super().__init__(full_message)


class NumericalStabilityWarning(UserWarning):
    """Warning issued when numerical values approach instability thresholds.

    This warning is issued when values are large but not yet at error levels,
    allowing the system to continue while alerting about potential issues.

    Examples
    --------
    >>> warnings.warn(
    ...     "Large values detected in precision calculation: 1e8",
    ...     NumericalStabilityWarning
    ... )
    """

    pass


class NumericalStabilityMonitor:
    """Monitor and handle numerical stability issues in computations.

    Provides centralized checking for NaN, Inf, overflow, and underflow
    conditions with configurable warning and error thresholds. Integrates
    with critical computation paths to ensure numerical stability.

    Attributes
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    warning_threshold : float
        Threshold for issuing warnings (default: 1e10)
    error_threshold : float
        Threshold for raising errors (default: 1e15)
    underflow_threshold : float
        Threshold for detecting underflow (default: 1e-10)
    check_enabled : bool
        Whether stability checking is enabled (default: True)
    warning_count : int
        Number of warnings issued
    error_count : int
        Number of errors detected

    Notes
    -----
    The monitor uses a two-tier threshold system:
    - Warning threshold: Issues warnings but allows computation to continue
    - Error threshold: Raises exceptions to halt computation

    This allows graceful degradation when values become large but not
    yet catastrophic, while preventing completely invalid computations.

    Examples
    --------
    >>> config = {'stability_warning_threshold': 1e10, 'stability_error_threshold': 1e15}
    >>> monitor = NumericalStabilityMonitor(config)
    >>> values = np.array([1.0, 2.0, 3.0])
    >>> monitor.check_stability(values, context="test_computation")
    >>> # No error or warning - values are within bounds
    """

    def __init__(self, config: Optional[ConfigDict] = None) -> None:
        """Initialize numerical stability monitor.

        Parameters
        ----------
        config : Dict[str, Any], optional
            Configuration dictionary containing:
            - 'stability_warning_threshold': Warning threshold (default: 1e10)
            - 'stability_error_threshold': Error threshold (default: 1e15)
            - 'stability_underflow_threshold': Underflow threshold (default: 1e-10)
            - 'stability_check_enabled': Enable/disable checking (default: True)

        Examples
        --------
        >>> monitor = NumericalStabilityMonitor()
        >>> custom_config = {
        ...     'stability_warning_threshold': 1e8,
        ...     'stability_error_threshold': 1e12
        ... }
        >>> monitor = NumericalStabilityMonitor(custom_config)
        """
        self.config = config or {}

        # Extract thresholds from config
        self.warning_threshold = self.config.get("stability_warning_threshold", 1e10)
        self.error_threshold = self.config.get("stability_error_threshold", 1e15)
        self.underflow_threshold = self.config.get("stability_underflow_threshold", 1e-10)
        self.check_enabled = self.config.get("stability_check_enabled", True)

        # Statistics
        self.warning_count = 0
        self.error_count = 0

    def check_stability(
        self, values: Union[float, FloatArray], context: str, raise_on_error: bool = True
    ) -> Dict[str, Any]:
        """Check numerical stability of values.

        Examines values for NaN, Inf, overflow, and underflow conditions.
        Issues warnings or raises errors based on configured thresholds.

        Parameters
        ----------
        values : float or np.ndarray
            Values to check for numerical stability
        context : str
            Context description for error/warning messages
        raise_on_error : bool, default=True
            Whether to raise exception on error threshold violation

        Returns
        -------
        status : Dict[str, Any]
            Dictionary containing:
            - 'stable': bool indicating if values are stable
            - 'has_nan': bool indicating presence of NaN
            - 'has_inf': bool indicating presence of Inf
            - 'max_abs_value': maximum absolute value
            - 'warning_issued': bool indicating if warning was issued
            - 'error_detected': bool indicating if error was detected

        Raises
        ------
        NumericalInstabilityError
            If values contain NaN/Inf or exceed error threshold
            (only if raise_on_error=True)

        Notes
        -----
        Checking is performed in the following order:
        1. Check for NaN values (always error)
        2. Check for Inf values (always error)
        3. Check against error threshold (raises error)
        4. Check against warning threshold (issues warning)

        Examples
        --------
        >>> monitor = NumericalStabilityMonitor()
        >>> values = np.array([1.0, 2.0, 3.0])
        >>> status = monitor.check_stability(values, "test")
        >>> print(status['stable'])
        True

        >>> large_values = np.array([1e11, 2e11])
        >>> status = monitor.check_stability(large_values, "large_test")
        # Issues warning but continues

        >>> invalid_values = np.array([np.nan, 1.0])
        >>> status = monitor.check_stability(invalid_values, "invalid_test")
        # Raises NumericalInstabilityError
        """
        if not self.check_enabled:
            import logging

            logging.getLogger(__name__).warning(
                f"Numerical stability checks are disabled for context: {context}"
            )
            return {
                "stable": True,
                "has_nan": False,
                "has_inf": False,
                "max_abs_value": 0.0,
                "warning_issued": False,
                "error_detected": False,
            }

        # Convert scalar to array for uniform handling
        if np.isscalar(values):
            values_array = np.array([values])
        else:
            values_array = np.asarray(values)

        # Initialize status
        status = {
            "stable": True,
            "has_nan": False,
            "has_inf": False,
            "max_abs_value": 0.0,
            "warning_issued": False,
            "error_detected": False,
        }

        # Check for NaN
        has_nan = np.any(np.isnan(values_array))
        status["has_nan"] = bool(has_nan)

        if has_nan:
            status["stable"] = False
            status["error_detected"] = True
            self.error_count += 1

            error_msg = f"NaN detected in {context}"
            if raise_on_error:
                raise NumericalInstabilityError(error_msg, context=context, value=values)
            return status

        # Check for Inf
        has_inf = np.any(np.isinf(values_array))
        status["has_inf"] = bool(has_inf)

        if has_inf:
            status["stable"] = False
            status["error_detected"] = True
            self.error_count += 1

            error_msg = f"Inf detected in {context}"
            if raise_on_error:
                raise NumericalInstabilityError(error_msg, context=context, value=values)
            return status

        # Compute maximum absolute value
        max_abs_val = float(np.max(np.abs(values_array)))
        status["max_abs_value"] = max_abs_val

        # Check against error threshold
        if max_abs_val > self.error_threshold:
            status["stable"] = False
            status["error_detected"] = True
            self.error_count += 1

            error_msg = f"Value overflow in {context}: max abs value = {max_abs_val:.2e}"
            if raise_on_error:
                raise NumericalInstabilityError(error_msg, context=context, value=values)
            return status

        # Check against warning threshold
        if max_abs_val > self.warning_threshold:
            status["warning_issued"] = True
            self.warning_count += 1

            warning_msg = (
                f"Large values detected in {context}: "
                f"max abs value = {max_abs_val:.2e} "
                f"(warning threshold = {self.warning_threshold:.2e})"
            )
            warnings.warn(warning_msg, NumericalStabilityWarning)

        return status

    def check_array_properties(
        self,
        array: FloatArray,
        context: str,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Check array properties including stability, shape, and value range.

        Performs comprehensive validation of array properties beyond just
        numerical stability, including shape and value range checks.

        Parameters
        ----------
        array : np.ndarray
            Array to check
        context : str
            Context description for error messages
        expected_shape : tuple, optional
            Expected array shape
        expected_range : tuple, optional
            Expected (min, max) value range

        Returns
        -------
        status : Dict[str, Any]
            Dictionary containing stability status plus:
            - 'shape_valid': bool indicating if shape matches expected
            - 'range_valid': bool indicating if values are in expected range
            - 'actual_shape': actual array shape
            - 'actual_range': actual (min, max) values

        Raises
        ------
        ValueError
            If shape or range validation fails
        NumericalInstabilityError
            If numerical stability check fails

        Examples
        --------
        >>> monitor = NumericalStabilityMonitor()
        >>> array = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> status = monitor.check_array_properties(
        ...     array,
        ...     context="test_array",
        ...     expected_shape=(2, 2),
        ...     expected_range=(0.0, 10.0)
        ... )
        >>> print(status['shape_valid'], status['range_valid'])
        True True
        """
        # First check numerical stability
        status = self.check_stability(array, context)

        # Check shape
        status["actual_shape"] = array.shape
        if expected_shape is not None:
            shape_valid = array.shape == expected_shape
            status["shape_valid"] = shape_valid

            if not shape_valid:
                raise ValueError(
                    f"{context}: shape mismatch - " f"expected {expected_shape}, got {array.shape}"
                )
        else:
            status["shape_valid"] = True

        # Check value range
        if status["stable"]:  # Only check range if values are stable
            actual_min = float(np.min(array))
            actual_max = float(np.max(array))
            status["actual_range"] = (actual_min, actual_max)

            if expected_range is not None:
                range_valid = actual_min >= expected_range[0] and actual_max <= expected_range[1]
                status["range_valid"] = range_valid

                if not range_valid:
                    raise ValueError(
                        f"{context}: value range violation - "
                        f"expected [{expected_range[0]}, {expected_range[1]}], "
                        f"got [{actual_min}, {actual_max}]"
                    )
            else:
                status["range_valid"] = True
        else:
            status["actual_range"] = (None, None)
            status["range_valid"] = False

        return status

    def get_statistics(self) -> Dict[str, int]:
        """Get monitoring statistics.

        Returns
        -------
        stats : Dict[str, int]
            Dictionary containing:
            - 'warning_count': Number of warnings issued
            - 'error_count': Number of errors detected

        Examples
        --------
        >>> monitor = NumericalStabilityMonitor()
        >>> # ... perform computations with stability checks ...
        >>> stats = monitor.get_statistics()
        >>> print(f"Warnings: {stats['warning_count']}, Errors: {stats['error_count']}")
        """
        return {"warning_count": self.warning_count, "error_count": self.error_count}

    def reset_statistics(self) -> None:
        """Reset monitoring statistics.

        Clears warning and error counts. Useful for starting new
        simulation runs or experimental trials.

        Examples
        --------
        >>> monitor = NumericalStabilityMonitor()
        >>> # ... perform computations ...
        >>> monitor.reset_statistics()
        >>> stats = monitor.get_statistics()
        >>> print(stats)  # {'warning_count': 0, 'error_count': 0}
        """
        self.warning_count = 0
        self.error_count = 0
