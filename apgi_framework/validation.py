"""Input Validation Infrastructure.

Provides centralized input validation for the APGI system to ensure
data integrity and catch errors early.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np


class InputValidator:
    """Centralized input validation for APGI system components.

    This class provides static methods for validating arrays and scalars,
    including shape checking, range validation, and NaN/Inf detection.

    Examples
    --------
    >>> validator = InputValidator()
    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> InputValidator.validate_array(arr, "test_array", expected_shape=(3,))
    >>> InputValidator.validate_scalar(1.5, "test_value", positive=True)
    """

    @staticmethod
    def validate_array(
        arr: np.ndarray[Any, np.dtype[Any]],
        name: str,
        expected_shape: Optional[Tuple[int, ...]] = None,
        value_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Validate numpy array inputs.

        Checks that the input is a numpy array, optionally validates its shape
        and value range, and ensures no NaN or Inf values are present.

        Parameters
        ----------
        arr : np.ndarray
            The array to validate
        name : str
            Name of the parameter (for error messages)
        expected_shape : Optional[Tuple[int, ...]], optional
            Expected shape of the array. If None, shape is not checked.
        value_range : Optional[Tuple[float, float]], optional
            Tuple of (min_value, max_value) for range checking.
            If None, range is not checked.

        Raises
        ------
        TypeError
            If arr is not a numpy array
        ValueError
            If shape doesn't match, values are out of range, or NaN/Inf detected

        Examples
        --------
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> InputValidator.validate_array(arr, "data", expected_shape=(3,))
        >>> InputValidator.validate_array(arr, "data", value_range=(0.0, 10.0))
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be numpy array, got {type(arr)}")

        if expected_shape and arr.shape != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: expected {expected_shape}, " f"got {arr.shape}"
            )

        if value_range:
            if np.any(arr < value_range[0]) or np.any(arr > value_range[1]):
                raise ValueError(
                    f"{name} values must be in range {value_range}, "
                    f"got min={arr.min()}, max={arr.max()}"
                )

        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError(f"{name} contains NaN or Inf values")

    @staticmethod
    def validate_scalar(
        value: Union[int, float, np.number],
        name: str,
        value_range: Optional[Tuple[float, float]] = None,
        positive: bool = False,
    ) -> None:
        """Validate scalar inputs.

        Checks that the input is numeric, optionally validates its range
        and positivity, and ensures it's not NaN or Inf.

        Parameters
        ----------
        value : Union[int, float, np.number]
            The scalar value to validate
        name : str
            Name of the parameter (for error messages)
        value_range : Optional[Tuple[float, float]], optional
            Tuple of (min_value, max_value) for range checking.
            If None, range is not checked.
        positive : bool, optional
            If True, value must be strictly positive (> 0), by default False

        Raises
        ------
        TypeError
            If value is not numeric
        ValueError
            If value is out of range, not positive when required, or is NaN/Inf

        Examples
        --------
        >>> InputValidator.validate_scalar(1.5, "alpha", positive=True)
        >>> InputValidator.validate_scalar(0.5, "probability", value_range=(0.0, 1.0))
        """
        if not isinstance(value, (int, float, np.number)):
            raise TypeError(f"{name} must be numeric, got {type(value)}")

        if np.isnan(value) or np.isinf(value):
            raise ValueError(f"{name} is NaN or Inf")

        if positive and value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

        if value_range:
            if value < value_range[0] or value > value_range[1]:
                raise ValueError(f"{name} must be in range {value_range}, got {value}")
