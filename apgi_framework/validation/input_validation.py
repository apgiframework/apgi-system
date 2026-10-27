"""Input validation utilities for the APGI Framework.

Provides common validation functions for user input, parameters,
and data to ensure security and data integrity.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..logging.standardized_logging import get_logger

# Initialize logger
logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""


def validate_file_path(
    file_path: Union[str, Path],
    must_exist: bool = True,
    allowed_extensions: Optional[List[str]] = None,
) -> Path:
    """Validate file path input.

    Args:
        file_path: Input file path
        must_exist: Whether file must exist
        allowed_extensions: List of allowed file extensions

    Returns:
        Validated Path object

    Raises:
        ValidationError: If validation fails
    """
    try:
        path = Path(file_path)

        # Check for path traversal attempts
        if ".." in str(path):
            raise ValidationError("Path traversal not allowed")

        # Check file existence if required
        if must_exist and not path.exists():
            raise ValidationError(f"File does not exist: {path}")

        # Check file extension if specified
        if allowed_extensions:
            if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                raise ValidationError(
                    f"File extension {path.suffix} not allowed. "
                    f"Allowed: {', '.join(allowed_extensions)}"
                )

        return path

    except (TypeError, ValueError) as e:
        raise ValidationError(f"Invalid file path: {e}")


def validate_numeric_input(
    value: Any,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_float: bool = True,
) -> float:
    """Validate numeric input.

    Args:
        value: Input value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        allow_float: Whether to allow floating point numbers

    Returns:
        Validated numeric value

    Raises:
        ValidationError: If validation fails
    """
    try:
        if allow_float:
            num_value = float(value)
        else:
            num_value = int(float(value))
            if float(value) != num_value:
                raise ValidationError("Integer required")

        # Check bounds
        if min_val is not None and num_value < min_val:
            raise ValidationError(f"Value {num_value} below minimum {min_val}")

        if max_val is not None and num_value > max_val:
            raise ValidationError(f"Value {num_value} above maximum {max_val}")

        return num_value

    except (TypeError, ValueError) as e:
        raise ValidationError(f"Invalid numeric input: {e}")


def validate_string_input(
    value: Any,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allowed_chars: Optional[str] = None,
    forbidden_patterns: Optional[List[str]] = None,
) -> str:
    """Validate string input.

    Args:
        value: Input value to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        allowed_chars: Regex pattern of allowed characters
        forbidden_patterns: List of regex patterns to forbid

    Returns:
        Validated string

    Raises:
        ValidationError: If validation fails
    """
    try:
        str_value = str(value)

        # Check length constraints
        if min_length is not None and len(str_value) < min_length:
            raise ValidationError(f"String length {len(str_value)} below minimum {min_length}")

        if max_length is not None and len(str_value) > max_length:
            raise ValidationError(f"String length {len(str_value)} above maximum {max_length}")

        # Check allowed characters
        if allowed_chars is not None:
            if not re.fullmatch(allowed_chars, str_value):
                raise ValidationError("String contains invalid characters")

        # Check forbidden patterns
        if forbidden_patterns is not None:
            for pattern in forbidden_patterns:
                if re.search(pattern, str_value, re.IGNORECASE):
                    raise ValidationError(f"String contains forbidden pattern: {pattern}")

        return str_value

    except (TypeError, ValueError) as e:
        raise ValidationError(f"Invalid string input: {e}")


def validate_experiment_parameters(params: dict) -> Dict[str, Any]:
    """Validate experiment parameters.

    Args:
        params: Dictionary of experiment parameters

    Returns:
        Validated parameters dictionary

    Raises:
        ValidationError: If validation fails
    """
    validated_params: Dict[str, Any] = {}

    # Validate common parameters
    if "n_participants" in params:
        validated_params["n_participants"] = validate_numeric_input(
            params["n_participants"], min_val=1, max_val=1000, allow_float=False
        )

    if "n_trials" in params:
        validated_params["n_trials"] = validate_numeric_input(
            params["n_trials"], min_val=1, max_val=10000, allow_float=False
        )

    # Validate APGI parameters
    apgi_params = [
        "extero_precision",
        "intero_precision",
        "somatic_gain",
        "threshold",
        "steepness",
    ]

    for param in apgi_params:
        if param in params:
            validated_params[param] = validate_numeric_input(
                params[param], min_val=0.001, max_val=100.0
            )

    # Copy any other parameters (with basic validation)
    for key, value in params.items():
        if key not in validated_params:
            if isinstance(value, (str, int, float, bool, list)):
                validated_params[key] = value
            else:
                raise ValidationError(f"Invalid parameter type for {key}: {type(value)}")

    return validated_params


def validate_confirmation_input(
    prompt: str,
    valid_yes: Optional[List[str]] = None,
    valid_no: Optional[List[str]] = None,
) -> bool:
    """Validate user confirmation input with retry logic.

    Args:
        prompt: Prompt message to display
        valid_yes: List of valid "yes" responses
        valid_no: List of valid "no" responses

    Returns:
        True for yes, False for no

    Raises:
        ValidationError: If input cannot be parsed after retries
    """
    if valid_yes is None:
        valid_yes = ["y", "yes", "Y", "YES"]
    if valid_no is None:
        valid_no = ["n", "no", "N", "NO"]

    max_attempts = 3
    attempts = 0

    while attempts < max_attempts:
        try:
            response = input(prompt).strip()

            if response in valid_yes:
                return True
            elif response in valid_no:
                return False
            else:
                attempts += 1
                remaining = max_attempts - attempts
                if remaining > 0:
                    logger.warning(
                        f"Invalid input. Please enter 'y' or 'n'. {remaining} attempts remaining."
                    )
                else:
                    raise ValidationError("Too many invalid attempts")

        except (EOFError, KeyboardInterrupt):
            raise ValidationError("Input interrupted")

    raise ValidationError("Failed to get valid confirmation")


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and invalid characters.

    Args:
        filename: Input filename

    Returns:
        Sanitized filename

    Raises:
        ValidationError: If filename cannot be sanitized
    """
    try:
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
        sanitized = re.sub(r"\.\.", "_", sanitized)  # Prevent path traversal
        sanitized = sanitized.strip(". ")  # Remove leading/trailing dots and spaces

        if not sanitized or sanitized in [".", ".."]:
            raise ValidationError("Invalid filename after sanitization")

        # Limit length
        if len(sanitized) > 255:
            sanitized = sanitized[:255]

        return sanitized

    except (TypeError, ValueError) as e:
        raise ValidationError(f"Cannot sanitize filename: {e}")


# Common validation patterns
SAFE_STRING_PATTERN = r"^[a-zA-Z0-9_\-\.]+$"
EXPERIMENT_NAME_PATTERN = r"^[a-zA-Z0-9_\-]{3,50}$"
PARTICIPANT_ID_PATTERN = r"^[a-zA-Z0-9_\-]{1,20}$"

# Forbidden patterns for security
FORBIDDEN_PATTERNS = [
    r"<script.*?>.*?</script>",  # Script tags
    r"javascript:",  # JavaScript URLs
    r"data:",  # Data URLs
    r"\.\./.*",  # Path traversal
    r"\.\.\\.*",  # Windows path traversal
]
