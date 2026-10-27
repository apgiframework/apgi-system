"""Comprehensive Input Validation Module for APGI Framework.

Provides centralized input validation with security considerations,
type checking, and user-friendly error messages.
"""

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from apgi_framework.logging.standardized_logging import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message: str, field_name: Optional[str] = None, value: Any = None):
        self.field_name = field_name
        self.value = value
        super().__init__(message)


class InputValidator:
    """Comprehensive input validation utility.

    Provides validation for common input types with security considerations
    and detailed error reporting.
    """

    # Security patterns for input sanitization
    DANGEROUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # XSS
        r"javascript:",  # JavaScript URLs
        r"on\w+\s*=",  # Event handlers
        r"expression\s*\(",  # CSS expressions
        r"@import",  # CSS imports
        r"union\s+select",  # SQL injection
        r"drop\s+table",  # SQL injection
        r"insert\s+into",  # SQL injection
        r"delete\s+from",  # SQL injection
    ]

    def __init__(self) -> None:
        """Initialize validator with compiled patterns."""
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.DANGEROUS_PATTERNS
        ]

    def validate_string(
        self,
        value: Any,
        field_name: str = "input",
        min_length: int = 0,
        max_length: int = 1000,
        allowed_chars: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> str:
        """Validate string input with comprehensive checks.

        Args:
            value: Input value to validate
            field_name: Name of the field for error messages
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            allowed_chars: String of allowed characters (whitelist)
            pattern: Regex pattern to match

        Returns:
            Validated string

        Raises:
            ValidationError: If validation fails
        """
        if value is None:
            raise ValidationError(f"{field_name} cannot be None", field_name, value)

        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", field_name, value)

        # Length validation
        if len(value) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters long",
                field_name,
                value,
            )

        if len(value) > max_length:
            raise ValidationError(
                f"{field_name} must be no more than {max_length} characters long",
                field_name,
                value,
            )

        # Security checks
        self._check_security(value, field_name)

        # Character whitelist
        if allowed_chars:
            if not all(char in allowed_chars for char in value):
                raise ValidationError(
                    f"{field_name} contains invalid characters", field_name, value
                )

        # Pattern matching
        if pattern:
            if not re.match(pattern, value):
                raise ValidationError(
                    f"{field_name} does not match required pattern", field_name, value
                )

        return value.strip()

    def validate_integer(
        self,
        value: Any,
        field_name: str = "input",
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> int:
        """Validate integer input.

        Args:
            value: Input value to validate
            field_name: Name of the field for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated integer

        Raises:
            ValidationError: If validation fails
        """
        if value is None:
            raise ValidationError(f"{field_name} cannot be None", field_name, value)

        try:
            int_value = int(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a valid integer", field_name, value)

        if min_value is not None and int_value < min_value:
            raise ValidationError(f"{field_name} must be at least {min_value}", field_name, value)

        if max_value is not None and int_value > max_value:
            raise ValidationError(
                f"{field_name} must be no more than {max_value}", field_name, value
            )

        return int_value

    def validate_float(
        self,
        value: Any,
        field_name: str = "input",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        """Validate float input.

        Args:
            value: Input value to validate
            field_name: Name of the field for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated float

        Raises:
            ValidationError: If validation fails
        """
        if value is None:
            raise ValidationError(f"{field_name} cannot be None", field_name, value)

        try:
            float_value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a valid number", field_name, value)

        if min_value is not None and float_value < min_value:
            raise ValidationError(f"{field_name} must be at least {min_value}", field_name, value)

        if max_value is not None and float_value > max_value:
            raise ValidationError(
                f"{field_name} must be no more than {max_value}", field_name, value
            )

        return float_value

    def validate_path(
        self,
        value: Any,
        field_name: str = "input",
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        create_if_missing: bool = False,
    ) -> Path:
        """Validate file path input.

        Args:
            value: Input value to validate
            field_name: Name of the field for error messages
            must_exist: Path must exist
            must_be_file: Path must be a file
            must_be_dir: Path must be a directory
            create_if_missing: Create directory if it doesn't exist

        Returns:
            Validated Path object

        Raises:
            ValidationError: If validation fails
        """
        if value is None:
            raise ValidationError(f"{field_name} cannot be None", field_name, value)

        try:
            path = Path(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a valid path", field_name, value)

        # Security check for path traversal
        self._check_path_security(path, field_name)

        if must_exist and not path.exists():
            if create_if_missing and not must_be_file:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise ValidationError(
                        f"Cannot create directory {field_name}: {e}", field_name, value
                    )
            else:
                raise ValidationError(f"{field_name} does not exist", field_name, value)

        if path.exists():
            if must_be_file and not path.is_file():
                raise ValidationError(f"{field_name} must be a file", field_name, value)

            if must_be_dir and not path.is_dir():
                raise ValidationError(f"{field_name} must be a directory", field_name, value)

        return path

    def validate_email(self, value: Any, field_name: str = "email") -> str:
        """Validate email address.

        Args:
            value: Input value to validate
            field_name: Name of the field for error messages

        Returns:
            Validated email string

        Raises:
            ValidationError: If validation fails
        """
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        email = self.validate_string(
            value, field_name, min_length=5, max_length=254, pattern=email_pattern
        )

        return email.lower()

    def validate_url(self, value: Any, field_name: str = "url") -> str:
        """Validate URL.

        Args:
            value: Input value to validate
            field_name: Name of the field for error messages

        Returns:
            Validated URL string

        Raises:
            ValidationError: If validation fails
        """
        url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"

        url = self.validate_string(
            value, field_name, min_length=10, max_length=2048, pattern=url_pattern
        )

        return url

    def validate_choice(self, value: Any, choices: List[str], field_name: str = "choice") -> str:
        """Validate choice from allowed options.

        Args:
            value: Input value to validate
            choices: List of allowed choices
            field_name: Name of the field for error messages

        Returns:
            Validated choice

        Raises:
            ValidationError: If validation fails
        """
        if value is None:
            raise ValidationError(f"{field_name} cannot be None", field_name, value)

        str_value = str(value)

        if str_value not in choices:
            raise ValidationError(
                f"{field_name} must be one of: {', '.join(choices)}", field_name, value
            )

        return str_value

    def validate_list(
        self,
        value: Any,
        field_name: str = "list",
        item_validator: Optional[Callable[[Any], Any]] = None,
        min_length: int = 0,
        max_length: int = 100,
    ) -> List[Any]:
        """Validate list input.

        Args:
            value: Input value to validate
            field_name: Name of the field for error messages
            item_validator: Function to validate each item
            min_length: Minimum list length
            max_length: Maximum list length

        Returns:
            Validated list

        Raises:
            ValidationError: If validation fails
        """
        if value is None:
            raise ValidationError(f"{field_name} cannot be None", field_name, value)

        if isinstance(value, str):
            # Try to parse as comma-separated
            try:
                value = [item.strip() for item in value.split(",")]
            except Exception:
                raise ValidationError(f"{field_name} must be a list", field_name, value)

        if not isinstance(value, (list, tuple)):
            raise ValidationError(f"{field_name} must be a list", field_name, value)

        if len(value) < min_length:
            raise ValidationError(
                f"{field_name} must contain at least {min_length} items",
                field_name,
                value,
            )

        if len(value) > max_length:
            raise ValidationError(
                f"{field_name} must contain no more than {max_length} items",
                field_name,
                value,
            )

        # Validate items if validator provided
        if item_validator:
            validated_items = []
            for i, item in enumerate(value):
                try:
                    validated_item = item_validator(item)
                    validated_items.append(validated_item)
                except ValidationError as e:
                    raise ValidationError(f"{field_name}[{i}]: {str(e)}", field_name, value)
            return validated_items

        return list(value)

    def validate_json(self, value: Any, field_name: str = "json") -> Dict[str, Any]:
        """Validate JSON input.

        Args:
            value: Input value to validate
            field_name: Name of the field for error messages

        Returns:
            Validated JSON dictionary

        Raises:
            ValidationError: If validation fails
        """
        import json

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValidationError(f"{field_name} must be valid JSON: {e}", field_name, value)
        elif isinstance(value, dict):
            parsed = value
        else:
            raise ValidationError(f"{field_name} must be JSON", field_name, value)

        if not isinstance(parsed, dict):
            raise ValidationError(f"{field_name} must be a JSON object", field_name, value)

        return parsed

    def _check_security(self, value: str, field_name: str) -> None:
        """Check for security issues in string input."""
        for pattern in self.compiled_patterns:
            if pattern.search(value):
                raise ValidationError(
                    f"{field_name} contains potentially dangerous content",
                    field_name,
                    value,
                )

    def _check_path_security(self, path: Path, field_name: str) -> None:
        """Check for path traversal security issues."""
        # Convert to string and check for dangerous patterns
        path_str = str(path)

        dangerous_patterns = ["..", "~/", "/etc/", "/var/", "/sys/", "/proc/"]

        for pattern in dangerous_patterns:
            if pattern in path_str:
                raise ValidationError(
                    f"{field_name} contains potentially dangerous path",
                    field_name,
                    path_str,
                )


# Global validator instance
validator = InputValidator()


# Convenience functions
def validate_string(value: Any, field_name: str = "input", **kwargs: Any) -> str:
    """Convenience function for string validation."""
    return validator.validate_string(value, field_name, **kwargs)


def validate_integer(value: Any, field_name: str = "input", **kwargs: Any) -> int:
    """Convenience function for integer validation."""
    return validator.validate_integer(value, field_name, **kwargs)


def validate_float(value: Any, field_name: str = "input", **kwargs: Any) -> float:
    """Convenience function for float validation."""
    return validator.validate_float(value, field_name, **kwargs)


def validate_path(value: Any, field_name: str = "input", **kwargs: Any) -> Path:
    """Convenience function for path validation."""
    return validator.validate_path(value, field_name, **kwargs)


def validate_email(value: Any, field_name: str = "email") -> str:
    """Convenience function for email validation."""
    return validator.validate_email(value, field_name)


def validate_choice(value: Any, choices: List[str], field_name: str = "choice") -> str:
    """Convenience function for choice validation."""
    return validator.validate_choice(value, choices, field_name)


def safe_input(
    prompt: str,
    validator_func: Optional[Callable[..., Any]] = None,
    **validator_kwargs: Any,
) -> Optional[Any]:
    """Safe input function with validation.

    Args:
        prompt: Input prompt to display
        validator_func: Validation function to use
        **validator_kwargs: Arguments to pass to validator

    Returns:
        Validated input
    """
    while True:
        try:
            value = input(prompt)
            if validator_func:
                return validator_func(value, **validator_kwargs)
            return value
        except ValidationError as e:
            logger.info(f"Validation error: {e}")
        except KeyboardInterrupt:
            logger.info("\nInput cancelled.")
            return None
        except Exception:
            return None
        except EOFError:
            logger.info("\nInput ended.")
            return None
