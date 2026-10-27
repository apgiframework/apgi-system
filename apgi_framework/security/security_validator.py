"""Security validation and sanitization utilities for APGI Framework.

Provides centralized security functions to validate inputs, sanitize data,
and prevent common security vulnerabilities including:
- Code injection attacks
- Unsafe pickle usage
- Path traversal
- Command injection
"""

import ast
import logging
import re
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Centralized security validation for the APGI Framework.

    Provides validation functions to prevent common security vulnerabilities
    and ensure safe data handling throughout the framework.
    """

    # Dangerous patterns that should never be allowed
    DANGEROUS_PATTERNS = {
        "code_injection": [
            "__import__",
            "eval(",
            "exec(",
            "compile(",
            "subprocess",
            "os.system",
            "os.popen",
            "os.spawn",
            "commands.getoutput",
            "popen",
            "call",
            "check_output",
            "input(",
            "raw_input(",
            "open(",
            "file(",
        ],
        "pickle_attacks": [
            "pickle.loads",
            "pickle.load",
            "pickle.dump",
            "pickle.dumps",
            "__reduce__",
            "__reduce_ex__",
            "__getstate__",
            "__setstate__",
        ],
        "file_operations": [
            "open(",
            "file(",
            "read(",
            "write(",
            "remove(",
            "unlink(",
            "rmdir(",
            "mkdir(",
            "makedirs(",
            "chmod(",
            "chown(",
        ],
        "network_operations": [
            "socket",
            "urllib",
            "http",
            "ftplib",
            "smtplib",
            "telnetlib",
            "requests",
            "httplib",
            "urlparse",
        ],
    }

    # Safe types for data validation
    SAFE_TYPES = {
        str,
        int,
        float,
        bool,
        type(None),
        list,
        tuple,
        dict,
        set,
        frozenset,
        bytes,
        bytearray,
    }

    def __init__(self, strict_mode: bool = True):
        """Initialize security validator.

        Args:
            strict_mode: If True, rejects any suspicious content. If False, logs warnings.
        """
        self.strict_mode = strict_mode
        self.validation_cache: Dict[str, Any] = {}

    def validate_string_input(self, input_string: str, context: str = "unknown") -> bool:
        """Validate string input for dangerous patterns.

        Args:
            input_string: String to validate
            context: Context description for logging

        Returns:
            True if safe, raises SecurityError if dangerous

        Raises:
            SecurityError: If dangerous patterns are detected
        """
        input_lower = input_string.lower()

        # Check for dangerous patterns
        for category, patterns in self.DANGEROUS_PATTERNS.items():
            for pattern in patterns:
                if pattern in input_lower:
                    error_msg = f"Dangerous pattern '{pattern}' detected in {context}"
                    if self.strict_mode:
                        logger.error(error_msg)
                        raise SecurityError(error_msg)
                    else:
                        logger.warning(error_msg)
                        return False

        return True

    def validate_data_structure(
        self, data: Any, max_depth: int = 10, current_depth: int = 0
    ) -> bool:
        """Recursively validate data structures for safety.

        Args:
            data: Data structure to validate
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth

        Returns:
            True if safe, raises SecurityError if dangerous
        """
        if current_depth > max_depth:
            error_msg = f"Data structure too deep (max depth: {max_depth})"
            if self.strict_mode:
                raise SecurityError(error_msg)
            else:
                logger.warning(error_msg)
                return False

        # Check basic types
        if isinstance(data, (int, float, bool, type(None), bytes, bytearray)):
            return True
        # Check strings for dangerous content
        if isinstance(data, str):
            return self.validate_string_input(data, "data structure validation")

        # Check collections recursively
        if isinstance(data, (list, tuple, set, frozenset)):
            return all(
                self.validate_data_structure(item, max_depth, current_depth + 1) for item in data
            )

        # Check dictionaries
        if isinstance(data, dict):
            return all(
                self.validate_data_structure(k, max_depth, current_depth + 1) for k in data.keys()
            ) and all(
                self.validate_data_structure(v, max_depth, current_depth + 1) for v in data.values()
            )

        # Check dataclasses
        if is_dataclass(data):
            fields = {
                field.name: getattr(data, field.name)
                for field in data.__dataclass_fields__.values()
            }
            return self.validate_data_structure(fields, max_depth, current_depth)

        # Unknown type - log and handle based on strict mode
        error_msg = f"Unknown data type: {type(data)}"
        if self.strict_mode:
            raise SecurityError(error_msg)
        else:
            logger.warning(error_msg)
            return False

    def validate_file_path(
        self,
        file_path: Union[str, Path],
        allowed_base_dirs: Optional[List[Union[str, Path]]] = None,
    ) -> bool:
        """Validate file paths to prevent directory traversal attacks.

        Args:
            file_path: File path to validate
            allowed_base_dirs: List of allowed base directories

        Returns:
            True if safe, raises SecurityError if dangerous
        """
        file_path = Path(file_path)

        # Convert to absolute path to detect traversal
        try:
            abs_path = file_path.resolve()
        except (OSError, RuntimeError):
            error_msg = f"Invalid file path: {file_path}"
            if self.strict_mode:
                raise SecurityError(error_msg)
            else:
                logger.warning(error_msg)
                return False

        # Check for directory traversal attempts in the original path
        path_str = str(file_path)
        if ".." in path_str or "../" in path_str or "..\\" in path_str:
            error_msg = f"Directory traversal attempt detected: {file_path}"
            if self.strict_mode:
                raise SecurityError(error_msg)
            else:
                logger.warning(error_msg)
                return False

        # Block dangerous system paths
        dangerous_system_paths = [
            "/etc/",
            "/private/etc/",
            "/usr/bin/",
            "/bin/",
            "/sbin/",
            "/sys/",
            "/proc/",
            "C:\\Windows\\",
            "C:\\Program Files\\",
            "C:\\System32\\",
        ]

        abs_path_str = str(abs_path)
        for dangerous_path in dangerous_system_paths:
            if abs_path_str.startswith(dangerous_path):
                error_msg = f"Access to system path blocked: {abs_path}"
                if self.strict_mode:
                    raise SecurityError(error_msg)
                else:
                    logger.warning(error_msg)
                    return False

        # Check against allowed base directories
        if allowed_base_dirs:
            allowed_base_dirs = [Path(d).resolve() for d in allowed_base_dirs]
            if not any(str(abs_path).startswith(str(base_dir)) for base_dir in allowed_base_dirs):
                error_msg = f"File path outside allowed directories: {file_path}"
                if self.strict_mode:
                    raise SecurityError(error_msg)
                else:
                    logger.warning(error_msg)
                    return False

        return True

    def safe_literal_eval(self, expression: str) -> Any:
        """Safely evaluate literal expressions with additional validation.

        Args:
            expression: String expression to evaluate

        Returns:
            Evaluated result

        Raises:
            SecurityError: If expression contains dangerous content
        """
        # First validate the string for dangerous patterns
        self.validate_string_input(expression, "literal evaluation")

        try:
            result = ast.literal_eval(expression)
            # Additional validation of the result
            self.validate_data_structure(result)
            return result
        except (ValueError, SyntaxError) as e:
            error_msg = f"Invalid literal expression: {e}"
            raise SecurityError(error_msg)

    def sanitize_for_logging(self, data: Any, max_length: int = 1000) -> str:
        """Sanitize data for safe logging (remove sensitive information).

        Args:
            data: Data to sanitize
            max_length: Maximum string length

        Returns:
            Sanitized string representation
        """
        try:
            # Convert to string and truncate
            result = str(data)
            if len(result) > max_length:
                result = result[:max_length] + "... [truncated]"

            # Enhanced sensitive patterns including variations
            sensitive_patterns = [
                # Basic patterns
                "password",
                "passwd",
                "pass",
                "pwd",
                "token",
                "auth_token",
                "api_token",
                "access_token",
                "key",
                "api_key",
                "secret_key",
                "private_key",
                "public_key",
                "secret",
                "credential",
                "credentials",
                # Database and environment patterns
                "db_password",
                "db_pass",
                "database_password",
                "postgres_password",
                "mysql_password",
                "sqlite_password",
                # Cloud and service patterns
                "aws_secret",
                "aws_key",
                "azure_secret",
                "gcp_key",
                "stripe_secret",
                "twilio_token",
                "sendgrid_key",
                # Generic sensitive patterns
                "bearer",
                "authorization",
                "auth",
                "session",
            ]

            result_lower = result.lower()

            for pattern in sensitive_patterns:
                if pattern in result_lower:
                    # Improved redaction - match key-value pairs and redact values
                    # Pattern matches: key="value", key='value', key=value, key: value, etc.
                    patterns_to_redact = [
                        rf"{re.escape(pattern)}\s*[:=]\s*['\"]?([^'\"\s]+)['\"]?",
                        rf"{re.escape(pattern)}\s*['\"]?([^'\"\s]+)['\"]?",
                        rf"(['\"]?{re.escape(pattern)}['\"]?\s*[:=]\s*['\"]?)([^'\"\s,]*)",
                    ]

                    for redact_pattern in patterns_to_redact:
                        result = re.sub(
                            redact_pattern,
                            r"\1[REDACTED]",
                            result,
                            flags=re.IGNORECASE,
                        )

            return result
        except Exception as e:
            logger.warning(f"Error sanitizing data for logging: {e}")
            return "[SANITIZATION_ERROR]"


class SecurityError(Exception):
    """Security-related validation errors."""


# Global validator instance
_default_validator = SecurityValidator(strict_mode=True)


def validate_input(input_data: Any, context: str = "unknown") -> bool:
    """Convenience function for input validation.

    Args:
        input_data: Data to validate
        context: Context description

    Returns:
        True if valid
    """
    return _default_validator.validate_data_structure(input_data)


def safe_eval_literal(expression: str) -> Any:
    """Convenience function for safe literal evaluation.

    Args:
        expression: Expression to evaluate

    Returns:
        Evaluated result
    """
    return _default_validator.safe_literal_eval(expression)


def validate_file_path(
    file_path: Union[str, Path], allowed_dirs: Optional[List[Union[str, Path]]] = None
) -> bool:
    """Convenience function for file path validation.

    Args:
        file_path: File path to validate
        allowed_dirs: Allowed base directories

    Returns:
        True if valid
    """
    return _default_validator.validate_file_path(file_path, allowed_dirs)


def sanitize_for_logging(data: Any, max_length: int = 1000) -> str:
    """Convenience function for log sanitization.

    Args:
        data: Data to sanitize
        max_length: Maximum length

    Returns:
        Sanitized string
    """
    return _default_validator.sanitize_for_logging(data, max_length)
