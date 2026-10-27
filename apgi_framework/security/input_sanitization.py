"""Input Sanitization Module for APGI Framework
Provides secure input validation and sanitization utilities.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Security-related exception for input validation failures."""


class InputSanitizer:
    """Sanitize and validate various types of user input."""

    def __init__(self, allowed_directories: Optional[List[Union[str, Path]]] = None):
        """Initialize input sanitizer.

        Args:
            allowed_directories: List of allowed base directories for file operations
        """
        self.allowed_directories = []
        if allowed_directories:
            self.allowed_directories = [Path(d).resolve() for d in allowed_directories]

    def sanitize_filename(self, filename: str, max_length: int = 255) -> str:
        """Sanitize filename to prevent path traversal and invalid characters.

        Args:
            filename: Input filename
            max_length: Maximum allowed filename length

        Returns:
            Sanitized filename

        Raises:
            SecurityError: If filename contains dangerous patterns
        """
        if not filename:
            raise SecurityError("Empty filename not allowed")

        # Check for dangerous patterns
        dangerous_patterns = [
            r"\.\.",  # Directory traversal
            r"[\\/]",  # Path separators
            r"^\.+$",  # Hidden files starting with dots
            r'[<>:"|?*]',  # Windows invalid characters
            r"\x00",  # Null bytes
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, filename):
                raise SecurityError(f"Dangerous pattern detected in filename: {pattern}")

        # Remove or replace invalid characters
        sanitized = re.sub(r"[^\w\-_\.]", "_", filename)

        # Remove leading dots and dashes
        sanitized = sanitized.lstrip("._-")

        # Ensure reasonable length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        # Ensure not empty after sanitization
        if not sanitized:
            raise SecurityError("Filename became empty after sanitization")

        return sanitized

    def sanitize_path(
        self,
        path: Union[str, Path],
        base_dir: Optional[Union[str, Path]] = None,
        allow_absolute: bool = False,
    ) -> Path:
        """Sanitize file path to prevent path traversal attacks.

        Args:
            path: Input path
            base_dir: Base directory to resolve relative paths against
            allow_absolute: Whether to allow absolute paths

        Returns:
            Sanitized Path object

        Raises:
            SecurityError: If path is dangerous or outside allowed directories
        """
        if not path:
            raise SecurityError("Empty path not allowed")

        # Convert to Path object
        try:
            input_path = Path(path)
        except Exception as e:
            raise SecurityError(f"Invalid path format: {e}")

        # Check for dangerous patterns in string representation
        path_str = str(input_path)

        # Check for null bytes
        if "\x00" in path_str:
            raise SecurityError("Null bytes not allowed in paths")

        # Resolve the path
        if base_dir:
            base_path = Path(base_dir).resolve()
            resolved_path = (base_path / input_path).resolve()
        else:
            resolved_path = input_path.resolve()

        # Check if absolute paths are allowed
        if not allow_absolute and not input_path.is_absolute() and not base_dir:
            raise SecurityError("Relative paths require base_dir when absolute paths not allowed")

        # Check for path traversal
        if base_dir:
            try:
                resolved_path.relative_to(base_path.resolve())
            except ValueError:
                raise SecurityError(f"Path traversal detected: {path} is outside {base_dir}")

        # Check against allowed directories
        if self.allowed_directories:
            is_allowed = False
            for allowed_dir in self.allowed_directories:
                try:
                    resolved_path.relative_to(allowed_dir)
                    is_allowed = True
                    break
                except ValueError:
                    continue

            if not is_allowed:
                allowed_str = ", ".join(str(d) for d in self.allowed_directories)
                raise SecurityError(f"Path not in allowed directories: {allowed_str}")

        return resolved_path

    def validate_file_extension(
        self,
        filename: str,
        allowed_extensions: List[str],
        case_sensitive: bool = False,
    ) -> bool:
        """Validate file extension against allowed list.

        Args:
            filename: Filename to check
            allowed_extensions: List of allowed extensions (with or without dot)
            case_sensitive: Whether to check case sensitively

        Returns:
            True if extension is allowed

        Raises:
            SecurityError: If extension is not allowed
        """
        if not filename or "." not in filename:
            raise SecurityError("Invalid filename or missing extension")

        # Extract extension
        extension = filename.split(".")[-1]

        # Normalize extensions (remove dots if present)
        normalized_allowed = [ext.lstrip(".") for ext in allowed_extensions]
        normalized_extension = extension.lstrip(".")

        if not case_sensitive:
            normalized_extension = normalized_extension.lower()
            normalized_allowed = [ext.lower() for ext in normalized_allowed]

        if normalized_extension not in normalized_allowed:
            raise SecurityError(
                f"File extension '{extension}' not allowed. Allowed: {allowed_extensions}"
            )

        return True

    def sanitize_text_input(
        self,
        text: str,
        max_length: int = 1000,
        allow_html: bool = False,
        allow_script: bool = False,
    ) -> str:
        """Sanitize text input to prevent injection attacks.

        Args:
            text: Input text
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML tags
            allow_script: Whether to allow script content

        Returns:
            Sanitized text

        Raises:
            SecurityError: If text contains dangerous content
        """
        if not isinstance(text, str):
            raise SecurityError("Text input must be a string")

        # Check length
        if len(text) > max_length:
            raise SecurityError(f"Text too long: {len(text)} > {max_length}")

        # Check for null bytes
        if "\x00" in text:
            raise SecurityError("Null bytes not allowed in text")

        sanitized = text

        # Remove or escape dangerous content
        if not allow_html:
            # Remove HTML tags
            sanitized = re.sub(r"<[^>]+>", "", sanitized)

        if not allow_script:
            # Remove script content and common attack patterns
            script_patterns = [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"vbscript:",
                r"onload\s*=",
                r"onerror\s*=",
                r"onclick\s*=",
                r"data:text/html",
            ]

            for pattern in script_patterns:
                if re.search(pattern, sanitized, re.IGNORECASE):
                    raise SecurityError(f"Script content detected: {pattern}")

        # Control characters (except common ones)
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", sanitized)

        return sanitized.strip()

    def validate_numeric_input(
        self,
        value: Any,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        allow_float: bool = True,
    ) -> Union[int, float]:
        """Validate and sanitize numeric input.

        Args:
            value: Input value
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            allow_float: Whether to allow floating point numbers

        Returns:
            Validated numeric value

        Raises:
            SecurityError: If value is invalid or out of range
        """
        try:
            if allow_float:
                num_value = float(value)
            else:
                num_value = int(value)
                if float(value) != num_value:
                    raise SecurityError("Integer required but float provided")
        except (ValueError, TypeError):
            raise SecurityError(f"Invalid numeric value: {value}")

        if min_val is not None and num_value < min_val:
            raise SecurityError(f"Value {num_value} below minimum {min_val}")

        if max_val is not None and num_value > max_val:
            raise SecurityError(f"Value {num_value} above maximum {max_val}")

        return num_value

    def sanitize_url(self, url: str, allowed_schemes: Optional[List[str]] = None) -> str:
        """Sanitize URL to prevent malicious URLs.

        Args:
            url: Input URL
            allowed_schemes: List of allowed URL schemes

        Returns:
            Sanitized URL

        Raises:
            SecurityError: If URL is dangerous or invalid
        """
        if not url:
            raise SecurityError("Empty URL not allowed")

        if allowed_schemes is None:
            allowed_schemes = ["http", "https", "ftp"]

        try:
            parsed = urlparse(url)
        except Exception as e:
            raise SecurityError(f"Invalid URL format: {e}")

        # Check scheme
        if parsed.scheme.lower() not in allowed_schemes:
            raise SecurityError(f"URL scheme '{parsed.scheme}' not allowed")

        # Check for dangerous content
        dangerous_patterns = [
            "javascript:",
            "vbscript:",
            "data:",
            "file:",
        ]

        for pattern in dangerous_patterns:
            if pattern in url.lower():
                raise SecurityError(f"Dangerous URL pattern detected: {pattern}")

        return url

    def validate_json_input(self, json_str: str, max_size: int = 1024 * 1024) -> dict:
        """Validate and parse JSON input.

        Args:
            json_str: JSON string to validate
            max_size: Maximum allowed JSON size in bytes

        Returns:
            Parsed JSON object

        Raises:
            SecurityError: If JSON is invalid or too large
        """
        if len(json_str.encode("utf-8")) > max_size:
            raise SecurityError(f"JSON too large: {len(json_str)} > {max_size}")

        try:
            import json

            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise SecurityError(f"Invalid JSON: {e}")

        return cast(Dict[Any, Any], parsed)

    def create_secure_filename(self, base_name: str, extension: str) -> str:
        """Create a secure filename with timestamp and random component.

        Args:
            base_name: Base name for the file
            extension: File extension

        Returns:
            Secure filename
        """
        import time
        import uuid

        # Sanitize inputs
        safe_base = self.sanitize_filename(base_name)
        safe_ext = extension.lstrip(".")

        # Add timestamp and UUID for uniqueness
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]

        return f"{safe_base}_{timestamp}_{unique_id}.{safe_ext}"


class SecureFileHandler:
    """Secure file operations with input validation."""

    def __init__(self, allowed_directories: Optional[List[Union[str, Path]]] = None):
        """Initialize secure file handler.

        Args:
            allowed_directories: List of allowed base directories
        """
        self.sanitizer = InputSanitizer(allowed_directories)

    def safe_read_file(
        self,
        file_path: Union[str, Path],
        base_dir: Optional[Union[str, Path]] = None,
        max_size: int = 10 * 1024 * 1024,
    ) -> bytes:
        """Safely read file with path validation.

        Args:
            file_path: Path to file
            base_dir: Base directory for relative paths
            max_size: Maximum file size in bytes

        Returns:
            File contents

        Raises:
            SecurityError: If path is dangerous or file too large
        """
        # Validate path
        safe_path = self.sanitizer.sanitize_path(file_path, base_dir)

        # Check if file exists and is within size limits
        if not safe_path.exists():
            raise SecurityError(f"File does not exist: {safe_path}")

        if not safe_path.is_file():
            raise SecurityError(f"Path is not a file: {safe_path}")

        file_size = safe_path.stat().st_size
        if file_size > max_size:
            raise SecurityError(f"File too large: {file_size} > {max_size}")

        try:
            with open(safe_path, "rb") as f:
                return f.read()
        except Exception as e:
            raise SecurityError(f"Failed to read file: {e}")

    def safe_write_file(
        self,
        file_path: Union[str, Path],
        content: Union[str, bytes],
        base_dir: Optional[Union[str, Path]] = None,
        allowed_extensions: Optional[List[str]] = None,
    ) -> Path:
        """Safely write file with path validation.

        Args:
            file_path: Path to write file
            content: File content
            base_dir: Base directory for relative paths
            allowed_extensions: Allowed file extensions

        Returns:
            Path to written file

        Raises:
            SecurityError: If path is dangerous or extension not allowed
        """
        # Validate path
        safe_path = self.sanitizer.sanitize_path(file_path, base_dir)

        # Validate extension if provided
        if allowed_extensions:
            self.sanitizer.validate_file_extension(safe_path.name, allowed_extensions)

        # Create parent directories if needed
        safe_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if isinstance(content, str):
                with open(safe_path, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                with open(safe_path, "wb") as f:
                    f.write(content)

            return safe_path
        except Exception as e:
            raise SecurityError(f"Failed to write file: {e}")

    def safe_list_directory(
        self,
        dir_path: Union[str, Path],
        base_dir: Optional[Union[str, Path]] = None,
        max_entries: int = 1000,
    ) -> List[Path]:
        """Safely list directory contents.

        Args:
            dir_path: Directory path
            base_dir: Base directory for relative paths
            max_entries: Maximum number of entries to return

        Returns:
            List of directory entries

        Raises:
            SecurityError: If path is dangerous or not a directory
        """
        # Validate path
        safe_path = self.sanitizer.sanitize_path(dir_path, base_dir)

        if not safe_path.exists():
            raise SecurityError(f"Directory does not exist: {safe_path}")

        if not safe_path.is_dir():
            raise SecurityError(f"Path is not a directory: {safe_path}")

        try:
            entries = list(safe_path.iterdir())

            if len(entries) > max_entries:
                raise SecurityError(f"Too many directory entries: {len(entries)} > {max_entries}")

            return entries
        except Exception as e:
            raise SecurityError(f"Failed to list directory: {e}")


# Global sanitizer instance
default_sanitizer = InputSanitizer()


# Convenience functions
def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Convenience function to sanitize filename."""
    return default_sanitizer.sanitize_filename(filename, max_length)


def sanitize_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Convenience function to sanitize path."""
    return default_sanitizer.sanitize_path(path, base_dir)


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Convenience function to validate file extension."""
    return default_sanitizer.validate_file_extension(filename, allowed_extensions)


def sanitize_text_input(text: str, max_length: int = 1000) -> str:
    """Convenience function to sanitize text input."""
    return default_sanitizer.sanitize_text_input(text, max_length)


if __name__ == "__main__":
    # Example usage
    sanitizer = InputSanitizer(allowed_directories=["/tmp", "./data"])  # nosec B108 - demo only

    # Test filename sanitization
    try:
        safe_name = sanitizer.sanitize_filename("../../../etc/passwd")
        logger.info(f"Sanitized filename: {safe_name}")
    except SecurityError as e:
        logger.info(f"Security error: {e}")

    # Test path sanitization
    try:
        safe_path = sanitizer.sanitize_path("../data/test.txt", base_dir="./safe")
        logger.info(f"Safe path: {safe_path}")
    except SecurityError as e:
        logger.info(f"Security error: {e}")

    # Test text sanitization
    try:
        safe_text = sanitizer.sanitize_text_input("<script>alert('xss')</script>Hello")
        logger.info(f"Sanitized text: {safe_text}")
    except SecurityError as e:
        logger.info(f"Security error: {e}")

    # Test secure file handler
    handler = SecureFileHandler(allowed_directories=["./test_data"])
    logger.info("Security tests completed.")
