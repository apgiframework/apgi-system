"""Production startup validation for APGI Framework.

Ensures the application is configured safely before starting in production mode.
"""

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ProductionValidator:
    """Validates configuration for production deployment."""

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_startup(self, environment: str = "production") -> bool:
        """Validate configuration before starting application.

        Args:
            environment: Current environment (production, development, testing)

        Returns:
            True if configuration is valid, False otherwise

        Raises:
            RuntimeError: If validation fails in production mode
        """
        self.errors.clear()
        self.warnings.clear()

        # Check environment-specific validations
        if environment == "production":
            self._validate_production_config()
        elif environment == "development":
            self._validate_development_config()

        # Log warnings
        for warning in self.warnings:
            logger.warning(f"Configuration Warning: {warning}")

        # Check for critical errors
        if self.errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {e}" for e in self.errors
            )
            if environment == "production":
                raise RuntimeError(error_msg)
            logger.error(error_msg)
            return False

        logger.info("Configuration validation passed")
        return True

    def _validate_production_config(self) -> None:
        """Validate production-specific configuration."""
        # Check for required environment variables
        required_vars = [
            "FLASK_SECRET_KEY",
        ]

        for var in required_vars:
            if not os.getenv(var):
                self.errors.append(f"Required environment variable not set: {var}")

        # Check for secure secret key
        secret_key = os.getenv("FLASK_SECRET_KEY")
        if secret_key and len(secret_key) < 32:
            self.errors.append("FLASK_SECRET_KEY is too short (minimum 32 characters)")

        # Check for debug mode
        debug_mode = os.getenv("DEBUG", "false").lower()
        if debug_mode == "true":
            self.errors.append("DEBUG mode is enabled in production - this is a security risk")

        # Check for database configuration
        if os.getenv("DATABASE_URL"):
            db_url = os.getenv("DATABASE_URL")
            if db_url and ("localhost" in db_url or "127.0.0.1" in db_url):
                self.warnings.append("Using localhost database in production")

        # Check for logging configuration
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        if log_level == "DEBUG":
            self.errors.append(
                "DEBUG logging is enabled in production - this may expose sensitive data"
            )

        # Check for CORS configuration
        cors_origins = os.getenv("CORS_ORIGINS", "")
        if cors_origins == "*" or cors_origins == "":
            self.errors.append("CORS origins set to wildcard (*) - this is a security risk")

    def _validate_development_config(self) -> None:
        """Validate development-specific configuration."""
        # Development is more permissive
        if not os.getenv("FLASK_SECRET_KEY"):
            self.warnings.append(
                "FLASK_SECRET_KEY not set - sessions will be invalidated on restart"
            )

    def validate_flask_config(self, config: Dict) -> bool:
        """Validate Flask application configuration.

        Args:
            config: Flask application configuration dictionary

        Returns:
            True if configuration is valid, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()

        # Check for SECRET_KEY
        if "SECRET_KEY" not in config:
            self.errors.append("SECRET_KEY not configured")
        elif len(config["SECRET_KEY"]) < 32:
            self.errors.append("SECRET_KEY is too short (minimum 32 characters)")

        # Check for DEBUG mode
        if config.get("DEBUG", False):
            self.warnings.append("DEBUG mode is enabled")

        # Check for MAX_CONTENT_LENGTH
        max_content = config.get("MAX_CONTENT_LENGTH", 0)
        if max_content > 100 * 1024 * 1024:  # 100MB
            self.warnings.append(
                f"MAX_CONTENT_LENGTH is very large: {max_content / (1024 * 1024):.1f}MB"
            )

        # Check for CORS configuration
        if "CORS_ORIGINS" in config:
            origins = config["CORS_ORIGINS"]
            if "*" in origins:
                self.errors.append("CORS origins set to wildcard (*)")

        return len(self.errors) == 0

    def validate_web_interface_config(self, config: Any) -> bool:
        """Validate web interface configuration.

        Args:
            config: WebInterfaceConfig object

        Returns:
            True if configuration is valid, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()

        # Check for valid port
        if config.port < 1 or config.port > 65535:
            self.errors.append(f"Invalid port number: {config.port}")

        # Check for reasonable file size limits
        if config.max_file_size > 1000 * 1024 * 1024:  # 1GB
            self.warnings.append(
                f"max_file_size is very large: {config.max_file_size / (1024 * 1024):.1f}MB"
            )

        # Check for allowed extensions
        dangerous_extensions = [".exe", ".bat", ".sh", ".cmd", ".com"]
        for ext in dangerous_extensions:
            if ext in config.allowed_extensions:
                self.errors.append(f"Dangerous file extension allowed: {ext}")

        return len(self.errors) == 0


# Global validator instance
_validator = None


def get_validator() -> ProductionValidator:
    """Get the global production validator instance."""
    global _validator
    if _validator is None:
        _validator = ProductionValidator()
    return _validator


def validate_startup(environment: str = "production") -> bool:
    """Validate startup configuration.

    Args:
        environment: Current environment (production, development, testing)

    Returns:
        True if validation passes, False otherwise

    Raises:
        RuntimeError: If validation fails in production mode
    """
    validator = get_validator()
    return validator.validate_startup(environment)


def validate_flask_config(config: Dict) -> bool:
    """Validate Flask configuration.

    Args:
        config: Flask application configuration

    Returns:
        True if configuration is valid, False otherwise
    """
    validator = get_validator()
    return validator.validate_flask_config(config)
