"""Advanced Configuration Management System for APGI Framework.

Provides centralized configuration management with validation,
environment-specific settings, and dynamic updates.
"""

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_type_hints

import yaml

try:
    from apgi_framework.logging.standardized_logging import get_logger
    from apgi_framework.validation.input_validator import (
        ValidationError,
        validate_path,
        validate_string,
    )

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore [assignment]

    # Fallback validation classes and functions
    class ValidationError(Exception):  # type: ignore
        pass

    class ConfigurationError(Exception):  # type: ignore
        pass

    def validate_string(value: Any, field_name: str = "input", **kwargs: Any) -> str:
        return str(value)

    def validate_path(value: Any, field_name: str = "input", **kwargs: Any) -> Path:
        return Path(value)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    host: str = "localhost"
    port: int = 5432
    database: str = "apgi_framework"
    username: str = ""  # Require explicit configuration
    password: str = ""  # Require explicit configuration
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


@dataclass
class MonitoringConfig:
    """Monitoring configuration settings."""

    enable_realtime: bool = True
    websocket_port: int = 8765
    data_retention_days: int = 30
    system_check_interval: int = 5
    alert_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
        }
    )


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    enable_input_validation: bool = True
    max_login_attempts: int = 3
    session_timeout: int = 3600
    allowed_hosts: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    encryption_key_rotation_days: int = 90


@dataclass
class ExperimentalConfig:
    """Experimental configuration settings."""

    default_trials: int = 1000
    max_participants: int = 1000
    timeout_seconds: int = 3600
    parallel_workers: int = 4
    data_export_format: str = "json"
    enable_backup: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration settings."""

    level: str = "INFO"
    enable_file_logging: bool = True
    log_directory: str = "logs"
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_structured_logging: bool = False
    log_to_database: bool = False


@dataclass
class APGIConfig:
    """Main configuration class for APGI Framework."""

    # Environment
    environment: str = "development"
    debug: bool = False
    strict_mode: bool = False  # Fail startup on configuration errors in production
    secret_key: str = ""

    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Paths
    data_directory: str = "data"
    output_directory: str = "output"
    backup_directory: str = "backups"
    temp_directory: str = "temp"

    # Feature flags
    features: Dict[str, bool] = field(
        default_factory=lambda: {
            "realtime_monitoring": True,
            "advanced_visualization": True,
            "ml_classification": False,
            "database_integration": False,
            "websocket_support": True,
            "gui_themes": True,
            "automated_backup": True,
            "performance_profiling": True,
        }
    )


class ConfigurationManager:
    """Advanced configuration management system.

    Handles loading, validation, and management of configuration
    from multiple sources with environment-specific overrides.
    """

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.

        Args:
            config_file: Path to configuration file (JSON or YAML)
        """
        self.config_file = Path(config_file) if config_file else None
        self.config = APGIConfig()
        self._config_sources: List[str] = []

        # Load configuration in priority order
        self._load_default_config()
        if self.config_file and self.config_file.exists():
            self._load_file_config()
        self._load_environment_config()
        self._load_command_line_config()

        logger.info(f"ConfigurationManager initialized with {len(self._config_sources)} sources")

    def _load_default_config(self) -> None:
        """Load default configuration."""
        self.config = APGIConfig()
        self._config_sources.append("defaults")
        logger.debug("Loaded default configuration")

    def _load_file_config(self) -> None:
        """Load configuration from file."""
        if not self.config_file or not self.config_file.exists():
            return

        try:
            with open(self.config_file, "r") as f:
                if self.config_file.suffix.lower() in [".yaml", ".yml"]:
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)

            self._update_config_from_dict(file_config)
            self._config_sources.append(f"file:{self.config_file}")
            logger.info(f"Loaded configuration from {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to load configuration from {self.config_file}: {e}")
            raise

    def _load_environment_config(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            "APGI_ENVIRONMENT": ("environment", str),
            "APGI_DEBUG": ("debug", bool),
            "APGI_STRICT_MODE": ("strict_mode", bool),
            "APGI_SECRET_KEY": ("secret_key", str),
            "APGI_DATA_DIR": ("data_directory", str),
            "APGI_OUTPUT_DIR": ("output_directory", str),
            "APGI_LOG_LEVEL": ("logging.level", str),
            "APGI_LOG_DIR": ("logging.log_directory", str),
            "APGI_DB_HOST": ("database.host", str),
            "APGI_DB_PORT": ("database.port", int),
            "APGI_DB_NAME": ("database.database", str),
            "APGI_MONITORING_PORT": ("monitoring.websocket_port", int),
            "APGI_MAX_TRIALS": ("experimental.default_trials", int),
            "APGI_MAX_PARTICIPANTS": ("experimental.max_participants", int),
        }

        env_config: Dict[str, Any] = {}
        for env_var, (config_path, value_type) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert to appropriate type
                if value_type == bool:
                    value_bool = value.lower() in ["true", "1", "yes", "on"]
                    value = value_bool  # type: ignore[assignment]
                elif value_type == int:
                    try:
                        value_int = int(value)
                        value = value_int  # type: ignore[assignment]
                    except ValueError:
                        logger.warning(f"Invalid integer value for {env_var}: {value}")
                        continue

                # Set nested configuration
                self._set_nested_config(env_config, config_path, value)

        if env_config:
            self._update_config_from_dict(env_config)
            self._config_sources.append("environment")
            logger.debug(f"Loaded {len(env_config)} environment variables")

    def _load_command_line_config(self) -> None:
        """Load configuration from command line arguments."""
        # This would be called from CLI with parsed arguments
        # Implementation depends on how CLI is structured

    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                attr = getattr(self.config, key)
                if isinstance(
                    attr,
                    (
                        DatabaseConfig,
                        MonitoringConfig,
                        SecurityConfig,
                        ExperimentalConfig,
                        LoggingConfig,
                    ),
                ):
                    # Update nested config object
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if hasattr(attr, sub_key):
                                # Get field type and coerce value
                                field_type = get_type_hints(type(attr)).get(sub_key)
                                if field_type:
                                    sub_value = self._coerce_value(sub_value, field_type)
                                setattr(attr, sub_key, sub_value)
                else:
                    # Update direct attribute
                    setattr(self.config, key, value)
            elif key == "features":
                # Update features
                if isinstance(value, dict):
                    self.config.features.update(value)

    def _set_nested_config(self, config_dict: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested configuration value."""
        keys = path.split(".")
        current = config_dict

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _coerce_value(self, value: Any, field_type: Any) -> Any:
        """Coerce configuration value to the expected type."""
        try:
            if field_type == bool:
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            elif field_type == int:
                return int(value)
            elif field_type == float:
                return float(value)
            elif field_type == str:
                return str(value)
            else:
                return value
        except (ValueError, TypeError):
            logger.warning(f"Could not coerce {value} to {field_type}, using as-is")
            return value

    def get_config(self) -> APGIConfig:
        """Get current configuration."""
        return self.config

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.config.database

    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        return self.config.monitoring

    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        return self.config.security

    def get_experimental_config(self) -> ExperimentalConfig:
        """Get experimental configuration."""
        return self.config.experimental

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.config.logging

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        return self.config.features.get(feature_name, False)

    def enable_feature(self, feature_name: str) -> None:
        """Enable a feature."""
        self.config.features[feature_name] = True
        logger.info(f"Feature enabled: {feature_name}")

    def disable_feature(self, feature_name: str) -> None:
        """Disable a feature."""
        self.config.features[feature_name] = False
        logger.info(f"Feature disabled: {feature_name}")

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with validation."""
        old_config = self.config
        try:
            self._update_config_from_dict(updates)
            self.save_config()  # Persist changes to file
            logger.info("Configuration updated successfully")
        except Exception as e:
            # Restore previous config on error
            self.config = old_config
            logger.error(f"Failed to update configuration: {e}")
            raise

    def save_config(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file."""
        save_path = Path(file_path) if file_path else self.config_file
        if not save_path:
            raise ValueError("No file path specified for saving configuration")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self.config)

        try:
            with open(save_path, "w") as f:
                if save_path.suffix.lower() in [".yaml", ".yml"]:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)

            logger.info(f"Configuration saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration to {save_path}: {e}")
            raise

    def validate_config(self) -> List[str]:
        """Validate current configuration.

        Returns:
            List of validation error messages

        Raises:
            ConfigurationError: If strict mode is enabled and validation fails
        """
        errors = []

        # Validate secret key for production
        if self.config.environment == "production" and not self.config.secret_key:
            errors.append("APGI_SECRET_KEY must be set in production environment")

        # Detect placeholder/default secret keys
        placeholder_patterns = [
            "your-secret-key-here",
            "your-new-secret-key-here",
            "change-this-to-a-secure",
            "replace-with-random",
            "default-secret-key",
            "dev-secret-key",
        ]

        if self.config.secret_key:
            secret_lower = self.config.secret_key.lower()
            for pattern in placeholder_patterns:
                if pattern in secret_lower:
                    if self.config.environment == "production":
                        errors.append(
                            "APGI_SECRET_KEY appears to be a placeholder value. "
                            "Replace with a secure random secret key (64 hex characters recommended)."
                        )
                    else:
                        logger.warning(
                            "APGI_SECRET_KEY appears to be a placeholder value. "
                            "Set a secure secret key for production deployment."
                        )
                    break

        # In strict mode, fail on missing secrets even in development
        if self.config.strict_mode and not self.config.secret_key:
            errors.append(
                "APGI_SECRET_KEY must be set in strict mode. "
                "Set APGI_SECRET_KEY environment variable or disable strict mode."
            )

        # Auto-generate secret key for development if not set (only in non-strict mode)
        if (
            self.config.environment == "development"
            and not self.config.secret_key
            and not self.config.strict_mode
        ):
            import secrets

            self.config.secret_key = secrets.token_hex(32)
            logger.warning(
                "Auto-generated development secret key - set APGI_SECRET_KEY for production"
            )

        # Validate paths
        try:
            validate_path(self.config.data_directory, "data_directory", must_exist=False)
        except ValidationError as e:
            errors.append(str(e))

        try:
            validate_path(self.config.output_directory, "output_directory", must_exist=False)
        except ValidationError as e:
            errors.append(str(e))

        # Validate logging level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.config.logging.level not in valid_levels:
            errors.append(f"Invalid logging level: {self.config.logging.level}")

        # Validate ports
        if not (1 <= self.config.monitoring.websocket_port <= 65535):
            errors.append(f"Invalid WebSocket port: {self.config.monitoring.websocket_port}")

        if not (1 <= self.config.database.port <= 65535):
            errors.append(f"Invalid database port: {self.config.database.port}")

        # Validate experimental parameters
        if self.config.experimental.default_trials <= 0:
            errors.append("Default trials must be positive")

        if self.config.experimental.max_participants <= 0:
            errors.append("Max participants must be positive")

        # Validate database credentials for non-SQLite backends
        # SQLite typically uses a file path for database name and empty host
        is_sqlite = (
            not self.config.database.host
            or Path(self.config.database.database).exists()
            or self.config.database.database.endswith(".db")
            or self.config.database.database.startswith("file:")
        )

        if not is_sqlite:
            if not self.config.database.username:
                errors.append("Database username is required for non-SQLite backends")
            if not self.config.database.password:
                errors.append("Database password is required for non-SQLite backends")

        # In strict mode, raise ConfigurationError instead of returning errors
        if self.config.strict_mode and errors:
            raise ConfigurationError(
                "Configuration validation failed in strict mode:\n" + "\n".join(errors)
            )

        return errors

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for display."""
        return {
            "environment": self.config.environment,
            "debug": self.config.debug,
            "data_directory": self.config.data_directory,
            "output_directory": self.config.output_directory,
            "monitoring": {
                "realtime_enabled": self.config.monitoring.enable_realtime,
                "websocket_port": self.config.monitoring.websocket_port,
            },
            "database": {
                "host": self.config.database.host,
                "port": self.config.database.port,
                "database": self.config.database.database,
            },
            "features": self.config.features,
            "config_sources": self._config_sources,
        }

    def reload_config(self) -> None:
        """Reload configuration from sources."""
        old_config = self.config
        self._config_sources.clear()

        try:
            self._load_default_config()
            if self.config_file and self.config_file.exists():
                self._load_file_config()
            self._load_environment_config()
            self._load_command_line_config()

            logger.info("Configuration reloaded successfully")

        except Exception as e:
            # Restore old config on error
            self.config = old_config
            logger.error(f"Failed to reload configuration: {e}")
            raise


# Global configuration manager instance
_global_config_manager: Optional[ConfigurationManager] = None
_config_lock = threading.Lock()


def get_config_manager(
    config_file: Optional[Union[str, Path]] = None,
) -> ConfigurationManager:
    """Get or create the global configuration manager."""
    global _global_config_manager
    with _config_lock:
        if _global_config_manager is None:
            _global_config_manager = ConfigurationManager(config_file)
    return _global_config_manager


def get_config() -> APGIConfig:
    """Get the current global configuration."""
    return get_config_manager().get_config()


def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled in the global configuration."""
    return get_config_manager().is_feature_enabled(feature_name)


def reload_config() -> None:
    """Reload the global configuration."""
    get_config_manager().reload_config()
