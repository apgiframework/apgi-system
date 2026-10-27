"""Configuration-specific exceptions.

This module contains exceptions specific to the configuration system.
"""

from typing import Any, Optional

from ..exceptions import APGIFrameworkError


class ConfigurationError(APGIFrameworkError):
    """Configuration related errors."""

    def __init__(self, message: str, config_file: Optional[str] = None, **kwargs: Any) -> None:
        if config_file:
            message = f"Configuration error in '{config_file}': {message}"
        super().__init__(message=message, category="CONFIGURATION", **kwargs)
