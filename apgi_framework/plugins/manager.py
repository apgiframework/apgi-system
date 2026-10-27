"""Plugin manager for lifecycle and configuration management."""

import logging
from typing import Any, Dict, List, Optional, Set

from apgi_framework.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


class PluginManager:
    """High-level plugin management interface.

    Handles:
    - Plugin lifecycle (load, enable, disable)
    - Configuration management
    - Health checks
    - Resource cleanup
    """

    def __init__(self, registry: Optional[PluginRegistry] = None):
        self.registry = registry or PluginRegistry()
        self._enabled: Set[str] = set()
        self._configs: Dict[str, Dict[str, Any]] = {}

    def load_all(self, auto_discover: bool = True) -> None:
        """Load all available plugins."""
        if auto_discover:
            self.registry.discover_plugins()

        for spec in self.registry.list_plugins():
            missing = self.registry.check_dependencies(spec)
            if missing:
                logger.warning(f"Plugin '{spec.name}' has missing dependencies: {missing}")
            else:
                self.enable(spec.name)

    def enable(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Enable a plugin."""
        try:
            spec = self.registry.get_plugin_spec(name)
            if not spec:
                logger.error(f"Plugin '{name}' not found")
                return False

            _ = self.registry.get_engine(name, config)
            self._enabled.add(name)
            if config:
                self._configs[name] = config

            logger.info(f"Enabled plugin: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to enable plugin '{name}': {e}")
            return False

    def disable(self, name: str) -> bool:
        """Disable a plugin."""
        if name in self._enabled:
            self._enabled.discard(name)
            if name in self._configs:
                del self._configs[name]
            logger.info(f"Disabled plugin: {name}")
            return True
        return False

    def is_enabled(self, name: str) -> bool:
        """Check if plugin is enabled."""
        return name in self._enabled

    def get_enabled(self) -> List[str]:
        """List enabled plugin names."""
        return list(self._enabled)

    def health_check(self, name: str) -> Dict[str, Any]:
        """Perform health check on a plugin."""
        result = {"name": name, "healthy": False, "error": None}

        if not self.is_enabled(name):
            result["error"] = "Plugin not enabled"
            return result

        try:
            engine = self.registry.get_engine(name)
            # Try to get basic info as health check
            _ = engine.name
            result["healthy"] = True
        except Exception as e:
            result["error"] = str(e)

        return result

    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health checks on all enabled plugins."""
        return {name: self.health_check(name) for name in self._enabled}

    def shutdown(self) -> None:
        """Shutdown all plugins."""
        self.registry.shutdown_all()
        self._enabled.clear()
        self._configs.clear()
