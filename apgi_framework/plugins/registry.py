"""Plugin registry for APGI Framework.

Provides entry point-based plugin discovery and registration.
"""

import importlib.metadata
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EngineInterface(ABC):
    """Abstract base class for simulation engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Engine version."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the engine with configuration."""
        pass

    @abstractmethod
    def run_simulation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run a simulation and return results."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the engine."""
        pass


@dataclass
class PluginSpec:
    """Specification for an APGI plugin."""

    name: str
    version: str
    description: str
    author: str
    engine_class: type
    entry_point: Optional[str] = None
    dependencies: Optional[List[str]] = None
    config_schema: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.dependencies is None:
            self.dependencies = []


class PluginRegistry:
    """Central registry for APGI plugins.

    Supports:
    - Entry point-based plugin discovery
    - Manual plugin registration
    - Lazy loading of plugins
    - Dependency checking

    Example:
        registry = PluginRegistry()
        registry.discover_plugins()
        engine = registry.get_engine("custom_neural_engine")
    """

    ENTRY_POINT_GROUP = "apgi.engines"

    def __init__(self) -> None:
        self._plugins: Dict[str, PluginSpec] = {}
        self._engines: Dict[str, EngineInterface] = {}
        self._initialized: bool = False

    def register(self, spec: PluginSpec) -> None:
        """Register a plugin specification."""
        if spec.name in self._plugins:
            logger.warning(f"Plugin '{spec.name}' already registered, overwriting")

        self._plugins[spec.name] = spec
        logger.info(f"Registered plugin: {spec.name} v{spec.version}")

    def unregister(self, name: str) -> None:
        """Unregister a plugin."""
        if name in self._plugins:
            del self._plugins[name]
            if name in self._engines:
                del self._engines[name]
            logger.info(f"Unregistered plugin: {name}")

    def discover_plugins(self) -> List[str]:
        """Discover plugins via entry points.

        Returns:
            List of discovered plugin names
        """
        discovered = []
        try:
            entry_points = importlib.metadata.entry_points()
            if hasattr(entry_points, "select"):
                # Python 3.10+ API
                eps = entry_points.select(group=self.ENTRY_POINT_GROUP)
            else:
                # Legacy API
                eps = entry_points.get(self.ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]

            for ep in eps:
                try:
                    plugin_module = ep.load()
                    if hasattr(plugin_module, "get_plugin_spec"):
                        spec = plugin_module.get_plugin_spec()
                        self.register(spec)
                        discovered.append(spec.name)
                    else:
                        logger.warning(f"Plugin {ep.name} missing get_plugin_spec()")
                except Exception as e:
                    logger.error(f"Failed to load plugin {ep.name}: {e}")

        except Exception as e:
            logger.error(f"Plugin discovery failed: {e}")

        return discovered

    def get_plugin_spec(self, name: str) -> Optional[PluginSpec]:
        """Get plugin specification by name."""
        return self._plugins.get(name)

    def get_engine(self, name: str, config: Optional[Dict[str, Any]] = None) -> EngineInterface:
        """Get or create an engine instance.

        Args:
            name: Plugin name
            config: Optional configuration override

        Returns:
            EngineInterface instance

        Raises:
            KeyError: If plugin not found
            RuntimeError: If engine initialization fails
        """
        if name in self._engines:
            return self._engines[name]

        spec = self._plugins.get(name)
        if not spec:
            raise KeyError(f"Plugin '{name}' not found")

        try:
            engine = spec.engine_class()
            engine.initialize(config or {})
            self._engines[name] = engine
            return engine
        except Exception as e:
            raise RuntimeError(f"Failed to initialize engine '{name}': {e}")

    def list_plugins(self) -> List[PluginSpec]:
        """List all registered plugins."""
        return list(self._plugins.values())

    def list_engines(self) -> List[str]:
        """List all initialized engine names."""
        return list(self._engines.keys())

    def shutdown_all(self) -> None:
        """Shutdown all initialized engines."""
        for name, engine in self._engines.items():
            try:
                engine.shutdown()
                logger.info(f"Shutdown engine: {name}")
            except Exception as e:
                logger.error(f"Error shutting down engine {name}: {e}")
        self._engines.clear()

    def check_dependencies(self, spec: PluginSpec) -> List[str]:
        """Check if plugin dependencies are satisfied.

        Returns:
            List of missing dependencies
        """
        missing = []
        for dep in spec.dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(dep)
        return missing


# Global registry instance
_global_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry
