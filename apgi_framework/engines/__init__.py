"""Engine Registry for the APGI Framework.

This module provides a centralized registry for all engine implementations
in the APGI Framework, including somatic marker processing, threshold management,
computation engines, and simulation engines.
"""

import logging
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

# Engine implementations
from .equation_engine import APGIEquation

# Core model imports (still in core)
from .models_engine import PredictiveIgnitionNetwork, SomaticAgent
from .precision_engine import NeuromodulatorType, PrecisionCalculator, PrecisionWeighting
from .prediction_error_engine import PredictionErrorProcessor
from .somatic_marker_engine import ContextType, SomaticMarkerEngine
from .threshold_engine import ThresholdAdaptationType, ThresholdManager

logger = logging.getLogger(__name__)


class EngineType(Enum):
    """Types of engines available in the framework."""

    SOMATIC_MARKER = auto()
    THRESHOLD = auto()
    EQUATION = auto()
    PRECISION = auto()
    PREDICTION_ERROR = auto()
    PREDICTIVE_IGNITION = auto()
    SIMULATION = auto()
    CUSTOM = auto()


T = TypeVar("T", bound=EngineType)


class EngineMetadata:
    """Metadata for registered engines."""

    def __init__(
        self,
        name: str,
        engine_type: EngineType,
        engine_class: Type[Any],
        description: str,
        version: str = "1.0.0",
        dependencies: Optional[List[str]] = None,
        config_schema: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.engine_type = engine_type
        self.engine_class = engine_class
        self.description = description
        self.version = version
        self.dependencies = dependencies or []
        self.config_schema = config_schema or {}


class EngineRegistry:
    """Central registry for all APGI Framework engines.

    This registry provides a unified interface for discovering,
    instantiating, and managing engines throughout the framework.
    """

    _instance: Optional["EngineRegistry"] = None
    _initialized: bool = False
    _engines: Dict[str, EngineMetadata] = {}
    _instances: Dict[str, Any] = {}

    def __new__(cls) -> "EngineRegistry":
        """Singleton pattern for engine registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the engine registry with default engines."""
        if self._initialized:
            return

        self._engines = {}
        self._instances = {}
        self._register_default_engines()
        self._initialized = True
        logger.info("EngineRegistry initialized with %d engines", len(self._engines))

    def _register_default_engines(self) -> None:
        """Register all default framework engines."""
        self.register(
            EngineMetadata(
                name="somatic_marker",
                engine_type=EngineType.SOMATIC_MARKER,
                engine_class=SomaticMarkerEngine,
                description="Processes somatic markers for consciousness assessment",
                version="1.0.0",
                dependencies=["numpy", "scipy"],
            )
        )

        self.register(
            EngineMetadata(
                name="threshold",
                engine_type=EngineType.THRESHOLD,
                engine_class=ThresholdManager,
                description="Manages adaptive threshold calculations",
                version="1.0.0",
                dependencies=["numpy"],
            )
        )

        self.register(
            EngineMetadata(
                name="equation",
                engine_type=EngineType.EQUATION,
                engine_class=APGIEquation,
                description="Core APGI ignition threshold equation implementation",
                version="1.0.0",
                dependencies=["numpy", "scipy"],
            )
        )

        self.register(
            EngineMetadata(
                name="precision",
                engine_type=EngineType.PRECISION,
                engine_class=PrecisionCalculator,
                description="Calculates measurement precision and confidence intervals",
                version="1.0.0",
                dependencies=["numpy", "scipy"],
            )
        )

        self.register(
            EngineMetadata(
                name="prediction_error",
                engine_type=EngineType.PREDICTION_ERROR,
                engine_class=PredictionErrorProcessor,
                description="Processes prediction error signals",
                version="1.0.0",
                dependencies=["numpy"],
            )
        )

        self.register(
            EngineMetadata(
                name="predictive_ignition",
                engine_type=EngineType.PREDICTIVE_IGNITION,
                engine_class=PredictiveIgnitionNetwork,
                description="Neural network for predictive ignition modeling",
                version="1.0.0",
                dependencies=["numpy", "scipy"],
            )
        )

    def register(self, metadata: EngineMetadata) -> None:
        """Register a new engine with the registry."""
        if metadata.name in self._engines:
            raise ValueError(f"Engine '{metadata.name}' is already registered")
        self._engines[metadata.name] = metadata
        logger.debug("Registered engine: %s", metadata.name)

    def unregister(self, name: str) -> None:
        """Unregister an engine from the registry."""
        if name in self._engines:
            del self._engines[name]
            self._instances.pop(name, None)
            logger.debug("Unregistered engine: %s", name)

    def get_engine(self, name: str) -> Optional[EngineMetadata]:
        """Get engine metadata by name."""
        return self._engines.get(name)

    def list_engines(self, engine_type: Optional[EngineType] = None) -> List[str]:
        """List all registered engine names."""
        if engine_type is None:
            return list(self._engines.keys())
        return [name for name, meta in self._engines.items() if meta.engine_type == engine_type]

    def get_engines_by_type(self, engine_type: EngineType) -> Dict[str, EngineMetadata]:
        """Get all engines of a specific type."""
        return {
            name: meta for name, meta in self._engines.items() if meta.engine_type == engine_type
        }

    def create_engine(
        self, name: str, config: Optional[Dict[str, Any]] = None, singleton: bool = True
    ) -> Any:
        """Instantiate an engine by name."""
        if singleton and name in self._instances:
            return self._instances[name]

        metadata = self._engines.get(name)
        if metadata is None:
            raise ValueError(f"Engine '{name}' not found in registry")

        try:
            config = config or {}
            instance = metadata.engine_class(**config)
            if singleton:
                self._instances[name] = instance
            logger.debug("Created engine instance: %s", name)
            return instance
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate engine '{name}': {e}") from e

    def dispose_engine(self, name: str) -> None:
        """Dispose of a cached engine instance."""
        instance = self._instances.pop(name, None)
        if instance and hasattr(instance, "dispose"):
            try:
                instance.dispose()
            except Exception as e:
                logger.warning("Error disposing engine %s: %s", name, e)

    def clear_instances(self) -> None:
        """Clear all cached engine instances."""
        for name in list(self._instances.keys()):
            self.dispose_engine(name)
        self._instances.clear()

    def is_registered(self, name: str) -> bool:
        """Check if an engine is registered."""
        return name in self._engines

    def get_engine_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered engines."""
        return {
            name: {
                "type": meta.engine_type.name,
                "description": meta.description,
                "version": meta.version,
                "dependencies": meta.dependencies,
                "class": meta.engine_class.__name__,
            }
            for name, meta in self._engines.items()
        }


__all__ = [
    "EngineRegistry",
    "EngineMetadata",
    "EngineType",
    "SomaticMarkerEngine",
    "ThresholdManager",
    "APGIEquation",
    "PrecisionCalculator",
    "PrecisionWeighting",
    "PredictionErrorProcessor",
    "PredictiveIgnitionNetwork",
    "SomaticAgent",
    "ContextType",
    "ThresholdAdaptationType",
    "NeuromodulatorType",
]
