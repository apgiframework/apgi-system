"""Optional Dependency Management and Validation.

Provides robust handling of optional dependencies with proper validation,
fallback behaviors, and consistent error handling across the APGI framework.
"""

import importlib
import importlib.metadata
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

from ..logging.standardized_logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class DependencyStatus(Enum):
    """Status of an optional dependency."""

    AVAILABLE = "available"
    MISSING = "missing"
    VERSION_MISMATCH = "version_mismatch"
    IMPORT_ERROR = "import_error"
    NOT_TESTED = "not_tested"


@dataclass
class DependencyInfo:
    """Information about an optional dependency."""

    name: str
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    description: str = ""
    required_features: Optional[List[str]] = None
    fallback_module: Optional[str] = None


@dataclass
class DependencyCheckResult:
    """Result of a dependency check."""

    name: str
    status: DependencyStatus
    version: Optional[str] = None
    required_version: Optional[str] = None
    error_message: Optional[str] = None
    fallback_available: bool = False


class OptionalDependencyManager:
    """Manages optional dependencies with robust validation and fallback handling.

    This class ensures that optional dependencies are properly validated before use
    and provides consistent fallback behaviors when dependencies are unavailable.
    """

    _instance: Optional["OptionalDependencyManager"] = None
    _dependencies: Dict[str, DependencyCheckResult] = {}
    _validators: Dict[str, Callable[[], bool]] = {}
    _initialized: bool = False

    def __new__(cls) -> "OptionalDependencyManager":
        """Singleton pattern for global dependency management."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._dependencies = {}
        self._validators = {}

    def register_dependency(
        self,
        name: str,
        min_version: Optional[str] = None,
        max_version: Optional[str] = None,
        description: str = "",
        validator: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Register an optional dependency for tracking."""
        info = DependencyInfo(
            name=name,
            min_version=min_version,
            max_version=max_version,
            description=description,
        )

        # Perform initial check
        result = self._check_dependency(info)
        self._dependencies[name] = result

        if validator:
            self._validators[name] = validator

        if result.status == DependencyStatus.AVAILABLE:
            logger.debug(f"Optional dependency '{name}' is available (v{result.version})")
        else:
            logger.debug(f"Optional dependency '{name}' unavailable: {result.error_message}")

    def _check_dependency(self, info: DependencyInfo) -> DependencyCheckResult:
        """Check if a dependency is available and meets version requirements."""
        try:
            # Try to import the module
            module = importlib.import_module(info.name)

            # Get version
            version = self._get_version(module, info.name)

            # Check version constraints
            if info.min_version and version:
                if not self._version_satisfies(version, info.min_version, ">="):
                    return DependencyCheckResult(
                        name=info.name,
                        status=DependencyStatus.VERSION_MISMATCH,
                        version=version,
                        required_version=info.min_version,
                        error_message=f"Version {version} < required {info.min_version}",
                    )

            if info.max_version and version:
                if not self._version_satisfies(version, info.max_version, "<="):
                    return DependencyCheckResult(
                        name=info.name,
                        status=DependencyStatus.VERSION_MISMATCH,
                        version=version,
                        required_version=info.max_version,
                        error_message=f"Version {version} > max allowed {info.max_version}",
                    )

            # Run custom validator if provided
            if info.name in self._validators:
                if not self._validators[info.name]():
                    return DependencyCheckResult(
                        name=info.name,
                        status=DependencyStatus.IMPORT_ERROR,
                        version=version,
                        error_message="Custom validation failed",
                    )

            return DependencyCheckResult(
                name=info.name,
                status=DependencyStatus.AVAILABLE,
                version=version,
            )

        except ImportError as e:
            return DependencyCheckResult(
                name=info.name,
                status=DependencyStatus.MISSING,
                error_message=str(e),
            )
        except Exception as e:
            return DependencyCheckResult(
                name=info.name,
                status=DependencyStatus.IMPORT_ERROR,
                error_message=f"Unexpected error: {e}",
            )

    def _get_version(self, module: Any, name: str) -> Optional[str]:
        """Extract version from a module."""
        # Try common version attributes
        for attr in ["__version__", "VERSION", "version", "_version"]:
            if hasattr(module, attr):
                version = getattr(module, attr)
                if isinstance(version, str):
                    return version
                elif isinstance(version, tuple):
                    return ".".join(map(str, version))

        # Try importlib.metadata
        try:
            return importlib.metadata.version(name)
        except Exception:
            pass

        return None

    def _version_satisfies(self, version: str, constraint: str, operator: str) -> bool:
        """Check if version satisfies constraint."""
        try:
            from packaging import version as pkg_version

            v = pkg_version.parse(version)
            c = pkg_version.parse(constraint)

            if operator == ">=":
                return v >= c
            elif operator == "<=":
                return v <= c
            elif operator == "==":
                return v == c
            elif operator == "!=":
                return v != c
        except ImportError:
            # Fallback to simple string comparison
            if operator == ">=":
                return version >= constraint
            elif operator == "<=":
                return version <= constraint
            elif operator == "==":
                return version == constraint
            elif operator == "!=":
                return version != constraint

        return False

    def is_available(self, name: str) -> bool:
        """Check if an optional dependency is available."""
        if name not in self._dependencies:
            # Auto-register if not already tracked
            self.register_dependency(name)

        return self._dependencies[name].status == DependencyStatus.AVAILABLE

    def get_status(self, name: str) -> DependencyCheckResult:
        """Get detailed status of an optional dependency."""
        if name not in self._dependencies:
            self.register_dependency(name)

        return self._dependencies[name]

    def require(
        self,
        name: str,
        feature: Optional[str] = None,
        error_on_missing: bool = False,
    ) -> bool:
        """Require an optional dependency, optionally raising an error if missing.

        Args:
            name: Name of the dependency
            feature: Feature that requires this dependency (for error messages)
            error_on_missing: Whether to raise an error if dependency is missing

        Returns:
            True if dependency is available, False otherwise
        """
        available = self.is_available(name)

        if not available and error_on_missing:
            status = self.get_status(name)
            feature_msg = f" for {feature}" if feature else ""
            raise ImportError(
                f"Required dependency '{name}'{feature_msg} is not available. "
                f"Status: {status.status.value}. "
                f"Error: {status.error_message}"
            )

        if not available:
            status = self.get_status(name)
            feature_msg = f" for {feature}" if feature else ""
            warnings.warn(
                f"Optional dependency '{name}'{feature_msg} is not available. "
                f"Some features may be limited. "
                f"Error: {status.error_message}",
                ImportWarning,
            )

        return available

    def get_all_status(self) -> Dict[str, DependencyCheckResult]:
        """Get status of all registered dependencies."""
        return self._dependencies.copy()


# Global instance
def get_dependency_manager() -> OptionalDependencyManager:
    """Get the global dependency manager instance."""
    return OptionalDependencyManager()


# Common optional dependencies for APGI
COMMON_OPTIONAL_DEPS = {
    "flask": {"min_version": "2.0.0", "description": "Web dashboard"},
    "flask_socketio": {"min_version": "5.1.0", "description": "Real-time web features"},
    "torchvision": {"min_version": "0.11.0", "description": "Computer vision models"},
    "torchaudio": {"min_version": "0.11.0", "description": "Audio processing"},
    "scikit-learn": {"min_version": "1.0.0", "description": "Machine learning"},
    "seaborn": {"min_version": "0.11.2", "description": "Statistical visualization"},
    "plotly": {"min_version": "5.10.0", "description": "Interactive plots"},
    "mne": {"min_version": "1.0.0", "description": "EEG/MEG analysis"},
    "nibabel": {"min_version": "3.2.2", "description": "Neuroimaging"},
    "cv2": {"min_version": "4.5.5", "description": "Computer vision"},
    "PIL": {"min_version": "9.0.0", "description": "Image processing"},
    "psutil": {"min_version": "5.8.0", "description": "System monitoring"},
    "memory_profiler": {"min_version": "0.60.0", "description": "Memory profiling"},
    "sphinx": {"min_version": "4.0.0", "description": "Documentation"},
    "black": {"min_version": "22.0.0", "description": "Code formatting"},
    "flake8": {"min_version": "5.0.0", "description": "Linting"},
    "mypy": {"min_version": "1.0.0", "description": "Type checking"},
}


def initialize_optional_deps() -> None:
    """Initialize all common optional dependencies."""
    manager = get_dependency_manager()
    for name, info in COMMON_OPTIONAL_DEPS.items():
        manager.register_dependency(
            name,
            min_version=info.get("min_version"),
            description=info.get("description", ""),
        )


# Utility decorator for optional dependency handling
def requires_dependency(
    name: str,
    feature: Optional[str] = None,
    fallback: Optional[Callable[..., T]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to mark a function as requiring an optional dependency.

    Args:
        name: Name of the required dependency
        feature: Feature description for error messages
        fallback: Fallback function to call if dependency is unavailable
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            manager = get_dependency_manager()

            if not manager.is_available(name):
                if fallback:
                    logger.warning(
                        f"Dependency '{name}' unavailable for {func.__name__}, " f"using fallback"
                    )
                    return fallback(*args, **kwargs)

                feature_msg = f" for {feature}" if feature else ""
                raise ImportError(f"Function '{func.__name__}' requires '{name}'{feature_msg}")

            return func(*args, **kwargs)

        # Preserve original function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator
