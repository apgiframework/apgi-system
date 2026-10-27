"""
Deprecation Utilities for APGI Framework

Provides standardized deprecation warnings and version-based deprecation management.
"""

import functools
import warnings
from typing import Any, Callable, Optional, Type, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T", bound=Type[Any])


class APGIDeprecationWarning(DeprecationWarning):
    """Custom deprecation warning for APGI framework."""

    pass


# Make APGIDeprecationWarning visible by default
warnings.simplefilter("always", APGIDeprecationWarning)


def deprecated(
    reason: str,
    version: str,
    removal_version: Optional[str] = None,
    alternative: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator to mark functions as deprecated.

    Args:
        reason: Explanation of why the function is deprecated
        version: Version when the deprecation was introduced
        removal_version: Planned version for removal
        alternative: Name of the recommended alternative function/class

    Returns:
        Decorated function that emits a deprecation warning

    Example:
        @deprecated(
            reason="Use new_analysis_engine instead",
            version="2.1.0",
            removal_version="3.0.0",
            alternative="new_analysis_engine"
        )
        def old_analysis_engine():
            pass
    """

    def decorator(func: F) -> F:
        message = f"{func.__name__} is deprecated since version {version}. {reason}"
        if removal_version:
            message += f" It will be removed in version {removal_version}."
        if alternative:
            message += f" Use {alternative} instead."

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                message,
                APGIDeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        # Add deprecation metadata
        wrapper.__deprecated__ = True  # type: ignore
        wrapper.__deprecated_version__ = version  # type: ignore
        wrapper.__deprecated_reason__ = reason  # type: ignore
        wrapper.__deprecated_removal__ = removal_version  # type: ignore
        wrapper.__deprecated_alternative__ = alternative  # type: ignore

        return wrapper  # type: ignore

    return decorator


def deprecated_class(
    reason: str,
    version: str,
    removal_version: Optional[str] = None,
    alternative: Optional[str] = None,
) -> Callable[[T], T]:
    """
    Decorator to mark classes as deprecated.

    Args:
        reason: Explanation of why the class is deprecated
        version: Version when the deprecation was introduced
        removal_version: Planned version for removal
        alternative: Name of the recommended alternative class

    Returns:
        Decorated class that emits a deprecation warning on instantiation
    """

    def decorator(cls: T) -> T:
        message = f"{cls.__name__} is deprecated since version {version}. {reason}"
        if removal_version:
            message += f" It will be removed in version {removal_version}."
        if alternative:
            message += f" Use {alternative} instead."

        original_init = cls.__init__

        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            warnings.warn(
                message,
                APGIDeprecationWarning,
                stacklevel=2,
            )
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init

        # Add deprecation metadata
        cls.__deprecated__ = True  # type: ignore
        cls.__deprecated_version__ = version  # type: ignore
        cls.__deprecated_reason__ = reason  # type: ignore
        cls.__deprecated_removal__ = removal_version  # type: ignore
        cls.__deprecated_alternative__ = alternative  # type: ignore

        return cls

    return decorator


def deprecated_parameter(
    param_name: str,
    reason: str,
    version: str,
    removal_version: Optional[str] = None,
    alternative: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator to mark function parameters as deprecated.

    Args:
        param_name: Name of the deprecated parameter
        reason: Explanation of why the parameter is deprecated
        version: Version when the deprecation was introduced
        removal_version: Planned version for removal
        alternative: Name of the recommended alternative parameter

    Returns:
        Decorated function that emits a deprecation warning when parameter is used
    """

    def decorator(func: F) -> F:
        message = (
            f"Parameter '{param_name}' in {func.__name__} is deprecated since version {version}. "
            f"{reason}"
        )
        if removal_version:
            message += f" It will be removed in version {removal_version}."
        if alternative:
            message += f" Use '{alternative}' instead."

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if param_name in kwargs:
                warnings.warn(
                    message,
                    APGIDeprecationWarning,
                    stacklevel=2,
                )
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def emit_deprecation_warning(
    message: str,
    version: Optional[str] = None,
    stacklevel: int = 2,
) -> None:
    """
    Emit a standardized deprecation warning.

    Args:
        message: The deprecation message
        version: Version when the deprecation was introduced
        stacklevel: Stack level for warning (default: 2)
    """
    full_message = message
    if version:
        full_message = f"[Deprecated since {version}] {message}"

    warnings.warn(
        full_message,
        APGIDeprecationWarning,
        stacklevel=stacklevel,
    )


_warned_modules: set = set()


def warn_deprecated_module(
    module_name: str,
    replacement: str,
    *,
    since: str,
    removal: Optional[str] = None,
    stacklevel: int = 3,
) -> bool:
    """Warn that a whole module is deprecated; once per module per process.

    Meant to be called at import time of a legacy or duplicate module. The use is recorded
    in the global :class:`DeprecatedFeatureManager` so :func:`get_deprecation_manager`
    reports which deprecated modules are still being imported.

    Args:
        module_name: Name of the deprecated module (usually ``__name__``).
        replacement: Module or entry point to use instead.
        since: Version in which the module was deprecated.
        removal: Version in which it will be removed, if known.
        stacklevel: Passed to :func:`warnings.warn`; 3 points at the importer.

    Returns:
        True if a warning was emitted, False if this module already warned or its
        warnings are suppressed in the manager.
    """
    manager = get_deprecation_manager()
    if module_name in _warned_modules or manager.is_suppressed(module_name):
        return False
    _warned_modules.add(module_name)
    message = (
        f"Module {module_name} is deprecated since version {since}. Use {replacement} instead."
    )
    if removal:
        message += f" It will be removed in version {removal}."
    manager.record_warning(module_name)
    warnings.warn(message, APGIDeprecationWarning, stacklevel=stacklevel)
    return True


def is_deprecated(obj: Any) -> bool:
    """
    Check if a function or class is marked as deprecated.

    Args:
        obj: Object to check

    Returns:
        True if deprecated, False otherwise
    """
    return getattr(obj, "__deprecated__", False)


def get_deprecation_info(obj: Any) -> Optional[dict]:
    """
    Get deprecation information for a deprecated object.

    Args:
        obj: Object to get info for

    Returns:
        Dictionary with deprecation metadata or None if not deprecated
    """
    if not is_deprecated(obj):
        return None

    return {
        "version": getattr(obj, "__deprecated_version__", None),
        "reason": getattr(obj, "__deprecated_reason__", None),
        "removal": getattr(obj, "__deprecated_removal__", None),
        "alternative": getattr(obj, "__deprecated_alternative__", None),
    }


def check_version_deprecation(
    current_version: str,
    deprecated_since: str,
    removal_version: Optional[str] = None,
) -> bool:
    """
    Check if a feature should be deprecated based on version comparison.

    Args:
        current_version: Current version of the software
        deprecated_since: Version when feature was deprecated
        removal_version: Planned removal version

    Returns:
        True if feature is deprecated in current version
    """
    from packaging import version as pkg_version

    current = pkg_version.parse(current_version)
    deprecated = pkg_version.parse(deprecated_since)

    # Check if we're past the deprecation point
    if current < deprecated:
        return False

    # Check if we're past the removal point
    if removal_version:
        removal = pkg_version.parse(removal_version)
        if current >= removal:
            raise RuntimeError(
                f"Feature deprecated in {deprecated_since} should have been removed in {removal_version}"
            )

    return True


class DeprecatedFeatureManager:
    """
    Manager for tracking and controlling deprecated features.

    Allows runtime control of deprecation warnings and tracks
    which deprecated features are still being used.
    """

    def __init__(self) -> None:
        """Initialize the deprecation manager."""
        self._suppressed_warnings: set = set()
        self._emitted_warnings: set = set()
        self._warning_count: dict = {}

    def suppress_warning(self, feature_name: str) -> None:
        """
        Suppress deprecation warnings for a specific feature.

        Args:
            feature_name: Name of the feature to suppress warnings for
        """
        self._suppressed_warnings.add(feature_name)

    def enable_warning(self, feature_name: str) -> None:
        """
        Re-enable deprecation warnings for a specific feature.

        Args:
            feature_name: Name of the feature to enable warnings for
        """
        self._suppressed_warnings.discard(feature_name)

    def is_suppressed(self, feature_name: str) -> bool:
        """
        Check if warnings are suppressed for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            True if suppressed, False otherwise
        """
        return feature_name in self._suppressed_warnings

    def record_warning(self, feature_name: str) -> None:
        """
        Record that a deprecation warning was emitted.

        Args:
            feature_name: Name of the feature that triggered the warning
        """
        self._emitted_warnings.add(feature_name)
        self._warning_count[feature_name] = self._warning_count.get(feature_name, 0) + 1

    def get_warning_stats(self) -> dict:
        """
        Get statistics on deprecation warnings.

        Returns:
            Dictionary with warning statistics
        """
        return {
            "unique_warnings": len(self._emitted_warnings),
            "total_count": sum(self._warning_count.values()),
            "per_feature": self._warning_count.copy(),
        }

    def reset_stats(self) -> None:
        """Reset warning statistics."""
        self._emitted_warnings.clear()
        self._warning_count.clear()


# Global deprecation manager instance
_deprecation_manager = DeprecatedFeatureManager()


def get_deprecation_manager() -> DeprecatedFeatureManager:
    """Get the global deprecation manager instance."""
    return _deprecation_manager
