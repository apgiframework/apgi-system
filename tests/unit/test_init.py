"""
Unit tests for apgi_framework package initialization.

This module tests that the package can be imported and that
all expected classes are available from the package level.
"""


def test_package_imports() -> None:
    """Test that main classes can be imported from apgi_framework package."""
    from apgi_framework import (
        ActiveInferenceEngine,
        FreeEnergyCalculator,
        HierarchicalPredictor,
        PrecisionWeighting,
    )

    # Verify classes are imported correctly
    assert ActiveInferenceEngine is not None
    assert FreeEnergyCalculator is not None
    assert HierarchicalPredictor is not None
    assert PrecisionWeighting is not None


def test_package_version() -> None:
    """Test that package version is defined."""
    import apgi_framework

    assert hasattr(apgi_framework, "__version__")
    assert isinstance(apgi_framework.__version__, str)
    assert len(apgi_framework.__version__) > 0


def test_package_all() -> None:
    """Test that __all__ is properly defined."""
    import apgi_framework

    assert hasattr(apgi_framework, "__all__")
    assert isinstance(apgi_framework.__all__, list)
    core = [
        "ActiveInferenceEngine",
        "FreeEnergyCalculator",
        "HierarchicalPredictor",
        "PrecisionWeighting",
    ]
    names = apgi_framework.__all__
    assert len(names) == len(set(names)), "__all__ must not contain duplicates"
    for name in core:
        assert name in names, f"{name} must stay exported"
    for name in names:
        assert hasattr(apgi_framework, name), f"__all__ lists missing attribute {name}"
