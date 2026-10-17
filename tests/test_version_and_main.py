"""
Tests for version and main entry points.
"""

import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from apgi_framework import __version__

# Importing the `apgi_framework.__version__` submodule would replace the package's
# `__version__` string with the module, so read the file without importing it.
version_module = SimpleNamespace(
    **runpy.run_path(
        str(Path(__file__).resolve().parent.parent / "apgi_framework" / "__version__.py")
    )
)


def test_version_constants() -> None:
    """Test that version constants are present and correctly formatted."""
    assert isinstance(__version__, str)
    assert hasattr(version_module, "__version_info__")
    assert isinstance(version_module.__version_info__, tuple)
    assert len(version_module.__version_info__) >= 3

    assert hasattr(version_module, "__build__")
    assert hasattr(version_module, "__release_date__")
    assert hasattr(version_module, "CORE_VERSION")
    assert hasattr(version_module, "TEST_ENHANCEMENT_VERSION")
    assert hasattr(version_module, "GUI_VERSION")
    assert hasattr(version_module, "CLI_VERSION")


@patch("apgi_framework.cli.main")
def test_main_execution(mock_cli_main: Any) -> None:
    """Test the __main__ entry point."""

    # Simulate running as __main__
    # Note: Since the module is already imported, it might not run the if __name__ == "__main__" block
    # during import unless we reload it or simulate it explicitly.
    # But we can at least test that it imports and calls main if we were to invoke it.

    # Manual call to test the logic in __main__.py if it were to be executed
    # In most cases, we just want to ensure it references cli.main
    from apgi_framework.__main__ import main as imported_main

    assert imported_main is mock_cli_main
