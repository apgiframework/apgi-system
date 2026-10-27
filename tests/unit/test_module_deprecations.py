"""Deprecated legacy entry points warn when loaded, using apgi_framework.utils.deprecation."""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

from apgi_framework.utils import deprecation
from apgi_framework.utils.deprecation import (
    APGIDeprecationWarning,
    get_deprecation_manager,
    warn_deprecated_module,
)

ROOT = Path(__file__).resolve().parents[2]
LEGACY = [
    "APGI_GUI",
    "APGI_Application_GUI",
    "Psychological_States_GUI",
    "Assistant_GUI",
    "GUI-Launcher",
    "Tests_GUI",
    "Utils_GUI",
]


@pytest.fixture(autouse=True)
def fresh_state():  # type: ignore[no-untyped-def]
    deprecation._warned_modules.clear()
    manager = get_deprecation_manager()
    manager.reset_stats()
    yield
    deprecation._warned_modules.clear()
    manager.reset_stats()
    for name in list(manager._suppressed_warnings):
        manager.enable_warning(name)


class TestWarnDeprecatedModule:
    def test_warns_once_per_module_with_replacement_and_removal(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert warn_deprecated_module("old_mod", "new.mod", since="1.1.0", removal="2.0.0")
            assert not warn_deprecated_module("old_mod", "new.mod", since="1.1.0", removal="2.0.0")
        assert len(caught) == 1 and caught[0].category is APGIDeprecationWarning
        text = str(caught[0].message)
        assert "old_mod" in text and "1.1.0" in text and "new.mod" in text and "2.0.0" in text

    def test_use_is_recorded_in_the_manager(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warn_deprecated_module("a", "b", since="1.0")
            warn_deprecated_module("c", "b", since="1.0")
        assert get_deprecation_manager().get_warning_stats()["per_feature"] == {"a": 1, "c": 1}

    def test_suppressed_module_does_not_warn(self) -> None:
        get_deprecation_manager().suppress_warning("quiet")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert not warn_deprecated_module("quiet", "b", since="1.0")
        assert caught == []

    def test_warning_points_at_the_importer(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_deprecated_module(
                "m", "n", since="1.0"
            )  # stacklevel=3 skips helper + this test's caller
        assert caught and caught[0].filename != deprecation.__file__


class TestLegacyEntryPoints:
    @pytest.mark.parametrize("name", LEGACY)
    def test_each_listed_legacy_module_calls_the_helper_with_its_own_name(self, name: str) -> None:
        tree = ast.parse((ROOT / f"{name}.py").read_text(encoding="utf-8"))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "warn_deprecated_module"
        ]
        assert len(calls) == 1
        args = calls[0].args
        assert ast.literal_eval(args[0]) == name and ast.literal_eval(args[1]) == "apgi_gui.main"

    def test_readme_lists_exactly_these_legacy_entry_points(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        section = readme.split("### Legacy Entry Points (Deprecated)")[1].split("###")[0]
        listed = {Path(tok).stem for tok in __import__("re").findall(r"`([\w-]+\.py)`", section)}
        assert listed == set(LEGACY)

    def test_importing_a_legacy_module_emits_the_warning(self, tmp_path: Path) -> None:
        import importlib.util
        import sys

        src = tmp_path / "Legacy_Demo.py"
        src.write_text(
            "try:\n"
            "    from apgi_framework.utils.deprecation import warn_deprecated_module\n"
            "    warn_deprecated_module('Legacy_Demo', 'apgi_gui.main', since='1.1.0', removal='2.0.0')\n"
            "except ImportError:\n"
            "    pass\n"
        )
        spec = importlib.util.spec_from_file_location("Legacy_Demo", src)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sys.modules["Legacy_Demo"] = module
            spec.loader.exec_module(module)
        sys.modules.pop("Legacy_Demo", None)
        assert [str(w.message) for w in caught if w.category is APGIDeprecationWarning] == [
            "Module Legacy_Demo is deprecated since version 1.1.0. Use apgi_gui.main instead. "
            "It will be removed in version 2.0.0."
        ]
