"""scripts/check_import_cycles.py: graph building, cycle detection and the baseline."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "check_import_cycles", ROOT / "scripts" / "check_import_cycles.py"
)
assert spec and spec.loader
cyc = importlib.util.module_from_spec(spec)
sys.modules["check_import_cycles"] = cyc
spec.loader.exec_module(cyc)


def make_tree(root: Path, files: Dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def cycles(root: Path) -> set:
    return {cyc.cycle_key(c) for c in cyc.strongly_connected(cyc.build_graph(root, ("pkg",)))}


def test_the_repository_has_no_import_cycle_outside_the_baseline() -> None:
    assert cyc.check(ROOT) == []


def test_two_module_cycle(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "from pkg import b\n",
            "pkg/b.py": "import pkg.a\n",
            "pkg/c.py": "from pkg import a\n",
        },
    )
    assert cycles(tmp_path) == {"pkg.a, pkg.b"}


def test_relative_imports_and_three_module_cycle(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/sub/__init__.py": "",
            "pkg/sub/x.py": "from ..y import f\n",
            "pkg/y.py": "from .sub.z import g\n",
            "pkg/sub/z.py": "from . import x\n",
        },
    )
    assert cycles(tmp_path) == {"pkg.sub.x, pkg.sub.z, pkg.y"}


def test_child_importing_names_from_its_parent_package_is_a_cycle(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "pkg/__init__.py": "from .child import Thing\n",
            "pkg/child.py": "from pkg import Other\nclass Thing: ...\n",
        },
    )
    assert cycles(tmp_path) == {"pkg, pkg.child"}


def test_child_importing_a_sibling_module_is_not(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "pkg/__init__.py": "from .child import Thing\n",
            "pkg/child.py": "from .other import x\nclass Thing: ...\n",
            "pkg/other.py": "x = 1\n",
        },
    )
    assert cycles(tmp_path) == set()


def test_type_checking_and_function_level_imports_do_not_count(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from pkg import b\n",
            "pkg/b.py": "from pkg import a\ndef f():\n    from pkg import a as again\n",
            "pkg/c.py": "def f():\n    import pkg.d\n",
            "pkg/d.py": "import pkg.c\n",
        },
    )
    assert cycles(tmp_path) == set()


def test_try_except_import_still_counts(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "try:\n    from pkg import b\nexcept ImportError:\n    b = None\n",
            "pkg/b.py": "from pkg import a\n",
        },
    )
    assert cycles(tmp_path) == {"pkg.a, pkg.b"}


def test_baseline_allows_known_cycles_and_flags_new_and_stale(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "from pkg import b\n",
            "pkg/b.py": "from pkg import a\n",
        },
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"cycles": []}))
    found = cyc.check(tmp_path, baseline, ("pkg",))
    assert [p.startswith("new import cycle") for p in found] == [True]
    cyc.write_baseline(tmp_path, baseline, ("pkg",))
    assert cyc.check(tmp_path, baseline, ("pkg",)) == []
    (tmp_path / "pkg" / "b.py").write_text("x = 1\n")
    stale = cyc.check(tmp_path, baseline, ("pkg",))
    assert len(stale) == 1 and "no longer exists" in stale[0]


def test_main_exit_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_tree(tmp_path, {"apgi_framework/__init__.py": "", "scripts/placeholder": ""})
    assert cyc.main(["--root", str(tmp_path)]) == 0
