#!/usr/bin/env python3
"""Detect import cycles between modules of the project's own packages.

Builds the module-level import graph with the AST (``if TYPE_CHECKING`` blocks and imports
inside functions are ignored, because they do not run at import time), finds strongly
connected components, and compares them with ``scripts/import_cycles_baseline.json``.

* a cycle not in the baseline fails the check (a new circular dependency);
* a baseline entry that no longer exists fails too, so the baseline only ever shrinks;
* ``--update-baseline`` rewrites the baseline from the current graph.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Set

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "import_cycles_baseline.json"
PACKAGES = ("apgi_framework", "apgi_gui", "api", "utils")
SKIP_PARTS = {"tests", "__pycache__"}


def module_name(root: Path, path: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def collect_modules(root: Path, packages: Iterable[str] = PACKAGES) -> Dict[str, Path]:
    modules: Dict[str, Path] = {}
    for package in packages:
        base = root / package
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if SKIP_PARTS & set(path.relative_to(root).parts):
                continue
            modules[module_name(root, path)] = path
    return modules


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_level_imports(body: List[ast.stmt]) -> Iterable[ast.stmt]:
    """Imports that execute when the module is imported (not in functions or TYPE_CHECKING)."""
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, ast.If):
            if _is_type_checking(node.test):
                yield from _module_level_imports(node.orelse)
            else:
                yield from _module_level_imports(node.body)
                yield from _module_level_imports(node.orelse)
        elif isinstance(node, ast.Try):
            for block in (node.body, node.orelse, node.finalbody):
                yield from _module_level_imports(block)
            for handler in node.handlers:
                yield from _module_level_imports(handler.body)
        elif isinstance(node, (ast.With, ast.For, ast.While)):
            yield from _module_level_imports(node.body)


def _resolve(modules: Dict[str, Path], current: str, is_package: bool, node: ast.stmt) -> Set[str]:
    """Modules whose *contents* ``node`` needs at import time.

    Python also runs every parent package's ``__init__`` when a submodule is imported, but
    that alone cannot fail on a cycle, so implicit parents are not edges. ``from pkg import
    name`` is an edge to ``pkg`` itself only when ``name`` is not a submodule of ``pkg``.
    """
    targets: Set[str] = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name in modules:
                targets.add(alias.name)
        return targets
    assert isinstance(node, ast.ImportFrom)
    if node.level:
        pkg_parts = current.split(".") if is_package else current.split(".")[:-1]
        if node.level > 1:
            pkg_parts = pkg_parts[: len(pkg_parts) - (node.level - 1)]
        base = ".".join(pkg_parts + ([node.module] if node.module else []))
    else:
        base = node.module or ""
    if not base:
        return targets
    needs_base = False
    for alias in node.names:
        candidate = f"{base}.{alias.name}"
        if candidate in modules:
            targets.add(candidate)
        else:
            needs_base = True
    if needs_base and base in modules:
        targets.add(base)
    return targets


def build_graph(root: Path, packages: Iterable[str] = PACKAGES) -> Dict[str, Set[str]]:
    modules = collect_modules(root, packages)
    graph: Dict[str, Set[str]] = {name: set() for name in modules}
    for name, path in modules.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        is_package = path.name == "__init__.py"
        for node in _module_level_imports(tree.body):
            for target in _resolve(modules, name, is_package, node):
                if target != name:
                    graph[name].add(target)
    return graph


def strongly_connected(graph: Dict[str, Set[str]]) -> List[FrozenSet[str]]:
    """Tarjan's algorithm, iterative; returns components with more than one module."""
    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    on_stack: Set[str] = set()
    stack: List[str] = []
    components: List[FrozenSet[str]] = []
    counter = 0
    for start in graph:
        if start in index:
            continue
        work = [(start, iter(sorted(graph[start])))]
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(graph[child]))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                members = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    members.append(member)
                    if member == node:
                        break
                if len(members) > 1:
                    components.append(frozenset(members))
    return components


def cycle_key(component: FrozenSet[str]) -> str:
    return ", ".join(sorted(component))


def load_baseline(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {cycle_key(frozenset(group)) for group in json.loads(path.read_text())["cycles"]}


def check(
    root: Path = ROOT, baseline_path: Path = BASELINE, packages: Iterable[str] = PACKAGES
) -> List[str]:
    components = strongly_connected(build_graph(root, packages))
    current = {cycle_key(c) for c in components}
    baseline = load_baseline(baseline_path)
    problems = [f"new import cycle: {key}" for key in sorted(current - baseline)]
    problems += [
        f"baseline cycle no longer exists (remove it from {baseline_path.name}): {key}"
        for key in sorted(baseline - current)
    ]
    return problems


def write_baseline(
    root: Path = ROOT, baseline_path: Path = BASELINE, packages: Iterable[str] = PACKAGES
) -> int:
    components = sorted(sorted(c) for c in strongly_connected(build_graph(root, packages)))
    baseline_path.write_text(json.dumps({"cycles": components}, indent=2) + "\n", encoding="utf-8")
    return len(components)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    baseline = args.root / "scripts" / "import_cycles_baseline.json"
    if args.update_baseline:
        print(f"baseline written: {write_baseline(args.root, baseline)} cycle(s)")
        return 0
    problems = check(args.root, baseline)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} import-cycle problem(s).", file=sys.stderr)
        return 1
    print("no new import cycles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
