#!/usr/bin/env python3
"""Fail unless every model class in apgi_framework is mapped to a registered model.

Reads config/model_registry.yaml. Class discovery is static (AST), so no framework
dependencies are imported. Problems reported:

* a candidate model class that is neither mapped to a model nor listed as a non-model;
* a mapped or non-model class path that does not exist in the source;
* a class listed as both, or mapped twice (rejected when the registry is parsed).
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

ROOT = Path(__file__).resolve().parent.parent


def _load_registry_module(root: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "_model_registry_standalone", root / "apgi_framework" / "model_registry.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def module_path(root: Path, dotted_module: str) -> Optional[Path]:
    base = root.joinpath(*dotted_module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def top_level_classes(path: Path) -> Dict[str, List[str]]:
    """Public and private top-level class names -> base class names."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out[node.name] = [ast.unparse(b).split(".")[-1] for b in node.bases]
    return out


def discover(root: Path, discovery: Dict[str, Any]) -> Set[str]:
    pattern = re.compile(discovery["name_pattern"])
    skip_bases = set(discovery.get("skip_bases", []))
    skip_dirs = set(discovery.get("skip_dirs", []))
    skip_files = set(discovery.get("skip_files", []))
    found: Set[str] = set()
    for top in discovery.get("roots", ["apgi_framework"]):
        for path in sorted((root / top).rglob("*.py")):
            rel = path.relative_to(root)
            if any(part in skip_dirs for part in rel.parts[:-1]) or rel.as_posix() in skip_files:
                continue
            try:
                classes = top_level_classes(path)
            except SyntaxError:
                continue
            module = ".".join(rel.with_suffix("").parts).removesuffix(".__init__")
            for name, bases in classes.items():
                if name.startswith("_") or not pattern.search(name):
                    continue
                if any(b in skip_bases or b.endswith(("Error", "Exception")) for b in bases):
                    continue
                found.add(f"{module}.{name}")
    return found


def check(root: Path = ROOT, registry_path: Optional[Path] = None) -> List[str]:
    mod = _load_registry_module(root)
    try:
        registry = mod.load_registry(registry_path or root / "config" / "model_registry.yaml")
    except mod.ModelRegistryError as exc:
        return [str(exc)]
    mapped = set(registry.class_index)
    non_models = set(registry.non_models)
    problems: List[str] = []
    for path in sorted(mapped | non_models):
        module, _, cls = path.rpartition(".")
        file = module_path(root, module)
        if file is None or cls not in top_level_classes(file):
            problems.append(f"{path}: listed in the registry but not found in the source")
    for path in sorted(discover(root, registry.discovery) - mapped - non_models):
        problems.append(
            f"{path}: model class is not mapped to a registered model "
            "(add it to a model's classes, or to non_models with a reason)"
        )
    return problems


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    problems = check(args.root)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} model-registry problem(s).", file=sys.stderr)
        return 1
    print("every model class is mapped to a registered model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
