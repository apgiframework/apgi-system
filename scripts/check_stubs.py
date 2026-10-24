"""Quality gate QG-01: fail on undocumented stubs in the shipped packages.

A stub is a function whose body is only ``pass``, ``...`` or
``raise NotImplementedError``. It must say so in its docstring (the words
"stub" or "not implemented"); abstract methods, ``typing.overload`` signatures
and ``Protocol`` members are exempt.

Stubs that predate the gate are recorded in ``scripts/stub_baseline.json`` and
must be documented or implemented over time; the gate fails on any stub outside
the baseline and on baseline entries that no longer exist (so the baseline can
only shrink). Regenerate it with ``--update-baseline`` after fixing stubs.
"""

import json
import ast
import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "stub_baseline.json"
PACKAGES = ("apgi_framework", "api", "apgi_gui")
MARKERS = ("stub", "not implemented")
EXEMPT_DECORATORS = {"abstractmethod", "overload"}


def _decorator_names(node):
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        yield getattr(target, "attr", getattr(target, "id", ""))


def _is_stub_body(body):
    statements = body[1:] if _has_docstring(body) else body
    if not statements:
        return True
    for stmt in statements:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            if stmt.value.value is Ellipsis:
                continue
        if isinstance(stmt, ast.Raise):
            exc = stmt.exc.func if isinstance(stmt.exc, ast.Call) else stmt.exc
            if getattr(exc, "id", "") == "NotImplementedError":
                continue
        return False
    return True


def _has_docstring(body):
    return (
        bool(body)
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    )


def undocumented_stubs(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    protocol_classes = {
        id(member)
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef)
        and any(getattr(base, "id", getattr(base, "attr", "")) == "Protocol" for base in cls.bases)
        for member in cls.body
    }
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if id(node) in protocol_classes or EXEMPT_DECORATORS & set(_decorator_names(node)):
            continue
        if not _is_stub_body(node.body):
            continue
        doc = (ast.get_docstring(node) or "").lower()
        if not any(marker in doc for marker in MARKERS):
            found.append((node.lineno, node.name))
    return found


def collect():
    counts = collections.Counter()
    for package in PACKAGES:
        for path in sorted((ROOT / package).rglob("*.py")):
            if "tests" in path.relative_to(ROOT).parts:
                continue
            for _, name in undocumented_stubs(path):
                counts[f"{path.relative_to(ROOT).as_posix()}::{name}"] += 1
    return counts


def main():
    counts = collect()
    if "--update-baseline" in sys.argv:
        BASELINE.write_text(json.dumps(dict(sorted(counts.items())), indent=2) + "\n")
        print(f"Baseline written: {sum(counts.values())} stub(s)")
        return 0
    baseline = collections.Counter(json.loads(BASELINE.read_text())) if BASELINE.exists() else {}
    new = {k: v - baseline.get(k, 0) for k, v in counts.items() if v > baseline.get(k, 0)}
    fixed = {k: v - counts.get(k, 0) for k, v in baseline.items() if v > counts.get(k, 0)}
    for key in sorted(new):
        print(f"undocumented stub not in baseline: {key}")
    for key in sorted(fixed):
        print(f"baseline entry no longer a stub (run --update-baseline): {key}")
    if new or fixed:
        print("QG-01 FAILED")
        return 1
    print(f"QG-01 passed: {sum(counts.values())} baselined stub(s), none new")
    return 0


if __name__ == "__main__":
    sys.exit(main())
