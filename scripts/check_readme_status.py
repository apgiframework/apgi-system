#!/usr/bin/env python3
"""Check that README feature claims carry a status and that "verified" claims are true.

Rules:

* Every bullet under the feature-list sections (see ``FEATURE_SECTIONS``) and every
  ``utils/*.py`` heading must be marked ``✅ verified`` or ``🧭 planned``.
* A line marked ``✅ verified`` may only cite repository paths that exist.
* A ``utils/*.py`` heading marked ``✅ verified`` needs the file; ``🧭 planned`` needs it absent.

Exit status 1 lists every violation as ``README.md:<line>: <problem>``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
VERIFIED, PLANNED = "✅ verified", "🧭 planned"
FEATURE_SECTIONS = {
    "Features",
    "Framework Overview",
    "Core Capabilities",
    "Research Applications",
    "Testing",
}
UTILS_HEADING = re.compile(r"^###\s+\d+\.\s+.*\(`(utils/[\w./-]+\.py)`\)")
CODE_SPAN = re.compile(r"`([^`\n]+)`")
PATHLIKE = re.compile(r"^[\w.-]+(/[\w.-]+)+/?$|^[\w-]+\.(py|md|yaml|yml|toml|json)$")


def cited_paths(line: str) -> List[str]:
    paths = []
    for token in CODE_SPAN.findall(line):
        token = token.strip().rstrip(":,.")
        if token.startswith(("http", "-")) or any(c in token for c in "*<>{} "):
            continue
        if PATHLIKE.match(token):
            paths.append(token)
    return paths


def check_lines(lines: List[str], root: Path) -> List[str]:
    problems: List[str] = []
    section: Optional[str] = None
    in_fence = False
    for i, line in enumerate(lines, 1):
        if line.startswith("```"):
            in_fence = not in_fence
        if in_fence:
            continue
        heading = re.match(r"^#{2,4}\s+(.*)$", line)
        if heading:
            section = heading.group(1).strip()
        marked = VERIFIED in line or PLANNED in line
        m = UTILS_HEADING.match(line)
        if m:
            exists = (root / m.group(1)).exists()
            if not marked:
                problems.append(f"README.md:{i}: {m.group(1)} has no status marker")
            elif exists and PLANNED in line:
                problems.append(f"README.md:{i}: {m.group(1)} exists but is marked planned")
            elif not exists and VERIFIED in line:
                problems.append(
                    f"README.md:{i}: {m.group(1)} is marked verified but does not exist"
                )
            continue
        if section in FEATURE_SECTIONS and line.startswith("- ") and not marked:
            problems.append(f"README.md:{i}: feature bullet has no ✅ verified / 🧭 planned marker")
        if VERIFIED in line and PLANNED not in line:
            for path in cited_paths(line):
                if not (root / path).exists():
                    problems.append(f"README.md:{i}: marked verified but {path} does not exist")
    return problems


def main(argv: Optional[Iterable[str]] = None) -> int:
    root = Path(list(argv)[0]) if argv else ROOT
    problems = check_lines((root / "README.md").read_text(encoding="utf-8").splitlines(), root)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} README status problem(s).", file=sys.stderr)
        return 1
    print("README status markers consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
