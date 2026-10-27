#!/usr/bin/env python3
"""Fail when documentation uses wording listed in config/forbidden_language.yaml.

See that file for the rules and for how to allow a justified exception. Exit status 1
lists each hit as ``path:line: [rule-id] matched text -- message``.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "forbidden_language.yaml"
ALLOW_RE = re.compile(r"forbidden-language:\s*allow\s+([\w,\s-]+?)\s*(?:-->|$)")


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    rule: str
    text: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.text!r} -- {self.message}"


def load_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    config: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    for rule in config["rules"]:
        rule["_re"] = re.compile(rule["pattern"], re.IGNORECASE)
    words = [n for n in config.get("negations", []) if not n.startswith("n'")]
    words_re = "|".join(re.escape(n) for n in words)
    config["_neg"] = re.compile(
        rf"(?:\b(?:{words_re})\b|n't\b)\W*(?:\w+\W+){{0,3}}$", re.IGNORECASE
    )
    return config


def _allowed(lines: Sequence[str], index: int, rule_id: str) -> bool:
    for probe in (index, index - 1):
        if probe < 0:
            continue
        m = ALLOW_RE.search(lines[probe])
        if m and rule_id in {r.strip() for r in m.group(1).split(",")}:
            return True
    return False


def scan_text(rel: str, text: str, config: Dict[str, Any]) -> List[Hit]:
    hits: List[Hit] = []
    lines = text.splitlines()
    window = int(config.get("negation_window_chars", 30))
    in_fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:  # code samples may quote forbidden words
            continue
        for rule in config["rules"]:
            for m in rule["_re"].finditer(line):
                before = line[max(0, m.start() - window) : m.start()]
                if config["_neg"].search(before):
                    continue
                if _allowed(lines, i, rule["id"]):
                    continue
                hits.append(
                    Hit(rel, i + 1, rule["id"], m.group(0), " ".join(rule["message"].split()))
                )
    return hits


def targets(root: Path, config: Dict[str, Any]) -> List[Path]:
    selected: Dict[Path, None] = {}
    for pattern in config["include"]:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                selected[path] = None
    for pattern in config.get("exclude", []):
        for path in root.glob(pattern):
            selected.pop(path, None)
    return list(selected)


def scan_repo(root: Path = ROOT, config_path: Path = DEFAULT_CONFIG) -> List[Hit]:
    config = load_config(config_path)
    hits: List[Hit] = []
    for path in targets(root, config):
        hits.extend(
            scan_text(path.relative_to(root).as_posix(), path.read_text(encoding="utf-8"), config)
        )
    return hits


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(list(argv) if argv is not None else None)
    hits = scan_repo(args.root, args.config)
    for hit in hits:
        print(hit, file=sys.stderr)
    if hits:
        print(f"\n{len(hits)} forbidden-language hit(s).", file=sys.stderr)
        return 1
    print("no forbidden language found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
