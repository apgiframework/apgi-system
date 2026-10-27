"""Repo-policy scripts: identifiers, README status markers, forbidden language."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ids_mod = load("sync_identifiers")
status_mod = load("check_readme_status")
lang_mod = load("check_forbidden_language")


class TestRealRepoIsClean:
    def test_identifiers_are_consistent(self) -> None:
        assert [str(f) for f in ids_mod.check(ROOT)] == []

    def test_readme_status_markers(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
        assert status_mod.check_lines(text, ROOT) == []

    def test_no_forbidden_language(self) -> None:
        assert [str(h) for h in lang_mod.scan_repo(ROOT)] == []


@pytest.fixture
def ids() -> Dict[str, Any]:
    data = ids_mod.load_identifiers(ROOT)
    return data


class TestIdentifiers:
    def test_orcid_checksum(self) -> None:
        assert ids_mod.orcid_checksum_ok("0000-0002-1825-0097")  # ORCID's documented sample iD
        assert ids_mod.orcid_checksum_ok("0009-0005-0328-2896")
        assert not ids_mod.orcid_checksum_ok("0009-0005-0328-2897")

    def test_divergent_doi_orcid_and_repo_are_reported(self, ids: Dict[str, Any]) -> None:
        text = "\n".join(
            [
                "doi 10.5281/zenodo.99999",
                "ORCID: 0000-0002-1825-0097",
                "git clone https://github.com/someone/apgi-system.git",
                "card 4532-1234-5678-9010",  # not an ORCID line
                "example 10.5281/zenodo.12345",  # allowed example
                "see https://github.com/your-org/apgi-experiments",  # allowed placeholder
                f"ok {ids['doi']} https://github.com/apgiframework/apgi-framework",
            ]
        )
        found = ids_mod.scan_text("x.md", text, ids)
        assert [f.line for f in found] == [1, 2, 3]
        assert "zenodo.99999" in found[0].message
        assert "someone/apgi-system" in found[2].message

    def test_check_detects_stale_readme_block_and_cff(
        self, tmp_path: Path, ids: Dict[str, Any]
    ) -> None:
        (tmp_path / "apgi_framework").mkdir()
        (tmp_path / "apgi_framework" / "__version__.py").write_text(
            '__version__ = "1.1.0"\n__release_date__ = "2026-10-17"\n'
        )
        data = json.loads((ROOT / "identifiers.json").read_text())
        (tmp_path / "identifiers.json").write_text(json.dumps(data))
        readme = f"# t\n{ids_mod.BEGIN}\nold\n{ids_mod.END}\n"
        (tmp_path / "README.md").write_text(readme)
        cff = (ROOT / "CITATION.cff").read_text().replace(ids["doi"], "10.5281/zenodo.1")
        (tmp_path / "CITATION.cff").write_text(cff)
        messages = [str(f) for f in ids_mod.check(tmp_path)]
        assert any("README.md" in m and "out of date" in m for m in messages)
        assert any("CITATION.cff" in m and "doi" in m for m in messages)

        assert set(ids_mod.write(tmp_path)) == {"README.md", "CITATION.cff"}
        assert ids_mod.check(tmp_path) == []
        assert ids_mod.write(tmp_path) == []  # idempotent

    def test_version_comes_from_the_package(self, ids: Dict[str, Any]) -> None:
        from apgi_framework.__version__ import __version__

        assert ids["version"] == __version__


class TestReadmeStatus:
    def lines(self, text: str) -> list:
        return text.splitlines()

    def test_unmarked_feature_bullet(self, tmp_path: Path) -> None:
        problems = status_mod.check_lines(self.lines("## Features\n\n- Fast\n"), tmp_path)
        assert len(problems) == 1 and "no ✅ verified" in problems[0]

    def test_verified_claim_must_cite_existing_paths(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "b.py").write_text("")
        ok = status_mod.check_lines(self.lines("- x ✅ verified (`a/b.py`)"), tmp_path)
        bad = status_mod.check_lines(self.lines("- x ✅ verified (`a/missing.py`)"), tmp_path)
        assert ok == [] and "missing.py" in bad[0]

    def test_utils_headings(self, tmp_path: Path) -> None:
        (tmp_path / "utils").mkdir()
        (tmp_path / "utils" / "here.py").write_text("")
        text = "\n".join(
            [
                "### 1. Here (`utils/here.py`) — ✅ verified",
                "### 2. Gone (`utils/gone.py`) — ✅ verified",
                "### 3. Planned but present (`utils/here.py`) — 🧭 planned",
                "### 4. Bare (`utils/here.py`)",
            ]
        )
        problems = status_mod.check_lines(self.lines(text), tmp_path)
        assert [p.split(":")[1] for p in problems] == ["2", "3", "4"]


class TestForbiddenLanguage:
    @pytest.fixture(scope="class")
    def config(self) -> Dict[str, Any]:
        return lang_mod.load_config()

    @pytest.mark.parametrize(
        "text,rule",
        [
            ("These results prove the theory.", "proof-claim"),
            ("The model is clinically validated.", "clinical-claim"),
            ("It can diagnose patients early.", "diagnostic-claim"),
            ("We guarantee convergence.", "certainty-claim"),
            ("A revolutionary approach.", "hype"),
            ("The tool measures consciousness directly.", "measures-consciousness"),
        ],
    )
    def test_rule_fires(self, config: Dict[str, Any], text: str, rule: str) -> None:
        assert [h.rule for h in lang_mod.scan_text("t.md", text, config)] == [rule]

    @pytest.mark.parametrize(
        "text",
        [
            "This does not prove anything.",
            "The tests can't guarantee correctness.",
            "No result here is conclusive.",
            "Run system validation to diagnose issues.",
            "A proof of concept.",
            "<!-- forbidden-language: allow hype -->\nA revolutionary approach.",
            "A revolutionary approach. <!-- forbidden-language: allow hype -->",
            "```\nthis proves it\n```",
        ],
    )
    def test_not_flagged(self, config: Dict[str, Any], text: str) -> None:
        assert lang_mod.scan_text("t.md", text, config) == []

    def test_allow_marker_is_rule_specific(self, config: Dict[str, Any]) -> None:
        text = "<!-- forbidden-language: allow hype -->\nWe guarantee a breakthrough."
        assert [h.rule for h in lang_mod.scan_text("t.md", text, config)] == ["certainty-claim"]

    def test_negation_does_not_leak_across_sentences(self, config: Dict[str, Any]) -> None:
        text = "Nothing else matters here and it proves a lot."
        assert [h.rule for h in lang_mod.scan_text("t.md", text, config)] == ["proof-claim"]

    def test_main_exit_codes(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        (tmp_path / "README.md").write_text("It will revolutionary. A revolutionary tool.\n")
        cfg = tmp_path / "c.yaml"
        cfg.write_text((ROOT / "config" / "forbidden_language.yaml").read_text())
        assert lang_mod.main(["--root", str(tmp_path), "--config", str(cfg)]) == 1
        (tmp_path / "README.md").write_text("Plain description.\n")
        assert lang_mod.main(["--root", str(tmp_path), "--config", str(cfg)]) == 0


class TestRegisterEvidence:
    def test_every_evidence_path_exists_and_markdown_is_current(self) -> None:
        reg = load("register_status")
        assert reg.problems(reg.load(ROOT), ROOT) == []
        assert reg.OUT.read_text(encoding="utf-8") == reg.render(reg.load(ROOT))

    def test_missing_path_and_unknown_status_are_reported(self) -> None:
        reg = load("register_status")
        bad = {
            "X": {"status": "done", "evidence": ["nope.py"], "note": ""},
            "Y": {"status": "implemented", "evidence": [], "note": ""},
        }
        found = reg.problems(bad, ROOT)
        assert len(found) == 3 and any("missing" in f for f in found)
