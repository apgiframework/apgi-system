"""Protocol specification 1.1.0: loader fields, verdicts and release metadata."""

import re
from pathlib import Path

import pytest

from apgi_framework import __version__
from apgi_framework.analysis.statistical_report_generator import (
    FalsificationConclusion,
    StatisticalReportGenerator,
)
from apgi_framework.config.protocol_spec import (
    SPECIFICATION_VERSION,
    Verdict,
    classify_prediction,
    load_all_protocols,
    load_protocol,
    registered_ranges,
)

ROOT = Path(__file__).resolve().parent.parent
FRAMEWORK_PROTOCOLS = ROOT.parent / "apgi-framework" / "protocols"


class TestSpecificationRelease:
    def test_every_protocol_is_release_1_1_0(self):
        specs = load_all_protocols()
        assert sorted(specs) == [f"APGI-P0{n}" for n in range(8)]
        for spec in specs.values():
            assert spec.version == SPECIFICATION_VERSION
            assert spec.is_current_specification
            assert spec.study_status == "pre_collection"

    def test_kappa_is_uncalibrated(self):
        for spec in load_all_protocols().values():
            kappa = spec.apgi_parameters["kappa"]
            assert kappa["value"] is None
            assert kappa["status"] == "uncalibrated"

    def test_registered_ranges(self):
        assert registered_ranges() == {
            "theta_0": (0.25, 0.85),
            "gamma_sig": (2, 7.5),
            "beta_SM": (0.3, 0.7),
        }

    @pytest.mark.parametrize(
        "protocol, expected",
        [(1, (62, 59, 48)), ("P3", (40, 36, 36)), ("APGI-P04", (64, 55, 48)), (5, (52, 44, 36))],
    )
    def test_cohort_sizes(self, protocol, expected):
        people = load_protocol(protocol).participants
        assert (people["n_target"], people["n_analyzable_target"], people["n_minimum"]) == expected

    def test_sub_prediction_fields(self):
        spec = load_protocol(1)
        pred = spec.get_prediction("Pred 1.A")
        assert pred is not None and pred.is_confirmatory
        assert "not evidence of falsification" in pred.indeterminate_rule
        assert pred.confirming_thresholds["alpha"] == pytest.approx(0.0125)
        assert not spec.get_prediction("Pred 1.E").is_confirmatory
        assert spec.get_prediction("Pred 9.Z") is None

    def test_unknown_protocol(self):
        with pytest.raises(KeyError):
            load_protocol(9)

    def test_missing_folder(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_all_protocols(tmp_path / "absent")

    @pytest.mark.skipif(not FRAMEWORK_PROTOCOLS.is_dir(), reason="apgi-framework not checked out")
    def test_byte_identical_to_apgi_framework(self):
        ours = sorted(
            p.relative_to(ROOT / "protocols") for p in (ROOT / "protocols").rglob("*.json")
        )
        theirs = sorted(
            p.relative_to(FRAMEWORK_PROTOCOLS) for p in FRAMEWORK_PROTOCOLS.rglob("*.json")
        )
        assert ours == theirs
        for rel in ours:
            assert (ROOT / "protocols" / rel).read_bytes() == (
                FRAMEWORK_PROTOCOLS / rel
            ).read_bytes()


class TestVerdicts:
    @pytest.mark.parametrize(
        "confirmed, falsified, expected",
        [
            (True, False, Verdict.CONFIRMED),
            (True, True, Verdict.CONFIRMED),
            (False, True, Verdict.FALSIFIED),
            (False, False, Verdict.INDETERMINATE),
        ],
    )
    def test_failure_to_confirm_is_not_falsification(self, confirmed, falsified, expected):
        assert classify_prediction(confirmed, falsified) is expected

    def test_nonsignificant_soma_bias_is_not_falsification(self):
        conclusion = StatisticalReportGenerator()._determine_falsification_conclusion(
            "soma_bias_absence",
            p_value=0.4,
            effect_size=1.02,
            evidence_strength=0.2,
            theoretical_predictions={},
        )
        assert conclusion is FalsificationConclusion.INCONCLUSIVE


class TestReleaseMetadata:
    def test_version_is_1_1_0(self):
        assert __version__ == "1.1.0"

    def test_versions_agree(self):
        import runpy

        import api

        # Read the file rather than importing the submodule, which would shadow the string.
        module = runpy.run_path(str(ROOT / "apgi_framework" / "__version__.py"))
        assert api.__version__ == module["__version__"] == __version__
        assert module["__version_info__"] == tuple(int(x) for x in __version__.split("."))
        cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        assert re.search(r'^version: "(.+)"$', cff, re.M).group(1) == __version__
        assert f"## [{__version__}]" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        assert f"version={{{__version__}}}" in (ROOT / "README.md").read_text(encoding="utf-8")
