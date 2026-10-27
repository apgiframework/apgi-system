"""Model registry: every model class is mapped; unmapped parameters raise."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from apgi_framework.model_registry import (
    ModelRegistryError,
    UnknownModelError,
    UnmappedModelError,
    UnmappedParameterError,
    load_registry,
    parse_registry,
    require_mapped_parameters,
    require_registered_model,
)
from utils.parameter_validator import parameter_validator, require_valid_parameters

ROOT = Path(__file__).resolve().parents[2]


def load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "check_model_registry", ROOT / "scripts" / "check_model_registry.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_model_registry"] = module
    spec.loader.exec_module(module)
    return module


check_script = load_script()


class TestEveryModelClassIsMapped:
    def test_repository_has_no_unmapped_model_class(self) -> None:
        assert check_script.check(ROOT) == []

    def test_runtime_lookup_of_real_classes(self) -> None:
        from apgi_framework.engines import PredictiveIgnitionNetwork, ThresholdManager
        from apgi_framework.interoception.body_model import BodyModel

        assert require_registered_model(ThresholdManager) == "M-APGI-FORMAL"
        assert require_registered_model(PredictiveIgnitionNetwork) == "M-APGI-FORMAL"
        assert require_registered_model(BodyModel) == "M-BODY"

    def test_an_instance_resolves_like_its_class(self) -> None:
        class Impostor:
            pass

        with pytest.raises(UnmappedModelError, match="not mapped to a registered model"):
            require_registered_model(Impostor())

    def test_new_unmapped_model_class_is_reported(self, tmp_path: Path) -> None:
        (tmp_path / "config").mkdir()
        (tmp_path / "apgi_framework").mkdir()
        (tmp_path / "apgi_framework" / "model_registry.py").write_text(
            (ROOT / "apgi_framework" / "model_registry.py").read_text()
        )
        (tmp_path / "apgi_framework" / "newmodel.py").write_text(
            "class ShinyModel:\n    pass\n\nclass _PrivateModel:\n    pass\n\n"
            "class ModelError(Exception):\n    pass\n\nclass Helper:\n    pass\n"
        )
        registry = {
            "version": 1,
            "discovery": {
                "name_pattern": "(Model)$",
                "roots": ["apgi_framework"],
                "skip_files": ["apgi_framework/model_registry.py"],
            },
            "models": [],
        }
        (tmp_path / "config" / "model_registry.yaml").write_text(yaml.safe_dump(registry))
        problems = check_script.check(tmp_path)
        assert len(problems) == 1 and "apgi_framework.newmodel.ShinyModel" in problems[0]

        registry["models"] = [
            {"id": "M-X", "classes": ["apgi_framework.newmodel.ShinyModel"], "parameters": []}
        ]
        (tmp_path / "config" / "model_registry.yaml").write_text(yaml.safe_dump(registry))
        assert check_script.check(tmp_path) == []

    def test_dangling_registry_entry_is_reported(self, tmp_path: Path) -> None:
        (tmp_path / "config").mkdir()
        (tmp_path / "apgi_framework").mkdir()
        (tmp_path / "apgi_framework" / "model_registry.py").write_text(
            (ROOT / "apgi_framework" / "model_registry.py").read_text()
        )
        registry = {
            "version": 1,
            "discovery": {"name_pattern": "(Model)$", "roots": ["apgi_framework"]},
            "models": [{"id": "M-X", "classes": ["apgi_framework.gone.GhostModel"]}],
        }
        (tmp_path / "config" / "model_registry.yaml").write_text(yaml.safe_dump(registry))
        assert "not found in the source" in check_script.check(tmp_path)[0]

    def test_class_cannot_be_mapped_twice_or_mapped_and_excluded(self) -> None:
        with pytest.raises(ModelRegistryError, match="mapped to both"):
            parse_registry(
                {"models": [{"id": "A", "classes": ["m.C"]}, {"id": "B", "classes": ["m.C"]}]}
            )
        with pytest.raises(ModelRegistryError, match="duplicate model id"):
            parse_registry({"models": [{"id": "A"}, {"id": "A"}]})
        with pytest.raises(ModelRegistryError, match="both mapped and listed as non-models"):
            parse_registry(
                {
                    "models": [{"id": "A", "classes": ["m.C"]}],
                    "non_models": [{"class": "m.C", "reason": "x"}],
                }
            )


class TestUnmappedParametersRaise:
    def test_mapped_parameters_pass(self) -> None:
        require_mapped_parameters("M-APGI-FORMAL", {"tau_S": 0.5, "Pi_e": 1.0})
        require_mapped_parameters("M-APGI-FORMAL", ["beta"])
        require_mapped_parameters("M-BODY", {})

    def test_unmapped_parameter_raises_and_names_every_offender(self) -> None:
        with pytest.raises(UnmappedParameterError) as info:
            require_mapped_parameters("M-APGI-FORMAL", {"tau_S": 0.5, "tau_X": 1, "zeta": 2})
        message = str(info.value)
        assert (
            "tau_X" in message
            and "zeta" in message
            and "tau_S" in message.split("mapped parameters")[1]
        )
        assert isinstance(info.value, ValueError)

    def test_a_parameter_mapped_for_another_model_is_still_unmapped(self) -> None:
        with pytest.raises(UnmappedParameterError):
            require_mapped_parameters("M-BODY", {"tau_S": 0.5})

    def test_unknown_model_id(self) -> None:
        with pytest.raises(UnknownModelError, match="M-NOPE"):
            require_mapped_parameters("M-NOPE", {})

    def test_registry_matches_the_parameter_validator_schema(self) -> None:
        formal = load_registry().model("M-APGI-FORMAL").parameters
        assert formal == frozenset(parameter_validator.list_valid_parameters())

    def test_require_valid_parameters_raises_instead_of_reporting(self) -> None:
        assert require_valid_parameters({"tau_S": 0.5})["valid"]
        with pytest.raises(UnmappedParameterError):
            require_valid_parameters({"made_up": 1.0})
        with pytest.raises(ValueError, match="above maximum"):
            require_valid_parameters({"tau_S": 5.0})
        # the non-raising API is unchanged
        assert not parameter_validator.validate({"made_up": 1.0})["valid"]
