"""Registry mapping model classes and their parameters to registered models (R-001).

``config/model_registry.yaml`` is the source of truth. This module loads it and offers
two guards:

* :func:`require_registered_model` - a class (or instance) must belong to a registered model;
* :func:`require_mapped_parameters` - every parameter passed to a model must be one the
  registry maps for it; an unmapped name raises :class:`UnmappedParameterError`.

The module imports only the standard library and PyYAML so build-time checks can load it
without importing the whole framework.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple, Union

import yaml

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "config" / "model_registry.yaml"


class ModelRegistryError(Exception):
    """Base class for registry problems."""


class UnknownModelError(ModelRegistryError, KeyError):
    """A model id is not registered."""


class UnmappedModelError(ModelRegistryError, TypeError):
    """A class does not belong to any registered model."""


class UnmappedParameterError(ModelRegistryError, ValueError):
    """A parameter name is not mapped for the model it was passed to."""


@dataclass(frozen=True)
class RegisteredModel:
    model_id: str
    name: str
    description: str
    parameters: FrozenSet[str]
    classes: Tuple[str, ...]
    equations: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelRegistry:
    models: Mapping[str, RegisteredModel]
    non_models: Mapping[str, str]  # dotted class path -> reason
    discovery: Mapping[str, Any]

    def model(self, model_id: str) -> RegisteredModel:
        try:
            return self.models[model_id]
        except KeyError:
            raise UnknownModelError(
                f"unknown model {model_id!r}; registered: {sorted(self.models)}"
            ) from None

    def model_id_for(self, cls_or_obj: Union[type, object]) -> Optional[str]:
        cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)
        path = f"{cls.__module__}.{cls.__qualname__}"
        for model in self.models.values():
            if path in model.classes:
                return model.model_id
        return None

    @property
    def class_index(self) -> Dict[str, str]:
        return {c: m.model_id for m in self.models.values() for c in m.classes}


def parse_registry(raw: Mapping[str, Any]) -> ModelRegistry:
    """Build a registry from the parsed YAML, rejecting duplicate ids and doubly-mapped classes."""
    models: Dict[str, RegisteredModel] = {}
    seen_classes: Dict[str, str] = {}
    for entry in raw.get("models", []):
        model_id = str(entry["id"])
        if model_id in models:
            raise ModelRegistryError(f"duplicate model id {model_id!r}")
        classes = tuple(str(c) for c in entry.get("classes", []))
        for cls in classes:
            if cls in seen_classes:
                raise ModelRegistryError(
                    f"{cls} is mapped to both {seen_classes[cls]} and {model_id}"
                )
            seen_classes[cls] = model_id
        models[model_id] = RegisteredModel(
            model_id=model_id,
            name=str(entry.get("name", model_id)),
            description=str(entry.get("description", "")),
            parameters=frozenset(str(p) for p in entry.get("parameters", [])),
            classes=classes,
            equations=tuple(str(e) for e in entry.get("equations", [])),
        )
    non_models = {str(n["class"]): str(n["reason"]) for n in raw.get("non_models", [])}
    both = sorted(set(non_models) & set(seen_classes))
    if both:
        raise ModelRegistryError(f"classes both mapped and listed as non-models: {both}")
    return ModelRegistry(models, non_models, dict(raw.get("discovery", {})))


@functools.lru_cache(maxsize=4)
def _load(path: str) -> ModelRegistry:
    return parse_registry(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load_registry(path: Optional[Path] = None) -> ModelRegistry:
    return _load(str(path or DEFAULT_REGISTRY_PATH))


def require_registered_model(
    cls_or_obj: Union[type, object], *, registry: Optional[ModelRegistry] = None
) -> str:
    """Return the model id for a class or instance, or raise :class:`UnmappedModelError`."""
    reg = registry or load_registry()
    model_id = reg.model_id_for(cls_or_obj)
    if model_id is None:
        cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)
        raise UnmappedModelError(
            f"{cls.__module__}.{cls.__qualname__} is not mapped to a registered model; "
            "add it to config/model_registry.yaml"
        )
    return model_id


def require_mapped_parameters(
    model_id: str,
    parameters: Union[Mapping[str, Any], Iterable[str]],
    *,
    registry: Optional[ModelRegistry] = None,
) -> None:
    """Raise :class:`UnmappedParameterError` if any name is not mapped for ``model_id``.

    ``parameters`` is a mapping (its keys are checked) or an iterable of names. All unmapped
    names are reported together so one failure shows the whole problem.
    """
    reg = registry or load_registry()
    allowed = reg.model(model_id).parameters
    names = list(parameters.keys()) if isinstance(parameters, Mapping) else list(parameters)
    unmapped = sorted({str(n) for n in names} - allowed)
    if unmapped:
        raise UnmappedParameterError(
            f"parameter(s) {unmapped} are not mapped for model {model_id}; "
            f"mapped parameters: {sorted(allowed)}"
        )
