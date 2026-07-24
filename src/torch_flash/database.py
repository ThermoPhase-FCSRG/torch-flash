"""Versioned YAML parameter-database loading.

Bundled model parameter sets are immutable scientific inputs. YAML documents
are parsed and validated once per process, while model constructors copy their
values into independent PyTorch tensors with the requested dtype, device, and
trainability.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cache, lru_cache
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias, cast

import yaml  # type: ignore[import-untyped]

from torch_flash.exceptions import ParameterDatabaseError

ParameterDocument: TypeAlias = Mapping[str, Any]

_MODEL_FORMAT = "torch-flash-model-parameters"
_INDEX_FORMAT = "torch-flash-parameter-index"
_SCHEMA_VERSION = 1
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MODEL_KINDS = frozenset(
    {
        "activity",
        "binary_interaction",
        "characterization",
        "cpa",
        "cubic",
        "group_contribution",
        "multifluid",
        "standard_state",
        "volume_translation",
    }
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


def _parse_yaml(text: str, source: str) -> dict[str, Any]:
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ParameterDatabaseError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise ParameterDatabaseError(f"{source} must contain a YAML mapping at its root")
    return cast(dict[str, Any], document)


def _require_string(document: Mapping[str, Any], key: str, source: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ParameterDatabaseError(f"{source} requires a non-empty {key!r} string")
    return value


def _validate_header(document: Mapping[str, Any], expected_format: str, source: str) -> None:
    if document.get("format") != expected_format:
        raise ParameterDatabaseError(
            f"{source} has format {document.get('format')!r}; expected {expected_format!r}"
        )
    if document.get("schema_version") != _SCHEMA_VERSION:
        raise ParameterDatabaseError(
            f"{source} has unsupported schema_version {document.get('schema_version')!r}"
        )


@dataclass(frozen=True)
class ModelParameterSet:
    """Validated model parameter set independent of tensor dtype and device.

    Parameters can be supplied directly through this dataclass, read from a
    custom YAML path, or selected from the bundled database by identifier.
    The stored mappings are recursively read-only. Use :meth:`as_dict` when a
    mutable deep copy is needed.
    """

    identifier: str
    model_kind: str
    model: str
    version: str
    parameters: ParameterDocument
    units: ParameterDocument = field(default_factory=dict)
    references: tuple[ParameterDocument, ...] = ()
    description: str = ""
    source: str = "<api>"

    def __post_init__(self) -> None:
        for field_name in ("identifier", "model_kind", "model", "version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ParameterDatabaseError(f"{field_name} must be a non-empty string")
        if _IDENTIFIER_PATTERN.fullmatch(self.identifier) is None:
            raise ParameterDatabaseError(
                "identifier must contain only lowercase letters, digits, '.', '_', or '-'"
            )
        if self.model_kind not in _MODEL_KINDS:
            raise ParameterDatabaseError(
                f"model_kind must be one of: {', '.join(sorted(_MODEL_KINDS))}"
            )
        if not isinstance(self.parameters, Mapping):
            raise ParameterDatabaseError("parameters must be a mapping")
        if not isinstance(self.units, Mapping):
            raise ParameterDatabaseError("units must be a mapping")
        if any(
            not isinstance(key, str) or not isinstance(value, str) or not key or not value
            for key, value in self.units.items()
        ):
            raise ParameterDatabaseError("unit names and values must be non-empty strings")
        if not isinstance(self.references, Sequence) or isinstance(self.references, str):
            raise ParameterDatabaseError("references must be a sequence of mappings")
        if any(not isinstance(reference, Mapping) for reference in self.references):
            raise ParameterDatabaseError("each reference must be a mapping")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for reference in self.references
            for key, value in reference.items()
        ):
            raise ParameterDatabaseError("reference names and values must be strings")
        if not isinstance(self.description, str):
            raise ParameterDatabaseError("description must be a string")
        object.__setattr__(self, "parameters", _freeze(self.parameters))
        object.__setattr__(self, "units", _freeze(self.units))
        object.__setattr__(
            self,
            "references",
            tuple(_freeze(reference) for reference in self.references),
        )

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, Any],
        *,
        source: str = "<api>",
    ) -> ModelParameterSet:
        """Validate a model-parameter YAML document represented as a mapping."""
        _validate_header(document, _MODEL_FORMAT, source)
        parameters = document.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ParameterDatabaseError(f"{source} requires a 'parameters' mapping")
        units = document.get("units", {})
        references = document.get("references", [])
        if not isinstance(units, Mapping):
            raise ParameterDatabaseError(f"{source} requires 'units' to be a mapping")
        if not isinstance(references, list):
            raise ParameterDatabaseError(f"{source} requires 'references' to be a list")
        description = document.get("description", "")
        if not isinstance(description, str):
            raise ParameterDatabaseError(f"{source} requires 'description' to be a string")
        return cls(
            identifier=_require_string(document, "id", source),
            model_kind=_require_string(document, "model_kind", source),
            model=_require_string(document, "model", source),
            version=_require_string(document, "version", source),
            parameters=parameters,
            units=units,
            references=tuple(cast(list[ParameterDocument], references)),
            description=description,
            source=source,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable deep copy of the numerical parameter payload."""
        return cast(dict[str, Any], _thaw(self.parameters))


ParameterSource: TypeAlias = str | Path | ModelParameterSet


def _model_root() -> Any:
    return files("torch_flash").joinpath("data", "models")


@lru_cache(maxsize=1)
def _index() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    resource = _model_root().joinpath("index.yaml")
    source = "bundled:data/models/index.yaml"
    document = _parse_yaml(resource.read_text(encoding="utf-8"), source)
    _validate_header(document, _INDEX_FORMAT, source)
    entries = document.get("parameter_sets")
    if not isinstance(entries, dict) or not entries:
        raise ParameterDatabaseError(f"{source} requires a non-empty 'parameter_sets' mapping")
    normalized_entries: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    for identifier, raw_entry in entries.items():
        if not isinstance(identifier, str) or not isinstance(raw_entry, dict):
            raise ParameterDatabaseError(f"{source} contains an invalid parameter-set entry")
        path = raw_entry.get("path")
        if not isinstance(path, str) or not path.endswith((".yaml", ".yml")):
            raise ParameterDatabaseError(f"{source} entry {identifier!r} requires a YAML path")
        normalized_entries[identifier] = raw_entry
        for alias in raw_entry.get("aliases", []):
            if not isinstance(alias, str):
                raise ParameterDatabaseError(f"{source} aliases must be strings")
            normalized = alias.strip().lower()
            if normalized in aliases and aliases[normalized] != identifier:
                raise ParameterDatabaseError(f"duplicate bundled parameter alias {alias!r}")
            aliases[normalized] = identifier
        aliases[identifier.lower()] = identifier
    return normalized_entries, aliases


@cache
def _load_builtin(identifier: str) -> ModelParameterSet:
    entries, _ = _index()
    resource_path = entries[identifier]["path"]
    resource = _model_root().joinpath(*resource_path.split("/"))
    source = f"bundled:data/models/{resource_path}"
    parameter_set = ModelParameterSet.from_document(
        _parse_yaml(resource.read_text(encoding="utf-8"), source),
        source=source,
    )
    if parameter_set.identifier != identifier:
        raise ParameterDatabaseError(
            f"{source} declares id {parameter_set.identifier!r}, expected {identifier!r}"
        )
    return parameter_set


@cache
def _load_path(path_string: str) -> ModelParameterSet:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"model parameter file does not exist: {path}")
    return ModelParameterSet.from_document(
        _parse_yaml(path.read_text(encoding="utf-8"), str(path)),
        source=str(path),
    )


def load_model_parameters(source: ParameterSource) -> ModelParameterSet:
    """Load a bundled identifier, custom YAML path, or explicit parameter set.

    Parsed bundled and custom files are cached by identifier or resolved path.
    Call :func:`clear_parameter_caches` after intentionally modifying a custom
    file in a long-running process.
    """
    if isinstance(source, ModelParameterSet):
        return source
    if isinstance(source, Path):
        return _load_path(str(source.expanduser().absolute()))
    _, aliases = _index()
    normalized = source.strip().lower()
    if normalized in aliases:
        return _load_builtin(aliases[normalized])
    candidate = Path(source).expanduser()
    if candidate.suffix.lower() in (".yaml", ".yml"):
        return _load_path(str(candidate.absolute()))
    available = ", ".join(available_parameter_sets())
    raise KeyError(f"unknown model parameter set {source!r}; available: {available}")


def available_parameter_sets(*, model_kind: str | None = None) -> tuple[str, ...]:
    """Return bundled parameter-set identifiers, optionally filtered by kind."""
    entries, _ = _index()
    identifiers = tuple(sorted(entries))
    if model_kind is None:
        return identifiers
    return tuple(
        identifier
        for identifier in identifiers
        if _load_builtin(identifier).model_kind == model_kind
    )


def clear_parameter_caches() -> None:
    """Clear parsed YAML caches, primarily for custom-file development."""
    _load_builtin.cache_clear()
    _load_path.cache_clear()
    _index.cache_clear()


__all__ = [
    "ModelParameterSet",
    "ParameterDocument",
    "ParameterSource",
    "available_parameter_sets",
    "clear_parameter_caches",
    "load_model_parameters",
]
