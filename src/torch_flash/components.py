"""Canonical component names and shared SI property databases."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache, lru_cache
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import torch
import yaml  # type: ignore[import-untyped]
from torch import Tensor

from torch_flash.config import resolve_tensor_options
from torch_flash.exceptions import ParameterDatabaseError

_COMPONENT_FORMAT = "torch-flash-component-database"
_SCHEMA_VERSION = 1
_EXPECTED_UNITS = {
    "critical_temperature": "K",
    "critical_pressure": "Pa",
    "acentric_factor": "dimensionless",
    "molar_mass": "kg mol^-1",
    "critical_volume": "m^3 mol^-1",
    "critical_density": "mol m^-3",
}


def _normalize_name(name: str) -> str:
    return name.lower().strip().replace("-", "_").replace(" ", "_")


def _optional_float(record: Mapping[str, Any], key: str, source: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ParameterDatabaseError(f"{source} property {key!r} must be numeric or null")
    return float(value)


def _required_positive(record: Mapping[str, Any], key: str, source: str) -> float:
    value = _optional_float(record, key, source)
    if value is None or value <= 0.0:
        raise ParameterDatabaseError(f"{source} property {key!r} must be positive")
    return value


@dataclass(frozen=True)
class Component:
    """Shared pure-component properties in SI units.

    ``acentric_factor`` may be unavailable for components whose bundled use is
    restricted to a model-specific multiparameter equation. Such a record can be
    resolved by name but cannot be used to construct a cubic EoS until the user
    supplies a complete custom component database or :class:`ComponentSet`.

    Attributes
    ----------
    name:
        Canonical lowercase snake-case component identifier.
    critical_temperature:
        Critical temperature in K.
    critical_pressure:
        Critical pressure in Pa.
    acentric_factor:
        Dimensionless acentric factor, or ``None`` when unavailable.
    molar_mass:
        Molar mass in kg/mol.
    critical_volume:
        Critical molar volume in m3/mol, or ``None`` when unavailable.
    aliases:
        Accepted alternative names, normalized only at API lookup boundaries.
    critical_density:
        Critical molar density in mol/m3, or ``None`` when unavailable.
    """

    name: str
    critical_temperature: float
    critical_pressure: float
    acentric_factor: float | None
    molar_mass: float
    critical_volume: float | None = None
    aliases: tuple[str, ...] = ()
    critical_density: float | None = None


@dataclass(frozen=True)
class ComponentDatabase:
    """Validated immutable canonical component database.

    Attributes
    ----------
    identifier:
        Stable database identifier.
    revision:
        Database revision string.
    components:
        Pure-component records in database order.
    units:
        Read-only mapping of property names to declared SI units.
    references:
        Read-only source records applying to the database.
    source:
        Bundled resource label, custom YAML path, or ``"<api>"``.

    Raises
    ------
    ParameterDatabaseError
        If identifiers, units, records, aliases, or required positive
        properties are invalid.
    """

    identifier: str
    revision: str
    components: tuple[Component, ...]
    units: Mapping[str, str]
    references: tuple[Mapping[str, str], ...] = ()
    source: str = "<api>"

    def __post_init__(self) -> None:
        if not self.identifier or not self.revision:
            raise ParameterDatabaseError("component database id and revision must be non-empty")
        if not self.components:
            raise ParameterDatabaseError("component database must contain at least one component")
        for property_name, expected in _EXPECTED_UNITS.items():
            if self.units.get(property_name) != expected:
                raise ParameterDatabaseError(
                    f"component database unit for {property_name!r} must be {expected!r}"
                )
        identifiers: set[str] = set()
        for record in self.components:
            if _normalize_name(record.name) != record.name:
                raise ParameterDatabaseError(
                    f"component name {record.name!r} must be canonical lowercase snake case"
                )
            current = (record.name, *(_normalize_name(alias) for alias in record.aliases))
            duplicate = next((name for name in current if name in identifiers), None)
            if duplicate is not None:
                raise ParameterDatabaseError(
                    f"component database contains duplicate name or alias {duplicate!r}"
                )
            identifiers.update(current)
            positive = (
                record.critical_temperature,
                record.critical_pressure,
                record.molar_mass,
            )
            if any(not math.isfinite(value) or value <= 0.0 for value in positive):
                raise ParameterDatabaseError(
                    f"component {record.name!r} requires finite positive Tcrit, Pcrit, and mass"
                )
            optional = (
                record.critical_volume,
                record.critical_density,
            )
            if any(
                value is not None and (not math.isfinite(value) or value <= 0.0)
                for value in optional
            ):
                raise ParameterDatabaseError(
                    f"component {record.name!r} optional volume/density must be positive"
                )
            if record.acentric_factor is not None and not math.isfinite(record.acentric_factor):
                raise ParameterDatabaseError(
                    f"component {record.name!r} acentric factor must be finite or null"
                )
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))
        object.__setattr__(
            self,
            "references",
            tuple(MappingProxyType(dict(reference)) for reference in self.references),
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Return canonical component names in database order.

        Returns
        -------
        tuple
            Immutable sequence of canonical identifiers.
        """
        return tuple(record.name for record in self.components)

    def lookup(self, name: str) -> Component:
        """Resolve a canonical name or alias.

        Parameters
        ----------
        name:
            Name or alias; matching is case-insensitive and treats spaces and
            hyphens as underscores.

        Returns
        -------
        Component
            Matching immutable component record.

        Raises
        ------
        KeyError
            If no component or alias matches.
        """
        normalized = _normalize_name(name)
        aliases: dict[str, Component] = {}
        for record in self.components:
            aliases[_normalize_name(record.name)] = record
            for alias in record.aliases:
                aliases[_normalize_name(alias)] = record
        try:
            return aliases[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(self.names))
            raise KeyError(f"unknown component {name!r}; available: {available}") from exc


@dataclass(frozen=True)
class ComponentSet:
    """Vectorized component constants used by native PyTorch models.

    All tensor fields have shape ``(ncomponents,)`` and share a dtype/device.
    Component order defines the final axis of every associated composition
    tensor.

    Attributes
    ----------
    names:
        Canonical component names in tensor order.
    critical_temperature:
        Critical temperatures in K.
    critical_pressure:
        Critical pressures in Pa.
    acentric_factor:
        Dimensionless acentric factors.
    molar_mass:
        Molar masses in kg/mol.
    critical_volume:
        Critical molar volumes in m3/mol, or ``None``.
    critical_density:
        Critical molar densities in mol/m3, or ``None``.
    """

    names: tuple[str, ...]
    critical_temperature: Tensor
    critical_pressure: Tensor
    acentric_factor: Tensor
    molar_mass: Tensor
    critical_volume: Tensor | None = None
    critical_density: Tensor | None = None

    @property
    def ncomponents(self) -> int:
        """Return the component-axis length.

        Returns
        -------
        int
            Number of entries in :attr:`names`.
        """
        return len(self.names)

    def to(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> ComponentSet:
        """Return a component set with all tensors converted together.

        Parameters
        ----------
        dtype:
            Target floating dtype. ``None`` preserves the current dtype.
        device:
            Target PyTorch device. ``None`` preserves the current device.

        Returns
        -------
        ComponentSet
            New immutable record; the original is unchanged.
        """
        return ComponentSet(
            self.names,
            self.critical_temperature.to(dtype=dtype, device=device),
            self.critical_pressure.to(dtype=dtype, device=device),
            self.acentric_factor.to(dtype=dtype, device=device),
            self.molar_mass.to(dtype=dtype, device=device),
            (
                None
                if self.critical_volume is None
                else self.critical_volume.to(dtype=dtype, device=device)
            ),
            (
                None
                if self.critical_density is None
                else self.critical_density.to(dtype=dtype, device=device)
            ),
        )


def _parse_component_database(text: str, source: str) -> ComponentDatabase:
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ParameterDatabaseError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise ParameterDatabaseError(f"{source} must contain a YAML mapping")
    if document.get("format") != _COMPONENT_FORMAT:
        raise ParameterDatabaseError(f"{source} is not a torch-flash component database")
    if document.get("schema_version") != _SCHEMA_VERSION:
        raise ParameterDatabaseError(
            f"{source} has unsupported schema_version {document.get('schema_version')!r}"
        )
    if document.get("unit_system") != "SI":
        raise ParameterDatabaseError(f"{source} must declare unit_system: SI")
    units = document.get("units")
    if not isinstance(units, dict):
        raise ParameterDatabaseError(f"{source} requires a units mapping")
    for property_name, expected in _EXPECTED_UNITS.items():
        if units.get(property_name) != expected:
            raise ParameterDatabaseError(
                f"{source} unit for {property_name!r} must be {expected!r}"
            )
    raw_components = document.get("components")
    if not isinstance(raw_components, dict) or not raw_components:
        raise ParameterDatabaseError(f"{source} requires a non-empty components mapping")

    components: list[Component] = []
    names_and_aliases: set[str] = set()
    for raw_name, raw_record in raw_components.items():
        if not isinstance(raw_name, str) or not isinstance(raw_record, dict):
            raise ParameterDatabaseError(f"{source} contains an invalid component record")
        name = _normalize_name(raw_name)
        if name != raw_name:
            raise ParameterDatabaseError(
                f"{source} component name {raw_name!r} is not canonical; use {name!r}"
            )
        aliases = raw_record.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
            raise ParameterDatabaseError(f"{source} aliases for {name!r} must be a string list")
        normalized_aliases = tuple(_normalize_name(alias) for alias in aliases)
        identifiers = (name, *normalized_aliases)
        duplicate = next((item for item in identifiers if item in names_and_aliases), None)
        if duplicate is not None:
            raise ParameterDatabaseError(f"{source} contains duplicate name or alias {duplicate!r}")
        names_and_aliases.update(identifiers)
        component_source = f"{source}:{name}"
        acentric_factor = _optional_float(raw_record, "acentric_factor", component_source)
        components.append(
            Component(
                name=name,
                critical_temperature=_required_positive(
                    raw_record, "critical_temperature", component_source
                ),
                critical_pressure=_required_positive(
                    raw_record, "critical_pressure", component_source
                ),
                acentric_factor=acentric_factor,
                molar_mass=_required_positive(raw_record, "molar_mass", component_source),
                critical_volume=_optional_float(raw_record, "critical_volume", component_source),
                aliases=normalized_aliases,
                critical_density=_optional_float(raw_record, "critical_density", component_source),
            )
        )

    references = document.get("references", [])
    if not isinstance(references, list) or any(
        not isinstance(reference, dict) for reference in references
    ):
        raise ParameterDatabaseError(f"{source} references must be a list of mappings")
    identifier = document.get("id")
    revision = document.get("revision")
    if not isinstance(identifier, str) or not isinstance(revision, str):
        raise ParameterDatabaseError(f"{source} requires string id and revision fields")
    return ComponentDatabase(
        identifier,
        revision,
        tuple(components),
        cast(dict[str, str], units),
        tuple(cast(list[Mapping[str, str]], references)),
        source,
    )


@lru_cache(maxsize=1)
def _default_component_database() -> ComponentDatabase:
    resource = files("torch_flash").joinpath("data", "components", "default.yaml")
    return _parse_component_database(
        resource.read_text(encoding="utf-8"),
        "bundled:data/components/default.yaml",
    )


@cache
def _component_database_from_path(path_string: str) -> ComponentDatabase:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"component database file does not exist: {path}")
    return _parse_component_database(path.read_text(encoding="utf-8"), str(path))


def load_component_database(
    source: str | Path | ComponentDatabase | None = None,
) -> ComponentDatabase:
    """Load and validate a bundled, custom, or explicit component database.

    Parameters
    ----------
    source:
        ``None``, ``"default"``, or ``"components.default"`` selects the
        bundled database. A path reads a custom torch-flash component YAML
        document. An existing :class:`ComponentDatabase` is returned unchanged.

    Returns
    -------
    ComponentDatabase
        Immutable validated database. Parsed files are cached by resolved path.

    Raises
    ------
    FileNotFoundError
        If a custom path does not exist.
    ParameterDatabaseError
        If the YAML schema, unit declarations, or component records are invalid.
    """
    if source is None or source in ("default", "components.default"):
        return _default_component_database()
    if isinstance(source, ComponentDatabase):
        return source
    return _component_database_from_path(str(Path(source).expanduser().absolute()))


def clear_component_caches() -> None:
    """Clear cached bundled and custom component YAML documents.

    Use this after intentionally modifying a custom database in a long-running
    process. Existing :class:`ComponentSet` and model instances are unaffected.

    Returns
    -------
    None
        The next database load reparses its source.
    """
    _default_component_database.cache_clear()
    _component_database_from_path.cache_clear()


DEFAULT_COMPONENT_DATABASE = load_component_database()
"""Validated bundled component database used by default constructors."""

_COMPONENTS = DEFAULT_COMPONENT_DATABASE.components
COMPONENTS: dict[str, Component] = {}
"""Canonical-name and alias lookup mapping for bundled component records."""

for _component in _COMPONENTS:
    COMPONENTS[_component.name] = _component
    for _alias in _component.aliases:
        COMPONENTS[_alias] = _component


def canonical_component_name(
    name: str,
    *,
    database: str | Path | ComponentDatabase | None = None,
    strict: bool = True,
) -> str:
    """Resolve a component name or alias to the canonical identifier.

    Parameters
    ----------
    name:
        Canonical name or declared alias.
    database:
        Bundled/custom database selection accepted by
        :func:`load_component_database`.
    strict:
        If ``True``, unknown names raise. If ``False``, an unknown name is
        normalized to lowercase snake case without asserting database presence.

    Returns
    -------
    str
        Canonical database name or, in non-strict mode, normalized input.

    Raises
    ------
    KeyError
        If ``name`` is unknown and ``strict=True``.
    """
    selected = load_component_database(database)
    try:
        return selected.lookup(name).name
    except KeyError:
        if strict:
            raise
        return _normalize_name(name)


def component(
    name: str,
    *,
    database: str | Path | ComponentDatabase | None = None,
) -> Component:
    """Look up one pure-component record by canonical name or alias.

    Parameters
    ----------
    name:
        Canonical name or alias to resolve.
    database:
        Bundled/custom database selection.

    Returns
    -------
    Component
        Immutable record from the selected database.

    Raises
    ------
    KeyError
        If the component is absent from the selected database.
    """
    return load_component_database(database).lookup(name)


def component_set(
    names: Iterable[str],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    database: str | Path | ComponentDatabase | None = None,
) -> ComponentSet:
    """Build vectorized cubic-EoS constants from a component database.

    Omitted tensor options follow the process-wide :mod:`torch_flash.config`
    policy. Explicit dtype or device arguments override that policy.

    Parameters
    ----------
    names:
        Nonempty iterable of canonical names or aliases. Order is preserved.
    dtype, device:
        Tensor options. Omitted values use the configured runtime policy.
    database:
        Bundled/custom database selection.

    Returns
    -------
    ComponentSet
        Vectorized SI constants in the requested order.

    Raises
    ------
    ValueError
        If no components are requested.
    KeyError
        If a name is unknown.
    ParameterDatabaseError
        If a requested component lacks the acentric factor required by cubic
        EoS construction.
    """
    dtype, device = resolve_tensor_options(dtype, device)
    items = tuple(component(name, database=database) for name in names)
    if not items:
        raise ValueError("at least one component is required")
    unavailable = tuple(item.name for item in items if item.acentric_factor is None)
    if unavailable:
        joined = ", ".join(unavailable)
        raise ParameterDatabaseError(
            f"cubic-EoS acentric factors are unavailable for: {joined}; "
            "supply a custom component database or ComponentSet"
        )

    def tensor(values: list[float]) -> Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    critical_volumes = [
        float("nan") if item.critical_volume is None else item.critical_volume for item in items
    ]
    critical_densities = [
        float("nan") if item.critical_density is None else item.critical_density for item in items
    ]
    return ComponentSet(
        tuple(item.name for item in items),
        tensor([item.critical_temperature for item in items]),
        tensor([item.critical_pressure for item in items]),
        tensor([cast(float, item.acentric_factor) for item in items]),
        tensor([item.molar_mass for item in items]),
        tensor(critical_volumes),
        tensor(critical_densities),
    )


__all__ = [
    "COMPONENTS",
    "DEFAULT_COMPONENT_DATABASE",
    "Component",
    "ComponentDatabase",
    "ComponentSet",
    "canonical_component_name",
    "clear_component_caches",
    "component",
    "component_set",
    "load_component_database",
]
