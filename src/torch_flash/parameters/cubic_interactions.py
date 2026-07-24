"""Generic cubic attraction and co-volume interaction parameter files."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.database import ParameterSource, load_model_parameters
from torch_flash.exceptions import ParameterDatabaseError


@dataclass(frozen=True)
class CubicInteractionParameters:
    """Symmetric dimensionless ``kij`` and ``lij`` matrices."""

    kij: Tensor
    lij: Tensor
    parameter_set: str

    def __post_init__(self) -> None:
        if self.kij.ndim != 2 or self.kij.shape[0] != self.kij.shape[1]:
            raise ValueError("kij must be a square matrix")
        if self.lij.shape != self.kij.shape:
            raise ValueError("lij must have the same square shape as kij")
        if not bool(torch.isfinite(self.kij).all()) or not bool(torch.isfinite(self.lij).all()):
            raise ValueError("interaction matrices must contain only finite values")
        if not torch.allclose(self.kij, self.kij.mT):
            raise ValueError("kij must be symmetric")
        if not torch.allclose(self.lij, self.lij.mT):
            raise ValueError("lij must be symmetric")
        if bool(torch.diagonal(self.kij).count_nonzero()) or bool(
            torch.diagonal(self.lij).count_nonzero()
        ):
            raise ValueError("interaction matrices must have zero diagonals")


def _component_order(value: object, source: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ParameterDatabaseError(f"{source} component_order must be a string list")
    result = tuple(value)
    if any(not isinstance(name, str) or not name for name in result):
        raise ParameterDatabaseError(f"{source} component_order must be a string list")
    if len(set(result)) != len(result):
        raise ParameterDatabaseError(f"{source} component_order contains duplicates")
    return result  # type: ignore[return-value]


def _finite_default(defaults: Mapping[str, object], key: str, source: str) -> float:
    value = defaults.get(key, 0.0)
    if not isinstance(value, int | float) or not math.isfinite(value):
        raise ParameterDatabaseError(f"{source} default {key} must be finite and numeric")
    return float(value)


def cubic_interaction_parameters(
    components: ComponentSet,
    source: ParameterSource,
) -> CubicInteractionParameters:
    """Load general van der Waals one-fluid ``kij`` and ``lij`` matrices.

    The YAML payload declares ``component_order``, optional numeric
    ``defaults``, and unordered ``first|second`` pair records. Missing pair
    fields use the declared defaults, which are zero when omitted.
    """
    loaded = load_model_parameters(source)
    if loaded.model_kind != "binary_interaction":
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} is {loaded.model_kind!r}, not 'binary_interaction'"
        )
    if loaded.model != "cubic-vdw-one-fluid":
        raise ParameterDatabaseError(f"{loaded.identifier!r} model must be 'cubic-vdw-one-fluid'")
    if loaded.units.get("kij") != "dimensionless" or loaded.units.get("lij") != "dimensionless":
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} must declare dimensionless kij and lij units"
        )
    parameters = loaded.parameters
    order = _component_order(parameters.get("component_order"), loaded.identifier)
    unsupported = sorted(set(components.names) - set(order))
    if unsupported:
        raise KeyError(
            f"{loaded.identifier} has no interaction parameters for: {', '.join(unsupported)}"
        )
    defaults = parameters.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise ParameterDatabaseError(f"{loaded.identifier} defaults must be a mapping")
    default_kij = _finite_default(defaults, "kij", loaded.identifier)
    default_lij = _finite_default(defaults, "lij", loaded.identifier)
    shape = (components.ncomponents, components.ncomponents)
    kij = torch.full(
        shape,
        default_kij,
        dtype=components.critical_temperature.dtype,
        device=components.critical_temperature.device,
    )
    lij = torch.full_like(kij, default_lij)
    kij.fill_diagonal_(0.0)
    lij.fill_diagonal_(0.0)
    pairs = parameters.get("pairs", {})
    if not isinstance(pairs, Mapping):
        raise ParameterDatabaseError(f"{loaded.identifier} pairs must be a mapping")
    seen: set[frozenset[str]] = set()
    selected = {name: index for index, name in enumerate(components.names)}
    allowed = set(order)
    for key, record in pairs.items():
        if not isinstance(key, str) or key.count("|") != 1:
            raise ParameterDatabaseError(
                f"{loaded.identifier} pair keys must have the form 'first|second'"
            )
        first, second = key.split("|")
        if first == second:
            raise ParameterDatabaseError(f"{loaded.identifier} cannot contain self-pairs")
        if first not in allowed or second not in allowed:
            raise ParameterDatabaseError(
                f"{loaded.identifier} pair {key!r} names a component outside component_order"
            )
        unordered = frozenset((first, second))
        if unordered in seen:
            raise ParameterDatabaseError(
                f"{loaded.identifier} contains duplicate pair {first}|{second}"
            )
        seen.add(unordered)
        if not isinstance(record, Mapping):
            raise ParameterDatabaseError(f"{loaded.identifier} pair {key!r} must be a mapping")
        values = {}
        for parameter, default in (("kij", default_kij), ("lij", default_lij)):
            value = record.get(parameter, default)
            if not isinstance(value, int | float) or not math.isfinite(value):
                raise ParameterDatabaseError(
                    f"{loaded.identifier} pair {key!r} {parameter} must be finite and numeric"
                )
            values[parameter] = float(value)
        if first not in selected or second not in selected:
            continue
        first_index = selected[first]
        second_index = selected[second]
        kij[first_index, second_index] = kij[second_index, first_index] = values["kij"]
        lij[first_index, second_index] = lij[second_index, first_index] = values["lij"]
    return CubicInteractionParameters(kij, lij, loaded.identifier)


__all__ = ["CubicInteractionParameters", "cubic_interaction_parameters"]
