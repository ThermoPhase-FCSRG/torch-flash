"""Published petroleum binary-interaction parameter sets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.database import ParameterSource, load_model_parameters
from torch_flash.exceptions import ParameterDatabaseError

PetroleumEOS = Literal["PR", "SRK"]


def _string_tuple(value: object, key: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ParameterDatabaseError(f"binary-interaction {key!r} must be a string list")
    result = tuple(value)
    if any(not isinstance(item, str) for item in result):
        raise ParameterDatabaseError(f"binary-interaction {key!r} must be a string list")
    return result


def binary_interaction(
    components: ComponentSet,
    parameter_set: ParameterSource,
    eos: PetroleumEOS = "PR",
) -> Tensor:
    """Return a symmetric ``kij`` matrix from YAML model parameters."""
    if eos not in ("PR", "SRK"):
        raise ValueError("eos must be 'PR' or 'SRK'")
    loaded = load_model_parameters(parameter_set)
    if loaded.model_kind != "binary_interaction":
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} is {loaded.model_kind!r}, not 'binary_interaction'"
        )
    parameters = loaded.parameters
    hydrocarbons = _string_tuple(parameters.get("hydrocarbons"), "hydrocarbons")
    nonhydrocarbons = _string_tuple(parameters.get("nonhydrocarbons"), "nonhydrocarbons")
    supported = frozenset((*hydrocarbons, *nonhydrocarbons))
    unsupported = sorted(set(components.names) - supported)
    if unsupported:
        names = ", ".join(unsupported)
        raise KeyError(f"{loaded.identifier} has no parameters for: {names}")
    aggregate_from = parameters.get("aggregate_from")
    if not isinstance(aggregate_from, str) or aggregate_from not in hydrocarbons:
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} requires a valid aggregate_from component"
        )
    aggregate_index = hydrocarbons.index(aggregate_from)
    eos_tables = parameters.get("eos")
    if not isinstance(eos_tables, Mapping):
        raise ParameterDatabaseError(f"{loaded.identifier!r} requires an eos mapping")
    table = eos_tables.get(eos)
    if not isinstance(table, Mapping):
        raise ParameterDatabaseError(f"{loaded.identifier!r} has no {eos} table")
    hydrocarbon_rows = table.get("hydrocarbon_rows")
    pair_values = table.get("nonhydrocarbon_pairs")
    if not isinstance(hydrocarbon_rows, Mapping) or not isinstance(pair_values, Mapping):
        raise ParameterDatabaseError(f"{loaded.identifier!r} has incomplete {eos} tables")

    kij = torch.zeros(
        (components.ncomponents, components.ncomponents),
        dtype=components.critical_temperature.dtype,
        device=components.critical_temperature.device,
    )
    for row, first in enumerate(components.names):
        for column in range(row):
            second = components.names[column]
            if first in hydrocarbons and second in hydrocarbons:
                value = 0.0
            elif first in nonhydrocarbons and second in nonhydrocarbons:
                key = f"{first}|{second}"
                reverse_key = f"{second}|{first}"
                raw_value = pair_values.get(key, pair_values.get(reverse_key))
                if not isinstance(raw_value, int | float):
                    raise ParameterDatabaseError(
                        f"{loaded.identifier!r} lacks pair {first}|{second}"
                    )
                value = float(raw_value)
            else:
                nonhydrocarbon = first if first in nonhydrocarbons else second
                hydrocarbon = second if first in nonhydrocarbons else first
                values = hydrocarbon_rows.get(nonhydrocarbon)
                if not isinstance(values, Sequence) or isinstance(values, str):
                    raise ParameterDatabaseError(
                        f"{loaded.identifier!r} lacks row {nonhydrocarbon!r}"
                    )
                hydrocarbon_index = min(hydrocarbons.index(hydrocarbon), aggregate_index)
                raw_value = values[hydrocarbon_index]
                if not isinstance(raw_value, int | float):
                    raise ParameterDatabaseError("binary-interaction values must be numeric")
                value = float(raw_value)
            kij[row, column] = value
            kij[column, row] = value
    return kij


def whitson_binary_interaction(
    components: ComponentSet,
    eos: PetroleumEOS = "PR",
) -> Tensor:
    """Return Whitson and Brule (2000), Table A-3, ``kij`` values.

    The source's C7+ values are applied from n-heptane through n-decane. The
    printed H2S/C7+ value is not silently extrapolated by carbon number.
    """
    return binary_interaction(components, "binary-interaction.whitson-2000", eos)


def pedersen_binary_interaction(
    components: ComponentSet,
    eos: PetroleumEOS = "PR",
) -> Tensor:
    """Return Pedersen et al. (2024), Table 4.2, ``kij`` values.

    This is a distinct parameterization from Whitson's Table A-3. The source's
    aggregate C7+ entries are used for n-heptane through n-decane.
    """
    return binary_interaction(components, "binary-interaction.pedersen-2024", eos)


__all__ = [
    "PetroleumEOS",
    "binary_interaction",
    "pedersen_binary_interaction",
    "whitson_binary_interaction",
]
