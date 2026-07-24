"""PPR78 group-contribution parameter databases and constructors.

The bundled set is the original six-group saturated-hydrocarbon
parameterization from Jaubert and Mutelet, *Fluid Phase Equilibria* 224
(2004), 285-304, doi:10.1016/j.fluid.2004.06.059. Later PPR78 extensions use
the same API through custom YAML parameter sets but must not be confused with
this 2004 fit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.database import ParameterSource, load_model_parameters
from torch_flash.exceptions import ParameterDatabaseError
from torch_flash.mixing import PPR78Mixing

DEFAULT_PPR78_GROUP_CONTRIBUTION = "group-contribution.ppr78-jaubert-mutelet-2004"


@dataclass(frozen=True)
class PPR78GroupContributionParameters:
    """Selected PPR78 group fractions and universal interaction tensors."""

    group_names: tuple[str, ...]
    group_fractions: Tensor
    group_a: Tensor
    group_b: Tensor
    reference_temperature: float
    parameter_set: str

    def __post_init__(self) -> None:
        if not self.group_names or len(set(self.group_names)) != len(self.group_names):
            raise ValueError("PPR78 group names must be non-empty and unique")
        expected = len(self.group_names)
        if self.group_fractions.ndim != 2 or self.group_fractions.shape[1] != expected:
            raise ValueError("PPR78 group fractions must match the group inventory")
        if self.group_a.shape != (expected, expected) or self.group_b.shape != self.group_a.shape:
            raise ValueError("PPR78 group interactions must match the group inventory")


def _group_names(value: object, source: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ParameterDatabaseError(f"{source} groups must be a string list")
    names = tuple(value)
    if any(not isinstance(name, str) or not name for name in names):
        raise ParameterDatabaseError(f"{source} groups must be a string list")
    if len(names) < 2 or len(set(names)) != len(names):
        raise ParameterDatabaseError(f"{source} groups must contain unique names")
    return names  # type: ignore[return-value]


def _finite(value: object, description: str) -> float:
    if not isinstance(value, int | float) or not math.isfinite(value):
        raise ParameterDatabaseError(f"{description} must be finite and numeric")
    return float(value)


def _interaction_matrices(
    value: object,
    groups: tuple[str, ...],
    source: str,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> tuple[Tensor, Tensor]:
    if not isinstance(value, Mapping):
        raise ParameterDatabaseError(f"{source} interactions must be a mapping")
    size = len(groups)
    group_index = {name: index for index, name in enumerate(groups)}
    group_a = torch.zeros((size, size), dtype=dtype, device=device)
    group_b = torch.zeros_like(group_a)
    seen: set[frozenset[str]] = set()
    for key, record in value.items():
        if not isinstance(key, str) or key.count("|") != 1:
            raise ParameterDatabaseError(
                f"{source} interaction keys must have the form 'first|second'"
            )
        first, second = key.split("|")
        if first == second or first not in group_index or second not in group_index:
            raise ParameterDatabaseError(f"{source} has invalid group interaction {key!r}")
        unordered = frozenset((first, second))
        if unordered in seen:
            raise ParameterDatabaseError(f"{source} contains duplicate interaction {key!r}")
        seen.add(unordered)
        if not isinstance(record, Mapping):
            raise ParameterDatabaseError(f"{source} interaction {key!r} must be a mapping")
        a_value = _finite(record.get("A"), f"{source} interaction {key!r} A")
        b_value = _finite(record.get("B"), f"{source} interaction {key!r} B")
        if a_value == 0.0 and b_value != 0.0:
            raise ParameterDatabaseError(
                f"{source} interaction {key!r} B must be zero when A is zero"
            )
        i = group_index[first]
        j = group_index[second]
        group_a[i, j] = group_a[j, i] = a_value
        group_b[i, j] = group_b[j, i] = b_value
    expected_pairs = size * (size - 1) // 2
    if len(seen) != expected_pairs:
        raise ParameterDatabaseError(
            f"{source} must explicitly define all {expected_pairs} unique group pairs"
        )
    return group_a, group_b


def _fraction_matrix(
    components: ComponentSet,
    groups: tuple[str, ...],
    value: object,
    source: str,
) -> Tensor:
    dtype = components.critical_temperature.dtype
    device = components.critical_temperature.device
    if isinstance(value, Tensor):
        counts = value.to(dtype=dtype, device=device)
        if counts.shape != (components.ncomponents, len(groups)):
            raise ValueError(
                "custom PPR78 group_counts tensor must have shape "
                f"({components.ncomponents}, {len(groups)})"
            )
    elif isinstance(value, Mapping):
        counts = torch.zeros(
            (components.ncomponents, len(groups)),
            dtype=dtype,
            device=device,
        )
        group_index = {name: index for index, name in enumerate(groups)}
        for component_index, component_name in enumerate(components.names):
            record = value.get(component_name)
            if not isinstance(record, Mapping):
                raise KeyError(f"{source} has no PPR78 decomposition for {component_name!r}")
            for group_name, raw_count in record.items():
                if group_name not in group_index:
                    raise ParameterDatabaseError(
                        f"{source} component {component_name!r} uses unknown group {group_name!r}"
                    )
                count = _finite(
                    raw_count,
                    f"{source} component {component_name!r} group {group_name!r}",
                )
                if count < 0.0:
                    raise ParameterDatabaseError("PPR78 group counts cannot be negative")
                counts[component_index, group_index[group_name]] = count
    else:
        raise ParameterDatabaseError(f"{source} component_groups must be a mapping")
    if not bool(torch.isfinite(counts).all()) or bool((counts < 0.0).any()):
        raise ValueError("PPR78 group counts must be finite and nonnegative")
    totals = counts.sum(dim=-1, keepdim=True)
    if bool((totals <= 0.0).any()):
        raise ValueError("each PPR78 component must contain at least one group")
    return counts / totals


def ppr78_group_contribution_parameters(
    components: ComponentSet,
    source: ParameterSource = DEFAULT_PPR78_GROUP_CONTRIBUTION,
    *,
    group_counts: Mapping[str, Mapping[str, float]] | Tensor | None = None,
) -> PPR78GroupContributionParameters:
    """Load PPR78 group parameters for ``components``.

    ``source`` may be the bundled identifier, a custom YAML path, or an
    in-memory :class:`~torch_flash.database.ModelParameterSet`. Explicit
    ``group_counts`` override only the component decompositions; the source
    still supplies the group inventory and interaction matrices.
    """
    loaded = load_model_parameters(source)
    if loaded.model_kind != "group_contribution":
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} is {loaded.model_kind!r}, not 'group_contribution'"
        )
    if loaded.model != "PPR78":
        raise ParameterDatabaseError(f"{loaded.identifier!r} model must be 'PPR78'")
    expected_units = {
        "A": "Pa",
        "B": "Pa",
        "group_fraction": "dimensionless",
        "reference_temperature": "K",
    }
    if any(loaded.units.get(key) != unit for key, unit in expected_units.items()):
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} must declare PPR78 A, B, fraction, and temperature units"
        )
    parameters = loaded.parameters
    groups = _group_names(parameters.get("groups"), loaded.identifier)
    reference_temperature = _finite(
        parameters.get("reference_temperature"),
        f"{loaded.identifier} reference_temperature",
    )
    if reference_temperature <= 0.0:
        raise ParameterDatabaseError(f"{loaded.identifier} reference_temperature must be positive")
    group_a, group_b = _interaction_matrices(
        parameters.get("interactions"),
        groups,
        loaded.identifier,
        dtype=components.critical_temperature.dtype,
        device=components.critical_temperature.device,
    )
    decompositions = parameters.get("component_groups") if group_counts is None else group_counts
    fractions = _fraction_matrix(
        components,
        groups,
        decompositions,
        loaded.identifier if group_counts is None else "<api group_counts>",
    )
    return PPR78GroupContributionParameters(
        groups,
        fractions,
        group_a,
        group_b,
        reference_temperature,
        loaded.identifier,
    )


def ppr78_mixing(
    components: ComponentSet,
    source: ParameterSource = DEFAULT_PPR78_GROUP_CONTRIBUTION,
    *,
    group_counts: Mapping[str, Mapping[str, float]] | Tensor | None = None,
    trainable: bool = False,
) -> PPR78Mixing:
    """Construct PPR78 mixing from bundled, custom-file, or API parameters."""
    parameters = ppr78_group_contribution_parameters(
        components,
        source,
        group_counts=group_counts,
    )
    return PPR78Mixing(
        parameters.group_fractions,
        parameters.group_a,
        parameters.group_b,
        reference_temperature=parameters.reference_temperature,
        trainable=trainable,
        parameter_set=parameters.parameter_set,
    )


__all__ = [
    "DEFAULT_PPR78_GROUP_CONTRIBUTION",
    "PPR78GroupContributionParameters",
    "ppr78_group_contribution_parameters",
    "ppr78_mixing",
]
