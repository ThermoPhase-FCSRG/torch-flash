"""PPR78 and E-PPR78 group-contribution databases and constructors.

The default bundled set is the original six-group saturated-hydrocarbon
parameterization from Jaubert and Mutelet, *Fluid Phase Equilibria* 224
(2004), 285-304, doi:10.1016/j.fluid.2004.06.059. The separate bundled
H2/N2/H2O submatrix combines the hydrogen extension of Qian et al.,
doi:10.1016/j.supflu.2012.12.014, with the water extension of Qian et al.,
doi:10.1021/ie402541h. Parameter-set identity remains explicit so these later
coefficients cannot be confused with the 2004 fit.

The global E-PPR78 set is the 40-group parameterization of Jaubert et al.,
*Fluid Phase Equilibria* 560 (2022), 113456,
doi:10.1016/j.fluid.2022.113456. Its explicitly unavailable group pairs are
kept distinct from fitted zero-valued interactions.
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
PPR78_HYDROGEN_WATER_GROUP_CONTRIBUTION = "group-contribution.ppr78-qian-2013-hydrogen-water"
DEFAULT_EPPR78_GROUP_CONTRIBUTION = "group-contribution.eppr78-privat-2022-40-group"
# The global 2022 E-PPR78 inventory includes the CCS groups introduced and
# assessed by Xu et al. (2017). This use-case alias intentionally resolves to
# the newer, openly licensed global parameter revision.
EPPR78_CCS_GROUP_CONTRIBUTION = DEFAULT_EPPR78_GROUP_CONTRIBUTION


@dataclass(frozen=True)
class PPR78GroupContributionParameters:
    """Selected PPR78 or E-PPR78 decompositions and interactions.

    Attributes
    ----------
    group_names
        Ordered group identifiers.
    group_fractions
        Normalized component group fractions with shape
        ``(ncomponents, ngroups)``.
    group_a, group_b
        Symmetric universal interaction matrices in Pa with shape
        ``(ngroups, ngroups)``.
    reference_temperature
        Positive interaction reference temperature in K.
    parameter_set
        Versioned parameter-set identifier.
    """

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
    unavailable_value: object,
    groups: tuple[str, ...],
    source: str,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> tuple[Tensor, Tensor, set[frozenset[str]]]:
    if not isinstance(value, Mapping):
        raise ParameterDatabaseError(f"{source} interactions must be a mapping")
    size = len(groups)
    group_index = {name: index for index, name in enumerate(groups)}
    group_a = torch.zeros((size, size), dtype=dtype, device=device)
    group_b = torch.zeros_like(group_a)
    seen: set[frozenset[str]] = set()
    for key, record in value.items():
        first, second, unordered = _interaction_pair(key, group_index, source)
        if unordered in seen:
            raise ParameterDatabaseError(f"{source} contains duplicate interaction {key!r}")
        seen.add(unordered)
        if not isinstance(record, Mapping):
            raise ParameterDatabaseError(f"{source} interaction {key!r} must be a mapping")
        a_value = _finite(record.get("A"), f"{source} interaction {key!r} A")
        b_value = _finite(record.get("B"), f"{source} interaction {key!r} B")
        if a_value == 0.0 and b_value != 0.0:
            raise ParameterDatabaseError(f"{source} interaction {key!r} has undefined B/A")
        i = group_index[first]
        j = group_index[second]
        group_a[i, j] = group_a[j, i] = a_value
        group_b[i, j] = group_b[j, i] = b_value
    unavailable = _unavailable_pairs(
        unavailable_value,
        group_index,
        source,
    )
    overlap = seen & unavailable
    if overlap:
        pair = sorted(overlap.pop())
        raise ParameterDatabaseError(
            f"{source} marks interaction {pair[0]!r}|{pair[1]!r} as both available and unavailable"
        )
    expected_pairs = size * (size - 1) // 2
    if len(seen) + len(unavailable) != expected_pairs:
        raise ParameterDatabaseError(
            f"{source} must explicitly account for all {expected_pairs} unique group pairs "
            "as interactions or unavailable_interactions"
        )
    return group_a, group_b, unavailable


def _interaction_pair(
    key: object,
    group_index: Mapping[str, int],
    source: str,
) -> tuple[str, str, frozenset[str]]:
    if not isinstance(key, str) or key.count("|") != 1:
        raise ParameterDatabaseError(f"{source} interaction keys must have the form 'first|second'")
    first, second = key.split("|")
    if first == second or first not in group_index or second not in group_index:
        raise ParameterDatabaseError(f"{source} has invalid group interaction {key!r}")
    return first, second, frozenset((first, second))


def _unavailable_pairs(
    value: object,
    group_index: Mapping[str, int],
    source: str,
) -> set[frozenset[str]]:
    if value is None:
        return set()
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ParameterDatabaseError(f"{source} unavailable_interactions must be a string list")
    unavailable: set[frozenset[str]] = set()
    for key in value:
        _, _, unordered = _interaction_pair(key, group_index, source)
        if unordered in unavailable:
            raise ParameterDatabaseError(
                f"{source} contains duplicate unavailable interaction {key!r}"
            )
        unavailable.add(unordered)
    return unavailable


def _select_available_active_groups(
    groups: tuple[str, ...],
    fractions: Tensor,
    group_a: Tensor,
    group_b: Tensor,
    unavailable: set[frozenset[str]],
    source: str,
) -> tuple[tuple[str, ...], Tensor, Tensor, Tensor]:
    """Reject unavailable active pairs and remove unused groups."""
    if not unavailable:
        return groups, fractions, group_a, group_b
    active_indices = torch.nonzero(fractions.ne(0.0).any(dim=0), as_tuple=False).flatten()
    active_names = tuple(groups[index] for index in active_indices.tolist())
    for first_index, first in enumerate(active_names):
        for second in active_names[first_index + 1 :]:
            if frozenset((first, second)) in unavailable:
                raise ParameterDatabaseError(
                    f"{source} has no E-PPR78 interaction for active groups {first!r}|{second!r}"
                )
    return (
        active_names,
        fractions.index_select(1, active_indices),
        group_a.index_select(0, active_indices).index_select(1, active_indices),
        group_b.index_select(0, active_indices).index_select(1, active_indices),
    )


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
    """Load PPR78 or E-PPR78 group parameters for ``components``.

    Parameters
    ----------
    components
        Ordered component set defining selected decompositions, dtype, and
        device.
    source
        Bundled identifier, custom YAML path, mapping, or loaded parameter set.
    group_counts
        Optional component-by-group counts as a mapping or tensor. Values are
        normalized into group fractions and override only source component
        decompositions.

    Returns
    -------
    PPR78GroupContributionParameters
        Selected fractions and universal interaction tensors. Parameter sets
        with explicitly unavailable pairs are reduced to the groups active in
        ``components`` after availability has been checked.

    Raises
    ------
    KeyError
        If a requested component lacks a decomposition.
    ValueError
        If explicit counts have invalid shapes or values.
    ParameterDatabaseError
        If source identity, units, group inventory, interactions, or the
        availability of an active group pair is invalid.

    Notes
    -----
    ``source`` may be the default 2004 saturated-hydrocarbon PPR78 set,
    :data:`PPR78_HYDROGEN_WATER_GROUP_CONTRIBUTION`, a custom YAML path, or an
    in-memory :class:`~torch_flash.database.ModelParameterSet`. The global
    E-PPR78 parameterization is selected with
    :data:`DEFAULT_EPPR78_GROUP_CONTRIBUTION`. The
    H2/N2/H2O set is the exact active-group submatrix of the 2013 hydrogen and
    water extensions, not a refit. Explicit ``group_counts`` override only
    the component decompositions; the source still supplies the group
    inventory and interaction matrices.
    """
    loaded = load_model_parameters(source)
    if loaded.model_kind != "group_contribution":
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} is {loaded.model_kind!r}, not 'group_contribution'"
        )
    if loaded.model not in {"PPR78", "E-PPR78"}:
        raise ParameterDatabaseError(f"{loaded.identifier!r} model must be 'PPR78' or 'E-PPR78'")
    expected_units = {
        "A": "Pa",
        "B": "Pa",
        "group_fraction": "dimensionless",
        "reference_temperature": "K",
    }
    if any(loaded.units.get(key) != unit for key, unit in expected_units.items()):
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} must declare PPR78/E-PPR78 A, B, fraction, "
            "and temperature units"
        )
    parameters = loaded.parameters
    groups = _group_names(parameters.get("groups"), loaded.identifier)
    reference_temperature = _finite(
        parameters.get("reference_temperature"),
        f"{loaded.identifier} reference_temperature",
    )
    if reference_temperature <= 0.0:
        raise ParameterDatabaseError(f"{loaded.identifier} reference_temperature must be positive")
    group_a, group_b, unavailable = _interaction_matrices(
        parameters.get("interactions"),
        parameters.get("unavailable_interactions"),
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
    groups, fractions, group_a, group_b = _select_available_active_groups(
        groups,
        fractions,
        group_a,
        group_b,
        unavailable,
        loaded.identifier,
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
    """Construct a differentiable PPR78 or E-PPR78 mixing rule.

    Parameters
    ----------
    components
        Ordered component set.
    source
        PPR78 or E-PPR78 parameter source.
    group_counts
        Optional explicit component group counts overriding stored
        decompositions. Counts are normalized to molecular group fractions.
    trainable
        Register the independent, active off-diagonal A/B group interactions
        as PyTorch parameters.

    Returns
    -------
    PPR78Mixing
        Mixing-rule module on the component set's dtype and device.

    Raises
    ------
    KeyError
        If a requested component lacks a stored decomposition.
    ValueError
        If explicit group counts or tensor values are invalid.
    ParameterDatabaseError
        If the source inventory is malformed or an interaction between two
        active E-PPR78 groups is unavailable.

    Notes
    -----
    The implemented correlation is Eq. (5) of Jaubert and Mutelet,
    *Fluid Phase Equilibria* 224 (2004) 285--304,
    doi:10.1016/j.fluid.2004.06.059. E-PPR78 uses the same equation with the
    global interactions in Jaubert, Qian, Lasala, and Privat,
    *Fluid Phase Equilibria* 560 (2022) 113456,
    doi:10.1016/j.fluid.2022.113456, Table S4. For a source with unavailable
    pairs, only the groups active in ``components`` are retained after every
    active pair has been validated. All tensors honor the component set's
    dtype and device.
    """
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
    "DEFAULT_EPPR78_GROUP_CONTRIBUTION",
    "DEFAULT_PPR78_GROUP_CONTRIBUTION",
    "EPPR78_CCS_GROUP_CONTRIBUTION",
    "PPR78_HYDROGEN_WATER_GROUP_CONTRIBUTION",
    "PPR78GroupContributionParameters",
    "ppr78_group_contribution_parameters",
    "ppr78_mixing",
]
