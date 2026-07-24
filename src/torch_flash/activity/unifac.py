"""Original-UNIFAC activity coefficients with PyTorch tensor kernels.

The equations follow Fredenslund, Jones, and Prausnitz, *AIChE Journal*
21 (1975), 1086-1099, doi:10.1002/aic.690210607.  The implementation uses
the notation summarized by Kontogeorgis and Folas, *Thermodynamic Models
for Industrial Applications* (2010), section 5.7, and the independently
derived ThermoPack memo by Hammer (2025).

UNIFAC variants are parameter sets, not interchangeable aliases.  This
module implements the original combinatorial and residual equations.  A
parameter database must identify the matching subgroup geometry and
directed main-group interaction table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import torch
from torch import Tensor, nn

from torch_flash.components import canonical_component_name
from torch_flash.config import resolve_tensor_options
from torch_flash.database import ModelParameterSet, ParameterSource, load_model_parameters
from torch_flash.exceptions import ParameterDatabaseError
from torch_flash.types import normalize_composition

GroupKey: TypeAlias = int | str
GroupAssignment: TypeAlias = Mapping[GroupKey, int | float]


class UNIFAC(nn.Module):
    r"""Original UNIFAC excess-Gibbs-energy model.

    Parameters
    ----------
    group_counts:
        Component-by-subgroup occurrence matrix :math:`\nu_k^{(i)}`.
    relative_volume, relative_surface_area:
        Selected subgroup :math:`R_k` and :math:`Q_k` values.
    subgroup_main_indices:
        Zero-based index of the selected main group for each subgroup.
    interaction:
        Directed selected-main-group matrix :math:`a_{mn}` in kelvin.
        Every required off-diagonal value must be supplied.
    coordination_number:
        Lattice coordination number :math:`z`; original UNIFAC uses 10.
    trainable:
        Store the main-group interaction matrix as an ``nn.Parameter``.

    Notes
    -----
    The model is vectorized over all leading dimensions of temperature and
    composition.  PyTorch autodiff can therefore differentiate activities,
    excess Gibbs energy, and temperature/composition derivatives on CPU or
    GPU.  Group assignment is discrete and is intentionally not trainable.
    """

    group_counts: Tensor
    relative_volume: Tensor
    relative_surface_area: Tensor
    subgroup_main_indices: Tensor
    interaction: Tensor
    subgroup_keys: tuple[str, ...]
    main_group_ids: tuple[int, ...]

    def __init__(
        self,
        group_counts: Tensor,
        relative_volume: Tensor,
        relative_surface_area: Tensor,
        subgroup_main_indices: Tensor,
        interaction: Tensor,
        *,
        coordination_number: float = 10.0,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if group_counts.ndim != 2 or not group_counts.shape[0] or not group_counts.shape[1]:
            raise ValueError("UNIFAC group counts must be a nonempty component-by-group matrix")
        group_shape = (group_counts.shape[1],)
        if relative_volume.shape != group_shape or relative_surface_area.shape != group_shape:
            raise ValueError("UNIFAC requires one R and Q value per selected subgroup")
        if subgroup_main_indices.shape != group_shape:
            raise ValueError("UNIFAC requires one main-group index per selected subgroup")
        if subgroup_main_indices.dtype != torch.long:
            raise ValueError("UNIFAC main-group indices must have torch.long dtype")
        if interaction.ndim != 2 or interaction.shape[0] != interaction.shape[1]:
            raise ValueError("UNIFAC interactions must be a square directed matrix")
        if not interaction.shape[0]:
            raise ValueError("UNIFAC requires at least one selected main group")
        tensors = (group_counts, relative_volume, relative_surface_area, interaction)
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("UNIFAC numerical inputs must be finite")
        if bool((group_counts < 0.0).any()) or bool((group_counts.sum(dim=-1) <= 0.0).any()):
            raise ValueError("each UNIFAC component needs nonnegative, nonempty group counts")
        if bool((relative_volume <= 0.0).any()) or bool((relative_surface_area < 0.0).any()):
            raise ValueError("UNIFAC R values must be positive and Q values nonnegative")
        if bool((subgroup_main_indices < 0).any()) or bool(
            (subgroup_main_indices >= interaction.shape[0]).any()
        ):
            raise ValueError("UNIFAC main-group indices are outside the interaction matrix")
        if not torch.allclose(
            torch.diagonal(interaction),
            torch.zeros(interaction.shape[0], dtype=interaction.dtype, device=interaction.device),
        ):
            raise ValueError("UNIFAC same-main-group interactions must be zero")
        if coordination_number <= 0.0:
            raise ValueError("UNIFAC coordination number must be positive")

        molecular_volume = group_counts @ relative_volume
        molecular_surface = group_counts @ relative_surface_area
        if bool((molecular_volume <= 0.0).any()) or bool((molecular_surface <= 0.0).any()):
            raise ValueError("each UNIFAC component must have positive molecular R and Q")

        self.coordination_number = float(coordination_number)
        self.subgroup_keys = ()
        self.main_group_ids = ()
        self.register_buffer("group_counts", group_counts.clone())
        self.register_buffer("relative_volume", relative_volume.clone())
        self.register_buffer("relative_surface_area", relative_surface_area.clone())
        self.register_buffer("subgroup_main_indices", subgroup_main_indices.clone())
        if trainable:
            self.interaction = nn.Parameter(interaction.clone())
        else:
            self.register_buffer("interaction", interaction.clone())

    @property
    def molecular_relative_volume(self) -> Tensor:
        """Return component molecular volume parameters :math:`r_i`."""
        return self.group_counts @ self.relative_volume

    @property
    def molecular_relative_surface_area(self) -> Tensor:
        """Return component molecular surface parameters :math:`q_i`."""
        return self.group_counts @ self.relative_surface_area

    def interaction_factors(self, temperature: Tensor) -> Tensor:
        r"""Return :math:`\Psi_{mn}=\exp(-a_{mn}/T)`."""
        if not torch.compiler.is_compiling() and bool((temperature <= 0.0).any()):
            raise ValueError("temperature must be positive")
        interaction = self.interaction - torch.diag_embed(torch.diagonal(self.interaction))
        return torch.exp(-interaction / temperature[..., None, None])

    def _state(self, temperature: Tensor, composition: Tensor) -> tuple[Tensor, Tensor]:
        x = normalize_composition(composition)
        if x.shape[-1] != self.group_counts.shape[0]:
            raise ValueError("UNIFAC composition size must match the component group matrix")
        if not torch.compiler.is_compiling() and bool((temperature <= 0.0).any()):
            raise ValueError("temperature must be positive")
        batch_shape = torch.broadcast_shapes(temperature.shape, x.shape[:-1])
        return temperature.expand(batch_shape), x.expand((*batch_shape, x.shape[-1]))

    def combinatorial_log_activity_coefficients(
        self,
        composition: Tensor,
    ) -> Tensor:
        r"""Return the original Flory-Huggins/Staverman-Guggenheim term."""
        x = normalize_composition(composition)
        if x.shape[-1] != self.group_counts.shape[0]:
            raise ValueError("UNIFAC composition size must match the component group matrix")
        r = self.molecular_relative_volume
        q = self.molecular_relative_surface_area
        r_bar = torch.sum(x * r, dim=-1, keepdim=True)
        q_bar = torch.sum(x * q, dim=-1, keepdim=True)
        phi_over_x = r / r_bar
        theta_over_phi = (q / r) * (r_bar / q_bar)
        return (
            torch.log(phi_over_x)
            + 1.0
            - phi_over_x
            + 0.5
            * self.coordination_number
            * q
            * (torch.log(theta_over_phi) - 1.0 + theta_over_phi.reciprocal())
        )

    def _group_log_activity_coefficients(
        self,
        surface_fractions: Tensor,
        interaction_factors: Tensor,
    ) -> Tensor:
        column_sum = torch.sum(
            surface_fractions[..., :, None] * interaction_factors,
            dim=-2,
        )
        correction = torch.sum(
            surface_fractions[..., None, :] * interaction_factors / column_sum[..., None, :],
            dim=-1,
        )
        return self.relative_surface_area * (1.0 - torch.log(column_sum) - correction)

    def residual_log_activity_coefficients(
        self,
        temperature: Tensor,
        composition: Tensor,
    ) -> Tensor:
        r"""Return the solution-of-groups residual contribution."""
        temperature, x = self._state(temperature, composition)
        group_abundance = torch.einsum("...i,ik->...k", x, self.group_counts)
        mixture_surface = self.relative_surface_area * group_abundance
        mixture_surface = mixture_surface / mixture_surface.sum(dim=-1, keepdim=True)

        main_factors = self.interaction_factors(temperature)
        subgroup_factors = main_factors[
            ...,
            self.subgroup_main_indices[:, None],
            self.subgroup_main_indices[None, :],
        ]
        mixture_log_group = self._group_log_activity_coefficients(
            mixture_surface,
            subgroup_factors,
        )

        pure_surface = self.group_counts * self.relative_surface_area
        pure_surface = pure_surface / pure_surface.sum(dim=-1, keepdim=True)
        batch_shape = x.shape[:-1]
        pure_surface = pure_surface.expand((*batch_shape, *pure_surface.shape))
        pure_log_group = self._group_log_activity_coefficients(
            pure_surface,
            subgroup_factors[..., None, :, :],
        )
        return torch.sum(
            self.group_counts * (mixture_log_group[..., None, :] - pure_log_group),
            dim=-1,
        )

    def log_activity_coefficients(self, temperature: Tensor, composition: Tensor) -> Tensor:
        """Return ``log(gamma)`` for original UNIFAC."""
        _, x = self._state(temperature, composition)
        return self.combinatorial_log_activity_coefficients(x) + (
            self.residual_log_activity_coefficients(temperature, x)
        )

    def excess_gibbs_rt(self, temperature: Tensor, composition: Tensor) -> Tensor:
        """Return dimensionless molar excess Gibbs energy, ``gE/(RT)``."""
        _, x = self._state(temperature, composition)
        return torch.sum(x * self.log_activity_coefficients(temperature, x), dim=-1)


def _numeric(value: object, context: str) -> float:
    if not isinstance(value, int | float):
        raise ParameterDatabaseError(f"{context} must be numeric")
    return float(value)


def _subgroup_records(
    loaded: ModelParameterSet,
) -> tuple[tuple[str, ...], tuple[Mapping[str, object], ...], dict[str, str]]:
    raw = loaded.parameters.get("subgroups")
    if not isinstance(raw, Mapping) or not raw:
        raise ParameterDatabaseError(f"{loaded.identifier!r} requires a nonempty subgroups mapping")
    keys: list[str] = []
    records: list[Mapping[str, object]] = []
    aliases: dict[str, str] = {}
    ambiguous: set[str] = set()
    for key, record in raw.items():
        if not isinstance(key, str) or not isinstance(record, Mapping):
            raise ParameterDatabaseError("UNIFAC subgroup keys and records must be mappings")
        keys.append(key)
        records.append(record)
        candidates = (key, record.get("number"), record.get("name"))
        for candidate in candidates:
            if candidate is None:
                continue
            normalized = str(candidate).strip().lower()
            if normalized in aliases and aliases[normalized] != key:
                ambiguous.add(normalized)
            else:
                aliases[normalized] = key
    for normalized in ambiguous:
        aliases.pop(normalized, None)
    return tuple(keys), tuple(records), aliases


def _resolve_assignment(
    assignment: GroupAssignment,
    aliases: Mapping[str, str],
    key_to_index: Mapping[str, int],
) -> dict[int, float]:
    resolved: dict[int, float] = {}
    if not isinstance(assignment, Mapping) or not assignment:
        raise ValueError("each UNIFAC component requires a nonempty group assignment")
    for raw_key, raw_count in assignment.items():
        normalized = str(raw_key).strip().lower()
        key = aliases.get(normalized)
        if key is None:
            raise KeyError(
                f"unknown or ambiguous UNIFAC subgroup {raw_key!r}; use its subgroup key or number"
            )
        count = _numeric(raw_count, f"UNIFAC count for {raw_key!r}")
        if count < 0.0:
            raise ValueError("UNIFAC group counts cannot be negative")
        if count:
            index = key_to_index[key]
            resolved[index] = resolved.get(index, 0.0) + count
    if not resolved:
        raise ValueError("each UNIFAC component requires at least one positive group count")
    return resolved


def _bundled_assignments(
    loaded: ModelParameterSet,
    names: Sequence[str],
) -> tuple[GroupAssignment, ...]:
    raw = loaded.parameters.get("component_assignments")
    if not isinstance(raw, Mapping):
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} has no bundled component_assignments mapping"
        )
    assignments: list[GroupAssignment] = []
    for name in names:
        canonical = canonical_component_name(name, strict=False)
        assignment = raw.get(canonical)
        if not isinstance(assignment, Mapping):
            raise KeyError(f"{loaded.identifier} has no UNIFAC group assignment for {canonical!r}")
        assignments.append(assignment)
    return tuple(assignments)


def _unifac_from_loaded(
    loaded: ModelParameterSet,
    *,
    names: Sequence[str] | None,
    group_assignments: Sequence[GroupAssignment] | None,
    dtype: torch.dtype,
    device: torch.device | str | None,
    trainable: bool,
) -> UNIFAC:
    if group_assignments is None:
        if names is None or not names:
            raise ValueError("UNIFAC requires component names or explicit group_assignments")
        group_assignments = _bundled_assignments(loaded, names)
    elif names is not None and len(names) != len(group_assignments):
        raise ValueError("UNIFAC names and group_assignments must have the same length")
    if not group_assignments:
        raise ValueError("UNIFAC requires at least one component group assignment")

    keys, records, aliases = _subgroup_records(loaded)
    key_to_index = {key: index for index, key in enumerate(keys)}
    resolved = tuple(
        _resolve_assignment(assignment, aliases, key_to_index) for assignment in group_assignments
    )
    selected_indices = tuple(sorted({index for assignment in resolved for index in assignment}))
    selected_lookup = {old: new for new, old in enumerate(selected_indices)}
    counts = torch.zeros(
        (len(resolved), len(selected_indices)),
        dtype=dtype,
        device=device,
    )
    for component_index, assignment in enumerate(resolved):
        for old_index, count in assignment.items():
            counts[component_index, selected_lookup[old_index]] = count

    selected_records = tuple(records[index] for index in selected_indices)
    volumes = torch.tensor(
        [
            _numeric(record.get("relative_volume"), "UNIFAC relative_volume")
            for record in selected_records
        ],
        dtype=dtype,
        device=device,
    )
    surfaces = torch.tensor(
        [
            _numeric(record.get("relative_surface_area"), "UNIFAC relative_surface_area")
            for record in selected_records
        ],
        dtype=dtype,
        device=device,
    )
    main_ids: list[int] = []
    for record in selected_records:
        main_group = record.get("main_group")
        if not isinstance(main_group, int):
            raise ParameterDatabaseError("UNIFAC subgroup main_group values must be integers")
        main_ids.append(main_group)
    selected_main_ids = tuple(sorted(set(main_ids)))
    main_lookup = {value: index for index, value in enumerate(selected_main_ids)}
    subgroup_main_indices = torch.tensor(
        [main_lookup[value] for value in main_ids],
        dtype=torch.long,
        device=device,
    )

    interaction_records = loaded.parameters.get("interactions")
    if not isinstance(interaction_records, Sequence) or isinstance(interaction_records, str):
        raise ParameterDatabaseError(f"{loaded.identifier!r} requires an interactions sequence")
    values: dict[tuple[int, int], float] = {}
    for row in interaction_records:
        if (
            not isinstance(row, Sequence)
            or isinstance(row, str)
            or len(row) != 3
            or not isinstance(row[0], int)
            or not isinstance(row[1], int)
        ):
            raise ParameterDatabaseError(
                "each UNIFAC interaction must be [main_group_i, main_group_j, a_ij]"
            )
        pair = (row[0], row[1])
        if pair in values:
            raise ParameterDatabaseError(f"duplicate UNIFAC interaction for main groups {pair}")
        values[pair] = _numeric(row[2], f"UNIFAC interaction {pair}")

    interaction = torch.zeros(
        (len(selected_main_ids), len(selected_main_ids)),
        dtype=dtype,
        device=device,
    )
    missing: list[tuple[int, int]] = []
    for row, main_i in enumerate(selected_main_ids):
        for column, main_j in enumerate(selected_main_ids):
            if main_i == main_j:
                continue
            value = values.get((main_i, main_j))
            if value is None:
                missing.append((main_i, main_j))
            else:
                interaction[row, column] = value
    if missing:
        pairs = ", ".join(f"{left}->{right}" for left, right in missing)
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} lacks required directed UNIFAC interactions: {pairs}"
        )
    coordination = _numeric(
        loaded.parameters.get("coordination_number", 10.0),
        "UNIFAC coordination_number",
    )
    model = UNIFAC(
        counts,
        volumes,
        surfaces,
        subgroup_main_indices,
        interaction,
        coordination_number=coordination,
        trainable=trainable,
    )
    model.subgroup_keys = tuple(keys[index] for index in selected_indices)
    model.main_group_ids = selected_main_ids
    return model


def unifac_model(
    parameter_set: ParameterSource = "unifac-original",
    names: Sequence[str] | None = None,
    *,
    group_assignments: Sequence[GroupAssignment] | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    trainable: bool = False,
) -> UNIFAC:
    """Construct original UNIFAC from cached YAML or explicit parameters.

    ``names`` select bundled, audited component fragmentations.  Arbitrary
    molecules use ``group_assignments``, where each mapping may be keyed by
    subgroup key, published subgroup number, or an unambiguous subgroup name.
    The optional :func:`unifac_groups_from_identifiers` adapter can generate
    these mappings with ``ugropy``.
    """
    dtype, device = resolve_tensor_options(dtype, device)
    loaded = load_model_parameters(parameter_set)
    if loaded.model_kind != "activity":
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} is {loaded.model_kind!r}, not 'activity'"
        )
    if loaded.model.strip().lower().replace("_", "-") not in ("unifac", "original-unifac"):
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} is {loaded.model!r}, not original UNIFAC"
        )
    return _unifac_from_loaded(
        loaded,
        names=names,
        group_assignments=group_assignments,
        dtype=dtype,
        device=device,
        trainable=trainable,
    )


def unifac_groups_from_identifiers(
    identifiers: Sequence[str],
    *,
    identifier_type: str = "smiles",
) -> tuple[dict[int, float], ...]:
    """Fragment molecules with the optional MIT-licensed ``ugropy`` package.

    SMILES inputs are recommended for reproducibility.  Name lookup delegates
    to PubChem and therefore requires network access and may change outside
    ``torch-flash``.  Fragmentation is discrete; inspect every returned map
    before using it in regression or safety-critical calculations.
    """
    try:
        from ugropy import unifac as ugropy_unifac
    except ImportError as exc:
        raise ImportError(
            "automatic UNIFAC fragmentation requires 'ugropy'; install torch-flash[groups]"
        ) from exc
    if identifier_type not in ("name", "smiles"):
        raise ValueError("identifier_type must be 'name' or 'smiles'")
    assignments: list[dict[int, float]] = []
    for identifier in identifiers:
        result = ugropy_unifac.get_groups(identifier, identifier_type)
        if isinstance(result, list):
            if len(result) != 1:
                raise ValueError(
                    f"ugropy returned {len(result)} fragmentations for {identifier!r}; "
                    "select a group assignment explicitly"
                )
            result = result[0]
        raw = getattr(result, "subgroups_num", None)
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError(f"ugropy could not assign original-UNIFAC groups to {identifier!r}")
        assignments.append({int(key): float(value) for key, value in raw.items()})
    return tuple(assignments)


__all__ = [
    "UNIFAC",
    "GroupAssignment",
    "GroupKey",
    "unifac_groups_from_identifiers",
    "unifac_model",
]
