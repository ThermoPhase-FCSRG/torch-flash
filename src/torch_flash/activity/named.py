"""Activity-model constructors backed by versioned YAML parameter sets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from torch_flash.components import canonical_component_name, component_set
from torch_flash.config import resolve_tensor_options
from torch_flash.constants import R
from torch_flash.database import ParameterSource, load_model_parameters
from torch_flash.exceptions import ParameterDatabaseError

from .models import NRTL, AnchoredHuronVidalNRTL, HuronVidalNRTL, Wilson
from .unifac import UNIFAC, GroupAssignment, _unifac_from_loaded

ActivityModel = AnchoredHuronVidalNRTL | HuronVidalNRTL | NRTL | UNIFAC | Wilson


def _numeric_matrix(
    parameters: Mapping[str, object],
    key: str,
    order: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> Tensor:
    values = parameters.get(key)
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise ParameterDatabaseError(f"activity parameter {key!r} must be a matrix")
    try:
        reordered = [[values[row][column] for column in order] for row in order]
        tensor = torch.tensor(reordered, dtype=dtype, device=device)
    except (IndexError, TypeError, ValueError) as exc:
        raise ParameterDatabaseError(f"activity parameter {key!r} is not a numeric matrix") from exc
    return tensor


def _numeric_vector(
    parameters: Mapping[str, object],
    key: str,
    order: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> Tensor:
    values = parameters.get(key)
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise ParameterDatabaseError(f"activity parameter {key!r} must be a vector")
    try:
        tensor = torch.tensor([values[index] for index in order], dtype=dtype, device=device)
    except (IndexError, TypeError, ValueError) as exc:
        raise ParameterDatabaseError(f"activity parameter {key!r} is not numeric") from exc
    return tensor


def activity_model(
    parameter_set: ParameterSource,
    names: tuple[str, ...] | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    trainable: bool = False,
    covolumes: Tensor | None = None,
    group_assignments: Sequence[GroupAssignment] | None = None,
) -> ActivityModel:
    """Construct NRTL, HV-NRTL, Wilson, or original UNIFAC from YAML.

    Parameters
    ----------
    parameter_set
        Bundled identifier, YAML path, mapping, or loaded activity parameter
        set.
    names
        Optional requested component order. Omitting it uses the stored order.
    dtype, device
        Tensor placement, defaulting to runtime configuration.
    trainable
        Register supported interaction tensors as PyTorch parameters.
    covolumes
        Optional HV-NRTL component covolumes in m3/mol.
    group_assignments
        Optional explicit UNIFAC subgroup counts, one mapping per component.

    Returns
    -------
    ActivityModel
        Model matching the parameter document's exact identity.

    Raises
    ------
    KeyError
        If requested components or UNIFAC groups are unsupported.
    ValueError
        If component order or model-specific overrides are inconsistent.
    ParameterDatabaseError
        If model identity, units, or parameter records are malformed.

    Notes
    -----
    The requested component order may differ from the stored order; vectors
    and matrices are permuted consistently. For custom fitting workflows, the
    model classes continue to accept explicit tensors directly. UNIFAC uses
    bundled fragmentations selected by ``names`` or explicit
    ``group_assignments``.
    """
    dtype, device = resolve_tensor_options(dtype, device)
    loaded = load_model_parameters(parameter_set)
    if loaded.model_kind != "activity":
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} is {loaded.model_kind!r}, not 'activity'"
        )
    parameters = loaded.parameters
    model_name = loaded.model.strip().lower().replace("_", "-")
    if model_name in ("unifac", "original-unifac"):
        return _unifac_from_loaded(
            loaded,
            names=names,
            group_assignments=group_assignments,
            dtype=dtype,
            device=device,
            trainable=trainable,
        )
    if group_assignments is not None:
        raise ValueError("group_assignments are only valid for UNIFAC")
    stored_names = parameters.get("components")
    if not isinstance(stored_names, Sequence) or isinstance(stored_names, str):
        raise ParameterDatabaseError(f"{loaded.identifier!r} requires a components list")
    canonical_stored = tuple(
        canonical_component_name(str(name), strict=False) for name in stored_names
    )
    selected = (
        canonical_stored
        if names is None
        else tuple(canonical_component_name(name, strict=False) for name in names)
    )
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("activity-model component names must be non-empty and unique")
    try:
        order = tuple(canonical_stored.index(name) for name in selected)
    except ValueError as exc:
        raise KeyError(
            f"{loaded.identifier} has no activity parameters for one or more of {selected!r}"
        ) from exc

    if model_name == "nrtl":
        return NRTL(
            _numeric_matrix(parameters, "interaction", order, dtype=dtype, device=device),
            _numeric_matrix(parameters, "nonrandomness", order, dtype=dtype, device=device),
            trainable=trainable,
        )
    if model_name == "wilson":
        return Wilson(
            _numeric_matrix(parameters, "interaction", order, dtype=dtype, device=device),
            _numeric_vector(parameters, "molar_volumes", order, dtype=dtype, device=device),
            trainable=trainable,
        )
    if model_name not in ("hv-nrtl", "huron-vidal-nrtl"):
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} has unsupported activity model {loaded.model!r}"
        )

    if covolumes is None and "covolumes" in parameters:
        covolumes = _numeric_vector(
            parameters,
            "covolumes",
            order,
            dtype=dtype,
            device=device,
        )
    if covolumes is None:
        cubic_source = parameters.get("covolume_cubic_parameter_set")
        if not isinstance(cubic_source, str):
            raise ParameterDatabaseError(
                f"{loaded.identifier!r} requires covolumes or 'covolume_cubic_parameter_set'"
            )
        cubic = load_model_parameters(cubic_source)
        omega_b = cubic.parameters.get("omega_b")
        if cubic.model_kind != "cubic" or not isinstance(omega_b, int | float):
            raise ParameterDatabaseError(f"{cubic.identifier!r} cannot provide cubic covolumes")
        components = component_set(selected, dtype=dtype, device=device)
        covolumes = (
            float(omega_b) * R * components.critical_temperature / components.critical_pressure
        )
    else:
        covolumes = covolumes.to(dtype=dtype, device=device)
    if covolumes.shape != (len(selected),):
        raise ValueError("one activity-model covolume is required per selected component")
    return HuronVidalNRTL(
        _numeric_matrix(parameters, "energy_over_r", order, dtype=dtype, device=device),
        _numeric_matrix(
            parameters,
            "temperature_coefficient",
            order,
            dtype=dtype,
            device=device,
        ),
        _numeric_matrix(parameters, "nonrandomness", order, dtype=dtype, device=device),
        covolumes,
        trainable=trainable,
    )


__all__ = ["ActivityModel", "activity_model"]
