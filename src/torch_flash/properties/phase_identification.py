"""Physical phase identification for homogeneous and flashed states.

Pedersen, Christensen, and Shaikh, *Phase Behavior of Petroleum Reservoir
Fluids*, 3rd ed. (2024), section 6.6, recommend ``V/b = 1.75`` as a practical
liquid/gas separator for the SRK and PR equations. This module implements that
criterion without confusing an EoS root label with physical phase identity.
For models without a cubic covolume, a multiphase result can only use the
weaker density-ordering convention described in the same section.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import torch
from torch import Tensor

from torch_flash.constants import R
from torch_flash.types import (
    ChemicalState,
    PhaseIdentification,
    PhaseIdentityKind,
    PhaseKind,
    PhaseProperties,
)

DEFAULT_VOLUME_TO_COVOLUME_THRESHOLD = 1.75
DEFAULT_AMBIGUITY_RELATIVE_TOLERANCE = 0.05


def _validate_options(threshold: float, ambiguity_relative_tolerance: float) -> None:
    if not torch.isfinite(torch.tensor(threshold)) or threshold <= 0.0:
        raise ValueError("volume-to-covolume threshold must be finite and positive")
    if (
        not torch.isfinite(torch.tensor(ambiguity_relative_tolerance))
        or ambiguity_relative_tolerance < 0.0
    ):
        raise ValueError("ambiguity relative tolerance must be finite and non-negative")


def _unavailable_identification() -> PhaseIdentification:
    return PhaseIdentification(
        kind="unknown",
        method="unavailable",
        criterion_value=None,
        threshold=None,
        ambiguous=True,
    )


def _covolume(model: object, state: ChemicalState) -> Tensor | None:
    mixture_parameters: Any = getattr(model, "mixture_parameters", None)
    if not callable(mixture_parameters):
        return None
    _, covolume = cast(
        tuple[Tensor, Tensor],
        mixture_parameters(
            state.temperature,
            state.composition,
        ),
    )
    if not bool(torch.isfinite(covolume).all()) or bool((covolume <= 0.0).any()):
        raise ValueError("model mixture covolume must be finite and positive")
    return covolume


def volume_to_covolume_ratio(
    model: object,
    state: ChemicalState,
    phase: PhaseKind = "stable",
) -> Tensor:
    """Return the differentiable cubic-family ``V/b`` phase diagnostic.

    Parameters
    ----------
    model
        Model implementing ``select_z`` and cubic ``mixture_parameters``.
    state
        Temperature in K, pressure in Pa, and mole fractions.
    phase
        Homogeneous root-selection request.

    Returns
    -------
    Tensor
        Positive dimensionless EOS volume-to-covolume ratio.

    Raises
    ------
    TypeError
        If the model lacks root selection or a mixture covolume.
    ValueError
        If the evaluated covolume or ratio is nonpositive/nonfinite.

    Notes
    -----
    Leading batch dimensions are supported. Cubic volume translations are
    excluded because Pedersen's criterion uses the volume entering the cubic
    repulsive term. A model without ``mixture_parameters`` has no defined
    covolume and raises ``TypeError``.
    """
    select_z: Any = getattr(model, "select_z", None)
    if not callable(select_z):
        raise TypeError("phase-identification model must implement select_z")
    covolume = _covolume(model, state)
    if covolume is None:
        raise TypeError("phase-identification model does not expose a mixture covolume")
    compressibility_factor = cast(
        Tensor,
        select_z(
            state.temperature,
            state.pressure,
            state.composition,
            phase,
        ),
    )
    equation_of_state_volume = compressibility_factor * R * state.temperature / state.pressure
    ratio = equation_of_state_volume / covolume
    if not bool(torch.isfinite(ratio).all()) or bool((ratio <= 0.0).any()):
        raise ValueError("volume-to-covolume ratio must be finite and positive")
    return ratio


def _from_state_values(
    model: object,
    state: ChemicalState,
    compressibility_factor: Tensor,
    *,
    threshold: float,
    ambiguity_relative_tolerance: float,
) -> PhaseIdentification:
    covolume = _covolume(model, state)
    if covolume is None:
        return _unavailable_identification()

    # Cubic volume translations are property corrections, not part of the
    # repulsive EoS volume used in Pedersen's V/b criterion. Z*RT/P therefore
    # gives the appropriate unshifted volume for translated cubic models.
    equation_of_state_volume = compressibility_factor * R * state.temperature / state.pressure
    ratio = equation_of_state_volume / covolume
    if ratio.ndim != 0 or not bool(torch.isfinite(ratio)) or float(ratio.detach()) <= 0.0:
        raise ValueError("volume-to-covolume ratio must be a finite positive scalar")
    threshold_tensor = ratio.new_tensor(threshold)
    log_margin = torch.log(ratio / threshold_tensor)
    ambiguity_limit = torch.log1p(ratio.new_tensor(ambiguity_relative_tolerance))
    kind: PhaseIdentityKind = "liquid" if float(ratio.detach()) < threshold else "vapor"
    return PhaseIdentification(
        kind=kind,
        method="pedersen-volume-to-covolume",
        criterion_value=ratio,
        threshold=threshold_tensor,
        ambiguous=bool((log_margin.abs() <= ambiguity_limit).detach()),
    )


def identify_phase(
    model: object,
    state: ChemicalState,
    phase: PhaseKind = "stable",
    *,
    threshold: float = DEFAULT_VOLUME_TO_COVOLUME_THRESHOLD,
    ambiguity_relative_tolerance: float = DEFAULT_AMBIGUITY_RELATIVE_TOLERANCE,
) -> PhaseIdentification:
    """Identify a scalar homogeneous state using the Pedersen ``V/b`` rule.

    Parameters
    ----------
    model
        Model implementing root selection and optionally cubic mixture
        covolume parameters.
    state
        One scalar TP state.
    phase
        Root-selection request.
    threshold
        Dimensionless ``V/b`` separator.
    ambiguity_relative_tolerance
        Relative band around the separator marked ambiguous.

    Returns
    -------
    PhaseIdentification
        Physical-identity diagnostic or ``unknown`` when a covolume is
        unavailable.

    Raises
    ------
    ValueError
        If options or state shapes are invalid.
    TypeError
        If the model does not implement root selection.

    Notes
    -----
    The default threshold of 1.75 is documented by Pedersen et al. for SRK and
    PR. Models that expose compatible cubic-family ``mixture_parameters`` can
    use the same machinery, but a non-SRK/PR threshold requires independent
    validation. If the model does not expose a covolume, ``unknown`` is
    returned rather than inferring physical identity from the selected root.
    """
    _validate_options(threshold, ambiguity_relative_tolerance)
    if state.temperature.ndim != 0 or state.pressure.ndim != 0 or state.composition.ndim != 1:
        raise ValueError("phase identification currently accepts one scalar T-P state")
    select_z: Any = getattr(model, "select_z", None)
    if not callable(select_z):
        raise TypeError("phase-identification model must implement select_z")
    compressibility_factor = select_z(
        state.temperature,
        state.pressure,
        state.composition,
        phase,
    )
    return _from_state_values(
        model,
        state,
        compressibility_factor,
        threshold=threshold,
        ambiguity_relative_tolerance=ambiguity_relative_tolerance,
    )


def identify_phase_from_properties(
    model: object,
    state: ChemicalState,
    properties: PhaseProperties,
    *,
    threshold: float = DEFAULT_VOLUME_TO_COVOLUME_THRESHOLD,
    ambiguity_relative_tolerance: float = DEFAULT_AMBIGUITY_RELATIVE_TOLERANCE,
) -> PhaseIdentification:
    """Identify a phase while reusing already evaluated state properties.

    Parameters
    ----------
    model
        Model that may expose cubic mixture covolume parameters.
    state
        Homogeneous scalar TP state associated with ``properties``.
    properties
        Previously evaluated properties supplying the compressibility factor.
    threshold
        Dimensionless ``V/b`` separator.
    ambiguity_relative_tolerance
        Relative band around ``threshold`` classified as ambiguous.

    Returns
    -------
    PhaseIdentification
        Pedersen ``V/b`` classification, or an unavailable/unknown result for
        batched states or models without a compatible covolume.
    """
    _validate_options(threshold, ambiguity_relative_tolerance)
    if state.temperature.ndim != 0 or state.pressure.ndim != 0 or state.composition.ndim != 1:
        return _unavailable_identification()
    return _from_state_values(
        model,
        state,
        properties.compressibility_factor,
        threshold=threshold,
        ambiguity_relative_tolerance=ambiguity_relative_tolerance,
    )


def identify_flash_phases(
    phases: tuple[PhaseProperties, ...],
    *,
    ambiguity_relative_tolerance: float = DEFAULT_AMBIGUITY_RELATIVE_TOLERANCE,
) -> tuple[PhaseProperties, ...]:
    """Fill unavailable multiphase identities by molar-volume ordering.

    Parameters
    ----------
    phases
        Homogeneous phase properties from one flash result.
    ambiguity_relative_tolerance
        Minimum relative molar-volume separation needed to label the
        least-dense phase as vapor.

    Returns
    -------
    tuple
        Phase records with physical-identification diagnostics filled where
        possible.

    Raises
    ------
    ValueError
        If phase volumes are invalid or the ambiguity tolerance is negative.

    Notes
    -----
    Existing ``V/b`` identifications are preserved. If no phase has a cubic
    covolume diagnostic, the least-dense phase (largest molar volume) is
    labeled vapor and the remaining phases liquid. Near-equal leading volumes
    are returned as ambiguous ``unknown`` because density ordering cannot
    distinguish liquid-liquid equilibrium from a vapor-liquid split.
    """
    _validate_options(DEFAULT_VOLUME_TO_COVOLUME_THRESHOLD, ambiguity_relative_tolerance)
    if not phases:
        return phases
    has_covolume_identification = tuple(
        phase.phase_identification is not None
        and phase.phase_identification.method != "unavailable"
        for phase in phases
    )
    if any(has_covolume_identification):
        return phases
    if len(phases) == 1:
        return phases

    volumes = torch.stack(tuple(phase.molar_volume for phase in phases))
    if volumes.ndim != 1 or not bool(torch.isfinite(volumes).all()) or bool((volumes <= 0.0).any()):
        raise ValueError("phase molar volumes must be finite positive scalars")
    sorted_volumes, _ = torch.sort(volumes)
    vapor_volume = sorted_volumes[-1]
    next_volume = sorted_volumes[-2]
    separator = torch.sqrt(vapor_volume * next_volume)
    separation = vapor_volume / next_volume
    ambiguous = bool(separation <= 1.0 + ambiguity_relative_tolerance)

    identified = []
    for phase, volume in zip(phases, volumes, strict=True):
        if ambiguous:
            kind: PhaseIdentityKind = "unknown"
        else:
            kind = "vapor" if bool(volume > separator) else "liquid"
        identification = PhaseIdentification(
            kind=kind,
            method="density-ordering",
            criterion_value=volume,
            threshold=separator,
            ambiguous=ambiguous,
        )
        identified.append(replace(phase, phase_identification=identification))
    return tuple(identified)
