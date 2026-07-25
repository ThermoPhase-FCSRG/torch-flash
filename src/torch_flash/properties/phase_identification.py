"""Physical phase-identification diagnostics for homogeneous and flashed states.

The five named criteria follow Bennett and Schmidt, *Energy & Fuels* 31
(2017), 3370-3379, doi:10.1021/acs.energyfuels.6b02316. Root selection,
physical phase identification, and the equilibrium phase count remain
separate concepts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.constants import R
from torch_flash.initialization import wilson_k_values
from torch_flash.types import (
    ChemicalState,
    PhaseIdentification,
    PhaseIdentificationCriterion,
    PhaseIdentityKind,
    PhaseKind,
    PhaseProperties,
    normalize_composition,
)

DEFAULT_PHASE_IDENTIFICATION_METHOD: PhaseIdentificationCriterion = "pedersen-volume-to-covolume"
DEFAULT_VOLUME_TO_COVOLUME_THRESHOLD = 1.75
DEFAULT_PSEUDO_CRITICAL_TEMPERATURE_FACTOR = 1.0
DEFAULT_AMBIGUITY_RELATIVE_TOLERANCE = 0.05


@dataclass(frozen=True)
class PhaseResponseDerivatives:
    """Autodiff response functions used by the two derivative criteria.

    Attributes
    ----------
    molar_volume
        Homogeneous physical molar volume in m3/mol.
    isothermal_compressibility
        ``kappa = -(1/V)(dV/dP)_T`` in 1/Pa.
    thermal_expansion_coefficient
        ``alpha = (1/V)(dV/dT)_P`` in 1/K.
    isothermal_compressibility_temperature_derivative
        ``(d kappa/dT)_P`` in 1/(Pa K).
    thermal_expansion_temperature_derivative
        ``(d alpha/dT)_P`` in 1/K^2.

    Notes
    -----
    The derivatives are assembled from first and second PyTorch autodiff
    derivatives of the model's explicit ``P(T, V, x)`` function. The supplied
    homogeneous root is evaluated once; no finite-difference step is used.
    """

    molar_volume: Tensor
    isothermal_compressibility: Tensor
    thermal_expansion_coefficient: Tensor
    isothermal_compressibility_temperature_derivative: Tensor
    thermal_expansion_temperature_derivative: Tensor


@dataclass(frozen=True)
class _PhaseResponseDetails:
    values: PhaseResponseDerivatives
    compressibility_derivative_scale: Tensor
    expansion_derivative_scale: Tensor


def _validate_options(threshold: float, ambiguity_relative_tolerance: float) -> None:
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("volume-to-covolume threshold must be finite and positive")
    if not math.isfinite(ambiguity_relative_tolerance) or ambiguity_relative_tolerance < 0.0:
        raise ValueError("ambiguity relative tolerance must be finite and non-negative")


def _validate_positive_factor(value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("pseudo-critical-temperature factor must be finite and positive")


def _unavailable_identification() -> PhaseIdentification:
    return PhaseIdentification(
        kind="unknown",
        method="unavailable",
        criterion_value=None,
        threshold=None,
        ambiguous=True,
    )


def _model_tensor(model: object, name: str) -> Tensor | None:
    value = getattr(model, name, None)
    return value if isinstance(value, Tensor) else None


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
    if not bool(torch.isfinite(covolume).all().detach()) or bool((covolume <= 0.0).any().detach()):
        raise ValueError("model mixture covolume must be finite and positive")
    return covolume


def li_pseudo_critical_temperature(
    model: object,
    composition: Tensor,
    *,
    factor: float = DEFAULT_PSEUDO_CRITICAL_TEMPERATURE_FACTOR,
    critical_volume: Tensor | None = None,
) -> Tensor:
    """Return Li's volume-weighted pseudo-critical temperature.

    Parameters
    ----------
    model
        Model exposing per-component critical temperatures and, unless
        supplied explicitly, critical molar volumes.
    composition
        Mole fractions with components on the final axis.
    factor
        Dimensionless tuning factor ``r1``. Bennett and Schmidt use one.
    critical_volume
        Optional per-component critical molar volumes in m3/mol.

    Returns
    -------
    Tensor
        Pseudo-critical temperature in K with the composition batch shape.

    Notes
    -----
    This implements ``Tc = r1*sum(xj*Vcj*Tcj)/sum(xj*Vcj)``. A phase is
    labeled vapor when its actual temperature exceeds this value.
    """
    _validate_positive_factor(factor)
    critical_temperature = _model_tensor(model, "critical_temperature")
    if critical_temperature is None:
        raise TypeError("pseudo-critical-temperature model lacks critical temperatures")
    volume = _model_tensor(model, "critical_volume") if critical_volume is None else critical_volume
    if volume is None:
        raise TypeError("pseudo-critical-temperature method requires critical volumes")
    if (
        critical_temperature.ndim != 1
        or volume.ndim != 1
        or critical_temperature.shape != volume.shape
    ):
        raise ValueError("critical temperatures and volumes must be equal-length vectors")
    if not bool(torch.isfinite(critical_temperature).all().detach()) or bool(
        (critical_temperature <= 0.0).any().detach()
    ):
        raise ValueError("critical temperatures must be finite and positive")
    if not bool(torch.isfinite(volume).all().detach()) or bool((volume <= 0.0).any().detach()):
        raise ValueError("critical volumes must be finite and positive")
    x = normalize_composition(composition)
    if x.shape[-1] != critical_temperature.numel():
        raise ValueError("composition and critical-property component counts differ")
    weights = x * volume
    return factor * torch.sum(weights * critical_temperature, dim=-1) / weights.sum(dim=-1)


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
    repulsive term.
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
    if not bool(torch.isfinite(ratio).all().detach()) or bool((ratio <= 0.0).any().detach()):
        raise ValueError("volume-to-covolume ratio must be finite and positive")
    return ratio


def negative_flash_residual(model: object, state: ChemicalState) -> Tensor:
    """Evaluate Perschke's negative-flash residual ``G(0.5)``.

    Wilson K-values are evaluated at the supplied ``T`` and ``P``. Positive
    residuals identify vapor and negative residuals identify liquid. The
    result is dimensionless and supports leading batch dimensions.
    """
    required = ("critical_temperature", "critical_pressure", "acentric_factor")
    if not all(isinstance(getattr(model, name, None), Tensor) for name in required):
        raise TypeError("negative-flash method requires critical constants and acentric factors")
    components = cast(ComponentSet, model)
    k_values = wilson_k_values(components, state.temperature, state.pressure)
    terms = state.composition * (k_values - 1.0) / (1.0 + 0.5 * (k_values - 1.0))
    return terms.sum(dim=-1)


def _negative_flash_scale(model: object, state: ChemicalState) -> Tensor:
    components = cast(ComponentSet, model)
    k_values = wilson_k_values(components, state.temperature, state.pressure)
    terms = state.composition * (k_values - 1.0) / (1.0 + 0.5 * (k_values - 1.0))
    return terms.abs().sum(dim=-1)


def _phase_response_details(
    model: object,
    state: ChemicalState,
    phase: PhaseKind,
    *,
    molar_volume: Tensor | None = None,
) -> _PhaseResponseDetails:
    if state.temperature.ndim != 0 or state.pressure.ndim != 0 or state.composition.ndim != 1:
        raise ValueError("phase-response derivatives require one scalar T-P state")
    pressure_function: Any = getattr(model, "pressure", None)
    volume_function: Any = getattr(model, "molar_volume", None)
    if not callable(pressure_function) or not callable(volume_function):
        raise TypeError("phase-response derivatives require pressure and molar_volume methods")
    volume = (
        cast(
            Tensor,
            volume_function(
                state.temperature,
                state.pressure,
                state.composition,
                phase,
            ),
        )
        if molar_volume is None
        else molar_volume
    )
    if (
        volume.ndim != 0
        or not bool(torch.isfinite(volume).detach())
        or bool((volume <= 0.0).detach())
    ):
        raise ValueError("phase molar volume must be a finite positive scalar")
    composition = state.composition

    def pressure_at_tv(temperature: Tensor, current_volume: Tensor) -> Tensor:
        return cast(Tensor, pressure_function(temperature, current_volume, composition))

    def response_at_tv(temperature: Tensor, current_volume: Tensor) -> Tensor:
        pressure_temperature, pressure_volume = torch.func.grad(
            pressure_at_tv,
            argnums=(0, 1),
        )(temperature, current_volume)
        volume_temperature = -pressure_temperature / pressure_volume
        compressibility = -1.0 / (current_volume * pressure_volume)
        expansion = volume_temperature / current_volume
        return torch.stack(
            (
                compressibility,
                expansion,
                volume_temperature,
                pressure_volume,
            )
        )

    response = response_at_tv(state.temperature, volume)
    if not bool(torch.isfinite(response).all().detach()):
        raise ValueError("phase-response autodiff produced nonfinite values")
    if bool((response[3] >= 0.0).detach()):
        raise ValueError("phase-response derivatives require a mechanically stable root")
    response_temperature, response_volume = torch.func.jacrev(
        response_at_tv,
        argnums=(0, 1),
    )(state.temperature, volume)
    volume_temperature = response[2]
    compressibility_terms = torch.stack(
        (
            response_temperature[0],
            response_volume[0] * volume_temperature,
        )
    )
    expansion_terms = torch.stack(
        (
            response_temperature[1],
            response_volume[1] * volume_temperature,
        )
    )
    compressibility_derivative = compressibility_terms.sum()
    expansion_derivative = expansion_terms.sum()
    derivatives = PhaseResponseDerivatives(
        molar_volume=volume,
        isothermal_compressibility=response[0],
        thermal_expansion_coefficient=response[1],
        isothermal_compressibility_temperature_derivative=compressibility_derivative,
        thermal_expansion_temperature_derivative=expansion_derivative,
    )
    if not bool(
        torch.isfinite(
            torch.stack(
                (
                    derivatives.isothermal_compressibility_temperature_derivative,
                    derivatives.thermal_expansion_temperature_derivative,
                )
            )
        )
        .all()
        .detach()
    ):
        raise ValueError("phase-response second derivatives are nonfinite")
    return _PhaseResponseDetails(
        derivatives,
        compressibility_terms.abs().sum(),
        expansion_terms.abs().sum(),
    )


def phase_response_derivatives(
    model: object,
    state: ChemicalState,
    phase: PhaseKind = "stable",
) -> PhaseResponseDerivatives:
    """Return autodiff response functions for one homogeneous state."""
    return _phase_response_details(model, state, phase).values


def _scalar_identification(
    *,
    criterion: Tensor,
    threshold: Tensor,
    method: PhaseIdentificationCriterion,
    vapor_if_positive: bool,
    ambiguity_margin: Tensor,
    ambiguity_limit: Tensor,
) -> PhaseIdentification:
    if criterion.ndim != 0 or threshold.ndim != 0:
        raise ValueError("phase identification currently accepts one scalar T-P state")
    if not bool(torch.isfinite(criterion).detach()) or not bool(torch.isfinite(threshold).detach()):
        raise ValueError("phase-identification criterion and threshold must be finite")
    positive = bool((criterion > threshold).detach())
    vapor = positive if vapor_if_positive else not positive
    kind: PhaseIdentityKind = "vapor" if vapor else "liquid"
    return PhaseIdentification(
        kind=kind,
        method=method,
        criterion_value=criterion,
        threshold=threshold,
        ambiguous=bool((ambiguity_margin.abs() <= ambiguity_limit).detach()),
    )


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
    equation_of_state_volume = compressibility_factor * R * state.temperature / state.pressure
    ratio = equation_of_state_volume / covolume
    if ratio.ndim != 0 or not bool(torch.isfinite(ratio).detach()) or bool((ratio <= 0.0).detach()):
        raise ValueError("volume-to-covolume ratio must be a finite positive scalar")
    threshold_tensor = ratio.new_tensor(threshold)
    return _scalar_identification(
        criterion=ratio,
        threshold=threshold_tensor,
        method="pedersen-volume-to-covolume",
        vapor_if_positive=True,
        ambiguity_margin=torch.log(ratio / threshold_tensor),
        ambiguity_limit=torch.log1p(ratio.new_tensor(ambiguity_relative_tolerance)),
    )


def identify_phase(
    model: object,
    state: ChemicalState,
    phase: PhaseKind = "stable",
    *,
    method: PhaseIdentificationCriterion = DEFAULT_PHASE_IDENTIFICATION_METHOD,
    threshold: float = DEFAULT_VOLUME_TO_COVOLUME_THRESHOLD,
    pseudo_critical_temperature_factor: float = DEFAULT_PSEUDO_CRITICAL_TEMPERATURE_FACTOR,
    ambiguity_relative_tolerance: float = DEFAULT_AMBIGUITY_RELATIVE_TOLERANCE,
) -> PhaseIdentification:
    """Identify one homogeneous state with a selected literature criterion.

    The default remains Pedersen's ``V/b`` method for backward compatibility.
    Every returned criterion tensor remains connected to the PyTorch graph;
    only the string label and ambiguity flag require a detached scalar
    decision.
    """
    _validate_options(threshold, ambiguity_relative_tolerance)
    _validate_positive_factor(pseudo_critical_temperature_factor)
    if state.temperature.ndim != 0 or state.pressure.ndim != 0 or state.composition.ndim != 1:
        raise ValueError("phase identification currently accepts one scalar T-P state")

    if method == "pedersen-volume-to-covolume":
        select_z: Any = getattr(model, "select_z", None)
        if not callable(select_z):
            raise TypeError("phase-identification model must implement select_z")
        compressibility_factor = cast(
            Tensor,
            select_z(
                state.temperature,
                state.pressure,
                state.composition,
                phase,
            ),
        )
        return _from_state_values(
            model,
            state,
            compressibility_factor,
            threshold=threshold,
            ambiguity_relative_tolerance=ambiguity_relative_tolerance,
        )

    if method == "li-pseudo-critical-temperature":
        if (
            _model_tensor(model, "critical_temperature") is None
            or _model_tensor(model, "critical_volume") is None
        ):
            return _unavailable_identification()
        pseudo_critical = li_pseudo_critical_temperature(
            model,
            state.composition,
            factor=pseudo_critical_temperature_factor,
        )
        ratio = state.temperature / pseudo_critical
        one = ratio.new_tensor(1.0)
        return _scalar_identification(
            criterion=ratio,
            threshold=one,
            method=method,
            vapor_if_positive=True,
            ambiguity_margin=torch.log(ratio),
            ambiguity_limit=torch.log1p(ratio.new_tensor(ambiguity_relative_tolerance)),
        )

    if method == "perschke-negative-flash":
        required = ("critical_temperature", "critical_pressure", "acentric_factor")
        if not all(_model_tensor(model, name) is not None for name in required):
            return _unavailable_identification()
        residual = negative_flash_residual(model, state)
        scale = _negative_flash_scale(model, state)
        zero = residual.new_zeros(())
        return _scalar_identification(
            criterion=residual,
            threshold=zero,
            method=method,
            vapor_if_positive=True,
            ambiguity_margin=residual,
            ambiguity_limit=scale * ambiguity_relative_tolerance,
        )

    if method in (
        "pasad-isothermal-compressibility-derivative",
        "bennett-thermal-expansion-derivative",
    ):
        if not callable(getattr(model, "pressure", None)) or not callable(
            getattr(model, "molar_volume", None)
        ):
            return _unavailable_identification()
        response = _phase_response_details(model, state, phase)
        if method == "pasad-isothermal-compressibility-derivative":
            criterion = response.values.isothermal_compressibility_temperature_derivative
            scale = response.compressibility_derivative_scale
        else:
            criterion = response.values.thermal_expansion_temperature_derivative
            scale = response.expansion_derivative_scale
        zero = criterion.new_zeros(())
        return _scalar_identification(
            criterion=criterion,
            threshold=zero,
            method=method,
            vapor_if_positive=False,
            ambiguity_margin=criterion,
            ambiguity_limit=scale * ambiguity_relative_tolerance,
        )

    raise ValueError(f"unknown phase-identification method {method!r}")


def identify_phase_from_properties(
    model: object,
    state: ChemicalState,
    properties: PhaseProperties,
    *,
    threshold: float = DEFAULT_VOLUME_TO_COVOLUME_THRESHOLD,
    ambiguity_relative_tolerance: float = DEFAULT_AMBIGUITY_RELATIVE_TOLERANCE,
) -> PhaseIdentification:
    """Apply the default ``V/b`` rule while reusing evaluated properties."""
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

    Existing named-method identifications are preserved. If none is
    available, the least-dense phase is labeled vapor only when its volume is
    sufficiently separated from the next-largest phase volume.
    """
    _validate_options(DEFAULT_VOLUME_TO_COVOLUME_THRESHOLD, ambiguity_relative_tolerance)
    if not phases:
        return phases
    has_named_identification = tuple(
        phase.phase_identification is not None
        and phase.phase_identification.method != "unavailable"
        for phase in phases
    )
    if any(has_named_identification):
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
