"""Saturation points and phase-envelope tracing.

Thermodynamic formulation and continuation context follow Michelsen and
Mollerup, *Thermodynamic Models: Fundamentals & Computational Aspects*,
2nd ed. (2007), chapter 12, ISBN 978-87-989961-3-2.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, cast

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.initialization import wilson_k_values
from torch_flash.properties.state import HelmholtzStateModel, StateModel
from torch_flash.solvers import damped_newton
from torch_flash.types import PhaseKind, normalize_composition

SaturationKind = Literal["bubble", "dew"]


@dataclass(frozen=True)
class SaturationPoint:
    """One bubble- or dew-point solution.

    Attributes
    ----------
    temperature:
        Specified absolute temperature in K.
    pressure:
        Solved saturation pressure in Pa.
    incipient_composition:
        Normalized composition of the incipient phase.
    k_values:
        Final equilibrium ratios in feed-component order.
    kind:
        ``"bubble"`` or ``"dew"``.
    iterations:
        Number of Newton iterations reported by the final solve.
    converged:
        Whether the complete equilibrium/closure residual met tolerance.
    residual_norm:
        Maximum absolute dimensionless residual.
    """

    temperature: Tensor
    pressure: Tensor
    incipient_composition: Tensor
    k_values: Tensor
    kind: SaturationKind
    iterations: int
    converged: bool
    residual_norm: Tensor


@dataclass(frozen=True)
class BinaryVLEPoint:
    """Coexisting binary liquid and vapor compositions at fixed ``T`` and ``P``.

    Attributes
    ----------
    temperature, pressure:
        Specified temperature in K and pressure in Pa.
    liquid_composition, vapor_composition:
        Two-component normalized mole-fraction vectors.
    iterations:
        Number of nonlinear iterations performed.
    converged:
        Whether fugacity equality converged and the phases remain separated by
        the requested minimum distance.
    residual_norm:
        Maximum absolute dimensionless equilibrium residual.
    """

    temperature: Tensor
    pressure: Tensor
    liquid_composition: Tensor
    vapor_composition: Tensor
    iterations: int
    converged: bool
    residual_norm: Tensor


@dataclass(frozen=True)
class BinaryBubblePoint:
    """Binary bubble pressure and incipient vapor at specified ``T, x``.

    Attributes
    ----------
    temperature:
        Specified temperature in K.
    pressure:
        Solved bubble pressure in Pa.
    liquid_composition:
        Specified normalized binary liquid composition.
    vapor_composition:
        Solved normalized incipient-vapor composition.
    iterations, converged, residual_norm:
        Nonlinear iteration count, convergence status, and maximum absolute
        dimensionless residual.
    """

    temperature: Tensor
    pressure: Tensor
    liquid_composition: Tensor
    vapor_composition: Tensor
    iterations: int
    converged: bool
    residual_norm: Tensor


@dataclass(frozen=True)
class BinaryHelmholtzBubblePoint(BinaryBubblePoint):
    """Binary bubble point solved with explicit coexisting phase volumes.

    The two molar volumes are retained as continuation variables. Reusing
    them at the next state avoids the nested liquid and vapor density
    inversions required by the pressure-based formulation.

    Attributes
    ----------
    temperature, pressure
        Specified temperature in K and solved bubble pressure in Pa.
    liquid_composition, vapor_composition
        Specified binary liquid composition and solved incipient-vapor
        composition.
    iterations, converged, residual_norm
        Nonlinear iteration count, convergence status, and maximum absolute
        dimensionless residual.
    liquid_molar_volume, vapor_molar_volume
        Coexisting molar volumes in m3/mol retained for continuation.
    """

    liquid_molar_volume: Tensor
    vapor_molar_volume: Tensor


@dataclass(frozen=True)
class BinaryPhaseEquilibriumPoint:
    """Coexisting binary compositions for a requested pair of phase roots.

    Attributes
    ----------
    temperature, pressure:
        Specified temperature in K and pressure in Pa.
    phase1_composition, phase2_composition:
        Solved normalized binary phase compositions.
    phase_kinds:
        Algebraic root requested for each phase.
    iterations, converged, residual_norm:
        Nonlinear iteration count, convergence/separation status, and maximum
        absolute dimensionless residual.
    """

    temperature: Tensor
    pressure: Tensor
    phase1_composition: Tensor
    phase2_composition: Tensor
    phase_kinds: tuple[PhaseKind, PhaseKind]
    iterations: int
    converged: bool
    residual_norm: Tensor


@dataclass(frozen=True)
class BinaryCriticalPoint:
    """Binary-mixture critical point at a specified overall composition.

    Attributes
    ----------
    temperature, pressure:
        Solved critical temperature in K and pressure in Pa.
    molar_volume:
        Solved critical molar volume in m3/mol.
    composition:
        Specified normalized binary composition.
    iterations:
        Number of nonlinear iterations performed.
    converged:
        Whether the criticality residual met tolerance.
    residual_norm:
        Maximum absolute scaled criticality residual.
    """

    temperature: Tensor
    pressure: Tensor
    molar_volume: Tensor
    composition: Tensor
    iterations: int
    converged: bool
    residual_norm: Tensor


def _components_from_model(model: StateModel) -> ComponentSet:
    required = ("critical_temperature", "critical_pressure", "acentric_factor")
    if not all(hasattr(model, name) for name in required):
        raise ValueError("saturation calculation needs an initial pressure or critical constants")
    return cast(ComponentSet, model)


def saturation_point(
    model: StateModel,
    temperature: Tensor,
    composition: Tensor,
    kind: SaturationKind,
    *,
    initial_pressure: Tensor | None = None,
    initial_k_values: Tensor | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 40,
    substitution_iterations: int | None = None,
) -> SaturationPoint:
    """Calculate an isothermal bubble or dew point with full Newton updates.

    Parameters
    ----------
    model
        Homogeneous-state model providing liquid and vapor fugacity
        coefficients and critical constants for default initialization.
    temperature
        Scalar temperature in K.
    composition
        Feed-phase mole-fraction vector.
    kind
        ``"bubble"`` for an incipient vapor or ``"dew"`` for an incipient
        liquid.
    initial_pressure
        Optional positive pressure estimate in Pa.
    initial_k_values
        Optional positive vapor-to-liquid equilibrium-ratio vector.
    tolerance
        Maximum absolute equilibrium/closure residual.
    max_iterations
        Maximum nonlinear iterations.
    substitution_iterations
        Number of Michelsen substitution passes before Newton. ``None`` uses
        up to 20 passes.

    Returns
    -------
    SaturationPoint
        Pressure, incipient composition, K values, and explicit nonlinear
        convergence diagnostics.

    Raises
    ------
    ValueError
        If kind, initialization, component data, or iteration controls are
        invalid.

    Notes
    -----
    ``substitution_iterations=None`` applies up to 20 Michelsen successive-
    substitution steps before Newton. A continuation driver with a nearby
    converged state can reduce this count; :func:`phase_envelope` does so only
    after a two-point secant predictor is available. Both the initializer and
    Newton iterates are confined to the same finite ``ln(K)`` and pressure
    bounds.
    """
    if kind not in ("bubble", "dew"):
        raise ValueError(f"unknown saturation kind {kind!r}")
    if substitution_iterations is not None and substitution_iterations < 0:
        raise ValueError("substitution_iterations must be nonnegative")
    z = normalize_composition(composition)
    components = _components_from_model(model)
    if initial_pressure is None:
        if not bool(
            torch.isfinite(components.critical_temperature).all()
            & torch.isfinite(components.critical_pressure).all()
            & torch.isfinite(components.acentric_factor).all()
        ):
            raise ValueError(
                "saturation calculation needs finite critical constants "
                "or an explicit initial pressure"
            )
        reference_pressure = torch.ones((), dtype=z.dtype, device=z.device)
        volatility = wilson_k_values(components, temperature, reference_pressure)
        if kind == "bubble":
            initial_pressure = torch.sum(z * volatility)
        elif kind == "dew":
            initial_pressure = 1.0 / torch.sum(z / volatility)
        else:  # pragma: no cover - validated above for type narrowing
            raise AssertionError
    if not bool(torch.isfinite(initial_pressure) & (initial_pressure > 0.0)):
        raise ValueError("initial saturation pressure must be finite and positive")
    if initial_k_values is None:
        initial_k = wilson_k_values(components, temperature, initial_pressure)
    else:
        if initial_k_values.shape != z.shape:
            raise ValueError("initial saturation K-values must match composition")
        if not bool(torch.isfinite(initial_k_values).all() & (initial_k_values > 0.0).all()):
            raise ValueError("initial saturation K-values must be finite and positive")
        initial_k = initial_k_values.to(dtype=z.dtype, device=z.device)
    pressure_ceiling = 50.0 * torch.max(components.critical_pressure)
    lower = torch.cat(
        (
            torch.full_like(initial_k, -50.0),
            initial_pressure.new_tensor([0.0]),
        )
    )
    upper = torch.cat(
        (
            torch.full_like(initial_k, 50.0),
            torch.log(pressure_ceiling).reshape(1),
        )
    )

    def project(current: Tensor) -> Tensor:
        return torch.minimum(torch.maximum(current, lower), upper)

    variables = torch.cat((torch.log(initial_k), torch.log(initial_pressure).reshape(1)))
    variables = project(variables)

    def residual(current: Tensor) -> Tensor:
        log_k = current[:-1]
        pressure = torch.exp(current[-1])
        k = torch.exp(log_k)
        if kind == "bubble":
            incipient = z * k
            incipient = incipient / incipient.sum()
            log_phi_feed = model.log_fugacity_coefficients(temperature, pressure, z, "liquid")
            log_phi_incipient = model.log_fugacity_coefficients(
                temperature, pressure, incipient, "vapor"
            )
            closure = torch.sum(z * k) - 1.0
            target_log_k = log_phi_feed - log_phi_incipient
        else:
            incipient = z / k
            incipient = incipient / incipient.sum()
            log_phi_feed = model.log_fugacity_coefficients(temperature, pressure, z, "vapor")
            log_phi_incipient = model.log_fugacity_coefficients(
                temperature, pressure, incipient, "liquid"
            )
            closure = torch.sum(z / k) - 1.0
            target_log_k = log_phi_incipient - log_phi_feed
        equilibrium = log_k - target_log_k
        return torch.cat((equilibrium, closure.reshape(1)))

    # Michelsen-style successive substitution provides a physical starting
    # branch and avoids Newton's trivial K=1 solution at extreme pressure.
    # The final full Newton solve retains the fast local convergence and
    # autodifferentiable residual formulation.
    substitution_count = min(
        20 if substitution_iterations is None else substitution_iterations,
        max_iterations,
    )
    for _ in range(substitution_count):
        value = residual(variables)
        if float(value.detach().abs().max()) <= tolerance:
            break
        log_k = variables[:-1] - value[:-1]
        k = torch.exp(log_k)
        closure_sum = torch.sum(z * k) if kind == "bubble" else torch.sum(z / k)
        pressure_update = (
            variables[-1] + torch.log(closure_sum)
            if kind == "bubble"
            else variables[-1] - torch.log(closure_sum)
        )
        target = torch.cat((log_k, pressure_update.reshape(1)))
        candidate = 0.5 * variables + 0.5 * target
        if not bool(torch.isfinite(candidate).all()):
            break
        variables = project(candidate)

    result = damped_newton(
        residual,
        variables,
        tolerance=tolerance,
        max_iterations=max_iterations,
        lower_bound=lower,
        upper_bound=upper,
    )
    variables = result.solution
    k = torch.exp(variables[:-1])
    pressure = torch.exp(variables[-1])
    incipient = z * k if kind == "bubble" else z / k
    incipient = incipient / incipient.sum()
    return SaturationPoint(
        temperature,
        pressure,
        incipient,
        k,
        kind,
        result.iterations,
        result.converged,
        result.residual_norm,
    )


def phase_envelope(
    model: StateModel,
    temperatures: Tensor,
    composition: Tensor,
    *,
    kinds: tuple[SaturationKind, ...] = ("bubble", "dew"),
    accelerated: bool = True,
) -> dict[SaturationKind, tuple[SaturationPoint, ...]]:
    """Trace bubble/dew branches over a specified temperature grid.

    Parameters
    ----------
    model
        Homogeneous-state model for saturation calculations.
    temperatures
        One-dimensional temperature sequence in K, in desired continuation
        order.
    composition
        Feed mole-fraction vector.
    kinds
        Bubble and/or dew branches to trace.
    accelerated
        Use two-point secant prediction after two converged points.

    Returns
    -------
    dict
        Mapping from saturation kind to one :class:`SaturationPoint` per input
        temperature. Failed points remain present and marked non-converged.

    Notes
    -----
    After two converged points, a secant predictor extrapolates ``ln(K)`` and
    ``ln(P)`` to the next temperature. Two successive-substitution corrections
    then keep the estimate on the physical branch before Newton. This avoids
    repeating the full 20-step initializer at every dense-grid point. A
    failed accelerated point, or one that abruptly collapses toward the
    algebraic ``K=1`` solution, is retried with the original robust
    initializer. Set ``accelerated=False`` to reproduce the former previous-
    point/full-initializer algorithm for numerical comparison.
    """
    branches: dict[SaturationKind, tuple[SaturationPoint, ...]] = {}
    pressure_ceiling_log = torch.log(
        50.0 * torch.max(_components_from_model(model).critical_pressure)
    )

    def collapsed_toward_trivial(candidate: SaturationPoint, reference: Tensor) -> bool:
        reference_scale = torch.max(torch.abs(reference[:-1]))
        candidate_scale = torch.max(torch.abs(torch.log(candidate.k_values.detach())))
        return bool((reference_scale > 1.0e-3) & (candidate_scale < 0.1 * reference_scale))

    for kind in kinds:
        points: list[SaturationPoint] = []
        history: list[tuple[Tensor, Tensor]] = []
        for temperature in temperatures:
            initial_pressure: Tensor | None = None
            initial_k_values: Tensor | None = None
            predicted: Tensor | None = None
            if history:
                predicted = history[-1][1]
                if accelerated and len(history) == 2:
                    temperature_step = history[-1][0] - history[-2][0]
                    if bool(temperature_step != 0.0):
                        ratio = (temperature - history[-1][0]) / temperature_step
                        predicted = history[-1][1] + ratio * (history[-1][1] - history[-2][1])
                initial_k_values = torch.exp(torch.clamp(predicted[:-1], -50.0, 50.0))
                initial_pressure = torch.exp(
                    torch.minimum(
                        torch.clamp_min(predicted[-1], 0.0),
                        pressure_ceiling_log,
                    )
                )
            point = saturation_point(
                model,
                temperature,
                composition,
                kind,
                initial_pressure=initial_pressure,
                initial_k_values=initial_k_values,
                substitution_iterations=2 if accelerated and len(history) == 2 else None,
            )

            predictor_collapsed = (
                accelerated
                and history
                and predicted is not None
                and point.converged
                and collapsed_toward_trivial(point, predicted)
            )
            if accelerated and history and (not point.converged or predictor_collapsed):
                previous = history[-1][1]
                point = saturation_point(
                    model,
                    temperature,
                    composition,
                    kind,
                    initial_pressure=torch.exp(previous[-1]),
                    initial_k_values=torch.exp(previous[:-1]),
                )
                if point.converged and collapsed_toward_trivial(point, previous):
                    point = replace(point, converged=False)
            points.append(point)
            if point.converged:
                solution = torch.cat(
                    (
                        torch.log(point.k_values.detach()),
                        torch.log(point.pressure.detach()).reshape(1),
                    )
                )
                history.append((temperature.detach(), solution))
                history = history[-2:]
        branches[kind] = tuple(points)
    return branches


def continue_saturation_branch(
    model: StateModel,
    composition: Tensor,
    initial_point: SaturationPoint,
    target_log_k_values: Tensor,
    *,
    controlled_component: int = 0,
    tolerance: float = 1.0e-9,
    max_iterations: int = 60,
    accelerated: bool = True,
) -> tuple[SaturationPoint, ...]:
    """Continue a saturation branch using one ``ln(K_i)`` as coordinate.

    Parameters
    ----------
    model
        Homogeneous-state model for saturation fugacity coefficients.
    composition
        Feed mole-fraction vector.
    initial_point
        Converged bubble or dew point defining the initial branch.
    target_log_k_values
        One-dimensional continuation coordinates in requested traversal order.
    controlled_component
        Component index whose log K value defines the coordinate.
    tolerance
        Maximum absolute augmented-system residual.
    max_iterations
        Maximum Newton iterations per target.
    accelerated
        Use a two-point secant predictor when available.

    Returns
    -------
    tuple
        One :class:`SaturationPoint` per target coordinate, including explicit
        non-converged points.

    Raises
    ------
    ValueError
        If shapes, component index, or initial saturation kind are invalid.

    Notes
    -----
    Temperature continuation becomes singular at a cricondentherm and can
    jump to the algebraic ``K=1`` solution near a mixture critical point.
    Replacing temperature by a selected log-K value yields a square
    ``(ln K, ln P, ln T)`` system that can pass both features. Points are
    returned in the order of ``target_log_k_values`` and must not be sorted by
    temperature when plotting a retrograde branch.

    The caller controls step size through the supplied targets. A failed point
    remains in the returned sequence with ``converged=False`` so scientific
    workflows can expose, rather than silently interpolate across, a
    continuation failure.

    The default secant predictor extrapolates all continuation variables from
    the previous two converged coordinates. Set ``accelerated=False`` to use
    the former previous-point initializer for performance comparisons.
    """
    z = normalize_composition(composition)
    if z.ndim != 1:
        raise ValueError("continuation requires one composition vector")
    if initial_point.k_values.shape != z.shape:
        raise ValueError("initial saturation K values must match composition")
    if target_log_k_values.ndim != 1:
        raise ValueError("target_log_k_values must be one-dimensional")
    if controlled_component < 0 or controlled_component >= z.numel():
        raise ValueError("controlled_component is outside the component range")
    if initial_point.kind not in ("bubble", "dew"):
        raise ValueError("initial point must be a bubble or dew point")

    variables = torch.cat(
        (
            torch.log(initial_point.k_values),
            torch.log(initial_point.pressure).reshape(1),
            torch.log(initial_point.temperature).reshape(1),
        )
    )
    ncomponents = z.numel()
    components = _components_from_model(model)
    lower = torch.cat(
        (
            torch.full_like(z, -50.0),
            torch.stack(
                (
                    torch.log(variables.new_tensor(1.0e2)),
                    torch.log(0.2 * torch.min(components.critical_temperature)),
                )
            ),
        )
    )
    upper = torch.cat(
        (
            torch.full_like(z, 50.0),
            torch.stack(
                (
                    torch.log(100.0 * torch.max(components.critical_pressure)),
                    torch.log(2.0 * torch.max(components.critical_temperature)),
                )
            ),
        )
    )
    points: list[SaturationPoint] = []
    history: list[tuple[Tensor, Tensor]] = [
        (
            variables[controlled_component].detach(),
            variables.detach(),
        )
    ]
    for target in target_log_k_values:
        previous_variables = variables
        used_predictor = False
        if accelerated and len(history) == 2:
            coordinate_step = history[-1][0] - history[-2][0]
            if bool(coordinate_step != 0.0):
                ratio = (target - history[-1][0]) / coordinate_step
                variables = history[-1][1] + ratio * (history[-1][1] - history[-2][1])
                variables = torch.minimum(torch.maximum(variables, lower), upper)
                used_predictor = True

        def residual(current: Tensor, target_value: Tensor = target) -> Tensor:
            log_k = current[:ncomponents]
            pressure = torch.exp(current[-2])
            temperature = torch.exp(current[-1])
            k_values = torch.exp(log_k)
            if initial_point.kind == "bubble":
                liquid = z
                vapor = normalize_composition(z * k_values)
                closure = torch.sum(z * k_values) - 1.0
            else:
                vapor = z
                liquid = normalize_composition(z / k_values)
                closure = torch.sum(z / k_values) - 1.0
            equilibrium = (
                log_k
                - model.log_fugacity_coefficients(
                    temperature,
                    pressure,
                    liquid,
                    "liquid",
                )
                + model.log_fugacity_coefficients(
                    temperature,
                    pressure,
                    vapor,
                    "vapor",
                )
            )
            coordinate = log_k[controlled_component] - target_value
            return torch.cat(
                (
                    equilibrium,
                    closure.reshape(1),
                    coordinate.reshape(1),
                )
            )

        result = damped_newton(
            residual,
            variables,
            tolerance=tolerance,
            max_iterations=max_iterations,
            lower_bound=lower,
            upper_bound=upper,
        )
        if accelerated and used_predictor and not result.converged:
            result = damped_newton(
                residual,
                previous_variables,
                tolerance=tolerance,
                max_iterations=max_iterations,
                lower_bound=lower,
                upper_bound=upper,
            )
        variables = result.solution.detach()
        k_values = torch.exp(variables[:ncomponents])
        pressure = torch.exp(variables[-2])
        temperature = torch.exp(variables[-1])
        incipient = (
            normalize_composition(z * k_values)
            if initial_point.kind == "bubble"
            else normalize_composition(z / k_values)
        )
        points.append(
            SaturationPoint(
                temperature,
                pressure,
                incipient,
                k_values,
                initial_point.kind,
                result.iterations,
                result.converged,
                result.residual_norm,
            )
        )
        if result.converged:
            history.append((target.detach(), variables))
            history = history[-2:]
    return tuple(points)


def binary_critical_point(
    model: StateModel,
    composition: Tensor,
    *,
    initial_temperature: Tensor | None = None,
    initial_pressure: Tensor | None = None,
    tolerance: float = 1.0e-9,
    max_iterations: int = 30,
) -> BinaryCriticalPoint:
    """Solve the binary mixture criticality conditions with autodiff.

    Parameters
    ----------
    model
        Smooth homogeneous-state model providing stable-root Gibbs properties.
    composition
        Strictly interior two-component mole-fraction vector.
    initial_temperature, initial_pressure
        Optional positive scalar estimates in K and Pa. Composition-weighted
        critical constants are used when omitted.
    tolerance
        Maximum absolute scaled criticality residual.
    max_iterations
        Maximum damped-Newton iterations.

    Returns
    -------
    BinaryCriticalPoint
        Solved temperature, pressure, volume, composition, and convergence
        diagnostics.

    Raises
    ------
    ValueError
        If composition is not binary/interior or initial estimates are not
        positive scalars.

    Notes
    -----
    At fixed ``T`` and ``P``, the second and third derivatives of the reduced
    molar Gibbs energy of mixing with respect to the first mole fraction both
    vanish at a binary critical point. Newton's Jacobian therefore uses up to
    fourth-order PyTorch derivatives of the model; no hand-coded EoS
    derivative expressions are required.

    This composition-space formulation is specific to a binary mixture and a
    smooth homogeneous root. It is not a general multicomponent critical-locus
    solver.
    """
    z = normalize_composition(composition)
    if z.shape != (2,):
        raise ValueError("binary_critical_point requires two components")
    if bool((z <= 0.0).any()):
        raise ValueError("binary_critical_point requires an interior composition")
    components = _components_from_model(model)
    if initial_temperature is None:
        initial_temperature = torch.sum(z * components.critical_temperature)
    if initial_pressure is None:
        initial_pressure = torch.sum(z * components.critical_pressure)
    if initial_temperature.ndim != 0 or initial_pressure.ndim != 0:
        raise ValueError("initial critical temperature and pressure must be scalar")
    if bool((initial_temperature <= 0.0) | (initial_pressure <= 0.0)):
        raise ValueError("initial critical temperature and pressure must be positive")

    first_fraction = z[0]

    def reduced_gibbs_of_mixing(
        first: Tensor,
        temperature: Tensor,
        pressure: Tensor,
    ) -> Tensor:
        current = torch.stack((first, 1.0 - first))
        log_phi = model.log_fugacity_coefficients(
            temperature,
            pressure,
            current,
            "stable",
        )
        return torch.sum(current * (torch.log(current) + log_phi))

    def residual(log_temperature_pressure: Tensor) -> Tensor:
        temperature = torch.exp(log_temperature_pressure[0])
        pressure = torch.exp(log_temperature_pressure[1])

        def gibbs(first: Tensor) -> Tensor:
            return reduced_gibbs_of_mixing(first, temperature, pressure)

        second = torch.func.grad(torch.func.grad(gibbs))
        third = torch.func.grad(second)
        return torch.stack((second(first_fraction), third(first_fraction)))

    variables = torch.stack((torch.log(initial_temperature), torch.log(initial_pressure)))
    minimum_temperature = 0.2 * torch.min(components.critical_temperature)
    maximum_temperature = 2.0 * torch.max(components.critical_temperature)
    minimum_pressure = variables.new_tensor(1.0e3)
    maximum_pressure = 100.0 * torch.max(components.critical_pressure)
    result = damped_newton(
        residual,
        variables,
        tolerance=tolerance,
        max_iterations=max_iterations,
        lower_bound=torch.stack((torch.log(minimum_temperature), torch.log(minimum_pressure))),
        upper_bound=torch.stack((torch.log(maximum_temperature), torch.log(maximum_pressure))),
    )
    temperature = torch.exp(result.solution[0])
    pressure = torch.exp(result.solution[1])
    volume = model.molar_volume(temperature, pressure, z, "stable")
    return BinaryCriticalPoint(
        temperature,
        pressure,
        volume,
        z,
        result.iterations,
        result.converged,
        result.residual_norm,
    )


def binary_phase_equilibrium_point(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    initial_phase1_composition: Tensor,
    initial_phase2_composition: Tensor,
    *,
    phase_kinds: tuple[PhaseKind, PhaseKind] = ("stable", "stable"),
    tolerance: float = 1.0e-8,
    max_iterations: int = 30,
    minimum_phase_separation: float = 1.0e-6,
) -> BinaryPhaseEquilibriumPoint:
    """Solve binary phase coexistence at fixed ``T`` and ``P``.

    Parameters
    ----------
    model
        Homogeneous-state fugacity model.
    temperature, pressure
        Scalar temperature in K and pressure in Pa.
    initial_phase1_composition, initial_phase2_composition
        Initial two-component mole-fraction vectors.
    phase_kinds
        Root request for each coexisting phase.
    tolerance
        Maximum absolute component log-fugacity residual.
    max_iterations
        Maximum substitution/Newton iterations.
    minimum_phase_separation
        Minimum absolute mole-fraction separation needed to reject the
        homogeneous algebraic solution.

    Returns
    -------
    BinaryPhaseEquilibriumPoint
        Coexisting compositions, root requests, and convergence diagnostics.

    Raises
    ------
    ValueError
        If roots, shapes, or the separation threshold are invalid.

    Notes
    -----
    ``("stable", "stable")`` is appropriate for mutual-solubility work where
    the hydrocarbon-rich phase may switch between liquid and vapor with
    conditions. Explicit roots support LLE (``"liquid", "liquid"``) and VLE
    (``"liquid", "vapor"``). The algebraic equal-composition solution is
    rejected because it is a homogeneous state, not phase coexistence.
    """
    if minimum_phase_separation < 0.0:
        raise ValueError("minimum phase separation must be nonnegative")
    if any(kind not in ("liquid", "vapor", "stable") for kind in phase_kinds):
        raise ValueError("binary phase kinds must be 'liquid', 'vapor', or 'stable'")
    phase1_initial = normalize_composition(initial_phase1_composition)
    phase2_initial = normalize_composition(initial_phase2_composition)
    if phase1_initial.shape != (2,) or phase2_initial.shape != (2,):
        raise ValueError("binary equilibrium requires two two-component composition vectors")
    epsilon = 32.0 * torch.finfo(phase1_initial.dtype).eps

    def logit(value: Tensor) -> Tensor:
        bounded = torch.clamp(value, epsilon, 1.0 - epsilon)
        return torch.log(bounded) - torch.log1p(-bounded)

    def unpack(current: Tensor) -> tuple[Tensor, Tensor]:
        first = torch.sigmoid(current)
        return torch.stack((first[0], 1.0 - first[0])), torch.stack((first[1], 1.0 - first[1]))

    variables = torch.stack((logit(phase1_initial[0]), logit(phase2_initial[0])))

    def residual(current: Tensor) -> Tensor:
        phase1, phase2 = unpack(current)
        log_phi_phase1 = model.log_fugacity_coefficients(
            temperature,
            pressure,
            phase1,
            phase_kinds[0],
        )
        log_phi_phase2 = model.log_fugacity_coefficients(
            temperature,
            pressure,
            phase2,
            phase_kinds[1],
        )
        return torch.log(phase1) + log_phi_phase1 - torch.log(phase2) - log_phi_phase2

    substitution_iterations = min(20, max_iterations)
    for iteration in range(1, substitution_iterations + 1):
        phase1, phase2 = unpack(variables)
        value = residual(variables)
        residual_norm = value.abs().max()
        if float(residual_norm.detach()) <= tolerance:
            phase_separation = torch.max(torch.abs(phase2 - phase1))
            return BinaryPhaseEquilibriumPoint(
                temperature,
                pressure,
                phase1,
                phase2,
                phase_kinds,
                iteration,
                bool(float(phase_separation.detach()) > minimum_phase_separation),
                residual_norm,
            )
        k_values = torch.exp(value) * phase2 / phase1
        denominator = k_values[0] - k_values[1]
        if float(denominator.detach().abs()) <= 1.0e-10:
            break
        phase1_first = (1.0 - k_values[1]) / denominator
        phase2_first = k_values[0] * phase1_first
        if not bool(
            torch.isfinite(phase1_first)
            & torch.isfinite(phase2_first)
            & (phase1_first > epsilon)
            & (phase1_first < 1.0 - epsilon)
            & (phase2_first > epsilon)
            & (phase2_first < 1.0 - epsilon)
        ):
            break
        target = torch.stack((logit(phase1_first), logit(phase2_first)))
        variables = 0.5 * variables + 0.5 * target

    result = damped_newton(
        residual,
        variables,
        tolerance=tolerance,
        max_iterations=max_iterations,
        lower_bound=torch.full_like(variables, -30.0),
        upper_bound=torch.full_like(variables, 30.0),
    )
    phase1, phase2 = unpack(result.solution)
    phase_separation = torch.max(torch.abs(phase2 - phase1))
    return BinaryPhaseEquilibriumPoint(
        temperature,
        pressure,
        phase1,
        phase2,
        phase_kinds,
        result.iterations,
        result.converged and bool(float(phase_separation.detach()) > minimum_phase_separation),
        result.residual_norm,
    )


def binary_vle_point(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    initial_liquid_composition: Tensor,
    initial_vapor_composition: Tensor,
    *,
    tolerance: float = 1.0e-8,
    max_iterations: int = 30,
    minimum_phase_separation: float = 1.0e-6,
) -> BinaryVLEPoint:
    """Solve a binary liquid-vapor coexistence point at fixed ``T`` and ``P``.

    Parameters
    ----------
    model
        Homogeneous-state model used for liquid and vapor fugacity
        coefficients.
    temperature
        Scalar temperature in K.
    pressure
        Scalar pressure in Pa.
    initial_liquid_composition, initial_vapor_composition
        Initial two-component mole-fraction vectors.
    tolerance
        Maximum absolute component log-fugacity residual.
    max_iterations
        Maximum substitution/Newton iterations.
    minimum_phase_separation
        Minimum absolute mole-fraction difference required to reject the
        algebraic homogeneous solution.

    Returns
    -------
    BinaryVLEPoint
        Liquid and vapor compositions plus convergence diagnostics.

    Notes
    -----
    A result is converged only when fugacity equality passes and the two
    compositions remain separated by more than
    ``minimum_phase_separation``.
    """
    result = binary_phase_equilibrium_point(
        model,
        temperature,
        pressure,
        initial_liquid_composition,
        initial_vapor_composition,
        phase_kinds=("liquid", "vapor"),
        tolerance=tolerance,
        max_iterations=max_iterations,
        minimum_phase_separation=minimum_phase_separation,
    )
    return BinaryVLEPoint(
        result.temperature,
        result.pressure,
        result.phase1_composition,
        result.phase2_composition,
        result.iterations,
        result.converged,
        result.residual_norm,
    )


def binary_bubble_point(
    model: StateModel,
    temperature: Tensor,
    liquid_composition: Tensor,
    *,
    initial_pressure: Tensor | None = None,
    initial_vapor_composition: Tensor | None = None,
    minimum_pressure: Tensor | float | None = None,
    maximum_pressure: Tensor | float | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 30,
) -> BinaryBubblePoint:
    """Solve binary bubble pressure and vapor composition at fixed ``T, x``.

    Parameters
    ----------
    model
        Homogeneous-state fugacity model with critical constants for default
        initialization.
    temperature
        Scalar temperature in K.
    liquid_composition
        Strictly positive binary liquid mole fractions.
    initial_pressure
        Optional positive pressure estimate in Pa.
    initial_vapor_composition
        Optional strictly positive binary vapor estimate.
    minimum_pressure, maximum_pressure
        Optional positive pressure bounds in Pa.
    tolerance
        Maximum absolute component log-fugacity residual.
    max_iterations
        Maximum damped-Newton iterations.

    Returns
    -------
    BinaryBubblePoint
        Bubble pressure, liquid/vapor compositions, and convergence
        diagnostics.

    Raises
    ------
    ValueError
        If composition, initialization, or pressure bounds are invalid.

    Notes
    -----
    The two unknowns are the log pressure and the logit of the first vapor
    mole fraction.  Unlike a fixed-``T,P`` coexistence calculation, this
    formulation remains well posed at a homogeneous azeotrope where
    ``x == y``. Optional pressure bounds can exclude disconnected,
    physically irrelevant roots in highly non-ideal systems.
    """
    x = normalize_composition(liquid_composition)
    if x.shape != (2,):
        raise ValueError("binary bubble point requires one two-component liquid composition")
    if not bool(torch.isfinite(x).all() & (x > 0.0).all()):
        raise ValueError("binary bubble-point liquid composition must be finite and positive")

    components = _components_from_model(model)
    if initial_pressure is None:
        reference_pressure = torch.ones((), dtype=x.dtype, device=x.device)
        volatility = wilson_k_values(components, temperature, reference_pressure)
        initial_pressure = torch.sum(x * volatility)
    initial_pressure = initial_pressure.to(dtype=x.dtype, device=x.device)
    if initial_pressure.ndim != 0 or not bool(
        torch.isfinite(initial_pressure) & (initial_pressure > 0.0)
    ):
        raise ValueError("initial binary bubble pressure must be one finite positive scalar")

    if initial_vapor_composition is None:
        volatility = wilson_k_values(components, temperature, initial_pressure)
        initial_y = normalize_composition(x * volatility)
    else:
        initial_y = normalize_composition(
            initial_vapor_composition.to(dtype=x.dtype, device=x.device)
        )
    if initial_y.shape != (2,) or not bool(
        torch.isfinite(initial_y).all() & (initial_y > 0.0).all()
    ):
        raise ValueError("initial binary bubble vapor composition must be finite and positive")

    def pressure_bound(value: Tensor | float | None) -> Tensor | None:
        if value is None:
            return None
        return torch.as_tensor(value, dtype=x.dtype, device=x.device)

    minimum = pressure_bound(minimum_pressure)
    maximum = pressure_bound(maximum_pressure)
    for name, value in (("minimum", minimum), ("maximum", maximum)):
        if value is not None and (
            value.ndim != 0 or not bool(torch.isfinite(value) & (value > 0.0))
        ):
            raise ValueError(f"{name} binary bubble pressure must be one finite positive scalar")
    if minimum is not None and maximum is not None and not bool(minimum < maximum):
        raise ValueError("minimum binary bubble pressure must be below maximum pressure")

    epsilon = 32.0 * torch.finfo(x.dtype).eps

    def logit(value: Tensor) -> Tensor:
        bounded = torch.clamp(value, epsilon, 1.0 - epsilon)
        return torch.log(bounded) - torch.log1p(-bounded)

    variables = torch.stack((logit(initial_y[0]), torch.log(initial_pressure)))

    def residual(current: Tensor) -> Tensor:
        y1 = torch.sigmoid(current[0])
        y = torch.stack((y1, 1.0 - y1))
        pressure = torch.exp(current[1])
        return (
            torch.log(x)
            + model.log_fugacity_coefficients(temperature, pressure, x, "liquid")
            - torch.log(y)
            - model.log_fugacity_coefficients(temperature, pressure, y, "vapor")
        )

    lower_pressure_log = variables.new_tensor(-torch.inf) if minimum is None else torch.log(minimum)
    upper_pressure_log = variables.new_tensor(torch.inf) if maximum is None else torch.log(maximum)
    lower_bound = torch.stack((variables.new_tensor(-30.0), lower_pressure_log))
    upper_bound = torch.stack((variables.new_tensor(30.0), upper_pressure_log))
    result = damped_newton(
        residual,
        variables,
        tolerance=tolerance,
        max_iterations=max_iterations,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    vapor_first = torch.sigmoid(result.solution[0])
    vapor = torch.stack((vapor_first, 1.0 - vapor_first))
    return BinaryBubblePoint(
        temperature,
        torch.exp(result.solution[1]),
        x,
        vapor,
        result.iterations,
        result.converged,
        result.residual_norm,
    )


def binary_helmholtz_bubble_point(
    model: HelmholtzStateModel,
    temperature: Tensor,
    liquid_composition: Tensor,
    *,
    initial_point: BinaryHelmholtzBubblePoint | None = None,
    initial_pressure: Tensor | None = None,
    initial_vapor_composition: Tensor | None = None,
    minimum_pressure: Tensor | float | None = None,
    maximum_pressure: Tensor | float | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 30,
) -> BinaryHelmholtzBubblePoint:
    """Solve a binary bubble point using phase volumes as Newton variables.

    Parameters
    ----------
    model
        Helmholtz state model providing residual Helmholtz energy, pressure,
        molar-volume roots, and critical constants.
    temperature
        Scalar temperature in K.
    liquid_composition
        Strictly positive binary liquid mole fractions.
    initial_point
        Previously converged volume-formulation point used as the continuation
        estimate. When omitted, the pressure formulation initializes the
        first point.
    initial_pressure
        Optional positive pressure estimate in Pa for first-point
        initialization.
    initial_vapor_composition
        Optional strictly positive vapor estimate for first-point
        initialization.
    minimum_pressure, maximum_pressure
        Optional positive pressure bounds in Pa.
    tolerance
        Maximum absolute dimensionless equilibrium residual.
    max_iterations
        Maximum damped-Newton iterations.

    Returns
    -------
    BinaryHelmholtzBubblePoint
        Bubble pressure, both phase compositions and volumes, and explicit
        nonlinear convergence diagnostics.

    Notes
    -----
    With no ``initial_point``, the pressure formulation supplies a robust
    first point and its two density roots. Continuation then solves directly
    for vapor composition, liquid density, and vapor density from equality of
    both component fugacities and pressure. This is the volume-based GERG
    phase-equilibrium formulation described in the GERG-2004 monograph and
    eliminates density inversion from every outer residual evaluation.

    A failed direct solve remains explicitly non-converged. Callers tracing a
    difficult branch can retry without ``initial_point`` to invoke the robust
    pressure initializer.
    """
    pressure_function = getattr(model, "pressure", None)
    if not callable(pressure_function):
        raise TypeError("volume-formulation bubble points require a Helmholtz pressure method")

    x = normalize_composition(liquid_composition)
    if x.shape != (2,):
        raise ValueError("binary bubble point requires one two-component liquid composition")
    if not bool(torch.isfinite(x).all() & (x > 0.0).all()):
        raise ValueError("binary bubble-point liquid composition must be finite and positive")

    def pressure_bound(value: Tensor | float | None, name: str) -> Tensor | None:
        if value is None:
            return None
        bound = torch.as_tensor(value, dtype=x.dtype, device=x.device)
        if bound.ndim != 0 or not bool(torch.isfinite(bound) & (bound > 0.0)):
            raise ValueError(f"{name} binary bubble pressure must be one finite positive scalar")
        return bound

    minimum = pressure_bound(minimum_pressure, "minimum")
    maximum = pressure_bound(maximum_pressure, "maximum")
    if minimum is not None and maximum is not None and not bool(minimum < maximum):
        raise ValueError("minimum binary bubble pressure must be below maximum pressure")

    if initial_point is None:
        initialized = binary_bubble_point(
            model,
            temperature,
            x,
            initial_pressure=initial_pressure,
            initial_vapor_composition=initial_vapor_composition,
            minimum_pressure=minimum,
            maximum_pressure=maximum,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        liquid_volume = model.molar_volume(
            temperature,
            initialized.pressure,
            x,
            "liquid",
        )
        vapor_volume = model.molar_volume(
            temperature,
            initialized.pressure,
            initialized.vapor_composition,
            "vapor",
        )
        return BinaryHelmholtzBubblePoint(
            initialized.temperature,
            initialized.pressure,
            initialized.liquid_composition,
            initialized.vapor_composition,
            initialized.iterations,
            initialized.converged,
            initialized.residual_norm,
            liquid_volume,
            vapor_volume,
        )

    initial_y = normalize_composition(
        initial_point.vapor_composition.to(dtype=x.dtype, device=x.device)
    )
    initial_liquid_volume = initial_point.liquid_molar_volume.to(
        dtype=x.dtype,
        device=x.device,
    )
    initial_vapor_volume = initial_point.vapor_molar_volume.to(
        dtype=x.dtype,
        device=x.device,
    )
    if initial_y.shape != (2,) or not bool(
        torch.isfinite(initial_y).all() & (initial_y > 0.0).all()
    ):
        raise ValueError("initial binary bubble vapor composition must be finite and positive")
    if not bool(
        torch.isfinite(initial_liquid_volume)
        & torch.isfinite(initial_vapor_volume)
        & (initial_liquid_volume > 0.0)
        & (initial_vapor_volume > 0.0)
    ):
        raise ValueError("initial binary bubble molar volumes must be finite and positive")

    epsilon = 32.0 * torch.finfo(x.dtype).eps

    def logit(value: Tensor) -> Tensor:
        bounded = torch.clamp(value, epsilon, 1.0 - epsilon)
        return torch.log(bounded) - torch.log1p(-bounded)

    variables = torch.stack(
        (
            logit(initial_y[0]),
            torch.log(initial_liquid_volume.reciprocal()),
            torch.log(initial_vapor_volume.reciprocal()),
        )
    )
    pressure_scale = torch.clamp_min(
        initial_point.pressure.to(dtype=x.dtype, device=x.device).detach().abs(),
        variables.new_tensor(1.0),
    )

    def residual(current: Tensor) -> Tensor:
        vapor_first = torch.sigmoid(current[0])
        vapor = torch.stack((vapor_first, 1.0 - vapor_first))
        liquid_volume = torch.exp(-current[1])
        vapor_volume = torch.exp(-current[2])
        liquid_mu_residual: Tensor = torch.func.grad(
            lambda moles: model.residual_helmholtz_rt(
                temperature,
                liquid_volume,
                moles,
            ).sum()
        )(x)
        vapor_mu_residual: Tensor = torch.func.grad(
            lambda moles: model.residual_helmholtz_rt(
                temperature,
                vapor_volume,
                moles,
            ).sum()
        )(vapor)
        log_fugacity_difference = (
            torch.log(x / liquid_volume)
            + liquid_mu_residual
            - torch.log(vapor / vapor_volume)
            - vapor_mu_residual
        )
        liquid_pressure = pressure_function(temperature, liquid_volume, x)
        vapor_pressure = pressure_function(temperature, vapor_volume, vapor)
        pressure_difference = (liquid_pressure - vapor_pressure) / pressure_scale
        return torch.cat((log_fugacity_difference, pressure_difference.reshape(1)))

    lower_bound = torch.stack(
        (
            variables.new_tensor(-30.0),
            variables.new_tensor(-20.0),
            variables.new_tensor(-20.0),
        )
    )
    upper_bound = torch.stack(
        (
            variables.new_tensor(30.0),
            variables.new_tensor(20.0),
            variables.new_tensor(20.0),
        )
    )
    result = damped_newton(
        residual,
        variables,
        tolerance=tolerance,
        max_iterations=max_iterations,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        jacobian_refresh_interval=4,
    )
    vapor_first = torch.sigmoid(result.solution[0])
    vapor = torch.stack((vapor_first, 1.0 - vapor_first))
    liquid_volume = torch.exp(-result.solution[1])
    vapor_volume = torch.exp(-result.solution[2])
    liquid_pressure = pressure_function(temperature, liquid_volume, x)
    vapor_pressure = pressure_function(temperature, vapor_volume, vapor)
    pressure = 0.5 * (liquid_pressure + vapor_pressure)
    pressure_in_range = torch.isfinite(pressure) & (pressure > 0.0)
    if minimum is not None:
        pressure_in_range = pressure_in_range & (pressure >= minimum)
    if maximum is not None:
        pressure_in_range = pressure_in_range & (pressure <= maximum)
    return BinaryHelmholtzBubblePoint(
        temperature,
        pressure,
        x,
        vapor,
        result.iterations,
        result.converged and bool(pressure_in_range),
        result.residual_norm,
        liquid_volume,
        vapor_volume,
    )
