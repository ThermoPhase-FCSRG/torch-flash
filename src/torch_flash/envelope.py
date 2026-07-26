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
class FixedVaporRatioVLEPoint:
    """VLE compositions at fixed ``T``, ``P``, and dry-vapor component ratio.

    Attributes
    ----------
    temperature, pressure
        Specified temperature in K and pressure in Pa.
    liquid_composition, vapor_composition
        Solved normalized liquid and vapor mole-fraction vectors. Ratios among
        the non-variable vapor components equal the specified reference ratio.
    variable_vapor_component
        Component index whose vapor fraction was solved independently.
    iterations
        Number of damped-Newton iterations performed.
    converged
        Whether the fugacity residual met tolerance and the two phases remain
        physically separated.
    residual_norm
        Maximum absolute dimensionless log-fugacity residual.
    phase_separation
        Maximum absolute component mole-fraction difference between phases.
    """

    temperature: Tensor
    pressure: Tensor
    liquid_composition: Tensor
    vapor_composition: Tensor
    variable_vapor_component: int
    iterations: int
    converged: bool
    residual_norm: Tensor
    phase_separation: Tensor


@dataclass(frozen=True)
class BinaryVLEPointWithVolumes(BinaryVLEPoint):
    """Binary fixed-pressure coexistence point retaining both molar volumes.

    This result extends :class:`BinaryVLEPoint` with the liquid and vapor
    molar volumes used by a volume-based equilibrium solver. The container is
    named for its reusable thermodynamic payload; the currently producing
    operation, :func:`binary_helmholtz_vle_point`, carries the Helmholtz-model
    restriction.

    Attributes
    ----------
    temperature, pressure
        Specified temperature in K and pressure in Pa.
    liquid_composition, vapor_composition
        Solved normalized binary phase compositions.
    iterations, converged, residual_norm
        Nonlinear iteration count, physical-convergence status, and maximum
        absolute dimensionless residual.
    liquid_molar_volume, vapor_molar_volume
        Coexisting molar volumes in m3/mol retained for continuation.
    """

    liquid_molar_volume: Tensor
    vapor_molar_volume: Tensor


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
class BinaryBubblePointWithVolumes(BinaryBubblePoint):
    """Binary bubble point retaining the coexisting phase molar volumes.

    This result extends :class:`BinaryBubblePoint` with the liquid and vapor
    molar volumes used by a volume-based equilibrium solver. The payload is a
    thermodynamic result rather than an equation-of-state implementation:
    any compatible algorithm can populate it. A Helmholtz-specific public
    operation currently produces it in :func:`binary_helmholtz_bubble_point`.

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
        Coexisting molar volumes in m3/mol. Reusing them as continuation
        variables can avoid nested liquid and vapor density inversions.
    """

    liquid_molar_volume: Tensor
    vapor_molar_volume: Tensor


@dataclass(frozen=True)
class BinaryPxyIsotherm:
    """Liquid-composition continuation of a binary pressure-composition trace.

    Attributes
    ----------
    temperature
        Specified scalar temperature in K.
    pressure
        Attempted bubble pressures in Pa.
    liquid_composition, vapor_composition
        Coexisting binary composition arrays with shape ``(points, 2)``.
    iterations
        Nonlinear iteration count for every attempted point.
    converged
        Physical-convergence mask requiring the nonlinear residual and minimum
        phase separation to pass.
    residual_norm
        Maximum absolute dimensionless equilibrium residual at every point.
    phase_separation
        Maximum absolute liquid-vapor mole-fraction difference.
    """

    temperature: Tensor
    pressure: Tensor
    liquid_composition: Tensor
    vapor_composition: Tensor
    iterations: Tensor
    converged: Tensor
    residual_norm: Tensor
    phase_separation: Tensor


@dataclass(frozen=True)
class BinaryFixedCompositionBoundary:
    """Pressure boundaries of a binary two-phase region at fixed composition.

    Attributes
    ----------
    temperature
        Requested one-dimensional temperature grid in K.
    bubble_pressure, dew_pressure
        Upper and lower boundary pressures in Pa. Missing crossings are NaN;
        an upper boundary above the requested reporting limit is clipped to
        that limit.
    bubble_converged, dew_converged
        Whether each boundary was resolved from physically separated,
        residual-converged bubble points.
    bubble_above_reporting_limit
        Whether the upper boundary lies at or above the reporting limit.
    dew_below_scan
        Whether the lower crossing lies below the liquid-composition scan.
    bubble_separation
        Vapor-liquid separation in the selected volatile-component coordinate
        at the upper boundary.
    bubble_residual, dew_residual
        Conservative dimensionless residual associated with each interpolated
        boundary.
    """

    temperature: Tensor
    bubble_pressure: Tensor
    dew_pressure: Tensor
    bubble_converged: Tensor
    bubble_above_reporting_limit: Tensor
    dew_converged: Tensor
    dew_below_scan: Tensor
    bubble_separation: Tensor
    bubble_residual: Tensor
    dew_residual: Tensor


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


def fixed_vapor_ratio_vle_point(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    dry_vapor_composition: Tensor,
    variable_vapor_component: int,
    *,
    initial_liquid_composition: Tensor | None = None,
    initial_variable_vapor_fraction: Tensor | None = None,
    phase_kinds: tuple[PhaseKind, PhaseKind] = ("liquid", "vapor"),
    tolerance: float = 1.0e-8,
    max_iterations: int = 40,
    minimum_phase_separation: float = 1.0e-6,
) -> FixedVaporRatioVLEPoint:
    """Solve VLE at fixed temperature, pressure, and dry-vapor composition.

    The selected variable vapor component is excluded from
    ``dry_vapor_composition``. Its saturated vapor mole fraction and every
    liquid-phase mole fraction are solved simultaneously from component
    fugacity equality. For example, a water-saturated H2/N2 gas is represented
    by ``dry_vapor_composition=[0.75, 0.25, 0]`` and
    ``variable_vapor_component=2`` in an ``(H2, N2, H2O)`` model.

    Parameters
    ----------
    model
        Homogeneous-state model providing liquid and vapor fugacity
        coefficients.
    temperature, pressure
        Finite positive scalar temperature in K and pressure in Pa.
    dry_vapor_composition
        One-dimensional nonnegative component ratio in model order. The
        variable component entry must be exactly zero; every other modeled
        component must be present with a positive ratio. The vector is
        normalized internally.
    variable_vapor_component
        Index of the vapor component whose saturated fraction is solved.
    initial_liquid_composition
        Optional strictly positive liquid composition. The default is 0.1 mol
        % dissolved dry gas, distributed in the specified dry-gas ratio.
    initial_variable_vapor_fraction
        Optional strictly interior estimate for the variable component vapor
        fraction. A Wilson estimate is used when omitted.
    phase_kinds
        Homogeneous root requested for the liquid-rich and vapor-rich states.
    tolerance
        Positive maximum absolute dimensionless log-fugacity residual.
    max_iterations
        Positive damped-Newton iteration limit.
    minimum_phase_separation
        Nonnegative minimum of ``max(abs(y - x))`` required for physical
        convergence.

    Returns
    -------
    FixedVaporRatioVLEPoint
        Coexisting compositions, solved component index, nonlinear
        diagnostics, and phase separation.

    Raises
    ------
    ValueError
        If scalar state inputs, component ratios, initial values, phase roots,
        or solver controls are invalid.

    Notes
    -----
    The square nonlinear system enforces
    ``ln(x_i phi_i^L) = ln(y_i phi_i^V)`` for every component. Independent
    liquid log-ratios and one vapor logit keep both compositions strictly
    inside their simplices. The vapor parametrization preserves all specified
    non-variable component ratios exactly, avoiding an arbitrary overall-feed
    assumption for experiments that report a saturated gas composition but
    not the coexisting liquid composition.

    Fugacity equality and fixed-composition phase-equilibrium context follow
    Michelsen and Mollerup, *Thermodynamic Models: Fundamentals &
    Computational Aspects*, 2nd ed. (2007), chapters 8 and 12,
    ISBN 978-87-989961-3-2. Newton Jacobians are assembled by PyTorch
    automatic differentiation; no equation-of-state-specific derivatives are
    embedded here.
    """
    if temperature.ndim != 0 or not bool(torch.isfinite(temperature) & (temperature > 0.0)):
        raise ValueError("fixed-vapor-ratio VLE requires one finite positive temperature")
    if pressure.ndim != 0 or not bool(torch.isfinite(pressure) & (pressure > 0.0)):
        raise ValueError("fixed-vapor-ratio VLE requires one finite positive pressure")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("fixed-vapor-ratio VLE requires positive solver controls")
    if minimum_phase_separation < 0.0:
        raise ValueError("minimum phase separation must be nonnegative")
    if any(kind not in ("liquid", "vapor", "stable") for kind in phase_kinds):
        raise ValueError("fixed-vapor-ratio phase kinds must be 'liquid', 'vapor', or 'stable'")
    if not dry_vapor_composition.is_floating_point() or dry_vapor_composition.ndim != 1:
        raise ValueError("dry vapor composition must be a one-dimensional floating tensor")
    ncomponents = dry_vapor_composition.numel()
    if variable_vapor_component < 0 or variable_vapor_component >= ncomponents:
        raise ValueError("variable vapor component is outside the component range")
    if not bool(torch.isfinite(dry_vapor_composition).all() & (dry_vapor_composition >= 0.0).all()):
        raise ValueError("dry vapor composition must be finite and nonnegative")
    if bool(dry_vapor_composition[variable_vapor_component] != 0.0):
        raise ValueError("variable component entry in dry vapor composition must be zero")
    dry_indices = tuple(index for index in range(ncomponents) if index != variable_vapor_component)
    dry_values = torch.stack(tuple(dry_vapor_composition[index] for index in dry_indices))
    if not bool((dry_values > 0.0).all()):
        raise ValueError("every non-variable dry vapor component must have a positive ratio")
    dry_values = dry_values / dry_values.sum()
    dry_composition = torch.stack(
        tuple(
            dry_vapor_composition.new_zeros(())
            if index == variable_vapor_component
            else dry_values[dry_indices.index(index)]
            for index in range(ncomponents)
        )
    )

    if initial_liquid_composition is None:
        initial_dry_total = dry_vapor_composition.new_tensor(1.0e-3)
        initial_liquid = torch.stack(
            tuple(
                1.0 - initial_dry_total
                if index == variable_vapor_component
                else initial_dry_total * dry_composition[index]
                for index in range(ncomponents)
            )
        )
    else:
        if (
            not initial_liquid_composition.is_floating_point()
            or initial_liquid_composition.shape != dry_vapor_composition.shape
            or not bool(
                torch.isfinite(initial_liquid_composition).all()
                & (initial_liquid_composition > 0.0).all()
            )
        ):
            raise ValueError(
                "initial liquid composition must be a finite positive vector "
                "matching the dry vapor composition"
            )
        initial_liquid = normalize_composition(
            initial_liquid_composition.to(
                dtype=dry_vapor_composition.dtype,
                device=dry_vapor_composition.device,
            )
        )

    epsilon = 32.0 * torch.finfo(dry_vapor_composition.dtype).eps
    if initial_variable_vapor_fraction is None:
        components = _components_from_model(model)
        initial_k = wilson_k_values(
            components,
            temperature.to(
                dtype=dry_vapor_composition.dtype,
                device=dry_vapor_composition.device,
            ),
            pressure.to(
                dtype=dry_vapor_composition.dtype,
                device=dry_vapor_composition.device,
            ),
        )
        initial_variable_fraction = (
            initial_liquid[variable_vapor_component] * initial_k[variable_vapor_component]
        )
    else:
        if initial_variable_vapor_fraction.ndim != 0 or not bool(
            torch.isfinite(initial_variable_vapor_fraction)
            & (initial_variable_vapor_fraction > 0.0)
            & (initial_variable_vapor_fraction < 1.0)
        ):
            raise ValueError(
                "initial variable vapor fraction must be one finite scalar inside (0, 1)"
            )
        initial_variable_fraction = initial_variable_vapor_fraction.to(
            dtype=dry_vapor_composition.dtype,
            device=dry_vapor_composition.device,
        )
    initial_variable_fraction = torch.clamp(
        initial_variable_fraction,
        epsilon,
        1.0 - epsilon,
    )
    initial_liquid_ratios = torch.stack(
        tuple(
            initial_liquid[index] / initial_liquid[variable_vapor_component]
            for index in dry_indices
        )
    )
    initial_variable_logit = torch.log(initial_variable_fraction) - torch.log1p(
        -initial_variable_fraction
    )
    variables = torch.cat((torch.log(initial_liquid_ratios), initial_variable_logit.reshape(1)))

    def unpack(current: Tensor) -> tuple[Tensor, Tensor]:
        liquid_ratios = torch.exp(current[:-1])
        liquid_variable = 1.0 / (1.0 + liquid_ratios.sum())
        liquid_dry = liquid_ratios * liquid_variable
        liquid = torch.stack(
            tuple(
                liquid_variable
                if index == variable_vapor_component
                else liquid_dry[dry_indices.index(index)]
                for index in range(ncomponents)
            )
        )
        vapor_variable = torch.sigmoid(current[-1])
        vapor = (1.0 - vapor_variable) * dry_composition
        vapor = torch.stack(
            tuple(
                vapor_variable if index == variable_vapor_component else vapor[index]
                for index in range(ncomponents)
            )
        )
        return liquid, vapor

    solve_temperature = temperature.to(
        dtype=dry_vapor_composition.dtype,
        device=dry_vapor_composition.device,
    )
    solve_pressure = pressure.to(
        dtype=dry_vapor_composition.dtype,
        device=dry_vapor_composition.device,
    )

    def residual(current: Tensor) -> Tensor:
        liquid, vapor = unpack(current)
        return (
            torch.log(liquid)
            + model.log_fugacity_coefficients(
                solve_temperature,
                solve_pressure,
                liquid,
                phase_kinds[0],
            )
            - torch.log(vapor)
            - model.log_fugacity_coefficients(
                solve_temperature,
                solve_pressure,
                vapor,
                phase_kinds[1],
            )
        )

    result = damped_newton(
        residual,
        variables,
        tolerance=tolerance,
        max_iterations=max_iterations,
        lower_bound=torch.full_like(variables, -50.0),
        upper_bound=torch.full_like(variables, 50.0),
    )
    liquid, vapor = unpack(result.solution)
    phase_separation = torch.max(torch.abs(vapor - liquid))
    return FixedVaporRatioVLEPoint(
        solve_temperature,
        solve_pressure,
        liquid,
        vapor,
        variable_vapor_component,
        result.iterations,
        result.converged and bool(float(phase_separation.detach()) > minimum_phase_separation),
        result.residual_norm,
        phase_separation,
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


def binary_helmholtz_vle_point(
    model: HelmholtzStateModel,
    temperature: Tensor,
    pressure: Tensor,
    initial_point: BinaryBubblePointWithVolumes | BinaryVLEPointWithVolumes,
    *,
    tolerance: float = 1.0e-8,
    max_iterations: int = 30,
    minimum_phase_separation: float = 1.0e-6,
) -> BinaryVLEPointWithVolumes:
    """Solve binary fixed-pressure coexistence using explicit phase volumes.

    Parameters
    ----------
    model
        Two-component Helmholtz state model providing residual Helmholtz
        energy and pressure as functions of temperature, molar volume, and
        composition.
    temperature
        One finite positive scalar temperature in K.
    pressure
        One finite positive scalar coexistence pressure in Pa.
    initial_point
        Nearby coexistence point containing strictly positive liquid/vapor
        compositions and molar volumes. It can be a bubble point or another
        fixed-pressure point from the same branch.
    tolerance
        Positive maximum absolute dimensionless residual for both component
        chemical-potential equalities and the two pressure equations.
    max_iterations
        Positive maximum number of damped-Newton iterations.
    minimum_phase_separation
        Nonnegative minimum ``max(abs(y - x))`` required for physical
        convergence.

    Returns
    -------
    BinaryVLEPointWithVolumes
        Specified state, solved coexisting compositions and molar volumes, and
        explicit convergence diagnostics. Floating outputs preserve the
        initial-point dtype and device.

    Raises
    ------
    TypeError
        If ``model`` does not expose the Helmholtz pressure operation.
    ValueError
        If the specified state, initial compositions or volumes, tolerance,
        iteration count, or separation threshold is invalid.

    Notes
    -----
    The four Newton variables are the liquid and vapor composition logits and
    the logarithms of both molar densities. The residual contains equality of
    the two component chemical potentials and equality of each phase pressure
    to the specified pressure. This is the volume-based phase-equilibrium
    formulation described by Kunz, Klimeck, Wagner, and Jaeschke, *The
    GERG-2004 Wide-Range Equation of State for Natural Gases and Other
    Mixtures*, GERG Technical Monograph 15 (2007), Sec. 5.4.3,
    ISBN 978-3-18-355706-6. Retaining both volumes removes nested density-root
    inversions from continuation residual evaluations.

    The discrete physical-separation decision is non-differentiable. The
    returned tensors otherwise preserve their PyTorch graphs.
    """
    pressure_function = getattr(model, "pressure", None)
    if not callable(pressure_function):
        raise TypeError("volume-formulation VLE points require a Helmholtz pressure method")

    x = normalize_composition(initial_point.liquid_composition)
    y = normalize_composition(initial_point.vapor_composition.to(dtype=x.dtype, device=x.device))
    specified_temperature = temperature.to(dtype=x.dtype, device=x.device)
    specified_pressure = pressure.to(dtype=x.dtype, device=x.device)
    liquid_volume = initial_point.liquid_molar_volume.to(dtype=x.dtype, device=x.device)
    vapor_volume = initial_point.vapor_molar_volume.to(dtype=x.dtype, device=x.device)
    if specified_temperature.ndim != 0 or not bool(
        torch.isfinite(specified_temperature) & (specified_temperature > 0.0)
    ):
        raise ValueError("binary Helmholtz VLE temperature must be one finite positive scalar")
    if specified_pressure.ndim != 0 or not bool(
        torch.isfinite(specified_pressure) & (specified_pressure > 0.0)
    ):
        raise ValueError("binary Helmholtz VLE pressure must be one finite positive scalar")
    if (
        x.shape != (2,)
        or y.shape != (2,)
        or not bool(
            torch.isfinite(x).all() & torch.isfinite(y).all() & (x > 0.0).all() & (y > 0.0).all()
        )
    ):
        raise ValueError("initial binary Helmholtz VLE compositions must be finite and positive")
    if not bool(
        torch.isfinite(liquid_volume)
        & torch.isfinite(vapor_volume)
        & (liquid_volume > 0.0)
        & (vapor_volume > 0.0)
    ):
        raise ValueError("initial binary Helmholtz VLE molar volumes must be finite and positive")
    if tolerance <= 0.0 or not torch.isfinite(torch.tensor(tolerance)):
        raise ValueError("binary Helmholtz VLE tolerance must be finite and positive")
    if max_iterations < 1:
        raise ValueError("binary Helmholtz VLE max_iterations must be positive")
    if minimum_phase_separation < 0.0 or not torch.isfinite(torch.tensor(minimum_phase_separation)):
        raise ValueError(
            "binary Helmholtz VLE minimum phase separation must be finite and nonnegative"
        )

    epsilon = 32.0 * torch.finfo(x.dtype).eps

    def logit(value: Tensor) -> Tensor:
        bounded = torch.clamp(value, epsilon, 1.0 - epsilon)
        return torch.log(bounded) - torch.log1p(-bounded)

    variables = torch.stack(
        (
            logit(x[0]),
            logit(y[0]),
            torch.log(liquid_volume.reciprocal()),
            torch.log(vapor_volume.reciprocal()),
        )
    )
    pressure_scale = torch.clamp_min(
        specified_pressure.detach().abs(),
        variables.new_tensor(1.0),
    )

    def residual(current: Tensor) -> Tensor:
        liquid_first = torch.sigmoid(current[0])
        vapor_first = torch.sigmoid(current[1])
        liquid = torch.stack((liquid_first, 1.0 - liquid_first))
        vapor = torch.stack((vapor_first, 1.0 - vapor_first))
        current_liquid_volume = torch.exp(-current[2])
        current_vapor_volume = torch.exp(-current[3])
        liquid_mu_residual: Tensor = torch.func.grad(
            lambda moles: model.residual_helmholtz_rt(
                specified_temperature,
                current_liquid_volume,
                moles,
            ).sum()
        )(liquid)
        vapor_mu_residual: Tensor = torch.func.grad(
            lambda moles: model.residual_helmholtz_rt(
                specified_temperature,
                current_vapor_volume,
                moles,
            ).sum()
        )(vapor)
        log_fugacity_difference = (
            torch.log(liquid / current_liquid_volume)
            + liquid_mu_residual
            - torch.log(vapor / current_vapor_volume)
            - vapor_mu_residual
        )
        liquid_pressure_residual = (
            pressure_function(
                specified_temperature,
                current_liquid_volume,
                liquid,
            )
            - specified_pressure
        ) / pressure_scale
        vapor_pressure_residual = (
            pressure_function(
                specified_temperature,
                current_vapor_volume,
                vapor,
            )
            - specified_pressure
        ) / pressure_scale
        return torch.cat(
            (
                log_fugacity_difference,
                liquid_pressure_residual.reshape(1),
                vapor_pressure_residual.reshape(1),
            )
        )

    result = damped_newton(
        residual,
        variables,
        tolerance=tolerance,
        max_iterations=max_iterations,
        lower_bound=torch.stack(
            (
                variables.new_tensor(-30.0),
                variables.new_tensor(-30.0),
                variables.new_tensor(-20.0),
                variables.new_tensor(-20.0),
            )
        ),
        upper_bound=torch.stack(
            (
                variables.new_tensor(30.0),
                variables.new_tensor(30.0),
                variables.new_tensor(20.0),
                variables.new_tensor(20.0),
            )
        ),
        jacobian_refresh_interval=4,
    )
    liquid_first = torch.sigmoid(result.solution[0])
    vapor_first = torch.sigmoid(result.solution[1])
    liquid = torch.stack((liquid_first, 1.0 - liquid_first))
    vapor = torch.stack((vapor_first, 1.0 - vapor_first))
    solved_liquid_volume = torch.exp(-result.solution[2])
    solved_vapor_volume = torch.exp(-result.solution[3])
    phase_separation = torch.max(torch.abs(vapor - liquid))
    return BinaryVLEPointWithVolumes(
        specified_temperature,
        specified_pressure,
        liquid,
        vapor,
        result.iterations,
        result.converged and bool(phase_separation > minimum_phase_separation),
        result.residual_norm,
        solved_liquid_volume,
        solved_vapor_volume,
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
    initial_point: BinaryBubblePointWithVolumes | None = None,
    initial_pressure: Tensor | None = None,
    initial_vapor_composition: Tensor | None = None,
    minimum_pressure: Tensor | float | None = None,
    maximum_pressure: Tensor | float | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 30,
) -> BinaryBubblePointWithVolumes:
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
    BinaryBubblePointWithVolumes
        Bubble pressure, both phase compositions and volumes, and explicit
        nonlinear convergence diagnostics.

    Notes
    -----
    With no ``initial_point``, the pressure formulation supplies a robust
    first point and its two density roots. Continuation then solves directly
    for vapor composition, liquid density, and vapor density from equality of
    both component fugacities and pressure. This is the volume-based GERG
    phase-equilibrium formulation described by Kunz, Klimeck, Wagner, and
    Jaeschke, *The GERG-2004 Wide-Range Equation of State for Natural Gases
    and Other Mixtures*, GERG Technical Monograph 15 (2007), Sec. 5.4.3,
    ISBN 978-3-18-355706-6. It eliminates density inversion from every outer
    residual evaluation.

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
        return BinaryBubblePointWithVolumes(
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
    return BinaryBubblePointWithVolumes(
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


def trace_binary_helmholtz_pxy_isotherm(
    model: HelmholtzStateModel,
    temperature: Tensor,
    liquid_compositions: Tensor,
    *,
    minimum_pressure: Tensor | float | None = None,
    maximum_pressure: Tensor | float | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 30,
    minimum_phase_separation: float = 1.0e-5,
    stop_on_failure: bool = True,
    composition_failure_refinement_steps: int = 0,
    continue_in_pressure_on_failure: bool = False,
    pressure_continuation_points: int = 25,
    pressure_failure_refinement_steps: int = 0,
) -> BinaryPxyIsotherm:
    """Trace a binary pressure-composition isotherm by bubble continuation.

    Each row of ``liquid_compositions`` specifies a successive liquid point.
    The first state uses the pressure-formulation initializer; later states
    reuse both coexisting phase volumes in
    :func:`binary_helmholtz_bubble_point`. This exposes a high-level ``p-x-y``
    workflow without embedding continuation logic in an application study.

    Parameters
    ----------
    model
        Two-component Helmholtz state model with pressure, molar-volume, and
        residual-Helmholtz methods.
    temperature
        One finite positive scalar temperature in K.
    liquid_compositions
        Ordered, strictly positive liquid mole fractions with shape
        ``(points, 2)``. Rows are normalized internally.
    minimum_pressure, maximum_pressure
        Optional finite positive pressure bounds in Pa. When a converged point
        reaches 99.5% of ``maximum_pressure``, tracing stops.
    tolerance
        Positive maximum absolute dimensionless equilibrium residual.
    max_iterations
        Positive maximum damped-Newton iteration count per point.
    minimum_phase_separation
        Nonnegative minimum of ``max(abs(y - x))`` required for physical
        convergence.
    stop_on_failure
        Stop after the first rejected point once a physical branch has begun.
    composition_failure_refinement_steps
        Number of bisection refinements between the last accepted liquid
        composition and the first rejected one. Refinement approaches a
        critical endpoint or a turning point in the liquid-composition
        parametrization without requiring a uniformly fine input grid.
    continue_in_pressure_on_failure
        After refinement, continue at increasing fixed pressures from the last
        accepted coexistence state to ``maximum_pressure``. This follows open
        phase boundaries through a turning point in liquid composition.
        Requires ``stop_on_failure=True`` and a finite ``maximum_pressure``.
    pressure_continuation_points
        Number of logarithmically spaced fixed-pressure solves used by the
        pressure continuation, including its final pressure.
    pressure_failure_refinement_steps
        Number of logarithmic bisection refinements between the last accepted
        and first rejected fixed-pressure states. This approaches the critical
        endpoint of a closed curve without uniformly refining the entire
        pressure path.

    Returns
    -------
    BinaryPxyIsotherm
        Attempted pressures, both phase compositions, iterations, residuals,
        separation, and a physical-convergence mask. All floating tensors
        retain the input composition dtype and device.

    Raises
    ------
    ValueError
        If temperature, compositions, tolerances, or iteration settings are
        invalid, or if pressure continuation is requested without compatible
        stopping and pressure-bound settings. Pressure-bound validation is
        otherwise delegated to the point solver.

    Notes
    -----
    The equilibrium equations and reuse of prior solutions as initial
    estimates follow Kunz, Klimeck, Wagner, and Jaeschke, *The GERG-2004
    Wide-Range Equation of State for Natural Gases and Other Mixtures*, GERG
    Technical Monograph 15 (2007), Sec. 5.4.3, ISBN 978-3-18-355706-6, and
    Michelsen and Mollerup, *Thermodynamic Models*, 2nd ed. (2007), chapter
    12, ISBN 978-87-989961-3-2.

    Liquid-composition continuation becomes singular at a fold of an open or
    closed phase boundary. The optional fallback changes the continuation
    coordinate to pressure and solves coexistence with
    :func:`binary_helmholtz_vle_point`. An open branch continues to the
    requested pressure bound; a closed branch is refined toward its critical
    endpoint after the first rejected pressure. The routine never forces a
    physically open branch to close. Bisection and the discrete
    accept/stop/coordinate-switch decisions are not differentiable, but
    accepted equilibrium tensors preserve their PyTorch graphs.
    """
    if temperature.ndim != 0 or not bool(torch.isfinite(temperature) & (temperature > 0.0)):
        raise ValueError("binary p-x-y isotherm requires one finite positive temperature")
    if not liquid_compositions.is_floating_point():
        raise ValueError("binary p-x-y liquid compositions must use a floating dtype")
    compositions = normalize_composition(
        liquid_compositions.to(dtype=liquid_compositions.dtype, device=liquid_compositions.device)
    )
    if compositions.ndim != 2 or compositions.shape[0] < 1 or compositions.shape[1] != 2:
        raise ValueError("binary p-x-y liquid compositions must have nonempty shape (points, 2)")
    if not bool(torch.isfinite(compositions).all() & (compositions > 0.0).all()):
        raise ValueError("binary p-x-y liquid compositions must be finite and positive")
    if tolerance <= 0.0 or not torch.isfinite(torch.tensor(tolerance)):
        raise ValueError("binary p-x-y tolerance must be finite and positive")
    if max_iterations < 1:
        raise ValueError("binary p-x-y max_iterations must be positive")
    if minimum_phase_separation < 0.0 or not torch.isfinite(torch.tensor(minimum_phase_separation)):
        raise ValueError("binary p-x-y minimum phase separation must be finite and nonnegative")
    if composition_failure_refinement_steps < 0:
        raise ValueError("binary p-x-y composition failure refinement steps must be nonnegative")
    if pressure_continuation_points < 1:
        raise ValueError("binary p-x-y pressure continuation points must be positive")
    if pressure_failure_refinement_steps < 0:
        raise ValueError("binary p-x-y pressure failure refinement steps must be nonnegative")
    if continue_in_pressure_on_failure and not stop_on_failure:
        raise ValueError("binary p-x-y pressure continuation requires stop_on_failure=True")
    if continue_in_pressure_on_failure and maximum_pressure is None:
        raise ValueError("binary p-x-y pressure continuation requires maximum_pressure")

    maximum = (
        None
        if maximum_pressure is None
        else torch.as_tensor(
            maximum_pressure,
            dtype=compositions.dtype,
            device=compositions.device,
        )
    )
    points: list[BinaryBubblePointWithVolumes | BinaryVLEPoint] = []
    accepted: list[Tensor] = []
    separations: list[Tensor] = []
    previous: BinaryBubblePointWithVolumes | None = None
    failed_liquid: Tensor | None = None

    def physical_diagnostics(
        point: BinaryBubblePoint | BinaryVLEPoint,
    ) -> tuple[bool, Tensor]:
        separation = torch.max(torch.abs(point.vapor_composition - point.liquid_composition))
        physical = (
            point.converged
            and bool(point.residual_norm <= tolerance)
            and bool(separation > minimum_phase_separation)
        )
        return physical, separation

    for liquid in compositions:
        point = binary_helmholtz_bubble_point(
            model,
            temperature.to(dtype=liquid.dtype, device=liquid.device),
            liquid,
            initial_point=previous,
            minimum_pressure=minimum_pressure,
            maximum_pressure=maximum_pressure,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        physical, separation = physical_diagnostics(point)
        points.append(point)
        accepted.append(liquid.new_tensor(physical, dtype=torch.bool))
        separations.append(separation)
        if physical:
            previous = point
        elif previous is not None and stop_on_failure:
            failed_liquid = liquid
            break
        if physical and maximum is not None and bool(point.pressure >= 0.995 * maximum):
            break

    if failed_liquid is not None and previous is not None:
        rejected_liquid = failed_liquid
        for _ in range(composition_failure_refinement_steps):
            candidate_liquid = normalize_composition(
                0.5 * (previous.liquid_composition + rejected_liquid)
            )
            point = binary_helmholtz_bubble_point(
                model,
                temperature.to(
                    dtype=candidate_liquid.dtype,
                    device=candidate_liquid.device,
                ),
                candidate_liquid,
                initial_point=previous,
                minimum_pressure=minimum_pressure,
                maximum_pressure=maximum_pressure,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            physical, separation = physical_diagnostics(point)
            points.append(point)
            accepted.append(candidate_liquid.new_tensor(physical, dtype=torch.bool))
            separations.append(separation)
            if physical:
                previous = point
            else:
                rejected_liquid = candidate_liquid

    if (
        continue_in_pressure_on_failure
        and failed_liquid is not None
        and previous is not None
        and maximum is not None
        and bool(previous.pressure < 0.995 * maximum)
    ):
        start_pressure = torch.minimum(previous.pressure * 1.01, maximum)
        continuation_pressures = torch.exp(
            torch.linspace(
                torch.log(start_pressure),
                torch.log(maximum),
                pressure_continuation_points,
                dtype=compositions.dtype,
                device=compositions.device,
            )
        )
        pressure_initial_point: BinaryBubblePointWithVolumes | BinaryVLEPointWithVolumes = previous
        rejected_pressure: Tensor | None = None
        for pressure in continuation_pressures:
            vle_point = binary_helmholtz_vle_point(
                model,
                temperature.to(dtype=pressure.dtype, device=pressure.device),
                pressure,
                pressure_initial_point,
                tolerance=tolerance,
                max_iterations=max_iterations,
                minimum_phase_separation=minimum_phase_separation,
            )
            physical, separation = physical_diagnostics(vle_point)
            points.append(vle_point)
            accepted.append(pressure.new_tensor(physical, dtype=torch.bool))
            separations.append(separation)
            if physical:
                pressure_initial_point = vle_point
            elif stop_on_failure:
                rejected_pressure = pressure
                break
        if rejected_pressure is not None:
            for _ in range(pressure_failure_refinement_steps):
                candidate_pressure = torch.sqrt(pressure_initial_point.pressure * rejected_pressure)
                vle_point = binary_helmholtz_vle_point(
                    model,
                    temperature.to(
                        dtype=candidate_pressure.dtype,
                        device=candidate_pressure.device,
                    ),
                    candidate_pressure,
                    pressure_initial_point,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    minimum_phase_separation=minimum_phase_separation,
                )
                physical, separation = physical_diagnostics(vle_point)
                points.append(vle_point)
                accepted.append(candidate_pressure.new_tensor(physical, dtype=torch.bool))
                separations.append(separation)
                if physical:
                    pressure_initial_point = vle_point
                else:
                    rejected_pressure = candidate_pressure

    return BinaryPxyIsotherm(
        temperature.to(dtype=compositions.dtype, device=compositions.device),
        torch.stack(tuple(point.pressure for point in points)),
        torch.stack(tuple(point.liquid_composition for point in points)),
        torch.stack(tuple(point.vapor_composition for point in points)),
        torch.tensor(
            [point.iterations for point in points],
            dtype=torch.int64,
            device=compositions.device,
        ),
        torch.stack(accepted),
        torch.stack(tuple(point.residual_norm for point in points)),
        torch.stack(separations),
    )


@dataclass(frozen=True)
class _BoundaryScanPoint:
    difference: Tensor
    log_pressure: Tensor
    residual: Tensor
    liquid_fraction: Tensor


def trace_binary_helmholtz_fixed_composition_boundary(
    model: HelmholtzStateModel,
    temperatures: Tensor,
    overall_composition: Tensor,
    *,
    volatile_component_index: int = 1,
    reporting_pressure_limit: Tensor | float,
    maximum_pressure: Tensor | float,
    minimum_pressure: Tensor | float = 1.0e3,
    minimum_volatile_liquid_fraction: float = 1.0e-5,
    transition_volatile_liquid_fraction: float = 0.05,
    lean_scan_points: int = 12,
    rich_scan_points: int = 15,
    tolerance: float = 1.0e-8,
    max_iterations: int = 25,
    minimum_phase_separation: float = 1.0e-5,
) -> BinaryFixedCompositionBoundary:
    """Trace a binary two-phase pressure band at fixed overall composition.

    At each temperature, this routine continues the bubble locus from a
    volatile-component-lean liquid toward the specified overall composition.
    The first crossing of ``y[volatile_component_index]`` through the overall
    fraction gives the lower (dew) boundary. The upper boundary is the bubble
    state at ``x == z`` or, beyond a critical composition, the return crossing
    of the vapor branch. Previous-temperature solutions initialize matching
    liquid-composition points.

    Parameters
    ----------
    model
        Two-component Helmholtz state model.
    temperatures
        Nonempty one-dimensional ordered temperature grid in K. Either
        increasing or decreasing order is accepted.
    overall_composition
        Strictly positive binary overall mole fractions.
    volatile_component_index
        Component coordinate expected to be enriched in the vapor phase,
        either ``0`` or ``1``.
    reporting_pressure_limit
        Positive reporting limit in Pa. Upper boundaries beyond it are clipped
        and identified by ``bubble_above_reporting_limit``.
    maximum_pressure
        Positive numerical bubble-solver ceiling in Pa, strictly greater than
        or equal to ``reporting_pressure_limit``.
    minimum_pressure
        Positive numerical bubble-solver floor in Pa.
    minimum_volatile_liquid_fraction
        Positive starting mole fraction for the volatile-component scan.
    transition_volatile_liquid_fraction
        Positive fraction separating geometric lean-side spacing from linear
        spacing toward the overall composition.
    lean_scan_points, rich_scan_points
        Number of geometric and linear scan points. Each must be at least two.
    tolerance
        Positive maximum absolute dimensionless equilibrium residual.
    max_iterations
        Positive maximum damped-Newton iterations per bubble point.
    minimum_phase_separation
        Positive minimum ``y_volatile - x_volatile`` for an accepted state.

    Returns
    -------
    BinaryFixedCompositionBoundary
        Boundary pressures in Pa plus convergence, out-of-range, separation,
        and residual diagnostics on the requested temperature grid. Floating
        outputs preserve the input dtype and device.

    Raises
    ------
    ValueError
        If state grids, compositions, pressure limits, scan settings, or
        nonlinear tolerances are invalid.

    Notes
    -----
    This is a fixed-composition specialization of bubble-locus continuation
    and crossing interpolation. The phase-equilibrium equations,
    Newton-Raphson solution, and continuation initial estimates follow Kunz,
    Klimeck, Wagner, and Jaeschke, *The GERG-2004 Wide-Range Equation of State
    for Natural Gases and Other Mixtures*, GERG Technical Monograph 15 (2007),
    Sec. 5.4.3 and Table 5.3, ISBN 978-3-18-355706-6. Log-pressure
    interpolation and pressure-limit clipping are numerical continuation
    choices made explicitly by `torch-flash`, not additional equilibrium
    equations. Discrete branch and clipping decisions are non-differentiable.
    """
    if not temperatures.is_floating_point() or temperatures.ndim != 1 or temperatures.numel() < 1:
        raise ValueError("binary fixed-composition temperatures must be a nonempty float vector")
    if not bool(torch.isfinite(temperatures).all() & (temperatures > 0.0).all()):
        raise ValueError("binary fixed-composition temperatures must be finite and positive")
    if not overall_composition.is_floating_point():
        raise ValueError("binary fixed-composition overall composition must use a floating dtype")
    composition = normalize_composition(
        overall_composition.to(dtype=temperatures.dtype, device=temperatures.device)
    )
    if composition.shape != (2,) or not bool(
        torch.isfinite(composition).all() & (composition > 0.0).all()
    ):
        raise ValueError(
            "binary fixed-composition overall composition must contain two finite positive values"
        )
    if volatile_component_index not in (0, 1):
        raise ValueError("binary fixed-composition volatile_component_index must be 0 or 1")
    if lean_scan_points < 2 or rich_scan_points < 2:
        raise ValueError("binary fixed-composition scan point counts must each be at least two")
    if max_iterations < 1:
        raise ValueError("binary fixed-composition max_iterations must be positive")
    scalar_settings = (
        ("minimum_volatile_liquid_fraction", minimum_volatile_liquid_fraction),
        (
            "transition_volatile_liquid_fraction",
            transition_volatile_liquid_fraction,
        ),
        ("tolerance", tolerance),
        ("minimum_phase_separation", minimum_phase_separation),
    )
    for name, value in scalar_settings:
        if value <= 0.0 or not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"binary fixed-composition {name} must be finite and positive")

    lower_pressure = torch.as_tensor(
        minimum_pressure,
        dtype=temperatures.dtype,
        device=temperatures.device,
    )
    reported_limit = torch.as_tensor(
        reporting_pressure_limit,
        dtype=temperatures.dtype,
        device=temperatures.device,
    )
    solver_limit = torch.as_tensor(
        maximum_pressure,
        dtype=temperatures.dtype,
        device=temperatures.device,
    )
    if any(
        value.ndim != 0 or not bool(torch.isfinite(value) & (value > 0.0))
        for value in (lower_pressure, reported_limit, solver_limit)
    ):
        raise ValueError("binary fixed-composition pressure limits must be finite positive scalars")
    if not bool(lower_pressure < reported_limit <= solver_limit):
        raise ValueError(
            "binary fixed-composition pressures require minimum < reporting <= maximum"
        )

    target_fraction = composition[volatile_component_index]
    if not bool(minimum_volatile_liquid_fraction < target_fraction):
        raise ValueError(
            "binary fixed-composition minimum volatile liquid fraction must be below "
            "the overall fraction"
        )
    transition = torch.minimum(
        target_fraction,
        target_fraction.new_tensor(transition_volatile_liquid_fraction),
    )
    transition = torch.maximum(
        transition,
        target_fraction.new_tensor(minimum_volatile_liquid_fraction),
    )
    lean_fractions = torch.exp(
        torch.linspace(
            torch.log(target_fraction.new_tensor(minimum_volatile_liquid_fraction)),
            torch.log(transition),
            lean_scan_points,
            dtype=temperatures.dtype,
            device=temperatures.device,
        )
    )
    if bool(target_fraction > transition):
        rich_fractions = torch.linspace(
            transition,
            target_fraction,
            rich_scan_points,
            dtype=temperatures.dtype,
            device=temperatures.device,
        )[1:]
        scan_fractions = torch.cat((lean_fractions, rich_fractions))
    else:
        scan_fractions = lean_fractions
        scan_fractions = torch.cat((scan_fractions[:-1], target_fraction.reshape(1)))

    temperature_points: list[BinaryBubblePointWithVolumes | None] = [None] * len(scan_fractions)
    branch_above_limit = False
    bubble_history: list[tuple[Tensor, Tensor]] = []
    output_rows: list[tuple[Tensor, Tensor, bool, bool, bool, bool, Tensor, Tensor, Tensor]] = []

    for temperature in temperatures:
        if len(bubble_history) == 2:
            previous_temperature, previous_log_pressure = bubble_history[-1]
            earlier_temperature, earlier_log_pressure = bubble_history[-2]
            temperature_step = previous_temperature - earlier_temperature
            if bool(temperature_step != 0.0):
                ratio = (temperature - previous_temperature) / temperature_step
                predicted_log_pressure = previous_log_pressure + ratio * (
                    previous_log_pressure - earlier_log_pressure
                )
                branch_above_limit = branch_above_limit or bool(
                    predicted_log_pressure >= torch.log(reported_limit)
                )

        previous_scan: _BoundaryScanPoint | None = None
        scan_point: BinaryBubblePointWithVolumes | None = None
        next_temperature_points: list[BinaryBubblePointWithVolumes | None] = [None] * len(
            scan_fractions
        )
        first_difference: Tensor | None = None
        nan = temperature.new_tensor(torch.nan)
        bubble_pressure = reported_limit if branch_above_limit else nan
        bubble_residual = nan
        bubble_separation = nan
        bubble_converged = False
        bubble_above_reporting_limit = branch_above_limit
        final_scan_pressure = nan
        dew_pressure = nan
        dew_residual = nan
        dew_converged = False

        for scan_index, liquid_fraction in enumerate(scan_fractions):
            liquid = (
                torch.stack((liquid_fraction, 1.0 - liquid_fraction))
                if volatile_component_index == 0
                else torch.stack((1.0 - liquid_fraction, liquid_fraction))
            )
            initial_point = temperature_points[scan_index]
            if initial_point is None:
                initial_point = scan_point
            scan = binary_helmholtz_bubble_point(
                model,
                temperature,
                liquid,
                initial_point=initial_point,
                minimum_pressure=lower_pressure,
                maximum_pressure=solver_limit,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            phase_separation = scan.vapor_composition[volatile_component_index] - liquid_fraction
            invalid_scan = (
                not scan.converged
                or bool(scan.residual_norm > tolerance)
                or bool(phase_separation <= minimum_phase_separation)
            )
            if (
                invalid_scan
                and temperature_points[scan_index] is not None
                and scan_point is not None
            ):
                scan = binary_helmholtz_bubble_point(
                    model,
                    temperature,
                    liquid,
                    initial_point=scan_point,
                    minimum_pressure=lower_pressure,
                    maximum_pressure=solver_limit,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
                phase_separation = (
                    scan.vapor_composition[volatile_component_index] - liquid_fraction
                )
                invalid_scan = (
                    not scan.converged
                    or bool(scan.residual_norm > tolerance)
                    or bool(phase_separation <= minimum_phase_separation)
                )
            branch_terminated = (
                invalid_scan
                and scan_point is not None
                and previous_scan is not None
                and bool(previous_scan.difference > 0.0)
                and (
                    dew_converged or (first_difference is not None and bool(first_difference > 0.0))
                )
            )
            if branch_terminated:
                break
            if invalid_scan:
                pressure_seed = scan_point if scan_point is not None else initial_point
                scan = binary_helmholtz_bubble_point(
                    model,
                    temperature,
                    liquid,
                    initial_pressure=(None if pressure_seed is None else pressure_seed.pressure),
                    initial_vapor_composition=(
                        None if pressure_seed is None else pressure_seed.vapor_composition
                    ),
                    minimum_pressure=lower_pressure,
                    maximum_pressure=solver_limit,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
                phase_separation = (
                    scan.vapor_composition[volatile_component_index] - liquid_fraction
                )
                invalid_scan = (
                    not scan.converged
                    or bool(scan.residual_norm > tolerance)
                    or bool(phase_separation <= minimum_phase_separation)
                )
            at_target = scan_index == len(scan_fractions) - 1
            if at_target:
                final_scan_pressure = scan.pressure
            if invalid_scan:
                continue

            current_scan = _BoundaryScanPoint(
                scan.vapor_composition[volatile_component_index] - target_fraction,
                torch.log(scan.pressure),
                scan.residual_norm,
                liquid_fraction,
            )
            if first_difference is None:
                first_difference = current_scan.difference
            if (
                not dew_converged
                and previous_scan is not None
                and bool(previous_scan.difference <= 0.0)
                and bool(current_scan.difference >= 0.0)
            ):
                fraction = -previous_scan.difference / (
                    current_scan.difference - previous_scan.difference
                )
                log_pressure = previous_scan.log_pressure + fraction * (
                    current_scan.log_pressure - previous_scan.log_pressure
                )
                dew_pressure = torch.exp(log_pressure)
                dew_residual = torch.maximum(
                    previous_scan.residual,
                    current_scan.residual,
                )
                dew_converged = True
            elif (
                dew_converged
                and not bubble_converged
                and previous_scan is not None
                and bool(previous_scan.difference >= 0.0)
                and bool(current_scan.difference <= 0.0)
            ):
                fraction = previous_scan.difference / (
                    previous_scan.difference - current_scan.difference
                )
                log_pressure = previous_scan.log_pressure + fraction * (
                    current_scan.log_pressure - previous_scan.log_pressure
                )
                liquid_at_boundary = previous_scan.liquid_fraction + fraction * (
                    current_scan.liquid_fraction - previous_scan.liquid_fraction
                )
                final_scan_pressure = torch.exp(log_pressure)
                bubble_pressure = final_scan_pressure
                bubble_residual = torch.maximum(
                    previous_scan.residual,
                    current_scan.residual,
                )
                bubble_separation = target_fraction - liquid_at_boundary
                bubble_converged = True
            if at_target and not bubble_converged:
                bubble_pressure = final_scan_pressure
                bubble_residual = scan.residual_norm
                bubble_separation = phase_separation
                bubble_converged = True

            previous_scan = current_scan
            scan_point = scan
            next_temperature_points[scan_index] = scan
            positive_first_difference = first_difference is not None and bool(
                first_difference > 0.0
            )
            if branch_above_limit and (dew_converged or positive_first_difference):
                break
            if bool(scan.pressure >= reported_limit) and (
                dew_converged or positive_first_difference
            ):
                bubble_pressure = reported_limit
                bubble_above_reporting_limit = True
                branch_above_limit = True
                break
            if bubble_converged and not at_target:
                break

        dew_below_scan = (
            not dew_converged and first_difference is not None and bool(first_difference > 0.0)
        )
        if bool(final_scan_pressure >= reported_limit):
            bubble_above_reporting_limit = True
            bubble_pressure = reported_limit
            branch_above_limit = True
        elif bubble_converged:
            bubble_history.append((temperature, torch.log(final_scan_pressure)))
            bubble_history = bubble_history[-2:]
        output_rows.append(
            (
                bubble_pressure,
                dew_pressure,
                bubble_converged,
                bubble_above_reporting_limit,
                dew_converged,
                dew_below_scan,
                bubble_separation,
                bubble_residual,
                dew_residual,
            )
        )
        for scan_index, next_point in enumerate(next_temperature_points):
            if next_point is not None:
                temperature_points[scan_index] = next_point

    return BinaryFixedCompositionBoundary(
        temperatures,
        torch.stack(tuple(row[0] for row in output_rows)),
        torch.stack(tuple(row[1] for row in output_rows)),
        torch.tensor(
            [row[2] for row in output_rows],
            dtype=torch.bool,
            device=temperatures.device,
        ),
        torch.tensor(
            [row[3] for row in output_rows],
            dtype=torch.bool,
            device=temperatures.device,
        ),
        torch.tensor(
            [row[4] for row in output_rows],
            dtype=torch.bool,
            device=temperatures.device,
        ),
        torch.tensor(
            [row[5] for row in output_rows],
            dtype=torch.bool,
            device=temperatures.device,
        ),
        torch.stack(tuple(row[6] for row in output_rows)),
        torch.stack(tuple(row[7] for row in output_rows)),
        torch.stack(tuple(row[8] for row in output_rows)),
    )
