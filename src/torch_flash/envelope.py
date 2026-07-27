"""Saturation points and phase-envelope tracing.

Thermodynamic formulation and continuation context follow Michelsen and
Mollerup, *Thermodynamic Models: Fundamentals & Computational Aspects*,
2nd ed. (2007), chapter 12, ISBN 978-87-989961-3-2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.exceptions import InvalidStateError
from torch_flash.flash.grid import (
    solve_batched_binary_three_phase_invariants,
    solve_binary_three_phase_invariant,
)
from torch_flash.initialization import wilson_k_values
from torch_flash.properties.state import HelmholtzStateModel, StateModel
from torch_flash.solvers import batched_damped_newton, damped_newton
from torch_flash.types import PhaseKind, normalize_composition

SaturationKind = Literal["bubble", "dew"]
PhaseBoundaryKind = Literal["liquid-liquid", "liquid-vapor", "liquid-liquid-vapor"]
ThreePhaseSolver = Literal["newton", "newton-trust-region"]


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
class PhaseTransitionPoint:
    """One locally solved incipient-phase pressure at specified ``T`` and feed.

    Attributes
    ----------
    temperature
        Specified absolute temperature in K.
    pressure
        Solved transition pressure in Pa.
    parent_composition
        Specified normalized composition of the disappearing parent phase.
    incipient_composition
        Solved normalized composition of the emerging phase.
    phase_kinds
        Algebraic EoS roots requested for the parent and incipient phases.
    iterations
        Number of damped-Newton iterations performed.
    converged
        Whether both the fugacity residual and requested phase separation
        passed.
    residual_norm
        Maximum absolute dimensionless log-fugacity residual.
    phase_separation
        Maximum absolute component mole-fraction difference between phases.
    """

    temperature: Tensor
    pressure: Tensor
    parent_composition: Tensor
    incipient_composition: Tensor
    phase_kinds: tuple[PhaseKind, PhaseKind]
    iterations: int
    converged: bool
    residual_norm: Tensor
    phase_separation: Tensor


@dataclass(frozen=True)
class PhaseTransitionBatch:
    """Independent local transition-pressure solutions in one tensor batch.

    Attributes
    ----------
    temperature
        Absolute temperatures with shape ``(batch,)`` in K.
    pressure
        Solved transition pressures with shape ``(batch,)`` in Pa.
    parent_composition, incipient_composition
        Normalized compositions with shape ``(batch, components)``.
    phase_kinds
        Common algebraic EoS roots for the parent and incipient phases.
    iterations
        Per-state damped-Newton iteration counts with shape ``(batch,)``.
    solver_converged
        Per-state fugacity-residual convergence flags.
    converged
        Per-state flags requiring both fugacity convergence and the requested
        minimum phase separation.
    residual_norm
        Per-state maximum dimensionless log-fugacity residual.
    phase_separation
        Per-state maximum absolute component-composition difference.
    """

    temperature: Tensor
    pressure: Tensor
    parent_composition: Tensor
    incipient_composition: Tensor
    phase_kinds: tuple[PhaseKind, PhaseKind]
    iterations: Tensor
    solver_converged: Tensor
    converged: Tensor
    residual_norm: Tensor
    phase_separation: Tensor


@dataclass(frozen=True)
class PhaseTransitionState:
    """One reference state for a local phase-transition calculation.

    Attributes
    ----------
    temperature
        Scalar absolute temperature in K.
    reference_pressure
        Positive scalar pressure in Pa used to initialize and, when multiple
        disconnected roots are found, identify the corresponding local branch.
    parent_composition
        Strictly positive parent/overall composition vector.
    boundary_kind
        ``"liquid-vapor"``, ``"liquid-liquid"``, or
        ``"liquid-liquid-vapor"``.
    minimum_pressure, maximum_pressure
        Optional positive scalar pressure bounds in Pa. If omitted, the
        evaluator uses one quarter and four times ``reference_pressure``,
        clipped to 0.2 and 80 MPa.
    initial_incipient_compositions
        Optional two-phase incipient-composition starts. Generic Wilson,
        uniform, and component-rich starts are used when empty.
    initial_three_phase_compositions
        Optional binary three-phase starts with shape ``(3, 2)``. Generic
        separated liquid-liquid-vapor starts are used when empty.
    """

    temperature: Tensor
    reference_pressure: Tensor
    parent_composition: Tensor
    boundary_kind: PhaseBoundaryKind
    minimum_pressure: Tensor | float | None = None
    maximum_pressure: Tensor | float | None = None
    initial_incipient_compositions: tuple[Tensor, ...] = ()
    initial_three_phase_compositions: tuple[Tensor, ...] = ()


@dataclass(frozen=True)
class PhaseTransitionEvaluation:
    """Selected physical result for one :class:`PhaseTransitionState`.

    Attributes
    ----------
    state
        Reference state and branch specification.
    pressure
        Selected calculated transition pressure in Pa, or NaN if no candidate
        produced a finite pressure.
    phase_compositions
        Selected incipient composition with shape ``(1, n)`` for a two-phase
        boundary, or all three sorted binary compositions with shape
        ``(3, 2)`` for a three-phase invariant.
    residual_norm
        Maximum absolute dimensionless log-fugacity mismatch.
    phase_separation
        Maximum two-phase component difference or minimum adjacent binary
        three-phase composition difference.
    iterations
        Nonlinear iterations reported by the selected candidate.
    solver_converged
        Whether fugacity equations converged before the phase-separation gate.
    converged
        Whether fugacity and physical phase-separation gates both passed.
    solver
        Nonlinear method that produced the selected candidate.
    """

    state: PhaseTransitionState
    pressure: Tensor
    phase_compositions: Tensor
    residual_norm: Tensor
    phase_separation: Tensor
    iterations: int
    solver_converged: bool
    converged: bool
    solver: Literal["newton", "trust-region"] = "newton"


@dataclass(frozen=True)
class PhaseEnvelopeSet:
    """Fixed-composition vapor-liquid and liquid-liquid phase boundaries.

    Attributes
    ----------
    vapor_liquid
        Bubble and dew branches on the requested vapor-liquid temperature
        sequence. Failed points remain explicit.
    liquid_liquid
        Independently continued liquid-liquid branches, each ordered by
        increasing temperature and retaining explicit failed points.
    """

    vapor_liquid: dict[SaturationKind, tuple[SaturationPoint, ...]]
    liquid_liquid: tuple[tuple[PhaseTransitionPoint, ...], ...]


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
class BinaryBubbleTemperature:
    """Binary bubble temperature and incipient vapor at specified ``P, x``.

    Attributes
    ----------
    temperature
        Solved bubble temperature in K.
    pressure
        Specified pressure in Pa.
    liquid_composition
        Specified normalized binary liquid composition.
    vapor_composition
        Solved normalized incipient-vapor composition.
    iterations, converged, residual_norm
        Nonlinear iteration count, physical-convergence status, and maximum
        absolute dimensionless log-fugacity residual.
    phase_separation
        Maximum absolute liquid-vapor mole-fraction difference. This is
        reported separately because the algebraic homogeneous solution is not
        a two-phase bubble point.
    """

    temperature: Tensor
    pressure: Tensor
    liquid_composition: Tensor
    vapor_composition: Tensor
    iterations: int
    converged: bool
    residual_norm: Tensor
    phase_separation: Tensor


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
class BinaryTxyIsobar:
    """Liquid-composition continuation of a binary temperature-composition trace.

    Attributes
    ----------
    pressure
        Specified scalar pressure in Pa.
    temperature
        Attempted bubble temperatures in K.
    liquid_composition, vapor_composition
        Coexisting binary composition arrays with shape ``(points, 2)``.
    iterations
        Nonlinear iteration count for every attempted point.
    converged
        Physical-convergence mask requiring both the nonlinear residual and
        minimum phase separation to pass.
    residual_norm
        Maximum absolute dimensionless equilibrium residual at every point.
    phase_separation
        Maximum absolute liquid-vapor mole-fraction difference.
    """

    pressure: Tensor
    temperature: Tensor
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


def phase_transition_pressure(
    model: StateModel,
    temperature: Tensor,
    parent_composition: Tensor,
    *,
    phase_kinds: tuple[PhaseKind, PhaseKind] = ("liquid", "vapor"),
    initial_pressure: Tensor | None = None,
    initial_incipient_composition: Tensor | None = None,
    minimum_pressure: Tensor | float | None = None,
    maximum_pressure: Tensor | float | None = None,
    minimum_phase_separation: float = 1.0e-6,
    tolerance: float = 1.0e-8,
    max_iterations: int = 40,
) -> PhaseTransitionPoint:
    """Solve a local incipient-phase pressure at specified ``T`` and composition.

    Parameters
    ----------
    model
        Homogeneous-state fugacity model.
    temperature
        Scalar absolute temperature in K.
    parent_composition
        Strictly positive parent-phase mole fractions. At an incipient
        transition this equals the specified overall composition.
    phase_kinds
        Algebraic EoS roots for the parent and incipient phases. Use
        ``("liquid", "vapor")`` for an L-V boundary or
        ``("liquid", "liquid")`` for an L-L boundary.
    initial_pressure
        Positive pressure estimate in Pa. It is required unless
        ``phase_kinds`` is liquid-vapor or vapor-liquid and the model exposes
        finite critical constants for Wilson initialization.
    initial_incipient_composition
        Strictly positive incipient-phase composition estimate. It is required
        when both phases request the same root because the homogeneous
        composition is also an algebraic solution.
    minimum_pressure, maximum_pressure
        Optional positive pressure bounds in Pa. Bounds are important when
        disconnected L-L branches exist.
    minimum_phase_separation
        Minimum maximum-component mole-fraction difference required to accept
        the result as two distinct phases.
    tolerance
        Maximum absolute dimensionless log-fugacity residual.
    max_iterations
        Maximum damped-Newton iterations.

    Returns
    -------
    PhaseTransitionPoint
        Pressure, incipient composition, roots, and explicit residual,
        iteration, convergence, and phase-separation diagnostics.

    Raises
    ------
    ValueError
        If a state, root, initialization, bound, or numerical control is
        invalid.

    Notes
    -----
    The formulation follows the isofugacity and incipient-phase equations in
    Michelsen and Mollerup, *Thermodynamic Models: Fundamentals &
    Computational Aspects*, 2nd ed. (2007), chapter 12,
    ISBN 978-87-989961-3-2. The first ``n-1`` unknowns are incipient-phase
    composition logits and the final unknown is logarithmic pressure.

    This is a local branch calculation, not a stability proof or automatic
    phase-topology search. Initial estimates and pressure bounds select among
    disconnected roots. A residual-converged homogeneous solution is returned
    with ``converged=False`` when it fails ``minimum_phase_separation``.
    """
    if temperature.ndim != 0 or not bool(torch.isfinite(temperature) & (temperature > 0.0)):
        raise ValueError("phase-transition temperature must be one finite positive scalar")
    if tolerance <= 0.0 or max_iterations <= 0:
        raise ValueError("phase-transition tolerance and max_iterations must be positive")
    if minimum_phase_separation < 0.0:
        raise ValueError("minimum phase separation must be nonnegative")
    if any(kind not in ("liquid", "vapor", "stable") for kind in phase_kinds):
        raise ValueError("phase kinds must be 'liquid', 'vapor', or 'stable'")

    parent = normalize_composition(parent_composition)
    if parent.ndim != 1 or parent.numel() < 2:
        raise ValueError("phase transition requires one multicomponent composition vector")
    if not bool(torch.isfinite(parent).all() & (parent > 0.0).all()):
        raise ValueError("parent-phase composition must be finite and strictly positive")

    components = _components_from_model(model)
    reference_pressure = torch.ones((), dtype=parent.dtype, device=parent.device)
    volatility: Tensor | None = None
    if phase_kinds in (("liquid", "vapor"), ("vapor", "liquid")):
        if not bool(
            torch.isfinite(components.critical_temperature).all()
            & torch.isfinite(components.critical_pressure).all()
            & torch.isfinite(components.acentric_factor).all()
        ):
            if initial_pressure is None or initial_incipient_composition is None:
                raise ValueError(
                    "phase transition needs finite critical constants or explicit "
                    "pressure and composition estimates"
                )
        else:
            volatility = wilson_k_values(components, temperature, reference_pressure)

    if initial_pressure is None:
        if volatility is None:
            raise ValueError(
                "initial pressure is required unless Wilson liquid-vapor initialization applies"
            )
        initial_pressure = (
            torch.sum(parent * volatility)
            if phase_kinds == ("liquid", "vapor")
            else 1.0 / torch.sum(parent / volatility)
        )
    initial_pressure = torch.as_tensor(
        initial_pressure,
        dtype=parent.dtype,
        device=parent.device,
    )
    if initial_pressure.ndim != 0 or not bool(
        torch.isfinite(initial_pressure) & (initial_pressure > 0.0)
    ):
        raise ValueError("initial phase-transition pressure must be one finite positive scalar")

    if initial_incipient_composition is None:
        if volatility is None:
            raise ValueError(
                "initial incipient composition is required for same-root phase transitions"
            )
        initial_incipient = normalize_composition(
            parent * volatility if phase_kinds == ("liquid", "vapor") else parent / volatility
        )
    else:
        initial_incipient = normalize_composition(
            initial_incipient_composition.to(dtype=parent.dtype, device=parent.device)
        )
    if initial_incipient.shape != parent.shape or not bool(
        torch.isfinite(initial_incipient).all() & (initial_incipient > 0.0).all()
    ):
        raise ValueError(
            "initial incipient composition must match the finite positive parent composition"
        )

    def pressure_bound(value: Tensor | float | None, name: str) -> Tensor:
        if value is None:
            return parent.new_tensor(-torch.inf if name == "minimum" else torch.inf)
        result = torch.as_tensor(value, dtype=parent.dtype, device=parent.device)
        if result.ndim != 0 or not bool(torch.isfinite(result) & (result > 0.0)):
            raise ValueError(f"{name} phase-transition pressure must be one finite positive scalar")
        return torch.log(result)

    minimum_log_pressure = pressure_bound(minimum_pressure, "minimum")
    maximum_log_pressure = pressure_bound(maximum_pressure, "maximum")
    if not bool(minimum_log_pressure < maximum_log_pressure):
        raise ValueError("minimum phase-transition pressure must be below maximum pressure")

    def composition_logits(composition: Tensor) -> Tensor:
        return torch.log(composition[:-1]) - torch.log(composition[-1])

    def composition_from_logits(logits: Tensor) -> Tensor:
        return torch.softmax(torch.cat((logits, logits.new_zeros(1))), dim=0)

    variables = torch.cat(
        (
            composition_logits(initial_incipient),
            torch.log(initial_pressure).reshape(1),
        )
    )
    lower_bound = torch.cat(
        (
            torch.full_like(variables[:-1], -30.0),
            minimum_log_pressure.reshape(1),
        )
    )
    upper_bound = torch.cat(
        (
            torch.full_like(variables[:-1], 30.0),
            maximum_log_pressure.reshape(1),
        )
    )

    def residual(current: Tensor) -> Tensor:
        incipient = composition_from_logits(current[:-1])
        pressure = torch.exp(current[-1])
        parent_log_phi = model.log_fugacity_coefficients(
            temperature,
            pressure,
            parent,
            phase_kinds[0],
        )
        incipient_log_phi = model.log_fugacity_coefficients(
            temperature,
            pressure,
            incipient,
            phase_kinds[1],
        )
        return torch.log(parent) + parent_log_phi - torch.log(incipient) - incipient_log_phi

    result = damped_newton(
        residual,
        variables,
        tolerance=tolerance,
        max_iterations=max_iterations,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    incipient = composition_from_logits(result.solution[:-1])
    separation = torch.max(torch.abs(incipient - parent))
    return PhaseTransitionPoint(
        temperature,
        torch.exp(result.solution[-1]),
        parent,
        incipient,
        phase_kinds,
        result.iterations,
        result.converged and bool(separation.detach() > minimum_phase_separation),
        result.residual_norm,
        separation,
    )


def solve_batched_phase_transition_pressures(
    model: StateModel,
    temperature: Tensor,
    parent_composition: Tensor,
    *,
    phase_kinds: tuple[PhaseKind, PhaseKind],
    initial_pressure: Tensor,
    initial_incipient_composition: Tensor,
    minimum_pressure: Tensor | float | None = None,
    maximum_pressure: Tensor | float | None = None,
    minimum_phase_separation: float = 1.0e-6,
    tolerance: float = 1.0e-8,
    max_iterations: int = 40,
) -> PhaseTransitionBatch:
    """Solve independent local transition pressures in one PyTorch batch.

    Parameters
    ----------
    model
        Homogeneous-state fugacity model shared by every state.
    temperature
        Absolute temperatures with shape ``(batch,)`` in K.
    parent_composition
        Strictly positive parent-phase mole fractions with shape
        ``(batch, components)``.
    phase_kinds
        Common algebraic roots for parent and incipient phases. Batch
        liquid-vapor and liquid-liquid states separately.
    initial_pressure
        Positive branch pressure estimates with shape ``(batch,)`` in Pa.
    initial_incipient_composition
        Strictly positive branch composition estimates with shape
        ``(batch, components)``.
    minimum_pressure, maximum_pressure
        Optional positive scalar or ``(batch,)`` pressure bounds in Pa.
    minimum_phase_separation
        Minimum maximum-component mole-fraction difference required for a
        physical two-phase result.
    tolerance
        Maximum per-state dimensionless log-fugacity residual.
    max_iterations
        Maximum batched damped-Newton iterations.

    Returns
    -------
    PhaseTransitionBatch
        Pressures, phase compositions, and explicit per-state diagnostics.

    Raises
    ------
    ValueError
        If tensor shapes, states, roots, bounds, or numerical controls are
        invalid.

    Notes
    -----
    This is the batched form of :func:`phase_transition_pressure`. It solves
    the same local isofugacity equations and preserves gradients through every
    state. All states execute together in one tensor batch; they do not share
    material balance or alter one another's branch seed.
    """
    if temperature.ndim != 1 or temperature.numel() == 0:
        raise ValueError("batched phase-transition temperature must have shape (batch,)")
    if parent_composition.ndim != 2 or parent_composition.shape[0] != temperature.shape[0]:
        raise ValueError("batched parent composition must have shape (batch, components)")
    if parent_composition.shape[1] < 2:
        raise ValueError("batched phase transitions require at least two components")
    if any(kind not in ("liquid", "vapor", "stable") for kind in phase_kinds):
        raise ValueError("phase kinds must be 'liquid', 'vapor', or 'stable'")
    if tolerance <= 0.0 or max_iterations <= 0 or minimum_phase_separation < 0.0:
        raise ValueError("batched phase-transition controls are invalid")

    parent = normalize_composition(parent_composition)
    initial_incipient = normalize_composition(
        initial_incipient_composition.to(
            dtype=parent.dtype,
            device=parent.device,
        )
    )
    temperature = temperature.to(dtype=parent.dtype, device=parent.device)
    initial_pressure = initial_pressure.to(dtype=parent.dtype, device=parent.device)
    if (
        initial_incipient.shape != parent.shape
        or initial_pressure.shape != temperature.shape
        or not bool(
            torch.isfinite(temperature).all()
            & (temperature > 0.0).all()
            & torch.isfinite(parent).all()
            & (parent > 0.0).all()
            & torch.isfinite(initial_incipient).all()
            & (initial_incipient > 0.0).all()
            & torch.isfinite(initial_pressure).all()
            & (initial_pressure > 0.0).all()
        )
    ):
        raise ValueError("batched phase-transition states must be finite and positive")

    def pressure_bound(
        value: Tensor | float | None,
        *,
        lower: bool,
    ) -> Tensor:
        if value is None:
            fill = -torch.inf if lower else torch.inf
            return parent.new_full(temperature.shape, fill)
        result = torch.as_tensor(value, dtype=parent.dtype, device=parent.device)
        try:
            result = torch.broadcast_to(result, temperature.shape)
        except RuntimeError as error:
            raise ValueError("batched transition pressure bound is not broadcastable") from error
        if not bool(torch.isfinite(result).all() & (result > 0.0).all()):
            raise ValueError("batched transition pressure bounds must be finite and positive")
        return torch.log(result)

    minimum_log_pressure = pressure_bound(minimum_pressure, lower=True)
    maximum_log_pressure = pressure_bound(maximum_pressure, lower=False)
    if not bool((minimum_log_pressure < maximum_log_pressure).all()):
        raise ValueError("minimum transition pressures must be below maximum pressures")

    initial_logits = torch.log(initial_incipient[:, :-1]) - torch.log(initial_incipient[:, -1:])
    variables = torch.cat(
        (initial_logits, torch.log(initial_pressure).unsqueeze(-1)),
        dim=-1,
    )
    lower_bound = torch.cat(
        (
            torch.full_like(initial_logits, -30.0),
            minimum_log_pressure.unsqueeze(-1),
        ),
        dim=-1,
    )
    upper_bound = torch.cat(
        (
            torch.full_like(initial_logits, 30.0),
            maximum_log_pressure.unsqueeze(-1),
        ),
        dim=-1,
    )

    def residual(
        current: Tensor,
        state_temperature: Tensor,
        state_parent: Tensor,
    ) -> Tensor:
        incipient = torch.softmax(
            torch.cat(
                (
                    current[:, :-1],
                    current.new_zeros((current.shape[0], 1)),
                ),
                dim=-1,
            ),
            dim=-1,
        )
        pressure = torch.exp(current[:, -1])
        parent_log_phi = model.log_fugacity_coefficients(
            state_temperature,
            pressure,
            state_parent,
            phase_kinds[0],
        )
        incipient_log_phi = model.log_fugacity_coefficients(
            state_temperature,
            pressure,
            incipient,
            phase_kinds[1],
        )
        return torch.log(state_parent) + parent_log_phi - torch.log(incipient) - incipient_log_phi

    result = batched_damped_newton(
        residual,
        variables,
        temperature,
        parent,
        tolerance=tolerance,
        max_iterations=max_iterations,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    incipient = torch.softmax(
        torch.cat(
            (
                result.solution[:, :-1],
                result.solution.new_zeros((result.solution.shape[0], 1)),
            ),
            dim=-1,
        ),
        dim=-1,
    )
    separation = torch.abs(incipient - parent).amax(dim=-1)
    physical = result.converged & (separation > minimum_phase_separation)
    return PhaseTransitionBatch(
        temperature,
        torch.exp(result.solution[:, -1]),
        parent,
        incipient,
        phase_kinds,
        result.iterations,
        result.converged,
        physical,
        result.residual_norm,
        separation,
    )


def _default_two_phase_starts(state: PhaseTransitionState) -> tuple[Tensor | None, ...]:
    parent = normalize_composition(state.parent_composition)
    starts: list[Tensor | None] = []
    if state.initial_incipient_compositions:
        starts.extend(state.initial_incipient_compositions)
        return tuple(starts)
    if state.boundary_kind == "liquid-vapor":
        starts.append(None)
    if state.boundary_kind == "liquid-liquid" and parent.numel() == 2:
        starts.extend(
            parent.new_tensor((fraction, 1.0 - fraction))
            for fraction in (0.01, 0.20, 0.50, 0.80, 0.99)
        )
        return tuple(starts)
    starts.append(torch.full_like(parent, 1.0 / parent.numel()))
    for component_index in range(parent.numel()):
        rich = torch.full_like(parent, 0.03 / (parent.numel() - 1))
        rich[component_index] = 0.97
        starts.append(rich)
    return tuple(starts)


def _default_three_phase_starts(state: PhaseTransitionState) -> tuple[Tensor, ...]:
    if state.initial_three_phase_compositions:
        return state.initial_three_phase_compositions
    reference = state.parent_composition
    return (
        reference.new_tensor(((0.85, 0.15), (0.99, 0.01), (0.99999, 0.00001))),
        reference.new_tensor(((0.82, 0.18), (0.98, 0.02), (0.9999, 0.0001))),
        reference.new_tensor(((0.75, 0.25), (0.98, 0.02), (0.99999, 0.00001))),
        reference.new_tensor(((0.02, 0.98), (0.70, 0.30), (0.999, 0.001))),
        reference.new_tensor(((0.10, 0.90), (0.90, 0.10), (0.995, 0.005))),
        reference.new_tensor(((0.25, 0.75), (0.75, 0.25), (0.999, 0.001))),
    )


def _trust_region_three_phase_candidates(
    model: StateModel,
    state: PhaseTransitionState,
    candidates: Sequence[PhaseTransitionEvaluation],
    *,
    tolerance: float,
    max_iterations: int,
    minimum_phase_separation: float,
) -> tuple[PhaseTransitionEvaluation, ...]:
    """Polish or recover one binary invariant with the dense trust region."""
    physical_newton = tuple(candidate for candidate in candidates if candidate.converged)
    starts = (
        (
            min(
                physical_newton,
                key=lambda candidate: abs(
                    float(torch.log(candidate.pressure.detach() / state.reference_pressure))
                ),
            ).phase_compositions,
        )
        if physical_newton
        else _default_three_phase_starts(state)
    )
    recovered = []
    for start in starts:
        try:
            result = solve_binary_three_phase_invariant(
                model,
                state.temperature,
                state.reference_pressure,
                start,
                tolerance=tolerance,
                max_iterations=max(100, max_iterations),
                method="trust-region",
            )
        except (InvalidStateError, RuntimeError, torch.linalg.LinAlgError):
            continue
        separation = torch.diff(result.phase_compositions[:, 0]).abs().min()
        physical = bool(
            result.converged
            and result.residual_norm <= tolerance
            and separation > minimum_phase_separation
        )
        recovered.append(
            PhaseTransitionEvaluation(
                state,
                result.pressure,
                result.phase_compositions,
                result.residual_norm,
                separation,
                result.iterations,
                result.converged,
                physical,
                "trust-region",
            )
        )
        if physical:
            break
    return tuple(recovered)


def _batched_trust_region_three_phase_candidates(
    model: StateModel,
    states: Sequence[PhaseTransitionState],
    state_indices: Sequence[int],
    candidate_groups: Sequence[list[PhaseTransitionEvaluation]],
    *,
    tolerance: float,
    max_iterations: int,
    minimum_phase_separation: float,
) -> None:
    """Polish or recover binary invariants with vectorized exact Hessians."""
    preferred_starts: dict[int, Tensor] = {}
    for state_index in state_indices:
        physical_newton = tuple(
            candidate for candidate in candidate_groups[state_index] if candidate.converged
        )
        if physical_newton:
            preferred_starts[state_index] = min(
                physical_newton,
                key=lambda candidate: abs(
                    float(
                        torch.log(
                            candidate.pressure.detach() / states[state_index].reference_pressure
                        )
                    )
                ),
            ).phase_compositions
        else:
            preferred_starts[state_index] = _default_three_phase_starts(states[state_index])[0]

    remaining = list(state_indices)
    maximum_recovery_starts = max(
        len(_default_three_phase_starts(states[index])) for index in state_indices
    )
    for recovery_index in range(maximum_recovery_starts + 1):
        if not remaining:
            break
        active_indices = tuple(remaining)
        starts = torch.stack(
            tuple(
                (
                    preferred_starts[index]
                    if recovery_index == 0
                    else _default_three_phase_starts(states[index])[recovery_index - 1]
                )
                for index in active_indices
            )
        )
        result = solve_batched_binary_three_phase_invariants(
            model,
            torch.stack(tuple(states[index].temperature for index in active_indices)),
            torch.stack(tuple(states[index].reference_pressure for index in active_indices)),
            starts,
            tolerance=tolerance,
            max_iterations=max(100, max_iterations),
            method="trust-region",
        )
        separations = torch.diff(result.phase_compositions[:, :, 0], dim=-1).abs().amin(dim=-1)
        recovered_indices = []
        for batch_index, state_index in enumerate(active_indices):
            physical = bool(
                result.converged[batch_index]
                and separations[batch_index] > minimum_phase_separation
            )
            candidate_groups[state_index].append(
                PhaseTransitionEvaluation(
                    states[state_index],
                    result.pressure[batch_index],
                    result.phase_compositions[batch_index],
                    result.residual_norm[batch_index],
                    separations[batch_index],
                    int(result.iterations[batch_index]),
                    bool(result.converged[batch_index]),
                    physical,
                    "trust-region",
                )
            )
            if physical:
                recovered_indices.append(state_index)
        recovered_set = set(recovered_indices)
        remaining = [index for index in remaining if index not in recovered_set]


def evaluate_phase_transition_state(
    model: StateModel,
    state: PhaseTransitionState,
    *,
    tolerance: float = 1.0e-8,
    max_iterations: int = 40,
    minimum_phase_separation: float = 2.0e-3,
    exhaustive_two_phase_starts: bool = False,
    three_phase_solver: ThreePhaseSolver = "newton",
) -> PhaseTransitionEvaluation:
    """Evaluate and select a physical branch at one reference state.

    Parameters
    ----------
    model
        Homogeneous-state fugacity model.
    state
        Temperature, reference pressure, composition, transition identity,
        optional bounds, and optional multi-start compositions.
    tolerance
        Maximum absolute dimensionless log-fugacity residual.
    max_iterations
        Maximum nonlinear iterations per candidate.
    minimum_phase_separation
        Minimum two-phase component difference or adjacent binary
        three-phase composition difference.
    exhaustive_two_phase_starts
        Evaluate every declared liquid-vapor start before selecting the
        closest physical pressure. By default, the ordered Wilson, uniform,
        and component-rich starts are lazy fallbacks: the first physical
        liquid-vapor result is returned. Liquid-liquid and three-phase
        searches remain exhaustive.
    three_phase_solver
        ``"newton"`` uses the fast local invariant solver.
        ``"newton-trust-region"`` keeps Newton as the fast path, then polishes
        its selected physical LLV invariant or invokes an exact-Hessian dense
        trust-region recovery when Newton produced no separated branch.

    Returns
    -------
    PhaseTransitionEvaluation
        First physical ordered liquid-vapor candidate unless exhaustive search
        is requested. Exhaustive searches select the candidate closest in
        logarithmic pressure to ``reference_pressure`` among those passing
        residual and phase-separation gates. If none pass, the finite candidate
        with the closest pressure and then smallest residual is returned
        explicitly as non-converged.

    Raises
    ------
    ValueError
        If the boundary identity, state tensors, numerical controls, or a
        three-phase non-binary composition are invalid.

    Notes
    -----
    Two-phase candidates solve the incipient equations documented by
    :func:`phase_transition_pressure`. Binary three-phase candidates use
    :func:`torch_flash.flash.solve_binary_three_phase_invariant`. Both follow
    the isofugacity formulation of Michelsen and Mollerup,
    *Thermodynamic Models: Fundamentals & Computational Aspects*, 2nd ed.
    (2007), chapter 12, ISBN 978-87-989961-3-2.

    The reference pressure identifies a disconnected experimental or
    application branch; it is not included as an equilibrium residual.
    Candidate selection is discrete and therefore not differentiable. The
    selected pressure and residual retain their PyTorch parameter graph.
    """
    if state.boundary_kind not in (
        "liquid-liquid",
        "liquid-vapor",
        "liquid-liquid-vapor",
    ):
        raise ValueError("unknown phase-boundary kind")
    if tolerance <= 0.0 or max_iterations <= 0 or minimum_phase_separation < 0.0:
        raise ValueError("phase-transition evaluation controls are invalid")
    if three_phase_solver not in ("newton", "newton-trust-region"):
        raise ValueError("unknown three-phase phase-transition solver")
    temperature = state.temperature
    reference_pressure = state.reference_pressure
    parent = normalize_composition(state.parent_composition)
    if temperature.ndim != 0 or not bool(torch.isfinite(temperature) & (temperature > 0.0)):
        raise ValueError("phase-transition temperature must be one finite positive scalar")
    if reference_pressure.ndim != 0 or not bool(
        torch.isfinite(reference_pressure) & (reference_pressure > 0.0)
    ):
        raise ValueError("phase-transition reference pressure must be finite and positive")
    if (
        parent.ndim != 1
        or parent.numel() < 2
        or not bool(torch.isfinite(parent).all() & (parent > 0.0).all())
    ):
        raise ValueError("phase-transition parent composition must be finite and positive")

    minimum_pressure = (
        max(0.2e6, 0.25 * float(reference_pressure.detach()))
        if state.minimum_pressure is None
        else state.minimum_pressure
    )
    maximum_pressure = (
        min(80.0e6, 4.0 * float(reference_pressure.detach()))
        if state.maximum_pressure is None
        else state.maximum_pressure
    )
    candidates: list[PhaseTransitionEvaluation] = []

    def failed(compositions: Tensor) -> PhaseTransitionEvaluation:
        return PhaseTransitionEvaluation(
            state,
            reference_pressure.new_tensor(torch.nan),
            compositions,
            reference_pressure.new_tensor(torch.inf),
            reference_pressure.new_zeros(()),
            0,
            False,
            False,
        )

    if state.boundary_kind == "liquid-liquid-vapor":
        if parent.numel() != 2:
            raise ValueError("liquid-liquid-vapor invariant evaluation requires a binary state")
        for three_phase_start in _default_three_phase_starts(state):
            try:
                invariant_result = solve_binary_three_phase_invariant(
                    model,
                    temperature,
                    reference_pressure,
                    three_phase_start,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
            except (InvalidStateError, RuntimeError, torch.linalg.LinAlgError):
                candidates.append(failed(three_phase_start))
                continue
            separation = torch.diff(invariant_result.phase_compositions[:, 0]).abs().min()
            physical = bool(
                invariant_result.converged
                and invariant_result.residual_norm <= tolerance
                and separation > minimum_phase_separation
            )
            candidates.append(
                PhaseTransitionEvaluation(
                    state,
                    invariant_result.pressure,
                    invariant_result.phase_compositions,
                    invariant_result.residual_norm,
                    separation,
                    invariant_result.iterations,
                    invariant_result.converged,
                    physical,
                )
            )
    else:
        phase_kinds: tuple[PhaseKind, PhaseKind] = (
            ("liquid", "liquid") if state.boundary_kind == "liquid-liquid" else ("liquid", "vapor")
        )
        for two_phase_start in _default_two_phase_starts(state):
            fallback_composition = parent if two_phase_start is None else two_phase_start
            try:
                transition_result = phase_transition_pressure(
                    model,
                    temperature,
                    parent,
                    phase_kinds=phase_kinds,
                    initial_pressure=reference_pressure,
                    initial_incipient_composition=two_phase_start,
                    minimum_pressure=minimum_pressure,
                    maximum_pressure=maximum_pressure,
                    minimum_phase_separation=minimum_phase_separation,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
            except (InvalidStateError, RuntimeError, torch.linalg.LinAlgError):
                candidates.append(failed(fallback_composition.reshape(1, -1)))
                continue
            physical = bool(
                transition_result.converged
                and transition_result.residual_norm <= tolerance
                and transition_result.phase_separation > minimum_phase_separation
            )
            candidate = PhaseTransitionEvaluation(
                state,
                transition_result.pressure,
                transition_result.incipient_composition.reshape(1, -1),
                transition_result.residual_norm,
                transition_result.phase_separation,
                transition_result.iterations,
                bool(transition_result.residual_norm <= tolerance),
                physical,
            )
            candidates.append(candidate)
            if (
                physical
                and state.boundary_kind == "liquid-vapor"
                and not exhaustive_two_phase_starts
            ):
                return candidate

    if state.boundary_kind == "liquid-liquid-vapor" and three_phase_solver == "newton-trust-region":
        candidates.extend(
            _trust_region_three_phase_candidates(
                model,
                state,
                candidates,
                tolerance=tolerance,
                max_iterations=max_iterations,
                minimum_phase_separation=minimum_phase_separation,
            )
        )

    def score(candidate: PhaseTransitionEvaluation) -> tuple[int, int, float, float]:
        branch_distance = (
            abs(float(torch.log(candidate.pressure.detach() / reference_pressure)))
            if bool(torch.isfinite(candidate.pressure.detach()))
            else float("inf")
        )
        return (
            0 if candidate.converged else 1,
            (
                0
                if three_phase_solver == "newton-trust-region"
                and candidate.solver == "trust-region"
                else 1
            ),
            branch_distance,
            float(candidate.residual_norm.detach()),
        )

    return min(candidates, key=score)


def evaluate_phase_transition_states(
    model: StateModel,
    states: Sequence[PhaseTransitionState],
    *,
    tolerance: float = 1.0e-8,
    max_iterations: int = 40,
    minimum_phase_separation: float = 2.0e-3,
    exhaustive_two_phase_starts: bool = False,
    three_phase_solver: ThreePhaseSolver = "newton",
) -> tuple[PhaseTransitionEvaluation, ...]:
    """Evaluate independent phase-transition reference states.

    Parameters
    ----------
    model
        Homogeneous-state model shared by all supplied states.
    states
        Independent states whose composition length must match ``model``.
    tolerance, max_iterations, minimum_phase_separation,
    exhaustive_two_phase_starts, three_phase_solver
        Controls passed to :func:`evaluate_phase_transition_state`.

    Returns
    -------
    tuple
        One explicit evaluation per input state, preserving input order.

    Notes
    -----
    States are independent and may select disconnected branches through their
    reference pressures. No material balance is imposed between states.
    """
    if tolerance <= 0.0 or max_iterations <= 0 or minimum_phase_separation < 0.0:
        raise ValueError("phase-transition evaluation controls are invalid")
    if three_phase_solver not in ("newton", "newton-trust-region"):
        raise ValueError("unknown three-phase phase-transition solver")
    if not states:
        return ()

    components = _components_from_model(model)
    candidate_groups: list[list[PhaseTransitionEvaluation]] = [[] for _ in states]
    selected: list[PhaseTransitionEvaluation | None] = [None for _ in states]

    def pressure_bounds(
        state: PhaseTransitionState,
    ) -> tuple[Tensor | float, Tensor | float]:
        minimum_pressure = (
            max(0.2e6, 0.25 * float(state.reference_pressure.detach()))
            if state.minimum_pressure is None
            else state.minimum_pressure
        )
        maximum_pressure = (
            min(80.0e6, 4.0 * float(state.reference_pressure.detach()))
            if state.maximum_pressure is None
            else state.maximum_pressure
        )
        return minimum_pressure, maximum_pressure

    def failed(
        state: PhaseTransitionState,
        compositions: Tensor,
    ) -> PhaseTransitionEvaluation:
        return PhaseTransitionEvaluation(
            state,
            state.reference_pressure.new_tensor(torch.nan),
            compositions,
            state.reference_pressure.new_tensor(torch.inf),
            state.reference_pressure.new_zeros(()),
            0,
            False,
            False,
        )

    for boundary_kind in ("liquid-liquid", "liquid-vapor"):
        state_indices = [
            index for index, state in enumerate(states) if state.boundary_kind == boundary_kind
        ]
        if not state_indices:
            continue
        two_phase_starts = {
            index: _default_two_phase_starts(states[index]) for index in state_indices
        }
        maximum_starts = max(len(state_starts) for state_starts in two_phase_starts.values())
        for start_index in range(maximum_starts):
            active_indices = [
                index
                for index in state_indices
                if start_index < len(two_phase_starts[index])
                and (
                    boundary_kind == "liquid-liquid"
                    or exhaustive_two_phase_starts
                    or selected[index] is None
                )
            ]
            if not active_indices:
                continue

            temperatures = torch.stack(tuple(states[index].temperature for index in active_indices))
            parents = torch.stack(
                tuple(states[index].parent_composition for index in active_indices)
            )
            reference_pressures = torch.stack(
                tuple(states[index].reference_pressure for index in active_indices)
            )
            incipient_starts = []
            minimum_pressures = []
            maximum_pressures = []
            for index, state_temperature, parent in zip(
                active_indices,
                temperatures,
                parents,
                strict=True,
            ):
                start = two_phase_starts[index][start_index]
                if start is None:
                    volatility = wilson_k_values(
                        components,
                        state_temperature,
                        state_temperature.new_ones(()),
                    )
                    start = normalize_composition(parent * volatility)
                incipient_starts.append(start)
                lower, upper = pressure_bounds(states[index])
                minimum_pressures.append(reference_pressures.new_tensor(lower))
                maximum_pressures.append(reference_pressures.new_tensor(upper))
            try:
                two_phase_batch_result = solve_batched_phase_transition_pressures(
                    model,
                    temperatures,
                    parents,
                    phase_kinds=(
                        ("liquid", "liquid")
                        if boundary_kind == "liquid-liquid"
                        else ("liquid", "vapor")
                    ),
                    initial_pressure=reference_pressures,
                    initial_incipient_composition=torch.stack(tuple(incipient_starts)),
                    minimum_pressure=torch.stack(tuple(minimum_pressures)),
                    maximum_pressure=torch.stack(tuple(maximum_pressures)),
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    minimum_phase_separation=minimum_phase_separation,
                )
                for batch_index, state_index in enumerate(active_indices):
                    candidate = PhaseTransitionEvaluation(
                        states[state_index],
                        two_phase_batch_result.pressure[batch_index],
                        two_phase_batch_result.incipient_composition[batch_index].reshape(1, -1),
                        two_phase_batch_result.residual_norm[batch_index],
                        two_phase_batch_result.phase_separation[batch_index],
                        int(two_phase_batch_result.iterations[batch_index]),
                        bool(two_phase_batch_result.solver_converged[batch_index]),
                        bool(two_phase_batch_result.converged[batch_index]),
                    )
                    candidate_groups[state_index].append(candidate)
                    if (
                        boundary_kind == "liquid-vapor"
                        and not exhaustive_two_phase_starts
                        and candidate.converged
                    ):
                        selected[state_index] = candidate
            except (InvalidStateError, RuntimeError, torch.linalg.LinAlgError):
                for state_index, start in zip(
                    active_indices,
                    incipient_starts,
                    strict=True,
                ):
                    candidate_groups[state_index].append(
                        failed(states[state_index], start.reshape(1, -1))
                    )

    three_phase_indices = [
        index for index, state in enumerate(states) if state.boundary_kind == "liquid-liquid-vapor"
    ]
    if three_phase_indices:
        if any(states[index].parent_composition.numel() != 2 for index in three_phase_indices):
            raise ValueError("liquid-liquid-vapor invariant evaluation requires binary states")
        three_phase_starts = {
            index: _default_three_phase_starts(states[index]) for index in three_phase_indices
        }
        maximum_starts = max(len(state_starts) for state_starts in three_phase_starts.values())
        for start_index in range(maximum_starts):
            active_indices = [
                index
                for index in three_phase_indices
                if start_index < len(three_phase_starts[index])
            ]
            temperatures = torch.stack(tuple(states[index].temperature for index in active_indices))
            reference_pressures = torch.stack(
                tuple(states[index].reference_pressure for index in active_indices)
            )
            phase_starts = torch.stack(
                tuple(three_phase_starts[index][start_index] for index in active_indices)
            )
            try:
                three_phase_batch_result = solve_batched_binary_three_phase_invariants(
                    model,
                    temperatures,
                    reference_pressures,
                    phase_starts,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
                separations = (
                    torch.diff(
                        three_phase_batch_result.phase_compositions[:, :, 0],
                        dim=-1,
                    )
                    .abs()
                    .amin(dim=-1)
                )
                for batch_index, state_index in enumerate(active_indices):
                    physical = bool(
                        three_phase_batch_result.converged[batch_index]
                        and separations[batch_index] > minimum_phase_separation
                    )
                    candidate_groups[state_index].append(
                        PhaseTransitionEvaluation(
                            states[state_index],
                            three_phase_batch_result.pressure[batch_index],
                            three_phase_batch_result.phase_compositions[batch_index],
                            three_phase_batch_result.residual_norm[batch_index],
                            separations[batch_index],
                            int(three_phase_batch_result.iterations[batch_index]),
                            bool(three_phase_batch_result.converged[batch_index]),
                            physical,
                        )
                    )
            except (InvalidStateError, RuntimeError, torch.linalg.LinAlgError):
                for state_index, start in zip(
                    active_indices,
                    phase_starts,
                    strict=True,
                ):
                    candidate_groups[state_index].append(failed(states[state_index], start))

    unknown = [
        state.boundary_kind
        for state in states
        if state.boundary_kind
        not in (
            "liquid-liquid",
            "liquid-vapor",
            "liquid-liquid-vapor",
        )
    ]
    if unknown:
        raise ValueError("unknown phase-boundary kind")

    if three_phase_solver == "newton-trust-region" and three_phase_indices:
        _batched_trust_region_three_phase_candidates(
            model,
            states,
            three_phase_indices,
            candidate_groups,
            tolerance=tolerance,
            max_iterations=max_iterations,
            minimum_phase_separation=minimum_phase_separation,
        )

    def score(
        candidate: PhaseTransitionEvaluation,
    ) -> tuple[int, int, float, float]:
        reference_pressure = candidate.state.reference_pressure
        branch_distance = (
            abs(float(torch.log(candidate.pressure.detach() / reference_pressure)))
            if bool(torch.isfinite(candidate.pressure.detach()))
            else float("inf")
        )
        return (
            0 if candidate.converged else 1,
            (
                0
                if three_phase_solver == "newton-trust-region"
                and candidate.solver == "trust-region"
                else 1
            ),
            branch_distance,
            float(candidate.residual_norm.detach()),
        )

    return tuple(
        (candidate if candidate is not None else min(candidate_groups[index], key=score))
        for index, candidate in enumerate(selected)
    )


def continue_phase_transition_branch(
    model: StateModel,
    temperatures: Tensor,
    initial_point: PhaseTransitionPoint,
    *,
    minimum_pressure: Tensor | float | None = None,
    maximum_pressure: Tensor | float | None = None,
    minimum_phase_separation: float = 1.0e-6,
    tolerance: float = 1.0e-8,
    max_iterations: int = 40,
) -> tuple[PhaseTransitionPoint, ...]:
    """Continue one incipient two-phase branch over a temperature sequence.

    Parameters
    ----------
    model
        Homogeneous-state fugacity model used for every continuation point.
    temperatures
        Nonempty one-dimensional temperature sequence in K, ordered in the
        desired continuation direction.
    initial_point
        Converged branch seed supplying the parent composition, phase-root
        pair, pressure, and incipient-phase composition.
    minimum_pressure, maximum_pressure
        Optional positive pressure bounds in Pa applied to every point.
        Bounds can isolate a disconnected liquid-liquid branch.
    minimum_phase_separation
        Minimum maximum-component mole-fraction difference required to accept
        two physically distinct phases.
    tolerance
        Maximum absolute dimensionless log-fugacity residual.
    max_iterations
        Maximum damped-Newton iterations per temperature.

    Returns
    -------
    tuple
        One :class:`PhaseTransitionPoint` per requested temperature. Failed
        points remain present with ``converged=False`` and do not replace the
        last converged continuation state.

    Raises
    ------
    ValueError
        If the temperature grid is invalid or ``initial_point`` is not a
        converged, phase-separated seed.

    Notes
    -----
    The local equilibrium equations are those of
    :func:`phase_transition_pressure`, following Michelsen and Mollerup,
    *Thermodynamic Models: Fundamentals & Computational Aspects*, 2nd ed.
    (2007), chapter 12, ISBN 978-87-989961-3-2. Previous-point continuation is
    a numerical branch-tracking choice made explicitly by ``torch-flash``.
    The caller controls direction, spacing, and pressure bounds because these
    choices can select among disconnected phase-transition branches.

    Continuation initial states are detached between points. Each returned
    solve can still carry parameter derivatives, but the discrete decision to
    retain or reject an initializer is not differentiable.
    """
    if temperatures.ndim != 1 or temperatures.numel() < 1:
        raise ValueError("phase-transition temperatures must be a nonempty vector")
    if not temperatures.is_floating_point() or not bool(
        torch.isfinite(temperatures).all() & (temperatures > 0.0).all()
    ):
        raise ValueError("phase-transition temperatures must be finite and positive")
    if not initial_point.converged or not bool(
        torch.isfinite(initial_point.pressure)
        & (initial_point.pressure > 0.0)
        & torch.isfinite(initial_point.incipient_composition).all()
        & (initial_point.phase_separation > minimum_phase_separation)
    ):
        raise ValueError("initial phase-transition point must be converged and separated")

    parent = initial_point.parent_composition.to(
        dtype=temperatures.dtype,
        device=temperatures.device,
    )
    previous_pressure = initial_point.pressure.to(
        dtype=temperatures.dtype,
        device=temperatures.device,
    ).detach()
    previous_incipient = initial_point.incipient_composition.to(
        dtype=temperatures.dtype,
        device=temperatures.device,
    ).detach()
    points: list[PhaseTransitionPoint] = []
    for temperature in temperatures:
        try:
            point = phase_transition_pressure(
                model,
                temperature,
                parent,
                phase_kinds=initial_point.phase_kinds,
                initial_pressure=previous_pressure,
                initial_incipient_composition=previous_incipient,
                minimum_pressure=minimum_pressure,
                maximum_pressure=maximum_pressure,
                minimum_phase_separation=minimum_phase_separation,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        except (InvalidStateError, RuntimeError, torch.linalg.LinAlgError):
            point = PhaseTransitionPoint(
                temperature,
                previous_pressure.new_tensor(torch.nan),
                parent,
                previous_incipient,
                initial_point.phase_kinds,
                0,
                False,
                previous_pressure.new_tensor(torch.inf),
                torch.max(torch.abs(previous_incipient - parent)),
            )
        points.append(point)
        if point.converged:
            previous_pressure = point.pressure.detach()
            previous_incipient = point.incipient_composition.detach()
    return tuple(points)


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


def trace_phase_envelope_set(
    model: StateModel,
    composition: Tensor,
    vapor_liquid_temperatures: Tensor,
    *,
    liquid_liquid_seeds: Sequence[PhaseTransitionEvaluation] = (),
    liquid_liquid_temperatures: Tensor | None = None,
    minimum_pressure: Tensor | float | None = None,
    maximum_pressure: Tensor | float | None = None,
    minimum_phase_separation: float = 2.0e-3,
    tolerance: float = 1.0e-8,
    max_iterations: int = 40,
    vapor_liquid_closure_points: int = 20,
    vapor_liquid_closure_log_k: float = 1.0e-3,
) -> PhaseEnvelopeSet:
    """Trace a complete fixed-composition fluid phase-boundary set.

    Parameters
    ----------
    model
        Homogeneous-state fugacity model.
    composition
        Fixed overall/parent mole-fraction vector.
    vapor_liquid_temperatures
        Ordered temperature grid in K for both bubble and dew branches.
    liquid_liquid_seeds
        Residual- and separation-converged liquid-liquid reference
        evaluations. If several observations converge to the same branch,
        only the lowest- and highest-pressure distinct seeds are retained.
    liquid_liquid_temperatures
        Ordered temperature grid in K for each liquid-liquid branch. Required
        when ``liquid_liquid_seeds`` is nonempty.
    minimum_pressure, maximum_pressure
        Optional liquid-liquid continuation bounds in Pa.
    minimum_phase_separation
        Minimum maximum-component composition difference for a physical
        liquid-liquid point.
    tolerance
        Maximum absolute dimensionless log-fugacity residual.
    max_iterations
        Maximum nonlinear iterations per liquid-liquid continuation point.
    vapor_liquid_closure_points
        Number of logarithmically spaced log-K continuation points used to
        approach each bubble/dew critical endpoint after fixed-temperature
        tracing.
    vapor_liquid_closure_log_k
        Positive absolute controlled-component log-K target at the final
        critical-closure point.

    Returns
    -------
    PhaseEnvelopeSet
        Bubble/dew branches and every distinct seeded liquid-liquid branch.
        Failed continuation points remain explicit.

    Raises
    ------
    ValueError
        If liquid-liquid seeds do not match the supplied composition or a
        continuation grid is missing.

    Notes
    -----
    Bubble/dew tracing starts with :func:`phase_envelope`, then switches from
    temperature to a selected log-K continuation coordinate through
    :func:`continue_saturation_branch`. This passes the cricondentherm and
    approaches the mixture critical endpoint without accepting the algebraic
    ``K=1`` root. Fixed-temperature points after the closure seed are replaced
    by the ordered log-K continuation; failed continuation points remain
    explicit. Liquid-liquid tracing uses
    :func:`continue_phase_transition_branch` in both temperature directions
    from each selected seed. The formulation follows Michelsen and Mollerup,
    *Thermodynamic Models: Fundamentals & Computational Aspects*, 2nd ed.
    (2007), chapter 12, ISBN 978-87-989961-3-2.

    Temperature continuation can end at a critical merge or fail at a turning
    point. Such points are retained as non-converged; the result does not
    silently interpolate across a missing segment.
    """
    if vapor_liquid_closure_points < 1 or vapor_liquid_closure_log_k <= 0.0:
        raise ValueError("vapor-liquid closure controls must be positive")
    parent = normalize_composition(composition)
    fixed_temperature_vapor_liquid = phase_envelope(
        model,
        vapor_liquid_temperatures,
        parent,
        kinds=("bubble", "dew"),
    )
    vapor_liquid: dict[SaturationKind, tuple[SaturationPoint, ...]] = {}
    for kind, points in fixed_temperature_vapor_liquid.items():
        physical_indices = [
            index
            for index, point in enumerate(points)
            if point.converged
            and bool(point.residual_norm <= tolerance)
            and bool(
                torch.max(torch.abs(point.incipient_composition - parent))
                > minimum_phase_separation
            )
        ]
        if not physical_indices:
            vapor_liquid[kind] = points
            continue
        seed_index = max(
            physical_indices,
            key=lambda index: float(points[index].temperature.detach()),
        )
        vapor_seed = points[seed_index]
        log_k = torch.log(vapor_seed.k_values.detach())
        controlled_component = int(torch.argmax(torch.abs(log_k)))
        initial_coordinate = log_k[controlled_component]
        target_magnitude = initial_coordinate.new_tensor(vapor_liquid_closure_log_k)
        if bool(torch.abs(initial_coordinate) <= target_magnitude):
            vapor_liquid[kind] = points[: seed_index + 1]
            continue
        magnitudes = torch.logspace(
            torch.log10(0.9 * torch.abs(initial_coordinate)),
            torch.log10(target_magnitude),
            vapor_liquid_closure_points,
            dtype=initial_coordinate.dtype,
            device=initial_coordinate.device,
        )
        targets = torch.sign(initial_coordinate) * magnitudes
        closure = continue_saturation_branch(
            model,
            parent,
            vapor_seed,
            targets,
            controlled_component=controlled_component,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        vapor_liquid[kind] = (*points[: seed_index + 1], *closure)

    accepted_seeds = [
        seed
        for seed in liquid_liquid_seeds
        if seed.converged and seed.state.boundary_kind == "liquid-liquid"
    ]
    if not accepted_seeds:
        return PhaseEnvelopeSet(vapor_liquid, ())
    if liquid_liquid_temperatures is None:
        raise ValueError("liquid-liquid temperatures are required with branch seeds")
    for accepted_seed in accepted_seeds:
        if accepted_seed.state.parent_composition.shape != parent.shape or not bool(
            torch.allclose(
                normalize_composition(accepted_seed.state.parent_composition).detach(),
                parent.detach(),
                rtol=1.0e-10,
                atol=1.0e-12,
            )
        ):
            raise ValueError("liquid-liquid seed composition must match the envelope composition")

    ordered_seeds = sorted(
        accepted_seeds,
        key=lambda item: float(item.pressure.detach()),
    )
    selected_seeds = [ordered_seeds[0]]
    if (
        len(ordered_seeds) > 1
        and float(ordered_seeds[-1].pressure.detach() / ordered_seeds[0].pressure.detach()) > 1.05
    ):
        selected_seeds.append(ordered_seeds[-1])

    liquid_liquid: list[tuple[PhaseTransitionPoint, ...]] = []
    for evaluation in selected_seeds:
        state = evaluation.state
        liquid_seed = PhaseTransitionPoint(
            state.temperature,
            evaluation.pressure,
            parent,
            evaluation.phase_compositions[0],
            ("liquid", "liquid"),
            evaluation.iterations,
            evaluation.converged,
            evaluation.residual_norm,
            evaluation.phase_separation,
        )
        descending_temperatures = liquid_liquid_temperatures[
            liquid_liquid_temperatures < liquid_seed.temperature
        ].flip(0)
        ascending_temperatures = liquid_liquid_temperatures[
            liquid_liquid_temperatures > liquid_seed.temperature
        ]
        descending = (
            continue_phase_transition_branch(
                model,
                descending_temperatures,
                liquid_seed,
                minimum_pressure=minimum_pressure,
                maximum_pressure=maximum_pressure,
                minimum_phase_separation=minimum_phase_separation,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            if descending_temperatures.numel()
            else ()
        )
        ascending = (
            continue_phase_transition_branch(
                model,
                ascending_temperatures,
                liquid_seed,
                minimum_pressure=minimum_pressure,
                maximum_pressure=maximum_pressure,
                minimum_phase_separation=minimum_phase_separation,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            if ascending_temperatures.numel()
            else ()
        )
        liquid_liquid.append(
            tuple(
                sorted(
                    (*descending, liquid_seed, *ascending),
                    key=lambda point: float(point.temperature.detach()),
                )
            )
        )
    return PhaseEnvelopeSet(vapor_liquid, tuple(liquid_liquid))


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

    Before the coupled solve, each supplied phase volume is pressure-matched
    by a bounded one-dimensional Newton correction at its supplied
    composition. This retains the nearby density root while preventing modest
    continuation-predictor errors in both volumes from steering the coupled
    solve toward the homogeneous root.

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

    pressure_scale = torch.clamp_min(
        specified_pressure.detach().abs(),
        x.new_tensor(1.0),
    )
    density_lower_bound = x.new_tensor(-20.0)
    density_upper_bound = x.new_tensor(20.0)

    def pressure_matched_log_density(composition: Tensor, volume: Tensor) -> Tensor:
        initial_log_density = torch.log(volume.reciprocal()).reshape(1)

        def pressure_residual(current: Tensor) -> Tensor:
            current_volume = torch.exp(-current[0])
            current_pressure: Tensor = pressure_function(
                specified_temperature,
                current_volume,
                composition,
            )
            return ((current_pressure - specified_pressure) / pressure_scale).reshape(1)

        initial_residual_norm = pressure_residual(initial_log_density).abs().max()
        if bool(initial_residual_norm <= tolerance):
            return initial_log_density[0]
        density_result = damped_newton(
            pressure_residual,
            initial_log_density,
            tolerance=tolerance,
            max_iterations=min(max_iterations, 12),
            lower_bound=density_lower_bound.reshape(1),
            upper_bound=density_upper_bound.reshape(1),
            jacobian_refresh_interval=1,
        )
        if bool(
            torch.isfinite(density_result.residual_norm)
            & (density_result.residual_norm < initial_residual_norm)
        ):
            return density_result.solution[0]
        return initial_log_density[0]

    variables = torch.stack(
        (
            logit(x[0]),
            logit(y[0]),
            pressure_matched_log_density(x, liquid_volume),
            pressure_matched_log_density(y, vapor_volume),
        )
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
        jacobian_refresh_interval=1,
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


def binary_bubble_temperature(
    model: StateModel,
    pressure: Tensor,
    liquid_composition: Tensor,
    *,
    initial_temperature: Tensor | None = None,
    initial_vapor_composition: Tensor | None = None,
    minimum_temperature: Tensor | float | None = None,
    maximum_temperature: Tensor | float | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 30,
    minimum_phase_separation: float = 1.0e-6,
) -> BinaryBubbleTemperature:
    """Solve binary bubble temperature and vapor composition at fixed ``P, x``.

    Parameters
    ----------
    model
        Homogeneous-state fugacity model with critical constants for default
        initialization.
    pressure
        Specified finite positive scalar pressure in Pa.
    liquid_composition
        Strictly positive binary liquid mole fractions.
    initial_temperature
        Optional finite positive temperature estimate in K. The
        composition-weighted critical temperature is used when omitted.
    initial_vapor_composition
        Optional strictly positive binary vapor estimate. Wilson equilibrium
        ratios initialize the vapor when omitted.
    minimum_temperature, maximum_temperature
        Optional finite positive temperature bounds in K.
    tolerance
        Positive maximum absolute component log-fugacity residual.
    max_iterations
        Positive damped-Newton iteration limit.
    minimum_phase_separation
        Nonnegative minimum of ``max(abs(y - x))`` required for physical
        convergence.

    Returns
    -------
    BinaryBubbleTemperature
        Bubble temperature, specified pressure, coexisting compositions, and
        explicit convergence diagnostics.

    Raises
    ------
    ValueError
        If the state, composition, bounds, or solver controls are invalid.

    Notes
    -----
    The unknowns are the log temperature and the logit of the first vapor
    mole fraction. Fugacity equality is enforced with liquid and vapor roots,
    following Michelsen and Mollerup, *Thermodynamic Models: Fundamentals &
    Computational Aspects*, 2nd ed. (2007), chapter 12,
    ISBN 978-87-989961-3-2. The Jacobian is assembled by PyTorch automatic
    differentiation, so equation-of-state-specific temperature derivatives
    are not embedded in this solver.

    A converged nonlinear residual is not sufficient near a critical endpoint:
    the homogeneous ``x == y`` solution is rejected unless the requested
    ``minimum_phase_separation`` is exceeded.
    """
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("binary bubble temperature requires positive solver controls")
    if minimum_phase_separation < 0.0:
        raise ValueError("minimum phase separation must be nonnegative")
    if pressure.ndim != 0 or not bool(torch.isfinite(pressure) & (pressure > 0.0)):
        raise ValueError("binary bubble pressure must be one finite positive scalar")

    x = normalize_composition(liquid_composition)
    if x.shape != (2,) or not bool(torch.isfinite(x).all() & (x > 0.0).all()):
        raise ValueError(
            "binary bubble-temperature liquid composition must be a finite "
            "positive two-component vector"
        )
    solve_pressure = pressure.to(dtype=x.dtype, device=x.device)
    components = _components_from_model(model)

    if initial_temperature is None:
        lower_guess = 0.2 * torch.min(components.critical_temperature)
        upper_guess = 2.0 * torch.max(components.critical_temperature)

        def wilson_closure(temperature: Tensor) -> Tensor:
            return torch.sum(x * wilson_k_values(components, temperature, solve_pressure)) - 1.0

        lower_closure = wilson_closure(lower_guess)
        upper_closure = wilson_closure(upper_guess)
        if bool((lower_closure <= 0.0) & (upper_closure >= 0.0)):
            for _ in range(48):
                midpoint = 0.5 * (lower_guess + upper_guess)
                if bool(wilson_closure(midpoint) < 0.0):
                    lower_guess = midpoint
                else:
                    upper_guess = midpoint
            initial_temperature = 0.5 * (lower_guess + upper_guess)
        else:
            initial_temperature = torch.sum(x * components.critical_temperature)
    solve_initial_temperature = initial_temperature.to(dtype=x.dtype, device=x.device)
    if solve_initial_temperature.ndim != 0 or not bool(
        torch.isfinite(solve_initial_temperature) & (solve_initial_temperature > 0.0)
    ):
        raise ValueError("initial binary bubble temperature must be one finite positive scalar")

    if initial_vapor_composition is None:
        initial_k = wilson_k_values(components, solve_initial_temperature, solve_pressure)
        initial_y = normalize_composition(x * initial_k)
    else:
        initial_y = normalize_composition(
            initial_vapor_composition.to(dtype=x.dtype, device=x.device)
        )
    if initial_y.shape != (2,) or not bool(
        torch.isfinite(initial_y).all() & (initial_y > 0.0).all()
    ):
        raise ValueError("initial binary bubble vapor composition must be finite and positive")

    def temperature_bound(value: Tensor | float | None) -> Tensor | None:
        if value is None:
            return None
        return torch.as_tensor(value, dtype=x.dtype, device=x.device)

    minimum = temperature_bound(minimum_temperature)
    maximum = temperature_bound(maximum_temperature)
    for name, value in (("minimum", minimum), ("maximum", maximum)):
        if value is not None and (
            value.ndim != 0 or not bool(torch.isfinite(value) & (value > 0.0))
        ):
            raise ValueError(f"{name} binary bubble temperature must be one finite positive scalar")
    if minimum is not None and maximum is not None and not bool(minimum < maximum):
        raise ValueError("minimum binary bubble temperature must be below maximum temperature")

    epsilon = 32.0 * torch.finfo(x.dtype).eps

    def logit(value: Tensor) -> Tensor:
        bounded = torch.clamp(value, epsilon, 1.0 - epsilon)
        return torch.log(bounded) - torch.log1p(-bounded)

    variables = torch.stack((logit(initial_y[0]), torch.log(solve_initial_temperature)))

    def residual(current: Tensor) -> Tensor:
        y1 = torch.sigmoid(current[0])
        y = torch.stack((y1, 1.0 - y1))
        temperature = torch.exp(current[1])
        return (
            torch.log(x)
            + model.log_fugacity_coefficients(temperature, solve_pressure, x, "liquid")
            - torch.log(y)
            - model.log_fugacity_coefficients(temperature, solve_pressure, y, "vapor")
        )

    lower_temperature_log = (
        variables.new_tensor(-torch.inf) if minimum is None else torch.log(minimum)
    )
    upper_temperature_log = (
        variables.new_tensor(torch.inf) if maximum is None else torch.log(maximum)
    )
    result = damped_newton(
        residual,
        variables,
        tolerance=tolerance,
        max_iterations=max_iterations,
        lower_bound=torch.stack((variables.new_tensor(-30.0), lower_temperature_log)),
        upper_bound=torch.stack((variables.new_tensor(30.0), upper_temperature_log)),
    )
    vapor_first = torch.sigmoid(result.solution[0])
    vapor = torch.stack((vapor_first, 1.0 - vapor_first))
    phase_separation = torch.max(torch.abs(vapor - x))
    return BinaryBubbleTemperature(
        torch.exp(result.solution[1]),
        solve_pressure,
        x,
        vapor,
        result.iterations,
        result.converged and bool(float(phase_separation.detach()) > minimum_phase_separation),
        result.residual_norm,
        phase_separation,
    )


def trace_binary_pxy_isotherm(
    model: StateModel,
    temperature: Tensor,
    liquid_first_fractions: Tensor,
    *,
    initial_pressure: Tensor | None = None,
    initial_vapor_composition: Tensor | None = None,
    minimum_pressure: Tensor | float | None = None,
    maximum_pressure: Tensor | float | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 30,
    minimum_phase_separation: float = 1.0e-6,
) -> BinaryPxyIsotherm:
    """Trace a binary ``P-x-y`` isotherm with liquid-composition continuation.

    Parameters
    ----------
    model
        Homogeneous-state fugacity model.
    temperature
        Specified finite positive scalar temperature in K.
    liquid_first_fractions
        One-dimensional, strictly interior liquid mole fractions of the first
        component, in the requested continuation order.
    initial_pressure, initial_vapor_composition
        Optional estimates for the first point. Every physically converged
        point initializes the next one.
    minimum_pressure, maximum_pressure
        Optional positive pressure bounds in Pa passed to every bubble solve.
    tolerance, max_iterations
        Positive damped-Newton controls.
    minimum_phase_separation
        Nonnegative minimum of ``max(abs(y - x))`` used to reject homogeneous
        algebraic roots.

    Returns
    -------
    BinaryPxyIsotherm
        Attempted pressures, compositions, and pointwise convergence
        diagnostics in the same order as ``liquid_first_fractions``.

    Notes
    -----
    This fugacity-based driver is valid for any :class:`StateModel`.
    :func:`trace_binary_helmholtz_pxy_isotherm` is a distinct volume-based
    continuation algorithm for :class:`HelmholtzStateModel` models and can
    follow composition folds that are not single-valued in liquid
    composition.

    The phase-equilibrium formulation follows Michelsen and Mollerup,
    *Thermodynamic Models: Fundamentals & Computational Aspects*, 2nd ed.
    (2007), chapter 12, ISBN 978-87-989961-3-2.
    """
    if temperature.ndim != 0 or not bool(torch.isfinite(temperature) & (temperature > 0.0)):
        raise ValueError("binary P-x-y trace requires one finite positive temperature")
    if (
        not liquid_first_fractions.is_floating_point()
        or liquid_first_fractions.ndim != 1
        or liquid_first_fractions.numel() == 0
        or not bool(
            torch.isfinite(liquid_first_fractions).all()
            & (liquid_first_fractions > 0.0).all()
            & (liquid_first_fractions < 1.0).all()
        )
    ):
        raise ValueError(
            "binary P-x-y trace requires a nonempty floating vector of "
            "strictly interior liquid first-component fractions"
        )
    if minimum_phase_separation < 0.0:
        raise ValueError("minimum phase separation must be nonnegative")

    pressures: list[Tensor] = []
    liquid_compositions: list[Tensor] = []
    vapor_compositions: list[Tensor] = []
    iterations: list[int] = []
    converged: list[bool] = []
    residuals: list[Tensor] = []
    separations: list[Tensor] = []
    next_pressure = initial_pressure
    next_vapor = initial_vapor_composition
    for first_fraction in liquid_first_fractions:
        liquid = torch.stack((first_fraction, 1.0 - first_fraction))
        point = binary_bubble_point(
            model,
            temperature,
            liquid,
            initial_pressure=next_pressure,
            initial_vapor_composition=next_vapor,
            minimum_pressure=minimum_pressure,
            maximum_pressure=maximum_pressure,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        separation = torch.max(torch.abs(point.vapor_composition - point.liquid_composition))
        physical = point.converged and bool(float(separation.detach()) > minimum_phase_separation)
        pressures.append(point.pressure)
        liquid_compositions.append(point.liquid_composition)
        vapor_compositions.append(point.vapor_composition)
        iterations.append(point.iterations)
        converged.append(physical)
        residuals.append(point.residual_norm)
        separations.append(separation)
        if physical:
            next_pressure = point.pressure
            next_vapor = point.vapor_composition

    return BinaryPxyIsotherm(
        temperature,
        torch.stack(pressures),
        torch.stack(liquid_compositions),
        torch.stack(vapor_compositions),
        torch.tensor(iterations, dtype=torch.int64, device=liquid_first_fractions.device),
        torch.tensor(converged, dtype=torch.bool, device=liquid_first_fractions.device),
        torch.stack(residuals),
        torch.stack(separations),
    )


def trace_binary_txy_isobar(
    model: StateModel,
    pressure: Tensor,
    liquid_first_fractions: Tensor,
    *,
    initial_temperature: Tensor | None = None,
    initial_vapor_composition: Tensor | None = None,
    minimum_temperature: Tensor | float | None = None,
    maximum_temperature: Tensor | float | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 30,
    minimum_phase_separation: float = 1.0e-6,
) -> BinaryTxyIsobar:
    """Trace a binary ``T-x-y`` isobar with liquid-composition continuation.

    Parameters
    ----------
    model
        Homogeneous-state fugacity model.
    pressure
        Specified finite positive scalar pressure in Pa.
    liquid_first_fractions
        One-dimensional, strictly interior liquid mole fractions of the first
        component, in the requested continuation order.
    initial_temperature, initial_vapor_composition
        Optional estimates for the first point. Every physically converged
        point initializes the next one.
    minimum_temperature, maximum_temperature
        Optional positive temperature bounds in K passed to every bubble
        solve.
    tolerance, max_iterations
        Positive damped-Newton controls.
    minimum_phase_separation
        Nonnegative minimum of ``max(abs(y - x))`` used to reject homogeneous
        algebraic roots.

    Returns
    -------
    BinaryTxyIsobar
        Attempted temperatures, compositions, and pointwise convergence
        diagnostics in the same order as ``liquid_first_fractions``.

    Notes
    -----
    Every point is solved by :func:`binary_bubble_temperature`; both liquid
    and vapor branches therefore come from one fugacity-equality calculation.
    The continuation order is scientifically significant near critical
    endpoints and should be selected to begin on a well-separated branch.

    The phase-equilibrium formulation follows Michelsen and Mollerup,
    *Thermodynamic Models: Fundamentals & Computational Aspects*, 2nd ed.
    (2007), chapter 12, ISBN 978-87-989961-3-2.
    """
    if pressure.ndim != 0 or not bool(torch.isfinite(pressure) & (pressure > 0.0)):
        raise ValueError("binary T-x-y trace requires one finite positive pressure")
    if (
        not liquid_first_fractions.is_floating_point()
        or liquid_first_fractions.ndim != 1
        or liquid_first_fractions.numel() == 0
        or not bool(
            torch.isfinite(liquid_first_fractions).all()
            & (liquid_first_fractions > 0.0).all()
            & (liquid_first_fractions < 1.0).all()
        )
    ):
        raise ValueError(
            "binary T-x-y trace requires a nonempty floating vector of "
            "strictly interior liquid first-component fractions"
        )
    if minimum_phase_separation < 0.0:
        raise ValueError("minimum phase separation must be nonnegative")

    temperatures: list[Tensor] = []
    liquid_compositions: list[Tensor] = []
    vapor_compositions: list[Tensor] = []
    iterations: list[int] = []
    converged: list[bool] = []
    residuals: list[Tensor] = []
    separations: list[Tensor] = []
    next_temperature = initial_temperature
    next_vapor = initial_vapor_composition
    for first_fraction in liquid_first_fractions:
        liquid = torch.stack((first_fraction, 1.0 - first_fraction))
        point = binary_bubble_temperature(
            model,
            pressure,
            liquid,
            initial_temperature=next_temperature,
            initial_vapor_composition=next_vapor,
            minimum_temperature=minimum_temperature,
            maximum_temperature=maximum_temperature,
            tolerance=tolerance,
            max_iterations=max_iterations,
            minimum_phase_separation=minimum_phase_separation,
        )
        temperatures.append(point.temperature)
        liquid_compositions.append(point.liquid_composition)
        vapor_compositions.append(point.vapor_composition)
        iterations.append(point.iterations)
        converged.append(point.converged)
        residuals.append(point.residual_norm)
        separations.append(point.phase_separation)
        if point.converged:
            next_temperature = point.temperature
            next_vapor = point.vapor_composition

    return BinaryTxyIsobar(
        pressure,
        torch.stack(temperatures),
        torch.stack(liquid_compositions),
        torch.stack(vapor_compositions),
        torch.tensor(iterations, dtype=torch.int64, device=liquid_first_fractions.device),
        torch.tensor(converged, dtype=torch.bool, device=liquid_first_fractions.device),
        torch.stack(residuals),
        torch.stack(separations),
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
    minimum_pressure_step_separation_ratio: float = 0.1,
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
    minimum_pressure_step_separation_ratio
        Positive ratio, no greater than one, between a candidate phase
        separation and the preceding accepted separation during fixed-pressure
        continuation. A larger contraction is treated as a branch jump and
        triggers pressure refinement instead of accepting a nearly homogeneous
        algebraic root.
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
    endpoint after the first rejected pressure. Fixed-pressure initial states
    use a bounded secant extrapolation of both composition logits and both
    logarithmic phase volumes. A candidate whose phase split contracts by more
    than ``minimum_pressure_step_separation_ratio`` is rejected as a possible
    jump to the algebraic homogeneous root and causes pressure refinement. The
    routine never forces a physically open branch to close. Bisection and the
    discrete accept/stop/coordinate-switch decisions are not differentiable,
    but accepted equilibrium tensors preserve their PyTorch graphs.
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
    if (
        minimum_pressure_step_separation_ratio <= 0.0
        or minimum_pressure_step_separation_ratio > 1.0
        or not torch.isfinite(torch.tensor(minimum_pressure_step_separation_ratio))
    ):
        raise ValueError(
            "binary p-x-y minimum pressure-step separation ratio must be finite "
            "and in the interval (0, 1]"
        )
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
    physical_volume_points: list[BinaryBubblePointWithVolumes | BinaryVLEPointWithVolumes] = []
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

    def pressure_continuation_diagnostics(
        point: BinaryVLEPointWithVolumes,
        preceding_point: BinaryBubblePointWithVolumes | BinaryVLEPointWithVolumes,
    ) -> tuple[bool, Tensor]:
        physical, separation = physical_diagnostics(point)
        preceding_separation = torch.max(
            torch.abs(preceding_point.vapor_composition - preceding_point.liquid_composition)
        )
        branch_continuous = bool(
            separation >= minimum_pressure_step_separation_ratio * preceding_separation
        )
        return physical and branch_continuous, separation

    def pressure_predictor(
        target_pressure: Tensor,
        fallback: BinaryBubblePointWithVolumes | BinaryVLEPointWithVolumes,
    ) -> BinaryVLEPointWithVolumes:
        """Extrapolate the last two physical states in logarithmic pressure."""
        if len(physical_volume_points) < 2:
            return BinaryVLEPointWithVolumes(
                fallback.temperature,
                target_pressure,
                fallback.liquid_composition,
                fallback.vapor_composition,
                fallback.iterations,
                fallback.converged,
                fallback.residual_norm,
                fallback.liquid_molar_volume,
                fallback.vapor_molar_volume,
            )

        predecessor, current = physical_volume_points[-2:]
        log_pressure_step = torch.log(current.pressure) - torch.log(predecessor.pressure)
        if not bool(
            torch.isfinite(log_pressure_step)
            & (torch.abs(log_pressure_step) > torch.finfo(log_pressure_step.dtype).eps)
        ):
            factor = log_pressure_step.new_tensor(0.0)
        else:
            factor = (torch.log(target_pressure) - torch.log(current.pressure)) / log_pressure_step
            factor = torch.clamp(factor, 0.0, 2.0)

        epsilon = 32.0 * torch.finfo(current.liquid_composition.dtype).eps

        def logit(value: Tensor) -> Tensor:
            bounded = torch.clamp(value, epsilon, 1.0 - epsilon)
            return torch.log(bounded) - torch.log1p(-bounded)

        def extrapolate_composition(previous_composition: Tensor, composition: Tensor) -> Tensor:
            predicted_first = torch.sigmoid(
                logit(composition[0])
                + factor * (logit(composition[0]) - logit(previous_composition[0]))
            )
            return torch.stack((predicted_first, 1.0 - predicted_first))

        predicted_liquid = extrapolate_composition(
            predecessor.liquid_composition,
            current.liquid_composition,
        )
        predicted_vapor = extrapolate_composition(
            predecessor.vapor_composition,
            current.vapor_composition,
        )
        predicted_liquid_volume = torch.exp(
            torch.log(current.liquid_molar_volume)
            + factor
            * (torch.log(current.liquid_molar_volume) - torch.log(predecessor.liquid_molar_volume))
        )
        predicted_vapor_volume = torch.exp(
            torch.log(current.vapor_molar_volume)
            + factor
            * (torch.log(current.vapor_molar_volume) - torch.log(predecessor.vapor_molar_volume))
        )
        return BinaryVLEPointWithVolumes(
            current.temperature,
            target_pressure,
            predicted_liquid,
            predicted_vapor,
            current.iterations,
            current.converged,
            current.residual_norm,
            predicted_liquid_volume,
            predicted_vapor_volume,
        )

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
            physical_volume_points.append(point)
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
                physical_volume_points.append(point)
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
            predicted_initial_point = pressure_predictor(pressure, pressure_initial_point)
            vle_point = binary_helmholtz_vle_point(
                model,
                temperature.to(dtype=pressure.dtype, device=pressure.device),
                pressure,
                predicted_initial_point,
                tolerance=tolerance,
                max_iterations=max_iterations,
                minimum_phase_separation=minimum_phase_separation,
            )
            physical, separation = pressure_continuation_diagnostics(
                vle_point,
                pressure_initial_point,
            )
            points.append(vle_point)
            accepted.append(pressure.new_tensor(physical, dtype=torch.bool))
            separations.append(separation)
            if physical:
                pressure_initial_point = vle_point
                physical_volume_points.append(vle_point)
            elif stop_on_failure:
                rejected_pressure = pressure
                break
        if rejected_pressure is not None:
            for _ in range(pressure_failure_refinement_steps):
                candidate_pressure = torch.sqrt(pressure_initial_point.pressure * rejected_pressure)
                predicted_initial_point = pressure_predictor(
                    candidate_pressure,
                    pressure_initial_point,
                )
                vle_point = binary_helmholtz_vle_point(
                    model,
                    temperature.to(
                        dtype=candidate_pressure.dtype,
                        device=candidate_pressure.device,
                    ),
                    candidate_pressure,
                    predicted_initial_point,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    minimum_phase_separation=minimum_phase_separation,
                )
                physical, separation = pressure_continuation_diagnostics(
                    vle_point,
                    pressure_initial_point,
                )
                points.append(vle_point)
                accepted.append(candidate_pressure.new_tensor(physical, dtype=torch.bool))
                separations.append(separation)
                if physical:
                    pressure_initial_point = vle_point
                    physical_volume_points.append(vle_point)
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
