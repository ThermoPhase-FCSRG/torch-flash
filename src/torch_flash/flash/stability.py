"""Michelsen tangent-plane-distance stability analysis.

Reference
---------
M. L. Michelsen, "The isothermal flash problem. Part I. Stability",
*Fluid Phase Equilibria* 9 (1982), 1-19,
doi:10.1016/0378-3812(82)85001-2.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.initialization import wilson_k_values
from torch_flash.properties.state import StateModel
from torch_flash.types import ChemicalState, PhaseKind, StabilityResult


def _independent_softmax(coordinates: Tensor) -> Tensor:
    # Bounding log-ratios prevents trial states from reaching exact zero during
    # a globalization step while retaining more than enough dynamic range for
    # double-precision phase-stability calculations.
    bounded = torch.clamp(coordinates, -80.0, 80.0)
    augmented = torch.cat((bounded, coordinates.new_zeros(1)))
    return torch.softmax(augmented, dim=0)


def _newton_minimize(
    objective: object,
    initial: Tensor,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[Tensor, Tensor, int, bool]:
    coordinates = initial.clone()
    converged = False
    value = objective(coordinates)  # type: ignore[operator]
    for _iteration in range(1, max_iterations + 1):
        gradient = torch.func.grad(objective)(coordinates)  # type: ignore[arg-type]
        if float(gradient.abs().max()) <= tolerance:
            converged = True
            break
        hessian = torch.func.hessian(objective)(coordinates)  # type: ignore[arg-type]
        eye = torch.eye(coordinates.numel(), dtype=coordinates.dtype, device=coordinates.device)
        regularization = 1.0e-9
        try:
            direction = torch.linalg.solve(hessian + regularization * eye, -gradient)
        except torch.linalg.LinAlgError:
            direction = -gradient
        if (
            not bool(torch.isfinite(direction).all())
            or float(torch.dot(direction, gradient)) >= 0.0
        ):
            direction = -gradient
        # A trust-region bound is important near pure-component trial states,
        # where the log-composition Hessian may be extremely ill-conditioned.
        direction_norm = torch.linalg.vector_norm(direction)
        direction = direction * torch.clamp_max(10.0 / torch.clamp_min(direction_norm, 1.0), 1.0)
        accepted = False
        step = 1.0
        for _ in range(20):
            candidate = torch.clamp(coordinates + step * direction, -80.0, 80.0)
            candidate_value = objective(candidate)  # type: ignore[operator]
            if bool(torch.isfinite(candidate_value)) and float(candidate_value) < float(
                value + 1.0e-4 * step * torch.dot(gradient, direction)
            ):
                coordinates = candidate
                value = candidate_value
                accepted = True
                break
            step *= 0.5
        if not accepted:
            finite_gradient = torch.nan_to_num(gradient, nan=0.0, posinf=1.0, neginf=-1.0)
            coordinates = torch.clamp(coordinates - 0.05 * finite_gradient, -80.0, 80.0)
            value = objective(coordinates)  # type: ignore[operator]
    return coordinates, value, _iteration, converged


def tangent_plane_stability(
    model: StateModel,
    state: ChemicalState,
    *,
    reference_phase: PhaseKind = "stable",
    initial_compositions: tuple[Tensor, ...] | None = None,
    tolerance: float = 1.0e-9,
    max_iterations: int = 40,
) -> StabilityResult:
    """Assess local phase stability by minimizing tangent-plane distance.

    The feed composition is evaluated as the reference state, and each trial
    composition is optimized in independent log-ratio coordinates. A negative
    normalized tangent-plane distance (TPD) identifies a composition that can
    lower the Gibbs energy. Multiple starting points are therefore important:
    the optimization is local and a single stationary point is not a global
    stability proof.

    Parameters
    ----------
    model
        Homogeneous-state model providing log fugacity coefficients.
    state
        Feed state. Temperature is in K, pressure is in Pa, and
        ``state.composition`` must be a one-dimensional mole-fraction vector.
    reference_phase
        Root requested when evaluating the feed fugacity coefficients.
        Trial phases are evaluated with the model's ``"stable"`` root.
    initial_compositions
        Optional trial mole-fraction vectors. Values are made positive and
        normalized before optimization. When omitted, the feed, Wilson-based
        vapor/liquid trials when available, and component-rich trials are used.
    tolerance
        Maximum absolute gradient component used by each local minimization.
        The stability decision allows a numerical margin of ``10 * tolerance``.
    max_iterations
        Maximum Newton/globalization iterations for each starting point.

    Returns
    -------
    StabilityResult
        Lowest TPD found, its trial composition, local-solver diagnostics, and
        the resulting stability classification.

    Raises
    ------
    ValueError
        If the feed composition is batched rather than one-dimensional.

    Notes
    -----
    ``StabilityResult.converged`` describes the local minimization associated
    with the reported minimum. Callers should not interpret a non-converged
    nonnegative TPD as conclusive evidence of stability.
    """
    if state.composition.ndim != 1:
        raise ValueError("stability analysis currently accepts one composition vector")
    z = state.composition
    log_phi_z = model.log_fugacity_coefficients(
        state.temperature, state.pressure, z, reference_phase
    )
    d = torch.log(torch.clamp_min(z, torch.finfo(z.dtype).tiny)) + log_phi_z

    if initial_compositions is None:
        candidates: list[Tensor] = [z]
        if all(
            hasattr(model, attribute)
            for attribute in (
                "critical_temperature",
                "critical_pressure",
                "acentric_factor",
                "molar_mass",
                "names",
            )
        ):
            components = cast(ComponentSet, model)
            k = wilson_k_values(components, state.temperature, state.pressure)
            candidates.extend((z * k, z / k))
        eye = torch.eye(z.numel(), dtype=z.dtype, device=z.device)
        candidates.extend(0.95 * eye[i] + 0.05 * z for i in range(z.numel()))
        initial_compositions = tuple(candidates)

    best_value = torch.tensor(torch.inf, dtype=z.dtype, device=z.device)
    best_composition = z
    best_iterations = 0
    best_converged = False
    for initial_composition in initial_compositions:
        trial = torch.clamp_min(initial_composition, 1.0e-16)
        trial = trial / trial.sum()
        coordinates = torch.log(trial[:-1]) - torch.log(trial[-1])

        def objective(q: Tensor) -> Tensor:
            w = _independent_softmax(q)
            # Stable-root trial evaluation catches both vapor- and liquid-like minima.
            log_phi = model.log_fugacity_coefficients(
                state.temperature, state.pressure, w, "stable"
            )
            return torch.sum(
                w * (torch.log(torch.clamp_min(w, torch.finfo(w.dtype).tiny)) + log_phi - d)
            )

        q, value, iterations, converged = _newton_minimize(
            objective,
            coordinates,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        if float(value) < float(best_value):
            best_value = value
            best_composition = _independent_softmax(q)
            best_iterations = iterations
            best_converged = converged

    stability_tolerance = 10.0 * tolerance
    return StabilityResult(
        bool(float(best_value) >= -stability_tolerance),
        best_value,
        best_composition,
        best_iterations,
        best_converged,
    )
