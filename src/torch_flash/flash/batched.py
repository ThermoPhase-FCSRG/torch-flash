"""Vectorized fixed-two-phase flash calculations.

This solver is intended for grids whose two-phase region has already been
identified, for example from bubble/dew branches. It deliberately omits a
per-state stability test; use :func:`torch_flash.two_phase_flash` when phase
discovery is required.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.initialization import wilson_k_values
from torch_flash.material_balance import rachford_rice
from torch_flash.properties.state import StateModel
from torch_flash.types import BatchedTwoPhaseFlashResult, ChemicalState, PhaseKind


def _model_components(model: StateModel) -> ComponentSet:
    required = (
        "critical_temperature",
        "critical_pressure",
        "acentric_factor",
    )
    if not all(hasattr(model, item) for item in required):
        raise ValueError("initial K values are required for a model without critical constants")
    return cast(ComponentSet, model)


def _batch_inputs(state: ChemicalState) -> tuple[Tensor, Tensor, Tensor]:
    if state.temperature.ndim != 1 or state.pressure.ndim != 1:
        raise ValueError("batched_two_phase_flash requires one-dimensional T and P batches")
    if state.temperature.shape != state.pressure.shape:
        raise ValueError("temperature and pressure batches must have the same shape")
    batch_size = state.temperature.numel()
    if state.composition.ndim == 1:
        composition = state.composition.expand(batch_size, -1)
    elif state.composition.ndim == 2 and state.composition.shape[0] == batch_size:
        composition = state.composition
    else:
        raise ValueError("composition must be one vector or one vector per T-P state")
    return state.temperature, state.pressure, composition


def _straddles_unity(log_k: Tensor) -> Tensor:
    k_values = torch.exp(_bounded_log_k(log_k))
    return (k_values.amin(dim=-1) < 1.0) & (k_values.amax(dim=-1) > 1.0)


def _bounded_log_k(log_k: Tensor) -> Tensor:
    """Keep K ratios positive and finite in double-precision trial states."""
    return torch.clamp(log_k, -80.0, 80.0)


def _admissible_update(current: Tensor, target: Tensor) -> Tensor:
    """Backtrack only rows that would lose their finite RR root."""
    factor = torch.ones(current.shape[:-1], dtype=current.dtype, device=current.device)
    for _ in range(16):
        candidate = _bounded_log_k(current + factor[..., None] * (target - current))
        valid = _straddles_unity(candidate)
        if bool(valid.all()):
            return candidate
        factor = torch.where(valid, factor, 0.5 * factor)
    candidate = _bounded_log_k(current + factor[..., None] * (target - current))
    valid = _straddles_unity(candidate)
    return torch.where(valid[..., None], candidate, current)


def batched_two_phase_flash(
    model: StateModel,
    state: ChemicalState,
    *,
    initial_k_values: Tensor | None = None,
    phase_roots: tuple[PhaseKind, PhaseKind] = ("liquid", "vapor"),
    tolerance: float = 1.0e-8,
    substitution_iterations: int = 12,
    newton_iterations: int = 16,
) -> BatchedTwoPhaseFlashResult:
    """Solve independent known-two-phase TP states in one tensor batch.

    The hybrid algorithm performs vectorized successive substitution followed
    by block-diagonal Newton updates obtained with PyTorch autodiff. Inputs
    must have ``Kmin < 1 < Kmax`` for every state.

    Parameters
    ----------
    model
        Homogeneous-state model supporting batched log fugacity coefficients.
    state
        Temperatures in K and pressures in Pa as one-dimensional tensors of
        shape ``(batch,)``. Composition may have shape ``(ncomponents,)`` for
        a common feed or ``(batch, ncomponents)`` for state-specific feeds.
    initial_k_values
        Optional positive vapor-to-liquid ratios of shape
        ``(batch, ncomponents)``. Wilson estimates are used when omitted.
    phase_roots
        Cubic-root requests for the denominator and numerator phases,
        respectively. The default ``("liquid", "vapor")`` solves a known
        vapor-liquid branch. ``("stable", "stable")`` is useful when a
        preceding stability calculation supplies a generic phase-split seed.
    tolerance
        Per-state convergence threshold for the maximum absolute
        log-fugacity residual.
    substitution_iterations
        Maximum vectorized successive-substitution passes.
    newton_iterations
        Maximum block-diagonal Newton passes after substitution.

    Returns
    -------
    BatchedTwoPhaseFlashResult
        Per-state phase fractions, compositions, K values, convergence flags,
        residuals, and the shared number of executed iteration passes.

    Raises
    ------
    ValueError
        If batch shapes are inconsistent, iteration controls are invalid,
        initial K values are invalid, or a state does not satisfy
        ``Kmin < 1 < Kmax``.

    Notes
    -----
    ``converged`` is reported per state and additionally requires a physical
    vapor fraction in ``[0, 1]``.

    This routine does not perform tangent-plane stability analysis. That
    separation is intentional: an envelope or another phase classifier can
    cheaply select thousands of known two-phase grid cells, while ambiguous
    cells can be sent to the full scalar stability-tested flash.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if substitution_iterations < 0 or newton_iterations < 0:
        raise ValueError("iteration counts must be nonnegative")
    if len(phase_roots) != 2 or any(
        phase not in ("liquid", "vapor", "stable") for phase in phase_roots
    ):
        raise ValueError("phase_roots must contain two valid phase-root requests")
    temperature, pressure, composition = _batch_inputs(state)
    if initial_k_values is None:
        k_values = wilson_k_values(
            _model_components(model),
            temperature,
            pressure,
        )
    else:
        if initial_k_values.shape != composition.shape:
            raise ValueError("initial K values must have one row per state and component")
        if bool((~torch.isfinite(initial_k_values)).any() | (initial_k_values <= 0.0).any()):
            raise ValueError("initial K values must be finite and strictly positive")
        k_values = initial_k_values.to(
            dtype=composition.dtype,
            device=composition.device,
        )
    log_k = _bounded_log_k(torch.log(k_values))
    if not bool(_straddles_unity(log_k).all()):
        raise ValueError("every batched state requires Kmin < 1 < Kmax")

    def equilibrium_residual(current_log_k: Tensor) -> Tensor:
        split = rachford_rice(
            composition,
            torch.exp(current_log_k),
            tolerance=1.0e-13,
        )
        log_phi_liquid = model.log_fugacity_coefficients(
            temperature,
            pressure,
            split.liquid_composition,
            phase_roots[0],
        )
        log_phi_vapor = model.log_fugacity_coefficients(
            temperature,
            pressure,
            split.vapor_composition,
            phase_roots[1],
        )
        return current_log_k - (log_phi_liquid - log_phi_vapor)

    completed_iterations = 0
    for index in range(substitution_iterations):
        residual = equilibrium_residual(log_k)
        norm = residual.abs().amax(dim=-1)
        active = norm > tolerance
        if not bool(active.any()):
            completed_iterations = index + 1
            break
        damping = 0.8 if index < 4 else 0.5
        target = log_k - damping * residual
        updated = _admissible_update(log_k, target)
        log_k = torch.where(active[..., None], updated, log_k)
        completed_iterations = index + 1

    ncomponents = composition.shape[-1]
    eye = torch.eye(
        ncomponents,
        dtype=composition.dtype,
        device=composition.device,
    )
    for index in range(newton_iterations):
        current = log_k.detach().requires_grad_(True)
        residual = equilibrium_residual(current)
        norm = residual.abs().amax(dim=-1)
        active = norm > tolerance
        if not bool(active.any()):
            log_k = current.detach()
            completed_iterations += index
            break
        jacobian_rows = tuple(
            torch.autograd.grad(
                residual[:, component].sum(),
                current,
                retain_graph=component + 1 < ncomponents,
            )[0]
            for component in range(ncomponents)
        )
        jacobian = torch.stack(jacobian_rows, dim=-2)
        try:
            step = torch.linalg.solve(
                jacobian + 1.0e-10 * eye,
                -residual[..., None],
            ).squeeze(-1)
        except torch.linalg.LinAlgError:
            step = -0.25 * residual
        step = torch.clamp(step, -3.0, 3.0)

        factor = torch.ones_like(norm)
        accepted = ~active
        next_log_k = current.detach()
        for _ in range(12):
            trial = _bounded_log_k(current.detach() + factor[..., None] * step.detach())
            straddles = _straddles_unity(trial)
            safe_trial = torch.where(straddles[..., None], trial, current.detach())
            trial_norm = equilibrium_residual(safe_trial).detach().abs().amax(dim=-1)
            improved = active & ~accepted & straddles & (trial_norm < norm.detach())
            next_log_k = torch.where(improved[..., None], trial, next_log_k)
            accepted = accepted | improved
            if bool(accepted.all()):
                break
            factor = torch.where(accepted, factor, 0.5 * factor)
        log_k = next_log_k
        completed_iterations += 1

    final_residual = equilibrium_residual(log_k)
    residual_norm = final_residual.abs().amax(dim=-1)
    split = rachford_rice(
        composition,
        torch.exp(log_k),
        tolerance=1.0e-13,
    )
    physical_fraction = (split.vapor_fraction >= -10.0 * tolerance) & (
        split.vapor_fraction <= 1.0 + 10.0 * tolerance
    )
    converged = (residual_norm <= tolerance) & physical_fraction
    return BatchedTwoPhaseFlashResult(
        torch.clamp(split.vapor_fraction, 0.0, 1.0),
        torch.clamp(split.liquid_fraction, 0.0, 1.0),
        split.liquid_composition,
        split.vapor_composition,
        torch.exp(log_k),
        completed_iterations,
        converged,
        residual_norm,
    )
