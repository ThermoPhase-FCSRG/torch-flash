"""Generalized Rachford-Rice and fixed-phase-count multiphase flash."""

from __future__ import annotations

import warnings
from typing import cast

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.exceptions import ConvergenceWarning, ExperimentalModelWarning
from torch_flash.initialization import wilson_k_values
from torch_flash.properties import identify_flash_phases, phase_properties
from torch_flash.properties.state import StateModel
from torch_flash.types import ChemicalState, FlashResult, PhaseKind


def solve_generalized_rachford_rice(
    composition: Tensor,
    k_values: Tensor,
    *,
    tolerance: float = 1.0e-12,
    max_iterations: int = 100,
) -> tuple[Tensor, Tensor, int, bool]:
    """Solve multiphase material balance for phase ratios to a reference phase.

    Parameters
    ----------
    composition
        Overall mole fractions with shape ``(ncomponents,)``. The vector is
        normalized internally.
    k_values
        Positive ratios to the reference-phase composition, with shape
        ``(nphases - 1, ncomponents)``.
    tolerance
        Convergence threshold for the maximum absolute generalized
        Rachford--Rice residual.
    max_iterations
        Maximum safeguarded Newton iterations.

    Returns
    -------
    tuple
        ``(phase_fractions, phase_compositions, iterations, converged)``.
        Fractions have shape ``(nphases,)`` and place the reference phase
        first. Compositions have shape ``(nphases, ncomponents)``.

    Raises
    ------
    ValueError
        If shapes are inconsistent or any equilibrium ratio is nonpositive.

    Notes
    -----
    ``converged`` reports the material-balance solve only. Fugacity equality
    must be checked separately by the enclosing multiphase flash.
    """
    if composition.ndim != 1 or k_values.ndim != 2:
        raise ValueError("expected a composition vector and a K-value matrix")
    if k_values.shape[1] != composition.numel():
        raise ValueError("K-value component dimension does not match composition")
    if bool((k_values.detach() <= 0.0).any()):
        raise ValueError("all generalized K values must be positive")
    z = composition / composition.sum()
    differences = k_values - 1.0
    n_other = k_values.shape[0]
    beta = torch.full(
        (n_other,),
        1.0 / (n_other + 1),
        dtype=z.dtype,
        device=z.device,
    )
    converged = False
    for _iteration in range(1, max_iterations + 1):
        denominator = 1.0 + torch.einsum("p,pi->i", beta, differences)
        residual = torch.sum(z[None, :] * differences / denominator[None, :], dim=1)
        if float(residual.detach().abs().max()) <= tolerance:
            converged = True
            break
        jacobian = -torch.einsum(
            "i,pi,qi->pq",
            z / denominator.square(),
            differences,
            differences,
        )
        try:
            step = torch.linalg.solve(jacobian, -residual)
        except torch.linalg.LinAlgError:
            step = -0.05 * residual
        factor = 1.0
        accepted = False
        for _ in range(30):
            candidate = beta + factor * step
            candidate_denominator = 1.0 + torch.einsum("p,pi->i", candidate, differences)
            if (
                bool((candidate.detach() > 0.0).all())
                and float(candidate.detach().sum()) < 1.0
                and bool((candidate_denominator.detach() > 0.0).all())
            ):
                beta = candidate
                accepted = True
                break
            factor *= 0.5
        if not accepted:
            beta = torch.clamp(beta - 0.01 * residual, min=1.0e-10)
            beta = beta / max(1.0, float(beta.sum() + 1.0e-10)) * 0.95
    iteration = _iteration
    fractions = torch.cat(((1.0 - beta.sum()).reshape(1), beta))
    denominator = 1.0 + torch.einsum("p,pi->i", beta, differences)
    reference_composition = z / denominator
    phase_compositions = torch.cat(
        (reference_composition[None, :], k_values * reference_composition[None, :]),
        dim=0,
    )
    return fractions, phase_compositions, iteration, converged


def _default_multiphase_k(model: StateModel, state: ChemicalState) -> Tensor:
    required = ("critical_temperature", "critical_pressure", "acentric_factor", "names")
    if not all(hasattr(model, item) for item in required):
        raise ValueError("initial K values are required for this model")
    components = cast(ComponentSet, model)
    vapor_k = wilson_k_values(components, state.temperature, state.pressure)
    names = components.names
    aqueous_k = torch.full_like(vapor_k, 1.0e-5)
    if "water" in names:
        aqueous_k[names.index("water")] = 1.0e5
    else:
        aqueous_k = vapor_k.reciprocal()
    return torch.stack((vapor_k, aqueous_k))


def _equilibrium_residual(
    model: StateModel,
    state: ChemicalState,
    log_k: Tensor,
) -> tuple[Tensor, Tensor, Tensor, int, bool]:
    fractions, compositions, rr_iterations, rr_converged = solve_generalized_rachford_rice(
        state.composition,
        torch.exp(log_k),
        tolerance=1.0e-12,
    )
    if not rr_converged:
        return torch.full_like(log_k, torch.inf), fractions, compositions, rr_iterations, False
    normalized = compositions / compositions.sum(dim=1, keepdim=True)
    log_phi_reference = model.log_fugacity_coefficients(
        state.temperature,
        state.pressure,
        normalized[0],
        "liquid",
    )
    targets = []
    for phase_index in range(1, normalized.shape[0]):
        phase_kind: PhaseKind = "vapor" if phase_index == 1 else "liquid"
        log_phi = model.log_fugacity_coefficients(
            state.temperature,
            state.pressure,
            normalized[phase_index],
            phase_kind,
        )
        targets.append(log_phi_reference - log_phi)
    residual = log_k - torch.stack(targets)
    return residual, fractions, normalized, rr_iterations, True


def multiphase_flash(
    model: StateModel,
    state: ChemicalState,
    *,
    initial_k_values: Tensor | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 100,
) -> FlashResult:
    """Solve a fixed-phase-count PT flash by generalized substitution.

    Parameters
    ----------
    model
        Homogeneous-state model providing phase fugacity coefficients and
        properties.
    state
        Feed state with temperature in K, pressure in Pa, and a
        one-dimensional overall mole-fraction vector.
    initial_k_values
        Positive composition ratios to the reference liquid phase, with shape
        ``(nphases - 1, ncomponents)``. Omitting this argument requests a
        three-phase heuristic containing vapor-like and aqueous-like trials.
    tolerance
        Convergence threshold for the maximum absolute log-fugacity residual.
    max_iterations
        Maximum generalized substitution/Newton iterations.

    Returns
    -------
    FlashResult
        Fractions, phase properties, convergence status, fugacity residual,
        and generalized material-balance diagnostics.

    Raises
    ------
    ValueError
        If the composition is batched or default K values cannot be generated.

    Warns
    -----
    ExperimentalModelWarning
        On every call, because automatic phase discovery is not yet a global
        stability algorithm.
    ConvergenceWarning
        If the fixed-phase-count equilibrium iterations do not converge.

    Notes
    -----
    The number of phases is fixed by ``initial_k_values``; this routine does
    not prove that the selected phase count is globally stable. For VLL or VLW
    calculations, supply physically informed initial ratios and inspect
    ``FlashResult.converged``, the residual, phase fractions, material balance,
    and an independent stability analysis.
    """
    warnings.warn(
        "automatic multiphase phase discovery is experimental; inspect stability "
        "and material-balance diagnostics",
        ExperimentalModelWarning,
        stacklevel=2,
    )
    if state.composition.ndim != 1:
        raise ValueError("multiphase_flash currently accepts one composition vector")
    log_k = torch.log(
        _default_multiphase_k(model, state) if initial_k_values is None else initial_k_values
    )
    converged = False
    residual_norm = torch.tensor(torch.inf, dtype=log_k.dtype, device=log_k.device)
    fractions = torch.empty(0, dtype=log_k.dtype, device=log_k.device)
    compositions = torch.empty(0, dtype=log_k.dtype, device=log_k.device)
    rr_iterations = 0
    newton_steps = 0

    for iteration in range(1, max_iterations + 1):
        residual, fractions, compositions, rr_iterations, rr_converged = _equilibrium_residual(
            model,
            state,
            log_k,
        )
        if not rr_converged:
            log_k *= 0.8
            continue
        residual_norm = residual.abs().max()
        if float(residual_norm) <= tolerance:
            converged = True
            break

        # A full Jacobian is worthwhile only for three or more phases. It is
        # generated through the generalized material balance and fugacity
        # calculations by PyTorch, following Michelsen's Newton acceleration
        # principle without coding model-specific derivatives.
        if log_k.shape[0] > 1 and iteration >= 3:
            try:
                jacobian = torch.func.jacrev(
                    lambda current_log_k: _equilibrium_residual(
                        model,
                        state,
                        current_log_k,
                    )[0]
                )(log_k)
                flattened = jacobian.reshape(log_k.numel(), log_k.numel())
                step = torch.linalg.solve(flattened, -residual.reshape(-1)).reshape_as(log_k)
                factor = 1.0
                for _ in range(12):
                    candidate = log_k + factor * step
                    (
                        candidate_residual,
                        candidate_fractions,
                        candidate_compositions,
                        candidate_rr_iterations,
                        candidate_rr_converged,
                    ) = _equilibrium_residual(model, state, candidate)
                    candidate_norm = candidate_residual.detach().abs().max()
                    if (
                        candidate_rr_converged
                        and bool(torch.isfinite(candidate_norm))
                        and float(candidate_norm) < float(residual_norm.detach())
                    ):
                        log_k = candidate
                        fractions = candidate_fractions
                        compositions = candidate_compositions
                        rr_iterations = candidate_rr_iterations
                        newton_steps += 1
                        break
                    factor *= 0.5
                else:
                    log_k = log_k - (0.5 if iteration < 20 else 0.2) * residual
                continue
            except torch.linalg.LinAlgError:
                pass
        log_k = log_k - (0.5 if iteration < 20 else 0.2) * residual

    phases = []
    for phase_index, composition in enumerate(compositions):
        kind: PhaseKind = "vapor" if phase_index == 1 else "liquid"
        phase_state = ChemicalState(state.temperature, state.pressure, composition)
        phases.append(phase_properties(model, phase_state, kind, caloric=False))
    if not converged:
        warnings.warn(
            f"multiphase flash did not converge in {max_iterations} iterations",
            ConvergenceWarning,
            stacklevel=2,
        )
    identified_phases = identify_flash_phases(tuple(phases))
    return FlashResult(
        fractions,
        identified_phases,
        converged,
        iteration,
        residual_norm,
        converged,
        {
            "generalized_rachford_rice_iterations": rr_iterations,
            "autodiff_newton_steps": newton_steps,
        },
    )
