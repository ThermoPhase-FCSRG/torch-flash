"""Small dense trust-region minimization with exact PyTorch derivatives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrustRegionResult:
    """Result and convergence diagnostics from dense trust-region minimization.

    Attributes
    ----------
    solution
        Final variable vector. Its dtype, device, and autodiff graph follow
        ``initial`` and the supplied objective.
    objective
        Scalar objective evaluated at ``solution``.
    gradient_norm
        Infinity norm of the final objective gradient.
    minimum_hessian_eigenvalue
        Smallest eigenvalue of the final symmetrized Hessian. A converged
        local minimum requires it to be nonnegative within dtype-aware
        roundoff.
    iterations
        Number of outer trust-region iterations.
    converged
        Whether ``gradient_norm`` met the requested tolerance.
    accepted_steps, rejected_steps
        Counts based on agreement between the exact objective reduction and
        its local quadratic model.
    trust_radius
        Final Euclidean trust-region radius.
    """

    solution: Tensor
    objective: Tensor
    gradient_norm: Tensor
    minimum_hessian_eigenvalue: Tensor
    iterations: int
    converged: bool
    accepted_steps: int
    rejected_steps: int
    trust_radius: Tensor


@dataclass(frozen=True)
class BatchedTrustRegionResult:
    """Independent dense trust-region minimizations executed in one batch.

    Attributes
    ----------
    solution
        Final variables with shape ``(batch, variables)``.
    objective
        Final scalar objective per independent state with shape ``(batch,)``.
    gradient_norm
        Final infinity-norm objective gradient per state.
    minimum_hessian_eigenvalue
        Smallest eigenvalue of each final symmetrized Hessian.
    iterations
        Outer trust-region iterations performed per state.
    converged
        Per-state flags requiring the gradient and local-curvature gates.
    accepted_steps, rejected_steps
        Per-state counts based on actual versus quadratic-model reduction.
    trust_radius
        Final Euclidean trust-region radius per state.
    """

    solution: Tensor
    objective: Tensor
    gradient_norm: Tensor
    minimum_hessian_eigenvalue: Tensor
    iterations: Tensor
    converged: Tensor
    accepted_steps: Tensor
    rejected_steps: Tensor
    trust_radius: Tensor


def _dense_trust_region_step(
    gradient: Tensor,
    hessian: Tensor,
    radius: Tensor,
    *,
    subproblem_tolerance: float,
    max_subproblem_iterations: int,
) -> Tensor:
    """Return the exact small dense quadratic trust-region step."""
    symmetric_hessian = 0.5 * (hessian + hessian.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_hessian)
    transformed_gradient = eigenvectors.mT @ gradient
    scale = torch.maximum(
        eigenvalues.detach().abs().amax(),
        eigenvalues.new_tensor(1.0),
    )
    spectral_tolerance = 32.0 * torch.finfo(eigenvalues.dtype).eps * scale
    minimum_eigenvalue = eigenvalues[0]

    if bool(minimum_eigenvalue.detach() > spectral_tolerance):
        newton_coordinates = -transformed_gradient / eigenvalues
        if bool(torch.linalg.vector_norm(newton_coordinates).detach() <= radius.detach()):
            return cast(Tensor, eigenvectors @ newton_coordinates)

    minimum_mask = torch.abs(eigenvalues - minimum_eigenvalue) <= spectral_tolerance
    minimum_gradient = transformed_gradient[minimum_mask]
    lower_shift = torch.clamp_min(-minimum_eigenvalue, 0.0)
    hard_case = bool(
        torch.linalg.vector_norm(minimum_gradient).detach() <= spectral_tolerance.detach()
    )
    if hard_case:
        regular = ~minimum_mask
        base_coordinates = torch.zeros_like(transformed_gradient)
        if bool(regular.any()):
            base_coordinates[regular] = -transformed_gradient[regular] / (
                eigenvalues[regular] + lower_shift
            )
        base_norm = torch.linalg.vector_norm(base_coordinates)
        if bool(base_norm.detach() <= radius.detach()):
            remaining = torch.sqrt(torch.clamp_min(radius.square() - base_norm.square(), 0.0))
            minimum_index = int(torch.nonzero(minimum_mask)[0])
            base_coordinates[minimum_index] = remaining
            return cast(Tensor, eigenvectors @ base_coordinates)

    shift_margin = torch.maximum(
        spectral_tolerance,
        torch.finfo(eigenvalues.dtype).eps * scale,
    )
    lower = lower_shift + shift_margin

    def shifted_norm(shift: Tensor) -> Tensor:
        return cast(
            Tensor,
            torch.linalg.vector_norm(transformed_gradient / (eigenvalues + shift)),
        )

    upper = torch.maximum(lower + scale, eigenvalues.new_tensor(1.0))
    for _ in range(max_subproblem_iterations):
        if bool(shifted_norm(upper).detach() <= radius.detach()):
            break
        upper = 2.0 * upper

    shift = 0.5 * (lower + upper)
    for _ in range(max_subproblem_iterations):
        norm = shifted_norm(shift)
        relative_error = torch.abs(norm - radius) / torch.clamp_min(radius, 1.0)
        if bool(relative_error.detach() <= subproblem_tolerance):
            break
        if bool(norm.detach() > radius.detach()):
            lower = shift
        else:
            upper = shift
        shift = 0.5 * (lower + upper)
    coordinates = -transformed_gradient / (eigenvalues + shift)
    return cast(Tensor, eigenvectors @ coordinates)


def minimize_dense_trust_region(
    objective: Callable[[Tensor], Tensor],
    initial: Tensor,
    *,
    is_feasible: Callable[[Tensor], bool] | None = None,
    initial_radius: float | Tensor | None = None,
    maximum_radius: float | Tensor | None = None,
    gradient_tolerance: float = 1.0e-9,
    max_iterations: int = 100,
    acceptance_threshold: float = 0.0,
    subproblem_tolerance: float = 1.0e-8,
    max_subproblem_iterations: int = 80,
) -> TrustRegionResult:
    """Minimize a scalar objective with an exact small dense trust region.

    This operation implements the restricted-step formulation used for phase
    stability and multiphase Gibbs minimization by M. Petitfrere and
    D. V. Nichita, "Robust and efficient Trust-Region based stability
    analysis and multiphase flash calculations", *Fluid Phase Equilibria*
    362 (2014), 51-68, sections 3.1-3.4,
    doi:10.1016/j.fluid.2013.08.039. The dense quadratic subproblem follows
    the Moré-Sorensen eigenvalue-shift conditions. It is intended for the
    small systems typical of compositional phase calculations, not large
    sparse optimization.

    Parameters
    ----------
    objective
        Differentiable mapping from one variable vector to one finite scalar.
        Gradients and exact Hessians are assembled with PyTorch autodiff.
    initial
        One-dimensional starting vector.
    is_feasible
        Optional predicate for hard variable bounds. Infeasible trial steps
        are rejected without evaluating ``objective``. The initial vector
        must be feasible.
    initial_radius
        Positive initial Euclidean radius. The paper's flash choice
        ``||initial|| / 10`` is used when omitted, with a finite floor for a
        zero initial vector.
    maximum_radius
        Positive maximum radius. Ten times the initial radius is used when
        omitted.
    gradient_tolerance
        Convergence threshold for the infinity norm of the objective gradient.
    max_iterations
        Maximum accepted-or-rejected outer iterations.
    acceptance_threshold
        Minimum actual-to-predicted reduction ratio for accepting a step.
        Zero matches the phase-equilibrium algorithm in section 3.4.
    subproblem_tolerance
        Relative radius tolerance for the shifted-Hessian subproblem.
    max_subproblem_iterations
        Maximum safeguarded scalar shift iterations.

    Returns
    -------
    TrustRegionResult
        Final variables, objective and gradient diagnostics, iteration and
        step counts, and the final trust radius.

    Raises
    ------
    ValueError
        If the initial tensor is not a finite vector, controls are invalid, or
        the objective is not a finite scalar.

    Notes
    -----
    The method is a local minimizer. It does not prove that a tangent-plane or
    Gibbs minimum is global; phase calculations still require multiple starts,
    material-balance checks, and stability tests. A negative or poorly
    predicted step is rejected rather than returned as convergence.
    """
    if initial.ndim != 1 or initial.numel() == 0 or not bool(torch.isfinite(initial).all()):
        raise ValueError("dense trust-region initial values must be one finite vector")
    if is_feasible is not None and not is_feasible(initial):
        raise ValueError("dense trust-region initial values must be feasible")
    if (
        gradient_tolerance <= 0.0
        or max_iterations <= 0
        or not 0.0 <= acceptance_threshold < 0.25
        or subproblem_tolerance <= 0.0
        or max_subproblem_iterations <= 0
    ):
        raise ValueError("dense trust-region controls are invalid")

    variables = initial.clone()
    initial_norm = torch.linalg.vector_norm(variables.detach())
    default_radius = torch.clamp_min(initial_norm / 10.0, 0.1)
    radius = torch.as_tensor(
        default_radius if initial_radius is None else initial_radius,
        dtype=variables.dtype,
        device=variables.device,
    )
    maximum = torch.as_tensor(
        10.0 * radius if maximum_radius is None else maximum_radius,
        dtype=variables.dtype,
        device=variables.device,
    )
    if (
        radius.ndim != 0
        or maximum.ndim != 0
        or not bool(
            torch.isfinite(radius) & torch.isfinite(maximum) & (radius > 0.0) & (maximum >= radius)
        )
    ):
        raise ValueError("dense trust-region radii must be finite positive scalars")

    value = objective(variables)
    if value.ndim != 0 or not bool(torch.isfinite(value)):
        raise ValueError("dense trust-region objective must return one finite scalar")
    accepted_steps = 0
    rejected_steps = 0
    converged = False
    gradient = torch.func.grad(objective)(variables)
    gradient_norm = gradient.abs().max()
    minimum_hessian_eigenvalue = variables.new_tensor(torch.nan)

    for _iteration in range(1, max_iterations + 1):
        gradient = torch.func.grad(objective)(variables)
        gradient_norm = gradient.abs().max()
        hessian = torch.func.hessian(objective)(variables)
        symmetric_hessian = 0.5 * (hessian + hessian.mT)
        hessian_eigenvalues = torch.linalg.eigvalsh(symmetric_hessian)
        minimum_hessian_eigenvalue = hessian_eigenvalues[0]
        curvature_scale = torch.maximum(
            hessian_eigenvalues.detach().abs().amax(),
            hessian_eigenvalues.new_tensor(1.0),
        )
        curvature_tolerance = torch.finfo(hessian_eigenvalues.dtype).eps ** 0.5 * curvature_scale
        if bool(
            (gradient_norm.detach() <= gradient_tolerance)
            & (minimum_hessian_eigenvalue.detach() >= -curvature_tolerance.detach())
        ):
            converged = True
            break
        step = _dense_trust_region_step(
            gradient,
            hessian,
            radius,
            subproblem_tolerance=subproblem_tolerance,
            max_subproblem_iterations=max_subproblem_iterations,
        )
        predicted_reduction = -(torch.dot(gradient, step) + 0.5 * torch.dot(step, hessian @ step))
        candidate = variables + step
        candidate_is_feasible = is_feasible is None or is_feasible(candidate)
        candidate_value = (
            objective(candidate) if candidate_is_feasible else value.new_tensor(torch.inf)
        )
        actual_reduction = value - candidate_value
        valid_prediction = bool(
            candidate_is_feasible
            and torch.isfinite(candidate_value).detach()
            & torch.isfinite(predicted_reduction).detach()
            & (predicted_reduction.detach() > 0.0)
        )
        ratio = (
            actual_reduction / predicted_reduction
            if valid_prediction
            else value.new_tensor(-torch.inf)
        )
        step_norm = torch.linalg.vector_norm(step)
        reduction_resolution = (
            64.0
            * torch.finfo(value.dtype).eps
            * torch.maximum(value.detach().abs(), value.new_tensor(1.0))
        )
        roundoff_stationary_step = False
        if (
            valid_prediction
            and bool(predicted_reduction.detach() <= reduction_resolution)
            and bool(actual_reduction.detach().abs() <= reduction_resolution)
        ):
            candidate_gradient = torch.func.grad(objective)(candidate)
            roundoff_stationary_step = bool(
                torch.isfinite(candidate_gradient).all().detach()
                & (candidate_gradient.detach().abs().max() < gradient_norm.detach())
            )
        if roundoff_stationary_step:
            pass
        elif not valid_prediction or bool(ratio.detach() < 0.25):
            radius = torch.clamp_min(radius / 4.0, torch.finfo(radius.dtype).eps)
        elif bool((ratio.detach() > 0.75) & (step_norm.detach() >= 0.99 * radius.detach())):
            radius = torch.minimum(2.0 * radius, maximum)

        if valid_prediction and (
            roundoff_stationary_step or bool(ratio.detach() > acceptance_threshold)
        ):
            variables = candidate
            value = candidate_value
            accepted_steps += 1
        else:
            rejected_steps += 1
    gradient = torch.func.grad(objective)(variables)
    gradient_norm = gradient.abs().max()
    final_hessian = torch.func.hessian(objective)(variables)
    final_hessian_eigenvalues = torch.linalg.eigvalsh(0.5 * (final_hessian + final_hessian.mT))
    minimum_hessian_eigenvalue = final_hessian_eigenvalues[0]
    final_curvature_scale = torch.maximum(
        final_hessian_eigenvalues.detach().abs().amax(),
        final_hessian_eigenvalues.new_tensor(1.0),
    )
    final_curvature_tolerance = (
        torch.finfo(final_hessian_eigenvalues.dtype).eps ** 0.5 * final_curvature_scale
    )
    converged = converged or bool(
        (gradient_norm.detach() <= gradient_tolerance)
        & (minimum_hessian_eigenvalue.detach() >= -final_curvature_tolerance.detach())
    )
    return TrustRegionResult(
        variables,
        objective(variables),
        gradient_norm,
        minimum_hessian_eigenvalue,
        _iteration,
        converged,
        accepted_steps,
        rejected_steps,
        radius,
    )


def minimize_batched_dense_trust_region(
    objective: Callable[..., Tensor],
    initial: Tensor,
    *objective_arguments: Tensor,
    initial_radius: float | Tensor = 0.5,
    maximum_radius: float | Tensor = 5.0,
    gradient_tolerance: float = 1.0e-9,
    max_iterations: int = 100,
    acceptance_threshold: float = 0.0,
    subproblem_tolerance: float = 1.0e-8,
    max_subproblem_iterations: int = 80,
) -> BatchedTrustRegionResult:
    """Minimize independent small dense objectives in one PyTorch batch.

    Parameters
    ----------
    objective
        Mapping from the complete variable tensor and optional complete
        ``objective_arguments`` to one scalar objective per batch row. Rows
        must be mathematically independent.
    initial
        Finite initial variables with shape ``(batch, variables)``.
    *objective_arguments
        Tensors sharing the leading batch size.
    initial_radius, maximum_radius
        Positive scalars or per-state tensors with shape ``(batch,)``.
    gradient_tolerance, max_iterations, acceptance_threshold,
    subproblem_tolerance, max_subproblem_iterations
        Controls with the same meaning as
        :func:`minimize_dense_trust_region`.

    Returns
    -------
    BatchedTrustRegionResult
        Per-state solutions and convergence, curvature, iteration, step, and
        radius diagnostics.

    Raises
    ------
    ValueError
        If batch shapes, radii, controls, or objective values are invalid.

    Notes
    -----
    PyTorch evaluates the objective, gradients, and exact block-diagonal
    Hessians over the complete state batch. Only the tiny Moré-Sorensen
    spectral subproblem is dispatched row by row. This avoids coupling
    independent phase states while keeping the thermodynamic kernels
    vectorized on CPU or accelerator devices.
    """
    if (
        initial.ndim != 2
        or initial.shape[0] == 0
        or initial.shape[1] == 0
        or not bool(torch.isfinite(initial).all())
    ):
        raise ValueError("batched trust-region initial values must have shape (batch, variables)")
    if any(
        argument.ndim == 0 or argument.shape[0] != initial.shape[0]
        for argument in objective_arguments
    ):
        raise ValueError("batched trust-region arguments must share the leading batch size")
    if (
        gradient_tolerance <= 0.0
        or max_iterations <= 0
        or not 0.0 <= acceptance_threshold < 0.25
        or subproblem_tolerance <= 0.0
        or max_subproblem_iterations <= 0
    ):
        raise ValueError("batched trust-region controls are invalid")

    batch_size, variable_count = initial.shape
    variables = initial.clone()

    def batch_scalar(value: float | Tensor, name: str) -> Tensor:
        tensor = torch.as_tensor(value, dtype=initial.dtype, device=initial.device)
        try:
            tensor = torch.broadcast_to(tensor, (batch_size,)).clone()
        except RuntimeError as error:
            raise ValueError(f"batched trust-region {name} is not broadcastable") from error
        return tensor

    radius = batch_scalar(initial_radius, "initial radius")
    maximum = batch_scalar(maximum_radius, "maximum radius")
    if not bool(
        torch.isfinite(radius).all()
        & torch.isfinite(maximum).all()
        & (radius > 0.0).all()
        & (maximum >= radius).all()
    ):
        raise ValueError("batched trust-region radii must be finite and positive")

    def values(current: Tensor) -> Tensor:
        result = objective(current, *objective_arguments)
        if result.shape != (batch_size,):
            raise ValueError("batched trust-region objective must return shape (batch,)")
        return result

    def derivatives(current: Tensor) -> tuple[Tensor, Tensor]:
        def total(candidate: Tensor) -> Tensor:
            return values(candidate).sum()

        gradient_function = torch.func.grad(total)
        gradient = gradient_function(current)
        hessian = torch.stack(
            tuple(
                torch.func.grad(
                    lambda candidate, index=index: gradient_function(candidate)[:, index].sum()
                )(current)
                for index in range(variable_count)
            ),
            dim=-2,
        )
        return gradient, 0.5 * (hessian + hessian.mT)

    value = values(variables)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("batched trust-region objective values must be finite")
    iterations = torch.zeros(batch_size, dtype=torch.int64, device=initial.device)
    accepted_steps = torch.zeros_like(iterations)
    rejected_steps = torch.zeros_like(iterations)
    converged = torch.zeros(batch_size, dtype=torch.bool, device=initial.device)

    for iteration in range(1, max_iterations + 1):
        gradient, hessian = derivatives(variables)
        gradient_norm = gradient.abs().amax(dim=-1)
        hessian_eigenvalues = torch.linalg.eigvalsh(hessian)
        minimum_hessian_eigenvalue = hessian_eigenvalues[:, 0]
        curvature_scale = torch.maximum(
            hessian_eigenvalues.detach().abs().amax(dim=-1),
            hessian_eigenvalues.new_ones((batch_size,)),
        )
        curvature_tolerance = torch.finfo(initial.dtype).eps ** 0.5 * curvature_scale
        converged = (
            torch.isfinite(gradient_norm)
            & (gradient_norm <= gradient_tolerance)
            & (minimum_hessian_eigenvalue >= -curvature_tolerance)
        )
        active = ~converged
        if not bool(active.any().detach()):
            break

        step = torch.stack(
            tuple(
                _dense_trust_region_step(
                    gradient[index],
                    hessian[index],
                    radius[index],
                    subproblem_tolerance=subproblem_tolerance,
                    max_subproblem_iterations=max_subproblem_iterations,
                )
                if bool(active[index].detach())
                else torch.zeros_like(gradient[index])
                for index in range(batch_size)
            )
        )
        predicted_reduction = -(
            torch.einsum("bi,bi->b", gradient, step)
            + 0.5 * torch.einsum("bi,bij,bj->b", step, hessian, step)
        )
        candidate = variables + step
        candidate_value = values(candidate)
        actual_reduction = value - candidate_value
        valid_prediction = (
            active
            & torch.isfinite(candidate_value)
            & torch.isfinite(predicted_reduction)
            & (predicted_reduction > 0.0)
        )
        ratio = torch.where(
            valid_prediction,
            actual_reduction / predicted_reduction,
            actual_reduction.new_full((), -torch.inf),
        )
        step_norm = torch.linalg.vector_norm(step, dim=-1)
        reduction_resolution = (
            64.0
            * torch.finfo(value.dtype).eps
            * torch.maximum(value.detach().abs(), value.new_ones((batch_size,)))
        )
        roundoff_candidate = (
            valid_prediction
            & (predicted_reduction <= reduction_resolution)
            & (actual_reduction.abs() <= reduction_resolution)
        )
        if bool(roundoff_candidate.any().detach()):
            candidate_gradient, _ = derivatives(candidate)
            roundoff_stationary_step = (
                roundoff_candidate
                & torch.isfinite(candidate_gradient).all(dim=-1)
                & (candidate_gradient.detach().abs().amax(dim=-1) < gradient_norm.detach())
            )
        else:
            roundoff_stationary_step = torch.zeros_like(active)

        shrink = active & ~roundoff_stationary_step & (~valid_prediction | (ratio < 0.25))
        grow = active & ~shrink & (ratio > 0.75) & (step_norm >= 0.99 * radius)
        radius = torch.where(
            shrink,
            torch.clamp_min(radius / 4.0, torch.finfo(radius.dtype).eps),
            torch.where(grow, torch.minimum(2.0 * radius, maximum), radius),
        )
        accepted = (
            active & valid_prediction & (roundoff_stationary_step | (ratio > acceptance_threshold))
        )
        variables = torch.where(accepted.unsqueeze(-1), candidate, variables)
        value = values(variables)
        iterations = torch.where(active, iterations.new_full((), iteration), iterations)
        accepted_steps = accepted_steps + accepted.to(dtype=torch.int64)
        rejected_steps = rejected_steps + (active & ~accepted).to(dtype=torch.int64)

    gradient, hessian = derivatives(variables)
    gradient_norm = gradient.abs().amax(dim=-1)
    hessian_eigenvalues = torch.linalg.eigvalsh(hessian)
    minimum_hessian_eigenvalue = hessian_eigenvalues[:, 0]
    curvature_scale = torch.maximum(
        hessian_eigenvalues.detach().abs().amax(dim=-1),
        hessian_eigenvalues.new_ones((batch_size,)),
    )
    curvature_tolerance = torch.finfo(initial.dtype).eps ** 0.5 * curvature_scale
    converged = (
        torch.isfinite(gradient_norm)
        & (gradient_norm <= gradient_tolerance)
        & (minimum_hessian_eigenvalue >= -curvature_tolerance)
    )
    return BatchedTrustRegionResult(
        variables,
        values(variables),
        gradient_norm,
        minimum_hessian_eigenvalue,
        iterations,
        converged,
        accepted_steps,
        rejected_steps,
        radius,
    )


__all__ = [
    "BatchedTrustRegionResult",
    "TrustRegionResult",
    "minimize_batched_dense_trust_region",
    "minimize_dense_trust_region",
]
