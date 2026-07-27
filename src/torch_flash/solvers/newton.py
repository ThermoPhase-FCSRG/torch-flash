"""Small dense Newton solver using PyTorch Jacobians."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class NewtonResult:
    """Solution and diagnostics from :func:`damped_newton`.

    Attributes
    ----------
    solution
        Final iterate.
    residual
        Residual vector evaluated at ``solution``.
    residual_norm
        Maximum absolute residual component.
    iterations
        Number of outer Newton iterations executed.
    converged
        Whether ``residual_norm <= tolerance``.
    """

    solution: Tensor
    residual: Tensor
    residual_norm: Tensor
    iterations: int
    converged: bool


@dataclass(frozen=True)
class BatchedNewtonResult:
    """Solutions and diagnostics from :func:`batched_damped_newton`.

    Attributes
    ----------
    solution
        Final variables with shape ``(batch, variables)``.
    residual
        Residual vectors evaluated at ``solution`` with the same shape.
    residual_norm
        Per-system infinity norms with shape ``(batch,)``.
    iterations
        Per-system outer iteration counts with shape ``(batch,)``.
    converged
        Per-system convergence flags with shape ``(batch,)``.
    """

    solution: Tensor
    residual: Tensor
    residual_norm: Tensor
    iterations: Tensor
    converged: Tensor


def batched_damped_newton(
    residual_function: Callable[..., Tensor],
    initial: Tensor,
    *residual_arguments: Tensor,
    tolerance: float = 1.0e-8,
    max_iterations: int = 50,
    max_line_search: int = 16,
    lower_bound: Tensor | None = None,
    upper_bound: Tensor | None = None,
) -> BatchedNewtonResult:
    """Solve independent small nonlinear systems in one PyTorch batch.

    Parameters
    ----------
    residual_function
        Batched mapping. Its first argument is the complete variable tensor;
        subsequent arguments are the complete tensors in
        ``residual_arguments``. It must return a tensor matching the variable
        shape and must not couple distinct leading-batch rows.
    initial
        Initial variables with shape ``(batch, variables)``.
    *residual_arguments
        Tensors whose leading dimension equals ``batch``. They are vectorized
        together with the variables.
    tolerance
        Absolute per-system infinity-norm convergence threshold.
    max_iterations
        Maximum damped-Newton iterations.
    max_line_search
        Maximum residual-decreasing backtracking trials per iteration.
    lower_bound, upper_bound
        Optional bounds broadcastable to ``initial``.

    Returns
    -------
    BatchedNewtonResult
        Per-system solutions, residuals, iteration counts, and convergence.

    Raises
    ------
    ValueError
        If shapes or numerical controls are invalid.

    Notes
    -----
    Exact per-system Jacobian blocks come from
    :func:`torch.func.jacrev` of each residual component summed across the
    independent batch. This evaluates only the block diagonal implied by
    independent systems rather than constructing and discarding cross-batch
    zero blocks. Gradients through model parameters and accepted Newton
    iterates are preserved. A singular direct solve falls back to a batched
    least-squares step. Failed systems remain explicit in the returned
    diagnostics.
    """
    if initial.ndim != 2 or initial.shape[0] == 0 or initial.shape[1] == 0:
        raise ValueError("batched Newton initial values must have shape (batch, variables)")
    if tolerance <= 0.0 or max_iterations <= 0 or max_line_search <= 0:
        raise ValueError("batched Newton controls must be positive")
    if any(
        argument.ndim == 0 or argument.shape[0] != initial.shape[0]
        for argument in residual_arguments
    ):
        raise ValueError("batched Newton residual arguments must share the leading batch size")

    variables = initial.clone()

    def broadcast_bound(bound: Tensor | None, name: str) -> Tensor | None:
        if bound is None:
            return None
        try:
            return torch.broadcast_to(bound, initial.shape)
        except RuntimeError as error:
            raise ValueError(f"batched Newton {name} bound is not broadcastable") from error

    lower = broadcast_bound(lower_bound, "lower")
    upper = broadcast_bound(upper_bound, "upper")

    def project(candidate: Tensor) -> Tensor:
        if lower is not None:
            candidate = torch.maximum(candidate, lower)
        if upper is not None:
            candidate = torch.minimum(candidate, upper)
        return candidate

    variables = project(variables)
    value = residual_function(variables, *residual_arguments)
    if value.shape != variables.shape:
        raise ValueError("batched Newton residuals must match the initial variable shape")

    batch_size = initial.shape[0]
    iterations = torch.zeros(batch_size, dtype=torch.int64, device=initial.device)
    residual_norm = value.abs().amax(dim=-1)
    converged = torch.isfinite(residual_norm) & (residual_norm <= tolerance)
    for iteration in range(1, max_iterations + 1):
        active = ~converged
        if bool(torch.all(~active).detach()):
            break

        jacobian = torch.stack(
            tuple(
                torch.func.jacrev(
                    lambda current, index=residual_index: residual_function(
                        current,
                        *residual_arguments,
                    )[:, index].sum()
                )(variables)
                for residual_index in range(initial.shape[1])
            ),
            dim=-2,
        )
        expected_jacobian_shape = (*initial.shape, initial.shape[1])
        if jacobian.shape != expected_jacobian_shape:
            raise ValueError("batched Newton Jacobian shape is inconsistent")
        direct_step, information = torch.linalg.solve_ex(
            jacobian,
            -value.unsqueeze(-1),
            check_errors=False,
        )
        direct_step = direct_step.squeeze(-1)
        least_squares_step = torch.linalg.lstsq(
            jacobian,
            -value.unsqueeze(-1),
        ).solution.squeeze(-1)
        direct_is_finite = torch.isfinite(direct_step).all(dim=-1)
        step = torch.where(
            ((information == 0) & direct_is_finite).unsqueeze(-1),
            direct_step,
            least_squares_step,
        )
        step = torch.where(
            torch.isfinite(step),
            step,
            -0.1 * torch.nan_to_num(value),
        )
        step = torch.where(active.unsqueeze(-1), step, torch.zeros_like(step))

        accepted = torch.zeros(batch_size, dtype=torch.bool, device=initial.device)
        accepted_variables = variables
        factor = torch.ones(batch_size, dtype=initial.dtype, device=initial.device)
        for _ in range(max_line_search):
            candidate = project(variables + factor.unsqueeze(-1) * step)
            candidate_value = residual_function(candidate, *residual_arguments)
            candidate_norm = candidate_value.abs().amax(dim=-1)
            improving = (
                active
                & ~accepted
                & torch.isfinite(candidate_value).all(dim=-1)
                & (candidate_norm < residual_norm)
            )
            accepted_variables = torch.where(
                improving.unsqueeze(-1),
                candidate,
                accepted_variables,
            )
            accepted = accepted | improving
            factor = torch.where(accepted, factor, 0.5 * factor)
            if bool(torch.all(accepted | ~active).detach()):
                break

        fallback_variables = project(variables + factor.unsqueeze(-1) * step)
        next_variables = torch.where(
            accepted.unsqueeze(-1),
            accepted_variables,
            fallback_variables,
        )
        variables = torch.where(active.unsqueeze(-1), next_variables, variables)
        value = residual_function(variables, *residual_arguments)
        iterations = torch.where(
            active,
            iterations.new_full((), iteration),
            iterations,
        )
        residual_norm = value.abs().amax(dim=-1)
        converged = torch.isfinite(residual_norm) & (residual_norm <= tolerance)

    residual_norm = value.abs().amax(dim=-1)
    converged = torch.isfinite(residual_norm) & (residual_norm <= tolerance)
    return BatchedNewtonResult(
        variables,
        value,
        residual_norm,
        iterations,
        converged,
    )


def damped_newton(
    residual_function: Callable[[Tensor], Tensor],
    initial: Tensor,
    *,
    tolerance: float = 1.0e-8,
    max_iterations: int = 50,
    max_line_search: int = 16,
    lower_bound: Tensor | None = None,
    upper_bound: Tensor | None = None,
    jacobian_refresh_interval: int = 1,
) -> NewtonResult:
    """Solve a square nonlinear system with autodiff and backtracking.

    Parameters
    ----------
    residual_function
        Differentiable mapping from the variable tensor to an equally sized
        residual tensor.
    initial
        Initial variable tensor; dtype and device are preserved.
    tolerance
        Absolute infinity-norm convergence threshold.
    max_iterations
        Maximum Newton iterations.
    max_line_search
        Maximum residual-decreasing backtracking trials per iteration.
    lower_bound, upper_bound
        Optional elementwise projection bounds broadcastable to ``initial``.
    jacobian_refresh_interval
        Recompute the exact autodiff Jacobian after this many accepted
        iterations. Values above one use rank-one Broyden updates between
        refreshes; the default preserves full Newton behavior.

    Returns
    -------
    NewtonResult
        Final iterate and explicit convergence diagnostics.

    Notes
    -----
    A singular square solve falls back to a least-squares step. Failure to
    converge is reported in the result and is never silently converted into a
    successful solve.
    """
    if jacobian_refresh_interval < 1:
        raise ValueError("Jacobian refresh interval must be positive")
    variables = initial.clone()

    def project(candidate: Tensor) -> Tensor:
        if lower_bound is not None:
            candidate = torch.maximum(candidate, lower_bound)
        if upper_bound is not None:
            candidate = torch.minimum(candidate, upper_bound)
        return candidate

    variables = project(variables)
    converged = False
    value = residual_function(variables)
    jacobian: Tensor | None = None
    accepted_since_refresh = jacobian_refresh_interval
    for _iteration in range(1, max_iterations + 1):
        residual_norm = value.abs().max()
        if float(residual_norm.detach()) <= tolerance:
            converged = True
            break
        if jacobian is None or accepted_since_refresh >= jacobian_refresh_interval:
            jacobian = torch.func.jacrev(residual_function)(variables)
            accepted_since_refresh = 0
        try:
            step = torch.linalg.solve(jacobian, -value)
        except torch.linalg.LinAlgError:
            step = torch.linalg.lstsq(jacobian, -value.unsqueeze(-1)).solution.squeeze(-1)
        if not bool(torch.isfinite(step).all()):
            step = -0.1 * torch.nan_to_num(value)
        accepted = False
        factor = 1.0
        for _ in range(max_line_search):
            candidate = project(variables + factor * step)
            candidate_value = residual_function(candidate)
            if bool(torch.isfinite(candidate_value.detach()).all()) and float(
                candidate_value.detach().abs().max()
            ) < float(residual_norm.detach()):
                accepted_step = candidate - variables
                residual_step = candidate_value - value
                denominator = torch.dot(accepted_step, accepted_step)
                if bool(
                    torch.isfinite(denominator)
                    & (denominator > torch.finfo(denominator.dtype).tiny)
                ):
                    jacobian = (
                        jacobian
                        + torch.outer(
                            residual_step - jacobian @ accepted_step,
                            accepted_step,
                        )
                        / denominator
                    )
                    accepted_since_refresh += 1
                else:
                    jacobian = None
                    accepted_since_refresh = jacobian_refresh_interval
                variables = candidate
                value = candidate_value
                accepted = True
                break
            factor *= 0.5
        if not accepted:
            variables = project(variables + factor * step)
            value = residual_function(variables)
            jacobian = None
            accepted_since_refresh = jacobian_refresh_interval
    residual_norm = value.abs().max()
    if float(residual_norm.detach()) <= tolerance:
        converged = True
    return NewtonResult(variables, value, residual_norm, _iteration, converged)
