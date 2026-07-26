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
