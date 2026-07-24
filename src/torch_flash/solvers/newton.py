"""Small dense Newton solver using PyTorch Jacobians."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class NewtonResult:
    """Solution and convergence diagnostics from :func:`damped_newton`."""

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
) -> NewtonResult:
    """Solve a square nonlinear system with autodiff and backtracking."""
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
    for _iteration in range(1, max_iterations + 1):
        residual_norm = value.abs().max()
        if float(residual_norm.detach()) <= tolerance:
            converged = True
            break
        jacobian = torch.func.jacrev(residual_function)(variables)
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
                variables = candidate
                value = candidate_value
                accepted = True
                break
            factor *= 0.5
        if not accepted:
            variables = project(variables + factor * step)
            value = residual_function(variables)
    residual_norm = value.abs().max()
    if float(residual_norm.detach()) <= tolerance:
        converged = True
    return NewtonResult(variables, value, residual_norm, _iteration, converged)
