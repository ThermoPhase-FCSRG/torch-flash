"""Autodifferentiable thermodynamic parameter estimation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from torch_flash.properties.state import StateModel
from torch_flash.types import PhaseKind, normalize_composition


@dataclass(frozen=True)
class FitResult:
    """Optimization history and final convergence diagnostics.

    Attributes
    ----------
    losses
        Scalar objective value after each Adam iteration.
    converged
        Whether successive objective values met the relative stopping test.
    iterations
        Number of optimizer iterations executed.
    final_loss
        Last objective value.

    Notes
    -----
    Optimizer convergence does not establish parameter identifiability or
    model validation. Inspect sensitivities, parameter correlations, and
    independent holdout behavior separately.
    """

    losses: tuple[float, ...]
    converged: bool
    iterations: int
    final_loss: float


def least_squares_loss(
    prediction: Tensor,
    observation: Tensor,
    *,
    scale: Tensor | float = 1.0,
    weights: Tensor | None = None,
) -> Tensor:
    """Return a dimensionless weighted mean-square residual.

    Parameters
    ----------
    prediction, observation
        Broadcast-compatible model predictions and observations in matching
        physical units.
    scale
        Positive residual scale in the same units. Scalar or broadcastable
        tensor.
    weights
        Optional nonnegative statistical weights, broadcastable to the
        residual shape.

    Returns
    -------
    Tensor
        Scalar mean of the squared, scaled residuals; weights enter through
        their square roots.
    """
    residual = (prediction - observation) / scale
    if weights is not None:
        residual = residual * torch.sqrt(weights)
    return torch.mean(residual.square())


def phase_equilibrium_residual(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    phase1_composition: Tensor,
    phase2_composition: Tensor,
    *,
    phase_kinds: tuple[PhaseKind, PhaseKind] = ("liquid", "vapor"),
) -> Tensor:
    """Return component log-fugacity equalities for measured phase pairs.

    Parameters
    ----------
    model
        Homogeneous-state fugacity model.
    temperature, pressure
        Scalar or batched temperatures in K and pressures in Pa.
    phase1_composition, phase2_composition
        Equally shaped, strictly positive phase mole fractions.
    phase_kinds
        Root request corresponding to each measured phase.

    Returns
    -------
    Tensor
        Dimensionless component residual
        ``ln(x1_i phi1_i) - ln(x2_i phi2_i)``.

    Raises
    ------
    ValueError
        If phase compositions differ in shape or are nonpositive/nonfinite.

    Notes
    -----
    Inputs may contain independent batched states.  The residual is
    dimensionless and is zero when every component has equal fugacity in the
    two requested phases.  Strictly positive compositions are required
    because the thermodynamic equality is evaluated in logarithmic form.
    """
    phase1 = normalize_composition(phase1_composition)
    phase2 = normalize_composition(phase2_composition)
    if phase1.shape != phase2.shape:
        raise ValueError("phase-equilibrium compositions must have equal shapes")
    if not bool(
        torch.isfinite(phase1).all()
        & torch.isfinite(phase2).all()
        & (phase1 > 0.0).all()
        & (phase2 > 0.0).all()
    ):
        raise ValueError("phase-equilibrium compositions must be finite and positive")
    return (
        torch.log(phase1)
        + model.log_fugacity_coefficients(
            temperature,
            pressure,
            phase1,
            phase_kinds[0],
        )
        - torch.log(phase2)
        - model.log_fugacity_coefficients(
            temperature,
            pressure,
            phase2,
            phase_kinds[1],
        )
    )


def fit_parameters(
    parameters: Iterable[nn.Parameter],
    closure: Callable[[], Tensor],
    *,
    learning_rate: float = 0.05,
    max_iterations: int = 500,
    tolerance: float = 1.0e-10,
) -> FitResult:
    """Fit arbitrary PyTorch thermodynamic parameters with Adam.

    Parameters
    ----------
    parameters
        Trainable tensors to pass to :class:`torch.optim.Adam`.
    closure
        Zero-argument function that recomputes one finite differentiable
        scalar loss.
    learning_rate
        Adam learning rate.
    max_iterations
        Maximum optimizer steps.
    tolerance
        Relative successive-loss stopping threshold.

    Returns
    -------
    FitResult
        Loss history and explicit stopping diagnostics.

    Raises
    ------
    ValueError
        If no parameters are supplied or the closure returns a nonfinite or
        nonscalar loss.

    Notes
    -----
    The closure must recompute and return a scalar differentiable loss. Bounds
    can be imposed by parameterizing physical values through sigmoid/softplus
    transforms in the model.
    """
    trainable = tuple(parameters)
    if not trainable:
        raise ValueError("at least one trainable parameter is required")
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    history: list[float] = []
    converged = False
    previous = torch.inf
    for _iteration in range(1, max_iterations + 1):
        optimizer.zero_grad()
        loss = closure()
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise ValueError("fitting closure must return one finite scalar loss")
        loss.backward()
        optimizer.step()
        current = float(loss.detach())
        history.append(current)
        if abs(float(previous) - current) <= tolerance * max(1.0, abs(current)):
            converged = True
            break
        previous = current
    return FitResult(tuple(history), converged, _iteration, history[-1])


__all__ = [
    "FitResult",
    "fit_parameters",
    "least_squares_loss",
    "phase_equilibrium_residual",
]
