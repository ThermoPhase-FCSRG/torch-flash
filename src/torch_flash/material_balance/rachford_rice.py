"""Safeguarded Rachford-Rice solvers.

The bounds span the interval between the nearest denominator singularities,
so the solver also supports the mathematically useful negative flash.

References
----------
H. H. Rachford and J. D. Rice (1952), doi:10.2118/952327-G.
C. F. Leibovici and J. Neoschil (1992),
doi:10.1016/0378-3812(92)85069-K.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from torch_flash.types import RachfordRiceResult, normalize_composition


def _validate_inputs(composition: Tensor, k_values: Tensor) -> tuple[Tensor, Tensor]:
    z = normalize_composition(composition)
    if z.shape != k_values.shape:
        raise ValueError("composition and K values must have the same shape")
    if not torch.is_floating_point(k_values):
        k_values = k_values.to(z.dtype)
    if not torch.isfinite(k_values).all() or bool((k_values <= 0.0).any()):
        raise ValueError("K values must be finite and strictly positive")
    kmin = k_values.amin(dim=-1)
    kmax = k_values.amax(dim=-1)
    if bool(((kmin >= 1.0) | (kmax <= 1.0)).any()):
        raise ValueError("a finite Rachford-Rice root requires Kmin < 1 < Kmax")
    return z, k_values


def rachford_rice(
    composition: Tensor,
    k_values: Tensor,
    *,
    tolerance: float | None = None,
    max_iterations: int = 100,
) -> RachfordRiceResult:
    """Solve the two-phase material balance with safeguarded Newton steps.

    Leading dimensions are treated as independent batches. Computation remains
    on the input device and is differentiable through the executed iterations.
    """
    z, k = _validate_inputs(composition, k_values)
    eps = torch.finfo(z.dtype).eps
    tol = 8.0 * eps if tolerance is None else tolerance
    km1 = k - 1.0
    lower = 1.0 / (1.0 - k.amax(dim=-1))
    upper = 1.0 / (1.0 - k.amin(dim=-1))
    vapor = 0.5 * (lower + upper)
    converged = torch.zeros_like(vapor, dtype=torch.bool)
    residual = torch.full_like(vapor, torch.inf)
    iterations = 0

    for _iterations in range(1, max_iterations + 1):
        denominator = 1.0 + vapor[..., None] * km1
        terms = km1 / denominator
        residual = torch.sum(z * terms, dim=-1)
        derivative = -torch.sum(z * terms.square(), dim=-1)
        converged = residual.abs() <= tol
        if bool(converged.all()):
            break

        lower = torch.where(residual > 0.0, vapor, lower)
        upper = torch.where(residual < 0.0, vapor, upper)
        newton = vapor - residual / derivative
        midpoint = 0.5 * (lower + upper)
        admissible = torch.isfinite(newton) & (newton > lower) & (newton < upper)
        candidate = torch.where(admissible, newton, midpoint)
        vapor = torch.where(converged, vapor, candidate)

    iterations = _iterations
    denominator = 1.0 + vapor[..., None] * km1
    liquid_composition = z / denominator
    vapor_composition = k * liquid_composition
    liquid = 1.0 - vapor
    return RachfordRiceResult(
        vapor,
        liquid,
        liquid_composition,
        vapor_composition,
        iterations,
        converged,
        residual,
    )


def rachford_rice_numpy(
    composition: NDArray[np.float64],
    k_values: NDArray[np.float64],
    *,
    tolerance: float = 1.0e-15,
    max_iterations: int = 100,
) -> tuple[int, NDArray[np.float64], NDArray[np.float64], float, float]:
    """NumPy compatibility wrapper matching the Whitson contest signature.

    This exact-compatibility path delegates to the MIT-licensed
    ``chemicals`` dependency. Its implementation applies double-double
    arithmetic to the transformed/bounded formulation of Leibovici and Neoschil,
    *Fluid Phase Equilibria* 74 (1992), 303-308,
    doi:10.1016/0378-3812(92)85069-K. The dependency notice is retained in
    ``THIRD_PARTY_NOTICES.md``.
    """
    from chemicals.rachford_rice import (
        Rachford_Rice_solution_Leibovici_Neoschil_dd,
    )

    z = np.asarray(composition, dtype=np.float64)
    k = np.asarray(k_values, dtype=np.float64)
    if z.ndim != 1 or z.shape != k.shape:
        raise ValueError("the NumPy wrapper accepts equally sized one-dimensional arrays")
    z = z / z.sum()
    if np.any(k <= 0.0) or np.min(k) >= 1.0 or np.max(k) <= 1.0:
        raise ValueError("a finite root requires positive K values with Kmin < 1 < Kmax")
    del tolerance, max_iterations
    liquid, vapor, x, y = Rachford_Rice_solution_Leibovici_Neoschil_dd(z.tolist(), k.tolist())
    return (
        1,
        np.asarray(y, dtype=np.float64),
        np.asarray(x, dtype=np.float64),
        float(vapor),
        float(liquid),
    )
