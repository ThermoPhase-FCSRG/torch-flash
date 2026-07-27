"""Reusable autodifferentiable nonlinear solvers."""

from .newton import BatchedNewtonResult, NewtonResult, batched_damped_newton, damped_newton
from .trust_region import (
    BatchedTrustRegionResult,
    TrustRegionResult,
    minimize_batched_dense_trust_region,
    minimize_dense_trust_region,
)

__all__ = [
    "BatchedNewtonResult",
    "BatchedTrustRegionResult",
    "NewtonResult",
    "TrustRegionResult",
    "batched_damped_newton",
    "damped_newton",
    "minimize_batched_dense_trust_region",
    "minimize_dense_trust_region",
]
