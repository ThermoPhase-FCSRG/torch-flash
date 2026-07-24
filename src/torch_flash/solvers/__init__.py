"""Reusable autodifferentiable nonlinear solvers."""

from .newton import NewtonResult, damped_newton

__all__ = ["NewtonResult", "damped_newton"]
