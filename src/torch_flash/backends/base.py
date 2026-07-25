"""Shared contracts for optional, non-PyTorch validation backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCapabilities:
    """Capabilities declared by an optional validation backend.

    Attributes
    ----------
    autodiff
        Whether backend outputs participate in PyTorch automatic
        differentiation.
    gpu
        Whether the backend can execute on accelerator tensors.
    fugacity_coefficients
        Whether component fugacity coefficients are exposed.
    exact_model
        Model identity implemented by the backend, for matched-input
        verification rather than name-only comparison.
    """

    autodiff: bool
    gpu: bool
    fugacity_coefficients: bool
    exact_model: str
