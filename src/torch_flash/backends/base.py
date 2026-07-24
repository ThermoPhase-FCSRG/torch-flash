"""Shared contracts for optional, non-PyTorch validation backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCapabilities:
    """Explicit numerical and model-identity capability flags."""

    autodiff: bool
    gpu: bool
    fugacity_coefficients: bool
    exact_model: str
