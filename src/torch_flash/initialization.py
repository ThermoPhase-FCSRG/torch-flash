"""Phase-equilibrium initial estimates."""

from __future__ import annotations

import torch
from torch import Tensor

from torch_flash.components import ComponentSet


def wilson_k_values(
    components: ComponentSet,
    temperature: Tensor,
    pressure: Tensor,
) -> Tensor:
    """Evaluate Wilson initial vapor-to-liquid equilibrium ratios.

    Parameters
    ----------
    components
        Ordered critical temperatures, critical pressures, and acentric
        factors.
    temperature
        Temperature in K with arbitrary leading batch dimensions.
    pressure
        Positive pressure in Pa, broadcast-compatible with ``temperature``.

    Returns
    -------
    Tensor
        Positive dimensionless K-value estimates with shape
        ``batch_shape + (ncomponents,)``.

    Notes
    -----
    These are initialization estimates, not converged equilibrium ratios and
    not a phase-stability criterion.
    """
    return (
        components.critical_pressure
        / pressure[..., None]
        * torch.exp(
            5.373
            * (1.0 + components.acentric_factor)
            * (1.0 - components.critical_temperature / temperature[..., None])
        )
    )


__all__ = ["wilson_k_values"]
