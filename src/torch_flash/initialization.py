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
    """Return Wilson's dilute-solution equilibrium-ratio estimate."""
    return (
        components.critical_pressure
        / pressure[..., None]
        * torch.exp(
            5.373
            * (1.0 + components.acentric_factor)
            * (1.0 - components.critical_temperature / temperature[..., None])
        )
    )
