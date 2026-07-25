"""Isolated CoolProp adapter for independent CPU comparisons."""

from __future__ import annotations

from typing import ClassVar, Literal

import torch
from torch import Tensor

from torch_flash.exceptions import ModelCapabilityError
from torch_flash.types import PhaseKind, normalize_composition

from .base import BackendCapabilities


class CoolPropBackend:
    """CoolProp ``AbstractState`` wrapper.

    This backend is intended for independent validation and frozen baselines.
    It is CPU-only and not differentiable. ``HEOS`` uses multifluid mixture
    machinery with GERG-style reducing/departure functions but is not labeled
    as the published GERG-2008 coefficient set. Exact GERG through REFPROP
    requires a licensed REFPROP installation configured by the user.

    Parameters
    ----------
    names
        Canonical component names in composition-axis order.
    backend
        CoolProp ``"HEOS"`` or user-configured ``"REFPROP"`` backend.

    Notes
    -----
    Inputs are copied through CPU scalars/lists. Returned tensors match the
    caller's dtype/device but are detached from PyTorch autograd.
    """

    _ALIASES: ClassVar[dict[str, str]] = {
        "carbon_dioxide": "CarbonDioxide",
        "n_butane": "n-Butane",
        "n_pentane": "n-Pentane",
        "n_hexane": "n-Hexane",
        "n_heptane": "n-Heptane",
        "n_octane": "n-Octane",
        "n_decane": "n-Decane",
    }

    def __init__(
        self,
        names: tuple[str, ...],
        *,
        backend: Literal["HEOS", "REFPROP"] = "HEOS",
    ) -> None:
        try:
            from CoolProp import CoolProp
        except ImportError as exc:
            raise ImportError(
                "CoolPropBackend requires the optional 'external' dependency"
            ) from exc
        self._cp = CoolProp
        self.names = names
        fluids = "&".join(self._ALIASES.get(name, name.title()) for name in names)
        self._state = CoolProp.AbstractState(backend, fluids)
        self.capabilities = BackendCapabilities(
            autodiff=False,
            gpu=False,
            fugacity_coefficients=True,
            exact_model="REFPROP-selected model" if backend == "REFPROP" else "CoolProp HEOS",
        )

    def _update(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind,
    ) -> None:
        if temperature.ndim or pressure.ndim or composition.ndim != 1:
            raise ValueError("CoolProp adapter accepts one scalar T-P state")
        x = normalize_composition(composition)
        self._state.set_mole_fractions(x.detach().cpu().tolist())
        if phase == "liquid":
            self._state.specify_phase(self._cp.iphase_liquid)
        elif phase == "vapor":
            self._state.specify_phase(self._cp.iphase_gas)
        else:
            self._state.unspecify_phase()
        self._state.update(
            self._cp.PT_INPUTS,
            float(pressure.detach()),
            float(temperature.detach()),
        )

    @staticmethod
    def _result(value: float, like: Tensor) -> Tensor:
        return torch.tensor(value, dtype=like.dtype, device=like.device)

    def select_z(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return CoolProp compressibility factor."""
        self._update(temperature, pressure, composition, phase)
        return self._result(self._state.compressibility_factor(), temperature)

    def molar_volume(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return CoolProp molar volume."""
        self._update(temperature, pressure, composition, phase)
        return self._result(1.0 / self._state.rhomolar(), temperature)

    def log_fugacity_coefficients(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return CoolProp log fugacity coefficients."""
        self._update(temperature, pressure, composition, phase)
        try:
            values = [self._state.fugacity_coefficient(index) for index in range(len(self.names))]
        except (AttributeError, ValueError) as exc:
            raise ModelCapabilityError(
                "selected CoolProp backend does not expose fugacity coefficients"
            ) from exc
        return torch.log(torch.tensor(values, dtype=temperature.dtype, device=temperature.device))
