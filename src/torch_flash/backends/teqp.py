"""NIST teqp adapter for independent PR, GERG-2008, and EOS-CG comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.constants import R
from torch_flash.exceptions import ConvergenceError, InvalidStateError
from torch_flash.types import PhaseKind, normalize_composition

from .base import BackendCapabilities


class TeqpBackend:
    """Wrap a teqp residual model as a homogeneous-state model.

    teqp is deliberately optional and all values cross a CPU/NumPy boundary,
    so this adapter is for validation rather than differentiable production
    calculations. The GERG constructor uses the published hard-coded
    GERG-2008 residual coefficient set shipped by teqp.

    Parameters
    ----------
    names
        Canonical component names in composition-axis order.
    model
        Constructed teqp residual model object.
    exact_model
        Explicit coefficient/model identity reported through capabilities.

    Notes
    -----
    Inputs cross a CPU/NumPy boundary. Returned tensors match the caller's
    dtype/device but are not differentiable.
    """

    _GERG_NAMES: ClassVar[dict[str, str]] = {
        "carbon_dioxide": "carbondioxide",
        "n_butane": "n-butane",
        "n_pentane": "n-pentane",
        "n_hexane": "n-hexane",
        "n_heptane": "n-heptane",
        "n_octane": "n-octane",
        "n_decane": "n-decane",
    }
    _EOSCG_NAMES: ClassVar[dict[str, str]] = {
        "carbon_dioxide": "CarbonDioxide",
        "water": "Water",
        "nitrogen": "Nitrogen",
        "oxygen": "Oxygen",
        "argon": "Argon",
        "carbon_monoxide": "CarbonMonoxide",
    }

    def __init__(self, names: tuple[str, ...], model: Any, *, exact_model: str) -> None:
        self.names = names
        self._model = model
        self.capabilities = BackendCapabilities(
            autodiff=False,
            gpu=False,
            fugacity_coefficients=True,
            exact_model=exact_model,
        )

    @classmethod
    def canonical_peng_robinson(cls, components: ComponentSet) -> TeqpBackend:
        """Construct teqp's canonical Peng-Robinson model."""
        try:
            import teqp
        except ImportError as exc:
            raise ImportError("TeqpBackend requires the optional 'external' dependency") from exc
        model = teqp.canonical_PR(
            components.critical_temperature.detach().cpu().numpy(),
            components.critical_pressure.detach().cpu().numpy(),
            components.acentric_factor.detach().cpu().numpy(),
        )
        return cls(components.names, model, exact_model="teqp canonical Peng-Robinson")

    @classmethod
    def gerg2008(cls, names: tuple[str, ...]) -> TeqpBackend:
        """Construct teqp's exact GERG-2008 residual model."""
        try:
            import teqp
        except ImportError as exc:
            raise ImportError("TeqpBackend requires the optional 'external' dependency") from exc
        gerg_names = [cls._GERG_NAMES.get(name, name) for name in names]
        model = teqp.make_model({"kind": "GERG2008resid", "model": {"names": gerg_names}})
        return cls(names, model, exact_model="GERG-2008 residual (teqp)")

    @classmethod
    def eoscg_2015(cls, names: tuple[str, ...]) -> TeqpBackend:
        """Construct the complete 2015 EOS-CG multiparameter mixture model.

        The six-component scope is CO2, water, nitrogen, oxygen, argon, and
        carbon monoxide.  teqp loads the pure-fluid Helmholtz equations plus
        the Gernert--Span binary reducing and departure parameters through its
        multifluid model factory and versioned CoolProp-format data files.
        """
        try:
            import teqp
        except ImportError as exc:
            raise ImportError("TeqpBackend requires the optional 'external' dependency") from exc
        try:
            component_names = [cls._EOSCG_NAMES[name] for name in names]
        except KeyError as exc:
            supported = ", ".join(cls._EOSCG_NAMES)
            raise ValueError(f"EOS-CG-2015 component must be one of: {supported}") from exc
        data_path = Path(teqp.get_datapath())
        model = teqp.build_multifluid_model(
            component_names,
            str(data_path),
            str(data_path / "dev" / "mixtures" / "mixture_binary_pairs.json"),
            departurepath=str(data_path / "dev" / "mixtures" / "mixture_departure_functions.json"),
        )
        return cls(
            names,
            model,
            exact_model="EOS-CG-2015 multiparameter mixture model (teqp)",
        )

    @staticmethod
    def _numpy_state(
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
    ) -> tuple[float, float, np.ndarray]:
        if temperature.ndim or pressure.ndim or composition.ndim != 1:
            raise ValueError("teqp adapter accepts one scalar T-P state")
        x = normalize_composition(composition)
        return (
            float(temperature.detach().cpu()),
            float(pressure.detach().cpu()),
            np.asarray(x.detach().cpu().numpy(), dtype=np.float64),
        )

    def _pressure(self, temperature: float, density: float, composition: np.ndarray) -> float:
        gas_constant = float(self._model.get_R(composition))
        return (
            density
            * gas_constant
            * temperature
            * (1.0 + float(self._model.get_Ar01(temperature, density, composition)))
        )

    def _density_roots(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
    ) -> tuple[float, ...]:
        t, p, x = self._numpy_state(temperature, pressure, composition)
        ideal_density = p / (R * t)
        grid = np.geomspace(max(ideal_density * 1.0e-5, 1.0e-8), 1.0e5, 400)

        def residual(density: float) -> float:
            return self._pressure(t, density, x) - p

        roots: list[float] = []
        left = float(grid[0])
        left_value = residual(left)
        for right_value_raw in grid[1:]:
            right = float(right_value_raw)
            right_value = residual(right)
            if (
                np.isfinite(left_value)
                and np.isfinite(right_value)
                and np.signbit(left_value) != np.signbit(right_value)
            ):
                low, high = left, right
                low_value = left_value
                for _ in range(100):
                    midpoint = 0.5 * (low + high)
                    midpoint_value = residual(midpoint)
                    if abs(midpoint_value) <= 1.0e-13 * max(p, 1.0):
                        break
                    if np.signbit(low_value) != np.signbit(midpoint_value):
                        high = midpoint
                    else:
                        low = midpoint
                        low_value = midpoint_value
                if abs(residual(midpoint)) <= 1.0e-10 * max(p, 1.0):
                    if not roots or abs(midpoint - roots[-1]) > 1.0e-7 * midpoint:
                        roots.append(midpoint)
            left, left_value = right, right_value
        if not roots:
            raise ConvergenceError("teqp density scan found no pressure root")
        return tuple(roots)

    def _select_density(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind,
    ) -> float:
        roots = self._density_roots(temperature, pressure, composition)
        if phase == "liquid":
            return roots[-1]
        if phase == "vapor":
            return roots[0]
        if phase != "stable":
            raise ValueError(f"unknown phase root {phase!r}")
        t, _, x = self._numpy_state(temperature, pressure, composition)
        with np.errstate(divide="ignore", invalid="ignore"):
            gibbs = [
                float(
                    np.dot(
                        x,
                        np.log(self._model.get_fugacity_coefficients(t, density * x)),
                    )
                )
                for density in roots
            ]
        return roots[int(np.argmin(gibbs))]

    def molar_volume(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return a teqp molar-volume root in m3/mol."""
        density = self._select_density(temperature, pressure, composition, phase)
        return torch.tensor(
            1.0 / density,
            dtype=temperature.dtype,
            device=temperature.device,
        )

    def pressure(self, temperature: Tensor, molar_volume: Tensor, composition: Tensor) -> Tensor:
        """Return pressure at a prescribed homogeneous density."""
        if temperature.ndim or molar_volume.ndim or composition.ndim != 1:
            raise ValueError("teqp adapter accepts one scalar T-volume state")
        x = normalize_composition(composition)
        value = self._pressure(
            float(temperature.detach().cpu()),
            1.0 / float(molar_volume.detach().cpu()),
            np.asarray(x.detach().cpu().numpy(), dtype=np.float64),
        )
        return torch.tensor(value, dtype=temperature.dtype, device=temperature.device)

    def select_z(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return compressibility factor."""
        volume = self.molar_volume(temperature, pressure, composition, phase)
        z_factor = pressure * volume / (R * temperature)
        if not bool(torch.isfinite(z_factor)):
            raise InvalidStateError("teqp produced a non-finite compressibility factor")
        return z_factor

    def log_fugacity_coefficients(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return teqp residual-model fugacity coefficients."""
        t, _, x = self._numpy_state(temperature, pressure, composition)
        density = self._select_density(temperature, pressure, composition, phase)
        values = np.asarray(self._model.get_fugacity_coefficients(t, density * x))
        if np.any(values <= 0.0) or not np.isfinite(values).all():
            raise InvalidStateError("teqp produced invalid fugacity coefficients")
        return torch.log(torch.tensor(values, dtype=temperature.dtype, device=temperature.device))
