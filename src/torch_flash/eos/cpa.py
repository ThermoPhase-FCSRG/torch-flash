"""Cubic-plus-association (CPA) equation of state.

The physical term is SRK and the association term is Wertheim TPT1. Site
fractions are solved by differentiable fixed-point iterations, while pressure
and chemical potentials are obtained from the residual Helmholtz energy with
PyTorch automatic differentiation.

The CPA equation is defined by Kontogeorgis et al., *Industrial & Engineering
Chemistry Research* 35 (1996), 4310-4318, doi:10.1021/ie9600203. Bundled
pure-component and cross-association parameters are from Folas et al.,
*Industrial & Engineering Chemistry Research* 44 (2005), 3823-3833,
doi:10.1021/ie048832j.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import Tensor, nn

from torch_flash.components import canonical_component_name, component
from torch_flash.config import resolve_tensor_options
from torch_flash.constants import R
from torch_flash.database import ParameterSource, load_model_parameters
from torch_flash.exceptions import ConvergenceError, InvalidStateError, ParameterDatabaseError
from torch_flash.mixing import QuadraticMixing, TemperatureDependentQuadraticMixing
from torch_flash.types import PhaseKind, normalize_composition

CombiningRule = Literal["CR1", "ECR"]
AssociationScheme: TypeAlias = Literal["none", "1A", "1B", "2B", "3B", "4C"]


@dataclass(frozen=True)
class CPAComponent:
    """Pure-component CPA parameters and optional petroleum pseudo-properties.

    ``critical_pressure``, ``acentric_factor``, and ``molar_mass`` normally
    come from the shared component database. Petroleum pseudo-components can
    instead carry those three initialization properties directly.
    """

    name: str
    critical_temperature: float
    a0: float
    b: float
    c1: float
    association_energy: float = 0.0
    association_volume: float = 0.0
    scheme: AssociationScheme = "none"
    critical_pressure: float | None = None
    acentric_factor: float | None = None
    molar_mass: float | None = None


def _site_types(scheme: str) -> tuple[int, int, int, int]:
    if scheme == "none":
        return (-1, -1, -1, -1)
    if scheme == "1A":
        return (0, -1, -1, -1)
    if scheme == "1B":
        return (1, -1, -1, -1)
    if scheme == "2B":
        return (0, 1, -1, -1)
    if scheme == "3B":
        return (0, 1, 1, -1)
    if scheme == "4C":
        return (0, 0, 1, 1)
    raise ValueError(f"unsupported association scheme {scheme!r}")


class CPAEOS(nn.Module):
    """SRK cubic-plus-association mixture model."""

    critical_temperature: Tensor
    critical_pressure: Tensor
    acentric_factor: Tensor
    molar_mass: Tensor
    a0: Tensor
    b: Tensor
    c1: Tensor
    association_energy: Tensor
    association_volume: Tensor
    site_types: Tensor
    cross_association_mask: Tensor
    cross_association_energy: Tensor
    cross_association_volume: Tensor
    mixing: QuadraticMixing | TemperatureDependentQuadraticMixing

    def __init__(
        self,
        parameters: tuple[CPAComponent, ...],
        *,
        kij: Tensor | None = None,
        kij_a: Tensor | None = None,
        kij_b: Tensor | None = None,
        lij: Tensor | None = None,
        cross_association_energy: Tensor | None = None,
        cross_association_volume: Tensor | None = None,
        combining_rule: CombiningRule = "ECR",
        trainable: bool = False,
        trainable_lij: bool = False,
        association_iterations: int = 10,
        association_newton_iterations: int = 8,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not parameters:
            raise ValueError("CPA requires at least one component")
        if combining_rule not in ("CR1", "ECR"):
            raise ValueError(f"unknown CPA combining rule {combining_rule!r}")
        if association_iterations < 0 or association_newton_iterations < 0:
            raise ValueError("CPA association iteration counts must be nonnegative")
        if kij is not None and (kij_a is not None or kij_b is not None):
            raise ValueError("pass either fixed kij or temperature-dependent kij A/B, not both")
        if (kij_a is None) != (kij_b is None):
            raise ValueError("temperature-dependent CPA interactions require both kij_a and kij_b")
        if (cross_association_energy is None) != (cross_association_volume is None):
            raise ValueError("cross-association overrides require both energy and volume matrices")
        self.names = tuple(item.name for item in parameters)
        self.combining_rule = combining_rule
        self.association_iterations = association_iterations
        self.association_newton_iterations = association_newton_iterations
        supplied_tensors = tuple(
            value
            for value in (
                kij,
                kij_a,
                kij_b,
                lij,
                cross_association_energy,
                cross_association_volume,
            )
            if value is not None
        )
        runtime_dtype, runtime_device = resolve_tensor_options(dtype, device)
        dtype = supplied_tensors[0].dtype if dtype is None and supplied_tensors else runtime_dtype
        device = (
            supplied_tensors[0].device if device is None and supplied_tensors else runtime_device
        )

        def tensor(values: list[float]) -> Tensor:
            return torch.tensor(values, dtype=dtype, device=device)

        numeric_rows = (
            (
                item.critical_temperature,
                item.a0,
                item.b,
                item.c1,
                item.association_energy,
                item.association_volume,
            )
            for item in parameters
        )
        if any(
            not all(torch.isfinite(torch.tensor(row)))
            or row[0] <= 0.0
            or row[1] <= 0.0
            or row[2] <= 0.0
            or row[4] < 0.0
            or row[5] < 0.0
            for row in numeric_rows
        ):
            raise ValueError(
                "CPA temperatures, a0, and b must be positive; association parameters "
                "must be finite and nonnegative"
            )
        self.register_buffer(
            "critical_temperature",
            tensor([item.critical_temperature for item in parameters]),
        )
        fitted_parameters = {
            "a0": tensor([item.a0 for item in parameters]),
            "b": tensor([item.b for item in parameters]),
            "c1": tensor([item.c1 for item in parameters]),
            "association_energy": tensor([item.association_energy for item in parameters]),
            "association_volume": tensor([item.association_volume for item in parameters]),
        }
        for name, value in fitted_parameters.items():
            if trainable:
                setattr(self, name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)
        self.register_buffer(
            "site_types",
            torch.tensor(
                [_site_types(item.scheme) for item in parameters],
                dtype=torch.int64,
                device=device,
            ),
        )

        critical_pressure: list[float] = []
        acentric_factor: list[float] = []
        molar_mass: list[float] = []
        for item in parameters:
            if (
                item.critical_pressure is None
                or item.acentric_factor is None
                or item.molar_mass is None
            ):
                try:
                    shared = component(item.name)
                except KeyError as exc:
                    raise ParameterDatabaseError(
                        f"custom CPA component {item.name!r} must provide critical_pressure, "
                        "acentric_factor, and molar_mass"
                    ) from exc
                shared_pressure = shared.critical_pressure
                shared_acentric = shared.acentric_factor
                shared_mass = shared.molar_mass
            else:
                shared_pressure = item.critical_pressure
                shared_acentric = item.acentric_factor
                shared_mass = item.molar_mass
            resolved_pressure = (
                item.critical_pressure if item.critical_pressure is not None else shared_pressure
            )
            resolved_acentric = (
                item.acentric_factor if item.acentric_factor is not None else shared_acentric
            )
            resolved_mass = item.molar_mass if item.molar_mass is not None else shared_mass
            if resolved_acentric is None:
                raise ParameterDatabaseError(
                    f"CPA component {item.name!r} requires an acentric factor"
                )
            if (
                not torch.isfinite(torch.tensor(resolved_pressure))
                or not torch.isfinite(torch.tensor(resolved_acentric))
                or not torch.isfinite(torch.tensor(resolved_mass))
                or resolved_pressure <= 0.0
                or resolved_mass <= 0.0
            ):
                raise ValueError("CPA critical pressure and molar mass must be finite and positive")
            critical_pressure.append(float(resolved_pressure))
            acentric_factor.append(float(resolved_acentric))
            molar_mass.append(float(resolved_mass))
        self.register_buffer("critical_pressure", tensor(critical_pressure))
        self.register_buffer("acentric_factor", tensor(acentric_factor))
        self.register_buffer("molar_mass", tensor(molar_mass))

        size = len(parameters)
        if kij_a is not None and kij_b is not None:
            self.mixing = TemperatureDependentQuadraticMixing(
                kij_a.to(dtype=dtype, device=device),
                kij_b.to(dtype=dtype, device=device),
                None if lij is None else lij.to(dtype=dtype, device=device),
                trainable=trainable,
                trainable_lij=trainable_lij,
            )
        else:
            fixed_kij = (
                torch.zeros((size, size), dtype=dtype, device=device)
                if kij is None
                else kij.to(dtype=dtype, device=device)
            )
            self.mixing = QuadraticMixing(
                fixed_kij,
                None if lij is None else lij.to(dtype=dtype, device=device),
                trainable=trainable,
                trainable_lij=trainable_lij,
            )

        if cross_association_energy is None or cross_association_volume is None:
            cross_mask = torch.zeros((size, size), dtype=torch.bool, device=device)
            cross_energy = torch.zeros((size, size), dtype=dtype, device=device)
            cross_volume = torch.zeros((size, size), dtype=dtype, device=device)
        else:
            supplied_energy = cross_association_energy.to(dtype=dtype, device=device)
            supplied_volume = cross_association_volume.to(dtype=dtype, device=device)
            if supplied_energy.shape != (size, size) or supplied_volume.shape != (size, size):
                raise ValueError("cross-association matrices must match the CPA component count")
            energy_mask = torch.isfinite(supplied_energy)
            volume_mask = torch.isfinite(supplied_volume)
            if not torch.equal(energy_mask, volume_mask):
                raise ValueError("cross-association energy and volume masks must match")
            if not torch.equal(energy_mask, energy_mask.mT):
                raise ValueError("cross-association overrides must be symmetric")
            if bool(torch.diagonal(energy_mask).any()):
                raise ValueError("cross-association overrides cannot replace pure-component pairs")
            cross_mask = energy_mask
            cross_energy = torch.where(
                energy_mask, supplied_energy, torch.zeros_like(supplied_energy)
            )
            cross_volume = torch.where(
                volume_mask, supplied_volume, torch.zeros_like(supplied_volume)
            )
            if (
                not torch.allclose(cross_energy, cross_energy.mT)
                or not torch.allclose(cross_volume, cross_volume.mT)
                or bool((cross_energy[cross_mask] <= 0.0).any())
                or bool((cross_volume[cross_mask] <= 0.0).any())
            ):
                raise ValueError(
                    "cross-association energies and volumes must be symmetric and positive"
                )
        self.register_buffer("cross_association_mask", cross_mask)
        if trainable:
            self.cross_association_energy = nn.Parameter(cross_energy)
            self.cross_association_volume = nn.Parameter(cross_volume)
        else:
            self.register_buffer("cross_association_energy", cross_energy)
            self.register_buffer("cross_association_volume", cross_volume)

    @property
    def ncomponents(self) -> int:
        """Number of mixture components."""
        return len(self.names)

    def pure_parameters(self, temperature: Tensor) -> tuple[Tensor, Tensor]:
        """Return CPA's fitted SRK energy and covolume parameters."""
        reduced = temperature[..., None] / self.critical_temperature
        a = self.a0 * (1.0 + self.c1 * (1.0 - torch.sqrt(reduced))).square()
        return a, self.b

    def mixture_parameters(self, temperature: Tensor, composition: Tensor) -> tuple[Tensor, Tensor]:
        """Return conventionally mixed physical-term parameters."""
        pure_a, pure_b = self.pure_parameters(temperature)
        result: tuple[Tensor, Tensor] = self.mixing(temperature, composition, pure_a, pure_b)
        return result

    def binary_interaction(self, temperature: Tensor) -> Tensor:
        """Return the symmetric physical-term ``kij`` matrix at ``temperature``."""
        if isinstance(self.mixing, TemperatureDependentQuadraticMixing):
            return self.mixing.kij(temperature)
        return self.mixing.kij

    def association_strength(
        self, temperature: Tensor, molar_density: Tensor, composition: Tensor
    ) -> Tensor:
        """Return ``Delta[i, A, j, B]`` association strengths in m3/mol."""
        x = normalize_composition(composition)
        _, bm = self.mixture_parameters(temperature, x)
        packing_fraction = 0.25 * bm * molar_density
        radial_distribution = 1.0 / (1.0 - 1.9 * packing_fraction)
        bij = 0.5 * (self.b[:, None] + self.b[None, :])
        temperature_pair = temperature[..., None, None]
        if self.combining_rule == "CR1":
            epsilon = 0.5 * (self.association_energy[:, None] + self.association_energy[None, :])
            beta = torch.sqrt(self.association_volume[:, None] * self.association_volume[None, :])
            pair_strength = (
                radial_distribution[..., None, None]
                * torch.expm1(epsilon / (R * temperature_pair))
                * bij
                * beta
            )
        elif self.combining_rule == "ECR":
            pure_factor = (
                torch.expm1(self.association_energy / (R * temperature[..., None]))
                * self.b
                * self.association_volume
            )
            pair_strength = radial_distribution[..., None, None] * torch.sqrt(
                pure_factor[..., :, None] * pure_factor[..., None, :]
            )
        else:  # pragma: no cover - constructor validation protects this branch
            raise ValueError(f"unknown CPA combining rule {self.combining_rule!r}")

        # Modified CR-1 solvation overrides use the published pair energy and
        # volume directly in Delta = g [exp(epsilon/RT)-1] bij beta.
        override_strength = (
            radial_distribution[..., None, None]
            * torch.expm1(self.cross_association_energy / (R * temperature_pair))
            * bij
            * self.cross_association_volume
        )
        pair_strength = torch.where(
            self.cross_association_mask,
            override_strength,
            pair_strength,
        )
        types_i = self.site_types[:, :, None, None]
        types_j = self.site_types[None, None, :, :]
        compatible = (types_i >= 0) & (types_j >= 0) & (types_i != types_j)
        return pair_strength[..., :, None, :, None] * compatible

    def site_fractions(
        self, temperature: Tensor, molar_density: Tensor, composition: Tensor
    ) -> Tensor:
        """Solve the CPA mass-action equations for unbonded site fractions.

        Leading dimensions are broadcast as independent homogeneous states.
        The component axis is the final composition dimension.
        """
        x = normalize_composition(composition)
        if x.shape[-1] != self.ncomponents:
            raise ValueError("CPA composition has the wrong number of components")
        batch_shape = torch.broadcast_shapes(
            temperature.shape,
            molar_density.shape,
            x.shape[:-1],
        )
        temperature = torch.broadcast_to(temperature, batch_shape)
        molar_density = torch.broadcast_to(molar_density, batch_shape)
        x = torch.broadcast_to(x, (*batch_shape, self.ncomponents))
        strength = self.association_strength(temperature, molar_density, x)
        active = self.site_types >= 0
        site_fraction = torch.ones(
            (*batch_shape, self.ncomponents, self.site_types.shape[-1]),
            dtype=x.dtype,
            device=x.device,
        )
        for _ in range(self.association_iterations):
            bonded_sum = torch.einsum(
                "...j,...jb,...iajb->...ia",
                x,
                site_fraction,
                strength,
            )
            update = 1.0 / (1.0 + molar_density[..., None, None] * bonded_sum)
            update = torch.where(active, update, torch.ones_like(update))
            site_fraction = 0.5 * site_fraction + 0.5 * update

        # Newton refinement uses the analytic mass-action Jacobian. A
        # positivity-limited step replaces dozens of slowly convergent Picard
        # iterations in strongly associating liquids while retaining a fixed,
        # differentiable workload suitable for torch.compile and accelerators.
        nsites = self.site_types.shape[-1]
        system_size = self.ncomponents * nsites
        flattened_strength = strength.reshape(*batch_shape, system_size, system_size)
        site_weights = (
            x[..., :, None]
            .expand(*batch_shape, self.ncomponents, nsites)
            .reshape(*batch_shape, system_size)
        )
        flattened_active = active.reshape(system_size)
        identity = torch.eye(system_size, dtype=x.dtype, device=x.device)
        for _ in range(self.association_newton_iterations):
            flattened_sites = site_fraction.reshape(*batch_shape, system_size)
            bonded_sum = torch.einsum(
                "...j,...jb,...iajb->...ia",
                x,
                site_fraction,
                strength,
            ).reshape(*batch_shape, system_size)
            residual = flattened_sites.reciprocal() - 1.0 - molar_density[..., None] * bonded_sum
            jacobian = -torch.diag_embed(flattened_sites.reciprocal().square())
            jacobian = jacobian - (
                molar_density[..., None, None] * flattened_strength * site_weights[..., None, :]
            )
            residual = torch.where(flattened_active, residual, flattened_sites - 1.0)
            jacobian = torch.where(flattened_active[:, None], jacobian, identity)
            step = torch.linalg.solve(jacobian, -residual)
            decreases_site_fraction = (step < 0.0) & flattened_active
            safe_step = torch.where(
                decreases_site_fraction,
                step,
                -torch.ones_like(step),
            )
            positivity_limits = torch.where(
                decreases_site_fraction,
                -0.9 * flattened_sites / safe_step,
                torch.full_like(step, torch.inf),
            )
            step_scale = torch.minimum(
                torch.ones_like(positivity_limits[..., 0]),
                positivity_limits.amin(dim=-1),
            )
            site_fraction = (flattened_sites + step_scale[..., None] * step).reshape(
                *batch_shape, self.ncomponents, nsites
            )
            site_fraction = torch.where(active, site_fraction, torch.ones_like(site_fraction))
        return site_fraction

    def residual_helmholtz_rt(self, temperature: Tensor, volume: Tensor, moles: Tensor) -> Tensor:
        """Return extensive ``Ares/(RT)`` for the combined CPA model."""
        if moles.ndim != 1 or volume.ndim != 0:
            raise ValueError("CPA Helmholtz kernel currently accepts one homogeneous state")
        total = moles.sum()
        x = moles / total
        molar_volume = volume / total
        am, bm = self.mixture_parameters(temperature, x)
        if bool(molar_volume <= bm):
            raise InvalidStateError("CPA molar volume must exceed mixture covolume")
        physical = -total * torch.log1p(-bm / molar_volume)
        physical = physical - total * am / (R * temperature * bm) * torch.log(
            (molar_volume + bm) / molar_volume
        )
        site_fraction = self.site_fractions(temperature, molar_volume.reciprocal(), x)
        active = self.site_types >= 0
        association_terms = torch.where(
            active,
            torch.log(site_fraction) - 0.5 * site_fraction + 0.5,
            torch.zeros_like(site_fraction),
        )
        association = torch.sum(moles[:, None] * association_terms)
        return physical + association

    def pressure(self, temperature: Tensor, molar_volume: Tensor, composition: Tensor) -> Tensor:
        """Evaluate the published CPA pressure equation."""
        x = normalize_composition(composition)
        am, bm = self.mixture_parameters(temperature, x)
        density = molar_volume.reciprocal()
        sites = self.site_fractions(temperature, density, x)
        active = self.site_types >= 0
        unbonded = torch.where(active, 1.0 - sites, torch.zeros_like(sites))
        association_sum = torch.sum(x[..., :, None] * unbonded, dim=(-2, -1))
        packing_fraction = 0.25 * bm * density
        density_dlogg = 1.9 * packing_fraction / (1.0 - 1.9 * packing_fraction)
        physical = R * temperature / (molar_volume - bm) - am / (molar_volume * (molar_volume + bm))
        association = (
            -0.5 * R * temperature / molar_volume * (1.0 + density_dlogg) * association_sum
        )
        return physical + association

    def _volume_roots(
        self, temperature: Tensor, pressure: Tensor, composition: Tensor
    ) -> tuple[Tensor, ...]:
        """Locate and polish all positive CPA pressure roots."""
        x = normalize_composition(composition)
        _, bm = self.mixture_parameters(temperature, x)
        minimum = bm * (1.0 + 1.0e-9)
        maximum = torch.maximum(100.0 * R * temperature / pressure, minimum * 1.0e6)
        grid = torch.logspace(
            float(torch.log10(minimum.detach())),
            float(torch.log10(maximum.detach())),
            96,
            dtype=temperature.dtype,
            device=temperature.device,
        )

        def residual(volume: Tensor) -> Tensor:
            return self.pressure(temperature, volume, x) - pressure

        # The scan is one batched pressure call. Association-site iterations
        # therefore operate on the entire grid instead of repeating Python and
        # dispatcher overhead for every candidate volume.
        values = residual(grid)
        finite = torch.isfinite(values[:-1]) & torch.isfinite(values[1:])
        changes_sign = torch.signbit(values[:-1]) != torch.signbit(values[1:])
        exact = (values[:-1] == 0.0) | (values[1:] == 0.0)
        bracket_indices = torch.nonzero(finite & (changes_sign | exact)).flatten().tolist()
        brackets = [(grid[index], grid[index + 1]) for index in bracket_indices]
        if not brackets:
            raise ConvergenceError("CPA volume scan found no pressure root")

        roots: list[Tensor] = []
        for left, right in brackets:
            left_value = residual(left)
            for _ in range(80):
                midpoint = 0.5 * (left + right)
                midpoint_value = residual(midpoint)
                if float((midpoint_value.abs() / pressure).detach()) <= 1.0e-12:
                    break
                if bool(torch.signbit(left_value) != torch.signbit(midpoint_value)):
                    right = midpoint
                else:
                    left = midpoint
                    left_value = midpoint_value

            # Bisection is used only for robust root selection. Newton polishing
            # restores the implicit parameter derivatives of p(T, v, x)=P.
            volume = midpoint
            for _ in range(4):
                value = residual(volume)
                slope = torch.func.grad(residual)(volume)
                volume = volume - value / slope
            roots.append(volume)
        return tuple(roots)

    def _phase_volume_newton(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: Literal["liquid", "vapor"],
    ) -> Tensor | None:
        """Try the fast differentiable phase-specific volume solve."""
        _, bm = self.mixture_parameters(temperature, composition)
        minimum = bm * (1.0 + 1.0e-9)
        ideal = R * temperature / pressure
        initial = minimum * 1.2 if phase == "liquid" else torch.maximum(ideal, minimum * 2.0)
        log_minimum = torch.log(minimum)
        log_volume = torch.log(initial)

        def residual(current: Tensor) -> Tensor:
            volume = torch.exp(current)
            return (self.pressure(temperature, volume, composition) - pressure) / pressure

        for _ in range(40):
            value = residual(log_volume)
            if float(value.detach().abs()) <= 1.0e-10:
                return torch.exp(log_volume)
            slope: Tensor = torch.func.grad(residual)(log_volume)
            step = torch.clamp(-value / slope, -0.5, 0.5)
            if not bool(torch.isfinite(step)):
                return None
            log_volume = torch.maximum(log_volume + step, log_minimum)
        return None

    def molar_volume(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Solve the CPA volume root for a specified phase."""
        if temperature.ndim != 0 or pressure.ndim != 0 or composition.ndim != 1:
            raise ValueError("CPA volume solver currently accepts one scalar T-P state")
        x = normalize_composition(composition)
        if phase in ("liquid", "vapor"):
            fast_volume = self._phase_volume_newton(
                temperature,
                pressure,
                x,
                phase,
            )
            if fast_volume is not None:
                return fast_volume
        roots = self._volume_roots(temperature, pressure, x)
        if phase == "liquid":
            return roots[0]
        if phase == "vapor":
            return roots[-1]
        if phase != "stable":
            raise ValueError(f"unknown phase root {phase!r}")

        gibbs: list[Tensor] = []
        for volume in roots:
            z = pressure * volume / (R * temperature)
            residual_helmholtz = self.residual_helmholtz_rt(temperature, volume, x)
            gibbs.append(residual_helmholtz + z - 1.0 - torch.log(z))
        index = int(torch.argmin(torch.stack(gibbs)).detach())
        return roots[index]

    def select_z(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return the CPA compressibility factor."""
        volume = self.molar_volume(temperature, pressure, composition, phase)
        return pressure * volume / (R * temperature)

    def log_fugacity_coefficients(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return CPA log fugacity coefficients from Helmholtz derivatives."""
        x = normalize_composition(composition)
        volume = self.molar_volume(temperature, pressure, x, phase)

        def at_fixed_volume(moles: Tensor) -> Tensor:
            return self.residual_helmholtz_rt(temperature, volume, moles)

        residual_mu_rt: Tensor = torch.func.grad(at_fixed_volume)(x)
        z = pressure * volume / (R * temperature)
        return residual_mu_rt - torch.log(z)


def _cpa_component(record_name: str, record: Mapping[str, object]) -> CPAComponent:
    required = (
        "critical_temperature",
        "a0",
        "b",
        "c1",
        "association_energy",
        "association_volume",
    )
    if any(not isinstance(record.get(key), int | float) for key in required):
        raise ParameterDatabaseError(f"CPA component {record_name!r} has non-numeric parameters")
    scheme = record.get("scheme", "none")
    if scheme not in ("none", "1A", "1B", "2B", "3B", "4C"):
        raise ParameterDatabaseError(
            f"CPA component {record_name!r} has unsupported association scheme {scheme!r}"
        )
    association_scheme: AssociationScheme = scheme
    numeric = {key: float(record[key]) for key in required}  # type: ignore[arg-type]
    optional: dict[str, float | None] = {}
    for key in ("critical_pressure", "acentric_factor", "molar_mass"):
        value = record.get(key)
        if value is not None and not isinstance(value, int | float):
            raise ParameterDatabaseError(
                f"CPA component {record_name!r} has non-numeric optional {key!r}"
            )
        optional[key] = None if value is None else float(value)
    return CPAComponent(
        record_name,
        numeric["critical_temperature"],
        numeric["a0"],
        numeric["b"],
        numeric["c1"],
        numeric["association_energy"],
        numeric["association_volume"],
        association_scheme,
        optional["critical_pressure"],
        optional["acentric_factor"],
        optional["molar_mass"],
    )


def _pair_indices(key: object, selected: tuple[str, ...], source: str) -> tuple[int, int] | None:
    if not isinstance(key, str) or key.count("|") != 1:
        raise ParameterDatabaseError(f"{source} pair keys must have the form 'first|second'")
    first_raw, second_raw = key.split("|")
    first = canonical_component_name(first_raw)
    second = canonical_component_name(second_raw)
    if first == second:
        raise ParameterDatabaseError(f"{source} cannot contain a self-pair {key!r}")
    if first not in selected or second not in selected:
        return None
    return selected.index(first), selected.index(second)


def _database_binary_interactions(
    document: object,
    selected: tuple[str, ...],
    source: str,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    if document is None:
        return None, None, None
    if not isinstance(document, Mapping):
        raise ParameterDatabaseError(f"{source} binary_interactions must be a mapping")
    kind = document.get("kind")
    pairs = document.get("pairs")
    if not isinstance(pairs, Mapping):
        raise ParameterDatabaseError(f"{source} binary_interactions requires a pairs mapping")
    size = len(selected)
    if kind == "constant":
        values = torch.zeros((size, size), dtype=dtype, device=device)
        for key, record in pairs.items():
            indices = _pair_indices(key, selected, f"{source} binary_interactions")
            if indices is None:
                continue
            if not isinstance(record, int | float):
                raise ParameterDatabaseError(
                    f"{source} constant interaction {key!r} must be numeric"
                )
            first, second = indices
            values[first, second] = values[second, first] = float(record)
        return values, None, None
    if kind == "a_plus_b_over_temperature":
        a = torch.zeros((size, size), dtype=dtype, device=device)
        b = torch.zeros((size, size), dtype=dtype, device=device)
        for key, record in pairs.items():
            indices = _pair_indices(key, selected, f"{source} binary_interactions")
            if indices is None:
                continue
            if (
                not isinstance(record, Mapping)
                or not isinstance(record.get("a"), int | float)
                or not isinstance(record.get("b"), int | float)
            ):
                raise ParameterDatabaseError(
                    f"{source} temperature-dependent interaction {key!r} requires numeric a and b"
                )
            first, second = indices
            a[first, second] = a[second, first] = float(record["a"])
            b[first, second] = b[second, first] = float(record["b"])
        return None, a, b
    raise ParameterDatabaseError(
        f"{source} binary_interactions kind must be 'constant' or 'a_plus_b_over_temperature'"
    )


def _database_cross_association(
    document: object,
    selected: tuple[str, ...],
    source: str,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[Tensor | None, Tensor | None]:
    if document is None:
        return None, None
    if not isinstance(document, Mapping):
        raise ParameterDatabaseError(f"{source} cross_association must be a mapping")
    pairs = document.get("pairs")
    if not isinstance(pairs, Mapping):
        raise ParameterDatabaseError(f"{source} cross_association requires a pairs mapping")
    size = len(selected)
    energy = torch.full((size, size), torch.nan, dtype=dtype, device=device)
    volume = torch.full((size, size), torch.nan, dtype=dtype, device=device)
    for key, record in pairs.items():
        indices = _pair_indices(key, selected, f"{source} cross_association")
        if indices is None:
            continue
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("association_energy"), int | float)
            or not isinstance(record.get("association_volume"), int | float)
        ):
            raise ParameterDatabaseError(
                f"{source} cross-association pair {key!r} requires numeric "
                "association_energy and association_volume"
            )
        first, second = indices
        energy[first, second] = energy[second, first] = float(record["association_energy"])
        volume[first, second] = volume[second, first] = float(record["association_volume"])
    if not bool(torch.isfinite(energy).any()):
        return None, None
    return energy, volume


def cpa_eos(
    names: tuple[str, ...],
    parameter_set: ParameterSource = "cpa.folas-2005",
    *,
    kij: Tensor | None = None,
    kij_a: Tensor | None = None,
    kij_b: Tensor | None = None,
    lij: Tensor | None = None,
    cross_association_energy: Tensor | None = None,
    cross_association_volume: Tensor | None = None,
    combining_rule: CombiningRule | None = None,
    trainable: bool = False,
    trainable_lij: bool = False,
    association_iterations: int = 10,
    association_newton_iterations: int = 8,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> CPAEOS:
    """Construct CPA from a bundled/custom YAML set or use :class:`CPAEOS`.

    Direct construction with a tuple of :class:`CPAComponent` remains the
    explicit in-memory API for fitting new species or parameterizations.
    ``lij`` optionally modifies the physical SRK co-volume rule;
    ``trainable_lij`` controls it independently of the attraction parameters.
    """
    supplied_tensors = tuple(
        value
        for value in (
            kij,
            kij_a,
            kij_b,
            lij,
            cross_association_energy,
            cross_association_volume,
        )
        if value is not None
    )
    runtime_dtype, runtime_device = resolve_tensor_options(dtype, device)
    dtype = supplied_tensors[0].dtype if dtype is None and supplied_tensors else runtime_dtype
    device = supplied_tensors[0].device if device is None and supplied_tensors else runtime_device
    selected = tuple(canonical_component_name(name) for name in names)
    loaded = load_model_parameters(parameter_set)
    if loaded.model_kind != "cpa":
        raise ParameterDatabaseError(f"{loaded.identifier!r} is {loaded.model_kind!r}, not 'cpa'")
    records = loaded.parameters.get("components")
    if not isinstance(records, Mapping):
        raise ParameterDatabaseError(f"{loaded.identifier!r} requires a components mapping")
    parameters: list[CPAComponent] = []
    for name in selected:
        record = records.get(name)
        if not isinstance(record, Mapping):
            raise KeyError(f"{loaded.identifier} CPA parameters are unavailable for {name!r}")
        parameters.append(_cpa_component(name, record))
    default_rule = loaded.parameters.get("default_combining_rule", "ECR")
    resolved_rule = default_rule if combining_rule is None else combining_rule
    if resolved_rule not in ("CR1", "ECR"):
        raise ParameterDatabaseError(f"unsupported CPA combining rule {resolved_rule!r}")
    if kij is None and kij_a is None and kij_b is None:
        kij, kij_a, kij_b = _database_binary_interactions(
            loaded.parameters.get("binary_interactions"),
            selected,
            loaded.identifier,
            dtype=dtype,
            device=device,
        )
    if cross_association_energy is None and cross_association_volume is None:
        cross_association_energy, cross_association_volume = _database_cross_association(
            loaded.parameters.get("cross_association"),
            selected,
            loaded.identifier,
            dtype=dtype,
            device=device,
        )
    return CPAEOS(
        tuple(parameters),
        kij=kij,
        kij_a=kij_a,
        kij_b=kij_b,
        lij=lij,
        cross_association_energy=cross_association_energy,
        cross_association_volume=cross_association_volume,
        combining_rule=resolved_rule,
        trainable=trainable,
        trainable_lij=trainable_lij,
        association_iterations=association_iterations,
        association_newton_iterations=association_newton_iterations,
        dtype=dtype,
        device=device,
    )


def cpa_folas_2005(
    names: tuple[str, ...],
    *,
    kij: Tensor | None = None,
    lij: Tensor | None = None,
    combining_rule: CombiningRule = "ECR",
    trainable: bool = False,
    trainable_lij: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> CPAEOS:
    """Construct CPA with the Folas et al. (2005), Table 1, pure parameters.

    Reference: doi:10.1021/ie048832j.
    """
    return cpa_eos(
        names,
        "cpa.folas-2005",
        kij=kij,
        lij=lij,
        combining_rule=combining_rule,
        trainable=trainable,
        trainable_lij=trainable_lij,
        dtype=dtype,
        device=device,
    )


def cpa_oliveira_2007(
    names: tuple[str, ...],
    *,
    kij: Tensor | None = None,
    lij: Tensor | None = None,
    cross_association_energy: Tensor | None = None,
    cross_association_volume: Tensor | None = None,
    trainable: bool = False,
    trainable_lij: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> CPAEOS:
    """Construct the hydrocarbon-water CPA parameterization of Oliveira et al.

    Pure parameters and constant physical-term interactions are from Tables
    1-3 of Oliveira, Coutinho, and Queimada, Fluid Phase Equilibria 258
    (2007) 58-66, doi:10.1016/j.fluid.2007.05.023. Aromatic-water solvation
    uses the paper's modified CR-1 cross-association parameters.
    """
    return cpa_eos(
        names,
        "cpa.oliveira-2007-hydrocarbon-water",
        kij=kij,
        lij=lij,
        cross_association_energy=cross_association_energy,
        cross_association_volume=cross_association_volume,
        combining_rule="CR1",
        trainable=trainable,
        trainable_lij=trainable_lij,
        dtype=dtype,
        device=device,
    )


def cpa_yan_2009(
    names: tuple[str, ...],
    *,
    kij: Tensor | None = None,
    kij_a: Tensor | None = None,
    kij_b: Tensor | None = None,
    lij: Tensor | None = None,
    trainable: bool = False,
    trainable_lij: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> CPAEOS:
    """Construct reservoir-fluid CPA with Yan et al.'s ``A + B/T`` water BIPs.

    Reference: Yan, Kontogeorgis, and Stenby, Fluid Phase Equilibria 276
    (2009) 75-85, doi:10.1016/j.fluid.2008.10.007.
    """
    return cpa_eos(
        names,
        "cpa.yan-2009-reservoir-fluids",
        kij=kij,
        kij_a=kij_a,
        kij_b=kij_b,
        lij=lij,
        trainable=trainable,
        trainable_lij=trainable_lij,
        dtype=dtype,
        device=device,
    )
