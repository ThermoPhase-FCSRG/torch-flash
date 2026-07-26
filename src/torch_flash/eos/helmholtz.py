"""Pure-fluid fundamental Helmholtz equations of state."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from torch_flash.types import PhaseKind

from .multifluid import MultiFluidEOS


@dataclass(frozen=True)
class PureFluidHelmholtzMetadata:
    """Identity and scope of one pure-fluid fundamental equation.

    Attributes
    ----------
    model
        Published equation or parameter-set identity.
    reference
        Defining bibliographic reference.
    version
        Version of the serialized coefficient set.
    fluid
        Canonical pure-fluid name.
    """

    model: str
    reference: str
    version: str
    fluid: str


class PureFluidHelmholtzEOS(nn.Module):
    """One-component thermodynamic Helmholtz equation.

    The internal kernel shares the differentiable Helmholtz term evaluators
    used by multifluid models, while this public wrapper exposes pure-fluid
    signatures without a redundant composition argument.

    Parameters
    ----------
    kernel
        One-component native Helmholtz evaluation kernel.
    metadata
        Published model identity and canonical pure-fluid name.
    """

    _composition: Tensor

    def __init__(
        self,
        kernel: MultiFluidEOS,
        metadata: PureFluidHelmholtzMetadata,
    ) -> None:
        super().__init__()
        if len(kernel.names) != 1:
            raise ValueError("a pure-fluid Helmholtz kernel requires exactly one component")
        if kernel.names[0] != metadata.fluid:
            raise ValueError("pure-fluid metadata must match the kernel component")
        self.kernel = kernel
        self.metadata = metadata
        self.register_buffer("_composition", kernel.gas_constant.new_ones(1))

    @property
    def gas_constant(self) -> Tensor:
        """Equation gas constant in J/(mol K)."""
        return self.kernel.gas_constant

    @property
    def critical_temperature(self) -> Tensor:
        """Critical temperature in K."""
        return self.kernel.critical_temperature[0]

    @property
    def critical_pressure(self) -> Tensor:
        """Critical pressure in Pa."""
        return self.kernel.critical_pressure[0]

    @property
    def critical_density(self) -> Tensor:
        """Critical molar density in mol/m3."""
        return self.kernel.critical_density[0]

    @property
    def molar_mass(self) -> Tensor:
        """Molar mass in kg/mol."""
        return self.kernel.molar_mass[0]

    @property
    def residual_term_count(self) -> int:
        """Number of nonzero residual Helmholtz terms."""
        return int(torch.count_nonzero(self.kernel.pure_n[0]))

    def alpha_ideal(self, temperature: Tensor, molar_density: Tensor) -> Tensor:
        """Return dimensionless ideal molar Helmholtz energy."""
        return self.kernel.alpha_ideal(temperature, molar_density, self._composition)

    def alpha_residual(self, temperature: Tensor, molar_density: Tensor) -> Tensor:
        """Return dimensionless residual molar Helmholtz energy."""
        return self.kernel.alpha_residual(temperature, molar_density, self._composition)

    def alpha_total(self, temperature: Tensor, molar_density: Tensor) -> Tensor:
        """Return total dimensionless molar Helmholtz energy."""
        return self.kernel.alpha_total(temperature, molar_density, self._composition)

    def pressure(self, temperature: Tensor, molar_volume: Tensor) -> Tensor:
        """Return pressure in Pa at the supplied homogeneous state."""
        return self.kernel.pressure(temperature, molar_volume, self._composition)

    def molar_volume(
        self,
        temperature: Tensor,
        pressure: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return the selected molar-volume root in m3/mol."""
        return self.kernel.molar_volume(
            temperature,
            pressure,
            self._composition,
            phase,
        )

    def compressibility_factor(self, temperature: Tensor, molar_volume: Tensor) -> Tensor:
        """Return the homogeneous-state compressibility factor."""
        pressure = self.pressure(temperature, molar_volume)
        return pressure * molar_volume / (self.gas_constant * temperature)

    def molar_helmholtz_energy(
        self,
        temperature: Tensor,
        molar_density: Tensor,
    ) -> Tensor:
        """Return molar Helmholtz energy in J/mol."""
        return self.kernel.molar_helmholtz_energy(
            temperature,
            molar_density,
            self._composition,
        )

    def molar_entropy(self, temperature: Tensor, molar_density: Tensor) -> Tensor:
        """Return molar entropy in J/(mol K)."""
        return self.kernel.molar_entropy(temperature, molar_density, self._composition)

    def molar_internal_energy(
        self,
        temperature: Tensor,
        molar_density: Tensor,
    ) -> Tensor:
        """Return molar internal energy in J/mol."""
        return self.kernel.molar_internal_energy(
            temperature,
            molar_density,
            self._composition,
        )

    def molar_enthalpy(self, temperature: Tensor, molar_density: Tensor) -> Tensor:
        """Return molar enthalpy in J/mol."""
        return self.kernel.molar_enthalpy(temperature, molar_density, self._composition)

    def molar_gibbs_energy(
        self,
        temperature: Tensor,
        molar_density: Tensor,
    ) -> Tensor:
        """Return molar Gibbs energy in J/mol."""
        return self.kernel.molar_gibbs_energy(
            temperature,
            molar_density,
            self._composition,
        )

    def molar_heat_capacity_cv(
        self,
        temperature: Tensor,
        molar_density: Tensor,
    ) -> Tensor:
        """Return isochoric molar heat capacity in J/(mol K)."""
        return self.kernel.molar_heat_capacity_cv(
            temperature,
            molar_density,
            self._composition,
        )

    def molar_heat_capacity_cp(
        self,
        temperature: Tensor,
        molar_density: Tensor,
    ) -> Tensor:
        """Return isobaric molar heat capacity in J/(mol K)."""
        return self.kernel.molar_heat_capacity_cp(
            temperature,
            molar_density,
            self._composition,
        )

    def speed_of_sound(self, temperature: Tensor, molar_density: Tensor) -> Tensor:
        """Return homogeneous-phase speed of sound in m/s."""
        return self.kernel.speed_of_sound(
            temperature,
            molar_density,
            self._composition,
        )


__all__ = [
    "PureFluidHelmholtzEOS",
    "PureFluidHelmholtzMetadata",
]
