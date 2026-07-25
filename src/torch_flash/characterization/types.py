"""Data structures shared by heavy-end characterization methods."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor


@dataclass(frozen=True)
class PseudoComponentCut:
    """One measured or estimated heavy-end cut in SI units.

    Attributes
    ----------
    name:
        Nonempty pseudo-component identifier.
    mole_fraction:
        Nonnegative fraction of the complete fluid composition.
    normal_boiling_temperature:
        Normal boiling temperature in K.
    specific_gravity:
        Dimensionless liquid specific gravity under the source convention.
    molar_mass:
        Molar mass in kg/mol.
    """

    name: str
    mole_fraction: float
    normal_boiling_temperature: float
    specific_gravity: float
    molar_mass: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("pseudo-component cut name must be a non-empty string")
        values = (
            self.mole_fraction,
            self.normal_boiling_temperature,
            self.specific_gravity,
            self.molar_mass,
        )
        if any(not isfinite(value) for value in values):
            raise ValueError("pseudo-component cut values must be finite")
        if self.mole_fraction < 0.0:
            raise ValueError("pseudo-component cut mole fraction must be nonnegative")
        if any(value <= 0.0 for value in values[1:]):
            raise ValueError(
                "pseudo-component cut temperature, gravity, and molar mass must be positive"
            )


@dataclass(frozen=True)
class SCNDistribution:
    """Discrete single-carbon-number representation of a plus fraction.

    Attributes
    ----------
    carbon_numbers:
        Strictly increasing one-dimensional integer-like carbon-number labels.
    mole_fractions:
        Nonnegative cut mole fractions; their sum is the represented plus
        fraction, not generally one.
    molar_masses:
        Positive cut molar masses in kg/mol.
    densities:
        Optional positive cut mass densities in kg/m3.
    """

    carbon_numbers: Tensor
    mole_fractions: Tensor
    molar_masses: Tensor
    densities: Tensor | None = None

    def __post_init__(self) -> None:
        vectors = (self.carbon_numbers, self.mole_fractions, self.molar_masses)
        if any(value.ndim != 1 for value in vectors):
            raise ValueError("SCN distribution values must be one-dimensional")
        if not self.carbon_numbers.numel() or any(
            value.shape != self.carbon_numbers.shape for value in vectors[1:]
        ):
            raise ValueError("SCN distribution vectors must have the same nonzero length")
        if self.densities is not None and self.densities.shape != self.carbon_numbers.shape:
            raise ValueError("SCN densities must match the carbon-number vector")
        continuous = [self.mole_fractions, self.molar_masses]
        if self.densities is not None:
            continuous.append(self.densities)
        if any(not bool(torch.isfinite(value).all()) for value in continuous):
            raise ValueError("SCN distribution values must be finite")
        if bool((self.mole_fractions < 0.0).any()) or not bool(self.mole_fractions.sum() > 0.0):
            raise ValueError("SCN mole fractions must be nonnegative with a positive sum")
        if bool((self.molar_masses <= 0.0).any()):
            raise ValueError("SCN molar masses must be positive")
        if self.densities is not None and bool((self.densities <= 0.0).any()):
            raise ValueError("SCN densities must be positive")
        if self.carbon_numbers.numel() > 1 and not bool(
            (self.carbon_numbers[1:] > self.carbon_numbers[:-1]).all()
        ):
            raise ValueError("SCN carbon numbers must be strictly increasing")

    @property
    def total_mole_fraction(self) -> Tensor:
        """Return the total feed fraction represented by all SCN cuts.

        Returns
        -------
        Tensor
            Scalar sum of ``mole_fractions``.
        """
        return self.mole_fractions.sum()

    @property
    def average_molar_mass(self) -> Tensor:
        """Return the mole-average molar mass.

        Returns
        -------
        Tensor
            Plus-fraction average in kg/mol.
        """
        return torch.sum(self.mole_fractions * self.molar_masses) / self.total_mole_fraction

    @property
    def bulk_density(self) -> Tensor | None:
        """Return ideal-volume-mixed bulk mass density.

        Returns
        -------
        Tensor or None
            Density in kg/m3, or ``None`` when cut densities are unavailable.
        """
        if self.densities is None:
            return None
        mass = self.mole_fractions * self.molar_masses
        return mass.sum() / torch.sum(mass / self.densities)

    def to(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> SCNDistribution:
        """Return the distribution on a common dtype and device.

        Parameters
        ----------
        dtype:
            Target floating dtype for fractions, masses, and densities.
        device:
            Target device for every tensor.

        Returns
        -------
        SCNDistribution
            New immutable distribution. Carbon numbers retain integer dtype.
        """
        return SCNDistribution(
            self.carbon_numbers.to(device=device),
            self.mole_fractions.to(dtype=dtype, device=device),
            self.molar_masses.to(dtype=dtype, device=device),
            None if self.densities is None else self.densities.to(dtype=dtype, device=device),
        )


@dataclass(frozen=True)
class LumpedDistribution:
    """Contiguous pseudo-components produced by a lumping rule.

    Attributes
    ----------
    names:
        Pseudo-component labels in increasing carbon-number order.
    carbon_number_bounds:
        Inclusive lower/upper carbon-number bounds for each lump.
    mole_fractions:
        Lump mole fractions.
    molar_masses:
        Mole-averaged lump molar masses in kg/mol.
    densities:
        Ideal-volume-preserving lump densities in kg/m3, or ``None``.
    properties:
        Additional named, mass-weighted property tensors.
    """

    names: tuple[str, ...]
    carbon_number_bounds: tuple[tuple[int, int], ...]
    mole_fractions: Tensor
    molar_masses: Tensor
    densities: Tensor | None
    properties: dict[str, Tensor]
