"""Classical, predictive, and excess-Gibbs-energy cubic mixing rules.

The infinite-pressure rule is due to Huron and Vidal, *Fluid Phase
Equilibria* 3 (1979), 255-271, doi:10.1016/0378-3812(79)80001-1.
The cross-covolume convention follows Privat and Jaubert, *Fluid Phase
Equilibria* 570 (2023), 113697, doi:10.1016/j.fluid.2022.113697.
PPR78 follows Jaubert and Mutelet, *Fluid Phase Equilibria* 224 (2004),
285-304, doi:10.1016/j.fluid.2004.06.059.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, nn

from torch_flash.constants import R
from torch_flash.types import normalize_composition


class ActivityModel(Protocol):
    """Protocol needed by the Huron-Vidal mixing rule."""

    def excess_gibbs_rt(self, temperature: Tensor, composition: Tensor) -> Tensor:
        """Return ``gE/(RT)``."""


class QuadraticMixing(nn.Module):
    """van der Waals one-fluid mixing with ``kij`` and ``lij`` parameters.

    The unlike attraction and covolume parameters are

    ``aij = sqrt(ai*aj)*(1-kij)`` and
    ``bij = 0.5*(bi+bj)*(1-lij)``, respectively.

    ``lij=0`` recovers the conventional linear covolume rule exactly and uses
    a dedicated O(N) evaluation path.
    This convention is also exposed by ThermoPack's ``get_lij``/``set_lij``
    cubic interface.

    Parameters
    ----------
    kij
        Symmetric dimensionless attraction-interaction matrix.
    lij
        Optional symmetric dimensionless cross-covolume interactions.
    trainable, trainable_lij
        Register attraction or covolume interactions as trainable parameters.
    """

    def __init__(
        self,
        kij: Tensor,
        lij: Tensor | None = None,
        *,
        trainable: bool = False,
        trainable_lij: bool = False,
    ) -> None:
        super().__init__()
        if kij.ndim != 2 or kij.shape[0] != kij.shape[1]:
            raise ValueError("kij must be a square matrix")
        if not bool(torch.isfinite(kij).all()):
            raise ValueError("kij must contain only finite values")
        if not torch.allclose(kij, kij.mT):
            raise ValueError("kij must be symmetric")
        if lij is None:
            lij = torch.zeros_like(kij)
        elif lij.shape != kij.shape:
            raise ValueError("lij must have the same square shape as kij")
        elif not bool(torch.isfinite(lij).all()):
            raise ValueError("lij must contain only finite values")
        elif not torch.allclose(lij, lij.mT):
            raise ValueError("lij must be symmetric")
        if trainable:
            self.raw_kij = nn.Parameter(kij.clone())
        else:
            self.register_buffer("raw_kij", kij.clone())
        if trainable_lij:
            self.raw_lij = nn.Parameter(lij.clone())
        else:
            self.register_buffer("raw_lij", lij.clone())
        self._linear_covolume = not trainable_lij and not bool(torch.count_nonzero(lij))

    @staticmethod
    def _symmetric_zero_diagonal(values: Tensor) -> Tensor:
        symmetric = 0.5 * (values + values.mT)
        return symmetric - torch.diag_embed(torch.diagonal(symmetric))

    @property
    def kij(self) -> Tensor:
        """Symmetrized interaction matrix with an exactly zero diagonal."""
        return self._symmetric_zero_diagonal(self.raw_kij)

    @property
    def lij(self) -> Tensor:
        """Symmetrized covolume interaction matrix with zero diagonal."""
        return self._symmetric_zero_diagonal(self.raw_lij)

    def forward(
        self,
        temperature: Tensor,
        composition: Tensor,
        pure_a: Tensor,
        pure_b: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return mixture ``a`` and ``b`` parameters."""
        del temperature
        x = normalize_composition(composition)
        aij = self.cross_a(pure_a)
        am = torch.einsum("...i,...ij,...j->...", x, aij, x)
        if self._linear_covolume:
            bm = torch.sum(x * pure_b, dim=-1)
        else:
            bm = torch.einsum("...i,...ij,...j->...", x, self.cross_b(pure_b), x)
        return am, bm

    def cross_a(self, pure_a: Tensor) -> Tensor:
        """Return the matrix of unlike energy parameters ``aij``."""
        return torch.sqrt(pure_a[..., :, None] * pure_a[..., None, :]) * (1.0 - self.kij)

    def cross_b(self, pure_b: Tensor) -> Tensor:
        """Return unlike covolumes ``bij = (bi+bj)*(1-lij)/2``."""
        return 0.5 * (pure_b[..., :, None] + pure_b[..., None, :]) * (1.0 - self.lij)

    def partial_b(self, composition: Tensor, pure_b: Tensor) -> Tensor:
        """Return partial molar covolumes ``d(n*b_mix)/d(n_i)``."""
        x = normalize_composition(composition)
        if self._linear_covolume:
            return pure_b + torch.zeros_like(x)
        bij = self.cross_b(pure_b)
        bm = torch.einsum("...i,...ij,...j->...", x, bij, x)
        return 2.0 * torch.einsum("...j,...ij->...i", x, bij) - bm[..., None]


class TemperatureDependentQuadraticMixing(nn.Module):
    """van der Waals mixing with ``kij(T) = Aij + Bij/T``.

    ``Aij`` is dimensionless and ``Bij`` is in kelvin. This form is used by
    Yan et al. for petroleum CPA calculations involving light hydrocarbons
    and water (Fluid Phase Equilibria 276, 2009, 75-85,
    doi:10.1016/j.fluid.2008.10.007).

    Parameters
    ----------
    a
        Symmetric dimensionless ``Aij`` matrix.
    b
        Symmetric ``Bij`` matrix in K.
    lij
        Optional symmetric dimensionless cross-covolume interactions.
    trainable, trainable_lij
        Register attraction or covolume interactions as trainable parameters.
    """

    def __init__(
        self,
        a: Tensor,
        b: Tensor,
        lij: Tensor | None = None,
        *,
        trainable: bool = False,
        trainable_lij: bool = False,
    ) -> None:
        super().__init__()
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("temperature-dependent kij A must be a square matrix")
        if b.shape != a.shape:
            raise ValueError("temperature-dependent kij A and B matrices must have equal shapes")
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            raise ValueError("temperature-dependent kij A and B matrices must be finite")
        if not torch.allclose(a, a.mT) or not torch.allclose(b, b.mT):
            raise ValueError("temperature-dependent kij A and B matrices must be symmetric")
        if lij is None:
            lij = torch.zeros_like(a)
        elif lij.shape != a.shape:
            raise ValueError("lij must have the same square shape as temperature-dependent kij")
        elif not bool(torch.isfinite(lij).all()):
            raise ValueError("lij must contain only finite values")
        elif not torch.allclose(lij, lij.mT):
            raise ValueError("lij must be symmetric")
        if trainable:
            self.raw_a = nn.Parameter(a.clone())
            self.raw_b = nn.Parameter(b.clone())
        else:
            self.register_buffer("raw_a", a.clone())
            self.register_buffer("raw_b", b.clone())
        if trainable_lij:
            self.raw_lij = nn.Parameter(lij.clone())
        else:
            self.register_buffer("raw_lij", lij.clone())
        self._linear_covolume = not trainable_lij and not bool(torch.count_nonzero(lij))

    @staticmethod
    def _symmetric_zero_diagonal(values: Tensor) -> Tensor:
        symmetric = 0.5 * (values + values.mT)
        return symmetric - torch.diag_embed(torch.diagonal(symmetric))

    @property
    def a(self) -> Tensor:
        """Return the symmetric dimensionless ``Aij`` matrix."""
        return self._symmetric_zero_diagonal(self.raw_a)

    @property
    def b(self) -> Tensor:
        """Return the symmetric ``Bij`` matrix in kelvin."""
        return self._symmetric_zero_diagonal(self.raw_b)

    @property
    def lij(self) -> Tensor:
        """Return the symmetric dimensionless covolume interaction matrix."""
        return self._symmetric_zero_diagonal(self.raw_lij)

    def kij(self, temperature: Tensor) -> Tensor:
        """Evaluate the interaction matrix at temperature in kelvin."""
        if bool((~torch.isfinite(temperature) | (temperature <= 0.0)).any()):
            raise ValueError("temperature must be finite and positive")
        return self.a + self.b / temperature[..., None, None]

    def forward(
        self,
        temperature: Tensor,
        composition: Tensor,
        pure_a: Tensor,
        pure_b: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return mixture ``a`` and ``b`` parameters."""
        x = normalize_composition(composition)
        aij = torch.sqrt(pure_a[..., :, None] * pure_a[..., None, :]) * (
            1.0 - self.kij(temperature)
        )
        am = torch.einsum("...i,...ij,...j->...", x, aij, x)
        if self._linear_covolume:
            bm = torch.sum(x * pure_b, dim=-1)
        else:
            bm = torch.einsum("...i,...ij,...j->...", x, self.cross_b(pure_b), x)
        return am, bm

    def cross_a(self, temperature: Tensor, pure_a: Tensor) -> Tensor:
        """Return temperature-dependent unlike energy parameters."""
        return torch.sqrt(pure_a[..., :, None] * pure_a[..., None, :]) * (
            1.0 - self.kij(temperature)
        )

    def cross_b(self, pure_b: Tensor) -> Tensor:
        """Return unlike covolumes ``bij = (bi+bj)*(1-lij)/2``."""
        return 0.5 * (pure_b[..., :, None] + pure_b[..., None, :]) * (1.0 - self.lij)

    def partial_b(self, composition: Tensor, pure_b: Tensor) -> Tensor:
        """Return partial molar covolumes ``d(n*b_mix)/d(n_i)``."""
        x = normalize_composition(composition)
        if self._linear_covolume:
            return pure_b + torch.zeros_like(x)
        bij = self.cross_b(pure_b)
        bm = torch.einsum("...i,...ij,...j->...", x, bij, x)
        return 2.0 * torch.einsum("...j,...ij->...i", x, bij) - bm[..., None]


class PPR78Mixing(nn.Module):
    r"""Predictive PR78 group-contribution attraction mixing.

    This is Eq. (5) of Jaubert and Mutelet (2004),
    doi:10.1016/j.fluid.2004.06.059. For component group fractions
    :math:`\\alpha_{ik}`, the method evaluates

    .. math::

       k_{ij}(T) =
       \\frac{
       -\\frac12\\sum_{kl}\\Delta\\alpha_{ij,k}\\Delta\\alpha_{ij,l}
       A_{kl}(T_r/T)^{B_{kl}/A_{kl}-1}
       -(\\sqrt{a_i}/b_i-\\sqrt{a_j}/b_j)^2
       }{
       2\\sqrt{a_i a_j}/(b_i b_j)
       }.

    ``group_a`` and ``group_b`` are symmetric pressure-valued group
    interaction matrices. Only their unique off-diagonal entries are stored,
    avoiding redundant degrees of freedom during fitting. The paper derives
    the correlation with the conventional linear covolume rule, which this
    class preserves.

    Parameters
    ----------
    group_fractions
        Normalized component-by-group fraction matrix.
    group_a, group_b
        Symmetric pressure-valued universal interaction matrices in Pa.
    reference_temperature
        Positive correlation reference temperature in K.
    trainable
        Register unique off-diagonal group interactions as trainable
        parameters.
    parameter_set
        Optional source parameter-set identifier.
    """

    group_fractions: Tensor
    pair_indices: Tensor
    reference_temperature: Tensor
    raw_group_a: Tensor
    raw_group_b: Tensor

    def __init__(
        self,
        group_fractions: Tensor,
        group_a: Tensor,
        group_b: Tensor,
        *,
        reference_temperature: float = 298.15,
        trainable: bool = False,
        parameter_set: str | None = None,
    ) -> None:
        super().__init__()
        if group_fractions.ndim != 2:
            raise ValueError("PPR78 group_fractions must be a component-by-group matrix")
        component_count, group_count = group_fractions.shape
        if component_count == 0 or group_count < 2:
            raise ValueError("PPR78 requires at least one component and two groups")
        if group_a.shape != (group_count, group_count) or group_b.shape != group_a.shape:
            raise ValueError("PPR78 group A and B must be square matrices matching the groups")
        if not bool(
            torch.isfinite(group_fractions).all()
            & torch.isfinite(group_a).all()
            & torch.isfinite(group_b).all()
        ):
            raise ValueError("PPR78 fractions and group interactions must be finite")
        if bool((group_fractions < 0.0).any()):
            raise ValueError("PPR78 group fractions cannot be negative")
        row_sums = group_fractions.sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums)):
            raise ValueError("each PPR78 component group-fraction row must sum to one")
        if not torch.allclose(group_a, group_a.mT) or not torch.allclose(group_b, group_b.mT):
            raise ValueError("PPR78 group A and B matrices must be symmetric")
        if bool(torch.diagonal(group_a).count_nonzero()) or bool(
            torch.diagonal(group_b).count_nonzero()
        ):
            raise ValueError("PPR78 group A and B matrices must have zero diagonals")
        if not torch.isfinite(torch.tensor(reference_temperature)) or reference_temperature <= 0.0:
            raise ValueError("PPR78 reference_temperature must be finite and positive")
        undefined = (group_a == 0.0) & (group_b != 0.0)
        if bool(undefined.any()):
            raise ValueError("PPR78 group B must be zero wherever group A is zero")

        indices = torch.triu_indices(
            group_count,
            group_count,
            offset=1,
            device=group_a.device,
        )
        self.register_buffer("group_fractions", group_fractions.clone())
        self.register_buffer("pair_indices", indices)
        self.register_buffer(
            "reference_temperature",
            group_a.new_tensor(reference_temperature),
        )
        if trainable:
            self.raw_group_a = nn.Parameter(group_a[indices[0], indices[1]].clone())
            self.raw_group_b = nn.Parameter(group_b[indices[0], indices[1]].clone())
        else:
            self.register_buffer(
                "raw_group_a",
                group_a[indices[0], indices[1]].clone(),
            )
            self.register_buffer(
                "raw_group_b",
                group_b[indices[0], indices[1]].clone(),
            )
        self.parameter_set = parameter_set

    @property
    def ncomponents(self) -> int:
        """Number of component decompositions."""
        return int(self.group_fractions.shape[0])

    @property
    def ngroups(self) -> int:
        """Number of structural groups."""
        return int(self.group_fractions.shape[1])

    def _symmetric_matrix(self, values: Tensor) -> Tensor:
        matrix = values.new_zeros((self.ngroups, self.ngroups))
        matrix = matrix.index_put(
            (self.pair_indices[0], self.pair_indices[1]),
            values,
        )
        return matrix + matrix.mT

    @property
    def group_a(self) -> Tensor:
        """Return the symmetric :math:`A_{kl}` matrix in pascals."""
        return self._symmetric_matrix(self.raw_group_a)

    @property
    def group_b(self) -> Tensor:
        """Return the symmetric :math:`B_{kl}` matrix in pascals."""
        return self._symmetric_matrix(self.raw_group_b)

    def group_interaction_energy(self, temperature: Tensor) -> Tensor:
        """Return :math:`A_{kl}(298.15/T)^{B_{kl}/A_{kl}-1}` in pascals."""
        if bool((~torch.isfinite(temperature) | (temperature <= 0.0)).any()):
            raise ValueError("temperature must be finite and positive")
        group_a = self.group_a
        group_b = self.group_b
        is_zero = group_a == 0.0
        safe_a = torch.where(is_zero, torch.ones_like(group_a), group_a)
        exponent = group_b / safe_a - 1.0
        ratio = self.reference_temperature / temperature[..., None, None]
        value = group_a * ratio.pow(exponent)
        return torch.where(is_zero, torch.zeros_like(value), value)

    def kij(
        self,
        temperature: Tensor,
        pure_a: Tensor,
        pure_b: Tensor,
    ) -> Tensor:
        """Evaluate the symmetric component BIP matrix.

        ``pure_a`` and ``pure_b`` must be the PR78 pure-component parameters
        at ``temperature``. This dependence is essential: PPR78 is not a
        standalone ``A + B/T`` law.
        """
        if pure_a.shape[-1] != self.ncomponents or pure_b.shape[-1] != self.ncomponents:
            raise ValueError("PPR78 pure parameters must have one value per component")
        if not bool(
            torch.isfinite(pure_a).all()
            & torch.isfinite(pure_b).all()
            & (pure_a > 0.0).all()
            & (pure_b > 0.0).all()
        ):
            raise ValueError("PPR78 pure a and b parameters must be finite and positive")

        differences = self.group_fractions[:, None, :] - self.group_fractions[None, :, :]
        group_energy = self.group_interaction_energy(temperature)
        group_term = -0.5 * torch.einsum(
            "ijg,...gh,ijh->...ij",
            differences,
            group_energy,
            differences,
        )
        delta = torch.sqrt(pure_a) / pure_b
        pure_term = (delta[..., :, None] - delta[..., None, :]).square()
        denominator = (
            2.0
            * torch.sqrt(pure_a[..., :, None] * pure_a[..., None, :])
            / (pure_b[..., :, None] * pure_b[..., None, :])
        )
        result = (group_term - pure_term) / denominator
        return result - torch.diag_embed(torch.diagonal(result, dim1=-2, dim2=-1))

    def cross_a(
        self,
        temperature: Tensor,
        pure_a: Tensor,
        pure_b: Tensor,
    ) -> Tensor:
        """Return PPR78 unlike attraction parameters."""
        return torch.sqrt(pure_a[..., :, None] * pure_a[..., None, :]) * (
            1.0 - self.kij(temperature, pure_a, pure_b)
        )

    def partial_b(self, composition: Tensor, pure_b: Tensor) -> Tensor:
        """Return partial covolumes for the PPR78 linear covolume rule."""
        x = normalize_composition(composition)
        return pure_b + torch.zeros_like(x)

    def forward(
        self,
        temperature: Tensor,
        composition: Tensor,
        pure_a: Tensor,
        pure_b: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return PPR78 mixture ``a`` and linear mixture ``b``."""
        x = normalize_composition(composition)
        aij = self.cross_a(temperature, pure_a, pure_b)
        am = torch.einsum("...i,...ij,...j->...", x, aij, x)
        bm = torch.sum(x * pure_b, dim=-1)
        return am, bm


class HuronVidalMixing(nn.Module):
    """Original infinite-pressure Huron-Vidal mixing rule.

    The implementation follows Michelsen and Mollerup, chapter 7, Eq. 7:

    ``a/(bRT) = sum(x_i a_i/(b_i RT)) - gE/(Delta RT)``.

    Primary source: Huron and Vidal (1979),
    doi:10.1016/0378-3812(79)80001-1. The implementation convention follows
    Michelsen and Mollerup, *Thermodynamic Models*, 2nd ed. (2007),
    chapter 7, ISBN 978-87-989961-3-2.

    Parameters
    ----------
    activity_model
        Excess-Gibbs model compatible with the HV infinite-pressure reference.
    delta1, delta2
        Distinct generalized-cubic denominator-shape constants.
    """

    def __init__(
        self,
        activity_model: ActivityModel,
        *,
        delta1: float,
        delta2: float,
    ) -> None:
        super().__init__()
        if delta1 == delta2:
            raise ValueError("cubic delta parameters must be distinct")
        self.activity_model = activity_model  # type: ignore[assignment]
        self.delta1 = float(delta1)
        self.delta2 = float(delta2)
        ratio = (1.0 + delta2) / (1.0 + delta1)
        self.delta = float(torch.log(torch.tensor(ratio)) / (delta2 - delta1))

    def forward(
        self,
        temperature: Tensor,
        composition: Tensor,
        pure_a: Tensor,
        pure_b: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return mixture ``a`` and ``b`` parameters."""
        x = normalize_composition(composition)
        bm = torch.sum(x * pure_b, dim=-1)
        pure_alpha = pure_a / (pure_b * R * temperature[..., None])
        ge_rt = self.activity_model.excess_gibbs_rt(temperature, x)
        alpha = torch.sum(x * pure_alpha, dim=-1) - ge_rt / self.delta
        return alpha * bm * R * temperature, bm
