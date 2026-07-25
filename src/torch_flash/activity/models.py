"""Differentiable activity-coefficient model kernels.

Primary model definitions are Renon and Prausnitz (1968),
doi:10.1002/aic.690140124, for NRTL and Wilson (1964),
doi:10.1021/ja01056a002, for the Wilson equation. The covolume-weighted NRTL
variant is the infinite-pressure construction of Huron and Vidal (1979),
doi:10.1016/0378-3812(79)80001-1.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from torch_flash.constants import R
from torch_flash.types import normalize_composition


def _huron_vidal_excess_gibbs_rt(
    composition: Tensor,
    tau: Tensor,
    nonrandomness: Tensor,
    covolumes: Tensor,
) -> Tensor:
    """Evaluate the covolume-weighted local-composition expression."""
    weights = torch.exp(-nonrandomness * tau)
    weighted_composition = composition * covolumes
    denominator = torch.einsum("...k,...ki->...i", weighted_composition, weights)
    numerator = torch.einsum(
        "...j,...ji,...ji->...i",
        weighted_composition,
        tau,
        weights,
    )
    return torch.sum(composition * numerator / denominator, dim=-1)


class NRTL(nn.Module):
    r"""Non-random two-liquid excess Gibbs energy model.

    Parameters
    ----------
    interaction
        Matrix storing :math:`g_{ij}-g_{jj}` in J/mol.
    nonrandomness
        Dimensionless :math:`\alpha_{ij}` matrix.
    trainable
        Register the interaction-energy matrix as a trainable parameter.

    References
    ----------
    Renon and Prausnitz, *AIChE Journal* 14 (1968), 135-144,
    doi:10.1002/aic.690140124.
    """

    interaction: Tensor
    nonrandomness: Tensor

    def __init__(
        self,
        interaction: Tensor,
        nonrandomness: Tensor,
        *,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if interaction.shape != nonrandomness.shape or interaction.ndim != 2:
            raise ValueError("NRTL parameter arrays must be equally sized matrices")
        value = interaction.clone()
        if trainable:
            self.interaction = nn.Parameter(value)
        else:
            self.register_buffer("interaction", value)
        self.register_buffer("nonrandomness", nonrandomness.clone())

    def excess_gibbs_rt(self, temperature: Tensor, composition: Tensor) -> Tensor:
        """Evaluate dimensionless molar excess Gibbs energy.

        Parameters
        ----------
        temperature
            Temperature in K.
        composition
            Mole fractions with components on the final axis.

        Returns
        -------
        Tensor
            Dimensionless ``g^E/(RT)`` with the broadcast leading shape.
        """
        x = normalize_composition(composition)
        tau = self.interaction / (R * temperature[..., None, None])
        weights = torch.exp(-self.nonrandomness * tau)
        denominator = torch.einsum("...k,...ki->...i", x, weights)
        numerator = torch.einsum("...j,...ji,...ji->...i", x, tau, weights)
        return torch.sum(x * numerator / denominator, dim=-1)

    def log_activity_coefficients(self, temperature: Tensor, composition: Tensor) -> Tensor:
        """Return component log activity coefficients.

        Parameters
        ----------
        temperature
            Temperature in K.
        composition
            Mole fractions with components on the final axis.

        Returns
        -------
        Tensor
            Dimensionless ``ln(gamma_i)`` obtained from derivatives of the
            extensive excess Gibbs energy.
        """
        x = normalize_composition(composition)

        def extensive(moles: Tensor) -> Tensor:
            total = moles.sum(dim=-1)
            fractions = moles / total[..., None]
            return (total * self.excess_gibbs_rt(temperature, fractions)).sum()

        result: Tensor = torch.func.grad(extensive)(x)
        return result


class HuronVidalNRTL(nn.Module):
    r"""Covolume-weighted NRTL model used by the original HV formulation.

    The dimensionless interaction energy is

    .. math::

        \tau_{ji}(T) = A_{ji}/T + B_{ji},

    where ``energy_over_r`` stores :math:`A` in kelvin and
    ``temperature_coefficient`` stores the dimensionless linear coefficient
    :math:`B`.  The excess Gibbs energy is the modified NRTL expression from
    Huron and Vidal (1979), with :math:`x_j b_j` replacing :math:`x_j` in the
    local-composition sums.  Consequently its parameters must be fitted for
    the HV infinite-pressure reference; ordinary low-pressure NRTL parameters
    are not interchangeable.

    Reference: Huron and Vidal, *Fluid Phase Equilibria* 3 (1979), 255-271,
    doi:10.1016/0378-3812(79)80001-1.

    Parameters
    ----------
    energy_over_r
        Matrix ``A`` in K.
    temperature_coefficient
        Dimensionless matrix ``B``.
    nonrandomness
        Dimensionless NRTL nonrandomness matrix.
    covolumes
        Positive component cubic covolumes in m3/mol.
    trainable
        Register ``A`` and ``B`` as trainable parameters.
    """

    energy_over_r: Tensor
    temperature_coefficient: Tensor
    nonrandomness: Tensor
    covolumes: Tensor

    def __init__(
        self,
        energy_over_r: Tensor,
        temperature_coefficient: Tensor,
        nonrandomness: Tensor,
        covolumes: Tensor,
        *,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if (
            energy_over_r.ndim != 2
            or energy_over_r.shape[0] != energy_over_r.shape[1]
            or temperature_coefficient.shape != energy_over_r.shape
            or nonrandomness.shape != energy_over_r.shape
        ):
            raise ValueError("HV-NRTL interaction arrays must be equally sized square matrices")
        if covolumes.shape != (energy_over_r.shape[0],):
            raise ValueError("one positive HV-NRTL covolume is required per component")
        if bool((covolumes <= 0.0).any()):
            raise ValueError("HV-NRTL covolumes must be positive")
        if trainable:
            self.energy_over_r = nn.Parameter(energy_over_r.clone())
            self.temperature_coefficient = nn.Parameter(temperature_coefficient.clone())
        else:
            self.register_buffer("energy_over_r", energy_over_r.clone())
            self.register_buffer("temperature_coefficient", temperature_coefficient.clone())
        self.register_buffer("nonrandomness", nonrandomness.clone())
        self.register_buffer("covolumes", covolumes.clone())

    def tau_matrix(self, temperature: Tensor) -> Tensor:
        """Evaluate the dimensionless HV-NRTL interaction matrix.

        Parameters
        ----------
        temperature
            Temperature in K.

        Returns
        -------
        Tensor
            ``tau(T) = A/T + B`` with trailing component-pair axes.
        """
        return self.energy_over_r / temperature[..., None, None] + self.temperature_coefficient

    def excess_gibbs_rt(self, temperature: Tensor, composition: Tensor) -> Tensor:
        """Evaluate covolume-weighted HV excess Gibbs energy.

        Parameters
        ----------
        temperature
            Temperature in K.
        composition
            Mole fractions on the final axis.

        Returns
        -------
        Tensor
            Dimensionless molar ``g^E/(RT)``.
        """
        x = normalize_composition(composition)
        tau = self.tau_matrix(temperature)
        return _huron_vidal_excess_gibbs_rt(
            x,
            tau,
            self.nonrandomness,
            self.covolumes,
        )

    def log_activity_coefficients(self, temperature: Tensor, composition: Tensor) -> Tensor:
        """Return HV-NRTL component log activity coefficients.

        Parameters
        ----------
        temperature
            Temperature in K.
        composition
            Mole fractions on the final axis.

        Returns
        -------
        Tensor
            Dimensionless ``ln(gamma_i)`` from the extensive HV excess Gibbs
            energy derivative.
        """
        x = normalize_composition(composition)

        def extensive(moles: Tensor) -> Tensor:
            total = moles.sum(dim=-1)
            fractions = moles / total[..., None]
            return (total * self.excess_gibbs_rt(temperature, fractions)).sum()

        result: Tensor = torch.func.grad(extensive)(x)
        return result


class AnchoredHuronVidalNRTL(nn.Module):
    r"""Fit-ready HV-NRTL model anchored at two temperatures.

    Directly regressing :math:`A_{ij}` and :math:`B_{ij}` in
    :math:`\tau_{ij}=A_{ij}/T+B_{ij}` is poorly scaled and can be strongly
    correlated over a finite experimental temperature interval. This exactly
    equivalent parameterization stores trainable values of :math:`\tau_{ij}`
    at two user-selected temperatures and interpolates linearly in
    :math:`1/T`.

    The optional trainable non-randomness matrix is kept symmetric and within
    explicit open bounds by a sigmoid transform. Diagonal interactions remain
    exactly zero. Call :meth:`freeze` after fitting to obtain the standard
    :class:`HuronVidalNRTL` representation used by parameter databases.

    Parameters
    ----------
    tau_at_lower_temperature, tau_at_upper_temperature
        Dimensionless interaction matrices at the two anchors.
    nonrandomness
        Fixed or initially bounded-symmetric nonrandomness matrix.
    covolumes
        Positive component cubic covolumes in m3/mol.
    lower_temperature, upper_temperature
        Positive increasing anchor temperatures in K.
    trainable_nonrandomness
        Fit the symmetric nonrandomness matrix through a bounded transform.
    nonrandomness_bounds
        Open lower/upper bounds for fitted off-diagonal nonrandomness.
    """

    raw_tau_at_lower_temperature: nn.Parameter
    raw_tau_at_upper_temperature: nn.Parameter
    raw_nonrandomness: nn.Parameter | None
    lower_temperature: Tensor
    upper_temperature: Tensor
    covolumes: Tensor
    _off_diagonal: Tensor
    _fixed_nonrandomness: Tensor

    def __init__(
        self,
        tau_at_lower_temperature: Tensor,
        tau_at_upper_temperature: Tensor,
        nonrandomness: Tensor,
        covolumes: Tensor,
        *,
        lower_temperature: Tensor | float,
        upper_temperature: Tensor | float,
        trainable_nonrandomness: bool = False,
        nonrandomness_bounds: tuple[float, float] = (0.05, 0.60),
    ) -> None:
        super().__init__()
        if (
            tau_at_lower_temperature.ndim != 2
            or tau_at_lower_temperature.shape[0] != tau_at_lower_temperature.shape[1]
            or tau_at_upper_temperature.shape != tau_at_lower_temperature.shape
            or nonrandomness.shape != tau_at_lower_temperature.shape
        ):
            raise ValueError("anchored HV-NRTL arrays must be equally sized square matrices")
        if covolumes.shape != (tau_at_lower_temperature.shape[0],):
            raise ValueError("one positive anchored HV-NRTL covolume is required per component")
        if bool((covolumes <= 0.0).any()):
            raise ValueError("anchored HV-NRTL covolumes must be positive")

        lower = torch.as_tensor(
            lower_temperature,
            dtype=tau_at_lower_temperature.dtype,
            device=tau_at_lower_temperature.device,
        )
        upper = torch.as_tensor(
            upper_temperature,
            dtype=tau_at_lower_temperature.dtype,
            device=tau_at_lower_temperature.device,
        )
        if (
            lower.ndim != 0
            or upper.ndim != 0
            or not bool(
                torch.isfinite(lower) & torch.isfinite(upper) & (lower > 0.0) & (upper > lower)
            )
        ):
            raise ValueError(
                "anchored HV-NRTL temperatures must be finite, positive, and increasing"
            )

        self.raw_tau_at_lower_temperature = nn.Parameter(tau_at_lower_temperature.clone())
        self.raw_tau_at_upper_temperature = nn.Parameter(tau_at_upper_temperature.clone())
        self.register_buffer("lower_temperature", lower.clone())
        self.register_buffer("upper_temperature", upper.clone())
        self.register_buffer("covolumes", covolumes.clone())
        size = tau_at_lower_temperature.shape[0]
        mask = torch.ones(
            (size, size),
            dtype=tau_at_lower_temperature.dtype,
            device=tau_at_lower_temperature.device,
        ) - torch.eye(
            size,
            dtype=tau_at_lower_temperature.dtype,
            device=tau_at_lower_temperature.device,
        )
        self.register_buffer("_off_diagonal", mask)

        if trainable_nonrandomness:
            lower_bound, upper_bound = nonrandomness_bounds
            if (
                not 0.0 <= lower_bound < upper_bound
                or not torch.isfinite(torch.tensor([lower_bound, upper_bound])).all()
            ):
                raise ValueError(
                    "anchored HV-NRTL nonrandomness bounds must be finite and increasing"
                )
            off_diagonal_values = nonrandomness[mask.bool()]
            if not bool(
                (off_diagonal_values > lower_bound).all()
                & (off_diagonal_values < upper_bound).all()
            ):
                raise ValueError(
                    "initial trainable HV-NRTL nonrandomness must lie inside its bounds"
                )
            if not torch.allclose(nonrandomness, nonrandomness.mT):
                raise ValueError("trainable HV-NRTL nonrandomness must be symmetric")
            scaled = (nonrandomness - lower_bound) / (upper_bound - lower_bound)
            scaled = torch.where(mask.bool(), scaled, torch.full_like(scaled, 0.5))
            raw_nonrandomness = torch.log(scaled) - torch.log1p(-scaled)
            self.raw_nonrandomness = nn.Parameter(raw_nonrandomness)
            self.register_buffer(
                "_fixed_nonrandomness",
                torch.empty(
                    0,
                    dtype=nonrandomness.dtype,
                    device=nonrandomness.device,
                ),
            )
        else:
            self.register_parameter("raw_nonrandomness", None)
            self.register_buffer("_fixed_nonrandomness", nonrandomness.clone())
        self.nonrandomness_bounds = (float(nonrandomness_bounds[0]), float(nonrandomness_bounds[1]))

    @property
    def tau_at_lower_temperature(self) -> Tensor:
        """Return the lower-anchor interaction matrix with an exact zero diagonal."""
        return self.raw_tau_at_lower_temperature * self._off_diagonal

    @property
    def tau_at_upper_temperature(self) -> Tensor:
        """Return the upper-anchor interaction matrix with an exact zero diagonal."""
        return self.raw_tau_at_upper_temperature * self._off_diagonal

    @property
    def nonrandomness(self) -> Tensor:
        """Return the fixed or bounded-symmetric non-randomness matrix."""
        if self.raw_nonrandomness is None:
            return self._fixed_nonrandomness
        lower, upper = self.nonrandomness_bounds
        transformed = lower + (upper - lower) * torch.sigmoid(self.raw_nonrandomness)
        return 0.5 * (transformed + transformed.mT) * self._off_diagonal

    @property
    def energy_over_r(self) -> Tensor:
        """Return :math:`A_{ij}` in kelvin for ``tau = A/T + B``."""
        inverse_interval = self.lower_temperature.reciprocal() - self.upper_temperature.reciprocal()
        return (self.tau_at_lower_temperature - self.tau_at_upper_temperature) / inverse_interval

    @property
    def temperature_coefficient(self) -> Tensor:
        """Return dimensionless :math:`B_{ij}` for ``tau = A/T + B``."""
        return self.tau_at_lower_temperature - self.energy_over_r / self.lower_temperature

    def tau_matrix(self, temperature: Tensor) -> Tensor:
        """Evaluate interactions by interpolation in inverse temperature.

        Parameters
        ----------
        temperature
            Positive temperature in K.

        Returns
        -------
        Tensor
            Dimensionless interaction matrices with trailing component axes.

        Raises
        ------
        ValueError
            If any temperature is nonpositive.
        """
        if bool((temperature <= 0.0).any()):
            raise ValueError("HV-NRTL temperature must be positive")
        inverse_fraction = (temperature.reciprocal() - self.upper_temperature.reciprocal()) / (
            self.lower_temperature.reciprocal() - self.upper_temperature.reciprocal()
        )
        return (
            inverse_fraction[..., None, None] * self.tau_at_lower_temperature
            + (1.0 - inverse_fraction[..., None, None]) * self.tau_at_upper_temperature
        )

    def excess_gibbs_rt(self, temperature: Tensor, composition: Tensor) -> Tensor:
        """Evaluate anchored HV dimensionless excess Gibbs energy.

        Parameters
        ----------
        temperature
            Positive temperature in K.
        composition
            Mole fractions on the final axis.

        Returns
        -------
        Tensor
            Dimensionless molar ``g^E/(RT)``.
        """
        x = normalize_composition(composition)
        return _huron_vidal_excess_gibbs_rt(
            x,
            self.tau_matrix(temperature),
            self.nonrandomness,
            self.covolumes,
        )

    def log_activity_coefficients(self, temperature: Tensor, composition: Tensor) -> Tensor:
        """Return anchored HV component log activity coefficients.

        Parameters
        ----------
        temperature
            Positive temperature in K.
        composition
            Mole fractions on the final axis.

        Returns
        -------
        Tensor
            Dimensionless ``ln(gamma_i)``.
        """
        x = normalize_composition(composition)

        def extensive(moles: Tensor) -> Tensor:
            total = moles.sum(dim=-1)
            fractions = moles / total[..., None]
            return (total * self.excess_gibbs_rt(temperature, fractions)).sum()

        result: Tensor = torch.func.grad(extensive)(x)
        return result

    def freeze(self) -> HuronVidalNRTL:
        """Convert fitted anchor parameters to a detached standard HV model.

        Returns
        -------
        HuronVidalNRTL
            Prediction/storage model containing detached ``A``, ``B``,
            nonrandomness, and covolume tensors.
        """
        return HuronVidalNRTL(
            self.energy_over_r.detach(),
            self.temperature_coefficient.detach(),
            self.nonrandomness.detach(),
            self.covolumes.detach(),
        )


class Wilson(nn.Module):
    """Wilson excess Gibbs model with energy and molar-volume parameters.

    Parameters
    ----------
    interaction
        Square energy-difference matrix in J/mol.
    molar_volumes
        Positive pure-liquid molar volumes in m3/mol.
    trainable
        Register ``interaction`` as a trainable PyTorch parameter.

    References
    ----------
    Wilson, *Journal of the American Chemical Society* 86 (1964), 127-130,
    doi:10.1021/ja01056a002.
    """

    interaction: Tensor
    molar_volumes: Tensor

    def __init__(
        self,
        interaction: Tensor,
        molar_volumes: Tensor,
        *,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if interaction.ndim != 2 or interaction.shape[0] != interaction.shape[1]:
            raise ValueError("Wilson interaction must be a square matrix")
        if molar_volumes.shape != (interaction.shape[0],):
            raise ValueError("one Wilson molar volume is required per component")
        if trainable:
            self.interaction = nn.Parameter(interaction.clone())
        else:
            self.register_buffer("interaction", interaction.clone())
        self.register_buffer("molar_volumes", molar_volumes.clone())

    def lambda_matrix(self, temperature: Tensor) -> Tensor:
        """Evaluate the temperature-dependent Wilson Lambda matrix.

        Parameters
        ----------
        temperature
            Temperature in K.

        Returns
        -------
        Tensor
            Dimensionless Lambda matrices with trailing component-pair axes.
        """
        volume_ratio = self.molar_volumes[None, :] / self.molar_volumes[:, None]
        return volume_ratio * torch.exp(-self.interaction / (R * temperature[..., None, None]))

    def excess_gibbs_rt(self, temperature: Tensor, composition: Tensor) -> Tensor:
        """Evaluate dimensionless molar excess Gibbs energy.

        Parameters
        ----------
        temperature
            Temperature in K.
        composition
            Mole fractions on the final axis.

        Returns
        -------
        Tensor
            Dimensionless ``g^E/(RT)``.
        """
        x = normalize_composition(composition)
        lambda_matrix = self.lambda_matrix(temperature)
        sums = torch.einsum("...j,...ij->...i", x, lambda_matrix)
        return -torch.sum(x * torch.log(sums), dim=-1)

    def log_activity_coefficients(self, temperature: Tensor, composition: Tensor) -> Tensor:
        """Evaluate analytical Wilson log activity coefficients.

        Parameters
        ----------
        temperature
            Temperature in K.
        composition
            Mole fractions on the final axis.

        Returns
        -------
        Tensor
            Dimensionless ``ln(gamma_i)``.
        """
        x = normalize_composition(composition)
        lambda_matrix = self.lambda_matrix(temperature)
        row_sums = torch.einsum("...j,...ij->...i", x, lambda_matrix)
        correction = torch.einsum("...j,...ji,...j->...i", x, lambda_matrix, row_sums.reciprocal())
        return 1.0 - torch.log(row_sums) - correction
