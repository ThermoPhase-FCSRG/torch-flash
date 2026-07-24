"""Autodifferentiable generalized two-parameter cubic equations of state.

SRK is defined by Soave (1972), doi:10.1016/0009-2509(72)80096-4. PR76 is
defined by Peng and Robinson (1976), doi:10.1021/i160057a011. PR78 denotes
the acentric-factor extension in Robinson and Peng, GPA Research Report RR-28
(1978).

Cross-covolume interactions use Privat and Jaubert's reviewed convention,
doi:10.1016/j.fluid.2022.113697.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from torch_flash.components import ComponentSet
from torch_flash.constants import R
from torch_flash.database import ParameterSource, load_model_parameters
from torch_flash.exceptions import InvalidStateError, ParameterDatabaseError
from torch_flash.mixing import (
    HuronVidalMixing,
    PPR78Mixing,
    QuadraticMixing,
    TemperatureDependentQuadraticMixing,
)
from torch_flash.parameters.group_contribution import (
    DEFAULT_PPR78_GROUP_CONTRIBUTION,
    ppr78_mixing,
)
from torch_flash.types import PhaseKind, normalize_composition

from .volume_translation import VolumeTranslation

AlphaKind = Literal["srk", "pr76", "pr78"]
CubicMixing = QuadraticMixing | TemperatureDependentQuadraticMixing | PPR78Mixing | HuronVidalMixing


@dataclass(frozen=True)
class CubicConstants:
    """Definition of a generalized cubic EoS."""

    name: str
    omega_a: float
    omega_b: float
    delta1: float
    delta2: float
    alpha_kind: AlphaKind
    alpha_low: tuple[float, float, float]
    alpha_high: tuple[float, float, float, float] | None = None
    alpha_switch: float | None = None
    parameter_set: str | None = None


def cubic_constants(source: ParameterSource) -> CubicConstants:
    """Construct typed cubic constants from YAML or an explicit parameter set."""
    parameter_set = load_model_parameters(source)
    if parameter_set.model_kind != "cubic":
        raise ParameterDatabaseError(
            f"{parameter_set.identifier!r} is {parameter_set.model_kind!r}, not 'cubic'"
        )
    parameters = parameter_set.parameters
    alpha = parameters.get("alpha")
    if not isinstance(alpha, Mapping):
        raise ParameterDatabaseError(f"{parameter_set.identifier!r} requires an alpha mapping")
    alpha_kind = alpha.get("kind")
    if alpha_kind not in ("srk", "pr76", "pr78"):
        raise ParameterDatabaseError(
            f"{parameter_set.identifier!r} has unsupported alpha kind {alpha_kind!r}"
        )

    def coefficients(key: str, size: int) -> tuple[float, ...]:
        block = alpha.get(key)
        if not isinstance(block, Mapping):
            raise ParameterDatabaseError(f"{parameter_set.identifier!r} requires alpha.{key}")
        values = block.get("coefficients")
        if not isinstance(values, tuple | list) or len(values) != size:
            raise ParameterDatabaseError(
                f"{parameter_set.identifier!r} alpha.{key}.coefficients must have length {size}"
            )
        if any(not isinstance(value, int | float) for value in values):
            raise ParameterDatabaseError("cubic alpha coefficients must be numeric")
        return tuple(float(value) for value in values)

    low = coefficients("low", 3)
    high = coefficients("high", 4) if alpha_kind == "pr78" else None
    switch = alpha.get("switch_acentric_factor")
    if alpha_kind == "pr78" and not isinstance(switch, int | float):
        raise ParameterDatabaseError(
            f"{parameter_set.identifier!r} requires alpha.switch_acentric_factor"
        )
    required = ("omega_a", "omega_b", "delta1", "delta2")
    if any(not isinstance(parameters.get(key), int | float) for key in required):
        raise ParameterDatabaseError(
            f"{parameter_set.identifier!r} cubic constants must be numeric"
        )
    return CubicConstants(
        parameter_set.model,
        float(parameters["omega_a"]),
        float(parameters["omega_b"]),
        float(parameters["delta1"]),
        float(parameters["delta2"]),
        alpha_kind,
        (low[0], low[1], low[2]),
        None if high is None else (high[0], high[1], high[2], high[3]),
        None if switch is None else float(switch),
        parameter_set.identifier,
    )


# Loaded once from separate, versioned YAML documents. Full-precision
# critical-point constraints avoid amplified dense-phase pressure errors.
SRK = cubic_constants("cubic.srk-1972")
PR76 = cubic_constants("cubic.pr-1976")
PR78 = cubic_constants("cubic.pr-1978")


def _cbrt(value: Tensor) -> Tensor:
    # The exact derivative is singular at zero. An epsilon floor only regularizes
    # that multiple-root point and prevents NaNs leaking through an inactive
    # ``torch.where`` branch during higher-order autodiff.
    magnitude = torch.clamp_min(torch.abs(value), torch.finfo(value.dtype).eps)
    return torch.sign(value) * magnitude.pow(1.0 / 3.0)


def cubic_real_roots(c2: Tensor, c1: Tensor, c0: Tensor) -> Tensor:
    """Return the three sorted real-root slots of a monic cubic.

    For a one-real-root polynomial, that root is repeated in all slots. This
    representation keeps downstream phase-root selection vectorizable.
    """
    p = c1 - c2.square() / 3.0
    q = 2.0 * c2.pow(3) / 27.0 - c2 * c1 / 3.0 + c0
    discriminant = q.square() / 4.0 + p.pow(3) / 27.0
    # In the three-real-root region the Cardano branch is inactive. Giving its
    # square root a constant finite argument prevents an infinite derivative
    # at ``sqrt(0)`` from contaminating second-order thermal derivatives
    # through ``0*inf`` in autograd.
    safe_discriminant = torch.where(
        discriminant > 0.0,
        discriminant,
        torch.ones_like(discriminant),
    )
    sqrt_disc = torch.sqrt(safe_discriminant)
    one = _cbrt(-q / 2.0 + sqrt_disc) + _cbrt(-q / 2.0 - sqrt_disc) - c2 / 3.0

    # ``tiny**1.5`` underflows in the inactive three-root branch. Machine
    # epsilon is still negligible on the scale of cubic coefficients and keeps
    # all intermediate derivatives finite.
    safe_neg_p = torch.clamp_min(-p / 3.0, torch.finfo(p.dtype).eps)
    radius = 2.0 * torch.sqrt(safe_neg_p)
    angular_epsilon = 16.0 * torch.finfo(c2.dtype).eps
    cosine = torch.clamp(
        -q / (2.0 * safe_neg_p.pow(1.5)),
        -1.0 + angular_epsilon,
        1.0 - angular_epsilon,
    )
    theta = torch.acos(cosine) / 3.0
    offsets = torch.tensor(
        [0.0, -2.0 * torch.pi / 3.0, -4.0 * torch.pi / 3.0],
        dtype=c2.dtype,
        device=c2.device,
    )
    three = radius[..., None] * torch.cos(theta[..., None] + offsets) - c2[..., None] / 3.0
    repeated = one[..., None].expand_as(three)
    roots = torch.where((discriminant <= 0.0)[..., None], three, repeated)
    return torch.sort(roots, dim=-1).values


class CubicEOS(nn.Module):
    """Generalized cubic equation of state with differentiable parameters."""

    critical_temperature: Tensor
    critical_pressure: Tensor
    acentric_factor: Tensor
    molar_mass: Tensor
    volume_translation: Tensor
    volume_translation_slope: Tensor
    volume_translation_reference_temperature: Tensor
    mixing: CubicMixing

    def __init__(
        self,
        components: ComponentSet,
        constants: CubicConstants,
        *,
        mixing: CubicMixing | None = None,
        volume_translation: Tensor | VolumeTranslation | None = None,
    ) -> None:
        super().__init__()
        self.names = components.names
        self.constants = constants
        self.register_buffer("critical_temperature", components.critical_temperature.clone())
        self.register_buffer("critical_pressure", components.critical_pressure.clone())
        self.register_buffer("acentric_factor", components.acentric_factor.clone())
        self.register_buffer("molar_mass", components.molar_mass.clone())
        if mixing is None:
            zeros = torch.zeros(
                (components.ncomponents, components.ncomponents),
                dtype=components.critical_temperature.dtype,
                device=components.critical_temperature.device,
            )
            mixing = QuadraticMixing(zeros)
        self.mixing = mixing
        if isinstance(volume_translation, VolumeTranslation):
            translation = volume_translation.reference_shift.to(
                dtype=components.critical_temperature.dtype,
                device=components.critical_temperature.device,
            )
            translation_slope = volume_translation.temperature_slope.to(
                dtype=components.critical_temperature.dtype,
                device=components.critical_temperature.device,
            )
            reference_temperature = volume_translation.reference_temperature
        else:
            translation = (
                torch.zeros_like(components.critical_temperature)
                if volume_translation is None
                else volume_translation
            )
            translation_slope = torch.zeros_like(translation)
            reference_temperature = 288.15
        if translation.shape != components.critical_temperature.shape:
            raise ValueError("volume_translation must have one value per component")
        if translation_slope.shape != components.critical_temperature.shape:
            raise ValueError("volume_translation slope must have one value per component")
        self.register_buffer("volume_translation", translation.clone())
        self.register_buffer("volume_translation_slope", translation_slope.clone())
        self.register_buffer(
            "volume_translation_reference_temperature",
            components.critical_temperature.new_tensor(reference_temperature),
        )

    @property
    def ncomponents(self) -> int:
        """Number of modeled components."""
        return len(self.names)

    def component_volume_translation(self, temperature: Tensor) -> Tensor:
        """Return additive component volume shifts at ``temperature``."""
        return (
            self.volume_translation
            + (temperature[..., None] - self.volume_translation_reference_temperature)
            * self.volume_translation_slope
        )

    def _kappa(self) -> Tensor:
        omega = self.acentric_factor
        low_coefficients = self.constants.alpha_low
        low = (
            low_coefficients[0] + low_coefficients[1] * omega + low_coefficients[2] * omega.square()
        )
        if self.constants.alpha_kind != "pr78":
            return low
        if self.constants.alpha_high is None or self.constants.alpha_switch is None:
            raise ParameterDatabaseError("PR78 requires high-alpha coefficients and a switch")
        high_coefficients = self.constants.alpha_high
        high = (
            high_coefficients[0]
            + high_coefficients[1] * omega
            + high_coefficients[2] * omega.square()
            + high_coefficients[3] * omega.pow(3)
        )
        return torch.where(omega <= self.constants.alpha_switch, low, high)

    def pure_parameters(self, temperature: Tensor) -> tuple[Tensor, Tensor]:
        """Return temperature-dependent pure ``a_i`` and constant ``b_i``."""
        if bool((temperature <= 0.0).any()):
            raise InvalidStateError("temperature must be positive")
        reduced = temperature[..., None] / self.critical_temperature
        alpha = (1.0 + self._kappa() * (1.0 - torch.sqrt(reduced))).square()
        a = (
            self.constants.omega_a
            * R**2
            * self.critical_temperature.square()
            / self.critical_pressure
            * alpha
        )
        b = self.constants.omega_b * R * self.critical_temperature / self.critical_pressure
        return a, b

    def mixture_parameters(self, temperature: Tensor, composition: Tensor) -> tuple[Tensor, Tensor]:
        """Return mixed attraction and covolume parameters."""
        x = normalize_composition(composition)
        pure_a, pure_b = self.pure_parameters(temperature)
        result: tuple[Tensor, Tensor] = self.mixing(temperature, x, pure_a, pure_b)
        return result

    def dimensionless_parameters(
        self, temperature: Tensor, pressure: Tensor, composition: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return conventional cubic parameters ``A`` and ``B``."""
        if bool((pressure <= 0.0).any()):
            raise InvalidStateError("pressure must be positive")
        am, bm = self.mixture_parameters(temperature, composition)
        return am * pressure / (R * temperature).square(), bm * pressure / (R * temperature)

    def z_factors(self, temperature: Tensor, pressure: Tensor, composition: Tensor) -> Tensor:
        """Return sorted real compressibility-factor roots."""
        a, b = self.dimensionless_parameters(temperature, pressure, composition)
        u = self.constants.delta1 + self.constants.delta2
        w = self.constants.delta1 * self.constants.delta2
        c2 = (u - 1.0) * b - 1.0
        c1 = a - u * b + (w - u) * b.square()
        c0 = -(a * b + w * b.square() * (1.0 + b))
        return cubic_real_roots(c2, c1, c0)

    def _residual_gibbs_rt_from_z(self, z: Tensor, a: Tensor, b: Tensor) -> Tensor:
        d1 = self.constants.delta1
        d2 = self.constants.delta2
        safe_b = torch.clamp_min(b, torch.finfo(b.dtype).tiny)
        attraction = a / (safe_b * (d1 - d2))
        log_ratio = torch.log((z + d1 * b) / (z + d2 * b))
        value = z - 1.0 - torch.log(z - b) - attraction * log_ratio
        ideal_b_limit = z - 1.0 - torch.log(z) - a / z
        return torch.where(b.abs() > 10.0 * torch.finfo(b.dtype).eps, value, ideal_b_limit)

    def select_z(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Select a physical liquid, vapor, or minimum-Gibbs root."""
        roots = self.z_factors(temperature, pressure, composition)
        _, b = self.dimensionless_parameters(temperature, pressure, composition)
        valid = roots > b[..., None] * (1.0 + 32.0 * torch.finfo(roots.dtype).eps)
        if phase == "liquid":
            selected = torch.where(valid, roots, torch.inf).amin(dim=-1)
        elif phase == "vapor":
            selected = torch.where(valid, roots, -torch.inf).amax(dim=-1)
        elif phase == "stable":
            a, b = self.dimensionless_parameters(temperature, pressure, composition)
            gibbs = self._residual_gibbs_rt_from_z(roots, a[..., None], b[..., None])
            gibbs = torch.where(valid, gibbs, torch.inf)
            selected = torch.gather(roots, -1, gibbs.argmin(dim=-1, keepdim=True)).squeeze(-1)
        else:
            raise ValueError(f"unknown phase root {phase!r}")
        if not bool(torch.isfinite(selected).all()):
            raise InvalidStateError("cubic EoS produced no physical volume root")
        return selected

    def molar_volume(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return translated molar volume in m3/mol."""
        x = normalize_composition(composition)
        z = self.select_z(temperature, pressure, x, phase)
        unshifted = z * R * temperature / pressure
        translation = self.component_volume_translation(temperature)
        return unshifted + torch.sum(x * translation, dim=-1)

    def pressure(self, temperature: Tensor, molar_volume: Tensor, composition: Tensor) -> Tensor:
        """Evaluate pressure at a homogeneous ``T, v, x`` state."""
        x = normalize_composition(composition)
        shift = torch.sum(x * self.component_volume_translation(temperature), dim=-1)
        volume = molar_volume - shift
        am, bm = self.mixture_parameters(temperature, x)
        d1 = self.constants.delta1
        d2 = self.constants.delta2
        if bool((volume <= bm).any()):
            raise InvalidStateError("molar volume must exceed the mixture covolume")
        return R * temperature / (volume - bm) - am / ((volume + d1 * bm) * (volume + d2 * bm))

    def residual_helmholtz_rt(self, temperature: Tensor, volume: Tensor, moles: Tensor) -> Tensor:
        """Return extensive residual Helmholtz energy divided by ``RT``."""
        total = moles.sum(dim=-1)
        x = moles / total[..., None]
        translation = self.component_volume_translation(temperature)
        shift = torch.sum(x * translation, dim=-1)
        unshifted_volume = volume - total * shift
        molar_volume = unshifted_volume / total
        am, bm = self.mixture_parameters(temperature, x)
        d1 = self.constants.delta1
        d2 = self.constants.delta2
        repulsive = -total * torch.log1p(-bm / molar_volume)
        logarithm = torch.log((molar_volume + d1 * bm) / (molar_volume + d2 * bm))
        attractive = -total * am / (R * temperature * bm * (d1 - d2)) * logarithm
        # A constant Peneloux-style translation maps the physical volume V to
        # the parent-EoS volume V0 = V - sum(n_i*c_i).  The final logarithm is
        # the ideal-gas reference-volume correction required when A^R is
        # defined relative to an ideal gas at the *physical* V.  Including it
        # makes -dA/dV, fugacity, and the TP/TV routes mutually consistent.
        reference_volume_correction = total * torch.log(volume / unshifted_volume)
        return repulsive + attractive + reference_volume_correction

    def log_fugacity_coefficients(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return component log fugacity coefficients.

        The quadratic rule uses the standard closed form. Non-quadratic mixing
        rules use an autodifferentiated residual Helmholtz energy, preserving
        the exact composition dependence of the selected rule.
        """
        x = normalize_composition(composition)
        z = self.select_z(temperature, pressure, x, phase)
        a, b = self.dimensionless_parameters(temperature, pressure, x)
        if isinstance(
            self.mixing,
            QuadraticMixing | TemperatureDependentQuadraticMixing | PPR78Mixing,
        ):
            pure_a, pure_b = self.pure_parameters(temperature)
            am, bm = self.mixture_parameters(temperature, x)
            if isinstance(self.mixing, PPR78Mixing):
                aij = self.mixing.cross_a(temperature, pure_a, pure_b)
            elif isinstance(self.mixing, TemperatureDependentQuadraticMixing):
                aij = self.mixing.cross_a(temperature, pure_a)
            else:
                aij = self.mixing.cross_a(pure_a)
            sum_aij = torch.einsum("...j,...ij->...i", x, aij)
            partial_b_over_b = self.mixing.partial_b(x, pure_b) / bm[..., None]
            composition_term = 2.0 * sum_aij / am[..., None] - partial_b_over_b
            d1 = self.constants.delta1
            d2 = self.constants.delta2
            log_ratio = torch.log((z + d1 * b) / (z + d2 * b))
            unshifted_log_phi = (
                partial_b_over_b * (z - 1.0)[..., None]
                - torch.log(z - b)[..., None]
                - (a / (b * (d1 - d2)) * log_ratio)[..., None] * composition_term
            )
            # For v = v0 + sum(x_i*c_i), thermodynamic consistency requires
            # ln(phi_i) = ln(phi_i,0) + P*c_i/(R*T).  A common sign error is
            # avoided here by defining ``volume_translation`` as the quantity
            # *added* to the parent-EoS volume in ``molar_volume``.
            return unshifted_log_phi + (pressure / (R * temperature))[
                ..., None
            ] * self.component_volume_translation(temperature)

        translation = self.component_volume_translation(temperature)
        volume = z * R * temperature / pressure + torch.sum(x * translation, dim=-1)

        def at_fixed_volume(moles: Tensor) -> Tensor:
            # Batched states are independent, so differentiating their sum
            # returns the same row-wise chemical potentials as evaluating
            # each scalar state separately. ``torch.func.grad`` requires a
            # scalar output and therefore cannot consume the unsummed batch.
            return self.residual_helmholtz_rt(temperature, volume, moles).sum()

        residual_mu_rt: Tensor = torch.func.grad(at_fixed_volume)(x)
        physical_z = pressure * volume / (R * temperature)
        return residual_mu_rt - torch.log(physical_z)[..., None]


def _resolve_mixing(
    components: ComponentSet,
    kij: Tensor | None,
    kij_a: Tensor | None,
    kij_b: Tensor | None,
    lij: Tensor | None,
    trainable: bool,
    trainable_lij: bool,
    mixing: CubicMixing | None,
) -> CubicMixing:
    has_temperature_dependent_kij = kij_a is not None or kij_b is not None
    if mixing is not None:
        if (
            kij is not None
            or has_temperature_dependent_kij
            or lij is not None
            or trainable
            or trainable_lij
        ):
            raise ValueError(
                "kij/kij_a/kij_b/lij/trainable options and an explicit mixing rule "
                "are mutually exclusive"
            )
        return mixing
    if kij is not None and has_temperature_dependent_kij:
        raise ValueError(
            "constant kij and temperature-dependent kij_a/kij_b are mutually exclusive"
        )
    if (kij_a is None) != (kij_b is None):
        raise ValueError("temperature-dependent mixing requires both kij_a and kij_b")
    if kij_a is not None and kij_b is not None:
        return TemperatureDependentQuadraticMixing(
            kij_a,
            kij_b,
            lij,
            trainable=trainable,
            trainable_lij=trainable_lij,
        )
    if kij is None:
        kij = torch.zeros(
            (components.ncomponents, components.ncomponents),
            dtype=components.critical_temperature.dtype,
            device=components.critical_temperature.device,
        )
    return QuadraticMixing(
        kij,
        lij,
        trainable=trainable,
        trainable_lij=trainable_lij,
    )


def cubic_eos(
    components: ComponentSet,
    parameter_set: ParameterSource | CubicConstants,
    *,
    kij: Tensor | None = None,
    kij_a: Tensor | None = None,
    kij_b: Tensor | None = None,
    lij: Tensor | None = None,
    trainable: bool = False,
    trainable_lij: bool = False,
    mixing: CubicMixing | None = None,
    volume_translation: Tensor | VolumeTranslation | None = None,
) -> CubicEOS:
    """Construct a cubic EoS from bundled YAML or explicit typed constants.

    Supply ``kij`` for a constant quadratic interaction matrix, or both
    ``kij_a`` and ``kij_b`` for ``kij(T) = kij_a + kij_b/T``. The latter
    coefficients are dimensionless and kelvin, respectively. Supply ``lij``
    for ``bij = 0.5*(bi+bj)*(1-lij)``; omitting it recovers the conventional
    linear covolume rule. ``trainable`` controls attraction interactions;
    ``trainable_lij`` independently enables co-volume fitting from the
    supplied matrix or from zero.
    """
    constants = (
        parameter_set
        if isinstance(parameter_set, CubicConstants)
        else cubic_constants(parameter_set)
    )
    return CubicEOS(
        components,
        constants,
        mixing=_resolve_mixing(
            components,
            kij,
            kij_a,
            kij_b,
            lij,
            trainable,
            trainable_lij,
            mixing,
        ),
        volume_translation=volume_translation,
    )


def soave_redlich_kwong(
    components: ComponentSet,
    *,
    kij: Tensor | None = None,
    kij_a: Tensor | None = None,
    kij_b: Tensor | None = None,
    lij: Tensor | None = None,
    trainable: bool = False,
    trainable_lij: bool = False,
    mixing: CubicMixing | None = None,
    volume_translation: Tensor | VolumeTranslation | None = None,
) -> CubicEOS:
    """Construct the 1972 Soave-Redlich-Kwong equation of state."""
    return cubic_eos(
        components,
        SRK,
        kij=kij,
        kij_a=kij_a,
        kij_b=kij_b,
        lij=lij,
        trainable=trainable,
        trainable_lij=trainable_lij,
        mixing=mixing,
        volume_translation=volume_translation,
    )


def peng_robinson_1976(
    components: ComponentSet,
    *,
    kij: Tensor | None = None,
    kij_a: Tensor | None = None,
    kij_b: Tensor | None = None,
    lij: Tensor | None = None,
    trainable: bool = False,
    trainable_lij: bool = False,
    mixing: CubicMixing | None = None,
    volume_translation: Tensor | VolumeTranslation | None = None,
) -> CubicEOS:
    """Construct the original 1976 Peng-Robinson equation of state."""
    return cubic_eos(
        components,
        PR76,
        kij=kij,
        kij_a=kij_a,
        kij_b=kij_b,
        lij=lij,
        trainable=trainable,
        trainable_lij=trainable_lij,
        mixing=mixing,
        volume_translation=volume_translation,
    )


def peng_robinson_1978(
    components: ComponentSet,
    *,
    kij: Tensor | None = None,
    kij_a: Tensor | None = None,
    kij_b: Tensor | None = None,
    lij: Tensor | None = None,
    trainable: bool = False,
    trainable_lij: bool = False,
    mixing: CubicMixing | None = None,
    volume_translation: Tensor | VolumeTranslation | None = None,
) -> CubicEOS:
    """Construct the 1978 Peng-Robinson acentric-factor variant."""
    return cubic_eos(
        components,
        PR78,
        kij=kij,
        kij_a=kij_a,
        kij_b=kij_b,
        lij=lij,
        trainable=trainable,
        trainable_lij=trainable_lij,
        mixing=mixing,
        volume_translation=volume_translation,
    )


def predictive_peng_robinson_1978(
    components: ComponentSet,
    *,
    parameter_set: ParameterSource = DEFAULT_PPR78_GROUP_CONTRIBUTION,
    group_counts: Mapping[str, Mapping[str, float]] | Tensor | None = None,
    trainable: bool = False,
    volume_translation: Tensor | VolumeTranslation | None = None,
) -> CubicEOS:
    """Construct PPR78 with group-contribution ``kij(T)``.

    The default parameter set is the original six-group saturated-hydrocarbon
    fit of Jaubert and Mutelet (2004), doi:10.1016/j.fluid.2004.06.059.
    Custom YAML parameter sets and explicit component group counts use the
    same path. ``trainable=True`` exposes the unique universal A/B group
    interactions as PyTorch parameters.

    PPR78 was derived with the linear covolume rule, so co-volume interactions
    are intentionally not accepted by this named constructor.
    """
    return CubicEOS(
        components,
        PR78,
        mixing=ppr78_mixing(
            components,
            parameter_set,
            group_counts=group_counts,
            trainable=trainable,
        ),
        volume_translation=volume_translation,
    )
