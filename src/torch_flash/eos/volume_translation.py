"""Cubic-EoS volume-translation correlations in explicit SI conventions.

Péneloux et al. (1982) and Whitson and Brulé (2000) write the corrected
volume as ``v = v_eos - sum(x_i c_i)``.  :class:`VolumeTranslation` stores the
quantity *added* to ``v_eos`` instead, so its ``reference_shift`` is ``-c``.
Keeping this conversion at the correlation boundary avoids the common
fugacity-sign error.

Pedersen et al. (2024), Eq. 5.6, prints ``c = M/rho - v_eos`` despite defining
``v_pen = v_eos - c`` in Eq. 4.44 and using ``c = 148 - 130`` cm3/mol in
Figure 4.7.  The density-matching functions below follow the latter,
thermodynamically consistent convention: published ``c = v_eos - M/rho``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.constants import R
from torch_flash.database import ParameterSource, load_model_parameters
from torch_flash.exceptions import InvalidStateError, ParameterDatabaseError

if TYPE_CHECKING:
    from torch_flash.eos.cubic import CubicEOS

VolumeTranslationEOS = Literal["srk", "pr"]
HydrocarbonFamily = Literal["paraffin", "naphthene", "aromatic"]

_PEDERSEN_SOURCE = "volume-translation.pedersen-2024"
_WHITSON_SOURCE = "volume-translation.whitson-2000"


@dataclass(frozen=True)
class VolumeTranslation:
    """Component translations in the additive torch-flash convention.

    ``reference_shift`` is in m3/mol, ``temperature_slope`` in
    m3/(mol K), and ``reference_temperature`` in K.  A published positive
    Péneloux ``c`` therefore appears as a negative ``reference_shift``.
    """

    reference_shift: Tensor
    temperature_slope: Tensor
    reference_temperature: float = 288.15
    source: str = "custom"

    def __post_init__(self) -> None:
        if self.reference_shift.ndim != 1:
            raise ValueError("reference_shift must be a one-dimensional component vector")
        if self.temperature_slope.shape != self.reference_shift.shape:
            raise ValueError("temperature_slope must match reference_shift")
        if not bool(
            torch.isfinite(self.reference_shift).all()
            & torch.isfinite(self.temperature_slope).all()
        ):
            raise ValueError("volume-translation coefficients must be finite")
        if not torch.is_floating_point(self.reference_shift):
            raise TypeError("volume-translation coefficients must use a floating dtype")
        if (
            self.temperature_slope.dtype != self.reference_shift.dtype
            or self.temperature_slope.device != self.reference_shift.device
        ):
            raise ValueError("volume-translation tensors must share dtype and device")
        if not torch.isfinite(torch.tensor(self.reference_temperature)):
            raise ValueError("reference_temperature must be finite")
        if self.reference_temperature <= 0.0:
            raise ValueError("reference_temperature must be positive")

    @classmethod
    def constant(cls, shift: Tensor, *, source: str = "custom") -> VolumeTranslation:
        """Construct a temperature-independent additive translation."""
        return cls(shift, torch.zeros_like(shift), source=source)

    def at_temperature(self, temperature: Tensor) -> Tensor:
        """Return each component's additive shift at ``temperature``."""
        return (
            self.reference_shift
            + (temperature[..., None] - self.reference_temperature) * self.temperature_slope
        )

    def to(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> VolumeTranslation:
        """Move translation coefficients to a common dtype and device."""
        return VolumeTranslation(
            self.reference_shift.to(dtype=dtype, device=device),
            self.temperature_slope.to(dtype=dtype, device=device),
            self.reference_temperature,
            self.source,
        )


def _parameter_mapping(source: ParameterSource, expected_model: str) -> Mapping[str, object]:
    parameter_set = load_model_parameters(source)
    if parameter_set.model_kind != "volume_translation":
        raise ParameterDatabaseError(
            f"{parameter_set.identifier!r} is {parameter_set.model_kind!r}, "
            "not 'volume_translation'"
        )
    if parameter_set.model != expected_model:
        raise ParameterDatabaseError(
            f"{parameter_set.identifier!r} describes {parameter_set.model!r}, "
            f"not {expected_model!r}"
        )
    return cast(Mapping[str, object], parameter_set.parameters)


def _mapping(parent: Mapping[str, object], key: str, source: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ParameterDatabaseError(f"{source} requires a {key!r} mapping")
    return cast(Mapping[str, object], value)


def _number(parent: Mapping[str, object], key: str, source: str) -> float:
    value = parent.get(key)
    if not isinstance(value, int | float) or not math.isfinite(value):
        raise ParameterDatabaseError(f"{source} requires finite numeric {key!r}")
    return float(value)


def rackett_compressibility_factor(
    acentric_factor: Tensor,
    *,
    source: ParameterSource = _PEDERSEN_SOURCE,
) -> Tensor:
    """Return ``Z_RA = 0.29056 - 0.08775 omega`` from Pedersen Eq. 4.47."""
    parameters = _parameter_mapping(source, "Pedersen-Peneloux")
    rackett = _mapping(parameters, "rackett", str(source))
    intercept = _number(rackett, "intercept", str(source))
    slope = _number(rackett, "acentric_slope", str(source))
    return intercept + slope * acentric_factor


def pedersen_peneloux_translation(
    components: ComponentSet,
    eos: VolumeTranslationEOS,
    *,
    rackett_factor: Tensor | None = None,
    source: ParameterSource = _PEDERSEN_SOURCE,
) -> VolumeTranslation:
    """Return the Pedersen light-component SRK/PR Péneloux translation.

    SRK uses Pedersen Eq. 4.46 and PR uses the Jhaveri-Youngren Eq. 4.49.
    These correlations were fitted for nonhydrocarbons and hydrocarbons
    lighter than C7; heavier fractions should use density matching or a
    characterized parameter set.
    """
    parameters = _parameter_mapping(source, "Pedersen-Peneloux")
    correlations = _mapping(parameters, "correlations", str(source))
    correlation = _mapping(correlations, eos, str(source))
    z_ra = (
        rackett_compressibility_factor(components.acentric_factor, source=source)
        if rackett_factor is None
        else rackett_factor.to(
            dtype=components.critical_temperature.dtype,
            device=components.critical_temperature.device,
        )
    )
    if z_ra.shape != components.critical_temperature.shape:
        raise ValueError("rackett_factor must have one value per component")
    if not bool(torch.isfinite(z_ra).all()):
        raise ValueError("rackett_factor must be finite")
    scale = _number(correlation, "scale", str(source))
    target = _number(correlation, "target", str(source))
    published_c = (
        scale * R * components.critical_temperature / components.critical_pressure * (target - z_ra)
    )
    return VolumeTranslation.constant(
        -published_c,
        source=f"{load_model_parameters(source).identifier}:{eos}",
    )


def whitson_volume_translation(
    components: ComponentSet,
    eos: VolumeTranslationEOS,
    *,
    heavy_families: Mapping[str, HydrocarbonFamily] | None = None,
    source: ParameterSource = _WHITSON_SOURCE,
) -> VolumeTranslation:
    """Return Whitson Tables 4.2-4.3 translations.

    Named light components and normal paraffins through n-decane use the
    tabulated ``s_i = c_i/b_i`` values.  Any other component requires a family
    entry and uses ``s_i = 1 - A0/M_i**A1`` with ``M`` in g/mol.
    """
    parameters = _parameter_mapping(source, "Whitson-Peneloux")
    eos_parameters = _mapping(_mapping(parameters, "eos", str(source)), eos, str(source))
    factors = _mapping(eos_parameters, "pure_shift_factors", str(source))
    family_parameters = _mapping(parameters, "heavy_families", str(source))
    supplied_families = {} if heavy_families is None else dict(heavy_families)
    shift_factors: list[Tensor] = []
    for index, name in enumerate(components.names):
        tabulated = factors.get(name)
        if isinstance(tabulated, int | float):
            shift_factors.append(components.molar_mass.new_tensor(float(tabulated)))
            continue
        family = supplied_families.get(name)
        if family is None:
            raise ValueError(
                f"Whitson has no tabulated shift factor for {name!r}; provide heavy_families[name]"
            )
        family_values = _mapping(family_parameters, family, str(source))
        a0 = _number(family_values, "A0", str(source))
        a1 = _number(family_values, "A1", str(source))
        molar_mass_g = 1000.0 * components.molar_mass[index]
        shift_factors.append(1.0 - a0 / molar_mass_g.pow(a1))
    factor = torch.stack(shift_factors)
    omega_b = _number(eos_parameters, "covolume_factor", str(source))
    covolume = omega_b * R * components.critical_temperature / components.critical_pressure
    published_c = factor * covolume
    return VolumeTranslation.constant(
        -published_c,
        source=f"{load_model_parameters(source).identifier}:{eos}",
    )


def density_matched_translation(
    eos_molar_volume: Tensor,
    molar_mass: Tensor,
    mass_density: Tensor,
    *,
    source: str = "density-match",
) -> VolumeTranslation:
    """Match a reference density with an additive component translation.

    All inputs use SI units.  The returned shift is
    ``M/rho - v_eos``, while the equivalent published Péneloux coefficient is
    its negative.
    """
    if eos_molar_volume.shape != molar_mass.shape or mass_density.shape != molar_mass.shape:
        raise ValueError("volume, molar mass, and density vectors must have the same shape")
    if not bool(
        torch.isfinite(eos_molar_volume).all()
        & torch.isfinite(molar_mass).all()
        & torch.isfinite(mass_density).all()
    ):
        raise ValueError("density-matching inputs must be finite")
    if bool(
        (eos_molar_volume <= 0.0).any() | (molar_mass <= 0.0).any() | (mass_density <= 0.0).any()
    ):
        raise ValueError("density-matching inputs must be positive")
    target_volume = molar_mass / mass_density
    return VolumeTranslation.constant(target_volume - eos_molar_volume, source=source)


def pedersen_temperature_dependent_translation(
    model: CubicEOS,
    reference_density: Tensor,
    *,
    pressure: float = 101_325.0,
    source: ParameterSource = _PEDERSEN_SOURCE,
) -> VolumeTranslation:
    """Fit Pedersen's linear C7+ translation using Eqs. 5.7-5.9.

    ``reference_density`` is in kg/m3 at 288.15 K by default.  The ASTM
    correlation supplies the target density at 353.15 K, and the parent cubic
    EoS supplies the unshifted liquid volumes at both temperatures.
    """
    parameters = _parameter_mapping(source, "Pedersen-Peneloux")
    temperature_parameters = _mapping(
        parameters,
        "temperature_dependent",
        str(source),
    )
    reference_temperature = _number(
        temperature_parameters,
        "reference_temperature",
        str(source),
    )
    target_temperature = _number(
        temperature_parameters,
        "target_temperature",
        str(source),
    )
    astm_constant = _number(
        temperature_parameters,
        "astm_density_constant",
        str(source),
    )
    nonlinear_factor = _number(
        temperature_parameters,
        "nonlinear_factor",
        str(source),
    )
    if not 0.0 < reference_temperature < target_temperature:
        raise ParameterDatabaseError(
            "temperature-dependent translation requires 0 < "
            "reference_temperature < target_temperature"
        )
    if astm_constant <= 0.0 or nonlinear_factor < 0.0:
        raise ParameterDatabaseError(
            "temperature-dependent translation requires a positive ASTM "
            "constant and non-negative nonlinear factor"
        )
    density = reference_density.to(
        dtype=model.critical_temperature.dtype,
        device=model.critical_temperature.device,
    )
    if density.shape != model.critical_temperature.shape:
        raise ValueError("reference_density must have one value per component")
    if not bool(torch.isfinite(density).all() & (density > 0.0).all()):
        raise ValueError("reference_density must be finite and positive")
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("pressure must be finite and positive")
    if bool(
        (model.volume_translation != 0.0).any() | (model.volume_translation_slope != 0.0).any()
    ):
        raise ValueError(
            "Pedersen temperature-dependent matching requires an untranslated parent EoS"
        )
    if bool((model.critical_temperature <= target_temperature).any()):
        raise InvalidStateError(
            "Pedersen temperature-dependent density matching requires "
            "target_temperature below every component critical temperature"
        )

    ncomponents = model.ncomponents
    identity = torch.eye(
        ncomponents,
        dtype=model.critical_temperature.dtype,
        device=model.critical_temperature.device,
    )
    pressure_vector = model.critical_temperature.new_full((ncomponents,), pressure)

    def parent_volume(temperature_value: float) -> Tensor:
        temperature_vector = model.critical_temperature.new_full(
            (ncomponents,),
            temperature_value,
        )
        z = model.select_z(
            temperature_vector,
            pressure_vector,
            identity,
            "liquid",
        )
        return z * R * temperature_vector / pressure_vector

    reference_parent_volume = parent_volume(reference_temperature)
    target_parent_volume = parent_volume(target_temperature)
    reference_shift = model.molar_mass / density - reference_parent_volume
    delta_temperature = target_temperature - reference_temperature
    expansion = astm_constant / density.square()
    target_density = density * torch.exp(
        -expansion * delta_temperature * (1.0 + nonlinear_factor * expansion * delta_temperature)
    )
    target_shift = model.molar_mass / target_density - target_parent_volume
    slope = (target_shift - reference_shift) / delta_temperature
    return VolumeTranslation(
        reference_shift,
        slope,
        reference_temperature,
        f"{load_model_parameters(source).identifier}:temperature-dependent",
    )


__all__ = [
    "HydrocarbonFamily",
    "VolumeTranslation",
    "VolumeTranslationEOS",
    "density_matched_translation",
    "pedersen_peneloux_translation",
    "pedersen_temperature_dependent_translation",
    "rackett_compressibility_factor",
    "whitson_volume_translation",
]
