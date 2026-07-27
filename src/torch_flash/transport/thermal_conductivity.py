"""Methane-reference corresponding-states thermal-conductivity models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol, cast

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.constants import R
from torch_flash.database import load_model_parameters
from torch_flash.exceptions import InvalidStateError
from torch_flash.types import normalize_composition

from .viscosity import (
    _METHANE_CRITICAL_MASS_DENSITY,
    _METHANE_MOLAR_MASS_G,
    _METHANE_PC,
    _METHANE_TC,
    _methane_bwr_density_derivative,
    _mixture_pseudocritical,
    corresponding_states_viscosity,
    methane_bwr_density,
    methane_bwr_pressure,
    methane_viscosity,
)

_PARAMETERS = load_model_parameters("transport.pedersen-2024").parameters
_METHANE = cast(Mapping[str, object], _PARAMETERS["methane"])
_CRITICAL_DENSITY_G_CM3 = cast(float, _METHANE["critical_mass_density_g_cm3"])
_AVOGADRO_CONSTANT = 6.02214076e23
_BOLTZMANN_CONSTANT = 1.380649e-23


class HeatCapacityModel(Protocol):
    """Component ideal-gas heat-capacity evaluator used by transport models."""

    def heat_capacity(self, temperature: Tensor) -> Tensor:
        """Return component heat capacities in J/(mol K)."""


def _values(name: str, reference: Tensor) -> Tensor:
    return torch.as_tensor(_METHANE[name], dtype=reference.dtype, device=reference.device)


def _methane_conductivity_background(
    temperature: Tensor,
    density_mol_l: Tensor,
) -> Tensor:
    mass_density = density_mol_l * _METHANE_MOLAR_MASS_G / 1000.0
    dilute_coefficients = _values("conductivity_dilute", temperature)
    powers = torch.arange(-3, 6, dtype=temperature.dtype, device=temperature.device) / 3.0
    dilute = torch.sum(dilute_coefficients * temperature[..., None].pow(powers), dim=-1)
    first = _values("conductivity_first_density", temperature)
    first_density = first[0] + first[1] * (first[2] - torch.log(temperature / first[3])).square()
    dense_coefficients = _values("conductivity_dense", temperature)
    theta = (mass_density - _CRITICAL_DENSITY_G_CM3) / _CRITICAL_DENSITY_G_CM3
    dense = torch.exp(dense_coefficients[0] + dense_coefficients[3] / temperature) * (
        torch.exp(
            mass_density.pow(0.1)
            * (dense_coefficients[1] + dense_coefficients[2] / temperature.pow(1.5))
            + theta
            * mass_density.sqrt()
            * (
                dense_coefficients[4]
                + dense_coefficients[5] / temperature
                + dense_coefficients[6] / temperature.square()
            )
        )
        - 1.0
    )
    low_coefficients = _values("conductivity_low_temperature", temperature)
    low_dense = torch.exp(low_coefficients[0] + low_coefficients[3] / temperature) * (
        torch.exp(
            mass_density.pow(0.1)
            * (low_coefficients[1] + low_coefficients[2] / temperature.pow(1.5))
            + mass_density.sqrt()
            * (
                low_coefficients[4]
                + low_coefficients[5] / temperature
                + low_coefficients[6] / temperature.square()
            )
        )
        - 1.0
    )
    transition = 0.5 * (torch.tanh(temperature - 91.0) + 1.0)
    dense = transition * dense + (1.0 - transition) * low_dense
    return 1.0e-3 * (dilute + first_density * mass_density + dense)


def _pressure_temperature_derivative(
    temperature: Tensor,
    density_mol_l: Tensor,
) -> Tensor:
    with torch.enable_grad():
        if temperature.requires_grad:
            differentiable_temperature = temperature
        else:
            differentiable_temperature = temperature.detach().requires_grad_(True)
        pressure_atm = methane_bwr_pressure(differentiable_temperature, density_mol_l)
        derivative = torch.autograd.grad(
            pressure_atm.sum(),
            differentiable_temperature,
            create_graph=temperature.requires_grad,
        )[0]
    return 101_325.0 * derivative


def _scaled_isothermal_compressibility(
    temperature: Tensor,
    density_mol_l: Tensor,
    direct_compressibility: Tensor,
) -> Tensor:
    mass_density = density_mol_l * _METHANE_MOLAR_MASS_G / 1000.0
    reduced_temperature_distance = torch.abs(temperature - _METHANE_TC) / _METHANE_TC
    reduced_density_distance = (
        torch.abs(mass_density - _CRITICAL_DENSITY_G_CM3) / _CRITICAL_DENSITY_G_CM3
    )
    block = cast(Mapping[str, float], _METHANE["critical_enhancement"])
    beta = block["beta"]
    delta = block["delta"]
    x0 = block["x0"]
    e1 = block["E1"]
    e2 = block["E2"]
    gamma_prime = block["gamma_prime"]
    safe_density_distance = torch.clamp_min(
        reduced_density_distance,
        torch.finfo(temperature.dtype).eps,
    )
    x = reduced_temperature_distance / safe_density_distance.pow(1.0 / beta)
    u = (x + x0) / x0
    q = 1.0 + e2 * u.pow(2.0 * beta)
    exponent = -(gamma_prime - 1.0) / (2.0 * beta)
    h = e1 * u.pow(2.0 * beta) * q.pow(exponent)
    derivative = e1 * (2.0 * beta / x0) * u.pow(2.0 * beta - 1.0) * q.pow(exponent) + e1 * u.pow(
        2.0 * beta
    ) * exponent * q.pow(exponent - 1.0) * e2 * (2.0 * beta / x0) * u.pow(2.0 * beta - 1.0)
    inverse_reduced = safe_density_distance.pow(delta - 1.0) * (delta * h - x * derivative / beta)
    density_ratio = mass_density / _CRITICAL_DENSITY_G_CM3
    scaled = 1.0 / (
        _METHANE_PC
        * density_ratio.square()
        * torch.clamp_min(
            inverse_reduced,
            torch.finfo(temperature.dtype).tiny,
        )
    )
    use_scaling = (reduced_temperature_distance <= 0.025) & (reduced_density_distance <= 0.25)
    return torch.where(use_scaling, scaled, direct_compressibility)


def methane_critical_thermal_conductivity_enhancement(
    temperature: Tensor,
    density_mol_l: Tensor,
) -> Tensor:
    """Return the methane critical thermal-conductivity enhancement.

    Parameters
    ----------
    temperature
        Methane temperature in K.
    density_mol_l
        Methane molar density in mol/L.

    Returns
    -------
    Tensor
        Critical enhancement in W/(m K).

    Raises
    ------
    InvalidStateError
        If the supplied state is nonpositive or has nonpositive isothermal
        compressibility.

    Notes
    -----
    Implements Hanley, McCarty, and Haynes, *Cryogenics* 15 (1975),
    Appendix C, Eqs. 10-17, doi:10.1016/0011-2275(75)90010-7. The
    Vicentini-Missoni scaling form replaces the BWR compressibility only
    inside ``|T-Tc|/Tc <= 0.025`` and ``|rho-rhoc|/rhoc <= 0.25``. The
    correlation is singular at the mathematical critical point.
    """
    temperature, density = torch.broadcast_tensors(temperature, density_mol_l)
    if bool(
        (
            (~torch.isfinite(temperature))
            | (~torch.isfinite(density))
            | (temperature <= 0.0)
            | (density <= 0.0)
        ).any()
    ):
        raise InvalidStateError("methane critical enhancement requires positive finite state data")
    density_mol_m3 = 1000.0 * density
    derivative = _methane_bwr_density_derivative(temperature, density)
    pressure_density_derivative = 101_325.0 * derivative / 1000.0
    direct_compressibility = 1.0 / (density_mol_m3 * pressure_density_derivative)
    compressibility = _scaled_isothermal_compressibility(
        temperature,
        density,
        direct_compressibility,
    )
    if bool((~torch.isfinite(compressibility) | (compressibility <= 0.0)).any()):
        raise InvalidStateError("methane BWR state has non-positive isothermal compressibility")
    mass_density_g_cm3 = density * _METHANE_MOLAR_MASS_G / 1000.0
    mass_density_kg_m3 = 1000.0 * mass_density_g_cm3
    viscosity = methane_viscosity(temperature, density)
    pressure_temperature = _pressure_temperature_derivative(temperature, density)
    correlation_length = 1.465 * torch.sqrt(mass_density_g_cm3 / temperature) * 1.0e-8
    prefactor = torch.sqrt(
        (_METHANE_MOLAR_MASS_G / 1000.0)
        / (mass_density_kg_m3 * _AVOGADRO_CONSTANT * _BOLTZMANN_CONSTANT * temperature)
    )
    reduced_temperature_distance = torch.abs(temperature - _METHANE_TC) / _METHANE_TC
    reduced_density_distance = (
        torch.abs(mass_density_g_cm3 - _CRITICAL_DENSITY_G_CM3) / _CRITICAL_DENSITY_G_CM3
    )
    damping = torch.exp(
        -18.66 * reduced_temperature_distance.square() - 4.25 * reduced_density_distance.pow(4)
    )
    return (
        prefactor
        * _BOLTZMANN_CONSTANT
        * temperature.square()
        / (6.0 * torch.pi * viscosity * correlation_length)
        * pressure_temperature.square()
        * torch.sqrt(compressibility)
        * damping
    )


def methane_thermal_conductivity(
    temperature: Tensor,
    density_mol_l: Tensor,
    *,
    include_critical_enhancement: bool = True,
) -> Tensor:
    """Evaluate the Hanley methane reference thermal conductivity.

    Parameters
    ----------
    temperature
        Methane temperature in K.
    density_mol_l
        Methane molar density in mol/L.
    include_critical_enhancement
        Include the pure-fluid critical anomaly from Hanley Appendix C.

    Returns
    -------
    Tensor
        Thermal conductivity in W/(m K).

    Notes
    -----
    The background implements Pedersen et al. (2024), Eqs. 10.69-10.74,
    doi:10.1201/9780429457418-10. ``GT(9)=5.311764e-2`` follows the primary
    Hanley table, doi:10.1016/0011-2275(75)90010-7. The critical term is a
    pure-methane correction and is excluded from the mixture model as directed
    by Pedersen section 10.2.
    """
    temperature, density = torch.broadcast_tensors(temperature, density_mol_l)
    if bool(
        (
            (~torch.isfinite(temperature))
            | (~torch.isfinite(density))
            | (temperature <= 0.0)
            | (density < 0.0)
        ).any()
    ):
        raise InvalidStateError("methane conductivity requires positive T and non-negative density")
    background = _methane_conductivity_background(temperature, density)
    if not include_critical_enhancement or bool((density == 0.0).all()):
        return background
    enhancement = torch.where(
        density > 0.0,
        methane_critical_thermal_conductivity_enhancement(
            temperature,
            torch.clamp_min(density, torch.finfo(density.dtype).tiny),
        ),
        torch.zeros_like(density),
    )
    return background + enhancement


def _effective_mixture_molar_mass(
    composition: Tensor,
    components: ComponentSet,
    critical_temperature: Tensor,
    critical_pressure: Tensor,
) -> Tensor:
    z = composition
    molecular_weight = 1000.0 * components.molar_mass
    component_pressure_atm = components.critical_pressure / 101_325.0
    mixture_pressure_atm = critical_pressure / 101_325.0
    weights = z[..., :, None] * z[..., None, :]
    numerator = torch.sqrt(1.0 / molecular_weight[:, None] + 1.0 / molecular_weight[None, :]) * (
        components.critical_temperature[:, None] * components.critical_temperature[None, :]
    ).pow(0.25)
    denominator = (
        (components.critical_temperature / component_pressure_atm).pow(1.0 / 3.0)[:, None]
        + (components.critical_temperature / component_pressure_atm).pow(1.0 / 3.0)[None, :]
    ).square()
    collision_sum = torch.sum(weights * numerator / denominator, dim=(-2, -1))
    return (
        1.0
        / 16.0
        * collision_sum.pow(-2.0)
        * critical_temperature.pow(-1.0 / 3.0)
        * mixture_pressure_atm.pow(4.0 / 3.0)
    )


def _internal_energy_conductivity(
    dilute_viscosity: Tensor,
    heat_capacity: Tensor,
    effective_molar_mass_g: Tensor,
    reduced_density: Tensor,
) -> Tensor:
    density_function = (
        1.0
        + 0.053432 * reduced_density
        - 0.030182 * reduced_density.square()
        - 0.029725 * reduced_density.pow(3)
    )
    viscosity_cp = 1000.0 * dilute_viscosity
    return (
        1.0e-3
        * 1.18653
        * viscosity_cp
        * (heat_capacity - 2.5 * R)
        * density_function
        / effective_molar_mass_g
    )


def corresponding_states_thermal_conductivity(
    temperature: Tensor,
    pressure: Tensor,
    composition: Tensor,
    components: ComponentSet,
    ideal_gas: HeatCapacityModel,
    methane_ideal_gas: HeatCapacityModel,
    *,
    phase: Literal["liquid", "vapor"] = "vapor",
) -> Tensor:
    """Evaluate the Christensen-Fredenslund mixture thermal conductivity.

    Parameters
    ----------
    temperature, pressure
        Positive state variables in K and Pa.
    composition
        Mole fractions with components on the final axis.
    components
        Critical properties and molar masses matching ``composition``.
    ideal_gas
        Component ideal-gas heat-capacity model for the mixture.
    methane_ideal_gas
        One-component methane ideal-gas heat-capacity model.
    phase
        Methane BWR reference-density branch.

    Returns
    -------
    Tensor
        Mixture thermal conductivity in W/(m K).

    Raises
    ------
    ValueError
        If composition or heat-capacity component dimensions disagree.
    InvalidStateError
        If the model produces a nonpositive result.

    Notes
    -----
    Implements Christensen and Fredenslund, "A corresponding states model for
    the thermal conductivity of gases and liquids," *Chemical Engineering
    Science* 35 (1980), 871-875, and Pedersen et al. (2024),
    Eqs. 10.59-10.74. As specified for mixtures, the pure-fluid critical
    enhancement is omitted. The model requires homogeneous phase identity;
    it does not perform a flash.
    """
    if composition.ndim < 1 or composition.shape[-1] != components.ncomponents:
        raise ValueError("composition and component set sizes must match")
    z = normalize_composition(composition)
    mixture_tc, mixture_pc = _mixture_pseudocritical(components, z)
    mapped_temperature = temperature * _METHANE_TC / mixture_tc
    mapped_pressure = pressure * _METHANE_PC / mixture_pc
    initial_density = methane_bwr_density(mapped_temperature, mapped_pressure, phase=phase)
    reduced_density = (
        initial_density * _METHANE_MOLAR_MASS_G / 1000.0 / _METHANE_CRITICAL_MASS_DENSITY
    )
    molecular_weight_g = 1000.0 * components.molar_mass
    alpha_components = 1.0 + 0.0006004 * reduced_density[..., None].pow(
        2.043
    ) * molecular_weight_g.pow(1.086)
    weights = z[..., :, None] * z[..., None, :]
    alpha_mixture = torch.sum(
        weights * torch.sqrt(alpha_components[..., :, None] * alpha_components[..., None, :]),
        dim=(-2, -1),
    )
    alpha_methane = 1.0 + 0.0006004 * reduced_density.pow(2.043) * temperature.new_tensor(
        _METHANE_MOLAR_MASS_G
    ).pow(1.086)
    reference_temperature = temperature * _METHANE_TC * alpha_methane / (mixture_tc * alpha_mixture)
    reference_pressure = pressure * _METHANE_PC * alpha_methane / (mixture_pc * alpha_mixture)
    reference_density = methane_bwr_density(
        reference_temperature,
        reference_pressure,
        phase=phase,
    )
    reference_conductivity = methane_thermal_conductivity(
        reference_temperature,
        reference_density,
        include_critical_enhancement=False,
    )
    effective_mass = _effective_mixture_molar_mass(z, components, mixture_tc, mixture_pc)
    mixture_component_heat_capacity = ideal_gas.heat_capacity(temperature)
    if (
        mixture_component_heat_capacity.ndim < 1
        or mixture_component_heat_capacity.shape[-1] != components.ncomponents
    ):
        raise ValueError("mixture heat-capacity model must return one value per component")
    methane_component_heat_capacity = methane_ideal_gas.heat_capacity(reference_temperature)
    if methane_component_heat_capacity.ndim < 1 or methane_component_heat_capacity.shape[-1] != 1:
        raise ValueError("methane heat-capacity model must return one component value")
    mixture_heat_capacity = torch.sum(z * mixture_component_heat_capacity, dim=-1)
    methane_heat_capacity = methane_component_heat_capacity.squeeze(-1)
    mixture_dilute_viscosity = corresponding_states_viscosity(
        temperature,
        temperature.new_tensor(101_325.0),
        z,
        components,
        phase="vapor",
    )
    methane_dilute_viscosity = methane_viscosity(
        reference_temperature,
        methane_bwr_density(
            reference_temperature,
            reference_temperature.new_tensor(101_325.0),
            phase="vapor",
        ),
    )
    mixture_internal = _internal_energy_conductivity(
        mixture_dilute_viscosity,
        mixture_heat_capacity,
        effective_mass,
        reduced_density,
    )
    reference_internal = _internal_energy_conductivity(
        methane_dilute_viscosity,
        methane_heat_capacity,
        reference_temperature.new_tensor(_METHANE_MOLAR_MASS_G / 2.0),
        reduced_density,
    )
    scale = (
        (mixture_pc / _METHANE_PC).pow(2.0 / 3.0)
        / (
            (mixture_tc / _METHANE_TC).pow(1.0 / 6.0)
            * (effective_mass / (_METHANE_MOLAR_MASS_G / 2.0)).sqrt()
        )
        * alpha_mixture
        / alpha_methane
    )
    conductivity = scale * (reference_conductivity - reference_internal) + mixture_internal
    if bool((~torch.isfinite(conductivity) | (conductivity <= 0.0)).any()):
        raise InvalidStateError("corresponding-states model produced non-positive conductivity")
    return conductivity
