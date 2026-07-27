"""Surface- and interfacial-tension correlations for petroleum fluids."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.database import load_model_parameters
from torch_flash.exceptions import InvalidStateError
from torch_flash.types import normalize_composition

_PARAMETERS = load_model_parameters("transport.pedersen-2024").parameters
_INTERFACIAL = cast(Mapping[str, object], _PARAMETERS["interfacial_tension"])


def riedel_parameter(
    normal_boiling_temperature: Tensor,
    critical_temperature: Tensor,
    critical_pressure: Tensor,
) -> Tensor:
    """Return the dimensionless Riedel parameter.

    Parameters
    ----------
    normal_boiling_temperature, critical_temperature
        Normal boiling and critical temperatures in K.
    critical_pressure
        Critical pressure in Pa.

    Returns
    -------
    Tensor
        Dimensionless Riedel parameter.

    Notes
    -----
    Implements Pedersen et al. (2024), Eq. 10.79, from Riedel (1954). The
    numerical form uses critical pressure in atm internally.
    """
    boiling, critical, pressure = torch.broadcast_tensors(
        normal_boiling_temperature,
        critical_temperature,
        critical_pressure,
    )
    if bool(
        (
            (~torch.isfinite(boiling))
            | (~torch.isfinite(critical))
            | (~torch.isfinite(pressure))
            | (boiling <= 0.0)
            | (critical <= 0.0)
            | (boiling >= critical)
            | (pressure <= 0.0)
        ).any()
    ):
        raise InvalidStateError("Riedel inputs require 0 < Tb < Tc and positive Pc")
    reduced_boiling = boiling / critical
    result: Tensor = 0.9076 * (
        1.0 + reduced_boiling * torch.log(pressure / 101_325.0) / (1.0 - reduced_boiling)
    )
    return result


def brock_bird_surface_tension(
    temperature: Tensor,
    critical_temperature: Tensor,
    critical_pressure: Tensor,
    normal_boiling_temperature: Tensor,
) -> Tensor:
    """Evaluate Brock-Bird pure nonpolar-fluid surface tension.

    Parameters
    ----------
    temperature, critical_temperature, normal_boiling_temperature
        Temperatures in K.
    critical_pressure
        Critical pressure in Pa.

    Returns
    -------
    Tensor
        Surface tension in N/m.

    Raises
    ------
    InvalidStateError
        If the state is not below the critical temperature or the defining
        properties are nonphysical.

    Notes
    -----
    Implements Brock and Bird, "Surface Tension and the Principle of
    Corresponding States," *AIChE Journal* 1 (1955), 174-177, through
    Pedersen et al. (2024), Eqs. 10.77-10.79. The published numerical
    equation yields dyn/cm from K and atm; the result is converted to N/m.
    """
    temperature, critical, pressure, boiling = torch.broadcast_tensors(
        temperature,
        critical_temperature,
        critical_pressure,
        normal_boiling_temperature,
    )
    if bool(
        ((~torch.isfinite(temperature)) | (temperature <= 0.0) | (temperature >= critical)).any()
    ):
        raise InvalidStateError("Brock-Bird requires a positive subcritical temperature")
    alpha = riedel_parameter(boiling, critical, pressure)
    amplitude = (pressure / 101_325.0).pow(2.0 / 3.0) * critical.pow(1.0 / 3.0)
    amplitude = amplitude * (0.133 * alpha - 0.281)
    tension_dyn_cm = amplitude * (1.0 - temperature / critical).pow(11.0 / 9.0)
    if bool((~torch.isfinite(tension_dyn_cm) | (tension_dyn_cm < 0.0)).any()):
        raise InvalidStateError("Brock-Bird produced a negative surface tension")
    result: Tensor = 1.0e-3 * tension_dyn_cm
    return result


def parachor_from_molar_mass(molar_mass: Tensor) -> Tensor:
    """Estimate a C7+ petroleum-fraction parachor from molar mass.

    Parameters
    ----------
    molar_mass
        Fraction molar mass in kg/mol.

    Returns
    -------
    Tensor
        Parachor in the Weinaug-Katz ``(dyn/cm)**0.25 cm3/mol`` convention.

    Notes
    -----
    Implements Pedersen et al. (2024), Eq. 10.85. This is a petroleum
    pseudo-component correlation, not a replacement for published
    pure-component parachors.
    """
    if bool(((~torch.isfinite(molar_mass)) | (molar_mass <= 0.0)).any()):
        raise InvalidStateError("molar mass must be finite and positive")
    return 59.3 + 2.34 * (1000.0 * molar_mass)


def published_parachors(
    names: Sequence[str],
    *,
    like: Tensor,
) -> Tensor:
    """Return the Pedersen Table 10.19 parachors for supported components.

    Raises
    ------
    KeyError
        If any component is outside the published table. Callers may instead
        provide explicit parachors or use :func:`parachor_from_molar_mass` for
        characterized C7+ fractions.
    """
    values = cast(Mapping[str, float], _INTERFACIAL["parachor"])
    missing = [name for name in names if name not in values]
    if missing:
        raise KeyError(f"no published parachor for: {', '.join(missing)}")
    return like.new_tensor([values[name] for name in names])


def published_lee_chien_b(
    names: Sequence[str],
    *,
    like: Tensor,
) -> Tensor:
    """Return Lee-Chien ``B`` values reproduced in Pedersen Table 10.20.

    Raises
    ------
    KeyError
        If a component lies outside the eight-component published table.
        Explicit user-supplied values are required for additional compounds.
    """
    values = cast(Mapping[str, float], _INTERFACIAL["lee_chien_b"])
    missing = [name for name in names if name not in values]
    if missing:
        raise KeyError(f"no published Lee-Chien B value for: {', '.join(missing)}")
    return like.new_tensor([values[name] for name in names])


def weinaug_katz_interfacial_tension(
    liquid_molar_density: Tensor,
    vapor_molar_density: Tensor,
    liquid_composition: Tensor,
    vapor_composition: Tensor,
    parachor: Tensor,
    *,
    liquid_mass_density: Tensor | None = None,
    vapor_mass_density: Tensor | None = None,
    danesh_exponent: bool = False,
) -> Tensor:
    """Evaluate Weinaug-Katz gas-oil interfacial tension.

    Parameters
    ----------
    liquid_molar_density, vapor_molar_density
        Coexisting phase molar densities in mol/m3.
    liquid_composition, vapor_composition
        Phase mole fractions with components on the final axis.
    parachor
        Component parachors in the conventional
        ``(dyn/cm)**0.25 cm3/mol`` units.
    liquid_mass_density, vapor_mass_density
        Optional phase mass densities in kg/m3 required by the Danesh
        gas-condensate exponent.
    danesh_exponent
        Use Pedersen Eq. 10.98 instead of the fixed one-quarter exponent.

    Returns
    -------
    Tensor
        Interfacial tension in N/m.

    Raises
    ------
    ValueError
        If component dimensions disagree or Danesh densities are absent.
    InvalidStateError
        If phase densities are nonphysical.

    Notes
    -----
    Implements the Macleod-Sugden/Weinaug-Katz relation as reproduced in
    Pedersen et al. (2024), Eqs. 10.83-10.85. The optional Danesh et al.
    modification follows Eq. 10.98. This operation requires an already solved
    pair of coexisting phases and does not identify or flash phases.
    """
    if liquid_composition.shape != vapor_composition.shape:
        raise ValueError("liquid and vapor compositions must have the same shape")
    if liquid_composition.ndim < 1 or parachor.shape != liquid_composition.shape[-1:]:
        raise ValueError("parachor must have one value per composition component")
    liquid = normalize_composition(liquid_composition)
    vapor = normalize_composition(vapor_composition)
    rho_l, rho_v = torch.broadcast_tensors(liquid_molar_density, vapor_molar_density)
    if bool(
        (
            (~torch.isfinite(rho_l))
            | (~torch.isfinite(rho_v))
            | (rho_l <= 0.0)
            | (rho_v < 0.0)
            | (rho_l <= rho_v)
        ).any()
    ):
        raise InvalidStateError("interfacial tension requires rho_liquid > rho_vapor >= 0")
    density_liquid_mol_cm3 = rho_l / 1.0e6
    density_vapor_mol_cm3 = rho_v / 1.0e6
    contrast = torch.sum(
        parachor
        * (density_liquid_mol_cm3[..., None] * liquid - density_vapor_mol_cm3[..., None] * vapor),
        dim=-1,
    ).abs()
    exponent = contrast.new_tensor(0.25)
    if danesh_exponent:
        if liquid_mass_density is None or vapor_mass_density is None:
            raise ValueError("Danesh exponent requires liquid and vapor mass densities")
        mass_liquid, mass_vapor = torch.broadcast_tensors(
            liquid_mass_density,
            vapor_mass_density,
        )
        if bool(
            (
                (~torch.isfinite(mass_liquid))
                | (~torch.isfinite(mass_vapor))
                | (mass_liquid <= mass_vapor)
                | (mass_vapor < 0.0)
            ).any()
        ):
            raise InvalidStateError("Danesh exponent requires rho_liquid > rho_vapor >= 0")
        density_difference_g_cm3 = (mass_liquid - mass_vapor) / 1000.0
        exponent = 1.0 / (3.583 + 0.16 * density_difference_g_cm3)
    tension_dyn_cm = contrast.pow(1.0 / exponent)
    return 1.0e-3 * tension_dyn_cm


def lee_chien_interfacial_tension(
    liquid_molar_density: Tensor,
    vapor_molar_density: Tensor,
    liquid_composition: Tensor,
    vapor_composition: Tensor,
    components: ComponentSet,
    riedel_parameters: Tensor,
    b_parameters: Tensor,
) -> Tensor:
    """Evaluate the Lee-Chien phase-parachor interfacial tension.

    Parameters
    ----------
    liquid_molar_density, vapor_molar_density
        Coexisting molar densities in mol/m3.
    liquid_composition, vapor_composition
        Phase mole fractions with components on the final axis.
    components
        Critical temperatures, pressures, and molar volumes.
    riedel_parameters
        Dimensionless per-component Riedel parameters.
    b_parameters
        Per-component Lee-Chien ``B`` parameters.

    Returns
    -------
    Tensor
        Interfacial tension in N/m.

    Notes
    -----
    Implements Lee and Chien, SPE-12643-MS (1984), through Pedersen et al.
    (2024), Eqs. 10.86-10.97. The bundled ``B`` table is deliberately limited
    to the eight components reproduced in Pedersen Table 10.20; additional
    components require explicit values. C7+ whole-fraction construction is a
    characterization step and is not inferred silently.
    """
    if liquid_composition.shape != vapor_composition.shape:
        raise ValueError("liquid and vapor compositions must have the same shape")
    if liquid_composition.ndim < 1 or liquid_composition.shape[-1] != components.ncomponents:
        raise ValueError("composition and component set sizes must match")
    expected = components.critical_temperature.shape
    if riedel_parameters.shape != expected or b_parameters.shape != expected:
        raise ValueError("Riedel and B parameters must have one value per component")
    if components.critical_volume is None:
        raise ValueError("Lee-Chien requires component critical molar volumes")
    critical_volume = components.critical_volume
    if bool(
        (
            (~torch.isfinite(riedel_parameters))
            | (~torch.isfinite(b_parameters))
            | (b_parameters <= 0.0)
        ).any()
    ):
        raise ValueError("Lee-Chien parameters must be finite and B positive")
    liquid = normalize_composition(liquid_composition)
    vapor = normalize_composition(vapor_composition)

    def phase_parachor(composition: Tensor) -> Tensor:
        pressure_atm = torch.sum(
            composition * (components.critical_pressure / 101_325.0),
            dim=-1,
        )
        critical_temperature = torch.sum(
            composition * components.critical_temperature,
            dim=-1,
        )
        alpha = torch.sum(composition * riedel_parameters, dim=-1)
        critical_volume_cm3 = torch.sum(
            composition * critical_volume * 1.0e6,
            dim=-1,
        )
        b = torch.sum(composition * b_parameters, dim=-1)
        amplitude = (
            pressure_atm.pow(2.0 / 3.0)
            * critical_temperature.pow(1.0 / 3.0)
            * (0.133 * alpha - 0.281)
        )
        return amplitude.pow(0.25) * critical_volume_cm3 / b

    liquid_parachor = phase_parachor(liquid)
    vapor_parachor = phase_parachor(vapor)
    rho_l, rho_v = torch.broadcast_tensors(liquid_molar_density, vapor_molar_density)
    if bool(
        (
            (~torch.isfinite(rho_l)) | (~torch.isfinite(rho_v)) | (rho_l <= rho_v) | (rho_v < 0.0)
        ).any()
    ):
        raise InvalidStateError("Lee-Chien requires rho_liquid > rho_vapor >= 0")
    contrast = (rho_l / 1.0e6 * liquid_parachor - rho_v / 1.0e6 * vapor_parachor).abs()
    result: Tensor = 1.0e-3 * contrast.pow(4)
    if bool((~torch.isfinite(result) | (result < 0.0)).any()):
        raise InvalidStateError("Lee-Chien produced a non-finite interfacial tension")
    return result
