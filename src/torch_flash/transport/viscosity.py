"""Pedersen corresponding-states and Lohrenz-Bray-Clark viscosity models.

Equations and constants follow Pedersen, *Phase Behavior of Petroleum
Reservoir Fluids*, 3rd ed. (2024), section 10.1.1,
doi:10.1201/9780429457418. The corresponding-states mixture model originates
with Pedersen et al., *Chemical Engineering Science* 39 (1984), 1011-1016,
doi:10.1016/0009-2509(84)87009-8.

The LBC implementation follows Pedersen (2024), section 10.1.3,
Eqs. 10.38-10.45, and is checked against Whitson and Brulé, *Phase
Behavior*, SPE Monograph 20 (2000), Appendix B, Problem 7.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, cast

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.constants import R
from torch_flash.database import load_model_parameters
from torch_flash.eos.cubic import CubicEOS
from torch_flash.exceptions import InvalidStateError
from torch_flash.types import PhaseKind, normalize_composition

_METHANE_TC = 190.564
_METHANE_PC = 4.5992e6
_METHANE_MOLAR_MASS_G = 16.04246
_METHANE_CRITICAL_MASS_DENSITY = 0.16266  # kg/L = g/cm3
_R_L_ATM = 0.08205616
_GAMMA = 0.0096
_METHANE_BWR_DENSITY_SCAN_LOG10_MAX = math.log10(50.0)

_N = (
    -1.8439486666e-2,
    1.0510162064,
    -1.6057820303e1,
    8.4844027563e2,
    -4.2738409106e4,
    7.6565285254e-4,
    -4.8360724197e-1,
    8.5195473835e1,
    -1.6607434721e4,
    -3.7521074532e-5,
    2.8616309259e-2,
    -2.8685298973,
    1.1906973942e-4,
    -8.5315715698e-3,
    3.8365063841,
    2.4986828379e-5,
    5.7974531455e-6,
    -7.1648329297e-3,
    1.2577853784e-4,
    2.2240102466e4,
    -1.4800512328e6,
    5.0498054887e1,
    1.6428375992e6,
    2.1325387196e-1,
    3.7791273422e1,
    -1.1857016815e-5,
    -3.1630780767e1,
    -4.1006782941e-6,
    1.4870043284e-3,
    3.1512261532e-9,
    -2.1670774745e-6,
    2.4000551079e-5,
)

_GV = (
    -2.090975e5,
    2.647269e5,
    -1.472818e5,
    4.716740e4,
    -9.491872e3,
    1.219979e3,
    -9.627993e1,
    4.274152,
    -8.141531e-2,
)
_J = (-10.3506, 17.5716, -3019.39, 188.730, 0.0429036, 145.290, 6127.68)
_LBC = (0.10230, 0.023364, 0.058533, -0.040758, 0.0093324)
_FT3_PER_LBMOL_TO_M3_PER_MOL = 0.028316846592 / 453.59237
_LB_FT3_TO_KG_M3 = 16.01846337396014
_R_BAR_CM3 = 83.1446261815324
_TRANSPORT_PARAMETERS = load_model_parameters("transport.pedersen-2024").parameters


def _bwr_coefficients(temperature: Tensor) -> Tensor:
    n = torch.tensor(_N, dtype=temperature.dtype, device=temperature.device)
    t = temperature
    values = (
        _R_L_ATM * t,
        n[0] * t + n[1] * torch.sqrt(t) + n[2] + n[3] / t + n[4] / t.square(),
        n[5] * t + n[6] + n[7] / t + n[8] / t.square(),
        n[9] * t + n[10] + n[11] / t,
        n[12].expand_as(t),
        n[13] / t + n[14] / t.square(),
        n[15] / t,
        n[16] / t + n[17] / t.square(),
        n[18] / t.square(),
        n[19] / t.square() + n[20] / t.pow(3),
        n[21] / t.square() + n[22] / t.pow(4),
        n[23] / t.square() + n[24] / t.pow(3),
        n[25] / t.square() + n[26] / t.pow(4),
        n[27] / t.square() + n[28] / t.pow(3),
        n[29] / t.square() + n[30] / t.pow(3) + n[31] / t.pow(4),
    )
    return torch.stack(values, dim=-1)


def methane_bwr_pressure(temperature: Tensor, density_mol_l: Tensor) -> Tensor:
    """Evaluate the McCarty BWR methane pressure correlation.

    Parameters
    ----------
    temperature
        Scalar temperature in K.
    density_mol_l
        Methane molar density in mol/L.

    Returns
    -------
    Tensor
        Pressure in atm.
    """
    coefficients = _bwr_coefficients(temperature)
    low_powers = torch.arange(1, 10, dtype=temperature.dtype, device=temperature.device)
    high_powers = torch.arange(3, 14, 2, dtype=temperature.dtype, device=temperature.device)
    density = density_mol_l[..., None]
    polynomial = torch.sum(coefficients[..., :9] * density.pow(low_powers), dim=-1)
    exponential = torch.sum(
        coefficients[..., 9:] * density.pow(high_powers) * torch.exp(-_GAMMA * density.square()),
        dim=-1,
    )
    return polynomial + exponential


def _methane_bwr_density_derivative(temperature: Tensor, density_mol_l: Tensor) -> Tensor:
    coefficients = _bwr_coefficients(temperature)
    low_powers = torch.arange(1, 10, dtype=temperature.dtype, device=temperature.device)
    high_powers = torch.arange(3, 14, 2, dtype=temperature.dtype, device=temperature.device)
    density = density_mol_l[..., None]
    polynomial = torch.sum(
        coefficients[..., :9] * low_powers * density.pow(low_powers - 1.0),
        dim=-1,
    )
    exponential_factor = torch.exp(-_GAMMA * density.square())
    exponential = torch.sum(
        coefficients[..., 9:]
        * exponential_factor
        * (
            high_powers * density.pow(high_powers - 1.0)
            - 2.0 * _GAMMA * density.pow(high_powers + 1.0)
        ),
        dim=-1,
    )
    return polynomial + exponential


def methane_bwr_density(
    temperature: Tensor,
    pressure: Tensor,
    *,
    phase: Literal["liquid", "vapor"] = "vapor",
) -> Tensor:
    """Solve the McCarty BWR methane density at SI pressure.

    Parameters
    ----------
    temperature
        Positive temperature in K with arbitrary leading batch dimensions.
    pressure
        Positive pressure in Pa, broadcastable with ``temperature``.
    phase
        Select the lowest-density vapor root or highest-density liquid root.

    Returns
    -------
    Tensor
        Methane molar density in mol/L with the broadcast state shape.

    Raises
    ------
    InvalidStateError
        If the state is nonpositive or the density scan finds no pressure root.
    ValueError
        If ``phase`` is not ``"liquid"`` or ``"vapor"``.
    """
    if bool((temperature <= 0.0).any()) or bool((pressure <= 0.0).any()):
        raise InvalidStateError("temperature and pressure must be positive")
    if phase not in ("liquid", "vapor"):
        raise ValueError(f"unknown viscosity phase {phase!r}")
    broadcast_temperature, broadcast_pressure = torch.broadcast_tensors(temperature, pressure)
    pressure_atm = broadcast_pressure / 101_325.0
    grid_values = torch.logspace(
        -10.0,
        _METHANE_BWR_DENSITY_SCAN_LOG10_MAX,
        240,
        dtype=temperature.dtype,
        device=temperature.device,
    )
    grid = grid_values.expand((*broadcast_temperature.shape, grid_values.numel()))
    values = methane_bwr_pressure(broadcast_temperature[..., None], grid) - pressure_atm[..., None]
    sign_change = (
        torch.isfinite(values[..., :-1])
        & torch.isfinite(values[..., 1:])
        & (torch.signbit(values[..., :-1]) != torch.signbit(values[..., 1:]))
    )
    if bool((~sign_change.any(dim=-1)).any()):
        raise InvalidStateError("methane BWR density scan found no pressure root")
    first = torch.argmax(sign_change.to(torch.int64), dim=-1)
    reverse = torch.argmax(torch.flip(sign_change, dims=(-1,)).to(torch.int64), dim=-1)
    last = sign_change.shape[-1] - 1 - reverse
    index = first if phase == "vapor" else last
    left = torch.gather(grid, -1, index[..., None]).squeeze(-1)
    right = torch.gather(grid, -1, (index + 1)[..., None]).squeeze(-1)
    left_value = methane_bwr_pressure(broadcast_temperature, left) - pressure_atm
    residual_tolerance = 1.0e-11 * torch.clamp_min(pressure_atm.detach().abs(), 1.0)
    for iteration in range(80):
        density = 0.5 * (left + right)
        value = methane_bwr_pressure(broadcast_temperature, density) - pressure_atm
        if (iteration + 1) % 8 == 0 and bool((value.detach().abs() <= residual_tolerance).all()):
            break
        changes_left = torch.signbit(left_value) != torch.signbit(value)
        right = torch.where(changes_left, density, right)
        left = torch.where(changes_left, left, density)
        left_value = torch.where(changes_left, left_value, value)
    for _ in range(4):
        residual = methane_bwr_pressure(broadcast_temperature, density) - pressure_atm
        derivative = _methane_bwr_density_derivative(broadcast_temperature, density)
        density = density - residual / derivative
    return density


def methane_viscosity(
    temperature: Tensor,
    density_mol_l: Tensor,
) -> Tensor:
    """Return methane dynamic viscosity in Pa s from Eq. 10.6.

    Parameters
    ----------
    temperature
        Temperature in K.
    density_mol_l
        Methane molar density in mol/L.

    Returns
    -------
    Tensor
        Dynamic viscosity in Pa s.

    Notes
    -----
    Although the 2024 textbook prose labels the correlation density as mol/L,
    the published Hanley coefficients require mass density in kg/L (numerically
    equal to g/cm3). Using molar density inside the fractional powers produces
    viscosities two orders of magnitude too large at reservoir pressures.
    """
    gv = torch.tensor(_GV, dtype=temperature.dtype, device=temperature.device)
    powers = torch.arange(-3, 6, dtype=temperature.dtype, device=temperature.device) / 3.0
    dilute = torch.sum(gv * temperature[..., None].pow(powers), dim=-1)
    eta1 = 1.696985927 - 0.133372346 * (1.4 - torch.log(temperature / 168.0)).square()
    mass_density = density_mol_l * _METHANE_MOLAR_MASS_G / 1000.0
    theta = (mass_density - _METHANE_CRITICAL_MASS_DENSITY) / _METHANE_CRITICAL_MASS_DENSITY
    j = torch.tensor(_J, dtype=temperature.dtype, device=temperature.device)
    dense = torch.exp(j[0] + j[3] / temperature) * (
        torch.exp(
            mass_density.pow(0.1) * (j[1] + j[2] / temperature.pow(1.5))
            + theta
            * mass_density.sqrt()
            * (j[4] + j[5] / temperature + j[6] / temperature.square())
        )
        - 1.0
    )
    low_values = cast(Mapping[str, object], _TRANSPORT_PARAMETERS["methane"])[
        "viscosity_low_temperature"
    ]
    low = torch.as_tensor(low_values, dtype=temperature.dtype, device=temperature.device)
    low_dense = torch.exp(low[0] + low[3] / temperature) * (
        torch.exp(
            mass_density.pow(0.1) * (low[1] + low[2] / temperature.pow(1.5))
            + mass_density.sqrt() * (low[4] + low[5] / temperature + low[6] / temperature.square())
        )
        - 1.0
    )
    transition = 0.5 * (torch.tanh(temperature - 91.0) + 1.0)
    dense = transition * dense + (1.0 - transition) * low_dense
    viscosity_cp = 1.0e-4 * (dilute + eta1 * mass_density + dense)
    return viscosity_cp * 1.0e-3


def _mixture_pseudocritical(components: ComponentSet, composition: Tensor) -> tuple[Tensor, Tensor]:
    z = composition / composition.sum(dim=-1, keepdim=True)
    tc = components.critical_temperature
    pc = components.critical_pressure
    q = (tc / pc).pow(1.0 / 3.0)
    qsum_cubed = (q[:, None] + q[None, :]).pow(3)
    tcij = torch.sqrt(tc[:, None] * tc[None, :])
    weights = z[..., :, None] * z[..., None, :]
    volume_sum = torch.sum(weights * qsum_cubed, dim=(-2, -1))
    numerator = torch.sum(weights * qsum_cubed * tcij, dim=(-2, -1))
    tc_mix = numerator / volume_sum
    pc_mix = 8.0 * numerator / volume_sum.square()
    return tc_mix, pc_mix


def _pedersen_molecular_weight(molar_mass_kg: Tensor, composition: Tensor) -> Tensor:
    z = composition / composition.sum(dim=-1, keepdim=True)
    molar_mass_g = 1000.0 * molar_mass_kg
    mn = torch.sum(z * molar_mass_g, dim=-1)
    mw = torch.sum(z * molar_mass_g.square(), dim=-1) / mn
    return 1.304e-4 * (mw.pow(2.303) - mn.pow(2.303)) + mn


def corresponding_states_viscosity(
    temperature: Tensor,
    pressure: Tensor,
    composition: Tensor,
    components: ComponentSet,
    *,
    phase: Literal["liquid", "vapor"] = "vapor",
) -> Tensor:
    """Evaluate the Pedersen corresponding-states mixture viscosity.

    Parameters
    ----------
    temperature
        Positive temperature in K with arbitrary leading batch dimensions.
    pressure
        Positive pressure in Pa, broadcastable with the state/composition
        batch.
    composition
        Mole fractions with components on the final axis.
    components
        Ordered critical-property and molar-mass data matching ``composition``.
    phase
        Methane-reference density root, ``"liquid"`` or ``"vapor"``.

    Returns
    -------
    Tensor
        Mixture dynamic viscosity in Pa s with the broadcast batch shape.

    Raises
    ------
    ValueError
        If the composition has no component axis or its component count
        differs from ``components``.
    InvalidStateError
        If a required methane reference-state density cannot be solved.

    Notes
    -----
    This is a homogeneous-phase correlation and does not perform phase
    equilibrium. Supply the composition and phase root returned by the
    relevant state or flash calculation.
    """
    if composition.ndim < 1:
        raise ValueError("composition must have a final component axis")
    if composition.shape[-1] != components.ncomponents:
        raise ValueError("composition and component set sizes must match")
    z = normalize_composition(composition)
    tc_mix, pc_mix = _mixture_pseudocritical(components, z)
    mapped_t = temperature * _METHANE_TC / tc_mix
    mapped_p = pressure * _METHANE_PC / pc_mix
    reference_density = methane_bwr_density(mapped_t, mapped_p, phase=phase)
    reference_mass_density = reference_density * _METHANE_MOLAR_MASS_G / 1000.0
    reduced_density = reference_mass_density / _METHANE_CRITICAL_MASS_DENSITY
    mixture_molar_mass = _pedersen_molecular_weight(components.molar_mass, z)
    alpha_mix = 1.0 + 7.378e-3 * reduced_density.pow(1.847) * mixture_molar_mass.pow(0.5173)
    alpha_reference = 1.0 + 7.378e-3 * reduced_density.pow(1.847) * temperature.new_tensor(
        _METHANE_MOLAR_MASS_G
    ).pow(0.5173)
    reference_pressure = pressure * _METHANE_PC * alpha_reference / (pc_mix * alpha_mix)
    reference_temperature = temperature * _METHANE_TC * alpha_reference / (tc_mix * alpha_mix)
    density = methane_bwr_density(reference_temperature, reference_pressure, phase=phase)
    reference_viscosity = methane_viscosity(reference_temperature, density)
    return (
        (tc_mix / _METHANE_TC).pow(-1.0 / 6.0)
        * (pc_mix / _METHANE_PC).pow(2.0 / 3.0)
        * (mixture_molar_mass / _METHANE_MOLAR_MASS_G).sqrt()
        * (alpha_mix / alpha_reference)
        * reference_viscosity
    )


def lbc_pseudocomponent_critical_volume(
    molar_mass: Tensor,
    standard_liquid_density: Tensor,
) -> Tensor:
    """Estimate a C7+ critical molar volume from Pedersen Eq. 10.41.

    Parameters
    ----------
    molar_mass
        Pseudo-component molar mass in kg/mol.
    standard_liquid_density
        Standard liquid mass density in kg/m3.

    Returns
    -------
    Tensor
        Estimated critical molar volume in m3/mol.

    Raises
    ------
    ValueError
        If either input is nonfinite or nonpositive.

    Notes
    -----
    Parameters are SI: molar mass in kg/mol and liquid density in kg/m3.
    The returned critical molar volume is in m3/mol. The underlying empirical
    equation uses g/mol, g/cm3, and ft3/lbmol.
    """
    if bool(
        (
            (~torch.isfinite(molar_mass))
            | (~torch.isfinite(standard_liquid_density))
            | (molar_mass <= 0.0)
            | (standard_liquid_density <= 0.0)
        ).any()
    ):
        raise ValueError("molar mass and standard liquid density must be finite and positive")
    molar_mass_g = 1000.0 * molar_mass
    density_g_cm3 = standard_liquid_density / 1000.0
    volume_ft3_lbmol = (
        21.573
        + 0.015122 * molar_mass_g
        - 27.656 * density_g_cm3
        + 0.070615 * molar_mass_g * density_g_cm3
    )
    return volume_ft3_lbmol * _FT3_PER_LBMOL_TO_M3_PER_MOL


def lbc_viscosity(
    temperature: Tensor,
    molar_density: Tensor,
    composition: Tensor,
    components: ComponentSet,
    *,
    critical_volume: Tensor | None = None,
    coefficients: Tensor | None = None,
) -> Tensor:
    """Return Lohrenz-Bray-Clark mixture viscosity in Pa s.

    Parameters
    ----------
    temperature
        Temperature in K.
    molar_density
        Homogeneous-phase molar density in mol/m3, normally obtained from the
        same EoS used by the flash calculation.
    composition
        Mole fractions with components on the final axis.
    components
        Critical temperatures, pressures, molar masses, and volumes.
    critical_volume
        Optional m3/mol values overriding the component data. This is the
        principal LBC tuning parameter for petroleum pseudocomponents.
    coefficients
        Optional trainable or fitted ``a1...a5`` tensor replacing the original
        LBC constants.

    Returns
    -------
    Tensor
        Mixture dynamic viscosity in Pa s.

    Raises
    ------
    ValueError
        If component shapes, critical volumes, or coefficient tensors are
        inconsistent.
    InvalidStateError
        If temperature/density or a required reducing property is outside its
        physical domain.

    Notes
    -----
    The Stiel-Thodos reducing parameters use K, atm, and g/mol, as required by
    the published numerical constants. LBC is empirical and often needs
    pseudocomponent critical-volume or coefficient tuning for heavy oils.
    """
    if composition.ndim < 1:
        raise ValueError("composition must have a final component axis")
    if composition.shape[-1] != components.ncomponents:
        raise ValueError("composition and component set sizes must match")
    if bool(
        (
            (~torch.isfinite(temperature))
            | (~torch.isfinite(molar_density))
            | (temperature <= 0.0)
            | (molar_density < 0.0)
        ).any()
    ):
        raise InvalidStateError("temperature must be positive and molar density non-negative")

    volumes = components.critical_volume if critical_volume is None else critical_volume
    if volumes is None:
        raise ValueError("critical molar volumes are required by the LBC correlation")
    if volumes.shape != components.critical_temperature.shape:
        raise ValueError("critical_volume must have one value per component")
    if bool(((~torch.isfinite(volumes)) | (volumes <= 0.0)).any()):
        raise ValueError("critical molar volumes must be finite and positive")

    parameters = (
        temperature.new_tensor(_LBC)
        if coefficients is None
        else coefficients.to(dtype=temperature.dtype, device=temperature.device)
    )
    if parameters.shape != (5,) or bool((~torch.isfinite(parameters)).any()):
        raise ValueError("LBC coefficients must be a finite five-element tensor")

    z = normalize_composition(composition)
    tc = components.critical_temperature
    pc_atm = components.critical_pressure / 101_325.0
    molar_mass_g = 1000.0 * components.molar_mass
    xi_components = tc.pow(1.0 / 6.0) / (molar_mass_g.sqrt() * pc_atm.pow(2.0 / 3.0))
    reduced_temperature = temperature[..., None] / tc
    dilute_components_cp = torch.where(
        reduced_temperature <= 1.5,
        34.0e-5 * reduced_temperature.pow(0.94) / xi_components,
        17.78e-5 * (4.58 * reduced_temperature - 1.67).pow(5.0 / 8.0) / xi_components,
    )
    square_root_mass = molar_mass_g.sqrt()
    dilute_mixture_cp = torch.sum(
        z * dilute_components_cp * square_root_mass,
        dim=-1,
    ) / torch.sum(z * square_root_mass, dim=-1)

    mixture_xi = torch.sum(z * tc, dim=-1).pow(1.0 / 6.0) / (
        torch.sum(z * molar_mass_g, dim=-1).sqrt() * torch.sum(z * pc_atm, dim=-1).pow(2.0 / 3.0)
    )
    reduced_density = molar_density * torch.sum(z * volumes, dim=-1)
    polynomial = parameters[0] + reduced_density * (
        parameters[1]
        + reduced_density
        * (parameters[2] + reduced_density * (parameters[3] + reduced_density * parameters[4]))
    )
    dense_increment_cp = (polynomial.pow(4) - 1.0e-4) / mixture_xi
    viscosity_cp = dilute_mixture_cp + dense_increment_cp
    if bool((~torch.isfinite(viscosity_cp) | (viscosity_cp <= 0.0)).any()):
        raise InvalidStateError("LBC correlation produced a non-positive viscosity")
    return viscosity_cp * 1.0e-3


def kinematic_viscosity(dynamic_viscosity: Tensor, mass_density: Tensor) -> Tensor:
    """Return kinematic viscosity from dynamic viscosity and mass density.

    Parameters
    ----------
    dynamic_viscosity
        Dynamic viscosity in Pa s.
    mass_density
        Homogeneous-phase mass density in kg/m3, broadcastable with
        ``dynamic_viscosity``.

    Returns
    -------
    Tensor
        Kinematic viscosity in m2/s with the broadcast input shape.

    Raises
    ------
    InvalidStateError
        If viscosity is negative or either input is nonfinite or density is
        nonpositive.

    Notes
    -----
    This is the SI definition in Pedersen et al. (2024), section 10.1, rather
    than an equilibrium or empirical transport model.
    """
    viscosity, density = torch.broadcast_tensors(dynamic_viscosity, mass_density)
    invalid = (
        (~torch.isfinite(viscosity))
        | (~torch.isfinite(density))
        | (viscosity < 0.0)
        | (density <= 0.0)
    )
    if bool(invalid.any()):
        raise InvalidStateError("viscosity must be non-negative and density finite and positive")
    result: Tensor = viscosity / density
    return result


def lee_gas_viscosity(
    temperature: Tensor,
    mass_density: Tensor,
    molar_mass: Tensor,
) -> Tensor:
    """Evaluate the Lee-Gonzalez-Eakin natural-gas viscosity correlation.

    Parameters
    ----------
    temperature
        Gas temperature in K.
    mass_density
        Gas mass density in kg/m3.
    molar_mass
        Mixture molar mass in kg/mol.

    Returns
    -------
    Tensor
        Dynamic viscosity in Pa s with the broadcast input shape.

    Raises
    ------
    InvalidStateError
        If temperature or molar mass is nonpositive, density is negative, or
        any input or output is nonfinite.

    Notes
    -----
    Implements Lee, Gonzalez, and Eakin, "The Viscosity of Natural Gases,"
    *J. Petroleum Technology* 18 (1966), 997-1000,
    doi:10.2118/1340-PA, as reproduced in Pedersen et al. (2024),
    Eqs. 10.46-10.49. The published equation uses degrees Rankine, lb/ft3,
    g/mol, and cP; this API converts SI inputs and returns SI viscosity. It is
    a gas correlation and is not valid for a condensed liquid phase.
    """
    temperature, density, molecular_weight = torch.broadcast_tensors(
        temperature,
        mass_density,
        molar_mass,
    )
    invalid = (
        (~torch.isfinite(temperature))
        | (~torch.isfinite(density))
        | (~torch.isfinite(molecular_weight))
        | (temperature <= 0.0)
        | (density < 0.0)
        | (molecular_weight <= 0.0)
    )
    if bool(invalid.any()):
        raise InvalidStateError(
            "temperature and molar mass must be positive and gas density non-negative"
        )
    temperature_rankine = 1.8 * temperature
    molecular_weight_g = 1000.0 * molecular_weight
    density_lb_ft3 = density / _LB_FT3_TO_KG_M3
    x = 3.5 + 986.0 / temperature_rankine + 0.01 * molecular_weight_g
    y = 2.4 - 0.2 * x
    k = (
        (9.4 + 0.02 * molecular_weight_g)
        * temperature_rankine.pow(1.5)
        / (209.0 + 19.0 * molecular_weight_g + temperature_rankine)
    )
    viscosity_cp = 1.0e-4 * k * torch.exp(x * (density_lb_ft3 / 62.4).pow(y))
    if bool((~torch.isfinite(viscosity_cp) | (viscosity_cp <= 0.0)).any()):
        raise InvalidStateError("Lee gas correlation produced a non-positive viscosity")
    result: Tensor = 1.0e-3 * viscosity_cp
    return result


def stabilized_heavy_oil_viscosity(
    temperature: Tensor,
    pressure: Tensor,
    number_average_molar_mass: Tensor,
    weight_average_molar_mass: Tensor,
    *,
    third_csp: Tensor | float = 1.0,
    fourth_csp: Tensor | float = 1.0,
) -> Tensor:
    """Evaluate the Lindeloff stabilized/live-heavy-oil viscosity branch.

    Parameters
    ----------
    temperature
        Temperature in K.
    pressure
        Pressure in Pa.
    number_average_molar_mass, weight_average_molar_mass
        Mixture molecular-weight averages in kg/mol.
    third_csp, fourth_csp
        Dimensionless fitted factors in Pedersen et al. (2024),
        Eqs. 10.34-10.36. Defaults reproduce the published predictive branch.

    Returns
    -------
    Tensor
        Dynamic viscosity in Pa s.

    Raises
    ------
    InvalidStateError
        If the state or molecular-weight averages are outside their physical
        domains, or the correlation returns a nonfinite result.

    Notes
    -----
    Implements Lindeloff et al., "The corresponding states viscosity model
    applied to heavy oil systems," *J. Can. Petroleum Technology* 43 (2004),
    47-53, through Pedersen et al. (2024), Eqs. 10.33-10.37. This function is
    the empirical heavy-oil branch only; use
    :func:`heavy_oil_corresponding_states_viscosity` for the published
    low-reference-temperature blending protocol.
    """
    third = torch.as_tensor(third_csp, dtype=temperature.dtype, device=temperature.device)
    fourth = torch.as_tensor(fourth_csp, dtype=temperature.dtype, device=temperature.device)
    temperature, pressure, mn, mw, third, fourth = torch.broadcast_tensors(
        temperature,
        pressure,
        number_average_molar_mass,
        weight_average_molar_mass,
        third,
        fourth,
    )
    invalid = (
        (~torch.isfinite(temperature))
        | (~torch.isfinite(pressure))
        | (~torch.isfinite(mn))
        | (~torch.isfinite(mw))
        | (~torch.isfinite(third))
        | (~torch.isfinite(fourth))
        | (temperature <= 0.0)
        | (pressure <= 0.0)
        | (mn <= 0.0)
        | (mw < mn)
        | (third <= 0.0)
        | (fourth <= 0.0)
    )
    if bool(invalid.any()):
        raise InvalidStateError(
            "heavy-oil state, molecular-weight averages, and CSP factors must be physical"
        )
    mn_g = 1000.0 * mn
    mw_g = 1000.0 * mw
    visfac3 = 0.2252 * temperature / mn_g + 0.9738
    visfac4 = 0.5354 * visfac3 - 0.1170
    ratio = mw_g / mn_g
    low_base = 1.5 / (visfac3 * third)
    high_base = mw_g / (visfac3 * third * mn_g)
    representative_mass = mn_g * torch.where(
        ratio <= 1.5,
        low_base.pow(visfac4 * fourth),
        high_base.pow(visfac4 * fourth),
    )
    mass_sign = torch.where(temperature > 564.49, 1.0, -1.0)
    log10_viscosity_cp = (
        -0.07995
        + mass_sign * 0.01101 * representative_mass
        - 371.8 / temperature
        + 6.215 * representative_mass / temperature
    )
    viscosity_atmospheric_cp = torch.pow(temperature.new_tensor(10.0), log10_viscosity_cp)
    pressure_atm = pressure / 101_325.0
    pressure_factor = torch.exp(0.00384 * (pressure_atm.pow(0.8226) - 1.0) / 0.8226)
    viscosity = 1.0e-3 * viscosity_atmospheric_cp * pressure_factor
    if bool((~torch.isfinite(viscosity) | (viscosity <= 0.0)).any()):
        raise InvalidStateError("heavy-oil correlation produced a non-positive viscosity")
    return viscosity


def heavy_oil_corresponding_states_viscosity(
    temperature: Tensor,
    pressure: Tensor,
    composition: Tensor,
    components: ComponentSet,
    *,
    phase: Literal["liquid", "vapor"] = "liquid",
    third_csp: Tensor | float = 1.0,
    fourth_csp: Tensor | float = 1.0,
) -> Tensor:
    """Evaluate the blended corresponding-states heavy-oil viscosity model.

    Parameters
    ----------
    temperature, pressure
        Positive SI state variables in K and Pa.
    composition
        Mole fractions with components on the final axis.
    components
        Critical properties and molar masses in the same order.
    phase
        Methane-reference BWR density branch.
    third_csp, fourth_csp
        Dimensionless Lindeloff tuning factors.

    Returns
    -------
    Tensor
        Dynamic viscosity in Pa s over the broadcast state/composition batch.

    Notes
    -----
    Pedersen et al. (2024), section 10.1.2, retains the conventional
    corresponding-states result above a methane reference temperature of
    75 K, uses the Lindeloff branch below 65 K, and linearly blends the two
    between 65 and 75 K. Calibration factors are explicit tensors so fitting
    preserves PyTorch gradients.
    """
    if composition.ndim < 1 or composition.shape[-1] != components.ncomponents:
        raise ValueError("composition and component set sizes must match")
    z = normalize_composition(composition)
    tc_mix, pc_mix = _mixture_pseudocritical(components, z)
    mapped_temperature = temperature * _METHANE_TC / tc_mix
    mapped_pressure = pressure * _METHANE_PC / pc_mix
    initial_density = methane_bwr_density(mapped_temperature, mapped_pressure, phase=phase)
    reduced_density = (
        initial_density * _METHANE_MOLAR_MASS_G / 1000.0 / _METHANE_CRITICAL_MASS_DENSITY
    )
    molar_mass_g = 1000.0 * components.molar_mass
    mn_g = torch.sum(z * molar_mass_g, dim=-1)
    mw_g = torch.sum(z * molar_mass_g.square(), dim=-1) / mn_g
    mixture_mass = 1.304e-4 * (mw_g.pow(2.303) - mn_g.pow(2.303)) + mn_g
    alpha_mix = 1.0 + 7.378e-3 * reduced_density.pow(1.847) * mixture_mass.pow(0.5173)
    alpha_reference = 1.0 + 7.378e-3 * reduced_density.pow(1.847) * temperature.new_tensor(
        _METHANE_MOLAR_MASS_G
    ).pow(0.5173)
    reference_temperature = temperature * _METHANE_TC * alpha_reference / (tc_mix * alpha_mix)
    conventional = corresponding_states_viscosity(
        temperature,
        pressure,
        z,
        components,
        phase=phase,
    )
    heavy = stabilized_heavy_oil_viscosity(
        temperature,
        pressure,
        mn_g / 1000.0,
        mw_g / 1000.0,
        third_csp=third_csp,
        fourth_csp=fourth_csp,
    )
    heavy_weight = torch.clamp((75.0 - reference_temperature) / 10.0, 0.0, 1.0)
    return torch.lerp(conventional, heavy, heavy_weight)


def _friction_reduced_coefficients(
    temperature: Tensor,
    eos: CubicEOS,
    family: Literal["SRK", "PR"],
) -> tuple[Tensor, Tensor, Tensor]:
    block = cast(
        Mapping[str, object],
        cast(Mapping[str, object], _TRANSPORT_PARAMETERS["friction_theory"])[family],
    )
    critical = torch.as_tensor(
        block["critical"],
        dtype=temperature.dtype,
        device=temperature.device,
    )
    attractive = torch.as_tensor(
        block["attractive"],
        dtype=temperature.dtype,
        device=temperature.device,
    )
    repulsive = torch.as_tensor(
        block["repulsive"],
        dtype=temperature.dtype,
        device=temperature.device,
    )
    quadratic = temperature.new_tensor(cast(float, block["quadratic_repulsive"]))
    gamma = eos.critical_temperature / temperature[..., None]
    psi = _R_BAR_CM3 * eos.critical_temperature / (eos.critical_pressure / 1.0e5)

    def residual(parameters: Tensor) -> Tensor:
        return (
            parameters[0] * (gamma - 1.0)
            + (parameters[1] + parameters[2] * psi) * torch.expm1(gamma - 1.0)
            + (parameters[3] + parameters[4] * psi + parameters[5] * psi.square())
            * torch.expm1(2.0 * gamma - 2.0)
        )

    ka = critical[0] + residual(attractive)
    kr = critical[1] + residual(repulsive)
    krr = critical[2] + quadratic * psi * torch.expm1(2.0 * gamma) * (gamma - 1.0).square()
    return kr, ka, krr


def _chung_dilute_viscosity(
    temperature: Tensor,
    eos: CubicEOS,
    critical_volume: Tensor,
) -> Tensor:
    molecular_weight = 1000.0 * eos.molar_mass
    volume_cm3_mol = 1.0e6 * critical_volume
    reduced_temperature = 1.2593 * temperature[..., None] / eos.critical_temperature
    collision = (
        1.16145 / reduced_temperature.pow(0.14874)
        + 0.52487 / torch.exp(0.77320 * reduced_temperature)
        + 2.16178 / torch.exp(2.43787 * reduced_temperature)
        - 6.435e-4
        * reduced_temperature.pow(0.14874)
        * torch.sin(18.0323 * reduced_temperature.pow(-0.76830) - 7.27371)
    )
    correction = 1.0 - 0.2756 * eos.acentric_factor
    viscosity_micropoise = (
        40.785
        * torch.sqrt(molecular_weight * temperature[..., None])
        * correction
        / (volume_cm3_mol.pow(2.0 / 3.0) * collision)
    )
    return 1.0e-7 * viscosity_micropoise


def friction_theory_viscosity(
    temperature: Tensor,
    pressure: Tensor,
    composition: Tensor,
    eos: CubicEOS,
    *,
    phase: PhaseKind = "stable",
    critical_viscosity: Tensor | None = None,
    critical_volume: Tensor | None = None,
) -> Tensor:
    """Evaluate the one-parameter friction-theory cubic-EOS viscosity.

    Parameters
    ----------
    temperature, pressure
        Positive state variables in K and Pa.
    composition
        Nonpolar-fluid mole fractions with components on the final axis.
    eos
        Native SRK or PR76 :class:`~torch_flash.eos.cubic.CubicEOS`. Its
        mixing rule and binary interactions also define the pressure split.
    phase
        Cubic root selection for the supplied homogeneous state.
    critical_viscosity
        Optional characteristic component viscosities in Pa s. When omitted,
        the n-alkane correlation of Eq. 21 is used.
    critical_volume
        Optional component critical volumes in m3/mol for the Chung dilute-gas
        term. Defaults to the component database.

    Returns
    -------
    Tensor
        Dynamic viscosity in Pa s for each broadcast state.

    Raises
    ------
    ValueError
        If the EOS is not the published SRK or PR76 family, or component
        vectors have inconsistent shapes.
    InvalidStateError
        If a required volume or state is nonphysical or the result is
        nonpositive.

    Notes
    -----
    Implements Quiñones-Cisneros, Zéberg-Mikkelsen, and Stenby,
    "One parameter friction theory models for viscosity," *Fluid Phase
    Equilibria* 178 (2001), Eqs. 1-30 and Tables 1-2,
    doi:10.1016/S0378-3812(00)00474-X. The parameterization was developed for
    nonpolar fluids and n-alkane mixtures. PR78 and PRSV alpha functions are
    not silently identified with the paper's PR76 model.
    """
    if eos.constants.alpha_kind == "srk":
        family: Literal["SRK", "PR"] = "SRK"
    elif eos.constants.alpha_kind == "pr76":
        family = "PR"
    else:
        raise ValueError("friction theory supports only the published SRK and PR76 families")
    if composition.ndim < 1 or composition.shape[-1] != eos.ncomponents:
        raise ValueError("composition and cubic EOS component sizes must match")
    if bool(
        (
            (~torch.isfinite(temperature))
            | (~torch.isfinite(pressure))
            | (temperature <= 0.0)
            | (pressure <= 0.0)
        ).any()
    ):
        raise InvalidStateError(
            "friction-theory temperature and pressure must be finite and positive"
        )
    z = normalize_composition(composition)
    volumes = eos.critical_volume if critical_volume is None else critical_volume
    if volumes is None or volumes.shape != eos.critical_temperature.shape:
        raise ValueError("friction theory requires one critical volume per component")
    if bool(((~torch.isfinite(volumes)) | (volumes <= 0.0)).any()):
        raise ValueError("critical volumes must be finite and positive")
    if critical_viscosity is None:
        pressure_bar = eos.critical_pressure / 1.0e5
        molecular_weight_g = 1000.0 * eos.molar_mass
        eta_c = 1.0e-7 * 0.597556 * pressure_bar * molecular_weight_g.pow(0.601652)
    else:
        eta_c = critical_viscosity.to(dtype=temperature.dtype, device=temperature.device)
        if eta_c.shape != eos.critical_temperature.shape:
            raise ValueError("critical_viscosity must have one value per component")
        if bool(((~torch.isfinite(eta_c)) | (eta_c <= 0.0)).any()):
            raise ValueError("critical viscosities must be finite and positive")
    kr_hat, ka_hat, krr_hat = _friction_reduced_coefficients(
        temperature,
        eos,
        family,
    )
    molecular_weight_g = 1000.0 * eos.molar_mass
    epsilon = 0.30
    denominator = torch.sum(z / molecular_weight_g.pow(epsilon), dim=-1, keepdim=True)
    weights = z / (molecular_weight_g.pow(epsilon) * denominator)
    kr = torch.sum(weights * eta_c * kr_hat / eos.critical_pressure, dim=-1)
    ka = torch.sum(weights * eta_c * ka_hat / eos.critical_pressure, dim=-1)
    krr = torch.sum(weights * eta_c * krr_hat / eos.critical_pressure.square(), dim=-1)
    dilute_components = _chung_dilute_viscosity(temperature, eos, volumes)
    dilute = torch.exp(torch.sum(z * torch.log(dilute_components), dim=-1))
    physical_volume = eos.molar_volume(temperature, pressure, z, phase)
    translation = torch.sum(
        z * eos.component_volume_translation(temperature),
        dim=-1,
    )
    volume = physical_volume - translation
    attraction, covolume = eos.mixture_parameters(temperature, z)
    repulsive_pressure = R * temperature / (volume - covolume)
    attractive_pressure = -attraction / (
        (volume + eos.constants.delta1 * covolume) * (volume + eos.constants.delta2 * covolume)
    )
    viscosity = (
        dilute
        + kr * repulsive_pressure
        + ka * attractive_pressure
        + krr * repulsive_pressure.square()
    )
    if bool((~torch.isfinite(viscosity) | (viscosity <= 0.0)).any()):
        raise InvalidStateError("friction theory produced a non-positive viscosity")
    return viscosity
