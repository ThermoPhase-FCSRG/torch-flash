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

from typing import Literal

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.exceptions import InvalidStateError
from torch_flash.types import normalize_composition

_METHANE_TC = 190.564
_METHANE_PC = 4.5992e6
_METHANE_MOLAR_MASS_G = 16.04246
_METHANE_CRITICAL_MASS_DENSITY = 0.16266  # kg/L = g/cm3
_R_L_ATM = 0.08205616
_GAMMA = 0.0096

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


def _bwr_coefficients(temperature: Tensor) -> Tensor:
    n = torch.tensor(_N, dtype=temperature.dtype, device=temperature.device)
    t = temperature
    values = (
        _R_L_ATM * t,
        n[0] * t + n[1] * torch.sqrt(t) + n[2] + n[3] / t + n[4] / t.square(),
        n[5] * t + n[6] + n[7] / t + n[8] / t.square(),
        n[9] * t + n[10] + n[11] / t,
        n[12],
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
    return torch.stack(values)


def methane_bwr_pressure(temperature: Tensor, density_mol_l: Tensor) -> Tensor:
    """Return McCarty BWR methane pressure in atm."""
    coefficients = _bwr_coefficients(temperature)
    low_powers = torch.arange(1, 10, dtype=temperature.dtype, device=temperature.device)
    high_powers = torch.arange(3, 14, 2, dtype=temperature.dtype, device=temperature.device)
    polynomial = torch.sum(coefficients[:9] * density_mol_l.pow(low_powers))
    exponential = torch.sum(
        coefficients[9:]
        * density_mol_l.pow(high_powers)
        * torch.exp(-_GAMMA * density_mol_l.square())
    )
    return polynomial + exponential


def methane_bwr_density(
    temperature: Tensor,
    pressure: Tensor,
    *,
    phase: Literal["liquid", "vapor"] = "vapor",
) -> Tensor:
    """Solve the methane BWR density in mol/L at SI pressure."""
    if bool((temperature <= 0.0).any()) or bool((pressure <= 0.0).any()):
        raise InvalidStateError("temperature and pressure must be positive")
    if phase not in ("liquid", "vapor"):
        raise ValueError(f"unknown viscosity phase {phase!r}")
    pressure_atm = pressure / 101_325.0
    grid = torch.logspace(
        -10.0,
        torch.log10(pressure.new_tensor(50.0)).item(),
        240,
        dtype=temperature.dtype,
        device=temperature.device,
    )

    def residual(density: Tensor) -> Tensor:
        return methane_bwr_pressure(temperature, density) - pressure_atm

    brackets: list[tuple[Tensor, Tensor]] = []
    left = grid[0]
    left_value = residual(left)
    for right in grid[1:]:
        right_value = residual(right)
        if bool(torch.isfinite(left_value) & torch.isfinite(right_value)) and bool(
            torch.signbit(left_value) != torch.signbit(right_value)
        ):
            brackets.append((left, right))
        left, left_value = right, right_value
    if not brackets:
        raise InvalidStateError("methane BWR density scan found no pressure root")
    left, right = brackets[0] if phase == "vapor" else brackets[-1]
    left_value = residual(left)
    for _ in range(80):
        density = 0.5 * (left + right)
        value = residual(density)
        if float(value.detach().abs()) <= 1.0e-11 * max(float(pressure_atm.detach()), 1.0):
            break
        if bool(torch.signbit(left_value) != torch.signbit(value)):
            right = density
        else:
            left = density
            left_value = value
    for _ in range(4):
        density = density - residual(density) / torch.func.grad(residual)(density)
    return density


def methane_viscosity(
    temperature: Tensor,
    density_mol_l: Tensor,
) -> Tensor:
    """Return methane dynamic viscosity in Pa s from Eq. 10.6.

    Although the 2024 textbook prose labels the correlation density as mol/L,
    the published Hanley coefficients require mass density in kg/L (numerically
    equal to g/cm3). Using molar density inside the fractional powers produces
    viscosities two orders of magnitude too large at reservoir pressures.
    """
    gv = torch.tensor(_GV, dtype=temperature.dtype, device=temperature.device)
    powers = torch.arange(-3, 6, dtype=temperature.dtype, device=temperature.device) / 3.0
    dilute = torch.sum(gv * temperature.pow(powers))
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
    viscosity_cp = 1.0e-4 * (dilute + eta1 * mass_density + dense)
    return viscosity_cp * 1.0e-3


def _mixture_pseudocritical(components: ComponentSet, composition: Tensor) -> tuple[Tensor, Tensor]:
    z = composition / composition.sum()
    tc = components.critical_temperature
    pc = components.critical_pressure
    q = (tc / pc).pow(1.0 / 3.0)
    qsum_cubed = (q[:, None] + q[None, :]).pow(3)
    tcij = torch.sqrt(tc[:, None] * tc[None, :])
    weights = z[:, None] * z[None, :]
    volume_sum = torch.sum(weights * qsum_cubed)
    tc_mix = torch.sum(weights * qsum_cubed * tcij) / volume_sum
    pc_mix = 8.0 * torch.sum(weights * qsum_cubed * tcij) / volume_sum.square()
    return tc_mix, pc_mix


def _pedersen_molecular_weight(molar_mass_kg: Tensor, composition: Tensor) -> Tensor:
    z = composition / composition.sum()
    molar_mass_g = 1000.0 * molar_mass_kg
    mn = torch.sum(z * molar_mass_g)
    mw = torch.sum(z * molar_mass_g.square()) / mn
    return 1.304e-4 * (mw.pow(2.303) - mn.pow(2.303)) + mn


def pedersen_viscosity(
    temperature: Tensor,
    pressure: Tensor,
    composition: Tensor,
    components: ComponentSet,
    *,
    phase: Literal["liquid", "vapor"] = "vapor",
) -> Tensor:
    """Return Pedersen CSP mixture viscosity in Pa s."""
    if composition.ndim != 1:
        raise ValueError("Pedersen viscosity currently accepts one composition vector")
    if composition.numel() != components.ncomponents:
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
        One mole-fraction vector.
    components
        Critical temperatures, pressures, molar masses, and volumes.
    critical_volume
        Optional m3/mol values overriding the component data. This is the
        principal LBC tuning parameter for petroleum pseudocomponents.
    coefficients
        Optional trainable or fitted ``a1...a5`` tensor replacing the original
        LBC constants.

    Notes
    -----
    The Stiel-Thodos reducing parameters use K, atm, and g/mol, as required by
    the published numerical constants. LBC is empirical and often needs
    pseudocomponent critical-volume or coefficient tuning for heavy oils.
    """
    if composition.ndim != 1:
        raise ValueError("LBC viscosity currently accepts one composition vector")
    if composition.numel() != components.ncomponents:
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
    dilute_mixture_cp = torch.sum(z * dilute_components_cp * square_root_mass, dim=-1) / torch.sum(
        z * square_root_mass
    )

    mixture_xi = torch.sum(z * tc).pow(1.0 / 6.0) / (
        torch.sum(z * molar_mass_g).sqrt() * torch.sum(z * pc_atm).pow(2.0 / 3.0)
    )
    reduced_density = molar_density * torch.sum(z * volumes)
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
