"""Thermodynamic properties evaluated without solving phase equilibrium."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

import torch
from torch import Tensor

from torch_flash.constants import STANDARD_PRESSURE, R
from torch_flash.standard_state import StandardState
from torch_flash.types import ChemicalState, PhaseKind, PhaseProperties

from .phase_identification import identify_phase_from_properties


class StateModel(Protocol):
    """Minimal homogeneous-state model interface."""

    def select_z(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return a compressibility factor."""

    def molar_volume(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return molar volume."""

    def log_fugacity_coefficients(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return component log fugacity coefficients."""


class HelmholtzStateModel(StateModel, Protocol):
    """Homogeneous-state model with an extensive residual Helmholtz function."""

    def residual_helmholtz_rt(
        self,
        temperature: Tensor,
        volume: Tensor,
        moles: Tensor,
    ) -> Tensor:
        """Return extensive ``A^R/(R*T)`` at fixed ``T``, total ``V``, and ``n``."""


@dataclass(frozen=True)
class ThermodynamicDerivatives:
    """First homogeneous-state derivatives.

    Component derivatives are provided in both unconstrained softmax-logit
    coordinates and ``n-1`` independent mole fractions, where the last mole
    fraction is ``1 - sum(x_independent)``. Temperature and pressure
    derivatives hold composition fixed.

    Mole-number derivatives hold ``T`` and ``P`` fixed and are evaluated at
    ``n_i = x_i mol`` (a one-mole total basis). Because all returned
    properties are intensive, the corresponding derivative at another total
    amount is this value divided by that amount in mol.

    Fugacity derivatives have Pa per coordinate units.
    ``dlog_fugacity_*`` differentiates the dimensionless
    ``ln(f_i / p_standard)``. Chemical-potential derivatives have J/mol per
    coordinate units; the reduced variants differentiate ``mu_i / (R*T)``.
    Molar-volume derivatives have m3/mol per coordinate units.
    """

    dfugacity_coefficient_dlogits: Tensor
    dfugacity_coefficient_dindependent_composition: Tensor
    dfugacity_coefficient_dtemperature: Tensor
    dfugacity_coefficient_dpressure: Tensor
    dfugacity_coefficient_dmoles: Tensor
    dlog_fugacity_coefficient_dlogits: Tensor
    dlog_fugacity_coefficient_dindependent_composition: Tensor
    dlog_fugacity_coefficient_dtemperature: Tensor
    dlog_fugacity_coefficient_dpressure: Tensor
    dlog_fugacity_coefficient_dmoles: Tensor
    dfugacity_dlogits: Tensor
    dfugacity_dindependent_composition: Tensor
    dfugacity_dtemperature: Tensor
    dfugacity_dpressure: Tensor
    dfugacity_dmoles: Tensor
    dlog_fugacity_dlogits: Tensor
    dlog_fugacity_dindependent_composition: Tensor
    dlog_fugacity_dtemperature: Tensor
    dlog_fugacity_dpressure: Tensor
    dlog_fugacity_dmoles: Tensor

    dchemical_potential_dlogits: Tensor
    dchemical_potential_dindependent_composition: Tensor
    dchemical_potential_dtemperature: Tensor
    dchemical_potential_dpressure: Tensor
    dchemical_potential_dmoles: Tensor
    dreduced_chemical_potential_dlogits: Tensor
    dreduced_chemical_potential_dindependent_composition: Tensor
    dreduced_chemical_potential_dtemperature: Tensor
    dreduced_chemical_potential_dpressure: Tensor
    dreduced_chemical_potential_dmoles: Tensor
    dmolar_volume_dlogits: Tensor
    dmolar_volume_dindependent_composition: Tensor
    dmolar_volume_dtemperature: Tensor
    dmolar_volume_dpressure: Tensor
    dmolar_volume_dmoles: Tensor
    dgibbs_dtemperature: Tensor
    dgibbs_dpressure: Tensor


def _log_fugacities_from_coefficients(
    log_fugacity_coefficients: Tensor,
    pressure: Tensor,
    composition: Tensor,
) -> Tensor:
    """Return ``ln(f_i / p_standard)`` from log fugacity coefficients."""
    safe_x = torch.clamp_min(composition, torch.finfo(composition.dtype).tiny)
    reduced_pressure = torch.log(pressure / STANDARD_PRESSURE)[..., None]
    return torch.log(safe_x) + log_fugacity_coefficients + reduced_pressure


def _log_fugacities(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    composition: Tensor,
    phase: PhaseKind,
) -> Tensor:
    """Return ``ln(f_i / p_standard)`` using the model fugacity coefficients."""
    log_phi = model.log_fugacity_coefficients(temperature, pressure, composition, phase)
    return _log_fugacities_from_coefficients(log_phi, pressure, composition)


def _chemical_potentials_from_log_fugacities(
    log_fugacity: Tensor,
    temperature: Tensor,
    standard_state: StandardState | None,
) -> Tensor:
    """Return chemical potentials from dimensionless log fugacities."""
    temperature_by_component = temperature[..., None]
    departure = R * temperature_by_component * log_fugacity
    if standard_state is None:
        return departure
    return standard_state.chemical_potential(temperature) + departure


def _chemical_potentials(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    composition: Tensor,
    phase: PhaseKind,
    standard_state: StandardState | None = None,
) -> Tensor:
    log_fugacity = _log_fugacities(model, temperature, pressure, composition, phase)
    return _chemical_potentials_from_log_fugacities(
        log_fugacity,
        temperature,
        standard_state,
    )


def log_fugacities_tv(
    model: HelmholtzStateModel,
    temperature: Tensor,
    volume: Tensor,
    moles: Tensor,
) -> Tensor:
    """Return ``ln(f_i/p_standard)`` from an explicit ``T-V-n`` state.

    ``volume`` is the total physical volume in m3 and ``moles`` are component
    amounts in mol. The identity

    ``ln(f_i/p_standard) = ln(n_i*R*T/(V*p_standard)) + d(A^R/RT)/dn_i``

    provides an independent Helmholtz-route check of the usual TP fugacity
    calculation. The model's residual Helmholtz energy must be extensive and
    referenced to an ideal gas at the same physical volume.
    """
    if temperature.ndim != 0 or volume.ndim != 0:
        raise ValueError("log_fugacities_tv currently accepts scalar temperature and volume")
    if moles.ndim != 1:
        raise ValueError("log_fugacities_tv currently accepts one mole-number vector")
    if bool((temperature <= 0.0) | (volume <= 0.0)):
        raise ValueError("temperature and volume must be positive")
    if bool((~torch.isfinite(moles)).any() | (moles <= 0.0).any()):
        raise ValueError("moles must be finite and strictly positive")

    residual_mu_rt: Tensor = torch.func.jacrev(
        lambda current_moles: model.residual_helmholtz_rt(
            temperature,
            volume,
            current_moles,
        )
    )(moles)
    ideal_log_fugacity = torch.log(moles * R * temperature / (volume * STANDARD_PRESSURE))
    return ideal_log_fugacity + residual_mu_rt


def fugacities_tv(
    model: HelmholtzStateModel,
    temperature: Tensor,
    volume: Tensor,
    moles: Tensor,
) -> Tensor:
    """Return component fugacities in Pa from an explicit ``T-V-n`` state."""
    return STANDARD_PRESSURE * torch.exp(log_fugacities_tv(model, temperature, volume, moles))


def phase_properties(
    model: StateModel,
    state: ChemicalState,
    phase: PhaseKind = "stable",
    *,
    caloric: bool = True,
    standard_state: StandardState | None = None,
) -> PhaseProperties:
    """Evaluate a homogeneous state, without any equilibrium calculation.

    Fugacities are returned in Pa and logarithmic fugacities are
    ``ln(f_i / p_standard)`` with ``p_standard = 1 bar``. Chemical potentials
    and total free energies use a zero ideal-gas standard chemical potential
    at that pressure unless ``standard_state`` is supplied. The corresponding
    dimensionless chemical potential is ``mu_i / (R*T)``; it is the
    thermodynamically meaningful alternative to ``ln(mu_i)``, which is not
    generally defined for a dimensional, reference-dependent quantity that
    may be negative.

    Absolute caloric reference terms are deliberately not invented; returned
    enthalpy and entropy are residual quantities.

    The reduced molar free energies are
    ``reduced_gibbs_energy = g/(R*T)`` and
    ``reduced_helmholtz_energy = a/(R*T)``, with ``a = g - P*v``. The
    reference-independent departures follow
    ``g^R/(R*T) = sum(x_i*ln(phi_i))`` and
    ``a^R/(R*T) = g^R/(R*T) - Z + 1 + ln(Z)``, where
    ``Z = P*v/(R*T)``.

    At an exactly zero mole fraction, the logarithmic component properties
    use the smallest positive value of the tensor dtype as a finite trace
    limit. ``state_derivatives`` instead requires a strictly positive,
    interior composition because logarithmic composition derivatives are
    singular on the simplex boundary.
    """
    temperature = state.temperature
    pressure = state.pressure
    composition = state.composition
    z_factor = model.select_z(temperature, pressure, composition, phase)
    volume = model.molar_volume(temperature, pressure, composition, phase)
    log_phi = model.log_fugacity_coefficients(temperature, pressure, composition, phase)
    log_fugacity = _log_fugacities_from_coefficients(log_phi, pressure, composition)
    fugacity = STANDARD_PRESSURE * torch.exp(log_fugacity)
    chemical_potentials = _chemical_potentials_from_log_fugacities(
        log_fugacity,
        temperature,
        standard_state,
    )
    reduced_chemical_potentials = chemical_potentials / (R * temperature[..., None])
    gibbs = torch.sum(composition * chemical_potentials, dim=-1)
    pressure_volume_over_rt = pressure * volume / (R * temperature)
    helmholtz = gibbs - pressure * volume
    reduced_gibbs = gibbs / (R * temperature)
    reduced_helmholtz = helmholtz / (R * temperature)
    reduced_residual_gibbs = torch.sum(composition * log_phi, dim=-1)
    reduced_residual_helmholtz = (
        reduced_residual_gibbs - pressure_volume_over_rt + 1.0 + torch.log(pressure_volume_over_rt)
    )

    residual_enthalpy: Tensor | None = None
    residual_entropy: Tensor | None = None
    if caloric and temperature.ndim == 0 and composition.ndim == 1:

        def residual_gibbs_over_t(current_temperature: Tensor) -> Tensor:
            current_log_phi = model.log_fugacity_coefficients(
                current_temperature, pressure, composition, phase
            )
            return R * torch.sum(composition * current_log_phi)

        derivative = torch.func.grad(residual_gibbs_over_t)(temperature)
        residual_enthalpy = -temperature.square() * derivative
        residual_gibbs = R * temperature * torch.sum(composition * log_phi)
        residual_entropy = (residual_enthalpy - residual_gibbs) / temperature

    properties = PhaseProperties(
        kind=phase,
        composition=composition,
        compressibility_factor=z_factor,
        molar_volume=volume,
        log_fugacity_coefficients=log_phi,
        fugacities=fugacity,
        log_fugacities=log_fugacity,
        chemical_potentials=chemical_potentials,
        reduced_chemical_potentials=reduced_chemical_potentials,
        molar_gibbs_energy=gibbs,
        molar_helmholtz_energy=helmholtz,
        reduced_gibbs_energy=reduced_gibbs,
        reduced_helmholtz_energy=reduced_helmholtz,
        reduced_residual_gibbs_energy=reduced_residual_gibbs,
        reduced_residual_helmholtz_energy=reduced_residual_helmholtz,
        residual_enthalpy=residual_enthalpy,
        residual_entropy=residual_entropy,
    )
    return replace(
        properties,
        phase_identification=identify_phase_from_properties(model, state, properties),
    )


def state_derivatives(
    model: StateModel,
    state: ChemicalState,
    phase: PhaseKind = "stable",
    *,
    standard_state: StandardState | None = None,
) -> ThermodynamicDerivatives:
    """Autodifferentiate TP state properties in several composition coordinates."""
    if state.temperature.ndim != 0 or state.pressure.ndim != 0:
        raise ValueError("state_derivatives currently accepts one scalar T-P state")
    if state.composition.ndim != 1:
        raise ValueError("state_derivatives currently accepts one composition vector")
    if bool((state.composition <= 0.0).any()):
        raise ValueError("state_derivatives requires strictly positive mole fractions")

    def composition_from_logits(logit_values: Tensor) -> Tensor:
        return torch.softmax(logit_values, dim=-1)

    def composition_from_independent(independent: Tensor) -> Tensor:
        return torch.cat((independent, (1.0 - independent.sum()).reshape(1)))

    def composition_from_moles(moles: Tensor) -> Tensor:
        return moles / moles.sum()

    def log_phi_at(composition: Tensor, temperature: Tensor, pressure: Tensor) -> Tensor:
        return model.log_fugacity_coefficients(
            temperature,
            pressure,
            composition,
            phase,
        )

    def log_fugacity_at(composition: Tensor, temperature: Tensor, pressure: Tensor) -> Tensor:
        return _log_fugacities(
            model,
            temperature,
            pressure,
            composition,
            phase,
        )

    def mu_at(composition: Tensor, temperature: Tensor, pressure: Tensor) -> Tensor:
        return _chemical_potentials(
            model,
            temperature,
            pressure,
            composition,
            phase,
            standard_state,
        )

    def volume_at(composition: Tensor, temperature: Tensor, pressure: Tensor) -> Tensor:
        return model.molar_volume(temperature, pressure, composition, phase)

    logits = torch.log(torch.clamp_min(state.composition, 1.0e-300))
    independent = state.composition[:-1]
    unit_moles = state.composition

    PropertyFunction = Callable[[Tensor, Tensor, Tensor], Tensor]

    def apply_logits(function: PropertyFunction, current: Tensor) -> Tensor:
        return function(
            composition_from_logits(current),
            state.temperature,
            state.pressure,
        )

    def apply_independent(function: PropertyFunction, current: Tensor) -> Tensor:
        return function(
            composition_from_independent(current),
            state.temperature,
            state.pressure,
        )

    def apply_moles(function: PropertyFunction, current: Tensor) -> Tensor:
        return function(
            composition_from_moles(current),
            state.temperature,
            state.pressure,
        )

    dlog_phi_dlogits = torch.func.jacrev(lambda value: apply_logits(log_phi_at, value))(logits)
    dlog_phi_dindependent = torch.func.jacrev(lambda value: apply_independent(log_phi_at, value))(
        independent
    )
    dlog_phi_dmoles = torch.func.jacrev(lambda value: apply_moles(log_phi_at, value))(unit_moles)
    dlog_phi_dt = torch.func.jacrev(lambda t: log_phi_at(state.composition, t, state.pressure))(
        state.temperature
    )
    dlog_phi_dp = torch.func.jacrev(lambda p: log_phi_at(state.composition, state.temperature, p))(
        state.pressure
    )

    dlog_fugacity_dlogits = torch.func.jacrev(lambda value: apply_logits(log_fugacity_at, value))(
        logits
    )
    dlog_fugacity_dindependent = torch.func.jacrev(
        lambda value: apply_independent(log_fugacity_at, value)
    )(independent)
    dlog_fugacity_dmoles = torch.func.jacrev(lambda value: apply_moles(log_fugacity_at, value))(
        unit_moles
    )
    dlog_fugacity_dt = torch.func.jacrev(
        lambda t: log_fugacity_at(state.composition, t, state.pressure)
    )(state.temperature)
    dlog_fugacity_dp = torch.func.jacrev(
        lambda p: log_fugacity_at(state.composition, state.temperature, p)
    )(state.pressure)

    dmu_dlogits = torch.func.jacrev(lambda value: apply_logits(mu_at, value))(logits)
    dmu_dindependent = torch.func.jacrev(lambda value: apply_independent(mu_at, value))(independent)
    dmu_dmoles = torch.func.jacrev(lambda value: apply_moles(mu_at, value))(unit_moles)
    dmu_dt = torch.func.jacrev(lambda t: mu_at(state.composition, t, state.pressure))(
        state.temperature
    )
    dmu_dp = torch.func.jacrev(lambda p: mu_at(state.composition, state.temperature, p))(
        state.pressure
    )

    dvolume_dlogits = torch.func.jacrev(lambda value: apply_logits(volume_at, value))(logits)
    dvolume_dindependent = torch.func.jacrev(lambda value: apply_independent(volume_at, value))(
        independent
    )
    dvolume_dmoles = torch.func.jacrev(lambda value: apply_moles(volume_at, value))(unit_moles)
    dvolume_dt = torch.func.jacrev(lambda t: volume_at(state.composition, t, state.pressure))(
        state.temperature
    )
    dvolume_dp = torch.func.jacrev(lambda p: volume_at(state.composition, state.temperature, p))(
        state.pressure
    )

    def gibbs(t: Tensor, p: Tensor) -> Tensor:
        mu = mu_at(state.composition, t, p)
        return torch.sum(state.composition * mu)

    dg_dt = torch.func.grad(lambda t: gibbs(t, state.pressure))(state.temperature)
    dg_dp = torch.func.grad(lambda p: gibbs(state.temperature, p))(state.pressure)

    log_fugacity = _log_fugacities(
        model,
        state.temperature,
        state.pressure,
        state.composition,
        phase,
    )
    fugacity = STANDARD_PRESSURE * torch.exp(log_fugacity)
    log_phi = log_phi_at(state.composition, state.temperature, state.pressure)
    phi = torch.exp(log_phi)
    dphi_dlogits = phi[:, None] * dlog_phi_dlogits
    dphi_dindependent = phi[:, None] * dlog_phi_dindependent
    dphi_dmoles = phi[:, None] * dlog_phi_dmoles
    dphi_dt = phi * dlog_phi_dt
    dphi_dp = phi * dlog_phi_dp
    dfugacity_dlogits = fugacity[:, None] * dlog_fugacity_dlogits
    dfugacity_dindependent = fugacity[:, None] * dlog_fugacity_dindependent
    dfugacity_dmoles = fugacity[:, None] * dlog_fugacity_dmoles
    dfugacity_dt = fugacity * dlog_fugacity_dt
    dfugacity_dp = fugacity * dlog_fugacity_dp

    chemical_potential = _chemical_potentials(
        model,
        state.temperature,
        state.pressure,
        state.composition,
        phase,
        standard_state,
    )
    rt = R * state.temperature
    dreduced_mu_dlogits = dmu_dlogits / rt
    dreduced_mu_dindependent = dmu_dindependent / rt
    dreduced_mu_dmoles = dmu_dmoles / rt
    dreduced_mu_dt = dmu_dt / rt - chemical_potential / (R * state.temperature.square())
    dreduced_mu_dp = dmu_dp / rt

    return ThermodynamicDerivatives(
        dfugacity_coefficient_dlogits=dphi_dlogits,
        dfugacity_coefficient_dindependent_composition=dphi_dindependent,
        dfugacity_coefficient_dtemperature=dphi_dt,
        dfugacity_coefficient_dpressure=dphi_dp,
        dfugacity_coefficient_dmoles=dphi_dmoles,
        dlog_fugacity_coefficient_dlogits=dlog_phi_dlogits,
        dlog_fugacity_coefficient_dindependent_composition=dlog_phi_dindependent,
        dlog_fugacity_coefficient_dtemperature=dlog_phi_dt,
        dlog_fugacity_coefficient_dpressure=dlog_phi_dp,
        dlog_fugacity_coefficient_dmoles=dlog_phi_dmoles,
        dfugacity_dlogits=dfugacity_dlogits,
        dfugacity_dindependent_composition=dfugacity_dindependent,
        dfugacity_dtemperature=dfugacity_dt,
        dfugacity_dpressure=dfugacity_dp,
        dfugacity_dmoles=dfugacity_dmoles,
        dlog_fugacity_dlogits=dlog_fugacity_dlogits,
        dlog_fugacity_dindependent_composition=dlog_fugacity_dindependent,
        dlog_fugacity_dtemperature=dlog_fugacity_dt,
        dlog_fugacity_dpressure=dlog_fugacity_dp,
        dlog_fugacity_dmoles=dlog_fugacity_dmoles,
        dchemical_potential_dlogits=dmu_dlogits,
        dchemical_potential_dindependent_composition=dmu_dindependent,
        dchemical_potential_dtemperature=dmu_dt,
        dchemical_potential_dpressure=dmu_dp,
        dchemical_potential_dmoles=dmu_dmoles,
        dreduced_chemical_potential_dlogits=dreduced_mu_dlogits,
        dreduced_chemical_potential_dindependent_composition=dreduced_mu_dindependent,
        dreduced_chemical_potential_dtemperature=dreduced_mu_dt,
        dreduced_chemical_potential_dpressure=dreduced_mu_dp,
        dreduced_chemical_potential_dmoles=dreduced_mu_dmoles,
        dmolar_volume_dlogits=dvolume_dlogits,
        dmolar_volume_dindependent_composition=dvolume_dindependent,
        dmolar_volume_dtemperature=dvolume_dt,
        dmolar_volume_dpressure=dvolume_dp,
        dmolar_volume_dmoles=dvolume_dmoles,
        dgibbs_dtemperature=dg_dt,
        dgibbs_dpressure=dg_dp,
    )
