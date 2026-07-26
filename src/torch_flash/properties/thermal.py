"""Autodifferentiable caloric and thermal properties.

The definitions follow Chapter 8 of Pedersen, *Phase Behavior of Petroleum
Reservoir Fluids*, 3rd ed. (2024), doi:10.1201/9780429457418. Temperature and
pressure derivatives are evaluated by PyTorch autodiff instead of coding
model-specific analytic derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from torch_flash.constants import STANDARD_PRESSURE, R
from torch_flash.exceptions import InvalidStateError
from torch_flash.properties.state import StateModel
from torch_flash.types import ChemicalState, PhaseKind, normalize_composition


class CaloricStandardState(Protocol):
    """Ideal-gas caloric functions required by Chapter 8 properties."""

    def heat_capacity(self, temperature: Tensor) -> Tensor:
        """Return component ideal-gas heat capacities in J/(mol K)."""

    def enthalpy(self, temperature: Tensor) -> Tensor:
        """Return component ideal-gas enthalpies in J/mol."""

    def entropy(self, temperature: Tensor) -> Tensor:
        """Return component ideal-gas entropies in J/(mol K)."""


@dataclass(frozen=True)
class ThermalProperties:
    """Caloric and response properties of one homogeneous phase.

    All quantities are molar SI except the Joule-Thomson coefficient (K/Pa)
    and speed of sound (m/s). ``reduced_*`` free energies are dimensionless
    molar quantities divided by ``R*T``.

    Attributes
    ----------
    molar_enthalpy, molar_internal_energy
        Total energies in J/mol.
    molar_entropy
        Total entropy in J/(mol K).
    molar_helmholtz_energy, molar_gibbs_energy
        Total free energies in J/mol.
    reduced_helmholtz_energy, reduced_gibbs_energy
        Dimensionless total free energies divided by ``RT``.
    reduced_residual_helmholtz_energy, reduced_residual_gibbs_energy
        Dimensionless EOS departure free energies.
    isobaric_heat_capacity, isochoric_heat_capacity
        ``C_p`` and ``C_v`` in J/(mol K).
    joule_thomson_coefficient
        Isenthalpic pressure response in K/Pa.
    speed_of_sound
        Speed in m/s, or ``None`` when molar mass is unavailable.
    residual_enthalpy
        EOS departure enthalpy in J/mol.
    residual_entropy
        EOS departure entropy in J/(mol K).
    """

    molar_enthalpy: Tensor
    molar_internal_energy: Tensor
    molar_entropy: Tensor
    molar_helmholtz_energy: Tensor
    molar_gibbs_energy: Tensor
    reduced_helmholtz_energy: Tensor
    reduced_gibbs_energy: Tensor
    reduced_residual_helmholtz_energy: Tensor
    reduced_residual_gibbs_energy: Tensor
    isobaric_heat_capacity: Tensor
    isochoric_heat_capacity: Tensor
    joule_thomson_coefficient: Tensor
    speed_of_sound: Tensor | None
    residual_enthalpy: Tensor
    residual_entropy: Tensor


def _weighted_log_fugacity(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    composition: Tensor,
    phase: PhaseKind,
) -> Tensor:
    log_phi = model.log_fugacity_coefficients(temperature, pressure, composition, phase)
    return torch.sum(composition * log_phi, dim=-1)


def _residual_terms(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    composition: Tensor,
    phase: PhaseKind,
) -> tuple[Tensor, Tensor]:
    weighted_log_phi = _weighted_log_fugacity(
        model,
        temperature,
        pressure,
        composition,
        phase,
    )
    # Temperature is scalar while the weighted fugacity may contain a large
    # state batch. Forward-mode AD therefore propagates one input tangent and
    # avoids the output-sized reverse sweeps required by a full Jacobian.
    derivative = torch.func.jacfwd(
        lambda current_temperature: _weighted_log_fugacity(
            model,
            current_temperature,
            pressure,
            composition,
            phase,
        )
    )(temperature)
    # A thermodynamically consistent translated cubic already carries
    # P*c_i/(R*T) in ln(phi_i). Its temperature derivative therefore supplies
    # the P*c_mix enthalpy shift exactly once.
    residual_enthalpy = -R * temperature.square() * derivative
    residual_entropy = residual_enthalpy / temperature - R * weighted_log_phi
    return residual_enthalpy, residual_entropy


def _enthalpy(
    model: StateModel,
    standard_state: CaloricStandardState,
    temperature: Tensor,
    pressure: Tensor,
    composition: Tensor,
    phase: PhaseKind,
) -> Tensor:
    residual_enthalpy, _ = _residual_terms(
        model,
        temperature,
        pressure,
        composition,
        phase,
    )
    return torch.sum(composition * standard_state.enthalpy(temperature)) + residual_enthalpy


def molar_enthalpy_of_mixing(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    composition: Tensor,
    phase: PhaseKind = "stable",
) -> Tensor:
    r"""Return the isothermal-isobaric molar enthalpy change on mixing.

    Parameters
    ----------
    model
        Homogeneous-state model used for the mixture and pure-component
        reference states.
    temperature, pressure
        Finite positive scalar temperature in K and pressure in Pa.
    composition
        Nonnegative mole fractions with components on the final axis. Leading
        dimensions are evaluated as a batch of independent compositions at
        the supplied temperature and pressure.
    phase
        Root-selection request applied independently to the mixture and every
        pure-component reference state.

    Returns
    -------
    Tensor
        Molar enthalpy of mixing in J/mol. The result has the leading shape of
        ``composition`` and is scalar for a single composition,

        .. math::

            h^\mathrm{M}(T,P,\mathbf{x})
            = h(T,P,\mathbf{x})
            - \sum_i x_i h_i(T,P).

    Raises
    ------
    ValueError
        If temperature, pressure, composition, or root selection is invalid.

    Notes
    -----
    Ideal-gas component enthalpies cancel exactly between the mixture and its
    unmixed pure-component reference states. The implementation therefore
    evaluates only residual enthalpies from

    .. math::

        h^\mathrm{R}
        = -R T^2
          \left(\frac{\partial \sum_i x_i \ln\phi_i}
          {\partial T}\right)_{P,\mathbf{x}},

    with the temperature derivative obtained by PyTorch automatic
    differentiation. This preserves derivatives with respect to equation-of-
    state parameters and avoids requiring an arbitrary ideal-gas caloric
    reference.

    The definition matches the fixed-temperature, fixed-pressure enthalpy
    change on mixing used to parameterize E-PPR78 by Xu, Lasala, Privat, and
    Jaubert, *International Journal of Greenhouse Gas Control* 56 (2017)
    126-154, doi:10.1016/j.ijggc.2016.11.015. The requested homogeneous root
    must represent the experimental single-phase state; this function does
    not perform a flash or silently average phase enthalpies inside a
    two-phase region.
    """
    if phase not in ("liquid", "vapor", "stable"):
        raise ValueError("enthalpy of mixing phase must be 'liquid', 'vapor', or 'stable'")
    if temperature.ndim != 0 or not bool(torch.isfinite(temperature) & (temperature > 0.0)):
        raise ValueError("enthalpy of mixing requires one finite positive temperature")
    if pressure.ndim != 0 or not bool(torch.isfinite(pressure) & (pressure > 0.0)):
        raise ValueError("enthalpy of mixing requires one finite positive pressure")
    if not composition.is_floating_point() or composition.ndim < 1:
        raise ValueError(
            "enthalpy of mixing composition must be a floating tensor with a final component axis"
        )
    if not bool(torch.isfinite(composition).all() & (composition >= 0.0).all()):
        raise ValueError("enthalpy of mixing composition must be finite and nonnegative")

    normalized = normalize_composition(composition)
    component_count = normalized.shape[-1]
    if component_count < 2:
        raise ValueError("enthalpy of mixing requires at least two components")
    solve_temperature = temperature.to(dtype=normalized.dtype, device=normalized.device)
    solve_pressure = pressure.to(dtype=normalized.dtype, device=normalized.device)
    mixture_enthalpy, _ = _residual_terms(
        model,
        solve_temperature,
        solve_pressure,
        normalized,
        phase,
    )
    pure_compositions = torch.eye(
        component_count,
        dtype=normalized.dtype,
        device=normalized.device,
    )
    pure_enthalpies, _ = _residual_terms(
        model,
        solve_temperature,
        solve_pressure,
        pure_compositions,
        phase,
    )
    return mixture_enthalpy - torch.sum(normalized * pure_enthalpies, dim=-1)


def _mixture_molar_mass(
    model: StateModel,
    composition: Tensor,
    supplied: Tensor | None,
) -> Tensor | None:
    value = getattr(model, "molar_mass", None) if supplied is None else supplied
    if not isinstance(value, Tensor):
        return None
    if value.ndim == 0:
        mixture = value
    elif value.shape == composition.shape:
        mixture = torch.sum(composition * value)
    else:
        raise ValueError("molar_mass must be scalar or have one value per component")
    if bool((~torch.isfinite(mixture) | (mixture <= 0.0)).detach()):
        raise ValueError("molar_mass must be finite and positive")
    return mixture


def thermal_properties(
    model: StateModel,
    state: ChemicalState,
    standard_state: CaloricStandardState,
    phase: PhaseKind = "stable",
    *,
    reference_pressure: float = STANDARD_PRESSURE,
    molar_mass: Tensor | None = None,
) -> ThermalProperties:
    """Evaluate Pedersen Chapter 8 properties at a specified state.

    Parameters
    ----------
    model
        Homogeneous-state model.
    state
        One scalar TP state with temperature in K, pressure in Pa, and one
        composition vector.
    standard_state
        Component ideal-gas heat capacity, enthalpy, and entropy functions.
    phase
        Root-selection request held fixed during differentiation.
    reference_pressure
        Positive ideal-gas entropy reference pressure in Pa.
    molar_mass
        Optional scalar mixture or component molar masses in kg/mol. When
        omitted, the model's ``molar_mass`` attribute is used if available.

    Returns
    -------
    ThermalProperties
        Caloric energies, response functions, residual terms, and optional
        speed of sound in SI units.

    Raises
    ------
    ValueError
        If state shapes, reference pressure, standard-state component count,
        or molar-mass data are invalid.
    InvalidStateError
        If response functions violate positive heat-capacity or mechanical
        stability requirements.

    Notes
    -----
    No phase-equilibrium calculation is performed. The requested phase root
    remains fixed while differentiating. A volume-translated cubic receives
    Pedersen's ``P*DeltaV`` enthalpy correction (Eq. 8.12).

    The entropy pressure term in Eq. 8.17 is interpreted as
    ``-R*ln(P/Pref)``; the glyph in the printed 2024 edition can resemble
    ``T``, but dimensional consistency and the ideal-gas identity require
    the gas constant. Helmholtz energy follows ``a = g - P*v``. Reduced total
    free energies use the supplied caloric reference, while reduced residual
    free energies are reference-independent EoS departures.
    """
    if state.temperature.ndim != 0 or state.pressure.ndim != 0:
        raise ValueError("thermal_properties currently accepts one scalar T-P state")
    if state.composition.ndim != 1:
        raise ValueError("thermal_properties currently accepts one composition vector")
    if reference_pressure <= 0.0:
        raise ValueError("reference_pressure must be positive")

    temperature = state.temperature
    pressure = state.pressure
    composition = state.composition
    ideal_enthalpy = standard_state.enthalpy(temperature)
    ideal_entropy = standard_state.entropy(temperature)
    if ideal_enthalpy.shape != composition.shape or ideal_entropy.shape != composition.shape:
        raise ValueError("standard-state component count must match the composition")

    residual_enthalpy, residual_entropy = _residual_terms(
        model,
        temperature,
        pressure,
        composition,
        phase,
    )
    enthalpy = torch.sum(composition * ideal_enthalpy) + residual_enthalpy
    safe_composition = torch.clamp_min(composition, torch.finfo(composition.dtype).tiny)
    entropy = (
        torch.sum(composition * ideal_entropy)
        - R * torch.log(pressure / reference_pressure)
        - R * torch.sum(composition * torch.log(safe_composition))
        + residual_entropy
    )
    volume = model.molar_volume(temperature, pressure, composition, phase)
    internal_energy = enthalpy - pressure * volume
    gibbs = enthalpy - temperature * entropy
    helmholtz = gibbs - pressure * volume
    reduced_gibbs = gibbs / (R * temperature)
    reduced_helmholtz = helmholtz / (R * temperature)
    pressure_volume_over_rt = pressure * volume / (R * temperature)
    reduced_residual_gibbs = _weighted_log_fugacity(
        model,
        temperature,
        pressure,
        composition,
        phase,
    )
    reduced_residual_helmholtz = (
        reduced_residual_gibbs - pressure_volume_over_rt + 1.0 + torch.log(pressure_volume_over_rt)
    )

    enthalpy_at_temperature = lambda current_temperature: _enthalpy(  # noqa: E731
        model,
        standard_state,
        current_temperature,
        pressure,
        composition,
        phase,
    )
    enthalpy_at_pressure = lambda current_pressure: _enthalpy(  # noqa: E731
        model,
        standard_state,
        temperature,
        current_pressure,
        composition,
        phase,
    )
    cp = torch.func.grad(enthalpy_at_temperature)(temperature)
    dh_dp = torch.func.grad(enthalpy_at_pressure)(pressure)
    joule_thomson = -dh_dp / cp

    volume_at_temperature = lambda current_temperature: model.molar_volume(  # noqa: E731
        current_temperature,
        pressure,
        composition,
        phase,
    )
    volume_at_pressure = lambda current_pressure: model.molar_volume(  # noqa: E731
        temperature,
        current_pressure,
        composition,
        phase,
    )
    dv_dt = torch.func.grad(volume_at_temperature)(temperature)
    dv_dp = torch.func.grad(volume_at_pressure)(pressure)
    cv = cp + temperature * dv_dt.square() / dv_dp

    response_values = torch.stack((cp, cv, dv_dp))
    if bool(
        (
            (~torch.isfinite(response_values)).any() | (cp <= 0.0) | (cv <= 0.0) | (dv_dp >= 0.0)
        ).detach()
    ):
        raise InvalidStateError(
            "thermal response functions require positive Cp/Cv and "
            "negative isothermal compressibility"
        )

    mixture_molar_mass = _mixture_molar_mass(model, composition, molar_mass)
    speed_of_sound = None
    if mixture_molar_mass is not None:
        speed_squared = -volume.square() * cp / (mixture_molar_mass * cv * dv_dp)
        if bool((~torch.isfinite(speed_squared) | (speed_squared <= 0.0)).detach()):
            raise InvalidStateError("speed-of-sound expression is not positive and finite")
        speed_of_sound = torch.sqrt(speed_squared)

    return ThermalProperties(
        enthalpy,
        internal_energy,
        entropy,
        helmholtz,
        gibbs,
        reduced_helmholtz,
        reduced_gibbs,
        reduced_residual_helmholtz,
        reduced_residual_gibbs,
        cp,
        cv,
        joule_thomson,
        speed_of_sound,
        residual_enthalpy,
        residual_entropy,
    )
