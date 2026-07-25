"""Stability-tested isothermal two-phase flash.

The phase-split strategy follows M. L. Michelsen, "The isothermal flash
problem. Part II. Phase-split calculation", *Fluid Phase Equilibria* 9
(1982), 21-40, doi:10.1016/0378-3812(82)85002-4. Preliminary stability uses
Michelsen Part I, doi:10.1016/0378-3812(82)85001-2.
"""

from __future__ import annotations

import warnings
from typing import cast

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.exceptions import ConvergenceWarning
from torch_flash.initialization import wilson_k_values
from torch_flash.material_balance import rachford_rice
from torch_flash.properties import identify_flash_phases, phase_properties
from torch_flash.properties.state import StateModel
from torch_flash.types import ChemicalState, FlashResult

from .stability import tangent_plane_stability


def _model_components(model: StateModel) -> ComponentSet:
    required = (
        "critical_temperature",
        "critical_pressure",
        "acentric_factor",
        "molar_mass",
        "names",
    )
    if not all(hasattr(model, item) for item in required):
        raise ValueError("initial K values are required for a model without critical constants")
    return cast(ComponentSet, model)


def two_phase_flash(
    model: StateModel,
    state: ChemicalState,
    *,
    initial_k_values: Tensor | None = None,
    check_stability: bool = True,
    tolerance: float = 1.0e-8,
    max_iterations: int = 100,
    raise_on_failure: bool = False,
) -> FlashResult:
    """Solve a scalar isothermal-isobaric two-phase flash.

    The solver optionally applies Michelsen tangent-plane stability analysis,
    solves the material balance with the safeguarded Rachford--Rice solver,
    and enforces equality of component fugacities using successive
    substitution followed by autodiff-assembled Newton steps.

    Parameters
    ----------
    model
        Homogeneous-state model providing phase fugacity coefficients and
        properties.
    state
        Feed state with temperature in K, pressure in Pa, and one
        mole-fraction vector of shape ``(ncomponents,)``.
    initial_k_values
        Positive initial vapor-to-liquid equilibrium ratios with shape
        ``(ncomponents,)``. Wilson estimates are used when omitted, which
        requires critical constants and acentric factors on ``model``.
    check_stability
        Run tangent-plane stability analysis before attempting a split. A
        stable feed returns a one-phase result.
    tolerance
        Convergence threshold for the maximum absolute log-fugacity residual.
    max_iterations
        Maximum phase-equilibrium iterations after initialization.
    raise_on_failure
        Raise ``RuntimeError`` instead of emitting ``ConvergenceWarning`` when
        the phase-equilibrium iterations do not converge.

    Returns
    -------
    FlashResult
        Phase fractions, identified phase properties, convergence status,
        iteration count, and residual diagnostics. For a two-phase result,
        fractions are ordered liquid then vapor before phase identification.

    Raises
    ------
    ValueError
        If the composition is batched, suitable initial K values cannot be
        constructed, or no finite two-phase material-balance root is found.
    RuntimeError
        If convergence fails and ``raise_on_failure`` is true.

    Warns
    -----
    ConvergenceWarning
        If the requested split does not converge and ``raise_on_failure`` is
        false. The returned result is explicitly marked non-converged.

    Notes
    -----
    The reported equilibrium residual is
    ``max_i |log(K_i) - (log(phi_i^L) - log(phi_i^V))|``. A converged
    Rachford--Rice solve is necessary but is not by itself evidence of phase
    equilibrium.
    """
    if state.composition.ndim != 1:
        raise ValueError("two_phase_flash currently accepts one composition vector")
    z = state.composition
    if check_stability:
        stability = tangent_plane_stability(model, state)
        if stability.stable:
            phase = phase_properties(model, state, "stable")
            return FlashResult(
                torch.ones(1, dtype=z.dtype, device=z.device),
                (phase,),
                stability.converged,
                stability.iterations,
                torch.clamp_min(stability.minimum_tpd, 0.0),
                True,
                {"minimum_tpd": float(stability.minimum_tpd)},
            )

    if initial_k_values is None:
        initial_k_values = wilson_k_values(
            _model_components(model), state.temperature, state.pressure
        )
    log_k = torch.log(torch.clamp_min(initial_k_values, 1.0e-30))
    converged = False
    residual_norm = torch.tensor(torch.inf, dtype=z.dtype, device=z.device)
    rr = None

    def equilibrium_residual(current_log_k: Tensor) -> Tensor:
        current_k = torch.exp(current_log_k)
        split = rachford_rice(z, current_k, tolerance=1.0e-13)
        x = split.liquid_composition
        y = split.vapor_composition
        log_phi_l = model.log_fugacity_coefficients(state.temperature, state.pressure, x, "liquid")
        log_phi_v = model.log_fugacity_coefficients(state.temperature, state.pressure, y, "vapor")
        return current_log_k - (log_phi_l - log_phi_v)

    for iteration in range(1, max_iterations + 1):
        k = torch.exp(log_k)
        try:
            rr = rachford_rice(z, k, tolerance=1.0e-13)
        except ValueError:
            # Pull non-straddling Wilson estimates toward a neutral split.
            log_k = log_k - torch.sum(z * log_k)
            continue
        x = rr.liquid_composition
        y = rr.vapor_composition
        log_phi_l = model.log_fugacity_coefficients(state.temperature, state.pressure, x, "liquid")
        log_phi_v = model.log_fugacity_coefficients(state.temperature, state.pressure, y, "vapor")
        residual = log_k - (log_phi_l - log_phi_v)
        residual_norm = residual.abs().max()
        if float(residual_norm) <= tolerance:
            converged = True
            break

        if iteration <= 12:
            damping = 0.8 if iteration < 5 else 0.5
            log_k = log_k - damping * residual
            continue

        jacobian = torch.func.jacrev(equilibrium_residual)(log_k)
        eye = torch.eye(log_k.numel(), dtype=log_k.dtype, device=log_k.device)
        try:
            step = torch.linalg.solve(jacobian + 1.0e-10 * eye, -residual)
        except torch.linalg.LinAlgError:
            step = -0.25 * residual
        accepted = False
        factor = 1.0
        for _ in range(12):
            candidate = log_k + factor * step
            try:
                candidate_residual = equilibrium_residual(candidate)
            except ValueError:
                factor *= 0.5
                continue
            if float(candidate_residual.abs().max()) < float(residual_norm):
                log_k = candidate
                accepted = True
                break
            factor *= 0.5
        if not accepted:
            log_k = log_k - 0.2 * residual

    if rr is None:
        raise ValueError("could not construct a two-phase material-balance split")
    final_k = torch.exp(log_k)
    rr = rachford_rice(z, final_k, tolerance=1.0e-13)
    liquid_state = ChemicalState(state.temperature, state.pressure, rr.liquid_composition)
    vapor_state = ChemicalState(state.temperature, state.pressure, rr.vapor_composition)
    liquid = phase_properties(model, liquid_state, "liquid", caloric=False)
    vapor = phase_properties(model, vapor_state, "vapor", caloric=False)
    if not converged:
        message = (
            f"two-phase flash did not converge in {max_iterations} iterations "
            f"(log-fugacity residual {float(residual_norm):.3e})"
        )
        if raise_on_failure:
            raise RuntimeError(message)
        warnings.warn(message, ConvergenceWarning, stacklevel=2)
    fractions = torch.stack((rr.liquid_fraction, rr.vapor_fraction))
    identified_phases = identify_flash_phases((liquid, vapor))
    return FlashResult(
        fractions,
        identified_phases,
        converged,
        iteration,
        residual_norm,
        converged,
        {"rachford_rice_iterations": rr.iterations},
    )
