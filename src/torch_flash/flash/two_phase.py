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
from torch_flash.solvers import minimize_dense_trust_region
from torch_flash.types import ChemicalState, FlashResult, PhaseKind

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


def two_phase_trust_region_flash(
    model: StateModel,
    state: ChemicalState,
    *,
    initial_k_values: Tensor | None = None,
    phase_roots: tuple[PhaseKind, PhaseKind] = ("liquid", "vapor"),
    check_stability: bool = True,
    tolerance: float = 1.0e-8,
    max_iterations: int = 100,
    raise_on_failure: bool = False,
) -> FlashResult:
    """Minimize two-phase Gibbs energy with an exact dense trust region.

    The formulation follows M. Petitfrere and D. V. Nichita, "Robust and
    efficient Trust-Region based stability analysis and multiphase flash
    calculations", *Fluid Phase Equilibria* 362 (2014), 51-68, equations
    (8)-(15) and sections 3.1-3.4,
    doi:10.1016/j.fluid.2013.08.039. Direct component mole amounts in the
    initially smaller phase are independent variables; the corresponding
    amounts in the reference phase are dependent, so component material
    balance is exact at every accepted step. PyTorch autodiff supplies the
    exact gradient and Hessian.

    Parameters
    ----------
    model
        Homogeneous-state fugacity model.
    state
        Scalar temperature in K, pressure in Pa, and one feed-composition
        vector.
    initial_k_values
        Optional positive vapor-to-liquid ratios. Wilson values are used when
        omitted.
    phase_roots
        Root requests for the two candidate phases.
    check_stability
        Run trust-region tangent-plane stability first. A stable feed returns
        one homogeneous phase without minimizing a split.
    tolerance
        Maximum dimensionless log-fugacity residual and trust-region gradient
        tolerance.
    max_iterations
        Maximum trust-region iterations.
    raise_on_failure
        Raise ``RuntimeError`` rather than emitting ``ConvergenceWarning`` for
        a non-converged or phase-collapsed result.

    Returns
    -------
    FlashResult
        Phase fractions, properties, convergence flag, fugacity residual, and
        trust-region diagnostics.

    Raises
    ------
    ValueError
        If state shapes, phase roots, iteration controls, or initial ratios
        are invalid.
    RuntimeError
        If the split fails and ``raise_on_failure`` is true.

    Warns
    -----
    ConvergenceWarning
        If the trust-region stationary point does not also pass fugacity and
        phase-fraction gates.

    Notes
    -----
    This is a local two-phase minimization. It does not replace multistart
    stability analysis or prove that no third phase has lower Gibbs energy.
    The default :func:`two_phase_flash` remains preferable for ordinary
    well-conditioned vapor-liquid states unless matched benchmarks establish
    a trust-region advantage.
    """
    if state.composition.ndim != 1:
        raise ValueError("two-phase trust-region flash accepts one composition vector")
    if len(phase_roots) != 2 or any(
        phase not in ("liquid", "vapor", "stable") for phase in phase_roots
    ):
        raise ValueError("phase_roots must contain two valid phase-root requests")
    if tolerance <= 0.0 or max_iterations <= 0:
        raise ValueError("two-phase trust-region controls must be positive")
    z = state.composition / state.composition.sum()
    if not bool(torch.isfinite(z).all() & (z > 0.0).all()):
        raise ValueError(
            "two-phase trust-region flash requires finite strictly positive feed fractions"
        )
    if check_stability:
        stability = tangent_plane_stability(
            model,
            state,
            minimizer="trust-region",
            tolerance=min(tolerance, 1.0e-9),
            max_iterations=max_iterations,
        )
        if stability.stable and stability.converged:
            phase = phase_properties(model, state, "stable")
            return FlashResult(
                torch.ones(1, dtype=z.dtype, device=z.device),
                (phase,),
                True,
                stability.iterations,
                torch.clamp_min(stability.minimum_tpd, 0.0),
                True,
                {
                    "minimum_tpd": float(stability.minimum_tpd),
                    "trust_region_stability": True,
                },
            )

    if initial_k_values is None:
        initial_k_values = wilson_k_values(
            _model_components(model),
            state.temperature,
            state.pressure,
        )
    if initial_k_values.shape != z.shape or not bool(
        torch.isfinite(initial_k_values).all() & (initial_k_values > 0.0).all()
    ):
        raise ValueError("initial K values must be a finite positive component vector")
    log_k = torch.log(initial_k_values)
    log_k = log_k - torch.sum(z * log_k)
    split = rachford_rice(z, torch.exp(log_k), tolerance=1.0e-13)
    first_initial_moles = split.liquid_fraction * split.liquid_composition
    second_initial_moles = split.vapor_fraction * split.vapor_composition
    independent_is_first = first_initial_moles <= second_initial_moles
    initial = torch.where(
        independent_is_first,
        first_initial_moles,
        second_initial_moles,
    )
    mole_floor = 16.0 * torch.finfo(z.dtype).eps * z
    initial = torch.minimum(
        torch.maximum(initial, mole_floor),
        z - mole_floor,
    )

    def quantities(independent_moles: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        first_moles = torch.where(
            independent_is_first,
            independent_moles,
            z - independent_moles,
        )
        second_moles = z - first_moles
        phase_moles = torch.stack((first_moles, second_moles))
        fractions = phase_moles.sum(dim=-1)
        compositions = phase_moles / fractions[:, None]
        log_phi = torch.stack(
            tuple(
                model.log_fugacity_coefficients(
                    state.temperature,
                    state.pressure,
                    composition,
                    phase_kind,
                )
                for composition, phase_kind in zip(
                    compositions,
                    phase_roots,
                    strict=True,
                )
            )
        )
        chemical_potential = torch.log(compositions) + log_phi
        gibbs = torch.sum(phase_moles * chemical_potential)
        return gibbs, fractions, compositions

    result = minimize_dense_trust_region(
        lambda independent_moles: quantities(independent_moles)[0],
        initial,
        is_feasible=lambda independent_moles: bool(
            (
                torch.isfinite(independent_moles)
                & (independent_moles > mole_floor)
                & (independent_moles < z - mole_floor)
            )
            .detach()
            .all()
        ),
        gradient_tolerance=tolerance,
        max_iterations=max_iterations,
    )
    _, fractions, compositions = quantities(result.solution)
    log_phi = torch.stack(
        tuple(
            model.log_fugacity_coefficients(
                state.temperature,
                state.pressure,
                composition,
                phase_kind,
            )
            for composition, phase_kind in zip(
                compositions,
                phase_roots,
                strict=True,
            )
        )
    )
    chemical_potential = torch.log(compositions) + log_phi
    residual_norm = (chemical_potential[1] - chemical_potential[0]).abs().max()
    phase_fraction_tolerance = max(10.0 * tolerance, 1.0e-10)
    physical = bool((fractions.detach() > phase_fraction_tolerance).all())
    converged = result.converged and physical and bool(residual_norm.detach() <= tolerance)
    phases = tuple(
        phase_properties(
            model,
            ChemicalState(state.temperature, state.pressure, composition),
            phase_kind,
            caloric=False,
        )
        for composition, phase_kind in zip(
            compositions,
            phase_roots,
            strict=True,
        )
    )
    if not converged:
        message = (
            "two-phase trust-region flash did not produce a physical "
            f"residual-converged split in {result.iterations} iterations "
            f"(log-fugacity residual {float(residual_norm):.3e})"
        )
        if raise_on_failure:
            raise RuntimeError(message)
        warnings.warn(message, ConvergenceWarning, stacklevel=2)
    return FlashResult(
        fractions,
        identify_flash_phases(phases),
        converged,
        result.iterations,
        residual_norm,
        converged,
        {
            "trust_region_accepted_steps": result.accepted_steps,
            "trust_region_rejected_steps": result.rejected_steps,
            "trust_region_gradient_norm": float(result.gradient_norm),
            "trust_region_minimum_hessian_eigenvalue": float(result.minimum_hessian_eigenvalue),
        },
    )
