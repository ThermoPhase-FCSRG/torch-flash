"""High-level phase-aware calibration workflows for heavy-oil CSP viscosity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from torch_flash.components import ComponentSet
from torch_flash.envelope import saturation_point
from torch_flash.exceptions import ConvergenceError, InvalidStateError
from torch_flash.fitting import FitResult, OptimizerKind, fit_parameters
from torch_flash.flash import batched_two_phase_flash
from torch_flash.properties.state import StateModel
from torch_flash.types import ChemicalState, normalize_composition

from .viscosity import heavy_oil_corresponding_states_viscosity


@dataclass(frozen=True)
class HeavyOilCSPProfile:
    """Phase-aware heavy-oil viscosity states and numerical diagnostics.

    Attributes
    ----------
    viscosity:
        Dynamic viscosity in Pa s, one value per requested state.
    liquid_composition:
        Feed composition above the bubble point and equilibrium liquid
        composition below it, with shape ``(states, components)``.
    bubble_pressure:
        Bubble pressure in Pa at the corresponding state temperature.
    vapor_fraction:
        Equilibrium vapor fraction below the bubble point and zero above it.
    converged:
        Per-state phase-boundary and sub-bubble flash convergence flag.
    residual_norm:
        Maximum dimensionless log-fugacity residual for each sub-bubble
        flash; zero for homogeneous states above the bubble point.
    """

    viscosity: Tensor
    liquid_composition: Tensor
    bubble_pressure: Tensor
    vapor_fraction: Tensor
    converged: Tensor
    residual_norm: Tensor


@dataclass(frozen=True)
class HeavyOilCSPCalibrationResult:
    """Joint bounded calibration of the two Lindeloff CSP factors.

    Attributes
    ----------
    third_csp, fourth_csp:
        Selected positive dimensionless factors in Pedersen et al. (2024),
        Eqs. 10.34-10.35.
    prediction:
        Fitted dynamic viscosities in Pa s at every calibration state.
    fit:
        PyTorch optimizer history and stopping diagnostics.
    sensitivity_matrix:
        Jacobian of log viscosity with respect to the two physical factors,
        with shape ``(states, 2)``.
    sensitivity_singular_values:
        Singular values of the sensitivity matrix.
    sensitivity_rank:
        Dtype-aware numerical rank.
    sensitivity_condition_number:
        Largest-to-smallest singular-value ratio, or infinity for a
        rank-deficient calibration.

    Notes
    -----
    Agreement with the supplied observations is calibration evidence, not
    independent validation.
    """

    third_csp: Tensor
    fourth_csp: Tensor
    prediction: Tensor
    fit: FitResult
    sensitivity_matrix: Tensor
    sensitivity_singular_values: Tensor
    sensitivity_rank: int
    sensitivity_condition_number: float


def evaluate_heavy_oil_corresponding_states_profile(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    feed_composition: Tensor,
    components: ComponentSet,
    *,
    third_csp: Tensor | float = 1.0,
    fourth_csp: Tensor | float = 1.0,
    initial_bubble_pressure: Tensor | float | None = None,
    tolerance: float = 1.0e-8,
    saturation_iterations: int = 50,
    substitution_iterations: int = 20,
    newton_iterations: int = 20,
    raise_on_failure: bool = True,
) -> HeavyOilCSPProfile:
    """Evaluate a phase-aware live-heavy-oil viscosity pressure profile.

    A liquid-vapor bubble point is solved once per unique temperature. States
    at or above that boundary retain the feed composition. All sub-bubble
    states at the same temperature are flashed simultaneously with
    :func:`torch_flash.batched_two_phase_flash`, and their equilibrium liquid
    compositions are passed in one batch to the heavy-oil CSP correlation.

    Parameters
    ----------
    model:
        Homogeneous-state fugacity model used for bubble points and flashes.
    temperature, pressure:
        One-dimensional state batches in K and Pa with matching shapes.
    feed_composition:
        One strictly positive feed mole-fraction vector.
    components:
        Transport critical properties and molar masses in the same order as
        the model and feed.
    third_csp, fourth_csp:
        Dimensionless Lindeloff calibration factors. Tensor inputs preserve
        gradients through the viscosity calculation.
    initial_bubble_pressure:
        Optional positive scalar pressure estimate in Pa reused at each
        temperature. Wilson initialization is used when omitted.
    tolerance:
        Dimensionless bubble-point and log-fugacity tolerance.
    saturation_iterations:
        Maximum iterations for each bubble-point solve.
    substitution_iterations, newton_iterations:
        Batched sub-bubble flash iteration limits.
    raise_on_failure:
        Raise :class:`ConvergenceError` if any boundary or flash fails. When
        false, failed-state viscosities are returned as NaN and diagnostics
        remain explicit.

    Returns
    -------
    HeavyOilCSPProfile
        Phase compositions, viscosities, and convergence diagnostics.

    Raises
    ------
    ValueError:
        If shapes, component order, or numerical controls are invalid.
    InvalidStateError:
        If state values or the feed are nonphysical.
    ConvergenceError:
        If a bubble point or sub-bubble flash fails and
        ``raise_on_failure=True``.

    Notes
    -----
    The viscosity formulation is Lindeloff et al., "The Corresponding States
    Viscosity Model Applied to Heavy Oil Systems" (2004), Paper 2003-150,
    through Pedersen, Christensen, and Shaikh (2024), section 10.1.2,
    doi:10.1201/9780429457418. Phase equilibrium is a separate cubic-EoS
    calculation; its model and characterization parameter set must therefore
    be reported with any result.
    """
    if (
        temperature.ndim != 1
        or pressure.ndim != 1
        or temperature.shape != pressure.shape
        or temperature.numel() == 0
    ):
        raise ValueError(
            "temperature and pressure must be nonempty matching one-dimensional batches"
        )
    if feed_composition.ndim != 1 or feed_composition.numel() != components.ncomponents:
        raise ValueError("feed composition must match the component set")
    if hasattr(model, "names") and tuple(model.names) != components.names:
        raise ValueError("phase model and transport component order must match")
    if (
        tolerance <= 0.0
        or saturation_iterations <= 0
        or substitution_iterations < 0
        or newton_iterations < 0
    ):
        raise ValueError("phase-equilibrium tolerances and iteration controls are invalid")
    invalid_state = (
        (~torch.isfinite(temperature))
        | (~torch.isfinite(pressure))
        | (temperature <= 0.0)
        | (pressure <= 0.0)
    )
    if bool(invalid_state.any()):
        raise InvalidStateError("heavy-oil profile temperatures and pressures must be positive")
    if not bool(
        torch.isfinite(feed_composition).all()
        & (feed_composition > 0.0).all()
        & torch.isfinite(feed_composition.sum())
    ):
        raise InvalidStateError("heavy-oil feed fractions must be finite and strictly positive")
    feed = normalize_composition(feed_composition)
    initial_pressure = (
        None
        if initial_bubble_pressure is None
        else torch.as_tensor(
            initial_bubble_pressure,
            dtype=temperature.dtype,
            device=temperature.device,
        )
    )
    if initial_pressure is not None and (
        initial_pressure.ndim
        or not bool(torch.isfinite(initial_pressure) & (initial_pressure > 0.0))
    ):
        raise ValueError("initial bubble pressure must be a positive finite scalar")

    state_count = temperature.numel()
    liquid_composition = feed.expand(state_count, -1).clone()
    bubble_pressure = torch.full_like(temperature, torch.nan)
    vapor_fraction = torch.zeros_like(temperature)
    residual_norm = torch.zeros_like(temperature)
    converged = torch.ones(state_count, dtype=torch.bool, device=temperature.device)

    for current_temperature in torch.unique(temperature):
        temperature_mask = temperature == current_temperature
        boundary = saturation_point(
            model,
            current_temperature,
            feed,
            "bubble",
            initial_pressure=initial_pressure,
            tolerance=tolerance,
            max_iterations=saturation_iterations,
        )
        bubble_pressure[temperature_mask] = boundary.pressure
        if not boundary.converged:
            converged[temperature_mask] = False
            residual_norm[temperature_mask] = boundary.residual_norm
            continue
        two_phase_mask = temperature_mask & (pressure < boundary.pressure)
        positions = torch.where(two_phase_mask)[0]
        if positions.numel() == 0:
            continue
        flash = batched_two_phase_flash(
            model,
            ChemicalState(
                temperature[positions],
                pressure[positions],
                feed,
            ),
            initial_k_values=boundary.k_values.expand(positions.numel(), -1),
            tolerance=tolerance,
            substitution_iterations=substitution_iterations,
            newton_iterations=newton_iterations,
        )
        liquid_composition[positions] = flash.liquid_composition
        vapor_fraction[positions] = flash.vapor_fraction
        residual_norm[positions] = flash.residual_norm
        converged[positions] = flash.converged

    if raise_on_failure and not bool(converged.all()):
        failed = torch.where(~converged)[0].detach().cpu().tolist()
        worst = float(residual_norm[~converged].detach().max())
        raise ConvergenceError(
            f"heavy-oil phase preparation failed at state indices {failed} "
            f"(maximum residual {worst:.3e})"
        )
    viscosity = heavy_oil_corresponding_states_viscosity(
        temperature,
        pressure,
        liquid_composition,
        components,
        phase="liquid",
        third_csp=third_csp,
        fourth_csp=fourth_csp,
    )
    viscosity = torch.where(converged, viscosity, torch.full_like(viscosity, torch.nan))
    return HeavyOilCSPProfile(
        viscosity,
        liquid_composition,
        bubble_pressure,
        vapor_fraction,
        converged,
        residual_norm,
    )


def fit_heavy_oil_csp_factors(
    temperature: Tensor,
    pressure: Tensor,
    liquid_composition: Tensor,
    components: ComponentSet,
    observed_viscosity: Tensor,
    *,
    initial_factors: tuple[float, float] = (1.0, 1.0),
    factor_bounds: tuple[float, float] = (0.1, 5.0),
    optimizer: OptimizerKind = "lbfgs",
    learning_rate: float = 0.3,
    max_iterations: int = 200,
    tolerance: float = 1.0e-10,
    no_improvement_patience: int = 30,
) -> HeavyOilCSPCalibrationResult:
    """Fit both heavy-oil CSP factors to all supplied states simultaneously.

    The full-batch objective is the mean squared logarithmic viscosity ratio,
    so every experimental point contributes in every optimizer step without
    a unit-dependent scale. Physical factors are smoothly bounded with a
    sigmoid transform. PyTorch supplies optimizer gradients and the final
    local sensitivity matrix.

    Parameters
    ----------
    temperature, pressure:
        State batches in K and Pa.
    liquid_composition:
        Feed or equilibrium liquid composition for every state, with
        components on the final axis.
    components:
        Critical properties and molar masses in matching order.
    observed_viscosity:
        Positive experimental dynamic viscosities in Pa s.
    initial_factors:
        Initial third and fourth CSP factors.
    factor_bounds:
        Shared positive lower and upper bounds for both factors.
    optimizer:
        PyTorch optimizer identifier. Full-batch LBFGS with strong-Wolfe line
        search is the default for this two-parameter smooth problem.
    learning_rate:
        Optimizer learning rate.
    max_iterations:
        Maximum optimizer steps; the default is 200.
    tolerance:
        Relative successive-loss stopping tolerance.
    no_improvement_patience:
        Early-stop patience; the default is 30 evaluated iterates.

    Returns
    -------
    HeavyOilCSPCalibrationResult
        Selected factors, fitted predictions, history, and identifiability
        diagnostics.

    Raises
    ------
    ValueError:
        If shapes, bounds, factors, or optimizer controls are invalid.
    InvalidStateError:
        If observations are nonpositive or nonfinite.

    Notes
    -----
    The fitted factors are the multipliers in Pedersen, Christensen, and
    Shaikh (2024), Eqs. 10.34-10.35,
    doi:10.1201/9780429457418. Lindeloff et al. (2004), Paper 2003-150,
    likewise fit both factors to all three Oil 5 temperature series
    simultaneously. This API does not refit the phase-equilibrium model.
    """
    if (
        temperature.ndim != 1
        or pressure.shape != temperature.shape
        or observed_viscosity.shape != temperature.shape
        or liquid_composition.shape != (temperature.numel(), components.ncomponents)
    ):
        raise ValueError("heavy-oil calibration state, observation, and composition shapes differ")
    if not bool(torch.isfinite(observed_viscosity).all() & (observed_viscosity > 0.0).all()):
        raise InvalidStateError("observed viscosities must be finite and positive")
    lower, upper = factor_bounds
    if (
        not 0.0 < lower < upper
        or len(initial_factors) != 2
        or any(not lower < value < upper for value in initial_factors)
    ):
        raise ValueError("initial CSP factors must lie strictly inside positive ordered bounds")
    lower_tensor = temperature.new_tensor(lower)
    span = temperature.new_tensor(upper - lower)
    initial = temperature.new_tensor(initial_factors)
    initial_unit = (initial - lower_tensor) / span
    unconstrained = nn.Parameter(torch.logit(initial_unit))

    def physical_factors(raw: Tensor) -> Tensor:
        return lower_tensor + span * torch.sigmoid(raw)

    def closure() -> Tensor:
        factors = physical_factors(unconstrained)
        prediction = heavy_oil_corresponding_states_viscosity(
            temperature,
            pressure,
            liquid_composition,
            components,
            phase="liquid",
            third_csp=factors[0],
            fourth_csp=factors[1],
        )
        return torch.mean(torch.log(prediction / observed_viscosity).square())

    fit = fit_parameters(
        (unconstrained,),
        closure,
        optimizer=optimizer,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
        no_improvement_patience=no_improvement_patience,
    )
    factors = physical_factors(unconstrained).detach()
    prediction = heavy_oil_corresponding_states_viscosity(
        temperature,
        pressure,
        liquid_composition,
        components,
        phase="liquid",
        third_csp=factors[0],
        fourth_csp=factors[1],
    ).detach()

    def log_prediction(current_factors: Tensor) -> Tensor:
        return torch.log(
            heavy_oil_corresponding_states_viscosity(
                temperature,
                pressure,
                liquid_composition,
                components,
                phase="liquid",
                third_csp=current_factors[0],
                fourth_csp=current_factors[1],
            )
        )

    sensitivity = torch.func.jacrev(log_prediction)(factors)
    singular_values = torch.linalg.svdvals(sensitivity)
    rank_tolerance = (
        max(sensitivity.shape) * torch.finfo(sensitivity.dtype).eps * singular_values.max()
    )
    rank = int(torch.sum(singular_values > rank_tolerance))
    condition_number = (
        float(torch.inf) if rank < 2 else float((singular_values[0] / singular_values[-1]).detach())
    )
    return HeavyOilCSPCalibrationResult(
        factors[0],
        factors[1],
        prediction,
        fit,
        sensitivity,
        singular_values,
        rank,
        condition_number,
    )


__all__ = [
    "HeavyOilCSPCalibrationResult",
    "HeavyOilCSPProfile",
    "evaluate_heavy_oil_corresponding_states_profile",
    "fit_heavy_oil_csp_factors",
]
