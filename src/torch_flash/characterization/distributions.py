"""Model-neutral plus-fraction splitting correlations."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import torch
from torch import Tensor

from torch_flash.characterization.types import SCNDistribution
from torch_flash.config import resolve_tensor_options
from torch_flash.database import ParameterSource, load_model_parameters
from torch_flash.exceptions import ConvergenceError, ParameterDatabaseError


def _characterization_parameters(source: ParameterSource) -> tuple[Mapping[str, object], str]:
    loaded = load_model_parameters(source)
    if loaded.model_kind != "characterization":
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} is {loaded.model_kind!r}, not 'characterization'"
        )
    return loaded.parameters, loaded.identifier


def _positive_number(record: Mapping[str, object], key: str, source: str) -> float:
    value = record.get(key)
    if not isinstance(value, int | float) or not isfinite(value) or value <= 0.0:
        raise ParameterDatabaseError(f"{source} {key!r} must be finite and positive")
    return float(value)


def pedersen_logarithmic_split(
    plus_mole_fraction: float | Tensor,
    plus_molar_mass: float | Tensor,
    *,
    first_carbon_number: int = 7,
    max_carbon_number: int | None = None,
    parameter_set: ParameterSource = "characterization.pedersen-2024",
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> SCNDistribution:
    """Split a plus fraction with Pedersen's logarithmic molar distribution.

    The finite distribution satisfies both plus-fraction mole and molar-mass
    balances (Pedersen et al., 2024, Eqs. 5.10-5.12). Molecular weights use
    Eq. 5.22. Measured extended compositions should be preferred whenever
    available, as emphasized by the source.
    """
    dtype, device = resolve_tensor_options(dtype, device)
    parameters, source = _characterization_parameters(parameter_set)
    split = parameters.get("plus_split")
    if not isinstance(split, Mapping):
        raise ParameterDatabaseError(f"{source} requires a plus_split mapping")
    if max_carbon_number is None:
        raw_max = split.get("default_max_carbon_number")
        if not isinstance(raw_max, int):
            raise ParameterDatabaseError(f"{source} default_max_carbon_number must be an integer")
        max_carbon_number = raw_max
    if (
        not isinstance(first_carbon_number, int)
        or not isinstance(max_carbon_number, int)
        or first_carbon_number < 1
        or max_carbon_number < first_carbon_number
    ):
        raise ValueError("carbon-number bounds must be ordered positive integers")
    relation = split.get("molecular_weight")
    if not isinstance(relation, Mapping):
        raise ParameterDatabaseError(f"{source} requires a molecular_weight mapping")
    slope = _positive_number(relation, "slope", source)
    intercept_value = relation.get("intercept")
    if not isinstance(intercept_value, int | float) or not isfinite(intercept_value):
        raise ParameterDatabaseError(f"{source} molecular-weight intercept must be finite")
    intercept = float(intercept_value)

    fraction = torch.as_tensor(plus_mole_fraction, dtype=dtype, device=device)
    target_mass = torch.as_tensor(plus_molar_mass, dtype=dtype, device=device)
    if fraction.ndim or target_mass.ndim:
        raise ValueError("plus mole fraction and molar mass must be scalar")
    if not bool(torch.isfinite(fraction) & (fraction > 0.0)):
        raise ValueError("plus mole fraction must be finite and positive")
    if not bool(torch.isfinite(target_mass) & (target_mass > 0.0)):
        raise ValueError("plus molar mass must be finite and positive")
    carbon_numbers = torch.arange(
        first_carbon_number,
        max_carbon_number + 1,
        dtype=dtype,
        device=device,
    )
    molar_masses = (slope * carbon_numbers + intercept) * 1.0e-3
    if not bool(
        (target_mass >= molar_masses[0] - 1.0e-12) & (target_mass <= molar_masses[-1] + 1.0e-12)
    ):
        raise ValueError("plus molar mass lies outside the selected finite SCN range")

    # Fixed Newton iterations retain an autodiff path through the moment
    # constraint. d<E[M]>/d lambda is cov(M, carbon number).
    log_slope = target_mass.new_zeros(())
    centered_carbon = carbon_numbers - carbon_numbers[0]
    for _ in range(40):
        weights = torch.softmax(log_slope * centered_carbon, dim=0)
        mean_mass = torch.sum(weights * molar_masses)
        mean_carbon = torch.sum(weights * centered_carbon)
        covariance = torch.sum(
            weights * (molar_masses - mean_mass) * (centered_carbon - mean_carbon)
        )
        step = torch.clamp((mean_mass - target_mass) / covariance, -2.0, 2.0)
        log_slope = torch.clamp(log_slope - step, -50.0, 50.0)
    weights = torch.softmax(log_slope * centered_carbon, dim=0)
    mole_fractions = fraction * weights
    if float(torch.abs(torch.sum(weights * molar_masses) - target_mass).detach()) > 1.0e-9:
        raise ConvergenceError("Pedersen logarithmic split did not satisfy its mass balance")
    return SCNDistribution(
        carbon_numbers.to(torch.int64),
        mole_fractions,
        molar_masses,
    )


def pedersen_density_split(
    distribution: SCNDistribution,
    plus_density: float | Tensor,
    *,
    anchor_density: float | Tensor | None = None,
    anchor_carbon_number: int | None = None,
    parameter_set: ParameterSource = "characterization.pedersen-2024",
) -> SCNDistribution:
    """Assign ``rho_N = C + D ln(CN)`` while matching bulk plus density.

    Density inputs and results are SI (kg/m3). If no measured anchor is
    provided, Pedersen's suggested C6 density ratio is applied to the carbon
    number immediately preceding the plus fraction.
    """
    parameters, source = _characterization_parameters(parameter_set)
    split = parameters.get("plus_split")
    if not isinstance(split, Mapping):
        raise ParameterDatabaseError(f"{source} requires a plus_split mapping")
    if split.get("density_log_carbon_number") is not True:
        raise ParameterDatabaseError(f"{source} does not define the logarithmic density rule")
    target = torch.as_tensor(
        plus_density,
        dtype=distribution.mole_fractions.dtype,
        device=distribution.mole_fractions.device,
    )
    if target.ndim or not bool(torch.isfinite(target) & (target > 0.0)):
        raise ValueError("plus density must be a finite positive scalar")
    if anchor_density is None:
        ratio = _positive_number(split, "default_anchor_density_ratio", source)
        anchor = ratio * target
    else:
        anchor = torch.as_tensor(
            anchor_density,
            dtype=target.dtype,
            device=target.device,
        )
    if anchor.ndim or not bool(torch.isfinite(anchor) & (anchor > 0.0)):
        raise ValueError("anchor density must be a finite positive scalar")
    if anchor_carbon_number is None:
        anchor_carbon_number = int(distribution.carbon_numbers[0]) - 1
    if anchor_carbon_number < 1:
        raise ValueError("anchor carbon number must be positive")

    log_delta = torch.log(distribution.carbon_numbers.to(target)) - torch.log(
        target.new_tensor(float(anchor_carbon_number))
    )
    mass = distribution.mole_fractions * distribution.molar_masses

    def residual(density_slope: Tensor) -> Tensor:
        densities = anchor + density_slope * log_delta
        bulk = mass.sum() / torch.sum(mass / densities)
        return bulk - target

    density_slope = (target - anchor) / torch.clamp(log_delta.mean(), min=1.0e-6)
    lower = -0.95 * anchor / torch.clamp(log_delta.max(), min=1.0e-6)
    upper = 10.0 * target
    for _ in range(30):
        value = residual(density_slope)
        derivative = torch.func.grad(residual)(density_slope)
        density_slope = torch.clamp(density_slope - value / derivative, lower, upper)
    densities = anchor + density_slope * log_delta
    if (
        bool((densities <= 0.0).any())
        or float(torch.abs(residual(density_slope) / target).detach()) > 1.0e-10
    ):
        raise ConvergenceError("Pedersen density split did not satisfy its volume balance")
    return SCNDistribution(
        distribution.carbon_numbers,
        distribution.mole_fractions,
        distribution.molar_masses,
        densities,
    )


def whitson_gamma_split(
    plus_mole_fraction: float | Tensor,
    plus_molar_mass: float | Tensor,
    *,
    first_carbon_number: int = 7,
    max_carbon_number: int = 80,
    shape: float | Tensor | None = None,
    minimum_molar_mass: float | Tensor | None = None,
    parameter_set: ParameterSource = "characterization.whitson-2000",
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> SCNDistribution:
    """Discretize Whitson's shifted gamma distribution into SCN-like bins.

    The last requested bin contains the complete tail to infinite molecular
    weight, so total moles and average molar mass are preserved exactly apart
    from floating-point roundoff. ``shape=1`` gives the exponential special
    case discussed by both Whitson and Pedersen.
    """
    dtype, device = resolve_tensor_options(dtype, device)
    parameters, source = _characterization_parameters(parameter_set)
    gamma = parameters.get("gamma_distribution")
    if not isinstance(gamma, Mapping):
        raise ParameterDatabaseError(f"{source} requires a gamma_distribution mapping")
    if shape is None:
        shape = _positive_number(gamma, "default_shape", source)
    alpha = torch.as_tensor(shape, dtype=dtype, device=device)
    fraction = torch.as_tensor(plus_mole_fraction, dtype=dtype, device=device)
    average_mass = torch.as_tensor(plus_molar_mass, dtype=dtype, device=device)
    if any(value.ndim for value in (alpha, fraction, average_mass)):
        raise ValueError("gamma shape and plus-fraction inputs must be scalar")
    if not bool(
        torch.isfinite(alpha)
        & (alpha > 0.0)
        & torch.isfinite(fraction)
        & (fraction > 0.0)
        & torch.isfinite(average_mass)
        & (average_mass > 0.0)
    ):
        raise ValueError("gamma shape, mole fraction, and molar mass must be finite and positive")
    if first_carbon_number < 1 or max_carbon_number < first_carbon_number:
        raise ValueError("carbon-number bounds must be ordered positive integers")
    if minimum_molar_mass is None:
        relation = gamma.get("recommended_minimum_molecular_weight_relation")
        if not isinstance(relation, Mapping):
            raise ParameterDatabaseError(
                f"{source} requires recommended minimum-molecular-weight coefficients"
            )
        scale = _positive_number(relation, "scale", source)
        multiplier = _positive_number(relation, "multiplier", source)
        exponent = _positive_number(relation, "exponent", source)
        eta = 1.0e-3 * scale * (1.0 - 1.0 / (1.0 + multiplier / alpha**exponent))
    else:
        eta = torch.as_tensor(minimum_molar_mass, dtype=dtype, device=device)
    if eta.ndim or not bool(torch.isfinite(eta) & (eta > 0.0) & (eta < average_mass)):
        raise ValueError("minimum molar mass must be finite, positive, and below the average")
    beta = (average_mass - eta) / alpha

    carbon_numbers = torch.arange(
        first_carbon_number,
        max_carbon_number + 1,
        dtype=dtype,
        device=device,
    )
    boundary_increment = _positive_number(
        gamma,
        "molecular_weight_boundary_increment",
        source,
    )
    # Whitson and Brule (2000), Table 5.4 and program GAMSPL, place
    # molecular-weight boundaries at eta, eta + 14, eta + 28, ... g/mol.
    # The carbon numbers are therefore nominal labels for consecutive bins;
    # they do not define midpoints through the n-paraffin relation.
    offsets = torch.arange(
        carbon_numbers.numel(),
        dtype=dtype,
        device=device,
    )
    lower = eta + boundary_increment * offsets
    upper = torch.cat(
        (
            lower[1:],
            torch.full_like(eta.reshape(1), torch.inf),
        )
    )
    y_lower = torch.clamp((lower - eta) / beta, min=0.0)
    y_upper = torch.clamp((upper - eta) / beta, min=0.0)
    p_lower = torch.special.gammainc(alpha, y_lower)
    p_upper = torch.special.gammainc(alpha, y_upper)
    probabilities = p_upper - p_lower
    first_moment_probability = torch.special.gammainc(alpha + 1.0, y_upper)
    first_moment_probability = first_moment_probability - torch.special.gammainc(
        alpha + 1.0, y_lower
    )
    bin_masses = eta + alpha * beta * first_moment_probability / probabilities
    return SCNDistribution(
        carbon_numbers.to(torch.int64),
        fraction * probabilities,
        bin_masses,
    )
