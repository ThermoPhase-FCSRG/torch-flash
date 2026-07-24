"""Heavy-end pseudo-component characterization for CPA.

The correlations implement Yan, Kontogeorgis, and Stenby, *Fluid Phase
Equilibria* 276 (2009) 75-85, Eqs. 10-15,
doi:10.1016/j.fluid.2008.10.007. They map a narrow heavy-end cut's normal
boiling temperature and specific gravity to CPA monomer properties. The
resulting pseudo-components use the conventional SRK form and can therefore
be passed directly to :class:`torch_flash.eos.cpa.CPAEOS`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log, sqrt

import torch
from torch import Tensor

from torch_flash.characterization import PseudoComponentCut
from torch_flash.config import resolve_tensor_options
from torch_flash.constants import R
from torch_flash.database import ParameterSource, load_model_parameters
from torch_flash.eos.cpa import CPAComponent
from torch_flash.exceptions import ConvergenceError, ParameterDatabaseError


@dataclass(frozen=True)
class CPAHeavyEndCorrelations:
    """Published coefficients used by the CPA C7+ monomer correlations."""

    normal_boiling_pressure: float
    srk_omega_a: float
    srk_omega_b: float
    srk_m: tuple[float, float, float]
    normal_alkane_temperature: tuple[float, float, float]
    log_normal_alkane_pressure_bar: tuple[float, float, float, float, float]
    normal_alkane_acentric: tuple[float, float, float]
    normal_alkane_specific_gravity: tuple[float, float, float, float, float]
    critical_temperature_numerator: tuple[float, float, float]
    critical_temperature_denominator: tuple[float, float, float]
    log_critical_pressure_ratio: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class CPAMonomerProperties:
    """Intermediate and final CPA monomer characterization results."""

    normal_boiling_temperature: float
    specific_gravity: float
    normal_alkane_specific_gravity: float
    specific_gravity_perturbation: float
    normal_alkane_temperature: float
    normal_alkane_pressure: float
    critical_temperature: float
    critical_pressure: float
    acentric_factor: float
    used_boiling_point_match: bool
    correlations: CPAHeavyEndCorrelations

    @property
    def m(self) -> float:
        """Return the SRK ``m`` coefficient corresponding to ``omega``."""
        omega = self.acentric_factor
        first, second, third = self.correlations.srk_m
        return first + second * omega + third * omega * omega

    @property
    def a0(self) -> float:
        """Return CPA/SRK energy parameter in Pa m6 mol-2."""
        return (
            self.correlations.srk_omega_a
            * (R * self.critical_temperature) ** 2
            / self.critical_pressure
        )

    @property
    def b(self) -> float:
        """Return CPA/SRK covolume in m3 mol-1."""
        return (
            self.correlations.srk_omega_b * R * self.critical_temperature / self.critical_pressure
        )


@dataclass(frozen=True)
class CPACharacterizedComponents:
    """Characterized pseudo-components and their normalized cut fractions."""

    components: tuple[CPAComponent, ...]
    mole_fractions: Tensor
    monomer_properties: tuple[CPAMonomerProperties, ...]


def _numeric_tuple(
    value: object,
    length: int,
    key: str,
    source: str,
) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or len(value) != length
        or any(not isinstance(item, int | float) for item in value)
    ):
        raise ParameterDatabaseError(
            f"{source} heavy_end_correlations {key!r} requires {length} numeric values"
        )
    result = tuple(float(item) for item in value)
    if any(not isfinite(item) for item in result):
        raise ParameterDatabaseError(f"{source} heavy_end_correlations {key!r} must be finite")
    return result


def cpa_heavy_end_correlations(
    parameter_set: ParameterSource = "cpa.yan-2009-reservoir-fluids",
) -> CPAHeavyEndCorrelations:
    """Load and validate CPA heavy-end correlation coefficients."""
    loaded = load_model_parameters(parameter_set)
    if loaded.model_kind != "cpa":
        raise ParameterDatabaseError(f"{loaded.identifier!r} is not a CPA parameter set")
    record = loaded.parameters.get("heavy_end_correlations")
    if not isinstance(record, Mapping):
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} requires a heavy_end_correlations mapping"
        )

    def positive(key: str) -> float:
        value = record.get(key)
        if not isinstance(value, int | float) or not isfinite(value) or value <= 0.0:
            raise ParameterDatabaseError(
                f"{loaded.identifier!r} heavy_end_correlations {key!r} must be finite and positive"
            )
        return float(value)

    temperature_ratio = record.get("critical_temperature_ratio")
    if not isinstance(temperature_ratio, Mapping):
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} requires critical_temperature_ratio coefficients"
        )
    return CPAHeavyEndCorrelations(
        positive("normal_boiling_pressure"),
        positive("srk_omega_a"),
        positive("srk_omega_b"),
        _numeric_tuple(record.get("srk_m"), 3, "srk_m", loaded.identifier),  # type: ignore[arg-type]
        _numeric_tuple(
            record.get("normal_alkane_temperature"),
            3,
            "normal_alkane_temperature",
            loaded.identifier,
        ),  # type: ignore[arg-type]
        _numeric_tuple(
            record.get("log_normal_alkane_pressure_bar"),
            5,
            "log_normal_alkane_pressure_bar",
            loaded.identifier,
        ),  # type: ignore[arg-type]
        _numeric_tuple(
            record.get("normal_alkane_acentric"),
            3,
            "normal_alkane_acentric",
            loaded.identifier,
        ),  # type: ignore[arg-type]
        _numeric_tuple(
            record.get("normal_alkane_specific_gravity"),
            5,
            "normal_alkane_specific_gravity",
            loaded.identifier,
        ),  # type: ignore[arg-type]
        _numeric_tuple(
            temperature_ratio.get("numerator"),
            3,
            "critical_temperature_ratio.numerator",
            loaded.identifier,
        ),  # type: ignore[arg-type]
        _numeric_tuple(
            temperature_ratio.get("denominator"),
            3,
            "critical_temperature_ratio.denominator",
            loaded.identifier,
        ),  # type: ignore[arg-type]
        _numeric_tuple(
            record.get("log_critical_pressure_ratio"),
            5,
            "log_critical_pressure_ratio",
            loaded.identifier,
        ),  # type: ignore[arg-type]
    )


def _srk_m(acentric_factor: float, coefficients: tuple[float, float, float]) -> float:
    first, second, third = coefficients
    return first + second * acentric_factor + third * acentric_factor * acentric_factor


def _pure_srk_fugacity_difference(
    acentric_factor: float,
    temperature: float,
    pressure: float,
    critical_temperature: float,
    critical_pressure: float,
    correlations: CPAHeavyEndCorrelations,
) -> float | None:
    reduced_temperature = temperature / critical_temperature
    reduced_pressure = pressure / critical_pressure
    alpha = (
        1.0 + _srk_m(acentric_factor, correlations.srk_m) * (1.0 - sqrt(reduced_temperature))
    ) ** 2
    a = (
        correlations.srk_omega_a
        * reduced_pressure
        * alpha
        / (reduced_temperature * reduced_temperature)
    )
    b = correlations.srk_omega_b * reduced_pressure / reduced_temperature
    coefficient = a - b - b * b
    companion = torch.tensor(
        (
            (0.0, 0.0, a * b),
            (1.0, 0.0, -coefficient),
            (0.0, 1.0, 1.0),
        ),
        dtype=torch.float64,
    )
    eigenvalues = torch.linalg.eigvals(companion)
    roots = sorted(
        float(value.real)
        for value in eigenvalues
        if abs(float(value.imag)) <= 1.0e-9 and float(value.real) > b
    )
    if len(roots) < 2:
        return None

    def log_fugacity(z: float) -> float:
        return z - 1.0 - log(z - b) - a / b * log(1.0 + b / z)

    return log_fugacity(roots[0]) - log_fugacity(roots[-1])


def _match_normal_boiling_point(
    temperature: float,
    critical_temperature: float,
    critical_pressure: float,
    correlations: CPAHeavyEndCorrelations,
) -> float:
    grid = [float(value) for value in torch.linspace(-0.49, 2.5, 300, dtype=torch.float64)]
    previous_omega: float | None = None
    previous_value: float | None = None
    bracket: tuple[float, float] | None = None
    for omega in grid:
        value = _pure_srk_fugacity_difference(
            omega,
            temperature,
            correlations.normal_boiling_pressure,
            critical_temperature,
            critical_pressure,
            correlations,
        )
        if value is None or not isfinite(value):
            continue
        if value == 0.0:
            return omega
        if (
            previous_omega is not None
            and previous_value is not None
            and (value > 0.0) != (previous_value > 0.0)
        ):
            bracket = (previous_omega, omega)
            break
        previous_omega = omega
        previous_value = value
    if bracket is None:
        raise ConvergenceError(
            "could not match the normal boiling point with the characterized CPA monomer"
        )
    lower, upper = bracket
    lower_value = _pure_srk_fugacity_difference(
        lower,
        temperature,
        correlations.normal_boiling_pressure,
        critical_temperature,
        critical_pressure,
        correlations,
    )
    if lower_value is None:  # pragma: no cover - bracket construction guarantees this
        raise AssertionError
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        midpoint_value = _pure_srk_fugacity_difference(
            midpoint,
            temperature,
            correlations.normal_boiling_pressure,
            critical_temperature,
            critical_pressure,
            correlations,
        )
        if midpoint_value is None:  # pragma: no cover - interior of a valid bracket
            raise ConvergenceError("CPA boiling-point match lost its two-phase root")
        if abs(midpoint_value) <= 1.0e-12:
            return midpoint
        if (midpoint_value > 0.0) == (lower_value > 0.0):
            lower = midpoint
            lower_value = midpoint_value
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def cpa_monomer_properties(
    normal_boiling_temperature: float,
    specific_gravity: float,
    parameter_set: ParameterSource = "cpa.yan-2009-reservoir-fluids",
) -> CPAMonomerProperties:
    """Calculate CPA monomer properties for one narrow heavy-end cut.

    Parameters
    ----------
    normal_boiling_temperature
        Atmospheric normal boiling temperature in K.
    specific_gravity
        Dimensionless liquid specific gravity used by the source
        characterization.
    """
    tb = float(normal_boiling_temperature)
    sg = float(specific_gravity)
    if not isfinite(tb) or not isfinite(sg) or tb <= 0.0 or sg <= 0.0:
        raise ValueError("boiling temperature and specific gravity must be finite and positive")

    correlations = cpa_heavy_end_correlations(parameter_set)
    temperature_constant, temperature_slope, temperature_denominator = (
        correlations.normal_alkane_temperature
    )
    normal_temperature = (
        (temperature_constant + temperature_slope * tb) * tb / (temperature_denominator + tb)
    )
    p4, p3, p2, p1, p0 = correlations.log_normal_alkane_pressure_bar
    log_normal_pressure_bar = p4 * tb**4 + p3 * tb**3 + p2 * tb**2 + p1 * tb + p0
    normal_pressure = 1.0e5 * exp(log_normal_pressure_bar)
    sg_scale, sg_constant, sg_linear, sg_inverse, sg_inverse_square = (
        correlations.normal_alkane_specific_gravity
    )
    normal_specific_gravity = (sg_scale * tb) ** (1.0 / 3.0) / (
        sg_constant + sg_linear * tb + sg_inverse / tb + sg_inverse_square / tb**2
    )
    perturbation = sg - normal_specific_gravity
    tn1, tn2, tn3 = correlations.critical_temperature_numerator
    td1, td2, td3 = correlations.critical_temperature_denominator
    temperature_ratio = (
        1.0 + tn1 * perturbation + tn2 * perturbation**2 + tn3 * perturbation**3
    ) / (1.0 + td1 * perturbation + td2 * perturbation**2 + td3 * perturbation**3)
    critical_temperature = normal_temperature * temperature_ratio
    pr0, pr1, pr2, pr3, pr4 = correlations.log_critical_pressure_ratio
    log_pressure_ratio = (
        perturbation
        * (pr0 + (pr1 + pr2 / sg) * perturbation)
        / (1.0 + pr3 * perturbation + pr4 * perturbation**2)
    )
    critical_pressure = normal_pressure * exp(log_pressure_ratio)
    if (
        not isfinite(critical_temperature)
        or not isfinite(critical_pressure)
        or critical_temperature <= 0.0
        or critical_pressure <= 0.0
    ):
        raise ValueError("Yan CPA characterization is outside its physical correlation domain")

    if tb < critical_temperature:
        acentric_factor = _match_normal_boiling_point(
            tb,
            critical_temperature,
            critical_pressure,
            correlations,
        )
        used_match = True
    else:
        omega_constant, omega_slope, omega_denominator = correlations.normal_alkane_acentric
        acentric_factor = exp((omega_constant + omega_slope * tb) / (omega_denominator + tb))
        used_match = False
    return CPAMonomerProperties(
        tb,
        sg,
        normal_specific_gravity,
        perturbation,
        normal_temperature,
        normal_pressure,
        critical_temperature,
        critical_pressure,
        acentric_factor,
        used_match,
        correlations,
    )


def cpa_pseudocomponent(
    name: str,
    normal_boiling_temperature: float,
    specific_gravity: float,
    molar_mass: float,
    parameter_set: ParameterSource = "cpa.yan-2009-reservoir-fluids",
) -> CPAComponent:
    """Create a non-associating CPA pseudo-component for a heavy-end cut."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("pseudo-component name must be a non-empty string")
    mass = float(molar_mass)
    if not isfinite(mass) or mass <= 0.0:
        raise ValueError("pseudo-component molar mass must be finite and positive")
    properties = cpa_monomer_properties(
        normal_boiling_temperature,
        specific_gravity,
        parameter_set,
    )
    return CPAComponent(
        name=name,
        critical_temperature=properties.critical_temperature,
        a0=properties.a0,
        b=properties.b,
        c1=properties.m,
        critical_pressure=properties.critical_pressure,
        acentric_factor=properties.acentric_factor,
        molar_mass=mass,
    )


def cpa_components_from_cuts(
    cuts: tuple[PseudoComponentCut, ...],
    *,
    parameter_set: ParameterSource = "cpa.yan-2009-reservoir-fluids",
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> CPACharacterizedComponents:
    """Characterize analyzed C7+ cuts and normalize their internal fractions.

    This function covers the EoS-parameter step of a plus-fraction workflow.
    It deliberately does not invent a carbon-number distribution when only a
    bulk C7+ molecular weight is known; splitting or lumping must be performed
    upstream with measured cuts or an explicitly selected Whitson/Pedersen
    distribution model.
    """
    dtype, device = resolve_tensor_options(dtype, device)
    if not cuts:
        raise ValueError("at least one pseudo-component cut is required")
    if any(not isfinite(cut.mole_fraction) or cut.mole_fraction < 0.0 for cut in cuts):
        raise ValueError("pseudo-component cut mole fractions must be finite and nonnegative")
    fractions = torch.tensor(
        [cut.mole_fraction for cut in cuts],
        dtype=dtype,
        device=device,
    )
    if not bool(fractions.sum() > 0.0):
        raise ValueError("pseudo-component cut mole fractions must have a positive sum")
    properties = tuple(
        cpa_monomer_properties(
            cut.normal_boiling_temperature,
            cut.specific_gravity,
            parameter_set,
        )
        for cut in cuts
    )
    components = tuple(
        CPAComponent(
            name=cut.name,
            critical_temperature=item.critical_temperature,
            a0=item.a0,
            b=item.b,
            c1=item.m,
            critical_pressure=item.critical_pressure,
            acentric_factor=item.acentric_factor,
            molar_mass=cut.molar_mass,
        )
        for cut, item in zip(cuts, properties, strict=True)
    )
    return CPACharacterizedComponents(
        components,
        fractions / fractions.sum(),
        properties,
    )
