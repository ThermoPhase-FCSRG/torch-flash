"""Cubic-EoS property adapters for characterized heavy-end fractions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from torch_flash.characterization.types import SCNDistribution
from torch_flash.database import ParameterSource, load_model_parameters
from torch_flash.exceptions import ParameterDatabaseError

CubicCharacterization = Literal["SRK", "PR"]


@dataclass(frozen=True)
class CubicFractionProperties:
    """Critical properties and acentric factors for cubic-EoS adapters.

    Attributes
    ----------
    critical_temperature:
        Cut critical temperatures in K.
    critical_pressure:
        Cut critical pressures in Pa.
    acentric_factor:
        Dimensionless cut acentric factors.
    m:
        Dimensionless alpha-function parameter associated with the requested
        SRK or PR correlation.
    """

    critical_temperature: Tensor
    critical_pressure: Tensor
    acentric_factor: Tensor
    m: Tensor


def _coefficients(
    record: Mapping[str, object],
    key: str,
    length: int,
    source: str,
) -> tuple[float, ...]:
    value = record.get(key)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or len(value) != length
        or any(not isinstance(item, int | float) for item in value)
    ):
        raise ParameterDatabaseError(f"{source} {key!r} requires {length} numeric coefficients")
    return tuple(float(item) for item in value)


def pedersen_cubic_properties(
    distribution: SCNDistribution,
    eos: CubicCharacterization,
    parameter_set: ParameterSource = "characterization.pedersen-2024",
) -> CubicFractionProperties:
    """Map characterized SCN cuts to SRK or PR properties.

    Implements Pedersen et al. (2024), Eqs. 5.1-5.5 and Table 5.3.
    The correlation coefficients are EoS-specific; the input distribution and
    density split remain model-neutral.

    Parameters
    ----------
    distribution:
        SCN distribution with molar masses in kg/mol and densities in kg/m3.
    eos:
        ``"SRK"`` or ``"PR"``; each selects a distinct coefficient table.
    parameter_set:
        Characterization parameter identifier, YAML path, or explicit record.

    Returns
    -------
    CubicFractionProperties
        Cut properties with the same shape, dtype, and device as the
        distribution.

    Raises
    ------
    ValueError
        If the EoS name is invalid, densities are absent, or the fitted
        acentric-factor inversion has no real solution.
    ParameterDatabaseError
        If the selected document is not a compatible characterization set.
    """
    if eos not in ("SRK", "PR"):
        raise ValueError("eos must be 'SRK' or 'PR'")
    if distribution.densities is None:
        raise ValueError("Pedersen cubic properties require characterized SCN densities")
    loaded = load_model_parameters(parameter_set)
    if loaded.model_kind != "characterization":
        raise ParameterDatabaseError(f"{loaded.identifier!r} is not a characterization set")
    tables = loaded.parameters.get("cubic_properties")
    if not isinstance(tables, Mapping):
        raise ParameterDatabaseError(f"{loaded.identifier!r} requires cubic_properties")
    record = tables.get(eos)
    if not isinstance(record, Mapping):
        raise ParameterDatabaseError(f"{loaded.identifier!r} has no {eos} property table")
    c1, c2, c3, c4 = _coefficients(record, "critical_temperature", 4, loaded.identifier)
    d1, d2, d3, d4, d5 = _coefficients(record, "log_critical_pressure", 5, loaded.identifier)
    e1, e2, e3, e4 = _coefficients(record, "m", 4, loaded.identifier)
    w0, w1, w2 = _coefficients(record, "m_to_acentric_factor", 3, loaded.identifier)
    molar_mass = 1.0e3 * distribution.molar_masses
    density = 1.0e-3 * distribution.densities
    critical_temperature = c1 * density + c2 * torch.log(molar_mass) + c3 * molar_mass
    critical_temperature = critical_temperature + c4 / molar_mass
    log_pressure_atm = d1 + d2 * density**d5 + d3 / molar_mass + d4 / molar_mass.square()
    critical_pressure = 101_325.0 * torch.exp(log_pressure_atm)
    m = e1 + e2 * molar_mass + e3 * density + e4 * molar_mass.square()
    discriminant = w1 * w1 - 4.0 * w2 * (w0 - m)
    if bool((discriminant < 0.0).any()):
        raise ValueError("Pedersen cubic correlation produced no real acentric factor")
    root_first = (-w1 + torch.sqrt(discriminant)) / (2.0 * w2)
    root_second = (-w1 - torch.sqrt(discriminant)) / (2.0 * w2)
    acentric = torch.where(
        torch.abs(root_first) <= torch.abs(root_second),
        root_first,
        root_second,
    )
    return CubicFractionProperties(
        critical_temperature,
        critical_pressure,
        acentric,
        m,
    )
