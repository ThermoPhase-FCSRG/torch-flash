"""Ideal-gas standard-state contributions and caloric reference choices.

The polynomial integrations implement Pedersen, *Phase Behavior of Petroleum
Reservoir Fluids*, 3rd ed. (2024), Eqs. 8.2-8.4,
doi:10.1201/9780429457418. The named coefficients are frozen from the Poling
data bank distributed by ``chemicals`` 1.5.2 and trace back to Poling et al.,
*The Properties of Gases and Liquids*, 5th ed. (2001).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import torch
from torch import Tensor, nn

from torch_flash.components import component
from torch_flash.config import resolve_tensor_options
from torch_flash.constants import R
from torch_flash.database import ParameterSource, load_model_parameters
from torch_flash.exceptions import ParameterDatabaseError


class StandardState(Protocol):
    """Protocol for component standard chemical potentials.

    Implementations return one value per modeled component, retain leading
    temperature batch dimensions, and use the same dtype and device as the
    input temperature.

    Methods
    -------
    chemical_potential
        Return component standard chemical potentials in J/mol.
    """

    def chemical_potential(self, temperature: Tensor) -> Tensor:
        """Return component standard chemical potentials in J/mol."""


class IdealGasPolynomial(nn.Module):
    """Polynomial ideal-gas heat-capacity standard state.

    For ``m`` columns, the heat capacity is
    ``sum(c[j]*T**j, j=0..m-1)`` in J/(mol K). Four columns recover
    Pedersen's Eq. 8.4; five columns support the Poling data bank. Reference
    enthalpies and entropies make the otherwise arbitrary caloric datum
    explicit and allow the coefficients to be fitted with PyTorch.

    Parameters
    ----------
    heat_capacity_coefficients
        Matrix of coefficients in ascending temperature power, with one row
        per component and SI units that make the evaluated heat capacity
        J/(mol K).
    reference_enthalpy
        Component enthalpies in J/mol at ``reference_temperature``.
    reference_entropy
        Component standard-pressure entropies in J/(mol K) at the reference.
    reference_temperature
        Positive caloric reference temperature in K.
    trainable
        Register heat-capacity coefficients as trainable parameters.
    """

    heat_capacity_coefficients: Tensor
    reference_enthalpy: Tensor
    reference_entropy: Tensor

    def __init__(
        self,
        heat_capacity_coefficients: Tensor,
        reference_enthalpy: Tensor,
        reference_entropy: Tensor,
        *,
        reference_temperature: float = 298.15,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if heat_capacity_coefficients.ndim != 2 or heat_capacity_coefficients.shape[1] < 1:
            raise ValueError("heat-capacity coefficients must be a nonempty matrix")
        expected = (heat_capacity_coefficients.shape[0],)
        if reference_enthalpy.shape != expected or reference_entropy.shape != expected:
            raise ValueError("one reference enthalpy and entropy are required per component")
        if reference_temperature <= 0.0:
            raise ValueError("reference temperature must be positive")
        self.reference_temperature = float(reference_temperature)
        if trainable:
            self.heat_capacity_coefficients = nn.Parameter(heat_capacity_coefficients.clone())
        else:
            self.register_buffer(
                "heat_capacity_coefficients",
                heat_capacity_coefficients.clone(),
            )
        self.register_buffer("reference_enthalpy", reference_enthalpy.clone())
        self.register_buffer("reference_entropy", reference_entropy.clone())

    def heat_capacity(self, temperature: Tensor) -> Tensor:
        """Evaluate component ideal-gas heat capacities.

        Parameters
        ----------
        temperature
            Temperature in K with arbitrary leading batch dimensions.

        Returns
        -------
        Tensor
            Heat capacities in J/(mol K), with a final component axis.
        """
        powers = torch.arange(
            self.heat_capacity_coefficients.shape[1],
            dtype=temperature.dtype,
            device=temperature.device,
        )
        return torch.sum(
            self.heat_capacity_coefficients * temperature[..., None, None].pow(powers),
            dim=-1,
        )

    def enthalpy(self, temperature: Tensor) -> Tensor:
        """Integrate component ideal-gas enthalpies from the reference state.

        Parameters
        ----------
        temperature
            Temperature in K.

        Returns
        -------
        Tensor
            Component enthalpies in J/mol with a final component axis.
        """
        powers = torch.arange(
            1,
            self.heat_capacity_coefficients.shape[1] + 1,
            dtype=temperature.dtype,
            device=temperature.device,
        )
        reference = temperature.new_tensor(self.reference_temperature)
        integral = (
            self.heat_capacity_coefficients
            * (temperature[..., None, None].pow(powers) - reference.pow(powers))
            / powers
        )
        return self.reference_enthalpy + torch.sum(integral, dim=-1)

    def entropy(self, temperature: Tensor) -> Tensor:
        """Integrate component ideal-gas entropies at standard pressure.

        Parameters
        ----------
        temperature
            Positive temperature in K.

        Returns
        -------
        Tensor
            Component standard-state entropies in J/(mol K).
        """
        reference = temperature.new_tensor(self.reference_temperature)
        coefficients = self.heat_capacity_coefficients
        leading = coefficients[:, 0] * torch.log(temperature[..., None] / reference)
        powers = torch.arange(
            1,
            coefficients.shape[1],
            dtype=temperature.dtype,
            device=temperature.device,
        )
        remaining = (
            coefficients[:, 1:]
            * (temperature[..., None, None].pow(powers) - reference.pow(powers))
            / powers
        )
        return self.reference_entropy + leading + torch.sum(remaining, dim=-1)

    def chemical_potential(self, temperature: Tensor) -> Tensor:
        """Return ideal-gas standard chemical potentials.

        Parameters
        ----------
        temperature
            Temperature in K.

        Returns
        -------
        Tensor
            ``h_i - T s_i`` in J/mol with a final component axis.
        """
        return self.enthalpy(temperature) - temperature[..., None] * self.entropy(temperature)


def ideal_gas_polynomial(
    names: tuple[str, ...] | list[str],
    parameter_set: ParameterSource,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    reference_temperature: float | None = None,
    trainable: bool = False,
) -> IdealGasPolynomial:
    """Construct an ideal-gas polynomial from versioned parameters.

    Parameters
    ----------
    names
        Component names in the desired output order. Aliases are canonicalized
        through the component database.
    parameter_set
        Bundled identifier, YAML path, mapping, or loaded standard-state
        parameter set.
    dtype, device
        Tensor placement. Omitted values use the configured runtime defaults.
    reference_temperature
        Caloric reference temperature in K. When omitted, the parameter set's
        default is required.
    trainable
        Register heat-capacity coefficients as trainable parameters. Reference
        enthalpies and entropies remain buffers.

    Returns
    -------
    IdealGasPolynomial
        Standard-state module ordered according to ``names``.

    Raises
    ------
    ParameterDatabaseError
        If the source has the wrong model kind or malformed/missing entries.
    KeyError
        If a requested component is absent from the parameter set.
    """
    dtype, device = resolve_tensor_options(dtype, device)
    loaded = load_model_parameters(parameter_set)
    if loaded.model_kind != "standard_state":
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} is {loaded.model_kind!r}, not 'standard_state'"
        )
    records = loaded.parameters.get("components")
    if not isinstance(records, Mapping):
        raise ParameterDatabaseError(f"{loaded.identifier!r} requires a components mapping")
    canonical = tuple(component(name).name for name in names)
    coefficients: list[list[float]] = []
    reference_enthalpy: list[float] = []
    reference_entropy: list[float] = []
    for name in canonical:
        record = records.get(name)
        if not isinstance(record, Mapping):
            raise KeyError(f"no {loaded.identifier} heat-capacity coefficients for: {name}")
        values = record.get("coefficients")
        if (
            not isinstance(values, Sequence)
            or isinstance(values, str)
            or not values
            or any(not isinstance(value, int | float) for value in values)
        ):
            raise ParameterDatabaseError(
                f"{loaded.identifier!r} coefficients for {name!r} must be numeric"
            )
        coefficients.append([float(value) for value in values])
        enthalpy = record.get("reference_enthalpy", 0.0)
        entropy = record.get("reference_entropy", 0.0)
        if not isinstance(enthalpy, int | float) or not isinstance(entropy, int | float):
            raise ParameterDatabaseError(
                f"{loaded.identifier!r} reference properties for {name!r} must be numeric"
            )
        reference_enthalpy.append(float(enthalpy))
        reference_entropy.append(float(entropy))
    if len({len(values) for values in coefficients}) != 1:
        raise ParameterDatabaseError("ideal-gas polynomial orders must match")
    resolved_reference = reference_temperature
    if resolved_reference is None:
        stored_reference = loaded.parameters.get("default_reference_temperature")
        if not isinstance(stored_reference, int | float):
            raise ParameterDatabaseError(
                f"{loaded.identifier!r} requires default_reference_temperature"
            )
        resolved_reference = float(stored_reference)
    dimensionless = torch.tensor(coefficients, dtype=dtype, device=device)
    return IdealGasPolynomial(
        R * dimensionless,
        torch.tensor(reference_enthalpy, dtype=dtype, device=device),
        torch.tensor(reference_entropy, dtype=dtype, device=device),
        reference_temperature=resolved_reference,
        trainable=trainable,
    )


def poling_ideal_gas(
    names: tuple[str, ...] | list[str],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    reference_temperature: float = 273.15,
    trainable: bool = False,
) -> IdealGasPolynomial:
    """Construct a named Poling ideal-gas standard state.

    Parameters
    ----------
    names
        Component names in desired tensor order.
    dtype, device
        Tensor placement.
    reference_temperature
        Caloric reference temperature in K.
    trainable
        Register heat-capacity coefficients as trainable parameters.

    Returns
    -------
    IdealGasPolynomial
        Poling coefficient model for the selected components.

    Raises
    ------
    KeyError
        If a selected component has no frozen Poling coefficient record.

    Notes
    -----
    The tabulated fits cover 50-1000 K for gases through propane and
    200-1000 K for C4-C10. Argon and helium are the monatomic ideal-gas
    limits. This function deliberately does not extrapolate or clip
    temperatures; callers remain responsible for checking those source
    ranges in their application.
    """
    try:
        return ideal_gas_polynomial(
            names,
            "standard-state.poling-2001",
            dtype=dtype,
            device=device,
            reference_temperature=reference_temperature,
            trainable=trainable,
        )
    except KeyError as exc:
        canonical = tuple(component(name).name for name in names)
        raise KeyError(
            f"no frozen Poling heat-capacity coefficients for: {', '.join(canonical)}"
        ) from exc
