"""Lumping rules for characterized heavy-end distributions."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise

import torch
from torch import Tensor

from torch_flash.characterization.types import (
    LumpedDistribution,
    SCNDistribution,
)


def equal_weight_lump(
    distribution: SCNDistribution,
    groups: int,
    *,
    properties: Mapping[str, Tensor] | None = None,
) -> LumpedDistribution:
    """Lump contiguous SCN cuts into approximately equal-weight groups.

    Critical properties supplied through ``properties`` use Pedersen's
    weight-average rule (2024, Eqs. 5.29-5.31). Molar mass is mole averaged;
    density, when present, preserves ideal mixed volume.
    """
    count = distribution.carbon_numbers.numel()
    if not isinstance(groups, int) or groups < 1 or groups > count:
        raise ValueError("groups must be an integer from one through the SCN count")
    supplied = {} if properties is None else dict(properties)
    for name, values in supplied.items():
        if not isinstance(name, str) or not name:
            raise ValueError("lumped property names must be non-empty strings")
        if values.shape != distribution.mole_fractions.shape:
            raise ValueError(f"lumped property {name!r} must match the SCN distribution")

    mass = distribution.mole_fractions * distribution.molar_masses
    cumulative = torch.cumsum(mass, dim=0)
    targets = (
        mass.sum()
        * torch.arange(
            1,
            groups,
            dtype=mass.dtype,
            device=mass.device,
        )
        / groups
    )
    boundaries = [0]
    for target in targets:
        candidate = int(torch.searchsorted(cumulative, target).detach()) + 1
        minimum = boundaries[-1] + 1
        maximum = count - (groups - len(boundaries))
        boundaries.append(min(max(candidate, minimum), maximum))
    boundaries.append(count)

    names: list[str] = []
    bounds: list[tuple[int, int]] = []
    fractions: list[Tensor] = []
    molar_masses: list[Tensor] = []
    densities: list[Tensor] = []
    lumped_properties: dict[str, list[Tensor]] = {name: [] for name in supplied}
    for start, stop in pairwise(boundaries):
        current_fraction = distribution.mole_fractions[start:stop]
        current_mass = mass[start:stop]
        total_fraction = current_fraction.sum()
        total_mass = current_mass.sum()
        lower = int(distribution.carbon_numbers[start])
        upper = int(distribution.carbon_numbers[stop - 1])
        names.append(f"C{lower}" if lower == upper else f"C{lower}-C{upper}")
        bounds.append((lower, upper))
        fractions.append(total_fraction)
        molar_masses.append(total_mass / total_fraction)
        if distribution.densities is not None:
            densities.append(
                total_mass / torch.sum(current_mass / distribution.densities[start:stop])
            )
        for name, values in supplied.items():
            lumped_properties[name].append(
                torch.sum(current_mass * values[start:stop]) / total_mass
            )
    return LumpedDistribution(
        tuple(names),
        tuple(bounds),
        torch.stack(fractions),
        torch.stack(molar_masses),
        None if distribution.densities is None else torch.stack(densities),
        {name: torch.stack(values) for name, values in lumped_properties.items()},
    )
