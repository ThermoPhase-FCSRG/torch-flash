"""Split, characterize, and lump a petroleum plus fraction."""

import torch

from torch_flash import (
    equal_weight_lump,
    pedersen_cubic_properties,
    pedersen_density_split,
    pedersen_logarithmic_split,
)

# SI units: the C20+ fraction has z+=0.00833 and M+=0.377 kg/mol.
split = pedersen_logarithmic_split(
    plus_mole_fraction=0.00833,
    plus_molar_mass=torch.tensor(0.377, dtype=torch.float64),
    first_carbon_number=20,
    max_carbon_number=80,
)
with_density = pedersen_density_split(
    split,
    plus_density=873.0,  # kg/m^3
    anchor_density=841.0,
    anchor_carbon_number=19,
)
pr_properties = pedersen_cubic_properties(with_density, "PR")
cuts = equal_weight_lump(
    with_density,
    groups=3,
    properties={
        "critical_temperature": pr_properties.critical_temperature,
        "critical_pressure": pr_properties.critical_pressure,
        "acentric_factor": pr_properties.acentric_factor,
    },
)

print("pseudo-component names =", cuts.names)
print("mole fractions =", cuts.mole_fractions)
print("molar masses [kg/mol] =", cuts.molar_masses)
print("densities [kg/m^3] =", cuts.densities)
torch.testing.assert_close(cuts.mole_fractions.sum(), split.total_mole_fraction)
