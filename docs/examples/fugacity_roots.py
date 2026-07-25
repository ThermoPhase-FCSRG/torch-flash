"""Compare liquid-like and vapor-like EoS roots at one homogeneous state."""

import torch

from torch_flash import (
    ChemicalState,
    component_set,
    configure,
    peng_robinson_1978,
    phase_properties,
)

runtime = configure(device="cpu", dtype=torch.float64)
model = peng_robinson_1978(component_set(("n_butane",)))
state = ChemicalState(
    runtime.tensor(300.0),
    runtime.tensor(2.0e5),
    runtime.tensor([1.0]),
)

liquid_like = phase_properties(model, state, phase="liquid", caloric=False)
vapor_like = phase_properties(model, state, phase="vapor", caloric=False)

print("liquid-like ln(phi) =", liquid_like.log_fugacity_coefficients)
print("vapor-like ln(phi) =", vapor_like.log_fugacity_coefficients)
print("molar volumes [m^3/mol] =", liquid_like.molar_volume, vapor_like.molar_volume)

# These roots have the same overall composition, so this is not a coexistence
# calculation. A flash must solve separate phase compositions.
assert liquid_like.molar_volume < vapor_like.molar_volume
