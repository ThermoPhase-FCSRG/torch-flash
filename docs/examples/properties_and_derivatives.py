"""Evaluate homogeneous phase properties and their TP derivatives."""

import torch

from torch_flash import (
    ChemicalState,
    component_set,
    configure,
    peng_robinson_1978,
    phase_properties,
    state_derivatives,
)

runtime = configure(device="cpu", dtype=torch.float64)
model = peng_robinson_1978(component_set(("methane", "n_butane")))
state = ChemicalState(
    temperature=runtime.tensor(300.0),  # K
    pressure=runtime.tensor(5.0e6),  # Pa
    composition=runtime.tensor([0.70, 0.30]),  # mol/mol
)

# This evaluates the supplied homogeneous state; it does not run a flash.
properties = phase_properties(model, state, phase="stable")
derivatives = state_derivatives(model, state, phase="stable")

print(f"Z = {float(properties.compressibility_factor):.6f}")
print(f"molar volume = {float(properties.molar_volume):.6e} m^3/mol")
print("fugacities [Pa] =", properties.fugacities)
print("d ln(f_i/p°) / dT [1/K] =", derivatives.dlog_fugacity_dtemperature)
print("dV_m / dP [m^3/(mol Pa)] =", derivatives.dmolar_volume_dpressure)

# For a binary mixture, the single independent coordinate is x_methane;
# x_n-butane = 1 - x_methane.
print(
    "d mu_i / d x_methane [J/mol] =",
    derivatives.dchemical_potential_dindependent_composition[:, 0],
)
