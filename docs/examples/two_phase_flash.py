"""Solve a known two-phase methane/n-butane TP state."""

import torch

from torch_flash import (
    ChemicalState,
    component_set,
    configure,
    peng_robinson_1978,
    two_phase_flash,
)

runtime = configure(device="cpu", dtype=torch.float64)
model = peng_robinson_1978(component_set(("methane", "n_butane")))
state = ChemicalState(
    temperature=runtime.tensor(270.0),  # K
    pressure=runtime.tensor(3.0e6),  # Pa
    composition=runtime.tensor([0.50, 0.50]),  # overall z
)

# This example deliberately solves a state already known to be on a
# two-phase branch. Leave check_stability=True when the phase count is unknown.
result = two_phase_flash(
    model,
    state,
    check_stability=False,
    tolerance=1.0e-10,
    raise_on_failure=True,
)

liquid, vapor = result.phases
print("phase fractions [liquid, vapor] =", result.phase_fractions)
print("liquid x =", liquid.composition)
print("vapor y =", vapor.composition)
print("maximum log-fugacity residual =", float(result.residual_norm))

# Always check material balance and the numerical result before using it.
reconstructed_feed = (
    result.phase_fractions[0] * liquid.composition + result.phase_fractions[1] * vapor.composition
)
torch.testing.assert_close(reconstructed_feed, state.composition, atol=1.0e-9, rtol=0.0)
assert result.converged
