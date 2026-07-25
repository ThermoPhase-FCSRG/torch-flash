"""Evaluate an original-UNIFAC activity model and differentiate it."""

import torch

from torch_flash import activity_model, configure

runtime = configure(device="cpu", dtype=torch.float64)
model = activity_model(
    "activity.unifac-original-public-2026",
    ("ethanol", "n_heptane"),
)
temperature = runtime.tensor(323.15, requires_grad=True)  # K
composition = runtime.tensor([0.40, 0.60], requires_grad=True)

log_gamma = model.log_activity_coefficients(temperature, composition)
excess_gibbs_rt = model.excess_gibbs_rt(temperature, composition)
gradient = torch.autograd.grad(excess_gibbs_rt, (temperature, composition))

print("ln(gamma) =", log_gamma)
print("g^E/(RT) =", excess_gibbs_rt)
print("d[g^E/(RT)]/dT =", gradient[0])
print("d[g^E/(RT)]/dx =", gradient[1])
