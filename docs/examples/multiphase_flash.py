"""Solve a fixed three-phase flash initialized from a published example."""

import torch

from torch_flash import ChemicalState, ComponentSet, multiphase_flash, peng_robinson_1978

# Pedersen, Christensen, and Shaikh (2024), Tables 6.5-6.6. The published
# constants and phase compositions are rounded, so they are used here as a
# physically informed initializer rather than an exact numerical baseline.
names = (
    "nitrogen",
    "carbon_dioxide",
    "methane",
    "ethane",
    "propane",
    "isobutane",
    "n_butane",
    "isopentane",
    "n_pentane",
    "n_hexane",
    "n_heptane",
    "n_octane",
    "n_decane",
)
components = ComponentSet(
    names,
    (
        torch.tensor(
            [
                -147.0,
                31.1,
                -82.6,
                32.3,
                96.7,
                135.0,
                152.1,
                187.3,
                196.5,
                234.3,
                280.4,
                352.5,
                473.1,
            ],
            dtype=torch.float64,
        )
        + 273.15
    ),
    torch.tensor(
        [
            33.94,
            73.76,
            46.00,
            48.84,
            42.46,
            36.48,
            38.00,
            33.84,
            33.74,
            29.69,
            26.72,
            21.29,
            16.67,
        ],
        dtype=torch.float64,
    )
    * 1.0e5,
    torch.tensor(
        [
            0.040,
            0.225,
            0.008,
            0.098,
            0.152,
            0.176,
            0.193,
            0.227,
            0.251,
            0.296,
            0.373,
            0.518,
            0.803,
        ],
        dtype=torch.float64,
    ),
    torch.ones(13, dtype=torch.float64),
)

feed = torch.tensor(
    [0.08, 2.01, 82.51, 5.81, 2.88, 0.56, 1.24, 0.52, 0.60, 0.72, 1.66, 0.91, 0.49],
    dtype=torch.float64,
)
vapor_guess = torch.tensor(
    [
        0.18,
        1.08,
        96.45,
        1.86,
        0.33,
        0.03,
        0.05,
        0.01,
        0.01,
        1.0e-4,
        1.0e-4,
        1.0e-4,
        1.0e-4,
    ],
    dtype=torch.float64,
)
liquid_1_guess = torch.tensor(
    [0.08, 1.88, 87.95, 5.28, 2.17, 0.38, 0.76, 0.28, 0.29, 0.29, 0.48, 0.14, 0.01],
    dtype=torch.float64,
)
liquid_2_guess = torch.tensor(
    [0.05, 2.36, 75.66, 7.28, 4.00, 0.81, 1.83, 0.79, 0.93, 1.14, 2.73, 1.54, 0.87],
    dtype=torch.float64,
)

reference = liquid_1_guess / liquid_1_guess.sum()
initial_k = torch.stack(
    (
        vapor_guess / vapor_guess.sum() / reference,
        liquid_2_guess / liquid_2_guess.sum() / reference,
    )
)
state = ChemicalState(
    torch.tensor(201.15, dtype=torch.float64),
    torch.tensor(5.2e6, dtype=torch.float64),
    feed,
)
result = multiphase_flash(
    peng_robinson_1978(components),
    state,
    initial_k_values=initial_k,
    tolerance=1.0e-10,
    max_iterations=30,
)

print("phase fractions =", result.phase_fractions)
print("phase kinds =", result.phase_kinds)
print("maximum log-fugacity residual =", float(result.residual_norm))
assert result.converged
assert result.nphases == 3

reconstructed_feed = sum(
    fraction * phase.composition
    for fraction, phase in zip(result.phase_fractions, result.phases, strict=True)
)
torch.testing.assert_close(reconstructed_feed, state.composition, atol=2.0e-8, rtol=0.0)
