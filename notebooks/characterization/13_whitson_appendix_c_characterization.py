# %% [markdown]
# # Whitson Appendix C: petroleum-fluid characterization
#
# This study reconstructs the two equation-of-state applications in Appendix C
# of C. H. Whitson and M. R. Brulé, *Phase Behavior*, SPE Monograph 20 (2000),
# ISBN 978-1-55563-087-4:
#
# 1. the 15-component gas-condensate characterization in Tables C-7/C-8; and
# 2. the 13-component reservoir-oil PR characterization in Tables C-9/C-11.
#
# All values below are embedded from the published tables so the notebook does
# not depend on untracked local reference material.
#
# **Source anomaly.** Table C-7 visually assigns the N₂ row the CO₂ molecular
# weight and critical constants, and vice versa. Tables A-1 and C-10 agree with
# each other and with the chemical identities. We therefore retain the C-7
# compositions but use the consistent A-1/C-10 N₂ and CO₂ property rows.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display

from torch_flash.components import ComponentSet
from torch_flash.envelope import saturation_point
from torch_flash.eos import peng_robinson_1978

torch.set_default_dtype(torch.float64)
plt.style.use("seaborn-v0_8-whitegrid")
PSIA_TO_PA = 6894.757293168
R = 8.31446261815324


def appendix_c_kij(
    size: int,
    nitrogen: list[float],
    carbon_dioxide: list[float],
    methane_heavy: list[float],
) -> torch.Tensor:
    """Build the sparse, symmetric BIP matrices printed in Tables C-8/C-11."""
    kij = torch.zeros((size, size))
    for column in range(size):
        kij[0, column] = kij[column, 0] = nitrogen[column]
        kij[1, column] = kij[column, 1] = carbon_dioxide[column]
    heavy_start = size - len(methane_heavy)
    for column, value in enumerate(methane_heavy, start=heavy_start):
        kij[2, column] = kij[column, 2] = value
    return kij


def whitson_translation(
    critical_temperature: torch.Tensor,
    critical_pressure: torch.Tensor,
    s_over_b: torch.Tensor,
) -> torch.Tensor:
    """Convert printed s=c/b to torch-flash's additive ΔV=-c convention."""
    b = 0.07780 * R * critical_temperature / critical_pressure
    return -s_over_b * b


# %% [markdown]
# ## Gas condensate: Tables C-7 and C-8
#
# The unregressed characterization predicts a dew point of 3535 psia at the
# reservoir temperature of 186°F, about 500 psi below the measured 4015 psia.
# Appendix C then multiplies methane/heavy-fraction BIPs by 2.09. We evaluate
# both definitions without further tuning.

# %%
gas_names = (
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
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
)
gas_z = torch.tensor(
    [
        0.0018,
        0.0013,
        0.6192,
        0.1408,
        0.0835,
        0.0097,
        0.0341,
        0.0084,
        0.0148,
        0.0179,
        0.024227,
        0.028921,
        0.012852,
        0.002367,
        0.000132,
    ]
)
gas_molar_mass = (
    torch.tensor(
        [
            28.01,
            44.01,
            16.04,
            30.07,
            44.10,
            58.12,
            58.12,
            72.15,
            72.15,
            86.18,
            98.55,
            135.84,
            206.65,
            319.83,
            500.00,
        ]
    )
    / 1000.0
)
gas_tc = (
    torch.tensor(
        [
            227.3,
            547.6,
            343.0,
            549.8,
            665.7,
            734.7,
            765.3,
            828.8,
            845.4,
            913.4,
            1004.4,
            1135.1,
            1309.6,
            1490.2,
            1670.4,
        ]
    )
    / 1.8
)
gas_pc = (
    torch.tensor(
        [
            493.0,
            1070.6,
            667.8,
            707.8,
            616.3,
            529.1,
            550.7,
            490.4,
            488.6,
            436.9,
            441.5,
            362.7,
            266.9,
            191.1,
            140.3,
        ]
    )
    * PSIA_TO_PA
)
gas_omega = torch.tensor(
    [
        0.0450,
        0.2310,
        0.0115,
        0.0908,
        0.1454,
        0.1756,
        0.1928,
        0.2273,
        0.2510,
        0.2957,
        0.2864,
        0.3882,
        0.5756,
        0.8316,
        1.1188,
    ]
)
gas_s_over_b = torch.tensor(
    [
        -0.0577,
        -0.1752,
        -0.1651,
        -0.1070,
        -0.0848,
        -0.0686,
        -0.0686,
        -0.0410,
        -0.0410,
        -0.0154,
        0.0322,
        0.0552,
        0.1075,
        0.1544,
        0.1599,
    ]
)
gas_components = ComponentSet(
    gas_names,
    gas_tc,
    gas_pc,
    gas_omega,
    gas_molar_mass,
)
gas_kij = appendix_c_kij(
    15,
    [
        0,
        0,
        0.025,
        0.010,
        0.090,
        0.095,
        0.095,
        0.100,
        0.110,
        0.110,
        0.110,
        0.110,
        0.110,
        0.110,
        0.110,
    ],
    [
        0,
        0,
        0.105,
        0.130,
        0.125,
        0.120,
        0.115,
        0.115,
        0.115,
        0.115,
        0.115,
        0.115,
        0.115,
        0.115,
        0.115,
    ],
    [0.030, 0.042, 0.058, 0.076, 0.095],
)
gas_translation = whitson_translation(gas_tc, gas_pc, gas_s_over_b)
gas_models = {
    "Table C-8": peng_robinson_1978(
        gas_components,
        kij=gas_kij,
        volume_translation=gas_translation,
    ),
    "C-8 methane/F \N{MULTIPLICATION SIGN} 2.09": peng_robinson_1978(
        gas_components,
        kij=gas_kij
        + (2.09 - 1.0)
        * (
            torch.nn.functional.one_hot(torch.tensor(2), 15)[:, None]
            * torch.cat((torch.zeros(10), gas_kij[2, 10:]))[None, :]
            + torch.cat((torch.zeros(10), gas_kij[2, 10:]))[:, None]
            * torch.nn.functional.one_hot(torch.tensor(2), 15)[None, :]
        ),
        volume_translation=gas_translation,
    ),
}
assert torch.isclose(gas_z.sum(), torch.tensor(1.0))

# %%
reservoir_temperature = torch.tensor((186.0 + 459.67) / 1.8)
dew_rows = []
for model_name, model in gas_models.items():
    initial = 3535.0 if model_name == "Table C-8" else 4015.0
    point = saturation_point(
        model,
        reservoir_temperature,
        gas_z,
        "dew",
        initial_pressure=torch.tensor(initial * PSIA_TO_PA),
    )
    dew_rows.append(
        {
            "model": model_name,
            "dew_pressure_psia": float(point.pressure / PSIA_TO_PA),
            "converged": point.converged,
        }
    )
dew_comparison = pd.DataFrame(dew_rows)
dew_comparison.loc[len(dew_comparison)] = ["Appendix C unregressed text", 3535.0, True]
dew_comparison.loc[len(dew_comparison)] = ["measured", 4015.0, True]
display(dew_comparison)

# %%
temperature_grid = torch.linspace(330.0, 375.0, 7)
dew_trace_rows = []
for model_name, model in gas_models.items():
    pressure_guess = torch.tensor(3000.0 * PSIA_TO_PA)
    for temperature in temperature_grid:
        point = saturation_point(
            model,
            temperature,
            gas_z,
            "dew",
            initial_pressure=pressure_guess,
        )
        dew_trace_rows.append(
            {
                "model": model_name,
                "temperature_K": float(temperature),
                "pressure_psia": float(point.pressure / PSIA_TO_PA),
                "converged": point.converged,
            }
        )
        pressure_guess = point.pressure.detach()
dew_trace = pd.DataFrame(dew_trace_rows)
display(dew_trace.groupby("model")["converged"].all())

fig, axis = plt.subplots(figsize=(7.5, 4.8))
for model_name, group in dew_trace.groupby("model"):
    axis.plot(
        group["temperature_K"], group["pressure_psia"], marker="o", label=model_name
    )
axis.scatter(
    [float(reservoir_temperature)],
    [4015.0],
    color="black",
    marker="x",
    s=75,
    label="measured reservoir dew point",
)
axis.set(
    xlabel="Temperature / K",
    ylabel="Dew pressure / psia",
    title="Appendix C gas condensate: PR78 dew branch",
)
axis.legend()
plt.show()

# %% [markdown]
# The 2.09 multiplier moves the dew branch in the intended direction but does
# not force an exact 4015-psia result when only the rounded Table C-7 constants
# are available. This is a reproducibility limit of the printed precision.

# %% [markdown]
# ## Reservoir oil: Tables C-9 through C-11
#
# The reported state is the 2600-psia bubble point at 220°F. This is a strong
# check because Table C-9 publishes the entire incipient-gas composition.

# %%
oil_names = (
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
    "F1",
    "F2",
    "F3",
)
oil_z = (
    torch.tensor(
        [
            0.16,
            0.91,
            36.47,
            9.67,
            6.95,
            1.44,
            3.93,
            1.44,
            1.41,
            4.33,
            15.91,
            14.28,
            3.11,
        ]
    )
    / 100.0
)
oil_molar_mass = (
    torch.tensor(
        [
            28.01,
            44.01,
            16.04,
            30.07,
            44.10,
            58.12,
            58.12,
            72.15,
            72.15,
            86.18,
            120.08,
            255.96,
            545.00,
        ]
    )
    / 1000.0
)
oil_tc = (
    torch.tensor(
        [
            227.3,
            547.6,
            343.0,
            549.8,
            665.7,
            734.7,
            765.3,
            828.8,
            845.4,
            913.4,
            1086.6,
            1401.5,
            1707.3,
        ]
    )
    / 1.8
)
oil_pc = (
    torch.tensor(
        [
            493.0,
            1070.6,
            667.8,
            707.8,
            616.3,
            529.1,
            550.7,
            490.4,
            488.6,
            436.9,
            397.1,
            230.0,
            137.0,
        ]
    )
    * PSIA_TO_PA
)
oil_omega = torch.tensor(
    [
        0.0450,
        0.2310,
        0.0115,
        0.0908,
        0.1454,
        0.1756,
        0.1928,
        0.2273,
        0.2510,
        0.2957,
        0.3419,
        0.6866,
        1.2213,
    ]
)
oil_s_over_b = torch.tensor(
    [
        -0.1930,
        -0.0820,
        -0.1590,
        -0.1130,
        -0.0860,
        -0.0840,
        -0.0670,
        -0.0610,
        -0.0390,
        -0.0080,
        0.0403,
        0.1255,
        0.1326,
    ]
)
oil_components = ComponentSet(
    oil_names,
    oil_tc,
    oil_pc,
    oil_omega,
    oil_molar_mass,
)
oil_kij = appendix_c_kij(
    13,
    [0, 0, 0.025, 0.010, 0.090, 0.095, 0.095, 0.100, 0.110, 0.110, 0.110, 0.110, 0.110],
    [0, 0, 0.105, 0.130, 0.125, 0.120, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115],
    [0.035, 0.063, 0.092],
)
oil_model = peng_robinson_1978(
    oil_components,
    kij=oil_kij,
    volume_translation=whitson_translation(oil_tc, oil_pc, oil_s_over_b),
)
oil_bubble = saturation_point(
    oil_model,
    torch.tensor((220.0 + 459.67) / 1.8),
    oil_z,
    "bubble",
    initial_pressure=torch.tensor(2600.0 * PSIA_TO_PA),
)
oil_reference_y = (
    torch.tensor(
        [
            0.52,
            1.31,
            77.13,
            10.16,
            4.87,
            0.77,
            1.85,
            0.51,
            0.46,
            1.00,
            1.35,
            0.0623,
            0.000050,
        ]
    )
    / 100.0
)
print(f"Calculated bubble pressure: {float(oil_bubble.pressure / PSIA_TO_PA):.2f} psia")
print("Table C-9 bubble pressure: 2600.00 psia")

# %%
oil_comparison = pd.DataFrame(
    {
        "component": oil_names,
        "Table C-9 gas mole_percent": 100.0 * oil_reference_y.numpy(),
        "torch-flash gas mole_percent": 100.0
        * oil_bubble.incipient_composition.numpy(),
    }
)
display(oil_comparison)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
positions = np.arange(len(oil_names))
axes[0].semilogy(
    positions,
    np.maximum(100.0 * oil_reference_y.numpy(), 1.0e-6),
    "kx",
    label="Table C-9",
)
axes[0].semilogy(
    positions,
    np.maximum(100.0 * oil_bubble.incipient_composition.numpy(), 1.0e-6),
    "o",
    fillstyle="none",
    label="torch-flash",
)
axes[0].set(
    xticks=positions,
    xticklabels=oil_names,
    xlabel="Component",
    ylabel="Incipient-gas composition / mol%",
    title="Reservoir-oil bubble gas",
)
axes[0].tick_params(axis="x", rotation=70)
axes[0].legend()
axes[1].loglog(
    100.0 * oil_reference_y,
    100.0 * oil_bubble.incipient_composition,
    "o",
)
limits = (1.0e-5, 100.0)
axes[1].plot(limits, limits, "k--", linewidth=0.8)
axes[1].set(
    xlim=limits,
    ylim=limits,
    aspect="equal",
    xlabel="Table C-9 / mol%",
    ylabel="torch-flash / mol%",
    title="Incipient-gas parity",
)
plt.show()

# %% [markdown]
# ## Conclusions
#
# - The unregressed gas-condensate dew pressure and the reservoir-oil bubble
#   pressure are recovered from independently encoded Appendix C parameters.
# - The full reservoir-oil incipient-gas composition agrees over more than six
#   orders of magnitude.
# - Printed precision is sufficient for the oil bubble example but not for an
#   exact reconstruction of the gas-condensate regression multiplier.
