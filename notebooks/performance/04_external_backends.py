# %% [markdown]
# # Independent comparisons: teqp, ThermoPack, and NeqSim
#
# Independent implementations are valuable for catching algebra and unit
# errors. This study compares:
#
# 1. homogeneous canonical PR values against NIST `teqp`;
# 2. one methane/n-butane TP flash against ThermoPack and NeqSim;
# 3. frozen CSV baselines used by tests, so CI does not require those external
#    runtimes.
#
# Differences in component databases and PR alpha conventions prevent a
# bitwise comparison of complete flashes.
#
# Sources and exact software records:
#
# - Peng–Robinson model: <https://doi.org/10.1021/i160057a011>
# - teqp canonical cubic formulation:
#   <https://pages.nist.gov/teqp-docs/en/main/models/cubics.html>
# - ThermoPack 2.2.3:
#   <https://pypi.org/project/thermopack/2.2.3/>
# - neqsim-python 3.16.0:
#   <https://pypi.org/project/neqsim/3.16.0/>

# %%
from importlib.metadata import version
from pathlib import Path
from statistics import median
from timeit import repeat

import pandas as pd
import torch
from IPython.display import display

from torch_flash import (
    ChemicalState,
    component_set,
    peng_robinson_1978,
    two_phase_flash,
)
from torch_flash.backends import TeqpBackend

torch.set_default_dtype(torch.float64)
repo_root = next(
    candidate
    for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "pyproject.toml").is_file()
)
print(
    {
        package: version(package)
        for package in ("torch-flash", "teqp", "thermopack", "neqsim")
    }
)

# %% [markdown]
# ## Canonical PR homogeneous roots
#
# The in-package PR constants retain the full critical-point precision. With
# the same critical constants and acentric factors, they can be compared
# directly with `teqp.canonical_PR`.

# %%
components = component_set(("methane", "n_butane"))
torch_model = peng_robinson_1978(components)
teqp_model = TeqpBackend.canonical_peng_robinson(components)
states = pd.read_csv(repo_root / "tests" / "data" / "teqp_pr_binary.csv")

records = []
for row in states.itertuples(index=False):
    temperature = torch.tensor(row.temperature_K)
    pressure = torch.tensor(row.pressure_Pa)
    composition = torch.tensor([row.x_methane, row.x_n_butane])
    phase = row.phase
    z_torch = torch_model.select_z(temperature, pressure, composition, phase).item()
    z_teqp = teqp_model.select_z(temperature, pressure, composition, phase).item()
    phi_torch = torch_model.log_fugacity_coefficients(
        temperature, pressure, composition, phase
    )
    phi_teqp = teqp_model.log_fugacity_coefficients(
        temperature, pressure, composition, phase
    )
    records.append(
        {
            "T [K]": row.temperature_K,
            "P [MPa]": row.pressure_Pa / 1.0e6,
            "phase": phase,
            "|dZ|": abs(z_torch - z_teqp),
            "max |d ln(phi)|": (phi_torch - phi_teqp).abs().max().item(),
        }
    )
canonical_comparison = pd.DataFrame(records)
display(canonical_comparison)
assert canonical_comparison["|dZ|"].max() < 2.0e-12
assert canonical_comparison["max |d ln(phi)|"].max() < 2.0e-11

# %% [markdown]
# ## Complete TP flash

# %%
state = ChemicalState(
    torch.tensor(270.0),
    torch.tensor(3.0e6),
    torch.tensor([0.5, 0.5]),
)
ours = two_phase_flash(torch_model, state, check_stability=False)
ours_row = {
    "implementation": "torch-flash PR78",
    "beta_vapor": ours.phase_fractions[1].item(),
    "x_methane": ours.phases[0].composition[0].item(),
    "y_methane": ours.phases[1].composition[0].item(),
}

rows = [ours_row]
for filename, label in (
    ("thermopack_pr_flash.csv", "ThermoPack 2.2.3 frozen"),
    ("neqsim_pr_flash.csv", "NeqSim 3.16.0 frozen"),
):
    baseline = pd.read_csv(repo_root / "tests" / "data" / filename).iloc[0]
    rows.append(
        {
            "implementation": label,
            "beta_vapor": baseline.beta_vapor,
            "x_methane": baseline.x_methane,
            "y_methane": baseline.y_methane,
        }
    )
comparison = pd.DataFrame(rows).set_index("implementation")
comparison

# %%
from thermopack.cubic import cubic

thermopack_model = cubic("C1,NC4", eos="PR", mixing="vdW", alpha="Classic")
thermopack_result = thermopack_model.two_phase_tpflash(270.0, 3.0e6, [0.5, 0.5])
live_thermopack = {
    "beta_vapor": thermopack_result.betaV,
    "x_methane": thermopack_result.x[0],
    "y_methane": thermopack_result.y[0],
}
live_thermopack

# %%
import os

os.environ.setdefault("NEQSIM_JVM_ARGS", "--enable-native-access=ALL-UNNAMED")

from neqsim.thermo import TPflash, addComponent, fluid

neqsim_model = fluid(
    "pr", temperature=270.0, pressure=30.0
)  # NeqSim pressure input is bar
neqsim_model.useVolumeCorrection(False)
addComponent(neqsim_model, "methane", 0.5)
addComponent(neqsim_model, "n-butane", 0.5)
neqsim_model.setMixingRule("classic")
TPflash(neqsim_model)
phases = {
    neqsim_model.getPhase(index).getPhaseTypeName(): neqsim_model.getPhase(index)
    for index in range(neqsim_model.getNumberOfPhases())
}
live_neqsim = {
    "beta_vapor": phases["gas"].getBeta(),
    "x_methane": phases["oil"].getComponent(0).getx(),
    "y_methane": phases["gas"].getComponent(0).getx(),
}
live_neqsim

# %%
for key in live_thermopack:
    assert (
        abs(live_thermopack[key] - comparison.loc["ThermoPack 2.2.3 frozen", key])
        < 1e-7
    )
    assert abs(live_neqsim[key] - comparison.loc["NeqSim 3.16.0 frozen", key]) < 1e-7

# %% [markdown]
# ## Scalar TP-flash timing against ThermoPack
#
# Both models are warmed before timing. The torch-flash call skips its optional
# preliminary tangent-plane test because this state is already known to split;
# ThermoPack controls its own phase-detection workflow. The measurements are
# therefore useful implementation-level timings, not a claim of identical
# solver work.

# %%
for _ in range(3):
    two_phase_flash(torch_model, state, check_stability=False)
    thermopack_model.two_phase_tpflash(270.0, 3.0e6, [0.5, 0.5])

number = 10
torch_seconds = (
    median(
        repeat(
            lambda: two_phase_flash(torch_model, state, check_stability=False),
            repeat=5,
            number=number,
        )
    )
    / number
)
thermopack_seconds = (
    median(
        repeat(
            lambda: thermopack_model.two_phase_tpflash(270.0, 3.0e6, [0.5, 0.5]),
            repeat=5,
            number=number,
        )
    )
    / number
)
pd.Series(
    {
        "torch-flash median [s/flash]": torch_seconds,
        "ThermoPack median [s/flash]": thermopack_seconds,
        "torch-flash / ThermoPack": torch_seconds / thermopack_seconds,
    }
)

# %% [markdown]
# The torch-flash result lies close to both independent packages. Residual
# differences are expected because each package owns its component constants
# and model defaults. This notebook explicitly disables NeqSim volume
# translation and chooses classic mixing to make the comparison meaningful.
# ThermoPack is expected to be much faster for one scalar flash in this initial
# version. The PyTorch design instead targets differentiability and batched
# state evaluation; iterative-flash vectorization remains an optimization
# milestone.
