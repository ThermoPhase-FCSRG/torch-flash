# %% [markdown]
# # Whitson Rachford–Rice Contest: all 10,008 cases
#
# The contest stresses extreme compositions and K-values. Its pass criteria
# are tighter than a typical engineering flash calculation. `torch-flash`
# therefore provides two paths:
#
# - a batched, differentiable PyTorch solver for normal modeling workloads;
# - a NumPy compatibility wrapper that delegates the exact contest path to
#   `chemicals` 1.5.2's MIT-licensed double-double Leibovici–Neoschil solver.
#
# The original contest corpus is optional and intentionally not redistributed.
#
# Sources:
#
# - original material-balance equation:
#   <https://doi.org/10.2118/952327-G>
# - bounded Leibovici–Neoschil formulation:
#   <https://doi.org/10.1016/0378-3812(92)85069-K>
# - delegated `chemicals` solver and license:
#   <https://github.com/CalebBell/chemicals>
# - contest corpus, pinned at commit `503b92f`:
#   <https://github.com/WhitsonAS/Rachford-Rice-Contest/tree/503b92f1b2847c4459326841f538739bcb9d629f>

# %%
import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from IPython.display import display

from torch_flash import rachford_rice, rachford_rice_numpy

torch.set_default_dtype(torch.float64)
repo_root = next(
    candidate
    for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "pyproject.toml").is_file()
)
configured_contest_root = os.environ.get("TORCH_FLASH_WHITSON_DATA")
contest_root = (
    Path(configured_contest_root)
    if configured_contest_root
    else repo_root / "tests" / "data" / "whitson"
)
contest_source = (
    "TORCH_FLASH_WHITSON_DATA"
    if configured_contest_root
    else "optional tests/data/whitson fallback"
)
print("contest corpus source:", contest_source, "available:", contest_root.exists())


# %%
def normalized_contest_residuals(z, k, vapor, liquid, beta_v, beta_l):
    """Return the five normalized residuals used by the contest checker."""
    n = z.size
    phase_tolerance = 1.0e-15 + n * np.finfo(float).eps
    return np.asarray(
        [
            abs(1.0 - vapor.sum()) / phase_tolerance,
            abs(1.0 - liquid.sum()) / phase_tolerance,
            abs(beta_v + beta_l - 1.0) / (abs(beta_v) + abs(beta_l) + 1.0) / 1.0e-15,
            np.max(
                np.abs(beta_v * vapor + beta_l * liquid - z)
                / (np.abs(beta_v * vapor) + np.abs(beta_l * liquid) + z)
            )
            / 1.0e-15,
            np.max(np.abs(vapor - k * liquid) / (np.abs(vapor) + np.abs(k * liquid)))
            / 1.0e-15,
        ]
    )


# %%
if contest_root.exists():
    with (contest_root / "compositions.csv").open() as stream:
        composition_rows = list(csv.reader(stream))[1:]
    with (contest_root / "k-values.csv").open() as stream:
        k_rows = list(csv.reader(stream))[1:]

    maxima = []
    failures = []
    started = time.perf_counter()
    for case, (composition_row, k_row) in enumerate(
        zip(composition_rows, k_rows, strict=True), 1
    ):
        n = int(float(composition_row[0]))
        z = np.asarray(composition_row[1 : n + 1], dtype=float)
        z /= z.sum()
        k = np.asarray(k_row[:n], dtype=float)
        _, vapor, liquid, beta_v, beta_l = rachford_rice_numpy(z, k)
        residuals = normalized_contest_residuals(z, k, vapor, liquid, beta_v, beta_l)
        worst = residuals.max()
        maxima.append(worst)
        lower = 1.0 / (1.0 - k.max())
        upper = 1.0 / (1.0 - k.min())
        if worst > 1.0 or not lower < beta_v < upper:
            failures.append((case, worst))

    elapsed = time.perf_counter() - started
    contest_summary = pd.Series(
        {
            "cases": len(maxima),
            "failures": len(failures),
            "worst normalized residual": max(maxima),
            "wall time [s]": elapsed,
            "cases/s": len(maxima) / elapsed,
        }
    )
    display(contest_summary)
    assert len(maxima) == 10_008 and not failures
else:
    print(
        "Skipped: set TORCH_FLASH_WHITSON_DATA to a contest checkout "
        "to reproduce this cell."
    )

# %% [markdown]
# ## Differentiable path
#
# This ordinary three-component case demonstrates gradients through the
# PyTorch material-balance iterations. It is not claimed to meet the contest's
# double-double acceptance threshold.

# %%
z = torch.tensor([0.50, 0.30, 0.20], requires_grad=True)
k = torch.tensor([3.0, 0.8, 0.2])
split = rachford_rice(z, k)
gradient = torch.autograd.grad(split.vapor_fraction, z)[0]
pd.Series(
    {
        "vapor fraction": split.vapor_fraction.item(),
        "RR residual": split.residual.item(),
        "iterations": split.iterations,
        "d betaV / dz0": gradient[0].item(),
        "d betaV / dz1": gradient[1].item(),
        "d betaV / dz2": gradient[2].item(),
    }
)

# %% [markdown]
# The native solver permits negative-flash roots between the nearest
# denominator singularities. A physical TP-flash layer must still decide
# stability and phase existence; Rachford–Rice alone does not do so.
