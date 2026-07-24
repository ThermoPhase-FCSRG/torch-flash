from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from torch_flash.backends import TeqpBackend


@pytest.mark.external
def test_eoscg_2015_all_published_pressure_verification_states(
    not_cleared_data: Path,
):
    """Reproduce all 30 pressure entries in Gernert--Span Table 8."""
    pytest.importorskip("teqp")
    path = not_cleared_data / "gernert_span_2016_eoscg_table8.csv"
    with path.open(newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    assert len(rows) == 30
    predictions = []
    references = []
    models: dict[tuple[str, str], TeqpBackend] = {}
    for row in rows:
        names = (row["component_1"], row["component_2"])
        if names not in models:
            models[names] = TeqpBackend.eoscg_2015(names)
        model = models[names]
        density = float(row["molar_density_mol_m3"])
        predictions.append(
            model.pressure(
                torch.tensor(float(row["temperature_K"]), dtype=torch.float64),
                torch.tensor(1.0 / density, dtype=torch.float64),
                torch.tensor([float(row["x1"]), float(row["x2"])], dtype=torch.float64),
            )
            / 1.0e6
        )
        references.append(float(row["pressure_MPa"]))
    # The table is rounded to six decimals at low density and four at high
    # density. This tolerance is tighter than its final printed digit.
    torch.testing.assert_close(
        torch.stack(predictions),
        torch.tensor(references, dtype=torch.float64),
        rtol=4.0e-6,
        atol=5.0e-7,
    )
