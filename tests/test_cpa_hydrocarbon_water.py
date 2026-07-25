from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from torch_flash import binary_phase_equilibrium_point, cpa_yan_2009

DTYPE = torch.float64


@pytest.mark.serial
def test_cpa_light_hydrocarbon_water_against_four_experimental_systems(
    not_cleared_data: Path,
):
    """Check four light-hydrocarbon systems against digitized experiment."""
    with (not_cleared_data / "cpa_yan_2009_water_content_digitized.csv").open() as stream:
        all_rows = list(csv.DictReader(stream))
    deviations = []
    for hydrocarbon in ("methane", "ethane", "propane", "n_butane"):
        rows = [
            row
            for row in all_rows
            if row["component"] == hydrocarbon and row["temperature_K"] == "344.26"
        ]
        row = rows[len(rows) // 2]
        measured = float(row["water_mole_fraction"])
        model = cpa_yan_2009((hydrocarbon, "water"))
        hydrocarbon_phase = "vapor" if hydrocarbon in ("methane", "ethane") else "liquid"
        point = binary_phase_equilibrium_point(
            model,
            torch.tensor(float(row["temperature_K"]), dtype=DTYPE),
            torch.tensor(float(row["pressure_bar"]) * 1.0e5, dtype=DTYPE),
            torch.tensor([1.0e-3, 1.0 - 1.0e-3], dtype=DTYPE),
            torch.tensor([1.0 - measured, measured], dtype=DTYPE),
            phase_kinds=("liquid", hydrocarbon_phase),
            max_iterations=30,
        )
        assert point.converged
        predicted = float(point.phase2_composition[1])
        deviation = 100.0 * abs(predicted / measured - 1.0)
        deviations.append(deviation)
        assert deviation < 35.0

    # The paper reports 6.6-12.6% AAD for water in hydrocarbon over its full
    # database. This four-system smoke subset is broader than a single fit but
    # remains subject to visual-digitization and pure-parameter differences.
    assert sum(deviations) / len(deviations) < 20.0
