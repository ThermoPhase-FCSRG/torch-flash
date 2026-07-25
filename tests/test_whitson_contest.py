from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from torch_flash.material_balance.rachford_rice import rachford_rice_numpy


@pytest.mark.serial
@pytest.mark.slow
def test_all_10008_whitson_contest_cases():
    configured = os.environ.get("TORCH_FLASH_WHITSON_DATA")
    root = Path(configured) if configured else Path(__file__).parent / "data" / "whitson"
    if not root.exists():
        pytest.skip("set TORCH_FLASH_WHITSON_DATA to the optional Whitson contest checkout")
    with (root / "compositions.csv").open() as stream:
        composition_rows = list(csv.reader(stream))[1:]
    with (root / "k-values.csv").open() as stream:
        k_rows = list(csv.reader(stream))[1:]

    failures = []
    for case, (composition_row, k_row) in enumerate(
        zip(composition_rows, k_rows, strict=True),
        1,
    ):
        ncomponents = int(float(composition_row[0]))
        composition = np.asarray(composition_row[1 : ncomponents + 1], dtype=float)
        composition /= composition.sum()
        k_values = np.asarray(k_row[:ncomponents], dtype=float)
        _, vapor, liquid, beta_v, beta_l = rachford_rice_numpy(composition, k_values)
        phase_tolerance = 1.0e-15 + ncomponents * np.finfo(float).eps
        normalized_residuals = (
            abs(1.0 - vapor.sum()) / phase_tolerance,
            abs(1.0 - liquid.sum()) / phase_tolerance,
            abs(beta_v + beta_l - 1.0) / (abs(beta_v) + abs(beta_l) + 1.0) / 1.0e-15,
            np.max(
                np.abs(beta_v * vapor + beta_l * liquid - composition)
                / (np.abs(beta_v * vapor) + np.abs(beta_l * liquid) + composition)
            )
            / 1.0e-15,
            np.max(np.abs(vapor - k_values * liquid) / (np.abs(vapor) + np.abs(k_values * liquid)))
            / 1.0e-15,
        )
        lower = 1.0 / (1.0 - k_values.max())
        upper = 1.0 / (1.0 - k_values.min())
        if max(normalized_residuals) > 1.0 or not lower < beta_v < upper:
            failures.append((case, max(normalized_residuals)))
    assert not failures
    assert len(composition_rows) == 10_008
