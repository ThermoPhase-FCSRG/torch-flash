from __future__ import annotations

import warnings

import pytest
import torch

from torch_flash import (
    ChemicalState,
    multiphase_flash,
    peng_robinson_1978,
    rachford_rice,
    two_phase_flash,
)
from torch_flash.components import ComponentSet
from torch_flash.exceptions import ExperimentalModelWarning

PSIA_TO_PA = 6894.757293168


def test_whitson_appendix_b_problem_15_rachford_rice():
    result = rachford_rice(
        torch.tensor([0.20, 0.32, 0.48], dtype=torch.float64),
        torch.tensor([9.208, 1.439, 0.358], dtype=torch.float64),
    )
    # Table B-20 gives 0.48242, while its three-decimal K values solve to
    # 0.48291; the difference is the expected loss from the printed rounding.
    assert float(result.vapor_fraction) == pytest.approx(0.4829102182, abs=2.0e-10)
    assert abs(float(result.vapor_fraction) - 0.48242) < 5.0e-4
    torch.testing.assert_close(
        result.liquid_composition,
        torch.tensor([0.0403, 0.2641, 0.6956], dtype=torch.float64),
        atol=1.3e-4,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.vapor_composition,
        torch.tensor([0.3713, 0.3800, 0.2487], dtype=torch.float64),
        atol=4.5e-4,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("pressure_psia", "expected_beta", "expected_x", "expected_y"),
    [
        (
            500.0,
            0.853401,
            [0.08588, 0.46349, 0.45064],
            [0.57114, 0.41253, 0.01633],
        ),
        (
            1500.0,
            0.566844,
            [0.330082, 0.513307, 0.156611],
            [0.629843, 0.348699, 0.021457],
        ),
    ],
)
def test_whitson_appendix_b_problem_18_pr_flash(
    pressure_psia,
    expected_beta,
    expected_x,
    expected_y,
):
    components = ComponentSet(
        ("methane", "n_butane", "n_decane"),
        torch.tensor([343.0, 765.3, 1111.8], dtype=torch.float64) / 1.8,
        torch.tensor([667.8, 550.7, 304.0], dtype=torch.float64) * PSIA_TO_PA,
        torch.tensor([0.0115, 0.1928, 0.4902], dtype=torch.float64),
        torch.tensor([16.04, 58.12, 142.29], dtype=torch.float64) / 1000.0,
    )
    result = two_phase_flash(
        peng_robinson_1978(components),
        ChemicalState(
            torch.tensor(740.0 / 1.8, dtype=torch.float64),
            torch.tensor(pressure_psia * PSIA_TO_PA, dtype=torch.float64),
            torch.tensor([0.50, 0.42, 0.08], dtype=torch.float64),
        ),
        check_stability=False,
    )
    assert result.converged
    assert float(result.phase_fractions[1]) == pytest.approx(expected_beta, abs=1.2e-3)
    torch.testing.assert_close(
        result.phases[0].composition,
        torch.tensor(expected_x, dtype=torch.float64),
        atol=8.0e-4,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.phases[1].composition,
        torch.tensor(expected_y, dtype=torch.float64),
        atol=8.0e-4,
        rtol=0.0,
    )


def test_pedersen_tables_6_5_and_6_6_three_phase_flash():
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
        + 273.15,
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
    gas = torch.tensor(
        [0.18, 1.08, 96.45, 1.86, 0.33, 0.03, 0.05, 0.01, 0.01, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4],
        dtype=torch.float64,
    )
    liquid_1 = torch.tensor(
        [0.08, 1.88, 87.95, 5.28, 2.17, 0.38, 0.76, 0.28, 0.29, 0.29, 0.48, 0.14, 0.01],
        dtype=torch.float64,
    )
    liquid_2 = torch.tensor(
        [0.05, 2.36, 75.66, 7.28, 4.00, 0.81, 1.83, 0.79, 0.93, 1.14, 2.73, 1.54, 0.87],
        dtype=torch.float64,
    )
    initial_k = torch.stack(
        (
            gas / gas.sum() / (liquid_1 / liquid_1.sum()),
            liquid_2 / liquid_2.sum() / (liquid_1 / liquid_1.sum()),
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalModelWarning)
        result = multiphase_flash(
            peng_robinson_1978(components),
            ChemicalState(
                torch.tensor(201.15, dtype=torch.float64),
                torch.tensor(5.2e6, dtype=torch.float64),
                feed,
            ),
            initial_k_values=initial_k,
            tolerance=1.0e-10,
            max_iterations=30,
        )
    assert result.converged
    assert result.diagnostics["autodiff_newton_steps"] >= 1
    expected_fractions = torch.tensor([0.2615, 0.1751, 0.5633], dtype=torch.float64)
    torch.testing.assert_close(result.phase_fractions, expected_fractions, atol=1.7e-2, rtol=0.0)
    # The hydrocarbon partitions are reproducible from the rounded Table 6.5
    # constants; N2/CO2 are more sensitive to the unreported BIP convention.
    expected_hydrocarbons = torch.stack(
        (
            liquid_1[2:] / liquid_1.sum(),
            gas[2:] / gas.sum(),
            liquid_2[2:] / liquid_2.sum(),
        )
    )
    actual_hydrocarbons = torch.stack(tuple(phase.composition[2:] for phase in result.phases))
    torch.testing.assert_close(
        actual_hydrocarbons,
        expected_hydrocarbons,
        atol=5.5e-3,
        rtol=0.0,
    )
