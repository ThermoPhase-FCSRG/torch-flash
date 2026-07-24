from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from torch_flash.components import ComponentSet
from torch_flash.constants import R
from torch_flash.eos.cubic import (
    CubicEOS,
    peng_robinson_1976,
    peng_robinson_1978,
    soave_redlich_kwong,
)
from torch_flash.mixing import QuadraticMixing, TemperatureDependentQuadraticMixing

DATA = Path(__file__).parent / "data"


def _thermopack_components(row: dict[str, str]) -> ComponentSet:
    return ComponentSet(
        ("methane", "n_decane"),
        torch.tensor(
            [
                float(row["critical_temperature_1_K"]),
                float(row["critical_temperature_2_K"]),
            ],
            dtype=torch.float64,
        ),
        torch.tensor(
            [
                float(row["critical_pressure_1_Pa"]),
                float(row["critical_pressure_2_Pa"]),
            ],
            dtype=torch.float64,
        ),
        torch.tensor(
            [float(row["acentric_factor_1"]), float(row["acentric_factor_2"])],
            dtype=torch.float64,
        ),
        torch.tensor(
            [
                float(row["molar_mass_1_kg_mol"]),
                float(row["molar_mass_2_kg_mol"]),
            ],
            dtype=torch.float64,
        ),
    )


def test_covolume_matrix_mixing_and_zero_interaction_limit():
    kij = torch.tensor([[0.0, 0.04], [0.04, 0.0]], dtype=torch.float64)
    lij = torch.tensor([[0.0, 0.10], [0.10, 0.0]], dtype=torch.float64)
    pure_a = torch.tensor([1.0, 4.0], dtype=torch.float64)
    pure_b = torch.tensor([0.1, 0.3], dtype=torch.float64)
    composition = torch.tensor([0.25, 0.75], dtype=torch.float64)
    rule = QuadraticMixing(kij, lij)
    expected_bij = torch.tensor([[0.1, 0.18], [0.18, 0.3]], dtype=torch.float64)
    torch.testing.assert_close(rule.cross_b(pure_b), expected_bij)
    _, bm = rule(torch.tensor(300.0), composition, pure_a, pure_b)
    torch.testing.assert_close(
        bm,
        torch.einsum("i,ij,j", composition, expected_bij, composition),
    )

    linear_rule = QuadraticMixing(kij)
    _, linear_bm = linear_rule(torch.tensor(300.0), composition, pure_a, pure_b)
    torch.testing.assert_close(linear_bm, torch.dot(composition, pure_b))


def test_partial_molar_covolume_matches_extensive_autodiff():
    rule = QuadraticMixing(
        torch.zeros((3, 3), dtype=torch.float64),
        torch.tensor(
            [[0.0, 0.10, -0.03], [0.10, 0.0, 0.04], [-0.03, 0.04, 0.0]],
            dtype=torch.float64,
        ),
    )
    pure_a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    pure_b = torch.tensor([0.08, 0.12, 0.25], dtype=torch.float64)
    moles = torch.tensor([0.2, 0.7, 0.4], dtype=torch.float64)

    def extensive_b(current_moles: Tensor) -> Tensor:
        total = current_moles.sum()
        _, bm = rule(
            torch.tensor(320.0),
            current_moles / total,
            pure_a,
            pure_b,
        )
        return total * bm

    torch.testing.assert_close(
        rule.partial_b(moles / moles.sum(), pure_b),
        torch.func.grad(extensive_b)(moles),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_covolume_closed_form_fugacity_matches_helmholtz_autodiff(binary_components):
    kij = torch.tensor([[0.0, 0.05], [0.05, 0.0]], dtype=torch.float64)
    lij = torch.tensor([[0.0, 0.08], [0.08, 0.0]], dtype=torch.float64)
    model = peng_robinson_1978(binary_components, kij=kij, lij=lij)
    temperature = torch.tensor(310.0, dtype=torch.float64)
    pressure = torch.tensor(8.0e6, dtype=torch.float64)
    composition = torch.tensor([0.65, 0.35], dtype=torch.float64)
    volume = model.molar_volume(temperature, pressure, composition, "liquid")
    z_factor = pressure * volume / (R * temperature)

    residual_mu_rt = torch.func.grad(
        lambda moles: model.residual_helmholtz_rt(
            temperature,
            volume,
            moles,
        )
    )(composition)
    autodiff_log_phi = residual_mu_rt - torch.log(z_factor)
    closed_form = model.log_fugacity_coefficients(
        temperature,
        pressure,
        composition,
        "liquid",
    )
    torch.testing.assert_close(closed_form, autodiff_log_phi, rtol=2.0e-13, atol=2.0e-13)


def test_covolume_interaction_is_independently_trainable(binary_components):
    zeros = torch.zeros((2, 2), dtype=torch.float64)
    model = peng_robinson_1978(
        binary_components,
        kij=zeros,
        trainable_lij=True,
    )
    assert isinstance(model.mixing.raw_kij, Tensor)
    assert not isinstance(model.mixing.raw_kij, nn.Parameter)
    assert isinstance(model.mixing.raw_lij, nn.Parameter)
    loss = model.select_z(
        torch.tensor(300.0, dtype=torch.float64),
        torch.tensor(8.0e6, dtype=torch.float64),
        torch.tensor([0.6, 0.4], dtype=torch.float64),
        "liquid",
    )
    loss.backward()
    assert model.mixing.raw_lij.grad is not None
    assert torch.isfinite(model.mixing.raw_lij.grad).all()
    assert model.mixing.raw_lij.grad[0, 1] != 0.0


def test_covolume_validation_and_temperature_dependent_rule():
    zeros = torch.zeros((2, 2), dtype=torch.float64)
    asymmetric = torch.tensor([[0.0, 0.1], [0.0, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="same square shape"):
        QuadraticMixing(zeros, torch.zeros((3, 3)))
    with pytest.raises(ValueError, match="finite"):
        QuadraticMixing(torch.tensor([[0.0, torch.inf], [torch.inf, 0.0]]))
    with pytest.raises(ValueError, match="finite"):
        QuadraticMixing(zeros, torch.tensor([[0.0, torch.inf], [torch.inf, 0.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        QuadraticMixing(zeros, asymmetric)
    with pytest.raises(ValueError, match="same square shape"):
        TemperatureDependentQuadraticMixing(zeros, zeros, torch.zeros((3, 3)))
    with pytest.raises(ValueError, match="must be finite"):
        TemperatureDependentQuadraticMixing(
            zeros,
            torch.tensor([[0.0, torch.inf], [torch.inf, 0.0]]),
        )
    with pytest.raises(ValueError, match="finite"):
        TemperatureDependentQuadraticMixing(
            zeros,
            zeros,
            torch.tensor([[0.0, torch.inf], [torch.inf, 0.0]]),
        )
    with pytest.raises(ValueError, match="symmetric"):
        TemperatureDependentQuadraticMixing(zeros, zeros, asymmetric)

    rule = TemperatureDependentQuadraticMixing(
        zeros,
        zeros,
        torch.tensor([[0.0, 0.08], [0.08, 0.0]], dtype=torch.float64),
    )
    pure_b = torch.tensor([0.1, 0.2], dtype=torch.float64)
    torch.testing.assert_close(
        rule.cross_b(pure_b),
        torch.tensor([[0.1, 0.138], [0.138, 0.2]], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "factory",
    [soave_redlich_kwong, peng_robinson_1976, peng_robinson_1978],
)
def test_covolume_supported_by_all_cubic_factories(factory, binary_components):
    lij = torch.tensor([[0.0, 0.07], [0.07, 0.0]], dtype=torch.float64)
    model = factory(binary_components, lij=lij)
    temperature = torch.tensor(320.0, dtype=torch.float64)
    composition = torch.tensor([0.4, 0.6], dtype=torch.float64)
    _, pure_b = model.pure_parameters(temperature)
    _, mixed_b = model.mixture_parameters(temperature, composition)
    assert mixed_b < torch.dot(composition, pure_b)


def test_pr78_covolume_matches_frozen_thermopack_223_baseline(data_regression):
    maximum_volume_error = 0.0
    maximum_logphi_error = 0.0
    with (DATA / "thermopack_pr78_covolume.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        kij = torch.tensor(
            [[0.0, float(row["k12"])], [float(row["k12"]), 0.0]],
            dtype=torch.float64,
        )
        lij = torch.tensor(
            [[0.0, float(row["l12"])], [float(row["l12"]), 0.0]],
            dtype=torch.float64,
        )
        model = peng_robinson_1978(_thermopack_components(row), kij=kij, lij=lij)
        temperature = torch.tensor(float(row["temperature_K"]), dtype=torch.float64)
        pressure = torch.tensor(float(row["pressure_Pa"]), dtype=torch.float64)
        composition = torch.tensor(
            [float(row["x1"]), 1.0 - float(row["x1"])],
            dtype=torch.float64,
        )
        phase = row["phase"]
        volume = model.molar_volume(temperature, pressure, composition, phase)
        log_phi = model.log_fugacity_coefficients(
            temperature,
            pressure,
            composition,
            phase,
        )
        maximum_volume_error = max(
            maximum_volume_error,
            abs(float(volume) - float(row["molar_volume_m3_mol"])),
        )
        maximum_logphi_error = max(
            maximum_logphi_error,
            abs(float(log_phi[0]) - float(row["lnphi_1"])),
            abs(float(log_phi[1]) - float(row["lnphi_2"])),
        )
    assert maximum_volume_error < 3.0e-19
    assert maximum_logphi_error < 3.0e-14
    data_regression.check(
        {
            "baseline": "ThermoPack-2.2.3-PR78",
            "cases": len(rows),
            "maximum_logphi_error": maximum_logphi_error,
            "maximum_volume_error_m3_mol": maximum_volume_error,
        },
        basename="thermopack_pr78_covolume_summary",
    )


def test_fitted_covolume_improves_unseen_methane_decane_density_states(
    not_cleared_data: Path,
):
    with (not_cleared_data / "segovia_2017_methane_n_decane_density.csv").open() as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["series"] == "323_K_isotherm" and float(row["x_methane"]) > 0.0
        ]
    components = ComponentSet(
        ("methane", "n_decane"),
        torch.tensor([190.56, 617.70], dtype=torch.float64),
        torch.tensor([4.599e6, 2.110e6], dtype=torch.float64),
        torch.tensor([0.0115, 0.4923], dtype=torch.float64),
        torch.tensor([16.04246e-3, 142.28168e-3], dtype=torch.float64),
    )
    kij = torch.tensor([[0.0, 0.0409], [0.0409, 0.0]], dtype=torch.float64)
    fitted_l12 = 0.047567430964013162
    lij = torch.tensor([[0.0, fitted_l12], [fitted_l12, 0.0]], dtype=torch.float64)
    conventional = peng_robinson_1978(components, kij=kij)
    fitted = peng_robinson_1978(components, kij=kij, lij=lij)
    temperature = torch.tensor([float(row["T_K"]) for row in rows], dtype=torch.float64)
    pressure = torch.tensor(
        [float(row["P_MPa"]) * 1.0e6 for row in rows],
        dtype=torch.float64,
    )
    x1 = torch.tensor([float(row["x_methane"]) for row in rows], dtype=torch.float64)
    composition = torch.stack((x1, 1.0 - x1), dim=-1)
    observed = torch.tensor(
        [float(row["density_g_cm3"]) for row in rows],
        dtype=torch.float64,
    )
    molar_mass = torch.sum(composition * components.molar_mass, dim=-1)

    def mean_absolute_relative_error(model: CubicEOS) -> Tensor:
        volume = model.molar_volume(temperature, pressure, composition, "liquid")
        density = molar_mass / volume / 1000.0
        return torch.mean(torch.abs(density / observed - 1.0)) * 100.0

    conventional_error = mean_absolute_relative_error(conventional)
    fitted_error = mean_absolute_relative_error(fitted)
    assert len(rows) == 33
    assert conventional_error > 4.0
    assert fitted_error < 3.2
    assert conventional_error - fitted_error > 1.0
