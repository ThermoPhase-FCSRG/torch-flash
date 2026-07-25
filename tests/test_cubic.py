from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from torch_flash import component_set
from torch_flash.activity import Wilson
from torch_flash.constants import R
from torch_flash.eos.cubic import (
    PR78,
    CubicEOS,
    cubic_real_roots,
    peng_robinson_1976,
    peng_robinson_1978,
    soave_redlich_kwong,
)
from torch_flash.exceptions import InvalidStateError
from torch_flash.mixing import (
    HuronVidalMixing,
    QuadraticMixing,
    TemperatureDependentQuadraticMixing,
)

DATA = Path(__file__).parent / "data"


def test_cubic_real_roots_one_and_three():
    dtype = torch.float64
    roots = cubic_real_roots(
        torch.tensor(-6.0, dtype=dtype),
        torch.tensor(11.0, dtype=dtype),
        torch.tensor(-6.0, dtype=dtype),
    )
    torch.testing.assert_close(roots, torch.tensor([1.0, 2.0, 3.0], dtype=dtype))
    repeated = cubic_real_roots(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )
    torch.testing.assert_close(repeated, torch.full((3,), -1.0, dtype=dtype))


def test_pr_matches_frozen_teqp_baseline(num_regression):
    model = peng_robinson_1978(component_set(("methane", "n_butane")))
    maximum_z_error = 0.0
    maximum_logphi_error = 0.0
    calculated_z = []
    calculated_logphi_methane = []
    calculated_logphi_n_butane = []
    with (DATA / "teqp_pr_binary.csv").open() as stream:
        for row in csv.DictReader(stream):
            temperature = torch.tensor(float(row["temperature_K"]), dtype=torch.float64)
            pressure = torch.tensor(float(row["pressure_Pa"]), dtype=torch.float64)
            composition = torch.tensor(
                [float(row["x_methane"]), float(row["x_n_butane"])],
                dtype=torch.float64,
            )
            phase = row["phase"]
            z_factor = model.select_z(temperature, pressure, composition, phase)
            log_phi = model.log_fugacity_coefficients(
                temperature,
                pressure,
                composition,
                phase,
            )
            calculated_z.append(float(z_factor))
            calculated_logphi_methane.append(float(log_phi[0]))
            calculated_logphi_n_butane.append(float(log_phi[1]))
            maximum_z_error = max(maximum_z_error, abs(float(z_factor) - float(row["z"])))
            maximum_logphi_error = max(
                maximum_logphi_error,
                abs(float(log_phi[0]) - float(row["lnphi_methane"])),
                abs(float(log_phi[1]) - float(row["lnphi_n_butane"])),
            )
    assert maximum_z_error < 2.0e-14
    assert maximum_logphi_error < 2.0e-13
    num_regression.check(
        {
            "z": np.asarray(calculated_z),
            "lnphi_methane": np.asarray(calculated_logphi_methane),
            "lnphi_n_butane": np.asarray(calculated_logphi_n_butane),
        },
        basename="teqp_pr_outputs",
        default_tolerance={"rtol": 1.0e-8, "atol": 0.0},
    )


def test_cubic_equation_pressure_and_helmholtz_consistency(binary_model):
    temperature = torch.tensor(300.0, dtype=torch.float64)
    pressure = torch.tensor(5.0e6, dtype=torch.float64)
    composition = torch.tensor([0.7, 0.3], dtype=torch.float64)
    volume = binary_model.molar_volume(temperature, pressure, composition)
    torch.testing.assert_close(
        binary_model.pressure(temperature, volume, composition),
        pressure,
        rtol=2.0e-14,
        atol=1.0e-7,
    )
    helmholtz = binary_model.residual_helmholtz_rt(
        temperature,
        volume,
        composition,
    )
    assert torch.isfinite(helmholtz)
    derivative_pressure = R * temperature / volume - R * temperature * torch.func.grad(
        lambda current_volume: binary_model.residual_helmholtz_rt(
            temperature,
            current_volume,
            composition,
        )
    )(volume)
    torch.testing.assert_close(derivative_pressure, pressure, rtol=5.0e-14, atol=1.0e-6)


def test_cubic_constructors_alpha_variants_and_buffers():
    components = component_set(("ethanol",))
    temperature = torch.tensor(500.0, dtype=torch.float64)
    pr76 = peng_robinson_1976(components)
    pr78 = peng_robinson_1978(components, trainable=True)
    srk = soave_redlich_kwong(components)
    assert not torch.allclose(
        pr76.pure_parameters(temperature)[0], pr78.pure_parameters(temperature)[0]
    )
    assert srk.constants.name == "SRK"
    assert srk.pure_parameters(temperature)[0] > 0.0
    assert isinstance(pr78.mixing.raw_kij, torch.nn.Parameter)
    assert pr78.ncomponents == 1
    assert pr78.names == ("ethanol",)
    explicit = torch.zeros((1, 1), dtype=torch.float64)
    assert soave_redlich_kwong(components, kij=explicit).ncomponents == 1
    assert peng_robinson_1976(components, kij=explicit).ncomponents == 1


def test_cubic_batching_and_parameter_gradient(binary_components):
    kij = torch.tensor([[0.0, 0.05], [0.05, 0.0]], dtype=torch.float64)
    model = peng_robinson_1978(binary_components, kij=kij, trainable=True)
    temperature = torch.tensor([280.0, 320.0], dtype=torch.float64)
    pressure = torch.tensor([2.0e6, 4.0e6], dtype=torch.float64)
    composition = torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float64)
    roots = model.z_factors(temperature, pressure, composition)
    assert roots.shape == (2, 3)
    loss = model.select_z(temperature, pressure, composition, "vapor").sum()
    loss.backward()
    assert model.mixing.raw_kij.grad is not None
    assert torch.isfinite(model.mixing.raw_kij.grad).all()


def test_temperature_dependent_cubic_bip_factory_fugacity_and_gradients(binary_components):
    a = torch.tensor([[0.0, 0.02], [0.02, 0.0]], dtype=torch.float64)
    b = torch.tensor([[0.0, 12.0], [12.0, 0.0]], dtype=torch.float64)
    lij = torch.tensor([[0.0, 0.06], [0.06, 0.0]], dtype=torch.float64)
    model = peng_robinson_1978(
        binary_components,
        kij_a=a,
        kij_b=b,
        lij=lij,
        trainable=True,
        trainable_lij=True,
    )
    assert isinstance(model.mixing, TemperatureDependentQuadraticMixing)

    temperature = torch.tensor(300.0, dtype=torch.float64)
    pressure = torch.tensor(2.0e6, dtype=torch.float64)
    composition = torch.tensor([0.7, 0.3], dtype=torch.float64)
    evaluated = model.mixing.kij(temperature).detach()
    constant = peng_robinson_1978(binary_components, kij=evaluated, lij=lij)
    torch.testing.assert_close(
        model.log_fugacity_coefficients(temperature, pressure, composition, "vapor"),
        constant.log_fugacity_coefficients(temperature, pressure, composition, "vapor"),
    )
    linear_covolume_model = peng_robinson_1978(
        binary_components,
        kij_a=a,
        kij_b=b,
    )
    linear_covolume_constant = peng_robinson_1978(
        binary_components,
        kij=linear_covolume_model.mixing.kij(temperature).detach(),
    )
    torch.testing.assert_close(
        linear_covolume_model.log_fugacity_coefficients(
            temperature,
            pressure,
            composition,
            "vapor",
        ),
        linear_covolume_constant.log_fugacity_coefficients(
            temperature,
            pressure,
            composition,
            "vapor",
        ),
    )

    loss = (
        model.log_fugacity_coefficients(
            temperature,
            pressure,
            composition,
            "vapor",
        )
        .square()
        .sum()
    )
    loss.backward()
    assert model.mixing.raw_a.grad is not None
    assert model.mixing.raw_b.grad is not None
    assert model.mixing.raw_lij.grad is not None
    assert torch.isfinite(model.mixing.raw_a.grad).all()
    assert torch.isfinite(model.mixing.raw_b.grad).all()
    assert model.mixing.raw_a.grad[0, 1] != 0.0
    assert model.mixing.raw_b.grad[0, 1] != 0.0
    assert model.mixing.raw_lij.grad[0, 1] != 0.0


def test_temperature_dependent_cubic_bip_factory_validation(binary_components):
    zeros = torch.zeros((2, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="requires both"):
        peng_robinson_1978(binary_components, kij_a=zeros)
    with pytest.raises(ValueError, match="mutually exclusive"):
        peng_robinson_1978(binary_components, kij=zeros, kij_a=zeros, kij_b=zeros)
    with pytest.raises(ValueError, match="mutually exclusive"):
        peng_robinson_1978(
            binary_components,
            kij_a=zeros,
            kij_b=zeros,
            mixing=QuadraticMixing(zeros),
        )


def test_huron_vidal_autodiff_fugacity(binary_components):
    activity = Wilson(
        torch.tensor([[0.0, 500.0], [-200.0, 0.0]], dtype=torch.float64),
        torch.tensor([8.0e-5, 1.0e-4], dtype=torch.float64),
    )
    mixing = HuronVidalMixing(
        activity,
        delta1=PR78.delta1,
        delta2=PR78.delta2,
    )
    model = CubicEOS(binary_components, PR78, mixing=mixing)
    temperature = torch.tensor(320.0, dtype=torch.float64)
    pressure = torch.tensor(2.0e6, dtype=torch.float64)
    composition = torch.tensor([0.4, 0.6], dtype=torch.float64)
    log_phi = model.log_fugacity_coefficients(
        temperature,
        pressure,
        composition,
        "vapor",
    )
    assert log_phi.shape == (2,)
    assert torch.isfinite(log_phi).all()
    jacobian = torch.func.jacrev(
        lambda values: model.log_fugacity_coefficients(
            temperature,
            pressure,
            torch.softmax(values, dim=0),
            "vapor",
        )
    )(torch.log(composition))
    assert torch.isfinite(jacobian).all()


def test_volume_translation_changes_volume_and_pressure(binary_components):
    translation = torch.tensor([1.0e-6, 2.0e-6], dtype=torch.float64)
    model = CubicEOS(
        binary_components,
        PR78,
        mixing=QuadraticMixing(torch.zeros((2, 2), dtype=torch.float64)),
        volume_translation=translation,
    )
    temperature = torch.tensor(300.0, dtype=torch.float64)
    pressure = torch.tensor(1.0e6, dtype=torch.float64)
    composition = torch.tensor([0.25, 0.75], dtype=torch.float64)
    volume = model.molar_volume(temperature, pressure, composition, "vapor")
    unshifted = (
        model.select_z(temperature, pressure, composition, "vapor") * R * temperature / pressure
    )
    torch.testing.assert_close(volume - unshifted, torch.dot(composition, translation))
    torch.testing.assert_close(
        model.pressure(temperature, volume, composition),
        pressure,
        rtol=1.0e-13,
        atol=1.0e-7,
    )
    with pytest.raises(ValueError, match="one value"):
        CubicEOS(binary_components, PR78, volume_translation=torch.zeros(3))


def test_cubic_input_errors(binary_model, monkeypatch):
    x = torch.tensor([0.5, 0.5], dtype=torch.float64)
    with pytest.raises(InvalidStateError, match="temperature"):
        binary_model.pure_parameters(torch.tensor(0.0))
    with pytest.raises(InvalidStateError, match="pressure"):
        binary_model.dimensionless_parameters(
            torch.tensor(300.0),
            torch.tensor(0.0),
            x,
        )
    with pytest.raises(InvalidStateError, match="covolume"):
        binary_model.pressure(
            torch.tensor(300.0, dtype=torch.float64),
            torch.tensor(1.0e-12, dtype=torch.float64),
            x,
        )
    with pytest.raises(ValueError, match="unknown phase"):
        binary_model.select_z(
            torch.tensor(300.0),
            torch.tensor(1.0e5),
            x,
            "solid",
        )
    monkeypatch.setattr(
        binary_model,
        "z_factors",
        lambda temperature, pressure, composition: torch.zeros(
            3,
            dtype=torch.float64,
        ),
    )
    monkeypatch.setattr(
        binary_model,
        "dimensionless_parameters",
        lambda temperature, pressure, composition: (
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        ),
    )
    with pytest.raises(InvalidStateError, match="no physical"):
        binary_model.select_z(
            torch.tensor(300.0),
            torch.tensor(1.0e5),
            x,
            "vapor",
        )
