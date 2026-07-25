from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

import torch_flash.eos.named as named
from torch_flash.database import ModelParameterSet, load_model_parameters
from torch_flash.envelope import phase_envelope
from torch_flash.eos import (
    EOSCG2021_COMPONENTS,
    GERG2008_COMPONENTS,
    GaoBTerms,
    NonAnalyticTerms,
    eoscg2021,
    gerg2008,
)

DATA = Path(__file__).parent / "data"
DTYPE = torch.float64


def test_complete_named_model_inventories():
    gerg = gerg2008()
    assert gerg.names == GERG2008_COMPONENTS
    assert gerg.metadata.validated_components == GERG2008_COMPONENTS
    assert gerg.pure_n.shape[0] == 21
    assert gerg.beta_temperature.shape == (21, 21)
    assert torch.count_nonzero(torch.triu(gerg.departure_scale, diagonal=1)) == 15
    torch.testing.assert_close(gerg.gas_constant, torch.tensor(8.314472, dtype=DTYPE))
    assert gerg.has_ideal_terms
    assert gerg.ideal_lead_constant.shape == (21,)

    eoscg = eoscg2021()
    assert eoscg.names == EOSCG2021_COMPONENTS
    assert eoscg.metadata.validated_components == EOSCG2021_COMPONENTS
    assert eoscg.pure_n.shape[0] == 16
    assert eoscg.beta_temperature.shape == (16, 16)
    assert torch.count_nonzero(torch.triu(eoscg.departure_scale, diagonal=1)) == 21
    assert torch.isfinite(eoscg.critical_pressure).all()
    assert eoscg.has_ideal_terms
    assert eoscg.ideal_lead_constant.shape == (16,)


def test_named_models_aliases_order_device_and_trainability():
    model = gerg2008(("H2", "CH4"), dtype=torch.float32, device="cpu", trainable=True)
    assert model.names == ("hydrogen", "methane")
    assert model.pure_n.dtype == torch.float32
    assert model.pure_n.requires_grad
    assert model.departure_n.requires_grad
    assert model.ideal_lead_constant.requires_grad
    assert model.ideal_planck_n.requires_grad
    torch.testing.assert_close(
        model.beta_temperature[0, 1] * model.beta_temperature[1, 0],
        torch.tensor(1.0),
    )
    assert eoscg2021(("CO2", "H2O", "NH3")).names == (
        "carbon_dioxide",
        "water",
        "ammonia",
    )


@pytest.mark.parametrize(
    ("constructor", "names", "message"),
    [
        (gerg2008, (), "at least one"),
        (gerg2008, ("methane", "methane"), "unique"),
        (eoscg2021, ("ethane",), "unsupported"),
    ],
)
def test_named_model_name_validation(constructor, names, message):
    with pytest.raises(ValueError, match=message):
        constructor(names)


def test_bundled_inventory_guards(monkeypatch):
    broken_gerg = load_model_parameters("multifluid.gerg-2008").as_dict()
    broken_gerg["pairs"].pop(next(iter(broken_gerg["pairs"])))
    monkeypatch.setattr(named, "_read_data", lambda filename: broken_gerg)
    with pytest.raises(RuntimeError, match="GERG-2008"):
        gerg2008(("methane",))

    broken_eoscg = load_model_parameters("multifluid.eos-cg-2021").as_dict()
    broken_eoscg["components"].pop("mdea")
    monkeypatch.setattr(named, "_read_data", lambda filename: broken_eoscg)
    with pytest.raises(RuntimeError, match="EOS-CG-2021"):
        eoscg2021(("carbon_dioxide",))


def test_special_term_shape_validation_and_unknown_term():
    zeros = torch.zeros((2, 1), dtype=DTYPE)
    with pytest.raises(ValueError, match="Gao-B"):
        GaoBTerms(zeros, zeros[:1], zeros, zeros, zeros, zeros, zeros, zeros)
    with pytest.raises(ValueError, match="non-analytic"):
        NonAnalyticTerms(zeros, zeros, zeros, zeros, zeros, zeros[:1], zeros, zeros)
    with pytest.raises(ValueError, match="unsupported regular"):
        named._append_regular_terms([], {"type": "unknown", "n": [1], "d": [1], "t": [1]})
    with pytest.raises(ValueError, match="unsupported ideal"):
        named._canonical_ideal_blocks([{"type": "unknown"}], 300.0)


def test_ideal_cp0_parser_covers_all_exponents():
    lead, power, planck = named._canonical_ideal_blocks(
        [
            {"type": "IdealGasHelmholtzLead", "a1": 1.0, "a2": 2.0},
            {"type": "IdealGasHelmholtzLogTau", "a": 3.0},
            {
                "type": "IdealGasHelmholtzCP0PolyT",
                "T0": 300.0,
                "Tc": 600.0,
                "c": [4.0, 5.0, 6.0],
                "t": [0.0, -1.0, 2.0],
            },
            {"type": "IdealGasHelmholtzPower", "n": [7.0], "t": [0.5]},
            {
                "type": "IdealGasHelmholtzPlanckEinstein",
                "n": [8.0],
                "t": [9.0],
            },
            {
                "type": "IdealGasHelmholtzPlanckEinsteinFunctionT",
                "Tcrit": 600.0,
                "n": [10.0],
                "v": [1200.0],
            },
            {
                "type": "IdealGasHelmholtzEnthalpyEntropyOffset",
                "a1": 11.0,
                "a2": 12.0,
            },
        ],
        600.0,
    )
    assert lead["lead_constant"] != 0.0
    assert lead["lead_tau"] != 0.0
    assert lead["log_tau"] == pytest.approx(7.0)
    assert lead["tau_log_tau"] == pytest.approx(-5.0 / 600.0)
    assert len(power) == 2
    assert planck == [{"n": 8.0, "theta": 9.0}, {"n": 10.0, "theta": 2.0}]


def test_gerg_matches_independent_teqp_regression_values():
    model = gerg2008(("carbon_dioxide", "nitrogen", "methane"))
    composition = torch.tensor([0.7, 0.1, 0.2], dtype=DTYPE)
    temperature = torch.tensor(280.0, dtype=DTYPE)
    density = torch.tensor(5000.0, dtype=DTYPE)
    alpha = model.alpha_residual(temperature, density, composition)
    pressure = model.pressure(temperature, density.reciprocal(), composition)
    torch.testing.assert_close(
        alpha,
        torch.tensor(-0.4537589230201591, dtype=DTYPE),
        rtol=2.0e-14,
        atol=1.0e-15,
    )
    torch.testing.assert_close(
        pressure,
        torch.tensor(6_999_416.730999984, dtype=DTYPE),
        rtol=2.0e-14,
        atol=1.0e-7,
    )


@pytest.mark.serial
def test_gerg_ch4_co2_envelope_continuation_rejects_trivial_branch():
    model = gerg2008(("methane", "carbon_dioxide"))
    temperatures = torch.tensor([230.0, 225.0, 220.0], dtype=DTYPE)
    branches = phase_envelope(
        model,
        temperatures,
        torch.tensor([0.5, 0.5], dtype=DTYPE),
    )

    assert all(point.converged for points in branches.values() for point in points)
    for points in branches.values():
        assert all(float(torch.max(torch.abs(torch.log(point.k_values)))) > 0.1 for point in points)
        assert all(float(point.residual_norm) <= 1.0e-8 for point in points)

    torch.testing.assert_close(
        torch.stack(tuple(point.pressure for point in branches["bubble"])),
        torch.tensor(
            [6_882_489.569132089, 6_394_564.015031207, 5_909_357.895729891],
            dtype=DTYPE,
        ),
        rtol=2.0e-10,
        atol=2.0e-4,
    )
    torch.testing.assert_close(
        torch.stack(tuple(point.pressure for point in branches["dew"])),
        torch.tensor(
            [1_983_899.6943067016, 1_606_977.01573233, 1_291_635.255207501],
            dtype=DTYPE,
        ),
        rtol=2.0e-10,
        atol=2.0e-4,
    )


def test_gerg_co2_water_hou_states_match_teqp_and_thermopack_baseline():
    model = gerg2008(("carbon_dioxide", "water"))
    with (DATA / "gerg2008_co2_water_hou_teqp_thermopack.csv").open() as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 12
    for row in rows:
        temperature = torch.tensor(float(row["temperature_K"]), dtype=DTYPE)
        pressure = torch.tensor(float(row["pressure_Pa"]), dtype=DTYPE)
        x_co2 = float(row["x_co2"])
        composition = torch.tensor([x_co2, 1.0 - x_co2], dtype=DTYPE)
        phase = row["phase"]
        density = model.molar_volume(
            temperature,
            pressure,
            composition,
            phase,
        ).reciprocal()
        log_phi = model.log_fugacity_coefficients(
            temperature,
            pressure,
            composition,
            phase,
        )

        torch.testing.assert_close(
            density,
            torch.tensor(float(row["molar_density_mol_m3"]), dtype=DTYPE),
            rtol=5.0e-10,
            atol=2.0e-7,
        )
        torch.testing.assert_close(
            log_phi,
            torch.tensor(
                [float(row["ln_phi_co2"]), float(row["ln_phi_water"])],
                dtype=DTYPE,
            ),
            rtol=3.0e-10,
            atol=8.0e-11,
        )
        assert float(row["thermopack_max_abs_ln_fugacity_difference"]) < 6.0e-12


def test_gerg_accepts_custom_co2_water_departure_database_and_gradients():
    payload = load_model_parameters("multifluid.gerg-2008").as_dict()
    pair = payload["pairs"]["carbondioxide|water"]
    fitted_amplitudes = [
        0.015750,
        0.002791,
        0.000159,
        0.009612,
        -0.014064,
        -0.005002,
        -0.008033,
        -0.004769,
        -0.003736,
        -0.000308,
    ]
    pair["departure_scale"] = 1.0
    pair["departure"] = {
        "type": "ResidualHelmholtzGERG2008",
        "n": fitted_amplitudes,
        "d": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0],
        "t": [1.0, 1.55, 1.7, 0.25, 1.35, 0.0, 1.25, 0.0, 0.7, 5.4],
        "eta": [0.0] * 10,
        "epsilon": [0.0] * 10,
        "beta": [0.0] * 10,
        "gamma": [0.0] * 10,
    }
    parameter_set = ModelParameterSet(
        "multifluid.gerg-form-co2-water-test",
        "multifluid",
        "GERG-2008-form CO2-H2O custom departure test",
        "notebook-20-regression",
        payload,
    )
    fitted = gerg2008(
        ("carbon_dioxide", "water"),
        parameter_set=parameter_set,
        trainable=True,
    )
    published = gerg2008(("carbon_dioxide", "water"))

    torch.testing.assert_close(
        fitted.departure_n[0, 1],
        torch.tensor(fitted_amplitudes, dtype=DTYPE),
    )
    torch.testing.assert_close(
        fitted.departure_scale[0, 1],
        torch.tensor(1.0, dtype=DTYPE),
    )
    temperature = torch.tensor(323.15, dtype=DTYPE)
    density = torch.tensor(45_000.0, dtype=DTYPE)
    composition = torch.tensor([0.02, 0.98], dtype=DTYPE)
    fitted_alpha = fitted.alpha_residual(temperature, density, composition)
    published_alpha = published.alpha_residual(temperature, density, composition)
    assert not torch.isclose(fitted_alpha, published_alpha, rtol=1.0e-5, atol=1.0e-7)

    departure_gradient = torch.autograd.grad(
        fitted_alpha,
        fitted.departure_n,
    )[0][0, 1]
    assert torch.isfinite(departure_gradient).all()
    assert torch.count_nonzero(departure_gradient.abs() > 1.0e-12) == 10


@pytest.mark.serial
def test_gerg_batched_kernels_roots_and_compiled_graph_match_scalar_calls():
    model = gerg2008(("hydrogen", "methane"))
    temperatures = torch.tensor([280.0, 320.0, 360.0], dtype=DTYPE)
    densities = torch.tensor([100.0, 1000.0, 5000.0], dtype=DTYPE)
    compositions = torch.tensor(
        [[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]],
        dtype=DTYPE,
    )
    batched_pressure = model.pressure(
        temperatures,
        densities.reciprocal(),
        compositions,
    )
    scalar_pressure = torch.stack(
        [
            model.pressure(temperature, density.reciprocal(), composition)
            for temperature, density, composition in zip(
                temperatures,
                densities,
                compositions,
                strict=True,
            )
        ]
    )
    torch.testing.assert_close(batched_pressure, scalar_pressure, rtol=3.0e-14, atol=2.0e-7)

    compiled_pressure = torch.compile(
        lambda current_temperature, current_volume, current_composition: model.pressure(
            current_temperature,
            current_volume,
            current_composition,
        ),
        backend="eager",
        fullgraph=True,
    )
    torch.testing.assert_close(
        compiled_pressure(temperatures, densities.reciprocal(), compositions),
        batched_pressure,
        rtol=2.0e-14,
        atol=2.0e-7,
    )

    target_pressures = batched_pressure.detach().requires_grad_()
    differentiable_temperatures = temperatures.detach().requires_grad_()
    volumes = model.molar_volume(
        differentiable_temperatures,
        target_pressures,
        compositions,
        "vapor",
    )
    scalar_volumes = torch.stack(
        [
            model.molar_volume(temperature, pressure, composition, "vapor")
            for temperature, pressure, composition in zip(
                temperatures,
                target_pressures.detach(),
                compositions,
                strict=True,
            )
        ]
    )
    torch.testing.assert_close(volumes.detach(), scalar_volumes, rtol=2.0e-10, atol=1.0e-14)
    volumes.sum().backward()
    assert differentiable_temperatures.grad is not None
    assert target_pressures.grad is not None
    assert torch.isfinite(differentiable_temperatures.grad).all()
    assert torch.all(target_pressures.grad < 0.0)


@pytest.mark.parametrize(
    ("phase", "compositions"),
    [
        ("liquid", [[0.0129, 0.9871], [0.1775, 0.8225]]),
        ("vapor", [[0.3312, 0.6688], [0.6960, 0.3040]]),
    ],
)
def test_gerg_batched_fugacity_matches_scalar_and_retains_parameter_gradients(
    phase,
    compositions,
):
    model = gerg2008(("nitrogen", "carbon_dioxide"), trainable=True)
    temperatures = torch.full((2,), 240.0, dtype=DTYPE)
    pressures = torch.tensor([21.0e5, 107.28e5], dtype=DTYPE)
    composition = torch.tensor(compositions, dtype=DTYPE)
    batched = model.log_fugacity_coefficients(
        temperatures,
        pressures,
        composition,
        phase,
    )
    scalar = torch.stack(
        [
            model.log_fugacity_coefficients(temperature, pressure, current, phase)
            for temperature, pressure, current in zip(
                temperatures,
                pressures,
                composition,
                strict=True,
            )
        ]
    )
    torch.testing.assert_close(batched, scalar, rtol=2.0e-10, atol=2.0e-11)
    batched_gradient = torch.autograd.grad(
        batched.sum(),
        model.departure_scale,
        retain_graph=True,
    )[0]
    scalar_gradient = torch.autograd.grad(
        scalar.sum(),
        model.departure_scale,
    )[0]
    torch.testing.assert_close(
        batched_gradient,
        scalar_gradient,
        rtol=3.0e-9,
        atol=2.0e-10,
    )
    batched.sum().backward()
    assert model.departure_scale.grad is not None
    assert torch.isfinite(model.departure_scale.grad).all()


def test_gerg_batched_vapor_root_accepts_a_unique_dense_co2_rich_phase():
    model = gerg2008(("carbon_dioxide", "water"))
    temperatures = torch.full((2,), 300.95, dtype=DTYPE)
    pressures = torch.tensor([5.0e6, 34.753e6], dtype=DTYPE)
    compositions = torch.tensor([[0.99, 0.01], [0.99, 0.01]], dtype=DTYPE)

    batched = model.molar_volume(
        temperatures,
        pressures,
        compositions,
        "vapor",
    )
    scalar = torch.stack(
        [
            model.molar_volume(temperature, pressure, composition, "vapor")
            for temperature, pressure, composition in zip(
                temperatures,
                pressures,
                compositions,
                strict=True,
            )
        ]
    )
    torch.testing.assert_close(batched, scalar, rtol=2.0e-10, atol=1.0e-14)
    density_scale = torch.sum(compositions[1] * model.critical_density)
    assert batched[1].reciprocal() > density_scale


def test_complete_ideal_and_caloric_properties():
    gerg = gerg2008(("H2", "CH4"))
    temperature = torch.tensor(300.0, dtype=DTYPE)
    density = torch.tensor(1000.0, dtype=DTYPE)
    composition = torch.tensor([0.4, 0.6], dtype=DTYPE)
    torch.testing.assert_close(
        gerg.alpha_ideal(temperature, density, composition),
        torch.tensor(1.5303900108326565, dtype=DTYPE),
        rtol=2.0e-14,
        atol=1.0e-14,
    )
    torch.testing.assert_close(
        gerg.molar_heat_capacity_cv(temperature, density, composition),
        torch.tensor(24.830305964130297, dtype=DTYPE),
        rtol=2.0e-13,
        atol=1.0e-12,
    )

    eoscg = eoscg2021(("CO2",))
    x = torch.ones(1, dtype=DTYPE)
    temperature = torch.tensor(350.0, dtype=DTYPE)
    density = torch.tensor(1000.0, dtype=DTYPE)
    expected = {
        "alpha_ideal": -4.531744864885811,
        "molar_heat_capacity_cv": 32.64358740584746,
        "molar_heat_capacity_cp": 44.984101688619255,
        "speed_of_sound": 277.0841199245435,
    }
    for method, value in expected.items():
        torch.testing.assert_close(
            getattr(eoscg, method)(temperature, density, x),
            torch.tensor(value, dtype=DTYPE),
            rtol=3.0e-13,
            atol=1.0e-11,
        )
    helmholtz = eoscg.molar_helmholtz_energy(temperature, density, x)
    entropy = eoscg.molar_entropy(temperature, density, x)
    internal = eoscg.molar_internal_energy(temperature, density, x)
    enthalpy = eoscg.molar_enthalpy(temperature, density, x)
    gibbs = eoscg.molar_gibbs_energy(temperature, density, x)
    pressure = eoscg.pressure(temperature, density.reciprocal(), x)
    torch.testing.assert_close(internal, helmholtz + temperature * entropy)
    torch.testing.assert_close(enthalpy, internal + pressure / density)
    torch.testing.assert_close(gibbs, helmholtz + pressure / density)
    torch.testing.assert_close(
        eoscg.chemical_potentials(temperature, density.reciprocal(), x)[0],
        gibbs,
        rtol=2.0e-13,
        atol=1.0e-10,
    )


def test_mdea_ideal_table_from_source_paper():
    model = eoscg2021(("MDEA",))
    temperature = torch.tensor(350.0, dtype=DTYPE)
    density = torch.tensor(8500.0, dtype=DTYPE)
    x = torch.ones(1, dtype=DTYPE)
    torch.testing.assert_close(
        model.alpha_ideal(temperature, density, x),
        torch.tensor(30.61891174980149, dtype=DTYPE),
        rtol=2.0e-14,
        atol=1.0e-14,
    )
    assert model.molar_heat_capacity_cv(temperature, density, x) > 0.0
    assert model.molar_heat_capacity_cp(temperature, density, x) > 0.0
    assert model.speed_of_sound(temperature, density, x) > 0.0


def test_eoscg_co2_h2_dense_teqp_regression():
    model = eoscg2021(("carbon_dioxide", "hydrogen"))
    with (DATA / "eoscg2021_co2_h2_teqp_reference.csv").open() as stream:
        for row in csv.DictReader(stream):
            composition = torch.tensor(
                [float(row["co2_mole_fraction"]), 1.0 - float(row["co2_mole_fraction"])],
                dtype=DTYPE,
            )
            temperature = torch.tensor(float(row["temperature_K"]), dtype=DTYPE)
            density = torch.tensor(float(row["molar_density_mol_m3"]), dtype=DTYPE)
            torch.testing.assert_close(
                model.alpha_residual(temperature, density, composition),
                torch.tensor(float(row["alpha_residual"]), dtype=DTYPE),
                rtol=3.0e-14,
                atol=2.0e-15,
            )
            torch.testing.assert_close(
                model.pressure(temperature, density.reciprocal(), composition),
                torch.tensor(float(row["pressure_Pa"]), dtype=DTYPE),
                rtol=3.0e-14,
                atol=2.0e-7,
            )


@pytest.mark.parametrize(
    ("component", "temperature", "density", "expected"),
    [
        ("carbon_dioxide", 300.0, 1000.0, -0.11891319042065522),
        ("hydrogen", 300.0, 1000.0, 0.014722085995914617),
        ("water", 500.0, 1000.0, -0.178464537439297),
        ("ammonia", 300.0, 1000.0, -0.2503326235646698),
    ],
)
def test_eoscg_pure_residual_terms_match_coolprop(component, temperature, density, expected):
    model = eoscg2021((component,))
    actual = model.alpha_residual(
        torch.tensor(temperature, dtype=DTYPE),
        torch.tensor(density, dtype=DTYPE),
        torch.ones(1, dtype=DTYPE),
    )
    torch.testing.assert_close(
        actual,
        torch.tensor(expected, dtype=DTYPE),
        rtol=2.0e-14,
        atol=1e-15,
    )


@pytest.mark.serial
def test_eoscg_mdea_matches_all_experimental_density_points():
    model = eoscg2021(("mdea",))
    errors = []
    with (DATA / "eoscg2021_mdea_density_experimental.csv").open() as stream:
        for row in csv.DictReader(stream):
            temperature = torch.tensor(float(row["temperature_K"]), dtype=DTYPE)
            pressure = torch.tensor(float(row["pressure_Pa"]), dtype=DTYPE)
            molar_volume = model.molar_volume(
                temperature,
                pressure,
                torch.ones(1, dtype=DTYPE),
                "liquid",
            )
            predicted = model.molar_mass[0] / molar_volume
            reference = torch.tensor(float(row["density_kg_m3"]), dtype=DTYPE)
            errors.append(float(100.0 * (predicted - reference).abs() / reference))
    assert len(errors) == 35
    assert sum(errors) / len(errors) < 0.06
    assert max(errors) < 0.13


@pytest.mark.serial
def test_eoscg_mdea_matches_all_experimental_speed_of_sound_points():
    model = eoscg2021(("mdea",))
    composition = torch.ones(1, dtype=DTYPE)
    errors = []
    with (DATA / "eoscg2021_mdea_speed_of_sound_experimental.csv").open() as stream:
        for row in csv.DictReader(stream):
            temperature = torch.tensor(float(row["temperature_K"]), dtype=DTYPE)
            pressure = torch.tensor(float(row["pressure_Pa"]), dtype=DTYPE)
            molar_volume = model.molar_volume(
                temperature,
                pressure,
                composition,
                "liquid",
            )
            predicted = model.speed_of_sound(
                temperature,
                molar_volume.reciprocal(),
                composition,
            )
            reference = torch.tensor(float(row["speed_of_sound_m_s"]), dtype=DTYPE)
            errors.append(float(100.0 * (predicted - reference).abs() / reference))
    assert len(errors) == 44
    assert sum(errors) / len(errors) < 0.031
    assert max(errors) < 0.07


def test_gerg_hydrogen_methane_paper_bank_subset():
    model = gerg2008(("hydrogen", "methane"))
    density_errors = []
    heat_capacity_errors = []
    with (DATA / "gerg2008_h2_ch4_reference.csv").open() as stream:
        rows = list(csv.DictReader(stream))[::50]
    for row in rows:
        hydrogen = float(row["hydrogen_mole_fraction"])
        composition = torch.tensor([hydrogen, 1.0 - hydrogen], dtype=DTYPE)
        temperature = torch.tensor(float(row["temperature_K"]), dtype=DTYPE)
        pressure = torch.tensor(float(row["pressure_Pa"]), dtype=DTYPE)
        volume = model.molar_volume(temperature, pressure, composition, "vapor")
        molar_density = volume.reciprocal()
        molar_mass = torch.dot(composition, model.molar_mass)
        predicted_density = molar_mass * molar_density
        reference_density = torch.tensor(float(row["density_kg_m3"]), dtype=DTYPE)
        density_errors.append(
            float(100.0 * (predicted_density - reference_density).abs() / reference_density)
        )
        predicted_cp = (
            model.molar_heat_capacity_cp(temperature, molar_density, composition) / molar_mass
        )
        reference_cp = torch.tensor(float(row["heat_capacity_cp_J_kg_K"]), dtype=DTYPE)
        heat_capacity_errors.append(
            float(100.0 * (predicted_cp - reference_cp).abs() / reference_cp)
        )
    assert len(density_errors) == len(heat_capacity_errors) == 21
    assert max(density_errors) < 0.003
    assert max(heat_capacity_errors) < 0.003
