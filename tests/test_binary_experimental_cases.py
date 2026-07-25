from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch

from torch_flash import (
    activity_model,
    binary_bubble_point,
    binary_vle_point,
    component_set,
    saturation_point,
    soave_redlich_kwong,
)
from torch_flash.eos import eoscg2021
from torch_flash.eos.cubic import SRK
from torch_flash.mixing import HuronVidalMixing

DATA = Path(__file__).parent / "data"
DTYPE = torch.float64


def test_jaubert_hv_bac5_tables_are_complete(not_cleared_data: Path):
    with (not_cleared_data / "jaubert_2020_hv_bac5_vle.csv").open() as stream:
        rows = tuple(csv.DictReader(stream))

    systems = {}
    for row in rows:
        key = (row["component1"], row["component2"])
        systems.setdefault(key, []).append(row)
    assert {key: len(value) for key, value in systems.items()} == {
        ("n_butane", "water"): 80,
        ("ethanol", "n_heptane"): 120,
        ("methanol", "benzene"): 128,
    }
    assert {
        key: len({float(row["temperature_K"]) for row in value}) for key, value in systems.items()
    } == {
        ("n_butane", "water"): 9,
        ("ethanol", "n_heptane"): 4,
        ("methanol", "benzene"): 8,
    }
    assert sum(float(row["x1"]) == 0.0 for row in rows) == 10
    assert sum(row["flag"] == "AZEO" for row in rows) == 8
    assert not any(row["source_doi"] == "10.1021/je025604s" for row in rows)
    assert rows[0]["source_doi"] == "10.1021/ie50507a049"
    assert rows[-1]["source_doi"] == "10.1002/jctb.5010180402"


@pytest.mark.serial
@pytest.mark.parametrize(
    (
        "system",
        "parameter_set",
        "holdout_temperatures",
        "maximum_pressure_bar",
        "maximum_pressure_mape",
        "maximum_vapor_mae",
    ),
    [
        (
            ("ethanol", "n_heptane"),
            "jaubert-hv-ethanol-n-heptane",
            (313.15,),
            100.0,
            5.0,
            0.05,
        ),
        (
            ("methanol", "benzene"),
            "jaubert-hv-methanol-benzene",
            (328.15, 433.15),
            100.0,
            2.1,
            0.02,
        ),
    ],
)
def test_bundled_hv_models_improve_complete_temperature_holdouts(
    system,
    parameter_set,
    holdout_temperatures,
    maximum_pressure_bar,
    maximum_pressure_mape,
    maximum_vapor_mae,
    not_cleared_data: Path,
):
    """Regress each HV parameter set on complete, unseen experimental isotherms."""
    with (not_cleared_data / "jaubert_2020_hv_bac5_vle.csv").open() as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if (row["component1"], row["component2"]) == system
            and float(row["temperature_K"]) in holdout_temperatures
            and 0.0 < float(row["x1"]) < 1.0
            and 0.0 < float(row["y1"]) < 1.0
        ]
    assert rows

    components = component_set(system)
    activity = activity_model(parameter_set, system)
    hv_model = soave_redlich_kwong(
        components,
        mixing=HuronVidalMixing(
            activity,
            delta1=SRK.delta1,
            delta2=SRK.delta2,
        ),
    )
    models = (soave_redlich_kwong(components), hv_model)
    metrics = []
    for model in models:
        pressure_errors = []
        vapor_errors = []
        for row in rows:
            pressure_bar = float(row["pressure_bar"])
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            point = binary_bubble_point(
                model,
                torch.tensor(float(row["temperature_K"]), dtype=DTYPE),
                torch.tensor([x1, 1.0 - x1], dtype=DTYPE),
                initial_pressure=torch.tensor(
                    pressure_bar * 1.0e5,
                    dtype=DTYPE,
                ),
                initial_vapor_composition=torch.tensor(
                    [y1, 1.0 - y1],
                    dtype=DTYPE,
                ),
                minimum_pressure=1.0e2,
                maximum_pressure=maximum_pressure_bar * 1.0e5,
                max_iterations=15,
            )
            if point.converged:
                pressure_errors.append(
                    100.0 * abs(float(point.pressure) / 1.0e5 / pressure_bar - 1.0)
                )
                vapor_errors.append(abs(float(point.vapor_composition[0]) - y1))
        metrics.append(
            (
                len(pressure_errors),
                sum(pressure_errors) / len(pressure_errors),
                sum(vapor_errors) / len(vapor_errors),
            )
        )

    baseline, hv = metrics
    assert hv[0] >= baseline[0]
    assert hv[1] < baseline[1]
    assert hv[2] < baseline[2]
    assert hv[1] < maximum_pressure_mape
    assert hv[2] < maximum_vapor_mae


@pytest.mark.serial
def test_n_butane_water_hv_improves_fixed_tp_holdouts(not_cleared_data: Path):
    """Validate the rounded dilute table without inverting it as ``P(T, x)``."""
    system = ("n_butane", "water")
    holdout_temperatures = (410.93, 510.93)
    with (not_cleared_data / "jaubert_2020_hv_bac5_vle.csv").open() as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if (row["component1"], row["component2"]) == system
            and float(row["temperature_K"]) in holdout_temperatures
            and 0.0 < float(row["x1"]) < 1.0
            and 0.0 < float(row["y1"]) < 1.0
        ]
    assert len(rows) == 20

    components = component_set(system)
    activity = activity_model("jaubert-hv-n-butane-water", system)
    hv_model = soave_redlich_kwong(
        components,
        mixing=HuronVidalMixing(
            activity,
            delta1=SRK.delta1,
            delta2=SRK.delta2,
        ),
    )
    metrics = []
    for model in (soave_redlich_kwong(components), hv_model):
        liquid_errors = []
        vapor_errors = []
        converged = 0
        for row in rows:
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            point = binary_vle_point(
                model,
                torch.tensor(float(row["temperature_K"]), dtype=DTYPE),
                torch.tensor(float(row["pressure_bar"]) * 1.0e5, dtype=DTYPE),
                torch.tensor([x1, 1.0 - x1], dtype=DTYPE),
                torch.tensor([y1, 1.0 - y1], dtype=DTYPE),
                max_iterations=60,
            )
            if point.converged:
                converged += 1
                liquid_errors.append(abs(float(point.liquid_composition[0]) - x1))
                vapor_errors.append(abs(float(point.vapor_composition[0]) - y1))
        metrics.append(
            (
                converged,
                sum(liquid_errors) / len(liquid_errors),
                sum(vapor_errors) / len(vapor_errors),
            )
        )

    baseline, hv = metrics
    assert hv[0] == len(rows)
    assert hv[0] >= baseline[0]
    assert hv[1] < baseline[1]
    assert hv[2] < baseline[2]
    assert hv[1] < 1.5e-4
    assert hv[2] < 0.035


def test_co2_water_literature_tables_are_complete(not_cleared_data: Path):
    """Guard the row counts, units, and printed endpoints of four source tables."""
    with (not_cleared_data / "ahmadi_chapoy_2018_co2_water_solubility.csv").open() as stream:
        ahmadi = tuple(csv.DictReader(stream))
    assert len(ahmadi) == 29
    assert {float(row["temperature_K"]) for row in ahmadi} == {
        300.95,
        307.79,
        322.62,
        373.39,
        423.48,
    }
    assert ahmadi[0]["pressure_MPa"] == "2.071"
    assert ahmadi[-1]["x_co2_liquid"] == "0.02839"

    with (not_cleared_data / "wang_2021_co2_water_solubility_molality.csv").open() as stream:
        wang = tuple(csv.DictReader(stream))
    temperature_columns = tuple(key for key in wang[0] if key != "pressure_MPa")
    assert len(wang) * len(temperature_columns) == 240
    assert temperature_columns == (
        "313.15_K",
        "333.15_K",
        "353.15_K",
        "363.15_K",
        "373.15_K",
        "393.15_K",
        "413.15_K",
        "433.15_K",
        "453.15_K",
        "473.15_K",
    )
    assert wang[0]["313.15_K"] == "0.1169"
    assert wang[-1]["473.15_K"] == "5.3787"
    water_molar_mass = 0.018015268
    converted = float(wang[-1]["473.15_K"]) / (float(wang[-1]["473.15_K"]) + 1.0 / water_molar_mass)
    assert math.isclose(converted, 0.08833880471267616)

    with (not_cleared_data / "wang_2021_water_vapor_volume_fraction.csv").open() as stream:
        wang_vapor = tuple(csv.DictReader(stream))
    assert len(wang_vapor) * (len(wang_vapor[0]) - 1) == 56
    assert wang_vapor[0]["313.15_K"] == "0.017"
    assert wang_vapor[-1]["473.15_K"] == "0.116"

    with (not_cleared_data / "portier_rochelle_2005_utsira_brine_solubility.csv").open() as stream:
        portier = tuple(csv.DictReader(stream))
    assert len(portier) == 35
    assert {row["fluid"] for row in portier} == {"synthetic_Utsira_porewater"}
    assert math.isclose(
        sum(float(row["co2_molality_mol_per_kg_water"]) for row in portier),
        33.508,
    )


def _density_rows(system: tuple[str, str]) -> list[dict[str, str]]:
    with (DATA / "nist_thermoml_h2_binary_density.csv").open() as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if (row["component1"], row["component2"]) == system
        ]


@pytest.mark.serial
@pytest.mark.parametrize(
    ("system", "maximum_aard_percent"),
    [
        (("methane", "hydrogen"), 0.06),
        (("nitrogen", "hydrogen"), 0.05),
        (("hydrogen", "carbon_dioxide"), 0.20),
    ],
)
def test_eoscg_hydrogen_binary_density_against_experiment(
    system: tuple[str, str],
    maximum_aard_percent: float,
):
    """Regress three UHS-relevant binaries against primary density data."""
    model = eoscg2021(system)
    relative_deviations = []
    rows = _density_rows(system)
    assert rows
    for row in rows:
        fraction = float(row["mole_fraction_component1"])
        composition = torch.tensor([fraction, 1.0 - fraction], dtype=DTYPE)
        volume = model.molar_volume(
            torch.tensor(float(row["temperature_K"]), dtype=DTYPE),
            torch.tensor(float(row["pressure_Pa"]), dtype=DTYPE),
            composition,
            "vapor",
        )
        predicted = torch.dot(composition, model.molar_mass) / volume
        measured = predicted.new_tensor(float(row["density_kg_m3"]))
        relative_deviations.append(100.0 * torch.abs(predicted / measured - 1.0))

    aard = torch.stack(relative_deviations).mean()
    assert aard < maximum_aard_percent


@pytest.mark.serial
def test_eoscg_co2_methane_fixed_tp_vle_against_experiment(
    not_cleared_data: Path,
):
    """Regress a CCS binary at low, intermediate, and high pressure."""
    with (not_cleared_data / "jaubert_2020_co2_binary_vle.csv").open() as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if (row["component1"], row["component2"]) == ("methane", "carbon_dioxide")
        ]
    model = eoscg2021(("methane", "carbon_dioxide"))
    selected = (rows[2], rows[8], rows[14])
    liquid_errors = []
    vapor_errors = []
    for row in selected:
        liquid_first = float(row["x1"])
        vapor_first = float(row["y1"])
        point = binary_vle_point(
            model,
            torch.tensor(float(row["temperature_K"]), dtype=DTYPE),
            torch.tensor(float(row["pressure_bar"]) * 1.0e5, dtype=DTYPE),
            torch.tensor([liquid_first, 1.0 - liquid_first], dtype=DTYPE),
            torch.tensor([vapor_first, 1.0 - vapor_first], dtype=DTYPE),
        )
        assert point.converged
        liquid_errors.append(abs(float(point.liquid_composition[0]) - liquid_first))
        vapor_errors.append(abs(float(point.vapor_composition[0]) - vapor_first))

    assert sum(liquid_errors) / len(liquid_errors) < 0.012
    assert sum(vapor_errors) / len(vapor_errors) < 0.002


def test_binary_vle_rejects_nonbinary_compositions(binary_model):
    with pytest.raises(ValueError, match="two-component"):
        binary_vle_point(
            binary_model,
            torch.tensor(300.0, dtype=DTYPE),
            torch.tensor(1.0e6, dtype=DTYPE),
            torch.tensor([0.2, 0.3, 0.5], dtype=DTYPE),
            torch.tensor([0.4, 0.6], dtype=DTYPE),
        )


def test_binary_bubble_point_matches_general_saturation_solver(binary_model):
    temperature = torch.tensor(270.0, dtype=torch.float64)
    liquid = torch.tensor([0.2, 0.8], dtype=torch.float64)
    reference = saturation_point(
        binary_model,
        temperature,
        liquid,
        "bubble",
    )
    point = binary_bubble_point(
        binary_model,
        temperature,
        liquid,
        initial_pressure=reference.pressure,
        initial_vapor_composition=reference.incipient_composition,
    )

    assert point.converged
    torch.testing.assert_close(point.pressure, reference.pressure)
    torch.testing.assert_close(point.vapor_composition, reference.incipient_composition)
    torch.testing.assert_close(point.liquid_composition, liquid)


def test_binary_bubble_point_validates_inputs(binary_model):
    temperature = torch.tensor(270.0, dtype=torch.float64)
    liquid = torch.tensor([0.2, 0.8], dtype=torch.float64)
    with pytest.raises(ValueError, match="two-component"):
        binary_bubble_point(
            binary_model,
            temperature,
            torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="liquid composition"):
        binary_bubble_point(
            binary_model,
            temperature,
            torch.tensor([0.0, 1.0], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="pressure"):
        binary_bubble_point(
            binary_model,
            temperature,
            liquid,
            initial_pressure=torch.tensor(-1.0, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="vapor composition"):
        binary_bubble_point(
            binary_model,
            temperature,
            liquid,
            initial_vapor_composition=torch.tensor([0.0, 1.0], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="minimum binary bubble pressure"):
        binary_bubble_point(
            binary_model,
            temperature,
            liquid,
            minimum_pressure=-1.0,
        )
    with pytest.raises(ValueError, match="below maximum"):
        binary_bubble_point(
            binary_model,
            temperature,
            liquid,
            minimum_pressure=2.0e6,
            maximum_pressure=1.0e6,
        )


def test_binary_vle_reports_unfinished_newton_polish(binary_model):
    point = binary_vle_point(
        binary_model,
        torch.tensor(270.0, dtype=DTYPE),
        torch.tensor(3.0e6, dtype=DTYPE),
        torch.tensor([0.2, 0.8], dtype=DTYPE),
        torch.tensor([0.8, 0.2], dtype=DTYPE),
        tolerance=0.0,
        max_iterations=1,
    )
    assert point.iterations == 1
    assert not point.converged
    assert point.residual_norm >= 0.0


def test_binary_vle_rejects_trivial_single_phase_root(binary_model):
    composition = torch.tensor([0.4, 0.6], dtype=DTYPE)
    point = binary_vle_point(
        binary_model,
        torch.tensor(500.0, dtype=DTYPE),
        torch.tensor(1.0e5, dtype=DTYPE),
        composition,
        composition,
    )
    assert point.residual_norm < 1.0e-8
    assert torch.allclose(point.liquid_composition, point.vapor_composition)
    assert not point.converged


def test_binary_vle_rejects_negative_phase_separation(binary_model):
    with pytest.raises(ValueError, match="phase separation"):
        binary_vle_point(
            binary_model,
            torch.tensor(300.0, dtype=DTYPE),
            torch.tensor(1.0e6, dtype=DTYPE),
            torch.tensor([0.2, 0.8], dtype=DTYPE),
            torch.tensor([0.8, 0.2], dtype=DTYPE),
            minimum_phase_separation=-1.0,
        )


class _IdealFugacityModel:
    def log_fugacity_coefficients(self, temperature, pressure, composition, phase="stable"):
        return torch.zeros_like(composition)


class _NoBinaryClosureModel(_IdealFugacityModel):
    def log_fugacity_coefficients(self, temperature, pressure, composition, phase="stable"):
        if phase == "liquid":
            return torch.log(composition.new_tensor([2.0, 3.0]))
        return torch.zeros_like(composition)


@pytest.mark.parametrize("model", [_IdealFugacityModel(), _NoBinaryClosureModel()])
def test_binary_vle_substitution_fallbacks(model):
    point = binary_vle_point(
        model,
        torch.tensor(300.0, dtype=DTYPE),
        torch.tensor(1.0e6, dtype=DTYPE),
        torch.tensor([0.2, 0.8], dtype=DTYPE),
        torch.tensor([0.8, 0.2], dtype=DTYPE),
        max_iterations=1,
    )
    assert not point.converged
