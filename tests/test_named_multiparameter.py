from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import torch_flash.envelope as envelope
import torch_flash.eos.named as named
from torch_flash.database import ModelParameterSet, load_model_parameters
from torch_flash.envelope import (
    BinaryBubblePointWithVolumes,
    binary_bubble_point,
    binary_helmholtz_bubble_point,
    binary_helmholtz_vle_point,
    binary_vle_point,
    phase_envelope,
    trace_binary_helmholtz_fixed_composition_boundary,
    trace_binary_helmholtz_pxy_isotherm,
)
from torch_flash.eos import (
    EOSCG2021_COMPONENTS,
    GERG2008_COMPONENTS,
    GERG2008_HYDROGEN_2021_COMPONENTS,
    GaoBTerms,
    NonAnalyticTerms,
    eoscg2021,
    gerg2008,
    gerg2008_hydrogen_2021,
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

    hydrogen_gerg = gerg2008_hydrogen_2021()
    assert hydrogen_gerg.names == GERG2008_HYDROGEN_2021_COMPONENTS
    assert hydrogen_gerg.metadata.validated_components == GERG2008_HYDROGEN_2021_COMPONENTS
    assert hydrogen_gerg.pure_n.shape[0] == 5
    assert hydrogen_gerg.beta_temperature.shape == (5, 5)
    assert torch.count_nonzero(torch.triu(hydrogen_gerg.departure_scale, diagonal=1)) == 7
    torch.testing.assert_close(
        hydrogen_gerg.gas_constant,
        torch.tensor(8.314472, dtype=DTYPE),
    )
    assert hydrogen_gerg.has_ideal_terms
    torch.testing.assert_close(
        hydrogen_gerg.pure_n[-1, 11],
        torch.tensor(0.032187, dtype=DTYPE),
    )


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
    assert gerg2008_hydrogen_2021(("H2", "CO2")).names == (
        "hydrogen",
        "carbon_dioxide",
    )


@pytest.mark.parametrize(
    ("component", "temperature", "density", "pressure_mpa"),
    [
        ("methane", 250.0, 20_000.0, 54.33554554),
        ("nitrogen", 250.0, 20_000.0, 65.60743718),
        ("carbon_monoxide", 250.0, 20_000.0, 62.38126662),
        ("carbon_dioxide", 350.0, 20_000.0, 69.13317365),
    ],
)
def test_hydrogen_tailored_gerg_matches_paper_table12_pressure_checks(
    component,
    temperature,
    density,
    pressure_mpa,
):
    """Check the four Table 12 states also embedded in the authors' supplement."""
    model = gerg2008_hydrogen_2021((component, "hydrogen"))
    composition = torch.tensor([0.6, 0.4], dtype=DTYPE)
    predicted = model.pressure(
        torch.tensor(temperature, dtype=DTYPE),
        torch.tensor(density, dtype=DTYPE).reciprocal(),
        composition,
    )
    torch.testing.assert_close(
        predicted,
        torch.tensor(pressure_mpa * 1.0e6, dtype=DTYPE),
        rtol=3.0e-10,
        atol=3.0e-2,
    )


def test_hydrogen_tailored_gerg_co2_pair_parameters_and_autodiff():
    model = gerg2008_hydrogen_2021(("carbon_dioxide", "hydrogen"), trainable=True)
    torch.testing.assert_close(model.beta_temperature[0, 1], torch.tensor(0.964, dtype=DTYPE))
    torch.testing.assert_close(model.gamma_temperature[0, 1], torch.tensor(2.014, dtype=DTYPE))
    torch.testing.assert_close(model.beta_volume[0, 1], torch.tensor(1.200, dtype=DTYPE))
    torch.testing.assert_close(model.gamma_volume[0, 1], torch.tensor(0.825, dtype=DTYPE))
    temperature = torch.tensor(350.0, dtype=DTYPE)
    density = torch.tensor(20_000.0, dtype=DTYPE)
    composition = torch.tensor([0.6, 0.4], dtype=DTYPE)
    pressure = model.pressure(temperature, density.reciprocal(), composition)
    pressure.backward()
    assert model.departure_n.grad is not None
    assert torch.isfinite(model.departure_n.grad).all()


def test_hydrogen_tailored_gerg_volume_bubble_continuation_matches_pressure_form():
    model = gerg2008_hydrogen_2021(("nitrogen", "hydrogen"))
    temperature = torch.tensor(90.8, dtype=DTYPE)
    initial_liquid = torch.tensor([0.79, 0.21], dtype=DTYPE)
    continued_liquid = torch.tensor([0.75, 0.25], dtype=DTYPE)
    initial = binary_helmholtz_bubble_point(
        model,
        temperature,
        initial_liquid,
        minimum_pressure=1.0e3,
        maximum_pressure=3.0e8,
    )
    volume_result = binary_helmholtz_bubble_point(
        model,
        temperature,
        continued_liquid,
        initial_point=initial,
        minimum_pressure=1.0e3,
        maximum_pressure=3.0e8,
    )
    pressure_result = binary_bubble_point(
        model,
        temperature,
        continued_liquid,
        initial_pressure=initial.pressure,
        initial_vapor_composition=initial.vapor_composition,
        minimum_pressure=1.0e3,
        maximum_pressure=3.0e8,
    )
    assert initial.converged
    assert volume_result.converged
    torch.testing.assert_close(
        volume_result.pressure,
        pressure_result.pressure,
        rtol=3.0e-10,
        atol=2.0e-3,
    )
    torch.testing.assert_close(
        volume_result.vapor_composition,
        pressure_result.vapor_composition,
        rtol=3.0e-10,
        atol=2.0e-11,
    )
    temperature_step = torch.tensor(1.0e-4, dtype=DTYPE)

    def continued_pressure(current_temperature):
        return binary_helmholtz_bubble_point(
            model,
            current_temperature,
            continued_liquid,
            initial_point=initial,
            tolerance=1.0e-10,
        ).pressure

    autodiff = torch.func.grad(continued_pressure)(temperature)
    finite_difference = (
        continued_pressure(temperature + temperature_step)
        - continued_pressure(temperature - temperature_step)
    ) / (2.0 * temperature_step)
    torch.testing.assert_close(
        autodiff,
        finite_difference,
        rtol=3.0e-8,
        atol=2.0e-3,
    )


def test_hydrogen_tailored_gerg_high_level_pxy_isotherm_and_autodiff():
    model = gerg2008_hydrogen_2021(("nitrogen", "hydrogen"))
    fractions = torch.linspace(0.002, 0.47, 5, dtype=DTYPE, requires_grad=True)
    liquid = torch.stack((1.0 - fractions, fractions), dim=-1)
    result = trace_binary_helmholtz_pxy_isotherm(
        model,
        torch.tensor(90.8, dtype=DTYPE),
        liquid,
        minimum_pressure=1.0e3,
        maximum_pressure=1.05e8,
        max_iterations=25,
    )
    assert result.pressure.shape == (5,)
    assert result.liquid_composition.shape == (5, 2)
    assert result.vapor_composition.shape == (5, 2)
    assert result.iterations.dtype == torch.int64
    assert result.converged.tolist() == [True] * 5
    assert torch.all(result.residual_norm <= 1.0e-8)
    assert torch.all(result.phase_separation > 1.0e-5)
    torch.testing.assert_close(
        result.pressure,
        torch.tensor(
            [
                473604.53876677965,
                4860041.687633225,
                9005470.309133109,
                12691482.24207196,
                15030260.898923744,
            ],
            dtype=DTYPE,
        ),
        rtol=2.0e-10,
        atol=2.0e-3,
    )
    gradient = torch.autograd.grad(result.pressure.sum(), fractions)[0]
    assert torch.isfinite(gradient).all()
    assert torch.all(gradient > 0.0)


def test_gerg2008_open_pxy_branch_continues_through_liquid_composition_fold():
    parameter = torch.linspace(0.0, 1.0, 61, dtype=DTYPE)
    liquid_hydrogen = 0.002 + (0.98 - 0.002) * parameter.pow(2.5)
    result = trace_binary_helmholtz_pxy_isotherm(
        gerg2008(("nitrogen", "hydrogen")),
        torch.tensor(70.4, dtype=DTYPE),
        torch.stack((1.0 - liquid_hydrogen, liquid_hydrogen), dim=-1),
        minimum_pressure=1.0e3,
        maximum_pressure=105.0e6,
        max_iterations=40,
        composition_failure_refinement_steps=8,
        continue_in_pressure_on_failure=True,
        pressure_continuation_points=10,
    )
    accepted = result.converged
    assert int(accepted.sum()) >= 20
    torch.testing.assert_close(
        result.pressure[accepted][-1],
        torch.tensor(105.0e6, dtype=DTYPE),
        rtol=2.0e-10,
        atol=2.0e-2,
    )
    assert torch.all(result.residual_norm[accepted] <= 1.0e-8)
    accepted_liquid = result.liquid_composition[accepted, 1]
    assert accepted_liquid[-1] < accepted_liquid.max()
    assert result.vapor_composition[accepted, 1][-1] > result.vapor_composition[accepted, 1][-10]


def test_hydrogen_tailored_gerg_pxy_closes_and_resolves_low_pressure_vapor_branch():
    parameter = torch.linspace(0.0, 1.0, 31, dtype=DTYPE)
    liquid_hydrogen = 0.002 + (0.98 - 0.002) * parameter.pow(2.5)
    result = trace_binary_helmholtz_pxy_isotherm(
        gerg2008_hydrogen_2021(("carbon_dioxide", "hydrogen")),
        torch.tensor(235.0, dtype=DTYPE),
        torch.stack((1.0 - liquid_hydrogen, liquid_hydrogen), dim=-1),
        minimum_pressure=1.0e3,
        maximum_pressure=315.0e6,
        max_iterations=40,
        composition_failure_refinement_steps=8,
        continue_in_pressure_on_failure=True,
        pressure_continuation_points=4,
        pressure_failure_refinement_steps=14,
    )
    accepted = result.converged
    liquid = result.liquid_composition[accepted, 1]
    vapor = result.vapor_composition[accepted, 1]
    pressure_MPa = result.pressure[accepted] / 1.0e6
    assert result.phase_separation[accepted][-1] <= 1.0e-3
    torch.testing.assert_close(
        result.pressure[accepted][-1],
        torch.tensor(182.36462144e6, dtype=DTYPE),
        rtol=2.0e-8,
        atol=2.0,
    )
    torch.testing.assert_close(
        liquid[-1],
        vapor[-1],
        rtol=0.0,
        atol=1.0e-3,
    )
    low_pressure_vapor = vapor[pressure_MPa <= 30.0]
    assert torch.abs(torch.diff(low_pressure_vapor)).max() <= 0.15


def test_helmholtz_volume_vle_point_matches_pressure_form():
    model = gerg2008(("nitrogen", "hydrogen"))
    temperature = torch.tensor(70.4, dtype=DTYPE)
    initial = binary_helmholtz_bubble_point(
        model,
        temperature,
        torch.tensor([0.942, 0.058], dtype=DTYPE),
        minimum_pressure=1.0e3,
        maximum_pressure=105.0e6,
    )
    pressure = torch.tensor(40.0e6, dtype=DTYPE)
    volume_result = binary_helmholtz_vle_point(
        model,
        temperature,
        pressure,
        initial,
        max_iterations=40,
    )
    pressure_result = binary_vle_point(
        model,
        temperature,
        pressure,
        initial.liquid_composition,
        initial.vapor_composition,
        max_iterations=40,
    )
    assert initial.converged
    assert volume_result.converged and pressure_result.converged
    assert volume_result.liquid_molar_volume > 0.0
    assert volume_result.vapor_molar_volume > 0.0
    torch.testing.assert_close(
        volume_result.liquid_composition,
        pressure_result.liquid_composition,
        rtol=3.0e-8,
        atol=2.0e-9,
    )
    torch.testing.assert_close(
        volume_result.vapor_composition,
        pressure_result.vapor_composition,
        rtol=3.0e-8,
        atol=2.0e-9,
    )
    assert volume_result.residual_norm <= 1.0e-8


def test_helmholtz_volume_vle_point_validates_inputs():
    model = gerg2008(("nitrogen", "hydrogen"))
    temperature = torch.tensor(70.4, dtype=DTYPE)
    pressure = torch.tensor(40.0e6, dtype=DTYPE)
    initial = binary_helmholtz_bubble_point(
        model,
        temperature,
        torch.tensor([0.942, 0.058], dtype=DTYPE),
        minimum_pressure=1.0e3,
        maximum_pressure=105.0e6,
    )
    with pytest.raises(TypeError, match="Helmholtz pressure method"):
        binary_helmholtz_vle_point(object(), temperature, pressure, initial)
    with pytest.raises(ValueError, match="temperature"):
        binary_helmholtz_vle_point(model, temperature.reshape(1), pressure, initial)
    with pytest.raises(ValueError, match="pressure"):
        binary_helmholtz_vle_point(model, temperature, pressure.reshape(1), initial)
    invalid_liquid = replace(
        initial,
        liquid_composition=torch.tensor([0.2, 0.3, 0.5], dtype=DTYPE),
    )
    with pytest.raises(ValueError, match="compositions"):
        binary_helmholtz_vle_point(model, temperature, pressure, invalid_liquid)
    invalid_vapor = replace(
        initial,
        vapor_composition=torch.tensor([1.0, 0.0], dtype=DTYPE),
    )
    with pytest.raises(ValueError, match="compositions"):
        binary_helmholtz_vle_point(model, temperature, pressure, invalid_vapor)
    invalid_volume = replace(
        initial,
        liquid_molar_volume=torch.tensor(torch.nan, dtype=DTYPE),
    )
    with pytest.raises(ValueError, match="molar volumes"):
        binary_helmholtz_vle_point(model, temperature, pressure, invalid_volume)
    with pytest.raises(ValueError, match="tolerance"):
        binary_helmholtz_vle_point(model, temperature, pressure, initial, tolerance=0.0)
    with pytest.raises(ValueError, match="max_iterations"):
        binary_helmholtz_vle_point(model, temperature, pressure, initial, max_iterations=0)
    with pytest.raises(ValueError, match="phase separation"):
        binary_helmholtz_vle_point(
            model,
            temperature,
            pressure,
            initial,
            minimum_phase_separation=-1.0,
        )


def test_high_level_pxy_isotherm_stops_on_failure_and_pressure_limit(monkeypatch):
    calls = 0

    def fake_bubble_point(model, temperature, liquid, **options):
        del model, options
        nonlocal calls
        calls += 1
        converged = calls != 2
        pressure = liquid.new_tensor(1.0e6 * calls)
        vapor = torch.stack((liquid[0] - 0.1, liquid[1] + 0.1))
        return BinaryBubblePointWithVolumes(
            temperature,
            pressure,
            liquid,
            vapor,
            2,
            converged,
            liquid.new_tensor(1.0e-10 if converged else 1.0),
            liquid.new_tensor(1.0e-4),
            liquid.new_tensor(1.0e-3),
        )

    monkeypatch.setattr(envelope, "binary_helmholtz_bubble_point", fake_bubble_point)
    liquid = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]], dtype=DTYPE)
    failed = trace_binary_helmholtz_pxy_isotherm(
        object(),
        torch.tensor(100.0, dtype=DTYPE),
        liquid,
    )
    assert failed.converged.tolist() == [True, False]

    calls = 0

    def pressure_limited_point(model, temperature, composition, **options):
        del model, options
        nonlocal calls
        calls += 1
        vapor = torch.stack((composition[0] - 0.1, composition[1] + 0.1))
        return BinaryBubblePointWithVolumes(
            temperature,
            composition.new_tensor(9.96e6),
            composition,
            vapor,
            1,
            True,
            composition.new_tensor(1.0e-10),
            composition.new_tensor(1.0e-4),
            composition.new_tensor(1.0e-3),
        )

    monkeypatch.setattr(envelope, "binary_helmholtz_bubble_point", pressure_limited_point)
    limited = trace_binary_helmholtz_pxy_isotherm(
        object(),
        torch.tensor(100.0, dtype=DTYPE),
        liquid,
        maximum_pressure=1.0e7,
    )
    assert calls == 1
    assert limited.converged.tolist() == [True]


def test_hydrogen_tailored_gerg_high_level_fixed_composition_boundary():
    result = trace_binary_helmholtz_fixed_composition_boundary(
        gerg2008_hydrogen_2021(("methane", "hydrogen")),
        torch.linspace(155.0, 60.0, 7, dtype=DTYPE),
        torch.tensor([0.5, 0.5], dtype=DTYPE),
        reporting_pressure_limit=70.0e6,
        maximum_pressure=75.0e6,
    )
    assert result.bubble_converged.tolist() == [
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert result.bubble_above_reporting_limit.tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert result.dew_converged.tolist() == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert result.dew_below_scan.tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    torch.testing.assert_close(
        result.bubble_pressure[1:4],
        torch.tensor(
            [31128040.37708861, 41983556.14551598, 63452327.12045838],
            dtype=DTYPE,
        ),
        rtol=3.0e-9,
        atol=3.0e-2,
    )
    torch.testing.assert_close(
        result.dew_pressure[:5],
        torch.tensor(
            [
                3353956.8690104545,
                1398163.4914000928,
                513559.3041102533,
                147002.6978222458,
                26947.28598205394,
            ],
            dtype=DTYPE,
        ),
        rtol=3.0e-9,
        atol=3.0e-3,
    )
    assert torch.all(result.bubble_separation[result.bubble_converged] > 1.0e-5)
    finite_residuals = torch.cat(
        (
            result.bubble_residual[result.bubble_converged],
            result.dew_residual[result.dew_converged],
        )
    )
    assert torch.all(finite_residuals <= 1.0e-8)


def test_fixed_composition_boundary_interpolates_return_crossing(monkeypatch):
    def fake_bubble_point(model, temperature, liquid, **options):
        del model, options
        fraction = float(liquid[1])
        if fraction < 0.01:
            vapor_fraction = 0.3
        elif fraction < 0.1:
            vapor_fraction = 0.7
        else:
            vapor_fraction = 0.3
        vapor = torch.stack(
            (liquid.new_tensor(1.0 - vapor_fraction), liquid.new_tensor(vapor_fraction))
        )
        pressure = liquid.new_tensor(1.0e6 * (1.0 + 5.0 * fraction))
        return BinaryBubblePointWithVolumes(
            temperature,
            pressure,
            liquid,
            vapor,
            1,
            True,
            liquid.new_tensor(1.0e-10),
            liquid.new_tensor(1.0e-4),
            liquid.new_tensor(1.0e-3),
        )

    monkeypatch.setattr(envelope, "binary_helmholtz_bubble_point", fake_bubble_point)
    result = trace_binary_helmholtz_fixed_composition_boundary(
        object(),
        torch.tensor([100.0], dtype=DTYPE),
        torch.tensor([0.5, 0.5], dtype=DTYPE),
        reporting_pressure_limit=10.0e6,
        maximum_pressure=11.0e6,
        lean_scan_points=2,
        rich_scan_points=4,
    )
    assert result.dew_converged.tolist() == [True]
    assert result.bubble_converged.tolist() == [True]
    assert 0.0 < float(result.bubble_pressure[0]) < 10.0e6
    assert float(result.bubble_separation[0]) > 0.0


def test_fixed_composition_boundary_retries_failed_initializations(monkeypatch):
    calls_by_state = {}

    def fake_bubble_point(model, temperature, liquid, **options):
        del model
        fraction = round(float(liquid[1]), 8)
        key = (float(temperature), fraction)
        calls_by_state[key] = calls_by_state.get(key, 0) + 1
        initial_point = options.get("initial_point")
        first_state_call = calls_by_state[key] == 1
        first_temperature_start = float(temperature) == 100.0 and fraction < 1.1e-5
        stale_temperature_guess = (
            float(temperature) == 99.0
            and initial_point is not None
            and float(initial_point.temperature) == 100.0
            and fraction > 1.1e-5
        )
        converged = not (
            (first_temperature_start and first_state_call)
            or (stale_temperature_guess and first_state_call)
        )
        vapor_fraction = min(fraction + 0.01, 0.99)
        vapor = torch.stack(
            (liquid.new_tensor(1.0 - vapor_fraction), liquid.new_tensor(vapor_fraction))
        )
        return BinaryBubblePointWithVolumes(
            temperature,
            liquid.new_tensor(2.0e6 + 1.0e5 * fraction),
            liquid,
            vapor,
            1,
            converged,
            liquid.new_tensor(1.0e-10 if converged else 1.0),
            liquid.new_tensor(1.0e-4),
            liquid.new_tensor(1.0e-3),
        )

    monkeypatch.setattr(envelope, "binary_helmholtz_bubble_point", fake_bubble_point)
    result = trace_binary_helmholtz_fixed_composition_boundary(
        object(),
        torch.tensor([100.0, 99.0], dtype=DTYPE),
        torch.tensor([0.98, 0.02], dtype=DTYPE),
        reporting_pressure_limit=10.0e6,
        maximum_pressure=11.0e6,
        lean_scan_points=3,
        rich_scan_points=3,
    )
    assert calls_by_state[(100.0, 1.0e-5)] == 2
    assert any(
        count == 2 for (temperature, _), count in calls_by_state.items() if temperature == 99.0
    )
    assert result.bubble_converged.tolist() == [True, True]
    assert result.dew_converged.tolist() == [True, True]


def test_fixed_composition_boundary_clips_midscan_pressure(monkeypatch):
    def fake_bubble_point(model, temperature, liquid, **options):
        del model, options
        fraction = float(liquid[1])
        vapor_fraction = min(fraction + 0.4, 0.99)
        vapor = torch.stack(
            (liquid.new_tensor(1.0 - vapor_fraction), liquid.new_tensor(vapor_fraction))
        )
        return BinaryBubblePointWithVolumes(
            temperature,
            liquid.new_tensor(12.0e6),
            liquid,
            vapor,
            1,
            True,
            liquid.new_tensor(1.0e-10),
            liquid.new_tensor(1.0e-4),
            liquid.new_tensor(1.0e-3),
        )

    monkeypatch.setattr(envelope, "binary_helmholtz_bubble_point", fake_bubble_point)
    result = trace_binary_helmholtz_fixed_composition_boundary(
        object(),
        torch.tensor([100.0], dtype=DTYPE),
        torch.tensor([0.5, 0.5], dtype=DTYPE),
        reporting_pressure_limit=10.0e6,
        maximum_pressure=15.0e6,
        lean_scan_points=2,
        rich_scan_points=3,
    )
    torch.testing.assert_close(result.bubble_pressure, torch.tensor([10.0e6], dtype=DTYPE))
    assert result.bubble_above_reporting_limit.tolist() == [True]


@pytest.mark.parametrize(
    ("temperature", "liquid", "options", "message"),
    [
        (
            torch.tensor([90.8], dtype=DTYPE),
            torch.tensor([[0.8, 0.2]], dtype=DTYPE),
            {},
            "one finite positive temperature",
        ),
        (
            torch.tensor(90.8, dtype=DTYPE),
            torch.tensor([[8, 2]]),
            {},
            "floating dtype",
        ),
        (
            torch.tensor(90.8, dtype=DTYPE),
            torch.empty((0, 2), dtype=DTYPE),
            {},
            r"shape \(points, 2\)",
        ),
        (
            torch.tensor(90.8, dtype=DTYPE),
            torch.tensor([[1.0, 0.0]], dtype=DTYPE),
            {},
            "finite and positive",
        ),
        (
            torch.tensor(90.8, dtype=DTYPE),
            torch.tensor([[0.8, 0.2]], dtype=DTYPE),
            {"tolerance": 0.0},
            "tolerance",
        ),
        (
            torch.tensor(90.8, dtype=DTYPE),
            torch.tensor([[0.8, 0.2]], dtype=DTYPE),
            {"max_iterations": 0},
            "max_iterations",
        ),
        (
            torch.tensor(90.8, dtype=DTYPE),
            torch.tensor([[0.8, 0.2]], dtype=DTYPE),
            {"minimum_phase_separation": -1.0},
            "phase separation",
        ),
        (
            torch.tensor(90.8, dtype=DTYPE),
            torch.tensor([[0.8, 0.2]], dtype=DTYPE),
            {"composition_failure_refinement_steps": -1},
            "refinement steps",
        ),
        (
            torch.tensor(90.8, dtype=DTYPE),
            torch.tensor([[0.8, 0.2]], dtype=DTYPE),
            {"pressure_continuation_points": 0},
            "continuation points",
        ),
        (
            torch.tensor(90.8, dtype=DTYPE),
            torch.tensor([[0.8, 0.2]], dtype=DTYPE),
            {"pressure_failure_refinement_steps": -1},
            "pressure failure refinement steps",
        ),
        (
            torch.tensor(90.8, dtype=DTYPE),
            torch.tensor([[0.8, 0.2]], dtype=DTYPE),
            {"continue_in_pressure_on_failure": True},
            "requires maximum_pressure",
        ),
        (
            torch.tensor(90.8, dtype=DTYPE),
            torch.tensor([[0.8, 0.2]], dtype=DTYPE),
            {
                "continue_in_pressure_on_failure": True,
                "maximum_pressure": 1.0e8,
                "stop_on_failure": False,
            },
            "requires stop_on_failure",
        ),
    ],
)
def test_trace_binary_helmholtz_pxy_isotherm_validates_high_level_inputs(
    temperature,
    liquid,
    options,
    message,
):
    with pytest.raises(ValueError, match=message):
        trace_binary_helmholtz_pxy_isotherm(
            gerg2008_hydrogen_2021(("nitrogen", "hydrogen")),
            temperature,
            liquid,
            **options,
        )


@pytest.mark.parametrize(
    ("temperatures", "composition", "options", "message"),
    [
        (
            torch.tensor([100]),
            torch.tensor([0.5, 0.5], dtype=DTYPE),
            {},
            "nonempty float vector",
        ),
        (
            torch.tensor([[100.0]], dtype=DTYPE),
            torch.tensor([0.5, 0.5], dtype=DTYPE),
            {},
            "nonempty float vector",
        ),
        (
            torch.tensor([100.0], dtype=DTYPE),
            torch.tensor([1, 1]),
            {},
            "floating dtype",
        ),
        (
            torch.tensor([100.0], dtype=DTYPE),
            torch.tensor([1.0, 0.0], dtype=DTYPE),
            {},
            "two finite positive",
        ),
        (
            torch.tensor([100.0], dtype=DTYPE),
            torch.tensor([0.5, 0.5], dtype=DTYPE),
            {"volatile_component_index": 2},
            "must be 0 or 1",
        ),
        (
            torch.tensor([100.0], dtype=DTYPE),
            torch.tensor([0.5, 0.5], dtype=DTYPE),
            {"lean_scan_points": 1},
            "at least two",
        ),
        (
            torch.tensor([100.0], dtype=DTYPE),
            torch.tensor([0.5, 0.5], dtype=DTYPE),
            {"max_iterations": 0},
            "max_iterations",
        ),
        (
            torch.tensor([100.0], dtype=DTYPE),
            torch.tensor([0.5, 0.5], dtype=DTYPE),
            {"minimum_phase_separation": 0.0},
            "minimum_phase_separation",
        ),
        (
            torch.tensor([100.0], dtype=DTYPE),
            torch.tensor([0.5, 0.5], dtype=DTYPE),
            {"reporting_pressure_limit": torch.tensor([70.0e6], dtype=DTYPE)},
            "pressure limits",
        ),
        (
            torch.tensor([100.0], dtype=DTYPE),
            torch.tensor([0.5, 0.5], dtype=DTYPE),
            {"maximum_pressure": 60.0e6},
            "minimum < reporting <= maximum",
        ),
        (
            torch.tensor([100.0], dtype=DTYPE),
            torch.tensor([0.999995, 0.000005], dtype=DTYPE),
            {},
            "minimum volatile liquid fraction",
        ),
    ],
)
def test_trace_binary_helmholtz_fixed_composition_boundary_validates_high_level_inputs(
    temperatures,
    composition,
    options,
    message,
):
    settings = {
        "reporting_pressure_limit": 70.0e6,
        "maximum_pressure": 75.0e6,
    }
    settings.update(options)
    with pytest.raises(ValueError, match=message):
        trace_binary_helmholtz_fixed_composition_boundary(
            gerg2008_hydrogen_2021(("methane", "hydrogen")),
            temperatures,
            composition,
            **settings,
        )


def test_eoscg_volume_bubble_continuation_matches_pressure_form():
    model = eoscg2021(("carbon_dioxide", "hydrogen"))
    temperature = torch.tensor(260.0, dtype=DTYPE)
    initial = binary_helmholtz_bubble_point(
        model,
        temperature,
        torch.tensor([0.98, 0.02], dtype=DTYPE),
        minimum_pressure=1.0e3,
        maximum_pressure=3.0e8,
    )
    liquid = torch.tensor([0.94, 0.06], dtype=DTYPE)
    volume_result = binary_helmholtz_bubble_point(
        model,
        temperature,
        liquid,
        initial_point=initial,
        minimum_pressure=1.0e3,
        maximum_pressure=3.0e8,
    )
    pressure_result = binary_bubble_point(
        model,
        temperature,
        liquid,
        initial_pressure=initial.pressure,
        initial_vapor_composition=initial.vapor_composition,
        minimum_pressure=1.0e3,
        maximum_pressure=3.0e8,
    )
    assert initial.converged and volume_result.converged
    torch.testing.assert_close(
        volume_result.pressure,
        pressure_result.pressure,
        rtol=3.0e-8,
        atol=2.0e-2,
    )
    torch.testing.assert_close(
        volume_result.vapor_composition,
        pressure_result.vapor_composition,
        rtol=3.0e-8,
        atol=2.0e-9,
    )


def test_hydrogen_gerg_batched_stable_roots_match_scalar_and_reject_discontinuity():
    model = gerg2008_hydrogen_2021(("methane", "hydrogen"))
    temperature = torch.tensor([60.0, 380.0, 700.0], dtype=DTYPE)
    pressure = torch.tensor([0.1e6, 35.05e6, 70.0e6], dtype=DTYPE)
    composition = torch.tensor(
        [[0.75, 0.25], [0.75, 0.25], [0.25, 0.75]],
        dtype=DTYPE,
    )
    batched = model.molar_volume(
        temperature,
        pressure,
        composition,
        "stable",
    )
    scalar = torch.stack(
        [
            model.molar_volume(current_temperature, current_pressure, current_x, "stable")
            for current_temperature, current_pressure, current_x in zip(
                temperature,
                pressure,
                composition,
                strict=True,
            )
        ]
    )
    torch.testing.assert_close(batched, scalar, rtol=2.0e-11, atol=1.0e-15)
    pressure_residual = (model.pressure(temperature, batched, composition) - pressure) / pressure
    assert pressure_residual.abs().max() < 1.0e-8
    # At the cold state a pressure discontinuity changes sign near
    # 10.9 kmol/m3; it is not a root. The admissible stable root is the
    # lower-Gibbs liquid root near 33.6 kmol/m3.
    torch.testing.assert_close(
        batched[0].reciprocal(),
        torch.tensor(33_596.7603953449, dtype=DTYPE),
        rtol=2.0e-11,
        atol=2.0e-7,
    )


@pytest.mark.parametrize("constructor", [gerg2008, gerg2008_hydrogen_2021])
def test_gerg_batched_stable_subreducing_multiple_roots_match_scalar(constructor):
    model = constructor(("methane", "hydrogen"))
    temperature = torch.tensor(
        [100.33351709028807, 61.08398971620922, 80.47340262213496, 82.23703959248205],
        dtype=DTYPE,
    )
    pressure = (
        torch.tensor(
            [1.6402384884188868, 31.818839712354517, 50.08630196825132, 31.508474133469587],
            dtype=DTYPE,
        )
        * 1.0e6
    )
    hydrogen = torch.tensor(
        [0.3834694208197475, 0.5143382254152788, 0.17002802054457583, 0.17602746517778295],
        dtype=DTYPE,
    )
    composition = torch.stack((1.0 - hydrogen, hydrogen), dim=-1)
    batched = model.molar_volume(temperature, pressure, composition, "stable")
    scalar = torch.stack(
        [
            model.molar_volume(current_temperature, current_pressure, current_x, "stable")
            for current_temperature, current_pressure, current_x in zip(
                temperature,
                pressure,
                composition,
                strict=True,
            )
        ]
    )
    torch.testing.assert_close(batched, scalar, rtol=3.0e-11, atol=2.0e-15)
    pressure_residual = (model.pressure(temperature, batched, composition) - pressure) / pressure
    assert pressure_residual.abs().max() < 1.0e-8


@pytest.mark.parametrize(
    ("constructor", "names"),
    [
        (gerg2008_hydrogen_2021, ("carbon_dioxide", "hydrogen")),
        (eoscg2021, ("carbon_dioxide", "water")),
    ],
)
def test_multiparameter_analytic_pressure_derivative_matches_autodiff(constructor, names):
    model = constructor(names)
    temperature = torch.tensor([310.0, 450.0], dtype=DTYPE)
    density = torch.tensor([500.0, 12_000.0], dtype=DTYPE)
    composition = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=DTYPE)
    autodiff_derivative = torch.func.grad(
        lambda current: model.alpha_residual(
            temperature,
            current,
            composition,
        ).sum()
    )(density)
    autodiff_pressure = (
        model.gas_constant * temperature * density * (1.0 + density * autodiff_derivative)
    )
    analytic_pressure = model.pressure(
        temperature,
        density.reciprocal(),
        composition,
    )
    torch.testing.assert_close(
        analytic_pressure,
        autodiff_pressure,
        rtol=3.0e-13,
        atol=2.0e-4,
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
    broken_gerg = load_model_parameters("multiparameter.gerg-2008").as_dict()
    broken_gerg["pairs"].pop(next(iter(broken_gerg["pairs"])))
    monkeypatch.setattr(named, "_read_data", lambda filename: broken_gerg)
    with pytest.raises(RuntimeError, match="GERG-2008"):
        gerg2008(("methane",))

    broken_eoscg = load_model_parameters("multiparameter.eos-cg-2021").as_dict()
    broken_eoscg["components"].pop("mdea")
    monkeypatch.setattr(named, "_read_data", lambda filename: broken_eoscg)
    with pytest.raises(RuntimeError, match="EOS-CG-2021"):
        eoscg2021(("carbon_dioxide",))

    with pytest.raises(named.ParameterDatabaseError, match="H2-tailored GERG"):
        gerg2008_hydrogen_2021(parameter_set="multiparameter.gerg-2008")


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
    payload = load_model_parameters("multiparameter.gerg-2008").as_dict()
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
        "multiparameter.gerg-form-co2-water-test",
        "multiparameter",
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
