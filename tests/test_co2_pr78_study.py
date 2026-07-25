from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from torch_flash import (
    ChemicalState,
    ComponentSet,
    batched_two_phase_flash,
    binary_critical_point,
    continue_saturation_branch,
    fugacities_tv,
    log_fugacities_tv,
    peng_robinson_1978,
    phase_envelope,
    phase_properties,
    saturation_point,
    state_derivatives,
    two_phase_flash,
)
from torch_flash.constants import R
from torch_flash.flash.batched import _admissible_update


def _thermopack_pr78_model():
    """PR78 inputs used by ThermoPack 2.2.3 for the external CO2/N2 study."""
    dtype = torch.float64
    components = ComponentSet(
        ("carbon_dioxide", "nitrogen"),
        torch.tensor([304.2, 126.161], dtype=dtype),
        torch.tensor([7_376_500.0, 3_394_400.0], dtype=dtype),
        torch.tensor([0.225, 0.04], dtype=dtype),
        torch.tensor([0.0440095, 0.0280134], dtype=dtype),
    )
    kij = torch.tensor([[0.0, -0.036], [-0.036, 0.0]], dtype=dtype)
    # ThermoPack reports v_shifted = v_parent - ci. torch-flash defines the
    # API value as what is added to the parent-EoS volume.
    translation = torch.tensor(
        [2.146134330985881e-6, 4.6863936895102035e-6],
        dtype=dtype,
    )
    return peng_robinson_1978(
        components,
        kij=kij,
        volume_translation=translation,
    )


def test_translated_pr78_matches_frozen_thermopack_state_and_derivatives():
    model = _thermopack_pr78_model()
    state = ChemicalState(
        torch.tensor(283.15, dtype=torch.float64),
        torch.tensor(6.0e6, dtype=torch.float64),
        torch.tensor([0.9, 0.1], dtype=torch.float64),
    )
    result = two_phase_flash(
        model,
        state,
        check_stability=False,
        tolerance=1.0e-12,
    )
    assert result.converged
    torch.testing.assert_close(
        result.phase_fractions,
        torch.tensor(
            [0.44283271690073256, 0.55716728309926755],
            dtype=torch.float64,
        ),
        rtol=2.0e-12,
        atol=2.0e-13,
    )
    expected = {
        "liquid": {
            "composition": [0.96485875491431383, 0.035141245085686057],
            "log_phi": [-0.56849369396741345, 1.6644536891198392],
            "volume": 5.7713502290530615e-05,
            "dlogphi_dt": [0.017698089457743073, -0.023466918178234764],
            "dlogphi_dp": [-1.4431158481578368e-07, -8.285606271316724e-08],
            "dlogphi_dn": [
                [-0.006524555247235386, 0.17914203770716375],
                [0.1791420377071633, -4.918629463284182],
            ],
            "dv_dt": 9.484061022081443e-07,
            "dv_dp": -2.0345586534739677e-12,
            "dv_dn": [5.2629229915959413e-05, 1.9731028382210592e-04],
        },
        "vapor": {
            "composition": [0.84845070856684301, 0.1515492914331569],
            "log_phi": [-0.43992396185514993, 0.2029182753722194],
            "volume": 2.3673156051655026e-04,
            "dlogphi_dt": [0.006941074123498618, -0.006878280476208866],
            "dlogphi_dp": [-9.539949846306906e-08, 9.785930397858242e-08],
            "dlogphi_dn": [
                [-0.03304061489467508, 0.18497831862998571],
                [0.18497831862998571, -1.0356035585975625],
            ],
            "dv_dt": 4.142261221795269e-06,
            "dv_dp": -7.850440818937791e-11,
            "dv_dn": [1.677800245092073e-04, 6.227576450265246e-04],
        },
    }
    for phase, properties in zip(("liquid", "vapor"), result.phases, strict=True):
        reference = expected[phase]
        derivatives = state_derivatives(
            model,
            ChemicalState(state.temperature, state.pressure, properties.composition),
            phase,
        )
        torch.testing.assert_close(
            properties.composition,
            torch.tensor(reference["composition"], dtype=torch.float64),
            rtol=2.0e-12,
            atol=2.0e-13,
        )
        torch.testing.assert_close(
            properties.log_fugacity_coefficients,
            torch.tensor(reference["log_phi"], dtype=torch.float64),
            rtol=3.0e-12,
            atol=3.0e-13,
        )
        assert float(properties.molar_volume) == pytest.approx(reference["volume"], rel=3.0e-12)
        torch.testing.assert_close(
            derivatives.dlog_fugacity_coefficient_dtemperature,
            torch.tensor(reference["dlogphi_dt"], dtype=torch.float64),
            rtol=2.0e-11,
            atol=2.0e-13,
        )
        torch.testing.assert_close(
            derivatives.dlog_fugacity_coefficient_dpressure,
            torch.tensor(reference["dlogphi_dp"], dtype=torch.float64),
            rtol=3.0e-11,
            atol=2.0e-17,
        )
        torch.testing.assert_close(
            derivatives.dlog_fugacity_coefficient_dmoles,
            torch.tensor(reference["dlogphi_dn"], dtype=torch.float64),
            rtol=3.0e-11,
            atol=2.0e-12,
        )
        assert float(derivatives.dmolar_volume_dtemperature) == pytest.approx(
            reference["dv_dt"], rel=3.0e-11
        )
        assert float(derivatives.dmolar_volume_dpressure) == pytest.approx(
            reference["dv_dp"], rel=3.0e-11
        )
        # ThermoPack's dvdn is for total V on a one-mole basis:
        # dV/dn_j = v + dv_molar/dn_j.
        torch.testing.assert_close(
            properties.molar_volume + derivatives.dmolar_volume_dmoles,
            torch.tensor(reference["dv_dn"], dtype=torch.float64),
            rtol=3.0e-11,
            atol=2.0e-13,
        )


def test_translated_pr78_tp_tv_fugacity_and_free_energy_consistency():
    model = _thermopack_pr78_model()
    temperature = torch.tensor(283.15, dtype=torch.float64)
    pressure = torch.tensor(6.0e6, dtype=torch.float64)
    composition = torch.tensor(
        [0.96485875491431383, 0.035141245085686057],
        dtype=torch.float64,
    )
    state = ChemicalState(temperature, pressure, composition)
    properties = phase_properties(model, state, "liquid", caloric=False)
    torch.testing.assert_close(
        log_fugacities_tv(
            model,
            temperature,
            properties.molar_volume,
            composition,
        ),
        properties.log_fugacities,
        rtol=2.0e-13,
        atol=1.0e-13,
    )
    torch.testing.assert_close(
        fugacities_tv(
            model,
            temperature,
            properties.molar_volume,
            composition,
        ),
        properties.fugacities,
        rtol=2.0e-13,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        model.residual_helmholtz_rt(
            temperature,
            properties.molar_volume,
            composition,
        ),
        properties.reduced_residual_helmholtz_energy,
        rtol=2.0e-13,
        atol=1.0e-13,
    )
    torch.testing.assert_close(
        pressure,
        R * temperature * composition.sum() / properties.molar_volume
        - torch.func.grad(
            lambda volume: (
                R * temperature * model.residual_helmholtz_rt(temperature, volume, composition)
            )
        )(properties.molar_volume),
        rtol=2.0e-12,
        atol=2.0e-7,
    )


def test_binary_critical_point_matches_frozen_thermopack_reference():
    model = _thermopack_pr78_model()
    result = binary_critical_point(
        model,
        torch.tensor([0.9, 0.1], dtype=torch.float64),
        tolerance=1.0e-10,
    )
    assert result.converged
    assert float(result.temperature) == pytest.approx(296.73751605547142, rel=2.0e-8)
    assert float(result.pressure) == pytest.approx(8_769_014.866603531, rel=2.0e-7)
    assert float(result.molar_volume) == pytest.approx(
        9.825174252413657e-05,
        rel=4.0e-7,
    )


def test_batched_two_phase_flash_matches_scalar_grid():
    model = _thermopack_pr78_model()
    z = torch.tensor([0.9, 0.1], dtype=torch.float64)
    temperatures = torch.tensor([260.0, 270.0, 283.15, 290.0], dtype=torch.float64)
    envelope = phase_envelope(model, temperatures, z)
    dew = torch.stack(tuple(point.pressure for point in envelope["dew"]))
    bubble = torch.stack(tuple(point.pressure for point in envelope["bubble"]))
    pressures = torch.sqrt(dew * bubble)
    fraction = (torch.log(pressures) - torch.log(dew)) / (torch.log(bubble) - torch.log(dew))
    dew_k = torch.stack(tuple(point.k_values for point in envelope["dew"]))
    bubble_k = torch.stack(tuple(point.k_values for point in envelope["bubble"]))
    initial_log_k = (1.0 - fraction[:, None]) * torch.log(dew_k) + fraction[:, None] * torch.log(
        bubble_k
    )
    batched = batched_two_phase_flash(
        model,
        ChemicalState(temperatures, pressures, z),
        initial_k_values=torch.exp(initial_log_k),
        tolerance=2.0e-9,
    )
    assert bool(batched.converged.all())
    for index in range(temperatures.numel()):
        scalar = two_phase_flash(
            model,
            ChemicalState(temperatures[index], pressures[index], z),
            initial_k_values=batched.k_values[index],
            check_stability=False,
            tolerance=2.0e-10,
        )
        assert scalar.converged
        torch.testing.assert_close(
            batched.vapor_fraction[index],
            scalar.phase_fractions[1],
            rtol=2.0e-8,
            atol=2.0e-9,
        )
        torch.testing.assert_close(
            batched.liquid_composition[index],
            scalar.phases[0].composition,
            rtol=2.0e-8,
            atol=2.0e-9,
        )
        torch.testing.assert_close(
            batched.vapor_composition[index],
            scalar.phases[1].composition,
            rtol=2.0e-8,
            atol=2.0e-9,
        )


def test_log_k_continuation_reaches_retrograde_cricondentherm():
    model = _thermopack_pr78_model()
    z = torch.tensor([0.9, 0.1], dtype=torch.float64)
    branch = phase_envelope(
        model,
        torch.linspace(250.0, 296.4, 48, dtype=torch.float64),
        z,
        kinds=("bubble",),
    )["bubble"]
    start = branch[-1]
    targets = torch.linspace(
        torch.log(start.k_values[0]),
        0.03,
        60,
        dtype=torch.float64,
    )[1:]
    continuation = continue_saturation_branch(
        model,
        z,
        start,
        targets,
    )
    assert all(point.converged for point in continuation)
    temperatures = torch.stack(tuple(point.temperature for point in continuation))
    pressures = torch.stack(tuple(point.pressure for point in continuation))
    maximum = int(torch.argmax(temperatures))
    # ThermoPack's independently traced cricondentherm is approximately
    # 297.194 K and 85.47 bar for this exact parameter set.
    assert float(temperatures[maximum]) == pytest.approx(297.194, abs=0.015)
    assert float(pressures[maximum] / 1.0e5) == pytest.approx(85.47, abs=0.35)
    assert float(temperatures[-1]) < float(temperatures[maximum])


def test_accelerated_envelope_matches_legacy_continuation():
    model = _thermopack_pr78_model()
    z = torch.tensor([0.9, 0.1], dtype=torch.float64)
    temperatures = torch.linspace(250.0, 296.4, 16, dtype=torch.float64)
    accelerated = phase_envelope(
        model,
        temperatures,
        z,
        kinds=("bubble",),
    )["bubble"]
    legacy = phase_envelope(
        model,
        temperatures,
        z,
        kinds=("bubble",),
        accelerated=False,
    )["bubble"]
    assert all(point.converged for point in (*accelerated, *legacy))
    for predicted, reference in zip(accelerated, legacy, strict=True):
        torch.testing.assert_close(predicted.pressure, reference.pressure)
        torch.testing.assert_close(
            predicted.k_values,
            reference.k_values,
            rtol=1.0e-6,
            atol=1.0e-8,
        )
        torch.testing.assert_close(
            predicted.incipient_composition,
            reference.incipient_composition,
            rtol=1.0e-6,
            atol=1.0e-8,
        )

    start = accelerated[-1]
    targets = torch.linspace(
        torch.log(start.k_values[0]),
        0.03,
        20,
        dtype=torch.float64,
    )[1:]
    predicted_continuation = continue_saturation_branch(
        model,
        z,
        start,
        targets,
    )
    reference_continuation = continue_saturation_branch(
        model,
        z,
        start,
        targets,
        accelerated=False,
    )
    assert all(point.converged for point in (*predicted_continuation, *reference_continuation))
    for predicted, reference in zip(
        predicted_continuation,
        reference_continuation,
        strict=True,
    ):
        torch.testing.assert_close(predicted.temperature, reference.temperature)
        torch.testing.assert_close(
            predicted.pressure,
            reference.pressure,
            rtol=1.0e-6,
            atol=1.0,
        )
        torch.testing.assert_close(
            predicted.k_values,
            reference.k_values,
            rtol=1.0e-6,
            atol=1.0e-8,
        )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda model: log_fugacities_tv(
                model,
                torch.tensor([300.0]),
                torch.tensor(1.0),
                torch.tensor([0.5, 0.5]),
            ),
            "scalar temperature",
        ),
        (
            lambda model: log_fugacities_tv(
                model,
                torch.tensor(300.0),
                torch.tensor(1.0),
                torch.tensor([[0.5, 0.5]]),
            ),
            "mole-number vector",
        ),
        (
            lambda model: log_fugacities_tv(
                model,
                torch.tensor(300.0),
                torch.tensor(-1.0),
                torch.tensor([0.5, 0.5]),
            ),
            "must be positive",
        ),
        (
            lambda model: log_fugacities_tv(
                model,
                torch.tensor(300.0),
                torch.tensor(1.0),
                torch.tensor([0.5, 0.0]),
            ),
            "strictly positive",
        ),
    ],
)
def test_tv_fugacity_validation(call, message):
    with pytest.raises(ValueError, match=message):
        call(_thermopack_pr78_model())


def test_binary_critical_and_batched_flash_validation():
    model = _thermopack_pr78_model()
    with pytest.raises(ValueError, match="substitution_iterations"):
        saturation_point(
            model,
            torch.tensor(280.0, dtype=torch.float64),
            torch.tensor([0.9, 0.1], dtype=torch.float64),
            "bubble",
            substitution_iterations=-1,
        )
    with pytest.raises(ValueError, match="two components"):
        binary_critical_point(
            model,
            torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="interior"):
        binary_critical_point(
            model,
            torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="must be scalar"):
        binary_critical_point(
            model,
            torch.tensor([0.9, 0.1], dtype=torch.float64),
            initial_temperature=torch.tensor([300.0], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="must be positive"):
        binary_critical_point(
            model,
            torch.tensor([0.9, 0.1], dtype=torch.float64),
            initial_temperature=torch.tensor(-1.0, dtype=torch.float64),
        )

    state = ChemicalState(
        torch.tensor([270.0, 280.0], dtype=torch.float64),
        torch.tensor([5.0e6, 6.0e6], dtype=torch.float64),
        torch.tensor([0.9, 0.1], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="iteration counts"):
        batched_two_phase_flash(model, state, substitution_iterations=-1)
    with pytest.raises(ValueError, match="tolerance"):
        batched_two_phase_flash(model, state, tolerance=0.0)
    with pytest.raises(ValueError, match="one row"):
        batched_two_phase_flash(
            model,
            state,
            initial_k_values=torch.ones((1, 2), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="Kmin"):
        batched_two_phase_flash(
            model,
            state,
            initial_k_values=torch.full((2, 2), 2.0, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="finite and strictly positive"):
        batched_two_phase_flash(
            model,
            state,
            initial_k_values=torch.tensor(
                [[2.0, 0.5], [float("nan"), 0.5]],
                dtype=torch.float64,
            ),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        batched_two_phase_flash(
            model,
            ChemicalState(
                torch.tensor(280.0, dtype=torch.float64),
                torch.tensor(6.0e6, dtype=torch.float64),
                torch.tensor([0.9, 0.1], dtype=torch.float64),
            ),
        )
    with pytest.raises(ValueError, match="same shape"):
        batched_two_phase_flash(
            model,
            ChemicalState(
                torch.tensor([270.0, 280.0], dtype=torch.float64),
                torch.tensor([6.0e6], dtype=torch.float64),
                torch.tensor([0.9, 0.1], dtype=torch.float64),
            ),
        )
    with pytest.raises(ValueError, match="one vector per"):
        batched_two_phase_flash(
            model,
            ChemicalState(
                torch.tensor([270.0, 280.0], dtype=torch.float64),
                torch.tensor([5.0e6, 6.0e6], dtype=torch.float64),
                torch.tensor(
                    [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
                    dtype=torch.float64,
                ),
            ),
        )

    start = phase_envelope(
        model,
        torch.tensor([280.0], dtype=torch.float64),
        torch.tensor([0.9, 0.1], dtype=torch.float64),
        kinds=("bubble",),
    )["bubble"][0]
    with pytest.raises(ValueError, match="one composition vector"):
        continue_saturation_branch(
            model,
            torch.tensor([[0.9, 0.1]], dtype=torch.float64),
            start,
            torch.tensor([0.0], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        continue_saturation_branch(
            model,
            torch.tensor([0.9, 0.1], dtype=torch.float64),
            start,
            torch.tensor([[0.0]], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="match composition"):
        continue_saturation_branch(
            model,
            torch.tensor([0.9, 0.1], dtype=torch.float64),
            replace(
                start,
                k_values=torch.tensor([2.0, 1.0, 0.5], dtype=torch.float64),
            ),
            torch.tensor([0.0], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="outside"):
        continue_saturation_branch(
            model,
            torch.tensor([0.9, 0.1], dtype=torch.float64),
            start,
            torch.tensor([0.0], dtype=torch.float64),
            controlled_component=2,
        )
    with pytest.raises(ValueError, match="bubble or dew"):
        continue_saturation_branch(
            model,
            torch.tensor([0.9, 0.1], dtype=torch.float64),
            replace(start, kind="solid"),  # type: ignore[arg-type]
            torch.tensor([0.0], dtype=torch.float64),
        )


def test_batched_flash_secondary_paths_and_dew_continuation(monkeypatch):
    model = _thermopack_pr78_model()
    z = torch.tensor([0.9, 0.1], dtype=torch.float64)
    temperatures = torch.tensor([275.0, 283.15], dtype=torch.float64)
    pressures = torch.tensor([5.0e6, 6.0e6], dtype=torch.float64)
    state = ChemicalState(
        temperatures,
        pressures,
        z.expand(2, -1).clone(),
    )

    # Wilson initialization exercises the no-explicit-K path and the
    # per-state composition-matrix path.
    wilson = batched_two_phase_flash(
        model,
        state,
        tolerance=1.0e-7,
    )
    assert wilson.k_values.shape == (2, 2)

    # A polished K matrix covers the substitution and Newton early-exit paths.
    polished = batched_two_phase_flash(
        model,
        state,
        initial_k_values=wilson.k_values,
        tolerance=1.0e-6,
        substitution_iterations=2,
        newton_iterations=0,
    )
    assert bool(polished.converged.all())
    immediate_newton = batched_two_phase_flash(
        model,
        state,
        initial_k_values=polished.k_values,
        phase_roots=("stable", "stable"),
        tolerance=1.0e-6,
        substitution_iterations=0,
        newton_iterations=2,
    )
    assert bool(immediate_newton.converged.all())
    with pytest.raises(ValueError, match="phase_roots"):
        batched_two_phase_flash(
            model,
            state,
            initial_k_values=polished.k_values,
            phase_roots=("liquid", "invalid"),  # type: ignore[arg-type]
        )

    original_solve = torch.linalg.solve

    def singular_solve(*args, **kwargs):
        raise torch.linalg.LinAlgError("forced singular matrix")

    monkeypatch.setattr(torch.linalg, "solve", singular_solve)
    fallback = batched_two_phase_flash(
        model,
        state,
        initial_k_values=torch.tensor(
            [[0.7, 4.0], [0.7, 4.0]],
            dtype=torch.float64,
        ),
        substitution_iterations=0,
        newton_iterations=1,
    )
    assert fallback.residual_norm.shape == (2,)
    monkeypatch.setattr(torch.linalg, "solve", original_solve)

    current = torch.tensor([[-1.0, 1.0]], dtype=torch.float64)
    exhausted = _admissible_update(
        current,
        torch.full_like(current, 1.0e12),
    )
    torch.testing.assert_close(exhausted, current)
    assert bool(((exhausted.amin(-1) < 0.0) & (exhausted.amax(-1) > 0.0)).all())

    dew_start = phase_envelope(
        model,
        torch.tensor([280.0], dtype=torch.float64),
        z,
        kinds=("dew",),
    )["dew"][0]
    dew_target = torch.log(dew_start.k_values[0]).reshape(1)
    dew_continuation = continue_saturation_branch(
        model,
        z,
        dew_start,
        dew_target,
    )
    assert dew_continuation[0].converged

    explicit_critical = binary_critical_point(
        model,
        z,
        initial_temperature=torch.tensor(296.7, dtype=torch.float64),
        initial_pressure=torch.tensor(8.77e6, dtype=torch.float64),
    )
    assert explicit_critical.converged


class _NoCriticalConstants:
    def log_fugacity_coefficients(
        self,
        temperature,
        pressure,
        composition,
        phase="stable",
    ):
        return torch.zeros_like(composition)


def test_batched_flash_without_constants_requires_explicit_k():
    state = ChemicalState(
        torch.tensor([280.0], dtype=torch.float64),
        torch.tensor([5.0e6], dtype=torch.float64),
        torch.tensor([0.9, 0.1], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="without critical constants"):
        batched_two_phase_flash(_NoCriticalConstants(), state)
