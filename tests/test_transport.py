"""Verification and validation tests for public transport-property APIs."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import torch_flash.transport.heavy_oil as heavy_oil_module
import torch_flash.transport.thermal_conductivity as thermal_module
import torch_flash.transport.viscosity as viscosity_module
from torch_flash import (
    ComponentSet,
    available_parameter_sets,
    brock_bird_surface_tension,
    component_set,
    corresponding_states_thermal_conductivity,
    corresponding_states_viscosity,
    evaluate_heavy_oil_corresponding_states_profile,
    fit_heavy_oil_csp_factors,
    friction_theory_viscosity,
    hayduk_minhas_n_paraffin_diffusion_coefficient,
    heavy_oil_corresponding_states_viscosity,
    kinematic_viscosity,
    lbc_pseudocomponent_critical_volume,
    lbc_viscosity,
    lee_chien_interfacial_tension,
    lee_gas_viscosity,
    load_model_parameters,
    methane_critical_thermal_conductivity_enhancement,
    methane_thermal_conductivity,
    parachor_from_molar_mass,
    peng_robinson_1976,
    peng_robinson_1978,
    poling_ideal_gas,
    published_lee_chien_b,
    published_parachors,
    riedel_parameter,
    soave_redlich_kwong,
    stabilized_heavy_oil_viscosity,
    weinaug_katz_interfacial_tension,
)
from torch_flash.exceptions import ConvergenceError, InvalidStateError
from torch_flash.transport.viscosity import (
    methane_bwr_density,
    methane_bwr_pressure,
    methane_viscosity,
)

DTYPE = torch.float64


def test_transport_parameter_document_is_registered():
    assert "transport.pedersen-2024" in available_parameter_sets(model_kind="transport")
    parameters = load_model_parameters("transport.pedersen-2024")
    assert parameters.model_kind == "transport"
    assert parameters.parameters["diffusion"]["temperature_exponent"] == pytest.approx(1.47)
    assert parameters.parameters["diffusion"]["volume_exponent"] == pytest.approx(0.71)
    assert parameters.references[1]["doi"] == "10.1016/0011-2275(75)90010-7"


def test_transport_structured_numerical_result_banks(num_regression):
    hexane = component_set(("n_hexane",), dtype=DTYPE)
    temperature = torch.tensor([350.0, 400.0, 450.0], dtype=DTYPE)
    pressure = torch.tensor([1.0e6, 1.0e7, 2.0e7], dtype=DTYPE)
    composition = torch.ones((3, 1), dtype=DTYPE)
    pr = peng_robinson_1976(hexane)
    friction = friction_theory_viscosity(
        temperature,
        pressure,
        composition,
        pr,
        phase="liquid",
    )
    corresponding_states = corresponding_states_viscosity(
        temperature,
        pressure,
        composition,
        hexane,
        phase="liquid",
    )
    lbc = lbc_viscosity(
        temperature,
        torch.tensor([4000.0, 5000.0, 6000.0], dtype=DTYPE),
        composition,
        hexane,
    )
    lee = lee_gas_viscosity(
        temperature,
        torch.tensor([10.0, 100.0, 300.0], dtype=DTYPE),
        torch.full((3,), hexane.molar_mass[0], dtype=DTYPE),
    )
    num_regression.check(
        {
            "friction_theory_Pa_s": friction.detach().numpy(),
            "corresponding_states_Pa_s": corresponding_states.detach().numpy(),
            "lbc_Pa_s": lbc.detach().numpy(),
            "lee_gas_Pa_s": lee.detach().numpy(),
        },
        basename="transport_viscosity_outputs",
        default_tolerance={"rtol": 1.0e-5, "atol": 0.0},
    )

    methane_temperature = torch.tensor([150.0, 190.0, 250.0, 300.0], dtype=DTYPE)
    methane_density = torch.tensor([20.0, 10.0, 2.0, 0.0], dtype=DTYPE)
    background = methane_thermal_conductivity(
        methane_temperature,
        methane_density,
        include_critical_enhancement=False,
    )
    complete = methane_thermal_conductivity(
        methane_temperature,
        methane_density,
    )
    num_regression.check(
        {
            "temperature_K": methane_temperature.numpy(),
            "density_mol_L": methane_density.numpy(),
            "background_W_m_K": background.detach().numpy(),
            "complete_W_m_K": complete.detach().numpy(),
        },
        basename="transport_methane_conductivity_outputs",
        default_tolerance={"rtol": 1.0e-5, "atol": 0.0},
    )

    liquid = torch.tensor(
        [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]],
        dtype=DTYPE,
    )
    vapor = torch.tensor(
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
        dtype=DTYPE,
    )
    phase_components = component_set(("methane", "n_butane"), dtype=DTYPE)
    parachor = published_parachors(phase_components.names, like=liquid)
    riedel = riedel_parameter(
        torch.tensor([111.66, 272.65], dtype=DTYPE),
        phase_components.critical_temperature,
        phase_components.critical_pressure,
    )
    b = published_lee_chien_b(phase_components.names, like=liquid)
    liquid_density = torch.tensor([8000.0, 7500.0, 7000.0], dtype=DTYPE)
    vapor_density = torch.tensor([500.0, 800.0, 1000.0], dtype=DTYPE)
    num_regression.check(
        {
            "weinaug_katz_N_m": weinaug_katz_interfacial_tension(
                liquid_density,
                vapor_density,
                liquid,
                vapor,
                parachor,
            )
            .detach()
            .numpy(),
            "lee_chien_N_m": lee_chien_interfacial_tension(
                liquid_density,
                vapor_density,
                liquid,
                vapor,
                phase_components,
                riedel,
                b,
            )
            .detach()
            .numpy(),
            "diffusion_m2_s": hayduk_minhas_n_paraffin_diffusion_coefficient(
                torch.tensor([280.0, 320.0, 360.0], dtype=DTYPE),
                torch.tensor([2.0e-3, 1.0e-3, 5.0e-4], dtype=DTYPE),
                torch.tensor([2.0e-4, 3.0e-4, 4.0e-4], dtype=DTYPE),
            )
            .detach()
            .numpy(),
        },
        basename="transport_interfacial_diffusion_outputs",
        default_tolerance={"rtol": 1.0e-5, "atol": 0.0},
    )


def test_kinematic_lee_gas_and_diffusion_si_values_and_autodiff():
    viscosity = torch.tensor([1.0e-3, 2.0e-3], dtype=DTYPE)
    density = torch.tensor([1000.0, 800.0], dtype=DTYPE)
    torch.testing.assert_close(
        kinematic_viscosity(viscosity, density),
        torch.tensor([1.0e-6, 2.5e-6], dtype=DTYPE),
        rtol=0.0,
        atol=0.0,
    )

    temperature = torch.tensor([300.0, 373.15], dtype=DTYPE, requires_grad=True)
    gas = lee_gas_viscosity(
        temperature,
        torch.tensor([0.0, 100.0], dtype=DTYPE),
        torch.tensor([0.018, 0.020], dtype=DTYPE),
    )
    torch.testing.assert_close(
        gas,
        torch.tensor(
            [1.1225758805417914e-5, 1.690271635191726e-5],
            dtype=DTYPE,
        ),
        rtol=2.0e-14,
        atol=0.0,
    )

    diffusion = hayduk_minhas_n_paraffin_diffusion_coefficient(
        temperature,
        torch.tensor([1.0e-3, 2.0e-3], dtype=DTYPE),
        torch.tensor([1.2e-4, 1.5e-4], dtype=DTYPE),
    )
    torch.testing.assert_close(
        diffusion,
        torch.tensor(
            [1.9453654600442582e-9, 1.3862795788270774e-9],
            dtype=DTYPE,
        ),
        rtol=2.0e-14,
        atol=0.0,
    )
    gradient = torch.autograd.grad((gas + diffusion).sum(), temperature)[0]
    assert torch.isfinite(gradient).all()
    assert gas.dtype == DTYPE


@pytest.mark.parametrize(
    ("operation", "arguments", "match"),
    [
        (
            kinematic_viscosity,
            (torch.tensor(-1.0), torch.tensor(1000.0)),
            "non-negative",
        ),
        (
            lee_gas_viscosity,
            (torch.tensor(300.0), torch.tensor(-1.0), torch.tensor(0.02)),
            "gas density",
        ),
        (
            hayduk_minhas_n_paraffin_diffusion_coefficient,
            (torch.tensor(300.0), torch.tensor(0.0), torch.tensor(1.0e-3)),
            "positive finite",
        ),
    ],
)
def test_simple_transport_models_reject_nonphysical_inputs(operation, arguments, match):
    with pytest.raises(InvalidStateError, match=match):
        operation(*arguments)


def test_lee_gas_rejects_overflowed_result():
    with pytest.raises(InvalidStateError, match="non-positive viscosity"):
        lee_gas_viscosity(
            torch.tensor(300.0, dtype=DTYPE),
            torch.tensor(1.0e100, dtype=DTYPE),
            torch.tensor(0.02, dtype=DTYPE),
        )


def test_stabilized_heavy_oil_branches_values_and_parameter_gradients():
    temperature = torch.tensor([350.0, 600.0], dtype=DTYPE)
    pressure = torch.tensor([101_325.0, 1.0e7], dtype=DTYPE)
    third = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    fourth = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    viscosity = stabilized_heavy_oil_viscosity(
        temperature,
        pressure,
        torch.tensor([0.30, 0.30], dtype=DTYPE),
        torch.tensor([0.36, 0.60], dtype=DTYPE),
        third_csp=third,
        fourth_csp=fourth,
    )
    torch.testing.assert_close(
        viscosity,
        torch.tensor(
            [0.012782700971863578, 23353.684177136885],
            dtype=DTYPE,
        ),
        rtol=3.0e-14,
        atol=0.0,
    )
    gradients = torch.autograd.grad(viscosity.sum(), (third, fourth))
    assert all(torch.isfinite(gradient) for gradient in gradients)


def test_stabilized_heavy_oil_validation_and_overflow():
    with pytest.raises(InvalidStateError, match="must be physical"):
        stabilized_heavy_oil_viscosity(
            torch.tensor(300.0),
            torch.tensor(1.0e5),
            torch.tensor(0.4),
            torch.tensor(0.3),
        )
    with pytest.raises(InvalidStateError, match="non-positive viscosity"):
        stabilized_heavy_oil_viscosity(
            torch.tensor(600.0, dtype=DTYPE),
            torch.tensor(1.0e300, dtype=DTYPE),
            torch.tensor(0.3, dtype=DTYPE),
            torch.tensor(0.6, dtype=DTYPE),
        )


def test_heavy_oil_blending_is_batched_and_differentiable():
    components = component_set(("n_decane",), dtype=DTYPE)
    temperature = torch.tensor([190.0, 300.0, 500.0], dtype=DTYPE)
    pressure = torch.full((3,), 1.0e6, dtype=DTYPE)
    composition = torch.ones((3, 1), dtype=DTYPE)
    third = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    result = heavy_oil_corresponding_states_viscosity(
        temperature,
        pressure,
        composition,
        components,
        third_csp=third,
    )
    assert result.shape == (3,)
    assert torch.isfinite(result).all()
    assert (result > 0.0).all()
    gradient = torch.autograd.grad(result.sum(), third)[0]
    assert torch.isfinite(gradient)
    assert gradient != 0.0
    with pytest.raises(ValueError, match="component set sizes"):
        heavy_oil_corresponding_states_viscosity(
            temperature,
            pressure,
            torch.ones((3, 2), dtype=DTYPE),
            components,
        )


def test_phase_aware_heavy_oil_profile_and_joint_factor_fit(num_regression):
    methane = component_set(("methane",), dtype=DTYPE)
    components = ComponentSet(
        (*methane.names, "heavy_cut"),
        torch.cat((methane.critical_temperature, torch.tensor([1000.0], dtype=DTYPE))),
        torch.cat((methane.critical_pressure, torch.tensor([1.2e6], dtype=DTYPE))),
        torch.cat((methane.acentric_factor, torch.tensor([0.8], dtype=DTYPE))),
        torch.cat((methane.molar_mass, torch.tensor([0.8], dtype=DTYPE))),
    )
    model = peng_robinson_1976(components)
    temperature = torch.tensor([310.0, 310.0, 330.0, 330.0], dtype=DTYPE)
    pressure = torch.tensor([1.0e6, 8.0e6, 1.0e6, 8.0e6], dtype=DTYPE)
    feed = torch.tensor([0.2, 0.8], dtype=DTYPE)
    profile = evaluate_heavy_oil_corresponding_states_profile(
        model,
        temperature,
        pressure,
        feed,
        components,
    )
    assert bool(profile.converged.all())
    assert float(profile.residual_norm.max()) < 1.0e-10
    assert bool((profile.vapor_fraction[[0, 2]] > 0.0).all())
    assert bool((profile.vapor_fraction[[1, 3]] == 0.0).all())

    observations = heavy_oil_corresponding_states_viscosity(
        temperature,
        pressure,
        profile.liquid_composition,
        components,
        third_csp=1.3,
        fourth_csp=1.8,
    )
    calibration = fit_heavy_oil_csp_factors(
        temperature,
        pressure,
        profile.liquid_composition,
        components,
        observations,
        tolerance=1.0e-8,
    )
    assert float(calibration.third_csp) == pytest.approx(1.3, rel=2.0e-3)
    assert float(calibration.fourth_csp) == pytest.approx(1.8, rel=2.0e-3)
    assert calibration.fit.iterations <= 200
    assert calibration.sensitivity_rank == 2
    torch.testing.assert_close(calibration.prediction, observations, rtol=3.0e-4, atol=0.0)
    num_regression.check(
        {
            "bubble_pressure_Pa": profile.bubble_pressure.detach().numpy(),
            "vapor_fraction": profile.vapor_fraction.detach().numpy(),
            "predictive_viscosity_Pa_s": profile.viscosity.detach().numpy(),
            "calibrated_viscosity_Pa_s": calibration.prediction.detach().numpy(),
            "fitted_factors": torch.stack((calibration.third_csp, calibration.fourth_csp)).numpy(),
            "sensitivity_singular_values": calibration.sensitivity_singular_values.detach().numpy(),
        },
        basename="transport_phase_aware_heavy_oil_profile",
        default_tolerance={"rtol": 1.0e-5, "atol": 0.0},
    )


def test_phase_aware_heavy_oil_profile_and_fit_validate_inputs():
    components = component_set(("methane",), dtype=DTYPE)
    model = peng_robinson_1976(components)
    temperature = torch.tensor([300.0], dtype=DTYPE)
    pressure = torch.tensor([1.0e6], dtype=DTYPE)
    composition = torch.ones((1, 1), dtype=DTYPE)
    with pytest.raises(ValueError, match="one-dimensional"):
        evaluate_heavy_oil_corresponding_states_profile(
            model,
            temperature[0],
            pressure,
            composition[0],
            components,
        )
    with pytest.raises(InvalidStateError, match="positive"):
        evaluate_heavy_oil_corresponding_states_profile(
            model,
            temperature,
            -pressure,
            composition[0],
            components,
        )
    with pytest.raises(ValueError, match="feed composition"):
        evaluate_heavy_oil_corresponding_states_profile(
            model,
            temperature,
            pressure,
            torch.ones(2, dtype=DTYPE),
            components,
        )
    ethane_model = peng_robinson_1976(component_set(("ethane",), dtype=DTYPE))
    with pytest.raises(ValueError, match="component order"):
        evaluate_heavy_oil_corresponding_states_profile(
            ethane_model,
            temperature,
            pressure,
            composition[0],
            components,
        )
    with pytest.raises(ValueError, match="iteration controls"):
        evaluate_heavy_oil_corresponding_states_profile(
            model,
            temperature,
            pressure,
            composition[0],
            components,
            tolerance=0.0,
        )
    with pytest.raises(InvalidStateError, match="feed fractions"):
        evaluate_heavy_oil_corresponding_states_profile(
            model,
            temperature,
            pressure,
            torch.zeros(1, dtype=DTYPE),
            components,
        )
    with pytest.raises(ValueError, match="initial bubble pressure"):
        evaluate_heavy_oil_corresponding_states_profile(
            model,
            temperature,
            pressure,
            composition[0],
            components,
            initial_bubble_pressure=-1.0,
        )
    with pytest.raises(ValueError, match="initial bubble pressure"):
        evaluate_heavy_oil_corresponding_states_profile(
            model,
            temperature,
            pressure,
            composition[0],
            components,
            initial_bubble_pressure=torch.ones(2, dtype=DTYPE),
        )
    with pytest.raises(ValueError, match="shapes differ"):
        fit_heavy_oil_csp_factors(
            temperature,
            pressure,
            composition,
            components,
            torch.ones(2, dtype=DTYPE),
        )
    with pytest.raises(InvalidStateError, match="observed viscosities"):
        fit_heavy_oil_csp_factors(
            temperature,
            pressure,
            composition,
            components,
            torch.zeros(1, dtype=DTYPE),
        )
    with pytest.raises(ValueError, match="strictly inside"):
        fit_heavy_oil_csp_factors(
            temperature,
            pressure,
            composition,
            components,
            torch.ones(1, dtype=DTYPE),
            initial_factors=(0.1, 1.0),
        )


def test_phase_aware_heavy_oil_profile_boundary_failure_and_no_flash(monkeypatch):
    components = component_set(("methane",), dtype=DTYPE)
    model = peng_robinson_1976(components)
    temperature = torch.tensor([300.0], dtype=DTYPE)
    pressure = torch.tensor([3.0e6], dtype=DTYPE)
    feed = torch.ones(1, dtype=DTYPE)

    def boundary(converged):
        return SimpleNamespace(
            pressure=torch.tensor(2.0e6, dtype=DTYPE),
            converged=converged,
            residual_norm=torch.tensor(2.0e-4, dtype=DTYPE),
            k_values=torch.ones(1, dtype=DTYPE),
        )

    monkeypatch.setattr(
        heavy_oil_module,
        "saturation_point",
        lambda *args, **kwargs: boundary(True),
    )
    homogeneous = evaluate_heavy_oil_corresponding_states_profile(
        model,
        temperature,
        pressure,
        feed,
        components,
    )
    assert bool(homogeneous.converged.all())
    assert homogeneous.vapor_fraction.item() == 0.0
    torch.testing.assert_close(homogeneous.liquid_composition, feed.reshape(1, 1))

    monkeypatch.setattr(
        heavy_oil_module,
        "saturation_point",
        lambda *args, **kwargs: boundary(False),
    )
    failed = evaluate_heavy_oil_corresponding_states_profile(
        model,
        temperature,
        pressure,
        feed,
        components,
        raise_on_failure=False,
    )
    assert not bool(failed.converged.any())
    assert torch.isnan(failed.viscosity).all()
    torch.testing.assert_close(failed.residual_norm, torch.tensor([2.0e-4], dtype=DTYPE))
    with pytest.raises(ConvergenceError, match="state indices"):
        evaluate_heavy_oil_corresponding_states_profile(
            model,
            temperature,
            pressure,
            feed,
            components,
        )


def test_lbc_batch_matches_independent_scalar_calls():
    components = component_set(("methane", "carbon_dioxide"), dtype=DTYPE)
    temperature = torch.tensor([240.0, 300.0], dtype=DTYPE)
    density = torch.tensor([1000.0, 5000.0], dtype=DTYPE)
    composition = torch.tensor([[0.8, 0.2], [0.4, 0.6]], dtype=DTYPE)
    batched = lbc_viscosity(temperature, density, composition, components)
    scalar = torch.stack(
        [
            lbc_viscosity(temperature[index], density[index], composition[index], components)
            for index in range(2)
        ]
    )
    torch.testing.assert_close(batched, scalar, rtol=2.0e-15, atol=0.0)


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (
            soave_redlich_kwong,
            [1.9705717124290543e-4, 1.5302749238149430e-4, 1.2842530285274306e-4],
        ),
        (
            peng_robinson_1976,
            [1.9490625802224252e-4, 1.5218919371664494e-4, 1.2849885260946106e-4],
        ),
    ],
)
def test_one_parameter_friction_theory_published_families(factory, expected):
    eos = factory(component_set(("n_hexane",), dtype=DTYPE))
    temperature = torch.tensor([350.0, 400.0, 450.0], dtype=DTYPE, requires_grad=True)
    viscosity = friction_theory_viscosity(
        temperature,
        torch.tensor([1.0e6, 1.0e7, 2.0e7], dtype=DTYPE),
        torch.ones((3, 1), dtype=DTYPE),
        eos,
        phase="liquid",
    )
    torch.testing.assert_close(
        viscosity,
        torch.tensor(expected, dtype=DTYPE),
        rtol=3.0e-14,
        atol=0.0,
    )
    gradient = torch.autograd.grad(viscosity.sum(), temperature)[0]
    assert torch.isfinite(gradient).all()


def test_friction_theory_custom_critical_viscosity_preserves_gradient():
    eos = soave_redlich_kwong(component_set(("n_hexane",), dtype=DTYPE))
    critical_viscosity = torch.tensor([2.0e-5], dtype=DTYPE, requires_grad=True)
    value = friction_theory_viscosity(
        torch.tensor(400.0, dtype=DTYPE),
        torch.tensor(1.0e7, dtype=DTYPE),
        torch.tensor([1.0], dtype=DTYPE),
        eos,
        phase="liquid",
        critical_viscosity=critical_viscosity,
        critical_volume=eos.critical_volume,
    )
    gradient = torch.autograd.grad(value, critical_viscosity)[0]
    assert torch.isfinite(value)
    assert torch.isfinite(gradient).all()


def test_friction_theory_validation_paths(monkeypatch):
    components = component_set(("n_hexane",), dtype=DTYPE)
    srk = soave_redlich_kwong(components)
    with pytest.raises(ValueError, match="SRK and PR76"):
        friction_theory_viscosity(
            torch.tensor(400.0, dtype=DTYPE),
            torch.tensor(1.0e7, dtype=DTYPE),
            torch.tensor([1.0], dtype=DTYPE),
            peng_robinson_1978(components),
        )
    with pytest.raises(ValueError, match="component sizes"):
        friction_theory_viscosity(
            torch.tensor(400.0, dtype=DTYPE),
            torch.tensor(1.0e7, dtype=DTYPE),
            torch.tensor([0.5, 0.5], dtype=DTYPE),
            srk,
        )
    with pytest.raises(InvalidStateError, match="finite and positive"):
        friction_theory_viscosity(
            torch.tensor(400.0, dtype=DTYPE),
            torch.tensor(-1.0, dtype=DTYPE),
            torch.tensor([1.0], dtype=DTYPE),
            srk,
        )
    with pytest.raises(ValueError, match="one critical volume"):
        friction_theory_viscosity(
            torch.tensor(400.0, dtype=DTYPE),
            torch.tensor(1.0e7, dtype=DTYPE),
            torch.tensor([1.0], dtype=DTYPE),
            srk,
            critical_volume=torch.ones(2, dtype=DTYPE),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        friction_theory_viscosity(
            torch.tensor(400.0, dtype=DTYPE),
            torch.tensor(1.0e7, dtype=DTYPE),
            torch.tensor([1.0], dtype=DTYPE),
            srk,
            critical_volume=torch.tensor([float("nan")], dtype=DTYPE),
        )
    with pytest.raises(ValueError, match="one value per component"):
        friction_theory_viscosity(
            torch.tensor(400.0, dtype=DTYPE),
            torch.tensor(1.0e7, dtype=DTYPE),
            torch.tensor([1.0], dtype=DTYPE),
            srk,
            critical_viscosity=torch.ones(2, dtype=DTYPE),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        friction_theory_viscosity(
            torch.tensor(400.0, dtype=DTYPE),
            torch.tensor(1.0e7, dtype=DTYPE),
            torch.tensor([1.0], dtype=DTYPE),
            srk,
            critical_viscosity=torch.tensor([0.0], dtype=DTYPE),
        )

    no_volume_components = ComponentSet(
        components.names,
        components.critical_temperature,
        components.critical_pressure,
        components.acentric_factor,
        components.molar_mass,
    )
    with pytest.raises(ValueError, match="one critical volume"):
        friction_theory_viscosity(
            torch.tensor(400.0, dtype=DTYPE),
            torch.tensor(1.0e7, dtype=DTYPE),
            torch.tensor([1.0], dtype=DTYPE),
            soave_redlich_kwong(no_volume_components),
        )

    monkeypatch.setattr(
        viscosity_module,
        "_chung_dilute_viscosity",
        lambda temperature, eos, volume: (
            -torch.ones(
                (*temperature.shape, eos.ncomponents),
                dtype=temperature.dtype,
                device=temperature.device,
            )
        ),
    )
    with pytest.raises(InvalidStateError, match="non-positive viscosity"):
        friction_theory_viscosity(
            torch.tensor(400.0, dtype=DTYPE),
            torch.tensor(1.0e7, dtype=DTYPE),
            torch.tensor([1.0], dtype=DTYPE),
            srk,
        )


def test_methane_thermal_conductivity_primary_coefficient_and_critical_term():
    temperature = torch.tensor([80.0, 300.0], dtype=DTYPE)
    density = torch.tensor([20.0, 0.0], dtype=DTYPE)
    background = methane_thermal_conductivity(
        temperature,
        density,
        include_critical_enhancement=False,
    )
    assert torch.isfinite(background).all()
    assert background.dtype == DTYPE
    assert float(background[1]) == pytest.approx(0.034689404044635011, rel=2.0e-14)

    near_critical_temperature = torch.tensor(190.0, dtype=DTYPE, requires_grad=True)
    enhancement = methane_critical_thermal_conductivity_enhancement(
        near_critical_temperature,
        torch.tensor(10.0, dtype=DTYPE),
    )
    assert float(enhancement.detach()) == pytest.approx(0.45976107131980365, rel=3.0e-13)
    gradient = torch.autograd.grad(enhancement, near_critical_temperature)[0]
    assert torch.isfinite(gradient)

    mixed_density = methane_thermal_conductivity(
        torch.tensor([300.0, 300.0], dtype=DTYPE),
        torch.tensor([0.0, 1.0], dtype=DTYPE),
    )
    assert mixed_density[1] > mixed_density[0]
    torch.testing.assert_close(mixed_density[0], background[1], rtol=0.0, atol=0.0)


def test_methane_thermal_conductivity_validation_paths():
    with pytest.raises(InvalidStateError, match="positive T"):
        methane_thermal_conductivity(torch.tensor(0.0), torch.tensor(1.0))
    with pytest.raises(InvalidStateError, match="positive finite state"):
        methane_critical_thermal_conductivity_enhancement(
            torch.tensor(190.0),
            torch.tensor(0.0),
        )
    with pytest.raises(InvalidStateError, match="non-positive isothermal"):
        methane_critical_thermal_conductivity_enhancement(
            torch.tensor(150.0, dtype=DTYPE),
            torch.tensor(5.0, dtype=DTYPE),
        )


def test_co2_methane_thermal_conductivity_against_pedersen_table_10_17():
    temperature = torch.tensor(
        [267.12, 246.77, 266.93, 228.32, 246.75, 253.94],
        dtype=DTYPE,
        requires_grad=True,
    )
    pressure = 1.0e5 * torch.tensor(
        [17.91, 11.12, 12.14, 2.64, 2.86, 2.95],
        dtype=DTYPE,
    )
    composition = torch.tensor([0.4939, 0.5061], dtype=DTYPE).expand(6, -1)
    components = component_set(("carbon_dioxide", "methane"), dtype=DTYPE)
    conductivity = corresponding_states_thermal_conductivity(
        temperature,
        pressure,
        composition,
        components,
        poling_ideal_gas(["carbon_dioxide", "methane"], dtype=DTYPE),
        poling_ideal_gas(["methane"], dtype=DTYPE),
    )
    torch.testing.assert_close(
        1000.0 * conductivity.detach(),
        torch.tensor(
            [
                24.043617174795013,
                21.71496495501753,
                23.326168796568908,
                19.04220978244893,
                20.507558764459755,
                21.08221214621377,
            ],
            dtype=DTYPE,
        ),
        rtol=4.0e-9,
        atol=2.0e-8,
    )
    measured = torch.tensor(
        [24.45, 21.77, 23.39, 19.14, 20.65, 21.68],
        dtype=DTYPE,
    )
    mean_absolute_percentage_error = torch.mean(
        torch.abs(1000.0 * conductivity.detach() - measured) / measured
    )
    assert float(mean_absolute_percentage_error) < 0.012
    gradient = torch.autograd.grad(conductivity.sum(), temperature)[0]
    assert torch.isfinite(gradient).all()


def test_mixture_thermal_conductivity_validation_paths(monkeypatch):
    components = component_set(("carbon_dioxide", "methane"), dtype=DTYPE)
    composition = torch.tensor([0.4939, 0.5061], dtype=DTYPE)
    ideal = poling_ideal_gas(["carbon_dioxide", "methane"], dtype=DTYPE)
    methane_ideal = poling_ideal_gas(["methane"], dtype=DTYPE)
    with pytest.raises(ValueError, match="component set sizes"):
        corresponding_states_thermal_conductivity(
            torch.tensor(250.0, dtype=DTYPE),
            torch.tensor(1.0e5, dtype=DTYPE),
            torch.tensor([1.0], dtype=DTYPE),
            components,
            ideal,
            methane_ideal,
        )

    bad_ideal = Mock()
    bad_ideal.heat_capacity.return_value = torch.tensor([30.0], dtype=DTYPE)
    with pytest.raises(ValueError, match="one value per component"):
        corresponding_states_thermal_conductivity(
            torch.tensor(250.0, dtype=DTYPE),
            torch.tensor(1.0e5, dtype=DTYPE),
            composition,
            components,
            bad_ideal,
            methane_ideal,
        )

    bad_methane = Mock()
    bad_methane.heat_capacity.return_value = torch.tensor([30.0, 31.0], dtype=DTYPE)
    with pytest.raises(ValueError, match="one component value"):
        corresponding_states_thermal_conductivity(
            torch.tensor(250.0, dtype=DTYPE),
            torch.tensor(1.0e5, dtype=DTYPE),
            composition,
            components,
            ideal,
            bad_methane,
        )

    monkeypatch.setattr(
        thermal_module,
        "methane_bwr_density",
        lambda temperature, pressure, phase="vapor": torch.ones_like(temperature),
    )
    monkeypatch.setattr(
        thermal_module,
        "methane_thermal_conductivity",
        lambda temperature, density, include_critical_enhancement=False: (
            -torch.ones_like(temperature)
        ),
    )
    monkeypatch.setattr(
        thermal_module,
        "corresponding_states_viscosity",
        lambda *args, **kwargs: torch.ones_like(args[0]),
    )
    monkeypatch.setattr(
        thermal_module,
        "methane_viscosity",
        lambda temperature, density: torch.ones_like(temperature),
    )
    monkeypatch.setattr(
        thermal_module,
        "_internal_energy_conductivity",
        lambda viscosity, heat_capacity, mass, density: torch.zeros_like(viscosity),
    )
    with pytest.raises(InvalidStateError, match="non-positive conductivity"):
        corresponding_states_thermal_conductivity(
            torch.tensor(250.0, dtype=DTYPE),
            torch.tensor(1.0e5, dtype=DTYPE),
            composition,
            components,
            ideal,
            methane_ideal,
        )


def test_brock_bird_riedel_and_parachor_values():
    alpha = riedel_parameter(
        torch.tensor(341.88, dtype=DTYPE),
        torch.tensor(507.6, dtype=DTYPE),
        torch.tensor(3.025e6, dtype=DTYPE),
    )
    assert float(alpha) == pytest.approx(7.266815581318145, rel=2.0e-14)
    tension = brock_bird_surface_tension(
        torch.tensor(300.0, dtype=DTYPE),
        torch.tensor(507.6, dtype=DTYPE),
        torch.tensor(3.025e6, dtype=DTYPE),
        torch.tensor(341.88, dtype=DTYPE),
    )
    assert float(tension) == pytest.approx(0.017643986859287919, rel=2.0e-14)
    assert float(parachor_from_molar_mass(torch.tensor(0.2, dtype=DTYPE))) == pytest.approx(527.3)


def test_surface_tension_parameter_tables_and_validation():
    like = torch.tensor(0.0, dtype=DTYPE)
    torch.testing.assert_close(
        published_parachors(("methane", "n_hexane"), like=like),
        torch.tensor([77.3, 271.0], dtype=DTYPE),
    )
    torch.testing.assert_close(
        published_lee_chien_b(("methane", "n_hexane"), like=like),
        torch.tensor([3.403, 3.726], dtype=DTYPE),
    )
    with pytest.raises(KeyError, match="no published parachor"):
        published_parachors(("water",), like=like)
    with pytest.raises(KeyError, match="no published Lee-Chien"):
        published_lee_chien_b(("water",), like=like)
    with pytest.raises(InvalidStateError, match="finite and positive"):
        parachor_from_molar_mass(torch.tensor(-1.0))
    with pytest.raises(InvalidStateError, match="0 < Tb < Tc"):
        riedel_parameter(torch.tensor(600.0), torch.tensor(500.0), torch.tensor(1.0e6))
    with pytest.raises(InvalidStateError, match="subcritical"):
        brock_bird_surface_tension(
            torch.tensor(510.0),
            torch.tensor(507.6),
            torch.tensor(3.025e6),
            torch.tensor(341.88),
        )
    with pytest.raises(InvalidStateError, match="negative surface tension"):
        brock_bird_surface_tension(
            torch.tensor(300.0),
            torch.tensor(500.0),
            torch.tensor(101_325.0),
            torch.tensor(350.0),
        )


def test_weinaug_katz_fixed_and_danesh_exponents_and_gradients():
    liquid_density = torch.tensor([8000.0, 7500.0], dtype=DTYPE, requires_grad=True)
    vapor_density = torch.tensor([500.0, 800.0], dtype=DTYPE)
    liquid = torch.tensor([[0.2, 0.8], [0.3, 0.7]], dtype=DTYPE)
    vapor = torch.tensor([[0.9, 0.1], [0.8, 0.2]], dtype=DTYPE)
    parachor = published_parachors(("methane", "n_butane"), like=liquid)
    fixed = weinaug_katz_interfacial_tension(
        liquid_density,
        vapor_density,
        liquid,
        vapor,
        parachor,
    )
    danesh = weinaug_katz_interfacial_tension(
        liquid_density,
        vapor_density,
        liquid,
        vapor,
        parachor,
        liquid_mass_density=torch.tensor([600.0, 550.0], dtype=DTYPE),
        vapor_mass_density=torch.tensor([40.0, 60.0], dtype=DTYPE),
        danesh_exponent=True,
    )
    torch.testing.assert_close(
        fixed.detach(),
        torch.tensor([0.0029108874800407502, 0.0014651970521238247], dtype=DTYPE),
        rtol=3.0e-14,
        atol=0.0,
    )
    assert (danesh < fixed).all()
    gradient = torch.autograd.grad((fixed + danesh).sum(), liquid_density)[0]
    assert torch.isfinite(gradient).all()


def test_weinaug_katz_validation_paths():
    composition = torch.tensor([0.5, 0.5])
    parachor = torch.tensor([77.3, 191.7])
    with pytest.raises(ValueError, match="same shape"):
        weinaug_katz_interfacial_tension(
            torch.tensor(8000.0),
            torch.tensor(500.0),
            composition,
            torch.tensor([[0.5, 0.5]]),
            parachor,
        )
    with pytest.raises(ValueError, match="one value"):
        weinaug_katz_interfacial_tension(
            torch.tensor(8000.0),
            torch.tensor(500.0),
            composition,
            composition,
            torch.tensor([77.3]),
        )
    with pytest.raises(InvalidStateError, match="rho_liquid"):
        weinaug_katz_interfacial_tension(
            torch.tensor(500.0),
            torch.tensor(8000.0),
            composition,
            composition,
            parachor,
        )
    with pytest.raises(ValueError, match="requires liquid"):
        weinaug_katz_interfacial_tension(
            torch.tensor(8000.0),
            torch.tensor(500.0),
            composition,
            composition,
            parachor,
            danesh_exponent=True,
        )
    with pytest.raises(InvalidStateError, match="rho_liquid"):
        weinaug_katz_interfacial_tension(
            torch.tensor(8000.0),
            torch.tensor(500.0),
            composition,
            composition,
            parachor,
            liquid_mass_density=torch.tensor(50.0),
            vapor_mass_density=torch.tensor(60.0),
            danesh_exponent=True,
        )


def test_lee_chien_values_and_validation_paths():
    names = ("methane", "n_butane")
    components = component_set(names, dtype=DTYPE)
    liquid = torch.tensor([[0.2, 0.8], [0.3, 0.7]], dtype=DTYPE)
    vapor = torch.tensor([[0.9, 0.1], [0.8, 0.2]], dtype=DTYPE)
    riedel = riedel_parameter(
        torch.tensor([111.66, 272.65], dtype=DTYPE),
        components.critical_temperature,
        components.critical_pressure,
    )
    b = published_lee_chien_b(names, like=liquid)
    result = lee_chien_interfacial_tension(
        torch.tensor([8000.0, 7500.0], dtype=DTYPE),
        torch.tensor([500.0, 800.0], dtype=DTYPE),
        liquid,
        vapor,
        components,
        riedel,
        b,
    )
    torch.testing.assert_close(
        result,
        torch.tensor([0.0026535248305568056, 0.001326063272785827], dtype=DTYPE),
        rtol=3.0e-14,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="same shape"):
        lee_chien_interfacial_tension(
            torch.tensor(8000.0),
            torch.tensor(500.0),
            liquid[0],
            vapor,
            components,
            riedel,
            b,
        )
    with pytest.raises(ValueError, match="component set sizes"):
        lee_chien_interfacial_tension(
            torch.tensor(8000.0),
            torch.tensor(500.0),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            components,
            riedel,
            b,
        )
    with pytest.raises(ValueError, match="one value per component"):
        lee_chien_interfacial_tension(
            torch.tensor(8000.0),
            torch.tensor(500.0),
            liquid[0],
            vapor[0],
            components,
            torch.ones(1),
            b,
        )
    no_volumes = ComponentSet(
        components.names,
        components.critical_temperature,
        components.critical_pressure,
        components.acentric_factor,
        components.molar_mass,
    )
    with pytest.raises(ValueError, match="critical molar volumes"):
        lee_chien_interfacial_tension(
            torch.tensor(8000.0),
            torch.tensor(500.0),
            liquid[0],
            vapor[0],
            no_volumes,
            riedel,
            b,
        )
    with pytest.raises(ValueError, match="finite and B positive"):
        lee_chien_interfacial_tension(
            torch.tensor(8000.0),
            torch.tensor(500.0),
            liquid[0],
            vapor[0],
            components,
            riedel,
            torch.zeros_like(b),
        )
    with pytest.raises(InvalidStateError, match="rho_liquid"):
        lee_chien_interfacial_tension(
            torch.tensor(500.0),
            torch.tensor(8000.0),
            liquid[0],
            vapor[0],
            components,
            riedel,
            b,
        )
    with pytest.raises(InvalidStateError, match="non-finite"):
        lee_chien_interfacial_tension(
            torch.tensor(8000.0),
            torch.tensor(500.0),
            liquid[0],
            vapor[0],
            components,
            torch.zeros_like(riedel),
            b,
        )


@pytest.mark.parametrize(
    ("temperature", "pressure", "expected"),
    [
        (300.0, 1.0e5, 1.1257424092147265e-5),
        (300.0, 1.0e7, 1.3985981420095590e-5),
        (200.0, 5.0e6, 1.0817631902125182e-5),
        (400.0, 2.0e7, 1.8367241007012082e-5),
    ],
)
def test_methane_reference_against_frozen_independent_values(
    temperature,
    pressure,
    expected,
):
    state_temperature = torch.tensor(temperature, dtype=DTYPE)
    state_pressure = torch.tensor(pressure, dtype=DTYPE)
    density = methane_bwr_density(state_temperature, state_pressure)
    torch.testing.assert_close(
        methane_bwr_pressure(state_temperature, density),
        state_pressure / 101_325.0,
        rtol=2.0e-11,
        atol=1.0e-10,
    )
    viscosity = methane_viscosity(state_temperature, density)
    assert float(viscosity) == pytest.approx(expected, rel=2.0e-12)


def test_methane_density_liquid_branch_and_autodiff():
    temperature = torch.tensor(150.0, dtype=DTYPE, requires_grad=True)
    pressure = torch.tensor(5.0e6, dtype=DTYPE)
    density = methane_bwr_density(temperature, pressure, phase="liquid")
    assert density > 10.0
    viscosity = methane_viscosity(temperature, density)
    gradient = torch.autograd.grad(viscosity, temperature)[0]
    assert torch.isfinite(gradient)


def test_corresponding_states_pure_methane_identity_and_mixture():
    temperature = torch.tensor(300.0, dtype=DTYPE)
    pressure = torch.tensor(1.0e7, dtype=DTYPE)
    methane = component_set(("methane",), dtype=DTYPE)
    mixture_value = corresponding_states_viscosity(
        temperature,
        pressure,
        torch.tensor([1.0], dtype=DTYPE),
        methane,
    )
    direct = methane_viscosity(
        temperature,
        methane_bwr_density(temperature, pressure),
    )
    torch.testing.assert_close(mixture_value, direct, rtol=5.0e-14, atol=0.0)

    components = component_set(("methane", "n_decane"), dtype=DTYPE)
    heavy = corresponding_states_viscosity(
        temperature,
        pressure,
        torch.tensor([0.7, 0.3], dtype=DTYPE),
        components,
        phase="liquid",
    )
    assert torch.isfinite(heavy)
    assert heavy > 0.0


def test_transport_validation(monkeypatch):
    with pytest.raises(InvalidStateError, match="positive"):
        methane_bwr_density(torch.tensor(0.0), torch.tensor(1.0e5))
    with pytest.raises(ValueError, match="unknown viscosity"):
        methane_bwr_density(
            torch.tensor(300.0),
            torch.tensor(1.0e5),
            phase="solid",
        )
    with pytest.raises(ValueError, match="final component axis"):
        corresponding_states_viscosity(
            torch.tensor(300.0),
            torch.tensor(1.0e5),
            torch.tensor(1.0),
            component_set(("methane",)),
        )
    with pytest.raises(ValueError, match="sizes"):
        corresponding_states_viscosity(
            torch.tensor(300.0),
            torch.tensor(1.0e5),
            torch.tensor([0.5, 0.5]),
            component_set(("methane",)),
        )
    monkeypatch.setattr(
        viscosity_module,
        "methane_bwr_pressure",
        lambda temperature, density: density.new_tensor(0.0),
    )
    with pytest.raises(InvalidStateError, match="no pressure root"):
        methane_bwr_density(torch.tensor(300.0), torch.tensor(1.0e5))


def test_lbc_against_whitson_appendix_b_problem_7_and_autodiff():
    names = (
        "methane",
        "ethane",
        "propane",
        "isobutane",
        "n_butane",
        "isopentane",
        "n_pentane",
        "n_hexane",
        "n_octane",
    )
    composition = torch.tensor(
        [0.875, 0.083, 0.021, 0.006, 0.008, 0.003, 0.002, 0.001, 0.001],
        dtype=DTYPE,
    )
    conversion = 0.028316846592 / 453.59237
    critical_volume = (
        torch.tensor(
            [1.590, 2.370, 3.250, 4.208, 4.080, 4.899, 4.870, 5.929, 7.882],
            dtype=DTYPE,
        )
        * conversion
    )
    density = torch.tensor(0.627 / (1.752 * conversion), dtype=DTYPE)
    coefficients = torch.tensor(
        [0.10230, 0.023364, 0.058533, -0.040758, 0.0093324],
        dtype=DTYPE,
        requires_grad=True,
    )
    viscosity = lbc_viscosity(
        torch.tensor(620.0 / 1.8, dtype=DTYPE),
        density,
        composition,
        component_set(names, dtype=DTYPE),
        critical_volume=critical_volume,
        coefficients=coefficients,
    )
    # Whitson Table B-12 reports 0.0166 cP; intermediate values are rounded.
    assert float((1000.0 * viscosity).detach()) == pytest.approx(0.0166, abs=2.5e-4)
    gradient = torch.autograd.grad(viscosity, coefficients)[0]
    assert torch.isfinite(gradient).all()


def test_lbc_pseudocomponent_volume_and_validation():
    conversion = 0.028316846592 / 453.59237
    volume = lbc_pseudocomponent_critical_volume(
        torch.tensor(0.114, dtype=DTYPE),
        torch.tensor(780.0, dtype=DTYPE),
    )
    assert float(volume / conversion) == pytest.approx(8.0043138, rel=2.0e-7)
    with pytest.raises(ValueError, match="finite and positive"):
        lbc_pseudocomponent_critical_volume(torch.tensor(-1.0), torch.tensor(800.0))


def test_lbc_validation_paths():
    methane = component_set(("methane",))
    temperature = torch.tensor(300.0)
    density = torch.tensor(10.0)
    with pytest.raises(ValueError, match="final component axis"):
        lbc_viscosity(temperature, density, torch.tensor(1.0), methane)
    with pytest.raises(ValueError, match="sizes"):
        lbc_viscosity(temperature, density, torch.tensor([0.5, 0.5]), methane)
    with pytest.raises(InvalidStateError, match="temperature"):
        lbc_viscosity(torch.tensor(0.0), density, torch.tensor([1.0]), methane)
    no_volumes = type(methane)(
        methane.names,
        methane.critical_temperature,
        methane.critical_pressure,
        methane.acentric_factor,
        methane.molar_mass,
    )
    with pytest.raises(ValueError, match="required"):
        lbc_viscosity(temperature, density, torch.tensor([1.0]), no_volumes)
    with pytest.raises(ValueError, match="one value"):
        lbc_viscosity(
            torch.tensor(1.0),
            density,
            torch.tensor([1.0]),
            methane,
            critical_volume=torch.ones(2),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        lbc_viscosity(
            temperature,
            density,
            torch.tensor([1.0]),
            methane,
            critical_volume=torch.tensor([float("nan")]),
        )
    with pytest.raises(ValueError, match="five-element"):
        lbc_viscosity(
            temperature,
            density,
            torch.tensor([1.0]),
            methane,
            coefficients=torch.ones(4),
        )
    with pytest.raises(InvalidStateError, match="non-positive"):
        lbc_viscosity(
            torch.tensor(1.0),
            density,
            torch.tensor([1.0]),
            methane,
            coefficients=torch.zeros(5),
        )
