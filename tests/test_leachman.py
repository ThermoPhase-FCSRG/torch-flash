import pytest
import torch

from torch_flash.database import load_model_parameters
from torch_flash.eos import (
    DEFAULT_LEACHMAN_NORMAL_HYDROGEN,
    PureFluidHelmholtzEOS,
    PureFluidHelmholtzMetadata,
    gerg2008_hydrogen_2021,
    leachman_normal_hydrogen,
)
from torch_flash.exceptions import ParameterDatabaseError

DTYPE = torch.float64


def test_leachman_normal_hydrogen_parameter_identity_and_terms():
    model = leachman_normal_hydrogen(dtype=DTYPE)

    assert DEFAULT_LEACHMAN_NORMAL_HYDROGEN == ("pure-helmholtz.leachman-2009-normal-hydrogen")
    assert model.metadata.model == "Leachman normal hydrogen (2009)"
    assert model.metadata.reference == "doi:10.1063/1.3160306"
    assert model.metadata.fluid == "hydrogen"
    torch.testing.assert_close(model.gas_constant, torch.tensor(8.314472, dtype=DTYPE))
    torch.testing.assert_close(
        model.critical_temperature,
        torch.tensor(33.145, dtype=DTYPE),
    )
    torch.testing.assert_close(
        model.critical_density,
        torch.tensor(15_508.0, dtype=DTYPE),
    )
    torch.testing.assert_close(
        model.critical_pressure,
        torch.tensor(1_296_400.0, dtype=DTYPE),
    )
    assert model.residual_term_count == 14
    parameters = load_model_parameters(DEFAULT_LEACHMAN_NORMAL_HYDROGEN)
    power, gaussian = parameters.parameters["components"]["hydrogen"]["residual"]
    torch.testing.assert_close(
        torch.tensor([*power["n"], *gaussian["n"]], dtype=DTYPE),
        torch.tensor(
            [
                -6.93643,
                0.01,
                2.1101,
                4.52059,
                0.732564,
                -1.34086,
                0.130985,
                -0.777414,
                0.351944,
                -0.0211716,
                0.0226312,
                0.032187,
                -0.0231752,
                0.0557346,
            ],
            dtype=DTYPE,
        ),
    )


def test_leachman_parameters_are_h2_tailored_source_of_truth():
    leachman = load_model_parameters(DEFAULT_LEACHMAN_NORMAL_HYDROGEN)
    tailored = load_model_parameters("multiparameter.gerg-2008-hydrogen-2021")

    assert (
        leachman.parameters["components"]["hydrogen"]
        == tailored.parameters["components"]["hydrogen"]
    )


def test_leachman_pure_properties_match_h2_tailored_pure_limit():
    standalone = leachman_normal_hydrogen(dtype=DTYPE)
    mixture_component = gerg2008_hydrogen_2021(("hydrogen",), dtype=DTYPE)
    temperature = torch.tensor([40.0, 100.0, 300.0], dtype=DTYPE)
    density = torch.tensor([100.0, 1_000.0, 5_000.0], dtype=DTYPE)
    composition = torch.ones((3, 1), dtype=DTYPE)

    calculations = (
        (
            lambda: standalone.alpha_ideal(temperature, density),
            lambda: mixture_component.alpha_ideal(temperature, density, composition),
        ),
        (
            lambda: standalone.alpha_residual(temperature, density),
            lambda: mixture_component.alpha_residual(temperature, density, composition),
        ),
        (
            lambda: standalone.alpha_total(temperature, density),
            lambda: mixture_component.alpha_total(temperature, density, composition),
        ),
        (
            lambda: standalone.pressure(temperature, density.reciprocal()),
            lambda: mixture_component.pressure(
                temperature,
                density.reciprocal(),
                composition,
            ),
        ),
        (
            lambda: standalone.compressibility_factor(
                temperature,
                density.reciprocal(),
            ),
            lambda: (
                mixture_component.pressure(
                    temperature,
                    density.reciprocal(),
                    composition,
                )
                * density.reciprocal()
                / (mixture_component.gas_constant * temperature)
            ),
        ),
        (
            lambda: standalone.molar_helmholtz_energy(temperature, density),
            lambda: mixture_component.molar_helmholtz_energy(
                temperature,
                density,
                composition,
            ),
        ),
        (
            lambda: standalone.molar_internal_energy(temperature, density),
            lambda: mixture_component.molar_internal_energy(
                temperature,
                density,
                composition,
            ),
        ),
        (
            lambda: standalone.molar_enthalpy(temperature, density),
            lambda: mixture_component.molar_enthalpy(temperature, density, composition),
        ),
        (
            lambda: standalone.molar_entropy(temperature, density),
            lambda: mixture_component.molar_entropy(temperature, density, composition),
        ),
        (
            lambda: standalone.molar_gibbs_energy(temperature, density),
            lambda: mixture_component.molar_gibbs_energy(
                temperature,
                density,
                composition,
            ),
        ),
        (
            lambda: standalone.molar_heat_capacity_cp(temperature, density),
            lambda: mixture_component.molar_heat_capacity_cp(
                temperature,
                density,
                composition,
            ),
        ),
        (
            lambda: standalone.speed_of_sound(temperature, density),
            lambda: mixture_component.speed_of_sound(temperature, density, composition),
        ),
    )
    for calculate_standalone, calculate_mixture in calculations:
        torch.testing.assert_close(
            calculate_standalone(),
            calculate_mixture(),
            rtol=0.0,
            atol=0.0,
        )


def test_pure_helmholtz_wrapper_rejects_incompatible_kernels():
    metadata = PureFluidHelmholtzMetadata(
        model="test",
        reference="test",
        version="test",
        fluid="hydrogen",
    )
    binary = gerg2008_hydrogen_2021(("hydrogen", "methane"), dtype=DTYPE)
    with pytest.raises(ValueError, match="exactly one component"):
        PureFluidHelmholtzEOS(binary, metadata)

    hydrogen = gerg2008_hydrogen_2021(("hydrogen",), dtype=DTYPE)
    wrong_fluid = PureFluidHelmholtzMetadata(
        model="test",
        reference="test",
        version="test",
        fluid="methane",
    )
    with pytest.raises(ValueError, match="metadata"):
        PureFluidHelmholtzEOS(hydrogen, wrong_fluid)


def test_leachman_reproduces_paper_table14_at_25_kelvin():
    """Verify both saturation branches against Leachman et al. Table 14."""
    model = leachman_normal_hydrogen(dtype=DTYPE)
    mass_density = torch.tensor([64.701, 3.8938], dtype=DTYPE)
    temperature = torch.full_like(mass_density, 25.0)
    molar_density = mass_density / model.molar_mass

    calculated = torch.stack(
        (
            model.molar_enthalpy(temperature, molar_density) / (1.0e3 * model.molar_mass),
            model.molar_entropy(temperature, molar_density) / (1.0e3 * model.molar_mass),
            model.molar_heat_capacity_cv(temperature, molar_density) / (1.0e3 * model.molar_mass),
            model.molar_heat_capacity_cp(temperature, molar_density) / (1.0e3 * model.molar_mass),
            model.speed_of_sound(temperature, molar_density),
        ),
        dim=-1,
    )
    published = torch.tensor(
        [
            [54.161, 2.2417, 5.9521, 13.298, 964.22],
            [463.37, 18.610, 6.7578, 15.289, 375.20],
        ],
        dtype=DTYPE,
    )
    tolerance = torch.tensor(
        [0.002, 1.0e-4, 2.0e-4, 2.0e-4, 0.03],
        dtype=DTYPE,
    )
    assert torch.all(torch.abs(calculated - published) < tolerance)

    published_pressure = torch.tensor(321_000.0, dtype=DTYPE)
    pressure = model.pressure(temperature, molar_density.reciprocal())
    assert torch.all(torch.abs(pressure - published_pressure) < 210.0)

    rooted_density = torch.stack(
        (
            model.molar_mass / model.molar_volume(temperature[0], published_pressure, "liquid"),
            model.molar_mass / model.molar_volume(temperature[0], published_pressure, "vapor"),
        )
    )
    torch.testing.assert_close(
        rooted_density,
        mass_density,
        rtol=0.0,
        atol=5.0e-4,
    )


def test_leachman_normal_hydrogen_batches_and_preserves_autodiff():
    model = leachman_normal_hydrogen(dtype=DTYPE, trainable=True)
    temperature = torch.tensor([50.0, 100.0, 300.0], dtype=DTYPE, requires_grad=True)
    density = torch.tensor([50.0, 500.0, 2_000.0], dtype=DTYPE, requires_grad=True)

    pressure = model.pressure(temperature, density.reciprocal())
    heat_capacity = model.molar_heat_capacity_cp(temperature, density)
    assert pressure.shape == (3,)
    assert heat_capacity.shape == (3,)
    assert torch.isfinite(pressure).all()
    assert torch.isfinite(heat_capacity).all()

    (pressure.sum() + heat_capacity.sum()).backward()
    assert temperature.grad is not None and torch.isfinite(temperature.grad).all()
    assert density.grad is not None and torch.isfinite(density.grad).all()
    parameter_gradients = [parameter.grad for parameter in model.parameters()]
    assert parameter_gradients
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)


def test_leachman_normal_hydrogen_rejects_another_parameter_identity():
    with pytest.raises(ParameterDatabaseError, match="Leachman normal-hydrogen"):
        leachman_normal_hydrogen(parameter_set="multiparameter.gerg-2008")
