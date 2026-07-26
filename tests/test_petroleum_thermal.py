from __future__ import annotations

import pytest
import torch

from torch_flash import (
    ChemicalState,
    component_set,
    enhanced_predictive_peng_robinson_1978,
    molar_enthalpy_of_mixing,
    pedersen_binary_interaction,
    peng_robinson_1978,
    phase_properties,
    poling_ideal_gas,
    thermal_properties,
    whitson_binary_interaction,
)
from torch_flash.constants import R
from torch_flash.exceptions import InvalidStateError
from torch_flash.standard_state import IdealGasPolynomial


class _IdealModel:
    molar_mass = torch.tensor([0.028], dtype=torch.float64)

    def log_fugacity_coefficients(self, temperature, pressure, composition, phase="stable"):
        return torch.zeros_like(composition)

    def select_z(self, temperature, pressure, composition, phase="stable"):
        return temperature.new_tensor(1.0)

    def molar_volume(self, temperature, pressure, composition, phase="stable"):
        return R * temperature / pressure


class _IdealModelWithoutMass:
    def log_fugacity_coefficients(self, temperature, pressure, composition, phase="stable"):
        return torch.zeros_like(composition)

    def select_z(self, temperature, pressure, composition, phase="stable"):
        return temperature.new_tensor(1.0)

    def molar_volume(self, temperature, pressure, composition, phase="stable"):
        return R * temperature / pressure


def test_whitson_and_pedersen_binary_interaction_tables():
    components = component_set(("nitrogen", "co2", "h2s", "methane", "n_decane"))
    whitson_pr = whitson_binary_interaction(components, "PR")
    whitson_srk = whitson_binary_interaction(components, "SRK")
    pedersen_pr = pedersen_binary_interaction(components, "PR")
    pedersen_srk = pedersen_binary_interaction(components, "SRK")

    assert whitson_pr[0, 2] == pytest.approx(0.130)
    assert whitson_srk[1, 3] == pytest.approx(0.120)
    assert pedersen_pr[0, 1] == pytest.approx(0.0170)
    assert pedersen_pr[1, 4] == pytest.approx(0.0100)
    assert pedersen_srk[0, 1] == pytest.approx(-0.0315)
    assert torch.equal(torch.diagonal(whitson_pr), torch.zeros(5, dtype=torch.float64))
    torch.testing.assert_close(pedersen_pr, pedersen_pr.mT)


@pytest.mark.parametrize("function", [whitson_binary_interaction, pedersen_binary_interaction])
def test_petroleum_binary_interaction_validation(function):
    with pytest.raises(ValueError, match=r"PR.*SRK"):
        function(component_set(("methane",)), "CPA")
    with pytest.raises(KeyError, match="no parameters"):
        function(component_set(("hydrogen", "methane")))


def test_poling_standard_state_and_ideal_gas_thermal_identities():
    standard = IdealGasPolynomial(
        torch.tensor([[3.5 * R]], dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        reference_temperature=273.15,
    )
    state = ChemicalState(
        torch.tensor(350.0, dtype=torch.float64),
        torch.tensor(2.0e5, dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
    )
    result = thermal_properties(_IdealModel(), state, standard)
    expected_h = 3.5 * R * (350.0 - 273.15)
    assert float(result.molar_enthalpy) == pytest.approx(expected_h)
    assert float(result.isobaric_heat_capacity) == pytest.approx(3.5 * R)
    assert float(result.isochoric_heat_capacity) == pytest.approx(2.5 * R)
    assert float(result.joule_thomson_coefficient) == pytest.approx(0.0, abs=1.0e-16)
    assert result.speed_of_sound is not None
    expected_sound = (3.5 / 2.5 * R * 350.0 / 0.028) ** 0.5
    assert float(result.speed_of_sound) == pytest.approx(expected_sound)
    torch.testing.assert_close(result.residual_enthalpy, torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(result.residual_entropy, torch.tensor(0.0, dtype=torch.float64))
    direct = phase_properties(
        _IdealModel(),
        state,
        standard_state=standard,
    )
    torch.testing.assert_close(
        result.molar_gibbs_energy,
        result.molar_enthalpy - state.temperature * result.molar_entropy,
    )
    torch.testing.assert_close(
        result.molar_helmholtz_energy,
        result.molar_gibbs_energy - state.pressure * direct.molar_volume,
    )
    torch.testing.assert_close(
        result.reduced_gibbs_energy,
        result.molar_gibbs_energy / (R * state.temperature),
    )
    torch.testing.assert_close(
        result.reduced_helmholtz_energy,
        result.molar_helmholtz_energy / (R * state.temperature),
    )
    torch.testing.assert_close(
        result.reduced_residual_gibbs_energy,
        torch.tensor(0.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.reduced_residual_helmholtz_energy,
        torch.tensor(0.0, dtype=torch.float64),
    )
    torch.testing.assert_close(result.molar_gibbs_energy, direct.molar_gibbs_energy)
    torch.testing.assert_close(
        result.molar_helmholtz_energy,
        direct.molar_helmholtz_energy,
    )
    torch.testing.assert_close(
        result.reduced_gibbs_energy,
        direct.reduced_gibbs_energy,
    )
    torch.testing.assert_close(
        result.reduced_helmholtz_energy,
        direct.reduced_helmholtz_energy,
    )

    no_sound = thermal_properties(_IdealModelWithoutMass(), state, standard)
    assert no_sound.speed_of_sound is None

    poling = poling_ideal_gas(["co2"])
    assert float(
        poling.heat_capacity(torch.tensor(300.0, dtype=torch.float64))[0]
    ) == pytest.approx(
        37.10,
        rel=5.0e-4,
    )
    with pytest.raises(KeyError, match="no frozen Poling"):
        poling_ideal_gas(["methanol"])


@pytest.mark.parametrize(
    ("temperature_c", "pressure_bar", "reference_c_per_bar"),
    [
        (49.85, 74.0, 0.8587),
        (49.85, 101.3, 0.5273),
        (99.85, 141.9, 0.4097),
        (149.85, 202.7, 0.2522),
    ],
)
def test_co2_joule_thomson_against_pedersen_table_8_1(
    temperature_c,
    pressure_bar,
    reference_c_per_bar,
):
    components = component_set(("co2",))
    model = peng_robinson_1978(components)
    state = ChemicalState(
        torch.tensor(temperature_c + 273.15, dtype=torch.float64),
        torch.tensor(pressure_bar * 1.0e5, dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
    )
    result = thermal_properties(model, state, poling_ideal_gas(["co2"]))
    predicted_c_per_bar = 1.0e5 * float(result.joule_thomson_coefficient)
    assert predicted_c_per_bar == pytest.approx(reference_c_per_bar, rel=1.2e-2)


def test_propane_vapor_response_functions_remain_finite_in_three_root_region():
    """Guard the second derivatives used for Pedersen Table 8.2.

    At this state PR78 has three real volume roots.  Inactive Cardano branches
    must not contaminate the vapor-root second derivatives with NaNs.
    """
    components = component_set(("propane",))
    model = peng_robinson_1978(components)
    state = ChemicalState(
        torch.tensor(294.25, dtype=torch.float64),
        torch.tensor(1.72e5, dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
    )
    result = thermal_properties(model, state, poling_ideal_gas(["propane"]), "vapor")

    assert result.speed_of_sound is not None
    responses = torch.stack(
        (
            result.isobaric_heat_capacity,
            result.isochoric_heat_capacity,
            result.joule_thomson_coefficient,
            result.speed_of_sound,
        )
    )
    assert torch.isfinite(responses).all()
    assert 1.0e5 * float(result.joule_thomson_coefficient) == pytest.approx(
        1.5966,
        rel=2.0e-4,
    )


def test_volume_translation_enthalpy_correction():
    components = component_set(("methane",))
    state = ChemicalState(
        torch.tensor(300.0, dtype=torch.float64),
        torch.tensor(5.0e6, dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
    )
    standard = poling_ideal_gas(["methane"])
    base = thermal_properties(peng_robinson_1978(components), state, standard, "vapor")
    shift = torch.tensor([-1.0e-6], dtype=torch.float64)
    translated = thermal_properties(
        peng_robinson_1978(components, volume_translation=shift),
        state,
        standard,
        "vapor",
    )
    torch.testing.assert_close(
        translated.molar_enthalpy - base.molar_enthalpy,
        state.pressure * shift[0],
    )
    torch.testing.assert_close(translated.molar_internal_energy, base.molar_internal_energy)
    torch.testing.assert_close(translated.molar_entropy, base.molar_entropy)


def test_molar_enthalpy_of_mixing_matches_epr78_figure2_scale_and_autodiff():
    model = enhanced_predictive_peng_robinson_1978(
        component_set(("nitrogen", "methane"), dtype=torch.float64)
    )
    temperature = torch.tensor(91.5, dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor(8.22e5, dtype=torch.float64)
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)

    value = molar_enthalpy_of_mixing(
        model,
        temperature,
        pressure,
        composition,
    )
    derivative = torch.autograd.grad(value, temperature)[0]

    assert float(value.detach()) == pytest.approx(142.85, abs=0.15)
    assert torch.isfinite(derivative)
    torch.testing.assert_close(
        molar_enthalpy_of_mixing(
            model,
            temperature.detach(),
            pressure,
            torch.tensor([1.0, 0.0], dtype=torch.float64),
        ),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1.0e-10,
        rtol=0.0,
    )

    ideal = molar_enthalpy_of_mixing(
        _IdealModel(),
        torch.tensor(300.0, dtype=torch.float64),
        torch.tensor(1.0e5, dtype=torch.float64),
        torch.tensor([0.25, 0.75], dtype=torch.float64),
    )
    torch.testing.assert_close(ideal, torch.tensor(0.0, dtype=torch.float64))


def test_molar_enthalpy_of_mixing_batches_compositions_with_scalar_parity():
    model = enhanced_predictive_peng_robinson_1978(
        component_set(("nitrogen", "methane"), dtype=torch.float64)
    )
    temperature = torch.tensor(195.15, dtype=torch.float64)
    pressure = torch.tensor(50.66e5, dtype=torch.float64)
    fractions = torch.linspace(0.05, 0.95, 7, dtype=torch.float64)
    compositions = torch.stack((fractions, 1.0 - fractions), dim=-1)

    batched = molar_enthalpy_of_mixing(
        model,
        temperature,
        pressure,
        compositions,
    )
    scalar = torch.stack(
        tuple(
            molar_enthalpy_of_mixing(
                model,
                temperature,
                pressure,
                composition,
            )
            for composition in compositions
        )
    )

    assert batched.shape == (7,)
    torch.testing.assert_close(batched, scalar, rtol=2.0e-13, atol=2.0e-11)


@pytest.mark.parametrize(
    ("temperature", "pressure", "composition", "phase", "match"),
    [
        (
            torch.tensor(-1.0),
            torch.tensor(1.0e5),
            torch.tensor([0.5, 0.5]),
            "stable",
            "temperature",
        ),
        (
            torch.tensor(300.0),
            torch.tensor([1.0e5]),
            torch.tensor([0.5, 0.5]),
            "stable",
            "pressure",
        ),
        (
            torch.tensor(300.0),
            torch.tensor(1.0e5),
            torch.tensor([1, 1]),
            "stable",
            "floating tensor",
        ),
        (
            torch.tensor(300.0),
            torch.tensor(1.0e5),
            torch.tensor([0.5, -0.5]),
            "stable",
            "nonnegative",
        ),
        (
            torch.tensor(300.0),
            torch.tensor(1.0e5),
            torch.tensor([1.0]),
            "stable",
            "at least two",
        ),
        (
            torch.tensor(300.0),
            torch.tensor(1.0e5),
            torch.tensor([0.5, 0.5]),
            "solid",
            "phase",
        ),
    ],
)
def test_molar_enthalpy_of_mixing_validation(
    temperature,
    pressure,
    composition,
    phase,
    match,
):
    with pytest.raises(ValueError, match=match):
        molar_enthalpy_of_mixing(
            _IdealModel(),
            temperature,
            pressure,
            composition,
            phase,
        )


def test_thermal_property_validation():
    standard = IdealGasPolynomial(
        torch.tensor([[3.5 * R]], dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
    )
    model = _IdealModel()
    with pytest.raises(ValueError, match="scalar T-P"):
        thermal_properties(
            model,
            ChemicalState(torch.tensor([300.0]), torch.tensor([1.0e5]), torch.tensor([1.0])),
            standard,
        )
    with pytest.raises(ValueError, match="composition vector"):
        thermal_properties(
            model,
            ChemicalState(torch.tensor(300.0), torch.tensor(1.0e5), torch.tensor([[1.0]])),
            standard,
        )
    state = ChemicalState(torch.tensor(300.0), torch.tensor(1.0e5), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="reference_pressure"):
        thermal_properties(model, state, standard, reference_pressure=0.0)
    two_component_standard = IdealGasPolynomial(
        torch.ones((2, 1)),
        torch.zeros(2),
        torch.zeros(2),
    )
    with pytest.raises(ValueError, match="component count"):
        thermal_properties(model, state, two_component_standard)
    with pytest.raises(ValueError, match="scalar or have one"):
        thermal_properties(model, state, standard, molar_mass=torch.ones(2))
    with pytest.raises(ValueError, match="finite and positive"):
        thermal_properties(model, state, standard, molar_mass=torch.tensor(0.0))

    class ExpandingModel(_IdealModel):
        def molar_volume(self, temperature, pressure, composition, phase="stable"):
            return R * temperature * pressure

    with pytest.raises(InvalidStateError, match="response functions"):
        thermal_properties(ExpandingModel(), state, standard)
