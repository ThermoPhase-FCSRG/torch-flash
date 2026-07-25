from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
import torch

from torch_flash import (
    ChemicalState,
    FlashResult,
    PhaseIdentification,
    PhaseIdentificationCriterion,
    component_set,
    identify_flash_phases,
    identify_phase,
    li_pseudo_critical_temperature,
    negative_flash_residual,
    peng_robinson_1978,
    phase_identification_parameter,
    phase_properties,
    phase_response_derivatives,
    two_phase_flash,
    volume_to_covolume_ratio,
)
from torch_flash.constants import R


class _NoCovolumeModel:
    def _volume(self, temperature, pressure, phase):
        multiplier = 0.05 if phase == "liquid" else 1.0
        return multiplier * R * temperature / pressure

    def select_z(self, temperature, pressure, composition, phase="stable"):
        return pressure * self._volume(temperature, pressure, phase) / (R * temperature)

    def molar_volume(self, temperature, pressure, composition, phase="stable"):
        return self._volume(temperature, pressure, phase)

    def log_fugacity_coefficients(self, temperature, pressure, composition, phase="stable"):
        return torch.zeros_like(composition)


class _InvalidCovolumeModel(_NoCovolumeModel):
    def mixture_parameters(self, temperature, composition):
        return temperature.new_tensor(1.0), temperature.new_tensor(-1.0)


class _ZeroRatioModel(_NoCovolumeModel):
    def select_z(self, temperature, pressure, composition, phase="stable"):
        return temperature.new_tensor(0.0)

    def mixture_parameters(self, temperature, composition):
        return temperature.new_tensor(1.0), temperature.new_tensor(1.0)


class _IdealGasResponseModel:
    def pressure(self, temperature, molar_volume, composition):
        del composition
        return R * temperature / molar_volume

    def molar_volume(self, temperature, pressure, composition, phase="stable"):
        del composition, phase
        return R * temperature / pressure


class _VanDerWaalsResponseModel:
    def __init__(self, molar_volume: float, attraction: float, covolume: float):
        self.volume = torch.tensor(molar_volume, dtype=torch.float64)
        self.attraction = torch.tensor(attraction, dtype=torch.float64)
        self.covolume = torch.tensor(covolume, dtype=torch.float64)

    def pressure(self, temperature, molar_volume, composition):
        del composition
        return (
            R * temperature / (molar_volume - self.covolume)
            - self.attraction / molar_volume.square()
        )

    def molar_volume(self, temperature, pressure, composition, phase="stable"):
        del temperature, pressure, composition, phase
        return self.volume


class _TemperatureIndependentStableModel:
    def pressure(self, temperature, molar_volume, composition):
        del temperature, composition
        return molar_volume.reciprocal()

    def molar_volume(self, temperature, pressure, composition, phase="stable"):
        del pressure, composition, phase
        return temperature.new_tensor(1.0e-3)


class _FixedVolumePathologicalModel:
    def __init__(self, behavior: str):
        self.behavior = behavior

    def pressure(self, temperature, molar_volume, composition):
        del composition
        if self.behavior == "nonfinite":
            return temperature * molar_volume * temperature.new_tensor(float("nan"))
        return R * temperature * molar_volume

    def molar_volume(self, temperature, pressure, composition, phase="stable"):
        del pressure, composition, phase
        if self.behavior == "invalid-volume":
            return temperature.new_tensor(0.0)
        return temperature.new_tensor(1.0e-3)


class _CriticalTemperatureOnly:
    critical_temperature = torch.tensor([190.0, 304.0], dtype=torch.float64)


def _state(temperature: float, pressure: float = 1.0e5) -> ChemicalState:
    return ChemicalState(
        torch.tensor(temperature, dtype=torch.float64),
        torch.tensor(pressure, dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )


def test_pedersen_volume_to_covolume_identifies_liquid_and_vapor():
    components = component_set(("methane", "n_butane"), dtype=torch.float64)
    model = peng_robinson_1978(components)
    liquid = identify_phase(model, _state(220.0, 8.0e6))
    vapor = identify_phase(model, _state(450.0))

    assert liquid.kind == "liquid"
    assert vapor.kind == "vapor"
    assert liquid.method == vapor.method == "pedersen-volume-to-covolume"
    assert liquid.criterion_value is not None
    assert liquid.threshold is not None
    assert liquid.criterion_value < liquid.threshold
    assert vapor.criterion_value is not None
    assert vapor.threshold is not None
    assert vapor.criterion_value > vapor.threshold
    assert not liquid.ambiguous
    assert not vapor.ambiguous

    boundary = identify_phase(
        model,
        _state(220.0, 8.0e6),
        threshold=float(liquid.criterion_value),
    )
    assert boundary.ambiguous


def test_identification_uses_untranslated_cubic_volume_and_retains_autodiff():
    components = component_set(("methane", "n_butane"), dtype=torch.float64)
    base = peng_robinson_1978(components)
    translated = peng_robinson_1978(
        components,
        volume_translation=torch.tensor([2.0e-5, 3.0e-5], dtype=torch.float64),
    )
    temperature = torch.tensor(270.0, dtype=torch.float64, requires_grad=True)
    state = ChemicalState(
        temperature,
        torch.tensor(3.0e6, dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    base_identity = identify_phase(base, state)
    translated_identity = identify_phase(translated, state)
    assert base_identity.criterion_value is not None
    assert translated_identity.criterion_value is not None
    torch.testing.assert_close(
        translated_identity.criterion_value,
        base_identity.criterion_value,
    )
    translated_identity.criterion_value.backward()
    assert temperature.grad is not None
    assert torch.isfinite(temperature.grad)

    batched_state = ChemicalState(
        torch.tensor([250.0, 350.0], dtype=torch.float64),
        torch.tensor([1.0e6, 1.0e6], dtype=torch.float64),
        torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64),
    )
    ratio = volume_to_covolume_ratio(base, batched_state)
    assert ratio.shape == (2,)
    assert torch.isfinite(ratio).all()


def test_all_phase_identification_methods_identify_clear_methane_states():
    components = component_set(("methane",), dtype=torch.float64)
    model = peng_robinson_1978(components)
    assert model.critical_volume is not None
    torch.testing.assert_close(model.critical_volume, components.critical_volume)
    methods: tuple[PhaseIdentificationCriterion, ...] = (
        "li-pseudo-critical-temperature",
        "pedersen-volume-to-covolume",
        "perschke-negative-flash",
        "pasad-isothermal-compressibility-derivative",
        "bennett-thermal-expansion-derivative",
        "venkatarathnam-oellrich-phase-identification-parameter",
    )
    liquid = ChemicalState(
        torch.tensor(120.0, dtype=torch.float64),
        torch.tensor(5.0e6, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
    )
    vapor = ChemicalState(
        torch.tensor(500.0, dtype=torch.float64),
        torch.tensor(1.0e5, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
    )

    for method in methods:
        liquid_result = identify_phase(
            model,
            liquid,
            method=method,
            ambiguity_relative_tolerance=0.0,
        )
        vapor_result = identify_phase(
            model,
            vapor,
            method=method,
            ambiguity_relative_tolerance=0.0,
        )
        assert liquid_result.kind == "liquid"
        assert vapor_result.kind == "vapor"
        assert liquid_result.method == vapor_result.method == method
        assert liquid_result.criterion_value is not None
        assert vapor_result.criterion_value is not None
        assert torch.isfinite(liquid_result.criterion_value)
        assert torch.isfinite(vapor_result.criterion_value)


def test_li_pseudo_critical_temperature_and_negative_flash_match_equations():
    components = component_set(("methane", "carbon_dioxide"), dtype=torch.float64)
    model = peng_robinson_1978(components)
    composition = torch.tensor(
        [[0.25, 0.75], [0.80, 0.20]],
        dtype=torch.float64,
    )
    critical_volume = torch.tensor([1.0e-4, 2.0e-4], dtype=torch.float64)
    calculated = li_pseudo_critical_temperature(
        model,
        composition,
        factor=1.1,
        critical_volume=critical_volume,
    )
    weights = composition * critical_volume
    expected = 1.1 * (weights * components.critical_temperature).sum(dim=-1) / weights.sum(dim=-1)
    torch.testing.assert_close(calculated, expected)

    state = ChemicalState(
        torch.tensor([180.0, 250.0], dtype=torch.float64),
        torch.tensor([2.0e6, 5.0e6], dtype=torch.float64),
        composition,
    )
    k_values = (
        components.critical_pressure
        / state.pressure[..., None]
        * torch.exp(
            5.373
            * (1.0 + components.acentric_factor)
            * (1.0 - components.critical_temperature / state.temperature[..., None])
        )
    )
    expected_residual = (composition * (k_values - 1.0) / (1.0 + 0.5 * (k_values - 1.0))).sum(
        dim=-1
    )
    torch.testing.assert_close(negative_flash_residual(model, state), expected_residual)


def test_phase_response_autodiff_recovers_ideal_gas_derivatives():
    temperature = torch.tensor(350.0, dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor(2.0e6, dtype=torch.float64)
    state = ChemicalState(temperature, pressure, torch.ones(1, dtype=torch.float64))
    response = phase_response_derivatives(_IdealGasResponseModel(), state)

    torch.testing.assert_close(response.molar_volume, R * temperature / pressure)
    torch.testing.assert_close(
        response.isothermal_compressibility,
        pressure.reciprocal(),
    )
    torch.testing.assert_close(
        response.thermal_expansion_coefficient,
        temperature.reciprocal(),
    )
    torch.testing.assert_close(
        response.isothermal_compressibility_temperature_derivative,
        torch.zeros_like(temperature),
        atol=1.0e-18,
        rtol=0.0,
    )
    torch.testing.assert_close(
        response.thermal_expansion_temperature_derivative,
        -temperature.reciprocal().square(),
    )
    response.thermal_expansion_temperature_derivative.backward()
    assert temperature.grad is not None
    torch.testing.assert_close(
        temperature.grad,
        2.0 / temperature.detach().pow(3),
    )


def test_batched_phase_response_matches_scalar_reverse_mode_oracle():
    components = component_set(("methane", "carbon_dioxide"), dtype=torch.float64)
    model = peng_robinson_1978(
        components,
        kij=torch.tensor([[0.0, 0.08], [0.08, 0.0]], dtype=torch.float64),
    )
    state = ChemicalState(
        torch.tensor([180.0, 320.0], dtype=torch.float64),
        torch.tensor([3.0e6, 1.0e6], dtype=torch.float64),
        torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float64),
    )
    batched = phase_response_derivatives(model, state)

    oracle_rows = []
    for index in range(state.temperature.numel()):
        temperature = state.temperature[index]
        pressure = state.pressure[index]
        composition = state.composition[index]
        volume = model.molar_volume(
            temperature,
            pressure,
            composition,
            "stable",
        )

        def pressure_at_tv(
            current_temperature,
            current_volume,
            current_composition=composition,
        ):
            return model.pressure(
                current_temperature,
                current_volume,
                current_composition,
            )

        def response_at_tv(current_temperature, current_volume):
            pressure_temperature, pressure_volume = torch.func.grad(
                pressure_at_tv,
                argnums=(0, 1),
            )(current_temperature, current_volume)
            volume_temperature = -pressure_temperature / pressure_volume
            return torch.stack(
                (
                    -1.0 / (current_volume * pressure_volume),
                    volume_temperature / current_volume,
                    volume_temperature,
                )
            )

        response = response_at_tv(temperature, volume)
        response_temperature, response_volume = torch.func.jacrev(
            response_at_tv,
            argnums=(0, 1),
        )(temperature, volume)
        oracle_rows.append(
            torch.stack(
                (
                    volume,
                    response[0],
                    response[1],
                    response_temperature[0] + response_volume[0] * response[2],
                    response_temperature[1] + response_volume[1] * response[2],
                )
            )
        )

    oracle = torch.stack(oracle_rows)
    calculated = torch.stack(
        (
            batched.molar_volume,
            batched.isothermal_compressibility,
            batched.thermal_expansion_coefficient,
            batched.isothermal_compressibility_temperature_derivative,
            batched.thermal_expansion_temperature_derivative,
        ),
        dim=-1,
    )
    torch.testing.assert_close(calculated, oracle, rtol=2.0e-12, atol=1.0e-18)


def test_phase_identification_parameter_matches_analytic_van_der_waals_derivatives():
    temperature = torch.tensor(300.0, dtype=torch.float64)
    volume = 3.0e-4
    attraction = 0.05
    covolume = 4.0e-5
    model = _VanDerWaalsResponseModel(volume, attraction, covolume)
    pressure = model.pressure(
        temperature,
        model.volume,
        torch.ones(1, dtype=torch.float64),
    )
    state = ChemicalState(
        temperature,
        pressure,
        torch.ones(1, dtype=torch.float64),
    )

    parameter = phase_identification_parameter(model, state)
    pressure_temperature = R / (volume - covolume)
    pressure_volume = (
        -R * float(temperature) / (volume - covolume) ** 2 + 2.0 * attraction / volume**3
    )
    pressure_volume_temperature = -R / (volume - covolume) ** 2
    pressure_volume_second = (
        2.0 * R * float(temperature) / (volume - covolume) ** 3 - 6.0 * attraction / volume**4
    )
    expected = volume * (
        pressure_volume_temperature / pressure_temperature
        - pressure_volume_second / pressure_volume
    )
    torch.testing.assert_close(
        parameter,
        torch.tensor(expected, dtype=torch.float64),
        rtol=2.0e-14,
        atol=0.0,
    )


def test_phase_identification_parameter_recovers_ideal_gas_limit():
    temperature = torch.tensor(350.0, dtype=torch.float64, requires_grad=True)
    state = ChemicalState(
        temperature,
        torch.tensor(2.0e6, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
    )
    parameter = phase_identification_parameter(_IdealGasResponseModel(), state)
    torch.testing.assert_close(
        parameter,
        torch.ones((), dtype=torch.float64),
        atol=2.0e-15,
        rtol=0.0,
    )
    identification = identify_phase(
        _IdealGasResponseModel(),
        state,
        method="venkatarathnam-oellrich-phase-identification-parameter",
    )
    assert identification.kind == "vapor"
    assert identification.ambiguous
    parameter.backward()
    assert temperature.grad is not None
    torch.testing.assert_close(
        temperature.grad,
        torch.zeros_like(temperature),
        atol=1.0e-16,
        rtol=0.0,
    )


def test_derivative_identification_retains_trainable_model_gradient():
    components = component_set(("methane", "carbon_dioxide"), dtype=torch.float64)
    model = peng_robinson_1978(
        components,
        kij=torch.tensor([[0.0, 0.08], [0.08, 0.0]], dtype=torch.float64),
        trainable=True,
    )
    state = ChemicalState(
        torch.tensor(180.0, dtype=torch.float64),
        torch.tensor(3.0e6, dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    result = identify_phase(
        model,
        state,
        method="bennett-thermal-expansion-derivative",
    )
    assert result.criterion_value is not None
    result.criterion_value.backward()
    raw_kij = model.mixing.raw_kij
    assert isinstance(raw_kij, torch.Tensor)
    assert raw_kij.grad is not None
    assert torch.isfinite(raw_kij.grad).all()
    assert float(raw_kij.grad[0, 1].abs()) > 0.0

    batched_temperature = torch.tensor(
        [180.0, 240.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    batched_response = phase_response_derivatives(
        model,
        ChemicalState(
            batched_temperature,
            torch.tensor([3.0e6, 2.0e6], dtype=torch.float64),
            torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float64),
        ),
    )
    temperature_gradient, interaction_gradient = torch.autograd.grad(
        batched_response.thermal_expansion_temperature_derivative.sum(),
        (batched_temperature, model.mixing.raw_kij),
    )
    assert torch.isfinite(temperature_gradient).all()
    assert torch.isfinite(interaction_gradient).all()
    assert float(temperature_gradient.abs().max()) > 0.0
    assert float(interaction_gradient[0, 1].abs()) > 0.0


def test_phase_identification_parameter_retains_state_and_model_gradients():
    components = component_set(("methane", "carbon_dioxide"), dtype=torch.float64)
    model = peng_robinson_1978(
        components,
        kij=torch.tensor([[0.0, 0.08], [0.08, 0.0]], dtype=torch.float64),
        trainable=True,
    )
    temperature = torch.tensor(240.0, dtype=torch.float64, requires_grad=True)
    state = ChemicalState(
        temperature,
        torch.tensor(3.0e6, dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    result = identify_phase(
        model,
        state,
        method="venkatarathnam-oellrich-phase-identification-parameter",
    )
    assert result.criterion_value is not None
    temperature_gradient, interaction_gradient = torch.autograd.grad(
        result.criterion_value,
        (temperature, model.mixing.raw_kij),
    )
    assert torch.isfinite(temperature_gradient)
    assert torch.isfinite(interaction_gradient).all()
    assert float(temperature_gradient.abs()) > 0.0
    assert float(interaction_gradient[0, 1].abs()) > 0.0


@pytest.mark.parametrize(
    ("threshold", "ambiguity", "message"),
    [
        (0.0, 0.05, "threshold"),
        (float("inf"), 0.05, "threshold"),
        (1.75, -0.1, "ambiguity"),
        (1.75, float("nan"), "ambiguity"),
    ],
)
def test_phase_identification_option_validation(threshold, ambiguity, message):
    with pytest.raises(ValueError, match=message):
        identify_phase(
            _NoCovolumeModel(),
            _state(300.0),
            threshold=threshold,
            ambiguity_relative_tolerance=ambiguity,
        )


def test_phase_identification_model_and_state_validation():
    batched = ChemicalState(
        torch.tensor([300.0, 310.0]),
        torch.tensor([1.0e5, 1.0e5]),
        torch.tensor([[0.5, 0.5], [0.4, 0.6]]),
    )
    with pytest.raises(ValueError, match="scalar T-P state"):
        identify_phase(_NoCovolumeModel(), batched)
    with pytest.raises(TypeError, match="select_z"):
        identify_phase(object(), _state(300.0))
    with pytest.raises(TypeError, match="select_z"):
        volume_to_covolume_ratio(object(), _state(300.0))
    with pytest.raises(TypeError, match="covolume"):
        volume_to_covolume_ratio(_NoCovolumeModel(), _state(300.0))
    with pytest.raises(ValueError, match="covolume"):
        identify_phase(_InvalidCovolumeModel(), _state(300.0))
    with pytest.raises(ValueError, match="ratio"):
        volume_to_covolume_ratio(_ZeroRatioModel(), _state(300.0))
    with pytest.raises(ValueError, match="ratio"):
        identify_phase(_ZeroRatioModel(), _state(300.0))
    with pytest.raises(ValueError, match="unknown phase-identification"):
        identify_phase(
            _NoCovolumeModel(),
            _state(300.0),
            method="not-a-method",  # type: ignore[arg-type]
        )


def test_new_method_input_validation_and_unavailable_results():
    components = component_set(("methane", "carbon_dioxide"), dtype=torch.float64)
    model = peng_robinson_1978(components)
    with pytest.raises(ValueError, match="factor"):
        li_pseudo_critical_temperature(model, torch.tensor([0.5, 0.5]), factor=0.0)
    with pytest.raises(ValueError, match="component counts"):
        li_pseudo_critical_temperature(model, torch.ones(1))
    with pytest.raises(ValueError, match="equal-length"):
        li_pseudo_critical_temperature(
            model,
            torch.tensor([0.5, 0.5]),
            critical_volume=torch.ones(3),
        )
    with pytest.raises(TypeError, match="critical temperatures"):
        li_pseudo_critical_temperature(object(), torch.ones(1))
    with pytest.raises(TypeError, match="critical volumes"):
        li_pseudo_critical_temperature(
            _CriticalTemperatureOnly(),
            torch.tensor([0.5, 0.5]),
        )
    for critical_temperature in (
        torch.tensor([float("nan"), 304.0], dtype=torch.float64),
        torch.tensor([-190.0, 304.0], dtype=torch.float64),
    ):
        invalid_temperature_model = replace(
            components,
            critical_temperature=critical_temperature,
        )
        with pytest.raises(ValueError, match="critical temperatures"):
            li_pseudo_critical_temperature(
                invalid_temperature_model,
                torch.tensor([0.5, 0.5], dtype=torch.float64),
            )
    for critical_volume in (
        torch.tensor([float("inf"), 1.0e-4], dtype=torch.float64),
        torch.tensor([0.0, 1.0e-4], dtype=torch.float64),
    ):
        with pytest.raises(ValueError, match="critical volumes"):
            li_pseudo_critical_temperature(
                model,
                torch.tensor([0.5, 0.5], dtype=torch.float64),
                critical_volume=critical_volume,
            )
    with pytest.raises(TypeError, match="critical constants"):
        negative_flash_residual(object(), _state(300.0))
    response_temperature = torch.tensor([300.0, 310.0])
    response_pressure = torch.tensor([1.0e5, 2.0e5])
    batched_response = phase_response_derivatives(
        _IdealGasResponseModel(),
        ChemicalState(
            response_temperature,
            response_pressure,
            torch.tensor([[0.5, 0.5], [0.4, 0.6]]),
        ),
    )
    torch.testing.assert_close(
        batched_response.isothermal_compressibility,
        response_pressure.reciprocal(),
    )
    torch.testing.assert_close(
        batched_response.thermal_expansion_coefficient,
        response_temperature.reciprocal(),
    )
    torch.testing.assert_close(
        batched_response.isothermal_compressibility_temperature_derivative,
        torch.zeros_like(response_temperature),
        atol=5.0e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        batched_response.thermal_expansion_temperature_derivative,
        -response_temperature.reciprocal().square(),
    )
    with pytest.raises(ValueError, match="shapes"):
        phase_response_derivatives(
            _IdealGasResponseModel(),
            ChemicalState(
                torch.tensor([300.0, 310.0]),
                torch.tensor([1.0e5, 1.0e5, 1.0e5]),
                torch.ones((2, 1)),
            ),
        )
    with pytest.raises(TypeError, match="pressure and molar_volume"):
        phase_response_derivatives(object(), _state(300.0))
    with pytest.raises(TypeError, match="pressure and molar_volume"):
        phase_identification_parameter(object(), _state(300.0))
    batched_parameter = phase_identification_parameter(
        _IdealGasResponseModel(),
        ChemicalState(
            torch.tensor([300.0, 310.0]),
            torch.tensor([1.0e5, 1.0e5]),
            torch.tensor([[0.5, 0.5], [0.4, 0.6]]),
        ),
    )
    torch.testing.assert_close(batched_parameter, torch.ones(2))
    with pytest.raises(ValueError, match="shapes"):
        phase_identification_parameter(
            _IdealGasResponseModel(),
            ChemicalState(
                torch.tensor([300.0, 310.0]),
                torch.tensor([1.0e5, 1.0e5, 1.0e5]),
                torch.ones((2, 1)),
            ),
        )
    with pytest.raises(ValueError, match="relative tolerance"):
        phase_identification_parameter(
            _IdealGasResponseModel(),
            ChemicalState(
                torch.tensor(300.0),
                torch.tensor(1.0e5),
                torch.ones(1),
            ),
            denominator_relative_tolerance=-1.0,
        )
    with pytest.raises(ValueError, match="finite positive"):
        phase_identification_parameter(
            _FixedVolumePathologicalModel("invalid-volume"),
            _state(300.0),
        )
    with pytest.raises(ValueError, match="nonfinite"):
        phase_identification_parameter(
            _FixedVolumePathologicalModel("nonfinite"),
            _state(300.0),
        )
    with pytest.raises(ValueError, match="mechanically stable"):
        phase_identification_parameter(
            _FixedVolumePathologicalModel("unstable"),
            _state(300.0),
        )
    for behavior, message in (
        ("invalid-volume", "finite positive"),
        ("nonfinite", "nonfinite"),
        ("unstable", "mechanically stable"),
    ):
        with pytest.raises(ValueError, match=message):
            phase_response_derivatives(
                _FixedVolumePathologicalModel(behavior),
                _state(300.0),
            )

    singular_state = ChemicalState(
        torch.tensor(300.0, dtype=torch.float64),
        torch.tensor(1.0e5, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="singular"):
        phase_identification_parameter(
            _TemperatureIndependentStableModel(),
            singular_state,
        )
    singular_identification = identify_phase(
        _TemperatureIndependentStableModel(),
        singular_state,
        method="venkatarathnam-oellrich-phase-identification-parameter",
    )
    assert singular_identification.kind == "unknown"
    assert singular_identification.criterion_value is None
    assert singular_identification.ambiguous

    unavailable_methods: tuple[PhaseIdentificationCriterion, ...] = (
        "li-pseudo-critical-temperature",
        "perschke-negative-flash",
        "pasad-isothermal-compressibility-derivative",
        "bennett-thermal-expansion-derivative",
        "venkatarathnam-oellrich-phase-identification-parameter",
    )
    for method in unavailable_methods:
        result = identify_phase(_NoCovolumeModel(), _state(300.0), method=method)
        assert result.kind == "unknown"
        assert result.method == "unavailable"


def test_unavailable_single_phase_and_density_ordering_fallback():
    model = _NoCovolumeModel()
    state = _state(300.0)
    liquid = phase_properties(model, state, "liquid", caloric=False)
    vapor = phase_properties(model, state, "vapor", caloric=False)
    assert liquid.phase_identification is not None
    assert liquid.phase_identification.kind == "unknown"
    assert liquid.phase_identification.method == "unavailable"
    assert identify_flash_phases(()) == ()
    assert identify_flash_phases((liquid,)) == (liquid,)

    identified = identify_flash_phases((liquid, vapor))
    identities = tuple(
        cast(PhaseIdentification, phase.phase_identification) for phase in identified
    )
    assert tuple(identity.kind for identity in identities) == (
        "liquid",
        "vapor",
    )
    assert all(identity.method == "density-ordering" for identity in identities)
    assert not any(identity.ambiguous for identity in identities)

    equal_volume = replace(vapor, molar_volume=liquid.molar_volume)
    ambiguous = identify_flash_phases((liquid, equal_volume))
    ambiguous_identities = tuple(
        cast(PhaseIdentification, phase.phase_identification) for phase in ambiguous
    )
    assert tuple(identity.kind for identity in ambiguous_identities) == (
        "unknown",
        "unknown",
    )
    assert all(identity.ambiguous for identity in ambiguous_identities)

    invalid = replace(vapor, molar_volume=torch.tensor(float("nan")))
    with pytest.raises(ValueError, match="molar volumes"):
        identify_flash_phases((liquid, invalid))


def test_flash_result_phase_identification_and_regime(binary_model, two_phase_state):
    result = two_phase_flash(binary_model, two_phase_state, check_stability=False)
    assert result.phase_kinds == ("liquid", "vapor")
    assert result.phase_regime == "vapor-liquid"
    assert all(identity is not None for identity in result.phase_identifications)

    stable = two_phase_flash(binary_model, _state(450.0))
    assert stable.phase_kinds == ("vapor",)
    assert stable.phase_regime == "vapor"

    unknown_phase = phase_properties(
        _NoCovolumeModel(),
        _state(300.0),
        caloric=False,
    )
    unknown = FlashResult(
        torch.ones(1),
        (unknown_phase,),
        True,
        1,
        torch.tensor(0.0),
        True,
    )
    assert unknown.phase_kinds == ("unknown",)
    assert unknown.phase_regime == "unknown"
    two_unknown = replace(unknown, phases=(unknown_phase, unknown_phase))
    assert two_unknown.phase_regime == "2-phase-unknown"
    assert replace(unknown, phases=()).phase_regime == "unknown"
