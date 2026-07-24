from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from torch_flash import (
    ChemicalState,
    FlashResult,
    component_set,
    identify_flash_phases,
    identify_phase,
    peng_robinson_1978,
    phase_properties,
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
    assert tuple(phase.phase_identification.kind for phase in identified) == (
        "liquid",
        "vapor",
    )
    assert all(phase.phase_identification.method == "density-ordering" for phase in identified)
    assert not any(phase.phase_identification.ambiguous for phase in identified)

    equal_volume = replace(vapor, molar_volume=liquid.molar_volume)
    ambiguous = identify_flash_phases((liquid, equal_volume))
    assert tuple(phase.phase_identification.kind for phase in ambiguous) == (
        "unknown",
        "unknown",
    )
    assert all(phase.phase_identification.ambiguous for phase in ambiguous)

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
