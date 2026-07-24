from __future__ import annotations

import pytest
import torch

import torch_flash
from torch_flash.components import ComponentSet, component, component_set
from torch_flash.types import (
    ChemicalState,
    FlashResult,
    PhaseProperties,
    RachfordRiceResult,
    StabilityResult,
    normalize_composition,
)


def test_package_metadata_and_public_surface():
    assert torch_flash.__version__ == "0.1.0"
    assert "two_phase_flash" in torch_flash.__all__


def test_component_alias_lookup_and_set_conversion():
    methane = component(" CH4 ")
    assert methane.name == "methane"
    assert component("n-butane").name == "n_butane"
    components = component_set(("C1", "co2"), dtype=torch.float32)
    assert components.names == ("methane", "carbon_dioxide")
    assert components.ncomponents == 2
    converted = components.to(dtype=torch.float64, device="cpu")
    assert converted.critical_temperature.dtype == torch.float64
    assert converted.molar_mass.device.type == "cpu"
    assert converted.critical_volume is not None
    assert converted.critical_volume.dtype == torch.float64
    assert component("methane").critical_volume == pytest.approx(9.93e-5)
    assert isinstance(converted, ComponentSet)

    without_volume = ComponentSet(
        ("test",),
        torch.ones(1),
        torch.ones(1),
        torch.zeros(1),
        torch.ones(1),
    ).to(dtype=torch.float64)
    assert without_volume.critical_volume is None


def test_component_errors():
    with pytest.raises(KeyError, match="unknown component"):
        component("unobtainium")
    with pytest.raises(ValueError, match="at least one"):
        component_set(())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([2, 3], [0.4, 0.6]),
        ([0.2, -1.0e-16, 0.8], [0.2, 0.0, 0.8]),
    ],
)
def test_normalize_composition(value, expected):
    actual = normalize_composition(torch.tensor(value))
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=actual.dtype))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (torch.tensor(1.0), "at least one dimension"),
        (torch.tensor([float("nan"), 1.0]), "finite"),
        (torch.tensor([-0.1, 1.1]), "negative"),
        (torch.tensor([0.0, 0.0]), "positive sum"),
    ],
)
def test_normalize_composition_errors(value, message):
    with pytest.raises(ValueError, match=message):
        normalize_composition(value)


def test_normalize_composition_compiler_numerical_path(monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    actual = normalize_composition(torch.tensor([2, 3]))
    torch.testing.assert_close(actual, torch.tensor([0.4, 0.6]))


def test_state_and_result_properties():
    with pytest.raises(ValueError, match="temperature"):
        ChemicalState(torch.tensor(0.0), torch.tensor(1.0), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="pressure"):
        ChemicalState(torch.tensor(1.0), torch.tensor(0.0), torch.tensor([1.0]))

    phase = PhaseProperties(
        kind="vapor",
        composition=torch.tensor([1.0]),
        compressibility_factor=torch.tensor(1.0),
        molar_volume=torch.tensor(1.0),
        log_fugacity_coefficients=torch.tensor([0.2]),
        fugacities=torch.tensor([1.0e5]),
        log_fugacities=torch.tensor([0.0]),
        chemical_potentials=torch.tensor([0.0]),
        reduced_chemical_potentials=torch.tensor([0.0]),
        molar_gibbs_energy=torch.tensor(0.0),
        molar_helmholtz_energy=torch.tensor(-1.0),
        reduced_gibbs_energy=torch.tensor(0.0),
        reduced_helmholtz_energy=torch.tensor(-1.0),
        reduced_residual_gibbs_energy=torch.tensor(0.2),
        reduced_residual_helmholtz_energy=torch.tensor(0.2),
    )
    torch.testing.assert_close(phase.fugacity_coefficients, torch.exp(torch.tensor([0.2])))
    torch.testing.assert_close(phase.fugacities, torch.tensor([1.0e5]))
    rr = RachfordRiceResult(
        torch.tensor(0.5),
        torch.tensor(0.5),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        2,
        torch.tensor(True),
        torch.tensor(0.0),
    )
    stability = StabilityResult(True, torch.tensor(0.0), torch.tensor([1.0]), 1, True)
    result = FlashResult(
        torch.tensor([1.0]),
        (phase,),
        True,
        1,
        torch.tensor(0.0),
        stability.stable,
        {"rr": rr.iterations},
    )
    assert result.nphases == 1
