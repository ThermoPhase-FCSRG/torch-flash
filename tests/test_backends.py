from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from torch_flash.backends import BackendCapabilities, CoolPropBackend, TeqpBackend
from torch_flash.components import component_set
from torch_flash.constants import R
from torch_flash.exceptions import ConvergenceError, InvalidStateError, ModelCapabilityError


class _FakeCoolPropState:
    def __init__(self):
        self.phase = None
        self.fractions = None
        self.inputs = None

    def set_mole_fractions(self, values):
        self.fractions = values

    def specify_phase(self, phase):
        self.phase = phase

    def unspecify_phase(self):
        self.phase = None

    def update(self, inputs, pressure, temperature):
        self.inputs = (inputs, pressure, temperature)

    def compressibility_factor(self):
        return 0.9

    def rhomolar(self):
        return 1000.0

    def fugacity_coefficient(self, index):
        return [0.8, 1.2][index]


class _FakeCoolProp:
    iphase_liquid = 1
    iphase_gas = 2
    PT_INPUTS = 3

    def __init__(self):
        self.state = _FakeCoolPropState()

    def AbstractState(self, backend, fluids):
        self.backend = backend
        self.fluids = fluids
        return self.state


def test_coolprop_adapter_with_fake_module(monkeypatch):
    fake = _FakeCoolProp()
    module = types.ModuleType("CoolProp")
    module.CoolProp = fake
    monkeypatch.setitem(sys.modules, "CoolProp", module)
    backend = CoolPropBackend(("carbon_dioxide", "n_butane"))
    assert fake.fluids == "CarbonDioxide&n-Butane"
    assert backend.capabilities == BackendCapabilities(False, False, True, "CoolProp HEOS")
    temperature = torch.tensor(300.0, dtype=torch.float64)
    pressure = torch.tensor(1.0e5, dtype=torch.float64)
    composition = torch.tensor([0.4, 0.6], dtype=torch.float64)
    for phase, expected in (("liquid", 1), ("vapor", 2), ("stable", None)):
        assert backend.select_z(temperature, pressure, composition, phase) == 0.9
        assert fake.state.phase == expected
    assert backend.molar_volume(temperature, pressure, composition) == 1.0e-3
    torch.testing.assert_close(
        backend.log_fugacity_coefficients(temperature, pressure, composition),
        torch.log(torch.tensor([0.8, 1.2], dtype=torch.float64)),
    )
    refprop = CoolPropBackend(("methane",), backend="REFPROP")
    assert refprop.capabilities.exact_model == "REFPROP-selected model"
    with pytest.raises(ValueError, match="one scalar"):
        backend.select_z(temperature[None], pressure, composition)


def test_coolprop_import_and_capability_errors(monkeypatch):
    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "CoolProp":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "CoolProp", raising=False)
    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImportError, match="optional"):
        CoolPropBackend(("methane",))
    monkeypatch.setattr(builtins, "__import__", original_import)

    fake = _FakeCoolProp()
    fake.state.fugacity_coefficient = lambda index: (_ for _ in ()).throw(ValueError)
    module = types.ModuleType("CoolProp")
    module.CoolProp = fake
    monkeypatch.setitem(sys.modules, "CoolProp", module)
    backend = CoolPropBackend(("methane",))
    with pytest.raises(ModelCapabilityError, match="does not expose"):
        backend.log_fugacity_coefficients(
            torch.tensor(300.0),
            torch.tensor(1.0e5),
            torch.tensor([1.0]),
        )


class _FakeTeqpModel:
    def __init__(self, fugacity=(0.8, 1.2)):
        self.fugacity = np.asarray(fugacity)

    def get_R(self, composition):
        return R

    def get_Ar01(self, temperature, density, composition):
        return 0.0

    def get_fugacity_coefficients(self, temperature, densities):
        return self.fugacity[: len(densities)]


def test_teqp_adapter_ideal_fake_and_constructors(monkeypatch):
    calls = {}
    module = types.ModuleType("teqp")

    def canonical_pr(*args):
        calls["pr"] = args
        return _FakeTeqpModel()

    module.canonical_PR = canonical_pr

    def make_model(config):
        calls["gerg"] = config
        return _FakeTeqpModel()

    module.make_model = make_model
    module.get_datapath = lambda: "/coefficient-data"

    def build_multifluid_model(components, root, binary_path, *, departurepath):
        calls["eoscg"] = (components, root, binary_path, departurepath)
        return _FakeTeqpModel()

    module.build_multifluid_model = build_multifluid_model
    monkeypatch.setitem(sys.modules, "teqp", module)
    components = component_set(("methane", "carbon_dioxide"))
    pr = TeqpBackend.canonical_peng_robinson(components)
    assert pr.capabilities.exact_model == "teqp canonical Peng-Robinson"
    gerg = TeqpBackend.gerg2008(("methane", "carbon_dioxide"))
    assert calls["gerg"]["model"]["names"] == ["methane", "carbondioxide"]
    assert gerg.capabilities.exact_model == "GERG-2008 residual (teqp)"
    eoscg = TeqpBackend.eoscg_2015(("carbon_dioxide", "water"))
    coefficient_data = Path("/coefficient-data")
    assert calls["eoscg"] == (
        ["CarbonDioxide", "Water"],
        str(coefficient_data),
        str(coefficient_data / "dev" / "mixtures" / "mixture_binary_pairs.json"),
        str(coefficient_data / "dev" / "mixtures" / "mixture_departure_functions.json"),
    )
    assert eoscg.capabilities.exact_model == "EOS-CG-2015 multifluid (teqp)"
    with pytest.raises(ValueError, match="component must be"):
        TeqpBackend.eoscg_2015(("methane",))

    temperature = torch.tensor(300.0, dtype=torch.float64)
    pressure = torch.tensor(1.0e5, dtype=torch.float64)
    composition = torch.tensor([0.4, 0.6], dtype=torch.float64)
    for phase in ("liquid", "vapor", "stable"):
        volume = gerg.molar_volume(temperature, pressure, composition, phase)
        torch.testing.assert_close(volume, R * temperature / pressure, rtol=2.0e-10, atol=0.0)
        torch.testing.assert_close(
            gerg.select_z(temperature, pressure, composition, phase),
            torch.tensor(1.0, dtype=torch.float64),
            rtol=2.0e-10,
            atol=0.0,
        )
    torch.testing.assert_close(
        gerg.log_fugacity_coefficients(temperature, pressure, composition),
        torch.log(torch.tensor([0.8, 1.2], dtype=torch.float64)),
    )
    torch.testing.assert_close(
        gerg.pressure(temperature, R * temperature / pressure, composition),
        pressure,
    )
    with pytest.raises(ValueError, match="one scalar"):
        gerg.select_z(temperature[None], pressure, composition)
    with pytest.raises(ValueError, match="one scalar"):
        gerg.pressure(temperature[None], R * temperature / pressure, composition)
    with pytest.raises(ValueError, match="unknown phase"):
        gerg.molar_volume(temperature, pressure, composition, "solid")


def test_teqp_errors(monkeypatch):
    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "teqp":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "teqp", raising=False)
    monkeypatch.setattr(builtins, "__import__", missing)
    components = component_set(("methane",))
    with pytest.raises(ImportError, match="optional"):
        TeqpBackend.canonical_peng_robinson(components)
    with pytest.raises(ImportError, match="optional"):
        TeqpBackend.gerg2008(("methane",))
    with pytest.raises(ImportError, match="optional"):
        TeqpBackend.eoscg_2015(("carbon_dioxide", "water"))

    model = _FakeTeqpModel((0.0,))
    backend = TeqpBackend(("methane",), model, exact_model="fake")
    temperature = torch.tensor(300.0, dtype=torch.float64)
    pressure = torch.tensor(1.0e5, dtype=torch.float64)
    composition = torch.tensor([1.0], dtype=torch.float64)
    with pytest.raises(InvalidStateError, match="invalid fugacity"):
        backend.log_fugacity_coefficients(temperature, pressure, composition)
    monkeypatch.setattr(
        backend,
        "_pressure",
        lambda temperature, density, composition: 0.0,
    )
    with pytest.raises(ConvergenceError, match="no pressure root"):
        backend.molar_volume(temperature, pressure, composition)
    monkeypatch.setattr(
        backend,
        "molar_volume",
        lambda *args, **kwargs: torch.tensor(float("nan")),
    )
    with pytest.raises(InvalidStateError, match="non-finite"):
        backend.select_z(temperature, pressure, composition)
