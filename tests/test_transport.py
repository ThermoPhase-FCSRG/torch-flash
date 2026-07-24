from __future__ import annotations

import pytest
import torch

from torch_flash import (
    component_set,
    lbc_pseudocomponent_critical_volume,
    lbc_viscosity,
)
from torch_flash.exceptions import InvalidStateError
from torch_flash.transport.viscosity import (
    methane_bwr_density,
    methane_bwr_pressure,
    methane_viscosity,
    pedersen_viscosity,
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
    t = torch.tensor(temperature, dtype=torch.float64)
    p = torch.tensor(pressure, dtype=torch.float64)
    density = methane_bwr_density(t, p)
    torch.testing.assert_close(
        methane_bwr_pressure(t, density),
        p / 101_325.0,
        rtol=2.0e-11,
        atol=1.0e-10,
    )
    viscosity = methane_viscosity(t, density)
    assert float(viscosity) == pytest.approx(expected, rel=2.0e-12)
    viscosity.backward() if viscosity.requires_grad else None


def test_methane_density_liquid_branch_and_autodiff():
    temperature = torch.tensor(150.0, dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor(5.0e6, dtype=torch.float64)
    density = methane_bwr_density(temperature, pressure, phase="liquid")
    assert density > 10.0
    viscosity = methane_viscosity(temperature, density)
    gradient = torch.autograd.grad(viscosity, temperature)[0]
    assert torch.isfinite(gradient)


def test_pedersen_pure_methane_identity_and_mixture():
    temperature = torch.tensor(300.0, dtype=torch.float64)
    pressure = torch.tensor(1.0e7, dtype=torch.float64)
    methane = component_set(("methane",))
    mixture_value = pedersen_viscosity(
        temperature,
        pressure,
        torch.tensor([1.0], dtype=torch.float64),
        methane,
    )
    direct = methane_viscosity(
        temperature,
        methane_bwr_density(temperature, pressure),
    )
    torch.testing.assert_close(mixture_value, direct, rtol=5.0e-14, atol=0.0)

    components = component_set(("methane", "n_decane"))
    heavy = pedersen_viscosity(
        temperature,
        pressure,
        torch.tensor([0.7, 0.3], dtype=torch.float64),
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
    with pytest.raises(ValueError, match="composition vector"):
        pedersen_viscosity(
            torch.tensor(300.0),
            torch.tensor(1.0e5),
            torch.tensor([[1.0]]),
            component_set(("methane",)),
        )
    with pytest.raises(ValueError, match="sizes"):
        pedersen_viscosity(
            torch.tensor(300.0),
            torch.tensor(1.0e5),
            torch.tensor([0.5, 0.5]),
            component_set(("methane",)),
        )
    monkeypatch.setattr(
        "torch_flash.transport.viscosity.methane_bwr_pressure",
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
        dtype=torch.float64,
    )
    conversion = 0.028316846592 / 453.59237
    critical_volume = (
        torch.tensor(
            [1.590, 2.370, 3.250, 4.208, 4.080, 4.899, 4.870, 5.929, 7.882],
            dtype=torch.float64,
        )
        * conversion
    )
    density = torch.tensor(0.627 / (1.752 * conversion), dtype=torch.float64)
    coefficients = torch.tensor(
        [0.10230, 0.023364, 0.058533, -0.040758, 0.0093324],
        dtype=torch.float64,
        requires_grad=True,
    )
    viscosity = lbc_viscosity(
        torch.tensor(620.0 / 1.8, dtype=torch.float64),
        density,
        composition,
        component_set(names),
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
        torch.tensor(0.114, dtype=torch.float64),
        torch.tensor(780.0, dtype=torch.float64),
    )
    assert float(volume / conversion) == pytest.approx(8.0043138, rel=2.0e-7)
    with pytest.raises(ValueError, match="finite and positive"):
        lbc_pseudocomponent_critical_volume(torch.tensor(-1.0), torch.tensor(800.0))


def test_lbc_validation_paths():
    methane = component_set(("methane",))
    temperature = torch.tensor(300.0)
    density = torch.tensor(10.0)
    with pytest.raises(ValueError, match="composition vector"):
        lbc_viscosity(temperature, density, torch.tensor([[1.0]]), methane)
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
