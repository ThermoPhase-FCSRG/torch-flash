from __future__ import annotations

import pytest
import torch

from torch_flash.eos.multiparameter import (
    GaoBTerms,
    HelmholtzTerms,
    IdealHelmholtzTerms,
    MultiparameterEOS,
    MultiparameterMetadata,
    NonAnalyticTerms,
)
from torch_flash.exceptions import ConvergenceError


def _terms(shape, value=0.0):
    return HelmholtzTerms(
        torch.full(shape, value, dtype=torch.float64),
        torch.ones(shape, dtype=torch.float64),
        torch.zeros(shape, dtype=torch.float64),
        torch.zeros(shape, dtype=torch.float64),
    )


def _model(*, trainable=False, residual=0.0):
    pure = _terms((2, 1), residual)
    departure = _terms((2, 2, 1), residual)
    ones = torch.ones((2, 2), dtype=torch.float64)
    return MultiparameterEOS(
        ("methane", "carbon_dioxide"),
        torch.tensor([190.564, 304.1282], dtype=torch.float64),
        torch.tensor([10_139.0, 10_625.0], dtype=torch.float64),
        torch.tensor([0.01604246, 0.0440095], dtype=torch.float64),
        pure,
        departure,
        ones,
        ones,
        ones,
        ones,
        torch.zeros_like(ones),
        MultiparameterMetadata(
            "test-multiparameter",
            "synthetic regression",
            "1",
            ("methane", "carbon_dioxide"),
        ),
        trainable=trainable,
    )


def test_helmholtz_term_shape_validation():
    with pytest.raises(ValueError, match="equal shape"):
        HelmholtzTerms(
            torch.zeros(1),
            torch.zeros(2),
            torch.zeros(1),
            torch.zeros(1),
        )
    with pytest.raises(ValueError, match="Gaussian"):
        HelmholtzTerms(
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            eta=torch.zeros(1),
        )
    with pytest.raises(ValueError, match="linear-density"):
        HelmholtzTerms(
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            linear_density=torch.zeros(1),
        )
    vector = torch.zeros(2)
    table = torch.zeros((2, 1))
    with pytest.raises(ValueError, match="lead arrays"):
        IdealHelmholtzTerms(
            vector,
            vector[:1],
            vector,
            vector,
            table,
            table,
            table,
            table,
            table,
            table,
            table,
            vector,
        )
    with pytest.raises(ValueError, match="term tables"):
        IdealHelmholtzTerms(
            vector,
            vector,
            vector,
            vector,
            table,
            table[:1],
            table,
            table,
            table,
            table,
            table,
            vector,
        )


def test_multiparameter_validation_errors():
    pure = _terms((1, 1))
    departure = _terms((2, 2, 1))
    ones = torch.ones((2, 2), dtype=torch.float64)
    metadata = MultiparameterMetadata("x", "x", "x", ())
    arguments = (
        ("a", "b"),
        torch.ones(2),
        torch.ones(2),
        torch.ones(2),
        pure,
        departure,
        ones,
        ones,
        ones,
        ones,
        ones,
        metadata,
    )
    with pytest.raises(ValueError, match="one row"):
        MultiparameterEOS(*arguments)
    valid_pure = _terms((2, 1))
    bad_departure = _terms((1, 2, 1))
    with pytest.raises(ValueError, match="component, component"):
        MultiparameterEOS(*arguments[:4], valid_pure, bad_departure, *arguments[6:])
    with pytest.raises(ValueError, match="square"):
        MultiparameterEOS(
            *arguments[:4],
            valid_pure,
            departure,
            torch.ones(1),
            *arguments[7:],
        )
    special = torch.zeros((1, 1))
    with pytest.raises(ValueError, match="Gao-B term table"):
        MultiparameterEOS(
            *arguments[:4],
            valid_pure,
            departure,
            *arguments[6:],
            pure_gaob_terms=GaoBTerms(*(special for _ in range(8))),
        )
    with pytest.raises(ValueError, match="non-analytic term table"):
        MultiparameterEOS(
            *arguments[:4],
            valid_pure,
            departure,
            *arguments[6:],
            pure_nonanalytic_terms=NonAnalyticTerms(*(special for _ in range(8))),
        )
    vector = torch.zeros(1)
    table = torch.zeros((1, 1))
    with pytest.raises(ValueError, match="ideal Helmholtz tables"):
        MultiparameterEOS(
            *arguments[:4],
            valid_pure,
            departure,
            *arguments[6:],
            ideal_terms=IdealHelmholtzTerms(
                vector,
                vector,
                vector,
                vector,
                table,
                table,
                table,
                table,
                table,
                table,
                table,
                vector,
            ),
        )


def test_multiparameter_ideal_limit_and_state_methods():
    model = _model()
    temperature = torch.tensor(300.0, dtype=torch.float64)
    pressure = torch.tensor(2.0e6, dtype=torch.float64)
    composition = torch.tensor([0.8, 0.2], dtype=torch.float64)
    reducing_temperature, reducing_density = model.reducing_functions(composition)
    assert reducing_temperature > 0.0
    assert reducing_density > 0.0
    torch.testing.assert_close(
        model.alpha_residual(temperature, torch.tensor(1000.0), composition),
        torch.tensor(0.0, dtype=torch.float64),
    )
    ideal_volume = 8.31446261815324 * temperature / pressure
    torch.testing.assert_close(
        model.pressure(temperature, ideal_volume, composition),
        pressure,
    )
    for phase in ("vapor", "liquid", "stable"):
        volume = model.molar_volume(temperature, pressure, composition, phase)
        torch.testing.assert_close(volume, ideal_volume, rtol=1.0e-10, atol=0.0)
        torch.testing.assert_close(
            model.select_z(temperature, pressure, composition, phase),
            torch.tensor(1.0, dtype=torch.float64),
            rtol=1.0e-10,
            atol=0.0,
        )
        torch.testing.assert_close(
            model.log_fugacity_coefficients(temperature, pressure, composition, phase),
            torch.zeros(2, dtype=torch.float64),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
    with pytest.raises(ValueError, match="unknown phase"):
        model.molar_volume(temperature, pressure, composition, "solid")
    assert model.metadata.model == "test-multiparameter"
    with pytest.raises(RuntimeError, match="no ideal"):
        model.alpha_ideal(temperature, torch.tensor(1000.0), composition)


def test_multiparameter_residual_and_trainable_gradients():
    model = _model(trainable=True, residual=0.01)
    temperature = torch.tensor(300.0, dtype=torch.float64)
    density = torch.tensor(500.0, dtype=torch.float64)
    composition = torch.tensor([0.6, 0.4], dtype=torch.float64)
    alpha = model.alpha_residual(temperature, density, composition)
    assert alpha > 0.0
    alpha.backward()
    assert model.pure_n.grad is not None
    assert model.departure_n.grad is not None


def test_multiparameter_analytic_reduced_density_derivative_matches_autodiff():
    model = _model(residual=0.01)
    temperature = torch.tensor(280.0, dtype=torch.float64)
    density = torch.tensor(4_000.0, dtype=torch.float64)
    composition = torch.tensor([0.35, 0.65], dtype=torch.float64)
    _, reducing_density = model.reducing_functions(composition)
    autodiff = (
        torch.func.grad(
            lambda current_density: model.alpha_residual(
                temperature,
                current_density,
                composition,
            )
        )(density)
        * reducing_density
    )
    analytic = model.alpha_residual_delta(temperature, density, composition)
    torch.testing.assert_close(analytic, autodiff, rtol=2.0e-13, atol=1.0e-15)


def test_multiparameter_gaussian_terms():
    shape = (2, 1)
    pure = HelmholtzTerms(
        torch.ones(shape, dtype=torch.float64),
        torch.ones(shape, dtype=torch.float64),
        torch.zeros(shape, dtype=torch.float64),
        torch.zeros(shape, dtype=torch.float64),
        eta=torch.ones(shape, dtype=torch.float64),
        epsilon=torch.full(shape, 0.5, dtype=torch.float64),
        beta=torch.ones(shape, dtype=torch.float64),
        gamma=torch.ones(shape, dtype=torch.float64),
    )
    departure = _terms((2, 2, 1))
    ones = torch.ones((2, 2), dtype=torch.float64)
    model = MultiparameterEOS(
        ("a", "b"),
        torch.tensor([200.0, 300.0], dtype=torch.float64),
        torch.tensor([10_000.0, 12_000.0], dtype=torch.float64),
        torch.tensor([0.02, 0.03], dtype=torch.float64),
        pure,
        departure,
        ones,
        ones,
        ones,
        ones,
        torch.zeros_like(ones),
        MultiparameterMetadata("gaussian", "synthetic", "1", ("a", "b")),
    )
    value = model.alpha_residual(
        torch.tensor(250.0, dtype=torch.float64),
        torch.tensor(5000.0, dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    assert value > 0.0


def test_multiparameter_density_failure(monkeypatch):
    model = _model()
    monkeypatch.setattr(
        model,
        "pressure",
        lambda temperature, volume, composition: temperature.new_tensor(0.0) + 0.0 * volume,
    )
    with pytest.raises(ConvergenceError, match="did not converge"):
        model.molar_volume(
            torch.tensor(300.0, dtype=torch.float64),
            torch.tensor(1.0e5, dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
        )


def test_multiparameter_batched_stable_liquid_and_scalar_fallbacks(monkeypatch):
    model = _model()
    temperatures = torch.tensor([300.0, 320.0], dtype=torch.float64)
    pressures = torch.tensor([1.0e5, 2.0e5], dtype=torch.float64)
    compositions = torch.tensor([[0.8, 0.2], [0.4, 0.6]], dtype=torch.float64)

    stable = model.molar_volume(temperatures, pressures, compositions, "stable")
    expected = model.gas_constant * temperatures / pressures
    torch.testing.assert_close(stable, expected, rtol=1.0e-10, atol=0.0)
    liquid = model.molar_volume(temperatures, pressures, compositions, "liquid")
    torch.testing.assert_close(liquid, expected, rtol=1.0e-10, atol=0.0)

    with pytest.raises(ValueError, match="wrong number"):
        model.molar_volume(
            torch.tensor(300.0),
            torch.tensor(1.0e5),
            torch.ones(3),
        )

    def subcritical_reducing(composition):
        shape = composition.shape[:-1]
        return (
            torch.full(shape, 1000.0, dtype=composition.dtype),
            torch.full(shape, 10_000.0, dtype=composition.dtype),
        )

    monkeypatch.setattr(model, "reducing_functions", subcritical_reducing)
    fallback_liquid = model.molar_volume(
        temperatures,
        pressures,
        compositions,
        "liquid",
    )
    torch.testing.assert_close(fallback_liquid, expected, rtol=1.0e-10, atol=0.0)

    original_molar_volume = model.molar_volume
    scalar_fallbacks = []

    def one_bad_batched_state(temperature, volume, composition):
        ideal_pressure = model.gas_constant * temperature / volume
        if volume.ndim == 0:
            return ideal_pressure
        return torch.where(
            torch.broadcast_to(temperature, volume.shape) > 310.0,
            torch.full_like(volume, torch.nan),
            ideal_pressure,
        )

    def track_scalar_fallbacks(temperature, pressure, composition, phase="stable"):
        if temperature.ndim == 0:
            scalar_fallbacks.append(float(temperature))
        return original_molar_volume(temperature, pressure, composition, phase)

    monkeypatch.setattr(model, "pressure", one_bad_batched_state)
    monkeypatch.setattr(model, "molar_volume", track_scalar_fallbacks)
    partial_fallback = original_molar_volume(
        temperatures,
        pressures,
        compositions,
        "vapor",
    )
    torch.testing.assert_close(partial_fallback, expected, rtol=1.0e-10, atol=0.0)
    assert scalar_fallbacks == [320.0]


def test_multiparameter_batched_stable_volume_falls_back_only_failed_states(monkeypatch):
    model = _model()
    temperatures = torch.tensor([300.0, 320.0], dtype=torch.float64)
    pressures = torch.tensor([1.0e5, 2.0e5], dtype=torch.float64)
    compositions = torch.tensor([[0.8, 0.2], [0.4, 0.6]], dtype=torch.float64)
    expected = model.gas_constant * temperatures / pressures

    def partial_phase_volume(
        temperature,
        pressure,
        composition,
        phase,
        *,
        return_convergence=False,
    ):
        del composition, phase
        volumes = model.gas_constant * temperature / pressure
        converged = torch.tensor([True, False], device=temperature.device)
        return (volumes, converged) if return_convergence else volumes

    monkeypatch.setattr(model, "_batched_phase_volume", partial_phase_volume)
    stable = model._batched_stable_volume(temperatures, pressures, compositions)
    torch.testing.assert_close(stable, expected, rtol=1.0e-10, atol=0.0)


def test_multiparameter_nonfinite_scalar_newton_step_falls_back(monkeypatch):
    model = _model()
    monkeypatch.setattr(
        model,
        "pressure",
        lambda temperature, volume, composition: torch.full_like(volume, torch.nan),
    )
    with pytest.raises(ConvergenceError, match="did not converge"):
        model.molar_volume(
            torch.tensor(300.0, dtype=torch.float64),
            torch.tensor(1.0e5, dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            "liquid",
        )
