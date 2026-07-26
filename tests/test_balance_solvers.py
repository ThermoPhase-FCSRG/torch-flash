from __future__ import annotations

import numpy as np
import pytest
import torch

from torch_flash.flash.multiphase import solve_generalized_rachford_rice
from torch_flash.material_balance.rachford_rice import (
    rachford_rice,
    rachford_rice_numpy,
)
from torch_flash.solvers import damped_newton


def test_rachford_rice_known_solution_and_gradient():
    composition = torch.tensor([0.4, 0.6], dtype=torch.float64)
    k_values = torch.tensor([2.0, 0.5], dtype=torch.float64, requires_grad=True)
    result = rachford_rice(composition, k_values)
    torch.testing.assert_close(result.vapor_fraction, torch.tensor(0.2, dtype=torch.float64))
    torch.testing.assert_close(result.liquid_fraction, torch.tensor(0.8, dtype=torch.float64))
    torch.testing.assert_close(
        result.liquid_composition.sum(),
        composition.new_tensor(1.0),
    )
    torch.testing.assert_close(
        result.vapor_composition.sum(),
        composition.new_tensor(1.0),
    )
    result.vapor_fraction.backward()
    assert k_values.grad is not None
    assert torch.isfinite(k_values.grad).all()
    assert bool(result.converged)


def test_rachford_rice_batch_and_bisection_path():
    composition = torch.tensor([[0.4, 0.6], [0.6, 0.4]], dtype=torch.float64)
    k_values = torch.tensor([[2.0, 0.5], [4.0, 0.25]], dtype=torch.float64)
    result = rachford_rice(composition, k_values, max_iterations=100)
    assert result.vapor_fraction.shape == (2,)
    assert bool(result.converged.all())
    assert result.iterations > 0
    with pytest.raises(ValueError, match="strictly positive"):
        rachford_rice(
            torch.tensor([0.4, 0.6]),
            torch.tensor([2, 0]),
        )


@pytest.mark.parametrize(
    ("composition", "k_values", "message"),
    [
        ([0.5, 0.5], [2.0], "same shape"),
        ([0.5, 0.5], [2.0, 0.0], "strictly positive"),
        ([0.5, 0.5], [2.0, float("nan")], "finite"),
        ([0.5, 0.5], [2.0, 3.0], "Kmin"),
        ([0.5, 0.5], [0.2, 0.3], "Kmin"),
    ],
)
def test_rachford_rice_validation(composition, k_values, message):
    with pytest.raises(ValueError, match=message):
        rachford_rice(torch.tensor(composition), torch.tensor(k_values))


def test_numpy_contest_wrapper_signature_and_errors():
    iterations, vapor, liquid, beta_v, beta_l = rachford_rice_numpy(
        np.array([0.4, 0.6]),
        np.array([2.0, 0.5]),
    )
    assert iterations == 1
    np.testing.assert_allclose(vapor.sum(), 1.0, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(liquid.sum(), 1.0, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(beta_v + beta_l, 1.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        rachford_rice_numpy(np.ones((1, 2)), np.ones(2))
    with pytest.raises(ValueError, match="finite root"):
        rachford_rice_numpy(np.array([0.5, 0.5]), np.array([2.0, 3.0]))


def test_generalized_rachford_rice_recovers_three_phase_balance():
    x0 = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    phases = torch.tensor(
        [[0.6, 0.2, 0.2], [0.1, 0.4, 0.5]],
        dtype=torch.float64,
    )
    k_values = phases / x0
    expected_fractions = torch.tensor([0.5, 0.2, 0.3], dtype=torch.float64)
    overall = expected_fractions[0] * x0 + torch.einsum(
        "p,pi->i",
        expected_fractions[1:],
        phases,
    )
    fractions, compositions, _, converged = solve_generalized_rachford_rice(
        overall,
        k_values,
    )
    assert converged
    torch.testing.assert_close(fractions, expected_fractions)
    torch.testing.assert_close(compositions, torch.cat((x0[None], phases)))
    torch.testing.assert_close(
        torch.einsum("p,pi->i", fractions, compositions),
        overall,
    )


@pytest.mark.parametrize(
    ("composition", "k_values", "message"),
    [
        (torch.ones((1, 2)), torch.ones((2, 2)), "composition vector"),
        (torch.ones(2), torch.ones((2, 3)), "component dimension"),
        (torch.ones(2), torch.tensor([[1.0, 0.0]]), "positive"),
    ],
)
def test_generalized_rachford_rice_validation(composition, k_values, message):
    with pytest.raises(ValueError, match=message):
        solve_generalized_rachford_rice(composition, k_values)


def test_generalized_rachford_rice_linear_solve_fallback(monkeypatch):
    monkeypatch.setattr(
        torch.linalg,
        "solve",
        lambda *args, **kwargs: (_ for _ in ()).throw(torch.linalg.LinAlgError("singular")),
    )
    fractions, _, iterations, converged = solve_generalized_rachford_rice(
        torch.tensor([0.4, 0.6], dtype=torch.float64),
        torch.tensor([[2.0, 0.5]], dtype=torch.float64),
        max_iterations=2,
    )
    assert iterations == 2
    assert not converged
    assert torch.isfinite(fractions).all()


def test_damped_newton_solution_bounds_and_failure_paths(monkeypatch):
    result = damped_newton(
        lambda value: value.square() - 4.0,
        torch.tensor([3.0], dtype=torch.float64),
    )
    assert result.converged
    torch.testing.assert_close(result.solution, torch.tensor([2.0], dtype=torch.float64))

    broyden = damped_newton(
        lambda value: value.square() - 4.0,
        torch.tensor([3.0], dtype=torch.float64),
        jacobian_refresh_interval=4,
    )
    assert broyden.converged
    torch.testing.assert_close(
        broyden.solution,
        torch.tensor([2.0], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="refresh interval"):
        damped_newton(
            lambda value: value,
            torch.tensor([1.0], dtype=torch.float64),
            jacobian_refresh_interval=0,
        )

    final_iteration = damped_newton(
        lambda value: value - 2.0,
        torch.tensor([0.0], dtype=torch.float64),
        max_iterations=1,
    )
    assert final_iteration.converged
    torch.testing.assert_close(
        final_iteration.residual,
        final_iteration.solution - 2.0,
    )

    bounded = damped_newton(
        lambda value: value - 5.0,
        torch.tensor([0.0], dtype=torch.float64),
        lower_bound=torch.tensor([-1.0], dtype=torch.float64),
        upper_bound=torch.tensor([2.0], dtype=torch.float64),
        max_iterations=3,
    )
    assert not bounded.converged
    torch.testing.assert_close(bounded.solution, torch.tensor([2.0], dtype=torch.float64))

    original_solve = torch.linalg.solve
    monkeypatch.setattr(
        torch.linalg,
        "solve",
        lambda *args, **kwargs: (_ for _ in ()).throw(torch.linalg.LinAlgError("singular")),
    )
    fallback = damped_newton(
        lambda value: 2.0 * value - 2.0,
        torch.tensor([0.0], dtype=torch.float64),
    )
    monkeypatch.setattr(torch.linalg, "solve", original_solve)
    assert fallback.converged

    monkeypatch.setattr(
        torch.linalg,
        "solve",
        lambda jacobian, value: torch.full_like(value, float("nan")),
    )
    nonfinite = damped_newton(
        lambda value: value - 1.0,
        torch.tensor([0.0], dtype=torch.float64),
        max_iterations=2,
    )
    assert torch.isfinite(nonfinite.solution).all()
