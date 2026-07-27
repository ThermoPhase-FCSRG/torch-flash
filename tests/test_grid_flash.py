from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import torch_flash.flash.multiphase as multiphase_module
import torch_flash.properties.phase_identification as phase_identification_module
from torch_flash import (
    ChemicalState,
    ComponentSet,
    GridFlashOptions,
    PhaseIdentificationCriterion,
    flash_grid,
    flash_grid_oracle,
    identify_grid_phases,
    identify_phase,
    multiphase_trust_region_flash,
    peng_robinson_1978,
    refine_flash_grid_phase_boundaries,
    soave_redlich_kwong,
    solve_batched_binary_three_phase_invariants,
    solve_binary_three_phase_invariant,
)
from torch_flash.constants import R
from torch_flash.exceptions import ConvergenceWarning, ExperimentalModelWarning
from torch_flash.flash import grid as grid_module


def _binary_model():
    components = ComponentSet(
        ("methane", "carbon_dioxide"),
        torch.tensor([190.6, 304.2], dtype=torch.float64),
        101325.0 * torch.tensor([45.4, 72.9], dtype=torch.float64),
        torch.tensor([0.008, 0.228], dtype=torch.float64),
        torch.tensor([0.01604, 0.04401], dtype=torch.float64),
        torch.tensor([9.93e-5, 9.40e-5], dtype=torch.float64),
    )
    interaction = torch.tensor(
        [[0.0, 0.12], [0.12, 0.0]],
        dtype=torch.float64,
    )
    return soave_redlich_kwong(components, kij=interaction)


def _north_ward_estes_model():
    critical_temperature = torch.tensor(
        [304.2, 190.6, 343.64, 466.41, 603.07, 733.79, 923.2],
        dtype=torch.float64,
    )
    critical_pressure = 1.0e5 * torch.tensor(
        [73.77, 46.0, 45.05, 33.51, 24.24, 18.03, 17.26],
        dtype=torch.float64,
    )
    components = ComponentSet(
        ("carbon_dioxide", "methane", "pc1", "pc2", "pc3", "pc4", "pc5"),
        critical_temperature,
        critical_pressure,
        torch.tensor(
            [0.225, 0.008, 0.13, 0.244, 0.6, 0.903, 1.229],
            dtype=torch.float64,
        ),
        1.0e-3
        * torch.tensor(
            [44.01, 16.04, 38.4, 72.82, 135.82, 257.75, 479.95],
            dtype=torch.float64,
        ),
        0.3074 * R * critical_temperature / critical_pressure,
    )
    interaction = torch.zeros((7, 7), dtype=torch.float64)
    interaction[0, 1:] = torch.tensor(
        [0.12, 0.12, 0.12, 0.09, 0.09, 0.09],
        dtype=torch.float64,
    )
    interaction[:, 0] = interaction[0]
    return peng_robinson_1978(components, kij=interaction)


def _north_ward_estes_state() -> ChemicalState:
    oil = torch.tensor(
        [0.0077, 0.2025, 0.1180, 0.1484, 0.2863, 0.1490, 0.0881],
        dtype=torch.float64,
    )
    injection_gas = torch.tensor(
        [0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    injection_fraction = (0.80866 - oil[0]) / (injection_gas[0] - oil[0])
    feed = (1.0 - injection_fraction) * oil + injection_fraction * injection_gas
    return ChemicalState(
        torch.tensor([301.48], dtype=torch.float64),
        torch.tensor([79.0e5], dtype=torch.float64),
        feed,
    )


@pytest.mark.serial
def test_grid_flash_recovers_published_north_ward_estes_three_phase_state():
    model = _north_ward_estes_model()
    state = _north_ward_estes_state()
    result = flash_grid(model, state)
    oracle = flash_grid_oracle(model, state)

    assert result.grid_shape == (1,)
    assert result.phase_counts.tolist() == [3]
    assert oracle.phase_counts.tolist() == [3]
    assert result.converged.tolist() == [True]
    assert oracle.converged.tolist() == [True]
    assert result.fugacity_residual[0] <= 1.0e-8
    assert result.material_balance_residual[0] <= 5.0e-11
    torch.testing.assert_close(
        result.phase_fractions[0].sum(),
        torch.tensor(1.0, dtype=torch.float64),
    )
    reconstructed = torch.sum(
        result.phase_fractions[0, :, None] * result.phase_compositions[0].nan_to_num(),
        dim=0,
    )
    torch.testing.assert_close(reconstructed, result.feeds[0], atol=5.0e-11, rtol=0.0)
    torch.testing.assert_close(
        oracle.gibbs_reduction,
        result.gibbs_reduction,
        atol=2.0e-7,
        rtol=0.0,
    )

    identification = identify_grid_phases(
        model,
        result,
        pip_autodiff_chunk_size=1,
        response_autodiff_chunk_size=1,
    )
    assert identification.region_codes.shape == (6, 1)
    assert identification.region_codes[:, 0].tolist() == [4, 4, 4, 4, 4, 4]
    assert torch.all(identification.phase_identity_codes[:, 0, :3] >= 0)
    assert torch.isfinite(identification.criterion_values[:, 0, :3]).all()
    assert torch.equal(
        identification.phase_identity_codes[3],
        identification.phase_identity_codes[5],
    )
    for method_index in range(6):
        method = identification.methods[method_index]
        for phase_index in range(3):
            scalar = identify_phase(
                model,
                ChemicalState(
                    result.temperatures[0],
                    result.pressures[0],
                    result.phase_compositions[0, phase_index],
                ),
                method=method,
            )
            assert scalar.criterion_value is not None
            torch.testing.assert_close(
                identification.criterion_values[method_index, 0, phase_index],
                scalar.criterion_value,
                rtol=2.0e-12,
                atol=1.0e-18,
            )
    assert len(identification.method_elapsed_seconds) == len(identification.methods)
    assert all(seconds >= 0.0 for seconds in identification.method_elapsed_seconds)
    assert sum(identification.method_elapsed_seconds) <= identification.elapsed_seconds


@pytest.mark.serial
def test_grid_continuation_recovers_north_ward_estes_llv_boundary_state():
    model = _north_ward_estes_model()
    oil = torch.tensor(
        [0.0077, 0.2025, 0.1180, 0.1484, 0.2863, 0.1490, 0.0881],
        dtype=torch.float64,
    )
    injection_gas = torch.tensor(
        [0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    injected_fraction = torch.tensor([[0.733, 0.775]], dtype=torch.float64)
    feeds = (1.0 - injected_fraction[..., None]) * oil + injected_fraction[
        ..., None
    ] * injection_gas
    state = ChemicalState(
        torch.full_like(injected_fraction, 301.48),
        torch.full_like(injected_fraction, 81.56312625250501e5),
        feeds,
    )

    result = flash_grid(model, state)
    oracle = flash_grid_oracle(
        model,
        ChemicalState(
            state.temperature[:, :1].reshape(1),
            state.pressure[:, :1].reshape(1),
            state.composition[:, :1].reshape(1, 7),
        ),
    )

    assert result.phase_counts.tolist() == [3, 3]
    assert result.topology_audit_count == 1
    assert result.topology_audit_replacements == 1
    assert result.phase_fractions[0].amin() > 1.0e-4
    assert result.fugacity_residual[0] <= 1.0e-8
    assert result.material_balance_residual[0] <= 5.0e-11
    torch.testing.assert_close(
        result.gibbs_reduction[0],
        oracle.gibbs_reduction[0],
        atol=2.0e-7,
        rtol=0.0,
    )


@pytest.mark.serial
def test_binary_invariant_autodiff_newton_and_grid_lever_rule():
    model = _binary_model()
    invariant = solve_binary_three_phase_invariant(
        model,
        torch.tensor(180.0, dtype=torch.float64),
        torch.tensor(2.73e6, dtype=torch.float64),
        torch.tensor(
            [[0.199, 0.801], [0.781, 0.219], [0.958, 0.042]],
            dtype=torch.float64,
        ),
    )

    assert invariant.converged
    assert invariant.iterations <= 12
    assert invariant.residual_norm <= 1.0e-11
    assert float(invariant.pressure / 1.0e5) == pytest.approx(27.37022, rel=2.0e-6)
    torch.testing.assert_close(
        invariant.phase_compositions[:, 0],
        torch.tensor([0.1949229, 0.7849824, 0.9584729], dtype=torch.float64),
        atol=2.0e-7,
        rtol=0.0,
    )

    batched = solve_batched_binary_three_phase_invariants(
        model,
        torch.tensor([180.0, 180.0], dtype=torch.float64),
        torch.tensor([2.73e6, 2.73e6], dtype=torch.float64),
        torch.stack(
            (
                torch.tensor(
                    [[0.199, 0.801], [0.781, 0.219], [0.958, 0.042]],
                    dtype=torch.float64,
                ),
                torch.tensor(
                    [[0.200, 0.800], [0.780, 0.220], [0.960, 0.040]],
                    dtype=torch.float64,
                ),
            )
        ),
    )
    assert batched.converged.tolist() == [True, True]
    torch.testing.assert_close(
        batched.pressure,
        invariant.pressure.expand(2),
        rtol=2.0e-10,
        atol=0.0,
    )
    torch.testing.assert_close(
        batched.phase_compositions[:, :, 0],
        invariant.phase_compositions[:, 0].expand(2, -1),
        atol=2.0e-10,
        rtol=0.0,
    )

    trust_invariant = solve_binary_three_phase_invariant(
        model,
        invariant.temperature,
        invariant.pressure,
        invariant.phase_compositions,
        method="trust-region",
        max_iterations=100,
    )
    trust_batch = solve_batched_binary_three_phase_invariants(
        model,
        invariant.temperature.expand(2),
        invariant.pressure.expand(2),
        invariant.phase_compositions.expand(2, -1, -1),
        method="trust-region",
        max_iterations=100,
    )
    assert trust_invariant.converged
    assert trust_batch.converged.tolist() == [True, True]
    torch.testing.assert_close(
        trust_invariant.pressure,
        invariant.pressure,
        rtol=2.0e-10,
        atol=0.0,
    )
    torch.testing.assert_close(
        trust_batch.pressure,
        invariant.pressure.expand(2),
        rtol=2.0e-10,
        atol=0.0,
    )

    methane_fraction = torch.tensor([0.3, 0.6, 0.9], dtype=torch.float64)
    state = ChemicalState(
        torch.full((1, 3), 180.0, dtype=torch.float64),
        invariant.pressure.expand(1, 3),
        torch.stack((methane_fraction, 1.0 - methane_fraction), dim=-1).reshape(1, 3, 2),
    )
    result = flash_grid(model, state, binary_invariants=(invariant,))
    oracle = flash_grid_oracle(model, state, binary_invariants=(invariant,))

    assert result.grid_shape == (1, 3)
    assert result.phase_counts.tolist() == [3, 3, 3]
    assert oracle.phase_counts.tolist() == [3, 3, 3]
    assert result.converged.all()
    assert oracle.converged.all()
    assert result.material_balance_residual.max() <= 5.0e-11

    homogeneous = flash_grid_oracle(
        model,
        ChemicalState(
            torch.tensor([500.0], dtype=torch.float64),
            torch.tensor([1.0e5], dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
        ),
    )
    assert homogeneous.phase_counts.tolist() == [1]


@pytest.mark.serial
def test_multiphase_trust_region_recovers_binary_three_phase_invariant():
    model = _binary_model()
    invariant = solve_binary_three_phase_invariant(
        model,
        torch.tensor(180.0, dtype=torch.float64),
        torch.tensor(2.73e6, dtype=torch.float64),
        torch.tensor(
            [[0.199, 0.801], [0.781, 0.219], [0.958, 0.042]],
            dtype=torch.float64,
        ),
    )
    expected_fractions = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    phase_order = torch.tensor([0, 2, 1])
    expected_compositions = invariant.phase_compositions[phase_order]
    composition = torch.einsum(
        "p,pi->i",
        expected_fractions,
        expected_compositions,
    )
    initial_compositions = torch.tensor(
        [[0.20, 0.80], [0.95, 0.05], [0.78, 0.22]],
        dtype=torch.float64,
    )
    initial_k_values = initial_compositions[1:] / initial_compositions[0]

    with pytest.warns(ExperimentalModelWarning):
        result = multiphase_trust_region_flash(
            model,
            ChemicalState(invariant.temperature, invariant.pressure, composition),
            initial_k_values=initial_k_values,
        )

    assert result.converged
    assert result.nphases == 3
    assert result.residual_norm <= 1.0e-8
    assert result.diagnostics["material_balance_residual"] <= 2.0e-15
    for actual, expected in zip(
        result.phases,
        expected_compositions,
        strict=True,
    ):
        torch.testing.assert_close(
            actual.composition,
            expected,
            rtol=2.0e-8,
            atol=2.0e-10,
        )

    scalar_state = ChemicalState(
        invariant.temperature,
        invariant.pressure,
        composition,
    )
    with pytest.raises(ValueError, match="composition vector"):
        multiphase_trust_region_flash(
            model,
            ChemicalState(
                invariant.temperature,
                invariant.pressure,
                composition[None],
            ),
            initial_k_values=initial_k_values,
        )
    with pytest.raises(ValueError, match="controls"):
        multiphase_trust_region_flash(
            model,
            scalar_state,
            initial_k_values=initial_k_values,
            tolerance=0.0,
        )
    with pytest.raises(ValueError, match="K values"):
        multiphase_trust_region_flash(
            model,
            scalar_state,
            initial_k_values=initial_k_values[:, :1],
        )
    with pytest.raises(ValueError, match="roots"):
        multiphase_trust_region_flash(
            model,
            scalar_state,
            initial_k_values=initial_k_values,
            phase_roots=("liquid", "vapor"),
        )
    with pytest.raises(RuntimeError, match="did not produce"):
        multiphase_trust_region_flash(
            model,
            scalar_state,
            initial_k_values=initial_k_values,
            max_iterations=1,
            raise_on_failure=True,
        )


def test_multiphase_trust_region_validates_balance_and_warns(
    monkeypatch,
):
    model = _binary_model()
    state = ChemicalState(
        torch.tensor(180.0, dtype=torch.float64),
        torch.tensor(2.73e6, dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    initial_k_values = torch.tensor(
        [[0.95 / 0.20, 0.05 / 0.80], [0.78 / 0.20, 0.22 / 0.80]],
        dtype=torch.float64,
    )
    with (
        pytest.warns(ExperimentalModelWarning),
        pytest.raises(
            ValueError,
            match="strictly positive",
        ),
    ):
        multiphase_trust_region_flash(
            model,
            ChemicalState(
                state.temperature,
                state.pressure,
                torch.tensor([1.0, 0.0], dtype=torch.float64),
            ),
            initial_k_values=initial_k_values,
        )

    monkeypatch.setattr(
        multiphase_module,
        "solve_generalized_rachford_rice",
        lambda z, k: (
            z.new_tensor([0.3, 0.3, 0.4]),
            z.new_tensor([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]]),
            100,
            False,
        ),
    )
    with (
        pytest.warns(ExperimentalModelWarning),
        pytest.raises(
            ValueError,
            match="material-balance",
        ),
    ):
        multiphase_trust_region_flash(
            model,
            state,
            initial_k_values=initial_k_values,
        )

    monkeypatch.undo()
    with pytest.warns((ExperimentalModelWarning, ConvergenceWarning)):
        result = multiphase_trust_region_flash(
            model,
            state,
            initial_k_values=initial_k_values,
            max_iterations=1,
        )
    assert not result.converged


def test_batched_binary_invariant_validates_batch_contract():
    model = _binary_model()
    temperature = torch.tensor([180.0], dtype=torch.float64)
    pressure = torch.tensor([2.73e6], dtype=torch.float64)
    compositions = torch.tensor(
        [[[0.199, 0.801], [0.781, 0.219], [0.958, 0.042]]],
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="shape"):
        solve_batched_binary_three_phase_invariants(
            model,
            temperature[0],
            pressure,
            compositions,
        )
    with pytest.raises(ValueError, match="guesses"):
        solve_batched_binary_three_phase_invariants(
            model,
            temperature,
            pressure,
            compositions[:, :2],
        )
    with pytest.raises(ValueError, match="controls"):
        solve_batched_binary_three_phase_invariants(
            model,
            temperature,
            pressure,
            compositions,
            tolerance=0.0,
        )
    with pytest.raises(ValueError, match="roots"):
        solve_batched_binary_three_phase_invariants(
            model,
            temperature,
            pressure,
            compositions,
            phase_roots=("liquid", "liquid", "invalid"),
        )
    with pytest.raises(ValueError, match="finite and interior"):
        solve_batched_binary_three_phase_invariants(
            model,
            temperature,
            torch.tensor([torch.nan], dtype=torch.float64),
            compositions,
        )
    with pytest.raises(ValueError, match="method"):
        solve_batched_binary_three_phase_invariants(
            model,
            temperature,
            pressure,
            compositions,
            method="invalid",
        )


def test_pure_grid_oracle_preserves_dtype_device_and_shape():
    components = ComponentSet(
        ("methane",),
        torch.tensor([190.6], dtype=torch.float64),
        torch.tensor([4.6e6], dtype=torch.float64),
        torch.tensor([0.008], dtype=torch.float64),
        torch.tensor([0.01604], dtype=torch.float64),
        torch.tensor([9.93e-5], dtype=torch.float64),
    )
    model = peng_robinson_1978(components)
    state = ChemicalState(
        torch.tensor([[150.0, 250.0]], dtype=torch.float64),
        torch.tensor([[1.0e5, 5.0e6]], dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
    )

    result = flash_grid_oracle(model, state)
    fast = flash_grid(model, state)

    assert result.grid_shape == (1, 2)
    assert result.phase_fractions.dtype == torch.float64
    assert result.phase_fractions.device == state.temperature.device
    assert result.phase_counts.tolist() == [1, 1]
    assert torch.equal(result.phase_counts, fast.phase_counts)
    torch.testing.assert_close(
        result.phase_compositions[:, 0],
        torch.ones((2, 1), dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "options",
    [
        GridFlashOptions(chunk_size=1),
        GridFlashOptions(random_allocation_starts=0),
        GridFlashOptions(fallback_workers=2),
    ],
)
def test_grid_flash_options_accept_valid_controls(options):
    assert options.chunk_size > 0
    assert options.random_allocation_starts >= 0
    assert options.fallback_workers > 0


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("chunk_size", 0),
        ("fallback_workers", 0),
        ("random_allocation_starts", -1),
        ("fugacity_tolerance", 0.0),
    ],
)
def test_grid_flash_options_reject_invalid_controls(keyword, value):
    with pytest.raises(ValueError):
        GridFlashOptions(**{keyword: value})


def test_grid_flash_and_identification_validate_batch_inputs():
    model = _binary_model()
    with pytest.raises(ValueError, match="non-scalar"):
        flash_grid(
            model,
            ChemicalState(
                torch.tensor(180.0, dtype=torch.float64),
                torch.tensor(2.0e6, dtype=torch.float64),
                torch.tensor([0.5, 0.5], dtype=torch.float64),
            ),
        )
    with pytest.raises(ValueError, match="equal, non-scalar"):
        flash_grid(
            model,
            ChemicalState(
                torch.ones(2, dtype=torch.float64) * 180.0,
                torch.ones(3, dtype=torch.float64) * 2.0e6,
                torch.tensor([0.5, 0.5], dtype=torch.float64),
            ),
        )
    with pytest.raises(ValueError, match="composition batch"):
        flash_grid(
            model,
            ChemicalState(
                torch.ones(2, dtype=torch.float64) * 180.0,
                torch.ones(2, dtype=torch.float64) * 2.0e6,
                torch.full((3, 2), 0.5, dtype=torch.float64),
            ),
        )

    equilibrium = flash_grid(
        model,
        ChemicalState(
            torch.tensor([180.0], dtype=torch.float64),
            torch.tensor([2.0e6], dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
        ),
    )
    with pytest.raises(ValueError, match="at least one"):
        identify_grid_phases(model, equilibrium, methods=())
    with pytest.raises(ValueError, match="unique"):
        identify_grid_phases(
            model,
            equilibrium,
            methods=(
                "pedersen-volume-to-covolume",
                "pedersen-volume-to-covolume",
            ),
        )
    with pytest.raises(ValueError, match="chunk"):
        identify_grid_phases(
            model,
            equilibrium,
            pip_autodiff_chunk_size=0,
        )
    with pytest.raises(ValueError, match="chunk"):
        identify_grid_phases(
            model,
            equilibrium,
            response_autodiff_chunk_size=0,
        )
    for options, message in (
        ({"volume_to_covolume_threshold": 0.0}, "threshold"),
        ({"pseudo_critical_temperature_factor": 0.0}, "factor"),
        ({"ambiguity_relative_tolerance": -1.0}, "ambiguity"),
        ({"pip_denominator_relative_tolerance": -1.0}, "PIP"),
    ):
        with pytest.raises(ValueError, match=message):
            identify_grid_phases(
                model,
                equilibrium,
                **options,
            )
    pip_method: tuple[PhaseIdentificationCriterion, ...] = (
        "venkatarathnam-oellrich-phase-identification-parameter",
    )
    singular = identify_grid_phases(
        model,
        equilibrium,
        methods=pip_method,
        pip_denominator_relative_tolerance=1.0e6,
    )
    assert singular.region_codes.tolist() == [[5]]

    nonconverged = replace(
        equilibrium,
        converged=torch.zeros_like(equilibrium.converged),
    )
    for nonconverged_method in (
        "pedersen-volume-to-covolume",
        "pasad-isothermal-compressibility-derivative",
        "venkatarathnam-oellrich-phase-identification-parameter",
    ):
        unavailable = identify_grid_phases(
            model,
            nonconverged,
            methods=(nonconverged_method,),
        )
        assert unavailable.region_codes.tolist() == [[5]]
    for unavailable_method in (
        "pasad-isothermal-compressibility-derivative",
        "bennett-thermal-expansion-derivative",
        "venkatarathnam-oellrich-phase-identification-parameter",
    ):
        unavailable = identify_grid_phases(
            object(),  # type: ignore[arg-type]
            equilibrium,
            methods=(unavailable_method,),
        )
        assert unavailable.region_codes.tolist() == [[5]]


def test_grid_phase_identification_falls_back_after_batched_physical_failure(monkeypatch):
    model = _binary_model()
    equilibrium = flash_grid(
        model,
        ChemicalState(
            torch.tensor([180.0], dtype=torch.float64),
            torch.tensor([2.0e6], dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
        ),
    )

    original_ratio = phase_identification_module.volume_to_covolume_ratio

    def scalar_only_ratio(model, state, phase="stable"):
        if state.temperature.ndim > 0:
            raise ValueError("forced batched physical failure")
        return original_ratio(model, state, phase)

    monkeypatch.setattr(
        phase_identification_module,
        "volume_to_covolume_ratio",
        scalar_only_ratio,
    )
    pedersen = identify_grid_phases(
        model,
        equilibrium,
        methods=("pedersen-volume-to-covolume",),
    )
    assert pedersen.region_codes.tolist() != [[5]]

    monkeypatch.setattr(
        phase_identification_module,
        "volume_to_covolume_ratio",
        original_ratio,
    )
    original_response = phase_identification_module._phase_response_details

    def scalar_only_response(model, state, phase="stable", *, molar_volume=None):
        if state.temperature.ndim > 0:
            raise ValueError("forced batched physical failure")
        return original_response(
            model,
            state,
            phase,
            molar_volume=molar_volume,
        )

    monkeypatch.setattr(
        phase_identification_module,
        "_phase_response_details",
        scalar_only_response,
    )
    response = identify_grid_phases(
        model,
        equilibrium,
        methods=("pasad-isothermal-compressibility-derivative",),
    )
    assert response.region_codes.tolist() != [[5]]


def test_flash_grid_phase_boundary_refinement_localizes_and_connects_curves(monkeypatch):
    model = _binary_model()
    temperature = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    pressure = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    region_codes = torch.tensor(
        [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
        dtype=torch.int8,
    )
    converged = torch.ones_like(region_codes, dtype=torch.bool)

    def fake_flash(_model, state, *, options=None):
        del options
        state_temperature, state_pressure = torch.broadcast_tensors(
            state.temperature,
            state.pressure,
        )
        return SimpleNamespace(
            temperatures=state_temperature.reshape(-1),
            pressures=state_pressure.reshape(-1),
            converged=torch.ones(state_temperature.numel(), dtype=torch.bool),
            grid_shape=state_temperature.shape,
        )

    def fake_identification(_model, equilibrium, *, methods):
        assert methods == ("pedersen-volume-to-covolume",)
        inside = (
            (equilibrium.temperatures > 1.5)
            & (equilibrium.temperatures < 2.5)
            & (equilibrium.pressures > 1.5)
            & (equilibrium.pressures < 2.5)
        )
        codes = torch.where(
            inside,
            torch.tensor(2, dtype=torch.int8),
            torch.tensor(0, dtype=torch.int8),
        )
        return SimpleNamespace(region_codes=codes.unsqueeze(0))

    monkeypatch.setattr(grid_module, "flash_grid", fake_flash)
    monkeypatch.setattr(grid_module, "identify_grid_phases", fake_identification)
    boundaries = refine_flash_grid_phase_boundaries(
        model,
        temperature,
        pressure,
        composition,
        region_codes,
        converged,
        refinement_iterations=3,
    )

    assert boundaries.refined_edge_count == 4
    assert boundaries.refinement_state_count == 12
    assert boundaries.failed_midpoint_count == 0
    assert boundaries.ambiguous_cell_count == 0
    assert boundaries.maximum_temperature_bracket == pytest.approx(0.125)
    assert boundaries.maximum_pressure_bracket == pytest.approx(0.125)
    assert len(boundaries.curves) == 1
    curve = boundaries.curves[0]
    assert curve.region_code == 2
    assert curve.closed
    assert curve.temperature.shape == curve.pressure.shape == (5,)
    torch.testing.assert_close(curve.temperature[0], curve.temperature[-1])
    torch.testing.assert_close(curve.pressure[0], curve.pressure[-1])

    empty = refine_flash_grid_phase_boundaries(
        model,
        temperature,
        pressure,
        composition,
        torch.zeros_like(region_codes),
        converged,
    )
    assert empty.curves == ()
    assert empty.refined_edge_count == 0

    ambiguous = refine_flash_grid_phase_boundaries(
        model,
        temperature[:2],
        pressure[:2],
        composition,
        torch.tensor([[0, 2], [2, 0]], dtype=torch.int8),
        torch.ones((2, 2), dtype=torch.bool),
        region_labels=("V", "LV"),
        refinement_iterations=1,
    )
    assert ambiguous.ambiguous_cell_count == 2


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"temperature_axis": torch.tensor([1.0])}, "axes"),
        ({"pressure_axis": torch.tensor([2.0, 1.0])}, "axes"),
        ({"region_codes": torch.zeros((2, 2), dtype=torch.bool)}, "dtypes"),
        ({"converged": torch.ones((2, 3), dtype=torch.bool)}, "classifications"),
        ({"composition": torch.tensor([1.0, 0.0])}, "composition"),
        ({"refinement_iterations": 0}, "iterations"),
        ({"region_labels": ("unknown",)}, "labels"),
        ({"identification_method": "unknown"}, "identification"),
    ],
)
def test_flash_grid_phase_boundary_refinement_validates_inputs(updates, match):
    arguments = {
        "model": _binary_model(),
        "temperature_axis": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "pressure_axis": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "composition": torch.tensor([0.5, 0.5], dtype=torch.float64),
        "region_codes": torch.zeros((2, 2), dtype=torch.int8),
        "converged": torch.ones((2, 2), dtype=torch.bool),
    }
    arguments.update(updates)
    with pytest.raises(ValueError, match=match):
        refine_flash_grid_phase_boundaries(**arguments)


def test_binary_invariant_validates_inputs():
    model = _binary_model()
    temperature = torch.tensor(180.0, dtype=torch.float64)
    pressure = torch.tensor(2.73e6, dtype=torch.float64)
    guesses = torch.tensor(
        [[0.199, 0.801], [0.781, 0.219], [0.958, 0.042]],
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="scalar"):
        solve_binary_three_phase_invariant(
            model,
            temperature[None],
            pressure,
            guesses,
        )
    with pytest.raises(ValueError, match="three two-component"):
        solve_binary_three_phase_invariant(model, temperature, pressure, guesses[:2])
    with pytest.raises(ValueError, match="strictly interior"):
        solve_binary_three_phase_invariant(
            model,
            temperature,
            pressure,
            torch.tensor(
                [[0.0, 1.0], [0.5, 0.5], [0.9, 0.1]],
                dtype=torch.float64,
            ),
        )
    with pytest.raises(ValueError, match="positive"):
        solve_binary_three_phase_invariant(
            model,
            temperature,
            pressure,
            guesses,
            max_iterations=0,
        )
    with pytest.raises(ValueError, match="method"):
        solve_binary_three_phase_invariant(
            model,
            temperature,
            pressure,
            guesses,
            method="invalid",
        )


def test_grid_phase_identification_covers_single_two_phase_and_unavailable_regions():
    model = peng_robinson_1978(
        ComponentSet(
            ("methane",),
            torch.tensor([190.6], dtype=torch.float64),
            torch.tensor([4.6e6], dtype=torch.float64),
            torch.tensor([0.008], dtype=torch.float64),
            torch.tensor([0.01604], dtype=torch.float64),
            torch.tensor([9.93e-5], dtype=torch.float64),
        )
    )
    temperatures = torch.tensor(
        [120.0, 500.0, 120.0, 500.0, 300.0],
        dtype=torch.float64,
    )
    pressures = torch.tensor(
        [5.0e6, 1.0e5, 5.0e6, 1.0e5, 1.0e5],
        dtype=torch.float64,
    )
    feeds = torch.ones((5, 1), dtype=torch.float64)
    phase_fractions = torch.zeros((5, 3), dtype=torch.float64)
    phase_fractions[:2, 0] = 1.0
    phase_fractions[2:4, :2] = 0.5
    phase_fractions[4, 0] = 1.0
    phase_compositions = torch.full((5, 3, 1), torch.nan, dtype=torch.float64)
    phase_compositions[:, 0] = 1.0
    phase_compositions[2:4, 1] = 1.0
    equilibrium = grid_module.GridEquilibrium(
        temperatures,
        pressures,
        feeds,
        (5,),
        phase_fractions,
        phase_compositions,
        torch.tensor([1, 1, 2, 2, 1]),
        torch.zeros(5),
        torch.zeros(5),
        torch.zeros(5),
        torch.tensor([True, True, True, True, False]),
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
        0,
    )

    result = identify_grid_phases(
        model,
        equilibrium,
        methods=("pedersen-volume-to-covolume",),
        ambiguity_relative_tolerance=0.0,
    )
    assert result.region_codes.tolist() == [[1, 0, 3, 2, 5]]

    unavailable_equilibrium = replace(
        equilibrium,
        temperatures=equilibrium.temperatures[:1],
        pressures=equilibrium.pressures[:1],
        feeds=equilibrium.feeds[:1],
        grid_shape=(1,),
        phase_fractions=equilibrium.phase_fractions[:1],
        phase_compositions=equilibrium.phase_compositions[:1],
        phase_counts=torch.ones(1, dtype=torch.int64),
        gibbs_reduction=equilibrium.gibbs_reduction[:1],
        fugacity_residual=equilibrium.fugacity_residual[:1],
        material_balance_residual=equilibrium.material_balance_residual[:1],
        converged=equilibrium.converged[:1],
    )
    unavailable = identify_grid_phases(
        object(),
        unavailable_equilibrium,
        methods=("li-pseudo-critical-temperature",),
    )
    assert unavailable.region_codes.tolist() == [[5]]

    nonconverged = identify_grid_phases(
        model,
        replace(
            unavailable_equilibrium,
            converged=torch.tensor([False]),
        ),
        methods=(
            "li-pseudo-critical-temperature",
            "perschke-negative-flash",
        ),
    )
    assert nonconverged.region_codes.tolist() == [[5], [5]]

    with pytest.raises(ValueError, match="phase counts"):
        identify_grid_phases(
            model,
            replace(equilibrium, phase_counts=torch.ones(2, dtype=torch.int64)),
            methods=("pedersen-volume-to-covolume",),
        )


def test_grid_two_phase_batch_isolates_non_straddling_k_values():
    model = _binary_model()
    temperatures = torch.full((2,), 180.0, dtype=torch.float64)
    pressures = torch.full((2,), 2.0e6, dtype=torch.float64)
    feeds = torch.full((2, 2), 0.5, dtype=torch.float64)
    result = grid_module._batched_two_phase_in_chunks(
        model,
        temperatures,
        pressures,
        feeds,
        torch.tensor(
            [
                [2.0, 0.5],
                [1.0, 1.0],
            ],
            dtype=torch.float64,
        ),
        GridFlashOptions(),
    )

    assert result.converged.shape == (2,)
    assert torch.isfinite(result.residual_norm[0])
    assert not bool(result.converged[1])
    assert torch.isinf(result.residual_norm[1])
    torch.testing.assert_close(result.liquid_composition[1], feeds[1])
    torch.testing.assert_close(result.vapor_composition[1], feeds[1])


def test_grid_internal_phase_merge_refinement_and_topology_helpers():
    options = GridFlashOptions(random_allocation_starts=0)
    fractions, _compositions = grid_module._merge_candidate_phases(
        torch.tensor([0.60, 0.399999, 1.0e-6], dtype=torch.float64),
        torch.tensor(
            [[0.2, 0.8], [0.2005, 0.7995], [0.9, 0.1]],
            dtype=torch.float64,
        ),
        options,
    )
    assert fractions.numel() == 1
    torch.testing.assert_close(fractions, torch.tensor([0.999999], dtype=torch.float64))
    with pytest.raises(RuntimeError, match="no active phase"):
        grid_module._merge_candidate_phases(
            torch.zeros(3, dtype=torch.float64),
            torch.full((3, 2), 0.5, dtype=torch.float64),
            options,
        )

    model = _binary_model()
    temperature = torch.tensor(350.0, dtype=torch.float64)
    pressure = torch.tensor(1.0e5, dtype=torch.float64)
    feed = torch.tensor([0.5, 0.5], dtype=torch.float64)
    one_fraction = torch.ones(1, dtype=torch.float64)
    one_composition = feed[None]
    refined = grid_module._refine_phase_equilibrium(
        model,
        temperature,
        pressure,
        feed,
        one_fraction,
        one_composition,
        options,
    )
    assert refined[0].numel() == 1
    assert (
        grid_module._fugacity_residual(
            model,
            temperature,
            pressure,
            one_composition,
        )
        == 0.0
    )

    initial_logits = torch.zeros((3, 2), dtype=torch.float64)
    _, robust_fractions, _ = grid_module._refine_and_reduce_candidate_robust(
        model,
        temperature,
        pressure,
        feed,
        initial_logits,
        options,
    )
    assert robust_fractions.numel() == 1

    counts = torch.tensor(
        [
            [3, 1, 3, 1],
            [3, 2, 3, 1],
            [3, 1, 3, 3],
        ],
        dtype=torch.int64,
    ).reshape(-1)
    bracketed = grid_module._bracketed_lower_phase_indices(counts, 3, 4)
    assert 1 in bracketed
    assert 5 in bracketed
    assert grid_module._three_phase_boundary_indices(
        torch.tensor([[3, 2, 2, 2]], dtype=torch.int64).reshape(-1),
        1,
        4,
    ) == [1]
    phase_fractions = torch.zeros((12, 3), dtype=torch.float64)
    phase_fractions[:, 0] = 1.0
    phase_fractions[4, :3] = torch.tensor([0.2, 0.3, 0.5])
    phase_compositions = torch.full((12, 3, 2), torch.nan, dtype=torch.float64)
    phase_compositions[:, 0] = torch.tensor([0.5, 0.5])
    phase_compositions[4] = torch.tensor([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
    seeds = grid_module._continuation_seed_logits(
        counts,
        phase_fractions,
        phase_compositions,
        [1, 5, 7],
        3,
        4,
    )
    assert 5 in seeds
    assert seeds[5].shape == (3, 2)
    vertical_counts = counts.clone()
    vertical_counts[1] = 3
    phase_fractions[1, :3] = torch.tensor([0.2, 0.3, 0.5])
    phase_compositions[1] = torch.tensor([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
    vertical_seed = grid_module._continuation_seed_logits(
        vertical_counts,
        phase_fractions,
        phase_compositions,
        [5],
        3,
        4,
    )
    assert 5 in vertical_seed
    assert (
        grid_module._continuation_seed_logits(
            torch.full((4,), 2, dtype=torch.int64),
            torch.full((4, 3), 1.0 / 3.0),
            torch.full((4, 3, 2), 0.5),
            [0],
            2,
            2,
        )
        == {}
    )

    nwe_model = _north_ward_estes_model()
    nwe_temperatures, nwe_pressures, nwe_feeds = grid_module._grid_states(_north_ward_estes_state())
    starts = grid_module._allocation_initial_logits(
        nwe_model,
        nwe_temperatures,
        nwe_pressures,
        nwe_feeds,
        options,
    )
    _, fast_fractions, fast_compositions = grid_module._refine_and_reduce_candidate(
        nwe_model,
        nwe_temperatures[0],
        nwe_pressures[0],
        nwe_feeds[0],
        starts[0, 0],
        options,
    )
    normalized, balance, fugacity = grid_module._candidate_diagnostics(
        nwe_model,
        nwe_temperatures[0],
        nwe_pressures[0],
        nwe_feeds[0],
        fast_fractions,
        fast_compositions,
    )
    torch.testing.assert_close(normalized.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.isfinite(balance)
    assert torch.isfinite(fugacity)


def test_binary_invariant_split_rejects_inapplicable_invariants():
    model = _binary_model()
    invariant = solve_binary_three_phase_invariant(
        model,
        torch.tensor(180.0, dtype=torch.float64),
        torch.tensor(2.73e6, dtype=torch.float64),
        torch.tensor(
            [[0.199, 0.801], [0.781, 0.219], [0.958, 0.042]],
            dtype=torch.float64,
        ),
    )
    options = GridFlashOptions()
    assert (
        grid_module._binary_invariant_split(
            (invariant,),
            torch.tensor(181.0, dtype=torch.float64),
            invariant.pressure,
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            options,
        )
        is None
    )
    assert (
        grid_module._binary_invariant_split(
            (replace(invariant, converged=False),),
            invariant.temperature,
            invariant.pressure,
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            options,
        )
        is None
    )
    assert (
        grid_module._binary_invariant_split(
            (invariant,),
            invariant.temperature,
            invariant.pressure,
            torch.tensor([0.99, 0.01], dtype=torch.float64),
            options,
        )
        is None
    )
    assert (
        grid_module._binary_invariant_split(
            (invariant,),
            invariant.temperature,
            invariant.pressure,
            torch.ones(3, dtype=torch.float64) / 3.0,
            options,
        )
        is None
    )

    unconverged = solve_binary_three_phase_invariant(
        model,
        torch.tensor(180.0, dtype=torch.float64),
        torch.tensor(2.73e6, dtype=torch.float64),
        torch.tensor(
            [[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]],
            dtype=torch.float64,
        ),
        tolerance=1.0e-30,
        max_iterations=1,
    )
    assert not unconverged.converged


def test_newton_regularization_and_invariant_line_search_failure(monkeypatch):
    model = _binary_model()
    temperature = torch.tensor(180.0, dtype=torch.float64)
    pressure = torch.tensor(2.0e6, dtype=torch.float64)
    feed = torch.tensor([0.5, 0.5], dtype=torch.float64)
    original_solve = torch.linalg.solve
    calls = 0

    def singular_once(matrix, right_hand_side):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise torch.linalg.LinAlgError("forced singular Newton matrix")
        return original_solve(matrix, right_hand_side)

    monkeypatch.setattr(grid_module.torch.linalg, "solve", singular_once)
    fractions, compositions, gibbs = grid_module._refine_phase_equilibrium(
        model,
        temperature,
        pressure,
        feed,
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        torch.tensor([[0.2, 0.8], [0.8, 0.2]], dtype=torch.float64),
        GridFlashOptions(),
    )
    assert calls >= 2
    assert torch.isfinite(fractions).all()
    assert torch.isfinite(compositions).all()
    assert torch.isfinite(gibbs)

    monkeypatch.setattr(
        grid_module.torch.linalg,
        "solve",
        lambda matrix, right_hand_side: torch.zeros_like(right_hand_side),
    )
    invariant = solve_binary_three_phase_invariant(
        model,
        temperature,
        torch.tensor(2.73e6, dtype=torch.float64),
        torch.tensor(
            [[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]],
            dtype=torch.float64,
        ),
    )
    assert not invariant.converged
    assert invariant.iterations == 1


def test_sparse_gibbs_fallback_vectorized_two_phase_and_scalar_paths(
    monkeypatch,
    capsys,
):
    model = _binary_model()
    temperatures = torch.tensor([250.0, 260.0], dtype=torch.float64)
    pressures = torch.tensor([2.0e6, 2.5e6], dtype=torch.float64)
    feeds = torch.full((2, 2), 0.5, dtype=torch.float64)
    one_phase_gibbs = torch.zeros(2, dtype=torch.float64)
    distinct = torch.tensor(
        [[0.1, 0.9], [0.4, 0.6], [0.9, 0.1]],
        dtype=torch.float64,
    )
    identical = torch.full((3, 2), 0.5, dtype=torch.float64)
    allocation_mode = {"distinct": True}

    def initial_logits(_model, temperature, _pressure, feed, _options):
        return torch.zeros(
            (temperature.numel(), 1, 3, feed.shape[-1]),
            dtype=feed.dtype,
            device=feed.device,
        )

    def allocation_quantities(_model, temperature, _pressure, feed, logits):
        state_count, start_count = logits.shape[:2]
        gibbs = logits.square().sum(dim=(-1, -2)) - 1.0
        fractions = torch.softmax(logits.sum(dim=-1), dim=-1)
        base = distinct if allocation_mode["distinct"] else identical
        compositions = base.to(feed).expand(state_count, start_count, -1, -1)
        return gibbs, fractions, compositions

    def batched_refine(_model, temperature, _pressure, feed, _fractions, _compositions, _options):
        state_count = temperature.numel()
        return (
            torch.full((state_count, 3), 1.0 / 3.0, dtype=feed.dtype),
            distinct.to(feed).expand(state_count, -1, -1),
            torch.full((state_count,), -1.0, dtype=feed.dtype),
            torch.ones(state_count, dtype=feed.dtype),
        )

    monkeypatch.setattr(grid_module, "_allocation_initial_logits", initial_logits)
    monkeypatch.setattr(grid_module, "_allocation_quantities", allocation_quantities)
    monkeypatch.setattr(grid_module, "_batched_refine_three_phase", batched_refine)

    def two_phase_result(_model, temperature, _pressure, feed, _k, _options):
        state_count = temperature.numel()
        liquid = torch.tensor([0.2, 0.8], dtype=feed.dtype).expand(state_count, -1)
        vapor = torch.tensor([0.8, 0.2], dtype=feed.dtype).expand(state_count, -1)
        return grid_module.BatchedTwoPhaseFlashResult(
            torch.full((state_count,), 0.5, dtype=feed.dtype),
            torch.full((state_count,), 0.5, dtype=feed.dtype),
            liquid,
            vapor,
            vapor / liquid,
            1,
            torch.ones(state_count, dtype=torch.bool),
            torch.zeros(state_count, dtype=feed.dtype),
        )

    monkeypatch.setattr(
        grid_module,
        "_batched_two_phase_in_chunks",
        two_phase_result,
    )
    vectorized = grid_module._gibbs_fallback_grid_states(
        model,
        temperatures[:1],
        pressures[:1],
        feeds[:1],
        [0],
        one_phase_gibbs[:1],
        one_phase_gibbs[:1],
        torch.zeros(1, dtype=torch.bool),
        GridFlashOptions(
            random_allocation_starts=0,
            gibbs_fallback_adam_iterations=1,
            debug=True,
        ),
        (),
        {0: torch.zeros((3, 2), dtype=torch.float64)},
    )
    assert vectorized[0][0].numel() == 2
    assert "Batched three-phase Newton" in capsys.readouterr().out

    allocation_mode["distinct"] = False

    def diagnostics(_model, _temperature, _pressure, feed, fractions, compositions):
        normalized = fractions / fractions.sum()
        if compositions.shape[0] == 1:
            return normalized, feed.new_zeros(()), feed.new_zeros(())
        return normalized, feed.new_ones(()), feed.new_ones(())

    def fast_refine(_model, _temperature, _pressure, feed, _logits, _options):
        return (
            feed.new_tensor(-1.0),
            feed.new_tensor([0.5, 0.5]),
            torch.stack((feed.new_tensor([0.25, 0.75]), feed.new_tensor([0.75, 0.25]))),
        )

    def robust_refine(_model, _temperature, _pressure, feed, _logits, _options):
        return feed.new_tensor(-1.0), feed.new_ones(1), feed[None]

    monkeypatch.setattr(grid_module, "_candidate_diagnostics", diagnostics)
    monkeypatch.setattr(grid_module, "_refine_and_reduce_candidate", fast_refine)
    monkeypatch.setattr(
        grid_module,
        "_refine_and_reduce_candidate_robust",
        robust_refine,
    )
    monkeypatch.setattr(
        grid_module,
        "_candidate_gibbs_energy",
        lambda *_args: torch.tensor(-1.0, dtype=torch.float64),
    )
    scalar = grid_module._gibbs_fallback_grid_states(
        model,
        temperatures,
        pressures,
        feeds,
        [0, 1],
        one_phase_gibbs,
        one_phase_gibbs,
        torch.zeros(2, dtype=torch.bool),
        GridFlashOptions(
            random_allocation_starts=0,
            gibbs_fallback_adam_iterations=1,
            fallback_workers=2,
            debug=True,
        ),
        (),
    )
    assert sorted(scalar) == [0, 1]
    assert all(result[0].numel() == 1 for result in scalar.values())
    assert "Fallback candidates" in capsys.readouterr().out

    monkeypatch.setattr(
        grid_module,
        "_candidate_diagnostics",
        lambda _model, _temperature, _pressure, feed, fractions, _compositions: (
            fractions / fractions.sum(),
            feed.new_ones(()),
            feed.new_ones(()),
        ),
    )
    rejected = grid_module._gibbs_fallback_grid_states(
        model,
        temperatures[:1],
        pressures[:1],
        feeds[:1],
        [0],
        one_phase_gibbs[:1],
        one_phase_gibbs[:1],
        torch.zeros(1, dtype=torch.bool),
        GridFlashOptions(
            random_allocation_starts=0,
            gibbs_fallback_adam_iterations=1,
            fallback_workers=1,
        ),
        (),
    )
    assert rejected == {}


def test_phase_reduction_handles_repeated_phase_disappearance(monkeypatch):
    model = _binary_model()
    temperature = torch.tensor(250.0, dtype=torch.float64)
    pressure = torch.tensor(2.0e6, dtype=torch.float64)
    feed = torch.tensor([0.5, 0.5], dtype=torch.float64)
    options = GridFlashOptions()

    def phases(count):
        fractions = torch.full((count,), 1.0 / count, dtype=torch.float64)
        compositions = torch.stack(
            tuple(
                torch.tensor(
                    [0.2 + 0.6 * index / max(count - 1, 1), 0.8 - 0.6 * index / max(count - 1, 1)],
                    dtype=torch.float64,
                )
                for index in range(count)
            )
        )
        return fractions, compositions

    three = phases(3)
    two = phases(2)
    one = phases(1)
    monkeypatch.setattr(
        grid_module,
        "_allocation_quantities",
        lambda *_args: (
            torch.zeros((1, 1), dtype=torch.float64),
            three[0].reshape(1, 1, 3),
            three[1].reshape(1, 1, 3, 2),
        ),
    )
    fast_merge_returns = iter((three, two, two, one))
    monkeypatch.setattr(
        grid_module,
        "_merge_candidate_phases",
        lambda *_args: next(fast_merge_returns),
    )
    monkeypatch.setattr(
        grid_module,
        "_refine_phase_equilibrium",
        lambda _model, _temperature, _pressure, _feed, fractions, compositions, _options: (
            fractions,
            compositions,
            torch.tensor(-1.0, dtype=torch.float64),
        ),
    )
    _, fast_fractions, _ = grid_module._refine_and_reduce_candidate(
        model,
        temperature,
        pressure,
        feed,
        torch.zeros((3, 2), dtype=torch.float64),
        options,
    )
    assert fast_fractions.numel() == 1

    allocation_returns = iter(
        (
            (torch.tensor(-1.0), *three),
            (torch.tensor(-1.0), *three),
            (torch.tensor(-1.0), *three),
        )
    )
    robust_merge_returns = iter((three, two, three, two))
    monkeypatch.setattr(
        grid_module,
        "_refine_state_allocation",
        lambda *_args: next(allocation_returns),
    )
    monkeypatch.setattr(
        grid_module,
        "_merge_candidate_phases",
        lambda *_args: next(robust_merge_returns),
    )
    _, robust_fractions, _ = grid_module._refine_and_reduce_candidate_robust(
        model,
        temperature,
        pressure,
        feed,
        torch.zeros((3, 2), dtype=torch.float64),
        options,
    )
    assert robust_fractions.numel() == 3

    single_merge_returns = iter((three, one))
    monkeypatch.setattr(
        grid_module,
        "_refine_state_allocation",
        lambda *_args: (torch.tensor(-1.0), *three),
    )
    monkeypatch.setattr(
        grid_module,
        "_merge_candidate_phases",
        lambda *_args: next(single_merge_returns),
    )
    _, disappeared_fractions, _ = grid_module._refine_and_reduce_candidate_robust(
        model,
        temperature,
        pressure,
        feed,
        torch.zeros((3, 2), dtype=torch.float64),
        options,
    )
    assert disappeared_fractions.numel() == 1


def test_flash_grid_runs_and_rejects_topology_audit_replacement(monkeypatch):
    model = _binary_model()
    state = ChemicalState(
        torch.full((1, 3), 250.0, dtype=torch.float64),
        torch.full((1, 3), 2.0e6, dtype=torch.float64),
        torch.full((1, 3, 2), 0.5, dtype=torch.float64),
    )

    def stable_result(_model, temperatures, _pressures, compositions, _options):
        state_count = temperatures.numel()
        return grid_module.BatchedStabilityResult(
            torch.ones(state_count, dtype=torch.bool),
            torch.zeros(state_count, dtype=torch.float64),
            compositions,
            1,
            torch.ones(state_count, dtype=torch.bool),
            torch.zeros(state_count, dtype=torch.float64),
        )

    invariant_fractions = torch.full((3,), 1.0 / 3.0, dtype=torch.float64)
    invariant_compositions = torch.full((3, 2), 0.5, dtype=torch.float64)
    monkeypatch.setattr(grid_module, "_batched_stability_in_chunks", stable_result)
    monkeypatch.setattr(
        grid_module,
        "_binary_invariant_split",
        lambda *_args: (invariant_fractions, invariant_compositions),
    )
    calls = 0

    def fallback(
        _model,
        _temperatures,
        _pressures,
        feeds,
        state_indices,
        _one_phase_gibbs,
        current_gibbs,
        _current_converged,
        _options,
        _binary_invariants,
        _seed_logits=None,
    ):
        nonlocal calls
        calls += 1
        if calls == 1:
            results = {}
            for index, count in enumerate((3, 2, 3)):
                fractions = torch.full((count,), 1.0 / count, dtype=feeds.dtype)
                compositions = feeds[index].expand(count, -1)
                results[index] = fractions, compositions, current_gibbs[index]
            return results
        assert state_indices == [1]
        return {
            1: (
                invariant_fractions,
                invariant_compositions,
                current_gibbs[1],
            )
        }

    monkeypatch.setattr(grid_module, "_gibbs_fallback_grid_states", fallback)
    result = flash_grid(model, state)

    assert calls == 2
    assert result.topology_audit_count == 1
    assert result.initial_fallback_replacements == 3
    assert result.topology_audit_replacements == 0
    assert result.phase_counts.tolist() == [3, 2, 3]


def test_flash_grid_advances_only_lower_gibbs_three_phase_boundary(monkeypatch):
    model = _binary_model()
    state = ChemicalState(
        torch.full((1, 4), 250.0, dtype=torch.float64),
        torch.full((1, 4), 2.0e6, dtype=torch.float64),
        torch.full((1, 4, 2), 0.5, dtype=torch.float64),
    )

    def stable_result(_model, temperatures, _pressures, compositions, _options):
        state_count = temperatures.numel()
        return grid_module.BatchedStabilityResult(
            torch.ones(state_count, dtype=torch.bool),
            torch.zeros(state_count, dtype=torch.float64),
            compositions,
            1,
            torch.ones(state_count, dtype=torch.bool),
            torch.zeros(state_count, dtype=torch.float64),
        )

    three_fractions = torch.full((3,), 1.0 / 3.0, dtype=torch.float64)
    two_fractions = torch.full((2,), 0.5, dtype=torch.float64)
    three_compositions = torch.full((3, 2), 0.5, dtype=torch.float64)
    two_compositions = torch.full((2, 2), 0.5, dtype=torch.float64)
    monkeypatch.setattr(grid_module, "_batched_stability_in_chunks", stable_result)
    monkeypatch.setattr(
        grid_module,
        "_binary_invariant_split",
        lambda *_args: (three_fractions, three_compositions),
    )
    calls = 0

    def fallback(
        _model,
        _temperatures,
        _pressures,
        _feeds,
        state_indices,
        _one_phase_gibbs,
        current_gibbs,
        _current_converged,
        _options,
        _binary_invariants,
        _seed_logits=None,
    ):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                index: (
                    (three_fractions, three_compositions)
                    if index == 0
                    else (two_fractions, two_compositions)
                )
                + (current_gibbs[index],)
                for index in range(4)
            }
        if calls == 2:
            assert state_indices == [1]
            return {
                1: (
                    three_fractions,
                    three_compositions,
                    current_gibbs[1] - 1.0e-3,
                )
            }
        assert calls == 3
        assert state_indices == [2]
        return {
            2: (
                three_fractions,
                three_compositions,
                current_gibbs[2],
            )
        }

    monkeypatch.setattr(grid_module, "_gibbs_fallback_grid_states", fallback)
    result = flash_grid(model, state)

    assert calls == 3
    assert result.topology_audit_count == 2
    assert result.initial_fallback_replacements == 4
    assert result.topology_audit_replacements == 1
    assert result.phase_counts.tolist() == [3, 3, 2, 2]
