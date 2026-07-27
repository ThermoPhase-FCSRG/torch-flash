from __future__ import annotations

import csv
import warnings
from pathlib import Path

import pytest
import torch

import torch_flash.envelope as envelope
import torch_flash.fitting as fitting_module
from torch_flash import (
    BinaryThreePhaseInvariantBatch,
    ChemicalState,
    CubicInteractionFitSystem,
    batched_tangent_plane_stability,
    binary_bubble_point,
    binary_bubble_temperature,
    binary_phase_equilibrium_point,
    build_cubic_interaction_models,
    component_set,
    continue_phase_transition_branch,
    evaluate_phase_transition_state,
    evaluate_phase_transition_states,
    fit_cubic_phase_transition_interactions,
    fixed_vapor_ratio_vle_point,
    peng_robinson_1978,
    phase_properties,
    phase_transition_pressure,
    solve_batched_phase_transition_pressures,
    state_derivatives,
    tangent_plane_stability,
    trace_binary_pxy_isotherm,
    trace_binary_txy_isobar,
    trace_phase_envelope_set,
    two_phase_flash,
    two_phase_trust_region_flash,
)
from torch_flash.constants import STANDARD_PRESSURE, R
from torch_flash.envelope import phase_envelope, saturation_point
from torch_flash.exceptions import ConvergenceWarning, ExperimentalModelWarning
from torch_flash.flash.grid import BinaryThreePhaseInvariant
from torch_flash.flash.multiphase import _default_multiphase_k, multiphase_flash
from torch_flash.flash.stability import _newton_minimize
from torch_flash.standard_state import IdealGasPolynomial

DATA = Path(__file__).parent / "data"


def test_fixed_vapor_ratio_vle_matches_binary_equilibrium():
    model = peng_robinson_1978(component_set(("hydrogen", "water"), dtype=torch.float64))
    temperature = torch.tensor(323.15, dtype=torch.float64)
    pressure = torch.tensor(10.0e6, dtype=torch.float64)
    result = fixed_vapor_ratio_vle_point(
        model,
        temperature,
        pressure,
        torch.tensor([2.0, 0.0], dtype=torch.float64),
        1,
    )
    reference = binary_phase_equilibrium_point(
        model,
        temperature,
        pressure,
        result.liquid_composition,
        result.vapor_composition,
        phase_kinds=("liquid", "vapor"),
    )
    assert result.converged
    assert result.variable_vapor_component == 1
    assert result.phase_separation > 0.99
    assert result.residual_norm <= 1.0e-8
    torch.testing.assert_close(result.liquid_composition, reference.phase1_composition)
    torch.testing.assert_close(result.vapor_composition, reference.phase2_composition)
    with pytest.raises(ValueError, match="phase kinds"):
        binary_phase_equilibrium_point(
            model,
            temperature,
            pressure,
            result.liquid_composition,
            result.vapor_composition,
            phase_kinds=("liquid", "invalid"),
        )


def test_fixed_vapor_ratio_vle_preserves_ternary_dry_gas_ratio():
    model = peng_robinson_1978(
        component_set(("hydrogen", "nitrogen", "water"), dtype=torch.float64)
    )
    dry_composition = torch.tensor([0.746, 0.254, 0.0], dtype=torch.float64)
    result = fixed_vapor_ratio_vle_point(
        model,
        torch.tensor(298.15, dtype=torch.float64),
        torch.tensor(10.1325e6, dtype=torch.float64),
        dry_composition,
        2,
        initial_liquid_composition=torch.tensor(
            [7.46e-4, 2.54e-4, 0.999],
            dtype=torch.float64,
        ),
        initial_variable_vapor_fraction=torch.tensor(3.85e-4, dtype=torch.float64),
        phase_kinds=("stable", "stable"),
    )
    predicted_dry = result.vapor_composition[:2] / result.vapor_composition[:2].sum()
    assert result.converged
    torch.testing.assert_close(predicted_dry, dry_composition[:2])
    torch.testing.assert_close(
        result.liquid_composition.sum(),
        torch.tensor(1.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.vapor_composition.sum(),
        torch.tensor(1.0, dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("temperature", "pressure", "dry_composition", "variable_component", "options", "match"),
    [
        (
            torch.tensor([298.15], dtype=torch.float64),
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            1,
            {},
            "temperature",
        ),
        (
            torch.tensor(298.15, dtype=torch.float64),
            torch.tensor(-1.0e6, dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            1,
            {},
            "pressure",
        ),
        (
            torch.tensor(298.15, dtype=torch.float64),
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            1,
            {"tolerance": 0.0},
            "solver controls",
        ),
        (
            torch.tensor(298.15, dtype=torch.float64),
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            1,
            {"minimum_phase_separation": -1.0},
            "phase separation",
        ),
        (
            torch.tensor(298.15, dtype=torch.float64),
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            1,
            {"phase_kinds": ("invalid", "vapor")},
            "phase kinds",
        ),
        (
            torch.tensor(298.15, dtype=torch.float64),
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([1, 0]),
            1,
            {},
            "floating tensor",
        ),
        (
            torch.tensor(298.15, dtype=torch.float64),
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            2,
            {},
            "outside",
        ),
        (
            torch.tensor(298.15, dtype=torch.float64),
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([float("nan"), 0.0], dtype=torch.float64),
            1,
            {},
            "finite and nonnegative",
        ),
        (
            torch.tensor(298.15, dtype=torch.float64),
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([0.9, 0.1], dtype=torch.float64),
            1,
            {},
            "must be zero",
        ),
        (
            torch.tensor(298.15, dtype=torch.float64),
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            2,
            {},
            "positive ratio",
        ),
        (
            torch.tensor(298.15, dtype=torch.float64),
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            1,
            {
                "initial_liquid_composition": torch.tensor(
                    [1.0, 0.0],
                    dtype=torch.float64,
                )
            },
            "finite positive vector",
        ),
        (
            torch.tensor(298.15, dtype=torch.float64),
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            1,
            {"initial_variable_vapor_fraction": torch.tensor(1.0)},
            "inside",
        ),
    ],
)
def test_fixed_vapor_ratio_vle_validates_inputs(
    binary_model,
    temperature,
    pressure,
    dry_composition,
    variable_component,
    options,
    match,
):
    with pytest.raises(ValueError, match=match):
        fixed_vapor_ratio_vle_point(
            binary_model,
            temperature,
            pressure,
            dry_composition,
            variable_component,
            **options,
        )


def test_direct_properties_and_derivatives(binary_model):
    state = ChemicalState(
        torch.tensor(320.0, dtype=torch.float64),
        torch.tensor(2.0e6, dtype=torch.float64),
        torch.tensor([0.7, 0.3], dtype=torch.float64),
    )
    properties = phase_properties(binary_model, state, "vapor")
    assert properties.compressibility_factor > 0.0
    assert properties.molar_volume > 0.0
    assert properties.residual_enthalpy is not None
    assert properties.residual_entropy is not None
    expected_fugacity = state.composition * properties.fugacity_coefficients * state.pressure
    torch.testing.assert_close(properties.fugacities, expected_fugacity)
    torch.testing.assert_close(
        properties.log_fugacities,
        torch.log(properties.fugacities / STANDARD_PRESSURE),
    )
    torch.testing.assert_close(
        properties.chemical_potentials,
        R * state.temperature * properties.log_fugacities,
    )
    torch.testing.assert_close(
        properties.reduced_chemical_potentials,
        properties.chemical_potentials / (R * state.temperature),
    )
    torch.testing.assert_close(
        properties.molar_gibbs_energy,
        torch.dot(state.composition, properties.chemical_potentials),
    )
    pressure_volume = state.pressure * properties.molar_volume
    rt = R * state.temperature
    torch.testing.assert_close(
        properties.molar_helmholtz_energy,
        properties.molar_gibbs_energy - pressure_volume,
    )
    torch.testing.assert_close(
        properties.reduced_gibbs_energy,
        properties.molar_gibbs_energy / rt,
    )
    torch.testing.assert_close(
        properties.reduced_helmholtz_energy,
        properties.molar_helmholtz_energy / rt,
    )
    torch.testing.assert_close(
        properties.reduced_residual_gibbs_energy,
        torch.dot(state.composition, properties.log_fugacity_coefficients),
    )
    torch.testing.assert_close(
        properties.reduced_residual_helmholtz_energy,
        properties.reduced_residual_gibbs_energy
        - pressure_volume / rt
        + 1.0
        + torch.log(pressure_volume / rt),
    )
    torch.testing.assert_close(
        properties.reduced_residual_helmholtz_energy,
        binary_model.residual_helmholtz_rt(
            state.temperature,
            properties.molar_volume,
            state.composition,
        ),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    no_caloric = phase_properties(binary_model, state, "vapor", caloric=False)
    assert no_caloric.residual_enthalpy is None
    assert no_caloric.residual_entropy is None

    standard = IdealGasPolynomial(
        torch.zeros((2, 4), dtype=torch.float64),
        torch.tensor([100.0, 200.0], dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
    )
    referenced = phase_properties(
        binary_model,
        state,
        "vapor",
        standard_state=standard,
    )
    torch.testing.assert_close(
        referenced.chemical_potentials - properties.chemical_potentials,
        torch.tensor([100.0, 200.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        referenced.reduced_residual_gibbs_energy,
        properties.reduced_residual_gibbs_energy,
    )
    torch.testing.assert_close(
        referenced.reduced_residual_helmholtz_energy,
        properties.reduced_residual_helmholtz_energy,
    )
    derivatives = state_derivatives(
        binary_model,
        state,
        "vapor",
        standard_state=standard,
    )
    assert derivatives.dchemical_potential_dlogits.shape == (2, 2)
    assert derivatives.dchemical_potential_dindependent_composition.shape == (2, 1)
    assert derivatives.dchemical_potential_dtemperature.shape == (2,)
    assert derivatives.dchemical_potential_dpressure.shape == (2,)
    assert derivatives.dfugacity_dlogits.shape == (2, 2)
    assert derivatives.dfugacity_dindependent_composition.shape == (2, 1)
    assert derivatives.dfugacity_dtemperature.shape == (2,)
    assert derivatives.dfugacity_dpressure.shape == (2,)
    assert derivatives.dlog_fugacity_dlogits.shape == (2, 2)
    assert derivatives.dlog_fugacity_dindependent_composition.shape == (2, 1)
    assert derivatives.dlog_fugacity_dtemperature.shape == (2,)
    assert derivatives.dlog_fugacity_dpressure.shape == (2,)
    assert derivatives.dreduced_chemical_potential_dlogits.shape == (2, 2)
    assert derivatives.dreduced_chemical_potential_dindependent_composition.shape == (2, 1)
    assert derivatives.dreduced_chemical_potential_dtemperature.shape == (2,)
    assert derivatives.dreduced_chemical_potential_dpressure.shape == (2,)
    assert derivatives.dlog_fugacity_coefficient_dlogits.shape == (2, 2)
    assert derivatives.dlog_fugacity_coefficient_dindependent_composition.shape == (2, 1)
    assert derivatives.dlog_fugacity_coefficient_dtemperature.shape == (2,)
    assert derivatives.dlog_fugacity_coefficient_dpressure.shape == (2,)
    assert derivatives.dlog_fugacity_coefficient_dmoles.shape == (2, 2)
    assert derivatives.dfugacity_coefficient_dlogits.shape == (2, 2)
    assert derivatives.dfugacity_coefficient_dindependent_composition.shape == (2, 1)
    assert derivatives.dfugacity_coefficient_dtemperature.shape == (2,)
    assert derivatives.dfugacity_coefficient_dpressure.shape == (2,)
    assert derivatives.dfugacity_coefficient_dmoles.shape == (2, 2)
    assert derivatives.dlog_fugacity_dmoles.shape == (2, 2)
    assert derivatives.dfugacity_dmoles.shape == (2, 2)
    assert derivatives.dchemical_potential_dmoles.shape == (2, 2)
    assert derivatives.dreduced_chemical_potential_dmoles.shape == (2, 2)
    assert derivatives.dmolar_volume_dlogits.shape == (2,)
    assert derivatives.dmolar_volume_dindependent_composition.shape == (1,)
    assert derivatives.dmolar_volume_dtemperature.ndim == 0
    assert derivatives.dmolar_volume_dpressure.ndim == 0
    assert derivatives.dmolar_volume_dmoles.shape == (2,)
    torch.testing.assert_close(
        derivatives.dfugacity_dlogits,
        properties.fugacities[:, None] * derivatives.dlog_fugacity_dlogits,
    )
    torch.testing.assert_close(
        derivatives.dfugacity_dindependent_composition,
        properties.fugacities[:, None] * derivatives.dlog_fugacity_dindependent_composition,
    )
    torch.testing.assert_close(
        derivatives.dfugacity_dtemperature,
        properties.fugacities * derivatives.dlog_fugacity_dtemperature,
    )
    torch.testing.assert_close(
        derivatives.dfugacity_dpressure,
        properties.fugacities * derivatives.dlog_fugacity_dpressure,
    )
    torch.testing.assert_close(
        derivatives.dfugacity_coefficient_dlogits,
        properties.fugacity_coefficients[:, None] * derivatives.dlog_fugacity_coefficient_dlogits,
    )
    torch.testing.assert_close(
        derivatives.dfugacity_coefficient_dindependent_composition,
        properties.fugacity_coefficients[:, None]
        * derivatives.dlog_fugacity_coefficient_dindependent_composition,
    )
    torch.testing.assert_close(
        derivatives.dfugacity_coefficient_dtemperature,
        properties.fugacity_coefficients * derivatives.dlog_fugacity_coefficient_dtemperature,
    )
    torch.testing.assert_close(
        derivatives.dfugacity_coefficient_dpressure,
        properties.fugacity_coefficients * derivatives.dlog_fugacity_coefficient_dpressure,
    )
    torch.testing.assert_close(
        derivatives.dfugacity_coefficient_dmoles,
        properties.fugacity_coefficients[:, None] * derivatives.dlog_fugacity_coefficient_dmoles,
    )
    torch.testing.assert_close(
        derivatives.dfugacity_dmoles,
        properties.fugacities[:, None] * derivatives.dlog_fugacity_dmoles,
    )
    torch.testing.assert_close(
        derivatives.dchemical_potential_dlogits,
        R * state.temperature * derivatives.dlog_fugacity_dlogits,
    )
    torch.testing.assert_close(
        derivatives.dchemical_potential_dindependent_composition,
        R * state.temperature * derivatives.dlog_fugacity_dindependent_composition,
    )
    torch.testing.assert_close(
        derivatives.dchemical_potential_dpressure,
        R * state.temperature * derivatives.dlog_fugacity_dpressure,
    )
    torch.testing.assert_close(
        derivatives.dchemical_potential_dmoles,
        R * state.temperature * derivatives.dlog_fugacity_dmoles,
    )
    torch.testing.assert_close(
        derivatives.dreduced_chemical_potential_dlogits,
        derivatives.dchemical_potential_dlogits / (R * state.temperature),
    )
    torch.testing.assert_close(
        derivatives.dreduced_chemical_potential_dindependent_composition,
        derivatives.dchemical_potential_dindependent_composition / (R * state.temperature),
    )
    torch.testing.assert_close(
        derivatives.dreduced_chemical_potential_dpressure,
        derivatives.dchemical_potential_dpressure / (R * state.temperature),
    )
    torch.testing.assert_close(
        derivatives.dreduced_chemical_potential_dmoles,
        derivatives.dchemical_potential_dmoles / (R * state.temperature),
    )
    dlogx_dmoles = torch.diag(1.0 / state.composition) - torch.ones((2, 2), dtype=torch.float64)
    torch.testing.assert_close(
        derivatives.dlog_fugacity_dmoles,
        derivatives.dlog_fugacity_coefficient_dmoles + dlogx_dmoles,
    )
    torch.testing.assert_close(
        state.composition @ derivatives.dlog_fugacity_coefficient_dmoles,
        torch.zeros(2, dtype=torch.float64),
        atol=3.0e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        state.composition @ derivatives.dmolar_volume_dmoles,
        torch.zeros((), dtype=torch.float64),
        atol=2.0e-18,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.matmul(
            state.composition,
            derivatives.dchemical_potential_dindependent_composition,
        ),
        torch.zeros(1, dtype=torch.float64),
        atol=2.0e-10,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.dot(state.composition, derivatives.dchemical_potential_dpressure),
        referenced.molar_volume,
        rtol=2.0e-10,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        derivatives.dreduced_chemical_potential_dtemperature,
        derivatives.dchemical_potential_dtemperature / (R * state.temperature)
        - referenced.chemical_potentials / (R * state.temperature.square()),
    )
    torch.testing.assert_close(
        derivatives.dgibbs_dpressure,
        referenced.molar_volume,
        rtol=2.0e-10,
        atol=1.0e-12,
    )


def test_fugacity_and_chemical_potential_derivatives_match_finite_differences(
    binary_model,
):
    dtype = torch.float64
    state = ChemicalState(
        torch.tensor(320.0, dtype=dtype),
        torch.tensor(2.0e6, dtype=dtype),
        torch.tensor([0.7, 0.3], dtype=dtype),
    )
    derivatives = state_derivatives(binary_model, state, "vapor")

    def values(
        temperature: torch.Tensor,
        pressure: torch.Tensor,
        first_mole_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        composition = torch.stack((first_mole_fraction, 1.0 - first_mole_fraction))
        properties = phase_properties(
            binary_model,
            ChemicalState(temperature, pressure, composition),
            "vapor",
            caloric=False,
        )
        return (
            properties.fugacities,
            properties.log_fugacities,
            properties.chemical_potentials,
            properties.reduced_chemical_potentials,
        )

    base_arguments = (state.temperature, state.pressure, state.composition[0])
    steps = (
        torch.tensor(1.0e-3, dtype=dtype),
        torch.tensor(10.0, dtype=dtype),
        torch.tensor(1.0e-5, dtype=dtype),
    )
    expected = (
        (
            derivatives.dfugacity_dtemperature,
            derivatives.dlog_fugacity_dtemperature,
            derivatives.dchemical_potential_dtemperature,
            derivatives.dreduced_chemical_potential_dtemperature,
        ),
        (
            derivatives.dfugacity_dpressure,
            derivatives.dlog_fugacity_dpressure,
            derivatives.dchemical_potential_dpressure,
            derivatives.dreduced_chemical_potential_dpressure,
        ),
        (
            derivatives.dfugacity_dindependent_composition[:, 0],
            derivatives.dlog_fugacity_dindependent_composition[:, 0],
            derivatives.dchemical_potential_dindependent_composition[:, 0],
            derivatives.dreduced_chemical_potential_dindependent_composition[:, 0],
        ),
    )
    for coordinate, (step, expected_derivatives) in enumerate(zip(steps, expected, strict=True)):
        upper_arguments = list(base_arguments)
        lower_arguments = list(base_arguments)
        upper_arguments[coordinate] = upper_arguments[coordinate] + step
        lower_arguments[coordinate] = lower_arguments[coordinate] - step
        upper = values(*upper_arguments)
        lower = values(*lower_arguments)
        for upper_value, lower_value, expected_derivative in zip(
            upper,
            lower,
            expected_derivatives,
            strict=True,
        ):
            finite_difference = (upper_value - lower_value) / (2.0 * step)
            torch.testing.assert_close(
                finite_difference,
                expected_derivative,
                rtol=2.0e-7,
                atol=2.0e-8,
            )


def test_direct_free_energies_support_batched_states(binary_model):
    pressure = torch.tensor(
        [1.0e6, 3.0e6],
        dtype=torch.float64,
        requires_grad=True,
    )
    state = ChemicalState(
        torch.tensor([300.0, 340.0], dtype=torch.float64),
        pressure,
        torch.tensor([[0.8, 0.2], [0.4, 0.6]], dtype=torch.float64),
    )
    batched = phase_properties(binary_model, state, "vapor", caloric=False)
    assert batched.molar_gibbs_energy.shape == (2,)
    assert batched.molar_helmholtz_energy.shape == (2,)
    assert batched.reduced_gibbs_energy.shape == (2,)
    assert batched.reduced_helmholtz_energy.shape == (2,)
    assert batched.fugacities.shape == (2, 2)
    assert batched.log_fugacities.shape == (2, 2)
    assert batched.reduced_chemical_potentials.shape == (2, 2)

    for index in range(2):
        scalar = phase_properties(
            binary_model,
            ChemicalState(
                state.temperature[index],
                state.pressure[index],
                state.composition[index],
            ),
            "vapor",
            caloric=False,
        )
        torch.testing.assert_close(
            batched.molar_gibbs_energy[index],
            scalar.molar_gibbs_energy,
        )
        torch.testing.assert_close(
            batched.molar_helmholtz_energy[index],
            scalar.molar_helmholtz_energy,
        )
        torch.testing.assert_close(
            batched.fugacities[index],
            scalar.fugacities,
        )
        torch.testing.assert_close(
            batched.log_fugacities[index],
            scalar.log_fugacities,
        )
        torch.testing.assert_close(
            batched.reduced_chemical_potentials[index],
            scalar.reduced_chemical_potentials,
        )
    batched.molar_helmholtz_energy.sum().backward()
    assert pressure.grad is not None
    assert torch.isfinite(pressure.grad).all()


def test_state_derivative_shape_errors(binary_model):
    with pytest.raises(ValueError, match="scalar T-P"):
        state_derivatives(
            binary_model,
            ChemicalState(
                torch.tensor([300.0]),
                torch.tensor([1.0e5]),
                torch.tensor([0.5, 0.5]),
            ),
        )
    with pytest.raises(ValueError, match="composition vector"):
        state_derivatives(
            binary_model,
            ChemicalState(
                torch.tensor(300.0),
                torch.tensor(1.0e5),
                torch.tensor([[0.5, 0.5]]),
            ),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        state_derivatives(
            binary_model,
            ChemicalState(
                torch.tensor(300.0),
                torch.tensor(1.0e5),
                torch.tensor([1.0, 0.0]),
            ),
        )


@pytest.mark.serial
def test_tangent_plane_stability_stable_and_unstable(binary_model, two_phase_state):
    unstable = tangent_plane_stability(binary_model, two_phase_state)
    assert not unstable.stable
    assert unstable.converged
    assert unstable.minimum_tpd < 0.0
    stable_state = ChemicalState(
        torch.tensor(450.0, dtype=torch.float64),
        torch.tensor(1.0e5, dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    stable = tangent_plane_stability(
        binary_model,
        stable_state,
        initial_compositions=(stable_state.composition,),
    )
    assert stable.stable
    trust_region = tangent_plane_stability(
        binary_model,
        two_phase_state,
        minimizer="trust-region",
        max_iterations=80,
    )
    assert not trust_region.stable
    assert trust_region.converged
    assert trust_region.minimum_tpd < 0.0
    with pytest.raises(ValueError, match="local minimizer"):
        tangent_plane_stability(
            binary_model,
            stable_state,
            minimizer="unknown",
        )
    with pytest.raises(ValueError, match="composition vector"):
        tangent_plane_stability(
            binary_model,
            ChemicalState(
                torch.tensor(300.0),
                torch.tensor(1.0e5),
                torch.tensor([[0.5, 0.5]]),
            ),
        )


def test_stability_newton_uses_gradient_fallback_after_rejected_line_search():
    initial = torch.tensor([1.0], dtype=torch.float64)

    def objective(coordinates):
        return torch.where(
            coordinates[0] == initial[0],
            coordinates[0],
            coordinates.new_tensor(torch.nan),
        )

    coordinates, value, iterations, converged = _newton_minimize(
        objective,
        initial,
        tolerance=1.0e-12,
        max_iterations=1,
    )
    assert iterations == 1
    assert not converged
    assert coordinates[0] < initial[0]
    assert torch.isnan(value)


def test_batched_tangent_plane_stability_matches_scalar_classification(
    binary_model,
    two_phase_state,
):
    temperatures = torch.stack(
        (two_phase_state.temperature, torch.tensor(450.0, dtype=torch.float64))
    )
    pressures = torch.stack((two_phase_state.pressure, torch.tensor(1.0e5, dtype=torch.float64)))
    result = batched_tangent_plane_stability(
        binary_model,
        ChemicalState(temperatures, pressures, two_phase_state.composition),
        max_iterations=80,
    )
    assert result.stable.tolist() == [False, True]
    assert bool(result.converged.all())
    assert result.minimum_tpd[0] < 0.0
    assert result.minimum_tpd[1] >= -1.0e-6
    torch.testing.assert_close(
        result.trial_composition.sum(dim=-1),
        torch.ones(2, dtype=torch.float64),
    )
    assert result.iterations <= 80
    assert bool((result.residual_norm <= 1.0e-7).all())

    common_trials = torch.stack(
        (
            two_phase_state.composition,
            torch.tensor([0.95, 0.05], dtype=torch.float64),
        )
    )
    custom = batched_tangent_plane_stability(
        binary_model,
        ChemicalState(temperatures, pressures, two_phase_state.composition),
        initial_compositions=common_trials,
        max_iterations=80,
    )
    assert custom.minimum_tpd.shape == (2,)
    batched_trials = common_trials[None, :, :].expand(2, -1, -1)
    batched_custom = batched_tangent_plane_stability(
        binary_model,
        ChemicalState(temperatures, pressures, two_phase_state.composition),
        initial_compositions=batched_trials,
        max_iterations=1,
    )
    assert batched_custom.minimum_tpd.shape == (2,)
    assert batched_custom.iterations == 1


@pytest.mark.parametrize(
    ("trials", "match"),
    [
        (torch.ones((2, 3), dtype=torch.float64), "component count"),
        (torch.ones((1, 2, 2), dtype=torch.float64), "shape"),
        (torch.ones(2, dtype=torch.float64), "shape"),
        (torch.tensor([[0.5, -0.5]], dtype=torch.float64), "nonnegative"),
        (torch.zeros((1, 2), dtype=torch.float64), "positive sum"),
    ],
)
def test_batched_tangent_plane_stability_rejects_invalid_trials(
    binary_model,
    two_phase_state,
    trials,
    match,
):
    state = ChemicalState(
        two_phase_state.temperature.repeat(2),
        two_phase_state.pressure.repeat(2),
        two_phase_state.composition,
    )
    with pytest.raises(ValueError, match=match):
        batched_tangent_plane_stability(
            binary_model,
            state,
            initial_compositions=trials,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "match"),
    [
        ("tolerance", 0.0, "tolerance"),
        ("max_iterations", 0, "max_iterations"),
    ],
)
def test_batched_tangent_plane_stability_rejects_invalid_controls(
    binary_model,
    two_phase_state,
    keyword,
    value,
    match,
):
    state = ChemicalState(
        two_phase_state.temperature[None],
        two_phase_state.pressure[None],
        two_phase_state.composition,
    )
    with pytest.raises(ValueError, match=match):
        batched_tangent_plane_stability(binary_model, state, **{keyword: value})


def test_stability_newton_fallback_and_model_without_constants(monkeypatch):
    monkeypatch.setattr(
        torch.linalg,
        "solve",
        lambda *args, **kwargs: (_ for _ in ()).throw(torch.linalg.LinAlgError("singular")),
    )
    coordinates, _, _, converged = _newton_minimize(
        lambda value: (value - 1.0).square().sum(),
        torch.tensor([0.0], dtype=torch.float64),
        tolerance=1.0e-10,
        max_iterations=20,
    )
    assert converged
    torch.testing.assert_close(coordinates, torch.tensor([1.0], dtype=torch.float64))

    model = _NoConstantsModel()
    state = ChemicalState(
        torch.tensor(300.0, dtype=torch.float64),
        torch.tensor(1.0e5, dtype=torch.float64),
        torch.tensor([0.4, 0.6], dtype=torch.float64),
    )
    result = tangent_plane_stability(model, state, max_iterations=3)
    assert result.stable


@pytest.mark.parametrize(
    "baseline_name",
    ["thermopack_pr_flash.csv", "neqsim_pr_flash.csv"],
)
def test_two_phase_flash_external_regressions(binary_model, two_phase_state, baseline_name):
    result = two_phase_flash(binary_model, two_phase_state, check_stability=False)
    assert result.converged
    assert result.nphases == 2
    torch.testing.assert_close(
        result.phase_fractions.sum(),
        result.phase_fractions.new_tensor(1.0),
    )
    liquid, vapor = result.phases
    torch.testing.assert_close(
        liquid.composition * torch.exp(liquid.log_fugacity_coefficients),
        vapor.composition * torch.exp(vapor.log_fugacity_coefficients),
        rtol=2.0e-8,
        atol=1.0e-10,
    )
    with (DATA / baseline_name).open() as stream:
        baseline = next(csv.DictReader(stream))
    assert float(result.phase_fractions[1]) == pytest.approx(
        float(baseline["beta_vapor"]),
        rel=0.03,
    )
    assert float(liquid.composition[0]) == pytest.approx(
        float(baseline["x_methane"]),
        abs=0.02,
    )
    assert float(vapor.composition[0]) == pytest.approx(
        float(baseline["y_methane"]),
        abs=0.01,
    )


def test_two_phase_flash_stable_and_failure_paths(binary_model):
    stable_state = ChemicalState(
        torch.tensor(450.0, dtype=torch.float64),
        torch.tensor(1.0e5, dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    stable = two_phase_flash(binary_model, stable_state)
    assert stable.nphases == 1
    assert stable.stable

    state = ChemicalState(
        torch.tensor(270.0, dtype=torch.float64),
        torch.tensor(3.0e6, dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    with pytest.warns(ConvergenceWarning):
        failed = two_phase_flash(
            binary_model,
            state,
            check_stability=False,
            max_iterations=1,
        )
    assert not failed.converged
    with pytest.raises(RuntimeError, match="did not converge"):
        two_phase_flash(
            binary_model,
            state,
            check_stability=False,
            max_iterations=1,
            raise_on_failure=True,
        )
    with pytest.raises(ValueError, match="material-balance split"):
        two_phase_flash(
            binary_model,
            state,
            initial_k_values=torch.tensor([2.0, 3.0], dtype=torch.float64),
            check_stability=False,
            max_iterations=1,
        )
    with pytest.raises(ValueError, match="composition vector"):
        two_phase_flash(
            binary_model,
            ChemicalState(
                torch.tensor(300.0),
                torch.tensor(1.0e5),
                torch.tensor([[0.5, 0.5]]),
            ),
        )


def test_two_phase_flash_newton_fallbacks(binary_model, two_phase_state, monkeypatch):
    monkeypatch.setattr(
        torch.linalg,
        "solve",
        lambda *args, **kwargs: (_ for _ in ()).throw(torch.linalg.LinAlgError("singular")),
    )
    with pytest.warns(ConvergenceWarning):
        result = two_phase_flash(
            binary_model,
            two_phase_state,
            check_stability=False,
            max_iterations=14,
        )
    assert not result.converged

    monkeypatch.setattr(
        torch.linalg,
        "solve",
        lambda matrix, value: torch.zeros_like(value),
    )
    with pytest.warns(ConvergenceWarning):
        result = two_phase_flash(
            binary_model,
            two_phase_state,
            check_stability=False,
            max_iterations=13,
        )
    assert not result.converged


def test_saturation_points_and_envelope(binary_model):
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    temperature = torch.tensor(270.0, dtype=torch.float64)
    bubble = saturation_point(binary_model, temperature, composition, "bubble")
    dew = saturation_point(binary_model, temperature, composition, "dew")
    assert bubble.converged and dew.converged
    assert bubble.pressure > dew.pressure
    branches = phase_envelope(
        binary_model,
        torch.tensor([265.0, 270.0], dtype=torch.float64),
        composition,
    )
    assert set(branches) == {"bubble", "dew"}
    assert len(branches["bubble"]) == 2
    with pytest.raises(ValueError, match="unknown saturation"):
        saturation_point(binary_model, temperature, composition, "solid")
    with pytest.raises(ValueError, match="match composition"):
        saturation_point(
            binary_model,
            temperature,
            composition,
            "bubble",
            initial_pressure=torch.tensor(1.0e6, dtype=torch.float64),
            initial_k_values=torch.tensor([2.0], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        saturation_point(
            binary_model,
            temperature,
            composition,
            "bubble",
            initial_pressure=torch.tensor(1.0e6, dtype=torch.float64),
            initial_k_values=torch.tensor([2.0, -0.5], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="pressure must be finite and positive"):
        saturation_point(
            binary_model,
            temperature,
            composition,
            "bubble",
            initial_pressure=torch.tensor(-1.0, dtype=torch.float64),
        )


def test_phase_transition_pressure_matches_binary_bubble_point(binary_model):
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    temperature = torch.tensor(270.0, dtype=torch.float64)
    reference = binary_bubble_point(binary_model, temperature, composition)
    result = phase_transition_pressure(
        binary_model,
        temperature,
        composition,
        initial_pressure=reference.pressure * 0.9,
        initial_incipient_composition=reference.vapor_composition,
        minimum_pressure=reference.pressure * 0.5,
        maximum_pressure=reference.pressure * 1.5,
    )

    assert result.converged
    assert result.phase_kinds == ("liquid", "vapor")
    assert result.phase_separation > 0.4
    assert result.residual_norm <= 1.0e-8
    torch.testing.assert_close(result.pressure, reference.pressure, rtol=2.0e-9, atol=0.0)
    torch.testing.assert_close(
        result.incipient_composition,
        reference.vapor_composition,
        rtol=1.0e-8,
        atol=1.0e-10,
    )


def test_two_phase_trust_region_flash_matches_standard_split(
    binary_model,
    two_phase_state,
):
    reference = two_phase_flash(
        binary_model,
        two_phase_state,
        check_stability=False,
    )
    result = two_phase_trust_region_flash(
        binary_model,
        two_phase_state,
        check_stability=False,
        max_iterations=100,
    )

    assert result.converged
    assert result.residual_norm <= 1.0e-8
    torch.testing.assert_close(
        result.phase_fractions,
        reference.phase_fractions,
        rtol=2.0e-6,
        atol=2.0e-8,
    )
    for actual, expected in zip(result.phases, reference.phases, strict=True):
        torch.testing.assert_close(
            actual.composition,
            expected.composition,
            rtol=2.0e-6,
            atol=2.0e-8,
        )


def test_two_phase_trust_region_flash_validates_and_reports_failure(
    binary_model,
    two_phase_state,
):
    with pytest.raises(ValueError, match="composition vector"):
        two_phase_trust_region_flash(
            binary_model,
            ChemicalState(
                two_phase_state.temperature,
                two_phase_state.pressure,
                two_phase_state.composition[None],
            ),
        )
    with pytest.raises(ValueError, match="phase_roots"):
        two_phase_trust_region_flash(
            binary_model,
            two_phase_state,
            phase_roots=("liquid", "invalid"),
        )
    with pytest.raises(ValueError, match="controls"):
        two_phase_trust_region_flash(
            binary_model,
            two_phase_state,
            tolerance=0.0,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        two_phase_trust_region_flash(
            binary_model,
            ChemicalState(
                two_phase_state.temperature,
                two_phase_state.pressure,
                torch.tensor([1.0, 0.0], dtype=torch.float64),
            ),
        )
    with pytest.raises(ValueError, match="initial K values"):
        two_phase_trust_region_flash(
            binary_model,
            two_phase_state,
            initial_k_values=torch.tensor([1.0], dtype=torch.float64),
        )

    stable_state = ChemicalState(
        torch.tensor(450.0, dtype=torch.float64),
        torch.tensor(1.0e5, dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    stable = two_phase_trust_region_flash(binary_model, stable_state)
    assert stable.converged
    assert stable.nphases == 1
    assert stable.diagnostics["trust_region_stability"]

    with pytest.raises(RuntimeError, match="did not produce"):
        two_phase_trust_region_flash(
            binary_model,
            two_phase_state,
            check_stability=False,
            max_iterations=1,
            raise_on_failure=True,
        )
    with pytest.warns(ConvergenceWarning):
        failed = two_phase_trust_region_flash(
            binary_model,
            two_phase_state,
            check_stability=False,
            max_iterations=1,
        )
    assert not failed.converged


def test_batched_phase_transition_pressures_match_scalar_and_preserve_gradients(
    binary_components,
):
    interaction = torch.tensor(0.02, dtype=torch.float64, requires_grad=True)
    kij = interaction * torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float64,
    )
    model = peng_robinson_1978(binary_components, kij=kij)
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    temperatures = torch.tensor([265.0, 270.0], dtype=torch.float64)
    references = tuple(
        binary_bubble_point(model, temperature, composition) for temperature in temperatures
    )
    reference_pressures = torch.stack(tuple(reference.pressure for reference in references))
    reference_compositions = torch.stack(
        tuple(reference.vapor_composition for reference in references)
    )

    result = solve_batched_phase_transition_pressures(
        model,
        temperatures,
        composition.expand(2, -1),
        phase_kinds=("liquid", "vapor"),
        initial_pressure=0.9 * reference_pressures,
        initial_incipient_composition=reference_compositions,
        minimum_pressure=0.5 * reference_pressures,
        maximum_pressure=1.5 * reference_pressures,
    )

    assert result.converged.tolist() == [True, True]
    assert bool(result.solver_converged.all())
    torch.testing.assert_close(
        result.pressure,
        reference_pressures,
        rtol=3.0e-9,
        atol=0.0,
    )
    assert torch.isfinite(torch.autograd.grad(result.pressure.sum(), interaction)[0])


def test_batched_phase_transition_pressures_validate_batch_contract(binary_model):
    temperature = torch.tensor([270.0], dtype=torch.float64)
    parent = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    reference = binary_bubble_point(binary_model, temperature[0], parent[0])
    pressure = reference.pressure.reshape(1)
    incipient = reference.vapor_composition.reshape(1, 2)

    def solve(**overrides):
        arguments = {
            "phase_kinds": ("liquid", "vapor"),
            "initial_pressure": pressure,
            "initial_incipient_composition": incipient,
        }
        arguments.update(overrides)
        return solve_batched_phase_transition_pressures(
            binary_model,
            temperature,
            parent,
            **arguments,
        )

    with pytest.raises(ValueError, match="temperature"):
        solve_batched_phase_transition_pressures(
            binary_model,
            temperature[0],
            parent,
            phase_kinds=("liquid", "vapor"),
            initial_pressure=pressure,
            initial_incipient_composition=incipient,
        )
    with pytest.raises(ValueError, match="parent composition"):
        solve_batched_phase_transition_pressures(
            binary_model,
            temperature,
            parent[0],
            phase_kinds=("liquid", "vapor"),
            initial_pressure=pressure,
            initial_incipient_composition=incipient,
        )
    with pytest.raises(ValueError, match="at least two"):
        solve_batched_phase_transition_pressures(
            binary_model,
            temperature,
            parent[:, :1],
            phase_kinds=("liquid", "vapor"),
            initial_pressure=pressure,
            initial_incipient_composition=parent[:, :1],
        )
    with pytest.raises(ValueError, match="phase kinds"):
        solve(phase_kinds=("liquid", "invalid"))
    with pytest.raises(ValueError, match="controls"):
        solve(tolerance=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        solve(initial_pressure=torch.tensor([torch.nan], dtype=torch.float64))
    with pytest.raises(ValueError, match="not broadcastable"):
        solve(minimum_pressure=torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="finite and positive"):
        solve(minimum_pressure=-1.0)
    with pytest.raises(ValueError, match="below maximum"):
        solve(minimum_pressure=3.0e6, maximum_pressure=2.0e6)

    result = solve()
    assert result.converged.tolist() == [True]


def test_phase_transition_pressure_rejects_homogeneous_same_root(binary_model):
    composition = torch.tensor([0.4, 0.6], dtype=torch.float64)
    result = phase_transition_pressure(
        binary_model,
        torch.tensor(270.0, dtype=torch.float64),
        composition,
        phase_kinds=("liquid", "liquid"),
        initial_pressure=torch.tensor(3.0e6, dtype=torch.float64),
        initial_incipient_composition=composition,
        minimum_pressure=2.0e6,
        maximum_pressure=4.0e6,
    )

    assert not result.converged
    assert result.residual_norm <= 1.0e-8
    assert result.phase_separation <= 1.0e-6


def test_continue_phase_transition_branch_matches_phase_envelope(binary_model):
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    temperatures = torch.tensor([270.0, 271.0], dtype=torch.float64)
    reference = phase_envelope(
        binary_model,
        temperatures,
        composition,
        kinds=("bubble",),
        accelerated=False,
    )["bubble"]
    initial = phase_transition_pressure(
        binary_model,
        temperatures[0],
        composition,
        initial_pressure=reference[0].pressure,
        initial_incipient_composition=reference[0].incipient_composition,
    )

    continued = continue_phase_transition_branch(
        binary_model,
        temperatures,
        initial,
    )

    assert all(point.converged for point in continued)
    assert all(point.phase_kinds == ("liquid", "vapor") for point in continued)
    torch.testing.assert_close(
        torch.stack(tuple(point.pressure for point in continued)),
        torch.stack(tuple(point.pressure for point in reference)),
        rtol=2.0e-8,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("temperatures", "converged", "match"),
    [
        (torch.tensor([], dtype=torch.float64), True, "nonempty vector"),
        (torch.tensor([-1.0], dtype=torch.float64), True, "finite and positive"),
        (torch.tensor([270.0], dtype=torch.float64), False, "converged and separated"),
    ],
)
def test_continue_phase_transition_branch_validates_seed_and_grid(
    binary_model,
    temperatures,
    converged,
    match,
):
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    reference = binary_bubble_point(
        binary_model,
        torch.tensor(270.0, dtype=torch.float64),
        composition,
    )
    initial = envelope.PhaseTransitionPoint(
        torch.tensor(270.0, dtype=torch.float64),
        reference.pressure,
        composition,
        reference.vapor_composition,
        ("liquid", "vapor"),
        reference.iterations,
        converged,
        reference.residual_norm,
        torch.max(torch.abs(composition - reference.vapor_composition)),
    )

    with pytest.raises(ValueError, match=match):
        continue_phase_transition_branch(binary_model, temperatures, initial)


def test_continue_phase_transition_branch_records_model_failure(binary_model, monkeypatch):
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    reference = binary_bubble_point(
        binary_model,
        torch.tensor(270.0, dtype=torch.float64),
        composition,
    )
    initial = envelope.PhaseTransitionPoint(
        torch.tensor(270.0, dtype=torch.float64),
        reference.pressure,
        composition,
        reference.vapor_composition,
        ("liquid", "vapor"),
        reference.iterations,
        True,
        reference.residual_norm,
        torch.max(torch.abs(composition - reference.vapor_composition)),
    )

    def fail_transition(*args, **kwargs):
        raise RuntimeError("trial state failed")

    monkeypatch.setattr(envelope, "phase_transition_pressure", fail_transition)
    result = continue_phase_transition_branch(
        binary_model,
        torch.tensor([271.0], dtype=torch.float64),
        initial,
    )

    assert len(result) == 1
    assert not result[0].converged
    assert torch.isnan(result[0].pressure)
    assert torch.isinf(result[0].residual_norm)


def test_phase_transition_state_evaluation_and_envelope_set(binary_model):
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    temperature = torch.tensor(270.0, dtype=torch.float64)
    reference = binary_bubble_point(binary_model, temperature, composition)
    state = envelope.PhaseTransitionState(
        temperature,
        reference.pressure,
        composition,
        "liquid-vapor",
    )

    evaluation = evaluate_phase_transition_state(binary_model, state)
    evaluations = evaluate_phase_transition_states(binary_model, (state, state))
    phase_boundaries = trace_phase_envelope_set(
        binary_model,
        composition,
        torch.tensor([269.0, 270.0], dtype=torch.float64),
    )

    assert evaluation.converged
    assert evaluation.phase_compositions.shape == (1, 2)
    assert len(evaluations) == 2
    assert set(phase_boundaries.vapor_liquid) == {"bubble", "dew"}
    assert not phase_boundaries.liquid_liquid
    torch.testing.assert_close(evaluation.pressure, reference.pressure, rtol=2.0e-9, atol=0.0)


@pytest.mark.parametrize(
    ("state_updates", "options", "match"),
    [
        ({"boundary_kind": "solid-liquid"}, {}, "unknown phase-boundary"),
        ({}, {"tolerance": 0.0}, "controls"),
        ({"temperature": torch.tensor([-1.0], dtype=torch.float64)}, {}, "temperature"),
        ({"reference_pressure": torch.tensor(-1.0, dtype=torch.float64)}, {}, "pressure"),
        (
            {"parent_composition": torch.tensor([1.0, 0.0], dtype=torch.float64)},
            {},
            "composition",
        ),
    ],
)
def test_phase_transition_state_evaluation_validates_inputs(
    binary_model,
    state_updates,
    options,
    match,
):
    values = {
        "temperature": torch.tensor(270.0, dtype=torch.float64),
        "reference_pressure": torch.tensor(2.0e6, dtype=torch.float64),
        "parent_composition": torch.tensor([0.5, 0.5], dtype=torch.float64),
        "boundary_kind": "liquid-vapor",
    }
    values.update(state_updates)
    state = envelope.PhaseTransitionState(**values)

    with pytest.raises(ValueError, match=match):
        evaluate_phase_transition_state(binary_model, state, **options)


def test_phase_transition_state_evaluation_handles_three_phase_and_solver_failures(
    binary_model,
    monkeypatch,
):
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    temperature = torch.tensor(270.0, dtype=torch.float64)
    pressure = torch.tensor(2.0e6, dtype=torch.float64)
    starts = (
        torch.tensor([[0.02, 0.98], [0.70, 0.30], [0.99, 0.01]], dtype=torch.float64),
        torch.tensor([[0.10, 0.90], [0.80, 0.20], [0.995, 0.005]], dtype=torch.float64),
    )
    state = envelope.PhaseTransitionState(
        temperature,
        pressure,
        composition,
        "liquid-liquid-vapor",
        initial_three_phase_compositions=starts,
    )
    calls = 0

    def invariant_result(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("trial state failed")
        return BinaryThreePhaseInvariant(
            temperature,
            pressure * 1.1,
            starts[1],
            pressure.new_tensor(1.0e-10),
            3,
            True,
        )

    monkeypatch.setattr(envelope, "solve_binary_three_phase_invariant", invariant_result)
    result = evaluate_phase_transition_state(binary_model, state)

    assert result.converged
    assert result.phase_compositions.shape == (3, 2)
    assert result.phase_separation > 0.1
    trust_result = evaluate_phase_transition_state(
        binary_model,
        state,
        three_phase_solver="newton-trust-region",
    )
    assert trust_result.converged
    assert trust_result.solver == "trust-region"

    ternary_state = envelope.PhaseTransitionState(
        temperature,
        pressure,
        torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64),
        "liquid-liquid-vapor",
    )
    with pytest.raises(ValueError, match="binary"):
        evaluate_phase_transition_state(binary_model, ternary_state)

    two_phase_state = envelope.PhaseTransitionState(
        temperature,
        pressure,
        composition,
        "liquid-liquid",
        initial_incipient_compositions=(torch.tensor([0.1, 0.9], dtype=torch.float64),),
    )
    monkeypatch.setattr(
        envelope,
        "phase_transition_pressure",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("trial state failed")),
    )
    failed = evaluate_phase_transition_state(binary_model, two_phase_state)
    assert not failed.converged
    assert torch.isnan(failed.pressure)


def test_phase_transition_state_batch_validates_and_reports_failed_candidates(
    binary_model,
    monkeypatch,
):
    temperature = torch.tensor(270.0, dtype=torch.float64)
    pressure = torch.tensor(2.0e6, dtype=torch.float64)
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    liquid_state = envelope.PhaseTransitionState(
        temperature,
        pressure,
        composition,
        "liquid-liquid",
    )
    starts = envelope._default_two_phase_starts(liquid_state)
    assert len(starts) == 5
    torch.testing.assert_close(starts[0], torch.tensor([0.01, 0.99], dtype=torch.float64))
    torch.testing.assert_close(starts[-1], torch.tensor([0.99, 0.01], dtype=torch.float64))

    with pytest.raises(ValueError, match="controls"):
        evaluate_phase_transition_states(binary_model, (liquid_state,), tolerance=0.0)
    with pytest.raises(ValueError, match="three-phase"):
        evaluate_phase_transition_states(
            binary_model,
            (liquid_state,),
            three_phase_solver="invalid",
        )

    ternary_state = envelope.PhaseTransitionState(
        temperature,
        pressure,
        torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64),
        "liquid-liquid-vapor",
    )
    with pytest.raises(ValueError, match="binary"):
        evaluate_phase_transition_states(binary_model, (ternary_state,))

    unknown_state = envelope.PhaseTransitionState(
        temperature,
        pressure,
        composition,
        "solid-liquid",
    )
    with pytest.raises(ValueError, match="unknown"):
        evaluate_phase_transition_states(binary_model, (unknown_state,))

    vapor_state = envelope.PhaseTransitionState(
        temperature,
        pressure,
        composition,
        "liquid-vapor",
        initial_incipient_compositions=(torch.tensor([0.1, 0.9], dtype=torch.float64),),
    )
    monkeypatch.setattr(
        envelope,
        "solve_batched_phase_transition_pressures",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("trial state failed")),
    )
    failed = evaluate_phase_transition_states(binary_model, (vapor_state,))
    assert len(failed) == 1
    assert not failed[0].converged
    assert torch.isnan(failed[0].pressure)


def test_batched_three_phase_trust_region_recovers_remaining_starts(
    binary_model,
    monkeypatch,
):
    temperature = torch.tensor(270.0, dtype=torch.float64)
    pressure = torch.tensor(2.0e6, dtype=torch.float64)
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    states = tuple(
        envelope.PhaseTransitionState(
            temperature + index,
            pressure,
            composition,
            "liquid-liquid-vapor",
        )
        for index in range(2)
    )
    separated = torch.tensor(
        [[0.10, 0.90], [0.80, 0.20], [0.99, 0.01]],
        dtype=torch.float64,
    )
    newton_candidate = envelope.PhaseTransitionEvaluation(
        states[0],
        pressure,
        separated,
        pressure.new_tensor(1.0e-10),
        pressure.new_tensor(0.19),
        3,
        True,
        True,
    )
    candidate_groups = [[newton_candidate], []]
    calls = 0

    def batched_invariant(
        model,
        temperatures,
        initial_pressures,
        initial_compositions,
        **kwargs,
    ):
        nonlocal calls
        del model, initial_compositions, kwargs
        calls += 1
        count = temperatures.shape[0]
        phase_compositions = separated.expand(count, -1, -1).clone()
        converged = torch.ones(count, dtype=torch.bool)
        if calls == 1:
            phase_compositions[1] = torch.tensor(
                [[0.50, 0.50], [0.50, 0.50], [0.50, 0.50]],
                dtype=torch.float64,
            )
            converged[1] = False
        return BinaryThreePhaseInvariantBatch(
            temperatures,
            initial_pressures,
            phase_compositions,
            temperatures.new_full((count,), 1.0e-10),
            torch.ones(count, dtype=torch.int64),
            converged,
        )

    monkeypatch.setattr(
        envelope,
        "solve_batched_binary_three_phase_invariants",
        batched_invariant,
    )
    envelope._batched_trust_region_three_phase_candidates(
        binary_model,
        states,
        (0, 1),
        candidate_groups,
        tolerance=1.0e-8,
        max_iterations=20,
        minimum_phase_separation=2.0e-3,
    )

    assert calls == 2
    assert candidate_groups[0][-1].solver == "trust-region"
    assert candidate_groups[0][-1].converged
    assert not candidate_groups[1][0].converged
    assert candidate_groups[1][-1].converged


def test_scalar_three_phase_trust_region_reports_recovery_failure(
    binary_model,
    monkeypatch,
):
    state = envelope.PhaseTransitionState(
        torch.tensor(270.0, dtype=torch.float64),
        torch.tensor(2.0e6, dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        "liquid-liquid-vapor",
    )
    monkeypatch.setattr(
        envelope,
        "solve_binary_three_phase_invariant",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    assert (
        envelope._trust_region_three_phase_candidates(
            binary_model,
            state,
            (),
            tolerance=1.0e-8,
            max_iterations=20,
            minimum_phase_separation=2.0e-3,
        )
        == ()
    )


def test_phase_transition_state_evaluation_uses_lazy_vapor_fallback(
    binary_model,
    monkeypatch,
):
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    pressure = torch.tensor(2.0e6, dtype=torch.float64)
    starts = (
        torch.tensor([0.1, 0.9], dtype=torch.float64),
        torch.tensor([0.2, 0.8], dtype=torch.float64),
    )
    state = envelope.PhaseTransitionState(
        torch.tensor(270.0, dtype=torch.float64),
        pressure,
        composition,
        "liquid-vapor",
        initial_incipient_compositions=starts,
    )
    calls = 0

    def transition(*args, **kwargs):
        nonlocal calls
        index = calls
        calls += 1
        return envelope.PhaseTransitionPoint(
            state.temperature,
            pressure * (1.5 if index == 0 else 1.05),
            composition,
            starts[index],
            ("liquid", "vapor"),
            2,
            True,
            pressure.new_tensor(1.0e-10),
            pressure.new_tensor(0.4),
        )

    monkeypatch.setattr(envelope, "phase_transition_pressure", transition)
    lazy = evaluate_phase_transition_state(binary_model, state)
    assert calls == 1
    torch.testing.assert_close(lazy.pressure, pressure * 1.5)

    calls = 0
    exhaustive = evaluate_phase_transition_state(
        binary_model,
        state,
        exhaustive_two_phase_starts=True,
    )
    assert calls == 2
    torch.testing.assert_close(exhaustive.pressure, pressure * 1.05)


def test_trace_phase_envelope_set_orchestrates_distinct_liquid_branches(
    binary_model,
    monkeypatch,
):
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    temperatures = torch.tensor([269.0, 270.0, 271.0], dtype=torch.float64)
    state = envelope.PhaseTransitionState(
        temperatures[1],
        torch.tensor(2.0e6, dtype=torch.float64),
        composition,
        "liquid-liquid",
    )

    def evaluation(pressure):
        return envelope.PhaseTransitionEvaluation(
            state,
            torch.tensor(pressure, dtype=torch.float64),
            torch.tensor([[0.1, 0.9]], dtype=torch.float64),
            torch.tensor(1.0e-10, dtype=torch.float64),
            torch.tensor(0.4, dtype=torch.float64),
            3,
            True,
            True,
        )

    low_seed = evaluation(2.0e6)
    high_seed = evaluation(3.0e6)
    fake_vle = {"bubble": (), "dew": ()}
    monkeypatch.setattr(envelope, "phase_envelope", lambda *args, **kwargs: fake_vle)

    def continuation(model, requested_temperatures, initial_point, **kwargs):
        return tuple(
            envelope.PhaseTransitionPoint(
                temperature,
                initial_point.pressure,
                initial_point.parent_composition,
                initial_point.incipient_composition,
                initial_point.phase_kinds,
                1,
                True,
                initial_point.residual_norm,
                initial_point.phase_separation,
            )
            for temperature in requested_temperatures
        )

    monkeypatch.setattr(envelope, "continue_phase_transition_branch", continuation)
    result = trace_phase_envelope_set(
        binary_model,
        composition,
        temperatures,
        liquid_liquid_seeds=(high_seed, low_seed),
        liquid_liquid_temperatures=temperatures,
    )

    assert result.vapor_liquid == fake_vle
    assert len(result.liquid_liquid) == 2
    assert all(len(branch) == 3 for branch in result.liquid_liquid)

    with pytest.raises(ValueError, match="temperatures"):
        trace_phase_envelope_set(
            binary_model,
            composition,
            temperatures,
            liquid_liquid_seeds=(low_seed,),
        )

    mismatched_state = envelope.PhaseTransitionState(
        temperatures[1],
        torch.tensor(2.0e6, dtype=torch.float64),
        torch.tensor([0.4, 0.6], dtype=torch.float64),
        "liquid-liquid",
    )
    mismatched = envelope.PhaseTransitionEvaluation(
        mismatched_state,
        low_seed.pressure,
        low_seed.phase_compositions,
        low_seed.residual_norm,
        low_seed.phase_separation,
        3,
        True,
        True,
    )
    with pytest.raises(ValueError, match="must match"):
        trace_phase_envelope_set(
            binary_model,
            composition,
            temperatures,
            liquid_liquid_seeds=(mismatched,),
            liquid_liquid_temperatures=temperatures,
        )


def test_trace_phase_envelope_set_closes_vapor_liquid_branches(
    binary_model,
    monkeypatch,
):
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    temperature = torch.tensor(270.0, dtype=torch.float64)
    bubble = saturation_point(binary_model, temperature, composition, "bubble")
    dew = saturation_point(binary_model, temperature, composition, "dew")
    monkeypatch.setattr(
        envelope,
        "phase_envelope",
        lambda *args, **kwargs: {"bubble": (bubble,), "dew": (dew,)},
    )
    calls = []

    def closure(model, parent, seed, targets, **kwargs):
        calls.append((seed.kind, targets))
        return (
            envelope.SaturationPoint(
                seed.temperature + 1.0,
                seed.pressure,
                seed.incipient_composition,
                seed.k_values,
                seed.kind,
                1,
                True,
                seed.residual_norm,
            ),
        )

    monkeypatch.setattr(envelope, "continue_saturation_branch", closure)
    result = trace_phase_envelope_set(
        binary_model,
        composition,
        torch.tensor([270.0], dtype=torch.float64),
        vapor_liquid_closure_points=3,
    )

    assert len(calls) == 2
    assert all(targets.numel() == 3 for _, targets in calls)
    assert all(len(points) == 2 for points in result.vapor_liquid.values())

    with pytest.raises(ValueError, match="closure controls"):
        trace_phase_envelope_set(
            binary_model,
            composition,
            torch.tensor([270.0], dtype=torch.float64),
            vapor_liquid_closure_points=0,
        )


def test_cubic_phase_transition_fit_uses_public_high_level_api(binary_components):
    model = peng_robinson_1978(binary_components)
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    temperature = torch.tensor(270.0, dtype=torch.float64)
    reference = binary_bubble_point(model, temperature, composition)
    state = envelope.PhaseTransitionState(
        temperature,
        reference.pressure,
        composition,
        "liquid-vapor",
    )
    system = CubicInteractionFitSystem(binary_components, (0, 1), (state,))

    result = fit_cubic_phase_transition_interactions(
        peng_robinson_1978,
        (system,),
        torch.tensor([0.01], dtype=torch.float64),
        torch.tensor([-0.2], dtype=torch.float64),
        torch.tensor([0.2], dtype=torch.float64),
        kij_pairs=((0, 1),),
        learning_rate=1.0e-3,
        max_iterations=2,
    )

    assert result.parameters.shape == (1,)
    assert len(result.parameter_history) == 3
    assert result.selected_iteration in (0, 1, 2)
    assert result.optimizer_stopping_reason == "iteration-limit"
    assert result.sensitivity_matrix.shape == (1, 1)
    assert result.sensitivity_rank == 1
    assert torch.isfinite(result.parameters).all()


def test_cubic_phase_transition_fit_can_optimize_observed_states_simultaneously(
    binary_components,
):
    model = peng_robinson_1978(binary_components)
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    temperature = torch.tensor(270.0, dtype=torch.float64)
    reference = binary_bubble_point(model, temperature, composition)
    state = envelope.PhaseTransitionState(
        temperature,
        reference.pressure,
        composition,
        "liquid-vapor",
    )
    system = CubicInteractionFitSystem(binary_components, (0, 1), (state,))

    result = fit_cubic_phase_transition_interactions(
        peng_robinson_1978,
        (system,),
        torch.tensor([0.01], dtype=torch.float64),
        torch.tensor([-0.2], dtype=torch.float64),
        torch.tensor([0.2], dtype=torch.float64),
        kij_pairs=((0, 1),),
        objective="observed-state-fugacity",
        learning_rate=1.0e-2,
        max_iterations=4,
        no_improvement_patience=None,
        parameter_prior_weight=1.0,
    )

    fitted_state = result.fitted_states[0][0]
    fitted_composition = fitted_state.initial_incipient_compositions[0]
    assert result.calibration_objective == "observed-state-fugacity"
    assert result.optimizer == "adam"
    assert result.sensitivity_kind == "observed-state-fugacity"
    assert result.selected_loss <= result.parameter_history[0]
    assert torch.max(torch.abs(fitted_composition - composition)) > 2.0e-3
    assert result.sensitivity_matrix.shape == (2, 1)
    assert torch.isfinite(result.sensitivity_matrix).all()

    def residual(parameter):
        fitted_model = build_cubic_interaction_models(
            peng_robinson_1978,
            (system,),
            parameter,
            kij_pairs=((0, 1),),
        )[0]
        return (
            torch.log(composition)
            + fitted_model.log_fugacity_coefficients(
                temperature,
                reference.pressure,
                composition,
                "liquid",
            )
            - torch.log(fitted_composition)
            - fitted_model.log_fugacity_coefficients(
                temperature,
                reference.pressure,
                fitted_composition,
                "vapor",
            )
        ) / 2.0**0.5

    step = torch.tensor([1.0e-5], dtype=torch.float64)
    finite_difference = (
        residual(result.parameters + step) - residual(result.parameters - step)
    ) / (2.0 * step)
    torch.testing.assert_close(
        result.sensitivity_matrix[:, 0],
        finite_difference,
        rtol=2.0e-5,
        atol=2.0e-7,
    )


def test_cubic_observed_state_seed_and_residual_helpers_cover_fallbacks(
    binary_components,
):
    state = envelope.PhaseTransitionState(
        torch.tensor(270.0, dtype=torch.float64),
        torch.tensor(2.0e6, dtype=torch.float64),
        torch.tensor([0.8, 0.2], dtype=torch.float64),
        "liquid-vapor",
        initial_incipient_compositions=(torch.tensor([0.8, 0.2], dtype=torch.float64),),
    )
    failed_evaluation = envelope.PhaseTransitionEvaluation(
        state,
        state.reference_pressure,
        state.parent_composition[None],
        state.reference_pressure.new_tensor(torch.inf),
        state.reference_pressure.new_tensor(0.0),
        1,
        False,
        False,
    )
    seed = fitting_module._separated_two_phase_seed(
        state,
        failed_evaluation,
        minimum_phase_separation=2.0e-3,
    )
    assert torch.max(torch.abs(seed - state.parent_composition)) > 2.0e-3
    torch.testing.assert_close(seed.sum(), seed.new_tensor(1.0))

    ternary_components = component_set(
        ("methane", "carbon_dioxide", "n_butane"),
        dtype=torch.float64,
    )
    ternary_state = envelope.PhaseTransitionState(
        state.temperature,
        state.reference_pressure,
        torch.tensor([0.4, 0.3, 0.3], dtype=torch.float64),
        "liquid-vapor",
        initial_incipient_compositions=(torch.tensor([0.1, 0.6, 0.3], dtype=torch.float64),),
    )
    ternary_system = CubicInteractionFitSystem(
        ternary_components,
        (0, 1, 2),
        (ternary_state,),
    )
    batch = fitting_module._build_cubic_observed_state_batches(
        (ternary_system,),
        minimum_phase_separation=2.0e-3,
    )[0]
    encoded = fitting_module._encode_separated_composition(
        batch,
        minimum_phase_separation=2.0e-3,
    )
    decoded = fitting_module._decode_separated_composition(
        batch,
        encoded,
        minimum_phase_separation=2.0e-3,
    )
    torch.testing.assert_close(decoded, batch.initial_incipient_composition)

    with pytest.raises(ValueError, match="at least one two-phase"):
        fitting_module._cubic_observed_state_fugacity_loss(
            peng_robinson_1978,
            (CubicInteractionFitSystem(binary_components, (0, 1), ()),),
            (),
            torch.zeros(1, dtype=torch.float64),
            (),
            kij_pairs=((0, 1),),
            lij_pairs=(),
            minimum_phase_separation=2.0e-3,
        )


def test_cubic_fit_batches_three_phase_invariants_and_reports_solver_failure(
    binary_components,
    monkeypatch,
):
    temperature = torch.tensor(270.0, dtype=torch.float64)
    pressure = torch.tensor(2.0e6, dtype=torch.float64)
    phase_compositions = torch.tensor(
        [[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]],
        dtype=torch.float64,
    )
    state = envelope.PhaseTransitionState(
        temperature,
        pressure,
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        "liquid-liquid-vapor",
        initial_three_phase_compositions=(phase_compositions,),
    )
    system = CubicInteractionFitSystem(binary_components, (0, 1), (state,))
    batches = fitting_module._build_cubic_transition_batches((system,))
    model = peng_robinson_1978(binary_components)

    def solved_invariants(
        model,
        temperatures,
        initial_pressures,
        initial_compositions,
        **kwargs,
    ):
        count = temperatures.shape[0]
        return BinaryThreePhaseInvariantBatch(
            temperatures,
            initial_pressures,
            initial_compositions,
            temperatures.new_full((count,), 1.0e-10),
            torch.ones(count, dtype=torch.int64),
            torch.ones(count, dtype=torch.bool),
        )

    monkeypatch.setattr(
        fitting_module,
        "solve_batched_binary_three_phase_invariants",
        solved_invariants,
    )
    evaluation = fitting_module._evaluate_cubic_transition_batches(
        (model,),
        batches,
        tolerance=1.0e-8,
        max_iterations=20,
        minimum_phase_separation=2.0e-3,
    )[0]
    assert evaluation.converged.tolist() == [True]
    torch.testing.assert_close(
        evaluation.phase_separation,
        torch.tensor([0.4], dtype=torch.float64),
    )

    monkeypatch.setattr(
        fitting_module,
        "solve_batched_binary_three_phase_invariants",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("trial state failed")),
    )
    failed = fitting_module._evaluate_cubic_transition_batches(
        (model,),
        batches,
        tolerance=1.0e-8,
        max_iterations=20,
        minimum_phase_separation=2.0e-3,
    )[0]
    assert failed.converged.tolist() == [False]
    assert torch.isnan(failed.pressure).all()
    assert torch.isinf(failed.residual_norm).all()


def test_cubic_interaction_model_and_fit_validate_inputs(binary_components):
    state = envelope.PhaseTransitionState(
        torch.tensor(270.0, dtype=torch.float64),
        torch.tensor(2.0e6, dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        "liquid-vapor",
    )
    system = CubicInteractionFitSystem(binary_components, (0, 1), (state,))
    parameters = torch.tensor([0.01], dtype=torch.float64)

    invalid_builds = (
        ((system,), torch.tensor([0.01, 0.02]), ((0, 1),), (), "vector"),
        ((), parameters, ((0, 1),), (), "at least one"),
        ((system,), parameters, ((0, 2),), (), "indices"),
        (
            (system,),
            torch.tensor([0.01, 0.02]),
            ((0, 1), (0, 1)),
            (),
            "unique",
        ),
        (
            (CubicInteractionFitSystem(binary_components, (0, 0), (state,)),),
            torch.empty(0, dtype=torch.float64),
            (),
            (),
            "uniquely match",
        ),
    )
    for systems, values, kij_pairs, lij_pairs, match in invalid_builds:
        with pytest.raises(ValueError, match=match):
            build_cubic_interaction_models(
                peng_robinson_1978,
                systems,
                values,
                kij_pairs=kij_pairs,
                lij_pairs=lij_pairs,
            )

    simultaneous = build_cubic_interaction_models(
        peng_robinson_1978,
        (system,),
        torch.tensor([0.01, 0.02], dtype=torch.float64),
        kij_pairs=((0, 1),),
        lij_pairs=((0, 1),),
    )
    assert len(simultaneous) == 1

    common = {
        "constructor": peng_robinson_1978,
        "systems": (system,),
        "initial_parameters": parameters,
        "lower_bounds": torch.tensor([-0.2], dtype=torch.float64),
        "upper_bounds": torch.tensor([0.2], dtype=torch.float64),
        "kij_pairs": ((0, 1),),
        "max_iterations": 1,
    }
    invalid_fits = (
        ({"upper_bounds": torch.tensor([0.2, 0.3], dtype=torch.float64)}, "equally shaped"),
        ({"initial_parameters": torch.tensor([0.3], dtype=torch.float64)}, "strictly inside"),
        (
            {
                "systems": (
                    CubicInteractionFitSystem(
                        binary_components,
                        (0, 1),
                        (
                            envelope.PhaseTransitionState(
                                state.temperature,
                                state.reference_pressure,
                                state.parent_composition,
                                "liquid-liquid-vapor",
                            ),
                        ),
                    ),
                )
            },
            "three-phase fit states require",
        ),
        ({"learning_rate": 0.0}, "controls"),
        ({"no_improvement_patience": 0}, "controls"),
        ({"parameter_prior_weight": -1.0}, "controls"),
        ({"objective": "unknown"}, "objective"),
        ({"optimizer": "unknown"}, "optimizer"),
    )
    for updates, match in invalid_fits:
        arguments = {**common, **updates}
        with pytest.raises(ValueError, match=match):
            fit_cubic_phase_transition_interactions(**arguments)

    empty_system = CubicInteractionFitSystem(binary_components, (0, 1), ())
    with pytest.raises(ValueError, match="at least one phase-transition"):
        fit_cubic_phase_transition_interactions(
            peng_robinson_1978,
            (empty_system,),
            parameters,
            common["lower_bounds"],
            common["upper_bounds"],
            kij_pairs=((0, 1),),
            max_iterations=1,
        )

    three_phase_with_start = envelope.PhaseTransitionState(
        state.temperature,
        state.reference_pressure,
        state.parent_composition,
        "liquid-liquid-vapor",
        initial_three_phase_compositions=(
            torch.tensor(
                [[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]],
                dtype=torch.float64,
            ),
        ),
    )
    with pytest.raises(ValueError, match="only two-phase"):
        fit_cubic_phase_transition_interactions(
            peng_robinson_1978,
            (
                CubicInteractionFitSystem(
                    binary_components,
                    (0, 1),
                    (three_phase_with_start,),
                ),
            ),
            parameters,
            common["lower_bounds"],
            common["upper_bounds"],
            kij_pairs=((0, 1),),
            objective="observed-state-fugacity",
            max_iterations=1,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"phase_kinds": ("liquid", "solid")}, "phase kinds"),
        (
            {
                "phase_kinds": ("liquid", "liquid"),
                "initial_pressure": torch.tensor(1.0e6, dtype=torch.float64),
            },
            "initial incipient composition",
        ),
        (
            {"phase_kinds": ("liquid", "liquid")},
            "initial pressure",
        ),
        (
            {
                "minimum_pressure": 2.0e6,
                "maximum_pressure": 1.0e6,
            },
            "minimum phase-transition pressure",
        ),
        ({"minimum_phase_separation": -1.0}, "minimum phase separation"),
        ({"temperature": torch.tensor([-1.0], dtype=torch.float64)}, "temperature"),
        ({"tolerance": 0.0}, "tolerance"),
        (
            {"parent_composition": torch.tensor([1.0], dtype=torch.float64)},
            "multicomponent",
        ),
        (
            {"parent_composition": torch.tensor([1.0, 0.0], dtype=torch.float64)},
            "strictly positive",
        ),
        (
            {"initial_pressure": torch.tensor(-1.0, dtype=torch.float64)},
            "initial phase-transition pressure",
        ),
        (
            {"initial_incipient_composition": torch.tensor([1.0], dtype=torch.float64)},
            "initial incipient composition",
        ),
        ({"minimum_pressure": -1.0}, "minimum phase-transition pressure"),
    ],
)
def test_phase_transition_pressure_validates_inputs(
    binary_model,
    kwargs,
    match,
):
    temperature = kwargs.pop(
        "temperature",
        torch.tensor(270.0, dtype=torch.float64),
    )
    parent_composition = kwargs.pop(
        "parent_composition",
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match=match):
        phase_transition_pressure(
            binary_model,
            temperature,
            parent_composition,
            **kwargs,
        )


def test_phase_transition_pressure_requires_explicit_guesses_without_constants(
    binary_model,
    monkeypatch,
):
    monkeypatch.setattr(
        binary_model,
        "critical_temperature",
        torch.full_like(binary_model.critical_temperature, torch.nan),
    )
    with pytest.raises(ValueError, match="critical constants"):
        phase_transition_pressure(
            binary_model,
            torch.tensor(270.0, dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
        )


def test_binary_bubble_temperature_recovers_isothermal_bubble_point(binary_model):
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    expected_temperature = torch.tensor(270.0, dtype=torch.float64)
    reference = binary_bubble_point(binary_model, expected_temperature, composition)

    recovered = binary_bubble_temperature(
        binary_model,
        reference.pressure,
        composition,
    )
    bounded = binary_bubble_temperature(
        binary_model,
        reference.pressure,
        composition,
        initial_temperature=torch.tensor(260.0, dtype=torch.float64),
        initial_vapor_composition=reference.vapor_composition,
        minimum_temperature=240.0,
        maximum_temperature=torch.tensor(300.0, dtype=torch.float64),
    )

    assert recovered.converged and bounded.converged
    assert recovered.phase_separation > 0.4
    torch.testing.assert_close(recovered.temperature, expected_temperature, rtol=2.0e-9, atol=0.0)
    torch.testing.assert_close(
        recovered.vapor_composition,
        reference.vapor_composition,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    torch.testing.assert_close(bounded.temperature, expected_temperature)

    rejected = binary_bubble_temperature(
        binary_model,
        reference.pressure,
        composition,
        initial_temperature=expected_temperature,
        initial_vapor_composition=reference.vapor_composition,
        minimum_phase_separation=1.0,
    )
    assert not rejected.converged
    assert rejected.residual_norm < 1.0e-8


def test_binary_bubble_temperature_uses_critical_weighted_wilson_fallback(
    binary_model,
    monkeypatch,
):
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    reference = binary_bubble_point(
        binary_model,
        torch.tensor(270.0, dtype=torch.float64),
        composition,
    )
    monkeypatch.setattr(
        envelope,
        "wilson_k_values",
        lambda components, temperature, pressure: torch.full_like(
            components.critical_temperature,
            2.0,
        ),
    )
    result = binary_bubble_temperature(
        binary_model,
        reference.pressure,
        composition,
        max_iterations=1,
    )
    assert torch.isfinite(result.temperature)


@pytest.mark.parametrize(
    ("pressure", "composition", "options", "match"),
    [
        (
            torch.tensor(-1.0, dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            {},
            "pressure",
        ),
        (
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64),
            {},
            "composition",
        ),
        (
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            {},
            "composition",
        ),
        (
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            {"tolerance": 0.0},
            "solver controls",
        ),
        (
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            {"minimum_phase_separation": -1.0},
            "phase separation",
        ),
        (
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            {"initial_temperature": torch.tensor(-1.0, dtype=torch.float64)},
            "initial.*temperature",
        ),
        (
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            {"initial_vapor_composition": torch.tensor([1.0, 0.0], dtype=torch.float64)},
            "vapor composition",
        ),
        (
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            {"minimum_temperature": -1.0},
            "minimum.*temperature",
        ),
        (
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            {"maximum_temperature": torch.tensor([300.0], dtype=torch.float64)},
            "maximum.*temperature",
        ),
        (
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            {"minimum_temperature": 300.0, "maximum_temperature": 200.0},
            "below maximum",
        ),
    ],
)
def test_binary_bubble_temperature_rejects_invalid_inputs(
    binary_model,
    pressure,
    composition,
    options,
    match,
):
    with pytest.raises(ValueError, match=match):
        binary_bubble_temperature(binary_model, pressure, composition, **options)


def test_generic_binary_pxy_and_txy_traces_use_physical_continuation(binary_model):
    temperature = torch.tensor(270.0, dtype=torch.float64)
    fractions = torch.tensor([0.45, 0.50, 0.55], dtype=torch.float64)
    reference = binary_bubble_point(
        binary_model,
        temperature,
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )

    pxy = trace_binary_pxy_isotherm(binary_model, temperature, fractions)
    txy = trace_binary_txy_isobar(binary_model, reference.pressure, fractions)

    assert bool(pxy.converged.all()) and bool(txy.converged.all())
    assert pxy.pressure.shape == fractions.shape
    assert txy.temperature.shape == fractions.shape
    assert pxy.liquid_composition.shape == (3, 2)
    assert txy.vapor_composition.shape == (3, 2)
    assert pxy.iterations.dtype == torch.int64
    assert txy.converged.dtype == torch.bool
    torch.testing.assert_close(pxy.temperature, temperature)
    torch.testing.assert_close(txy.pressure, reference.pressure)
    torch.testing.assert_close(txy.temperature[1], temperature, rtol=2.0e-9, atol=0.0)

    rejected_pxy = trace_binary_pxy_isotherm(
        binary_model,
        temperature,
        fractions[:1],
        initial_pressure=reference.pressure,
        initial_vapor_composition=reference.vapor_composition,
        minimum_phase_separation=1.0,
    )
    rejected_txy = trace_binary_txy_isobar(
        binary_model,
        reference.pressure,
        fractions[:1],
        initial_temperature=temperature,
        initial_vapor_composition=reference.vapor_composition,
        minimum_phase_separation=1.0,
    )
    assert not bool(rejected_pxy.converged.any())
    assert not bool(rejected_txy.converged.any())


@pytest.mark.parametrize(
    ("function", "fixed_value", "fractions", "options", "match"),
    [
        (
            trace_binary_pxy_isotherm,
            torch.tensor(-1.0, dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            {},
            "temperature",
        ),
        (
            trace_binary_pxy_isotherm,
            torch.tensor(270.0, dtype=torch.float64),
            torch.tensor([], dtype=torch.float64),
            {},
            "fractions",
        ),
        (
            trace_binary_pxy_isotherm,
            torch.tensor(270.0, dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            {"minimum_phase_separation": -1.0},
            "phase separation",
        ),
        (
            trace_binary_txy_isobar,
            torch.tensor(-1.0, dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            {},
            "pressure",
        ),
        (
            trace_binary_txy_isobar,
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
            {},
            "fractions",
        ),
        (
            trace_binary_txy_isobar,
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            {"minimum_phase_separation": -1.0},
            "phase separation",
        ),
    ],
)
def test_generic_binary_composition_traces_reject_invalid_inputs(
    binary_model,
    function,
    fixed_value,
    fractions,
    options,
    match,
):
    with pytest.raises(ValueError, match=match):
        function(binary_model, fixed_value, fractions, **options)


def test_full_phase_envelope_against_thermopack_baseline(binary_model):
    with (DATA / "thermopack_pr_phase_envelope.csv").open() as stream:
        rows = tuple(csv.DictReader(stream))
    temperatures = torch.tensor(
        [float(row["temperature_K"]) for row in rows],
        dtype=torch.float64,
    )
    result = phase_envelope(
        binary_model,
        temperatures,
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    predicted_pressure = torch.stack(
        tuple(point.pressure for kind in ("bubble", "dew") for point in result[kind])
    )
    reference_pressure = torch.tensor(
        [
            *[float(row["bubble_pressure_Pa"]) for row in rows],
            *[float(row["dew_pressure_Pa"]) for row in rows],
        ],
        dtype=torch.float64,
    )
    predicted_composition = torch.stack(
        tuple(
            point.incipient_composition[0] for kind in ("bubble", "dew") for point in result[kind]
        )
    )
    reference_composition = torch.tensor(
        [
            *[float(row["bubble_y_methane"]) for row in rows],
            *[float(row["dew_x_methane"]) for row in rows],
        ],
        dtype=torch.float64,
    )
    # Independent component databases account for the systematic offset.
    torch.testing.assert_close(
        predicted_pressure,
        reference_pressure,
        rtol=5.0e-2,
        atol=0.0,
    )
    torch.testing.assert_close(
        predicted_composition,
        reference_composition,
        rtol=3.2e-2,
        atol=1.0e-5,
    )


class _NoConstantsModel:
    def log_fugacity_coefficients(self, temperature, pressure, composition, phase="stable"):
        return torch.zeros_like(composition)

    def select_z(self, temperature, pressure, composition, phase="stable"):
        return temperature.new_tensor(1.0)

    def molar_volume(self, temperature, pressure, composition, phase="stable"):
        return 8.31446261815324 * temperature / pressure


def test_models_without_initialization_constants(two_phase_state):
    model = _NoConstantsModel()
    with pytest.raises(ValueError, match="critical constants"):
        saturation_point(
            model,
            two_phase_state.temperature,
            two_phase_state.composition,
            "bubble",
        )
    with pytest.raises(ValueError, match="initial K values"):
        two_phase_flash(model, two_phase_state, check_stability=False, max_iterations=1)
    with pytest.raises(ValueError, match="initial K values"):
        _default_multiphase_k(model, two_phase_state)


class _FixedKModel(_NoConstantsModel):
    names = ("a", "b")
    critical_temperature = torch.tensor([200.0, 400.0], dtype=torch.float64)
    critical_pressure = torch.tensor([5.0e6, 4.0e6], dtype=torch.float64)
    acentric_factor = torch.tensor([0.1, 0.2], dtype=torch.float64)
    log_k = torch.log(torch.tensor([2.0, 0.5], dtype=torch.float64))

    def log_fugacity_coefficients(self, temperature, pressure, composition, phase="stable"):
        return self.log_k if phase == "liquid" else torch.zeros_like(composition)


class _ExplosiveSubstitutionModel(_FixedKModel):
    log_k = torch.tensor([100.0, -100.0], dtype=torch.float64)

    def __init__(self):
        self.maximum_pressure = 50.0 * float(torch.max(self.critical_pressure))
        self.maximum_evaluated_pressure = 0.0

    def log_fugacity_coefficients(self, temperature, pressure, composition, phase="stable"):
        pressure_value = float(pressure.detach())
        self.maximum_evaluated_pressure = max(
            self.maximum_evaluated_pressure,
            pressure_value,
        )
        if pressure_value > self.maximum_pressure * (1.0 + 1.0e-12):
            raise RuntimeError("saturation initializer escaped its pressure bounds")
        return super().log_fugacity_coefficients(
            temperature,
            pressure,
            composition,
            phase,
        )


class _NonfiniteConstantsModel(_FixedKModel):
    critical_temperature = torch.tensor([float("nan"), 400.0], dtype=torch.float64)


def test_saturation_nonfinite_constants_and_exact_initial_k():
    temperature = torch.tensor(300.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="finite critical constants"):
        saturation_point(
            _NonfiniteConstantsModel(),
            temperature,
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            "bubble",
        )

    model = _FixedKModel()
    point = saturation_point(
        model,
        temperature,
        torch.tensor([1.0 / 3.0, 2.0 / 3.0], dtype=torch.float64),
        "bubble",
        initial_pressure=torch.tensor(1.0e5, dtype=torch.float64),
        initial_k_values=torch.exp(model.log_k),
    )
    assert point.converged
    assert point.residual_norm < 1.0e-12


def test_saturation_substitution_is_projected_to_newton_bounds():
    model = _ExplosiveSubstitutionModel()
    point = saturation_point(
        model,
        torch.tensor(300.0, dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        "bubble",
        initial_pressure=torch.tensor(1.0e5, dtype=torch.float64),
        initial_k_values=torch.ones(2, dtype=torch.float64),
        substitution_iterations=2,
        max_iterations=2,
    )
    assert torch.isfinite(point.pressure)
    assert torch.isfinite(point.incipient_composition).all()
    assert model.maximum_evaluated_pressure <= model.maximum_pressure * (1.0 + 1.0e-12)


def test_multiphase_flash_fixed_phase_count_and_warnings():
    model = _FixedKModel()
    state = ChemicalState(
        torch.tensor(300.0, dtype=torch.float64),
        torch.tensor(1.0e5, dtype=torch.float64),
        torch.tensor([0.4, 0.6], dtype=torch.float64),
    )
    with pytest.warns(ExperimentalModelWarning):
        result = multiphase_flash(
            model,
            state,
            initial_k_values=torch.tensor([[2.0, 0.5]], dtype=torch.float64),
        )
    assert result.converged
    assert result.nphases == 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        failed = multiphase_flash(
            model,
            state,
            initial_k_values=torch.tensor([[1.1, 0.9]], dtype=torch.float64),
            max_iterations=1,
        )
    assert not failed.converged
    assert any(item.category is ExperimentalModelWarning for item in caught)
    assert any(item.category is ConvergenceWarning for item in caught)
    with pytest.raises(ValueError, match="composition vector"):
        multiphase_flash(
            model,
            ChemicalState(
                torch.tensor(300.0),
                torch.tensor(1.0e5),
                torch.tensor([[0.5, 0.5]]),
            ),
        )


def test_default_multiphase_initialization_water_branch():
    model = peng_robinson_1978(component_set(("methane", "water")))
    state = ChemicalState(
        torch.tensor(350.0, dtype=torch.float64),
        torch.tensor(1.0e6, dtype=torch.float64),
        torch.tensor([0.8, 0.2], dtype=torch.float64),
    )
    values = _default_multiphase_k(model, state)
    assert values.shape == (2, 2)
    assert values[1, 1] == 1.0e5
    hydrocarbon = peng_robinson_1978(component_set(("methane", "n_butane")))
    reciprocal = _default_multiphase_k(
        hydrocarbon,
        ChemicalState(
            torch.tensor(300.0, dtype=torch.float64),
            torch.tensor(1.0e6, dtype=torch.float64),
            torch.tensor([0.8, 0.2], dtype=torch.float64),
        ),
    )
    torch.testing.assert_close(reciprocal[1], reciprocal[0].reciprocal())


def test_multiphase_late_iteration_damping():
    model = _FixedKModel()
    state = ChemicalState(
        torch.tensor(300.0, dtype=torch.float64),
        torch.tensor(1.0e5, dtype=torch.float64),
        torch.tensor([0.4, 0.6], dtype=torch.float64),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = multiphase_flash(
            model,
            state,
            initial_k_values=torch.tensor([[3.0, 0.4]], dtype=torch.float64),
            tolerance=0.0,
            max_iterations=21,
        )
    assert result.iterations == 21
