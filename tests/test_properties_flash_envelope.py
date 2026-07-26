from __future__ import annotations

import csv
import warnings
from pathlib import Path

import pytest
import torch

from torch_flash import (
    ChemicalState,
    batched_tangent_plane_stability,
    binary_bubble_point,
    binary_bubble_temperature,
    binary_phase_equilibrium_point,
    component_set,
    fixed_vapor_ratio_vle_point,
    peng_robinson_1978,
    phase_properties,
    state_derivatives,
    tangent_plane_stability,
    trace_binary_pxy_isotherm,
    trace_binary_txy_isobar,
    two_phase_flash,
)
from torch_flash.constants import STANDARD_PRESSURE, R
from torch_flash.envelope import phase_envelope, saturation_point
from torch_flash.exceptions import ConvergenceWarning, ExperimentalModelWarning
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
    with pytest.raises(ValueError, match="composition vector"):
        tangent_plane_stability(
            binary_model,
            ChemicalState(
                torch.tensor(300.0),
                torch.tensor(1.0e5),
                torch.tensor([[0.5, 0.5]]),
            ),
        )


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
