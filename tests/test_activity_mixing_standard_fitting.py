from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from torch_flash import (
    ChemicalState,
    activity_model,
    binary_vle_point,
    component_set,
    peng_robinson_1978,
    soave_redlich_kwong,
    two_phase_flash,
)
from torch_flash.activity import (
    NRTL,
    AnchoredHuronVidalNRTL,
    HuronVidalNRTL,
    Wilson,
)
from torch_flash.constants import R
from torch_flash.eos.cubic import SRK
from torch_flash.fitting import (
    fit_parameters,
    least_squares_loss,
    phase_equilibrium_residual,
)
from torch_flash.mixing import (
    HuronVidalMixing,
    QuadraticMixing,
    TemperatureDependentQuadraticMixing,
)
from torch_flash.standard_state import IdealGasPolynomial


def test_nrtl_ideal_and_trainable_gradient():
    zeros = torch.zeros((2, 2), dtype=torch.float64)
    model = NRTL(zeros, torch.full_like(zeros, 0.3), trainable=True)
    temperature = torch.tensor(300.0, dtype=torch.float64)
    composition = torch.tensor([0.4, 0.6], dtype=torch.float64)
    torch.testing.assert_close(
        model.excess_gibbs_rt(temperature, composition),
        temperature.new_tensor(0.0),
    )
    torch.testing.assert_close(
        model.log_activity_coefficients(temperature, composition),
        torch.zeros(2, dtype=torch.float64),
    )
    model.excess_gibbs_rt(temperature, composition).backward()
    assert model.interaction.grad is not None


def test_nrtl_validation_and_buffer_mode():
    with pytest.raises(ValueError, match="equally sized"):
        NRTL(torch.zeros(2), torch.zeros((2, 2)))
    model = NRTL(torch.zeros((2, 2)), torch.zeros((2, 2)))
    assert "interaction" in dict(model.named_buffers())


def test_wilson_ideal_and_validation():
    temperature = torch.tensor(320.0, dtype=torch.float64)
    composition = torch.tensor([0.25, 0.75], dtype=torch.float64)
    volumes = torch.ones(2, dtype=torch.float64)
    model = Wilson(torch.zeros((2, 2), dtype=torch.float64), volumes, trainable=True)
    torch.testing.assert_close(model.lambda_matrix(temperature), temperature.new_ones((2, 2)))
    torch.testing.assert_close(
        model.excess_gibbs_rt(temperature, composition),
        temperature.new_tensor(0.0),
    )
    torch.testing.assert_close(
        model.log_activity_coefficients(temperature, composition),
        torch.zeros(2, dtype=torch.float64),
    )
    assert isinstance(model.interaction, nn.Parameter)
    with pytest.raises(ValueError, match="square"):
        Wilson(torch.zeros(2), volumes)
    with pytest.raises(ValueError, match="one Wilson"):
        Wilson(torch.zeros((2, 2)), torch.ones(3))


def test_quadratic_mixing_and_trainable_symmetry():
    kij = torch.tensor([[0.0, 0.1], [0.1, 0.0]], dtype=torch.float64)
    rule = QuadraticMixing(kij, trainable=True)
    with torch.no_grad():
        rule.raw_kij[0, 1] = 0.2
    torch.testing.assert_close(
        rule.kij,
        kij.new_tensor([[0.0, 0.15], [0.15, 0.0]]),
    )
    a = torch.tensor([1.0, 4.0], dtype=torch.float64)
    b = torch.tensor([0.1, 0.2], dtype=torch.float64)
    x = torch.tensor([0.25, 0.75], dtype=torch.float64)
    am, bm = rule(torch.tensor(300.0), x, a, b)
    torch.testing.assert_close(bm, torch.tensor(0.175, dtype=torch.float64))
    torch.testing.assert_close(am, torch.einsum("i,ij,j", x, rule.cross_a(a), x))
    with pytest.raises(ValueError, match="square"):
        QuadraticMixing(torch.zeros(2))
    with pytest.raises(ValueError, match="symmetric"):
        QuadraticMixing(torch.tensor([[0.0, 1.0], [0.0, 0.0]]))


def test_temperature_dependent_quadratic_mixing_validation_and_cross_parameters():
    matrix = torch.tensor([[0.0, 0.1], [0.1, 0.0]], dtype=torch.float64)
    slope = torch.tensor([[0.0, 30.0], [30.0, 0.0]], dtype=torch.float64)
    rule = TemperatureDependentQuadraticMixing(matrix, slope)
    temperature = torch.tensor(300.0, dtype=torch.float64)
    pure_a = torch.tensor([1.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(
        rule.cross_a(temperature, pure_a),
        torch.sqrt(pure_a[:, None] * pure_a[None, :]) * (1.0 - rule.kij(temperature)),
    )

    with pytest.raises(ValueError, match="square"):
        TemperatureDependentQuadraticMixing(torch.zeros(2), torch.zeros(2))
    with pytest.raises(ValueError, match="equal shapes"):
        TemperatureDependentQuadraticMixing(matrix, torch.zeros((3, 3)))
    with pytest.raises(ValueError, match="symmetric"):
        TemperatureDependentQuadraticMixing(
            torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            torch.zeros((2, 2)),
        )


def test_huron_vidal_reduces_to_pure_weighted_alpha():
    activity = Wilson(
        torch.zeros((2, 2), dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
    )
    rule = HuronVidalMixing(
        activity,
        delta1=1.0 + 2.0**0.5,
        delta2=1.0 - 2.0**0.5,
    )
    temperature = torch.tensor(300.0, dtype=torch.float64)
    x = torch.tensor([0.3, 0.7], dtype=torch.float64)
    a = torch.tensor([1.0, 2.0], dtype=torch.float64)
    b = torch.tensor([0.1, 0.2], dtype=torch.float64)
    am, bm = rule(temperature, x, a, b)
    expected = bm * R * temperature * torch.sum(x * a / (b * R * temperature))
    torch.testing.assert_close(am, expected)
    with pytest.raises(ValueError, match="distinct"):
        HuronVidalMixing(activity, delta1=1.0, delta2=1.0)


def test_huron_vidal_cubic_integration_and_factory_validation():
    activity = Wilson(
        torch.zeros((2, 2), dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        trainable=True,
    )
    rule = HuronVidalMixing(
        activity,
        delta1=1.0 + 2.0**0.5,
        delta2=1.0 - 2.0**0.5,
    )
    model = peng_robinson_1978(
        component_set(("methane", "n_butane")),
        mixing=rule,
        volume_translation=torch.tensor([1.0e-7, 2.0e-7], dtype=torch.float64),
    )
    temperature = torch.tensor(300.0, dtype=torch.float64)
    pressure = torch.tensor(1.0e6, dtype=torch.float64)
    composition = torch.tensor([0.4, 0.6], dtype=torch.float64)
    assert torch.isfinite(model.log_fugacity_coefficients(temperature, pressure, composition)).all()
    model.log_fugacity_coefficients(temperature, pressure, composition).sum().backward()
    assert activity.interaction.grad is not None
    with pytest.raises(ValueError, match="mutually exclusive"):
        peng_robinson_1978(
            component_set(("methane", "n_butane")),
            kij=torch.zeros((2, 2), dtype=torch.float64),
            mixing=rule,
        )


def test_huron_vidal_batched_fugacity_matches_scalar_evaluation():
    matrix = torch.zeros((2, 2), dtype=torch.float64)
    activity = HuronVidalNRTL(
        matrix,
        torch.tensor([[0.0, -0.4], [0.7, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.3], [0.3, 0.0]], dtype=torch.float64),
        torch.tensor([7.0e-5, 3.0e-5], dtype=torch.float64),
        trainable=True,
    )
    rule = HuronVidalMixing(
        activity,
        delta1=SRK.delta1,
        delta2=SRK.delta2,
    )
    model = soave_redlich_kwong(
        component_set(("water", "propane")),
        mixing=rule,
    )
    temperatures = torch.tensor([340.0, 390.0], dtype=torch.float64)
    pressures = torch.tensor([2.0e6, 5.0e6], dtype=torch.float64)
    compositions = torch.tensor(
        [[0.98, 0.02], [0.85, 0.15]],
        dtype=torch.float64,
    )

    batched = model.log_fugacity_coefficients(
        temperatures,
        pressures,
        compositions,
        "liquid",
    )
    scalar = torch.stack(
        [
            model.log_fugacity_coefficients(
                temperatures[index],
                pressures[index],
                compositions[index],
                "liquid",
            )
            for index in range(len(temperatures))
        ]
    )

    torch.testing.assert_close(batched, scalar)
    batched.sum().backward()
    assert activity.temperature_coefficient.grad is not None


def test_phase_equilibrium_residual_batches_and_validates_compositions():
    model = _pedersen_propane_water_hv()
    temperatures = torch.tensor([369.6, 394.3], dtype=torch.float64)
    pressures = torch.tensor([6.9e5, 20.7e5], dtype=torch.float64)
    liquid = torch.tensor(
        [[1.0 - 5.8e-5, 5.8e-5], [1.0 - 4.4e-4, 4.4e-4]],
        dtype=torch.float64,
    )
    vapor = torch.tensor(
        [[0.133, 0.867], [0.0841, 0.9159]],
        dtype=torch.float64,
    )
    residual = phase_equilibrium_residual(
        model,
        temperatures,
        pressures,
        liquid,
        vapor,
    )

    assert residual.shape == liquid.shape
    assert torch.isfinite(residual).all()
    with pytest.raises(ValueError, match="equal shapes"):
        phase_equilibrium_residual(
            model,
            temperatures[0],
            pressures[0],
            liquid[0],
            torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="positive"):
        phase_equilibrium_residual(
            model,
            temperatures[0],
            pressures[0],
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            vapor[0],
        )


def _pedersen_propane_water_hv():
    components = component_set(("water", "propane"))
    covolumes = SRK.omega_b * R * components.critical_temperature / components.critical_pressure
    activity = HuronVidalNRTL(
        energy_over_r=torch.tensor(
            [[0.0, -2026.0], [6065.0, 0.0]],
            dtype=torch.float64,
        ),
        temperature_coefficient=torch.tensor(
            [[0.0, -3.82], [-3.92, 0.0]],
            dtype=torch.float64,
        ),
        nonrandomness=torch.tensor(
            [[0.0, 0.05], [0.05, 0.0]],
            dtype=torch.float64,
        ),
        covolumes=covolumes,
    )
    mixing = HuronVidalMixing(
        activity,
        delta1=SRK.delta1,
        delta2=SRK.delta2,
    )
    return soave_redlich_kwong(components, mixing=mixing)


def _jaubert_n_butane_water_hv():
    system = ("n_butane", "water")
    components = component_set(system)
    activity = activity_model(
        "activity.hv-nrtl-jaubert-2020-n-butane-water",
        system,
    )
    return soave_redlich_kwong(
        components,
        mixing=HuronVidalMixing(
            activity,
            delta1=SRK.delta1,
            delta2=SRK.delta2,
        ),
    )


def test_huron_vidal_nrtl_validation_and_gradient():
    matrix = torch.zeros((2, 2), dtype=torch.float64)
    covolumes = torch.tensor([1.0, 2.0], dtype=torch.float64)
    model = HuronVidalNRTL(matrix, matrix, matrix, covolumes, trainable=True)
    temperature = torch.tensor(350.0, dtype=torch.float64)
    composition = torch.tensor([0.3, 0.7], dtype=torch.float64)
    torch.testing.assert_close(
        model.tau_matrix(temperature),
        matrix,
    )
    value = model.excess_gibbs_rt(temperature, composition)
    torch.testing.assert_close(value, torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(
        model.log_activity_coefficients(temperature, composition),
        matrix[0],
    )
    value.backward()
    assert model.energy_over_r.grad is not None
    assert model.temperature_coefficient.grad is not None
    with pytest.raises(ValueError, match="square"):
        HuronVidalNRTL(matrix[0], matrix, matrix, covolumes)
    with pytest.raises(ValueError, match="one positive"):
        HuronVidalNRTL(matrix, matrix, matrix, torch.ones(3))
    with pytest.raises(ValueError, match="must be positive"):
        HuronVidalNRTL(matrix, matrix, matrix, torch.tensor([1.0, 0.0]))


def test_anchored_huron_vidal_matches_standard_model_and_freezes():
    lower_temperature = 300.0
    upper_temperature = 650.0
    energy_over_r = torch.tensor(
        [[0.0, 4031.9527248050813], [2533.8452008416716, 0.0]],
        dtype=torch.float64,
    )
    temperature_coefficient = torch.tensor(
        [[0.0, -3.8580511357065994], [-1.6877087017239711, 0.0]],
        dtype=torch.float64,
    )
    nonrandomness = torch.tensor(
        [[0.0, 0.2069410807024626], [0.2069410807024626, 0.0]],
        dtype=torch.float64,
    )
    covolumes = torch.tensor([8.0e-5, 3.0e-5], dtype=torch.float64)
    lower_tau = energy_over_r / lower_temperature + temperature_coefficient
    upper_tau = energy_over_r / upper_temperature + temperature_coefficient
    anchored = AnchoredHuronVidalNRTL(
        lower_tau,
        upper_tau,
        nonrandomness,
        covolumes,
        lower_temperature=lower_temperature,
        upper_temperature=upper_temperature,
        trainable_nonrandomness=True,
    )
    standard = HuronVidalNRTL(
        energy_over_r,
        temperature_coefficient,
        nonrandomness,
        covolumes,
    )
    temperatures = torch.tensor([310.93, 477.59, 637.15], dtype=torch.float64)
    compositions = torch.tensor(
        [[0.0001, 0.9999], [0.0010, 0.9990], [0.041, 0.959]],
        dtype=torch.float64,
    )

    torch.testing.assert_close(anchored.energy_over_r, energy_over_r)
    torch.testing.assert_close(
        anchored.temperature_coefficient,
        temperature_coefficient,
    )
    torch.testing.assert_close(anchored.nonrandomness, nonrandomness)
    torch.testing.assert_close(
        anchored.excess_gibbs_rt(temperatures, compositions),
        standard.excess_gibbs_rt(temperatures, compositions),
    )
    torch.testing.assert_close(
        anchored.log_activity_coefficients(temperatures, compositions),
        standard.log_activity_coefficients(temperatures, compositions),
    )
    anchored.excess_gibbs_rt(temperatures, compositions).sum().backward()
    assert anchored.raw_tau_at_lower_temperature.grad is not None
    assert anchored.raw_tau_at_upper_temperature.grad is not None
    assert anchored.raw_nonrandomness is not None
    assert anchored.raw_nonrandomness.grad is not None

    frozen = anchored.freeze()
    torch.testing.assert_close(frozen.energy_over_r, energy_over_r)
    torch.testing.assert_close(
        frozen.temperature_coefficient,
        temperature_coefficient,
    )
    torch.testing.assert_close(frozen.nonrandomness, nonrandomness)
    assert not tuple(frozen.parameters())


def test_anchored_huron_vidal_validates_inputs_and_fixed_alpha():
    matrix = torch.zeros((2, 2), dtype=torch.float64)
    covolumes = torch.ones(2, dtype=torch.float64)
    fixed = AnchoredHuronVidalNRTL(
        matrix,
        matrix,
        matrix,
        covolumes,
        lower_temperature=300.0,
        upper_temperature=600.0,
    )
    assert fixed.raw_nonrandomness is None
    torch.testing.assert_close(fixed.nonrandomness, matrix)
    with pytest.raises(ValueError, match="temperature must be positive"):
        fixed.tau_matrix(torch.tensor(-1.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="square"):
        AnchoredHuronVidalNRTL(
            matrix[0],
            matrix,
            matrix,
            covolumes,
            lower_temperature=300.0,
            upper_temperature=600.0,
        )
    with pytest.raises(ValueError, match="one positive"):
        AnchoredHuronVidalNRTL(
            matrix,
            matrix,
            matrix,
            torch.ones(3, dtype=torch.float64),
            lower_temperature=300.0,
            upper_temperature=600.0,
        )
    with pytest.raises(ValueError, match="must be positive"):
        AnchoredHuronVidalNRTL(
            matrix,
            matrix,
            matrix,
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            lower_temperature=300.0,
            upper_temperature=600.0,
        )
    with pytest.raises(ValueError, match="temperatures"):
        AnchoredHuronVidalNRTL(
            matrix,
            matrix,
            matrix,
            covolumes,
            lower_temperature=600.0,
            upper_temperature=300.0,
        )
    with pytest.raises(ValueError, match="bounds"):
        AnchoredHuronVidalNRTL(
            matrix,
            matrix,
            torch.full_like(matrix, 0.3),
            covolumes,
            lower_temperature=300.0,
            upper_temperature=600.0,
            trainable_nonrandomness=True,
            nonrandomness_bounds=(0.6, 0.05),
        )
    asymmetric = torch.tensor([[0.0, 0.2], [0.3, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="symmetric"):
        AnchoredHuronVidalNRTL(
            matrix,
            matrix,
            asymmetric,
            covolumes,
            lower_temperature=300.0,
            upper_temperature=600.0,
            trainable_nonrandomness=True,
        )
    outside = torch.tensor([[0.0, 0.8], [0.8, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="inside"):
        AnchoredHuronVidalNRTL(
            matrix,
            matrix,
            outside,
            covolumes,
            lower_temperature=300.0,
            upper_temperature=600.0,
            trainable_nonrandomness=True,
        )


def test_huron_vidal_homogeneous_states_match_thermopack_2_2_3():
    """Verify both cubic roots across the complete frozen 24-state grid."""
    model = _jaubert_n_butane_water_hv()
    path = Path(__file__).parent / "data" / "thermopack_2_2_3_srk_hv_n_butane_water_states.csv"
    with path.open(newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    assert len(rows) == 24
    for phase in ("liquid", "vapor"):
        selected = tuple(row for row in rows if row["phase"] == phase)
        temperature = torch.tensor(
            [float(row["temperature_K"]) for row in selected],
            dtype=torch.float64,
        )
        pressure = torch.tensor(
            [float(row["pressure_Pa"]) for row in selected],
            dtype=torch.float64,
        )
        first = torch.tensor(
            [float(row["x1"]) for row in selected],
            dtype=torch.float64,
        )
        composition = torch.stack((first, 1.0 - first), dim=-1)
        reference_volume = torch.tensor(
            [float(row["molar_volume_m3_mol"]) for row in selected],
            dtype=torch.float64,
        )
        reference_log_phi = torch.tensor(
            [[float(row["lnphi_1"]), float(row["lnphi_2"])] for row in selected],
            dtype=torch.float64,
        )

        torch.testing.assert_close(
            model.molar_volume(temperature, pressure, composition, phase),
            reference_volume,
            rtol=5.0e-8,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            model.log_fugacity_coefficients(
                temperature,
                pressure,
                composition,
                phase,
            ),
            reference_log_phi,
            rtol=1.0e-6,
            atol=5.0e-8,
        )


def test_huron_vidal_fixed_tp_branches_match_all_thermopack_states():
    """Verify all 70 positive Jaubert states, including both held-out isotherms."""
    model = _jaubert_n_butane_water_hv()
    path = Path(__file__).parent / "data" / "thermopack_2_2_3_srk_hv_n_butane_water_flash.csv"
    with path.open(newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    assert len(rows) == 70
    liquid_errors = []
    vapor_errors = []
    for row in rows:
        reference_liquid = float(row["liquid_x1"])
        reference_vapor = float(row["vapor_y1"])
        point = binary_vle_point(
            model,
            torch.tensor(float(row["temperature_K"]), dtype=torch.float64),
            torch.tensor(float(row["pressure_Pa"]), dtype=torch.float64),
            torch.tensor(
                [reference_liquid, 1.0 - reference_liquid],
                dtype=torch.float64,
            ),
            torch.tensor(
                [reference_vapor, 1.0 - reference_vapor],
                dtype=torch.float64,
            ),
        )
        assert point.converged
        liquid_errors.append(abs(float(point.liquid_composition[0]) - reference_liquid))
        vapor_errors.append(abs(float(point.vapor_composition[0]) - reference_vapor))

    assert max(liquid_errors) < 2.0e-8
    assert max(vapor_errors) < 4.0e-8


@pytest.mark.serial
def test_huron_vidal_full_pedersen_table_against_reference_data(
    not_cleared_data: Path,
):
    """Validate all 14 Pedersen states, not selected single-point examples."""
    model = _pedersen_propane_water_hv()
    path = not_cleared_data / "pedersen_2024_hv_propane_water.csv"
    with path.open(newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    assert len(rows) == 14
    predicted: list[Tensor] = []
    reference: list[Tensor] = []
    experimental: list[Tensor] = []
    for row in rows:
        temperature = torch.tensor(float(row["temperature_K"]), dtype=torch.float64)
        pressure = torch.tensor(float(row["pressure_bar"]) * 1.0e5, dtype=torch.float64)
        x_propane = float(row["x_propane_water_experimental"])
        y_water = float(row["x_water_propane_experimental"])
        initial_k = torch.tensor(
            [
                y_water / (1.0 - x_propane),
                (1.0 - y_water) / x_propane,
            ],
            dtype=torch.float64,
        )
        result = two_phase_flash(
            model,
            ChemicalState(
                temperature,
                pressure,
                torch.tensor([0.5, 0.5], dtype=torch.float64),
            ),
            initial_k_values=initial_k,
            check_stability=False,
            raise_on_failure=True,
        )
        assert result.converged
        predicted.append(
            torch.stack(
                (
                    result.phases[0].composition[1],
                    result.phases[1].composition[0],
                )
            )
        )
        reference.append(
            torch.tensor(
                [
                    float(row["x_propane_water_thermopack"]),
                    float(row["x_water_propane_thermopack"]),
                ],
                dtype=torch.float64,
            )
        )
        experimental.append(
            torch.tensor(
                [x_propane, y_water],
                dtype=torch.float64,
            )
        )
    # The small residual difference comes from independently stored pure-fluid
    # constants; the complete composition trends agree within 0.4%.
    predicted_tensor = torch.stack(predicted)
    torch.testing.assert_close(
        predicted_tensor,
        torch.stack(reference),
        rtol=4.0e-3,
        atol=2.0e-10,
    )
    # These deliberately loose experimental limits document, rather than hide,
    # the known weakness of this classic-alpha parameterization for the dilute
    # propane-in-water branch.
    experimental_aard = 100.0 * (
        (predicted_tensor - torch.stack(experimental)) / torch.stack(experimental)
    ).abs().mean(dim=0)
    assert experimental_aard[0] < 50.0
    assert experimental_aard[1] < 8.5


def test_ideal_gas_polynomial_integrals_and_validation():
    coefficients = torch.tensor([[10.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    zeros = torch.zeros(1, dtype=torch.float64)
    model = IdealGasPolynomial(
        coefficients,
        zeros,
        zeros,
        reference_temperature=300.0,
        trainable=True,
    )
    temperature = torch.tensor(330.0, dtype=torch.float64)
    torch.testing.assert_close(model.heat_capacity(temperature), temperature.new_tensor([10.0]))
    torch.testing.assert_close(model.enthalpy(temperature), temperature.new_tensor([300.0]))
    torch.testing.assert_close(
        model.entropy(temperature),
        torch.tensor([10.0 * torch.log(torch.tensor(1.1, dtype=torch.float64))]),
    )
    torch.testing.assert_close(
        model.chemical_potential(temperature),
        model.enthalpy(temperature) - temperature * model.entropy(temperature),
    )
    assert isinstance(model.heat_capacity_coefficients, nn.Parameter)
    with pytest.raises(ValueError, match="nonempty matrix"):
        IdealGasPolynomial(torch.zeros((1, 0)), zeros, zeros)
    with pytest.raises(ValueError, match="one reference"):
        IdealGasPolynomial(torch.zeros((2, 4)), zeros, zeros)
    with pytest.raises(ValueError, match="positive"):
        IdealGasPolynomial(coefficients, zeros, zeros, reference_temperature=0.0)


def test_least_squares_and_parameter_fitting():
    prediction = torch.tensor([1.0, 3.0])
    observation = torch.tensor([0.0, 1.0])
    weights = torch.tensor([1.0, 4.0])
    assert least_squares_loss(prediction, observation, scale=2.0).item() == pytest.approx(0.625)
    assert least_squares_loss(
        prediction,
        observation,
        scale=2.0,
        weights=weights,
    ).item() == pytest.approx(2.125)

    parameter = nn.Parameter(torch.tensor(0.0))
    result = fit_parameters(
        [parameter],
        lambda: (parameter - 2.0).square(),
        learning_rate=0.2,
        max_iterations=300,
        tolerance=1.0e-8,
    )
    assert result.losses
    assert result.final_loss < 1.0e-5
    assert result.iterations <= 300
    unfinished_parameter = nn.Parameter(torch.tensor(0.0))
    unfinished = fit_parameters(
        [unfinished_parameter],
        lambda: (unfinished_parameter - 2.0).square(),
        max_iterations=1,
        tolerance=0.0,
    )
    assert not unfinished.converged
    with pytest.raises(ValueError, match="at least one"):
        fit_parameters([], lambda: torch.tensor(0.0))
    bad = nn.Parameter(torch.tensor(1.0))
    with pytest.raises(ValueError, match="finite scalar"):
        fit_parameters([bad], lambda: torch.tensor([1.0]))
