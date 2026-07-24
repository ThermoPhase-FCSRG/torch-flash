from __future__ import annotations

from copy import deepcopy

import pytest
import torch

import torch_flash.eos.cpa_heavy_end as cpa_heavy_module
from torch_flash import PseudoComponentCut
from torch_flash.database import ModelParameterSet, load_model_parameters
from torch_flash.envelope import saturation_point
from torch_flash.eos.cpa import (
    CPAEOS,
    CPAComponent,
    cpa_eos,
    cpa_folas_2005,
    cpa_oliveira_2007,
    cpa_yan_2009,
)
from torch_flash.eos.cpa_heavy_end import (
    cpa_components_from_cuts,
    cpa_heavy_end_correlations,
    cpa_monomer_properties,
    cpa_pseudocomponent,
)
from torch_flash.exceptions import (
    ConvergenceError,
    InvalidStateError,
    ParameterDatabaseError,
)
from torch_flash.solvers import damped_newton


def test_cpa_folas_site_fractions_strength_and_roots():
    model = cpa_folas_2005(("water", "methanol"))
    temperature = torch.tensor(350.0, dtype=torch.float64)
    pressure = torch.tensor(1.0e5, dtype=torch.float64)
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    density = torch.tensor(30_000.0, dtype=torch.float64)
    strength = model.association_strength(temperature, density, composition)
    assert strength.shape == (2, 4, 2, 4)
    assert torch.all(strength >= 0.0)
    sites = model.site_fractions(temperature, density, composition)
    assert sites.shape == (2, 4)
    assert bool(((sites > 0.0) & (sites <= 1.0)).all())

    liquid_volume = model.molar_volume(temperature, pressure, composition, "liquid")
    vapor_volume = model.molar_volume(temperature, pressure, composition, "vapor")
    stable_volume = model.molar_volume(temperature, pressure, composition, "stable")
    assert model.select_z(temperature, pressure, composition, "vapor") > 0.0
    assert liquid_volume < vapor_volume
    torch.testing.assert_close(stable_volume, vapor_volume)
    for volume in (liquid_volume, vapor_volume):
        torch.testing.assert_close(
            model.pressure(temperature, volume, composition),
            pressure,
            rtol=1.0e-10,
            atol=1.0e-5,
        )
    assert model.ncomponents == 2
    assert torch.isfinite(
        model.log_fugacity_coefficients(temperature, pressure, composition, "vapor")
    ).all()


def test_cpa_cr1_and_nonassociating_sites():
    parameters = (
        CPAComponent("propane", 369.83, 0.9, 5.7e-5, 0.63),
        CPAComponent(
            "methanol",
            512.64,
            0.4,
            3.1e-5,
            0.43,
            24_591.0,
            0.0161,
            "2B",
        ),
    )
    model = CPAEOS(parameters, combining_rule="CR1", association_iterations=10)
    temperature = torch.tensor(320.0, dtype=torch.float64)
    density = torch.tensor(1000.0, dtype=torch.float64)
    composition = torch.tensor([0.4, 0.6], dtype=torch.float64)
    strength = model.association_strength(temperature, density, composition)
    assert torch.count_nonzero(strength[0]) == 0
    sites = model.site_fractions(temperature, density, composition)
    torch.testing.assert_close(sites[0], torch.ones(4, dtype=torch.float64))
    pure_a, pure_b = model.pure_parameters(temperature)
    assert pure_a.shape == pure_b.shape == (2,)
    am, bm = model.mixture_parameters(temperature, composition)
    assert am > 0.0 and bm > 0.0


def test_cpa_batched_association_matches_scalar_states_and_gradients():
    model = cpa_folas_2005(("water", "methanol"))
    temperatures = torch.tensor([300.0, 350.0, 425.0], dtype=torch.float64)
    densities = torch.tensor([500.0, 10_000.0, 30_000.0], dtype=torch.float64)
    compositions = torch.tensor(
        [[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]],
        dtype=torch.float64,
    )
    batched_sites = model.site_fractions(temperatures, densities, compositions)
    scalar_sites = torch.stack(
        [
            model.site_fractions(temperature, density, composition)
            for temperature, density, composition in zip(
                temperatures,
                densities,
                compositions,
                strict=True,
            )
        ]
    )
    torch.testing.assert_close(batched_sites, scalar_sites, rtol=2.0e-14, atol=2.0e-15)

    volumes = densities.reciprocal().requires_grad_()
    batched_pressure = model.pressure(temperatures, volumes, compositions)
    scalar_pressure = torch.stack(
        [
            model.pressure(temperature, volume, composition)
            for temperature, volume, composition in zip(
                temperatures,
                volumes,
                compositions,
                strict=True,
            )
        ]
    )
    torch.testing.assert_close(batched_pressure, scalar_pressure, rtol=2.0e-14, atol=1.0e-7)
    batched_pressure.sum().backward()
    assert volumes.grad is not None
    assert torch.isfinite(volumes.grad).all()


def test_cpa_paper_azeotrope_regression():
    kij = torch.tensor([[0.0, -0.11], [-0.11, 0.0]], dtype=torch.float64)
    model = cpa_folas_2005(("ethanol", "water"), kij=kij)
    point = saturation_point(
        model,
        torch.tensor(333.15, dtype=torch.float64),
        torch.tensor([0.91, 0.09], dtype=torch.float64),
        "bubble",
        tolerance=1.0e-7,
        max_iterations=12,
    )
    assert point.converged
    # Folas et al. Table 3: experimental 0.47 bar and x_ethanol=0.910;
    # ECR with one k12 reports 0.478 bar and x_ethanol=0.92.
    assert float(point.pressure / 1.0e5) == pytest.approx(0.478, rel=0.03)
    assert float(point.incipient_composition[0]) == pytest.approx(0.91, abs=0.02)


@pytest.mark.parametrize(
    ("components", "temperature", "initial_x", "initial_pressure_bar", "kij"),
    [
        (("2_propanol", "water"), 423.15, 0.68, 9.293, -0.16),
        (("ethanol", "water"), 523.15, 0.72, 69.68, -0.11),
    ],
)
def test_cpa_exact_azeotropes_across_system_and_temperature(
    components,
    temperature,
    initial_x,
    initial_pressure_bar,
    kij,
):
    interaction = torch.tensor([[0.0, kij], [kij, 0.0]], dtype=torch.float64)
    model = cpa_folas_2005(components, kij=interaction)
    temperature_tensor = torch.tensor(temperature, dtype=torch.float64)

    def residual(variables):
        x_alcohol = torch.sigmoid(variables[0])
        composition = torch.stack((x_alcohol, 1.0 - x_alcohol))
        pressure = torch.exp(variables[1])
        return model.log_fugacity_coefficients(
            temperature_tensor, pressure, composition, "liquid"
        ) - model.log_fugacity_coefficients(temperature_tensor, pressure, composition, "vapor")

    result = damped_newton(
        residual,
        torch.tensor(
            [
                torch.logit(torch.tensor(initial_x, dtype=torch.float64)),
                torch.log(torch.tensor(initial_pressure_bar * 1.0e5, dtype=torch.float64)),
            ],
            dtype=torch.float64,
        ),
        tolerance=1.0e-7,
        max_iterations=10,
    )
    assert result.converged
    # Folas et al. Table 3 reports rounded one-kij ECR results.
    assert float(torch.sigmoid(result.solution[0])) == pytest.approx(initial_x, abs=0.01)
    assert float(torch.exp(result.solution[1]) / 1.0e5) == pytest.approx(
        initial_pressure_bar,
        rel=0.01,
    )


def test_cpa_validation_errors(monkeypatch):
    with pytest.raises(ValueError, match="at least one"):
        CPAEOS(())
    with pytest.raises(ValueError, match="unsupported"):
        CPAEOS((CPAComponent("water", 647.0, 0.1, 1.0e-5, 0.5, scheme="3C"),))
    with pytest.raises(KeyError, match="unavailable"):
        cpa_folas_2005(("hydrogen",))

    model = cpa_folas_2005(("water",))
    temperature = torch.tensor(350.0, dtype=torch.float64)
    pressure = torch.tensor(1.0e5, dtype=torch.float64)
    composition = torch.tensor([1.0], dtype=torch.float64)
    model.combining_rule = "invalid"
    with pytest.raises(ValueError, match="combining rule"):
        model.association_strength(temperature, torch.tensor(1000.0), composition)
    model.combining_rule = "ECR"
    batched_sites = model.site_fractions(
        temperature.expand(3),
        torch.tensor([500.0, 1000.0, 2000.0], dtype=torch.float64),
        composition,
    )
    assert batched_sites.shape == (3, 1, 4)
    with pytest.raises(ValueError, match="wrong number"):
        model.site_fractions(temperature, torch.tensor(1000.0), torch.tensor([0.5, 0.5]))
    with pytest.raises(ValueError, match="homogeneous state"):
        model.residual_helmholtz_rt(
            temperature[None],
            torch.tensor([1.0e-3]),
            composition[None, :],
        )
    with pytest.raises(InvalidStateError, match="covolume"):
        model.residual_helmholtz_rt(temperature, torch.tensor(1.0e-8), composition)
    with pytest.raises(ValueError, match="scalar T-P"):
        model.molar_volume(temperature[None], pressure, composition)
    with pytest.raises(ValueError, match="unknown phase"):
        model.molar_volume(temperature, pressure, composition, "solid")

    monkeypatch.setattr(
        model,
        "_phase_volume_newton",
        lambda temperature, pressure, composition, phase: None,
    )
    roots = model._volume_roots(temperature, pressure, composition)
    torch.testing.assert_close(
        model.molar_volume(temperature, pressure, composition, "liquid"),
        roots[0],
    )
    torch.testing.assert_close(
        model.molar_volume(temperature, pressure, composition, "vapor"),
        roots[-1],
    )

    monkeypatch.setattr(
        model,
        "pressure",
        lambda temperature, volume, composition: torch.ones_like(volume),
    )
    with pytest.raises(ConvergenceError, match="no pressure root"):
        model.molar_volume(temperature, pressure, composition)


def test_cpa_fast_volume_failure_paths(monkeypatch):
    model = cpa_folas_2005(("water",))
    temperature = torch.tensor(350.0, dtype=torch.float64)
    pressure = torch.tensor(1.0e5, dtype=torch.float64)
    composition = torch.ones(1, dtype=torch.float64)

    monkeypatch.setattr(
        model,
        "pressure",
        lambda temperature, volume, composition: torch.full_like(volume, torch.nan),
    )
    assert (
        model._phase_volume_newton(
            temperature,
            pressure,
            composition,
            "vapor",
        )
        is None
    )

    monkeypatch.setattr(
        model,
        "pressure",
        lambda temperature, volume, composition: 2.0 * pressure + 0.0 * volume,
    )
    assert (
        model._phase_volume_newton(
            temperature,
            pressure,
            composition,
            "liquid",
        )
        is None
    )


def test_cpa_trainable_kij_and_parameter_error():
    kij = torch.zeros((2, 2), dtype=torch.float32)
    model = cpa_folas_2005(("ethanol", "water"), kij=kij, trainable=True)
    assert isinstance(model.mixing.raw_kij, torch.nn.Parameter)
    assert isinstance(model.a0, torch.nn.Parameter)
    assert isinstance(model.association_energy, torch.nn.Parameter)


def test_cpa_covolume_interaction_api_and_gradient():
    lij = torch.tensor([[0.0, 0.06], [0.06, 0.0]], dtype=torch.float64)
    model = cpa_folas_2005(
        ("ethanol", "water"),
        lij=lij,
        trainable_lij=True,
        dtype=torch.float64,
    )
    temperature = torch.tensor(350.0, dtype=torch.float64)
    composition = torch.tensor([0.4, 0.6], dtype=torch.float64)
    _, pure_b = model.pure_parameters(temperature)
    _, mixed_b = model.mixture_parameters(temperature, composition)
    assert isinstance(model.mixing.raw_lij, torch.nn.Parameter)
    assert mixed_b < torch.dot(composition, pure_b)
    mixed_b.backward()
    assert model.mixing.raw_lij.grad is not None
    assert model.mixing.raw_lij.grad[0, 1] < 0.0


def test_cpa_hydrocarbon_water_parameterizations_and_solvation():
    temperature = torch.tensor(344.26, dtype=torch.float64)
    yan = cpa_yan_2009(("methane", "water"))
    expected = 0.6769 - 213.5 / temperature
    torch.testing.assert_close(yan.binary_interaction(temperature)[0, 1], expected)
    assert yan.binary_interaction(temperature[None]).shape == (1, 2, 2)

    oliveira = cpa_oliveira_2007(("benzene", "water"))
    torch.testing.assert_close(
        oliveira.binary_interaction(temperature),
        torch.tensor([[0.0, 0.047], [0.047, 0.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        oliveira.cross_association_energy,
        torch.tensor([[0.0, 8327.5], [8327.5, 0.0]], dtype=torch.float64),
    )
    strength = oliveira.association_strength(
        temperature,
        torch.tensor(1000.0, dtype=torch.float64),
        torch.tensor([0.1, 0.9], dtype=torch.float64),
    )
    # Aromatic 1B accepts from water donor sites, but does not bond to water's
    # electron-acceptor sites or self-associate.
    assert strength[0, 0, 1, 0] > 0.0
    assert strength[0, 0, 1, 2] == 0.0
    assert torch.count_nonzero(strength[0, :, 0, :]) == 0


def test_cpa_ecr_combines_complete_pure_association_strength():
    model = cpa_folas_2005(("water", "methanol"))
    temperature = torch.tensor(350.0, dtype=torch.float64)
    density = torch.tensor(5000.0, dtype=torch.float64)
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    strength = model.association_strength(temperature, density, composition)
    _, bm = model.mixture_parameters(temperature, composition)
    radial = 1.0 / (1.0 - 1.9 * 0.25 * bm * density)
    pure = (
        torch.expm1(model.association_energy / (8.31446261815324 * temperature))
        * model.b
        * model.association_volume
    )
    expected = radial * torch.sqrt(pure[0] * pure[1])
    torch.testing.assert_close(strength[0, 0, 1, 1], expected)


def test_cpa_heavy_end_adapter_and_custom_correlation_set():
    correlations = cpa_heavy_end_correlations()
    assert correlations.srk_omega_a == pytest.approx(0.42748)
    normal = cpa_monomer_properties(371.58, 0.684)
    assert normal.used_boiling_point_match
    assert normal.critical_temperature > 500.0
    assert normal.critical_pressure > 1.0e6
    assert 0.0 < normal.acentric_factor < 1.0

    pseudo = cpa_pseudocomponent("C7-C10", 400.0, 0.75, 0.12)
    model = CPAEOS((pseudo,))
    assert model.names == ("C7-C10",)
    torch.testing.assert_close(model.molar_mass, torch.tensor([0.12], dtype=torch.float64))

    cuts = (
        PseudoComponentCut("C7-C10", 0.4, 400.0, 0.75, 0.12),
        PseudoComponentCut("C11+", 0.6, 550.0, 0.85, 0.25),
    )
    characterized = cpa_components_from_cuts(cuts, dtype=torch.float32)
    assert len(characterized.components) == len(characterized.monomer_properties) == 2
    assert characterized.mole_fractions.dtype == torch.float32
    torch.testing.assert_close(
        characterized.mole_fractions,
        torch.tensor([0.4, 0.6], dtype=torch.float32),
    )

    custom = load_model_parameters("cpa-yan-2009").as_dict()
    custom["heavy_end_correlations"]["srk_omega_a"] = 0.5
    parameter_set = ModelParameterSet(
        "cpa.custom-heavy-end",
        "cpa",
        "CPA",
        "1",
        custom,
    )
    changed = cpa_monomer_properties(371.58, 0.684, parameter_set)
    assert changed.a0 / normal.a0 == pytest.approx(0.5 / 0.42748)


def _custom_component(name: str, scheme: str = "none") -> CPAComponent:
    return CPAComponent(
        name,
        500.0,
        1.0,
        5.0e-5,
        0.7,
        15_000.0 if scheme != "none" else 0.0,
        0.02 if scheme != "none" else 0.0,
        scheme,  # type: ignore[arg-type]
        4.0e6,
        0.2,
        0.1,
    )


def test_cpa_extended_schemes_and_constructor_validation():
    schemes = ("1A", "1B", "3B")
    model = CPAEOS(tuple(_custom_component(name, name) for name in schemes))
    assert torch.equal(
        model.site_types,
        torch.tensor(
            [[0, -1, -1, -1], [1, -1, -1, -1], [0, 1, 1, -1]],
            dtype=torch.int64,
        ),
    )
    with pytest.raises(ValueError, match="combining rule"):
        CPAEOS((_custom_component("x"),), combining_rule="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        CPAEOS((_custom_component("x"),), association_iterations=-1)
    matrix = torch.zeros((1, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="either fixed"):
        CPAEOS((_custom_component("x"),), kij=matrix, kij_a=matrix, kij_b=matrix)
    with pytest.raises(ValueError, match="require both"):
        CPAEOS((_custom_component("x"),), kij_a=matrix)
    with pytest.raises(ValueError, match="require both energy"):
        CPAEOS((_custom_component("x"),), cross_association_energy=matrix)
    with pytest.raises(ValueError, match="must be positive"):
        CPAEOS((CPAComponent("x", 500.0, -1.0, 5.0e-5, 0.7, critical_pressure=1.0),))
    with pytest.raises(ParameterDatabaseError, match="must provide"):
        CPAEOS((CPAComponent("unknown-cut", 500.0, 1.0, 5.0e-5, 0.7),))
    with pytest.raises(ValueError, match="critical pressure"):
        CPAEOS(
            (
                CPAComponent(
                    "x",
                    500.0,
                    1.0,
                    5.0e-5,
                    0.7,
                    critical_pressure=-1.0,
                    acentric_factor=0.2,
                    molar_mass=0.1,
                ),
            )
        )

    two = (_custom_component("a", "1A"), _custom_component("b", "1B"))
    nan = torch.full((2, 2), torch.nan, dtype=torch.float64)
    with pytest.raises(ValueError, match="component count"):
        CPAEOS(
            two,
            cross_association_energy=torch.full((1, 1), torch.nan),
            cross_association_volume=torch.full((1, 1), torch.nan),
        )
    mismatched = nan.clone()
    mismatched[0, 1] = 1000.0
    with pytest.raises(ValueError, match="masks"):
        CPAEOS(
            two,
            cross_association_energy=mismatched,
            cross_association_volume=nan,
        )
    asymmetric = nan.clone()
    asymmetric[0, 1] = 1000.0
    asymmetric_volume = nan.clone()
    asymmetric_volume[0, 1] = 0.1
    with pytest.raises(ValueError, match="symmetric"):
        CPAEOS(
            two,
            cross_association_energy=asymmetric,
            cross_association_volume=asymmetric_volume,
        )
    diagonal = nan.clone()
    diagonal[0, 0] = 1000.0
    with pytest.raises(ValueError, match="pure-component"):
        CPAEOS(two, cross_association_energy=diagonal, cross_association_volume=diagonal)
    negative = nan.clone()
    negative[0, 1] = negative[1, 0] = -1.0
    with pytest.raises(ValueError, match="positive"):
        CPAEOS(two, cross_association_energy=negative, cross_association_volume=negative)

    energy = nan.clone()
    volume = nan.clone()
    energy[0, 1] = energy[1, 0] = 1000.0
    volume[0, 1] = volume[1, 0] = 0.01
    trainable = CPAEOS(
        two,
        kij_a=torch.zeros((2, 2)),
        kij_b=torch.zeros((2, 2)),
        cross_association_energy=energy,
        cross_association_volume=volume,
        trainable=True,
    )
    assert isinstance(trainable.cross_association_energy, torch.nn.Parameter)
    assert isinstance(trainable.mixing.raw_a, torch.nn.Parameter)
    with pytest.raises(ValueError, match="temperature"):
        trainable.binary_interaction(torch.tensor(-1.0))


def test_cpa_heavy_end_input_and_payload_errors():
    with pytest.raises(ValueError, match="name"):
        cpa_pseudocomponent("", 400.0, 0.8, 0.2)
    with pytest.raises(ValueError, match="molar mass"):
        cpa_pseudocomponent("bad", 400.0, 0.8, -1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        cpa_monomer_properties(-1.0, 0.8)
    with pytest.raises(ValueError, match="at least one"):
        cpa_components_from_cuts(())
    bad_cut = PseudoComponentCut("bad", 0.0, 400.0, 0.8, 0.2)
    with pytest.raises(ValueError, match="positive sum"):
        cpa_components_from_cuts((bad_cut,))

    wrong_kind = ModelParameterSet("wrong.kind", "activity", "NRTL", "1", {})
    with pytest.raises(ParameterDatabaseError, match="not a CPA"):
        cpa_heavy_end_correlations(wrong_kind)
    base = load_model_parameters("cpa-yan-2009").as_dict()
    missing = dict(base)
    missing.pop("heavy_end_correlations")
    with pytest.raises(ParameterDatabaseError, match="heavy_end_correlations"):
        cpa_heavy_end_correlations(
            ModelParameterSet("cpa.missing-heavy", "cpa", "CPA", "1", missing)
        )
    malformed = load_model_parameters("cpa-yan-2009").as_dict()
    malformed["heavy_end_correlations"]["srk_m"] = [1.0]
    with pytest.raises(ParameterDatabaseError, match="requires 3"):
        cpa_heavy_end_correlations(ModelParameterSet("cpa.bad-heavy", "cpa", "CPA", "1", malformed))

    nonfinite = load_model_parameters("cpa-yan-2009").as_dict()
    nonfinite["heavy_end_correlations"]["srk_m"] = [1.0, float("nan"), 1.0]
    with pytest.raises(ParameterDatabaseError, match="must be finite"):
        cpa_heavy_end_correlations(
            ModelParameterSet("cpa.nonfinite-heavy", "cpa", "CPA", "1", nonfinite)
        )

    nonpositive = load_model_parameters("cpa-yan-2009").as_dict()
    nonpositive["heavy_end_correlations"]["srk_omega_a"] = 0.0
    with pytest.raises(ParameterDatabaseError, match="finite and positive"):
        cpa_heavy_end_correlations(
            ModelParameterSet("cpa.nonpositive-heavy", "cpa", "CPA", "1", nonpositive)
        )

    missing_ratio = load_model_parameters("cpa-yan-2009").as_dict()
    missing_ratio["heavy_end_correlations"].pop("critical_temperature_ratio")
    with pytest.raises(ParameterDatabaseError, match="critical_temperature_ratio"):
        cpa_heavy_end_correlations(
            ModelParameterSet("cpa.missing-ratio", "cpa", "CPA", "1", missing_ratio)
        )

    invalid_cut = PseudoComponentCut("C7+", 1.0, 400.0, 0.8, 0.2)
    object.__setattr__(invalid_cut, "mole_fraction", float("nan"))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        cpa_components_from_cuts((invalid_cut,))


def test_cpa_heavy_end_correlation_domain_and_boiling_match_paths(monkeypatch):
    correlations = cpa_heavy_end_correlations()
    assert (
        cpa_heavy_module._pure_srk_fugacity_difference(
            0.1,
            100.0,
            correlations.normal_boiling_pressure,
            500.0,
            3.0e6,
            correlations,
        )
        is None
    )

    invalid_domain = load_model_parameters("cpa-yan-2009").as_dict()
    invalid_domain["heavy_end_correlations"]["critical_temperature_ratio"]["numerator"] = [
        -100.0,
        0.0,
        0.0,
    ]
    invalid_domain["heavy_end_correlations"]["critical_temperature_ratio"]["denominator"] = [
        0.0,
        0.0,
        0.0,
    ]
    with pytest.raises(ValueError, match="physical correlation domain"):
        cpa_monomer_properties(
            400.0,
            0.8,
            ModelParameterSet("cpa.invalid-domain", "cpa", "CPA", "1", invalid_domain),
        )

    extrapolated = load_model_parameters("cpa-yan-2009").as_dict()
    extrapolated["heavy_end_correlations"]["normal_alkane_temperature"] = [
        1.0,
        0.0,
        1.0,
    ]
    result = cpa_monomer_properties(
        400.0,
        0.8,
        ModelParameterSet("cpa.extrapolated", "cpa", "CPA", "1", extrapolated),
    )
    assert not result.used_boiling_point_match
    assert result.acentric_factor > 0.0

    monkeypatch.setattr(
        cpa_heavy_module,
        "_pure_srk_fugacity_difference",
        lambda *args: None,
    )
    with pytest.raises(ConvergenceError, match="could not match"):
        cpa_monomer_properties(371.58, 0.684)


def _cpa_test_parameters(payload):
    return ModelParameterSet("cpa.test-payload", "cpa", "CPA", "1", payload)


def test_cpa_database_component_and_document_errors():
    wrong_kind = ModelParameterSet("not.cpa", "activity", "NRTL", "1", {})
    with pytest.raises(ParameterDatabaseError, match="not 'cpa'"):
        cpa_eos(("methane",), wrong_kind)

    base = load_model_parameters("cpa-yan-2009").as_dict()
    missing_components = deepcopy(base)
    missing_components.pop("components")
    with pytest.raises(ParameterDatabaseError, match="components mapping"):
        cpa_eos(("methane",), _cpa_test_parameters(missing_components))
    with pytest.raises(KeyError, match="unavailable"):
        cpa_eos(("carbon_dioxide",), _cpa_test_parameters(base))

    bad_optional = deepcopy(base)
    bad_optional["components"]["methane"]["critical_pressure"] = "not-numeric"
    with pytest.raises(ParameterDatabaseError, match="non-numeric optional"):
        cpa_eos(("methane",), _cpa_test_parameters(bad_optional))

    bad_rule = deepcopy(base)
    bad_rule["default_combining_rule"] = "invalid"
    with pytest.raises(ParameterDatabaseError, match="combining rule"):
        cpa_eos(("methane",), _cpa_test_parameters(bad_rule))


@pytest.mark.parametrize(
    ("binary_interactions", "message"),
    [
        ([], "must be a mapping"),
        ({"kind": "constant"}, "requires a pairs mapping"),
        ({"kind": "constant", "pairs": {"invalid": 0.1}}, "must have the form"),
        (
            {"kind": "constant", "pairs": {"methane|methane": 0.1}},
            "self-pair",
        ),
        (
            {"kind": "constant", "pairs": {"methane|water": []}},
            "must be numeric",
        ),
        (
            {
                "kind": "a_plus_b_over_temperature",
                "pairs": {"methane|water": {"a": 0.1}},
            },
            "requires numeric a and b",
        ),
        ({"kind": "unknown", "pairs": {}}, "kind must be"),
    ],
)
def test_cpa_database_binary_interaction_errors(binary_interactions, message):
    payload = load_model_parameters("cpa-yan-2009").as_dict()
    payload["binary_interactions"] = binary_interactions
    with pytest.raises(ParameterDatabaseError, match=message):
        cpa_eos(("methane", "water"), _cpa_test_parameters(payload))


@pytest.mark.parametrize(
    ("cross_association", "message"),
    [
        ([], "must be a mapping"),
        ({}, "requires a pairs mapping"),
        (
            {"pairs": {"methane|water": {"association_energy": 1000.0}}},
            "requires numeric",
        ),
    ],
)
def test_cpa_database_cross_association_errors(cross_association, message):
    payload = load_model_parameters("cpa-yan-2009").as_dict()
    payload["cross_association"] = cross_association
    with pytest.raises(ParameterDatabaseError, match=message):
        cpa_eos(("methane", "water"), _cpa_test_parameters(payload))


def test_cpa_database_ignores_unselected_pair_records():
    payload = load_model_parameters("cpa-yan-2009").as_dict()
    payload["cross_association"] = {
        "pairs": {
            "ethane|water": {
                "association_energy": 1000.0,
                "association_volume": 0.01,
            }
        }
    }
    model = cpa_eos(("methane",), _cpa_test_parameters(payload))
    assert torch.count_nonzero(model.cross_association_energy) == 0
