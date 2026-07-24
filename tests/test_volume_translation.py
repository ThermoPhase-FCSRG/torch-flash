from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from torch_flash import (
    ComponentSet,
    ModelParameterSet,
    VolumeTranslation,
    component_set,
    density_matched_translation,
    pedersen_peneloux_translation,
    pedersen_temperature_dependent_translation,
    peng_robinson_1978,
    rackett_compressibility_factor,
    saturation_point,
    soave_redlich_kwong,
    whitson_volume_translation,
)
from torch_flash.constants import R
from torch_flash.exceptions import InvalidStateError, ParameterDatabaseError


def _pedersen_light_components() -> ComponentSet:
    return ComponentSet(
        (
            "nitrogen",
            "carbon_dioxide",
            "methane",
            "ethane",
            "propane",
            "isobutane",
            "n_butane",
            "isopentane",
            "n_pentane",
            "c6",
        ),
        torch.tensor(
            [
                -147.0,
                31.1,
                -82.6,
                32.3,
                96.7,
                135.0,
                152.1,
                187.3,
                196.5,
                234.3,
            ],
            dtype=torch.float64,
        )
        + 273.15,
        torch.tensor(
            [33.94, 73.76, 46.00, 48.84, 42.46, 36.48, 38.00, 33.84, 33.74, 29.69],
            dtype=torch.float64,
        )
        * 1.0e5,
        torch.tensor(
            [0.040, 0.225, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 0.251, 0.296],
            dtype=torch.float64,
        ),
        torch.tensor(
            [28.0, 44.0, 16.0, 30.1, 44.1, 58.1, 58.1, 72.2, 72.2, 86.2],
            dtype=torch.float64,
        )
        / 1000.0,
    )


def test_volume_translation_value_object():
    reference = torch.tensor([-2.0e-6, 1.0e-6], dtype=torch.float64)
    slope = torch.tensor([1.0e-8, -2.0e-8], dtype=torch.float64)
    translation = VolumeTranslation(reference, slope, 300.0, "test")
    torch.testing.assert_close(
        translation.at_temperature(torch.tensor(310.0, dtype=torch.float64)),
        reference + 10.0 * slope,
    )
    converted = translation.to(dtype=torch.float32, device="cpu")
    assert converted.reference_shift.dtype == torch.float32
    assert converted.source == "test"
    constant = VolumeTranslation.constant(reference)
    torch.testing.assert_close(constant.temperature_slope, torch.zeros_like(reference))


@pytest.mark.parametrize(
    ("reference", "slope", "temperature", "error"),
    [
        (torch.ones((1, 1)), torch.ones((1, 1)), 300.0, ValueError),
        (torch.ones(2), torch.ones(1), 300.0, ValueError),
        (torch.tensor([float("nan")]), torch.zeros(1), 300.0, ValueError),
        (torch.ones(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64), 300.0, TypeError),
        (torch.ones(1), torch.zeros(1, dtype=torch.float64), 300.0, ValueError),
        (torch.ones(1), torch.zeros(1), float("inf"), ValueError),
        (torch.ones(1), torch.zeros(1), 0.0, ValueError),
    ],
)
def test_volume_translation_value_errors(reference, slope, temperature, error):
    with pytest.raises(error):
        VolumeTranslation(reference, slope, temperature)


def test_pedersen_correlations_reproduce_light_hydrocarbon_table():
    components = _pedersen_light_components()
    rackett = rackett_compressibility_factor(components.acentric_factor)
    torch.testing.assert_close(
        rackett,
        0.29056 - 0.08775 * components.acentric_factor,
    )

    srk_published_c = -pedersen_peneloux_translation(components, "srk").reference_shift * 1.0e6
    pr_published_c = -pedersen_peneloux_translation(components, "pr").reference_shift * 1.0e6
    # Pedersen et al. (2024), Table 7.5, cm3/mol. C6 is a lump rather than
    # pure n-hexane and CO2 uses a tabulated Rackett value, so the direct
    # Eq. 4.47 comparison is restricted to C1-C5.
    torch.testing.assert_close(
        srk_published_c[2:9],
        torch.tensor(
            [0.63, 2.63, 5.06, 7.29, 7.86, 10.93, 12.18],
            dtype=torch.float64,
        ),
        rtol=1.0e-2,
        atol=0.04,
    )
    torch.testing.assert_close(
        pr_published_c[2:9],
        torch.tensor(
            [-5.20, -5.79, -6.35, -7.18, -6.49, -6.20, -5.12],
            dtype=torch.float64,
        ),
        rtol=2.0e-3,
        atol=0.02,
    )

    custom_rackett = rackett + 0.001
    custom = pedersen_peneloux_translation(
        components,
        "srk",
        rackett_factor=custom_rackett,
    )
    default = pedersen_peneloux_translation(components, "srk")
    assert not torch.equal(custom.reference_shift, default.reference_shift)
    with pytest.raises(ValueError, match="one value"):
        pedersen_peneloux_translation(
            components,
            "srk",
            rackett_factor=torch.ones(2),
        )
    with pytest.raises(ValueError, match="finite"):
        pedersen_peneloux_translation(
            components,
            "srk",
            rackett_factor=torch.full((components.ncomponents,), float("nan")),
        )


def test_pedersen_hexane_figure_4_7_volume():
    components = _pedersen_light_components()
    hexane = ComponentSet(
        ("c6",),
        components.critical_temperature[-1:],
        components.critical_pressure[-1:],
        components.acentric_factor[-1:],
        components.molar_mass[-1:],
    )
    parent = soave_redlich_kwong(hexane)
    translated = soave_redlich_kwong(
        hexane,
        volume_translation=pedersen_peneloux_translation(hexane, "srk"),
    )
    temperature = torch.tensor(288.15, dtype=torch.float64)
    pressure = torch.tensor(1.0e5, dtype=torch.float64)
    composition = torch.ones(1, dtype=torch.float64)
    parent_cm3 = parent.molar_volume(temperature, pressure, composition, "liquid") * 1.0e6
    translated_cm3 = translated.molar_volume(temperature, pressure, composition, "liquid") * 1.0e6
    assert float(parent_cm3) == pytest.approx(148.0, abs=0.2)
    assert float(translated_cm3) == pytest.approx(130.0, abs=0.7)


@pytest.mark.parametrize(
    ("eos", "expected"),
    [
        (
            "pr",
            [
                -0.1927,
                -0.0817,
                -0.1288,
                -0.1595,
                -0.1134,
                -0.0863,
                -0.0844,
                -0.0675,
                -0.0608,
                -0.0390,
                -0.0080,
                0.0033,
                0.0314,
                0.0408,
                0.0655,
            ],
        ),
        (
            "srk",
            [
                -0.0079,
                0.0833,
                0.0466,
                0.0234,
                0.0605,
                0.0825,
                0.0830,
                0.0975,
                0.1022,
                0.1209,
                0.1467,
                0.1554,
                0.1794,
                0.1868,
                0.2080,
            ],
        ),
    ],
)
def test_whitson_table_4_3_shift_factors(eos, expected):
    components = component_set(
        (
            "nitrogen",
            "carbon_dioxide",
            "hydrogen_sulfide",
            "methane",
            "ethane",
            "propane",
            "isobutane",
            "n_butane",
            "isopentane",
            "n_pentane",
            "n_hexane",
            "n_heptane",
            "n_octane",
            "n_nonane",
            "n_decane",
        ),
        dtype=torch.float64,
    )
    translation = whitson_volume_translation(components, eos)
    omega_b = 0.07779607390388846 if eos == "pr" else 0.08664034996495773
    covolume = omega_b * R * components.critical_temperature / components.critical_pressure
    recovered_shift_factor = -translation.reference_shift / covolume
    torch.testing.assert_close(
        recovered_shift_factor,
        torch.tensor(expected, dtype=torch.float64),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_whitson_heavy_family_correlation():
    components = ComponentSet(
        ("c20_plus",),
        torch.tensor([800.0], dtype=torch.float64),
        torch.tensor([1.0e6], dtype=torch.float64),
        torch.tensor([0.9], dtype=torch.float64),
        torch.tensor([0.300], dtype=torch.float64),
    )
    translation = whitson_volume_translation(
        components,
        "pr",
        heavy_families={"c20_plus": "paraffin"},
    )
    covolume = 0.07779607390388846 * R * 800.0 / 1.0e6
    expected_factor = 1.0 - 2.258 / 300.0**0.1823
    assert float(-translation.reference_shift[0] / covolume) == pytest.approx(
        expected_factor,
        rel=1.0e-13,
    )
    with pytest.raises(ValueError, match="heavy_families"):
        whitson_volume_translation(components, "pr")


def test_density_matched_translation_and_errors():
    eos_volume = torch.tensor([150.0e-6], dtype=torch.float64)
    molar_mass = torch.tensor([0.086], dtype=torch.float64)
    density = torch.tensor([660.0], dtype=torch.float64)
    translation = density_matched_translation(eos_volume, molar_mass, density)
    torch.testing.assert_close(
        eos_volume + translation.reference_shift,
        molar_mass / density,
    )
    with pytest.raises(ValueError, match="same shape"):
        density_matched_translation(eos_volume, molar_mass, torch.ones(2))
    with pytest.raises(ValueError, match="finite"):
        density_matched_translation(eos_volume, molar_mass, torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="positive"):
        density_matched_translation(eos_volume, molar_mass, torch.tensor([0.0]))


def test_pedersen_temperature_dependent_density_targets():
    components = component_set(("n_decane",), dtype=torch.float64)
    parent = soave_redlich_kwong(components)
    density_at_288 = torch.tensor([734.2], dtype=torch.float64)
    translation = pedersen_temperature_dependent_translation(
        parent,
        density_at_288,
        pressure=1.0e5,
    )
    translated = soave_redlich_kwong(components, volume_translation=translation)
    composition = torch.ones(1, dtype=torch.float64)
    pressure = torch.tensor(1.0e5, dtype=torch.float64)
    densities = []
    for value in (288.15, 353.15):
        temperature = torch.tensor(value, dtype=torch.float64)
        volume = translated.molar_volume(temperature, pressure, composition, "liquid")
        densities.append(components.molar_mass[0] / volume)
    expansion = 613.9723 / density_at_288.square()
    target = density_at_288 * torch.exp(-expansion * 65.0 * (1.0 + 0.8 * expansion * 65.0))
    torch.testing.assert_close(densities[0], density_at_288[0])
    torch.testing.assert_close(densities[1], target[0])


def test_pedersen_temperature_dependent_input_errors():
    decane = component_set(("n_decane",), dtype=torch.float64)
    model = soave_redlich_kwong(decane)
    with pytest.raises(ValueError, match="one value"):
        pedersen_temperature_dependent_translation(model, torch.ones(2))
    with pytest.raises(ValueError, match="finite and positive"):
        pedersen_temperature_dependent_translation(model, torch.tensor([0.0]))
    with pytest.raises(ValueError, match="pressure"):
        pedersen_temperature_dependent_translation(model, torch.tensor([700.0]), pressure=0.0)
    with pytest.raises(ValueError, match="pressure"):
        pedersen_temperature_dependent_translation(
            model,
            torch.tensor([700.0]),
            pressure=float("nan"),
        )
    shifted = soave_redlich_kwong(
        decane,
        volume_translation=torch.tensor([-1.0e-6]),
    )
    with pytest.raises(ValueError, match="untranslated parent"):
        pedersen_temperature_dependent_translation(shifted, torch.tensor([700.0]))

    methane = component_set(("methane",), dtype=torch.float64)
    with pytest.raises(InvalidStateError, match="below every"):
        pedersen_temperature_dependent_translation(
            soave_redlich_kwong(methane),
            torch.tensor([400.0]),
        )


def test_constant_translation_preserves_binary_saturation():
    components = component_set(("methane", "n_decane"), dtype=torch.float64)
    kij = torch.tensor([[0.0, 0.0409], [0.0409, 0.0]], dtype=torch.float64)
    parent = peng_robinson_1978(components, kij=kij)
    translated = peng_robinson_1978(
        components,
        kij=kij,
        volume_translation=whitson_volume_translation(components, "pr"),
    )
    temperature = torch.tensor(310.9277777778, dtype=torch.float64)
    composition = torch.tensor([0.4, 0.6], dtype=torch.float64)
    parent_point = saturation_point(
        parent,
        temperature,
        composition,
        "bubble",
        tolerance=1.0e-10,
        max_iterations=80,
    )
    translated_point = saturation_point(
        translated,
        temperature,
        composition,
        "bubble",
        tolerance=1.0e-10,
        max_iterations=80,
    )
    assert parent_point.converged and translated_point.converged
    torch.testing.assert_close(
        translated_point.pressure,
        parent_point.pressure,
        rtol=1.0e-12,
        atol=1.0e-5,
    )
    torch.testing.assert_close(
        translated_point.incipient_composition,
        parent_point.incipient_composition,
        rtol=1.0e-12,
        atol=1.0e-13,
    )


def test_whitson_translation_against_segovia_methane_decane_density(
    not_cleared_data: Path,
):
    """Validate both cubic translations on 86 primary experimental states."""
    path = not_cleared_data / "segovia_2017_methane_n_decane_density.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["series"] in {"80_MPa_isobar", "323_K_isotherm"}
        ]
    temperature = torch.tensor([float(row["T_K"]) for row in rows], dtype=torch.float64)
    pressure = torch.tensor([float(row["P_MPa"]) for row in rows], dtype=torch.float64) * 1.0e6
    methane = torch.tensor([float(row["x_methane"]) for row in rows], dtype=torch.float64)
    composition = torch.stack((methane, 1.0 - methane), dim=-1)
    measured = (
        torch.tensor([float(row["density_g_cm3"]) for row in rows], dtype=torch.float64) * 1000.0
    )
    # Segovia et al. (2017), Table 2 model inputs, retained exactly as printed.
    components = ComponentSet(
        ("methane", "n_decane"),
        torch.tensor([190.56, 617.70], dtype=torch.float64),
        torch.tensor([4.599, 2.110], dtype=torch.float64) * 1.0e6,
        torch.tensor([0.0115, 0.4923], dtype=torch.float64),
        torch.tensor([16.04246, 142.28168], dtype=torch.float64) / 1000.0,
    )
    mixture_mass = torch.sum(composition * components.molar_mass, dim=-1)

    def density_aad(model) -> torch.Tensor:
        predicted = mixture_mass / model.molar_volume(
            temperature,
            pressure,
            composition,
            "liquid",
        )
        return (100.0 * (predicted - measured) / measured).abs().mean()

    srk_kij = torch.tensor([[0.0, 0.0411], [0.0411, 0.0]], dtype=torch.float64)
    pr_kij = torch.tensor([[0.0, 0.0409], [0.0409, 0.0]], dtype=torch.float64)
    srk_parent = soave_redlich_kwong(components, kij=srk_kij)
    srk_translated = soave_redlich_kwong(
        components,
        kij=srk_kij,
        volume_translation=whitson_volume_translation(components, "srk"),
    )
    pr_parent = peng_robinson_1978(components, kij=pr_kij)
    pr_translated = peng_robinson_1978(
        components,
        kij=pr_kij,
        volume_translation=whitson_volume_translation(components, "pr"),
    )
    srk_parent_aad = density_aad(srk_parent)
    srk_translated_aad = density_aad(srk_translated)
    pr_parent_aad = density_aad(pr_parent)
    pr_translated_aad = density_aad(pr_translated)
    assert float(srk_translated_aad) < 2.5
    assert float(pr_translated_aad) < 2.0
    assert srk_translated_aad < srk_parent_aad
    assert pr_translated_aad < pr_parent_aad


def test_temperature_dependent_translation_pressure_and_fugacity_identities():
    components = component_set(("n_decane",), dtype=torch.float64)
    parent = soave_redlich_kwong(components)
    translation = pedersen_temperature_dependent_translation(
        parent,
        torch.tensor([734.2], dtype=torch.float64),
        pressure=1.0e5,
    )
    model = soave_redlich_kwong(components, volume_translation=translation)
    temperature = torch.tensor(330.0, dtype=torch.float64)
    pressure = torch.tensor(5.0e6, dtype=torch.float64)
    composition = torch.ones(1, dtype=torch.float64)
    volume = model.molar_volume(temperature, pressure, composition, "liquid")
    torch.testing.assert_close(
        model.pressure(temperature, volume, composition),
        pressure,
        rtol=1.0e-12,
        atol=1.0e-5,
    )
    pressure_from_helmholtz = R * temperature / volume - R * temperature * torch.func.grad(
        lambda current_volume: model.residual_helmholtz_rt(
            temperature,
            current_volume,
            composition,
        )
    )(volume)
    torch.testing.assert_close(
        pressure_from_helmholtz,
        pressure,
        rtol=2.0e-12,
        atol=1.0e-5,
    )
    log_phi_difference = model.log_fugacity_coefficients(
        temperature,
        pressure,
        composition,
        "liquid",
    ) - parent.log_fugacity_coefficients(
        temperature,
        pressure,
        composition,
        "liquid",
    )
    expected = pressure * translation.at_temperature(temperature) / (R * temperature)
    torch.testing.assert_close(log_phi_difference, expected)


def test_volume_translation_parameter_source_errors():
    with pytest.raises(ParameterDatabaseError, match="not 'volume_translation'"):
        rackett_compressibility_factor(
            torch.tensor([0.1]),
            source="cubic.pr-1978",
        )
    wrong_model = ModelParameterSet(
        "volume-translation.wrong",
        "volume_translation",
        "wrong",
        "1",
        {},
    )
    with pytest.raises(ParameterDatabaseError, match="not 'Pedersen-Peneloux'"):
        rackett_compressibility_factor(torch.tensor([0.1]), source=wrong_model)
    missing_mapping = ModelParameterSet(
        "volume-translation.missing-mapping",
        "volume_translation",
        "Pedersen-Peneloux",
        "1",
        {"rackett": []},
    )
    with pytest.raises(ParameterDatabaseError, match="requires a 'rackett' mapping"):
        rackett_compressibility_factor(torch.tensor([0.1]), source=missing_mapping)
    nonnumeric = ModelParameterSet(
        "volume-translation.nonnumeric",
        "volume_translation",
        "Pedersen-Peneloux",
        "1",
        {"rackett": {"intercept": "not-a-number", "acentric_slope": -0.08775}},
    )
    with pytest.raises(ParameterDatabaseError, match="numeric 'intercept'"):
        rackett_compressibility_factor(torch.tensor([0.1]), source=nonnumeric)

    invalid_temperatures = ModelParameterSet(
        "volume-translation.invalid-temperatures",
        "volume_translation",
        "Pedersen-Peneloux",
        "1",
        {
            "temperature_dependent": {
                "reference_temperature": 353.15,
                "target_temperature": 288.15,
                "astm_density_constant": 613.9723,
                "nonlinear_factor": 0.8,
            }
        },
    )
    decane = component_set(("n_decane",), dtype=torch.float64)
    with pytest.raises(ParameterDatabaseError, match="reference_temperature"):
        pedersen_temperature_dependent_translation(
            soave_redlich_kwong(decane),
            torch.tensor([700.0]),
            source=invalid_temperatures,
        )
    invalid_astm = ModelParameterSet(
        "volume-translation.invalid-astm",
        "volume_translation",
        "Pedersen-Peneloux",
        "1",
        {
            "temperature_dependent": {
                "reference_temperature": 288.15,
                "target_temperature": 353.15,
                "astm_density_constant": 0.0,
                "nonlinear_factor": -0.1,
            }
        },
    )
    with pytest.raises(ParameterDatabaseError, match="positive ASTM"):
        pedersen_temperature_dependent_translation(
            soave_redlich_kwong(decane),
            torch.tensor([700.0]),
            source=invalid_astm,
        )
