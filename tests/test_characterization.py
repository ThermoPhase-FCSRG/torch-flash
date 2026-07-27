from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from torch_flash import (
    PseudoComponentCut,
    SCNDistribution,
    equal_weight_lump,
    pedersen_cubic_properties,
    pedersen_density_split,
    pedersen_logarithmic_split,
    whitson_gamma_split,
)
from torch_flash.database import ModelParameterSet, load_model_parameters
from torch_flash.exceptions import ConvergenceError, ParameterDatabaseError

DTYPE = torch.float64


def test_pedersen_split_float32_mass_balance_respects_dtype_precision():
    target_mass = torch.tensor(0.2, dtype=torch.float32)
    split = pedersen_logarithmic_split(
        0.1,
        target_mass,
        max_carbon_number=30,
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        split.average_molar_mass,
        target_mass,
        rtol=8.0 * torch.finfo(torch.float32).eps,
        atol=0.0,
    )


def test_pedersen_split_density_cubic_adapters_and_lumping():
    target_mass = torch.tensor(0.377, dtype=DTYPE, requires_grad=True)
    split = pedersen_logarithmic_split(
        0.00833,
        target_mass,
        first_carbon_number=20,
        max_carbon_number=80,
    )
    torch.testing.assert_close(split.total_mole_fraction, torch.tensor(0.00833, dtype=DTYPE))
    torch.testing.assert_close(split.average_molar_mass, target_mass, atol=1.0e-12, rtol=0.0)
    split.average_molar_mass.backward()
    torch.testing.assert_close(
        target_mass.grad, torch.ones_like(target_mass), atol=2.0e-10, rtol=0.0
    )

    with_density = pedersen_density_split(
        split,
        873.0,
        anchor_density=841.0,
        anchor_carbon_number=19,
    )
    assert with_density.bulk_density is not None
    torch.testing.assert_close(
        with_density.bulk_density,
        torch.tensor(873.0, dtype=DTYPE),
        atol=1.0e-8,
        rtol=0.0,
    )
    srk = pedersen_cubic_properties(with_density, "SRK")
    pr = pedersen_cubic_properties(with_density, "PR")
    assert bool((srk.critical_temperature > 0.0).all())
    assert bool((pr.critical_pressure > 0.0).all())
    assert not torch.allclose(srk.critical_temperature, pr.critical_temperature)
    # The first characterized C20 state is consistent with Pedersen Table 5.8
    # (about 734 K, 14.9 bar, and omega about 0.93).
    assert float(srk.critical_temperature[0].detach()) == pytest.approx(734.0, rel=0.01)
    assert float((srk.critical_pressure[0] / 1.0e5).detach()) == pytest.approx(14.9, rel=0.03)
    assert float(srk.acentric_factor[0].detach()) == pytest.approx(0.93, rel=0.04)

    lumped = equal_weight_lump(
        with_density,
        3,
        properties={
            "critical_temperature": srk.critical_temperature,
            "critical_pressure": srk.critical_pressure,
            "acentric_factor": srk.acentric_factor,
        },
    )
    assert lumped.names == ("C20-C24", "C25-C31", "C32-C80")
    torch.testing.assert_close(lumped.mole_fractions.sum(), split.total_mole_fraction)
    torch.testing.assert_close(
        torch.sum(lumped.mole_fractions * lumped.molar_masses),
        torch.sum(split.mole_fractions * split.molar_masses),
    )
    assert lumped.densities is not None
    assert set(lumped.properties) == {
        "critical_temperature",
        "critical_pressure",
        "acentric_factor",
    }
    moved = with_density.to(dtype=torch.float32)
    assert moved.mole_fractions.dtype == torch.float32
    assert moved.carbon_numbers.dtype == torch.int64


def test_pedersen_heavy_aromatic_c200_structured_regression(num_regression):
    distribution = pedersen_logarithmic_split(
        0.7934,
        0.5302,
        first_carbon_number=7,
        parameter_set="pedersen-c200",
    )
    distribution = pedersen_density_split(
        distribution,
        1009.0,
        parameter_set="pedersen-c200",
    )
    selected = torch.tensor([0, 73, 193])
    srk = pedersen_cubic_properties(distribution, "SRK", "pedersen-c200")
    pr = pedersen_cubic_properties(distribution, "PR", "pedersen-c200")
    num_regression.check(
        {
            "carbon_number": distribution.carbon_numbers[selected].numpy(),
            "mole_fraction": distribution.mole_fractions[selected].detach().numpy(),
            "density_kg_m3": distribution.densities[selected].detach().numpy(),
            "srk_critical_temperature_K": srk.critical_temperature[selected].detach().numpy(),
            "srk_critical_pressure_Pa": srk.critical_pressure[selected].detach().numpy(),
            "srk_acentric_factor": srk.acentric_factor[selected].detach().numpy(),
            "pr_critical_temperature_K": pr.critical_temperature[selected].detach().numpy(),
            "pr_critical_pressure_Pa": pr.critical_pressure[selected].detach().numpy(),
            "pr_acentric_factor": pr.acentric_factor[selected].detach().numpy(),
        },
        basename="pedersen_heavy_aromatic_c200",
        default_tolerance={"rtol": 1.0e-5, "atol": 0.0},
    )
    assert distribution.carbon_numbers[-1] == 200
    assert pr.m[-1] < pr.m[-2]


def test_whitson_gamma_distribution_preserves_moments_and_tail():
    distribution = whitson_gamma_split(
        0.041,
        0.227,
        first_carbon_number=7,
        max_carbon_number=40,
        shape=0.817,
    )
    torch.testing.assert_close(
        distribution.total_mole_fraction,
        torch.tensor(0.041, dtype=DTYPE),
        atol=1.0e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        distribution.average_molar_mass,
        torch.tensor(0.227, dtype=DTYPE),
        atol=1.0e-13,
        rtol=0.0,
    )
    # The final bin includes the complete infinite tail and therefore has an
    # average mass above its nominal C40 centre.
    assert distribution.molar_masses[-1] > (14.0 * 40.0 - 4.0) * 1.0e-3

    exponential = whitson_gamma_split(
        0.1,
        0.2,
        first_carbon_number=7,
        max_carbon_number=20,
        shape=1.0,
        minimum_molar_mass=0.09,
        dtype=torch.float32,
    )
    assert exponential.mole_fractions.dtype == torch.float32
    torch.testing.assert_close(
        exponential.average_molar_mass,
        torch.tensor(0.2, dtype=torch.float32),
        atol=2.0e-7,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("shape", "published_mole_fractions", "published_molar_masses"),
    [
        (
            0.5,
            [
                0.2787233,
                0.1073842,
                0.0772607,
                0.0610991,
                0.0505020,
                0.0428377,
                0.0369618,
                0.0322804,
                0.0284480,
                0.0252470,
                0.0225321,
                0.0202013,
                0.0181808,
                0.0164152,
                0.0148619,
                0.0134879,
                0.0122665,
                0.0111762,
                0.0101996,
                0.1199341,
            ],
            [
                94.588,
                110.525,
                124.690,
                138.758,
                152.796,
                166.819,
                180.836,
                194.848,
                208.857,
                222.864,
                236.870,
                250.875,
                264.879,
                278.883,
                292.886,
                306.888,
                320.890,
                334.892,
                348.894,
                539.651,
            ],
        ),
        (
            1.0,
            [
                0.1195065,
                0.1052247,
                0.0926497,
                0.0815774,
                0.0718284,
                0.0632444,
                0.0556863,
                0.0490314,
                0.0431719,
                0.0380125,
                0.0334698,
                0.0294699,
                0.0259481,
                0.0228471,
                0.0201167,
                0.0177127,
                0.0155959,
                0.0137321,
                0.0120910,
                0.0890834,
            ],
            [
                96.852,
                110.852,
                124.852,
                138.852,
                152.852,
                166.852,
                180.852,
                194.852,
                208.852,
                222.852,
                236.852,
                250.852,
                264.852,
                278.852,
                292.852,
                306.852,
                320.852,
                334.852,
                348.852,
                466.000,
            ],
        ),
        (
            2.0,
            [
                0.0273900,
                0.0655834,
                0.0852269,
                0.0927292,
                0.0925552,
                0.0877762,
                0.0804707,
                0.0720157,
                0.0632969,
                0.0548597,
                0.0470180,
                0.0399302,
                0.0336535,
                0.0281813,
                0.0234690,
                0.0194514,
                0.0160543,
                0.0132017,
                0.0108204,
                0.0463166,
            ],
            [
                99.132,
                111.490,
                125.172,
                139.038,
                152.963,
                166.916,
                180.883,
                194.859,
                208.841,
                222.826,
                236.814,
                250.805,
                264.797,
                278.790,
                292.784,
                306.778,
                320.774,
                334.770,
                348.766,
                420.424,
            ],
        ),
    ],
)
def test_whitson_table_5_4_gamma_split(
    shape,
    published_mole_fractions,
    published_molar_masses,
):
    """Reproduce all 20 rows of Whitson and Brule (2000), Table 5.4."""
    distribution = whitson_gamma_split(
        1.0,
        0.200,
        first_carbon_number=7,
        max_carbon_number=26,
        shape=shape,
        minimum_molar_mass=0.090,
    )
    torch.testing.assert_close(
        distribution.mole_fractions,
        torch.tensor(published_mole_fractions, dtype=DTYPE),
        atol=6.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        distribution.molar_masses * 1.0e3,
        torch.tensor(published_molar_masses, dtype=DTYPE),
        atol=1.0e-3,
        rtol=0.0,
    )


def test_pedersen_tables_5_8_and_5_9_characterization_and_lumping():
    """Reproduce Pedersen et al. (2024) North Sea condensate example."""
    measured = SCNDistribution(
        torch.arange(7, 20),
        torch.tensor(
            [
                0.95,
                1.08,
                0.78,
                0.592,
                0.467,
                0.345,
                0.375,
                0.304,
                0.237,
                0.208,
                0.220,
                0.169,
                0.140,
            ],
            dtype=DTYPE,
        )
        / 100.0,
        torch.tensor(
            [95, 106, 116, 133, 152, 164, 179, 193, 209, 218, 239, 250, 264],
            dtype=DTYPE,
        )
        * 1.0e-3,
        torch.tensor(
            [
                0.726,
                0.747,
                0.769,
                0.781,
                0.778,
                0.785,
                0.802,
                0.815,
                0.817,
                0.824,
                0.825,
                0.831,
                0.841,
            ],
            dtype=DTYPE,
        )
        * 1.0e3,
    )
    plus = pedersen_logarithmic_split(
        0.00833,
        0.377,
        first_carbon_number=20,
        max_carbon_number=80,
    )
    plus = pedersen_density_split(
        plus,
        873.0,
        anchor_density=841.0,
        anchor_carbon_number=19,
    )
    assert plus.densities is not None
    characterized = SCNDistribution(
        torch.cat((measured.carbon_numbers, plus.carbon_numbers)),
        torch.cat((measured.mole_fractions, plus.mole_fractions)),
        torch.cat((measured.molar_masses, plus.molar_masses)),
        torch.cat((measured.densities, plus.densities)),  # type: ignore[arg-type]
    )
    properties = pedersen_cubic_properties(characterized, "SRK")

    # Table 5.8 prints rounded values. These selected rows span the split.
    for carbon_number, mole_percent, density, tc_celsius, pc_bar, omega in (
        (20, 0.1010, 0.845, 460.8, 14.87, 0.932),
        (28, 0.0359, 0.873, 546.0, 13.12, 1.151),
        (40, 0.00761, 0.902, 656.3, 12.17, 1.344),
        (60, 0.000574, 0.936, 819.7, 11.75, 1.253),
        (80, 0.0000432, 0.960, 970.7, 11.74, 0.697),
    ):
        index = carbon_number - 7
        assert float(characterized.mole_fractions[index].detach() * 100.0) == pytest.approx(
            mole_percent, rel=0.012
        )
        assert float(characterized.densities[index].detach() / 1.0e3) == pytest.approx(
            density, abs=0.002
        )
        assert float(properties.critical_temperature[index].detach() - 273.15) == pytest.approx(
            tc_celsius, abs=1.2
        )
        assert float(properties.critical_pressure[index].detach() / 1.0e5) == pytest.approx(
            pc_bar, abs=0.08
        )
        assert float(properties.acentric_factor[index].detach()) == pytest.approx(omega, abs=0.006)

    lumped = equal_weight_lump(
        characterized,
        3,
        properties={
            "critical_temperature": properties.critical_temperature,
            "critical_pressure": properties.critical_pressure,
            "acentric_factor": properties.acentric_factor,
        },
    )
    assert lumped.names == ("C7-C11", "C12-C18", "C19-C80")
    torch.testing.assert_close(
        lumped.mole_fractions * 100.0,
        torch.tensor([3.87, 1.86, 0.97], dtype=DTYPE),
        atol=0.006,
        rtol=0.0,
    )
    torch.testing.assert_close(
        lumped.properties["critical_temperature"],
        torch.tensor([568.0, 668.9, 817.3], dtype=DTYPE),
        atol=0.12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        lumped.properties["critical_pressure"] / 1.0e5,
        torch.tensor([26.8, 17.4, 13.5], dtype=DTYPE),
        atol=0.04,
        rtol=0.0,
    )
    torch.testing.assert_close(
        lumped.properties["acentric_factor"],
        torch.tensor([0.530, 0.762, 1.108], dtype=DTYPE),
        atol=6.0e-4,
        rtol=0.0,
    )


def test_characterization_data_structure_validation():
    cut = PseudoComponentCut("C20+", 0.1, 650.0, 0.9, 0.35)
    assert cut.name == "C20+"
    with pytest.raises(ValueError, match="name"):
        PseudoComponentCut("", 0.1, 650.0, 0.9, 0.35)
    with pytest.raises(ValueError, match="finite"):
        PseudoComponentCut("bad", float("nan"), 650.0, 0.9, 0.35)
    with pytest.raises(ValueError, match="nonnegative"):
        PseudoComponentCut("bad", -0.1, 650.0, 0.9, 0.35)
    with pytest.raises(ValueError, match="positive"):
        PseudoComponentCut("bad", 0.1, 0.0, 0.9, 0.35)

    carbon = torch.tensor([7, 8])
    fractions = torch.tensor([0.1, 0.2], dtype=DTYPE)
    masses = torch.tensor([0.09, 0.11], dtype=DTYPE)
    base = SCNDistribution(carbon, fractions, masses)
    assert base.bulk_density is None
    with pytest.raises(ValueError, match="one-dimensional"):
        SCNDistribution(carbon[None, :], fractions, masses)
    with pytest.raises(ValueError, match="same nonzero"):
        SCNDistribution(carbon, fractions[:1], masses)
    with pytest.raises(ValueError, match="densities"):
        SCNDistribution(carbon, fractions, masses, torch.ones(1))
    with pytest.raises(ValueError, match="finite"):
        SCNDistribution(carbon, fractions, torch.tensor([0.09, torch.nan]))
    with pytest.raises(ValueError, match="nonnegative"):
        SCNDistribution(carbon, -fractions, masses)
    with pytest.raises(ValueError, match="molar masses"):
        SCNDistribution(carbon, fractions, -masses)
    with pytest.raises(ValueError, match="densities"):
        SCNDistribution(carbon, fractions, masses, -torch.ones(2))
    with pytest.raises(ValueError, match="increasing"):
        SCNDistribution(carbon.flip(0), fractions, masses)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"plus_mole_fraction": 0.0, "plus_molar_mass": 0.2}, "mole fraction"),
        ({"plus_mole_fraction": 0.1, "plus_molar_mass": 0.0}, "molar mass"),
        (
            {
                "plus_mole_fraction": 0.1,
                "plus_molar_mass": 0.2,
                "first_carbon_number": 20,
                "max_carbon_number": 10,
            },
            "bounds",
        ),
        (
            {
                "plus_mole_fraction": 0.1,
                "plus_molar_mass": 2.0,
                "max_carbon_number": 20,
            },
            "outside",
        ),
    ],
)
def test_pedersen_split_input_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        pedersen_logarithmic_split(**kwargs)


def test_density_cubic_and_lumping_input_errors():
    distribution = pedersen_logarithmic_split(0.1, 0.2, max_carbon_number=30)
    with pytest.raises(ValueError, match="plus density"):
        pedersen_density_split(distribution, -1.0)
    with pytest.raises(ValueError, match="anchor density"):
        pedersen_density_split(distribution, 850.0, anchor_density=-1.0)
    with pytest.raises(ValueError, match="anchor carbon"):
        pedersen_density_split(distribution, 850.0, anchor_carbon_number=0)
    with pytest.raises(ValueError, match="require characterized"):
        pedersen_cubic_properties(distribution, "SRK")
    with pytest.raises(ValueError, match="eos"):
        pedersen_cubic_properties(
            pedersen_density_split(distribution, 850.0),
            "CPA",
        )
    with pytest.raises(ValueError, match="groups"):
        equal_weight_lump(distribution, 0)
    with pytest.raises(ValueError, match="groups"):
        equal_weight_lump(distribution, 100)
    with pytest.raises(ValueError, match="names"):
        equal_weight_lump(distribution, 2, properties={"": distribution.molar_masses})
    with pytest.raises(ValueError, match="must match"):
        equal_weight_lump(distribution, 2, properties={"bad": torch.ones(1)})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"plus_mole_fraction": 0.0, "plus_molar_mass": 0.2}, "finite and positive"),
        ({"plus_mole_fraction": 0.1, "plus_molar_mass": 0.2, "shape": 0.0}, "finite and positive"),
        (
            {
                "plus_mole_fraction": 0.1,
                "plus_molar_mass": 0.2,
                "first_carbon_number": 20,
                "max_carbon_number": 10,
            },
            "bounds",
        ),
        (
            {
                "plus_mole_fraction": 0.1,
                "plus_molar_mass": 0.2,
                "minimum_molar_mass": 0.3,
            },
            "minimum molar mass",
        ),
    ],
)
def test_whitson_gamma_input_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        whitson_gamma_split(**kwargs)


def test_characterization_parameter_payload_errors(monkeypatch):
    wrong_kind = ModelParameterSet("bad.kind", "cpa", "CPA", "1", {})
    with pytest.raises(ParameterDatabaseError, match="not 'characterization'"):
        pedersen_logarithmic_split(0.1, 0.2, parameter_set=wrong_kind)

    base = load_model_parameters("pedersen-characterization").as_dict()
    missing_split = deepcopy(base)
    missing_split.pop("plus_split")
    with pytest.raises(ParameterDatabaseError, match="plus_split"):
        pedersen_logarithmic_split(
            0.1,
            0.2,
            parameter_set=ModelParameterSet(
                "bad.split", "characterization", "test", "1", missing_split
            ),
        )
    bad_mass = deepcopy(base)
    bad_mass["plus_split"]["molecular_weight"] = []
    with pytest.raises(ParameterDatabaseError, match="molecular_weight"):
        pedersen_logarithmic_split(
            0.1,
            0.2,
            parameter_set=ModelParameterSet("bad.mass", "characterization", "test", "1", bad_mass),
        )
    bad_density = deepcopy(base)
    bad_density["plus_split"]["density_log_carbon_number"] = False
    distribution = pedersen_logarithmic_split(0.1, 0.2)
    with pytest.raises(ParameterDatabaseError, match="logarithmic density"):
        pedersen_density_split(
            distribution,
            850.0,
            parameter_set=ModelParameterSet(
                "bad.density", "characterization", "test", "1", bad_density
            ),
        )

    with_density = pedersen_density_split(distribution, 850.0)
    no_cubic = deepcopy(base)
    no_cubic.pop("cubic_properties")
    with pytest.raises(ParameterDatabaseError, match="cubic_properties"):
        pedersen_cubic_properties(
            with_density,
            "SRK",
            ModelParameterSet("bad.cubic", "characterization", "test", "1", no_cubic),
        )

    monkeypatch.setattr(
        "torch_flash.characterization.distributions.torch.abs",
        lambda value: torch.ones_like(value),
    )
    with pytest.raises(ConvergenceError, match="mass balance"):
        pedersen_logarithmic_split(0.1, 0.2)


def test_characterization_additional_payload_and_shape_errors(monkeypatch):
    pedersen = load_model_parameters("pedersen-characterization").as_dict()

    bad_default = deepcopy(pedersen)
    bad_default["plus_split"]["default_max_carbon_number"] = 80.0
    with pytest.raises(ParameterDatabaseError, match="must be an integer"):
        pedersen_logarithmic_split(
            0.1,
            0.2,
            parameter_set=ModelParameterSet(
                "bad.default", "characterization", "test", "1", bad_default
            ),
        )

    bad_slope = deepcopy(pedersen)
    bad_slope["plus_split"]["molecular_weight"]["slope"] = -1.0
    with pytest.raises(ParameterDatabaseError, match="finite and positive"):
        pedersen_logarithmic_split(
            0.1,
            0.2,
            max_carbon_number=30,
            parameter_set=ModelParameterSet(
                "bad.slope", "characterization", "test", "1", bad_slope
            ),
        )

    bad_intercept = deepcopy(pedersen)
    bad_intercept["plus_split"]["molecular_weight"]["intercept"] = "bad"
    with pytest.raises(ParameterDatabaseError, match="intercept"):
        pedersen_logarithmic_split(
            0.1,
            0.2,
            max_carbon_number=30,
            parameter_set=ModelParameterSet(
                "bad.intercept", "characterization", "test", "1", bad_intercept
            ),
        )

    with pytest.raises(ValueError, match="must be scalar"):
        pedersen_logarithmic_split(torch.tensor([0.1]), 0.2, max_carbon_number=30)

    distribution = pedersen_logarithmic_split(0.1, 0.2, max_carbon_number=30)
    no_split = deepcopy(pedersen)
    no_split.pop("plus_split")
    with pytest.raises(ParameterDatabaseError, match="plus_split"):
        pedersen_density_split(
            distribution,
            850.0,
            parameter_set=ModelParameterSet(
                "bad.no-density-split",
                "characterization",
                "test",
                "1",
                no_split,
            ),
        )

    monkeypatch.setattr(
        "torch_flash.characterization.distributions.torch.func.grad",
        lambda function: lambda value: torch.zeros_like(value),
    )
    with pytest.raises(ConvergenceError, match="volume balance"):
        pedersen_density_split(distribution, 850.0)


def test_whitson_additional_payload_and_shape_errors():
    whitson = load_model_parameters("whitson-characterization").as_dict()

    missing_gamma = deepcopy(whitson)
    missing_gamma.pop("gamma_distribution")
    with pytest.raises(ParameterDatabaseError, match="gamma_distribution"):
        whitson_gamma_split(
            0.1,
            0.2,
            parameter_set=ModelParameterSet(
                "bad.gamma", "characterization", "test", "1", missing_gamma
            ),
        )

    with pytest.raises(ValueError, match="must be scalar"):
        whitson_gamma_split(torch.tensor([0.1]), 0.2)

    missing_relation = deepcopy(whitson)
    missing_relation["gamma_distribution"].pop("recommended_minimum_molecular_weight_relation")
    with pytest.raises(ParameterDatabaseError, match="minimum-molecular-weight"):
        whitson_gamma_split(
            0.1,
            0.2,
            parameter_set=ModelParameterSet(
                "bad.eta", "characterization", "test", "1", missing_relation
            ),
        )

    bad_increment = deepcopy(whitson)
    bad_increment["gamma_distribution"]["molecular_weight_boundary_increment"] = 0.0
    with pytest.raises(ParameterDatabaseError, match="finite and positive"):
        whitson_gamma_split(
            0.1,
            0.2,
            minimum_molar_mass=0.09,
            parameter_set=ModelParameterSet(
                "bad.increment",
                "characterization",
                "test",
                "1",
                bad_increment,
            ),
        )


def test_pedersen_cubic_adapter_payload_errors():
    distribution = pedersen_density_split(
        pedersen_logarithmic_split(0.1, 0.2, max_carbon_number=30),
        850.0,
    )
    wrong_kind = ModelParameterSet("wrong.cubic", "cpa", "CPA", "1", {})
    with pytest.raises(ParameterDatabaseError, match="not a characterization"):
        pedersen_cubic_properties(distribution, "SRK", wrong_kind)

    pedersen = load_model_parameters("pedersen-characterization").as_dict()
    missing_srk = deepcopy(pedersen)
    missing_srk["cubic_properties"].pop("SRK")
    with pytest.raises(ParameterDatabaseError, match="no SRK"):
        pedersen_cubic_properties(
            distribution,
            "SRK",
            ModelParameterSet("bad.no-srk", "characterization", "test", "1", missing_srk),
        )

    bad_coefficients = deepcopy(pedersen)
    bad_coefficients["cubic_properties"]["SRK"]["critical_temperature"] = [1.0]
    with pytest.raises(ParameterDatabaseError, match="requires 4"):
        pedersen_cubic_properties(
            distribution,
            "SRK",
            ModelParameterSet(
                "bad.coefficients",
                "characterization",
                "test",
                "1",
                bad_coefficients,
            ),
        )

    negative_discriminant = deepcopy(pedersen)
    negative_discriminant["cubic_properties"]["SRK"]["m_to_acentric_factor"] = [100.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="no real acentric"):
        pedersen_cubic_properties(
            distribution,
            "SRK",
            ModelParameterSet(
                "bad.discriminant",
                "characterization",
                "test",
                "1",
                negative_discriminant,
            ),
        )

    heavy = load_model_parameters("pedersen-c200").as_dict()
    bad_threshold = deepcopy(heavy)
    bad_threshold["cubic_properties"]["PR"]["inverse_mass_m_threshold"] = -1.0
    heavy_distribution = pedersen_density_split(
        pedersen_logarithmic_split(
            0.5,
            0.53,
            first_carbon_number=7,
            parameter_set="pedersen-c200",
        ),
        1009.0,
        parameter_set="pedersen-c200",
    )
    with pytest.raises(ParameterDatabaseError, match="inverse_mass_m_threshold"):
        pedersen_cubic_properties(
            heavy_distribution,
            "PR",
            ModelParameterSet(
                "bad.threshold",
                "characterization",
                "test",
                "1",
                bad_threshold,
            ),
        )
