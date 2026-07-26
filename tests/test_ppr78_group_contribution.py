from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
import yaml

from torch_flash import (
    DEFAULT_EPPR78_GROUP_CONTRIBUTION,
    EPPR78_CCS_GROUP_CONTRIBUTION,
    PPR78_HYDROGEN_WATER_GROUP_CONTRIBUTION,
    ComponentSet,
    ModelParameterSet,
    PPR78GroupContributionParameters,
    available_parameter_sets,
    binary_bubble_point,
    component_set,
    enhanced_predictive_peng_robinson_1978,
    load_model_parameters,
    peng_robinson_1978,
    ppr78_group_contribution_parameters,
    predictive_peng_robinson_1978,
)
from torch_flash.constants import R
from torch_flash.exceptions import ParameterDatabaseError
from torch_flash.mixing import PPR78Mixing

DTYPE = torch.float64
DATA = Path(__file__).parent / "data"
PPR78_ID = "group-contribution.ppr78-jaubert-mutelet-2004"


def test_ppr78_database_preserves_table_1_and_component_decompositions():
    components = component_set(
        ("methane", "ethane", "propane", "isobutane", "n_decane"),
        dtype=DTYPE,
    )
    parameters = ppr78_group_contribution_parameters(components, "ppr78")

    assert PPR78_ID in available_parameter_sets(model_kind="group_contribution")
    assert parameters.parameter_set == PPR78_ID
    assert parameters.group_names == ("CH3", "CH2", "CH", "C", "CH4", "C2H6")
    torch.testing.assert_close(
        parameters.group_fractions,
        torch.tensor(
            [
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [2 / 3, 1 / 3, 0, 0, 0, 0],
                [3 / 4, 0, 1 / 4, 0, 0, 0],
                [0.2, 0.8, 0, 0, 0, 0],
            ],
            dtype=DTYPE,
        ),
    )
    assert parameters.reference_temperature == 298.15
    assert parameters.group_a[0, 1] == 74.81e6
    assert parameters.group_b[0, 1] == 165.7e6
    assert parameters.group_a[2, 3] == -305.7e6
    assert parameters.group_b[0, 4] == -35.0e6
    torch.testing.assert_close(parameters.group_a, parameters.group_a.mT)
    torch.testing.assert_close(parameters.group_b, parameters.group_b.mT)
    assert not bool(torch.diagonal(parameters.group_a).count_nonzero())
    assert not bool(torch.diagonal(parameters.group_b).count_nonzero())


def test_ppr78_hydrogen_water_database_preserves_published_submatrix():
    components = component_set(("hydrogen", "nitrogen", "water"), dtype=DTYPE)
    parameters = ppr78_group_contribution_parameters(
        components,
        "ppr78-hydrogen-water",
    )

    assert PPR78_HYDROGEN_WATER_GROUP_CONTRIBUTION in available_parameter_sets(
        model_kind="group_contribution"
    )
    assert parameters.parameter_set == PPR78_HYDROGEN_WATER_GROUP_CONTRIBUTION
    assert parameters.group_names == ("H2", "N2", "H2O")
    torch.testing.assert_close(parameters.group_fractions, torch.eye(3, dtype=DTYPE))
    torch.testing.assert_close(
        parameters.group_a,
        torch.tensor(
            [
                [0.0, 65.20e6, 830.8e6],
                [65.20e6, 0.0, 2574.0e6],
                [830.8e6, 2574.0e6, 0.0],
            ],
            dtype=DTYPE,
        ),
    )
    torch.testing.assert_close(
        parameters.group_b,
        torch.tensor(
            [
                [0.0, 70.10e6, -137.9e6],
                [70.10e6, 0.0, 5490.0e6],
                [-137.9e6, 5490.0e6, 0.0],
            ],
            dtype=DTYPE,
        ),
    )


def test_eppr78_global_database_preserves_inventory_and_ccs_parameters():
    loaded = load_model_parameters(DEFAULT_EPPR78_GROUP_CONTRIBUTION)
    payload = loaded.as_dict()

    assert EPPR78_CCS_GROUP_CONTRIBUTION == DEFAULT_EPPR78_GROUP_CONTRIBUTION
    assert load_model_parameters("eppr78-ccs").identifier == DEFAULT_EPPR78_GROUP_CONTRIBUTION
    assert DEFAULT_EPPR78_GROUP_CONTRIBUTION in available_parameter_sets(
        model_kind="group_contribution"
    )
    assert loaded.model == "E-PPR78"
    assert loaded.version == "2022-global-40-group"
    assert len(payload["groups"]) == 40
    assert len(payload["interactions"]) == 356
    assert len(payload["unavailable_interactions"]) == 424
    assert len(payload["interactions"]) + len(payload["unavailable_interactions"]) == 780
    assert payload["interactions"]["CH4|CO2"] == {
        "A": 136.6e6,
        "B": 214.8e6,
    }
    assert payload["interactions"]["CO2|H2"] == {
        "A": 261.1e6,
        "B": 300.9e6,
    }
    assert payload["interactions"]["CO2|O2"] == {
        "A": 154.4e6,
        "B": 109.8e6,
    }
    assert payload["interactions"]["H2O|CO"] == {
        "A": 715.1e6,
        "B": -89.90e6,
    }
    assert "O2|NH3" in payload["unavailable_interactions"]


def test_eppr78_selects_active_groups_and_builds_ccs_model():
    components = component_set(
        ("carbon_dioxide", "hydrogen", "water"),
        dtype=DTYPE,
    )
    parameters = ppr78_group_contribution_parameters(
        components,
        DEFAULT_EPPR78_GROUP_CONTRIBUTION,
    )

    assert parameters.group_names == ("CO2", "H2O", "H2")
    torch.testing.assert_close(
        parameters.group_fractions,
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=DTYPE,
        ),
    )
    torch.testing.assert_close(
        parameters.group_a,
        torch.tensor(
            [
                [0.0, 559.3e6, 261.1e6],
                [559.3e6, 0.0, 830.8e6],
                [261.1e6, 830.8e6, 0.0],
            ],
            dtype=DTYPE,
        ),
    )
    model = enhanced_predictive_peng_robinson_1978(components, trainable=True)
    assert isinstance(model.mixing, PPR78Mixing)
    assert model.mixing.parameter_set == DEFAULT_EPPR78_GROUP_CONTRIBUTION
    assert model.mixing.ngroups == 3
    assert model.mixing.raw_group_a.numel() == 3
    kij = model.mixing.kij(
        torch.tensor(300.0, dtype=DTYPE),
        *model.pure_parameters(torch.tensor(300.0, dtype=DTYPE)),
    )
    assert bool(torch.isfinite(kij).all())


def test_eppr78_rejects_an_unavailable_active_ccs_pair():
    with pytest.raises(
        ParameterDatabaseError,
        match=r"no E-PPR78 interaction for active groups 'H2S'\\|'O2'",
    ):
        enhanced_predictive_peng_robinson_1978(
            component_set(("hydrogen_sulfide", "oxygen"), dtype=DTYPE)
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda parameters: parameters.update(unavailable_interactions="H2S|O2"),
            "must be a string list",
        ),
        (
            lambda parameters: parameters["unavailable_interactions"].append("invalid"),
            "keys must have the form",
        ),
        (
            lambda parameters: parameters["unavailable_interactions"].append(
                parameters["unavailable_interactions"][0]
            ),
            "duplicate unavailable interaction",
        ),
        (
            lambda parameters: parameters["unavailable_interactions"].append("CH3|CH2"),
            "both available and unavailable",
        ),
        (
            lambda parameters: parameters["unavailable_interactions"].pop(),
            "explicitly account for all",
        ),
    ],
)
def test_eppr78_rejects_malformed_availability_inventory(mutation, message):
    bundled = load_model_parameters(DEFAULT_EPPR78_GROUP_CONTRIBUTION)
    parameters = bundled.as_dict()
    mutation(parameters)
    invalid = ModelParameterSet(
        identifier="group-contribution.invalid-eppr78",
        model_kind="group_contribution",
        model="E-PPR78",
        version="invalid",
        parameters=parameters,
        units=dict(bundled.units),
    )
    with pytest.raises(ParameterDatabaseError, match=message):
        ppr78_group_contribution_parameters(
            component_set(("propane", "n_butane"), dtype=DTYPE),
            invalid,
        )


def test_eppr78_single_active_group_is_a_valid_pure_component_model():
    model = enhanced_predictive_peng_robinson_1978(component_set(("carbon_dioxide",), dtype=DTYPE))
    assert isinstance(model.mixing, PPR78Mixing)
    assert model.mixing.ngroups == 1
    assert model.mixing.raw_group_a.numel() == 0
    torch.testing.assert_close(
        model.mixing.group_interaction_energy(torch.tensor(300.0, dtype=DTYPE)),
        torch.zeros((1, 1), dtype=DTYPE),
    )


@pytest.mark.parametrize(
    ("components", "printed_kij"),
    [
        (
            ("hydrogen", "water"),
            {
                310.93: -0.9490,
                323.15: -0.8770,
                366.48: -0.6143,
                422.04: -0.2600,
                423.15: -0.2527,
                448.15: -0.0862,
                473.15: 0.0848,
                498.15: 0.2606,
                523.15: 0.4412,
                548.15: 0.6269,
                573.15: 0.8178,
            },
        ),
        (
            ("hydrogen", "nitrogen"),
            {
                63.15: 0.0141,
                63.19: 0.0142,
                70.35: 0.0244,
                77.35: 0.0340,
                77.55: 0.0343,
                83.67: 0.0423,
                86.10: 0.0455,
                90.79: 0.0515,
                99.82: 0.0626,
                100.00: 0.0628,
                109.00: 0.0735,
                110.30: 0.0750,
            },
        ),
    ],
)
def test_ppr78_hydrogen_water_extension_matches_published_figure_kij(
    components,
    printed_kij,
):
    model = predictive_peng_robinson_1978(
        component_set(components, dtype=DTYPE),
        parameter_set=PPR78_HYDROGEN_WATER_GROUP_CONTRIBUTION,
    )
    errors = []
    for temperature_value, reference in printed_kij.items():
        temperature = torch.tensor(temperature_value, dtype=DTYPE)
        pure_a, pure_b = model.pure_parameters(temperature)
        predicted = float(model.mixing.kij(temperature, pure_a, pure_b)[0, 1])
        errors.append(abs(predicted - reference))

    # Figures 7 and 19 print four decimals and the articles use Poling et al.
    # pure constants rather than the package's shared component compilation.
    assert max(errors) < 4.5e-3
    assert sum(errors) / len(errors) < 2.0e-3


def test_ppr78_reproduces_appendix_a_propane_n_butane_calculation():
    # Appendix A uses these rounded pure constants, which intentionally differ
    # slightly from the package's general component database.
    components = ComponentSet(
        ("propane", "n_butane"),
        torch.tensor([369.83, 425.12], dtype=DTYPE),
        torch.tensor([42.48e5, 37.96e5], dtype=DTYPE),
        torch.tensor([0.152, 0.200], dtype=DTYPE),
        torch.tensor([0.04409562, 0.0581222], dtype=DTYPE),
    )
    model = predictive_peng_robinson_1978(components)
    temperature = torch.tensor(303.15, dtype=DTYPE)
    pure_a, pure_b = model.pure_parameters(temperature)
    mixing = model.mixing
    assert isinstance(mixing, PPR78Mixing)

    differences = mixing.group_fractions[0] - mixing.group_fractions[1]
    double_sum = -0.5 * torch.einsum(
        "i,ij,j->",
        differences,
        mixing.group_interaction_energy(temperature),
        differences,
    )
    kij = mixing.kij(temperature, pure_a, pure_b)

    torch.testing.assert_close(
        pure_a,
        torch.tensor([1.1371, 1.8361], dtype=DTYPE),
        rtol=4e-4,
        atol=0,
    )
    torch.testing.assert_close(
        pure_b,
        torch.tensor([5.6313e-5, 7.2440e-5], dtype=DTYPE),
        rtol=8e-5,
        atol=0,
    )
    assert float(double_sum) == pytest.approx(2.036e6, rel=8e-4)
    assert float(kij[0, 1]) == pytest.approx(0.0028, abs=1.0e-5)
    assert float(kij[1, 0]) == pytest.approx(0.0028, abs=1.0e-5)
    assert torch.equal(torch.diagonal(kij), torch.zeros(2, dtype=DTYPE))


def test_ppr78_reproduces_all_figure_3_printed_kij_values(
    not_cleared_data: Path,
):
    with (not_cleared_data / "jaubert_mutelet_2004_ppr78_kij.csv").open() as stream:
        rows = tuple(csv.DictReader(stream))
    assert len(rows) == 15
    assert {row["source_kind"] for row in rows} == {"published PPR78 model value"}

    errors = []
    for system in {(row["component1"], row["component2"]) for row in rows}:
        model = predictive_peng_robinson_1978(component_set(system, dtype=DTYPE))
        assert isinstance(model.mixing, PPR78Mixing)
        for row in rows:
            if (row["component1"], row["component2"]) != system:
                continue
            temperature = torch.tensor(float(row["temperature_K"]), dtype=DTYPE)
            pure_a, pure_b = model.pure_parameters(temperature)
            predicted = float(model.mixing.kij(temperature, pure_a, pure_b)[0, 1])
            errors.append(abs(predicted - float(row["kij_printed"])))
    # The paper prints only 2-4 decimal places and uses a slightly different
    # pure-property compilation than the general package database.
    assert max(errors) < 5.0e-4
    assert sum(errors) / len(errors) < 1.5e-4


def test_ppr78_is_batched_differentiable_and_has_unique_trainable_pairs():
    components = component_set(("methane", "n_decane"), dtype=DTYPE)
    model = predictive_peng_robinson_1978(components, trainable=True)
    mixing = model.mixing
    assert isinstance(mixing, PPR78Mixing)
    assert mixing.raw_group_a.shape == (15,)
    assert mixing.raw_group_b.shape == (15,)
    assert sum(parameter.numel() for parameter in mixing.parameters()) == 30

    temperature = torch.tensor([300.0, 450.0, 600.0], dtype=DTYPE, requires_grad=True)
    pure_a, pure_b = model.pure_parameters(temperature)
    kij = mixing.kij(temperature, pure_a, pure_b)
    assert kij.shape == (3, 2, 2)
    torch.testing.assert_close(kij, kij.mT)
    torch.testing.assert_close(
        torch.diagonal(kij, dim1=-2, dim2=-1),
        torch.zeros((3, 2), dtype=DTYPE),
    )

    loss = kij[:, 0, 1].square().sum()
    gradients = torch.autograd.grad(
        loss,
        (temperature, mixing.raw_group_a, mixing.raw_group_b),
    )
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert all(bool(gradient.abs().sum() > 0.0) for gradient in gradients)


def test_ppr78_closed_form_fugacity_matches_helmholtz_autodiff():
    model = predictive_peng_robinson_1978(component_set(("methane", "n_decane"), dtype=DTYPE))
    temperature = torch.tensor(450.0, dtype=DTYPE)
    pressure = torch.tensor(8.0e6, dtype=DTYPE)
    composition = torch.tensor([0.35, 0.65], dtype=DTYPE)
    z = model.select_z(temperature, pressure, composition, "liquid")
    volume = z * R * temperature / pressure

    def residual(moles: torch.Tensor) -> torch.Tensor:
        return model.residual_helmholtz_rt(temperature, volume, moles)

    autodiff = torch.func.grad(residual)(composition) - torch.log(z)
    closed_form = model.log_fugacity_coefficients(
        temperature,
        pressure,
        composition,
        "liquid",
    )
    torch.testing.assert_close(closed_form, autodiff, rtol=2e-12, atol=2e-12)


def test_ppr78_custom_group_counts_mapping_tensor_and_yaml(tmp_path):
    components = component_set(("propane", "n_butane"), dtype=DTYPE)
    mapping = {
        "propane": {"CH3": 2.0, "CH2": 1.0},
        "n_butane": {"CH3": 2.0, "CH2": 2.0},
    }
    from_mapping = ppr78_group_contribution_parameters(
        components,
        group_counts=mapping,
    )
    from_tensor = ppr78_group_contribution_parameters(
        components,
        group_counts=torch.tensor(
            [[2, 1, 0, 0, 0, 0], [2, 2, 0, 0, 0, 0]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(from_mapping.group_fractions, from_tensor.group_fractions)

    bundled = load_model_parameters(PPR78_ID)
    document = {
        "format": "torch-flash-model-parameters",
        "schema_version": 1,
        "id": "group-contribution.custom-ppr78",
        "model_kind": "group_contribution",
        "model": "PPR78",
        "version": "test",
        "description": "Test copy of the published parameter set.",
        "units": dict(bundled.units),
        "references": [dict(reference) for reference in bundled.references],
        "parameters": bundled.as_dict(),
    }
    path = tmp_path / "custom-ppr78.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    custom = ppr78_group_contribution_parameters(components, path)
    assert custom.parameter_set == "group-contribution.custom-ppr78"
    torch.testing.assert_close(custom.group_a, from_mapping.group_a)
    torch.testing.assert_close(custom.group_b, from_mapping.group_b)


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        (
            lambda document: document.update(model_kind="binary_interaction"),
            ParameterDatabaseError,
            "not 'group_contribution'",
        ),
        (
            lambda document: document.update(model="not-PPR78"),
            ParameterDatabaseError,
            "model must be 'PPR78'",
        ),
        (
            lambda document: document["units"].update(A="bar"),
            ParameterDatabaseError,
            "must declare PPR78",
        ),
        (
            lambda document: document["parameters"].update(reference_temperature=0.0),
            ParameterDatabaseError,
            "must be positive",
        ),
        (
            lambda document: document["parameters"]["interactions"].pop("CH3|CH2"),
            ParameterDatabaseError,
            "explicitly account for all",
        ),
    ],
)
def test_ppr78_parameter_document_errors(mutation, error, message):
    bundled = load_model_parameters(PPR78_ID)
    document = {
        "identifier": "group-contribution.invalid",
        "model_kind": bundled.model_kind,
        "model": bundled.model,
        "version": "invalid",
        "parameters": bundled.as_dict(),
        "units": dict(bundled.units),
    }
    mutation(document)
    parameter_set = ModelParameterSet(
        identifier=document["identifier"],
        model_kind=document["model_kind"],
        model=document["model"],
        version=document["version"],
        parameters=document["parameters"],
        units=document["units"],
    )
    with pytest.raises(error, match=message):
        ppr78_group_contribution_parameters(
            component_set(("propane", "n_butane"), dtype=DTYPE),
            parameter_set,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda parameters: parameters.update(groups=None),
            "groups must be a string list",
        ),
        (
            lambda parameters: parameters.update(groups=["CH3", 2]),
            "groups must be a string list",
        ),
        (
            lambda parameters: parameters.update(groups=["CH3", "CH3"]),
            "groups must contain unique names",
        ),
        (
            lambda parameters: parameters.update(reference_temperature=float("nan")),
            "must be finite and numeric",
        ),
        (
            lambda parameters: parameters.update(interactions=[]),
            "interactions must be a mapping",
        ),
        (
            lambda parameters: parameters["interactions"].update(
                {"invalid-key": {"A": 1.0, "B": 1.0}}
            ),
            "interaction keys must have the form",
        ),
        (
            lambda parameters: parameters["interactions"].update({"CH3|CH3": {"A": 1.0, "B": 1.0}}),
            "invalid group interaction",
        ),
        (
            lambda parameters: parameters["interactions"].update({"CH2|CH3": {"A": 1.0, "B": 1.0}}),
            "duplicate interaction",
        ),
        (
            lambda parameters: parameters["interactions"].update({"CH3|CH2": []}),
            "must be a mapping",
        ),
        (
            lambda parameters: parameters["interactions"].update({"CH3|CH2": {"A": 0.0, "B": 1.0}}),
            "undefined B/A",
        ),
    ],
)
def test_ppr78_rejects_malformed_group_parameter_payloads(mutation, message):
    bundled = load_model_parameters(PPR78_ID)
    parameters = bundled.as_dict()
    mutation(parameters)
    invalid = ModelParameterSet(
        identifier="group-contribution.invalid-payload",
        model_kind="group_contribution",
        model="PPR78",
        version="invalid",
        parameters=parameters,
        units=dict(bundled.units),
    )
    with pytest.raises(ParameterDatabaseError, match=message):
        ppr78_group_contribution_parameters(
            component_set(("propane", "n_butane"), dtype=DTYPE),
            invalid,
        )


def test_ppr78_parameter_dataclass_validates_internal_tensor_shapes():
    fractions = torch.tensor([[1.0, 0.0]], dtype=DTYPE)
    interactions = torch.zeros((2, 2), dtype=DTYPE)
    arguments = (fractions, interactions, interactions, 298.15, "test")
    with pytest.raises(ValueError, match="non-empty and unique"):
        PPR78GroupContributionParameters(("CH3", "CH3"), *arguments)
    with pytest.raises(ValueError, match="fractions must match"):
        PPR78GroupContributionParameters(("CH3", "CH2"), fractions[:, :1], *arguments[1:])
    with pytest.raises(ValueError, match="interactions must match"):
        PPR78GroupContributionParameters(
            ("CH3", "CH2"),
            fractions,
            torch.zeros((3, 3), dtype=DTYPE),
            torch.zeros((3, 3), dtype=DTYPE),
            298.15,
            "test",
        )


def test_ppr78_rejects_unsupported_or_invalid_group_decompositions():
    with pytest.raises(KeyError, match="no PPR78 decomposition"):
        predictive_peng_robinson_1978(component_set(("methane", "carbon_dioxide"), dtype=DTYPE))
    components = component_set(("propane", "n_butane"), dtype=DTYPE)
    with pytest.raises(ValueError, match="shape"):
        ppr78_group_contribution_parameters(
            components,
            group_counts=torch.ones((2, 3)),
        )
    with pytest.raises(ParameterDatabaseError, match="unknown group"):
        ppr78_group_contribution_parameters(
            components,
            group_counts={
                "propane": {"unknown": 1.0},
                "n_butane": {"CH3": 2.0, "CH2": 2.0},
            },
        )
    with pytest.raises(ValueError, match="at least one group"):
        ppr78_group_contribution_parameters(
            components,
            group_counts={
                "propane": {},
                "n_butane": {"CH3": 2.0, "CH2": 2.0},
            },
        )
    with pytest.raises(ParameterDatabaseError, match="cannot be negative"):
        ppr78_group_contribution_parameters(
            components,
            group_counts={
                "propane": {"CH3": -1.0},
                "n_butane": {"CH3": 2.0, "CH2": 2.0},
            },
        )
    with pytest.raises(ParameterDatabaseError, match="must be a mapping"):
        ppr78_group_contribution_parameters(
            components,
            group_counts=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ppr78_group_contribution_parameters(
            components,
            group_counts=torch.tensor(
                [[float("nan"), 1, 0, 0, 0, 0], [2, 2, 0, 0, 0, 0]],
                dtype=DTYPE,
            ),
        )


def test_ppr78_mixing_validates_tensor_contracts():
    fractions = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    zeros = torch.zeros((2, 2), dtype=DTYPE)
    with pytest.raises(ValueError, match="component-by-group"):
        PPR78Mixing(fractions[0], zeros, zeros)
    with pytest.raises(ValueError, match="at least one component"):
        PPR78Mixing(fractions[:0], zeros, zeros)
    with pytest.raises(ValueError, match="one group"):
        PPR78Mixing(fractions[:, :0], zeros[:0, :0], zeros[:0, :0])
    with pytest.raises(ValueError, match="matching the groups"):
        PPR78Mixing(fractions, torch.zeros((3, 3)), torch.zeros((3, 3)))
    with pytest.raises(ValueError, match="must be finite"):
        PPR78Mixing(fractions, torch.tensor([[0.0, float("nan")], [float("nan"), 0.0]]), zeros)
    with pytest.raises(ValueError, match="cannot be negative"):
        PPR78Mixing(torch.tensor([[1.1, -0.1], [0.0, 1.0]]), zeros, zeros)
    with pytest.raises(ValueError, match="sum to one"):
        PPR78Mixing(fractions * 0.5, zeros, zeros)
    with pytest.raises(ValueError, match="symmetric"):
        PPR78Mixing(fractions, torch.tensor([[0.0, 1.0], [0.0, 0.0]]), zeros)
    with pytest.raises(ValueError, match="zero diagonals"):
        PPR78Mixing(fractions, torch.eye(2), zeros)
    with pytest.raises(ValueError, match="wherever group A is zero"):
        PPR78Mixing(fractions, zeros, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="reference_temperature"):
        PPR78Mixing(fractions, zeros, zeros, reference_temperature=float("nan"))
    mixing = PPR78Mixing(fractions, zeros, zeros)
    with pytest.raises(ValueError, match="temperature"):
        mixing.group_interaction_energy(torch.tensor(0.0))
    with pytest.raises(ValueError, match="one value per component"):
        mixing.kij(torch.tensor(300.0), torch.ones(3), torch.ones(3))
    with pytest.raises(ValueError, match="finite and positive"):
        mixing.kij(torch.tensor(300.0), torch.tensor([1.0, -1.0]), torch.ones(2))


@pytest.mark.serial
def test_ppr78_improves_all_selected_methane_n_decane_experimental_isotherms(
    not_cleared_data: Path,
):
    with (not_cleared_data / "jaubert_ppr78_hydrocarbon_vle.csv").open() as stream:
        all_rows = tuple(csv.DictReader(stream))
    rows = tuple(
        row for row in all_rows if (row["component1"], row["component2"]) == ("methane", "n_decane")
    )
    assert len(all_rows) == 103
    assert len(rows) == 65
    assert {float(row["temperature_K"]) for row in rows} == {
        410.93,
        477.59,
        510.93,
        563.25,
    }
    assert len({row["citation"] for row in all_rows}) == 4

    components = component_set(("methane", "n_decane"), dtype=DTYPE)
    metrics = []
    for model in (
        peng_robinson_1978(components),
        predictive_peng_robinson_1978(components),
    ):
        pressure_errors = []
        vapor_errors = []
        for row in rows:
            pressure = float(row["pressure_bar"]) * 1.0e5
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            point = binary_bubble_point(
                model,
                torch.tensor(float(row["temperature_K"]), dtype=DTYPE),
                torch.tensor([x1, 1.0 - x1], dtype=DTYPE),
                initial_pressure=torch.tensor(pressure, dtype=DTYPE),
                initial_vapor_composition=torch.tensor([y1, 1.0 - y1], dtype=DTYPE),
                minimum_pressure=1.0e3,
                maximum_pressure=1.0e8,
                max_iterations=30,
            )
            assert point.converged
            pressure_errors.append(100.0 * abs(float(point.pressure) / pressure - 1.0))
            vapor_errors.append(abs(float(point.vapor_composition[0]) - y1))
        metrics.append(
            (
                sum(pressure_errors) / len(pressure_errors),
                sum(vapor_errors) / len(vapor_errors),
            )
        )
    zero_kij, ppr78 = metrics
    assert ppr78[0] < 0.4 * zero_kij[0]
    assert ppr78[1] < zero_kij[1]
    assert ppr78[0] < 2.6
    assert ppr78[1] < 0.007
