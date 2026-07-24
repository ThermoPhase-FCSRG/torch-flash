from __future__ import annotations

import builtins
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from torch_flash import ModelParameterSet, activity_model, load_model_parameters, unifac_model
from torch_flash.activity import UNIFAC, unifac_groups_from_identifiers
from torch_flash.exceptions import ParameterDatabaseError

DTYPE = torch.float64


@pytest.mark.parametrize(
    ("assignments", "temperature", "composition", "expected"),
    [
        (
            ({1: 2, 2: 4}, {1: 1, 2: 1, 18: 1}),
            333.15,
            (0.5, 0.5),
            (1.4276025835624173, 1.3646545010104225),
        ),
        (
            ({1: 1, 2: 1, 14: 1}, {9: 6}),
            345.0,
            (0.2, 0.8),
            (2.90999524962436, 1.1038643452317465),
        ),
        (
            ({1: 2, 3: 1, 14: 1}, {16: 1}),
            353.52,
            (0.1, 0.9),
            (5.09713626128305, 1.05863779262131),
        ),
        (
            ({1: 1, 18: 1}, {1: 2, 2: 3}),
            307.0,
            (0.047, 0.953),
            (4.992034311484559, 1.00526021118788),
        ),
    ],
)
def test_original_unifac_literature_examples(
    assignments,
    temperature,
    composition,
    expected,
):
    """Reproduce one DDBST example and three thermo 0.6.0 reference states."""
    model = unifac_model(group_assignments=assignments, dtype=DTYPE)
    result = torch.exp(
        model.log_activity_coefficients(
            torch.tensor(temperature, dtype=DTYPE),
            torch.tensor(composition, dtype=DTYPE),
        )
    )
    torch.testing.assert_close(result, torch.tensor(expected, dtype=DTYPE), rtol=2.0e-13, atol=0.0)


def test_unifac_bundled_names_batch_autodiff_and_trainable_interactions():
    model = unifac_model(names=("ethanol", "benzene"), dtype=DTYPE, trainable=True)
    assert isinstance(model, UNIFAC)
    assert isinstance(model.interaction, torch.nn.Parameter)
    assert model.subgroup_keys == ("sg_001", "sg_002", "sg_009", "sg_014")
    assert model.main_group_ids == (1, 3, 5)
    torch.testing.assert_close(
        model.molecular_relative_volume,
        torch.tensor([2.5755, 3.1878], dtype=DTYPE),
    )
    torch.testing.assert_close(
        model.molecular_relative_surface_area,
        torch.tensor([2.588, 2.4], dtype=DTYPE),
    )

    temperatures = torch.tensor([330.0, 345.0], dtype=DTYPE)
    compositions = torch.tensor([[0.5, 0.5], [0.2, 0.8]], dtype=DTYPE)
    batched = model.log_activity_coefficients(temperatures, compositions)
    scalar = torch.stack(
        [
            model.log_activity_coefficients(temperatures[index], compositions[index])
            for index in range(2)
        ]
    )
    torch.testing.assert_close(batched, scalar)

    temperature = temperatures[1]
    composition = compositions[1].clone().requires_grad_(True)
    log_gamma = model.log_activity_coefficients(temperature, composition)

    def extensive(moles):
        total = moles.sum()
        return total * model.excess_gibbs_rt(temperature, moles / total)

    torch.testing.assert_close(torch.func.grad(extensive)(composition), log_gamma)
    jacobian = torch.func.jacrev(
        lambda values: model.log_activity_coefficients(temperature, values)
    )(composition)
    assert bool(torch.isfinite(jacobian).all())
    model.excess_gibbs_rt(temperature, composition).backward()
    assert model.interaction.grad is not None
    torch.testing.assert_close(
        torch.diagonal(model.interaction.grad),
        torch.zeros(len(model.main_group_ids), dtype=DTYPE),
    )


def test_unifac_pure_limit_group_aliases_and_activity_factory():
    by_name = activity_model("unifac", ("n_hexane", "2_butanone"), dtype=DTYPE)
    by_number = unifac_model(
        group_assignments=({"CH3": 2, "CH2": 4}, {"sg_001": 1, 2: 1, 18: 1}),
        dtype=DTYPE,
    )
    temperature = torch.tensor(333.15, dtype=DTYPE)
    composition = torch.tensor([0.5, 0.5], dtype=DTYPE)
    torch.testing.assert_close(
        by_name.log_activity_coefficients(temperature, composition),
        by_number.log_activity_coefficients(temperature, composition),
    )
    pure = unifac_model(group_assignments=({1: 1, 2: 1, 14: 1},), dtype=DTYPE)
    torch.testing.assert_close(
        pure.log_activity_coefficients(temperature, torch.ones(1, dtype=DTYPE)),
        torch.zeros(1, dtype=DTYPE),
        atol=4.0e-15,
        rtol=0.0,
    )
    factors = by_name.interaction_factors(temperature)
    torch.testing.assert_close(torch.diagonal(factors), torch.ones(2, dtype=DTYPE))
    by_name.interaction.diagonal().fill_(123.0)
    torch.testing.assert_close(
        torch.diagonal(by_name.interaction_factors(temperature)),
        torch.ones(2, dtype=DTYPE),
    )


def _simple_parameter_set(**parameter_updates) -> ModelParameterSet:
    parameters = {
        "coordination_number": 10.0,
        "subgroups": {
            "a": {
                "number": 1,
                "name": "A",
                "main_group": 1,
                "relative_volume": 1.0,
                "relative_surface_area": 1.0,
            },
            "b": {
                "number": 2,
                "name": "B",
                "main_group": 2,
                "relative_volume": 1.2,
                "relative_surface_area": 1.1,
            },
        },
        "interactions": [[1, 2, 100.0], [2, 1, -50.0]],
        "component_assignments": {"first": {1: 1}, "second": {2: 1}},
    }
    parameters.update(parameter_updates)
    return ModelParameterSet(
        "activity.test-unifac",
        "activity",
        "original-UNIFAC",
        "test",
        parameters,
    )


def test_unifac_custom_parameter_set_and_factory_errors():
    model = unifac_model(_simple_parameter_set(), names=("first", "second"), dtype=DTYPE)
    assert model.main_group_ids == (1, 2)
    with pytest.raises(ParameterDatabaseError, match="lacks required directed"):
        unifac_model(
            _simple_parameter_set(interactions=[[1, 2, 100.0]]),
            group_assignments=({1: 1}, {2: 1}),
        )
    with pytest.raises(ParameterDatabaseError, match="duplicate UNIFAC"):
        unifac_model(
            _simple_parameter_set(interactions=[[1, 2, 100.0], [1, 2, 101.0], [2, 1, -50.0]]),
            group_assignments=({1: 1}, {2: 1}),
        )
    with pytest.raises(ParameterDatabaseError, match="must be \\[main_group_i"):
        unifac_model(
            _simple_parameter_set(interactions=[[1, 2]]),
            group_assignments=({1: 1}, {2: 1}),
        )
    with pytest.raises(ParameterDatabaseError, match="nonempty subgroups"):
        unifac_model(_simple_parameter_set(subgroups={}), group_assignments=({1: 1},))
    with pytest.raises(ParameterDatabaseError, match="interactions sequence"):
        unifac_model(_simple_parameter_set(interactions={}), group_assignments=({1: 1},))
    with pytest.raises(KeyError, match="no UNIFAC group assignment"):
        unifac_model(_simple_parameter_set(), names=("unknown",))
    with pytest.raises(ValueError, match="names and group_assignments"):
        unifac_model(
            _simple_parameter_set(),
            names=("first",),
            group_assignments=({1: 1}, {2: 1}),
        )
    with pytest.raises(ValueError, match="component names or explicit"):
        unifac_model(_simple_parameter_set())
    with pytest.raises(ParameterDatabaseError, match="not original UNIFAC"):
        unifac_model(
            ModelParameterSet("activity.not-unifac", "activity", "NRTL", "1", {}),
            group_assignments=({1: 1},),
        )
    with pytest.raises(ParameterDatabaseError, match="not 'activity'"):
        unifac_model(load_model_parameters("cubic.pr-1978"), group_assignments=({1: 1},))
    with pytest.raises(ValueError, match="only valid for UNIFAC"):
        activity_model(
            "activity.hv-nrtl-jaubert-2020-methanol-benzene",
            ("methanol", "benzene"),
            group_assignments=({15: 1}, {9: 6}),
        )


def test_unifac_group_assignment_and_record_validation():
    with pytest.raises(KeyError, match="unknown or ambiguous"):
        unifac_model(group_assignments=({"CHO": 1},), dtype=DTYPE)
    with pytest.raises(ValueError, match="nonempty group assignment"):
        unifac_model(group_assignments=({},), dtype=DTYPE)
    with pytest.raises(ValueError, match="cannot be negative"):
        unifac_model(group_assignments=({1: -1},), dtype=DTYPE)
    with pytest.raises(ValueError, match="positive group count"):
        unifac_model(group_assignments=({1: 0},), dtype=DTYPE)
    with pytest.raises(KeyError, match="unknown or ambiguous"):
        unifac_model(group_assignments=({9999: 1},), dtype=DTYPE)
    with pytest.raises(ParameterDatabaseError, match="must be numeric"):
        unifac_model(group_assignments=({1: "one"},), dtype=DTYPE)  # type: ignore[dict-item]
    bad_record = _simple_parameter_set(
        subgroups={
            "a": {
                "number": 1,
                "name": "A",
                "main_group": "one",
                "relative_volume": 1.0,
                "relative_surface_area": 1.0,
            }
        },
        interactions=[],
    )
    with pytest.raises(ParameterDatabaseError, match="main_group values"):
        unifac_model(bad_record, group_assignments=({1: 1},))
    with pytest.raises(ParameterDatabaseError, match="keys and records"):
        unifac_model(
            _simple_parameter_set(subgroups={"a": 1}),
            group_assignments=({"a": 1},),
        )
    incomplete_alias_record = _simple_parameter_set(
        subgroups={
            "a": {
                "main_group": 1,
                "relative_volume": 1.0,
                "relative_surface_area": 1.0,
            }
        },
        interactions=[],
    )
    assert unifac_model(
        incomplete_alias_record,
        group_assignments=({"a": 1},),
    ).subgroup_keys == ("a",)
    with pytest.raises(ParameterDatabaseError, match="no bundled component_assignments"):
        unifac_model(
            _simple_parameter_set(component_assignments=None),
            names=("first",),
        )
    with pytest.raises(ValueError, match="at least one component"):
        unifac_model(_simple_parameter_set(), group_assignments=())


def _base_unifac_tensors():
    return (
        torch.tensor([[1.0, 1.0], [2.0, 0.0]], dtype=DTYPE),
        torch.tensor([1.0, 1.2], dtype=DTYPE),
        torch.tensor([0.8, 1.1], dtype=DTYPE),
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([[0.0, 100.0], [-50.0, 0.0]], dtype=DTYPE),
    )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda values: values.__setitem__(0, torch.ones(2)), "component-by-group"),
        (lambda values: values.__setitem__(1, torch.ones(3)), "one R and Q"),
        (lambda values: values.__setitem__(3, torch.zeros(3, dtype=torch.long)), "main-group"),
        (lambda values: values.__setitem__(3, torch.zeros(2)), "torch.long"),
        (lambda values: values.__setitem__(4, torch.ones(2)), "square directed"),
        (
            lambda values: values.__setitem__(
                4,
                torch.empty((0, 0), dtype=DTYPE),
            ),
            "at least one selected main group",
        ),
        (lambda values: values.__setitem__(0, torch.tensor([[math.nan, 1.0]])), "finite"),
        (lambda values: values.__setitem__(0, -torch.ones((2, 2))), "nonnegative"),
        (lambda values: values.__setitem__(1, torch.tensor([0.0, 1.0])), "R values"),
        (lambda values: values.__setitem__(3, torch.tensor([0, 2])), "outside"),
        (
            lambda values: values.__setitem__(
                4, torch.tensor([[1.0, 100.0], [-50.0, 0.0]], dtype=DTYPE)
            ),
            "same-main-group",
        ),
    ],
)
def test_unifac_low_level_validation(mutator, match):
    values = list(_base_unifac_tensors())
    mutator(values)
    with pytest.raises(ValueError, match=match):
        UNIFAC(*values)


def test_unifac_low_level_state_validation():
    values = _base_unifac_tensors()
    with pytest.raises(ValueError, match="coordination"):
        UNIFAC(*values, coordination_number=0.0)
    zero_q = list(values)
    zero_q[2] = torch.zeros(2, dtype=DTYPE)
    with pytest.raises(ValueError, match="positive molecular"):
        UNIFAC(*zero_q)
    model = UNIFAC(*values)
    with pytest.raises(ValueError, match="temperature must be positive"):
        model.interaction_factors(torch.tensor(0.0, dtype=DTYPE))
    with pytest.raises(ValueError, match="temperature must be positive"):
        model.log_activity_coefficients(
            torch.tensor(-1.0, dtype=DTYPE),
            torch.tensor([0.5, 0.5], dtype=DTYPE),
        )
    with pytest.raises(ValueError, match="composition size"):
        model.log_activity_coefficients(
            torch.tensor(300.0, dtype=DTYPE),
            torch.ones(3, dtype=DTYPE),
        )
    with pytest.raises(ValueError, match="composition size"):
        model.combinatorial_log_activity_coefficients(torch.ones(3, dtype=DTYPE))


def test_ugropy_optional_adapter(monkeypatch):
    real_import = builtins.__import__

    def missing_import(name, *args, **kwargs):
        if name == "ugropy":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_import)
    with pytest.raises(ImportError, match="torch-flash\\[groups\\]"):
        unifac_groups_from_identifiers(("CCO",))
    monkeypatch.setattr(builtins, "__import__", real_import)

    module = ModuleType("ugropy")
    module.unifac = SimpleNamespace(
        get_groups=lambda identifier, identifier_type: SimpleNamespace(
            subgroups_num={1: len(identifier), 2: 1}
        )
    )
    monkeypatch.setitem(sys.modules, "ugropy", module)
    assert unifac_groups_from_identifiers(("CCO",)) == ({1: 3.0, 2: 1.0},)
    with pytest.raises(ValueError, match="identifier_type"):
        unifac_groups_from_identifiers(("CCO",), identifier_type="inchi")

    module.unifac = SimpleNamespace(
        get_groups=lambda identifier, identifier_type: [SimpleNamespace(subgroups_num={1: 1})]
    )
    assert unifac_groups_from_identifiers(("CC",)) == ({1: 1.0},)
    module.unifac = SimpleNamespace(get_groups=lambda identifier, identifier_type: [1, 2])
    with pytest.raises(ValueError, match="returned 2 fragmentations"):
        unifac_groups_from_identifiers(("CC",))
    module.unifac = SimpleNamespace(
        get_groups=lambda identifier, identifier_type: SimpleNamespace(subgroups_num={})
    )
    with pytest.raises(ValueError, match="could not assign"):
        unifac_groups_from_identifiers(("invalid",))
