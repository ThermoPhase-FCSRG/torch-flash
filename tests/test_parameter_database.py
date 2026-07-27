from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

import torch_flash
import torch_flash.database as database_module
import torch_flash.eos.named as named_module
from torch_flash import (
    Component,
    ComponentDatabase,
    ModelParameterSet,
    activity_model,
    available_parameter_sets,
    binary_interaction,
    canonical_component_name,
    clear_parameter_caches,
    component,
    component_set,
    cpa_eos,
    cubic_constants,
    cubic_eos,
    cubic_interaction_parameters,
    ideal_gas_polynomial,
    load_component_database,
    load_model_parameters,
    multiparameter_eos,
    peng_robinson_1978,
)
from torch_flash.activity import NRTL, HuronVidalNRTL, Wilson
from torch_flash.components import clear_component_caches
from torch_flash.database import _parse_yaml
from torch_flash.eos import (
    CPAEOS,
    CPAComponent,
    CubicConstants,
    MultiparameterEOS,
    MultiparameterMetadata,
)
from torch_flash.eos.named import eoscg2021, gerg2008
from torch_flash.exceptions import ParameterDatabaseError
from torch_flash.parameters import CubicInteractionParameters

DTYPE = torch.float64


def _model_document(**updates):
    document = {
        "format": "torch-flash-model-parameters",
        "schema_version": 1,
        "id": "test.parameters",
        "model_kind": "activity",
        "model": "NRTL",
        "version": "1",
        "description": "test",
        "units": {"interaction": "J mol^-1"},
        "references": [{"citation": "test reference"}],
        "parameters": {
            "components": ["water", "methanol"],
            "interaction": [[0.0, 100.0], [200.0, 0.0]],
            "nonrandomness": [[0.0, 0.3], [0.3, 0.0]],
        },
    }
    document.update(updates)
    return document


def _component_document(**updates):
    document = {
        "format": "torch-flash-component-database",
        "schema_version": 1,
        "id": "components.test",
        "revision": "1",
        "description": "test",
        "unit_system": "SI",
        "units": {
            "critical_temperature": "K",
            "critical_pressure": "Pa",
            "acentric_factor": "dimensionless",
            "molar_mass": "kg mol^-1",
            "critical_volume": "m^3 mol^-1",
            "critical_density": "mol m^-3",
        },
        "references": [{"citation": "test reference"}],
        "components": {
            "test_fluid": {
                "aliases": ["tf"],
                "critical_temperature": 400.0,
                "critical_pressure": 5.0e6,
                "acentric_factor": 0.2,
                "molar_mass": 0.05,
                "critical_volume": 2.0e-4,
                "critical_density": 5000.0,
            }
        },
    }
    document.update(updates)
    return document


def _write_yaml(path: Path, document) -> Path:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _component_units():
    return _component_document()["units"]


def test_bundled_registry_caching_metadata_and_immutability():
    identifiers = available_parameter_sets()
    assert identifiers == tuple(sorted(identifiers))
    assert set(available_parameter_sets(model_kind="cubic")) == {
        "cubic.pr-1976",
        "cubic.pr-1978",
        "cubic.srk-1972",
    }
    first = load_model_parameters("gerg2008")
    second = load_model_parameters("multiparameter.gerg-2008")
    assert first is second
    assert first.model == "GERG-2008"
    assert first.units["critical_temperature"] == "K"
    assert first.references[0]["doi"] == "10.1021/je300655b"
    unifac = load_model_parameters("unifac")
    assert unifac.identifier == "activity.unifac-original-public-2026"
    assert len(unifac.parameters["subgroups"]) == 113
    assert len(unifac.parameters["interactions"]) == 1270
    assert unifac.parameters["subgroups"]["sg_018"]["relative_surface_area"] == pytest.approx(1.488)
    assert unifac.parameters["provenance_notes"][1].startswith("Missing directed pairs")
    with pytest.raises(TypeError):
        first.parameters["gas_constant"] = 1.0
    copied = first.as_dict()
    copied["gas_constant"] = 1.0
    assert first.parameters["gas_constant"] == pytest.approx(8.314472)
    clear_parameter_caches()
    assert load_model_parameters("gerg2008") is not first


@pytest.mark.parametrize(
    ("identifier", "names", "expected_tau_01_at_350_k"),
    [
        (
            "jaubert-hv-n-butane-water",
            ("n_butane", "water"),
            4032.109951499353 / 350.0 - 3.8583578360824635,
        ),
        (
            "jaubert-hv-ethanol-n-heptane",
            ("ethanol", "n_heptane"),
            234.4139461863396 / 350.0 - 0.3714375290648538,
        ),
        (
            "jaubert-hv-methanol-benzene",
            ("methanol", "benzene"),
            512.3267983829446 / 350.0 - 1.4666938931099835,
        ),
    ],
)
def test_bundled_jaubert_hv_parameter_sets(
    identifier,
    names,
    expected_tau_01_at_350_k,
):
    model = activity_model(identifier, names)
    assert isinstance(model, HuronVidalNRTL)
    tau = model.tau_matrix(torch.tensor(350.0, dtype=DTYPE))
    assert float(tau[0, 1]) == pytest.approx(expected_tau_01_at_350_k)
    assert float(tau[0, 0]) == 0.0
    loaded = load_model_parameters(identifier)
    assert loaded.references[0]["doi"] == "10.1021/acs.iecr.0c01734"
    assert loaded.parameters["fit_metadata"]["objective"] == (
        "measured-phase log-fugacity residual"
    )
    if identifier == "jaubert-hv-ethanol-n-heptane":
        assert loaded.parameters["fit_metadata"]["excluded_source_rows"] == 23


def test_custom_model_yaml_and_explicit_parameter_object(tmp_path):
    path = _write_yaml(tmp_path / "nrtl.yaml", _model_document())
    first = load_model_parameters(path)
    second = load_model_parameters(str(path))
    assert first is second
    assert first.source == str(path.resolve())
    path.unlink()
    assert load_model_parameters(path) is first
    model = activity_model(path, ("methanol", "water"), trainable=True)
    assert isinstance(model, NRTL)
    assert model.interaction.requires_grad
    torch.testing.assert_close(
        model.interaction,
        torch.tensor([[0.0, 200.0], [100.0, 0.0]], dtype=DTYPE),
    )

    explicit = ModelParameterSet(
        "api.wilson",
        "activity",
        "Wilson",
        "fit-1",
        {
            "components": ["water", "methanol"],
            "interaction": [[0.0, 50.0], [75.0, 0.0]],
            "molar_volumes": [1.8e-5, 4.1e-5],
        },
    )
    wilson = activity_model(explicit)
    assert isinstance(wilson, Wilson)
    assert wilson.molar_volumes.shape == (2,)


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"format": "wrong"}, "format"),
        ({"schema_version": 2}, "schema_version"),
        ({"id": ""}, "non-empty"),
        ({"parameters": []}, "parameters"),
        ({"units": []}, "units"),
        ({"references": {}}, "references"),
        ({"description": []}, "description"),
    ],
)
def test_model_document_validation(tmp_path, update, match):
    path = _write_yaml(tmp_path / "invalid.yaml", _model_document(**update))
    with pytest.raises(ParameterDatabaseError, match=match):
        load_model_parameters(path)


def test_model_database_lookup_and_yaml_errors(tmp_path):
    with pytest.raises(KeyError, match="unknown model parameter"):
        load_model_parameters("not-a-model")
    with pytest.raises(FileNotFoundError):
        load_model_parameters(tmp_path / "missing.yaml")
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("x: [", encoding="utf-8")
    with pytest.raises(ParameterDatabaseError, match="invalid YAML"):
        load_model_parameters(malformed)
    with pytest.raises(ParameterDatabaseError, match="mapping"):
        _parse_yaml("- one\n- two\n", "inline")
    with pytest.raises(ParameterDatabaseError, match="parameters"):
        ModelParameterSet("x", "cubic", "x", "1", [])
    with pytest.raises(ParameterDatabaseError, match="reference"):
        ModelParameterSet("x", "cubic", "x", "1", {}, references=("bad",))


def test_default_and_custom_component_databases(tmp_path):
    database = load_component_database()
    assert database is load_component_database("default")
    assert database.identifier == "components.default"
    assert database.units["critical_pressure"] == "Pa"
    assert canonical_component_name(" CO2 ") == "carbon_dioxide"
    assert canonical_component_name("unknown-fluid", strict=False) == "unknown_fluid"
    with pytest.raises(KeyError, match="unknown component"):
        canonical_component_name("unknown-fluid")

    ammonia = component("NH3")
    assert ammonia.name == "ammonia"
    assert ammonia.acentric_factor is None
    with pytest.raises(ParameterDatabaseError, match="acentric"):
        component_set(("ammonia",))

    path = _write_yaml(tmp_path / "components.yaml", _component_document())
    custom = load_component_database(path)
    assert custom is load_component_database(str(path))
    assert custom.lookup("TF").name == "test_fluid"
    components = component_set(("tf",), database=custom)
    assert components.names == ("test_fluid",)
    torch.testing.assert_close(
        components.critical_density,
        torch.tensor([5000.0], dtype=DTYPE),
    )
    moved = components.to(dtype=torch.float32)
    assert moved.critical_volume is not None
    assert moved.critical_volume.dtype == torch.float32
    assert moved.critical_density is not None
    assert moved.critical_density.dtype == torch.float32
    clear_component_caches()
    assert load_component_database(path) is not custom


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda doc: doc.update(format="wrong"), "component database"),
        (lambda doc: doc.update(schema_version=2), "schema_version"),
        (lambda doc: doc.update(unit_system="field"), "unit_system"),
        (lambda doc: doc.update(units=[]), "units"),
        (
            lambda doc: doc["units"].update(critical_pressure="bar"),
            "critical_pressure",
        ),
        (lambda doc: doc.update(components=[]), "components"),
        (lambda doc: doc.update(references=["bad"]), "references"),
        (lambda doc: doc.update(id=3), "id and revision"),
    ],
)
def test_component_database_document_validation(tmp_path, mutator, match):
    document = _component_document()
    mutator(document)
    path = _write_yaml(tmp_path / "invalid-components.yaml", document)
    with pytest.raises(ParameterDatabaseError, match=match):
        load_component_database(path)


def test_component_database_record_validation(tmp_path):
    cases = []
    noncanonical = _component_document()
    noncanonical["components"] = {"Test Fluid": noncanonical["components"]["test_fluid"]}
    cases.append((noncanonical, "not canonical"))
    duplicate = _component_document()
    duplicate["components"]["second"] = {
        **duplicate["components"]["test_fluid"],
        "aliases": ["tf"],
    }
    cases.append((duplicate, "duplicate"))
    bad_alias = _component_document()
    bad_alias["components"]["test_fluid"]["aliases"] = "tf"
    cases.append((bad_alias, "aliases"))
    bad_numeric = _component_document()
    bad_numeric["components"]["test_fluid"]["critical_pressure"] = "5 MPa"
    cases.append((bad_numeric, "numeric"))
    nonpositive = _component_document()
    nonpositive["components"]["test_fluid"]["molar_mass"] = 0.0
    cases.append((nonpositive, "positive"))
    for index, (document, match) in enumerate(cases):
        path = _write_yaml(tmp_path / f"record-{index}.yaml", document)
        with pytest.raises(ParameterDatabaseError, match=match):
            load_component_database(path)


def test_cubic_yaml_and_explicit_parameter_apis():
    components = component_set(("methane", "n_butane"))
    from_yaml = cubic_eos(components, "pr78")
    named = peng_robinson_1978(components)
    temperature = torch.tensor(300.0, dtype=DTYPE)
    pressure = torch.tensor(4.0e6, dtype=DTYPE)
    composition = torch.tensor([0.7, 0.3], dtype=DTYPE)
    torch.testing.assert_close(
        from_yaml.select_z(temperature, pressure, composition),
        named.select_z(temperature, pressure, composition),
    )
    constants = cubic_constants("cubic.pr-1978")
    assert cubic_eos(components, constants).constants is constants
    assert constants.parameter_set == "cubic.pr-1978"


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda data: data.pop("alpha"), "alpha"),
        (lambda data: data["alpha"].update(kind="bad"), "alpha kind"),
        (lambda data: data["alpha"]["low"].update(coefficients=[1.0]), "length 3"),
        (lambda data: data["alpha"]["high"].update(coefficients=[1.0]), "length 4"),
        (lambda data: data["alpha"].pop("switch_acentric_factor"), "switch"),
        (lambda data: data.update(omega_a="bad"), "must be numeric"),
    ],
)
def test_custom_cubic_parameter_validation(mutator, match):
    data = load_model_parameters("pr78").as_dict()
    mutator(data)
    parameter_set = ModelParameterSet("custom.pr", "cubic", "custom", "1", data)
    with pytest.raises(ParameterDatabaseError, match=match):
        cubic_constants(parameter_set)
    with pytest.raises(ParameterDatabaseError, match="not 'cubic'"):
        cubic_constants(_model_parameter_kind("activity"))


def _model_parameter_kind(kind: str) -> ModelParameterSet:
    return ModelParameterSet("test.kind", kind, "test", "1", {})


def test_cpa_database_factory_and_validation():
    loaded = cpa_eos(("H2O", "MEOH"))
    named = cpa_eos(("water", "methanol"), load_model_parameters("folas-2005"))
    assert isinstance(loaded, CPAEOS)
    torch.testing.assert_close(loaded.a0, named.a0)
    with pytest.raises(ParameterDatabaseError, match="not 'cpa'"):
        cpa_eos(("water",), _model_parameter_kind("activity"))

    malformed = load_model_parameters("folas-2005").as_dict()
    malformed["components"]["water"]["a0"] = "bad"
    with pytest.raises(ParameterDatabaseError, match="non-numeric"):
        cpa_eos(
            ("water",),
            ModelParameterSet("custom.cpa", "cpa", "CPA", "1", malformed),
        )
    with pytest.raises(KeyError, match="unavailable"):
        cpa_eos(("hydrogen",))


def test_activity_database_factory_hv_and_error_paths():
    model = activity_model(
        "pedersen-hv-propane-water",
        ("propane", "water"),
        trainable=True,
    )
    assert isinstance(model, HuronVidalNRTL)
    assert model.energy_over_r.requires_grad
    torch.testing.assert_close(
        model.energy_over_r,
        torch.tensor([[0.0, 6065.0], [-2026.0, 0.0]], dtype=DTYPE),
    )
    explicit_covolumes = torch.tensor([8.0e-5, 2.0e-5], dtype=DTYPE)
    explicit = activity_model(
        "pedersen-hv-propane-water",
        ("propane", "water"),
        covolumes=explicit_covolumes,
    )
    torch.testing.assert_close(explicit.covolumes, explicit_covolumes)

    with pytest.raises(ParameterDatabaseError, match="not 'activity'"):
        activity_model(_model_parameter_kind("cubic"))
    with pytest.raises(ValueError, match="non-empty and unique"):
        activity_model("pedersen-hv-propane-water", ())
    with pytest.raises(KeyError, match="no activity parameters"):
        activity_model("pedersen-hv-propane-water", ("water", "methane"))
    with pytest.raises(ValueError, match="one activity-model covolume"):
        activity_model(
            "pedersen-hv-propane-water",
            covolumes=torch.ones(3, dtype=DTYPE),
        )


def test_activity_custom_nrtl_wilson_and_invalid_models():
    nrtl = activity_model(ModelParameterSet.from_document(_model_document()))
    assert isinstance(nrtl, NRTL)
    wilson_document = _model_document(
        id="test.wilson",
        model="Wilson",
        parameters={
            "components": ["water", "methanol"],
            "interaction": [[0.0, 1.0], [2.0, 0.0]],
            "molar_volumes": [1.0e-5, 2.0e-5],
        },
    )
    assert isinstance(
        activity_model(ModelParameterSet.from_document(wilson_document)),
        Wilson,
    )
    unsupported = _model_document(model="UNIQUAC")
    with pytest.raises(ParameterDatabaseError, match="unsupported activity"):
        activity_model(ModelParameterSet.from_document(unsupported))
    missing_components = _model_document(parameters={"components": "water"})
    with pytest.raises(ParameterDatabaseError, match="components list"):
        activity_model(ModelParameterSet.from_document(missing_components))
    bad_matrix = _model_document()
    bad_matrix["parameters"]["interaction"] = ["bad"]
    with pytest.raises(ParameterDatabaseError, match="numeric matrix"):
        activity_model(ModelParameterSet.from_document(bad_matrix))


def test_binary_interaction_generic_database_factory():
    components = component_set(("nitrogen", "co2", "methane", "n_decane"))
    matrix = binary_interaction(components, "whitson-bip", "PR")
    assert matrix[0, 2] == pytest.approx(0.025)
    assert matrix[1, 3] == pytest.approx(0.115)
    torch.testing.assert_close(matrix, matrix.mT)
    with pytest.raises(ParameterDatabaseError, match="not 'binary_interaction'"):
        binary_interaction(components, _model_parameter_kind("cubic"))


def _cubic_interaction_set(
    parameters,
    *,
    model_kind="binary_interaction",
    model="cubic-vdw-one-fluid",
    units=None,
):
    return ModelParameterSet(
        "binary.cubic-custom",
        model_kind,
        model,
        "1",
        parameters,
        {"kij": "dimensionless", "lij": "dimensionless"} if units is None else units,
    )


def test_cubic_attraction_and_covolume_interaction_database_factory():
    components = component_set(("n_decane", "methane"))
    interactions = cubic_interaction_parameters(
        components,
        "methane-n-decane-covolume",
    )
    expected_kij = torch.tensor([[0.0, 0.0409], [0.0409, 0.0]], dtype=DTYPE)
    expected_lij = torch.tensor(
        [[0.0, 0.047567430964013162], [0.047567430964013162, 0.0]],
        dtype=DTYPE,
    )
    torch.testing.assert_close(interactions.kij, expected_kij)
    torch.testing.assert_close(interactions.lij, expected_lij)
    assert interactions.parameter_set == "binary-interaction.segovia-2017-methane-n-decane"
    model = peng_robinson_1978(
        components,
        kij=interactions.kij,
        lij=interactions.lij,
    )
    assert (
        model.select_z(
            torch.tensor(323.15, dtype=DTYPE),
            torch.tensor(80.0e6, dtype=DTYPE),
            torch.tensor([0.5, 0.5], dtype=DTYPE),
            "liquid",
        )
        > 0.0
    )


def test_cubic_interaction_database_defaults_subsets_and_validation():
    components = component_set(("methane", "n_decane"))
    base = {
        "component_order": ["methane", "n_decane", "ethane"],
        "defaults": {"kij": 0.01, "lij": -0.02},
        "pairs": {
            "methane|n_decane": {"kij": 0.04, "lij": 0.05},
            "methane|ethane": {"kij": 0.03},
        },
    }
    result = cubic_interaction_parameters(components, _cubic_interaction_set(base))
    assert result.kij[0, 1] == pytest.approx(0.04)
    assert result.lij[0, 1] == pytest.approx(0.05)

    cases = [
        (
            _cubic_interaction_set(base, model_kind="cubic"),
            "not 'binary_interaction'",
        ),
        (
            _cubic_interaction_set(base, model="other"),
            "cubic-vdw-one-fluid",
        ),
        (
            _cubic_interaction_set(base, units={"kij": "K", "lij": "dimensionless"}),
            "dimensionless",
        ),
        (
            _cubic_interaction_set({**base, "component_order": "methane"}),
            "string list",
        ),
        (
            _cubic_interaction_set({**base, "component_order": ["methane", 1]}),
            "string list",
        ),
        (
            _cubic_interaction_set({**base, "component_order": ["methane", "methane"]}),
            "duplicates",
        ),
        (
            _cubic_interaction_set({**base, "defaults": "bad"}),
            "defaults",
        ),
        (
            _cubic_interaction_set({**base, "defaults": {"kij": float("nan")}}),
            "default kij",
        ),
        (
            _cubic_interaction_set({**base, "pairs": "bad"}),
            "pairs",
        ),
        (
            _cubic_interaction_set({**base, "pairs": {"bad": {}}}),
            "first\\|second",
        ),
        (
            _cubic_interaction_set({**base, "pairs": {"methane|methane": {}}}),
            "self-pairs",
        ),
        (
            _cubic_interaction_set({**base, "pairs": {"methane|water": {}}}),
            "outside component_order",
        ),
        (
            _cubic_interaction_set(
                {
                    **base,
                    "pairs": {
                        "methane|n_decane": {},
                        "n_decane|methane": {},
                    },
                }
            ),
            "duplicate pair",
        ),
        (
            _cubic_interaction_set({**base, "pairs": {"methane|n_decane": "bad"}}),
            "must be a mapping",
        ),
        (
            _cubic_interaction_set({**base, "pairs": {"methane|n_decane": {"lij": float("inf")}}}),
            "lij must be finite",
        ),
    ]
    for source, message in cases:
        with pytest.raises(ParameterDatabaseError, match=message):
            cubic_interaction_parameters(components, source)

    unsupported = component_set(("methane", "water"))
    with pytest.raises(KeyError, match="water"):
        cubic_interaction_parameters(unsupported, _cubic_interaction_set(base))


def test_cubic_interaction_typed_container_validation():
    zeros = torch.zeros((2, 2), dtype=DTYPE)
    asymmetric = torch.tensor([[0.0, 0.1], [0.0, 0.0]], dtype=DTYPE)
    with pytest.raises(ValueError, match="square"):
        CubicInteractionParameters(torch.zeros(2), torch.zeros(2), "test")
    with pytest.raises(ValueError, match="same square shape"):
        CubicInteractionParameters(zeros, torch.zeros((3, 3)), "test")
    with pytest.raises(ValueError, match="finite"):
        CubicInteractionParameters(
            torch.tensor([[0.0, torch.inf], [torch.inf, 0.0]]),
            zeros,
            "test",
        )
    with pytest.raises(ValueError, match="finite"):
        CubicInteractionParameters(
            zeros,
            torch.tensor([[0.0, torch.inf], [torch.inf, 0.0]]),
            "test",
        )
    with pytest.raises(ValueError, match="kij must be symmetric"):
        CubicInteractionParameters(asymmetric, zeros, "test")
    with pytest.raises(ValueError, match="lij must be symmetric"):
        CubicInteractionParameters(zeros, asymmetric, "test")
    with pytest.raises(ValueError, match="zero diagonals"):
        CubicInteractionParameters(torch.eye(2), zeros, "test")
    with pytest.raises(ValueError, match="zero diagonals"):
        CubicInteractionParameters(zeros, torch.eye(2), "test")


def test_standard_state_database_and_custom_reference_values():
    assert load_model_parameters("poling").identifier == "standard-state.poling-2001"
    transport = load_model_parameters("pedersen-transport")
    assert transport.identifier == "transport.pedersen-2024"
    assert transport.model_kind == "transport"
    model = ideal_gas_polynomial(("co2",), "poling")
    torch.testing.assert_close(
        model.heat_capacity(torch.tensor(300.0, dtype=DTYPE)),
        torch.tensor([37.100429258368185], dtype=DTYPE),
        rtol=2.0e-8,
        atol=1.0e-8,
    )
    custom = ModelParameterSet(
        "standard.custom",
        "standard_state",
        "ideal-gas-cp-polynomial",
        "1",
        {
            "default_reference_temperature": 300.0,
            "components": {
                "water": {
                    "coefficients": [3.5, 1.0e-3],
                    "reference_enthalpy": 100.0,
                    "reference_entropy": 2.0,
                }
            },
        },
    )
    fitted = ideal_gas_polynomial(("water",), custom, trainable=True)
    assert fitted.reference_temperature == 300.0
    assert fitted.heat_capacity_coefficients.requires_grad
    torch.testing.assert_close(
        fitted.reference_enthalpy,
        torch.tensor([100.0], dtype=DTYPE),
    )
    with pytest.raises(ParameterDatabaseError, match="not 'standard_state'"):
        ideal_gas_polynomial(("water",), _model_parameter_kind("cubic"))


def test_generic_multiparameter_database_factory_and_model_guards():
    direct = multiparameter_eos("gerg2008", ("h2", "ch4"))
    named = gerg2008(("hydrogen", "methane"))
    torch.testing.assert_close(direct.pure_n, named.pure_n)
    eoscg = multiparameter_eos(load_model_parameters("eoscg2021"), ("co2", "h2"))
    torch.testing.assert_close(
        eoscg.pure_n,
        eoscg2021(("carbon_dioxide", "hydrogen")).pure_n,
    )
    with pytest.raises(ParameterDatabaseError, match="not 'multiparameter'"):
        multiparameter_eos(_model_parameter_kind("cubic"))
    with pytest.raises(ParameterDatabaseError, match="unsupported multiparameter"):
        multiparameter_eos(ModelParameterSet("custom", "multiparameter", "unknown", "1", {}))

    gerg_data = load_model_parameters("gerg2008").as_dict()
    gerg2004 = ModelParameterSet(
        "multiparameter.custom-gerg-2004",
        "multiparameter",
        "GERG-2004",
        "2004",
        gerg_data,
    )
    assert multiparameter_eos(gerg2004, ("methane",)).metadata.version == "2004"
    with pytest.raises(ParameterDatabaseError, match="requires a GERG-2008"):
        gerg2008(("methane",), parameter_set=gerg2004)
    with pytest.raises(ParameterDatabaseError, match="requires an EOS-CG-2021"):
        eoscg2021(("carbon_dioxide",), parameter_set=gerg2004)


def test_legacy_multifluid_api_and_parameter_ids_redirect_to_multiparameter():
    assert torch_flash.MultiFluidEOS is MultiparameterEOS
    assert torch_flash.MultifluidMetadata is MultiparameterMetadata

    with pytest.warns(DeprecationWarning, match="multiparameter_eos"):
        legacy_factory_model = torch_flash.multifluid_eos(
            "multiparameter.gerg-2008",
            ("methane",),
        )
    assert isinstance(legacy_factory_model, MultiparameterEOS)
    with pytest.warns(DeprecationWarning, match="'multifluid.' parameter prefix"):
        assert (
            load_model_parameters("multifluid.gerg-2008").identifier == "multiparameter.gerg-2008"
        )
    with pytest.warns(DeprecationWarning, match="model_kind='multifluid'"):
        assert available_parameter_sets(model_kind="multifluid") == (
            "multiparameter.eos-cg-2021",
            "multiparameter.gerg-2008",
            "multiparameter.gerg-2008-hydrogen-2021",
        )

    canonical = load_model_parameters("multiparameter.gerg-2008")
    legacy_kind = ModelParameterSet(
        "multifluid.custom-gerg-2008",
        "multifluid",
        canonical.model,
        canonical.version,
        canonical.parameters,
        canonical.units,
        canonical.references,
    )
    with pytest.warns(DeprecationWarning, match="model_kind='multifluid'"):
        legacy_kind_model = multiparameter_eos(legacy_kind, ("methane",))
    assert isinstance(legacy_kind_model, MultiparameterEOS)


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"identifier": ""}, "non-empty"),
        ({"identifier": "Bad ID"}, "lowercase"),
        ({"model_kind": "unknown"}, "model_kind"),
        ({"units": {"pressure": 1}}, "unit names"),
        ({"references": "citation"}, "references"),
        ({"references": ({"doi": 1},)}, "reference names"),
        ({"description": 1}, "description"),
    ],
)
def test_explicit_model_parameter_set_validation(arguments, match):
    values = {
        "identifier": "test.valid",
        "model_kind": "cubic",
        "model": "test",
        "version": "1",
        "parameters": {},
    }
    values.update(arguments)
    with pytest.raises(ParameterDatabaseError, match=match):
        ModelParameterSet(**values)


def test_custom_parameter_index_validation(tmp_path, monkeypatch):
    index_path = tmp_path / "index.yaml"
    monkeypatch.setattr(database_module, "_model_root", lambda: tmp_path)

    invalid_documents = [
        {
            "format": "torch-flash-parameter-index",
            "schema_version": 1,
            "parameter_sets": {},
        },
        {
            "format": "torch-flash-parameter-index",
            "schema_version": 1,
            "parameter_sets": {"x": []},
        },
        {
            "format": "torch-flash-parameter-index",
            "schema_version": 1,
            "parameter_sets": {"x": {"path": "x.json", "aliases": []}},
        },
        {
            "format": "torch-flash-parameter-index",
            "schema_version": 1,
            "parameter_sets": {"x": {"path": "x.yaml", "aliases": [1]}},
        },
        {
            "format": "torch-flash-parameter-index",
            "schema_version": 1,
            "parameter_sets": {
                "x": {"path": "x.yaml", "aliases": ["same"]},
                "y": {"path": "y.yaml", "aliases": ["same"]},
            },
        },
    ]
    matches = ["non-empty", "invalid", "YAML path", "aliases", "duplicate"]
    for document, match in zip(invalid_documents, matches, strict=True):
        _write_yaml(index_path, document)
        database_module._index.cache_clear()
        with pytest.raises(ParameterDatabaseError, match=match):
            database_module._index()

    _write_yaml(
        index_path,
        {
            "format": "torch-flash-parameter-index",
            "schema_version": 1,
            "parameter_sets": {"expected": {"path": "actual.yaml", "aliases": []}},
        },
    )
    _write_yaml(tmp_path / "actual.yaml", _model_document(id="actual"))
    database_module._index.cache_clear()
    database_module._load_builtin.cache_clear()
    with pytest.raises(ParameterDatabaseError, match="declares id"):
        database_module._load_builtin("expected")
    database_module.clear_parameter_caches()


def test_explicit_component_database_validation_and_api():
    record = Component(
        "test_fluid",
        400.0,
        5.0e6,
        0.2,
        0.05,
        2.0e-4,
        ("tf",),
        5000.0,
    )
    database = ComponentDatabase(
        "components.api",
        "1",
        (record,),
        _component_units(),
    )
    assert load_component_database(database) is database
    assert component_set(("tf",), database=database).names == ("test_fluid",)

    cases = [
        ({"identifier": ""}, "id and revision"),
        ({"components": ()}, "at least one"),
        ({"units": {}}, "unit"),
        (
            {"components": (Component("Test Fluid", 400.0, 5.0e6, 0.2, 0.05),)},
            "canonical",
        ),
        (
            {
                "components": (
                    record,
                    Component("second", 410.0, 5.1e6, 0.1, 0.06, aliases=("tf",)),
                )
            },
            "duplicate",
        ),
        (
            {"components": (Component("bad", 0.0, 5.0e6, 0.1, 0.05),)},
            "finite positive",
        ),
        (
            {"components": (Component("bad", 400.0, 5.0e6, 0.1, 0.05, -1.0),)},
            "volume/density",
        ),
        (
            {"components": (Component("bad", 400.0, 5.0e6, math.nan, 0.05),)},
            "acentric",
        ),
    ]
    for update, match in cases:
        values = {
            "identifier": "components.test",
            "revision": "1",
            "components": (record,),
            "units": _component_units(),
        }
        values.update(update)
        with pytest.raises(ParameterDatabaseError, match=match):
            ComponentDatabase(**values)


def test_component_yaml_low_level_errors(tmp_path):
    malformed = tmp_path / "malformed-components.yaml"
    malformed.write_text("components: [", encoding="utf-8")
    with pytest.raises(ParameterDatabaseError, match="invalid YAML"):
        load_component_database(malformed)
    sequence = tmp_path / "component-sequence.yaml"
    sequence.write_text("- component", encoding="utf-8")
    with pytest.raises(ParameterDatabaseError, match="mapping"):
        load_component_database(sequence)
    invalid_record = _component_document()
    invalid_record["components"]["test_fluid"] = []
    with pytest.raises(ParameterDatabaseError, match="invalid component record"):
        load_component_database(_write_yaml(tmp_path / "invalid-record.yaml", invalid_record))
    with pytest.raises(FileNotFoundError):
        load_component_database(tmp_path / "missing-components.yaml")


def test_activity_parameter_payload_error_branches():
    nrtl_nonmatrix = _model_document()
    nrtl_nonmatrix["parameters"]["interaction"] = 1.0
    with pytest.raises(ParameterDatabaseError, match="must be a matrix"):
        activity_model(ModelParameterSet.from_document(nrtl_nonmatrix))

    wilson_nonvector = _model_document(
        id="test.wilson",
        model="Wilson",
        parameters={
            "components": ["water"],
            "interaction": [[0.0]],
            "molar_volumes": 1.0,
        },
    )
    with pytest.raises(ParameterDatabaseError, match="must be a vector"):
        activity_model(ModelParameterSet.from_document(wilson_nonvector))
    wilson_bad_vector = deepcopy(wilson_nonvector)
    wilson_bad_vector["parameters"]["molar_volumes"] = [["bad"]]
    with pytest.raises(ParameterDatabaseError, match="not numeric"):
        activity_model(ModelParameterSet.from_document(wilson_bad_vector))

    hv_base = load_model_parameters("pedersen-hv-propane-water").as_dict()
    hv_base["covolumes"] = [1.8e-5, 5.7e-5]
    hv = ModelParameterSet("activity.hv-explicit", "activity", "HV-NRTL", "1", hv_base)
    assert isinstance(activity_model(hv), HuronVidalNRTL)

    missing_source = deepcopy(hv_base)
    missing_source.pop("covolumes")
    missing_source.pop("covolume_cubic_parameter_set")
    with pytest.raises(ParameterDatabaseError, match="requires covolumes"):
        activity_model(
            ModelParameterSet(
                "activity.hv-missing",
                "activity",
                "HV-NRTL",
                "1",
                missing_source,
            )
        )
    bad_source = deepcopy(missing_source)
    bad_source["covolume_cubic_parameter_set"] = "poling"
    with pytest.raises(ParameterDatabaseError, match="cannot provide"):
        activity_model(ModelParameterSet("activity.hv-bad", "activity", "HV-NRTL", "1", bad_source))


def test_cpa_parameter_payload_error_branches():
    with pytest.raises(ParameterDatabaseError, match="acentric"):
        CPAEOS((CPAComponent("ammonia", 405.56, 0.1, 1.0e-5, 0.5),))
    missing_components = ModelParameterSet("cpa.missing", "cpa", "CPA", "1", {})
    with pytest.raises(ParameterDatabaseError, match="components mapping"):
        cpa_eos(("water",), missing_components)
    invalid_scheme = load_model_parameters("folas-2005").as_dict()
    invalid_scheme["components"]["water"]["scheme"] = "3C"
    with pytest.raises(ParameterDatabaseError, match="association scheme"):
        cpa_eos(
            ("water",),
            ModelParameterSet("cpa.scheme", "cpa", "CPA", "1", invalid_scheme),
        )
    invalid_rule = load_model_parameters("folas-2005").as_dict()
    invalid_rule["default_combining_rule"] = "bad"
    with pytest.raises(ParameterDatabaseError, match="combining rule"):
        cpa_eos(
            ("water",),
            ModelParameterSet("cpa.rule", "cpa", "CPA", "1", invalid_rule),
        )


def test_cubic_parameter_payload_remaining_error_branches():
    data = load_model_parameters("pr78").as_dict()
    data["alpha"].pop("low")
    with pytest.raises(ParameterDatabaseError, match=r"alpha\.low"):
        cubic_constants(ModelParameterSet("cubic.no-low", "cubic", "PR", "1", data))
    data = load_model_parameters("pr78").as_dict()
    data["alpha"]["low"]["coefficients"][0] = "bad"
    with pytest.raises(ParameterDatabaseError, match="coefficients must be numeric"):
        cubic_constants(ModelParameterSet("cubic.bad-low", "cubic", "PR", "1", data))

    incomplete = CubicConstants(
        "bad-pr78",
        0.45,
        0.078,
        2.4,
        -0.4,
        "pr78",
        (0.37, 1.54, -0.27),
    )
    model = cubic_eos(component_set(("methane",)), incomplete)
    with pytest.raises(ParameterDatabaseError, match="high-alpha"):
        model.pure_parameters(torch.tensor(300.0, dtype=DTYPE))


def _binary_parameter_set(data, identifier="binary.custom"):
    return ModelParameterSet(
        identifier,
        "binary_interaction",
        "petroleum-cubic-kij",
        "1",
        data,
    )


def test_binary_interaction_payload_error_branches():
    components = component_set(("nitrogen", "carbon_dioxide", "methane"))
    base = load_model_parameters("whitson-bip").as_dict()
    cases = []
    invalid_hydrocarbons = deepcopy(base)
    invalid_hydrocarbons["hydrocarbons"] = "methane"
    cases.append((invalid_hydrocarbons, "string list"))
    invalid_names = deepcopy(base)
    invalid_names["hydrocarbons"] = [1]
    cases.append((invalid_names, "string list"))
    invalid_aggregate = deepcopy(base)
    invalid_aggregate["aggregate_from"] = "unknown"
    cases.append((invalid_aggregate, "aggregate_from"))
    invalid_eos = deepcopy(base)
    invalid_eos["eos"] = []
    cases.append((invalid_eos, "eos mapping"))
    missing_table = deepcopy(base)
    missing_table["eos"].pop("PR")
    cases.append((missing_table, "no PR table"))
    incomplete = deepcopy(base)
    incomplete["eos"]["PR"].pop("hydrocarbon_rows")
    cases.append((incomplete, "incomplete"))
    for index, (data, match) in enumerate(cases):
        with pytest.raises(ParameterDatabaseError, match=match):
            binary_interaction(
                components,
                _binary_parameter_set(data, f"binary.case-{index}"),
            )

    missing_pair = deepcopy(base)
    missing_pair["eos"]["PR"]["nonhydrocarbon_pairs"].clear()
    with pytest.raises(ParameterDatabaseError, match="lacks pair"):
        binary_interaction(
            component_set(("nitrogen", "carbon_dioxide")),
            _binary_parameter_set(missing_pair, "binary.no-pair"),
        )
    missing_row = deepcopy(base)
    missing_row["eos"]["PR"]["hydrocarbon_rows"].pop("nitrogen")
    with pytest.raises(ParameterDatabaseError, match="lacks row"):
        binary_interaction(
            component_set(("nitrogen", "methane")),
            _binary_parameter_set(missing_row, "binary.no-row"),
        )
    bad_value = deepcopy(base)
    bad_value["eos"]["PR"]["hydrocarbon_rows"]["nitrogen"][0] = "bad"
    with pytest.raises(ParameterDatabaseError, match="must be numeric"):
        binary_interaction(
            component_set(("nitrogen", "methane")),
            _binary_parameter_set(bad_value, "binary.bad-value"),
        )


def test_standard_state_payload_error_branches():
    with pytest.raises(ParameterDatabaseError, match="components mapping"):
        ideal_gas_polynomial(
            ("water",),
            ModelParameterSet("standard.empty", "standard_state", "ideal", "1", {}),
        )
    with pytest.raises(KeyError, match="no standard"):
        ideal_gas_polynomial(("methanol",), "poling")

    base = load_model_parameters("poling").as_dict()
    base["components"]["water"]["coefficients"] = []
    with pytest.raises(ParameterDatabaseError, match="must be numeric"):
        ideal_gas_polynomial(
            ("water",),
            ModelParameterSet("standard.bad-coeff", "standard_state", "ideal", "1", base),
        )
    base = load_model_parameters("poling").as_dict()
    base["components"]["water"]["reference_enthalpy"] = "bad"
    with pytest.raises(ParameterDatabaseError, match="reference properties"):
        ideal_gas_polynomial(
            ("water",),
            ModelParameterSet("standard.bad-ref", "standard_state", "ideal", "1", base),
        )
    base = load_model_parameters("poling").as_dict()
    base["components"]["water"]["coefficients"] = [1.0]
    with pytest.raises(ParameterDatabaseError, match="orders must match"):
        ideal_gas_polynomial(
            ("water", "methane"),
            ModelParameterSet("standard.orders", "standard_state", "ideal", "1", base),
        )
    base = load_model_parameters("poling").as_dict()
    base.pop("default_reference_temperature")
    with pytest.raises(ParameterDatabaseError, match="default_reference_temperature"):
        ideal_gas_polynomial(
            ("water",),
            ModelParameterSet("standard.no-temp", "standard_state", "ideal", "1", base),
        )


def _custom_multiparameter(identifier, model, data, references=()):
    return ModelParameterSet(
        identifier,
        "multiparameter",
        model,
        "test",
        data,
        references=references,
    )


def test_multiparameter_parameter_payload_error_and_fallback_branches():
    gerg = load_model_parameters("gerg2008").as_dict()
    no_order = deepcopy(gerg)
    no_order.pop("component_order")
    with pytest.raises(ParameterDatabaseError, match="component_order"):
        multiparameter_eos(_custom_multiparameter("multiparameter.no-order", "GERG-test", no_order))
    duplicate_order = deepcopy(gerg)
    duplicate_order["component_order"][1] = duplicate_order["component_order"][0]
    with pytest.raises(ParameterDatabaseError, match="must be unique"):
        multiparameter_eos(
            _custom_multiparameter("multiparameter.duplicate", "GERG-test", duplicate_order)
        )
    no_components = deepcopy(gerg)
    no_components["components"] = []
    with pytest.raises(ParameterDatabaseError, match="components and pairs"):
        multiparameter_eos(
            _custom_multiparameter("multiparameter.no-components", "GERG-test", no_components)
        )
    no_gas = deepcopy(gerg)
    no_gas.pop("gas_constant")
    with pytest.raises(ParameterDatabaseError, match="gas_constant"):
        multiparameter_eos(
            _custom_multiparameter("multiparameter.no-gas", "GERG-test", no_gas),
            ("methane",),
        )
    no_reference = deepcopy(gerg)
    no_reference.pop("reference")
    fallback = multiparameter_eos(
        _custom_multiparameter(
            "multiparameter.reference",
            "GERG-test",
            no_reference,
            ({"doi": "10.test/gerg"},),
        ),
        ("methane",),
    )
    assert fallback.metadata.reference == "10.test/gerg"

    eoscg = load_model_parameters("eoscg2021").as_dict()
    no_components = deepcopy(eoscg)
    no_components["pairs"] = []
    with pytest.raises(ParameterDatabaseError, match="components and pairs"):
        multiparameter_eos(
            _custom_multiparameter("multiparameter.eoscg-no-pairs", "EOS-CG-test", no_components)
        )
    no_gas = deepcopy(eoscg)
    no_gas.pop("gas_constant")
    with pytest.raises(ParameterDatabaseError, match="gas_constant"):
        multiparameter_eos(
            _custom_multiparameter("multiparameter.eoscg-no-gas", "EOS-CG-test", no_gas),
            ("carbon_dioxide",),
        )
    no_reference = deepcopy(eoscg)
    no_reference.pop("reference")
    fallback = multiparameter_eos(
        _custom_multiparameter(
            "multiparameter.eoscg-reference",
            "EOS-CG-test",
            no_reference,
            ({"citation": "EOS-CG test"},),
        ),
        ("carbon_dioxide",),
    )
    assert fallback.metadata.reference == "EOS-CG test"

    pressure, acentric = named_module._initialization_tensors(
        ("not_in_database",),
        torch.tensor([math.nan], dtype=DTYPE),
    )
    assert torch.isnan(pressure).all() and torch.isnan(acentric).all()


@pytest.mark.parametrize("departure", [1.0, ["not-a-mapping"]])
def test_multiparameter_departure_payload_validation(departure):
    data = {
        "pairs": {
            "a|b": {
                "first": "a",
                "second": "b",
                "beta_temperature": 1.0,
                "gamma_temperature": 1.0,
                "beta_volume": 1.0,
                "gamma_volume": 1.0,
                "departure_scale": 0.0,
                "departure": departure,
            }
        }
    }
    with pytest.raises(ParameterDatabaseError, match="departure"):
        named_module._mixture_tables(
            data,
            ("a", "b"),
            dtype=DTYPE,
            device=None,
        )
