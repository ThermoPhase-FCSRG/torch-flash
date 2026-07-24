"""Complete native GERG-2008 and EOS-CG-2021 model constructors.

Primary mixture sources are Kunz and Wagner (2012),
doi:10.1021/je300655b, and Neumann et al. (2023),
doi:10.1007/s10765-023-03263-6. The latter's supplementary coefficient tables
are linked from ``src/torch_flash/eos/data/README.md``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from torch_flash.components import canonical_component_name, component
from torch_flash.config import resolve_tensor_options
from torch_flash.database import (
    ModelParameterSet,
    ParameterSource,
    load_model_parameters,
)
from torch_flash.exceptions import ParameterDatabaseError

from .multifluid import (
    GaoBTerms,
    HelmholtzTerms,
    IdealHelmholtzTerms,
    MultiFluidEOS,
    MultifluidMetadata,
    NonAnalyticTerms,
)

GERG2008_COMPONENTS = (
    "methane",
    "nitrogen",
    "carbon_dioxide",
    "ethane",
    "propane",
    "n_butane",
    "isobutane",
    "n_pentane",
    "isopentane",
    "n_hexane",
    "n_heptane",
    "n_octane",
    "hydrogen",
    "oxygen",
    "carbon_monoxide",
    "water",
    "helium",
    "argon",
    "hydrogen_sulfide",
    "n_nonane",
    "n_decane",
)

EOSCG2021_COMPONENTS = (
    "carbon_dioxide",
    "water",
    "nitrogen",
    "oxygen",
    "argon",
    "carbon_monoxide",
    "hydrogen",
    "methane",
    "hydrogen_sulfide",
    "sulfur_dioxide",
    "mea",
    "dea",
    "hydrogen_chloride",
    "chlorine",
    "ammonia",
    "mdea",
)

_GERG_TO_CANONICAL = {
    "carbondioxide": "carbon_dioxide",
    "carbonmonoxide": "carbon_monoxide",
    "hydrogensulfide": "hydrogen_sulfide",
    "n-butane": "n_butane",
    "n-pentane": "n_pentane",
    "n-hexane": "n_hexane",
    "n-heptane": "n_heptane",
    "n-octane": "n_octane",
    "n-nonane": "n_nonane",
    "n-decane": "n_decane",
}
_GERG_FROM_CANONICAL = {canonical: raw for raw, canonical in _GERG_TO_CANONICAL.items()}

_ALIASES = {
    "ar": "argon",
    "ch4": "methane",
    "cl2": "chlorine",
    "co": "carbon_monoxide",
    "co2": "carbon_dioxide",
    "h2": "hydrogen",
    "h2o": "water",
    "h2s": "hydrogen_sulfide",
    "hcl": "hydrogen_chloride",
    "n2": "nitrogen",
    "nh3": "ammonia",
    "o2": "oxygen",
    "so2": "sulfur_dioxide",
    "i_butane": "isobutane",
    "i_pentane": "isopentane",
}

_TERM_KEYS = (
    "n",
    "d",
    "t",
    "decay",
    "eta",
    "epsilon",
    "beta",
    "gamma",
    "linear_density",
    "linear_shift",
)


def _read_data(source: ParameterSource) -> Mapping[str, Any]:
    """Return the cached, recursively read-only coefficient payload."""
    return load_model_parameters(source).parameters


def _canonical_name(name: str) -> str:
    normalized = canonical_component_name(name, strict=False)
    return _ALIASES.get(normalized, normalized)


def _validate_names(names: tuple[str, ...] | None, supported: tuple[str, ...]) -> tuple[str, ...]:
    selected = supported if names is None else tuple(_canonical_name(name) for name in names)
    if not selected:
        raise ValueError("at least one component is required")
    if len(set(selected)) != len(selected):
        raise ValueError("component names must be unique")
    unsupported = tuple(name for name in selected if name not in supported)
    if unsupported:
        raise ValueError(f"unsupported components: {', '.join(unsupported)}")
    return selected


def _empty_record() -> dict[str, float]:
    return dict.fromkeys(_TERM_KEYS, 0.0)


def _append_regular_terms(records: list[dict[str, float]], block: Mapping[str, Any]) -> None:
    kind = block["type"]
    count = len(block["n"])
    for index in range(count):
        record = _empty_record()
        record.update(
            n=float(block["n"][index]),
            d=float(block["d"][index]),
            t=float(block["t"][index]),
        )
        if kind == "ResidualHelmholtzPower":
            record["decay"] = float(block["l"][index])
        elif kind == "ResidualHelmholtzGaussian":
            for key in ("eta", "epsilon", "beta", "gamma"):
                record[key] = float(block[key][index])
        elif kind == "ResidualHelmholtzGERG2008":
            record["eta"] = float(block["eta"][index])
            record["epsilon"] = float(block["epsilon"][index])
            record["linear_density"] = float(block["beta"][index])
            record["linear_shift"] = float(block["gamma"][index])
        elif kind == "Gaussian+Exponential":
            gaussian_values = tuple(
                float(block[key][index]) for key in ("eta", "epsilon", "beta", "gamma")
            )
            if all(value == 0.0 for value in gaussian_values):
                record["decay"] = float(block["l"][index])
            else:
                for key, value in zip(
                    ("eta", "epsilon", "beta", "gamma"), gaussian_values, strict=True
                ):
                    record[key] = value
        else:
            raise ValueError(f"unsupported regular Helmholtz term type: {kind}")
        records.append(record)


def _pad_records(
    rows: list[list[dict[str, float]]],
    leading_shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> HelmholtzTerms:
    width = max(1, *(len(row) for row in rows))
    table = {key: torch.zeros((len(rows), width), dtype=dtype, device=device) for key in _TERM_KEYS}
    for row_index, row in enumerate(rows):
        for column_index, record in enumerate(row):
            for key in _TERM_KEYS:
                table[key][row_index, column_index] = record[key]
    shaped = {key: value.reshape(*leading_shape, width) for key, value in table.items()}
    return HelmholtzTerms(
        shaped["n"],
        shaped["d"],
        shaped["t"],
        shaped["decay"],
        eta=shaped["eta"],
        epsilon=shaped["epsilon"],
        beta=shaped["beta"],
        gamma=shaped["gamma"],
        linear_density=shaped["linear_density"],
        linear_shift=shaped["linear_shift"],
    )


def _pad_special(
    rows: list[list[dict[str, float]]],
    keys: tuple[str, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
    safe_one: tuple[str, ...] = (),
) -> dict[str, Tensor]:
    width = max(1, *(len(row) for row in rows))
    result = {}
    for key in keys:
        fill = 1.0 if key in safe_one else 0.0
        result[key] = torch.full((len(rows), width), fill, dtype=dtype, device=device)
    for row_index, row in enumerate(rows):
        for column_index, record in enumerate(row):
            for key in keys:
                result[key][row_index, column_index] = record[key]
    return result


def _canonical_ideal_blocks(
    blocks: Sequence[Mapping[str, Any]], critical_temperature: float
) -> tuple[dict[str, float], list[dict[str, float]], list[dict[str, float]]]:
    lead = {
        "lead_constant": 0.0,
        "lead_tau": 0.0,
        "log_tau": 0.0,
        "tau_log_tau": 0.0,
    }
    power: list[dict[str, float]] = []
    planck: list[dict[str, float]] = []
    for block in blocks:
        kind = block["type"]
        if kind in ("IdealGasHelmholtzLead", "IdealGasHelmholtzEnthalpyEntropyOffset"):
            lead["lead_constant"] += float(block["a1"])
            lead["lead_tau"] += float(block["a2"])
        elif kind == "IdealGasHelmholtzLogTau":
            lead["log_tau"] += float(block["a"])
        elif kind == "IdealGasHelmholtzPower":
            power.extend(
                {"n": float(n), "t": float(t)} for n, t in zip(block["n"], block["t"], strict=True)
            )
        elif kind in (
            "IdealGasHelmholtzPlanckEinstein",
            "IdealGasHelmholtzPlanckEinsteinFunctionT",
        ):
            theta = (
                block["t"]
                if kind == "IdealGasHelmholtzPlanckEinstein"
                else [float(value) / float(block["Tcrit"]) for value in block["v"]]
            )
            planck.extend(
                {"n": float(n), "theta": float(value)}
                for n, value in zip(block["n"], theta, strict=True)
            )
        elif kind == "IdealGasHelmholtzCP0PolyT":
            reference_temperature = float(block["T0"])
            reducing_temperature = float(block.get("Tc", critical_temperature))
            for coefficient, exponent in zip(block["c"], block["t"], strict=True):
                coefficient = float(coefficient)
                exponent = float(exponent)
                if exponent == 0.0:
                    ratio = reference_temperature / reducing_temperature
                    lead["lead_constant"] += coefficient * (1.0 + math.log(ratio))
                    lead["lead_tau"] -= coefficient * ratio
                    lead["log_tau"] += coefficient
                elif exponent == -1.0:
                    tau0 = reducing_temperature / reference_temperature
                    scaled = coefficient / reducing_temperature
                    lead["lead_constant"] -= scaled * tau0
                    lead["lead_tau"] += scaled * (math.log(tau0) + 1.0)
                    lead["tau_log_tau"] -= scaled
                else:
                    lead["lead_constant"] += (
                        coefficient * reference_temperature**exponent / exponent
                    )
                    lead["lead_tau"] -= (
                        coefficient
                        * reference_temperature ** (exponent + 1.0)
                        / (reducing_temperature * (exponent + 1.0))
                    )
                    power.append(
                        {
                            "n": -coefficient
                            * reducing_temperature**exponent
                            / (exponent * (exponent + 1.0)),
                            "t": -exponent,
                        }
                    )
        else:
            raise ValueError(f"unsupported ideal Helmholtz term type: {kind}")
    return lead, power, planck


def _pad_ideal(
    leads: list[dict[str, float]],
    power_rows: list[list[dict[str, float]]],
    planck_rows: list[list[dict[str, float]]],
    gerg_rows: list[list[dict[str, float]]],
    gas_scale: list[float],
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> IdealHelmholtzTerms:
    count = len(leads)

    def table(rows: list[list[dict[str, float]]], key: str, *, fill: float = 0.0) -> Tensor:
        width = max(1, *(len(row) for row in rows))
        result = torch.full((count, width), fill, dtype=dtype, device=device)
        for row_index, row in enumerate(rows):
            for column_index, record in enumerate(row):
                result[row_index, column_index] = record[key]
        return result

    return IdealHelmholtzTerms(
        **{
            key: torch.tensor([lead[key] for lead in leads], dtype=dtype, device=device)
            for key in ("lead_constant", "lead_tau", "log_tau", "tau_log_tau")
        },
        power_n=table(power_rows, "n"),
        power_t=table(power_rows, "t"),
        planck_n=table(planck_rows, "n"),
        planck_theta=table(planck_rows, "theta", fill=1.0),
        gerg_n=table(gerg_rows, "n"),
        gerg_theta=table(gerg_rows, "theta", fill=1.0),
        gerg_sign=table(gerg_rows, "sign"),
        gas_scale=torch.tensor(gas_scale, dtype=dtype, device=device),
    )


def _ideal_terms_eoscg(
    data: Mapping[str, Any],
    selected: tuple[str, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> IdealHelmholtzTerms:
    leads = []
    power_rows = []
    planck_rows = []
    gas_scale = []
    for name in selected:
        component = data["components"][name]
        lead, power, planck = _canonical_ideal_blocks(
            component["ideal"], float(component["critical_temperature"])
        )
        leads.append(lead)
        power_rows.append(power)
        planck_rows.append(planck)
        gas_scale.append(float(component["source_gas_constant"]) / float(data["gas_constant"]))
    return _pad_ideal(
        leads,
        power_rows,
        planck_rows,
        [[] for _ in selected],
        gas_scale,
        dtype=dtype,
        device=device,
    )


def _ideal_terms_gerg(
    data: Mapping[str, Any],
    selected: tuple[str, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> IdealHelmholtzTerms:
    leads = []
    gerg_rows = []
    gas_scale = []
    for name in selected:
        raw_name = _GERG_FROM_CANONICAL.get(name, name)
        ideal = data["components"][raw_name]["ideal"]
        n0 = ideal["n0"]
        theta0 = ideal["theta0"]
        leads.append(
            {
                "lead_constant": float(n0[1]),
                "lead_tau": float(n0[2]),
                "log_tau": float(n0[3]),
                "tau_log_tau": 0.0,
            }
        )
        gerg_rows.append(
            [
                {"n": float(n0[index]), "theta": abs(float(theta0[index])), "sign": sign}
                for index, sign in ((4, 1.0), (5, -1.0), (6, 1.0), (7, -1.0))
                if float(n0[index]) != 0.0
            ]
        )
        gas_scale.append(float(ideal["source_gas_constant"]) / float(data["gas_constant"]))
    return _pad_ideal(
        leads,
        [[] for _ in selected],
        [[] for _ in selected],
        gerg_rows,
        gas_scale,
        dtype=dtype,
        device=device,
    )


def _pure_rows_eoscg(
    data: Mapping[str, Any], selected: tuple[str, ...]
) -> tuple[
    list[list[dict[str, float]]],
    list[list[dict[str, float]]],
    list[list[dict[str, float]]],
]:
    regular_rows: list[list[dict[str, float]]] = []
    gaob_rows: list[list[dict[str, float]]] = []
    nonanalytic_rows: list[list[dict[str, float]]] = []
    for name in selected:
        regular: list[dict[str, float]] = []
        gaob: list[dict[str, float]] = []
        nonanalytic: list[dict[str, float]] = []
        for block in data["components"][name]["residual"]:
            kind = block["type"]
            if kind == "ResidualHelmholtzGaoB":
                for index in range(len(block["n"])):
                    gaob.append(
                        {
                            key: float(block[key][index])
                            for key in ("n", "d", "t", "eta", "epsilon", "beta", "gamma", "b")
                        }
                    )
            elif kind == "ResidualHelmholtzNonAnalytic":
                for index in range(len(block["n"])):
                    nonanalytic.append(
                        {
                            "n": float(block["n"][index]),
                            "capital_a": float(block["A"][index]),
                            "capital_b": float(block["B"][index]),
                            "capital_c": float(block["C"][index]),
                            "capital_d": float(block["D"][index]),
                            "a": float(block["a"][index]),
                            "b": float(block["b"][index]),
                            "beta": float(block["beta"][index]),
                        }
                    )
            else:
                _append_regular_terms(regular, block)
        regular_rows.append(regular)
        gaob_rows.append(gaob)
        nonanalytic_rows.append(nonanalytic)
    return regular_rows, gaob_rows, nonanalytic_rows


def _pure_rows_gerg(
    data: Mapping[str, Any], selected: tuple[str, ...]
) -> list[list[dict[str, float]]]:
    rows = []
    for name in selected:
        raw_name = _GERG_FROM_CANONICAL.get(name, name)
        component = data["components"][raw_name]
        records = []
        for n, d, t, coefficient, exponent in zip(
            component["n"],
            component["d"],
            component["t"],
            component["c"],
            component["l"],
            strict=True,
        ):
            record = _empty_record()
            record.update(n=float(n), d=float(d), t=float(t))
            record["decay"] = float(exponent) if coefficient != 0.0 else 0.0
            records.append(record)
        rows.append(records)
    return rows


def _pair_lookup(
    data: Mapping[str, Any], to_canonical: dict[str, str] | None = None
) -> dict[frozenset[str], Mapping[str, Any]]:
    mapping = {} if to_canonical is None else to_canonical
    result: dict[frozenset[str], Mapping[str, Any]] = {}
    for pair in data["pairs"].values():
        first = mapping.get(pair["first"], pair["first"])
        second = mapping.get(pair["second"], pair["second"])
        result[frozenset((first, second))] = {**pair, "first": first, "second": second}
    return result


def _mixture_tables(
    data: Mapping[str, Any],
    selected: tuple[str, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
    to_canonical: dict[str, str] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, HelmholtzTerms]:
    count = len(selected)
    ones = torch.ones((count, count), dtype=dtype, device=device)
    beta_temperature = ones.clone()
    gamma_temperature = ones.clone()
    beta_volume = ones.clone()
    gamma_volume = ones.clone()
    scale = torch.zeros_like(ones)
    departure_rows: list[list[dict[str, float]]] = [[] for _ in range(count * count)]
    lookup = _pair_lookup(data, to_canonical)
    for i, first in enumerate(selected):
        for j in range(i + 1, count):
            second = selected[j]
            pair = lookup[frozenset((first, second))]
            forward = first == pair["first"]
            for target, key in (
                (beta_temperature, "beta_temperature"),
                (beta_volume, "beta_volume"),
            ):
                value = float(pair[key])
                target[i, j] = value if forward else 1.0 / value
                target[j, i] = 1.0 / target[i, j]
            gamma_temperature[i, j] = gamma_temperature[j, i] = float(pair["gamma_temperature"])
            gamma_volume[i, j] = gamma_volume[j, i] = float(pair["gamma_volume"])
            scale[i, j] = scale[j, i] = float(pair["departure_scale"])
            blocks = pair.get("departure", [])
            if isinstance(blocks, Mapping):
                blocks = (blocks,)
            if not isinstance(blocks, Sequence):
                raise ParameterDatabaseError("multifluid departure terms must be a sequence")
            records: list[dict[str, float]] = []
            for block in blocks:
                if not isinstance(block, Mapping):
                    raise ParameterDatabaseError("multifluid departure term must be a mapping")
                if "type" not in block:
                    block = {**block, "type": "ResidualHelmholtzGERG2008"}
                _append_regular_terms(records, block)
            departure_rows[i * count + j] = records
            departure_rows[j * count + i] = records
    terms = _pad_records(departure_rows, (count, count), dtype=dtype, device=device)
    return (
        beta_temperature,
        gamma_temperature,
        beta_volume,
        gamma_volume,
        scale,
        terms,
    )


def _component_tensors(
    data: Mapping[str, Any],
    selected: tuple[str, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
    from_canonical: dict[str, str] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mapping = {} if from_canonical is None else from_canonical
    components = [data["components"][mapping.get(name, name)] for name in selected]

    def tensor(key: str, default: float = float("nan")) -> Tensor:
        return torch.tensor(
            [component.get(key, default) for component in components],
            dtype=dtype,
            device=device,
        )

    return (
        tensor("critical_temperature"),
        tensor("critical_density"),
        tensor("molar_mass"),
        tensor("critical_pressure"),
    )


def _initialization_tensors(
    selected: tuple[str, ...],
    critical_pressure: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fill Wilson-initialization constants from the auditable component table."""
    pressure = critical_pressure.clone()
    acentric_factor = torch.full_like(pressure, torch.nan)
    for index, name in enumerate(selected):
        try:
            record = component(name)
        except KeyError:
            continue
        if not bool(torch.isfinite(pressure[index])):
            pressure[index] = record.critical_pressure
        if record.acentric_factor is not None:
            acentric_factor[index] = record.acentric_factor
    return pressure, acentric_factor


def _component_order(parameter_set: ModelParameterSet, data: Mapping[str, Any]) -> tuple[str, ...]:
    order = data.get("component_order")
    if not isinstance(order, Sequence) or isinstance(order, str):
        raise ParameterDatabaseError(
            f"{parameter_set.identifier!r} requires a component_order list"
        )
    canonical = tuple(_canonical_name(str(name)) for name in order)
    if len(canonical) != len(set(canonical)):
        raise ParameterDatabaseError(f"{parameter_set.identifier!r} component_order must be unique")
    return canonical


def _gerg_eos(
    parameter_set: ModelParameterSet,
    names: tuple[str, ...] | None,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
    trainable: bool,
) -> MultiFluidEOS:
    data = _read_data(parameter_set)
    supported = _component_order(parameter_set, data)
    selected = _validate_names(names, supported)
    components = data.get("components")
    pairs = data.get("pairs")
    if not isinstance(components, Mapping) or not isinstance(pairs, Mapping):
        raise ParameterDatabaseError(
            f"{parameter_set.identifier!r} requires components and pairs mappings"
        )
    expected_pairs = len(supported) * (len(supported) - 1) // 2
    if len(components) != len(supported) or len(pairs) != expected_pairs:
        raise RuntimeError(f"{parameter_set.model} coefficient inventory is incomplete")
    pure_rows = _pure_rows_gerg(data, selected)
    pure_terms = _pad_records(pure_rows, (len(selected),), dtype=dtype, device=device)
    tables = _mixture_tables(
        data,
        selected,
        dtype=dtype,
        device=device,
        to_canonical=_GERG_TO_CANONICAL,
    )
    critical_temperature, critical_density, molar_mass, critical_pressure = _component_tensors(
        data,
        selected,
        dtype=dtype,
        device=device,
        from_canonical=_GERG_FROM_CANONICAL,
    )
    critical_pressure, acentric_factor = _initialization_tensors(selected, critical_pressure)
    reference = data.get("reference")
    if not isinstance(reference, str):
        reference = "; ".join(
            str(item.get("doi", item.get("citation", ""))) for item in parameter_set.references
        )
    gas_constant = data.get("gas_constant")
    if not isinstance(gas_constant, int | float):
        raise ParameterDatabaseError(f"{parameter_set.identifier!r} requires gas_constant")
    return MultiFluidEOS(
        selected,
        critical_temperature,
        critical_density,
        molar_mass,
        pure_terms,
        tables[5],
        *tables[:5],
        MultifluidMetadata(
            parameter_set.model,
            reference,
            parameter_set.version,
            supported,
        ),
        trainable=trainable,
        gas_constant=float(gas_constant),
        ideal_terms=_ideal_terms_gerg(data, selected, dtype=dtype, device=device),
        critical_pressure=critical_pressure,
        acentric_factor=acentric_factor,
    )


def _eoscg_eos(
    parameter_set: ModelParameterSet,
    names: tuple[str, ...] | None,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
    trainable: bool,
) -> MultiFluidEOS:
    data = _read_data(parameter_set)
    supported = _component_order(parameter_set, data)
    selected = _validate_names(names, supported)
    components = data.get("components")
    pairs = data.get("pairs")
    if not isinstance(components, Mapping) or not isinstance(pairs, Mapping):
        raise ParameterDatabaseError(
            f"{parameter_set.identifier!r} requires components and pairs mappings"
        )
    expected_pairs = len(supported) * (len(supported) - 1) // 2
    if len(components) != len(supported) or len(pairs) != expected_pairs:
        raise RuntimeError(f"{parameter_set.model} coefficient inventory is incomplete")
    regular_rows, gaob_rows, nonanalytic_rows = _pure_rows_eoscg(data, selected)
    pure_terms = _pad_records(regular_rows, (len(selected),), dtype=dtype, device=device)
    gaob_values = _pad_special(
        gaob_rows,
        ("n", "d", "t", "eta", "epsilon", "beta", "gamma", "b"),
        dtype=dtype,
        device=device,
        safe_one=("b",),
    )
    nonanalytic_values = _pad_special(
        nonanalytic_rows,
        ("n", "capital_a", "capital_b", "capital_c", "capital_d", "a", "b", "beta"),
        dtype=dtype,
        device=device,
        safe_one=("a", "b", "beta"),
    )
    tables = _mixture_tables(data, selected, dtype=dtype, device=device)
    critical_temperature, critical_density, molar_mass, critical_pressure = _component_tensors(
        data, selected, dtype=dtype, device=device
    )
    critical_pressure, acentric_factor = _initialization_tensors(selected, critical_pressure)
    reference = data.get("reference")
    if not isinstance(reference, str):
        reference = "; ".join(
            str(item.get("doi", item.get("citation", ""))) for item in parameter_set.references
        )
    gas_constant = data.get("gas_constant")
    if not isinstance(gas_constant, int | float):
        raise ParameterDatabaseError(f"{parameter_set.identifier!r} requires gas_constant")
    return MultiFluidEOS(
        selected,
        critical_temperature,
        critical_density,
        molar_mass,
        pure_terms,
        tables[5],
        *tables[:5],
        MultifluidMetadata(
            parameter_set.model,
            reference,
            parameter_set.version,
            supported,
        ),
        trainable=trainable,
        gas_constant=float(gas_constant),
        pure_gaob_terms=GaoBTerms(**gaob_values),
        pure_nonanalytic_terms=NonAnalyticTerms(**nonanalytic_values),
        ideal_terms=_ideal_terms_eoscg(data, selected, dtype=dtype, device=device),
        critical_pressure=critical_pressure,
        acentric_factor=acentric_factor,
    )


def multifluid_eos(
    parameter_set: ParameterSource,
    names: tuple[str, ...] | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    trainable: bool = False,
) -> MultiFluidEOS:
    """Construct a native multifluid EoS from YAML or explicit parameters."""
    dtype, device = resolve_tensor_options(dtype, device)
    loaded = load_model_parameters(parameter_set)
    if loaded.model_kind != "multifluid":
        raise ParameterDatabaseError(
            f"{loaded.identifier!r} is {loaded.model_kind!r}, not 'multifluid'"
        )
    normalized = loaded.model.strip().lower().replace("_", "-")
    if normalized.startswith("gerg"):
        return _gerg_eos(
            loaded,
            names,
            dtype=dtype,
            device=device,
            trainable=trainable,
        )
    if normalized.startswith("eos-cg") or normalized.startswith("eoscg"):
        return _eoscg_eos(
            loaded,
            names,
            dtype=dtype,
            device=device,
            trainable=trainable,
        )
    raise ParameterDatabaseError(
        f"{loaded.identifier!r} has unsupported multifluid model {loaded.model!r}"
    )


def gerg2008(
    names: tuple[str, ...] | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    trainable: bool = False,
    parameter_set: ParameterSource = "multifluid.gerg-2008",
) -> MultiFluidEOS:
    """Construct the complete native 21-component GERG-2008 Helmholtz model.

    Reference: Kunz and Wagner (2012), doi:10.1021/je300655b.
    """
    dtype, device = resolve_tensor_options(dtype, device)
    loaded = load_model_parameters(parameter_set)
    if not loaded.model.upper().startswith("GERG-2008"):
        raise ParameterDatabaseError(
            f"gerg2008 requires a GERG-2008 parameter set, got {loaded.model!r}"
        )
    return _gerg_eos(
        loaded,
        names,
        dtype=dtype,
        device=device,
        trainable=trainable,
    )


def eoscg2021(
    names: tuple[str, ...] | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    trainable: bool = False,
    parameter_set: ParameterSource = "multifluid.eos-cg-2021",
) -> MultiFluidEOS:
    """Construct the complete native 16-component EOS-CG-2021 Helmholtz model.

    Reference: Neumann et al. (2023),
    doi:10.1007/s10765-023-03263-6, including its supplementary tables.
    """
    dtype, device = resolve_tensor_options(dtype, device)
    loaded = load_model_parameters(parameter_set)
    if not loaded.model.upper().startswith("EOS-CG-2021"):
        raise ParameterDatabaseError(
            f"eoscg2021 requires an EOS-CG-2021 parameter set, got {loaded.model!r}"
        )
    return _eoscg_eos(
        loaded,
        names,
        dtype=dtype,
        device=device,
        trainable=trainable,
    )


__all__ = [
    "EOSCG2021_COMPONENTS",
    "GERG2008_COMPONENTS",
    "eoscg2021",
    "gerg2008",
    "multifluid_eos",
]
