"""Generate the Beckmüller et al. (2021) H2-tailored GERG parameter document.

The four new binary reducing and departure functions are transcribed from
Tables 2 and 3 of doi:10.1063/5.0040533. The GERG-2008 pure-fluid inventories
for CH4, N2, CO, and CO2 and their unaffected binary pairs are selected from
the bundled GERG-2008 document. Normal-hydrogen coefficients are selected
from the standalone bundled Leachman parameter document.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "src" / "torch_flash" / "data" / "models" / "multiparameter"
OUTPUT = MODEL_ROOT / "gerg-2008-hydrogen-2021.yaml"

COMPONENT_ORDER = (
    "methane",
    "nitrogen",
    "carbon_monoxide",
    "carbon_dioxide",
    "hydrogen",
)
RAW_NAMES = {
    "methane": "methane",
    "nitrogen": "nitrogen",
    "carbon_monoxide": "carbonmonoxide",
    "carbon_dioxide": "carbondioxide",
    "hydrogen": "hydrogen",
}

TAILORED_PAIRS: dict[str, dict[str, Any]] = {
    "methane|hydrogen": {
        "first": "methane",
        "second": "hydrogen",
        "beta_temperature": 1.033,
        "gamma_temperature": 1.335,
        "beta_volume": 1.001,
        "gamma_volume": 1.075,
        "n": [1.690, -1.240, 4.630, 2.900, -3.620, 5.613, -1.040, -8.670],
        "t": [0.269, 0.410, 1.550, 2.120, 0.039, 0.320, 0.414, 0.774],
        "d": [1, 2, 1, 2, 1, 2, 3, 1],
        "l": [0, 0, 1, 1, 0, 0, 0, 0],
        "eta": [0, 0, 0, 0, 0.2080, 0.0327, 0.0770, 0.1540],
        "beta": [0, 0, 0, 0, 0.640, 0.369, 0.359, 0.374],
        "gamma": [0, 0, 0, 0, 1.224, 1.603, 1.655, 2.270],
        "epsilon": [0, 0, 0, 0, 1.59, 0.13, 1.70, 0.08],
    },
    "nitrogen|hydrogen": {
        "first": "nitrogen",
        "second": "hydrogen",
        "beta_temperature": 1.022,
        "gamma_temperature": 1.250,
        "beta_volume": 0.986,
        "gamma_volume": 0.783,
        "n": [-1.812, -0.612, -0.485, 0.157, 2.762, 5.195, -3.751, -5.506],
        "t": [0.924, 0.411, 2.846, 3.565, 3.186, 0.748, 2.532, 1.114],
        "d": [1, 2, 1, 2, 1, 1, 1, 1],
        "l": [1, 1, 2, 2, 0, 0, 0, 0],
        "eta": [0, 0, 0, 0, 1.83, 0.07, 1.82, 0.17],
        "beta": [0, 0, 0, 0, 1.08, 0.31, 1.14, 0.21],
        "gamma": [0, 0, 0, 0, 1.37, 0.89, 1.55, 0.21],
        "epsilon": [0, 0, 0, 0, 2.50, 1.45, 2.50, 1.55],
    },
    "carbonmonoxide|hydrogen": {
        "first": "carbonmonoxide",
        "second": "hydrogen",
        "beta_temperature": 1.078,
        "gamma_temperature": 1.105,
        "beta_volume": 1.037,
        "gamma_volume": 1.040,
        "n": [-0.521, -0.387, -2.590, 4.350],
        "t": [2.250, 0.473, 0.585, 0.091],
        "d": [1, 2, 1, 2],
        "l": [0, 0, 0, 0],
        "eta": [0, 0, 0.647, 0.344],
        "beta": [0, 0, 0.751, 0.660],
        "gamma": [0, 0, 1.86, 2.23],
        "epsilon": [0, 0, 1.380, 0.773],
    },
    "carbondioxide|hydrogen": {
        "first": "carbondioxide",
        "second": "hydrogen",
        "beta_temperature": 0.964,
        "gamma_temperature": 2.014,
        "beta_volume": 1.200,
        "gamma_volume": 0.825,
        "n": [3.56, -0.97, -4.56, 12.12, -2.43, -3.17],
        "t": [1.40, 1.12, 1.87, 0.25, 1.53, 2.28],
        "d": [1, 2, 1, 2, 3, 1],
        "l": [0, 0, 0, 0, 0, 0],
        "eta": [0, 0, 0.575, 0.210, 0.295, 0.135],
        "beta": [0, 0, 0.510, 0.826, 0.410, 1.0],
        "gamma": [0, 0, 0.22, 2.12, 1.44, 1.70],
        "epsilon": [0, 0, 0.52, 0.15, 0.23, 0.14],
    },
}


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return document


def _pair_lookup(pairs: dict[str, Any]) -> dict[frozenset[str], dict[str, Any]]:
    return {frozenset((pair["first"], pair["second"])): pair for pair in pairs.values()}


def _tailored_pair(values: dict[str, Any]) -> dict[str, Any]:
    interaction_keys = (
        "beta_temperature",
        "gamma_temperature",
        "beta_volume",
        "gamma_volume",
    )
    departure_keys = ("n", "t", "d", "l", "eta", "beta", "gamma", "epsilon")
    return {
        "first": values["first"],
        "second": values["second"],
        **{key: values[key] for key in interaction_keys},
        "departure_scale": 1.0,
        "departure": {
            "type": "Gaussian+Exponential",
            **{key: values[key] for key in departure_keys},
        },
    }


def build_document() -> dict[str, Any]:
    """Return the complete five-component parameter document."""
    gerg = _load(MODEL_ROOT / "gerg-2008.yaml")["parameters"]
    leachman = _load(MODEL_ROOT.parent / "pure_helmholtz" / "leachman-2009-normal-hydrogen.yaml")[
        "parameters"
    ]

    components = {
        RAW_NAMES[name]: deepcopy(gerg["components"][RAW_NAMES[name]])
        for name in COMPONENT_ORDER
        if name != "hydrogen"
    }
    components["hydrogen"] = deepcopy(leachman["components"]["hydrogen"])

    base_pairs = _pair_lookup(gerg["pairs"])
    pairs: dict[str, Any] = {}
    raw_order = tuple(RAW_NAMES[name] for name in COMPONENT_ORDER)
    for index, first in enumerate(raw_order):
        for second in raw_order[index + 1 :]:
            key = f"{first}|{second}"
            tailored = TAILORED_PAIRS.get(key)
            pairs[key] = (
                _tailored_pair(tailored)
                if tailored is not None
                else deepcopy(base_pairs[frozenset((first, second))])
            )

    return {
        "format": "torch-flash-model-parameters",
        "schema_version": 1,
        "id": "multiparameter.gerg-2008-hydrogen-2021",
        "model_kind": "multiparameter",
        "model": "GERG-2008 H2-tailored (Beckmüller et al. 2021)",
        "version": "2021",
        "description": (
            "Five-component H2-tailored GERG mixture model using GERG-2008 "
            "pure fluids for CH4, N2, CO, and CO2 and Leachman normal hydrogen."
        ),
        "units": {
            "gas_constant": "J mol^-1 K^-1",
            "critical_temperature": "K",
            "critical_pressure": "Pa",
            "critical_density": "mol m^-3",
            "molar_mass": "kg mol^-1",
            "helmholtz_coefficients": ("dimensionless unless term definition states otherwise"),
        },
        "references": [
            {
                "citation": (
                    "Beckmüller et al., Journal of Physical and Chemical "
                    "Reference Data 50, 013102 (2021)"
                ),
                "doi": "10.1063/5.0040533",
            },
            {
                "citation": (
                    "Kunz and Wagner, Journal of Chemical and Engineering Data 57, 3032-3091 (2012)"
                ),
                "doi": "10.1021/je300655b",
            },
            {
                "citation": (
                    "Leachman et al., Journal of Physical and Chemical "
                    "Reference Data 38, 721-748 (2009)"
                ),
                "doi": "10.1063/1.3160306",
            },
        ],
        "parameters": {
            "component_order": list(COMPONENT_ORDER),
            "components": components,
            "gas_constant": 8.314472,
            "pairs": pairs,
            "reference": "doi:10.1063/5.0040533, Tables 1-3",
        },
    }


def main() -> int:
    """Write the generated parameter document."""
    document = build_document()
    with OUTPUT.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=False, allow_unicode=False)
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
