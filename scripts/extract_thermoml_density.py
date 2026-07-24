"""Normalize binary experimental mass-density data from NIST ThermoML JSON.

The NIST ThermoML archive is an official transcription of data reported by
cooperating journals: https://trc.nist.gov/ThermoML/Browse.  This read-only
utility accepts a locally downloaded DOI JSON file and emits a compact CSV
without requiring a network connection or a third-party JSON package.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _canonical_name(compound: dict[str, Any]) -> str:
    names = compound.get("sCommonName", ())
    if isinstance(names, str):
        names = (names,)
    name = names[0] if names else compound["sFormulaMolec"]
    return str(name).lower().replace(" ", "_")


def _typed_value(container: dict[str, Any]) -> tuple[str, Any]:
    return next((key, value) for key, value in container.items() if key != "tml_elements")


def density_records(document: dict[str, Any]) -> list[dict[str, str | float]]:
    """Return normalized binary gas-density records from a ThermoML document."""
    compounds = {
        int(item["RegNum"]["nOrgNum"]): _canonical_name(item) for item in document["Compound"]
    }
    if len(compounds) != 2:
        raise ValueError("density extraction requires an exactly binary ThermoML document")
    component_numbers = tuple(sorted(compounds))
    first_number = component_numbers[0]
    citation = document["Citation"]
    records: list[dict[str, str | float]] = []
    for dataset in document.get("PureOrMixtureData", ()):
        properties: dict[int, tuple[str, str, str]] = {}
        for item in dataset["Property"]:
            group = item["Property-MethodID"]["PropertyGroup"]
            _, method_block = _typed_value(group)
            properties[int(item["nPropNumber"])] = (
                str(method_block["ePropName"]),
                str(method_block.get("eMethodName", "")),
                str(item.get("PropPhaseID", {}).get("ePropPhase", "")),
            )
        density_numbers = {
            number for number, (name, _, _) in properties.items() if name == "Mass density, kg/m3"
        }
        if not density_numbers:
            continue

        variables: dict[int, tuple[str, int | None]] = {}
        for item in dataset["Variable"]:
            variable_type = item["VariableID"]["VariableType"]
            _, name = _typed_value(variable_type)
            reg_num = item["VariableID"].get("RegNum", {}).get("nOrgNum")
            variables[int(item["nVarNumber"])] = (
                str(name),
                None if reg_num is None else int(reg_num),
            )

        constrained_composition: dict[int, float] = {}
        for item in dataset.get("Constraint", ()):
            identifier = item["ConstraintID"]
            constraint_type = identifier["ConstraintType"]
            key, name = _typed_value(constraint_type)
            if key == "eComponentComposition" and name == "Mole fraction":
                constrained_composition[int(identifier["RegNum"]["nOrgNum"])] = float(
                    item["nConstraintValue"]
                )

        for values in dataset["NumValues"]:
            temperature = pressure_kpa = None
            composition = constrained_composition.copy()
            for item in values["VariableValue"]:
                name, reg_num = variables[int(item["nVarNumber"])]
                value = float(item["nVarValue"])
                if name == "Temperature, K":
                    temperature = value
                elif name == "Pressure, kPa":
                    pressure_kpa = value
                elif name == "Mole fraction" and reg_num is not None:
                    composition[reg_num] = value
            if temperature is None or pressure_kpa is None:
                raise ValueError("density record is missing temperature or pressure")
            if first_number not in composition:
                other_number = component_numbers[1]
                composition[first_number] = 1.0 - composition[other_number]

            for item in values["PropertyValue"]:
                number = int(item["nPropNumber"])
                if number not in density_numbers:
                    continue
                _, method, phase = properties[number]
                uncertainty = item.get("CombinedUncertainty", {}).get(
                    "nCombExpandUncertValue",
                    "",
                )
                records.append(
                    {
                        "doi": citation["sDOI"],
                        "component1": compounds[first_number],
                        "component2": compounds[component_numbers[1]],
                        "temperature_K": temperature,
                        "pressure_Pa": pressure_kpa * 1000.0,
                        "mole_fraction_component1": composition[first_number],
                        "density_kg_m3": float(item["nPropValue"]),
                        "expanded_uncertainty_density_kg_m3": uncertainty,
                        "phase": phase,
                        "method": method,
                    }
                )
    return records


def main() -> int:
    """Emit normalized mass-density rows as CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file", type=Path)
    args = parser.parse_args()
    document = json.loads(args.json_file.read_text(encoding="utf-8"))
    records = density_records(document)
    if not records:
        raise ValueError("no binary mass-density records were found")
    writer = csv.DictWriter(sys.stdout, fieldnames=tuple(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
