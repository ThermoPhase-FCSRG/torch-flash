"""Generate an offline teqp H2/CO isothermal VLE reference.

This optional benchmark uses teqp's multifluid model factory and is not needed
at package runtime.  The trace starts from the pure-CO saturation state, as
recommended by the teqp VLE documentation:
https://pages.nist.gov/teqp-docs/en/latest/algorithms/VLE.html
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import teqp


def trace_isotherm(temperature: float) -> list[dict[str, float | str]]:
    """Trace one H2/CO isotherm from its subcritical pure-CO endpoint."""
    names = ["Hydrogen", "CarbonMonoxide"]
    model = teqp.build_multifluid_model(names, teqp.get_datapath())
    pure_co = teqp.build_multifluid_model(["CarbonMonoxide"], teqp.get_datapath())
    ancillary = pure_co.build_ancillaries()
    liquid_density, vapor_density = pure_co.pure_VLE_T(
        temperature,
        ancillary.rhoL(temperature),
        ancillary.rhoV(temperature),
        20,
    )
    liquid = np.array([0.0, liquid_density])
    vapor = np.array([0.0, vapor_density])
    options = teqp.TVLEOptions()
    options.p_termination = 5.0e7
    options.crit_termination = 1.0e-8
    options.calc_criticality = True
    trace = model.trace_VLE_isotherm_binary(temperature, liquid, vapor, options)

    records = []
    for point in trace:
        pressure_atm = float(point["pL / Pa"]) / 101_325.0
        if pressure_atm > 235.0:
            continue
        records.append(
            {
                "generator": f"teqp-{teqp.__version__}",
                "model": "multifluid-CoolProp-GERG-BIP",
                "temperature_K": temperature,
                "pressure_atm": pressure_atm,
                "x_hydrogen": float(point["xL_0 / mole frac."]),
                "y_hydrogen": float(point["xV_0 / mole frac."]),
            }
        )
    return records


def main() -> int:
    """Write both external-reference isotherms to the requested CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = [record for temperature in (73.2, 83.2) for record in trace_isotherm(temperature)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
