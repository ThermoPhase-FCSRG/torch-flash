"""Regenerate the frozen PR78 co-volume-interaction software baseline.

Run with ``pixi run -e benchmarks python
scripts/generate_thermopack_covolume_reference.py``. ThermoPack uses
one-based component indices and reports specific volume in m3/mol.
"""

from __future__ import annotations

import csv
from importlib.metadata import version
from pathlib import Path

from thermopack.cubic import PengRobinson78

TARGET = Path(__file__).parents[1] / "tests/data/thermopack_pr78_covolume.csv"
FIELDS = (
    "thermopack_version",
    "components",
    "critical_temperature_1_K",
    "critical_temperature_2_K",
    "critical_pressure_1_Pa",
    "critical_pressure_2_Pa",
    "acentric_factor_1",
    "acentric_factor_2",
    "molar_mass_1_kg_mol",
    "molar_mass_2_kg_mol",
    "k12",
    "l12",
    "temperature_K",
    "pressure_Pa",
    "x1",
    "phase",
    "molar_volume_m3_mol",
    "lnphi_1",
    "lnphi_2",
)


def main() -> None:
    """Generate independent volume and fugacity results with ThermoPack 2.2.3."""
    model = PengRobinson78("C1,NC10")
    constants = {
        "thermopack_version": version("thermopack"),
        "components": "C1,NC10",
        "critical_temperature_1_K": model.critical_temperature(1),
        "critical_temperature_2_K": model.critical_temperature(2),
        "critical_pressure_1_Pa": model.critical_pressure(1),
        "critical_pressure_2_Pa": model.critical_pressure(2),
        "acentric_factor_1": model.acentric_factor(1),
        "acentric_factor_2": model.acentric_factor(2),
        "molar_mass_1_kg_mol": model.compmoleweight(1) / 1000.0,
        "molar_mass_2_kg_mol": model.compmoleweight(2) / 1000.0,
        "k12": model.get_kij(1, 2),
    }
    states = (
        (350.0, 1.0e7, 0.60, "vapor"),
        (450.0, 2.0e7, 0.25, "liquid"),
    )
    rows: list[dict[str, str | float]] = []
    for l12 in (-0.05, 0.04, 0.10):
        model.set_lij(1, 2, l12)
        for temperature, pressure, x1, phase in states:
            composition = [x1, 1.0 - x1]
            phase_flag = model.VAPPH if phase == "vapor" else model.LIQPH
            volume = model.specific_volume(
                temperature,
                pressure,
                composition,
                phase_flag,
            )[0]
            log_phi = model.thermo(
                temperature,
                pressure,
                composition,
                phase_flag,
            )[0]
            rows.append(
                {
                    **constants,
                    "l12": l12,
                    "temperature_K": temperature,
                    "pressure_Pa": pressure,
                    "x1": x1,
                    "phase": phase,
                    "molar_volume_m3_mol": volume,
                    "lnphi_1": log_phi[0],
                    "lnphi_2": log_phi[1],
                }
            )
    with TARGET.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
