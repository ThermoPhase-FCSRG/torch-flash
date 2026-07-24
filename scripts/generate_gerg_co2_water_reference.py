"""Generate the frozen independent GERG-2008 CO2/H2O state audit."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from thermopack.multiparameter import multiparam

from torch_flash.backends import TeqpBackend

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "tests" / "data"
SOURCE = DATA / "not-cleared" / "jaubert_2020_co2_binary_vle.csv"
TARGET = DATA / "gerg2008_co2_water_hou_teqp_thermopack.csv"
FIELDS = (
    "temperature_K",
    "pressure_Pa",
    "phase",
    "x_co2",
    "molar_density_mol_m3",
    "ln_phi_co2",
    "ln_phi_water",
    "thermopack_max_abs_ln_fugacity_difference",
)


def main() -> None:
    """Write teqp values and the independent ThermoPack discrepancy."""
    torch.set_default_dtype(torch.float64)
    teqp_model = TeqpBackend.gerg2008(("carbon_dioxide", "water"))
    thermopack_model = multiparam("CO2,H2O", "GERG2008")
    with SOURCE.open(newline="") as stream:
        experimental = [
            row
            for row in csv.DictReader(stream)
            if row["component1"] == "carbon_dioxide" and row["component2"] == "water"
        ]

    records = []
    for row in experimental:
        temperature = torch.tensor(float(row["temperature_K"]))
        pressure = torch.tensor(float(row["pressure_bar"]) * 1.0e5)
        for phase, x_co2, thermopack_phase in (
            ("liquid", float(row["x1"]), thermopack_model.LIQPH),
            ("vapor", float(row["y1"]), thermopack_model.VAPPH),
        ):
            composition = torch.tensor([x_co2, 1.0 - x_co2])
            volume = teqp_model.molar_volume(
                temperature,
                pressure,
                composition,
                phase,
            )
            log_phi = teqp_model.log_fugacity_coefficients(
                temperature,
                pressure,
                composition,
                phase,
            )
            thermopack_volume = thermopack_model.specific_volume(
                temperature.item(),
                pressure.item(),
                composition.tolist(),
                thermopack_phase,
            )[0]
            thermopack_log_fugacity = thermopack_model.fugacity_tv(
                temperature.item(),
                thermopack_volume,
                composition.tolist(),
            )[0]
            teqp_log_fugacity = (torch.log(pressure * composition) + log_phi).detach().numpy()
            records.append(
                {
                    "temperature_K": temperature.item(),
                    "pressure_Pa": pressure.item(),
                    "phase": phase,
                    "x_co2": x_co2,
                    "molar_density_mol_m3": volume.reciprocal().item(),
                    "ln_phi_co2": log_phi[0].item(),
                    "ln_phi_water": log_phi[1].item(),
                    "thermopack_max_abs_ln_fugacity_difference": float(
                        np.max(np.abs(thermopack_log_fugacity - teqp_log_fugacity))
                    ),
                }
            )

    with TARGET.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows to {TARGET.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
