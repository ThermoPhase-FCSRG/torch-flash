"""Generate frozen ThermoPack SRK/HV-NRTL verification tables.

Run only from the ``benchmarks`` Pixi environment:

``pixi run -e benchmarks python scripts/generate_thermopack_hv_reference.py``.

ThermoPack is initialized with pseudo-components carrying the exact
``torch-flash`` critical constants, acentric factors, molar masses, and fitted
HV parameters. This removes database and convention differences from the
implementation comparison.
"""

from __future__ import annotations

import csv
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import torch
from thermopack.cubic import cubic

from torch_flash import activity_model, component_set

DATA = Path(__file__).parents[1] / "tests" / "data"
EXPERIMENT = DATA / "not-cleared" / "jaubert_2020_hv_bac5_vle.csv"
FLASH_TARGET = DATA / "thermopack_2_2_3_srk_hv_n_butane_water_flash.csv"
STATE_TARGET = DATA / "thermopack_2_2_3_srk_hv_n_butane_water_states.csv"
SYSTEM = ("n_butane", "water")
PARAMETER_SET = "activity.hv-nrtl-jaubert-2020-n-butane-water"


def thermopack_model():
    """Return ThermoPack configured with exactly matched model inputs."""
    components = component_set(SYSTEM)
    activity = activity_model(PARAMETER_SET, SYSTEM)
    model = cubic("PSEUDO,PSEUDO", "SRK", mixing="HV", alpha="Classic")
    model.init_pseudo(
        "NC4,H2O",
        components.critical_temperature.tolist(),
        components.critical_pressure.tolist(),
        components.acentric_factor.tolist(),
        (1000.0 * components.molar_mass).tolist(),
        mixing="HV",
        alpha="Classic",
    )
    energy = activity.energy_over_r.detach().cpu().numpy()
    coefficient = activity.temperature_coefficient.detach().cpu().numpy()
    nonrandomness = activity.nonrandomness.detach().cpu().numpy()
    model.set_hv_param(
        1,
        2,
        nonrandomness[0, 1],
        nonrandomness[1, 0],
        energy[0, 1],
        energy[1, 0],
        coefficient[0, 1],
        coefficient[1, 0],
        0.0,
        0.0,
    )
    return model


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a deterministic CSV from records with a shared schema."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_flash_reference(model) -> None:
    """Freeze all 70 positive-composition fixed-TP coexistence states."""
    data = pd.read_csv(EXPERIMENT)
    data = data[
        (data["component1"] == SYSTEM[0])
        & (data["component2"] == SYSTEM[1])
        & (data["x1"] > 0.0)
        & (data["y1"] > 0.0)
    ]
    rows: list[dict[str, object]] = []
    package_version = version("thermopack")
    for row in data.itertuples(index=False):
        overall_first = 0.5 * (row.x1 + row.y1)
        result = model.two_phase_tpflash(
            row.temperature_K,
            row.pressure_bar * 1.0e5,
            [overall_first, 1.0 - overall_first],
        )
        if result.phase != model.TWOPH:
            raise RuntimeError(
                f"ThermoPack did not find two phases at {row.temperature_K} K, "
                f"{row.pressure_bar} bar"
            )
        rows.append(
            {
                "thermopack_version": package_version,
                "parameter_set": PARAMETER_SET,
                "temperature_K": f"{row.temperature_K:.12g}",
                "pressure_Pa": f"{row.pressure_bar * 1.0e5:.12g}",
                "overall_x1": f"{overall_first:.17g}",
                "liquid_x1": f"{result.x[0]:.17g}",
                "vapor_y1": f"{result.y[0]:.17g}",
                "beta_vapor": f"{result.betaV:.17g}",
            }
        )
    if len(rows) != 70:
        raise RuntimeError(f"expected 70 positive experimental states, found {len(rows)}")
    write_rows(FLASH_TARGET, rows)


def generate_homogeneous_reference(model) -> None:
    """Freeze a broad homogeneous-state fugacity grid for both root choices."""
    package_version = version("thermopack")
    states = (
        (310.93, 1.0e5, 0.0001),
        (310.93, 3.0e5, 0.50),
        (344.26, 8.0e5, 0.9990),
        (377.59, 1.5e6, 0.01),
        (410.93, 2.0e6, 0.001),
        (410.93, 5.0e6, 0.80),
        (444.26, 1.0e7, 0.0004),
        (477.59, 3.0e7, 0.10),
        (510.93, 6.0e7, 0.0017),
        (510.93, 8.0e7, 0.90),
        (628.15, 2.55e7, 0.025),
        (637.15, 8.30e7, 0.318),
    )
    rows: list[dict[str, object]] = []
    for temperature, pressure, first in states:
        composition = [first, 1.0 - first]
        for phase, phase_id in (("liquid", model.LIQPH), ("vapor", model.VAPPH)):
            log_fugacity = model.thermo(
                temperature,
                pressure,
                composition,
                phase_id,
            )[0]
            volume = model.specific_volume(
                temperature,
                pressure,
                composition,
                phase_id,
            )[0]
            rows.append(
                {
                    "thermopack_version": package_version,
                    "parameter_set": PARAMETER_SET,
                    "temperature_K": f"{temperature:.12g}",
                    "pressure_Pa": f"{pressure:.12g}",
                    "x1": f"{first:.17g}",
                    "phase": phase,
                    "molar_volume_m3_mol": f"{volume:.17g}",
                    "lnphi_1": f"{log_fugacity[0]:.17g}",
                    "lnphi_2": f"{log_fugacity[1]:.17g}",
                }
            )
    write_rows(STATE_TARGET, rows)


def main() -> None:
    """Generate both frozen verification datasets."""
    torch.set_default_dtype(torch.float64)
    model = thermopack_model()
    generate_flash_reference(model)
    generate_homogeneous_reference(model)
    print(f"Wrote {FLASH_TARGET}")
    print(f"Wrote {STATE_TARGET}")


if __name__ == "__main__":
    main()
