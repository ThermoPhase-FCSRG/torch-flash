"""Publish selected plot outputs from the executed validation notebooks.

The local-only notebooks and their research inputs are intentionally excluded
from Git and package distributions.  This script copies only selected PNG
outputs into the documentation and records the source notebook, cell, data
dependencies, and publication form in a machine-auditable manifest.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = REPO_ROOT / "docs" / "assets" / "validation"
MANIFEST_PATH = OUTPUT_DIRECTORY / "manifest.yaml"


@dataclass(frozen=True)
class FigureSpec:
    """Describe one notebook plot selected for public documentation."""

    filename: str
    study_id: str
    notebook: str
    cell_index: int
    image_index: int
    evidence_class: str
    observation_level_markers: bool
    data_artifacts: tuple[str, ...]
    source_material: tuple[str, ...]
    description: str


FIGURES = (
    FigureSpec(
        "05_gerg_pedersen_validation.png",
        "05_gerg_eoscg_ccs",
        "notebooks/local-only/validation/05_gerg_eoscg_ccs.ipynb",
        5,
        0,
        "validation",
        True,
        ("pedersen_2024_gerg_z.csv",),
        ("https://doi.org/10.1201/9780429457418",),
        "Measured and GERG-2008 compressibility factors for the Pedersen mixture.",
    ),
    FigureSpec(
        "05_eoscg_table8_verification.png",
        "05_gerg_eoscg_ccs",
        "notebooks/local-only/validation/05_gerg_eoscg_ccs.ipynb",
        10,
        0,
        "verification",
        True,
        ("gernert_span_2016_eoscg_table8.csv",),
        ("https://doi.org/10.1016/j.jct.2015.05.015",),
        "EOS-CG parity and residual plots for the published Table 8 verification states.",
    ),
    FigureSpec(
        "08_eoscg2021_co2_h2_verification.png",
        "08_eoscg2021_native_validation",
        "notebooks/validation/08_eoscg2021_native_validation.ipynb",
        5,
        0,
        "verification",
        True,
        ("eoscg2021_co2_h2_teqp_reference.csv",),
        ("https://doi.org/10.1007/s10765-023-03263-6",),
        "Native EOS-CG-2021 parity against an independently generated teqp reference.",
    ),
    FigureSpec(
        "08_eoscg2021_mdea_density.png",
        "08_eoscg2021_native_validation",
        "notebooks/validation/08_eoscg2021_native_validation.ipynb",
        7,
        0,
        "validation",
        True,
        ("eoscg2021_mdea_density_experimental.csv",),
        ("https://doi.org/10.1007/s10765-021-02933-7",),
        "Pure-MDEA liquid-density predictions against CC BY 4.0 measurements.",
    ),
    FigureSpec(
        "08_eoscg2021_mdea_speed_of_sound.png",
        "08_eoscg2021_native_validation",
        "notebooks/validation/08_eoscg2021_native_validation.ipynb",
        9,
        0,
        "validation",
        True,
        ("eoscg2021_mdea_speed_of_sound_experimental.csv",),
        ("https://doi.org/10.1007/s10765-021-02933-7",),
        "Pure-MDEA speed-of-sound predictions against CC BY 4.0 measurements.",
    ),
    FigureSpec(
        "15_thermal_co2_comparison.png",
        "15_pedersen_chapter_8_thermal_properties",
        "notebooks/local-only/validation/15_pedersen_chapter_8_thermal_properties.ipynb",
        4,
        0,
        "validation",
        True,
        (),
        (
            "https://doi.org/10.1201/9780429457418",
            "https://doi.org/10.1016/j.jcou.2017.04.007",
        ),
        "CO2 Joule-Thomson predictions against the complete reported comparison.",
    ),
    FigureSpec(
        "15_thermal_co2_error.png",
        "15_pedersen_chapter_8_thermal_properties",
        "notebooks/local-only/validation/15_pedersen_chapter_8_thermal_properties.ipynb",
        5,
        0,
        "validation",
        True,
        (),
        (
            "https://doi.org/10.1201/9780429457418",
            "https://doi.org/10.1016/j.jcou.2017.04.007",
        ),
        "CO2 Joule-Thomson parity and signed-error structure.",
    ),
    FigureSpec(
        "15_thermal_propane_comparison.png",
        "15_pedersen_chapter_8_thermal_properties",
        "notebooks/local-only/validation/15_pedersen_chapter_8_thermal_properties.ipynb",
        8,
        0,
        "validation",
        True,
        (),
        (
            "https://doi.org/10.1201/9780429457418",
            "https://doi.org/10.1021/ie50317a026",
        ),
        "Propane Joule-Thomson predictions over all reported vapor-state isotherms.",
    ),
    FigureSpec(
        "15_thermal_property_profile.png",
        "15_pedersen_chapter_8_thermal_properties",
        "notebooks/local-only/validation/15_pedersen_chapter_8_thermal_properties.ipynb",
        10,
        0,
        "application",
        False,
        (),
        ("https://doi.org/10.1201/9780429457418",),
        "Autodifferentiated thermal-property profiles at supplied homogeneous states.",
    ),
    FigureSpec(
        "22_hv_full_pxy.png",
        "22_huron_vidal_alcohol_hydrocarbon",
        "notebooks/local-only/validation/22_huron_vidal_alcohol_hydrocarbon.ipynb",
        11,
        0,
        "validation",
        True,
        ("jaubert_2020_hv_bac5_vle.csv",),
        ("https://doi.org/10.1021/acs.iecr.0c01734",),
        "Full experimental and modeled P-x-y behavior for the twelve BAC-5 isotherms.",
    ),
    FigureSpec(
        "22_hv_parity_transfer.png",
        "22_huron_vidal_alcohol_hydrocarbon",
        "notebooks/local-only/validation/22_huron_vidal_alcohol_hydrocarbon.ipynb",
        13,
        0,
        "validation",
        True,
        ("jaubert_2020_hv_bac5_vle.csv",),
        ("https://doi.org/10.1021/acs.iecr.0c01734",),
        "Pressure and vapor-composition parity with complete-temperature holdouts.",
    ),
    FigureSpec(
        "22_hv_residuals.png",
        "22_huron_vidal_alcohol_hydrocarbon",
        "notebooks/local-only/validation/22_huron_vidal_alcohol_hydrocarbon.ipynb",
        14,
        0,
        "validation",
        True,
        ("jaubert_2020_hv_bac5_vle.csv",),
        ("https://doi.org/10.1021/acs.iecr.0c01734",),
        "Temperature-resolved pressure and vapor-composition residuals.",
    ),
    FigureSpec(
        "24_volume_translation_density_curves.png",
        "24_cubic_volume_translation_pedersen_whitson",
        "notebooks/local-only/validation/24_cubic_volume_translation_pedersen_whitson.ipynb",
        15,
        0,
        "validation",
        True,
        ("segovia_2017_methane_n_decane_density.csv",),
        ("https://doi.org/10.1016/j.jct.2017.01.022",),
        "Methane-n-decane density curves with untranslated and translated cubic models.",
    ),
    FigureSpec(
        "24_volume_translation_isotherm.png",
        "24_cubic_volume_translation_pedersen_whitson",
        "notebooks/local-only/validation/24_cubic_volume_translation_pedersen_whitson.ipynb",
        16,
        0,
        "validation",
        True,
        ("segovia_2017_methane_n_decane_density.csv",),
        ("https://doi.org/10.1016/j.jct.2017.01.022",),
        "Translated-model comparison along the experimental methane-n-decane isotherm.",
    ),
    FigureSpec(
        "24_volume_translation_parity.png",
        "24_cubic_volume_translation_pedersen_whitson",
        "notebooks/local-only/validation/24_cubic_volume_translation_pedersen_whitson.ipynb",
        17,
        0,
        "validation",
        True,
        ("segovia_2017_methane_n_decane_density.csv",),
        ("https://doi.org/10.1016/j.jct.2017.01.022",),
        "Density parity and signed residuals for the translated cubic models.",
    ),
    FigureSpec(
        "25_covolume_density_curves.png",
        "25_cubic_covolume_interaction",
        "notebooks/local-only/validation/25_cubic_covolume_interaction.ipynb",
        15,
        0,
        "validation",
        True,
        ("segovia_2017_methane_n_decane_density.csv",),
        ("https://doi.org/10.1016/j.jct.2017.01.022",),
        "PR78 density curves before and after fitting the cross-co-volume interaction.",
    ),
    FigureSpec(
        "25_covolume_density_parity.png",
        "25_cubic_covolume_interaction",
        "notebooks/local-only/validation/25_cubic_covolume_interaction.ipynb",
        16,
        0,
        "validation",
        True,
        ("segovia_2017_methane_n_decane_density.csv",),
        ("https://doi.org/10.1016/j.jct.2017.01.022",),
        "Density parity for the estimation isobar and disjoint validation isotherm.",
    ),
    FigureSpec(
        "25_covolume_validation_residuals.png",
        "25_cubic_covolume_interaction",
        "notebooks/local-only/validation/25_cubic_covolume_interaction.ipynb",
        17,
        0,
        "validation",
        True,
        ("segovia_2017_methane_n_decane_density.csv",),
        ("https://doi.org/10.1016/j.jct.2017.01.022",),
        "Pressure-resolved density residuals on the disjoint validation isotherm.",
    ),
    FigureSpec(
        "25_covolume_parameter_sensitivity.png",
        "25_cubic_covolume_interaction",
        "notebooks/local-only/validation/25_cubic_covolume_interaction.ipynb",
        19,
        0,
        "application",
        False,
        (),
        ("https://doi.org/10.1016/j.fluid.2022.113697",),
        "Training and validation sensitivity to the cross-co-volume parameter.",
    ),
    FigureSpec(
        "26_unifac_ethanol_heptane_pxy.png",
        "26_unifac_activity_validation",
        "notebooks/local-only/validation/26_unifac_activity_validation.ipynb",
        17,
        0,
        "validation",
        True,
        ("jaubert_2020_hv_bac5_vle.csv",),
        ("https://doi.org/10.1021/acs.iecr.0c01734",),
        "Predictive original-UNIFAC P-x-y curves for ethanol-n-heptane.",
    ),
    FigureSpec(
        "26_unifac_methanol_benzene_pxy.png",
        "26_unifac_activity_validation",
        "notebooks/local-only/validation/26_unifac_activity_validation.ipynb",
        17,
        1,
        "validation",
        True,
        ("jaubert_2020_hv_bac5_vle.csv",),
        ("https://doi.org/10.1021/acs.iecr.0c01734",),
        "Predictive original-UNIFAC P-x-y curves for methanol-benzene.",
    ),
    FigureSpec(
        "26_unifac_parity_residuals.png",
        "26_unifac_activity_validation",
        "notebooks/local-only/validation/26_unifac_activity_validation.ipynb",
        20,
        0,
        "validation",
        True,
        ("jaubert_2020_hv_bac5_vle.csv",),
        ("https://doi.org/10.1021/acs.iecr.0c01734",),
        "Original-UNIFAC parity and residual structure with extrapolative states marked.",
    ),
    FigureSpec(
        "26_unifac_activity_surfaces.png",
        "26_unifac_activity_validation",
        "notebooks/local-only/validation/26_unifac_activity_validation.ipynb",
        22,
        0,
        "application",
        False,
        (),
        (
            "https://doi.org/10.1002/aic.690210607",
            "https://doi.org/10.1021/ie00058a017",
        ),
        "Predicted activity-coefficient surfaces over composition and temperature.",
    ),
    FigureSpec(
        "28_ppr78_full_pxy.png",
        "28_ppr78_hydrocarbon_vle",
        "notebooks/local-only/validation/28_ppr78_hydrocarbon_vle.ipynb",
        7,
        0,
        "validation",
        True,
        ("jaubert_ppr78_hydrocarbon_vle.csv",),
        ("https://doi.org/10.1021/acs.iecr.0c01734",),
        "Experimental and predicted P-x-y behavior over six hydrocarbon isotherms.",
    ),
    FigureSpec(
        "28_ppr78_parity.png",
        "28_ppr78_hydrocarbon_vle",
        "notebooks/local-only/validation/28_ppr78_hydrocarbon_vle.ipynb",
        9,
        0,
        "validation",
        True,
        ("jaubert_ppr78_hydrocarbon_vle.csv",),
        ("https://doi.org/10.1021/acs.iecr.0c01734",),
        "Pressure and vapor-composition parity for zero-BIP PR78 and predictive PPR78.",
    ),
    FigureSpec(
        "28_ppr78_interactions.png",
        "28_ppr78_hydrocarbon_vle",
        "notebooks/local-only/validation/28_ppr78_hydrocarbon_vle.ipynb",
        11,
        0,
        "application",
        False,
        (),
        ("https://doi.org/10.1016/j.fluid.2004.06.059",),
        "Temperature-dependent PPR78 binary interactions from the group contribution.",
    ),
    FigureSpec(
        "29_phase_identification_north_ward_estes.png",
        "29_bennett_phase_identification_methods",
        "notebooks/verification/29_bennett_phase_identification_methods.ipynb",
        17,
        2,
        "verification",
        False,
        (),
        (
            "https://doi.org/10.1021/acs.energyfuels.6b02316",
            "https://doi.org/10.1016/j.fluid.2010.12.001",
            "https://doi.org/10.2118/129844-PA",
        ),
        "Six physical phase-identification diagnostics applied after equilibrium "
        "flashes on the North Ward Estes 500 by 500 injection grid.",
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_payloads(cell: dict[str, Any]) -> list[bytes]:
    payloads: list[bytes] = []
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        encoded = data.get("image/png")
        if encoded is None:
            continue
        if isinstance(encoded, list):
            encoded = "".join(encoded)
        payloads.append(base64.b64decode(encoded))
    return payloads


def _extract(spec: FigureSpec) -> tuple[str, dict[str, Any]]:
    notebook_path = REPO_ROOT / spec.notebook
    if not notebook_path.is_file():
        raise FileNotFoundError(
            f"{spec.notebook} is required; restore the paired local-only notebook"
        )
    document = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = document.get("cells")
    if not isinstance(cells, list) or spec.cell_index >= len(cells):
        raise RuntimeError(f"{spec.notebook}: missing cell {spec.cell_index}")
    cell = cells[spec.cell_index]
    images = _image_payloads(cell)
    if spec.image_index >= len(images):
        raise RuntimeError(
            f"{spec.notebook}: cell {spec.cell_index} has {len(images)} PNG outputs, "
            f"not image {spec.image_index}"
        )
    payload = images[spec.image_index]
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"{spec.notebook}: selected output is not a PNG")
    (OUTPUT_DIRECTORY / spec.filename).write_bytes(payload)

    record = asdict(spec)
    record.pop("filename")
    record["executed_artifact_sha256"] = _sha256(notebook_path)
    record["publication_form"] = "rendered-plot"
    record["machine_readable_observation_values"] = False
    record["data_tables_distributed"] = False
    return spec.filename, record


def main() -> int:
    """Extract all declared figures and write their provenance manifest."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    records = dict(_extract(spec) for spec in FIGURES)
    manifest = {
        "schema_version": 2,
        "policy": "docs/licensing.md",
        "publication_boundary": {
            "public_outputs": "selected rendered plots only",
            "machine_readable_observation_values": False,
            "data_tables_distributed": False,
            "local_sources": "notebooks/local-only and tests/data/not-cleared",
        },
        "figures": records,
    }
    MANIFEST_PATH.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Generated {len(records)} validation-report figures in {OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
