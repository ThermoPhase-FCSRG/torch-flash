"""Extract the reviewed Huron-Vidal validation subset from Jaubert et al.

The source is Supporting Information workbook ``ie0c01734_si_001.xlsx`` for
Jaubert et al., *Industrial & Engineering Chemistry Research* 59 (2020),
14981-15027, doi:10.1021/acs.iecr.0c01734.  The workbook is untracked; the
normalized CSV is the shipped, auditable notebook and regression input.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from extract_jaubert_binary_data import XlsxValues, isothermal_vle_records

_SYSTEMS = {
    "5_1921": ("n_butane", "water"),
    "1102_17": ("ethanol", "n_heptane"),
    "1101_501": ("methanol", "benzene"),
}

# Worksheet 1102_17 assigns 23 rows at 483.15, 508.15, and 523.15 K to
# Seo et al., doi:10.1021/je025604s.  That paper reports 2-propanol +
# n-hexane, not
# ethanol + n-heptane.  Retaining those rows would turn a source-identity
# conflict into purported experimental validation, so the normalized subset
# excludes them and guards the reviewed count below.
_EXCLUDED_SOURCE_IDENTITIES = {
    ("1102_17", "10.1021/je025604s"): 23,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Write the selected complete isothermal ``P, x1, y1`` tables."""
    args = _parse_args()
    fields = (
        "sheet",
        "component1",
        "component2",
        "temperature_K",
        "pressure_bar",
        "x1",
        "y1",
        "flag",
        "source_doi",
        "citation",
    )
    rows: list[dict[str, str | float]] = []
    excluded_counts = {source_identity: 0 for source_identity in _EXCLUDED_SOURCE_IDENTITIES}
    with XlsxValues(args.workbook) as workbook:
        for sheet, expected_components in _SYSTEMS.items():
            records = isothermal_vle_records(workbook.rows(sheet), sheet)
            if not records:
                raise ValueError(f"{sheet} contains no simple isothermal VLE rows")
            observed_components = (records[0].component1, records[0].component2)
            normalized_observed = tuple(name.replace("-", "_") for name in observed_components)
            if normalized_observed != expected_components:
                raise ValueError(
                    f"{sheet} contains {observed_components!r}, expected {expected_components!r}"
                )
            retained_records = []
            for record in records:
                source_identity = (record.sheet, record.doi)
                if source_identity in excluded_counts:
                    excluded_counts[source_identity] += 1
                else:
                    retained_records.append(record)
            rows.extend(
                {
                    "sheet": record.sheet,
                    "component1": expected_components[0],
                    "component2": expected_components[1],
                    "temperature_K": record.temperature_k,
                    "pressure_bar": record.pressure_bar,
                    "x1": record.x1,
                    "y1": record.y1,
                    "flag": record.flag,
                    "source_doi": record.doi,
                    "citation": record.citation,
                }
                for record in retained_records
            )

    if excluded_counts != _EXCLUDED_SOURCE_IDENTITIES:
        raise ValueError(
            "reviewed source-identity exclusions changed: "
            f"{excluded_counts!r} != {_EXCLUDED_SOURCE_IDENTITIES!r}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
