"""Extract complete experimental PPR78 hydrocarbon VLE isotherms.

The optional source workbook is the supporting information for Jaubert et al.
(2020), doi:10.1021/acs.iecr.0c01734. The selected methane/ethane and
methane/n-decane systems are both part of the 2004 PPR78 parameter database
and are highlighted in that paper's Figure 3. Only high-quality benchmark
isotherms at or near the temperatures in Figure 3 are retained.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from extract_jaubert_binary_data import XlsxValues, isothermal_vle_records

SELECTIONS = {
    "1_2": (199.93, 230.0),
    "1_56": (410.93, 477.59, 510.93, 563.25),
}
EXPECTED_ROWS = 103


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Write the reviewed experimental subset as a provenance-rich CSV."""
    args = _parse_args()
    records = []
    with XlsxValues(args.workbook) as workbook:
        for sheet, temperatures in SELECTIONS.items():
            sheet_records = isothermal_vle_records(workbook.rows(sheet), sheet)
            records.extend(
                record for record in sheet_records if record.temperature_k in temperatures
            )
    if len(records) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, extracted {len(records)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
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
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sheet": record.sheet,
                    "component1": record.component1.replace("-", "_"),
                    "component2": record.component2.replace("-", "_"),
                    "temperature_K": record.temperature_k,
                    "pressure_bar": record.pressure_bar,
                    "x1": record.x1,
                    "y1": record.y1,
                    "flag": record.flag,
                    "source_doi": record.doi,
                    "citation": record.citation,
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
