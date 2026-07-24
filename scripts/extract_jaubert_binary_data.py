"""Read the Jaubert et al. binary-system benchmark workbook without Excel.

The source workbook is Supporting Information file ``ie0c01734_si_001.xlsx``
for Jaubert et al., *Industrial & Engineering Chemistry Research* 59 (2020),
14981-15027, doi:10.1021/acs.iecr.0c01734.

The workbook remains an optional, untracked source artifact. This module uses
only the Python standard library so contributors can audit or regenerate
selected CSV regression subsets without adding an Excel runtime dependency.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_COLUMN_RE = re.compile(r"([A-Z]+)")
_DOI_RE = re.compile(r"doi:(10\.\d{4,9}/\S+)", re.IGNORECASE)


def _column_index(reference: str) -> int:
    """Return a zero-based column index from an A1 cell reference."""
    match = _COLUMN_RE.match(reference)
    if match is None:
        raise ValueError(f"invalid cell reference {reference!r}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _numeric_value(text: str) -> int | float | str:
    """Parse an Excel numeric value while retaining nonnumeric cached text."""
    try:
        value = float(text)
    except ValueError:
        return text
    return int(value) if value.is_integer() else value


class XlsxValues:
    """Minimal read-only OOXML value reader."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._archive = zipfile.ZipFile(path)
        self._shared_strings = self._read_shared_strings()
        self._sheet_paths = self._read_sheet_paths()

    def __enter__(self) -> XlsxValues:
        return self

    def __exit__(self, *_: object) -> None:
        self._archive.close()

    @property
    def sheet_names(self) -> tuple[str, ...]:
        """Return workbook sheet names in workbook order."""
        return tuple(self._sheet_paths)

    def _read_shared_strings(self) -> tuple[str, ...]:
        try:
            root = ElementTree.fromstring(self._archive.read("xl/sharedStrings.xml"))
        except KeyError:
            return ()
        strings = []
        for item in root.findall(f"{{{_MAIN_NS}}}si"):
            strings.append("".join(node.text or "" for node in item.iter(f"{{{_MAIN_NS}}}t")))
        return tuple(strings)

    def _read_sheet_paths(self) -> dict[str, str]:
        workbook = ElementTree.fromstring(self._archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(self._archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relationships.findall(f"{{{_PKG_REL_NS}}}Relationship")
            if relation.attrib.get("Type", "").endswith("/worksheet")
        }
        paths: dict[str, str] = {}
        for sheet in workbook.findall(f".//{{{_MAIN_NS}}}sheet"):
            name = sheet.attrib["name"]
            relationship_id = sheet.attrib[f"{{{_DOC_REL_NS}}}id"]
            target = PurePosixPath(targets[relationship_id])
            paths[name] = str(PurePosixPath("xl") / target)
        return paths

    def rows(self, sheet_name: str) -> list[list[int | float | str | None]]:
        """Return cached values from one worksheet as rectangular rows."""
        try:
            sheet_path = self._sheet_paths[sheet_name]
        except KeyError as exc:
            raise KeyError(f"worksheet {sheet_name!r} is not present") from exc
        root = ElementTree.fromstring(self._archive.read(sheet_path))
        rows: list[list[int | float | str | None]] = []
        for row_node in root.findall(f".//{{{_MAIN_NS}}}sheetData/{{{_MAIN_NS}}}row"):
            values: dict[int, int | float | str | None] = {}
            for cell in row_node.findall(f"{{{_MAIN_NS}}}c"):
                column = _column_index(cell.attrib["r"])
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    values[column] = "".join(
                        node.text or "" for node in cell.iter(f"{{{_MAIN_NS}}}t")
                    )
                    continue
                value_node = cell.find(f"{{{_MAIN_NS}}}v")
                if value_node is None or value_node.text is None:
                    values[column] = None
                    continue
                text = value_node.text
                if cell_type == "s":
                    values[column] = self._shared_strings[int(text)]
                elif cell_type == "b":
                    values[column] = bool(int(text))
                elif cell_type == "str":
                    values[column] = text
                else:
                    values[column] = _numeric_value(text)
            width = max(values, default=-1) + 1
            rows.append([values.get(column) for column in range(width)])
        return rows


@dataclass(frozen=True)
class IsothermalVLE:
    """One isothermal pressure-composition row from a binary worksheet."""

    sheet: str
    component1: str
    component2: str
    temperature_k: float
    pressure_bar: float
    x1: float
    y1: float
    flag: str
    doi: str
    citation: str


def isothermal_vle_records(
    rows: list[list[int | float | str | None]],
    sheet_name: str,
) -> list[IsothermalVLE]:
    """Normalize simple ``P, x1, y1`` VLE blocks from one worksheet.

    Partial-solubility and multiphase columns are intentionally excluded; the
    Jaubert benchmark protocol treats those observables differently.
    """
    if not rows or len(rows[0]) < 3:
        raise ValueError(f"worksheet {sheet_name!r} has no binary-system header")
    component1 = str(rows[0][1]).strip().lower().replace(" ", "_")
    component2 = str(rows[0][2]).strip().lower().replace(" ", "_")
    citation = ""
    doi = ""
    temperature: float | None = None
    simple_block = False
    records: list[IsothermalVLE] = []
    for row in rows:
        first = "" if not row or row[0] is None else str(row[0]).strip()
        remaining_empty = all(value is None for value in row[1:])
        is_citation = (
            remaining_empty
            and "," in first
            and not first.startswith(("T / K", "P / bar", "Name", "CAS", "InChiKey"))
        )
        if is_citation:
            citation = first
            match = _DOI_RE.search(first)
            doi = "" if match is None else match.group(1).rstrip(".,;")
        if first.startswith("T / K") and len(row) > 1 and isinstance(row[1], int | float):
            temperature = float(row[1])
            simple_block = False
            continue
        if first == "P / bar":
            headings = tuple("" if value is None else str(value).strip() for value in row[:3])
            simple_block = headings == ("P / bar", "x₁", "y₁")
            continue
        if (
            simple_block
            and temperature is not None
            and len(row) >= 3
            and all(isinstance(value, int | float) for value in row[:3])
        ):
            flag = "" if len(row) < 4 or row[3] is None else str(row[3]).strip()
            records.append(
                IsothermalVLE(
                    sheet_name,
                    component1,
                    component2,
                    temperature,
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    flag,
                    doi,
                    citation,
                )
            )
        elif not first:
            simple_block = False
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list-sheets", action="store_true")
    action.add_argument("--sheet")
    parser.add_argument(
        "--isothermal-vle",
        action="store_true",
        help="normalize simple isothermal P-x-y blocks instead of raw cells",
    )
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main() -> int:
    """List worksheets or emit one worksheet as UTF-8 CSV."""
    args = _parse_args()
    with XlsxValues(args.workbook) as workbook:
        if args.list_sheets:
            sys.stdout.write("\n".join(workbook.sheet_names) + "\n")
            return 0
        rows = workbook.rows(args.sheet)
    if args.isothermal_vle:
        records = isothermal_vle_records(rows, args.sheet)
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=tuple(IsothermalVLE.__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(record.__dict__ for record in records)
        return 0
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    writer = csv.writer(sys.stdout)
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
