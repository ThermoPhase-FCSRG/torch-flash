"""Digitize the plotted markers in Xu et al. (2017), Figure 2.

The source is Xu, Lasala, Privat, and Jaubert, *International Journal of
Greenhouse Gas Control* 56 (2017) 126-154,
doi:10.1016/j.ijggc.2016.11.015. The script operates on a lawful
user-supplied PDF; it does not download or redistribute the article.

Figure 2 is vector artwork. Poppler converts page 10 to SVG, after which this
script reads the centers of the colored ``+``, ``*``, and ``x`` marker paths.
Axes are calibrated from their vector borders and printed limits. This is a
visual digitization of the plotted symbols, not recovery of the authors'
underlying experimental database. Conservative coordinate uncertainties are
written beside every value.

Usage
-----
```
python scripts/extract_xu_2017_eppr78_figure2.py ARTICLE.pdf OUTPUT.csv
```
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DOI = "10.1016/j.ijggc.2016.11.015"
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"
FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")
MATRIX_PATTERN = re.compile(r"matrix\(([^)]*)\)")
EXPECTED_OBSERVATION_COUNTS = {
    ("a", "bubble"): 34,
    ("a", "dew"): 34,
    ("b", "bubble"): 28,
    ("b", "dew"): 28,
    ("c", "bubble"): 147,
    ("c", "dew"): 145,
    ("d", "mixing_enthalpy"): 12,
    ("e", "mixing_enthalpy"): 20,
    ("f", "mixing_enthalpy"): 20,
    ("g", "mixing_enthalpy"): 20,
    ("h", "bubble"): 35,
    ("h", "dew"): 22,
}

COLORS = {
    "rgb(100%, 0%, 0%)": "red",
    "rgb(0%, 100%, 0%)": "green",
    "rgb(0%, 0%, 100%)": "blue",
    "rgb(0%, 100%, 100%)": "cyan",
    "rgb(100%, 0%, 100%)": "magenta",
    "rgb(50.195312%, 0%, 50.195312%)": "purple",
}


@dataclass(frozen=True)
class Panel:
    """Source-specific Figure 2 panel calibration and series metadata."""

    label: str
    component2: str
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float
    y_min: float
    y_max: float
    ordinate: str
    series: dict[str, tuple[float, float]]
    ordinate_uncertainty: float

    def contains(self, x_coordinate: float, y_coordinate: float) -> bool:
        """Return whether one SVG coordinate is inside the plotting frame."""
        return (
            self.x_left < x_coordinate < self.x_right and self.y_top < y_coordinate < self.y_bottom
        )

    def fraction(self, x_coordinate: float) -> float:
        """Map an SVG abscissa to the printed zero-to-one composition axis."""
        return (x_coordinate - self.x_left) / (self.x_right - self.x_left)

    def ordinate_value(self, y_coordinate: float) -> float:
        """Map an SVG ordinate to the printed linear property axis."""
        fraction = (self.y_bottom - y_coordinate) / (self.y_bottom - self.y_top)
        return self.y_min + fraction * (self.y_max - self.y_min)


PANELS = {
    "a": Panel(
        "a",
        "methane",
        147.4609,
        295.0508,
        60.2578,
        178.3711,
        75.0,
        175.0,
        "temperature",
        {
            "red": (float("nan"), 1.01),
            "green": (float("nan"), 4.90),
            "blue": (float("nan"), 10.34),
            "cyan": (float("nan"), 17.24),
            "magenta": (float("nan"), 27.58),
        },
        0.25,
    ),
    "b": Panel(
        "b",
        "methane",
        327.6641,
        475.2500,
        60.2578,
        178.3711,
        115.0,
        200.0,
        "temperature",
        {
            "red": (float("nan"), 24.13),
            "green": (float("nan"), 31.03),
            "blue": (float("nan"), 37.92),
            "cyan": (float("nan"), 44.82),
            "magenta": (float("nan"), 48.26),
        },
        0.25,
    ),
    "c": Panel(
        "c",
        "methane",
        147.4609,
        295.0508,
        205.3789,
        323.3008,
        0.0,
        60.0,
        "pressure",
        {
            "red": (130.0, float("nan")),
            "green": (140.0, float("nan")),
            "blue": (150.0, float("nan")),
            "cyan": (160.0, float("nan")),
            "magenta": (170.0, float("nan")),
            "purple": (180.0, float("nan")),
        },
        0.15,
    ),
    "d": Panel(
        "d",
        "methane",
        327.6641,
        475.2500,
        205.3789,
        323.3008,
        0.0,
        150.0,
        "enthalpy",
        {"red": (91.50, 8.22), "green": (105.00, 21.03)},
        0.5,
    ),
    "e": Panel(
        "e",
        "methane",
        147.4609,
        295.0508,
        350.5000,
        468.4219,
        0.0,
        1600.0,
        "enthalpy",
        {
            "red": (195.15, 20.27),
            "green": (195.15, 40.53),
            "blue": (195.15, 60.80),
            "cyan": (195.15, 81.06),
            "magenta": (195.15, 101.33),
        },
        5.0,
    ),
    "f": Panel(
        "f",
        "methane",
        327.6641,
        475.2500,
        350.5000,
        468.4219,
        0.0,
        300.0,
        "enthalpy",
        {
            "red": (253.15, 20.27),
            "green": (253.15, 40.53),
            "blue": (253.15, 60.80),
            "cyan": (253.15, 81.06),
            "magenta": (253.15, 101.33),
        },
        1.0,
    ),
    "g": Panel(
        "g",
        "methane",
        147.4609,
        295.0508,
        495.4023,
        613.5195,
        0.0,
        110.0,
        "enthalpy",
        {
            "red": (313.15, 20.27),
            "green": (313.15, 40.53),
            "blue": (313.15, 60.80),
            "cyan": (313.15, 81.06),
            "magenta": (313.15, 101.33),
        },
        0.5,
    ),
    "h": Panel(
        "h",
        "carbon_monoxide",
        327.6641,
        475.2500,
        495.4023,
        613.5195,
        0.0,
        2.1,
        "pressure",
        {
            "red": (68.90, float("nan")),
            "green": (75.00, float("nan")),
            "blue": (83.82, float("nan")),
        },
        0.008,
    ),
}


@dataclass
class Marker:
    """Cluster of SVG line segments forming one plotted marker."""

    panel: str
    color: str
    x_coordinate: float
    y_coordinate: float
    segments: int = 1

    def add(self, x_coordinate: float, y_coordinate: float) -> None:
        """Add one coincident segment midpoint to the marker centroid."""
        self.x_coordinate = (self.x_coordinate * self.segments + x_coordinate) / (self.segments + 1)
        self.y_coordinate = (self.y_coordinate * self.segments + y_coordinate) / (self.segments + 1)
        self.segments += 1


def _segment_midpoints(svg: Path) -> list[tuple[str, str, float, float]]:
    """Return panel, color, and midpoint for every short colored SVG segment."""
    root = ET.parse(svg).getroot()
    midpoints: list[tuple[str, str, float, float]] = []
    for element in root.iter(f"{SVG_NAMESPACE}path"):
        stroke = element.get("stroke")
        if stroke not in COLORS:
            continue
        path_data = element.get("d", "")
        if path_data.count("M") != 1 or path_data.count("L") != 1:
            continue
        coordinates = [float(value) for value in FLOAT_PATTERN.findall(path_data)]
        matrix_match = MATRIX_PATTERN.search(element.get("transform", ""))
        if len(coordinates) < 4 or matrix_match is None:
            continue
        matrix = [float(value) for value in FLOAT_PATTERN.findall(matrix_match.group(1))]
        if len(matrix) != 6:
            continue
        x1, y1, x2, y2 = coordinates[:4]
        a, b, c, d, e, f = matrix
        local_x = 0.5 * (x1 + x2)
        local_y = 0.5 * (y1 + y2)
        x_coordinate = a * local_x + c * local_y + e
        y_coordinate = b * local_x + d * local_y + f
        for panel in PANELS.values():
            if panel.contains(x_coordinate, y_coordinate) and COLORS[stroke] in panel.series:
                midpoints.append((panel.label, COLORS[stroke], x_coordinate, y_coordinate))
                break
    return midpoints


def _cluster_markers(svg: Path) -> list[Marker]:
    """Cluster coincident marker-segment midpoints with a vector-scale tolerance."""
    markers: list[Marker] = []
    for panel, color, x_coordinate, y_coordinate in _segment_midpoints(svg):
        for marker in markers:
            distance_squared = (marker.x_coordinate - x_coordinate) ** 2 + (
                marker.y_coordinate - y_coordinate
            ) ** 2
            if marker.panel == panel and marker.color == color and distance_squared < 0.12**2:
                marker.add(x_coordinate, y_coordinate)
                break
        else:
            markers.append(Marker(panel, color, x_coordinate, y_coordinate))
    return markers


def _observation_types(panel: Panel, segments: int) -> tuple[str, ...]:
    """Interpret marker geometry as mixing enthalpy, bubble, dew, or both."""
    if panel.ordinate == "enthalpy":
        return ("mixing_enthalpy",)
    if segments >= 6:
        return ("bubble", "dew")
    if segments >= 4:
        return ("dew",)
    return ("bubble",)


def extract(pdf: Path, output: Path) -> int:
    """Extract Figure 2 markers from ``pdf`` and write normalized ``output``."""
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="torch-flash-xu-2017-") as temporary:
        svg = Path(temporary) / "figure2-page.svg"
        subprocess.run(
            [
                "pdftocairo",
                "-f",
                "10",
                "-l",
                "10",
                "-svg",
                str(pdf),
                str(svg),
            ],
            check=True,
        )
        markers = _cluster_markers(svg)

    rows: list[dict[str, object]] = []
    for marker in markers:
        panel = PANELS[marker.panel]
        temperature, pressure = panel.series[marker.color]
        ordinate = panel.ordinate_value(marker.y_coordinate)
        if panel.ordinate == "temperature":
            temperature = ordinate
        elif panel.ordinate == "pressure":
            pressure = ordinate
        for observation_type in _observation_types(panel, marker.segments):
            rows.append(
                {
                    "panel": panel.label,
                    "component1": "nitrogen",
                    "component2": panel.component2,
                    "observation_type": observation_type,
                    "temperature_K": round(temperature, 4),
                    "pressure_bar": round(pressure, 5),
                    "first_component_fraction": round(panel.fraction(marker.x_coordinate), 6),
                    "enthalpy_of_mixing_J_mol": (
                        round(ordinate, 4) if panel.ordinate == "enthalpy" else ""
                    ),
                    "fraction_uncertainty": 0.002,
                    "temperature_uncertainty_K": (
                        panel.ordinate_uncertainty if panel.ordinate == "temperature" else 0.0
                    ),
                    "pressure_uncertainty_bar": (
                        panel.ordinate_uncertainty if panel.ordinate == "pressure" else 0.0
                    ),
                    "enthalpy_uncertainty_J_mol": (
                        panel.ordinate_uncertainty if panel.ordinate == "enthalpy" else 0.0
                    ),
                    "source_figure": "Figure 2",
                    "source_doi": DOI,
                    "extraction_method": "vector-marker visual digitization",
                }
            )

    rows.sort(
        key=lambda row: (
            str(row["panel"]),
            float(row["temperature_K"]),
            float(row["pressure_bar"]),
            str(row["observation_type"]),
            float(row["first_component_fraction"]),
        )
    )
    expected_panels = set(PANELS)
    found_panels = {str(row["panel"]) for row in rows}
    if found_panels != expected_panels:
        raise RuntimeError(
            f"expected Figure 2 panels {sorted(expected_panels)}, found {sorted(found_panels)}"
        )
    observation_counts = Counter((str(row["panel"]), str(row["observation_type"])) for row in rows)
    if observation_counts != EXPECTED_OBSERVATION_COUNTS:
        raise RuntimeError(
            "Figure 2 marker geometry does not match the audited panel counts: "
            f"expected {EXPECTED_OBSERVATION_COUNTS}, found {dict(observation_counts)}"
        )

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    """Parse command-line arguments and run the Figure 2 extraction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="user-supplied Xu et al. article PDF")
    parser.add_argument("output", type=Path, help="local output CSV")
    arguments = parser.parse_args()
    count = extract(arguments.pdf, arguments.output)
    print(f"wrote {count} digitized Figure 2 observations to {arguments.output}")


if __name__ == "__main__":
    main()
