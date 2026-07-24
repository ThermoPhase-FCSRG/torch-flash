"""Reproduce the visual digitization of Yan et al. water-content figures.

This script is not part of CI because the copyright-controlled source figures
are user-supplied and are neither tracked nor distributed. The committed CSV
is the reviewable output. Marker centres were isolated from the solid-symbol
isotherms, checked visually, and recorded below in source-image pixels. Axis
calibration is linear in pressure and logarithmic in water mole fraction.

Reference: Yan, Kontogeorgis, and Stenby, Fluid Phase Equilibria 276
(2009) 75-85, doi:10.1016/j.fluid.2008.10.007, Figs. 3-4 and supplementary
Figs. 3-4.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DigitizedCurve:
    """One visually digitized experimental curve and its pixel calibration."""

    component: str
    temperature: float
    figure: str
    pressure_axis: tuple[float, float, float]
    log_fraction_axis: tuple[float, float]
    pixels: tuple[tuple[float, float], ...]


METHANE_X = (
    (142, 89),
    (149, 104),
    (156, 114),
    (163, 122),
    (172, 129),
    (181, 135),
    (199, 144),
    (217, 151),
    (235, 156),
    (253, 160),
    (270.7, 163.3),
    (307, 168.5),
    (342.5, 172.5),
    (378.5, 175.5),
    (414.5, 178),
    (450, 180),
    (486, 182),
)
METHANE_BLUE = (
    (142, 131),
    (149, 145),
    (156, 155),
    (163, 163),
    (172, 170),
    (181, 176),
    (199, 185),
    (217, 191),
    (235, 196),
    (252.8, 199.8),
    (270.8, 202.8),
    (306.3, 207.7),
    (342.5, 211),
    (378.5, 214),
    (414, 216),
    (449.5, 218),
    (486, 219),
)
METHANE_RED = (
    (134.5, 153.5),
    (141.5, 179),
    (148, 192),
    (155.7, 202.3),
    (163.3, 209.7),
    (172.3, 216.7),
    (181.3, 222.3),
    (199, 231),
    (217, 237),
    (235, 241),
    (253, 244),
    (271, 247),
    (307, 251),
    (343, 254.5),
    (378.5, 257),
    (414.5, 259),
    (450, 260),
    (486.3, 260.3),
)
METHANE_BLACK = (
    (134.5, 215.5),
    (142, 241),
    (149, 254),
    (156, 263),
    (163, 270),
    (172, 276.5),
    (181, 281.5),
    (199, 288.5),
    (217, 293.5),
    (235, 297),
    (253, 300),
    (271, 302.5),
    (307, 305.5),
    (343, 308),
    (378.5, 310.5),
    (414.5, 311.5),
    (450.5, 313),
    (486, 314),
)

PROPANE_MAGENTA = (
    (217.5, 213),
    (229, 214),
    (244, 214),
    (258.5, 215),
    (273, 215),
    (309, 216),
    (346, 217),
    (418.5, 218.5),
    (564, 220),
)
PROPANE_BLUE = (
    (186, 240),
    (200, 240),
    (215, 241),
    (229, 241),
    (244, 241),
    (258.5, 242),
    (273, 242),
    (309, 242),
    (346, 243),
    (419, 243),
    (564, 244),
)
PROPANE_RED = (
    (171.3, 267.3),
    (185.7, 267.3),
    (200.3, 267.7),
    (214.7, 267.7),
    (229.3, 268.3),
    (243.7, 268.3),
    (258.7, 268.3),
    (273, 268.5),
    (309.5, 268.5),
    (345.5, 269),
    (418.7, 269.3),
    (564, 269),
)
PROPANE_BLACK = (
    (157, 300),
    (171, 300),
    (185.5, 300),
    (200.5, 300),
    (217.5, 300),
    (229, 300),
    (244, 300),
    (258.5, 300),
    (273, 300),
    (309, 300.5),
    (346, 300.5),
    (418.5, 300.5),
    (564, 300.5),
)

ETHANE_MAGENTA = (
    (223, 203),
    (240, 212),
    (257, 221),
    (290, 235),
    (324, 244.5),
    (358, 252),
    (392, 258),
    (426, 262),
    (459, 266),
    (493, 269),
    (561, 274),
    (628, 277),
    (696, 280),
    (763, 283),
    (831, 286),
)
ETHANE_BLUE = (
    (239.5, 274),
    (256, 282),
    (290.3, 295.7),
    (324.5, 303.5),
    (358, 311),
    (392, 316),
    (426, 320),
    (459.5, 324),
    (493, 326),
    (561, 330),
    (628, 333),
    (696, 335),
    (763, 338),
    (831, 340),
)
ETHANE_RED = (
    (221.5, 334.5),
    (239.7, 343.3),
    (256.5, 351.5),
    (290, 363),
    (324.3, 370.7),
    (358, 376),
    (392, 380),
    (425.5, 383),
    (459.5, 386),
    (493, 388),
    (561, 390),
    (628, 392),
    (696, 393),
    (763.3, 394.3),
    (831.3, 395.3),
)
ETHANE_BLACK = (
    (196, 398),
    (209.5, 412),
    (223, 421.5),
    (240, 430.5),
    (256.5, 437.5),
    (290.5, 447.5),
    (324, 452.5),
    (358, 456),
    (392, 459),
    (425.5, 460.5),
    (459.5, 461.5),
    (493, 462.5),
    (560.6, 462.8),
    (628, 464),
    (695.5, 462),
    (763, 461.5),
    (831, 461.5),
)

BUTANE_MAGENTA = (
    (182, 242),
    (183.5, 243),
    (187.5, 244),
    (198, 248),
    (209, 252),
    (222, 257),
    (236, 261),
    (249, 264),
    (263, 267),
    (289.5, 271),
    (316, 275),
    (370, 281),
    (423.5, 285),
    (477, 289),
    (531, 293),
    (584.5, 296),
    (638, 300),
    (692, 303),
)
BUTANE_BLUE = (
    (174.5, 314),
    (183, 317),
    (187, 317),
    (198.5, 320),
    (209, 320),
    (222, 322),
    (236, 324),
    (249.5, 326),
    (263, 327),
    (290, 331),
    (316, 334),
    (370, 340),
    (424, 345),
    (477, 349),
    (531, 351),
    (584.5, 353),
    (638, 355),
    (692, 357),
)
BUTANE_RED = (
    (166, 394),
    (171.5, 394),
    (178, 394),
    (181, 394),
    (187, 394),
    (198, 394),
    (209, 394),
    (236, 394),
    (263, 394),
    (316, 394),
    (370, 394),
    (423.5, 394),
    (477, 394),
    (585, 394),
    (692, 394),
)
BUTANE_BLACK = (
    (183.5, 475),
    (236, 475),
    (262.5, 475),
    (316, 475),
    (370, 475),
    (423.5, 475),
    (477, 475),
    (584.5, 474),
    (691.5, 475),
)

CURVES = (
    DigitizedCurve("methane", 410.93, "Figure 3", (127, 648, 1000), (9, 370), METHANE_X),
    DigitizedCurve("methane", 377.59, "Figure 3", (127, 648, 1000), (9, 370), METHANE_BLUE),
    DigitizedCurve("methane", 344.26, "Figure 3", (127, 648, 1000), (9, 370), METHANE_RED),
    DigitizedCurve("methane", 310.93, "Figure 3", (127, 648, 1000), (9, 370), METHANE_BLACK),
    DigitizedCurve("propane", 360.93, "Figure 4", (127, 655, 250), (9, 370), PROPANE_MAGENTA),
    DigitizedCurve("propane", 344.26, "Figure 4", (127, 655, 250), (9, 370), PROPANE_BLUE),
    DigitizedCurve("propane", 327.59, "Figure 4", (127, 655, 250), (9, 370), PROPANE_RED),
    DigitizedCurve("propane", 310.93, "Figure 4", (127, 655, 250), (9, 370), PROPANE_BLACK),
    DigitizedCurve(
        "ethane",
        410.93,
        "Supplementary Figure 3",
        (156, 940, 800),
        (31, 569),
        ETHANE_MAGENTA,
    ),
    DigitizedCurve(
        "ethane",
        377.59,
        "Supplementary Figure 3",
        (156, 940, 800),
        (31, 569),
        ETHANE_BLUE,
    ),
    DigitizedCurve(
        "ethane",
        344.26,
        "Supplementary Figure 3",
        (156, 940, 800),
        (31, 569),
        ETHANE_RED,
    ),
    DigitizedCurve(
        "ethane",
        310.93,
        "Supplementary Figure 3",
        (156, 940, 800),
        (31, 569),
        ETHANE_BLACK,
    ),
    DigitizedCurve(
        "n_butane",
        410.93,
        "Supplementary Figure 4",
        (156, 934, 1000),
        (31, 569),
        BUTANE_MAGENTA,
    ),
    DigitizedCurve(
        "n_butane",
        377.59,
        "Supplementary Figure 4",
        (156, 934, 1000),
        (31, 569),
        BUTANE_BLUE,
    ),
    DigitizedCurve(
        "n_butane",
        344.26,
        "Supplementary Figure 4",
        (156, 934, 1000),
        (31, 569),
        BUTANE_RED,
    ),
    DigitizedCurve(
        "n_butane",
        310.93,
        "Supplementary Figure 4",
        (156, 934, 1000),
        (31, 569),
        BUTANE_BLACK,
    ),
)


def main() -> None:
    """Write calibrated values to the committed validation CSV."""
    output = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "data"
        / "not-cleared"
        / "cpa_yan_2009_water_content_digitized.csv"
    )
    fields = (
        "component",
        "temperature_K",
        "pressure_bar",
        "water_mole_fraction",
        "figure",
        "pressure_uncertainty_bar",
        "water_fraction_relative_uncertainty_percent",
        "digitization_method",
        "reference",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for curve in CURVES:
            x0, x1, maximum_pressure = curve.pressure_axis
            y0, y1 = curve.log_fraction_axis
            pressure_uncertainty = 2.0 * maximum_pressure / (x1 - x0)
            for x_pixel, y_pixel in curve.pixels:
                pressure = maximum_pressure * (x_pixel - x0) / (x1 - x0)
                log_fraction = -4.0 * (y_pixel - y0) / (y1 - y0)
                writer.writerow(
                    {
                        "component": curve.component,
                        "temperature_K": f"{curve.temperature:.2f}",
                        "pressure_bar": f"{pressure:.6g}",
                        "water_mole_fraction": f"{10.0**log_fraction:.8g}",
                        "figure": curve.figure,
                        "pressure_uncertainty_bar": f"{pressure_uncertainty:.3g}",
                        "water_fraction_relative_uncertainty_percent": "6",
                        "digitization_method": "visually checked marker-centre pixel calibration",
                        "reference": "Yan et al. (2009), doi:10.1016/j.fluid.2008.10.007",
                    }
                )
    print(f"wrote {sum(len(curve.pixels) for curve in CURVES)} points to {output}")


if __name__ == "__main__":
    main()
