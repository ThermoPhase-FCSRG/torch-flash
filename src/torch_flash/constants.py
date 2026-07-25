"""Physical constants used throughout :mod:`torch_flash`."""

from __future__ import annotations

# CODATA 2018 exact molar gas constant, J mol-1 K-1.
R = 8.31446261815324
"""Exact molar gas constant in J/(mol K), CODATA 2018."""

STANDARD_PRESSURE = 100_000.0
"""Package standard pressure in Pa (1 bar)."""

__all__ = ["STANDARD_PRESSURE", "R"]
