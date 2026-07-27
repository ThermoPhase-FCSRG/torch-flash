"""Stability-tested phase-equilibrium flash calculations."""

from .batched import batched_tangent_plane_stability, batched_two_phase_flash
from .grid import (
    DEFAULT_GRID_PHASE_IDENTIFICATION_METHODS,
    GRID_PHASE_REGION_LABELS,
    BinaryThreePhaseInvariant,
    BinaryThreePhaseInvariantBatch,
    GridEquilibrium,
    GridFlashOptions,
    GridPhaseIdentification,
    PhaseRegionBoundaryCurve,
    PhaseRegionBoundarySet,
    TrustRegionGridPolishResult,
    flash_grid,
    flash_grid_oracle,
    identify_grid_phases,
    polish_grid_equilibrium_with_trust_region,
    refine_flash_grid_phase_boundaries,
    solve_batched_binary_three_phase_invariants,
    solve_binary_three_phase_invariant,
)
from .multiphase import (
    multiphase_flash,
    multiphase_trust_region_flash,
    solve_generalized_rachford_rice,
)
from .stability import tangent_plane_stability
from .two_phase import two_phase_flash, two_phase_trust_region_flash

__all__ = [
    "DEFAULT_GRID_PHASE_IDENTIFICATION_METHODS",
    "GRID_PHASE_REGION_LABELS",
    "BinaryThreePhaseInvariant",
    "BinaryThreePhaseInvariantBatch",
    "GridEquilibrium",
    "GridFlashOptions",
    "GridPhaseIdentification",
    "PhaseRegionBoundaryCurve",
    "PhaseRegionBoundarySet",
    "TrustRegionGridPolishResult",
    "batched_tangent_plane_stability",
    "batched_two_phase_flash",
    "flash_grid",
    "flash_grid_oracle",
    "identify_grid_phases",
    "multiphase_flash",
    "multiphase_trust_region_flash",
    "polish_grid_equilibrium_with_trust_region",
    "refine_flash_grid_phase_boundaries",
    "solve_batched_binary_three_phase_invariants",
    "solve_binary_three_phase_invariant",
    "solve_generalized_rachford_rice",
    "tangent_plane_stability",
    "two_phase_flash",
    "two_phase_trust_region_flash",
]
