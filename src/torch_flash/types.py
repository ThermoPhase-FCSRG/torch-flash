"""Typed result containers shared by thermodynamic domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

PhaseKind = Literal["liquid", "vapor", "stable"]
PhaseIdentityKind = Literal["liquid", "vapor", "unknown"]
PhaseIdentificationMethod = Literal[
    "pedersen-volume-to-covolume",
    "density-ordering",
    "unavailable",
]


def normalize_composition(
    composition: Tensor,
    *,
    atol: float = 1.0e-14,
) -> Tensor:
    """Return a normalized, non-negative mole-fraction tensor.

    The last axis is interpreted as the component axis. Tiny negative values
    caused by roundoff are clipped; materially negative or zero-sum
    compositions are rejected.
    """
    if not torch.is_floating_point(composition):
        composition = composition.to(torch.get_default_dtype())
    # Tensor-to-Python validation branches are intentionally excluded from
    # compiled graphs. Public eager calls retain the strict validation below;
    # compiled kernels assume the same documented input contract. Keeping the
    # numerical normalization in the graph lets torch.compile fuse the small
    # tensor operations that dominate multifluid and association models.
    if not torch.compiler.is_compiling():
        if composition.ndim < 1:
            raise ValueError("composition must have at least one dimension")
        if not torch.isfinite(composition).all():
            raise ValueError("composition must contain only finite values")
        if bool((composition < -atol).any()):
            raise ValueError("composition cannot contain negative mole fractions")
    clipped = torch.clamp_min(composition, 0.0)
    total = clipped.sum(dim=-1, keepdim=True)
    if not torch.compiler.is_compiling() and bool((total <= 0.0).any()):
        raise ValueError("composition must have a positive sum")
    return clipped / total


@dataclass(frozen=True)
class ChemicalState:
    """Specified temperature, pressure, and overall composition."""

    temperature: Tensor
    pressure: Tensor
    composition: Tensor

    def __post_init__(self) -> None:
        if bool((self.temperature <= 0.0).any()):
            raise ValueError("temperature must be positive")
        if bool((self.pressure <= 0.0).any()):
            raise ValueError("pressure must be positive")
        object.__setattr__(self, "composition", normalize_composition(self.composition))


@dataclass(frozen=True)
class PhaseIdentification:
    """Likely physical identity of a homogeneous phase.

    ``kind`` is deliberately separate from the EoS root requested through
    :class:`PhaseProperties`. For the Pedersen cubic-EoS criterion,
    ``criterion_value`` is ``V/b`` and ``threshold`` is normally 1.75. For
    density ordering, they are the phase molar volume and the geometric-mean
    separator between the two least-dense phases. The Boolean ``ambiguous``
    marks values within the configured relative band around the separator.

    Phase identification is a naming diagnostic; it does not change the
    equilibrium calculation or any thermodynamic property.
    """

    kind: PhaseIdentityKind
    method: PhaseIdentificationMethod
    criterion_value: Tensor | None
    threshold: Tensor | None
    ambiguous: bool


@dataclass(frozen=True)
class PhaseProperties:
    """Thermodynamic properties of one homogeneous phase.

    Fugacities are in Pa and ``log_fugacities`` are the dimensionless
    ``ln(f_i / p_standard)`` values. Chemical potentials and molar free
    energies are in J/mol. ``reduced_*`` chemical potentials and energies are
    dimensionless quantities divided by ``R*T``. Total quantities use the
    standard-state convention selected by ``phase_properties``; the reduced
    residual quantities are reference-independent EoS departures.
    """

    kind: str
    composition: Tensor
    compressibility_factor: Tensor
    molar_volume: Tensor
    log_fugacity_coefficients: Tensor
    fugacities: Tensor
    log_fugacities: Tensor
    chemical_potentials: Tensor
    reduced_chemical_potentials: Tensor
    molar_gibbs_energy: Tensor
    molar_helmholtz_energy: Tensor
    reduced_gibbs_energy: Tensor
    reduced_helmholtz_energy: Tensor
    reduced_residual_gibbs_energy: Tensor
    reduced_residual_helmholtz_energy: Tensor
    residual_enthalpy: Tensor | None = None
    residual_entropy: Tensor | None = None
    phase_identification: PhaseIdentification | None = None

    @property
    def fugacity_coefficients(self) -> Tensor:
        """Fugacity coefficients, ``exp(log(phi))``."""
        return torch.exp(self.log_fugacity_coefficients)


@dataclass(frozen=True)
class RachfordRiceResult:
    """Solution of the scalar two-phase material-balance equation."""

    vapor_fraction: Tensor
    liquid_fraction: Tensor
    liquid_composition: Tensor
    vapor_composition: Tensor
    iterations: int
    converged: Tensor
    residual: Tensor


@dataclass(frozen=True)
class BatchedTwoPhaseFlashResult:
    """Fixed two-phase flash results for a batch of known unstable states."""

    vapor_fraction: Tensor
    liquid_fraction: Tensor
    liquid_composition: Tensor
    vapor_composition: Tensor
    k_values: Tensor
    iterations: int
    converged: Tensor
    residual_norm: Tensor


@dataclass(frozen=True)
class StabilityResult:
    """Result of tangent-plane-distance minimization."""

    stable: bool
    minimum_tpd: Tensor
    trial_composition: Tensor
    iterations: int
    converged: bool


@dataclass(frozen=True)
class FlashResult:
    """Equilibrium phases and numerical diagnostics."""

    phase_fractions: Tensor
    phases: tuple[PhaseProperties, ...]
    converged: bool
    iterations: int
    residual_norm: Tensor
    stable: bool
    diagnostics: dict[str, float | int | bool | str] = field(default_factory=dict)

    @property
    def nphases(self) -> int:
        """Number of equilibrium phases."""
        return len(self.phases)

    @property
    def phase_identifications(self) -> tuple[PhaseIdentification | None, ...]:
        """Per-phase physical-identification diagnostics."""
        return tuple(phase.phase_identification for phase in self.phases)

    @property
    def phase_kinds(self) -> tuple[PhaseIdentityKind, ...]:
        """Likely physical phase kinds, using ``unknown`` when unavailable."""
        return tuple(
            "unknown" if phase.phase_identification is None else phase.phase_identification.kind
            for phase in self.phases
        )

    @property
    def phase_regime(self) -> str:
        """Return a compact overall label such as ``vapor-liquid``."""
        kinds = self.phase_kinds
        if not kinds:
            return "unknown"
        if len(kinds) == 1:
            return kinds[0]
        if "unknown" in kinds:
            return f"{len(kinds)}-phase-unknown"
        vapor_count = kinds.count("vapor")
        liquid_count = kinds.count("liquid")
        labels = ["vapor"] * vapor_count + ["liquid"] * liquid_count
        return "-".join(labels)
