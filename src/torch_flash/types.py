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

    Parameters
    ----------
    composition
        Mole amounts or fractions with components on the final axis.
    atol
        Absolute magnitude below which negative roundoff is clipped to zero.

    Returns
    -------
    Tensor
        Nonnegative final-axis mole fractions normalized to unit sum.

    Raises
    ------
    ValueError
        In eager execution, if the tensor has no component axis, contains
        nonfinite/materially negative values, or a row has zero total.

    Notes
    -----
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
    """Specified temperature, pressure, and molar composition.

    The state is an immutable input record. Temperature and pressure must be
    positive. Composition is normalized along its final axis by
    :func:`normalize_composition`; tiny negative roundoff is clipped and
    materially negative or zero-sum rows are rejected.

    Attributes
    ----------
    temperature:
        Absolute temperature in K. Individual calculation APIs document
        whether scalar or batched temperatures are accepted.
    pressure:
        Absolute pressure in Pa, with shape compatible with ``temperature``.
    composition:
        Mole fractions with components on the final axis. The stored tensor is
        nonnegative and normalized to unit sum.
    """

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

    Attributes
    ----------
    kind:
        Likely physical identity: ``"liquid"``, ``"vapor"``, or ``"unknown"``.
    method:
        Diagnostic rule used to assign ``kind``.
    criterion_value:
        Value compared with ``threshold``, or ``None`` when unavailable.
    threshold:
        Decision threshold in the same units as ``criterion_value``.
    ambiguous:
        Whether the criterion lies within the configured ambiguity band.
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

    Attributes
    ----------
    kind:
        Algebraic root requested from the model: ``"liquid"``, ``"vapor"``,
        or ``"stable"``.
    composition:
        Phase mole fractions, normalized on the final component axis.
    compressibility_factor:
        Dimensionless compressibility factor ``Z = P*v/(R*T)``.
    molar_volume:
        Molar volume in m3/mol.
    log_fugacity_coefficients:
        Dimensionless ``ln(phi_i)`` values.
    fugacities:
        Component fugacities in Pa.
    log_fugacities:
        Dimensionless ``ln(f_i/p_standard)``, with ``p_standard = 1 bar``.
    chemical_potentials:
        Component chemical potentials in J/mol under the selected standard
        state.
    reduced_chemical_potentials:
        Dimensionless ``mu_i/(R*T)``.
    molar_gibbs_energy, molar_helmholtz_energy:
        Total molar free energies in J/mol.
    reduced_gibbs_energy, reduced_helmholtz_energy:
        Dimensionless total molar free energies divided by ``R*T``.
    reduced_residual_gibbs_energy, reduced_residual_helmholtz_energy:
        Reference-independent dimensionless EoS departures.
    residual_enthalpy:
        Residual molar enthalpy in J/mol, or ``None`` when caloric evaluation
        was disabled or unsupported for the state shape.
    residual_entropy:
        Residual molar entropy in J/(mol K), with the same availability as
        ``residual_enthalpy``.
    phase_identification:
        Optional physical-identity diagnostic, separate from ``kind``.
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
        """Return component fugacity coefficients.

        Returns
        -------
        Tensor
            Dimensionless ``phi_i = exp(log(phi_i))``.
        """
        return torch.exp(self.log_fugacity_coefficients)


@dataclass(frozen=True)
class RachfordRiceResult:
    """Solution of the two-phase Rachford-Rice material balance.

    Attributes
    ----------
    vapor_fraction, liquid_fraction:
        Phase molar fractions. For batched inputs these retain the leading
        state dimensions.
    liquid_composition, vapor_composition:
        Normalized phase mole fractions with components on the final axis.
    iterations:
        Maximum iteration count used by the solve.
    converged:
        Boolean tensor indicating convergence for each state.
    residual:
        Final scalar Rachford-Rice residual for each state.
    """

    vapor_fraction: Tensor
    liquid_fraction: Tensor
    liquid_composition: Tensor
    vapor_composition: Tensor
    iterations: int
    converged: Tensor
    residual: Tensor


@dataclass(frozen=True)
class BatchedTwoPhaseFlashResult:
    """Fixed two-phase flash results for a batch of known unstable states.

    This record does not imply that phase stability was checked. It is
    returned by :func:`torch_flash.flash.batched_two_phase_flash`, whose input
    contract assumes every state belongs to the requested two-phase branch.

    Attributes
    ----------
    vapor_fraction, liquid_fraction:
        Phase fractions with one value per input state.
    liquid_composition, vapor_composition:
        Phase mole fractions with shape ``batch_shape + (ncomponents,)``.
    k_values:
        Final positive equilibrium ratios ``K_i = y_i/x_i``.
    iterations:
        Number of fixed substitution/Newton iterations performed.
    converged:
        Boolean tensor with one convergence flag per state.
    residual_norm:
        Maximum absolute log-fugacity residual for each state.
    """

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
    """Result of tangent-plane-distance minimization.

    Attributes
    ----------
    stable:
        Whether the minimum tangent-plane distance is nonnegative within the
        solver tolerance.
    minimum_tpd:
        Smallest dimensionless tangent-plane distance found.
    trial_composition:
        Normalized trial composition at ``minimum_tpd``.
    iterations:
        Iterations associated with the selected trial minimization.
    converged:
        Whether that minimization met its numerical stopping criterion.

    Notes
    -----
    Stability is only as strong as the explored trial-composition basins. A
    small residual or equal-fugacity split is not a substitute for this test.
    """

    stable: bool
    minimum_tpd: Tensor
    trial_composition: Tensor
    iterations: int
    converged: bool


@dataclass(frozen=True)
class FlashResult:
    """Equilibrium phases and numerical diagnostics.

    Attributes
    ----------
    phase_fractions:
        Molar phase fractions in the same order as ``phases``.
    phases:
        Homogeneous properties and compositions for each returned phase.
    converged:
        Whether the flash iteration satisfied its numerical criterion.
    iterations:
        Number of outer flash iterations performed.
    residual_norm:
        Maximum absolute dimensionless log-fugacity mismatch.
    stable:
        Stability status reported by the solver. Its exact meaning depends on
        whether the selected flash API performed phase-stability analysis.
    diagnostics:
        Additional scalar algorithm diagnostics, such as Rachford-Rice or
        autodiff-Newton iteration counts.

    Notes
    -----
    ``phase_kinds`` are post-solve physical-identification diagnostics.
    They do not alter the equilibrium equations or prove the global phase
    count.
    """

    phase_fractions: Tensor
    phases: tuple[PhaseProperties, ...]
    converged: bool
    iterations: int
    residual_norm: Tensor
    stable: bool
    diagnostics: dict[str, float | int | bool | str] = field(default_factory=dict)

    @property
    def nphases(self) -> int:
        """Return the number of phases in the result.

        Returns
        -------
        int
            Length of :attr:`phases`.
        """
        return len(self.phases)

    @property
    def phase_identifications(self) -> tuple[PhaseIdentification | None, ...]:
        """Return physical-identification diagnostics in phase order.

        Returns
        -------
        tuple
            One optional :class:`PhaseIdentification` per phase.
        """
        return tuple(phase.phase_identification for phase in self.phases)

    @property
    def phase_kinds(self) -> tuple[PhaseIdentityKind, ...]:
        """Return likely physical phase identities.

        Returns
        -------
        tuple
            ``"liquid"``, ``"vapor"``, or ``"unknown"`` for each phase.
        """
        return tuple(
            "unknown" if phase.phase_identification is None else phase.phase_identification.kind
            for phase in self.phases
        )

    @property
    def phase_regime(self) -> str:
        """Return a compact aggregate phase-regime label.

        Returns
        -------
        str
            Examples include ``"vapor"``, ``"vapor-liquid"``, and
            ``"3-phase-unknown"``.
        """
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
