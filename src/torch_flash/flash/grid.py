"""Batched stability-tested one-, two-, and three-phase grid flashes.

The fast hierarchy combines tangent-plane stability, vectorized two-phase
flash, sparse Gibbs-allocation minimization, and autodiff Newton refinement.
A full multistart Gibbs calculation remains available as a correctness oracle.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from torch_flash.components import ComponentSet
from torch_flash.flash.batched import (
    batched_tangent_plane_stability,
    batched_two_phase_flash,
)
from torch_flash.initialization import wilson_k_values
from torch_flash.properties.state import StateModel
from torch_flash.types import (
    BatchedStabilityResult,
    BatchedTwoPhaseFlashResult,
    ChemicalState,
    PhaseIdentificationCriterion,
    PhaseKind,
)

DEFAULT_GRID_PHASE_IDENTIFICATION_METHODS: tuple[PhaseIdentificationCriterion, ...] = (
    "li-pseudo-critical-temperature",
    "pedersen-volume-to-covolume",
    "perschke-negative-flash",
    "pasad-isothermal-compressibility-derivative",
    "bennett-thermal-expansion-derivative",
)

GRID_PHASE_REGION_LABELS = ("V", "L", "LV", "LL", "three-phase", "unavailable")


@dataclass(frozen=True)
class GridFlashOptions:
    """Numerical controls for batched phase-count-discovering TP flashes.

    The defaults target float64 reference calculations. Tolerances are applied
    to dimensionless log-fugacity or composition residuals unless stated
    otherwise. Changing a start count, merge tolerance, or topology-audit
    setting can change which Gibbs basin is selected; these fields are
    therefore part of the scientific configuration rather than hidden
    performance switches.

    Attributes
    ----------
    chunk_size
        Maximum number of independent states passed to one batched stability
        or fixed two-phase kernel. It bounds temporary tensor size; it does not
        subdivide or couple the thermodynamic problem.
    random_allocation_starts
        Number of reproducible random three-phase component-allocation starts
        added to the deterministic Wilson and component-rich starts.
    random_seed
        PyTorch generator seed used only for random allocation starts.
    fallback_workers
        Number of Python threads used for the final sparse scalar refinement.
        The batched search is already vectorized. Values above one may be
        slower when PyTorch intra-operation threading is also enabled.
    debug
        Print fallback candidate counts and Newton acceptance summaries.
        Library results and numerical tolerances are unchanged.
    phase_fraction_tolerance
        Minimum retained molar phase fraction. Smaller phases are treated as
        disappearing and removed after refinement.
    phase_composition_merge_tolerance
        Maximum absolute mole-fraction difference for merging two candidate
        phase compositions.
    gibbs_reduction_tolerance
        Minimum dimensionless reduction in ``G/(R*T)`` required to replace a
        converged lower-phase-count state.
    fugacity_tolerance
        Maximum absolute log-fugacity equality residual accepted for a
        multiphase result.
    material_balance_tolerance
        Maximum absolute component mole-fraction material-balance residual.
    flash_newton_tolerance
        Maximum residual used to terminate each scalar equal-fugacity Newton
        refinement.
    stability_tolerance
        Tangent-plane-distance and stability-iteration tolerance. A state is
        treated as unstable when its minimum TPD is less than the negative of
        this value.
    independent_reflash_starts
        Number of lowest-Gibbs starts independently refined by
        :func:`flash_grid_oracle` for each multicomponent state.
    stability_iterations
        Maximum successive-substitution passes in each batched stability
        calculation.
    two_phase_substitution_iterations
        Maximum fixed-substitution passes before the batched two-phase Newton
        correction.
    two_phase_newton_iterations
        Maximum batched autodiff-Newton corrections for a known two-phase
        state.
    gibbs_fallback_adam_iterations
        Adam iterations applied simultaneously to the sparse difficult-state
        component-allocation problems.
    three_phase_newton_iterations
        Maximum batched autodiff-Newton corrections for candidate three-phase
        states.

    Raises
    ------
    ValueError
        If an iteration, chunk, worker, or independent-start count is not
        positive; if ``random_allocation_starts`` is negative; or if a
        tolerance is not positive.
    """

    chunk_size: int = 2048
    random_allocation_starts: int = 8
    random_seed: int = 20260724
    fallback_workers: int = 1
    debug: bool = False
    phase_fraction_tolerance: float = 1.0e-4
    phase_composition_merge_tolerance: float = 2.0e-3
    gibbs_reduction_tolerance: float = 2.0e-7
    fugacity_tolerance: float = 1.0e-8
    material_balance_tolerance: float = 5.0e-11
    flash_newton_tolerance: float = 1.0e-11
    stability_tolerance: float = 1.0e-7
    independent_reflash_starts: int = 4
    stability_iterations: int = 40
    two_phase_substitution_iterations: int = 30
    two_phase_newton_iterations: int = 8
    gibbs_fallback_adam_iterations: int = 80
    three_phase_newton_iterations: int = 48

    def __post_init__(self) -> None:
        positive_integers = (
            self.chunk_size,
            self.fallback_workers,
            self.independent_reflash_starts,
            self.stability_iterations,
            self.two_phase_substitution_iterations,
            self.two_phase_newton_iterations,
            self.gibbs_fallback_adam_iterations,
            self.three_phase_newton_iterations,
        )
        if any(value <= 0 for value in positive_integers):
            raise ValueError(
                "grid-flash iteration, chunk, start, and worker counts must be positive"
            )
        if self.random_allocation_starts < 0:
            raise ValueError("random_allocation_starts must be nonnegative")
        positive_tolerances = (
            self.phase_fraction_tolerance,
            self.phase_composition_merge_tolerance,
            self.gibbs_reduction_tolerance,
            self.fugacity_tolerance,
            self.material_balance_tolerance,
            self.flash_newton_tolerance,
            self.stability_tolerance,
        )
        if any(value <= 0.0 for value in positive_tolerances):
            raise ValueError("grid-flash tolerances must be positive")


@dataclass(frozen=True)
class BinaryThreePhaseInvariant:
    """Solved binary three-phase invariant used by a grid flash.

    Attributes
    ----------
    temperature
        Scalar invariant temperature in K.
    pressure
        Scalar invariant pressure in Pa.
    phase_compositions
        Tensor with shape ``(3, 2)`` containing normalized binary phase mole
        fractions. Rows are sorted by increasing first-component mole
        fraction; root labels are not encoded in this ordering.
    residual_norm
        Maximum absolute dimensionless log-fugacity mismatch among the three
        phases.
    iterations
        Number of damped autodiff-Newton iterations attempted.
    converged
        Whether ``residual_norm`` satisfies the requested solver tolerance.

    Notes
    -----
    A binary three-phase invariant fixes phase compositions and pressure at a
    specified temperature, but it does not uniquely fix all three phase
    fractions. :func:`flash_grid` applies a positive centered lever-rule
    representative only for feeds strictly inside the outer phase-composition
    interval.
    """

    temperature: Tensor
    pressure: Tensor
    phase_compositions: Tensor
    residual_norm: Tensor
    iterations: int
    converged: bool


@dataclass(frozen=True)
class GridEquilibrium:
    """Padded phase-equilibrium results over independent TP states.

    State fields are flattened in row-major order. ``grid_shape`` records the
    original temperature/pressure batch shape so callers can reshape any
    per-state tensor without reconstructing axis lengths.

    Attributes
    ----------
    temperatures
        Flattened temperatures in K with shape ``(nstates,)``.
    pressures
        Flattened pressures in Pa with shape ``(nstates,)``.
    feeds
        Normalized overall mole fractions with shape
        ``(nstates, ncomponents)``.
    grid_shape
        Original non-scalar batch shape of the input temperatures and
        pressures.
    phase_fractions
        Molar phase fractions with shape ``(nstates, 3)``. Entries beyond
        ``phase_counts`` are zero.
    phase_compositions
        Equilibrium mole fractions with shape
        ``(nstates, 3, ncomponents)``. Padded phase rows are NaN.
    phase_counts
        Number of retained distinct phases per state, from one through three.
    gibbs_reduction
        Dimensionless reduction in ``G/(R*T)`` relative to the homogeneous
        feed state.
    fugacity_residual
        Maximum absolute log-fugacity mismatch among retained phases.
    material_balance_residual
        Maximum absolute component mole-fraction material-balance residual.
    converged
        Per-state flag requiring both fugacity and material-balance gates.
        Callers must inspect this tensor before interpreting phase counts.
    elapsed_seconds
        Total wall-clock seconds for the grid call.
    batched_search_seconds
        Wall-clock seconds spent in stability, vectorized two-phase work, and
        initial Gibbs candidate discovery.
    refinement_seconds
        Wall-clock seconds spent refining and auditing difficult states.
    difficult_state_count
        Number of states sent to the difficult-state Gibbs fallback. For
        :func:`flash_grid_oracle`, this is the number of multicomponent states
        examined independently.
    topology_audit_count
        Number of additional cells selected by the two-dimensional topology
        audit. This is zero for :func:`flash_grid_oracle`.
    initial_fallback_replacements
        Number of first-pass grid candidates replaced by a lower-Gibbs,
        converged difficult-state result.
    topology_audit_replacements
        Number of already-installed grid candidates replaced by a lower-Gibbs,
        converged result during the subsequent topology audit.

    Notes
    -----
    Root selection, physical phase identification, and equilibrium phase count
    are separate. Use :func:`identify_grid_phases` to label the returned
    equilibrium compositions. Grid flashes are solver operations with
    discrete phase-count decisions; returned tensors are diagnostic results,
    not a differentiable end-to-end flash mapping.
    """

    temperatures: Tensor
    pressures: Tensor
    feeds: Tensor
    grid_shape: tuple[int, ...]
    phase_fractions: Tensor
    phase_compositions: Tensor
    phase_counts: Tensor
    gibbs_reduction: Tensor
    fugacity_residual: Tensor
    material_balance_residual: Tensor
    converged: Tensor
    elapsed_seconds: float
    batched_search_seconds: float
    refinement_seconds: float
    difficult_state_count: int
    topology_audit_count: int
    initial_fallback_replacements: int
    topology_audit_replacements: int


@dataclass(frozen=True)
class GridPhaseIdentification:
    """Physical identities evaluated at every equilibrium composition.

    Attributes
    ----------
    methods
        Ordered phase-identification criteria. This is the leading axis of all
        returned tensors.
    phase_identity_codes
        Integer tensor with shape ``(nmethods, nstates, 3)``. Codes are -1 for
        unknown or padded, 0 for liquid, and 1 for vapor.
    criterion_values
        Native scalar criterion values with the same padded shape as
        ``phase_identity_codes``. Units depend on the method: dimensionless
        ``T/Tc``, dimensionless ``V/b``, dimensionless negative-flash
        residual, ``1/(Pa K)``, or ``1/K^2``.
    thresholds
        Decision threshold in the same units as each criterion value.
    ambiguous
        Boolean ambiguity flags with shape ``(nmethods, nstates, 3)``. Padded,
        unavailable, and non-converged entries are true.
    region_codes
        Integer tensor with shape ``(nmethods, *grid_shape)`` indexing
        :data:`GRID_PHASE_REGION_LABELS`.
    elapsed_seconds
        Wall-clock seconds spent evaluating all requested methods.

    Notes
    -----
    The ``"three-phase"`` region is deliberately identity-neutral. Individual
    phase codes retain whether a criterion returned LLV, LLL, or another
    combination. Scalar diagnostic values are detached to avoid retaining one
    higher-order autodiff graph per grid cell. Call :func:`identify_phase`
    directly when derivatives of a criterion with respect to state or model
    parameters are required.
    """

    methods: tuple[PhaseIdentificationCriterion, ...]
    phase_identity_codes: Tensor
    criterion_values: Tensor
    thresholds: Tensor
    ambiguous: Tensor
    region_codes: Tensor
    elapsed_seconds: float


MAX_PHASES = 3


def identify_grid_phases(
    model: StateModel,
    equilibrium: GridEquilibrium,
    *,
    methods: tuple[PhaseIdentificationCriterion, ...] = DEFAULT_GRID_PHASE_IDENTIFICATION_METHODS,
    phase: PhaseKind = "stable",
    volume_to_covolume_threshold: float = 1.75,
    pseudo_critical_temperature_factor: float = 1.0,
    ambiguity_relative_tolerance: float = 0.05,
) -> GridPhaseIdentification:
    """Identify each equilibrium phase and summarize paper-style regions.

    Equilibrium phase count and physical identity remain separate: this
    function never changes the flash result. Each method is applied to every
    converged phase at its equilibrium composition.

    Parameters
    ----------
    model
        Homogeneous-state model used to produce ``equilibrium``. It must expose
        the properties required by every requested criterion.
    equilibrium
        Result returned by :func:`flash_grid` or
        :func:`flash_grid_oracle`.
    methods
        Unique ordered criteria to evaluate. The default is all five Bennett
        and Schmidt methods.
    phase
        Homogeneous EoS root request used while evaluating each already
        separated equilibrium composition. ``"stable"`` is normally
        appropriate.
    volume_to_covolume_threshold
        Dimensionless Pedersen ``V/b`` separator. Values above the threshold
        are vapor-like.
    pseudo_critical_temperature_factor
        Positive factor multiplying Li's volume-weighted pseudo-critical
        temperature. Bennett and Schmidt use one.
    ambiguity_relative_tolerance
        Nonnegative relative band around each criterion threshold. It changes
        only ``ambiguous`` flags, not the sign-based identity.

    Returns
    -------
    GridPhaseIdentification
        Per-phase identities, native criterion values and thresholds,
        ambiguity flags, paper-style region codes, and elapsed time.

    Raises
    ------
    ValueError
        If no method is requested, a method is repeated, the equilibrium
        arrays are inconsistent, or a criterion option is invalid.
    TypeError
        If the model lacks a property required by a requested method and that
        method cannot report itself as unavailable.

    Notes
    -----
    A non-converged equilibrium state or an unavailable/unknown phase identity
    maps to the ``"unavailable"`` region. For two phases, any vapor-like phase
    maps to ``"LV"`` and otherwise to ``"LL"``. Three phases share one plotting
    category; inspect ``phase_identity_codes`` for the exact labels.

    The two derivative criteria evaluate first and second PyTorch autodiff
    derivatives at every equilibrium composition. The returned grid tensors
    are detached intentionally; the scalar :func:`identify_phase` API retains
    its autodiff graph.
    """
    from torch_flash.properties.phase_identification import identify_phase

    if not methods:
        raise ValueError("at least one phase-identification method is required")
    if len(set(methods)) != len(methods):
        raise ValueError("phase-identification methods must be unique")
    state_count = equilibrium.temperatures.numel()
    if equilibrium.phase_counts.shape != (state_count,):
        raise ValueError("equilibrium phase counts do not match the state batch")

    result_shape = (len(methods), state_count, MAX_PHASES)
    identities = torch.full(
        result_shape,
        -1,
        dtype=torch.int8,
        device=equilibrium.feeds.device,
    )
    criterion_values = equilibrium.temperatures.new_full(result_shape, torch.nan)
    thresholds = equilibrium.temperatures.new_full(result_shape, torch.nan)
    ambiguous = torch.ones(
        result_shape,
        dtype=torch.bool,
        device=equilibrium.feeds.device,
    )
    region_codes = torch.full(
        (len(methods), state_count),
        GRID_PHASE_REGION_LABELS.index("unavailable"),
        dtype=torch.int8,
        device=equilibrium.feeds.device,
    )
    started = time.perf_counter()
    for method_index, method in enumerate(methods):
        for state_index in range(state_count):
            if not bool(equilibrium.converged[state_index]):
                continue
            phase_count = int(equilibrium.phase_counts[state_index])
            available = True
            for phase_index in range(phase_count):
                state = ChemicalState(
                    equilibrium.temperatures[state_index],
                    equilibrium.pressures[state_index],
                    equilibrium.phase_compositions[state_index, phase_index],
                )
                identification = identify_phase(
                    model,
                    state,
                    phase,
                    method=method,
                    threshold=volume_to_covolume_threshold,
                    pseudo_critical_temperature_factor=(pseudo_critical_temperature_factor),
                    ambiguity_relative_tolerance=ambiguity_relative_tolerance,
                )
                if identification.kind == "unknown":
                    available = False
                    break
                identities[method_index, state_index, phase_index] = (
                    1 if identification.kind == "vapor" else 0
                )
                ambiguous[method_index, state_index, phase_index] = identification.ambiguous
                if identification.criterion_value is not None:
                    criterion_values[method_index, state_index, phase_index] = (
                        identification.criterion_value.detach()
                    )
                if identification.threshold is not None:
                    thresholds[method_index, state_index, phase_index] = (
                        identification.threshold.detach()
                    )
            if not available:
                continue
            has_vapor = bool((identities[method_index, state_index, :phase_count] == 1).any())
            if phase_count == 1:
                label = "V" if has_vapor else "L"
            elif phase_count == 2:
                label = "LV" if has_vapor else "LL"
            else:
                label = "three-phase"
            region_codes[method_index, state_index] = GRID_PHASE_REGION_LABELS.index(label)

    return GridPhaseIdentification(
        methods=methods,
        phase_identity_codes=identities,
        criterion_values=criterion_values,
        thresholds=thresholds,
        ambiguous=ambiguous,
        region_codes=region_codes.reshape((len(methods), *equilibrium.grid_shape)),
        elapsed_seconds=time.perf_counter() - started,
    )


def _batched_stability_in_chunks(
    model: StateModel,
    temperatures: Tensor,
    pressures: Tensor,
    compositions: Tensor,
    options: GridFlashOptions,
) -> BatchedStabilityResult:
    """Run vectorized stability screening with a bounded peak batch size."""
    results = []
    for start in range(0, temperatures.numel(), options.chunk_size):
        stop = min(start + options.chunk_size, temperatures.numel())
        with torch.no_grad():
            results.append(
                batched_tangent_plane_stability(
                    model,
                    ChemicalState(
                        temperatures[start:stop],
                        pressures[start:stop],
                        compositions[start:stop],
                    ),
                    tolerance=options.stability_tolerance,
                    max_iterations=options.stability_iterations,
                )
            )
    return BatchedStabilityResult(
        torch.cat(tuple(result.stable for result in results)),
        torch.cat(tuple(result.minimum_tpd for result in results)),
        torch.cat(tuple(result.trial_composition for result in results)),
        max(result.iterations for result in results),
        torch.cat(tuple(result.converged for result in results)),
        torch.cat(tuple(result.residual_norm for result in results)),
    )


def _batched_two_phase_in_chunks(
    model: StateModel,
    temperatures: Tensor,
    pressures: Tensor,
    compositions: Tensor,
    initial_k_values: Tensor,
    options: GridFlashOptions,
) -> BatchedTwoPhaseFlashResult:
    """Run independent two-phase flashes with a bounded autodiff batch."""
    results = []
    for start in range(0, temperatures.numel(), options.chunk_size):
        stop = min(start + options.chunk_size, temperatures.numel())
        results.append(
            batched_two_phase_flash(
                model,
                ChemicalState(
                    temperatures[start:stop],
                    pressures[start:stop],
                    compositions[start:stop],
                ),
                initial_k_values=initial_k_values[start:stop],
                phase_roots=("stable", "stable"),
                tolerance=options.fugacity_tolerance,
                substitution_iterations=options.two_phase_substitution_iterations,
                newton_iterations=options.two_phase_newton_iterations,
            )
        )
    return BatchedTwoPhaseFlashResult(
        torch.cat(tuple(result.vapor_fraction for result in results)),
        torch.cat(tuple(result.liquid_fraction for result in results)),
        torch.cat(tuple(result.liquid_composition for result in results)),
        torch.cat(tuple(result.vapor_composition for result in results)),
        torch.cat(tuple(result.k_values for result in results)),
        max(result.iterations for result in results),
        torch.cat(tuple(result.converged for result in results)),
        torch.cat(tuple(result.residual_norm for result in results)),
    )


def _grid_states(state: ChemicalState) -> tuple[Tensor, Tensor, Tensor]:
    """Validate and flatten one independent state per batch row."""
    if state.temperature.shape != state.pressure.shape or state.temperature.ndim == 0:
        raise ValueError("flash_grid requires equal, non-scalar temperature and pressure batches")
    state_count = state.temperature.numel()
    temperatures = state.temperature.reshape(-1)
    pressures = state.pressure.reshape(-1)
    if state.composition.ndim == 1:
        feeds = state.composition.expand(state_count, -1)
    elif state.composition.shape[:-1] == state.temperature.shape:
        feeds = state.composition.reshape(state_count, state.composition.shape[-1])
    else:
        raise ValueError("flash_grid composition batch must match temperature and pressure")
    return temperatures, pressures, feeds


def _allocation_initial_logits(
    model: StateModel,
    temperatures: Tensor,
    pressures: Tensor,
    feeds: Tensor,
    options: GridFlashOptions,
) -> Tensor:
    """Construct volatility, component-rich, and randomized phase partitions."""
    required = ("critical_temperature", "critical_pressure", "acentric_factor")
    if not all(hasattr(model, item) for item in required):
        raise ValueError("grid flash requires model critical constants for Wilson starts")
    components = cast(ComponentSet, model)
    k_values = wilson_k_values(components, temperatures, pressures)
    score = torch.log(k_values)
    score = score - score.mean(dim=-1, keepdim=True)
    score = score / torch.clamp_min(score.abs().amax(dim=-1, keepdim=True), 1.0)

    centers = score.new_tensor([-1.0, 0.0, 1.0])
    starts = []
    for scale in (2.0, 6.0):
        starts.append(-scale * (score[:, None, :] - centers[None, :, None]).square())
    for scale in (2.0, 5.0):
        starts.append(
            torch.stack(
                (
                    -scale * score,
                    scale * score,
                    torch.zeros_like(score),
                ),
                dim=1,
            )
        )

    # Three-phase petroleum states can contain a component-rich liquid that a
    # volatility-only partition misses. Give every component an independent
    # deterministic trial as the third phase; absent/trace feed components are
    # harmless because the component-allocation parameterization preserves
    # their total amounts exactly.
    for component_index in range(feeds.shape[-1]):
        component_rich = torch.stack(
            (
                -1.5 * score,
                1.5 * score,
                torch.full_like(score, -4.0),
            ),
            dim=1,
        )
        component_rich[:, 2, component_index] = 8.0
        starts.append(component_rich)

    generator = torch.Generator(device=score.device)
    generator.manual_seed(options.random_seed)
    for _ in range(options.random_allocation_starts):
        random_partition = 5.0 * torch.randn(
            (MAX_PHASES, score.shape[1]),
            dtype=score.dtype,
            device=score.device,
            generator=generator,
        )
        random_logits = (
            random_partition[None, :, :]
            .expand(
                score.shape[0],
                -1,
                -1,
            )
            .clone()
        )
        starts.append(random_logits)
    return torch.stack(starts, dim=1)


def _allocation_quantities(
    model: StateModel,
    temperatures: Tensor,
    pressures: Tensor,
    feeds: Tensor,
    logits: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    bounded_logits = torch.clamp(
        torch.nan_to_num(logits, nan=0.0, posinf=60.0, neginf=-60.0),
        -60.0,
        60.0,
    )
    allocations = torch.softmax(bounded_logits, dim=-2)
    phase_moles = allocations * feeds[:, None, None, :]
    phase_fractions = torch.clamp_min(phase_moles.sum(dim=-1), 1.0e-30)
    phase_compositions = phase_moles / phase_fractions[..., None]
    phase_compositions = torch.clamp_min(phase_compositions, 1.0e-30)
    phase_compositions = phase_compositions / phase_compositions.sum(
        dim=-1,
        keepdim=True,
    )
    expanded_temperature = temperatures[:, None, None].expand_as(phase_fractions)
    expanded_pressure = pressures[:, None, None].expand_as(phase_fractions)
    log_phi = model.log_fugacity_coefficients(
        expanded_temperature,
        expanded_pressure,
        phase_compositions,
        "stable",
    )
    gibbs = torch.sum(
        phase_moles * (torch.log(phase_compositions) + log_phi),
        dim=(-1, -2),
    )
    return gibbs, phase_fractions, phase_compositions


def _merge_candidate_phases(
    fractions: Tensor,
    compositions: Tensor,
    options: GridFlashOptions,
) -> tuple[Tensor, Tensor]:
    """Remove vanishing phases and merge duplicate optimized phases."""
    groups: list[tuple[Tensor, Tensor]] = []
    for phase_index in torch.argsort(fractions, descending=True):
        fraction = fractions[phase_index]
        if float(fraction) <= options.phase_fraction_tolerance:
            continue
        composition = compositions[phase_index]
        for group_index, (group_fraction, group_composition) in enumerate(groups):
            if (
                float(torch.max(torch.abs(composition - group_composition)))
                <= options.phase_composition_merge_tolerance
            ):
                merged_fraction = group_fraction + fraction
                merged_composition = (
                    group_fraction * group_composition + fraction * composition
                ) / merged_fraction
                groups[group_index] = merged_fraction, merged_composition
                break
        else:
            groups.append((fraction, composition))
    if not groups:
        raise RuntimeError("Gibbs minimization returned no active phase")
    return (
        torch.stack(tuple(item[0] for item in groups)),
        torch.stack(tuple(item[1] for item in groups)),
    )


def _refine_state_allocation(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    feed: Tensor,
    initial_logits: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Refine one candidate split to a strict autodiff stationarity tolerance."""
    logits = torch.nn.Parameter(initial_logits.clone())
    optimizer = torch.optim.LBFGS(
        (logits,),
        lr=0.5,
        max_iter=220,
        history_size=20,
        tolerance_grad=2.0e-12,
        tolerance_change=2.0e-15,
        line_search_fn="strong_wolfe",
    )

    def quantities() -> tuple[Tensor, Tensor, Tensor]:
        gibbs, fractions, compositions = _allocation_quantities(
            model,
            temperature[None],
            pressure[None],
            feed[None, :],
            logits[None, None, :, :],
        )
        return gibbs[0, 0], fractions[0, 0], compositions[0, 0]

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        gibbs, _, _ = quantities()
        gibbs.backward()
        return gibbs

    optimizer.step(closure)
    with torch.no_grad():
        return quantities()


def _logits_from_phases(
    fractions: Tensor,
    compositions: Tensor,
) -> Tensor:
    component_allocations = fractions[:, None] * compositions
    component_allocations = component_allocations / component_allocations.sum(
        dim=0,
        keepdim=True,
    )
    return torch.log(torch.clamp_min(component_allocations, 1.0e-12))


def _flash_quantities(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    feed: Tensor,
    variables: Tensor,
    phase_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return residual, fractions, compositions, and Gibbs energy for a PT flash."""
    component_count = feed.numel()
    log_k_size = (phase_count - 1) * component_count
    log_k = variables[:log_k_size].reshape(phase_count - 1, component_count)
    fraction_coordinates = variables[log_k_size:]
    fractions = torch.softmax(
        torch.cat((variables.new_zeros(1), fraction_coordinates)),
        dim=0,
    )
    ratios = torch.cat((torch.ones_like(log_k[:1]), torch.exp(log_k)), dim=0)
    denominator = torch.sum(fractions[:, None] * ratios, dim=0)
    raw_compositions = ratios * feed[None, :] / denominator[None, :]
    compositions = raw_compositions / raw_compositions.sum(dim=1, keepdim=True)

    phase_temperature = temperature.expand(phase_count)
    phase_pressure = pressure.expand(phase_count)
    chemical_potentials = torch.log(compositions) + model.log_fugacity_coefficients(
        phase_temperature,
        phase_pressure,
        compositions,
        "stable",
    )
    fugacity_residuals = (chemical_potentials[1:] - chemical_potentials[0]).reshape(-1)
    normalization_residuals = raw_compositions[1:].sum(dim=1) - 1.0
    residual = torch.cat((fugacity_residuals, normalization_residuals))
    phase_moles = fractions[:, None] * compositions
    gibbs = torch.sum(phase_moles * chemical_potentials)
    return residual, fractions, compositions, gibbs


def _batched_three_phase_quantities(
    model: StateModel,
    temperatures: Tensor,
    pressures: Tensor,
    feeds: Tensor,
    variables: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate independent three-phase PT residuals in one tensor batch."""
    state_count, component_count = feeds.shape
    log_k_size = 2 * component_count
    log_k = variables[:, :log_k_size].reshape(state_count, 2, component_count)
    fraction_coordinates = variables[:, log_k_size:]
    fractions = torch.softmax(
        torch.cat(
            (
                variables.new_zeros((state_count, 1)),
                fraction_coordinates,
            ),
            dim=-1,
        ),
        dim=-1,
    )
    ratios = torch.cat(
        (
            torch.ones_like(log_k[:, :1]),
            torch.exp(log_k),
        ),
        dim=1,
    )
    denominator = torch.sum(fractions[..., None] * ratios, dim=1)
    raw_compositions = ratios * feeds[:, None, :] / denominator[:, None, :]
    compositions = raw_compositions / raw_compositions.sum(dim=-1, keepdim=True)
    chemical_potentials = torch.log(compositions) + model.log_fugacity_coefficients(
        temperatures[:, None],
        pressures[:, None],
        compositions,
        "stable",
    )
    fugacity_residuals = (chemical_potentials[:, 1:] - chemical_potentials[:, :1]).reshape(
        state_count, -1
    )
    normalization_residuals = raw_compositions[:, 1:].sum(dim=-1) - 1.0
    residual = torch.cat(
        (
            fugacity_residuals,
            normalization_residuals,
        ),
        dim=-1,
    )
    phase_moles = fractions[..., None] * compositions
    gibbs = torch.sum(phase_moles * chemical_potentials, dim=(-1, -2))
    return residual, fractions, compositions, gibbs


def _batched_refine_three_phase(
    model: StateModel,
    temperatures: Tensor,
    pressures: Tensor,
    feeds: Tensor,
    initial_fractions: Tensor,
    initial_compositions: Tensor,
    options: GridFlashOptions,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Polish independent three-phase seeds with block-diagonal autodiff Newton."""
    component_count = feeds.shape[-1]
    log_k = torch.log(
        torch.clamp_min(
            initial_compositions[:, 1:] / initial_compositions[:, :1],
            1.0e-30,
        )
    ).reshape(feeds.shape[0], -1)
    fraction_coordinates = torch.log(
        torch.clamp_min(
            initial_fractions[:, 1:] / initial_fractions[:, :1],
            1.0e-30,
        )
    )
    variables = torch.cat((log_k, fraction_coordinates), dim=-1)
    variable_count = variables.shape[-1]
    log_k_size = 2 * component_count
    identity = torch.eye(
        variable_count,
        dtype=variables.dtype,
        device=variables.device,
    )

    for _ in range(options.three_phase_newton_iterations):
        current = variables.detach().requires_grad_(True)
        residual, _, _, _ = _batched_three_phase_quantities(
            model,
            temperatures,
            pressures,
            feeds,
            current,
        )
        norm = residual.detach().abs().amax(dim=-1)
        active = norm > options.flash_newton_tolerance
        if not bool(active.any()):
            variables = current.detach()
            break
        jacobian_rows = tuple(
            torch.autograd.grad(
                residual[:, residual_index].sum(),
                current,
                retain_graph=residual_index + 1 < variable_count,
            )[0]
            for residual_index in range(variable_count)
        )
        jacobian = torch.stack(jacobian_rows, dim=-2)
        direction, info = torch.linalg.solve_ex(
            jacobian + 1.0e-10 * identity,
            -residual[..., None],
        )
        direction = direction.squeeze(-1)
        direction = torch.where(
            (info == 0)[:, None] & torch.isfinite(direction),
            direction,
            -0.1 * residual,
        )
        direction_norm = torch.linalg.vector_norm(direction, dim=-1)
        direction = (
            direction
            * torch.clamp_max(
                8.0 / torch.clamp_min(direction_norm, 1.0),
                1.0,
            )[:, None]
        )

        accepted = ~active
        next_variables = current.detach()
        factor = torch.ones_like(norm)
        for _ in range(16):
            candidate = current.detach() + factor[:, None] * direction.detach()
            candidate = torch.cat(
                (
                    torch.clamp(candidate[:, :log_k_size], -200.0, 200.0),
                    torch.clamp(candidate[:, log_k_size:], -50.0, 50.0),
                ),
                dim=-1,
            )
            candidate_norm = (
                _batched_three_phase_quantities(
                    model,
                    temperatures,
                    pressures,
                    feeds,
                    candidate,
                )[0]
                .detach()
                .abs()
                .amax(dim=-1)
            )
            improved = active & ~accepted & torch.isfinite(candidate_norm) & (candidate_norm < norm)
            next_variables = torch.where(
                improved[:, None],
                candidate,
                next_variables,
            )
            accepted = accepted | improved
            if bool(accepted.all()):
                break
            factor = torch.where(accepted, factor, 0.5 * factor)
        variables = next_variables

    residual, fractions, compositions, gibbs = _batched_three_phase_quantities(
        model,
        temperatures,
        pressures,
        feeds,
        variables,
    )
    return (
        fractions,
        compositions,
        gibbs,
        residual.abs().amax(dim=-1),
    )


def _refine_phase_equilibrium(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    feed: Tensor,
    fractions: Tensor,
    compositions: Tensor,
    options: GridFlashOptions,
) -> tuple[Tensor, Tensor, Tensor]:
    """Solve material balance and equal fugacity from a Gibbs phase seed."""
    phase_count, component_count = compositions.shape
    if phase_count == 1:
        log_phi = model.log_fugacity_coefficients(
            temperature,
            pressure,
            feed,
            "stable",
        )
        gibbs = torch.sum(feed * (torch.log(feed) + log_phi))
        return fractions, compositions, gibbs

    log_k = torch.log(compositions[1:] / compositions[0]).reshape(-1)
    fraction_coordinates = torch.log(fractions[1:] / fractions[0])
    variables = torch.cat((log_k, fraction_coordinates))
    log_k_size = (phase_count - 1) * component_count

    for _ in range(80):
        residual, current_fractions, current_compositions, gibbs = _flash_quantities(
            model,
            temperature,
            pressure,
            feed,
            variables,
            phase_count,
        )
        residual_norm = residual.detach().abs().max()
        if float(residual_norm) <= options.flash_newton_tolerance:
            return current_fractions, current_compositions, gibbs

        jacobian = torch.func.jacrev(
            lambda current: _flash_quantities(
                model,
                temperature,
                pressure,
                feed,
                current,
                phase_count,
            )[0]
        )(variables)
        try:
            direction = torch.linalg.solve(jacobian, -residual)
        except torch.linalg.LinAlgError:
            regularization = 1.0e-10 * torch.eye(
                variables.numel(),
                dtype=variables.dtype,
                device=variables.device,
            )
            direction = torch.linalg.solve(
                jacobian.mT @ jacobian + regularization,
                -(jacobian.mT @ residual),
            )
        direction_norm = torch.linalg.vector_norm(direction)
        direction = direction * torch.clamp_max(
            direction.new_tensor(8.0) / torch.clamp_min(direction_norm, 1.0),
            1.0,
        )

        accepted = False
        factor = 1.0
        for _ in range(24):
            candidate = variables + factor * direction
            candidate = torch.cat(
                (
                    torch.clamp(candidate[:log_k_size], -200.0, 200.0),
                    torch.clamp(candidate[log_k_size:], -50.0, 50.0),
                )
            )
            candidate_residual = _flash_quantities(
                model,
                temperature,
                pressure,
                feed,
                candidate,
                phase_count,
            )[0]
            candidate_norm = candidate_residual.detach().abs().max()
            if bool(torch.isfinite(candidate_norm)) and float(candidate_norm) < float(
                residual_norm
            ):
                variables = candidate
                accepted = True
                break
            factor *= 0.5
        if not accepted:
            break

    _, current_fractions, current_compositions, gibbs = _flash_quantities(
        model,
        temperature,
        pressure,
        feed,
        variables,
        phase_count,
    )
    return current_fractions, current_compositions, gibbs


def _refine_and_reduce_candidate(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    feed: Tensor,
    initial_logits: Tensor,
    options: GridFlashOptions,
) -> tuple[Tensor, Tensor, Tensor]:
    """Newton-refine the globally optimized split and remove duplicate phases."""
    with torch.no_grad():
        candidate_gibbs, candidate_fractions, candidate_compositions = _allocation_quantities(
            model,
            temperature[None],
            pressure[None],
            feed[None, :],
            initial_logits[None, None, :, :],
        )
        gibbs = candidate_gibbs[0, 0]
        fractions = candidate_fractions[0, 0]
        compositions = candidate_compositions[0, 0]

    for _ in range(MAX_PHASES - 1):
        merged_fractions, merged_compositions = _merge_candidate_phases(
            fractions,
            compositions,
            options,
        )
        fractions, compositions, gibbs = _refine_phase_equilibrium(
            model,
            temperature,
            pressure,
            feed,
            merged_fractions,
            merged_compositions,
            options,
        )
        post_fractions, post_compositions = _merge_candidate_phases(
            fractions,
            compositions,
            options,
        )
        if post_fractions.numel() == fractions.numel():
            return gibbs, post_fractions, post_compositions
        fractions, compositions = post_fractions, post_compositions
    return gibbs, fractions, compositions


def _refine_and_reduce_candidate_robust(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    feed: Tensor,
    initial_logits: Tensor,
    options: GridFlashOptions,
) -> tuple[Tensor, Tensor, Tensor]:
    """Use strict per-state LBFGS when the fast Newton refinement fails."""
    gibbs, fractions, compositions = _refine_state_allocation(
        model,
        temperature,
        pressure,
        feed,
        initial_logits,
    )
    for _ in range(MAX_PHASES - 1):
        merged_fractions, merged_compositions = _merge_candidate_phases(
            fractions,
            compositions,
            options,
        )
        if merged_fractions.numel() == 1:
            return gibbs, merged_fractions, merged_compositions
        fractions, compositions, gibbs = _refine_phase_equilibrium(
            model,
            temperature,
            pressure,
            feed,
            merged_fractions,
            merged_compositions,
            options,
        )
        post_fractions, post_compositions = _merge_candidate_phases(
            fractions,
            compositions,
            options,
        )
        if post_fractions.numel() == fractions.numel():
            return gibbs, post_fractions, post_compositions
        if post_fractions.numel() == 1:
            return gibbs, post_fractions, post_compositions
        gibbs, fractions, compositions = _refine_state_allocation(
            model,
            temperature,
            pressure,
            feed,
            _logits_from_phases(post_fractions, post_compositions),
        )
    return gibbs, fractions, compositions


def _fugacity_residual(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    compositions: Tensor,
) -> Tensor:
    if compositions.shape[0] == 1:
        return temperature.new_zeros(())
    temperatures = temperature.expand(compositions.shape[0])
    pressures = pressure.expand(compositions.shape[0])
    chemical_potential = torch.log(compositions) + model.log_fugacity_coefficients(
        temperatures,
        pressures,
        compositions,
        "stable",
    )
    return (chemical_potential[1:] - chemical_potential[0]).abs().amax()


def solve_binary_three_phase_invariant(
    model: StateModel,
    temperature: Tensor,
    initial_pressure: Tensor,
    initial_phase_compositions: Tensor,
    *,
    phase_roots: tuple[PhaseKind, PhaseKind, PhaseKind] = (
        "liquid",
        "liquid",
        "vapor",
    ),
    tolerance: float = 1.0e-11,
    max_iterations: int = 12,
) -> BinaryThreePhaseInvariant:
    """Close a binary three-phase invariant with an autodiff Newton solve.

    The unknowns are the first-component mole fraction of each phase and the
    logarithm of pressure. Temperature is fixed. Equal component fugacities
    provide the four residual equations.

    Parameters
    ----------
    model
        Two-component homogeneous-state model supplying differentiable
        log-fugacity coefficients.
    temperature
        Scalar invariant temperature in K.
    initial_pressure
        Positive scalar pressure guess in Pa.
    initial_phase_compositions
        Strictly interior normalized guesses with shape ``(3, 2)``. The rows
        should correspond to ``phase_roots`` and should lie near the desired
        three-phase branch.
    phase_roots
        EoS roots used for the three guessed phases. The default represents
        two liquid roots and one vapor root.
    tolerance
        Maximum absolute dimensionless log-fugacity mismatch required for
        convergence.
    max_iterations
        Maximum damped Newton iterations.

    Returns
    -------
    BinaryThreePhaseInvariant
        Solved pressure, phase compositions sorted by first-component mole
        fraction, residual, iteration count, and convergence flag. A failed
        solve is returned explicitly with ``converged=False``.

    Raises
    ------
    ValueError
        If temperature or pressure is not scalar, guesses do not have shape
        ``(3, 2)``, a first-component guess is not strictly between zero and
        one, or a numerical control is not positive.
    RuntimeError
        If PyTorch cannot solve the Newton linear system or the model fails at
        a trial state.

    Notes
    -----
    :func:`torch.func.jacrev` differentiates all four fugacity residuals with
    respect to three composition logits and log pressure. A backtracking line
    search accepts only a decrease in squared residual norm. This is a local
    branch solve: multiple invariants or a poor initial branch guess are not
    discovered automatically.
    """
    if temperature.ndim != 0 or initial_pressure.ndim != 0:
        raise ValueError("binary invariant temperature and pressure must be scalar")
    if initial_phase_compositions.shape != (MAX_PHASES, 2):
        raise ValueError("binary invariant requires three two-component phase guesses")
    if tolerance <= 0.0 or max_iterations <= 0:
        raise ValueError("binary invariant tolerance and max_iterations must be positive")
    first_fractions = initial_phase_compositions[:, 0]
    if bool(((first_fractions <= 0.0) | (first_fractions >= 1.0)).any()):
        raise ValueError("binary invariant phase guesses must be strictly interior")

    def residual(variables: Tensor) -> Tensor:
        component_fractions = torch.sigmoid(variables[:MAX_PHASES])
        compositions = torch.stack(
            (component_fractions, 1.0 - component_fractions),
            dim=-1,
        )
        pressure = torch.exp(variables[MAX_PHASES])
        chemical_potentials = torch.stack(
            tuple(
                torch.log(compositions[index])
                + model.log_fugacity_coefficients(
                    temperature,
                    pressure,
                    compositions[index],
                    phase_roots[index],
                )
                for index in range(MAX_PHASES)
            )
        )
        return torch.cat(
            (
                chemical_potentials[1] - chemical_potentials[0],
                chemical_potentials[2] - chemical_potentials[0],
            )
        )

    variables = torch.cat((torch.logit(first_fractions), torch.log(initial_pressure).reshape(1)))
    converged = False
    for _iteration in range(1, max_iterations + 1):
        current_residual = residual(variables)
        if float(current_residual.detach().abs().max()) <= tolerance:
            converged = True
            break
        step = torch.linalg.solve(
            torch.func.jacrev(residual)(variables),
            -current_residual,
        )
        baseline = float(current_residual.detach().square().sum())
        for line_search in range(14):
            trial = variables + (0.5**line_search) * step
            if float(residual(trial).detach().square().sum()) < baseline:
                variables = trial
                break
        else:
            break

    final_residual = residual(variables)
    converged = bool(final_residual.detach().abs().max() <= tolerance)
    first_fractions = torch.sigmoid(variables[:MAX_PHASES])
    order = torch.argsort(first_fractions)
    phase_compositions = torch.stack(
        (first_fractions[order], 1.0 - first_fractions[order]),
        dim=-1,
    )
    return BinaryThreePhaseInvariant(
        temperature=temperature,
        pressure=torch.exp(variables[MAX_PHASES]),
        phase_compositions=phase_compositions,
        residual_norm=final_residual.abs().max(),
        iterations=_iteration,
        converged=converged,
    )


def _binary_invariant_split(
    invariants: tuple[BinaryThreePhaseInvariant, ...],
    temperature: Tensor,
    pressure: Tensor,
    feed: Tensor,
    options: GridFlashOptions,
) -> tuple[Tensor, Tensor] | None:
    """Resolve a supplied binary invariant with a centered positive lever rule."""
    if feed.numel() != 2:
        return None
    for invariant in invariants:
        if not invariant.converged:
            continue
        state_matches = bool(
            torch.isclose(temperature, invariant.temperature, rtol=1.0e-10, atol=1.0e-8)
            & torch.isclose(pressure, invariant.pressure, rtol=1.0e-10, atol=1.0e-3)
        )
        if not state_matches:
            continue
        phase_coordinate = invariant.phase_compositions[:, 0]
        feed_coordinate = feed[0]
        tolerance = options.phase_fraction_tolerance
        if not bool(
            (feed_coordinate > phase_coordinate[0] + tolerance)
            & (feed_coordinate < phase_coordinate[2] - tolerance)
        ):
            continue
        middle_maximum = torch.minimum(
            (phase_coordinate[2] - feed_coordinate) / (phase_coordinate[2] - phase_coordinate[1]),
            (feed_coordinate - phase_coordinate[0]) / (phase_coordinate[1] - phase_coordinate[0]),
        )
        middle_fraction = 0.5 * middle_maximum
        first_fraction = (
            phase_coordinate[2]
            - feed_coordinate
            - (phase_coordinate[2] - phase_coordinate[1]) * middle_fraction
        ) / (phase_coordinate[2] - phase_coordinate[0])
        fractions = torch.stack(
            (
                first_fraction,
                middle_fraction,
                1.0 - first_fraction - middle_fraction,
            )
        )
        if bool(torch.all(fractions > tolerance)):
            return fractions, invariant.phase_compositions
    return None


def _candidate_diagnostics(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    feed: Tensor,
    fractions: Tensor,
    compositions: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize a candidate and return balance and fugacity residuals."""
    normalized_fractions = fractions / fractions.sum()
    balance = torch.sum(normalized_fractions[:, None] * compositions, dim=0)
    balance_residual = torch.max(torch.abs(balance - feed))
    equilibrium_residual = _fugacity_residual(
        model,
        temperature,
        pressure,
        compositions,
    )
    return normalized_fractions, balance_residual, equilibrium_residual


def _candidate_gibbs_energy(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    fractions: Tensor,
    compositions: Tensor,
) -> Tensor:
    """Evaluate the total reduced Gibbs energy of a normalized split."""
    phase_count = compositions.shape[0]
    log_phi = model.log_fugacity_coefficients(
        temperature.expand(phase_count),
        pressure.expand(phase_count),
        compositions,
        "stable",
    )
    phase_gibbs = torch.sum(
        compositions * (torch.log(compositions) + log_phi),
        dim=-1,
    )
    return torch.sum(fractions * phase_gibbs)


def _bracketed_lower_phase_indices(
    phase_counts: Tensor,
    vertical_count: int,
    horizontal_count: int,
) -> list[int]:
    """Find lower-phase-count cells enclosed by higher-count grid states.

    This is an audit trigger, not a phase-topology assumption. Every proposed
    replacement must still have lower Gibbs energy and pass the independent
    fugacity and material-balance gates. Searching both complete row and
    column rays also catches runs of adjacent missed cells rather than only
    isolated single-pixel holes.
    """
    counts = phase_counts.reshape(vertical_count, horizontal_count)
    indices: list[int] = []
    for vertical_index in range(vertical_count):
        for horizontal_index in range(horizontal_count):
            current_count = int(counts[vertical_index, horizontal_index])
            if current_count >= MAX_PHASES:
                continue
            if current_count == 1:
                horizontally_bracketed = (
                    0 < horizontal_index < horizontal_count - 1
                    and int(counts[vertical_index, horizontal_index - 1]) > 1
                    and int(counts[vertical_index, horizontal_index + 1]) > 1
                )
                vertically_bracketed = (
                    0 < vertical_index < vertical_count - 1
                    and int(counts[vertical_index - 1, horizontal_index]) > 1
                    and int(counts[vertical_index + 1, horizontal_index]) > 1
                )
            else:
                horizontally_bracketed = (
                    0 < horizontal_index < horizontal_count - 1
                    and int(counts[vertical_index, :horizontal_index].max()) == MAX_PHASES
                    and int(counts[vertical_index, horizontal_index + 1 :].max()) == MAX_PHASES
                )
                vertically_bracketed = (
                    0 < vertical_index < vertical_count - 1
                    and int(counts[:vertical_index, horizontal_index].max()) == MAX_PHASES
                    and int(counts[vertical_index + 1 :, horizontal_index].max()) == MAX_PHASES
                )
            if horizontally_bracketed or vertically_bracketed:
                indices.append(vertical_index * horizontal_count + horizontal_index)
    return indices


def _continuation_seed_logits(
    phase_counts: Tensor,
    phase_fractions: Tensor,
    phase_compositions: Tensor,
    target_indices: list[int],
    vertical_count: int,
    horizontal_count: int,
) -> dict[int, Tensor]:
    """Seed missed lower-count cells from the nearest resolved three-phase state."""
    counts = phase_counts.reshape(vertical_count, horizontal_count)
    seeds: dict[int, Tensor] = {}
    for target_index in target_indices:
        vertical_index, horizontal_index = divmod(target_index, horizontal_count)
        if int(counts[vertical_index, horizontal_index]) != MAX_PHASES - 1:
            continue
        candidates: list[tuple[int, int, int]] = []
        for candidate_horizontal in range(horizontal_count):
            if int(counts[vertical_index, candidate_horizontal]) == MAX_PHASES:
                candidate_index = vertical_index * horizontal_count + candidate_horizontal
                # At fixed T and P, three-phase compositions do not depend on
                # the overall feed while phase fractions do. Prefer a seed on
                # the same pressure row, then its closest composition.
                candidates.append(
                    (
                        0,
                        abs(candidate_horizontal - horizontal_index),
                        candidate_index,
                    )
                )
        for candidate_vertical in range(vertical_count):
            if int(counts[candidate_vertical, horizontal_index]) == MAX_PHASES:
                candidate_index = candidate_vertical * horizontal_count + horizontal_index
                candidates.append(
                    (
                        1,
                        abs(candidate_vertical - vertical_index),
                        candidate_index,
                    )
                )
        if not candidates:
            continue
        _, _, seed_index = min(candidates)
        seeds[target_index] = _logits_from_phases(
            phase_fractions[seed_index, :MAX_PHASES],
            phase_compositions[seed_index, :MAX_PHASES],
        )
    return seeds


def _independent_multistart_reflash(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    feed: Tensor,
    optimized_logits: Tensor,
    one_phase_gibbs: Tensor,
    options: GridFlashOptions,
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Independently optimize the best starts for a suspicious one-phase cell."""
    with torch.no_grad():
        candidate_gibbs, _, _ = _allocation_quantities(
            model,
            temperature[None],
            pressure[None],
            feed[None, :],
            optimized_logits[None, :, :, :],
        )
        start_order = torch.argsort(candidate_gibbs[0])[: options.independent_reflash_starts]

    best_energy = one_phase_gibbs
    best_result: tuple[Tensor, Tensor, Tensor] | None = None
    for start_index in start_order:
        _, fractions, compositions = _refine_and_reduce_candidate_robust(
            model,
            temperature,
            pressure,
            feed,
            optimized_logits[start_index],
            options,
        )
        fractions, balance_residual, equilibrium_residual = _candidate_diagnostics(
            model,
            temperature,
            pressure,
            feed,
            fractions,
            compositions,
        )
        energy = _candidate_gibbs_energy(
            model,
            temperature,
            pressure,
            fractions,
            compositions,
        )
        valid = bool(
            (equilibrium_residual <= options.fugacity_tolerance)
            & (balance_residual <= options.material_balance_tolerance)
        )
        if (
            valid
            and fractions.numel() > 1
            and bool(energy < best_energy - options.gibbs_reduction_tolerance)
        ):
            best_energy = energy
            best_result = fractions, compositions, energy
    return best_result


def flash_grid_oracle(
    model: StateModel,
    state: ChemicalState,
    *,
    options: GridFlashOptions | None = None,
    binary_invariants: tuple[BinaryThreePhaseInvariant, ...] = (),
) -> GridEquilibrium:
    """Independently apply multistart Gibbs minimization to every TP state.

    This deliberately slower path bypasses stability and two-phase screening.
    It is intended as a small-batch correctness oracle for :func:`flash_grid`.

    Parameters
    ----------
    model
        Homogeneous-state model supplying differentiable log-fugacity
        coefficients and Wilson critical-property initialization data.
    state
        Batched TP states. Temperature is in K, pressure is in Pa, and
        composition contains mole fractions on the final axis. Temperature
        and pressure must share one non-scalar batch shape; composition can be
        common or have matching leading dimensions.
    options
        Numerical controls. ``independent_reflash_starts`` bounds how many
        lowest initial Gibbs allocations are strictly refined per state.
    binary_invariants
        Optional converged binary invariant solutions. A matching temperature
        and pressure uses their fixed phase compositions and a positive
        centered lever rule.

    Returns
    -------
    GridEquilibrium
        Padded one-, two-, or three-phase results with explicit fugacity,
        material-balance, Gibbs-reduction, convergence, and timing diagnostics.

    Raises
    ------
    ValueError
        If batch shapes, compositions, options, or a supplied invariant are
        invalid.
    RuntimeError
        If a model evaluation or strict scalar refinement fails.

    Notes
    -----
    Every multicomponent state is optimized independently from the same
    deterministic and seeded-random start library. The oracle is algorithmic
    verification, not an independent physical model. Compare phase count,
    Gibbs energy, and residuals; agreement alone does not validate the EoS
    parameterization.

    Candidate minimization and equal-fugacity Newton Jacobians use PyTorch
    autodiff. Phase selection and padding are discrete, so the returned record
    is not an end-to-end differentiable flash result.
    """
    resolved_options = GridFlashOptions() if options is None else options
    temperatures, pressures, feeds = _grid_states(state)
    grid_shape = tuple(state.temperature.shape)
    state_count, component_count = feeds.shape
    started = time.perf_counter()
    padded_fractions = temperatures.new_zeros((state_count, MAX_PHASES))
    padded_compositions = temperatures.new_full(
        (state_count, MAX_PHASES, component_count),
        torch.nan,
    )
    phase_counts = torch.ones(state_count, dtype=torch.int64, device=feeds.device)
    gibbs_reduction = temperatures.new_zeros(state_count)
    fugacity_residual = temperatures.new_zeros(state_count)
    material_balance_residual = temperatures.new_zeros(state_count)
    converged = torch.ones(state_count, dtype=torch.bool, device=feeds.device)

    if component_count == 1:
        padded_fractions[:, 0] = 1.0
        padded_compositions[:, 0] = feeds
        return GridEquilibrium(
            temperatures,
            pressures,
            feeds,
            grid_shape,
            padded_fractions,
            padded_compositions,
            phase_counts,
            gibbs_reduction,
            fugacity_residual,
            material_balance_residual,
            converged,
            time.perf_counter() - started,
            0.0,
            0.0,
            0,
            0,
            0,
            0,
        )

    with torch.no_grad():
        one_phase_log_phi = model.log_fugacity_coefficients(
            temperatures,
            pressures,
            feeds,
            "stable",
        )
        one_phase_gibbs = torch.sum(
            feeds * (torch.log(feeds) + one_phase_log_phi),
            dim=-1,
        )
        multistart_logits = _allocation_initial_logits(
            model,
            temperatures,
            pressures,
            feeds,
            resolved_options,
        )
    batched_search_seconds = time.perf_counter() - started
    refinement_started = time.perf_counter()
    audited = 0
    replacements = 0
    for state_index in range(state_count):
        invariant_split = _binary_invariant_split(
            binary_invariants,
            temperatures[state_index],
            pressures[state_index],
            feeds[state_index],
            resolved_options,
        )
        if invariant_split is None:
            audited += 1
            replacement = _independent_multistart_reflash(
                model,
                temperatures[state_index],
                pressures[state_index],
                feeds[state_index],
                multistart_logits[state_index],
                one_phase_gibbs[state_index],
                resolved_options,
            )
            if replacement is None:
                fractions = temperatures.new_ones(1)
                compositions = feeds[state_index][None, :]
                split_gibbs = one_phase_gibbs[state_index]
            else:
                fractions, compositions, split_gibbs = replacement
                replacements += 1
        else:
            fractions, compositions = invariant_split
            split_gibbs = _candidate_gibbs_energy(
                model,
                temperatures[state_index],
                pressures[state_index],
                fractions,
                compositions,
            )

        fractions, balance_residual, equilibrium_residual = _candidate_diagnostics(
            model,
            temperatures[state_index],
            pressures[state_index],
            feeds[state_index],
            fractions,
            compositions,
        )
        count = fractions.numel()
        padded_fractions[state_index, :count] = fractions
        padded_compositions[state_index, :count] = compositions
        phase_counts[state_index] = count
        gibbs_reduction[state_index] = torch.clamp_min(
            one_phase_gibbs[state_index] - split_gibbs,
            0.0,
        )
        fugacity_residual[state_index] = equilibrium_residual
        material_balance_residual[state_index] = balance_residual
        converged[state_index] = bool(
            (equilibrium_residual <= resolved_options.fugacity_tolerance)
            & (balance_residual <= resolved_options.material_balance_tolerance)
        )

    refinement_seconds = time.perf_counter() - refinement_started
    return GridEquilibrium(
        temperatures,
        pressures,
        feeds,
        grid_shape,
        padded_fractions,
        padded_compositions,
        phase_counts,
        gibbs_reduction,
        fugacity_residual,
        material_balance_residual,
        converged,
        time.perf_counter() - started,
        batched_search_seconds,
        refinement_seconds,
        0,
        audited,
        replacements,
        0,
    )


def _gibbs_fallback_grid_states(
    model: StateModel,
    temperatures: Tensor,
    pressures: Tensor,
    feeds: Tensor,
    state_indices: list[int],
    one_phase_gibbs: Tensor,
    current_gibbs: Tensor,
    current_converged: Tensor,
    options: GridFlashOptions,
    binary_invariants: tuple[BinaryThreePhaseInvariant, ...],
    seed_logits: dict[int, Tensor] | None = None,
) -> dict[int, tuple[Tensor, Tensor, Tensor]]:
    """Discover up to three phases for a sparse difficult-state subset."""
    if not state_indices:
        return {}
    indices = torch.tensor(
        sorted(set(state_indices)),
        dtype=torch.long,
        device=feeds.device,
    )
    subset_temperature = temperatures[indices]
    subset_pressure = pressures[indices]
    subset_feed = feeds[indices]
    initial_logits = _allocation_initial_logits(
        model,
        subset_temperature,
        subset_pressure,
        subset_feed,
        options,
    )
    if seed_logits:
        targeted = initial_logits[:, 0].clone()
        for local_index, global_index_tensor in enumerate(indices):
            global_index = int(global_index_tensor)
            if global_index in seed_logits:
                targeted[local_index] = seed_logits[global_index]
        initial_logits = torch.cat((initial_logits, targeted[:, None]), dim=1)
    logits = torch.nn.Parameter(initial_logits)
    optimizer = torch.optim.Adam(
        (logits,),
        lr=0.08,
    )
    for _ in range(options.gibbs_fallback_adam_iterations):
        optimizer.zero_grad(set_to_none=True)
        candidate_gibbs, _, _ = _allocation_quantities(
            model,
            subset_temperature,
            subset_pressure,
            subset_feed,
            logits,
        )
        loss = candidate_gibbs.sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        (
            candidate_gibbs,
            candidate_fractions,
            candidate_compositions,
        ) = _allocation_quantities(
            model,
            subset_temperature,
            subset_pressure,
            subset_feed,
            logits,
        )
        best_start = candidate_gibbs.argmin(dim=1)
        subset_index = torch.arange(indices.numel(), device=feeds.device)
        best_candidate_gibbs = candidate_gibbs[subset_index, best_start]
        best_logits = logits.detach()[subset_index, best_start]
        best_fractions = candidate_fractions[subset_index, best_start]
        best_compositions = candidate_compositions[subset_index, best_start]

    (
        batched_fractions,
        batched_compositions,
        batched_gibbs,
        batched_newton_residual,
    ) = _batched_refine_three_phase(
        model,
        subset_temperature,
        subset_pressure,
        subset_feed,
        best_fractions,
        best_compositions,
        options,
    )
    if options.debug:
        merged_seed_counts = [
            int(
                _merge_candidate_phases(
                    best_fractions[index],
                    best_compositions[index],
                    options,
                )[0].numel()
            )
            for index in range(indices.numel())
        ]
        print(
            "Batched three-phase Newton:",
            int((batched_newton_residual <= options.fugacity_tolerance).sum()),
            "/",
            indices.numel(),
            "within the fugacity tolerance",
            "; Adam seed phase counts:",
            {count: merged_seed_counts.count(count) for count in sorted(set(merged_seed_counts))},
            flush=True,
        )

    pair_distance = torch.stack(
        (
            torch.abs(batched_compositions[:, 0] - batched_compositions[:, 1]).amax(dim=-1),
            torch.abs(batched_compositions[:, 0] - batched_compositions[:, 2]).amax(dim=-1),
            torch.abs(batched_compositions[:, 1] - batched_compositions[:, 2]).amax(dim=-1),
        ),
        dim=-1,
    )
    batched_balance_residual = torch.abs(
        torch.sum(batched_fractions[..., None] * batched_compositions, dim=1) - subset_feed
    ).amax(dim=-1)
    batched_reduction = one_phase_gibbs[indices] - batched_gibbs
    invariant_results: dict[
        int,
        tuple[Tensor, Tensor, Tensor],
    ] = {}
    invariant_split = torch.zeros_like(batched_newton_residual, dtype=torch.bool)
    for local_index, global_index_tensor in enumerate(indices):
        result = _binary_invariant_split(
            binary_invariants,
            subset_temperature[local_index],
            subset_pressure[local_index],
            subset_feed[local_index],
            options,
        )
        if result is not None:
            fractions, compositions = result
            invariant_split[local_index] = True
            invariant_results[int(global_index_tensor)] = (
                fractions,
                compositions,
                _candidate_gibbs_energy(
                    model,
                    subset_temperature[local_index],
                    subset_pressure[local_index],
                    fractions,
                    compositions,
                ),
            )

    direct_three_phase = (
        (batched_newton_residual <= options.fugacity_tolerance)
        & (batched_balance_residual <= options.material_balance_tolerance)
        & (batched_fractions.amin(dim=-1) > options.phase_fraction_tolerance)
        & (pair_distance.amin(dim=-1) > options.phase_composition_merge_tolerance)
        & (batched_reduction > options.gibbs_reduction_tolerance)
        & torch.isfinite(batched_gibbs)
        & torch.isfinite(batched_compositions).all(dim=(-1, -2))
        & ~invariant_split
    )
    promising_candidate = (
        best_candidate_gibbs < current_gibbs[indices] - options.gibbs_reduction_tolerance
    )
    requires_failure_resolution = ~current_converged[indices]
    resolved_without_scalar = direct_three_phase | invariant_split
    scalar_refinement = ~resolved_without_scalar & (
        promising_candidate | requires_failure_resolution
    )
    if options.debug:
        print(
            "Fallback candidates:",
            int(resolved_without_scalar.sum()),
            "accepted in batch,",
            int(scalar_refinement.sum()),
            "sent to scalar refinement,",
            int((~resolved_without_scalar & ~scalar_refinement).sum()),
            "rejected without a lower Gibbs candidate",
            flush=True,
        )
    direct_results = {
        int(indices[local_index]): (
            batched_fractions[local_index].detach(),
            batched_compositions[local_index].detach(),
            batched_gibbs[local_index].detach(),
        )
        for local_index in torch.nonzero(direct_three_phase).flatten().tolist()
    }
    direct_results.update(invariant_results)
    alternative_two_phase = torch.zeros_like(direct_three_phase)
    alternative_results: dict[
        int,
        tuple[Tensor, Tensor, Tensor],
    ] = {}
    alternative_local_indices = torch.nonzero(scalar_refinement).flatten()
    if alternative_local_indices.numel():
        two_phase_seed_compositions = []
        for local_index in alternative_local_indices.tolist():
            seed_fractions, seed_compositions = _merge_candidate_phases(
                best_fractions[local_index],
                best_compositions[local_index],
                options,
            )
            if seed_fractions.numel() == MAX_PHASES:
                distances = torch.stack(
                    (
                        torch.abs(seed_compositions[0] - seed_compositions[1]).amax(),
                        torch.abs(seed_compositions[0] - seed_compositions[2]).amax(),
                        torch.abs(seed_compositions[1] - seed_compositions[2]).amax(),
                    )
                )
                first, second = ((0, 1), (0, 2), (1, 2))[int(distances.argmin())]
                retained = 3 - first - second
                merged_fraction = seed_fractions[first] + seed_fractions[second]
                merged_composition = (
                    seed_fractions[first] * seed_compositions[first]
                    + seed_fractions[second] * seed_compositions[second]
                ) / merged_fraction
                seed_fractions = torch.stack(
                    (
                        merged_fraction,
                        seed_fractions[retained],
                    )
                )
                seed_compositions = torch.stack(
                    (
                        merged_composition,
                        seed_compositions[retained],
                    )
                )
            if seed_fractions.numel() != 2:
                two_phase_seed_compositions.append(
                    torch.stack((subset_feed[local_index], subset_feed[local_index]))
                )
            else:
                two_phase_seed_compositions.append(seed_compositions)

        two_phase_seed = torch.stack(two_phase_seed_compositions)
        trial_k = two_phase_seed[:, 1] / torch.clamp_min(two_phase_seed[:, 0], 1.0e-30)
        straddles = (trial_k.amin(dim=-1) < 1.0) & (trial_k.amax(dim=-1) > 1.0)
        if bool(straddles.any()):
            trial_local = alternative_local_indices[straddles]
            alternative_flash = _batched_two_phase_in_chunks(
                model,
                subset_temperature[trial_local],
                subset_pressure[trial_local],
                subset_feed[trial_local],
                trial_k[straddles],
                options,
            )
            alternative_fractions = torch.stack(
                (
                    alternative_flash.liquid_fraction,
                    alternative_flash.vapor_fraction,
                ),
                dim=-1,
            )
            alternative_compositions = torch.stack(
                (
                    alternative_flash.liquid_composition,
                    alternative_flash.vapor_composition,
                ),
                dim=1,
            )
            with torch.no_grad():
                alternative_log_phi = model.log_fugacity_coefficients(
                    subset_temperature[trial_local, None],
                    subset_pressure[trial_local, None],
                    alternative_compositions,
                    "stable",
                )
                alternative_phase_gibbs = torch.sum(
                    alternative_compositions
                    * (torch.log(alternative_compositions) + alternative_log_phi),
                    dim=-1,
                )
                alternative_gibbs = torch.sum(
                    alternative_fractions * alternative_phase_gibbs,
                    dim=-1,
                )
                alternative_balance = torch.abs(
                    torch.sum(
                        alternative_fractions[..., None] * alternative_compositions,
                        dim=1,
                    )
                    - subset_feed[trial_local]
                ).amax(dim=-1)
                alternative_distance = torch.abs(
                    alternative_compositions[:, 0] - alternative_compositions[:, 1]
                ).amax(dim=-1)
                acceptable = (
                    alternative_flash.converged
                    & (alternative_balance <= options.material_balance_tolerance)
                    & (alternative_fractions.amin(dim=-1) > options.phase_fraction_tolerance)
                    & (alternative_distance > options.phase_composition_merge_tolerance)
                    & (
                        (
                            alternative_gibbs
                            < current_gibbs[indices[trial_local]]
                            - options.gibbs_reduction_tolerance
                        )
                        | requires_failure_resolution[trial_local]
                    )
                )
            for candidate_index in torch.nonzero(acceptable).flatten().tolist():
                local_index = int(trial_local[candidate_index])
                global_index = int(indices[local_index])
                alternative_two_phase[local_index] = True
                alternative_results[global_index] = (
                    alternative_fractions[candidate_index].detach(),
                    alternative_compositions[candidate_index].detach(),
                    alternative_gibbs[candidate_index].detach(),
                )
    direct_results.update(alternative_results)

    def solve_candidate(
        item: tuple[int, Tensor],
    ) -> tuple[int, tuple[Tensor, Tensor, Tensor] | None]:
        local_index, global_index_tensor = item
        global_index = int(global_index_tensor)
        fractions, compositions = _merge_candidate_phases(
            batched_fractions[local_index],
            batched_compositions[local_index],
            options,
        )
        fractions, balance_residual, equilibrium_residual = _candidate_diagnostics(
            model,
            subset_temperature[local_index],
            subset_pressure[local_index],
            subset_feed[local_index],
            fractions,
            compositions,
        )
        if bool(
            (equilibrium_residual > options.fugacity_tolerance)
            | (balance_residual > options.material_balance_tolerance)
        ):
            _, fractions, compositions = _refine_and_reduce_candidate(
                model,
                subset_temperature[local_index],
                subset_pressure[local_index],
                subset_feed[local_index],
                best_logits[local_index],
                options,
            )
            fractions, balance_residual, equilibrium_residual = _candidate_diagnostics(
                model,
                subset_temperature[local_index],
                subset_pressure[local_index],
                subset_feed[local_index],
                fractions,
                compositions,
            )
        if bool(
            (equilibrium_residual > options.fugacity_tolerance)
            | (balance_residual > options.material_balance_tolerance)
        ):
            _, fractions, compositions = _refine_and_reduce_candidate_robust(
                model,
                subset_temperature[local_index],
                subset_pressure[local_index],
                subset_feed[local_index],
                best_logits[local_index],
                options,
            )
            fractions, balance_residual, equilibrium_residual = _candidate_diagnostics(
                model,
                subset_temperature[local_index],
                subset_pressure[local_index],
                subset_feed[local_index],
                fractions,
                compositions,
            )
        split_gibbs = _candidate_gibbs_energy(
            model,
            subset_temperature[local_index],
            subset_pressure[local_index],
            fractions,
            compositions,
        )
        reduction = one_phase_gibbs[global_index] - split_gibbs
        valid = bool(
            (equilibrium_residual <= options.fugacity_tolerance)
            & (balance_residual <= options.material_balance_tolerance)
        )
        if valid and (
            fractions.numel() == 1 or bool(reduction > options.gibbs_reduction_tolerance)
        ):
            return global_index, (fractions, compositions, split_gibbs)
        return global_index, None

    items = [
        (local_index, global_index)
        for local_index, global_index in enumerate(indices)
        if bool(scalar_refinement[local_index] & ~alternative_two_phase[local_index])
    ]
    if not items:
        return direct_results
    if options.fallback_workers == 1 or len(items) == 1:
        solved = list(map(solve_candidate, items))
    else:
        with ThreadPoolExecutor(
            max_workers=min(options.fallback_workers, len(items)),
            thread_name_prefix="torch-flash-grid",
        ) as executor:
            solved = list(executor.map(solve_candidate, items))
    direct_results.update(
        {global_index: result for global_index, result in solved if result is not None}
    )
    return direct_results


def flash_grid(
    model: StateModel,
    state: ChemicalState,
    *,
    options: GridFlashOptions | None = None,
    binary_invariants: tuple[BinaryThreePhaseInvariant, ...] = (),
) -> GridEquilibrium:
    """Flash independent states with batched screening and sparse Gibbs fallback.

    The temperature and pressure tensors must have the same non-scalar batch
    shape. The composition can be common to the batch or have that batch shape
    as leading dimensions. Two-dimensional batches additionally receive a
    topology audit that independently reflashes isolated lower-phase cells.

    Parameters
    ----------
    model
        Homogeneous-state thermodynamic model supplying differentiable
        log-fugacity coefficients, cubic critical constants for Wilson starts,
        and stable-root selection.
    state
        Batched TP states. Temperature is in K, pressure is in Pa, and
        composition contains mole fractions on the final axis. Temperature
        and pressure must have the same non-scalar batch shape. Composition
        can be common to all states or have matching leading dimensions.
    options
        Numerical controls. Defaults target float64 reference calculations.
        The model, state, and all options are reused across the batch without
        changing process-wide PyTorch runtime settings.
    binary_invariants
        Optional converged binary three-phase solutions. A matching temperature
        and pressure uses their fixed phase compositions and a positive
        centered lever rule for feeds inside the outer composition interval.

    Returns
    -------
    GridEquilibrium
        Padded phase fractions, compositions, convergence diagnostics, and
        timing counters flattened in row-major order with ``grid_shape``
        recording the original batch shape.

    Raises
    ------
    ValueError
        If temperature and pressure are scalar or have different shapes, the
        composition batch does not match, a numerical option is invalid, or a
        supplied binary invariant is malformed.
    RuntimeError
        If model evaluation, an optimizer, or an autodiff Newton system fails.

    Notes
    -----
    The hierarchy is:

    1. batched tangent-plane stability screening;
    2. a batched known-two-phase flash for unstable feeds;
    3. stability screening of both returned phases;
    4. sparse three-phase Gibbs-allocation minimization and autodiff Newton
       refinement for failed or child-unstable states; and
    5. for two-dimensional batches, an independent reflash of lower-phase
       cells bracketed by higher-phase states.

    The topology audit only proposes candidates. A replacement still requires
    lower Gibbs energy plus fugacity and material-balance convergence; visual
    smoothness never overrides the thermodynamic gates.

    Forward-only screening runs without graph retention. Gibbs minimization
    uses PyTorch autograd and Newton Jacobians use ``torch.func.jacrev``.
    Phase-count decisions, merging, and padded output assembly are discrete,
    so call-level gradients through ``GridEquilibrium`` are not defined. Use
    homogeneous property APIs and :func:`identify_phase` when a differentiable
    diagnostic is required.

    ``chunk_size`` limits memory but does not introduce thermodynamic coupling.
    The states remain independent. On CPU, configure dtype and thread count
    once before constructing the model; library calls never change global
    PyTorch thread settings.
    """
    resolved_options = options or GridFlashOptions()
    temperatures, pressures, feeds = _grid_states(state)
    grid_shape = tuple(state.temperature.shape)
    state_count, component_count = feeds.shape
    started = time.perf_counter()
    if component_count == 1:
        return flash_grid_oracle(
            model,
            state,
            options=resolved_options,
            binary_invariants=binary_invariants,
        )

    with torch.no_grad():
        one_phase_log_phi = model.log_fugacity_coefficients(
            temperatures,
            pressures,
            feeds,
            "stable",
        )
        one_phase_gibbs = torch.sum(
            feeds * (torch.log(feeds) + one_phase_log_phi),
            dim=-1,
        )
    padded_fractions = temperatures.new_zeros((state_count, MAX_PHASES))
    padded_fractions[:, 0] = 1.0
    padded_compositions = temperatures.new_full(
        (state_count, MAX_PHASES, component_count),
        torch.nan,
    )
    padded_compositions[:, 0, :] = feeds
    phase_counts = torch.ones(
        state_count,
        dtype=torch.int64,
        device=feeds.device,
    )
    gibbs_reduction = temperatures.new_zeros(state_count)
    fugacity_residual = temperatures.new_zeros(state_count)
    material_balance_residual = temperatures.new_zeros(state_count)
    converged = torch.ones(
        state_count,
        dtype=torch.bool,
        device=feeds.device,
    )

    stability = _batched_stability_in_chunks(
        model,
        temperatures,
        pressures,
        feeds,
        resolved_options,
    )
    unstable_indices = torch.nonzero(
        stability.minimum_tpd < -resolved_options.stability_tolerance,
    ).flatten()
    difficult_indices: set[int] = set()
    fallback_seed_logits: dict[int, Tensor] = {}

    if unstable_indices.numel():
        converged[unstable_indices] = False
        initial_k = stability.trial_composition[unstable_indices] / torch.clamp_min(
            feeds[unstable_indices],
            1.0e-30,
        )
        two_phase = _batched_two_phase_in_chunks(
            model,
            temperatures[unstable_indices],
            pressures[unstable_indices],
            feeds[unstable_indices],
            initial_k,
            resolved_options,
        )
        converged_two_phase = two_phase.converged
        failed_local = torch.nonzero(~converged_two_phase).flatten()
        difficult_indices.update(int(index) for index in unstable_indices[failed_local])

        if bool(converged_two_phase.any()):
            local = torch.nonzero(converged_two_phase).flatten()
            global_indices = unstable_indices[local]
            fractions = torch.stack(
                (
                    two_phase.liquid_fraction[local],
                    two_phase.vapor_fraction[local],
                ),
                dim=-1,
            )
            compositions = torch.stack(
                (
                    two_phase.liquid_composition[local],
                    two_phase.vapor_composition[local],
                ),
                dim=1,
            )
            with torch.no_grad():
                phase_log_phi = model.log_fugacity_coefficients(
                    temperatures[global_indices, None],
                    pressures[global_indices, None],
                    compositions,
                    "stable",
                )
                phase_gibbs = torch.sum(
                    compositions * (torch.log(compositions) + phase_log_phi),
                    dim=-1,
                )
                split_gibbs = torch.sum(fractions * phase_gibbs, dim=-1)
            reduction = one_phase_gibbs[global_indices] - split_gibbs
            balance = torch.sum(fractions[..., None] * compositions, dim=1)
            balance_residual = torch.abs(
                balance - feeds[global_indices],
            ).amax(dim=-1)
            composition_distance = torch.abs(
                compositions[:, 0] - compositions[:, 1],
            ).amax(dim=-1)
            active = (
                (fractions.amin(dim=-1) > resolved_options.phase_fraction_tolerance)
                & (composition_distance > resolved_options.phase_composition_merge_tolerance)
                & (reduction > resolved_options.gibbs_reduction_tolerance)
                & (balance_residual <= resolved_options.material_balance_tolerance)
            )
            converged[global_indices] = True
            fugacity_residual[global_indices] = two_phase.residual_norm[local]
            material_balance_residual[global_indices] = balance_residual
            active_local = torch.nonzero(active).flatten()
            active_split_indices = global_indices[active_local]
            if active_split_indices.numel():
                active_fractions = fractions[active_local]
                active_compositions = compositions[active_local]
                padded_fractions[active_split_indices, :2] = active_fractions
                padded_compositions[active_split_indices, :2] = active_compositions
                phase_counts[active_split_indices] = 2
                gibbs_reduction[active_split_indices] = reduction[active_local]

                phase_feed = torch.cat(
                    (
                        active_compositions[:, 0],
                        active_compositions[:, 1],
                    ),
                    dim=0,
                )
                phase_temperature = temperatures[active_split_indices].repeat(2)
                phase_pressure = pressures[active_split_indices].repeat(2)
                phase_stability = _batched_stability_in_chunks(
                    model,
                    phase_temperature,
                    phase_pressure,
                    phase_feed,
                    resolved_options,
                )
                child_tpd = phase_stability.minimum_tpd.reshape(2, -1)
                child_composition = phase_stability.trial_composition.reshape(
                    2,
                    -1,
                    component_count,
                )
                child_unstable = (child_tpd < -resolved_options.stability_tolerance).any(dim=0)
                child_candidates = torch.nonzero(child_unstable).flatten()
                difficult_indices.update(
                    int(index) for index in active_split_indices[child_candidates]
                )
                for child_index_tensor in child_candidates:
                    child_index = int(child_index_tensor)
                    unstable_phase = int(torch.argmin(child_tpd[:, child_index]))
                    seed_fractions = torch.cat(
                        (
                            0.95 * active_fractions[child_index],
                            active_fractions.new_tensor([0.05]),
                        )
                    )
                    seed_compositions = torch.cat(
                        (
                            active_compositions[child_index],
                            child_composition[unstable_phase, child_index][None, :],
                        )
                    )
                    fallback_seed_logits[int(active_split_indices[child_index])] = (
                        _logits_from_phases(
                            seed_fractions,
                            seed_compositions,
                        )
                    )

    invariant_indices = [
        state_index
        for state_index in range(state_count)
        if _binary_invariant_split(
            binary_invariants,
            temperatures[state_index],
            pressures[state_index],
            feeds[state_index],
            resolved_options,
        )
        is not None
    ]
    difficult_indices.update(invariant_indices)
    if invariant_indices:
        converged[invariant_indices] = False

    batched_search_seconds = time.perf_counter() - started
    refinement_started = time.perf_counter()
    replacements = _gibbs_fallback_grid_states(
        model,
        temperatures,
        pressures,
        feeds,
        list(difficult_indices),
        one_phase_gibbs,
        one_phase_gibbs - gibbs_reduction,
        converged,
        resolved_options,
        binary_invariants,
        fallback_seed_logits,
    )

    def install_replacements(
        current: dict[int, tuple[Tensor, Tensor, Tensor]],
    ) -> int:
        installed = 0
        for state_index, (fractions, compositions, split_gibbs) in current.items():
            fractions, balance_residual, equilibrium_residual = _candidate_diagnostics(
                model,
                temperatures[state_index],
                pressures[state_index],
                feeds[state_index],
                fractions,
                compositions,
            )
            reduction = one_phase_gibbs[state_index] - split_gibbs
            existing_gibbs = one_phase_gibbs[state_index] - gibbs_reduction[state_index]
            lower_gibbs = bool(
                split_gibbs < existing_gibbs - resolved_options.gibbs_reduction_tolerance
            )
            resolves_failure = not bool(converged[state_index])
            if not (lower_gibbs or resolves_failure):
                continue
            count = fractions.numel()
            padded_fractions[state_index] = 0.0
            padded_compositions[state_index] = torch.nan
            padded_fractions[state_index, :count] = fractions
            padded_compositions[state_index, :count] = compositions
            phase_counts[state_index] = count
            gibbs_reduction[state_index] = torch.clamp_min(reduction, 0.0)
            fugacity_residual[state_index] = equilibrium_residual
            material_balance_residual[state_index] = balance_residual
            converged[state_index] = bool(
                (equilibrium_residual <= resolved_options.fugacity_tolerance)
                & (balance_residual <= resolved_options.material_balance_tolerance)
            )
            installed += 1
        return installed

    fallback_replacements = install_replacements(replacements)
    audited_indices: set[int] = set()
    audit_replacements = 0
    if len(grid_shape) == 2:
        n_rows, n_columns = grid_shape
        while True:
            audit_indices = [
                index
                for index in _bracketed_lower_phase_indices(
                    phase_counts,
                    n_rows,
                    n_columns,
                )
                if index not in audited_indices
            ]
            if not audit_indices:
                break
            audited_indices.update(audit_indices)
            audit_seed_logits = _continuation_seed_logits(
                phase_counts,
                padded_fractions,
                padded_compositions,
                audit_indices,
                n_rows,
                n_columns,
            )
            audit_results = _gibbs_fallback_grid_states(
                model,
                temperatures,
                pressures,
                feeds,
                audit_indices,
                one_phase_gibbs,
                one_phase_gibbs - gibbs_reduction,
                converged,
                resolved_options,
                binary_invariants,
                audit_seed_logits,
            )
            audit_replacements += install_replacements(audit_results)

    refinement_seconds = time.perf_counter() - refinement_started
    return GridEquilibrium(
        temperatures,
        pressures,
        feeds,
        grid_shape,
        padded_fractions,
        padded_compositions,
        phase_counts,
        gibbs_reduction,
        fugacity_residual,
        material_balance_residual,
        converged,
        time.perf_counter() - started,
        batched_search_seconds,
        refinement_seconds,
        len(difficult_indices),
        len(audited_indices),
        fallback_replacements,
        audit_replacements,
    )
