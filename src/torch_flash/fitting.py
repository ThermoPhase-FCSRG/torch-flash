"""Autodifferentiable thermodynamic parameter estimation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

import torch
from torch import Tensor, nn

from torch_flash.components import ComponentSet
from torch_flash.envelope import (
    PhaseBoundaryKind,
    PhaseTransitionEvaluation,
    PhaseTransitionState,
    evaluate_phase_transition_states,
    solve_batched_phase_transition_pressures,
)
from torch_flash.exceptions import InvalidStateError
from torch_flash.flash import solve_batched_binary_three_phase_invariants
from torch_flash.properties.state import StateModel
from torch_flash.types import PhaseKind, normalize_composition

OptimizerKind = Literal["adam", "adamw", "sgd", "lbfgs"]
CalibrationObjective = Literal["transition-pressure", "observed-state-fugacity"]


@dataclass(frozen=True)
class FitResult:
    """Optimization history and final convergence diagnostics.

    Attributes
    ----------
    losses
        Scalar objective at the initial point followed by each accepted
        optimizer iterate.
    converged
        Whether successive objective values met the relative stopping test.
    iterations
        Number of optimizer iterations executed.
    final_loss
        Minimum objective value retained in the supplied parameter tensors.
    stopping_reason
        ``"tolerance"``, ``"patience"``, or ``"iteration-limit"``.

    Notes
    -----
    Optimizer convergence does not establish parameter identifiability or
    model validation. Inspect sensitivities, parameter correlations, and
    independent holdout behavior separately.
    """

    losses: tuple[float, ...]
    converged: bool
    iterations: int
    final_loss: float
    stopping_reason: Literal["tolerance", "patience", "iteration-limit"]


@dataclass(frozen=True)
class CubicInteractionFitSystem:
    """One component subset and transition-state group in a cubic fit.

    Attributes
    ----------
    components
        Ordered component set used to construct this system's cubic model.
    component_indices
        Indices mapping ``components`` into the master interaction matrix.
    states
        Phase-transition states used by the joint fit and sensitivity audit.
    """

    components: ComponentSet
    component_indices: tuple[int, ...]
    states: tuple[PhaseTransitionState, ...]


@dataclass(frozen=True)
class CubicPhaseTransitionFitResult:
    """Bounded cubic-interaction fit and local identifiability diagnostics.

    Attributes
    ----------
    parameters
        Selected physical interaction vector in the declared pair order.
    parameter_history
        Scalar objective at the initial point followed by each accepted
        optimizer iterate.
    selected_iteration
        Zero-based optimizer-step index of the retained minimum-loss point.
        Zero identifies the initial point.
    selected_loss
        Loss at ``parameters``.
    optimizer_converged
        Whether the strict successive-loss stopping test was met.
    iterations
        Number of Adam iterations executed.
    optimizer_stopping_reason
        ``"tolerance"``, ``"patience"``, or ``"iteration-limit"``.
    seed_evaluations
        Source-parameter branch evaluations in system/state order.
    sensitivity_matrix
        Jacobian of fitted log pressure ratios with respect to physical
        interaction parameters. Nonfinite rows are excluded.
    sensitivity_singular_values
        Singular values of ``sensitivity_matrix``.
    sensitivity_rank
        Numerical matrix rank using dtype-aware SVD tolerance.
    sensitivity_condition_number
        Largest/smallest singular-value ratio, or infinity when rank deficient.
    calibration_objective
        Residual formulation minimized during calibration.
    optimizer
        PyTorch optimizer used for parameter updates.
    sensitivity_kind
        Scientific meaning of the rows in ``sensitivity_matrix``.
    fitted_states
        Phase-transition states in input system/state order with the selected
        incipient branch compositions installed as explicit initializers.
    """

    parameters: Tensor
    parameter_history: tuple[float, ...]
    selected_iteration: int
    selected_loss: float
    optimizer_converged: bool
    iterations: int
    optimizer_stopping_reason: Literal["tolerance", "patience", "iteration-limit"]
    seed_evaluations: tuple[tuple[PhaseTransitionEvaluation, ...], ...]
    sensitivity_matrix: Tensor
    sensitivity_singular_values: Tensor
    sensitivity_rank: int
    sensitivity_condition_number: float
    calibration_objective: CalibrationObjective
    optimizer: OptimizerKind
    sensitivity_kind: Literal["log-transition-pressure", "observed-state-fugacity"]
    fitted_states: tuple[tuple[PhaseTransitionState, ...], ...]


@dataclass(frozen=True)
class _CubicTransitionBatch:
    system_index: int
    boundary_kind: PhaseBoundaryKind
    temperature: Tensor
    reference_pressure: Tensor
    parent_composition: Tensor
    initial_phase_composition: Tensor
    minimum_pressure: Tensor
    maximum_pressure: Tensor


@dataclass(frozen=True)
class _CubicTransitionBatchEvaluation:
    pressure: Tensor
    residual_norm: Tensor
    phase_separation: Tensor
    solver_converged: Tensor
    converged: Tensor


@dataclass(frozen=True)
class _CubicObservedStateBatch:
    system_index: int
    phase_kinds: tuple[PhaseKind, PhaseKind]
    temperature: Tensor
    pressure: Tensor
    parent_composition: Tensor
    initial_incipient_composition: Tensor
    separation_component: Tensor
    separation_direction: Tensor
    other_component_indices: Tensor


def _build_cubic_transition_batches(
    systems: Sequence[CubicInteractionFitSystem],
) -> tuple[_CubicTransitionBatch, ...]:
    batches = []
    boundary_order: tuple[PhaseBoundaryKind, ...] = (
        "liquid-liquid",
        "liquid-vapor",
        "liquid-liquid-vapor",
    )
    for system_index, system in enumerate(systems):
        for boundary_kind in boundary_order:
            states = tuple(state for state in system.states if state.boundary_kind == boundary_kind)
            if not states:
                continue
            reference_pressure = torch.stack(tuple(state.reference_pressure for state in states))
            minimum_pressure = torch.stack(
                tuple(
                    (
                        reference_pressure.new_tensor(state.minimum_pressure)
                        if state.minimum_pressure is not None
                        else torch.maximum(
                            reference_pressure.new_tensor(0.2e6),
                            0.25 * state.reference_pressure,
                        )
                    )
                    for state in states
                )
            )
            maximum_pressure = torch.stack(
                tuple(
                    (
                        reference_pressure.new_tensor(state.maximum_pressure)
                        if state.maximum_pressure is not None
                        else torch.minimum(
                            reference_pressure.new_tensor(80.0e6),
                            4.0 * state.reference_pressure,
                        )
                    )
                    for state in states
                )
            )
            if boundary_kind == "liquid-liquid-vapor":
                initial_phase_composition = torch.stack(
                    tuple(state.initial_three_phase_compositions[0] for state in states)
                )
            else:
                initial_phase_composition = torch.stack(
                    tuple(state.initial_incipient_compositions[0] for state in states)
                )
            batches.append(
                _CubicTransitionBatch(
                    system_index,
                    boundary_kind,
                    torch.stack(tuple(state.temperature for state in states)),
                    reference_pressure,
                    torch.stack(tuple(state.parent_composition for state in states)),
                    initial_phase_composition,
                    minimum_pressure,
                    maximum_pressure,
                )
            )
    return tuple(batches)


def _separated_two_phase_seed(
    state: PhaseTransitionState,
    evaluation: PhaseTransitionEvaluation,
    *,
    minimum_phase_separation: float,
) -> Tensor:
    parent = normalize_composition(state.parent_composition)
    candidate = normalize_composition(evaluation.phase_compositions[0])
    separation = torch.abs(candidate - parent)
    if bool(
        torch.isfinite(candidate).all()
        & (candidate > 0.0).all()
        & (separation.max() > minimum_phase_separation)
    ):
        return candidate

    enriched_component = int(torch.argmin(parent))
    target = torch.maximum(
        parent[enriched_component] + 2.0 * minimum_phase_separation,
        parent.new_tensor(0.5),
    )
    target = torch.minimum(target, parent.new_tensor(0.95))
    remaining = torch.arange(parent.numel(), device=parent.device) != enriched_component
    seed = parent.new_zeros(parent.shape)
    seed[enriched_component] = target
    seed[remaining] = (1.0 - target) * parent[remaining] / parent[remaining].sum()
    return seed


def _build_cubic_observed_state_batches(
    systems: Sequence[CubicInteractionFitSystem],
    *,
    minimum_phase_separation: float,
) -> tuple[_CubicObservedStateBatch, ...]:
    batches = []
    phase_groups: tuple[
        tuple[PhaseBoundaryKind, tuple[PhaseKind, PhaseKind]],
        ...,
    ] = (
        ("liquid-liquid", ("liquid", "liquid")),
        ("liquid-vapor", ("liquid", "vapor")),
    )
    for system_index, system in enumerate(systems):
        for boundary_kind, phase_kinds in phase_groups:
            states = tuple(state for state in system.states if state.boundary_kind == boundary_kind)
            if not states:
                continue
            initial = torch.stack(
                tuple(state.initial_incipient_compositions[0] for state in states)
            )
            parent = torch.stack(tuple(state.parent_composition for state in states))
            differences = initial - parent
            separation_component = torch.argmax(torch.abs(differences), dim=-1)
            selected_difference = differences.gather(
                1,
                separation_component.unsqueeze(-1),
            ).squeeze(-1)
            fallback_direction = torch.where(
                parent.gather(1, separation_component.unsqueeze(-1)).squeeze(-1) < 0.5,
                torch.ones_like(selected_difference),
                -torch.ones_like(selected_difference),
            )
            separation_direction = torch.where(
                selected_difference == 0.0,
                fallback_direction,
                torch.sign(selected_difference),
            )
            component_indices = torch.arange(
                parent.shape[-1],
                dtype=torch.long,
                device=parent.device,
            ).expand(parent.shape[0], -1)
            other_component_indices = component_indices[
                component_indices != separation_component.unsqueeze(-1)
            ].reshape(parent.shape[0], parent.shape[-1] - 1)
            batches.append(
                _CubicObservedStateBatch(
                    system_index,
                    phase_kinds,
                    torch.stack(tuple(state.temperature for state in states)),
                    torch.stack(tuple(state.reference_pressure for state in states)),
                    parent,
                    initial,
                    separation_component,
                    separation_direction,
                    other_component_indices,
                )
            )
    return tuple(batches)


def _encode_separated_composition(
    batch: _CubicObservedStateBatch,
    *,
    minimum_phase_separation: float,
) -> Tensor:
    parent_selected = batch.parent_composition.gather(
        1,
        batch.separation_component.unsqueeze(-1),
    ).squeeze(-1)
    initial_selected = batch.initial_incipient_composition.gather(
        1,
        batch.separation_component.unsqueeze(-1),
    ).squeeze(-1)
    tiny = torch.finfo(parent_selected.dtype).eps
    upper = torch.clamp(
        parent_selected - minimum_phase_separation,
        min=tiny,
        max=1.0 - tiny,
    )
    lower = torch.clamp(
        parent_selected + minimum_phase_separation,
        min=tiny,
        max=1.0 - tiny,
    )
    negative_fraction = torch.clamp(initial_selected / upper, 1.0e-6, 1.0 - 1.0e-6)
    positive_fraction = torch.clamp(
        (initial_selected - lower) / (1.0 - lower),
        1.0e-6,
        1.0 - 1.0e-6,
    )
    selected_fraction = torch.where(
        batch.separation_direction < 0.0,
        negative_fraction,
        positive_fraction,
    )
    selected_raw = torch.logit(selected_fraction).unsqueeze(-1)

    initial_other = batch.initial_incipient_composition.gather(
        1,
        batch.other_component_indices,
    )
    other_fraction = normalize_composition(initial_other)
    if other_fraction.shape[-1] == 1:
        return selected_raw
    other_logits = torch.log(other_fraction[:, :-1]) - torch.log(other_fraction[:, -1:])
    return torch.cat((selected_raw, other_logits), dim=-1)


def _decode_separated_composition(
    batch: _CubicObservedStateBatch,
    raw_composition: Tensor,
    *,
    minimum_phase_separation: float,
) -> Tensor:
    parent_selected = batch.parent_composition.gather(
        1,
        batch.separation_component.unsqueeze(-1),
    ).squeeze(-1)
    tiny = torch.finfo(parent_selected.dtype).eps
    upper = torch.clamp(
        parent_selected - minimum_phase_separation,
        min=tiny,
        max=1.0 - tiny,
    )
    lower = torch.clamp(
        parent_selected + minimum_phase_separation,
        min=tiny,
        max=1.0 - tiny,
    )
    fraction = torch.sigmoid(raw_composition[:, 0])
    selected = torch.where(
        batch.separation_direction < 0.0,
        upper * fraction,
        lower + (1.0 - lower) * fraction,
    )
    other_logits = torch.cat(
        (
            raw_composition[:, 1:],
            raw_composition.new_zeros((raw_composition.shape[0], 1)),
        ),
        dim=-1,
    )
    other = (1.0 - selected).unsqueeze(-1) * torch.softmax(other_logits, dim=-1)
    composition = raw_composition.new_zeros(batch.parent_composition.shape)
    composition = composition.scatter(1, batch.other_component_indices, other)
    return composition.scatter(
        1,
        batch.separation_component.unsqueeze(-1),
        selected.unsqueeze(-1),
    )


def _cubic_observed_state_fugacity_residuals(
    constructor: Callable[..., StateModel],
    systems: Sequence[CubicInteractionFitSystem],
    batches: Sequence[_CubicObservedStateBatch],
    physical_parameters: Tensor,
    raw_compositions: Sequence[Tensor],
    *,
    kij_pairs: Sequence[tuple[int, int]],
    lij_pairs: Sequence[tuple[int, int]],
    minimum_phase_separation: float,
) -> tuple[Tensor, ...]:
    models = build_cubic_interaction_models(
        constructor,
        systems,
        physical_parameters,
        kij_pairs=kij_pairs,
        lij_pairs=lij_pairs,
    )
    residuals = []
    for batch, raw_composition in zip(batches, raw_compositions, strict=True):
        incipient = _decode_separated_composition(
            batch,
            raw_composition,
            minimum_phase_separation=minimum_phase_separation,
        )
        model = models[batch.system_index]
        parent_log_phi = model.log_fugacity_coefficients(
            batch.temperature,
            batch.pressure,
            batch.parent_composition,
            batch.phase_kinds[0],
        )
        incipient_log_phi = model.log_fugacity_coefficients(
            batch.temperature,
            batch.pressure,
            incipient,
            batch.phase_kinds[1],
        )
        residuals.append(
            torch.log(batch.parent_composition)
            + parent_log_phi
            - torch.log(incipient)
            - incipient_log_phi
        )
    return tuple(residuals)


def _cubic_observed_state_fugacity_loss(
    constructor: Callable[..., StateModel],
    systems: Sequence[CubicInteractionFitSystem],
    batches: Sequence[_CubicObservedStateBatch],
    physical_parameters: Tensor,
    raw_compositions: Sequence[Tensor],
    *,
    kij_pairs: Sequence[tuple[int, int]],
    lij_pairs: Sequence[tuple[int, int]],
    minimum_phase_separation: float,
) -> Tensor:
    residuals = _cubic_observed_state_fugacity_residuals(
        constructor,
        systems,
        batches,
        physical_parameters,
        raw_compositions,
        kij_pairs=kij_pairs,
        lij_pairs=lij_pairs,
        minimum_phase_separation=minimum_phase_separation,
    )
    if not residuals:
        raise ValueError("at least one two-phase transition fit state is required")
    return torch.cat(tuple(residual.square().mean(dim=-1) for residual in residuals)).mean()


def _evaluate_cubic_transition_batches(
    models: Sequence[StateModel],
    batches: Sequence[_CubicTransitionBatch],
    *,
    tolerance: float,
    max_iterations: int,
    minimum_phase_separation: float,
) -> tuple[_CubicTransitionBatchEvaluation, ...]:
    evaluations = []
    for batch in batches:
        model = models[batch.system_index]
        try:
            if batch.boundary_kind == "liquid-liquid-vapor":
                invariant_result = solve_batched_binary_three_phase_invariants(
                    model,
                    batch.temperature,
                    batch.reference_pressure,
                    batch.initial_phase_composition,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
                separation = (
                    torch.diff(
                        invariant_result.phase_compositions[:, :, 0],
                        dim=-1,
                    )
                    .abs()
                    .amin(dim=-1)
                )
                solver_converged = invariant_result.converged
                converged = solver_converged & (separation > minimum_phase_separation)
                evaluations.append(
                    _CubicTransitionBatchEvaluation(
                        invariant_result.pressure,
                        invariant_result.residual_norm,
                        separation,
                        solver_converged,
                        converged,
                    )
                )
            else:
                transition_result = solve_batched_phase_transition_pressures(
                    model,
                    batch.temperature,
                    batch.parent_composition,
                    phase_kinds=(
                        ("liquid", "liquid")
                        if batch.boundary_kind == "liquid-liquid"
                        else ("liquid", "vapor")
                    ),
                    initial_pressure=batch.reference_pressure,
                    initial_incipient_composition=batch.initial_phase_composition,
                    minimum_pressure=batch.minimum_pressure,
                    maximum_pressure=batch.maximum_pressure,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    minimum_phase_separation=minimum_phase_separation,
                )
                evaluations.append(
                    _CubicTransitionBatchEvaluation(
                        transition_result.pressure,
                        transition_result.residual_norm,
                        transition_result.phase_separation,
                        transition_result.solver_converged,
                        transition_result.converged,
                    )
                )
        except (InvalidStateError, RuntimeError, torch.linalg.LinAlgError):
            count = batch.temperature.shape[0]
            evaluations.append(
                _CubicTransitionBatchEvaluation(
                    batch.reference_pressure.new_full((count,), torch.nan),
                    batch.reference_pressure.new_full((count,), torch.inf),
                    batch.reference_pressure.new_zeros(count),
                    torch.zeros(
                        count,
                        dtype=torch.bool,
                        device=batch.reference_pressure.device,
                    ),
                    torch.zeros(
                        count,
                        dtype=torch.bool,
                        device=batch.reference_pressure.device,
                    ),
                )
            )
    return tuple(evaluations)


def _cubic_transition_loss(
    constructor: Callable[..., StateModel],
    systems: Sequence[CubicInteractionFitSystem],
    batches: Sequence[_CubicTransitionBatch],
    physical_parameters: Tensor,
    *,
    kij_pairs: Sequence[tuple[int, int]],
    lij_pairs: Sequence[tuple[int, int]],
    fugacity_tolerance: float,
    solver_iterations: int,
    minimum_phase_separation: float,
    fugacity_penalty: float,
    phase_merge_penalty: float,
) -> Tensor:
    models = build_cubic_interaction_models(
        constructor,
        systems,
        physical_parameters,
        kij_pairs=kij_pairs,
        lij_pairs=lij_pairs,
    )
    evaluations = _evaluate_cubic_transition_batches(
        models,
        batches,
        tolerance=fugacity_tolerance,
        max_iterations=solver_iterations,
        minimum_phase_separation=minimum_phase_separation,
    )
    if not evaluations:
        raise ValueError("at least one phase-transition fit state is required")
    pressure_terms = torch.cat(
        tuple(
            torch.nan_to_num(
                torch.log(evaluation.pressure / batch.reference_pressure),
                nan=1.0,
                posinf=1.0,
                neginf=-1.0,
            ).square()
            for batch, evaluation in zip(batches, evaluations, strict=True)
        )
    )
    closure_terms = torch.cat(
        tuple(
            torch.nan_to_num(
                evaluation.residual_norm,
                nan=10.0,
                posinf=10.0,
                neginf=-10.0,
            ).square()
            for evaluation in evaluations
        )
    )
    separation_terms = torch.cat(
        tuple(
            torch.relu(
                evaluation.pressure.new_tensor(minimum_phase_separation)
                - evaluation.phase_separation
            ).square()
            for evaluation in evaluations
        )
    )
    return (
        pressure_terms.mean()
        + fugacity_penalty * closure_terms.mean()
        + phase_merge_penalty * separation_terms.mean()
    )


def _interaction_matrix(
    values: Tensor,
    pairs: Sequence[tuple[int, int]],
    component_count: int,
) -> Tensor:
    matrix = values.new_zeros((component_count, component_count))
    for value, (first, second) in zip(values, pairs, strict=True):
        mask = values.new_zeros((component_count, component_count))
        mask[first, second] = 1.0
        mask[second, first] = 1.0
        matrix = matrix + value * mask
    return matrix


def build_cubic_interaction_models(
    constructor: Callable[..., StateModel],
    systems: Sequence[CubicInteractionFitSystem],
    parameters: Tensor,
    *,
    kij_pairs: Sequence[tuple[int, int]],
    lij_pairs: Sequence[tuple[int, int]] = (),
) -> tuple[StateModel, ...]:
    """Construct matched cubic systems from one shared interaction vector.

    Parameters
    ----------
    constructor
        Cubic model constructor accepting ``components``, ``kij``, and ``lij``.
    systems
        Component subsets and their master-component index mappings.
    parameters
        One-dimensional tensor containing all ``kij_pairs`` values followed
        by all ``lij_pairs`` values.
    kij_pairs, lij_pairs
        Unique off-diagonal master-component index pairs. Returned models use
        symmetric zero-diagonal matrices.

    Returns
    -------
    tuple
        One constructed model per system in input order. Parameter gradients,
        dtype, and device are preserved.

    Raises
    ------
    ValueError
        If parameters, system mappings, or interaction pairs are invalid.
    """
    if parameters.ndim != 1 or parameters.numel() != len(kij_pairs) + len(lij_pairs):
        raise ValueError("interaction parameter vector does not match declared pairs")
    if not systems:
        raise ValueError("at least one cubic interaction fit system is required")
    component_count = max(max(system.component_indices) for system in systems) + 1
    all_pairs = (*kij_pairs, *lij_pairs)
    if any(
        first < 0
        or second < 0
        or first >= component_count
        or second >= component_count
        or first == second
        for first, second in all_pairs
    ):
        raise ValueError("interaction pairs must be distinct master-component indices")
    if len(set(kij_pairs)) != len(kij_pairs) or len(set(lij_pairs)) != len(lij_pairs):
        raise ValueError("interaction pairs must be unique within each interaction matrix")
    for system in systems:
        if len(system.component_indices) != len(system.components.names) or len(
            set(system.component_indices)
        ) != len(system.component_indices):
            raise ValueError("system component indices must uniquely match its component set")

    kij = _interaction_matrix(parameters[: len(kij_pairs)], kij_pairs, component_count)
    lij = _interaction_matrix(parameters[len(kij_pairs) :], lij_pairs, component_count)
    models = []
    for system in systems:
        indices = torch.tensor(
            system.component_indices,
            dtype=torch.long,
            device=parameters.device,
        )
        models.append(
            constructor(
                system.components,
                kij=kij[indices][:, indices],
                lij=lij[indices][:, indices],
            )
        )
    return tuple(models)


def fit_cubic_phase_transition_interactions(
    constructor: Callable[..., StateModel],
    systems: Sequence[CubicInteractionFitSystem],
    initial_parameters: Tensor,
    lower_bounds: Tensor,
    upper_bounds: Tensor,
    *,
    kij_pairs: Sequence[tuple[int, int]],
    lij_pairs: Sequence[tuple[int, int]] = (),
    objective: CalibrationObjective = "transition-pressure",
    optimizer: OptimizerKind = "adam",
    learning_rate: float = 5.0e-3,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    max_iterations: int = 30,
    stopping_tolerance: float = 1.0e-9,
    no_improvement_patience: int | None = 6,
    fugacity_tolerance: float = 1.0e-7,
    solver_iterations: int = 24,
    minimum_phase_separation: float = 2.0e-3,
    fugacity_penalty: float = 0.05,
    phase_merge_penalty: float = 5.0,
    parameter_prior_weight: float = 0.0,
) -> CubicPhaseTransitionFitResult:
    """Fit shared conventional-cubic interactions to phase-transition states.

    Parameters
    ----------
    constructor
        Cubic model constructor accepting component, ``kij``, and ``lij``
        arguments.
    systems
        Component subsets, master-index mappings, and two-phase transition
        states. Two-phase boundaries and binary three-phase invariants are
        included in one joint objective.
    initial_parameters
        Interior physical interaction vector in ``kij_pairs`` then
        ``lij_pairs`` order.
    lower_bounds, upper_bounds
        Finite physical bounds with the same shape as ``initial_parameters``.
    kij_pairs, lij_pairs
        Shared master-component pair identities.
    objective
        ``"transition-pressure"`` solves a local transition pressure at every
        trial parameter vector and minimizes squared log-pressure ratios.
        ``"observed-state-fugacity"`` simultaneously optimizes the shared
        interactions and one physically separated incipient composition per
        two-phase observation at its measured temperature and pressure.
    optimizer
        PyTorch optimizer: ``"adam"``, ``"adamw"``, ``"sgd"``, or
        strong-Wolfe ``"lbfgs"``.
    learning_rate
        Optimizer learning rate in unconstrained coordinates.
    weight_decay
        Nonnegative optimizer weight decay. Keep zero unless a regularization
        prior on the transformed variables is scientifically justified.
    momentum
        SGD momentum. Ignored by Adam and AdamW.
    max_iterations
        Maximum optimizer iterations.
    stopping_tolerance
        Relative successive-loss stopping threshold.
    no_improvement_patience
        Stop after this many evaluated losses without a new minimum. ``None``
        disables this criterion. The best evaluated iterate is retained in
        either case.
    fugacity_tolerance, solver_iterations
        Local transition-pressure solver controls. They apply to the
        ``"transition-pressure"`` objective and the post-fit pressure
        sensitivity audit.
    minimum_phase_separation
        Minimum selected-component mole-fraction difference. The simultaneous
        observed-state objective enforces it by construction, rather than
        allowing the algebraic homogeneous solution.
    fugacity_penalty
        Weight on mean squared dimensionless fugacity residual.
    phase_merge_penalty
        Weight on squared phase-separation shortfall.
    parameter_prior_weight
        Nonnegative quadratic prior weight on the bounded physical interaction
        displacement from ``initial_parameters``, scaled by each half-range.
        Zero disables the prior.

    Returns
    -------
    CubicPhaseTransitionFitResult
        Minimum-loss bounded parameters, complete optimization history,
        source-seed branch results, and a local sensitivity SVD.

    Raises
    ------
    ValueError
        If bounds, parameters, pair identities, states, objective, optimizer,
        or numerical controls are invalid, or the selected objective cannot
        represent a requested state.

    Notes
    -----
    Branches are identified once at ``initial_parameters`` using
    :func:`torch_flash.envelope.evaluate_phase_transition_state`; the selected
    or fallback physically separated incipient composition initializes the
    same branch during autodiff optimization. The simultaneous observed-state
    formulation evaluates every experimental state in every optimizer step,
    avoids differentiating through a failed nested pressure solve, and keeps
    the latent incipient compositions strictly separated from the parent.

    The minimum-loss evaluated iterate is retained even when a later
    first-order step increases the objective. SVD rank and condition number
    diagnose only the reported local residual Jacobian; they do not establish
    a unique global fit.
    """
    if (
        initial_parameters.ndim != 1
        or lower_bounds.shape != initial_parameters.shape
        or upper_bounds.shape != initial_parameters.shape
    ):
        raise ValueError("initial cubic interactions and bounds must be equally shaped vectors")
    if not bool(
        torch.isfinite(initial_parameters).all()
        & torch.isfinite(lower_bounds).all()
        & torch.isfinite(upper_bounds).all()
        & (lower_bounds < initial_parameters).all()
        & (initial_parameters < upper_bounds).all()
    ):
        raise ValueError("initial cubic interactions must lie strictly inside finite bounds")
    if (
        learning_rate <= 0.0
        or max_iterations <= 0
        or stopping_tolerance <= 0.0
        or (no_improvement_patience is not None and no_improvement_patience <= 0)
        or fugacity_tolerance <= 0.0
        or solver_iterations <= 0
        or minimum_phase_separation < 0.0
        or fugacity_penalty < 0.0
        or phase_merge_penalty < 0.0
        or parameter_prior_weight < 0.0
        or weight_decay < 0.0
        or momentum < 0.0
    ):
        raise ValueError("cubic phase-transition fitting controls are invalid")
    if objective not in ("transition-pressure", "observed-state-fugacity"):
        raise ValueError("unknown cubic phase-transition calibration objective")
    if optimizer not in ("adam", "adamw", "sgd", "lbfgs"):
        raise ValueError("unknown PyTorch fitting optimizer")
    if objective != "transition-pressure" and any(
        state.boundary_kind == "liquid-liquid-vapor"
        for system in systems
        for state in system.states
    ):
        raise ValueError(
            "simultaneous observed-state calibration currently accepts only two-phase states"
        )

    midpoint = 0.5 * (lower_bounds + upper_bounds)
    half_width = 0.5 * (upper_bounds - lower_bounds)

    def to_physical(raw: Tensor) -> Tensor:
        return midpoint + half_width * torch.tanh(raw)

    normalized = torch.clamp(
        (initial_parameters - midpoint) / half_width,
        -1.0 + 1.0e-10,
        1.0 - 1.0e-10,
    )
    raw = nn.Parameter(torch.atanh(normalized).detach().clone())
    initial_models = build_cubic_interaction_models(
        constructor,
        systems,
        initial_parameters,
        kij_pairs=kij_pairs,
        lij_pairs=lij_pairs,
    )
    seed_evaluations = tuple(
        evaluate_phase_transition_states(
            model,
            system.states,
            exhaustive_two_phase_starts=True,
        )
        for model, system in zip(initial_models, systems, strict=True)
    )
    if any(
        state.boundary_kind == "liquid-liquid-vapor" and not evaluation.converged
        for system, system_evaluations in zip(
            systems,
            seed_evaluations,
            strict=True,
        )
        for state, evaluation in zip(
            system.states,
            system_evaluations,
            strict=True,
        )
    ):
        raise ValueError(
            "three-phase fit states require residual-converged, non-degenerate seed branches"
        )
    fixed_systems = tuple(
        CubicInteractionFitSystem(
            system.components,
            system.component_indices,
            tuple(
                replace(
                    state,
                    initial_incipient_compositions=(
                        ()
                        if state.boundary_kind == "liquid-liquid-vapor"
                        else (
                            _separated_two_phase_seed(
                                state,
                                evaluation,
                                minimum_phase_separation=minimum_phase_separation,
                            ),
                        )
                    ),
                    initial_three_phase_compositions=(
                        (evaluation.phase_compositions,)
                        if state.boundary_kind == "liquid-liquid-vapor"
                        else ()
                    ),
                )
                for state, evaluation in zip(
                    system.states,
                    system_evaluations,
                    strict=True,
                )
            ),
        )
        for system, system_evaluations in zip(systems, seed_evaluations, strict=True)
    )
    fit_batches = _build_cubic_transition_batches(fixed_systems)
    observed_batches = (
        _build_cubic_observed_state_batches(
            fixed_systems,
            minimum_phase_separation=minimum_phase_separation,
        )
        if objective != "transition-pressure"
        else ()
    )
    raw_compositions = tuple(
        nn.Parameter(
            _encode_separated_composition(
                batch,
                minimum_phase_separation=minimum_phase_separation,
            )
        )
        for batch in observed_batches
    )

    def loss_at(raw_parameters: Tensor, compositions: Sequence[Tensor]) -> Tensor:
        physical_parameters = to_physical(raw_parameters)
        if objective == "observed-state-fugacity":
            loss = _cubic_observed_state_fugacity_loss(
                constructor,
                fixed_systems,
                observed_batches,
                physical_parameters,
                compositions,
                kij_pairs=kij_pairs,
                lij_pairs=lij_pairs,
                minimum_phase_separation=minimum_phase_separation,
            )
        else:
            loss = _cubic_transition_loss(
                constructor,
                fixed_systems,
                fit_batches,
                physical_parameters,
                kij_pairs=kij_pairs,
                lij_pairs=lij_pairs,
                fugacity_tolerance=fugacity_tolerance,
                solver_iterations=solver_iterations,
                minimum_phase_separation=minimum_phase_separation,
                fugacity_penalty=fugacity_penalty,
                phase_merge_penalty=phase_merge_penalty,
            )
        parameter_prior = ((physical_parameters - initial_parameters) / half_width).square().mean()
        return loss + parameter_prior_weight * parameter_prior

    optimization = fit_parameters(
        (raw, *raw_compositions),
        lambda: loss_at(raw, raw_compositions),
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        max_iterations=max_iterations,
        tolerance=stopping_tolerance,
        no_improvement_patience=no_improvement_patience,
    )
    selected_index = min(
        range(len(optimization.losses)),
        key=optimization.losses.__getitem__,
    )
    selected_parameters = to_physical(raw).detach()
    selected_raw_compositions = tuple(
        raw_composition.detach().clone() for raw_composition in raw_compositions
    )
    if objective != "transition-pressure":
        decoded_compositions = tuple(
            _decode_separated_composition(
                batch,
                raw_composition,
                minimum_phase_separation=minimum_phase_separation,
            ).detach()
            for batch, raw_composition in zip(
                observed_batches,
                selected_raw_compositions,
                strict=True,
            )
        )
        decoded_by_group = {
            (batch.system_index, batch.phase_kinds): compositions
            for batch, compositions in zip(
                observed_batches,
                decoded_compositions,
                strict=True,
            )
        }
        group_positions: dict[tuple[int, tuple[PhaseKind, PhaseKind]], int] = {}
        fitted_states_rows = []
        for system_index, system in enumerate(fixed_systems):
            system_states = []
            for state in system.states:
                phase_kinds: tuple[PhaseKind, PhaseKind] = (
                    ("liquid", "liquid")
                    if state.boundary_kind == "liquid-liquid"
                    else ("liquid", "vapor")
                )
                group_key = (system_index, phase_kinds)
                group_position = group_positions.get(group_key, 0)
                composition = decoded_by_group[group_key][group_position]
                group_positions[group_key] = group_position + 1
                system_states.append(
                    replace(
                        state,
                        initial_incipient_compositions=(composition,),
                    )
                )
            fitted_states = tuple(system_states)
            fitted_states_rows.append(fitted_states)
        result_fitted_states = tuple(fitted_states_rows)
    else:
        result_fitted_states = tuple(system.states for system in fixed_systems)

    sensitivity_parameters = selected_parameters.clone().requires_grad_(True)
    if objective != "transition-pressure":
        residual_rows = torch.cat(
            tuple(
                (residual / residual.shape[-1] ** 0.5).reshape(-1)
                for residual in _cubic_observed_state_fugacity_residuals(
                    constructor,
                    fixed_systems,
                    observed_batches,
                    sensitivity_parameters,
                    selected_raw_compositions,
                    kij_pairs=kij_pairs,
                    lij_pairs=lij_pairs,
                    minimum_phase_separation=minimum_phase_separation,
                )
            )
        )
        sensitivity_outputs = residual_rows
        sensitivity_kind: Literal[
            "log-transition-pressure",
            "observed-state-fugacity",
        ] = "observed-state-fugacity"
    else:
        sensitivity_models = build_cubic_interaction_models(
            constructor,
            fixed_systems,
            sensitivity_parameters,
            kij_pairs=kij_pairs,
            lij_pairs=lij_pairs,
        )
        sensitivity_evaluations = _evaluate_cubic_transition_batches(
            sensitivity_models,
            fit_batches,
            tolerance=fugacity_tolerance,
            max_iterations=solver_iterations,
            minimum_phase_separation=minimum_phase_separation,
        )
        sensitivity_outputs = torch.cat(
            tuple(
                torch.log(evaluation.pressure / batch.reference_pressure)
                for batch, evaluation in zip(
                    fit_batches,
                    sensitivity_evaluations,
                    strict=True,
                )
            )
        )
        sensitivity_kind = "log-transition-pressure"
    finite_sensitivity_outputs = torch.isfinite(sensitivity_outputs.detach())
    if bool(finite_sensitivity_outputs.any()):
        finite_outputs = sensitivity_outputs[finite_sensitivity_outputs]
        sensitivity = torch.autograd.grad(
            finite_outputs,
            sensitivity_parameters,
            grad_outputs=torch.eye(
                finite_outputs.numel(),
                dtype=finite_outputs.dtype,
                device=finite_outputs.device,
            ),
            is_grads_batched=True,
        )[0].detach()
        sensitivity = sensitivity[torch.isfinite(sensitivity).all(dim=-1)]
    else:
        sensitivity = selected_parameters.new_empty((0, selected_parameters.numel()))
    singular_values = (
        torch.linalg.svdvals(sensitivity)
        if sensitivity.numel()
        else selected_parameters.new_empty((0,))
    )
    if singular_values.numel():
        threshold = singular_values[0] * max(sensitivity.shape) * torch.finfo(sensitivity.dtype).eps
        rank = int((singular_values > threshold).sum())
        condition = (
            float(singular_values[0] / singular_values[-1])
            if rank == selected_parameters.numel()
            else float("inf")
        )
    else:
        rank = 0
        condition = float("inf")
    return CubicPhaseTransitionFitResult(
        selected_parameters,
        optimization.losses,
        selected_index,
        optimization.losses[selected_index],
        optimization.converged,
        optimization.iterations,
        optimization.stopping_reason,
        seed_evaluations,
        sensitivity,
        singular_values,
        rank,
        condition,
        objective,
        optimizer,
        sensitivity_kind,
        result_fitted_states,
    )


def least_squares_loss(
    prediction: Tensor,
    observation: Tensor,
    *,
    scale: Tensor | float = 1.0,
    weights: Tensor | None = None,
) -> Tensor:
    """Return a dimensionless weighted mean-square residual.

    Parameters
    ----------
    prediction, observation
        Broadcast-compatible model predictions and observations in matching
        physical units.
    scale
        Positive residual scale in the same units. Scalar or broadcastable
        tensor.
    weights
        Optional nonnegative statistical weights, broadcastable to the
        residual shape.

    Returns
    -------
    Tensor
        Scalar mean of the squared, scaled residuals; weights enter through
        their square roots.
    """
    residual = (prediction - observation) / scale
    if weights is not None:
        residual = residual * torch.sqrt(weights)
    return torch.mean(residual.square())


def phase_equilibrium_residual(
    model: StateModel,
    temperature: Tensor,
    pressure: Tensor,
    phase1_composition: Tensor,
    phase2_composition: Tensor,
    *,
    phase_kinds: tuple[PhaseKind, PhaseKind] = ("liquid", "vapor"),
) -> Tensor:
    """Return component log-fugacity equalities for measured phase pairs.

    Parameters
    ----------
    model
        Homogeneous-state fugacity model.
    temperature, pressure
        Scalar or batched temperatures in K and pressures in Pa.
    phase1_composition, phase2_composition
        Equally shaped, strictly positive phase mole fractions.
    phase_kinds
        Root request corresponding to each measured phase.

    Returns
    -------
    Tensor
        Dimensionless component residual
        ``ln(x1_i phi1_i) - ln(x2_i phi2_i)``.

    Raises
    ------
    ValueError
        If phase compositions differ in shape or are nonpositive/nonfinite.

    Notes
    -----
    Inputs may contain independent batched states.  The residual is
    dimensionless and is zero when every component has equal fugacity in the
    two requested phases.  Strictly positive compositions are required
    because the thermodynamic equality is evaluated in logarithmic form.
    """
    phase1 = normalize_composition(phase1_composition)
    phase2 = normalize_composition(phase2_composition)
    if phase1.shape != phase2.shape:
        raise ValueError("phase-equilibrium compositions must have equal shapes")
    if not bool(
        torch.isfinite(phase1).all()
        & torch.isfinite(phase2).all()
        & (phase1 > 0.0).all()
        & (phase2 > 0.0).all()
    ):
        raise ValueError("phase-equilibrium compositions must be finite and positive")
    return (
        torch.log(phase1)
        + model.log_fugacity_coefficients(
            temperature,
            pressure,
            phase1,
            phase_kinds[0],
        )
        - torch.log(phase2)
        - model.log_fugacity_coefficients(
            temperature,
            pressure,
            phase2,
            phase_kinds[1],
        )
    )


def fit_parameters(
    parameters: Iterable[nn.Parameter],
    closure: Callable[[], Tensor],
    *,
    optimizer: OptimizerKind = "adam",
    learning_rate: float = 0.05,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    max_iterations: int = 500,
    tolerance: float = 1.0e-10,
    no_improvement_patience: int | None = None,
) -> FitResult:
    """Fit arbitrary PyTorch thermodynamic parameters with a PyTorch optimizer.

    Parameters
    ----------
    parameters
        Trainable tensors passed to the selected optimizer.
    closure
        Zero-argument function that recomputes one finite differentiable
        scalar loss.
    optimizer
        ``"adam"``, ``"adamw"``, full-batch ``"sgd"``, or limited-memory
        ``"lbfgs"`` with a strong-Wolfe line search.
    learning_rate
        Optimizer learning rate.
    weight_decay
        Nonnegative optimizer weight decay. A nonzero value is a regularizer,
        not a numerical acceleration, and requires a defensible prior.
    momentum
        SGD momentum in ``[0, 1)``. Ignored by Adam and AdamW.
    max_iterations
        Maximum optimizer steps.
    tolerance
        Relative successive-loss stopping threshold.
    no_improvement_patience
        Optional number of consecutive evaluated losses without a new
        minimum before stopping. The returned ``converged`` flag remains
        ``False`` for this performance termination.

    Returns
    -------
    FitResult
        Initial and accepted-iterate loss history plus explicit stopping
        diagnostics. The supplied tensors are restored to the minimum-loss
        accepted point.

    Raises
    ------
    ValueError
        If no parameters are supplied or the closure returns a nonfinite or
        nonscalar loss.

    Notes
    -----
    The closure must recompute and return a scalar differentiable loss. Bounds
    can be imposed by parameterizing physical values through sigmoid/softplus
    transforms in the model.
    """
    trainable = tuple(parameters)
    if not trainable:
        raise ValueError("at least one trainable parameter is required")
    if (
        optimizer not in ("adam", "adamw", "sgd", "lbfgs")
        or learning_rate <= 0.0
        or weight_decay < 0.0
        or not 0.0 <= momentum < 1.0
        or max_iterations <= 0
        or tolerance < 0.0
    ):
        raise ValueError("fitting controls must be nonnegative with positive rate and iterations")
    if no_improvement_patience is not None and no_improvement_patience <= 0:
        raise ValueError("no-improvement patience must be positive when provided")
    if optimizer == "lbfgs" and (weight_decay != 0.0 or momentum != 0.0):
        raise ValueError("LBFGS does not accept weight decay or momentum")
    torch_optimizer: torch.optim.Optimizer
    if optimizer == "adam":
        torch_optimizer = torch.optim.Adam(
            trainable,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer == "adamw":
        torch_optimizer = torch.optim.AdamW(
            trainable,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer == "sgd":
        torch_optimizer = torch.optim.SGD(
            trainable,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        torch_optimizer = torch.optim.LBFGS(
            trainable,
            lr=learning_rate,
            max_iter=1,
            max_eval=10,
            history_size=10,
            line_search_fn="strong_wolfe",
        )
    initial_loss = closure()
    if initial_loss.ndim != 0 or not bool(torch.isfinite(initial_loss)):
        raise ValueError("fitting closure must return one finite scalar loss")
    initial = float(initial_loss.detach())
    history = [initial]
    converged = False
    previous = initial
    best = initial
    best_parameters = tuple(parameter.detach().clone() for parameter in trainable)
    evaluations_without_improvement = 0
    stopping_reason: Literal["tolerance", "patience", "iteration-limit"] = "iteration-limit"
    for _iteration in range(1, max_iterations + 1):
        if optimizer == "lbfgs":

            def line_search_closure() -> Tensor:
                torch_optimizer.zero_grad()
                trial_loss = closure()
                if trial_loss.ndim != 0 or not bool(torch.isfinite(trial_loss)):
                    raise ValueError("fitting closure must return one finite scalar loss")
                trial_loss.backward()
                return trial_loss

            cast(torch.optim.LBFGS, torch_optimizer).step(line_search_closure)
            loss = closure()
        else:
            torch_optimizer.zero_grad()
            loss = closure()
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError("fitting closure must return one finite scalar loss")
            loss.backward()
            cast(torch.optim.Optimizer, torch_optimizer).step()
            loss = closure()
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise ValueError("fitting closure must return one finite scalar loss")
        current = float(loss.detach())
        history.append(current)
        if current < best:
            best = current
            best_parameters = tuple(parameter.detach().clone() for parameter in trainable)
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1
        if abs(float(previous) - current) <= tolerance * max(1.0, abs(current)):
            converged = True
            stopping_reason = "tolerance"
            break
        if (
            no_improvement_patience is not None
            and evaluations_without_improvement >= no_improvement_patience
        ):
            stopping_reason = "patience"
            break
        previous = current
    with torch.no_grad():
        for parameter, selected in zip(trainable, best_parameters, strict=True):
            parameter.copy_(selected)
    return FitResult(
        tuple(history),
        converged,
        _iteration,
        best,
        stopping_reason,
    )


__all__ = [
    "CubicInteractionFitSystem",
    "CubicPhaseTransitionFitResult",
    "FitResult",
    "build_cubic_interaction_models",
    "fit_cubic_phase_transition_interactions",
    "fit_parameters",
    "least_squares_loss",
    "phase_equilibrium_residual",
]
