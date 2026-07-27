"""Benchmark conventional and trust-region flash paths on matched states."""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
import warnings
from collections.abc import Callable
from typing import Any

import torch

from torch_flash import (
    ChemicalState,
    ComponentSet,
    component_set,
    configure,
    multiphase_flash,
    multiphase_trust_region_flash,
    peng_robinson_1978,
    soave_redlich_kwong,
    solve_binary_three_phase_invariant,
    two_phase_flash,
    two_phase_trust_region_flash,
)
from torch_flash.exceptions import ConvergenceWarning, ExperimentalModelWarning
from torch_flash.types import FlashResult

REPEATS = 3
GROUP_SIZE = 12


def _timed(
    operation: Callable[[], FlashResult],
    *,
    repeats: int = REPEATS,
) -> tuple[FlashResult, float]:
    elapsed = []
    result: FlashResult | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        elapsed.append(time.perf_counter() - started)
    if result is None:
        raise RuntimeError("benchmark operation did not execute")
    return result, statistics.median(elapsed)


def _two_phase_benchmark() -> dict[str, Any]:
    model = peng_robinson_1978(component_set(("methane", "n_butane")))
    composition = torch.tensor([0.5, 0.5], dtype=torch.float64)
    states = tuple(
        ChemicalState(temperature, pressure, composition)
        for temperature in torch.linspace(230.0, 320.0, 13, dtype=torch.float64)
        for pressure in torch.linspace(1.0e6, 7.0e6, 13, dtype=torch.float64)
    )
    converged_states = []
    for state in states:
        try:
            result = two_phase_flash(
                model,
                state,
                check_stability=False,
                max_iterations=100,
            )
        except ValueError:
            continue
        if result.converged and result.nphases == 2:
            converged_states.append((state, result.iterations))
    converged_states.sort(key=lambda item: item[1])
    groups = {
        "ordinary": converged_states[:GROUP_SIZE],
        "higher_iteration": converged_states[-GROUP_SIZE:],
    }

    records: dict[str, Any] = {}
    for group_name, rows in groups.items():
        standard_results = []
        trust_results = []
        for state, _ in rows:
            standard, standard_seconds = _timed(
                lambda state=state: two_phase_flash(
                    model,
                    state,
                    check_stability=False,
                    max_iterations=100,
                )
            )
            trust, trust_seconds = _timed(
                lambda state=state: two_phase_trust_region_flash(
                    model,
                    state,
                    check_stability=False,
                    max_iterations=100,
                )
            )
            standard_results.append((standard, standard_seconds))
            trust_results.append((trust, trust_seconds))
        records[group_name] = {
            "states": len(rows),
            "standard_converged": sum(result.converged for result, _ in standard_results),
            "trust_region_converged": sum(result.converged for result, _ in trust_results),
            "median_standard_iterations": statistics.median(
                result.iterations for result, _ in standard_results
            ),
            "median_trust_region_iterations": statistics.median(
                result.iterations for result, _ in trust_results
            ),
            "median_standard_ms": 1.0e3
            * statistics.median(seconds for _, seconds in standard_results),
            "median_trust_region_ms": 1.0e3
            * statistics.median(seconds for _, seconds in trust_results),
            "maximum_trust_region_residual": max(
                float(result.residual_norm) for result, _ in trust_results
            ),
            "maximum_phase_fraction_difference": max(
                float(torch.max(torch.abs(trust.phase_fractions - standard.phase_fractions)))
                for (standard, _), (trust, _) in zip(
                    standard_results,
                    trust_results,
                    strict=True,
                )
            ),
        }
    return records


def _three_phase_benchmark() -> dict[str, Any]:
    components = ComponentSet(
        ("methane", "carbon_dioxide"),
        torch.tensor([190.6, 304.2], dtype=torch.float64),
        101325.0 * torch.tensor([45.4, 72.9], dtype=torch.float64),
        torch.tensor([0.008, 0.228], dtype=torch.float64),
        torch.tensor([0.01604, 0.04401], dtype=torch.float64),
        torch.tensor([9.93e-5, 9.40e-5], dtype=torch.float64),
    )
    model = soave_redlich_kwong(
        components,
        kij=torch.tensor([[0.0, 0.12], [0.12, 0.0]], dtype=torch.float64),
    )
    invariant = solve_binary_three_phase_invariant(
        model,
        torch.tensor(180.0, dtype=torch.float64),
        torch.tensor(2.73e6, dtype=torch.float64),
        torch.tensor(
            [[0.199, 0.801], [0.781, 0.219], [0.958, 0.042]],
            dtype=torch.float64,
        ),
    )
    if not invariant.converged:
        raise RuntimeError("three-phase invariant benchmark seed did not converge")
    phase_order = torch.tensor([0, 2, 1])
    phase_compositions = invariant.phase_compositions[phase_order]
    fractions = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    composition = torch.einsum("p,pi->i", fractions, phase_compositions)
    state = ChemicalState(invariant.temperature, invariant.pressure, composition)
    initial_compositions = torch.tensor(
        [[0.20, 0.80], [0.95, 0.05], [0.78, 0.22]],
        dtype=torch.float64,
    )
    initial_k_values = initial_compositions[1:] / initial_compositions[0]
    standard, standard_seconds = _timed(
        lambda: multiphase_flash(
            model,
            state,
            initial_k_values=initial_k_values,
            max_iterations=100,
        )
    )
    trust, trust_seconds = _timed(
        lambda: multiphase_trust_region_flash(
            model,
            state,
            initial_k_values=initial_k_values,
            max_iterations=100,
        )
    )
    maximum_composition_difference = max(
        float(torch.max(torch.abs(actual.composition - expected)))
        for actual, expected in zip(
            trust.phases,
            phase_compositions,
            strict=True,
        )
    )
    return {
        "temperature_K": float(invariant.temperature),
        "pressure_MPa": float(invariant.pressure / 1.0e6),
        "standard_converged": standard.converged,
        "standard_iterations": standard.iterations,
        "standard_residual": float(standard.residual_norm),
        "median_standard_seconds": standard_seconds,
        "trust_region_converged": trust.converged,
        "trust_region_iterations": trust.iterations,
        "trust_region_residual": float(trust.residual_norm),
        "median_trust_region_seconds": trust_seconds,
        "material_balance_residual": trust.diagnostics["material_balance_residual"],
        "maximum_phase_composition_difference": maximum_composition_difference,
    }


def main() -> int:
    """Run matched float64 CPU benchmarks and print one JSON document."""
    threads = int(os.environ.get("TORCH_FLASH_BENCHMARK_THREADS", "4"))
    if threads < 1:
        raise ValueError("TORCH_FLASH_BENCHMARK_THREADS must be positive")
    configure(dtype=torch.float64, device="cpu", num_threads=threads)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", ExperimentalModelWarning)
        results = {
            "runtime": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "pytorch": torch.__version__,
                "dtype": "float64",
                "device": "cpu",
                "threads": threads,
                "repeats": REPEATS,
            },
            "two_phase": _two_phase_benchmark(),
            "three_phase": _three_phase_benchmark(),
        }
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
