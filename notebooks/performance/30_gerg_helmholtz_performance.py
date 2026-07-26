# %% [markdown]
# # GERG Helmholtz performance: profiling, AD modes, compilation, and Numba
#
# This experiment asks how to reduce the latency of the H2-tailored GERG
# phase-equilibrium calculations without changing their thermodynamic root.
# The correctness oracle is equality with the pressure-based bubble solver,
# an equilibrium residual no larger than \(10^{-8}\), and automatic-derivative
# parity with centered finite differences.
#
# The workloads are kept distinct:
#
# 1. one homogeneous pressure call;
# 2. one binary bubble point at specified \(T,x\);
# 3. a 25-point binary \(p-x-y\) isotherm.
#
# Cold construction/JIT time and warmed latency are reported separately.
# Timings are hardware observations, not package-wide guarantees.

# %%
from __future__ import annotations

import cProfile
import math
import os
import platform
import pstats
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from statistics import median

import numpy as np
import pandas as pd
import torch
from IPython.display import display
from numba import njit
from torch import Tensor
from torch.profiler import ProfilerActivity, profile

from torch_flash import (
    binary_bubble_point,
    binary_helmholtz_bubble_point,
    eoscg2021,
    gerg2008_hydrogen_2021,
)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
DTYPE = torch.float64
TOLERANCE = 1.0e-8
REPEATS = 9

environment = pd.Series(
    {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical CPUs": os.cpu_count(),
        "PyTorch intra-op threads": torch.get_num_threads(),
        "PyTorch inter-op threads": torch.get_num_interop_threads(),
        "dtype": str(DTYPE),
        "torch": torch.__version__,
        "torch-flash": version("torch-flash"),
        "numba": version("numba"),
        "numpy": np.__version__,
    },
    name="value",
)
display(environment.to_frame())


def timed(
    callable_, *, warmup: int = 2, repeats: int = REPEATS
) -> tuple[float, list[float]]:
    """Return median warmed wall time and all samples."""
    for _ in range(warmup):
        callable_()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_()
        samples.append(time.perf_counter() - started)
    return median(samples), samples


# %% [markdown]
# ## Baseline and optimized continuation
#
# The original pressure formulation uses unknowns \((\ln P,\operatorname{logit}
# y_1)\). Every outer residual therefore performs liquid and vapor density
# inversions. The optimized GERG-2004 volume formulation retains
# \((\operatorname{logit}y_1,\ln\rho_L,\ln\rho_V)\), enforces two fugacity
# equalities plus \(P_L=P_V\), and reuses the converged volumes at the next
# composition. The first point still uses the conservative pressure solver.
# Between exact autodiff Jacobian refreshes, safeguarded Broyden updates avoid
# repeating nearly identical second-derivative work.

# %%
model = gerg2008_hydrogen_2021(("nitrogen", "hydrogen"))
temperature = torch.tensor(90.8)
liquid_start = torch.tensor([0.79, 0.21])
liquid_next = torch.tensor([0.75, 0.25])

initial_volume_point = binary_helmholtz_bubble_point(
    model,
    temperature,
    liquid_start,
    minimum_pressure=1.0e3,
    maximum_pressure=3.0e8,
    tolerance=TOLERANCE,
)


def pressure_point():
    return binary_bubble_point(
        model,
        temperature,
        liquid_next,
        initial_pressure=initial_volume_point.pressure,
        initial_vapor_composition=initial_volume_point.vapor_composition,
        minimum_pressure=1.0e3,
        maximum_pressure=3.0e8,
        tolerance=TOLERANCE,
        max_iterations=25,
    )


def volume_point():
    return binary_helmholtz_bubble_point(
        model,
        temperature,
        liquid_next,
        initial_point=initial_volume_point,
        minimum_pressure=1.0e3,
        maximum_pressure=3.0e8,
        tolerance=TOLERANCE,
        max_iterations=25,
    )


pressure_seconds, _ = timed(pressure_point)
volume_seconds, _ = timed(volume_point)
pressure_result = pressure_point()
volume_result = volume_point()
point_metrics = pd.Series(
    {
        "pressure formulation / s": pressure_seconds,
        "volume formulation / s": volume_seconds,
        "speedup": pressure_seconds / volume_seconds,
        "relative pressure difference": float(
            (volume_result.pressure - pressure_result.pressure).abs()
            / pressure_result.pressure
        ),
        "maximum vapor-composition difference": float(
            (volume_result.vapor_composition - pressure_result.vapor_composition)
            .abs()
            .max()
        ),
        "volume residual": float(volume_result.residual_norm),
        "volume iterations": volume_result.iterations,
    },
    name="value",
)
display(point_metrics.to_frame())
assert pressure_result.converged and volume_result.converged
assert point_metrics["relative pressure difference"] < 3.0e-10
assert float(volume_result.residual_norm) <= TOLERANCE


def trace_pressure_isotherm(
    points: int = 25,
    maximum_liquid_hydrogen: float = 0.49,
):
    previous_pressure = None
    previous_vapor = None
    results = []
    for hydrogen in torch.linspace(0.002, maximum_liquid_hydrogen, points):
        liquid = torch.stack((1.0 - hydrogen, hydrogen))
        point = binary_bubble_point(
            model,
            temperature,
            liquid,
            initial_pressure=previous_pressure,
            initial_vapor_composition=previous_vapor,
            minimum_pressure=1.0e3,
            maximum_pressure=3.0e8,
            tolerance=TOLERANCE,
            max_iterations=25,
        )
        results.append(point)
        if point.converged:
            previous_pressure = point.pressure.detach()
            previous_vapor = point.vapor_composition.detach()
            if float(point.vapor_composition[1] - hydrogen) <= 0.0:
                break
    return results


def trace_volume_isotherm(
    points: int = 25,
    maximum_liquid_hydrogen: float = 0.49,
):
    previous = None
    results = []
    for hydrogen in torch.linspace(0.002, maximum_liquid_hydrogen, points):
        liquid = torch.stack((1.0 - hydrogen, hydrogen))
        point = binary_helmholtz_bubble_point(
            model,
            temperature,
            liquid,
            initial_point=previous,
            minimum_pressure=1.0e3,
            maximum_pressure=3.0e8,
            tolerance=TOLERANCE,
            max_iterations=25,
        )
        if not point.converged and previous is not None:
            point = binary_helmholtz_bubble_point(
                model,
                temperature,
                liquid,
                initial_pressure=previous.pressure,
                initial_vapor_composition=previous.vapor_composition,
                minimum_pressure=1.0e3,
                maximum_pressure=3.0e8,
                tolerance=TOLERANCE,
                max_iterations=25,
            )
        results.append(point)
        if point.converged:
            previous = point
            if float(point.vapor_composition[1] - hydrogen) <= 0.0:
                break
    return results


pressure_curve_seconds, _ = timed(trace_pressure_isotherm, warmup=0, repeats=1)
volume_curve_seconds, _ = timed(trace_volume_isotherm, warmup=1, repeats=3)
pressure_curve = trace_pressure_isotherm()
volume_curve = trace_volume_isotherm()
curve_pressure = torch.stack([point.pressure for point in pressure_curve])
volume_pressure = torch.stack([point.pressure for point in volume_curve])
curve_metrics = pd.Series(
    {
        "points": len(volume_curve),
        "pressure curve / s": pressure_curve_seconds,
        "volume curve / s": volume_curve_seconds,
        "curve speedup": pressure_curve_seconds / volume_curve_seconds,
        "maximum relative pressure difference": float(
            ((volume_pressure - curve_pressure) / curve_pressure).abs().max()
        ),
        "maximum volume residual": max(
            float(point.residual_norm) for point in volume_curve
        ),
        "all volume points converged": all(point.converged for point in volume_curve),
    },
    name="value",
)
display(curve_metrics.to_frame())
assert curve_metrics["all volume points converged"]
assert curve_metrics["maximum relative pressure difference"] < 2.0e-8

# %%
eoscg = eoscg2021(("carbon_dioxide", "hydrogen"))
eoscg_temperature = torch.tensor(260.0)
eoscg_initial = binary_helmholtz_bubble_point(
    eoscg,
    eoscg_temperature,
    torch.tensor([0.98, 0.02]),
    minimum_pressure=1.0e3,
    maximum_pressure=3.0e8,
)
eoscg_liquid = torch.tensor([0.94, 0.06])


def eoscg_volume_point():
    return binary_helmholtz_bubble_point(
        eoscg,
        eoscg_temperature,
        eoscg_liquid,
        initial_point=eoscg_initial,
        minimum_pressure=1.0e3,
        maximum_pressure=3.0e8,
    )


def eoscg_pressure_point():
    return binary_bubble_point(
        eoscg,
        eoscg_temperature,
        eoscg_liquid,
        initial_pressure=eoscg_initial.pressure,
        initial_vapor_composition=eoscg_initial.vapor_composition,
        minimum_pressure=1.0e3,
        maximum_pressure=3.0e8,
    )


eoscg_volume_seconds, _ = timed(eoscg_volume_point)
eoscg_pressure_seconds, _ = timed(eoscg_pressure_point)
eoscg_volume_result = eoscg_volume_point()
eoscg_pressure_result = eoscg_pressure_point()
eoscg_metrics = pd.Series(
    {
        "EOS-CG pressure formulation / s": eoscg_pressure_seconds,
        "EOS-CG volume formulation / s": eoscg_volume_seconds,
        "EOS-CG speedup": eoscg_pressure_seconds / eoscg_volume_seconds,
        "EOS-CG relative pressure difference": float(
            (eoscg_volume_result.pressure - eoscg_pressure_result.pressure).abs()
            / eoscg_pressure_result.pressure
        ),
        "EOS-CG volume residual": float(eoscg_volume_result.residual_norm),
    },
    name="value",
)
display(eoscg_metrics.to_frame())
assert eoscg_initial.converged and eoscg_volume_result.converged
assert eoscg_metrics["EOS-CG relative pressure difference"] < 3.0e-8

# %% [markdown]
# ## Native Python and PyTorch profiles
#
# Both profiles cover the same warmed continued point. `cProfile` exposes
# Python and transform overhead; the PyTorch profiler resolves dispatcher and
# autograd operations.

# %%
native_profiler = cProfile.Profile()
native_profiler.enable()
profiled_result = volume_point()
native_profiler.disable()
native_stats = pstats.Stats(native_profiler)
native_rows = []
for (filename, line, function), (
    _primitive_calls,
    total_calls,
    total_time,
    cumulative_time,
    _,
) in native_stats.stats.items():
    native_rows.append(
        {
            "function": f"{os.path.basename(filename)}:{line}({function})",
            "calls": total_calls,
            "self / ms": 1.0e3 * total_time,
            "cumulative / ms": 1.0e3 * cumulative_time,
        }
    )
native_profile = (
    pd.DataFrame(native_rows)
    .sort_values("cumulative / ms", ascending=False)
    .head(15)
    .reset_index(drop=True)
)
display(native_profile)

with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
) as torch_profile:
    profiled_result = volume_point()
print(
    torch_profile.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=15,
    )
)
assert profiled_result.converged

# %% [markdown]
# The remaining cost is dominated by many tiny elementwise operations and
# reverse-mode engine calls, rather than dense linear algebra. That profile is
# why the implementation fuses exact density derivatives, skips absent
# critical terms, reuses reducing functions, and refreshes the small Newton
# Jacobian only periodically.

# %% [markdown]
# ## Forward versus reverse automatic differentiation
#
# The direct volume residual maps three unknowns to three equations, so output
# and input dimensions alone do not predict a winner. Its two chemical
# potentials are themselves scalar-to-vector reverse gradients. We therefore
# time the complete nested transform and verify the Jacobians agree.

# %%
pressure_scale = initial_volume_point.pressure.detach()
initial_variables = torch.stack(
    (
        torch.logit(initial_volume_point.vapor_composition[0]),
        torch.log(initial_volume_point.liquid_molar_volume.reciprocal()),
        torch.log(initial_volume_point.vapor_molar_volume.reciprocal()),
    )
)


def volume_residual(current: Tensor) -> Tensor:
    vapor_first = torch.sigmoid(current[0])
    vapor = torch.stack((vapor_first, 1.0 - vapor_first))
    liquid_volume = torch.exp(-current[1])
    vapor_volume = torch.exp(-current[2])
    liquid_mu = torch.func.grad(
        lambda moles: model.residual_helmholtz_rt(
            temperature,
            liquid_volume,
            moles,
        ).sum()
    )(liquid_next)
    vapor_mu = torch.func.grad(
        lambda moles: model.residual_helmholtz_rt(
            temperature,
            vapor_volume,
            moles,
        ).sum()
    )(vapor)
    liquid_pressure = model.pressure(temperature, liquid_volume, liquid_next)
    vapor_pressure = model.pressure(temperature, vapor_volume, vapor)
    return torch.cat(
        (
            torch.log(liquid_next / liquid_volume)
            + liquid_mu
            - torch.log(vapor / vapor_volume)
            - vapor_mu,
            ((liquid_pressure - vapor_pressure) / pressure_scale).reshape(1),
        )
    )


reverse_jacobian = torch.func.jacrev(volume_residual)
forward_jacobian = torch.func.jacfwd(volume_residual)
reverse_seconds, _ = timed(
    lambda: reverse_jacobian(initial_variables),
    repeats=20,
)
forward_seconds, _ = timed(
    lambda: forward_jacobian(initial_variables),
    repeats=20,
)
reverse_value = reverse_jacobian(initial_variables)
forward_value = forward_jacobian(initial_variables)
ad_metrics = pd.Series(
    {
        "reverse Jacobian / ms": 1.0e3 * reverse_seconds,
        "forward Jacobian / ms": 1.0e3 * forward_seconds,
        "forward / reverse time": forward_seconds / reverse_seconds,
        "maximum Jacobian difference": float(
            (reverse_value - forward_value).abs().max()
        ),
    },
    name="value",
)
display(ad_metrics.to_frame())
torch.testing.assert_close(
    reverse_value,
    forward_value,
    rtol=3.0e-12,
    atol=3.0e-12,
)

# %%
temperature_step = torch.tensor(1.0e-4)


def continued_pressure(current_temperature):
    return binary_helmholtz_bubble_point(
        model,
        current_temperature,
        liquid_next,
        initial_point=initial_volume_point,
        tolerance=1.0e-10,
        max_iterations=30,
    ).pressure


autodiff_temperature = torch.func.grad(continued_pressure)(temperature)
finite_difference_temperature = (
    continued_pressure(temperature + temperature_step)
    - continued_pressure(temperature - temperature_step)
) / (2.0 * temperature_step)
derivative_metrics = pd.Series(
    {
        "autodiff dP/dT / Pa K-1": float(autodiff_temperature),
        "finite-difference dP/dT / Pa K-1": float(finite_difference_temperature),
        "relative difference": float(
            (autodiff_temperature - finite_difference_temperature).abs()
            / finite_difference_temperature.abs()
        ),
    },
    name="value",
)
display(derivative_metrics.to_frame())
assert derivative_metrics["relative difference"] < 3.0e-8

# %% [markdown]
# ## `torch.compile`: cold cost versus warmed fusion
#
# PyTorch can compile a transform when the outer `jacrev` is included in the
# compiled callable. Compiling only the pressure function and then applying
# eager `jacrev` outside is unsupported. The experiment below records the
# one-time compilation cost separately from warmed latency.

# %%
state_variables = torch.tensor([90.8, -9.9, 0.2, 0.8])


def pressure_from_state(current):
    return model.pressure(
        current[0],
        torch.exp(current[1]),
        torch.softmax(current[2:], dim=0),
    )


eager_pressure_jacobian = torch.func.jacrev(pressure_from_state)
original_inductor_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
with tempfile.TemporaryDirectory(prefix="torch-flash-inductor-") as cache_directory:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_directory
    torch._dynamo.reset()
    compiled_pressure_jacobian = torch.compile(
        eager_pressure_jacobian,
        fullgraph=True,
    )
    compile_started = time.perf_counter()
    compiled_first = compiled_pressure_jacobian(state_variables)
    compile_seconds = time.perf_counter() - compile_started
    compiled_seconds, _ = timed(
        lambda: compiled_pressure_jacobian(state_variables),
        repeats=100,
    )
if original_inductor_cache is None:
    del os.environ["TORCHINDUCTOR_CACHE_DIR"]
else:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = original_inductor_cache
eager_jacobian_seconds, _ = timed(
    lambda: eager_pressure_jacobian(state_variables),
    repeats=100,
)
eager_value = eager_pressure_jacobian(state_variables)
compile_metrics = pd.Series(
    {
        "cold compile plus first call / s": compile_seconds,
        "warmed compiled Jacobian / us": 1.0e6 * compiled_seconds,
        "eager Jacobian / us": 1.0e6 * eager_jacobian_seconds,
        "warmed speedup": eager_jacobian_seconds / compiled_seconds,
        "maximum relative Jacobian difference": float(
            (
                (compiled_first - eager_value).abs() / eager_value.abs().clamp_min(1.0)
            ).max()
        ),
    },
    name="value",
)
display(compile_metrics.to_frame())
torch.testing.assert_close(
    compiled_first,
    eager_value,
    rtol=3.0e-12,
    atol=2.0e-7,
)

# %% [markdown]
# ## Numba/PyTorch coupling experiment
#
# Numba 0.66 can fuse the many small GERG term operations into one native CPU
# kernel. The prototype below specializes immutable H2-tailored GERG arrays
# and uses a zero-copy NumPy view of a CPU tensor. It intentionally evaluates
# only the regular residual terms present in this two-component model.
#
# This is not a production backend: captured coefficients are invisible to
# PyTorch, the function is CPU-only, and no parameter or state derivative is
# registered. Current PyTorch integration requires a custom operator plus fake,
# autograd, and often `vmap` registrations, followed by `opcheck` and
# `gradcheck`. A Numba CUDA kernel can exchange device memory through the CUDA
# Array Interface, but stream synchronization still has to be correct.
#
# Official references:
#
# - PyTorch custom operators and registered autograd:
#   <https://docs.pytorch.org/docs/stable/library.html>
# - PyTorch custom-operator workflow:
#   <https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html>
# - Numba CUDA Array Interface:
#   <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>
# - Numba performance guidance:
#   <https://numba.readthedocs.io/en/stable/user/performance-tips.html>

# %%
tc = model.critical_temperature.numpy()
rhoc = model.critical_density.numpy()
beta_t = model.beta_temperature.numpy()
gamma_t = model.gamma_temperature.numpy()
beta_v = model.beta_volume.numpy()
gamma_v = model.gamma_volume.numpy()
departure_scale = model.departure_scale.numpy()
tc_pair = model._critical_temperature_pair.numpy()
inverse_density_pair = model._inverse_density_pair.numpy()
pure_n = model.pure_n.numpy()
pure_d = model.pure_d.numpy()
pure_t = model.pure_t.numpy()
pure_decay = model.pure_decay.numpy()
pure_eta = model.pure_eta.numpy()
pure_epsilon = model.pure_epsilon.numpy()
pure_beta = model.pure_beta.numpy()
pure_gamma = model.pure_gamma.numpy()
pure_linear_density = model.pure_linear_density.numpy()
pure_linear_shift = model.pure_linear_shift.numpy()
departure_n = model.departure_n.numpy()
departure_d = model.departure_d.numpy()
departure_t = model.departure_t.numpy()
departure_decay = model.departure_decay.numpy()
departure_eta = model.departure_eta.numpy()
departure_epsilon = model.departure_epsilon.numpy()
departure_beta = model.departure_beta.numpy()
departure_gamma = model.departure_gamma.numpy()
departure_linear_density = model.departure_linear_density.numpy()
departure_linear_shift = model.departure_linear_shift.numpy()
gas_constant = float(model.gas_constant)


@njit(cache=False, fastmath=False)
def numba_pressure(temperature_value, molar_volume_value, input_composition):
    """Specialized regular-term pressure kernel for this two-component model."""
    composition = np.empty(2, np.float64)
    total = input_composition[0] + input_composition[1]
    composition[0] = input_composition[0] / total
    composition[1] = input_composition[1] / total
    reducing_temperature = 0.0
    inverse_reducing_density = 0.0
    for i in range(2):
        reducing_temperature += composition[i] ** 2 * tc[i]
        inverse_reducing_density += composition[i] ** 2 / rhoc[i]
    for i in range(2):
        for j in range(i + 1, 2):
            temperature_fraction = (
                2.0
                * composition[i]
                * composition[j]
                * beta_t[i, j]
                * gamma_t[i, j]
                * (composition[i] + composition[j])
                / (beta_t[i, j] ** 2 * composition[i] + composition[j])
            )
            volume_fraction = (
                2.0
                * composition[i]
                * composition[j]
                * beta_v[i, j]
                * gamma_v[i, j]
                * (composition[i] + composition[j])
                / (beta_v[i, j] ** 2 * composition[i] + composition[j])
            )
            reducing_temperature += temperature_fraction * tc_pair[i, j]
            inverse_reducing_density += volume_fraction * inverse_density_pair[i, j]
    density = 1.0 / molar_volume_value
    reducing_density = 1.0 / inverse_reducing_density
    delta = density / reducing_density
    tau = reducing_temperature / temperature_value
    alpha_delta = 0.0
    for i in range(2):
        row = 0.0
        for k in range(pure_n.shape[1]):
            active_decay = pure_decay[i, k] != 0.0
            exponent = (
                -(delta ** pure_decay[i, k] if active_decay else 0.0)
                - pure_eta[i, k] * (delta - pure_epsilon[i, k]) ** 2
                - pure_beta[i, k] * (tau - pure_gamma[i, k]) ** 2
                - pure_linear_density[i, k] * (delta - pure_linear_shift[i, k])
            )
            term = (
                pure_n[i, k]
                * delta ** pure_d[i, k]
                * tau ** pure_t[i, k]
                * math.exp(exponent)
            )
            logarithmic_derivative = (
                pure_d[i, k] / delta
                - (
                    pure_decay[i, k] * delta ** (pure_decay[i, k] - 1.0)
                    if active_decay
                    else 0.0
                )
                - 2.0 * pure_eta[i, k] * (delta - pure_epsilon[i, k])
                - pure_linear_density[i, k]
            )
            row += term * logarithmic_derivative
        alpha_delta += composition[i] * row
    for i in range(2):
        for j in range(i + 1, 2):
            row = 0.0
            for k in range(departure_n.shape[2]):
                active_decay = departure_decay[i, j, k] != 0.0
                exponent = (
                    -(delta ** departure_decay[i, j, k] if active_decay else 0.0)
                    - departure_eta[i, j, k] * (delta - departure_epsilon[i, j, k]) ** 2
                    - departure_beta[i, j, k] * (tau - departure_gamma[i, j, k]) ** 2
                    - departure_linear_density[i, j, k]
                    * (delta - departure_linear_shift[i, j, k])
                )
                term = (
                    departure_n[i, j, k]
                    * delta ** departure_d[i, j, k]
                    * tau ** departure_t[i, j, k]
                    * math.exp(exponent)
                )
                logarithmic_derivative = (
                    departure_d[i, j, k] / delta
                    - (
                        departure_decay[i, j, k]
                        * delta ** (departure_decay[i, j, k] - 1.0)
                        if active_decay
                        else 0.0
                    )
                    - 2.0
                    * departure_eta[i, j, k]
                    * (delta - departure_epsilon[i, j, k])
                    - departure_linear_density[i, j, k]
                )
                row += term * logarithmic_derivative
            alpha_delta += composition[i] * composition[j] * departure_scale[i, j] * row
    return gas_constant * temperature_value * density * (1.0 + delta * alpha_delta)


numba_temperature = torch.tensor(90.8)
numba_volume = initial_volume_point.liquid_molar_volume.detach()
numba_composition = liquid_start.detach()
compile_started = time.perf_counter()
numba_first = numba_pressure(
    float(numba_temperature),
    float(numba_volume),
    numba_composition.numpy(),
)
numba_compile_seconds = time.perf_counter() - compile_started
numba_kernel_seconds, _ = timed(
    lambda: numba_pressure(
        float(numba_temperature),
        float(numba_volume),
        numba_composition.numpy(),
    ),
    repeats=20_000,
)
torch_pressure_seconds, _ = timed(
    lambda: model.pressure(
        numba_temperature,
        numba_volume,
        numba_composition,
    ),
    repeats=3_000,
)
torch_pressure_value = model.pressure(
    numba_temperature,
    numba_volume,
    numba_composition,
)
numba_metrics = pd.Series(
    {
        "Numba cold compile plus first call / s": numba_compile_seconds,
        "Numba plus Tensor boundary / us": 1.0e6 * numba_kernel_seconds,
        "PyTorch eager pressure / us": 1.0e6 * torch_pressure_seconds,
        "forward-only warmed speedup": torch_pressure_seconds / numba_kernel_seconds,
        "absolute pressure difference / Pa": abs(
            numba_first - float(torch_pressure_value)
        ),
        "relative pressure difference": abs(numba_first - float(torch_pressure_value))
        / abs(float(torch_pressure_value)),
    },
    name="value",
)
display(numba_metrics.to_frame())
assert numba_metrics["relative pressure difference"] < 3.0e-12


# %%
@torch.library.custom_op(
    "torch_flash_experiment::gerg_pressure_numba",
    mutates_args=(),
    device_types="cpu",
)
def gerg_pressure_numba(
    temperature_input: Tensor,
    volume_input: Tensor,
    composition_input: Tensor,
) -> Tensor:
    value = numba_pressure(
        float(temperature_input),
        float(volume_input),
        composition_input.detach().numpy(),
    )
    return temperature_input.new_tensor(value)


@gerg_pressure_numba.register_fake
def _(
    temperature_input: Tensor,
    volume_input: Tensor,
    composition_input: Tensor,
) -> Tensor:
    return torch.empty_like(temperature_input)


torch.library.opcheck(
    gerg_pressure_numba,
    (numba_temperature, numba_volume, numba_composition),
    test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
)
custom_value = gerg_pressure_numba(
    numba_temperature,
    numba_volume,
    numba_composition,
)
torch.testing.assert_close(
    custom_value,
    torch_pressure_value,
    rtol=3.0e-12,
    atol=2.0e-7,
)
gradient_error = None
try:
    differentiable_temperature = numba_temperature.clone().requires_grad_()
    gerg_pressure_numba(
        differentiable_temperature,
        numba_volume,
        numba_composition,
    ).backward()
except RuntimeError as error:
    gradient_error = str(error).splitlines()[0]
print("Expected missing-autograd result:", gradient_error)
assert gradient_error is not None

# %% [markdown]
# ## Thread-level branch parallelism
#
# Tiny nested-autodiff systems are latency-bound and contend in Python when
# dispatched through a thread pool. Three independent volume curves are
# compared against sequential execution; this determines the notebook default
# rather than assuming more host threads help.

# %%
temperatures = (70.4, 90.8, 110.3)


def independent_curve(temperature_value):
    local_model = gerg2008_hydrogen_2021(("nitrogen", "hydrogen"))
    local_temperature = torch.tensor(temperature_value)
    previous = None
    for hydrogen in torch.linspace(0.002, 0.60, 12):
        liquid = torch.stack((1.0 - hydrogen, hydrogen))
        point = binary_helmholtz_bubble_point(
            local_model,
            local_temperature,
            liquid,
            initial_point=previous,
            minimum_pressure=1.0e3,
            maximum_pressure=3.0e8,
            tolerance=TOLERANCE,
            max_iterations=25,
        )
        if point.converged:
            previous = point
    return previous


started = time.perf_counter()
sequential_points = [independent_curve(value) for value in temperatures]
sequential_seconds = time.perf_counter() - started
started = time.perf_counter()
with ThreadPoolExecutor(max_workers=3) as executor:
    threaded_points = list(executor.map(independent_curve, temperatures))
threaded_seconds = time.perf_counter() - started
thread_metrics = pd.Series(
    {
        "three sequential curves / s": sequential_seconds,
        "three threaded curves / s": threaded_seconds,
        "threaded speedup": sequential_seconds / threaded_seconds,
        "all sequential endpoints converged": all(
            point is not None and point.converged for point in sequential_points
        ),
        "all threaded endpoints converged": all(
            point is not None and point.converged for point in threaded_points
        ),
    },
    name="value",
)
display(thread_metrics.to_frame())

# %% [markdown]
# ## Conclusion
#
# The default remains native PyTorch:
#
# - exact analytical density differentiation and volume continuation remove
#   the dominant nested density work while retaining parameter/state autodiff;
# - reverse mode is the correct choice for the nested GERG residual on this
#   workload;
# - `torch.compile` has a large warmed benefit, but its cold cost only amortizes
#   in long-lived repeated workloads;
# - Numba demonstrates that scalar forward fusion has substantial headroom,
#   but a complete backend must implement and test state and parameter
#   gradients, higher-order derivatives needed by phase equilibrium, fake
#   tensors, and batching. The forward-only prototype is therefore evidence,
#   not a package backend.
#
# The next structural optimization, if repeated warmed envelopes dominate a
# deployment, is a cached compiled residual-plus-Jacobian object or a fused
# custom operator with an independently verified analytic backward. A
# Numba-only EOS is not substituted for the differentiable PyTorch model.
