# High-performance setup

`torch-flash` is optimized for differentiable and batched thermodynamic work.
It does not claim to beat compiled Fortran/C++ libraries on a one-off scalar
state.

## Setup workflow

Native `torch-flash` models use PyTorch tensors and therefore inherit the
execution device, batching, automatic differentiation, and compilation
capabilities of the installed PyTorch build. Acceleration is not enabled by a
separate `torch-flash` backend switch.

### Select the environment

For CPU execution, the default installation is sufficient:

```bash
python -m pip install torch-flash
```

For a repository-managed NVIDIA GPU environment:

```bash
pixi install -e gpu
pixi run -e gpu python your_calculation.py
```

For pip installations, first select a PyTorch build compatible with the
machine and accelerator using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/),
then install the optional supporting packages if they are required by the
application:

```bash
python -m pip install "torch-flash[gpu]"
```

The `gpu` extra adds `nvmath-python` and `torch-sla`; it does not replace
PyTorch, install a GPU driver, move existing tensors, or automatically route a
calculation through those packages. Native CUDA execution only requires a
working accelerator-enabled PyTorch installation. Confirm the actual runtime
before benchmarking:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
```

### Configure once, then construct

Call `configure()` before creating components, models, state tensors, or
worker-process workloads. Float64 is the reference precision for equilibrium,
derivative, and near-critical calculations:

```python
import torch

from torch_flash import configure
from torch_flash.eos import gerg2008

runtime = configure(
    device="cpu",        # use "gpu" to require a compatible accelerator
    dtype=torch.float64,
    num_threads=8,
    deterministic=False,
)
model = gerg2008(("hydrogen", "methane"))
```

On CPU, benchmark `num_threads` with the deployed batch shape; the physical
core count is not automatically the fastest setting. Configure each spawned
worker separately and avoid oversubscribing cores by combining many worker
processes with many PyTorch threads. On an accelerator, construct model and
input tensors on the final device and keep subsequent calculations there.
Repeated host/device transfers, `.cpu()`, `.numpy()`, and `.item()` calls
synchronize execution and can dominate small thermodynamic kernels.

### Batch independent states

Leading tensor dimensions represent independent states for the kernels that
document batching support. Prefer one tensor call over a Python loop:

```python
temperature = torch.linspace(300.0, 400.0, 1010, **runtime.tensor_options)
pressure = torch.linspace(1.0e4, 100.0e6, 1010, **runtime.tensor_options)
hydrogen = torch.linspace(0.1, 0.9, 1010, **runtime.tensor_options)
composition = torch.stack((hydrogen, 1.0 - hydrogen), dim=-1)

volume = model.molar_volume(temperature, pressure, composition, "vapor")
```

Use `batched_two_phase_flash` only for independent states already known to be
two-phase and satisfying its documented initialization contract. Full
stability analysis, phase-envelope continuation, and branch-following remain
scientifically different workloads and may require sequential decisions.

### Compile a stable repeated kernel

Compile tensor kernels that are called repeatedly with stable shapes. Keep an
eager result as the correctness reference, and measure compilation separately
from warmed execution:

```python
def pressure_kernel(current_temperature, current_volume, current_composition):
    return model.pressure(
        current_temperature,
        current_volume,
        current_composition,
    )


eager_pressure = pressure_kernel(temperature, volume, composition)
compiled_pressure = torch.compile(pressure_kernel, fullgraph=True)

# First call includes tracing and compilation.
compiled_result = compiled_pressure(temperature, volume, composition)
torch.testing.assert_close(compiled_result, eager_pressure)

# Time repeated calls only after this warm-up.
compiled_pressure(temperature, volume, composition)
```

[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
uses TorchInductor by default. Graph breaks, changing shapes, and one-off calls
can erase its benefit. Validate untrusted inputs eagerly before sending them to
a compiled hot loop because compiled thermodynamic kernels assume the
documented input contract.

Use the default `fullgraph=False` when first compiling an application-level
workflow. Use `fullgraph=True` as an audit when a selected tensor kernel is
expected to form one graph. PyTorch's
[compile programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html)
documents guards, graph breaks, and recompilation.

For forward-only property evaluation where no derivatives or parameter
gradients will be requested, benchmark `torch.inference_mode()`. Do not use it
around fitting, sensitivity, or differentiable flash calculations.

### Measure the real workload

Record cold construction/compilation separately from warmed latency. Use
representative batch shapes, dtype, device, thread count, derivative order,
and phase region. PyTorch's
[`torch.utils.benchmark`](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
performs warm-up and accelerator synchronization; manual CUDA timings must
synchronize before reading the clock. Always report pressure, fugacity,
material-balance, and convergence residuals with timing.

CUDA execution is asynchronous, so an unsynchronized wall clock can stop
before the kernel has completed. Use CUDA events or synchronize around the
measured region, following PyTorch's
[CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution):

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
compiled_pressure(temperature, volume, composition)
end.record()
torch.cuda.synchronize(runtime.device)
print(f"{start.elapsed_time(end):.3f} ms")
```

The practical order of optimization is:

1. verify the model, phase branch, dtype, and residuals;
2. remove scalar Python loops by batching independent states;
3. reuse model and tensor allocations;
4. tune CPU threads or keep a sufficiently large workload device-resident;
5. compile the repeated tensor kernel; and
6. profile again before considering a more complex backend.

## Execution paths

- CPA site fractions broadcast over leading state dimensions. Ten damped
  fixed-point updates are followed by eight Newton updates using the analytic
  mass-action Jacobian and a positivity-limited step.
- CPA's conservative volume grid is evaluated as one tensor workload.
  Explicit liquid/vapor roots first use a phase-specific Newton solve; the
  complete scan remains the fallback and stable-root path.
- GERG-2008 and EOS-CG-2021 reducing functions, pure/departure Helmholtz terms,
  ideal terms, pressure, and caloric derivatives broadcast over state axes.
- Batched multiparameter density roots use a fixed vector workload. Root location
  is detached from the graph; if an input or fitted parameter requires a
  gradient, one exact Newton correction restores the implicit derivative.
- Composition validation remains strict in eager calls. Compiled graphs assume
  that documented input contract so the numerical normalization can be fused.

Set CPU intra-operation threads with `configure(num_threads=...)` before
thermodynamic work. More threads help sufficiently large batches, but small
kernels plateau quickly. Do not change the process-wide thread count inside a
library call. The [runtime guide](runtime.md) also covers the stricter
one-time inter-operation thread setting.

### Dense trust-region scope

The Nichita-style stability and flash path uses exact dense PyTorch Hessians
and `torch.linalg.eigh`. This matches the small compositional systems for which
the method was proposed. `torch-sla` is not selected automatically: its sparse
adjoint solve is better aligned with large sparse implicit systems than with
independent two-to-four-variable flash systems. A future coupled sparse
transport/flash problem may justify that backend, but using it for the current
dense subproblems would add conversion and dependency overhead without
removing the thermodynamic Hessian evaluations.

An Apple M4 Pro, float64 CPU, four-thread benchmark with PyTorch 2.12.1
separated ordinary and higher-iteration methane/n-butane PR78 states. Each
reported state timing is a warmed median of three repetitions. The complete
matched runner is
[`scripts/benchmark_trust_region_flash.py`](https://github.com/ThermoPhase-FCSRG/torch-flash/blob/main/scripts/benchmark_trust_region_flash.py).

| Two-phase group | States | \(\ln K\) iterations | Trust-region iterations | \(\ln K\) median / ms | Trust-region median / ms |
|---|---:|---:|---:|---:|---:|
| Ordinary | 12 | 14.0 | 8.5 | 12.57 | 230.35 |
| Higher-iteration | 12 | 15.0 | 7.0 | 16.40 | 191.32 |

Both methods converged on all 24 states. The largest phase-fraction difference
was \(6.5\times10^{-9}\), and the largest trust-region fugacity residual was
\(8.5\times10^{-9}\). Fewer nonlinear iterations therefore do not imply lower
latency when every iteration assembles an exact Hessian.

The complementary three-phase test is where the method helps. At the
methane/CO2 SRK three-phase invariant at 180 K and 2.737 MPa, from a deliberately
offset but physical three-phase initialization, the generalized-substitution
path did not converge in 100 iterations (3.95 s, residual 0.18). The
direct-mole trust-region path recovered all three invariant compositions in 32
iterations (1.20 s), with a \(5.1\times10^{-9}\) chemical-potential residual,
a maximum phase-composition difference of \(2.2\times10^{-10}\) from the
invariant solution, and exact component material balance. This supports an opt-in difficult
multiphase path, not replacing the default ordinary two-phase flash.

## Executed benchmark scope

[`16_performance_backends.ipynb`](https://github.com/ThermoPhase-FCSRG/torch-flash/blob/main/notebooks/performance/16_performance_backends.ipynb)
records hardware, package versions, cold compilation, warmed medians,
pressure/root residuals, and plots. Its Apple arm64 results are approximately:

| Workload | torch-flash eager | torch-flash compiled | ThermoPack | teqp | NeqSim |
|---|---:|---:|---:|---:|---:|
| GERG pressure, same coefficients | 1.0 ms | 38–40 µs | 6 µs | 1.1 µs | n/a |
| CPA pressure workload | 1.0 ms | 105–120 µs | 21–22 µs | n/a | n/a |
| GERG scalar TP root | 12 ms | n/a | 11–12 µs | 10–11 µs | 0.3–0.45 ms* |

\* The NeqSim call includes setting the state and running `TPflash`, so it is
broader than a single homogeneous density-root call.

The 1,010-state native GERG density solve reaches about 4,200 states/s with
maximum normalized pressure residual below `3e-15`. A 4,096-state pressure
kernel gains about 2× from 4–8 CPU intra-operation threads relative to one
thread on that host.

The executed CO2-H2O fitting notebook measures its exact 52-state
autodifferentiable GERG residual graph separately. Batched eager evaluation
takes `0.299 s`, versus `5.437 s` for an explicit scalar loop (`18.17×`);
objective plus backward takes `0.322 s`. The maximum batched/scalar residual
difference is `1.94e-10`. This is the workload used by the regression rather
than a synthetic pressure-only kernel.

The complete CO2/N2 PR78 envelope in
[`23_synthetic_co2_pr78_derivatives.ipynb`](https://github.com/ThermoPhase-FCSRG/torch-flash/blob/main/notebooks/verification/23_synthetic_co2_pr78_derivatives.ipynb)
is a different workload: 469 ordered bubble, dew, and fixed-log-K
continuation solves, each only three or four variables wide. The continuation
path uses a secant predictor, two successive-substitution corrections, and an
autodiff Newton solve. A failed prediction is retried with the conservative
full initializer. The recorded Apple arm64 run takes `4.713 s`.
`accelerated=False` selects full initialization at every continuation point
for numerical audits.

This envelope is intrinsically sequential: each physical solution selects the
branch for the next state. Moving the individual tiny Newton systems to a GPU
would add kernel-launch and host-synchronization overhead, while independently
batching all temperatures can converge to the algebraic \(K_i=1\) solution
instead of the physical branch. GPU execution remains appropriate for the
large homogeneous state grids evaluated after the envelope is known.

These figures are measurements, not CI thresholds. Exact latency depends on
hardware, PyTorch version, compiler cache, batch shape, composition, phase
region, and requested derivative order.

## GPU precision

The Pixi `gpu` environment targets CUDA and retains float64. GPU launch and
transfer overhead mean that batching and device-resident downstream work are
necessary to amortize costs. The benchmark checks CUDA float64 when available.

The Apple MPS runtime on the executed host rejects float64 tensors. A float32
MPS result is not an equal-accuracy thermodynamic comparison and is therefore
not included in the primary timing ratios. Near critical loci, spinodals, and
coalescing phase roots, float32 can be inadequate.
