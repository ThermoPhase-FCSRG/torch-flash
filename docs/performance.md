# Performance

`torch-flash` is optimized for differentiable and batched thermodynamic work.
It does not claim to beat compiled Fortran/C++ libraries on a one-off scalar
state.

## Execution paths

- CPA site fractions broadcast over leading state dimensions. Ten damped
  fixed-point updates are followed by eight Newton updates using the analytic
  mass-action Jacobian and a positivity-limited step.
- CPA's conservative volume grid is evaluated as one tensor workload.
  Explicit liquid/vapor roots first use a phase-specific Newton solve; the
  complete scan remains the fallback and stable-root path.
- GERG-2008 and EOS-CG-2021 reducing functions, pure/departure Helmholtz terms,
  ideal terms, pressure, and caloric derivatives broadcast over state axes.
- Batched multifluid density roots use a fixed vector workload. Root location
  is detached from the graph; if an input or fitted parameter requires a
  gradient, one exact Newton correction restores the implicit derivative.
- Composition validation remains strict in eager calls. Compiled graphs assume
  that documented input contract so the numerical normalization can be fused.

## Recommended use

```python
import torch

from torch_flash import configure
from torch_flash.eos import gerg2008

runtime = configure(device="cpu", dtype=torch.float64, num_threads=8)
model = gerg2008(("hydrogen", "methane"))

temperature = torch.linspace(300.0, 400.0, 1010, **runtime.tensor_options)
pressure = torch.linspace(1.0e4, 100.0e6, 1010, **runtime.tensor_options)
hydrogen = torch.linspace(0.1, 0.9, 1010, **runtime.tensor_options)
composition = torch.stack((hydrogen, 1.0 - hydrogen), dim=-1)

volume = model.molar_volume(temperature, pressure, composition, "vapor")

compiled_pressure = torch.compile(
    lambda current_temperature, current_volume, current_composition: model.pressure(
        current_temperature,
        current_volume,
        current_composition,
    ),
    fullgraph=True,
)
compiled_pressure(temperature, volume, composition)
```

Compile a repeated homogeneous kernel, not a one-off call.
[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
uses TorchInductor by default and its first call can take seconds. Validate
untrusted compositions with an eager call before passing them to a compiled
hot loop.

Set CPU intra-operation threads with `configure(num_threads=...)` before
thermodynamic work. More threads help sufficiently large batches, but small
kernels plateau quickly. Do not change the process-wide thread count inside a
library call. The [runtime guide](runtime.md) also covers the stricter
one-time inter-operation thread setting.

## Executed benchmark scope

[`16_performance_backends.ipynb`](https://github.com/volpatto/torch-flash/blob/main/notebooks/performance/16_performance_backends.ipynb)
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
[`23_synthetic_co2_pr78_derivatives.ipynb`](https://github.com/volpatto/torch-flash/blob/main/notebooks/verification/23_synthetic_co2_pr78_derivatives.ipynb)
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
