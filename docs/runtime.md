# Runtime configuration

Call `configure()` before constructing models and state tensors to select the
process-wide native `torch-flash` tensor policy:

```python
import torch

from torch_flash import ChemicalState, component_set, configure, peng_robinson_1978

runtime = configure(
    device="cpu",
    dtype=torch.float64,
    num_threads=8,
    deterministic=False,
)
model = peng_robinson_1978(component_set(("methane", "n_butane")))
state = ChemicalState(
    runtime.tensor(270.0),
    runtime.tensor(3.0e6),
    runtime.tensor([0.5, 0.5]),
)
```

The returned `RuntimeConfig` is an immutable snapshot. Its `tensor()` and
`as_tensor()` helpers create application data on the same device and with the
same dtype as the model. `get_config()` retrieves the current policy and actual
PyTorch thread/determinism settings.

## Device selection

`device` accepts a PyTorch device or one of two selectors:

- `"auto"` tries CUDA, XPU, and MPS in that order, then falls back to CPU.
- `"gpu"` tries the same accelerators but raises if none supports the selected
  dtype.

Availability is checked by a real zero-length allocation with the requested
dtype. Consequently, `configure(device="auto", dtype=torch.float64)` falls
back to CPU rather than selecting an accelerator that cannot represent
float64. An explicit unavailable or incompatible device raises before a model
is constructed.

```python
runtime = configure(device="gpu", dtype=torch.float64)
model = component_set(("hydrogen", "methane"))  # allocated on runtime.device

temperature = torch.linspace(
    280.0,
    400.0,
    4096,
    **runtime.tensor_options,
)
```

The policy is read once by named component, cubic, activity, CPA, GERG,
EOS-CG, standard-state, and characterization factories. Explicit factory
arguments override the corresponding global option:

```python
configure(device="cuda", dtype=torch.float64)
cpu_components = component_set(
    ("methane", "n_butane"),
    device="cpu",
    dtype=torch.float32,
)
```

Changing the configuration does not move existing models or optimizer
parameters. Use the normal PyTorch `model.to(...)` operation when a deliberate
post-construction move is needed. Low-level model constructors that accept
explicit tensors continue to follow those tensors.

`configure()` sets PyTorch's default floating dtype, but deliberately does not
call
[`torch.set_default_device`](https://docs.pytorch.org/docs/stable/generated/torch.set_default_device.html).
PyTorch documents a small cost on every Python API call after changing its
global default device. Use `RuntimeConfig.tensor()`, `as_tensor()`, or
`tensor_options` for state and batch construction instead. This keeps device
placement explicit and avoids adding dispatch overhead to thermodynamic hot
paths.

## CPU threading

`num_threads` controls intra-operation CPU parallelism. It maps to
[`torch.set_num_threads`](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html),
which PyTorch requires to be called before eager, JIT, or autograd work for
reliable effect. Large batches can benefit from several threads; scalar and
small-state workloads can be slower when over-threaded, so benchmark the
deployed workload and hardware.

`num_interop_threads` controls inter-operation parallelism. PyTorch permits
[`torch.set_num_interop_threads`](https://docs.pytorch.org/docs/stable/generated/torch.set_num_interop_threads.html)
to be called only once and before inter-operation parallel work starts.
`torch-flash` therefore changes it only when the requested value differs and
raises an explanatory error if configuration occurs too late. It cannot be
safely used as a temporary or repeatedly tuned setting.

Both thread controls are process-wide PyTorch state, including for code outside
`torch-flash`. Configure each spawned worker process independently.

## Determinism and precision

`deterministic=True` calls `torch.use_deterministic_algorithms`. By default an
operation without a deterministic implementation raises;
`deterministic_warn_only=True` changes that behavior to a warning. Determinism
can reduce performance and does not guarantee identical floating-point results
across PyTorch versions, platforms, or CPU/GPU devices, as stated in the
[PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).
It also does not seed PyTorch, NumPy, or Python random generators.

Only float32 and float64 are accepted as global thermodynamic policies.
Float64 remains the default and is strongly recommended for phase boundaries,
near-critical roots, stability calculations, and parameter regression.
Float32 accelerator runs are an explicit accuracy/performance tradeoff and
should be validated against float64 for the intended state range.

External ThermoPack, NeqSim, teqp, and CoolProp backends retain their own
runtime controls; the configuration applies to native PyTorch models and
torch-flash-created tensors.
