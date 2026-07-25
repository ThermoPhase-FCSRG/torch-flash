# Batched grid flash and phase identification

`flash_grid` discovers one, two, or three phases independently at every point
of a non-scalar temperature-pressure batch. It is intended for phase maps and
other state sweeps where the phase count is not known in advance.

Phase equilibrium and physical phase identification are separate operations:

```text
batched feed states
       │
       ▼
  flash_grid
       │  equilibrium phase fractions and compositions
       ▼
identify_grid_phases
       │  liquid/vapor diagnostics for every returned phase
       ▼
 phase-region map
```

The flash determines compositions from material balance and fugacity equality.
The identification criteria label those compositions afterward; they do not
enter or alter the equilibrium equations.

## Complete grid example

Configure float64 and CPU threading before constructing tensors or the model.
Temperature is in K, pressure is in Pa, and composition is mole fraction:

```python
import torch

from torch_flash import (
    ChemicalState,
    GridFlashOptions,
    component_set,
    configure,
    flash_grid,
    identify_grid_phases,
    peng_robinson_1978,
)

runtime = configure(dtype=torch.float64, device="cpu", num_threads=1)
components = component_set(
    ("methane", "n_butane"),
    **runtime.tensor_options,
)
model = peng_robinson_1978(components)

temperature_axis = torch.linspace(180.0, 420.0, 50, **runtime.tensor_options)
pressure_axis = torch.linspace(1.0e5, 15.0e6, 50, **runtime.tensor_options)
pressure, temperature = torch.meshgrid(
    pressure_axis,
    temperature_axis,
    indexing="ij",
)
feed = torch.tensor([0.65, 0.35], **runtime.tensor_options)
state = ChemicalState(temperature, pressure, feed)

options = GridFlashOptions(
    chunk_size=2048,
    random_allocation_starts=8,
)
equilibrium = flash_grid(model, state, options=options)

if not bool(equilibrium.converged.all()):
    failed = torch.nonzero(
        ~equilibrium.converged.reshape(equilibrium.grid_shape)
    )
    raise RuntimeError(f"non-converged grid cells: {failed.tolist()}")

assert float(equilibrium.fugacity_residual.max()) <= options.fugacity_tolerance
assert (
    float(equilibrium.material_balance_residual.max())
    <= options.material_balance_tolerance
)

identification = identify_grid_phases(model, equilibrium)
pedersen_index = identification.methods.index(
    "pedersen-volume-to-covolume"
)
pedersen_regions = identification.region_codes[pedersen_index]
```

`pedersen_regions` has the original `(pressure, temperature)` grid shape.
Its integer values index `GRID_PHASE_REGION_LABELS`:

| Code | Label | Meaning |
|---:|---|---|
| 0 | `V` | one vapor-like phase |
| 1 | `L` | one liquid-like phase |
| 2 | `LV` | two phases with at least one vapor-like identity |
| 3 | `LL` | two liquid-like phases |
| 4 | `three-phase` | three equilibrium phases; inspect individual identities |
| 5 | `unavailable` | flash did not converge or identification was unavailable |

The exact liquid/vapor identity of every padded phase is in
`phase_identity_codes`, where `-1` is unknown/padded, `0` is liquid, and `1`
is vapor.

## Input and output shapes

Let the common batch shape be `B` and the component count be `C`.

| Quantity | Input/output shape |
|---|---|
| input temperature | `B` |
| input pressure | `B` |
| input composition | `(C,)` or `B + (C,)` |
| flattened temperatures/pressures | `(N,)`, where `N = prod(B)` |
| feeds | `(N, C)` |
| phase fractions | `(N, 3)` |
| phase compositions | `(N, 3, C)` |
| phase counts and residuals | `(N,)` |
| region codes | `(methods,) + B` |
| phase identity codes | `(methods, N, 3)` |

Phase-fraction padding is zero and phase-composition padding is NaN. Always
slice by `phase_counts` before using a state's returned phases.

## Numerical hierarchy

The fast path performs batched tangent-plane stability calculations, flashes
the unstable states together on a known two-phase branch, and tests both
returned phases for further instability. Only failed or child-unstable states
enter the three-phase Gibbs-allocation fallback.

The fallback parameterizes nonnegative phase/component amounts by

\[
n_{pi}=z_i\operatorname{softmax}_p(q_{pi}),
\]

which satisfies component material balance by construction. Adam searches the
difficult states as a tensor batch. Equal-fugacity refinements use Jacobians
from `torch.func.jacrev`. Vanishing phases are removed using
`phase_fraction_tolerance`; composition duplicates are merged only after
refinement.

For a two-dimensional batch, lower-phase-count cells bracketed by
higher-phase-count cells are independently reflashed. This is an audit trigger,
not a topology constraint. A replacement must still reduce Gibbs energy and
pass the same fugacity and material-balance gates.

`flash_grid_oracle` bypasses stability and known-two-phase screening and
strictly refines multiple Gibbs starts for every multicomponent state. Use it
on a small representative batch to verify the fast hierarchy:

```python
from torch_flash import flash_grid_oracle

sample = ChemicalState(
    temperature[::10, ::10],
    pressure[::10, ::10],
    feed,
)
oracle = flash_grid_oracle(model, sample, options=options)
assert bool(oracle.converged.all())
```

The oracle shares the same EoS and low-level equations, so agreement is
verification of the algorithm, not independent model validation.

## Binary three-phase invariants

At fixed temperature, a binary three-phase state is an invariant pressure and
three invariant compositions. The four equal-fugacity equations solve three
composition logits and log pressure:

```python
from torch_flash import (
    ComponentSet,
    soave_redlich_kwong,
    solve_binary_three_phase_invariant,
)

binary_components = ComponentSet(
    ("methane", "carbon_dioxide"),
    torch.tensor([190.6, 304.2], **runtime.tensor_options),
    101325.0 * torch.tensor([45.4, 72.9], **runtime.tensor_options),
    torch.tensor([0.008, 0.228], **runtime.tensor_options),
    torch.tensor([0.01604, 0.04401], **runtime.tensor_options),
    torch.tensor([9.93e-5, 9.40e-5], **runtime.tensor_options),
)
binary_kij = torch.tensor(
    [[0.0, 0.12], [0.12, 0.0]],
    **runtime.tensor_options,
)
binary_model = soave_redlich_kwong(binary_components, kij=binary_kij)

invariant = solve_binary_three_phase_invariant(
    binary_model,
    temperature=torch.tensor(180.0, **runtime.tensor_options),
    initial_pressure=torch.tensor(2.7e6, **runtime.tensor_options),
    initial_phase_compositions=torch.tensor(
        [
            [0.20, 0.80],
            [0.78, 0.22],
            [0.96, 0.04],
        ],
        **runtime.tensor_options,
    ),
)
if not invariant.converged:
    raise RuntimeError(
        f"binary invariant residual={float(invariant.residual_norm):.3e}"
    )
```

Pass only an invariant produced for the same model and parameter set:

```python
methane_fraction = torch.linspace(0.25, 0.95, 50, **runtime.tensor_options)
binary_state = ChemicalState(
    torch.full_like(methane_fraction, 180.0),
    invariant.pressure.expand_as(methane_fraction),
    torch.stack((methane_fraction, 1.0 - methane_fraction), dim=-1),
)
equilibrium = flash_grid(
    binary_model,
    binary_state,
    options=options,
    binary_invariants=(invariant,),
)
```

The solver is local. Initial rows must correspond to the requested EoS roots
and lie near the desired branch. A binary invariant does not uniquely fix all
three phase fractions, so the grid API returns a positive centered
lever-rule representative for feeds strictly inside the outer composition
interval.

## Convergence and failure handling

Do not infer success from a plausible phase count or smooth image. A usable
cell requires:

- `converged=True`;
- `fugacity_residual <= options.fugacity_tolerance`;
- `material_balance_residual <= options.material_balance_tolerance`; and
- positive retained phase fractions and normalized compositions.

`gibbs_reduction` is dimensionless `G/(R*T)` relative to the homogeneous feed.
A split replaces a converged state only if its reduction exceeds
`gibbs_reduction_tolerance`. Non-converged cells remain explicit and map to
`unavailable`; they are never silently colored as a physical phase.

Near critical points, phase compositions coalesce and phase disappearance is
ill-conditioned. Tightening a merge tolerance is not equivalent to resolving
a critical endpoint. Compare branches, residuals, and float64 results rather
than interpreting a single near-critical pixel in isolation.

## Autodiff boundary

PyTorch autodiff supplies Gibbs gradients, equal-fugacity Newton Jacobians,
the two response-derivative phase-identification criteria, and the
Venkatarathnam-Oellrich pressure-derivative parameter. Forward-only stability
screening avoids retaining a graph.

The complete grid result is intentionally not an end-to-end differentiable
mapping: phase-count selection, topology auditing, phase merging, and padded
assembly are discrete. `identify_grid_phases` also detaches its grid of scalar
criterion values to avoid retaining one higher-order graph per cell. Use
`identify_phase` directly on a selected equilibrium composition when a
criterion gradient with respect to temperature or a trainable model parameter
is needed.

## Performance controls

Independent states are already tensor-batched. `chunk_size` limits peak
temporary size; it does not create thermodynamic coupling. The sparse scalar
fallback can use `fallback_workers`, but Python threads can oversubscribe
PyTorch's intra-operation pool. Benchmark one thread first for the small dense
linear systems typical of cubic-EoS flashes.

Record `elapsed_seconds`, `batched_search_seconds`, `refinement_seconds`,
`GridPhaseIdentification.method_elapsed_seconds`, dtype, device, PyTorch
thread count, grid shape, and all residual maxima. The method timings follow
the exact order in `GridPhaseIdentification.methods`. Changing the number of
Gibbs starts or numerical tolerances changes the scientific workload and must
be reported with timing.

`pip_autodiff_chunk_size` bounds the number of independent equilibrium phases
passed to one nested forward-mode JVP evaluation of the
Venkatarathnam-Oellrich parameter. It is a memory/performance control only:
states remain independent and the equations are unchanged.

`response_autodiff_chunk_size` applies the same memory/performance separation
to either response-derivative criterion. Pedersen \(V/b\) evaluates all active
phase roots in one leading batch. Each requested method remains a complete,
independent pass: timings do not assume that another identification method was
evaluated or that derivative work was shared between methods.

Configure process-wide dtype, device, and threads once before model
construction. `flash_grid` never changes global PyTorch runtime settings.

## Physical phase-identification methods

`identify_grid_phases` defaults to the five criteria compared by Bennett and
Schmidt plus the Venkatarathnam-Oellrich phase-identification parameter:

- Li volume-weighted pseudo-critical temperature;
- Pedersen volume-to-covolume ratio;
- Perschke negative flash;
- temperature derivative of isothermal compressibility; and
- temperature derivative of thermal expansion; and
- the dimensionless Venkatarathnam-Oellrich pressure-derivative parameter.

Each criterion is evaluated at each returned equilibrium composition. The
native value, threshold, and ambiguity flag remain available in
`GridPhaseIdentification`. The detailed equations and scalar differentiability
behavior are documented in the
[phase properties API](api/properties.md) and the Bennett and Schmidt entry in
[Scientific references](references.md).
