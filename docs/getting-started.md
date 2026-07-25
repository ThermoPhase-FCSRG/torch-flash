# Getting started

This guide takes a new user from installation to the calculations most often
needed in a thermodynamic workflow:

1. configure PyTorch and construct a model;
2. evaluate a supplied homogeneous state and its derivatives;
3. solve a known two-phase state;
4. solve a fixed three-phase problem; and
5. check convergence, fugacity residuals, and material balance.

All public thermodynamic inputs and outputs use SI units. The examples use
`torch.float64`, the reference precision for phase-equilibrium calculations.

## Install torch-flash

`torch-flash` requires Python 3.11 or newer. Install the normal package from
PyPI:

```bash
python -m pip install torch-flash
```

The optional pip extras are intentionally narrower than the Pixi development
environments:

```bash
# Group-contribution helpers
python -m pip install "torch-flash[groups]"

# Optional sparse linear algebra on Linux or Windows
python -m pip install "torch-flash[intel]"

# Optional CUDA-oriented solver integrations on Linux or Windows
python -m pip install "torch-flash[gpu]"
```

The `gpu` extra does not install a GPU driver or make every calculation faster.
Use a PyTorch build compatible with the local CUDA driver, keep tensors on the
device, and batch enough work to amortize launch and transfer costs. See
[high-performance setup](performance.md) before selecting an accelerator.

For a source checkout, use the locked Pixi environment:

```bash
pixi install -e default
pixi run -e default python -c "import torch_flash; print(torch_flash.__version__)"
```

## Configure the runtime first

Runtime dtype, device, thread counts, and deterministic behavior affect every
model tensor created afterward. Configure them before constructing component
sets or models:

```python
import torch

from torch_flash import configure

runtime = configure(
    device="cpu",
    dtype=torch.float64,
    num_threads=4,
)
temperature = runtime.tensor(300.0)
```

`device="auto"` selects the first available CUDA, XPU, or MPS device that
supports the chosen dtype, then falls back to CPU. `device="gpu"` requires an
accelerator and raises if none supports that dtype. Float64 is not available
on every accelerator; do not silently replace it with float32 near critical
states or coalescing roots.

## The model-state-calculation pattern

Most `torch-flash` calculations follow the same pattern:

```python
import torch

from torch_flash import ChemicalState, component_set, configure, peng_robinson_1978

runtime = configure(device="cpu", dtype=torch.float64)
components = component_set(("methane", "n_butane"))
model = peng_robinson_1978(components)
state = ChemicalState(
    temperature=runtime.tensor(300.0),  # K
    pressure=runtime.tensor(5.0e6),  # Pa
    composition=runtime.tensor([0.70, 0.30]),  # mol/mol
)
```

The order of every composition vector is the order supplied to
`component_set`. Component names are canonicalized at this API boundary.
Named constructors use the versioned parameter databases documented in the
[parameter guide](parameters.md).

## Compute properties and derivatives

`phase_properties` evaluates the supplied homogeneous state. It does **not**
perform a flash. `state_derivatives` then differentiates the same state with
respect to temperature, pressure, composition coordinates, and mole numbers.

```python
--8<-- "docs/examples/properties_and_derivatives.py"
```

The default chemical-potential convention uses a zero ideal-gas standard
chemical potential at 1 bar. It is not a formation-property reference.
Reference-independent residual energies are provided separately.

Composition derivatives require strictly positive mole fractions. For
\(n\) components, the independent-composition Jacobian has \(n-1\) columns,
with the final fraction defined by
\(x_n=1-\sum_{i=1}^{n-1}x_i\). See
[Phase Properties and Their Derivatives](theory/phase-properties-derivatives.md)
for the complete coordinate and unit conventions.

## Solve a two-phase flash

At fixed temperature, pressure, and overall composition, a two-phase TP flash
finds the phase fraction and phase compositions that satisfy material balance
and equality of component fugacities.

```python
--8<-- "docs/examples/two_phase_flash.py"
```

This compact example sets `check_stability=False` because it deliberately
solves a state already known to lie on a two-phase branch. Keep the default
`check_stability=True` when the phase count is unknown. Disabling stability
analysis is not a general shortcut: fugacity equality is necessary, but a
lower-Gibbs-energy phase may still exist.

Do not use a result unless:

- `result.converged` is true;
- `result.residual_norm` is appropriate for the model and dtype;
- all phase fractions and compositions are physical;
- the reconstructed feed matches the specified feed; and
- the selected roots and phase identities make physical sense.

The [Two-Phase Flash](theory/two-phase-flash.md) page derives the
Rachford-Rice balance and documents the solver sequence.

## Solve a fixed multiphase flash

`multiphase_flash` accepts one row of equilibrium ratios for every phase after
the reference phase. Therefore, an initial \(K\) matrix with shape
`(2, n_components)` requests a fixed three-phase solve.

The following example reproduces the model definition and rounded
initialization from Tables 6.5-6.6 of Pedersen, Christensen, and Shaikh. The
source values initialize the calculation; the converged result is checked by
fugacity residual and material balance.

??? example "Complete runnable fixed-three-phase example"

    ```python
    --8<-- "docs/examples/multiphase_flash.py"
    ```

The emitted `ExperimentalModelWarning` is intentional. The current API does
not discover the correct phase count automatically, and a converged
fixed-phase solution is not by itself a global-stability proof. Physical phase
identification is also a diagnostic rather than an equilibrium equation; with
the rounded constants in this example, the current heuristic labels the
converged phases `vapor`, `vapor`, and `liquid`. Inspect densities, roots,
stability, and the application context before assigning physical names.

See [Multiphase Flash](theory/multiphase-flash.md) for the generalized
Rachford-Rice equations, initialization contract, and limitations.

## Differentiate a custom objective

Native model calculations remain in the PyTorch graph. For example, an
optimization objective can depend directly on a fugacity coefficient:

```python
import torch

from torch_flash import component_set, configure, peng_robinson_1978

runtime = configure(device="cpu", dtype=torch.float64)
model = peng_robinson_1978(component_set(("methane", "n_butane")))
temperature = runtime.tensor(300.0, requires_grad=True)
pressure = runtime.tensor(5.0e6)
composition = runtime.tensor([0.70, 0.30])

log_phi = model.log_fugacity_coefficients(
    temperature,
    pressure,
    composition,
    phase="stable",
)
objective = log_phi.square().sum()
objective.backward()
print(temperature.grad)
```

Avoid `.item()`, NumPy conversion, or `detach()` inside a differentiable
calculation. Use them only at an intentional reporting or control-flow
boundary.

## Where to go next

- [Chemical Equilibrium Overview](theory/chemical-equilibrium.md) explains
  equilibrium, fugacity equality, stability, and the tangent-plane distance.
- [Activity Models](theory/activity-models.md) covers NRTL, Wilson, original
  UNIFAC, and Huron-Vidal activity terms.
- [Fugacity Models](theory/fugacity-models.md) connects equation-of-state roots
  to phase fugacity.
- [Characterization and Pseudo-Components](theory/characterization-pseudocomponents.md)
  covers plus-fraction splitting, property correlations, and lumping.
- [High-performance setup](performance.md) covers batching, compilation,
  threading, accelerators, gradients, and trustworthy timing.
- [Model scope](model-scope.md) and [validation evidence](validation.md)
  define which scientific claims are currently supported.
