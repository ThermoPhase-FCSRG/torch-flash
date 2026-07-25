# API Reference

The API reference is generated directly from the public signatures,
annotations, and NumPy-style docstrings in `src/torch_flash`. Use the
[Getting Started](getting-started.md) and
[Theoretical Background](theory/index.md) sections for task-oriented examples
and derivations; use these pages for the exact callable contract.

Most commonly used names are re-exported from `torch_flash`:

```python
from torch_flash import ChemicalState, phase_properties, two_phase_flash
```

The domain modules below remain useful for discovering related types and
advanced constructors.

## API domains

| Domain | Contents |
|---|---|
| [Runtime, components, and databases](api/runtime-components.md) | Constants, device/dtype policy, canonical components, parameter documents, exceptions, and cache control |
| [Equations of state](api/equations-of-state.md) | Cubic, CPA, GERG-2008, EOS-CG-2021, and volume-translation APIs |
| [Activity models and mixing rules](api/activity-mixing.md) | NRTL, Wilson, original UNIFAC, Huron-Vidal, and cubic mixing protocols |
| [Phase properties and derivatives](api/properties.md) | Homogeneous-state properties, caloric properties, standard states, and phase identification |
| [Flash and equilibrium](api/flash-equilibrium.md) | Initial estimates, stability, two-phase and multiphase flash, material balance, saturation, envelopes, and binary loci |
| [Characterization, parameters, and fitting](api/characterization-parameters.md) | Heavy-end splitting/lumping, cubic property adapters, interactions, PPR78 groups, and calibration helpers |
| [Transport properties](api/transport.md) | Pedersen and Lohrenz-Bray-Clark viscosity calculations |
| [Optional backends and numerical solvers](api/backends-solvers.md) | Capability reporting, external homogeneous-state adapters, and damped Newton |
| [State and result types](api/results.md) | Chemical states, phase properties, convergence results, and equilibrium-point records |

## Shared conventions

- Public thermodynamic inputs and outputs use SI units.
- Component-axis order follows the associated `ComponentSet`.
- Float64 is the reference precision for equilibrium and derivatives.
- Leading dimensions are batch dimensions only where the individual API
  explicitly documents batching.
- A returned convergence flag and residual are part of the scientific result.
- Physical phase identification is a diagnostic and is separate from root
  selection and equilibrium phase count.

Every model constructor records its parameter identity. Consult
[Parameter databases](parameters.md) for bundled identifiers and
[Model scope](model-scope.md) for supported scientific ranges.
