# torch-flash

[![Tests](https://github.com/ThermoPhase-FCSRG/torch-flash/actions/workflows/tests.yml/badge.svg)](https://github.com/ThermoPhase-FCSRG/torch-flash/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/ThermoPhase-FCSRG/torch-flash/branch/main/graph/badge.svg)](https://codecov.io/gh/ThermoPhase-FCSRG/torch-flash)
[![Supported OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-blue)](https://github.com/ThermoPhase-FCSRG/torch-flash/actions/workflows/tests.yml)
[![Docs](https://github.com/ThermoPhase-FCSRG/torch-flash/actions/workflows/docs.yml/badge.svg)](https://thermophase-fcsrg.github.io/torch-flash/)
[![PyPI](https://img.shields.io/pypi/v/torch-flash.svg)](https://pypi.org/project/torch-flash/)
[![DOI](https://zenodo.org/badge/1309653496.svg)](https://doi.org/10.5281/zenodo.21541547)
[![Python](https://img.shields.io/pypi/pyversions/torch-flash.svg)](https://pypi.org/project/torch-flash/)
[![License: LGPL-2.1](https://img.shields.io/badge/license-LGPL--2.1-blue.svg)](LICENSE)

`torch-flash` is a typed, differentiable thermodynamics and phase-equilibrium
library built on PyTorch. Homogeneous-state properties are independent of
equilibrium solvers: a supplied composition, temperature, and pressure can be
evaluated directly and differentiated for optimization, machine learning, or
coupled simulation.

The current model coverage, assumptions, and limitations are documented in [Model scope](docs/model-scope.md).

## Installation

```bash
python -m pip install torch-flash
```

Optional package capabilities are available through the `groups`, `intel`,
and `gpu` extras:

```bash
python -m pip install "torch-flash[groups]"
python -m pip install "torch-flash[intel]"
python -m pip install "torch-flash[gpu]"
```

The `intel` and `gpu` dependencies are currently published for Linux and
Windows. GPU use additionally requires a compatible CUDA runtime and device.

Development, documentation, notebook, external-comparison, and benchmark
dependencies remain in their dedicated [Pixi](https://pixi.sh) environments:

```bash
pixi install -e test
pixi run -e test test-cov
```

For maintainers, `pixi.toml` is the dependency source of truth and the package
version is synchronized across release metadata through Pixi tasks:

```bash
pixi run sync-deps
pixi run sync-deps-check
pixi run bump-version 0.1.3 --dry-run
pixi run bump-version 0.1.3
```

See the [contribution guide](CONTRIBUTING.md) for the dependency-export
boundary, synchronized version files, and release checks.

## Quick start

```python
import torch

from torch_flash import (
    ChemicalState,
    component_set,
    configure,
    peng_robinson_1978,
    phase_properties,
    state_derivatives,
    two_phase_flash,
)

runtime = configure(device="cpu", dtype=torch.float64)
model = peng_robinson_1978(component_set(("methane", "n_butane")))
state = ChemicalState(
    temperature=runtime.tensor(270.0),       # K
    pressure=runtime.tensor(3.0e6),          # Pa
    composition=runtime.tensor([0.5, 0.5]),  # mole fractions
)

# Evaluate the supplied homogeneous state without solving an equilibrium.
properties = phase_properties(model, state, phase="stable")
print(properties.compressibility_factor)
print(properties.fugacities, properties.log_fugacities)
print(properties.chemical_potentials)
print(properties.molar_helmholtz_energy, properties.molar_gibbs_energy)
print(properties.reduced_helmholtz_energy, properties.reduced_gibbs_energy)

# Derivatives are evaluated with PyTorch automatic differentiation.
derivatives = state_derivatives(model, state)
print(derivatives.dfugacity_dpressure)
print(derivatives.dchemical_potential_dtemperature)

# Stability-tested two-phase TP flash.
equilibrium = two_phase_flash(model, state)
print(equilibrium.phase_fractions)
print(equilibrium.phase_kinds, equilibrium.phase_regime)
for phase in equilibrium.phases:
    print(phase.composition)
```

Thermodynamic inputs and outputs use SI units. Float64 is strongly recommended
for phase-equilibrium calculations. Device, dtype, CPU-thread, and
deterministic-execution policies are set before model construction; see
[Runtime configuration](docs/runtime.md).

## Capabilities

- SRK, PR76, and PR78 cubic equations of state, with attractive and co-volume
  binary interactions, PPR78 group-contribution BIPs, and Pedersen or
  Péneloux/Whitson volume translation
  ([Soave, 1972](https://doi.org/10.1016/0009-2509%2872%2980096-4);
  [Peng and Robinson, 1976](https://doi.org/10.1021/i160057a011);
  [Jaubert and Mutelet, 2004](https://doi.org/10.1016/j.fluid.2004.06.059)).
- NRTL, Wilson, original VLE-UNIFAC, and Huron–Vidal mixing rules
  ([Renon and Prausnitz, 1968](https://doi.org/10.1002/aic.690140124);
  [Huron and Vidal, 1979](https://doi.org/10.1016/0378-3812%2879%2980001-1);
  [Hansen et al., 1991](https://doi.org/10.1021/ie00058a017)).
- SRK-CPA association models with configurable site schemes, cross-association
  rules, hydrocarbon/water parameters, and heavy-cut adapters
  ([Kontogeorgis et al., 1996](https://doi.org/10.1021/ie9600203)).
- Native GERG-2008 and EOS-CG-2021 multiparameter Helmholtz mixture models
  ([Kunz and Wagner, 2012](https://doi.org/10.1021/je300655b);
  [Neumann et al., 2023](https://doi.org/10.1007/s10765-023-03263-6)).
- Fugacity, chemical potential, compressibility, molar volume, Helmholtz and
  Gibbs energies, caloric and response properties, and PyTorch derivatives at
  a specified state.
- Tangent-plane stability, two-phase TP flash, fixed-phase-count multiphase
  flash, phase identification, bubble/dew calculations, phase-envelope
  continuation, and a binary critical-point solver
  ([Michelsen, 1982](https://doi.org/10.1016/0378-3812%2882%2985001-2)).
- Model-neutral pseudo-component splitting, characterization, lumping, and
  model-specific property adapters.
- Differentiable LBC and corresponding-states viscosity models.
- Trainable interaction parameters and Helmholtz coefficients for
  reparameterization against user data.

The package does not currently provide automatic global phase-count discovery,
reactive equilibrium, or a general multicomponent critical-locus arclength
solver. Fixed-phase-count multiphase calculations require physically
meaningful initialization. Detailed model-specific boundaries are stated in
[Model scope](docs/model-scope.md).

## Parameter databases

Component properties and bundled model parameterizations are stored in
versioned YAML with canonical component names, declared SI units, model
identity, and source metadata. Models may use packaged parameter sets, custom
YAML files, typed parameter objects, or user-supplied tensors:

```python
from torch_flash import (
    available_parameter_sets,
    component_set,
    cubic_eos,
    cubic_interaction_parameters,
    multiparameter_eos,
)

components = component_set(("methane", "n_decane"))
interactions = cubic_interaction_parameters(
    components,
    "binary-interaction.segovia-2017-methane-n-decane",
)
pr78 = cubic_eos(
    components,
    "cubic.pr-1978",
    kij=interactions.kij,
    lij=interactions.lij,
)
gerg = multiparameter_eos("gerg2008", ("CO2", "CH4"))
print(available_parameter_sets(model_kind="cubic"))
```

Parsed database documents are cached; model instances and their tensors are
not shared. Custom schemas, canonical naming, units, cache controls, and
trainable-parameter conventions are described in
[Component and model parameter databases](docs/parameters.md).

Bundled parameters, repository-only scientific data, and independently
generated software baselines have distinct provenance and redistribution
conditions. See [Licensing and scientific-data provenance](docs/licensing.md).
Research studies and test data are not included in the PyPI wheel or source
distribution.

## Documentation

- [Rendered documentation](https://thermophase-fcsrg.github.io/torch-flash/)
- [Getting started](docs/getting-started.md)
- [Theoretical background](docs/theory/index.md)
- [Model scope and numerical assumptions](docs/model-scope.md)
- [Python API](docs/api.md)
- [Parameter and component databases](docs/parameters.md)
- [Runtime configuration](docs/runtime.md)
- [High-performance PyTorch setup and benchmarks](docs/performance.md)
- [Verification and validation evidence](docs/validation.md)
- [Scientific references and data provenance](docs/references.md)
- [Licensing and redistribution policy](docs/licensing.md)
- [Contributing](docs/contributing.md)

## Generative-AI disclosure

Following the transparency and responsibility principles in the
[CNPq Policy for Integrity in Scientific Activity](https://www.gov.br/cnpq/pt-br/assuntos/noticias/cnpq-em-acao/cnpq-publica-portaria-que-institui-politica-de-integridade-na-atividade-cientifica),
generative-AI tools—OpenAI Codex and GPT models—assisted code implementation,
refactoring, and documentation. Human contributors remain responsible for
scientific choices, source verification, tests, review, and released
artifacts. Contributions produced with AI assistance are held to the same
review and reproducibility requirements as all other work.

## License

Source code is distributed under the
[GNU Lesser General Public License v2.1](LICENSE). Third-party notices and
data-specific terms are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Institutional Support

`torch-flash` receives institutional support from the
[Laboratório Nacional de Computação Científica (LNCC)](https://www.gov.br/lncc/pt-br),
a research unit of the Ministério da Ciência, Tecnologia e Inovação (MCTI),
Brazil, and is partially developed at the
[Universidade do Estado de Santa Catarina (UDESC)](https://www.udesc.br/).
The project is developed by the
[ThermoPhase — Fluid and Complex Systems Research Group](https://github.com/ThermoPhase-FCSRG).

<div align="center">
  <table>
    <tr>
      <td align="center" bgcolor="#ffffff">
        <a href="https://github.com/ThermoPhase-FCSRG">
          <img
            src="resources/logo/TP%20LOGO-%20Horizontal,%20colorful,%20transparent%20background-%20extended%20version.png"
            alt="ThermoPhase — Fluid and Complex Systems Research Group"
            width="680"
          />
        </a>
        <br />
        <a href="https://www.gov.br/lncc/pt-br">
          <img
            src="resources/logo/lncc.svg"
            alt="Laboratório Nacional de Computação Científica — LNCC"
            width="260"
          />
        </a>
        <a href="https://www.udesc.br/">
          <img
            src="resources/logo/udesc-horizontal.jpg"
            alt="Universidade do Estado de Santa Catarina — UDESC"
            width="260"
          />
        </a>
      </td>
    </tr>
  </table>
</div>
