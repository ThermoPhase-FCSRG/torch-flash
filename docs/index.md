<section class="tp-hero">
  <div class="tp-hero__copy">
    <span class="tp-kicker">Scientific Python · Native PyTorch</span>
    <h1>Differentiable thermodynamics, from state properties to phase equilibrium</h1>
    <p class="tp-hero__lead">
      <code>torch-flash</code> provides typed, differentiable thermodynamic
      models and phase-equilibrium solvers for research, optimization, and
      coupled simulation.
    </p>
    <p class="tp-hero__actions">
      <a href="getting-started/" class="md-button md-button--primary">Start a calculation</a>
      <a href="validation/" class="md-button">Explore validation</a>
    </p>
  </div>
  <div class="tp-hero__identity">
    <div class="tp-logo-plate">
      <a href="./" class="tp-logo-plate__product">
        <img
          src="assets/branding/torch-flash-logo.svg"
          alt="torch-flash — Differentiable Thermodynamics powered by PyTorch"
        />
      </a>
      <div class="tp-affiliation">
        <span>Research developed by and at</span>
        <div class="tp-affiliation__logos tp-affiliation__logos--three">
          <a
            href="https://github.com/ThermoPhase-FCSRG"
            class="tp-affiliation__logo--thermophase"
          >
            <img
              src="assets/branding/thermophase-horizontal.png"
              alt="ThermoPhase — Fluid and Complex Systems Research Group"
            />
          </a>
          <a href="https://www.gov.br/lncc/pt-br">
            <img
              src="assets/branding/lncc.svg"
              alt="Laboratório Nacional de Computação Científica — LNCC"
            />
          </a>
          <a href="https://www.udesc.br/">
            <img
              src="assets/branding/udesc-horizontal.jpg"
              alt="Universidade do Estado de Santa Catarina — UDESC"
            />
          </a>
        </div>
      </div>
    </div>
  </div>
</section>

<div class="tp-feature-grid">
  <div class="tp-feature">
    <strong>Native PyTorch</strong>
    <span>Float64 reference calculations with device, dtype, batching, and automatic differentiation kept explicit.</span>
  </div>
  <div class="tp-feature">
    <strong>Thermodynamic scope</strong>
    <span>Cubic, association, activity-coefficient, and multiparameter models behind a consistent typed interface.</span>
  </div>
  <div class="tp-feature">
    <strong>Auditable evidence</strong>
    <span>Verification, experimental validation, numerical diagnostics, provenance, and limitations documented together.</span>
  </div>
</div>

## Installation and package extras

Install the default package capability from PyPI:

```bash
python -m pip install torch-flash
```

Three optional pip extras expose package-level integrations:

```bash
python -m pip install "torch-flash[groups]"
python -m pip install "torch-flash[intel]"
python -m pip install "torch-flash[gpu]"
```

The `intel` and `gpu` extras are currently active on Linux and Windows. The GPU
extra also requires a compatible CUDA runtime and device.

The normal installation is the `default` capability; it is not named
`torch-flash[default]`. Development, testing, documentation, notebooks,
external comparisons, and benchmarks use dedicated Pixi environments rather
than pip extras. Maintainers should follow the
[dependency and release metadata workflow](contributing.md#dependency-and-release-metadata).

## Scientific software with explicit thermodynamic state

`torch-flash` provides differentiable thermodynamic state models and
phase-equilibrium calculations on top of PyTorch.

The central design rule is that homogeneous-state properties do not require an
equilibrium solve. A supplied `(T, P, x)` state can be evaluated directly for
compressibility, fugacity coefficients, fugacities, dimensionless log
fugacities, chemical potentials, reduced chemical potentials, molar Helmholtz
and Gibbs energies, reduced free energies, residual caloric properties, and
PyTorch derivatives. Flash and saturation solvers are separate consumers of
the same model interface.

```python
import torch
from torch_flash import ChemicalState, component_set, configure, peng_robinson_1978
from torch_flash import (
    log_fugacities_tv,
    phase_properties,
    poling_ideal_gas,
    state_derivatives,
    thermal_properties,
)

runtime = configure(device="cpu", dtype=torch.float64)
model = peng_robinson_1978(component_set(("methane", "n_butane")))
state = ChemicalState(
    runtime.tensor(300.0),
    runtime.tensor(5.0e6),
    runtime.tensor([0.7, 0.3]),
)

properties = phase_properties(model, state)
print(properties.fugacities, properties.log_fugacities)
print(properties.chemical_potentials, properties.reduced_chemical_potentials)
print(properties.molar_helmholtz_energy, properties.molar_gibbs_energy)
print(properties.reduced_helmholtz_energy, properties.reduced_gibbs_energy)
derivatives = state_derivatives(model, state)
print(derivatives.dfugacity_dpressure)
print(derivatives.dlog_fugacity_dtemperature)
print(derivatives.dlog_fugacity_coefficient_dmoles)
print(derivatives.dmolar_volume_dpressure)
print(derivatives.dchemical_potential_dindependent_composition)
print(
    log_fugacities_tv(
        model,
        state.temperature,
        properties.molar_volume,
        state.composition,
    )
)
thermal = thermal_properties(
    model,
    state,
    poling_ideal_gas(["methane", "n_butane"]),
)
```

Version 0.1 is an alpha research release. Read [model scope](model-scope.md)
before selecting a model and [validation](validation.md) before relying on it
outside the tested range.
The [Getting Started guide](getting-started.md) gives complete runnable
examples for homogeneous properties, automatic derivatives, two-phase flash,
and fixed-phase-count multiphase flash.
The [Theoretical Background](theory/index.md) connects chemical equilibrium,
flash algorithms, fugacity and activity models, property derivatives, and
heavy-end characterization to the corresponding `torch-flash` APIs.
The [parameter database guide](parameters.md) documents the versioned YAML
schemas, canonical component names, SI-unit validation, and custom-parameter
APIs.
The [runtime configuration guide](runtime.md) documents construction-time
device/dtype policy, CPU threading, GPU selection, and deterministic execution.
The [scientific reference and data-provenance index](references.md) identifies
primary equation, parameter, experimental-data, and software-baseline sources
separately.
The [high-performance setup and benchmark guide](performance.md) documents
hardware-aware installation, native batching, `torch.compile`, CPU-thread
selection, device-resident GPU execution, precision limits, and benchmark
conditions for ThermoPack, teqp, and NeqSim comparisons.

Detailed equation checks, worked-example reproductions, experimental
comparisons, and fitted-model studies are catalogued in
[verification and validation evidence](validation.md). That page distinguishes
implementation verification from validation against independent measurements
and records the applicable model, data, parameter, and temperature ranges.
