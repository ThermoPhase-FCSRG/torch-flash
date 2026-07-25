# Scientific notebooks

The notebooks are grouped by the scientific question they answer. Every
executed `.ipynb` artifact is paired with a reviewable Jupytext
`py:percent` source of the same name.

Studies that require research inputs without a verified redistribution basis
keep **both** paired files under `local-only/<category>/`. That subtree is
excluded from Git, while selected plot outputs can be published as documented
validation reports. Their local inputs belong under the likewise ignored
`tests/data/not-cleared/` directory and must be obtained or generated from a
lawfully available source; they are not repository fixtures. Tests that need
those files skip when the directory is absent, while executing the local-only
notebooks requires the corresponding files to be present. The regular
category folders contain only distributable notebook pairs.

| Directory | Scope |
|---|---|
| `equilibrium/` | Direct property evaluation and two-/three-phase equilibrium calculations |
| `verification/` | Analytical identities, published worked examples, and independent equation/reference-software reproduction: does the implementation solve the stated equations correctly? |
| `validation/` | Comparisons against primary experimental measurements: does the selected physical model represent real systems within quantified error? |
| `solubility/` | Experimental phase-composition, density, and mutual-solubility studies |
| `fitting/` | Autodifferentiable parameter-estimation studies with held-out data |
| `characterization/` | Heavy-end splitting, pseudo-components, and lumping |
| `performance/` | External-backend comparisons, batching, compilation, threading, and GPU studies |
| `local-only/<category>/` | Paired studies whose research inputs or executed tables are excluded from public distribution |

The numeric filename prefix preserves the historical reading order across
directories. In particular:

- `equilibrium/01_differentiable_pr_flash` demonstrates that requested EoS
  roots and physical phase identities are separate, tabulates the Pedersen
  \(V/b\) diagnostic, and renders a colored liquid/vapor/two-phase map with
  stability-tested flash checks.
- `local-only/fitting/19_co2_binary_parameter_fitting` retains the N2-CO2 case and fits
  PR78 and a narrowly modified GERG-2008 form.
- `local-only/fitting/20_co2_water_parameter_fitting` independently fits the more
  CCS-relevant CO2-H2O case using 275 pure-water liquid observations. It keeps
  mole fractions, vapor volume fractions, and Utsira-brine molalities
  separate, and plots both Hou mole-fraction branches explicitly.
- `local-only/solubility/21_huron_vidal_n_butane_water` fits SRK/HV-NRTL to 80
  n-butane/water states over nine isotherms and plots every P-x-y table,
  including complete-temperature holdouts.
- `local-only/validation/22_huron_vidal_alcohol_hydrocarbon` fits and validates
  ethanol/n-heptane and methanol/benzene on 248 retained experimental states
  over 12 isotherms. It also documents and excludes 23 databank rows whose
  cited paper names a different binary system.
- `verification/23_synthetic_co2_pr78_derivatives` reproduces the external
  ThermoPack CO2/N2 PR78 study with exact source parameters, translated
  TP/TV fugacity consistency, all scalar derivatives, a closed retrograde
  envelope through the cricondentherm, an accelerated-versus-legacy timing
  and coordinate audit, and an executed 100 by 100 derivative grid.
- `local-only/validation/24_cubic_volume_translation_pedersen_whitson` checks the
  Pedersen and Whitson parameter equations, then validates translated SRK and
  PR78 densities against 86 methane/n-decane measurements. Its theoretical
  fugacity and VLE identities support the implementation audit, but the
  experimental density comparison makes validation the notebook's primary
  evidence class.
- `local-only/validation/25_cubic_covolume_interaction` verifies the PR78 `lij`
  convention against frozen ThermoPack volume/fugacity results and independent
  Helmholtz derivatives. It then fits one methane/n-decane `l12` on 30
  experimental density states and plots before/after curves, parity,
  residuals, and parameter sensitivity on 33 untouched states.
- `local-only/validation/26_unifac_activity_validation` verifies the DDBST P05.22a
  calculation, three independent `thermo` 0.6.0 states, and the extensive
  excess-Gibbs autodiff identity,
  then predicts all 248 retained ethanol/n-heptane and methanol/benzene
  states over 12 isotherms. Every experimental \(P\)-\(x\)-\(y\) diagram,
  parity plot, and residual plot is executed; the 29 observations above the
  original model's approximately 425 K recommendation are visibly separated.
- `local-only/verification/27_ppr78_group_contribution` checks the published PPR78
  Appendix A propane/n-butane calculation and all 15 printed Figure 3
  interaction values, then exercises temperature, structural, and parameter
  derivatives.
- `local-only/validation/28_ppr78_hydrocarbon_vle` compares conventional zero-BIP PR78
  and PPR78 against all 103 retained methane/ethane and methane/n-decane
  measurements over six isotherms, with full P-x-y, parity, residual, and
  temperature-dependent-interaction plots. These systems contributed to the
  original parameter fit, so this is calibration-domain validation rather
  than an independent blind test.
- `verification/29_bennett_phase_identification_methods` flashes every grid
  feed and applies all five homogeneous-state criteria to every returned
  equilibrium phase for the five fluid cases reported by Bennett and Schmidt.
  It records the directly available inputs and explicit pseudo-component
  assumptions, enforces fugacity and material-balance residual gates, plots
  every method on matched 33 by 33 grids, reports batched-search/refinement
  timings, and verifies a higher-order gradient through a trainable binary
  interaction. The maps compare diagnostic phase labels; phase count still
  comes from the preceding equilibrium calculation.

The fitted GERG variants in these two studies are not parameter sets published
as GERG-2008. They are deliberately local sensitivity demonstrations. The
CO2-H2O GERG-form fit remains trained on phase-complete Hou data, while the
expanded multi-temperature liquid datasets provide saturation-residual
evaluation and expose its transfer limitations.

Install and execute the complete suite, including the optional external
backends, with:

```bash
pixi install -e benchmarks
pixi run -e benchmarks notebooks-run
```

Synchronize all nested `.ipynb`/`.py` pairs after editing either
representation with:

```bash
pixi run -e benchmarks notebooks-sync
```

Publish the selected plot outputs from the freshly executed validation
notebooks with:

```bash
pixi run -e notebooks validation-figures
```

This task writes the figure-only validation reports and their provenance
manifest under `docs/assets/validation/`. It does not copy notebook tables,
research CSVs, or either member of a `local-only` notebook pair into the
documentation.
