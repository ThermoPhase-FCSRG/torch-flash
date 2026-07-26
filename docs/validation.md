# Validation reports

These reports show the saved visual results of the `torch-flash` validation
notebooks. Experimental markers, model curves, parity plots, and residual
plots are retained so the tested systems and model behavior remain visible.
The underlying research CSVs, numerical tables, and local-only notebook pairs
are not published.

The reports distinguish three kinds of evidence:

- **validation** compares a selected model and parameter set with experimental
  observations;
- **verification** checks equations against a published calculation or
  independently generated software reference; and
- **application** demonstrates a computed property, sensitivity, or model
  surface without claiming experimental agreement.

## Available reports

- [GERG-2008 and EOS-CG](validation-reports/05-gerg-eoscg.md) — measured
  compressibility factors and the complete EOS-CG Table 8 verification set.
- [Native EOS-CG-2021](validation-reports/08-eoscg2021.md) — dense CO2/H2
  numerical verification plus experimental MDEA density and speed of sound.
- [Native GERG-2008 for H2/CH4](validation-reports/09-gerg2008-hydrogen.md) —
  full density and isobaric-heat-capacity verification over the 1,010-state
  H2ThermoBank reference.
- [Autodifferentiated thermal properties](validation-reports/15-thermal-properties.md)
  — CO2 and propane Joule-Thomson validation and homogeneous-state property
  profiles.
- [Huron-Vidal mixing](validation-reports/22-huron-vidal.md) — phase diagrams,
  parity, and complete-temperature transfer for two alcohol/hydrocarbon
  binaries.
- [Cubic volume translation](validation-reports/24-volume-translation.md) —
  methane/n-decane density response for SRK and PR78 translations.
- [Cross-co-volume interaction](validation-reports/25-covolume-interaction.md)
  — before/after density fitting with a disjoint validation isotherm.
- [Original UNIFAC](validation-reports/26-unifac.md) — predictive
  alcohol/hydrocarbon phase diagrams and activity-coefficient behavior.
- [Predictive Peng-Robinson 1978](validation-reports/28-ppr78.md) —
  calibration-domain hydrocarbon VLE comparisons.
- [H2-tailored GERG (2021)](validation-reports/29-hydrogen-tailored-gerg.md) —
  Table 12 verification and reproductions of Figures 4, 7, and 16, including
  N2/H2 and CO2/H2 phase-equilibrium results.
- [Hydrogen-water systems across applicable models](validation-reports/32-hydrogen-water-all-models.md)
  — H2-H2O, H2-CO-H2O, and H2-N2-H2O composition parity and water-content
  curves, including the global 40-group E-PPR78 revision.
- [E-PPR78 for CCS mixtures](validation-reports/33-eppr78-ccs-figure2.md) —
  all eight panels of Xu et al. Figure 2, including N2-CH4 phase behavior and
  mixing enthalpy plus N2-CO phase behavior.

## Publication and reproduction boundary

The figures are selected PNG outputs copied directly from top-to-bottom
executed notebooks. A marker visibly represents an observation coordinate,
but the source row, machine-readable value, CSV, and notebook output table are
not included in the public documentation. This figure-only decision does not
change the source dataset's license or establish a general right to
redistribute it; see [licensing and data provenance](licensing.md).

The local-only notebook pairs can be synchronized, executed, and republished
in a checkout containing the lawful research inputs:

```bash
pixi run -e benchmarks notebooks-sync
pixi run -e benchmarks notebooks-run
pixi run -e notebooks validation-figures
pixi run -e default check-data-rights
```

The [figure manifest](assets/validation/manifest.yaml) records the exact
executed-notebook checksum, source cell, evidence class, and data dependencies
for every published plot.
