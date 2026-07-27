# Validation reports

These reports show validation results produced with `torch-flash`.
Experimental markers, model curves, parity plots, residual plots, and
aggregate metrics are included where appropriate so the tested systems and
model behavior remain visible.

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
- [CO2 pre-salt phase behavior across applicable models](validation-reports/34-co2-presalt-all-models.md)
  — aggregate unfitted and fitted results for all 157 binary, ternary, and
  quaternary transition observations from Simoncelli et al. Tables 7-9.
- [Trust-Region phase envelopes and phase identification](validation-reports/35-nichita-trust-region-phase-envelopes.md)
  — verification of Petitfrere-Nichita Figures 3 and 4 plus a comparison of
  all six physical phase-identification criteria.

## Report traceability

Each report records the model identity, evidence class, conditions, units,
solver acceptance criteria, quantitative metrics, and scientific limitations
needed to interpret its `torch-flash` results. Published validation figures
also have an entry in the
[figure manifest](assets/validation/manifest.yaml) with their evidence class,
source cell, inputs, and checksum.
