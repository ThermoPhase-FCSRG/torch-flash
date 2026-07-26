# Frozen scientific baselines

This manifest distinguishes experimental measurements, values printed by a
published thermodynamic model, and independently generated software
baselines. The CSV files are regression inputs; their presence does not turn a
software or model-reproduction comparison into experimental validation.

`segovia_2017_methane_n_decane_density.csv` contains experimental
compressed-liquid densities for methane + n-decane, transcribed from Table 4
of Segovia et al., *Journal of Chemical Thermodynamics* 109 (2017) 113-128,
[doi:10.1016/j.jct.2017.01.022](https://doi.org/10.1016/j.jct.2017.01.022).
The file retains the two slices plotted in that paper's Figure 4: the complete
80 MPa isobar across temperature and the complete 323.15 K isotherm across
pressure for all six reported methane mole fractions. It also retains the six
0.1 MPa pure-n-decane states needed to test the temperature-dependent
Pedersen translation rather than extrapolating from a single datum. Duplicate
intersection points retain a `series` label so either panel can be reproduced
independently.
`U_density_g_cm3` is the expanded density uncertainty (\(k=2\)) stated in the
Table 4 footnote; the paper also reports \(U(x)=3\times10^{-4}\),
\(u(T)=0.02\) K, and \(u(P)=0.08\) MPa.
Notebook 25 estimates one PR78 co-volume interaction from the 30 non-pure
80 MPa states excluding 323.15 K, then reserves all 33 non-pure states on the
323.15 K isotherm for validation. The split and fitted parameter are recorded
in the notebook and in the explicitly local
`binary-interaction.segovia-2017-methane-n-decane` YAML.

## Distribution and rights status

The tracked files in this directory are repository research material and are
excluded from both the PyPI wheel and source distribution. Inputs whose
redistribution status is unresolved live only under
`tests/data/not-cleared/`; that whole subdirectory is ignored by Git as well
as excluded from package distributions. Keeping data out of a package does
not by itself grant permission to publish it in a public Git repository.

The machine-readable [`rights.yaml`](rights.yaml) ledger covers all 35
declared CSV artifacts: 5 have a verified open basis, 12 are project-generated
software outputs, and 18 are marked `not-cleared`. The audit scans recursively,
fails if a CSV is added without an explicit rights decision, and enforces that
locally present `not-cleared` files are confined to the ignored subdirectory.
Tests that require those inputs skip when it is absent.

Executed notebook tables and plots made from a `not-cleared` fixture are not
automatically cleared by transformation or by exclusion from PyPI. The
project's current documentation decision publishes selected rendered plots,
including experimental markers, but not the source CSVs, notebook pairs,
machine-readable observation values, or output tables. Each plot's notebook
checksum, source cell, and data dependency are audited through
`docs/assets/validation/manifest.yaml`. This figure-only boundary is a
repository decision, not a declaration that the underlying dataset is openly
licensed.

The following sources have an explicit reusable-data or open-content basis:

- `nist_thermoml_h2_binary_density.csv`: NIST ThermoML public-domain archive;
- `gerg2008_h2_ch4_reference.csv`: H2ThermoBank Figshare archive,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
- `moortgat_2025_h2_h2o_vle.csv`: PLOS article and supplement,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); and
- `eoscg2021_mdea_density_experimental.csv` and
  `eoscg2021_mdea_speed_of_sound_experimental.csv`: Springer article and
  supplement,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Files identified below as independently generated software baselines are
project-generated numerical outputs, not copies of the external
implementations. For the remaining literature transcriptions, workbook
subsets, and figure digitizations, the primary citation is recorded but an
explicit redistribution license has not been verified. Before publishing a
repository data archive, replace those files with openly licensed data, make
the extraction depend on a lawful user-supplied source, obtain permission, or
remove them. See
[`docs/licensing.md`](../../docs/licensing.md) for the release policy.

In particular, the three normalized Jaubert workbook extracts are not cleared
for a public data release. The article is identified as CC BY-NC-ND 4.0, no
separate permissive workbook license was found, and normalization creates an
adapted dataset. The scripts therefore treat the workbook as a user-supplied
research input; citation alone is not permission to redistribute the outputs.

The file-specific descriptions below identify the modifications: selections,
normalization, transcriptions, unit conversions, and missing-value handling.
The source authors do not endorse `torch-flash`.

## Published experimental and model tables

- `beckmuller_2021_h2_gerg_table12.csv` retains all 16 single-phase
  implementation states from Table 12 for CH4, N2, CO, and CO2 mixed with
  40 mol % H2. The local-only notebook evaluates pressure, isobaric heat
  capacity, speed of sound, enthalpy, entropy, and molar Helmholtz energy
  without fitting an offset.
- `beckmuller_2021_h2_vle_digitized.csv` contains approximate visual
  digitizations of the experimental markers in Figures 7 and 16. Pixel
  calibration used the linear H2-composition axis and logarithmic pressure
  ticks; the CSV records the conservative digitization uncertainty used by
  the notebook. It is not a substitute for the primary experimental tables.
  [doi:10.1063/5.0040533](https://doi.org/10.1063/5.0040533).
- `jaubert_ppr78_hydrocarbon_vle.csv` retains 103 phase-complete experimental
  VLE states used in notebook 28: methane/ethane at 199.93 and 230.00 K
  (38 states), and methane/n-decane at 410.93, 477.59, 510.93, and 563.25 K
  (65 states). Pressures and both coexisting compositions come from four
  primary experimental tables preserved in the Jaubert et al. binary
  databank. `scripts/extract_ppr78_validation_data.py` deterministically
  rebuilds the CSV from an optional user-supplied workbook and guards the
  selected row count; tests and notebooks read the resulting local-only CSV
  from `tests/data/not-cleared/`. Because both systems were among those used
  to estimate the published PPR78 group parameters, agreement is
  calibration-domain validation rather than a blind predictive test.
  [Databank article](https://doi.org/10.1021/acs.iecr.0c01734);
  [Jaubert and Mutelet PPR78 correlation](https://doi.org/10.1016/j.fluid.2004.06.059).
- `jaubert_mutelet_2004_ppr78_kij.csv` records all 15 PPR78 interaction values
  printed in Figure 3 for propane/n-butane, methane/ethane, and
  methane/n-decane. They are rounded outputs of the published model, not
  experimental measurements. The file exists solely to verify Eq. 5 and the
  complete Table 1 parameter transcription.
  [doi:10.1016/j.fluid.2004.06.059](https://doi.org/10.1016/j.fluid.2004.06.059).
- `jaubert_2020_co2_binary_vle.csv` is a curated, normalized subset of the
  binary VLE workbook distributed with Jaubert et al. (2020). It contains 79
  measured states for six CCS-relevant systems: CH4/CO2 (Wei et al., 1995,
  27 points), N2/CO2 (Alsahhaf et al., 1983, 17), O2/CO2 (Fredenslund and
  Sather, 1970, 11), CO/CO2 (Christiansen et al., 1995, 9), CO2/H2O
  (Hou et al., 2013, 6), and CO2/H2S (Chapoy et al., 2013, 9). Component order,
  source sheet, temperature, pressure, both coexisting compositions, source
  key, and DOI are retained. Only complete, isothermal `P, x1, y1` tables are
  included; VLLE and one-sided solubility tables are deliberately excluded.
  The source workbook remains the authoritative full databank and is not
  shipped.
  [Databank article doi:10.1021/acs.iecr.0c01734](https://doi.org/10.1021/acs.iecr.0c01734);
  [Wei et al.](https://doi.org/10.1021/je00020a002);
  [Alsahhaf et al.](https://doi.org/10.1021/i100012a004);
  [Fredenslund and Sather](https://doi.org/10.1021/je60044a024);
  [Hou et al.](https://doi.org/10.1016/j.supflu.2012.11.011);
  [Chapoy et al.](https://doi.org/10.1016/j.fluid.2013.07.050).
- `yucelen_kidnay_1999_n2_co2_vle.csv` transcribes all 21 N2/CO2 states in
  Tables C.1 and C.2 of Yücelen's thesis, including one-sided phase
  compositions rather than silently discarding them. The 240 and 270 K
  isotherms make a two-coefficient temperature-dependent PR78 interaction
  law estimable, although only six non-pure states report both coexisting
  compositions and the resulting identifiability must still be diagnosed.
  Pressure is retained in MPa; the reported composition accuracy is 0.002
  mole fraction. The thesis also reports independently fitted constant PR
  interactions of -0.029 and -0.024 at 240 and 270 K, respectively.
  [Yücelen and Kidnay article doi:10.1021/je980321e](https://doi.org/10.1021/je980321e);
  [Colorado School of Mines thesis](https://repository.mines.edu/entities/publication/8fdfb765-076f-4e23-a49f-c905501731ca).
- `jaubert_2020_hv_bac5_vle.csv` retains 328 phase-complete VLE states from
  three BAC-5 systems selected for Huron-Vidal validation: n-butane/water
  (80 states, 9 isotherms), ethanol/n-heptane (120, 4), and
  methanol/benzene (128, 8). The file preserves both coexisting
  compositions, flags, worksheet, DOI, and full citation. Ten n-butane
  liquid compositions printed as zero remain zero; they are plotted but
  excluded from logarithmic fugacity fitting. Eight reported azeotropes are
  retained. The extraction deliberately rejects 23 nominal
  ethanol/n-heptane rows at 483.15, 508.15, and 523.15 K because worksheet
  `1102_17` assigns them to
  [Seo et al. (2003)](https://doi.org/10.1021/je025604s), a
  2-propanol/n-hexane paper. The exclusion count is guarded in
  `scripts/extract_jaubert_hv_cases.py`; neither component identity nor
  provenance is silently inferred.
  [Databank article](https://doi.org/10.1021/acs.iecr.0c01734);
  [Reamer et al.](https://doi.org/10.1021/ie50507a049);
  [Danneil et al.](https://doi.org/10.1002/cite.330391309);
  [Berro et al.](https://doi.org/10.1016/0378-3812(82)80005-8);
  [Ratcliff and Chao](https://doi.org/10.1002/cjce.5450470208);
  [Toghiani et al.](https://doi.org/10.1021/je00013a018);
  [Strubl et al.](https://doi.org/10.1135/cccc19723522);
  [Scatchard and Wood](https://doi.org/10.1021/ja01214a024);
  [Butcher and Medani](https://doi.org/10.1002/jctb.5010180402).
- `ahmadi_chapoy_2018_co2_water_solubility.csv` transcribes all 29 pure-water
  measurements and their expanded mole-fraction uncertainties from Table 6.
  The five isotherms span 300.95-423.48 K and pressures through 42.077 MPa.
  Temperatures and liquid CO2 mole fractions are retained as printed; pressure
  remains in MPa.
  [doi:10.1016/j.fluid.2018.02.002](https://doi.org/10.1016/j.fluid.2018.02.002).
- `wang_2021_co2_water_solubility_molality.csv` transcribes the complete
  24-pressure by 10-temperature Table 1: 240 liquid-phase CO2 molalities from
  313.15 to 473.15 K and 0.5 to 200 MPa. The wide layout mirrors the printed
  table. The fitting notebook converts molality \(m\) to binary liquid mole
  fraction with \(x_{\mathrm{CO2}}=m/(m+1/M_{\mathrm{H2O}})\).
  `wang_2021_water_vapor_volume_fraction.csv` separately transcribes all 56
  entries in Table 2. Those values are measured H2O **volume fractions**, not
  equilibrium mole fractions; the paper itself warns that the two are not
  interchangeable at high pressure. They are therefore plotted in their
  reported units and are not used as vapor-composition fitting targets.
  [doi:10.1016/j.apgeochem.2021.105005](https://doi.org/10.1016/j.apgeochem.2021.105005).
- `portier_rochelle_2005_utsira_brine_solubility.csv` transcribes all 35
  replicate measurements in Table 2 for synthetic Utsira porewater. This is a
  mixed-electrolyte solution, not the binary CO2-H2O system. It is retained as
  CCS context and excluded from salt-free PR78/GERG fitting; treating it as
  pure water would confound binary interaction parameters with salting-out.
  [doi:10.1016/j.chemgeo.2004.12.007](https://doi.org/10.1016/j.chemgeo.2004.12.007).
- `nist_thermoml_h2_binary_density.csv` contains 99 primary mass-density
  measurements for three UHS-relevant binaries from NIST's ThermoML archive:
  H2/CH4 (40 selected from 391 original states), H2/N2 (40 selected from 399),
  and H2/CO2 (all 19 states). For H2/CH4 and H2/N2, the retained rows are the
  minimum- and maximum-pressure observation for every reported
  temperature/composition group; this deterministic selection spans the full
  experimental domain without silently treating a model-generated grid as
  experiment. Temperatures, pressures, composition, density, expanded
  density uncertainty, component order, and DOI are preserved.
  [H2/CH4 doi:10.1021/acs.jced.7b01125](https://doi.org/10.1021/acs.jced.7b01125);
  [H2/N2 doi:10.1021/acs.jced.7b00694](https://doi.org/10.1021/acs.jced.7b00694);
  [H2/CO2 doi:10.1021/acs.jced.7b00213](https://doi.org/10.1021/acs.jced.7b00213);
  [NIST ThermoML archive](https://trc.nist.gov/ThermoML/Browse).
- `verschoyle_1932_h2_co_vle.csv` transcribes all 39 coexisting compositions
  printed at 73.2 and 83.2 K in Verschoyle's H2/CO Table I. Pressure remains in
  standard atmospheres and compositions are converted from mole percent to
  mole fraction. The contemporary scan is difficult to parse automatically,
  so totals (`x_H2+x_CO=1` and `y_H2+y_CO=1`) and printed K ratios were used as
  transcription cross-checks. The historical table does not provide
  pointwise uncertainties. Values were checked against the US National Bureau
  of Standards compilation rather than against an equation-of-state output.
  [Original article doi:10.1098/rsta.1932.0006](https://doi.org/10.1098/rsta.1932.0006);
  [NBS Technical Note 108](https://www.govinfo.gov/content/pkg/GOVPUB-C13-43afd4702e63123a12bf51e0c21d182d/pdf/GOVPUB-C13-43afd4702e63123a12bf51e0c21d182d.pdf).
- `teqp_0.23.2_h2_co_vle.csv` is an offline external-software benchmark
  generated by `scripts/generate_teqp_h2_co_reference.py`. It contains
  isothermal traces started from the pure-CO saturation endpoint with teqp
  0.23.2's CoolProp pure-fluid equations and GERG binary reducing parameters.
  It is not experimental data and is not numerically identical to native
  GERG-2008, whose original simplified hydrogen pure-fluid equation is
  retained. The generator, model label, temperature, pressure, and both phase
  compositions are recorded in every row.
  [teqp VLE tracing documentation](https://pages.nist.gov/teqp-docs/en/latest/algorithms/VLE.html);
  [teqp repository](https://github.com/usnistgov/teqp).
- `moortgat_2025_h2_h2o_vle.csv` transcribes all 60 Table S1 rows at 273.15,
  323.15, 366.48, 422.89, and 473.15 K from Moortgat (2025). The two measured
  quantities are H2 mole fraction in liquid water and H2O mole fraction in
  the gas, both reported by the paper in per mille; blank cells preserve
  unreported measurements. Table S1 compiles six earlier experimental
  sources but does not map individual rows to those sources, so the CSV cites
  the review DOI at row level and the notebook does not claim finer
  provenance.
  [doi:10.1371/journal.pone.0332157](https://doi.org/10.1371/journal.pone.0332157).
- `folas_2005_azeotropes.csv` transcribes all seven ethanol–water and
  2-propanol–water states in Folas et al. (2005), Table 3. The experimental
  composition and pressure columns retain blank cells where the paper reports
  no value. The `ecr_single` columns are the paper's rounded ECR-CPA model
  results, not measurements. Pure CPA parameters come from Table 1 and the
  fitted binary interaction parameters from Table 3.
  [doi:10.1021/ie048832j](https://doi.org/10.1021/ie048832j).
- `cpa_yan_2009_water_content_digitized.csv` contains 236 experimental marker
  centres from four solid-symbol isotherms for each of methane, ethane,
  propane, and n-butane in Yan, Kontogeorgis, and Stenby (2009), Figs. 3-4
  and supplementary Figs. 3-4. These are visual digitizations, not
  author-supplied tabular values. The CSV reports conservative pressure and
  relative-composition reading uncertainties. Pixel centres and axis
  calibrations are reviewable in `scripts/digitize_cpa_yan_2009.py`; the
  script is not a CI input because the copyright-controlled source figures are
  intentionally neither tracked nor shipped.
  [doi:10.1016/j.fluid.2008.10.007](https://doi.org/10.1016/j.fluid.2008.10.007).
- `pedersen_2024_hv_propane_water.csv` transcribes all 14 experimental
  mutual-solubility states in Pedersen et al. (2024), Table 16.2. The book
  reproduces measurements by
  [Kobayashi and Katz (1953)](https://doi.org/10.1021/ie50518a051);
  `torch-flash` uses the SRK/HV parameters in Table 16.3. The two `thermopack`
  columns were generated separately with ThermoPack 2.2, SRK/HV, its default
  classical alpha correlations, and those parameters. They are software
  baselines, not experimental data.
  [Book doi:10.1201/9780429457418](https://doi.org/10.1201/9780429457418).
- `thermopack_2_2_3_srk_hv_n_butane_water_flash.csv` freezes all 70
  positive-composition fixed-\(T,P\) coexistence calculations from the
  Jaubert n-butane/water subset. ThermoPack receives pseudo-components with
  the exact package critical constants, acentric factors, molar masses, SRK
  classic-alpha convention, and fitted HV-NRTL parameters. The midpoint of
  each reported tie line is used only as an overall composition inside the
  two-phase region. `thermopack_2_2_3_srk_hv_n_butane_water_states.csv`
  separately freezes liquid- and vapor-root volume and log-fugacity results
  for 24 homogeneous states over 310.93--637.15 K and 1--830 bar. Both files
  are implementation-verification baselines, not experiments, and can be
  regenerated by `scripts/generate_thermopack_hv_reference.py` in the
  `benchmarks` Pixi environment.
  [ThermoPack cubic API](https://thermotools.github.io/thermopack/vcurrent/cubic_methods.html).
- `pedersen_2024_gerg_z.csv` transcribes all 21 pressures and the experimental
  and rounded GERG-2008 compressibility factors in Pedersen et al. (2024),
  Table 7.20. The composition is 2.20 mol% N2, 89.92 mol% CH4, 6.28 mol% C2H6,
  and 1.60 mol% C3H8 at 315.00 K. No interpolation is applied.
  [Book doi:10.1201/9780429457418](https://doi.org/10.1201/9780429457418).
- `gernert_span_2016_eoscg_table8.csv` transcribes all 30 pressure
  verification states in Gernert and Span (2016), Table 8. Component names
  are normalized to package identifiers; the printed temperature, density,
  composition, and pressure values are otherwise retained without
  interpolation. These are published numerical verification states, not
  experiments.
  [doi:10.1016/j.jct.2015.05.015](https://doi.org/10.1016/j.jct.2015.05.015).
- `eoscg2021_mdea_density_experimental.csv` transcribes all 35 MDEA density
  measurements and expanded uncertainties from Neumann et al. (2022),
  Table 1. Pressure is converted from MPa to Pa; temperature and density are
  retained in the paper's units.
  [doi:10.1007/s10765-021-02933-7](https://doi.org/10.1007/s10765-021-02933-7).
- `eoscg2021_mdea_speed_of_sound_experimental.csv` transcribes all 44 MDEA
  speed-of-sound measurements and expanded uncertainties from Neumann et al.
  (2022), Tables 4 and 5. The `apparatus` column preserves the two reported
  experimental series; pressure alone is converted from MPa to Pa.
  [doi:10.1007/s10765-021-02933-7](https://doi.org/10.1007/s10765-021-02933-7).

## Published model-generated databank

- `gerg2008_h2_ch4_reference.csv` contains every one of the 1,010 H2/CH4
  states in the hydrogen-stream databank published with
  Hassanpouryouzband et al. (2020). The source bank was itself generated with
  GERG-2008, so density and heat-capacity comparisons are model-reproduction
  checks rather than experimental validation. Pressure is converted from MPa
  to Pa; the source density, viscosity, thermal conductivity, heat capacity,
  enthalpy, and entropy values are retained.
  [Article doi:10.1038/s41597-020-0568-6](https://doi.org/10.1038/s41597-020-0568-6);
  [figshare archive](https://doi.org/10.6084/m9.figshare.12063297).

## Independently generated software baselines

- `thermopack_pr78_covolume.csv` contains six homogeneous PR78 states
  generated with ThermoPack 2.2.3 for C1/NC10, its own frozen pure constants,
  its default `k12=0.04361`, and `l12=-0.05`, `0.04`, and `0.10`. Each
  co-volume setting includes one liquid and one vapor root request, molar
  volume, and both log fugacity coefficients. Regenerate it with
  `scripts/generate_thermopack_covolume_reference.py` in the `benchmarks`
  environment. It verifies the convention
  \(b_{ij}=(b_i+b_j)(1-l_{ij})/2\) and is not experimental validation.
  [ThermoPack cubic API](https://thermotools.github.io/thermopack/vcurrent/cubic_methods.html#get-lij);
  [Privat and Jaubert (2023)](https://doi.org/10.1016/j.fluid.2022.113697).
- `gerg2008_co2_water_hou_teqp_thermopack.csv` contains density and
  fugacity-coefficient values at both measured phase compositions of all six
  Hou CO2/H2O states. Values were generated with teqp 0.23.2's exact
  `GERG2008resid` model and independently checked with ThermoPack 2.2.3
  `GERG2008`; the maximum difference in \(\ln f_i\) is retained per row. This
  is a software/model baseline, not experimental data. The experimental
  compositions remain in `jaubert_2020_co2_binary_vle.csv`; regenerate the
  audit with `scripts/generate_gerg_co2_water_reference.py` in the
  `benchmarks` environment.
  [GERG-2008 doi:10.1021/je300655b](https://doi.org/10.1021/je300655b);
  [Hou et al.](https://doi.org/10.1016/j.supflu.2012.11.011);
  [teqp 0.23.2](https://github.com/usnistgov/teqp/tree/v0.23.2);
  [ThermoPack 2.2.3](https://pypi.org/project/thermopack/2.2.3/).
- `teqp_pr_binary.csv` contains four homogeneous states generated with
  [teqp 0.23.2](https://github.com/usnistgov/teqp/tree/v0.23.2)
  `canonical_PR`. The teqp calculation received the exact critical
  temperatures, critical pressures, and acentric factors stored by
  `torch-flash`; this isolates algebra and root selection from component-data
  differences.
  [Canonical-cubic documentation](https://pages.nist.gov/teqp-docs/en/main/models/cubics.html).
- `eoscg2021_co2_h2_teqp_reference.csv` contains 36 residual-Helmholtz and
  pressure states generated independently with teqp 0.23.2 from the published
  EOS-CG-2021 coefficient tables. The grid is frozen before native-model
  evaluation and is not fitted by `torch-flash`.
  [EOS-CG-2021](https://doi.org/10.1007/s10765-023-03263-6);
  [supplementary coefficient tables](https://static-content.springer.com/esm/art%3A10.1007%2Fs10765-023-03263-6/MediaObjects/10765_2023_3263_MOESM1_ESM.pdf);
  [teqp multifluid formulation](https://pages.nist.gov/teqp-docs/en/main/models/multifluid.html).
- `thermopack_pr_flash.csv` contains one methane/n-butane TP flash generated
  with
  [ThermoPack 2.2.3](https://pypi.org/project/thermopack/2.2.3/),
  component identifiers `C1,NC4`, PR, `Classic` alpha, and van der Waals
  mixing. ThermoPack's component database differs slightly from the bundled
  constants, so the regression uses an engineering tolerance rather than
  bitwise equality.
- `thermopack_pr_phase_envelope.csv` contains eight isothermal bubble points
  and eight dew points generated with ThermoPack 2.2 for equimolar methane +
  n-butane using PR, classic alpha, and van der Waals mixing. Incipient-phase
  compositions and pressures are both retained.
  [ThermoPack releases](https://github.com/thermotools/thermopack/releases).
- `thermopack_2_2_3_pr78_co2_n2_state.csv` freezes the complete homogeneous
  phase-property output used in the external CO2/N2 derivative study at
  283.15 K, 6 MPa, and `z=[0.9, 0.1]`: flash fractions and compositions,
  translated volume, `ln(phi)`, and all ThermoPack `d/dT`, `d/dP`, and
  one-mole-basis `d/dn` values. The exact ThermoPack 2.2.3 defaults are
  retained explicitly: `Tc=[304.2, 126.161] K`,
  `Pc=[7.3765, 3.3944] MPa`, `omega=[0.225, 0.04]`,
  `kij(CO2,N2)=-0.036`, and volume-shift signs converted to the
  `torch-flash` added-volume convention.
- `thermopack_2_2_3_pr78_co2_n2_envelope.csv` freezes all 55 points returned
  by ThermoPack 2.2.3 `get_envelope_twophase(minP=1 bar, maxP=200 bar,
  calc_v=True)` for the same mixture. The rows remain in continuation order;
  they must not be sorted by temperature across the retrograde turn. The
  independently returned critical state is 296.737516055 K,
  8.7690148666 MPa, and 9.8251742524e-5 m3/mol.
  [ThermoPack 2.2.3](https://pypi.org/project/thermopack/2.2.3/);
  [phase-envelope API](https://thermotools.github.io/thermopack/vcurrent/getting_started.html).
- `neqsim_pr_flash.csv` contains one methane/n-butane TP flash generated with
  [neqsim-python 3.16.0](https://pypi.org/project/neqsim/3.16.0/), the PR
  model, classic mixing, disabled volume correction, and components `methane`
  and `n-butane`. NeqSim's independent component database likewise precludes
  a bitwise comparison.

## Optional stress corpus

The exhaustive Rachford–Rice corpus is not redistributed. It is fetched from
[`WhitsonAS/Rachford-Rice-Contest`](https://github.com/WhitsonAS/Rachford-Rice-Contest)
at commit
[`503b92f1b2847c4459326841f538739bcb9d629f`](https://github.com/WhitsonAS/Rachford-Rice-Contest/tree/503b92f1b2847c4459326841f538739bcb9d629f).
CI downloads it to the runner's temporary directory, and local runs select it
with `TORCH_FLASH_WHITSON_DATA`. No regression depends on untracked local
reference material.
