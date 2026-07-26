# H2-tailored GERG (2021)

This report reproduces selected numerical and graphical results from
R. Beckmüller et al., “New Equations of State for Binary Hydrogen Mixtures
Containing Methane, Nitrogen, Carbon Monoxide, and Carbon Dioxide,” *Journal of
Physical and Chemical Reference Data* **50** (2021), 013102,
[doi:10.1063/5.0040533](https://doi.org/10.1063/5.0040533). The article
develops hydrogen-tailored multifluid equations for the CH4/H2, N2/H2, CO/H2,
and CO2/H2 binaries to improve their representation relative to GERG-2008.

The `torch-flash` study specifically reproduces the paper's Table 12 and
Figures 4, 7, and 16. Table 12 is the fundamental numerical implementation
criterion; Figure 4 compares calculated CH4/H2 densities and phase boundaries
from GERG-2008 and the H2-tailored model; Figures 7 and 16 compare calculated
N2/H2 and CO2/H2 phase-equilibrium curves with the experimental markers shown
in the article.

The implementation uses the article's main five-component parameterization:
GERG-2008 pure-fluid equations for CH4, N2, CO, and CO2; the Leachman equation
for normal hydrogen; and the published binary reducing and departure
parameters. The supplementary parameterization based on newer reference
equations is a distinct model and is not substituted here.

## Table 12 implementation verification

![H2-tailored GERG Table 12 property parity](../assets/validation/29_h2_gerg_table12_parity.png)

All 16 states and all six reported properties are shown: pressure, isobaric
heat capacity, speed of sound, enthalpy, entropy, and molar Helmholtz energy.
This is verification against published model-calculation values.

## Figure 4: CH4/H2 density difference

![H2-tailored GERG Figure 4 density-difference reproduction](../assets/validation/29_h2_gerg_figure4_density_difference.png)

The contours compare GERG-2008 with the H2-tailored parameterization at 25,
50, and 75 mol % H2. The shaded regions and boundary curves are calculated
phase-equilibrium results rather than experimental observations.

## Figure 7: N2/H2 phase equilibrium

![H2-tailored GERG Figure 7 N2-H2 phase-equilibrium reproduction](../assets/validation/29_h2_gerg_figure7_n2_h2_vle.png)

The native H2-tailored and GERG-2008 traces are shown at 70.4, 90.8, and
110.3 K together with visually digitized experimental markers.

## Figure 16: CO2/H2 phase equilibrium

![H2-tailored GERG Figure 16 CO2-H2 phase-equilibrium reproduction](../assets/validation/29_h2_gerg_figure16_co2_h2_vle.png)

The corresponding CO2/H2 comparison covers 235, 260, and 295.7 K. The VLE
markers were visually digitized from the paper because numerical phase-
equilibrium data were unavailable; the plotted comparison therefore includes
digitization uncertainty and should not be read as a replacement for the
authors' original data.
