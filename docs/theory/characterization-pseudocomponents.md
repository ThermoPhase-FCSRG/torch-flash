# Characterization and Pseudo-Components

## Why characterization is needed

A petroleum plus fraction such as C7+ or C20+ represents material unresolved
by detailed compositional analysis. It is not a single chemical compound.
Characterization replaces that aggregate with a discrete distribution and
then with a manageable set of pseudo-components for an equation of state.

The operations answer different questions:

1. **splitting** estimates a single-carbon-number-like distribution;
2. **property characterization** assigns density and EoS properties;
3. **lumping** combines contiguous cuts while preserving selected moments; and
4. **fluid-model construction** combines the pseudo-components with defined
   light components and interaction parameters.

Whitson and Brulé develop the shifted-gamma workflow in chapter 5 of
*Phase Behavior*. Pedersen, Christensen, and Shaikh develop logarithmic
splitting, property correlations, and lumping in chapter 5
([book DOI](https://doi.org/10.1201/9780429457418)).

## Moment constraints

For a plus fraction with total mole fraction \(z_+\), mean molar mass
\(\overline{M}_+\), and discrete cuts \(N\),

\[
\sum_N z_N=z_+,
\qquad
\frac{\sum_N z_NM_N}{z_+}=\overline{M}_+.
\]

A density assignment should also preserve the supplied bulk density under the
chosen volume-mixing convention:

\[
\rho_+
=
\frac{\sum_N z_NM_N}
{\sum_N z_NM_N/\rho_N}.
\]

These are conservation constraints, not model validation. Different
distribution families can satisfy the same moments while giving different
phase behavior.

## Implemented split families

`pedersen_logarithmic_split` constructs a finite molar distribution while
matching the specified plus-fraction mole and molar-mass balances.
`pedersen_density_split` assigns a logarithmic carbon-number density trend and
matches the specified bulk density.

`whitson_gamma_split` discretizes a shifted gamma distribution. Its final
requested bin includes the remaining infinite tail, so truncating the nominal
carbon-number labels does not discard the tail mass.

`pedersen_cubic_properties` then maps molar mass and density to EoS-specific
critical temperature, critical pressure, and acentric factor. The SRK and PR
coefficient sets are distinct; they must not be swapped.

## Lumping

`equal_weight_lump` forms contiguous pseudo-components with approximately
equal mass. It preserves total moles and mass, uses ideal-volume mixing for
density, and can carry additional mass-weighted characterized properties.

A lumping scheme is part of the model definition. Changing boundaries changes
critical properties, binary interactions, and phase behavior even if total
plus-fraction mass is unchanged.

## Runnable characterization example

```python
--8<-- "docs/examples/characterization.py"
```

The example uses only SI units:

- molar mass in kg/mol;
- density in kg/m3;
- critical temperature in K; and
- critical pressure in Pa.

The resulting `LumpedDistribution` contains pseudo-component names, carbon
number bounds, mole fractions, molar masses, densities, and the supplied
characterized properties. It is immutable scientific input for subsequent
model construction.

## Scientific and numerical cautions

- Prefer measured extended compositions when available; a correlation should
  not replace resolved measurements.
- Record the plus-fraction definition, measured moments, split family,
  truncation/tail convention, density anchor, EoS property correlation, and
  lump boundaries.
- Refit or validate binary interactions after changing characterization.
- Preserve mole, mass, and chosen volume moments numerically.
- Diagnose sensitivity to the number and placement of lumps.
- Label pseudo-components by their actual construction; do not present them as
  pure compounds.
- A fit to the same PVT data used to characterize the heavy end is
  calibration, not independent validation.
