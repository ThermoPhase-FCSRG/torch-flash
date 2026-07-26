# Fugacity Models

## Why fugacity is the equilibrium variable

For a real mixture at fixed temperature,

\[
d\mu_i=RT\,d\ln f_i.
\]

Fugacity has pressure units and becomes partial pressure in the ideal-gas
limit. In an equation-of-state formulation,

\[
f_i=x_i\phi_iP,
\]

where the fugacity coefficient \(\phi_i\) measures departure from ideal-mixture
behavior for the selected phase root. Equality of chemical potential between
phases is therefore implemented as equality of fugacity.

The thermodynamic derivation is given by Michelsen and Mollerup in their
partial-molar-property and fugacity sections. Cubic equation-of-state
expressions follow their defining model sources, including
[Peng and Robinson (1976)](https://doi.org/10.1021/i160057a011) and
[Soave (1972)](https://doi.org/10.1016/0009-2509%2872%2980096-4).

## Phi-phi and gamma-phi formulations

In a \(\phi\)-\(\phi\) flash, the same equation-of-state family supplies both
phase fugacity coefficients:

\[
x_i\phi_i^LP=y_i\phi_i^VP.
\]

In a \(\gamma\)-\(\phi\) flash, a liquid activity model supplies

\[
f_i^L=x_i\gamma_i f_i^\mathrm{ref},
\]

while an equation of state supplies vapor fugacity. The reference fugacity
must specify every correction and standard-state convention. These two
formulations are not interchangeable merely because both return an
equilibrium ratio.

Huron-Vidal models in `torch-flash` instead embed an excess-Gibbs contribution
in a cubic-EoS mixing rule, producing EoS fugacity coefficients for both roots.

## Roots and physical phase identity

A cubic equation of state can have several admissible volume roots. Selecting
`phase="liquid"`, `"vapor"`, or `"stable"` determines which root is evaluated.
This algebraic root choice is distinct from:

- the number of equilibrium phases;
- a density- or \(V/b\)-based physical phase label; and
- global Gibbs-energy stability.

Near critical conditions, roots coalesce and phase identification becomes
ambiguous. A homogeneous \(x=y\) solution is not evidence of two-phase
coexistence.

## Inspect fugacity equality in a flash

The complete [two-phase example](two-phase-flash.md#runnable-example) returns
phase property objects. Its equilibrium residual can also be checked directly:

```python
liquid, vapor = result.phases
log_fugacity_mismatch = (
    liquid.log_fugacities - vapor.log_fugacities
).abs().max()
print(log_fugacity_mismatch)
```

For a supplied homogeneous state, inspect both roots without solving
equilibrium:

```python
--8<-- "docs/examples/fugacity_roots.py"
```

Both calls evaluate the same \(T\), \(P\), and composition on different roots.
They do not assert that the roots coexist.

## Available model families

The package's native fugacity-capable models include:

- SRK and Peng-Robinson cubic families, including predictive and translated
  variants;
- cubic-plus-association models for associating mixtures;
- GERG-2008 and EOS-CG multiparameter Helmholtz mixture models; and
- cubic models coupled to Huron-Vidal activity terms.

Each family has a different parameter identity and validity range. Use
[model scope](../model-scope.md), the [parameter database](../parameters.md),
and [validation evidence](../validation.md) before choosing one.

## Practical checks

- Match component constants, alpha function, mixing rule, binary interactions,
  and volume translation before comparing two fugacity calculations.
- Keep `log_fugacities` dimensionless as
  \(\ln(f_i/p^\circ)\); do not take a logarithm of a dimensional fugacity
  without a reference pressure.
- Inspect root admissibility and phase identity separately.
- Compare log fugacities rather than raw fugacity ratios when components span
  many orders of magnitude.
- Treat critical-region derivatives and float32 results as separately
  conditioned calculations.
