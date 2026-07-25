# Phase Properties and Their Derivatives

## Homogeneous state before equilibrium

A thermodynamic property at a supplied \((T,P,\boldsymbol{x})\) state is a
different question from the equilibrium phase split of an overall
\((T,P,\boldsymbol{z})\) feed. `phase_properties` evaluates the former
directly and never calls a flash solver.

For a selected equation-of-state root,

\[
Z=\frac{Pv}{RT},\qquad
f_i=x_i\phi_iP,
\]

and, with standard pressure \(p^\circ=1\ \mathrm{bar}\),

\[
\ln\left(\frac{f_i}{p^\circ}\right)
=
\ln x_i+\ln\phi_i+\ln\left(\frac{P}{p^\circ}\right).
\]

The default standard-state convention defines

\[
\mu_i
=
RT\ln\left(\frac{f_i}{p^\circ}\right).
\]

It intentionally does not invent ideal-gas formation terms. Supply an
explicit standard state when absolute chemical potentials are required.

The reference-independent residual Gibbs departure is

\[
\frac{g^R}{RT}=\sum_i x_i\ln\phi_i,
\]

and the corresponding residual Helmholtz departure is

\[
\frac{a^R}{RT}
=
\frac{g^R}{RT}-Z+1+\ln Z.
\]

These relationships follow the partial-molar-property and fugacity treatment
in Michelsen and Mollerup, chapter 1.

## Derivative coordinates

`state_derivatives` returns first derivatives at one scalar-\(T\), scalar-\(P\)
state in three composition coordinate systems:

- **softmax logits:** unconstrained coordinates mapped to normalized
  composition;
- **independent mole fractions:** \(x_1,\ldots,x_{n-1}\), with
  \(x_n=1-\sum_{i=1}^{n-1}x_i\);
- **mole numbers:** evaluated at \(n_i=x_i\) mol on a one-mole basis.

Temperature and pressure derivatives hold composition fixed. Mole-number
derivatives hold \(T\) and \(P\) fixed. Because the differentiated properties
are intensive, a mole-number derivative at another total amount is the
one-mole-basis result divided by that amount in mol.

The returned families include derivatives of:

- fugacity coefficients and their logarithms;
- fugacities and dimensionless log fugacities;
- chemical potentials and reduced chemical potentials;
- molar volume; and
- molar Gibbs energy with respect to \(T\) and \(P\).

Fugacity derivatives have Pa per coordinate units, chemical-potential
derivatives have J/mol per coordinate units, and molar-volume derivatives
have m3/mol per coordinate units. Composition must be strictly positive
because logarithmic derivatives are singular on the simplex boundary.

## Runnable example

```python
--8<-- "docs/examples/properties_and_derivatives.py"
```

The derivative implementation uses `torch.func` transformations rather than
finite differences. Native model operations remain differentiable with
respect to state variables and trainable model tensors unless the API
explicitly documents a nondifferentiable root-location boundary.

## Caloric properties

For scalar states, `phase_properties(..., caloric=True)` evaluates residual
enthalpy and entropy from temperature derivatives of the residual Gibbs
energy. Absolute enthalpy, entropy, heat capacity, speed of sound, and
Joule-Thomson quantities require an ideal-gas reference model and are exposed
through `thermal_properties`.

```python
from torch_flash import poling_ideal_gas, thermal_properties

ideal = poling_ideal_gas(["methane", "n_butane"])
thermal = thermal_properties(model, state, ideal, phase="stable")
print(thermal.molar_enthalpy)
print(thermal.isobaric_heat_capacity)
print(thermal.speed_of_sound)
```

The source and validity of the ideal-gas polynomial matter just as much as the
residual equation of state. See the [parameter guide](../parameters.md) and
[reference index](../references.md).

## Numerical cautions

- A derivative of a selected root is local to that branch; it is not a
  derivative through a phase switch.
- Root coalescence and critical conditions can make response functions
  ill-conditioned.
- Float32 is a separate accuracy study, not an interchangeable equilibrium
  reference precision.
- Do not detach tensors or convert them to NumPy inside a differentiable
  objective.
- Validate derivative shape, units, and coordinates before coupling the
  result to an optimizer or simulator.
