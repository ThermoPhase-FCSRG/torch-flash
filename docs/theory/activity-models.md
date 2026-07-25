# Activity Models

## Excess Gibbs energy and activity coefficients

An activity-coefficient model represents liquid-phase nonideality through
excess Gibbs energy:

\[
\frac{g^E}{RT}=\sum_i x_i\ln\gamma_i.
\]

For a thermodynamically consistent model, component activity coefficients are
partial-molar derivatives of the extensive excess Gibbs energy. In a
liquid-phase fugacity formulation,

\[
f_i^L=x_i\gamma_i f_i^{\mathrm{ref}},
\]

where the reference fugacity may also include vapor-pressure, fugacity, and
Poynting corrections. An activity model by itself therefore does not define a
complete flash: the standard state and vapor-phase model must also be named.

The excess-Gibbs framework and local-composition interpretation are developed
in Kontogeorgis and Folas, chapters 4-6
([book DOI](https://doi.org/10.1002/9780470747537)).

## Wilson

The Wilson model uses local-composition ratios \(\Lambda_{ij}\):

\[
\ln\gamma_i
=
1-\ln\left(\sum_j x_j\Lambda_{ij}\right)
-
\sum_k
\frac{x_k\Lambda_{ki}}
{\sum_j x_j\Lambda_{kj}}.
\]

Its asymmetric interaction parameters combine energetic and molar-volume
effects. Wilson's original formulation is given in
[Wilson (1964)](https://doi.org/10.1021/ja01056a002).

## NRTL

For NRTL,

\[
G_{ij}=\exp(-\alpha_{ij}\tau_{ij}),
\]

and

\[
\ln\gamma_i
=
\sum_j
\frac{x_j\tau_{ji}G_{ji}}{\sum_k x_kG_{ki}}
+
\sum_j
\frac{x_jG_{ij}}{\sum_k x_kG_{kj}}
\left[
\tau_{ij}
-
\frac{\sum_m x_m\tau_{mj}G_{mj}}{\sum_k x_kG_{kj}}
\right].
\]

\(\tau_{ij}\) is a directed interaction and \(\alpha_{ij}\) controls
nonrandomness. Parameter units and temperature dependence are part of the
model definition, not interchangeable implementation details. See
[Renon and Prausnitz (1968)](https://doi.org/10.1002/aic.690140124).

## Original UNIFAC

Original UNIFAC decomposes the activity coefficient into combinatorial and
residual contributions:

\[
\ln\gamma_i
=
\ln\gamma_i^\mathrm{C}
+
\ln\gamma_i^\mathrm{R}.
\]

The combinatorial term uses molecular group volumes and surfaces. The
residual term uses subgroup counts and directed main-group interaction
energies. A component fragmentation is therefore scientific input: the same
name with a different structural assignment is a different calculation.

`torch-flash` provides the original VLE-UNIFAC form and does not treat
Dortmund UNIFAC, PSRK, or other variants as parameter substitutions. The
defining and revision sources are
[Fredenslund et al. (1975)](https://doi.org/10.1002/aic.690210607),
[Hansen et al. (1991)](https://doi.org/10.1021/ie00058a017), and
[Wittig et al. (2003)](https://doi.org/10.1021/ie020506l).

## Huron-Vidal coupling

Huron-Vidal mixing uses an excess-Gibbs model to define the attractive mixing
term of a cubic equation of state at an infinite-pressure reference. This
provides an EoS fugacity model for strongly nonideal mixtures rather than
merely adding a standalone \(\gamma_i\) to a cubic calculation. The original
mixing rule is due to
[Huron and Vidal (1979)](https://doi.org/10.1016/0378-3812%2879%2980001-1).

The `activity_model` constructor supports NRTL, Wilson, original UNIFAC, and
Huron-Vidal NRTL parameter documents. The bundled release contains the
public original-UNIFAC table and the documented Huron-Vidal sets; custom NRTL
and Wilson documents use the same versioned schema. Use the parameter-set
identifier to keep the model, units, components, source, and fit identity
explicit.

## Runnable original-UNIFAC example

```python
--8<-- "docs/examples/activity_model.py"
```

The example differentiates \(g^E/(RT)\) with respect to both temperature and
composition. The unconstrained PyTorch composition gradient is not the same
object as a derivative restricted to the normalized composition simplex;
choose coordinates appropriate to the downstream calculation.

## Model-selection cautions

- Do not combine interaction parameters from different model variants.
- Check the parameter temperature range and component identities.
- A calibration fit is not independent validation.
- A standalone activity model does not define vapor fugacity or a complete
  phase-equilibrium standard state.
- For high-pressure work, use a documented EoS mixing rule or fugacity
  formulation rather than extrapolating a low-pressure shortcut without
  evidence.
