# Theoretical Background

This section connects the thermodynamic equations to the corresponding
`torch-flash` APIs. It is a working theory guide, not a replacement for the
primary literature.

The principal textbook sources are:

- Michelsen and Mollerup, *Thermodynamic Models: Fundamentals &
  Computational Aspects*, especially the chapters on partial molar
  properties, fugacity, phase equilibrium, stability, and isothermal flash;
- Kontogeorgis and Folas, *Thermodynamic Models for Industrial Applications*,
  especially the activity-coefficient, mixing-rule, and association chapters;
- Whitson and Brulé, *Phase Behavior*, especially heavy-end characterization
  and reservoir-fluid calculation procedures; and
- Pedersen, Christensen, and Shaikh, *Phase Behavior of Petroleum Reservoir
  Fluids*, especially characterization, cubic-EoS phase behavior, and
  multiphase examples.

Full bibliographic records and primary model papers are collected in the
[scientific reference index](../references.md). Equations below are
paraphrased and organized around the implementation; published coefficient
tables remain in their cited parameter sources.

## Scope map

| Question | Theory page | Main API |
|---|---|---|
| Is a state at equilibrium and stable? | [Chemical Equilibrium Overview](chemical-equilibrium.md) | `tangent_plane_stability` |
| How is a vapor-liquid split computed? | [Two-Phase Flash](two-phase-flash.md) | `two_phase_flash`, `rachford_rice` |
| How is a fixed phase count handled? | [Multiphase Flash](multiphase-flash.md) | `multiphase_flash`, `solve_generalized_rachford_rice` |
| How are properties and sensitivities evaluated? | [Phase Properties and Their Derivatives](phase-properties-derivatives.md) | `phase_properties`, `state_derivatives` |
| How is liquid nonideality represented? | [Activity Models](activity-models.md) | `activity_model` |
| How does an EoS provide equilibrium fugacity? | [Fugacity Models](fugacity-models.md) | `log_fugacity_coefficients` |
| How are unresolved heavy fractions represented? | [Characterization and Pseudo-Components](characterization-pseudocomponents.md) | characterization and lumping APIs |

## Common notation

- \(T\): absolute temperature in K.
- \(P\): pressure in Pa.
- \(z_i\): overall mole fraction.
- \(x_i^{(\alpha)}\): mole fraction in phase \(\alpha\).
- \(\beta_\alpha\): molar fraction of phase \(\alpha\).
- \(f_i^{(\alpha)}\): component fugacity in phase \(\alpha\), in Pa.
- \(\phi_i^{(\alpha)}\): fugacity coefficient.
- \(\gamma_i\): liquid-phase activity coefficient.
- \(\mu_i\): chemical potential in J/mol.
- \(K_i^{(\alpha/0)}=x_i^{(\alpha)}/x_i^{(0)}\): equilibrium
  ratio relative to a reference phase.

Compositions are normalized molar fractions. All equations assume the model,
parameter set, standard state, and phase-root convention named by the
calculation.

## A practical reading order

For a first calculation, read the [Getting Started](../getting-started.md)
guide, then Chemical Equilibrium, Fugacity Models, and the relevant flash
page. Read Activity Models before selecting an excess-Gibbs or Huron-Vidal
model, and read Characterization before creating heavy pseudo-components.

Numerical convergence and physical validity are separate questions throughout
this section. A small algebraic residual does not establish global stability,
validate a model against experiment, or prove that a phase label is
unambiguous.
