# Two-Phase Flash

## Problem statement

An isothermal-isobaric two-phase flash specifies
\((T,P,\boldsymbol{z})\) and solves for liquid composition
\(\boldsymbol{x}\), vapor composition \(\boldsymbol{y}\), and vapor fraction
\(\beta\). The component balances are

\[
z_i=(1-\beta)x_i+\beta y_i.
\]

Defining equilibrium ratios \(K_i=y_i/x_i\) gives

\[
x_i=\frac{z_i}{1+\beta(K_i-1)},\qquad
y_i=K_i x_i.
\]

Normalization of both phases produces the Rachford-Rice equation

\[
F(\beta)
=
\sum_i
\frac{z_i(K_i-1)}
     {1+\beta(K_i-1)}
=0.
\]

The material-balance equation originates with
[Rachford and Rice (1952)](https://doi.org/10.2118/952327-G). Its admissible
root must keep all denominators and phase compositions positive.

## Coupling material balance and thermodynamics

For a \(\phi\)-\(\phi\) equation-of-state calculation,

\[
f_i^L=x_i\phi_i^L P,\qquad
f_i^V=y_i\phi_i^V P.
\]

The equilibrium condition \(f_i^L=f_i^V\) implies

\[
K_i=\frac{\phi_i^L}{\phi_i^V}.
\]

Because each fugacity coefficient depends on its phase composition, the
Rachford-Rice balance and the fugacity equations must be iterated together.
`torch-flash` uses:

1. optional tangent-plane stability analysis;
2. explicit initial \(K\) values or a Wilson estimate when critical constants
   are available;
3. a bounded Rachford-Rice solve;
4. damped successive substitution in \(\ln K\); and
5. an autodifferentiated Newton correction with line search.

This follows the phase-split strategy of
[Michelsen, Part II](https://doi.org/10.1016/0378-3812%2882%2985002-4).
The reported residual is

\[
\max_i\left|
\ln K_i-\left(\ln\phi_i^L-\ln\phi_i^V\right)
\right|,
\]

which is the maximum dimensionless log-fugacity mismatch at the current
split.

## Runnable example

```python
--8<-- "docs/examples/two_phase_flash.py"
```

The explicit `check_stability=False` makes this a solve on a known two-phase
branch. For a general feed state, retain the default stability test:

```python
result = two_phase_flash(
    model,
    state,
    check_stability=True,
    tolerance=1.0e-8,
    raise_on_failure=True,
)
```

If the stability result is homogeneous, `result.nphases == 1`. If a split is
returned, phase fractions are ordered with the liquid-like solver root first
and the vapor-like root second; `result.phase_kinds` contains the separate
physical-identification diagnostic.

## Failure modes and interpretation

- **No Rachford-Rice bracket:** the current \(K\) values do not support two
  positive phases for the specified feed.
- **Trivial solution:** \(K_i\rightarrow1\) and
  \(\boldsymbol{x}\rightarrow\boldsymbol{y}\). Do not interpret this
  algebraic state as coexistence without a dedicated critical calculation.
- **Near-critical ill-conditioning:** phase compositions and roots coalesce,
  so small residuals need not imply well-resolved phase properties.
- **Metastable split:** equality of fugacities can hold even though another
  phase or phase count has lower Gibbs energy.
- **Non-convergence:** use `raise_on_failure=True` for workflows that must not
  continue with an invalid result; otherwise inspect the emitted convergence
  warning and `result.converged`.

For batched states already known to be two phase, use
`batched_two_phase_flash`. It intentionally does not perform per-state phase
discovery.

## Trust-region alternative for difficult states

`two_phase_trust_region_flash` minimizes the two-phase Gibbs energy in direct
component mole amounts. The initially larger phase amount for each component
is dependent, so every accepted iterate satisfies material balance exactly.
The dense trust-region subproblem uses the exact PyTorch-autodifferentiated
gradient and Hessian. `tangent_plane_stability(..., minimizer="trust-region")`
selects the corresponding formal-mole-number TPD minimizer.

```python
from torch_flash import two_phase_trust_region_flash

result = two_phase_trust_region_flash(
    model,
    state,
    check_stability=True,
    tolerance=1.0e-8,
    raise_on_failure=True,
)
```

The formulation follows Petitfrere and Nichita's restricted-step phase
equilibrium method
([doi:10.1016/j.fluid.2013.08.039](https://doi.org/10.1016/j.fluid.2013.08.039)).
It is an opt-in path, not the default replacement for the \(\ln K\) flash:
exact Hessians cost more than the small linear systems in ordinary
well-conditioned two-phase states.

For a fixed three-or-more-phase hypothesis,
`multiphase_trust_region_flash` applies the paper's per-component reference
phase and direct-mole Gibbs formulation. It returns explicit residual,
material-balance, accepted/rejected-step, and Hessian-curvature diagnostics.
The requested phase count and roots remain scientific inputs; convergence of
this local minimum is not proof of global stability.
