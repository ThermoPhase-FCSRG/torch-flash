# Chemical Equilibrium Overview

## Equilibrium at fixed temperature and pressure

For a closed multicomponent system at fixed \(T\), \(P\), and total component
amounts, the equilibrium state minimizes total Gibbs energy subject to the
component material balances. For every component present in coexisting
phases, the stationarity conditions are

\[
T^{(\alpha)}=T,\qquad
P^{(\alpha)}=P,\qquad
\mu_i^{(\alpha)}=\mu_i^{(\beta)}.
\]

Chemical potential is related to fugacity by

\[
\mu_i(T,P,\boldsymbol{x})
=
\mu_i^\circ(T)
+
RT\ln\left(\frac{f_i}{f^\circ}\right),
\]

so equality of chemical potentials is equivalent to

\[
f_i^{(\alpha)}=f_i^{(\beta)}
\quad\text{for every component }i.
\]

These conditions follow from constrained Gibbs-energy minimization. The
derivation and its computational consequences are developed in Michelsen and
Mollerup, chapters 1 and 8, and in Kontogeorgis and Folas, chapter 1. See the
[bibliographic records](../references.md#equilibrium-algorithms-and-classical-models).

## Necessary conditions are not enough

Equal fugacities establish stationarity, not global stability. A solver can
converge to a local stationary point, a metastable state, or the algebraic
trivial solution in which nominally distinct phases have the same
composition. This becomes especially important near critical conditions,
where phase compositions and roots coalesce.

For an equation-of-state \(\phi\)-\(\phi\) formulation, a dimensionless
tangent-plane distance for a normalized trial composition
\(\boldsymbol{w}\) can be written

\[
\operatorname{TPD}(\boldsymbol{w})
=
\sum_i w_i
\left[
\ln w_i
+
\ln\phi_i(T,P,\boldsymbol{w})
-
\ln z_i
-
\ln\phi_i(T,P,\boldsymbol{z})
\right].
\]

The reference phase is stable only if the global minimum of TPD over all
admissible trial compositions is nonnegative, within a tolerance chosen for
the model and dtype. A materially negative minimum identifies a composition
direction that lowers Gibbs energy. This criterion and practical minimization
algorithms follow Michelsen's stability analysis
([Part I](https://doi.org/10.1016/0378-3812%2882%2985001-2)).

Numerically, a nonnegative minimum is meaningful only if the minimization
itself converged and explored the relevant composition basins. Zero
compositions require careful limiting treatment because the logarithms are
singular.

## Stability in torch-flash

```python
import torch

from torch_flash import (
    ChemicalState,
    component_set,
    configure,
    peng_robinson_1978,
    tangent_plane_stability,
)

runtime = configure(device="cpu", dtype=torch.float64)
model = peng_robinson_1978(component_set(("methane", "n_butane")))
state = ChemicalState(
    runtime.tensor(450.0),
    runtime.tensor(3.0e6),
    runtime.tensor([0.50, 0.50]),
)

stability = tangent_plane_stability(model, state)
print(stability.stable)
print(stability.minimum_tpd)
print(stability.trial_composition)
print(stability.converged, stability.iterations)
```

`two_phase_flash` performs this test by default and returns one homogeneous
phase when the test is stable. When a branch is already known—for example,
during a controlled continuation study—the stability test can be disabled,
but that changes the scientific question from phase discovery to a
branch-constrained split.

## What to verify

For any equilibrium result:

1. check solver convergence and the reported residual;
2. reconstruct the overall material balance;
3. compare component fugacities between phases;
4. reject nonpositive or nonnormalized compositions;
5. distinguish phase-root selection from physical phase identification; and
6. perform a stability analysis when the phase count was not externally
   established.

The next pages apply these conditions to [two phases](two-phase-flash.md) and
to a [fixed multiphase count](multiphase-flash.md).
