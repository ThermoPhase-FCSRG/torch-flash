# Multiphase Flash

## Generalized material balance

For \(N_p\) phases, select phase 0 as a reference and define

\[
K_{ji}=\frac{x_i^{(j)}}{x_i^{(0)}},
\qquad j=1,\ldots,N_p-1.
\]

If \(\beta_j\) is the fraction of non-reference phase \(j\), the reference
fraction is

\[
\beta_0=1-\sum_{j=1}^{N_p-1}\beta_j.
\]

The component balance gives

\[
x_i^{(0)}
=
\frac{z_i}
{1+\sum_{j=1}^{N_p-1}\beta_j(K_{ji}-1)}
=
\frac{z_i}{D_i}.
\]

Normalization of every non-reference phase yields the generalized
Rachford-Rice residuals

\[
F_j(\boldsymbol{\beta})
=
\sum_i\frac{z_i(K_{ji}-1)}{D_i}
=0,
\qquad j=1,\ldots,N_p-1.
\]

An admissible solution requires every phase fraction and every \(D_i\) to be
positive. Thermodynamic equilibrium additionally requires, for each component
and phase pair,

\[
\ln K_{ji}
=
\ln\phi_i^{(0)}-\ln\phi_i^{(j)}
\]

in a \(\phi\)-\(\phi\) formulation.

The Gibbs-energy basis and multiphase algorithms are discussed by Michelsen
and Mollerup in their phase-equilibrium chapters. The reservoir-fluid example
below follows Pedersen, Christensen, and Shaikh, Tables 6.5-6.6
([book DOI](https://doi.org/10.1201/9780429457418)).

## Current torch-flash contract

`multiphase_flash` is a **fixed-phase-count** solver:

- `initial_k_values.shape == (nphases - 1, ncomponents)`;
- the first phase is the reference phase;
- the solver alternates generalized material balance, fugacity-coefficient
  updates, and autodifferentiated Newton steps;
- the returned residual is the maximum log-fugacity mismatch;
- physical phase kinds are assigned afterward as diagnostics; and
- automatic discovery, addition, or deletion of phases is experimental.

The function emits `ExperimentalModelWarning` so this limitation cannot be
missed. A converged three-phase solve does not prove that three is the globally
stable phase count.

## Runnable fixed-three-phase example

??? example "Pedersen Tables 6.5-6.6 initialization"

    ```python
    --8<-- "docs/examples/multiphase_flash.py"
    ```

The table compositions are rounded and their binary-interaction convention is
not fully identified in the source. The example therefore uses zero binary
interactions and treats the reported compositions only as an initializer. The
result is accepted by its own material balance and fugacity residual, not by
forcing equality with every printed digit.

The current root-based phase-identification heuristic labels the converged
phases `vapor`, `vapor`, and `liquid` for these rounded inputs. This illustrates
why `nphases`, solver root, and physical phase identity are separate concepts.
Do not relabel a phase solely to match an expected narrative.

## Choosing initial equilibrium ratios

For a known three-phase vapor-liquid-liquid problem, a physically informed
matrix is

\[
\boldsymbol{K}
=
\begin{bmatrix}
\boldsymbol{y}/\boldsymbol{x}^{(0)}\\
\boldsymbol{x}^{(2)}/\boldsymbol{x}^{(0)}
\end{bmatrix}.
\]

Possible sources are a nearby converged state, a lower-dimensional boundary,
or a documented engineering estimate. Continuation is preferable to resetting
every state from a generic initializer because multiphase equations can have
several basins and trivial solutions.

## Required diagnostics

Before accepting a multiphase result:

1. require `result.converged`;
2. compare the residual with the requested tolerance and float precision;
3. verify \(\sum_\alpha\beta_\alpha=1\);
4. verify
   \(\boldsymbol{z}=\sum_\alpha\beta_\alpha\boldsymbol{x}^{(\alpha)}\);
5. check positivity and normalization of each composition;
6. inspect phase roots, densities, and `phase_identifications`; and
7. perform stability/phase-discovery analysis outside this fixed-count solve
   when the phase count was not known in advance.
