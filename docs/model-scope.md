# Model scope

## Equations of state

The native cubic kernel implements SRK, PR76, and PR78 in generalized
two-parameter form. It supports quadratic van der Waals mixing, symmetric
constant BIPs or `kij(T) = Aij + Bij/T` with dimensionless `Aij` and `Bij`
in kelvin, symmetric dimensionless cross-co-volume interactions `lij`,
component-linear volume translation, and the original
infinite-pressure Huron–Vidal rule coupled to NRTL, Wilson, or the original
covolume-weighted HV-NRTL activity model. HV-specific parameters are not
interchangeable with ordinary low-pressure activity-model parameters.
Closed-form fugacity coefficients are used for constant and
temperature-dependent quadratic mixing and PPR78; PyTorch derivatives of the
residual Helmholtz energy are used for non-quadratic mixing.

The predictive PR78 constructor evaluates the Jaubert-Mutelet
group-contribution correlation
\[
k_{ij}(T)=
\frac{
-\tfrac12\sum_{kl}\Delta\alpha_{ij,k}\Delta\alpha_{ij,l}A_{kl}
\left(\frac{298.15}{T}\right)^{B_{kl}/A_{kl}-1}
-\left(\frac{\sqrt{a_i(T)}}{b_i}
-\frac{\sqrt{a_j(T)}}{b_j}\right)^2
}{
2\sqrt{a_i(T)a_j(T)}/(b_i b_j)
}.
\]
Here \(\Delta\alpha_{ij,k}=\alpha_{ik}-\alpha_{jk}\), \(T\) is in
kelvin, and \(A_{kl},B_{kl}\) are in pascals. The squared pure-component
term follows Eq. 5 and Appendix A of the
[primary paper](https://doi.org/10.1016/j.fluid.2004.06.059). The bundled
parameter set is the paper's original six-group set, while the YAML/API schema
accepts a separately sourced, versioned larger group inventory. The
derivation uses
\(b_m=\sum_i x_i b_i\), so PPR78 is deliberately not combined with a
cross-co-volume `lij` in the named constructor.

The implemented one-fluid co-volume convention is
\[
b_{ij}=\frac{b_i+b_j}{2}(1-l_{ij}),\qquad
b_m=\sum_i\sum_jx_i x_jb_{ij}.
\]
It follows
[Privat and Jaubert (2023)](https://doi.org/10.1016/j.fluid.2022.113697)
and the independently exercised
[ThermoPack convention](https://thermotools.github.io/thermopack/vcurrent/cubic_methods.html#get-lij).
At `lij=0`, the expression reduces exactly to Pedersen's linear
\(b_m=\sum_i x_i b_i\) rule. For nonzero `lij`, closed-form fugacity uses the
partial molar co-volume
\(\bar b_i=2\sum_jx_jb_{ij}-b_m\); the test suite independently differentiates
the extensive residual Helmholtz energy to guard this composition term.
`lij` is distinct from volume translation: it changes pressure, roots,
fugacity, and phase equilibrium, whereas translation maps the physical volume
of the parent EoS.

`VolumeTranslation` supports constant or linear-in-temperature additive
component shifts,
\[
v=v_0+\sum_i x_i d_i(T),\qquad
d_i(T)=d_{i,0}+d_{i,1}(T-T_{\mathrm{ref}}).
\]
The residual-Helmholtz reference-volume term and
\(P d_i(T)/(RT)\) log-fugacity correction are included consistently. The
literature normally writes \(v=v_0-\sum_i x_i c_i\); therefore the public
torch-flash coefficient is \(d_i=-c_i\). This explicit conversion follows
[Péneloux et al. (1982)](https://doi.org/10.1016/0378-3812%2882%2980002-2)
and the independent
[ThermoPack derivation](https://thermotools.github.io/thermopack/memo/peneloux/peneloux.pdf).

Named factories provide the Pedersen Rackett-based light-component SRK/PR
correlations, Pedersen's ASTM-anchored linear \(C7+\) correction, Whitson's
Table 4.3 pure-component factors, and Whitson's Table 4.2
paraffin/naphthene/aromatic heavy-family correlation. Pedersen Eq. 5.6 prints
the density-matching sign opposite to the book's Eq. 4.44 and n-hexane worked
example; `density_matched_translation()` follows the latter consistent pair,
\(d=M/\rho-v_{\mathrm{EOS}}\). Constant composition-linear shifts leave
fixed-temperature VLE ratios unchanged; they correct volume and density but
cannot repair a poor phase-equilibrium model. Temperature-dependent shifts
remain differentiable, but their extrapolated isotherms require physical
inspection.
The primary definitions are
[Soave (1972)](https://doi.org/10.1016/0009-2509%2872%2980096-4),
[Peng and Robinson (1976)](https://doi.org/10.1021/i160057a011),
[Robinson and Peng (1978)](https://books.google.com/books?id=bE-_HAAACAAJ),
and [Huron and Vidal (1979)](https://doi.org/10.1016/0378-3812%2879%2980001-1).
The activity-model definitions are due to
[Renon and Prausnitz (1968)](https://doi.org/10.1002/aic.690140124) and
[Wilson (1964)](https://doi.org/10.1021/ja01056a002).

## Activity models

`UNIFAC` implements the original Fredenslund solution-of-groups residual term
and original Flory-Huggins/Staverman-Guggenheim combinatorial term with
\(z=10\). The kernel accepts batch dimensions, runs on the configured PyTorch
device/dtype, exposes \(\ln\gamma_i\) and \(g^E/(RT)\), and makes the directed
main-group interaction matrix trainable on request. The extensive identity
\[
\ln\gamma_i =
\frac{\partial\left[n g^E/(RT)\right]}{\partial n_i}
\]
is checked directly with autodiff.

The bundled parameter file is original VLE-UNIFAC, not Dortmund UNIFAC,
modified UNIFAC, PSRK, or an aerosol extension. Those variants change the
combinatorial expression, temperature law, subgroup inventory, or several of
these simultaneously. The `torch-flash` original-UNIFAC validation treats
states above roughly 425 K as extrapolative, following the application range
documented by Kontogeorgis and Folas (2010, section 5.7). UNIFAC supplies a
liquid-phase excess-Gibbs model, not by itself vapor fugacity, Poynting
corrections, association physics, electrolytes, or a high-pressure flash.

The equations originate with
[Fredenslund et al. (1975)](https://doi.org/10.1002/aic.690210607).
The bundled public inventory follows the revisions of
[Hansen et al. (1991)](https://doi.org/10.1021/ie00058a017) and
[Wittig et al. (2003)](https://doi.org/10.1021/ie020506l), with exact source
identity and table discrepancies recorded in the
[parameter documentation](parameters.md#original-unifac-payload).

The CPA kernel uses an SRK physical term and Wertheim TPT1 association. It
supports `none`, 1A, 1B, 2B, 3B, and 4C site schemes; CR1/ECR combining;
explicit modified-CR1 cross parameters; constant or \(A+B/T\) attractive
BIPs; and an optional `lij` in the physical SRK term. The association
strength's pure-\(b_i\) combining convention is unchanged by that optional
physical-term parameter. The
bundled sets cover the Folas alcohol/water systems, Oliveira aliphatic and
aromatic hydrocarbon/water systems, and Yan light-hydrocarbon/water reservoir
systems. The equation follows
[Kontogeorgis et al. (1996)](https://doi.org/10.1021/ie9600203); the parameter
and solvation conventions follow
[Folas et al. (2005)](https://doi.org/10.1021/ie048832j),
[Oliveira et al. (2007)](https://doi.org/10.1016/j.fluid.2007.05.023), and
[Yan et al. (2009)](https://doi.org/10.1016/j.fluid.2008.10.007).

## Heavy-end characterization

`torch_flash.characterization` is independent of CPA, PR, SRK, or a flash
solver. It implements Pedersen's logarithmic SCN split and density balance,
Whitson's shifted-gamma discretization, contiguous equal-weight lumping, and
Pedersen SRK/PR critical-property adapters. The published definitions and
worked validation cases are in
[Whitson and Brulé (2000), chapter 5](https://books.google.com/books?id=Z4cQAQAAMAAJ)
and
[Pedersen et al. (2024), chapter 5](https://doi.org/10.1201/9780429457418).

The inferred distribution is not unique. Extended measured compositions take
precedence; the chosen endpoint, gamma shape, density anchor, and lump count
must be recorded. The last Whitson bin contains the full infinite
molecular-weight tail, which preserves total moles and mass but gives that bin
a conditional average mass above its nominal SCN label. CPA-specific mapping
of characterized boiling-point/gravity cuts is kept in a separate adapter.

## Multifluid models

`MultiFluidEOS` implements differentiable ideal and residual Helmholtz
energies, multifluid reducing functions, power/exponential/Gaussian terms,
GERG binary terms, Gao-B terms, and Span-Wagner non-analytic terms.

`gerg2008()` bundles the complete 21-component thermodynamic model: all pure
ideal and residual equations, 210 binary reducing-parameter sets, and 15
nonzero departure functions, as defined by
[Kunz and Wagner (2012)](https://doi.org/10.1021/je300655b).
`eoscg2021()` bundles the 16-component
EOS-CG-2021 model: all pure ideal and residual equations, 120 binary
reducing-parameter sets, and 21 departure functions from
[Neumann et al. (2023)](https://doi.org/10.1007/s10765-023-03263-6) and its
[supplementary tables](https://static-content.springer.com/esm/art%3A10.1007%2Fs10765-023-03263-6/MediaObjects/10765_2023_3263_MOESM1_ESM.pdf).
Both run natively in PyTorch without an external runtime and expose total Helmholtz energy,
chemical potentials, entropy, internal energy, enthalpy, Gibbs energy,
isochoric/isobaric heat capacity, and speed of sound in addition to pressure,
density, compressibility, and fugacity.

The coefficient inventories are complete for the thermodynamic Helmholtz
mixture equations. They do not bundle every optional ancillary saturation or
transport correlation from the pure-fluid source files. The optional
`TeqpBackend` remains useful as an independent CPU reference, including its
GERG-2008 and six-component
[Gernert–Span EOS-CG](https://doi.org/10.1016/j.jct.2015.05.015) adapters.
The source assigned to every EOS-CG-2021 pure-fluid equation is listed in
[Scientific references and data provenance](references.md).

## Equilibrium

The two-phase TP flash uses tangent-plane stability followed by
successive-substitution and damped Newton updates. The generalized multiphase
solver supports a caller-specified phase count and VLL/VLW-style initial
K-value matrices. The stability and phase-split algorithms are based on
[Michelsen's Part I](https://doi.org/10.1016/0378-3812%2882%2985001-2) and
[Part II](https://doi.org/10.1016/0378-3812%2882%2985002-4), respectively;
the material-balance reduction originates with
[Rachford and Rice (1952)](https://doi.org/10.2118/952327-G).

Automatic global phase discovery is experimental. For difficult polar or
multiphase systems, provide physically informed initialization and inspect
material-balance, fugacity, stability, and convergence diagnostics.

An EoS root label and a physical phase name are kept separate.
`PhaseProperties.kind` records the requested root, whereas
`PhaseProperties.phase_identification` records a likely physical identity,
the method, its numerical criterion, and an ambiguity flag. `FlashResult`
collects these through `phase_identifications`, `phase_kinds`, and
`phase_regime`. For SRK and PR, the default single-phase rule is the
[Pedersen section 6.6](https://doi.org/10.1201/9780429457418) criterion
\(V/b<1.75\) for liquid and \(V/b>1.75\) for vapor. The differentiable
`volume_to_covolume_ratio` function supports batches; cubic volume
translations are excluded from \(V\) because they are not part of the
repulsive EoS volume.

The 5% band around the divider is marked ambiguous without changing the
deterministic label. For a multiphase model that has no cubic covolume,
`identify_flash_phases` falls back to molar-volume ordering: the least-dense
phase is likely vapor. Near-equal phase volumes remain `unknown`. This fallback
cannot in general distinguish VLE from LLE, and the 1.75 threshold is only
documented for the SRK/PR forms; CPA or a custom cubic-family model requires
its own validation. Identification changes no thermodynamic property or
equilibrium equation.

Phase envelopes can be traced over a supplied temperature grid. Near a
critical point or cricondentherm, `continue_saturation_branch` replaces
temperature by a selected \(\ln K_i\) continuation coordinate so the ordered
branch can pass a temperature turn without being sorted into a spurious
curve. By default, both temperature-grid and fixed-\(\ln K_i\) continuation
use a secant predictor from the previous two converged states; the
temperature-grid path applies two successive-substitution corrections before
Newton and retries a failed prediction with the robust full initializer.
Set `accelerated=False` to select full initialization at every continuation
point for numerical audits. `binary_critical_point` solves the binary second- and third-Gibbs-
derivative conditions with higher-order PyTorch autodiff. These are not a
general multicomponent critical-locus/arclength implementation. The
thermodynamic formulation and continuation context are described by
[Michelsen and Mollerup (2007), chapter 12](https://books.google.com/books?id=qjmeOgAACAAJ).
For binary experimental tables that specify both temperature and pressure,
`binary_vle_point` solves the two component-fugacity equalities for the
coexisting compositions with K-value substitution followed by an autodiff
Newton polish. The caller supplies initial phase compositions to select the
physical branch; convergence and the final fugacity residual are returned
explicitly. Because the fixed-\(T,P\) fugacity equations also possess the
algebraic single-phase solution \(x=y\), the solver does not report that
trivial root as converged VLE. The default minimum phase-composition
separation is \(10^{-6}\); near-critical studies can lower it explicitly but
should use a dedicated critical continuation to locate the endpoint.

## Properties and fitting

`phase_properties` evaluates one homogeneous state. Its chemical potentials
use a zero ideal-gas reference chemical potential at 1 bar unless an explicit
standard-state model is supplied. Enthalpy and entropy are residual properties
unless the caller supplies a consistent ideal-gas standard state. It exposes
component fugacities \(f_i=x_i\phi_iP\) in Pa and dimensionless
\(\ln(f_i/p^\circ)\), where \(p^\circ=1\) bar. It also exposes chemical
potentials in J/mol and the reduced values \(\mu_i/(RT)\). Under the default
zero standard-state convention,

\[
\frac{\mu_i}{RT}=\ln\left(\frac{f_i}{p^\circ}\right).
\]

A literal logarithm of chemical potential is intentionally not defined:
\(\mu_i\) is dimensional, depends on the selected reference state, and may be
negative or zero. The reduced chemical potential is the physically meaningful
dimensionless quantity for log-domain optimization and machine-learning
features.

`state_derivatives` returns first derivatives of \(\phi_i\), \(\ln\phi_i\),
\(f_i\), \(\ln(f_i/p^\circ)\), \(\mu_i\), \(\mu_i/(RT)\), and molar volume
with respect to temperature, pressure, softmax-logit composition coordinates,
\(n-1\) independent mole fractions, and component mole numbers. Temperature
and pressure derivatives hold composition fixed. Mole-number derivatives use
\(n_i=x_i\) mol, a one-mole total basis; for intensive quantities they scale
inversely with another chosen total amount. Composition derivatives require a
strictly positive interior composition; logarithmic derivatives are singular
at a zero mole fraction.

`log_fugacities_tv` and `fugacities_tv` independently evaluate an explicit
\(T,V,\mathbf n\) state from the extensive residual Helmholtz energy,
\[
\ln(f_i/p^\circ)=\ln(n_iRT/(Vp^\circ))
+\partial(A^R/RT)/\partial n_i|_{T,V,n_{j\ne i}}.
\]
They provide a direct TP/TV consistency check for any model that implements
the residual-Helmholtz protocol.

The state-property calculation additionally exposes
the dimensional molar free energies \(a\) and \(g\), their dimensionless
reduced forms \(a/(RT)\) and \(g/(RT)\), and the reference-independent
residual departures \(a^R/(RT)\) and \(g^R/(RT)\). The identities
\(a=g-Pv\), \(g^R/(RT)=\sum_i x_i\ln\phi_i\), and
\(a^R/(RT)=g^R/(RT)-Z+1+\ln Z\) are enforced directly.
Because `ChemicalState` specifies composition but no total amount, the
dimensional results are molar; an extensive \(A\) or \(G\) is obtained by
multiplying by the system mole amount.

`thermal_properties` combines an explicit ideal-gas caloric standard state
with EoS departure terms to return enthalpy, internal energy, entropy,
Helmholtz and Gibbs energies and their reduced forms, \(C_p\), \(C_v\),
Joule-Thomson coefficient, and speed of sound.
Temperature and pressure derivatives follow Pedersen Chapter 8 but are
evaluated by PyTorch autodiff. The phase root is held fixed during
differentiation. The bundled Poling heat-capacity fits cover the named
petroleum components only over their documented temperature ranges; the
function does not silently clip or extrapolate them.

Interaction tensors and Helmholtz amplitudes can be trainable PyTorch
parameters. `fit_parameters` is a small Adam driver; applications remain
responsible for parameter transformations, physical bounds, uncertainty
analysis, train/test splits, and identifiability.

### Cubic-EoS tuning discipline

The BIP workflows summarized by the
[Whitson BIP review](https://wiki.whitson.com/eos/bips/) are implemented as
composable parameterizations rather than a single opaque automatic tuner.
A defensible calibration should:

1. freeze component identity, characterization, alpha convention, volume
   translation, co-volume rule, and units before changing attractive BIPs;
2. begin from zero, a published matrix, or PPR78, and plot that unmodified
   baseline against every measured phase branch;
3. select observables that identify the requested parameter class—use volume
   translation for density only after a sensitivity audit, rather than asking
   an attractive BIP to absorb a volumetric error;
4. tune the smallest sensitive subset first. For petroleum fluids, the
   [Katz-Firoozabadi](https://doi.org/10.2118/6721-PA) lightest-hydrocarbon
   versus heavy-\(C_{n+}\) interaction is a useful hypothesis, not a
   universal constraint;
5. introduce a temperature law only when multiple temperatures identify it,
   scale its coefficients, and inspect the autodiff Jacobian rank and
   condition number;
6. constrain symmetry and zero diagonals by construction, regularize toward
   the correlation or published prior, and reserve entire temperatures or
   fluids as holdouts; and
7. after fitting, rerun stability/flash calculations over a wider
   \(T,P,z\) domain and inspect phase envelopes, root continuity, BIP trends,
   and hydrocarbon K-value ordering. A crossing is an engineering warning,
   not by itself a mathematical proof of impossibility.

`torch-flash` permits physically meaningful negative BIPs while requiring
holdout and extrapolation checks in addition to the calibration residual.
PPR78's trainable unique \(A_{kl},B_{kl}\) parameters make a global group fit
possible; because one group parameter affects many binary pairs, transfer
validation is more important than for a single pair-specific `kij`.

## Transport

`corresponding_states_viscosity` implements the Pedersen corresponding-states
method and maps a mixture to the McCarty/Hanley methane reference
correlation. It is a petroleum-fluid engineering correlation, not a universal
transport model. Dense methane fractional-power terms use mass density in
kg/L, which is numerically equal to g/cm3. The mixture model originates with
[Pedersen et al. (1984)](https://doi.org/10.1016/0009-2509%2884%2987009-8);
the implemented constants and unit convention follow
[Pedersen et al. (2024), section 10.1.1](https://doi.org/10.1201/9780429457418).

The LBC implementation follows Pedersen section 10.1.3 and the original
[Lohrenz-Bray-Clark correlation](https://doi.org/10.2118/915-PA). It consumes
a homogeneous molar density from the caller's EoS. Its critical volumes and
five polynomial coefficients are explicit and differentiable so they can be
fitted, but this does not turn the empirical correlation into a generally
predictive heavy-oil model. Built-in petroleum critical volumes are from
Whitson and Brulé Table A-1B; pseudo-components require their own
characterization.

For complete bibliographic metadata and the distinction between model,
experimental-data, and software-baseline sources, see
[Scientific references and data provenance](references.md).
