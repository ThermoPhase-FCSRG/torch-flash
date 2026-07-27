# Transport Properties

Transport properties are evaluated for a specified homogeneous phase or an
already resolved pair of coexisting phases. They are not equilibrium
conditions: a viscosity or interfacial-tension correlation does not establish
phase stability, choose an EoS root, or perform a flash. The caller must supply
the phase composition, density, pressure, and root required by the selected
model.

`torch-flash` uses SI at the public boundary:

- dynamic viscosity \(\eta\) in Pa s;
- kinematic viscosity \(\nu\) in m²/s;
- thermal conductivity \(\lambda\) in W/(m K);
- surface or interfacial tension \(\sigma\) in N/m; and
- diffusion coefficient \(D\) in m²/s.

The packaged coefficients are identified by
`transport.pedersen-2024`. The common chapter-level source is Pedersen,
Christensen, and Shaikh, *Phase Behavior of Petroleum Reservoir Fluids*, 3rd
ed. (2024), chapter 10,
[doi:10.1201/9780429457418](https://doi.org/10.1201/9780429457418).
Primary equation sources are cited with each model.

## Viscosity

### Dynamic and kinematic viscosity

Dynamic viscosity relates shear stress to velocity gradient. Kinematic
viscosity divides out the homogeneous phase mass density:

\[
\nu = \frac{\eta}{\rho_m}.
\]

`kinematic_viscosity` implements this SI definition and requires
\(\eta\geq0\) and \(\rho_m>0\).

### Pedersen corresponding states

`corresponding_states_viscosity` maps a petroleum mixture to a methane
reference state. In compact form, its reference coordinates are

\[
T_0 =
T\frac{T_{c,0}}{T_{c,m}}\frac{\alpha_0}{\alpha_m},
\qquad
P_0 =
P\frac{P_{c,0}}{P_{c,m}}\frac{\alpha_0}{\alpha_m},
\]

and the mixture viscosity is

\[
\eta_m =
\left(\frac{T_{c,m}}{T_{c,0}}\right)^{-1/6}
\left(\frac{P_{c,m}}{P_{c,0}}\right)^{2/3}
\left(\frac{M_m}{M_0}\right)^{1/2}
\frac{\alpha_m}{\alpha_0}\,
\eta_0(T_0,\rho_0).
\]

Here \(T_{c,m}\) and \(P_{c,m}\) are the Pedersen mixture reducing
properties, \(M_m\) is the correlation's mixture molecular-weight parameter,
and \(\alpha_m/\alpha_0\) is the dense-fluid shape correction. The methane
reference density \(\rho_0\) is solved with the published BWR reference
equation before the Hanley methane viscosity is evaluated. The formulation
originates with Pedersen et al.,
[doi:10.1016/0009-2509(84)87009-8](https://doi.org/10.1016/0009-2509%2884%2987009-8);
the methane reference equation is from Hanley, McCarty, and Haynes,
[doi:10.1016/0011-2275(75)90010-7](https://doi.org/10.1016/0011-2275%2875%2990010-7).

The `phase` argument selects the methane-reference density branch. It does
not identify the physical phase of the supplied mixture.

`heavy_oil_corresponding_states_viscosity` retains the conventional
corresponding-states result above a 75 K methane-reference temperature, uses
the Lindeloff heavy-oil branch below 65 K, and linearly blends the two in
between. Its dimensionless third and fourth CSP factors are explicit tensors.
Changing them is calibration and requires a separately recorded dataset,
objective, bounds, and validation result.

For the low-reference-temperature branch, the representative molecular mass
is

\[
M =
M_n
\left[
\frac{M_w}{\mathrm{Visfac}_3(3\mathrm{rd\ CSP})M_n}
\right]^{\mathrm{Visfac}_4(4\mathrm{th\ CSP})}
\]

when \(M_w/M_n>1.5\), with the corresponding capped-ratio expression from
Eq. 10.34 below that threshold. Here

\[
\mathrm{Visfac}_3=0.2252\,T/M_n+0.9738,
\qquad
\mathrm{Visfac}_4=0.5354\,\mathrm{Visfac}_3-0.1170.
\]

`evaluate_heavy_oil_corresponding_states_profile` is the high-level
phase-aware pressure-profile operation. It solves one bubble point per unique
temperature, retains the feed above the bubble point, batches all sub-bubble
flashes at that temperature, and passes their equilibrium liquid
compositions to the viscosity model. Boundary and flash convergence,
vapor fraction, and log-fugacity residual remain explicit in
`HeavyOilCSPProfile`.

`fit_heavy_oil_csp_factors` calibrates both factors against every supplied
state simultaneously. Its full-batch objective is the mean squared
logarithmic viscosity ratio, and smooth bounds keep both physical factors
positive. Full-batch PyTorch LBFGS with a strong-Wolfe line search is the
default for this two-parameter problem; the result also reports the
log-viscosity sensitivity matrix, singular values, numerical rank, and
condition number. The phase model is prepared separately and is not silently
refitted.

The Heavy Oil 5 study uses the bundled
`characterization.pedersen-heavy-aromatic-2004` C200 correlations and PR78.
The heavy-aromatic characterization is defined by Pedersen, Milter, and
Sørensen,
[doi:10.2118/88364-PA](https://doi.org/10.2118/88364-PA).

### Lohrenz-Bray-Clark

`lbc_viscosity` combines a dilute-gas mixture contribution with a reduced
density polynomial:

\[
\eta =
\eta^\star +
\frac{
\left(a_1+a_2\rho_r+a_3\rho_r^2+a_4\rho_r^3+a_5\rho_r^4\right)^4
-10^{-4}
}{\xi_m},
\qquad
\rho_r=\rho_n\sum_i x_iV_{c,i}.
\]

The numerical polynomial uses the conventional LBC source units internally;
the API accepts molar density in mol/m³ and critical volume in m³/mol and
returns Pa s. Critical volume is therefore a scientific input, not a generic
EoS placeholder. Characterized C7+ cuts can use
`lbc_pseudocomponent_critical_volume`, while measured or fitted critical
volumes can be supplied explicitly.

The model is from Lohrenz, Bray, and Clark,
[doi:10.2118/915-PA](https://doi.org/10.2118/915-PA). Its five coefficients
can be passed as a differentiable tensor, but an untuned LBC calculation is
not automatically predictive for heavy oils.

### Lee gas correlation

`lee_gas_viscosity` implements the Lee-Gonzalez-Eakin natural-gas
correlation:

\[
\eta = 10^{-4}K\exp\left[
X\left(\frac{\rho_m}{62.4}\right)^Y
\right]
\quad\text{cP},
\]

with \(K\), \(X\), and \(Y\) defined from temperature in degrees Rankine and
mixture molar mass in g/mol. The public function converts SI temperature,
mass density, and molar mass into the published units and converts the result
back to Pa s. It is a gas correlation and must not be used for a condensed
liquid merely because the inputs are finite. The source is Lee, Gonzalez, and
Eakin, *J. Petroleum Technology* 18 (1966), 997-1000,
[doi:10.2118/1340-PA](https://doi.org/10.2118/1340-PA).

### One-parameter friction theory

`friction_theory_viscosity` decomposes the cubic-EoS pressure into repulsive
and attractive contributions:

\[
\eta =
\eta_0 +
\kappa_r P_r +
\kappa_a P_a +
\kappa_{rr}P_r^2.
\]

The temperature-dependent friction coefficients are mixed from the published
component reducing relations. The same cubic EoS, mixing rule, interactions,
and phase root determine \(P_r\) and \(P_a\). The bundled parameterization is
defined only for SRK and PR76; PR78 is not silently treated as PR76.

The implementation follows the one-parameter model of
Quiñones-Cisneros, Zéberg-Mikkelsen, and Stenby,
[doi:10.1016/S0378-3812(00)00474-X](https://doi.org/10.1016/S0378-3812%2800%2900474-X).
Its reported scope is nonpolar fluids and n-alkane mixtures.

## Thermal conductivity

### Methane reference correlation

`methane_thermal_conductivity` evaluates the Hanley methane correlation as a
sum of dilute, first-density, dense-fluid, low-temperature, and optional
critical-enhancement terms:

\[
\lambda(T,\rho) =
\lambda^{(0)}(T)
+\lambda^{(1)}(T)\rho
+\Delta\lambda_{\mathrm{dense}}(T,\rho)
+\Delta\lambda_{\mathrm{low}\ T}(T,\rho)
+\Delta\lambda_c(T,\rho).
\]

The public density is mol/L, matching the numerical form of the reference
correlation. `methane_critical_thermal_conductivity_enhancement` exposes the
Vicentini-Missoni enhancement separately and rejects states whose BWR
isothermal compressibility is nonpositive. The defining methane equations are
from Hanley, McCarty, and Haynes,
[doi:10.1016/0011-2275(75)90010-7](https://doi.org/10.1016/0011-2275%2875%2990010-7).

### Christensen-Fredenslund mixture model

`corresponding_states_thermal_conductivity` maps a homogeneous mixture to the
methane reference and applies a conductivity scale:

\[
\lambda_m =
F_\lambda
\left(\lambda_0-\lambda_{\mathrm{int},0}\right)
+\lambda_{\mathrm{int},m}.
\]

\(F_\lambda\) contains the mixture critical-property, effective-mass, and
shape-factor ratios. The two internal-energy terms use ideal-gas heat
capacities supplied by the caller. Following the published mixture model, the
pure-methane critical enhancement is omitted from the mapped reference term.
The model source is Christensen and Fredenslund, *Chemical Engineering
Science* 35 (1980), 871-875; the validation measurements are separate and
originate in
[doi:10.1021/je60083a034](https://doi.org/10.1021/je60083a034).

## Surface and interfacial tension

### Brock-Bird pure-fluid surface tension

For a nonpolar pure fluid, `brock_bird_surface_tension` evaluates

\[
\sigma =
P_c^{2/3}T_c^{1/3}(0.133\alpha-0.281)
(1-T_r)^{11/9},
\]

with the Riedel parameter

\[
\alpha =
0.9076\left[
1+\frac{T_b}{T_c-T_b}\ln\left(\frac{P_c}{1\ {\rm atm}}\right)
\right].
\]

The source equation produces dyn/cm from K and atm; the API converts to N/m.
It requires \(0<T<T_c\) and \(0<T_b<T_c\). The defining paper is Brock and
Bird,
[doi:10.1002/aic.690010208](https://doi.org/10.1002/aic.690010208).

### Weinaug-Katz and Danesh

For already coexisting liquid and vapor phases,
`weinaug_katz_interfacial_tension` forms the phase parachor contrast

\[
\mathcal{P} =
\sum_i P_i\left(\rho_Lx_i-\rho_Vy_i\right),
\qquad
\sigma=\lvert\mathcal{P}\rvert^4
\]

in the conventional mol/cm³ and dyn/cm units before returning SI. Published
pure-component parachors are available through `published_parachors`;
petroleum pseudo-components can use `parachor_from_molar_mass`.

With `danesh_exponent=True`, the fixed fourth power is replaced by the Danesh
gas-condensate density-dependent exponent. Both liquid and vapor mass
densities must then be supplied explicitly. Neither form performs a flash or
infers phase identity.

### Lee-Chien

`lee_chien_interfacial_tension` constructs separate liquid and vapor phase
parachors from critical properties, Riedel parameters, and the Lee-Chien
\(B_i\) coefficients before applying the phase density contrast. The bundled
`published_lee_chien_b` table contains only the eight components reproduced
in Pedersen Table 10.20. Additional compounds require explicit coefficients;
they are never inferred from another component family.

The implementation follows Lee and Chien, SPE-12643-MS (1984),
[doi:10.2118/12643-MS](https://doi.org/10.2118/12643-MS), through Pedersen
chapter 10.

## Liquid n-paraffin diffusion

`hayduk_minhas_n_paraffin_diffusion_coefficient` evaluates the
infinite-dilution normal-paraffin correlation

\[
D_{AB}^{\infty} =
13.3\times10^{-12}
\frac{
T^{1.47}
\eta_B^{\,10.2/V_A-0.791}
}{
V_A^{0.71}
}.
\]

In the defining numerical equation, \(T\) is in K, solvent or bulk-phase
viscosity \(\eta_B\) is in cP, and the diffusing solute's normal-boiling molar
volume \(V_A\) is in cm³/mol. The public API accepts Pa s and m³/mol and
returns m²/s. \(V_A\) is a solute descriptor: it is not the current bulk
specific volume or EoS phase volume.

The source is Hayduk and Minhas,
[doi:10.1002/cjce.5450600213](https://doi.org/10.1002/cjce.5450600213).
The correlation is not a concentrated-mixture Maxwell-Stefan model.

## PyTorch behavior and failure modes

All public transport functions preserve dtype, device, leading batch
dimensions, and gradients through tensor inputs and explicit fitted
parameters. Float64 is the reference precision for dense and near-critical
states. A transport calculation can still be scientifically invalid even
when its tensor result is finite:

- the supplied EoS root may not represent the intended homogeneous phase;
- a methane reference mapping can approach a branch or critical
  ill-conditioning;
- a required critical volume, parachor, or Lee-Chien coefficient may be
  unavailable;
- heavy-oil factors or LBC coefficients may be unidentified for the fluid;
  and
- an empirical correlation may be outside its fluid-family or state range.

Nonphysical inputs and nonpositive results raise explicit errors. The
[transport validation report](../validation-reports/36-transport-properties.md)
shows the experimental agreement and model-form limitations for the selected
reservoir-gas, heavy-oil, and CO2/methane cases.
