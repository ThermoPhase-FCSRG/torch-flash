# Component and model parameter databases

`torch-flash` separates three quantities that are easy to conflate:

1. shared component identity and SI properties;
2. coefficients belonging to a particular published model parameterization;
3. tensors supplied explicitly by an application or fitted with PyTorch.

This distinction matters scientifically. For example, the reducing
temperature stored in GERG-2008 is part of that Helmholtz model and is not
silently replaced by the general critical temperature used by PR78.

## Bundled data

The component inventory is
[`src/torch_flash/data/components/default.yaml`](https://github.com/ThermoPhase-FCSRG/torch-flash/blob/main/src/torch_flash/data/components/default.yaml).
Its canonical names are used by every public named constructor. Aliases such
as `CO2`, `H2`, `CH4`, and `NH3` resolve to those canonical names. All values
declare SI units; `null` means that a property is unavailable.

Published model sets are separate YAML documents under
[`src/torch_flash/data/models`](https://github.com/ThermoPhase-FCSRG/torch-flash/tree/main/src/torch_flash/data/models):

| Identifier | Contents |
|---|---|
| `cubic.srk-1972` | SRK constants and alpha coefficients |
| `cubic.pr-1976` | original PR constants and alpha coefficients |
| `cubic.pr-1978` | PR high-acentric-factor extension |
| `group-contribution.ppr78-jaubert-mutelet-2004` | original six-group predictive PR78 BIP correlation |
| `cpa.folas-2005` | Folas Table 1 SRK-CPA pure parameters |
| `cpa.oliveira-2007-hydrocarbon-water` | alkane/aromatic-water pure, BIP, and modified-CR1 parameters |
| `cpa.yan-2009-reservoir-fluids` | temperature-dependent light-hydrocarbon/water BIPs and CPA heavy-cut correlations |
| `characterization.pedersen-2024` | logarithmic split, density, SRK/PR property, and lumping coefficients |
| `characterization.whitson-2000` | shifted-gamma split and grouping defaults |
| `activity.unifac-original-public-2026` | original VLE-UNIFAC: 113 subgroups, 54 main groups, and 1,270 directed interactions |
| `activity.hv-nrtl-pedersen-2024-propane-water` | Pedersen Table 16.3 HV-NRTL parameters |
| `binary-interaction.whitson-2000` | Whitson Table A-3 PR/SRK BIPs |
| `binary-interaction.pedersen-2024` | Pedersen Table 4.2 PR/SRK BIPs |
| `binary-interaction.segovia-2017-methane-n-decane` | Segovia PR78 `kij` plus a torch-flash `lij` density fit with a disjoint validation isotherm |
| `volume-translation.pedersen-2024` | Rackett-based SRK/PR light-component and ASTM-anchored C7+ translation coefficients |
| `volume-translation.whitson-2000` | Whitson Tables 4.2-4.3 pure-component and heavy-family shift factors |
| `multiparameter.gerg-2008` | complete 21-component GERG-2008 inventory |
| `multiparameter.gerg-2008-hydrogen-2021` | five-component H2-tailored GERG model for CH4, N2, CO, CO2, and normal H2 |
| `multiparameter.eos-cg-2021` | complete 16-component EOS-CG-2021 inventory |
| `standard-state.poling-2001` | ideal-gas heat-capacity polynomials |

The primary sources include
[Soave (1972)](https://doi.org/10.1016/0009-2509%2872%2980096-4),
[Peng and Robinson (1976)](https://doi.org/10.1021/i160057a011),
[Jaubert and Mutelet (2004)](https://doi.org/10.1016/j.fluid.2004.06.059),
[Folas et al. (2005)](https://doi.org/10.1021/ie048832j),
[Oliveira et al. (2007)](https://doi.org/10.1016/j.fluid.2007.05.023),
[Yan et al. (2009)](https://doi.org/10.1016/j.fluid.2008.10.007),
[Whitson and Brulé (2000)](https://books.google.com/books?id=Z4cQAQAAMAAJ),
[Pedersen et al. (2024)](https://doi.org/10.1201/9780429457418),
[Kunz and Wagner (2012)](https://doi.org/10.1021/je300655b),
[Beckmüller et al. (2021)](https://doi.org/10.1063/5.0040533), and
[Neumann et al. (2023)](https://doi.org/10.1007/s10765-023-03263-6).
Every YAML file carries its own source metadata.

Machine-readable JSON Schema 2020-12 documents are packaged in
[`src/torch_flash/data/schemas`](https://github.com/ThermoPhase-FCSRG/torch-flash/tree/main/src/torch_flash/data/schemas).
Runtime
validation does not require the optional `jsonschema` package; the loaders
enforce the same common envelope and each constructor validates its
model-specific payload.

Inspect the registry without constructing a model:

```python
from torch_flash import available_parameter_sets, load_model_parameters

print(available_parameter_sets(model_kind="multiparameter"))
parameters = load_model_parameters("gerg2008")  # aliases are accepted
print(parameters.identifier, parameters.version, parameters.units)
```

Parsed files are cached after their first read. Model instances are not shared:
each constructor creates independent tensors with the dtype/device selected by
the [runtime policy](runtime.md), or by explicit constructor overrides, and
with its requested `trainable` state.

## Construct from bundled or custom YAML

The same constructor accepts a bundled identifier or a custom path:

```python
import torch

from torch_flash import component_set, cubic_eos, multiparameter_eos

components = component_set(("CO2", "CH4"), dtype=torch.float64)
pr = cubic_eos(components, "cubic.pr-1978")
gerg = multiparameter_eos("multiparameter.gerg-2008", ("CO2", "CH4"))

# Files using the documented schemas are handled identically.
custom_components = component_set(
    ("species_a", "species_b"),
    database="project-components.yaml",
)
custom_pr = cubic_eos(custom_components, "project-pr-fit.yaml")
```

A model YAML document has this envelope:

```yaml
format: torch-flash-model-parameters
schema_version: 1
id: cubic.project-pr-fit
model_kind: cubic
model: Project-PR
version: "fit-2026-07"
description: Parameters fitted to the project's VLE measurements.
units:
  omega_a: dimensionless
  omega_b: dimensionless
references:
  - citation: Internal experiment campaign 2026
parameters:
  omega_a: 0.4572355289213822
  omega_b: 0.07779607390388846
  delta1: 2.414213562373095
  delta2: -0.41421356237309515
  alpha:
    kind: pr78
    switch_acentric_factor: 0.491
    low:
      coefficients: [0.37464, 1.54226, -0.26992]
    high:
      coefficients: [0.379642, 1.48503, -0.164423, 0.016666]
```

The component schema similarly requires
`format: torch-flash-component-database`, `schema_version: 1`,
`unit_system: SI`, the complete unit map, and a `components` mapping. Load it
directly when validation without model construction is useful:

```python
from torch_flash import load_component_database

database = load_component_database("project-components.yaml")
print(database.names)
```

## Explicit in-memory and fitting APIs

YAML is optional. Explicit typed objects and tensors remain first-class:

```python
from torch_flash import ModelParameterSet, component_set, cubic_eos

fit = ModelParameterSet(
    identifier="cubic.current-fit",
    model_kind="cubic",
    model="Project-PR",
    version="optimizer-step-500",
    parameters={
        "omega_a": 0.4572355289213822,
        "omega_b": 0.07779607390388846,
        "delta1": 2.414213562373095,
        "delta2": -0.41421356237309515,
        "alpha": {
            "kind": "pr78",
            "switch_acentric_factor": 0.491,
            "low": {"coefficients": [0.37464, 1.54226, -0.26992]},
            "high": {
                "coefficients": [0.379642, 1.48503, -0.164423, 0.016666]
            },
        },
    },
)
model = cubic_eos(component_set(("methane", "n_butane")), fit)
```

`CPAEOS`, `NRTL`, `HuronVidalNRTL`, `AnchoredHuronVidalNRTL`, `Wilson`, `CubicEOS`, and
`MultiparameterEOS` also retain their explicit tensor constructors. This is the
preferred route while parameters are PyTorch `nn.Parameter` objects being
optimized. Serialize a finalized, reviewed fit to YAML only after recording
its units, data provenance, objective, and version.

For HV-NRTL, prefer `AnchoredHuronVidalNRTL` during calibration. It represents
the same \(\tau_{ij}=A_{ij}/T+B_{ij}\) function by its values at two selected
temperatures, reducing the numerical correlation between \(A_{ij}\) and
\(B_{ij}\). With `trainable_nonrandomness=True`, a sigmoid keeps the symmetric
non-randomness matrix inside explicit bounds. After optimization, `freeze()`
returns a standard `HuronVidalNRTL` instance whose `energy_over_r`,
`temperature_coefficient`, and `nonrandomness` tensors can be reviewed and
serialized to the documented YAML schema. The executed n-butane/water
notebook demonstrates the full workflow and complete-temperature holdouts.

`clear_parameter_caches()` and `clear_component_caches()` are available for a
long-running development process that intentionally rewrites a custom file.
Production code should treat parameter documents as immutable.

## Common model-document standard

Every model document has the following required fields:

| Field | Standard |
|---|---|
| `format` | exactly `torch-flash-model-parameters` |
| `schema_version` | integer `1`; changes only for an incompatible document format |
| `id` | stable lowercase identifier, conventionally `<kind>.<source-version>` |
| `model_kind` | `cubic`, `cpa`, `activity`, `binary_interaction`, `group_contribution`, `characterization`, `volume_translation`, `multiparameter`, `standard_state`, or `transport` |
| `model` | scientific model/family name, independent of the file name |
| `version` | publication or fit version; changing coefficients requires a new version |
| `units` | explicit unit for every dimensional parameter family |
| `references` | list of mappings such as `citation`, `doi`, `url`, or data-set identifier |
| `parameters` | model-specific mapping described below |

`description` is optional but strongly recommended. `schema_version` describes
the data format, while `version` describes the scientific parameterization.
They must not be used interchangeably.

YAML is loaded with `yaml.safe_load`; Python-specific YAML tags and executable
constructors are not supported. Strings that resemble dates or ambiguous
chemical tokens should be quoted. Missing values use YAML `null`, never a
numeric sentinel. Model files must already use the units they declare:
`torch-flash` does not silently infer that a pressure is in bar or convert a
legacy table during loading.

## Component database standard

The component document uses:

| Field | Requirement |
|---|---|
| `format` | `torch-flash-component-database` |
| `schema_version` | `1` |
| `id`, `revision` | stable database identity and scientific revision |
| `unit_system` | exactly `SI` |
| `units` | the complete fixed unit map shown below |
| `references` | provenance mappings |
| `components` | mapping keyed by canonical component name |

The fixed unit map is:

```yaml
critical_temperature: K
critical_pressure: Pa
acentric_factor: dimensionless
molar_mass: kg mol^-1
critical_volume: m^3 mol^-1
critical_density: mol m^-3
```

Canonical names are lowercase snake case, for example `carbon_dioxide`,
`n_butane`, and `hydrogen_sulfide`. Each component record contains `aliases`
and all six property keys. `critical_temperature`, `critical_pressure`, and
`molar_mass` must be positive. The acentric factor, critical volume, or
critical density may be `null` when they were not established for the shared
database.

Aliases are normalized only for lookup; the canonical key is always stored in
`ComponentSet.names`. Alias/name collisions are rejected. A cubic constructor
also rejects a selected component whose acentric factor is `null`, rather than
using zero.

Custom component databases can be passed as files or constructed in memory:

```python
from torch_flash import Component, ComponentDatabase, component_set

custom_database = ComponentDatabase(
    identifier="components.project",
    revision="fit-3",
    components=(
        Component(
            name="species_a",
            critical_temperature=410.0,
            critical_pressure=4.8e6,
            acentric_factor=0.21,
            molar_mass=0.052,
            critical_volume=2.1e-4,
            aliases=("a",),
        ),
    ),
    units={
        "critical_temperature": "K",
        "critical_pressure": "Pa",
        "acentric_factor": "dimensionless",
        "molar_mass": "kg mol^-1",
        "critical_volume": "m^3 mol^-1",
        "critical_density": "mol m^-3",
    },
)
components = component_set(("a",), database=custom_database)
```

## Cubic parameter payload

`model_kind: cubic` requires:

```yaml
parameters:
  omega_a: 0.4572355289213822
  omega_b: 0.07779607390388846
  delta1: 2.414213562373095
  delta2: -0.41421356237309515
  alpha:
    kind: pr78
    switch_acentric_factor: 0.491
    low:
      coefficients: [0.37464, 1.54226, -0.26992]
    high:
      coefficients: [0.379642, 1.48503, -0.164423, 0.016666]
```

`omega_a`, `omega_b`, `delta1`, and `delta2` are dimensionless generalized
cubic constants. Alpha coefficients are ordered by increasing power of the
acentric factor. `srk` and `pr76` use the three `low` coefficients. `pr78`
also requires four `high` coefficients and a switching acentric factor.
Full-precision critical-point constants are retained in YAML.

The applied component translation remains a cubic-constructor argument. It may
be an explicit additive tensor or a `VolumeTranslation` returned by one of the
named/caller-supplied factories documented below. Cubic constructors accept
either constant `kij`, or both `kij_a` and `kij_b` for
`kij(T) = kij_a + kij_b/T`; `kij_a` is dimensionless and `kij_b` is in
kelvin. The alternatives are mutually exclusive. An optional dimensionless
`lij` matrix defines
\(b_{ij}=(b_i+b_j)(1-l_{ij})/2\); omitting it gives the conventional linear
co-volume rule. `trainable=True` controls the attractive interaction tensors,
while `trainable_lij=True` independently creates a trainable co-volume matrix
from the supplied value or from zero. Published reusable constant interactions
use the separate binary-interaction format below; fitted temperature-law or
co-volume matrices should retain their form, SI units, data domain, objective,
and provenance in custom model metadata as demonstrated by notebooks 19, 20,
and 25.

## PPR78 and E-PPR78 group-contribution payload

`model_kind: group_contribution` with `model: PPR78` or `model: E-PPR78`
stores the universal group parameters separately from the PR78 pure-component
constants:

```yaml
model_kind: group_contribution
model: PPR78
units:
  A: Pa
  B: Pa
  group_fraction: dimensionless
  reference_temperature: K
parameters:
  reference_temperature: 298.15
  groups: [CH3, CH2, CH, C, CH4, C2H6]
  interactions:
    "CH3|CH2": {A: 74810000.0, B: 165700000.0}
    # Every other unordered group pair is also explicit.
  component_groups:
    propane: {CH3: 2, CH2: 1}
    n_butane: {CH3: 2, CH2: 2}
```

`A` and `B` are pressures in pascals. A key `first|second` denotes one
unordered pair; diagonal terms are zero by definition, reversed duplicates
are rejected, and all \(N_g(N_g-1)/2\) off-diagonal pairs must be accounted
for. A complete inventory supplies every pair under `interactions`. An
incomplete published inventory divides them exactly between `interactions`
and `unavailable_interactions`; an unavailable pair is never interpreted as
a fitted zero.
`component_groups` contains nonnegative structural counts. The constructor
normalizes each component row to fractions
\(\alpha_{ik}=N_{ik}/\sum_kN_{ik}\); it never guesses a missing
decomposition or treats a missing interaction as zero.

The default PPR78 document is exactly the original six-group parameterization of
[Jaubert and Mutelet (2004)](https://doi.org/10.1016/j.fluid.2004.06.059),
not a later extension. The separate H2/N2/H2O document preserves the exact
active submatrix from Qian et al.'s 2013 hydrogen and water extensions.

The global E-PPR78 document reproduces all 40 groups, 356 available pairs, and
424 unavailable pairs in Table S4 of
[Jaubert et al. (2022)](https://doi.org/10.1016/j.fluid.2022.113456).
It includes the principal CCS groups studied by
[Xu et al. (2017)](https://doi.org/10.1016/j.ijggc.2016.11.015), but uses the
later 2022 global coefficient revision. The 2022 article and supplement are
CC BY 4.0; the separately copyrighted 2017 table is not duplicated in the
runtime database.

```python
import torch

from torch_flash import (
    component_set,
    enhanced_predictive_peng_robinson_1978,
    ppr78_group_contribution_parameters,
    predictive_peng_robinson_1978,
)

components = component_set(("methane", "n_decane"), dtype=torch.float64)
model = predictive_peng_robinson_1978(components)

# Override only structural counts while retaining the selected universal set.
custom = predictive_peng_robinson_1978(
    components,
    group_counts={
        "methane": {"CH4": 1},
        "n_decane": {"CH3": 2, "CH2": 8},
    },
    trainable=True,
)

# A custom YAML path or in-memory ModelParameterSet uses the same validation.
parameters = ppr78_group_contribution_parameters(
    components,
    "project-ppr78.yaml",
)

# Global E-PPR78 for a hydrogen-bearing CCS stream.
ccs_components = component_set(
    ("carbon_dioxide", "hydrogen", "water"),
    dtype=torch.float64,
)
ccs_model = enhanced_predictive_peng_robinson_1978(ccs_components)
```

`trainable=True` creates 30 independent parameters for the six-group set:
the 15 unique \(A_{kl}\) and 15 unique \(B_{kl}\) values. Symmetry and zero
diagonals are reconstructed on every evaluation. A fit of universal group
parameters can affect every component pair using those groups, so transfer
systems and temperature holdouts must be checked before publishing a revised
YAML parameterization.

For an E-PPR78 component selection, only active groups are materialized after
their pairwise availability has been verified. Thus the three pure groups in
the CCS example create three independent A/B pairs rather than 780 candidate
pairs. A request containing an explicitly unavailable active pair raises
`ParameterDatabaseError` with the group names.

## Volume-translation parameter payload

`model_kind: volume_translation` stores reusable correlation coefficients, not
the already evaluated component shifts. This preserves component identity,
dtype, and device until the factory is called. The bundled Pedersen document
uses:

```yaml
model_kind: volume_translation
model: Pedersen-Peneloux
units:
  rackett_intercept: dimensionless
  rackett_acentric_slope: dimensionless
  correlation_scale: dimensionless
  correlation_target: dimensionless
  reference_temperature: K
  target_temperature: K
  astm_density_constant: kg^2 m^-6 K^-1
parameters:
  rackett:
    intercept: 0.29056
    acentric_slope: -0.08775
  correlations:
    srk: {scale: 0.40768, target: 0.29441}
    pr: {scale: 0.50033, target: 0.25969}
  temperature_dependent:
    reference_temperature: 288.15
    target_temperature: 353.15
    astm_density_constant: 613.9723
    nonlinear_factor: 0.8
```

The Whitson document has an `eos` mapping. Each `pr`/`srk` record contains a
dimensionless `covolume_factor` and a `pure_shift_factors` mapping keyed by
canonical component name. Its `heavy_families` mapping contains `A0` and `A1`
for `paraffin`, `naphthene`, and `aromatic` families; the correlation evaluates
\(s=1-A_0/M^{A_1}\) with \(M\) in g/mol.

Both sources use the published convention
\(v=v_{\mathrm{EOS}}-\sum_i x_i c_i\). `VolumeTranslation.reference_shift`
uses m3/mol and stores \(d_i=-c_i\); `temperature_slope` uses
m3/(mol K). A custom database can be supplied by path or as a
`ModelParameterSet`:

```python
from torch_flash import (
    component_set,
    pedersen_peneloux_translation,
    peng_robinson_1978,
    whitson_volume_translation,
)

components = component_set(("methane", "n_decane"))
published = whitson_volume_translation(
    components,
    "pr",
    source="project-whitson-variant.yaml",
)
model = peng_robinson_1978(components, volume_translation=published)

# The same API accepts a custom Pedersen-format parameter document.
light = pedersen_peneloux_translation(
    components,
    "pr",
    source="project-pedersen-variant.yaml",
)
```

For a direct density anchor, `density_matched_translation()` accepts
parent-EoS molar volume in m3/mol, molar mass in kg/mol, and density in kg/m3.
`pedersen_temperature_dependent_translation()` uses the same component order
as its untranslated parent cubic model and requires one 288.15 K reference
density per component. Passing an already translated parent is rejected to
prevent an accidental double correction. It evaluates new tensors once during
model construction; cached YAML parsing does not add per-state disk I/O.

## CPA parameter payload

`model_kind: cpa` contains a `components` mapping. Each record requires:

| Key | SI definition |
|---|---|
| `critical_temperature` | K, as used by this CPA fit |
| `a0` | Pa m6 mol-2 |
| `b` | m3 mol-1 |
| `c1` | dimensionless alpha coefficient |
| `association_energy` | J mol-1 |
| `association_volume` | dimensionless |
| `scheme` | `none`, `1A`, `1B`, `2B`, `3B`, or `4C` |

`default_combining_rule` is `CR1` or `ECR`. A constructor argument may
override it. `binary_interactions.kind` is either `constant` with numeric pair
values or `a_plus_b_over_temperature` with pair mappings `{a, b}` defining
\(k_{ij}=a+b/T\). `cross_association.pairs` may provide explicit
`association_energy` in J/mol and dimensionless `association_volume` for a
modified-CR1 pair. Missing pairs use the selected combining rule; explicit
cross-pair tensors can also be supplied through the Python constructor. CPA
constructors also accept physical-term `lij` and `trainable_lij`; this changes
the SRK mixture co-volume but does not redefine the association-energy or
association-volume combining rules.

`heavy_end_correlations` is optional and is consumed only by
`cpa_pseudocomponent()` or `cpa_components_from_cuts()`. It maps an already
split cut's normal boiling temperature and specific gravity to a
non-associating CPA monomer. It does not decide the upstream SCN distribution
or lump boundaries. A mixture-specific CPA interaction remains an explicit
tensor so it can be fitted without rewriting the published file. Users fitting
new pure species may pass `CPAComponent` objects directly to `CPAEOS`.

## Heavy-end characterization payload

`model_kind: characterization` stores coefficients for splitting, property
estimation, or lumping independently of a phase model. The two bundled
payloads intentionally have different structures because they represent
different published methods:

- `characterization.pedersen-2024` contains `plus_split`, the
  \(M_N=14CN-4\) g/mol relation, the logarithmic density rule and anchor
  default, EoS-specific `cubic_properties` tables for SRK and PR, and the
  weight-based lumping convention.
- `characterization.pedersen-heavy-aromatic-2004` extends the logarithmic
  split through C200 and contains the separately published SRK and PR
  critical-property correlations for highly aromatic and HT/HP fluids. Above
  1000 g/mol, the cubic alpha parameter follows the source's inverse-mass
  continuation.
- `characterization.whitson-2000` contains the shifted-gamma shape default and
  range, the recommended \(\eta(\alpha)\) relation, the 0.014 kg/mol
  molecular-weight boundary increment used by GAMSPL, and grouping defaults.

All public inputs use SI: mole fractions are dimensionless, molar masses are
kg/mol, and densities are kg/m³. `pedersen_logarithmic_split()` and
`whitson_gamma_split()` return `SCNDistribution`; the latter places the entire
remaining gamma tail in the final requested bin so both zeroth and first
moments are preserved. `pedersen_density_split()` enforces the bulk ideal
volume balance. `equal_weight_lump()` operates on any `SCNDistribution`;
`pedersen_cubic_properties()` is a deliberately model-specific SRK/PR adapter.

The YAML document may be replaced by a custom path or an in-memory
`ModelParameterSet` using the same function arguments as other model kinds.
Measured extended cuts should be preferred to an inferred distribution.
`first_carbon_number`, `max_carbon_number`, gamma `shape`, density anchor, and
lump count remain explicit API choices because changing them changes the
scientific characterization.

## Activity/excess-Gibbs parameter payloads

Every activity payload starts with an ordered `components` list. Rows and
columns of every matrix follow this order. `activity_model(..., names=...)`
may request a different order; all vectors and matrices are permuted together.

Supported payloads are:

- `model: NRTL`: `interaction` in J mol-1 and dimensionless
  `nonrandomness`. `interaction[i][j]` follows the
  \(g_{ij}-g_{jj}\) convention used by `NRTL`.
- `model: HV-NRTL`: `energy_over_r` in K,
  dimensionless `temperature_coefficient`, and dimensionless
  `nonrandomness`. The implemented convention is
  \(\tau_{ji}=A_{ji}/T+B_{ji}\). Supply either `covolumes` in m3 mol-1 or
  `covolume_cubic_parameter_set`, whose `omega_b` and the shared component
  critical constants define the covolumes.
- `model: Wilson`: `interaction` in J mol-1 and `molar_volumes` in
  m3 mol-1.

Ordinary low-pressure NRTL parameters and HV infinite-pressure parameters are
not interchangeable even though both use local-composition matrices.

### Original UNIFAC payload

`model: original-UNIFAC` uses a different payload because components are
assembled from functional subgroups:

```yaml
format: torch-flash-model-parameters
schema_version: 1
id: activity.project-original-unifac
model_kind: activity
model: original-UNIFAC
version: "reviewed-1"
units:
  relative_volume: dimensionless
  relative_surface_area: dimensionless
  interaction: K
  coordination_number: dimensionless
references:
  - citation: Fredenslund, Jones, and Prausnitz (1975)
    doi: 10.1002/aic.690210607
parameters:
  variant: original-vle
  coordination_number: 10.0
  subgroups:
    alkyl_ch3:
      number: 1
      name: CH3
      main_group: 1
      main_group_name: CH2
      relative_volume: 0.9011
      relative_surface_area: 0.848
    alkyl_ch2:
      number: 2
      name: CH2
      main_group: 1
      main_group_name: CH2
      relative_volume: 0.6744
      relative_surface_area: 0.540
  interactions: []
  component_assignments:
    ethane: {1: 2}
```

The abbreviated example contains one main group, so its directed interaction
list is empty. With two or more selected main groups, `interactions` contains
rows `[main_group_i, main_group_j, a_ij]`. Direction matters:
\(a_{ij}\ne a_{ji}\) in general. Every off-diagonal directed pair required by
the selected components must be present; an absent pair raises
`ParameterDatabaseError` and is never interpreted as zero.

Subgroup mapping keys are stable database identifiers. A component assignment
may use one of those keys, a published subgroup `number`, or an unambiguous
subgroup `name`. Counts must be nonnegative with at least one positive count
per component. Integer counts are the normal molecular interpretation;
fractional counts are accepted for explicit pseudo-component workflows but
must be scientifically justified by the caller. `component_assignments` is
optional convenience data and is not intended to be a structure database:

```python
from torch_flash import unifac_model

manual = unifac_model(
    "activity.project-original-unifac.yaml",
    group_assignments=(
        {1: 2, 2: 4},          # n-hexane
        {1: 1, 2: 1, 18: 1},  # 2-butanone
    ),
)
```

The bundled `activity.unifac-original-public-2026` table pins the public
original-UNIFAC inventory accessed on 2026-07-24. It was cross-checked against
the MIT-licensed `thermo` 0.6.0 table and the
[DDBST published table](https://www.ddbst.com/published-parameters-unifac.html).
It uses the original-UNIFAC subgroup identities and includes
\(Q_{\mathrm{CH_3CO}}=1.488\). Dortmund, PSRK, and UEA/AIM are separate
parameterizations and are not merged into this model.

`unifac_model(..., trainable=True)` makes the selected directed interaction
matrix an `nn.Parameter`; subgroup geometry and counts remain buffers.
Composition-, temperature-, and parameter-gradients then use ordinary
PyTorch autodiff. Parsed YAML documents are cached by the common database
loader, while every constructor returns independent tensors.

Automatic group assignment is intentionally optional:

```bash
pip install "torch-flash[groups]"
# or
pixi install -e groups
```

```python
from torch_flash import unifac_groups_from_identifiers

assignments = unifac_groups_from_identifiers(
    ("CCO", "c1ccccc1"),
    identifier_type="smiles",
)
```

This adapter delegates to MIT-licensed
[`ugropy`](https://github.com/ipqa-research/ugropy). SMILES is recommended:
name lookup may contact PubChem. Fragmentation is discrete and is not
differentiable; review and version the returned assignments before fitting,
regression testing, or safety-critical use.

## Binary-interaction payload

`model_kind: binary_interaction` keeps BIPs separate from the cubic equation
definition. The petroleum schema contains:

```yaml
parameters:
  hydrocarbons: [methane, ethane, propane, ...]
  nonhydrocarbons: [nitrogen, carbon_dioxide, hydrogen_sulfide]
  aggregate_from: n_heptane
  eos:
    PR:
      hydrocarbon_rows:
        nitrogen: [...]
      nonhydrocarbon_pairs:
        nitrogen|carbon_dioxide: 0.017
    SRK: ...
```

The source order fixes each row's meaning. `aggregate_from` documents where a
published C7+ value begins to be reused for represented heavier normal
alkanes. Hydrocarbon/hydrocarbon entries are zero for these two particular
tables; that is a property of the data files, not an assumption of
`QuadraticMixing`. Pair keys are order-independent and the constructed tensor
is symmetric with an exactly zero diagonal.

The general `cubic-vdw-one-fluid` schema stores both attraction and co-volume
interactions without embedding them in a cubic pure-parameter file:

```yaml
format: torch-flash-model-parameters
schema_version: 1
id: binary-interaction.project-fit
model_kind: binary_interaction
model: cubic-vdw-one-fluid
version: "fit-1"
units:
  kij: dimensionless
  lij: dimensionless
references:
  - citation: Project experimental dataset and fitting protocol
parameters:
  component_order: [methane, n_decane]
  defaults: {kij: 0.0, lij: 0.0}
  pairs:
    methane|n_decane: {kij: 0.0409, lij: 0.0475}
```

Names in `component_order` are canonical package names and cannot repeat.
Each unordered pair may occur once; self-pairs are rejected. Missing pair
fields use `defaults`, and missing defaults are zero. All values are finite
and dimensionless. `cubic_interaction_parameters()` validates a bundled YAML,
custom path, or in-memory `ModelParameterSet`, reorders it to a requested
`ComponentSet`, and returns independent tensors:

```python
from torch_flash import (
    component_set,
    cubic_interaction_parameters,
    peng_robinson_1978,
)

components = component_set(("methane", "n_decane"))
interaction = cubic_interaction_parameters(components, "project-fit.yaml")
model = peng_robinson_1978(
    components,
    kij=interaction.kij,
    lij=interaction.lij,
)
```

The bundled Segovia file is labeled as a local density fit. It uses
`kij=0.0409` and an `lij=0.047567430964013162` estimated by notebook 25 from
30 non-pure 80 MPa states. Its 33 non-pure 323.15 K states are untouched
validation data. Neither that `lij` nor any fitted BIP should be transferred
to a new system or temperature/pressure domain without validation.

## Multiparameter payload

`model_kind: multiparameter` is the native GERG/EOS-CG coefficient inventory.
GERG is constructed with the published multi-fluid approximation, while
EOS-CG is classified by its authors as a multiparameter mixture model. The
canonical parameter kind names their shared Helmholtz EOS abstraction. Its
payload requires:

- `component_order`: canonical public names and the default model order;
- `gas_constant`: the model's fitted gas constant in J mol-1 K-1;
- `components`: model-specific critical/reducing constants, molar mass, ideal
  Helmholtz blocks, and pure residual Helmholtz terms;
- `pairs`: exactly \(N(N-1)/2\) binary records containing `first`, `second`,
  temperature/volume reducing parameters, departure scale, and optional
  departure terms;
- `reference`: the model-level bibliographic source string.

Supported residual term block types are `ResidualHelmholtzPower`,
`ResidualHelmholtzGaussian`, `ResidualHelmholtzGERG2008`, and
`Gaussian+Exponential`. Supported ideal blocks are
`IdealGasHelmholtzLead`, `IdealGasHelmholtzLogTau`,
`IdealGasHelmholtzPower`, `IdealGasHelmholtzPlanckEinstein`,
`IdealGasHelmholtzPlanckEinsteinFunctionT`,
`IdealGasHelmholtzCP0PolyT`, and
`IdealGasHelmholtzEnthalpyEntropyOffset`.

Arrays within one term block must have identical lengths. Pair direction is
defined by `first` and `second`; beta reducing parameters are inverted when a
requested component order reverses that direction, while gamma and departure
scales remain symmetric. Critical constants here are model parameters and
deliberately do not inherit from the general component database.

`multiparameter_eos()` dispatches compatible GERG-family and EOS-CG-family files.
The stricter `gerg2008()` and `eoscg2021()` wrappers reject a file with a
different declared model version.

### Custom GERG-form binary departure terms

A custom GERG-form regression may replace one binary `pairs` record in a copy
of the GERG parameter payload. The `departure_scale` and coefficient
amplitudes are multiplicatively degenerate, so fix one scale convention
before fitting. Notebook 20 fixes `departure_scale: 1.0` and estimates the
`n` amplitudes:

```yaml
carbondioxide|water:
  first: carbon_dioxide
  second: water
  beta_temperature: 1.026187
  gamma_temperature: 0.798725
  beta_volume: 0.868702
  gamma_volume: 1.253512
  departure_scale: 1.0
  departure:
    type: ResidualHelmholtzGERG2008
    n: [0.015750, 0.002791, 0.000159, 0.009612, -0.014064,
        -0.005002, -0.008033, -0.004769, -0.003736, -0.000308]
    d: [1, 1, 1, 2, 2, 3, 3, 4, 4, 4]
    t: [1, 1.55, 1.7, 0.25, 1.35, 0, 1.25, 0, 0.7, 5.4]
    eta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    epsilon: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    beta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    gamma: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

All arrays in the departure block must have equal lengths. These example
values define a VLE-only CO2-H2O fit and are not published GERG-2008
coefficients. Assign a distinct `id` and `version` and preserve the
description, source DOIs, fitted temperature/composition range, objective, and
validation metrics in any saved model document. The fit is locally full rank
but has condition number \(5.91\times10^7\); more digits do not imply
independently identifiable coefficients.

## Standard-state payload

`model_kind: standard_state` with
`model: ideal-gas-cp-polynomial` stores:

```yaml
parameters:
  default_reference_temperature: 273.15
  components:
    methane:
      coefficients: [a0, a1, a2, a3, a4]
      temperature_range: [50.0, 1000.0]
      reference_enthalpy: 0.0   # optional
      reference_entropy: 0.0    # optional
```

The coefficients define \(C_p^\circ/R=\sum_k a_k T^k\); coefficient units
therefore depend on polynomial order. `temperature_range` records the source
fit range and is metadata rather than an automatic clipping rule. Reference
enthalpy and entropy default to zero, and a constructor argument may choose a
different positive reference temperature.

## Transport payload

`model_kind: transport` stores correlation constants with their
model-specific units. The bundled `transport.pedersen-2024` document contains
the chapter 10 methane-reference, viscosity, thermal-conductivity,
surface/interfacial-tension, and n-paraffin-diffusion constants used by the
public transport functions.

Fit-ready quantities remain explicit function arguments. For example,
`lbc_viscosity(..., coefficients=...)` accepts the five LBC coefficients, and
`fit_heavy_oil_csp_factors()` returns the calibrated third and fourth CSP
factors without rewriting the immutable source parameter document. Persisting
a calibration requires a distinct identifier and the dataset, objective,
bounds, split, and fit diagnostics.

## Caching and custom-database lifecycle

Bundled identifiers are resolved through the packaged index without probing
the working directory. Bundled YAML is read and parsed once per process.
Custom YAML is cached by its absolute path; after the first call, repeated
loads return the same immutable `ModelParameterSet` or `ComponentDatabase`
without another file read.

Constructors consume those immutable mappings directly but create new PyTorch
tensors for each model. This avoids disk/parsing/deep-copy overhead without
sharing trainable tensors, devices, gradients, or mutable model state.

During deliberate custom-file editing, call:

```python
from torch_flash import clear_component_caches, clear_parameter_caches

clear_component_caches()
clear_parameter_caches()
```

Caching assumes parameter files are immutable in production. It does not watch
modification times, because a hidden mid-simulation parameter change would
make a thermodynamic calculation irreproducible.
