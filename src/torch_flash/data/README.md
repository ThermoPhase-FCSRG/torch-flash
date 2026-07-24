# Bundled scientific parameter data

All runtime scientific databases use YAML and declare a schema version, model
identity, parameter-set version, SI units, and primary references.

```text
components/default.yaml
models/index.yaml
models/activity/
models/binary_interaction/
models/cpa/
models/cubic/
models/group_contribution/
models/multifluid/
models/standard_state/
schemas/
```

Component records use canonical `torch-flash` names. Aliases are lookup aids
only and never appear in a constructed model's `names` tuple. A `null` property
means unavailable, not zero and not estimated.

Each model file is independent. Model-specific critical and reducing constants
in GERG or EOS-CG therefore remain in that model's file even when a general
critical constant also appears in the shared component database. This prevents
silently replacing constants fitted as part of a published equation.

The public loaders use `yaml.safe_load`, validate schema and units, and cache
the parsed document. Constructors copy the selected values into their own
PyTorch tensors.

`models/activity/unifac-original-public-2026.yaml` is explicitly the original
VLE-UNIFAC parameter identity. It contains 113 subgroup records, 54 selected
main-group identities, and 1,270 directed interaction records. Missing
directed pairs mean unavailable parameters, not zero. Its provenance notes
also document why values and numbering from the UEA/AIM extension are not
merged into the pinned DDBST/`thermo` table.

`models/group_contribution/ppr78-jaubert-mutelet-2004.yaml` contains the
complete six-group parameterization published by Jaubert and Mutelet (2004):
all 15 unique \(A_{kl},B_{kl}\) pairs and the audited normal-alkane
decompositions used by the bundled examples. Later extended PPR78 group
inventories are distinct parameterizations and are not silently combined
with this file.

`schemas/` contains JSON Schema 2020-12 documents written in YAML for editor
and external validation tooling. Model-specific payload requirements are
documented in `docs/parameters.md`.
