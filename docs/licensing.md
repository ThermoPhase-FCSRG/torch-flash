# Licensing and scientific-data provenance

This page records the release boundary for code, runtime parameter databases,
validation data, and external-software results. It is a conservative
engineering audit, not legal advice. Copyright, database rights, contract
terms, and fair-use or quotation exceptions vary by jurisdiction; obtain
qualified advice before publishing an artifact whose permission remains
unclear.

## Release boundary

| Artifact class | Git repository | PyPI wheel and source distribution | Policy |
|---|---:|---:|---|
| `src/torch_flash/**/*.py` | Yes | Yes | Original package implementation under LGPL-2.1-only |
| Runtime YAML under `src/torch_flash/data/` | Yes | Yes | Minimal model/component parameters, with primary scientific citations and applicable third-party notices |
| Cleared literature-derived CSV under `tests/data/` | Yes, subject to the rights audit below | No | Repository validation fixture with a recorded reusable basis |
| Unresolved research CSV under `tests/data/not-cleared/` | No | No | Local-only input; the entire subdirectory is ignored by Git |
| Independently generated software baselines under `tests/data/` | Yes | No | Project-generated numerical output; generator, software, version, and model inputs must be recorded |
| Notebooks, tests, scripts, and rendered outputs | Yes | No | Research/development material, not package runtime content |
| Books, papers, publisher supplements, and source workbooks | No | No | Never vendored; users obtain them from the cited publisher or archive |

The build is checked in CI so that `.ipynb`, CSV, workbook, image, and PDF
research artifacts cannot enter a Python distribution accidentally.

## Current data-rights audit

The detailed per-file scientific provenance is in the repository's
[`tests/data/README.md`](https://github.com/ThermoPhase-FCSRG/torch-flash/blob/main/tests/data/README.md).
The machine-readable
[`tests/data/rights.yaml`](https://github.com/ThermoPhase-FCSRG/torch-flash/blob/main/tests/data/rights.yaml)
ledger covers all 40 declared CSV artifacts: 5 have a verified open basis, 12
are project-generated, and 23 are marked `not-cleared`. CI rejects an
unclassified CSV and enforces that locally present `not-cleared` inputs are
confined to the ignored `tests/data/not-cleared/` subdirectory. Tests that
consume those inputs skip in a public checkout where the directory is absent.
The rights result does not change whether a file is experimental or
model-generated.

### Explicitly reusable sources

- NIST ThermoML data are obtained from NIST's public ThermoML archive. The
  archive still requires the original articles to be cited and the values to
  be independently checked.
- The H2ThermoBank archive, DOI
  [10.6084/m9.figshare.12063297](https://doi.org/10.6084/m9.figshare.12063297),
  declares CC BY 4.0.
- Moortgat's H2/H2O article and supporting table, DOI
  [10.1371/journal.pone.0332157](https://doi.org/10.1371/journal.pone.0332157),
  are CC BY 4.0.
- The EOS-CG-2021 article and its supplementary coefficient tables, DOI
  [10.1007/s10765-023-03263-6](https://doi.org/10.1007/s10765-023-03263-6),
  are CC BY 4.0 unless a third-party credit line says otherwise.
- The global E-PPR78 article and supplementary 40-group parameter table, DOI
  [10.1016/j.fluid.2022.113456](https://doi.org/10.1016/j.fluid.2022.113456),
  are CC BY 4.0. The separately copyrighted 2017 CCS table is cited as the
  predecessor CCS application but is not the bundled numerical source.
- The MDEA article, experimental tables, and parameter supplement, DOI
  [10.1007/s10765-021-02933-7](https://doi.org/10.1007/s10765-021-02933-7),
  are CC BY 4.0 unless a third-party credit line says otherwise.
- The bundled original-UNIFAC table is derived from the MIT-licensed
  `thermo` data distribution. The scientific publications and DDBST page are
  cited as model provenance; the applicable software/data notice is retained
  in `THIRD_PARTY_NOTICES.md`.

Attribution, license links, and modification notices must remain with
CC BY material. Public availability alone is not interpreted as an open
license.

### Project-generated numerical baselines

Files whose names identify ThermoPack, teqp, or NeqSim are independently
generated outputs used for numerical comparison. They do not contain copied
source code. The frozen-data manifest records the exact software version and
model inputs or points to the generator. The third-party programs themselves
are optional and are not bundled in `torch-flash`.

### Redistribution permission not yet established

Several validation CSV files are transcriptions, normalized subsets, or visual
digitizations of publisher-controlled articles, books, theses, or supporting
workbooks. A DOI and full citation establish provenance, but no blanket
redistribution license has been verified for these sources. They remain
excluded from Git and every Python distribution under
`tests/data/not-cleared/`. For a public repository archive or data release,
each such file must be replaced by an openly licensed source, generated after
a user supplies or downloads the source lawfully, covered by written
permission, or kept outside the release.

The three normalized Jaubert-databank fixtures require particular caution:
`jaubert_2020_co2_binary_vle.csv`, `jaubert_2020_hv_bac5_vle.csv`, and
`jaubert_ppr78_hydrocarbon_vle.csv`. The ACS article is identified as
CC BY-NC-ND 4.0, while no separate permissive license was found for the
supporting workbook. The normalized subsets are adaptations and the package's
license does not impose a noncommercial restriction. Consequently, this audit
does not clear those three files for a public data release; obtain explicit
permission or replace them with a compatibly licensed source first.

This unresolved group currently includes the Segovia density table, Jaubert
binary/PPR78/Huron-Vidal subsets, Yücelen N2/CO2, Ahmadi and Wang CO2/H2O,
Portier brine, Verschoyle H2/CO, Folas and Yan CPA tables/digitizations,
Gillespie-Wilson and Bartlett hydrogen/water tables, Pedersen and Whitson book
tables, and the Gernert-Span Table 8 transcription.
Their scientific use in a private checkout is distinct from permission to
redistribute the normalized tables publicly.

Executed notebook outputs that reproduce `not-cleared` values share the same
unresolved publication risk. Excluding notebooks from PyPI prevents package
redistribution but does not itself clear publication in a public Git
repository.

The current documentation boundary publishes selected rendered plots,
including observation markers, while withholding the paired local-only
notebooks, source CSVs, machine-readable observation values, and notebook
tables. A marker still graphically communicates an approximate observation
coordinate; converting a table into a plot does not change the source
dataset's license. Each published figure therefore declares its executed
notebook checksum, source cell, inputs, evidence class, and publication form
in `docs/assets/validation/manifest.yaml`. CI enforces the declared
figure-only boundary and rejects tracked `not-cleared` research CSVs. This is
a repository publication decision, not a determination that the source data
are openly licensed or that an exception applies in every jurisdiction.

## Runtime parameter databases

Published equations and numerical constants are cited at the file level. When
a bundled coefficient inventory was transcribed or cross-checked with a
licensed software database, the software/data license is also retained:

- `chemicals` 1.5.2 (MIT) for selected shared component and ideal-gas data;
- `thermo` 0.6.0 (MIT) for the original-UNIFAC parameter table;
- Clapeyron.jl (MIT) for cross-checking the EOS-CG and E-PPR78 coefficient
  inventories; and
- NIST teqp 0.23.2 (NIST notice) for cross-checking the GERG inventory.

Cross-checking identifies the independent numerical source; it does not mean
that external implementation code was translated into this package. A
2026-07-24 audit compared whitespace-normalized sequences of six consecutive
nonblank source lines against the locally inspected ThermoPack, NeqSim, teqp,
and Clapeyron source trees. It found no exact implementation-block matches.
That scan is evidence against obvious copying, not a proof of independent
authorship or a substitute for contributor review.

Minimal parameter subsets transcribed from books and journal tables remain
scientifically cited. The larger E-PPR78 inventory is bundled under the 2022
source's CC BY 4.0 license with attribution and a modification notice.
Because compilation or database rights can apply even where individual
numerical facts are not protected, other larger imports require an explicit
license or permission before bundling.

## Contribution checklist

For every new dataset or parameter file:

1. identify the primary scientific source and persistent identifier;
2. distinguish measurement, published-model output, fitted project output,
   and external-software output;
3. record units, transformations, selection rules, uncertainty, and generator;
4. record an explicit license or written permission for redistribution;
5. retain required copyright and license notices; and
6. keep the artifact outside package distributions unless it is required at
   runtime and cleared for redistribution.
