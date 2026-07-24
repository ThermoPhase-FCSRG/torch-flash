# Bundled multifluid coefficient data

`../../data/models/multifluid/gerg-2008.yaml` contains the complete
21-component GERG-2008 Helmholtz
coefficient inventory: all pure-fluid ideal and residual equations, all 210
binary reducing-parameter sets, and the 15 nonzero binary departure
functions. The primary source is Kunz and Wagner, *J. Chem. Eng. Data* 57
(2012), 3032–3091,
[doi:10.1021/je300655b](https://doi.org/10.1021/je300655b), and the GERG
monograph. The published constants were cross-checked against the NIST
implementation in
[NIST `teqp` 0.23.2](https://github.com/usnistgov/teqp/tree/v0.23.2).
The applicable NIST notice is reproduced in `THIRD_PARTY_NOTICES.md`.

`../../data/models/multifluid/eos-cg-2021.yaml` contains the complete
16-component EOS-CG-2021 Helmholtz
inventory: all pure-fluid ideal and residual equations, all 120 binary
reducing-parameter sets, and the 21 departure functions. The primary mixture
source is Neumann et al., *Int. J. Thermophys.* 44, 178 (2023), including its
supplementary material
([doi:10.1007/s10765-023-03263-6](https://doi.org/10.1007/s10765-023-03263-6);
[coefficient tables](https://static-content.springer.com/esm/art%3A10.1007%2Fs10765-023-03263-6/MediaObjects/10765_2023_3263_MOESM1_ESM.pdf)).
Pure-fluid equations come from the references assigned in the main paper's
Table 3; all 16 source assignments are enumerated in
[`docs/references.md`](../../../docs/references.md). The MDEA ideal and
residual equations are from Neumann et al.,
*Int. J. Thermophys.* 43, 10 (2022), and its open REFPROP/TREND parameter
file
([doi:10.1007/s10765-021-02933-7](https://doi.org/10.1007/s10765-021-02933-7)).
The remaining tables were cross-checked against the MIT-licensed
[Clapeyron.jl EOS-CG database](https://github.com/ClapeyronThermo/Clapeyron.jl).

“Complete” here refers to the thermodynamic Helmholtz mixture model. Optional
ancillary saturation correlations and transport-property correlations from
the pure-fluid source files are deliberately outside these coefficient
inventories.

No external implementation code is included in these YAML files, and no
external thermodynamic package is imported at runtime. The coefficient
sources, license notices, and redistribution boundary are documented in
`docs/licensing.md`.
