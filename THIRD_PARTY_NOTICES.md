# Third-party notices

`torch-flash` is distributed under the GNU Lesser General Public License
version 2.1. The notices below cover institutional marks, separately licensed
parameter/data sources used by bundled runtime databases, and one optional
integration. External thermodynamic programs used only to generate repository
comparison results are not bundled.

## Institutional marks

The LNCC and UDESC names and logos are included solely to identify the
institutions where `torch-flash` is supported or partially developed. These
institutional marks remain the property of their respective owners and are not
covered by the package's LGPL-2.1 license.

- LNCC: <https://www.gov.br/lncc/pt-br>
- UDESC logo source and usage guidance: <https://www.udesc.br/marcaudesc>

Their inclusion does not imply endorsement of a particular software release
or transfer any trademark rights.

## MIT-licensed parameter and component-data sources

### thermo

The original-UNIFAC parameter YAML was mechanically generated from and
cross-checked against `thermo` 0.6.0:

- Project: <https://github.com/CalebBell/thermo>
- License: MIT
- Copyright (C) 2016-2020 Caleb Bell

No `thermo` Python source code is copied into `torch-flash`. The frozen YAML
also cites the original publications and the public DDBST table so that the
scientific provenance is independent of the software provenance.

### chemicals

`chemicals` is a runtime dependency. Selected component and Poling ideal-gas
values in the bundled YAML were obtained from `chemicals` 1.5.2:

- Project: <https://github.com/CalebBell/chemicals>
- License: MIT
- Copyright (C) 2016-2021 Caleb Bell

No `chemicals` Python implementation is copied into `torch-flash`.

### Clapeyron.jl

The EOS-CG-2021 and E-PPR78 coefficient inventories were independently
cross-checked against the Clapeyron.jl database:

- Project: <https://github.com/ClapeyronThermo/Clapeyron.jl>
- License: MIT
- Copyright (c) 2020 Hon Wa Yew and Pierre Walker

No Clapeyron.jl implementation code is translated or bundled. The respective
CC BY 4.0 papers, supplements, and assigned pure-fluid sources remain the
primary scientific references. Differences in the E-PPR78 inventory were
resolved against the defining 2022 supplementary Table S4.

### MIT license text

The following terms apply to the MIT-licensed resources identified above:

> MIT License
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to
> deal in the Software without restriction, including without limitation the
> rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
> sell copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
> FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
> IN THE SOFTWARE.

## E-PPR78 parameter material

The global E-PPR78 article and its supplementary 40-group coefficient table
are licensed under Creative Commons Attribution 4.0 International:

- J.-N. Jaubert, J.-W. Qian, S. Lasala, and R. Privat, "The impressive
  impact of including enthalpy and heat capacity of mixing data when
  parameterising equations of state. Application to the development of the
  E-PPR78 model," *Fluid Phase Equilibria* 560, 113456 (2022),
  <https://doi.org/10.1016/j.fluid.2022.113456>.
- License: <https://creativecommons.org/licenses/by/4.0/>.

`torch-flash` mechanically transcribed the 356 available group-pair
coefficients from supplementary Table S4, converted MPa to Pa, represented
all 424 unavailable pairs explicitly, normalized group identifiers, and
independently checked the inventory. The 2017 E-PPR78 CCS article,
<https://doi.org/10.1016/j.ijggc.2016.11.015>, establishes the CCS scope and
predecessor parameter revision; its separately copyrighted coefficient table
is not bundled. The source authors do not endorse this package.

## EOS-CG-2021 and MDEA parameter material

The EOS-CG-2021 article and supplementary coefficient tables are licensed
under Creative Commons Attribution 4.0 International:

- T. Neumann et al., "EOS-CG-2021: A Mixture Model for the Calculation of
  Thermodynamic Properties of CCS Mixtures," *International Journal of
  Thermophysics* 44, 178 (2023),
  <https://doi.org/10.1007/s10765-023-03263-6>.
- License: <https://creativecommons.org/licenses/by/4.0/>

The MDEA pure-fluid coefficient source used by EOS-CG-2021 is likewise
licensed under CC BY 4.0:

- T. Neumann et al., "Thermodynamic Properties of Methyl Diethanolamine,"
  *International Journal of Thermophysics* 43, 10 (2022),
  <https://doi.org/10.1007/s10765-021-02933-7>.
- License: <https://creativecommons.org/licenses/by/4.0/>

`torch-flash` transcribed the coefficient tables into its versioned YAML
schema, normalized component identifiers, converted metadata to explicit SI
units, and independently checked the numerical inventory. The source authors
do not endorse this package. Third-party material credited separately in the
articles is not relicensed by these notices.

## Repository-only CC BY 4.0 hydrogen data

The following sources support repository research fixtures but are excluded
from Python distributions:

- A. Hassanpouryouzband et al., “Thermodynamic and transport properties of
  hydrogen containing streams,” *Scientific Data* 7, 222 (2020), and the
  associated H2ThermoBank archive,
  <https://doi.org/10.6084/m9.figshare.12063297>.
- Joachim Moortgat, “Vapor-liquid equilibrium of water-hydrogen mixtures,”
  *PLOS ONE* 20, e0332157 (2025),
  <https://doi.org/10.1371/journal.pone.0332157>.
- License: <https://creativecommons.org/licenses/by/4.0/>.

The repository CSV files normalize column names and SI units and select model
validation fields. Attribution, this modification notice, and the CC BY 4.0
license link must remain with redistributed copies.

## NIST teqp

The GERG-2008 coefficient inventory was independently cross-checked against
NIST `teqp` 0.23.2. No teqp implementation code is translated or bundled.

- Project: <https://github.com/usnistgov/teqp>
- License: NIST Disclaimer of Copyright and Warranty

> This software was developed by employees of the National Institute of
> Standards and Technology (NIST), an agency of the Federal Government and is
> being made available as a public service. Pursuant to title 17 United States
> Code Section 105, works of NIST employees are not subject to copyright
> protection in the United States. This software may be subject to foreign
> copyright. Permission in the United States and in foreign countries, to the
> extent that NIST may hold copyright, to use, copy, modify, create derivative
> works, and distribute this software and its documentation without fee is
> hereby granted on a non-exclusive basis, provided that this notice and
> disclaimer of warranty appears in all copies.
>
> THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER
> EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY
> THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM
> INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE
> SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE. IN NO EVENT
> SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT,
> INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM,
> OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON
> WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED
> BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED
> FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES
> PROVIDED HEREUNDER.

## Optional ugropy integration

`ugropy` 3.x is an optional runtime dependency used only by
`unifac_groups_from_identifiers`:

- Project: <https://github.com/ipqa-research/ugropy>
- License: MIT
- Copyright (c) 2026 Salvador Eduardo Brandolin

No `ugropy` source code or parameter data is bundled. Users who install the
`groups` extra receive it under its own license and should inspect generated
functional-group assignments before scientific use.
