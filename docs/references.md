# Scientific references and data provenance

Scientific claims in `torch-flash` cite the source that defines the equation,
reports the parameter set, or supplies the experimental/reference data. A
software comparison is identified separately from experimental validation.
The frozen CSV provenance is recorded in
[`tests/data/README.md`](https://github.com/ThermoPhase-FCSRG/torch-flash/blob/main/tests/data/README.md).

## Equilibrium algorithms and classical models

| Topic | Primary source |
|---|---|
| Rachford-Rice material balance | H. H. Rachford and J. D. Rice, "Procedure for Use of Electronic Digital Computers in Calculating Flash Vaporization Hydrocarbon Equilibrium," *J. Pet. Technol.* 4 (1952). [doi:10.2118/952327-G](https://doi.org/10.2118/952327-G) |
| Transformed/bounded Rachford-Rice formulation | C. F. Leibovici and J. Neoschil, "A new look at the Rachford-Rice equation," *Fluid Phase Equilib.* 74 (1992) 303-308. [doi:10.1016/0378-3812(92)85069-K](https://doi.org/10.1016/0378-3812%2892%2985069-K) |
| Tangent-plane stability | M. L. Michelsen, "The isothermal flash problem. Part I. Stability," *Fluid Phase Equilib.* 9 (1982) 1-19. [doi:10.1016/0378-3812(82)85001-2](https://doi.org/10.1016/0378-3812%2882%2985001-2) |
| Isothermal phase split | M. L. Michelsen, "The isothermal flash problem. Part II. Phase-split calculation," *Fluid Phase Equilib.* 9 (1982) 21-40. [doi:10.1016/0378-3812(82)85002-4](https://doi.org/10.1016/0378-3812%2882%2985002-4) |
| Dense trust-region stability and multiphase flash | M. Petitfrere and D. V. Nichita, "Robust and efficient Trust-Region based stability analysis and multiphase flash calculations," *Fluid Phase Equilib.* 362 (2014) 51-68. The implementation uses the modified-TPD and improved mole-number Gibbs objectives with exact dense PyTorch Hessians and the Moré-Sorensen restricted step. [doi:10.1016/j.fluid.2013.08.039](https://doi.org/10.1016/j.fluid.2013.08.039) |
| Physical phase identification | J. Bennett and Z. Schmidt, "Comparison of Phase Identification Methods Used in Petroleum Reservoir Simulation," *Energy & Fuels* 31 (2017) 3370-3379. The implementation exposes all five compared homogeneous-state criteria; the two response-derivative criteria use PyTorch autodiff. [doi:10.1021/acs.energyfuels.6b02316](https://doi.org/10.1021/acs.energyfuels.6b02316). G. Venkatarathnam and L. R. Oellrich, "Identification of the phase of a fluid using partial derivatives of pressure, volume and temperature without reference to saturation properties: Applications in phase equilibria calculations," *Fluid Phase Equilib.* 301 (2011) 225-233, defines the dimensionless \(\Pi\) pressure-derivative parameter; the implementation evaluates its first and second pressure derivatives with PyTorch autodiff. [doi:10.1016/j.fluid.2010.12.001](https://doi.org/10.1016/j.fluid.2010.12.001). K. S. Pedersen, P. L. Christensen, and J. A. Shaikh, *Phase Behavior of Petroleum Reservoir Fluids*, 3rd ed., CRC Press (2024), section 6.6, supplies the default SRK/PR \(V/b=1.75\) rule. [doi:10.1201/9780429457418](https://doi.org/10.1201/9780429457418) |
| Thermodynamic derivations and implementation conventions | M. L. Michelsen and J. M. Mollerup, *Thermodynamic Models: Fundamentals & Computational Aspects*, 2nd ed., Tie-Line Publications (2007), ISBN 978-87-989961-3-2. [Bibliographic record](https://books.google.com/books?id=qjmeOgAACAAJ) |
| Activity coefficients, mixing rules, and association theory | G. M. Kontogeorgis and G. K. Folas, *Thermodynamic Models for Industrial Applications: From Classical and Advanced Mixing Rules to Association Theories*, Wiley (2010). [doi:10.1002/9780470747537](https://doi.org/10.1002/9780470747537) |
| SRK | G. Soave, "Equilibrium constants from a modified Redlich-Kwong equation of state," *Chem. Eng. Sci.* 27 (1972) 1197-1203. [doi:10.1016/0009-2509(72)80096-4](https://doi.org/10.1016/0009-2509%2872%2980096-4) |
| PR76 | D.-Y. Peng and D. B. Robinson, "A New Two-Constant Equation of State," *Ind. Eng. Chem. Fundam.* 15 (1976) 59-64. [doi:10.1021/i160057a011](https://doi.org/10.1021/i160057a011) |
| PR78 acentric-factor extension | D. B. Robinson and D.-Y. Peng, *The Characterization of the Heptanes and Heavier Fractions for the GPA Peng-Robinson Programs*, GPA Research Report RR-28 (1978). [Bibliographic record](https://books.google.com/books?id=bE-_HAAACAAJ) |
| PPR78 predictive group-contribution BIPs | J.-N. Jaubert and F. Mutelet, "VLE predictions with the Peng-Robinson equation of state and temperature dependent \(k_{ij}\) calculated through a group contribution method," *Fluid Phase Equilib.* 224 (2004) 285-304. The bundled original parameterization is Table 1; Eq. 5 and Appendix A define and audit the correlation. [doi:10.1016/j.fluid.2004.06.059](https://doi.org/10.1016/j.fluid.2004.06.059) |
| E-PPR78 global 40-group parameters | J.-N. Jaubert, J.-W. Qian, S. Lasala, and R. Privat, "The impressive impact of including enthalpy and heat capacity of mixing data when parameterising equations of state. Application to the development of the E-PPR78 model," *Fluid Phase Equilib.* 560 (2022) 113456. Equation 5 defines the correlation and supplementary Table S4 supplies the bundled 40-group inventory. [doi:10.1016/j.fluid.2022.113456](https://doi.org/10.1016/j.fluid.2022.113456) |
| E-PPR78 for CCS fluids | X. Xu, S. Lasala, R. Privat, and J.-N. Jaubert, "E-PPR78: A proper cubic EoS for modelling fluids involved in the design and operation of carbon dioxide capture and storage (CCS) processes," *Int. J. Greenhouse Gas Control* 56 (2017) 126-154. This establishes the CCS application scope; `torch-flash` uses the later 2022 global coefficient revision. [doi:10.1016/j.ijggc.2016.11.015](https://doi.org/10.1016/j.ijggc.2016.11.015) |
| Petroleum BIP tuning | D. L. Katz and A. Firoozabadi, "Predicting Phase Behavior of Condensate/Crude-Oil Systems Using Methane Interaction Coefficients," *J. Pet. Technol.* 30 (1978) 1649-1655. [doi:10.2118/6721-PA](https://doi.org/10.2118/6721-PA). R. Gani and A. Fredenslund, "Thermodynamics of Petroleum Mixtures Containing Heavy Hydrocarbons: An Expert Tuning System," *Ind. Eng. Chem. Res.* 26 (1987) 1304-1312. [doi:10.1021/ie00067a008](https://doi.org/10.1021/ie00067a008). [Whitson BIP workflow](https://wiki.whitson.com/eos/bips/). |
| Cubic cross-co-volume interaction | R. Privat and J.-N. Jaubert, "The state of the art of cubic equations of state with temperature-dependent binary interaction coefficients: From correlation to prediction," *Fluid Phase Equilib.* 570 (2023) 113697. Equations 9-11 define \(b_m=\sum_i\sum_jx_i x_jb_{ij}\), \(b_{ij}=(b_i+b_j)(1-l_{ij})/2\), and its linear `lij=0` limit. [doi:10.1016/j.fluid.2022.113697](https://doi.org/10.1016/j.fluid.2022.113697) |
| Cubic volume translation | A. Péneloux, E. Rauzy, and R. Fréze, "A consistent correction for Redlich-Kwong-Soave volumes," *Fluid Phase Equilib.* 8 (1982) 7-23. [doi:10.1016/0378-3812(82)80002-2](https://doi.org/10.1016/0378-3812%2882%2980002-2) |
| PR and heavy-fraction volume-shift factors | B. S. Jhaveri and G. K. Youngren, "Three-Parameter Modification of the Peng-Robinson Equation of State To Improve Volumetric Predictions," *SPE Reservoir Engineering* 3 (1988) 1033-1040. [doi:10.2118/13118-PA](https://doi.org/10.2118/13118-PA). Whitson and Brulé (2000), section 4.2.6 and Tables 4.2-4.3, supply the implemented pure-component and family parameter tables. |
| NRTL | H. Renon and J. M. Prausnitz, "Local compositions in thermodynamic excess functions for liquid mixtures," *AIChE J.* 14 (1968) 135-144. [doi:10.1002/aic.690140124](https://doi.org/10.1002/aic.690140124) |
| Wilson activity model | G. M. Wilson, "Vapor-Liquid Equilibrium. XI. A New Expression for the Excess Free Energy of Mixing," *J. Am. Chem. Soc.* 86 (1964) 127-130. [doi:10.1021/ja01056a002](https://doi.org/10.1021/ja01056a002) |
| Original UNIFAC equation | A. Fredenslund, R. L. Jones, and J. M. Prausnitz, "Group-Contribution Estimation of Activity Coefficients in Nonideal Liquid Mixtures," *AIChE J.* 21 (1975) 1086-1099. [doi:10.1002/aic.690210607](https://doi.org/10.1002/aic.690210607) |
| Original UNIFAC revision 5 | H. K. Hansen et al., "Vapor-Liquid Equilibria by UNIFAC Group Contribution. 5. Revision and Extension," *Ind. Eng. Chem. Res.* 30 (1991) 2352-2355. [doi:10.1021/ie00058a017](https://doi.org/10.1021/ie00058a017) |
| Original UNIFAC revision 6 | R. Wittig, J. Lohmann, and J. Gmehling, "Vapor-Liquid Equilibria by UNIFAC Group Contribution. 6. Revision and Extension," *Ind. Eng. Chem. Res.* 42 (2003) 183-188. [doi:10.1021/ie020506l](https://doi.org/10.1021/ie020506l) |
| UNIFAC formulation cross-check | M. Hammer, "UNIFAC – From Extensive Residual Helmholtz Energy to Chemical Potential," ThermoPack memo (2025-01-03). [PDF](https://thermotools.github.io/thermopack/memo/UNIFAC/unifac.pdf) |
| Huron-Vidal mixing | M.-J. Huron and J. Vidal, "New mixing rules in simple equations of state for representing vapour-liquid equilibria of strongly non-ideal mixtures," *Fluid Phase Equilib.* 3 (1979) 255-271. [doi:10.1016/0378-3812(79)80001-1](https://doi.org/10.1016/0378-3812%2879%2980001-1) |
| Original CPA equation | G. M. Kontogeorgis et al., "An Equation of State for Associating Fluids," *Ind. Eng. Chem. Res.* 35 (1996) 4310-4318. [doi:10.1021/ie9600203](https://doi.org/10.1021/ie9600203) |
| CPA cross-association parameters and validation | G. K. Folas et al., "Application of the Cubic-Plus-Association (CPA) Equation of State to Cross-Associating Systems," *Ind. Eng. Chem. Res.* 44 (2005) 3823-3833. [doi:10.1021/ie048832j](https://doi.org/10.1021/ie048832j) |
| CPA hydrocarbon/water mutual solubility and modified CR1 | M. B. Oliveira, J. A. P. Coutinho, and A. J. Queimada, "Mutual solubilities of hydrocarbons and water with the CPA EoS," *Fluid Phase Equilib.* 258 (2007) 58-66. [doi:10.1016/j.fluid.2007.05.023](https://doi.org/10.1016/j.fluid.2007.05.023) |
| CPA reservoir fluids, temperature-dependent BIPs, and heavy-cut adapter | W. Yan, G. M. Kontogeorgis, and E. H. Stenby, "Application of the CPA equation of state to reservoir fluids in presence of water and polar chemicals," *Fluid Phase Equilib.* 276 (2009) 75-85. [doi:10.1016/j.fluid.2008.10.007](https://doi.org/10.1016/j.fluid.2008.10.007) |
| Heavy-end gamma characterization, critical properties, PR/SRK BIPs, and worked equilibrium examples | C. H. Whitson and M. R. Brulé, *Phase Behavior*, SPE Monograph Series, volume 20, Society of Petroleum Engineers (2000), ISBN 978-1-55563-087-4. Chapter 5, Tables A-1B/A-3, and Appendices B/C. [Bibliographic record](https://books.google.com/books?id=Z4cQAQAAMAAJ) |
| Heavy-end logarithmic characterization/lumping, BIPs, multiphase examples, and thermal-property equations | K. S. Pedersen, P. L. Christensen, and J. A. Shaikh, *Phase Behavior of Petroleum Reservoir Fluids*, 3rd ed., CRC Press (2024). Chapters 4-8 and 10. [doi:10.1201/9780429457418](https://doi.org/10.1201/9780429457418) |
| Ideal-gas heat-capacity polynomials | B. E. Poling, J. M. Prausnitz, and J. P. O'Connell, *The Properties of Gases and Liquids*, 5th ed., McGraw-Hill (2001). Frozen coefficients are taken from the Poling data bank distributed by `chemicals` 1.5.2; `torch-flash` records that software/data version instead of presenting the values as a new fit. |
| Pedersen corresponding-states viscosity | K. S. Pedersen et al., "Viscosity of crude oils," *Chem. Eng. Sci.* 39 (1984) 1011-1016. [doi:10.1016/0009-2509(84)87009-8](https://doi.org/10.1016/0009-2509%2884%2987009-8) |
| Lohrenz-Bray-Clark viscosity | J. Lohrenz, B. G. Bray, and C. R. Clark, "Calculating Viscosities of Reservoir Fluids From Their Compositions," *J. Pet. Technol.* 16 (1964) 1171-1176. [doi:10.2118/915-PA](https://doi.org/10.2118/915-PA). The implemented numerical form and C7+ critical-volume estimator follow Pedersen (2024), section 10.1.3, and are checked against Whitson Appendix B, Problem 7. |
| Heavy-oil corresponding states | N. Lindeloff, K. S. Pedersen, H. P. Rønningsen, and J. Milter, "The Corresponding States Viscosity Model Applied to Heavy Oil Systems," *J. Can. Pet. Technol.* 43 (2004) 47-53, Paper 2003-150. Pedersen et al. (2024), Eqs. 10.33-10.37, supply the implemented factor and pressure forms. |
| Heavy-aromatic C200 characterization | K. S. Pedersen, J. Milter, and H. Sørensen, "Cubic Equations of State Applied to HT/HP and Highly Aromatic Reservoir Fluids," *SPE J.* 9 (2004) 186-192. [doi:10.2118/88364-PA](https://doi.org/10.2118/88364-PA) |
| Lee natural-gas viscosity | A. Lee, M. Gonzalez, and B. Eakin, "The Viscosity of Natural Gases," *J. Pet. Technol.* 18 (1966) 997-1000. [doi:10.2118/1340-PA](https://doi.org/10.2118/1340-PA) |
| One-parameter friction-theory viscosity | S. E. Quiñones-Cisneros, C. K. Zéberg-Mikkelsen, and E. H. Stenby, "One parameter friction theory models for viscosity," *Fluid Phase Equilib.* 178 (2001) 1-16. [doi:10.1016/S0378-3812(00)00474-X](https://doi.org/10.1016/S0378-3812%2800%2900474-X) |
| Methane-reference transport | H. J. M. Hanley, R. D. McCarty, and W. M. Haynes, "Equation for the viscosity and thermal conductivity coefficients of methane," *Cryogenics* 15 (1975) 413-417. [doi:10.1016/0011-2275(75)90010-7](https://doi.org/10.1016/0011-2275%2875%2990010-7) |
| Corresponding-states mixture thermal conductivity | P. L. Christensen and A. Fredenslund, "A corresponding states model for the thermal conductivity of gases and liquids," *Chem. Eng. Sci.* 35 (1980) 871-875. The validation measurements are from the authors' independent CO2/methane dataset, [doi:10.1021/je60083a034](https://doi.org/10.1021/je60083a034). |
| Pure-fluid surface tension | J. R. Brock and R. B. Bird, "Surface Tension and the Principle of Corresponding States," *AIChE J.* 1 (1955) 174-177. [doi:10.1002/aic.690010208](https://doi.org/10.1002/aic.690010208) |
| Multicomponent interfacial tension | C. F. Weinaug and D. L. Katz, "Surface Tensions of Methane-Propane Mixtures," *Ind. Eng. Chem.* 35 (1943) 239-246; S. T. Lee and M. C. H. Chien, "A New Multicomponent Surface Tension Correlation Based on Scaling Theory," SPE-12643-MS (1984). [doi:10.2118/12643-MS](https://doi.org/10.2118/12643-MS) |
| Infinite-dilution n-paraffin diffusion | W. Hayduk and B. S. Minhas, "Correlations for Prediction of Molecular Diffusivities in Liquids," *Can. J. Chem. Eng.* 60 (1982) 295-299. [doi:10.1002/cjce.5450600213](https://doi.org/10.1002/cjce.5450600213) |

## Multiparameter Helmholtz mixture equations of state

| Model or dataset | Primary source |
|---|---|
| GERG-2008 | O. Kunz and W. Wagner, "The GERG-2008 Wide-Range Equation of State for Natural Gases and Other Mixtures: An Expansion of GERG-2004," *J. Chem. Eng. Data* 57 (2012) 3032-3091. [doi:10.1021/je300655b](https://doi.org/10.1021/je300655b) |
| GERG-2004 algorithms | O. Kunz, R. Klimeck, W. Wagner, and M. Jaeschke, *The GERG-2004 Wide-Range Equation of State for Natural Gases and Other Mixtures*, GERG Technical Monograph 15, VDI Verlag (2007), ISBN 978-3-18-355706-6. |
| EOS-CG (2015 formulation) | J. Gernert and R. Span, "EOS-CG: A Helmholtz energy mixture model for humid gases and CCS mixtures," *J. Chem. Thermodyn.* 93 (2016) 274-293. [doi:10.1016/j.jct.2015.05.015](https://doi.org/10.1016/j.jct.2015.05.015) |
| EOS-CG-2021 | T. Neumann et al., "EOS-CG-2021: A Mixture Model for the Calculation of Thermodynamic Properties of CCS Mixtures," *Int. J. Thermophys.* 44 (2023), article 178. [doi:10.1007/s10765-023-03263-6](https://doi.org/10.1007/s10765-023-03263-6); [supplementary coefficient and data tables](https://static-content.springer.com/esm/art%3A10.1007%2Fs10765-023-03263-6/MediaObjects/10765_2023_3263_MOESM1_ESM.pdf) |
| MDEA pure-fluid equation and experiments | T. Neumann et al., "Thermodynamic Properties of Methyl Diethanolamine," *Int. J. Thermophys.* 43 (2022), article 10. [doi:10.1007/s10765-021-02933-7](https://doi.org/10.1007/s10765-021-02933-7) |
| Hydrogen-containing GERG databank | A. Hassanpouryouzband et al., "Thermodynamic and transport properties of hydrogen containing streams," *Sci. Data* 7 (2020), article 222. [doi:10.1038/s41597-020-0568-6](https://doi.org/10.1038/s41597-020-0568-6); [figshare data archive](https://doi.org/10.6084/m9.figshare.12063297) |
| Reference equations for H2/CH4, H2/N2, H2/CO, and H2/CO2 | R. Beckmüller et al., "New Equations of State for Binary Hydrogen Mixtures Containing Methane, Nitrogen, Carbon Monoxide, and Carbon Dioxide," *J. Phys. Chem. Ref. Data* 50 (2021), 013102. [doi:10.1063/5.0040533](https://doi.org/10.1063/5.0040533); [NIST record](https://www.nist.gov/publications/new-equations-state-binary-hydrogen-mixtures-containing-methane-nitrogen-carbon) |

### Pure-fluid equations used by EOS-CG-2021

EOS-CG-2021 Table 3 assigns a specific pure-fluid Helmholtz equation to every
component. The native coefficient inventory follows that assignment; the
mixture paper and its supplement are not substitutes for these pure-fluid
sources.

| Component(s) | Pure-fluid source used by EOS-CG-2021 |
|---|---|
| CO2 | R. Span and W. Wagner, *J. Phys. Chem. Ref. Data* 25 (1996) 1509. [doi:10.1063/1.555991](https://doi.org/10.1063/1.555991) |
| H2O | W. Wagner and A. Pruss, *J. Phys. Chem. Ref. Data* 31 (2002) 387. [doi:10.1063/1.1461829](https://doi.org/10.1063/1.1461829) |
| N2 | R. Span et al., *J. Phys. Chem. Ref. Data* 29 (2000) 1361. [doi:10.1063/1.1349047](https://doi.org/10.1063/1.1349047) |
| O2 | R. Schmidt and W. Wagner, *Fluid Phase Equilib.* 19 (1985) 175. [doi:10.1016/0378-3812(85)87016-3](https://doi.org/10.1016/0378-3812%2885%2987016-3) |
| Ar | C. Tegeler, R. Span, and W. Wagner, *J. Phys. Chem. Ref. Data* 28 (1999) 779. [doi:10.1063/1.556037](https://doi.org/10.1063/1.556037) |
| CO and H2S | E. W. Lemmon and R. Span, *J. Chem. Eng. Data* 51 (2006) 785. [doi:10.1021/je050186n](https://doi.org/10.1021/je050186n) |
| H2 | J. W. Leachman et al., *J. Phys. Chem. Ref. Data* 38 (2009) 721. [doi:10.1063/1.3160306](https://doi.org/10.1063/1.3160306) |
| CH4 | U. Setzmann and W. Wagner, *J. Phys. Chem. Ref. Data* 20 (1991) 1061. [doi:10.1063/1.555898](https://doi.org/10.1063/1.555898) |
| SO2 | K. Gao et al., *J. Chem. Eng. Data* 61 (2016) 2859. [doi:10.1021/acs.jced.6b00195](https://doi.org/10.1021/acs.jced.6b00195) |
| MEA | S. Herrig, *New Helmholtz-Energy Equations of State for Pure Fluids and CCS-Relevant Mixtures*, doctoral thesis, Ruhr University Bochum (2018/2019). [German National Library record and full text](https://d-nb.info/1180028023/34) |
| DEA | M. Kortmann, *Development of Empirical Multiparameter Equations of State for Monoethanolamine and Diethanolamine*, master's thesis, Ruhr University Bochum (2016); bibliographic assignment in [EOS-CG-2021 Table 3](https://doi.org/10.1007/s10765-023-03263-6) |
| HCl | M. Thol et al., *J. Chem. Eng. Data* 63 (2018) 2533. [doi:10.1021/acs.jced.7b01031](https://doi.org/10.1021/acs.jced.7b01031) |
| Cl2 | M. Thol et al., *AIChE J.* 67 (2021), e17326. [doi:10.1002/aic.17326](https://doi.org/10.1002/aic.17326) |
| NH3 | K. Gao et al., *J. Phys. Chem. Ref. Data* 52 (2023), 013102. [doi:10.1063/5.0128269](https://doi.org/10.1063/5.0128269) |
| MDEA | T. Neumann et al., *Int. J. Thermophys.* 43 (2022), article 10. [doi:10.1007/s10765-021-02933-7](https://doi.org/10.1007/s10765-021-02933-7) |

## Experimental and numerical validation sources

- The PPR78 hydrocarbon VLE validation retains all 103 phase-complete states
  at the selected isotherms: methane/ethane measurements from
  [Wichterle and Kobayashi (1972)](https://doi.org/10.1021/je60052a022) and
  [Wei et al. (1995)](https://doi.org/10.1021/je00020a002), and
  methane/n-decane measurements from
  [Reamer et al. (1942)](https://doi.org/10.1021/ie50396a025) and
  [Lin et al. (1979)](https://doi.org/10.1021/je60081a004). The normalized
  source tables are distributed by the
  [Jaubert et al. databank](https://doi.org/10.1021/acs.iecr.0c01734).
  Both binaries contributed to the original PPR78 fit, so this is explicitly
  calibration-domain validation.
- The methane+n-decane compressed-density validation transcribes 92 states
  from Segovia et al., including the complete 80 MPa isobar and 323.15 K
  isotherm used for the 86-state mixture metrics and six 0.1 MPa pure-n-decane
  states used for the temperature correction.
  [doi:10.1016/j.jct.2017.01.022](https://doi.org/10.1016/j.jct.2017.01.022).
- The curated CO2 binary VLE subset is drawn from the electronic workbook
  accompanying A. Jaubert et al., "Benchmark Database Containing Binary-System-
  High-Quality-Certified Data for Cross-Comparing Thermodynamic Models and
  Assessing Their Accuracy," *Ind. Eng. Chem. Res.* 59 (2020) 14981-15027.
  [doi:10.1021/acs.iecr.0c01734](https://doi.org/10.1021/acs.iecr.0c01734).
  The selected primary tables are CH4/CO2 from
  [Wei et al. (1995)](https://doi.org/10.1021/je00020a002), N2/CO2 from
  [Alsahhaf et al. (1983)](https://doi.org/10.1021/i100012a004), O2/CO2 from
  [Fredenslund and Sather (1970)](https://doi.org/10.1021/je60044a024),
  CO2/H2O from [Hou et al. (2013)](https://doi.org/10.1016/j.supflu.2012.11.011),
  and CO2/H2S from [Chapoy et al. (2013)](https://doi.org/10.1016/j.fluid.2013.07.050).
  `torch-flash` records the selected Christiansen et al. CO/CO2 table without
  an inferred DOI.
- The Huron-Vidal BAC-5 subset uses the same Jaubert et al. workbook and
  retains 328 states for n-butane/water, ethanol/n-heptane, and
  methanol/benzene. Primary sources with persistent identifiers are listed in
  the [frozen-data manifest](https://github.com/ThermoPhase-FCSRG/torch-flash/blob/main/tests/data/README.md).
  The curated subset excludes 23 rows whose component identity cannot be
  established consistently from the workbook metadata and its cited primary
  source.
- The experimental H2/CH4, H2/N2, and H2/CO2 density records are distributed
  through the [NIST ThermoML archive](https://trc.nist.gov/ThermoML/Browse)
  and originate in
  [doi:10.1021/acs.jced.7b01125](https://doi.org/10.1021/acs.jced.7b01125),
  [doi:10.1021/acs.jced.7b00694](https://doi.org/10.1021/acs.jced.7b00694),
  and [doi:10.1021/acs.jced.7b00213](https://doi.org/10.1021/acs.jced.7b00213),
  respectively.
- H2/CO coexisting compositions are from T. T. H. Verschoyle, "The Ternary
  System Carbon Monoxide-Nitrogen-Hydrogen and the Component Binary Systems
  between Temperatures of -185 and -215 C, and between Pressures of 0 and 225
  Atm," *Phil. Trans. R. Soc. A* 230 (1932) 189-222.
  [doi:10.1098/rsta.1932.0006](https://doi.org/10.1098/rsta.1932.0006).
  Transcription was cross-checked against
  [US NBS Technical Note 108](https://www.govinfo.gov/content/pkg/GOVPUB-C13-43afd4702e63123a12bf51e0c21d182d/pdf/GOVPUB-C13-43afd4702e63123a12bf51e0c21d182d.pdf).
- H2/H2O VLE data and the PR-CPA comparison are from J. Moortgat,
  "Vapor-Liquid Equilibrium of Water-Hydrogen Mixtures: A Review of
  Experimental Data and Modeling with a Cubic-Plus-Association
  Equation-of-State," *PLOS ONE* 20 (2025), e0332157.
  [doi:10.1371/journal.pone.0332157](https://doi.org/10.1371/journal.pone.0332157).
  Its PR-based CPA convention is not identical to the package's current
  SRK-CPA convention; the notebooks keep those models distinct.
- The propane-water mutual-solubility measurements are from R. Kobayashi and
  D. L. Katz, "Vapor-Liquid Equilibria For Binary Hydrocarbon-Water Systems,"
  *Ind. Eng. Chem.* 45 (1953) 440-446.
  [doi:10.1021/ie50518a051](https://doi.org/10.1021/ie50518a051). The
  `torch-flash` transcription follows Pedersen et al. (2024), Table 16.2.
- The EOS-CG-2015 computer-verification states are all 30 rows of Gernert and
  Span (2016), Table 8; they are numerical reference values, not experiment.
- The MDEA density values are Neumann et al. (2022), Table 1, and the
  speed-of-sound values are Tables 4 and 5. Their reported expanded
  uncertainties are retained in the CSV files.
- The H2/CH4 table is the GERG-generated model databank distributed with
  Hassanpouryouzband et al. (2020), not an experimental dataset.
- The Rachford-Rice stress corpus is the
  [Whitson Rachford-Rice Contest](https://github.com/WhitsonAS/Rachford-Rice-Contest)
  pinned at commit
  [`503b92f`](https://github.com/WhitsonAS/Rachford-Rice-Contest/tree/503b92f1b2847c4459326841f538739bcb9d629f).
- Whitson Appendix B values used in the executed notebook are Problems 7,
  15, and 18 (Tables B-11/B-12, B-18 through B-21, and B-28 through B-32).
  Appendix C uses Tables C-7 through C-11. The notebook records two source
  limitations: the printed Problem 15 K values are rounded, and the N2/CO2
  property rows appear interchanged in Table C-7 relative to Tables A-1/C-10.
- Pedersen Table 6.5 supplies the rounded pseudo-component characterization
  and Table 6.6 the VLL result. Because Table 6.6 does not identify its BIP
  convention, zero BIPs and the independently tabulated Table 4.2 BIPs are
  reported as separate model definitions. Chapter 8 Table 8.1 reproduces CO2
  Joule-Thomson observations from
  [Wang et al. (2017)](https://doi.org/10.1016/j.jcou.2017.04.007), while
  Table 8.2 reproduces propane observations from
  [Sage et al. (1936)](https://doi.org/10.1021/ie50317a026).

## External implementation baselines

- NIST [`teqp`](https://pages.nist.gov/teqp-docs/en/main/) supplies canonical
  cubic and multiparameter Helmholtz numerical checks. Frozen values record the exact
  package version used.
- [ThermoPack](https://thermotools.github.io/thermopack/) and
  [NeqSim](https://github.com/equinor/neqsim-python) are independent
  implementation comparisons. Their component databases and defaults are not
  assumed to match `torch-flash`.
- CoolProp's Helmholtz implementation is described by I. H. Bell et al.,
  *Ind. Eng. Chem. Res.* 53 (2014) 2498-2508.
  [doi:10.1021/ie4033999](https://doi.org/10.1021/ie4033999).
