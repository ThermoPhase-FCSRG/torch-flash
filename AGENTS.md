# Repository instructions for AI coding agents

## Scope and purpose

This file applies to the entire `torch-flash` repository. Keep one root
instruction file unless a genuinely independent subtree later requires
different rules.

`torch-flash` is scientific software for differentiable thermodynamic
properties and phase-equilibrium calculations on PyTorch. Treat correctness,
traceability, numerical robustness, and reproducibility as product
requirements, not optional polish.

The current checkout is the source of truth. Before changing anything, inspect
the relevant implementation, tests, parameter documents, notebooks, and
documentation. Do not rely on remembered repository state or an earlier
conversation when the files can be checked directly.

## Collaboration style

- Work autonomously within the requested scope. Make reasonable,
  evidence-based assumptions when they are reversible and do not alter the
  scientific question.
- Lead status updates and the final response with the concrete outcome.
  Mention changed files, executed checks, measured results, and remaining
  limitations.
- Be concise but scientifically explicit. State assumptions, units, parameter
  identities, numerical tolerances, and possible failure modes when they
  affect interpretation.
- Do not stop at a plausible-looking implementation. Continue through tests,
  numerical checks, notebook execution, plots, documentation, and packaging
  checks appropriate to the change.
- Do not claim completion when a required numerical gate, validation,
  notebook execution, or final QA step has not run. Report unexecuted expensive
  or unavailable checks precisely.
- Ask for user input only when a missing choice would materially change the
  scientific result or authorize a broader/destructive action.
- Preserve unrelated user changes. Inspect `git status` before editing, avoid
  destructive Git operations, and do not rewrite or delete work merely to
  obtain a clean tree.
- Use fast repository searches such as `rg` and `rg --files`. Prefer small,
  reviewable patches over bulk rewrites.

## Non-negotiable scientific standards

### Reference hierarchy and independent implementation

1. Prefer the primary paper, book equation, official supplement, or standard
   that defines the model and parameter set.
2. Use authoritative open implementations and official documentation to
   cross-check interpretation.
3. Use ThermoPack, teqp, NeqSim, CoolProp, `thermo`, and similar packages as
   independent numerical references, with exact versions and settings
   recorded.

Never translate or lightly rewrite third-party implementation code. Implement
the published mathematics independently, cite the defining source, and use
external software only to check behavior. Do not describe a generic cubic,
association, activity, or Helmholtz form as a named model unless its required
published terms and coefficients are present.

When sources disagree:

- inspect the primary equations and errata;
- record the exact convention selected;
- add a regression that distinguishes the alternatives; and
- describe the `torch-flash` formulation directly, with its defining citation.

In user-facing documentation, do not editorialize about errors, omissions, or
shortcomings in another source or implementation. State what `torch-flash`
implements and validates. Keep any unavoidable source-identity or provenance
decision neutral and confined to the relevant validation or data manifest.

Never fabricate, infer, or silently fit a missing published coefficient. A
custom fit must have a distinct identifier and must be labeled as, for
example, “GERG-2008 form after fit,” not as the published GERG-2008
parameterization.

### Units, states, and thermodynamic meaning

- Public thermodynamic inputs and outputs use SI units.
- Record any source-unit conversion next to the ingestion code or parameter
  metadata and test at least one conversion.
- Use canonical component names from the component database. Preserve aliases
  only at API boundaries.
- Keep the supplied homogeneous-state API independent of equilibrium solving.
  Properties at a specified `(T, P, x)` or `(T, V, n)` state must not require a
  flash.
- Keep root selection, physical phase identification, and equilibrium regime
  as separate concepts. Phase identification is a diagnostic with ambiguity,
  not a new equilibrium equation.
- Do not hide critical-region ill-conditioning, multiple roots, metastability,
  or phase disappearance behind broad tolerances.
- Reject algebraic homogeneous `x = y` solutions as two-phase coexistence
  unless a dedicated critical calculation establishes the endpoint.

### Numerical solver behavior

- Every iterative solver must expose convergence, iteration count, and a
  physically meaningful residual. Include material-balance and fugacity
  residuals where applicable.
- Never silently return a failed solve as valid. Use a domain-specific warning,
  explicit non-converged result, or exception according to the public API.
- Initial guesses and continuation choices are scientific configuration.
  Keep them explicit when they affect branch selection.
- Test positivity, normalization, mass balance, root admissibility, and
  invariance to harmless component permutations where applicable.
- Choose tolerances from equation conditioning, float precision, and source
  uncertainty. Explain non-obvious tolerances in the test.
- Near critical points, verify branch continuity and closure; do not replace a
  physical branch with the trivial solution merely because Newton converges.

## Software architecture and API

- Organize source by thermodynamic domain: equations of state, activity
  models, mixing rules, material balance, flash/stability, properties,
  characterization, transport, solvers, databases, and optional backends.
- Prefer typed free functions for calculations. Use frozen dataclasses or
  small classes when state, parameters, or a complex abstraction genuinely
  benefits from encapsulation.
- Keep native PyTorch models independent of optional external backends.
- Maintain the separation between:
  - immutable scientific parameter metadata;
  - per-model tensors, including trainable tensors;
  - runtime device/dtype/thread configuration; and
  - numerical solver options.
- Public APIs must be typed, documented, exported intentionally, and covered
  by tests. Avoid exposing notebook-only helpers as package API.
- Name public functions, methods, classes, result types, parameters, and
  variables for their scientific action and scope. In particular, function
  names must use a clear action and expose important protocol/model
  restrictions such as Helmholtz-only, cubic-only, binary-only, or
  homogeneous-state behavior; do not rely on type annotations or docstrings
  to repair an ambiguous name.
- Name result and data containers for the reusable scientific quantity they
  represent, not for whichever model or algorithm currently produces them.
  Add a model or formulation qualifier to a container only when its fields or
  scientific meaning genuinely depend on that model. Keep real applicability
  restrictions on the producing operation instead; for example, a
  Helmholtz-only solver may return a generic bubble-point-with-volumes result.
- Every public function and class must have a detailed docstring that explains
  its scientific purpose, parameters, units, tensor shapes, return values,
  convergence or failure behavior, assumptions, and important numerical or
  physical limitations. A signature summary alone is not sufficient.
- When a public API implements or materially follows an algorithm, equation,
  correlation, or parameterization from the literature, its docstring must
  identify the defining publication and cite it with enough precision to
  locate the method: authors, work, year, equation/section/table where
  applicable, and a DOI or other persistent identifier. Distinguish the
  equation source from parameter and validation-data sources.
- Preserve backwards compatibility unless the task explicitly authorizes an
  API break. If a correction changes numerical meaning, document the migration
  and add a regression for the old failure.
- Keep dependencies purposeful. Prefer existing PyTorch operations and the
  repository solver abstractions before adding a new numerical dependency.

## PyTorch, autodiff, batching, and runtime configuration

- Preserve gradients through thermodynamic quantities and fitted parameters.
  Do not detach, convert to NumPy, call `.item()`, or recreate tensors inside a
  differentiable hot path unless the non-differentiable boundary is deliberate
  and documented.
- Honor the caller's device and dtype. Construct tensors from existing tensors
  or configured tensor options so CPU, CUDA, and supported accelerators behave
  consistently.
- Support leading batch dimensions where the governing equations permit it.
  Avoid Python loops over states in performance-sensitive kernels.
- Use PyTorch autodiff instead of hand-derived model derivatives unless an
  analytic expression is required for robustness or speed and is independently
  checked against autodiff.
- Configure device, dtype, CPU threading, and deterministic behavior before
  model construction through the runtime configuration API. Library functions
  must not repeatedly change process-wide thread settings.
- Float64 is the reference precision for equilibrium and derivative work.
  Treat float32 results near critical points or coalescing roots as a separate
  accuracy study, not an equivalent benchmark.
- `torch.compile` and GPU execution are workload-dependent. Compile repeated
  tensor kernels and use GPUs for sufficiently large device-resident batches;
  do not assume that tiny sequential Newton systems become faster on a GPU.

## Parameters and component databases

- Store bundled component and model parameters in versioned YAML, using the
  schemas under `src/torch_flash/data/schemas/`.
- Every parameter document must identify its model, version, SI units,
  components, source references, and applicable validity or fitting range.
- Shared component properties and model-specific constants are distinct.
  Never substitute one for the other merely because they have the same name.
- All packaged database constructors must also accept custom user databases or
  explicit typed parameters/tensors.
- Cache parsed immutable documents to avoid repeated disk I/O, but create
  independent model instances and tensors. Never share a trainable tensor
  accidentally through a cache.
- Fitted parameters need a new identifier/version plus the dataset, objective,
  split, bounds or priors, and parameterization recorded.
- Diagnose fit identifiability with sensitivity/Jacobian rank, conditioning,
  parameter correlations, and holdout behavior when the number of adjustable
  parameters is substantial.

## Verification, validation, and tests

Use the terms precisely:

- **Verification** checks that code solves the stated equations: analytical
  limits, identities, published worked examples, autodiff parity, or
  matched-input independent-software results.
- **Validation** checks whether a selected model and parameter set represent
  independent experimental observations.
- **Calibration** estimates parameters. Agreement on calibration data is not
  independent validation.
- A model-generated reference bank is verification, not experimental data.

For each model or bug fix, add the smallest useful combination of:

- unit tests for equations and edge cases;
- regression tests for the reported failure;
- autodiff-versus-finite-difference or analytic checks;
- integration tests for state properties and equilibrium;
- published worked examples;
- independent software baselines with identical constants and conventions;
  and
- experimental validation over a meaningful range.

Requirements:

- Maintain at least 99% branch-aware coverage from the start.
- Do not weaken tolerances, delete difficult states, or exclude branches merely
  to restore coverage.
- Normal tests must not import ThermoPack, teqp, or NeqSim. Generate frozen,
  versioned baselines separately and record generator, version, complete model
  inputs, units, and residuals.
- Every new or materially changed numerical model, solver, property, or
  reference bank must include at least one `pytest-regressions` snapshot of a
  representative structured result bank. Keep equation, physics, tolerance,
  convergence, and derivative assertions as well; the snapshot complements
  rather than replaces them. Prefer `num_regression` for floating result
  arrays so behavioral changes produce a reviewable numerical diff. Use
  ``rtol=1e-5`` as the default snapshot tolerance; increase it only when a
  documented conditioning or cross-platform nonportability study justifies
  the wider regression envelope.
- A few hand-picked points are not sufficient for a validation study when a
  fuller table or curve is available.
- Match pure-component constants, alpha functions, mixing conventions,
  standard states, root selection, and interaction parameters before
  interpreting differences between packages.
- Always distinguish numerical solver failure from poor physical model
  agreement.

## Notebook study standard

### Application-level notebook boundary

Notebooks are application-level demonstrations for engineers and scientists
who may not be experienced programmers. They must show how to solve a
thermodynamic problem with the highest-level suitable public `torch-flash`
APIs, not implement the underlying algorithms inside the study.

- Keep thermodynamic algorithms, numerical solvers, continuation and tracing
  methods, derivative/property assembly, reusable data transformations,
  parallel execution kernels, and other core computational logic under
  `src/torch_flash/`, organized in the appropriate scientific domain.
- Notebook code should primarily configure a case, load study inputs, call
  public package APIs, orchestrate existing `torch-flash` features, inspect
  convergence and physical diagnostics, and present results.
- Small notebook-local helpers are acceptable only for genuinely
  study-specific presentation or input adaptation. They must not reimplement
  package behavior or hide a generally useful scientific or numerical method.
  Repeated or non-trivial `def`/`class` blocks are a prompt to stop and review
  the package API boundary.
- If a notebook needs a capability that `torch-flash` does not expose, first
  implement it in the package as a typed, reusable, batch-aware function or
  appropriately small abstraction. Generalize it across scientifically valid
  cases instead of encoding one paper figure, dataset, or composition unless
  the published method is intrinsically case-specific.
- Preserve PyTorch dtype, device, batching, and autodiff behavior in extracted
  code. Add unit, numerical-regression, difficult-state, and derivative tests
  as applicable, maintain the branch-aware coverage gate, document the public
  API, and only then reduce the notebook to a high-level consumer of it.
- Keep paper-specific plotting, labels, lawful local-data ingestion, and
  validation comparisons in the notebook or study tooling when they are not
  reusable package capabilities. Do not make the installable library depend
  on notebook-only plotting/dataframe packages or `not-cleared` research data.
- During review, judge a notebook by whether an application engineer can
  understand and adapt the workflow without understanding solver
  implementation details. A notebook that contains the implementation needed
  to produce its result is not complete, even if it executes correctly.

Notebook studies are paired Jupytext artifacts. Edit the percent-format `.py`
source first, synchronize it, execute the `.ipynb` from top to bottom, and
inspect the saved tables and plots. Keep filenames paired and place studies in
the appropriate category: `verification`, `validation`, `equilibrium`,
`solubility`, `fitting`, `characterization`, or `performance`.

Every scientific notebook should:

1. state the question, model identity, parameter source, and verification or
   validation class;
2. show equations or cite the exact defining equation/table;
3. record package versions, hardware where relevant, SI units, assumptions,
   solver settings, convergence criteria, and data provenance;
4. check convergence, residuals, phase separation, and material balance before
   plotting;
5. display the physical property or phase diagram, not only an error plot;
6. plot experimental/reference markers together with model predictions;
7. cover the available data and a scientifically useful range, with
   extrapolation clearly marked;
8. explain missing or non-converged curve segments instead of silently drawing
   incomplete lines; and
9. finish with quantitative metrics, limitations, and a reproducible
   conclusion.

Plot-specific expectations:

- Fitting plots show experimental data and clearly labeled **before** and
  **after** curves for every calibrated model.
- Binary VLE studies distinguish liquid `x` and vapor `y`, include complete
  `P-x-y` behavior when possible, and do not plot rejected `x = y` roots as
  coexistence.
- Phase envelopes should cover the full physically relevant two-phase region,
  including the high-temperature/high-pressure closure when the model has one.
- Density, viscosity, or thermal-property error plots must be accompanied by
  the corresponding property-versus-pressure or property-versus-temperature
  plot.
- Performance studies report cold and warmed timing separately, repeat counts,
  hardware, threads, dtype, batch shape, package versions, and accuracy or
  residual checks.

Do not manually edit notebook JSON to change a scientific result. Regenerate
from the paired source. If execution is too expensive for routine CI, provide
smoke-friendly settings and an explicit full-study path.

## Performance work

- Profile before optimizing and preserve an unoptimized correctness oracle or
  an independent numerical comparison.
- Optimize batch layout, tensor reuse, compiled kernels, initialization, and
  continuation before introducing more complex backends.
- Benchmark matched workloads. A homogeneous pressure call, a density root,
  and a complete TP flash are different operations and must not share a timing
  label.
- Report accuracy with timing. A faster calculation that selects another root
  or loses near-critical precision is not a valid speedup.
- Keep hardware-specific measurements in the performance documentation or
  notebooks, not in the README.

## Documentation and claims

- The README is a concise entry point: purpose, installation, minimal working
  example, current capability summary, documentation links, disclosure, and
  license.
- Do not put changelog narratives, former-versus-current timings, transient
  test counts, validation tables, or notebook inventories in the README.
- Documentation describes the current implementation. Historical change logs
  belong in release notes, not model or performance guides.
- Put detailed verification, validation, benchmark conditions, notebook
  studies, and scientific caveats in `docs/`.
- User-facing documentation and reports must mention only tracked repository
  artifacts and behavior shipped by `torch-flash`. Do not describe local-only,
  ignored, private, or untracked data paths, notebooks, figures, or execution
  artifacts as part of the published project.
- Cite claims near the relevant text. Prefer primary references and persistent
  DOIs. Distinguish equation source, parameter source, experimental source,
  and external-software baseline.
- Public user-facing documentation must describe only artifacts,
  capabilities, results, and workflows that are tracked in this repository or
  shipped by `torch-flash`. Do not discuss private, ignored, or local-only
  paths, source-clearance workflows, or internal publication boundaries in
  model guides or validation reports. Keep necessary legal and data-governance
  facts in the licensing documentation and machine-readable data-rights
  ledger.
- Do not mention private workstation paths or untracked reference locations in
  source, notebooks intended for release, documentation, logs, or package
  artifacts.
- Do not claim that a model is “complete” without naming the precise scope,
  such as thermodynamic Helmholtz terms versus ancillary or transport
  correlations.
- Preserve the generative-AI disclosure and human responsibility statement.

## Copyright, data rights, and release boundary

- Do not vendor books, papers, publisher supplements, source workbooks, or
  third-party source trees.
- Public availability and citation do not grant redistribution permission.
- Every research CSV requires an entry in `tests/data/rights.yaml` with source,
  basis, status, and license where applicable.
- `not-cleared` data must not be committed to a public repository. Keep
  reproducible extraction code that operates on a lawful user-supplied source,
  or replace the data with an openly licensed source.
- Executed notebook tables, markers, and densely reconstructed plots derived
  from `not-cleared` data require the same clearance. Stripping them from PyPI
  alone is not sufficient for a public Git repository.
- Project-generated external-software baselines may be retained when their
  generator, software version, inputs, and third-party notices are recorded.
- Retain attribution, license links, and modification notices for CC BY, MIT,
  NIST, and other reusable sources.
- Review substantial coefficient inventories for copyright, contract, and
  database rights even when individual numerical facts may not be protected.
- Wheels and source distributions must exclude notebooks, tests, scripts,
  research CSVs, workbooks, plots, PDFs, and private reference material.
- Never remove or rewrite legal notices to make a distribution check pass.

Consult `docs/licensing.md`, `THIRD_PARTY_NOTICES.md`, and
`tests/data/rights.yaml` before adding data or parameter tables.

## Reproducible environments and cross-platform support

Use Pixi environments and tasks rather than an ad hoc environment. Keep the
manifest and lock file synchronized. The supported CI baseline includes Linux,
Windows, and `macos-latest`; do not introduce shell-only assumptions into
portable Python code or tests.

Common commands:

```bash
pixi install
pixi run -e default lint
pixi run -e default format-check
pixi run -e default typecheck
pixi run -e default test-cov
pixi run -e default check-data-rights
pixi run -e default sync-deps-check
pixi run -e docs mkdocs build --strict
```

Treat `pixi.toml` as the dependency source of truth. Run `pixi run sync-deps`
after dependency changes; the generated pip metadata intentionally contains
the normal `core` requirements and only the `groups`, `intel`, and `gpu`
extras. Use `pixi run bump-version <version>` to synchronize
`pyproject.toml`, `pixi.toml`, `src/torch_flash/__init__.py`, and
`CITATION.cff`, then refresh and check `pixi.lock`.

For notebooks and optional external comparisons:

```bash
pixi install -e benchmarks
pixi run -e benchmarks notebooks-sync
pixi run -e benchmarks notebooks-run
```

For a release-boundary check:

```bash
pixi run -e default build
pixi run -e default twine check dist/*
pixi run -e default check-dist
```

The exhaustive Whitson Rachford–Rice corpus is optional and supplied through
`TORCH_FLASH_WHITSON_DATA`; do not vendor that external checkout.

Prefer a targeted test while iterating, then run the complete relevant gate.
Documentation-only work does not require the full numerical suite, but it does
require a strict docs build and package metadata check when the README or
release metadata changes.

## Definition of done

Before declaring a task complete, confirm as applicable:

- governing equations and parameter identities match their cited sources;
- units, tensor shapes, dtype, device, and gradients are correct;
- public function/class docstrings fully document behavior and cite the
  defining literature for publication-derived algorithms;
- convergence and physical residuals pass over representative and difficult
  states;
- verification and validation evidence are classified correctly;
- a regression covers every fixed bug;
- branch-aware coverage remains at least 99%;
- notebook source and executed artifact are synchronized and visually
  inspected;
- notebooks consume high-level package APIs and contain no reusable
  thermodynamic, numerical, continuation, derivative, or parallel algorithms;
- experimental/reference data and model curves are visible and correctly
  labeled;
- documentation reflects the current behavior and known limitations;
- data rights and third-party notices are complete;
- wheel and source-distribution boundaries pass;
- Linux, Windows, and macOS implications were considered; and
- the final response reports what ran, what passed, and anything that remains
  uncertain.
