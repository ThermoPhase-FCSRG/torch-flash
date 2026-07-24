# Contributing

Contributions should preserve scientific traceability as well as software
quality. New models or parameter sets need a primary reference, explicit units,
the exact component and validity scope, and an independent numerical
comparison. Do not describe a generic Helmholtz or cubic form as a named model
without the complete published coefficient set.

## Development workflow

Install Pixi, then create the default and benchmark environments:

```bash
pixi install
pixi install -e benchmarks
```

Before opening a pull request:

```bash
pixi run lint
pixi run format-check
pixi run typecheck
pixi run test-cov
pixi run sync-deps-check
pixi run build
pixi run check-dist
pixi run -e benchmarks notebooks-sync
```

Tests must retain at least 99% branch-aware coverage. Numerical tests should
state why their tolerances are physically and computationally appropriate.
External software should normally be used to generate a versioned, documented
CSV baseline; the normal test suite must remain independent of that software.
Before committing literature-derived data, record the source, the exact
extraction or transformation, and an explicit redistribution license or
permission. A citation establishes scientific provenance but does not by
itself grant redistribution rights. If permission is unclear, use a
user-supplied/downloaded source outside version control and keep only the
reproducible extraction code.

Notebooks are paired through Jupytext. Edit the `.py` source, synchronize, and
execute the notebook from top to bottom. Record package versions, SI units,
reference provenance, assumptions, and limitations.

## Dependency metadata

Declare dependencies in `pixi.toml`; do not maintain parallel constraints
manually in `pyproject.toml`. Synchronize the pip-facing metadata after changing
a dependency:

```bash
pixi run sync-deps
pixi run sync-deps-check
pixi lock --check
```

The ordinary `pip install torch-flash` dependencies come from Pixi's `core`
feature. Only the `groups`, `intel`, and `gpu` features are exported as pip
extras:

```bash
python -m pip install "torch-flash[groups]"
python -m pip install "torch-flash[intel]"
python -m pip install "torch-flash[gpu]"
```

There is no `default` extra: the default capability is the package's normal
dependency set. Test, development, notebook, documentation,
external-comparison, and benchmark features remain Pixi-only. The Python
interpreter and Conda-only accelerator selectors such as `mkl`, `cuda-version`,
and `pytorch-gpu` are not copied into pip metadata; their pip-installable
companion packages are exported where applicable. Name or specifier
translations belong in `scripts/sync_deps.py`. The `intel` and `gpu` extras
currently target Linux and Windows; GPU execution also requires a compatible
CUDA runtime and device.

## Version and release metadata

Use the version task instead of editing version strings individually:

```bash
pixi run bump-version 0.1.3 --dry-run
pixi run bump-version 0.1.3
pixi lock
```

The task updates `pyproject.toml`, `pixi.toml`,
`src/torch_flash/__init__.py`, and `CITATION.cff`. It refuses to proceed if
those files already disagree, so resolve any inconsistency deliberately before
starting a new bump.

Before creating the matching `v<version>` tag, run:

```bash
pixi run sync-deps-check
pixi lock --check
pixi run lint
pixi run format-check
pixi run typecheck
pixi run test-cov
pixi run build
pixi run -e default twine check dist/*
pixi run check-dist
```

## Pull requests

Keep changes scoped by thermodynamic domain. Describe:

- the governing equations and literature source;
- parameter sources and unit conversions;
- numerical method and failure diagnostics;
- independent validation and observed errors;
- known validity limits and unimplemented terms.
- the license or written permission for every new bundled parameter table or
  repository data fixture.

By contributing, you agree that your work is distributed under LGPL-2.1-only.
