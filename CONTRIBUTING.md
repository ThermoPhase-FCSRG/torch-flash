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
pixi run build
pixi run check-dist
pixi run -e default python scripts/sync_deps.py --check
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
