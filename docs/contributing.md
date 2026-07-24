# Contributing

The complete repository contribution policy is available in
[CONTRIBUTING.md](https://github.com/ThermoPhase-FCSRG/torch-flash/blob/main/CONTRIBUTING.md).

Scientific additions need primary-source provenance, explicit SI conversion,
model/parameter identity, independent numerical validation, and stated
limitations. Pull requests must retain the 99% branch-aware coverage gate.

## Development environments

Install the default environment for source, lint, typing, and tests. Install
the benchmark environment only when working with notebooks or external
comparison packages:

```bash
pixi install
pixi install -e benchmarks
```

The common quality gates are:

```bash
pixi run lint
pixi run format-check
pixi run typecheck
pixi run test-cov
pixi run sync-deps-check
pixi run -e docs mkdocs build --strict
```

## Dependency and release metadata

`pixi.toml` is the source of truth for dependency constraints. After editing
it, regenerate and verify the pip-facing metadata:

```bash
pixi run sync-deps
pixi run sync-deps-check
pixi lock --check
```

The `core` feature becomes the dependency set installed by
`pip install torch-flash`. The public pip extras are limited to:

| Capability | Pip installation | Pixi environment |
| --- | --- | --- |
| Default runtime | `pip install torch-flash` | `default` |
| Group identification | `pip install "torch-flash[groups]"` | `groups` |
| Intel sparse solver | `pip install "torch-flash[intel]"` | `intel` |
| GPU linear algebra | `pip install "torch-flash[gpu]"` | `gpu` |

`default` is the ordinary installation, not a pip extra. Test, development,
notebook, documentation, external-comparison, and benchmark dependencies stay
in Pixi. The `intel` and `gpu` extras currently apply to Linux and Windows, and
GPU execution requires a compatible CUDA runtime and device. Conda-only
interpreter, runtime, and accelerator selectors are filtered from pip
metadata; `scripts/sync_deps.py` records the intentional package-name,
version, and platform-marker translations.

Synchronize a release version with:

```bash
pixi run bump-version 0.1.3 --dry-run
pixi run bump-version 0.1.3
pixi lock
```

This updates the package metadata in `pyproject.toml`, the Pixi workspace,
`torch_flash.__version__`, and `CITATION.cff` together. The task stops if their
current versions differ, preventing a new bump from concealing an existing
metadata mismatch.

Before tagging `v<version>`, verify:

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

The distribution checks enforce the public package boundary: notebooks,
tests, scripts, research datasets, and rendered research artifacts are not
included in wheels or source distributions.
