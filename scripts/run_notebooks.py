"""Execute every research notebook from top to bottom."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
REQUIRED_EXTERNAL_PACKAGES = ("neqsim", "teqp", "thermopack")


def _missing_external_packages() -> tuple[str, ...]:
    missing = []
    for package in REQUIRED_EXTERNAL_PACKAGES:
        try:
            version(package)
        except PackageNotFoundError:
            missing.append(package)
    return tuple(missing)


def main() -> int:
    """Execute all notebooks in deterministic filename order."""
    notebooks = sorted(
        NOTEBOOKS_DIR.rglob("*.ipynb"),
        key=lambda path: (path.name, path.as_posix()),
    )
    if not notebooks:
        print("No notebooks found to execute.")
        return 0
    missing = _missing_external_packages()
    if missing:
        joined = ", ".join(missing)
        print(
            "The complete notebook suite requires the benchmark backends "
            f"({joined} missing). Run `pixi run -e benchmarks notebooks-run`.",
            file=sys.stderr,
        )
        return 2
    for notebook in notebooks:
        command = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=600",
            str(notebook),
        ]
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
