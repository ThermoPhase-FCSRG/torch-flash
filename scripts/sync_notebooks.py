"""Synchronize paired Jupytext notebooks and percent-format sources."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def main() -> int:
    """Synchronize all paired notebooks in the repository."""
    notebooks = sorted(
        NOTEBOOKS_DIR.rglob("*.ipynb"),
        key=lambda path: (path.name, path.as_posix()),
    )
    ipynb_files = [str(path) for path in notebooks]
    if not ipynb_files:
        print("No notebooks found to sync.")
        return 0

    command = ["jupytext", "--sync", *ipynb_files]
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
