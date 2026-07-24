"""Reject research artifacts and private paths in built Python distributions."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Protocol

PRIVATE_DIRECTORY_NAME = "." + "tmp"
FORBIDDEN_DIRECTORIES = frozenset(
    {"notebooks", "tests", "scripts", ".github", "tmp", PRIVATE_DIRECTORY_NAME}
)
FORBIDDEN_SUFFIXES = frozenset(
    {".csv", ".ipynb", ".jpeg", ".jpg", ".pdf", ".png", ".svg", ".xls", ".xlsx"}
)
TEXT_SUFFIXES = frozenset({".cff", ".cfg", ".ini", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"})
PRIVATE_PATH_TOKEN = PRIVATE_DIRECTORY_NAME.encode()
REQUIRED_LEGAL_FILES = (
    "COPYRIGHT",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "docs/licensing.md",
)


class ArchiveReader(Protocol):
    """Minimal common interface for zip and tar readers."""

    def names(self) -> Iterable[str]:
        """Yield member names."""

    def read(self, name: str) -> bytes:
        """Read one regular-file member."""


class ZipReader:
    """Read a wheel through :mod:`zipfile`."""

    def __init__(self, path: Path) -> None:
        self._archive = zipfile.ZipFile(path)

    def names(self) -> Iterable[str]:
        """Yield regular-file names."""
        return (info.filename for info in self._archive.infolist() if not info.is_dir())

    def read(self, name: str) -> bytes:
        """Read one member."""
        return self._archive.read(name)


class TarReader:
    """Read a source distribution through :mod:`tarfile`."""

    def __init__(self, path: Path) -> None:
        self._archive = tarfile.open(path, mode="r:*")

    def names(self) -> Iterable[str]:
        """Yield regular-file names."""
        return (member.name for member in self._archive.getmembers() if member.isfile())

    def read(self, name: str) -> bytes:
        """Read one member."""
        member = self._archive.getmember(name)
        extracted = self._archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"could not read archive member {name!r}")
        return extracted.read()


def _relative_member(name: str, *, source_distribution: bool) -> PurePosixPath:
    """Remove the conventional top-level source-distribution directory."""
    path = PurePosixPath(name)
    if source_distribution and len(path.parts) > 1:
        return PurePosixPath(*path.parts[1:])
    return path


def audit_distribution(path: Path) -> tuple[str, ...]:
    """Return release-boundary violations found in one built artifact."""
    source_distribution = path.name.endswith(".tar.gz")
    if path.suffix == ".whl":
        archive: ArchiveReader = ZipReader(path)
    elif source_distribution:
        archive = TarReader(path)
    else:
        return (f"{path}: unsupported distribution format",)

    violations: list[str] = []
    names = tuple(archive.names())
    if not names:
        return (f"{path}: archive is empty",)
    relatives = tuple(
        _relative_member(name, source_distribution=source_distribution) for name in names
    )

    for name, relative in zip(names, relatives, strict=True):
        if FORBIDDEN_DIRECTORIES.intersection(relative.parts):
            violations.append(f"{path}: forbidden directory in {relative}")
        if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"{path}: forbidden research artifact {relative}")
        if relative.suffix.lower() in TEXT_SUFFIXES:
            if PRIVATE_PATH_TOKEN in archive.read(name):
                violations.append(f"{path}: private-path token in {relative}")

    package_prefix = (
        PurePosixPath("src/torch_flash") if source_distribution else PurePosixPath("torch_flash")
    )
    if not any(
        relative == package_prefix or package_prefix in relative.parents for relative in relatives
    ):
        violations.append(f"{path}: torch_flash package source is missing")
    for required in REQUIRED_LEGAL_FILES:
        if source_distribution:
            present = PurePosixPath(required) in relatives
        else:
            suffix = f".dist-info/licenses/{required}"
            present = any(relative.as_posix().endswith(suffix) for relative in relatives)
        if not present:
            violations.append(f"{path}: required legal file is missing: {required}")
    return tuple(violations)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distributions", nargs="+", type=Path)
    return parser


def main() -> int:
    """Audit all requested artifacts and return a shell status."""
    args = _parser().parse_args()
    violations = tuple(
        violation
        for distribution in args.distributions
        for violation in audit_distribution(distribution)
    )
    if violations:
        for violation in violations:
            print(violation)
        return 1
    print(f"distribution audit passed for {len(args.distributions)} artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
