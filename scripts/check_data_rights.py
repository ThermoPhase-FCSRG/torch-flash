"""Audit research CSVs and public figures against their rights decisions."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import yaml

VALID_STATUSES = frozenset({"open", "generated", "not-cleared"})
REQUIRED_FIELDS = frozenset({"status", "basis", "source"})
PUBLIC_FIGURE_STATUSES = frozenset({"open", "generated"})
FIGURE_EVIDENCE_CLASSES = frozenset({"application", "validation", "verification"})
AGGREGATE_PUBLICATION_FORM = "non-reconstructive-aggregate"
RENDERED_PLOT_PUBLICATION_FORM = "rendered-plot"


def audit_data_rights(data_directory: Path, ledger_path: Path) -> tuple[str, ...]:
    """Return coverage and schema violations for the research-data ledger."""
    document: Any
    with ledger_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        return (f"{ledger_path}: root must be a mapping",)
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, dict):
        return (f"{ledger_path}: artifacts must be a mapping",)

    violations: list[str] = []
    csv_paths_by_name: dict[str, list[Path]] = {}
    for path in data_directory.rglob("*.csv"):
        csv_paths_by_name.setdefault(path.name, []).append(path)
    csv_names = set(csv_paths_by_name)
    ledger_names = set(artifacts)
    for name, paths in sorted(csv_paths_by_name.items()):
        if len(paths) > 1:
            relative_paths = ", ".join(
                path.relative_to(data_directory).as_posix() for path in sorted(paths)
            )
            violations.append(f"{data_directory}: duplicate CSV basename {name}: {relative_paths}")
    for name in sorted(csv_names - ledger_names):
        violations.append(f"{ledger_path}: missing rights record for {name}")
    for name in sorted(ledger_names - csv_names):
        record = artifacts[name]
        if not isinstance(record, dict) or record.get("status") != "not-cleared":
            violations.append(f"{ledger_path}: rights record has no CSV: {name}")

    for name, record in sorted(artifacts.items()):
        if not isinstance(name, str) or not name.endswith(".csv"):
            violations.append(f"{ledger_path}: invalid artifact name {name!r}")
            continue
        if not isinstance(record, dict):
            violations.append(f"{ledger_path}: {name} record must be a mapping")
            continue
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            violations.append(f"{ledger_path}: {name} missing {', '.join(sorted(missing))}")
        status = record.get("status")
        if status not in VALID_STATUSES:
            violations.append(f"{ledger_path}: {name} has invalid status {status!r}")
        for path in csv_paths_by_name.get(name, ()):
            relative_path = path.relative_to(data_directory)
            relative_label = relative_path.as_posix()
            if status == "not-cleared" and path.parent != data_directory / "not-cleared":
                violations.append(f"{relative_label}: not-cleared CSV must be under not-cleared/")
            if status in PUBLIC_FIGURE_STATUSES and path.parent != data_directory:
                violations.append(
                    f"{relative_label}: {status} CSV must be directly under {data_directory}"
                )
        source = record.get("source")
        if isinstance(source, str) and not source.startswith(("http://", "https://")):
            if not Path(source).is_file():
                violations.append(f"{ledger_path}: {name} source does not exist: {source}")
        if (
            status == "open"
            and not record.get("license")
            and "NIST" not in str(record.get("basis", ""))
        ):
            violations.append(f"{ledger_path}: {name} open record requires a license")

    return tuple(violations)


def audit_public_figure_manifest(
    figure_manifest_path: Path,
    ledger_path: Path,
) -> tuple[str, ...]:
    """Return violations in the public figure/data dependency manifest."""
    with ledger_path.open(encoding="utf-8") as stream:
        ledger: Any = yaml.safe_load(stream)
    with figure_manifest_path.open(encoding="utf-8") as stream:
        manifest: Any = yaml.safe_load(stream)
    artifacts = ledger.get("artifacts") if isinstance(ledger, dict) else None
    figures = manifest.get("figures") if isinstance(manifest, dict) else None
    if not isinstance(artifacts, dict):
        return (f"{ledger_path}: artifacts must be a mapping",)
    if not isinstance(figures, dict):
        return (f"{figure_manifest_path}: figures must be a mapping",)

    violations: list[str] = []
    for name, record in sorted(figures.items()):
        figure_path = figure_manifest_path.parent / name
        if not figure_path.is_file():
            violations.append(f"{figure_manifest_path}: missing public figure {name}")
        if not isinstance(record, dict):
            violations.append(f"{figure_manifest_path}: {name} record must be a mapping")
            continue
        evidence_class = record.get("evidence_class")
        if evidence_class not in FIGURE_EVIDENCE_CLASSES:
            violations.append(
                f"{figure_manifest_path}: {name} has invalid evidence class {evidence_class!r}"
            )
        dependencies = record.get("data_artifacts")
        if not isinstance(dependencies, list) or not all(
            isinstance(dependency, str) for dependency in dependencies
        ):
            violations.append(
                f"{figure_manifest_path}: {name} data_artifacts must be a string list"
            )
            continue
        source_material = record.get("source_material", [])
        if not isinstance(source_material, list) or not all(
            isinstance(source, str) for source in source_material
        ):
            violations.append(
                f"{figure_manifest_path}: {name} source_material must be a string list"
            )
            source_material = []
        if evidence_class == "validation" and not dependencies and not source_material:
            violations.append(
                f"{figure_manifest_path}: {name} validation requires experimental data"
            )
        publication_form = record.get("publication_form")
        observation_level_values = record.get("observation_level_values")
        observation_level_markers = record.get("observation_level_markers")
        machine_readable_values = record.get("machine_readable_observation_values")
        data_tables_distributed = record.get("data_tables_distributed")
        for dependency in dependencies:
            rights_record = artifacts.get(dependency)
            if not isinstance(rights_record, dict):
                violations.append(
                    f"{figure_manifest_path}: {name} has unclassified dependency {dependency}"
                )
                continue
            status = rights_record.get("status")
            aggregate_only = (
                status == "not-cleared"
                and publication_form == AGGREGATE_PUBLICATION_FORM
                and observation_level_values is False
            )
            rendered_plot_only = (
                status == "not-cleared"
                and publication_form == RENDERED_PLOT_PUBLICATION_FORM
                and isinstance(observation_level_markers, bool)
                and machine_readable_values is False
                and data_tables_distributed is False
            )
            if (
                status not in PUBLIC_FIGURE_STATUSES
                and not aggregate_only
                and not rendered_plot_only
            ):
                violations.append(
                    f"{figure_manifest_path}: {name} depends on {dependency} "
                    f"with status {status!r} outside the declared figure-only boundary"
                )
    return tuple(violations)


def audit_tracked_data_rights(
    tracked_paths: tuple[str, ...],
    ledger_path: Path,
) -> tuple[str, ...]:
    """Reject tracked research CSVs whose redistribution is not cleared."""
    with ledger_path.open(encoding="utf-8") as stream:
        ledger: Any = yaml.safe_load(stream)
    artifacts = ledger.get("artifacts") if isinstance(ledger, dict) else None
    if not isinstance(artifacts, dict):
        return (f"{ledger_path}: artifacts must be a mapping",)

    violations: list[str] = []
    for tracked_path in tracked_paths:
        name = Path(tracked_path).name
        record = artifacts.get(name)
        if isinstance(record, dict) and record.get("status") == "not-cleared":
            violations.append(f"{tracked_path}: not-cleared research data must not be tracked")
    return tuple(violations)


def _tracked_repository_paths(repository_root: Path) -> tuple[str, ...]:
    """Return Git-tracked paths, or none outside a Git working tree."""
    result = subprocess.run(
        ["git", "ls-files", "--", "tests/data"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return ()
    return tuple(line for line in result.stdout.splitlines() if line)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("tests/data"))
    parser.add_argument("--ledger", type=Path, default=Path("tests/data/rights.yaml"))
    parser.add_argument(
        "--public-figures",
        type=Path,
        default=Path("docs/assets/validation/manifest.yaml"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    return parser


def main() -> int:
    """Audit the ledger and return a shell status."""
    args = _parser().parse_args()
    violations = (
        *audit_data_rights(args.data, args.ledger),
        *audit_public_figure_manifest(args.public_figures, args.ledger),
        *audit_tracked_data_rights(
            _tracked_repository_paths(args.repository_root),
            args.ledger,
        ),
    )
    if violations:
        for violation in violations:
            print(violation)
        return 1
    print("research-data rights ledger and public-figure boundary audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
