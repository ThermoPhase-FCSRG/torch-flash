import runpy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AUDIT_MODULE = runpy.run_path(str(ROOT / "scripts" / "check_data_rights.py"))
audit_data_rights = AUDIT_MODULE["audit_data_rights"]
audit_public_figure_manifest = AUDIT_MODULE["audit_public_figure_manifest"]
audit_tracked_data_rights = AUDIT_MODULE["audit_tracked_data_rights"]


def test_research_data_rights_ledger_is_complete() -> None:
    assert not audit_data_rights(
        ROOT / "tests" / "data",
        ROOT / "tests" / "data" / "rights.yaml",
    )


def test_public_figures_stay_within_declared_release_boundary() -> None:
    assert not audit_public_figure_manifest(
        ROOT / "docs" / "assets" / "validation" / "manifest.yaml",
        ROOT / "tests" / "data" / "rights.yaml",
    )


def test_public_figure_audit_rejects_uncleared_undeclared_point_level_data(
    tmp_path: Path,
) -> None:
    (tmp_path / "unsafe.png").write_bytes(b"placeholder")
    manifest = {
        "figures": {
            "unsafe.png": {
                "evidence_class": "validation",
                "publication_form": "point-level",
                "observation_level_values": True,
                "data_artifacts": ["pedersen_2024_gerg_z.csv"],
            }
        }
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    violations = audit_public_figure_manifest(
        manifest_path,
        ROOT / "tests" / "data" / "rights.yaml",
    )

    assert any("outside the declared figure-only boundary" in item for item in violations)


def test_public_figure_audit_allows_declared_rendered_plot(
    tmp_path: Path,
) -> None:
    (tmp_path / "plot.png").write_bytes(b"placeholder")
    manifest = {
        "figures": {
            "plot.png": {
                "evidence_class": "validation",
                "publication_form": "rendered-plot",
                "observation_level_markers": True,
                "machine_readable_observation_values": False,
                "data_tables_distributed": False,
                "data_artifacts": ["pedersen_2024_gerg_z.csv"],
            }
        }
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    assert not audit_public_figure_manifest(
        manifest_path,
        ROOT / "tests" / "data" / "rights.yaml",
    )


def test_public_figure_audit_rejects_rendered_plot_with_table_values(
    tmp_path: Path,
) -> None:
    (tmp_path / "unsafe.png").write_bytes(b"placeholder")
    manifest = {
        "figures": {
            "unsafe.png": {
                "evidence_class": "validation",
                "publication_form": "rendered-plot",
                "observation_level_markers": True,
                "machine_readable_observation_values": False,
                "data_tables_distributed": True,
                "data_artifacts": ["pedersen_2024_gerg_z.csv"],
            }
        }
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    violations = audit_public_figure_manifest(
        manifest_path,
        ROOT / "tests" / "data" / "rights.yaml",
    )

    assert any("outside the declared figure-only boundary" in item for item in violations)


def test_repository_audit_rejects_tracked_uncleared_csv() -> None:
    violations = audit_tracked_data_rights(
        ("tests/data/not-cleared/pedersen_2024_gerg_z.csv",),
        ROOT / "tests" / "data" / "rights.yaml",
    )

    assert violations == (
        "tests/data/not-cleared/pedersen_2024_gerg_z.csv: "
        "not-cleared research data must not be tracked",
    )


def test_ledger_allows_absent_uncleared_csv(tmp_path: Path) -> None:
    ledger = {
        "artifacts": {
            "private.csv": {
                "status": "not-cleared",
                "basis": "permission not established",
                "source": "https://example.com/private",
            }
        }
    }
    ledger_path = tmp_path / "rights.yaml"
    ledger_path.write_text(yaml.safe_dump(ledger), encoding="utf-8")

    assert not audit_data_rights(tmp_path, ledger_path)


def test_ledger_allows_uncleared_csv_only_in_dedicated_subdirectory(
    tmp_path: Path,
) -> None:
    not_cleared = tmp_path / "not-cleared"
    not_cleared.mkdir()
    (not_cleared / "private.csv").write_text("value\n1\n", encoding="utf-8")
    ledger = {
        "artifacts": {
            "private.csv": {
                "status": "not-cleared",
                "basis": "permission not established",
                "source": "https://example.com/private",
            }
        }
    }
    ledger_path = tmp_path / "rights.yaml"
    ledger_path.write_text(yaml.safe_dump(ledger), encoding="utf-8")

    assert not audit_data_rights(tmp_path, ledger_path)


def test_ledger_rejects_uncleared_csv_at_data_root(tmp_path: Path) -> None:
    (tmp_path / "private.csv").write_text("value\n1\n", encoding="utf-8")
    ledger = {
        "artifacts": {
            "private.csv": {
                "status": "not-cleared",
                "basis": "permission not established",
                "source": "https://example.com/private",
            }
        }
    }
    ledger_path = tmp_path / "rights.yaml"
    ledger_path.write_text(yaml.safe_dump(ledger), encoding="utf-8")

    violations = audit_data_rights(tmp_path, ledger_path)

    assert violations == ("private.csv: not-cleared CSV must be under not-cleared/",)


def test_ledger_rejects_open_csv_in_not_cleared_subdirectory(
    tmp_path: Path,
) -> None:
    not_cleared = tmp_path / "not-cleared"
    not_cleared.mkdir()
    (not_cleared / "open.csv").write_text("value\n1\n", encoding="utf-8")
    ledger = {
        "artifacts": {
            "open.csv": {
                "status": "open",
                "basis": "reusable source",
                "license": "CC0-1.0",
                "source": "https://example.com/open",
            }
        }
    }
    ledger_path = tmp_path / "rights.yaml"
    ledger_path.write_text(yaml.safe_dump(ledger), encoding="utf-8")

    violations = audit_data_rights(tmp_path, ledger_path)

    assert violations == (f"not-cleared/open.csv: open CSV must be directly under {tmp_path}",)
