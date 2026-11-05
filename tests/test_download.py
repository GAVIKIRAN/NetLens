"""The dataset installer's offline paths.

Only the local-archive path is exercised: the test-suite never touches the
network, so downloading is out of scope here.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import download_dataset as installer  # noqa: E402
from tests.fixture_dataset import write_dataset  # noqa: E402


@pytest.fixture()
def nested_archive(tmp_path: Path) -> Path:
    """A zip laid out like the real MachineLearningCSV.zip (CSVs one folder deep)."""
    source = tmp_path / "src"
    write_dataset(source, files=2, rows_per_file=60, seed=4)
    archive = tmp_path / "MachineLearningCSV.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for csv in sorted(source.glob("*.csv")):
            bundle.write(csv, arcname=f"MachineLearningCVE/{csv.name}")
        bundle.writestr("MachineLearningCVE/readme.txt", "not a csv")
    return archive


def test_install_flattens_csvs_and_ignores_other_members(nested_archive, tmp_path):
    raw = tmp_path / "raw"
    installed = installer.install(nested_archive, raw)

    assert len(installed) == 2
    assert all(path.parent == raw for path in installed)          # flattened
    assert all(path.suffix == ".csv" for path in installed)
    assert not (raw / "readme.txt").exists()                      # non-CSV skipped
    assert not (raw / "MachineLearningCVE").exists()              # no nesting


def test_installed_files_are_readable_by_the_loader(nested_archive, tmp_path, capsys):
    raw = tmp_path / "raw"
    installer.install(nested_archive, raw)
    assert installer.verify(raw) == 0
    output = capsys.readouterr().out
    assert "CSV file(s) in" in output
    assert "label=" in output


def test_verify_reports_an_empty_directory(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert installer.verify(empty) == 1
    assert "No CSV files found" in capsys.readouterr().out


def test_archive_catalogue_is_well_formed():
    assert "MachineLearningCSV.zip" in installer.ARCHIVES
    for name, spec in installer.ARCHIVES.items():
        assert name.endswith(".zip")
        assert int(spec["size_mb"]) > 0
        assert spec["urls"], f"{name} has no mirrors"
        assert all(str(url).startswith("http") for url in spec["urls"])


def test_links_output_includes_the_official_page_and_citation(capsys):
    installer.print_links()
    output = capsys.readouterr().out
    assert installer.LANDING_PAGE in output
    assert "MachineLearningCSV.zip" in output
    assert "Sharafaldin" in output
    assert "--zip" in output


def test_relaxed_tls_is_scoped_to_the_bare_ip_host():
    """Certificate checking is only relaxed for the host that mis-serves its cert."""
    assert installer._ssl_context("https://cicresearch.ca/x.zip") is None
    assert installer._ssl_context("http://205.174.165.80/x.zip") is None
    context = installer._ssl_context("https://205.174.165.80/x.zip")
    assert context is not None and context.check_hostname is False
