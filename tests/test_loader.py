"""Dataset discovery, loading, cleaning and profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import add_binary_label, clean_dataset
from src.data.loader import detect_label_column, discover_csv_files, load_dataset
from src.data.profiler import profile_dataset, save_profile
from src.data.splitter import split_dataset
from src.utils.errors import DatasetError


def test_discovery_finds_every_csv(raw_dataset):
    files = discover_csv_files(raw_dataset)
    assert len(files) == 2
    assert all(path.suffix.lower() == ".csv" for path in files)


def test_discovery_raises_on_missing_directory(tmp_path):
    with pytest.raises(DatasetError):
        discover_csv_files(tmp_path / "nope")


def test_discovery_raises_on_empty_directory(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(DatasetError):
        discover_csv_files(tmp_path / "empty")


def test_label_column_detection():
    assert detect_label_column(["flow_duration", "label"]) == "label"
    with pytest.raises(DatasetError):
        detect_label_column(["a", "b"])


def test_load_dataset_subsamples_and_canonicalises(raw_dataset):
    result = load_dataset(raw_dataset, max_rows_per_class=120, max_total_rows=600)
    assert result.rows_loaded <= 600
    assert result.label_column == "label"
    assert "flow_duration" in result.frame.columns
    assert "flow_bytes_s" in result.frame.columns
    assert result.total_rows_available >= result.rows_loaded
    assert len(result.class_counts_loaded) >= 2
    # every class capped
    assert max(result.class_counts_loaded.values()) <= 120


def test_cleaning_removes_identifiers_and_degenerate_columns(raw_dataset):
    result = load_dataset(raw_dataset, max_rows_per_class=80, max_total_rows=400)
    frame, report = clean_dataset(result.frame, result.label_column)

    for dropped in ("flow_id", "source_ip", "destination_ip", "source_port",
                    "timestamp", "cwe_flag_count", "fwd_header_length_1",
                    "fwd_avg_bytes_bulk"):
        assert dropped not in frame.columns
        assert dropped in report.columns_dropped

    assert report.rows_out <= report.rows_in
    assert report.duplicates_removed >= 0
    numeric = frame.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy(dtype=float, na_value=0.0)).any()


def test_cleaning_fixes_the_cp1252_dash_in_web_attack_labels(raw_dataset):
    result = load_dataset(raw_dataset, max_rows_per_class=200, max_total_rows=1400)
    frame, _report = clean_dataset(result.frame, result.label_column,
                                   min_samples_per_class=1)
    labels = set(frame[result.label_column])
    assert not any("\x96" in label for label in labels)
    assert any(label.startswith("Web Attack - ") for label in labels)
    # ASCII hyphens in the published class names survive untouched
    assert "SSH-Patator" in labels or "FTP-Patator" in labels


def test_cleaning_merges_rare_classes():
    frame = pd.DataFrame(
        {
            "flow_duration": np.arange(60, dtype=float),
            "total_fwd_packets": np.arange(60, dtype=float),
            "label": ["BENIGN"] * 55 + ["Heartbleed"] * 5,
        }
    )
    cleaned, report = clean_dataset(frame, "label", min_samples_per_class=10)
    assert "Heartbleed" in report.merged_rare_classes
    assert "Rare_Attack" in set(cleaned["label"])


def test_cleaning_rejects_a_single_class_dataset():
    frame = pd.DataFrame({"flow_duration": [1.0, 2.0, 3.0], "label": ["BENIGN"] * 3})
    with pytest.raises(DatasetError):
        clean_dataset(frame, "label")


def test_binary_label_helper():
    frame = pd.DataFrame({"label": ["BENIGN", "DDoS"]})
    out = add_binary_label(frame, "label")
    assert list(out["binary_label"]) == ["BENIGN", "ATTACK"]


def test_profiler_reports_real_statistics(raw_dataset, tmp_path):
    result = load_dataset(raw_dataset, max_rows_per_class=80, max_total_rows=400)
    frame, _ = clean_dataset(result.frame, result.label_column)
    profile = profile_dataset(frame, result.label_column)

    assert profile.n_records == len(frame)
    assert profile.n_features == frame.shape[1] - 1
    assert sum(profile.class_distribution.values()) == len(frame)
    assert profile.imbalance_ratio >= 1.0
    assert profile.feature_statistics
    path = save_profile(profile, tmp_path)
    assert path.exists()


def test_split_is_disjoint_and_stratified(raw_dataset):
    result = load_dataset(raw_dataset, max_rows_per_class=120, max_total_rows=700)
    frame, _ = clean_dataset(result.frame, result.label_column)
    features = frame.drop(columns=[result.label_column])
    labels = frame[result.label_column]

    splits = split_dataset(features, labels, test_size=0.2, validation_size=0.2)
    sizes = splits.sizes()
    assert sum(sizes.values()) == len(frame)
    assert sizes["test"] == pytest.approx(len(frame) * 0.2, rel=0.15)
    # Every split sees more than one class
    for part in splits.class_distribution().values():
        assert len(part) >= 2
