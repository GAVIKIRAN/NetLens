"""Label repair, using the byte sequences found in real copies of CIC-IDS2017."""

from __future__ import annotations

import pytest

from src.config.settings import DEFAULT_CONFIG_PATH, Settings
from src.inference.risk import RiskScorer
from src.utils.helpers import normalise_label


@pytest.mark.parametrize(
    "raw",
    [
        b"Web Attack \xef\xbf\xbd Brute Force",  # the MachineLearningCSV.zip copy
        b"Web Attack \x96 Brute Force",
        b"Web Attack \xe2\x80\x93 Brute Force",
    ],
)
def test_damaged_web_attack_dashes_are_repaired(raw):
    assert normalise_label(raw.decode("latin-1")) == "Web Attack - Brute Force"


@pytest.mark.parametrize("label", ["SSH-Patator", "FTP-Patator", "BENIGN", "DoS slowloris"])
def test_other_labels_are_left_alone(label):
    assert normalise_label(label) == label


@pytest.mark.parametrize(
    "raw, expected",
    [
        (b"Web Attack \xef\xbf\xbd Brute Force", 0.75),
        (b"Web Attack \xef\xbf\xbd XSS", 0.80),
        (b"Web Attack \xef\xbf\xbd Sql Injection", 0.90),
    ],
)
def test_repaired_labels_get_their_configured_severity(raw, expected):
    settings = Settings.load(DEFAULT_CONFIG_PATH, use_env=False)
    scorer = RiskScorer.from_settings(settings)
    assert scorer.severity_of(normalise_label(raw.decode("latin-1"))) == expected
