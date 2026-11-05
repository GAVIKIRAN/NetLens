"""Configuration loading, validation and environment overrides."""

from __future__ import annotations

import pytest

from src.config.settings import DEFAULT_CONFIG_PATH, Settings
from src.utils.errors import ConfigurationError


def test_default_config_loads():
    settings = Settings.load(DEFAULT_CONFIG_PATH, use_env=False)
    assert settings.get("project.short_name") == "NetLens"
    assert settings.get("inference.benign_label") == "BENIGN"


def test_paths_are_absolute():
    settings = Settings.load(DEFAULT_CONFIG_PATH, use_env=False)
    assert settings.path("raw_data_dir").is_absolute()
    assert settings.path("database_path").is_absolute()


def test_dotted_access_and_defaults():
    settings = Settings.load(DEFAULT_CONFIG_PATH, use_env=False)
    assert settings.get("network.flow_timeout") == 120.0
    assert settings.get("does.not.exist", "fallback") == "fallback"


def test_environment_override(monkeypatch):
    monkeypatch.setenv("NIDS_NETWORK__INTERFACE", "Ethernet 2")
    monkeypatch.setenv("NIDS_INFERENCE__ATTACK_PROBABILITY_THRESHOLD", "0.85")
    settings = Settings.load(DEFAULT_CONFIG_PATH, use_env=True)
    assert settings.get("network.interface") == "Ethernet 2"
    assert settings.get("inference.attack_probability_threshold") == 0.85


def test_risk_weights_must_sum_to_one():
    with pytest.raises(ConfigurationError):
        Settings.load(
            DEFAULT_CONFIG_PATH,
            overrides={"risk": {"weights": {"severity": 0.9, "confidence": 0.9, "volume": 0.9}}},
            use_env=False,
        )


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"risk": {"bands": {"low": 60, "medium": 50, "high": 75}}}, "risk.bands"),
        ({"risk": {"alert_min_band": "SEVERE"}}, "alert_min_band"),
        ({"risk": {"severity": {"DDoS": 1.5}}}, "risk.severity.DDoS"),
        ({"inference": {"attack_probability_threshold": 2.0}}, "attack_probability_threshold"),
        ({"network": {"flow_timeout": 0}}, "network.flow_timeout"),
        ({"network": {"flow_timeout": "soon"}}, "network.flow_timeout"),
    ],
)
def test_bad_values_fail_with_a_clear_message(overrides, fragment):
    with pytest.raises(ConfigurationError, match=fragment.replace(".", r"\.")):
        Settings.load(DEFAULT_CONFIG_PATH, overrides=overrides, use_env=False)


def test_missing_section_is_rejected():
    with pytest.raises(ConfigurationError):
        Settings({"paths": {}})


def test_runtime_set_and_path_resolution(tmp_path):
    settings = Settings.load(DEFAULT_CONFIG_PATH, use_env=False)
    settings.set("paths.model_dir", str(tmp_path / "models"))
    assert settings.path("model_dir") == tmp_path / "models"
