"""Configuration loading and access.

The whole application reads its settings through a single :class:`Settings`
object built from ``config/config.yaml``.  Values can be overridden by
environment variables using the ``NIDS_`` prefix and ``__`` as the nesting
separator, which makes the app configurable on Windows without editing files::

    set NIDS_NETWORK__INTERFACE=Ethernet
    set NIDS_PATHS__RAW_DATA_DIR=D:\\datasets\\CICIDS2017
    set NIDS_INFERENCE__ATTACK_PROBABILITY_THRESHOLD=0.75
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from src.utils.errors import ConfigurationError

ENV_PREFIX = "NIDS_"
ENV_SEPARATOR = "__"

#: Repository root (the directory that contains ``config/``).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

#: Keys under ``paths:`` that are resolved relative to :data:`PROJECT_ROOT`.
_PATH_KEYS = (
    "raw_data_dir",
    "processed_data_dir",
    "demo_data_dir",
    "model_dir",
    "preprocessing_dir",
    "reports_dir",
    "log_dir",
    "database_path",
)

#: Minimal top-level sections that must exist for the app to run.
_REQUIRED_SECTIONS = (
    "paths",
    "dataset",
    "features",
    "preprocessing",
    "training",
    "inference",
    "risk",
    "network",
    "xai",
    "database",
    "dashboard",
    "demo",
)


def _coerce(text: str) -> Any:
    """Convert an environment-variable string to bool/int/float/None/str."""
    lowered = text.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", ""}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _deep_update(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` (in place)."""
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _env_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    """Build a nested override dict from ``NIDS_*`` environment variables."""
    overrides: dict[str, Any] = {}
    for raw_key, raw_value in environ.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        path = raw_key[len(ENV_PREFIX) :].lower().split(ENV_SEPARATOR)
        path = [part for part in path if part]
        if not path:
            continue
        cursor = overrides
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):  # pragma: no cover - defensive
                break
        if isinstance(cursor, dict):
            cursor[path[-1]] = _coerce(raw_value)
    return overrides


class Settings:
    """Immutable-ish view over the merged configuration dictionary."""

    def __init__(self, data: Mapping[str, Any], source: Path | None = None) -> None:
        self._data: dict[str, Any] = copy.deepcopy(dict(data))
        self.source = source
        self.project_root = PROJECT_ROOT
        self._validate()
        self._resolve_paths()

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: Mapping[str, Any] | None = None,
        use_env: bool = True,
    ) -> "Settings":
        """Read the YAML file, apply env overrides, then explicit overrides."""
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}",
                hint="Copy the shipped config/config.yaml into place.",
            )
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"config.yaml is not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigurationError("config.yaml must contain a top-level mapping.")

        if use_env:
            _deep_update(data, _env_overrides(os.environ))
        if overrides:
            _deep_update(data, overrides)
        return cls(data, source=config_path)

    # -------------------------------------------------------------- internal
    def _validate(self) -> None:
        missing = [s for s in _REQUIRED_SECTIONS if s not in self._data]
        if missing:
            raise ConfigurationError(
                "config.yaml is missing required section(s): " + ", ".join(missing)
            )
        weights = self._data.get("risk", {}).get("weights", {})
        total = sum(float(weights.get(k, 0.0)) for k in ("severity", "confidence", "volume"))
        if abs(total - 1.0) > 1e-6:
            raise ConfigurationError(
                f"risk.weights must sum to 1.0 (got {total:.4f}); "
                "adjust severity/confidence/volume."
            )
        test_size = self._number("preprocessing.test_size", 0.2)
        val_size = self._number("preprocessing.validation_size", 0.2)
        if not 0.0 < test_size < 0.9:
            raise ConfigurationError("preprocessing.test_size must be in (0, 0.9).")
        if not 0.0 < val_size < 0.9:
            raise ConfigurationError("preprocessing.validation_size must be in (0, 0.9).")

        bands = self._data["risk"].get("bands", {}) or {}
        low = self._number("risk.bands.low", 25)
        medium = self._number("risk.bands.medium", 50)
        high = self._number("risk.bands.high", 75)
        if not 0.0 < low < medium < high < 100.0:
            raise ConfigurationError(
                f"risk.bands must satisfy 0 < low < medium < high < 100 (got {bands})."
            )
        band = str(self._data["risk"].get("alert_min_band", "MEDIUM")).upper()
        if band not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            raise ConfigurationError(
                f"risk.alert_min_band must be LOW, MEDIUM, HIGH or CRITICAL (got '{band}')."
            )
        for label, value in (self._data["risk"].get("severity", {}) or {}).items():
            severity = self._number(f"risk.severity.{label}", value)
            if not 0.0 <= severity <= 1.0:
                raise ConfigurationError(
                    f"risk.severity.{label} must be between 0 and 1 (got {severity})."
                )
        threshold = self._number("inference.attack_probability_threshold", 0.6)
        if not 0.0 <= threshold <= 1.0:
            raise ConfigurationError(
                "inference.attack_probability_threshold must be between 0 and 1 "
                f"(got {threshold})."
            )
        for key, default in (
            ("flow_timeout", 120.0), ("activity_timeout", 5.0), ("subflow_timeout", 1.0),
        ):
            if self._number(f"network.{key}", default) <= 0.0:
                raise ConfigurationError(f"network.{key} must be greater than 0.")
        models = (self._data["training"].get("models", {}) or {})
        if models and not any((cfg or {}).get("enabled", True) for cfg in models.values()):
            raise ConfigurationError("training.models must enable at least one model.")

    def _number(self, dotted: str, default: Any) -> float:
        """Read a numeric setting, turning a bad value into a clear error."""
        value = self.get(dotted, default)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{dotted} must be a number (got {value!r}).") from exc

    def _resolve_paths(self) -> None:
        paths = self._data.setdefault("paths", {})
        for key in _PATH_KEYS:
            value = paths.get(key)
            if value is None:
                continue
            candidate = Path(str(value)).expanduser()
            if not candidate.is_absolute():
                candidate = (self.project_root / candidate).resolve()
            paths[key] = str(candidate)
        demo = self._data.setdefault("demo", {})
        src_file = demo.get("source_file")
        if src_file:
            candidate = Path(str(src_file)).expanduser()
            if not candidate.is_absolute():
                candidate = (self.project_root / candidate).resolve()
            demo["source_file"] = str(candidate)

    # ------------------------------------------------------------- accessors
    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value with a dotted path, e.g. ``"network.interface"``."""
        cursor: Any = self._data
        for part in dotted.split("."):
            if isinstance(cursor, Mapping) and part in cursor:
                cursor = cursor[part]
            else:
                return default
        return cursor

    def section(self, name: str) -> dict[str, Any]:
        """Return a *copy* of a top-level section (never ``None``)."""
        value = self._data.get(name, {})
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    def path(self, key: str) -> Path:
        """Return one of the resolved ``paths:`` entries as a :class:`Path`."""
        value = self._data.get("paths", {}).get(key)
        if value is None:
            raise ConfigurationError(f"paths.{key} is not configured.")
        return Path(value)

    def set(self, dotted: str, value: Any) -> None:
        """Override a value at runtime (used by the Settings dashboard page)."""
        parts = dotted.split(".")
        cursor = self._data
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[parts[-1]] = value
        if parts[0] == "paths" or dotted == "demo.source_file":
            self._resolve_paths()

    def as_dict(self) -> dict[str, Any]:
        """Deep copy of the whole configuration."""
        return copy.deepcopy(self._data)

    def ensure_directories(self, keys: Iterable[str] | None = None) -> None:
        """Create the configured directories (and the DB parent) if missing."""
        keys = keys or (
            "raw_data_dir",
            "processed_data_dir",
            "demo_data_dir",
            "model_dir",
            "preprocessing_dir",
            "reports_dir",
            "log_dir",
        )
        for key in keys:
            try:
                self.path(key).mkdir(parents=True, exist_ok=True)
            except ConfigurationError:
                continue
        try:
            self.path("database_path").parent.mkdir(parents=True, exist_ok=True)
        except ConfigurationError:
            pass

    # ---------------------------------------------------------------- dunder
    def __contains__(self, key: str) -> bool:  # pragma: no cover - trivial
        return key in self._data

    def __getitem__(self, key: str) -> Any:  # pragma: no cover - trivial
        return self._data[key]

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<Settings source={self.source} sections={sorted(self._data)}>"


_CACHED: Settings | None = None


def get_settings(path: str | Path | None = None, reload: bool = False) -> Settings:
    """Process-wide singleton accessor for :class:`Settings`."""
    global _CACHED
    if _CACHED is None or reload or path is not None:
        _CACHED = Settings.load(path)
    return _CACHED


__all__ = ["Settings", "get_settings", "PROJECT_ROOT", "DEFAULT_CONFIG_PATH"]
