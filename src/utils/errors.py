"""Project-wide exception hierarchy.

Every module raises one of these instead of a bare ``Exception`` so that the
dashboard and the CLI scripts can present an actionable message to the user
while the full traceback goes to the log file.
"""

from __future__ import annotations


class NIDSError(Exception):
    """Base class for every error raised by this application."""

    #: Short, user-facing hint shown in the dashboard. Subclasses override it.
    hint: str = ""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if hint is not None:
            self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.hint:
            return f"{self.message}  ({self.hint})"
        return self.message


class ConfigurationError(NIDSError):
    """Raised when the configuration file is missing, malformed or invalid."""

    hint = "Check config/config.yaml against the shipped default."


class DatasetError(NIDSError):
    """Raised when the dataset cannot be found, read or understood."""

    hint = "Verify paths.raw_data_dir and that the CSV files are readable."


class PreprocessingError(NIDSError):
    """Raised when the preprocessing pipeline cannot be fitted or applied."""

    hint = "The incoming records do not match the persisted feature contract."


class ModelError(NIDSError):
    """Raised when a model artefact is missing, corrupted or incompatible."""

    hint = "Re-run `python scripts/train_model.py` to rebuild the artefacts."


class FeatureContractError(NIDSError):
    """Raised when a record violates the canonical feature contract."""

    hint = "Feature names/order must match src/features/contract.py."


class CaptureError(NIDSError):
    """Raised when live packet capture cannot be started or sustained."""

    hint = (
        "On Windows install Npcap (https://npcap.com) and run the terminal "
        "as Administrator."
    )


class DatabaseError(NIDSError):
    """Raised when the SQLite store cannot be opened or written."""

    hint = "Check paths.database_path and that the file is not read-only."


class ExplainerError(NIDSError):
    """Raised when SHAP or LIME cannot explain a prediction."""

    hint = "Explanations are optional; detection continues without them."


__all__ = [
    "NIDSError",
    "ConfigurationError",
    "DatasetError",
    "PreprocessingError",
    "ModelError",
    "FeatureContractError",
    "CaptureError",
    "DatabaseError",
    "ExplainerError",
]
