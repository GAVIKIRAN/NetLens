"""Shared, cached application state for the Streamlit dashboard.

``st.cache_resource`` objects are created once per Streamlit *server*, not per
browser session, which is exactly what the monitor needs: the background
capture thread keeps running while pages re-render and even while the browser
reloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.config.settings import Settings, get_settings
from src.database.database import Database, get_database
from src.inference.predictor import Predictor
from src.network.monitor import MonitorService
from src.utils.errors import NIDSError
from src.utils.logger import configure_from_mapping

STYLE_PATH = Path(__file__).resolve().parent / "styles" / "theme.css"


@st.cache_resource(show_spinner=False)
def load_settings() -> Settings:
    """Load ``config/config.yaml`` once and configure logging."""
    settings = get_settings()
    configure_from_mapping(settings.section("logging"), settings.path("log_dir"))
    settings.ensure_directories()
    return settings


@st.cache_resource(show_spinner=False)
def load_database() -> Database:
    return get_database(load_settings())


@st.cache_resource(show_spinner=False)
def load_monitor() -> MonitorService:
    settings = load_settings()
    return MonitorService(settings, database=load_database())


@st.cache_resource(show_spinner="Loading the trained model ...")
def load_predictor() -> Predictor | None:
    """Load the predictor, or ``None`` when the model has not been trained."""
    settings = load_settings()
    if not Predictor.artefacts_exist(settings):
        return None
    try:
        return Predictor.load(settings)
    except NIDSError as exc:
        st.session_state["model_error"] = str(exc)
        return None


@st.cache_resource(show_spinner="Preparing SHAP and LIME explainers ...")
def load_explainer():
    """Build the explanation service once (SHAP/LIME construction is not free)."""
    predictor = load_predictor()
    if predictor is None:
        return None
    from src.xai.explanation_engine import ExplanationService

    return ExplanationService(predictor, load_settings())


@st.cache_data(show_spinner=False)
def load_training_metrics() -> dict[str, Any] | None:
    from src.training.train import load_metrics

    return load_metrics(load_settings().path("model_dir"))


@st.cache_data(show_spinner=False)
def load_training_metadata() -> dict[str, Any] | None:
    from src.training.train import load_metadata

    return load_metadata(load_settings().path("model_dir"))


@st.cache_data(show_spinner=False)
def load_dataset_profile() -> dict[str, Any] | None:
    from src.data.profiler import load_profile

    return load_profile(load_settings().path("reports_dir"))


def inject_styles() -> None:
    """Inject the dashboard stylesheet exactly once per rerun."""
    try:
        css = STYLE_PATH.read_text(encoding="utf-8")
    except OSError:
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def clear_caches() -> None:
    """Drop cached model/metric artefacts after re-training."""
    for func in (
        load_predictor, load_explainer, load_training_metrics,
        load_training_metadata, load_dataset_profile,
    ):
        try:
            func.clear()
        except Exception:  # pragma: no cover - streamlit version differences
            pass


def session_default(key: str, value: Any) -> Any:
    """``st.session_state.setdefault`` that also returns the value."""
    if key not in st.session_state:
        st.session_state[key] = value
    return st.session_state[key]


__all__ = [
    "load_settings",
    "load_database",
    "load_monitor",
    "load_predictor",
    "load_explainer",
    "load_training_metrics",
    "load_training_metadata",
    "load_dataset_profile",
    "inject_styles",
    "clear_caches",
    "session_default",
]
