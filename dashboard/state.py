"""Shared, cached application state for the Streamlit dashboard.

``st.cache_resource`` objects are created once per Streamlit *server*, not per
browser session, which is exactly what the monitor needs: the background
capture thread keeps running while pages re-render and even while the browser
reloads.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from src.config.settings import Settings, get_settings
from src.database.database import Database, get_database
from src.inference.predictor import Predictor
from src.network.monitor import MonitorService
from src.utils.errors import NIDSError
from src.utils.logger import configure_from_mapping

@st.cache_resource(show_spinner=False)
def load_settings() -> Settings:
    """Load ``config/config.yaml`` once and configure logging."""
    settings = get_settings()
    configure_from_mapping(settings.section("logging"), settings.path("log_dir"))
    settings.ensure_directories()
    return settings


@st.cache_resource(show_spinner=False)
def _raw_database() -> Database:
    """The real database. The monitor writes through this one directly."""
    return get_database(load_settings())


# ---------------------------------------------------------------- read cache
# Streamlit reruns a whole page on every click, and the heavy aggregates
# (statistics, protocol mix, top talkers, timelines) scan every flow - about
# a second each on a 130k-flow database. The data only changes when the
# monitor writes rows or the UI edits something, so reads are cached against a
# cheap *data version*:
#   * monitor stopped - the newest row id of each table plus a counter that UI
#     writes bump. Unchanged data is never re-queried.
#   * monitor running - a time bucket of ``dashboard.refresh_interval``
#     seconds, so pages refresh at the live rate but never query faster.

#: Database methods that only read. Every other method is treated as a write.
_READ_METHODS = frozenset({
    "statistics", "attack_distribution", "class_distribution", "timeline",
    "protocol_distribution", "top_talkers", "recent_predictions", "recent_alerts",
    "recent_events", "recent_behaviour_events", "get_prediction", "query", "counts",
    "list_sessions", "export_session",
})
_VERSION_SQL = (
    "SELECT (SELECT MAX(id) FROM flows) AS f, (SELECT MAX(id) FROM predictions) AS p, "
    "(SELECT MAX(id) FROM alerts) AS a, (SELECT MAX(id) FROM sessions) AS s, "
    "(SELECT COUNT(ended_at) FROM sessions) AS se, (SELECT MAX(id) FROM system_events) AS e, "
    "(SELECT MAX(id) FROM behaviour_events) AS b"
)


@st.cache_resource(show_spinner=False)
def _write_counter() -> dict[str, int]:
    return {"n": 0}


@st.cache_data(show_spinner=False, ttl=1)
def _stored_version() -> tuple:
    """Newest row ids - primary-key lookups, effectively free."""
    try:
        row = _raw_database().query(_VERSION_SQL)
        return tuple(None if v != v else v for v in row.iloc[0].tolist())  # NaN -> None
    except Exception:  # noqa: BLE001 - an unreadable version just means no caching
        return (time.time(),)


def _data_version() -> tuple:
    writes = _write_counter()["n"]
    try:
        monitor = load_monitor()
        if monitor.is_running:
            interval = max(1.0, float(load_settings().get("dashboard.refresh_interval", 5)))
            return ("live", int(time.time() // interval), writes)
    except Exception:  # noqa: BLE001
        pass
    return ("stored", _stored_version(), writes)


@st.cache_data(show_spinner=False, max_entries=512)
def _cached_read(method: str, args: tuple, kwargs: tuple, version: tuple) -> Any:
    # ``version`` is only part of the cache key.
    return getattr(_raw_database(), method)(*args, **dict(kwargs))


class CachedDatabase:
    """Drop-in for :class:`Database`: reads are cached, writes pass straight
    through and invalidate the cache. Results are copies, safe to mutate."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._database, name)
        if not callable(attr):
            return attr
        if name in _READ_METHODS:
            def read(*args: Any, **kwargs: Any) -> Any:
                return _cached_read(name, args, tuple(sorted(kwargs.items())), _data_version())
            return read

        def write(*args: Any, **kwargs: Any) -> Any:
            try:
                return attr(*args, **kwargs)
            finally:
                _write_counter()["n"] += 1
        return write


@st.cache_resource(show_spinner=False)
def load_database() -> CachedDatabase:
    """The database as pages see it: cached reads, pass-through writes."""
    return CachedDatabase(_raw_database())


@st.cache_resource(show_spinner=False)
def load_monitor() -> MonitorService:
    settings = load_settings()
    return MonitorService(settings, database=_raw_database())


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


@st.cache_resource(show_spinner=False)
def warm_up_in_background() -> bool:
    """Build the slow, server-wide resources on a background thread once per
    server, so the first visitor to AI Explanation does not wait ~7 s for the
    SHAP/LIME explainers. A page that needs one before it is ready simply
    waits on the same cache entry; nothing is built twice."""
    import threading

    def _work() -> None:
        try:
            load_explainer()
        except Exception:  # noqa: BLE001 - the page reports real errors itself
            pass

    threading.Thread(target=_work, name="netlens-warmup", daemon=True).start()
    return True


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
    """Inject the Axelar stylesheet (tokens + components) once per rerun."""
    from dashboard.styles.theme import inject_css

    inject_css()


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
