"""XAI-NIDS dashboard entry point.

Run it from the project root with::

    streamlit run app.py

The dashboard is a *read-mostly* view over the SQLite store plus a control
surface for the monitoring service.  All detection work happens in
:mod:`src`; nothing in ``dashboard/`` trains, captures or classifies on its own.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from dashboard.components import metrics  # noqa: E402
from dashboard.components.refresh import fragment  # noqa: E402
from dashboard.pages_impl import (  # noqa: E402
    ai_explanation,
    alert_center,
    live_monitor,
    model_performance,
    overview,
    sessions,
    settings_page,
    traffic_analytics,
)
from dashboard.state import (  # noqa: E402
    inject_styles,
    load_monitor,
    load_predictor,
    load_settings,
)
from src.utils.errors import NIDSError  # noqa: E402

PAGES = {
    "Network Security Overview": overview,
    "Live Monitor": live_monitor,
    "Sessions": sessions,
    "Alert Center": alert_center,
    "AI Explanation": ai_explanation,
    "Model Performance": model_performance,
    "Traffic Analytics": traffic_analytics,
    "System Settings": settings_page,
}

#: Page -> (short name, Material icon, one-line description).
PAGE_INFO = {
    "Network Security Overview": (
        "Overview", "shield", "Threat level, attack share and alerts across classified flows",
    ),
    "Live Monitor": (
        "Live Monitor", "monitoring", "Start monitoring and watch flows being classified",
    ),
    "Sessions": (
        "Sessions", "history", "Every monitoring run, saved and downloadable as CSV",
    ),
    "Alert Center": (
        "Alert Center", "notifications", "Filter, review and export stored alerts",
    ),
    "AI Explanation": (
        "AI Explanation", "account_tree", "Why the model classified a stored flow the way it did",
    ),
    "Model Performance": (
        "Model Performance", "query_stats", "Measured results for every trained model",
    ),
    "Traffic Analytics": (
        "Traffic Analytics", "lan", "Protocols, top talkers and traffic trends",
    ),
    "System Settings": (
        "System Settings", "tune", "Capture, detection and data retention settings",
    ),
}


def _nav_label(name: str) -> str:
    short, icon, _ = PAGE_INFO[name]
    return f":material/{icon}: {short}"


def _system_status() -> None:
    status = load_monitor().status()
    stats = status["stats"]
    predictor = load_predictor()
    metrics.system_card(
        bool(status["running"]),
        str(status["mode"]),
        int(stats["flows_processed"]),
        int(stats["attacks_detected"]),
        int(stats["alerts_raised"]),
        predictor.model_name if predictor else "",
        len(predictor.classes) if predictor else None,
    )
    if predictor is None:
        st.caption("Run `python scripts/train_model.py`")


def _sidebar(settings) -> str:
    with st.sidebar:
        metrics.brand(
            str(settings.get("project.short_name", "NetLens")),
            "Explainable network intrusion detection",
        )
        choice = st.radio(
            "Navigation",
            list(PAGES),
            format_func=_nav_label,
            label_visibility="collapsed",
            key="nav",
        )
        st.markdown('<div class="nids-nav-title">System</div>', unsafe_allow_html=True)

        # While monitoring runs, this card refreshes on its own timer so its counters
        # stay live on every page; a stopped monitor has nothing to refresh.
        running = bool(load_monitor().status()["running"])
        interval = float(settings.get("dashboard.refresh_interval", 5)) if running else None
        fragment(run_every=interval)(_system_status)()

        st.markdown(
            '<div class="nids-footnote">Defensive monitoring only. This application '
            "captures and classifies traffic; it never generates, injects or replays "
            f"traffic onto a network.<br>v{settings.get('project.version', '1.0.0')}</div>",
            unsafe_allow_html=True,
        )
    return choice


def main() -> None:
    settings = load_settings()

    st.set_page_config(
        page_title=settings.get("dashboard.title", "NetLens · Explainable NIDS"),
        page_icon=":material/shield:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    inject_styles()

    choice = _sidebar(settings)
    monitor = load_monitor()
    status = monitor.status()
    predictor = load_predictor()

    pills = [
        metrics.pill(
            ("Monitoring · " + status["mode"]) if status["running"] else "Monitoring stopped",
            "ok" if status["running"] else "idle",
        ),
        metrics.pill(
            f"Model: {predictor.model_name}" if predictor else "No model",
            "info" if predictor else "danger",
        ),
    ]
    if status["stats"]["errors"]:
        pills.append(metrics.pill(f"{status['stats']['errors']} error(s)", "warn"))

    short_name, _, description = PAGE_INFO[choice]
    metrics.header(short_name, description, pills)

    module = PAGES[choice]
    try:
        module.render()
    except NIDSError as exc:
        st.error(str(exc))
        st.caption("Full technical detail has been written to logs/nids.log.")
    except Exception as exc:  # pragma: no cover - top-level guard
        st.error(f"This page could not be rendered: {exc}")
        with st.expander("Technical detail"):
            st.code(traceback.format_exc(), language="text")


# Streamlit executes this file top to bottom on every rerun, so main() is
# called unconditionally rather than behind an ``if __name__`` guard.
main()
