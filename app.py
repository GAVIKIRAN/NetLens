"""XAI-NIDS dashboard entry point.

Run it from the project root with::

    streamlit run app.py

Two separate pages share one server:

* ``/``        - the landing page (``dashboard/pages_impl/landing.py``)
* ``/console`` - the working console with its sidebar of tools

The dashboard is a *read-mostly* view over the SQLite store plus a control
surface for the monitoring service.  All detection work happens in
:mod:`src`; nothing in ``dashboard/`` trains, captures or classifies on its own.
"""

from __future__ import annotations

import html
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402


def _reload_edited_ui_modules() -> None:
    """Pick up edited dashboard UI modules without a server restart.

    Streamlit reruns this script on save but on this setup it kept serving
    stale copies of imported modules (ImportError for names that exist on
    disk). Each module under the UI packages is reloaded when its source file
    is newer than the copy in memory - styles first, then components, then
    pages, so every page binds to fresh helpers. ``dashboard.state`` and
    ``src`` are never touched: they hold the cached monitor and model.
    """
    import importlib
    import os

    for prefix in ("dashboard.styles.", "dashboard.components.", "dashboard.pages_impl."):
        for name in sorted(n for n in list(sys.modules) if n.startswith(prefix)):
            module = sys.modules.get(name)
            path = getattr(module, "__file__", None)
            if not path or not os.path.exists(path):
                continue
            mtime = os.path.getmtime(path)
            # A module seen for the first time is reloaded once: it may have been
            # imported before its file was last edited.
            if getattr(module, "__nl_mtime__", None) != mtime:
                try:
                    module = importlib.reload(module)
                except Exception:  # noqa: BLE001 - the import below reports real errors
                    continue
                module.__nl_mtime__ = mtime


_reload_edited_ui_modules()

from dashboard.components import metrics, ui  # noqa: E402
from dashboard.components.side_nav import console_rail  # noqa: E402
from dashboard.pages_impl import (  # noqa: E402
    ai_explanation,
    alert_center,
    home,
    landing,
    live_monitor,
    model_performance,
    overview,
    sessions,
    settings_page,
    traffic_analytics,
)
from dashboard import state as _state  # noqa: E402
from dashboard.state import load_monitor, load_predictor, load_settings  # noqa: E402
from dashboard.styles.theme import inject_css  # noqa: E402
from src.network.monitor import MODE_DEMO, MODE_LIVE  # noqa: E402
from src.utils.errors import NIDSError  # noqa: E402

#: Internal page id (the navigation radio's value, also set by buttons that
#: jump between pages) -> module. Dict order is the sidebar order.
PAGES = {
    "Home": home,
    "Network Security Overview": overview,
    "Live Monitor": live_monitor,
    "Alert Center": alert_center,
    "AI Explanation": ai_explanation,
    "Traffic Analytics": traffic_analytics,
    "Sessions": sessions,
    "Model Performance": model_performance,
    "System Settings": settings_page,
}

#: Page id -> (label, Material icon, one-line description).
PAGE_INFO = {
    "Home": ("Command center", "radar", "The network at a glance and where to start"),
    "Network Security Overview": (
        "Threat overview", "shield", "Threat level, attack share and alerts across classified flows",
    ),
    "Live Monitor": ("Live monitor", "monitoring", "Start monitoring and watch flows being classified"),
    "Alert Center": ("Alerts", "notifications", "Filter, review and export stored alerts"),
    "AI Explanation": (
        "Explain a prediction", "account_tree", "Why the model classified a stored flow the way it did",
    ),
    "Traffic Analytics": ("Traffic analytics", "lan", "Protocols, top talkers and traffic trends"),
    "Sessions": ("Sessions", "history", "Every monitoring run, saved and downloadable as CSV"),
    "Model Performance": ("Model performance", "query_stats", "Measured results for every trained model"),
    "System Settings": ("Settings", "tune", "Capture, detection and data retention settings"),
}

#: Rail groups: (label, first page id in the group). A hairline separates them.
NAV_GROUPS = (
    ("Monitor", "Home"),
    ("Investigate", "AI Explanation"),
    ("Model and system", "Model Performance"),
)


def _select_page(page: str) -> None:
    st.session_state["nav"] = page


def _rail() -> str:
    """The console's floating icon rail; returns the selected page id."""
    choice = st.session_state.get("nav")
    if choice not in PAGES:
        choice = st.session_state["nav"] = "Home"
    console_rail(
        [(page, PAGE_INFO[page][0], PAGE_INFO[page][1]) for page in PAGES],
        active=choice,
        on_select=_select_page,
        group_starts=[first for _, first in NAV_GROUPS[1:]],
        home_href="./",
    )
    return choice


def _status_meta(status: dict, predictor) -> list[str]:
    """Quiet status for the right side of the page header."""
    if status["running"]:
        mode = str(status["mode"]).upper()
        if mode == MODE_LIVE:
            iface = (status.get("capture") or {}).get("interface")
            text = "Live capture" + (f" on {iface}" if iface else "")
        elif mode == MODE_DEMO:
            text = "Demo replay running"
        else:
            text = f"{mode} replay running"
        meta = [ui.status_text(text, "live")]
    else:
        meta = [ui.status_text("Monitoring stopped", "idle")]
    if predictor is not None:
        meta.append(
            f'<span class="nl-meta">{html.escape(predictor.model_name)} · '
            f"{len(predictor.classes)} classes</span>"
        )
    else:
        meta.append(ui.status_text("No trained model", "error"))
    errors = int(status["stats"]["errors"])
    if errors:
        meta.append(ui.status_text(f"{errors} error{'s' if errors != 1 else ''} this session", "error"))
    return meta


def console(settings, landing_page) -> None:
    """The working page: a floating icon rail over every tool."""
    inject_css()
    choice = _rail()
    module = PAGES[choice]
    status = load_monitor().status()
    predictor = load_predictor()

    if module is not home:  # the command center opens with its own hero
        label, _, description = PAGE_INFO[choice]
        metrics.header(label, description, _status_meta(status, predictor))

    try:
        module.render()
    except NIDSError as exc:
        st.error(str(exc))
        st.caption("Full technical detail has been written to logs/nids.log.")
    except Exception as exc:  # pragma: no cover - top-level guard
        st.error(f"This page could not be rendered: {exc}")
        with st.expander("Technical detail"):
            st.code(traceback.format_exc(), language="text")


def main() -> None:
    settings = load_settings()
    st.set_page_config(
        page_title=settings.get("dashboard.title", "NetLens · Explainable NIDS"),
        page_icon=":material/shield:",
        layout="wide",
        initial_sidebar_state="auto",
    )

    pages: dict[str, st.Page] = {}

    def landing_view() -> None:
        inject_css()
        landing.render(pages["console"], settings)

    def console_view() -> None:
        console(settings, pages["landing"])

    pages["landing"] = st.Page(
        landing_view, title="NetLens · Explainable NIDS", icon=":material/shield:", default=True,
    )
    pages["console"] = st.Page(
        console_view, title="Console · NetLens", icon=":material/radar:", url_path="console",
    )
    # Hidden: the landing page has its own nav bar and the console its own sidebar.
    # Explainers build while the visitor reads the page. Looked up at run time:
    # dashboard.state is never hot-reloaded (it owns the running monitor), so a
    # server started before this function existed must still run.
    warm_up = getattr(_state, "warm_up_in_background", None)
    if warm_up is not None:
        warm_up()
    st.navigation(list(pages.values()), position="hidden").run()


# Streamlit executes this file top to bottom on every rerun, so main() is
# called unconditionally rather than behind an ``if __name__`` guard.
main()
