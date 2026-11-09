"""COMMAND CENTER - where the dashboard opens.

An asymmetric split: what NetLens does and the one action that starts it on
the left; on the right, nothing: the page opens on its words and actions. Below: the detection pipeline in real totals, the latest alerts and the
classes the model recognises. Every figure is read from the database or the
trained model; nothing is a placeholder.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from dashboard.components import ui
from dashboard.state import load_database, load_monitor, load_predictor, load_training_metrics
from dashboard.styles import theme
from src.network.monitor import MODE_DEMO
from src.utils.errors import NIDSError

PAGE_TITLE = "Command center"
_e = html.escape

#: Shown when no model has been trained yet: the CIC-IDS2017 classes.
FALLBACK_CLASSES = (
    "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "DDoS", "PortScan",
    "FTP-Patator", "SSH-Patator", "Web Attack - Brute Force", "Web Attack - XSS", "Bot",
)


def _go(page: str) -> None:
    st.session_state["nav"] = page


def _start_demo() -> None:
    """Start safe demo replay and move to the page that shows it."""
    st.session_state.pop("home_start_error", None)
    try:
        load_monitor().start(MODE_DEMO)
        st.session_state["nav"] = "Live Monitor"
    except NIDSError as exc:
        st.session_state["home_start_error"] = str(exc)


def _hero(model_name: str | None) -> None:
    monitor = load_monitor()
    running = bool(monitor.status()["running"])
    model = f"a trained {_e(model_name)} model" if model_name else "a trained model"
    ui.render(
        '<div class="nl-hero"><h1>Intrusion detection that explains every alert</h1>'
        f"<p>NetLens sorts traffic into flows, classifies each one with {model}, scores its "
        "risk and shows which measurements drove the verdict.</p></div>"
    )
    with st.container(horizontal=True, gap="small", key="home_ctas"):
        if running:
            ui.primary_button("Open live monitor", on_click=_go, args=("Live Monitor",),
                              key="home_cta_live")
        else:
            ui.primary_button("Start demo traffic", on_click=_start_demo,
                              disabled=not monitor.model_ready(), key="home_cta_demo")
        ui.secondary_button("View threat overview", on_click=_go,
                            args=("Network Security Overview",), key="home_cta_overview")
    if st.session_state.get("home_start_error"):
        st.error(st.session_state["home_start_error"])
    if not running and not monitor.model_ready():
        st.caption("Train a model first: `python scripts/train_model.py`")


def _right_now(database, stats: dict, predictor) -> None:
    """Where the map was: the state of this installation in one panel -
    stored alerts by severity on the left, the read-outs that matter on the right."""
    bands = database.query(
        "SELECT UPPER(severity) AS band, COUNT(*) AS n FROM alerts GROUP BY UPPER(severity)"
    )
    counts = dict(zip(bands["band"], bands["n"])) if not bands.empty else {}
    last = database.recent_alerts(1)
    last_seen = str(last.iloc[0].get("timestamp", ""))[:19] if not last.empty else "none yet"
    status = load_monitor().status()
    if status["running"]:
        state = ui.status_text(f"{str(status['mode']).title()} monitoring running", "live")
    else:
        state = ui.status_text("Monitoring stopped", "idle")
    model = f"{_e(predictor.model_name)} · {len(predictor.classes)} classes" if predictor else "not trained"
    readout = ui.kv_html([
        ("Model", model),
        ("Average risk", f"{float(stats.get('avg_risk_score') or 0):.1f}"),
        ("Highest risk", f"{float(stats.get('max_risk_score') or 0):.1f}"),
        ("Model confidence", f"{100 * float(stats.get('avg_confidence') or 0):.1f}%"),
        ("Last alert", _e(last_seen)),
    ])
    ui.render(
        '<section class="nl-panel nl-now">'
        + ui.card_header_html("Right now", f"{int(stats.get('total_alerts') or 0):,} stored alerts")
        + f'<div class="nl-now-state">{state}</div>'
        + '<div class="nl-now-grid">'
        + f'<div><div class="nl-now-k">Alerts by severity</div>{ui.severity_list_html(counts)}</div>'
        + f'<div><div class="nl-now-k">Read-outs</div>{readout}</div>'
        + "</div></section>"
    )


def _pipeline(database, stats: dict) -> None:
    explained = database.query(
        "SELECT COUNT(*) AS n FROM alerts WHERE explanation IS NOT NULL AND explanation != ''"
    )
    explained_n = int(explained.iloc[0]["n"]) if not explained.empty else 0
    ui.section("Pipeline", "all recorded traffic")
    ui.stats([
        {"label": "Captured", "value": f"{int(stats.get('total_packets') or 0):,}",
         "note": f"packets metered into {int(stats.get('total_flows') or 0):,} flows, read-only"},
        {"label": "Classified as attack", "value": f"{float(stats.get('attack_percentage') or 0):.1f}%",
         "note": f"{int(stats.get('attack_flows') or 0):,} flows, scored for risk"},
        {"label": "Alerts explained", "value": f"{explained_n:,}",
         "note": f"of {int(stats.get('total_alerts') or 0):,} stored alerts"},
    ])


def _port(value) -> str:
    try:
        port = int(float(value))
    except (TypeError, ValueError):
        return ""
    return f":{port}" if port else ""


def _latest_alerts(database, total_alerts: int) -> None:
    ui.section("Latest alerts", f"{total_alerts:,} stored")
    alerts = database.recent_alerts(6)
    rows = []
    for _, row in alerts.iterrows():
        rows.append({
            "time": str(row.get("timestamp", ""))[11:19],
            "band": row.get("severity", ""),
            "score": row.get("risk_score"),
            "label": row.get("attack_type", ""),
            "flow": f"{row.get('src_ip', '')}{_port(row.get('src_port'))} → "
                    f"{row.get('dst_ip', '')}{_port(row.get('dst_port'))}",
            "confidence": row.get("confidence") if pd.notna(row.get("prediction_id")) else None,
        })
    ui.panel(ui.log_html(rows, empty="No alerts have been raised yet"))
    if rows:
        ui.secondary_button("Review all alerts", on_click=_go, args=("Alert Center",),
                            key="home_cta_alerts")


def _coverage(classes: list[str]) -> None:
    attacks = [label for label in classes if theme.attack_family(label) != "Benign"]
    metrics = (load_training_metrics() or {}).get("best_model", {}).get("test") or {}
    f1 = metrics.get("f1_macro")
    meta = f"{len(classes)} classes" + (f" · macro F1 {f1:.3f} on held-out test" if f1 is not None else "")
    ui.section("What the model recognises", meta)
    tags = "".join(ui.attack_badge(label) for label in attacks)
    ui.panel(f'<div class="nl-tags">{tags}</div>')


def render() -> None:
    database = load_database()
    predictor = load_predictor()
    stats = database.statistics()
    classes = list(predictor.classes) if predictor else list(FALLBACK_CLASSES)

    text, side = st.columns([1, 1], vertical_alignment="center", gap="large")
    with text:
        _hero(predictor.model_name if predictor else None)
    with side:
        _right_now(database, stats, predictor)

    _pipeline(database, stats)
    _latest_alerts(database, int(stats.get("total_alerts") or 0))
    _coverage(classes)


__all__ = ["render", "PAGE_TITLE"]
