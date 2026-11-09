"""THREAT OVERVIEW - posture, the attack timeline and recent alerts.

One main chart (the attack timeline, two thirds wide) with smaller blocks of a
different kind beside and below it: attack types and top features as ranked
lists, traffic volume as a secondary chart.
"""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import streamlit as st

from dashboard.components import charts, metrics, tables, ui
from dashboard.state import load_database, load_monitor, load_settings
from src.network.monitor import MODE_DEMO
from src.utils.errors import NIDSError
from src.utils.helpers import format_duration, human_bytes

PAGE_TITLE = "Threat overview"


def _aggregate_alert_features(alerts: pd.DataFrame, top_k: int = 10) -> list[dict]:
    """Mean |contribution| per feature across the stored alert explanations."""
    totals: Counter = Counter()
    counts: Counter = Counter()
    labels: dict[str, str] = {}
    for raw in alerts.get("top_features_json", pd.Series(dtype=object)).dropna():
        try:
            entries = json.loads(raw)
        except (TypeError, ValueError):
            continue
        for entry in entries or []:
            feature = str(entry.get("feature", ""))
            if not feature:
                continue
            totals[feature] += abs(float(entry.get("contribution", 0.0)))
            counts[feature] += 1
            labels[feature] = str(entry.get("label") or feature)
    rows = [
        {"feature": feature, "label": labels.get(feature, feature),
         "importance": totals[feature] / max(1, counts[feature])}
        for feature in totals
    ]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows[:top_k]


def _start_demo() -> None:
    """Button callback: start safe demo replay and go to the page that shows it."""
    st.session_state.pop("ov_start_error", None)
    try:
        load_monitor().start(MODE_DEMO)
        st.session_state["nav"] = "Live Monitor"
    except NIDSError as exc:
        st.session_state["ov_start_error"] = str(exc)


def render() -> None:
    settings = load_settings()
    database = load_database()
    monitor = load_monitor()

    status = monitor.status()
    stats = status["stats"]
    db_stats = database.statistics()

    total = db_stats["total_flows"]
    attacks = db_stats["attack_flows"]
    alerts_frame = database.recent_alerts(500)

    if total == 0:
        metrics.empty_state(
            "No traffic classified yet",
            "Replay safe, pre-recorded flow records to see detection, risk scoring and "
            "explanations working. Nothing is captured or sent on your network.",
        )
        with st.container(key="empty_actions"):
            ui.primary_button(
                "Start demo traffic",
                on_click=_start_demo,
                disabled=not monitor.model_ready(),
                key="ov_start_demo",
            )
            if st.session_state.get("ov_start_error"):
                st.error(st.session_state["ov_start_error"])
            st.caption("To capture real traffic, use LIVE mode on the Live monitor page")
        return

    current_band = "LOW"
    if not alerts_frame.empty:
        for band in ("CRITICAL", "HIGH", "MEDIUM"):
            if (alerts_frame["severity"].astype(str).str.upper() == band).any():
                current_band = band
                break
    severity_counts = (
        alerts_frame["severity"].astype(str).str.upper().value_counts().to_dict()
        if not alerts_frame.empty else {}
    )

    # --------------------------------------------------------------- posture
    metrics.posture_banner(
        current_band,
        float(db_stats["attack_percentage"]),
        int(db_stats["benign_flows"]),
        int(attacks),
        severity_counts,
    )
    metrics.stat_row([
        {"label": "Total flows", "value": f"{total:,}", "delta": "classified end to end"},
        {"label": "Traffic seen", "value": human_bytes(db_stats["total_bytes"]),
         "delta": f"{db_stats['total_packets']:,} packets"},
        {"label": "Average risk score", "value": f"{db_stats['avg_risk_score']:.1f}",
         "delta": f"peak {db_stats['max_risk_score']:.0f} / 100"},
        {"label": "Stored alerts", "value": f"{db_stats['total_alerts']:,}",
         "delta": "at or above the alert floor"},
        {"label": "This session", "value": f"{stats['flows_processed']:,}",
         "delta": (f"{status['mode'].title()} · {format_duration(stats['uptime_seconds'])}"
                   if status["running"] else "monitoring stopped")},
        {"label": "Errors", "value": f"{stats['errors']:,}",
         "delta": stats["last_error"][:48] or "none this session"},
    ])

    # ------------------------------------------- main chart + ranked list
    row1 = st.columns([2, 1])
    with row1[0]:
        timeline = database.timeline("minute")
        tables.chart_with_table(
            charts.timeline_chart(timeline, title="Attacks and traffic over time", height=340),
            timeline, key="ov_timeline", meta="per minute",
        )
    with row1[1]:
        distribution = database.attack_distribution()
        rows = (
            list(zip(distribution["attack_type"], distribution["count"]))
            if not distribution.empty else []
        )
        tables.list_card(
            "Attack types",
            ui.ranked_list_html(rows, label_html=ui.attack_badge, limit=8,
                                empty="No attacks detected yet"),
            key="ov_attack_types", meta="flows",
        )

    row2 = st.columns([1, 1])
    with row2[0]:
        volume = database.query(
            "SELECT strftime('%Y-%m-%dT%H:%M', timestamp) AS bucket, "
            "SUM(byte_count) AS bytes, SUM(packet_count) AS packets "
            "FROM flows GROUP BY bucket ORDER BY bucket"
        )
        tables.chart_with_table(
            charts.volume_area_chart(
                volume, x_column="bucket", y_column="bytes",
                title="Traffic volume", y_title="bytes", height=260,
            ),
            volume, key="ov_volume", meta="bytes per minute",
        )
    with row2[1]:
        contributors = _aggregate_alert_features(alerts_frame)
        tables.list_card(
            "Features that drove alerts",
            ui.ranked_list_html(
                [(row["label"], row["importance"]) for row in contributors], fmt="{:.4f}",
                empty="Appears once alerts with inline SHAP explanations are stored",
            ),
            key="ov_contrib", meta="mean |SHAP|",
        )

    # --------------------------------------------------------- recent alerts
    st.subheader("Recent alerts")
    recent = database.recent_alerts(int(settings.get("dashboard.recent_rows", 200)))
    if recent.empty:
        st.caption("No alerts have been raised yet")
    else:
        tables.dataframe(tables.format_alerts(recent.head(15)), height=420)
        st.caption(
            f"The 15 most recent of {len(recent):,} stored alerts. "
            "The Alerts page has filtering and the full history."
        )


__all__ = ["render", "PAGE_TITLE"]
