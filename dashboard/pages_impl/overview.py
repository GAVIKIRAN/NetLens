"""NETWORK SECURITY OVERVIEW - the dashboard's landing page."""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import streamlit as st

from dashboard.components import charts, metrics, tables
from dashboard.state import load_database, load_monitor, load_settings
from src.network.monitor import MODE_DEMO
from src.utils.errors import NIDSError
from src.utils.helpers import format_duration, human_bytes

PAGE_TITLE = "Network Security Overview"


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

    # ------------------------------------------------------------- headline
    total = db_stats["total_flows"]
    attacks = db_stats["attack_flows"]
    alerts_frame = database.recent_alerts(500)

    current_band = "LOW"
    if not alerts_frame.empty:
        for band in ("CRITICAL", "HIGH", "MEDIUM"):
            if (alerts_frame["severity"].astype(str).str.upper() == band).any():
                current_band = band
                break

    if total == 0:
        metrics.empty_state(
            "No traffic classified yet",
            "Replay safe, pre-recorded flow records to see detection, risk scoring and "
            "explanations working. Nothing is captured or sent on your network.",
        )
        with st.container(key="empty_actions"):
            st.button(
                "Start demo traffic",
                type="primary",
                on_click=_start_demo,
                disabled=not monitor.model_ready(),
                key="ov_start_demo",
            )
            if st.session_state.get("ov_start_error"):
                st.error(st.session_state["ov_start_error"])
            st.caption("To capture real traffic, use LIVE mode on the Live Monitor page.")
        return

    severity_counts = (
        alerts_frame["severity"].astype(str).str.upper().value_counts().to_dict()
        if not alerts_frame.empty
        else {}
    )
    metrics.posture_banner(
        current_band,
        float(db_stats["attack_percentage"]),
        int(db_stats["benign_flows"]),
        int(attacks),
        severity_counts,
    )
    metrics.stat_row(
        [
            {"label": "Total flows", "value": f"{total:,}",
             "delta": "classified end to end", "accent": "blue"},
            {"label": "Traffic seen", "value": human_bytes(db_stats["total_bytes"]),
             "delta": f"{db_stats['total_packets']:,} packets", "accent": "blue"},
            {"label": "Average risk score", "value": f"{db_stats['avg_risk_score']:.1f}",
             "delta": f"peak {db_stats['max_risk_score']:.0f} / 100", "accent": "high"},
            {"label": "Stored alerts", "value": f"{db_stats['total_alerts']:,}",
             "delta": "at or above alert floor", "accent": "red"},
            {"label": "This session", "value": f"{stats['flows_processed']:,}",
             "delta": (f"{status['mode']} · {format_duration(stats['uptime_seconds'])}"
                       if status["running"] else "monitoring stopped"),
             "accent": "low" if status["running"] else ""},
            {"label": "Errors", "value": f"{stats['errors']:,}",
             "delta": stats["last_error"][:48] or "none this session",
             "accent": "critical" if stats["errors"] else ""},
        ]
    )

    # --------------------------------------------------------------- charts
    row1 = st.columns([2, 1])
    with row1[0]:
        timeline = database.timeline("minute")
        tables.chart_with_table(
            charts.timeline_chart(timeline, height=380),
            timeline,
            key="ov_timeline",
        )
    with row1[1]:
        distribution = database.attack_distribution()
        tables.chart_with_table(
            charts.class_distribution_bar(distribution, "attack_type", "count", height=380),
            distribution,
            key="ov_attack_dist",
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
                title="Traffic volume over time", y_title="bytes", height=320,
            ),
            volume,
            key="ov_volume",
        )
    with row2[1]:
        contributors = _aggregate_alert_features(alerts_frame)
        tables.chart_with_table(
            charts.importance_bar(
                contributors,
                title="Top contributing features across alerts",
                height=320,
            ),
            pd.DataFrame(contributors),
            key="ov_contrib",
        )
        if not contributors:
            metrics.note(
                "Feature contributions appear here once alerts with inline SHAP "
                "explanations have been stored."
            )

    # -------------------------------------------------------- recent alerts
    st.subheader("Recent alerts")
    recent = database.recent_alerts(int(settings.get("dashboard.recent_rows", 200)))
    if recent.empty:
        st.caption("No alerts have been raised yet.")
    else:
        tables.dataframe(tables.format_alerts(recent.head(15)), height=420)
        st.caption(
            f"Showing the 15 most recent of {len(recent):,} stored alert(s). "
            "The Alert Center page has filtering and the full history."
        )


__all__ = ["render", "PAGE_TITLE"]
