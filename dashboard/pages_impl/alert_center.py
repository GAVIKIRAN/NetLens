"""ALERT CENTER - filterable alert history."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from dashboard.components import metrics, tables, ui
from dashboard.state import load_database, load_settings

PAGE_TITLE = "Alerts"
SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def render() -> None:
    database = load_database()
    settings = load_settings()

    all_alerts = database.recent_alerts(5000)
    if all_alerts.empty:
        st.info(
            "No alerts have been raised yet. Alerts are stored when a flow is "
            f"classified as an attack **and** its risk band reaches "
            f"**{settings.get('risk.alert_min_band', 'MEDIUM')}** "
            "(configurable as `risk.alert_min_band`)."
        )
        return

    # ------------------------------------------------------------ filters --
    with st.container(key="card_ac_filters"):
        ui.render(ui.card_header_html("Filters"))
        columns = st.columns([1.2, 1.6, 1, 1])
        with columns[0]:
            severities = st.multiselect(
                "Severity",
                SEVERITIES,
                default=[s for s in SEVERITIES if s in set(all_alerts["severity"].astype(str))],
                key="ac_sev",
            )
        with columns[1]:
            attack_types = st.multiselect(
                "Attack type",
                sorted(all_alerts["attack_type"].astype(str).unique().tolist()),
                key="ac_type",
            )
        with columns[2]:
            modes = st.multiselect(
                "Mode", sorted(all_alerts["mode"].astype(str).unique().tolist()), key="ac_mode"
            )
        with columns[3]:
            limit = st.number_input(
                "Max rows", min_value=20, max_value=5000, value=300, step=20, key="ac_limit"
            )

        search = st.text_input(
            "Search by address", value="", key="ac_search",
            placeholder="192.168.10.50",
        )


    frame = all_alerts.copy()
    if severities:
        frame = frame[frame["severity"].astype(str).isin(severities)]
    if attack_types:
        frame = frame[frame["attack_type"].astype(str).isin(attack_types)]
    if modes:
        frame = frame[frame["mode"].astype(str).isin(modes)]
    if search.strip():
        needle = search.strip()
        frame = frame[
            frame["src_ip"].astype(str).str.contains(needle, case=False, na=False)
            | frame["dst_ip"].astype(str).str.contains(needle, case=False, na=False)
        ]
    frame = frame.head(int(limit))

    # ------------------------------------------------------------- summary
    counts = frame["severity"].astype(str).str.upper().value_counts().to_dict()
    metrics.stat_row(
        [
            {"label": "Matching alerts", "value": f"{len(frame):,}",
             "delta": f"of {len(all_alerts):,} stored"},
            {"label": "Critical", "value": f"{counts.get('CRITICAL', 0):,}",
             "delta": "risk score 75 and above", "band": "CRITICAL"},
            {"label": "High", "value": f"{counts.get('HIGH', 0):,}",
             "delta": "risk score 50 to 74", "band": "HIGH"},
            {"label": "Medium", "value": f"{counts.get('MEDIUM', 0):,}",
             "delta": "risk score 25 to 49", "band": "MEDIUM"},
        ]
    )

    if frame.empty:
        st.info("No alerts match the current filters")
        return

    split = st.columns([1, 1])
    with split[0]:
        by_type = frame["attack_type"].astype(str).value_counts()
        tables.list_card(
            "Alerts by attack type",
            ui.ranked_list_html(by_type.items(), label_html=ui.attack_badge, limit=8),
            key="ac_types", meta=f"{len(frame):,} alerts",
        )
    with split[1]:
        tables.list_card(
            "Alerts by severity", ui.severity_list_html(counts),
            key="ac_bands", meta=f"{len(frame):,} alerts",
        )

    st.subheader("Alerts")
    tables.dataframe(tables.format_alerts(frame), height=480)

    ui.export_button(
        "Download filtered alerts (CSV)",
        data=tables.format_alerts(frame).to_csv(index=False).encode("utf-8"),
        file_name="nids_alerts.csv",
        mime="text/csv",
    )

    # ------------------------------------------------------- alert detail
    st.subheader("Alert detail")
    options = frame["id"].astype(int).tolist()
    selected = st.selectbox(
        "Alert", options, format_func=lambda i: f"#{i}", key="ac_detail"
    )
    row = frame[frame["id"] == selected].iloc[0]
    from_model = pd.notna(row.get("prediction_id"))

    def _port(value) -> str:
        try:
            port = int(float(value))
        except (TypeError, ValueError):
            return ""
        return f":{port}" if port else ""

    flow = (
        f"{row['src_ip']}{_port(row.get('src_port'))} → {row['dst_ip']}{_port(row.get('dst_port'))}"
        f" · {row.get('protocol_name') or ''}"
    )
    facts = [
        ("Severity", ui.severity_badge(str(row["severity"]), float(row["risk_score"]))),
        ("Attack type", ui.attack_badge(str(row["attack_type"]))),
        ("ML confidence", f"{float(row['confidence']):.1%}") if from_model
        else ("Detected by", "behaviour rules, across many connections"),
        ("Flow", html.escape(flow)),
        ("Time (UTC)", html.escape(str(row.get("timestamp", "")).replace("T", " ")[:19])),
        ("Mode", html.escape(str(row.get("mode", "")))),
    ]

    detail = st.columns([1, 1.6])
    with detail[0]:
        ui.panel(ui.kv_html(facts), title=f"Alert #{int(selected)}")
    with detail[1]:
        with st.container(key="card_ac_explanation"):
            ui.render(ui.card_header_html("Explanation"))
            explanation = str(row.get("explanation") or "")
            if explanation:
                st.markdown(explanation)
            elif from_model:
                st.caption(
                    "No inline explanation was stored for this alert. Open "
                    "Explain a prediction and select prediction "
                    f"#{int(row['prediction_id'])} to compute SHAP and LIME on demand."
                )
            else:
                st.caption("Behaviour detections are explained by the rule that fired, shown in the flow table")
            if from_model:
                st.caption(
                    "Feature contributions describe how the model weighed this flow. "
                    "They are not a claim that any feature caused the behaviour."
                )


__all__ = ["render", "PAGE_TITLE"]
