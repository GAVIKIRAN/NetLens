"""ALERT CENTER - filterable alert history."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components import charts, metrics, tables
from dashboard.state import load_database, load_settings

PAGE_TITLE = "Alert Center"
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

    # ---------------------------------------------------- one filter row --
    st.markdown("##### Filters")
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
        "Search source / destination address", value="", key="ac_search",
        placeholder="e.g. 192.168.10.50",
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
             "delta": f"of {len(all_alerts):,} stored", "accent": "blue"},
            {"label": "Critical", "value": f"{counts.get('CRITICAL', 0):,}",
             "delta": "risk score >= 75", "accent": "critical"},
            {"label": "High", "value": f"{counts.get('HIGH', 0):,}",
             "delta": "risk score 50-74", "accent": "high"},
            {"label": "Medium", "value": f"{counts.get('MEDIUM', 0):,}",
             "delta": "risk score 25-49", "accent": "medium"},
        ]
    )

    if frame.empty:
        st.warning("No alerts match the current filters.")
        return

    st.divider()
    split = st.columns([1, 1])
    with split[0]:
        distribution = (
            frame.groupby("attack_type", as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        tables.chart_with_table(
            charts.class_distribution_bar(
                distribution, "attack_type", "count",
                title="Alerts by attack type", height=320,
            ),
            distribution,
            key="ac_types",
        )
    with split[1]:
        tables.chart_with_table(
            charts.risk_band_bar(counts, height=320),
            pd.DataFrame([{"severity": k, "alerts": v} for k, v in counts.items()]),
            key="ac_bands",
        )

    st.subheader("Alerts")
    tables.dataframe(tables.format_alerts(frame), height=480)

    st.download_button(
        "Download filtered alerts (CSV)",
        data=tables.format_alerts(frame).to_csv(index=False).encode("utf-8"),
        file_name="nids_alerts.csv",
        mime="text/csv",
    )

    # ------------------------------------------------------- alert detail
    st.divider()
    st.subheader("Alert detail")
    options = frame["id"].astype(int).tolist()
    selected = st.selectbox(
        "Alert ID", options, format_func=lambda i: f"#{i}", key="ac_detail"
    )
    row = frame[frame["id"] == selected].iloc[0]

    detail = st.columns([1, 2])
    with detail[0]:
        st.markdown(
            metrics.risk_badge(str(row["severity"]), float(row["risk_score"])),
            unsafe_allow_html=True,
        )
        metrics.stat_tile("Attack type", str(row["attack_type"]), accent="red")
        from_model = pd.notna(row.get("prediction_id"))
        if from_model:
            metrics.stat_tile("Model confidence", f"{float(row['confidence']):.1%}",
                              "probability of the predicted class", accent="blue")
        else:
            metrics.stat_tile("Detected by", "Behaviour rules",
                              "across many connections, not by the model", accent="blue")
        if from_model:
            metrics.stat_tile(
                "Connection",
                f"{row['src_ip']}:{int(row['src_port'])}",
                f"to {row['dst_ip']}:{int(row['dst_port'])} over {row['protocol_name']}",
            )
        else:
            port = f":{int(row['dst_port'])}" if int(row["dst_port"]) else ""
            metrics.stat_tile(
                "Source", str(row["src_ip"]), f"to {row['dst_ip']}{port} over {row['protocol_name']}"
            )
    with detail[1]:
        explanation = str(row.get("explanation") or "")
        if explanation:
            st.markdown(explanation)
        elif from_model:
            st.caption(
                "No inline explanation was stored for this alert. Open the "
                "**AI Explanation** page and select prediction "
                f"#{int(row['prediction_id'])} to compute SHAP and LIME on demand."
            )
        if from_model:
            st.caption(
                "Feature contributions describe how the model weighed this flow. "
                "They are not a claim that any feature caused the behaviour."
            )


__all__ = ["render", "PAGE_TITLE"]
