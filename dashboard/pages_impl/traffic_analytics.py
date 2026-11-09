"""TRAFFIC ANALYTICS - protocol, volume, duration and trend analysis."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import html

from dashboard.components import charts, metrics, tables, ui
from dashboard.state import load_database
from src.utils.helpers import human_bytes

PAGE_TITLE = "Traffic Analytics"


def render() -> None:
    database = load_database()
    stats = database.statistics()

    if stats["captured_flows"] == 0:
        st.info(
            "No flows have been recorded yet. Start monitoring from the "
            "Live monitor page (DEMO mode replays saved records safely)."
        )
        return

    metrics.stat_row(
        [
            {"label": "Flows recorded", "value": f"{stats['captured_flows']:,}"},
            {"label": "Packets", "value": f"{stats['total_packets']:,}"},
            {"label": "Bytes", "value": human_bytes(stats["total_bytes"])},
            {"label": "Attack share", "value": f"{stats['attack_percentage']:.1f}%",
             "accent": True},
        ]
    )

    volume_row = st.columns([2, 1])
    with volume_row[0]:
        volume = database.query(
            "SELECT strftime('%Y-%m-%dT%H:%M', timestamp) AS bucket, "
            "SUM(byte_count) AS bytes, SUM(packet_count) AS packets, COUNT(*) AS flows "
            "FROM flows GROUP BY bucket ORDER BY bucket"
        )
        tables.chart_with_table(
            charts.volume_area_chart(
                volume, x_column="bucket", y_column="packets",
                title="Packet volume over time", y_title="packets",
            ),
            volume, key="ta_packets", meta="per minute",
        )
    with volume_row[1]:
        protocols = database.protocol_distribution()
        tables.list_card(
            "Protocols",
            ui.ranked_list_html(zip(protocols["protocol"], protocols["flows"])),
            key="ta_proto", meta="flows",
        )

    trend_row = st.columns([2, 1])
    with trend_row[0]:
        timeline = database.timeline("minute")
        tables.chart_with_table(
            charts.timeline_chart(timeline, title="Attack trend over time"),
            timeline, key="ta_trend", meta="per minute",
        )
    with trend_row[1]:
        talkers = database.top_talkers(10)
        tables.list_card(
            "Top sources by volume",
            ui.ranked_list_html(zip(talkers["src_ip"], talkers["bytes"].fillna(0)),
                                label_html=lambda ip: f'<span class="mono">{html.escape(ip)}</span>'),
            key="ta_talkers", meta="bytes",
        )

    st.subheader("Flow duration and classes")
    duration = database.query(
        """
        SELECT CASE
                 WHEN duration_seconds < 0.1  THEN '< 0.1 s'
                 WHEN duration_seconds < 1    THEN '0.1 - 1 s'
                 WHEN duration_seconds < 10   THEN '1 - 10 s'
                 WHEN duration_seconds < 60   THEN '10 - 60 s'
                 WHEN duration_seconds < 300  THEN '1 - 5 min'
                 ELSE '> 5 min'
               END AS bucket,
               COUNT(*) AS flows,
               AVG(packet_count) AS avg_packets,
               AVG(byte_count) AS avg_bytes
          FROM flows
         GROUP BY bucket
        """
    )
    order = ["< 0.1 s", "0.1 - 1 s", "1 - 10 s", "10 - 60 s", "1 - 5 min", "> 5 min"]
    duration["_order"] = duration["bucket"].map({b: i for i, b in enumerate(order)})
    duration = duration.sort_values("_order").drop(columns=["_order"])

    row3 = st.columns(2)
    with row3[0]:
        tables.list_card(
            "Flow duration",
            ui.ranked_list_html(zip(duration["bucket"], duration["flows"]),
                                sort=False, accent_first=False),
            key="ta_duration", meta="flows",
        )
    with row3[1]:
        classes = database.class_distribution()
        tables.list_card(
            "Predicted classes",
            ui.ranked_list_html(zip(classes["prediction"], classes["count"]),
                                label_html=ui.attack_badge, limit=8),
            key="ta_classes", meta="all flows",
        )

    st.subheader("Flow records")
    limit = st.slider("Rows", 50, 2000, 300, 50, key="ta_rows")
    flows = database.query(
        "SELECT id, timestamp, src_ip, src_port, dst_ip, dst_port, protocol_name, "
        "duration_seconds, packet_count, byte_count, termination, mode "
        "FROM flows ORDER BY id DESC LIMIT ?",
        (int(limit),),
    )
    tables.dataframe(flows, height=440)
    ui.export_button(
        "Download flow records (CSV)",
        data=flows.to_csv(index=False).encode("utf-8"),
        file_name="nids_flows.csv",
        mime="text/csv",
    )


__all__ = ["render", "PAGE_TITLE"]
