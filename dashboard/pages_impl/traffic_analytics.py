"""TRAFFIC ANALYTICS - protocol, volume, duration and trend analysis."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components import charts, metrics, tables
from dashboard.state import load_database
from src.utils.helpers import human_bytes

PAGE_TITLE = "Traffic Analytics"


def render() -> None:
    database = load_database()
    stats = database.statistics()

    if stats["captured_flows"] == 0:
        st.info(
            "No flows have been recorded yet. Start monitoring from the "
            "**Live Monitor** page (DEMO mode replays saved records safely)."
        )
        return

    metrics.stat_row(
        [
            {"label": "Flows recorded", "value": f"{stats['captured_flows']:,}", "accent": "blue"},
            {"label": "Packets", "value": f"{stats['total_packets']:,}"},
            {"label": "Bytes", "value": human_bytes(stats["total_bytes"])},
            {"label": "Attack share", "value": f"{stats['attack_percentage']:.1f}%", "accent": "red"},
        ]
    )

    st.divider()
    row1 = st.columns(2)
    with row1[0]:
        protocols = database.protocol_distribution()
        tables.chart_with_table(
            charts.magnitude_bar(
                list(protocols["protocol"]),
                list(protocols["flows"]),
                title="Protocol distribution",
                x_title="flows",
                height=280,
                hover_suffix=" flows",
            ),
            protocols,
            key="ta_proto",
        )
    with row1[1]:
        talkers = database.top_talkers(10)
        tables.chart_with_table(
            charts.magnitude_bar(
                list(talkers["src_ip"]),
                list(talkers["bytes"].fillna(0)),
                title="Top sources by volume",
                x_title="bytes",
                height=340,
                value_format="{:,.0f}",
                hover_suffix=" bytes",
            ),
            talkers,
            key="ta_talkers",
        )

    row2 = st.columns(2)
    with row2[0]:
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
            volume,
            key="ta_packets",
        )
    with row2[1]:
        timeline = database.timeline("minute")
        tables.chart_with_table(
            charts.timeline_chart(timeline, title="Attack trend over time"),
            timeline,
            key="ta_trend",
        )

    st.divider()
    st.subheader("Flow duration and size")
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
        tables.chart_with_table(
            charts.magnitude_bar(
                list(duration["bucket"]), list(duration["flows"]),
                title="Flow duration distribution", x_title="flows", height=300,
            ),
            duration,
            key="ta_duration",
        )
    with row3[1]:
        classes = database.class_distribution()
        tables.chart_with_table(
            charts.class_distribution_bar(
                classes, "prediction", "count",
                title="Predicted class distribution (all flows)", height=340,
            ),
            classes,
            key="ta_classes",
        )

    st.divider()
    st.subheader("Flow records")
    limit = st.slider("Rows", 50, 2000, 300, 50, key="ta_rows")
    flows = database.query(
        "SELECT id, timestamp, src_ip, src_port, dst_ip, dst_port, protocol_name, "
        "duration_seconds, packet_count, byte_count, termination, mode "
        "FROM flows ORDER BY id DESC LIMIT ?",
        (int(limit),),
    )
    tables.dataframe(flows, height=440)
    st.download_button(
        "Download flow records (CSV)",
        data=flows.to_csv(index=False).encode("utf-8"),
        file_name="nids_flows.csv",
        mime="text/csv",
    )


__all__ = ["render", "PAGE_TITLE"]
