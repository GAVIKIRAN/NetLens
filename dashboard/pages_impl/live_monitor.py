"""LIVE MONITOR - start/stop capture and watch flows being classified."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components import charts, metrics, tables
from dashboard.components.refresh import auto_refresh_supported, fragment, refresh_controls
from dashboard.state import load_database, load_monitor, load_settings
from src.network.interfaces import (
    capture_driver_status,
    default_interface,
    list_interfaces,
    rank_interfaces,
    scapy_available,
)
from src.network.monitor import MODE_DEMO, MODE_LIVE, MODE_PCAP
from src.utils.errors import NIDSError
from src.utils.helpers import format_duration

PAGE_TITLE = "Live Monitor"


def _mode_help(mode: str) -> str:
    return {
        MODE_DEMO: (
            "**DEMO** replays saved flow records (real CIC-IDS2017 samples when the "
            "dataset is present, otherwise coherent synthetic records). Nothing is "
            "captured and nothing is transmitted - this is the safe presentation mode."
        ),
        MODE_LIVE: (
            "**LIVE** captures packets from a real interface, read-only. On Windows "
            "this needs the Npcap driver and an Administrator terminal."
        ),
        MODE_PCAP: (
            "**PCAP** replays a saved capture file through the same flow meter. "
            "Useful for analysing traffic that was recorded elsewhere."
        ),
    }.get(mode, "")


def _controls() -> None:
    settings = load_settings()
    monitor = load_monitor()
    status = monitor.status()

    columns = st.columns([1.1, 2, 1, 1])
    with columns[0]:
        mode = st.selectbox(
            "Mode",
            [MODE_DEMO, MODE_LIVE, MODE_PCAP],
            index=[MODE_DEMO, MODE_LIVE, MODE_PCAP].index(
                str(settings.get("dashboard.default_mode", MODE_DEMO)).upper()
                if str(settings.get("dashboard.default_mode", MODE_DEMO)).upper()
                in (MODE_DEMO, MODE_LIVE, MODE_PCAP)
                else MODE_DEMO
            ),
            disabled=status["running"],
            key="lm_mode",
        )

    interface = None
    pcap_path = None
    replay_rate = None

    with columns[1]:
        if mode == MODE_LIVE:
            available, message = scapy_available()
            if not available:
                st.error(f"Scapy unavailable: {message}")
            else:
                driver_ok, driver_message = capture_driver_status()
                if not driver_ok:
                    st.warning(
                        f"**Live capture isn't ready.** {driver_message}, then start the "
                        "dashboard with `scripts\\start_live.bat`."
                    )
                interfaces = rank_interfaces(list_interfaces(), default_interface())
                if interfaces:
                    options = [i.display() for i in interfaces]
                    choice = st.selectbox(
                        "Network interface", options, disabled=status["running"], key="lm_iface"
                    )
                    interface = interfaces[options.index(choice)].name
        elif mode == MODE_PCAP:
            pcap_path = st.text_input(
                "PCAP file path",
                value=st.session_state.get("lm_pcap", ""),
                placeholder=r"C:\captures\sample.pcap",
                disabled=status["running"],
                key="lm_pcap",
            )
        else:
            replay_rate = st.slider(
                "Replay rate (flows/second)",
                min_value=1.0,
                max_value=60.0,
                value=float(settings.get("demo.replay_rate", 4.0)),
                step=1.0,
                disabled=status["running"],
                key="lm_rate",
            )

    with columns[2]:
        st.write("")
        if status["running"]:
            if st.button("Stop", type="primary"):
                monitor.stop()
                st.rerun()
        else:
            if st.button("Start", type="primary"):
                try:
                    monitor.start(
                        mode,
                        interface=interface,
                        pcap_path=pcap_path or None,
                        replay_rate=replay_rate,
                    )
                    st.rerun()
                except NIDSError as exc:
                    st.error(str(exc))
                except Exception as exc:  # pragma: no cover - defensive
                    st.error(f"Could not start monitoring: {exc}")

    with columns[3]:
        st.write("")
        if st.button("Classify 25 demo flows",
                     disabled=status["running"],
                     help="Runs a single batch through the full pipeline without "
                          "starting continuous monitoring."):
            try:
                count = monitor.demo_burst(25)
                st.success(f"Classified {count} demo flow(s).")
            except NIDSError as exc:
                st.error(str(exc))

    metrics.note(_mode_help(mode))


def _live_body() -> None:
    monitor = load_monitor()
    database = load_database()
    settings = load_settings()
    status = monitor.status()
    stats = status["stats"]

    pills = []
    if "capture" in status:
        capture = status["capture"]
        pills.append(metrics.pill(f"iface {capture['interface']}", "info"))
        if capture["last_error"]:
            pills.append(metrics.pill("capture error", "danger"))
    if pills:
        st.markdown(" ".join(pills), unsafe_allow_html=True)

    metrics.stat_row(
        [
            {"label": "Flows this session", "value": f"{stats['flows_processed']:,}",
             "delta": f"uptime {format_duration(stats['uptime_seconds'])}", "accent": "blue"},
            {"label": "Attack flows (final)", "value": f"{stats['final_attacks']:,}",
             "delta": f"{stats['final_suspicious']:,} suspicious · "
                      f"{stats['attacks_detected']:,} flagged by the model", "accent": "red"},
            {"label": "Behaviour detections", "value": f"{stats['behaviour_findings']:,}",
             "delta": f"{status.get('behaviour', {}).get('connection_attempts', 0):,} "
                      "connection attempts tracked", "accent": "high"},
            {"label": "Alerts raised", "value": f"{stats['alerts_raised']:,}",
             "delta": f"at or above {str(status['alert_min_band']).title()}", "accent": "high"},
            {"label": "Packets seen", "value": f"{stats['packets_seen']:,}",
             "delta": (f"{status.get('flows', {}).get('active_flows', 0):,} flows in assembly")
             if "flows" in status else "demo mode uses stored flow records"},
        ]
    )

    if status["errors"]:
        with st.expander(f"{len(status['errors'])} error(s) reported", expanded=False):
            for message in reversed(status["errors"]):
                st.write(f"- {message}")

    st.subheader("Behavioural detections")
    findings = monitor.findings_frame()
    if not findings.empty:
        tables.dataframe(tables.format_session_findings(findings), height=200)
    elif status["running"]:
        # Only this session's stored findings, so a new run starts clean.
        current = getattr(database, "current_session_id", None)
        stored = database.recent_behaviour_events(20, session_id=current)
        if stored.empty:
            st.caption("No behavioural pattern has been detected in this session.")
        else:
            tables.dataframe(tables.format_behaviour_events(stored), height=200)
    else:
        st.caption("No behavioural pattern has been detected in this session.")

    st.subheader("Live flow table")
    frame = monitor.recent_frame(int(settings.get("dashboard.recent_rows", 200)))
    if frame.empty:
        recent = database.recent_predictions(int(settings.get("dashboard.recent_rows", 200)))
        if recent.empty:
            st.caption("No flows classified yet. Press Start above.")
            return
        tables.dataframe(tables.format_predictions(recent), height=460)
        return

    display = frame.rename(
        columns={
            "prediction_id": "ID",
            "timestamp": "Time",
            "src": "Source",
            "dst": "Destination",
            "protocol": "Proto",
            "prediction": "ML prediction",
            "behaviour": "Behaviour",
            "final": "Final decision",
            "reason": "Reason",
            "confidence": "Confidence",
            "risk_score": "Risk",
            "risk_level": "Band",
            "packets": "Packets",
            "bytes": "Bytes",
            "ground_truth": "Demo label",
        }
    )
    columns = [
        c for c in (
            "ID", "Time", "Source", "Destination", "Proto", "ML prediction", "Behaviour",
            "Final decision", "Confidence", "Risk", "Band", "Reason",
            "Packets", "Bytes", "Demo label",
        ) if c in display.columns
    ]
    tables.dataframe(display[columns], height=460)

    split = st.columns(2)
    with split[0]:
        counts = frame["prediction"].value_counts().reset_index()
        counts.columns = ["prediction", "count"]
        tables.chart_with_table(
            charts.class_distribution_bar(
                counts, "prediction", "count",
                title="Predictions in the live buffer", height=300,
            ),
            counts,
            key="lm_dist",
        )
    with split[1]:
        bands = frame["risk_level"].value_counts().to_dict()
        tables.chart_with_table(
            charts.risk_band_bar(bands, height=300),
            pd.DataFrame([{"band": k, "flows": v} for k, v in bands.items()]),
            key="lm_bands",
        )


def render() -> None:
    with st.container(key="card_lm_controls"):
        _controls()
    enabled, interval = refresh_controls("lm", default_interval=5)
    if enabled and auto_refresh_supported():
        fragment(run_every=interval)(_live_body)()
    else:
        _live_body()
        if enabled and not auto_refresh_supported():
            st.caption(
                "This Streamlit build has no fragment timer, so use **Refresh now**."
            )


__all__ = ["render", "PAGE_TITLE"]
