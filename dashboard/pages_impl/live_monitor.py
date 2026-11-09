"""LIVE MONITOR - start/stop capture and watch flows being classified."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components import metrics, tables, ui
from dashboard.styles import theme
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

PAGE_TITLE = "Live monitor"


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

    columns = st.columns([1.1, 2, 1, 1.2], vertical_alignment="bottom")
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
        if status["running"]:
            if ui.ghost_button("Stop monitoring", width="stretch"):
                monitor.stop()
                st.rerun()
        else:
            if ui.primary_button("Start monitoring", width="stretch"):
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
        if ui.ghost_button("Classify 25 demo flows",
                     width="stretch",
                     disabled=status["running"],
                     help="Runs a single batch through the full pipeline without "
                          "starting continuous monitoring."):
            try:
                count = monitor.demo_burst(25)
                st.success(f"Classified {count} demo flow{'s' if count != 1 else ''}")
            except NIDSError as exc:
                st.error(str(exc))

    metrics.note(_mode_help(mode))


def _live_body() -> None:
    monitor = load_monitor()
    database = load_database()
    settings = load_settings()
    status = monitor.status()
    stats = status["stats"]

    # Running state is in the page header; this line adds only what is new.
    strip = [
        (f"uptime {format_duration(stats['uptime_seconds'])}", None),
        (f"alert floor {str(status['alert_min_band']).upper()}", None),
    ]
    if "capture" in status:
        capture = status["capture"]
        strip.append((f"iface {capture['interface']}", None))
        if capture["last_error"]:
            strip.append(("capture error", theme.NODE_MAGENTA))
    metrics.status_strip(strip)

    metrics.stat_row(
        [
            {"label": "Flows this session", "value": f"{stats['flows_processed']:,}",
             "delta": f"uptime {format_duration(stats['uptime_seconds'])}"},
            {"label": "Attack flows, final decision", "value": f"{stats['final_attacks']:,}",
             "delta": f"{stats['final_suspicious']:,} suspicious · "
                      f"{stats['attacks_detected']:,} flagged by the model", "accent": True},
            {"label": "Behaviour detections", "value": f"{stats['behaviour_findings']:,}",
             "delta": f"{status.get('behaviour', {}).get('connection_attempts', 0):,} "
                      "connection attempts tracked"},
            {"label": "Alerts raised", "value": f"{stats['alerts_raised']:,}",
             "delta": f"at or above {str(status['alert_min_band']).upper()}"},
            {"label": "Packets seen", "value": f"{stats['packets_seen']:,}",
             "delta": (f"{status.get('flows', {}).get('active_flows', 0):,} flows in assembly")
             if "flows" in status else "demo mode uses stored flow records"},
        ]
    )

    if status["errors"]:
        with st.expander(f"{len(status['errors'])} error(s) reported", expanded=False):
            for message in reversed(status["errors"]):
                st.write(f"- {message}")

    limit = int(settings.get("dashboard.recent_rows", 200))
    frame = monitor.recent_frame(limit)
    if not frame.empty:
        _alert_feed(frame, benign=str(settings.get("inference.benign_label", "BENIGN")))
        split = st.columns(2)
        with split[0]:
            counts = frame["prediction"].astype(str).value_counts()
            tables.list_card(
                "Predictions in the live buffer",
                ui.ranked_list_html(counts.items(), label_html=ui.attack_badge, limit=8),
                key="lm_dist", meta=f"last {len(frame):,} flows",
            )
        with split[1]:
            bands = frame["risk_level"].astype(str).str.upper().value_counts().to_dict()
            tables.list_card(
                "Risk bands in the live buffer",
                ui.severity_list_html(bands, empty="No flows scored yet"),
                key="lm_bands", meta="flows",
            )

    st.subheader("Behavioural detections")
    findings = monitor.findings_frame()
    if not findings.empty:
        tables.dataframe(tables.format_session_findings(findings), height=200)
    elif status["running"]:
        # Only this session's stored findings, so a new run starts clean.
        current = getattr(database, "current_session_id", None)
        stored = database.recent_behaviour_events(20, session_id=current)
        if stored.empty:
            st.caption("No behavioural pattern has been detected in this session")
        else:
            tables.dataframe(tables.format_behaviour_events(stored), height=200)
    else:
        st.caption("No behavioural pattern has been detected in this session")

    st.subheader("Live flow table")
    if frame.empty:
        recent = database.recent_predictions(limit)
        if recent.empty:
            st.caption("No flows classified yet. Press Start monitoring above.")
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


def _alert_feed(frame: pd.DataFrame, *, benign: str, limit: int = 8) -> None:
    """The newest non-benign decisions in the live buffer, one line each.

    Severity is the meter; ML confidence is its own column. It refreshes with
    the live view, so nothing in it animates.
    """
    if "final" not in frame.columns:
        return
    flagged = frame[frame["final"].astype(str).str.upper() != benign.upper()].head(limit)
    st.subheader("Live alert feed")
    rows = [
        {
            "time": str(row.get("timestamp", ""))[11:19],
            "band": row.get("risk_level", ""),
            "score": row.get("risk_score"),
            "label": row.get("final", ""),
            "flow": f"{row.get('src', '')} → {row.get('dst', '')} · {row.get('protocol', '')}",
            "confidence": row.get("confidence"),
        }
        for _, row in flagged.iterrows()
    ]
    ui.panel(ui.log_html(rows, empty="Nothing but benign traffic in the live buffer"))


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
