"""SYSTEM SETTINGS - runtime configuration, artefact status and maintenance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.components import metrics, tables
from dashboard.state import (
    clear_caches,
    load_database,
    load_monitor,
    load_predictor,
    load_settings,
)
from src.inference.risk import RiskScorer
from src.network.interfaces import default_interface, list_interfaces, scapy_available
from src.network.monitor import MODE_DEMO, MODE_LIVE, MODE_PCAP
from src.utils.errors import NIDSError

PAGE_TITLE = "System Settings"


def render() -> None:
    settings = load_settings()
    monitor = load_monitor()
    database = load_database()

    st.caption(
        "Changes made here apply to the running session immediately. They are "
        "**not** written back to `config/config.yaml`; edit that file (or set a "
        "`NIDS_*` environment variable) to make a change permanent."
    )

    tabs = st.tabs(
        ["Capture", "Detection & risk", "Paths & artefacts", "Database", "Environment"]
    )

    # ------------------------------------------------------------- capture
    with tabs[0]:
        available, message = scapy_available()
        if not available:
            st.warning(f"Scapy is not usable: {message}. DEMO mode still works.")
        else:
            interfaces = list_interfaces()
            st.markdown(f"Scapy default interface: `{default_interface()}`")
            if interfaces:
                tables.dataframe(pd.DataFrame([i.to_dict() for i in interfaces]))
            else:
                st.warning(
                    "No interfaces were reported. On Windows install Npcap from "
                    "https://npcap.com with 'WinPcap API-compatible Mode' enabled."
                )

        columns = st.columns(2)
        with columns[0]:
            options = ["<Scapy default>"] + [i.name for i in (interfaces if available else [])]
            current = settings.get("network.interface")
            index = options.index(current) if current in options else 0
            chosen = st.selectbox("Network interface", options, index=index, key="set_iface")
            bpf = st.text_input(
                "BPF capture filter", value=str(settings.get("network.bpf_filter", "ip")),
                help="'ip' ignores ARP/LLDP noise; leave empty to capture everything.",
                key="set_bpf",
            )
            capture_duration = st.number_input(
                "Capture duration (seconds, 0 = until stopped)",
                min_value=0, max_value=86400,
                value=int(settings.get("network.capture_duration", 0)), step=10,
                key="set_duration",
            )
        with columns[1]:
            flow_timeout = st.number_input(
                "Flow timeout (s)", min_value=5.0, max_value=600.0,
                value=float(settings.get("network.flow_timeout", 120.0)), step=5.0,
                help="Idle time after which an open flow is closed and classified. "
                     "CICFlowMeter used 120 s for CIC-IDS2017.",
                key="set_flow_timeout",
            )
            activity_timeout = st.number_input(
                "Activity threshold (s)", min_value=0.5, max_value=60.0,
                value=float(settings.get("network.activity_timeout", 5.0)), step=0.5,
                help="Gap that separates 'active' bursts from 'idle' periods.",
                key="set_activity",
            )
            refresh_interval = st.number_input(
                "Dashboard refresh interval (s)", min_value=2, max_value=120,
                value=int(settings.get("dashboard.refresh_interval", 5)), step=1,
                key="set_refresh",
            )

        default_mode = st.radio(
            "Default monitoring mode", [MODE_DEMO, MODE_LIVE, MODE_PCAP],
            index=[MODE_DEMO, MODE_LIVE, MODE_PCAP].index(
                str(settings.get("dashboard.default_mode", MODE_DEMO)).upper()
            ) if str(settings.get("dashboard.default_mode", MODE_DEMO)).upper()
            in (MODE_DEMO, MODE_LIVE, MODE_PCAP) else 0,
            horizontal=True, key="set_mode",
        )

        if st.button("Apply capture settings", type="primary"):
            settings.set(
                "network.interface", None if chosen == "<Scapy default>" else chosen
            )
            settings.set("network.bpf_filter", bpf)
            settings.set("network.capture_duration", int(capture_duration))
            settings.set("network.flow_timeout", float(flow_timeout))
            settings.set("network.activity_timeout", float(activity_timeout))
            settings.set("dashboard.refresh_interval", int(refresh_interval))
            settings.set("dashboard.default_mode", default_mode)
            st.success("Applied. Restart monitoring for capture changes to take effect.")

    # ------------------------------------------------------ detection & risk
    with tabs[1]:
        scorer = RiskScorer.from_settings(settings)
        columns = st.columns(2)
        with columns[0]:
            threshold = st.slider(
                "Attack probability threshold",
                min_value=0.0, max_value=0.99,
                value=float(settings.get("inference.attack_probability_threshold", 0.6)),
                step=0.01,
                help="An attack class is only reported when its probability reaches "
                     "this value. Raise it to reduce false positives on a live network.",
                key="set_threshold",
            )
            alert_band = st.select_slider(
                "Raise alerts at or above",
                options=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                value=str(settings.get("risk.alert_min_band", "MEDIUM")),
                key="set_band",
            )
        with columns[1]:
            st.markdown("###### Risk scoring policy")
            st.json(scorer.describe(), expanded=False)

        metrics.panel(
            "Risk score is not model confidence",
            "<b>Confidence</b> is the probability the classifier assigns to the "
            "predicted class - a property of the model. <b>Risk score</b> combines "
            "class severity, that confidence and the flow's traffic volume - a "
            "property of the situation. Both are stored and displayed separately.",
        )

        if st.button("Apply detection settings", type="primary"):
            settings.set("inference.attack_probability_threshold", float(threshold))
            settings.set("risk.alert_min_band", alert_band)
            predictor = load_predictor()
            if predictor is not None:
                predictor.attack_threshold = float(threshold)
            monitor.alert_min_band = alert_band
            monitor.risk_scorer = RiskScorer.from_settings(settings)
            st.success("Applied to the running session.")

    # ------------------------------------------------------------ artefacts
    with tabs[2]:
        rows = []
        for key in (
            "raw_data_dir", "processed_data_dir", "demo_data_dir", "model_dir",
            "preprocessing_dir", "reports_dir", "log_dir", "database_path",
        ):
            path = settings.path(key)
            rows.append(
                {
                    "setting": f"paths.{key}",
                    "path": str(path),
                    "exists": path.exists(),
                }
            )
        demo_file = Path(str(settings.get("demo.source_file")))
        rows.append(
            {"setting": "demo.source_file", "path": str(demo_file), "exists": demo_file.exists()}
        )
        tables.dataframe(pd.DataFrame(rows))

        predictor = load_predictor()
        if predictor is None:
            st.warning(
                "No trained model is loaded. Run `python scripts/train_model.py`."
            )
        else:
            st.markdown("###### Loaded model")
            st.json(predictor.describe(), expanded=False)

        st.caption(
            "Paths are read-only here because the model artefacts, the database and "
            "the dataset must stay consistent with one another. Change them in "
            "`config/config.yaml` (or with `NIDS_PATHS__*` environment variables) "
            "and restart the app."
        )
        if st.button("Reload model and metric caches"):
            clear_caches()
            st.success("Caches cleared. The next page render reloads the artefacts.")

    # ------------------------------------------------------------- database
    with tabs[3]:
        counts = database.counts()
        metrics.stat_row(
            [
                {"label": "Flows", "value": f"{counts['flows']:,}", "accent": "blue"},
                {"label": "Predictions", "value": f"{counts['predictions']:,}"},
                {"label": "Alerts", "value": f"{counts['alerts']:,}", "accent": "red"},
                {"label": "System events", "value": f"{counts['system_events']:,}"},
            ]
        )
        st.caption(f"Database file: `{settings.path('database_path')}`")

        columns = st.columns(3)
        with columns[0]:
            retention = st.number_input(
                "Retention (days, 0 = keep everything)", min_value=0, max_value=3650,
                value=int(settings.get("database.retention_days", 30)), step=1,
                key="set_retention",
            )
            if st.button("Prune old rows"):
                try:
                    removed = database.prune(int(retention))
                    st.success(f"Pruned {removed:,} row(s).")
                except NIDSError as exc:
                    st.error(str(exc))
        with columns[1]:
            st.write("")
            confirm = st.checkbox("I understand this deletes all history", key="set_confirm")
            if st.button("Clear all data", disabled=not confirm):
                if monitor.is_running:
                    st.error("Stop monitoring before clearing the database.")
                else:
                    database.clear()
                    st.success("All flow, prediction, alert and event rows deleted.")
        with columns[2]:
            st.write("")
            if st.button("Show recent system events"):
                st.session_state["set_events"] = True

        if st.session_state.get("set_events"):
            tables.dataframe(database.recent_events(50), height=320)

    # ----------------------------------------------------------- environment
    with tabs[4]:
        from src.training.train import load_metadata

        metadata = load_metadata(settings.path("model_dir")) or {}
        st.markdown("###### Runtime")
        st.json(metadata.get("environment", {"note": "train a model to record this"}))
        st.markdown("###### Monitor status")
        st.json(monitor.status(), expanded=False)
        st.markdown("###### Effective configuration")
        st.json(settings.as_dict(), expanded=False)


__all__ = ["render", "PAGE_TITLE"]
