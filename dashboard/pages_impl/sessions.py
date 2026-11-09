"""SESSIONS - one row per monitoring run, with a per-session CSV export.

Every time monitoring starts, a new session is opened in the database and all of
that run's flows, predictions, alerts and behaviour events are tagged with its
id.  Previous sessions are kept, so each run can be reviewed and downloaded on
its own.  Rows recorded before sessions existed appear as "Legacy".
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components import metrics, tables, ui
from dashboard.state import load_database

PAGE_TITLE = "Sessions"


def _label(row: pd.Series) -> str:
    started = str(row.get("started_at", ""))[:19].replace("T", " ")
    mode = row.get("mode", "")
    extra = f" - {row['label']}" if row.get("label") else ""
    status = "running" if not row.get("ended_at") else "ended"
    return f"#{int(row['id'])} - {mode} - {started} ({status}){extra}"


def render() -> None:
    database = load_database()
    sessions = database.list_sessions()

    if sessions.empty:
        metrics.empty_state(
            "No sessions recorded yet",
            "Start monitoring from the Live Monitor, or classify a demo burst, "
            "and each run will be saved here as its own session.",
        )
        # Legacy rows (recorded before sessions existed) can still be exported.
        legacy = database.export_session(None)
        if not legacy.empty:
            ui.export_button(
                f"Download pre-session data ({len(legacy):,} flows, CSV)",
                data=legacy.to_csv(index=False).encode("utf-8"),
                file_name="nids_legacy.csv",
                mime="text/csv",
            )
        return

    st.subheader("Recorded sessions")
    summary = sessions.rename(
        columns={
            "id": "Session",
            "mode": "Mode",
            "started_at": "Started",
            "ended_at": "Ended",
            "label": "Label",
            "flows_processed": "Flows",
            "attacks_detected": "Attacks",
            "alerts_raised": "Alerts",
            "behaviour_findings": "Behaviour",
        }
    )
    for col in ("Started", "Ended"):
        summary[col] = summary[col].astype(str).str.slice(0, 19).str.replace("T", " ")
    tables.dataframe(summary, height=260)

    st.subheader("Review and download a session")

    options = list(sessions["id"])
    choice = st.selectbox(
        "Session",
        options,
        format_func=lambda sid: _label(sessions[sessions["id"] == sid].iloc[0]),
        key="sess_pick",
    )

    row = sessions[sessions["id"] == choice].iloc[0]
    metrics.stat_row(
        [
            {"label": "Flows", "value": f"{int(row['flows_processed']):,}"},
            {"label": "Attacks", "value": f"{int(row['attacks_detected']):,}"},
            {"label": "Alerts", "value": f"{int(row['alerts_raised']):,}"},
            {"label": "Behaviour", "value": f"{int(row['behaviour_findings']):,}"},
        ]
    )

    export = database.export_session(int(choice))
    if export.empty:
        st.caption("This session recorded no classified flows")
    else:
        tables.dataframe(export, height=380)

    ui.export_button(
        f"Download session #{int(choice)} (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name=f"nids_session_{int(choice)}.csv",
        mime="text/csv",
        disabled=export.empty,
    )
