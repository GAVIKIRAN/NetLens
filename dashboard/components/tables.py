"""Table rendering helpers.

Every chart in this dashboard is paired with a table view, which is both the
accessibility fallback for colour-encoded values and the fastest way for an
analyst to copy a number out.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from dashboard.components import compat, ui
from dashboard.styles import theme

RISK_ORDER = list(theme.RISK_ORDER)
#: Severity node lights; each passes 4.5:1 as text on the Carbon table surface.
BAND_TEXT = dict(theme.STATUS)


def chart_with_table(
    figure,
    frame: pd.DataFrame | None,
    *,
    key: str,
    table_label: str = "Data table",
    meta: str = "",
) -> None:
    """A Plotly figure in a card, with its table-view twin in a quiet disclosure.

    The figure's title is lifted out of Plotly into the card header, so chart
    cards and list cards share one heading style and alignment.
    """
    title = str(figure.layout.title.text or "")
    figure.update_layout(title_text="", margin_t=8)
    with card(key):
        if title:
            ui.render(ui.card_header_html(title, meta))
        compat.plotly_chart(figure, key=key)
        if frame is not None and not frame.empty:
            with st.expander(table_label, expanded=False):
                compat.dataframe(frame)


def list_card(title: str, body_html: str, *, key: str, meta: str = "") -> None:
    """A card holding a ranked list (or any HTML), the same height as its row."""
    with card(key):
        ui.render(ui.card_header_html(title, meta) + body_html)


def card(key: str):
    """A stretching container; the ``card_`` key makes the stylesheet draw it as a panel."""
    try:
        return st.container(key=f"card_{key}", height="stretch")
    except TypeError:  # pragma: no cover - older Streamlit
        return st.container(key=f"card_{key}")


_card = card


def _band_style(value: Any) -> str:
    colour = BAND_TEXT.get(str(value).upper())
    return f"color: {colour}" if colour else ""


def _band_label(value: Any) -> str:
    """The severity meter as text (``●●●○  HIGH``), for canvas-drawn table cells."""
    text = str(value)
    icon = theme.STATUS_ICON.get(text.upper())
    return f"{icon}  {text.upper()}" if icon else text


def _default_columns(frame: pd.DataFrame) -> dict[str, Any]:
    """Show risk as a bar for scanning, and confidence as a percentage.

    Confidence is not a bar because it is nearly always close to 100%, so the bars
    would all look alike.
    """
    config: dict[str, Any] = {}
    if "Confidence" in frame.columns:
        config["Confidence"] = st.column_config.NumberColumn("Confidence", format="percent")
    if "Risk" in frame.columns:
        config["Risk"] = st.column_config.ProgressColumn(
            "Risk", min_value=0.0, max_value=100.0, format="%.0f", width="small"
        )
    return config


def dataframe(
    frame: pd.DataFrame,
    *,
    height: int | None = None,
    column_config: Mapping[str, Any] | None = None,
    hide_index: bool = True,
) -> None:
    """Thin wrapper so every table in the app looks the same."""
    if frame is None or frame.empty:
        st.caption("No rows to display.")
        return
    config = {**_default_columns(frame), **dict(column_config or {})}
    bands = [c for c in ("Band", "Severity") if c in frame.columns]
    styled = (
        frame.style.map(_band_style, subset=bands).format(_band_label, subset=bands)
        if bands else frame
    )
    compat.dataframe(
        styled,
        height=height,
        hide_index=hide_index,
        column_config=config or None,
    )


def format_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare the prediction table for display."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if "timestamp" in data.columns:
        data["timestamp"] = data["timestamp"].astype(str).str.replace("T", " ", regex=False)
        data["timestamp"] = data["timestamp"].str.slice(0, 19)
    if {"src_ip", "src_port"}.issubset(data.columns):
        data["source"] = (
            data["src_ip"].fillna("").astype(str) + ":" + data["src_port"].fillna(0).astype(int).astype(str)
        )
    if {"dst_ip", "dst_port"}.issubset(data.columns):
        data["destination"] = (
            data["dst_ip"].fillna("").astype(str) + ":" + data["dst_port"].fillna(0).astype(int).astype(str)
        )
    if "confidence" in data.columns:
        data["confidence"] = data["confidence"].astype(float).round(4)
    if "risk_score" in data.columns:
        data["risk_score"] = data["risk_score"].astype(float).round(1)
    columns = [
        c
        for c in (
            "id", "timestamp", "source", "destination", "protocol_name",
            "prediction", "behaviour_result", "final_classification", "confidence",
            "risk_score", "risk_level", "detection_reason",
            "ground_truth", "packet_count", "byte_count", "mode",
        )
        if c in data.columns
    ]
    return data[columns].rename(
        columns={
            "id": "ID",
            "timestamp": "Time (UTC)",
            "source": "Source",
            "destination": "Destination",
            "protocol_name": "Proto",
            "prediction": "ML prediction",
            "behaviour_result": "Behaviour",
            "final_classification": "Final decision",
            "confidence": "Confidence",
            "risk_score": "Risk",
            "risk_level": "Band",
            "detection_reason": "Reason",
            "ground_truth": "Demo label",
            "packet_count": "Packets",
            "byte_count": "Bytes",
            "mode": "Mode",
        }
    )


def format_session_findings(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare this session's behavioural detections for display."""
    return frame.rename(
        columns={
            "timestamp": "Time",
            "source": "Source",
            "target": "Target",
            "detection": "Detection",
            "decision": "Final decision",
            "risk_score": "Risk",
            "risk_level": "Band",
            "reason": "Reason",
        }
    )


def format_behaviour_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare stored behavioural detections for display."""
    from src.detection.behavior import KIND_LABELS

    data = frame.copy()
    data["timestamp"] = data["timestamp"].astype(str).str.replace("T", " ", regex=False).str.slice(0, 19)
    data["kind"] = data["kind"].map(lambda kind: KIND_LABELS.get(kind, kind))
    data["target"] = [
        (ip or "multiple hosts") + (f":{int(port)}" if port else "")
        for ip, port in zip(data["dst_ip"], data["dst_port"])
    ]
    return data[
        ["timestamp", "src_ip", "target", "kind", "decision", "risk_score", "risk_level", "reason"]
    ].rename(
        columns={
            "timestamp": "Time (UTC)",
            "src_ip": "Source",
            "target": "Target",
            "kind": "Detection",
            "decision": "Final decision",
            "risk_score": "Risk",
            "risk_level": "Band",
            "reason": "Reason",
        }
    )


def format_alerts(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare the alert table for display."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if "timestamp" in data.columns:
        data["timestamp"] = data["timestamp"].astype(str).str.replace("T", " ", regex=False).str.slice(0, 19)
    if {"src_ip", "src_port"}.issubset(data.columns):
        data["source"] = data["src_ip"].astype(str) + ":" + data["src_port"].astype(int).astype(str)
    if {"dst_ip", "dst_port"}.issubset(data.columns):
        data["destination"] = data["dst_ip"].astype(str) + ":" + data["dst_port"].astype(int).astype(str)
    for column in ("confidence", "risk_score"):
        if column in data.columns:
            data[column] = data[column].astype(float).round(3)
    columns = [
        c
        for c in (
            "id", "timestamp", "severity", "attack_type", "source", "destination",
            "protocol_name", "confidence", "risk_score", "prediction_id", "mode",
        )
        if c in data.columns
    ]
    return data[columns].rename(
        columns={
            "id": "Alert",
            "timestamp": "Time (UTC)",
            "severity": "Severity",
            "attack_type": "Attack type",
            "source": "Source",
            "destination": "Destination",
            "protocol_name": "Proto",
            "confidence": "Confidence",
            "risk_score": "Risk",
            "prediction_id": "Prediction",
            "mode": "Mode",
        }
    )


def contribution_table(contributions: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Tabular twin of the contribution chart."""
    rows = []
    for entry in contributions or []:
        rows.append(
            {
                "Feature": entry.get("label") or entry.get("feature", ""),
                "Canonical name": entry.get("feature", ""),
                "Value": entry.get("value"),
                "Contribution": round(float(entry.get("contribution", 0.0)), 6),
                "Direction": entry.get("direction", ""),
                "Condition": entry.get("condition", ""),
            }
        )
    frame = pd.DataFrame(rows)
    if "Condition" in frame.columns and not frame["Condition"].astype(str).str.strip().any():
        frame = frame.drop(columns=["Condition"])
    return frame


def sort_by_risk(frame: pd.DataFrame, column: str = "Band") -> pd.DataFrame:
    """Sort a formatted table by severity band, most severe first."""
    if frame is None or frame.empty or column not in frame.columns:
        return frame
    order = {band: index for index, band in enumerate(reversed(RISK_ORDER))}
    data = frame.copy()
    data["_rank"] = data[column].astype(str).str.upper().map(order).fillna(-1)
    return data.sort_values("_rank", ascending=False).drop(columns=["_rank"])


__all__ = [
    "chart_with_table",
    "list_card",
    "card",
    "dataframe",
    "format_predictions",
    "format_alerts",
    "format_behaviour_events",
    "format_session_findings",
    "contribution_table",
    "sort_by_risk",
    "RISK_ORDER",
]
