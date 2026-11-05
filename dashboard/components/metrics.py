"""Stat tiles, hero figures, status pills and risk badges.

Per the visualisation guidance a single headline number is a **stat tile**, not
a one-bar chart, and a status colour is always paired with its text label so the
meaning never rests on colour alone.
"""

from __future__ import annotations

import html
import re
from typing import Any, Mapping, Sequence

import streamlit as st

from dashboard.styles import theme

ACCENTS = {
    "blue": "accent-blue",
    "red": "accent-red",
    "low": "accent-low",
    "medium": "accent-medium",
    "high": "accent-high",
    "critical": "accent-critical",
}


def stat_tile(label: str, value: Any, delta: str = "", accent: str = "") -> None:
    """One stat tile: an uppercase label, a large value and an optional note."""
    accent_class = ACCENTS.get(str(accent).lower(), "")
    st.markdown(
        f"""
        <div class="nids-tile {accent_class}">
          <div class="label">{html.escape(str(label))}</div>
          <div class="value">{html.escape(str(value))}</div>
          <div class="delta">{html.escape(str(delta))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_row(tiles: Sequence[Mapping[str, Any]]) -> None:
    """Render a KPI row of stat tiles."""
    if not tiles:
        return
    cells = "".join(
        f'<div class="nids-tile"><div class="label">{html.escape(str(tile.get("label", "")))}</div>'
        f'<div class="value">{html.escape(str(tile.get("value", "")))}</div>'
        f'<div class="delta">{html.escape(str(tile.get("delta", "")))}</div></div>'
        for tile in tiles
    )
    st.markdown(f'<div class="nids-stats">{cells}</div>', unsafe_allow_html=True)


def hero(value: Any, label: str, colour: str | None = None) -> None:
    """The one number a page leads with."""
    style = f' style="color:{colour}"' if colour else ""
    st.markdown(
        f"""
        <div>
          <div class="nids-hero-label">{html.escape(str(label))}</div>
          <div class="nids-hero"{style}>{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(level: str, score: float | None = None) -> str:
    """Inline HTML badge - colour plus the band's own name, never colour alone."""
    band = str(level).upper()
    icon = theme.STATUS_ICON.get(band, "●")
    text = f"{icon} {band.title()}" + (f" · {score:.0f}" if score is not None else "")
    return f'<span class="nids-risk {band}">{html.escape(text)}</span>'


def pill(text: str, kind: str = "info") -> str:
    """Inline status pill (``ok``/``idle``/``warn``/``danger``/``info``)."""
    return f'<span class="nids-pill {kind}">{html.escape(str(text))}</span>'


def header(title: str, subtitle: str, pills: Sequence[str] = (), eyebrow: str = "") -> None:
    """The page heading shown on every page."""
    eyebrow_html = f'<div class="eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    st.markdown(
        f"""<div class="nids-header"><div>{eyebrow_html}<h1>{html.escape(title)}</h1>
        <div class="sub">{html.escape(subtitle)}</div></div>
        <div class="pills">{"".join(pills)}</div></div>""",
        unsafe_allow_html=True,
    )


#: Inline so it renders with no external asset and no internet connection.
SHIELD_SVG = (
    '<svg viewBox="0 0 40 40" fill="none" role="img" aria-label="NetLens shield" '
    'stroke="currentColor" stroke-width="2.500" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M20 4 33 9v10c0 8-5.400 14-13 17C12.400 33 7 27 7 19V9l13-5Z"/>'
    '<path d="M12.500 20.500h4l2.200-5.500 3.600 9.500 2.200-4h3"/></svg>'
)


def brand(name: str, tagline: str) -> None:
    """The product mark at the top of the sidebar."""
    st.markdown(
        f"""<div class="nids-brand">{SHIELD_SVG}<div>
        <div class="name">{html.escape(name)}</div>
        <div class="tag">{html.escape(tagline)}</div></div></div>""",
        unsafe_allow_html=True,
    )


def system_card(
    running: bool, mode: str, flows: int, attacks: int, alerts: int, model: str, classes: int | None
) -> None:
    """Live monitoring state and session counters for the sidebar."""
    state = f"Monitoring · {mode}" if running else "Monitoring stopped"
    model_row = (
        f"<span>Model</span><b>{html.escape(model)} · {classes} classes</b>"
        if classes is not None
        else '<span>Model</span><b>Not trained</b>'
    )
    st.markdown(
        f"""<div class="nids-system">
        <div class="row"><span class="state"><i class="nids-dot{' on' if running else ''}"></i>
        {html.escape(state)}</span></div>
        <div class="mini">
        <div><b>{flows:,}</b><span>flows</span></div>
        <div><b>{attacks:,}</b><span>attacks</span></div>
        <div><b>{alerts:,}</b><span>alerts</span></div></div>
        <div class="model">{model_row}</div></div>""",
        unsafe_allow_html=True,
    )


def posture_banner(
    band: str, attack_share: float, benign: int, attack: int, band_counts: Mapping[str, int]
) -> None:
    """Threat posture: the page's headline, the traffic split and alerts by severity."""
    level = str(band).upper()
    benign_share = max(0.0, 100.0 - float(attack_share))
    segments = "".join(
        f'<i style="width:{share:.2f}%;background:{colour}"></i>'
        for share, colour in (
            (benign_share, theme.BENIGN), (float(attack_share), theme.ATTACK)
        )
        if share > 0
    )
    bands = "".join(
        f"<div>{risk_badge(name)}<b>{int(band_counts.get(name, 0)):,}</b></div>"
        for name in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    )
    summary = {
        "CRITICAL": "Critical alerts are on record",
        "HIGH": "High-risk alerts are on record",
        "MEDIUM": "Medium-risk alerts are on record",
    }.get(level, "No alerts at medium risk or above")
    st.markdown(
        f"""<div class="nids-posture {level}">
        <div><div class="k">Threat level</div>
        <div class="band"><span class="mark">{theme.STATUS_ICON.get(level, "●")}</span>{level.title()}</div>
        <div class="why">{summary}</div></div>
        <div><div class="k">Attack share of classified flows</div>
        <div class="share"><b>{float(attack_share):.1f}%</b>
        <span>{attack:,} of {benign + attack:,} flows</span></div>
        <div class="nids-split">{segments}</div>
        <div class="nids-legend">
        <span><i style="background:{theme.BENIGN}"></i>Benign {benign:,}</span>
        <span><i style="background:{theme.ATTACK}"></i>Attack {attack:,}</span></div></div>
        <div><div class="k" style="margin-bottom:.45rem">Stored alerts by severity</div>
        <div class="nids-bands">{bands}</div></div></div>""",
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str) -> None:
    """Friendly first-run message; the caller adds the action underneath."""
    st.markdown(
        f"""<div class="nids-empty">{SHIELD_SVG}<h3>{html.escape(title)}</h3>
        <p>{html.escape(body)}</p></div>""",
        unsafe_allow_html=True,
    )


def panel(title: str, body_markdown: str) -> None:
    """A bordered explanatory panel."""
    st.markdown(
        f"""
        <div class="nids-panel">
          <h4>{html.escape(title)}</h4>
          <div class="nids-note">{body_markdown}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    # Markdown is not parsed inside an HTML block, so **bold** is converted here.
    body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", str(text))
    st.markdown(f'<div class="nids-note">{body}</div>', unsafe_allow_html=True)


def confidence_vs_risk(confidence: float, risk_score: float, risk_level: str) -> None:
    """Show confidence and risk side by side - they are different quantities."""
    columns = st.columns(2)
    with columns[0]:
        stat_tile(
            "Model confidence",
            f"{float(confidence):.1%}",
            "probability the model assigns to the predicted class",
            accent="blue",
        )
    with columns[1]:
        stat_tile(
            "Risk score",
            f"{float(risk_score):.0f} / 100",
            f"severity band: {str(risk_level).title()}",
            accent=str(risk_level).lower(),
        )


__all__ = [
    "stat_tile",
    "stat_row",
    "hero",
    "risk_badge",
    "pill",
    "header",
    "brand",
    "system_card",
    "posture_banner",
    "empty_state",
    "panel",
    "note",
    "confidence_vs_risk",
]
