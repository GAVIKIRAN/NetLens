"""Figures, status, severity and page chrome, built from :mod:`ui`.

Two rules hold everywhere:

* severity is the four-light meter plus the band code - never colour alone;
* **risk score** (a property of the situation) and **ML confidence** (a
  property of the classifier) are separate figures with separate encodings.
"""

from __future__ import annotations

import html
import re
from typing import Any, Mapping, Sequence

from dashboard.components import ui
from dashboard.styles import theme

_e = html.escape


# ================================================================== figures ==
def _cell(tile: Mapping[str, Any]) -> dict[str, Any]:
    band = tile.get("band")
    return {
        "label": tile.get("label", ""),
        "value": tile.get("value", ""),
        "note": tile.get("delta", tile.get("note", "")),
        "badge": tile.get("badge") or (ui.pips(band) if band else ""),
        "accent": bool(tile.get("accent") or tile.get("highlight")),
    }


def stat_row(tiles: Sequence[Mapping[str, Any]]) -> None:
    """One strip of figures. Keys: label, value, delta, band, badge, accent."""
    ui.stats([_cell(tile) for tile in tiles])


def stat_tile(label: str, value: Any, delta: str = "", **extra: Any) -> None:
    """A single figure (a one-cell strip)."""
    stat_row([{"label": label, "value": value, "delta": delta, **extra}])


def risk_badge(level: str, score: float | None = None) -> str:
    return ui.severity_badge(level, score)


def pill(text: str, kind: str = "info") -> str:
    """Quiet status text (kept for older call sites)."""
    return ui.status_text(text, {"ok": "live", "danger": "error"}.get(kind, "info"))


def status_strip(items: Sequence[tuple[str, str | None]]) -> None:
    """A mono meta line from ``(text, light colour or None)`` pairs."""
    ui.meta_line(ui.meta_item(text, colour) for text, colour in items)


# ============================================================== page chrome ==
def header(title: str, subtitle: str, meta: Sequence[str] = ()) -> None:
    ui.page_header(title, subtitle, meta)


def brand(name: str) -> None:
    ui.render(f'<div class="nl-brand">{ui.wordmark_html(name)}<span class="nl-brand-tag">Console</span></div>')


def system_card(
    running: bool, mode: str, flows: int, attacks: int, alerts: int, model: str, classes: int | None
) -> None:
    """Sidebar status: what is running, this session's counters, the model."""
    state = ui.status_text(
        f"{mode.title()} monitoring running" if running else "Monitoring stopped",
        "live" if running else "idle",
    )
    model_value = f"{_e(model)} · {classes} classes" if classes is not None else "not trained"
    ui.render(
        '<div class="nl-side">'
        '<div class="nl-side-label">Status</div>'
        f'<div class="nl-side-state">{state}</div>'
        + ui.kv_html(
            [("Flows", f"{flows:,}"), ("Attacks", f"{attacks:,}"),
             ("Alerts", f"{alerts:,}"), ("Model", model_value)],
            compact=True,
        )
        + "</div>"
    )


def posture_banner(
    band: str, attack_share: float, benign: int, attack: int, band_counts: Mapping[str, int]
) -> None:
    """Threat posture in one panel: level, traffic split, alerts by band."""
    level = str(band).upper()
    share = max(0.0, min(100.0, float(attack_share)))
    segments = "".join(
        f'<i style="width:{width:.2f}%;background:{colour}"></i>'
        for width, colour in ((100.0 - share, theme.BENIGN), (share, theme.ATTACK))
        if width > 0
    )
    summary = {
        "CRITICAL": "Critical alerts are on record",
        "HIGH": "High-risk alerts are on record",
        "MEDIUM": "Medium-risk alerts are on record",
    }.get(level, "No alerts at medium risk or above")
    ui.render(
        '<section class="nl-panel nl-posture">'
        '<div><div class="k">Threat level</div>'
        f'<div class="nl-level">{ui.pips(level, large=True)}<b>{_e(level.title())}</b></div>'
        f'<div class="note">{_e(summary)}</div></div>'
        '<div><div class="k">Attack share of classified flows</div>'
        f'<div class="fig">{share:.1f}%</div>'
        f'<div class="nl-split" aria-hidden="true">{segments}</div>'
        '<div class="nl-legend">'
        f'<span class="nl-tag">{ui.dot(theme.BENIGN)}<span>Benign <b class="mono">{benign:,}</b></span></span>'
        f'<span class="nl-tag">{ui.dot(theme.ATTACK)}<span>Attack <b class="mono">{attack:,}</b></span></span>'
        "</div></div>"
        '<div><div class="k">Stored alerts by severity</div>'
        f"{ui.severity_list_html(band_counts)}</div>"
        "</section>"
    )


def empty_state(title: str, body: str) -> None:
    ui.empty_state(title, body)


def panel(title: str, body_markup: str) -> None:
    """A panel of explanatory text. ``body_markup`` is trusted HTML from page code."""
    ui.panel(f'<div class="nl-note">{body_markup}</div>', title=title)


def note(text: str) -> None:
    # Markdown is not parsed inside an HTML block, so **bold** is converted here.
    body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", str(text))
    ui.render(f'<div class="nl-note">{body}</div>')


def confidence_vs_risk(confidence: float, risk_score: float, risk_level: str) -> None:
    """ML confidence and risk side by side, as separate figures."""
    ui.stats([
        {"label": "ML confidence", "value": f"{float(confidence):.1%}",
         "note": "probability of the predicted class"},
        {"label": "Risk score", "value": f"{float(risk_score):.0f} / 100",
         "badge": ui.severity_badge(risk_level),
         "note": "class severity, confidence and traffic volume"},
    ])


__all__ = [
    "stat_tile", "stat_row", "risk_badge", "pill", "status_strip", "header", "brand",
    "system_card", "posture_banner", "empty_state", "panel", "note", "confidence_vs_risk",
]
