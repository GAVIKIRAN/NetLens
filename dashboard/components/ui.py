"""NetLens component helpers - the only place page code gets custom markup from.

``*_html`` functions return an HTML string, so components compose (a severity
meter inside a list row, a tag inside a stat). The other functions render.
Everything interpolated into markup is escaped unless the parameter says it
takes HTML. Styles live in ``dashboard/styles/theme.css``.

Redesigned with unslop by Damir Sokolovsky (@heydamir), CC BY 4.0.
"""

from __future__ import annotations

import html
import math
import re
import zlib
from typing import Any, Callable, Iterable, Mapping, Sequence

import streamlit as st

from dashboard.styles import theme

_e = html.escape


def render(markup: str) -> None:
    """Render trusted markup. Whitespace between tags is removed so Markdown
    can never mistake an indented line for a code block."""
    st.markdown(re.sub(r">\s+<", "><", markup.strip()), unsafe_allow_html=True)


# ================================================================== lights ==
def dot(colour: str, *, ring: bool = False) -> str:
    """An 8 px status light. ``ring`` draws it hollow, for secondary categories."""
    fill = "transparent" if ring else colour
    return (
        f'<i class="nl-dot{" ring" if ring else ""}" '
        f'style="background:{fill};border-color:{colour}" aria-hidden="true"></i>'
    )


def pips(level: str, *, large: bool = False) -> str:
    """The signature severity meter: 1-4 of four lights, lit in the band's colour."""
    band = str(level).upper()
    lit = theme.SEVERITY_LEVEL.get(band, 0)
    lights = "".join(f'<i class="{"on" if i < lit else ""}"></i>' for i in range(4))
    return (
        f'<span class="nl-pips{" nl-pips--lg" if large else ""}" '
        f'style="--pip:{theme.risk_color(band)}" role="img" '
        f'aria-label="{_e(band.title())} severity, {lit} of 4">{lights}</span>'
    )


def severity_badge(level: str, score: float | None = None) -> str:
    """Meter + band code (+ risk score). The code is data, so it stays as stored."""
    band = str(level).upper()
    text = band + (f" · {float(score):.0f}" if score is not None else "")
    return f'<span class="nl-sev">{pips(band)}<span>{_e(text)}</span></span>'


def attack_badge(label: str) -> str:
    """Attack class: the family's light + the exact class name."""
    colour, ring = theme.ATTACK_FAMILIES[theme.attack_family(label)]
    return f'<span class="nl-tag">{dot(colour, ring=ring)}<span>{_e(str(label))}</span></span>'


def status_text(text: str, state: str = "idle") -> str:
    """Quiet status: a light and plain text. ``state``: live | idle | error | info."""
    colour = {
        "live": theme.NODE_GREEN, "error": theme.NODE_MAGENTA, "info": theme.STEEL,
    }.get(state, theme.STEEL)
    return f'<span class="nl-status {state}">{dot(colour)}<span>{_e(text)}</span></span>'


# =================================================================== text ==
def page_header(title: str, subtitle: str, meta: Iterable[str] = ()) -> None:
    """Page title, one line of description, and quiet status on the right."""
    render(
        f'<header class="nl-header"><div class="nl-header-text"><h1>{_e(title)}</h1>'
        f'<p>{_e(subtitle)}</p></div>'
        f'<div class="nl-header-meta">{"".join(meta)}</div></header>'
    )


def section(title: str, meta: str = "") -> None:
    """Section heading with optional mono meta on the right."""
    tail = f'<span class="nl-meta">{_e(meta)}</span>' if meta else ""
    render(f'<div class="nl-section"><h3>{_e(title)}</h3>{tail}</div>')


def meta_line(items: Iterable[str]) -> None:
    """A line of mono meta. Items are HTML (use :func:`meta_item`)."""
    cells = "".join(f"<span>{item}</span>" for item in items if item)
    if cells:
        render(f'<div class="nl-metaline">{cells}</div>')


def meta_item(text: str, colour: str | None = None) -> str:
    return (dot(colour) if colour else "") + _e(str(text))


# ============================================================== containers ==
def card_header_html(title: str, meta: str = "") -> str:
    tail = f'<span class="nl-meta">{_e(meta)}</span>' if meta else ""
    return f'<div class="nl-card-head"><h4>{_e(title)}</h4>{tail}</div>'


def panel_html(body: str, *, title: str = "", meta: str = "", cls: str = "") -> str:
    """A carbon panel. ``body`` is HTML."""
    head = card_header_html(title, meta) if title else ""
    return f'<section class="nl-panel {cls}">{head}{body}</section>'


def panel(body: str, *, title: str = "", meta: str = "", cls: str = "") -> None:
    render(panel_html(body, title=title, meta=meta, cls=cls))


# ==================================================================== data ==
def stats_html(cells: Sequence[Mapping[str, Any]]) -> str:
    """One strip of figures divided by hairlines.

    Cell keys: ``label``, ``value``, optional ``note``, ``badge`` (HTML shown
    next to the label) and ``accent`` (the one figure that matters, in orange).
    """
    out = []
    for cell in cells:
        badge = cell.get("badge") or ""
        accent = " accent" if cell.get("accent") else ""
        note = cell.get("note") or cell.get("delta") or ""
        out.append(
            '<div class="cell">'
            f'<div class="k"><span>{_e(str(cell.get("label", "")))}</span>{badge}</div>'
            f'<div class="fig{accent}">{_e(str(cell.get("value", "")))}</div>'
            + (f'<div class="note">{_e(str(note))}</div>' if note else "")
            + "</div>"
        )
    return f'<div class="nl-stats">{"".join(out)}</div>'


def stats(cells: Sequence[Mapping[str, Any]]) -> None:
    if cells:
        render(stats_html(cells))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def ranked_list_html(
    rows: Iterable[tuple[Any, Any]],
    *,
    fmt: str = "{:,.0f}",
    label_html: Callable[[str], str] | None = None,
    accent_first: bool = True,
    sort: bool = True,
    limit: int | None = None,
    empty: str = "No data yet",
) -> str:
    """A ranked list with thin bars - the compact alternative to a second chart.

    The first row carries the accent when the list is sorted; the rest stay
    neutral. Past ``limit`` rows the tail folds into one "Other" row.
    """
    items = [(str(label), v) for label, value in rows if (v := _finite(value)) is not None]
    if not items:
        return f'<p class="nl-empty-line">{_e(empty)}</p>'
    if sort:
        items.sort(key=lambda item: item[1], reverse=True)
    if limit and len(items) > limit:
        tail = sum(v for _, v in items[limit:])
        items = items[:limit] + [("Other", tail)]
    top = max(abs(v) for _, v in items) or 1.0
    lis = []
    for index, (label, value) in enumerate(items):
        lead = accent_first and sort and index == 0 and label != "Other"
        width = 0.0 if value == 0 else max(1.5, 100.0 * abs(value) / top)
        shown = label_html(label) if label_html and label != "Other" else _e(label)
        lis.append(
            f'<li class="{"lead" if lead else ""}"><span class="lbl">{shown}</span>'
            f'<span class="val">{_e(fmt.format(value))}</span>'
            f'<span class="bar" aria-hidden="true"><i style="width:{width:.1f}%"></i></span></li>'
        )
    return f'<ol class="nl-rank">{"".join(lis)}</ol>'


def severity_list_html(counts: Mapping[str, Any], *, empty: str = "No alerts yet") -> str:
    """Alerts by band, most severe first, each row led by the meter."""
    rows = [(band, int(counts.get(band, 0) or 0)) for band in reversed(theme.RISK_ORDER)]
    if not any(count for _, count in rows):
        return f'<p class="nl-empty-line">{_e(empty)}</p>'
    return ranked_list_html(
        rows, label_html=lambda band: severity_badge(band), accent_first=False, sort=False,
    )


def kv_html(pairs: Iterable[tuple[str, str]], *, compact: bool = False) -> str:
    """Key-value rows. Values are HTML - escape plain text with :func:`html.escape`."""
    rows = "".join(f"<div><dt>{_e(k)}</dt><dd>{v}</dd></div>" for k, v in pairs)
    return f'<dl class="nl-kv{" compact" if compact else ""}">{rows}</dl>'


def log_html(rows: Sequence[Mapping[str, Any]], *, empty: str = "Nothing to show") -> str:
    """Alert log: time, severity, class, flow and ML confidence on one line each.

    Severity (the meter) and confidence (a plain percentage) never share a column.
    """
    if not rows:
        return f'<p class="nl-empty-line">{_e(empty)}</p>'
    head = (
        '<div class="row head" role="row"><span class="t">Time</span><span class="s">Severity</span>'
        '<span class="c">Class</span><span class="f">Flow</span>'
        '<span class="p">Confidence</span></div>'
    )
    body = []
    for row in rows:
        score = _finite(row.get("score"))
        conf = _finite(row.get("confidence"))
        body.append(
            '<div class="row" role="row">'
            f'<span class="t mono">{_e(str(row.get("time", "")))}</span>'
            f'<span class="s">{severity_badge(str(row.get("band", "")), score)}</span>'
            f'<span class="c">{attack_badge(str(row.get("label", "")))}</span>'
            f'<span class="f mono">{_e(str(row.get("flow", "")))}</span>'
            f'<span class="p mono">{"" if conf is None else f"{conf:.1%}"}</span></div>'
        )
    return f'<div class="nl-log" role="table">{head}{"".join(body)}</div>'


# ================================================================ identity ==
#: Flat mark: a shield with a pulse line, drawn in one stroke colour.
SHIELD_SVG = (
    '<svg viewBox="0 0 40 40" fill="none" aria-hidden="true" stroke="currentColor" '
    'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M20 4 33 9v10c0 8-5.4 14-13 17C12.4 33 7 27 7 19V9l13-5Z"/>'
    '<path d="M12.5 20.5h4l2.2-5.5 3.6 9.5 2.2-4h3"/></svg>'
)


def wordmark_html(name: str) -> str:
    return f'<span class="nl-wordmark">{SHIELD_SVG}<b>{_e(name)}</b></span>'


def empty_state(title: str, body: str) -> None:
    """First-run message; the caller adds the action underneath."""
    render(f'<div class="nl-empty">{SHIELD_SVG}<h3>{_e(title)}</h3><p>{_e(body)}</p></div>')


# ============================================================= network map ==
def lattice_slots(cols: int = 10, rows: int = 8, cell: int = 56) -> list[tuple[float, float]]:
    """Lattice intersections strictly inside the frame. Two 45-degree line
    families cross where the half-cell x and y indices share parity."""
    half = cell / 2
    return [
        (x * half, y * half)
        for y in range(1, rows)
        for x in range(1, 2 * cols)
        if (x + y) % 2 == 0
    ]


def map_capacity(cols: int = 10, rows: int = 8) -> int:
    """How many hosts the map draws: 60% of the intersections, so it never looks full."""
    return int(len(lattice_slots(cols, rows)) * 0.6)


def lattice_map(
    hosts: Sequence[Mapping[str, Any]], *, cols: int = 10, rows: int = 8, cell: int = 56
) -> str:
    """The command center's moment: the network drawn as a diamond lattice.

    Each source host from recent traffic sits on a lattice intersection chosen
    from a hash of its address, so a host keeps its place between visits.
    Hosts that raised alerts light up in the colour of their worst band; the
    rest are small neutral points. ``hosts``: dicts with ``ip``, ``band``
    (or None) and ``flows``.
    """
    half = cell / 2
    width, height = cols * cell, rows * half
    lines = " ".join(
        f"M{i * cell} 0L{i * cell + height} {height}M{i * cell} 0L{i * cell - height} {height}"
        for i in range(-rows, cols + rows + 1)
    )
    slots = lattice_slots(cols, rows, cell)
    ranked = sorted(
        hosts,
        key=lambda h: (-theme.SEVERITY_LEVEL.get(str(h.get("band") or "").upper(), 0),
                       -int(h.get("flows") or 0)),
    )
    placed: dict[int, Mapping[str, Any]] = {}
    for host in ranked[: map_capacity(cols, rows)]:
        slot = zlib.crc32(str(host.get("ip", "")).encode()) % len(slots)
        while slot in placed:
            slot = (slot + 1) % len(slots)
        placed[slot] = host

    marks, lit = [], 0
    for slot, host in sorted(placed.items()):
        x, y = slots[slot]
        band = str(host.get("band") or "").upper()
        level = theme.SEVERITY_LEVEL.get(band, 0)
        tip = f"{host.get('ip', '')} · {int(host.get('flows') or 0):,} flows"
        if level:
            colour = theme.risk_color(band)
            ring = (
                f'<circle cx="{x}" cy="{y}" r="{9 + level}" fill="none" stroke="{colour}" '
                'stroke-opacity=".35"/>' if level >= 3 else ""
            )
            marks.append(
                f'<g class="node lit" style="--i:{min(lit, 6)}"><title>{_e(tip)} · {band}</title>'
                f'{ring}<circle cx="{x}" cy="{y}" r="{3.5 + level * 0.5}" fill="{colour}"/></g>'
            )
            lit += 1
        else:
            marks.append(
                f'<circle class="node" cx="{x}" cy="{y}" r="2.5" fill="{theme.STEEL}">'
                f"<title>{_e(tip)}</title></circle>"
            )
    flagged = sum(1 for h in placed.values() if h.get("band"))
    label = f"Network map: {len(placed)} source hosts, {flagged} raised alerts"
    return (
        f'<svg class="nl-map-svg" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMidYMid slice" role="img" aria-label="{_e(label)}">'
        f'<path d="{lines}" stroke="{theme.CHARCOAL}" stroke-width="1" fill="none"/>'
        f'{"".join(marks)}</svg>'
    )


# ================================================================= buttons ==
# Streamlit's own buttons are restyled by CSS; these wrappers name the intent.
def primary_button(label: str, **kwargs) -> bool:
    """Signal Orange - the one main action in a group."""
    return st.button(label, type="primary", **kwargs)


def secondary_button(label: str, **kwargs) -> bool:
    """Raised neutral surface with a hairline."""
    return st.button(label, type="secondary", **kwargs)


ghost_button = secondary_button


def export_button(label: str, **kwargs) -> bool:
    """Export is a primary action (the brief names it alongside Start and Run)."""
    return st.download_button(label, type="primary", **kwargs)


__all__ = [
    "render", "dot", "pips", "severity_badge", "attack_badge", "status_text",
    "page_header", "section", "meta_line", "meta_item",
    "card_header_html", "panel_html", "panel",
    "stats_html", "stats", "ranked_list_html", "severity_list_html", "kv_html", "log_html",
    "SHIELD_SVG", "wordmark_html", "empty_state", "lattice_slots", "map_capacity", "lattice_map",
    "primary_button", "secondary_button", "ghost_button", "export_button",
]
