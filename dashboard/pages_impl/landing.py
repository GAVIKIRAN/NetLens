"""LANDING - the public front of NetLens, separate from the console.

Built to DESIGN.md: a floating capsule nav, a full-width headline hero, a light section of
three Signal Orange product cards, a 2-column grid of dark category cards for
the attack classes, the real totals from the database, and a closing call to
action. Every figure is read from the database or the trained model.

The page writes nothing to the sidebar, so Streamlit shows none here.
"""

from __future__ import annotations

import html
import math
import random
from pathlib import Path

import streamlit as st

from dashboard.components import ui
from dashboard.components.side_nav import side_nav
from dashboard.state import load_database, load_monitor, load_predictor, load_training_metrics
from dashboard.styles import theme
from src.network.monitor import MODE_DEMO
from src.utils.errors import NIDSError

_e = html.escape

#: Shown when no model has been trained yet: the CIC-IDS2017 classes.
FALLBACK_CLASSES = (
    "BENIGN", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "DDoS",
    "PortScan", "FTP-Patator", "SSH-Patator", "Web Attack - Brute Force", "Web Attack - XSS", "Bot",
)


LANDING_CSS = Path(__file__).resolve().parents[1] / "styles" / "landing.css"


# ================================================================ helpers ==
def _inject_css() -> None:
    """The landing page's own sheet, layered on top of the global one."""
    try:
        sheet = LANDING_CSS.read_text(encoding="utf-8")
    except OSError:
        return
    st.markdown(f"<style>{sheet}</style>", unsafe_allow_html=True)


def _safe(fn, default):
    """Landing must render even before the database or model exist."""
    try:
        return fn()
    except Exception:  # noqa: BLE001 - a front page never shows a traceback
        return default


def _num(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else number


def _lattice_svg(cols: int = 9, rows: int = 12, cell: int = 64) -> str:
    """Decorative hero lattice: rhombus outlines in charcoal, a few node lights
    at intersections and the one magenta glow diamond DESIGN.md allows."""
    half = cell / 2
    width, height = cols * cell, rows * half
    lines = " ".join(
        f"M{i * cell} 0L{i * cell + height} {height}M{i * cell} 0L{i * cell - height} {height}"
        for i in range(-rows, cols + rows + 1)
    )
    slots = [(x * half, y * half) for y in range(1, rows) for x in range(1, 2 * cols)
             if (x + y) % 2 == 0]
    rng = random.Random(7)  # fixed seed: the picture is the same on every visit
    picks = rng.sample(slots, 14)
    palette = [theme.NODE_GREEN, theme.NODE_VIOLET, theme.SIGNAL_ORANGE, theme.NODE_CYAN,
               theme.STEEL, theme.STEEL]
    nodes = "".join(
        f'<circle class="lp-node" style="--i:{i}" cx="{x}" cy="{y}" r="{4 if c != theme.STEEL else 2.5}" fill="{c}"/>'
        for i, ((x, y), c) in enumerate(zip(picks, palette * 3))
    )
    gx, gy = (cols // 2) * cell, height - cell
    glow = (
        f'<path d="M{gx} {gy - half}L{gx + half} {gy}L{gx} {gy + half}L{gx - half} {gy}Z" '
        'fill="url(#lp-glow)" stroke="#ff2ad4" stroke-opacity=".6"/>'
    )
    return (
        f'<svg class="lp-lattice" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid slice" aria-hidden="true">'
        '<defs><radialGradient id="lp-glow"><stop offset="0" stop-color="#ff2ad4" stop-opacity=".9"/>'
        '<stop offset=".6" stop-color="#ff6314" stop-opacity=".35"/>'
        '<stop offset="1" stop-color="#ff2ad4" stop-opacity="0"/></radialGradient></defs>'
        f'<path d="{lines}" stroke="{theme.CHARCOAL}" stroke-width="1" fill="none"/>'
        f"{glow}{nodes}</svg>"
    )


def _dots_icon(pattern: tuple[str, ...]) -> str:
    """A white pixel-grid halftone icon (DESIGN.md product-card imagery)."""
    size = 6
    cells = "".join(
        f'<circle cx="{x * size + 3}" cy="{y * size + 3}" r="2"/>'
        for y, row in enumerate(pattern) for x, ch in enumerate(row) if ch == "#"
    )
    w, h = len(pattern[0]) * size, len(pattern) * size
    return f'<svg class="lp-dots" viewBox="0 0 {w} {h}" fill="currentColor" aria-hidden="true">{cells}</svg>'


ICON_CAPTURE = ("..##..##..", ".########.", "##......##", "#..####..#", "##......##",
                ".########.", "..##..##..")
ICON_CLASSIFY = ("#.#.#.#.#.", ".#.#.#.#.#", "#.#.###.#.", ".#.#####.#", "#.#.###.#.",
                 ".#.#.#.#.#", "#.#.#.#.#.")
ICON_EXPLAIN = ("##########", "..........", "#######...", "..........", "#####.....",
                "..........", "###.......")


# ================================================================ actions ==
def _start_demo() -> None:
    st.session_state.pop("lp_start_error", None)
    try:
        load_monitor().start(MODE_DEMO)
    except NIDSError as exc:
        st.session_state["lp_start_error"] = str(exc)
        return
    st.session_state["lp_go_live"] = True


# =============================================================== sections ==
def _hero(console, predictor, f1) -> None:
    monitor = _safe(load_monitor, None)
    ready = bool(monitor and _safe(monitor.model_ready, False))
    model = predictor.model_name if predictor else None
    classes = len(predictor.classes) if predictor else len(FALLBACK_CLASSES)

    text, visual = st.columns([1.05, 1], vertical_alignment="center", gap="large")
    with text:
        ui.render(
            '<div class="lp-hero" id="top">'
            '<span class="lp-eyebrow"><i class="lp-dot" style="background:var(--color-node-green)"></i>'
            "Explainable network intrusion detection</span>"
            "<h1>Every alert,<br>explained</h1>"
            "<h2>See what is on your network and why the model flagged it.</h2>"
            "<p>NetLens groups traffic into flows, classifies each one with "
            + (f"a trained {_e(model)} model" if model else "a trained model")
            + ", scores its risk and shows the measurements that drove every verdict.</p></div>"
        )
        with st.container(horizontal=True, gap="small", key="lp_ctas"):
            if ui.primary_button("Launch console", key="lp_launch"):
                st.switch_page(console)
            ui.secondary_button("Run demo traffic", key="lp_demo", disabled=not ready,
                                on_click=_start_demo)
        if st.session_state.pop("lp_go_live", False):
            st.session_state["nav"] = "Live Monitor"
            st.switch_page(console)
        if st.session_state.get("lp_start_error"):
            st.error(st.session_state["lp_start_error"])
        if not ready:
            st.caption("Train a model first to run the demo: `python scripts/train_model.py`")

        trust = [
            ("Read-only capture", "M3 12h4l3-8 4 16 3-8h4"),
            (f"{classes} traffic classes", "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z"),
            (f"Macro F1 {f1:.3f} on held-out test" if f1 is not None else "SHAP and LIME explanations",
             "M4 20V10M10 20V4M16 20v-7M22 20H2"),
        ]
        ui.render(
            '<ul class="lp-trust">' + "".join(
                f'<li><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
                f'aria-hidden="true"><path d="{d}"/></svg>{_e(label)}</li>' for label, d in trust
            ) + "</ul>"
        )
    with visual:
        ui.render(f'<div class="lp-visual">{_lattice_svg()}</div>')


def _how_it_works(classes: int) -> None:
    cards = [
        ("1", "Capture", "Watch the wire", ICON_CAPTURE,
         "Packets from a live interface or a recorded replay are grouped into flows and "
         "measured: duration, sizes, flags and rates. Nothing is ever sent."),
        ("2", "Classify", "Score every flow", ICON_CLASSIFY,
         f"A model trained on CIC-IDS2017 labels each flow as benign or one of {classes - 1} "
         "attack classes, and a risk score sets its severity."),
        ("3", "Explain", "Show the why", ICON_EXPLAIN,
         "SHAP and LIME rank the measurements that pushed each verdict, so an analyst can "
         "check an alert instead of taking it on trust."),
    ]
    body = "".join(
        f'<article class="lp-product"><div class="lp-product-top"><span class="lp-step">{n}</span>'
        f"{_dots_icon(icon)}</div><span class=\"lp-cat\">{_e(cat)}</span>"
        f"<h3>{_e(title)}</h3><p>{_e(text)}</p></article>"
        for n, cat, title, icon, text in cards
    )
    ui.render(
        '<section class="lp-light" id="how-it-works">'
        '<div class="lp-section-head"><span class="lp-kicker">How it works</span>'
        "<h2>From packet to explanation in three steps</h2></div>"
        f'<div class="lp-products">{body}</div></section>'
    )


def _coverage(classes: list[str]) -> None:
    families: dict[str, list[str]] = {}
    for label in classes:
        family = theme.attack_family(label)
        if family != "Benign":
            families.setdefault(family, []).append(label)
    cards = []
    for family, labels in families.items():
        colour, ring = theme.ATTACK_FAMILIES.get(family, (theme.STEEL, True))
        style = f"border:2px solid {colour};background:transparent" if ring else f"background:{colour}"
        cells = "".join(f'<div class="lp-cell">{_e(label)}</div>' for label in labels)
        if len(labels) % 2:
            cells += '<div class="lp-cell lp-cell--empty"></div>'
        cards.append(
            f'<article class="lp-family"><header><i class="lp-dot" style="{style}"></i>'
            f'<span>{_e(family)}</span><b>{len(labels)}</b></header>'
            f'<div class="lp-cells">{cells}</div></article>'
        )
    ui.render(
        '<section class="lp-section" id="coverage">'
        '<div class="lp-section-head"><span class="lp-kicker">Coverage</span>'
        "<h2>What the model recognises</h2>"
        f"<p>{sum(len(v) for v in families.values())} attack classes in {len(families)} families, "
        "plus benign traffic.</p></div>"
        f'<div class="lp-families">{"".join(cards)}</div></section>'
    )


def _numbers(console, stats: dict) -> None:
    ui.render(
        '<section class="lp-section" id="numbers"><div class="lp-section-head">'
        '<span class="lp-kicker">Numbers</span><h2>The network so far</h2>'
        "<p>Read live from this installation's database.</p></div>"
        + ui.stats_html([
            {"label": "Packets captured", "value": f"{int(_num(stats.get('total_packets'))):,}"},
            {"label": "Flows classified", "value": f"{int(_num(stats.get('total_flows'))):,}"},
            {"label": "Classified as attack", "value": f"{_num(stats.get('attack_percentage')):.1f}%",
             "accent": True},
            {"label": "Alerts stored", "value": f"{int(_num(stats.get('total_alerts'))):,}"},
        ])
        + "</section>"
    )


def _closing(console, version: str) -> None:
    with st.container(key="lp_close"):
        ui.render(
            '<div class="lp-close-text"><h2>Open the console</h2>'
            "<p>Start monitoring, review alerts and ask the model why.</p></div>"
        )
        if ui.primary_button("Launch console", key="lp_launch_2"):
            st.switch_page(console)
    ui.render(
        '<footer class="lp-footer"><span>NetLens · explainable NIDS</span>'
        "<span>Defensive monitoring only. NetLens never generates, injects or replays "
        "traffic onto a network.</span>"
        f'<span class="mono">v{_e(version)}</span></footer>'
    )


# ================================================================= render ==
def render(console, settings) -> None:
    _inject_css()
    predictor = _safe(load_predictor, None)
    database = _safe(load_database, None)
    stats = _safe(database.statistics, {}) if database else {}
    metrics = _safe(load_training_metrics, None) or {}
    f1 = ((metrics.get("best_model") or {}).get("test") or {}).get("f1_macro")
    classes = list(predictor.classes) if predictor else list(FALLBACK_CLASSES)
    name = str(settings.get("project.short_name", "NetLens"))

    side_nav()
    _hero(console, predictor, f1)
    _how_it_works(len(classes))
    _coverage(classes)
    _numbers(console, stats)
    _closing(console, str(settings.get("project.version", "1.0.0")))


__all__ = ["render"]
