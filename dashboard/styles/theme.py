"""Chart and UI palette for the dashboard.

The categorical, sequential, diverging and status values below are the
validated dark-mode palette: the eight-slot categorical order passes the
lightness-band, chroma, CVD-separation, normal-vision and contrast checks
against the chart surface.  Red/green was rejected for benign/attack precisely
because it fails CVD separation (deutan Delta E 4.1), so benign is a neutral grey
and only attacks carry colour.

Rules applied throughout the dashboard:

* surfaces, chrome and ordinary data stay in one graphite ramp;
* one accent per chart: the thing that matters is coloured and the rest is neutral;
* categorical hues are assigned in fixed order and never cycled - past seven
  classes the tail folds into a neutral "Other";
* the diverging blue<->red pair is reserved for signed values (SHAP/LIME
  contributions), with a neutral grey midpoint;
* status colours are reserved for risk bands and always ship with their text
  label, never colour alone.
"""

from __future__ import annotations

from typing import Sequence

# --------------------------------------------------------------- surfaces ---
SURFACE = "#19191b"
SURFACE_ALT = "#232325"
GRID = "#2a2a2d"
TEXT_PRIMARY = "#ededed"
TEXT_SECONDARY = "#a1a1a6"
TEXT_MUTED = "#85858b"
ACCENT = "#4f93e4"
FONT_MONO = "'IBM Plex Mono', Consolas, ui-monospace, monospace"

# ------------------------------------------------------------- categorical ---
SERIES: tuple[str, ...] = (
    "#3987e5",  # 1 blue
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#008300",  # 6 green
    "#9085e9",  # 7 violet
    "#e66767",  # 8 red
)
#: Neutral used for the folded tail. Never a generated 9th hue.
OTHER = "#4e4e53"
MAX_SERIES = 7

# --------------------------------------------------------- semantic pairing ---
BENIGN = "#6e6e75"
ATTACK = "#e5484d"

# --------------------------------------------------------------- sequential ---
SEQUENTIAL: tuple[str, ...] = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)
#: Ordinary bars and lines are neutral; ``ACCENT`` marks the one that matters.
SEQUENTIAL_MAIN = "#6e6e75"

# ---------------------------------------------------------------- diverging ---
DIVERGING_NEGATIVE = "#4f93e4"
DIVERGING_MID = "#4e4e53"
DIVERGING_POSITIVE = "#e5484d"

# ------------------------------------------------------------------- status ---
STATUS = {
    "LOW": "#4cc38a",
    "MEDIUM": "#e2a336",
    "HIGH": "#e8804f",
    "CRITICAL": "#e5484d",
}
STATUS_ICON = {"LOW": "●", "MEDIUM": "▲", "HIGH": "▲▲", "CRITICAL": "■"}


def risk_color(level: str) -> str:
    """Status colour for a risk band (always shown with its text label)."""
    return STATUS.get(str(level).upper(), TEXT_MUTED)


def sequential_scale() -> list[list]:
    """Plotly colour scale for the dark surface: low values recede, high ones glow."""
    ramp = list(reversed(SEQUENTIAL))
    steps = len(ramp) - 1
    return [[index / steps, colour] for index, colour in enumerate(ramp)]


def categorical(n: int) -> list[str]:
    """First ``n`` categorical slots in fixed order, folding the tail to grey."""
    n = max(0, int(n))
    if n <= len(SERIES):
        return list(SERIES[:n])
    return list(SERIES) + [OTHER] * (n - len(SERIES))


def fold_tail(labels: Sequence[str], max_items: int = MAX_SERIES) -> list[str]:
    """Colours for ``labels`` where anything past ``max_items`` becomes grey."""
    colours: list[str] = []
    for index, label in enumerate(labels):
        if str(label).lower() == "other" or index >= max_items:
            colours.append(OTHER)
        else:
            colours.append(SERIES[index % len(SERIES)])
    return colours


#: Shared Plotly layout: recessive solid hairline grid, no dashes, generous padding.
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="'IBM Plex Sans', 'Segoe UI', system-ui, -apple-system, Roboto, sans-serif",
        size=13,
        color=TEXT_SECONDARY,
    ),
    title=dict(
        font=dict(size=14, color=TEXT_PRIMARY, weight=500),
        # Measured from the card, so every title lines up with its card's edge.
        x=0.004, xanchor="left", xref="container",
    ),
    margin=dict(l=4, r=12, t=44, b=6),
    hoverlabel=dict(
        bgcolor=SURFACE_ALT, bordercolor=GRID, font=dict(color=TEXT_PRIMARY, size=12)
    ),
    # The legend sits *below* the plot so it can never collide with the title.
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.16,
        xanchor="left",
        x=0,
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_SECONDARY, size=12),
    ),
    xaxis=dict(
        gridcolor=GRID, griddash="solid", zeroline=False,
        linecolor=GRID, tickfont=dict(color=TEXT_MUTED, size=11, family=FONT_MONO),
        title=dict(font=dict(color=TEXT_MUTED, size=12)),
    ),
    yaxis=dict(
        gridcolor=GRID, griddash="solid", zeroline=False,
        linecolor=GRID, tickfont=dict(color=TEXT_MUTED, size=11, family=FONT_MONO),
        title=dict(font=dict(color=TEXT_MUTED, size=12)),
    ),
)

__all__ = [
    "SURFACE", "SURFACE_ALT", "GRID", "TEXT_PRIMARY", "TEXT_SECONDARY", "TEXT_MUTED",
    "ACCENT", "FONT_MONO",
    "SERIES", "OTHER", "MAX_SERIES", "BENIGN", "ATTACK",
    "SEQUENTIAL", "SEQUENTIAL_MAIN", "sequential_scale",
    "DIVERGING_NEGATIVE", "DIVERGING_MID", "DIVERGING_POSITIVE",
    "STATUS", "STATUS_ICON", "risk_color", "categorical", "fold_tail",
    "PLOTLY_LAYOUT",
]
