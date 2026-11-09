"""Plotly figures and the shared chart style for the dashboard.

Every chart uses the ``netlens`` Plotly template registered below (matplotlib
gets matching rcParams from :func:`apply_matplotlib_style`):

* backgrounds are Carbon, the card surface charts sit on; gridlines are
  Charcoal hairlines; tick labels are muted DM Mono;
* one accent per chart: the series that matters (attack traffic, the leading
  bar) is Signal Orange and everything else is a neutral step;
* signed SHAP/LIME weights use the cyan <-> orange diverging pair;
* fills are flat, with 2 px surface-coloured gaps between adjacent marks.

Secondary comparisons (class counts, protocols, top talkers) are drawn as
ranked lists by :func:`dashboard.components.ui.ranked_list_html`, so each page
has one main chart and smaller data blocks of a different kind.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from dashboard.styles import theme
from src.features.contract import label_for

BAR_GAP = 2  # px, drawn in the surface colour to separate adjacent fills
BAR_RADIUS = 4  # px - the "small" radius from DESIGN.md
FEW_POINTS = 12  # below this, line charts also draw markers so a point is visible
TEMPLATE_NAME = "netlens"


def _rgba(hex_colour: str, alpha: float) -> str:
    value = hex_colour.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _register_template() -> None:
    layout = {k: (dict(v) if isinstance(v, dict) else v) for k, v in theme.PLOTLY_LAYOUT.items()}
    template = go.layout.Template(layout=layout)
    template.layout.colorscale = dict(
        sequential=theme.sequential_scale(),
        diverging=[[i / 9, c] for i, c in enumerate(theme.DIVERGING)],
    )
    pio.templates[TEMPLATE_NAME] = template
    pio.templates.default = TEMPLATE_NAME


def apply_matplotlib_style() -> None:
    """Matching dark rcParams for any matplotlib figure (e.g. SHAP's own plots)."""
    try:
        import matplotlib as mpl
    except ImportError:  # pragma: no cover - matplotlib is a hard dependency
        return
    mpl.rcParams.update({
        "figure.facecolor": theme.SURFACE,
        "axes.facecolor": theme.SURFACE,
        "savefig.facecolor": theme.SURFACE,
        "axes.edgecolor": theme.GRID,
        "axes.labelcolor": theme.TEXT_MUTED,
        "axes.titlecolor": theme.TEXT_PRIMARY,
        "axes.grid": True,
        "grid.color": theme.GRID,
        "grid.linewidth": 0.8,
        "xtick.color": theme.TEXT_MUTED,
        "ytick.color": theme.TEXT_MUTED,
        "text.color": theme.TEXT_SECONDARY,
        "legend.facecolor": theme.SURFACE,
        "legend.edgecolor": theme.GRID,
        "axes.prop_cycle": mpl.cycler(color=list(theme.SERIES)),
        "font.family": ["Inter", "DejaVu Sans", "sans-serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


_register_template()
apply_matplotlib_style()


def _base_figure(title: str = "", height: int = 320) -> go.Figure:
    figure = go.Figure(layout=dict(template=TEMPLATE_NAME))
    figure.update_layout(height=height, title_text=title, showlegend=False)
    return figure


def empty_figure(message: str = "No data yet", height: int = 260) -> go.Figure:
    """Placeholder shown before any traffic has been classified."""
    figure = _base_figure(height=height)
    figure.add_annotation(
        text=message,
        showarrow=False,
        font=dict(color=theme.TEXT_MUTED, size=13),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


# ===========================================================================
#  Part-to-whole: benign vs attack
# ===========================================================================
def attack_split_bar(benign: int, attack: int, height: int = 176) -> go.Figure:
    """One horizontal stacked bar: the benign/attack split of classified flows."""
    total = int(benign) + int(attack)
    if total <= 0:
        return empty_figure("No classified flows yet", height=height)

    figure = _base_figure("Benign vs attack traffic", height=height)
    for name, value, colour in (
        ("Benign", int(benign), theme.BENIGN),
        ("Attack", int(attack), theme.ATTACK),
    ):
        share = 100.0 * value / total
        figure.add_trace(
            go.Bar(
                x=[value],
                y=["flows"],
                name=name,
                orientation="h",
                marker=dict(
                    color=colour,
                    line=dict(color=theme.SURFACE, width=BAR_GAP),
                ),
                text=[f"{name} · {value:,} ({share:.1f}%)"] if share >= 12 else [""],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(
                    color=theme.ABYSS if colour == theme.ATTACK else theme.WHITE,
                    size=12, family=theme.FONT_MONO,
                ),
                hovertemplate=f"{name}: %{{x:,}} flows ({share:.1f}%)<extra></extra>",
            )
        )
    figure.update_layout(
        barmode="stack",
        showlegend=True,
        margin=dict(l=8, r=8, t=44, b=54),
        bargap=0.35,
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


# ===========================================================================
#  Magnitude: single-hue horizontal bars
# ===========================================================================
def magnitude_bar(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str = "",
    x_title: str = "",
    height: int = 340,
    colour: str | None = None,
    colours: Sequence[str] | None = None,
    value_format: str = "{:,.0f}",
    hover_suffix: str = "",
) -> go.Figure:
    """Horizontal bar chart for magnitude comparison (one hue by default)."""
    if len(labels) == 0:
        return empty_figure("No data yet", height=height)
    order = np.argsort(np.asarray(values, dtype=float))
    labels = [str(labels[i]) for i in order]
    values = [float(values[i]) for i in order]
    if colours is not None:
        marker_colour = [list(colours)[i] for i in order]
    elif colour is not None:
        marker_colour = colour
    else:
        # One accent per chart: the largest bar leads and the rest stay neutral.
        marker_colour = [theme.SEQUENTIAL_MAIN] * (len(values) - 1) + [theme.ACCENT]

    figure = _base_figure(title, height=height)
    figure.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=marker_colour,
                line=dict(color=theme.SURFACE, width=BAR_GAP),
                cornerradius=BAR_RADIUS,
            ),
            text=[value_format.format(v) for v in values],
            textposition="outside",
            textfont=dict(color=theme.TEXT_SECONDARY, size=11, family=theme.FONT_MONO),
            cliponaxis=False,
            hovertemplate="%{y}: %{x:,}" + hover_suffix + "<extra></extra>",
        )
    )
    figure.update_layout(margin=dict(l=4, r=64, t=44, b=6), bargap=0.56)
    figure.update_xaxes(title_text=x_title, showgrid=True, nticks=4)
    # Names, not figures, so they stay in the sans; mono would squeeze a narrow card.
    figure.update_yaxes(
        showgrid=False,
        tickfont=dict(family=theme.FONT_BODY, color=theme.TEXT_SECONDARY, size=12),
    )
    return figure


def class_distribution_bar(
    frame: pd.DataFrame,
    label_column: str,
    value_column: str,
    *,
    title: str = "Attack type distribution",
    height: int = 360,
    max_items: int = theme.MAX_SERIES,
) -> go.Figure:
    """Top-N classes by count; everything past ``max_items`` folds into 'Other'."""
    if frame is None or frame.empty:
        return empty_figure("No attacks detected yet", height=height)
    data = frame[[label_column, value_column]].copy()
    data.columns = ["label", "value"]
    data = data.groupby("label", as_index=False)["value"].sum()
    data = data.sort_values("value", ascending=False)
    if len(data) > max_items:
        head = data.head(max_items)
        tail_total = float(data.tail(len(data) - max_items)["value"].sum())
        data = pd.concat(
            [head, pd.DataFrame([{"label": "Other", "value": tail_total}])],
            ignore_index=True,
        )
    # Sorted largest first, so the first bar is the one that leads.
    colours = [
        theme.OTHER if str(label) == "Other"
        else (theme.ACCENT if index == 0 else theme.SEQUENTIAL_MAIN)
        for index, label in enumerate(data["label"])
    ]
    return magnitude_bar(
        list(data["label"]),
        list(data["value"]),
        title=title,
        x_title="flows",
        height=height,
        colours=colours,
        hover_suffix=" flows",
    )


# ===========================================================================
#  Trend over time
# ===========================================================================
def timeline_chart(
    frame: pd.DataFrame,
    *,
    bucket_column: str = "bucket",
    total_column: str = "total",
    attack_column: str = "attacks",
    title: str = "Alerts and traffic over time",
    height: int = 320,
) -> go.Figure:
    """Two-series line chart: all classified flows and the attack subset."""
    if frame is None or frame.empty:
        return empty_figure("No timeline data yet", height=height)
    data = frame.copy().sort_values(bucket_column)
    # A line needs two points, so a short history must also draw its markers.
    mode = "lines+markers" if len(data) < FEW_POINTS else "lines"
    marker = dict(size=8, line=dict(color=theme.SURFACE, width=2))
    figure = _base_figure(title, height=height)
    figure.add_trace(
        go.Scatter(
            x=data[bucket_column],
            y=data[total_column],
            name="All flows",
            mode=mode,
            marker=dict(color=theme.BENIGN, **marker),
            line=dict(color=theme.BENIGN, width=1.5, shape="spline", smoothing=0.4),
            hovertemplate="All flows: %{y:,}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=data[bucket_column],
            y=data[attack_column],
            name="Attack flows",
            mode=mode,
            marker=dict(color=theme.ATTACK, **marker),
            line=dict(color=theme.ATTACK, width=2, shape="spline", smoothing=0.4),
            fill="tozeroy",
            fillcolor=_rgba(theme.ATTACK, 0.12),
            hovertemplate="Attack flows: %{y:,}<extra></extra>",
        )
    )
    # Selective direct label: the final point of the attack series only.
    if len(data) > 1:
        figure.add_annotation(
            x=data[bucket_column].iloc[-1],
            y=float(data[attack_column].iloc[-1]),
            text=f"{int(data[attack_column].iloc[-1]):,}",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(color=theme.ATTACK, size=11),
        )
    figure.update_layout(
        showlegend=True,
        hovermode="x unified",
        margin=dict(l=4, r=32, t=44, b=62),
        legend=dict(y=-0.24, traceorder="normal"),
    )
    # Flat, sparser ticks: rotated ones would run into the legend on a phone.
    figure.update_xaxes(title_text="", nticks=6, tickangle=0)
    figure.update_yaxes(title_text="flows", rangemode="tozero")
    return figure


def volume_area_chart(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    title: str = "Traffic volume",
    y_title: str = "bytes",
    height: int = 300,
) -> go.Figure:
    """Single-series area chart (sequential hue, no legend needed)."""
    if frame is None or frame.empty:
        return empty_figure("No volume data yet", height=height)
    data = frame.copy().sort_values(x_column)
    figure = _base_figure(title, height=height)
    figure.add_trace(
        go.Scatter(
            x=data[x_column],
            y=data[y_column],
            mode="lines+markers" if len(data) < FEW_POINTS else "lines",
            marker=dict(size=8, color=theme.CLOUD, line=dict(color=theme.SURFACE, width=2)),
            line=dict(color=theme.CLOUD, width=1.5, shape="spline", smoothing=0.4),
            fill="tozeroy",
            fillcolor=_rgba(theme.STEEL, 0.14),
            hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
        )
    )
    figure.update_xaxes(nticks=6, tickangle=0)
    figure.update_yaxes(title_text=y_title, rangemode="tozero")
    return figure


# ===========================================================================
#  Signed values: SHAP / LIME contributions (diverging)
# ===========================================================================
def contribution_chart(
    contributions: Sequence[Mapping[str, Any]],
    *,
    title: str = "Feature contributions",
    height: int | None = None,
    top_k: int = 10,
) -> go.Figure:
    """Diverging horizontal bar chart of signed feature attributions."""
    rows = list(contributions or [])[: int(top_k)]
    if not rows:
        return empty_figure("No attribution available for this prediction", height=260)
    rows = sorted(rows, key=lambda r: float(r.get("contribution", 0.0)))
    labels = [str(r.get("label") or label_for(str(r.get("feature", "")))) for r in rows]
    values = [float(r.get("contribution", 0.0)) for r in rows]
    colours = [
        theme.DIVERGING_POSITIVE if value > 0 else theme.DIVERGING_NEGATIVE
        for value in values
    ]
    height = height or max(240, 34 * len(rows) + 90)

    figure = _base_figure(title, height=height)
    figure.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=colours,
                line=dict(color=theme.SURFACE, width=BAR_GAP),
                cornerradius=BAR_RADIUS,
            ),
            text=[f"{v:+.4f}" for v in values],
            textposition="outside",
            textfont=dict(color=theme.TEXT_SECONDARY, size=11, family=theme.FONT_MONO),
            cliponaxis=False,
            hovertemplate="%{y}<br>weight %{x:+.4f}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_width=1, line_color=theme.DIVERGING_MID)
    figure.update_layout(margin=dict(l=4, r=76, t=44, b=28), bargap=0.36)
    # Room for outside labels: a negative bar's label would otherwise land on the
    # feature names to its left.
    low, high = min(values + [0.0]), max(values + [0.0])
    span = (high - low) or 1.0
    figure.update_xaxes(
        title_text="← reduces likelihood · increases likelihood →",
        range=[low - (0.24 * span if low < 0 else 0.0), high + (0.04 * span if high > 0 else 0.0)],
    )
    figure.update_yaxes(
        showgrid=False,
        tickfont=dict(family=theme.FONT_BODY, color=theme.TEXT_SECONDARY, size=12),
    )
    return figure


# ===========================================================================
#  Matrix: confusion heatmap (sequential, one hue)
# ===========================================================================
def confusion_heatmap(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    title: str = "Confusion matrix (test split)",
    normalise: bool = True,
    height: int = 520,
) -> go.Figure:
    """Sequential single-hue heatmap with in-cell counts."""
    array = np.asarray(matrix, dtype=float)
    if array.size == 0:
        return empty_figure("No confusion matrix available", height=height)
    labels = [str(label) for label in labels]

    if normalise:
        totals = array.sum(axis=1, keepdims=True)
        display = np.divide(array, np.clip(totals, 1, None))
        colour_title = "row share"
        hover = "True %{y}<br>Predicted %{x}<br>%{customdata:,.0f} flows (%{z:.1%})<extra></extra>"
    else:
        display = array
        colour_title = "flows"
        hover = "True %{y}<br>Predicted %{x}<br>%{z:,.0f} flows<extra></extra>"

    figure = _base_figure(title, height=height)
    figure.add_trace(
        go.Heatmap(
            z=display,
            x=labels,
            y=labels,
            customdata=array,
            colorscale=theme.sequential_scale(),
            reversescale=False,
            xgap=BAR_GAP,
            ygap=BAR_GAP,
            hovertemplate=hover,
            colorbar=dict(
                title=dict(text=colour_title, font=dict(color=theme.TEXT_MUTED, size=11)),
                tickfont=dict(color=theme.TEXT_MUTED, size=10, family=theme.FONT_MONO),
                outlinewidth=0,
                thickness=12,
            ),
        )
    )
    # In-cell counts: the text is the accessible twin of the colour encoding.
    threshold = float(np.nanmax(display)) * 0.55 if np.nanmax(display) else 0.0
    for row_index, row_label in enumerate(labels):
        for col_index, col_label in enumerate(labels):
            count = array[row_index, col_index]
            if count == 0:
                continue
            figure.add_annotation(
                x=col_label,
                y=row_label,
                text=f"{int(count):,}",
                showarrow=False,
                font=dict(
                    # The scale brightens with value, so bright cells need dark text.
                    color=theme.ABYSS if display[row_index, col_index] >= threshold else theme.TEXT_PRIMARY,
                    size=10, family=theme.FONT_MONO,
                ),
            )
    figure.update_layout(margin=dict(l=8, r=8, t=44, b=8))
    figure.update_xaxes(title_text="Predicted class", showgrid=False, tickangle=-35)
    figure.update_yaxes(title_text="True class", showgrid=False, autorange="reversed")
    return figure


# ===========================================================================
#  Small helpers used by several pages
# ===========================================================================
def probability_bar(
    probabilities: Mapping[str, float],
    *,
    title: str = "Class probabilities",
    top_k: int = 8,
    height: int = 300,
) -> go.Figure:
    """Model output distribution for one prediction (single hue)."""
    if not probabilities:
        return empty_figure("No probabilities available", height=height)
    items = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[: int(top_k)]
    return magnitude_bar(
        [k for k, _ in items],
        [100.0 * float(v) for _, v in items],
        title=title,
        x_title="probability (%)",
        height=height,
        value_format="{:,.1f}%",
        hover_suffix=" %",
    )


def risk_band_bar(counts: Mapping[str, int], *, height: int = 260) -> go.Figure:
    """Ordered LOW -> CRITICAL bar chart. One hue: the axis names the bands."""
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    values = [int(counts.get(band, 0)) for band in order]
    if sum(values) == 0:
        return empty_figure("No alerts yet", height=height)
    figure = _base_figure("Alerts by severity band", height=height)
    figure.add_trace(
        go.Bar(
            x=list(order),
            y=values,
            marker=dict(
                color=[theme.risk_color(band) for band in order],
                line=dict(color=theme.SURFACE, width=BAR_GAP),
                cornerradius=BAR_RADIUS,
            ),
            text=[f"{v:,}" for v in values],
            textposition="outside",
            textfont=dict(color=theme.TEXT_SECONDARY, size=11, family=theme.FONT_MONO),
            cliponaxis=False,
            hovertemplate="%{x}: %{y:,} alerts<extra></extra>",
        )
    )
    figure.update_layout(margin=dict(l=4, r=8, t=44, b=28), bargap=0.55)
    figure.update_yaxes(title_text="alerts", rangemode="tozero")
    return figure


def importance_bar(
    items: Sequence[Mapping[str, Any]],
    *,
    title: str = "Top contributing features (mean |SHAP|)",
    height: int = 400,
    top_k: int = 12,
) -> go.Figure:
    """Global feature ranking - magnitude, so one hue."""
    rows = list(items or [])[: int(top_k)]
    if not rows:
        return empty_figure("Global importance not computed yet", height=height)
    return magnitude_bar(
        [str(r.get("label") or r.get("feature", "")) for r in rows],
        [float(r.get("importance", 0.0)) for r in rows],
        title=title,
        x_title="mean |SHAP value|",
        height=height,
        value_format="{:,.4f}",
    )


__all__ = [
    "TEMPLATE_NAME",
    "apply_matplotlib_style",
    "empty_figure",
    "attack_split_bar",
    "magnitude_bar",
    "class_distribution_bar",
    "timeline_chart",
    "volume_area_chart",
    "contribution_chart",
    "confusion_heatmap",
    "probability_bar",
    "risk_band_bar",
    "importance_bar",
]
