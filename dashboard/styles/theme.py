"""Design tokens for the NetLens dashboard.

The one place where colours, type, radii, spacing and motion are defined.
``inject_css()`` turns them into CSS custom properties and loads ``theme.css``,
which only ever reads ``var(--...)``.

Direction
---------
* Product   - watches network traffic, flags intrusions with a trained model and
              explains each alert, for the analyst on watch.
* Character - watchful, exact, unhurried.
* World     - a network operations room at night: status lights on dark racks,
              instrument read-outs.
* Moments   - the live network map on the command center, and the main chart of
              each page (attack timeline, SHAP contributions, confusion matrix).
* Signature - the four-light severity meter (``●●●○ HIGH``), like the status
              lights on a rack unit. Used wherever severity appears, nowhere else.

Rules
-----
* Surfaces stay in the neutral Axelar stack (abyss -> carbon -> obsidian ->
  charcoal hairlines). Depth is tonal, never a shadow or a glow.
* Signal Orange is the only chromatic accent in the chrome: the primary action,
  the selected navigation item, focus, and the one figure or series that matters.
* Node lights (green, cyan, violet, magenta) appear only as small status lights
  and always travel with a text label.
* Risk/severity and ML confidence are different quantities and never share an
  encoding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

# =================================================================== palette ==
SIGNAL_ORANGE = "#ff6314"
NODE_GREEN = "#33ffac"
NODE_VIOLET = "#5b76ff"
NODE_MAGENTA = "#ff2ad4"
NODE_CYAN = "#16c4ff"

ABYSS = "#04070a"      # page and sidebar
VOID = "#000000"       # never a background; SVG fills only
CARBON = "#090c0f"     # panels and cards
OBSIDIAN = "#0f1214"   # raised: inputs, active navigation row, code
CHARCOAL = "#1a1d1f"   # hairlines
STEEL = "#676f7a"      # neutral data marks, placeholders, disabled
CLOUD = "#c6ced7"      # secondary text
MIST = "#e1e6eb"
SNOW = "#f0f5fa"       # primary text
WHITE = "#ffffff"

COLORS: dict[str, str] = {
    "signal-orange": SIGNAL_ORANGE, "node-green": NODE_GREEN, "node-violet": NODE_VIOLET,
    "node-magenta": NODE_MAGENTA, "node-cyan": NODE_CYAN, "abyss": ABYSS, "void": VOID,
    "obsidian": OBSIDIAN, "charcoal": CHARCOAL, "carbon": CARBON, "steel": STEEL,
    "cloud": CLOUD, "mist": MIST, "snow": SNOW, "white": WHITE,
}

# ================================================================ text roles ==
TEXT_1 = SNOW          # primary: headings, figures, body
TEXT_2 = CLOUD         # secondary: descriptions, labels (12:1 on carbon)
TEXT_3 = "#8a929d"     # muted: meta and captions. Steel lifted one step in the
                       # same cool grey so 12 px text clears 4.5:1 (6.2:1 on carbon)
HAIRLINE = CHARCOAL
HAIRLINE_STRONG = "#2c3136"   # input and button outlines
PIP_OFF = "#262a2f"           # an unlit severity light
ORANGE_HOVER = "#ff7a36"      # the accent one step lighter, for hover only

# ================================================================ typography ==
# One interface family plus one partner: Inter for the interface, DM Mono for
# figures, addresses, timestamps and meta - the instrument read-out voice.
FONT_UI = "'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
FONT_MONO = "'DM Mono', 'JetBrains Mono', ui-monospace, Consolas, monospace"
FONT_BODY = FONT_UI
#: DESIGN.md: Clashgrotesk for headings, navigation and buttons. It is not on
#: Google Fonts, so Space Grotesk (the documented substitute) is loaded behind it.
FONT_DISPLAY = "'Clashgrotesk', 'Space Grotesk', ui-sans-serif, system-ui, 'Segoe UI', sans-serif"
GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600"
    "&family=Space+Grotesk:wght@500;600"
    "&family=DM+Mono:wght@400;500&display=swap"
)

#: role -> (size px, line height px, letter spacing em). Tracking is size
#: specific: zero below 16 px, tightening only for large display text.
TYPE_SCALE: dict[str, tuple[int, int, float]] = {
    "meta": (12, 16, 0.0),
    "caption": (13, 20, 0.0),
    "body": (14, 20, 0.0),
    "heading": (16, 24, -0.005),
    "lead": (18, 28, -0.005),
    "title": (28, 34, -0.02),
    "figure": (28, 32, -0.01),
    "display": (52, 56, -0.03),
}

# ======================================================= spacing and shapes ==
SPACING = (4, 8, 12, 16, 20, 24, 32, 40, 48, 64)   # the 4 px grid
RADIUS_SMALL = 4      # bars, keycaps, small inline elements
RADIUS_CONTROL = 8    # navigation rows, menus
RADIUS_PILL = 24      # DESIGN.md: every button and single-line input is a pill
RADIUS_CARD = 8       # DESIGN.md: cards
RADIUS_BADGE = 100    # pill badges
PAGE_MAX_WIDTH = 1280  # DESIGN.md page max-width
EASE = "cubic-bezier(0.22, 1, 0.36, 1)"
DURATIONS = {"fast": 140, "menu": 180, "enter": 360}

# ============================================ severity: the signature meter ==
STATUS: dict[str, str] = {
    "LOW": NODE_GREEN,
    "MEDIUM": NODE_CYAN,
    "HIGH": NODE_VIOLET,
    "CRITICAL": NODE_MAGENTA,
}
RISK_ORDER = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
#: How many of the four lights each band lights. The count, not the hue,
#: carries the order, so severity reads without colour.
SEVERITY_LEVEL = {band: index + 1 for index, band in enumerate(RISK_ORDER)}
#: Text form of the meter for places that cannot draw (dataframe cells).
STATUS_ICON = {band: "●" * n + "○" * (4 - n) for band, n in SEVERITY_LEVEL.items()}


def risk_color(level: str) -> str:
    """Status light for a severity band (always shown with its label)."""
    return STATUS.get(str(level).upper(), STEEL)


# ============================================ attack classes: category lights ==
#: Attack family -> (light colour, hollow ring?). The class name always follows.
ATTACK_FAMILIES: dict[str, tuple[str, bool]] = {
    "Benign": (STEEL, False),
    "DoS": (NODE_MAGENTA, False),
    "DDoS": (NODE_VIOLET, False),
    "PortScan": (NODE_CYAN, False),
    "Brute Force": (NODE_GREEN, False),
    "Web Attack": (NODE_CYAN, True),
    "Botnet": (NODE_MAGENTA, True),
    "Other": (STEEL, True),
}


def attack_family(label: str) -> str:
    """Map a CIC-IDS2017 class name (``DoS Hulk``, ``SSH-Patator`` ...) to its family."""
    text = str(label).strip().lower()
    if text in ("benign", "normal"):
        return "Benign"
    if text.startswith("ddos"):
        return "DDoS"
    if text.startswith("dos"):
        return "DoS"
    if "portscan" in text or "port scan" in text:
        return "PortScan"
    if text.startswith("web attack") or "xss" in text or "sql" in text:
        return "Web Attack"
    if "patator" in text or "brute" in text:
        return "Brute Force"
    if text.startswith("bot"):
        return "Botnet"
    return "Other"


# ===================================================================== charts ==
SURFACE = CARBON          # charts sit inside cards
SURFACE_ALT = OBSIDIAN    # hover labels
GRID = CHARCOAL
TEXT_PRIMARY = TEXT_1
TEXT_SECONDARY = TEXT_2
TEXT_MUTED = TEXT_3
ACCENT = SIGNAL_ORANGE    # the one series or bar that matters

#: Categorical order when a chart must tell series apart: orange leads.
SERIES: tuple[str, ...] = (SIGNAL_ORANGE, NODE_CYAN, NODE_GREEN, NODE_VIOLET, NODE_MAGENTA, CLOUD)
OTHER = "#3a4047"
MAX_SERIES = len(SERIES)

BENIGN = STEEL            # ordinary traffic stays neutral
ATTACK = SIGNAL_ORANGE    # attack traffic is the series that matters

SEQUENTIAL_MAIN = "#3d434a"   # neutral bars and tracks
SEQUENTIAL: tuple[str, ...] = (
    "#14171a", "#2a1d17", "#40231a", "#57291b", "#6e2f1b",
    "#86361a", "#9f3d19", "#b94518", "#d94f16", SIGNAL_ORANGE,
)
#: Signed SHAP/LIME weights: cyan lowers the predicted class, orange raises it.
DIVERGING_NEGATIVE = NODE_CYAN
DIVERGING_MID = STEEL
DIVERGING_POSITIVE = SIGNAL_ORANGE
DIVERGING: tuple[str, ...] = (
    "#16c4ff", "#1ba3d3", "#1f82a7", "#24617b", "#2f3a44",
    "#4a3127", "#7a3a1f", "#aa431b", "#d94f16", SIGNAL_ORANGE,
)


def sequential_scale() -> list[list]:
    """Plotly colour scale: low values sink into the card, high values glow orange."""
    steps = len(SEQUENTIAL) - 1
    return [[index / steps, colour] for index, colour in enumerate(SEQUENTIAL)]


def categorical(n: int) -> list[str]:
    """First ``n`` categorical slots in fixed order, folding the tail to neutral."""
    n = max(0, int(n))
    if n <= len(SERIES):
        return list(SERIES[:n])
    return list(SERIES) + [OTHER] * (n - len(SERIES))


def fold_tail(labels: Sequence[str], max_items: int = MAX_SERIES) -> list[str]:
    """Colours for ``labels`` where anything past ``max_items`` becomes neutral."""
    return [
        OTHER if str(label).lower() == "other" or index >= max_items
        else SERIES[index % len(SERIES)]
        for index, label in enumerate(labels)
    ]


#: Shared Plotly layout (registered as the ``netlens`` template in charts.py).
PLOTLY_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family=FONT_UI, size=13, color=TEXT_2),
    title=dict(
        font=dict(family=FONT_UI, size=14, color=TEXT_1, weight=500),
        x=0, xanchor="left", xref="container",
    ),
    colorway=list(SERIES),
    margin=dict(l=4, r=12, t=12, b=6),
    hoverlabel=dict(
        bgcolor=SURFACE_ALT, bordercolor=HAIRLINE_STRONG,
        font=dict(color=TEXT_1, size=12, family=FONT_MONO),
    ),
    legend=dict(
        orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0,
        bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_2, size=12, family=FONT_UI),
    ),
    xaxis=dict(
        gridcolor=GRID, griddash="solid", zeroline=False, linecolor=GRID,
        tickfont=dict(color=TEXT_3, size=11, family=FONT_MONO),
        title=dict(font=dict(color=TEXT_3, size=12, family=FONT_UI)),
    ),
    yaxis=dict(
        gridcolor=GRID, griddash="solid", zeroline=False, linecolor=GRID,
        tickfont=dict(color=TEXT_3, size=11, family=FONT_MONO),
        title=dict(font=dict(color=TEXT_3, size=12, family=FONT_UI)),
    ),
)

# ======================================================================= CSS ==
STYLE_PATH = Path(__file__).resolve().parent / "theme.css"
LANDING_STYLE_PATH = Path(__file__).resolve().parent / "landing.css"
#: The project's design tokens (DESIGN.md quick start, Tailwind ``@theme`` form).
DESIGN_TOKENS_PATH = Path(__file__).resolve().parents[2] / "theme.css"


def design_tokens() -> str:
    """The root ``theme.css`` token block, served as plain ``:root`` variables
    (``--color-*``, ``--text-*``, ``--spacing-*``, ``--radius-*`` ...)."""
    try:
        sheet = DESIGN_TOKENS_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""
    return sheet.replace("@theme", ":root", 1) + "\n"


def css_variables() -> str:
    """The token set as a ``:root`` block."""
    lines = [f"  --color-{name}: {value};" for name, value in COLORS.items()]
    lines += [
        f"  --text-1: {TEXT_1};", f"  --text-2: {TEXT_2};", f"  --text-3: {TEXT_3};",
        f"  --hairline: {HAIRLINE};", f"  --hairline-strong: {HAIRLINE_STRONG};",
        f"  --pip-off: {PIP_OFF};", f"  --orange-hover: {ORANGE_HOVER};",
        f"  --font-ui: {FONT_UI};", f"  --font-mono: {FONT_MONO};",
        f"  --font-display: {FONT_DISPLAY};",
        f"  --r-pill: {RADIUS_PILL}px;", f"  --r-badge: {RADIUS_BADGE}px;",
        f"  --r-sm: {RADIUS_SMALL}px;", f"  --r-control: {RADIUS_CONTROL}px;",
        f"  --r-card: {RADIUS_CARD}px;", f"  --page-max: {PAGE_MAX_WIDTH}px;",
        f"  --ease: {EASE};",
    ]
    for role, (size, leading, tracking) in TYPE_SCALE.items():
        lines += [
            f"  --fs-{role}: {size}px;",
            f"  --lh-{role}: {leading}px;",
            f"  --ls-{role}: {tracking}em;",
        ]
    lines += [f"  --s-{step}: {step}px;" for step in SPACING]
    lines += [f"  --dur-{name}: {ms}ms;" for name, ms in DURATIONS.items()]
    lines += [f"  --sev-{band.lower()}: {colour};" for band, colour in STATUS.items()]
    return ":root {\n" + "\n".join(lines) + "\n}\n"


def build_css() -> str:
    """Font import + token variables + the component stylesheet."""
    try:
        sheet = STYLE_PATH.read_text(encoding="utf-8")
    except OSError:
        sheet = ""
    return f"@import url('{GOOGLE_FONTS_URL}');\n{design_tokens()}{css_variables()}{sheet}"


def inject_css(extra: str = "") -> None:
    """Inject the global stylesheet once per rerun, before any page renders."""
    import streamlit as st

    st.markdown(f"<style>{build_css()}{extra}</style>", unsafe_allow_html=True)


def inject_landing_css() -> None:
    """The landing page's own sheet, layered on top of the global one."""
    import streamlit as st

    try:
        sheet = LANDING_STYLE_PATH.read_text(encoding="utf-8")
    except OSError:
        sheet = ""
    st.markdown(f"<style>{sheet}</style>", unsafe_allow_html=True)


__all__ = [
    "COLORS", "SIGNAL_ORANGE", "NODE_GREEN", "NODE_VIOLET", "NODE_MAGENTA", "NODE_CYAN",
    "ABYSS", "VOID", "CARBON", "OBSIDIAN", "CHARCOAL", "STEEL", "CLOUD", "MIST", "SNOW", "WHITE",
    "TEXT_1", "TEXT_2", "TEXT_3", "HAIRLINE", "HAIRLINE_STRONG", "PIP_OFF", "ORANGE_HOVER",
    "FONT_UI", "FONT_MONO", "FONT_BODY", "FONT_DISPLAY", "GOOGLE_FONTS_URL", "TYPE_SCALE",
    "SPACING", "RADIUS_SMALL", "RADIUS_CONTROL", "RADIUS_PILL", "RADIUS_CARD", "RADIUS_BADGE",
    "PAGE_MAX_WIDTH",
    "EASE", "DURATIONS",
    "STATUS", "RISK_ORDER", "SEVERITY_LEVEL", "STATUS_ICON", "risk_color",
    "ATTACK_FAMILIES", "attack_family",
    "SURFACE", "SURFACE_ALT", "GRID", "TEXT_PRIMARY", "TEXT_SECONDARY", "TEXT_MUTED", "ACCENT",
    "SERIES", "OTHER", "MAX_SERIES", "BENIGN", "ATTACK",
    "SEQUENTIAL", "SEQUENTIAL_MAIN", "sequential_scale",
    "DIVERGING", "DIVERGING_NEGATIVE", "DIVERGING_MID", "DIVERGING_POSITIVE",
    "categorical", "fold_tail", "PLOTLY_LAYOUT",
    "design_tokens", "css_variables", "build_css", "inject_css", "inject_landing_css",
]
