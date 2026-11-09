"""Floating vertical side navigation - landing page and console.

A fixed stack of square icon tiles on the left edge (a bottom dock under
768 px), topped by the NetLens logo tile. Styles live in
``dashboard/styles/side_nav.css`` and read only the project's design tokens.

* ``side_nav()``     - landing page. Tiles are anchors; clicking smooth-scrolls
  to the section and an IntersectionObserver lights the tile in view. Rendered
  with ``st.html(..., unsafe_allow_javascript=True)`` so it sits in the page DOM
  (not an iframe). Markup and script are static strings from this module.
* ``console_rail()`` - console. Tiles are native ``st.button``s (keyboard and
  screen-reader behaviour come with them) inside one fixed container; the
  active page is ``st.session_state["nav"]``.
"""

from __future__ import annotations

import html
from pathlib import Path
from urllib.parse import quote
from typing import Callable, Sequence

import streamlit as st

STYLE_PATH = Path(__file__).resolve().parents[1] / "styles" / "side_nav.css"
LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "netlens-logo.svg"

#: The NetLens mark (also saved as dashboard/assets/netlens-logo.svg).
_LOGO_SOURCE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="9" fill="#ff6314"/>'
    '<circle cx="14" cy="14" r="7.5" fill="none" stroke="#04070a" stroke-width="2.6"/>'
    '<path d="M19.6 19.6 25 25" stroke="#04070a" stroke-width="3.2" stroke-linecap="round"/>'
    '<path d="M14 10.6 17.4 14 14 17.4 10.6 14Z" fill="#04070a"/></svg>'
)
#: st.html strips inline <svg>, so the mark and the icons are drawn by CSS
#: (background / mask images) and the markup only carries empty spans.
LOGO_SVG = '<span class="sn-logo" aria-hidden="true"></span>'

#: Lucide icons (ISC licence), 24 px grid. Shapes with class "duo" get a soft
#: fill for the duotone look; the rest are strokes. Used as CSS masks.
ICONS: dict[str, str] = {
    "home": (
        '<path class="duo" d="M3 10a2 2 0 0 1 .709-1.528l7-5.999a2 2 0 0 1 2.582 0l7 5.999'
        'A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
        '<path d="M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8"/>'
    ),
    "workflow": (
        '<rect class="duo" width="8" height="8" x="3" y="3" rx="2"/>'
        '<path d="M7 11v4a2 2 0 0 0 2 2h4"/><rect class="duo" width="8" height="8" x="13" y="13" rx="2"/>'
    ),
    "shield": (
        '<path class="duo" d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6'
        'a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>'
        '<path d="m9 12 2 2 4-4"/>'
    ),
    "chart": (
        '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>'
    ),
    "arrow": '<path d="M7 17 17 7"/><path d="M7 7h10v10"/>',
}

#: (section id, label, icon) for the landing page, in page order.
LANDING_ITEMS: tuple[tuple[str, str, str], ...] = (
    ("top", "Home", "home"),
    ("how-it-works", "How it works", "workflow"),
    ("coverage", "Coverage", "shield"),
    ("numbers", "Numbers", "chart"),
)

_SCRIPT = """
<script>
(() => {
  const nav = document.querySelector('.sn[data-sn]');
  if (!nav) return;
  // Reruns re-insert the nav: drop the previous observer first.
  if (window.__snObserver) window.__snObserver.disconnect();
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const tiles = [...nav.querySelectorAll('a[data-target]')];
  const setActive = (id) => tiles.forEach((t) => {
    if (t.dataset.target === id) t.setAttribute('aria-current', 'true');
    else t.removeAttribute('aria-current');
  });
  nav.querySelectorAll('a[data-scroll]').forEach((t) => t.addEventListener('click', (event) => {
    const target = document.getElementById(t.dataset.scroll);
    if (!target) return;
    event.preventDefault();
    target.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
    if (t.dataset.target) setActive(t.dataset.target);
  }));
  // A section counts as current once it crosses the upper-middle of the viewport.
  const seen = new Map();
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((e) => seen.set(e.target.id, e.isIntersecting));
    const current = tiles.map((t) => t.dataset.target).filter((id) => seen.get(id));
    if (current.length) setActive(current[current.length - 1]);
  }, { rootMargin: '-35% 0px -55% 0px' });
  const watch = () => tiles.forEach((t) => {
    const el = document.getElementById(t.dataset.target);
    if (el) observer.observe(el);
  });
  watch();
  // Sections render after this element on first load; observe them as they arrive.
  setTimeout(watch, 400);
  setTimeout(watch, 1500);
  window.__snObserver = observer;
})();
</script>
"""


def _data_uri(svg: str) -> str:
    return "data:image/svg+xml," + quote(svg, safe="")


def _icon_svg(key: str) -> str:
    body = ICONS[key].replace('class="duo"', 'fill="#000" fill-opacity=".22"')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#000" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )


def _icon_css() -> str:
    """One mask rule per icon plus the logo background, built from the sources above."""
    rules = [f'.sn-ico--{key} {{ --sn-mask: url("{_data_uri(_icon_svg(key))}"); }}' for key in ICONS]
    rules.append(f'.sn-logo {{ background-image: url("{_data_uri(_LOGO_SOURCE)}"); }}')
    return "\n".join(rules)


def _css() -> str:
    try:
        sheet = STYLE_PATH.read_text(encoding="utf-8")
    except OSError:
        sheet = ""
    return sheet + "\n" + _icon_css()


def _icon(key: str) -> str:
    return f'<span class="sn-ico sn-ico--{key}" aria-hidden="true"></span>'


def side_nav(
    items: Sequence[tuple[str, str, str]] = LANDING_ITEMS,
    *,
    active: str | None = None,
    cta: tuple[str, str] | None = ("console", "Open console"),
) -> None:
    """Landing-page rail. ``items``: (section id, label, icon key);
    ``cta``: (href, label) for the accent tile at the foot, or None."""
    active = active or (items[0][0] if items else None)
    logo = (
        '<li class="sn-brand"><a class="sn-tile sn-tile--logo" href="#top" target="_self" '
        'data-scroll="top" aria-label="NetLens, back to top">'
        f'{LOGO_SVG}<span class="sn-label" aria-hidden="true">NetLens</span></a></li>'
    )
    tiles = []
    for target, label, icon in items:
        current = ' aria-current="true"' if target == active else ""
        t = html.escape(target)
        tiles.append(
            f'<li><a class="sn-tile" href="#{t}" target="_self" data-target="{t}" data-scroll="{t}" '
            f'aria-label="{html.escape(label)}"{current}>{_icon(icon)}'
            f'<span class="sn-label" aria-hidden="true">{html.escape(label)}</span></a></li>'
        )
    foot = ""
    if cta:
        href, label = cta
        foot = (
            f'<li class="sn-foot"><a class="sn-tile sn-tile--cta" href="{html.escape(href)}" target="_self" '
            f'aria-label="{html.escape(label)}">{_icon("arrow")}'
            f'<span class="sn-label" aria-hidden="true">{html.escape(label)}</span></a></li>'
        )
    st.html(
        f"<style>{_css()}</style>"
        f'<nav class="sn" data-sn aria-label="Primary"><ul>{logo}{"".join(tiles)}{foot}</ul></nav>'
        + _SCRIPT,
        unsafe_allow_javascript=True,
    )


def console_rail(
    pages: Sequence[tuple[str, str, str]],
    *,
    active: str,
    on_select: Callable[[str], None],
    group_starts: Sequence[str] = (),
    home_href: str = "./",
) -> None:
    """Console rail. ``pages``: (page id, label, Material icon name).
    ``group_starts``: page ids that open a new group (a hairline goes above)."""
    st.html(f"<style>{_css()}</style>")
    with st.container(key="rail", gap=None):
        st.html(
            f'<a class="sn-tile sn-tile--logo" href="{html.escape(home_href)}" target="_self" '
            f'aria-label="NetLens, back to site">{LOGO_SVG}'
            '<span class="sn-label" aria-hidden="true">Back to site</span></a>'
        )
        for page_id, label, icon in pages:
            if page_id in group_starts:
                st.html('<hr class="sn-sep">')
            st.button(
                label,
                icon=f":material/{icon}:",
                type="primary" if page_id == active else "secondary",
                key=f"rail_{page_id.lower().replace(' ', '_')}",
                on_click=on_select,
                args=(page_id,),
            )


__all__ = ["side_nav", "console_rail", "LANDING_ITEMS", "ICONS", "LOGO_SVG", "LOGO_PATH"]
