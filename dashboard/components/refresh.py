"""Auto-refresh helpers.

Streamlit's fragment API (``st.fragment(run_every=...)``) re-runs only the
decorated block on a timer, which is exactly what a live table needs.  It is
not available on every Streamlit release, so this module degrades to a manual
refresh button rather than failing.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st


def fragment(run_every: float | None = None):
    """Return ``st.fragment(run_every=...)`` or an identity decorator."""
    factory = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
    if factory is None:
        def _identity(func: Callable) -> Callable:
            return func
        return _identity
    try:
        return factory(run_every=run_every) if run_every else factory()
    except TypeError:  # pragma: no cover - older signature
        return factory()


def auto_refresh_supported() -> bool:
    return hasattr(st, "fragment") or hasattr(st, "experimental_fragment")


def refresh_controls(key: str, default_interval: int = 5) -> tuple[bool, int]:
    """Render the 'auto refresh' toggle + interval selector. Returns both."""
    columns = st.columns([1, 1, 3])
    with columns[0]:
        enabled = st.toggle(
            "Auto-refresh",
            value=st.session_state.get(f"{key}_auto", True),
            key=f"{key}_auto",
            help=(
                "Re-renders the live view on a timer."
                if auto_refresh_supported()
                else "This Streamlit version has no fragment timer; use Refresh now."
            ),
        )
    with columns[1]:
        interval = st.number_input(
            "Every (s)",
            min_value=2,
            max_value=120,
            value=int(st.session_state.get(f"{key}_interval", default_interval)),
            step=1,
            key=f"{key}_interval",
        )
    with columns[2]:
        st.write("")
        if st.button("Refresh now", key=f"{key}_manual"):
            st.rerun()
    return bool(enabled), int(interval)


__all__ = ["fragment", "auto_refresh_supported", "refresh_controls"]
