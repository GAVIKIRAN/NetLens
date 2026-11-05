"""Streamlit API compatibility helpers.

Streamlit renamed the width argument on ``dataframe``/``plotly_chart`` from
``use_container_width=True`` to ``width="stretch"``.  The project supports a
wide range of Streamlit versions, so the correct spelling is discovered once at
runtime instead of being hard-coded (which would either spam deprecation
warnings on new builds or raise ``TypeError`` on old ones).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

_WIDTH_STYLE: str | None = None


def _detect_width_style() -> str:
    """Return ``"stretch"`` for modern Streamlit, ``"legacy"`` otherwise."""
    global _WIDTH_STYLE
    if _WIDTH_STYLE is not None:
        return _WIDTH_STYLE
    try:
        import inspect

        parameters = inspect.signature(st.dataframe).parameters
        width = parameters.get("width")
        # The annotation can be an alias such as ``Width``, so read the default too.
        modern = width is not None and (
            width.default in ("stretch", "content")
            or "str" in str(width.annotation)
            or "Literal" in str(width.annotation)
        )
        _WIDTH_STYLE = "stretch" if modern else "legacy"
    except Exception:  # pragma: no cover - defensive
        _WIDTH_STYLE = "legacy"
    return _WIDTH_STYLE


def _width_kwargs() -> dict[str, Any]:
    return {"width": "stretch"} if _detect_width_style() == "stretch" else {
        "use_container_width": True
    }


def dataframe(
    frame: pd.DataFrame,
    *,
    height: int | None = None,
    hide_index: bool = True,
    column_config: dict[str, Any] | None = None,
) -> None:
    """``st.dataframe`` that works on old and new Streamlit and skips None height."""
    kwargs: dict[str, Any] = dict(_width_kwargs())
    if height is not None:
        kwargs["height"] = int(height)
    if hide_index:
        kwargs["hide_index"] = True
    if column_config:
        kwargs["column_config"] = column_config
    try:
        st.dataframe(frame, **kwargs)
    except TypeError:  # pragma: no cover - very old Streamlit
        st.dataframe(frame)


def plotly_chart(figure, *, key: str | None = None) -> None:
    """``st.plotly_chart`` with the right width spelling for this Streamlit."""
    kwargs: dict[str, Any] = dict(_width_kwargs())
    kwargs["config"] = {"displayModeBar": False, "displaylogo": False, "responsive": True}
    if key:
        kwargs["key"] = key
    try:
        st.plotly_chart(figure, **kwargs)
    except TypeError:  # pragma: no cover - very old Streamlit
        st.plotly_chart(figure)


__all__ = ["dataframe", "plotly_chart"]
