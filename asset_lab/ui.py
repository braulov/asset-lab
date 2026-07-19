from __future__ import annotations

from numbers import Real
from typing import Any, Callable

import streamlit as st


ACCENT = "#ff5c68"
ACCENT_SECONDARY = "#ff9f5a"


def apply_asset_lab_style() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --asset-accent: {ACCENT};
            --asset-accent-secondary: {ACCENT_SECONDARY};
            --asset-panel: rgba(23, 28, 39, 0.72);
            --asset-border: rgba(255, 255, 255, 0.085);
            --asset-muted: rgba(235, 239, 247, 0.62);
        }}

        .block-container {{
            max-width: 1240px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }}

        [data-testid="stSidebar"] {{
            border-right: 1px solid var(--asset-border);
        }}

        [data-testid="stMetric"] {{
            min-height: 92px;
            padding: 0.85rem 1rem;
            border: 1px solid var(--asset-border);
            border-radius: 14px;
            background: linear-gradient(145deg, rgba(24, 30, 42, 0.88), rgba(18, 22, 31, 0.66));
        }}

        [data-testid="stMetricLabel"] {{
            color: var(--asset-muted);
        }}

        [data-testid="stExpander"] {{
            border: 1px solid var(--asset-border);
            border-radius: 14px;
            overflow: hidden;
            background: rgba(17, 21, 30, 0.50);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.45rem;
            border-bottom: 1px solid var(--asset-border);
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 2.85rem;
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }}

        .asset-callout {{
            padding: 1rem 1.1rem;
            border: 1px solid var(--asset-border);
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(255, 92, 104, 0.08), rgba(255, 159, 90, 0.035));
            margin: 0.55rem 0 1rem 0;
        }}

        .asset-callout strong {{
            color: rgba(255, 255, 255, 0.96);
        }}

        .asset-kicker {{
            color: var(--asset-muted);
            font-size: 0.79rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }}

        .asset-range-wrap {{
            margin-top: -0.55rem;
            margin-bottom: 0.45rem;
        }}

        .asset-range-track {{
            width: 100%;
            height: 4px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.10);
        }}

        .asset-range-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--asset-accent), var(--asset-accent-secondary));
        }}

        .asset-range-labels {{
            display: flex;
            justify-content: space-between;
            margin-top: 0.22rem;
            color: rgba(235, 239, 247, 0.46);
            font-size: 0.66rem;
            line-height: 1;
        }}

        .asset-model-note {{
            padding: 0.9rem 1rem;
            border-left: 3px solid var(--asset-accent);
            border-radius: 0 10px 10px 0;
            background: rgba(255, 255, 255, 0.035);
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid var(--asset-border);
            border-radius: 12px;
            overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_bound(value: Real, format_string: str | None) -> str:
    if format_string is None:
        return f"{value:g}"
    try:
        return format_string % value
    except (TypeError, ValueError):
        return f"{value:g}"


def bounded_number_input(
    label: str,
    *,
    min_value: int | float,
    max_value: int | float,
    value: int | float,
    step: int | float | None = None,
    key: str | None = None,
    format: str | None = None,
    help: str | None = None,
    on_change: Callable[..., Any] | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    disabled: bool = False,
    label_visibility: str = "visible",
) -> int | float:
    result = st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        key=key,
        format=format,
        help=help,
        on_change=on_change,
        args=args,
        kwargs=kwargs,
        disabled=disabled,
        label_visibility=label_visibility,
    )
    span = float(max_value) - float(min_value)
    fraction = 0.0 if span <= 0 else (float(result) - float(min_value)) / span
    percentage = min(100.0, max(0.0, fraction * 100.0))
    left = _format_bound(min_value, format)
    right = _format_bound(max_value, format)
    st.markdown(
        f"""
        <div aria-hidden="true" style="margin-top:-0.55rem;margin-bottom:0.45rem">
            <div style="width:100%;height:4px;border-radius:999px;overflow:hidden;background:rgba(255,255,255,0.10)">
                <div style="width:{percentage:.2f}%;height:100%;border-radius:999px;background:linear-gradient(90deg,{ACCENT},{ACCENT_SECONDARY})"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:0.22rem;color:rgba(235,239,247,0.46);font-size:0.66rem;line-height:1">
                <span>{left}</span><span>{right}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return result
