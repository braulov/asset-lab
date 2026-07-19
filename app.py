from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Asset Lab v6", page_icon="📈", layout="wide")

st.title("Asset Lab v6")
st.caption("MOEX volatility, moments, mobility and shock-relaxation research laboratory")

st.markdown(
    """
Version 6 separates two complementary workflows.

### Daily Lab
The validated v5 pipeline remains intact:

- several daily OHLC variance estimators;
- past-only HAR forecasts and volatility innovations;
- asymmetric price-trend regressions;
- HAR, matched-control and local-projection counterfactuals;
- guarded relaxation-law fitting and event-level cross-validation.

### Hourly Moments
The new page implements the findings from the 25-stock hourly study:

- past-only hour-of-day standardisation;
- separate **M2**, **M3**, skewness and downside variance;
- overnight and within-session asymmetry;
- constant, proportional and affine mobility models;
- the finite-horizon M3 implication of `M(F)=a+bF`;
- abrupt/preheated and, with a multi-asset ZIP, external/internal process proxies;
- matched-control M2/M3 trajectories;
- exponential, constrained-mobility, stress-fed and shifted-power relaxation models;
- an optional market-level M3 and synchrony panel.

Use the page selector in the sidebar.
"""
)

left, middle, right = st.columns(3)
left.metric("Daily workflow", "3 counterfactuals")
middle.metric("Hourly moments", "M2 + M3")
right.metric("Mechanistic models", "4 relaxation laws")

st.info(
    "The spectral graph branch remains an advanced diagnostic rather than the main predictor: "
    "broad daily and hourly tests did not show robust incremental forecasting value."
)
