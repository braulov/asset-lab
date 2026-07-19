from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from asset_lab.analysis.events import (
    aggregate_counterfactual_trajectories,
    detect_volatility_shocks,
    horizon_comparisons,
)
from asset_lab.analysis.features import (
    calendar_gap_report,
    exclude_current_unfinished_daily_candle,
    log_returns,
    rolling_moments,
    validate_candles,
)
from asset_lab.analysis.har import expanding_har_innovations
from asset_lab.analysis.regression import (
    fit_har_trend_regression,
    fit_har_trend_regression_by_period,
    regression_interpretation,
    walk_forward_comparison,
)
from asset_lab.analysis.relaxation import (
    cross_validate_relaxation_models,
    fit_relaxation_models,
)
from asset_lab.analysis.shock_models import (
    aftershock_excess_rate,
    build_har_median_counterfactual_trajectories,
    build_matched_control_trajectories,
    consensus_decision,
    fit_baseline_omori,
    fit_local_projection_responses,
    local_projection_aggregate,
    matching_balance_summary,
    method_early_response_summary,
    select_matched_controls,
)
from asset_lab.analysis.trend import (
    build_trend_variance_frame,
    correlation_summary,
    decile_summary,
)
from asset_lab.analysis.volatility import (
    PROXY_LABELS,
    ohlc_variance_proxies,
    proxy_correlation_matrix,
    proxy_disagreement_days,
    rolling_asymmetry_metrics,
    rolling_volatility_from_variance,
    yang_zhang_volatility,
)
from asset_lab.charts.figures import (
    aftershock_excess_figure,
    asymmetric_coefficients_figure,
    counterfactual_event_figure,
    local_projection_difference_figure,
    matching_balance_figure,
    multi_method_response_figure,
    decile_figure,
    har_shock_score_figure,
    heatmap_figure,
    line_figure,
    price_figure,
    trend_scatter,
)
from asset_lab.data.moex import InstrumentRoute, MoexApiError, MoexClient


st.set_page_config(page_title="Asset Lab v6 · Daily", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(
    secid: str,
    start_date: date,
    end_date: date,
    interval: int,
    preferred_board: str | None,
) -> tuple[InstrumentRoute, pd.DataFrame]:
    client = MoexClient()
    route = client.resolve_route(secid, preferred_board=preferred_board or None)
    candles = client.load_candles(route, start_date, end_date, interval=interval)
    return route, candles


@st.cache_data(show_spinner=False)
def cached_har_diagnostics(
    variance_proxy: pd.Series,
    initial_training: int,
    refit_every: int,
    mad_window: int,
) -> pd.DataFrame:
    return expanding_har_innovations(
        variance_proxy,
        initial_training=initial_training,
        refit_every=refit_every,
        mad_window=mad_window,
        robust=True,
    )


@st.cache_data(show_spinner=False)
def cached_har_median_trajectories(
    variance_proxy: pd.Series,
    diagnostics: pd.DataFrame,
    events: pd.DataFrame,
    future_window: int,
    post_days: int,
    stride: int,
) -> pd.DataFrame:
    return build_har_median_counterfactual_trajectories(
        variance_proxy,
        diagnostics,
        events,
        future_window=future_window,
        post_days=post_days,
        stride=stride,
    )


@st.cache_data(show_spinner=False)
def cached_matches(
    variance_proxy: pd.Series,
    returns: pd.Series,
    diagnostics: pd.DataFrame,
    events: pd.DataFrame,
    future_window: int,
    post_days: int,
    controls_per_event: int,
    trend_window: int,
    maximum_position_distance: int,
    control_score_cap: float,
    exclusion_radius: int,
) -> pd.DataFrame:
    return select_matched_controls(
        variance_proxy,
        returns,
        diagnostics,
        events,
        future_window=future_window,
        post_days=post_days,
        controls_per_event=controls_per_event,
        trend_window=trend_window,
        maximum_position_distance=maximum_position_distance,
        control_score_cap=control_score_cap,
        exclusion_radius=exclusion_radius,
    )


@st.cache_data(show_spinner=False)
def cached_matched_trajectories(
    variance_proxy: pd.Series,
    events: pd.DataFrame,
    matches: pd.DataFrame,
    future_window: int,
    post_days: int,
    stride: int,
) -> pd.DataFrame:
    return build_matched_control_trajectories(
        variance_proxy,
        events,
        matches,
        future_window=future_window,
        post_days=post_days,
        stride=stride,
    )


@st.cache_data(show_spinner=False)
def cached_local_projections(
    variance_proxy: pd.Series,
    returns: pd.Series,
    diagnostics: pd.DataFrame,
    events: pd.DataFrame,
    future_window: int,
    post_days: int,
    stride: int,
    trend_window: int,
    contamination_radius: int,
) -> pd.DataFrame:
    return fit_local_projection_responses(
        variance_proxy,
        returns,
        diagnostics,
        events,
        future_window=future_window,
        post_days=post_days,
        stride=stride,
        trend_window=trend_window,
        contamination_radius=contamination_radius,
    )


INTERVALS = {"1 day": 24}

st.title("Asset Lab v6 · Daily")
st.caption(
    "MOEX OHLC → multiple variance estimators → HAR persistence → asymmetric trend forecast "
    "→ volatility innovations → three-method shock identification"
)

with st.sidebar:
    st.header("Data")
    secid = st.text_input("SECID", value="SBER").strip().upper()
    preferred_board = st.text_input(
        "Board (optional)",
        value="",
        help="Leave empty to use the primary board. SBER commonly resolves to TQBR.",
    ).strip().upper()
    default_end = date.today()
    default_start = default_end - timedelta(days=365 * 10)
    start_date = st.date_input("From", value=default_start)
    end_date = st.date_input("Till", value=default_end)
    interval_label = st.selectbox("Interval", list(INTERVALS), index=0)

    with st.expander("Daily-data cleaning", expanded=True):
        exclude_long_gaps = st.checkbox(
            "Exclude gap-dependent observations after long calendar gaps",
            value=True,
        )
        max_gap_days = st.slider(
            "Maximum calendar gap kept",
            min_value=3,
            max_value=30,
            value=7,
            disabled=not exclude_long_gaps,
        )
        exclude_unfinished_candle = st.checkbox(
            "Exclude the still-forming current daily candle",
            value=True,
        )
        yz_center_window = st.slider(
            "Yang–Zhang contribution centering window",
            min_value=10,
            max_value=120,
            value=20,
        )

    st.button("Load / refresh", type="primary", use_container_width=True)
    st.caption("Daily page preserves the validated v5 counterfactual workflow. Use Hourly Moments for M2/M3, mobility and process classification.")

if start_date >= end_date:
    st.error("`From` must be earlier than `Till`.")
    st.stop()
if not secid:
    st.info("Enter a MOEX SECID.")
    st.stop()

try:
    with st.spinner("Loading candles from MOEX ISS…"):
        route, candles = load_market_data(
            secid,
            start_date,
            end_date,
            INTERVALS[interval_label],
            preferred_board or None,
        )
except MoexApiError as exc:
    st.error(str(exc))
    st.stop()

if candles.empty:
    st.warning("MOEX ISS returned no candles for this route and period.")
    st.stop()

unfinished_candle_excluded = False
if exclude_unfinished_candle:
    candles, unfinished_candle_excluded = exclude_current_unfinished_daily_candle(candles)
    if candles.empty:
        st.warning("Only an unfinished current-day candle was available.")
        st.stop()

warnings = validate_candles(candles)
gap_report = calendar_gap_report(candles, threshold_days=max_gap_days)
raw_returns = log_returns(candles)
returns = log_returns(
    candles,
    max_calendar_gap_days=max_gap_days if exclude_long_gaps else None,
)
if "begin" in candles.columns:
    time_axis = pd.DatetimeIndex(pd.to_datetime(candles["begin"], errors="coerce"))
    raw_returns.index = time_axis
    returns.index = time_axis
else:
    time_axis = returns.index

proxies = ohlc_variance_proxies(
    candles,
    yz_center_window=yz_center_window,
    max_calendar_gap_days=max_gap_days if exclude_long_gaps else None,
)

route_label = f"{route.secid} · {route.engine}/{route.market}/{route.board}"
st.subheader(route_label)
if route.board_title:
    st.caption(route.board_title)

metric_columns = st.columns(6)
metric_columns[0].metric("Candles", f"{len(candles):,}")
metric_columns[1].metric("From", str(time_axis.min())[:10])
metric_columns[2].metric("Till", str(time_axis.max())[:10])
metric_columns[3].metric("Missing close", int(candles["close"].isna().sum()))
metric_columns[4].metric(
    "Excluded long-gap returns",
    int(raw_returns.notna().sum() - returns.notna().sum()),
)
metric_columns[5].metric(
    "Excluded live candle",
    int(unfinished_candle_excluded),
)

for warning in warnings:
    st.warning(warning)

if not gap_report.empty:
    action = "excluded from gap-dependent measures" if exclude_long_gaps else "kept"
    st.warning(
        f"Found {len(gap_report)} close-to-close observation(s) spanning more than "
        f"{max_gap_days} calendar days; they are currently {action}."
    )
    with st.expander("Long calendar gaps"):
        st.dataframe(gap_report, use_container_width=True, hide_index=True)

proxy_options = list(PROXY_LABELS)
proxy_format = lambda key: PROXY_LABELS.get(key, key)

overview_tab, volatility_tab, moments_tab, trend_tab, shock_tab, data_tab = st.tabs(
    [
        "Overview",
        "Volatility laboratory",
        "M2 / asymmetry",
        "Trend and forecast",
        "Shock and relaxation",
        "Raw data",
    ]
)

with overview_tab:
    st.plotly_chart(
        price_figure(candles, f"{route.secid}: price and volume"),
        use_container_width=True,
    )
    st.plotly_chart(
        line_figure(
            time_axis,
            {"cleaned log return": returns, "raw log return": raw_returns},
            "Logarithmic close-to-close returns",
            "log(Pₜ/Pₜ₋₁)",
            zero_line=True,
        ),
        use_container_width=True,
    )

    overview_window = st.slider("Rolling volatility window", 5, 120, 20, key="overview_window")
    selected_overview = st.multiselect(
        "Daily variance proxies shown",
        proxy_options,
        default=["close_to_close", "parkinson", "gap_rogers_satchell"],
        format_func=proxy_format,
    )
    volatility_lines = {
        PROXY_LABELS[key]: rolling_volatility_from_variance(proxies[key], overview_window)
        for key in selected_overview
    }
    volatility_lines["Yang–Zhang rolling"] = yang_zhang_volatility(
        candles,
        overview_window,
        max_calendar_gap_days=max_gap_days if exclude_long_gaps else None,
    )
    st.plotly_chart(
        line_figure(
            time_axis,
            volatility_lines,
            "Rolling annualised volatility estimates",
            "Volatility",
        ),
        use_container_width=True,
    )

with volatility_tab:
    st.subheader("What each estimator measures")
    st.markdown(
        "**Default rolling state:** Yang–Zhang, because it includes opening gaps and is "
        "drift-robust over a window. **Default daily event proxy:** gap² + "
        "Rogers–Satchell, because it separates overnight and intraday variation. "
        "Parkinson and Garman–Klass remain efficiency-oriented robustness checks, not "
        "the sole definition of volatility."
    )
    lab_window = st.slider("Comparison window", 5, 120, 20, key="lab_window")
    all_lines = {
        PROXY_LABELS[key]: rolling_volatility_from_variance(proxies[key], lab_window)
        for key in proxy_options
    }
    all_lines["Yang–Zhang rolling"] = yang_zhang_volatility(
        candles,
        lab_window,
        max_calendar_gap_days=max_gap_days if exclude_long_gaps else None,
    )
    st.plotly_chart(
        line_figure(time_axis, all_lines, "All supported OHLC volatility estimates", "Annualised volatility"),
        use_container_width=True,
    )

    correlation = proxy_correlation_matrix(proxies)
    st.plotly_chart(
        heatmap_figure(correlation, "Spearman correlation between daily variance proxies"),
        use_container_width=True,
    )
    with st.expander("Dates with the strongest estimator disagreement"):
        st.dataframe(proxy_disagreement_days(proxies, top_n=25), use_container_width=True, hide_index=True)

    decomposition = pd.DataFrame(
        {
            "overnight gap²": proxies["overnight_gap_squared"],
            "intraday Rogers–Satchell": proxies["intraday_rs"],
        }
    )
    st.plotly_chart(
        line_figure(
            time_axis,
            decomposition.to_dict("series"),
            "Daily overnight and intraday variance components",
            "Squared-log-return units",
        ),
        use_container_width=True,
    )

with moments_tab:
    moment_window = st.slider("Moment / asymmetry window", 5, 180, 20, key="moment_window")
    moments = rolling_moments(returns, moment_window)
    asymmetry = rolling_asymmetry_metrics(returns, moment_window)

    st.plotly_chart(
        line_figure(time_axis, {"M2": moments["m2"]}, "Rolling second central moment", "M2"),
        use_container_width=True,
    )
    st.plotly_chart(
        line_figure(
            time_axis,
            {"standardised skewness": moments["skewness"]},
            "Rolling standardised skewness M3 / M2^(3/2)",
            "Skewness",
            zero_line=True,
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        line_figure(
            time_axis,
            {
                "downside variance share": asymmetry["downside_variance_share"],
            },
            "Share of squared returns contributed by negative days",
            "Downside share",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        line_figure(
            time_axis,
            {
                "downside/upside log ratio": asymmetry["downside_upside_log_ratio"],
                "quantile skewness": asymmetry["quantile_skewness"],
            },
            "Robust asymmetry diagnostics",
            "Asymmetry",
            zero_line=True,
        ),
        use_container_width=True,
    )

    with st.expander("Raw M3 diagnostic", expanded=False):
        m3_view = st.radio(
            "Display scale",
            ["Signed cube-root", "Central 98% clipped", "Raw full scale"],
            horizontal=True,
        )
        raw_m3 = moments["m3"]
        if m3_view == "Signed cube-root":
            displayed = moments["signed_cuberoot_m3"]
            y_title = "sign(M3) · |M3|^(1/3)"
        elif m3_view == "Central 98% clipped":
            finite = raw_m3.dropna()
            displayed = raw_m3 if finite.empty else raw_m3.clip(*finite.quantile([0.01, 0.99]))
            y_title = "M3 clipped for display"
        else:
            displayed = raw_m3
            y_title = "M3"
        st.plotly_chart(
            line_figure(time_axis, {"M3": displayed}, "Rolling raw third central moment", y_title, zero_line=True),
            use_container_width=True,
        )

with trend_tab:
    st.subheader("Does trend add information beyond volatility persistence?")
    left, middle, right = st.columns(3)
    with left:
        trend_proxy_key = st.selectbox(
            "Variance proxy",
            proxy_options,
            index=proxy_options.index("gap_rogers_satchell"),
            format_func=proxy_format,
            key="trend_proxy",
        )
    with middle:
        trend_window = st.slider("Past trend window", 3, 180, 20)
    with right:
        future_horizon = st.slider("Future variance horizon", 1, 60, 5)

    trend_proxy = proxies[trend_proxy_key]
    relation = build_trend_variance_frame(returns, trend_proxy, trend_window, future_horizon)
    summary = correlation_summary(relation)
    summary_columns = st.columns(3)
    summary_columns[0].metric("Observations", f"{len(relation):,}")
    summary_columns[1].metric("Pearson", f"{summary['pearson']:.3f}")
    summary_columns[2].metric("Spearman", f"{summary['spearman']:.3f}")

    if len(relation) >= 10:
        st.plotly_chart(trend_scatter(relation), use_container_width=True)
        buckets = decile_summary(relation)
        if not buckets.empty:
            st.plotly_chart(decile_figure(buckets), use_container_width=True)
    else:
        st.info("Not enough observations for trend analysis.")

    st.latex(
        r"\log FV^2_{t,h}=a+HAR_t+b_-\max(-T_{t,L},0)+b_+\max(T_{t,L},0)+\varepsilon_t"
    )
    hac_lags = max(1, future_horizon - 1)
    regression = fit_har_trend_regression(relation, hac_lags=hac_lags)
    if regression is None:
        st.info("Not enough complete observations for HAR + asymmetric trend regression.")
    else:
        coefficient_map = regression.coefficients.set_index("term")
        metrics = st.columns(6)
        metrics[0].metric("Regression N", f"{regression.nobs:,}")
        metrics[1].metric("HAR-only R²", f"{regression.baseline_r_squared:.3f}")
        metrics[2].metric("Full R²", f"{regression.r_squared:.3f}")
        metrics[3].metric("ΔR² from trend", f"{regression.delta_r_squared:.3f}")
        metrics[4].metric("b−", f"{coefficient_map.loc['negative_trend', 'estimate']:.3f}")
        metrics[5].metric("p-value b− = b+", f"{regression.difference_p_value:.3g}")
        st.info(regression_interpretation(regression))
        st.dataframe(regression.coefficients, use_container_width=True, hide_index=True)

        if isinstance(relation.index, pd.DatetimeIndex):
            minimum_year = int(relation.index.year.min())
            periods = [
                (f"{minimum_year}–2021", f"{minimum_year}-01-01", "2021-12-31"),
                ("2022", "2022-01-01", "2022-12-31"),
                ("2023–latest", "2023-01-01", None),
            ]
            period_results = fit_har_trend_regression_by_period(
                relation,
                periods,
                hac_lags=hac_lags,
                minimum_observations=80,
            )
            if period_results:
                all_results = [regression, *period_results]
                st.plotly_chart(asymmetric_coefficients_figure(all_results), use_container_width=True)
                st.dataframe(
                    pd.DataFrame([result.summary_record() for result in all_results]),
                    use_container_width=True,
                    hide_index=True,
                )

        walk_forward = walk_forward_comparison(relation)
        st.subheader("Walk-forward out-of-sample comparison")
        if walk_forward.empty:
            st.info("The sample is too short for a yearly expanding-window evaluation.")
        else:
            st.dataframe(walk_forward, use_container_width=True, hide_index=True)
            mean_delta_qlike = walk_forward["delta_qlike_full_minus_baseline"].mean()
            mean_delta_mse = walk_forward["delta_log_mse_full_minus_baseline"].mean()
            st.caption(
                f"Average full-minus-baseline ΔQLIKE = {mean_delta_qlike:+.4f}; "
                f"Δlog-MSE = {mean_delta_mse:+.4f}. Negative values mean the trend model improved forecasts."
            )

with shock_tab:
    st.subheader("Volatility shocks: identification by three counterfactual methods")
    st.markdown(
        "The event definition is unchanged: a daily OHLC variance proxy must exceed a "
        "strictly past HAR forecast by an unusually large robust residual. The response is "
        "now estimated independently by **(1) a non-compounding HAR conditional-median "
        "path, (2) nearest matched no-shock days, and (3) HAC local projections**. A decay "
        "law is fitted only when the methods agree on a positive early response."
    )

    row1 = st.columns(4)
    with row1[0]:
        shock_proxy_key = st.selectbox(
            "Daily event variance proxy",
            proxy_options,
            index=proxy_options.index("gap_rogers_satchell"),
            format_func=proxy_format,
            key="shock_proxy",
        )
    with row1[1]:
        initial_training = st.slider("Initial HAR training", 120, 750, 252, 10)
    with row1[2]:
        refit_every = st.slider("Refit HAR every N days", 1, 60, 20)
    with row1[3]:
        mad_window = st.slider("Innovation MAD window", 60, 750, 250, 10)

    row2 = st.columns(4)
    with row2[0]:
        shock_quantile = st.slider("Common shock-score quantile", 0.80, 0.995, 0.95, 0.005)
    with row2[1]:
        cooldown = st.slider("Event cooldown", 0, 60, 10)
    with row2[2]:
        future_window = st.slider("Forward variance window", 1, 20, 5)
    with row2[3]:
        post_days = st.slider("Post-shock horizon", 10, 90, 30)
    stride = st.slider(
        "Trajectory step",
        1,
        max(1, future_window),
        1,
        help=(
            "Step 1 gives the most detailed curve. Setting the step equal to the forward "
            "window yields non-overlapping displayed windows; HAC/local bootstrap still "
            "handles dependence in estimation."
        ),
    )

    with st.expander("Matched-control and local-projection settings"):
        settings = st.columns(5)
        with settings[0]:
            controls_per_event = st.slider("Controls per event", 3, 30, 10)
        with settings[1]:
            matching_trend_window = st.slider("Pre-event trend window", 5, 120, 20)
        with settings[2]:
            maximum_position_distance = st.slider(
                "Maximum local date distance",
                126,
                1260,
                504,
                21,
                help="Trading observations; 504 is approximately two years.",
            )
        with settings[3]:
            control_score_cap = st.slider(
                "Maximum control shock score",
                -1.0,
                1.5,
                1.0,
                0.1,
            )
        with settings[4]:
            contamination_radius = st.slider(
                "Exclude days near selected shocks",
                0,
                60,
                max(10, cooldown),
            )

    shock_proxy = proxies[shock_proxy_key]
    with st.spinner("Fitting expanding past-only HAR forecasts…"):
        diagnostics = cached_har_diagnostics(
            shock_proxy,
            initial_training,
            refit_every,
            mad_window,
        )
    diagnostics = diagnostics.join(
        proxies[["overnight_gap_squared", "intraday_rs"]],
        how="left",
    )
    events = detect_volatility_shocks(
        diagnostics,
        returns,
        quantile=shock_quantile,
        cooldown=cooldown,
    )
    threshold = float(events["threshold"].iloc[0]) if not events.empty else None
    st.plotly_chart(
        har_shock_score_figure(diagnostics, events, threshold),
        use_container_width=True,
    )

    negative_events = int((events["direction"] == "negative").sum()) if not events.empty else 0
    positive_events = int((events["direction"] == "positive").sum()) if not events.empty else 0
    metrics = st.columns(4)
    metrics[0].metric("All selected shocks", len(events))
    metrics[1].metric("Negative-price shocks", negative_events)
    metrics[2].metric("Positive-price shocks", positive_events)
    metrics[3].metric("Common threshold", "—" if threshold is None else f"{threshold:.2f}")

    if events.empty:
        st.info("No volatility innovations exceed the selected threshold.")
    else:
        with st.expander("Detected shock table"):
            st.dataframe(events, use_container_width=True, hide_index=True)

        with st.spinner("Building non-compounding HAR median counterfactuals…"):
            har_trajectories = cached_har_median_trajectories(
                shock_proxy,
                diagnostics,
                events,
                future_window,
                post_days,
                stride,
            )

        with st.spinner("Selecting matched no-shock control days…"):
            matches = cached_matches(
                shock_proxy,
                returns,
                diagnostics,
                events,
                future_window,
                post_days,
                controls_per_event,
                matching_trend_window,
                maximum_position_distance,
                control_score_cap,
                contamination_radius,
            )
            matched_trajectories = cached_matched_trajectories(
                shock_proxy,
                events,
                matches,
                future_window,
                post_days,
                stride,
            )

        with st.spinner("Estimating HAC local projections by horizon…"):
            local_projections = cached_local_projections(
                shock_proxy,
                returns,
                diagnostics,
                events,
                future_window,
                post_days,
                stride,
                matching_trend_window,
                contamination_radius,
            )

        har_aggregate = (
            aggregate_counterfactual_trajectories(har_trajectories, bootstrap_samples=500)
            if not har_trajectories.empty
            else pd.DataFrame()
        )
        matched_aggregate = (
            aggregate_counterfactual_trajectories(matched_trajectories, bootstrap_samples=500)
            if not matched_trajectories.empty
            else pd.DataFrame()
        )
        local_aggregate = local_projection_aggregate(local_projections)
        aggregates = {
            "har_conditional_median": har_aggregate,
            "matched_controls": matched_aggregate,
            "local_projection": local_aggregate,
        }

        usable = st.columns(4)
        usable[0].metric(
            "HAR complete events",
            har_trajectories["event_id"].nunique() if not har_trajectories.empty else 0,
        )
        usable[1].metric(
            "Matched complete events",
            matched_trajectories["event_id"].nunique() if not matched_trajectories.empty else 0,
        )
        usable[2].metric(
            "Events with controls",
            matches["event_id"].nunique() if not matches.empty else 0,
        )
        usable[3].metric("Local-projection horizons", len(local_projections))

        balance = matching_balance_summary(matches)
        if not balance.empty:
            st.plotly_chart(matching_balance_figure(balance), use_container_width=True)
            maximum_smd = float(balance["standardized_mean_difference"].abs().max())
            if maximum_smd <= 0.10:
                st.success(f"Matched-control balance is good: maximum |SMD| = {maximum_smd:.3f}.")
            else:
                st.warning(
                    f"Some matched features remain imbalanced: maximum |SMD| = {maximum_smd:.3f}."
                )

        available_aggregates = {
            method: aggregate
            for method, aggregate in aggregates.items()
            if not aggregate.empty
        }
        if not available_aggregates:
            st.info("No complete response estimates are available. Lower the threshold or shorten the horizon.")
        else:
            st.plotly_chart(
                multi_method_response_figure(available_aggregates),
                use_container_width=True,
            )

            early_summary = method_early_response_summary(available_aggregates, early_end=5)
            st.subheader("Cross-method early-response summary")
            display_summary = early_summary.copy()
            if not display_summary.empty:
                display_summary["early_mean_excess"] *= 100.0
                st.dataframe(display_summary, use_container_width=True, hide_index=True)

            consensus_allowed, consensus_reason = consensus_decision(
                early_summary,
                direction="negative",
            )
            if consensus_allowed:
                st.success(f"Negative-shock response identified across methods: {consensus_reason}")
            else:
                st.warning(f"No cross-method negative-shock identification: {consensus_reason}")

            if not local_projections.empty:
                st.plotly_chart(
                    local_projection_difference_figure(local_projections),
                    use_container_width=True,
                )
                with st.expander("Local-projection coefficient table"):
                    lp_display = local_projections.copy()
                    st.dataframe(lp_display, use_container_width=True, hide_index=True)

            st.subheader("Decay models for the matched-control negative response")
            negative_matched = (
                matched_aggregate.loc[matched_aggregate["direction"] == "negative"]
                if not matched_aggregate.empty
                else pd.DataFrame()
            )
            if consensus_allowed and not negative_matched.empty:
                fits, fit_decision = fit_relaxation_models(
                    negative_matched,
                    minimum_events=20,
                )
                if fit_decision.allowed:
                    st.plotly_chart(
                        counterfactual_event_figure(
                            matched_aggregate,
                            fits,
                            fit_direction="negative",
                        ),
                        use_container_width=True,
                    )
                    st.caption(
                        "AIC/BIC are descriptive because curve points are dependent. "
                        "The event-level held-out RMSE table below is the stronger comparison."
                    )
                    if fits:
                        st.dataframe(
                            pd.DataFrame([fit.as_record() for fit in fits]),
                            use_container_width=True,
                            hide_index=True,
                        )
                    cv_table = cross_validate_relaxation_models(
                        matched_trajectories,
                        fit_decision,
                        direction="negative",
                        folds=5,
                    )
                    if not cv_table.empty:
                        st.subheader("Event-level held-out decay comparison")
                        st.dataframe(cv_table, use_container_width=True, hide_index=True)
                else:
                    st.warning(f"No relaxation law fitted: {fit_decision.reason}")
            else:
                st.info(
                    "Decay fitting is disabled until the early negative response is positive "
                    "under at least two methods and agrees with the local projection."
                )

            st.subheader("Negative versus positive response by broad horizon")
            comparison_rows: list[dict[str, object]] = []
            for method, trajectories in [
                ("HAR conditional median", har_trajectories),
                ("Matched controls", matched_trajectories),
            ]:
                for comparison in horizon_comparisons(trajectories):
                    record = comparison.as_record()
                    record["method"] = method
                    comparison_rows.append(record)
            if comparison_rows:
                comparison_frame = pd.DataFrame(comparison_rows)
                percent_columns = [
                    "negative_median_excess",
                    "positive_median_excess",
                    "negative_minus_positive",
                    "difference_ci_low",
                    "difference_ci_high",
                ]
                comparison_frame[percent_columns] *= 100.0
                st.dataframe(comparison_frame, use_container_width=True, hide_index=True)

            with st.expander("Matched-control diagnostics and raw response tables"):
                if not balance.empty:
                    st.markdown("**Matching balance**")
                    st.dataframe(balance, use_container_width=True, hide_index=True)
                if not matches.empty:
                    st.markdown("**Selected control matches**")
                    st.dataframe(matches, use_container_width=True, hide_index=True)
                if not har_aggregate.empty:
                    st.markdown("**HAR conditional-median aggregate**")
                    st.dataframe(har_aggregate, use_container_width=True, hide_index=True)
                if not matched_aggregate.empty:
                    st.markdown("**Matched-control aggregate**")
                    st.dataframe(matched_aggregate, use_container_width=True, hide_index=True)

        st.subheader("Aftershock exceedance rate")
        aftershock_quantile = st.slider(
            "Aftershock score quantile",
            0.75,
            0.99,
            0.90,
            0.01,
            help="Main events use the common shock threshold above; later aftershocks may use a lower threshold.",
        )
        rate = aftershock_excess_rate(
            diagnostics,
            events,
            horizon=min(post_days, 60),
            aftershock_quantile=aftershock_quantile,
        )
        omori_fit = fit_baseline_omori(rate)
        st.plotly_chart(aftershock_excess_figure(rate, omori_fit), use_container_width=True)
        if omori_fit is not None:
            if omori_fit.meaningful_decay:
                st.success(omori_fit.reason)
            else:
                st.warning(omori_fit.reason)
            st.dataframe(
                pd.DataFrame([omori_fit.as_record()]),
                use_container_width=True,
                hide_index=True,
            )
        st.caption(
            "The Omori comparison includes the unconditional background exceedance rate "
            "and must improve AIC over a flat-rate model before being called decay."
        )

with data_tab:
    export = candles.copy()
    for column in proxies.columns:
        export[column] = proxies[column].to_numpy()
    export["raw_log_return"] = raw_returns.to_numpy()
    export["cleaned_log_return"] = returns.to_numpy()
    st.dataframe(export, use_container_width=True, hide_index=True)
    st.download_button(
        "Download candles and computed proxies as CSV",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name=f"{route.secid}_asset_lab_v6_daily.csv",
        mime="text/csv",
    )
