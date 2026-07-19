from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from asset_lab.analysis.amplitude_relaxation import (
    MODEL_LABELS,
    cross_validate_amplitude_models,
    fit_amplitude_models,
    predict_amplitude_model,
)
from asset_lab.analysis.hourly import (
    affine_m3_horizon_test,
    aggregate_moment_trajectories,
    build_matched_moment_trajectories,
    detect_moment_shocks,
    fit_mobility_models,
    market_moment_frame,
    mobility_price_bins,
    overnight_moment_summary,
    panel_synchrony,
    past_hourly_standardisation,
    prepare_hourly_candles,
    rolling_standardised_moments,
    session_moment_summary,
)
from asset_lab.charts.figures import line_figure, price_figure
from asset_lab.data.moex import InstrumentRoute, MoexApiError, MoexClient
from asset_lab.ui import apply_asset_lab_style, bounded_number_input


st.set_page_config(page_title="Asset Lab v6 · Hourly", page_icon="🕐", layout="wide")
apply_asset_lab_style()


@st.cache_data(ttl=3600, show_spinner=False)
def load_hourly_market_data(
    secid: str,
    start_date: date,
    end_date: date,
    preferred_board: str | None,
) -> tuple[InstrumentRoute, pd.DataFrame]:
    client = MoexClient(timeout_seconds=60)
    route = client.resolve_route(secid, preferred_board=preferred_board or None)
    candles = client.load_candles(route, start_date, end_date, interval=60)
    return route, candles


@st.cache_data(show_spinner=False)
def read_hourly_zip(payload: bytes) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        for name in archive.namelist():
            lower = name.lower()
            if lower.endswith("/") or not (lower.endswith(".csv") or lower.endswith(".csv.gz")):
                continue
            if "/candles/" not in f"/{lower}" and "candle" not in lower:
                continue
            raw = archive.read(name)
            compression = "gzip" if lower.endswith(".gz") else None
            try:
                frame = pd.read_csv(BytesIO(raw), compression=compression)
            except (ValueError, OSError):
                continue
            if not {"begin", "open", "close", "high", "low", "volume"}.issubset(frame.columns):
                continue
            filename = name.rsplit("/", 1)[-1]
            asset = filename.removesuffix(".csv.gz").removesuffix(".csv").upper()
            frames[asset] = frame
    return frames


@st.cache_data(show_spinner=False)
def prepare_asset(
    candles: pd.DataFrame,
    regular_start: int,
    regular_end: int,
    scale_window: int,
) -> pd.DataFrame:
    prepared = prepare_hourly_candles(
        candles,
        regular_start_hour=regular_start,
        regular_end_hour=regular_end,
    )
    return past_hourly_standardisation(prepared, scale_window=scale_window)


@st.cache_data(show_spinner=False)
def prepare_panel(
    frames: dict[str, pd.DataFrame],
    regular_start: int,
    regular_end: int,
    scale_window: int,
) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for asset, candles in frames.items():
        try:
            prepared = prepare_asset(candles, regular_start, regular_end, scale_window)
        except ValueError:
            continue
        regular = prepared.loc[prepared["regular_session"]]
        if len(regular) < max(500, scale_window * 2):
            continue
        values = regular.set_index("begin")["z_return"]
        series[asset] = values[~values.index.duplicated(keep="last")]
    if not series:
        return pd.DataFrame()
    panel = pd.concat(series, axis=1).sort_index()
    minimum_assets = max(3, int(np.ceil(0.60 * panel.shape[1])))
    return panel.loc[panel.notna().sum(axis=1) >= minimum_assets]


@st.cache_data(show_spinner=False)
def cached_mobility_analysis(frame: pd.DataFrame, bins: int):
    binned = mobility_price_bins(frame, bins=bins)
    fits, cross_validation = fit_mobility_models(binned)
    return binned, fits, cross_validation


@st.cache_data(show_spinner=False)
def cached_process_analysis(
    frame: pd.DataFrame,
    quantile: float,
    cooldown: int,
    precursor_window: int,
    post_horizon: int,
    controls: int,
    synchrony: pd.Series | None,
    marketwide_threshold: float,
    isolated_threshold: float,
):
    events = detect_moment_shocks(
        frame,
        quantile=quantile,
        cooldown=cooldown,
        precursor_window=precursor_window,
        post_horizon=post_horizon,
        synchrony=synchrony,
        marketwide_threshold=marketwide_threshold,
        isolated_threshold=isolated_threshold,
    )
    trajectories, matches = build_matched_moment_trajectories(
        frame,
        events,
        controls_per_event=controls,
        pre_horizon=min(12, precursor_window),
        post_horizon=post_horizon,
        exclusion_radius=max(cooldown, 12),
    )
    aggregate = aggregate_moment_trajectories(trajectories, bootstrap_samples=250)
    return events, trajectories, matches, aggregate


def trajectory_figure(
    aggregate: pd.DataFrame,
    value: str,
    low: str,
    high: str,
    title: str,
    y_title: str,
) -> go.Figure:
    figure = go.Figure()
    for (process_class, direction), group in aggregate.groupby(["process_class", "direction"]):
        group = group.sort_values("offset")
        label = f"{process_class} · {direction}"
        figure.add_trace(
            go.Scatter(
                x=group["offset"],
                y=group[value],
                mode="lines",
                name=label,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=pd.concat([group["offset"], group["offset"].iloc[::-1]]),
                y=pd.concat([group[high], group[low].iloc[::-1]]),
                fill="toself",
                line={"width": 0},
                opacity=0.12,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.add_vline(x=0, line_dash="dash")
    figure.add_hline(y=0, line_dash="dot")
    figure.update_layout(title=title, xaxis_title="Hourly offset", yaxis_title=y_title)
    return figure


def mobility_figure(bins: pd.DataFrame, fits) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=bins["price"],
            y=bins["mobility"],
            mode="markers",
            marker={"size": np.sqrt(bins["observations"]).clip(5, 18)},
            text=bins["year"].astype(str),
            name="Price-bin estimate",
        )
    )
    grid = np.linspace(bins["price"].min(), bins["price"].max(), 200)
    for fit in fits:
        if fit.model == "constant":
            prediction = np.full(len(grid), fit.parameters["a"])
        elif fit.model == "proportional":
            prediction = fit.parameters["b"] * grid
        else:
            prediction = fit.parameters["a"] + fit.parameters["b"] * grid
        figure.add_trace(go.Scatter(x=grid, y=prediction, mode="lines", name=fit.model))
    figure.update_layout(
        title="Conditional price mobility",
        xaxis_title="Price F",
        yaxis_title="Estimated mobility of ΔF",
    )
    return figure


def amplitude_fit_figure(
    aggregate: pd.DataFrame,
    trajectories: pd.DataFrame,
    process_class: str,
    direction: str,
    fits,
    fit_end: int,
    initial_hours: int = 3,
) -> go.Figure:
    selected = aggregate.loc[
        (aggregate["process_class"] == process_class)
        & (aggregate["direction"] == direction)
        & aggregate["offset"].between(0, fit_end)
    ].sort_values("offset")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=selected["offset"],
            y=selected["median_abnormal_m2"],
            mode="markers+lines",
            name="Matched response",
            line={"width": 2.5},
        )
    )

    selected_events = trajectories.loc[
        (trajectories["process_class"] == process_class)
        & (trajectories["direction"] == direction)
        & trajectories["offset"].between(1, initial_hours)
    ]
    x0_by_event = selected_events.groupby("event_id")["abnormal_m2"].median().clip(lower=0.0)
    median_x0 = float(x0_by_event.median()) if not x0_by_event.empty else float("nan")
    offsets = np.arange(initial_hours + 1, fit_end + 1, dtype=int)
    times = offsets - initial_hours
    if np.isfinite(median_x0):
        for fit in fits:
            prediction = predict_amplitude_model(
                fit.model,
                median_x0,
                times,
                fit.parameters,
            )
            figure.add_trace(
                go.Scatter(
                    x=offsets,
                    y=prediction,
                    mode="lines",
                    name=fit.label,
                )
            )

    figure.add_vrect(
        x0=1,
        x1=initial_hours,
        fillcolor="rgba(255,255,255,0.04)",
        line_width=0,
        annotation_text="x₀ window",
        annotation_position="top left",
    )
    figure.add_hline(y=0, line_dash="dot")
    figure.update_layout(
        title=f"Amplitude-aware relaxation · {process_class} · {direction}",
        xaxis_title="Hours after shock",
        yaxis_title="Abnormal M2",
        legend_title="Model",
    )
    return figure


def model_explainer() -> None:
    with st.expander("Why Landau is not constrained mobility"):
        left, right = st.columns(2)
        with left:
            st.markdown("**Kurbakovsky constrained mobility**")
            st.latex(r"x(t)=2q e^{-t/\tau}+q^2e^{-2t/\tau},\quad q=\sqrt{1+x_0}-1")
            st.caption(
                "Mobility relaxes exponentially and variance is its square. This is a tightly "
                "constrained double exponential: both time scales and weights are linked."
            )
        with right:
            st.markdown("**Amplitude-aware Landau**")
            st.latex(
                r"x(t)=\frac{x_0e^{-t/\tau}}{\sqrt{1+\theta x_0^2(1-e^{-2t/\tau})}}"
            )
            st.caption(
                "The fractional decay rate depends nonlinearly on the event amplitude x₀. "
                "It is a separate model, not a special case hidden inside constrained mobility."
            )
        st.markdown(
            '<div class="asset-model-note">The earlier research win was specific: Landau was most '
            'useful for preheated negative events and their integrated 12-hour excess burden. '
            'For abrupt or ordinary moderate shocks, the improvement was small, so the app keeps '
            'the held-out comparison visible instead of declaring a universal law.</div>',
            unsafe_allow_html=True,
        )


st.title("Asset Lab v6 · Hourly Moments")
st.caption(
    "Hourly MOEX OHLC → past-only standardisation → M2/M3 → mobility → "
    "abrupt/preheated processes → amplitude-aware relaxation"
)

with st.sidebar:
    st.header("Hourly data")
    source = st.radio("Source", ["MOEX live", "Multi-asset ZIP"], horizontal=False)
    uploaded = None
    panel_frames: dict[str, pd.DataFrame] = {}
    route_label = ""

    if source == "MOEX live":
        secid = st.text_input("SECID", value="SBER").strip().upper()
        board = st.text_input("Board (optional)", value="").strip().upper()
        end_date = st.date_input("Till", value=date.today())
        start_date = st.date_input("From", value=end_date - timedelta(days=365 * 4))
    else:
        uploaded = st.file_uploader(
            "Hourly ZIP",
            type=["zip"],
            help="Supports the moex_hourly_core exporter layout and ordinary CSV/CSV.GZ candle files.",
        )
        if uploaded is not None:
            panel_frames = read_hourly_zip(uploaded.getvalue())
        asset_options = sorted(panel_frames)
        secid = st.selectbox("Displayed asset", asset_options) if asset_options else ""
        board = ""
        start_date = None
        end_date = None

    with st.expander("Session and scaling", expanded=False):
        regular_start = int(
            st.number_input("Regular session starts", min_value=9, max_value=15, value=10, step=1)
        )
        regular_end = int(
            st.number_input("Regular session ends", min_value=16, max_value=23, value=18, step=1)
        )
        scale_window = int(
            st.number_input(
                "Past robust-scale window",
                min_value=120,
                max_value=750,
                value=252,
                step=12,
            )
        )
    st.caption("Default regular session: 10:00–18:59. Opening, evening and overnight stay separate.")

if source == "Multi-asset ZIP" and uploaded is None:
    st.info("Upload the hourly ZIP produced by the exporter.")
    st.stop()
if not secid:
    st.warning("No usable hourly candle file was found in the ZIP.")
    st.stop()

try:
    if source == "MOEX live":
        if start_date >= end_date:
            st.error("`From` must be earlier than `Till`.")
            st.stop()
        with st.spinner("Loading hourly candles from MOEX ISS…"):
            route, candles = load_hourly_market_data(secid, start_date, end_date, board or None)
        route_label = f"{route.secid} · {route.engine}/{route.market}/{route.board}"
    else:
        candles = panel_frames[secid]
        route_label = f"{secid} · uploaded hourly panel"
    frame = prepare_asset(candles, regular_start, regular_end, scale_window)
except (MoexApiError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

regular = frame.loc[frame["regular_session"]].reset_index(drop=True)
if len(regular) < 500:
    st.warning("At least 500 regular-session candles are needed for the hourly laboratory.")
    st.stop()

panel = pd.DataFrame()
synchrony = None
if source == "Multi-asset ZIP" and len(panel_frames) >= 3:
    with st.spinner("Preparing the cross-asset moment panel…"):
        panel = prepare_panel(panel_frames, regular_start, regular_end, scale_window)
    if not panel.empty and secid in panel.columns:
        synchrony = panel_synchrony(panel).reindex(regular["begin"]).set_axis(regular["begin"])

st.subheader(route_label)
metrics = st.columns(6)
metrics[0].metric("All candles", f"{len(frame):,}")
metrics[1].metric("Regular candles", f"{len(regular):,}")
metrics[2].metric("From", str(frame["begin"].min())[:10])
metrics[3].metric("Till", str(frame["begin"].max())[:10])
metrics[4].metric("Years", int(frame["year"].nunique()))
metrics[5].metric("Panel assets", panel.shape[1] if not panel.empty else 1)

overview_tab, moments_tab, mobility_tab, processes_tab, market_tab, data_tab = st.tabs(
    [
        "Overview",
        "M2 / M3 and sessions",
        "Mobility and Kurbakovsky M3",
        "Two-process shocks",
        "Market moments",
        "Raw data",
    ]
)

with overview_tab:
    st.plotly_chart(price_figure(frame, f"{secid}: hourly price and volume"), use_container_width=True)
    st.plotly_chart(
        line_figure(
            frame["begin"],
            {"open-to-close log return": frame["log_return_oc"]},
            "Hourly within-candle returns",
            "log(close/open)",
            zero_line=True,
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        line_figure(
            regular["begin"],
            {"z return": regular["z_return"], "conditional scale": regular["conditional_scale"]},
            "Past-only hour-of-day standardisation",
            "Standardised return / scale",
            zero_line=True,
        ),
        use_container_width=True,
    )

with moments_tab:
    moment_window = int(
        st.number_input(
            "Rolling moment window (regular hours)",
            min_value=12,
            max_value=500,
            value=60,
            step=12,
        )
    )
    moments = rolling_standardised_moments(regular, moment_window)
    moment_axis = regular["begin"]
    st.plotly_chart(
        line_figure(moment_axis, {"M2": moments["m2"]}, "Rolling standardised M2", "M2"),
        use_container_width=True,
    )
    st.plotly_chart(
        line_figure(
            moment_axis,
            {"M3": moments["m3"], "skewness": moments["skewness"]},
            "Rolling M3 and standardised skewness",
            "Moment / skewness",
            zero_line=True,
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        line_figure(
            moment_axis,
            {"downside variance share": moments["downside_share"]},
            "Negative-return share of M2",
            "Share",
        ),
        use_container_width=True,
    )

    session_summary = session_moment_summary(frame)
    overnight_summary = overnight_moment_summary(frame)
    left, right = st.columns(2)
    with left:
        session_figure = go.Figure(go.Bar(x=session_summary["hour"], y=session_summary["skewness"]))
        session_figure.add_hline(y=0)
        session_figure.update_layout(
            title="Skewness by candle start hour",
            xaxis_title="Hour",
            yaxis_title="Robust skewness",
        )
        st.plotly_chart(session_figure, use_container_width=True)
        st.dataframe(session_summary, use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Overnight M3")
        st.latex(r"r_{night}=\log(\mathrm{first\ open}_t/\mathrm{last\ close}_{t-1})")
        st.dataframe(overnight_summary, use_container_width=True, hide_index=True)
        if not overnight_summary.empty:
            robust_row = overnight_summary.loc[
                overnight_summary["filter"] == "exclude_2022_and_20pct"
            ]
            if not robust_row.empty:
                st.metric("Robust overnight skewness", f"{float(robust_row.iloc[0]['skewness']):.3f}")

with mobility_tab:
    mobility_bin_count = int(
        st.number_input("Price quantile bins per year", min_value=5, max_value=12, value=8, step=1)
    )
    with st.spinner("Estimating conditional price mobility…"):
        binned, mobility_fits, mobility_cv = cached_mobility_analysis(frame, mobility_bin_count)
    if binned.empty:
        st.info("The sample is too short for mobility estimation by year and price bin.")
    else:
        st.latex(r"M(F)=a,\qquad M(F)=bF,\qquad M(F)=a+bF")
        st.plotly_chart(mobility_figure(binned, mobility_fits), use_container_width=True)
        st.dataframe(
            pd.DataFrame([fit.as_record() for fit in mobility_fits]),
            use_container_width=True,
            hide_index=True,
        )
        if not mobility_cv.empty:
            st.subheader("Leave-one-year-out mobility comparison")
            st.dataframe(mobility_cv, use_container_width=True, hide_index=True)
            winner = str(mobility_cv.iloc[0]["model"])
            if winner == "proportional":
                st.success("The proportional geometric law M(F)=bF is the best out-of-year model.")
            elif winner == "affine":
                st.success("The full affine law M(F)=a+bF adds out-of-year information.")
            else:
                st.info("A price-independent mobility is the best out-of-year model for this asset.")

        affine_fit = next((fit for fit in mobility_fits if fit.model == "affine"), None)
        if affine_fit is not None:
            st.subheader("Finite-horizon M3 implied by affine mobility")
            st.latex(r"\gamma_K(h)=\left(e^{b^2h}+2\right)\sqrt{e^{b^2h}-1}\geq0")
            horizon_test = affine_m3_horizon_test(frame, affine_fit)
            st.dataframe(horizon_test, use_container_width=True, hide_index=True)
            if not horizon_test.empty:
                figure = go.Figure()
                figure.add_trace(
                    go.Scatter(
                        x=horizon_test["horizon_hours"],
                        y=horizon_test["predicted_affine_skewness"],
                        mode="lines+markers",
                        name="Affine prediction",
                    )
                )
                figure.add_trace(
                    go.Scatter(
                        x=horizon_test["horizon_hours"],
                        y=horizon_test["observed_skewness"],
                        mode="lines+markers",
                        name="Observed",
                    )
                )
                figure.add_hline(y=0)
                figure.update_layout(
                    title="Does state-dependent mobility explain M3?",
                    xaxis_title="Horizon, regular-session hours",
                    yaxis_title="Skewness",
                )
                st.plotly_chart(figure, use_container_width=True)
                residual = horizon_test["observed_skewness"] - horizon_test["predicted_affine_skewness"]
                st.caption(
                    f"Mean observed-minus-affine skewness = {residual.mean():+.4f}. "
                    "A large residual indicates mechanisms beyond diffusion."
                )

with processes_tab:
    st.markdown(
        """
        <div class="asset-callout">
            <div class="asset-kicker">Process split</div>
            An <strong>abrupt-like</strong> shock starts from a relatively cold M2 state; a
            <strong>preheated-like</strong> shock arrives after elevated M2. With a multi-asset ZIP,
            synchrony additionally creates strict <strong>external-like</strong> and
            <strong>internal-like</strong> proxies.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if synchrony is None:
        marketwide_threshold = 0.40
        isolated_threshold = 0.20
        st.info("Upload a multi-asset ZIP to activate market-wide versus isolated classification.")

    with st.expander("Analysis parameters", expanded=False):
        st.caption(
            "Type exact values. The thin strip shows the value's position inside the admissible range; "
            "it is a scale, not a good/bad score."
        )
        settings = st.columns(5)
        with settings[0]:
            shock_quantile = float(
                bounded_number_input(
                    "|z| shock quantile",
                    min_value=0.95,
                    max_value=0.999,
                    value=0.99,
                    step=0.001,
                    key="shock_quantile",
                    format="%.3f",
                )
            )
        with settings[1]:
            cooldown = int(
                bounded_number_input(
                    "Cooldown (hours)",
                    min_value=3,
                    max_value=48,
                    value=12,
                    step=1,
                    key="cooldown",
                )
            )
        with settings[2]:
            precursor_window = int(
                bounded_number_input(
                    "Preheating window",
                    min_value=6,
                    max_value=36,
                    value=12,
                    step=1,
                    key="precursor_window",
                )
            )
        with settings[3]:
            post_horizon = int(
                bounded_number_input(
                    "Post-shock horizon",
                    min_value=12,
                    max_value=48,
                    value=24,
                    step=1,
                    key="post_horizon",
                )
            )
        with settings[4]:
            controls = int(
                bounded_number_input(
                    "Matched controls",
                    min_value=3,
                    max_value=20,
                    value=8,
                    step=1,
                    key="controls",
                )
            )

        if synchrony is not None:
            sync_settings = st.columns(2)
            with sync_settings[0]:
                marketwide_threshold = float(
                    bounded_number_input(
                        "Market-wide synchrony",
                        min_value=0.15,
                        max_value=0.80,
                        value=0.40,
                        step=0.05,
                        key="marketwide_threshold",
                        format="%.2f",
                    )
                )
            with sync_settings[1]:
                isolated_threshold = float(
                    bounded_number_input(
                        "Isolated synchrony",
                        min_value=0.05,
                        max_value=0.40,
                        value=0.20,
                        step=0.05,
                        key="isolated_threshold",
                        format="%.2f",
                    )
                )

    st.caption(
        f"Current setup: q ≥ {shock_quantile:.3f} · cooldown {cooldown}h · preheat {precursor_window}h "
        f"· horizon {post_horizon}h · {controls} controls"
    )

    with st.spinner("Detecting shocks and matching no-shock controls…"):
        events, trajectories, matches, aggregate = cached_process_analysis(
            frame,
            shock_quantile,
            cooldown,
            precursor_window,
            post_horizon,
            controls,
            synchrony,
            marketwide_threshold,
            isolated_threshold,
        )

    event_metrics = st.columns(5)
    event_metrics[0].metric("Selected shocks", len(events))
    event_metrics[1].metric(
        "Matched shocks",
        trajectories["event_id"].nunique() if not trajectories.empty else 0,
    )
    event_metrics[2].metric(
        "Abrupt-like",
        int((events.get("heating_class") == "abrupt-like").sum()) if not events.empty else 0,
    )
    event_metrics[3].metric(
        "Preheated-like",
        int((events.get("heating_class") == "preheated-like").sum()) if not events.empty else 0,
    )
    event_metrics[4].metric(
        "Negative shocks",
        int((events.get("direction") == "negative").sum()) if not events.empty else 0,
    )

    model_explainer()

    if events.empty or aggregate.empty:
        st.info("No complete matched shock trajectories were available under the selected settings.")
    else:
        st.plotly_chart(
            trajectory_figure(
                aggregate,
                "median_abnormal_m2",
                "m2_ci_low",
                "m2_ci_high",
                "Matched-control M2 trajectories",
                "Abnormal M2",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            trajectory_figure(
                aggregate,
                "mean_abnormal_m3",
                "m3_ci_low",
                "m3_ci_high",
                "Matched-control M3 trajectories",
                "Abnormal M3",
            ),
            use_container_width=True,
        )

        available_classes = sorted(
            aggregate.loc[aggregate["process_class"] != "other", "process_class"].unique()
        )
        if available_classes:
            selection = st.columns(2)
            process_class = selection[0].selectbox("Process fitted", available_classes)
            direction_options = sorted(
                aggregate.loc[aggregate["process_class"] == process_class, "direction"].unique()
            )
            direction = selection[1].selectbox("Price direction", direction_options)
            fit_end = min(24, post_horizon)
            fits = fit_amplitude_models(
                trajectories,
                process_class=process_class,
                direction=direction,
                initial_hours=3,
                fit_end=fit_end,
            )
            if fits:
                st.plotly_chart(
                    amplitude_fit_figure(
                        aggregate,
                        trajectories,
                        process_class,
                        direction,
                        fits,
                        fit_end,
                    ),
                    use_container_width=True,
                )
                fit_table = pd.DataFrame([fit.as_record() for fit in fits])
                st.dataframe(fit_table, use_container_width=True, hide_index=True)

                cv = cross_validate_amplitude_models(
                    trajectories,
                    process_class=process_class,
                    direction=direction,
                    initial_hours=3,
                    fit_end=fit_end,
                    folds=5,
                )
                if not cv.empty:
                    st.subheader("Event-level held-out relaxation comparison")
                    display_cv = cv[
                        ["label", "mean", "median", "std", "burden_mae_12", "events", "winner"]
                    ].rename(columns={"label": "model"})
                    st.dataframe(display_cv, use_container_width=True, hide_index=True)
                    best = cv.iloc[0]
                    spread = float(cv["mean"].max() - cv["mean"].min())
                    if best["model"] == "landau" and process_class in {
                        "preheated-like",
                        "internal-like",
                    }:
                        st.success(
                            f"Amplitude-aware Landau leads held-out RMSE for {process_class}. "
                            "Treat the practical 12-hour burden column as the stronger decision metric."
                        )
                    else:
                        st.info(
                            f"Held-out winner: {MODEL_LABELS[str(best['model'])]}. "
                            "Landau is retained as a candidate, not forced as a universal winner."
                        )
                    st.caption(
                        f"Best-to-worst held-out RMSE spread = {spread:.4f}. Small spreads mean "
                        "persistence is identified more clearly than a unique law."
                    )

        with st.expander("Event and matching tables"):
            st.markdown("**Detected events**")
            st.dataframe(events, use_container_width=True, hide_index=True)
            st.markdown("**Match quality**")
            st.dataframe(matches, use_container_width=True, hide_index=True)

with market_tab:
    if panel.empty:
        st.info("The market-moment panel requires a ZIP with at least three usable hourly assets.")
    else:
        market = market_moment_frame(panel)
        st.metric("Aligned panel assets", panel.shape[1])
        st.plotly_chart(
            line_figure(
                market.index,
                {"market M2": market["market_m2"], "market M3": market["market_m3"]},
                "Cross-asset market moments",
                "Standardised moment",
                zero_line=True,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            line_figure(
                market.index,
                {"95% synchrony": market["synchrony_95"]},
                "Share of assets in their own extreme-return tail",
                "Synchrony",
            ),
            use_container_width=True,
        )
        forecast_horizon = int(
            st.number_input("Future market-M2 horizon", min_value=1, max_value=24, value=6, step=1)
        )
        future_columns = [market["market_m2"].shift(-step) for step in range(1, forecast_horizon + 1)]
        future_m2 = pd.concat(future_columns, axis=1).mean(axis=1)
        past_m3 = market["market_m3"].rolling(6, min_periods=3).mean()
        diagnostic = pd.DataFrame(
            {"past_market_m3": past_m3, "future_market_m2": future_m2}
        ).dropna()
        correlation = diagnostic.corr(method="spearman").iloc[0, 1] if len(diagnostic) else np.nan
        st.metric("Spearman: recent market M3 vs future M2", f"{correlation:+.3f}")
        diagnostic["m3_decile"] = pd.qcut(
            diagnostic["past_market_m3"],
            10,
            labels=False,
            duplicates="drop",
        )
        deciles = (
            diagnostic.groupby("m3_decile")
            .agg(
                recent_market_m3=("past_market_m3", "mean"),
                future_market_m2=("future_market_m2", "mean"),
                observations=("future_market_m2", "size"),
            )
            .reset_index()
        )
        st.dataframe(deciles, use_container_width=True, hide_index=True)
        st.caption(
            "The 25-stock research panel found market-level M3 more useful than spectral graph "
            "features for future market M2. This panel exposes the same state variables live."
        )

with data_tab:
    export = frame.copy()
    st.dataframe(export, use_container_width=True, hide_index=True)
    st.download_button(
        "Download prepared hourly moments as CSV",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name=f"{secid}_asset_lab_v6_hourly.csv",
        mime="text/csv",
    )
