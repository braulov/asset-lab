from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from asset_lab.analysis.regression import AsymmetricRegressionResult
from asset_lab.analysis.relaxation import FitResult, omori_rate, predict_fit


def price_figure(candles: pd.DataFrame, title: str) -> go.Figure:
    has_volume = "volume" in candles.columns and candles["volume"].notna().any()
    rows = 2 if has_volume else 1
    heights = [0.78, 0.22] if has_volume else [1.0]
    figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
    )
    x = candles["begin"] if "begin" in candles.columns else candles.index
    figure.add_trace(
        go.Candlestick(
            x=x,
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    if has_volume:
        figure.add_trace(go.Bar(x=x, y=candles["volume"], name="Volume"), row=2, col=1)
        figure.update_yaxes(title_text="Volume", row=2, col=1)
    figure.update_yaxes(title_text="Price", row=1, col=1)
    figure.update_layout(title=title, xaxis_rangeslider_visible=False, hovermode="x unified")
    return figure


def line_figure(
    x: pd.Series | pd.Index,
    series: dict[str, pd.Series],
    title: str,
    y_title: str,
    zero_line: bool = False,
) -> go.Figure:
    figure = go.Figure()
    for name, values in series.items():
        figure.add_trace(go.Scatter(x=x, y=values, mode="lines", name=name))
    if zero_line:
        figure.add_hline(y=0)
    figure.update_layout(title=title, yaxis_title=y_title, hovermode="x unified")
    return figure


def heatmap_figure(frame: pd.DataFrame, title: str) -> go.Figure:
    figure = px.imshow(
        frame,
        text_auto=".2f",
        aspect="auto",
        title=title,
        zmin=-1.0,
        zmax=1.0,
    )
    return figure


def trend_scatter(frame: pd.DataFrame) -> go.Figure:
    figure = px.scatter(
        frame,
        x="normalized_trend",
        y="future_volatility",
        opacity=0.35,
        labels={
            "normalized_trend": "Volatility-normalised past trend",
            "future_volatility": "Future annualised volatility",
        },
        title="Past trend and future volatility",
    )
    figure.add_vline(x=0)
    return figure


def decile_figure(summary: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=summary["bucket"],
            y=summary["future_vol_mean"],
            mode="lines+markers",
            name="Mean",
            customdata=np.column_stack([summary["trend_mean"], summary["count"]]),
            hovertemplate=(
                "Group %{x}<br>Future vol: %{y:.3f}<br>"
                "Mean trend: %{customdata[0]:.3f}<br>N: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=summary["bucket"],
            y=summary["future_vol_median"],
            mode="lines+markers",
            name="Median",
        )
    )
    figure.update_layout(
        title="Future volatility by trend quantile",
        xaxis_title="Strong decline → strong rise",
        yaxis_title="Future annualised volatility",
    )
    return figure


def asymmetric_coefficients_figure(results: list[AsymmetricRegressionResult]) -> go.Figure:
    figure = go.Figure()
    frame = pd.DataFrame([result.summary_record() for result in results])
    if frame.empty:
        return figure
    figure.add_trace(
        go.Scatter(
            x=frame["period"],
            y=frame["negative_trend"],
            mode="markers+lines",
            name="Negative trend coefficient",
            error_y={
                "type": "data",
                "symmetric": False,
                "array": frame["negative_ci_high"] - frame["negative_trend"],
                "arrayminus": frame["negative_trend"] - frame["negative_ci_low"],
            },
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["period"],
            y=frame["positive_trend"],
            mode="markers+lines",
            name="Positive trend coefficient",
            error_y={
                "type": "data",
                "symmetric": False,
                "array": frame["positive_ci_high"] - frame["positive_trend"],
                "arrayminus": frame["positive_trend"] - frame["positive_ci_low"],
            },
        )
    )
    figure.add_hline(y=0)
    figure.update_layout(
        title="Asymmetric trend coefficients by period (95% HAC CI)",
        xaxis_title="Sample",
        yaxis_title="Coefficient in log future variance model",
        hovermode="x unified",
    )
    return figure


def har_shock_score_figure(
    diagnostics: pd.DataFrame,
    events: pd.DataFrame,
    threshold: float | None,
) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.62, 0.38],
    )
    figure.add_trace(
        go.Scatter(
            x=diagnostics.index,
            y=diagnostics["shock_score"],
            mode="lines",
            name="HAR innovation score",
        ),
        row=1,
        col=1,
    )
    if threshold is not None and np.isfinite(threshold):
        figure.add_hline(y=threshold, annotation_text=f"common threshold={threshold:.2f}", row=1, col=1)
    if not events.empty:
        for direction in ["negative", "positive"]:
            subset = events.loc[events["direction"] == direction]
            if subset.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=subset["timestamp"],
                    y=subset["shock_score"],
                    mode="markers",
                    name=f"{direction} price shock",
                    customdata=np.column_stack(
                        [subset["return"], subset["source"], subset["gap_share"]]
                    ),
                    hovertemplate=(
                        "%{x}<br>score=%{y:.2f}<br>return=%{customdata[0]:.3%}"
                        "<br>source=%{customdata[1]}<br>gap share=%{customdata[2]:.1%}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
    figure.add_trace(
        go.Scatter(
            x=diagnostics.index,
            y=diagnostics["innovation"],
            mode="lines",
            name="log-variance innovation",
        ),
        row=2,
        col=1,
    )
    figure.add_hline(y=0, row=2, col=1)
    figure.update_yaxes(title_text="Robust score", row=1, col=1)
    figure.update_yaxes(title_text="Innovation", row=2, col=1)
    figure.update_layout(
        title="Unexpected volatility innovations relative to a past-only HAR model",
        hovermode="x unified",
    )
    return figure


def counterfactual_event_figure(
    aggregate: pd.DataFrame,
    fits: list[FitResult] | None = None,
    fit_direction: str = "negative",
) -> go.Figure:
    figure = go.Figure()
    if aggregate.empty:
        return figure
    direction_names = {"negative": "Negative-price volatility shocks", "positive": "Positive-price volatility shocks"}
    for direction in ["negative", "positive"]:
        subset = aggregate.loc[aggregate["direction"] == direction].sort_values("offset")
        if subset.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=subset["offset"],
                y=subset["bootstrap_ci_high"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=subset["offset"],
                y=subset["bootstrap_ci_low"],
                mode="lines",
                fill="tonexty",
                line={"width": 0},
                name=f"{direction} 95% event-bootstrap CI",
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=subset["offset"],
                y=subset["median"],
                mode="lines+markers",
                name=direction_names[direction],
            )
        )
    figure.add_hline(y=0)

    if fits:
        fit_data = aggregate.loc[aggregate["direction"] == fit_direction]
        for fit in fits:
            if not fit.success or fit.fit_start is None or fit.fit_end is None:
                continue
            offsets = fit_data.loc[
                (fit_data["offset"] >= fit.fit_start)
                & (fit_data["offset"] <= fit.fit_end),
                "offset",
            ].to_numpy(dtype=float)
            relative = offsets - float(fit.fit_start)
            prediction = predict_fit(fit.model, relative, fit.parameters)
            figure.add_trace(
                go.Scatter(
                    x=offsets,
                    y=prediction,
                    mode="lines",
                    name=f"{fit_direction} {fit.model} (AIC={fit.aic:.1f})",
                )
            )

    figure.update_layout(
        title="Abnormal future variance relative to a recursive no-shock HAR counterfactual",
        xaxis_title="Trading periods after the shock",
        yaxis_title="Actual variance / no-shock variance − 1",
        hovermode="x unified",
    )
    return figure


def aftershock_rate_figure(rate_frame: pd.DataFrame, fit: FitResult | None) -> go.Figure:
    figure = go.Figure()
    if rate_frame.empty:
        return figure
    figure.add_trace(
        go.Scatter(
            x=rate_frame["offset"],
            y=rate_frame["rate"],
            mode="lines+markers",
            name="Observed exceedance rate",
        )
    )
    if fit is not None and fit.success:
        offsets = rate_frame["offset"].to_numpy(dtype=float)
        prediction = omori_rate(
            offsets,
            fit.parameters["scale"],
            fit.parameters["shift"],
            fit.parameters["exponent"],
        )
        figure.add_trace(
            go.Scatter(
                x=offsets,
                y=prediction,
                mode="lines",
                name=f"Omori fit p={fit.parameters['exponent']:.2f}",
            )
        )
    figure.update_layout(
        title="Rate of later volatility-shock exceedances",
        xaxis_title="Trading periods after the main shock",
        yaxis_title="Fraction of exposed events with another exceedance",
    )
    return figure


# Compatibility aliases retained for v3 imports.
def shock_score_figure(diagnostics: pd.DataFrame, events: pd.DataFrame, threshold: float | None) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=diagnostics.index, y=diagnostics["shock_score"], mode="lines"))
    if threshold is not None:
        figure.add_hline(y=threshold)
    return figure


def event_study_figure(aggregate: pd.DataFrame, fits: list[FitResult]) -> go.Figure:
    if "direction" in aggregate:
        return counterfactual_event_figure(aggregate, fits)
    figure = go.Figure()
    if not aggregate.empty:
        figure.add_trace(go.Scatter(x=aggregate["offset"], y=aggregate["median"], mode="lines+markers"))
    return figure


def multi_method_response_figure(
    aggregates: dict[str, pd.DataFrame],
) -> go.Figure:
    """Compare HAR-median, matched-control and local-projection responses by sign."""
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("Negative-price shocks", "Positive-price shocks"),
        horizontal_spacing=0.08,
    )
    method_labels = {
        "har_conditional_median": "HAR conditional median",
        "matched_controls": "Matched controls",
        "local_projection": "Local projection",
    }
    dash = {
        "har_conditional_median": "dot",
        "matched_controls": "dash",
        "local_projection": "solid",
    }
    for column, direction in enumerate(["negative", "positive"], start=1):
        for method, aggregate in aggregates.items():
            if aggregate.empty:
                continue
            subset = aggregate.loc[aggregate["direction"] == direction].sort_values("offset")
            if subset.empty:
                continue
            label = method_labels.get(method, method)
            if method == "local_projection":
                figure.add_trace(
                    go.Scatter(
                        x=subset["offset"],
                        y=subset["bootstrap_ci_high"],
                        mode="lines",
                        line={"width": 0},
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=column,
                )
                figure.add_trace(
                    go.Scatter(
                        x=subset["offset"],
                        y=subset["bootstrap_ci_low"],
                        mode="lines",
                        fill="tonexty",
                        line={"width": 0},
                        name=f"{label} 95% HAC CI" if column == 1 else None,
                        showlegend=column == 1,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=column,
                )
            figure.add_trace(
                go.Scatter(
                    x=subset["offset"],
                    y=subset["median"],
                    mode="lines+markers",
                    line={"dash": dash.get(method, "solid")},
                    name=label if column == 1 else None,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        figure.add_hline(y=0, row=1, col=column)
    figure.update_xaxes(title_text="Trading days after shock", row=1, col=1)
    figure.update_xaxes(title_text="Trading days after shock", row=1, col=2)
    figure.update_yaxes(title_text="Abnormal future variance", row=1, col=1)
    figure.update_layout(
        title="Post-shock response under three counterfactual methods",
        hovermode="x unified",
    )
    return figure


def local_projection_difference_figure(local_projections: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if local_projections.empty:
        return figure
    effect = np.exp(local_projections["difference"].to_numpy(float)) - 1.0
    low = np.exp(local_projections["difference_ci_low"].to_numpy(float)) - 1.0
    high = np.exp(local_projections["difference_ci_high"].to_numpy(float)) - 1.0
    figure.add_trace(
        go.Scatter(
            x=local_projections["offset"],
            y=high,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=local_projections["offset"],
            y=low,
            mode="lines",
            fill="tonexty",
            line={"width": 0},
            name="95% HAC CI",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=local_projections["offset"],
            y=effect,
            mode="lines+markers",
            name="Negative minus positive",
        )
    )
    figure.add_hline(y=0)
    figure.update_layout(
        title="Local-projection contrast: negative versus positive volatility shocks",
        xaxis_title="Trading days after shock",
        yaxis_title="Multiplicative variance contrast",
        hovermode="x unified",
    )
    return figure


def matching_balance_figure(balance: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if balance.empty:
        return figure
    figure.add_trace(
        go.Bar(
            x=balance["feature"],
            y=balance["standardized_mean_difference"],
            name="Standardised mean difference",
        )
    )
    figure.add_hline(y=0.1, line_dash="dash")
    figure.add_hline(y=-0.1, line_dash="dash")
    figure.update_layout(
        title="Matched-control balance on pre-event state variables",
        xaxis_title="Feature",
        yaxis_title="Standardised mean difference",
    )
    return figure


def aftershock_excess_figure(rate_frame: pd.DataFrame, fit) -> go.Figure:
    figure = go.Figure()
    if rate_frame.empty:
        return figure
    figure.add_trace(
        go.Scatter(
            x=rate_frame["offset"],
            y=rate_frame["rate"],
            mode="lines+markers",
            name="Observed aftershock rate",
        )
    )
    baseline = float(rate_frame["baseline_rate"].iloc[0])
    figure.add_hline(y=baseline, annotation_text=f"unconditional rate={baseline:.3f}")
    if fit is not None:
        from asset_lab.analysis.shock_models import baseline_omori_probability

        offsets = rate_frame["offset"].to_numpy(float)
        prediction = baseline_omori_probability(offsets, fit)
        figure.add_trace(
            go.Scatter(
                x=offsets,
                y=prediction,
                mode="lines",
                name=f"baseline + Omori (p={fit.exponent:.2f})",
            )
        )
    figure.update_layout(
        title="Aftershock exceedance rate versus unconditional background",
        xaxis_title="Trading days after main shock",
        yaxis_title="Probability of a later score exceedance",
        hovermode="x unified",
    )
    return figure
