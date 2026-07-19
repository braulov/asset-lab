from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit, gammaln

from asset_lab.analysis.har import (
    EPSILON,
    har_design,
    model_from_diagnostic_row,
    recursive_log_median_path,
)
from asset_lab.analysis.trend import normalized_trend


@dataclass(frozen=True)
class LocalProjectionResult:
    offset: int
    nobs: int
    negative_events: int
    positive_events: int
    negative_estimate: float
    negative_std_error: float
    negative_ci_low: float
    negative_ci_high: float
    negative_p_value: float
    positive_estimate: float
    positive_std_error: float
    positive_ci_low: float
    positive_ci_high: float
    positive_p_value: float
    difference: float
    difference_std_error: float
    difference_ci_low: float
    difference_ci_high: float
    difference_p_value: float
    r_squared: float

    def as_record(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class OmoriComparison:
    baseline_rate: float
    scale: float
    shift: float
    exponent: float
    log_likelihood: float
    aic: float
    flat_rate: float
    flat_log_likelihood: float
    flat_aic: float
    delta_aic_omori_minus_flat: float
    meaningful_decay: bool
    reason: str

    def as_record(self) -> dict[str, float | bool | str]:
        return asdict(self)


def _forward_average_variance(
    variance_proxy: pd.Series,
    *,
    offset: int,
    window: int,
) -> pd.Series:
    if offset < 1 or window < 1:
        raise ValueError("Offset and window must be positive.")
    q = pd.to_numeric(variance_proxy, errors="coerce").where(lambda item: item > 0.0)
    columns = [q.shift(-(offset + step)) for step in range(window)]
    frame = pd.concat(columns, axis=1)
    result = frame.mean(axis=1, skipna=False)
    result.name = f"future_variance_{offset}_{window}"
    return result


def _pre_event_feature_frame(
    variance_proxy: pd.Series,
    returns: pd.Series,
    diagnostics: pd.DataFrame,
    *,
    trend_window: int = 20,
) -> pd.DataFrame:
    q = pd.to_numeric(variance_proxy, errors="coerce").reindex(diagnostics.index)
    r = pd.to_numeric(returns, errors="coerce").reindex(diagnostics.index)
    state = har_design(q)[["log_daily", "log_weekly", "log_monthly"]]
    pretrend = normalized_trend(r.shift(1), trend_window, q.shift(1))
    return pd.DataFrame(
        {
            "forecast_log_variance": diagnostics["forecast_log_variance"],
            "log_daily": state["log_daily"],
            "log_weekly": state["log_weekly"],
            "log_monthly": state["log_monthly"],
            "pretrend": pretrend,
        },
        index=diagnostics.index,
    ).replace([np.inf, -np.inf], np.nan)


def _robust_standardise(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    median = frame.median(axis=0)
    mad = (frame - median).abs().median(axis=0)
    scale = (1.4826 * mad).replace(0.0, np.nan)
    fallback = frame.std(axis=0, ddof=0).replace(0.0, 1.0)
    scale = scale.fillna(fallback).fillna(1.0)
    return (frame - median) / scale, median, scale


def select_matched_controls(
    variance_proxy: pd.Series,
    returns: pd.Series,
    diagnostics: pd.DataFrame,
    events: pd.DataFrame,
    *,
    future_window: int = 5,
    post_days: int = 30,
    controls_per_event: int = 10,
    trend_window: int = 20,
    maximum_position_distance: int = 504,
    control_score_cap: float = 1.0,
    exclusion_radius: int = 10,
) -> pd.DataFrame:
    """Nearest-neighbour no-shock controls using strictly pre-event state variables.

    Controls are matched on the past-only HAR forecast, daily/weekly/monthly state and
    pre-event normalised trend.  Selected shock days and their immediate neighbourhoods
    are excluded.  A date-distance penalty prevents remote regimes from dominating the
    match while still allowing a global fallback when a local pool is too small.
    """
    if events.empty:
        return pd.DataFrame()
    if controls_per_event < 1:
        raise ValueError("At least one control per event is required.")

    q = pd.to_numeric(variance_proxy, errors="coerce").reindex(diagnostics.index)
    features = _pre_event_feature_frame(
        q,
        returns,
        diagnostics,
        trend_window=trend_window,
    )
    finite_features = features.notna().all(axis=1)
    valid_future = pd.Series(False, index=q.index)
    last_start = len(q) - post_days - future_window
    if last_start > 0:
        valid_future.iloc[: last_start + 1] = True

    score = pd.to_numeric(diagnostics["shock_score"], errors="coerce")
    candidate_mask = finite_features & valid_future & score.notna() & (score <= control_score_cap)

    event_positions = events["position"].astype(int).to_numpy()
    contaminated = np.zeros(len(q), dtype=bool)
    for position in event_positions:
        left = max(0, position - exclusion_radius)
        right = min(len(q), position + exclusion_radius + 1)
        contaminated[left:right] = True
    candidate_mask.iloc[contaminated] = False

    pool = features.loc[candidate_mask]
    if len(pool) < controls_per_event:
        return pd.DataFrame()
    standardised, median, scale = _robust_standardise(pool)

    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        event_position = int(event["position"])
        if event_position >= len(q) or not finite_features.iloc[event_position] or not valid_future.iloc[event_position]:
            continue
        event_features = features.iloc[event_position]
        event_z = (event_features - median) / scale

        positions = np.flatnonzero(candidate_mask.to_numpy())
        local = positions[np.abs(positions - event_position) <= maximum_position_distance]
        if len(local) < controls_per_event:
            local = positions
        if len(local) < controls_per_event:
            continue

        candidate_index = q.index[local]
        candidate_z = standardised.reindex(candidate_index)
        feature_distance = np.sqrt(
            np.square(candidate_z - event_z.to_numpy(dtype=float)).sum(axis=1)
        )
        time_penalty = 0.25 * np.abs(local - event_position) / max(maximum_position_distance, 1)
        distance = feature_distance.to_numpy(dtype=float) + time_penalty
        order = np.argsort(distance)[:controls_per_event]

        for rank, selected_index in enumerate(order, start=1):
            control_position = int(local[selected_index])
            record: dict[str, object] = {
                "event_id": int(event["event_id"]),
                "event_timestamp": event["timestamp"],
                "event_position": event_position,
                "direction": event["direction"],
                "control_rank": rank,
                "control_position": control_position,
                "control_timestamp": q.index[control_position],
                "distance": float(distance[selected_index]),
            }
            for column in features.columns:
                record[f"event_{column}"] = float(event_features[column])
                record[f"control_{column}"] = float(features.iloc[control_position][column])
            rows.append(record)
    return pd.DataFrame(rows)


def matching_balance_summary(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()
    features = [
        column.removeprefix("event_")
        for column in matches.columns
        if column.startswith("event_")
        and column not in {"event_id", "event_timestamp", "event_position"}
        and f"control_{column.removeprefix('event_')}" in matches.columns
    ]
    rows: list[dict[str, float | str]] = []
    for feature in features:
        event_values = pd.to_numeric(matches[f"event_{feature}"], errors="coerce")
        control_values = pd.to_numeric(matches[f"control_{feature}"], errors="coerce")
        pooled = np.sqrt((event_values.var(ddof=1) + control_values.var(ddof=1)) / 2.0)
        smd = float((event_values.mean() - control_values.mean()) / pooled) if pooled > 0 else 0.0
        rows.append(
            {
                "feature": feature,
                "event_mean": float(event_values.mean()),
                "control_mean": float(control_values.mean()),
                "standardized_mean_difference": smd,
                "mean_absolute_pair_difference": float(np.mean(np.abs(event_values - control_values))),
            }
        )
    return pd.DataFrame(rows)


def build_matched_control_trajectories(
    variance_proxy: pd.Series,
    events: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    future_window: int = 5,
    post_days: int = 30,
    stride: int = 1,
) -> pd.DataFrame:
    if events.empty or matches.empty:
        return pd.DataFrame()
    q = pd.to_numeric(variance_proxy, errors="coerce")
    rows: list[dict[str, object]] = []
    event_lookup = events.set_index("event_id")

    for event_id, event_matches in matches.groupby("event_id"):
        if event_id not in event_lookup.index:
            continue
        event = event_lookup.loc[event_id]
        event_position = int(event["position"])
        control_positions = event_matches["control_position"].astype(int).to_numpy()
        event_rows: list[dict[str, object]] = []
        complete = True
        for offset in range(1, post_days + 1, stride):
            actual = q.iloc[event_position + offset : event_position + offset + future_window].to_numpy(float)
            if len(actual) != future_window or not np.isfinite(actual).all() or np.any(actual <= 0.0):
                complete = False
                break
            control_windows: list[float] = []
            for control_position in control_positions:
                values = q.iloc[
                    control_position + offset : control_position + offset + future_window
                ].to_numpy(float)
                if len(values) != future_window or not np.isfinite(values).all() or np.any(values <= 0.0):
                    continue
                control_windows.append(float(np.mean(values)))
            if len(control_windows) < max(3, len(control_positions) // 2):
                complete = False
                break
            actual_variance = float(np.mean(actual))
            # A geometric-median reference is resistant to one volatile control path.
            control_reference = float(np.exp(np.median(np.log(control_windows))))
            abnormal_log = float(np.log(actual_variance / control_reference))
            event_rows.append(
                {
                    "event_id": int(event_id),
                    "event_timestamp": event["timestamp"],
                    "direction": event["direction"],
                    "shock_score": float(event["shock_score"]),
                    "offset": offset,
                    "actual_variance": actual_variance,
                    "counterfactual_variance": control_reference,
                    "abnormal_log_variance": abnormal_log,
                    "abnormal_excess_variance": float(np.exp(abnormal_log) - 1.0),
                    "method": "matched_controls",
                    "control_count": len(control_windows),
                }
            )
        if complete:
            rows.extend(event_rows)
    return pd.DataFrame(rows)


def build_har_median_counterfactual_trajectories(
    variance_proxy: pd.Series,
    diagnostics: pd.DataFrame,
    events: pd.DataFrame,
    *,
    future_window: int = 5,
    post_days: int = 30,
    stride: int = 1,
) -> pd.DataFrame:
    """No-shock HAR conditional-median path with no recursive smearing correction."""
    if events.empty:
        return pd.DataFrame()
    q = pd.to_numeric(variance_proxy, errors="coerce").reindex(diagnostics.index)
    rows: list[dict[str, object]] = []
    required_periods = post_days + future_window + 1

    for _, event in events.iterrows():
        position = int(event["position"])
        if position < 22 or position + post_days + future_window >= len(q):
            continue
        model = model_from_diagnostic_row(diagnostics.iloc[position])
        if model is None:
            continue
        path = recursive_log_median_path(q.iloc[:position], model, required_periods)
        if len(path) < required_periods:
            continue
        median_variance = path["forecast_median_variance"].to_numpy(float)

        event_rows: list[dict[str, object]] = []
        complete = True
        for offset in range(1, post_days + 1, stride):
            actual = q.iloc[position + offset : position + offset + future_window].to_numpy(float)
            expected = median_variance[offset : offset + future_window]
            if (
                len(actual) != future_window
                or len(expected) != future_window
                or not np.isfinite(actual).all()
                or not np.isfinite(expected).all()
                or np.any(actual <= 0.0)
                or np.any(expected <= 0.0)
            ):
                complete = False
                break
            actual_variance = float(np.mean(actual))
            counterfactual_variance = float(np.mean(expected))
            abnormal_log = float(np.log(actual_variance / counterfactual_variance))
            event_rows.append(
                {
                    "event_id": int(event["event_id"]),
                    "event_timestamp": event["timestamp"],
                    "direction": event["direction"],
                    "shock_score": float(event["shock_score"]),
                    "offset": offset,
                    "actual_variance": actual_variance,
                    "counterfactual_variance": counterfactual_variance,
                    "abnormal_log_variance": abnormal_log,
                    "abnormal_excess_variance": float(np.exp(abnormal_log) - 1.0),
                    "method": "har_conditional_median",
                }
            )
        if complete:
            rows.extend(event_rows)
    return pd.DataFrame(rows)


def _event_indicator_frame(index: pd.Index, events: pd.DataFrame) -> pd.DataFrame:
    indicators = pd.DataFrame(
        {"negative_event": 0.0, "positive_event": 0.0},
        index=index,
    )
    for _, event in events.iterrows():
        position = int(event["position"])
        direction = str(event["direction"])
        column = f"{direction}_event"
        if 0 <= position < len(indicators) and column in indicators.columns:
            indicators.iloc[position, indicators.columns.get_loc(column)] = 1.0
    return indicators


def fit_local_projection_responses(
    variance_proxy: pd.Series,
    returns: pd.Series,
    diagnostics: pd.DataFrame,
    events: pd.DataFrame,
    *,
    future_window: int = 5,
    post_days: int = 30,
    stride: int = 1,
    trend_window: int = 20,
    contamination_radius: int = 10,
    include_year_fixed_effects: bool = True,
    minimum_observations: int = 400,
) -> pd.DataFrame:
    """Local projections of future log variance on negative/positive shock indicators.

    The controls are known before the event day.  Cluster-neighbour days and unselected
    threshold exceedances are excluded from the control sample.  HAC lags grow with the
    response horizon and the forward averaging window.
    """
    if events.empty:
        return pd.DataFrame()
    q = pd.to_numeric(variance_proxy, errors="coerce").reindex(diagnostics.index)
    features = _pre_event_feature_frame(
        q,
        returns,
        diagnostics,
        trend_window=trend_window,
    )
    indicators = _event_indicator_frame(q.index, events)
    threshold = float(events["threshold"].iloc[0])
    selected_positions = set(events["position"].astype(int).tolist())

    contaminated = np.zeros(len(q), dtype=bool)
    for position in selected_positions:
        left = max(0, position - contamination_radius)
        right = min(len(q), position + contamination_radius + 1)
        contaminated[left:right] = True
        contaminated[position] = False
    unselected_exceedance = (
        pd.to_numeric(diagnostics["shock_score"], errors="coerce").to_numpy(float) >= threshold
    )
    for position in selected_positions:
        unselected_exceedance[position] = False

    year_dummies = pd.DataFrame(index=q.index)
    if include_year_fixed_effects and isinstance(q.index, pd.DatetimeIndex):
        year_dummies = pd.get_dummies(q.index.year, prefix="year", drop_first=True, dtype=float)
        year_dummies.index = q.index

    rows: list[dict[str, float | int]] = []
    for offset in range(1, post_days + 1, stride):
        target = np.log(_forward_average_variance(q, offset=offset, window=future_window) + EPSILON)
        work = pd.concat([target.rename("target"), indicators, features, year_dummies], axis=1)
        work = work.replace([np.inf, -np.inf], np.nan)
        eligible = ~pd.Series(contaminated | unselected_exceedance, index=q.index)
        selected_list = sorted(position for position in selected_positions if 0 <= position < len(q))
        if selected_list:
            eligible.iloc[selected_list] = True
        work = work.loc[eligible].dropna()
        if len(work) < minimum_observations:
            continue
        if work["negative_event"].sum() < 8 or work["positive_event"].sum() < 8:
            continue

        columns = [
            "negative_event",
            "positive_event",
            "forecast_log_variance",
            "log_daily",
            "log_weekly",
            "log_monthly",
            "pretrend",
            *year_dummies.columns.tolist(),
        ]
        x = sm.add_constant(work[columns], has_constant="add")
        hac_lags = max(future_window - 1, offset + future_window - 2, 1)
        fitted = sm.OLS(work["target"], x).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": int(hac_lags), "use_correction": True},
        )
        confidence = fitted.conf_int(alpha=0.05)

        order = list(fitted.params.index)
        contrast = np.zeros(len(order), dtype=float)
        contrast[order.index("negative_event")] = 1.0
        contrast[order.index("positive_event")] = -1.0
        test = fitted.t_test(contrast)
        difference_ci = np.asarray(test.conf_int(alpha=0.05)).reshape(-1, 2)[0]

        rows.append(
            LocalProjectionResult(
                offset=offset,
                nobs=int(fitted.nobs),
                negative_events=int(work["negative_event"].sum()),
                positive_events=int(work["positive_event"].sum()),
                negative_estimate=float(fitted.params["negative_event"]),
                negative_std_error=float(fitted.bse["negative_event"]),
                negative_ci_low=float(confidence.loc["negative_event", 0]),
                negative_ci_high=float(confidence.loc["negative_event", 1]),
                negative_p_value=float(fitted.pvalues["negative_event"]),
                positive_estimate=float(fitted.params["positive_event"]),
                positive_std_error=float(fitted.bse["positive_event"]),
                positive_ci_low=float(confidence.loc["positive_event", 0]),
                positive_ci_high=float(confidence.loc["positive_event", 1]),
                positive_p_value=float(fitted.pvalues["positive_event"]),
                difference=float(np.asarray(test.effect).reshape(-1)[0]),
                difference_std_error=float(np.asarray(test.sd).reshape(-1)[0]),
                difference_ci_low=float(difference_ci[0]),
                difference_ci_high=float(difference_ci[1]),
                difference_p_value=float(np.asarray(test.pvalue).reshape(-1)[0]),
                r_squared=float(fitted.rsquared),
            ).as_record()
        )
    return pd.DataFrame(rows)


def local_projection_aggregate(local_projections: pd.DataFrame) -> pd.DataFrame:
    if local_projections.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    for direction in ["negative", "positive"]:
        estimate = local_projections[f"{direction}_estimate"].to_numpy(float)
        low = local_projections[f"{direction}_ci_low"].to_numpy(float)
        high = local_projections[f"{direction}_ci_high"].to_numpy(float)
        # Convert log-variance effects to proportional variance effects.
        effect = np.exp(estimate) - 1.0
        effect_low = np.exp(low) - 1.0
        effect_high = np.exp(high) - 1.0
        event_count = local_projections[f"{direction}_events"].to_numpy(int)
        for idx, offset in enumerate(local_projections["offset"].to_numpy(int)):
            rows.append(
                {
                    "direction": direction,
                    "offset": int(offset),
                    "mean": float(effect[idx]),
                    "median": float(effect[idx]),
                    "q25": float(effect_low[idx]),
                    "q75": float(effect_high[idx]),
                    "bootstrap_ci_low": float(effect_low[idx]),
                    "bootstrap_ci_high": float(effect_high[idx]),
                    "count": int(event_count[idx]),
                    "method": "local_projection",
                }
            )
    return pd.DataFrame(rows)


def method_early_response_summary(
    aggregates: dict[str, pd.DataFrame],
    *,
    early_end: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    for method, aggregate in aggregates.items():
        if aggregate.empty:
            continue
        for direction in ["negative", "positive"]:
            subset = aggregate.loc[
                (aggregate["direction"] == direction)
                & (aggregate["offset"] <= early_end)
            ].sort_values("offset")
            if subset.empty:
                continue
            estimate = float(subset["median"].mean())
            # Conservative early-support rule: at least one early point has CI above zero.
            supported = bool((subset["bootstrap_ci_low"] > 0.0).any())
            rows.append(
                {
                    "method": method,
                    "direction": direction,
                    "early_mean_excess": estimate,
                    "positive_support": supported,
                    "minimum_events": int(subset["count"].min()),
                }
            )
    return pd.DataFrame(rows)


def consensus_decision(
    summaries: pd.DataFrame,
    *,
    direction: str = "negative",
) -> tuple[bool, str]:
    subset = summaries.loc[summaries["direction"] == direction]
    if subset.empty:
        return False, "No response estimates are available."
    positive_methods = int((subset["early_mean_excess"] > 0.0).sum())
    supported_methods = int(subset["positive_support"].sum())
    local = subset.loc[subset["method"] == "local_projection"]
    local_positive = not local.empty and float(local["early_mean_excess"].iloc[0]) > 0.0
    if local_positive and positive_methods >= 2 and supported_methods >= 1:
        return True, (
            f"{positive_methods} of {len(subset)} methods show a positive early response; "
            f"{supported_methods} have interval support and the local projection agrees."
        )
    return False, (
        f"Only {positive_methods} of {len(subset)} methods show a positive early response; "
        f"{supported_methods} have interval support. Cross-method identification is insufficient."
    )


def aftershock_excess_rate(
    diagnostics: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon: int = 30,
    aftershock_quantile: float = 0.90,
) -> pd.DataFrame:
    if events.empty or horizon < 1:
        return pd.DataFrame()
    scores = pd.to_numeric(diagnostics["shock_score"], errors="coerce")
    finite = scores.dropna()
    if finite.empty:
        return pd.DataFrame()
    threshold = float(finite.quantile(aftershock_quantile))
    baseline_rate = float((finite >= threshold).mean())
    values = scores.to_numpy(float)
    rows: list[dict[str, float | int]] = []
    for offset in range(1, horizon + 1):
        indicators: list[float] = []
        for position in events["position"].astype(int):
            target = position + offset
            if target >= len(values) or not np.isfinite(values[target]):
                continue
            indicators.append(float(values[target] >= threshold))
        if not indicators:
            continue
        observed = float(np.mean(indicators))
        rows.append(
            {
                "offset": offset,
                "rate": observed,
                "count": int(np.sum(indicators)),
                "exposed_events": len(indicators),
                "baseline_rate": baseline_rate,
                "excess_rate": observed - baseline_rate,
                "rate_ratio": observed / baseline_rate if baseline_rate > 0 else np.nan,
                "aftershock_threshold": threshold,
            }
        )
    return pd.DataFrame(rows)


def _binomial_log_likelihood(count: np.ndarray, exposure: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return float(
        np.sum(
            gammaln(exposure + 1)
            - gammaln(count + 1)
            - gammaln(exposure - count + 1)
            + count * np.log(probability)
            + (exposure - count) * np.log1p(-probability)
        )
    )


def fit_baseline_omori(rate_frame: pd.DataFrame) -> OmoriComparison | None:
    if rate_frame.empty or len(rate_frame) < 6:
        return None
    work = rate_frame.dropna(subset=["rate", "count", "exposed_events", "baseline_rate"])
    if len(work) < 6:
        return None
    t = work["offset"].to_numpy(float)
    count = work["count"].to_numpy(float)
    exposure = work["exposed_events"].to_numpy(float)
    baseline = float(work["baseline_rate"].iloc[0])

    def unpack(theta: np.ndarray) -> tuple[float, float, float]:
        # Smooth transforms keep probability valid and parameters positive.
        scale = (1.0 - baseline - 1e-6) * expit(theta[0])
        shift = float(np.exp(np.clip(theta[1], -5.0, 5.0)))
        exponent = float(0.05 + 4.95 * expit(theta[2]))
        return float(scale), shift, exponent

    def objective(theta: np.ndarray) -> float:
        scale, shift, exponent = unpack(theta)
        probability = baseline + scale / np.power(t + shift, exponent)
        return -_binomial_log_likelihood(count, exposure, probability)

    result = minimize(objective, np.array([-2.0, 0.0, -1.0]), method="BFGS")
    scale, shift, exponent = unpack(result.x)
    probability = baseline + scale / np.power(t + shift, exponent)
    log_likelihood = _binomial_log_likelihood(count, exposure, probability)
    aic = float(2 * 3 - 2 * log_likelihood)

    flat_rate = float(np.sum(count) / np.sum(exposure))
    flat_probability = np.full_like(t, flat_rate)
    flat_log_likelihood = _binomial_log_likelihood(count, exposure, flat_probability)
    flat_aic = float(2 * 1 - 2 * flat_log_likelihood)
    delta = aic - flat_aic
    meaningful = bool(delta < -2.0 and exponent > 0.1 and scale > 0.005)
    if meaningful:
        reason = "Baseline-plus-Omori decay improves AIC over a flat aftershock rate."
    elif delta >= -2.0:
        reason = "The decay model does not improve AIC by at least 2 over a flat rate."
    elif exponent <= 0.1:
        reason = "The fitted exponent is too close to zero to represent meaningful decay."
    else:
        reason = "The fitted excess aftershock amplitude is negligible."
    return OmoriComparison(
        baseline_rate=baseline,
        scale=scale,
        shift=shift,
        exponent=exponent,
        log_likelihood=log_likelihood,
        aic=aic,
        flat_rate=flat_rate,
        flat_log_likelihood=flat_log_likelihood,
        flat_aic=flat_aic,
        delta_aic_omori_minus_flat=delta,
        meaningful_decay=meaningful,
        reason=reason,
    )


def baseline_omori_probability(
    offsets: np.ndarray,
    fit: OmoriComparison,
) -> np.ndarray:
    t = np.asarray(offsets, dtype=float)
    return fit.baseline_rate + fit.scale / np.power(t + fit.shift, fit.exponent)
