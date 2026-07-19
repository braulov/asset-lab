from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from asset_lab.analysis.har import model_from_diagnostic_row, recursive_no_shock_path


@dataclass(frozen=True)
class HorizonComparison:
    horizon: str
    negative_median: float
    positive_median: float
    difference: float
    difference_ci_low: float
    difference_ci_high: float
    p_value: float
    negative_events: int
    positive_events: int

    def as_record(self) -> dict[str, float | int | str]:
        return {
            "horizon": self.horizon,
            "negative_median_excess": self.negative_median,
            "positive_median_excess": self.positive_median,
            "negative_minus_positive": self.difference,
            "difference_ci_low": self.difference_ci_low,
            "difference_ci_high": self.difference_ci_high,
            "bootstrap_p_value": self.p_value,
            "negative_events": self.negative_events,
            "positive_events": self.positive_events,
        }


def detect_volatility_shocks(
    diagnostics: pd.DataFrame,
    returns: pd.Series,
    *,
    quantile: float = 0.95,
    cooldown: int = 10,
) -> pd.DataFrame:
    """Detect positive HAR log-variance innovations, then label their price direction.

    The threshold is computed once from all shock scores.  Direction filtering must be
    performed only after this function, ensuring identical extremeness thresholds for
    negative and positive price moves.
    """
    if not 0.5 < quantile < 1.0:
        raise ValueError("Shock quantile must lie between 0.5 and 1.")
    if cooldown < 0:
        raise ValueError("Cooldown cannot be negative.")
    if "shock_score" not in diagnostics:
        raise ValueError("HAR diagnostics must contain shock_score.")

    scores = pd.to_numeric(diagnostics["shock_score"], errors="coerce")
    finite = scores.dropna()
    if finite.empty:
        return pd.DataFrame()
    threshold = float(finite.quantile(quantile))
    candidate_positions = np.flatnonzero(scores.to_numpy(dtype=float) >= threshold)
    if len(candidate_positions) == 0:
        return pd.DataFrame()

    score_array = scores.to_numpy(dtype=float)
    selected: list[int] = []
    for position in sorted(candidate_positions, key=lambda pos: score_array[pos], reverse=True):
        if all(abs(int(position) - previous) > cooldown for previous in selected):
            selected.append(int(position))
    selected.sort()

    returns_aligned = pd.to_numeric(returns, errors="coerce").reindex(diagnostics.index)
    rows: list[dict[str, object]] = []
    for event_id, position in enumerate(selected):
        price_return = float(returns_aligned.iloc[position])
        direction = "negative" if price_return < 0.0 else "positive" if price_return > 0.0 else "zero"
        overnight = diagnostics.iloc[position].get("overnight_gap_squared", np.nan)
        intraday = diagnostics.iloc[position].get("intraday_rs", np.nan)
        total_component = float(overnight + intraday) if np.isfinite(overnight) and np.isfinite(intraday) else np.nan
        gap_share = float(overnight / total_component) if np.isfinite(total_component) and total_component > 0 else np.nan
        rows.append(
            {
                "event_id": event_id,
                "position": position,
                "timestamp": diagnostics.index[position],
                "return": price_return,
                "direction": direction,
                "shock_score": float(score_array[position]),
                "innovation": float(diagnostics.iloc[position]["innovation"]),
                "variance_proxy": float(diagnostics.iloc[position]["variance_proxy"]),
                "forecast_variance": float(diagnostics.iloc[position]["forecast_variance"]),
                "threshold": threshold,
                "gap_share": gap_share,
                "source": (
                    "overnight-dominated"
                    if np.isfinite(gap_share) and gap_share >= 0.67
                    else "intraday-dominated"
                    if np.isfinite(gap_share) and gap_share <= 0.33
                    else "mixed"
                ),
            }
        )
    return pd.DataFrame(rows)


def filter_events(events: pd.DataFrame, direction: str) -> pd.DataFrame:
    if events.empty or direction == "all":
        return events.copy()
    if direction not in {"negative", "positive", "zero"}:
        raise ValueError("Unknown event direction.")
    return events.loc[events["direction"] == direction].reset_index(drop=True)


def build_counterfactual_trajectories(
    variance_proxy: pd.Series,
    diagnostics: pd.DataFrame,
    events: pd.DataFrame,
    *,
    future_window: int = 5,
    post_days: int = 30,
    stride: int = 1,
) -> pd.DataFrame:
    """Compare actual future variance with a recursively forecast no-shock HAR path."""
    if future_window < 1 or post_days < 1 or stride < 1:
        raise ValueError("Future window, post horizon and stride must be positive.")
    if events.empty:
        return pd.DataFrame()

    q = pd.to_numeric(variance_proxy, errors="coerce").reindex(diagnostics.index)
    rows: list[dict[str, object]] = []
    maximum_offset = post_days
    required_prediction_periods = post_days + future_window + 1

    for _, event in events.iterrows():
        position = int(event["position"])
        if position < 22 or position + post_days + future_window >= len(q):
            continue
        model = model_from_diagnostic_row(diagnostics.iloc[position])
        if model is None:
            continue
        history = q.iloc[:position]
        counterfactual = recursive_no_shock_path(
            history,
            model,
            periods=required_prediction_periods,
        )
        if len(counterfactual) < required_prediction_periods:
            continue

        event_rows: list[dict[str, object]] = []
        complete = True
        for offset in range(1, maximum_offset + 1, stride):
            actual_slice = q.iloc[
                position + offset : position + offset + future_window
            ].to_numpy(dtype=float)
            # counterfactual[0] is the event day, so offset k starts at index k.
            counterfactual_slice = counterfactual[offset : offset + future_window]
            if (
                len(actual_slice) != future_window
                or len(counterfactual_slice) != future_window
                or not np.isfinite(actual_slice).all()
                or not np.isfinite(counterfactual_slice).all()
            ):
                complete = False
                break
            actual_variance = float(np.mean(actual_slice))
            expected_variance = float(np.mean(counterfactual_slice))
            if actual_variance <= 0.0 or expected_variance <= 0.0:
                complete = False
                break
            abnormal_log = float(np.log(actual_variance / expected_variance))
            event_rows.append(
                {
                    "event_id": int(event["event_id"]),
                    "event_timestamp": event["timestamp"],
                    "direction": event["direction"],
                    "shock_score": float(event["shock_score"]),
                    "offset": offset,
                    "actual_variance": actual_variance,
                    "counterfactual_variance": expected_variance,
                    "abnormal_log_variance": abnormal_log,
                    "abnormal_excess_variance": float(np.exp(abnormal_log) - 1.0),
                }
            )
        if complete:
            rows.extend(event_rows)
    return pd.DataFrame(rows)


def aggregate_counterfactual_trajectories(
    trajectories: pd.DataFrame,
    *,
    bootstrap_samples: int = 500,
    random_seed: int = 1729,
) -> pd.DataFrame:
    """Aggregate by direction and bootstrap complete events, not individual offsets."""
    if trajectories.empty:
        return pd.DataFrame()
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples.")

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, float | int | str]] = []
    for direction, group in trajectories.groupby("direction"):
        pivot = group.pivot(index="event_id", columns="offset", values="abnormal_excess_variance")
        event_count = len(pivot)
        if event_count == 0:
            continue
        matrix = pivot.to_numpy(dtype=float)
        bootstrap_medians = np.empty((bootstrap_samples, matrix.shape[1]), dtype=float)
        for sample in range(bootstrap_samples):
            indices = rng.integers(0, event_count, size=event_count)
            bootstrap_medians[sample] = np.nanmedian(matrix[indices], axis=0)

        for column_position, offset in enumerate(pivot.columns):
            values = matrix[:, column_position]
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                continue
            rows.append(
                {
                    "direction": str(direction),
                    "offset": int(offset),
                    "mean": float(np.mean(finite)),
                    "median": float(np.median(finite)),
                    "q25": float(np.quantile(finite, 0.25)),
                    "q75": float(np.quantile(finite, 0.75)),
                    "bootstrap_ci_low": float(np.nanquantile(bootstrap_medians[:, column_position], 0.025)),
                    "bootstrap_ci_high": float(np.nanquantile(bootstrap_medians[:, column_position], 0.975)),
                    "count": int(len(finite)),
                }
            )
    return pd.DataFrame(rows).sort_values(["direction", "offset"]).reset_index(drop=True)


def _event_horizon_values(
    trajectories: pd.DataFrame,
    direction: str,
    start: int,
    end: int,
) -> np.ndarray:
    subset = trajectories.loc[
        (trajectories["direction"] == direction)
        & (trajectories["offset"] >= start)
        & (trajectories["offset"] <= end)
    ]
    if subset.empty:
        return np.asarray([], dtype=float)
    per_event_log = subset.groupby("event_id")["abnormal_log_variance"].mean()
    return np.exp(per_event_log.to_numpy(dtype=float)) - 1.0


def horizon_comparisons(
    trajectories: pd.DataFrame,
    *,
    horizons: tuple[tuple[str, int, int], ...] = (
        ("days 1–5", 1, 5),
        ("days 6–10", 6, 10),
        ("days 11–20", 11, 20),
    ),
    bootstrap_samples: int = 2_000,
    random_seed: int = 31415,
) -> list[HorizonComparison]:
    if trajectories.empty:
        return []
    rng = np.random.default_rng(random_seed)
    results: list[HorizonComparison] = []
    for label, start, end in horizons:
        negative = _event_horizon_values(trajectories, "negative", start, end)
        positive = _event_horizon_values(trajectories, "positive", start, end)
        if len(negative) < 3 or len(positive) < 3:
            continue
        differences = np.empty(bootstrap_samples, dtype=float)
        for sample in range(bootstrap_samples):
            negative_sample = rng.choice(negative, size=len(negative), replace=True)
            positive_sample = rng.choice(positive, size=len(positive), replace=True)
            differences[sample] = np.median(negative_sample) - np.median(positive_sample)
        observed = float(np.median(negative) - np.median(positive))
        lower = float(np.quantile(differences, 0.025))
        upper = float(np.quantile(differences, 0.975))
        probability_nonpositive = float(np.mean(differences <= 0.0))
        probability_nonnegative = float(np.mean(differences >= 0.0))
        p_value = min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))
        results.append(
            HorizonComparison(
                horizon=label,
                negative_median=float(np.median(negative)),
                positive_median=float(np.median(positive)),
                difference=observed,
                difference_ci_low=lower,
                difference_ci_high=upper,
                p_value=p_value,
                negative_events=len(negative),
                positive_events=len(positive),
            )
        )
    return results


def aftershock_rate(
    diagnostics: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon: int = 30,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Average rate of later HAR-score exceedances after each selected main shock."""
    if events.empty or horizon < 1:
        return pd.DataFrame()
    scores = pd.to_numeric(diagnostics["shock_score"], errors="coerce").to_numpy(dtype=float)
    if threshold is None:
        threshold = float(events["threshold"].iloc[0])
    rows: list[dict[str, float | int]] = []
    for offset in range(1, horizon + 1):
        indicators: list[float] = []
        for position in events["position"].astype(int):
            target = position + offset
            if target >= len(scores) or not np.isfinite(scores[target]):
                continue
            indicators.append(float(scores[target] >= threshold))
        if indicators:
            count = int(np.sum(indicators))
            rows.append(
                {
                    "offset": offset,
                    "rate": float(np.mean(indicators)),
                    "count": count,
                    "exposed_events": len(indicators),
                }
            )
    return pd.DataFrame(rows)


# Legacy compatibility helpers from v3.
def return_shock_score(returns: pd.Series, lookback: int = 60) -> pd.DataFrame:
    values = pd.to_numeric(returns, errors="coerce")
    past_sigma = values.shift(1).rolling(lookback, min_periods=lookback).std(ddof=0)
    return pd.DataFrame(
        {
            "return": values,
            "abs_return": values.abs(),
            "past_sigma": past_sigma,
            "shock_score": values.abs() / past_sigma.where(past_sigma > 0),
        },
        index=values.index,
    )


def detect_return_shocks(
    returns: pd.Series,
    lookback: int = 60,
    quantile: float = 0.95,
    cooldown: int = 10,
    direction: str = "all",
) -> pd.DataFrame:
    diagnostics = return_shock_score(returns, lookback)
    threshold = float(diagnostics["shock_score"].dropna().quantile(quantile))
    candidates = np.flatnonzero(diagnostics["shock_score"].to_numpy(dtype=float) >= threshold)
    selected: list[int] = []
    for position in sorted(candidates, key=lambda p: diagnostics["shock_score"].iloc[p], reverse=True):
        if all(abs(int(position) - previous) > cooldown for previous in selected):
            selected.append(int(position))
    rows = []
    for event_id, position in enumerate(sorted(selected)):
        value = float(returns.iloc[position])
        event_direction = "negative" if value < 0 else "positive"
        if direction != "all" and event_direction != direction:
            continue
        rows.append(
            {
                "event_id": event_id,
                "position": position,
                "timestamp": returns.index[position],
                "return": value,
                "abs_return": abs(value),
                "past_sigma": float(diagnostics["past_sigma"].iloc[position]),
                "shock_score": float(diagnostics["shock_score"].iloc[position]),
                "direction": event_direction,
                "threshold": threshold,
            }
        )
    return pd.DataFrame(rows)


def build_forward_volatility_trajectories(*args, **kwargs) -> pd.DataFrame:
    return pd.DataFrame()


def aggregate_forward_trajectories(trajectories: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame()


def detect_volatility_peaks(
    volatility: pd.Series,
    quantile: float = 0.95,
    cooldown: int = 10,
) -> pd.DataFrame:
    values = pd.to_numeric(volatility, errors="coerce").dropna()
    if values.empty:
        return pd.DataFrame(columns=["position", "timestamp", "volatility", "threshold"])
    threshold = float(values.quantile(quantile))
    raw = pd.to_numeric(volatility, errors="coerce").to_numpy(dtype=float)
    candidates = np.flatnonzero(raw >= threshold)
    selected: list[int] = []
    for position in sorted(candidates, key=lambda item: raw[item], reverse=True):
        if all(abs(int(position) - previous) > cooldown for previous in selected):
            selected.append(int(position))
    return pd.DataFrame(
        [
            {
                "position": position,
                "timestamp": volatility.index[position],
                "volatility": float(raw[position]),
                "threshold": threshold,
            }
            for position in sorted(selected)
        ]
    )


def build_event_trajectories(
    volatility: pd.Series,
    events: pd.DataFrame,
    pre_days: int = 20,
    post_days: int = 60,
) -> pd.DataFrame:
    values = pd.to_numeric(volatility, errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for event_number, event in events.reset_index(drop=True).iterrows():
        position = int(event["position"])
        if position - pre_days < 0 or position + post_days >= len(values):
            continue
        baseline = float(np.nanmedian(values[position - pre_days : position]))
        peak_excess = float(values[position] - baseline)
        if not np.isfinite(peak_excess) or peak_excess <= 0:
            continue
        for offset in range(-pre_days, post_days + 1):
            observed = float(values[position + offset])
            rows.append(
                {
                    "event_id": event_number,
                    "event_timestamp": event["timestamp"],
                    "offset": offset,
                    "volatility": observed,
                    "baseline": baseline,
                    "normalized_excess": (observed - baseline) / peak_excess,
                }
            )
    return pd.DataFrame(rows)


def aggregate_trajectories(trajectories: pd.DataFrame) -> pd.DataFrame:
    if trajectories.empty:
        return pd.DataFrame()
    return (
        trajectories.groupby("offset", as_index=False)
        .agg(
            mean=("normalized_excess", "mean"),
            median=("normalized_excess", "median"),
            q25=("normalized_excess", lambda x: x.quantile(0.25)),
            q75=("normalized_excess", lambda x: x.quantile(0.75)),
            count=("normalized_excess", "count"),
        )
        .sort_values("offset")
        .reset_index(drop=True)
    )


def build_forward_volatility_trajectories(
    returns: pd.Series,
    events: pd.DataFrame,
    baseline_window: int = 60,
    future_window: int = 5,
    post_days: int = 60,
    stride: int = 1,
    annualization: float = 252.0,
) -> pd.DataFrame:
    values = pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for event_number, event in events.reset_index(drop=True).iterrows():
        position = int(event["position"])
        if position - baseline_window < 0 or position + post_days + future_window >= len(values):
            continue
        baseline_slice = values[position - baseline_window : position]
        if not np.isfinite(baseline_slice).all():
            continue
        baseline = float(np.sqrt(annualization * np.mean(baseline_slice**2)))
        if baseline <= 0:
            continue
        for offset in range(0, post_days + 1, stride):
            future = values[position + 1 + offset : position + 1 + offset + future_window]
            if len(future) != future_window or not np.isfinite(future).all():
                continue
            future_volatility = float(np.sqrt(annualization * np.mean(future**2)))
            rows.append(
                {
                    "event_id": event_number,
                    "event_timestamp": event["timestamp"],
                    "direction": event.get("direction", "unknown"),
                    "offset": offset,
                    "future_volatility": future_volatility,
                    "baseline": baseline,
                    "excess_ratio": future_volatility / baseline - 1.0,
                }
            )
    return pd.DataFrame(rows)


def aggregate_forward_trajectories(trajectories: pd.DataFrame) -> pd.DataFrame:
    if trajectories.empty:
        return pd.DataFrame()
    return (
        trajectories.groupby("offset", as_index=False)
        .agg(
            mean=("excess_ratio", "mean"),
            median=("excess_ratio", "median"),
            q25=("excess_ratio", lambda x: x.quantile(0.25)),
            q75=("excess_ratio", lambda x: x.quantile(0.75)),
            count=("excess_ratio", "count"),
        )
        .sort_values("offset")
        .reset_index(drop=True)
    )
