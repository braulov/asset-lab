from __future__ import annotations

import numpy as np
import pandas as pd

from asset_lab.analysis.har import current_har_state


EPSILON = 1e-12


def cumulative_trend(returns: pd.Series, window: int) -> pd.Series:
    result = pd.to_numeric(returns, errors="coerce").rolling(window, min_periods=window).sum()
    result.name = "cumulative_trend"
    return result


def normalized_trend(
    returns: pd.Series,
    window: int,
    variance_proxy: pd.Series | None = None,
) -> pd.Series:
    """Signed cumulative return normalised by observed quadratic variation."""
    values = pd.to_numeric(returns, errors="coerce")
    total_return = values.rolling(window, min_periods=window).sum()
    if variance_proxy is None:
        denominator_squared = values.pow(2).rolling(window, min_periods=window).sum()
    else:
        q = pd.to_numeric(variance_proxy, errors="coerce").clip(lower=0.0)
        denominator_squared = q.rolling(window, min_periods=window).sum()
    result = total_return / np.sqrt(denominator_squared.where(denominator_squared > 0.0))
    result.name = "normalized_trend"
    return result


def future_average_variance(variance_proxy: pd.Series, horizon: int) -> pd.Series:
    if horizon < 1:
        raise ValueError("Future variance horizon must be positive.")
    q = pd.to_numeric(variance_proxy, errors="coerce").where(lambda item: item >= 0.0)
    result = q.rolling(horizon, min_periods=horizon).mean().shift(-horizon)
    result.name = "future_variance"
    return result


def build_trend_variance_frame(
    returns: pd.Series,
    variance_proxy: pd.Series,
    trend_window: int,
    future_horizon: int,
) -> pd.DataFrame:
    state = current_har_state(variance_proxy)
    frame = pd.concat(
        [
            normalized_trend(returns, trend_window, variance_proxy),
            state,
            future_average_variance(variance_proxy, future_horizon),
        ],
        axis=1,
    )
    frame["future_volatility"] = np.sqrt(252.0 * frame["future_variance"].clip(lower=0.0))
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def build_trend_volatility_frame(
    returns: pd.Series,
    trend_window: int,
    future_horizon: int,
    current_volatility_window: int | None = None,
    annualization: float = 252.0,
) -> pd.DataFrame:
    """Compatibility wrapper for earlier tests and clients."""
    values = pd.to_numeric(returns, errors="coerce")
    variance = values.pow(2)
    frame = build_trend_variance_frame(values, variance, trend_window, future_horizon)
    current_window = current_volatility_window or trend_window
    frame["current_volatility"] = np.sqrt(
        annualization
        * values.pow(2).rolling(current_window, min_periods=current_window).mean()
    )
    return frame.dropna()


def correlation_summary(frame: pd.DataFrame) -> dict[str, float]:
    if len(frame) < 3:
        return {"pearson": float("nan"), "spearman": float("nan")}
    target = "future_volatility" if "future_volatility" in frame else "future_variance"
    return {
        "pearson": float(frame["normalized_trend"].corr(frame[target])),
        "spearman": float(frame["normalized_trend"].corr(frame[target], method="spearman")),
    }


def decile_summary(frame: pd.DataFrame, groups: int = 10) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    target = "future_volatility" if "future_volatility" in frame else "future_variance"
    unique = frame["normalized_trend"].nunique()
    actual_groups = min(groups, unique)
    if actual_groups < 2:
        return pd.DataFrame()

    work = frame.copy()
    work["trend_bucket"] = pd.qcut(work["normalized_trend"], q=actual_groups, duplicates="drop")
    summary = (
        work.groupby("trend_bucket", observed=True)
        .agg(
            trend_mean=("normalized_trend", "mean"),
            future_vol_mean=(target, "mean"),
            future_vol_median=(target, "median"),
            count=(target, "size"),
        )
        .reset_index(drop=True)
    )
    summary["bucket"] = np.arange(1, len(summary) + 1)
    return summary


def future_realized_volatility(
    returns: pd.Series,
    horizon: int,
    annualization: float = 252.0,
) -> pd.Series:
    squared = pd.to_numeric(returns, errors="coerce").pow(2)
    future_mean_square = squared.rolling(horizon, min_periods=horizon).mean().shift(-horizon)
    result = np.sqrt(annualization * future_mean_square)
    result.name = "future_volatility"
    return result


def current_realized_volatility(
    returns: pd.Series,
    window: int,
    annualization: float = 252.0,
) -> pd.Series:
    values = pd.to_numeric(returns, errors="coerce")
    result = np.sqrt(
        annualization * values.pow(2).rolling(window, min_periods=window).mean()
    )
    result.name = "current_volatility"
    return result
