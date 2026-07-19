from __future__ import annotations

import math

import numpy as np
import pandas as pd


PROXY_LABELS: dict[str, str] = {
    "close_to_close": "Close-to-close squared return",
    "parkinson": "Parkinson",
    "garman_klass": "Garman–Klass",
    "rogers_satchell": "Rogers–Satchell",
    "gap_rogers_satchell": "Gap² + Rogers–Satchell",
    "yang_zhang_contribution": "Yang–Zhang daily contribution",
}


def candle_index(candles: pd.DataFrame) -> pd.Index:
    if "begin" in candles.columns:
        timestamps = pd.to_datetime(candles["begin"], errors="coerce")
        if not timestamps.isna().all():
            return pd.DatetimeIndex(timestamps)
    return candles.index


def _positive_log(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").where(lambda item: item > 0)
    return np.log(numeric)


def _long_gap_mask(candles: pd.DataFrame, max_calendar_gap_days: int | None) -> pd.Series:
    index = candle_index(candles)
    mask = pd.Series(False, index=index)
    if max_calendar_gap_days is None or not isinstance(index, pd.DatetimeIndex):
        return mask
    gap_days = index.to_series(index=index).diff().dt.total_seconds() / 86_400.0
    return gap_days > max_calendar_gap_days


def ohlc_variance_proxies(
    candles: pd.DataFrame,
    *,
    yz_center_window: int = 20,
    max_calendar_gap_days: int | None = None,
) -> pd.DataFrame:
    """Compute transparent daily variance proxies from OHLC candles.

    All outputs are in squared-log-return units.  The two gap-dependent proxies are
    masked after unusually long calendar gaps when ``max_calendar_gap_days`` is set.
    """
    if yz_center_window < 5:
        raise ValueError("Yang–Zhang centering window must be at least 5 periods.")

    required = {"open", "high", "low", "close"}
    missing = required.difference(candles.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {', '.join(sorted(missing))}")

    index = candle_index(candles)
    o = pd.Series(_positive_log(candles["open"]).to_numpy(), index=index, name="log_open")
    h = pd.Series(_positive_log(candles["high"]).to_numpy(), index=index, name="log_high")
    l = pd.Series(_positive_log(candles["low"]).to_numpy(), index=index, name="log_low")
    c = pd.Series(_positive_log(candles["close"]).to_numpy(), index=index, name="log_close")

    previous_close = c.shift(1)
    close_return = c - previous_close
    overnight = o - previous_close
    open_to_close = c - o
    log_range = h - l

    close_to_close = close_return.pow(2)
    parkinson = log_range.pow(2) / (4.0 * math.log(2.0))
    garman_klass = 0.5 * log_range.pow(2) - (2.0 * math.log(2.0) - 1.0) * open_to_close.pow(2)
    rogers_satchell = (h - o) * (h - c) + (l - o) * (l - c)

    # Small negatives can occur through inconsistent rounded OHLC inputs.  A variance
    # proxy cannot be negative, so clip only after retaining the raw ingredients.
    garman_klass = garman_klass.clip(lower=0.0)
    rogers_satchell = rogers_satchell.clip(lower=0.0)
    gap_rs = overnight.pow(2) + rogers_satchell

    k = 0.34 / (1.34 + (yz_center_window + 1.0) / (yz_center_window - 1.0))
    past_overnight_mean = overnight.shift(1).rolling(
        yz_center_window, min_periods=yz_center_window
    ).mean()
    past_open_close_mean = open_to_close.shift(1).rolling(
        yz_center_window, min_periods=yz_center_window
    ).mean()
    yz_contribution = (
        (overnight - past_overnight_mean).pow(2)
        + k * (open_to_close - past_open_close_mean).pow(2)
        + (1.0 - k) * rogers_satchell
    )

    long_gap = _long_gap_mask(candles, max_calendar_gap_days)
    for series in (close_to_close, gap_rs, yz_contribution):
        series.loc[long_gap] = np.nan

    result = pd.DataFrame(
        {
            "close_to_close": close_to_close,
            "parkinson": parkinson,
            "garman_klass": garman_klass,
            "rogers_satchell": rogers_satchell,
            "gap_rogers_satchell": gap_rs,
            "yang_zhang_contribution": yz_contribution,
            "overnight_gap_squared": overnight.pow(2).mask(long_gap),
            "intraday_rs": rogers_satchell,
            "close_return": close_return.mask(long_gap),
            "overnight_return": overnight.mask(long_gap),
            "open_to_close_return": open_to_close,
        },
        index=index,
    )
    return result.replace([np.inf, -np.inf], np.nan)


def rolling_volatility_from_variance(
    variance_proxy: pd.Series,
    window: int,
    *,
    annualization: float = 252.0,
) -> pd.Series:
    if window < 2:
        raise ValueError("Rolling volatility window must be at least 2 periods.")
    values = pd.to_numeric(variance_proxy, errors="coerce").clip(lower=0.0)
    rolling_variance = values.rolling(window, min_periods=window).mean()
    result = np.sqrt(annualization * rolling_variance)
    result.name = "rolling_volatility"
    return result


def yang_zhang_volatility(
    candles: pd.DataFrame,
    window: int,
    *,
    annualization: float = 252.0,
    max_calendar_gap_days: int | None = None,
) -> pd.Series:
    """Rolling Yang–Zhang volatility, including overnight gaps and drift robustness."""
    if window < 3:
        raise ValueError("Yang–Zhang window must be at least 3 periods.")

    index = candle_index(candles)
    o = pd.Series(_positive_log(candles["open"]).to_numpy(), index=index)
    h = pd.Series(_positive_log(candles["high"]).to_numpy(), index=index)
    l = pd.Series(_positive_log(candles["low"]).to_numpy(), index=index)
    c = pd.Series(_positive_log(candles["close"]).to_numpy(), index=index)

    overnight = o - c.shift(1)
    open_to_close = c - o
    rs = ((h - o) * (h - c) + (l - o) * (l - c)).clip(lower=0.0)
    long_gap = _long_gap_mask(candles, max_calendar_gap_days)
    overnight = overnight.mask(long_gap)

    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    overnight_variance = overnight.rolling(window, min_periods=window).var(ddof=1)
    open_close_variance = open_to_close.rolling(window, min_periods=window).var(ddof=1)
    rs_mean = rs.rolling(window, min_periods=window).mean()
    yz_variance = overnight_variance + k * open_close_variance + (1.0 - k) * rs_mean

    result = np.sqrt(annualization * yz_variance.clip(lower=0.0))
    result.name = "yang_zhang_volatility"
    return result


def proxy_correlation_matrix(proxies: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in PROXY_LABELS if column in proxies.columns]
    frame = proxies[columns].replace([np.inf, -np.inf], np.nan)
    correlation = frame.corr(method="spearman")
    correlation.index = [PROXY_LABELS.get(item, item) for item in correlation.index]
    correlation.columns = [PROXY_LABELS.get(item, item) for item in correlation.columns]
    return correlation


def proxy_disagreement_days(proxies: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Dates where estimators disagree most after normalising each by its median."""
    columns = [column for column in PROXY_LABELS if column in proxies.columns]
    frame = proxies[columns].copy()
    medians = frame.median(skipna=True).replace(0.0, np.nan)
    normalised = frame.divide(medians, axis="columns")
    disagreement = normalised.max(axis=1) - normalised.min(axis=1)
    output = frame.copy()
    output.insert(0, "disagreement", disagreement)
    output = output.nlargest(top_n, "disagreement").reset_index(names="timestamp")
    return output.rename(columns=PROXY_LABELS)


def rolling_asymmetry_metrics(returns: pd.Series, window: int) -> pd.DataFrame:
    """Robust complements to raw M3 for downside/upside asymmetry."""
    if window < 5:
        raise ValueError("Asymmetry window must be at least 5 periods.")
    values = pd.to_numeric(returns, errors="coerce")
    squared = values.pow(2)
    downside = squared.where(values < 0.0, 0.0).rolling(window, min_periods=window).sum()
    upside = squared.where(values > 0.0, 0.0).rolling(window, min_periods=window).sum()
    total = downside + upside
    downside_share = downside / total.where(total > 0.0)
    epsilon = np.finfo(float).eps
    semivariance_log_ratio = np.log((downside + epsilon) / (upside + epsilon))

    def quantile_skew(array: np.ndarray) -> float:
        q25, q50, q75 = np.quantile(array, [0.25, 0.5, 0.75])
        denominator = q75 - q25
        if denominator <= 0:
            return float("nan")
        return float((q75 + q25 - 2.0 * q50) / denominator)

    qskew = values.rolling(window, min_periods=window).apply(quantile_skew, raw=True)
    return pd.DataFrame(
        {
            "downside_variance_share": downside_share,
            "downside_upside_log_ratio": semivariance_log_ratio,
            "quantile_skewness": qskew,
        },
        index=values.index,
    )
