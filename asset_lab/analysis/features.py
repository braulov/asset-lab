from __future__ import annotations

import math

import numpy as np
import pandas as pd


REQUIRED_OHLC = ("open", "high", "low", "close")


def _candle_timestamps(candles: pd.DataFrame) -> pd.Series | None:
    if "begin" not in candles.columns:
        return None
    timestamps = pd.to_datetime(candles["begin"], errors="coerce")
    if timestamps.isna().all():
        return None
    return timestamps


def calendar_gap_report(
    candles: pd.DataFrame,
    threshold_days: int = 7,
    price_column: str = "close",
) -> pd.DataFrame:
    """Return returns whose endpoints are separated by a long calendar gap."""
    if threshold_days < 1:
        raise ValueError("Порог календарного разрыва должен быть положительным.")

    timestamps = _candle_timestamps(candles)
    if timestamps is None:
        return pd.DataFrame(
            columns=["previous_timestamp", "timestamp", "gap_days", "log_return"]
        )

    prices = pd.to_numeric(candles[price_column], errors="coerce").where(lambda x: x > 0)
    returns = np.log(prices).diff()
    gap_days = timestamps.diff().dt.total_seconds() / 86_400.0
    mask = gap_days > threshold_days

    report = pd.DataFrame(
        {
            "previous_timestamp": timestamps.shift(1),
            "timestamp": timestamps,
            "gap_days": gap_days,
            "log_return": returns,
        }
    )
    return report.loc[mask].reset_index(drop=True)


def validate_candles(candles: pd.DataFrame) -> list[str]:
    """Return human-readable data-quality warnings."""
    warnings: list[str] = []
    if candles.empty:
        return ["Ряд пуст."]

    missing = [column for column in REQUIRED_OHLC if column not in candles.columns]
    if missing:
        warnings.append(f"Отсутствуют OHLC-поля: {', '.join(missing)}.")
        return warnings

    if candles["close"].isna().any():
        warnings.append("В close есть пропущенные значения.")
    if (candles[["open", "high", "low", "close"]] <= 0).any().any():
        warnings.append("Есть неположительные OHLC-значения; логарифмические метрики искажены.")
    if "begin" in candles.columns and candles["begin"].duplicated().any():
        warnings.append("Есть повторяющиеся timestamps.")
    if (candles["high"] < candles[["open", "close", "low"]].max(axis=1)).any():
        warnings.append("Есть строки, где high меньше другого OHLC-значения.")
    if (candles["low"] > candles[["open", "close", "high"]].min(axis=1)).any():
        warnings.append("Есть строки, где low больше другого OHLC-значения.")
    return warnings


def log_returns(
    candles: pd.DataFrame,
    price_column: str = "close",
    max_calendar_gap_days: int | None = None,
) -> pd.Series:
    """Compute close-to-close log returns, optionally removing long-gap returns.

    A daily close-to-close return after a long trading suspension spans many calendar
    days and should not silently be treated as an ordinary one-day observation.
    """
    prices = pd.to_numeric(candles[price_column], errors="coerce")
    prices = prices.where(prices > 0)
    result = np.log(prices).diff()

    if max_calendar_gap_days is not None:
        if max_calendar_gap_days < 1:
            raise ValueError("max_calendar_gap_days должен быть положительным.")
        timestamps = _candle_timestamps(candles)
        if timestamps is not None:
            gap_days = timestamps.diff().dt.total_seconds() / 86_400.0
            result = result.mask(gap_days > max_calendar_gap_days)

    result.name = "log_return"
    return result


def rolling_moments(returns: pd.Series, window: int) -> pd.DataFrame:
    if window < 3:
        raise ValueError("Окно моментов должно быть не меньше 3.")

    values = pd.to_numeric(returns, errors="coerce")
    rolling_mean = values.rolling(window=window, min_periods=window).mean()
    # M2/M3 are population central moments inside each rolling window (ddof=0).
    m2 = values.rolling(window=window, min_periods=window).var(ddof=0)
    m3 = values.rolling(window=window, min_periods=window).apply(
        lambda x: float(np.mean((x - np.mean(x)) ** 3)),
        raw=True,
    )
    denominator = np.power(m2, 1.5)
    skewness = m3 / denominator.where(denominator > 0)
    signed_cuberoot_m3 = np.sign(m3) * np.cbrt(np.abs(m3))

    return pd.DataFrame(
        {
            "rolling_mean": rolling_mean,
            "m2": m2,
            "m3": m3,
            "signed_cuberoot_m3": signed_cuberoot_m3,
            "skewness": skewness,
        },
        index=returns.index,
    )


def close_to_close_volatility(
    returns: pd.Series,
    window: int,
    annualization: float = 252.0,
) -> pd.Series:
    variance = pd.to_numeric(returns, errors="coerce").rolling(
        window=window, min_periods=window
    ).var(ddof=0)
    result = np.sqrt(annualization * variance)
    result.name = "close_to_close_volatility"
    return result


def ewma_volatility(
    returns: pd.Series,
    span: int = 20,
    annualization: float = 252.0,
) -> pd.Series:
    if span < 2:
        raise ValueError("EWMA span должен быть не меньше 2.")
    variance = pd.to_numeric(returns, errors="coerce").ewm(
        span=span,
        adjust=False,
        min_periods=span,
    ).var(bias=True)
    result = np.sqrt(annualization * variance)
    result.name = "ewma_volatility"
    return result


def parkinson_volatility(
    candles: pd.DataFrame,
    window: int,
    annualization: float = 252.0,
) -> pd.Series:
    high = pd.to_numeric(candles["high"], errors="coerce").where(lambda x: x > 0)
    low = pd.to_numeric(candles["low"], errors="coerce").where(lambda x: x > 0)
    squared_range = np.log(high / low) ** 2
    variance = squared_range.rolling(window=window, min_periods=window).mean() / (
        4.0 * math.log(2.0)
    )
    result = np.sqrt(annualization * variance)
    result.name = "parkinson_volatility"
    return result


def exclude_current_unfinished_daily_candle(
    candles: pd.DataFrame,
    *,
    current_date: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Drop a candle whose trading date is the current date.

    MOEX ISS may expose a still-forming daily candle.  Research calculations should not
    mix that partial observation with completed daily candles.  Users can disable this
    conservative rule in the interface when they deliberately need the live candle.
    """
    if candles.empty or "begin" not in candles.columns:
        return candles.copy(), False
    today = pd.Timestamp(current_date).normalize() if current_date is not None else pd.Timestamp.today().normalize()
    begin = pd.to_datetime(candles["begin"], errors="coerce")
    if begin.empty or pd.isna(begin.iloc[-1]):
        return candles.copy(), False
    if begin.iloc[-1].normalize() >= today:
        return candles.iloc[:-1].copy(), True
    return candles.copy(), False
