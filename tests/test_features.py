import math

import numpy as np
import pandas as pd

from asset_lab.analysis.features import (
    close_to_close_volatility,
    log_returns,
    parkinson_volatility,
    rolling_moments,
)


def test_log_returns() -> None:
    candles = pd.DataFrame({"close": [100.0, 110.0, 121.0]})
    result = log_returns(candles)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest_approx(math.log(1.1))
    assert result.iloc[2] == pytest_approx(math.log(1.1))


def test_rolling_moments_for_symmetric_window() -> None:
    returns = pd.Series([-1.0, 0.0, 1.0])
    moments = rolling_moments(returns, window=3)
    assert moments["m2"].iloc[-1] == pytest_approx(2.0 / 3.0)
    assert moments["m3"].iloc[-1] == pytest_approx(0.0)
    assert moments["skewness"].iloc[-1] == pytest_approx(0.0)


def test_close_to_close_volatility_constant_returns_is_zero() -> None:
    returns = pd.Series([0.01] * 10)
    result = close_to_close_volatility(returns, window=5)
    assert result.dropna().abs().max() == pytest_approx(0.0)


def test_parkinson_volatility_constant_range() -> None:
    candles = pd.DataFrame({"high": [110.0] * 5, "low": [100.0] * 5})
    result = parkinson_volatility(candles, window=5, annualization=1.0)
    expected = abs(math.log(1.1)) / math.sqrt(4.0 * math.log(2.0))
    assert result.iloc[-1] == pytest_approx(expected)


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value)


def test_log_returns_can_exclude_long_calendar_gap() -> None:
    candles = pd.DataFrame(
        {
            "begin": ["2022-02-25", "2022-03-24", "2022-03-25"],
            "close": [100.0, 50.0, 55.0],
        }
    )
    result = log_returns(candles, max_calendar_gap_days=7)
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest_approx(math.log(1.1))

from asset_lab.analysis.features import exclude_current_unfinished_daily_candle


def test_current_day_candle_can_be_excluded() -> None:
    frame = pd.DataFrame(
        {
            "begin": pd.to_datetime(["2026-07-16", "2026-07-17"]),
            "close": [100.0, 101.0],
        }
    )
    cleaned, excluded = exclude_current_unfinished_daily_candle(
        frame,
        current_date="2026-07-17",
    )
    assert excluded
    assert len(cleaned) == 1
    assert cleaned["begin"].iloc[-1] == pd.Timestamp("2026-07-16")
