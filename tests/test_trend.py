import numpy as np
import pandas as pd
import pytest

from asset_lab.analysis.trend import future_realized_volatility, normalized_trend


def test_future_realized_volatility_uses_strict_future() -> None:
    returns = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    result = future_realized_volatility(returns, horizon=2, annualization=1.0)
    assert result.iloc[0] == pytest.approx(np.sqrt((1.0**2 + 2.0**2) / 2.0))
    assert result.iloc[1] == pytest.approx(np.sqrt((2.0**2 + 3.0**2) / 2.0))
    assert np.isnan(result.iloc[-2])
    assert np.isnan(result.iloc[-1])


def test_normalized_trend_zero_for_zero_sum() -> None:
    returns = pd.Series([-1.0, 0.0, 1.0])
    result = normalized_trend(returns, window=3)
    assert result.iloc[-1] == pytest.approx(0.0)

from asset_lab.analysis.trend import build_trend_volatility_frame


def test_trend_frame_contains_current_volatility() -> None:
    returns = pd.Series(np.linspace(-0.03, 0.03, 100))
    frame = build_trend_volatility_frame(
        returns,
        trend_window=10,
        future_horizon=5,
        current_volatility_window=12,
        annualization=1.0,
    )
    assert {"normalized_trend", "current_volatility", "future_volatility"}.issubset(
        frame.columns
    )
    assert not frame.empty
