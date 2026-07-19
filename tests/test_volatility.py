import numpy as np
import pandas as pd

from asset_lab.analysis.volatility import ohlc_variance_proxies, yang_zhang_volatility


def test_gap_rs_includes_overnight_gap() -> None:
    candles = pd.DataFrame(
        {
            "begin": pd.date_range("2024-01-01", periods=3),
            "open": [100.0, 110.0, 110.0],
            "high": [101.0, 111.0, 111.0],
            "low": [99.0, 109.0, 109.0],
            "close": [100.0, 110.0, 110.0],
        }
    )
    proxies = ohlc_variance_proxies(candles, yz_center_window=5)
    assert proxies.loc[pd.Timestamp("2024-01-02"), "gap_rogers_satchell"] > proxies.loc[
        pd.Timestamp("2024-01-02"), "rogers_satchell"
    ]


def test_yang_zhang_is_nonnegative() -> None:
    n = 30
    base = np.linspace(100.0, 110.0, n)
    candles = pd.DataFrame(
        {
            "begin": pd.date_range("2024-01-01", periods=n),
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base * 1.002,
        }
    )
    result = yang_zhang_volatility(candles, 10)
    assert (result.dropna() >= 0.0).all()
