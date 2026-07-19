from __future__ import annotations

import numpy as np
import pandas as pd

from asset_lab.analysis.hourly import (
    affine_mobility_skewness,
    detect_moment_shocks,
    fit_mobility_models,
    mobility_price_bins,
    past_hourly_standardisation,
    prepare_hourly_candles,
)


def synthetic_candles(days: int = 180, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = []
    for day in pd.bdate_range("2024-01-01", periods=days):
        for hour in range(10, 19):
            timestamps.append(day + pd.Timedelta(hours=hour))
    timestamps = pd.DatetimeIndex(timestamps)
    price = np.empty(len(timestamps))
    price[0] = 100.0
    returns = rng.normal(0.0, 0.004, len(timestamps))
    for index in range(1, len(price)):
        price[index] = price[index - 1] * np.exp(returns[index - 1])
    open_price = price
    close_price = open_price * np.exp(returns)
    high = np.maximum(open_price, close_price) * 1.001
    low = np.minimum(open_price, close_price) * 0.999
    return pd.DataFrame(
        {
            "begin": timestamps,
            "end": timestamps + pd.Timedelta(hours=1),
            "open": open_price,
            "close": close_price,
            "high": high,
            "low": low,
            "volume": 1_000.0,
            "value": 100_000.0,
        }
    )


def test_past_standardisation_has_no_future_lookahead() -> None:
    candles = synthetic_candles()
    prepared = prepare_hourly_candles(candles)
    baseline = past_hourly_standardisation(prepared, scale_window=90, minimum_periods=30)

    modified = candles.copy()
    modified.loc[modified.index[-1], "close"] *= 1.5
    prepared_modified = prepare_hourly_candles(modified)
    changed = past_hourly_standardisation(prepared_modified, scale_window=90, minimum_periods=30)

    np.testing.assert_allclose(
        baseline["conditional_scale"].iloc[:-1],
        changed["conditional_scale"].iloc[:-1],
        rtol=0,
        atol=0,
    )


def test_affine_skewness_is_nonnegative_and_increasing() -> None:
    horizons = np.array([1.0, 2.0, 4.0, 8.0])
    values = affine_mobility_skewness(0.03, horizons)
    assert np.all(values >= 0)
    assert np.all(np.diff(values) > 0)


def test_mobility_fit_recovers_proportional_synthetic_bins() -> None:
    prices = np.linspace(50.0, 200.0, 40)
    frame = pd.DataFrame(
        {
            "year": np.repeat(np.arange(2020, 2025), 8),
            "price": prices,
            "observations": 100,
            "mobility": 0.012 * prices,
        }
    )
    fits, cv = fit_mobility_models(frame)
    proportional = next(fit for fit in fits if fit.model == "proportional")
    assert abs(proportional.parameters["b"] - 0.012) < 1e-8
    assert cv.iloc[0]["model"] in {"proportional", "affine"}


def test_detect_moment_shocks_splits_cold_and_hot_precursors() -> None:
    candles = synthetic_candles(days=220)
    prepared = prepare_hourly_candles(candles)
    standardised = past_hourly_standardisation(prepared, scale_window=90, minimum_periods=30)
    regular = standardised.loc[standardised["regular_session"]].copy().reset_index(drop=True)

    # Inject separated extreme returns with very different precursor M2 states.
    regular.loc[700:711, "m2_clipped"] = 0.1
    regular.loc[712, "z_return"] = -8.0
    regular.loc[712, "m2_clipped"] = 36.0
    regular.loc[1200:1211, "m2_clipped"] = 10.0
    regular.loc[1212, "z_return"] = -8.0
    regular.loc[1212, "m2_clipped"] = 36.0

    events = detect_moment_shocks(
        regular,
        quantile=0.995,
        cooldown=12,
        precursor_window=12,
        post_horizon=12,
    )
    assert "abrupt-like" in set(events["heating_class"])
    assert "preheated-like" in set(events["heating_class"])


def test_mobility_bins_returns_expected_columns() -> None:
    candles = synthetic_candles(days=400)
    frame = prepare_hourly_candles(candles)
    bins = mobility_price_bins(frame, bins=6, minimum_bucket_size=20)
    assert {"year", "price", "mobility", "m2_price", "m3_price"}.issubset(bins.columns)
