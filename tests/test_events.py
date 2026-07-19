import numpy as np
import pandas as pd
import pytest

from asset_lab.analysis.events import (
    aggregate_trajectories,
    build_event_trajectories,
    detect_volatility_peaks,
)


def test_peak_detection_suppresses_neighbors() -> None:
    vol = pd.Series([1, 1, 5, 4, 1, 1, 6, 5, 1], dtype=float)
    events = detect_volatility_peaks(vol, quantile=0.7, cooldown=1)
    assert events["position"].tolist() == [2, 6]


def test_event_normalization_peak_is_one() -> None:
    values = np.ones(20)
    values[8] = 5.0
    vol = pd.Series(values)
    events = pd.DataFrame(
        [{"position": 8, "timestamp": 8, "volatility": 5.0, "threshold": 4.0}]
    )
    trajectories = build_event_trajectories(vol, events, pre_days=5, post_days=5)
    peak = trajectories.loc[trajectories["offset"] == 0, "normalized_excess"].iloc[0]
    assert peak == pytest.approx(1.0)
    aggregate = aggregate_trajectories(trajectories)
    assert aggregate.loc[aggregate["offset"] == 0, "median"].iloc[0] == pytest.approx(1.0)

from asset_lab.analysis.events import (
    build_forward_volatility_trajectories,
    detect_return_shocks,
    return_shock_score,
)


def test_return_shock_score_uses_strict_past_scale() -> None:
    returns = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0, 10.0])
    diagnostics = return_shock_score(returns, lookback=5)
    assert diagnostics.loc[5, "past_sigma"] == pytest.approx(np.std([1, -1, 1, -1, 1]))
    assert diagnostics.loc[5, "shock_score"] == pytest.approx(
        10.0 / np.std([1, -1, 1, -1, 1])
    )


def test_forward_response_excludes_event_return() -> None:
    values = np.array([1.0, -1.0, 1.0, -1.0, 1.0, 10.0, 1.0, -1.0, 1.0, -1.0])
    returns = pd.Series(values)
    events = pd.DataFrame(
        [
            {
                "position": 5,
                "timestamp": 5,
                "return": 10.0,
                "direction": "positive",
                "shock_score": 10.0,
            }
        ]
    )
    trajectories = build_forward_volatility_trajectories(
        returns,
        events,
        baseline_window=5,
        future_window=2,
        post_days=1,
        annualization=1.0,
    )
    assert trajectories.loc[trajectories["offset"] == 0, "future_volatility"].iloc[0] == pytest.approx(1.0)
    assert trajectories.loc[trajectories["offset"] == 0, "excess_ratio"].iloc[0] == pytest.approx(0.0)


def test_detect_return_shocks_can_filter_negative_events() -> None:
    returns = pd.Series(
        [0.01, -0.01] * 40 + [-0.20, 0.01, 0.18, 0.01],
        index=pd.date_range("2020-01-01", periods=84, freq="D"),
    )
    events = detect_return_shocks(
        returns,
        lookback=20,
        quantile=0.9,
        cooldown=2,
        direction="negative",
    )
    assert not events.empty
    assert (events["return"] < 0).all()
