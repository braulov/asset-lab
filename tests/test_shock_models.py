import numpy as np
import pandas as pd

from asset_lab.analysis.har import HarModel, recursive_log_median_path
from asset_lab.analysis.shock_models import (
    aftershock_excess_rate,
    fit_baseline_omori,
    fit_local_projection_responses,
    matching_balance_summary,
    select_matched_controls,
)


def test_recursive_log_median_does_not_compound_smearing() -> None:
    history = pd.Series(np.full(30, 0.01))
    params = pd.Series(
        {"const": np.log(0.01), "log_daily": 0.0, "log_weekly": 0.0, "log_monthly": 0.0}
    )
    path_one = recursive_log_median_path(
        history,
        HarModel(params=params, smearing=1.0, nobs=100, robust=True),
        periods=10,
    )
    path_large = recursive_log_median_path(
        history,
        HarModel(params=params, smearing=5.0, nobs=100, robust=True),
        periods=10,
    )
    assert np.allclose(path_one["forecast_median_variance"], 0.01)
    assert np.allclose(
        path_one["forecast_median_variance"],
        path_large["forecast_median_variance"],
    )


def test_matched_controls_exclude_event_neighbourhood_and_balance() -> None:
    rng = np.random.default_rng(4)
    index = pd.date_range("2018-01-01", periods=700, freq="B")
    q = pd.Series(np.exp(-8.0 + 0.1 * rng.normal(size=len(index))), index=index)
    returns = pd.Series(0.01 * rng.normal(size=len(index)), index=index)
    diagnostics = pd.DataFrame(
        {
            "forecast_log_variance": np.log(q).rolling(22, min_periods=1).mean(),
            "shock_score": 0.0,
        },
        index=index,
    )
    event_positions = [300, 450]
    events = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "position": position,
                "timestamp": index[position],
                "direction": "negative" if event_id == 0 else "positive",
                "shock_score": 3.0,
                "threshold": 2.0,
            }
            for event_id, position in enumerate(event_positions)
        ]
    )
    diagnostics.iloc[event_positions, diagnostics.columns.get_loc("shock_score")] = 3.0
    matches = select_matched_controls(
        q,
        returns,
        diagnostics,
        events,
        future_window=5,
        post_days=20,
        controls_per_event=5,
        exclusion_radius=10,
    )
    assert matches["event_id"].nunique() == 2
    for event_position in event_positions:
        assert (matches["control_position"] - event_position).abs().min() > 10
    balance = matching_balance_summary(matches)
    assert not balance.empty
    assert np.isfinite(balance["standardized_mean_difference"]).all()


def test_local_projection_recovers_larger_negative_event_effect() -> None:
    rng = np.random.default_rng(8)
    index = pd.date_range("2015-01-01", periods=1600, freq="B")
    q = np.exp(-8.0 + 0.12 * rng.normal(size=len(index)))
    returns = 0.01 * rng.normal(size=len(index))
    negative_positions = np.arange(300, 1100, 40)
    positive_positions = np.arange(320, 1120, 40)
    for position in negative_positions:
        q[position + 1 : position + 6] *= np.exp(0.55)
        returns[position] = -0.05
    for position in positive_positions:
        q[position + 1 : position + 6] *= np.exp(0.20)
        returns[position] = 0.05

    q_series = pd.Series(q, index=index)
    returns_series = pd.Series(returns, index=index)
    diagnostics = pd.DataFrame(
        {
            "forecast_log_variance": pd.Series(np.log(q), index=index).shift(1).rolling(22).mean(),
            "shock_score": 0.0,
        },
        index=index,
    )
    rows = []
    for event_id, position in enumerate(negative_positions):
        diagnostics.iloc[position, diagnostics.columns.get_loc("shock_score")] = 3.0
        rows.append(
            {
                "event_id": event_id,
                "position": int(position),
                "timestamp": index[position],
                "direction": "negative",
                "shock_score": 3.0,
                "threshold": 2.0,
            }
        )
    start = len(rows)
    for event_id, position in enumerate(positive_positions, start=start):
        diagnostics.iloc[position, diagnostics.columns.get_loc("shock_score")] = 3.0
        rows.append(
            {
                "event_id": event_id,
                "position": int(position),
                "timestamp": index[position],
                "direction": "positive",
                "shock_score": 3.0,
                "threshold": 2.0,
            }
        )
    events = pd.DataFrame(rows)
    response = fit_local_projection_responses(
        q_series,
        returns_series,
        diagnostics,
        events,
        future_window=3,
        post_days=3,
        stride=1,
        contamination_radius=5,
        include_year_fixed_effects=True,
        minimum_observations=500,
    )
    assert not response.empty
    first = response.iloc[0]
    assert first["negative_estimate"] > first["positive_estimate"] > 0.0


def test_flat_aftershock_rate_is_not_called_omori_decay() -> None:
    rng = np.random.default_rng(12)
    offsets = np.arange(1, 31)
    exposure = np.full(len(offsets), 100)
    count = rng.binomial(exposure, 0.10)
    rate = pd.DataFrame(
        {
            "offset": offsets,
            "rate": count / exposure,
            "count": count,
            "exposed_events": exposure,
            "baseline_rate": 0.10,
        }
    )
    fit = fit_baseline_omori(rate)
    assert fit is not None
    assert not fit.meaningful_decay
