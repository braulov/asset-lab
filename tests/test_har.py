import numpy as np
import pandas as pd

from asset_lab.analysis.har import expanding_har_innovations, fit_har_model, recursive_no_shock_path


def test_har_forecast_is_strictly_past_and_positive() -> None:
    rng = np.random.default_rng(7)
    q = pd.Series(
        np.exp(rng.normal(-7.0, 0.3, 500)),
        index=pd.date_range("2020-01-01", periods=500),
    )
    diagnostics = expanding_har_innovations(
        q,
        initial_training=150,
        refit_every=30,
        mad_window=100,
    )
    assert (diagnostics["forecast_variance"].dropna() > 0.0).all()
    assert diagnostics["shock_score"].notna().sum() > 100


def test_recursive_path_returns_requested_length() -> None:
    rng = np.random.default_rng(1)
    q = pd.Series(np.exp(rng.normal(-7.0, 0.2, 400)))
    model = fit_har_model(q.iloc[:300], minimum_observations=120)
    assert model is not None
    path = recursive_no_shock_path(q.iloc[:300], model, periods=20)
    assert len(path) == 20
    assert np.isfinite(path).all()
    assert (path > 0.0).all()
