import numpy as np
import pandas as pd

from asset_lab.analysis.regression import fit_asymmetric_regression


def test_asymmetric_regression_recovers_stronger_negative_effect() -> None:
    rng = np.random.default_rng(42)
    n = 2500
    trend = rng.normal(size=n)
    current_vol = np.exp(rng.normal(loc=-1.5, scale=0.25, size=n))
    down = np.maximum(-trend, 0.0)
    up = np.maximum(trend, 0.0)
    log_future = -0.4 + 0.35 * down + 0.05 * up + 0.7 * np.log(current_vol)
    log_future += rng.normal(scale=0.12, size=n)
    frame = pd.DataFrame(
        {
            "normalized_trend": trend,
            "current_volatility": current_vol,
            "future_volatility": np.exp(log_future),
        },
        index=pd.date_range("2010-01-01", periods=n, freq="D"),
    )

    result = fit_asymmetric_regression(frame, hac_lags=4)
    assert result is not None
    coefficients = result.coefficients.set_index("term")
    assert coefficients.loc["negative_trend", "estimate"] > 0.25
    assert coefficients.loc["positive_trend", "estimate"] < 0.15
    assert result.difference_estimate > 0.15
    assert result.difference_p_value < 0.01
