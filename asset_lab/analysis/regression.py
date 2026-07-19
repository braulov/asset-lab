from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


BASELINE_COLUMNS = ["log_daily", "log_weekly", "log_monthly"]
FULL_COLUMNS = [*BASELINE_COLUMNS, "negative_trend", "positive_trend"]


@dataclass(frozen=True)
class AsymmetricRegressionResult:
    label: str
    nobs: int
    r_squared: float
    hac_lags: int
    coefficients: pd.DataFrame
    difference_estimate: float
    difference_std_error: float
    difference_p_value: float
    difference_ci_low: float
    difference_ci_high: float
    baseline_r_squared: float = float("nan")
    delta_r_squared: float = float("nan")

    def summary_record(self) -> dict[str, object]:
        coefficient_map = self.coefficients.set_index("term")
        return {
            "period": self.label,
            "observations": self.nobs,
            "baseline_r_squared": self.baseline_r_squared,
            "full_r_squared": self.r_squared,
            "delta_r_squared": self.delta_r_squared,
            "negative_trend": coefficient_map.loc["negative_trend", "estimate"],
            "negative_ci_low": coefficient_map.loc["negative_trend", "ci_low"],
            "negative_ci_high": coefficient_map.loc["negative_trend", "ci_high"],
            "positive_trend": coefficient_map.loc["positive_trend", "estimate"],
            "positive_ci_low": coefficient_map.loc["positive_trend", "ci_low"],
            "positive_ci_high": coefficient_map.loc["positive_trend", "ci_high"],
            "difference_down_minus_up": self.difference_estimate,
            "difference_p_value": self.difference_p_value,
        }


def _prepare_har_trend_design(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"normalized_trend", "future_variance", *BASELINE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing regression columns: {', '.join(sorted(missing))}")
    work = frame.loc[:, sorted(required)].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work = work.loc[work["future_variance"] > 0.0].copy()
    work["negative_trend"] = np.maximum(-work["normalized_trend"], 0.0)
    work["positive_trend"] = np.maximum(work["normalized_trend"], 0.0)
    work["log_future_variance"] = np.log(work["future_variance"])
    return work


def _fit_ols_hac(y: pd.Series, x: pd.DataFrame, hac_lags: int):
    design = sm.add_constant(x, has_constant="add")
    return sm.OLS(y, design).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(hac_lags), "use_correction": True},
    )


def fit_har_trend_regression(
    frame: pd.DataFrame,
    *,
    label: str = "Full sample",
    hac_lags: int = 5,
    minimum_observations: int = 120,
) -> AsymmetricRegressionResult | None:
    if hac_lags < 0:
        raise ValueError("hac_lags cannot be negative")
    work = _prepare_har_trend_design(frame)
    if len(work) < minimum_observations:
        return None

    y = work["log_future_variance"]
    baseline = _fit_ols_hac(y, work[BASELINE_COLUMNS], hac_lags)
    fitted = _fit_ols_hac(y, work[FULL_COLUMNS], hac_lags)

    confidence = fitted.conf_int(alpha=0.05)
    rows: list[dict[str, float | str]] = []
    for parameter_name in fitted.params.index:
        rows.append(
            {
                "term": "intercept" if parameter_name == "const" else parameter_name,
                "estimate": float(fitted.params[parameter_name]),
                "std_error": float(fitted.bse[parameter_name]),
                "z_value": float(fitted.tvalues[parameter_name]),
                "p_value": float(fitted.pvalues[parameter_name]),
                "ci_low": float(confidence.loc[parameter_name, 0]),
                "ci_high": float(confidence.loc[parameter_name, 1]),
            }
        )

    order = list(fitted.params.index)
    contrast = np.zeros(len(order), dtype=float)
    contrast[order.index("negative_trend")] = 1.0
    contrast[order.index("positive_trend")] = -1.0
    test = fitted.t_test(contrast)
    interval = np.asarray(test.conf_int(alpha=0.05)).reshape(-1, 2)[0]

    return AsymmetricRegressionResult(
        label=label,
        nobs=int(fitted.nobs),
        r_squared=float(fitted.rsquared),
        baseline_r_squared=float(baseline.rsquared),
        delta_r_squared=float(fitted.rsquared - baseline.rsquared),
        hac_lags=int(hac_lags),
        coefficients=pd.DataFrame(rows),
        difference_estimate=float(np.asarray(test.effect).reshape(-1)[0]),
        difference_std_error=float(np.asarray(test.sd).reshape(-1)[0]),
        difference_p_value=float(np.asarray(test.pvalue).reshape(-1)[0]),
        difference_ci_low=float(interval[0]),
        difference_ci_high=float(interval[1]),
    )


def fit_har_trend_regression_by_period(
    frame: pd.DataFrame,
    periods: list[tuple[str, str | None, str | None]],
    *,
    hac_lags: int = 5,
    minimum_observations: int = 80,
) -> list[AsymmetricRegressionResult]:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Period regression requires a DatetimeIndex")
    results: list[AsymmetricRegressionResult] = []
    for label, start, end in periods:
        mask = pd.Series(True, index=frame.index)
        if start is not None:
            mask &= frame.index >= pd.Timestamp(start)
        if end is not None:
            mask &= frame.index <= pd.Timestamp(end)
        result = fit_har_trend_regression(
            frame.loc[mask],
            label=label,
            hac_lags=hac_lags,
            minimum_observations=minimum_observations,
        )
        if result is not None:
            results.append(result)
    return results


def _plain_fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    x_train = sm.add_constant(train[columns], has_constant="add")
    x_test = sm.add_constant(test[columns], has_constant="add")
    x_test = x_test.reindex(columns=x_train.columns, fill_value=1.0)
    fitted = sm.OLS(train["log_future_variance"], x_train).fit()
    return np.exp(np.clip(fitted.predict(x_test).to_numpy(dtype=float), -40.0, 20.0))


def _qlike(actual: np.ndarray, predicted: np.ndarray) -> float:
    ratio = actual / predicted
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def walk_forward_comparison(
    frame: pd.DataFrame,
    *,
    minimum_training_observations: int = 500,
) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        return pd.DataFrame()
    work = _prepare_har_trend_design(frame)
    if work.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int]] = []
    for year in sorted(work.index.year.unique()):
        train = work.loc[work.index.year < year]
        test = work.loc[work.index.year == year]
        if len(train) < minimum_training_observations or len(test) < 20:
            continue
        baseline_prediction = _plain_fit_predict(train, test, BASELINE_COLUMNS)
        full_prediction = _plain_fit_predict(train, test, FULL_COLUMNS)
        actual = test["future_variance"].to_numpy(dtype=float)
        baseline_log_mse = float(np.mean((np.log(actual) - np.log(baseline_prediction)) ** 2))
        full_log_mse = float(np.mean((np.log(actual) - np.log(full_prediction)) ** 2))
        baseline_qlike = _qlike(actual, baseline_prediction)
        full_qlike = _qlike(actual, full_prediction)
        rows.append(
            {
                "test_year": int(year),
                "train_n": int(len(train)),
                "test_n": int(len(test)),
                "baseline_qlike": baseline_qlike,
                "full_qlike": full_qlike,
                "delta_qlike_full_minus_baseline": full_qlike - baseline_qlike,
                "baseline_log_mse": baseline_log_mse,
                "full_log_mse": full_log_mse,
                "delta_log_mse_full_minus_baseline": full_log_mse - baseline_log_mse,
            }
        )
    return pd.DataFrame(rows)


def regression_interpretation(result: AsymmetricRegressionResult) -> str:
    coefficients = result.coefficients.set_index("term")
    down = float(coefficients.loc["negative_trend", "estimate"])
    up = float(coefficients.loc["positive_trend", "estimate"])
    down_variance_pct = 100.0 * (np.exp(down) - 1.0)
    up_variance_pct = 100.0 * (np.exp(up) - 1.0)
    down_vol_pct = 100.0 * (np.exp(down / 2.0) - 1.0)
    up_vol_pct = 100.0 * (np.exp(up / 2.0) - 1.0)

    if result.difference_p_value < 0.05:
        if result.difference_estimate > 0:
            comparison = "The negative-trend coefficient is statistically larger under HAC errors."
        else:
            comparison = "The positive-trend coefficient is statistically larger under HAC errors."
    else:
        comparison = "The down/up coefficient difference is not resolved at the 5% level."

    return (
        f"Holding the daily, weekly and monthly HAR state fixed, one unit of negative "
        f"normalised trend is associated with {down_variance_pct:+.1f}% future variance "
        f"({down_vol_pct:+.1f}% volatility). The corresponding positive-trend estimate is "
        f"{up_variance_pct:+.1f}% variance ({up_vol_pct:+.1f}% volatility). {comparison}"
    )


# Compatibility wrappers retained for notebooks/tests from v3.
def fit_asymmetric_regression(
    frame: pd.DataFrame,
    *,
    label: str = "Full sample",
    hac_lags: int = 5,
    minimum_observations: int = 80,
) -> AsymmetricRegressionResult | None:
    if {"future_variance", *BASELINE_COLUMNS}.issubset(frame.columns):
        return fit_har_trend_regression(
            frame, label=label, hac_lags=hac_lags, minimum_observations=minimum_observations
        )
    required = {"normalized_trend", "future_volatility", "current_volatility"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing regression columns: {', '.join(sorted(missing))}")
    converted = frame.copy()
    converted["future_variance"] = converted["future_volatility"].pow(2) / 252.0
    current_variance = converted["current_volatility"].pow(2) / 252.0
    converted["log_daily"] = np.log(current_variance)
    converted["log_weekly"] = converted["log_daily"]
    converted["log_monthly"] = converted["log_daily"]
    return fit_har_trend_regression(
        converted, label=label, hac_lags=hac_lags, minimum_observations=minimum_observations
    )


def fit_asymmetric_regression_by_period(
    frame: pd.DataFrame,
    periods: list[tuple[str, str | None, str | None]],
    *,
    hac_lags: int = 5,
    minimum_observations: int = 80,
) -> list[AsymmetricRegressionResult]:
    results: list[AsymmetricRegressionResult] = []
    for label, start, end in periods:
        mask = pd.Series(True, index=frame.index)
        if start is not None:
            mask &= frame.index >= pd.Timestamp(start)
        if end is not None:
            mask &= frame.index <= pd.Timestamp(end)
        result = fit_asymmetric_regression(
            frame.loc[mask], label=label, hac_lags=hac_lags,
            minimum_observations=minimum_observations
        )
        if result is not None:
            results.append(result)
    return results
