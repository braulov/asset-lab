from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


EPSILON = 1e-12
HAR_COLUMNS = ["const", "log_daily", "log_weekly", "log_monthly"]


@dataclass(frozen=True)
class HarModel:
    params: pd.Series
    smearing: float
    nobs: int
    robust: bool


def har_design(variance_proxy: pd.Series) -> pd.DataFrame:
    """One-step HAR design: every predictor for day t uses data no later than t-1."""
    q = pd.to_numeric(variance_proxy, errors="coerce").where(lambda item: item > 0.0)
    past = q.shift(1)
    frame = pd.DataFrame(index=q.index)
    frame["target_log_variance"] = np.log(q + EPSILON)
    frame["log_daily"] = np.log(past + EPSILON)
    frame["log_weekly"] = np.log(past.rolling(5, min_periods=5).mean() + EPSILON)
    frame["log_monthly"] = np.log(past.rolling(22, min_periods=22).mean() + EPSILON)
    return frame.replace([np.inf, -np.inf], np.nan)


def current_har_state(variance_proxy: pd.Series) -> pd.DataFrame:
    """HAR state known at t, for forecasting a future window beginning at t+1."""
    q = pd.to_numeric(variance_proxy, errors="coerce").where(lambda item: item > 0.0)
    return pd.DataFrame(
        {
            "log_daily": np.log(q + EPSILON),
            "log_weekly": np.log(q.rolling(5, min_periods=5).mean() + EPSILON),
            "log_monthly": np.log(q.rolling(22, min_periods=22).mean() + EPSILON),
        },
        index=q.index,
    ).replace([np.inf, -np.inf], np.nan)


def fit_har_model(
    variance_proxy: pd.Series,
    *,
    minimum_observations: int = 120,
    robust: bool = True,
) -> HarModel | None:
    frame = har_design(variance_proxy).dropna()
    if len(frame) < minimum_observations:
        return None

    x = sm.add_constant(frame[["log_daily", "log_weekly", "log_monthly"]], has_constant="add")
    y = frame["target_log_variance"]
    try:
        if robust:
            fitted = sm.RLM(y, x, M=sm.robust.norms.HuberT()).fit(maxiter=200)
        else:
            fitted = sm.OLS(y, x).fit()
        residuals = np.asarray(y - fitted.predict(x), dtype=float)
        smearing = float(np.nanmean(np.exp(np.clip(residuals, -20.0, 20.0))))
        if not np.isfinite(smearing) or smearing <= 0.0:
            smearing = 1.0
        params = pd.Series(fitted.params, index=x.columns, dtype=float)
        return HarModel(params=params, smearing=smearing, nobs=len(frame), robust=robust)
    except (ValueError, np.linalg.LinAlgError):
        if robust:
            return fit_har_model(
                variance_proxy,
                minimum_observations=minimum_observations,
                robust=False,
            )
        return None


def _state_from_history(history: list[float]) -> np.ndarray | None:
    finite = np.asarray(history, dtype=float)
    if len(finite) < 22 or not np.isfinite(finite[-22:]).all() or np.any(finite[-22:] <= 0.0):
        return None
    return np.asarray(
        [
            1.0,
            np.log(finite[-1] + EPSILON),
            np.log(np.mean(finite[-5:]) + EPSILON),
            np.log(np.mean(finite[-22:]) + EPSILON),
        ],
        dtype=float,
    )


def predict_next_variance(history: list[float], model: HarModel) -> float:
    state = _state_from_history(history)
    if state is None:
        return float("nan")
    parameters = model.params.reindex(HAR_COLUMNS).to_numpy(dtype=float)
    predicted_log = float(state @ parameters)
    predicted = float(np.exp(np.clip(predicted_log, -40.0, 20.0)) * model.smearing)
    return predicted if np.isfinite(predicted) and predicted > 0.0 else float("nan")




def predict_next_log_variance(history: list[float], model: HarModel) -> float:
    """Conditional median forecast in log-variance space.

    The lognormal smearing correction is deliberately not applied here.  A recursive
    state path must feed conditional medians back into the HAR state; recursively
    feeding bias-corrected means compounds the one-step correction and can dominate
    the counterfactual.
    """
    state = _state_from_history(history)
    if state is None:
        return float("nan")
    parameters = model.params.reindex(HAR_COLUMNS).to_numpy(dtype=float)
    predicted_log = float(state @ parameters)
    return predicted_log if np.isfinite(predicted_log) else float("nan")


def recursive_log_median_path(
    history_before_event: pd.Series,
    model: HarModel,
    periods: int,
) -> pd.DataFrame:
    """Recursive no-shock path without compounding a lognormal smearing factor.

    Returns both the conditional log-median and the corresponding variance level.
    The level is appended to the state history, while the one-step smearing factor is
    retained only as metadata for diagnostics.
    """
    if periods < 1:
        return pd.DataFrame(columns=["forecast_log_median", "forecast_median_variance"])
    history = pd.to_numeric(history_before_event, errors="coerce").dropna().tolist()
    rows: list[dict[str, float]] = []
    for _ in range(periods):
        predicted_log = predict_next_log_variance(history, model)
        if not np.isfinite(predicted_log):
            return pd.DataFrame(columns=["forecast_log_median", "forecast_median_variance"])
        predicted_median = float(np.exp(np.clip(predicted_log, -40.0, 20.0)))
        if not np.isfinite(predicted_median) or predicted_median <= 0.0:
            return pd.DataFrame(columns=["forecast_log_median", "forecast_median_variance"])
        rows.append(
            {
                "forecast_log_median": predicted_log,
                "forecast_median_variance": predicted_median,
                "one_step_smearing": float(model.smearing),
            }
        )
        history.append(predicted_median)
    return pd.DataFrame(rows)


def recursive_no_shock_path(
    history_before_event: pd.Series,
    model: HarModel,
    periods: int,
) -> np.ndarray:
    """Forecast the event day and later days while replacing the shock by forecasts."""
    if periods < 1:
        return np.asarray([], dtype=float)
    history = pd.to_numeric(history_before_event, errors="coerce").tolist()
    predictions: list[float] = []
    for _ in range(periods):
        predicted = predict_next_variance(history, model)
        if not np.isfinite(predicted):
            return np.asarray([], dtype=float)
        predictions.append(predicted)
        history.append(predicted)
    return np.asarray(predictions, dtype=float)


def expanding_har_innovations(
    variance_proxy: pd.Series,
    *,
    initial_training: int = 252,
    refit_every: int = 20,
    mad_window: int = 250,
    robust: bool = True,
) -> pd.DataFrame:
    """Strictly past HAR forecasts and robustly-standardised log-variance innovations."""
    if initial_training < 120:
        raise ValueError("Initial HAR training sample must be at least 120 periods.")
    if refit_every < 1:
        raise ValueError("HAR refit frequency must be positive.")
    if mad_window < 30:
        raise ValueError("Innovation MAD window must be at least 30 periods.")

    q = pd.to_numeric(variance_proxy, errors="coerce").where(lambda item: item > 0.0)
    output = pd.DataFrame(index=q.index)
    output["variance_proxy"] = q
    output["actual_log_variance"] = np.log(q + EPSILON)
    for column in [
        "forecast_variance",
        "forecast_log_variance",
        "innovation",
        "innovation_scale",
        "shock_score",
        "har_const",
        "har_daily",
        "har_weekly",
        "har_monthly",
        "har_smearing",
    ]:
        output[column] = np.nan

    model: HarModel | None = None
    residual_history: list[float] = []
    for position in range(len(q)):
        if position < initial_training:
            continue
        if model is None or (position - initial_training) % refit_every == 0:
            model = fit_har_model(
                q.iloc[:position],
                minimum_observations=max(120, min(initial_training, position // 2)),
                robust=robust,
            )
        if model is None:
            continue

        history = q.iloc[:position].dropna().tolist()
        forecast = predict_next_variance(history, model)
        actual = q.iloc[position]
        if not np.isfinite(forecast) or not np.isfinite(actual) or actual <= 0.0:
            continue

        forecast_log = float(np.log(forecast + EPSILON))
        innovation = float(np.log(actual + EPSILON) - forecast_log)
        output.iloc[position, output.columns.get_loc("forecast_variance")] = forecast
        output.iloc[position, output.columns.get_loc("forecast_log_variance")] = forecast_log
        output.iloc[position, output.columns.get_loc("innovation")] = innovation
        output.iloc[position, output.columns.get_loc("har_const")] = model.params.get("const", np.nan)
        output.iloc[position, output.columns.get_loc("har_daily")] = model.params.get("log_daily", np.nan)
        output.iloc[position, output.columns.get_loc("har_weekly")] = model.params.get("log_weekly", np.nan)
        output.iloc[position, output.columns.get_loc("har_monthly")] = model.params.get("log_monthly", np.nan)
        output.iloc[position, output.columns.get_loc("har_smearing")] = model.smearing

        if len(residual_history) >= 30:
            recent = np.asarray(residual_history[-mad_window:], dtype=float)
            median = float(np.nanmedian(recent))
            mad = float(np.nanmedian(np.abs(recent - median)))
            scale = 1.4826 * mad
            if np.isfinite(scale) and scale > 1e-8:
                output.iloc[position, output.columns.get_loc("innovation_scale")] = scale
                output.iloc[position, output.columns.get_loc("shock_score")] = (
                    innovation - median
                ) / scale
        residual_history.append(innovation)

    return output


def model_from_diagnostic_row(row: pd.Series) -> HarModel | None:
    values = {
        "const": row.get("har_const"),
        "log_daily": row.get("har_daily"),
        "log_weekly": row.get("har_weekly"),
        "log_monthly": row.get("har_monthly"),
    }
    if any(not np.isfinite(value) for value in values.values()):
        return None
    smearing = float(row.get("har_smearing", 1.0))
    if not np.isfinite(smearing) or smearing <= 0.0:
        smearing = 1.0
    return HarModel(params=pd.Series(values, dtype=float), smearing=smearing, nobs=0, robust=True)
