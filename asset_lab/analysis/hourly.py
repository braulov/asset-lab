from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize_scalar, nnls
from scipy.stats import skew


REQUIRED_CANDLE_COLUMNS = {
    "begin",
    "open",
    "close",
    "high",
    "low",
    "volume",
}


@dataclass(frozen=True)
class MobilityFit:
    model: str
    parameters: dict[str, float]
    r_squared: float
    rmse: float
    sample_size: int

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["parameters"] = ", ".join(
            f"{key}={value:.5g}" for key, value in self.parameters.items()
        )
        return record


@dataclass(frozen=True)
class ProcessFit:
    model: str
    parameters: dict[str, float]
    rmse: float
    sse: float
    sample_size: int

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["parameters"] = ", ".join(
            f"{key}={value:.5g}" for key, value in self.parameters.items()
        )
        return record


def _robust_scale(values: pd.Series) -> float:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return float("nan")
    centre = float(finite.median())
    mad = float((finite - centre).abs().median())
    if mad > 1e-12:
        return mad / 0.67448975
    standard_deviation = float(finite.std(ddof=0))
    return standard_deviation if standard_deviation > 1e-12 else 1.0


def prepare_hourly_candles(
    candles: pd.DataFrame,
    *,
    regular_start_hour: int = 10,
    regular_end_hour: int = 18,
) -> pd.DataFrame:
    """Clean MOEX hourly candles and add within-candle returns and session fields."""
    missing = REQUIRED_CANDLE_COLUMNS.difference(candles.columns)
    if missing:
        raise ValueError(f"Missing hourly candle columns: {sorted(missing)}")

    frame = candles.copy()
    frame["begin"] = pd.to_datetime(frame["begin"], errors="coerce")
    if "end" in frame:
        frame["end"] = pd.to_datetime(frame["end"], errors="coerce")

    numeric_columns = ["open", "close", "high", "low", "volume"]
    if "value" in frame:
        numeric_columns.append("value")
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.loc[
        frame["begin"].notna()
        & (frame["open"] > 0)
        & (frame["close"] > 0)
        & (frame["high"] > 0)
        & (frame["low"] > 0)
        & (frame["volume"] > 0)
    ].copy()
    frame = frame.sort_values("begin").drop_duplicates("begin").reset_index(drop=True)

    frame["date"] = frame["begin"].dt.normalize()
    frame["year"] = frame["begin"].dt.year
    frame["hour"] = frame["begin"].dt.hour
    frame["log_return_oc"] = np.log(frame["close"] / frame["open"])
    frame["price_increment"] = frame["close"] - frame["open"]
    frame["price"] = frame["open"]
    frame["regular_session"] = frame["hour"].between(
        regular_start_hour,
        regular_end_hour,
    )
    return frame


def past_hourly_standardisation(
    frame: pd.DataFrame,
    *,
    scale_window: int = 252,
    minimum_periods: int = 80,
    clip: float = 8.0,
) -> pd.DataFrame:
    """Past-only hour-of-day standardisation for intraday returns.

    The first scale is a shifted rolling MAD of |return|. A shifted robust RMS
    correction then makes the central conditional variance closer to one.
    """
    work = frame.copy().sort_values("begin").reset_index(drop=True)
    work["absolute_return"] = work["log_return_oc"].abs()
    work["scale_mad"] = np.nan

    fallback: dict[int, float] = {}
    for hour, indices in work.groupby("hour").groups.items():
        values = work.loc[indices, "absolute_return"]
        scale = values.rolling(scale_window, min_periods=minimum_periods).median().shift(1)
        scale = scale / 0.67448975
        work.loc[indices, "scale_mad"] = scale.to_numpy()
        fallback[int(hour)] = max(float(values.median() / 0.67448975), 1e-8)

    missing = work["scale_mad"].isna()
    if missing.any():
        work.loc[missing, "scale_mad"] = [
            fallback.get(int(hour), 1e-4) for hour in work.loc[missing, "hour"]
        ]

    work["preliminary_z"] = work["log_return_oc"] / work["scale_mad"]
    work["rms_correction"] = np.nan
    for _, indices in work.groupby("hour").groups.items():
        correction = (
            work.loc[indices, "preliminary_z"]
            .clip(-5.0, 5.0)
            .pow(2)
            .rolling(scale_window, min_periods=minimum_periods)
            .mean()
            .shift(1)
            .pow(0.5)
        )
        work.loc[indices, "rms_correction"] = correction.to_numpy()

    work["rms_correction"] = work["rms_correction"].fillna(1.0).clip(0.5, 3.0)
    work["conditional_scale"] = work["scale_mad"] * work["rms_correction"]
    work["z_return"] = (work["log_return_oc"] / work["conditional_scale"]).clip(
        -clip,
        clip,
    )
    work["m2_instant"] = work["z_return"].pow(2)
    work["m2_clipped"] = work["m2_instant"].clip(upper=36.0)
    work["m3_instant"] = work["z_return"].clip(-6.0, 6.0).pow(3)
    work["signed_m2"] = work["z_return"] * work["z_return"].abs()
    return work


def rolling_standardised_moments(
    frame: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    z = frame["z_return"].astype(float)
    minimum = max(10, window // 2)
    mean = z.rolling(window, min_periods=minimum).mean()
    centred = z - mean
    m2 = centred.pow(2).rolling(window, min_periods=minimum).mean()
    m3 = centred.pow(3).rolling(window, min_periods=minimum).mean()
    skewness = m3 / m2.pow(1.5).replace(0.0, np.nan)
    downside = z.where(z < 0.0, 0.0).pow(2).rolling(window, min_periods=minimum).mean()
    upside = z.where(z > 0.0, 0.0).pow(2).rolling(window, min_periods=minimum).mean()
    total = downside + upside
    return pd.DataFrame(
        {
            "m2": m2,
            "m3": m3,
            "skewness": skewness,
            "signed_m2": frame["signed_m2"].rolling(window, min_periods=minimum).mean(),
            "downside_share": downside / total.replace(0.0, np.nan),
        },
        index=frame.index,
    )


def session_moment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for hour, group in frame.groupby("hour"):
        returns = group["log_return_oc"].dropna()
        if len(returns) < 40:
            continue
        centre = float(returns.median())
        scale = _robust_scale(returns - centre)
        standardised = ((returns - centre) / scale).to_numpy(dtype=float)
        low, high = np.quantile(standardised, [0.005, 0.995])
        standardised = np.clip(standardised, low, high)
        rows.append(
            {
                "hour": int(hour),
                "observations": int(len(standardised)),
                "mean": float(np.mean(standardised)),
                "standard_deviation": float(np.std(standardised)),
                "skewness": float(skew(standardised, bias=False)),
                "m3": float(np.mean((standardised - np.mean(standardised)) ** 3)),
            }
        )
    return pd.DataFrame(rows).sort_values("hour").reset_index(drop=True)


def overnight_moment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.groupby("date").agg(
        first_open=("open", "first"),
        last_close=("close", "last"),
    )
    daily["overnight_return"] = np.log(daily["first_open"] / daily["last_close"].shift(1))

    filters = {
        "all": pd.Series(True, index=daily.index),
        "exclude_2022": daily.index.year != 2022,
        "absolute_gap_below_20pct": daily["overnight_return"].abs() < np.log(1.2),
        "exclude_2022_and_20pct": (
            (daily.index.year != 2022)
            & (daily["overnight_return"].abs() < np.log(1.2))
        ),
    }
    rows = []
    for label, mask in filters.items():
        values = daily.loc[mask, "overnight_return"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < 20:
            continue
        centre = float(values.median())
        scale = _robust_scale(values - centre)
        standardised = ((values - centre) / scale).to_numpy(dtype=float)
        low, high = np.quantile(standardised, [0.005, 0.995])
        standardised = np.clip(standardised, low, high)
        rows.append(
            {
                "filter": label,
                "observations": int(len(standardised)),
                "skewness": float(skew(standardised, bias=False)),
                "m3": float(np.mean((standardised - np.mean(standardised)) ** 3)),
                "mean": float(np.mean(standardised)),
            }
        )
    return pd.DataFrame(rows)


def mobility_price_bins(
    frame: pd.DataFrame,
    *,
    bins: int = 8,
    minimum_bucket_size: int = 40,
) -> pd.DataFrame:
    work = frame.loc[frame["regular_session"]].copy()
    overall_median = float(work["log_return_oc"].abs().median())
    hour_factor = work.groupby("hour")["log_return_oc"].apply(
        lambda values: max(float(values.abs().median() / overall_median), 0.2)
        if overall_median > 0
        else 1.0
    )
    work["hour_factor"] = work["hour"].map(hour_factor)
    work["adjusted_increment"] = work["price_increment"] / work["hour_factor"]

    rows: list[dict[str, float | int]] = []
    for (year,), group in work.groupby(["year"]):
        if len(group) < max(400, bins * minimum_bucket_size):
            continue
        lower, upper = group["adjusted_increment"].quantile([0.005, 0.995])
        trimmed = group.loc[group["adjusted_increment"].between(lower, upper)].copy()
        try:
            trimmed["price_bin"] = pd.qcut(trimmed["price"], bins, duplicates="drop")
        except ValueError:
            continue
        for _, bucket in trimmed.groupby("price_bin", observed=True):
            if len(bucket) < minimum_bucket_size:
                continue
            increments = bucket["adjusted_increment"].to_numpy(dtype=float)
            increments = increments - np.mean(increments)
            variance = float(np.mean(increments**2))
            third = float(np.mean(increments**3))
            rows.append(
                {
                    "year": int(year),
                    "price": float(bucket["price"].mean()),
                    "observations": int(len(bucket)),
                    "mobility": math.sqrt(max(variance, 0.0)),
                    "m2_price": variance,
                    "m3_price": third,
                    "skew_price": third / variance**1.5 if variance > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _fit_mobility_parameters(frame: pd.DataFrame, model: str) -> np.ndarray:
    price = frame["price"].to_numpy(dtype=float)
    mobility = frame["mobility"].to_numpy(dtype=float)
    weights = np.sqrt(frame["observations"].to_numpy(dtype=float))
    if model == "constant":
        return np.array([np.average(mobility, weights=weights**2)])
    if model == "proportional":
        design = price[:, None]
    elif model == "affine":
        design = np.column_stack([np.ones(len(price)), price])
    else:
        raise ValueError(f"Unknown mobility model: {model}")
    result = lsq_linear(
        design * weights[:, None],
        mobility * weights,
        bounds=(0.0, np.inf),
    )
    return result.x


def _mobility_prediction(parameters: np.ndarray, price: np.ndarray, model: str) -> np.ndarray:
    if model == "constant":
        return np.full(len(price), parameters[0])
    if model == "proportional":
        return parameters[0] * price
    return parameters[0] + parameters[1] * price


def fit_mobility_models(
    bins: pd.DataFrame,
) -> tuple[list[MobilityFit], pd.DataFrame]:
    if bins.empty:
        return [], pd.DataFrame()

    fits: list[MobilityFit] = []
    models = ["constant", "proportional", "affine"]
    weighted_mean = np.average(bins["mobility"], weights=bins["observations"])
    for model in models:
        parameters = _fit_mobility_parameters(bins, model)
        prediction = _mobility_prediction(parameters, bins["price"].to_numpy(float), model)
        residual = bins["mobility"].to_numpy(float) - prediction
        weights = bins["observations"].to_numpy(float)
        sse = float(np.sum(weights * residual**2))
        sst = float(np.sum(weights * (bins["mobility"] - weighted_mean) ** 2))
        parameter_names = {
            "constant": ["a"],
            "proportional": ["b"],
            "affine": ["a", "b"],
        }[model]
        fits.append(
            MobilityFit(
                model=model,
                parameters=dict(zip(parameter_names, map(float, parameters), strict=True)),
                r_squared=float(1.0 - sse / sst) if sst > 0 else float("nan"),
                rmse=float(math.sqrt(sse / max(float(weights.sum()), 1.0))),
                sample_size=int(len(bins)),
            )
        )

    years = sorted(bins["year"].unique())
    rows: list[dict[str, float | int | str]] = []
    if len(years) >= 2:
        for year in years:
            train = bins.loc[bins["year"] != year]
            test = bins.loc[bins["year"] == year]
            if len(train) < 12 or len(test) < 3:
                continue
            for model in models:
                parameters = _fit_mobility_parameters(train, model)
                prediction = _mobility_prediction(parameters, test["price"].to_numpy(float), model)
                rmse = math.sqrt(
                    np.average(
                        (test["mobility"].to_numpy(float) - prediction) ** 2,
                        weights=test["observations"],
                    )
                )
                normaliser = np.average(test["mobility"], weights=test["observations"])
                rows.append(
                    {
                        "held_out_year": int(year),
                        "model": model,
                        "normalised_rmse": float(rmse / normaliser) if normaliser > 0 else np.nan,
                    }
                )
    cv = pd.DataFrame(rows)
    if not cv.empty:
        summary = cv.groupby("model")["normalised_rmse"].agg(["mean", "median", "std"]).reset_index()
        summary["winner"] = summary["mean"] == summary["mean"].min()
        return fits, summary.sort_values("mean").reset_index(drop=True)
    return fits, cv


def affine_mobility_skewness(b: float, horizon: np.ndarray | float) -> np.ndarray:
    horizon_array = np.asarray(horizon, dtype=float)
    if b <= 1e-12:
        return np.zeros_like(horizon_array)
    exponential = np.exp(b * b * horizon_array)
    return (exponential + 2.0) * np.sqrt(np.maximum(exponential - 1.0, 0.0))


def affine_m3_horizon_test(
    frame: pd.DataFrame,
    affine_fit: MobilityFit,
    *,
    horizons: Iterable[int] = (1, 2, 4, 6, 8),
) -> pd.DataFrame:
    if affine_fit.model != "affine":
        raise ValueError("affine_m3_horizon_test requires the affine fit")
    a = float(affine_fit.parameters["a"])
    b = float(affine_fit.parameters["b"])
    work = frame.loc[frame["regular_session"]].sort_values("begin").reset_index(drop=True)
    rows = []
    for horizon in horizons:
        ending_close = work["close"].shift(-(horizon - 1))
        ending_time = work["begin"].shift(-(horizon - 1))
        valid = (
            ((ending_time - work["begin"]) == pd.Timedelta(hours=horizon - 1))
            & (ending_time.dt.normalize() == work["begin"].dt.normalize())
        )
        increments = ending_close.loc[valid].to_numpy(float) - work.loc[valid, "open"].to_numpy(float)
        starting_price = work.loc[valid, "price"].to_numpy(float)
        mobility = a + b * starting_price
        if b > 1e-12:
            conditional_sd = mobility / b * np.sqrt(np.exp(b * b * horizon) - 1.0)
        else:
            conditional_sd = mobility * math.sqrt(horizon)
        residual = increments / conditional_sd
        residual = residual[np.isfinite(residual)]
        if len(residual) < 50:
            continue
        lower, upper = np.quantile(residual, [0.005, 0.995])
        winsorised = np.clip(residual, lower, upper)
        rows.append(
            {
                "horizon_hours": int(horizon),
                "observations": int(len(residual)),
                "predicted_affine_skewness": float(affine_mobility_skewness(b, horizon)),
                "observed_skewness": float(skew(winsorised, bias=False)),
            }
        )
    return pd.DataFrame(rows)


def detect_moment_shocks(
    frame: pd.DataFrame,
    *,
    quantile: float = 0.99,
    cooldown: int = 12,
    precursor_window: int = 12,
    post_horizon: int = 24,
    synchrony: pd.Series | None = None,
    marketwide_threshold: float = 0.40,
    isolated_threshold: float = 0.20,
) -> pd.DataFrame:
    work = frame.loc[frame["regular_session"]].sort_values("begin").reset_index(drop=True)
    threshold = float(work["z_return"].abs().quantile(quantile))
    candidates = np.flatnonzero(work["z_return"].abs().to_numpy(float) >= threshold)
    rows = []
    last_position = -10**9
    for event_number, position in enumerate(candidates):
        if position - last_position < cooldown:
            continue
        if position < max(48, precursor_window) or position + post_horizon >= len(work):
            continue
        last_position = int(position)
        timestamp = pd.Timestamp(work.loc[position, "begin"])
        rows.append(
            {
                "event_id": event_number,
                "position": int(position),
                "timestamp": timestamp,
                "z_event": float(work.loc[position, "z_return"]),
                "direction": "negative" if work.loc[position, "z_return"] < 0 else "positive",
                "precursor_m2": float(
                    work["m2_clipped"].iloc[position - precursor_window : position].mean()
                ),
                "far_m2": float(work["m2_clipped"].iloc[position - 48 : position - 24].mean()),
                "synchrony": (
                    float(synchrony.get(timestamp, np.nan)) if synchrony is not None else np.nan
                ),
                "threshold": threshold,
            }
        )
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events["precursor_rank"] = events["precursor_m2"].rank(pct=True, method="average")
    events["heating_class"] = np.select(
        [events["precursor_rank"] <= 0.40, events["precursor_rank"] >= 0.60],
        ["abrupt-like", "preheated-like"],
        default="intermediate",
    )
    if synchrony is None:
        events["synchrony_class"] = "not available"
        events["process_class"] = events["heating_class"]
    else:
        events["synchrony_class"] = np.select(
            [
                events["synchrony"] >= marketwide_threshold,
                events["synchrony"] <= isolated_threshold,
            ],
            ["market-wide", "isolated"],
            default="intermediate",
        )
        events["process_class"] = np.select(
            [
                (events["heating_class"] == "abrupt-like")
                & (events["synchrony_class"] == "market-wide"),
                (events["heating_class"] == "preheated-like")
                & (events["synchrony_class"] == "isolated"),
            ],
            ["external-like", "internal-like"],
            default="other",
        )
    return events


def build_matched_moment_trajectories(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    *,
    controls_per_event: int = 8,
    pre_horizon: int = 12,
    post_horizon: int = 24,
    exclusion_radius: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = frame.loc[frame["regular_session"]].sort_values("begin").reset_index(drop=True).copy()
    work["pre_m2"] = work["m2_clipped"].rolling(12, min_periods=8).mean().shift(1)
    work["far_m2"] = work["m2_clipped"].rolling(24, min_periods=16).mean().shift(25)
    work["return_12"] = work["log_return_oc"].rolling(12, min_periods=8).sum().shift(1)
    work["log_pre_m2"] = np.log1p(work["pre_m2"].clip(lower=0.0))
    work["log_far_m2"] = np.log1p(work["far_m2"].clip(lower=0.0))
    work["scaled_return_12"] = work["return_12"] / np.sqrt(work["far_m2"].clip(lower=0.1))
    features = ["log_pre_m2", "log_far_m2", "scaled_return_12"]

    forbidden = np.zeros(len(work), dtype=bool)
    for position in events["position"].astype(int):
        forbidden[
            max(0, position - exclusion_radius) : min(len(work), position + post_horizon + 1)
        ] = True
    control_cap = float(work["z_return"].abs().quantile(0.95))
    candidates = work.loc[
        (~forbidden)
        & (work["z_return"].abs() < control_cap)
        & (work.index >= max(48, pre_horizon))
        & (work.index < len(work) - post_horizon)
    ].dropna(subset=features)
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()

    centre = candidates[features].median()
    scale = (candidates[features] - centre).abs().median().replace(0.0, 1.0)
    standardised_candidates = ((candidates[features] - centre) / scale).to_numpy(float)
    candidate_positions = candidates.index.to_numpy(int)
    candidate_hours = candidates["hour"].to_numpy(int)
    candidate_times = candidates["begin"].to_numpy()

    trajectory_rows = []
    match_rows = []
    for event in events.itertuples(index=False):
        position = int(event.position)
        event_features = work.loc[position, features]
        if event_features.isna().any():
            continue
        standardised_event = ((event_features - centre) / scale).to_numpy(float)
        distance = np.sum((standardised_candidates - standardised_event) ** 2, axis=1)
        distance[candidate_hours != int(work.loc[position, "hour"])] = np.inf
        calendar_distance = np.abs(
            (candidate_times - np.datetime64(work.loc[position, "begin"])) / np.timedelta64(1, "D")
        )
        distance += 0.03 * calendar_distance / 365.0
        ordering = np.argsort(distance)
        selected = ordering[np.isfinite(distance[ordering])][:controls_per_event]
        if len(selected) < max(3, controls_per_event // 2):
            continue
        controls = candidate_positions[selected]
        baseline_values = []
        for control in controls:
            baseline_values.extend(work["m2_clipped"].iloc[control - pre_horizon : control])
        baseline = max(float(np.mean(baseline_values)), 0.2)
        match_rows.append(
            {
                "event_id": int(event.event_id),
                "controls": int(len(controls)),
                "mean_distance": float(np.mean(distance[selected])),
            }
        )
        for offset in range(-pre_horizon, post_horizon + 1):
            control_m2 = work.loc[controls + offset, "m2_clipped"].to_numpy(float)
            control_m3 = work.loc[controls + offset, "m3_instant"].to_numpy(float)
            trajectory_rows.append(
                {
                    "event_id": int(event.event_id),
                    "timestamp": event.timestamp,
                    "direction": event.direction,
                    "heating_class": event.heating_class,
                    "synchrony_class": event.synchrony_class,
                    "process_class": event.process_class,
                    "offset": int(offset),
                    "abnormal_m2": float(
                        (work.loc[position + offset, "m2_clipped"] - np.median(control_m2))
                        / baseline
                    ),
                    "abnormal_m3": float(
                        (work.loc[position + offset, "m3_instant"] - np.mean(control_m3))
                        / baseline**1.5
                    ),
                }
            )
    return pd.DataFrame(trajectory_rows), pd.DataFrame(match_rows)


def aggregate_moment_trajectories(
    trajectories: pd.DataFrame,
    *,
    bootstrap_samples: int = 300,
    seed: int = 2026,
) -> pd.DataFrame:
    if trajectories.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    group_columns = ["process_class", "direction"]
    for keys, group in trajectories.groupby(group_columns):
        process_class, direction = keys
        event_ids = group["event_id"].unique()
        pivot_m2 = group.pivot(index="event_id", columns="offset", values="abnormal_m2")
        pivot_m3 = group.pivot(index="event_id", columns="offset", values="abnormal_m3")
        offsets = sorted(set(pivot_m2.columns).intersection(pivot_m3.columns))
        for offset in offsets:
            m2_values = pivot_m2[offset].dropna().to_numpy(float)
            m3_values = pivot_m3[offset].dropna().to_numpy(float)
            if len(m2_values) == 0:
                continue
            bootstrap_m2 = []
            bootstrap_m3 = []
            for _ in range(bootstrap_samples):
                sampled = rng.choice(event_ids, len(event_ids), replace=True)
                sample_m2 = pivot_m2.reindex(sampled)[offset].dropna().to_numpy(float)
                sample_m3 = pivot_m3.reindex(sampled)[offset].dropna().to_numpy(float)
                if len(sample_m2):
                    bootstrap_m2.append(float(np.median(sample_m2)))
                if len(sample_m3):
                    bootstrap_m3.append(float(np.mean(sample_m3)))
            rows.append(
                {
                    "process_class": process_class,
                    "direction": direction,
                    "offset": int(offset),
                    "events": int(len(m2_values)),
                    "median_abnormal_m2": float(np.median(m2_values)),
                    "m2_ci_low": float(np.quantile(bootstrap_m2, 0.025)),
                    "m2_ci_high": float(np.quantile(bootstrap_m2, 0.975)),
                    "mean_abnormal_m3": float(np.mean(m3_values)),
                    "m3_ci_low": float(np.quantile(bootstrap_m3, 0.025)),
                    "m3_ci_high": float(np.quantile(bootstrap_m3, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def _fit_process_curve(times: np.ndarray, response: np.ndarray, model: str) -> ProcessFit:
    times = np.asarray(times, dtype=float)
    response = np.asarray(response, dtype=float)
    tau_grid = np.geomspace(0.15, 120.0, 120)
    best: tuple[float, dict[str, float], np.ndarray] | None = None

    if model == "exponential":
        for tau in tau_grid:
            design = np.column_stack([np.ones_like(times), np.exp(-times / tau)])
            coefficients, _ = nnls(design, response)
            prediction = design @ coefficients
            sse = float(np.sum((response - prediction) ** 2))
            parameters = {
                "floor": float(coefficients[0]),
                "amplitude": float(coefficients[1]),
                "tau": float(tau),
            }
            if best is None or sse < best[0]:
                best = (sse, parameters, prediction)
    elif model == "constrained_mobility":
        for tau in tau_grid:
            slow = np.exp(-times / tau)
            fast = np.exp(-2.0 * times / tau)

            def objective(q: float) -> float:
                shape = 2.0 * q * slow + q * q * fast
                floor = max(0.0, float(np.mean(response - shape)))
                return float(np.sum((response - floor - shape) ** 2))

            optimum = minimize_scalar(objective, bounds=(0.0, 10.0), method="bounded")
            q = float(optimum.x)
            shape = 2.0 * q * slow + q * q * fast
            floor = max(0.0, float(np.mean(response - shape)))
            prediction = floor + shape
            parameters = {"floor": floor, "q": q, "tau": float(tau)}
            if best is None or float(optimum.fun) < best[0]:
                best = (float(optimum.fun), parameters, prediction)
    elif model == "stress_fed":
        for tau_fast in np.geomspace(0.1, 10.0, 28):
            for tau_slow in np.geomspace(2.0, 150.0, 36):
                if tau_slow <= 1.25 * tau_fast:
                    continue
                design = np.column_stack(
                    [
                        np.ones_like(times),
                        np.exp(-times / tau_fast),
                        np.exp(-times / tau_slow),
                    ]
                )
                coefficients, _ = nnls(design, response)
                prediction = design @ coefficients
                sse = float(np.sum((response - prediction) ** 2))
                parameters = {
                    "floor": float(coefficients[0]),
                    "amplitude_fast": float(coefficients[1]),
                    "tau_fast": float(tau_fast),
                    "amplitude_slow": float(coefficients[2]),
                    "tau_slow": float(tau_slow),
                }
                if best is None or sse < best[0]:
                    best = (sse, parameters, prediction)
    elif model == "shifted_power":
        for exponent in np.linspace(0.2, 3.2, 45):
            for tau in np.geomspace(0.1, 120.0, 55):
                basis = np.power(1.0 + times / tau, -exponent)
                design = np.column_stack([np.ones_like(times), basis])
                coefficients, _ = nnls(design, response)
                prediction = design @ coefficients
                sse = float(np.sum((response - prediction) ** 2))
                parameters = {
                    "floor": float(coefficients[0]),
                    "amplitude": float(coefficients[1]),
                    "tau": float(tau),
                    "exponent": float(exponent),
                }
                if best is None or sse < best[0]:
                    best = (sse, parameters, prediction)
    else:
        raise ValueError(model)

    assert best is not None
    return ProcessFit(
        model=model,
        parameters=best[1],
        rmse=float(math.sqrt(best[0] / len(response))),
        sse=float(best[0]),
        sample_size=int(len(response)),
    )


def process_curve_prediction(model: str, times: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    floor = parameters.get("floor", 0.0)
    if model == "exponential":
        return floor + parameters["amplitude"] * np.exp(-times / parameters["tau"])
    if model == "constrained_mobility":
        slow = np.exp(-times / parameters["tau"])
        return floor + 2.0 * parameters["q"] * slow + parameters["q"] ** 2 * slow**2
    if model == "stress_fed":
        return (
            floor
            + parameters["amplitude_fast"] * np.exp(-times / parameters["tau_fast"])
            + parameters["amplitude_slow"] * np.exp(-times / parameters["tau_slow"])
        )
    if model == "shifted_power":
        return floor + parameters["amplitude"] * np.power(
            1.0 + times / parameters["tau"],
            -parameters["exponent"],
        )
    raise ValueError(model)


def fit_process_models(
    aggregate: pd.DataFrame,
    *,
    process_class: str,
    direction: str = "negative",
    fit_start: int = 1,
    fit_end: int = 24,
) -> list[ProcessFit]:
    selected = aggregate.loc[
        (aggregate["process_class"] == process_class)
        & (aggregate["direction"] == direction)
        & aggregate["offset"].between(fit_start, fit_end)
    ].dropna(subset=["median_abnormal_m2"])
    if len(selected) < 6:
        return []
    response = selected["median_abnormal_m2"].clip(lower=0.0).to_numpy(float)
    times = selected["offset"].to_numpy(float) - fit_start + 1.0
    models = ["exponential", "constrained_mobility", "stress_fed", "shifted_power"]
    return sorted(
        [_fit_process_curve(times, response, model) for model in models],
        key=lambda fit: fit.rmse,
    )


def cross_validate_process_models(
    trajectories: pd.DataFrame,
    *,
    process_class: str,
    direction: str = "negative",
    fit_start: int = 1,
    fit_end: int = 24,
    folds: int = 5,
    seed: int = 2026,
) -> pd.DataFrame:
    selected = trajectories.loc[
        (trajectories["process_class"] == process_class)
        & (trajectories["direction"] == direction)
        & trajectories["offset"].between(fit_start, fit_end)
    ].copy()
    event_ids = selected["event_id"].unique()
    if len(event_ids) < max(10, folds * 2):
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rng.shuffle(event_ids)
    fold_ids = np.array_split(event_ids, folds)
    rows = []
    for fold_number, held_out in enumerate(fold_ids, start=1):
        training = selected.loc[~selected["event_id"].isin(held_out)]
        testing = selected.loc[selected["event_id"].isin(held_out)]
        training_curve = (
            training.groupby("offset")["abnormal_m2"].median().clip(lower=0.0).sort_index()
        )
        times = training_curve.index.to_numpy(float) - fit_start + 1.0
        for model in ["exponential", "constrained_mobility", "stress_fed", "shifted_power"]:
            fit = _fit_process_curve(times, training_curve.to_numpy(float), model)
            held_out_errors = []
            for _, event_curve in testing.groupby("event_id"):
                event_curve = event_curve.sort_values("offset")
                event_times = event_curve["offset"].to_numpy(float) - fit_start + 1.0
                prediction = process_curve_prediction(model, event_times, fit.parameters)
                held_out_errors.extend((event_curve["abnormal_m2"].to_numpy(float) - prediction) ** 2)
            rows.append(
                {
                    "fold": fold_number,
                    "model": model,
                    "held_out_rmse": float(math.sqrt(np.mean(held_out_errors))),
                }
            )
    fold_frame = pd.DataFrame(rows)
    summary = (
        fold_frame.groupby("model")["held_out_rmse"]
        .agg(["mean", "median", "std"])
        .reset_index()
        .sort_values("mean")
    )
    return summary


def panel_synchrony(
    standardised_returns: pd.DataFrame,
    *,
    tail_quantile: float = 0.95,
) -> pd.Series:
    thresholds = standardised_returns.abs().quantile(tail_quantile)
    return standardised_returns.abs().gt(thresholds, axis=1).mean(axis=1)


def market_moment_frame(standardised_returns: pd.DataFrame) -> pd.DataFrame:
    z = standardised_returns.clip(-6.0, 6.0)
    return pd.DataFrame(
        {
            "market_m2": z.pow(2).median(axis=1),
            "market_m3": z.pow(3).mean(axis=1),
            "negative_share": (z < 0.0).mean(axis=1),
            "synchrony_95": panel_synchrony(z),
        }
    ).sort_index()
