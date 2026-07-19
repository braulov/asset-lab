from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class FitResult:
    model: str
    parameters: dict[str, float]
    sse: float
    aic: float
    bic: float
    success: bool
    fit_start: int | None = None
    fit_end: int | None = None
    sample_size: int = 0
    error: str | None = None

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["parameters"] = ", ".join(
            f"{key}={value:.4g}" for key, value in self.parameters.items()
        )
        return record


@dataclass(frozen=True)
class FitDecision:
    allowed: bool
    reason: str
    fit_start: int | None = None
    fit_end: int | None = None


def exponential(t: np.ndarray, amplitude: float, tau: float) -> np.ndarray:
    return amplitude * np.exp(-t / tau)


def stretched_exponential(
    t: np.ndarray,
    amplitude: float,
    tau: float,
    beta: float,
) -> np.ndarray:
    return amplitude * np.exp(-np.power(t / tau, beta))


def double_exponential(
    t: np.ndarray,
    amplitude_fast: float,
    tau_fast: float,
    amplitude_slow: float,
    tau_slow: float,
) -> np.ndarray:
    return amplitude_fast * np.exp(-t / tau_fast) + amplitude_slow * np.exp(-t / tau_slow)


def shifted_power_law(
    t: np.ndarray,
    amplitude: float,
    shift: float,
    exponent: float,
) -> np.ndarray:
    return amplitude * np.power(t + shift, -exponent)


ModelSpec = tuple[
    Callable[..., np.ndarray],
    list[str],
    list[float],
    tuple[list[float], list[float]],
]


MODEL_SPECS: dict[str, ModelSpec] = {
    "exponential": (
        exponential,
        ["amplitude", "tau"],
        [0.2, 3.0],
        ([0.0, 0.05], [10.0, 1000.0]),
    ),
    "stretched_exponential": (
        stretched_exponential,
        ["amplitude", "tau", "beta"],
        [0.2, 3.0, 0.7],
        ([0.0, 0.05, 0.05], [10.0, 1000.0, 1.0]),
    ),
    "double_exponential": (
        double_exponential,
        ["amplitude_fast", "tau_fast", "amplitude_slow", "tau_slow"],
        [0.12, 1.5, 0.08, 10.0],
        ([0.0, 0.05, 0.0, 0.2], [10.0, 100.0, 10.0, 2000.0]),
    ),
    "shifted_power_law": (
        shifted_power_law,
        ["amplitude", "shift", "exponent"],
        [0.3, 1.0, 0.8],
        ([0.0, 0.05, 0.01], [20.0, 100.0, 10.0]),
    ),
}


def _information_criteria(
    sse: float,
    sample_size: int,
    parameter_count: int,
) -> tuple[float, float]:
    safe_sse = max(float(sse), np.finfo(float).tiny)
    aic = sample_size * np.log(safe_sse / sample_size) + 2 * parameter_count
    bic = sample_size * np.log(safe_sse / sample_size) + parameter_count * np.log(sample_size)
    return float(aic), float(bic)


def relaxation_fit_decision(
    aggregate: pd.DataFrame,
    *,
    minimum_events: int = 20,
    early_periods: int = 5,
    minimum_points: int = 5,
    nonpositive_run: int = 3,
) -> FitDecision:
    if aggregate.empty:
        return FitDecision(False, "No aggregated response is available.")
    required = {"offset", "median", "bootstrap_ci_low", "count"}
    if not required.issubset(aggregate.columns):
        return FitDecision(False, "The aggregate lacks bootstrap response columns.")

    post = aggregate.sort_values("offset").dropna(subset=["median"])
    if post.empty or int(post["count"].min()) < minimum_events:
        return FitDecision(False, f"Fewer than {minimum_events} events support the response.")

    early = post.loc[post["offset"] <= early_periods]
    if early.empty:
        return FitDecision(False, "No early post-shock response is available.")
    peak_row = early.loc[early["median"].idxmax()]
    peak_value = float(peak_row["median"])
    if peak_value <= 0.0:
        return FitDecision(False, "No positive early abnormal-volatility response was detected.")
    if not bool((early["bootstrap_ci_low"] > 0.0).any()):
        return FitDecision(
            False,
            "The early event-bootstrap interval does not rise above zero.",
        )

    fit_start = int(peak_row["offset"])
    decay = post.loc[post["offset"] >= fit_start].reset_index(drop=True)
    stop = len(decay)
    values = decay["median"].to_numpy(dtype=float)
    for position in range(minimum_points, len(decay) - nonpositive_run + 1):
        if np.all(values[position : position + nonpositive_run] <= 0.0):
            stop = position
            break
    segment = decay.iloc[:stop]
    if len(segment) < minimum_points:
        return FitDecision(False, "Too few positive-decay points remain after the early peak.")
    return FitDecision(
        True,
        "A positive early response with bootstrap support is followed by an identifiable decay segment.",
        fit_start=fit_start,
        fit_end=int(segment["offset"].iloc[-1]),
    )


def fit_relaxation_models(
    aggregate: pd.DataFrame,
    target_column: str = "median",
    *,
    minimum_events: int = 20,
) -> tuple[list[FitResult], FitDecision]:
    decision = relaxation_fit_decision(aggregate, minimum_events=minimum_events)
    if not decision.allowed or decision.fit_start is None or decision.fit_end is None:
        return [], decision

    segment = aggregate.loc[
        (aggregate["offset"] >= decision.fit_start)
        & (aggregate["offset"] <= decision.fit_end)
    ].dropna(subset=[target_column]).copy()
    segment["fit_time"] = segment["offset"] - decision.fit_start
    t = segment["fit_time"].to_numpy(dtype=float)
    y = segment[target_column].to_numpy(dtype=float)

    results: list[FitResult] = []
    initial_amplitude = max(float(y[0]), 0.02)
    for name, (function, parameter_names, initial, bounds) in MODEL_SPECS.items():
        if name == "double_exponential" and len(y) < 10:
            continue
        model_initial = list(initial)
        model_initial[0] = min(max(initial_amplitude, bounds[0][0] + 1e-6), bounds[1][0])
        try:
            parameters, _ = curve_fit(
                function,
                t,
                y,
                p0=model_initial,
                bounds=bounds,
                maxfev=100_000,
            )
            prediction = function(t, *parameters)
            sse = float(np.sum((y - prediction) ** 2))
            aic, bic = _information_criteria(sse, len(y), len(parameters))
            results.append(
                FitResult(
                    model=name,
                    parameters=dict(zip(parameter_names, map(float, parameters), strict=True)),
                    sse=sse,
                    aic=aic,
                    bic=bic,
                    success=True,
                    fit_start=decision.fit_start,
                    fit_end=decision.fit_end,
                    sample_size=len(y),
                )
            )
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            results.append(
                FitResult(
                    model=name,
                    parameters={},
                    sse=float("nan"),
                    aic=float("nan"),
                    bic=float("nan"),
                    success=False,
                    fit_start=decision.fit_start,
                    fit_end=decision.fit_end,
                    sample_size=len(y),
                    error=str(exc),
                )
            )
    return sorted(results, key=lambda result: result.aic if result.success else float("inf")), decision


def predict_fit(model: str, offsets: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    function, parameter_names, _, _ = MODEL_SPECS[model]
    ordered = [parameters[name] for name in parameter_names]
    return function(np.asarray(offsets, dtype=float), *ordered)


def omori_rate(t: np.ndarray, scale: float, shift: float, exponent: float) -> np.ndarray:
    return scale / np.power(t + shift, exponent)


def fit_omori_rate(rate_frame: pd.DataFrame) -> FitResult | None:
    if rate_frame.empty or len(rate_frame) < 5:
        return None
    work = rate_frame.loc[rate_frame["offset"] >= 1].dropna(subset=["rate"])
    if len(work) < 5 or float(work["rate"].max()) <= 0.0:
        return None
    t = work["offset"].to_numpy(dtype=float)
    y = work["rate"].to_numpy(dtype=float)
    try:
        parameters, _ = curve_fit(
            omori_rate,
            t,
            y,
            p0=[max(float(y[0]), 0.01), 1.0, 1.0],
            bounds=([0.0, 0.05, 0.01], [10.0, 100.0, 10.0]),
            maxfev=50_000,
        )
        prediction = omori_rate(t, *parameters)
        sse = float(np.sum((y - prediction) ** 2))
        aic, bic = _information_criteria(sse, len(y), len(parameters))
        return FitResult(
            model="omori_aftershock_rate",
            parameters=dict(zip(["scale", "shift", "exponent"], map(float, parameters), strict=True)),
            sse=sse,
            aic=aic,
            bic=bic,
            success=True,
            fit_start=int(t.min()),
            fit_end=int(t.max()),
            sample_size=len(y),
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None


def cross_validate_relaxation_models(
    trajectories: pd.DataFrame,
    decision: FitDecision,
    *,
    direction: str = "negative",
    folds: int = 5,
    random_seed: int = 1729,
) -> pd.DataFrame:
    """Event-level cross-validation of decay curves.

    Every fold holds out complete events.  Models are fitted to the training-event
    median response and evaluated against the held-out-event median on the fixed decay
    interval selected from the full response.  This is intentionally reported beside,
    not replaced by, AIC/BIC because response points are serially dependent.
    """
    if (
        trajectories.empty
        or not decision.allowed
        or decision.fit_start is None
        or decision.fit_end is None
        or folds < 2
    ):
        return pd.DataFrame()
    work = trajectories.loc[
        (trajectories["direction"] == direction)
        & (trajectories["offset"] >= decision.fit_start)
        & (trajectories["offset"] <= decision.fit_end)
    ]
    pivot = work.pivot(
        index="event_id",
        columns="offset",
        values="abnormal_excess_variance",
    ).dropna()
    if len(pivot) < max(10, folds * 2) or pivot.shape[1] < 5:
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    event_ids = pivot.index.to_numpy()
    rng.shuffle(event_ids)
    split_ids = [chunk for chunk in np.array_split(event_ids, min(folds, len(event_ids))) if len(chunk)]
    offsets = pivot.columns.to_numpy(dtype=float)
    t = offsets - float(decision.fit_start)

    scores: dict[str, list[float]] = {name: [] for name in MODEL_SPECS}
    for held_ids in split_ids:
        held = pivot.loc[held_ids]
        train = pivot.drop(index=held_ids)
        if len(train) < 5 or len(held) < 2:
            continue
        train_target = np.nanmedian(train.to_numpy(dtype=float), axis=0)
        held_target = np.nanmedian(held.to_numpy(dtype=float), axis=0)
        initial_amplitude = max(float(train_target[0]), 0.02)

        for name, (function, _parameter_names, initial, bounds) in MODEL_SPECS.items():
            if name == "double_exponential" and len(t) < 10:
                continue
            model_initial = list(initial)
            model_initial[0] = min(
                max(initial_amplitude, bounds[0][0] + 1e-6),
                bounds[1][0],
            )
            try:
                parameters, _ = curve_fit(
                    function,
                    t,
                    train_target,
                    p0=model_initial,
                    bounds=bounds,
                    maxfev=100_000,
                )
                prediction = function(t, *parameters)
                scores[name].append(float(np.sqrt(np.mean((held_target - prediction) ** 2))))
            except (RuntimeError, ValueError, FloatingPointError):
                continue

    rows: list[dict[str, float | int | str]] = []
    for name, values in scores.items():
        if not values:
            continue
        rows.append(
            {
                "model": name,
                "successful_folds": len(values),
                "mean_held_out_rmse": float(np.mean(values)),
                "median_held_out_rmse": float(np.median(values)),
                "rmse_std": float(np.std(values, ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_held_out_rmse").reset_index(drop=True)
