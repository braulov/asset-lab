from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


MODEL_LABELS: dict[str, str] = {
    "exponential": "Exponential",
    "kurbakovsky": "Kurbakovsky constrained mobility",
    "landau": "Amplitude-aware Landau",
    "double_exponential": "Two-reservoir / double exponential",
    "shifted_power": "Shifted power law",
}
MODEL_ORDER = tuple(MODEL_LABELS)


@dataclass(frozen=True)
class AmplitudeFit:
    model: str
    parameters: dict[str, float]
    rmse: float
    sse: float
    sample_size: int
    events: int

    @property
    def label(self) -> str:
        return MODEL_LABELS[self.model]

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["model"] = self.label
        record["parameters"] = ", ".join(
            f"{key}={value:.5g}" for key, value in self.parameters.items()
        )
        return record


@dataclass(frozen=True)
class _PathSample:
    event_ids: np.ndarray
    x0: np.ndarray
    targets: np.ndarray
    model_times: np.ndarray
    offsets: np.ndarray


def _predict_raw(model: str, raw: np.ndarray, x0: np.ndarray, times: np.ndarray) -> np.ndarray:
    x0 = np.asarray(x0, dtype=float)
    times = np.asarray(times, dtype=float)

    if model == "exponential":
        tau = float(np.exp(raw[0]))
        return x0 * np.exp(-times / tau)

    if model == "kurbakovsky":
        tau = float(np.exp(raw[0]))
        q = np.sqrt(np.maximum(1.0 + x0, 1e-12)) - 1.0
        decay = np.exp(-times / tau)
        return 2.0 * q * decay + q * q * decay * decay

    if model == "landau":
        tau = float(np.exp(raw[0]))
        theta = float(np.exp(raw[1]))
        decay = np.exp(-times / tau)
        denominator = np.sqrt(
            1.0 + theta * x0 * x0 * np.maximum(1.0 - decay * decay, 0.0)
        )
        return x0 * decay / denominator

    if model == "double_exponential":
        tau_fast = float(np.exp(raw[0]))
        tau_slow = tau_fast + float(np.exp(raw[1]))
        fast_weight = float(1.0 / (1.0 + np.exp(-raw[2])))
        return x0 * (
            fast_weight * np.exp(-times / tau_fast)
            + (1.0 - fast_weight) * np.exp(-times / tau_slow)
        )

    if model == "shifted_power":
        shift = float(np.exp(raw[0]))
        exponent = float(np.exp(raw[1]))
        return x0 * np.power(1.0 + times / shift, -exponent)

    raise ValueError(f"Unknown amplitude-aware model: {model}")


_MODEL_SPECS: dict[str, tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]] = {
    "exponential": (
        np.log([8.0]),
        (np.log([0.25]), np.log([256.0])),
    ),
    "kurbakovsky": (
        np.log([8.0]),
        (np.log([0.25]), np.log([256.0])),
    ),
    "landau": (
        np.log([8.0, 0.05]),
        (np.log([0.25, 1e-7]), np.log([256.0, 1000.0])),
    ),
    "double_exponential": (
        np.array([np.log(2.0), np.log(18.0), 0.0]),
        (
            np.array([np.log(0.25), np.log(0.1), -8.0]),
            np.array([np.log(128.0), np.log(256.0), 8.0]),
        ),
    ),
    "shifted_power": (
        np.log([4.0, 1.0]),
        (np.log([0.1, 0.03]), np.log([256.0, 12.0])),
    ),
}


def _decode_parameters(model: str, raw: np.ndarray) -> dict[str, float]:
    if model in {"exponential", "kurbakovsky"}:
        return {"tau": float(np.exp(raw[0]))}
    if model == "landau":
        return {
            "tau": float(np.exp(raw[0])),
            "theta": float(np.exp(raw[1])),
        }
    if model == "double_exponential":
        tau_fast = float(np.exp(raw[0]))
        return {
            "tau_fast": tau_fast,
            "tau_slow": tau_fast + float(np.exp(raw[1])),
            "fast_weight": float(1.0 / (1.0 + np.exp(-raw[2]))),
        }
    if model == "shifted_power":
        return {
            "shift": float(np.exp(raw[0])),
            "exponent": float(np.exp(raw[1])),
        }
    raise ValueError(model)


def predict_amplitude_model(
    model: str,
    x0: np.ndarray | float,
    times: np.ndarray | float,
    parameters: dict[str, float],
) -> np.ndarray:
    x0_array = np.asarray(x0, dtype=float)
    times_array = np.asarray(times, dtype=float)

    if model == "exponential":
        return x0_array * np.exp(-times_array / parameters["tau"])
    if model == "kurbakovsky":
        q = np.sqrt(np.maximum(1.0 + x0_array, 1e-12)) - 1.0
        decay = np.exp(-times_array / parameters["tau"])
        return 2.0 * q * decay + q * q * decay * decay
    if model == "landau":
        decay = np.exp(-times_array / parameters["tau"])
        denominator = np.sqrt(
            1.0
            + parameters["theta"]
            * x0_array
            * x0_array
            * np.maximum(1.0 - decay * decay, 0.0)
        )
        return x0_array * decay / denominator
    if model == "double_exponential":
        weight = parameters["fast_weight"]
        return x0_array * (
            weight * np.exp(-times_array / parameters["tau_fast"])
            + (1.0 - weight) * np.exp(-times_array / parameters["tau_slow"])
        )
    if model == "shifted_power":
        return x0_array * np.power(
            1.0 + times_array / parameters["shift"],
            -parameters["exponent"],
        )
    raise ValueError(model)


def _prepare_paths(
    trajectories: pd.DataFrame,
    *,
    process_class: str,
    direction: str,
    initial_hours: int,
    fit_end: int,
) -> _PathSample | None:
    if trajectories.empty or initial_hours < 1 or fit_end <= initial_hours + 2:
        return None

    selected = trajectories.loc[
        (trajectories["process_class"] == process_class)
        & (trajectories["direction"] == direction)
        & trajectories["offset"].between(1, fit_end)
    ].copy()
    if selected.empty:
        return None

    pivot = selected.pivot_table(
        index="event_id",
        columns="offset",
        values="abnormal_m2",
        aggfunc="first",
    )
    initial_offsets = list(range(1, initial_hours + 1))
    target_offsets = list(range(initial_hours + 1, fit_end + 1))
    required = initial_offsets + target_offsets
    if not set(required).issubset(pivot.columns):
        return None

    complete = pivot[required].replace([np.inf, -np.inf], np.nan).dropna()
    if complete.empty:
        return None

    x0 = complete[initial_offsets].median(axis=1).clip(lower=0.0).to_numpy(float)
    targets = complete[target_offsets].to_numpy(float)
    keep = np.isfinite(x0) & (x0 > 0.02) & np.isfinite(targets).all(axis=1)
    if not np.any(keep):
        return None

    return _PathSample(
        event_ids=complete.index.to_numpy()[keep],
        x0=x0[keep, None],
        targets=targets[keep],
        model_times=np.arange(1, len(target_offsets) + 1, dtype=float)[None, :],
        offsets=np.asarray(target_offsets, dtype=int),
    )


def _fit_raw(sample: _PathSample, model: str) -> np.ndarray:
    initial, bounds = _MODEL_SPECS[model]
    cap = max(float(np.nanquantile(np.abs(sample.targets), 0.99)), 0.5)
    robust_target = np.clip(sample.targets, -cap, cap)

    def residual(raw: np.ndarray) -> np.ndarray:
        prediction = _predict_raw(model, raw, sample.x0, sample.model_times)
        return ((prediction - robust_target) / math.sqrt(robust_target.shape[1])).ravel()

    result = least_squares(
        residual,
        initial,
        bounds=bounds,
        loss="soft_l1",
        f_scale=0.5,
        max_nfev=20_000,
    )
    return result.x


def fit_amplitude_models(
    trajectories: pd.DataFrame,
    *,
    process_class: str,
    direction: str = "negative",
    initial_hours: int = 3,
    fit_end: int = 24,
) -> list[AmplitudeFit]:
    sample = _prepare_paths(
        trajectories,
        process_class=process_class,
        direction=direction,
        initial_hours=initial_hours,
        fit_end=fit_end,
    )
    if sample is None or len(sample.event_ids) < 6:
        return []

    fits: list[AmplitudeFit] = []
    for model in MODEL_ORDER:
        raw = _fit_raw(sample, model)
        prediction = _predict_raw(model, raw, sample.x0, sample.model_times)
        errors = sample.targets - prediction
        sse = float(np.sum(errors**2))
        fits.append(
            AmplitudeFit(
                model=model,
                parameters=_decode_parameters(model, raw),
                rmse=float(math.sqrt(sse / errors.size)),
                sse=sse,
                sample_size=int(errors.size),
                events=int(len(sample.event_ids)),
            )
        )
    return sorted(fits, key=lambda fit: fit.rmse)


def cross_validate_amplitude_models(
    trajectories: pd.DataFrame,
    *,
    process_class: str,
    direction: str = "negative",
    initial_hours: int = 3,
    fit_end: int = 24,
    folds: int = 5,
    seed: int = 2026,
) -> pd.DataFrame:
    sample = _prepare_paths(
        trajectories,
        process_class=process_class,
        direction=direction,
        initial_hours=initial_hours,
        fit_end=fit_end,
    )
    if sample is None or len(sample.event_ids) < max(10, folds * 2):
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    order = np.arange(len(sample.event_ids))
    rng.shuffle(order)
    fold_indices = np.array_split(order, folds)
    records: list[dict[str, float | int | str]] = []

    for fold_number, held_out in enumerate(fold_indices, start=1):
        training_mask = np.ones(len(sample.event_ids), dtype=bool)
        training_mask[held_out] = False
        train = _PathSample(
            event_ids=sample.event_ids[training_mask],
            x0=sample.x0[training_mask],
            targets=sample.targets[training_mask],
            model_times=sample.model_times,
            offsets=sample.offsets,
        )
        test_x0 = sample.x0[held_out]
        test_targets = sample.targets[held_out]

        for model in MODEL_ORDER:
            raw = _fit_raw(train, model)
            prediction = _predict_raw(model, raw, test_x0, sample.model_times)
            event_rmse = np.sqrt(np.mean((prediction - test_targets) ** 2, axis=1))
            burden_width = min(12, test_targets.shape[1])
            actual_burden = np.maximum(test_targets[:, :burden_width], 0.0).sum(axis=1)
            predicted_burden = np.maximum(prediction[:, :burden_width], 0.0).sum(axis=1)
            burden_error = np.abs(predicted_burden - actual_burden)
            for local_index, sample_index in enumerate(held_out):
                records.append(
                    {
                        "fold": fold_number,
                        "event_id": str(sample.event_ids[sample_index]),
                        "model": model,
                        "event_rmse": float(event_rmse[local_index]),
                        "burden_mae_12": float(burden_error[local_index]),
                    }
                )

    event_scores = pd.DataFrame(records)
    summary = (
        event_scores.groupby("model")
        .agg(
            mean=("event_rmse", "mean"),
            median=("event_rmse", "median"),
            std=("event_rmse", "std"),
            burden_mae_12=("burden_mae_12", "mean"),
            events=("event_id", "size"),
        )
        .reset_index()
        .sort_values("mean")
        .reset_index(drop=True)
    )
    summary.insert(1, "label", summary["model"].map(MODEL_LABELS))
    summary["winner"] = summary["mean"] == summary["mean"].min()
    return summary
