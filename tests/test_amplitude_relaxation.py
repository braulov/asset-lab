from __future__ import annotations

import numpy as np
import pandas as pd

from asset_lab.analysis.amplitude_relaxation import (
    fit_amplitude_models,
    predict_amplitude_model,
)


def test_landau_starts_at_event_amplitude() -> None:
    amplitudes = np.array([0.1, 0.5, 1.5, 3.0])
    prediction = predict_amplitude_model(
        "landau",
        amplitudes,
        np.zeros_like(amplitudes),
        {"tau": 7.0, "theta": 0.4},
    )
    np.testing.assert_allclose(prediction, amplitudes)


def test_landau_tends_to_exponential_when_theta_is_tiny() -> None:
    times = np.arange(0.0, 24.0)
    landau = predict_amplitude_model(
        "landau",
        1.25,
        times,
        {"tau": 9.0, "theta": 1e-12},
    )
    exponential = predict_amplitude_model(
        "exponential",
        1.25,
        times,
        {"tau": 9.0},
    )
    np.testing.assert_allclose(landau, exponential, rtol=1e-10, atol=1e-12)


def synthetic_landau_trajectories(events: int = 30, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    offsets = np.arange(1, 25)
    for event_id in range(events):
        x0 = float(rng.uniform(0.25, 2.5))
        target = predict_amplitude_model(
            "landau",
            x0,
            np.maximum(offsets - 3, 0),
            {"tau": 7.5, "theta": 0.8},
        )
        target[:3] = x0
        target += rng.normal(0.0, 0.008, len(target))
        for offset, value in zip(offsets, target, strict=True):
            rows.append(
                {
                    "event_id": event_id,
                    "process_class": "preheated-like",
                    "direction": "negative",
                    "offset": int(offset),
                    "abnormal_m2": float(value),
                }
            )
    return pd.DataFrame(rows)


def test_landau_wins_on_landau_generated_event_paths() -> None:
    fits = fit_amplitude_models(
        synthetic_landau_trajectories(),
        process_class="preheated-like",
        direction="negative",
        initial_hours=3,
        fit_end=24,
    )
    assert fits
    assert fits[0].model == "landau"
    assert fits[0].rmse < 0.02
