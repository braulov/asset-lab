import numpy as np
import pandas as pd

from asset_lab.analysis.relaxation import fit_relaxation_models


def test_exponential_data_prefers_exponential_or_stretched() -> None:
    offsets = np.arange(1, 31, dtype=float)
    median = 0.5 * np.exp(-(offsets - 1) / 7.0)
    aggregate = pd.DataFrame(
        {
            "offset": offsets,
            "median": median,
            "bootstrap_ci_low": median * 0.7,
            "bootstrap_ci_high": median * 1.3,
            "count": 40,
        }
    )
    fits, decision = fit_relaxation_models(aggregate)
    assert decision.allowed
    assert fits
    assert fits[0].model in {"exponential", "stretched_exponential", "double_exponential"}
    assert fits[0].success

from asset_lab.analysis.relaxation import cross_validate_relaxation_models


def test_event_level_decay_cross_validation_returns_models() -> None:
    rng = np.random.default_rng(21)
    rows = []
    offsets = np.arange(1, 21)
    for event_id in range(40):
        amplitude = rng.lognormal(mean=np.log(0.5), sigma=0.15)
        values = amplitude * np.exp(-(offsets - 1) / 6.0) + rng.normal(0.0, 0.02, len(offsets))
        for offset, value in zip(offsets, values, strict=True):
            rows.append(
                {
                    "event_id": event_id,
                    "direction": "negative",
                    "offset": offset,
                    "abnormal_excess_variance": value,
                }
            )
    trajectories = pd.DataFrame(rows)
    aggregate = (
        trajectories.groupby("offset", as_index=False)
        .agg(
            median=("abnormal_excess_variance", "median"),
            bootstrap_ci_low=("abnormal_excess_variance", lambda x: x.quantile(0.10)),
            bootstrap_ci_high=("abnormal_excess_variance", lambda x: x.quantile(0.90)),
            count=("abnormal_excess_variance", "size"),
        )
    )
    _, decision = fit_relaxation_models(aggregate)
    table = cross_validate_relaxation_models(trajectories, decision, folds=5)
    assert not table.empty
    assert "exponential" in set(table["model"])
