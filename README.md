# Asset Lab v6

Interactive Streamlit laboratory for MOEX daily and hourly OHLC data.

Version 6 keeps the validated daily v5 workflow and adds a separate hourly research page for the mechanisms identified in the multi-asset study: M2/M3, price mobility, overnight asymmetry, preheated versus abrupt shocks and mechanistic relaxation models.

## Pages

### Daily Lab

The daily page preserves the v5 methodology:

- multiple OHLC variance proxies;
- exclusion of the still-forming daily candle;
- past-only expanding HAR forecasts;
- robust volatility-innovation scores;
- negative and positive trend hinges;
- HAC coefficient intervals and walk-forward forecasting;
- HAR conditional-median counterfactuals;
- matched no-shock controls;
- local projections;
- guarded relaxation fitting;
- event-level cross-validation;
- flat-rate versus Omori aftershock tests.

### Hourly Moments

The new hourly page supports either a live MOEX SECID or a multi-asset ZIP produced by the supplied hourly exporter.

It provides:

- past-only robust scaling separately by clock hour;
- regular-session M2, M3, skewness and downside M2 share;
- skewness by time of day;
- overnight M3 with robustness filters;
- three conditional-mobility models:
  - constant `M(F)=a`;
  - proportional `M(F)=bF`;
  - affine `M(F)=a+bF`;
- leave-one-year-out mobility comparison;
- the finite-horizon affine-mobility skewness prediction

```text
gamma_K(h) = (exp(b²h)+2) sqrt(exp(b²h)-1);
```

- observed-minus-affine M3 as a diagnostic for jumps, leverage, news and order flow;
- abrupt-like versus preheated-like shock classification;
- with a multi-asset ZIP:
  - market-wide versus isolated classification;
  - external-like = abrupt + market-wide;
  - internal-like = preheated + isolated;
- matched-control M2 and M3 trajectories;
- four relaxation laws:
  - exponential;
  - constrained mobility;
  - stress-fed / two-reservoir;
  - shifted power law;
- event-level held-out relaxation comparison;
- market M2, market M3 and extreme-return synchrony.

## Interpretation rules built into v6

- M2 and M3 are displayed separately.
- Overnight, regular-session and evening effects are not pooled without warning.
- `M(F)=bF` is treated as the main geometric historical null; affine mobility is a mechanistic competitor.
- A fitted law is not called uniquely identified when held-out RMSE differences are small.
- The process labels are explicitly proxies, not causal news classifications.
- Spectral graph quantities are not promoted to production predictors because broad daily and hourly tests did not show robust incremental forecasting value.

## Installation on Ubuntu

Python 3.10 or newer is supported.

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv unzip

unzip asset-lab-mvp-v6.zip
cd asset-lab-mvp-v6
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

The page selector appears in the Streamlit sidebar.

## Multi-asset hourly ZIP

The Hourly Moments page accepts ZIP files containing hourly candle CSV or CSV.GZ files. The expected columns are:

```text
begin, end, open, close, high, low, value, volume
```

Files under a `candles/` folder are detected automatically. The focused 25-stock exporter creates a compatible archive.

## Tests

```bash
source .venv/bin/activate
pytest
```

Version 6 has 33 tests. The added hourly tests cover:

- absence of future lookahead in robust scaling;
- monotonic nonnegative affine-mobility skewness;
- recovery of proportional mobility on synthetic data;
- cold versus preheated shock classification;
- construction of mobility price bins.

## Methodological limits

Hourly OHLC candles are not transaction-level data. They cannot identify participant IDs, order-flow imbalance, cancellations, queue position, spread or depth. The next mechanism-focused upgrade is five-minute TradeStats, OrderStats and OBStats or an anonymous order log for a small set of selected events.
