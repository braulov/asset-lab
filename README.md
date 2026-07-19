# Asset Lab v6

Interactive Streamlit laboratory for MOEX daily and hourly OHLC data.

Version 6 keeps the validated daily v5 workflow and adds a separate hourly research page for the mechanisms identified in the multi-asset study: M2/M3, price mobility, overnight asymmetry, preheated versus abrupt shocks and amplitude-aware relaxation models.

The application opens directly on **Daily Lab**. The navigation contains only two working pages:

- **Daily Lab**;
- **Hourly Moments**.

There is no separate landing page duplicating this README.

## Daily Lab

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

## Hourly Moments

The hourly page supports either a live MOEX SECID or a multi-asset ZIP produced by the supplied hourly exporter.

It provides:

- past-only robust scaling separately by clock hour;
- regular-session M2, M3, skewness and downside M2 share;
- skewness by time of day;
- overnight M3 with robustness filters;
- three conditional price-mobility models:
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
- five amplitude-aware relaxation candidates:
  - exponential;
  - Kurbakovsky constrained mobility;
  - Landau;
  - two-reservoir / double exponential;
  - shifted power law;
- event-level held-out RMSE and 12-hour positive-excess burden error;
- market M2, market M3 and extreme-return synchrony.

### Landau versus constrained mobility

These are separate models.

Kurbakovsky constrained mobility assumes that an underlying mobility relaxes exponentially and variance is its square:

```text
x(t) = 2q exp(-t/tau) + q² exp(-2t/tau),
q = sqrt(1+x0) - 1.
```

It is a tightly linked double exponential: its two time scales and weights are constrained.

The amplitude-aware Landau law is

```text
x(t) = x0 exp(-t/tau)
       / sqrt(1 + theta x0² (1-exp(-2t/tau))).
```

Its relaxation rate changes nonlinearly with the initial event amplitude. The UI therefore compares Landau explicitly instead of hiding it under the constrained-mobility label.

## Interface

Exact parameter values are entered through numeric fields. A thin strip below each field shows where the value lies inside its admissible range. The shock-analysis controls are collapsed below the process description, leaving the top of the page for the dataset summary and results.

## Interpretation rules built into v6

- M2 and M3 are displayed separately.
- Overnight, regular-session and evening effects are not pooled without warning.
- `M(F)=bF` is treated as the main geometric historical null; affine mobility is a mechanistic competitor.
- A fitted law is not called uniquely identified when held-out errors are close.
- Landau is not presented as a universal winner: its main empirical motivation is the preheated-negative branch and integrated short-horizon excess burden.
- The process labels are explicitly proxies, not causal news classifications.
- Spectral graph quantities are not promoted to production predictors because broad daily and hourly tests did not show robust incremental forecasting value.

## Installation on Ubuntu

Python 3.10 or newer is supported.

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv unzip

cd asset-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

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

The hourly tests cover past-only scaling, mobility recovery, shock classification, price bins and the amplitude-aware Landau implementation.

## Methodological limits

Hourly OHLC candles are not transaction-level data. They cannot identify participant IDs, order-flow imbalance, cancellations, queue position, spread or depth. The next mechanism-focused upgrade is five-minute TradeStats, OrderStats and OBStats or an anonymous order log for a small set of selected events.
