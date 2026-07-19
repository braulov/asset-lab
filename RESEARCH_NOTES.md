# Research choices encoded in v6

The application architecture follows the empirical tests performed before implementation.

1. A unique post-shock relaxation law is not identified reliably. The service reports event-level held-out errors for four candidate laws instead of declaring an in-sample winner.
2. Proportional price mobility `M(F)=bF` won leave-one-year-out comparison for most assets in the 25-stock hourly panel. The affine law is retained because its slope contains information about finite-horizon M3.
3. Affine diffusion predicts nonnegative skewness but materially underestimates observed M3. The residual is therefore shown explicitly rather than hidden inside a variance fit.
4. Strong negative M3 is concentrated in overnight and systemic external-like episodes. Hour-of-day and overnight panels are first-class objects.
5. Preheated/internal-like events are identified primarily by their prehistory. The service does not infer causal news origin from prices alone.
6. Market-level M3 was more useful for forecasting future market M2 than individual-asset M3. A multi-asset ZIP activates the market-moment panel.
7. Broad spectral graph tests did not improve robust QLIKE and did not show a reliable pre-shock precursor. Graph metrics remain outside the main v6 workflow.
