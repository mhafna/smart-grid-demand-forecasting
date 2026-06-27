# Chronological Modelling Split Policy

## Fixed UTC periods

Every row in the feature master is assigned once, before any target or feature
filter is applied:

| Split | First timestamp (inclusive) | Last timestamp (inclusive) |
|---|---:|---:|
| Training | `2022-01-01T00` UTC | `2023-12-31T23` UTC |
| Validation | `2024-01-01T00` UTC | `2024-06-30T23` UTC |
| Test | `2024-07-01T00` UTC | `2024-12-31T23` UTC |

Rows are never shuffled and random splitting is prohibited. Training must end
before validation begins, and validation must end before test begins. Missing
targets remain in the feature master and in the split timeline; they are
excluded only when a metric or fitted statistic requires an observed target.

## Baseline eligibility

Eligibility is calculated inside each already-defined split. A row contributes
to a baseline metric only when `target_available=True`, the target is numeric,
and that baseline's prediction is numeric. Missing targets and predictions are
not filled. Because each lag has a different history requirement, eligibility
is reported separately for persistence (1 hour), daily seasonal naive (24
hours), weekly seasonal naive (168 hours), and the training-fitted hour-of-week
mean.

## Training-only fitted statistic

UTC hour of week is `UTC day of week * 24 + UTC hour`, where Monday is day 0.
The 168 category means and the global fallback mean use only available targets
inside the training period. They are fitted once and held fixed for validation
and test. An unseen category uses the training global mean, and every fallback
is counted. Validation and test targets may not update either lookup.

## One-step-ahead interpretation

At timestamp `t`, persistence predicts `demand_lag_1h`, daily seasonal naive
predicts `demand_lag_24h`, and weekly seasonal naive predicts
`demand_lag_168h`. The true earlier demand is assumed to have arrived before
the prediction is issued. No same-hour solar, wind, combined renewable,
renewable share, or residual-demand measurement is used.

This is a one-hour-ahead evaluation with a newly observed history at every
step. It is not a recursive 24-hour forecast. In the later recursive setting,
some demand lags at later horizons will be unavailable and must be replaced by
earlier model predictions.

## Metrics

For actual value `y` and prediction `p`, error and bias use `p - y`:

- MAE is the mean of `|p - y|`.
- RMSE is the square root of the mean squared error.
- MAPE is `100 * mean(|p - y| / |y|)` on nonzero actuals.
- sMAPE is `100 * mean(2|p - y| / (|y| + |p|))` where the denominator is nonzero.
- Mean error (bias) is the mean of `p - y`.
- R² is `1 - sum((p-y)^2) / sum((y-mean(y))^2)`.

Models are ranked primarily by MAE. RMSE, MAPE, bias, R², and demand-tail
performance remain part of the interpretation.
