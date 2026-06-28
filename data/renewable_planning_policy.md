# Renewable-aware planning policy

## Purpose

This layer supports human review of 24-hour forecasts. It is not an automated grid controller and does not provide dispatch, safety, price, cost, or savings claims.

## Fixed inputs and splits

The layer reuses saved Recursive XGBoost demand predictions. Training is 2022-01-01 through 2023-12-31 UTC, validation is 2024-01-01 through 2024-06-30 UTC, and test is 2024-07-01 through 2024-12-31 UTC. Renewable method selection uses validation combined-renewable MAE only; test results cannot change it.

## Past-only renewable methods

Daily and weekly seasonal-naive forecasts read the reported value 24 or 168 hours before the target. The rolling method uses complete, same-UTC-hour observations in the interval `(origin - 28 days, origin]`. At least seven observations per resource are required for a rolling estimate or scenario. Missing values are retained, never filled. Negative reported solar values are preserved and are not clipped.

## Scenarios

Conservative, typical, and favourable renewable availability are the 25th, 50th, and 75th percentiles of the past-only same-hour sample, calculated separately for solar and wind and then combined. They are empirical planning scenarios, not statistically calibrated prediction intervals. A missing scenario retains nulls and records its history counts and reason.

## Residual demand and indicators

Residual demand equals forecast demand minus forecast solar-plus-wind. It is not floored at zero. It is not complete physical grid balance because it excludes other generation, imports, exports, storage, losses, and network constraints. Thresholds are fitted from training observations only. Alerts are descriptive planning indicators, not safety guarantees or operating instructions.
