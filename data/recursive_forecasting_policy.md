# Leakage-Safe Recursive 24-Hour Forecasting Policy

## Operational forecast definition

One forecast is issued at 23:00 UTC each day. The forecast origin is that
23:00 timestamp, and the 24 targets are the following UTC calendar day from
00:00 through 23:00. Validation forecast days are 2024-01-01 through
2024-06-30; test forecast days are 2024-07-01 through 2024-12-31. These fixed
periods are not changed after missing-data checks.

This is daily rolling-origin evaluation. At every new origin, reported demand
through the origin is available again. It is not a single six-month recursive
simulation.

## Information boundary

Calendar fields for a target timestamp are known in advance. Measured demand
after the forecast origin is forbidden. For each saved autoregressive model,
the first horizon uses only observed demand at or before the origin. After a
prediction is made, that prediction is added to a temporary, origin-specific
demand buffer. Later horizons obtain any post-origin lag or rolling-window
input from that buffer, never from the actual target column or precomputed
future features.

The saved feature order is authoritative. Demand lags are 1, 2, 3, 6, 12, 24,
48, and 168 hours. Rolling statistics use the 24 or 168 values immediately
before the target timestamp and calculate mean, sample standard deviation
(`ddof=1`), minimum, and maximum. This exactly preserves the one-step training
definition.

## Origin eligibility and fair comparison

An origin is eligible only when all 24 actual targets, the origin demand, all
required pre-origin demand history, and every daily/weekly seasonal-naive
source value are reported and finite. Missing history is never imputed. The
same eligible origins and target timestamps are used for both recursive models
and all three baselines. Ineligible origins are retained in an audit table with
their exact reason.

## Baselines

- Flat persistence repeats the observed demand at the origin for all horizons.
- Daily seasonal naive uses the observed demand at target minus 24 hours.
- Weekly seasonal naive uses the observed demand at target minus 168 hours.

All baseline source timestamps are at or before the origin.

## Selection and reporting

Recursive models are ranked by validation MAE over all 24 horizons. Test data
is used only for final reporting. No model is fitted, refitted, tuned, or
searched in this phase. Saved one-step artifacts and metadata are read-only.
Horizon-1 recursive predictions must match the saved legitimate one-step
predictions within a small numerical tolerance; horizons 2-24 are not expected
to match teacher-forced one-step predictions.
