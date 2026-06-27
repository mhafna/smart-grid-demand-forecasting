# Leakage-Safe Feature Engineering Policy

## Forecast-Time Rule

For a demand prediction at timestamp `t`, every measured input must come from a
timestamp before `t`. The current demand value is the target, not a predictor.
Calendar values for `t` are allowed because the calendar is known in advance.

The first application design will make a one-hour-ahead prediction and repeat
that process recursively to reach 24 hours. In recursive forecasting, a later
horizon may need to use an earlier model prediction in place of an unavailable
actual demand lag. Training and evaluation must reproduce that information
boundary; they must not substitute actual future demand values that would not
have been known at the forecast origin.

## Allowed Predictor Groups

The groups are cumulative:

1. **Calendar-only** uses UTC calendar fields and deterministic cyclical
   transformations. No encoder is fitted to the dataset.
2. **Autoregressive demand** adds strictly past demand lags and rolling
   statistics. Every rolling source is shifted one hour before the window is
   calculated. A 24-hour statistic at `t`, for example, uses `t-24` through
   `t-1`, never `t`.
3. **Renewable-history enhanced** adds lagged solar, wind, combined solar-plus-
   wind generation, and lagged solar-quality flags. These are observations from
   before the prediction time.

Actual future renewable generation is not assumed to be known. Same-hour actual
solar, wind, combined generation, renewable share, residual demand, and the
current negative-solar flag are prohibited as demand predictors. Renewable-
history features may use only observations available before the relevant
forecast origin. A future recursive implementation must explicitly carry that
origin-time availability rule across all 24 horizons.

## Missing And Reported Source Values

The canonical feature master retains every one of the 26,304 timestamps. It does
not interpolate, forward-fill, backward-fill, zero-fill, clip, or overwrite any
measurement. Natural nulls in early lags, full rolling windows, and periods
affected by documented source gaps remain null.

Negative solar values are official reported values. They are preserved exactly
in the source and lagged columns and accompanied by nullable reported-negative
flags. No zero-clipped solar feature belongs in the canonical feature master.

`target_available` is false when reported demand is unavailable. Models must
exclude those rows from fitting and evaluation, but the rows remain in the
master timeline. A model may also require complete values for its chosen
feature group.

## Splitting And Filtering Order

Validation, test, and eventual training periods must be chronological. Define
those time boundaries first. Only then, within each already-defined period,
exclude unavailable targets and rows missing features required by the selected
model. This order prevents data availability from silently changing the time
boundaries and keeps all eligibility counts auditable.

No performance conclusion follows from these features. Baselines must be built
and evaluated before more advanced models, in a later modelling task.
