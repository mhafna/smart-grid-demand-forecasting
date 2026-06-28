# One-Hour-Ahead Model Training Policy

## Fixed chronological periods

The feature master is split by UTC timestamp without shuffling:

| Split | Inclusive start | Inclusive end | Purpose |
|---|---|---|---|
| Training | `2022-01-01T00` | `2023-12-31T23` | Fit model parameters and the Linear Regression scaler |
| Validation | `2024-01-01T00` | `2024-06-30T23` | Select XGBoost settings and compare model variants |
| Test | `2024-07-01T00` | `2024-12-31T23` | Report final performance once, after selection |

These boundaries are fixed. The test set is not used to choose features,
hyperparameters, or model variants.

## Forecast definition and safe predictors

Each row predicts `target_demand_mwh` at timestamp `t` using only calendar
information known in advance and measurements from timestamps earlier than
`t`. The exact feature lists come from
`results/features/tables/feature_groups.csv`. Current demand, current renewable
generation, current residual demand, current renewable share, and the current
`solar_negative_reported` flag are prohibited predictors.

This is a one-step-ahead experiment: the true demand observed at `t-1` is
available for the next prediction. It is not a recursive 24-hour forecast.

## Eligibility and missing data

A row is eligible for a model only when `target_available=True`, the target is
finite and numeric, and every declared predictor is finite and numeric. No
target or predictor is imputed. The feature master remains unchanged.

Natural-eligibility metrics use all rows available to each model. Fair model
comparison uses a common validation subset and a common test subset on which
persistence and every requested model are eligible. Common-subset validation
MAE is the primary criterion for comparing final variants.

## Algorithms and fitting

Linear Regression uses `StandardScaler` followed by
`sklearn.linear_model.LinearRegression`. Both steps are fitted only on eligible
training rows. Scaling makes coefficient magnitudes comparable within a fitted
model, but correlated lag and rolling features prevent causal interpretation.

XGBoost candidates are fitted only on eligible training rows. A small candidate
grid is declared in `src/train_one_step_models.py` before evaluation. The best
candidate for each feature group is selected using validation MAE. The fixed
winner is then evaluated on the test set; test results never feed back into
selection. A fixed random seed is used.

## Metrics and benchmark

Reported metrics are observation count, MAE, RMSE, MAPE, sMAPE, mean error
(prediction minus actual), and R-squared. Model success is judged primarily by
common-subset MAE against one-hour persistence (`demand_lag_1h`), not by a high
R-squared value alone.

The established full-eligibility persistence results are retained as context:
validation MAE `780.46 MWh`, validation RMSE `973.42 MWh`, test MAE
`976.18 MWh`, test RMSE `1,203.21 MWh`, and test MAPE `3.469%`. Persistence is
recalculated on the new common subsets for exact like-for-like comparisons.
