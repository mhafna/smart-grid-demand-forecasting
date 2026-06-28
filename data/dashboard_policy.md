# Dashboard policy

## Purpose and scope

The Streamlit dashboard is a retrospective planning-support demonstration. It
reads saved historical, forecasting, renewable-planning, and evaluation tables.
It does not retrain a model, run recursive inference, download data, call an
external API, or connect to a live electricity grid.

The dates offered in the interface are evaluation dates already present in the
saved validation and test outputs. They are not future live forecasts. Every
timestamp, date boundary, peak time, and ramp time is labelled and interpreted
as Coordinated Universal Time (UTC).

## Demand, renewables, and residual demand

- **Forecast demand** estimates total electricity demand for an evaluation hour.
- **Forecast renewables** estimate solar and wind generation using the saved
  daily seasonal-naive method selected during the analytical pipeline.
- **Forecast residual demand** subtracts forecast solar and wind from forecast
  demand. It is useful for comparison and planning, but it is not a complete
  physical grid balance because it omits other generation, imports, exports,
  storage, losses, and operational constraints.

## Planning indicators

The high-demand, high-residual-demand, high-upward-residual-ramp, and
low-renewable-share indicators reuse thresholds fitted only on the training
period. They mark hours for review. They are descriptive signals, not emergency
alerts, safety guarantees, dispatch commands, or automated operating advice.

## Missing values and source limitations

Missing observations remain missing. The dashboard never converts them to zero,
interpolates them, or silently drops an affected forecast date. It warns when a
selected day lacks complete actual comparison measurements.

The historical source includes five timestamps with null demand, solar, and wind
measurements; a separate 24-hour block with absent SUN and WND rows; and 9,074
negative solar measurements preserved from the EIA source. The cause of the
negative values is unresolved. Percentage errors can also be unstable when solar
generation is close to zero.

Recursive uncertainty generally increases at later horizons, and saved peak
results show systematic underprediction. These limitations are shown rather than
hidden. The original historical and analytical output files remain read-only.
