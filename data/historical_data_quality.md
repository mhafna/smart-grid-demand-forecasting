# Historical EIA CISO Data Quality

## Policy

The master historical dataset preserves the official EIA source exactly where
measurements are unavailable. It keeps every demand timestamp, does not replace
missing values with zero, and does not interpolate or fabricate observations.

Two source-data exceptions are documented and allowed. Any deviation from
these exact exceptions fails validation and requires a new investigation.

## Present Rows With Null Measurements

At each timestamp below, the EIA response contains a demand row, a `SUN` row,
and a `WND` row, but each row's `value` field is JSON `null`:

- `2022-01-05T10`
- `2022-05-17T18`
- `2022-06-13T18`
- `2023-10-31T21`
- `2023-11-14T20`

These are five null demand measurements and ten null renewable measurements.
Their row schemas, respondent, categories, units, timestamps, and pagination
are otherwise valid.

## Absent Renewable Rows

Both `SUN` and `WND` rows are absent from `2024-11-02T08` through
`2024-11-03T07`, inclusive. This is one block of 24 timestamps and 48 absent
renewable rows. Demand rows remain present during this block.

## Why The Distinction Matters

A present row with a null value says that EIA returned the expected record but
did not provide its measurement. An absent row says that no matching source
record was returned for that timestamp and fuel. Both produce null renewable
columns in the processed CSV, but validation tracks them separately so a new
gap cannot be mistaken for a known null measurement.

Zero-filling would claim that measured demand or generation was zero.
Interpolation would create an estimate that was not reported by EIA. Either
choice would misrepresent the official source, so the master dataset does
neither.

## Processed Quality Flags

The processed CSV retains all 26,304 hourly timestamps and adds:

- `demand_data_complete`: true only when demand is present and numeric
- `renewable_data_complete`: true only when both solar and wind are present
  and numeric

Combined solar/wind generation is null unless both renewable measurements are
complete. Residual demand and solar/wind share are null unless demand, solar,
and wind are all complete.

## Later Analysis And Modelling

- Never train or evaluate a forecasting target where demand is null.
- Renewable-aware analysis requires `renewable_data_complete=True`.
- The master processed CSV must retain every timestamp.
- Filtering belongs only in downstream analysis or modelling datasets and must
  be reported with the affected row counts and reason.
- Chronological train, validation, and test splits remain required after any
  documented quality filtering.
