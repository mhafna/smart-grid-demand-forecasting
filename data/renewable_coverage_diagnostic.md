# Renewable Coverage Diagnostic

## Confirmed Result

The focused diagnostic examined CISO `SUN` and `WND` coverage for the inclusive
period 2022-01-01 through 2024-12-31.

- Demand: 26,304 observed of 26,304 theoretical hourly rows
- Solar (`SUN`): 26,280 observed of 26,304 theoretical hourly rows
- Wind (`WND`): 26,280 observed of 26,304 theoretical hourly rows
- Combined renewables: 52,560 observed of 52,608 theoretical rows
- Renewable shortfall: 48 rows, representing 24 hours for both fuels

## Exact Affected Timestamps

Both `SUN` and `WND` are missing at every timestamp in this inclusive block:

```text
2024-11-02T08
2024-11-02T09
2024-11-02T10
2024-11-02T11
2024-11-02T12
2024-11-02T13
2024-11-02T14
2024-11-02T15
2024-11-02T16
2024-11-02T17
2024-11-02T18
2024-11-02T19
2024-11-02T20
2024-11-02T21
2024-11-02T22
2024-11-02T23
2024-11-03T00
2024-11-03T01
2024-11-03T02
2024-11-03T03
2024-11-03T04
2024-11-03T05
2024-11-03T06
2024-11-03T07
```

No other missing timestamp is documented or allowed by the historical
validator.

## Cause

This is confirmed EIA source-data missingness, not a downloader defect:

- Focused pagination downloaded exactly the rows reported by EIA
  `response.total`.
- No duplicate timestamp/fuel combinations were found.
- The first and last requested days were complete.
- The missing rows form one internal block, so inclusive date-boundary handling
  did not cause the shortfall.

## Project Decision

The workflow preserves these 24 unavailable renewable hours as missing values.
It does not interpolate them, replace them with zeros, delete their demand
timestamps, or fabricate replacement observations. The processed dataset marks
the affected rows with `renewable_data_complete = False`.

Zero would claim that CISO reported no solar or wind generation during these
hours. The source instead supplied no observations, which is a different fact.

## Safe Use

The validator accepts only this exact block for both fuels. Any additional,
shorter, longer, shifted, or fuel-specific gap fails validation and must be
diagnosed before analysis continues.
