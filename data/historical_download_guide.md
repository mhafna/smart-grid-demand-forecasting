# Historical EIA CISO Download Guide

## Chosen Period

Use the complete three-year period from 2022-01-01 through 2024-12-31.

The download script treats the command-line end date as inclusive. That means
`2024-12-31` is converted to `2024-12-31T23`, so the last day is included all
the way through its final hourly record.

## Why Three Complete Years

Three complete calendar years give enough history to inspect weekday patterns,
seasonal patterns, and a leap year without mixing partial-year edges into the
first modeling dataset. The range includes 2024, so validation must see February
29, 2024.

This workflow does not train models or report findings. It only prepares the
raw download, validation, and processed CSV build steps.

## Expected Files

After a successful download:

- `data/raw/eia_ciso_hourly_demand_2022_2024.json`
- `data/raw/eia_ciso_hourly_renewable_generation_2022_2024.json`

After successful validation and build:

- `data/processed/eia_ciso_hourly_2022_2024.csv`

The processed CSV preserves the same column names as the sample CSV:

- `period`
- `demand_mwh`
- `solar_generation_mwh`
- `wind_generation_mwh`
- `solar_wind_generation_mwh`
- `residual_demand_after_solar_wind_mwh`
- `solar_wind_share_pct`

## Safe API-Key Handling

Set the EIA key only in your shell environment. The historical script reads
`EIA_API_KEY` from the environment and does not read it from a file.

PowerShell example:

```powershell
$env:EIA_API_KEY = "your-real-key-here"
```

Do not commit the key, paste it into documentation, or print it in terminal
logs. The script prints progress messages but never prints the key. If EIA
echoes the key in returned request metadata, the historical downloader redacts
that single metadata field before saving the raw page wrapper.

## Commands To Run Later

Download the raw historical files:

```powershell
python src\fetch_eia_history.py 2022-01-01 2024-12-31
```

Validate the raw historical files:

```powershell
python src\validate_eia_history.py 2022-01-01 2024-12-31
```

Build the processed historical CSV:

```powershell
python src\build_historical_dataset.py
```

## How Pagination Works

EIA JSON API responses are limited to 5,000 rows per request. A three-year
hourly file has more rows than that, so the script asks for the data in pages.

The first request starts at `offset=0` and asks for up to `length=5000` rows.
EIA also returns `response.total`, which says how many matching rows exist for
the full request. If `response.total` is larger than the rows already received,
the script asks for the next page using a larger offset. It keeps going until
the number of downloaded rows matches the expected total for the requested
complete hourly range.

Demand has one row per hour. Solar and wind generation have two rows per hour:
one for `SUN` and one for `WND`.

## Runtime And File-Size Considerations

The download should take longer than the seven-day sample because it makes
multiple API requests for demand and multiple API requests for renewable
generation. The raw files will also be larger than the sample files because
they preserve every EIA page response and its metadata.

Exact runtime and file size depend on EIA API response speed, local network
conditions, and the formatting of the returned JSON. Do not treat any rough
estimate as a result.

## Required Checks Before Exploratory Analysis

Before starting exploratory analysis:

- The download command must finish without errors.
- `src\validate_eia_history.py` must report `Overall result: PASS`.
- `src\build_historical_dataset.py` must write `data/processed/eia_ciso_hourly_2022_2024.csv`.
- The processed CSV should then be inspected with a focused validation step
  before modeling or feature engineering begins.

If any validation check fails, stop and inspect the raw files and error output.
Do not fill missing hours, remove duplicate rows, or change units without a
clear documented reason.
