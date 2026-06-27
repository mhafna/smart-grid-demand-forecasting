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

The processed CSV preserves the sample CSV's analytical column names and adds
one historical quality flag:

- `period`
- `demand_mwh`
- `solar_generation_mwh`
- `wind_generation_mwh`
- `renewable_data_complete` (historical quality flag)
- `solar_wind_generation_mwh`
- `residual_demand_after_solar_wind_mwh`
- `solar_wind_share_pct`

## Safe API-Key Handling

Set the EIA key only in your shell environment. The historical script reads
`EIA_API_KEY` from the environment and does not read it from a file.

PowerShell example:

```powershell
$secureKey = Read-Host "Enter EIA API key" -AsSecureString
$env:EIA_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
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

Remove the key from the current shell after the download:

```powershell
Remove-Item Env:EIA_API_KEY
Remove-Variable secureKey
```

## Expected Command Results

For the confirmed EIA source responses, the downloader should complete all
pages, report 26,304 demand rows, warn that 52,560 renewable rows are available
instead of the theoretical 52,608, and then save both raw files.

The validator should report:

```text
Download integrity: PASS
Demand coverage: PASS
Renewable source coverage: WARNING
Missing renewable timestamps: 24
Unexpected gaps: none
Overall result: PASS WITH DOCUMENTED SOURCE COVERAGE WARNING
```

The builder should write 26,304 processed rows and report 26,280 complete and
24 incomplete renewable rows.

## How Pagination Works

EIA JSON API responses are limited to 5,000 rows per request. A three-year
hourly file has more rows than that, so the script asks for the data in pages.

The first request starts at `offset=0` and asks for up to `length=5000` rows.
EIA also returns `response.total`, which says how many matching source rows
exist for the full request. If `response.total` is larger than the rows already
received, the script asks for the next page using a larger offset. It keeps
going until the number downloaded exactly matches `response.total`.

The script separately calculates theoretical complete hourly coverage. A
smaller EIA total produces a visible source-coverage warning, but it does not
make complete API pages look like a pagination failure. Changing totals,
duplicate rows, failed pages, or stopping before `response.total` remain hard
failures.

Demand has one row per hour. Solar and wind generation have two rows per hour:
one for `SUN` and one for `WND`.

## Documented Source Gap

Official EIA responses omit both `SUN` and `WND` from `2024-11-02T08` through
`2024-11-03T07`, inclusive. That is 24 demand timestamps and 48 renewable rows.
Source systems can contain gaps because publication pipelines may receive no
observation for a period even when the API itself is operating correctly.

The workflow preserves these values as null. Filling them with zero would
incorrectly assert that reported generation was zero rather than unavailable.
The processed CSV keeps all demand hours and sets `renewable_data_complete` to
`False` for the affected rows. Solar, wind, combined generation, residual
demand, and renewable share remain null there.

Later demand forecasting may retain these rows when it uses only complete
demand-derived features. Any feature or evaluation that depends on renewable
values must filter or otherwise handle `renewable_data_complete` explicitly in
a documented, chronological pipeline. Renewable analysis must exclude or
separately report incomplete rows; it must not silently treat them as zero.

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
- `src\validate_eia_history.py` must report `Download integrity: PASS`,
  `Demand coverage: PASS`, and
  `Overall result: PASS WITH DOCUMENTED SOURCE COVERAGE WARNING`.
- `src\build_historical_dataset.py` must write `data/processed/eia_ciso_hourly_2022_2024.csv`.
- The builder must report 26,280 complete and 24 incomplete renewable rows.
- The processed CSV should then be inspected with a focused validation step
  before modeling or feature engineering begins.

If any validation check fails, stop and inspect the raw files and error output.
Do not fill missing hours, remove duplicate rows, or change units without a
clear documented reason.
