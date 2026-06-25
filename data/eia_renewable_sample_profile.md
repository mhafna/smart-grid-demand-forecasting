# EIA California ISO Hourly Renewable Generation Sample Profile

Generated from local file only: `data/raw/eia_ciso_hourly_renewable_generation_sample.json`

No network requests were made for this profile, and the raw JSON was not changed.

## Row Count And Date Range

- Row count: 336
- Expected row count: 336 rows, from 7 days x 24 hours x 2 fuel categories
- Earliest timestamp: `2024-01-01T00`
- Latest timestamp: `2024-01-07T23`

## Fuel Codes And Labels

The machine-readable fuel category field is `fueltype`.

Observed fuel codes and labels:

| Fuel code | Observed label field | Label | Rows |
| --- | --- | --- | --- |
| `SUN` | `type-name` | Solar | 168 |
| `WND` | `type-name` | Wind | 168 |

The API response uses `type-name` as the human-readable fuel label field. It does not use `fueltype-name` in this downloaded sample.

## Units

- `megawatthours` in all 336 rows

## Gaps And Duplicates

- Duplicate timestamp/fuel combinations: none
- Hourly gaps for `SUN`: none
- Hourly gaps for `WND`: none
- Missing or null values: none
- Unexpected row fields: none
- Missing expected row fields: none
- Non-numeric generation values: none

## Timestamp Alignment With Demand

The renewable timestamps align with the existing demand sample:

- Demand sample path: `data/raw/eia_ciso_hourly_demand_sample.json`
- Renewable sample path: `data/raw/eia_ciso_hourly_renewable_generation_sample.json`
- Renewable timestamps missing from demand: none
- Demand timestamps missing from renewables: none

This means the renewable rows can later be joined to the demand rows by `period` after separating or pivoting the two `fueltype` categories.

## Validation Result

Command run:

```powershell
& 'C:\Users\marya\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' src\validate_eia_renewable_sample.py data\raw\eia_ciso_hourly_renewable_generation_sample.json
```

Output:

```text
EIA CISO hourly renewable generation sample validation
Renewable file: data\raw\eia_ciso_hourly_renewable_generation_sample.json
Demand file for timestamp alignment: data\raw\eia_ciso_hourly_demand_sample.json

Structure
- Top-level keys: ExcelAddInVersion, apiVersion, request, response
- Data rows section: response.data, or pages[].response.data for paginated files

Validation results
- Total rows: 336
- Expected rows for 7 days x 24 hours x 2 fuels: 336
- Earliest timestamp: 2024-01-01T00
- Latest timestamp: 2024-01-07T23
- All rows respondent CISO: yes
- Respondent values: CISO (336)
- Available fuel categories: SUN (168), WND (168)
- Fuel labels: Solar (168), Wind (168)
- Fuel labels by code:
  - SUN: Solar (168)
  - WND: Wind (168)
- Missing expected fuel categories: none
- Unexpected fuel categories: none
- Units: megawatthours (336)
- Observed row fields: fueltype, period, respondent, respondent-name, type-name, value, value-units
- Missing or null values: none
- Duplicate timestamp/fuel combinations: none
- Hourly gaps by fuel:
  - SUN: none
  - WND: none
- Renewable timestamps align with demand sample: yes
- Unexpected row fields: none
- Missing expected row fields: none
- Non-numeric generation values: none

Overall result: PASS
```
