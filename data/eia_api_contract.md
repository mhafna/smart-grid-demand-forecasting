# EIA API Contract: California ISO Hourly Grid Sample

Last verified: 2026-06-25

Sources used:
- EIA API technical documentation: https://www.eia.gov/opendata/documentation.php
- EIA OpenAPI YAML download linked from the technical documentation: https://www.eia.gov/opendata/eia-api-swagger.zip
- EIA API responses from `https://api.eia.gov/v2/electricity/rto/...` without an API key returned HTTP 403, confirming that a key is required before data or facet responses can be read.

## API Version And Key

- Current EIA API family: API v2.
- Current documentation page patch notes show API v2.1.12 in March 2026.
- The OpenAPI section still labels the downloadable specification as "API v2.1.0 (Current)".
- Route prefix: `https://api.eia.gov/v2/`
- API key: required. EIA says the key must be supplied as a URL parameter named `api_key`.
- Do not commit the real key. Store it locally as `EIA_API_KEY` in your shell environment or in an untracked `.env` file.

## California ISO Identifier

- Balancing authority / respondent identifier: `CISO`
- Human-readable name: California Independent System Operator

This identifier should be passed as:

```text
facets[respondent][]=CISO
```

## Hourly Electricity Demand Route

Official route from the EIA OpenAPI YAML:

```text
GET https://api.eia.gov/v2/electricity/rto/region-data/data
```

Minimum parameters for a small California ISO demand sample:

```text
api_key=<your key>
frequency=hourly
data[]=value
facets[respondent][]=CISO
facets[type][]=D
start=YYYY-MM-DDTHH
end=YYYY-MM-DDTHH
sort[0][column]=period
sort[0][direction]=asc
length=168
offset=0
```

Demand field identifier:

```text
facets[type][]=D
```

Expected response fields:

| Field | Meaning |
| --- | --- |
| `period` | Hour timestamp |
| `respondent` | Balancing authority code, expected `CISO` |
| `respondent-name` | Balancing authority name |
| `type` | Region-data measure code, expected `D` for demand |
| `type-name` | Region-data measure name |
| `value` | Reported hourly value |
| `value-units` | Unit for `value` |

Measurement unit:

- Demand values are reported in megawatthours according to the route's value-unit field for this EIA RTO data family.
- The script saves the raw response unchanged, including EIA's own `value-units` field, so the unit can be checked directly from the downloaded file.

## Hourly Solar And Wind Generation

Hourly solar and wind generation are available through a compatible EIA RTO route:

```text
GET https://api.eia.gov/v2/electricity/rto/fuel-type-data/data
```

Compatible parameters:

```text
api_key=<your key>
frequency=hourly
data[]=value
facets[respondent][]=CISO
facets[fueltype][]=SUN
facets[fueltype][]=WND
start=YYYY-MM-DDTHH
end=YYYY-MM-DDTHH
sort[0][column]=period
sort[0][direction]=asc
length=336
offset=0
```

Expected response fields:

| Field | Meaning |
| --- | --- |
| `period` | Hour timestamp |
| `respondent` | Balancing authority code, expected `CISO` |
| `respondent-name` | Balancing authority name |
| `fueltype` | Fuel type code |
| `fueltype-name` | Fuel type name |
| `value` | Reported hourly generation value |
| `value-units` | Unit for `value` |

Expected fuel type identifiers:

- `SUN`: solar
- `WND`: wind

The sample script intentionally fetches only hourly demand. Solar and wind are documented here so the compatible route is known, but they are not downloaded by the sample script.

## Timestamp Format And Timezone

- EIA API v2 data responses include a `dateFormat` field. For hourly RTO routes, the period format is expected to be `YYYY-MM-DDTHH`.
- Example parameter shape: `start=2024-01-01T00`.
- EIA's grid monitor application uses UTC chart time handling, but the downloaded OpenAPI YAML does not explicitly state the timezone for the `period` value. Treat the raw `period` values as EIA timestamps and confirm timezone from the downloaded response metadata before any modeling work.

## Pagination And Row Limits

- JSON responses return at most 5,000 rows per request.
- Use `length` to limit the number of rows returned.
- Use `offset` to page through results.
- The sample script uses `length=168`, which is seven days times 24 hours for one demand series.
- The script also validates that the requested date window is no more than seven days.

## Unresolved Details

- A live keyed facet response was not available during this setup because no valid `EIA_API_KEY` was provided. After you add a key, verify the returned `response.data` rows contain `respondent=CISO`, `type=D`, and EIA's own `value-units`.
- The exact timezone label is not present in the OpenAPI YAML. Confirm it against EIA response metadata or EIA support documentation before using the timestamps for modeling.
