# EIA API Contract: California ISO Hourly Grid Sample

Last verified: 2026-06-26

Sources used:
- EIA API technical documentation: https://www.eia.gov/opendata/documentation.php
- EIA OpenAPI YAML download linked from the technical documentation: https://www.eia.gov/opendata/eia-api-swagger.zip
- EIA Open Data browser: https://www.eia.gov/opendata/browser/electricity/rto/fuel-type-data
- EIA survey page for Form EIA-930: https://www.eia.gov/survey/
- Form EIA-930 data format and transmittal instructions: https://www.eia.gov/survey/form/eia_930/instructions.pdf
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

Hourly net generation by energy source is available through the EIA RTO fuel-type route. The EIA OpenAPI YAML lists both the metadata route and the data route:

```text
GET https://api.eia.gov/v2/electricity/rto/fuel-type-data
GET https://api.eia.gov/v2/electricity/rto/fuel-type-data/data
```

The OpenAPI YAML identifies the data route parameters as `data`, `facets`, `frequency`, `start`, `end`, `sort`, `length`, and `offset`; the global API security scheme requires `api_key` as a query parameter.

Parameters for the seven-day California ISO solar and wind sample:

```text
api_key=<your key>
frequency=hourly
data[]=value
facets[respondent][]=CISO
facets[fueltype][]=SUN
facets[fueltype][]=WND
start=2024-01-01T00
end=2024-01-07T23
sort[0][column]=period
sort[0][direction]=asc
sort[1][column]=fueltype
sort[1][direction]=asc
length=5000
offset=0
```

Expected row count for this narrow sample is 336 rows: 168 hourly timestamps times 2 fuel categories. The fetch script still checks EIA's `response.total` and can request additional pages if EIA ever returns fewer rows than the total.

Expected response fields:

| Field | Meaning |
| --- | --- |
| `period` | Hour timestamp, using EIA API's hourly period format |
| `respondent` | Balancing authority code, expected `CISO` |
| `respondent-name` | Balancing authority name |
| `fueltype` | Fuel type code |
| `fueltype-name` | Fuel type name |
| `value` | Reported hourly generation value |
| `value-units` | Unit for `value` |

Verified fuel type identifiers from Form EIA-930 instructions for Data Type `NG` net generation by energy source:

- `SUN`: solar without integrated battery storage
- `WND`: wind without integrated battery storage

Related EIA-930 codes that are not requested by the sample:

- `SNB`: solar with integrated battery storage
- `WNB`: wind with integrated battery storage
- `BAT`: battery storage

Measurement unit:

- Form EIA-930 instructions say to report hourly integrated values in megawatts by hour-ending time and to round reported megawatt data to the nearest integer.
- The API row's own `value-units` field must still be checked after download. The matching demand sample reported `megawatthours`, but the renewable sample should be validated from its own raw response instead of assumed.

Output target for the renewable fetch script:

```text
data/raw/eia_ciso_hourly_renewable_generation_sample.json
```

## Timestamp Format And Timezone

- EIA API v2 data responses include a `dateFormat` field. The validated demand sample returned `YYYY-MM-DD"T"HH24`, which corresponds to period strings such as `2024-01-01T00`.
- Form EIA-930 instructions say respondents report hourly date-time stamps using Coordinated Universal Time (UTC) that correlates with the respondent's local time.
- Form EIA-930 instructions also say the CSV `HR#` fields represent one value for each sequential hour of the day in the respondent's local time, for example hour-ending 5 a.m. for `HR5`.
- For this project, treat the API `period` values from the RTO hourly routes as UTC hourly timestamps for joining demand and renewable rows.
- Important distinction: the API route exposes a normalized hourly period, while the underlying EIA-930 collection is based on balancing-authority local hour-ending reporting converted to UTC. Keep that distinction visible when documenting features or later converting to local time.
- The downloaded API period strings do not include a literal `Z` or numeric timezone offset, so the UTC interpretation should be documented as coming from EIA-930 instructions and route context, not from the timestamp string alone.

## Pagination And Row Limits

- EIA's API documentation says JSON responses return at most 5,000 rows per request.
- Use `length` to limit the number of rows returned.
- Use `offset` to page through results.
- EIA returns `response.total`, which is the total number of rows responsive to the request even when `length` asks for only a subset.
- The sample script uses `length=168`, which is seven days times 24 hours for one demand series.
- The script also validates that the requested date window is no more than seven days.
- The renewable script uses a 5,000-row page size, which is far above the expected 336 rows for this narrow sample, but it checks totals and can continue with `offset` if needed.

## Unresolved Details

- A live keyed facet response was not available during this setup because no valid `EIA_API_KEY` was used. After you add a key, verify the returned renewable rows contain `respondent=CISO`, `fueltype` values `SUN` and `WND`, and EIA's own `value-units`.
- The API route's timestamp strings do not carry an explicit timezone suffix. The UTC convention is supported by EIA-930 instructions, but any later local-time feature engineering must carefully handle daylight saving time and hour-ending semantics.
