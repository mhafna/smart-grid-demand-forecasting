# EIA California ISO Hourly Demand Sample Profile

Generated from local file only: `data/raw/eia_ciso_hourly_demand_sample.json`

No network requests were made for this profile, and the raw JSON was not changed.

## JSON Structure

The downloaded file is a JSON object with these top-level sections:

- `response`: EIA response metadata and the actual data rows.
- `request`: EIA request metadata, including parameter names. The validator hides the API key value.
- `apiVersion`: EIA API version metadata.
- `ExcelAddInVersion`: EIA Excel add-in version metadata.

Inside `response`, the main sections are:

- `total`: reported number of matching rows.
- `dateFormat`: timestamp format metadata.
- `frequency`: reported data frequency.
- `data`: the list of hourly records.
- `description`: route description supplied by EIA.

The data-row section is `response.data`.

## What One Row Represents

One row represents one hourly EIA region-data demand observation for the California Independent System Operator balancing authority.

In this local sample, each row has:

- one hour in `period`,
- respondent code `CISO`,
- demand measure type `D`,
- one reported demand value,
- units from EIA's `value-units` field.

## Observed Fields

Observed row fields:

- `period`
- `respondent`
- `respondent-name`
- `type`
- `type-name`
- `value`
- `value-units`

No unexpected row fields were observed in the sample.

## Row Count And Date Range

- Row count: 168
- Earliest timestamp: `2024-01-01T00`
- Latest timestamp: `2024-01-07T23`
- Frequency metadata: `hourly`
- Date format metadata: `YYYY-MM-DD"T"HH24`

## Units

The observed unit is:

- `megawatthours` in all 168 rows

## Data-Quality Findings

The local sample passed the validation checks:

- All rows belong to respondent `CISO`.
- All rows have demand type `D`.
- All rows use type name `Demand`.
- All rows use respondent name `California Independent System Operator`.
- All rows have the expected fields.
- No null or missing values were found.
- No duplicate `period` / `respondent` / `type` combinations were found.
- No duplicate timestamps were found.
- Timestamps are sorted chronologically.
- Timestamps are hourly with no gaps.
- Demand values are numeric.

## Validation Script Results

Command run:

```powershell
.\.venv\Scripts\python.exe src\validate_eia_sample.py data\raw\eia_ciso_hourly_demand_sample.json
```

Output:

```text
EIA CISO hourly demand sample validation
File: data\raw\eia_ciso_hourly_demand_sample.json

Structure
- Top-level keys: ExcelAddInVersion, apiVersion, request, response
- Response keys: data, dateFormat, description, frequency, total
- Request keys: command, params
- Request parameter keys: api_key, data, end, facets, frequency, length, offset, sort, start
- API key present in request parameters: yes (value intentionally hidden)
- Data rows section: response.data

Validation results
- Total rows: 168
- Earliest timestamp: 2024-01-01T00
- Latest timestamp: 2024-01-07T23
- All rows respondent CISO: yes
- Respondent values: CISO (168)
- Respondent-name values: California Independent System Operator (168)
- All rows demand type D: yes
- Type values: D (168)
- Type-name values: Demand (168)
- Reported units: megawatthours (168)
- Observed row fields: period, respondent, respondent-name, type, type-name, value, value-units
- Null or missing values: none
- Duplicate timestamp/respondent/type combinations: none
- Duplicate timestamps: none
- Timestamps sorted chronologically: yes
- Timestamps hourly with no gaps: yes
- Unexpected row fields: none
- Missing expected row fields: none
- Non-numeric demand values: none

Metadata notes
- response.total: 168
- response.frequency: hourly
- response.dateFormat: YYYY-MM-DD"T"HH24
- Timestamp timezone: not stated in the period strings or validated by this local file

Overall result: PASS
```

## Confirmed Facts

- The sample contains exactly 168 hourly rows.
- The sample covers seven complete calendar days from `2024-01-01T00` through `2024-01-07T23`.
- The rows are chronological and form a continuous hourly series.
- The rows are for California ISO demand only, based on `respondent = CISO` and `type = D`.
- The observed unit is `megawatthours`.
- The raw response includes request metadata, and the validator does not print the API key value.

## Unresolved Issues Before Larger Downloads

- The timestamp timezone is still unresolved. The raw `period` strings do not include a timezone offset or timezone name.
- Before downloading several years of data, confirm EIA's intended timezone for this RTO hourly `period` field from official EIA documentation or support.
- Confirm whether `value-units = megawatthours` should be interpreted as hourly energy for each period or handled as an hourly demand-like series exactly as EIA labels it.
- Confirm whether future large downloads should page through results using `length` and `offset`, because EIA responses can have row limits.
