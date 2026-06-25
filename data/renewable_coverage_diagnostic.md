# Renewable Coverage Diagnostic

## Current Known Discrepancy

The main historical downloader stopped safely before saving files.

- Period: 2022-01-01 through 2024-12-31, inclusive
- Demand rows downloaded: 26,304 of 26,304
- EIA reported renewable rows: 52,560
- Theoretical renewable rows: 52,608
- Difference: 48 renewable rows

The downloader should remain strict until the missing coverage is understood.
No rows should be interpolated, filled, deleted, or fabricated.

## Diagnostic Script

Run:

```powershell
python src\diagnose_eia_renewable_coverage.py
```

The script reads `EIA_API_KEY` from the environment, prints a report, and does
not write or alter the main historical raw files.

## What The Diagnostic Checks

The diagnostic uses EIA's returned `response.total` as the API pagination truth.
It separately compares those totals with theoretical complete hourly coverage.

It checks:

- CISO `SUN` totals by year
- CISO `WND` totals by year
- monthly totals only inside deficient years
- row-level coverage only for affected months
- missing timestamp/fuel combinations
- duplicate timestamp/fuel combinations
- whether focused pagination matched EIA `response.total`
- whether the first and last requested days look complete

## Exact Affected Timestamps

Not yet determined in this repository because the diagnostic has not been run
with a valid `EIA_API_KEY`.

After running the script, copy the affected timestamp/fuel combinations from
the `Missing timestamp/fuel combinations` lines into this section. Do not infer
the missing hours from the 48-row difference alone.

## SUN, WND, Or Both

Not yet determined. The diagnostic requests `SUN` and `WND` separately by year,
then by month for any deficient year. This will show whether the shortfall is
solar only, wind only, or both.

## Source Missingness Versus Script Defect

Current evidence:

- Demand coverage completed exactly at the theoretical 26,304 rows.
- The renewable discrepancy came from EIA's own reported total being 52,560,
  not from a partial saved file.
- The main downloader did not silently drop rows; it stopped before writing.

The likely cause must remain unclassified until the diagnostic confirms whether
focused pagination is complete, whether duplicates exist, and whether the exact
missing timestamp/fuel combinations are inside the requested date boundaries.

## Safe Project Options

After the diagnostic identifies the exact issue, safe options include:

- Keep the strict downloader unchanged and choose a different complete
  historical window.
- Keep the strict downloader unchanged and document that this EIA route has
  missing renewable source rows for the affected timestamp/fuel combinations.
- Add an explicit, documented allowlist for known missing EIA source rows only
  if the project decides that downstream analysis can handle an incomplete
  renewable series.
- Use a different official source or route for renewable generation if it
  provides complete CISO hourly `SUN` and `WND` coverage for the chosen period.

Do not automatically interpolate, fill, or delete observations.
