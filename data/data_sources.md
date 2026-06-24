# Data Sources Research

Project goal: short-term electricity demand forecasting with renewable-aware grid planning for one clearly defined grid region.

This document evaluates official public sources only. No data has been downloaded, no API has been called for project data, and no modeling code has been written.

## Summary Recommendation

Use the U.S. Energy Information Administration (EIA) Hourly Electric Grid Monitor / U.S. Electric System Operating Data as the primary dataset, with California ISO as the preferred grid region.

Initial scope:

- Region: California ISO, after confirming the exact EIA region/facet code.
- Time period: approximately three complete recent calendar years, such as 2021-2023 or 2022-2024, after verifying complete availability.
- Resolution: hourly.
- Target: next-hour or next-day-ahead hourly electricity demand for the selected region.

This is the best starting point because it is official, national, API-friendly, and likely easier to make reproducible than manually collecting many CAISO web reports. CAISO should remain the main cross-check and possible later upgrade source.

## Candidate 1: EIA Hourly Electric Grid Monitor / U.S. Electric System Operating Data

Source links:

- EIA Open Data API dashboard for electricity RTO data: https://www.eia.gov/opendata/browser/electricity/rto/
- EIA API technical documentation: https://www.eia.gov/opendata/documentation.php
- EIA copyright and reuse guidance: https://www.eia.gov/about/copyrights_reuse.php

Official provider:

- U.S. Energy Information Administration (EIA), a U.S. government statistical agency.

Available variables:

- EIA lists "U.S. Electric System Operating Data" under its Open Data bulk downloads, split into "2019-present" and "before 2019" files.
- The EIA electricity category covers electricity demand and generation-related data.
- For this project, the expected useful variables are regional electricity demand and regional generation by energy source, including solar and wind if available for the selected region.
- Requires verification: the exact API route, facet names, region code for California ISO, and exact column names for demand, solar generation, and wind generation.

Time resolution:

- Expected to support hourly data because this product is the Hourly Electric Grid Monitor / electric system operating data.
- Requires verification: exact frequency value to use in the API request.

Available date range:

- EIA's Open Data dashboard lists bulk files for "U.S. Electric System Operating Data (2019-present)" and "U.S. Electric System Operating Data (before 2019)."
- Requires verification: the exact earliest available date and whether the preferred three-year period has complete California ISO demand, solar, and wind records.

Expected download or API method:

- API method: use EIA API v2 after confirming the correct route and facets. EIA documentation shows that API v2 uses route-style URLs, `/data` for data requests, `facets[...]` for filtering, `frequency`, `start`, `end`, `sort`, `offset`, and `length`.
- Bulk method: EIA also provides bulk ZIP downloads for the electric system operating data, but that should not be used until the exact scope is chosen.
- EIA documentation indicates an API key is used in API calls.

Whether demand, solar, and wind can be aligned:

- Likely yes, if the selected EIA RTO region has hourly demand and hourly generation-by-source data for the same timestamps.
- Requires verification before downloading: confirm that California ISO demand, solar, and wind are available at the same hourly timestamps and use the same time zone or documented timestamp convention.

Advantages:

- Official U.S. government source.
- API and bulk download options support reproducible workflows.
- Nationally standardized source, which may be easier than learning CAISO-specific reports first.
- Suitable for a portfolio project because the same method can later be repeated for another region.
- EIA reuse guidance says U.S. government publications are public domain, while recommending source acknowledgment.

Limitations:

- The EIA browser/API metadata must be checked carefully before choosing exact fields.
- Renewable categories may be aggregated or named differently than CAISO's native categories.
- Time zone handling must be verified.
- EIA may lag or transform ISO-reported data compared with CAISO's native systems.

Licensing or attribution considerations:

- EIA says its U.S. government publications are public domain and may be used or distributed, but reused information should acknowledge EIA and include a publication date when possible.
- Do not use the EIA logo without permission.

Likely difficulty for a beginner:

- Moderate.
- Easier than CAISO OASIS once the correct API route and filters are identified, but the first metadata lookup step must be done carefully.

## Candidate 2: California ISO Public Historical Data

Source links:

- CAISO Today's Outlook demand page: https://www.caiso.com/todays-outlook/demand
- CAISO Today's Outlook supply page: https://www.caiso.com/todays-outlook/supply
- CAISO Market reports library: https://www.caiso.com/library/market-reports
- CAISO OASIS: https://oasis.caiso.com/
- CAISO privacy, terms of use, and API terms: https://www.caiso.com/privacy-terms-of-use

Official provider:

- California Independent System Operator (California ISO / CAISO).

Available variables:

- CAISO Today's Outlook demand page documents system demand, forecasted demand, net demand, hour-ahead forecast, day-ahead forecast, day-ahead net forecast, demand response, and resource adequacy-related fields.
- CAISO Today's Outlook supply page documents current demand, current renewables, current solar, current wind, supply trend, renewables trend, hybrids trend, batteries trend, and imports trend.
- CAISO OASIS lists report categories including CAISO Demand Forecast, Wind and Solar Forecast, System Load and Resource Schedules, Wind and Solar Summary, and related market and system reports.
- Requires verification: exact OASIS report names, query parameters, downloadable file format, field names, and whether each chosen report provides actual demand, actual solar, and actual wind for the same timestamps.

Time resolution:

- Today's Outlook documents several 5-minute average series for demand, net demand, supply, and renewables trends.
- It also documents day-ahead forecast values as 1-hour averages.
- OASIS report resolution must be verified for the exact reports selected.

Available date range:

- CAISO OASIS states that current data can be accessed without the interface using report URLs, and that historical data beyond 39 months and as far back as 2016 is available through the Historical OASIS Data Downloader.
- Requires verification: whether the needed demand, solar, and wind reports are consistently available for the chosen three complete years.

Expected download or API method:

- OASIS web interface.
- Report URLs for current OASIS data.
- Historical OASIS Data Downloader for older historical data.
- CAISO Developer site for technical specifications, acceptable use, connectivity, sample code, and API details; CAISO says self-registration is required for the developer site.

Whether demand, solar, and wind can be aligned:

- Likely yes, because CAISO publishes demand, net demand, supply, renewable, wind, and solar information for its own balancing authority.
- Requires verification: exact reports and timestamps must be tested before committing to this as the main data source.

Advantages:

- Most region-specific source for California ISO.
- Directly supports renewable-aware questions such as demand minus wind and solar.
- Today's Outlook clearly defines net demand as demand minus wind and solar.
- OASIS is the official source CAISO points users to for official operational data.

Limitations:

- More complex for a beginner than EIA.
- Today's Outlook pages say their charts are informational, subject to change without notice, and should not be used for billing or operational planning; they point users to OASIS for official data.
- Historical download workflows may require learning OASIS-specific report names and parameters.
- Developer API details may require registration.

Licensing or attribution considerations:

- CAISO terms say website materials are provided as a public service and may be used if copyright/trademark notices remain intact and California ISO is credited.
- CAISO API terms provide a revocable, limited license and restrict excessive or adverse use.
- Any future use should cite California ISO and follow the current terms.

Likely difficulty for a beginner:

- Moderate to high.
- Good official source, but the reporting system is more specialized than EIA.

## Candidate 3: IESO Ontario Public Power Data

Source links:

- IESO Power Data: https://www.ieso.ca/power-data
- IESO Data Directory: https://www.ieso.ca/power-data/data-directory
- IESO Transmission-Connected Resources: https://www.ieso.ca/power-data/supply-overview/transmission-connected-generation
- IESO Terms of Use: https://www.ieso.ca/terms-of-use

Official provider:

- Independent Electricity System Operator (IESO), Ontario's electricity system operator.

Available variables:

- IESO Power Data includes actual and forecast electricity demand, supply mix, prices, Ontario demand, market demand, hourly output by fuel type, wind, solar, imports, and exports.
- The Data Directory includes "Ontario and Market Demand" and "Generator Output by Fuel Type" reports.
- The "Ontario and Market Demand" report provides Ontario demand and market demand for each hour of the day.
- The "Generator Output by Fuel Type" report provides hourly output grouped by fuel type for qualifying generating facilities.

Time resolution:

- Hourly demand is available in the Ontario and Market Demand report.
- Hourly output by fuel type is available through generator output by fuel type reporting.

Available date range:

- Ontario and Market Demand historical reports are listed as 2002-present, with a separate 1994-2002 historical demand report.
- Generator Output and Capability has monthly updated daily data from May 2019-present, with earlier hourly generator output and capability CSV files from 2010 to April 2019.
- Requires verification: exact date coverage for hourly generator output by fuel type, especially solar and wind, for a three-year period.

Expected download or API method:

- IESO Public Reports site, usually CSV or XML links from the Data Directory.
- Historical report links for older files.

Whether demand, solar, and wind can be aligned:

- Likely yes for Ontario, if hourly demand and hourly generation-by-fuel reports share compatible timestamps.
- Requires verification: whether solar and wind output fields are present and complete for the chosen years, and whether generator-output reports exclude embedded generation that matters for interpretation.

Advantages:

- Official system-operator source.
- Very beginner-friendly Data Directory pages.
- Clear hourly demand and generation-by-fuel reporting.
- Good meaningful advantage: it may be simpler to understand than CAISO OASIS while still supporting demand plus wind and solar analysis.

Limitations:

- It is Ontario, not California or a U.S. grid region.
- Transmission-connected reporting may exclude embedded/distribution-connected generation; IESO specifically notes that most solar facilities in Ontario are currently connected to the distribution system.
- It would shift the portfolio story away from California ISO.

Licensing or attribution considerations:

- Requires review of IESO Terms of Use before reuse in a published project.
- Use clear attribution to IESO and link to the source reports.

Likely difficulty for a beginner:

- Low to moderate.
- Easier documentation pages than CAISO OASIS, but the region is less aligned with the preferred California scope.

## Primary Dataset Recommendation

Primary dataset: EIA Hourly Electric Grid Monitor / U.S. Electric System Operating Data for California ISO.

Recommended first scope:

- One region: California ISO.
- Approximately three complete years: choose only after verifying complete hourly demand, solar, and wind availability. A likely first candidate is 2021-2023 or 2022-2024.
- One target: hourly electricity demand in megawatts.
- Renewable-aware variables: hourly solar generation and wind generation, if available and aligned for the same timestamps.

Why not start with CAISO directly:

- CAISO is the best native source for California, but OASIS is more specialized.
- CAISO Today's Outlook is useful for definitions and exploration, but CAISO itself says official data should come from OASIS.
- EIA is more likely to be beginner-friendly for a first reproducible data pipeline.

## Checks Required Before Downloading Anything

Before any data download or API call, verify:

1. The exact EIA API route for hourly regional operating data.
2. The exact California ISO region code or facet value in EIA.
3. The exact EIA field names for demand, solar, and wind.
4. The timestamp convention and time zone.
5. Whether demand, solar, and wind are all available for every hour in the chosen three-year period.
6. Whether daylight saving time creates missing or duplicate local hours.
7. Whether values are reported in MW, MWh, or another unit.
8. Whether forecasts are separate from actual values, so the target does not accidentally use future information.
9. The attribution text that should appear in the README and final project.

## Beginner-Friendly Interpretation

Recommended dataset:

- Use EIA's hourly electric grid data for California ISO as the first dataset, after confirming the exact API route and fields.

Why it fits:

- It is official, public, hourly, and likely contains the demand and renewable generation information needed for the project.
- It should let us build a clean first version before dealing with CAISO's more complex native reporting tools.

What one row would represent:

- One timestamp for one grid region, probably one hour for California ISO, with demand and possibly solar and wind values for that same hour.

Prediction target:

- The main prediction target should be future electricity demand, such as demand for the next hour or each hour of the next day.

Expected input variables:

- Past demand values.
- Calendar features such as hour of day, day of week, month, and holidays if added later.
- Past solar and wind generation values, if they are available at matching timestamps.
- Weather variables may be added later from a separate official or credible source, but they are not part of this dataset selection task.

What to verify first:

- Confirm that California ISO demand, solar, and wind are truly available from EIA for the same hourly timestamps and for the full chosen date range.
- Confirm the units and time zone before writing any code.
- Confirm that we use actual historical values correctly and do not leak future information into training features.
