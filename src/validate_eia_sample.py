"""Validate the local EIA California ISO hourly demand sample.

This script reads the raw JSON sample but does not modify it. It prints a
small profile of the file so we can decide whether the sample is clean enough
to use as the pattern for larger downloads.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_SAMPLE_PATH = Path("data/raw/eia_ciso_hourly_demand_sample.json")

# These are the row fields documented for the EIA RTO region-data demand route.
EXPECTED_ROW_FIELDS = {
    "period",
    "respondent",
    "respondent-name",
    "type",
    "type-name",
    "value",
    "value-units",
}

# These expected values come from the local sample request contract.
EXPECTED_RESPONDENT = "CISO"
EXPECTED_TYPE = "D"


def parse_period(value: Any, row_number: int) -> datetime | None:
    """Convert an EIA hourly period string into a datetime.

    EIA hourly periods in this sample look like "2024-01-01T00". The parsed
    datetime has no timezone because the raw string has no timezone marker.
    """
    if not isinstance(value, str):
        print(f"- Row {row_number}: period is not a string: {value!r}")
        return None

    try:
        return datetime.strptime(value, "%Y-%m-%dT%H")
    except ValueError:
        print(f"- Row {row_number}: period is not in YYYY-MM-DDTHH format: {value!r}")
        return None


def compact_counts(values: Counter[str]) -> str:
    """Format unique values without printing long raw records."""
    if not values:
        return "none observed"
    return ", ".join(f"{value} ({count})" for value, count in sorted(values.items()))


def validate_sample(path: Path) -> int:
    """Read the JSON file, run checks, and print a beginner-friendly report."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        print("FAIL: Top-level JSON value is not an object.")
        return 1

    top_level_keys = set(payload)
    response = payload.get("response")
    request = payload.get("request")

    if not isinstance(response, dict):
        print("FAIL: JSON does not contain a response object.")
        return 1

    rows = response.get("data")
    if not isinstance(rows, list):
        print("FAIL: response.data is missing or is not a list.")
        return 1

    print("EIA CISO hourly demand sample validation")
    print(f"File: {path}")
    print()

    print("Structure")
    print(f"- Top-level keys: {', '.join(sorted(top_level_keys))}")
    print(f"- Response keys: {', '.join(sorted(response))}")
    if isinstance(request, dict):
        request_keys = ", ".join(sorted(request))
        print(f"- Request keys: {request_keys}")
        params = request.get("params")
        if isinstance(params, dict):
            safe_param_keys = sorted(params)
            print(f"- Request parameter keys: {', '.join(safe_param_keys)}")
            if "api_key" in params:
                print("- API key present in request parameters: yes (value intentionally hidden)")
    print("- Data rows section: response.data")
    print()

    periods: list[datetime] = []
    period_strings: list[str] = []
    row_field_names: set[str] = set()
    unexpected_fields: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    null_or_missing: list[str] = []
    combo_counts: Counter[tuple[Any, Any, Any]] = Counter()
    respondents: Counter[str] = Counter()
    demand_types: Counter[str] = Counter()
    type_names: Counter[str] = Counter()
    respondent_names: Counter[str] = Counter()
    units: Counter[str] = Counter()
    non_numeric_values: list[str] = []

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            null_or_missing.append(f"row {index}: row is not an object")
            continue

        row_keys = set(row)
        row_field_names.update(row_keys)

        for field in row_keys - EXPECTED_ROW_FIELDS:
            unexpected_fields[field] += 1
        for field in EXPECTED_ROW_FIELDS - row_keys:
            missing_fields[field] += 1
            null_or_missing.append(f"row {index}: missing {field}")

        for field in EXPECTED_ROW_FIELDS & row_keys:
            if row.get(field) is None or row.get(field) == "":
                null_or_missing.append(f"row {index}: null or empty {field}")

        period_text = row.get("period")
        period = parse_period(period_text, index)
        if period is not None:
            periods.append(period)
            period_strings.append(period_text)

        respondent = row.get("respondent")
        demand_type = row.get("type")
        combo_counts[(period_text, respondent, demand_type)] += 1

        if isinstance(respondent, str):
            respondents[respondent] += 1
        if isinstance(demand_type, str):
            demand_types[demand_type] += 1
        if isinstance(row.get("type-name"), str):
            type_names[row["type-name"]] += 1
        if isinstance(row.get("respondent-name"), str):
            respondent_names[row["respondent-name"]] += 1
        if isinstance(row.get("value-units"), str):
            units[row["value-units"]] += 1

        value = row.get("value")
        try:
            float(value)
        except (TypeError, ValueError):
            non_numeric_values.append(f"row {index}: value {value!r}")

    duplicate_combos = {
        combo: count for combo, count in combo_counts.items() if count > 1
    }
    sorted_chronologically = periods == sorted(periods)

    hourly_gaps: list[str] = []
    duplicate_timestamps: list[str] = []
    unique_periods = sorted(set(periods))
    if len(unique_periods) != len(periods):
        seen_periods: Counter[datetime] = Counter(periods)
        duplicate_timestamps = [
            period.strftime("%Y-%m-%dT%H")
            for period, count in seen_periods.items()
            if count > 1
        ]

    for previous, current in zip(unique_periods, unique_periods[1:]):
        step = current - previous
        if step != timedelta(hours=1):
            hourly_gaps.append(
                f"{previous.strftime('%Y-%m-%dT%H')} to "
                f"{current.strftime('%Y-%m-%dT%H')} ({step})"
            )

    all_ciso = set(respondents) == {EXPECTED_RESPONDENT}
    all_demand = set(demand_types) == {EXPECTED_TYPE}
    all_rows_have_expected_fields = not missing_fields and not unexpected_fields
    values_are_numeric = not non_numeric_values
    no_nulls = not null_or_missing
    no_duplicate_combos = not duplicate_combos
    hourly_without_gaps = not hourly_gaps and not duplicate_timestamps

    print("Validation results")
    print(f"- Total rows: {len(rows)}")
    if periods:
        print(f"- Earliest timestamp: {min(periods).strftime('%Y-%m-%dT%H')}")
        print(f"- Latest timestamp: {max(periods).strftime('%Y-%m-%dT%H')}")
    else:
        print("- Earliest timestamp: not available")
        print("- Latest timestamp: not available")
    print(f"- All rows respondent CISO: {'yes' if all_ciso else 'no'}")
    print(f"- Respondent values: {compact_counts(respondents)}")
    print(f"- Respondent-name values: {compact_counts(respondent_names)}")
    print(f"- All rows demand type D: {'yes' if all_demand else 'no'}")
    print(f"- Type values: {compact_counts(demand_types)}")
    print(f"- Type-name values: {compact_counts(type_names)}")
    print(f"- Reported units: {compact_counts(units)}")
    print(f"- Observed row fields: {', '.join(sorted(row_field_names))}")
    print(f"- Null or missing values: {'none' if no_nulls else len(null_or_missing)}")
    if null_or_missing:
        for issue in null_or_missing[:10]:
            print(f"  - {issue}")
        if len(null_or_missing) > 10:
            print(f"  - ... {len(null_or_missing) - 10} more")
    print(
        "- Duplicate timestamp/respondent/type combinations: "
        f"{'none' if no_duplicate_combos else len(duplicate_combos)}"
    )
    print(f"- Duplicate timestamps: {'none' if not duplicate_timestamps else ', '.join(duplicate_timestamps)}")
    print(f"- Timestamps sorted chronologically: {'yes' if sorted_chronologically else 'no'}")
    print(f"- Timestamps hourly with no gaps: {'yes' if hourly_without_gaps else 'no'}")
    if hourly_gaps:
        for gap in hourly_gaps:
            print(f"  - Gap or non-hourly step: {gap}")
    print(f"- Unexpected row fields: {'none' if not unexpected_fields else compact_counts(unexpected_fields)}")
    print(f"- Missing expected row fields: {'none' if not missing_fields else compact_counts(missing_fields)}")
    print(f"- Non-numeric demand values: {'none' if values_are_numeric else len(non_numeric_values)}")
    if non_numeric_values:
        for issue in non_numeric_values[:10]:
            print(f"  - {issue}")

    print()
    print("Metadata notes")
    print(f"- response.total: {response.get('total')}")
    print(f"- response.frequency: {response.get('frequency')}")
    print(f"- response.dateFormat: {response.get('dateFormat')}")
    print("- Timestamp timezone: not stated in the period strings or validated by this local file")

    passed = all(
        [
            len(rows) > 0,
            periods,
            all_ciso,
            all_demand,
            no_nulls,
            no_duplicate_combos,
            hourly_without_gaps,
            sorted_chronologically,
            all_rows_have_expected_fields,
            values_are_numeric,
        ]
    )

    print()
    print(f"Overall result: {'PASS' if passed else 'CHECK ISSUES ABOVE'}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the local EIA CISO hourly demand JSON sample."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_SAMPLE_PATH,
        type=Path,
        help=f"Path to the raw sample JSON. Default: {DEFAULT_SAMPLE_PATH}",
    )
    args = parser.parse_args()
    return validate_sample(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
