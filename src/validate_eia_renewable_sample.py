"""Validate the local EIA CISO solar and wind generation sample.

Run this after fetching the renewable sample. The checks are intentionally
simple and readable so data problems are visible before any modeling work.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_RENEWABLE_PATH = Path(
    "data/raw/eia_ciso_hourly_renewable_generation_sample.json"
)
DEFAULT_DEMAND_PATH = Path("data/raw/eia_ciso_hourly_demand_sample.json")

EXPECTED_ROW_FIELDS = {
    "period",
    "respondent",
    "respondent-name",
    "fueltype",
    "type-name",
    "value",
    "value-units",
}
EXPECTED_RESPONDENT = "CISO"
EXPECTED_FUELS = {"SUN", "WND"}
EXPECTED_ROW_COUNT = 24 * 7 * len(EXPECTED_FUELS)


def parse_period(value: Any, row_number: int) -> datetime | None:
    """Parse EIA hourly period strings such as 2024-01-01T00."""
    if not isinstance(value, str):
        print(f"- Row {row_number}: period is not a string: {value!r}")
        return None

    try:
        return datetime.strptime(value, "%Y-%m-%dT%H")
    except ValueError:
        print(f"- Row {row_number}: period is not in YYYY-MM-DDTHH format: {value!r}")
        return None


def compact_counts(values: Counter[str]) -> str:
    """Make unique values easy to scan in the terminal output."""
    if not values:
        return "none observed"
    return ", ".join(f"{value} ({count})" for value, count in sorted(values.items()))


def load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load rows from either a normal one-page response or a pagination wrapper."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("Top-level JSON value is not an object.")

    if "pages" in payload:
        pages = payload.get("pages")
        if not isinstance(pages, list):
            raise ValueError("Pagination wrapper has a pages field, but it is not a list.")

        all_rows: list[dict[str, Any]] = []
        for page_number, page in enumerate(pages, start=1):
            if not isinstance(page, dict):
                raise ValueError(f"Page {page_number} is not a JSON object.")
            all_rows.extend(rows_from_response(page, f"page {page_number}"))
        return payload, all_rows

    return payload, rows_from_response(payload, "response")


def rows_from_response(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    """Read response.data from one EIA response object."""
    response = payload.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"{label} does not contain a response object.")

    rows = response.get("data")
    if not isinstance(rows, list):
        raise ValueError(f"{label} does not contain response.data as a list.")

    dict_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {index} is not an object.")
        dict_rows.append(row)

    return dict_rows


def load_demand_periods(path: Path) -> set[datetime]:
    """Load the timestamps from the already validated demand sample."""
    _, demand_rows = load_rows(path)
    periods: set[datetime] = set()
    for index, row in enumerate(demand_rows, start=1):
        period = parse_period(row.get("period"), index)
        if period is not None:
            periods.add(period)
    return periods


def gap_report(periods: list[datetime]) -> list[str]:
    """Find missing hourly steps within one fuel category."""
    gaps: list[str] = []
    unique_periods = sorted(set(periods))

    for previous, current in zip(unique_periods, unique_periods[1:]):
        step = current - previous
        if step != timedelta(hours=1):
            gaps.append(
                f"{previous.strftime('%Y-%m-%dT%H')} to "
                f"{current.strftime('%Y-%m-%dT%H')} ({step})"
            )

    return gaps


def validate_sample(renewable_path: Path, demand_path: Path) -> int:
    """Run the renewable checks and print a compact report."""
    try:
        payload, rows = load_rows(renewable_path)
        demand_periods = load_demand_periods(demand_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1

    top_level_keys = sorted(payload)
    periods_by_fuel: dict[str, list[datetime]] = defaultdict(list)
    all_periods: list[datetime] = []
    row_field_names: set[str] = set()
    unexpected_fields: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    null_or_missing: list[str] = []
    combo_counts: Counter[tuple[Any, Any]] = Counter()
    respondents: Counter[str] = Counter()
    fuel_types: Counter[str] = Counter()
    fuel_names: Counter[str] = Counter()
    labels_by_fuel: dict[str, Counter[str]] = defaultdict(Counter)
    units: Counter[str] = Counter()
    non_numeric_values: list[str] = []

    for index, row in enumerate(rows, start=1):
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

        period = parse_period(row.get("period"), index)
        fuel_type = row.get("fueltype")

        if period is not None:
            all_periods.append(period)
            if isinstance(fuel_type, str):
                periods_by_fuel[fuel_type].append(period)

        combo_counts[(row.get("period"), fuel_type)] += 1

        if isinstance(row.get("respondent"), str):
            respondents[row["respondent"]] += 1
        if isinstance(fuel_type, str):
            fuel_types[fuel_type] += 1
        fuel_label = row.get("type-name")
        if isinstance(fuel_label, str):
            fuel_names[fuel_label] += 1
            if isinstance(fuel_type, str):
                labels_by_fuel[fuel_type][fuel_label] += 1
        if isinstance(row.get("value-units"), str):
            units[row["value-units"]] += 1

        try:
            float(row.get("value"))
        except (TypeError, ValueError):
            non_numeric_values.append(f"row {index}: value {row.get('value')!r}")

    duplicate_combos = {
        combo: count for combo, count in combo_counts.items() if count > 1
    }
    missing_expected_fuels = EXPECTED_FUELS - set(fuel_types)
    unexpected_fuels = set(fuel_types) - EXPECTED_FUELS
    all_ciso = set(respondents) == {EXPECTED_RESPONDENT}

    gaps_by_fuel = {
        fuel_type: gap_report(periods)
        for fuel_type, periods in sorted(periods_by_fuel.items())
    }
    fuels_with_gaps = {
        fuel_type: gaps for fuel_type, gaps in gaps_by_fuel.items() if gaps
    }

    renewable_periods = set(all_periods)
    missing_from_renewables = sorted(demand_periods - renewable_periods)
    extra_renewable_periods = sorted(renewable_periods - demand_periods)
    timestamps_align_with_demand = (
        not missing_from_renewables and not extra_renewable_periods
    )

    print("EIA CISO hourly renewable generation sample validation")
    print(f"Renewable file: {renewable_path}")
    print(f"Demand file for timestamp alignment: {demand_path}")
    print()

    print("Structure")
    print(f"- Top-level keys: {', '.join(top_level_keys)}")
    print("- Data rows section: response.data, or pages[].response.data for paginated files")
    print()

    print("Validation results")
    print(f"- Total rows: {len(rows)}")
    print(f"- Expected rows for 7 days x 24 hours x 2 fuels: {EXPECTED_ROW_COUNT}")
    if all_periods:
        print(f"- Earliest timestamp: {min(all_periods).strftime('%Y-%m-%dT%H')}")
        print(f"- Latest timestamp: {max(all_periods).strftime('%Y-%m-%dT%H')}")
    else:
        print("- Earliest timestamp: not available")
        print("- Latest timestamp: not available")
    print(f"- All rows respondent CISO: {'yes' if all_ciso else 'no'}")
    print(f"- Respondent values: {compact_counts(respondents)}")
    print(f"- Available fuel categories: {compact_counts(fuel_types)}")
    print(f"- Fuel labels: {compact_counts(fuel_names)}")
    print("- Fuel labels by code:")
    for fuel_type in sorted(EXPECTED_FUELS | set(labels_by_fuel)):
        print(f"  - {fuel_type}: {compact_counts(labels_by_fuel[fuel_type])}")
    print(
        "- Missing expected fuel categories: "
        f"{'none' if not missing_expected_fuels else ', '.join(sorted(missing_expected_fuels))}"
    )
    print(
        "- Unexpected fuel categories: "
        f"{'none' if not unexpected_fuels else ', '.join(sorted(unexpected_fuels))}"
    )
    print(f"- Units: {compact_counts(units)}")
    print(f"- Observed row fields: {', '.join(sorted(row_field_names))}")
    print(f"- Missing or null values: {'none' if not null_or_missing else len(null_or_missing)}")
    if null_or_missing:
        for issue in null_or_missing[:10]:
            print(f"  - {issue}")
        if len(null_or_missing) > 10:
            print(f"  - ... {len(null_or_missing) - 10} more")
    print(
        "- Duplicate timestamp/fuel combinations: "
        f"{'none' if not duplicate_combos else len(duplicate_combos)}"
    )
    print("- Hourly gaps by fuel:")
    for fuel_type in sorted(EXPECTED_FUELS | set(periods_by_fuel)):
        gaps = gaps_by_fuel.get(fuel_type, [])
        print(f"  - {fuel_type}: {'none' if not gaps else '; '.join(gaps)}")
    print(
        "- Renewable timestamps align with demand sample: "
        f"{'yes' if timestamps_align_with_demand else 'no'}"
    )
    if missing_from_renewables:
        preview = ", ".join(period.strftime("%Y-%m-%dT%H") for period in missing_from_renewables[:10])
        print(f"  - Demand timestamps missing from renewables: {preview}")
    if extra_renewable_periods:
        preview = ", ".join(period.strftime("%Y-%m-%dT%H") for period in extra_renewable_periods[:10])
        print(f"  - Renewable timestamps not in demand sample: {preview}")
    print(f"- Unexpected row fields: {'none' if not unexpected_fields else compact_counts(unexpected_fields)}")
    print(f"- Missing expected row fields: {'none' if not missing_fields else compact_counts(missing_fields)}")
    print(f"- Non-numeric generation values: {'none' if not non_numeric_values else len(non_numeric_values)}")

    passed = all(
        [
            len(rows) == EXPECTED_ROW_COUNT,
            all_periods,
            all_ciso,
            not missing_expected_fuels,
            not unexpected_fuels,
            not null_or_missing,
            not duplicate_combos,
            not fuels_with_gaps,
            timestamps_align_with_demand,
            not unexpected_fields,
            not missing_fields,
            not non_numeric_values,
        ]
    )

    print()
    print(f"Overall result: {'PASS' if passed else 'CHECK ISSUES ABOVE'}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the local EIA CISO hourly solar/wind sample."
    )
    parser.add_argument(
        "renewable_path",
        nargs="?",
        default=DEFAULT_RENEWABLE_PATH,
        type=Path,
        help=f"Path to the renewable sample JSON. Default: {DEFAULT_RENEWABLE_PATH}",
    )
    parser.add_argument(
        "--demand-path",
        default=DEFAULT_DEMAND_PATH,
        type=Path,
        help=f"Path to the demand sample JSON. Default: {DEFAULT_DEMAND_PATH}",
    )
    args = parser.parse_args()
    return validate_sample(args.renewable_path, args.demand_path)


if __name__ == "__main__":
    raise SystemExit(main())
