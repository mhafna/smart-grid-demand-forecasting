"""Validate historical EIA CISO hourly demand and solar/wind raw JSON files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_DEMAND_PATH = Path("data/raw/eia_ciso_hourly_demand_2022_2024.json")
DEFAULT_RENEWABLE_PATH = Path(
    "data/raw/eia_ciso_hourly_renewable_generation_2022_2024.json"
)

EXPECTED_RESPONDENT = "CISO"
EXPECTED_DEMAND_TYPE = "D"
EXPECTED_FUELS = {"SUN", "WND"}
EXPECTED_DEMAND_FIELDS = {
    "period",
    "respondent",
    "respondent-name",
    "type",
    "type-name",
    "value",
    "value-units",
}
EXPECTED_RENEWABLE_FIELDS = {
    "period",
    "respondent",
    "respondent-name",
    "fueltype",
    "value",
    "value-units",
}


def parse_date(value: str, label: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY-MM-DD format.") from exc


def parse_period(value: Any, row_number: int, label: str) -> datetime | None:
    if not isinstance(value, str):
        print(f"- {label} row {row_number}: period is not a string: {value!r}")
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H")
    except ValueError:
        print(f"- {label} row {row_number}: period is not YYYY-MM-DDTHH: {value!r}")
        return None


def compact_counts(values: Counter[str]) -> str:
    if not values:
        return "none observed"
    return ", ".join(f"{value} ({count})" for value, count in sorted(values.items()))


def expected_periods(start_date: date, end_date: date) -> list[datetime]:
    start = datetime.combine(start_date, datetime.min.time())
    end = datetime.combine(end_date, datetime.min.time()) + timedelta(hours=23)
    periods: list[datetime] = []
    current = start
    while current <= end:
        periods.append(current)
        current += timedelta(hours=1)
    return periods


def load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} top-level JSON value is not an object.")

    if "pages" not in payload:
        return payload, rows_from_response(payload, f"{path} response")

    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"{path} has a pages field that is not a list.")

    rows: list[dict[str, Any]] = []
    for page_number, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            raise ValueError(f"{path} page {page_number} is not an object.")
        rows.extend(rows_from_response(page, f"{path} page {page_number}"))

    return payload, rows


def rows_from_response(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    response = payload.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"{label} does not contain a response object.")
    rows = response.get("data")
    if not isinstance(rows, list):
        raise ValueError(f"{label} does not contain response.data as a list.")

    dict_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {row_number} is not an object.")
        dict_rows.append(row)
    return dict_rows


def gap_report(periods: set[datetime]) -> list[str]:
    gaps: list[str] = []
    sorted_periods = sorted(periods)
    for previous, current in zip(sorted_periods, sorted_periods[1:]):
        step = current - previous
        if step != timedelta(hours=1):
            gaps.append(
                f"{previous.strftime('%Y-%m-%dT%H')} to "
                f"{current.strftime('%Y-%m-%dT%H')} ({step})"
            )
    return gaps


def preview_periods(periods: list[datetime] | set[datetime]) -> str:
    return ", ".join(period.strftime("%Y-%m-%dT%H") for period in sorted(periods)[:10])


def validate_history(
    demand_path: Path, renewable_path: Path, start_text: str, end_text: str
) -> int:
    try:
        start_date = parse_date(start_text, "start date")
        end_date = parse_date(end_text, "end date")
        if end_date < start_date:
            raise ValueError("end date must be same as or later than start date.")
        demand_payload, demand_rows = load_rows(demand_path)
        renewable_payload, renewable_rows = load_rows(renewable_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1

    expected = expected_periods(start_date, end_date)
    expected_set = set(expected)
    expected_hour_count = len(expected)
    expected_renewable_count = expected_hour_count * len(EXPECTED_FUELS)
    leap_year_expected = any(
        period.month == 2 and period.day == 29 for period in expected
    )

    demand_periods: list[datetime] = []
    demand_combo_counts: Counter[tuple[Any, Any, Any]] = Counter()
    demand_respondents: Counter[str] = Counter()
    demand_types: Counter[str] = Counter()
    demand_units: Counter[str] = Counter()
    demand_missing: list[str] = []
    demand_non_numeric: list[str] = []
    demand_unexpected_fields: Counter[str] = Counter()
    demand_missing_fields: Counter[str] = Counter()

    for index, row in enumerate(demand_rows, start=1):
        row_keys = set(row)
        for field in row_keys - EXPECTED_DEMAND_FIELDS:
            demand_unexpected_fields[field] += 1
        for field in EXPECTED_DEMAND_FIELDS - row_keys:
            demand_missing_fields[field] += 1
            demand_missing.append(f"row {index}: missing {field}")
        for field in EXPECTED_DEMAND_FIELDS & row_keys:
            if row.get(field) is None or row.get(field) == "":
                demand_missing.append(f"row {index}: blank {field}")

        period = parse_period(row.get("period"), index, "demand")
        if period is not None:
            demand_periods.append(period)
        demand_combo_counts[(row.get("period"), row.get("respondent"), row.get("type"))] += 1

        if isinstance(row.get("respondent"), str):
            demand_respondents[row["respondent"]] += 1
        if isinstance(row.get("type"), str):
            demand_types[row["type"]] += 1
        if isinstance(row.get("value-units"), str):
            demand_units[row["value-units"]] += 1
        try:
            float(row.get("value"))
        except (TypeError, ValueError):
            demand_non_numeric.append(f"row {index}: value {row.get('value')!r}")

    renewable_periods_by_fuel: dict[str, list[datetime]] = defaultdict(list)
    renewable_combo_counts: Counter[tuple[Any, Any, Any]] = Counter()
    renewable_respondents: Counter[str] = Counter()
    renewable_fuels: Counter[str] = Counter()
    renewable_units: Counter[str] = Counter()
    renewable_missing: list[str] = []
    renewable_non_numeric: list[str] = []
    renewable_unexpected_fields: Counter[str] = Counter()
    renewable_missing_fields: Counter[str] = Counter()

    for index, row in enumerate(renewable_rows, start=1):
        row_keys = set(row)
        for field in row_keys - EXPECTED_RENEWABLE_FIELDS:
            if field not in {"type-name", "fueltype-name"}:
                renewable_unexpected_fields[field] += 1
        for field in EXPECTED_RENEWABLE_FIELDS - row_keys:
            renewable_missing_fields[field] += 1
            renewable_missing.append(f"row {index}: missing {field}")
        for field in EXPECTED_RENEWABLE_FIELDS & row_keys:
            if row.get(field) is None or row.get(field) == "":
                renewable_missing.append(f"row {index}: blank {field}")

        period = parse_period(row.get("period"), index, "renewable")
        fuel = row.get("fueltype")
        if period is not None and isinstance(fuel, str):
            renewable_periods_by_fuel[fuel].append(period)
        renewable_combo_counts[(row.get("period"), row.get("respondent"), fuel)] += 1

        if isinstance(row.get("respondent"), str):
            renewable_respondents[row["respondent"]] += 1
        if isinstance(fuel, str):
            renewable_fuels[fuel] += 1
        if isinstance(row.get("value-units"), str):
            renewable_units[row["value-units"]] += 1
        try:
            float(row.get("value"))
        except (TypeError, ValueError):
            renewable_non_numeric.append(f"row {index}: value {row.get('value')!r}")

    demand_period_set = set(demand_periods)
    renewable_period_set = {
        period for periods in renewable_periods_by_fuel.values() for period in periods
    }
    duplicate_demand_combos = {
        combo: count for combo, count in demand_combo_counts.items() if count > 1
    }
    duplicate_renewable_combos = {
        combo: count for combo, count in renewable_combo_counts.items() if count > 1
    }

    demand_missing_expected = expected_set - demand_period_set
    demand_extra = demand_period_set - expected_set
    renewable_missing_by_fuel = {
        fuel: expected_set - set(renewable_periods_by_fuel.get(fuel, []))
        for fuel in EXPECTED_FUELS
    }
    renewable_extra_by_fuel = {
        fuel: set(periods) - expected_set
        for fuel, periods in renewable_periods_by_fuel.items()
    }
    gaps_demand = gap_report(demand_period_set)
    gaps_by_fuel = {
        fuel: gap_report(set(periods))
        for fuel, periods in sorted(renewable_periods_by_fuel.items())
    }
    timestamps_align = demand_period_set == renewable_period_set
    leap_year_observed = any(
        period.month == 2 and period.day == 29
        for period in demand_period_set | renewable_period_set
    )

    print("EIA CISO historical raw validation")
    print(f"Demand file: {demand_path}")
    print(f"Renewable file: {renewable_path}")
    print(f"Requested inclusive date range: {start_text} through {end_text}")
    print()

    print("Demand")
    print(f"- Row count: {len(demand_rows)}")
    print(f"- Expected hourly rows: {expected_hour_count}")
    if demand_periods:
        print(f"- Earliest timestamp: {min(demand_periods).strftime('%Y-%m-%dT%H')}")
        print(f"- Latest timestamp: {max(demand_periods).strftime('%Y-%m-%dT%H')}")
    print(f"- Respondents: {compact_counts(demand_respondents)}")
    print(f"- Demand type values: {compact_counts(demand_types)}")
    print(f"- Units: {compact_counts(demand_units)}")
    print(f"- Missing or blank values: {'none' if not demand_missing else len(demand_missing)}")
    print(f"- Non-numeric values: {'none' if not demand_non_numeric else len(demand_non_numeric)}")
    print(
        "- Duplicate timestamp/respondent/type combinations: "
        f"{'none' if not duplicate_demand_combos else len(duplicate_demand_combos)}"
    )
    print(f"- Hourly gaps: {'none' if not gaps_demand else '; '.join(gaps_demand[:10])}")
    print(
        "- Covers requested range: "
        f"{'yes' if not demand_missing_expected and not demand_extra else 'no'}"
    )
    if demand_missing_expected:
        print(f"  - Missing demand hours: {preview_periods(demand_missing_expected)}")
    if demand_extra:
        print(f"  - Demand hours outside request: {preview_periods(demand_extra)}")
    print(f"- Unexpected fields: {'none' if not demand_unexpected_fields else compact_counts(demand_unexpected_fields)}")
    print(f"- Missing expected fields: {'none' if not demand_missing_fields else compact_counts(demand_missing_fields)}")
    print()

    print("Solar and wind")
    print(f"- Row count: {len(renewable_rows)}")
    print(f"- Expected rows: {expected_renewable_count}")
    print(f"- Respondents: {compact_counts(renewable_respondents)}")
    print(f"- Fuel categories: {compact_counts(renewable_fuels)}")
    print(f"- Units: {compact_counts(renewable_units)}")
    for fuel in sorted(EXPECTED_FUELS):
        periods = renewable_periods_by_fuel.get(fuel, [])
        print(f"- {fuel} row count: {len(periods)}")
        if periods:
            print(f"  - {fuel} earliest: {min(periods).strftime('%Y-%m-%dT%H')}")
            print(f"  - {fuel} latest: {max(periods).strftime('%Y-%m-%dT%H')}")
        print(
            f"  - {fuel} hourly gaps: "
            f"{'none' if not gaps_by_fuel.get(fuel) else '; '.join(gaps_by_fuel[fuel][:10])}"
        )
        print(
            f"  - {fuel} covers requested range: "
            f"{'yes' if not renewable_missing_by_fuel[fuel] and not renewable_extra_by_fuel.get(fuel) else 'no'}"
        )
    print(f"- Missing or blank values: {'none' if not renewable_missing else len(renewable_missing)}")
    print(f"- Non-numeric values: {'none' if not renewable_non_numeric else len(renewable_non_numeric)}")
    print(
        "- Duplicate timestamp/respondent/fuel combinations: "
        f"{'none' if not duplicate_renewable_combos else len(duplicate_renewable_combos)}"
    )
    print(f"- Unexpected fields: {'none' if not renewable_unexpected_fields else compact_counts(renewable_unexpected_fields)}")
    print(f"- Missing expected fields: {'none' if not renewable_missing_fields else compact_counts(renewable_missing_fields)}")
    print()

    print("Cross-checks")
    print(f"- Demand and renewable timestamps align: {'yes' if timestamps_align else 'no'}")
    print(
        "- Leap-year handling: "
        f"{'Feb 29 expected and observed' if leap_year_expected and leap_year_observed else 'no Feb 29 expected'}"
    )
    if leap_year_expected and not leap_year_observed:
        print("  - Feb 29 is expected for this range but was not observed.")
    print(f"- Demand metadata top-level keys: {', '.join(sorted(demand_payload))}")
    print(f"- Renewable metadata top-level keys: {', '.join(sorted(renewable_payload))}")

    passed = all(
        [
            len(demand_rows) == expected_hour_count,
            len(renewable_rows) == expected_renewable_count,
            set(demand_respondents) == {EXPECTED_RESPONDENT},
            set(renewable_respondents) == {EXPECTED_RESPONDENT},
            set(demand_types) == {EXPECTED_DEMAND_TYPE},
            set(renewable_fuels) == EXPECTED_FUELS,
            not demand_missing,
            not renewable_missing,
            not demand_non_numeric,
            not renewable_non_numeric,
            not duplicate_demand_combos,
            not duplicate_renewable_combos,
            not gaps_demand,
            not any(gaps_by_fuel.values()),
            not demand_missing_expected,
            not demand_extra,
            not any(renewable_missing_by_fuel.values()),
            not any(renewable_extra_by_fuel.values()),
            timestamps_align,
            not demand_missing_fields,
            not renewable_missing_fields,
            not demand_unexpected_fields,
            not renewable_unexpected_fields,
            (not leap_year_expected or leap_year_observed),
        ]
    )

    print()
    print(f"Overall result: {'PASS' if passed else 'CHECK ISSUES ABOVE'}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate historical raw EIA CISO demand and solar/wind JSON."
    )
    parser.add_argument("start_date", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("end_date", help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument(
        "--demand-path",
        default=DEFAULT_DEMAND_PATH,
        type=Path,
        help=f"Demand JSON path. Default: {DEFAULT_DEMAND_PATH}",
    )
    parser.add_argument(
        "--renewable-path",
        default=DEFAULT_RENEWABLE_PATH,
        type=Path,
        help=f"Renewable JSON path. Default: {DEFAULT_RENEWABLE_PATH}",
    )
    args = parser.parse_args()
    return validate_history(args.demand_path, args.renewable_path, args.start_date, args.end_date)


if __name__ == "__main__":
    raise SystemExit(main())
