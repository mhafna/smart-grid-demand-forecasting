"""Validate historical EIA CISO demand and documented renewable coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
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
EXPECTED_UNITS = {"megawatthours"}

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
OPTIONAL_RENEWABLE_FIELDS = {"type-name", "fueltype-name"}

# Confirmed with the focused EIA diagnostic. This is the only accepted gap.
DOCUMENTED_GAP_START = datetime(2024, 11, 2, 8)
DOCUMENTED_GAP_END = datetime(2024, 11, 3, 7)


@dataclass
class PaginatedFile:
    """Rows plus pagination facts verified from a saved download wrapper."""

    payload: dict[str, Any]
    rows: list[dict[str, Any]]
    source_total: int


def parse_date(value: str, label: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY-MM-DD format.") from exc


def parse_period(value: Any, row_number: int, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} row {row_number} period is not a string: {value!r}")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H")
    except ValueError as exc:
        raise ValueError(
            f"{label} row {row_number} period is not YYYY-MM-DDTHH: {value!r}"
        ) from exc


def integer_metadata(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an integer: {value!r}") from exc
    if result < 0:
        raise ValueError(f"{label} cannot be negative: {result}")
    return result


def expected_periods(start_date: date, end_date: date) -> list[datetime]:
    start = datetime.combine(start_date, datetime.min.time())
    end = datetime.combine(end_date, datetime.min.time()) + timedelta(hours=23)
    periods: list[datetime] = []
    current = start
    while current <= end:
        periods.append(current)
        current += timedelta(hours=1)
    return periods


def documented_gap_periods() -> set[datetime]:
    periods: set[datetime] = set()
    current = DOCUMENTED_GAP_START
    while current <= DOCUMENTED_GAP_END:
        periods.add(current)
        current += timedelta(hours=1)
    return periods


def load_paginated_file(
    path: Path,
    label: str,
    theoretical_rows: int,
    expected_start: str,
    expected_end: str,
) -> PaginatedFile:
    """Load a wrapper and prove every saved page agrees with response.total."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} top-level JSON value is not an object.")

    pages = payload.get("pages")
    download = payload.get("download")
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"{path} does not contain a non-empty pages list.")
    if not isinstance(download, dict):
        raise ValueError(f"{path} does not contain download metadata.")

    rows: list[dict[str, Any]] = []
    source_total: int | None = None
    for page_number, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            raise ValueError(f"{path} page {page_number} is not an object.")
        response = page.get("response")
        if not isinstance(response, dict):
            raise ValueError(f"{path} page {page_number} has no response object.")
        page_rows = response.get("data")
        if not isinstance(page_rows, list):
            raise ValueError(f"{path} page {page_number} response.data is not a list.")
        page_total = integer_metadata(
            response.get("total"), f"{path} page {page_number} response.total"
        )
        if source_total is None:
            source_total = page_total
        elif page_total != source_total:
            raise ValueError(
                f"{path} response.total changed from {source_total} to {page_total} "
                f"on page {page_number}."
            )
        if not page_rows and len(rows) < source_total:
            raise ValueError(
                f"{path} page {page_number} is empty before response.total was reached."
            )
        for row_number, row in enumerate(page_rows, start=1):
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path} page {page_number} row {row_number} is not an object."
                )
            rows.append(row)
        if len(rows) > source_total:
            raise ValueError(f"{path} contains more rows than response.total={source_total}.")

        request = page.get("request")
        if isinstance(request, dict):
            params = request.get("params")
            if (
                isinstance(params, dict)
                and "api_key" in params
                and params["api_key"] != "[REDACTED]"
            ):
                raise ValueError(f"{path} contains an unredacted API key in page metadata.")

    if source_total is None or len(rows) != source_total:
        raise ValueError(
            f"{path} contains {len(rows)} rows but response.total is {source_total}."
        )

    expected_metadata = {
        "theoretical_rows": theoretical_rows,
        "source_total_rows": source_total,
        "downloaded_rows": source_total,
        "page_count": len(pages),
    }
    for field, expected_value in expected_metadata.items():
        actual = integer_metadata(download.get(field), f"{path} download.{field}")
        if actual != expected_value:
            raise ValueError(
                f"{path} download.{field} is {actual}, expected {expected_value}."
            )

    if download.get("start") != expected_start or download.get("end") != expected_end:
        raise ValueError(
            f"{path} metadata range is {download.get('start')!r} through "
            f"{download.get('end')!r}, expected {expected_start!r} through "
            f"{expected_end!r}."
        )

    shortfall = integer_metadata(
        download.get("source_coverage_shortfall_rows"),
        f"{path} download.source_coverage_shortfall_rows",
    )
    if shortfall != theoretical_rows - source_total:
        raise ValueError(
            f"{path} source coverage shortfall metadata is {shortfall}, expected "
            f"{theoretical_rows - source_total}."
        )

    return PaginatedFile(payload=payload, rows=rows, source_total=source_total)


def validate_history(
    demand_path: Path, renewable_path: Path, start_text: str, end_text: str
) -> int:
    try:
        start_date = parse_date(start_text, "start date")
        end_date = parse_date(end_text, "end date")
        if end_date < start_date:
            raise ValueError("end date must be same as or later than start date.")

        expected = expected_periods(start_date, end_date)
        expected_set = set(expected)
        expected_hours = len(expected)
        expected_renewable_rows = expected_hours * len(EXPECTED_FUELS)
        expected_start = f"{start_date:%Y-%m-%d}T00"
        expected_end = f"{end_date:%Y-%m-%d}T23"

        demand_file = load_paginated_file(
            demand_path, "demand", expected_hours, expected_start, expected_end
        )
        renewable_file = load_paginated_file(
            renewable_path,
            "renewable",
            expected_renewable_rows,
            expected_start,
            expected_end,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("Download integrity: FAIL")
        print(f"- {exc}")
        print("Overall result: FAIL")
        return 1

    demand_periods: list[datetime] = []
    demand_combos: Counter[tuple[Any, Any, Any]] = Counter()
    demand_respondents: Counter[str] = Counter()
    demand_types: Counter[str] = Counter()
    demand_units: Counter[str] = Counter()
    demand_content_issues: list[str] = []

    for index, row in enumerate(demand_file.rows, start=1):
        missing = EXPECTED_DEMAND_FIELDS - set(row)
        unexpected = set(row) - EXPECTED_DEMAND_FIELDS
        if missing or unexpected:
            demand_content_issues.append(
                f"row {index} fields missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        if any(row.get(field) is None or row.get(field) == "" for field in EXPECTED_DEMAND_FIELDS):
            demand_content_issues.append(f"row {index} has a missing or blank value")
        try:
            float(row.get("value"))
        except (TypeError, ValueError):
            demand_content_issues.append(f"row {index} has non-numeric demand")
        try:
            period = parse_period(row.get("period"), index, "demand")
            demand_periods.append(period)
        except ValueError as exc:
            demand_content_issues.append(str(exc))
        demand_combos[(row.get("period"), row.get("respondent"), row.get("type"))] += 1
        if isinstance(row.get("respondent"), str):
            demand_respondents[row["respondent"]] += 1
        if isinstance(row.get("type"), str):
            demand_types[row["type"]] += 1
        if isinstance(row.get("value-units"), str):
            demand_units[row["value-units"]] += 1

    renewable_periods_by_fuel: dict[str, list[datetime]] = defaultdict(list)
    renewable_combos: Counter[tuple[Any, Any, Any]] = Counter()
    renewable_respondents: Counter[str] = Counter()
    renewable_fuels: Counter[str] = Counter()
    renewable_units: Counter[str] = Counter()
    renewable_content_issues: list[str] = []

    allowed_renewable_fields = EXPECTED_RENEWABLE_FIELDS | OPTIONAL_RENEWABLE_FIELDS
    for index, row in enumerate(renewable_file.rows, start=1):
        missing = EXPECTED_RENEWABLE_FIELDS - set(row)
        unexpected = set(row) - allowed_renewable_fields
        if missing or unexpected:
            renewable_content_issues.append(
                f"row {index} fields missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        if any(row.get(field) is None or row.get(field) == "" for field in EXPECTED_RENEWABLE_FIELDS):
            renewable_content_issues.append(f"row {index} has a missing or blank value")
        try:
            float(row.get("value"))
        except (TypeError, ValueError):
            renewable_content_issues.append(f"row {index} has non-numeric generation")
        fuel = row.get("fueltype")
        try:
            period = parse_period(row.get("period"), index, "renewable")
            if isinstance(fuel, str):
                renewable_periods_by_fuel[fuel].append(period)
        except ValueError as exc:
            renewable_content_issues.append(str(exc))
        renewable_combos[(row.get("period"), row.get("respondent"), fuel)] += 1
        if isinstance(row.get("respondent"), str):
            renewable_respondents[row["respondent"]] += 1
        if isinstance(fuel, str):
            renewable_fuels[fuel] += 1
        if isinstance(row.get("value-units"), str):
            renewable_units[row["value-units"]] += 1

    duplicate_demand = {combo: count for combo, count in demand_combos.items() if count > 1}
    duplicate_renewable = {
        combo: count for combo, count in renewable_combos.items() if count > 1
    }
    demand_period_set = set(demand_periods)
    demand_missing = expected_set - demand_period_set
    demand_extra = demand_period_set - expected_set

    documented_missing = documented_gap_periods() & expected_set
    renewable_missing_by_fuel = {
        fuel: expected_set - set(renewable_periods_by_fuel.get(fuel, []))
        for fuel in EXPECTED_FUELS
    }
    renewable_extra_by_fuel = {
        fuel: set(periods) - expected_set
        for fuel, periods in renewable_periods_by_fuel.items()
    }
    renewable_period_set = set().union(
        *(set(periods) for periods in renewable_periods_by_fuel.values())
    )
    alignment_missing = demand_period_set - renewable_period_set
    alignment_extra = renewable_period_set - demand_period_set
    timestamp_alignment_pass = (
        alignment_missing == documented_missing and not alignment_extra
    )
    unexpected_missing = {
        fuel: periods - documented_missing
        for fuel, periods in renewable_missing_by_fuel.items()
    }
    undocumented_present = {
        fuel: documented_missing - periods
        for fuel, periods in renewable_missing_by_fuel.items()
    }

    content_integrity = all(
        [
            not demand_content_issues,
            not renewable_content_issues,
            not duplicate_demand,
            not duplicate_renewable,
            set(demand_respondents) == {EXPECTED_RESPONDENT},
            set(renewable_respondents) == {EXPECTED_RESPONDENT},
            set(demand_types) == {EXPECTED_DEMAND_TYPE},
            set(renewable_fuels) == EXPECTED_FUELS,
            set(demand_units) == EXPECTED_UNITS,
            set(renewable_units) == EXPECTED_UNITS,
        ]
    )
    demand_coverage_pass = all(
        [
            len(demand_file.rows) == expected_hours,
            not demand_missing,
            not demand_extra,
            len(demand_period_set) == expected_hours,
        ]
    )
    renewable_gap_matches = all(
        not unexpected_missing[fuel] and not undocumented_present[fuel]
        for fuel in EXPECTED_FUELS
    )
    renewable_coverage_pass = all(
        [
            renewable_gap_matches,
            not any(renewable_extra_by_fuel.values()),
            renewable_file.source_total
            == expected_renewable_rows - len(documented_missing) * len(EXPECTED_FUELS),
            timestamp_alignment_pass,
        ]
    )
    leap_expected = any(period.month == 2 and period.day == 29 for period in expected_set)
    leap_pass = not leap_expected or all(
        any(period.month == 2 and period.day == 29 for period in periods)
        for periods in [demand_periods, *renewable_periods_by_fuel.values()]
    )

    print("EIA CISO historical raw validation")
    print(f"Requested inclusive range: {start_text} through {end_text}")
    print()
    print(f"Download integrity: {'PASS' if content_integrity else 'FAIL'}")
    print(
        f"- Demand pagination: {len(demand_file.rows)}/{demand_file.source_total} "
        "rows downloaded"
    )
    print(
        f"- Renewable pagination: {len(renewable_file.rows)}/{renewable_file.source_total} "
        "rows downloaded"
    )
    print(f"- Demand duplicate combinations: {len(duplicate_demand)}")
    print(f"- Renewable duplicate combinations: {len(duplicate_renewable)}")
    print(f"- Demand content issues: {len(demand_content_issues)}")
    print(f"- Renewable content issues: {len(renewable_content_issues)}")
    print(f"- Demand units: {', '.join(sorted(demand_units)) or 'none'}")
    print(f"- Renewable units: {', '.join(sorted(renewable_units)) or 'none'}")
    print()

    print(f"Demand coverage: {'PASS' if demand_coverage_pass else 'FAIL'}")
    print(f"- Observed rows: {len(demand_file.rows)}")
    print(f"- Theoretical rows: {expected_hours}")
    print(f"- Missing demand timestamps: {len(demand_missing)}")
    print(f"- Demand timestamps outside range: {len(demand_extra)}")
    if demand_periods:
        print(f"- Earliest demand timestamp: {min(demand_periods):%Y-%m-%dT%H}")
        print(f"- Latest demand timestamp: {max(demand_periods):%Y-%m-%dT%H}")
    print()

    has_documented_warning = bool(documented_missing)
    renewable_label = "WARNING" if renewable_coverage_pass and has_documented_warning else (
        "PASS" if renewable_coverage_pass else "FAIL"
    )
    missing_renewable_timestamps = set().union(*renewable_missing_by_fuel.values())
    print(f"Renewable source coverage: {renewable_label}")
    print(f"- Observed rows: {len(renewable_file.rows)}")
    print(f"- Theoretical rows: {expected_renewable_rows}")
    for fuel in sorted(EXPECTED_FUELS):
        print(
            f"- {fuel}: {len(renewable_periods_by_fuel.get(fuel, []))}/{expected_hours} rows"
        )
    print(f"- Missing renewable timestamps: {len(missing_renewable_timestamps)}")
    if missing_renewable_timestamps:
        print(
            f"- Missing block: {min(missing_renewable_timestamps):%Y-%m-%dT%H} "
            f"through {max(missing_renewable_timestamps):%Y-%m-%dT%H}"
        )
    print(
        "- Unexpected gaps: "
        + ("none" if not any(unexpected_missing.values()) else "present")
    )
    print(
        "- Documented rows unexpectedly present: "
        + ("none" if not any(undocumented_present.values()) else "present")
    )
    print(
        "- Demand/renewable timestamp alignment: "
        + ("PASS (documented gap only)" if timestamp_alignment_pass else "FAIL")
    )
    print(f"- Leap-year handling: {'PASS' if leap_pass else 'FAIL'}")

    passed = all(
        [content_integrity, demand_coverage_pass, renewable_coverage_pass, leap_pass]
    )
    print()
    if passed and has_documented_warning:
        print("Overall result: PASS WITH DOCUMENTED SOURCE COVERAGE WARNING")
    else:
        print(f"Overall result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate historical raw EIA CISO demand and solar/wind JSON."
    )
    parser.add_argument("start_date", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("end_date", help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument(
        "--demand-path", default=DEFAULT_DEMAND_PATH, type=Path,
        help=f"Demand JSON path. Default: {DEFAULT_DEMAND_PATH}",
    )
    parser.add_argument(
        "--renewable-path", default=DEFAULT_RENEWABLE_PATH, type=Path,
        help=f"Renewable JSON path. Default: {DEFAULT_RENEWABLE_PATH}",
    )
    args = parser.parse_args()
    return validate_history(args.demand_path, args.renewable_path, args.start_date, args.end_date)


if __name__ == "__main__":
    raise SystemExit(main())
