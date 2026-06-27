"""Validate historical EIA CISO demand and documented renewable coverage."""

from __future__ import annotations

import argparse
import json
import math
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
DOCUMENTED_NULL_TIMESTAMPS = frozenset(
    {
        datetime(2022, 1, 5, 10),
        datetime(2022, 5, 17, 18),
        datetime(2022, 6, 13, 18),
        datetime(2023, 10, 31, 21),
        datetime(2023, 11, 14, 20),
    }
)


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


def print_issue_examples(label: str, issues: list[str], limit: int = 10) -> None:
    """Print concise row-level examples without dumping raw records or metadata."""
    if not issues:
        return
    print(f"- {label} issue details:")
    for issue in issues[:limit]:
        print(f"  - {issue}")
    if len(issues) > limit:
        print(f"  - ... {len(issues) - limit} more malformed rows")


def is_finite_number(value: Any) -> bool:
    """Return true only for values that can be parsed as finite numbers."""
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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
    demand_integrity_issues: list[str] = []
    demand_non_numeric_issues: list[str] = []
    demand_null_periods: set[datetime] = set()

    for index, row in enumerate(demand_file.rows, start=1):
        problems: list[str] = []
        missing = EXPECTED_DEMAND_FIELDS - set(row)
        unexpected = set(row) - EXPECTED_DEMAND_FIELDS
        if missing:
            problems.append(f"missing required fields: {', '.join(sorted(missing))}")
        if unexpected:
            problems.append(f"unexpected fields: {', '.join(sorted(unexpected))}")

        blank_identity_fields = sorted(
            field
            for field in (EXPECTED_DEMAND_FIELDS - {"value"}) & set(row)
            if row[field] is None or row[field] == ""
        )
        if blank_identity_fields:
            problems.append(
                "null or blank required fields: " + ", ".join(blank_identity_fields)
            )
        period: datetime | None = None
        try:
            period = parse_period(row.get("period"), index, "demand")
            demand_periods.append(period)
        except ValueError as exc:
            problems.append(str(exc))

        if "value" in row:
            if row["value"] is None:
                if period is not None:
                    demand_null_periods.add(period)
            elif not is_finite_number(row["value"]):
                demand_non_numeric_issues.append(
                    f"row {index} period={row.get('period')!r}: value "
                    f"{row['value']!r} is non-numeric (expected a finite number)"
                )
        if problems:
            demand_integrity_issues.append(
                f"row {index} period={row.get('period')!r}: " + "; ".join(problems)
            )
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
    renewable_integrity_issues: list[str] = []
    renewable_non_numeric_issues: list[str] = []
    renewable_null_combos: set[tuple[datetime, str]] = set()

    allowed_renewable_fields = EXPECTED_RENEWABLE_FIELDS | OPTIONAL_RENEWABLE_FIELDS
    for index, row in enumerate(renewable_file.rows, start=1):
        problems = []
        missing = EXPECTED_RENEWABLE_FIELDS - set(row)
        unexpected = set(row) - allowed_renewable_fields
        if missing:
            problems.append(f"missing required fields: {', '.join(sorted(missing))}")
        if unexpected:
            problems.append(f"unexpected fields: {', '.join(sorted(unexpected))}")

        blank_identity_fields = sorted(
            field
            for field in (EXPECTED_RENEWABLE_FIELDS - {"value"}) & set(row)
            if row[field] is None or row[field] == ""
        )
        if blank_identity_fields:
            problems.append(
                "null or blank required fields: " + ", ".join(blank_identity_fields)
            )
        fuel = row.get("fueltype")
        period = None
        try:
            period = parse_period(row.get("period"), index, "renewable")
            if isinstance(fuel, str):
                renewable_periods_by_fuel[fuel].append(period)
        except ValueError as exc:
            problems.append(str(exc))

        if "value" in row:
            if row["value"] is None:
                if period is not None and isinstance(fuel, str):
                    renewable_null_combos.add((period, fuel))
            elif not is_finite_number(row["value"]):
                renewable_non_numeric_issues.append(
                    f"row {index} period={row.get('period')!r} "
                    f"fueltype={fuel!r}: value {row['value']!r} is non-numeric "
                    "(expected a finite number)"
                )
        if problems:
            renewable_integrity_issues.append(
                f"row {index} period={row.get('period')!r} "
                f"fueltype={fuel!r}: " + "; ".join(problems)
            )
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

    documented_nulls = DOCUMENTED_NULL_TIMESTAMPS & expected_set
    expected_renewable_null_combos = {
        (period, fuel) for period in documented_nulls for fuel in EXPECTED_FUELS
    }
    unexpected_demand_nulls = demand_null_periods - documented_nulls
    documented_demand_nulls_missing = documented_nulls - demand_null_periods
    unexpected_renewable_nulls = (
        renewable_null_combos - expected_renewable_null_combos
    )
    documented_renewable_nulls_missing = (
        expected_renewable_null_combos - renewable_null_combos
    )

    download_integrity = all(
        [
            not demand_integrity_issues,
            not renewable_integrity_issues,
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
    demand_measurement_pass = all(
        [
            demand_null_periods == documented_nulls,
            not demand_non_numeric_issues,
        ]
    )
    renewable_measurement_pass = all(
        [
            renewable_null_combos == expected_renewable_null_combos,
            not renewable_non_numeric_issues,
        ]
    )
    leap_expected = any(period.month == 2 and period.day == 29 for period in expected_set)
    leap_pass = not leap_expected or all(
        any(period.month == 2 and period.day == 29 for period in periods)
        for periods in [demand_periods, *renewable_periods_by_fuel.values()]
    )
    unexpected_quality_problems = any(
        [
            any(unexpected_missing.values()),
            any(undocumented_present.values()),
            unexpected_demand_nulls,
            documented_demand_nulls_missing,
            unexpected_renewable_nulls,
            documented_renewable_nulls_missing,
            demand_non_numeric_issues,
            renewable_non_numeric_issues,
        ]
    )

    print("EIA CISO historical raw validation")
    print(f"Requested inclusive range: {start_text} through {end_text}")
    print()
    print(f"Download integrity: {'PASS' if download_integrity else 'FAIL'}")
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
    print(f"- Demand structural issues: {len(demand_integrity_issues)}")
    print(f"- Renewable structural issues: {len(renewable_integrity_issues)}")
    print_issue_examples("Demand structural", demand_integrity_issues)
    print_issue_examples("Renewable structural", renewable_integrity_issues)
    print(f"- Demand respondents: {', '.join(sorted(demand_respondents)) or 'none'}")
    print(f"- Demand types: {', '.join(sorted(demand_types)) or 'none'}")
    print(f"- Renewable respondents: {', '.join(sorted(renewable_respondents)) or 'none'}")
    print(f"- Renewable fuels: {', '.join(sorted(renewable_fuels)) or 'none'}")
    print(f"- Demand units: {', '.join(sorted(demand_units)) or 'none'}")
    print(f"- Renewable units: {', '.join(sorted(renewable_units)) or 'none'}")
    print()

    print(f"Demand timestamp coverage: {'PASS' if demand_coverage_pass else 'FAIL'}")
    print(f"- Observed rows: {len(demand_file.rows)}")
    print(f"- Theoretical rows: {expected_hours}")
    print(f"- Missing demand timestamps: {len(demand_missing)}")
    print(f"- Demand timestamps outside range: {len(demand_extra)}")
    if demand_periods:
        print(f"- Earliest demand timestamp: {min(demand_periods):%Y-%m-%dT%H}")
        print(f"- Latest demand timestamp: {max(demand_periods):%Y-%m-%dT%H}")
    print()

    demand_measurement_label = (
        "WARNING" if demand_measurement_pass and documented_nulls else
        ("PASS" if demand_measurement_pass else "FAIL")
    )
    print(f"Demand measurement completeness: {demand_measurement_label}")
    print(f"- Documented null values observed: {len(demand_null_periods)}")
    print(f"- Unexpected null values: {len(unexpected_demand_nulls)}")
    print(f"- Documented null values unexpectedly non-null: {len(documented_demand_nulls_missing)}")
    print(f"- Non-null, non-numeric values: {len(demand_non_numeric_issues)}")
    print_issue_examples("Demand measurement", demand_non_numeric_issues)
    print()

    has_documented_gap = bool(documented_missing)
    renewable_coverage_label = "WARNING" if renewable_coverage_pass and has_documented_gap else (
        "PASS" if renewable_coverage_pass else "FAIL"
    )
    missing_renewable_timestamps = set().union(*renewable_missing_by_fuel.values())
    print(f"Renewable row coverage: {renewable_coverage_label}")
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
    print()

    renewable_measurement_label = (
        "WARNING" if renewable_measurement_pass and documented_nulls else
        ("PASS" if renewable_measurement_pass else "FAIL")
    )
    print(f"Renewable measurement completeness: {renewable_measurement_label}")
    print(f"- Documented null values observed: {len(renewable_null_combos)}")
    print(f"- Affected documented timestamps: {len(documented_nulls)}")
    print(f"- Unexpected null values: {len(unexpected_renewable_nulls)}")
    print(
        "- Documented null values unexpectedly non-null: "
        f"{len(documented_renewable_nulls_missing)}"
    )
    print(f"- Non-null, non-numeric values: {len(renewable_non_numeric_issues)}")
    print_issue_examples("Renewable measurement", renewable_non_numeric_issues)
    print()

    print(
        "Unexpected gaps or nulls: "
        + ("present" if unexpected_quality_problems else "none")
    )
    print(f"- Leap-year handling: {'PASS' if leap_pass else 'FAIL'}")

    passed = all(
        [
            download_integrity,
            demand_coverage_pass,
            renewable_coverage_pass,
            demand_measurement_pass,
            renewable_measurement_pass,
            not unexpected_quality_problems,
            leap_pass,
        ]
    )
    has_documented_warning = bool(documented_missing or documented_nulls)
    print()
    if passed and has_documented_warning:
        print("Overall result: PASS WITH DOCUMENTED SOURCE DATA WARNINGS")
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
