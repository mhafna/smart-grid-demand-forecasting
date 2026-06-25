"""Diagnose EIA CISO hourly SUN/WND coverage without writing raw data files.

This script is intentionally separate from the main historical downloader. It
uses EIA's response.total as the pagination truth, then compares the returned
rows with theoretical complete hourly coverage so we can understand missingness
without filling, deleting, or fabricating observations.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from fetch_eia_history import (
    FUEL_TYPES,
    PAGE_SIZE,
    RENEWABLE_API_URL,
    RESPONDENT,
    fetch_page_with_retries,
    rows_from_payload,
    total_from_payload,
)


DEFAULT_START = date(2022, 1, 1)
DEFAULT_END = date(2024, 12, 31)


@dataclass
class TotalCheck:
    """Observed EIA total compared with theoretical hourly coverage."""

    label: str
    fuel_type: str
    start: date
    end: date
    eia_total: int
    theoretical_total: int

    @property
    def difference(self) -> int:
        return self.eia_total - self.theoretical_total

    @property
    def deficient(self) -> bool:
        return self.eia_total != self.theoretical_total


@dataclass
class FetchResult:
    """Rows and paging diagnostics for one focused EIA request."""

    label: str
    fuel_type: str
    start_hour: str
    end_hour: str
    response_total: int
    downloaded_rows: list[dict[str, Any]]
    page_row_counts: list[int]
    total_changed: bool
    empty_page_before_total: bool

    @property
    def downloaded_count(self) -> int:
        return len(self.downloaded_rows)

    @property
    def pagination_ok(self) -> bool:
        return (
            self.downloaded_count == self.response_total
            and not self.total_changed
            and not self.empty_page_before_total
        )


def parse_date(value: str, label: str) -> date:
    """Parse a date in YYYY-MM-DD format."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY-MM-DD format.") from exc


def inclusive_hour_strings(start_date: date, end_date: date) -> tuple[str, str]:
    """Convert inclusive dates to EIA hourly start/end strings."""
    return f"{start_date:%Y-%m-%d}T00", f"{end_date:%Y-%m-%d}T23"


def expected_hour_count(start_date: date, end_date: date) -> int:
    """Return the number of hourly timestamps in an inclusive date range."""
    return ((end_date - start_date).days + 1) * 24


def expected_periods(start_date: date, end_date: date) -> list[str]:
    """Build expected EIA period strings for an inclusive date range."""
    current = datetime.combine(start_date, datetime.min.time())
    end = datetime.combine(end_date, datetime.min.time()) + timedelta(hours=23)
    periods: list[str] = []
    while current <= end:
        periods.append(current.strftime("%Y-%m-%dT%H"))
        current += timedelta(hours=1)
    return periods


def month_windows(year: int) -> list[tuple[date, date]]:
    """Return inclusive month windows for one year."""
    windows: list[tuple[date, date]] = []
    for month in range(1, 13):
        start = date(year, month, 1)
        if month == 12:
            end = date(year, 12, 31)
        else:
            end = date(year, month + 1, 1) - timedelta(days=1)
        windows.append((start, end))
    return windows


def build_params(
    api_key: str,
    fuel_type: str,
    start_hour: str,
    end_hour: str,
    length: int,
    offset: int,
) -> list[tuple[str, str]]:
    """Build EIA parameters for one CISO renewable coverage request."""
    return [
        ("api_key", api_key),
        ("frequency", "hourly"),
        ("data[]", "value"),
        ("facets[respondent][]", RESPONDENT),
        ("facets[fueltype][]", fuel_type),
        ("start", start_hour),
        ("end", end_hour),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("sort[1][column]", "fueltype"),
        ("sort[1][direction]", "asc"),
        ("length", str(length)),
        ("offset", str(offset)),
    ]


def fetch_total(
    api_key: str, fuel_type: str, start_date: date, end_date: date, label: str
) -> TotalCheck:
    """Request a small page and read EIA response.total."""
    start_hour, end_hour = inclusive_hour_strings(start_date, end_date)
    url = (
        f"{RENEWABLE_API_URL}?"
        f"{urlencode(build_params(api_key, fuel_type, start_hour, end_hour, 1, 0))}"
    )
    payload = fetch_page_with_retries(url)
    total = total_from_payload(payload, f"{label} {fuel_type}")

    return TotalCheck(
        label=label,
        fuel_type=fuel_type,
        start=start_date,
        end=end_date,
        eia_total=total,
        theoretical_total=expected_hour_count(start_date, end_date),
    )


def fetch_rows_for_period(
    api_key: str, fuel_type: str, start_date: date, end_date: date, label: str
) -> FetchResult:
    """Fetch all rows for a focused period, using EIA total for pagination."""
    start_hour, end_hour = inclusive_hour_strings(start_date, end_date)
    offset = 0
    response_total: int | None = None
    total_changed = False
    empty_page_before_total = False
    rows: list[dict[str, Any]] = []
    page_row_counts: list[int] = []

    while response_total is None or offset < response_total:
        url = (
            f"{RENEWABLE_API_URL}?"
            f"{urlencode(build_params(api_key, fuel_type, start_hour, end_hour, PAGE_SIZE, offset))}"
        )
        payload = fetch_page_with_retries(url)
        page_rows = rows_from_payload(payload, f"{label} {fuel_type} offset {offset}")
        page_total = total_from_payload(payload, f"{label} {fuel_type} offset {offset}")

        if response_total is None:
            response_total = page_total
        elif page_total != response_total:
            total_changed = True

        if not page_rows and offset < page_total:
            empty_page_before_total = True
            break

        rows.extend(page_rows)
        page_row_counts.append(len(page_rows))
        offset += len(page_rows)

    if response_total is None:
        response_total = 0

    return FetchResult(
        label=label,
        fuel_type=fuel_type,
        start_hour=start_hour,
        end_hour=end_hour,
        response_total=response_total,
        downloaded_rows=rows,
        page_row_counts=page_row_counts,
        total_changed=total_changed,
        empty_page_before_total=empty_page_before_total,
    )


def row_combos(rows: list[dict[str, Any]], fuel_type: str) -> Counter[tuple[str, str]]:
    """Count timestamp/fuel combinations in returned rows."""
    combos: Counter[tuple[str, str]] = Counter()
    for row in rows:
        period = row.get("period")
        fuel = row.get("fueltype")
        if isinstance(period, str) and fuel == fuel_type:
            combos[(period, fuel_type)] += 1
    return combos


def print_total_table(title: str, checks: list[TotalCheck]) -> None:
    """Print observed totals without exposing the API key."""
    print(title)
    for check in checks:
        status = "OK" if not check.deficient else "CHECK"
        print(
            f"- {check.label} {check.fuel_type}: EIA total={check.eia_total}, "
            f"theoretical={check.theoretical_total}, "
            f"difference={check.difference} [{status}]"
        )
    print()


def preview(values: list[str], limit: int = 20) -> str:
    """Format a short preview of a list."""
    if not values:
        return "none"
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f"; ... {len(values) - limit} more"
    return ", ".join(shown) + suffix


def diagnose(api_key: str, start_date: date, end_date: date) -> int:
    """Run the renewable coverage diagnostic and print a concise report."""
    if end_date < start_date:
        raise ValueError("End date must be the same as or later than start date.")

    years = range(start_date.year, end_date.year + 1)
    year_checks: list[TotalCheck] = []

    print("EIA CISO SUN/WND renewable coverage diagnostic")
    print(f"Requested inclusive date range: {start_date:%Y-%m-%d} through {end_date:%Y-%m-%d}")
    print("This script prints API totals and focused row checks only; it writes no raw data files.")
    print()

    for year in years:
        year_start = max(start_date, date(year, 1, 1))
        year_end = min(end_date, date(year, 12, 31))
        for fuel_type in FUEL_TYPES:
            year_checks.append(
                fetch_total(api_key, fuel_type, year_start, year_end, str(year))
            )

    print_total_table("Year totals from EIA response.total", year_checks)

    deficient_years = [check for check in year_checks if check.deficient]
    if not deficient_years:
        print("No deficient years found by EIA totals.")
        return 0

    month_checks: list[TotalCheck] = []
    for year_check in deficient_years:
        for month_start, month_end in month_windows(year_check.start.year):
            if month_start < year_check.start or month_end > year_check.end:
                continue
            label = f"{month_start:%Y-%m}"
            month_checks.append(
                fetch_total(
                    api_key,
                    year_check.fuel_type,
                    month_start,
                    month_end,
                    label,
                )
            )

    print_total_table("Month totals inside deficient years", month_checks)

    affected_months = [check for check in month_checks if check.deficient]
    if not affected_months:
        print(
            "EIA year totals were deficient, but no deficient month was found. "
            "That points to an API total inconsistency rather than missing hourly rows."
        )
        return 1

    focused_results: list[FetchResult] = []
    all_missing: list[tuple[str, str]] = []
    all_duplicates: list[tuple[str, str, int]] = []
    pagination_problem = False

    print("Focused affected-period row checks")
    for check in affected_months:
        result = fetch_rows_for_period(
            api_key,
            check.fuel_type,
            check.start,
            check.end,
            check.label,
        )
        focused_results.append(result)
        combos = row_combos(result.downloaded_rows, check.fuel_type)
        expected = [(period, check.fuel_type) for period in expected_periods(check.start, check.end)]
        missing = [combo for combo in expected if combos[combo] == 0]
        duplicates = [
            (period, fuel_type, count)
            for (period, fuel_type), count in sorted(combos.items())
            if count > 1
        ]

        all_missing.extend(missing)
        all_duplicates.extend(duplicates)
        pagination_problem = pagination_problem or not result.pagination_ok

        observed_periods = sorted(
            period
            for period, fuel_type in combos
            if fuel_type == check.fuel_type
        )
        earliest = observed_periods[0] if observed_periods else "not observed"
        latest = observed_periods[-1] if observed_periods else "not observed"

        print(
            f"- {check.label} {check.fuel_type}: response.total={result.response_total}, "
            f"downloaded={result.downloaded_count}, theoretical={check.theoretical_total}"
        )
        print(f"  - Page row counts: {', '.join(str(count) for count in result.page_row_counts)}")
        print(f"  - Pagination OK: {'yes' if result.pagination_ok else 'no'}")
        print(f"  - Earliest observed period: {earliest}")
        print(f"  - Latest observed period: {latest}")
        print(f"  - Missing timestamp/fuel combinations: {len(missing)}")
        if missing:
            print(f"    - {preview([f'{period}/{fuel}' for period, fuel in missing])}")
        print(f"  - Duplicate timestamp/fuel combinations: {len(duplicates)}")
        if duplicates:
            print(
                "    - "
                + preview(
                    [
                        f"{period}/{fuel} count={count}"
                        for period, fuel, count in duplicates
                    ]
                )
            )
    print()

    full_range_total_checks = [
        fetch_total(api_key, fuel_type, start_date, end_date, "full range")
        for fuel_type in FUEL_TYPES
    ]
    print_total_table("Full-range totals by fuel", full_range_total_checks)

    first_day_checks = [
        fetch_total(api_key, fuel_type, start_date, start_date, "first day")
        for fuel_type in FUEL_TYPES
    ]
    last_day_checks = [
        fetch_total(api_key, fuel_type, end_date, end_date, "last day")
        for fuel_type in FUEL_TYPES
    ]
    print_total_table("Date-boundary checks", first_day_checks + last_day_checks)

    missing_count = len(all_missing)
    duplicate_count = len(all_duplicates)
    boundary_defects = [
        check
        for check in first_day_checks + last_day_checks
        if check.eia_total != check.theoretical_total
    ]

    print("Diagnosis")
    if pagination_problem:
        print("- Pagination problem: yes. Downloaded row counts did not match EIA response.total.")
    else:
        print("- Pagination problem: no. Focused downloads matched EIA response.total.")

    if duplicate_count:
        print("- Duplicate rows: yes. Duplicate timestamp/fuel combinations were returned.")
    else:
        print("- Duplicate rows: no duplicates found in focused affected periods.")

    if boundary_defects:
        print(
            "- Date-boundary handling: needs review because first-day or last-day "
            "EIA totals were not complete."
        )
    else:
        print("- Date-boundary handling: first-day and last-day EIA totals are complete.")

    if missing_count and not pagination_problem and not duplicate_count:
        print(
            "- Likely cause: source-data missingness in EIA renewable rows for the "
            "listed timestamp/fuel combinations."
        )
    elif missing_count:
        print(
            "- Missing combinations were found, but pagination or duplicate issues "
            "also need review before assigning cause."
        )
    else:
        print("- Missing source rows: none found in focused affected periods.")

    print()
    print("Overall result: DIAGNOSTIC COMPLETE")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose EIA CISO SUN/WND historical coverage by year, month, and affected rows."
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START.strftime("%Y-%m-%d"),
        help="Inclusive start date. Default: 2022-01-01.",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END.strftime("%Y-%m-%d"),
        help="Inclusive end date. Default: 2024-12-31.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print(
            "FAIL: Missing EIA_API_KEY. Set it in your shell environment before "
            "running this diagnostic. The key is never printed.",
            file=sys.stderr,
        )
        return 1

    try:
        start_date = parse_date(args.start_date, "start date")
        end_date = parse_date(args.end_date, "end date")
        return diagnose(api_key, start_date, end_date)
    except ValueError as exc:
        print(f"FAIL: Invalid request: {exc}", file=sys.stderr)
        return 1
    except HTTPError as exc:
        print(f"FAIL: EIA request failed with HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"FAIL: Could not reach EIA API: {exc.reason}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
