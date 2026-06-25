"""Download historical EIA CISO hourly demand plus solar/wind raw JSON files.

The command-line end date is inclusive. For example, an end date of 2024-12-31
requests through 2024-12-31T23 so the final calendar day is complete.

This script only downloads and saves raw EIA responses. It does not clean,
reshape, fill, or invent data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEMAND_API_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data"
RENEWABLE_API_URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data"

RESPONDENT = "CISO"
DEMAND_TYPE = "D"
FUEL_TYPES = ("SUN", "WND")

PAGE_SIZE = 5000
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 30
MAX_DAYS = 366 * 3

DEFAULT_DEMAND_OUTPUT = Path("data/raw/eia_ciso_hourly_demand_2022_2024.json")
DEFAULT_RENEWABLE_OUTPUT = Path(
    "data/raw/eia_ciso_hourly_renewable_generation_2022_2024.json"
)


def parse_date(value: str, label: str) -> date:
    """Parse a command-line date in YYYY-MM-DD format."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"{label} must use YYYY-MM-DD format, for example 2022-01-01."
        ) from exc


def date_window(start_text: str, end_text: str) -> tuple[date, date, int]:
    """Validate the date range and return inclusive day count."""
    start_date = parse_date(start_text, "start date")
    end_date = parse_date(end_text, "end date")

    if end_date < start_date:
        raise ValueError("End date must be the same as or later than start date.")

    days = (end_date - start_date).days + 1
    if days > MAX_DAYS:
        raise ValueError(
            f"Requested {days} days, which is broader than the {MAX_DAYS}-day "
            "safety limit. Split larger downloads into smaller reviewed ranges."
        )

    return start_date, end_date, days


def eia_start_end(start_date: date, end_date: date) -> tuple[str, str]:
    """Convert inclusive command-line dates to EIA hourly start/end strings."""
    return f"{start_date:%Y-%m-%d}T00", f"{end_date:%Y-%m-%d}T23"


def build_demand_params(
    api_key: str, start: str, end: str, offset: int
) -> list[tuple[str, str]]:
    """Build one demand page request."""
    return [
        ("api_key", api_key),
        ("frequency", "hourly"),
        ("data[]", "value"),
        ("facets[respondent][]", RESPONDENT),
        ("facets[type][]", DEMAND_TYPE),
        ("start", start),
        ("end", end),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("length", str(PAGE_SIZE)),
        ("offset", str(offset)),
    ]


def build_renewable_params(
    api_key: str, start: str, end: str, offset: int
) -> list[tuple[str, str]]:
    """Build one solar/wind generation page request."""
    params = [
        ("api_key", api_key),
        ("frequency", "hourly"),
        ("data[]", "value"),
        ("facets[respondent][]", RESPONDENT),
        ("start", start),
        ("end", end),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("sort[1][column]", "fueltype"),
        ("sort[1][direction]", "asc"),
        ("length", str(PAGE_SIZE)),
        ("offset", str(offset)),
    ]
    for fuel_type in FUEL_TYPES:
        params.append(("facets[fueltype][]", fuel_type))
    return params


def is_temporary_http_error(status: int) -> bool:
    """Return true for errors that are usually worth retrying briefly."""
    return status in {408, 429, 500, 502, 503, 504}


def request_json(url: str) -> dict[str, Any]:
    """Fetch one URL and validate that EIA returned a JSON object."""
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        status = response.status
        body = response.read()

    if status < 200 or status >= 300:
        raise RuntimeError(f"EIA request failed with HTTP status {status}.")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("EIA response was not valid JSON; nothing was saved.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("EIA response was not a JSON object; nothing was saved.")

    return payload


def fetch_page_with_retries(url: str) -> dict[str, Any]:
    """Retry only temporary API/network failures, with a small fixed limit."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return request_json(url)
        except HTTPError as exc:
            if not is_temporary_http_error(exc.code) or attempt == MAX_RETRIES:
                raise
            print(
                f"Temporary HTTP {exc.code}; retrying page "
                f"({attempt}/{MAX_RETRIES - 1})..."
            )
        except URLError as exc:
            if attempt == MAX_RETRIES:
                raise
            print(
                f"Temporary network problem: {exc.reason}; retrying page "
                f"({attempt}/{MAX_RETRIES - 1})..."
            )

        time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError("Retry loop ended unexpectedly.")


def rows_from_payload(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    """Return response.data rows and fail if the EIA response shape changed."""
    response = payload.get("response")
    if not isinstance(response, dict):
        raise RuntimeError(f"{label} did not contain a response object.")

    rows = response.get("data")
    if not isinstance(rows, list):
        raise RuntimeError(f"{label} did not contain response.data as a list.")

    dict_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise RuntimeError(f"{label} row {row_number} was not an object.")
        dict_rows.append(row)

    return dict_rows


def total_from_payload(payload: dict[str, Any], label: str) -> int:
    """Read EIA response.total, accepting string or integer forms."""
    response = payload.get("response")
    if not isinstance(response, dict):
        raise RuntimeError(f"{label} did not contain a response object.")

    total = response.get("total")
    try:
        return int(total)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} response.total was not an integer: {total!r}") from exc


def redact_api_key_metadata(payload: dict[str, Any]) -> None:
    """Remove echoed API keys from EIA request metadata before saving raw pages."""
    request = payload.get("request")
    if not isinstance(request, dict):
        return

    params = request.get("params")
    if isinstance(params, dict) and "api_key" in params:
        params["api_key"] = "[REDACTED]"


def fetch_all_pages(
    api_key: str,
    url: str,
    params_builder: Any,
    start: str,
    end: str,
    expected_rows: int,
    label: str,
) -> dict[str, Any]:
    """Fetch all EIA pages into memory before anything is written to disk."""
    pages: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    downloaded_rows = 0

    while total is None or offset < total:
        params = params_builder(api_key, start, end, offset)
        page_url = f"{url}?{urlencode(params)}"
        page_number = len(pages) + 1
        print(f"Requesting {label} page {page_number} at offset {offset}...")

        payload = fetch_page_with_retries(page_url)
        rows = rows_from_payload(payload, f"{label} page {page_number}")
        page_total = total_from_payload(payload, f"{label} page {page_number}")

        if total is None:
            total = page_total
            if total != expected_rows:
                raise RuntimeError(
                    f"EIA reported {total} {label} rows, but {expected_rows} were "
                    "expected for the requested complete hourly range. Nothing was saved."
                )
        elif page_total != total:
            raise RuntimeError(
                f"EIA response.total changed from {total} to {page_total} during "
                f"the {label} download. Nothing was saved."
            )

        if not rows and offset < total:
            raise RuntimeError(
                f"EIA returned no {label} rows at offset {offset}; pagination stopped "
                "making progress. Nothing was saved."
            )

        redact_api_key_metadata(payload)
        pages.append(payload)
        downloaded_rows += len(rows)
        offset += len(rows)
        print(f"Downloaded {downloaded_rows}/{total} {label} rows.")

    if downloaded_rows != expected_rows:
        raise RuntimeError(
            f"Downloaded {downloaded_rows} {label} rows, but expected {expected_rows}. "
            "Nothing was saved."
        )

    return {
        "note": (
            "Untouched EIA page responses. Rows are preserved in "
            "pages[].response.data exactly as returned by EIA."
        ),
        "download": {
            "respondent": RESPONDENT,
            "start": start,
            "end": end,
            "end_handling": "inclusive command-line end date converted to YYYY-MM-DDT23",
            "page_size": PAGE_SIZE,
            "expected_rows": expected_rows,
        },
        "pages": pages,
    }


def write_json_after_success(path: Path, payload: dict[str, Any]) -> None:
    """Write the completed payload to a temporary file, then replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download raw EIA CISO hourly historical demand and solar/wind JSON."
    )
    parser.add_argument("start_date", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("end_date", help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument(
        "--demand-output",
        default=DEFAULT_DEMAND_OUTPUT,
        type=Path,
        help=f"Demand output path. Default: {DEFAULT_DEMAND_OUTPUT}",
    )
    parser.add_argument(
        "--renewable-output",
        default=DEFAULT_RENEWABLE_OUTPUT,
        type=Path,
        help=f"Renewable output path. Default: {DEFAULT_RENEWABLE_OUTPUT}",
    )
    args = parser.parse_args()

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print(
            "FAIL: Missing EIA_API_KEY. Set it in your shell environment before "
            "running this script. The key is never read from a file or printed.",
            file=sys.stderr,
        )
        return 1

    try:
        start_date, end_date, days = date_window(args.start_date, args.end_date)
        start, end = eia_start_end(start_date, end_date)
        expected_hours = days * 24
        expected_renewable_rows = expected_hours * len(FUEL_TYPES)

        print(
            f"Downloading CISO hourly data from {start} through {end} "
            "(inclusive end date)."
        )
        print(f"Expected demand rows: {expected_hours}")
        print(f"Expected renewable rows: {expected_renewable_rows}")

        demand_payload = fetch_all_pages(
            api_key,
            DEMAND_API_URL,
            build_demand_params,
            start,
            end,
            expected_hours,
            "demand",
        )
        renewable_payload = fetch_all_pages(
            api_key,
            RENEWABLE_API_URL,
            build_renewable_params,
            start,
            end,
            expected_renewable_rows,
            "renewable",
        )

        write_json_after_success(args.demand_output, demand_payload)
        write_json_after_success(args.renewable_output, renewable_payload)
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
    except OSError as exc:
        print(f"FAIL: Could not save completed download: {exc}", file=sys.stderr)
        return 1

    print(f"Saved completed raw demand pages to {args.demand_output}")
    print(f"Saved completed raw renewable pages to {args.renewable_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
