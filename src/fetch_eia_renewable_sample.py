"""Fetch a seven-day EIA CISO hourly solar and wind generation sample.

The script requests only the small January 1-7, 2024 sample needed for this
project. It saves the EIA response under data/raw/ and does not clean, reshape,
or invent data.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data"
OUTPUT_PATH = Path("data/raw/eia_ciso_hourly_renewable_generation_sample.json")

RESPONDENT = "CISO"
FUEL_TYPES = ("SUN", "WND")
START = "2024-01-01T00"
END = "2024-01-07T23"

# EIA's JSON API limit is 5,000 rows. This sample should be only 336 rows,
# but using the limit keeps pagination simple and safe if the API returns more.
PAGE_SIZE = 5000
EXPECTED_MAX_ROWS = 24 * 7 * len(FUEL_TYPES)


def build_params(api_key: str, offset: int) -> list[tuple[str, str]]:
    """Build the query parameters for one EIA API page."""
    params = [
        ("api_key", api_key),
        ("frequency", "hourly"),
        ("data[]", "value"),
        ("facets[respondent][]", RESPONDENT),
        ("start", START),
        ("end", END),
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


def fetch_page(api_key: str, offset: int) -> tuple[bytes, dict[str, Any]]:
    """Fetch one API page and return both the raw bytes and parsed JSON."""
    url = f"{API_URL}?{urlencode(build_params(api_key, offset))}"
    request = Request(url, headers={"Accept": "application/json"})

    with urlopen(request, timeout=30) as response:
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

    return body, payload


def rows_from_payload(payload: dict[str, Any]) -> list[Any]:
    """Return response.data and fail if the API response shape is unexpected."""
    response = payload.get("response")
    if not isinstance(response, dict):
        raise RuntimeError("EIA response did not contain a response object.")

    rows = response.get("data")
    if not isinstance(rows, list):
        raise RuntimeError("EIA response did not contain response.data as a list.")

    return rows


def total_from_payload(payload: dict[str, Any]) -> int:
    """Read response.total, accepting EIA's string or integer forms."""
    response = payload.get("response")
    if not isinstance(response, dict):
        raise RuntimeError("EIA response did not contain a response object.")

    total = response.get("total")
    try:
        return int(total)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"EIA response.total was not an integer: {total!r}") from exc


def fetch_all_pages(api_key: str) -> tuple[bytes, list[dict[str, Any]]]:
    """Fetch the sample, following pagination only if EIA says it is needed."""
    first_body, first_payload = fetch_page(api_key, offset=0)
    pages = [first_payload]
    first_rows = rows_from_payload(first_payload)
    total = total_from_payload(first_payload)

    if total > EXPECTED_MAX_ROWS:
        raise RuntimeError(
            "EIA returned more rows than expected for seven days of two fuel "
            f"types: total={total}, expected at most {EXPECTED_MAX_ROWS}. "
            "Check the route parameters before saving data."
        )

    # The normal path is one page. In that case, save EIA's exact response bytes.
    if len(first_rows) >= total:
        return first_body, pages

    offset = len(first_rows)
    while offset < total:
        _, payload = fetch_page(api_key, offset=offset)
        rows = rows_from_payload(payload)
        if not rows:
            raise RuntimeError(
                f"EIA pagination stopped making progress at offset {offset}."
            )
        pages.append(payload)
        offset += len(rows)

    # Multiple pages are not expected for this sample. If they happen, preserve
    # each EIA page object in a small wrapper rather than merging or reshaping rows.
    paginated_body = json.dumps(
        {
            "note": "Multiple untouched EIA page responses were returned for this sample.",
            "pages": pages,
        },
        indent=2,
    ).encode("utf-8")
    return paginated_body, pages


def main() -> int:
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print(
            "Missing EIA_API_KEY. Set it securely in your shell environment "
            "before running this script.",
            file=sys.stderr,
        )
        return 1

    try:
        body, _ = fetch_all_pages(api_key)
    except HTTPError as exc:
        print(f"EIA request failed with HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"Could not reach EIA API: {exc.reason}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(body)
    print(f"Saved raw EIA renewable response to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
