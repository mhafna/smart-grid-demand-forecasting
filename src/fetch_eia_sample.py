"""Fetch a small raw EIA hourly demand sample for California ISO.

This script saves EIA's JSON response exactly as received. It does not clean,
reshape, or fabricate data.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data"
OUTPUT_PATH = Path("data/raw/eia_ciso_hourly_demand_sample.json")
MAX_HOURS = 24 * 7


def parse_hour(value: str) -> datetime:
    """Parse EIA hourly timestamps like 2024-01-01T00."""
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H")
    except ValueError as exc:
        raise ValueError(
            f"Invalid timestamp '{value}'. Use the format YYYY-MM-DDTHH, "
            "for example 2024-01-01T00."
        ) from exc


def build_url(api_key: str, start: str, end: str) -> str:
    start_time = parse_hour(start)
    end_time = parse_hour(end)
    hours = int((end_time - start_time).total_seconds() // 3600) + 1

    if hours <= 0:
        raise ValueError("The end timestamp must be after the start timestamp.")
    if hours > MAX_HOURS:
        raise ValueError("Request is too large. Use no more than seven days of hourly data.")

    params = [
        ("api_key", api_key),
        ("frequency", "hourly"),
        ("data[]", "value"),
        ("facets[respondent][]", "CISO"),
        ("facets[type][]", "D"),
        ("start", start),
        ("end", end),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("length", str(hours)),
        ("offset", "0"),
    ]
    return f"{API_URL}?{urlencode(params)}"


def fetch_json(url: str) -> bytes:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        status = response.status
        body = response.read()

    if status < 200 or status >= 300:
        raise RuntimeError(f"EIA request failed with HTTP status {status}.")

    try:
        json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("EIA response was not valid JSON; nothing was saved.") from exc

    return body


def main() -> int:
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print(
            "Missing EIA_API_KEY. Set it in your shell environment or in a local "
            ".env file that is not committed to Git.",
            file=sys.stderr,
        )
        return 1

    start = os.environ.get("EIA_SAMPLE_START", "2024-01-01T00")
    end = os.environ.get("EIA_SAMPLE_END", "2024-01-07T23")

    try:
        url = build_url(api_key, start, end)
        body = fetch_json(url)
    except ValueError as exc:
        print(f"Invalid request: {exc}", file=sys.stderr)
        return 1
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
    print(f"Saved raw EIA response to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
