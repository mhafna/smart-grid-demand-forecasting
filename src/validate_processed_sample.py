"""Validate the cleaned hourly demand plus solar/wind sample CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PROCESSED_PATH = Path("data/processed/eia_ciso_hourly_sample.csv")
REQUIRED_COLUMNS = [
    "period",
    "demand_mwh",
    "solar_generation_mwh",
    "wind_generation_mwh",
    "solar_wind_generation_mwh",
    "residual_demand_after_solar_wind_mwh",
    "solar_wind_share_pct",
]


def validate_processed_sample(path: Path) -> int:
    """Print row count, date range, duplicate timestamps, and missing values."""
    try:
        data = pd.read_csv(path)
    except OSError as exc:
        print(f"FAIL: {exc}")
        return 1

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        print("FAIL: processed CSV is missing required columns:")
        for column in missing_columns:
            print(f"- {column}")
        return 1

    timestamps = pd.to_datetime(data["period"], format="%Y-%m-%dT%H", errors="coerce")
    invalid_periods = data.loc[timestamps.isna(), "period"].astype(str).tolist()
    duplicate_periods = data.loc[data["period"].duplicated(), "period"].astype(str).tolist()
    missing_counts = data[REQUIRED_COLUMNS].isna().sum()
    columns_with_missing = missing_counts[missing_counts > 0]

    print("Processed EIA CISO hourly sample validation")
    print(f"File: {path}")
    print()
    print(f"- Row count: {len(data)}")
    if timestamps.notna().any():
        print(f"- Earliest period: {timestamps.min().strftime('%Y-%m-%dT%H')}")
        print(f"- Latest period: {timestamps.max().strftime('%Y-%m-%dT%H')}")
    else:
        print("- Earliest period: not available")
        print("- Latest period: not available")
    print(
        "- Duplicate timestamps: "
        f"{'none' if not duplicate_periods else ', '.join(duplicate_periods[:10])}"
    )
    print("- Missing values:")
    if columns_with_missing.empty:
        print("  - none")
    else:
        for column, count in columns_with_missing.items():
            print(f"  - {column}: {count}")
    print(
        "- Invalid period values: "
        f"{'none' if not invalid_periods else ', '.join(invalid_periods[:10])}"
    )

    passed = not duplicate_periods and columns_with_missing.empty and not invalid_periods
    print()
    print(f"Overall result: {'PASS' if passed else 'CHECK ISSUES ABOVE'}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the processed EIA CISO hourly sample CSV."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_PROCESSED_PATH,
        type=Path,
        help=f"Path to the processed CSV. Default: {DEFAULT_PROCESSED_PATH}",
    )
    args = parser.parse_args()
    return validate_processed_sample(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
