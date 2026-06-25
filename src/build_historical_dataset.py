"""Build the processed historical EIA CISO demand plus solar/wind CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from build_sample_dataset import (
    EXPECTED_RENEWABLE_FUELS,
    add_timestamp_column,
    numeric_column,
    require_fields,
)


DEFAULT_DEMAND_PATH = Path("data/raw/eia_ciso_hourly_demand_2022_2024.json")
DEFAULT_RENEWABLE_PATH = Path(
    "data/raw/eia_ciso_hourly_renewable_generation_2022_2024.json"
)
DEFAULT_OUTPUT_PATH = Path("data/processed/eia_ciso_hourly_2022_2024.csv")

DEMAND_REQUIRED_FIELDS = {"period", "value"}
RENEWABLE_REQUIRED_FIELDS = {"period", "fueltype", "value"}


def load_response_data(path: Path) -> list[dict[str, Any]]:
    """Load rows from one EIA response or from a historical pagination wrapper."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} has a top-level JSON value that is not an object.")

    if "pages" not in payload:
        return rows_from_response(payload, f"{path} response")

    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"{path} has a pages field that is not a list.")

    rows: list[dict[str, Any]] = []
    for page_number, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            raise ValueError(f"{path} page {page_number} is not an object.")
        rows.extend(rows_from_response(page, f"{path} page {page_number}"))
    return rows


def rows_from_response(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    response = payload.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"{label} is missing a response object.")
    rows = response.get("data")
    if not isinstance(rows, list):
        raise ValueError(f"{label} is missing response.data as a list.")

    dict_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {row_number} is not an object.")
        dict_rows.append(row)
    return dict_rows


def build_dataset(demand_path: Path, renewable_path: Path) -> pd.DataFrame:
    """Combine demand, solar, and wind by timestamp without filling gaps."""
    demand_rows = load_response_data(demand_path)
    renewable_rows = load_response_data(renewable_path)
    require_fields(demand_rows, DEMAND_REQUIRED_FIELDS, "Historical demand")
    require_fields(renewable_rows, RENEWABLE_REQUIRED_FIELDS, "Historical renewable")

    demand = pd.DataFrame(demand_rows)
    demand = add_timestamp_column(demand, "Historical demand")
    demand = numeric_column(demand, "value", "demand_mwh", "Historical demand")

    duplicate_demand = demand.loc[demand["timestamp"].duplicated(), "period"].tolist()
    if duplicate_demand:
        preview = ", ".join(duplicate_demand[:10])
        raise ValueError(f"Historical demand has duplicate hourly timestamps: {preview}")

    demand = demand[["timestamp", "period", "demand_mwh"]]

    renewable = pd.DataFrame(renewable_rows)
    renewable = renewable[renewable["fueltype"].isin(EXPECTED_RENEWABLE_FUELS)].copy()
    renewable = add_timestamp_column(renewable, "Historical renewable")
    renewable = numeric_column(
        renewable, "value", "generation_mwh", "Historical renewable"
    )

    observed_fuels = set(renewable["fueltype"])
    missing_fuels = EXPECTED_RENEWABLE_FUELS - observed_fuels
    if missing_fuels:
        raise ValueError(
            "Historical renewable data is missing expected fuel categories: "
            + ", ".join(sorted(missing_fuels))
        )

    duplicate_renewable = renewable.loc[
        renewable.duplicated(subset=["timestamp", "fueltype"]),
        ["period", "fueltype"],
    ]
    if not duplicate_renewable.empty:
        preview = ", ".join(
            f"{row.period}/{row.fueltype}"
            for row in duplicate_renewable.head(10).itertuples(index=False)
        )
        raise ValueError(
            "Historical renewable data has duplicate timestamp/fuel rows: " + preview
        )

    renewable_pivot = renewable.pivot(
        index="timestamp", columns="fueltype", values="generation_mwh"
    ).rename(
        columns={
            "SUN": "solar_generation_mwh",
            "WND": "wind_generation_mwh",
        }
    )

    combined = demand.merge(
        renewable_pivot,
        how="outer",
        left_on="timestamp",
        right_index=True,
        validate="one_to_one",
    )

    required_output_values = [
        "period",
        "demand_mwh",
        "solar_generation_mwh",
        "wind_generation_mwh",
    ]
    missing_rows = combined[combined[required_output_values].isna().any(axis=1)]
    if not missing_rows.empty:
        periods = combined.loc[missing_rows.index, "period"].fillna(
            combined.loc[missing_rows.index, "timestamp"].dt.strftime("%Y-%m-%dT%H")
        )
        preview = ", ".join(periods.astype(str).tolist()[:10])
        raise ValueError(
            "Historical demand, solar, and wind timestamps do not fully align. "
            f"Missing observations at: {preview}"
        )

    combined = combined.sort_values("timestamp").reset_index(drop=True)
    if combined["timestamp"].duplicated().any():
        duplicates = combined.loc[combined["timestamp"].duplicated(), "period"].tolist()
        raise ValueError(
            "Combined historical data has duplicate timestamps: "
            + ", ".join(duplicates[:10])
        )

    combined["solar_wind_generation_mwh"] = (
        combined["solar_generation_mwh"] + combined["wind_generation_mwh"]
    )
    combined["residual_demand_after_solar_wind_mwh"] = (
        combined["demand_mwh"] - combined["solar_wind_generation_mwh"]
    )
    combined["solar_wind_share_pct"] = (
        combined["solar_wind_generation_mwh"] / combined["demand_mwh"] * 100
    )

    return combined[
        [
            "period",
            "demand_mwh",
            "solar_generation_mwh",
            "wind_generation_mwh",
            "solar_wind_generation_mwh",
            "residual_demand_after_solar_wind_mwh",
            "solar_wind_share_pct",
        ]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build processed historical EIA CISO demand and solar/wind CSV."
    )
    parser.add_argument(
        "--demand-path",
        default=DEFAULT_DEMAND_PATH,
        type=Path,
        help=f"Historical demand JSON path. Default: {DEFAULT_DEMAND_PATH}",
    )
    parser.add_argument(
        "--renewable-path",
        default=DEFAULT_RENEWABLE_PATH,
        type=Path,
        help=f"Historical renewable JSON path. Default: {DEFAULT_RENEWABLE_PATH}",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        type=Path,
        help=f"Processed CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    args = parser.parse_args()

    try:
        dataset = build_dataset(args.demand_path, args.renewable_path)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(args.output_path, index=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"Wrote {len(dataset)} rows to {args.output_path}")
    print(
        "Residual demand note: this subtracts only reported solar and wind "
        "generation from demand; it does not account for other generation, "
        "storage, or interchange."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
