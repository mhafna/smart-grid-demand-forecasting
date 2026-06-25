"""Build the cleaned hourly demand plus solar/wind sample dataset.

The residual demand column created here is only demand minus reported solar and
wind generation. It is not a complete physical grid balance because it does not
account for other generation, storage, or interchange.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DEMAND_PATH = Path("data/raw/eia_ciso_hourly_demand_sample.json")
DEFAULT_RENEWABLE_PATH = Path(
    "data/raw/eia_ciso_hourly_renewable_generation_sample.json"
)
DEFAULT_OUTPUT_PATH = Path("data/processed/eia_ciso_hourly_sample.csv")

DEMAND_REQUIRED_FIELDS = {"period", "value"}
RENEWABLE_REQUIRED_FIELDS = {"period", "fueltype", "value"}
EXPECTED_RENEWABLE_FUELS = {"SUN", "WND"}


def load_response_data(path: Path) -> list[dict[str, Any]]:
    """Load response.data rows from a local EIA JSON file."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} has a top-level JSON value that is not an object.")

    response = payload.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"{path} is missing a response object.")

    rows = response.get("data")
    if not isinstance(rows, list):
        raise ValueError(f"{path} is missing response.data as a list.")

    dict_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path} response.data row {row_number} is not an object.")
        dict_rows.append(row)

    return dict_rows


def require_fields(rows: list[dict[str, Any]], required_fields: set[str], label: str) -> None:
    """Fail clearly if required fields are missing or blank."""
    problems: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        for field in sorted(required_fields):
            if field not in row:
                problems.append(f"row {row_number}: missing {field}")
            elif row[field] is None or row[field] == "":
                problems.append(f"row {row_number}: blank {field}")

    if problems:
        preview = "; ".join(problems[:10])
        extra = "" if len(problems) <= 10 else f"; ... {len(problems) - 10} more"
        raise ValueError(f"{label} has missing required values: {preview}{extra}")


def add_timestamp_column(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Parse period for sorting and joining while preserving the original string."""
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(
        frame["period"], format="%Y-%m-%dT%H", errors="coerce"
    )

    missing_timestamps = frame.loc[frame["timestamp"].isna(), "period"].tolist()
    if missing_timestamps:
        preview = ", ".join(repr(value) for value in missing_timestamps[:10])
        raise ValueError(f"{label} has missing or invalid hourly periods: {preview}")

    return frame


def numeric_column(frame: pd.DataFrame, source: str, target: str, label: str) -> pd.DataFrame:
    """Convert a value column to numbers and fail if any conversion is impossible."""
    frame = frame.copy()
    frame[target] = pd.to_numeric(frame[source], errors="coerce")

    bad_values = frame.loc[frame[target].isna(), source].tolist()
    if bad_values:
        preview = ", ".join(repr(value) for value in bad_values[:10])
        raise ValueError(f"{label} has non-numeric values: {preview}")

    return frame


def build_dataset(demand_path: Path, renewable_path: Path) -> pd.DataFrame:
    """Return a chronologically sorted hourly demand plus solar/wind table."""
    demand_rows = load_response_data(demand_path)
    renewable_rows = load_response_data(renewable_path)
    require_fields(demand_rows, DEMAND_REQUIRED_FIELDS, "Demand sample")
    require_fields(renewable_rows, RENEWABLE_REQUIRED_FIELDS, "Renewable sample")

    demand = pd.DataFrame(demand_rows)
    demand = add_timestamp_column(demand, "Demand sample")
    demand = numeric_column(demand, "value", "demand_mwh", "Demand sample")

    duplicate_demand = demand.loc[demand["timestamp"].duplicated(), "period"].tolist()
    if duplicate_demand:
        preview = ", ".join(duplicate_demand[:10])
        raise ValueError(f"Demand sample has duplicate hourly timestamps: {preview}")

    demand = demand[["timestamp", "period", "demand_mwh"]]

    renewable = pd.DataFrame(renewable_rows)
    renewable = renewable[renewable["fueltype"].isin(EXPECTED_RENEWABLE_FUELS)].copy()
    renewable = add_timestamp_column(renewable, "Renewable sample")
    renewable = numeric_column(
        renewable, "value", "generation_mwh", "Renewable sample"
    )

    observed_fuels = set(renewable["fueltype"])
    missing_fuels = EXPECTED_RENEWABLE_FUELS - observed_fuels
    if missing_fuels:
        raise ValueError(
            "Renewable sample is missing expected fuel categories: "
            + ", ".join(sorted(missing_fuels))
        )

    duplicate_renewable = renewable.loc[
        renewable.duplicated(subset=["timestamp", "fueltype"]),
        ["period", "fueltype"],
    ]
    if not duplicate_renewable.empty:
        preview = ", ".join(
            f"{row.period}/{row.fueltype}"
            for row in duplicate_renewable.itertuples(index=False)
        )
        raise ValueError(
            "Renewable sample has duplicate timestamp/fuel rows: " + preview
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
            "Demand, solar, and wind timestamps do not fully align. "
            f"Missing observations at: {preview}"
        )

    combined = combined.sort_values("timestamp").reset_index(drop=True)
    if combined["timestamp"].duplicated().any():
        duplicates = combined.loc[combined["timestamp"].duplicated(), "period"].tolist()
        raise ValueError("Combined data has duplicate timestamps: " + ", ".join(duplicates))

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
        description="Build the cleaned EIA CISO hourly demand and solar/wind sample CSV."
    )
    parser.add_argument(
        "--demand-path",
        default=DEFAULT_DEMAND_PATH,
        type=Path,
        help=f"Path to the raw demand JSON. Default: {DEFAULT_DEMAND_PATH}",
    )
    parser.add_argument(
        "--renewable-path",
        default=DEFAULT_RENEWABLE_PATH,
        type=Path,
        help=f"Path to the raw renewable JSON. Default: {DEFAULT_RENEWABLE_PATH}",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        type=Path,
        help=f"Path for the processed CSV. Default: {DEFAULT_OUTPUT_PATH}",
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
