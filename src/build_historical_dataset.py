"""Build the processed historical EIA CISO demand plus solar/wind CSV."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from build_sample_dataset import (
    EXPECTED_RENEWABLE_FUELS,
    add_timestamp_column,
)
from validate_eia_history import (
    DOCUMENTED_GAP_END,
    DOCUMENTED_GAP_START,
    DOCUMENTED_NULL_TIMESTAMPS,
)


DEFAULT_DEMAND_PATH = Path("data/raw/eia_ciso_hourly_demand_2022_2024.json")
DEFAULT_RENEWABLE_PATH = Path(
    "data/raw/eia_ciso_hourly_renewable_generation_2022_2024.json"
)
DEFAULT_OUTPUT_PATH = Path("data/processed/eia_ciso_hourly_2022_2024.csv")

DEMAND_REQUIRED_FIELDS = {"period", "value"}
RENEWABLE_REQUIRED_FIELDS = {"period", "fueltype", "value"}


def require_fields_present(
    rows: list[dict[str, Any]], required_fields: set[str], label: str
) -> None:
    """Require fields to exist while allowing documented null measurements."""
    problems: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        missing = required_fields - set(row)
        if missing:
            problems.append(
                f"row {row_number}: missing {', '.join(sorted(missing))}"
            )
    if problems:
        preview = "; ".join(problems[:10])
        raise ValueError(f"{label} has missing required fields: {preview}")


def numeric_column_preserving_nulls(
    frame: pd.DataFrame, source: str, target: str, label: str
) -> pd.DataFrame:
    """Convert numeric values while preserving JSON nulls as missing values."""
    frame = frame.copy()
    frame[target] = pd.to_numeric(frame[source], errors="coerce")
    invalid = frame[source].notna() & frame[target].isna()
    if invalid.any():
        bad_values = frame.loc[invalid, ["period", source]]
        preview = ", ".join(
            f"{row.period}={getattr(row, source)!r}"
            for row in bad_values.head(10).itertuples(index=False)
        )
        raise ValueError(f"{label} has non-null, non-numeric values: {preview}")
    return frame


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
    require_fields_present(demand_rows, DEMAND_REQUIRED_FIELDS, "Historical demand")
    require_fields_present(
        renewable_rows, RENEWABLE_REQUIRED_FIELDS, "Historical renewable"
    )

    demand = pd.DataFrame(demand_rows)
    demand = add_timestamp_column(demand, "Historical demand")
    demand = numeric_column_preserving_nulls(
        demand, "value", "demand_mwh", "Historical demand"
    )

    observed_demand_nulls = set(
        demand.loc[demand["value"].isna(), "timestamp"].dt.to_pydatetime()
    )
    if observed_demand_nulls != DOCUMENTED_NULL_TIMESTAMPS:
        raise ValueError(
            "Historical demand null timestamps differ from the documented EIA "
            "source exceptions. Run validate_eia_history.py for details."
        )

    duplicate_demand = demand.loc[demand["timestamp"].duplicated(), "period"].tolist()
    if duplicate_demand:
        preview = ", ".join(duplicate_demand[:10])
        raise ValueError(f"Historical demand has duplicate hourly timestamps: {preview}")

    demand = demand[["timestamp", "period", "demand_mwh"]]
    demand = demand.sort_values("timestamp").reset_index(drop=True)
    expected_start = pd.Timestamp("2022-01-01T00")
    expected_end = pd.Timestamp("2024-12-31T23")
    if (
        len(demand) != 26_304
        or demand["timestamp"].min() != expected_start
        or demand["timestamp"].max() != expected_end
        or not demand["timestamp"].diff().dropna().eq(timedelta(hours=1)).all()
    ):
        raise ValueError(
            "Historical demand must contain all 26,304 consecutive hourly "
            "timestamps from 2022-01-01T00 through 2024-12-31T23."
        )

    renewable = pd.DataFrame(renewable_rows)
    observed_fuels = set(renewable["fueltype"])
    if observed_fuels != EXPECTED_RENEWABLE_FUELS:
        raise ValueError(
            "Historical renewable fuel categories differ from SUN and WND: "
            + ", ".join(sorted(str(fuel) for fuel in observed_fuels))
        )
    renewable = add_timestamp_column(renewable, "Historical renewable")
    renewable = numeric_column_preserving_nulls(
        renewable, "value", "generation_mwh", "Historical renewable"
    )

    observed_renewable_nulls = set(
        zip(
            renewable.loc[renewable["value"].isna(), "timestamp"].dt.to_pydatetime(),
            renewable.loc[renewable["value"].isna(), "fueltype"],
        )
    )
    expected_renewable_nulls = {
        (period, fuel)
        for period in DOCUMENTED_NULL_TIMESTAMPS
        for fuel in EXPECTED_RENEWABLE_FUELS
    }
    if observed_renewable_nulls != expected_renewable_nulls:
        raise ValueError(
            "Historical renewable null timestamp/fuel combinations differ from "
            "the documented EIA source exceptions. Run validate_eia_history.py "
            "for details."
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

    renewable_outside_demand = renewable_pivot.index.difference(demand["timestamp"])
    if not renewable_outside_demand.empty:
        preview = ", ".join(
            timestamp.strftime("%Y-%m-%dT%H")
            for timestamp in renewable_outside_demand[:10]
        )
        raise ValueError(
            "Historical renewable data contains timestamps outside the demand "
            f"timeline: {preview}"
        )

    combined = demand.merge(
        renewable_pivot,
        how="left",
        left_on="timestamp",
        right_index=True,
        validate="one_to_one",
    )

    if combined["period"].isna().any():
        raise ValueError(
            "The demand timeline contains a missing period after joining."
        )

    documented_renewable_unavailable = pd.DatetimeIndex(
        sorted(
            set(pd.date_range(DOCUMENTED_GAP_START, DOCUMENTED_GAP_END, freq="h"))
            | {pd.Timestamp(period) for period in DOCUMENTED_NULL_TIMESTAMPS}
        )
    )
    for column in ["solar_generation_mwh", "wind_generation_mwh"]:
        missing_timestamps = pd.DatetimeIndex(
            combined.loc[combined[column].isna(), "timestamp"]
        )
        if not missing_timestamps.equals(documented_renewable_unavailable):
            preview = ", ".join(
                timestamp.strftime("%Y-%m-%dT%H")
                for timestamp in missing_timestamps[:10]
            )
            raise ValueError(
                f"{column} missing timestamps differ from the documented EIA "
                f"source gap. Observed: {preview or 'none'}"
            )

    if len(combined) != 26_304 or combined["timestamp"].duplicated().any():
        raise ValueError(
            "The combined dataset did not preserve the 26,304 unique demand timestamps."
        )

    combined["demand_data_complete"] = combined["demand_mwh"].notna()
    combined["renewable_data_complete"] = combined[
        ["solar_generation_mwh", "wind_generation_mwh"]
    ].notna().all(axis=1)
    combined["solar_wind_generation_mwh"] = (
        combined["solar_generation_mwh"] + combined["wind_generation_mwh"]
    ).where(combined["renewable_data_complete"])
    all_inputs_complete = (
        combined["demand_data_complete"] & combined["renewable_data_complete"]
    )
    combined["residual_demand_after_solar_wind_mwh"] = (
        combined["demand_mwh"] - combined["solar_wind_generation_mwh"]
    ).where(all_inputs_complete)
    combined["solar_wind_share_pct"] = (
        combined["solar_wind_generation_mwh"] / combined["demand_mwh"] * 100
    ).where(all_inputs_complete)

    return combined[
        [
            "period",
            "demand_mwh",
            "solar_generation_mwh",
            "wind_generation_mwh",
            "demand_data_complete",
            "renewable_data_complete",
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
    demand_complete_rows = int(dataset["demand_data_complete"].sum())
    renewable_complete_rows = int(dataset["renewable_data_complete"].sum())
    print(f"Demand data complete rows: {demand_complete_rows}")
    print(f"Demand data incomplete rows: {len(dataset) - demand_complete_rows}")
    print(f"Renewable data complete rows: {renewable_complete_rows}")
    print(
        f"Renewable data incomplete rows: {len(dataset) - renewable_complete_rows}"
    )
    print(
        "Residual demand note: this subtracts only reported solar and wind "
        "generation from demand; it does not account for other generation, "
        "storage, or interchange."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
