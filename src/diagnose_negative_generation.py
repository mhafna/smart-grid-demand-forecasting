"""Diagnose negative EIA CISO solar and wind generation values locally.

This script is read-only with respect to the processed CSV and raw JSON. It
compares processed renewable values with the historical raw response, measures
the temporal and numerical scope of negative values, and writes focused tables,
figures, and a Markdown report below results/data_quality/.

All timestamps and calendar groupings are labelled UTC. No conversion to or
interpretation as California local clock time is performed.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_PATH = (
    ROOT / "data" / "processed" / "eia_ciso_hourly_2022_2024.csv"
)
DEFAULT_RAW_RENEWABLE_PATH = (
    ROOT / "data" / "raw" / "eia_ciso_hourly_renewable_generation_2022_2024.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "data_quality"

SOLAR = "solar_generation_mwh"
WIND = "wind_generation_mwh"
COMBINED = "solar_wind_generation_mwh"
DEMAND = "demand_mwh"
RESIDUAL = "residual_demand_after_solar_wind_mwh"
SHARE = "solar_wind_share_pct"
METRICS = [SOLAR, WIND, COMBINED]
DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# This is an empirical diagnostic proxy, not a claim about California daylight.
LOW_OUTPUT_MEDIAN_THRESHOLD_MWH = 1.0
SUBSTANTIAL_NEGATIVE_THRESHOLD_MWH = -25.0
FORMULA_TOLERANCE = 1e-9


def load_processed(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load numeric and text copies so source values can be compared exactly."""
    required = {
        "period",
        DEMAND,
        SOLAR,
        WIND,
        COMBINED,
        RESIDUAL,
        SHARE,
        "demand_data_complete",
        "renewable_data_complete",
    }
    text = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(required.difference(text.columns))
    if missing:
        raise ValueError(f"Processed CSV is missing required columns: {missing}")

    frame = pd.read_csv(path, parse_dates=["period"])
    if frame["period"].isna().any():
        raise ValueError("Processed CSV contains an unparseable period.")
    if frame["period"].duplicated().any():
        raise ValueError("Processed CSV contains duplicate periods.")

    frame = frame.sort_values("period").reset_index(drop=True)
    text = text.set_index("period").loc[
        frame["period"].dt.strftime("%Y-%m-%dT%H")
    ].reset_index()
    expected = pd.date_range(frame["period"].min(), frame["period"].max(), freq="h")
    if not pd.DatetimeIndex(frame["period"]).equals(expected):
        raise ValueError("Processed CSV is not a continuous unique hourly series.")
    return add_utc_columns(frame), text


def add_utc_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar fields using the source timestamps without timezone conversion."""
    result = frame.copy()
    result["year"] = result["period"].dt.year
    result["month"] = result["period"].dt.month
    result["year_month_utc"] = result["period"].dt.to_period("M").astype(str)
    result["utc_hour"] = result["period"].dt.hour
    result["day_of_week_utc"] = result["period"].dt.day_name()
    return result


def rows_from_raw_json(path: Path) -> list[dict[str, Any]]:
    """Read only response data rows; request/download metadata is never printed."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Raw renewable JSON must have a top-level object.")

    pages = payload.get("pages")
    page_payloads = pages if pages is not None else [payload]
    if not isinstance(page_payloads, list):
        raise ValueError("Raw renewable JSON pages field must be a list.")

    rows: list[dict[str, Any]] = []
    for page_number, page in enumerate(page_payloads, start=1):
        if not isinstance(page, dict):
            raise ValueError(f"Raw renewable page {page_number} is not an object.")
        response = page.get("response")
        data = response.get("data") if isinstance(response, dict) else None
        if not isinstance(data, list):
            raise ValueError(
                f"Raw renewable page {page_number} has no response.data list."
            )
        for row in data:
            if not isinstance(row, dict):
                raise ValueError("Raw renewable response contains a non-object row.")
            rows.append(row)
    return rows


def load_raw_renewable(path: Path) -> pd.DataFrame:
    """Load only fields needed for local source preservation checks."""
    rows = rows_from_raw_json(path)
    required = {"period", "fueltype", "value"}
    for row_number, row in enumerate(rows, start=1):
        missing = required.difference(row)
        if missing:
            raise ValueError(
                f"Raw renewable row {row_number} is missing: {sorted(missing)}"
            )
    raw = pd.DataFrame(
        {
            "period": [row["period"] for row in rows],
            "fueltype": [row["fueltype"] for row in rows],
            "raw_value": [row["value"] for row in rows],
        }
    )
    raw = raw.loc[raw["fueltype"].isin(["SUN", "WND"])].copy()
    if raw.duplicated(["period", "fueltype"]).any():
        raise ValueError("Raw renewable data has duplicate period/fuel rows.")
    return raw


def as_decimal(value: Any) -> Decimal | None:
    """Convert a source value to Decimal while preserving missing values."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Non-numeric renewable value encountered: {text!r}") from exc


def source_preservation_tables(
    processed: pd.DataFrame, processed_text: pd.DataFrame, raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every processed SUN/WND value with its raw EIA value exactly."""
    summaries: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    processed_text_by_period = processed_text.set_index("period")

    for fuel, column in [("SUN", SOLAR), ("WND", WIND)]:
        fuel_raw = raw.loc[raw["fueltype"] == fuel]
        raw_lookup = dict(zip(fuel_raw["period"], fuel_raw["raw_value"]))
        comparable = 0
        exact_matches = 0
        equivalent_rows = 0
        processed_negative = 0
        negative_exact_matches = 0
        negative_missing_or_mismatch = 0

        for row in processed.itertuples(index=False):
            period = row.period.strftime("%Y-%m-%dT%H")
            processed_decimal = as_decimal(processed_text_by_period.at[period, column])
            raw_decimal = as_decimal(raw_lookup.get(period))
            if processed_decimal is None and raw_decimal is None:
                equivalent_rows += 1
            elif processed_decimal is not None and raw_decimal is not None:
                comparable += 1
                if processed_decimal == raw_decimal:
                    exact_matches += 1
                    equivalent_rows += 1
            if processed_decimal is not None and processed_decimal < 0:
                processed_negative += 1
                if raw_decimal is not None and processed_decimal == raw_decimal:
                    negative_exact_matches += 1
                else:
                    negative_missing_or_mismatch += 1

        summaries.append(
            {
                "fuel": fuel,
                "processed_column": column,
                "processed_rows": len(processed),
                "raw_source_rows": len(fuel_raw),
                "comparable_non_null_rows": comparable,
                "exact_non_null_matches": exact_matches,
                "null_or_exact_equivalent_rows": equivalent_rows,
                "mismatched_rows": len(processed) - equivalent_rows,
                "processed_negative_values": processed_negative,
                "negative_exact_raw_matches": negative_exact_matches,
                "negative_missing_or_mismatch": negative_missing_or_mismatch,
            }
        )

        if fuel == "SUN":
            negative = processed.loc[processed[column] < 0].sort_values(column)
            positions = [0, 1, 2, 3, 4]
            if len(negative) > 1:
                positions.extend(
                    round((len(negative) - 1) * quantile)
                    for quantile in [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
                )
            chosen = negative.iloc[sorted(set(positions))]
            for row in chosen.itertuples(index=False):
                period = row.period.strftime("%Y-%m-%dT%H")
                raw_value = raw_lookup.get(period)
                processed_value_text = processed_text_by_period.at[period, column]
                raw_decimal = as_decimal(raw_value)
                processed_decimal = as_decimal(processed_value_text)
                representative_rows.append(
                    {
                        "period_utc": period,
                        "raw_eia_value_as_stored": raw_value,
                        "processed_csv_value_as_stored": processed_value_text,
                        "raw_numeric_value_mwh": float(raw_decimal),
                        "processed_numeric_value_mwh": float(processed_decimal),
                        "exact_numeric_match": raw_decimal == processed_decimal,
                    }
                )

    return pd.DataFrame(summaries), pd.DataFrame(representative_rows)


def formula_checks(frame: pd.DataFrame) -> pd.DataFrame:
    """Confirm stored derived fields equal the documented arithmetic formulas."""
    renewable_complete = frame[[SOLAR, WIND, COMBINED]].notna().all(axis=1)
    all_complete = renewable_complete & frame[[DEMAND, RESIDUAL, SHARE]].notna().all(
        axis=1
    )
    expected = {
        COMBINED: (frame[SOLAR] + frame[WIND], renewable_complete),
        RESIDUAL: (frame[DEMAND] - frame[COMBINED], all_complete),
        SHARE: (frame[COMBINED] / frame[DEMAND] * 100, all_complete),
    }
    records: list[dict[str, Any]] = []
    for column, (calculated, valid) in expected.items():
        differences = (frame.loc[valid, column] - calculated.loc[valid]).abs()
        records.append(
            {
                "derived_column": column,
                "rows_checked": int(valid.sum()),
                "rows_with_difference_above_tolerance": int(
                    (differences > FORMULA_TOLERANCE).sum()
                ),
                "maximum_absolute_difference": float(differences.max()),
                "tolerance": FORMULA_TOLERANCE,
            }
        )
    return pd.DataFrame(records)


def negative_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize negative scope and useful negative-only quantiles."""
    records: list[dict[str, Any]] = []
    for metric in METRICS:
        complete = frame[metric].dropna()
        negative = complete.loc[complete < 0]
        record: dict[str, Any] = {
            "metric": metric,
            "complete_observations": len(complete),
            "negative_count": len(negative),
            "negative_pct_of_complete": len(negative) / len(complete) * 100,
            "minimum_negative_mwh": negative.min() if len(negative) else pd.NA,
            "maximum_negative_mwh": negative.max() if len(negative) else pd.NA,
            "mean_negative_mwh": negative.mean() if len(negative) else pd.NA,
            "median_negative_mwh": negative.median() if len(negative) else pd.NA,
        }
        for quantile in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
            label = f"q{int(quantile * 100):02d}_negative_mwh"
            record[label] = negative.quantile(quantile) if len(negative) else pd.NA
        records.append(record)
    return pd.DataFrame(records)


def grouped_negative_counts(
    frame: pd.DataFrame, group_column: str, ordered_values: list[Any] | None = None
) -> pd.DataFrame:
    """Count negative and complete values for each generation metric and group."""
    records: list[dict[str, Any]] = []
    values = ordered_values or sorted(frame[group_column].dropna().unique().tolist())
    for value in values:
        group = frame.loc[frame[group_column] == value]
        for metric in METRICS:
            complete = group[metric].notna()
            negative = complete & group[metric].lt(0)
            complete_count = int(complete.sum())
            records.append(
                {
                    group_column: value,
                    "metric": metric,
                    "complete_observations": complete_count,
                    "negative_count": int(negative.sum()),
                    "negative_pct_of_complete": (
                        float(negative.sum()) / complete_count * 100
                        if complete_count
                        else pd.NA
                    ),
                }
            )
    return pd.DataFrame(records)


def magnitude_bands(frame: pd.DataFrame) -> pd.DataFrame:
    """Count negative solar values in documented non-overlapping bands."""
    solar = frame.loc[frame[SOLAR] < 0, SOLAR]
    definitions = [
        ("-1 <= x < 0", solar.ge(-1) & solar.lt(0)),
        ("-5 <= x < -1", solar.ge(-5) & solar.lt(-1)),
        ("-10 <= x < -5", solar.ge(-10) & solar.lt(-5)),
        ("-25 <= x < -10", solar.ge(-25) & solar.lt(-10)),
        ("-50 <= x < -25", solar.ge(-50) & solar.lt(-25)),
        ("x < -50", solar.lt(-50)),
    ]
    return pd.DataFrame(
        [
            {
                "band_definition_mwh": label,
                "negative_solar_count": int(mask.sum()),
                "pct_of_all_negative_solar": float(mask.sum()) / len(solar) * 100,
            }
            for label, mask in definitions
        ]
    )


def solar_hour_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Build an empirical UTC-hour profile used for the low-output proxy."""
    profile = (
        frame.groupby("utc_hour")[SOLAR]
        .agg(
            complete_observations="count",
            median_solar_mwh="median",
            mean_solar_mwh="mean",
            p05_solar_mwh=lambda values: values.quantile(0.05),
            p25_solar_mwh=lambda values: values.quantile(0.25),
        )
        .reset_index()
    )
    profile["empirical_low_or_no_output_proxy"] = profile[
        "median_solar_mwh"
    ].le(LOW_OUTPUT_MEDIAN_THRESHOLD_MWH)
    return profile


def temporal_pattern_summary(
    frame: pd.DataFrame, hour_profile: pd.DataFrame
) -> pd.DataFrame:
    """Measure concentration in empirically low-output versus active UTC hours."""
    low_hours = set(
        hour_profile.loc[
            hour_profile["empirical_low_or_no_output_proxy"], "utc_hour"
        ]
    )
    negative = frame.loc[frame[SOLAR] < 0]
    in_low = negative["utc_hour"].isin(low_hours)
    active_negative = negative.loc[~in_low]
    substantial_active = active_negative.loc[
        active_negative[SOLAR] < SUBSTANTIAL_NEGATIVE_THRESHOLD_MWH
    ]
    records = [
        {
            "measure": "empirical_low_or_no_output_utc_hours",
            "value": ", ".join(f"{hour:02d}:00" for hour in sorted(low_hours)),
            "definition": (
                f"UTC hours with median solar <= {LOW_OUTPUT_MEDIAN_THRESHOLD_MWH:g} MWh"
            ),
        },
        {
            "measure": "negative_solar_in_low_or_no_output_proxy_count",
            "value": int(in_low.sum()),
            "definition": "Negative solar rows within the empirical proxy hours",
        },
        {
            "measure": "negative_solar_in_low_or_no_output_proxy_pct",
            "value": float(in_low.mean() * 100),
            "definition": "Percent of all negative solar rows within proxy hours",
        },
        {
            "measure": "negative_solar_in_empirically_active_hours_count",
            "value": len(active_negative),
            "definition": (
                f"Negative solar rows at UTC hours with median solar > "
                f"{LOW_OUTPUT_MEDIAN_THRESHOLD_MWH:g} MWh"
            ),
        },
        {
            "measure": "substantial_negative_in_empirically_active_hours_count",
            "value": len(substantial_active),
            "definition": (
                f"Solar < {SUBSTANTIAL_NEGATIVE_THRESHOLD_MWH:g} MWh at UTC hours "
                f"with median solar > {LOW_OUTPUT_MEDIAN_THRESHOLD_MWH:g} MWh"
            ),
        },
        {
            "measure": "most_negative_value_in_empirically_active_hours_mwh",
            "value": active_negative[SOLAR].min() if len(active_negative) else pd.NA,
            "definition": "Minimum solar among empirically active UTC hours",
        },
        {
            "measure": "substantial_active_utc_hours",
            "value": ", ".join(
                f"{hour:02d}:00" for hour in sorted(substantial_active["utc_hour"].unique())
            )
            or "none",
            "definition": "UTC hours containing substantial active-hour negatives",
        },
    ]
    return pd.DataFrame(records)


def negative_sequences(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find uninterrupted hourly runs of negative solar observations."""
    negative = frame[SOLAR].lt(0)
    starts = negative & (
        ~negative.shift(fill_value=False)
        | frame["period"].diff().ne(pd.Timedelta(hours=1))
    )
    sequence_id = starts.cumsum()
    rows: list[dict[str, Any]] = []
    for _, group in frame.loc[negative].groupby(sequence_id.loc[negative]):
        rows.append(
            {
                "start_period_utc": group["period"].min().strftime("%Y-%m-%dT%H"),
                "end_period_utc": group["period"].max().strftime("%Y-%m-%dT%H"),
                "consecutive_hours": len(group),
                "minimum_solar_mwh": group[SOLAR].min(),
                "mean_solar_mwh": group[SOLAR].mean(),
                "median_solar_mwh": group[SOLAR].median(),
            }
        )
    sequences = pd.DataFrame(rows).sort_values(
        ["consecutive_hours", "minimum_solar_mwh"], ascending=[False, True]
    )
    lengths = sequences["consecutive_hours"]
    summary = pd.DataFrame(
        [
            {
                "sequence_count": len(sequences),
                "single_hour_sequence_count": int(lengths.eq(1).sum()),
                "sequences_at_least_6_hours": int(lengths.ge(6).sum()),
                "sequences_at_least_12_hours": int(lengths.ge(12).sum()),
                "sequences_at_least_24_hours": int(lengths.ge(24).sum()),
                "median_sequence_hours": lengths.median(),
                "p90_sequence_hours": lengths.quantile(0.90),
                "p95_sequence_hours": lengths.quantile(0.95),
                "p99_sequence_hours": lengths.quantile(0.99),
                "longest_sequence_hours": int(lengths.max()),
                "longest_sequence_start_utc": sequences.iloc[0]["start_period_utc"],
                "longest_sequence_end_utc": sequences.iloc[0]["end_period_utc"],
            }
        ]
    )
    return summary, sequences.head(25).reset_index(drop=True)


def representative_records(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create concise record-level tables requested for manual review."""
    columns = ["period", DEMAND, SOLAR, WIND, COMBINED, RESIDUAL, SHARE]
    most_negative = frame.nsmallest(25, SOLAR)[columns].copy()

    representatives: list[pd.Series] = []
    for _, group in frame.loc[frame[SOLAR] < 0].groupby("utc_hour"):
        median = group[SOLAR].median()
        index = (group[SOLAR] - median).abs().idxmin()
        representatives.append(frame.loc[index, columns + ["utc_hour"]])
    by_hour = pd.DataFrame(representatives).sort_values("utc_hour")

    monthly = (
        frame.assign(negative_solar=frame[SOLAR].lt(0))
        .groupby("year_month_utc", as_index=False)
        .agg(
            complete_solar_observations=(SOLAR, "count"),
            negative_solar_count=("negative_solar", "sum"),
            minimum_solar_mwh=(SOLAR, "min"),
        )
    )
    monthly["negative_pct_of_complete"] = (
        monthly["negative_solar_count"]
        / monthly["complete_solar_observations"]
        * 100
    )

    hourly = (
        frame.assign(negative_solar=frame[SOLAR].lt(0))
        .groupby("utc_hour", as_index=False)
        .agg(
            complete_solar_observations=(SOLAR, "count"),
            negative_solar_count=("negative_solar", "sum"),
            minimum_solar_mwh=(SOLAR, "min"),
            median_solar_mwh=(SOLAR, "median"),
        )
    )
    hourly["negative_pct_of_complete"] = (
        hourly["negative_solar_count"] / hourly["complete_solar_observations"] * 100
    )

    negative_wind = frame.loc[frame[WIND] < 0, columns].copy()
    negative_combined = frame.loc[frame[COMBINED] < 0, columns].copy()
    for table in [most_negative, by_hour, negative_wind, negative_combined]:
        if "period" in table:
            table["period"] = table["period"].dt.strftime("%Y-%m-%dT%H")
            table.rename(columns={"period": "period_utc"}, inplace=True)
    return {
        "most_negative_solar_25": most_negative,
        "representative_negative_solar_by_utc_hour": by_hour,
        "monthly_negative_solar_counts": monthly,
        "hourly_negative_solar_counts": hourly,
        "negative_wind_observations": negative_wind,
        "negative_combined_generation_observations": negative_combined,
    }


def derived_metric_effects(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quantify observed anomalies and counterfactual clip-to-zero differences."""
    complete = frame[[DEMAND, SOLAR, WIND, COMBINED, RESIDUAL, SHARE]].notna().all(
        axis=1
    )
    conditions = {
        "combined_solar_wind_generation_is_negative": complete & frame[COMBINED].lt(0),
        "solar_wind_share_is_negative": complete & frame[SHARE].lt(0),
        "residual_demand_exceeds_demand": complete & frame[RESIDUAL].gt(frame[DEMAND]),
    }
    magnitudes = {
        "combined_solar_wind_generation_is_negative": -frame[COMBINED],
        "solar_wind_share_is_negative": -frame[SHARE],
        "residual_demand_exceeds_demand": frame[RESIDUAL] - frame[DEMAND],
    }
    units = {
        "combined_solar_wind_generation_is_negative": "MWh below zero",
        "solar_wind_share_is_negative": "percentage points below zero",
        "residual_demand_exceeds_demand": "MWh above demand",
    }
    records: list[dict[str, Any]] = []
    for condition, mask in conditions.items():
        values = magnitudes[condition].loc[mask]
        records.append(
            {
                "condition": condition,
                "affected_rows": int(mask.sum()),
                "pct_of_complete_rows": float(mask.sum()) / int(complete.sum()) * 100,
                "magnitude_unit": units[condition],
                "minimum_magnitude": values.min() if len(values) else pd.NA,
                "maximum_magnitude": values.max() if len(values) else pd.NA,
                "mean_magnitude": values.mean() if len(values) else pd.NA,
                "median_magnitude": values.median() if len(values) else pd.NA,
            }
        )
    condition_summary = pd.DataFrame(records)

    affected = complete & pd.concat(conditions, axis=1).any(axis=1)
    observation_columns = ["period", DEMAND, SOLAR, WIND, COMBINED, RESIDUAL, SHARE]
    observations = frame.loc[affected, observation_columns].copy()
    observations["combined_below_zero_mwh"] = -observations[COMBINED]
    observations["residual_above_demand_mwh"] = observations[RESIDUAL] - observations[DEMAND]
    observations["share_below_zero_percentage_points"] = -observations[SHARE]
    observations["period"] = observations["period"].dt.strftime("%Y-%m-%dT%H")
    observations.rename(columns={"period": "period_utc"}, inplace=True)

    negative_solar = complete & frame[SOLAR].lt(0)
    solar_magnitude = -frame.loc[negative_solar, SOLAR]
    share_reduction = (
        -frame.loc[negative_solar, SOLAR] / frame.loc[negative_solar, DEMAND] * 100
    )
    effect_definitions = [
        (
            COMBINED,
            "reported combined generation is lower than a clip-to-zero counterfactual",
            "MWh",
            solar_magnitude,
        ),
        (
            RESIDUAL,
            "reported residual demand is higher than a clip-to-zero counterfactual",
            "MWh",
            solar_magnitude,
        ),
        (
            SHARE,
            "reported renewable share is lower than a clip-to-zero counterfactual",
            "percentage points",
            share_reduction,
        ),
    ]
    effect_rows: list[dict[str, Any]] = []
    for metric, direction, unit, values in effect_definitions:
        effect_rows.append(
            {
                "derived_metric": metric,
                "counterfactual_interpretation_only": direction,
                "affected_negative_solar_rows": len(values),
                "effect_unit": unit,
                "minimum_absolute_effect": values.min(),
                "maximum_absolute_effect": values.max(),
                "mean_absolute_effect": values.mean(),
                "median_absolute_effect": values.median(),
            }
        )
    counterfactual_summary = pd.DataFrame(effect_rows)
    return condition_summary, observations, counterfactual_summary


def save_table(table: pd.DataFrame, path: Path) -> None:
    """Save one CSV table with consistent numeric precision."""
    table.to_csv(path, index=False, float_format="%.10g")


def style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_figures(
    frame: pd.DataFrame,
    hourly_counts: pd.DataFrame,
    monthly_counts: pd.DataFrame,
    profile: pd.DataFrame,
    figure_dir: Path,
) -> list[str]:
    """Create all six required PNG figures and close them after saving."""
    filenames: list[str] = []
    negative = frame.loc[frame[SOLAR] < 0].copy()
    negative["negative_magnitude_mwh"] = -negative[SOLAR]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(negative["negative_magnitude_mwh"], bins=40, color="tab:red", alpha=0.8)
    style_axes(
        ax,
        "Distribution of Negative Solar Magnitudes",
        "Absolute magnitude (MWh)",
        "Observation count",
    )
    filenames.append("01_negative_solar_magnitude_distribution.png")
    save_figure(fig, figure_dir / filenames[-1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hourly_counts["utc_hour"], hourly_counts["negative_solar_count"])
    ax.set_xticks(range(24))
    style_axes(
        ax,
        "Negative Solar Observations by UTC Hour",
        "UTC hour",
        "Negative observation count",
    )
    filenames.append("02_negative_solar_count_by_utc_hour.png")
    save_figure(fig, figure_dir / filenames[-1])

    month_aggregate = (
        monthly_counts.assign(
            month=monthly_counts["year_month_utc"].str[-2:].astype(int)
        )
        .groupby("month", as_index=False)["negative_solar_count"]
        .sum()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(month_aggregate["month"], month_aggregate["negative_solar_count"])
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(
        pd.date_range("2024-01-01", periods=12, freq="MS").month_name().str[:3]
    )
    style_axes(
        ax,
        "Negative Solar Observations by Calendar Month (UTC)",
        "Calendar month (UTC)",
        "Negative observation count",
    )
    filenames.append("03_negative_solar_count_by_month_utc.png")
    save_figure(fig, figure_dir / filenames[-1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        profile["utc_hour"],
        profile["median_solar_mwh"],
        marker="o",
        label="Median",
    )
    ax.plot(
        profile["utc_hour"],
        profile["p05_solar_mwh"],
        marker="o",
        label="5th percentile",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(24))
    style_axes(
        ax,
        "Solar Generation Profile by UTC Hour",
        "UTC hour",
        "Solar generation (MWh)",
    )
    ax.legend()
    filenames.append("04_solar_profile_by_utc_hour.png")
    save_figure(fig, figure_dir / filenames[-1])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.scatter(
        negative["period"],
        negative[SOLAR],
        s=5,
        alpha=0.45,
        color="tab:red",
        rasterized=True,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    style_axes(
        ax,
        "Timeline of Negative Solar Observations",
        "Date (UTC)",
        "Negative solar generation (MWh)",
    )
    filenames.append("05_negative_solar_timeline_utc.png")
    save_figure(fig, figure_dir / filenames[-1])

    years = sorted(negative["year"].unique())
    year_values = [
        negative.loc[negative["year"] == year, "negative_magnitude_mwh"]
        for year in years
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(year_values, tick_labels=[str(year) for year in years], showfliers=True)
    style_axes(
        ax,
        "Negative Solar Magnitude by Year",
        "Year (UTC)",
        "Absolute negative magnitude (MWh)",
    )
    filenames.append("06_negative_solar_magnitude_by_year.png")
    save_figure(fig, figure_dir / filenames[-1])
    return filenames


def fmt_number(value: Any, decimals: int = 2) -> str:
    if pd.isna(value):
        return "not applicable"
    return f"{float(value):,.{decimals}f}"


def write_report(
    path: Path,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    source_checks: pd.DataFrame,
    formula_check_table: pd.DataFrame,
    bands: pd.DataFrame,
    pattern: pd.DataFrame,
    sequence_summary: pd.DataFrame,
    effect_conditions: pd.DataFrame,
    counterfactual_effects: pd.DataFrame,
    figures: list[str],
    table_names: list[str],
) -> None:
    """Write a concise evidence-first diagnostic and policy recommendation."""
    solar = summary.loc[summary["metric"] == SOLAR].iloc[0]
    wind = summary.loc[summary["metric"] == WIND].iloc[0]
    combined = summary.loc[summary["metric"] == COMBINED].iloc[0]
    pattern_values = dict(zip(pattern["measure"], pattern["value"]))
    sequence = sequence_summary.iloc[0]
    source_solar = source_checks.loc[source_checks["fuel"] == "SUN"].iloc[0]
    source_wind = source_checks.loc[source_checks["fuel"] == "WND"].iloc[0]
    formula_failures = int(
        formula_check_table["rows_with_difference_above_tolerance"].sum()
    )
    effect_lookup = effect_conditions.set_index("condition")
    combined_effect = effect_lookup.loc[
        "combined_solar_wind_generation_is_negative"
    ]
    share_effect = effect_lookup.loc["solar_wind_share_is_negative"]
    residual_effect = effect_lookup.loc["residual_demand_exceeds_demand"]
    low_pct = float(pattern_values["negative_solar_in_low_or_no_output_proxy_pct"])
    mainly = "mainly" if low_pct > 50 else "not mainly"
    substantial_active = int(
        pattern_values["substantial_negative_in_empirically_active_hours_count"]
    )

    lines = [
        "# Negative Generation Diagnostic",
        "",
        "## Scope and safeguards",
        "",
        f"This local, read-only diagnostic covers **{len(frame):,} hourly rows** from "
        f"**{frame['period'].min():%Y-%m-%dT%H}** through "
        f"**{frame['period'].max():%Y-%m-%dT%H}**. All timestamps, hours, weekdays, "
        "months, and years in this report are **UTC**. No California local-clock "
        "conversion or interpretation was made.",
        "",
        "The master processed CSV and historical raw JSON were only read. No network "
        "request, data replacement, clipping, imputation, or model training was performed.",
        "",
        "## Evidence: source preservation",
        "",
        f"- Solar: **{int(source_solar['negative_exact_raw_matches']):,} of "
        f"{int(source_solar['processed_negative_values']):,}** processed negative values "
        "exactly equal the corresponding raw EIA numeric values. "
        f"All **{int(source_solar['null_or_exact_equivalent_rows']):,}** processed rows "
        "are either exact numeric matches or equivalent missing observations; "
        f"mismatches: **{int(source_solar['mismatched_rows'])}**.",
        f"- Wind: all **{int(source_wind['null_or_exact_equivalent_rows']):,}** processed "
        f"rows are exact numeric matches or equivalent missing observations; mismatches: "
        f"**{int(source_wind['mismatched_rows'])}**.",
        f"- The three stored derived columns match their documented arithmetic formulas "
        f"within {FORMULA_TOLERANCE:g}; rows above tolerance across all checks: "
        f"**{formula_failures}**.",
        "- Therefore, the local evidence shows that the negative renewable values were "
        "preserved from the raw EIA response rather than introduced by the historical "
        "processing arithmetic.",
        "",
        "Representative exact raw/processed comparisons are in "
        "`tables/source_preservation_representative_negative_solar.csv`. No request "
        "metadata is included in any output.",
        "",
        "## Evidence: scope and magnitude",
        "",
        f"- Solar has **{int(solar['negative_count']):,} negative values** "
        f"(**{solar['negative_pct_of_complete']:.2f}%** of "
        f"{int(solar['complete_observations']):,} complete observations), ranging from "
        f"**{fmt_number(solar['minimum_negative_mwh'])} MWh** to "
        f"**{fmt_number(solar['maximum_negative_mwh'])} MWh**. The negative-only mean "
        f"is **{fmt_number(solar['mean_negative_mwh'])} MWh** and median is "
        f"**{fmt_number(solar['median_negative_mwh'])} MWh**.",
        f"- Wind has **{int(wind['negative_count']):,} negative values** "
        f"(**{wind['negative_pct_of_complete']:.4f}%** of complete observations).",
        f"- Combined solar-plus-wind generation has **{int(combined['negative_count']):,} "
        f"negative values** (**{combined['negative_pct_of_complete']:.4f}%** of complete "
        "observations).",
        "- Quantiles and UTC counts by year, month, hour, and weekday are in the tables. "
        "Magnitude bands are non-overlapping: `-1 <= x < 0`, `-5 <= x < -1`, "
        "`-10 <= x < -5`, `-25 <= x < -10`, `-50 <= x < -25`, and `x < -50` MWh.",
        "",
        "## Evidence: temporal pattern",
        "",
        f"- The empirical low/no-output proxy is defined as UTC hours whose full-sample "
        f"median solar is at most **{LOW_OUTPUT_MEDIAN_THRESHOLD_MWH:g} MWh**. These are: "
        f"**{pattern_values['empirical_low_or_no_output_utc_hours']} UTC**.",
        f"- **{int(pattern_values['negative_solar_in_low_or_no_output_proxy_count']):,}** "
        f"negative solar rows (**{low_pct:.2f}%**) fall in those hours. On that explicit "
        f"empirical definition, the negatives occur **{mainly}** during low/no-output "
        "profile hours.",
        f"- **{int(pattern_values['negative_solar_in_empirically_active_hours_count']):,}** "
        "negative rows occur at UTC hours whose median solar is above the proxy threshold. "
        f"Among them, **{substantial_active:,}** are below "
        f"**{SUBSTANTIAL_NEGATIVE_THRESHOLD_MWH:g} MWh**; the most negative is "
        f"**{fmt_number(pattern_values['most_negative_value_in_empirically_active_hours_mwh'])} "
        "MWh**. This establishes substantial negatives during empirically solar-active "
        "UTC hours, but it does not label those hours as California local daytime.",
        f"- Negative solar frequently forms multi-hour runs: it has "
        f"**{int(sequence['sequence_count']):,}** sequences in total. The "
        f"longest lasts **{int(sequence['longest_sequence_hours'])} consecutive hours**, "
        f"from **{sequence['longest_sequence_start_utc']}** through "
        f"**{sequence['longest_sequence_end_utc']} UTC**. There are "
        f"**{int(sequence['sequences_at_least_12_hours']):,}** sequences lasting at least "
        "12 hours and **"
        f"{int(sequence['sequences_at_least_24_hours']):,}** lasting at least 24 hours.",
        "",
        "## Evidence: effects on derived metrics",
        "",
        f"- Rows where combined solar-plus-wind generation is negative: "
        f"**{int(combined_effect['affected_rows']):,}**. Maximum magnitude below zero: "
        f"**{fmt_number(combined_effect['maximum_magnitude'])}** (MWh below zero; "
        "not applicable when there are no affected rows).",
        f"- Rows where solar/wind share is negative: "
        f"**{int(share_effect['affected_rows']):,}**. Maximum magnitude below zero: "
        f"**{fmt_number(share_effect['maximum_magnitude'], 6)}** (percentage points "
        "below zero; not applicable when there are no affected rows).",
        f"- Rows where residual demand exceeds demand because combined generation is "
        f"negative: **{int(residual_effect['affected_rows']):,}**. Maximum excess: "
        f"**{fmt_number(residual_effect['maximum_magnitude'])}** (MWh above demand; "
        "not applicable when there are no affected rows).",
        "- For every complete negative-solar row, a diagnostic clip-to-zero "
        "counterfactual would raise combined generation and lower residual demand by the "
        "absolute solar magnitude. It would also raise renewable share by "
        "`abs(solar) / demand * 100` percentage points. This counterfactual was calculated "
        "only to quantify effects; no stored value was changed.",
        "",
        "## Inference and unresolved cause",
        "",
        "The timing, observed magnitudes, sequences, and exact source matches are consistent "
        "with a source reporting or accounting convention, measurement adjustment, or "
        "other operational effect. **The physical or accounting cause is not proven by "
        "these local files.** EIA series documentation and/or confirmation from EIA/CAISO "
        "domain experts is still required before assigning a cause or treating the values "
        "as errors.",
        "",
        "## Recommended downstream policy (not applied)",
        "",
        "- **Demand forecasting:** preserve the master values. If solar is not required "
        "as a predictor, omit it rather than editing it. If used, add a negative-solar "
        "flag and compare a derived clip-to-zero feature in validation only.",
        "- **Renewable-aware analysis:** preserve reported values and add an explicit "
        "quality/adjustment flag. Show sensitivity results using a separate derived "
        "clip-to-zero series. Replacing with null is not recommended without proof that "
        "the observations are invalid because it discards source information.",
        "- **Residual-demand modelling:** do not silently use clipped values. Build "
        "versioned derived inputs containing the original value, a negative-value flag, "
        "and optionally a clipped-to-zero sensitivity feature. Choose between them using "
        "chronological train/validation evaluation, with no future information entering "
        "lag or rolling features.",
        "- **Visual dashboard summaries:** keep the reported series available, visibly "
        "flag negatives, and explain the convention as unresolved. A display-only floor "
        "at zero may be offered for simple summaries only if clearly labelled and never "
        "substituted for the underlying values.",
        "",
        "## Issue to resolve before feature engineering",
        "",
        "Decide and document whether renewable predictors and residual-demand targets use "
        "reported values, flagged reported values, or a separate clipped sensitivity "
        "version. Domain/source confirmation should be sought before declaring the values "
        "invalid. Any eventual modelling comparison must use chronological splits and "
        "leakage-safe lag/rolling construction.",
        "",
        "## Generated artifacts",
        "",
        "Tables:",
        "",
        *[f"- `tables/{name}`" for name in table_names],
        "",
        "Figures:",
        "",
        *[f"- `figures/{name}`" for name in figures],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_diagnostic(processed_path: Path, raw_path: Path, output_dir: Path) -> None:
    """Run the complete local diagnostic and write reproducible artifacts."""
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    frame, processed_text = load_processed(processed_path)
    raw = load_raw_renewable(raw_path)

    source_checks, representative_matches = source_preservation_tables(
        frame, processed_text, raw
    )
    formulas = formula_checks(frame)
    summary = negative_summary(frame)
    by_year = grouped_negative_counts(frame, "year")
    by_month = grouped_negative_counts(frame, "month", list(range(1, 13)))
    by_hour = grouped_negative_counts(frame, "utc_hour", list(range(24)))
    by_day = grouped_negative_counts(frame, "day_of_week_utc", DAY_ORDER)
    bands = magnitude_bands(frame)
    profile = solar_hour_profile(frame)
    pattern = temporal_pattern_summary(frame, profile)
    sequence_summary, longest_sequences = negative_sequences(frame)
    records = representative_records(frame)
    condition_effects, effect_observations, counterfactual_effects = (
        derived_metric_effects(frame)
    )

    tables: dict[str, pd.DataFrame] = {
        "source_preservation_checks.csv": source_checks,
        "source_preservation_representative_negative_solar.csv": representative_matches,
        "derived_formula_checks.csv": formulas,
        "generation_negative_summary.csv": summary,
        "negative_counts_by_year_utc.csv": by_year,
        "negative_counts_by_calendar_month_utc.csv": by_month,
        "negative_counts_by_utc_hour.csv": by_hour,
        "negative_counts_by_day_of_week_utc.csv": by_day,
        "negative_solar_magnitude_bands.csv": bands,
        "solar_profile_by_utc_hour.csv": profile,
        "negative_solar_temporal_pattern_summary.csv": pattern,
        "negative_solar_sequence_summary.csv": sequence_summary,
        "longest_negative_solar_sequences_25.csv": longest_sequences,
        "derived_metric_anomaly_summary.csv": condition_effects,
        "derived_metric_anomaly_observations.csv": effect_observations,
        "negative_solar_counterfactual_effect_summary.csv": counterfactual_effects,
    }
    for name, table in records.items():
        tables[f"{name}.csv"] = table
    for name, table in tables.items():
        save_table(table, table_dir / name)

    figures = build_figures(
        frame,
        records["hourly_negative_solar_counts"],
        records["monthly_negative_solar_counts"],
        profile,
        figure_dir,
    )
    write_report(
        output_dir / "negative_generation_diagnostic.md",
        frame,
        summary,
        source_checks,
        formulas,
        bands,
        pattern,
        sequence_summary,
        condition_effects,
        counterfactual_effects,
        figures,
        list(tables),
    )

    solar_count = int((frame[SOLAR] < 0).sum())
    wind_count = int((frame[WIND] < 0).sum())
    combined_count = int((frame[COMBINED] < 0).sum())
    print(f"PASS: wrote diagnostic report to {output_dir}")
    print(
        f"Negative counts - solar: {solar_count}, wind: {wind_count}, "
        f"combined: {combined_count}"
    )
    print(f"Tables written: {len(tables)}; figures written: {len(figures)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose preserved negative EIA solar/wind values locally."
    )
    parser.add_argument("--processed-path", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--raw-renewable-path", type=Path, default=DEFAULT_RAW_RENEWABLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        run_diagnostic(args.processed_path, args.raw_renewable_path, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
