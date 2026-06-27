"""Build leakage-safe hourly features for CISO demand forecasting.

The feature master keeps every canonical timestamp. It does not impute, clip,
or otherwise change source measurements. Predictor features are either known
calendar values or strictly shifted historical observations.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT / "data" / "processed" / "eia_ciso_hourly_2022_2024.csv"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "processed" / "eia_ciso_hourly_features.csv"
DEFAULT_RESULTS_DIR = ROOT / "results" / "features"

EXPECTED_ROWS = 26_304
ORIGINAL_COLUMNS = [
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
CALENDAR_FEATURES = [
    "year",
    "month",
    "day",
    "day_of_year",
    "hour_utc",
    "day_of_week_utc",
    "is_weekend_utc",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]
DEMAND_LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS = [24, 168]
ROLLING_STATISTICS = ["mean", "std", "min", "max"]
RENEWABLE_LAG_HOURS = [1, 24, 168]


def demand_lag_features() -> list[str]:
    return [f"demand_lag_{hours}h" for hours in DEMAND_LAG_HOURS]


def demand_rolling_features() -> list[str]:
    return [
        f"demand_rolling_{window}h_{stat}"
        for window in ROLLING_WINDOWS
        for stat in ROLLING_STATISTICS
    ]


def renewable_history_features() -> list[str]:
    features: list[str] = []
    for prefix in ["solar", "wind", "solar_wind"]:
        features.extend(
            f"{prefix}_lag_{hours}h" for hours in RENEWABLE_LAG_HOURS
        )
    features.extend(
        f"solar_negative_reported_lag_{hours}h"
        for hours in RENEWABLE_LAG_HOURS
    )
    return features


DEMAND_HISTORY_FEATURES = demand_lag_features() + demand_rolling_features()
RENEWABLE_HISTORY_FEATURES = renewable_history_features()
FEATURE_GROUPS = {
    "calendar_only": CALENDAR_FEATURES,
    "autoregressive_demand": CALENDAR_FEATURES + DEMAND_HISTORY_FEATURES,
    "renewable_history_enhanced": (
        CALENDAR_FEATURES + DEMAND_HISTORY_FEATURES + RENEWABLE_HISTORY_FEATURES
    ),
}


def load_canonical(path: Path) -> pd.DataFrame:
    """Load the canonical CSV and verify its unchanged hourly timeline."""
    source = pd.read_csv(path)
    if source.columns.tolist() != ORIGINAL_COLUMNS:
        raise ValueError(
            "Canonical columns differ from the expected schema. "
            f"Observed: {source.columns.tolist()}"
        )
    if len(source) != EXPECTED_ROWS:
        raise ValueError(f"Canonical CSV has {len(source):,} rows; expected {EXPECTED_ROWS:,}.")

    parsed_period = pd.to_datetime(source["period"], format="%Y-%m-%dT%H", errors="raise")
    if parsed_period.duplicated().any():
        raise ValueError("Canonical CSV contains duplicate timestamps.")
    if not parsed_period.is_monotonic_increasing:
        raise ValueError("Canonical timestamps are not in chronological order.")
    expected = pd.date_range(parsed_period.iloc[0], parsed_period.iloc[-1], freq="h")
    if len(expected) != len(source) or not parsed_period.equals(pd.Series(expected)):
        raise ValueError("Canonical timestamps are not a continuous hourly timeline.")

    source["period"] = parsed_period
    return source


def add_calendar_features(features: pd.DataFrame) -> None:
    """Add deterministic UTC-labelled calendar and cyclical features."""
    period = features["period"]
    features["year"] = period.dt.year
    features["month"] = period.dt.month
    features["day"] = period.dt.day
    features["day_of_year"] = period.dt.dayofyear
    features["hour_utc"] = period.dt.hour
    features["day_of_week_utc"] = period.dt.dayofweek
    features["is_weekend_utc"] = period.dt.dayofweek.ge(5)

    hour_angle = features["hour_utc"] * (2.0 * math.pi / 24.0)
    day_of_week_angle = features["day_of_week_utc"] * (2.0 * math.pi / 7.0)
    days_in_year = period.dt.is_leap_year.map({True: 366, False: 365})
    day_of_year_angle = (features["day_of_year"] - 1) * (2.0 * math.pi) / days_in_year

    features["hour_sin"] = hour_angle.map(math.sin)
    features["hour_cos"] = hour_angle.map(math.cos)
    features["day_of_week_sin"] = day_of_week_angle.map(math.sin)
    features["day_of_week_cos"] = day_of_week_angle.map(math.cos)
    features["day_of_year_sin"] = day_of_year_angle.map(math.sin)
    features["day_of_year_cos"] = day_of_year_angle.map(math.cos)


def add_demand_history_features(features: pd.DataFrame) -> None:
    """Add demand lags and full-window statistics using only earlier rows."""
    demand = features["demand_mwh"]
    for hours in DEMAND_LAG_HOURS:
        features[f"demand_lag_{hours}h"] = demand.shift(hours)

    past_demand = demand.shift(1)
    for window in ROLLING_WINDOWS:
        rolling = past_demand.rolling(window=window, min_periods=window)
        features[f"demand_rolling_{window}h_mean"] = rolling.mean()
        features[f"demand_rolling_{window}h_std"] = rolling.std(ddof=1)
        features[f"demand_rolling_{window}h_min"] = rolling.min()
        features[f"demand_rolling_{window}h_max"] = rolling.max()


def add_renewable_history_features(features: pd.DataFrame) -> None:
    """Add optional renewable lags without changing negative solar reports."""
    solar = features["solar_generation_mwh"]
    features["solar_negative_reported"] = (
        solar.lt(0).where(solar.notna()).astype("boolean")
    )
    source_map = {
        "solar": "solar_generation_mwh",
        "wind": "wind_generation_mwh",
        "solar_wind": "solar_wind_generation_mwh",
    }
    for prefix, source_column in source_map.items():
        for hours in RENEWABLE_LAG_HOURS:
            features[f"{prefix}_lag_{hours}h"] = features[source_column].shift(hours)
    for hours in RENEWABLE_LAG_HOURS:
        features[f"solar_negative_reported_lag_{hours}h"] = features[
            "solar_negative_reported"
        ].shift(hours)


def add_availability_flags(features: pd.DataFrame) -> None:
    """Describe feature completeness without filtering or filling rows."""
    features["demand_lags_complete"] = features[demand_lag_features()].notna().all(axis=1)
    for window in ROLLING_WINDOWS:
        columns = [
            f"demand_rolling_{window}h_{stat}" for stat in ROLLING_STATISTICS
        ]
        features[f"demand_rolling_{window}h_complete"] = features[columns].notna().all(axis=1)
    features["renewable_lags_complete"] = features[
        RENEWABLE_HISTORY_FEATURES
    ].notna().all(axis=1)


def build_features(source: pd.DataFrame) -> pd.DataFrame:
    """Return the complete feature master without modifying the input frame."""
    features = source.copy()
    numeric_target = pd.to_numeric(features["demand_mwh"], errors="coerce")
    features["target_demand_mwh"] = features["demand_mwh"]
    features["target_available"] = numeric_target.notna() & features["demand_data_complete"]
    add_calendar_features(features)
    add_demand_history_features(features)
    add_renewable_history_features(features)
    add_availability_flags(features)
    return features


def feature_group_table() -> pd.DataFrame:
    """Describe predictor membership and forecast-time safety."""
    rows: list[dict[str, object]] = []
    for feature in CALENDAR_FEATURES:
        rows.append(
            {
                "feature": feature,
                "group": "calendar_only",
                "source_column": "period",
                "lookback": "known_in_advance",
                "safe_at_forecast_time": True,
                "calendar_only": True,
                "autoregressive_demand": True,
                "renewable_history_enhanced": True,
            }
        )
    for hours in DEMAND_LAG_HOURS:
        rows.append(
            {
                "feature": f"demand_lag_{hours}h",
                "group": "autoregressive_demand",
                "source_column": "demand_mwh",
                "lookback": f"{hours}h",
                "safe_at_forecast_time": True,
                "calendar_only": False,
                "autoregressive_demand": True,
                "renewable_history_enhanced": True,
            }
        )
    for window in ROLLING_WINDOWS:
        for stat in ROLLING_STATISTICS:
            rows.append(
                {
                    "feature": f"demand_rolling_{window}h_{stat}",
                    "group": "autoregressive_demand",
                    "source_column": "demand_mwh",
                    "lookback": f"1h-{window}h",
                    "safe_at_forecast_time": True,
                    "calendar_only": False,
                    "autoregressive_demand": True,
                    "renewable_history_enhanced": True,
                }
            )
    renewable_sources = {
        "solar": "solar_generation_mwh",
        "wind": "wind_generation_mwh",
        "solar_wind": "solar_wind_generation_mwh",
    }
    for prefix, source_column in renewable_sources.items():
        for hours in RENEWABLE_LAG_HOURS:
            rows.append(
                {
                    "feature": f"{prefix}_lag_{hours}h",
                    "group": "renewable_history_enhanced",
                    "source_column": source_column,
                    "lookback": f"{hours}h",
                    "safe_at_forecast_time": True,
                    "calendar_only": False,
                    "autoregressive_demand": False,
                    "renewable_history_enhanced": True,
                }
            )
    for hours in RENEWABLE_LAG_HOURS:
        rows.append(
            {
                "feature": f"solar_negative_reported_lag_{hours}h",
                "group": "renewable_history_enhanced",
                "source_column": "solar_negative_reported",
                "lookback": f"{hours}h",
                "safe_at_forecast_time": True,
                "calendar_only": False,
                "autoregressive_demand": False,
                "renewable_history_enhanced": True,
            }
        )
    rows.append(
        {
            "feature": "solar_negative_reported",
            "group": "quality_only_not_predictor",
            "source_column": "solar_generation_mwh",
            "lookback": "0h",
            "safe_at_forecast_time": False,
            "calendar_only": False,
            "autoregressive_demand": False,
            "renewable_history_enhanced": False,
        }
    )
    return pd.DataFrame(rows)


def eligibility_table(features: pd.DataFrame) -> pd.DataFrame:
    """Count complete predictors and model-eligible targets without filtering."""
    rows: list[dict[str, object]] = []
    for group, columns in FEATURE_GROUPS.items():
        predictors_complete = features[columns].notna().all(axis=1)
        model_eligible = predictors_complete & features["target_available"]
        first_complete = features.loc[predictors_complete, "period"].min()
        first_eligible = features.loc[model_eligible, "period"].min()
        rows.append(
            {
                "feature_group": group,
                "total_rows": len(features),
                "predictors_complete_rows": int(predictors_complete.sum()),
                "eligible_target_and_predictor_rows": int(model_eligible.sum()),
                "first_predictors_complete_timestamp": first_complete,
                "first_eligible_timestamp": first_eligible,
            }
        )
    return pd.DataFrame(rows)


def engineered_columns(features: pd.DataFrame) -> list[str]:
    return [column for column in features.columns if column not in ORIGINAL_COLUMNS]


def write_feature_csv(features: pd.DataFrame, source_path: Path, output_path: Path) -> None:
    """Write features while preserving canonical source-field text exactly."""
    output = features.copy()
    canonical_text = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    if canonical_text.columns.tolist() != ORIGINAL_COLUMNS or len(canonical_text) != len(output):
        raise ValueError("Canonical text reload differs from the validated source schema.")
    for column in ORIGINAL_COLUMNS:
        output[column] = canonical_text[column]
    output.to_csv(output_path, index=False)


def write_tables_and_summary(features: pd.DataFrame, results_dir: Path) -> None:
    """Write auditable metadata and actual feature counts."""
    table_dir = results_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    groups = feature_group_table()
    eligibility = eligibility_table(features)
    engineered = engineered_columns(features)
    null_counts = pd.DataFrame(
        {
            "feature": engineered,
            "null_count": [int(features[column].isna().sum()) for column in engineered],
            "available_count": [int(features[column].notna().sum()) for column in engineered],
        }
    )
    negative_counts = pd.DataFrame(
        [
            {
                "column": column,
                "negative_value_count": int(features[column].lt(0).sum()),
                "minimum_value_mwh": features[column].min(),
            }
            for column in [
                "solar_generation_mwh",
                *[f"solar_lag_{hours}h" for hours in RENEWABLE_LAG_HOURS],
            ]
        ]
    )
    availability_flags = pd.DataFrame(
        [
            {
                "flag": flag,
                "complete_rows": int(features[flag].sum()),
                "incomplete_rows": int((~features[flag]).sum()),
            }
            for flag in [
                "target_available",
                "demand_data_complete",
                "renewable_data_complete",
                "demand_lags_complete",
                "demand_rolling_24h_complete",
                "demand_rolling_168h_complete",
                "renewable_lags_complete",
            ]
        ]
    )

    groups.to_csv(table_dir / "feature_groups.csv", index=False)
    eligibility.to_csv(
        table_dir / "feature_group_eligibility.csv", index=False, date_format="%Y-%m-%dT%H"
    )
    null_counts.to_csv(table_dir / "engineered_feature_null_counts.csv", index=False)
    negative_counts.to_csv(table_dir / "negative_solar_preservation.csv", index=False)
    availability_flags.to_csv(table_dir / "feature_availability_counts.csv", index=False)

    eligibility_lines = "\n".join(
        f"- `{row.feature_group}`: {row.eligible_target_and_predictor_rows:,} rows "
        f"({row.predictors_complete_rows:,} rows have complete predictors); first "
        f"complete at `{row.first_predictors_complete_timestamp:%Y-%m-%dT%H}`."
        for row in eligibility.itertuples(index=False)
    )
    null_lines = "\n".join(
        f"- `{row.feature}`: {row.null_count:,}"
        for row in null_counts.itertuples(index=False)
    )
    negative_lines = "\n".join(
        f"- `{row.column}`: {row.negative_value_count:,} negative values; minimum "
        f"{row.minimum_value_mwh:,.0f} MWh."
        for row in negative_counts.itertuples(index=False)
    )
    summary = f"""# Leakage-Safe Feature Summary

## Dataset Size And Target Availability

- Total rows: {len(features):,}
- Total columns: {len(features.columns):,}
- Original source columns retained: {len(ORIGINAL_COLUMNS)}
- Engineered columns added: {len(engineered)}
- Target-available rows: {int(features['target_available'].sum()):,}
- Target-unavailable rows retained: {int((~features['target_available']).sum()):,}

The eligibility counts below are descriptive counts calculated before any split.
They do not filter the feature master. For modelling, chronological splits must be
defined first; target and selected-feature availability filtering happens within
those already-defined periods.

## Feature-Group Eligibility

{eligibility_lines}

## Null Counts For Every Engineered Column

{null_lines}

## Negative Solar Preservation

{negative_lines}

Negative source values are unchanged. No clipped solar feature exists.

## Validation Notes

The builder's schema, continuous-timeline, and source-preservation preconditions
passed. No unexpected builder validation result was observed. Independent
leakage validation is recorded in `tables/validation_results.csv` when
`src/validate_features.py` is run.
"""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "feature_summary.md").write_text(summary, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build leakage-safe CISO hourly features.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    try:
        source = load_canonical(args.input_path)
        features = build_features(source)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        write_feature_csv(features, args.input_path, args.output_path)
        write_tables_and_summary(features, args.results_dir)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"Feature build: FAIL - {exc}")
        return 1

    eligibility = eligibility_table(features)
    print(f"Feature build: PASS - wrote {len(features):,} rows and {len(features.columns)} columns")
    print(f"Target-available rows: {int(features['target_available'].sum()):,}")
    for row in eligibility.itertuples(index=False):
        print(
            f"{row.feature_group}: {row.eligible_target_and_predictor_rows:,} "
            "target-and-predictor eligible rows"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
