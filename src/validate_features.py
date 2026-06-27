"""Independently validate leakage-safe CISO hourly feature engineering."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import pandas as pd

from build_features import (
    CALENDAR_FEATURES,
    DEMAND_LAG_HOURS,
    EXPECTED_ROWS,
    FEATURE_GROUPS,
    ORIGINAL_COLUMNS,
    RENEWABLE_HISTORY_FEATURES,
    RENEWABLE_LAG_HOURS,
    ROLLING_STATISTICS,
    ROLLING_WINDOWS,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = ROOT / "data" / "processed" / "eia_ciso_hourly_2022_2024.csv"
DEFAULT_FEATURE_PATH = ROOT / "data" / "processed" / "eia_ciso_hourly_features.csv"
DEFAULT_GROUP_PATH = ROOT / "results" / "features" / "tables" / "feature_groups.csv"
DEFAULT_RESULTS_PATH = ROOT / "results" / "features" / "tables" / "validation_results.csv"
EXPECTED_SOURCE_SHA256 = "157dc6714f32db71f8b7d9aa4c82ac7f2fe7f48755d34db5e4da0bd4d686b511"

BOOLEAN_COLUMNS = [
    "demand_data_complete",
    "renewable_data_complete",
    "target_available",
    "is_weekend_utc",
    "solar_negative_reported",
    *[
        f"solar_negative_reported_lag_{hours}h"
        for hours in RENEWABLE_LAG_HOURS
    ],
    "demand_lags_complete",
    "demand_rolling_24h_complete",
    "demand_rolling_168h_complete",
    "renewable_lags_complete",
]


class ValidationReport:
    """Collect all validation outcomes instead of stopping at the first issue."""

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(self, check: str, passed: bool, details: str) -> None:
        self.rows.append(
            {"check": check, "status": "PASS" if passed else "FAIL", "details": details}
        )

    @property
    def passed(self) -> bool:
        return all(row["status"] == "PASS" for row in self.rows)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def series_match(
    actual: pd.Series,
    expected: pd.Series,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> tuple[bool, str]:
    """Compare a full column, including its null mask, and return useful detail."""
    try:
        pd.testing.assert_series_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            check_dtype=False,
            check_exact=rtol == 0.0 and atol == 0.0,
            rtol=rtol,
            atol=atol,
        )
    except AssertionError as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else "full-column mismatch"
        return False, first_line
    return True, f"all {len(actual):,} values and null positions match"


def load_inputs(source_path: Path, feature_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(source_path)
    feature_dtype = {column: "boolean" for column in BOOLEAN_COLUMNS}
    features = pd.read_csv(feature_path, dtype=feature_dtype)
    source["period"] = pd.to_datetime(source["period"], format="%Y-%m-%dT%H", errors="raise")
    features["period"] = pd.to_datetime(
        features["period"], format="%Y-%m-%dT%H", errors="raise"
    )
    return source, features


def validate_structure(
    report: ValidationReport, source: pd.DataFrame, features: pd.DataFrame
) -> None:
    report.add(
        "row_count_preserved",
        len(source) == EXPECTED_ROWS and len(features) == EXPECTED_ROWS,
        f"source={len(source):,}; features={len(features):,}; expected={EXPECTED_ROWS:,}",
    )
    prefix_ok = features.columns[: len(ORIGINAL_COLUMNS)].tolist() == ORIGINAL_COLUMNS
    report.add(
        "original_column_order_preserved",
        prefix_ok,
        "the first feature-master columns exactly match the canonical schema",
    )
    source_columns_ok = True
    mismatch_details: list[str] = []
    for column in ORIGINAL_COLUMNS:
        if column not in features:
            source_columns_ok = False
            mismatch_details.append(f"missing {column}")
            continue
        passed, detail = series_match(features[column], source[column])
        if not passed:
            source_columns_ok = False
            mismatch_details.append(f"{column}: {detail}")
    report.add(
        "original_source_values_unchanged",
        source_columns_ok,
        "; ".join(mismatch_details) if mismatch_details else "all original columns match exactly",
    )
    duplicate_count = int(features["period"].duplicated().sum())
    expected_periods = pd.Series(
        pd.date_range(features["period"].iloc[0], features["period"].iloc[-1], freq="h")
    )
    continuous, detail = series_match(features["period"], expected_periods)
    report.add(
        "unique_continuous_hourly_timestamps",
        duplicate_count == 0 and continuous,
        f"duplicate timestamps={duplicate_count}; {detail}",
    )


def validate_target(
    report: ValidationReport, source: pd.DataFrame, features: pd.DataFrame
) -> None:
    passed, detail = series_match(features["target_demand_mwh"], source["demand_mwh"])
    report.add("target_equals_reported_demand", passed, detail)
    expected_available = source["demand_mwh"].notna() & source["demand_data_complete"]
    passed, detail = series_match(features["target_available"], expected_available)
    report.add("target_available_formula", passed, detail)
    target_nulls = int(features["target_demand_mwh"].isna().sum())
    report.add(
        "five_unavailable_targets_retained",
        target_nulls == 5 and int((~features["target_available"]).sum()) == 5,
        f"null targets={target_nulls}; unavailable flags={int((~features['target_available']).sum())}",
    )


def expected_calendar(period: pd.Series) -> dict[str, pd.Series]:
    year = period.dt.year
    month = period.dt.month
    day = period.dt.day
    day_of_year = period.dt.dayofyear
    hour = period.dt.hour
    day_of_week = period.dt.dayofweek
    days_in_year = period.dt.is_leap_year.map({True: 366, False: 365})
    hour_angle = hour * (2.0 * math.pi / 24.0)
    day_of_week_angle = day_of_week * (2.0 * math.pi / 7.0)
    day_of_year_angle = (day_of_year - 1) * (2.0 * math.pi) / days_in_year
    return {
        "year": year,
        "month": month,
        "day": day,
        "day_of_year": day_of_year,
        "hour_utc": hour,
        "day_of_week_utc": day_of_week,
        "is_weekend_utc": day_of_week.ge(5),
        "hour_sin": hour_angle.map(math.sin),
        "hour_cos": hour_angle.map(math.cos),
        "day_of_week_sin": day_of_week_angle.map(math.sin),
        "day_of_week_cos": day_of_week_angle.map(math.cos),
        "day_of_year_sin": day_of_year_angle.map(math.sin),
        "day_of_year_cos": day_of_year_angle.map(math.cos),
    }


def validate_calendar(report: ValidationReport, features: pd.DataFrame) -> None:
    expected = expected_calendar(features["period"])
    for column in CALENDAR_FEATURES:
        tolerance = 1e-12 if column.endswith(("_sin", "_cos")) else 0.0
        passed, detail = series_match(
            features[column], expected[column], rtol=tolerance, atol=tolerance
        )
        report.add(f"calendar_formula_{column}", passed, detail)


def validate_demand_history(report: ValidationReport, features: pd.DataFrame) -> None:
    demand = features["demand_mwh"]
    all_lags_pass = True
    lag_details: list[str] = []
    for hours in DEMAND_LAG_HOURS:
        column = f"demand_lag_{hours}h"
        passed, detail = series_match(features[column], demand.shift(hours))
        all_lags_pass &= passed
        if not passed:
            lag_details.append(f"{column}: {detail}")
    report.add(
        "all_demand_lags_exact",
        all_lags_pass,
        "; ".join(lag_details) if lag_details else "every lag is an exact full-column shift",
    )

    past_demand = demand.shift(1)
    all_rolling_pass = True
    rolling_details: list[str] = []
    for window in ROLLING_WINDOWS:
        rolling = past_demand.rolling(window=window, min_periods=window)
        expected_map = {
            "mean": rolling.mean(),
            "std": rolling.std(ddof=1),
            "min": rolling.min(),
            "max": rolling.max(),
        }
        for stat in ROLLING_STATISTICS:
            column = f"demand_rolling_{window}h_{stat}"
            passed, detail = series_match(
                features[column], expected_map[stat], rtol=1e-12, atol=1e-12
            )
            all_rolling_pass &= passed
            if not passed:
                rolling_details.append(f"{column}: {detail}")
    report.add(
        "all_demand_rolling_features_past_only",
        all_rolling_pass,
        "; ".join(rolling_details)
        if rolling_details
        else "all windows exactly match demand.shift(1).rolling(full_window)",
    )

    initial_nulls_ok = all(
        features[f"demand_lag_{hours}h"].iloc[:hours].isna().all()
        for hours in DEMAND_LAG_HOURS
    ) and all(
        features[f"demand_rolling_{window}h_mean"].iloc[:window].isna().all()
        for window in ROLLING_WINDOWS
    )
    report.add(
        "natural_history_boundary_nulls_preserved",
        initial_nulls_ok,
        "initial lag and full-window rolling boundaries remain null",
    )


def validate_renewable_history(report: ValidationReport, features: pd.DataFrame) -> None:
    expected_negative = (
        features["solar_generation_mwh"]
        .lt(0)
        .where(features["solar_generation_mwh"].notna())
        .astype("boolean")
    )
    passed, detail = series_match(features["solar_negative_reported"], expected_negative)
    report.add("current_negative_solar_flag_exact", passed, detail)

    source_map = {
        "solar": "solar_generation_mwh",
        "wind": "wind_generation_mwh",
        "solar_wind": "solar_wind_generation_mwh",
    }
    all_lags_pass = True
    details: list[str] = []
    for prefix, source_column in source_map.items():
        for hours in RENEWABLE_LAG_HOURS:
            column = f"{prefix}_lag_{hours}h"
            passed, detail = series_match(
                features[column], features[source_column].shift(hours)
            )
            all_lags_pass &= passed
            if not passed:
                details.append(f"{column}: {detail}")
    for hours in RENEWABLE_LAG_HOURS:
        column = f"solar_negative_reported_lag_{hours}h"
        passed, detail = series_match(
            features[column], expected_negative.shift(hours)
        )
        all_lags_pass &= passed
        if not passed:
            details.append(f"{column}: {detail}")
    report.add(
        "all_renewable_lags_exact",
        all_lags_pass,
        "; ".join(details) if details else "all values and nullable flags are exact shifts",
    )

    negative_source_count = int(features["solar_generation_mwh"].lt(0).sum())
    negative_preserved = negative_source_count == 9_074
    for hours in RENEWABLE_LAG_HOURS:
        expected = features["solar_generation_mwh"].shift(hours)
        actual = features[f"solar_lag_{hours}h"]
        negative_preserved &= int(actual.lt(0).sum()) == int(expected.lt(0).sum())
        negative_preserved &= actual.min() == expected.min()
    report.add(
        "negative_solar_values_unclipped_and_preserved",
        negative_preserved,
        f"source negative count={negative_source_count:,}; source minimum={features['solar_generation_mwh'].min():,.0f}",
    )


def validate_availability_flags(report: ValidationReport, features: pd.DataFrame) -> None:
    lag_columns = [f"demand_lag_{hours}h" for hours in DEMAND_LAG_HOURS]
    expected_flags = {
        "demand_lags_complete": features[lag_columns].notna().all(axis=1),
        "renewable_lags_complete": features[RENEWABLE_HISTORY_FEATURES].notna().all(axis=1),
    }
    for window in ROLLING_WINDOWS:
        columns = [f"demand_rolling_{window}h_{stat}" for stat in ROLLING_STATISTICS]
        expected_flags[f"demand_rolling_{window}h_complete"] = (
            features[columns].notna().all(axis=1)
        )
    all_pass = True
    details: list[str] = []
    for column, expected in expected_flags.items():
        passed, detail = series_match(features[column], expected)
        all_pass &= passed
        if not passed:
            details.append(f"{column}: {detail}")
    report.add(
        "availability_flags_report_completeness_only",
        all_pass,
        "; ".join(details) if details else "all flags exactly match feature null masks",
    )


def validate_declared_groups(
    report: ValidationReport, features: pd.DataFrame, group_path: Path
) -> None:
    groups = pd.read_csv(group_path)
    required = {
        "feature",
        "group",
        "source_column",
        "lookback",
        "safe_at_forecast_time",
        *FEATURE_GROUPS,
    }
    schema_ok = required.issubset(groups.columns) and not groups["feature"].duplicated().any()
    report.add(
        "feature_group_table_schema",
        schema_ok,
        f"rows={len(groups)}; duplicate features={int(groups['feature'].duplicated().sum()) if 'feature' in groups else 'unknown'}",
    )
    if not schema_ok:
        return

    declared_ok = True
    for group, expected_features in FEATURE_GROUPS.items():
        declared = groups.loc[groups[group].astype(bool), "feature"].tolist()
        declared_ok &= declared == expected_features
    report.add(
        "declared_predictor_groups_exact",
        declared_ok,
        "calendar, autoregressive, and renewable-enhanced memberships match code constants",
    )
    predictor_mask = groups[list(FEATURE_GROUPS)].any(axis=1)
    safe_values = groups["safe_at_forecast_time"].astype(bool)
    all_declared_safe = bool(safe_values.loc[predictor_mask].all())
    report.add(
        "all_declared_predictors_safe_at_forecast_time",
        all_declared_safe,
        f"declared predictors={int(predictor_mask.sum())}; unsafe declared predictors={int((~safe_values.loc[predictor_mask]).sum())}",
    )

    predictor_names = set(groups.loc[predictor_mask, "feature"])
    contemporaneous_demand = {"demand_mwh", "target_demand_mwh"}
    contemporaneous_renewable = {
        "solar_generation_mwh",
        "wind_generation_mwh",
        "solar_wind_generation_mwh",
        "residual_demand_after_solar_wind_mwh",
        "solar_wind_share_pct",
        "solar_negative_reported",
    }
    demand_safe = predictor_names.isdisjoint(contemporaneous_demand)
    renewable_safe = predictor_names.isdisjoint(contemporaneous_renewable)
    report.add(
        "no_contemporaneous_demand_predictor",
        demand_safe,
        f"forbidden overlap={sorted(predictor_names & contemporaneous_demand)}",
    )
    report.add(
        "no_contemporaneous_renewable_predictor",
        renewable_safe,
        f"forbidden overlap={sorted(predictor_names & contemporaneous_renewable)}",
    )
    report.add(
        "all_declared_predictors_exist",
        predictor_names.issubset(features.columns),
        f"missing={sorted(predictor_names - set(features.columns))}",
    )


def validate_no_fill(report: ValidationReport, source: pd.DataFrame, features: pd.DataFrame) -> None:
    missing_demand_rows = source.index[source["demand_mwh"].isna()]
    demand_null_propagation = True
    for index in missing_demand_rows:
        for hours in DEMAND_LAG_HOURS:
            shifted_index = index + hours
            if shifted_index < len(features):
                demand_null_propagation &= pd.isna(
                    features.loc[shifted_index, f"demand_lag_{hours}h"]
                )
        for window in ROLLING_WINDOWS:
            end_index = min(index + window, len(features) - 1)
            demand_null_propagation &= features.loc[
                index + 1 : end_index, f"demand_rolling_{window}h_mean"
            ].isna().all()

    missing_renewable_rows = source.index[source["solar_generation_mwh"].isna()]
    renewable_null_propagation = True
    for index in missing_renewable_rows:
        for hours in RENEWABLE_LAG_HOURS:
            shifted_index = index + hours
            if shifted_index < len(features):
                renewable_null_propagation &= pd.isna(
                    features.loc[shifted_index, f"solar_lag_{hours}h"]
                )
                renewable_null_propagation &= pd.isna(
                    features.loc[shifted_index, f"wind_lag_{hours}h"]
                )
                renewable_null_propagation &= pd.isna(
                    features.loc[shifted_index, f"solar_negative_reported_lag_{hours}h"]
                )
    report.add(
        "documented_demand_nulls_propagate_without_fill",
        bool(demand_null_propagation),
        f"checked {len(missing_demand_rows)} canonical null-demand timestamps",
    )
    report.add(
        "documented_renewable_nulls_propagate_without_fill",
        bool(renewable_null_propagation),
        f"checked {len(missing_renewable_rows)} canonical incomplete-renewable timestamps",
    )
    report.add(
        "no_imputation_or_zero_fill_detected",
        bool(demand_null_propagation and renewable_null_propagation),
        "exact formula checks plus boundary/source-null checks preserve every expected null",
    )


def run_validation(
    source_path: Path, feature_path: Path, group_path: Path, results_path: Path
) -> bool:
    report = ValidationReport()
    observed_hash = sha256(source_path)
    report.add(
        "canonical_source_sha256_unchanged",
        observed_hash == EXPECTED_SOURCE_SHA256,
        f"observed={observed_hash}; expected={EXPECTED_SOURCE_SHA256}",
    )
    source, features = load_inputs(source_path, feature_path)
    validate_structure(report, source, features)
    validate_target(report, source, features)
    validate_calendar(report, features)
    validate_demand_history(report, features)
    validate_renewable_history(report, features)
    validate_availability_flags(report, features)
    validate_declared_groups(report, features, group_path)
    validate_no_fill(report, source, features)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    report.frame().to_csv(results_path, index=False)
    passed_count = sum(row["status"] == "PASS" for row in report.rows)
    failed_checks = [row["check"] for row in report.rows if row["status"] == "FAIL"]
    summary_path = results_path.parent.parent / "feature_summary.md"
    if summary_path.exists():
        summary_prefix = summary_path.read_text(encoding="utf-8").split(
            "## Validation Notes", maxsplit=1
        )[0].rstrip()
        unexpected = "None." if not failed_checks else ", ".join(failed_checks)
        summary_validation = (
            "\n\n## Validation Notes\n\n"
            f"- Independent checks passed: {passed_count}/{len(report.rows)}\n"
            f"- Overall independent validation: {'PASS' if report.passed else 'FAIL'}\n"
            f"- Unexpected validation results: {unexpected}\n"
            f"- Canonical source SHA256: `{observed_hash}`\n\n"
            "Detailed results are in `tables/validation_results.csv`.\n"
        )
        summary_path.write_text(summary_prefix + summary_validation, encoding="utf-8")
    print(f"Feature validation: {'PASS' if report.passed else 'FAIL'}")
    print(f"Checks passed: {passed_count}/{len(report.rows)}")
    print(f"Canonical SHA256: {observed_hash}")
    if not report.passed:
        for row in report.rows:
            if row["status"] == "FAIL":
                print(f"- FAIL {row['check']}: {row['details']}")
    print(f"Validation table: {results_path}")
    return report.passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate leakage-safe CISO features.")
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--group-path", type=Path, default=DEFAULT_GROUP_PATH)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()
    try:
        passed = run_validation(
            args.source_path, args.feature_path, args.group_path, args.results_path
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"Feature validation: FAIL - {exc}")
        return 1
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
