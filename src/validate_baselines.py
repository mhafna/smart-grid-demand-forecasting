"""Independently validate chronological splits and saved baseline results."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "data" / "processed" / "eia_ciso_hourly_features.csv"
POLICY_PATH = ROOT / "data" / "modelling_split_policy.md"
TABLE_DIR = ROOT / "results" / "baselines" / "tables"
VALIDATION_PATH = TABLE_DIR / "baseline_validation_results.csv"
EXPECTED_FEATURE_SHA256 = "e2e0f05a06add2bea0a0660c5d545c0a2fa1fe7d0ddceab2e45addd536fda6f2"
TARGET = "target_demand_mwh"
SPLIT_BOUNDS = {
    "train": (pd.Timestamp("2022-01-01T00"), pd.Timestamp("2023-12-31T23")),
    "validation": (pd.Timestamp("2024-01-01T00"), pd.Timestamp("2024-06-30T23")),
    "test": (pd.Timestamp("2024-07-01T00"), pd.Timestamp("2024-12-31T23")),
}
MODEL_SOURCES = {
    "persistence_1h": ("demand_lag_1h", 1),
    "daily_seasonal_naive_24h": ("demand_lag_24h", 24),
    "weekly_seasonal_naive_168h": ("demand_lag_168h", 168),
}
ALL_MODELS = [*MODEL_SOURCES, "train_hour_of_week_mean"]


class ValidationReport:
    """Collect every check so a failure does not hide later evidence."""

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(self, check: str, passed: bool, details: str) -> None:
        self.rows.append(
            {"check": check, "status": "PASS" if passed else "FAIL", "details": details}
        )

    @property
    def passed(self) -> bool:
        return all(row["status"] == "PASS" for row in self.rows)

    def save(self) -> None:
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(VALIDATION_PATH, index=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close_series(
    actual: pd.Series, expected: pd.Series, *, atol: float = 1e-8
) -> tuple[bool, str]:
    """Compare numeric values and null positions with saved-CSV tolerance."""
    actual_numeric = pd.to_numeric(actual, errors="coerce").reset_index(drop=True)
    expected_numeric = pd.to_numeric(expected, errors="coerce").reset_index(drop=True)
    nulls_match = actual_numeric.isna().equals(expected_numeric.isna())
    finite = actual_numeric.notna() & expected_numeric.notna()
    values_match = bool(
        np.allclose(
            actual_numeric.loc[finite], expected_numeric.loc[finite], rtol=1e-10, atol=atol
        )
    )
    max_difference = (
        float((actual_numeric.loc[finite] - expected_numeric.loc[finite]).abs().max())
        if finite.any()
        else 0.0
    )
    return nulls_match and values_match, f"null masks match={nulls_match}; max difference={max_difference:.3g}"


def calculate_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    valid = (
        group["target_available"].fillna(False).astype(bool)
        & group["actual_demand_mwh"].notna()
        & group["prediction_mwh"].notna()
    )
    used = group.loc[valid]
    actual = used["actual_demand_mwh"].astype(float)
    prediction = used["prediction_mwh"].astype(float)
    error = prediction - actual
    absolute_error = error.abs()
    nonzero = actual.ne(0)
    denominator = actual.abs() + prediction.abs()
    smape_valid = denominator.ne(0)
    ss_total = float(((actual - actual.mean()) ** 2).sum())
    ss_residual = float((error**2).sum())
    return {
        "observation_count": int(valid.sum()),
        "mae_mwh": float(absolute_error.mean()),
        "rmse_mwh": float(np.sqrt((error**2).mean())),
        "mape_pct": float((absolute_error[nonzero] / actual[nonzero].abs()).mean() * 100),
        "smape_pct": float(
            (2 * absolute_error[smape_valid] / denominator[smape_valid]).mean() * 100
        ),
        "mean_error_bias_mwh": float(error.mean()),
        "r_squared": float(1 - ss_residual / ss_total) if ss_total > 0 else np.nan,
    }


def load_inputs() -> dict[str, pd.DataFrame]:
    features = pd.read_csv(SOURCE_CSV, dtype={"target_available": "boolean"})
    features["period"] = pd.to_datetime(
        features["period"], format="%Y-%m-%dT%H", errors="raise"
    )
    files = {
        "features": features,
        "split_summary": pd.read_csv(TABLE_DIR / "split_summary.csv"),
        "lookup": pd.read_csv(TABLE_DIR / "hour_of_week_training_lookup.csv"),
        "metrics": pd.read_csv(TABLE_DIR / "baseline_metrics_all.csv"),
        "predictions": pd.read_csv(
            TABLE_DIR / "baseline_predictions_all.csv", dtype={"target_available": "boolean"}
        ),
        "predictions_validation": pd.read_csv(
            TABLE_DIR / "baseline_predictions_validation.csv",
            dtype={"target_available": "boolean"},
        ),
        "predictions_test": pd.read_csv(
            TABLE_DIR / "baseline_predictions_test.csv", dtype={"target_available": "boolean"}
        ),
        "metadata": pd.read_csv(TABLE_DIR / "baseline_run_metadata.csv"),
        "model_spec": pd.read_csv(TABLE_DIR / "baseline_model_specification.csv"),
    }
    for name in ["predictions", "predictions_validation", "predictions_test"]:
        files[name]["period"] = pd.to_datetime(
            files[name]["period"], format="%Y-%m-%dT%H", errors="raise"
        )
    return files


def validate_source_and_policy(
    report: ValidationReport, inputs: dict[str, pd.DataFrame], hash_start: str
) -> None:
    features = inputs["features"]
    metadata = inputs["metadata"].iloc[0]
    report.add(
        "feature_hash_matches_pre_run_record",
        hash_start == EXPECTED_FEATURE_SHA256,
        f"expected={EXPECTED_FEATURE_SHA256}; actual={hash_start}",
    )
    metadata_hashes_match = (
        str(metadata["source_sha256_before"]) == hash_start
        and str(metadata["source_sha256_after"]) == hash_start
        and bool(metadata["source_hash_unchanged"])
    )
    report.add(
        "feature_hash_matches_run_metadata",
        metadata_hashes_match,
        "run metadata before/after hashes both equal the independently calculated hash",
    )
    policy = POLICY_PATH.read_text(encoding="utf-8")
    documented = all(
        timestamp in policy
        for timestamp in [
            "2022-01-01T00",
            "2023-12-31T23",
            "2024-01-01T00",
            "2024-06-30T23",
            "2024-07-01T00",
            "2024-12-31T23",
        ]
    )
    report.add(
        "documented_split_boundaries_exact",
        documented,
        "all six fixed UTC boundary timestamps appear in the split policy",
    )
    expected = pd.date_range("2022-01-01T00", "2024-12-31T23", freq="h")
    timeline_ok = (
        len(features) == len(expected)
        and not features["period"].duplicated().any()
        and features["period"].reset_index(drop=True).equals(pd.Series(expected))
    )
    report.add(
        "source_timeline_complete_unique_chronological",
        timeline_ok,
        f"rows={len(features):,}; expected={len(expected):,}; duplicates={int(features['period'].duplicated().sum())}",
    )


def independent_split_assignment(features: pd.DataFrame) -> pd.Series:
    assigned = pd.Series(pd.NA, index=features.index, dtype="string")
    for split, (start, end) in SPLIT_BOUNDS.items():
        assigned.loc[features["period"].between(start, end, inclusive="both")] = split
    return assigned


def validate_splits(report: ValidationReport, inputs: dict[str, pd.DataFrame]) -> pd.Series:
    features = inputs["features"]
    summary = inputs["split_summary"].set_index("split")
    assigned = independent_split_assignment(features)
    membership_counts = pd.DataFrame(
        {
            split: features["period"].between(start, end, inclusive="both")
            for split, (start, end) in SPLIT_BOUNDS.items()
        }
    ).sum(axis=1)
    report.add(
        "every_timestamp_in_exactly_one_split",
        assigned.notna().all() and membership_counts.eq(1).all(),
        f"unassigned={int(assigned.isna().sum())}; overlapping={int(membership_counts.gt(1).sum())}",
    )

    boundary_ok = True
    count_details = []
    target_ok = features["target_available"].fillna(False) & features[TARGET].notna()
    for split, (start, end) in SPLIT_BOUNDS.items():
        mask = assigned.eq(split)
        row = summary.loc[split]
        saved_start = pd.Timestamp(row["start_utc"])
        saved_end = pd.Timestamp(row["end_utc"])
        expected_total = int(mask.sum())
        expected_targets = int((mask & target_ok).sum())
        current_ok = (
            saved_start == start
            and saved_end == end
            and int(row["total_rows"]) == expected_total
            and int(row["target_available_rows"]) == expected_targets
        )
        boundary_ok &= current_ok
        count_details.append(f"{split}={expected_total:,}/{expected_targets:,}")
    report.add(
        "split_boundaries_and_counts_match_policy",
        boundary_ok,
        "total/target rows: " + "; ".join(count_details),
    )
    chronological = (
        features.loc[assigned.eq("train"), "period"].max()
        < features.loc[assigned.eq("validation"), "period"].min()
        and features.loc[assigned.eq("validation"), "period"].max()
        < features.loc[assigned.eq("test"), "period"].min()
    )
    report.add(
        "split_order_strictly_chronological",
        chronological,
        "training ends before validation; validation ends before test",
    )
    return assigned


def validate_hour_of_week_fit(
    report: ValidationReport, inputs: dict[str, pd.DataFrame], assigned: pd.Series
) -> tuple[pd.Series, float]:
    features = inputs["features"].copy()
    lookup = inputs["lookup"].sort_values("hour_of_week_utc").reset_index(drop=True)
    features["split"] = assigned
    features["hour_of_week_utc"] = features["period"].dt.dayofweek * 24 + features["period"].dt.hour
    fit = features.loc[
        features["split"].eq("train")
        & features["target_available"].fillna(False)
        & features[TARGET].notna()
    ]
    expected = (
        fit.groupby("hour_of_week_utc", observed=True)[TARGET]
        .agg(
            training_observation_count="count",
            training_target_sum_mwh="sum",
            training_mean_demand_mwh="mean",
        )
        .reset_index()
        .sort_values("hour_of_week_utc")
        .reset_index(drop=True)
    )
    category_ok = lookup["hour_of_week_utc"].tolist() == list(range(168))
    count_ok = lookup["training_observation_count"].astype(int).equals(
        expected["training_observation_count"].astype(int)
    )
    sum_ok, sum_detail = close_series(
        lookup["training_target_sum_mwh"], expected["training_target_sum_mwh"]
    )
    mean_ok, mean_detail = close_series(
        lookup["training_mean_demand_mwh"], expected["training_mean_demand_mwh"]
    )
    report.add(
        "hour_of_week_lookup_exactly_recomputed_from_training",
        category_ok and count_ok and sum_ok and mean_ok,
        f"168 categories={category_ok}; counts={count_ok}; sums: {sum_detail}; means: {mean_detail}",
    )
    latest = pd.to_datetime(lookup["latest_contributing_timestamp_utc"])
    fit_bounds_ok = (
        lookup["fit_split"].eq("train").all()
        and pd.to_datetime(lookup["fit_start_utc"]).eq(SPLIT_BOUNDS["train"][0]).all()
        and pd.to_datetime(lookup["fit_end_utc"]).eq(SPLIT_BOUNDS["train"][1]).all()
        and latest.le(SPLIT_BOUNDS["train"][1]).all()
        and int(inputs["metadata"].iloc[0]["validation_or_test_targets_in_fitted_statistics"]) == 0
    )
    report.add(
        "no_validation_or_test_target_in_fitted_statistics",
        fit_bounds_ok,
        f"latest contributing timestamp={latest.max():%Y-%m-%dT%H}; fit rows={len(fit):,}",
    )
    global_mean = float(fit[TARGET].mean())
    saved_global = lookup["training_global_mean_demand_mwh"]
    global_ok = np.allclose(saved_global, global_mean, rtol=1e-10, atol=1e-8)
    report.add(
        "hour_of_week_global_fallback_training_only",
        global_ok,
        f"independent training mean={global_mean:.10f}; categories={len(lookup)}",
    )
    return expected.set_index("hour_of_week_utc")["training_mean_demand_mwh"], global_mean


def validate_prediction_structure(
    report: ValidationReport, inputs: dict[str, pd.DataFrame], assigned: pd.Series
) -> None:
    features = inputs["features"]
    predictions = inputs["predictions"]
    eval_rows = int(assigned.isin(["validation", "test"]).sum())
    unique_models = sorted(predictions["model_name"].unique())
    duplicate_keys = int(predictions.duplicated(["period", "split", "model_name"]).sum())
    expected_rows = eval_rows * len(ALL_MODELS)
    complete = (
        len(predictions) == expected_rows
        and unique_models == sorted(ALL_MODELS)
        and duplicate_keys == 0
        and predictions.groupby(["split", "model_name"], observed=True).size().eq(
            predictions.groupby("split")["period"].nunique()
        ).all()
    )
    report.add(
        "prediction_table_complete_and_unique",
        complete,
        f"rows={len(predictions):,}; expected={expected_rows:,}; duplicate keys={duplicate_keys}",
    )
    validation_saved = inputs["predictions_validation"].reset_index(drop=True)
    validation_from_all = predictions.loc[predictions["split"].eq("validation")].reset_index(drop=True)
    test_saved = inputs["predictions_test"].reset_index(drop=True)
    test_from_all = predictions.loc[predictions["split"].eq("test")].reset_index(drop=True)
    separated_ok = validation_saved.equals(validation_from_all) and test_saved.equals(test_from_all)
    report.add(
        "separate_prediction_files_match_combined_table",
        separated_ok,
        f"validation rows={len(validation_saved):,}; test rows={len(test_saved):,}",
    )


def validate_prediction_formulas(
    report: ValidationReport,
    inputs: dict[str, pd.DataFrame],
    assigned: pd.Series,
    hour_means: pd.Series,
    global_mean: float,
) -> None:
    features = inputs["features"].copy()
    predictions = inputs["predictions"]
    features["split"] = assigned
    features["hour_of_week_utc"] = features["period"].dt.dayofweek * 24 + features["period"].dt.hour
    eval_features = features.loc[features["split"].isin(["validation", "test"])].set_index("period")

    all_saved_lags_ok = True
    all_historical_lags_ok = True
    details = []
    for model, (column, hours) in MODEL_SOURCES.items():
        model_predictions = predictions.loc[predictions["model_name"].eq(model)].set_index("period")
        expected_feature = eval_features.loc[model_predictions.index, column]
        saved_ok, detail = close_series(model_predictions["prediction_mwh"], expected_feature)
        all_saved_lags_ok &= saved_ok
        historical = features["demand_mwh"].shift(hours)
        source_ok, source_detail = close_series(features[column], historical)
        all_historical_lags_ok &= source_ok
        details.append(f"{model}: saved {detail}; source {source_detail}")
    report.add(
        "lag_predictions_equal_required_feature_values",
        all_saved_lags_ok,
        "; ".join(details),
    )
    report.add(
        "lag_features_equal_exact_historical_demand_sources",
        all_historical_lags_ok,
        "1h, 24h, and 168h columns equal independent demand shifts including nulls",
    )

    hour_predictions = predictions.loc[
        predictions["model_name"].eq("train_hour_of_week_mean")
    ].set_index("period")
    mapped = eval_features.loc[hour_predictions.index, "hour_of_week_utc"].map(hour_means)
    expected_hour = mapped.fillna(global_mean)
    hour_ok, hour_detail = close_series(hour_predictions["prediction_mwh"], expected_hour)
    fallback_expected = mapped.isna().to_numpy()
    fallback_actual = hour_predictions["hour_of_week_global_mean_fallback"].astype(bool).to_numpy()
    report.add(
        "hour_of_week_predictions_use_frozen_training_lookup",
        hour_ok and np.array_equal(fallback_actual, fallback_expected),
        f"{hour_detail}; fallback rows={int(fallback_expected.sum())}",
    )


def validate_missingness_and_errors(
    report: ValidationReport, predictions: pd.DataFrame
) -> None:
    valid = (
        predictions["target_available"].fillna(False).astype(bool)
        & predictions["actual_demand_mwh"].notna()
        & predictions["prediction_mwh"].notna()
    )
    expected_error = (predictions["prediction_mwh"] - predictions["actual_demand_mwh"]).where(valid)
    error_ok, error_detail = close_series(predictions["error_mwh"], expected_error)
    abs_ok, abs_detail = close_series(predictions["absolute_error_mwh"], expected_error.abs())
    pct_valid = valid & predictions["actual_demand_mwh"].ne(0)
    expected_pct = (
        100 * expected_error / predictions["actual_demand_mwh"]
    ).where(pct_valid)
    pct_ok, pct_detail = close_series(predictions["percentage_error_pct"], expected_pct)
    invalid_outputs_null = predictions.loc[
        ~valid, ["error_mwh", "absolute_error_mwh", "percentage_error_pct"]
    ].isna().all().all()
    report.add(
        "missing_targets_and_predictions_excluded_not_filled",
        invalid_outputs_null and error_ok and abs_ok and pct_ok,
        f"invalid metric rows={int((~valid).sum())}; errors {error_detail}; absolute {abs_detail}; percentage {pct_detail}",
    )


def validate_metrics(report: ValidationReport, inputs: dict[str, pd.DataFrame]) -> None:
    predictions = inputs["predictions"]
    metrics = inputs["metrics"].set_index(["split", "model_name"])
    count_ok = True
    values_ok = True
    max_difference = 0.0
    for split in ["validation", "test"]:
        for model in ALL_MODELS:
            group = predictions.loc[
                predictions["split"].eq(split) & predictions["model_name"].eq(model)
            ]
            expected = calculate_metrics(group)
            saved = metrics.loc[(split, model)]
            count_ok &= int(saved["observation_count"]) == expected["observation_count"]
            for column, value in expected.items():
                if column == "observation_count":
                    continue
                difference = abs(float(saved[column]) - float(value))
                max_difference = max(max_difference, difference)
                values_ok &= bool(np.isclose(saved[column], value, rtol=1e-10, atol=1e-8))
    report.add(
        "metric_counts_match_prediction_eligibility",
        count_ok,
        "each saved count equals target_available AND numeric target AND numeric prediction",
    )
    report.add(
        "all_metrics_reproduce_from_prediction_table",
        values_ok,
        f"independent MAE/RMSE/MAPE/sMAPE/bias/R² maximum difference={max_difference:.3g}",
    )


def validate_predictor_scope(report: ValidationReport, inputs: dict[str, pd.DataFrame]) -> None:
    specification = inputs["model_spec"]
    allowed_sources = {
        "demand_lag_1h",
        "demand_lag_24h",
        "demand_lag_168h",
        "training-only hour-of-week demand mean",
    }
    no_renewable = (
        set(specification["prediction_source"]) == allowed_sources
        and not specification["uses_contemporaneous_renewable_measurement"].astype(bool).any()
        and int(inputs["metadata"].iloc[0]["contemporaneous_renewable_predictors_used"]) == 0
    )
    report.add(
        "no_contemporaneous_renewable_measurements_used",
        no_renewable,
        "all four independently reproduced predictions use only demand lags or training demand means",
    )


def main() -> None:
    hash_start = sha256(SOURCE_CSV)
    report = ValidationReport()
    try:
        inputs = load_inputs()
        validate_source_and_policy(report, inputs, hash_start)
        assigned = validate_splits(report, inputs)
        validate_prediction_structure(report, inputs, assigned)
        hour_means, global_mean = validate_hour_of_week_fit(report, inputs, assigned)
        validate_prediction_formulas(report, inputs, assigned, hour_means, global_mean)
        validate_missingness_and_errors(report, inputs["predictions"])
        validate_metrics(report, inputs)
        validate_predictor_scope(report, inputs)
        hash_end = sha256(SOURCE_CSV)
        report.add(
            "feature_hash_unchanged_during_validation",
            hash_start == hash_end == EXPECTED_FEATURE_SHA256,
            f"before={hash_start}; after={hash_end}",
        )
    except Exception as exc:
        report.add("validator_completed_without_exception", False, repr(exc))
    report.save()
    frame = pd.DataFrame(report.rows)
    print(frame.to_string(index=False))
    print(f"\nValidation checks: {(frame['status'] == 'PASS').sum()}/{len(frame)} passed")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
