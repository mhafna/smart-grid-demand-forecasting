"""Independently validate one-hour-ahead model artifacts and derived results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / "data" / "processed" / "eia_ciso_hourly_features.csv"
DECLARATION_FILE = ROOT / "results" / "features" / "tables" / "feature_groups.csv"
TABLE_DIR = ROOT / "results" / "models" / "tables"
MODEL_DIR = ROOT / "models" / "one_step"
OUTPUT_FILE = TABLE_DIR / "one_step_validation_results.csv"
TARGET = "target_demand_mwh"
EXPECTED_BOUNDS = {
    "train": (pd.Timestamp("2022-01-01T00"), pd.Timestamp("2023-12-31T23")),
    "validation": (pd.Timestamp("2024-01-01T00"), pd.Timestamp("2024-06-30T23")),
    "test": (pd.Timestamp("2024-07-01T00"), pd.Timestamp("2024-12-31T23")),
}
GROUPS = ["calendar_only", "autoregressive_demand", "renewable_history_enhanced"]
ALGORITHMS = ["linear_regression", "xgboost"]
PROHIBITED = {
    "demand_mwh", "target_demand_mwh", "solar_generation_mwh",
    "wind_generation_mwh", "solar_wind_generation_mwh",
    "residual_demand_after_solar_wind_mwh", "solar_wind_share_pct",
    "solar_negative_reported",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    actual = group["actual_demand_mwh"].to_numpy(dtype=float)
    prediction = group["prediction_mwh"].to_numpy(dtype=float)
    error = prediction - actual
    absolute = np.abs(error)
    nonzero = actual != 0
    smape_denominator = np.abs(actual) + np.abs(prediction)
    smape_valid = smape_denominator != 0
    r2_denominator = np.sum((actual - actual.mean()) ** 2)
    return {
        "observation_count": int(len(group)),
        "mae_mwh": float(absolute.mean()),
        "rmse_mwh": float(np.sqrt(np.mean(error ** 2))),
        "mape_pct": float(100 * np.mean(absolute[nonzero] / np.abs(actual[nonzero]))) if nonzero.any() else np.nan,
        "smape_pct": float(100 * np.mean(2 * absolute[smape_valid] / smape_denominator[smape_valid])) if smape_valid.any() else np.nan,
        "mean_error_bias_mwh": float(error.mean()),
        "r_squared": float(1 - np.sum(error ** 2) / r2_denominator) if r2_denominator else np.nan,
    }


def model_name(algorithm: str, group: str) -> str:
    return f"{algorithm}__{group}"


def finite_rows(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return pd.Series(np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1), index=frame.index)


def main() -> None:
    results: list[dict[str, Any]] = []

    def record(check: str, passed: bool, detail: str) -> None:
        results.append({"check": check, "passed": bool(passed), "detail": detail})

    def safe_check(check: str, function: Callable[[], tuple[bool, str]]) -> None:
        try:
            passed, detail = function()
            record(check, passed, detail)
        except Exception as exc:  # keep the complete audit table even after one failure
            record(check, False, f"{type(exc).__name__}: {exc}")

    declarations = pd.read_csv(DECLARATION_FILE)
    declared = pd.read_csv(TABLE_DIR / "model_feature_lists.csv")
    features = pd.read_csv(FEATURE_FILE, low_memory=False)
    features["period"] = pd.to_datetime(features["period"], format="%Y-%m-%dT%H", errors="raise")
    truth = features["target_available"].astype(str).str.lower().map({"true": True, "false": False})
    features["target_available"] = truth.astype(bool)
    numeric_columns = set(declared["feature"]) | {TARGET, "demand_lag_1h"}
    for column in numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features["split"] = pd.Series(pd.NA, index=features.index, dtype="string")
    for split, (start, end) in EXPECTED_BOUNDS.items():
        features.loc[features["period"].between(start, end, inclusive="both"), "split"] = split

    predictions = pd.read_csv(TABLE_DIR / "one_step_predictions_all.csv")
    predictions["period"] = pd.to_datetime(predictions["period"], format="%Y-%m-%dT%H", errors="raise")
    metrics = pd.read_csv(TABLE_DIR / "model_metrics_all.csv")
    eligibility_table = pd.read_csv(TABLE_DIR / "model_eligibility_by_timestamp.csv")
    eligibility_table["period"] = pd.to_datetime(eligibility_table["period"], format="%Y-%m-%dT%H", errors="raise")
    search = pd.read_csv(TABLE_DIR / "xgboost_validation_search.csv")
    run_metadata = json.loads((MODEL_DIR / "run_metadata.json").read_text(encoding="utf-8"))

    safe_check("fixed_split_boundaries", lambda: (
        run_metadata["fixed_split_bounds"] == {key: [str(value[0]), str(value[1])] for key, value in EXPECTED_BOUNDS.items()}
        and all(
            features.loc[features["split"] == split, "period"].min() == bounds[0]
            and features.loc[features["split"] == split, "period"].max() == bounds[1]
            for split, bounds in EXPECTED_BOUNDS.items()
        ),
        "Run metadata and observed split endpoints match the fixed policy."
    ))

    def check_split_overlap() -> tuple[bool, str]:
        intervals = list(EXPECTED_BOUNDS.items())
        no_overlap = all(intervals[index][1][1] < intervals[index + 1][1][0] for index in range(len(intervals) - 1))
        assigned_once = features["split"].notna().all() and not features["period"].duplicated().any()
        return no_overlap and assigned_once, "Each unique feature timestamp is assigned once and split intervals do not overlap."

    safe_check("no_split_overlap", check_split_overlap)
    safe_check("training_precedes_validation_and_test", lambda: (
        features.loc[features["split"] == "train", "period"].max()
        < features.loc[features["split"] == "validation", "period"].min()
        < features.loc[features["split"] == "test", "period"].min(),
        "The latest training timestamp is earlier than validation and test."
    ))

    def check_declared_features() -> tuple[bool, str]:
        safe_map = declarations.set_index("feature")["safe_at_forecast_time"].astype(str).str.lower().eq("true")
        failures = []
        for group in GROUPS:
            expected = declarations[
                declarations[group].astype(str).str.lower().eq("true")
                & declarations["safe_at_forecast_time"].astype(str).str.lower().eq("true")
            ]["feature"].tolist()
            observed = declared[declared["feature_group"] == group].sort_values("feature_order")["feature"].tolist()
            if observed != expected:
                failures.append(f"{group}: feature order/list differs")
            if any(not bool(safe_map.get(feature, False)) for feature in observed):
                failures.append(f"{group}: unsafe declaration selected")
        return not failures, "All exact feature lists match safe machine-readable declarations." if not failures else "; ".join(failures)

    safe_check("predictors_match_declared_safe_groups", check_declared_features)

    def check_prohibited() -> tuple[bool, str]:
        predictors = set(declared["feature"])
        direct = sorted(predictors & PROHIBITED)
        future_named = sorted(feature for feature in predictors if "lead" in feature.lower() or "future" in feature.lower())
        declaration_rows = declarations[declarations["feature"].isin(predictors)]
        zero_lookback_measurements = declaration_rows[
            declaration_rows["lookback"].astype(str).eq("0h")
            & ~declaration_rows["group"].astype(str).eq("calendar_only")
        ]["feature"].tolist()
        failures = direct + future_named + zero_lookback_measurements
        return not failures, "No current, zero-lookback measured, or future-named predictor is used." if not failures else f"Prohibited predictors: {failures}"

    safe_check("no_prohibited_same_hour_or_future_predictor", check_prohibited)

    target_ok = features["target_available"] & features[TARGET].notna() & np.isfinite(features[TARGET])
    group_eligibility = {
        group: target_ok & finite_rows(features, declared[declared["feature_group"] == group].sort_values("feature_order")["feature"].tolist())
        for group in GROUPS
    }

    def check_scalers() -> tuple[bool, str]:
        failures = []
        for group in GROUPS:
            name = model_name("linear_regression", group)
            metadata = json.loads((MODEL_DIR / f"{name}.metadata.json").read_text(encoding="utf-8"))
            pipeline = joblib.load(MODEL_DIR / f"{name}.joblib")
            scaler = pipeline.named_steps["scaler"]
            feature_list = metadata["feature_list"]
            training = features.loc[features["split"].eq("train") & group_eligibility[group], feature_list].astype(float)
            if metadata["scaler_fit_split"] != "train" or int(scaler.n_samples_seen_) != len(training):
                failures.append(f"{name}: fit split/count mismatch")
            if not np.allclose(scaler.mean_, training.mean(axis=0).to_numpy(), rtol=1e-10, atol=1e-8):
                failures.append(f"{name}: means do not reproduce from training rows")
            expected_scale = np.sqrt(training.var(axis=0, ddof=0).to_numpy())
            expected_scale[expected_scale == 0] = 1.0
            if not np.allclose(scaler.scale_, expected_scale, rtol=1e-9, atol=1e-8):
                failures.append(f"{name}: scales do not reproduce from training rows")
        return not failures, "Every scaler mean and scale reproduces from eligible training rows only." if not failures else "; ".join(failures)

    safe_check("scaler_fitted_on_training_only", check_scalers)

    def check_xgb_selection() -> tuple[bool, str]:
        failures = []
        required = search["fit_split"].eq("train").all() and search["selection_split"].eq("validation").all()
        if not required:
            failures.append("search fit/selection split labels are incorrect")
        for group in GROUPS:
            rows = search[search["feature_group"] == group]
            winner = rows[rows["selected"].astype(str).str.lower().eq("true")]
            minimum_id = int(rows.sort_values(["mae_mwh", "candidate_id"]).iloc[0]["candidate_id"])
            if len(winner) != 1 or int(winner.iloc[0]["candidate_id"]) != minimum_id:
                failures.append(f"{group}: selected candidate is not minimum validation MAE")
        return not failures, "Each group selects exactly the minimum validation-MAE candidate fitted on training." if not failures else "; ".join(failures)

    safe_check("xgboost_selection_uses_validation_only", check_xgb_selection)
    safe_check("test_not_used_for_hyperparameter_selection", lambda: (
        not any("test" in column.lower() for column in search.columns)
        and all(
            json.loads((MODEL_DIR / f"{model_name('xgboost', group)}.metadata.json").read_text(encoding="utf-8"))["test_used_for_selection"] is False
            for group in GROUPS
        ),
        "Search results contain no test metric and every XGBoost metadata file records test_used_for_selection=false."
    ))

    def check_prediction_splits() -> tuple[bool, str]:
        expected = predictions["split"].map({key: bounds for key, bounds in EXPECTED_BOUNDS.items()})
        in_range = [bounds[0] <= timestamp <= bounds[1] for timestamp, bounds in zip(predictions["period"], expected)]
        return all(in_range) and predictions["split"].isin(["validation", "test"]).all(), "Every prediction timestamp lies inside its labelled evaluation split."

    safe_check("prediction_timestamps_match_splits", check_prediction_splits)

    def check_metrics() -> tuple[bool, str]:
        failures = []
        keys = ["split", "model", "algorithm", "feature_group", "comparison_set_type"]
        saved = metrics.set_index(keys)
        for values, group in predictions.groupby(keys, observed=True, sort=False):
            reproduced = calculate_metrics(group)
            row = saved.loc[values]
            for metric, value in reproduced.items():
                if metric == "observation_count":
                    equal = int(row[metric]) == int(value)
                else:
                    equal = bool(np.isclose(float(row[metric]), float(value), rtol=1e-9, atol=1e-8, equal_nan=True))
                if not equal:
                    failures.append(f"{values}: {metric}")
        return not failures, "All metric counts and values reproduce from saved timestamp predictions." if not failures else f"Mismatches: {failures[:10]}"

    safe_check("metric_counts_and_values_reproduce", check_metrics)

    def check_persistence() -> tuple[bool, str]:
        persistence = predictions[
            (predictions["comparison_set_type"] == "common")
            & (predictions["model"] == "persistence_1h")
        ].merge(features[["period", "demand_lag_1h"]], on="period", how="left", validate="one_to_one")
        exact = np.array_equal(persistence["prediction_mwh"].to_numpy(), persistence["demand_lag_1h"].to_numpy())
        return exact, "Every common-subset persistence prediction exactly equals demand_lag_1h."

    safe_check("persistence_equals_demand_lag_1h", check_persistence)

    def check_common_timestamps() -> tuple[bool, str]:
        common = predictions[predictions["comparison_set_type"] == "common"]
        failures = []
        expected_models = {"persistence_1h", *[model_name(a, g) for a in ALGORITHMS for g in GROUPS]}
        for split in ["validation", "test"]:
            rows = common[common["split"] == split]
            model_sets = {model: tuple(rows.loc[rows["model"] == model, "period"]) for model in expected_models}
            reference = model_sets["persistence_1h"]
            if set(model_sets) != expected_models or any(values != reference for values in model_sets.values()):
                failures.append(split)
        return not failures, "All seven compared models have identical ordered common timestamps in each split." if not failures else f"Timestamp mismatch in: {failures}"

    safe_check("common_subset_timestamps_identical", check_common_timestamps)

    def check_artifacts() -> tuple[bool, str]:
        failures = []
        declared_lists = {
            group: declared[declared["feature_group"] == group].sort_values("feature_order")["feature"].tolist()
            for group in GROUPS
        }
        for algorithm in ALGORITHMS:
            for group in GROUPS:
                name = model_name(algorithm, group)
                metadata = json.loads((MODEL_DIR / f"{name}.metadata.json").read_text(encoding="utf-8"))
                artifact = MODEL_DIR / f"{name}.joblib"
                fitted = joblib.load(artifact)
                observed = list(fitted.feature_names_in_)
                expected = declared_lists[group]
                if metadata["feature_list"] != expected or observed != expected:
                    failures.append(f"{name}: feature list mismatch")
                if metadata["artifact_sha256"] != sha256(artifact):
                    failures.append(f"{name}: artifact hash mismatch")
                expected_train_count = int((features["split"].eq("train") & group_eligibility[group]).sum())
                if int(metadata["training_row_count"]) != expected_train_count:
                    failures.append(f"{name}: training count mismatch")
        return not failures, "All six artifacts, hashes, feature lists, and training counts match metadata." if not failures else "; ".join(failures)

    safe_check("model_artifacts_match_recorded_features", check_artifacts)

    def check_feature_hash() -> tuple[bool, str]:
        current = sha256(FEATURE_FILE)
        expected = run_metadata["feature_file_sha256_before"]
        metadata_hashes = []
        for algorithm in ALGORITHMS:
            for group in GROUPS:
                metadata = json.loads((MODEL_DIR / f"{model_name(algorithm, group)}.metadata.json").read_text(encoding="utf-8"))
                metadata_hashes.append(metadata["feature_file_sha256"])
        unchanged = (
            current == expected == run_metadata["feature_file_sha256_after"]
            and bool(run_metadata["feature_file_unchanged"])
            and all(value == current for value in metadata_hashes)
        )
        return unchanged, f"Current and recorded feature SHA-256: {current}."

    safe_check("feature_master_sha256_unchanged", check_feature_hash)

    output = pd.DataFrame(results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    failed = output[~output["passed"]]
    print(output.to_string(index=False))
    if not failed.empty:
        raise SystemExit(f"Independent validation failed {len(failed)} check(s).")
    print(f"Independent validation passed all {len(output)} checks.")


if __name__ == "__main__":
    main()
