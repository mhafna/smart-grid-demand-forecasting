"""Independently validate recursive 24-hour forecasting outputs."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / "data" / "processed" / "eia_ciso_hourly_features.csv"
FEATURE_LIST_FILE = ROOT / "results" / "models" / "tables" / "model_feature_lists.csv"
ONE_STEP_FILE = ROOT / "results" / "models" / "tables" / "one_step_predictions_all.csv"
MODEL_DIR = ROOT / "models" / "one_step"
TABLE_DIR = ROOT / "results" / "recursive" / "tables"
RUN_SCRIPT = ROOT / "src" / "run_recursive_forecasts.py"
OUTPUT_FILE = TABLE_DIR / "recursive_validation_results.csv"
TOLERANCE = 1e-6
EXPECTED_DAYS = {
    "validation": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30"), 182),
    "test": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31"), 184),
}
MODEL_FILES = {
    "recursive_linear_regression": (
        "linear_regression__autoregressive_demand.joblib",
        "linear_regression__autoregressive_demand.metadata.json",
        "linear_regression__autoregressive_demand",
    ),
    "recursive_xgboost": (
        "xgboost__autoregressive_demand.joblib",
        "xgboost__autoregressive_demand.metadata.json",
        "xgboost__autoregressive_demand",
    ),
}
BASELINES = {"flat_persistence", "daily_seasonal_naive", "weekly_seasonal_naive"}
ALL_MODELS = set(MODEL_FILES) | BASELINES
LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
WINDOWS = [24, 168]


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
    denominator = np.abs(actual) + np.abs(prediction)
    smape_valid = denominator != 0
    r2_denominator = np.sum((actual - actual.mean()) ** 2)
    return {
        "count": int(len(group)), "mae_mwh": float(absolute.mean()),
        "rmse_mwh": float(np.sqrt(np.mean(error ** 2))),
        "mape_pct": float(100 * np.mean(absolute[nonzero] / np.abs(actual[nonzero]))) if nonzero.any() else np.nan,
        "smape_pct": float(100 * np.mean(2 * absolute[smape_valid] / denominator[smape_valid])) if smape_valid.any() else np.nan,
        "bias_mwh": float(error.mean()),
        "r_squared": float(1 - np.sum(error ** 2) / r2_denominator) if r2_denominator else np.nan,
    }


def calendar_values(timestamp: pd.Timestamp) -> dict[str, float | int | bool]:
    hour_angle = timestamp.hour * 2 * math.pi / 24
    dow_angle = timestamp.dayofweek * 2 * math.pi / 7
    days = 366 if timestamp.is_leap_year else 365
    doy_angle = (timestamp.dayofyear - 1) * 2 * math.pi / days
    return {
        "year": timestamp.year, "month": timestamp.month, "day": timestamp.day,
        "day_of_year": timestamp.dayofyear, "hour_utc": timestamp.hour,
        "day_of_week_utc": timestamp.dayofweek,
        "is_weekend_utc": timestamp.dayofweek >= 5,
        "hour_sin": math.sin(hour_angle), "hour_cos": math.cos(hour_angle),
        "day_of_week_sin": math.sin(dow_angle), "day_of_week_cos": math.cos(dow_angle),
        "day_of_year_sin": math.sin(doy_angle), "day_of_year_cos": math.cos(doy_angle),
    }


def reproduce_features(
    target: pd.Timestamp, origin: pd.Timestamp, buffer: dict[pd.Timestamp, float]
) -> tuple[dict[str, float | int | bool], dict[str, Any]]:
    values = calendar_values(target)
    lag_observed = lag_predicted = rolling_observed = rolling_predicted = 0
    source_times: list[pd.Timestamp] = []
    for lag in LAGS:
        source = target - pd.Timedelta(hours=lag)
        values[f"demand_lag_{lag}h"] = float(buffer[source])
        lag_observed += int(source <= origin)
        lag_predicted += int(source > origin)
        source_times.append(source)
    for window in WINDOWS:
        times = [target - pd.Timedelta(hours=h) for h in range(1, window + 1)]
        data = np.array([buffer[t] for t in times], dtype=float)
        observed = sum(t <= origin for t in times)
        predicted = window - observed
        rolling_observed += observed * 4
        rolling_predicted += predicted * 4
        values.update({
            f"demand_rolling_{window}h_mean": float(np.mean(data)),
            f"demand_rolling_{window}h_std": float(np.std(data, ddof=1)),
            f"demand_rolling_{window}h_min": float(np.min(data)),
            f"demand_rolling_{window}h_max": float(np.max(data)),
        })
        source_times.extend(times)
    provenance = {
        "lag_observed_input_count": lag_observed,
        "lag_prediction_input_count": lag_predicted,
        "rolling_observed_input_count": rolling_observed,
        "rolling_prediction_input_count": rolling_predicted,
        "earliest_source_timestamp": min(source_times),
        "latest_source_timestamp": max(source_times),
        "all_sources_valid": max(source_times) < target,
    }
    return values, provenance


def main() -> None:
    results: list[dict[str, Any]] = []

    def record(check: str, passed: bool, detail: str) -> None:
        results.append({"check": check, "passed": bool(passed), "detail": detail})

    def safe(check: str, function: Callable[[], tuple[bool, str]]) -> None:
        try:
            passed, detail = function()
            record(check, passed, detail)
        except Exception as exc:
            record(check, False, f"{type(exc).__name__}: {exc}")

    features = pd.read_csv(FEATURE_FILE, low_memory=False)
    features["period"] = pd.to_datetime(features["period"], format="%Y-%m-%dT%H", errors="raise")
    features["demand_mwh"] = pd.to_numeric(features["demand_mwh"], errors="coerce")
    features["target_demand_mwh"] = pd.to_numeric(features["target_demand_mwh"], errors="coerce")
    features = features.set_index("period", drop=False)
    origins = pd.read_csv(TABLE_DIR / "forecast_origin_eligibility.csv")
    predictions = pd.read_csv(TABLE_DIR / "recursive_predictions.csv")
    provenance = pd.read_csv(TABLE_DIR / "recursive_feature_provenance.csv")
    metadata = pd.read_csv(TABLE_DIR / "recursive_run_metadata.csv")
    consistency = pd.read_csv(TABLE_DIR / "horizon_1_consistency.csv")
    hashes = pd.read_csv(TABLE_DIR / "upstream_file_hashes.csv")
    for table, columns in [
        (origins, ["forecast_date", "forecast_origin"]),
        (predictions, ["forecast_date", "forecast_origin", "target_timestamp"]),
        (provenance, ["forecast_origin", "target_timestamp", "earliest_source_timestamp", "latest_source_timestamp"]),
    ]:
        for column in columns: table[column] = pd.to_datetime(table[column], format="%Y-%m-%dT%H", errors="raise")

    def check_boundaries() -> tuple[bool, str]:
        failures = []
        for split, (start, end, count) in EXPECTED_DAYS.items():
            rows = origins[origins["split"].eq(split)].sort_values("forecast_date")
            if len(rows) != count or rows["forecast_date"].iloc[0] != start or rows["forecast_date"].iloc[-1] != end:
                failures.append(split)
        meta_ok = (
            pd.Timestamp(metadata.iloc[0]["validation_start"]) == pd.Timestamp("2024-01-01T00")
            and pd.Timestamp(metadata.iloc[0]["validation_end"]) == pd.Timestamp("2024-06-30T23")
            and pd.Timestamp(metadata.iloc[0]["test_start"]) == pd.Timestamp("2024-07-01T00")
            and pd.Timestamp(metadata.iloc[0]["test_end"]) == pd.Timestamp("2024-12-31T23")
        )
        return not failures and meta_ok, "Forecast-day and target-period boundaries exactly match the fixed policy." if not failures and meta_ok else f"Boundary failures: {failures}; metadata_ok={meta_ok}"

    safe("exact_validation_and_test_boundaries", check_boundaries)
    safe("origins_at_preceding_23_utc", lambda: (
        bool((origins["forecast_origin"].dt.hour.eq(23) & origins["forecast_origin"].eq(origins["forecast_date"] - pd.Timedelta(hours=1))).all()),
        "Every origin is 23:00 UTC on the day before its forecast date."
    ))

    def check_horizons() -> tuple[bool, str]:
        failures = []
        for keys, group in predictions.groupby(["split", "model", "forecast_origin"], sort=False):
            if group.sort_values("horizon")["horizon"].tolist() != list(range(1, 25)):
                failures.append(str(keys))
        return not failures, "Every model-origin forecast contains horizons 1-24 exactly once." if not failures else f"Failures: {failures[:5]}"

    safe("horizons_1_through_24_in_order", check_horizons)
    safe("target_equals_origin_plus_horizon", lambda: (
        bool(predictions["target_timestamp"].eq(predictions["forecast_origin"] + pd.to_timedelta(predictions["horizon"], unit="h")).all()),
        "Every target timestamp equals its forecast origin plus horizon hours."
    ))
    safe("no_origin_in_both_splits", lambda: (
        origins.groupby("forecast_origin")["split"].nunique().max() == 1,
        "No forecast origin appears in both validation and test."
    ))

    def check_common_timestamps() -> tuple[bool, str]:
        failures = []
        for split in EXPECTED_DAYS:
            rows = predictions[predictions["split"].eq(split)]
            sets = {
                model: tuple(rows[rows["model"].eq(model)].sort_values(["forecast_origin", "horizon"])[["forecast_origin", "target_timestamp", "horizon"]].itertuples(index=False, name=None))
                for model in ALL_MODELS
            }
            reference = next(iter(sets.values()))
            if set(rows["model"]) != ALL_MODELS or any(values != reference for values in sets.values()): failures.append(split)
        return not failures, "All recursive models and baselines use identical ordered eligible timestamps." if not failures else f"Timestamp mismatch: {failures}"

    safe("identical_primary_comparison_timestamps", check_common_timestamps)

    declared = pd.read_csv(FEATURE_LIST_FILE)
    feature_list = declared[declared["feature_group"].eq("autoregressive_demand")].sort_values("feature_order")["feature"].tolist()
    artifacts: dict[str, Any] = {}
    metadata_json: dict[str, dict[str, Any]] = {}
    for model, (artifact_name, metadata_name, _) in MODEL_FILES.items():
        artifacts[model] = joblib.load(MODEL_DIR / artifact_name)
        metadata_json[model] = json.loads((MODEL_DIR / metadata_name).read_text(encoding="utf-8"))

    def check_artifacts() -> tuple[bool, str]:
        failures = []
        for model, (artifact_name, _, _) in MODEL_FILES.items():
            meta = metadata_json[model]
            artifact_path = MODEL_DIR / artifact_name
            if meta["feature_list"] != feature_list or list(artifacts[model].feature_names_in_) != feature_list: failures.append(f"{model}: feature order")
            if meta["artifact_sha256"] != sha256(artifact_path): failures.append(f"{model}: artifact hash")
            if meta["feature_group"] != "autoregressive_demand": failures.append(f"{model}: group")
        return not failures, "Both saved autoregressive artifacts and exact ordered feature lists match metadata." if not failures else "; ".join(failures)

    safe("model_artifacts_and_features_match_metadata", check_artifacts)

    def check_recursive_reproduction() -> tuple[bool, str]:
        failures = []
        provenance_index = provenance.set_index(["model", "forecast_origin", "horizon"])
        prediction_index = predictions.set_index(["model", "forecast_origin", "horizon"])
        for origin_row in origins[origins["forecast_eligible"].astype(str).str.lower().eq("true")].itertuples(index=False):
            origin = origin_row.forecast_origin
            observed = {t: float(features.at[t, "demand_mwh"]) for t in pd.date_range(origin - pd.Timedelta(hours=167), origin, freq="h")}
            for model, fitted in artifacts.items():
                buffer = dict(observed)
                for horizon in range(1, 25):
                    target = origin + pd.Timedelta(hours=horizon)
                    values, expected_provenance = reproduce_features(target, origin, buffer)
                    row = pd.DataFrame([[values[name] for name in feature_list]], columns=feature_list)
                    reproduced = float(fitted.predict(row)[0])
                    saved_prediction = float(prediction_index.loc[(model, origin, horizon), "prediction_mwh"])
                    if not np.isclose(reproduced, saved_prediction, rtol=1e-10, atol=1e-7): failures.append(f"{model} {origin} h{horizon}: prediction")
                    saved_provenance = provenance_index.loc[(model, origin, horizon)]
                    for key, value in expected_provenance.items():
                        observed_value = saved_provenance[key]
                        if key.endswith("timestamp"):
                            equal = pd.Timestamp(observed_value) == value
                        elif key == "all_sources_valid":
                            equal = str(observed_value).lower() == str(value).lower()
                        else:
                            equal = int(observed_value) == int(value)
                        if not equal: failures.append(f"{model} {origin} h{horizon}: {key}")
                    if expected_provenance["latest_source_timestamp"] > origin and expected_provenance["lag_prediction_input_count"] + expected_provenance["rolling_prediction_input_count"] == 0:
                        failures.append(f"{model} {origin} h{horizon}: post-origin source not prediction")
                    buffer[target] = saved_prediction
        return not failures, "Every recursive feature and prediction reproduces using observed pre-origin demand or earlier same-origin predictions only." if not failures else f"Failures: {failures[:10]} (total {len(failures)})"

    safe("recursive_inputs_have_valid_provenance", check_recursive_reproduction)

    def check_baselines() -> tuple[bool, str]:
        failures = []
        for row in predictions[predictions["model"].isin(BASELINES)].itertuples(index=False):
            if row.model == "flat_persistence": source = row.forecast_origin
            elif row.model == "daily_seasonal_naive": source = row.target_timestamp - pd.Timedelta(hours=24)
            else: source = row.target_timestamp - pd.Timedelta(hours=168)
            expected = float(features.at[source, "demand_mwh"])
            if source > row.forecast_origin or row.prediction_mwh != expected: failures.append(f"{row.model} {row.forecast_origin} h{row.horizon}")
        return not failures, "Flat, daily, and weekly baselines exactly equal observed values known by each origin." if not failures else f"Failures: {failures[:10]}"

    safe("baselines_use_only_origin_known_values", check_baselines)

    def check_h1() -> tuple[bool, str]:
        saved = pd.read_csv(ONE_STEP_FILE)
        saved["period"] = pd.to_datetime(saved["period"], format="%Y-%m-%dT%H", errors="raise")
        saved = saved[saved["comparison_set_type"].eq("natural")]
        failures = []
        maximum = 0.0
        for model, (_, _, one_step_model) in MODEL_FILES.items():
            recursive = predictions[(predictions["model"].eq(model)) & predictions["horizon"].eq(1)]
            teacher = saved[saved["model"].eq(one_step_model)][["period", "split", "prediction_mwh"]]
            merged = recursive.merge(teacher, left_on=["target_timestamp", "split"], right_on=["period", "split"], suffixes=("_recursive", "_one_step"), validate="one_to_one")
            difference = np.abs(merged["prediction_mwh_recursive"] - merged["prediction_mwh_one_step"])
            maximum = max(maximum, float(difference.max()))
            if difference.max() > TOLERANCE: failures.append(model)
        saved_pass = consistency["passed"].astype(str).str.lower().eq("true").all() and consistency["maximum_absolute_difference_mwh"].max() <= TOLERANCE
        return not failures and saved_pass, f"Maximum independently reproduced horizon-1 difference: {maximum:.12g} MWh (tolerance {TOLERANCE})."

    safe("horizon_1_matches_legitimate_one_step", check_h1)

    metric_specs = {
        "metrics_overall.csv": ["split", "model", "model_label"],
        "metrics_by_horizon.csv": ["split", "model", "model_label", "horizon"],
        "metrics_by_forecast_date.csv": ["split", "model", "model_label", "forecast_date"],
        "metrics_by_target_utc_hour.csv": ["split", "model", "model_label", "target_utc_hour"],
        "metrics_by_calendar_month.csv": ["split", "model", "model_label", "calendar_month"],
    }

    def check_metrics() -> tuple[bool, str]:
        failures = []
        augmented = predictions.assign(
            target_utc_hour=predictions["target_timestamp"].dt.hour,
            calendar_month=predictions["target_timestamp"].dt.month,
        )
        for filename, keys in metric_specs.items():
            saved_table = pd.read_csv(TABLE_DIR / filename)
            for date_column in ["forecast_date"]:
                if date_column in saved_table: saved_table[date_column] = pd.to_datetime(saved_table[date_column], format="%Y-%m-%dT%H", errors="raise")
            saved_index = saved_table.set_index(keys)
            for values, group in augmented.groupby(keys, observed=True, sort=True):
                values = values if isinstance(values, tuple) else (values,)
                reproduced = calculate_metrics(group)
                row = saved_index.loc[values]
                for metric, value in reproduced.items():
                    if metric == "count": equal = int(row[metric]) == value
                    else: equal = np.isclose(float(row[metric]), float(value), rtol=1e-9, atol=1e-7, equal_nan=True)
                    if not equal: failures.append(f"{filename} {values} {metric}")
        return not failures, "Saved metric counts and values reproduce from the combined prediction table." if not failures else f"Failures: {failures[:10]}"

    safe("metric_counts_and_values_reproduce", check_metrics)

    def check_no_fit() -> tuple[bool, str]:
        tree = ast.parse(RUN_SCRIPT.read_text(encoding="utf-8"))
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform"}: forbidden.append(node.func.attr)
            if isinstance(node, ast.Name) and any(token in node.id.lower() for token in ["gridsearch", "randomizedsearch"]): forbidden.append(node.id)
        meta_ok = str(metadata.iloc[0]["models_fitted"]).lower() == "false" and str(metadata.iloc[0]["parameter_search_performed"]).lower() == "false"
        return not forbidden and meta_ok, "Run source contains no fit/search call and metadata records no fitting or parameter search." if not forbidden and meta_ok else f"Forbidden calls/names: {forbidden}; metadata_ok={meta_ok}"

    safe("no_model_fitting_or_parameter_search", check_no_fit)

    def check_hashes() -> tuple[bool, str]:
        failures = []
        for row in hashes.itertuples(index=False):
            path = ROOT / Path(row.file)
            current = sha256(path)
            if row.sha256_before != row.sha256_after or current != row.sha256_before or str(row.unchanged).lower() != "true": failures.append(row.file)
        return not failures, f"All {len(hashes)} upstream files retain identical before, after, and current SHA-256 hashes." if not failures else f"Changed: {failures}"

    safe("upstream_hashes_unchanged", check_hashes)
    safe("prediction_rows_are_eligible_and_available", lambda: (
        predictions["forecast_eligible"].astype(str).str.lower().eq("true").all()
        and predictions["target_available"].astype(str).str.lower().eq("true").all(),
        "Every primary-comparison row has an eligible forecast and available target."
    ))
    safe("origin_counts_match_prediction_rows", lambda: (
        all(
            len(predictions[predictions["split"].eq(split)])
            == int(origins[(origins["split"].eq(split)) & origins["forecast_eligible"].astype(str).str.lower().eq("true")].shape[0]) * 24 * len(ALL_MODELS)
            for split in EXPECTED_DAYS
        ),
        "Prediction row counts equal eligible origins x 24 horizons x five models."
    ))

    output = pd.DataFrame(results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    print(output.to_string(index=False))
    failed = output[~output["passed"]]
    if not failed.empty: raise SystemExit(f"Independent validation failed {len(failed)} check(s).")
    print(f"Independent validation passed all {len(output)} checks.")


if __name__ == "__main__":
    main()
