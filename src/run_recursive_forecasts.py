"""Evaluate saved autoregressive models with daily recursive 24-hour forecasts.

This script never fits a model. It reads the saved one-step artifacts, rebuilds
their features behind each forecast origin, and writes only recursive results.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / "data" / "processed" / "eia_ciso_hourly_features.csv"
DECLARATION_FILE = ROOT / "results" / "features" / "tables" / "feature_groups.csv"
FEATURE_LIST_FILE = ROOT / "results" / "models" / "tables" / "model_feature_lists.csv"
ONE_STEP_FILE = ROOT / "results" / "models" / "tables" / "one_step_predictions_all.csv"
MODEL_DIR = ROOT / "models" / "one_step"
RESULT_DIR = ROOT / "results" / "recursive"
TABLE_DIR = RESULT_DIR / "tables"
FIGURE_DIR = RESULT_DIR / "figures"
FINDINGS_FILE = RESULT_DIR / "recursive_forecasting_findings.md"
NOTEBOOK_FILE = ROOT / "notebooks" / "05_recursive_24_hour_forecasting.ipynb"

TARGET = "target_demand_mwh"
TOLERANCE = 1e-6
SPLIT_DAYS = {
    "validation": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30")),
    "test": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")),
}
FIXED_BOUNDS = {
    "train": (pd.Timestamp("2022-01-01T00"), pd.Timestamp("2023-12-31T23")),
    "validation": (pd.Timestamp("2024-01-01T00"), pd.Timestamp("2024-06-30T23")),
    "test": (pd.Timestamp("2024-07-01T00"), pd.Timestamp("2024-12-31T23")),
}
MODEL_SPECS = {
    "recursive_linear_regression": {
        "label": "Recursive Linear Regression",
        "artifact": "linear_regression__autoregressive_demand.joblib",
        "metadata": "linear_regression__autoregressive_demand.metadata.json",
        "one_step_model": "linear_regression__autoregressive_demand",
    },
    "recursive_xgboost": {
        "label": "Recursive XGBoost",
        "artifact": "xgboost__autoregressive_demand.joblib",
        "metadata": "xgboost__autoregressive_demand.metadata.json",
        "one_step_model": "xgboost__autoregressive_demand",
    },
}
BASELINES = {
    "flat_persistence": "Flat persistence",
    "daily_seasonal_naive": "Daily seasonal naive",
    "weekly_seasonal_naive": "Weekly seasonal naive",
}
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS = [24, 168]
ROLLING_STATS = ["mean", "std", "min", "max"]
CALENDAR_FEATURES = [
    "year", "month", "day", "day_of_year", "hour_utc", "day_of_week_utc",
    "is_weekend_utc", "hour_sin", "hour_cos", "day_of_week_sin",
    "day_of_week_cos", "day_of_year_sin", "day_of_year_cos",
]
UPSTREAM_FILES = [
    FEATURE_FILE, DECLARATION_FILE, FEATURE_LIST_FILE, ONE_STEP_FILE,
    MODEL_DIR / MODEL_SPECS["recursive_linear_regression"]["artifact"],
    MODEL_DIR / MODEL_SPECS["recursive_linear_regression"]["metadata"],
    MODEL_DIR / MODEL_SPECS["recursive_xgboost"]["artifact"],
    MODEL_DIR / MODEL_SPECS["recursive_xgboost"]["metadata"],
    MODEL_DIR / "run_metadata.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(
        TABLE_DIR / name, index=False, date_format="%Y-%m-%dT%H",
        float_format="%.10f",
    )


def markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    """Render a compact Markdown table without pandas' optional tabulate package."""
    display = frame.copy()
    for column in display.select_dtypes(include=["float", "float64"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}"
        )
    display = display.astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def calendar_values(timestamp: pd.Timestamp) -> dict[str, float | int | bool]:
    hour_angle = timestamp.hour * (2.0 * math.pi / 24.0)
    dow_angle = timestamp.dayofweek * (2.0 * math.pi / 7.0)
    days = 366 if timestamp.is_leap_year else 365
    doy_angle = (timestamp.dayofyear - 1) * (2.0 * math.pi / days)
    return {
        "year": timestamp.year,
        "month": timestamp.month,
        "day": timestamp.day,
        "day_of_year": timestamp.dayofyear,
        "hour_utc": timestamp.hour,
        "day_of_week_utc": timestamp.dayofweek,
        "is_weekend_utc": timestamp.dayofweek >= 5,
        "hour_sin": math.sin(hour_angle),
        "hour_cos": math.cos(hour_angle),
        "day_of_week_sin": math.sin(dow_angle),
        "day_of_week_cos": math.cos(dow_angle),
        "day_of_year_sin": math.sin(doy_angle),
        "day_of_year_cos": math.cos(doy_angle),
    }


def load_inputs() -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[str]]]:
    frame = pd.read_csv(FEATURE_FILE, low_memory=False)
    frame["period"] = pd.to_datetime(frame["period"], format="%Y-%m-%dT%H", errors="raise")
    if frame["period"].duplicated().any() or not frame["period"].is_monotonic_increasing:
        raise ValueError("Feature-master timestamps must be unique and chronological.")
    expected = pd.date_range(frame["period"].iloc[0], frame["period"].iloc[-1], freq="h")
    if not frame["period"].equals(pd.Series(expected)):
        raise ValueError("Feature master must retain its continuous hourly timeline.")
    frame["demand_mwh"] = pd.to_numeric(frame["demand_mwh"], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    truth = frame["target_available"].astype(str).str.lower().map({"true": True, "false": False})
    if truth.isna().any():
        raise ValueError("target_available contains an unexpected value.")
    frame["target_available"] = truth.astype(bool)
    frame = frame.set_index("period", drop=False)

    declared = pd.read_csv(FEATURE_LIST_FILE)
    declared_features = declared.loc[
        declared["feature_group"].eq("autoregressive_demand")
    ].sort_values("feature_order")["feature"].tolist()
    artifacts: dict[str, Any] = {}
    feature_lists: dict[str, list[str]] = {}
    for model, spec in MODEL_SPECS.items():
        metadata = json.loads((MODEL_DIR / spec["metadata"]).read_text(encoding="utf-8"))
        artifact_path = MODEL_DIR / spec["artifact"]
        if metadata["feature_group"] != "autoregressive_demand":
            raise ValueError(f"{model} is not the saved autoregressive variant.")
        if metadata["feature_list"] != declared_features:
            raise ValueError(f"{model} metadata feature list/order differs from declaration.")
        if metadata["artifact_sha256"] != sha256(artifact_path):
            raise ValueError(f"{model} artifact hash differs from metadata.")
        fitted = joblib.load(artifact_path)
        if list(fitted.feature_names_in_) != declared_features:
            raise ValueError(f"{model} artifact feature order differs from metadata.")
        artifacts[model] = fitted
        feature_lists[model] = list(metadata["feature_list"])
    if feature_lists["recursive_linear_regression"] != feature_lists["recursive_xgboost"]:
        raise ValueError("Selected model feature lists differ.")
    return frame, artifacts, feature_lists


def origin_eligibility(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    timeline = set(frame.index)
    for split, (start_day, end_day) in SPLIT_DAYS.items():
        for forecast_date in pd.date_range(start_day, end_day, freq="D"):
            origin = forecast_date - pd.Timedelta(hours=1)
            targets = [origin + pd.Timedelta(hours=h) for h in range(1, 25)]
            history = [origin - pd.Timedelta(hours=h) for h in range(0, 168)]
            reasons: list[str] = []
            absent_targets = [t for t in targets if t not in timeline]
            absent_history = [t for t in history if t not in timeline]
            if absent_targets:
                reasons.append(f"{len(absent_targets)} target timestamp(s) absent")
            if absent_history:
                reasons.append(f"{len(absent_history)} required history timestamp(s) absent")
            if not absent_targets:
                target_rows = frame.loc[targets]
                unavailable = (~target_rows["target_available"] | ~np.isfinite(target_rows[TARGET])).sum()
                if unavailable:
                    bad = target_rows.loc[
                        ~target_rows["target_available"] | ~np.isfinite(target_rows[TARGET]), "period"
                    ].dt.strftime("%Y-%m-%dT%H").tolist()
                    reasons.append(f"{unavailable} unavailable target(s): {', '.join(bad)}")
            if not absent_history:
                history_values = frame.loc[history, "demand_mwh"]
                missing = history_values[~np.isfinite(history_values)]
                if len(missing):
                    reasons.append(
                        f"{len(missing)} missing required observed demand value(s): "
                        + ", ".join(t.strftime("%Y-%m-%dT%H") for t in missing.index)
                    )
            rows.append({
                "split": split,
                "forecast_date": forecast_date,
                "forecast_origin": origin,
                "expected_target_count": 24,
                "forecast_eligible": not reasons,
                "ineligibility_reason": "; ".join(reasons),
            })
    return pd.DataFrame(rows)


def history_feature_values(
    target: pd.Timestamp,
    origin: pd.Timestamp,
    buffer: dict[pd.Timestamp, float],
) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    values: dict[str, float] = {}
    audit: list[dict[str, Any]] = []
    lag_observed = lag_predicted = rolling_observed = rolling_predicted = 0
    earliest: pd.Timestamp | None = None
    latest: pd.Timestamp | None = None

    for hours in LAG_HOURS:
        source = target - pd.Timedelta(hours=hours)
        source_type = "observed_pre_origin" if source <= origin else "recursive_prediction"
        values[f"demand_lag_{hours}h"] = float(buffer[source])
        lag_observed += int(source_type == "observed_pre_origin")
        lag_predicted += int(source_type == "recursive_prediction")
        audit.append({
            "feature": f"demand_lag_{hours}h", "source_start": source,
            "source_end": source, "source_count": 1,
            "observed_pre_origin_count": int(source_type == "observed_pre_origin"),
            "recursive_prediction_count": int(source_type == "recursive_prediction"),
        })
        earliest = source if earliest is None or source < earliest else earliest
        latest = source if latest is None or source > latest else latest

    for window in ROLLING_WINDOWS:
        source_times = [target - pd.Timedelta(hours=h) for h in range(1, window + 1)]
        source_values = np.array([buffer[t] for t in source_times], dtype=float)
        observed = sum(t <= origin for t in source_times)
        predicted = window - observed
        rolling_observed += observed * len(ROLLING_STATS)
        rolling_predicted += predicted * len(ROLLING_STATS)
        stats = {
            "mean": float(np.mean(source_values)),
            "std": float(np.std(source_values, ddof=1)),
            "min": float(np.min(source_values)),
            "max": float(np.max(source_values)),
        }
        for stat, value in stats.items():
            feature = f"demand_rolling_{window}h_{stat}"
            values[feature] = value
            audit.append({
                "feature": feature, "source_start": min(source_times),
                "source_end": max(source_times), "source_count": window,
                "observed_pre_origin_count": observed,
                "recursive_prediction_count": predicted,
            })
        earliest = min(source_times) if earliest is None or min(source_times) < earliest else earliest
        latest = max(source_times) if latest is None or max(source_times) > latest else latest

    provenance = {
        "lag_observed_input_count": lag_observed,
        "lag_prediction_input_count": lag_predicted,
        "rolling_observed_input_count": rolling_observed,
        "rolling_prediction_input_count": rolling_predicted,
        "earliest_source_timestamp": earliest,
        "latest_source_timestamp": latest,
        "all_sources_valid": latest is not None and latest < target,
    }
    return values, provenance, audit


def build_predictions(
    frame: pd.DataFrame,
    artifacts: dict[str, Any],
    feature_lists: dict[str, list[str]],
    origins: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    prediction_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    audit_candidates: list[dict[str, Any]] = []
    eligible = origins[origins["forecast_eligible"]].copy()
    representative_origin_keys = {
        split: group.sort_values("forecast_date").iloc[(len(group) - 1) // 2]["forecast_origin"]
        for split, group in eligible.groupby("split", sort=False)
    }

    for origin_row in eligible.itertuples(index=False):
        origin = origin_row.forecast_origin
        forecast_date = origin_row.forecast_date
        actual_history = {
            timestamp: float(frame.at[timestamp, "demand_mwh"])
            for timestamp in pd.date_range(origin - pd.Timedelta(hours=167), origin, freq="h")
        }
        for model, fitted in artifacts.items():
            buffer = dict(actual_history)
            for horizon in range(1, 25):
                target = origin + pd.Timedelta(hours=horizon)
                history_values, provenance, audit = history_feature_values(target, origin, buffer)
                feature_values = {**calendar_values(target), **history_values}
                feature_frame = pd.DataFrame(
                    [[feature_values[name] for name in feature_lists[model]]],
                    columns=feature_lists[model],
                )
                prediction = float(fitted.predict(feature_frame)[0])
                buffer[target] = prediction
                actual = float(frame.at[target, TARGET])
                prediction_rows.append(prediction_record(
                    origin, forecast_date, target, origin_row.split, horizon, model,
                    MODEL_SPECS[model]["label"], actual, prediction,
                    bool(frame.at[target, "target_available"]), True,
                ))
                provenance_rows.append({
                    "forecast_origin": origin, "target_timestamp": target,
                    "split": origin_row.split, "horizon": horizon, "model": model,
                    **provenance,
                })
                if (
                    origin == representative_origin_keys[origin_row.split]
                    and horizon in {1, 6, 12, 18, 24}
                ):
                    for item in audit:
                        audit_candidates.append({
                            "forecast_origin": origin, "forecast_date": forecast_date,
                            "target_timestamp": target, "split": origin_row.split,
                            "horizon": horizon, "model": model, **item,
                        })

        origin_value = float(frame.at[origin, "demand_mwh"])
        for horizon in range(1, 25):
            target = origin + pd.Timedelta(hours=horizon)
            actual = float(frame.at[target, TARGET])
            baseline_values = {
                "flat_persistence": origin_value,
                "daily_seasonal_naive": float(frame.at[target - pd.Timedelta(hours=24), "demand_mwh"]),
                "weekly_seasonal_naive": float(frame.at[target - pd.Timedelta(hours=168), "demand_mwh"]),
            }
            for model, prediction in baseline_values.items():
                prediction_rows.append(prediction_record(
                    origin, forecast_date, target, origin_row.split, horizon, model,
                    BASELINES[model], actual, prediction,
                    bool(frame.at[target, "target_available"]), True,
                ))
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["split", "forecast_origin", "horizon", "model"]
    ).reset_index(drop=True)
    provenance = pd.DataFrame(provenance_rows).sort_values(
        ["split", "forecast_origin", "model", "horizon"]
    ).reset_index(drop=True)
    return predictions, provenance, audit_candidates


def prediction_record(
    origin: pd.Timestamp, forecast_date: pd.Timestamp, target: pd.Timestamp,
    split: str, horizon: int, model: str, label: str, actual: float,
    prediction: float, target_available: bool, eligible: bool,
) -> dict[str, Any]:
    error = prediction - actual
    return {
        "forecast_origin": origin,
        "forecast_date": forecast_date,
        "target_timestamp": target,
        "split": split,
        "horizon": horizon,
        "model": model,
        "model_label": label,
        "actual_demand_mwh": actual,
        "prediction_mwh": prediction,
        "error_mwh": error,
        "absolute_error_mwh": abs(error),
        "percentage_error_pct": 100 * error / abs(actual) if actual != 0 else np.nan,
        "target_available": target_available,
        "forecast_eligible": eligible,
    }


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
        "count": int(len(group)),
        "mae_mwh": float(absolute.mean()),
        "rmse_mwh": float(np.sqrt(np.mean(error ** 2))),
        "mape_pct": float(100 * np.mean(absolute[nonzero] / np.abs(actual[nonzero]))) if nonzero.any() else np.nan,
        "smape_pct": float(100 * np.mean(2 * absolute[smape_valid] / denominator[smape_valid])) if smape_valid.any() else np.nan,
        "bias_mwh": float(error.mean()),
        "r_squared": float(1 - np.sum(error ** 2) / r2_denominator) if r2_denominator else np.nan,
    }


def grouped_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(keys, observed=True, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        rows.append({**dict(zip(keys, values)), **calculate_metrics(group)})
    return pd.DataFrame(rows)


def analyses(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    outputs["metrics_overall.csv"] = grouped_metrics(predictions, ["split", "model", "model_label"])
    outputs["metrics_by_horizon.csv"] = grouped_metrics(predictions, ["split", "model", "model_label", "horizon"])
    outputs["metrics_selected_horizons.csv"] = outputs["metrics_by_horizon.csv"].query("horizon in [1, 6, 12, 18, 24]").copy()
    outputs["metrics_by_forecast_date.csv"] = grouped_metrics(predictions, ["split", "model", "model_label", "forecast_date"])
    hourly = predictions.assign(target_utc_hour=predictions["target_timestamp"].dt.hour)
    outputs["metrics_by_target_utc_hour.csv"] = grouped_metrics(hourly, ["split", "model", "model_label", "target_utc_hour"])
    monthly = predictions.assign(calendar_month=predictions["target_timestamp"].dt.month)
    outputs["metrics_by_calendar_month.csv"] = grouped_metrics(monthly, ["split", "model", "model_label", "calendar_month"])

    thresholds = predictions.drop_duplicates(["split", "target_timestamp"])
    threshold_rows = []
    segment_frames = []
    peak_frames = []
    for split, group in thresholds.groupby("split", sort=False):
        bottom = float(group["actual_demand_mwh"].quantile(0.10))
        top = float(group["actual_demand_mwh"].quantile(0.90))
        threshold_rows.append({"split": split, "bottom_decile_threshold_mwh": bottom, "top_decile_threshold_mwh": top})
        split_rows = predictions[predictions["split"].eq(split)]
        bottom_rows = split_rows[split_rows["actual_demand_mwh"] <= bottom].assign(demand_segment="bottom_10_pct")
        top_rows = split_rows[split_rows["actual_demand_mwh"] >= top].assign(demand_segment="top_10_pct")
        segment_frames.extend([bottom_rows, top_rows])
        peak_frames.append(top_rows)
    outputs["demand_percentile_thresholds.csv"] = pd.DataFrame(threshold_rows)
    segments = pd.concat(segment_frames, ignore_index=True)
    outputs["metrics_demand_segments.csv"] = grouped_metrics(segments, ["split", "demand_segment", "model", "model_label"])
    peaks = pd.concat(peak_frames, ignore_index=True)
    outputs["peak_metrics_by_horizon.csv"] = grouped_metrics(peaks, ["split", "model", "model_label", "horizon"])

    daily = grouped_metrics(predictions, ["split", "model", "model_label", "forecast_date"])
    outputs["daily_forecast_metrics.csv"] = daily
    largest = daily.sort_values(["split", "model", "mae_mwh", "forecast_date"], ascending=[True, True, False, True]).groupby(["split", "model"], sort=False).head(10)
    outputs["largest_error_days.csv"] = largest.reset_index(drop=True)

    horizon = outputs["metrics_by_horizon.csv"]
    propagation_rows = []
    for (split, model, label), group in horizon.groupby(["split", "model", "model_label"], sort=False):
        ordered = group.sort_values("horizon")
        h1 = ordered.iloc[0]
        h24 = ordered.iloc[-1]
        diffs = ordered["mae_mwh"].diff().dropna()
        propagation_rows.append({
            "split": split, "model": model, "model_label": label,
            "horizon_1_mae_mwh": h1["mae_mwh"], "horizon_24_mae_mwh": h24["mae_mwh"],
            "mae_change_h1_to_h24_mwh": h24["mae_mwh"] - h1["mae_mwh"],
            "mae_change_h1_to_h24_pct": 100 * (h24["mae_mwh"] / h1["mae_mwh"] - 1),
            "rmse_change_h1_to_h24_mwh": h24["rmse_mwh"] - h1["rmse_mwh"],
            "bias_change_h1_to_h24_mwh": h24["bias_mwh"] - h1["bias_mwh"],
            "worst_mae_horizon": int(ordered.loc[ordered["mae_mwh"].idxmax(), "horizon"]),
            "mae_increases_every_horizon": bool((diffs >= 0).all()),
            "horizon_pattern": "steady nondecreasing" if (diffs >= 0).all() else "irregular",
        })
    outputs["error_propagation_summary.csv"] = pd.DataFrame(propagation_rows)
    return outputs


def one_step_comparisons(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    saved = pd.read_csv(ONE_STEP_FILE)
    saved["period"] = pd.to_datetime(saved["period"], format="%Y-%m-%dT%H", errors="raise")
    # Natural one-step rows are the exact selected model's legitimate timestamps.
    # The broader one-step "common" subset also required unrelated renewable
    # variants and therefore omits otherwise valid autoregressive timestamps.
    saved = saved[saved["comparison_set_type"].eq("natural")].copy()
    consistency_rows = []
    teacher_rows = []
    for recursive_model, spec in MODEL_SPECS.items():
        teacher = saved[saved["model"].eq(spec["one_step_model"])][
            ["period", "split", "prediction_mwh", "absolute_error_mwh"]
        ].rename(columns={"prediction_mwh": "teacher_prediction_mwh", "absolute_error_mwh": "teacher_absolute_error_mwh"})
        recursive = predictions[predictions["model"].eq(recursive_model)]
        merged = recursive.merge(
            teacher, left_on=["target_timestamp", "split"], right_on=["period", "split"],
            how="left", validate="one_to_one",
        )
        for split, group in merged[merged["horizon"].eq(1)].groupby("split", sort=False):
            difference = np.abs(group["prediction_mwh"] - group["teacher_prediction_mwh"])
            consistency_rows.append({
                "split": split, "model": recursive_model, "comparison_count": int(len(group)),
                "maximum_absolute_difference_mwh": float(difference.max()),
                "tolerance_mwh": TOLERANCE, "passed": bool(difference.max() <= TOLERANCE),
            })
        teacher_summary = grouped_metrics(
            merged.rename(columns={
                "prediction_mwh": "recursive_prediction_mwh",
                "teacher_prediction_mwh": "prediction_mwh",
            }),
            ["split", "horizon"],
        ).rename(columns={
            "mae_mwh": "teacher_forced_mae_mwh", "rmse_mwh": "teacher_forced_rmse_mwh",
            "bias_mwh": "teacher_forced_bias_mwh",
        })[["split", "horizon", "count", "teacher_forced_mae_mwh", "teacher_forced_rmse_mwh", "teacher_forced_bias_mwh"]]
        recursive_summary = grouped_metrics(recursive, ["split", "horizon"])[["split", "horizon", "mae_mwh", "rmse_mwh", "bias_mwh"]].rename(columns={
            "mae_mwh": "recursive_mae_mwh", "rmse_mwh": "recursive_rmse_mwh", "bias_mwh": "recursive_bias_mwh",
        })
        combined = teacher_summary.merge(recursive_summary, on=["split", "horizon"], validate="one_to_one")
        combined.insert(2, "model", recursive_model)
        teacher_rows.append(combined)
    consistency = pd.DataFrame(consistency_rows)
    if not consistency["passed"].all():
        raise RuntimeError("Horizon-1 recursive predictions differ from saved one-step outputs.")
    return consistency, pd.concat(teacher_rows, ignore_index=True)


def model_comparisons(overall: pd.DataFrame) -> pd.DataFrame:
    recursive = overall[overall["model"].isin(MODEL_SPECS)].copy()
    rows = []
    for item in recursive.itertuples(index=False):
        split_baselines = overall[(overall["split"].eq(item.split)) & overall["model"].isin(BASELINES)]
        for baseline in split_baselines.itertuples(index=False):
            rows.append({
                "split": item.split, "recursive_model": item.model,
                "recursive_model_label": item.model_label, "baseline": baseline.model,
                "baseline_label": baseline.model_label, "recursive_mae_mwh": item.mae_mwh,
                "baseline_mae_mwh": baseline.mae_mwh,
                "mae_improvement_vs_baseline_pct": 100 * (baseline.mae_mwh - item.mae_mwh) / baseline.mae_mwh,
            })
    return pd.DataFrame(rows)


def select_days(daily: pd.DataFrame, best_model: str) -> pd.DataFrame:
    rows = []
    for split in ["validation", "test"]:
        group = daily[(daily["split"].eq(split)) & (daily["model"].eq(best_model))].sort_values("forecast_date")
        median = group["mae_mwh"].median()
        representative = group.assign(distance=(group["mae_mwh"] - median).abs()).sort_values(["distance", "forecast_date"]).iloc[0]
        rows.append({"split": split, "selection": "representative_median_daily_mae", "model": best_model, "forecast_date": representative["forecast_date"], "daily_mae_mwh": representative["mae_mwh"]})
    test = daily[(daily["split"].eq("test")) & (daily["model"].eq(best_model))].sort_values(["mae_mwh", "forecast_date"], ascending=[False, True]).iloc[0]
    rows.append({"split": "test", "selection": "highest_error_test_day", "model": best_model, "forecast_date": test["forecast_date"], "daily_mae_mwh": test["mae_mwh"]})
    return pd.DataFrame(rows)


def plot_lines(table: pd.DataFrame, value: str, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, group in table.groupby("model_label", sort=False):
        ax.plot(group["horizon"], group[value], marker="o", markersize=3, linewidth=1.8, label=label)
    ax.set(title=title, xlabel="Forecast horizon (hours)", ylabel=ylabel)
    ax.set_xticks(range(1, 25))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def create_figures(predictions: pd.DataFrame, outputs: dict[str, pd.DataFrame], selected_days: pd.DataFrame, best_model: str, teacher: pd.DataFrame) -> None:
    overall = outputs["metrics_overall.csv"]
    horizon = outputs["metrics_by_horizon.csv"]
    for number, split in [(1, "validation"), (2, "test")]:
        table = overall[overall["split"].eq(split)].sort_values("mae_mwh")
        fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(table["model_label"], table["mae_mwh"])
        ax.set(title=f"{split.title()} overall recursive 24-hour MAE", ylabel="MAE (MWh)")
        ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(FIGURE_DIR / f"{number:02d}_{split}_overall_mae.png", dpi=160); plt.close(fig)
    plot_lines(horizon[horizon["split"].eq("validation")], "mae_mwh", "Validation MAE by forecast horizon", "MAE (MWh)", FIGURE_DIR / "03_validation_mae_by_horizon.png")
    plot_lines(horizon[horizon["split"].eq("test")], "mae_mwh", "Test MAE by forecast horizon", "MAE (MWh)", FIGURE_DIR / "04_test_mae_by_horizon.png")
    plot_lines(horizon[horizon["split"].eq("test")], "rmse_mwh", "Test RMSE by forecast horizon", "RMSE (MWh)", FIGURE_DIR / "05_test_rmse_by_horizon.png")
    plot_lines(horizon[horizon["split"].eq("test")], "bias_mwh", "Test bias by forecast horizon", "Bias: prediction - actual (MWh)", FIGURE_DIR / "06_test_bias_by_horizon.png")

    day_specs = [
        ("validation", "representative_median_daily_mae", "07_representative_validation_day.png", "Representative validation day"),
        ("test", "representative_median_daily_mae", "08_representative_test_day.png", "Representative test day"),
        ("test", "highest_error_test_day", "09_highest_error_test_day.png", "Highest-error test day"),
    ]
    for split, selection, filename, title in day_specs:
        day = pd.Timestamp(selected_days[(selected_days["split"].eq(split)) & selected_days["selection"].eq(selection)].iloc[0]["forecast_date"])
        rows = predictions[(predictions["split"].eq(split)) & predictions["forecast_date"].eq(day)]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        actual = rows.drop_duplicates("target_timestamp").sort_values("target_timestamp")
        ax.plot(actual["target_timestamp"], actual["actual_demand_mwh"], color="black", linewidth=2.5, label="Actual demand")
        for label, group in rows.groupby("model_label", sort=False):
            ax.plot(group["target_timestamp"], group["prediction_mwh"], linewidth=1.5, label=label)
        ax.set(title=f"{title}: {day:%Y-%m-%d} UTC", xlabel="Target timestamp (UTC)", ylabel="Demand (MWh)")
        ax.tick_params(axis="x", rotation=30); ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(FIGURE_DIR / filename, dpi=160); plt.close(fig)

    daily = outputs["daily_forecast_metrics.csv"]
    fig, ax = plt.subplots(figsize=(9, 5));
    for label, group in daily[daily["split"].eq("test")].groupby("model_label", sort=False):
        ax.hist(group["mae_mwh"], bins=24, alpha=0.35, label=label)
    ax.set(title="Distribution of daily test MAE", xlabel="Daily 24-hour MAE (MWh)", ylabel="Forecast days")
    ax.legend(fontsize=8); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(FIGURE_DIR / "10_daily_test_mae_distribution.png", dpi=160); plt.close(fig)

    peak = outputs["metrics_demand_segments.csv"].query("split == 'test' and demand_segment == 'top_10_pct'").sort_values("mae_mwh")
    fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(peak["model_label"], peak["mae_mwh"])
    ax.set(title="Test top-decile demand performance", ylabel="MAE (MWh)"); ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "11_test_peak_demand_performance.png", dpi=160); plt.close(fig)

    for number, key, x, title, filename in [
        (12, "metrics_by_target_utc_hour.csv", "target_utc_hour", "Test MAE by target UTC hour", "12_test_mae_by_target_utc_hour.png"),
        (13, "metrics_by_calendar_month.csv", "calendar_month", "Test MAE by calendar month", "13_test_mae_by_month.png"),
    ]:
        table = outputs[key].query("split == 'test'")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for label, group in table.groupby("model_label", sort=False): ax.plot(group[x], group["mae_mwh"], marker="o", label=label)
        ax.set(title=title, xlabel=x.replace("_", " ").title() + " (UTC)", ylabel="MAE (MWh)"); ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(FIGURE_DIR / filename, dpi=160); plt.close(fig)

    selected = horizon[(horizon["split"].eq("test")) & horizon["horizon"].isin([1, 24])].copy()
    pivot = selected.pivot(index="model_label", columns="horizon", values="mae_mwh").sort_values(24)
    fig, ax = plt.subplots(figsize=(9, 5)); pivot.rename(columns={1: "Horizon 1", 24: "Horizon 24"}).plot.bar(ax=ax)
    ax.set(title="Test horizon-1 versus horizon-24 MAE", xlabel="", ylabel="MAE (MWh)"); ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "14_horizon_1_vs_24_mae.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model, group in teacher[teacher["split"].eq("test")].groupby("model", sort=False):
        label = MODEL_SPECS[model]["label"]
        ax.plot(group["horizon"], group["recursive_mae_mwh"], marker="o", markersize=3, label=f"{label} recursive")
        ax.plot(group["horizon"], group["teacher_forced_mae_mwh"], linestyle="--", label=f"{label} teacher-forced one-step")
    ax.set(title="Recursive versus teacher-forced one-step test MAE", xlabel="Forecast horizon (hours)", ylabel="MAE (MWh)"); ax.set_xticks(range(1, 25)); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "15_recursive_vs_teacher_forced_error.png", dpi=160); plt.close(fig)


def write_findings(origins: pd.DataFrame, outputs: dict[str, pd.DataFrame], comparisons: pd.DataFrame, consistency: pd.DataFrame, selected_days: pd.DataFrame, best_model: str) -> None:
    overall = outputs["metrics_overall.csv"]
    horizon = outputs["metrics_by_horizon.csv"]
    propagation = outputs["error_propagation_summary.csv"]
    peaks = outputs["metrics_demand_segments.csv"].query("demand_segment == 'top_10_pct'")
    lines = [
        "# Recursive 24-Hour Forecasting Findings", "",
        "These are true daily rolling-origin 24-hour results. They are not the earlier teacher-forced one-step metrics.", "",
        "## Eligibility", "",
    ]
    for split in ["validation", "test"]:
        subset = origins[origins["split"].eq(split)]
        eligible = int(subset["forecast_eligible"].sum())
        lines.append(f"- {split.title()}: {eligible} eligible origins and {eligible * 24:,} hourly targets; {len(subset) - eligible} ineligible origins.")
    lines += ["", "## Overall metrics", "", markdown_table(overall), "", f"The best recursive model by validation all-horizon MAE is **{MODEL_SPECS[best_model]['label']}**.", "", "## Baseline comparison", "", markdown_table(comparisons), "", "Positive improvement percentages mean the recursive model has lower MAE; negative values mean degradation.", "", "## Selected horizons", "", markdown_table(outputs["metrics_selected_horizons.csv"]), "", "Complete horizon 1-24 results are in `tables/metrics_by_horizon.csv`.", "", "## Error propagation", "", markdown_table(propagation), ""]
    lines.append("Horizon behaviour is described as steady only when MAE never decreases between adjacent horizons; otherwise it is reported as irregular. No propagation claim is assumed in advance.")
    lines += ["", "## Peak demand", "", markdown_table(outputs["demand_percentile_thresholds.csv"]), "", markdown_table(peaks), ""]
    for row in peaks.itertuples(index=False):
        direction = "underpredicted" if row.bias_mwh < 0 else "overpredicted"
        lines.append(f"- {row.split.title()} {row.model_label}: peak demand was {direction} on average (bias {row.bias_mwh:.2f} MWh).")
    validation = overall[overall["split"].eq("validation")].set_index("model")
    test = overall[overall["split"].eq("test")].set_index("model")
    lines += ["", "## Validation-to-test change", ""]
    for model in MODEL_SPECS:
        change = 100 * (test.at[model, "mae_mwh"] / validation.at[model, "mae_mwh"] - 1)
        lines.append(f"- {MODEL_SPECS[model]['label']}: test MAE changed by {change:+.2f}% from validation.")
    stability = propagation[propagation["model"].isin(MODEL_SPECS)].copy()
    stability["absolute_mae_change"] = stability["mae_change_h1_to_h24_mwh"].abs()
    stable_by_split = stability.sort_values(["split", "absolute_mae_change"]).groupby("split").first().reset_index()
    lines += ["", "## Horizon-1 consistency", "", markdown_table(consistency, decimals=10), "", "Every horizon-1 comparison passed the numerical tolerance. Horizons 2-24 intentionally use recursive predictions and need not match teacher-forced outputs.", "", "## Largest-error days", "", markdown_table(outputs["largest_error_days.csv"].groupby(["split", "model"], sort=False).head(3)), "", "## Stability and application evidence", ""]
    for row in stable_by_split.itertuples(index=False): lines.append(f"- In {row.split}, {MODEL_SPECS.get(row.model, {'label': row.model})['label']} had the smaller absolute MAE change from horizon 1 to 24 among the recursive models.")
    best_test = test.loc[best_model]
    best_baseline = test.loc[list(BASELINES)].sort_values("mae_mwh").iloc[0]
    if best_test["mae_mwh"] < best_baseline["mae_mwh"]:
        lines.append(f"- The evidence supports carrying {MODEL_SPECS[best_model]['label']} into the Streamlit application as the recursive demand model: it was selected on validation and its test MAE ({best_test['mae_mwh']:.2f} MWh) beat the strongest baseline ({best_baseline['mae_mwh']:.2f} MWh). Operational monitoring is still needed.")
    else:
        lines.append(f"- The evidence does not yet support replacing the strongest simple baseline in the Streamlit application: the selected recursive model test MAE was {best_test['mae_mwh']:.2f} MWh versus {best_baseline['mae_mwh']:.2f} MWh for the best baseline.")
    lines += ["", "Representative days were selected deterministically as the eligible day closest to the selected model's median daily MAE. The highest-error test day is the maximum daily MAE, with earliest date used for ties.", "", "Selected days:", "", markdown_table(selected_days), ""]
    FINDINGS_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    hashes_before = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in UPSTREAM_FILES}
    frame, artifacts, feature_lists = load_inputs()
    origins = origin_eligibility(frame)
    predictions, provenance, audit = build_predictions(frame, artifacts, feature_lists, origins)
    outputs = analyses(predictions)
    consistency, teacher = one_step_comparisons(predictions)
    overall = outputs["metrics_overall.csv"]
    recursive_validation = overall[(overall["split"].eq("validation")) & overall["model"].isin(MODEL_SPECS)].sort_values(["mae_mwh", "model"])
    best_model = str(recursive_validation.iloc[0]["model"])
    comparisons = model_comparisons(overall)
    selected_days = select_days(outputs["daily_forecast_metrics.csv"], best_model)

    save_csv(origins, "forecast_origin_eligibility.csv")
    save_csv(predictions, "recursive_predictions.csv")
    save_csv(provenance, "recursive_feature_provenance.csv")
    save_csv(pd.DataFrame(audit), "representative_feature_provenance_audit.csv")
    for name, table in outputs.items(): save_csv(table, name)
    save_csv(consistency, "horizon_1_consistency.csv")
    save_csv(teacher, "recursive_vs_teacher_forced_by_horizon.csv")
    save_csv(comparisons, "baseline_comparisons.csv")
    save_csv(selected_days, "representative_days.csv")

    hashes_after = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in UPSTREAM_FILES}
    hash_table = pd.DataFrame([
        {"file": name, "sha256_before": value, "sha256_after": hashes_after[name], "unchanged": value == hashes_after[name]}
        for name, value in hashes_before.items()
    ])
    save_csv(hash_table, "upstream_file_hashes.csv")
    if not hash_table["unchanged"].all(): raise RuntimeError("An upstream file changed during recursive evaluation.")
    metadata = pd.DataFrame([{
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "forecast_type": "daily_rolling_origin_recursive_24_hour",
        "models_fitted": False, "parameter_search_performed": False,
        "validation_start": FIXED_BOUNDS["validation"][0], "validation_end": FIXED_BOUNDS["validation"][1],
        "test_start": FIXED_BOUNDS["test"][0], "test_end": FIXED_BOUNDS["test"][1],
        "best_recursive_model_by_validation_mae": best_model,
        "horizon_1_tolerance_mwh": TOLERANCE,
    }])
    save_csv(metadata, "recursive_run_metadata.csv")
    create_figures(predictions, outputs, selected_days, best_model, teacher)
    write_findings(origins, outputs, comparisons, consistency, selected_days, best_model)
    print(f"Completed recursive evaluation with {int(origins['forecast_eligible'].sum())} eligible origins.")
    print(f"Best recursive model by validation MAE: {best_model}")
    print(f"All {len(hash_table)} upstream hashes unchanged.")


if __name__ == "__main__":
    main()
