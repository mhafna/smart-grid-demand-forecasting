"""Train leakage-safe one-hour-ahead Linear Regression and XGBoost models.

The feature master, feature declarations, split policy, and baseline outputs are
read-only. This script writes only under results/models/ and models/one_step/.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / "data" / "processed" / "eia_ciso_hourly_features.csv"
FEATURE_GROUP_FILE = ROOT / "results" / "features" / "tables" / "feature_groups.csv"
RESULT_DIR = ROOT / "results" / "models"
TABLE_DIR = RESULT_DIR / "tables"
FIGURE_DIR = RESULT_DIR / "figures"
MODEL_DIR = ROOT / "models" / "one_step"
FINDINGS_FILE = RESULT_DIR / "one_step_model_findings.md"

TARGET = "target_demand_mwh"
RANDOM_SEED = 42
SPLIT_BOUNDS = {
    "train": (pd.Timestamp("2022-01-01T00"), pd.Timestamp("2023-12-31T23")),
    "validation": (pd.Timestamp("2024-01-01T00"), pd.Timestamp("2024-06-30T23")),
    "test": (pd.Timestamp("2024-07-01T00"), pd.Timestamp("2024-12-31T23")),
}
FEATURE_GROUPS = [
    "calendar_only",
    "autoregressive_demand",
    "renewable_history_enhanced",
]
ALGORITHMS = ["linear_regression", "xgboost"]
ALGORITHM_LABELS = {
    "linear_regression": "Linear Regression",
    "xgboost": "XGBoost",
    "persistence": "Persistence",
}
GROUP_LABELS = {
    "calendar_only": "calendar-only",
    "autoregressive_demand": "autoregressive demand",
    "renewable_history_enhanced": "renewable-history enhanced",
    "persistence": "1-hour lag",
}

# Declared before any test evaluation. Each candidate is intentionally modest.
XGBOOST_CANDIDATES: list[dict[str, Any]] = [
    {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05,
     "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1,
     "reg_lambda": 1.0},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03,
     "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1,
     "reg_lambda": 1.0},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
     "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 1,
     "reg_lambda": 1.0},
    {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.05,
     "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 5,
     "reg_lambda": 5.0},
    {"n_estimators": 250, "max_depth": 5, "learning_rate": 0.05,
     "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
     "reg_lambda": 5.0},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08,
     "subsample": 0.8, "colsample_bytree": 1.0, "min_child_weight": 1,
     "reg_lambda": 1.0},
]


def sha256(path: Path) -> str:
    """Return a file SHA-256 without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, filename: str) -> None:
    """Save a stable derived CSV table."""
    frame.to_csv(
        TABLE_DIR / filename,
        index=False,
        date_format="%Y-%m-%dT%H",
        float_format="%.10f",
    )


def model_id(algorithm: str, group: str) -> str:
    return f"{algorithm}__{group}"


def model_label(algorithm: str, group: str) -> str:
    if algorithm == "persistence":
        return "Persistence (1h)"
    return f"{ALGORITHM_LABELS[algorithm]} — {GROUP_LABELS[group]}"


def load_feature_declarations() -> dict[str, list[str]]:
    """Read exact predictor lists from the machine-readable declaration."""
    declarations = pd.read_csv(FEATURE_GROUP_FILE)
    required = {"feature", "safe_at_forecast_time", *FEATURE_GROUPS}
    missing = sorted(required.difference(declarations.columns))
    if missing:
        raise ValueError(f"Feature declaration is missing columns: {missing}")
    safe = declarations["safe_at_forecast_time"].astype(str).str.lower().eq("true")
    groups: dict[str, list[str]] = {}
    for group in FEATURE_GROUPS:
        included = declarations[group].astype(str).str.lower().eq("true")
        unsafe_selected = declarations.loc[included & ~safe, "feature"].tolist()
        if unsafe_selected:
            raise ValueError(f"Unsafe features declared for {group}: {unsafe_selected}")
        groups[group] = declarations.loc[included & safe, "feature"].tolist()
        if not groups[group]:
            raise ValueError(f"No features declared for {group}.")
    return groups


def load_data(feature_groups: dict[str, list[str]]) -> pd.DataFrame:
    """Load the feature master, enforce chronology, and coerce model fields."""
    frame = pd.read_csv(FEATURE_FILE, low_memory=False)
    required = {"period", TARGET, "target_available", "demand_lag_1h"}
    required.update(feature for group in feature_groups.values() for feature in group)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature master is missing required columns: {missing}")
    frame["period"] = pd.to_datetime(frame["period"], format="%Y-%m-%dT%H", errors="raise")
    if frame["period"].duplicated().any() or not frame["period"].is_monotonic_increasing:
        raise ValueError("Feature timestamps must be unique and chronological.")
    frame["split"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    for split, (start, end) in SPLIT_BOUNDS.items():
        mask = frame["period"].between(start, end, inclusive="both")
        if frame.loc[mask, "split"].notna().any():
            raise ValueError(f"Split overlap detected while assigning {split}.")
        frame.loc[mask, "split"] = split
    if frame["split"].isna().any():
        raise ValueError("Feature master contains timestamps outside the fixed split policy.")
    truth = frame["target_available"].astype(str).str.lower().map({"true": True, "false": False})
    if truth.isna().any():
        raise ValueError("target_available contains values other than True or False.")
    frame["target_available"] = truth.astype(bool)
    numeric_columns = {TARGET, "demand_lag_1h"}
    numeric_columns.update(feature for group in feature_groups.values() for feature in group)
    for column in sorted(numeric_columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def finite_rows(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].to_numpy(dtype=float)
    return pd.Series(np.isfinite(values).all(axis=1), index=frame.index)


def build_eligibility(
    frame: pd.DataFrame, feature_groups: dict[str, list[str]]
) -> tuple[dict[str, pd.Series], pd.Series, pd.DataFrame, pd.DataFrame]:
    """Build exact model eligibility and common comparison rows."""
    target_ok = (
        frame["target_available"]
        & frame[TARGET].notna()
        & np.isfinite(frame[TARGET].to_numpy(dtype=float))
    )
    group_ok = {
        group: target_ok & finite_rows(frame, features)
        for group, features in feature_groups.items()
    }
    persistence_ok = (
        target_ok
        & frame["demand_lag_1h"].notna()
        & np.isfinite(frame["demand_lag_1h"].to_numpy(dtype=float))
    )
    common_ok = persistence_ok.copy()
    for mask in group_ok.values():
        common_ok &= mask

    timestamp_table = frame[["period", "split"]].copy()
    timestamp_table["target_eligible"] = target_ok
    timestamp_table["persistence_eligible"] = persistence_ok
    for group, mask in group_ok.items():
        timestamp_table[f"{group}_eligible"] = mask
    timestamp_table["common_comparison_eligible"] = common_ok

    rows: list[dict[str, Any]] = []
    for split in SPLIT_BOUNDS:
        split_mask = frame["split"].eq(split)
        for algorithm in ALGORITHMS:
            for group in FEATURE_GROUPS:
                eligible = split_mask & group_ok[group]
                rows.append({
                    "split": split,
                    "algorithm": algorithm,
                    "model": model_id(algorithm, group),
                    "feature_group": group,
                    "split_total_rows": int(split_mask.sum()),
                    "target_available_rows": int((split_mask & target_ok).sum()),
                    "eligible_rows": int(eligible.sum()),
                    "ineligible_rows": int(split_mask.sum() - eligible.sum()),
                    "common_comparison_rows": int((split_mask & common_ok).sum()),
                })
        eligible = split_mask & persistence_ok
        rows.append({
            "split": split,
            "algorithm": "persistence",
            "model": "persistence_1h",
            "feature_group": "persistence",
            "split_total_rows": int(split_mask.sum()),
            "target_available_rows": int((split_mask & target_ok).sum()),
            "eligible_rows": int(eligible.sum()),
            "ineligible_rows": int(split_mask.sum() - eligible.sum()),
            "common_comparison_rows": int((split_mask & common_ok).sum()),
        })
    return group_ok, common_ok, timestamp_table, pd.DataFrame(rows)


def calculate_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    """Calculate baseline-compatible metrics from a prediction table."""
    actual = group["actual_demand_mwh"].to_numpy(dtype=float)
    prediction = group["prediction_mwh"].to_numpy(dtype=float)
    error = prediction - actual
    absolute_error = np.abs(error)
    nonzero = actual != 0
    smape_denominator = np.abs(actual) + np.abs(prediction)
    smape_valid = smape_denominator != 0
    denominator = np.sum((actual - actual.mean()) ** 2) if len(actual) else np.nan
    return {
        "observation_count": int(len(group)),
        "mae_mwh": float(np.mean(absolute_error)) if len(group) else np.nan,
        "rmse_mwh": float(np.sqrt(np.mean(error ** 2))) if len(group) else np.nan,
        "mape_pct": float(100 * np.mean(absolute_error[nonzero] / np.abs(actual[nonzero])))
        if nonzero.any() else np.nan,
        "smape_pct": float(100 * np.mean(2 * absolute_error[smape_valid] / smape_denominator[smape_valid]))
        if smape_valid.any() else np.nan,
        "mean_error_bias_mwh": float(np.mean(error)) if len(group) else np.nan,
        "r_squared": float(1 - np.sum(error ** 2) / denominator)
        if len(group) and denominator != 0 else np.nan,
    }


def prediction_table(
    frame: pd.DataFrame,
    mask: pd.Series,
    prediction: pd.Series,
    algorithm: str,
    group: str,
    comparison_type: str,
) -> pd.DataFrame:
    """Return eligible timestamp-level predictions in the required schema."""
    table = frame.loc[mask, ["period", "split", TARGET]].copy()
    table = table.rename(columns={TARGET: "actual_demand_mwh"})
    table["model"] = "persistence_1h" if algorithm == "persistence" else model_id(algorithm, group)
    table["algorithm"] = algorithm
    table["feature_group"] = group
    table["comparison_set_type"] = comparison_type
    table["prediction_mwh"] = prediction.loc[mask].to_numpy(dtype=float)
    table["error_mwh"] = table["prediction_mwh"] - table["actual_demand_mwh"]
    table["absolute_error_mwh"] = table["error_mwh"].abs()
    valid = table["actual_demand_mwh"].ne(0)
    table["percentage_error_pct"] = np.where(
        valid, 100 * table["error_mwh"] / table["actual_demand_mwh"], np.nan
    )
    return table[[
        "period", "split", "model", "algorithm", "feature_group",
        "comparison_set_type", "actual_demand_mwh", "prediction_mwh",
        "error_mwh", "absolute_error_mwh", "percentage_error_pct",
    ]]


def aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["split", "model", "algorithm", "feature_group", "comparison_set_type"]
    for values, group in predictions.groupby(keys, sort=False, observed=True):
        row = dict(zip(keys, values))
        row.update(calculate_metrics(group))
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(["comparison_set_type", "split", "mae_mwh", "model"]).reset_index(drop=True)


def dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "joblib": joblib.__version__,
    }


def base_metadata(
    algorithm: str,
    group: str,
    features: list[str],
    train_rows: pd.DataFrame,
    feature_hash: str,
) -> dict[str, Any]:
    return {
        "algorithm": ALGORITHM_LABELS[algorithm],
        "algorithm_key": algorithm,
        "feature_group": group,
        "feature_list": features,
        "training_range_utc": [str(SPLIT_BOUNDS["train"][0]), str(SPLIT_BOUNDS["train"][1])],
        "validation_range_utc": [str(SPLIT_BOUNDS["validation"][0]), str(SPLIT_BOUNDS["validation"][1])],
        "test_range_utc": [str(SPLIT_BOUNDS["test"][0]), str(SPLIT_BOUNDS["test"][1])],
        "training_row_count": int(len(train_rows)),
        "first_eligible_training_timestamp_utc": str(train_rows["period"].min()),
        "last_eligible_training_timestamp_utc": str(train_rows["period"].max()),
        "feature_file": str(FEATURE_FILE.relative_to(ROOT)).replace("\\", "/"),
        "feature_file_sha256": feature_hash,
        "dependency_versions": dependency_versions(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "forecast_horizon_hours": 1,
        "fit_split": "train",
        "test_used_for_selection": False,
    }


def save_model_and_metadata(name: str, fitted: Any, metadata: dict[str, Any]) -> None:
    artifact = MODEL_DIR / f"{name}.joblib"
    joblib.dump(fitted, artifact)
    metadata["artifact_file"] = str(artifact.relative_to(ROOT)).replace("\\", "/")
    metadata["artifact_sha256"] = sha256(artifact)
    with (MODEL_DIR / f"{name}.metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)


def train_models(
    frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    eligibility: dict[str, pd.Series],
    feature_hash: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit all models and select XGBoost settings with validation only."""
    fitted_models: dict[str, Any] = {}
    linear_coefficients: list[pd.DataFrame] = []
    xgb_importances: list[pd.DataFrame] = []
    search_rows: list[dict[str, Any]] = []

    # Linear models: training-only scaler and regression fit.
    for group, features in feature_groups.items():
        train_mask = frame["split"].eq("train") & eligibility[group]
        train = frame.loc[train_mask]
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ])
        pipeline.fit(train[features].astype(float), train[TARGET].astype(float))
        name = model_id("linear_regression", group)
        fitted_models[name] = pipeline
        scaler = pipeline.named_steps["scaler"]
        regressor = pipeline.named_steps["regressor"]
        coefficients = pd.DataFrame({
            "algorithm": "linear_regression",
            "feature_group": group,
            "model": name,
            "feature": features,
            "standardized_coefficient": regressor.coef_.astype(float),
            "absolute_standardized_coefficient": np.abs(regressor.coef_.astype(float)),
            "training_feature_mean": scaler.mean_.astype(float),
            "training_feature_scale": scaler.scale_.astype(float),
        }).sort_values("absolute_standardized_coefficient", ascending=False)
        linear_coefficients.append(coefficients)
        metadata = base_metadata("linear_regression", group, features, train, feature_hash)
        metadata.update({
            "selected_parameters": {"fit_intercept": bool(regressor.fit_intercept)},
            "random_seed": None,
            "pipeline_steps": ["StandardScaler", "LinearRegression"],
            "scaler_fit_split": "train",
            "scaler_training_row_count": int(scaler.n_samples_seen_),
            "scaler_feature_means": dict(zip(features, scaler.mean_.astype(float))),
            "scaler_feature_scales": dict(zip(features, scaler.scale_.astype(float))),
            "coefficient_interpretation": "Standardized association, not causal effect.",
        })
        save_model_and_metadata(name, pipeline, metadata)

    # XGBoost search: every candidate fits training and is ranked on validation.
    for group, features in feature_groups.items():
        train_mask = frame["split"].eq("train") & eligibility[group]
        validation_mask = frame["split"].eq("validation") & eligibility[group]
        train = frame.loc[train_mask]
        validation = frame.loc[validation_mask]
        candidates: list[tuple[float, int, XGBRegressor]] = []
        for candidate_id, parameters in enumerate(XGBOOST_CANDIDATES, start=1):
            model = XGBRegressor(
                **parameters,
                objective="reg:squarederror",
                random_state=RANDOM_SEED,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
            )
            model.fit(train[features].astype(float), train[TARGET].astype(float))
            prediction = model.predict(validation[features].astype(float))
            temporary = pd.DataFrame({
                "actual_demand_mwh": validation[TARGET].to_numpy(dtype=float),
                "prediction_mwh": prediction,
            })
            metrics = calculate_metrics(temporary)
            row = {
                "feature_group": group,
                "candidate_id": candidate_id,
                **parameters,
                "random_seed": RANDOM_SEED,
                "fit_split": "train",
                "selection_split": "validation",
                "selection_metric": "mae_mwh",
                **metrics,
            }
            search_rows.append(row)
            candidates.append((float(metrics["mae_mwh"]), candidate_id, model))
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, winner_id, winner = candidates[0]
        for row in search_rows:
            if row["feature_group"] == group:
                row["selected"] = row["candidate_id"] == winner_id
        name = model_id("xgboost", group)
        fitted_models[name] = winner
        winner_parameters = XGBOOST_CANDIDATES[winner_id - 1]
        booster_scores = winner.get_booster().get_score(importance_type="gain")
        gains = np.array([booster_scores.get(feature, 0.0) for feature in features], dtype=float)
        normalized = gains / gains.sum() if gains.sum() else gains
        importances = pd.DataFrame({
            "algorithm": "xgboost",
            "feature_group": group,
            "model": name,
            "feature": features,
            "gain": gains,
            "normalized_gain": normalized,
        }).sort_values("gain", ascending=False)
        xgb_importances.append(importances)
        metadata = base_metadata("xgboost", group, features, train, feature_hash)
        metadata.update({
            "selected_parameters": winner_parameters,
            "fixed_model_parameters": {
                "objective": "reg:squarederror", "tree_method": "hist", "n_jobs": 1
            },
            "random_seed": RANDOM_SEED,
            "candidate_grid_defined_in_source": True,
            "candidate_count": len(XGBOOST_CANDIDATES),
            "selected_candidate_id": winner_id,
            "selection_split": "validation",
            "selection_metric": "mae_mwh",
            "test_used_for_selection": False,
            "importance_type": "gain",
            "importance_interpretation": "Predictive split gain, not causal effect.",
        })
        save_model_and_metadata(name, winner, metadata)

    return (
        fitted_models,
        pd.concat(linear_coefficients, ignore_index=True),
        pd.concat(xgb_importances, ignore_index=True),
        pd.DataFrame(search_rows),
    )


def make_all_predictions(
    frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    eligibility: dict[str, pd.Series],
    common_ok: pd.Series,
    fitted_models: dict[str, Any],
) -> pd.DataFrame:
    """Predict validation and test after all validation-only selection is fixed."""
    predicted: dict[str, pd.Series] = {}
    evaluation = frame["split"].isin(["validation", "test"])
    for algorithm in ALGORITHMS:
        for group, features in feature_groups.items():
            name = model_id(algorithm, group)
            mask = evaluation & eligibility[group]
            values = pd.Series(np.nan, index=frame.index, dtype=float)
            values.loc[mask] = fitted_models[name].predict(frame.loc[mask, features].astype(float))
            predicted[name] = values

    tables: list[pd.DataFrame] = []
    for algorithm in ALGORITHMS:
        for group in FEATURE_GROUPS:
            name = model_id(algorithm, group)
            natural = evaluation & eligibility[group]
            common = evaluation & common_ok
            tables.append(prediction_table(frame, natural, predicted[name], algorithm, group, "natural"))
            tables.append(prediction_table(frame, common, predicted[name], algorithm, group, "common"))
    persistence = frame["demand_lag_1h"].astype(float)
    tables.append(prediction_table(
        frame, evaluation & common_ok, persistence, "persistence", "persistence", "common"
    ))
    return pd.concat(tables, ignore_index=True).sort_values(
        ["comparison_set_type", "split", "period", "model"]
    ).reset_index(drop=True)


def build_error_analysis(
    common_predictions: pd.DataFrame, selected_models: list[str]
) -> dict[str, pd.DataFrame]:
    """Analyse selected models and persistence on identical common timestamps."""
    names = ["persistence_1h", *selected_models]
    selected = common_predictions[common_predictions["model"].isin(names)].copy()
    selected["hour_utc"] = selected["period"].dt.hour
    selected["calendar_month"] = selected["period"].dt.month
    selected["day_of_week_utc"] = selected["period"].dt.dayofweek

    def grouped_metrics(column: str) -> pd.DataFrame:
        rows = []
        for keys, part in selected.groupby(["split", "model", column], observed=True):
            split, model, value = keys
            rows.append({"split": split, "model": model, column: value, **calculate_metrics(part)})
        return pd.DataFrame(rows)

    thresholds = []
    percentile_rows = []
    for split in ["validation", "test"]:
        split_rows = selected[(selected["split"] == split) & (selected["model"] == "persistence_1h")]
        actual = split_rows["actual_demand_mwh"]
        low = float(actual.quantile(0.10))
        high = float(actual.quantile(0.90))
        thresholds.append({
            "split": split,
            "comparison_set_type": "common",
            "observation_count": len(actual),
            "bottom_10_percent_threshold_mwh": low,
            "top_10_percent_threshold_mwh": high,
        })
        for model in names:
            model_rows = selected[(selected["split"] == split) & (selected["model"] == model)]
            for segment, mask in {
                "bottom_10_percent": model_rows["actual_demand_mwh"] <= low,
                "top_10_percent": model_rows["actual_demand_mwh"] >= high,
            }.items():
                percentile_rows.append({
                    "split": split, "model": model, "demand_segment": segment,
                    "threshold_mwh": low if segment.startswith("bottom") else high,
                    **calculate_metrics(model_rows.loc[mask]),
                })

    largest = selected.sort_values(
        ["split", "model", "absolute_error_mwh"], ascending=[True, True, False]
    ).groupby(["split", "model"], observed=True).head(20).reset_index(drop=True)

    degradation_rows = []
    metrics = aggregate_metrics(selected)
    for model in names:
        by_split = metrics[metrics["model"] == model].set_index("split")
        validation_mae = float(by_split.loc["validation", "mae_mwh"])
        test_mae = float(by_split.loc["test", "mae_mwh"])
        degradation_rows.append({
            "model": model,
            "validation_mae_mwh": validation_mae,
            "test_mae_mwh": test_mae,
            "test_minus_validation_mae_mwh": test_mae - validation_mae,
            "mae_change_pct": 100 * (test_mae - validation_mae) / validation_mae,
        })

    return {
        "error_by_utc_hour.csv": grouped_metrics("hour_utc"),
        "error_by_calendar_month.csv": grouped_metrics("calendar_month"),
        "error_by_day_of_week_utc.csv": grouped_metrics("day_of_week_utc"),
        "demand_percentile_thresholds.csv": pd.DataFrame(thresholds),
        "demand_percentile_performance.csv": pd.DataFrame(percentile_rows),
        "largest_absolute_errors.csv": largest,
        "validation_to_test_degradation.csv": pd.DataFrame(degradation_rows),
    }


def select_representative_weeks(common_predictions: pd.DataFrame) -> pd.DataFrame:
    """Choose the earliest complete Monday-to-Sunday UTC week per split."""
    persistence = common_predictions[common_predictions["model"] == "persistence_1h"]
    rows = []
    for split in ["validation", "test"]:
        timestamps = pd.DatetimeIndex(persistence.loc[persistence["split"] == split, "period"])
        available = set(timestamps)
        candidates = [timestamp for timestamp in timestamps if timestamp.dayofweek == 0 and timestamp.hour == 0]
        chosen = None
        for start in candidates:
            expected = pd.date_range(start, periods=168, freq="h")
            if all(timestamp in available for timestamp in expected):
                chosen = start
                break
        if chosen is None:
            raise ValueError(f"No complete common-subset UTC week found for {split}.")
        rows.append({
            "split": split,
            "week_start_utc": chosen,
            "week_end_utc": chosen + pd.Timedelta(hours=167),
            "selection_rule": "Earliest complete Monday 00:00 through Sunday 23:00 UTC week",
            "hour_count": 168,
        })
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_bar_metrics(metrics: pd.DataFrame, split: str, metric: str, title: str, filename: str) -> None:
    data = metrics[(metrics["split"] == split) & (metrics["comparison_set_type"] == "common")].copy()
    data["label"] = [model_label(a, g) for a, g in zip(data["algorithm"], data["feature_group"])]
    data = data.sort_values(metric)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(data["label"], data[metric], color=["#5B8FF9" if a != "persistence" else "#F6BD16" for a in data["algorithm"]])
    ax.set_xlabel("MWh")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, filename)


def create_figures(
    common_metrics: pd.DataFrame,
    common_predictions: pd.DataFrame,
    analyses: dict[str, pd.DataFrame],
    representative_weeks: pd.DataFrame,
    selected_models: list[str],
    linear_coefficients: pd.DataFrame,
    xgb_importances: pd.DataFrame,
) -> None:
    """Create the 13 required readable PNG figures."""
    plot_bar_metrics(common_metrics, "validation", "mae_mwh", "Validation common-subset MAE", "01_validation_common_mae_comparison.png")
    plot_bar_metrics(common_metrics, "test", "mae_mwh", "Test common-subset MAE", "02_test_common_mae_comparison.png")
    plot_bar_metrics(common_metrics, "validation", "rmse_mwh", "Validation common-subset RMSE", "03_validation_common_rmse_comparison.png")
    plot_bar_metrics(common_metrics, "test", "rmse_mwh", "Test common-subset RMSE", "04_test_common_rmse_comparison.png")

    names = ["persistence_1h", *selected_models]
    selected = common_predictions[common_predictions["model"].isin(names)].copy()
    for split, filename, number in [
        ("validation", "05_representative_validation_week_utc.png", "Validation"),
        ("test", "06_representative_test_week_utc.png", "Test"),
    ]:
        week = representative_weeks.set_index("split").loc[split]
        plot = selected[(selected["split"] == split) & selected["period"].between(week["week_start_utc"], week["week_end_utc"])]
        fig, ax = plt.subplots(figsize=(12, 5.5))
        actual = plot[plot["model"] == "persistence_1h"]
        ax.plot(actual["period"], actual["actual_demand_mwh"], color="black", linewidth=2, label="Actual")
        for model in names:
            rows = plot[plot["model"] == model]
            ax.plot(rows["period"], rows["prediction_mwh"], linewidth=1.2, label=model.replace("__", " — "))
        ax.set_title(f"{number} representative week (UTC): {week['week_start_utc']:%Y-%m-%d}")
        ax.set_ylabel("Demand (MWh)")
        ax.set_xlabel("Timestamp (UTC)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.2)
        save_figure(fig, filename)

    hour = analyses["error_by_utc_hour.csv"]
    hour = hour[(hour["split"] == "test") & hour["model"].isin(names)]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for model in names:
        rows = hour[hour["model"] == model]
        ax.plot(rows["hour_utc"], rows["mae_mwh"], marker="o", markersize=3, label=model.replace("__", " — "))
    ax.set(title="Test common-subset absolute error by UTC hour", xlabel="UTC hour", ylabel="MAE (MWh)")
    ax.set_xticks(range(24)); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    save_figure(fig, "07_test_absolute_error_by_utc_hour.png")

    month = analyses["error_by_calendar_month.csv"]
    month = month[(month["split"] == "test") & month["model"].isin(names)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model in names:
        rows = month[month["model"] == model]
        ax.plot(rows["calendar_month"], rows["mae_mwh"], marker="o", label=model.replace("__", " — "))
    ax.set(title="Test common-subset MAE by calendar month", xlabel="Calendar month (UTC)", ylabel="MAE (MWh)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    save_figure(fig, "08_test_mae_by_month.png")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    test = selected[selected["split"] == "test"]
    for model in names:
        errors = test.loc[test["model"] == model, "error_mwh"]
        ax.hist(errors, bins=50, alpha=0.42, density=True, label=model.replace("__", " — "))
    ax.axvline(0, color="black", linewidth=1)
    ax.set(title="Test common-subset prediction residual distributions", xlabel="Prediction minus actual (MWh)", ylabel="Density")
    ax.legend(fontsize=8)
    save_figure(fig, "09_prediction_residual_distributions.png")

    best_linear = selected_models[0]
    group = best_linear.split("__", 1)[1]
    coef = linear_coefficients[linear_coefficients["feature_group"] == group].nlargest(15, "absolute_standardized_coefficient").sort_values("standardized_coefficient")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(coef["feature"], coef["standardized_coefficient"], color=np.where(coef["standardized_coefficient"] >= 0, "#5B8FF9", "#E8684A"))
    ax.set(title=f"Linear Regression standardized coefficients: {GROUP_LABELS[group]}", xlabel="Standardized coefficient")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, "10_linear_regression_coefficient_importance.png")

    best_xgb = selected_models[1]
    group = best_xgb.split("__", 1)[1]
    importance = xgb_importances[xgb_importances["feature_group"] == group].nlargest(15, "gain").sort_values("gain")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance["feature"], importance["gain"], color="#61DDAA")
    ax.set(title=f"XGBoost feature importance by gain: {GROUP_LABELS[group]}", xlabel="Total gain")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, "11_xgboost_feature_importance_gain.png")

    peak = analyses["demand_percentile_performance.csv"]
    peak = peak[(peak["split"] == "test") & (peak["demand_segment"] == "top_10_percent")]
    peak = peak.set_index("model").loc[names].reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([m.replace("__", " — ") for m in peak["model"]], peak["mae_mwh"], color=["#F6BD16", "#5B8FF9", "#61DDAA"])
    ax.set(title="Test peak-demand performance (top 10%, common subset)", ylabel="MAE (MWh)")
    ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "12_peak_demand_performance_comparison.png")

    degrade = analyses["validation_to_test_degradation.csv"].set_index("model").loc[names].reset_index()
    x = np.arange(len(degrade)); width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, degrade["validation_mae_mwh"], width, label="Validation")
    ax.bar(x + width / 2, degrade["test_mae_mwh"], width, label="Test")
    ax.set_xticks(x, [m.replace("__", " — ") for m in degrade["model"]], rotation=15)
    ax.set(title="Validation-to-test common-subset MAE change", ylabel="MAE (MWh)")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "13_validation_to_test_mae_change.png")


def write_findings(
    eligibility_summary: pd.DataFrame,
    natural_metrics: pd.DataFrame,
    common_metrics: pd.DataFrame,
    selected_models: list[str],
    analyses: dict[str, pd.DataFrame],
    linear_coefficients: pd.DataFrame,
    xgb_importances: pd.DataFrame,
    search: pd.DataFrame,
    feature_hash: str,
) -> None:
    """Write an evidence-only findings report from calculated outputs."""
    best_linear, best_xgb = selected_models
    validation = common_metrics[common_metrics["split"] == "validation"].set_index("model")
    test = common_metrics[common_metrics["split"] == "test"].set_index("model")
    persistence_val = float(validation.loc["persistence_1h", "mae_mwh"])
    persistence_test = float(test.loc["persistence_1h", "mae_mwh"])

    def comparison(model: str, split_metrics: pd.DataFrame, persistence_mae: float) -> float:
        return 100 * (persistence_mae - float(split_metrics.loc[model, "mae_mwh"])) / persistence_mae

    lines = [
        "# One-Step Linear Regression and XGBoost Findings",
        "",
        "## Evaluation design",
        "",
        "All models predict one hour ahead on the fixed chronological splits. Predictors come only from the declared safe feature groups. Missing predictors and targets are not imputed. Model selection uses validation MAE; test results are reporting-only.",
        "",
        f"Feature master SHA-256: `{feature_hash}`.",
        "",
        "## Eligibility",
        "",
        "Natural eligibility counts vary only by feature group; Linear Regression and XGBoost use identical rows within a group.",
        "",
        "| Split | Feature group | Eligible rows | Common rows |",
        "|---|---|---:|---:|",
    ]
    one_algorithm = eligibility_summary[eligibility_summary["algorithm"] == "linear_regression"]
    for row in one_algorithm.itertuples():
        lines.append(f"| {row.split} | {GROUP_LABELS[row.feature_group]} | {row.eligible_rows:,} | {row.common_comparison_rows:,} |")

    lines += ["", "## Common-subset metrics", ""]
    for split, table in [("Validation", validation), ("Test", test)]:
        lines += [f"### {split}", "", "| Model | n | MAE | RMSE | MAPE | sMAPE | Bias | R² |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for model, row in table.sort_values("mae_mwh").iterrows():
            lines.append(
                f"| {model.replace('__', ' — ')} | {int(row.observation_count):,} | {row.mae_mwh:,.2f} | {row.rmse_mwh:,.2f} | {row.mape_pct:.3f}% | {row.smape_pct:.3f}% | {row.mean_error_bias_mwh:,.2f} | {row.r_squared:.4f} |"
            )
        lines.append("")

    lines += ["## Natural-eligibility metrics", "", "| Split | Model | n | MAE | RMSE | MAPE | sMAPE | Bias | R² |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in natural_metrics.sort_values(["split", "mae_mwh"]).itertuples():
        lines.append(
            f"| {row.split} | {row.model.replace('__', ' — ')} | {row.observation_count:,} | {row.mae_mwh:,.2f} | {row.rmse_mwh:,.2f} | {row.mape_pct:.3f}% | {row.smape_pct:.3f}% | {row.mean_error_bias_mwh:,.2f} | {row.r_squared:.4f} |"
        )

    selected_search = search[search["selected"]].set_index("feature_group")
    lines += [
        "",
        "## Selection and persistence comparison",
        "",
        f"The best Linear Regression variant by common-subset validation MAE is `{best_linear}`. Its validation change versus persistence is {comparison(best_linear, validation, persistence_val):+.2f}% (positive means improvement), and its test change is {comparison(best_linear, test, persistence_test):+.2f}%.",
        "",
        f"The best XGBoost variant is `{best_xgb}`. Its validation change versus persistence is {comparison(best_xgb, validation, persistence_val):+.2f}%, and its test change is {comparison(best_xgb, test, persistence_test):+.2f}%.",
        "",
        "A model is considered better only when its common-subset MAE is below persistence; high R² alone is not evidence of success.",
        "",
        "Selected XGBoost settings by feature group:",
        "",
        "```json",
        json.dumps({group: {key: (int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value) for key, value in XGBOOST_CANDIDATES[int(selected_search.loc[group, 'candidate_id']) - 1].items()} for group in FEATURE_GROUPS}, indent=2),
        "```",
    ]

    degradation = analyses["validation_to_test_degradation.csv"].set_index("model")
    peak = analyses["demand_percentile_performance.csv"]
    lines += ["", "## Error analysis", ""]
    for model in ["persistence_1h", best_linear, best_xgb]:
        row = degradation.loc[model]
        peak_row = peak[(peak["split"] == "test") & (peak["model"] == model) & (peak["demand_segment"] == "top_10_percent")].iloc[0]
        lines.append(f"- `{model}`: test MAE changed by {row.mae_change_pct:+.2f}% from validation; test top-10% demand MAE was {peak_row.mae_mwh:,.2f} MWh on {int(peak_row.observation_count):,} rows.")

    hour = analyses["error_by_utc_hour.csv"]
    month = analyses["error_by_calendar_month.csv"]
    for model in [best_linear, best_xgb]:
        worst_hour = hour[(hour["split"] == "test") & (hour["model"] == model)].sort_values("mae_mwh", ascending=False).iloc[0]
        worst_month = month[(month["split"] == "test") & (month["model"] == model)].sort_values("mae_mwh", ascending=False).iloc[0]
        lines.append(f"- `{model}` had its highest test hourly MAE at UTC hour {int(worst_hour.hour_utc):02d} and its highest monthly MAE in month {int(worst_month.calendar_month)}.")

    linear_group = best_linear.split("__", 1)[1]
    xgb_group = best_xgb.split("__", 1)[1]
    coef = linear_coefficients[linear_coefficients["feature_group"] == linear_group]
    positive = coef.nlargest(5, "standardized_coefficient")["feature"].tolist()
    negative = coef.nsmallest(5, "standardized_coefficient")["feature"].tolist()
    leading_xgb = xgb_importances[xgb_importances["feature_group"] == xgb_group].nlargest(10, "gain")["feature"].tolist()
    lines += [
        "",
        "## Interpretation",
        "",
        f"Largest positive standardized Linear Regression coefficients: {', '.join(f'`{item}`' for item in positive)}.",
        "",
        f"Largest negative standardized Linear Regression coefficients: {', '.join(f'`{item}`' for item in negative)}.",
        "",
        "Lag and rolling predictors are correlated, so coefficient size and sign are not causal effects and may be unstable across related specifications.",
        "",
        f"Leading XGBoost features by total gain: {', '.join(f'`{item}`' for item in leading_xgb)}. Gain measures predictive split contribution, not causality.",
        "",
        "## Limitations and next step",
        "",
        "One-step evaluation assumes the actual prior-hour demand becomes available before every next prediction. A recursive 24-hour forecast must feed predictions back into later horizons, so errors can compound and some observed lags will no longer be available. These results therefore do not guarantee 24-hour recursive performance.",
        "",
    ]
    beats_val = min(float(validation.loc[best_linear, "mae_mwh"]), float(validation.loc[best_xgb, "mae_mwh"])) < persistence_val
    lines.append(
        "The evidence supports proceeding to a separately designed recursive 24-hour experiment, while retaining persistence as the required benchmark."
        if beats_val else
        "Neither selected algorithm beats persistence on common-subset validation MAE, so the modelling approach should be reviewed before investing in recursive 24-hour forecasting."
    )
    FINDINGS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    feature_hash_before = sha256(FEATURE_FILE)
    feature_groups = load_feature_declarations()
    frame = load_data(feature_groups)
    eligibility, common_ok, eligibility_timestamps, eligibility_summary = build_eligibility(frame, feature_groups)

    save_csv(eligibility_timestamps, "model_eligibility_by_timestamp.csv")
    save_csv(eligibility_summary, "model_eligibility_summary.csv")
    common_timestamps = eligibility_timestamps[
        eligibility_timestamps["split"].isin(["validation", "test"])
        & eligibility_timestamps["common_comparison_eligible"]
    ][["period", "split"]]
    save_csv(common_timestamps, "common_comparison_timestamps.csv")
    common_summary = common_timestamps.groupby("split", observed=True).agg(
        common_row_count=("period", "size"),
        first_timestamp_utc=("period", "min"),
        last_timestamp_utc=("period", "max"),
    ).reset_index()
    save_csv(common_summary, "common_comparison_summary.csv")
    save_csv(pd.DataFrame([
        {"feature_group": group, "feature_order": order, "feature": feature}
        for group, features in feature_groups.items()
        for order, feature in enumerate(features, start=1)
    ]), "model_feature_lists.csv")

    fitted, coefficients, importances, search = train_models(
        frame, feature_groups, eligibility, feature_hash_before
    )
    save_csv(search.sort_values(["feature_group", "candidate_id"]), "xgboost_validation_search.csv")
    save_csv(coefficients, "linear_regression_standardized_coefficients_all.csv")
    save_csv(importances, "xgboost_feature_importance_gain_all.csv")
    for group in FEATURE_GROUPS:
        save_csv(coefficients[coefficients["feature_group"] == group], f"linear_regression_coefficients_{group}.csv")
        save_csv(importances[importances["feature_group"] == group], f"xgboost_gain_importance_{group}.csv")

    predictions = make_all_predictions(frame, feature_groups, eligibility, common_ok, fitted)
    for comparison in ["natural", "common"]:
        for split in ["validation", "test"]:
            part = predictions[(predictions["comparison_set_type"] == comparison) & (predictions["split"] == split)]
            save_csv(part, f"one_step_predictions_{comparison}_{split}.csv")
    save_csv(predictions, "one_step_predictions_all.csv")
    metrics = aggregate_metrics(predictions)
    natural_metrics = metrics[metrics["comparison_set_type"] == "natural"].copy()
    common_metrics = metrics[metrics["comparison_set_type"] == "common"].copy()
    save_csv(natural_metrics, "model_metrics_natural.csv")
    save_csv(common_metrics, "model_metrics_common.csv")
    save_csv(metrics, "model_metrics_all.csv")
    for comparison, table in [("natural", natural_metrics), ("common", common_metrics)]:
        for split in ["validation", "test"]:
            save_csv(table[table["split"] == split], f"model_metrics_{comparison}_{split}.csv")

    validation_common = common_metrics[common_metrics["split"] == "validation"]
    best_linear = validation_common[validation_common["algorithm"] == "linear_regression"].sort_values(["mae_mwh", "model"]).iloc[0]["model"]
    best_xgb = validation_common[validation_common["algorithm"] == "xgboost"].sort_values(["mae_mwh", "model"]).iloc[0]["model"]
    selected_models = [best_linear, best_xgb]
    selected_summary = validation_common[validation_common["model"].isin(selected_models)].copy()
    selected_summary["selection_rank_within_algorithm"] = selected_summary.groupby("algorithm")["mae_mwh"].rank(method="first")
    save_csv(selected_summary, "selected_model_variants.csv")

    common_predictions = predictions[predictions["comparison_set_type"] == "common"].copy()
    analyses = build_error_analysis(common_predictions, selected_models)
    for filename, table in analyses.items():
        save_csv(table, filename)
    representative_weeks = select_representative_weeks(common_predictions)
    save_csv(representative_weeks, "representative_weeks.csv")
    create_figures(
        common_metrics, common_predictions, analyses, representative_weeks,
        selected_models, coefficients, importances,
    )

    run_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_file_sha256_before": feature_hash_before,
        "feature_file_sha256_after": sha256(FEATURE_FILE),
        "feature_file_unchanged": sha256(FEATURE_FILE) == feature_hash_before,
        "fixed_split_bounds": {key: [str(value[0]), str(value[1])] for key, value in SPLIT_BOUNDS.items()},
        "random_seed": RANDOM_SEED,
        "xgboost_candidate_count_per_group": len(XGBOOST_CANDIDATES),
        "selected_linear_regression_variant": best_linear,
        "selected_xgboost_variant": best_xgb,
        "dependency_versions": dependency_versions(),
    }
    (MODEL_DIR / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")
    save_csv(pd.DataFrame([{
        "feature_file_sha256_before": feature_hash_before,
        "feature_file_sha256_after": run_metadata["feature_file_sha256_after"],
        "feature_file_unchanged": run_metadata["feature_file_unchanged"],
        "selected_linear_regression_variant": best_linear,
        "selected_xgboost_variant": best_xgb,
        **{f"version_{key}": value for key, value in dependency_versions().items()},
    }]), "one_step_run_metadata.csv")
    write_findings(
        eligibility_summary, natural_metrics, common_metrics, selected_models,
        analyses, coefficients, importances, search, feature_hash_before,
    )
    if sha256(FEATURE_FILE) != feature_hash_before:
        raise RuntimeError("Feature master SHA-256 changed during training.")
    print(f"Completed one-step modelling. Feature SHA-256 unchanged: {feature_hash_before}")


if __name__ == "__main__":
    main()
