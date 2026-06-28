"""Build a leakage-safe renewable-aware planning layer from saved forecasts.

This script does not train a demand model. It combines the saved recursive
XGBoost demand forecast with transparent, past-only renewable estimates.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_PATH = ROOT / "data" / "processed" / "eia_ciso_hourly_2022_2024.csv"
RECURSIVE_PATH = ROOT / "results" / "recursive" / "tables" / "recursive_predictions.csv"
SPLIT_POLICY_PATH = ROOT / "data" / "modelling_split_policy.md"
QUALITY_POLICY_PATH = ROOT / "data" / "historical_data_quality.md"
OUTPUT_DIR = ROOT / "results" / "planning"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
POLICY_PATH = ROOT / "data" / "renewable_planning_policy.md"
FINDINGS_PATH = OUTPUT_DIR / "renewable_planning_findings.md"

SPLITS = {
    "train": (pd.Timestamp("2022-01-01T00"), pd.Timestamp("2023-12-31T23")),
    "validation": (pd.Timestamp("2024-01-01T00"), pd.Timestamp("2024-06-30T23")),
    "test": (pd.Timestamp("2024-07-01T00"), pd.Timestamp("2024-12-31T23")),
}
METHODS = ["daily_seasonal_naive", "weekly_seasonal_naive", "past_same_hour_rolling_median"]
METHOD_LABELS = {
    "daily_seasonal_naive": "Daily seasonal naive (24 h)",
    "weekly_seasonal_naive": "Weekly seasonal naive (168 h)",
    "past_same_hour_rolling_median": "Past same-hour rolling median (28 d)",
}
MIN_SAME_HOUR_OBSERVATIONS = 7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(TABLE_DIR / name, index=False, float_format="%.10f")


def metric_row(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")
    valid = actual.notna() & predicted.notna()
    y = actual[valid].astype(float)
    p = predicted[valid].astype(float)
    error = p - y
    nonzero = y.ne(0)
    smape_valid = (y.abs() + p.abs()).ne(0)
    return {
        "count": int(valid.sum()),
        "mae_mwh": float(error.abs().mean()) if len(error) else np.nan,
        "rmse_mwh": float(np.sqrt(np.mean(np.square(error)))) if len(error) else np.nan,
        "bias_mwh": float(error.mean()) if len(error) else np.nan,
        "mape_pct": float((error[nonzero].abs() / y[nonzero].abs()).mean() * 100) if nonzero.any() else np.nan,
        "mape_count": int(nonzero.sum()),
        "smape_pct": float((2 * error[smape_valid].abs() / (y[smape_valid].abs() + p[smape_valid].abs())).mean() * 100) if smape_valid.any() else np.nan,
        "smape_count": int(smape_valid.sum()),
    }


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    required = [HISTORICAL_PATH, RECURSIVE_PATH, SPLIT_POLICY_PATH, QUALITY_POLICY_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in required}
    historical = pd.read_csv(HISTORICAL_PATH)
    predictions = pd.read_csv(RECURSIVE_PATH)
    historical["period"] = pd.to_datetime(historical["period"], errors="raise")
    for col in ["forecast_origin", "forecast_date", "target_timestamp"]:
        predictions[col] = pd.to_datetime(predictions[col], errors="raise")
    predictions = predictions.loc[predictions["model"].eq("recursive_xgboost")].copy()
    if predictions.empty:
        raise ValueError("No recursive_xgboost rows were found in the saved predictions.")
    return historical, predictions, hashes


def validate_input_shape(predictions: pd.DataFrame) -> None:
    expected = {
        "validation": (pd.Timestamp("2024-01-01T00"), pd.Timestamp("2024-06-30T00"), 182),
        "test": (pd.Timestamp("2024-07-01T00"), pd.Timestamp("2024-12-31T00"), 184),
    }
    if set(predictions["split"]) != {"validation", "test"}:
        raise ValueError("Saved XGBoost rows do not contain exactly validation and test splits.")
    for split, (start, end, days) in expected.items():
        rows = predictions.loc[predictions["split"].eq(split)]
        dates = rows["forecast_date"].drop_duplicates().sort_values()
        if len(dates) != days or dates.iloc[0] != start or dates.iloc[-1] != end:
            raise ValueError(f"{split} forecast dates differ from the fixed recursive output.")
        horizon_sets = rows.groupby("forecast_origin")["horizon"].apply(set)
        if not horizon_sets.map(lambda value: value == set(range(1, 25))).all():
            raise ValueError(f"{split} contains a forecast without horizons 1-24.")


def same_hour_history(
    historical_indexed: pd.DataFrame, origin: pd.Timestamp, target_hour: int
) -> pd.DataFrame:
    start = origin - pd.Timedelta(days=28)
    rows = historical_indexed.loc[
        (historical_indexed.index > start)
        & (historical_indexed.index <= origin)
        & (historical_indexed.index.hour == target_hour),
        ["solar_generation_mwh", "wind_generation_mwh"],
    ]
    return rows


def add_renewable_predictions(planning: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    history = historical.set_index("period").sort_index()
    solar_lookup = history["solar_generation_mwh"]
    wind_lookup = history["wind_generation_mwh"]
    records: list[dict[str, object]] = []
    for row in planning.itertuples(index=False):
        target = row.target_timestamp
        origin = row.forecast_origin
        daily_time = target - pd.Timedelta(hours=24)
        weekly_time = target - pd.Timedelta(hours=168)
        past = same_hour_history(history, origin, target.hour)
        solar_past = past["solar_generation_mwh"].dropna()
        wind_past = past["wind_generation_mwh"].dropna()
        solar_ok = len(solar_past) >= MIN_SAME_HOUR_OBSERVATIONS
        wind_ok = len(wind_past) >= MIN_SAME_HOUR_OBSERVATIONS
        scenario_ok = solar_ok and wind_ok
        reason = "complete"
        if not scenario_ok:
            reason = f"insufficient_history: solar={len(solar_past)}, wind={len(wind_past)}, required={MIN_SAME_HOUR_OBSERVATIONS}"
        record: dict[str, object] = {
            "daily_source_timestamp": daily_time,
            "weekly_source_timestamp": weekly_time,
            "same_hour_window_start_exclusive": origin - pd.Timedelta(days=28),
            "same_hour_window_end_inclusive": origin,
            "same_hour_solar_count": len(solar_past),
            "same_hour_wind_count": len(wind_past),
            "scenario_completeness_reason": reason,
            "daily_solar_forecast_mwh": solar_lookup.get(daily_time, np.nan),
            "daily_wind_forecast_mwh": wind_lookup.get(daily_time, np.nan),
            "weekly_solar_forecast_mwh": solar_lookup.get(weekly_time, np.nan),
            "weekly_wind_forecast_mwh": wind_lookup.get(weekly_time, np.nan),
            "rolling_median_solar_forecast_mwh": solar_past.median() if solar_ok else np.nan,
            "rolling_median_wind_forecast_mwh": wind_past.median() if wind_ok else np.nan,
        }
        for label, quantile in [("conservative", 0.25), ("typical", 0.50), ("favourable", 0.75)]:
            record[f"{label}_solar_scenario_mwh"] = solar_past.quantile(quantile) if scenario_ok else np.nan
            record[f"{label}_wind_scenario_mwh"] = wind_past.quantile(quantile) if scenario_ok else np.nan
        records.append(record)
    added = pd.DataFrame.from_records(records, index=planning.index)
    return pd.concat([planning, added], axis=1)


def method_columns(method: str) -> tuple[str, str]:
    if method == "daily_seasonal_naive":
        return "daily_solar_forecast_mwh", "daily_wind_forecast_mwh"
    if method == "weekly_seasonal_naive":
        return "weekly_solar_forecast_mwh", "weekly_wind_forecast_mwh"
    return "rolling_median_solar_forecast_mwh", "rolling_median_wind_forecast_mwh"


def renewable_method_metrics(planning: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    frame = planning.loc[planning["split"].eq(split)]
    for method in METHODS:
        solar_col, wind_col = method_columns(method)
        pairs = {
            "solar": ("actual_solar_mwh", frame[solar_col]),
            "wind": ("actual_wind_mwh", frame[wind_col]),
            "combined_solar_wind": (
                "actual_combined_renewable_mwh",
                frame[solar_col] + frame[wind_col],
            ),
        }
        for resource, (actual_col, predicted) in pairs.items():
            rows.append({
                "split": split,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "resource": resource,
                **metric_row(frame[actual_col], predicted),
            })
    return pd.DataFrame(rows)


def select_method(validation_metrics: pd.DataFrame, hashes: dict[str, str]) -> tuple[str, pd.DataFrame]:
    candidates = validation_metrics.loc[validation_metrics["resource"].eq("combined_solar_wind")]
    selected = str(candidates.sort_values(["mae_mwh", "method"]).iloc[0]["method"])
    selected_row = candidates.loc[candidates["method"].eq(selected)].iloc[0]
    metadata = pd.DataFrame([{
        "selected_method": selected,
        "selected_method_label": METHOD_LABELS[selected],
        "selection_split": "validation",
        "selection_metric": "combined_solar_wind_mae_mwh",
        "validation_combined_mae_mwh": selected_row["mae_mwh"],
        "validation_count": int(selected_row["count"]),
        "test_metrics_used_for_selection": False,
        "minimum_same_hour_observations": MIN_SAME_HOUR_OBSERVATIONS,
        "historical_source_sha256": hashes["data/processed/eia_ciso_hourly_2022_2024.csv"],
        "recursive_source_sha256": hashes["results/recursive/tables/recursive_predictions.csv"],
    }])
    return selected, metadata


def training_thresholds(historical: pd.DataFrame) -> pd.DataFrame:
    train = historical.loc[historical["period"].between(*SPLITS["train"], inclusive="both")].copy()
    train["actual_residual_mwh"] = train["demand_mwh"] - train["solar_generation_mwh"] - train["wind_generation_mwh"]
    train["actual_renewable_share_pct"] = (
        100 * (train["solar_generation_mwh"] + train["wind_generation_mwh"]) / train["demand_mwh"]
    )
    train["actual_residual_ramp_mwh"] = train["actual_residual_mwh"].diff()
    positive_ramps = train.loc[train["actual_residual_ramp_mwh"].gt(0), "actual_residual_ramp_mwh"].dropna()
    definitions = [
        ("high_demand_mwh", 0.90, train["demand_mwh"].dropna(), "demand_mwh"),
        ("high_residual_demand_mwh", 0.90, train["actual_residual_mwh"].dropna(), "demand - solar - wind"),
        ("high_positive_residual_ramp_mwh", 0.90, positive_ramps, "positive consecutive-hour residual-demand ramps"),
        ("low_renewable_share_pct", 0.10, train["actual_renewable_share_pct"].replace([np.inf, -np.inf], np.nan).dropna(), "100 * (solar + wind) / demand"),
    ]
    return pd.DataFrame([
        {
            "threshold_name": name,
            "threshold_value": float(values.quantile(quantile)),
            "quantile": quantile,
            "fit_split": "train",
            "fit_start_utc": SPLITS["train"][0],
            "fit_end_utc": SPLITS["train"][1],
            "training_row_count": int(len(values)),
            "source_definition": definition,
            "validation_rows_used": 0,
            "test_rows_used": 0,
        }
        for name, quantile, values, definition in definitions
    ])


def complete_planning_columns(
    planning: pd.DataFrame, historical: pd.DataFrame, selected: str, thresholds: pd.DataFrame
) -> pd.DataFrame:
    solar_col, wind_col = method_columns(selected)
    planning["selected_renewable_method"] = selected
    planning["selected_solar_forecast_mwh"] = planning[solar_col]
    planning["selected_wind_forecast_mwh"] = planning[wind_col]
    planning["selected_combined_renewable_forecast_mwh"] = planning[solar_col] + planning[wind_col]
    for scenario in ["conservative", "typical", "favourable"]:
        planning[f"{scenario}_combined_renewable_scenario_mwh"] = (
            planning[f"{scenario}_solar_scenario_mwh"] + planning[f"{scenario}_wind_scenario_mwh"]
        )
    planning["forecast_residual_demand_mwh"] = (
        planning["forecast_demand_mwh"] - planning["selected_combined_renewable_forecast_mwh"]
    )
    planning["conservative_residual_demand_scenario_mwh"] = (
        planning["forecast_demand_mwh"] - planning["conservative_combined_renewable_scenario_mwh"]
    )
    planning["typical_residual_demand_scenario_mwh"] = (
        planning["forecast_demand_mwh"] - planning["typical_combined_renewable_scenario_mwh"]
    )
    planning["favourable_residual_demand_scenario_mwh"] = (
        planning["forecast_demand_mwh"] - planning["favourable_combined_renewable_scenario_mwh"]
    )
    planning["forecast_renewable_share_pct"] = (
        100 * planning["selected_combined_renewable_forecast_mwh"] / planning["forecast_demand_mwh"]
    )
    planning = planning.sort_values(["forecast_origin", "horizon"]).reset_index(drop=True)
    planning["forecast_hourly_demand_ramp_mwh"] = planning.groupby("forecast_origin")["forecast_demand_mwh"].diff()
    planning["forecast_hourly_residual_demand_ramp_mwh"] = planning.groupby("forecast_origin")["forecast_residual_demand_mwh"].diff()

    history_actual = historical[["period", "demand_mwh", "solar_generation_mwh", "wind_generation_mwh", "demand_data_complete", "renewable_data_complete"]].rename(columns={
        "period": "target_timestamp",
        "demand_mwh": "historical_actual_demand_mwh",
        "solar_generation_mwh": "actual_solar_mwh",
        "wind_generation_mwh": "actual_wind_mwh",
    })
    planning = planning.drop(columns=[c for c in ["actual_solar_mwh", "actual_wind_mwh"] if c in planning])
    planning = planning.merge(history_actual, on="target_timestamp", how="left", validate="many_to_one")
    planning["actual_measurements_complete"] = (
        planning["historical_actual_demand_mwh"].notna()
        & planning["actual_solar_mwh"].notna()
        & planning["actual_wind_mwh"].notna()
    )
    complete = planning["actual_measurements_complete"]
    planning["actual_combined_renewable_mwh"] = np.where(
        complete, planning["actual_solar_mwh"] + planning["actual_wind_mwh"], np.nan
    )
    planning["actual_residual_demand_mwh"] = np.where(
        complete, planning["historical_actual_demand_mwh"] - planning["actual_combined_renewable_mwh"], np.nan
    )
    planning["actual_renewable_share_pct"] = np.where(
        complete,
        100 * planning["actual_combined_renewable_mwh"] / planning["historical_actual_demand_mwh"],
        np.nan,
    )
    planning["renewable_prediction_error_mwh"] = planning["selected_combined_renewable_forecast_mwh"] - planning["actual_combined_renewable_mwh"]
    planning["residual_demand_prediction_error_mwh"] = planning["forecast_residual_demand_mwh"] - planning["actual_residual_demand_mwh"]
    planning["renewable_share_prediction_error_pct_points"] = planning["forecast_renewable_share_pct"] - planning["actual_renewable_share_pct"]

    values = thresholds.set_index("threshold_name")["threshold_value"]
    planning["high_demand_alert"] = planning["forecast_demand_mwh"].gt(values["high_demand_mwh"])
    planning["high_residual_demand_alert"] = planning["forecast_residual_demand_mwh"].gt(values["high_residual_demand_mwh"])
    planning["high_upward_ramp_alert"] = planning["forecast_hourly_residual_demand_ramp_mwh"].gt(values["high_positive_residual_ramp_mwh"])
    planning["low_renewable_share_alert"] = planning["forecast_renewable_share_pct"].lt(values["low_renewable_share_pct"])
    planning["residual_demand_below_zero_diagnostic"] = planning["forecast_residual_demand_mwh"].lt(0)
    return planning


def planning_metrics(frame: pd.DataFrame, grouping: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    iterator = [((), frame)] if not grouping else frame.groupby(grouping, dropna=False, observed=True)
    for keys, group in iterator:
        if grouping:
            keys = keys if isinstance(keys, tuple) else (keys,)
            base = dict(zip(grouping, keys))
        else:
            base = {}
        for metric_name, actual, predicted, unit in [
            ("renewable_combined", "actual_combined_renewable_mwh", "selected_combined_renewable_forecast_mwh", "MWh"),
            ("residual_demand", "actual_residual_demand_mwh", "forecast_residual_demand_mwh", "MWh"),
            ("renewable_share", "actual_renewable_share_pct", "forecast_renewable_share_pct", "percentage_points"),
        ]:
            metric = metric_row(group[actual], group[predicted])
            rows.append({**base, "metric": metric_name, "unit": unit, **metric})
    return pd.DataFrame(rows)


def daily_summary(planning: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (split, forecast_date), day in planning.groupby(["split", "forecast_date"], sort=True):
        def peak(value_col: str) -> tuple[float, object]:
            valid = day[value_col].dropna()
            if valid.empty:
                return np.nan, pd.NaT
            idx = valid.idxmax()
            return float(day.loc[idx, value_col]), day.loc[idx, "target_timestamp"]
        def trough(value_col: str) -> tuple[float, object]:
            valid = day[value_col].dropna()
            if valid.empty:
                return np.nan, pd.NaT
            idx = valid.idxmin()
            return float(day.loc[idx, value_col]), day.loc[idx, "target_timestamp"]
        demand_peak, demand_time = peak("forecast_demand_mwh")
        residual_peak, residual_time = peak("forecast_residual_demand_mwh")
        share_low, share_time = trough("forecast_renewable_share_pct")
        ramp_peak, ramp_time = peak("forecast_hourly_residual_demand_ramp_mwh")
        conservative_peak, _ = peak("conservative_residual_demand_scenario_mwh")
        favourable_peak, _ = peak("favourable_residual_demand_scenario_mwh")
        complete_hours = int(day[["forecast_demand_mwh", "selected_combined_renewable_forecast_mwh", "conservative_combined_renewable_scenario_mwh", "typical_combined_renewable_scenario_mwh", "favourable_combined_renewable_scenario_mwh"]].notna().all(axis=1).sum())
        records.append({
            "split": split,
            "forecast_date": forecast_date,
            "demand_peak_mwh": demand_peak,
            "demand_peak_time_utc": demand_time,
            "residual_demand_peak_mwh": residual_peak,
            "residual_demand_peak_time_utc": residual_time,
            "lowest_forecast_renewable_share_pct": share_low,
            "lowest_forecast_renewable_share_time_utc": share_time,
            "maximum_upward_residual_ramp_mwh": ramp_peak,
            "maximum_upward_residual_ramp_time_utc": ramp_time,
            "average_forecast_renewable_share_pct": day["forecast_renewable_share_pct"].mean(),
            "hours_above_high_demand_threshold": int(day["high_demand_alert"].sum()),
            "hours_above_high_residual_threshold": int(day["high_residual_demand_alert"].sum()),
            "hours_with_high_upward_ramp_alert": int(day["high_upward_ramp_alert"].sum()),
            "hours_with_low_renewable_share_alert": int(day["low_renewable_share_alert"].sum()),
            "conservative_peak_residual_demand_mwh": conservative_peak,
            "favourable_peak_residual_demand_mwh": favourable_peak,
            "planning_complete_hours": complete_hours,
            "planning_data_completeness_pct": 100 * complete_hours / 24,
        })
    return pd.DataFrame(records)


def representative_days(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ["validation", "test"]:
        candidates = summary.loc[summary["split"].eq(split)].sort_values("forecast_date")
        row = candidates.iloc[len(candidates) // 2].to_dict()
        row["selection_rule"] = "chronological middle forecast day"
        row["operator_style_summary"] = (
            f"{split.title()} {pd.Timestamp(row['forecast_date']):%Y-%m-%d} UTC: forecast residual-demand peak "
            f"{row['residual_demand_peak_mwh']:,.0f} MWh at {pd.Timestamp(row['residual_demand_peak_time_utc']):%H:%M} UTC; "
            f"lowest renewable share {row['lowest_forecast_renewable_share_pct']:.1f}% at "
            f"{pd.Timestamp(row['lowest_forecast_renewable_share_time_utc']):%H:%M} UTC; "
            f"indicator hours — high demand {int(row['hours_above_high_demand_threshold'])}, high residual "
            f"{int(row['hours_above_high_residual_threshold'])}, upward ramp {int(row['hours_with_high_upward_ramp_alert'])}, "
            f"low renewable share {int(row['hours_with_low_renewable_share_alert'])}. This is planning context, not a dispatch instruction."
        )
        rows.append(row)
    return pd.DataFrame(rows)


def analysis_tables(planning: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    top_n = 50
    save_csv(planning.nlargest(top_n, "forecast_residual_demand_mwh"), "largest_residual_demand_hours.csv")
    save_csv(planning.nlargest(top_n, "forecast_hourly_residual_demand_ramp_mwh"), "largest_upward_ramps.csv")
    save_csv(planning.nsmallest(top_n, "forecast_renewable_share_pct"), "lowest_renewable_share_hours.csv")
    save_csv(planning.nlargest(top_n, "forecast_demand_mwh"), "largest_demand_hours.csv")
    peak_rows = []
    for split, group in planning.groupby("split"):
        threshold = group["actual_residual_demand_mwh"].dropna().quantile(0.90)
        actual_peak = group.loc[group["actual_residual_demand_mwh"].ge(threshold)]
        metrics = planning_metrics(actual_peak)
        metrics.insert(0, "split", split)
        metrics.insert(1, "actual_split_top_decile_threshold_mwh", threshold)
        peak_rows.append(metrics)
    save_csv(pd.concat(peak_rows, ignore_index=True), "actual_top_decile_residual_performance.csv")
    daily_errors = planning.groupby(["split", "forecast_date"], as_index=False).agg(
        residual_demand_mae_mwh=("residual_demand_prediction_error_mwh", lambda x: x.abs().mean()),
        renewable_mae_mwh=("renewable_prediction_error_mwh", lambda x: x.abs().mean()),
        complete_actual_hours=("actual_measurements_complete", "sum"),
    ).sort_values("residual_demand_mae_mwh", ascending=False)
    save_csv(daily_errors, "daily_planning_errors.csv")


def figure_setup(title: str, ylabel: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    return fig, ax


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_figures(
    planning: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    representatives: pd.DataFrame,
) -> None:
    for split, metrics, filename in [
        ("Validation", validation_metrics, "01_renewable_method_validation_comparison.png"),
        ("Test", test_metrics, "02_renewable_method_test_comparison.png"),
    ]:
        data = metrics.loc[metrics["resource"].eq("combined_solar_wind")]
        fig, ax = figure_setup(f"{split} combined-renewable method comparison", "MAE (MWh)")
        ax.bar(data["method_label"], data["mae_mwh"], color=["#4C78A8", "#F58518", "#54A24B"])
        ax.tick_params(axis="x", rotation=15)
        save_figure(fig, filename)

    representative_dates = dict(zip(representatives["split"], pd.to_datetime(representatives["forecast_date"])))
    for split, filename in [("validation", "03_representative_validation_day.png"), ("test", "04_representative_test_day.png")]:
        day = planning.loc[(planning["split"].eq(split)) & (planning["forecast_date"].eq(representative_dates[split]))]
        fig, ax = figure_setup(f"Representative {split} day: demand and renewable forecast", "MWh")
        ax.plot(day["target_timestamp"], day["forecast_demand_mwh"], label="Forecast demand", linewidth=2)
        ax.plot(day["target_timestamp"], day["selected_combined_renewable_forecast_mwh"], label="Forecast solar + wind", linewidth=2)
        ax.plot(day["target_timestamp"], day["actual_combined_renewable_mwh"], label="Actual solar + wind", linestyle="--")
        ax.set_xlabel("Target time (UTC)"); ax.legend(); ax.tick_params(axis="x", rotation=30)
        save_figure(fig, filename)

    day = planning.loc[(planning["split"].eq("test")) & (planning["forecast_date"].eq(representative_dates["test"]))]
    fig, ax = figure_setup("Representative test day: residual-demand scenarios", "Residual demand (MWh)")
    for col, label in [("conservative_residual_demand_scenario_mwh", "Conservative renewable"), ("typical_residual_demand_scenario_mwh", "Typical renewable"), ("favourable_residual_demand_scenario_mwh", "Favourable renewable")]:
        ax.plot(day["target_timestamp"], day[col], label=label)
    ax.set_xlabel("Target time (UTC)"); ax.legend(); ax.tick_params(axis="x", rotation=30)
    save_figure(fig, "05_representative_test_residual_scenarios.png")

    sample = planning.iloc[::24]
    fig, ax = figure_setup("Forecast versus actual residual demand", "Residual demand (MWh)")
    ax.scatter(sample["actual_residual_demand_mwh"], sample["forecast_residual_demand_mwh"], s=13, alpha=0.5)
    limits = [np.nanmin([sample["actual_residual_demand_mwh"].min(), sample["forecast_residual_demand_mwh"].min()]), np.nanmax([sample["actual_residual_demand_mwh"].max(), sample["forecast_residual_demand_mwh"].max()])]
    ax.plot(limits, limits, "k--", label="Perfect agreement"); ax.set_xlabel("Actual residual demand (MWh)"); ax.legend()
    save_figure(fig, "06_forecast_vs_actual_residual_demand.png")

    horizon = planning.groupby("horizon").agg(residual_mae=("residual_demand_prediction_error_mwh", lambda x: x.abs().mean()), renewable_mae=("renewable_prediction_error_mwh", lambda x: x.abs().mean())).reset_index()
    for value, title, ylabel, filename in [
        ("residual_mae", "Residual-demand error by horizon", "MAE (MWh)", "07_residual_error_by_horizon.png"),
        ("renewable_mae", "Renewable prediction error by horizon", "MAE (MWh)", "08_renewable_error_by_horizon.png"),
    ]:
        fig, ax = figure_setup(title, ylabel); ax.plot(horizon["horizon"], horizon[value], marker="o"); ax.set_xlabel("Forecast horizon (hours)"); ax.set_xticks(range(1, 25, 2)); save_figure(fig, filename)

    hour = planning.assign(utc_hour=planning["target_timestamp"].dt.hour).groupby("utc_hour")["forecast_renewable_share_pct"].mean()
    fig, ax = figure_setup("Average forecast renewable share by UTC hour", "Renewable share (%)"); ax.plot(hour.index, hour.values, marker="o"); ax.set_xlabel("UTC hour"); ax.set_xticks(range(24)); save_figure(fig, "09_renewable_share_by_utc_hour.png")

    month = planning.assign(month=planning["target_timestamp"].dt.month).groupby("month")["residual_demand_prediction_error_mwh"].apply(lambda x: x.abs().mean())
    fig, ax = figure_setup("Residual-demand MAE by month", "MAE (MWh)"); ax.bar(month.index.astype(str), month.values); ax.set_xlabel("UTC calendar month"); save_figure(fig, "10_residual_mae_by_month.png")

    peak = planning.loc[planning["actual_residual_demand_mwh"].ge(planning["actual_residual_demand_mwh"].quantile(0.90))]
    fig, ax = figure_setup("Performance during actual high residual-demand hours", "Forecast residual demand (MWh)"); ax.scatter(peak["actual_residual_demand_mwh"], peak["forecast_residual_demand_mwh"], s=14, alpha=0.5); ax.set_xlabel("Actual residual demand (MWh)"); save_figure(fig, "11_peak_residual_demand_performance.png")

    ramps = planning.copy()
    ramps["actual_residual_ramp_mwh"] = ramps.groupby("forecast_origin")["actual_residual_demand_mwh"].diff()
    ramps = ramps.dropna(subset=["forecast_hourly_residual_demand_ramp_mwh", "actual_residual_ramp_mwh"])
    fig, ax = figure_setup("Upward-ramp performance", "Forecast residual-demand ramp (MWh/h)"); ax.scatter(ramps["actual_residual_ramp_mwh"], ramps["forecast_hourly_residual_demand_ramp_mwh"], s=12, alpha=0.4); ax.axhline(0, color="black", linewidth=0.7); ax.axvline(0, color="black", linewidth=0.7); ax.set_xlabel("Actual residual-demand ramp (MWh/h)"); save_figure(fig, "12_upward_ramp_performance.png")

    alert_cols = ["hours_above_high_demand_threshold", "hours_above_high_residual_threshold", "hours_with_high_upward_ramp_alert", "hours_with_low_renewable_share_alert"]
    alert_labels = ["High demand", "High residual", "Upward ramp", "Low renewable share"]
    fig, ax = figure_setup("Daily planning-indicator counts", "Indicator hours per day")
    rolling = summary.set_index("forecast_date")[alert_cols].rolling(14, min_periods=1).mean()
    for col, label in zip(alert_cols, alert_labels): ax.plot(rolling.index, rolling[col], label=label)
    ax.set_xlabel("Forecast date (UTC, 14-day rolling average)"); ax.legend(ncol=2); save_figure(fig, "13_daily_planning_alert_counts.png")

    scenario_means = planning[["conservative_combined_renewable_scenario_mwh", "typical_combined_renewable_scenario_mwh", "favourable_combined_renewable_scenario_mwh"]].mean()
    fig, ax = figure_setup("Renewable scenario comparison", "Mean renewable availability (MWh)"); ax.bar(["Conservative (P25)", "Typical (P50)", "Favourable (P75)"], scenario_means.values, color=["#E45756", "#4C78A8", "#54A24B"]); save_figure(fig, "14_scenario_comparison.png")


def fmt(value: float, decimals: int = 2) -> str:
    return "not available" if pd.isna(value) else f"{value:,.{decimals}f}"


def write_policy() -> None:
    POLICY_PATH.write_text(
        """# Renewable-aware planning policy

## Purpose

This layer supports human review of 24-hour forecasts. It is not an automated grid controller and does not provide dispatch, safety, price, cost, or savings claims.

## Fixed inputs and splits

The layer reuses saved Recursive XGBoost demand predictions. Training is 2022-01-01 through 2023-12-31 UTC, validation is 2024-01-01 through 2024-06-30 UTC, and test is 2024-07-01 through 2024-12-31 UTC. Renewable method selection uses validation combined-renewable MAE only; test results cannot change it.

## Past-only renewable methods

Daily and weekly seasonal-naive forecasts read the reported value 24 or 168 hours before the target. The rolling method uses complete, same-UTC-hour observations in the interval `(origin - 28 days, origin]`. At least seven observations per resource are required for a rolling estimate or scenario. Missing values are retained, never filled. Negative reported solar values are preserved and are not clipped.

## Scenarios

Conservative, typical, and favourable renewable availability are the 25th, 50th, and 75th percentiles of the past-only same-hour sample, calculated separately for solar and wind and then combined. They are empirical planning scenarios, not statistically calibrated prediction intervals. A missing scenario retains nulls and records its history counts and reason.

## Residual demand and indicators

Residual demand equals forecast demand minus forecast solar-plus-wind. It is not floored at zero. It is not complete physical grid balance because it excludes other generation, imports, exports, storage, losses, and network constraints. Thresholds are fitted from training observations only. Alerts are descriptive planning indicators, not safety guarantees or operating instructions.
""",
        encoding="utf-8",
    )


def write_findings(
    selected: str,
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    overall: pd.DataFrame,
    by_horizon: pd.DataFrame,
    planning: pd.DataFrame,
    summary: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> None:
    val = validation_metrics.query("method == @selected and resource == 'combined_solar_wind'").iloc[0]
    test_combined = test_metrics.query("method == @selected and resource == 'combined_solar_wind'").iloc[0]
    test_solar = test_metrics.query("method == @selected and resource == 'solar'").iloc[0]
    test_wind = test_metrics.query("method == @selected and resource == 'wind'").iloc[0]
    residual = overall.query("split == 'test' and metric == 'residual_demand'").iloc[0]
    share = overall.query("split == 'test' and metric == 'renewable_share'").iloc[0]
    htest = by_horizon.query("split == 'test' and metric == 'residual_demand'")
    early = htest.loc[htest["horizon"].le(6), "mae_mwh"].mean()
    late = htest.loc[htest["horizon"].ge(19), "mae_mwh"].mean()
    later_statement = "higher" if late > early else "lower or equal"
    worst_days = planning.groupby(["split", "forecast_date"])["residual_demand_prediction_error_mwh"].apply(lambda x: x.abs().mean()).nlargest(5)
    indicator_counts = {col: int(planning[col].sum()) for col in ["high_demand_alert", "high_residual_demand_alert", "high_upward_ramp_alert", "low_renewable_share_alert"]}
    scenario_complete = planning.dropna(subset=["conservative_residual_demand_scenario_mwh", "favourable_residual_demand_scenario_mwh"])
    scenario_effect = (scenario_complete["conservative_residual_demand_scenario_mwh"] - scenario_complete["favourable_residual_demand_scenario_mwh"])
    negative_count = int((planning["actual_solar_mwh"] < 0).sum())
    incomplete_count = int((~planning["actual_measurements_complete"]).sum())
    largest = planning.nlargest(1, "forecast_residual_demand_mwh").iloc[0]
    ramp = planning.nlargest(1, "forecast_hourly_residual_demand_ramp_mwh").iloc[0]
    test_rows = planning.loc[planning["split"].eq("test")].copy()
    test_peak_threshold = test_rows["actual_residual_demand_mwh"].dropna().quantile(0.90)
    test_peak_rows = test_rows.loc[test_rows["actual_residual_demand_mwh"].ge(test_peak_threshold)]
    test_peak_metric = metric_row(test_peak_rows["actual_residual_demand_mwh"], test_peak_rows["forecast_residual_demand_mwh"])
    ramp_rows = planning.sort_values(["forecast_origin", "horizon"]).copy()
    ramp_rows["actual_residual_ramp_mwh"] = ramp_rows.groupby("forecast_origin")["actual_residual_demand_mwh"].diff()
    ramp_metric = metric_row(ramp_rows["actual_residual_ramp_mwh"], ramp_rows["forecast_hourly_residual_demand_ramp_mwh"])
    ramp_complete = ramp_rows.dropna(subset=["actual_residual_ramp_mwh", "forecast_hourly_residual_demand_ramp_mwh"])
    ramp_direction_agreement = (
        np.sign(ramp_complete["actual_residual_ramp_mwh"]) == np.sign(ramp_complete["forecast_hourly_residual_demand_ramp_mwh"])
    ).mean() * 100
    threshold_lines = "\n".join(f"- `{r.threshold_name}`: {r.threshold_value:,.3f} ({int(r.training_row_count):,} training rows)." for r in thresholds.itertuples())
    worst_lines = "\n".join(f"- {split} {pd.Timestamp(date):%Y-%m-%d}: residual-demand MAE {value:,.2f} MWh." for (split, date), value in worst_days.items())
    FINDINGS_PATH.write_text(f"""# Renewable-aware planning findings

## Method selection and renewable accuracy

The selected renewable method is **{METHOD_LABELS[selected]}**. It was selected before test evaluation because its validation combined solar-and-wind MAE was {val.mae_mwh:,.2f} MWh (RMSE {val.rmse_mwh:,.2f} MWh, bias {val.bias_mwh:,.2f} MWh; n={int(val['count']):,}). Selection used validation combined-renewable MAE only.

On test data, combined-renewable MAE was {test_combined.mae_mwh:,.2f} MWh, RMSE was {test_combined.rmse_mwh:,.2f} MWh, bias was {test_combined.bias_mwh:,.2f} MWh, MAPE was {test_combined.mape_pct:,.2f}%, and sMAPE was {test_combined.smape_pct:,.2f}% (n={int(test_combined['count']):,}). Test solar MAE was {test_solar.mae_mwh:,.2f} MWh and wind MAE was {test_wind.mae_mwh:,.2f} MWh.

## Residual demand and renewable share

Test residual-demand MAE was {residual.mae_mwh:,.2f} MWh, RMSE was {residual.rmse_mwh:,.2f} MWh, and bias was {residual.bias_mwh:,.2f} MWh (n={int(residual['count']):,}). Test renewable-share MAE was {share.mae_mwh:,.2f} percentage points, RMSE was {share.rmse_mwh:,.2f} points, and bias was {share.bias_mwh:,.2f} points (n={int(share['count']):,}).

Residual-demand MAE averaged {early:,.2f} MWh over horizons 1–6 and {late:,.2f} MWh over horizons 19–24, so later-horizon error was {later_statement} in this saved evaluation. This is descriptive, not causal.

## Peaks, ramps, and indicators

The largest forecast residual demand was {largest.forecast_residual_demand_mwh:,.2f} MWh at {largest.target_timestamp:%Y-%m-%d %H:%M} UTC ({largest['split']}). The largest forecast upward residual-demand ramp was {ramp.forecast_hourly_residual_demand_ramp_mwh:,.2f} MWh/h at {ramp.target_timestamp:%Y-%m-%d %H:%M} UTC. These identify review priorities, not dispatch commands.

During the test split's actual top-decile residual-demand hours (threshold {test_peak_threshold:,.2f} MWh), forecast residual-demand MAE was {test_peak_metric['mae_mwh']:,.2f} MWh and bias was {test_peak_metric['bias_mwh']:,.2f} MWh (n={test_peak_metric['count']:,}). Across complete within-forecast ramp comparisons, residual-ramp MAE was {ramp_metric['mae_mwh']:,.2f} MWh/h and forecast/actual ramp direction agreed {ramp_direction_agreement:,.2f}% of the time (n={ramp_metric['count']:,}).

Across {len(planning):,} planning hours, high-demand indicators occurred {indicator_counts['high_demand_alert']:,} times, high-residual indicators {indicator_counts['high_residual_demand_alert']:,} times, high-upward-ramp indicators {indicator_counts['high_upward_ramp_alert']:,} times, and low-renewable-share indicators {indicator_counts['low_renewable_share_alert']:,} times.

Training-only thresholds were:

{threshold_lines}

## Scenarios

The renewable P25/P50/P75 scenarios are empirical planning scenarios, not calibrated prediction intervals. Across {len(scenario_complete):,} complete scenario rows, conservative-renewable residual demand exceeded favourable-renewable residual demand by {scenario_effect.mean():,.2f} MWh on average (maximum {scenario_effect.max():,.2f} MWh). No residual-demand value was floored at zero.

## Greatest daily errors

{worst_lines}

## Data limitations and interpretation

There were {incomplete_count:,} planning rows without complete actual demand, solar, and wind measurements; they remain in the master table and are excluded only from metrics. The evaluation period contains {negative_count:,} negative reported solar observations. They remain unmodified in predictions, scenarios, actuals, and metrics. The source also has documented renewable gaps, so method counts can differ.

Residual demand here subtracts only solar and wind from demand. It is **not complete physical grid balance**: other generation, imports, exports, storage, losses, reserves, and network constraints are outside this dataset. No electricity prices, costs, dispatch decisions, savings, or automatic control actions are estimated.

## Streamlit-ready outputs

The dashboard can safely present the master hourly planning table, daily summaries, selected-method metadata, training-only thresholds, scenario bands, forecast-versus-actual comparisons, horizon/hour/month error views, and clearly labelled planning indicators. It should retain UTC labels, missing-data flags, the empirical-scenario disclaimer, and the residual-demand limitation. It should not convert indicators into automatic operating recommendations.
""", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    historical, recursive, hashes_before = load_inputs()
    validate_input_shape(recursive)
    planning = recursive.rename(columns={"prediction_mwh": "forecast_demand_mwh"}).copy()
    # Actual renewable columns are joined before method evaluation, but no target actual is used as a predictor.
    actual = historical[["period", "solar_generation_mwh", "wind_generation_mwh"]].rename(columns={"period": "target_timestamp", "solar_generation_mwh": "actual_solar_mwh", "wind_generation_mwh": "actual_wind_mwh"})
    planning = planning.merge(actual, on="target_timestamp", how="left", validate="many_to_one")
    planning["actual_combined_renewable_mwh"] = planning["actual_solar_mwh"] + planning["actual_wind_mwh"]
    planning = add_renewable_predictions(planning, historical)
    validation_metrics = renewable_method_metrics(planning, "validation")
    selected, selected_metadata = select_method(validation_metrics, hashes_before)
    # Test metrics are deliberately calculated only after the validation selection is frozen.
    test_metrics = renewable_method_metrics(planning, "test")
    thresholds = training_thresholds(historical)
    planning = complete_planning_columns(planning, historical, selected, thresholds)

    overall = planning_metrics(planning, ["split"])
    by_horizon = planning_metrics(planning, ["split", "horizon"])
    by_hour = planning_metrics(planning.assign(utc_hour=planning["target_timestamp"].dt.hour), ["split", "utc_hour"])
    by_month = planning_metrics(planning.assign(month=planning["target_timestamp"].dt.month), ["split", "month"])
    summary = daily_summary(planning)
    representatives = representative_days(summary)

    save_csv(validation_metrics, "renewable_method_validation_metrics.csv")
    save_csv(test_metrics, "renewable_method_test_metrics.csv")
    save_csv(selected_metadata, "selected_renewable_method.csv")
    save_csv(thresholds, "planning_thresholds.csv")
    save_csv(planning, "renewable_planning_predictions.csv")
    save_csv(summary, "daily_planning_summary.csv")
    save_csv(overall, "planning_metrics_overall.csv")
    save_csv(by_horizon, "planning_metrics_by_horizon.csv")
    save_csv(by_hour, "planning_metrics_by_utc_hour.csv")
    save_csv(by_month, "planning_metrics_by_month.csv")
    save_csv(representatives, "representative_planning_days.csv")
    analysis_tables(planning, thresholds)
    create_figures(planning, validation_metrics, test_metrics, summary, representatives)
    write_policy()
    write_findings(selected, validation_metrics, test_metrics, overall, by_horizon, planning, summary, thresholds)

    hashes_after = {path: sha256(ROOT / path) for path in hashes_before}
    hash_table = pd.DataFrame([
        {"file": path, "sha256_before": before, "sha256_after": hashes_after[path], "unchanged": before == hashes_after[path]}
        for path, before in hashes_before.items()
    ])
    save_csv(hash_table, "planning_upstream_hashes.csv")
    if not hash_table["unchanged"].all():
        raise RuntimeError("An upstream source changed during the planning run.")
    print(f"Renewable planning complete. Selected method: {selected}")
    print(f"Planning rows: {len(planning):,}; tables: {len(list(TABLE_DIR.glob('*.csv')))}; figures: {len(list(FIGURE_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()
