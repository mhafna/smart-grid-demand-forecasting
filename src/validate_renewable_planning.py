"""Independently validate the saved renewable-aware planning outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_PATH = ROOT / "data" / "processed" / "eia_ciso_hourly_2022_2024.csv"
RECURSIVE_PATH = ROOT / "results" / "recursive" / "tables" / "recursive_predictions.csv"
TABLE_DIR = ROOT / "results" / "planning" / "tables"
OUTPUT_PATH = TABLE_DIR / "planning_validation_results.csv"
MIN_HISTORY = 7
SPLITS = {
    "train": (pd.Timestamp("2022-01-01T00"), pd.Timestamp("2023-12-31T23")),
    "validation": (pd.Timestamp("2024-01-01T00"), pd.Timestamp("2024-06-30T23")),
    "test": (pd.Timestamp("2024-07-01T00"), pd.Timestamp("2024-12-31T23")),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: pd.Series, right: pd.Series, tolerance: float = 1e-7) -> bool:
    return bool(np.allclose(
        pd.to_numeric(left, errors="coerce"),
        pd.to_numeric(right, errors="coerce"),
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    ))


def independent_metric(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
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
        "rmse_mwh": float(np.sqrt(np.square(error).mean())) if len(error) else np.nan,
        "bias_mwh": float(error.mean()) if len(error) else np.nan,
        "mape_pct": float((error[nonzero].abs() / y[nonzero].abs()).mean() * 100) if nonzero.any() else np.nan,
        "mape_count": int(nonzero.sum()),
        "smape_pct": float((2 * error[smape_valid].abs() / (y[smape_valid].abs() + p[smape_valid].abs())).mean() * 100) if smape_valid.any() else np.nan,
        "smape_count": int(smape_valid.sum()),
    }


class Report:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(self, check: str, passed: bool, details: str) -> None:
        self.rows.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    def safe(self, check: str, function) -> None:
        try:
            passed, details = function()
            self.add(check, bool(passed), str(details))
        except Exception as exc:  # report every independent failure together
            self.add(check, False, f"{type(exc).__name__}: {exc}")


def load() -> dict[str, pd.DataFrame]:
    paths = {
        "planning": TABLE_DIR / "renewable_planning_predictions.csv",
        "selected": TABLE_DIR / "selected_renewable_method.csv",
        "validation_metrics": TABLE_DIR / "renewable_method_validation_metrics.csv",
        "test_metrics": TABLE_DIR / "renewable_method_test_metrics.csv",
        "thresholds": TABLE_DIR / "planning_thresholds.csv",
        "overall": TABLE_DIR / "planning_metrics_overall.csv",
        "horizon": TABLE_DIR / "planning_metrics_by_horizon.csv",
        "hashes": TABLE_DIR / "planning_upstream_hashes.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing planning outputs: {missing}")
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    frames["historical"] = pd.read_csv(HISTORICAL_PATH)
    frames["recursive"] = pd.read_csv(RECURSIVE_PATH)
    frames["historical"]["period"] = pd.to_datetime(frames["historical"]["period"])
    for col in ["forecast_origin", "forecast_date", "target_timestamp", "daily_source_timestamp", "weekly_source_timestamp", "same_hour_window_start_exclusive", "same_hour_window_end_inclusive"]:
        frames["planning"][col] = pd.to_datetime(frames["planning"][col])
    for col in ["forecast_origin", "forecast_date", "target_timestamp"]:
        frames["recursive"][col] = pd.to_datetime(frames["recursive"][col])
    return frames


def validate_sources(report: Report, f: dict[str, pd.DataFrame]) -> None:
    p = f["planning"]
    r = f["recursive"].loc[f["recursive"]["model"].eq("recursive_xgboost")].copy()

    report.safe("saved_xgboost_rows_reused_exactly", lambda: (
        len(p) == len(r) == 8784
        and not p.duplicated(["forecast_origin", "target_timestamp"]).any()
        and close(
            p.sort_values(["forecast_origin", "horizon"])["forecast_demand_mwh"].reset_index(drop=True),
            r.sort_values(["forecast_origin", "horizon"])["prediction_mwh"].reset_index(drop=True),
        ),
        f"planning rows={len(p):,}; saved XGBoost rows={len(r):,}; no demand retraining output introduced",
    ))

    def dates_and_horizons() -> tuple[bool, str]:
        expected = {"validation": ("2024-01-01", "2024-06-30", 182), "test": ("2024-07-01", "2024-12-31", 184)}
        problems = []
        for split, (start, end, days) in expected.items():
            rows = p.loc[p["split"].eq(split)]
            dates = rows["forecast_date"].dt.strftime("%Y-%m-%d").drop_duplicates()
            horizon_ok = rows.groupby("forecast_origin")["horizon"].apply(set).map(lambda value: value == set(range(1, 25))).all()
            if len(dates) != days or dates.min() != start or dates.max() != end or not horizon_ok:
                problems.append(split)
        return not problems, "Fixed validation/test dates and horizons 1-24 reproduced." if not problems else f"Problems: {problems}"
    report.safe("forecast_dates_and_horizons_unchanged", dates_and_horizons)

    report.safe("renewable_source_timestamps_at_or_before_origin", lambda: (
        p["daily_source_timestamp"].le(p["forecast_origin"]).all()
        and p["weekly_source_timestamp"].le(p["forecast_origin"]).all()
        and p["same_hour_window_end_inclusive"].le(p["forecast_origin"]).all(),
        f"latest daily offset={(p['daily_source_timestamp'] - p['forecast_origin']).max()}; latest weekly offset={(p['weekly_source_timestamp'] - p['forecast_origin']).max()}",
    ))
    report.safe("no_target_day_actual_renewable_used_as_predictor", lambda: (
        p["daily_source_timestamp"].dt.date.lt(p["target_timestamp"].dt.date).all()
        and p["weekly_source_timestamp"].dt.date.lt(p["target_timestamp"].dt.date).all()
        and p["same_hour_window_end_inclusive"].lt(p["target_timestamp"]).all(),
        "Seasonal sources are on earlier dates and rolling windows end before every target timestamp.",
    ))


def validate_rolling_and_scenarios(report: Report, f: dict[str, pd.DataFrame]) -> None:
    p = f["planning"]
    history = f["historical"].set_index("period").sort_index()
    expected: dict[str, list[float | int]] = {
        "solar_count": [], "wind_count": [], "solar_median": [], "wind_median": [],
        "solar_p25": [], "solar_p50": [], "solar_p75": [],
        "wind_p25": [], "wind_p50": [], "wind_p75": [],
    }
    latest_contributor_ok = True
    for row in p.itertuples(index=False):
        start = row.forecast_origin - pd.Timedelta(days=28)
        sample = history.loc[
            (history.index > start)
            & (history.index <= row.forecast_origin)
            & (history.index.hour == row.target_timestamp.hour)
        ]
        if len(sample) and sample.index.max() > row.forecast_origin:
            latest_contributor_ok = False
        solar = sample["solar_generation_mwh"].dropna()
        wind = sample["wind_generation_mwh"].dropna()
        solar_ok = len(solar) >= MIN_HISTORY
        wind_ok = len(wind) >= MIN_HISTORY
        both = solar_ok and wind_ok
        expected["solar_count"].append(len(solar)); expected["wind_count"].append(len(wind))
        expected["solar_median"].append(solar.median() if solar_ok else np.nan)
        expected["wind_median"].append(wind.median() if wind_ok else np.nan)
        for resource, values in [("solar", solar), ("wind", wind)]:
            for label, quantile in [("p25", .25), ("p50", .50), ("p75", .75)]:
                expected[f"{resource}_{label}"].append(values.quantile(quantile) if both else np.nan)

    rolling_ok = (
        latest_contributor_ok
        and np.array_equal(p["same_hour_solar_count"].to_numpy(), np.array(expected["solar_count"]))
        and np.array_equal(p["same_hour_wind_count"].to_numpy(), np.array(expected["wind_count"]))
        and close(p["rolling_median_solar_forecast_mwh"], pd.Series(expected["solar_median"]))
        and close(p["rolling_median_wind_forecast_mwh"], pd.Series(expected["wind_median"]))
    )
    report.add("rolling_same_hour_windows_past_only_and_exact", rolling_ok, f"Recomputed all {len(p):,} rolling windows; minimum required history={MIN_HISTORY}.")

    scenario_map = {
        "conservative_solar_scenario_mwh": "solar_p25", "typical_solar_scenario_mwh": "solar_p50", "favourable_solar_scenario_mwh": "solar_p75",
        "conservative_wind_scenario_mwh": "wind_p25", "typical_wind_scenario_mwh": "wind_p50", "favourable_wind_scenario_mwh": "wind_p75",
    }
    exact = all(close(p[column], pd.Series(expected[key])) for column, key in scenario_map.items())
    report.add("scenario_percentiles_exactly_recomputed", exact, "P25, P50, and P75 were independently recomputed for solar and wind from every past-only window.")

    complete = p[["conservative_combined_renewable_scenario_mwh", "typical_combined_renewable_scenario_mwh", "favourable_combined_renewable_scenario_mwh"]].notna().all(axis=1)
    renewable_order = (
        p.loc[complete, "conservative_combined_renewable_scenario_mwh"].le(p.loc[complete, "typical_combined_renewable_scenario_mwh"] + 1e-8).all()
        and p.loc[complete, "typical_combined_renewable_scenario_mwh"].le(p.loc[complete, "favourable_combined_renewable_scenario_mwh"] + 1e-8).all()
    )
    residual_order = (
        p.loc[complete, "conservative_residual_demand_scenario_mwh"].ge(p.loc[complete, "typical_residual_demand_scenario_mwh"] - 1e-8).all()
        and p.loc[complete, "typical_residual_demand_scenario_mwh"].ge(p.loc[complete, "favourable_residual_demand_scenario_mwh"] - 1e-8).all()
    )
    report.add("scenario_renewable_order_valid", renewable_order, f"Checked {int(complete.sum()):,} complete scenario rows: conservative <= typical <= favourable.")
    report.add("scenario_residual_order_reversed", residual_order, f"Checked {int(complete.sum()):,} complete rows: conservative residual >= typical residual >= favourable residual.")


def validate_selection_and_metrics(report: Report, f: dict[str, pd.DataFrame]) -> None:
    selected = f["selected"].iloc[0]
    validation = f["validation_metrics"]
    test = f["test_metrics"]
    combined = validation.loc[validation["resource"].eq("combined_solar_wind")].sort_values(["mae_mwh", "method"])
    expected = combined.iloc[0]["method"]
    report.add(
        "renewable_method_selected_on_validation_combined_mae_only",
        selected["selected_method"] == expected and selected["selection_split"] == "validation" and selected["selection_metric"] == "combined_solar_wind_mae_mwh",
        f"saved={selected['selected_method']}; independent validation winner={expected}",
    )
    report.add(
        "test_results_did_not_influence_selection",
        str(selected["test_metrics_used_for_selection"]).lower() == "false" and "test" not in str(selected["selection_split"]).lower(),
        "Selection metadata freezes validation before test evaluation and contains no test selection criterion.",
    )

    p = f["planning"]
    method_cols = {
        "daily_seasonal_naive": ("daily_solar_forecast_mwh", "daily_wind_forecast_mwh"),
        "weekly_seasonal_naive": ("weekly_solar_forecast_mwh", "weekly_wind_forecast_mwh"),
        "past_same_hour_rolling_median": ("rolling_median_solar_forecast_mwh", "rolling_median_wind_forecast_mwh"),
    }
    metric_ok = True
    count_ok = True
    for saved_table, split in [(validation, "validation"), (test, "test")]:
        frame = p.loc[p["split"].eq(split)]
        for saved in saved_table.itertuples(index=False):
            solar, wind = method_cols[saved.method]
            if saved.resource == "solar": actual, predicted = frame["actual_solar_mwh"], frame[solar]
            elif saved.resource == "wind": actual, predicted = frame["actual_wind_mwh"], frame[wind]
            else: actual, predicted = frame["actual_combined_renewable_mwh"], frame[solar] + frame[wind]
            calc = independent_metric(actual, predicted)
            count_ok &= int(saved.count) == calc["count"]
            metric_ok &= all(
                (pd.isna(getattr(saved, key)) and pd.isna(value)) or np.isclose(getattr(saved, key), value, rtol=1e-7, atol=1e-7)
                for key, value in calc.items()
            )
    report.add("renewable_metrics_exactly_reproduced", metric_ok, "All validation/test method-resource metrics independently reproduced.")
    report.add("renewable_metric_counts_match_predictions", count_ok, f"Counts checked against {len(p):,} saved planning rows.")


def validate_thresholds(report: Report, f: dict[str, pd.DataFrame]) -> None:
    h = f["historical"]
    train = h.loc[h["period"].between(*SPLITS["train"], inclusive="both")].copy()
    residual = train["demand_mwh"] - train["solar_generation_mwh"] - train["wind_generation_mwh"]
    share = 100 * (train["solar_generation_mwh"] + train["wind_generation_mwh"]) / train["demand_mwh"]
    ramps = residual.diff()
    expected = {
        "high_demand_mwh": (train["demand_mwh"].dropna().quantile(.9), train["demand_mwh"].notna().sum()),
        "high_residual_demand_mwh": (residual.dropna().quantile(.9), residual.notna().sum()),
        "high_positive_residual_ramp_mwh": (ramps[ramps.gt(0)].dropna().quantile(.9), ramps.gt(0).sum()),
        "low_renewable_share_pct": (share.replace([np.inf, -np.inf], np.nan).dropna().quantile(.1), share.replace([np.inf, -np.inf], np.nan).notna().sum()),
    }
    saved = f["thresholds"].set_index("threshold_name")
    exact = all(np.isclose(saved.loc[name, "threshold_value"], value) and int(saved.loc[name, "training_row_count"]) == int(count) for name, (value, count) in expected.items())
    metadata = saved["fit_split"].eq("train").all() and saved["validation_rows_used"].eq(0).all() and saved["test_rows_used"].eq(0).all()
    report.add("planning_thresholds_exactly_recomputed_from_training", exact and metadata, f"All four threshold values/counts reproduced from {SPLITS['train'][0]} through {SPLITS['train'][1]} UTC only.")


def validate_formulas_and_quality(report: Report, f: dict[str, pd.DataFrame]) -> None:
    p = f["planning"].sort_values(["forecast_origin", "horizon"]).reset_index(drop=True)
    checks = {
        "selected combined renewable": (p["selected_combined_renewable_forecast_mwh"], p["selected_solar_forecast_mwh"] + p["selected_wind_forecast_mwh"]),
        "forecast residual": (p["forecast_residual_demand_mwh"], p["forecast_demand_mwh"] - p["selected_combined_renewable_forecast_mwh"]),
        "conservative residual": (p["conservative_residual_demand_scenario_mwh"], p["forecast_demand_mwh"] - p["conservative_combined_renewable_scenario_mwh"]),
        "typical residual": (p["typical_residual_demand_scenario_mwh"], p["forecast_demand_mwh"] - p["typical_combined_renewable_scenario_mwh"]),
        "favourable residual": (p["favourable_residual_demand_scenario_mwh"], p["forecast_demand_mwh"] - p["favourable_combined_renewable_scenario_mwh"]),
        "forecast share": (p["forecast_renewable_share_pct"], 100 * p["selected_combined_renewable_forecast_mwh"] / p["forecast_demand_mwh"]),
        "actual combined": (p["actual_combined_renewable_mwh"], p["actual_solar_mwh"] + p["actual_wind_mwh"]),
        "actual residual": (p["actual_residual_demand_mwh"], p["historical_actual_demand_mwh"] - p["actual_combined_renewable_mwh"]),
        "renewable error": (p["renewable_prediction_error_mwh"], p["selected_combined_renewable_forecast_mwh"] - p["actual_combined_renewable_mwh"]),
        "residual error": (p["residual_demand_prediction_error_mwh"], p["forecast_residual_demand_mwh"] - p["actual_residual_demand_mwh"]),
        "demand ramp": (p["forecast_hourly_demand_ramp_mwh"], p.groupby("forecast_origin")["forecast_demand_mwh"].diff()),
        "residual ramp": (p["forecast_hourly_residual_demand_ramp_mwh"], p.groupby("forecast_origin")["forecast_residual_demand_mwh"].diff()),
    }
    failed = [name for name, (saved, expected) in checks.items() if not close(saved, expected)]
    report.add("all_planning_formulas_reproduce_exactly", not failed, "All core formulas reproduced." if not failed else f"Failed: {failed}")
    report.add("residual_demand_not_floored", close(p["forecast_residual_demand_mwh"], p["forecast_demand_mwh"] - p["selected_combined_renewable_forecast_mwh"]), f"Below-zero diagnostic rows={int(p['residual_demand_below_zero_diagnostic'].astype(str).str.lower().eq('true').sum())}.")

    incomplete = ~p["actual_measurements_complete"].astype(str).str.lower().eq("true")
    derived = ["actual_combined_renewable_mwh", "actual_residual_demand_mwh", "actual_renewable_share_pct", "renewable_prediction_error_mwh", "residual_demand_prediction_error_mwh"]
    null_ok = p.loc[incomplete, derived].isna().all().all()
    report.add("incomplete_actuals_retained_and_excluded_not_filled", null_ok and incomplete.sum() > 0, f"Incomplete rows retained={int(incomplete.sum()):,}; all derived actual/evaluation values remain null.")

    history = f["historical"].set_index("period")
    joined_actual = p.set_index("target_timestamp")["actual_solar_mwh"]
    expected_actual = history.reindex(joined_actual.index)["solar_generation_mwh"]
    negative_ok = close(joined_actual.reset_index(drop=True), expected_actual.reset_index(drop=True)) and (p["actual_solar_mwh"] < 0).any()
    report.add("negative_solar_values_preserved", negative_ok, f"Preserved {int((p['actual_solar_mwh'] < 0).sum()):,} negative actual solar rows; no clipping detected.")

    forbidden = [column for column in p.columns if any(word in column.lower() for word in ["price", "cost", "saving", "dispatch", "storage_action", "control_action"])]
    report.add("no_cost_price_savings_or_control_values_introduced", not forbidden, "No prohibited operational-value columns found." if not forbidden else f"Found: {forbidden}")

    overall_ok = True
    for saved in f["overall"].itertuples(index=False):
        frame = p.loc[p["split"].eq(saved.split)]
        pair = {
            "renewable_combined": ("actual_combined_renewable_mwh", "selected_combined_renewable_forecast_mwh"),
            "residual_demand": ("actual_residual_demand_mwh", "forecast_residual_demand_mwh"),
            "renewable_share": ("actual_renewable_share_pct", "forecast_renewable_share_pct"),
        }[saved.metric]
        calc = independent_metric(frame[pair[0]], frame[pair[1]])
        overall_ok &= int(saved.count) == calc["count"] and np.isclose(saved.mae_mwh, calc["mae_mwh"])
    report.add("planning_metric_counts_and_mae_reproduced", overall_ok, "Overall split/metric counts and MAE values independently reproduced.")


def validate_hashes(report: Report, f: dict[str, pd.DataFrame]) -> None:
    failures = []
    for row in f["hashes"].itertuples(index=False):
        current = sha256(ROOT / row.file)
        if current != row.sha256_before or current != row.sha256_after or str(row.unchanged).lower() != "true":
            failures.append(row.file)
    report.add("upstream_hashes_remain_unchanged", not failures, f"All {len(f['hashes'])} recorded sources match before, after, and current SHA-256 hashes." if not failures else f"Changed: {failures}")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    report = Report()
    try:
        frames = load()
        validate_sources(report, frames)
        validate_rolling_and_scenarios(report, frames)
        validate_selection_and_metrics(report, frames)
        validate_thresholds(report, frames)
        validate_formulas_and_quality(report, frames)
        validate_hashes(report, frames)
    except Exception as exc:
        report.add("validation_run_completed", False, f"{type(exc).__name__}: {exc}")
    output = pd.DataFrame(report.rows)
    output.to_csv(OUTPUT_PATH, index=False)
    failed = output.loc[output["status"].ne("PASS")]
    print(f"Independent renewable-planning validation: {len(output) - len(failed)}/{len(output)} checks passed.")
    if not failed.empty:
        print(failed.to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
