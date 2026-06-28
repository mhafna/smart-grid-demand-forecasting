"""Independent integrity checks for the read-only Streamlit dashboard."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard_data import (
    DAILY_NAIVE_MODEL,
    ROOT,
    SELECTED_DEMAND_MODEL,
    SOURCE_FILES,
    available_forecast_dates,
    dataframe_to_csv_bytes,
    data_quality_summary,
    headline_metrics,
    load_table,
    selected_day_planning,
    selected_day_recursive_predictions,
)


OUTPUT = ROOT / "results/dashboard/dashboard_validation_results.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: pd.Series, right: pd.Series, tolerance: float = 1e-7) -> bool:
    return bool(np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), rtol=tolerance, atol=tolerance, equal_nan=True))


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


@dataclass
class Report:
    rows: list[dict[str, str]] = field(default_factory=list)

    def add(self, check: str, passed: bool, detail: str) -> None:
        self.rows.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": detail})


def validate_sources(report: Report) -> dict[str, pd.DataFrame]:
    missing = [str(path.relative_to(ROOT)) for path in SOURCE_FILES.values() if not path.exists()]
    report.add("all_expected_dashboard_sources_exist", not missing, f"Checked {len(SOURCE_FILES)} saved source files." if not missing else f"Missing: {missing}")
    frames: dict[str, pd.DataFrame] = {}
    if missing:
        return frames
    try:
        frames = {name: load_table(name) for name in SOURCE_FILES}
        report.add("all_required_columns_present", True, "Every saved source passed the central schema checks.")
    except Exception as exc:
        report.add("all_required_columns_present", False, f"{type(exc).__name__}: {exc}")
    return frames


def validate_horizons_and_time(report: Report, frames: dict[str, pd.DataFrame]) -> None:
    recursive = frames["recursive_predictions"]
    selected = recursive.loc[recursive["model"].eq(SELECTED_DEMAND_MODEL)]
    groups = selected.groupby(["split", "forecast_date"])["horizon"].agg(list)
    recursive_ok = all(sorted(values) == list(range(1, 25)) and len(values) == 24 for values in groups)
    report.add("recursive_dates_have_exactly_horizons_1_to_24", recursive_ok, f"Checked {len(groups)} selected-model forecast dates.")

    planning = frames["planning_predictions"]
    plan_groups = planning.groupby(["split", "forecast_date"])["horizon"].agg(list)
    planning_ok = all(sorted(values) == list(range(1, 25)) and len(values) == 24 for values in plan_groups)
    report.add("planning_dates_have_exactly_horizons_1_to_24", planning_ok, f"Checked {len(plan_groups)} planning dates.")

    timestamp_columns = {
        "historical": ["period"],
        "recursive_predictions": ["forecast_origin", "forecast_date", "target_timestamp"],
        "planning_predictions": ["forecast_origin", "forecast_date", "target_timestamp"],
        "daily_planning_summary": ["forecast_date", "demand_peak_time_utc", "residual_demand_peak_time_utc"],
    }
    bad_timezone: list[str] = []
    for name, columns in timestamp_columns.items():
        for column in columns:
            if column in frames[name] and not isinstance(frames[name][column].dtype, pd.DatetimeTZDtype):
                bad_timezone.append(f"{name}.{column}")
    report.add("timestamps_are_explicitly_utc", not bad_timezone, "All parsed dashboard timestamps are timezone-aware UTC." if not bad_timezone else f"Not UTC-aware: {bad_timezone}")

    rec_sorted = all(group["target_timestamp"].is_monotonic_increasing for _, group in selected.groupby(["split", "forecast_date"]))
    plan_sorted = all(group["target_timestamp"].is_monotonic_increasing for _, group in planning.groupby(["split", "forecast_date"]))
    hist_sorted = frames["historical"]["period"].is_monotonic_increasing
    report.add("timestamps_are_chronologically_sorted", rec_sorted and plan_sorted and hist_sorted, "Historical, recursive-day, and planning-day timestamps are chronological.")

    left = selected.sort_values(["split", "forecast_date", "horizon"])[["split", "forecast_date", "horizon", "target_timestamp"]].reset_index(drop=True)
    right = planning.sort_values(["split", "forecast_date", "horizon"])[["split", "forecast_date", "horizon", "target_timestamp"]].reset_index(drop=True)
    report.add("forecast_and_planning_tables_align", left.equals(right), f"Compared {len(left):,} timestamp/horizon keys.")


def validate_headlines_and_formulas(report: Report, frames: dict[str, pd.DataFrame]) -> None:
    recursive_predictions = frames["recursive_predictions"]
    recursive_metrics = frames["recursive_metrics"]
    xgb = recursive_predictions.loc[(recursive_predictions["split"] == "test") & (recursive_predictions["model"] == SELECTED_DEMAND_MODEL)]
    saved_xgb = recursive_metrics.loc[(recursive_metrics["split"] == "test") & (recursive_metrics["model"] == SELECTED_DEMAND_MODEL)].iloc[0]
    calc_mae = (xgb["prediction_mwh"] - xgb["actual_demand_mwh"]).abs().mean()
    calc_rmse = np.sqrt(((xgb["prediction_mwh"] - xgb["actual_demand_mwh"]) ** 2).mean())
    recursive_ok = np.isclose(calc_mae, saved_xgb["mae_mwh"]) and np.isclose(calc_rmse, saved_xgb["rmse_mwh"])
    report.add("recursive_headline_metrics_match_predictions", recursive_ok, f"Reproduced test MAE={calc_mae:.6f} and RMSE={calc_rmse:.6f} MWh.")

    saved_naive = recursive_metrics.loc[(recursive_metrics["split"] == "test") & (recursive_metrics["model"] == DAILY_NAIVE_MODEL)].iloc[0]
    improvement = (saved_naive["mae_mwh"] - saved_xgb["mae_mwh"]) / saved_naive["mae_mwh"] * 100
    headlines = headline_metrics()
    report.add("daily_naive_improvement_matches_saved_metrics", np.isclose(improvement, headlines["improvement_vs_daily_naive_pct"]), f"Reproduced improvement={improvement:.6f}%.")

    planning = frames["planning_predictions"]
    saved_planning = frames["planning_metrics"]
    pairs = {
        "renewable_combined": ("actual_combined_renewable_mwh", "selected_combined_renewable_forecast_mwh"),
        "residual_demand": ("actual_residual_demand_mwh", "forecast_residual_demand_mwh"),
        "renewable_share": ("actual_renewable_share_pct", "forecast_renewable_share_pct"),
    }
    planning_ok = True
    details: list[str] = []
    for metric, (actual, predicted) in pairs.items():
        test = planning.loc[planning["split"].eq("test"), [actual, predicted]].dropna()
        value = (test[predicted] - test[actual]).abs().mean()
        saved = saved_planning.loc[(saved_planning["split"] == "test") & (saved_planning["metric"] == metric), "mae_mwh"].iloc[0]
        planning_ok &= np.isclose(value, saved)
        details.append(f"{metric}={value:.6f}")
    report.add("planning_headline_metrics_match_predictions", planning_ok, "Reproduced test MAE: " + "; ".join(details) + ".")

    formula_checks = {
        "combined renewable": (planning["selected_combined_renewable_forecast_mwh"], planning["selected_solar_forecast_mwh"] + planning["selected_wind_forecast_mwh"]),
        "forecast residual": (planning["forecast_residual_demand_mwh"], planning["forecast_demand_mwh"] - planning["selected_combined_renewable_forecast_mwh"]),
        "conservative residual": (planning["conservative_residual_demand_scenario_mwh"], planning["forecast_demand_mwh"] - planning["conservative_combined_renewable_scenario_mwh"]),
        "typical residual": (planning["typical_residual_demand_scenario_mwh"], planning["forecast_demand_mwh"] - planning["typical_combined_renewable_scenario_mwh"]),
        "favourable residual": (planning["favourable_residual_demand_scenario_mwh"], planning["forecast_demand_mwh"] - planning["favourable_combined_renewable_scenario_mwh"]),
    }
    failed = [name for name, (saved, calculated) in formula_checks.items() if not close(saved, calculated)]
    report.add("dashboard_calculations_match_saved_analytics", not failed, "All saved renewable/residual formulas reproduce." if not failed else f"Failed: {failed}")

    ordered = planning.sort_values(["forecast_origin", "horizon"]).copy()
    ordered["actual_ramp"] = ordered.groupby("forecast_origin")["actual_residual_demand_mwh"].diff()
    complete = ordered.dropna(subset=["actual_ramp", "forecast_hourly_residual_demand_ramp_mwh"])
    agreement = (np.sign(complete["actual_ramp"]) == np.sign(complete["forecast_hourly_residual_demand_ramp_mwh"])).mean() * 100
    report.add("ramp_direction_headline_reproduced", np.isclose(agreement, headlines["ramp_direction_agreement_pct"]), f"Reproduced direction agreement={agreement:.6f}% across {len(complete):,} comparisons.")


def validate_thresholds_and_missing(report: Report, frames: dict[str, pd.DataFrame]) -> None:
    planning = frames["planning_predictions"]
    values = frames["planning_thresholds"].set_index("threshold_name")["threshold_value"]
    expected = {
        "high_demand_alert": planning["forecast_demand_mwh"].gt(values["high_demand_mwh"]),
        "high_residual_demand_alert": planning["forecast_residual_demand_mwh"].gt(values["high_residual_demand_mwh"]),
        "high_upward_ramp_alert": planning["forecast_hourly_residual_demand_ramp_mwh"].gt(values["high_positive_residual_ramp_mwh"]),
        "low_renewable_share_alert": planning["forecast_renewable_share_pct"].lt(values["low_renewable_share_pct"]),
    }
    failures = [column for column, calculated in expected.items() if not truth(planning[column]).equals(calculated)]
    report.add("threshold_values_match_planning_indicators", not failures, "All four indicator columns match the saved training-derived thresholds." if not failures else f"Failed: {failures}")

    quality = data_quality_summary(frames["historical"])
    quality_ok = (
        quality["null_all_measurements_timestamps"] == 5
        and quality["missing_sun_wnd_timestamps"] == 24
        and quality["negative_solar_measurements"] == 9074
    )
    incomplete = ~truth(planning["actual_measurements_complete"])
    derived_actuals = ["actual_combined_renewable_mwh", "actual_residual_demand_mwh", "actual_renewable_share_pct"]
    preserved = incomplete.any() and planning.loc[incomplete, derived_actuals].isna().all().all()
    report.add("missing_values_are_preserved_not_filled", quality_ok and preserved, f"Historical exceptions={quality}; incomplete planning rows={int(incomplete.sum())} and derived actuals remain null.")


def validate_downloads_and_static_safety(report: Report, frames: dict[str, pd.DataFrame]) -> None:
    passed = True
    details: list[str] = []
    for split in ("validation", "test"):
        day = available_forecast_dates(split, frames["recursive_predictions"])[0]
        forecast = selected_day_recursive_predictions(split, day, predictions=frames["recursive_predictions"])
        planning = selected_day_planning(split, day, predictions=frames["planning_predictions"])
        for label, selected in (("forecast", forecast), ("planning", planning)):
            downloaded = pd.read_csv(BytesIO(dataframe_to_csv_bytes(selected)))
            passed &= len(downloaded) == 24 and downloaded.columns.tolist() == selected.columns.tolist()
            passed &= downloaded["target_timestamp"].str.endswith("Z").all()
            details.append(f"{split} {label}=24 rows")
    report.add("download_generation_preserves_selected_rows_columns_and_utc", passed, "; ".join(details) + ".")

    source_text = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in ["streamlit_app.py", "src/dashboard_data.py"])
    forbidden = {
        "model fitting": r"\.fit\s*\(",
        "HTTP URL": r"https?://",
        "requests client": r"\brequests\.",
        "urllib client": r"\burllib\.",
        "missing-value fill": r"\b(?:fillna|interpolate)\s*\(",
    }
    found = [label for label, pattern in forbidden.items() if re.search(pattern, source_text, flags=re.IGNORECASE)]
    report.add("dashboard_has_no_fitting_api_access_or_imputation", not found, "Static scan found no fitting, network, or missing-value fill calls." if not found else f"Found: {found}")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report = Report()
    existing = {name: path for name, path in SOURCE_FILES.items() if path.exists()}
    before = {name: sha256(path) for name, path in existing.items()}
    try:
        frames = validate_sources(report)
        if len(frames) == len(SOURCE_FILES):
            validate_horizons_and_time(report, frames)
            validate_headlines_and_formulas(report, frames)
            validate_thresholds_and_missing(report, frames)
            validate_downloads_and_static_safety(report, frames)
    except Exception as exc:
        report.add("validation_run_completed", False, f"{type(exc).__name__}: {exc}")

    after = {name: sha256(path) for name, path in existing.items()}
    changed = [name for name in before if before[name] != after[name]]
    report.add("upstream_sources_unchanged_during_validation", not changed, f"All {len(before)} dashboard source SHA-256 hashes are unchanged." if not changed else f"Changed: {changed}")

    result = pd.DataFrame(report.rows)
    result.to_csv(OUTPUT, index=False)
    failed = result.loc[result["status"].ne("PASS")]
    print(f"Dashboard validation: {len(result) - len(failed)}/{len(result)} checks passed.")
    if not failed.empty:
        print(failed.to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
