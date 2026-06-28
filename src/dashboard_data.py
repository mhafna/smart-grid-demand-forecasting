"""Read-only data access helpers for the Streamlit dashboard.

The dashboard deliberately consumes saved analytical outputs.  Nothing in this
module trains a model, calls an API, fills a missing value, or changes an
upstream file.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SELECTED_DEMAND_MODEL = "recursive_xgboost"
DAILY_NAIVE_MODEL = "daily_seasonal_naive"

SOURCE_FILES: dict[str, Path] = {
    "historical": ROOT / "data/processed/eia_ciso_hourly_2022_2024.csv",
    "recursive_predictions": ROOT / "results/recursive/tables/recursive_predictions.csv",
    "recursive_metrics": ROOT / "results/recursive/tables/metrics_overall.csv",
    "recursive_horizon_metrics": ROOT / "results/recursive/tables/metrics_by_horizon.csv",
    "planning_predictions": ROOT / "results/planning/tables/renewable_planning_predictions.csv",
    "daily_planning_summary": ROOT / "results/planning/tables/daily_planning_summary.csv",
    "planning_thresholds": ROOT / "results/planning/tables/planning_thresholds.csv",
    "planning_metrics": ROOT / "results/planning/tables/planning_metrics_overall.csv",
    "planning_horizon_metrics": ROOT / "results/planning/tables/planning_metrics_by_horizon.csv",
    "one_step_test_metrics": ROOT / "results/models/tables/model_metrics_common_test.csv",
    "one_step_validation_metrics": ROOT / "results/models/tables/model_metrics_common_validation.csv",
    "feature_importance": ROOT / "results/models/tables/xgboost_feature_importance_gain_all.csv",
}

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "historical": {
        "period", "demand_mwh", "solar_generation_mwh", "wind_generation_mwh",
        "demand_data_complete", "renewable_data_complete", "solar_wind_generation_mwh",
        "residual_demand_after_solar_wind_mwh", "solar_wind_share_pct",
    },
    "recursive_predictions": {
        "forecast_date", "target_timestamp", "split", "horizon", "model", "model_label",
        "actual_demand_mwh", "prediction_mwh", "error_mwh", "absolute_error_mwh",
    },
    "recursive_metrics": {"split", "model", "model_label", "count", "mae_mwh", "rmse_mwh", "bias_mwh"},
    "recursive_horizon_metrics": {"split", "model", "model_label", "horizon", "count", "mae_mwh", "rmse_mwh", "bias_mwh"},
    "planning_predictions": {
        "forecast_date", "target_timestamp", "split", "horizon", "forecast_demand_mwh",
        "selected_solar_forecast_mwh", "selected_wind_forecast_mwh",
        "selected_combined_renewable_forecast_mwh", "forecast_residual_demand_mwh",
        "actual_residual_demand_mwh", "forecast_renewable_share_pct",
        "conservative_residual_demand_scenario_mwh", "typical_residual_demand_scenario_mwh",
        "favourable_residual_demand_scenario_mwh", "forecast_hourly_residual_demand_ramp_mwh",
        "actual_measurements_complete", "high_demand_alert", "high_residual_demand_alert",
        "high_upward_ramp_alert", "low_renewable_share_alert",
    },
    "daily_planning_summary": {
        "split", "forecast_date", "demand_peak_mwh", "demand_peak_time_utc",
        "residual_demand_peak_mwh", "residual_demand_peak_time_utc",
        "lowest_forecast_renewable_share_pct", "lowest_forecast_renewable_share_time_utc",
        "maximum_upward_residual_ramp_mwh", "maximum_upward_residual_ramp_time_utc",
        "hours_above_high_demand_threshold", "hours_above_high_residual_threshold",
        "hours_with_high_upward_ramp_alert", "hours_with_low_renewable_share_alert",
    },
    "planning_thresholds": {"threshold_name", "threshold_value", "quantile", "fit_split", "source_definition"},
    "planning_metrics": {"split", "metric", "unit", "count", "mae_mwh", "rmse_mwh", "bias_mwh"},
    "planning_horizon_metrics": {"split", "horizon", "metric", "unit", "count", "mae_mwh", "rmse_mwh", "bias_mwh"},
    "one_step_test_metrics": {"split", "model", "algorithm", "feature_group", "observation_count", "mae_mwh", "rmse_mwh"},
    "one_step_validation_metrics": {"split", "model", "algorithm", "feature_group", "observation_count", "mae_mwh", "rmse_mwh"},
    "feature_importance": {"algorithm", "feature_group", "model", "feature", "gain", "normalized_gain"},
}

TIMESTAMP_COLUMNS: dict[str, tuple[str, ...]] = {
    "historical": ("period",),
    "recursive_predictions": ("forecast_origin", "forecast_date", "target_timestamp"),
    "planning_predictions": (
        "forecast_origin", "forecast_date", "target_timestamp", "daily_source_timestamp",
        "weekly_source_timestamp", "same_hour_window_start_exclusive", "same_hour_window_end_inclusive",
    ),
    "daily_planning_summary": (
        "forecast_date", "demand_peak_time_utc", "residual_demand_peak_time_utc",
        "lowest_forecast_renewable_share_time_utc", "maximum_upward_residual_ramp_time_utc",
    ),
}

SORT_COLUMNS: dict[str, list[str]] = {
    "historical": ["period"],
    "recursive_predictions": ["split", "model", "forecast_date", "horizon"],
    "planning_predictions": ["split", "forecast_date", "horizon"],
    "daily_planning_summary": ["split", "forecast_date"],
    "recursive_metrics": ["split", "model"],
    "recursive_horizon_metrics": ["split", "model", "horizon"],
    "planning_metrics": ["split", "metric"],
    "planning_horizon_metrics": ["split", "metric", "horizon"],
}


class DashboardDataError(RuntimeError):
    """Base error shown to dashboard users without an internal traceback."""


class MissingSourceError(DashboardDataError):
    """Raised when an expected saved output is absent."""


class MalformedSourceError(DashboardDataError):
    """Raised when a saved output has missing columns or invalid timestamps."""


class ForecastDayError(DashboardDataError):
    """Raised when a selected date is not a complete 24-hour forecast."""


def _missing_columns(frame: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return sorted(set(required).difference(frame.columns))


@lru_cache(maxsize=32)
def _read_table_cached(name: str, path_text: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns  # The value intentionally invalidates the cache after a file change.
    path = Path(path_text)
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise MalformedSourceError(f"Could not read {path}: {exc}") from exc

    missing = _missing_columns(frame, REQUIRED_COLUMNS[name])
    if missing:
        raise MalformedSourceError(f"{path.name} is missing required columns: {', '.join(missing)}")

    for column in TIMESTAMP_COLUMNS.get(name, ()):
        if column not in frame.columns:
            continue
        source_not_null = frame[column].notna()
        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
        invalid_count = int((source_not_null & parsed.isna()).sum())
        if invalid_count:
            raise MalformedSourceError(f"{path.name} has {invalid_count} invalid values in {column}.")
        frame[column] = parsed

    sort_columns = [column for column in SORT_COLUMNS.get(name, []) if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return frame


def load_table(name: str, path: Path | str | None = None) -> pd.DataFrame:
    """Load and validate one saved table, returning a defensive copy."""

    if name not in SOURCE_FILES:
        raise KeyError(f"Unknown dashboard source: {name}")
    source = Path(path) if path is not None else SOURCE_FILES[name]
    if not source.exists():
        raise MissingSourceError(f"Expected dashboard source is missing: {source}")
    return _read_table_cached(name, str(source.resolve()), source.stat().st_mtime_ns).copy()


def clear_caches() -> None:
    """Clear local read caches (primarily useful to isolated unit tests)."""

    _read_table_cached.cache_clear()
    _historical_range_cached.cache_clear()


def _normalise_split(split: str) -> str:
    value = str(split).strip().lower()
    if value not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")
    return value


def _normalise_forecast_date(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.normalize()


def available_forecast_dates(split: str, predictions: pd.DataFrame | None = None) -> list[pd.Timestamp]:
    """Return real saved forecast dates for the selected recursive model."""

    split = _normalise_split(split)
    frame = load_table("recursive_predictions") if predictions is None else predictions.copy()
    dates = frame.loc[(frame["split"] == split) & (frame["model"] == SELECTED_DEMAND_MODEL), "forecast_date"]
    dates = pd.to_datetime(dates, errors="coerce", utc=True).dropna().drop_duplicates().sort_values()
    return list(dates)


def _validate_24_hour_day(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    ordered = frame.sort_values("horizon", kind="stable").reset_index(drop=True)
    horizons = ordered["horizon"].tolist()
    if len(ordered) != 24 or horizons != list(range(1, 25)):
        raise ForecastDayError(f"{label} must contain exactly horizons 1–24; found {len(ordered)} rows with {horizons}.")
    if ordered["target_timestamp"].duplicated().any():
        raise ForecastDayError(f"{label} contains duplicate target timestamps.")
    if not ordered["target_timestamp"].is_monotonic_increasing:
        raise ForecastDayError(f"{label} target timestamps are not chronological.")
    return ordered


def selected_day_recursive_predictions(
    split: str,
    forecast_date: object,
    model: str = SELECTED_DEMAND_MODEL,
    predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one saved 24-hour recursive forecast for one model."""

    split = _normalise_split(split)
    day = _normalise_forecast_date(forecast_date)
    frame = load_table("recursive_predictions") if predictions is None else predictions.copy()
    frame["forecast_date"] = pd.to_datetime(frame["forecast_date"], errors="coerce", utc=True)
    frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"], errors="coerce", utc=True)
    chosen = frame.loc[(frame["split"] == split) & (frame["model"] == model) & (frame["forecast_date"] == day)].copy()
    if chosen.empty:
        raise ForecastDayError(f"No saved {model} forecast exists for {day:%Y-%m-%d} ({split}).")
    return _validate_24_hour_day(chosen, f"{day:%Y-%m-%d} {model} forecast")


def available_planning_dates(split: str, predictions: pd.DataFrame | None = None) -> list[pd.Timestamp]:
    split = _normalise_split(split)
    frame = load_table("planning_predictions") if predictions is None else predictions.copy()
    dates = frame.loc[frame["split"] == split, "forecast_date"]
    dates = pd.to_datetime(dates, errors="coerce", utc=True).dropna().drop_duplicates().sort_values()
    return list(dates)


def selected_day_planning(
    split: str,
    forecast_date: object,
    predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one saved renewable-planning day without filling incomplete actuals."""

    split = _normalise_split(split)
    day = _normalise_forecast_date(forecast_date)
    frame = load_table("planning_predictions") if predictions is None else predictions.copy()
    frame["forecast_date"] = pd.to_datetime(frame["forecast_date"], errors="coerce", utc=True)
    frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"], errors="coerce", utc=True)
    chosen = frame.loc[(frame["split"] == split) & (frame["forecast_date"] == day)].copy()
    if chosen.empty:
        raise ForecastDayError(f"No saved planning forecast exists for {day:%Y-%m-%d} ({split}).")
    return _validate_24_hour_day(chosen, f"{day:%Y-%m-%d} planning forecast")


def selected_daily_planning_summary(
    split: str,
    forecast_date: object,
    summaries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    split = _normalise_split(split)
    day = _normalise_forecast_date(forecast_date)
    frame = load_table("daily_planning_summary") if summaries is None else summaries.copy()
    frame["forecast_date"] = pd.to_datetime(frame["forecast_date"], errors="coerce", utc=True)
    selected = frame.loc[(frame["split"] == split) & (frame["forecast_date"] == day)].copy()
    if len(selected) != 1:
        raise ForecastDayError(f"Expected one daily planning summary for {day:%Y-%m-%d}; found {len(selected)}.")
    return selected.reset_index(drop=True)


def historical_bounds(path: Path | str | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    source = Path(path) if path is not None else SOURCE_FILES["historical"]
    if not source.exists():
        raise MissingSourceError(f"Expected dashboard source is missing: {source}")
    periods = pd.read_csv(source, usecols=["period"])["period"]
    parsed = pd.to_datetime(periods, errors="coerce", utc=True)
    if parsed.isna().any() or parsed.empty:
        raise MalformedSourceError(f"{source.name} contains invalid or empty period values.")
    return parsed.min(), parsed.max()


@lru_cache(maxsize=24)
def _historical_range_cached(path_text: str, modified_ns: int, start_text: str, end_text: str) -> pd.DataFrame:
    del modified_ns
    path = Path(path_text)
    start = pd.Timestamp(start_text)
    end = pd.Timestamp(end_text)
    chunks: list[pd.DataFrame] = []
    try:
        for chunk in pd.read_csv(path, chunksize=4096):
            missing = _missing_columns(chunk, REQUIRED_COLUMNS["historical"])
            if missing:
                raise MalformedSourceError(f"{path.name} is missing required columns: {', '.join(missing)}")
            chunk["period"] = pd.to_datetime(chunk["period"], errors="coerce", utc=True)
            if chunk["period"].isna().any():
                raise MalformedSourceError(f"{path.name} contains invalid period values.")
            selected = chunk.loc[chunk["period"].between(start, end, inclusive="both")]
            if not selected.empty:
                chunks.append(selected)
    except (OSError, pd.errors.ParserError) as exc:
        raise MalformedSourceError(f"Could not read {path}: {exc}") from exc
    if not chunks:
        return pd.DataFrame(columns=pd.read_csv(path, nrows=0).columns)
    return pd.concat(chunks, ignore_index=True).sort_values("period", kind="stable").reset_index(drop=True)


def historical_date_range(start: object, end: object, path: Path | str | None = None) -> pd.DataFrame:
    """Read only chunks that overlap a bounded inclusive UTC date range."""

    source = Path(path) if path is not None else SOURCE_FILES["historical"]
    if not source.exists():
        raise MissingSourceError(f"Expected dashboard source is missing: {source}")
    start_ts = _normalise_forecast_date(start)
    end_ts = _normalise_forecast_date(end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    if start_ts > end_ts:
        raise ValueError("Historical start date must not be after the end date.")
    return _historical_range_cached(str(source.resolve()), source.stat().st_mtime_ns, start_ts.isoformat(), end_ts.isoformat()).copy()


def model_metrics() -> dict[str, pd.DataFrame]:
    """Load the saved performance tables used by the performance page."""

    return {
        "one_step_test": load_table("one_step_test_metrics"),
        "one_step_validation": load_table("one_step_validation_metrics"),
        "recursive_overall": load_table("recursive_metrics"),
        "recursive_horizon": load_table("recursive_horizon_metrics"),
        "planning_overall": load_table("planning_metrics"),
        "planning_horizon": load_table("planning_horizon_metrics"),
        "feature_importance": load_table("feature_importance"),
    }


def planning_thresholds() -> pd.DataFrame:
    return load_table("planning_thresholds")


def data_quality_summary(historical: pd.DataFrame | None = None) -> dict[str, object]:
    """Calculate quality counts from preserved historical values and flags."""

    frame = load_table("historical") if historical is None else historical.copy()
    all_three_null = frame[["demand_mwh", "solar_generation_mwh", "wind_generation_mwh"]].isna().all(axis=1)
    absent_renewable_rows = (
        frame["demand_mwh"].notna()
        & frame["solar_generation_mwh"].isna()
        & frame["wind_generation_mwh"].isna()
    )
    return {
        "row_count": len(frame),
        "coverage_start": frame["period"].min(),
        "coverage_end": frame["period"].max(),
        "null_all_measurements_timestamps": int(all_three_null.sum()),
        "missing_sun_wnd_timestamps": int(absent_renewable_rows.sum()),
        "negative_solar_measurements": int(frame["solar_generation_mwh"].lt(0).sum()),
        "complete_demand_rows": int(frame["demand_mwh"].notna().sum()),
        "complete_renewable_rows": int(frame[["solar_generation_mwh", "wind_generation_mwh"]].notna().all(axis=1).sum()),
    }


def headline_metrics() -> dict[str, float | str]:
    """Read headline results from the saved tables and predictions."""

    recursive = load_table("recursive_metrics")
    selected = recursive.loc[(recursive["split"] == "test") & (recursive["model"] == SELECTED_DEMAND_MODEL)].iloc[0]
    naive = recursive.loc[(recursive["split"] == "test") & (recursive["model"] == DAILY_NAIVE_MODEL)].iloc[0]
    planning = load_table("planning_metrics")

    def planning_row(metric: str) -> pd.Series:
        return planning.loc[(planning["split"] == "test") & (planning["metric"] == metric)].iloc[0]

    renewable = planning_row("renewable_combined")
    residual = planning_row("residual_demand")
    share = planning_row("renewable_share")

    planning_predictions = load_table("planning_predictions").sort_values(["forecast_origin", "horizon"])
    planning_predictions["actual_residual_ramp_mwh"] = planning_predictions.groupby("forecast_origin")["actual_residual_demand_mwh"].diff()
    complete = planning_predictions.dropna(subset=["actual_residual_ramp_mwh", "forecast_hourly_residual_demand_ramp_mwh"])
    direction_agreement = (
        np.sign(complete["actual_residual_ramp_mwh"])
        == np.sign(complete["forecast_hourly_residual_demand_ramp_mwh"])
    ).mean() * 100

    return {
        "selected_model": str(selected["model_label"]),
        "recursive_test_mae_mwh": float(selected["mae_mwh"]),
        "recursive_test_rmse_mwh": float(selected["rmse_mwh"]),
        "improvement_vs_daily_naive_pct": float((naive["mae_mwh"] - selected["mae_mwh"]) / naive["mae_mwh"] * 100),
        "renewable_test_mae_mwh": float(renewable["mae_mwh"]),
        "residual_test_mae_mwh": float(residual["mae_mwh"]),
        "renewable_share_test_mae_pct_points": float(share["mae_mwh"]),
        "ramp_direction_agreement_pct": float(direction_agreement),
    }


def peak_demand_performance(predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarise daily peak forecasts directly from saved recursive predictions."""

    frame = load_table("recursive_predictions") if predictions is None else predictions.copy()
    grouped = frame.groupby(["split", "model", "model_label", "forecast_date"], as_index=False).agg(
        actual_peak_mwh=("actual_demand_mwh", "max"),
        forecast_peak_mwh=("prediction_mwh", "max"),
    )
    grouped["peak_error_mwh"] = grouped["forecast_peak_mwh"] - grouped["actual_peak_mwh"]
    grouped["absolute_peak_error_mwh"] = grouped["peak_error_mwh"].abs()
    return grouped.groupby(["split", "model", "model_label"], as_index=False).agg(
        days=("forecast_date", "count"),
        peak_mae_mwh=("absolute_peak_error_mwh", "mean"),
        peak_bias_mwh=("peak_error_mwh", "mean"),
    )


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize exactly the selected rows/columns with explicit UTC timestamps."""

    output = frame.copy()
    for column in output.columns:
        if isinstance(output[column].dtype, pd.DatetimeTZDtype):
            output[column] = output[column].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return output.to_csv(index=False).encode("utf-8")


def read_csv_download(payload: bytes) -> pd.DataFrame:
    """Small validation helper for round-tripping generated downloads."""

    return pd.read_csv(BytesIO(payload))
