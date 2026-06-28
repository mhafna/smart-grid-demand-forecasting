"""Focused standard-library tests for the dashboard's read-only data layer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.dashboard_data import (
    ForecastDayError,
    MalformedSourceError,
    MissingSourceError,
    available_forecast_dates,
    dataframe_to_csv_bytes,
    historical_date_range,
    load_table,
    read_csv_download,
    selected_day_recursive_predictions,
)


def recursive_fixture(rows: int = 24, date: str = "2024-01-01") -> pd.DataFrame:
    timestamps = pd.date_range(date, periods=rows, freq="h", tz="UTC")
    actual = np.arange(rows, dtype=float) + 20_000
    predicted = actual + 100
    return pd.DataFrame(
        {
            "forecast_origin": pd.Timestamp(date, tz="UTC") - pd.Timedelta(hours=1),
            "forecast_date": pd.Timestamp(date, tz="UTC"),
            "target_timestamp": timestamps,
            "split": "validation",
            "horizon": np.arange(1, rows + 1),
            "model": "recursive_xgboost",
            "model_label": "Recursive XGBoost",
            "actual_demand_mwh": actual,
            "prediction_mwh": predicted,
            "error_mwh": predicted - actual,
            "absolute_error_mwh": np.abs(predicted - actual),
        }
    )


def historical_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": ["2024-01-01T00", "2024-01-01T01", "2024-01-01T02"],
            "demand_mwh": [20_000.0, np.nan, 21_000.0],
            "solar_generation_mwh": [0.0, np.nan, -2.0],
            "wind_generation_mwh": [2_000.0, np.nan, 2_100.0],
            "demand_data_complete": [True, False, True],
            "renewable_data_complete": [True, False, True],
            "solar_wind_generation_mwh": [2_000.0, np.nan, 2_098.0],
            "residual_demand_after_solar_wind_mwh": [18_000.0, np.nan, 18_902.0],
            "solar_wind_share_pct": [10.0, np.nan, 9.99],
        }
    )


class DashboardDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_successful_loading_parses_utc(self) -> None:
        path = self.root / "recursive.csv"
        recursive_fixture().to_csv(path, index=False)
        loaded = load_table("recursive_predictions", path)
        self.assertEqual(len(loaded), 24)
        self.assertIsInstance(loaded["target_timestamp"].dtype, pd.DatetimeTZDtype)
        self.assertEqual(str(loaded["target_timestamp"].dt.tz), "UTC")

    def test_missing_file_has_useful_error(self) -> None:
        with self.assertRaises(MissingSourceError):
            load_table("recursive_predictions", self.root / "absent.csv")

    def test_malformed_columns_are_rejected(self) -> None:
        path = self.root / "malformed.csv"
        recursive_fixture().drop(columns="prediction_mwh").to_csv(path, index=False)
        with self.assertRaises(MalformedSourceError):
            load_table("recursive_predictions", path)

    def test_available_dates_are_extracted_and_sorted(self) -> None:
        later = recursive_fixture(date="2024-01-02")
        earlier = recursive_fixture(date="2024-01-01")
        combined = pd.concat([later, earlier], ignore_index=True)
        dates = available_forecast_dates("validation", combined)
        self.assertEqual([value.strftime("%Y-%m-%d") for value in dates], ["2024-01-01", "2024-01-02"])

    def test_incomplete_horizon_day_is_rejected(self) -> None:
        with self.assertRaises(ForecastDayError):
            selected_day_recursive_predictions("validation", "2024-01-01", predictions=recursive_fixture(rows=23))

    def test_selected_date_filtering_returns_only_requested_day(self) -> None:
        combined = pd.concat([recursive_fixture(date="2024-01-01"), recursive_fixture(date="2024-01-02")], ignore_index=True)
        selected = selected_day_recursive_predictions("validation", "2024-01-02", predictions=combined)
        self.assertEqual(len(selected), 24)
        self.assertTrue(selected["forecast_date"].eq(pd.Timestamp("2024-01-02", tz="UTC")).all())

    def test_selected_day_is_chronologically_ordered(self) -> None:
        shuffled = recursive_fixture().iloc[::-1].reset_index(drop=True)
        selected = selected_day_recursive_predictions("validation", "2024-01-01", predictions=shuffled)
        self.assertEqual(selected["horizon"].tolist(), list(range(1, 25)))
        self.assertTrue(selected["target_timestamp"].is_monotonic_increasing)

    def test_historical_null_values_are_preserved(self) -> None:
        path = self.root / "historical.csv"
        historical_fixture().to_csv(path, index=False)
        selected = historical_date_range("2024-01-01", "2024-01-01", path)
        self.assertEqual(len(selected), 3)
        self.assertEqual(int(selected["demand_mwh"].isna().sum()), 1)
        self.assertEqual(int(selected["solar_generation_mwh"].isna().sum()), 1)

    def test_csv_download_keeps_row_count_columns_and_utc(self) -> None:
        selected = selected_day_recursive_predictions("validation", "2024-01-01", predictions=recursive_fixture())
        downloaded = read_csv_download(dataframe_to_csv_bytes(selected))
        self.assertEqual(len(downloaded), 24)
        self.assertEqual(downloaded.columns.tolist(), selected.columns.tolist())
        self.assertTrue(downloaded["target_timestamp"].str.endswith("Z").all())


if __name__ == "__main__":
    unittest.main()
