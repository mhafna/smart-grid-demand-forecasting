"""Run leakage-safe one-hour-ahead baselines on fixed chronological splits.

The canonical feature master is read-only. This script writes only derived
tables, figures, metadata, and findings below results/baselines/.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "data" / "processed" / "eia_ciso_hourly_features.csv"
OUTPUT_DIR = ROOT / "results" / "baselines"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
FINDINGS_PATH = OUTPUT_DIR / "baseline_findings.md"

TARGET = "target_demand_mwh"
SPLIT_BOUNDS = {
    "train": (pd.Timestamp("2022-01-01T00"), pd.Timestamp("2023-12-31T23")),
    "validation": (pd.Timestamp("2024-01-01T00"), pd.Timestamp("2024-06-30T23")),
    "test": (pd.Timestamp("2024-07-01T00"), pd.Timestamp("2024-12-31T23")),
}
MODEL_SPECS = {
    "persistence_1h": {
        "prediction_source": "demand_lag_1h",
        "description": "Demand observed one hour earlier",
    },
    "daily_seasonal_naive_24h": {
        "prediction_source": "demand_lag_24h",
        "description": "Demand observed 24 hours earlier",
    },
    "weekly_seasonal_naive_168h": {
        "prediction_source": "demand_lag_168h",
        "description": "Demand observed 168 hours earlier",
    },
    "train_hour_of_week_mean": {
        "prediction_source": "training-only hour-of-week demand mean",
        "description": "Fixed UTC hour-of-week mean fitted on training targets only",
    },
}
MODEL_ORDER = list(MODEL_SPECS)
MODEL_LABELS = {
    "persistence_1h": "Persistence (1h)",
    "daily_seasonal_naive_24h": "Daily naive (24h)",
    "weekly_seasonal_naive_168h": "Weekly naive (168h)",
    "train_hour_of_week_mean": "Train hour-of-week mean",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest without changing the file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_table(table: pd.DataFrame, filename: str) -> None:
    """Save a derived table with stable timestamp and numeric formatting."""
    table.to_csv(
        TABLE_DIR / filename,
        index=False,
        date_format="%Y-%m-%dT%H",
        float_format="%.10f",
    )


def load_features() -> pd.DataFrame:
    """Load and minimally validate the feature master in chronological order."""
    required = {
        "period",
        TARGET,
        "target_available",
        "demand_lag_1h",
        "demand_lag_24h",
        "demand_lag_168h",
    }
    features = pd.read_csv(
        SOURCE_CSV,
        dtype={"target_available": "boolean"},
    )
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"Feature master is missing required columns: {missing}")
    features["period"] = pd.to_datetime(
        features["period"], format="%Y-%m-%dT%H", errors="raise"
    )
    if features["period"].duplicated().any():
        raise ValueError("Feature master contains duplicate timestamps.")
    if not features["period"].is_monotonic_increasing:
        raise ValueError("Feature master must already be in chronological order.")
    return features


def assign_splits(features: pd.DataFrame) -> pd.DataFrame:
    """Assign every row to exactly one fixed period without shuffling."""
    result = features.copy()
    result["split"] = pd.Series(pd.NA, index=result.index, dtype="string")
    for split, (start, end) in SPLIT_BOUNDS.items():
        mask = result["period"].between(start, end, inclusive="both")
        if result.loc[mask, "split"].notna().any():
            raise ValueError(f"Split {split} overlaps a previously assigned period.")
        result.loc[mask, "split"] = split
    if result["split"].isna().any():
        timestamps = result.loc[result["split"].isna(), "period"]
        raise ValueError(
            "One or more timestamps fall outside the fixed split policy: "
            f"{timestamps.min()} through {timestamps.max()}"
        )
    result["hour_of_week_utc"] = (
        result["period"].dt.dayofweek * 24 + result["period"].dt.hour
    )
    return result


def fit_hour_of_week(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, float]:
    """Fit 168 category means using available training targets only."""
    start, end = SPLIT_BOUNDS["train"]
    fit_mask = (
        features["split"].eq("train")
        & features["period"].between(start, end, inclusive="both")
        & features["target_available"].fillna(False)
        & features[TARGET].notna()
    )
    fit_rows = features.loc[fit_mask, ["period", "hour_of_week_utc", TARGET]].copy()
    global_mean = float(fit_rows[TARGET].mean())
    grouped = fit_rows.groupby("hour_of_week_utc", observed=True)[TARGET]
    lookup = grouped.agg(
        training_observation_count="count",
        training_target_sum_mwh="sum",
        training_mean_demand_mwh="mean",
    ).reset_index()
    lookup["fit_split"] = "train"
    lookup["fit_start_utc"] = start
    lookup["fit_end_utc"] = end
    lookup["latest_contributing_timestamp_utc"] = grouped.apply(
        lambda values: fit_rows.loc[values.index, "period"].max(), include_groups=False
    ).to_numpy()
    lookup["training_global_mean_demand_mwh"] = global_mean
    means = lookup.set_index("hour_of_week_utc")["training_mean_demand_mwh"]
    return lookup, means, global_mean


def prediction_columns(
    features: pd.DataFrame, hour_means: pd.Series, global_mean: float
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Create fixed baseline predictions; no current renewable value is used."""
    predictions = {
        "persistence_1h": features["demand_lag_1h"],
        "daily_seasonal_naive_24h": features["demand_lag_24h"],
        "weekly_seasonal_naive_168h": features["demand_lag_168h"],
    }
    mapped = features["hour_of_week_utc"].map(hour_means)
    fallback = mapped.isna()
    predictions["train_hour_of_week_mean"] = mapped.fillna(global_mean)
    return fallback, predictions


def build_split_summary(
    features: pd.DataFrame,
    predictions: dict[str, pd.Series],
    hour_fallback: pd.Series,
) -> pd.DataFrame:
    """Report the complete timeline, targets, and model-specific eligibility."""
    rows: list[dict[str, object]] = []
    target_ok = features["target_available"].fillna(False) & features[TARGET].notna()
    for split in SPLIT_BOUNDS:
        split_mask = features["split"].eq(split)
        row: dict[str, object] = {
            "split": split,
            "start_utc": features.loc[split_mask, "period"].min(),
            "end_utc": features.loc[split_mask, "period"].max(),
            "total_rows": int(split_mask.sum()),
            "target_available_rows": int((split_mask & target_ok).sum()),
        }
        eligible_masks = []
        for model in MODEL_ORDER:
            eligible = split_mask & target_ok & predictions[model].notna()
            row[f"{model}_eligible_rows"] = int(eligible.sum())
            eligible_masks.append(predictions[model].notna())
        row["all_baselines_eligible_rows"] = int(
            (split_mask & target_ok & pd.concat(eligible_masks, axis=1).all(axis=1)).sum()
        )
        row["hour_of_week_global_mean_fallback_rows"] = int(
            (split_mask & hour_fallback).sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_prediction_table(
    features: pd.DataFrame,
    predictions: dict[str, pd.Series],
    hour_fallback: pd.Series,
) -> pd.DataFrame:
    """Build one long-form row for every evaluation timestamp and model."""
    tables: list[pd.DataFrame] = []
    eval_mask = features["split"].isin(["validation", "test"])
    base = features.loc[eval_mask, ["period", "split", TARGET, "target_available"]].copy()
    base = base.rename(columns={TARGET: "actual_demand_mwh"})
    for model in MODEL_ORDER:
        table = base.copy()
        table["model_name"] = model
        table["prediction_mwh"] = predictions[model].loc[eval_mask].to_numpy()
        table["hour_of_week_global_mean_fallback"] = (
            hour_fallback.loc[eval_mask].to_numpy()
            if model == "train_hour_of_week_mean"
            else False
        )
        valid = (
            table["target_available"].fillna(False)
            & table["actual_demand_mwh"].notna()
            & table["prediction_mwh"].notna()
        )
        table["error_mwh"] = np.where(
            valid, table["prediction_mwh"] - table["actual_demand_mwh"], np.nan
        )
        table["absolute_error_mwh"] = table["error_mwh"].abs()
        pct_valid = valid & table["actual_demand_mwh"].ne(0)
        table["percentage_error_pct"] = np.where(
            pct_valid,
            100.0 * table["error_mwh"] / table["actual_demand_mwh"],
            np.nan,
        )
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    result["model_name"] = pd.Categorical(
        result["model_name"], categories=MODEL_ORDER, ordered=True
    )
    return result.sort_values(["split", "period", "model_name"]).reset_index(drop=True)


def calculate_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    """Calculate all metrics from rows with both an available target and prediction."""
    valid = (
        group["target_available"].fillna(False)
        & group["actual_demand_mwh"].notna()
        & group["prediction_mwh"].notna()
    )
    used = group.loc[valid]
    actual = used["actual_demand_mwh"].astype(float)
    prediction = used["prediction_mwh"].astype(float)
    error = prediction - actual
    absolute_error = error.abs()
    squared_error = error.pow(2)
    nonzero = actual.ne(0)
    smape_denominator = actual.abs() + prediction.abs()
    smape_valid = smape_denominator.ne(0)
    ss_total = float(((actual - actual.mean()) ** 2).sum())
    ss_residual = float(squared_error.sum())
    return {
        "observation_count": int(len(used)),
        "mae_mwh": float(absolute_error.mean()),
        "rmse_mwh": float(np.sqrt(squared_error.mean())),
        "mape_pct": float((absolute_error[nonzero] / actual[nonzero].abs()).mean() * 100),
        "smape_pct": float(
            (2 * absolute_error[smape_valid] / smape_denominator[smape_valid]).mean()
            * 100
        ),
        "mean_error_bias_mwh": float(error.mean()),
        "r_squared": float(1 - ss_residual / ss_total) if ss_total > 0 else np.nan,
    }


def build_metric_tables(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate split/model metrics and rank primarily by MAE."""
    rows = []
    for split in ["validation", "test"]:
        for model in MODEL_ORDER:
            group = predictions.loc[
                predictions["split"].eq(split)
                & predictions["model_name"].eq(model)
            ]
            rows.append({"split": split, "model_name": model, **calculate_metrics(group)})
    metrics = pd.DataFrame(rows)
    metrics["mae_rank"] = metrics.groupby("split")["mae_mwh"].rank(
        method="min", ascending=True
    ).astype(int)
    metrics = metrics.sort_values(["split", "mae_rank", "model_name"]).reset_index(drop=True)
    save_table(metrics.loc[metrics["split"].eq("validation")], "baseline_metrics_validation.csv")
    save_table(metrics.loc[metrics["split"].eq("test")], "baseline_metrics_test.csv")
    save_table(metrics, "baseline_metrics_all.csv")
    return metrics


def grouped_error_table(
    predictions: pd.DataFrame, group_column: str, output_name: str
) -> pd.DataFrame:
    """Create concise metrics for each split, model, and calendar group."""
    rows = []
    for (split, model, value), group in predictions.groupby(
        ["split", "model_name", group_column], observed=True, sort=True
    ):
        rows.append(
            {
                "split": split,
                "model_name": str(model),
                group_column: value,
                **calculate_metrics(group),
            }
        )
    result = pd.DataFrame(rows)
    save_table(result, output_name)
    return result


def build_error_analysis(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize distributions, calendar errors, and split-local demand tails."""
    analysis = predictions.copy()
    analysis["hour_utc"] = analysis["period"].dt.hour
    analysis["calendar_month"] = analysis["period"].dt.month
    analysis["day_of_week_utc"] = analysis["period"].dt.day_name()

    distribution_rows = []
    for (split, model), group in analysis.groupby(
        ["split", "model_name"], observed=True, sort=False
    ):
        valid_abs = group["absolute_error_mwh"].dropna()
        distribution_rows.append(
            {
                "split": split,
                "model_name": str(model),
                "observation_count": int(valid_abs.count()),
                "mean_absolute_error_mwh": float(valid_abs.mean()),
                "std_absolute_error_mwh": float(valid_abs.std(ddof=1)),
                "minimum_absolute_error_mwh": float(valid_abs.min()),
                "p25_absolute_error_mwh": float(valid_abs.quantile(0.25)),
                "median_absolute_error_mwh": float(valid_abs.median()),
                "p75_absolute_error_mwh": float(valid_abs.quantile(0.75)),
                "p90_absolute_error_mwh": float(valid_abs.quantile(0.90)),
                "p95_absolute_error_mwh": float(valid_abs.quantile(0.95)),
                "maximum_absolute_error_mwh": float(valid_abs.max()),
            }
        )
    distributions = pd.DataFrame(distribution_rows)
    save_table(distributions, "absolute_error_distribution.csv")

    by_hour = grouped_error_table(analysis, "hour_utc", "error_by_utc_hour.csv")
    by_month = grouped_error_table(analysis, "calendar_month", "error_by_calendar_month.csv")
    by_day = grouped_error_table(analysis, "day_of_week_utc", "error_by_day_of_week_utc.csv")

    tail_rows = []
    threshold_rows = []
    for split in ["validation", "test"]:
        split_actual = (
            analysis.loc[
                analysis["split"].eq(split)
                & analysis["target_available"].fillna(False),
                ["period", "actual_demand_mwh"],
            ]
            .drop_duplicates("period")
            .dropna()
        )
        low = float(split_actual["actual_demand_mwh"].quantile(0.10))
        high = float(split_actual["actual_demand_mwh"].quantile(0.90))
        threshold_rows.append(
            {
                "split": split,
                "bottom_10_percent_threshold_mwh": low,
                "top_10_percent_threshold_mwh": high,
                "target_observation_count": len(split_actual),
            }
        )
        split_rows = analysis.loc[analysis["split"].eq(split)].copy()
        conditions = {
            "bottom_10_percent": split_rows["actual_demand_mwh"].le(low),
            "middle_80_percent": split_rows["actual_demand_mwh"].gt(low)
            & split_rows["actual_demand_mwh"].lt(high),
            "top_10_percent": split_rows["actual_demand_mwh"].ge(high),
        }
        for model in MODEL_ORDER:
            model_rows = split_rows.loc[split_rows["model_name"].eq(model)]
            for demand_group, condition in conditions.items():
                group = model_rows.loc[condition.loc[model_rows.index]]
                tail_rows.append(
                    {
                        "split": split,
                        "model_name": model,
                        "demand_group": demand_group,
                        "bottom_threshold_mwh": low,
                        "top_threshold_mwh": high,
                        **calculate_metrics(group),
                    }
                )
    thresholds = pd.DataFrame(threshold_rows)
    tails = pd.DataFrame(tail_rows)
    save_table(thresholds, "demand_percentile_thresholds.csv")
    save_table(tails, "demand_percentile_performance.csv")
    return {
        "distributions": distributions,
        "by_hour": by_hour,
        "by_month": by_month,
        "by_day": by_day,
        "thresholds": thresholds,
        "tails": tails,
    }


def representative_week(split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Choose the central Monday-to-Sunday UTC week deterministically."""
    start, end = SPLIT_BOUNDS[split]
    midpoint = start + (end - start) / 2
    week_start = midpoint.normalize() - pd.Timedelta(days=midpoint.dayofweek)
    if week_start < start:
        week_start += pd.Timedelta(days=7)
    week_end = week_start + pd.Timedelta(hours=167)
    if week_end > end:
        week_start -= pd.Timedelta(days=7)
        week_end = week_start + pd.Timedelta(hours=167)
    return week_start, week_end


def style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(metrics: pd.DataFrame, split: str, filename: str) -> None:
    subset = metrics.loc[metrics["split"].eq(split)].set_index("model_name").loc[MODEL_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, column, title, ylabel in [
        (axes[0], "mae_mwh", "MAE", "MWh"),
        (axes[1], "rmse_mwh", "RMSE", "MWh"),
        (axes[2], "mape_pct", "MAPE", "Percent"),
    ]:
        ax.bar([MODEL_LABELS[m] for m in MODEL_ORDER], subset[column], color="tab:blue")
        style_axes(ax, f"{split.title()} {title}", "Baseline", ylabel)
        ax.tick_params(axis="x", rotation=28)
    save_figure(fig, filename)


def plot_representative_week(
    predictions: pd.DataFrame, split: str, filename: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = representative_week(split)
    week = predictions.loc[
        predictions["split"].eq(split)
        & predictions["period"].between(start, end, inclusive="both")
    ]
    actual = week.drop_duplicates("period").sort_values("period")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(
        actual["period"], actual["actual_demand_mwh"], color="black", linewidth=2.1, label="Actual"
    )
    for model in MODEL_ORDER:
        model_rows = week.loc[week["model_name"].eq(model)].sort_values("period")
        ax.plot(
            model_rows["period"], model_rows["prediction_mwh"], linewidth=1.1,
            alpha=0.85, label=MODEL_LABELS[model]
        )
    style_axes(
        ax,
        f"Actual vs Baseline Predictions: {split.title()} Representative Week",
        "Timestamp (UTC)",
        "Demand (MWh)",
    )
    ax.legend(ncol=3, fontsize=9)
    save_figure(fig, filename)
    return start, end


def build_figures(
    predictions: pd.DataFrame, metrics: pd.DataFrame, analysis: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Create the eight required, readable PNG figures."""
    plot_metric_comparison(metrics, "validation", "01_validation_metrics_comparison.png")
    plot_metric_comparison(metrics, "test", "02_test_metrics_comparison.png")
    validation_start, validation_end = plot_representative_week(
        predictions, "validation", "03_validation_representative_week.png"
    )
    test_start, test_end = plot_representative_week(
        predictions, "test", "04_test_representative_week.png"
    )

    by_hour = analysis["by_hour"].loc[lambda d: d["split"].eq("test")]
    fig, ax = plt.subplots(figsize=(11, 6))
    for model in MODEL_ORDER:
        group = by_hour.loc[by_hour["model_name"].eq(model)]
        ax.plot(group["hour_utc"], group["mae_mwh"], marker="o", label=MODEL_LABELS[model])
    ax.set_xticks(range(24))
    style_axes(ax, "Test Absolute Error by UTC Hour", "Hour (UTC)", "MAE (MWh)")
    ax.legend(fontsize=9)
    save_figure(fig, "05_test_absolute_error_by_utc_hour.png")

    by_month = analysis["by_month"].loc[lambda d: d["split"].eq("test")]
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODEL_ORDER:
        group = by_month.loc[by_month["model_name"].eq(model)]
        ax.plot(
            group["calendar_month"], group["mae_mwh"], marker="o", label=MODEL_LABELS[model]
        )
    ax.set_xticks(sorted(by_month["calendar_month"].unique()))
    style_axes(ax, "Test MAE by Calendar Month", "Calendar month (UTC)", "MAE (MWh)")
    ax.legend(fontsize=9)
    save_figure(fig, "06_test_mae_by_calendar_month.png")

    fig, ax = plt.subplots(figsize=(11, 6))
    test_errors = [
        predictions.loc[
            predictions["split"].eq("test") & predictions["model_name"].eq(model),
            "error_mwh",
        ].dropna()
        for model in MODEL_ORDER
    ]
    ax.boxplot(test_errors, tick_labels=[MODEL_LABELS[m] for m in MODEL_ORDER], showfliers=False)
    ax.axhline(0, color="black", linewidth=0.8)
    style_axes(ax, "Test Error Distribution (Outliers Hidden for Readability)", "Baseline", "Error: prediction - actual (MWh)")
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, "07_test_error_distribution.png")

    peak = analysis["tails"].loc[
        analysis["tails"]["split"].eq("test")
        & analysis["tails"]["demand_group"].eq("top_10_percent")
    ].set_index("model_name").loc[MODEL_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar([MODEL_LABELS[m] for m in MODEL_ORDER], peak["mae_mwh"], color="tab:red")
    style_axes(axes[0], "Top-10% Test Demand: MAE", "Baseline", "MAE (MWh)")
    axes[1].bar([MODEL_LABELS[m] for m in MODEL_ORDER], peak["rmse_mwh"], color="tab:orange")
    style_axes(axes[1], "Top-10% Test Demand: RMSE", "Baseline", "RMSE (MWh)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    save_figure(fig, "08_peak_demand_test_performance.png")

    weeks = pd.DataFrame(
        [
            {
                "split": "validation",
                "selection_rule": "central Monday-to-Sunday UTC week",
                "week_start_utc": validation_start,
                "week_end_utc": validation_end,
            },
            {
                "split": "test",
                "selection_rule": "central Monday-to-Sunday UTC week",
                "week_start_utc": test_start,
                "week_end_utc": test_end,
            },
        ]
    )
    save_table(weeks, "representative_weeks.csv")
    return weeks


def write_findings(
    split_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    analysis: dict[str, pd.DataFrame],
    weeks: pd.DataFrame,
    feature_hash: str,
) -> None:
    """Write calculated, restrained findings from the saved result tables."""
    validation = metrics.loc[metrics["split"].eq("validation")].sort_values("mae_rank")
    test = metrics.loc[metrics["split"].eq("test")].sort_values("mae_rank")
    best = test.iloc[0]
    persistence = test.loc[test["model_name"].eq("persistence_1h")].iloc[0]
    improvement_mwh = float(persistence["mae_mwh"] - best["mae_mwh"])
    improvement_pct = 100 * improvement_mwh / float(persistence["mae_mwh"])
    if best["model_name"] == "persistence_1h":
        improvement_text = (
            "No evaluated baseline improves on persistence. Persistence is itself "
            "the best baseline, so improvement over persistence is 0.00 MWh (0.00%)."
        )
    else:
        improvement_text = (
            f"Relative to persistence, its MAE is lower by {improvement_mwh:,.2f} MWh "
            f"({improvement_pct:.2f}%)."
        )
    peaks = analysis["tails"].loc[
        analysis["tails"]["split"].eq("test")
        & analysis["tails"]["demand_group"].eq("top_10_percent")
    ].sort_values("mae_mwh")
    best_peak = peaks.iloc[0]
    hour_errors = analysis["by_hour"].loc[
        analysis["by_hour"]["split"].eq("test")
        & analysis["by_hour"]["model_name"].eq(best["model_name"])
    ]
    worst_hour = hour_errors.sort_values("mae_mwh", ascending=False).iloc[0]
    month_errors = analysis["by_month"].loc[
        analysis["by_month"]["split"].eq("test")
        & analysis["by_month"]["model_name"].eq(best["model_name"])
    ]
    worst_month = month_errors.sort_values("mae_mwh", ascending=False).iloc[0]
    rankings_consistent = validation["model_name"].tolist() == test["model_name"].tolist()
    week_text = "; ".join(
        f"{row.split}: {row.week_start_utc:%Y-%m-%dT%H} to {row.week_end_utc:%Y-%m-%dT%H} UTC"
        for row in weeks.itertuples()
    )

    def metric_lines(frame: pd.DataFrame) -> str:
        return "\n".join(
            f"{int(row.mae_rank)}. `{row.model_name}` — MAE {row.mae_mwh:,.2f} MWh, "
            f"RMSE {row.rmse_mwh:,.2f} MWh, MAPE {row.mape_pct:.2f}%, "
            f"bias {row.mean_error_bias_mwh:,.2f} MWh, R² {row.r_squared:.4f} "
            f"(n={int(row.observation_count):,})"
            for row in frame.itertuples()
        )

    split_lines = "\n".join(
        f"- **{row.split.title()}:** {int(row.total_rows):,} total rows; "
        f"{int(row.target_available_rows):,} available targets; persistence/daily/weekly/"
        f"hour-of-week eligible = {int(row.persistence_1h_eligible_rows):,}/"
        f"{int(row.daily_seasonal_naive_24h_eligible_rows):,}/"
        f"{int(row.weekly_seasonal_naive_168h_eligible_rows):,}/"
        f"{int(row.train_hour_of_week_mean_eligible_rows):,}."
        for row in split_summary.itertuples()
    )
    peak_lines = "\n".join(
        f"- `{row.model_name}`: MAE {row.mae_mwh:,.2f} MWh; RMSE {row.rmse_mwh:,.2f} MWh; "
        f"MAPE {row.mape_pct:.2f}% (n={int(row.observation_count):,})."
        for row in peaks.itertuples()
    )
    content = f"""# Baseline Findings

## Evaluation design

This is an honest **one-hour-ahead** comparison on fixed, chronological UTC splits. Lag baselines may use the true earlier demand because that observation is assumed to be available before timestamp `t`. This is not recursive 24-hour forecasting: at later recursive horizons, unavailable future lags must eventually be replaced by earlier model predictions.

The hour-of-week lookup was fitted once from available training targets only and then frozen. No same-hour renewable measurement was used. Errors are `prediction - actual`, so positive bias means over-prediction. MAPE excludes a row only when its actual value is zero; no targets or predictions are filled.

## Split counts

{split_lines}

## Validation ranking (primary metric: MAE)

{metric_lines(validation)}

## Test ranking (primary metric: MAE)

{metric_lines(test)}

## Main result

The best test baseline is **`{best['model_name']}`** with MAE {best['mae_mwh']:,.2f} MWh. {improvement_text} RMSE and MAPE should be read alongside MAE because RMSE penalizes large misses more strongly, while MAPE reports error relative to observed demand.

For the best test baseline, the largest hourly MAE occurs at UTC hour {int(worst_hour['hour_utc']):02d}:00 ({worst_hour['mae_mwh']:,.2f} MWh), and the largest monthly MAE occurs in month {int(worst_month['calendar_month'])} ({worst_month['mae_mwh']:,.2f} MWh). These are descriptive test-period patterns, not claims about future years.

## Peak-demand performance

The test top-10% threshold is {float(peaks['top_threshold_mwh'].iloc[0]):,.2f} MWh, calculated from test targets only. Peak results are:

{peak_lines}

The lowest peak-period MAE belongs to **`{best_peak['model_name']}`** at {best_peak['mae_mwh']:,.2f} MWh.

## Stability and limitations

Validation and test MAE rankings are **{'consistent' if rankings_consistent else 'not identical'}**. The representative weeks were selected deterministically as the central Monday-to-Sunday UTC week in each split: {week_text}.

This benchmark tests one-step updates and therefore benefits from newly observed demand at every hour. It does not measure error accumulation across a recursive 24-hour forecast, and it does not establish causal effects. Future models should first beat the test MAE of **{best['mae_mwh']:,.2f} MWh** from `{best['model_name']}` while also checking RMSE, MAPE, bias, peak demand, and validation-to-test stability.

Feature-master SHA-256 recorded before and after this run: `{feature_hash}`.
"""
    FINDINGS_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    """Run the complete reproducible baseline workflow."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    hash_before = sha256(SOURCE_CSV)
    features = assign_splits(load_features())
    lookup, hour_means, global_mean = fit_hour_of_week(features)
    hour_fallback, prediction_map = prediction_columns(features, hour_means, global_mean)

    split_summary = build_split_summary(features, prediction_map, hour_fallback)
    predictions = build_prediction_table(features, prediction_map, hour_fallback)
    metrics = build_metric_tables(predictions)
    analysis = build_error_analysis(predictions)

    save_table(split_summary, "split_summary.csv")
    save_table(lookup, "hour_of_week_training_lookup.csv")
    save_table(
        pd.DataFrame(
            [
                {
                    "model_name": model,
                    "prediction_source": spec["prediction_source"],
                    "description": spec["description"],
                    "fit_split": "train" if model == "train_hour_of_week_mean" else "not_fitted",
                    "uses_contemporaneous_renewable_measurement": False,
                }
                for model, spec in MODEL_SPECS.items()
            ]
        ),
        "baseline_model_specification.csv",
    )
    save_table(
        predictions.loc[predictions["split"].eq("validation")],
        "baseline_predictions_validation.csv",
    )
    save_table(
        predictions.loc[predictions["split"].eq("test")],
        "baseline_predictions_test.csv",
    )
    save_table(predictions, "baseline_predictions_all.csv")
    weeks = build_figures(predictions, metrics, analysis)

    hash_after = sha256(SOURCE_CSV)
    metadata = pd.DataFrame(
        [
            {
                "source_feature_path": SOURCE_CSV.relative_to(ROOT).as_posix(),
                "source_sha256_before": hash_before,
                "source_sha256_after": hash_after,
                "source_hash_unchanged": hash_before == hash_after,
                "forecast_horizon_hours": 1,
                "evaluation_mode": "one_step_ahead_with_observed_lags",
                "hour_of_week_fit_split": "train",
                "hour_of_week_fit_start_utc": SPLIT_BOUNDS["train"][0],
                "hour_of_week_fit_end_utc": SPLIT_BOUNDS["train"][1],
                "hour_of_week_training_observations": int(
                    lookup["training_observation_count"].sum()
                ),
                "hour_of_week_training_global_mean_mwh": global_mean,
                "validation_or_test_targets_in_fitted_statistics": 0,
                "contemporaneous_renewable_predictors_used": 0,
                "error_definition": "prediction minus actual",
            }
        ]
    )
    save_table(metadata, "baseline_run_metadata.csv")
    if hash_before != hash_after:
        raise RuntimeError("Feature-master SHA-256 changed during the baseline run.")
    write_findings(split_summary, metrics, analysis, weeks, hash_after)

    print(f"Baseline evaluation complete: {OUTPUT_DIR}")
    print(f"Feature-master SHA-256 unchanged: {hash_after}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
