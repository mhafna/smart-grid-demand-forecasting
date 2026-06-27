"""Run reproducible exploratory data analysis for the CISO hourly dataset.

This script reads the validated master CSV without changing it. All derived
tables, figures, and written findings are saved below results/eda/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "data" / "processed" / "eia_ciso_hourly_2022_2024.csv"
OUTPUT_DIR = ROOT / "results" / "eda"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

DEMAND_COLUMN = "demand_mwh"
RENEWABLE_COLUMNS = ["solar_generation_mwh", "wind_generation_mwh"]
DERIVED_RENEWABLE_COLUMNS = [
    "solar_wind_generation_mwh",
    "residual_demand_after_solar_wind_mwh",
    "solar_wind_share_pct",
]
METRIC_COLUMNS = [DEMAND_COLUMN, *RENEWABLE_COLUMNS, *DERIVED_RENEWABLE_COLUMNS]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_ORDER = list(range(1, 13))
KNOWN_NULL_VALUE_TIMESTAMPS = pd.to_datetime(
    ["2022-01-05T10", "2022-05-17T18", "2022-06-13T18", "2023-10-31T21", "2023-11-14T20"]
)
KNOWN_MISSING_RENEWABLE_BLOCK = pd.date_range("2024-11-02T08", "2024-11-03T07", freq="h")


def load_and_validate() -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the master CSV and calculate validation checks without editing it."""
    required = {
        "period",
        *METRIC_COLUMNS,
        "demand_data_complete",
        "renewable_data_complete",
    }
    df = pd.read_csv(SOURCE_CSV, parse_dates=["period"])
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Source CSV is missing required columns: {missing_columns}")
    if df["period"].isna().any():
        raise ValueError("One or more period values could not be parsed as datetimes.")

    df = df.sort_values("period").reset_index(drop=True)
    expected_range = pd.date_range(df["period"].min(), df["period"].max(), freq="h")
    missing_hours = expected_range.difference(pd.DatetimeIndex(df["period"]))
    unexpected_hours = pd.DatetimeIndex(df["period"]).difference(expected_range)
    duplicate_count = int(df["period"].duplicated().sum())

    validation = {
        "total_rows": len(df),
        "earliest_timestamp": df["period"].min(),
        "latest_timestamp": df["period"].max(),
        "duplicate_timestamps": duplicate_count,
        "expected_hourly_rows": len(expected_range),
        "missing_hours": len(missing_hours),
        "unexpected_hours": len(unexpected_hours),
        "hourly_continuity": duplicate_count == 0 and len(missing_hours) == 0 and len(df) == len(expected_range),
        "demand_complete_rows": int(df["demand_data_complete"].sum()),
        "demand_incomplete_rows": int((~df["demand_data_complete"]).sum()),
        "renewable_complete_rows": int(df["renewable_data_complete"].sum()),
        "renewable_incomplete_rows": int((~df["renewable_data_complete"]).sum()),
    }
    return df, validation


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return an analytical copy with UTC-based calendar columns."""
    analysis = df.copy()
    analysis["year"] = analysis["period"].dt.year
    analysis["month"] = analysis["period"].dt.month
    analysis["month_name"] = analysis["period"].dt.month_name()
    analysis["day_of_week"] = analysis["period"].dt.day_name()
    analysis["hour"] = analysis["period"].dt.hour
    analysis["date"] = analysis["period"].dt.floor("D")
    return analysis


def save_table(table: pd.DataFrame, filename: str, index: bool = False) -> None:
    """Save a consistently formatted CSV table."""
    table.to_csv(TABLE_DIR / filename, index=index, date_format="%Y-%m-%dT%H:%M:%S")


def build_tables(df: pd.DataFrame, validation: dict[str, object]) -> dict[str, pd.DataFrame]:
    """Calculate and save validation, summary, time, renewable, and peak tables."""
    demand = df.loc[df["demand_data_complete"] & df[DEMAND_COLUMN].notna()].copy()
    renewable = df.loc[
        df["renewable_data_complete"] & df[RENEWABLE_COLUMNS].notna().all(axis=1)
    ].copy()
    combined = df.loc[
        df["demand_data_complete"]
        & df["renewable_data_complete"]
        & df[[DEMAND_COLUMN, *RENEWABLE_COLUMNS, *DERIVED_RENEWABLE_COLUMNS]].notna().all(axis=1)
    ].copy()

    masks = {
        DEMAND_COLUMN: df["demand_data_complete"] & df[DEMAND_COLUMN].notna(),
        "solar_generation_mwh": df["renewable_data_complete"] & df["solar_generation_mwh"].notna(),
        "wind_generation_mwh": df["renewable_data_complete"] & df["wind_generation_mwh"].notna(),
        "solar_wind_generation_mwh": df["renewable_data_complete"] & df["solar_wind_generation_mwh"].notna(),
        "residual_demand_after_solar_wind_mwh": (
            df["demand_data_complete"] & df["renewable_data_complete"]
            & df["residual_demand_after_solar_wind_mwh"].notna()
        ),
        "solar_wind_share_pct": (
            df["demand_data_complete"] & df["renewable_data_complete"] & df["solar_wind_share_pct"].notna()
        ),
    }
    descriptive = pd.DataFrame(
        {metric: df.loc[mask, metric].describe() for metric, mask in masks.items()}
    ).T.reset_index(names="metric")
    exclusion = pd.DataFrame(
        [
            {"analysis": metric, "total_rows": len(df), "rows_used": int(mask.sum()), "rows_excluded": int((~mask).sum())}
            for metric, mask in masks.items()
        ]
    )

    null_counts = df.isna().sum().rename("null_count").reset_index(name="null_count")
    null_counts = null_counts.rename(columns={"index": "column"})
    validation_summary = pd.DataFrame(
        [{"check": key, "value": value} for key, value in validation.items()]
    )
    quality_flag_counts = pd.DataFrame(
        [
            {"quality_flag": "demand_data_complete", "complete_rows": int(df.demand_data_complete.sum()), "incomplete_rows": int((~df.demand_data_complete).sum())},
            {"quality_flag": "renewable_data_complete", "complete_rows": int(df.renewable_data_complete.sum()), "incomplete_rows": int((~df.renewable_data_complete).sum())},
        ]
    )
    null_demand = df.loc[~df["demand_data_complete"], ["period", DEMAND_COLUMN, "demand_data_complete"]]
    null_renewable_values = df.loc[
        df["period"].isin(KNOWN_NULL_VALUE_TIMESTAMPS),
        ["period", *RENEWABLE_COLUMNS, "renewable_data_complete"],
    ]
    missing_renewable_block = df.loc[
        df["period"].isin(KNOWN_MISSING_RENEWABLE_BLOCK),
        ["period", DEMAND_COLUMN, *RENEWABLE_COLUMNS, "demand_data_complete", "renewable_data_complete"],
    ]

    demand_by_hour = demand.groupby("hour", as_index=False)[DEMAND_COLUMN].mean()
    demand_by_day = (
        demand.groupby("day_of_week", observed=True)[DEMAND_COLUMN].mean().reindex(DAY_ORDER).rename_axis("day_of_week").reset_index(name=DEMAND_COLUMN)
    )
    demand_by_month = demand.groupby(["month", "month_name"], as_index=False)[DEMAND_COLUMN].mean().sort_values("month")
    monthly_demand = demand.set_index("period").resample("MS")[DEMAND_COLUMN].agg(average_demand_mwh="mean", peak_demand_mwh="max").reset_index()
    daily_demand = demand.groupby("date", as_index=False)[DEMAND_COLUMN].mean().rename(columns={DEMAND_COLUMN: "daily_average_demand_mwh"})
    daily_demand["rolling_7_day_average_demand_mwh"] = daily_demand["daily_average_demand_mwh"].rolling(7, min_periods=7).mean()
    demand_by_year = demand.groupby("year")[DEMAND_COLUMN].agg(average_demand_mwh="mean", peak_demand_mwh="max", minimum_demand_mwh="min", valid_hours="count").reset_index()
    demand_year_month = demand.groupby(["year", "month"], as_index=False)[DEMAND_COLUMN].mean()
    heatmap = demand.pivot_table(index="day_of_week", columns="hour", values=DEMAND_COLUMN, aggfunc="mean").reindex(DAY_ORDER)

    renewable_by_hour = renewable.groupby("hour", as_index=False)[[*RENEWABLE_COLUMNS, "solar_wind_generation_mwh"]].mean()
    combined_by_hour = combined.groupby("hour", as_index=False)[["solar_wind_share_pct", "residual_demand_after_solar_wind_mwh"]].mean()
    monthly_renewable = renewable.groupby(["month", "month_name"], as_index=False)[[*RENEWABLE_COLUMNS, "solar_wind_generation_mwh"]].mean().sort_values("month")
    monthly_share = combined.groupby(["month", "month_name"], as_index=False)["solar_wind_share_pct"].mean().sort_values("month")

    top_demand = demand.nlargest(10, DEMAND_COLUMN)[["period", DEMAND_COLUMN, *RENEWABLE_COLUMNS, "solar_wind_generation_mwh", "residual_demand_after_solar_wind_mwh", "solar_wind_share_pct"]]
    top_residual = combined.nlargest(10, "residual_demand_after_solar_wind_mwh")[["period", DEMAND_COLUMN, *RENEWABLE_COLUMNS, "solar_wind_generation_mwh", "residual_demand_after_solar_wind_mwh", "solar_wind_share_pct"]]
    top_share = combined.nlargest(10, "solar_wind_share_pct")[["period", DEMAND_COLUMN, *RENEWABLE_COLUMNS, "solar_wind_generation_mwh", "residual_demand_after_solar_wind_mwh", "solar_wind_share_pct"]]

    tables = {
        "validation_summary": validation_summary,
        "null_counts": null_counts,
        "quality_flag_counts": quality_flag_counts,
        "analysis_exclusion_counts": exclusion,
        "descriptive_statistics": descriptive,
        "null_demand_timestamps": null_demand,
        "null_renewable_value_timestamps": null_renewable_values,
        "missing_renewable_row_block": missing_renewable_block,
        "demand_by_hour": demand_by_hour,
        "demand_by_day_of_week": demand_by_day,
        "demand_by_month": demand_by_month,
        "monthly_demand_summary": monthly_demand,
        "daily_demand_with_rolling_average": daily_demand,
        "demand_by_year": demand_by_year,
        "demand_by_year_and_month": demand_year_month,
        "demand_hour_day_heatmap": heatmap.reset_index(),
        "renewable_by_hour": renewable_by_hour,
        "renewable_share_and_residual_by_hour": combined_by_hour,
        "monthly_renewable_generation": monthly_renewable,
        "monthly_renewable_share": monthly_share,
        "top_10_demand_timestamps": top_demand,
        "top_10_residual_demand_timestamps": top_residual,
        "top_10_renewable_share_timestamps": top_share,
    }
    for name, table in tables.items():
        save_table(table, f"{name}.csv")
    return tables


def style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_figures(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Create and save all required matplotlib charts."""
    filenames: list[str] = []

    daily = tables["daily_demand_with_rolling_average"]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(daily["date"], daily["daily_average_demand_mwh"], linewidth=0.8, alpha=0.5, label="Daily average")
    ax.plot(daily["date"], daily["rolling_7_day_average_demand_mwh"], linewidth=1.8, label="7-day rolling average")
    style_axes(ax, "Daily Average CISO Demand and 7-Day Rolling Average", "Date (UTC)", "Demand (MWh per hour)")
    ax.legend()
    filenames.append("01_daily_average_demand_7_day_rolling.png"); save_figure(fig, filenames[-1])

    hourly = tables["demand_by_hour"]
    fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(hourly["hour"], hourly[DEMAND_COLUMN], marker="o")
    ax.set_xticks(range(0, 24, 2)); style_axes(ax, "Average CISO Demand by Hour of Day", "Hour of day (UTC)", "Average demand (MWh per hour)")
    filenames.append("02_average_demand_by_hour.png"); save_figure(fig, filenames[-1])

    day = tables["demand_by_day_of_week"]
    fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(day["day_of_week"], day[DEMAND_COLUMN]); ax.tick_params(axis="x", rotation=30)
    style_axes(ax, "Average CISO Demand by Day of Week", "Day of week (UTC)", "Average demand (MWh per hour)")
    filenames.append("03_average_demand_by_day_of_week.png"); save_figure(fig, filenames[-1])

    monthly = tables["monthly_demand_summary"]
    fig, ax = plt.subplots(figsize=(13, 5)); ax.plot(monthly["period"], monthly["average_demand_mwh"], marker="o", label="Monthly average"); ax.plot(monthly["period"], monthly["peak_demand_mwh"], marker="o", label="Monthly peak")
    style_axes(ax, "Monthly Average and Peak CISO Demand", "Month (UTC)", "Demand (MWh per hour)"); ax.legend()
    filenames.append("04_monthly_average_and_peak_demand.png"); save_figure(fig, filenames[-1])

    heat = tables["demand_hour_day_heatmap"].set_index("day_of_week")
    fig, ax = plt.subplots(figsize=(12, 5)); image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(24)); ax.set_xticklabels(range(24)); ax.set_yticks(range(7)); ax.set_yticklabels(heat.index)
    ax.set_title("Average CISO Demand by Hour and Day of Week"); ax.set_xlabel("Hour of day (UTC)"); ax.set_ylabel("Day of week (UTC)")
    colorbar = fig.colorbar(image, ax=ax); colorbar.set_label("Average demand (MWh per hour)")
    filenames.append("05_demand_hour_day_heatmap.png"); save_figure(fig, filenames[-1])

    renew_hour = tables["renewable_by_hour"]
    fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(renew_hour["hour"], renew_hour["solar_generation_mwh"], marker="o", label="Solar"); ax.plot(renew_hour["hour"], renew_hour["wind_generation_mwh"], marker="o", label="Wind"); ax.plot(renew_hour["hour"], renew_hour["solar_wind_generation_mwh"], marker="o", label="Solar + wind")
    ax.set_xticks(range(0, 24, 2)); style_axes(ax, "Average Solar and Wind Generation by Hour", "Hour of day (UTC)", "Average generation (MWh per hour)"); ax.legend()
    filenames.append("06_solar_wind_generation_by_hour.png"); save_figure(fig, filenames[-1])

    combined_hour = tables["renewable_share_and_residual_by_hour"]
    fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(combined_hour["hour"], combined_hour["solar_wind_share_pct"], marker="o", color="tab:green")
    ax.set_xticks(range(0, 24, 2)); style_axes(ax, "Average Solar and Wind Share by Hour", "Hour of day (UTC)", "Average solar + wind share (%)")
    filenames.append("07_renewable_share_by_hour.png"); save_figure(fig, filenames[-1])

    fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(combined_hour["hour"], combined_hour["residual_demand_after_solar_wind_mwh"], marker="o", color="tab:red")
    ax.set_xticks(range(0, 24, 2)); style_axes(ax, "Average Residual Demand After Solar and Wind by Hour", "Hour of day (UTC)", "Average residual demand (MWh per hour)")
    filenames.append("08_residual_demand_by_hour.png"); save_figure(fig, filenames[-1])

    month_renew = tables["monthly_renewable_generation"]
    fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(12); width = 0.36
    ax.bar(x - width / 2, month_renew["solar_generation_mwh"], width, label="Solar"); ax.bar(x + width / 2, month_renew["wind_generation_mwh"], width, label="Wind")
    ax.set_xticks(x); ax.set_xticklabels(month_renew["month_name"], rotation=30); style_axes(ax, "Average Renewable Generation by Month", "Month (UTC)", "Average generation (MWh per hour)"); ax.legend()
    filenames.append("09_monthly_renewable_generation.png"); save_figure(fig, filenames[-1])

    year_month = tables["demand_by_year_and_month"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for year, group in year_month.groupby("year"):
        ax.plot(group["month"], group[DEMAND_COLUMN], marker="o", label=str(year))
    ax.set_xticks(MONTH_ORDER); ax.set_xticklabels(pd.date_range("2024-01-01", periods=12, freq="MS").month_name().str[:3])
    style_axes(ax, "Monthly Demand Patterns Across 2022, 2023, and 2024", "Month (UTC)", "Average demand (MWh per hour)"); ax.legend(title="Year")
    filenames.append("10_demand_comparison_by_year.png"); save_figure(fig, filenames[-1])

    share_month = tables["monthly_renewable_share"]
    fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(share_month["month_name"], share_month["solar_wind_share_pct"], color="tab:green"); ax.tick_params(axis="x", rotation=30)
    style_axes(ax, "Average Solar and Wind Share by Month", "Month (UTC)", "Average solar + wind share (%)")
    filenames.append("11_monthly_renewable_share.png"); save_figure(fig, filenames[-1])
    return filenames


def write_findings(df: pd.DataFrame, validation: dict[str, object], tables: dict[str, pd.DataFrame], figures: list[str]) -> None:
    """Write calculated findings in plain language."""
    hour = tables["demand_by_hour"]
    day = tables["demand_by_day_of_week"]
    month = tables["demand_by_month"]
    years = tables["demand_by_year"]
    renew_hour = tables["renewable_by_hour"]
    combined_hour = tables["renewable_share_and_residual_by_hour"]
    renew_month = tables["monthly_renewable_generation"]
    share_month = tables["monthly_renewable_share"]
    top_demand = tables["top_10_demand_timestamps"].iloc[0]
    top_residual = tables["top_10_residual_demand_timestamps"].iloc[0]
    top_share = tables["top_10_renewable_share_timestamps"].iloc[0]

    peak_hour = hour.loc[hour[DEMAND_COLUMN].idxmax()]
    low_hour = hour.loc[hour[DEMAND_COLUMN].idxmin()]
    peak_day = day.loc[day[DEMAND_COLUMN].idxmax()]
    low_day = day.loc[day[DEMAND_COLUMN].idxmin()]
    peak_month = month.loc[month[DEMAND_COLUMN].idxmax()]
    low_month = month.loc[month[DEMAND_COLUMN].idxmin()]
    solar_peak = renew_hour.loc[renew_hour["solar_generation_mwh"].idxmax()]
    wind_peak = renew_hour.loc[renew_hour["wind_generation_mwh"].idxmax()]
    renew_peak_hour = renew_hour.loc[renew_hour["solar_wind_generation_mwh"].idxmax()]
    share_peak_hour = combined_hour.loc[combined_hour["solar_wind_share_pct"].idxmax()]
    residual_peak_hour = combined_hour.loc[combined_hour["residual_demand_after_solar_wind_mwh"].idxmax()]
    solar_peak_month = renew_month.loc[renew_month["solar_generation_mwh"].idxmax()]
    wind_peak_month = renew_month.loc[renew_month["wind_generation_mwh"].idxmax()]
    share_peak_month = share_month.loc[share_month["solar_wind_share_pct"].idxmax()]
    negative_solar = int((df.loc[df["renewable_data_complete"], "solar_generation_mwh"] < 0).sum())
    negative_wind = int((df.loc[df["renewable_data_complete"], "wind_generation_mwh"] < 0).sum())

    lines = [
        "# Exploratory Data Analysis Findings",
        "",
        "All timestamps and calendar groupings in this report use the source timestamp convention and are labelled UTC. They are not interpreted as local California clock time.",
        "",
        "## Dataset coverage and quality",
        "",
        f"- The dataset contains **{validation['total_rows']:,} hourly rows** from **{validation['earliest_timestamp']:%Y-%m-%dT%H}** through **{validation['latest_timestamp']:%Y-%m-%dT%H}**.",
        f"- It has **{validation['duplicate_timestamps']} duplicate timestamps** and **{validation['missing_hours']} missing hours** in the expected hourly index. Hourly continuity is therefore **{validation['hourly_continuity']}**.",
        f"- Demand is complete for **{validation['demand_complete_rows']:,} rows** and incomplete for **{validation['demand_incomplete_rows']} rows**. Renewable inputs are complete for **{validation['renewable_complete_rows']:,} rows** and incomplete for **{validation['renewable_incomplete_rows']} rows**.",
        "- The five incomplete demand timestamps are also the five timestamps where returned demand, solar, and wind values are null. A separate 24-hour block from 2024-11-02T08 through 2024-11-03T07 has demand but no source SUN or WND rows.",
        "- Metric-specific exclusions are recorded in `tables/analysis_exclusion_counts.csv`. Demand-only analysis excludes 5 rows; renewable and renewable-aware analysis excludes 29 rows.",
        "- The script only reads the master CSV and writes new artifacts under `results/eda`; it does not fill, interpolate, or overwrite source values.",
        "",
        "## Demand patterns",
        "",
        f"- Average demand by UTC hour is highest at **{int(peak_hour['hour']):02d}:00** ({peak_hour[DEMAND_COLUMN]:,.1f} MWh per hour) and lowest at **{int(low_hour['hour']):02d}:00** ({low_hour[DEMAND_COLUMN]:,.1f} MWh per hour).",
        f"- **{peak_day['day_of_week']}** has the highest day-of-week average ({peak_day[DEMAND_COLUMN]:,.1f} MWh per hour), while **{low_day['day_of_week']}** has the lowest ({low_day[DEMAND_COLUMN]:,.1f} MWh per hour).",
        f"- Across calendar months, **{peak_month['month_name']}** has the highest average ({peak_month[DEMAND_COLUMN]:,.1f} MWh per hour) and **{low_month['month_name']}** the lowest ({low_month[DEMAND_COLUMN]:,.1f} MWh per hour).",
        f"- Annual average demand was " + ", ".join(f"**{int(row.year)}: {row.average_demand_mwh:,.1f} MWh per hour**" for row in years.itertuples()) + ". These are descriptive comparisons, not evidence of a trend beyond the three observed years.",
        "",
        "## Solar and wind patterns",
        "",
        f"- Average solar generation reaches its hourly maximum at **{int(solar_peak['hour']):02d}:00 UTC** ({solar_peak['solar_generation_mwh']:,.1f} MWh per hour). Average wind generation is highest at **{int(wind_peak['hour']):02d}:00 UTC** ({wind_peak['wind_generation_mwh']:,.1f} MWh per hour).",
        f"- Average combined solar and wind generation peaks at **{int(renew_peak_hour['hour']):02d}:00 UTC** ({renew_peak_hour['solar_wind_generation_mwh']:,.1f} MWh per hour).",
        f"- By calendar month, average solar generation is highest in **{solar_peak_month['month_name']}** ({solar_peak_month['solar_generation_mwh']:,.1f} MWh per hour), and average wind generation is highest in **{wind_peak_month['month_name']}** ({wind_peak_month['wind_generation_mwh']:,.1f} MWh per hour).",
        "",
        "## Peak demand and residual demand",
        "",
        f"- The maximum observed demand is **{top_demand[DEMAND_COLUMN]:,.0f} MWh** at **{top_demand['period']:%Y-%m-%dT%H} UTC**. The top 10 demand observations are saved separately so nearby peak hours remain visible.",
        f"- The maximum observed residual demand after solar and wind is **{top_residual['residual_demand_after_solar_wind_mwh']:,.0f} MWh** at **{top_residual['period']:%Y-%m-%dT%H} UTC**; demand was {top_residual[DEMAND_COLUMN]:,.0f} MWh and combined solar/wind generation was {top_residual['solar_wind_generation_mwh']:,.0f} MWh.",
        f"- On an average hourly profile, residual demand is highest at **{int(residual_peak_hour['hour']):02d}:00 UTC** ({residual_peak_hour['residual_demand_after_solar_wind_mwh']:,.1f} MWh per hour).",
        "",
        "## Renewable share",
        "",
        f"- Average solar/wind share is highest at **{int(share_peak_hour['hour']):02d}:00 UTC** ({share_peak_hour['solar_wind_share_pct']:.2f}%). The highest monthly average share occurs in **{share_peak_month['month_name']}** ({share_peak_month['solar_wind_share_pct']:.2f}%).",
        f"- The maximum individual hourly solar/wind share is **{top_share['solar_wind_share_pct']:.2f}%** at **{top_share['period']:%Y-%m-%dT%H} UTC**, with combined generation of {top_share['solar_wind_generation_mwh']:,.0f} MWh and demand of {top_share[DEMAND_COLUMN]:,.0f} MWh.",
        "",
        "## Limitations and review items",
        "",
        "- This EDA describes three years of source observations. It does not establish why patterns occurred, and it does not test forecast performance.",
        "- UTC hour and calendar labels must not be reinterpreted as California local-clock labels. Any later timezone conversion should be explicit and should handle daylight-saving transitions deliberately.",
        f"- Complete renewable rows include **{negative_solar:,} negative solar values** and **{negative_wind:,} negative wind values**. They are preserved source measurements, not changed to zero, but their meaning should be reviewed before renewable-aware feature engineering.",
        "- The five missing demand targets must be excluded from target-based modelling. The 29 incomplete renewable rows need an explicit later modelling policy; no imputation policy is chosen in this EDA.",
        "",
        "## Implications for later feature engineering and forecasting",
        "",
        "- Hour-of-day, day-of-week, month, and year-aware features are plausible candidates because the descriptive group averages differ, but their predictive value must be tested chronologically.",
        "- Lag and rolling features must use only information available before each forecast timestamp. The descriptive 7-day rolling daily average here is for visualization and is not yet a model feature.",
        "- Build simple chronological baselines before advanced models, then use chronological train, validation, and test splits. Do not randomly shuffle this time series.",
        "- Renewable-aware models should retain quality flags or equivalent missingness handling and must never interpret missing generation as zero.",
        "",
        "## Generated artifacts",
        "",
        f"- Figures: **{len(figures)} PNG files** under `results/eda/figures/`.",
        f"- Tables: **{len(tables)} CSV files** under `results/eda/tables/`.",
    ]
    (OUTPUT_DIR / "eda_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df, validation = load_and_validate()
    analysis = add_time_columns(df)
    tables = build_tables(analysis, validation)
    figures = build_figures(tables)
    write_findings(analysis, validation, tables, figures)
    print(f"EDA complete: {len(analysis):,} rows read from {SOURCE_CSV}")
    print(f"Created {len(figures)} figures, {len(tables)} tables, and {OUTPUT_DIR / 'eda_findings.md'}")


if __name__ == "__main__":
    main()
