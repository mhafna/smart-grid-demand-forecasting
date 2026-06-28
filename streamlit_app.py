"""Retrospective planning dashboard for saved smart-grid forecasts."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_data import (
    DAILY_NAIVE_MODEL,
    SELECTED_DEMAND_MODEL,
    DashboardDataError,
    available_forecast_dates,
    available_planning_dates,
    dataframe_to_csv_bytes,
    data_quality_summary,
    headline_metrics,
    historical_bounds,
    historical_date_range,
    model_metrics,
    peak_demand_performance,
    planning_thresholds,
    selected_daily_planning_summary,
    selected_day_planning,
    selected_day_recursive_predictions,
)


COLORS = {
    "actual": "#F4F7FB",
    "demand": "#38BDF8",
    "solar": "#FBBF24",
    "wind": "#2DD4BF",
    "renewable": "#A3E635",
    "residual": "#F97316",
    "error": "#FB7185",
    "naive": "#A78BFA",
    "conservative": "#FB7185",
    "typical": "#F97316",
    "favourable": "#2DD4BF",
}


st.set_page_config(
    page_title="Smart Grid Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(145deg, #07111f 0%, #0b1728 52%, #0e1b2d 100%); }
      .block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 4rem; }
      [data-testid="stMetric"] { background: rgba(17, 37, 58, .72); border: 1px solid #25415f;
        border-radius: 12px; padding: .8rem 1rem; }
      [data-testid="stMetricLabel"] { color: #a9bdd1; white-space: normal; }
      [data-testid="stMetricValue"] { white-space: normal; overflow-wrap: anywhere; }
      .utc-note { border-left: 4px solid #38bdf8; background: rgba(56,189,248,.08);
        padding: .75rem 1rem; border-radius: 6px; margin: .6rem 0 1.2rem; }
      .concept { min-height: 125px; padding: 1rem; border-radius: 12px; border: 1px solid #25415f;
        background: rgba(13, 29, 48, .7); }
      .concept h4 { margin: 0 0 .35rem 0; color: #dceafb; }
      .concept p { margin: 0; color: #a9bdd1; line-height: 1.45; }
      .section-kicker { color: #38bdf8; letter-spacing: .08em; font-size: .78rem; font-weight: 700; }
      div[data-baseweb="tab-list"] { gap: .35rem; }
      button[data-baseweb="tab"] { background: rgba(17,37,58,.55); border-radius: 8px; padding: .55rem .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _chart(fig: go.Figure, *, height: int = 410) -> None:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=58, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D9E6F2"),
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)", title_font_color="#A9BDD1")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)", title_font_color="#A9BDD1")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True})


def _utc(value: object) -> str:
    if pd.isna(value):
        return "Unavailable"
    return pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")


def _hours(value: int) -> str:
    """Format an indicator count with correct singular/plural grammar."""
    return f"{value:,} {'hour' if value == 1 else 'hours'}"


def _model_name(value: object) -> str:
    """Keep the public XGBoost spelling consistent in visible labels."""
    return str(value).replace("Xgboost", "XGBoost").replace("xgboost", "XGBoost")


def _mwh(value: object, decimals: int = 2) -> str:
    return "Unavailable" if pd.isna(value) else f"{float(value):,.{decimals}f} MWh"


def _percent(value: object) -> str:
    return "Unavailable" if pd.isna(value) else f"{float(value):,.2f}%"


def _date_select(label: str, dates: list[pd.Timestamp], key: str) -> pd.Timestamp:
    if not dates:
        raise DashboardDataError("No saved dates are available for this selection.")
    return st.selectbox(label, dates, format_func=lambda value: value.strftime("%Y-%m-%d"), key=key)


def render_overview(headlines: dict[str, object], quality: dict[str, object]) -> None:
    st.markdown('<p class="section-kicker">RETROSPECTIVE PLANNING SUPPORT</p>', unsafe_allow_html=True)
    st.header("What this dashboard shows")
    st.write(
        "Explore saved California demand forecasts and renewable-aware planning outputs. "
        "The application reads completed evaluation results; it does not retrain models or control the grid."
    )
    st.markdown('<div class="utc-note"><b>Time standard:</b> Every timestamp and date boundary is shown in UTC.</div>', unsafe_allow_html=True)

    coverage_years = quality["coverage_end"].year - quality["coverage_start"].year + 1
    first_row = st.columns(3)
    first_row[0].metric("Selected model", _model_name(headlines["selected_model"]))
    first_row[1].metric("Recursive test MAE", _mwh(headlines["recursive_test_mae_mwh"]))
    first_row[2].metric("Improvement versus daily naive", f"{headlines['improvement_vs_daily_naive_pct']:,.2f}% better")
    second_row = st.columns(3)
    second_row[0].metric("Residual-demand MAE", _mwh(headlines["residual_test_mae_mwh"]))
    second_row[1].metric("Ramp-direction agreement", _percent(headlines["ramp_direction_agreement_pct"]))
    second_row[2].metric("Historical coverage", f"{coverage_years} years")
    second_row[2].caption(f"{quality['coverage_start']:%Y-%m-%d} to {quality['coverage_end']:%Y-%m-%d} UTC")

    st.subheader("Four ideas behind the planning view")
    cards = [
        ("Demand forecast", "Estimated total electricity demand for each of the next 24 evaluation hours."),
        ("Renewable forecast", "Estimated solar plus wind generation, using the saved daily seasonal-naive method."),
        ("Residual demand", "Forecast demand minus solar and wind. It is not a complete physical grid balance."),
        ("Planning indicators", "Training-derived flags that highlight hours for review. They are descriptive, not emergency alerts."),
    ]
    columns = st.columns(4)
    for column, (title, body) in zip(columns, cards):
        column.markdown(f'<div class="concept"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def render_forecast() -> None:
    st.header("24-Hour Demand Forecast")
    st.caption("Recursive forecasts use their own earlier predictions as later-horizon inputs. No model runs inside this page.")
    filter_a, filter_b = st.columns([1, 2])
    split = filter_a.selectbox("Evaluation split", ["validation", "test"], key="forecast_split")
    with filter_b:
        day = _date_select("Forecast date (UTC)", available_forecast_dates(split), "forecast_date")

    selected = selected_day_recursive_predictions(split, day)
    naive = selected_day_recursive_predictions(split, day, model=DAILY_NAIVE_MODEL)
    daily_mae = selected["absolute_error_mwh"].mean()
    naive_mae = naive["absolute_error_mwh"].mean()
    forecast_ramps = selected["prediction_mwh"].diff()
    peak = selected.loc[selected["prediction_mwh"].idxmax()]
    ramp_index = forecast_ramps.abs().idxmax()

    cols = st.columns(5)
    cols[0].metric("Daily MAE", f"{daily_mae:,.2f} MWh")
    cols[1].metric("Daily naive MAE", f"{naive_mae:,.2f} MWh")
    cols[2].metric("Difference vs naive", f"{naive_mae - daily_mae:,.2f} MWh")
    cols[3].metric("Forecast peak", f"{peak['prediction_mwh']:,.0f} MWh", help=_utc(peak["target_timestamp"]))
    cols[4].metric("Largest |ramp|", f"{abs(forecast_ramps.loc[ramp_index]):,.0f} MWh/h", help=_utc(selected.loc[ramp_index, "target_timestamp"]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected["target_timestamp"], y=selected["actual_demand_mwh"], name="Actual demand", line=dict(color=COLORS["actual"], width=3)))
    fig.add_trace(go.Scatter(x=selected["target_timestamp"], y=selected["prediction_mwh"], name="Recursive XGBoost", line=dict(color=COLORS["demand"], width=3)))
    fig.add_trace(go.Scatter(x=naive["target_timestamp"], y=naive["prediction_mwh"], name="Daily seasonal naive", line=dict(color=COLORS["naive"], dash="dot")))
    fig.update_layout(title=f"Demand forecast · {day:%Y-%m-%d} UTC", yaxis_title="Demand (MWh)", xaxis_title="Target time (UTC)")
    _chart(fig)

    error_fig = go.Figure(go.Bar(
        x=selected["horizon"], y=selected["error_mwh"], marker_color=[COLORS["error"] if value < 0 else COLORS["wind"] for value in selected["error_mwh"]],
        customdata=selected["target_timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC"),
        hovertemplate="Horizon %{x}<br>%{customdata}<br>Error %{y:,.0f} MWh<extra></extra>",
    ))
    error_fig.add_hline(y=0, line_color="#8BA3B8", line_width=1)
    error_fig.update_layout(title="Forecast error by horizon (forecast − actual)", xaxis_title="Forecast horizon", yaxis_title="Error (MWh)")
    _chart(error_fig, height=350)

    if not selected["actual_demand_mwh"].notna().all():
        st.warning("Actual demand is incomplete for this date, so comparison metrics use only available observations.")

    st.download_button(
        "Download selected 24-hour forecast (CSV)", dataframe_to_csv_bytes(selected),
        file_name=f"demand_forecast_{split}_{day:%Y-%m-%d}_utc.csv", mime="text/csv",
    )
    with st.expander("How to read the daily indicators"):
        st.write(f"The forecast peak occurs at **{_utc(peak['target_timestamp'])}**. The largest absolute forecast ramp ends at **{_utc(selected.loc[ramp_index, 'target_timestamp'])}**. A positive error means the forecast was above actual demand.")


def render_planning() -> None:
    st.header("Renewable-Aware Planning")
    st.caption("Saved demand and renewable forecasts are combined into residual-demand scenarios. These are review aids, not dispatch instructions.")
    col_a, col_b = st.columns([1, 2])
    split = col_a.selectbox("Evaluation split", ["validation", "test"], key="planning_split")
    with col_b:
        day = _date_select("Forecast date (UTC)", available_planning_dates(split), "planning_date")
    planning = selected_day_planning(split, day)
    daily = selected_daily_planning_summary(split, day)
    row = daily.iloc[0]

    if not planning["actual_measurements_complete"].astype(bool).all():
        st.warning("Actual demand, solar, or wind measurements are incomplete for this date. Missing actual comparisons remain blank.")

    top = st.columns(4)
    top[0].metric("Demand peak", f"{row['demand_peak_mwh']:,.0f} MWh", help=_utc(row["demand_peak_time_utc"]))
    top[1].metric("Residual-demand peak", f"{row['residual_demand_peak_mwh']:,.0f} MWh", help=_utc(row["residual_demand_peak_time_utc"]))
    top[2].metric("Lowest renewable share", f"{row['lowest_forecast_renewable_share_pct']:,.2f}%", help=_utc(row["lowest_forecast_renewable_share_time_utc"]))
    top[3].metric("Maximum upward residual ramp", f"{row['maximum_upward_residual_ramp_mwh']:,.0f} MWh/h", help=_utc(row["maximum_upward_residual_ramp_time_utc"]))

    main = go.Figure()
    series = [
        ("Forecast demand", "forecast_demand_mwh", COLORS["demand"]),
        ("Forecast residual demand", "forecast_residual_demand_mwh", COLORS["residual"]),
        ("Actual residual demand", "actual_residual_demand_mwh", COLORS["actual"]),
        ("Selected solar", "selected_solar_forecast_mwh", COLORS["solar"]),
        ("Selected wind", "selected_wind_forecast_mwh", COLORS["wind"]),
        ("Combined renewables", "selected_combined_renewable_forecast_mwh", COLORS["renewable"]),
    ]
    for name, column, color in series:
        main.add_trace(go.Scatter(x=planning["target_timestamp"], y=planning[column], name=name, line=dict(color=color, width=3 if "residual" in name.lower() else 2)))
    main.update_layout(title=f"Demand, renewables, and residual demand · {day:%Y-%m-%d} UTC", yaxis_title="Energy (MWh)", xaxis_title="Target time (UTC)")
    _chart(main, height=460)

    left, right = st.columns([2, 1])
    with left:
        scenario = go.Figure()
        for name, column, color, dash in [
            ("Conservative residual demand", "conservative_residual_demand_scenario_mwh", COLORS["conservative"], "dash"),
            ("Typical residual demand", "typical_residual_demand_scenario_mwh", COLORS["typical"], "solid"),
            ("Favourable residual demand", "favourable_residual_demand_scenario_mwh", COLORS["favourable"], "dash"),
        ]:
            scenario.add_trace(go.Scatter(x=planning["target_timestamp"], y=planning[column], name=name, line=dict(color=color, dash=dash, width=2.5)))
        scenario.update_layout(title="Residual-demand scenarios", yaxis_title="Residual demand (MWh)", xaxis_title="Target time (UTC)")
        _chart(scenario, height=390)
        st.caption("Conservative renewable availability produces the highest residual-demand scenario; favourable renewable availability produces the lowest.")
    with right:
        share = go.Figure(go.Scatter(x=planning["target_timestamp"], y=planning["forecast_renewable_share_pct"], fill="tozeroy", name="Forecast share", line=dict(color=COLORS["renewable"])))
        share.update_layout(title="Forecast renewable share", yaxis_title="Solar + wind share (%)", xaxis_title="Target time (UTC)")
        _chart(share, height=390)

    st.subheader("Planning indicators by hour")
    alert_columns = {
        "high_demand_alert": "High demand",
        "high_residual_demand_alert": "High residual demand",
        "high_upward_ramp_alert": "High upward residual ramp",
        "low_renewable_share_alert": "Low renewable share",
    }
    counts = st.columns(4)
    for column, (field, label) in zip(counts, alert_columns.items()):
        column.metric(label, _hours(int(planning[field].astype(bool).sum())))

    display = planning[["target_timestamp", "horizon", "forecast_demand_mwh", "forecast_residual_demand_mwh", "forecast_renewable_share_pct", *alert_columns]].copy()
    display = display.rename(columns={"target_timestamp": "UTC timestamp", "horizon": "Horizon", "forecast_demand_mwh": "Demand (MWh)", "forecast_residual_demand_mwh": "Residual demand (MWh)", "forecast_renewable_share_pct": "Renewable share (%)", **alert_columns})
    display["UTC timestamp"] = display["UTC timestamp"].map(_utc)
    display["Demand (MWh)"] = display["Demand (MWh)"].map(lambda value: _mwh(value).removesuffix(" MWh"))
    display["Residual demand (MWh)"] = display["Residual demand (MWh)"].map(lambda value: _mwh(value).removesuffix(" MWh"))
    display["Renewable share (%)"] = display["Renewable share (%)"].map(lambda value: _percent(value).removesuffix("%"))
    st.dataframe(display, use_container_width=True, hide_index=True, column_config={name: st.column_config.CheckboxColumn(name) for name in alert_columns.values()})

    with st.expander("Training-derived indicator thresholds"):
        thresholds = planning_thresholds().copy()
        thresholds["Meaning"] = [
            "High demand" if name == "high_demand_mwh" else
            "High residual demand" if name == "high_residual_demand_mwh" else
            "High upward residual ramp" if name == "high_positive_residual_ramp_mwh" else
            "Low renewable share"
            for name in thresholds["threshold_name"]
        ]
        thresholds["Threshold"] = [
            _percent(value) if name == "low_renewable_share_pct" else _mwh(value)
            for name, value in zip(thresholds["threshold_name"], thresholds["threshold_value"])
        ]
        thresholds["Quantile"] = thresholds["quantile"].map(lambda value: _percent(value * 100))
        thresholds["Fit split"] = thresholds["fit_split"].str.title()
        thresholds = thresholds.rename(columns={"source_definition": "Source definition"})
        st.dataframe(thresholds[["Meaning", "Threshold", "Quantile", "Fit split", "Source definition"]], use_container_width=True, hide_index=True)
        st.caption("Thresholds were fitted on training data only. Crossing one marks an hour for review; it is not a safety guarantee or automated instruction.")

    download_a, download_b = st.columns(2)
    download_a.download_button("Download selected planning table (CSV)", dataframe_to_csv_bytes(planning), file_name=f"renewable_planning_{split}_{day:%Y-%m-%d}_utc.csv", mime="text/csv")
    download_b.download_button("Download selected daily summary (CSV)", dataframe_to_csv_bytes(daily), file_name=f"planning_summary_{split}_{day:%Y-%m-%d}_utc.csv", mime="text/csv")


def render_history(quality: dict[str, object]) -> None:
    st.header("Historical Explorer")
    st.caption("Choose a bounded range. Only matching hourly rows are sent to the charts, and missing values stay missing.")
    start, end = historical_bounds()
    default_end = end.date()
    default_start = max(start.date(), default_end - timedelta(days=6))
    chosen = st.date_input("Historical date range (UTC)", value=(default_start, default_end), min_value=start.date(), max_value=end.date(), key="historical_range")
    if len(chosen) != 2:
        st.info("Choose both a start and end date to load the historical range.")
        return
    frame = historical_date_range(chosen[0], chosen[1])
    if frame.empty:
        st.warning("No historical rows fall inside this date range.")
        return

    stats = st.columns(5)
    stats[0].metric("Hourly rows", f"{len(frame):,}")
    stats[1].metric("Average demand", f"{frame['demand_mwh'].mean():,.0f} MWh")
    stats[2].metric("Peak demand", f"{frame['demand_mwh'].max():,.0f} MWh")
    stats[3].metric("Average renewable share", f"{frame['solar_wind_share_pct'].mean():,.2f}%")
    stats[4].metric("Incomplete renewable hours", f"{int(frame['renewable_data_complete'].astype(str).str.lower().ne('true').sum()):,}")

    demand = go.Figure()
    demand.add_trace(go.Scatter(x=frame["period"], y=frame["demand_mwh"], name="Demand", line=dict(color=COLORS["demand"])))
    demand.add_trace(go.Scatter(x=frame["period"], y=frame["residual_demand_after_solar_wind_mwh"], name="Residual demand", line=dict(color=COLORS["residual"])))
    demand.update_layout(title="Historical demand and residual demand", yaxis_title="Energy (MWh)", xaxis_title="Time (UTC)")
    _chart(demand)

    renewable = go.Figure()
    renewable.add_trace(go.Scatter(x=frame["period"], y=frame["solar_generation_mwh"], name="Solar", line=dict(color=COLORS["solar"])))
    renewable.add_trace(go.Scatter(x=frame["period"], y=frame["wind_generation_mwh"], name="Wind", line=dict(color=COLORS["wind"])))
    renewable.add_trace(go.Scatter(x=frame["period"], y=frame["solar_wind_generation_mwh"], name="Combined solar + wind", line=dict(color=COLORS["renewable"], width=3)))
    renewable.update_layout(title="Historical solar and wind", yaxis_title="Generation (MWh)", xaxis_title="Time (UTC)")
    _chart(renewable)

    share = px.area(frame, x="period", y="solar_wind_share_pct", labels={"period": "Time (UTC)", "solar_wind_share_pct": "Renewable share (%)"}, title="Historical solar + wind share")
    share.update_traces(line_color=COLORS["renewable"], fillcolor="rgba(163,230,53,.22)")
    _chart(share, height=340)

    flags = st.columns(3)
    flags[0].metric("Missing demand in range", int(frame["demand_mwh"].isna().sum()))
    flags[1].metric("Missing solar/wind in range", int(frame[["solar_generation_mwh", "wind_generation_mwh"]].isna().any(axis=1).sum()))
    flags[2].metric("Negative solar in range", int(frame["solar_generation_mwh"].lt(0).sum()))
    st.caption(f"Full historical source: {quality['row_count']:,} hours. Negative solar values are preserved source observations, not changed or silently clipped.")


def render_performance() -> None:
    st.header("Model Performance")
    st.info("One-step results use observed lagged demand (teacher forcing). Recursive results feed predictions forward for 24 hours. The two accuracies answer different questions.")
    metrics = model_metrics()

    left, right = st.columns(2)
    with left:
        one_step = metrics["one_step_test"].sort_values("mae_mwh")
        one_step["label"] = (
            one_step["algorithm"].str.replace("_", " ").str.title()
            + " · "
            + one_step["feature_group"].str.replace("_", " ")
        ).map(_model_name)
        fig = px.bar(one_step, x="mae_mwh", y="label", orientation="h", title="Teacher-forced one-step test MAE", labels={"mae_mwh": "MAE (MWh)", "label": "Model"}, color="mae_mwh", color_continuous_scale="Blues_r")
        fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total descending"})
        _chart(fig, height=430)
    with right:
        recursive = metrics["recursive_overall"].copy()
        fig = px.bar(recursive, x="model_label", y="mae_mwh", color="split", barmode="group", title="Recursive 24-hour MAE", labels={"model_label": "Saved forecast method", "mae_mwh": "MAE (MWh)", "split": "Split"}, color_discrete_map={"validation": COLORS["wind"], "test": COLORS["demand"]})
        fig.update_xaxes(tickangle=-25)
        _chart(fig, height=430)

    horizon = metrics["recursive_horizon"]
    horizon = horizon.loc[horizon["model"] == SELECTED_DEMAND_MODEL]
    fig = px.line(horizon, x="horizon", y="mae_mwh", color="split", markers=True, title="Recursive XGBoost MAE by forecast horizon", labels={"horizon": "Forecast horizon", "mae_mwh": "MAE (MWh)", "split": "Split"}, color_discrete_map={"validation": COLORS["wind"], "test": COLORS["demand"]})
    _chart(fig)
    st.caption("Later recursive horizons have higher error overall. The path is not perfectly monotonic, so horizon-specific values should be read from the chart.")

    validation = metrics["recursive_overall"].loc[metrics["recursive_overall"]["split"] == "validation", ["model", "model_label", "mae_mwh", "rmse_mwh"]]
    test = metrics["recursive_overall"].loc[metrics["recursive_overall"]["split"] == "test", ["model", "mae_mwh", "rmse_mwh"]]
    comparison = validation.merge(test, on="model", suffixes=("_validation", "_test"))
    comparison["MAE change (%)"] = (comparison["mae_mwh_test"] - comparison["mae_mwh_validation"]) / comparison["mae_mwh_validation"] * 100
    comparison = comparison.rename(columns={"model_label": "Method", "mae_mwh_validation": "Validation MAE (MWh)", "mae_mwh_test": "Test MAE (MWh)", "rmse_mwh_validation": "Validation RMSE (MWh)", "rmse_mwh_test": "Test RMSE (MWh)"})
    comparison["Method"] = comparison["Method"].map(_model_name)
    for column in ["Validation MAE (MWh)", "Test MAE (MWh)", "Validation RMSE (MWh)", "Test RMSE (MWh)"]:
        comparison[column] = comparison[column].map(lambda value: _mwh(value).removesuffix(" MWh"))
    comparison["MAE change (%)"] = comparison["MAE change (%)"].map(_percent)
    st.subheader("Validation-to-test behaviour")
    st.dataframe(comparison[["Method", "Validation MAE (MWh)", "Test MAE (MWh)", "MAE change (%)", "Validation RMSE (MWh)", "Test RMSE (MWh)"]], hide_index=True, use_container_width=True)

    peak = peak_demand_performance()
    peak = peak.loc[peak["model"] == SELECTED_DEMAND_MODEL]
    a, b = st.columns([1.15, 1.85])
    with a:
        st.subheader("Daily peak-demand performance")
        peak_display = peak[["split", "days", "peak_mae_mwh", "peak_bias_mwh"]].rename(columns={"split": "Split", "days": "Days", "peak_mae_mwh": "Peak MAE (MWh)", "peak_bias_mwh": "Peak bias (MWh)"}).copy()
        peak_display["Split"] = peak_display["Split"].str.title()
        peak_display["Peak MAE (MWh)"] = peak_display["Peak MAE (MWh)"].map(lambda value: _mwh(value).removesuffix(" MWh"))
        peak_display["Peak bias (MWh)"] = peak_display["Peak bias (MWh)"].map(lambda value: _mwh(value).removesuffix(" MWh"))
        st.dataframe(peak_display, hide_index=True, use_container_width=True)
        st.caption("Negative peak bias means peak demand was underpredicted on average.")
    with b:
        importance = metrics["feature_importance"]
        importance = importance.loc[importance["model"] == "xgboost__autoregressive_demand"].nlargest(10, "normalized_gain").sort_values("normalized_gain")
        fig = px.bar(importance, x="normalized_gain", y="feature", orientation="h", title="Leading XGBoost autoregressive feature importances", labels={"normalized_gain": "Normalised gain", "feature": "Feature"}, color="normalized_gain", color_continuous_scale="Teal")
        fig.update_layout(coloraxis_showscale=False)
        _chart(fig, height=350)
        st.caption("Gain describes how much a feature contributed to tree splits; it is not a causal effect.")


def render_quality(quality: dict[str, object]) -> None:
    st.header("Data Quality and Limitations")
    st.warning("This is retrospective planning support, not a live operational controller. It is not connected to a live grid.")
    limitations = [
        f"**{quality['null_all_measurements_timestamps']:,} timestamps** contain null demand, solar, and wind measurements.",
        f"**{quality['missing_sun_wnd_timestamps']:,} consecutive timestamps** contain missing SUN and WND rows while demand remains present.",
        f"**{quality['negative_solar_measurements']:,} negative solar measurements** were preserved from the EIA source; their physical or accounting cause is unresolved.",
        "Percentage errors can become unstable when solar values are close to zero, so MWh errors remain important.",
        "Later forecast horizons have higher error overall because recursive forecasts carry earlier predictions forward.",
        "Peak demand is systematically underpredicted in the saved evaluation; peak bias is shown on the performance page.",
        "Residual demand subtracts only solar and wind from demand. It is not a complete physical grid balance.",
        "Planning indicators describe training-derived threshold crossings. They are not emergency alerts, safety guarantees, or automated operating instructions.",
    ]
    for item in limitations:
        st.markdown(f"- {item}")
    st.subheader("Missing-data policy")
    st.write("Missing observations remain blank in tables, metrics, and charts. The dashboard does not replace them with zero and does not interpolate them. When a selected planning day has incomplete actuals, the page displays a warning.")


def main() -> None:
    st.title("⚡ Smart Grid Demand Forecasting")
    st.caption("California demand and renewable-aware planning · saved 2024 evaluation outputs · all times UTC")
    try:
        headlines = headline_metrics()
        quality = data_quality_summary()
        tabs = st.tabs(["Overview", "24-Hour Forecast", "Renewable Planning", "Historical Explorer", "Model Performance", "Data Quality & Limitations"])
        with tabs[0]:
            render_overview(headlines, quality)
        with tabs[1]:
            render_forecast()
        with tabs[2]:
            render_planning()
        with tabs[3]:
            render_history(quality)
        with tabs[4]:
            render_performance()
        with tabs[5]:
            render_quality(quality)
    except (DashboardDataError, OSError, ValueError, KeyError) as exc:
        st.error(f"The dashboard could not load a required saved output: {exc}")
        st.info("Check that the expected CSV files are present and unchanged, then restart the app.")
        st.stop()


if __name__ == "__main__":
    main()
