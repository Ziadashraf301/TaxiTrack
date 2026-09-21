"""Forecast visualization component comparing historical observations with ONNX predictions in a refined blue system."""
from typing import Any, Dict, List
import altair as alt
import pandas as pd
import streamlit as st


def render_forecast_chart(
    historical_data: List[Dict[str, Any]],
    forecast_data: List[Dict[str, Any]],
    zone_name: str,
    borough: str = "All",
    service_type: str = "All",
    start_date: str = "",
    end_date: str = "",
    **kwargs: Any,
) -> None:
    """Render dual-line chart comparing historical observed trips and model forecasts in clean blue design system."""
    if not historical_data and not forecast_data:
        st.info(f"No forecast data available for {zone_name} ({borough}) from {start_date} to {end_date}.")
        return

    records = []
    for h in historical_data:
        records.append({
            "datetime": pd.to_datetime(h["datetime"]),
            "trips": float(h["total_trips"]),
            "series": "Historical Observed",
        })

    for f in forecast_data:
        records.append({
            "datetime": pd.to_datetime(f["datetime"]),
            "trips": float(f["predicted_trips"]),
            "series": "ONNX Forecast",
        })

    df = pd.DataFrame(records)

    # 1. Historical Observed (Primary Solid Sapphire Blue #2563EB)
    hist_df = df[df["series"] == "Historical Observed"]

    hist_area = (
        alt.Chart(hist_df)
        .mark_area(
            opacity=0.08,
            color="#2563EB",
        )
        .encode(
            x=alt.X("datetime:T"),
            y=alt.Y("trips:Q", axis=None),
        )
    )

    hist_line = (
        alt.Chart(hist_df)
        .mark_line(color="#2563EB", strokeWidth=2.2)
        .encode(
            x=alt.X("datetime:T", title="Timeline", axis=alt.Axis(format="%b %d %H:%M", labelColor="#64748B", titleColor="#334155")),
            y=alt.Y("trips:Q", title="Demand (Trips/Hour)", axis=alt.Axis(labelColor="#64748B", titleColor="#334155", format="~s")),
            tooltip=[
                alt.Tooltip("datetime:T", title="Observed Time", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("trips:Q", title="Observed Trips", format=".1f"),
            ],
        )
    )

    # 2. ONNX Forecast (Distinct Bright Cyan-Blue #0EA5E9 with dashed stroke)
    fc_df = df[df["series"] == "ONNX Forecast"]

    fc_area = (
        alt.Chart(fc_df)
        .mark_area(
            opacity=0.06,
            color="#0EA5E9",
        )
        .encode(
            x=alt.X("datetime:T"),
            y=alt.Y("trips:Q", axis=None),
        )
    )

    fc_line = (
        alt.Chart(fc_df)
        .mark_line(
            color="#0EA5E9",
            strokeWidth=2.2,
            strokeDash=[5, 3],
            point=alt.OverlayMarkDef(filled=True, size=24, color="#0EA5E9"),
        )
        .encode(
            x=alt.X("datetime:T"),
            y=alt.Y("trips:Q"),
            tooltip=[
                alt.Tooltip("datetime:T", title="Forecast Time", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("trips:Q", title="Predicted Trips", format=".1f"),
            ],
        )
    )

    horizon_hours = len(forecast_data)
    if horizon_hours % 24 == 0 and horizon_hours > 0:
        horizon_label = f"+{horizon_hours // 24}D Horizon"
    else:
        horizon_label = f"+{horizon_hours}h Horizon"

    chart_title = alt.TitleParams(
        text=f"Hourly Demand Prediction: {zone_name} ({borough})",
        subtitle=f"Fleet: {service_type}  |  Observed (Solid Blue): {start_date} to {end_date}  |  Forecast (Dashed Cyan): {horizon_label}",
        color="#0F172A",
        subtitleColor="#64748B",
        fontSize=13.5,
        fontWeight=700,
        subtitleFontSize=11,
        anchor="start",
    )

    chart = (
        alt.layer(hist_area, hist_line, fc_area, fc_line)
        .properties(
            title=chart_title,
            height=320,
            background="#FFFFFF",
        )
        .interactive(bind_y=False)
        .configure_axis(
            gridColor="#F1F5F9",
            domainColor="#E2E8F0",
            tickColor="#E2E8F0",
        )
        .configure_view(
            strokeWidth=0,
        )
    )

    st.altair_chart(chart, use_container_width=True)
