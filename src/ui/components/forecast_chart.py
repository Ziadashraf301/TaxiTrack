"""Forecast visualization component comparing historical observations with ONNX predictions."""
from typing import Any, Dict, List
import altair as alt
import pandas as pd
import streamlit as st


def render_forecast_chart(
    historical_data: List[Dict[str, Any]],
    forecast_data: List[Dict[str, Any]],
    zone_name: str,
) -> None:
    """Render dual-line chart comparing historical observed trips and model forecasts."""
    if not historical_data and not forecast_data:
        st.info(f"No forecast data available for {zone_name}.")
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

    # Historical line (solid blue)
    hist_df = df[df["series"] == "Historical Observed"]
    hist_chart = (
        alt.Chart(hist_df)
        .mark_line(color="#38BDF8", strokeWidth=2.2)
        .encode(
            x=alt.X("datetime:T", title="Time", axis=alt.Axis(format="%b %d %H:%M")),
            y=alt.Y("trips:Q", title="Demand (Trips/Hour)"),
            tooltip=[
                alt.Tooltip("datetime:T", title="Timestamp", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("trips:Q", title="Observed Trips", format=".1f"),
            ],
        )
    )

    # Forecast line (dashed amber/orange)
    fc_df = df[df["series"] == "ONNX Forecast"]
    fc_chart = (
        alt.Chart(fc_df)
        .mark_line(color="#F59E0B", strokeWidth=2.5, strokeDash=[5, 3], point=alt.OverlayMarkDef(filled=True, size=35, color="#F59E0B"))
        .encode(
            x=alt.X("datetime:T"),
            y=alt.Y("trips:Q"),
            tooltip=[
                alt.Tooltip("datetime:T", title="Forecast Time", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("trips:Q", title="Predicted Trips", format=".1f"),
            ],
        )
    )

    chart = (
        alt.layer(hist_chart, fc_chart)
        .properties(
            title=f"Hourly Demand Prediction: {zone_name} (Actual vs. 24h Horizon)",
            height=320,
        )
        .configure_axis(gridColor="#2a324b", domainColor="#555")
        .configure_title(fontSize=14, color="#E2E8F0", anchor="start")
    )

    st.altair_chart(chart, use_container_width=True)
