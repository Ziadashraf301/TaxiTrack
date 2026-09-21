"""Performance time series chart component using Altair."""
from typing import Any, Dict, List
import altair as alt
import pandas as pd
import streamlit as st


def render_performance_chart(timeseries_data: List[Dict[str, Any]]) -> None:
    """Render interactive dual-axis or layered time series chart for trips and revenue."""
    if not timeseries_data:
        st.info("No time series data available for the selected range.")
        return

    df = pd.DataFrame(timeseries_data)
    df["date"] = pd.to_datetime(df["date"])

    # Base chart
    base = alt.Chart(df).encode(
        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%Y-%m-%d", labelAngle=-45))
    )

    # Trips line
    trips_line = base.mark_line(color="#FDB813", strokeWidth=2.5, point=alt.OverlayMarkDef(filled=True, size=40)).encode(
        y=alt.Y("num_trips:Q", title="Trip Volume", axis=alt.Axis(titleColor="#FDB813")),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("num_trips:Q", title="Trips", format=","),
            alt.Tooltip("total_revenue:Q", title="Revenue", format="$,.2f"),
            alt.Tooltip("total_passengers:Q", title="Passengers", format=","),
        ],
    )

    # Revenue line on second axis
    revenue_line = base.mark_line(color="#00D2FF", strokeWidth=2.0, strokeDash=[4, 4]).encode(
        y=alt.Y("total_revenue:Q", title="Revenue (USD)", axis=alt.Axis(titleColor="#00D2FF")),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("total_revenue:Q", title="Revenue", format="$,.2f"),
        ],
    )

    chart = (
        alt.layer(trips_line, revenue_line)
        .resolve_scale(y="independent")
        .properties(
            title="Operational Performance Trend (Trips vs. Gross Revenue)",
            height=320,
        )
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#2a324b", domainColor="#555")
        .configure_title(fontSize=14, color="#E2E8F0", anchor="start")
    )

    st.altair_chart(chart, use_container_width=True)
