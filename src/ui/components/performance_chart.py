"""Performance time series chart component with clean dual-axis separation and blue styling."""
from typing import Any, Dict, List
import altair as alt
import pandas as pd
import streamlit as st


def render_performance_chart(
    timeseries_data: List[Dict[str, Any]],
    borough: str = "All",
    service_type: str = "All",
    start_date: str = "",
    end_date: str = "",
    zone: str = "All",
    **kwargs: Any,
) -> None:
    """Render interactive dual-axis time series chart with left trips axis and right revenue axis."""
    if not timeseries_data:
        st.info(f"No time series data available for Borough: {borough}, Fleet: {service_type} ({start_date} to {end_date}).")
        return

    df = pd.DataFrame(timeseries_data)
    df["date"] = pd.to_datetime(df["date"])

    base = alt.Chart(df).encode(
        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%Y-%m-%d", labelAngle=-45, labelColor="#64748B", titleColor="#334155"))
    )

    # 1. Trips Volume (Left Y-Axis, Solid Royal Blue #2563EB)
    trips_area = base.mark_area(opacity=0.08, color="#2563EB").encode(
        y=alt.Y("num_trips:Q", axis=None)
    )

    trips_line = base.mark_line(
        color="#2563EB",
        strokeWidth=2.2,
        point=alt.OverlayMarkDef(filled=True, size=24, color="#2563EB"),
    ).encode(
        y=alt.Y(
            "num_trips:Q",
            axis=alt.Axis(
                orient="left",
                title="Trip Volume (Trips/Day)",
                titleColor="#2563EB",
                labelColor="#64748B",
                format="~s",
            ),
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("num_trips:Q", title="Trips", format=","),
            alt.Tooltip("total_revenue:Q", title="Revenue", format="$,.2f"),
            alt.Tooltip("total_passengers:Q", title="Passengers", format=","),
        ],
    )

    # 2. Gross Revenue (Right Y-Axis, Cyan-Blue #0284C7, Dashed)
    revenue_line = base.mark_line(
        color="#0284C7",
        strokeWidth=2.0,
        strokeDash=[4, 3],
        point=alt.OverlayMarkDef(filled=True, size=20, color="#0284C7"),
    ).encode(
        y=alt.Y(
            "total_revenue:Q",
            axis=alt.Axis(
                orient="right",
                title="Gross Revenue (USD/Day)",
                titleColor="#0284C7",
                labelColor="#64748B",
                format="$,.2s",
            ),
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("total_revenue:Q", title="Revenue", format="$,.2f"),
        ],
    )

    filter_desc = f"Borough: {borough}" + (f" (Zone: {zone})" if zone and zone != "All" else "")
    title_params = alt.TitleParams(
        text="Operational Performance Trend: Trip Volume vs. Gross Revenue",
        subtitle=f"{filter_desc}  |  Fleet: {service_type}  |  Period: {start_date} to {end_date}  |  Left: Trips (Blue), Right: Revenue (Cyan)",
        color="#0F172A",
        subtitleColor="#64748B",
        fontSize=13.5,
        fontWeight=700,
        subtitleFontSize=11,
        anchor="start",
    )

    chart = (
        alt.layer(trips_area, trips_line, revenue_line)
        .resolve_scale(y="independent")
        .properties(
            title=title_params,
            height=320,
            background="#FFFFFF",
        )
        .interactive(bind_y=False)
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="#F1F5F9",
            domainColor="#E2E8F0",
            tickColor="#E2E8F0",
        )
    )

    st.altair_chart(chart, use_container_width=True)
