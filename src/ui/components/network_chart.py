"""Network corridor visualization component using blue palette Altair heatmap."""
from typing import Any, Dict, List
import altair as alt
import pandas as pd
import streamlit as st


def render_network_chart(
    corridors: List[Dict[str, Any]],
    start_date: str = "",
    end_date: str = "",
    zone: str = "All",
    borough: str = "All",
    service_type: str = "All",
    **kwargs: Any,
) -> None:
    """Render origin-destination mobility flow heatmap in consistent blue palette."""
    if not corridors:
        st.info(f"No corridor mobility data available for {zone} ({borough}) from {start_date} to {end_date}.")
        return

    df = pd.DataFrame(corridors).head(15)

    # Clean long names for chart readability
    df["source_short"] = df["source"].str.slice(0, 18)
    df["target_short"] = df["target"].str.slice(0, 18)

    filter_desc = f"Zone: {zone} ({borough})" if zone and zone != "All" else (f"Borough: {borough}" if borough != "All" else "All NYC Zones")

    title_params = alt.TitleParams(
        text=f"Top Mobility Corridors: {filter_desc}",
        subtitle=f"Fleet: {service_type} (Combined Network)  |  Period: {start_date} to {end_date}",
        color="#0F172A",
        subtitleColor="#64748B",
        fontSize=14,
        fontWeight=700,
        subtitleFontSize=11,
        anchor="start",
    )

    heatmap = (
        alt.Chart(df)
        .mark_rect(cornerRadius=5)
        .encode(
            x=alt.X("target_short:N", title="Dropoff Zone (Target)", axis=alt.Axis(labelAngle=-40, labelColor="#64748B", titleColor="#475569")),
            y=alt.Y("source_short:N", title="Pickup Zone (Source)", axis=alt.Axis(labelColor="#64748B", titleColor="#475569")),
            color=alt.Color(
                "trip_count:Q",
                title="Trips",
                scale=alt.Scale(range=["#E0F2FE", "#BAE6FD", "#60A5FA", "#2563EB", "#1E3A8A"]),
            ),
            tooltip=[
                alt.Tooltip("source:N", title="Pickup Zone"),
                alt.Tooltip("target:N", title="Dropoff Zone"),
                alt.Tooltip("trip_count:Q", title="Total Trips", format=","),
                alt.Tooltip("avg_distance:Q", title="Avg Distance (mi)", format=".2f"),
                alt.Tooltip("avg_duration:Q", title="Avg Duration (min)", format=".1f"),
            ],
        )
        .properties(
            title=title_params,
            height=320,
            background="#FFFFFF",
        )
        .configure_axis(
            gridColor="#F1F5F9",
            domainColor="#E2E8F0",
        )
        .configure_view(
            strokeWidth=0,
        )
    )

    st.altair_chart(heatmap, use_container_width=True)
