"""Network corridor visualization component using Altair heatmap."""
from typing import Any, Dict, List
import altair as alt
import pandas as pd
import streamlit as st


def render_network_chart(corridors: List[Dict[str, Any]], month_str: str) -> None:
    """Render origin-destination mobility flow heatmap for top transit corridors."""
    if not corridors:
        st.info(f"No corridor mobility data available for {month_str}.")
        return

    df = pd.DataFrame(corridors).head(15)

    # Clean long names for chart readability
    df["source_short"] = df["source"].str.slice(0, 18)
    df["target_short"] = df["target"].str.slice(0, 18)

    heatmap = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("target_short:N", title="Dropoff Zone (Target)", axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("source_short:N", title="Pickup Zone (Source)"),
            color=alt.Color("trip_count:Q", title="Trips", scale=alt.Scale(scheme="goldorange")),
            tooltip=[
                alt.Tooltip("source:N", title="Pickup Zone"),
                alt.Tooltip("target:N", title="Dropoff Zone"),
                alt.Tooltip("trip_count:Q", title="Total Trips", format=","),
                alt.Tooltip("avg_distance:Q", title="Avg Distance (mi)", format=".2f"),
                alt.Tooltip("avg_duration:Q", title="Avg Duration (min)", format=".1f"),
            ],
        )
        .properties(
            title=f"Top Mobility Corridors ({month_str})",
            height=320,
        )
        .configure_axis(gridColor="#2a324b", domainColor="#555")
        .configure_title(fontSize=14, color="#E2E8F0", anchor="start")
    )

    st.altair_chart(heatmap, use_container_width=True)
