"""Borough and service type analytics breakdown table component."""
from typing import Any, Dict, List
import pandas as pd
import streamlit as st


def render_analytics_table(breakdown_data: List[Dict[str, Any]]) -> None:
    """Render formatted interactive breakdown table by Borough and Service Type."""
    st.subheader("Operational Breakdown by Borough & Fleet")

    if not breakdown_data:
        st.info("No breakdown data available for the selected period.")
        return

    df = pd.DataFrame(breakdown_data)

    st.dataframe(
        df,
        column_config={
            "borough": st.column_config.TextColumn("Borough", width="medium"),
            "service_type": st.column_config.TextColumn("Fleet / Type", width="small"),
            "num_trips": st.column_config.NumberColumn("Total Trips", format="%d"),
            "total_revenue": st.column_config.NumberColumn("Gross Revenue", format="$%.2f"),
            "avg_tip_rate": st.column_config.ProgressColumn(
                "Avg Tip Rate (%)",
                min_value=0.0,
                max_value=30.0,
                format="%.1f%%",
            ),
        },
        hide_index=True,
        use_container_width=True,
    )
