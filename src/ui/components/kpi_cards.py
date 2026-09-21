"""KPI metric cards row component."""
from typing import Any, Dict
import streamlit as st


def render_kpi_cards(overview: Dict[str, Any]) -> None:
    """Render 4 high-level KPI cards with deltas comparing to the prior period."""
    c1, c2, c3, c4 = st.columns(4)

    trips = overview.get("total_trips", 0)
    trips_delta = overview.get("trips_pct_change")
    trips_delta_str = f"{trips_delta:+.1f}%" if trips_delta is not None else None

    revenue = overview.get("total_revenue", 0.0)
    rev_delta = overview.get("revenue_pct_change")
    rev_delta_str = f"{rev_delta:+.1f}%" if rev_delta is not None else None

    passengers = overview.get("total_passengers", 0)
    pass_delta = overview.get("passengers_pct_change")
    pass_delta_str = f"{pass_delta:+.1f}%" if pass_delta is not None else None

    tip_rate = overview.get("avg_tip_rate", 0.0)
    tip_delta = overview.get("tip_rate_pct_change")
    tip_delta_str = f"{tip_delta:+.1f}%" if tip_delta is not None else None

    with c1:
        st.metric(
            label="🚕 Total Trips",
            value=f"{trips:,}",
            delta=trips_delta_str,
        )

    with c2:
        st.metric(
            label="💰 Gross Revenue",
            value=f"${revenue:,.2f}",
            delta=rev_delta_str,
        )

    with c3:
        st.metric(
            label="👥 Total Passengers",
            value=f"{passengers:,}",
            delta=pass_delta_str,
        )

    with c4:
        st.metric(
            label="💡 Avg Tip Rate",
            value=f"{tip_rate:.2f}%",
            delta=tip_delta_str,
        )
