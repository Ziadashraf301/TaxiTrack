"""Borough and service type analytics breakdown table component in refined blue styling."""
from typing import Any, Dict, List
import pandas as pd
import streamlit as st


def render_analytics_table(
    breakdown_data: List[Dict[str, Any]],
    borough: str = "All",
    service_type: str = "All",
    start_date: str = "",
    end_date: str = "",
    zone: str = "All",
    **kwargs: Any,
) -> None:
    """Render formatted interactive breakdown table with guaranteed blue progress bars and badges."""
    zone_desc = f"Zone: {zone} ({borough})" if zone and zone != "All" else f"Borough: {borough}"

    st.markdown(
        f"""
        <div style="margin-top: 4px; margin-bottom: 14px;">
            <div style="font-size: 16px; font-weight: 800; color: #0F172A; letter-spacing: -0.3px;">
                Operational Breakdown by Borough & Fleet
            </div>
            <div style="font-size: 12px; color: #64748B; margin-top: 2px;">
                📍 Active Filters: <b style="color: #1E293B;">{zone_desc}</b>  |  Fleet: <b style="color: #1E293B;">{service_type}</b>  |  Period: <b style="color: #1E293B;">{start_date} to {end_date}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not breakdown_data:
        st.info(f"No breakdown data available for Borough: {borough}, Fleet: {service_type} ({start_date} to {end_date}).")
        return

    # Build clean modern HTML table with custom Royal Blue progress bars
    rows_html = []
    for r in breakdown_data:
        b_name = r.get("borough", "Unknown")
        s_type = r.get("service_type", "")
        trips = r.get("num_trips", 0)
        rev = r.get("total_revenue", 0.0)
        tip_rate = r.get("avg_tip_rate", 0.0)
        tip_pct = min(max((tip_rate / 25.0) * 100.0, 2.0), 100.0)

        # Fleet Badge
        if "yellow" in s_type.lower():
            fleet_badge = '<span style="background:#FEF3C7;color:#B45309;padding:3px 8px;border-radius:6px;font-size:11.5px;font-weight:700;">yellow_trip</span>'
        else:
            fleet_badge = '<span style="background:#D1FAE5;color:#047857;padding:3px 8px;border-radius:6px;font-size:11.5px;font-weight:700;">green_trip</span>'

        row = (
            f'<tr style="border-bottom: 1px solid #F1F5F9; transition: background 0.15s ease;">'
            f'<td style="padding: 12px 14px; font-weight: 700; color: #1E293B; font-size: 13px;">{b_name}</td>'
            f'<td style="padding: 12px 14px;">{fleet_badge}</td>'
            f'<td style="padding: 12px 14px; font-weight: 600; color: #334155; font-size: 13px;">{trips:,}</td>'
            f'<td style="padding: 12px 14px; font-weight: 600; color: #334155; font-size: 13px;">${rev:,.2f}</td>'
            f'<td style="padding: 12px 14px;">'
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<div style="flex:1;background:#EFF6FF;border-radius:6px;height:8px;overflow:hidden;min-width:90px;">'
            f'<div style="background:linear-gradient(90deg, #3B82F6, #2563EB);width:{tip_pct:.1f}%;height:100%;border-radius:6px;"></div>'
            f'</div>'
            f'<span style="font-size:12px;font-weight:700;color:#1D4ED8;min-width:42px;text-align:right;">{tip_rate:.1f}%</span>'
            f'</div>'
            f'</td>'
            f'</tr>'
        )
        rows_html.append(row)

    table_html = (
        f'<div style="overflow-x:auto;border-radius:14px;border:1px solid #EEF2F6;box-shadow:0 2px 10px rgba(112,144,176,0.04);">'
        f'<table style="width:100%;border-collapse:collapse;text-align:left;background:#FFFFFF;">'
        f'<thead>'
        f'<tr style="background:#F8FAFC;border-bottom:1.5px solid #E2E8F0;">'
        f'<th style="padding:10px 14px;font-size:11.5px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;">Borough</th>'
        f'<th style="padding:10px 14px;font-size:11.5px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;">Fleet / Type</th>'
        f'<th style="padding:10px 14px;font-size:11.5px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;">Total Trips</th>'
        f'<th style="padding:10px 14px;font-size:11.5px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;">Gross Revenue</th>'
        f'<th style="padding:10px 14px;font-size:11.5px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;min-width:160px;">Avg Tip Rate (%)</th>'
        f'</tr>'
        f'</thead>'
        f'<tbody>'
        f'{"".join(rows_html)}'
        f'</tbody>'
        f'</table>'
        f'</div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)
