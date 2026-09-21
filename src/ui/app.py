"""TaxiTrack Streamlit Application - Modern SaaS Operational Intelligence Dashboard (Blue Theme)."""
import datetime
import os
import shutil
from typing import Any, Dict, List
import streamlit as st
from ui.api_client import APIClientError, TaxiTrackClient
from ui.components.analytics_table import render_analytics_table
from ui.components.forecast_chart import render_forecast_chart
from ui.components.kpi_cards import render_kpi_cards
from ui.components.network_chart import render_network_chart
from ui.components.performance_chart import render_performance_chart

# Page Configuration - Strict Light Theme
st.set_page_config(
    page_title="TaxiTrack — NYC Demand & Operations",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Enforced Modern Light Theme & Curated Blue Design System)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* STRICT LIGHT THEME ENFORCEMENT - Disable Dark Mode */
    :root {
        color-scheme: light !important;
        --primary-color: #2563EB !important;
        --background-color: #F8F9FD !important;
        --secondary-background-color: #FFFFFF !important;
        --text-color: #0F172A !important;
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #F8F9FD !important;
        color: #0F172A !important;
    }

    /* Main canvas layout */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1480px !important;
    }

    /* Modern Card Containers */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border-radius: 18px !important;
        border: 1px solid #EEF2F6 !important;
        box-shadow: 0 8px 24px -4px rgba(112, 144, 176, 0.08) !important;
        padding: 16px 20px !important;
        margin-bottom: 16px !important;
    }

    /* ==========================================================================
       SIDEBAR BLUE DESIGN UPGRADE
       ========================================================================== */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        box-shadow: 4px 0 24px rgba(112, 144, 176, 0.05) !important;
        padding-top: 1rem !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        background-color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #0F172A !important;
        font-weight: 800 !important;
        letter-spacing: -0.4px !important;
    }

    section[data-testid="stSidebar"] label {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        margin-bottom: 4px !important;
    }

    /* Sidebar form input boxes */
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stDateInput"] input {
        border-radius: 12px !important;
        border: 1.5px solid #E2E8F0 !important;
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stSelectbox"] > div:hover,
    div[data-testid="stDateInput"] input:hover {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }

    div[data-testid="stSelectbox"] > div:focus-within,
    div[data-testid="stDateInput"] input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2) !important;
    }

    /* Modern Blue Segmented Control for Fleet Service (No ugly radio circles) */
    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        background: #F1F5F9 !important;
        padding: 4px !important;
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        gap: 4px !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        flex: 1 !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        background: transparent !important;
        border-radius: 8px !important;
        padding: 7px 10px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }

    /* Hide standard radio dot circle */
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label p,
    div[data-testid="stRadio"] div[role="radiogroup"] > label span {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: #2563EB !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3) !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p,
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Force Radio Buttons from Red to Royal Blue */
    div[data-testid="stRadio"] [data-baseweb="radio"] div {
        border-color: #2563EB !important;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"] input:checked ~ div {
        border-color: #2563EB !important;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"] input:checked ~ div > div {
        background-color: #2563EB !important;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"] svg {
        fill: #2563EB !important;
        color: #2563EB !important;
    }
    div[data-testid="stRadio"] *[style*="rgb(255, 75, 75)"],
    div[data-testid="stRadio"] *[style*="rgb(255, 43, 43)"] {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
    }

    /* Fix Streamlit Slider from Red to Royal Blue */
    div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.5) !important;
        width: 18px !important;
        height: 18px !important;
    }

    div[data-testid="stSlider"] *[style*="rgb(255, 75, 75)"],
    div[data-testid="stSlider"] *[style*="rgb(255, 43, 43)"],
    div[data-testid="stSlider"] *[style*="rgb(255, 0, 0)"] {
        background-color: #2563EB !important;
        background: #2563EB !important;
        border-color: #2563EB !important;
    }

    div[data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: #1D4ED8 !important;
        font-weight: 700 !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #F1F5F9;
    }
    ::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94A3B8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Client
client = TaxiTrackClient()


# Cached API calls for sub-second page reactivity
@st.cache_data(ttl=300)
def cached_overview(start: str, end: str, borough: str, svc: str) -> Dict[str, Any]:
    return client.get_overview(start, end, borough=borough, service_type=svc)


@st.cache_data(ttl=300)
def cached_timeseries(start: str, end: str, gran: str, borough: str, svc: str) -> List[Dict[str, Any]]:
    return client.get_timeseries(start, end, granularity=gran, borough=borough, service_type=svc)


@st.cache_data(ttl=300)
def cached_breakdown(start: str, end: str, borough: str, svc: str) -> List[Dict[str, Any]]:
    return client.get_breakdown(start, end, borough=borough, service_type=svc)


@st.cache_data(ttl=3600)
def cached_zones() -> List[Dict[str, Any]]:
    try:
        return client.get_zones()
    except Exception:
        return []


@st.cache_data(ttl=300)
def cached_forecast(zone: str, borough: str, svc: str, horizon: int, end_date: str) -> List[Dict[str, Any]]:
    return client.get_forecast(zone, borough, svc, horizon_hours=horizon, end_date=end_date)


@st.cache_data(ttl=300)
def cached_historical(start: str, end: str, zone: str, borough: str, svc: str) -> List[Dict[str, Any]]:
    return client.get_historical(start, end, zone, borough, svc)


@st.cache_data(ttl=300)
def cached_corridors(start: str, end: str, zone: str, top_n: int = 15) -> List[Dict[str, Any]]:
    return client.get_top_corridors(start_date=start, end_date=end, zone=zone, top_n=top_n)


def render_top_header(start_str: str, end_str: str) -> None:
    """Render clean modern SaaS top navbar with date indicator and live status (removing redundant 'All' pills)."""
    header_html = (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'flex-wrap:wrap;gap:16px;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid #EEF2F6;">'
        f'<div style="display:flex;align-items:center;gap:14px;">'
        f'<div style="width:46px;height:46px;border-radius:14px;'
        f'background:linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);'
        f'display:flex;align-items:center;justify-content:center;font-size:22px;'
        f'box-shadow:0 8px 20px -3px rgba(37, 99, 235, 0.4);">🚕</div>'
        f'<div>'
        f'<h1 style="margin:0;font-size:24px;font-weight:800;color:#0F172A;'
        f'letter-spacing:-0.6px;line-height:1.2;">TaxiTrack Dashboard</h1>'
        f'<p style="margin:3px 0 0 0;font-size:13px;color:#64748B;font-weight:500;">'
        f'NYC Urban Transit & Demand Forecasting Intelligence</p>'
        f'</div>'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        f'<div style="display:flex;align-items:center;gap:6px;background:#FFFFFF;'
        f'border:1.5px solid #E2E8F0;padding:8px 16px;border-radius:30px;'
        f'box-shadow:0 2px 8px rgba(112, 144, 176, 0.06);font-size:12.5px;font-weight:700;color:#1E3A8A;">'
        f'<span>📅</span> {start_str} &nbsp;—&nbsp; {end_str}'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:6px;background:#EFF6FF;'
        f'border:1px solid #BFDBFE;padding:8px 14px;border-radius:30px;font-size:12px;font-weight:700;color:#1D4ED8;'
        f'box-shadow:0 2px 6px rgba(37, 99, 235, 0.08);">'
        f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
        f'background:#2563EB;box-shadow:0 0 6px #2563EB;"></span> Live Serving'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)


def main():
    # Load dynamic zone and borough lookup from database warehouse
    zones_list = cached_zones()
    zone_borough_map = {z["zone"]: z["borough"] for z in zones_list if z.get("zone") and z.get("borough")}
    all_boroughs = sorted(list(set(z["borough"] for z in zones_list if z.get("borough"))))

    # Sidebar Header
    st.sidebar.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
            <div style="width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg, #1E40AF, #3B82F6);display:flex;align-items:center;justify-content:center;color:white;font-size:16px;box-shadow:0 4px 10px rgba(37, 99, 235, 0.3);">
                🎛️
            </div>
            <div>
                <div style="font-size:16px;font-weight:800;color:#0F172A;line-height:1.2;">Analytics Filters</div>
                <div style="font-size:11.5px;color:#64748B;font-weight:500;">Global Scope Across All Charts</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Date Range Controls
    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.date(2019, 1, 1),
            min_value=datetime.date(2018, 1, 1),
            max_value=datetime.date(2025, 12, 31),
        )
    with col_d2:
        end_date = st.date_input(
            "End Date",
            value=datetime.date(2019, 3, 31),
            min_value=datetime.date(2018, 1, 1),
            max_value=datetime.date(2025, 12, 31),
        )

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Borough Filter
    borough_options = ["All"] + all_boroughs if all_boroughs else ["All", "Bronx", "Brooklyn", "EWR", "Manhattan", "Queens", "Staten Island"]
    selected_borough = st.sidebar.selectbox("Pickup Borough", borough_options, index=0)

    # Service Type (Segmented control)
    svc_options = ["All", "yellow_trip", "green_trip"]
    selected_service = st.sidebar.radio("Fleet Service", svc_options, index=0, horizontal=True)

    # Demand Forecast Controls Header
    st.sidebar.markdown("<div style='margin-top: 18px; margin-bottom: 12px; border-top: 1px solid #F1F5F9; padding-top: 14px;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
            <div style="width:28px;height:28px;border-radius:8px;background:#DBEAFE;display:flex;align-items:center;justify-content:center;color:#1D4ED8;font-size:14px;">
                🔮
            </div>
            <div style="font-size:15px;font-weight:800;color:#0F172A;">Demand Forecast Target</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dynamically filter zone options based on selected borough
    if selected_borough != "All":
        filtered_zones = sorted([z["zone"] for z in zones_list if z.get("borough") == selected_borough])
    else:
        filtered_zones = sorted(list(zone_borough_map.keys()))

    if not filtered_zones:
        filtered_zones = [
            "JFK Airport",
            "LaGuardia Airport",
            "Times Sq/Theatre District",
            "Upper East Side South",
            "Upper East Side North",
            "Midtown Center",
            "East Village",
        ]

    default_idx = 0
    if "LaGuardia Airport" in filtered_zones:
        default_idx = filtered_zones.index("LaGuardia Airport")
    elif "JFK Airport" in filtered_zones:
        default_idx = filtered_zones.index("JFK Airport")

    forecast_zone = st.sidebar.selectbox("Target Zone", filtered_zones, index=default_idx)
    forecast_borough = zone_borough_map.get(forecast_zone, selected_borough if selected_borough != "All" else "Queens")
    forecast_service = selected_service if selected_service != "All" else "yellow_trip"

    # Horizon Slider
    forecast_horizon_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=30, step=1)
    forecast_horizon = forecast_horizon_days * 24

    # Historical lookup window for forecast comparison: spans from start_date to end_date
    hist_start_str = start_str

    # Render Modern Top Header (Clean: date pill and live status, removing redundant 'All' pills)
    render_top_header(start_str, end_str)

    # Data Fetching & Rendering
    try:
        overview_data = cached_overview(start_str, end_str, selected_borough, selected_service)
        timeseries_data = cached_timeseries(start_str, end_str, "daily", selected_borough, selected_service)
        breakdown_data = cached_breakdown(start_str, end_str, selected_borough, selected_service)

        # 1. KPI Cards Row (4 Modern Blue Gradient Cards)
        render_kpi_cards(overview_data)
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # 2. Forecast & Network Split Row (White Card Containers)
        col_fc, col_net = st.columns([3, 2])

        with col_fc:
            with st.container(border=True):
                try:
                    hist_data = cached_historical(hist_start_str, end_str, forecast_zone, forecast_borough, forecast_service)
                    pred_data = cached_forecast(forecast_zone, forecast_borough, forecast_service, forecast_horizon, end_str)
                    render_forecast_chart(
                        hist_data,
                        pred_data,
                        zone_name=forecast_zone,
                        borough=forecast_borough,
                        service_type=forecast_service,
                        start_date=start_str,
                        end_date=end_str,
                    )
                except Exception as fc_err:
                    st.warning(f"Forecast unavailable: {fc_err}")

        with col_net:
            with st.container(border=True):
                try:
                    corridors_data = cached_corridors(start_str, end_str, forecast_zone, top_n=15)
                    render_network_chart(
                        corridors=corridors_data,
                        start_date=start_str,
                        end_date=end_str,
                        zone=forecast_zone,
                        borough=forecast_borough,
                        service_type=selected_service,
                    )
                except Exception as net_err:
                    st.warning(f"Mobility corridors unavailable: {net_err}")

        # 3. Performance Trend Chart (White Card Container)
        with st.container(border=True):
            render_performance_chart(
                timeseries_data,
                borough=selected_borough,
                service_type=selected_service,
                start_date=start_str,
                end_date=end_str,
                zone=forecast_zone if selected_borough != "All" else "All",
            )

        # 4. Analytics Breakdown Table (White Card Container)
        with st.container(border=True):
            render_analytics_table(
                breakdown_data,
                borough=selected_borough,
                service_type=selected_service,
                start_date=start_str,
                end_date=end_str,
                zone=forecast_zone if selected_borough != "All" else "All",
            )

    except APIClientError as api_err:
        st.error(f"⚠️ {api_err}")
        st.info("To run the backend API server: `uvicorn api.main:app --port 8000`")
    except Exception as general_err:
        st.error(f"Unexpected dashboard error: {general_err}")


if __name__ == "__main__":
    main()
