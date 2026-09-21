"""TaxiTrack Streamlit Application - Modern One-Page Operational Intelligence Dashboard."""
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

# Page Configuration
st.set_page_config(
    page_title="TaxiTrack Demand Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark modern theme)
st.markdown(
    """
    <style>
    .main {
        background-color: #0E131F;
    }
    .stMetric {
        background-color: #1A2234;
        border: 1px solid #2B384E;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .stMetric label {
        color: #94A3B8 !important;
        font-weight: 500;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Client
client = TaxiTrackClient()


def get_banner_image_path() -> str | None:
    """Resolve banner image path, copying from IDE artifacts if needed."""
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(curr_dir, "assets", "taxitrack_banner.jpg")

    if os.path.exists(target_path):
        return target_path

    source_artifact = r"C:\Users\MSI\.gemini\antigravity-ide\brain\4f616835-6abd-449c-bcd9-f2ead243f3aa\taxitrack_dashboard_banner_1789976258580.jpg"
    if os.path.exists(source_artifact):
        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy(source_artifact, target_path)
            return target_path
        except Exception:
            return source_artifact

    return None


# Cached API calls for sub-second page reactivity
@st.cache_data(ttl=300)
def cached_overview(start: str, end: str, borough: str, svc: str) -> Dict[str, Any]:
    return client.get_overview(start, end, borough=borough, service_type=svc)


@st.cache_data(ttl=300)
def cached_timeseries(start: str, end: str, gran: str, borough: str, svc: str) -> List[Dict[str, Any]]:
    return client.get_timeseries(start, end, granularity=gran, borough=borough, service_type=svc)


@st.cache_data(ttl=300)
def cached_breakdown(start: str, end: str) -> List[Dict[str, Any]]:
    return client.get_breakdown(start, end)


@st.cache_data(ttl=300)
def cached_forecast(zone: str, borough: str, svc: str, horizon: int) -> List[Dict[str, Any]]:
    return client.get_forecast(zone, borough, svc, horizon_hours=horizon)


@st.cache_data(ttl=300)
def cached_historical(start: str, end: str, zone: str, borough: str, svc: str) -> List[Dict[str, Any]]:
    return client.get_historical(start, end, zone, borough, svc)


@st.cache_data(ttl=3600)
def cached_corridors(month: str, top_n: int) -> List[Dict[str, Any]]:
    return client.get_top_corridors(month, top_n=top_n)


def main():
    # Banner Header
    banner_img = get_banner_image_path()
    if banner_img and os.path.exists(banner_img):
        st.image(banner_img, use_container_width=True)
    else:
        st.title("🚕 TaxiTrack — NYC Demand & Operations Intelligence")

    # Sidebar Filter Controls
    st.sidebar.header("🎛️ Analytics & Forecast Filters")

    # Date Range
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
    borough_options = ["All", "Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island"]
    selected_borough = st.sidebar.selectbox("Pickup Borough", borough_options, index=0)

    # Service Type
    svc_options = ["All", "yellow", "green"]
    selected_service = st.sidebar.radio("Fleet Service", svc_options, index=0, horizontal=True)

    # Forecasting Selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔮 Demand Forecast Controls")
    zone_options = [
        "JFK Airport",
        "LaGuardia Airport",
        "Times Sq/Theatre District",
        "Upper East Side South",
        "Upper East Side North",
        "Midtown Center",
        "East Village",
    ]
    forecast_zone = st.sidebar.selectbox("Target Zone", zone_options, index=0)
    zone_borough_map = {
        "JFK Airport": "Queens",
        "LaGuardia Airport": "Queens",
        "Times Sq/Theatre District": "Manhattan",
        "Upper East Side South": "Manhattan",
        "Upper East Side North": "Manhattan",
        "Midtown Center": "Manhattan",
        "East Village": "Manhattan",
    }
    forecast_borough = zone_borough_map.get(forecast_zone, selected_borough if selected_borough != "All" else "Manhattan")
    forecast_service = selected_service if selected_service != "All" else "yellow"
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Hours)", min_value=6, max_value=72, value=24, step=6)

    # Historical lookup window for forecast comparison
    hist_start_str = (end_date - datetime.timedelta(days=7)).strftime("%Y-%m-%d")

    # Month for network graph
    network_month_str = start_date.strftime("%Y%m")

    # Data Fetching & Rendering
    try:
        overview_data = cached_overview(start_str, end_str, selected_borough, selected_service)
        timeseries_data = cached_timeseries(start_str, end_str, "daily", selected_borough, selected_service)
        breakdown_data = cached_breakdown(start_str, end_str)

        # 1. KPI Cards Row
        render_kpi_cards(overview_data)
        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Forecast & Network Split Row
        col_fc, col_net = st.columns([3, 2])

        with col_fc:
            try:
                hist_data = cached_historical(hist_start_str, end_str, forecast_zone, forecast_borough, forecast_service)
                pred_data = cached_forecast(forecast_zone, forecast_borough, forecast_service, forecast_horizon)
                render_forecast_chart(hist_data, pred_data, forecast_zone)
            except Exception as fc_err:
                st.warning(f"Forecast unavailable: {fc_err}")

        with col_net:
            try:
                corridors_data = cached_corridors(network_month_str, top_n=15)
                render_network_chart(corridors_data, start_date.strftime("%B %Y"))
            except Exception as net_err:
                st.warning(f"Mobility corridors unavailable: {net_err}")

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Performance Trend Chart
        render_performance_chart(timeseries_data)

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. Analytics Breakdown Table
        render_analytics_table(breakdown_data)

    except APIClientError as api_err:
        st.error(f"⚠️ {api_err}")
        st.info("To run the backend API server: `uvicorn api.main:app --port 8000`")
    except Exception as general_err:
        st.error(f"Unexpected dashboard error: {general_err}")


if __name__ == "__main__":
    main()
