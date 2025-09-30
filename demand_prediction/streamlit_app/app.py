import os
import sys
import streamlit as st
import pandas as pd
import clickhouse_connect
import altair as alt
import gc
import traceback

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model_dev"))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import CLICKHOUSE_CONFIG, DATA_CONFIG
from forecast import TimeSeriesForecaster

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Taxi Demand Forecasting", 
    layout="wide",
    page_icon="🚕"
)
st.title("🚕 Taxi Demand Forecasting Dashboard")

# -----------------------------
# Improved Forecaster Management
# -----------------------------
def initialize_forecaster():
    """Initialize forecaster with proper error handling"""
    try:
        forecaster = TimeSeriesForecaster()
        forecaster._load_trained_artifacts()
        return forecaster
    except Exception as e:
        st.error(f"❌ Failed to initialize forecaster: {e}")
        return None

def safe_forecast(forecaster, groups, horizon_hours, model_name, start_date, end_date):
    """Run forecast with comprehensive error handling and memory management"""
    try:
        # Clear memory before forecast
        gc.collect()
        
        # Run forecast
        forecast_df = forecaster.forecast(
            groups=groups, 
            horizon_hours=horizon_hours, 
            model_name=model_name, 
            start_date=start_date, 
            end_date=end_date
        )
        
        # Get historical data
        historical_df = forecaster.get_latest_historical_data(
            groups=groups, 
            start_date=start_date, 
            end_date=end_date
        )
        
        return forecast_df, historical_df, None
        
    except Exception as e:
        return None, None, str(e)

# Initialize or reload forecaster
if 'forecaster' not in st.session_state or st.session_state.get('forecaster_failed', False):
    with st.spinner("🔄 Loading forecasting models..."):
        forecaster = initialize_forecaster()
        if forecaster:
            st.session_state.forecaster = forecaster
            st.session_state.forecaster_failed = False
            st.sidebar.success("✅ Forecaster initialized")
        else:
            st.session_state.forecaster_failed = True

# -----------------------------
# Sidebar - Controls
# -----------------------------
st.sidebar.header("📍 Location Selection")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_groups():
    try:
        client = clickhouse_connect.get_client(
            host='localhost',
            port=CLICKHOUSE_CONFIG["port"],
            username=CLICKHOUSE_CONFIG["username"],
            password=CLICKHOUSE_CONFIG["password"],
            database=CLICKHOUSE_CONFIG["database"]
        )
        query = """
        SELECT DISTINCT pickup_borough, pickup_zone, service_type
        FROM mart_demand_prediction
        """
        df = client.query_df(query)
        return df
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return pd.DataFrame()

# Fetch groups
df_groups = fetch_groups()

if df_groups.empty:
    st.error("No data available from database. Please check your connection.")
    st.stop()

# Borough selection
boroughs = sorted(df_groups["pickup_borough"].dropna().unique())
borough = st.sidebar.selectbox("Select Borough", boroughs, key="borough")

# Zone selection
zones = sorted(df_groups[df_groups["pickup_borough"] == borough]["pickup_zone"].dropna().unique())
zone = st.sidebar.selectbox("Select Zone", zones, key="zone")

# Service selection
services = sorted(
    df_groups[
        (df_groups["pickup_borough"] == borough) &
        (df_groups["pickup_zone"] == zone)
    ]["service_type"].dropna().unique()
)
service = st.sidebar.selectbox("Select Service Type", services, key="service")

# -----------------------------
# Forecast settings
# -----------------------------
st.sidebar.header("⚙️ Forecast Settings")

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (hours)", 
    min_value=1, 
    max_value=48, 
    value=2,
    help="How many hours into the future to forecast"
)

model_choice = st.sidebar.selectbox(
    "Choose Model", 
    ["decision_tree"],
    help="Select the machine learning model for forecasting"
)

# -----------------------------
# Date range selectors
# -----------------------------
st.sidebar.header("📅 Date Range")

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date", 
        pd.to_datetime("2022-01-01"),
        help="Start date for historical data"
    )
with col2:
    end_date = st.date_input(
        "End Date", 
        pd.to_datetime("2024-12-31"),
        help="End date for historical data"
    )

if start_date > end_date:
    st.sidebar.error("❌ Start date must be before end date")

# Convert to string format
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

# -----------------------------
# Recovery and Debugging Section
# -----------------------------
st.sidebar.header("🔧 System")

if st.sidebar.button("🔄 Restart Forecaster", help="Use if forecasts are failing"):
    if 'forecaster' in st.session_state:
        del st.session_state.forecaster
    if 'forecast_data' in st.session_state:
        del st.session_state.forecast_data
    st.session_state.forecaster_failed = False
    gc.collect()
    st.rerun()

if st.sidebar.button("🗑️ Clear Cache", help="Clear all cached data"):
    st.cache_data.clear()
    gc.collect()
    st.sidebar.success("Cache cleared")

# -----------------------------
# Main content area
# -----------------------------

# Display selected parameters
st.header("📋 Forecast Parameters")
param_col1, param_col2, param_col3, param_col4 = st.columns(4)

with param_col1:
    st.metric("Borough", borough)
with param_col2:
    st.metric("Zone", zone)
with param_col3:
    st.metric("Service", service)
with param_col4:
    st.metric("Horizon", f"{forecast_horizon} hours")

# -----------------------------
# Run forecasting with robust error handling
# -----------------------------
st.header("🎯 Generate Forecast")

if st.button("🚀 Run Forecast", type="primary", use_container_width=True):
    
    if 'forecaster' not in st.session_state or st.session_state.get('forecaster_failed', False):
        st.error("❌ Forecaster not available. Please click 'Restart Forecaster' in the sidebar.")
        st.stop()
    
    # Prepare groups
    groups = [{
        "pickup_zone": zone, 
        "pickup_borough": borough, 
        "service_type": service
    }]
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Initialize
        status_text.text("🔄 Initializing forecast...")
        progress_bar.progress(10)
        
        forecaster = st.session_state.forecaster
        
        # Step 2: Run safe forecast
        status_text.text("📊 Running forecast...")
        progress_bar.progress(30)
        
        forecast_df, historical_df, error = safe_forecast(
            forecaster, groups, forecast_horizon, model_choice, start_date_str, end_date_str
        )
        
        if error:
            st.error(f"❌ Forecast failed: {error}")
            
            # Show retry option
            if st.button("🔄 Retry Forecast"):
                if 'forecaster' in st.session_state:
                    del st.session_state.forecaster
                st.rerun()
            st.stop()
        
        progress_bar.progress(70)
        status_text.text("📈 Processing results...")
        
        # Step 3: Process results
        historical_df = historical_df.rename(columns={"total_trips": "trips"})
        forecast_df = forecast_df.rename(columns={"prediction": "trips"})
        
        combined_df = pd.concat([
            historical_df[["pickup_datetime", "trips"]].assign(type="historical"),
            forecast_df[["pickup_datetime", "trips", "prediction_lower", "prediction_upper"]].assign(type="forecast")
        ])
        
        combined_df = combined_df.sort_values("pickup_datetime")
        
        # Save results
        os.makedirs("results", exist_ok=True)
        combined_df.to_csv("results/combined_df.csv", index=False)
        
        progress_bar.progress(90)
        status_text.text("💾 Saving results...")
        
        # Store in session state
        st.session_state.forecast_data = combined_df
        st.session_state.last_forecast_params = {
            "borough": borough,
            "zone": zone,
            "service": service,
            "horizon": forecast_horizon,
            "historical_count": len(historical_df),
            "forecast_count": len(forecast_df)
        }
        
        progress_bar.progress(100)
        status_text.text("✅ Forecast completed!")
        
        st.success(f"✅ Forecast completed! Historical: {len(historical_df)} records, Forecast: {len(forecast_df)} future points")
        
        # Clear memory
        gc.collect()
        
    except Exception as e:
        progress_bar.progress(0)
        status_text.text("❌ Forecast failed")
        
        st.error(f"❌ Unexpected error during forecast: {str(e)}")
        
        # Detailed error information
        with st.expander("🔍 Technical Details"):
            st.code(traceback.format_exc())
        
        # Mark forecaster as failed
        st.session_state.forecaster_failed = True
        
        # Recovery options
        st.warning("""
        **Troubleshooting steps:**
        1. Click 'Restart Forecaster' in the sidebar
        2. Try reducing the forecast horizon
        3. Check if your model files are accessible
        """)

# -----------------------------
# Display results
# -----------------------------
if 'forecast_data' in st.session_state and st.session_state.forecast_data is not None:
    combined_df = st.session_state.forecast_data
    params = st.session_state.last_forecast_params
    
    st.header("📈 Forecast Results")
    
    # Create visualization
    try:
        chart_title = f"Taxi Trips Forecast - {params['borough']} / {params['zone']} / {params['service']}"
        
        line_chart = (
            alt.Chart(combined_df)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("pickup_datetime:T", title="Date & Time", axis=alt.Axis(format="%Y-%m-%d %H:%M")),
                y=alt.Y("trips:Q", title="Number of Trips", scale=alt.Scale(zero=False)),
                color=alt.Color("type:N", title="Data Type"),
                tooltip=[
                    alt.Tooltip("pickup_datetime:T", title="Datetime", format="%Y-%m-%d %H:%M"),
                    alt.Tooltip("trips:Q", title="Trips", format=".0f"),
                    alt.Tooltip("type:N", title="Type"),
                ]
            )
            .properties(title=chart_title, height=400)
            .interactive()
        )
        
        st.altair_chart(line_chart, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error creating chart: {e}")
    
    # Summary metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Historical Data")
        st.metric("Records", params["historical_count"])
        historical_trips = combined_df[combined_df['type'] == 'historical']['trips']
        if not historical_trips.empty:
            st.metric("Average Trips/Hour", f"{historical_trips.mean():.1f}")
    
    with col2:
        st.subheader("🔮 Forecast")
        st.metric("Future Points", params["forecast_count"])
        forecast_trips = combined_df[combined_df['type'] == 'forecast']['trips']
        if not forecast_trips.empty:
            st.metric("Average Predicted Trips", f"{forecast_trips.mean():.1f}")
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(combined_df.tail(72), use_container_width=True, hide_index=True)
    
    # Download
    st.download_button(
        label="📥 Download Forecast Data",
        data=combined_df.to_csv(index=False),
        file_name=f"taxi_forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# -----------------------------
# System status in sidebar
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("System Status")

if 'forecaster' in st.session_state and not st.session_state.get('forecaster_failed', False):
    st.sidebar.success("✅ Forecaster Ready")
else:
    st.sidebar.error("❌ Forecaster Issues")

if 'forecast_data' in st.session_state:
    st.sidebar.info(f"📊 Last Forecast: {st.session_state.get('last_forecast_params', {}).get('forecast_count', 0)} points")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Taxi Demand Forecasting Dashboard • Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)