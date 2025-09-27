import os
import sys
import streamlit as st
import pandas as pd
import clickhouse_connect
import altair as alt

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model_dev"))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import CLICKHOUSE_CONFIG, DATA_CONFIG
from forecast import TimeSeriesForecaster

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Taxi Demand Forecasting", layout="wide")
st.title("🚕 Taxi Demand Forecasting Dashboard")


# -----------------------------
# Dynamic group selector
# -----------------------------
@st.cache_data
def fetch_groups():
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

# Fetch groups
df_groups = fetch_groups()

# Borough selection
boroughs = sorted(df_groups["pickup_borough"].dropna().unique())
borough = st.sidebar.selectbox("Select Borough", boroughs)

# Zone selection - filtered by selected borough
zones = sorted(df_groups[df_groups["pickup_borough"] == borough]["pickup_zone"].dropna().unique())
zone = st.sidebar.selectbox("Select Zone", zones)

# Service selection - filtered by selected borough AND zone
services = sorted(
    df_groups[
        (df_groups["pickup_borough"] == borough) &
        (df_groups["pickup_zone"] == zone)
    ]["service_type"].dropna().unique()
)
service = st.sidebar.selectbox("Select Service Type", services)

# -----------------------------
# Forecast settings
# -----------------------------
forecast_horizon = st.sidebar.slider("Forecast Horizon (hours)", 1, 480, 12)
model_choice = st.sidebar.selectbox("Choose Model", ["xgboost"])

# -----------------------------
# Date range selectors
# -----------------------------
st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2020-01-01"))

if start_date > end_date:
    st.sidebar.error("Start date must be before end date")

# -----------------------------
# Run forecasting
# -----------------------------
if st.button("Run Forecast"):
    st.info("📊 Running forecast...")

    forecaster = TimeSeriesForecaster()
    groups = [{"pickup_zone": zone, "pickup_borough": borough, "service_type": service}]

    try:
        forecast_df = forecaster.forecast(
            groups = groups, horizon_hours=forecast_horizon, model_name=model_choice, start_date = start_date, end_date = end_date
        )
        historical_df = forecaster.get_latest_historical_data(groups = groups, start_date = start_date, end_date = end_date)

        # Unify trips column
        historical_df = historical_df.rename(columns={"total_trips": "trips"})
        forecast_df = forecast_df.rename(columns={"prediction": "trips"})

        combined_df = pd.concat([
            historical_df[["pickup_datetime", "trips"]].assign(type="historical"),
            forecast_df[["pickup_datetime", "trips"]].assign(type="forecast")
        ])

        combined_df = combined_df.sort_values("pickup_datetime")

        # Save debug
        os.makedirs("results", exist_ok=True)
        combined_df.to_csv("results/combined_df.csv", index=False)


        # Custom dynamic title
        chart_title = f"🚖 Taxi Trips Forecast - {borough} / {zone} / {service}"

        # Altair chart
        line_chart = (
            alt.Chart(combined_df)
            .mark_line(point=True)  # line + points
            .encode(
                x=alt.X("pickup_datetime:T", title="Pickup Datetime"),
                y=alt.Y("trips:Q", title="Number of Trips"),
                color=alt.Color("type:N", title="Data Type"),  # differentiate historical vs forecast
                tooltip=[
                    alt.Tooltip("pickup_datetime:T", title="Datetime", format="%Y-%m-%d %H:%M:%S"),
                    alt.Tooltip("trips:Q", title="Trips"),
                    alt.Tooltip("type:N", title="Type"),
                ]
            )
            .properties(title=chart_title)
            .interactive()  # zoom & pan
        )

        # Show chart
        st.altair_chart(line_chart, use_container_width=True)

        # Show last rows of data below chart
        st.subheader("📋 Data (Historical + Forecast)")
        st.dataframe(
            combined_df.sort_values("pickup_datetime").tail(20),  # show last 20 points
            use_container_width=True
        )

        st.success(f"✅ Forecast completed: {len(forecast_df)} future points predicted")

    except Exception as e:
        st.error(f"❌ Forecast failed: {e}")
        import traceback
        traceback.print_exc()
