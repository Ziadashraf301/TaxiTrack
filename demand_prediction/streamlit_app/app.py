# ============================================================================
# FILE: app/app.py
# ============================================================================
"""
Main Streamlit application for Taxi Demand Forecasting.
"""

import os
import sys
import logging
from typing import Optional, Tuple

import streamlit as st
import pandas as pd

# Import application modules
from appconfig import AppConfig
from models import ForecastGroup, ForecastResult
from database import DatabaseClient
from forecaster_manager import ForecasterManager
from data_processor import DataProcessor
from visualization import ChartBuilder
from ui_components import UIComponents

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model_dev"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import configurations from external module
try:
    from config import CLICKHOUSE_CONFIG, DATA_CONFIG
    from forecast import TimeSeriesForecaster
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()


# Configure logging
config = AppConfig()
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class TaxiForecastApp:
    """Main application class."""
    
    def __init__(self):
        self.config = AppConfig()
        self.db_client = None
    
    def setup_page(self):
        """Configure Streamlit page."""
        st.set_page_config(
            page_title=self.config.PAGE_TITLE,
            layout=self.config.LAYOUT,
            page_icon=self.config.PAGE_ICON
        )
        st.title(f"{self.config.PAGE_ICON} {self.config.PAGE_TITLE} Dashboard")
    
    def initialize_forecaster(self):
        """Initialize or reload forecaster."""
        if 'forecaster' not in st.session_state or st.session_state.get('forecaster_failed', False):
            with st.spinner("🔄 Loading forecasting models..."):
                forecaster = ForecasterManager.initialize(TimeSeriesForecaster)
                
                if forecaster:
                    # Validate forecaster
                    is_valid, error = ForecasterManager.validate_forecaster(forecaster)
                    
                    if is_valid:
                        st.session_state.forecaster = forecaster
                        st.session_state.forecaster_failed = False
                        st.sidebar.success("✅ Forecaster initialized")
                    else:
                        st.session_state.forecaster_failed = True
                        st.sidebar.error(f"❌ {error}")
                else:
                    st.session_state.forecaster_failed = True
                    st.sidebar.error("❌ Failed to initialize forecaster")
    
    @st.cache_data(ttl=AppConfig.CACHE_TTL)
    def fetch_groups(_self) -> pd.DataFrame:
        """Fetch available forecast groups (cached)."""
        try:
            if _self.db_client is None:
                _self.db_client = DatabaseClient(CLICKHOUSE_CONFIG)
            
            return _self.db_client.fetch_available_groups(_self.config.MIN_RECORD_COUNT)
        except Exception as e:
            logger.error(f"Failed to fetch groups: {e}", exc_info=True)
            st.error(f"❌ Database connection failed: {e}")
            return pd.DataFrame()
    
    def render_sidebar_controls(self, df_groups: pd.DataFrame) -> Tuple:
        """
        Render all sidebar controls.
        
        Returns:
            Tuple of (group, horizon, model, start_date, end_date)
        """
        # Location selection
        group = self._render_location_selector(df_groups)
        
        # Forecast settings
        horizon, model = self._render_forecast_settings()
        
        # Date range
        start_date, end_date = self._render_date_range_selector()
        
        # System controls
        UIComponents.render_system_controls()
        
        return group, horizon, model, start_date, end_date
    
    def _render_location_selector(self, df_groups: pd.DataFrame) -> ForecastGroup:
        """Render location selection sidebar."""
        st.sidebar.header("📍 Location Selection")
        
        boroughs = sorted(df_groups["pickup_borough"].dropna().unique())
        borough = st.sidebar.selectbox("Select Borough", boroughs, key="borough")
        
        zones = sorted(
            df_groups[df_groups["pickup_borough"] == borough]["pickup_zone"].dropna().unique()
        )
        zone = st.sidebar.selectbox("Select Zone", zones, key="zone")
        
        services = sorted(
            df_groups[
                (df_groups["pickup_borough"] == borough) &
                (df_groups["pickup_zone"] == zone)
            ]["service_type"].dropna().unique()
        )
        service = st.sidebar.selectbox("Select Service Type", services, key="service")
        
        return ForecastGroup(
            pickup_zone=zone,
            pickup_borough=borough,
            service_type=service
        )
    
    def _render_forecast_settings(self) -> Tuple[int, str]:
        """Render forecast settings sidebar."""
        st.sidebar.header("⚙️ Forecast Settings")
        
        horizon = st.sidebar.slider(
            "Forecast Horizon (hours)",
            min_value=self.config.MIN_FORECAST_HORIZON,
            max_value=self.config.MAX_FORECAST_HORIZON,
            value=self.config.DEFAULT_FORECAST_HORIZON,
            help="How many hours into the future to forecast"
        )
        
        model = st.sidebar.selectbox(
            "Choose Model",
            ["LIGHTGBM"],
            help="Select the machine learning model for forecasting"
        )
        
        return horizon, model
    
    def _render_date_range_selector(self) -> Tuple[str, str]:
        """Render date range selector."""
        st.sidebar.header("📅 Date Range")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                pd.to_datetime("2025-08-01"),
                help="Start date for historical data"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                pd.to_datetime("2025-09-01"),
                help="End date for historical data"
            )
        
        if start_date > end_date:
            st.sidebar.error("❌ Start date must be before end date")
        
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    
    def display_parameters(self, group: ForecastGroup, horizon: int):
        """Display selected forecast parameters."""
        st.header("📋 Forecast Parameters")
        
        metrics = {
            "Borough": group.pickup_borough,
            "Zone": group.pickup_zone,
            "Service": group.service_type,
            "Horizon": f"{horizon} hours"
        }
        
        UIComponents.render_metrics_grid(metrics, columns=4)
    
    def run_forecast_pipeline(
        self,
        group: ForecastGroup,
        horizon: int,
        model: str,
        start_date: str,
        end_date: str
    ):
        """Execute the complete forecasting pipeline."""
        if 'forecaster' not in st.session_state or st.session_state.get('forecaster_failed', False):
            st.error("❌ Forecaster not available. Please click 'Restart' in the sidebar.")
            return
        
        # Create progress tracker
        progress_bar, status_text = UIComponents.render_progress_tracker()
        
        try:
            # Step 1: Initialize
            UIComponents.update_progress(progress_bar, status_text, 10, "🔄 Initializing forecast...")
            forecaster = st.session_state.forecaster
            
            # Step 2: Run forecast
            UIComponents.update_progress(progress_bar, status_text, 30, "📊 Running forecast...")
            
            forecast_df, historical_df, error = ForecasterManager.run_forecast(
                forecaster=forecaster,
                groups=[group.to_dict()],
                horizon_hours=horizon,
                model_name=model,
                start_date=start_date,
                end_date=end_date
            )
            
            if error:
                st.error(f"❌ Forecast failed: {error}")
                if st.button("🔄 Retry Forecast"):
                    UIComponents.restart_forecaster()
                return
            
            # Step 3: Validate data
            UIComponents.update_progress(progress_bar, status_text, 50, "✅ Validating results...")
            
            hist_valid, hist_error = DataProcessor.validate_data(
                historical_df, ['pickup_datetime', 'total_trips']
            )
            fcst_valid, fcst_error = DataProcessor.validate_data(
                forecast_df, ['pickup_datetime', 'prediction']
            )
            
            if not hist_valid or not fcst_valid:
                st.error(f"❌ Data validation failed: {hist_error or fcst_error}")
                return
            
            # Step 4: Process results
            UIComponents.update_progress(progress_bar, status_text, 70, "📈 Processing results...")
            
            combined_df = DataProcessor.combine_historical_and_forecast(
                historical_df, forecast_df
            )
            
            # Step 5: Save results
            UIComponents.update_progress(progress_bar, status_text, 90, "💾 Saving results...")
            DataProcessor.save_results(combined_df, self.config.RESULTS_DIR)
            
            # Step 6: Create result object
            result = ForecastResult(
                combined_df=combined_df,
                forecast_count=len(forecast_df),
                historical_count=len(historical_df),
                group=group,
                horizon=horizon,
                model_name=model
            )
            
            # Store in session state
            st.session_state.forecast_result = result
            
            UIComponents.update_progress(progress_bar, status_text, 100, "✅ Forecast completed!")
            
            st.success(
                f"✅ Forecast completed! "
                f"Historical: {result.historical_count} records, "
                f"Forecast: {result.forecast_count} future points"
            )
            
        except Exception as e:
            UIComponents.display_error_with_traceback(e, "Unexpected error during forecast")
            st.session_state.forecaster_failed = True
            
            st.warning("""
            **Troubleshooting steps:**
            1. Click 'Restart' in the sidebar
            2. Try reducing the forecast horizon
            3. Check if your model files are accessible
            """)
    
    def display_results(self):
        """Display forecast results and visualizations."""
        if 'forecast_result' not in st.session_state:
            return
        
        result: ForecastResult = st.session_state.forecast_result
        
        st.header("📈 Forecast Results")
        
        # Create visualization
        try:
            chart_title = f"Taxi Trips Forecast - {result.group}"
            chart = ChartBuilder.create_forecast_chart(result.combined_df, chart_title)
            st.altair_chart(chart, use_container_width=True)
            
        except Exception as e:
            UIComponents.display_error_with_traceback(e, "Error creating chart")
        
        # Summary metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Historical Data")
            hist_stats = DataProcessor.calculate_statistics(result.combined_df, 'historical')
            st.metric("Records", hist_stats['count'])
            st.metric("Average Trips/Hour", f"{hist_stats['mean']:.1f}")
            st.metric("Std Deviation", f"{hist_stats['std']:.1f}")
        
        with col2:
            st.subheader("🔮 Forecast")
            fcst_stats = DataProcessor.calculate_statistics(result.combined_df, 'forecast')
            st.metric("Future Points", fcst_stats['count'])
            st.metric("Average Predicted Trips", f"{fcst_stats['mean']:.1f}")
            st.metric("Std Deviation", f"{fcst_stats['std']:.1f}")
        
        # Additional insights
        st.subheader("📊 Additional Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Min Predicted", f"{fcst_stats['min']:.0f}")
        with col2:
            st.metric("Max Predicted", f"{fcst_stats['max']:.0f}")
        with col3:
            change_pct = ((fcst_stats['mean'] - hist_stats['mean']) / hist_stats['mean'] * 100) if hist_stats['mean'] > 0 else 0
            st.metric("Avg Change", f"{change_pct:+.1f}%")
        
        # Data preview
        UIComponents.render_data_preview(result.combined_df, num_rows=72)
        
        # Download button
        UIComponents.render_download_button(result.combined_df, "taxi_forecast")
    
    def run(self):
        """Main application entry point."""
        # Setup
        self.setup_page()
        self.initialize_forecaster()
        
        # Fetch groups
        df_groups = self.fetch_groups()
        
        if df_groups.empty:
            st.error("No data available from database. Please check your connection.")
            st.stop()
        
        # Render sidebar controls
        group, horizon, model, start_date, end_date = self.render_sidebar_controls(df_groups)
        
        # Display parameters
        self.display_parameters(group, horizon)
        
        # Run forecast
        st.header("🎯 Generate Forecast")
        
        if st.button("🚀 Run Forecast", type="primary", use_container_width=True):
            self.run_forecast_pipeline(group, horizon, model, start_date, end_date)
        
        # Display results
        self.display_results()
        
        # System status
        forecaster_ready = (
            'forecaster' in st.session_state and 
            not st.session_state.get('forecaster_failed', False)
        )
        last_forecast_count = (
            st.session_state.get('forecast_result').forecast_count 
            if 'forecast_result' in st.session_state else 0
        )
        
        UIComponents.render_system_status(forecaster_ready, last_forecast_count)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: gray;'>"
            "Taxi Demand Forecasting Dashboard • Built with Streamlit"
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    app = TaxiForecastApp()
    app.run()