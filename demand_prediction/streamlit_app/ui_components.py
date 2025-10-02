# ============================================================================
# FILE: app/ui_components.py
# ============================================================================
"""
Reusable UI components for Streamlit.
"""

import logging
import gc
import traceback
from typing import Optional, Callable
from datetime import datetime
import streamlit as st
import pandas as pd

logger = logging.getLogger(__name__)


class UIComponents:
    """Reusable UI components."""
    
    @staticmethod
    def render_system_controls(
        on_restart: Optional[Callable] = None,
        on_clear_cache: Optional[Callable] = None
    ):
        """
        Render system control buttons in sidebar.
        
        Args:
            on_restart: Callback function for restart button
            on_clear_cache: Callback function for clear cache button
        """
        st.sidebar.header("🔧 System")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 Restart", help="Restart forecaster", use_container_width=True):
                if on_restart:
                    on_restart()
                else:
                    UIComponents.restart_forecaster()
        
        with col2:
            if st.button("🗑️ Clear Cache", help="Clear cached data", use_container_width=True):
                if on_clear_cache:
                    on_clear_cache()
                else:
                    UIComponents.clear_cache()
    
    @staticmethod
    def restart_forecaster():
        """Restart the forecaster by clearing session state."""
        keys_to_clear = ['forecaster', 'forecast_data', 'forecaster_failed', 'last_forecast_params']
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.session_state.forecaster_failed = False
        gc.collect()
        logger.info("Forecaster restarted")
        st.rerun()
    
    @staticmethod
    def clear_cache():
        """Clear all cached data."""
        st.cache_data.clear()
        gc.collect()
        st.sidebar.success("✅ Cache cleared")
        logger.info("Cache cleared")
    
    @staticmethod
    def render_progress_tracker():
        """
        Create progress tracking components.
        
        Returns:
            Tuple of (progress_bar, status_text)
        """
        progress_bar = st.progress(0)
        status_text = st.empty()
        return progress_bar, status_text
    
    @staticmethod
    def update_progress(progress_bar, status_text, progress: int, message: str):
        """
        Update progress bar and status message.
        
        Args:
            progress_bar: Streamlit progress bar component
            status_text: Streamlit text component
            progress: Progress value (0-100)
            message: Status message
        """
        progress_bar.progress(progress)
        status_text.text(message)
    
    @staticmethod
    def display_error_with_traceback(error: Exception, context: str = ""):
        """
        Display error message with expandable traceback.
        
        Args:
            error: Exception object
            context: Additional context about the error
        """
        error_msg = f"{context}: {str(error)}" if context else str(error)
        st.error(f"❌ {error_msg}")
        
        with st.expander("🔍 Technical Details"):
            st.code(traceback.format_exc())
        
        logger.error(f"{context} - {error}", exc_info=True)
    
    @staticmethod
    def render_metrics_grid(metrics: dict, columns: int = 4):
        """
        Render a grid of metrics.
        
        Args:
            metrics: Dictionary of {label: value}
            columns: Number of columns in the grid
        """
        cols = st.columns(columns)
        
        for idx, (label, value) in enumerate(metrics.items()):
            with cols[idx % columns]:
                st.metric(label, value)
    
    @staticmethod
    def render_data_preview(
        df: pd.DataFrame,
        title: str = "📋 Data Preview",
        num_rows: int = 72
    ):
        """
        Render data preview section.
        
        Args:
            df: DataFrame to display
            title: Section title
            num_rows: Number of rows to show
        """
        st.subheader(title)
        st.dataframe(df.tail(num_rows), use_container_width=True, hide_index=True)
    
    @staticmethod
    def render_download_button(
        df: pd.DataFrame,
        filename_prefix: str = "taxi_forecast",
        label: str = "📥 Download Forecast Data"
    ):
        """
        Render download button for DataFrame.
        
        Args:
            df: DataFrame to download
            filename_prefix: Prefix for filename
            label: Button label
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"{filename_prefix}_{timestamp}.csv"
        
        st.download_button(
            label=label,
            data=df.to_csv(index=False),
            file_name=filename,
            mime="text/csv",
            use_container_width=True
        )
    
    @staticmethod
    def render_system_status(forecaster_ready: bool, last_forecast_count: int = 0):
        """
        Display system status in sidebar.
        
        Args:
            forecaster_ready: Whether forecaster is ready
            last_forecast_count: Number of points in last forecast
        """
        st.sidebar.markdown("---")
        st.sidebar.subheader("System Status")
        
        if forecaster_ready:
            st.sidebar.success("✅ Forecaster Ready")
        else:
            st.sidebar.error("❌ Forecaster Issues")
        
        if last_forecast_count > 0:
            st.sidebar.info(f"📊 Last Forecast: {last_forecast_count} points")
