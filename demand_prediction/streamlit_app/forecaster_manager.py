# ============================================================================
# FILE: app/forecaster_manager.py
# ============================================================================
"""
Forecaster lifecycle and operations management.
"""

import logging
import gc
from typing import List, Dict, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class ForecasterManager:
    """Manages forecaster lifecycle and operations."""
    
    @staticmethod
    def initialize(forecaster_class, **kwargs) -> Optional[object]:
        """
        Initialize forecaster with error handling.
        
        Args:
            forecaster_class: The forecaster class to instantiate
            **kwargs: Additional arguments for forecaster initialization
            
        Returns:
            Forecaster instance or None if initialization fails
        """
        try:
            forecaster = forecaster_class(**kwargs)
            
            # Load trained artifacts if method exists
            if hasattr(forecaster, '_load_trained_artifacts'):
                forecaster._load_trained_artifacts()
            
            logger.info("Forecaster initialized successfully")
            return forecaster
        except Exception as e:
            logger.error(f"Failed to initialize forecaster: {e}", exc_info=True)
            return None
    
    @staticmethod
    def run_forecast(
        forecaster: object,
        groups: List[Dict],
        horizon_hours: int,
        model_name: str,
        start_date: str,
        end_date: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
        """
        Run forecast with comprehensive error handling.
        
        Args:
            forecaster: Forecaster instance
            groups: List of group dictionaries
            horizon_hours: Forecast horizon in hours
            model_name: Model to use for forecasting
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
        
        Returns:
            Tuple of (forecast_df, historical_df, error_message)
        """
        try:
            # Clear memory before forecast
            gc.collect()
            
            logger.info(f"Starting forecast for {len(groups)} groups, horizon: {horizon_hours}h")
            
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
            
            logger.info(
                f"Forecast completed: {len(forecast_df)} predictions, "
                f"{len(historical_df)} historical records"
            )
            
            return forecast_df, historical_df, None
            
        except Exception as e:
            error_msg = f"Forecast execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, None, error_msg
    
    @staticmethod
    def validate_forecaster(forecaster: object) -> Tuple[bool, Optional[str]]:
        """
        Validate that forecaster has required methods.
        
        Args:
            forecaster: Forecaster instance to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_methods = ['forecast', 'get_latest_historical_data']
        
        for method in required_methods:
            if not hasattr(forecaster, method):
                error_msg = f"Forecaster missing required method: {method}"
                logger.error(error_msg)
                return False, error_msg
        
        return True, None