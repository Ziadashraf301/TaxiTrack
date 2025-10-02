# ============================================================================
# FILE: app/data_processor.py
# ============================================================================
"""
Data transformation and processing operations.
"""

import os
import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data transformation and preparation."""
    
    @staticmethod
    def combine_historical_and_forecast(
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine historical and forecast data into a single DataFrame.
        
        Args:
            historical_df: Historical data with 'total_trips' column
            forecast_df: Forecast data with 'prediction' column
            
        Returns:
            Combined DataFrame with standardized column names
        """
        # Create copies to avoid modifying originals
        hist_df = historical_df.copy()
        fcst_df = forecast_df.copy()
        
        # Standardize column names
        hist_df = hist_df.rename(columns={"total_trips": "trips"})
        fcst_df = fcst_df.rename(columns={"prediction": "trips"})
        
        # Combine datasets
        combined_df = pd.concat([
            hist_df[["pickup_datetime", "trips"]].assign(type="historical"),
            fcst_df[["pickup_datetime", "trips", "prediction_lower", "prediction_upper"]].assign(type="forecast")
        ], ignore_index=True)
        
        # Sort by datetime
        combined_df = combined_df.sort_values("pickup_datetime").reset_index(drop=True)
        
        logger.info(f"Combined data: {len(combined_df)} total records")
        return combined_df
    
    @staticmethod
    def validate_data(df: pd.DataFrame, required_columns: list) -> tuple[bool, Optional[str]]:
        """
        Validate DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            return False, error_msg
        
        return True, None
    
    @staticmethod
    def save_results(df: pd.DataFrame, output_dir: str = "results", filename: str = "combined_df.csv") -> str:
        """
        Save results to CSV file.
        
        Args:
            df: DataFrame to save
            output_dir: Output directory path
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Results saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    @staticmethod
    def calculate_statistics(df: pd.DataFrame, data_type: str = 'historical') -> dict:
        """
        Calculate statistics for a given data type.
        
        Args:
            df: Combined DataFrame
            data_type: 'historical' or 'forecast'
            
        Returns:
            Dictionary of statistics
        """
        filtered_df = df[df['type'] == data_type]['trips']
        
        if filtered_df.empty:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0
            }
        
        return {
            'count': len(filtered_df),
            'mean': filtered_df.mean(),
            'median': filtered_df.median(),
            'min': filtered_df.min(),
            'max': filtered_df.max(),
            'std': filtered_df.std()
        }