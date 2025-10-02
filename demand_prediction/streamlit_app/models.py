# ============================================================================
# FILE: app/models.py
# ============================================================================
"""
Data models and type definitions.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd


@dataclass
class ForecastGroup:
    """Data class for forecast group parameters."""
    pickup_zone: str
    pickup_borough: str
    service_type: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format for API calls."""
        return {
            "pickup_zone": self.pickup_zone,
            "pickup_borough": self.pickup_borough,
            "service_type": self.service_type
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.pickup_borough} / {self.pickup_zone} / {self.service_type}"


@dataclass
class ForecastResult:
    """Container for forecast results."""
    combined_df: pd.DataFrame
    forecast_count: int
    historical_count: int
    group: ForecastGroup
    horizon: int
    model_name: str
    
    @property
    def historical_avg(self) -> float:
        """Average trips in historical data."""
        hist_data = self.combined_df[self.combined_df['type'] == 'historical']['trips']
        return hist_data.mean() if not hist_data.empty else 0.0
    
    @property
    def forecast_avg(self) -> float:
        """Average predicted trips."""
        forecast_data = self.combined_df[self.combined_df['type'] == 'forecast']['trips']
        return forecast_data.mean() if not forecast_data.empty else 0.0