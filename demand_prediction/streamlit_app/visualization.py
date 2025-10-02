# ============================================================================
# FILE: app/visualization.py
# ============================================================================
"""
Chart and visualization creation.
"""

import logging
from typing import Optional
import altair as alt
import pandas as pd

logger = logging.getLogger(__name__)


class ChartBuilder:
    """Builds Altair charts for visualization."""
    
    @staticmethod
    def create_forecast_chart(
        df: pd.DataFrame,
        title: str,
        height: int = 400,
        show_confidence_interval: bool = True
    ) -> alt.Chart:
        """
        Create interactive line chart for forecast visualization.
        
        Args:
            df: Combined DataFrame with historical and forecast data
            title: Chart title
            height: Chart height in pixels
            show_confidence_interval: Whether to show prediction intervals
            
        Returns:
            Altair Chart object
        """
        # Base line chart
        line_chart = (
            alt.Chart(df)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X(
                    "pickup_datetime:T",
                    title="Date & Time",
                    axis=alt.Axis(format="%Y-%m-%d %H:%M")
                ),
                y=alt.Y(
                    "trips:Q",
                    title="Number of Trips",
                    scale=alt.Scale(zero=False)
                ),
                color=alt.Color(
                    "type:N",
                    title="Data Type",
                    scale=alt.Scale(
                        domain=['historical', 'forecast'],
                        range=['#1f77b4', '#ff7f0e']
                    )
                ),
                tooltip=[
                    alt.Tooltip("pickup_datetime:T", title="Datetime", format="%Y-%m-%d %H:%M"),
                    alt.Tooltip("trips:Q", title="Trips", format=".0f"),
                    alt.Tooltip("type:N", title="Type"),
                ]
            )
            .properties(title=title, height=height)
        )
        
        # Add confidence interval if available and requested
        if show_confidence_interval and 'prediction_lower' in df.columns:
            forecast_data = df[df['type'] == 'forecast'].dropna(subset=['prediction_lower', 'prediction_upper'])
            
            if not forecast_data.empty:
                confidence_band = (
                    alt.Chart(forecast_data)
                    .mark_area(opacity=0.2, color='#ff7f0e')
                    .encode(
                        x="pickup_datetime:T",
                        y="prediction_lower:Q",
                        y2="prediction_upper:Q",
                        tooltip=[
                            alt.Tooltip("pickup_datetime:T", title="Datetime", format="%Y-%m-%d %H:%M"),
                            alt.Tooltip("prediction_lower:Q", title="Lower Bound", format=".0f"),
                            alt.Tooltip("prediction_upper:Q", title="Upper Bound", format=".0f"),
                        ]
                    )
                )
                
                line_chart = confidence_band + line_chart
        
        return line_chart.interactive()
    
    @staticmethod
    def create_comparison_chart(
        df: pd.DataFrame,
        metric: str = 'trips',
        height: int = 300
    ) -> alt.Chart:
        """
        Create bar chart comparing historical vs forecast averages.
        
        Args:
            df: Combined DataFrame
            metric: Metric to compare
            height: Chart height in pixels
            
        Returns:
            Altair Chart object
        """
        # Calculate averages by type
        avg_data = df.groupby('type')[metric].mean().reset_index()
        avg_data.columns = ['type', 'average']
        
        chart = (
            alt.Chart(avg_data)
            .mark_bar()
            .encode(
                x=alt.X('type:N', title='Data Type'),
                y=alt.Y('average:Q', title='Average Trips'),
                color=alt.Color('type:N', legend=None),
                tooltip=[
                    alt.Tooltip('type:N', title='Type'),
                    alt.Tooltip('average:Q', title='Average', format='.2f')
                ]
            )
            .properties(title='Average Trips Comparison', height=height)
        )
        
        return chart