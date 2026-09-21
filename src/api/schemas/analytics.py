"""Analytics schemas for historical performance metrics."""
from typing import Optional
from pydantic import BaseModel, Field


class KPIResponse(BaseModel):
    """High-level KPI response covering aggregate performance and period-over-period trends."""
    total_trips: int = Field(..., description="Total trip count in selected window")
    total_revenue: float = Field(..., description="Total gross revenue in USD")
    total_passengers: int = Field(..., description="Total passengers served")
    avg_tip_rate: float = Field(..., description="Average tip percentage")
    trips_pct_change: Optional[float] = Field(None, description="Percentage change in trips vs prior period")
    revenue_pct_change: Optional[float] = Field(None, description="Percentage change in revenue vs prior period")
    passengers_pct_change: Optional[float] = Field(None, description="Percentage change in passengers vs prior period")
    tip_rate_pct_change: Optional[float] = Field(None, description="Percentage change in tip rate vs prior period")


class TimeseriesPoint(BaseModel):
    """Point in time for aggregated trip volume and revenue."""
    date: str = Field(..., description="Date formatted as YYYY-MM-DD or YYYY-MM")
    num_trips: int = Field(..., description="Trip volume for this period")
    total_revenue: float = Field(..., description="Total revenue for this period")
    total_passengers: int = Field(..., description="Total passenger volume for this period")


class BreakdownRow(BaseModel):
    """Borough and service type granular analytics breakdown."""
    borough: str = Field(..., description="Pickup borough name")
    service_type: str = Field(..., description="Taxi service type (e.g. yellow, green)")
    num_trips: int = Field(..., description="Total trips")
    total_revenue: float = Field(..., description="Total revenue generated")
    avg_tip_rate: float = Field(..., description="Average tip percentage")
