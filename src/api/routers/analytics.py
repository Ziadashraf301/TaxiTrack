"""Analytics API endpoints querying historical taxi performance data."""
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from api.dependencies import get_analytics_service
from api.schemas.analytics import BreakdownRow, KPIResponse, TimeseriesPoint
from api.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/overview", response_model=KPIResponse)
def get_overview(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)", examples=["2019-01-01"]),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)", examples=["2019-03-01"]),
    borough: Optional[str] = Query(None, description="Optional borough filter"),
    service_type: Optional[str] = Query(None, description="Optional service type filter (yellow/green)"),
    service: AnalyticsService = Depends(get_analytics_service),
) -> KPIResponse:
    """Retrieve high-level operational KPIs and period-over-period percentage changes."""
    return service.get_overview(
        start_date=start_date,
        end_date=end_date,
        borough=borough,
        service_type=service_type,
    )


@router.get("/timeseries", response_model=List[TimeseriesPoint])
def get_timeseries(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)", examples=["2019-01-01"]),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)", examples=["2019-12-31"]),
    granularity: str = Query("daily", description="Time aggregation granularity ('daily' or 'monthly')"),
    borough: Optional[str] = Query(None, description="Optional borough filter"),
    service_type: Optional[str] = Query(None, description="Optional service type filter (yellow/green)"),
    service: AnalyticsService = Depends(get_analytics_service),
) -> List[TimeseriesPoint]:
    """Retrieve aggregated time series for trip volume, revenue, and passengers."""
    return service.get_timeseries(
        start_date=start_date,
        end_date=end_date,
        granularity=granularity,
        borough=borough,
        service_type=service_type,
    )


@router.get("/breakdown", response_model=List[BreakdownRow])
def get_breakdown(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)", examples=["2019-01-01"]),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)", examples=["2019-03-01"]),
    service: AnalyticsService = Depends(get_analytics_service),
) -> List[BreakdownRow]:
    """Retrieve cross-tabulated breakdown by borough and service type."""
    return service.get_breakdown(
        start_date=start_date,
        end_date=end_date,
    )
