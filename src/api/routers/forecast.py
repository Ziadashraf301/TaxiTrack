"""Forecast API endpoints for historical observations and real-time ONNX inferences."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from api.dependencies import get_forecast_service
from api.metrics import FORECAST_LATENCY, FORECAST_REQUESTS
from api.schemas.forecast import ForecastPoint, HistoricalPoint
from api.services.forecast_service import ForecastService

router = APIRouter(prefix="/api/forecast", tags=["forecast"])


@router.get("/historical", response_model=List[HistoricalPoint])
def get_historical(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)", examples=["2019-01-01"]),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)", examples=["2019-01-07"]),
    pickup_zone: str = Query(..., description="Pickup zone name", examples=["JFK Airport"]),
    pickup_borough: str = Query(..., description="Pickup borough name", examples=["Queens"]),
    service_type: str = Query(..., description="Fleet service type", examples=["yellow"]),
    service: ForecastService = Depends(get_forecast_service),
) -> List[HistoricalPoint]:
    """Retrieve historical observed hourly demand points for a specific zone and fleet."""
    return service.get_historical(
        start_date=start_date,
        end_date=end_date,
        pickup_zone=pickup_zone,
        pickup_borough=pickup_borough,
        service_type=service_type,
    )


@router.get("/zones")
def get_zones(
    service: ForecastService = Depends(get_forecast_service),
):
    """Retrieve all available boroughs and zones present in the demand warehouse."""
    return service.get_all_zones()


@router.get("/predict", response_model=List[ForecastPoint])
def predict(
    pickup_zone: str = Query(..., description="Pickup zone name", examples=["JFK Airport"]),
    pickup_borough: str = Query(..., description="Pickup borough name", examples=["Queens"]),
    service_type: str = Query(..., description="Fleet service type", examples=["yellow_trip"]),
    horizon_hours: int = Query(720, ge=1, le=744, description="Forecast horizon in hours (1-744)"),
    end_date: Optional[str] = Query(None, description="Anchor end date (YYYY-MM-DD)"),
    service: ForecastService = Depends(get_forecast_service),
) -> List[ForecastPoint]:
    """Generate forward-looking hourly demand predictions using the production ONNX model."""
    try:
        FORECAST_REQUESTS.labels(zone=pickup_zone, service_type=service_type).inc()
        with FORECAST_LATENCY.time():
            return service.predict(
                pickup_zone=pickup_zone,
                pickup_borough=pickup_borough,
                service_type=service_type,
                horizon_hours=horizon_hours,
                end_date=end_date,
            )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
