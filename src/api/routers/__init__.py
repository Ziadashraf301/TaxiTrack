"""TaxiTrack API Router modules."""
from api.routers.analytics import router as analytics_router
from api.routers.forecast import router as forecast_router
from api.routers.network import router as network_router

__all__ = ["analytics_router", "forecast_router", "network_router"]
