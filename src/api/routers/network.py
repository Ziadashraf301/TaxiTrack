"""Network API endpoints for spatial corridor analytics and NetworkX graph metrics."""
from typing import List
from fastapi import APIRouter, Depends, Query
from api.dependencies import get_network_service
from api.schemas.network import CentralityRow, CorridorRow
from api.services.network_service import NetworkService

router = APIRouter(prefix="/api/network", tags=["network"])


@router.get("/top-corridors", response_model=List[CorridorRow])
def get_top_corridors(
    pickup_month: str = Query(..., description="Target month as YYYYMM or YYYY-MM", examples=["201901"]),
    top_n: int = Query(20, ge=1, le=100, description="Top N corridors by trip volume"),
    service: NetworkService = Depends(get_network_service),
) -> List[CorridorRow]:
    """Retrieve top origin-destination transit corridors ranked by trip count."""
    return service.get_top_corridors(pickup_month=pickup_month, top_n=top_n)


@router.get("/centrality", response_model=List[CentralityRow])
def get_centrality(
    pickup_month: str = Query(..., description="Target month as YYYYMM or YYYY-MM", examples=["201901"]),
    service: NetworkService = Depends(get_network_service),
) -> List[CentralityRow]:
    """Retrieve NetworkX spatial graph centrality metrics across NYC taxi zones."""
    return service.get_centrality(pickup_month=pickup_month)
