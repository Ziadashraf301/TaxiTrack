"""Network schemas for spatial graph and OD corridor analytics."""
from pydantic import BaseModel, Field


class CorridorRow(BaseModel):
    """Origin-destination mobility corridor metric."""
    source: str = Field(..., description="Pickup zone or location")
    target: str = Field(..., description="Dropoff zone or location")
    trip_count: int = Field(..., description="Number of trips across this corridor")
    avg_distance: float = Field(..., description="Average trip distance in miles")
    avg_duration: float = Field(..., description="Average trip duration in minutes")


class CentralityRow(BaseModel):
    """Zone network centrality metrics computed via NetworkX."""
    zone: str = Field(..., description="Taxi zone name or identifier")
    pagerank: float = Field(..., description="PageRank centrality score")
    in_degree: float = Field(..., description="In-degree flow score")
    out_degree: float = Field(..., description="Out-degree flow score")
    asymmetry_score: float = Field(..., description="Flow asymmetry indicator")
    betweenness: float = Field(0.0, description="Betweenness centrality score")
