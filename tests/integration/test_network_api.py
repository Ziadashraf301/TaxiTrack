"""Integration tests for Network API and Prometheus Metrics endpoints."""
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from api.dependencies import get_network_service
from api.main import app
from api.schemas.network import CentralityRow, CorridorRow


@pytest.fixture
def mock_network_service():
    service = MagicMock()
    service.get_top_corridors.return_value = [
        CorridorRow(source="JFK Airport", target="Times Square", trip_count=1200, avg_distance=18.2, avg_duration=45.0),
        CorridorRow(source="LaGuardia", target="Midtown", trip_count=900, avg_distance=9.5, avg_duration=28.0),
    ]
    service.get_centrality.return_value = [
        CentralityRow(zone="JFK Airport", pagerank=0.085, in_degree=0.45, out_degree=0.52, asymmetry_score=-0.07, betweenness=0.03),
        CentralityRow(zone="Midtown", pagerank=0.092, in_degree=0.62, out_degree=0.48, asymmetry_score=0.14, betweenness=0.05),
    ]
    return service


@pytest.fixture
def client(mock_network_service):
    app.dependency_overrides[get_network_service] = lambda: mock_network_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_top_corridors_respects_top_n(client):
    response = client.get("/api/network/top-corridors?pickup_month=201901&top_n=5")
    assert response.status_code == 200
    corridors = response.json()
    assert len(corridors) <= 5
    assert corridors[0]["source"] == "JFK Airport"


def test_centrality_endpoint(client):
    response = client.get("/api/network/centrality?pickup_month=201901")
    assert response.status_code == 200
    centralities = response.json()
    assert len(centralities) == 2
    assert centralities[0]["zone"] == "JFK Airport"


def test_metrics_endpoint_exposed(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "taxitrack_forecast_requests_total" in response.text


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
