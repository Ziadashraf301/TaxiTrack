"""Integration tests for Analytics API endpoints."""
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from api.dependencies import get_analytics_service
from api.main import app
from api.schemas.analytics import BreakdownRow, KPIResponse, TimeseriesPoint


@pytest.fixture
def mock_analytics_service():
    service = MagicMock()
    service.get_overview.return_value = KPIResponse(
        total_trips=1250,
        total_revenue=31250.0,
        total_passengers=1800,
        avg_tip_rate=14.5,
        trips_pct_change=12.0,
        revenue_pct_change=8.5,
        passengers_pct_change=10.2,
        tip_rate_pct_change=-1.1,
    )
    service.get_timeseries.return_value = [
        TimeseriesPoint(date="2019-01-01", num_trips=100, total_revenue=2500.0, total_passengers=150),
        TimeseriesPoint(date="2019-01-02", num_trips=120, total_revenue=3000.0, total_passengers=170),
    ]
    service.get_breakdown.return_value = [
        BreakdownRow(borough="Manhattan", service_type="yellow", num_trips=850, total_revenue=21000.0, avg_tip_rate=16.0),
        BreakdownRow(borough="Queens", service_type="green", num_trips=400, total_revenue=10250.0, avg_tip_rate=12.5),
    ]
    return service


@pytest.fixture
def client(mock_analytics_service):
    app.dependency_overrides[get_analytics_service] = lambda: mock_analytics_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_overview_returns_200(client):
    response = client.get("/api/analytics/overview?start_date=2019-01-01&end_date=2019-01-31&borough=Manhattan")
    assert response.status_code == 200
    data = response.json()
    assert data["total_trips"] == 1250
    assert data["total_revenue"] == 31250.0
    assert data["trips_pct_change"] == 12.0


def test_overview_missing_params_returns_422(client):
    # Missing required start_date and end_date
    response = client.get("/api/analytics/overview")
    assert response.status_code == 422


def test_timeseries_endpoint(client):
    response = client.get("/api/analytics/timeseries?start_date=2019-01-01&end_date=2019-01-02&granularity=daily")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["num_trips"] == 100


def test_breakdown_endpoint(client):
    response = client.get("/api/analytics/breakdown?start_date=2019-01-01&end_date=2019-01-31")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["borough"] == "Manhattan"
