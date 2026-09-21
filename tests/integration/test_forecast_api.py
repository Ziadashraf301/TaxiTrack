"""Integration tests for Demand Forecast API endpoints."""
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from api.dependencies import get_forecast_service
from api.main import app
from api.schemas.forecast import ForecastPoint, HistoricalPoint


@pytest.fixture
def mock_forecast_service():
    service = MagicMock()

    def mock_predict(pickup_zone, pickup_borough, service_type, horizon_hours=24):
        if pickup_zone == "NONEXISTENT":
            raise KeyError(f"No historical demand records found for zone='{pickup_zone}'")
        return [
            ForecastPoint(
                datetime=f"2019-01-01 {h:02d}:00:00",
                predicted_trips=40.0 + h,
                model_version="test-v1",
            )
            for h in range(horizon_hours)
        ]

    service.predict.side_effect = mock_predict
    service.get_historical.return_value = [
        HistoricalPoint(datetime="2019-01-01 00:00:00", total_trips=35.0),
        HistoricalPoint(datetime="2019-01-01 01:00:00", total_trips=28.0),
    ]
    return service


@pytest.fixture
def client(mock_forecast_service):
    app.dependency_overrides[get_forecast_service] = lambda: mock_forecast_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_predict_returns_forecast_list(client):
    response = client.get(
        "/api/forecast/predict?pickup_zone=JFK Airport&pickup_borough=Queens&service_type=yellow&horizon_hours=24"
    )
    assert response.status_code == 200
    preds = response.json()
    assert len(preds) == 24
    assert all(p["predicted_trips"] >= 0 for p in preds)
    assert preds[0]["model_version"] == "test-v1"


def test_predict_unknown_zone_returns_404(client):
    response = client.get(
        "/api/forecast/predict?pickup_zone=NONEXISTENT&pickup_borough=X&service_type=yellow"
    )
    assert response.status_code == 404
    assert "NONEXISTENT" in response.json()["detail"]


def test_historical_returns_200(client):
    response = client.get(
        "/api/forecast/historical?start_date=2019-01-01&end_date=2019-01-02&pickup_zone=JFK Airport&pickup_borough=Queens&service_type=yellow"
    )
    assert response.status_code == 200
    assert len(response.json()) == 2
